// The Ampere set: all 152 weighted launches of Model A on kernels of ours, inside NVIDIA's runtime; see NrAmpere.h.
// Launch counting, the fp8_set<n> records and their upload follow NrFp8.cpp; our kernels are loaded like NrNet4.cpp's.
#include "NrWeightUpload.h"
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <nvapi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "NrAmpere.h"

namespace NrAmpere {
namespace {
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

constexpr unsigned kCreateModule = 0xad1a677d, kCreateFunction = 0xe2436e22, kLaunch = 0x24973538,
                   kDestroyModule = 0x41c65285, kDestroyFunction = 0xdf295ea6;
constexpr unsigned kMaxSets = 8;            // fp8_set1 .. fp8_set7, as NrFp8
constexpr unsigned kMaxLaunches = 256;      // launch numbers of an evaluation we can describe (Model A: 156)
constexpr uint16_t kBlendWeight = 0x39eb;   // float16 0.7397: the output block's history blend weight our models expect
constexpr size_t kMaxParams = 512;
constexpr uint32_t kMaxShared = 101376;     // sm_86's per-block maximum; also the most the 5090's NvAPI path takes (NrFfnV2)

void Check(HRESULT result, const char* what) {
    if (FAILED(result)) throw std::runtime_error(std::string(what) + " failed: D3D12 status " + std::to_string(unsigned(result)));
}
void NvCheck(NvAPI_Status result, const std::string& what) {
    if (result != NVAPI_OK) throw std::runtime_error(what + " failed: NVAPI status " + std::to_string(result));
}

std::vector<unsigned char> ReadFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("missing Ampere-set asset " + path.string());
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (bytes.empty() || !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
        throw std::runtime_error("cannot read Ampere-set asset " + path.string());
    return bytes;
}

// ---- fp8_set<n>.txt: the weighted launches and their records (NrFp8's format, read the same way) ----------------
struct Entry {
    bool present = false;
    std::string name;
    std::vector<std::pair<uint32_t, uint32_t>> layouts;     // (parameter block size, offset of the record pointer)
    uint64_t offset = 0, bytes = 0;
    uint32_t blend = 0;
};
struct Set {
    bool present = false;
    unsigned weighted = 0;
    std::vector<Entry> launches;
    fs::path bin;
    uint64_t binBytes = 0;
};

Set ReadSet(const fs::path& folder, unsigned n) {
    const std::string stem = "fp8_set" + std::to_string(n);
    Set set;
    set.bin = folder / (stem + ".bin");
    std::error_code error;
    set.binBytes = fs::file_size(set.bin, error);
    if (error || !set.binBytes) throw std::runtime_error("missing Ampere-set records " + set.bin.string());
    std::ifstream file(folder / (stem + ".txt"));
    if (!file) throw std::runtime_error("missing Ampere-set records " + (folder / (stem + ".txt")).string());
    unsigned evaluation = 0, number = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++number;
        std::istringstream words(line);
        std::string key;
        if (!(words >> key) || key[0] == '#') continue;
        const auto bad = [&](const char* why) { return std::runtime_error(stem + ".txt line " + std::to_string(number) + ": " + why); };
        if (key == "evaluation") {
            if (!(words >> evaluation) || evaluation == 0 || evaluation > kMaxLaunches) throw bad("a launch count from 1 to 256");
            set.launches.assign(evaluation, Entry{});
            continue;
        }
        if (key != "launch") throw bad("unknown line");
        unsigned launch = 0;
        if (!(words >> launch) || launch == 0 || launch >= set.launches.size()) throw bad("a launch number after `evaluation`, above 0");
        Entry e;
        e.present = true;
        bool haveOffset = false, haveBytes = false;
        for (std::string word; words >> word;) {
            if (word == "name") { if (!(words >> e.name)) throw bad("name"); }
            else if (word == "params") { uint32_t size = 0; if (!(words >> size) || size < 8 || size > kMaxParams) throw bad("params"); e.layouts.push_back({size, 0}); }
            else if (word == "record") {
                uint32_t at = 0;
                if (!(words >> at) || e.layouts.empty() || e.layouts.back().second || at + 8 > e.layouts.back().first) throw bad("record after params, inside them");
                e.layouts.back().second = at;
            }
            else if (word == "offset") { if (!(words >> e.offset)) throw bad("offset"); haveOffset = true; }
            else if (word == "bytes") { if (!(words >> e.bytes)) throw bad("bytes"); haveBytes = true; }
            else if (word == "blend") { if (!(words >> e.blend) || e.blend < 8) throw bad("blend"); }
            else throw bad("unknown word");
        }
        if (e.name.empty() || e.layouts.empty() || !haveOffset || !haveBytes || e.bytes == 0) throw bad("name, params, record, offset and bytes");
        for (const auto& [size, at] : e.layouts) if (!at || (e.blend && e.blend + 8 > size)) throw bad("a record (and blend) offset for every params");
        if (e.offset % 512 || e.offset > set.binBytes || e.bytes > set.binBytes - e.offset) throw bad("a 512-aligned record inside the .bin");
        if (set.launches[launch].present) throw bad("a launch given twice");
        set.launches[launch] = std::move(e);
        ++set.weighted;
    }
    if (!evaluation || !set.weighted) throw std::runtime_error(stem + ".txt has no launches");
    set.present = true;
    return set;
}

// ---- amp_*_manifest.txt: our kernel per launch ------------------------------------------------------------------
struct Dim { char mode = '='; uint32_t value = 0; };      // '=' the runtime's, 'n' this number, '*' / '/' the runtime's times / over value
using Dims = std::array<Dim, 3>;
struct Plan {
    bool present = false, nvidia = false;
    std::string file, entry;                // our kernel: cubin base name (no build ending) and symbol
    std::string name;                       // the runtime kernel it replaces, when the manifest says
    uint32_t params = 0, shared = 0;
    bool sharedRuntime = false;             // 'shared =': the runtime launch's own dynamic shared memory
    Dims grid{}, block{};
    std::string where;                      // manifest file and line, for messages
};
struct KernelDecl { std::string file, entry; uint32_t shared = 0; bool haveShared = false, sharedRuntime = false; Dims grid{}, block{}; bool haveGrid = false, haveBlock = false; };

std::string BaseName(std::string file) {
    for (const char* ending : {".sm120.cubin", ".sm86.cubin"}) {
        const size_t n = strlen(ending);
        if (file.size() > n && file.compare(file.size() - n, n, ending) == 0) return file.substr(0, file.size() - n);
    }
    return file;
}

Dim ParseDim(const std::string& word) {
    Dim d;
    const auto number = [&](const std::string& text) {
        size_t used = 0;
        const unsigned long value = std::stoul(text, &used, 10);
        if (used != text.size() || value == 0 || value > 0x7fffffff) throw std::runtime_error("bad dimension " + word);
        return uint32_t(value);
    };
    if (word == "=") return d;
    try {
        if (word[0] == '*' || word[0] == '/') { d.mode = word[0]; d.value = number(word.substr(1)); }
        else { d.mode = 'n'; d.value = number(word); }
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("bad dimension " + word);
    }
    return d;
}

Dims ParseDims(std::istringstream& words) {
    std::string first;
    if (!(words >> first)) throw std::runtime_error("three dimensions");
    std::vector<std::string> parts;
    if (first.find('x') != std::string::npos) {
        std::string part;
        std::istringstream split(first);
        while (std::getline(split, part, 'x')) parts.push_back(part);
    } else {
        parts.push_back(first);
        for (int i = 0; i < 2; ++i) { std::string w; if (!(words >> w)) throw std::runtime_error("three dimensions"); parts.push_back(w); }
    }
    if (parts.size() != 3) throw std::runtime_error("three dimensions");
    Dims dims;
    for (int i = 0; i < 3; ++i) dims[i] = ParseDim(parts[i]);
    return dims;
}

uint32_t Apply(const Dim& d, uint32_t runtime) {
    switch (d.mode) {
    case 'n': return d.value;
    case '*': return runtime * d.value;
    case '/': return (runtime + d.value - 1) / d.value;
    default: return runtime;
    }
}

std::vector<unsigned> ParseNumbers(const std::string& text) {
    std::vector<unsigned> numbers;
    std::string part;
    std::istringstream split(text);
    while (std::getline(split, part, ',')) {
        const size_t dash = part.find('-');
        size_t used = 0;
        const unsigned long a = std::stoul(part.substr(0, dash), &used, 10);
        if (used != (dash == std::string::npos ? part.size() : dash)) throw std::runtime_error("bad launch number " + part);
        unsigned long b = a;
        if (dash != std::string::npos) {
            const std::string tail = part.substr(dash + 1);
            b = std::stoul(tail, &used, 10);
            if (used != tail.size()) throw std::runtime_error("bad launch range " + part);
        }
        if (a == 0 || b < a || b >= kMaxLaunches) throw std::runtime_error("launch numbers from 1 to 255: " + part);
        for (unsigned long n = a; n <= b; ++n) numbers.push_back(unsigned(n));
    }
    if (numbers.empty()) throw std::runtime_error("no launch number");
    return numbers;
}

// 'shared <bytes>' (0..101376) or 'shared =' (the runtime launch's own amount, still capped at 101376 when launched).
void ReadShared(std::istringstream& words, uint32_t& bytes, bool& runtime) {
    std::string word;
    if (!(words >> word)) throw std::runtime_error("shared <bytes> or shared =");
    if (word == "=") { runtime = true; bytes = 0; return; }
    size_t used = 0;
    unsigned long value = 0;
    try { value = std::stoul(word, &used, 10); } catch (const std::exception&) { used = 0; }
    if (used != word.size() || value > kMaxShared) throw std::runtime_error("shared 0..101376 or =");
    bytes = uint32_t(value);
    runtime = false;
}

void ReadManifest(const fs::path& path, bool dry, std::array<Plan, kMaxLaunches>& plans, std::array<Plan, kMaxLaunches>& alts) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    const std::string stem = path.filename().string();
    std::map<std::string, KernelDecl> kernels;
    std::string line;
    unsigned number = 0;
    while (std::getline(file, line)) {
        ++number;
        const std::string where = stem + " line " + std::to_string(number);
        if (const size_t hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        std::istringstream words(line);
        std::string key;
        if (!(words >> key)) continue;
        try {
            if (key == "family") { std::string name; if (!(words >> name)) throw std::runtime_error("family <name>"); continue; }
            if (key == "kernel") {
                std::string id;
                if (!(words >> id)) throw std::runtime_error("kernel <id>");
                if (kernels.count(id)) throw std::runtime_error("kernel " + id + " given twice");
                KernelDecl k;
                for (std::string word; words >> word;) {
                    if (word == "file") { if (!(words >> k.file)) throw std::runtime_error("file"); k.file = BaseName(k.file); }
                    else if (word == "entry") { if (!(words >> k.entry)) throw std::runtime_error("entry"); }
                    else if (word == "shared") { ReadShared(words, k.shared, k.sharedRuntime); k.haveShared = true; }
                    else if (word == "grid") { k.grid = ParseDims(words); k.haveGrid = true; }
                    else if (word == "block") { k.block = ParseDims(words); k.haveBlock = true; }
                    else throw std::runtime_error("unknown word " + word);
                }
                if (k.file.empty() || k.entry.empty()) throw std::runtime_error("kernel <id> file <cubin> entry <symbol>");
                kernels[id] = k;
                continue;
            }
            if (key != "launch") throw std::runtime_error("unknown line " + key);
            std::string list;
            if (!(words >> list)) throw std::runtime_error("launch <numbers>");
            const std::vector<unsigned> launches = ParseNumbers(list);
            Plan p;
            p.present = true;
            p.where = where;
            bool haveShared = false, haveGrid = false, haveBlock = false;
            std::string kernelId;
            for (std::string word; words >> word;) {
                if (word == "kernel") { if (!(words >> kernelId)) throw std::runtime_error("kernel <id>"); }
                else if (word == "file") { if (!(words >> p.file)) throw std::runtime_error("file"); p.file = BaseName(p.file); }
                else if (word == "entry") { if (!(words >> p.entry)) throw std::runtime_error("entry"); }
                else if (word == "nvidia") p.nvidia = true;
                else if (word == "name") { if (!(words >> p.name)) throw std::runtime_error("name"); }
                else if (word == "params") { if (!(words >> p.params) || p.params < 8 || p.params > kMaxParams) throw std::runtime_error("params 8..512"); }
                else if (word == "shared") { ReadShared(words, p.shared, p.sharedRuntime); haveShared = true; }
                else if (word == "grid") { p.grid = ParseDims(words); haveGrid = true; }
                else if (word == "block") { p.block = ParseDims(words); haveBlock = true; }
                else throw std::runtime_error("unknown word " + word);
            }
            const int ways = int(!kernelId.empty()) + int(!p.file.empty() || !p.entry.empty()) + int(p.nvidia);
            if (ways != 1) throw std::runtime_error("one of: kernel <id>, file <cubin> entry <symbol>, nvidia");
            if (!kernelId.empty()) {
                const auto found = kernels.find(kernelId);
                if (found == kernels.end()) throw std::runtime_error("kernel " + kernelId + " is not declared above");
                const KernelDecl& k = found->second;
                p.file = k.file; p.entry = k.entry;
                if (!haveShared && k.haveShared) p.shared = k.shared, p.sharedRuntime = k.sharedRuntime;
                if (!haveGrid && k.haveGrid) p.grid = k.grid;
                if (!haveBlock && k.haveBlock) p.block = k.block;
            } else if (!p.nvidia && (p.file.empty() || p.entry.empty())) {
                throw std::runtime_error("file <cubin> entry <symbol>");
            }
            if (p.nvidia && !dry) throw std::runtime_error("'nvidia' (the runtime's own kernel) is a dry run and needs NR_AMP_DRY=1");
            if (p.nvidia && (haveShared || haveGrid || haveBlock)) throw std::runtime_error("'nvidia' keeps the runtime's launch shape");
            for (const unsigned n : launches) {
                if (plans[n].present) {
                    // A second line for the same launch is allowed for another runtime's parameter layout: both lines
                    // must give 'params', with different sizes (e.g. the input block: 272 bytes on 310.9.16, 264 on 310.8).
                    if (p.params && plans[n].params && p.params != plans[n].params && !alts[n].present) { alts[n] = p; continue; }
                    throw std::runtime_error("launch " + std::to_string(n) + " is already given by " + plans[n].where +
                                             " (a second line for a launch needs 'params' different from the first's)");
                }
                plans[n] = p;
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(where + ": " + error.what());
        }
    }
}

// ---- per-device state -------------------------------------------------------------------------------------------
struct Weights {
    bool ready = false;
    std::string failure;
    ComPtr<ID3D12Resource> buffer;
    uint64_t blendAt = 0;
};
struct Module { NVDX_ObjectHandle handle = nullptr; std::string failure; };
struct Kernel { NVDX_ObjectHandle handle = nullptr; std::string failure; };
struct Device {
    std::array<Weights, kMaxSets> sets;
    std::map<std::string, Module> modules;                                  // by cubin file name
    std::map<std::pair<std::string, std::string>, Kernel> kernels;          // by (cubin file name, entry)
};
struct ListState {
    ID3D12Device* device = nullptr;
    unsigned set = 0, index = 0, ours = 0, left = 0, dry = 0, extra = 0;
    bool failed = false;
    std::string reason;
};
struct Function { ID3D12Device* device = nullptr; std::string name; unsigned refs = 0; };

struct State {
    std::recursive_mutex mutex;
    bool enabled = false;
    Options options;
    std::array<Set, kMaxSets> sets;
    std::array<Plan, kMaxLaunches> plans;
    std::array<Plan, kMaxLaunches> alts;                                    // a second kernel for another parameter size
    unsigned planned = 0, plannedNvidia = 0, plannedExtra = 0;
    unsigned useSet = 0;
    std::map<NVDX_ObjectHandle, Function> functions;
    std::map<ID3D12Device*, Device> devices;
    std::map<ID3D12Device*, std::string> architecture;                      // CheckArchitecture's answer per device
    std::map<ID3D12GraphicsCommandList*, ListState> lists;
    uint64_t completed = 0, partial = 0, faults = 0, ours = 0, left = 0, dry = 0, extra = 0;
    std::string problem, notice, last;
    void (*log)(const char*) = nullptr;
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
};
State& S() { static State* state = new State; return *state; }

void Report(State& s, const std::string& text) noexcept {
    try {
        const std::string line = "Ampere set: " + text;
        if (s.log) s.log(line.c_str()); else fprintf(stderr, "%s\n", line.c_str());
    } catch (...) {
    }
}

const char* BuildName(Build b) { return b == Build::Sm86 ? "sm86" : "sm120"; }
bool Development(const Options& o) {
    return o.leaveMissing || o.dry || std::any_of(o.leave.begin(), o.leave.end(), [](uint64_t v) { return v != 0; });
}
bool Leave(const Options& o, unsigned n) { return n < 256 && ((o.leave[n / 64] >> (n % 64)) & 1); }
// The manifest line for launch n and the runtime's parameter size: the second line when it was written for that size.
const Plan& PlanFor(const State& s, unsigned n, uint32_t size) {
    const Plan& second = s.alts[n];
    return second.present && second.params == size ? second : s.plans[n];
}
// Our kernel takes exactly the parameter block its manifest line names: another size (another runtime's layout) would
// make it read the wrong fields, so it is a fault, never a launch.
void CheckParams(const Plan& p, unsigned n, uint32_t size) {
    if (p.params && p.params != size)
        throw std::runtime_error("launch " + std::to_string(n) + ": the runtime gave a " + std::to_string(size) + "-byte parameter block, " + p.entry +
                                 " takes " + std::to_string(p.params) + " bytes (" + p.where + ")");
}
fs::path CubinPath(const State& s, const std::string& base) { return s.options.assets / (base + "." + BuildName(s.options.build) + ".cubin"); }

std::string StatusLocked(State& s, bool counters);

void WriteReport(State& s) noexcept {
    if (s.options.report.empty()) return;
    try {
        std::ofstream file(s.options.report, std::ios::trunc);
        file << StatusLocked(s, true) << "\nlast: " << s.last << "\n";
    } catch (...) {
    }
}

// The records of set `n` on `device` (NrFp8's LoadWeights, with this layer's own buffer).
Weights& LoadWeights(State& s, Device& d, ID3D12Device* device, unsigned n) {
    Weights& w = d.sets[n];
    if (!w.failure.empty()) throw std::runtime_error(w.failure);
    if (w.ready) return w;
    try {
        const std::vector<unsigned char> blob = ReadFile(s.sets[n].bin);
        if (blob.size() != s.sets[n].binBytes) throw std::runtime_error(s.sets[n].bin.string() + " changed after it was checked");
        w.blendAt = (blob.size() + 511) / 512 * 512;
        const uint64_t bytes = (w.blendAt + 512 + 4096 + 255) / 256 * 256;    // (kernels read ahead of a record's end)
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&w.buffer)), "Ampere-set record buffer");
        ComPtr<ID3D12Resource> upload;
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "Ampere-set upload buffer");
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "Ampere-set upload map");
        memset(mapped, 0, size_t(bytes));
        memcpy(mapped, blob.data(), blob.size());
        memcpy(static_cast<unsigned char*>(mapped) + w.blendAt, &kBlendWeight, 2);
        upload->Unmap(0, nullptr);
        NrWeightUpload::Copy(device, w.buffer.Get(), upload.Get(), bytes);
        w.ready = true;
        Report(s, "records of fp8 set " + std::to_string(n) + " loaded (" + std::to_string(blob.size() >> 20) + " MB)");
        return w;
    } catch (const std::exception& error) {
        w.buffer.Reset();
        w.failure = error.what();
        throw;
    }
}

// Our kernel `entry` of cubin `base` (the chosen build) on `device`; a failure is remembered and thrown again.
NVDX_ObjectHandle KernelFor(State& s, Device& d, ID3D12Device* device, const std::string& base, const std::string& entry) {
    auto& k = d.kernels[{base, entry}];
    if (!k.failure.empty()) throw std::runtime_error(k.failure);
    if (k.handle) return k.handle;
    try {
        if (!s.createModule || !s.createFunction) throw std::runtime_error("the driver's CUDA entry points were not seen");
        auto& module = d.modules[base];
        if (!module.failure.empty()) throw std::runtime_error(module.failure);
        if (!module.handle) {
            try {
                const fs::path path = CubinPath(s, base);
                const auto cubin = ReadFile(path);
                NvCheck(s.createModule(device, cubin.data(), unsigned(cubin.size()), &module.handle),
                        "Ampere-set kernel module " + path.filename().string() + " (" + BuildName(s.options.build) + " build)");
            } catch (const std::exception& error) {
                module.handle = nullptr;
                module.failure = error.what();
                throw;
            }
        }
        NvCheck(s.createFunction(device, module.handle, entry.c_str(), &k.handle), "Ampere-set kernel " + entry + " in " + base);
        return k.handle;
    } catch (const std::exception& error) {
        k.handle = nullptr;
        k.failure = error.what();
        throw;
    }
}

// An evaluation with a set ends: wholly ours, partial (development switches), or a fault with its reason.
void Finish(State& s, ListState& l) {
    const Set& set = s.sets[l.set];
    if (!l.failed && l.index < set.launches.size())
        l.failed = true, l.reason = "the evaluation ended after " + std::to_string(l.index) + " launches, Model A's network has " + std::to_string(set.launches.size());
    if (!l.failed && l.ours + l.left + l.dry != set.weighted)
        l.failed = true, l.reason = std::to_string(l.ours + l.left + l.dry) + " of the " + std::to_string(set.weighted) + " weighted launches were seen";
    s.last = "set " + std::to_string(l.set) + ": ours " + std::to_string(l.ours) + ", left to NVIDIA " + std::to_string(l.left) +
             ", dry (NVIDIA's kernel by manifest) " + std::to_string(l.dry) + " of " + std::to_string(set.weighted) + " weighted launches: ";
    if (l.failed) {
        ++s.faults;
        s.problem = "Ampere set (fp8 set " + std::to_string(l.set) + "): " + l.reason;
        s.last += "FAULT, " + l.reason;
        if (s.notice != s.problem) { s.notice = s.problem; Report(s, s.problem); }
    } else if (Development(s.options)) {
        ++s.partial;
        s.last += "PARTIAL (development switches NR_AMP_SLOTS / NR_AMP_DRY are set: not wholly ours)";
        const std::string notice = "partial evaluation, not wholly ours: " + s.last;
        if (s.notice != notice) { s.notice = notice; Report(s, notice); }
    } else {
        ++s.completed;
        s.last += "wholly ours";
        if (s.completed == 1) Report(s, "first evaluation wholly on our kernels (" + s.last + ")");
    }
    WriteReport(s);
}

NvAPI_Status __cdecl CreateFunction(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& s = S();
    const auto result = s.createFunction(device, module, name, out);
    if (result == NVAPI_OK && name && out) {
        std::lock_guard guard(s.mutex);
        auto& f = s.functions[*out];
        if (f.refs && f.device == device && f.name == name) ++f.refs;
        else f = {device, name, 1u};
    }
    return result;
}

NvAPI_Status __cdecl DestroyFunction(ID3D12Device* device, NVDX_ObjectHandle function) {
    auto& s = S();
    const auto result = s.destroyFunction(device, function);
    if (result == NVAPI_OK) {
        std::lock_guard guard(s.mutex);
        const auto found = s.functions.find(function);
        if (found != s.functions.end() && --found->second.refs == 0) s.functions.erase(found);
    }
    return result;
}

NvAPI_Status __cdecl Launch(ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const Function* function = nullptr;
    if (kernels && count == 1) {
        const auto found = s.functions.find(kernels->hFunction);
        if (found != s.functions.end()) function = &found->second;
    }
    if (function && function->name == "cc_cb_clear") {         // a new evaluation
        const auto found = s.lists.find(commands);
        if (found != s.lists.end()) { Finish(s, found->second); s.lists.erase(found); }
        if (s.enabled && s.useSet && s.useSet < kMaxSets && s.sets[s.useSet].present) {
            ListState l;
            l.device = function->device;
            l.set = s.useSet;
            l.index = 1;
            const auto checked = s.architecture.find(l.device);
            if (checked == s.architecture.end()) l.failed = true, l.reason = "the GPU architecture was not checked for this device (CheckArchitecture)";
            else if (!checked->second.empty()) l.failed = true, l.reason = checked->second;
            s.lists[commands] = l;
        }
        return s.launch(commands, kernels, count);
    }
    const auto found = s.lists.find(commands);
    if (found == s.lists.end()) return s.launch(commands, kernels, count);
    ListState& l = found->second;
    const unsigned first = l.index;
    l.index += kernels ? count : 1;
    if (l.failed) return s.launch(commands, kernels, count);
    const Set& set = s.sets[l.set];
    unsigned char edited[kMaxParams];
    bool haveEdited = false;
    NVAPI_CU_KERNEL_LAUNCH_PARAMS theirs{};                    // the runtime's kernel and shape with our record
    try {
        if (!kernels || count == 0) throw std::runtime_error("launch " + std::to_string(first) + " has no kernel");
        if (first >= set.launches.size()) return s.launch(commands, kernels, count);   // after the network: the runtime's own work
        if (first + count > set.launches.size()) throw std::runtime_error("launch " + std::to_string(first) + " came in a chain past the network's end");
        if (count != 1) {
            for (unsigned k = 0; k < count; ++k)
                if (set.launches[first + k].present) throw std::runtime_error("launch " + std::to_string(first + k) + " came in a chain of " + std::to_string(count));
            return s.launch(commands, kernels, count);
        }
        const Entry& e = set.launches[first];
        const std::string name = function ? function->name : "an unknown kernel";
        if (!e.present) {                                               // a launch that reads no weight record (the repacks)
            const Plan& p = PlanFor(s, first, kernels->paramSize);
            if (!p.present || Leave(s.options, first)) return s.launch(commands, kernels, count);
            // A manifest may also give one of these a kernel of ours (optional; the parameter block stays the runtime's).
            if (name != p.name) throw std::runtime_error("launch " + std::to_string(first) + " is " + name + ", the manifest's is " + p.name);
            CheckParams(p, first, kernels->paramSize);
            NVAPI_CU_KERNEL_LAUNCH_PARAMS own = *kernels;
            own.hFunction = KernelFor(s, s.devices[l.device], l.device, p.file, p.entry);
            own.gridDim = {Apply(p.grid[0], kernels->gridDim.x), Apply(p.grid[1], kernels->gridDim.y), Apply(p.grid[2], kernels->gridDim.z)};
            own.blockDim = {Apply(p.block[0], kernels->blockDim.x), Apply(p.block[1], kernels->blockDim.y), Apply(p.block[2], kernels->blockDim.z)};
            own.dynSharedMemBytes = p.sharedRuntime ? kernels->dynSharedMemBytes : p.shared;
            const auto status = s.launch(commands, &own, 1);
            if (status != NVAPI_OK) throw std::runtime_error("launch " + std::to_string(first) + " (" + p.entry + " for " + name + ") failed: NVAPI status " + std::to_string(status));
            ++l.extra; ++s.extra;
            return status;
        }
        if (name != e.name) throw std::runtime_error("launch " + std::to_string(first) + " is " + name + ", Model A's is " + e.name);
        uint32_t at = 0;
        for (const auto& [size, record] : e.layouts) if (kernels->paramSize == size) at = record;
        if (!at || !kernels->pParams)
            throw std::runtime_error("launch " + std::to_string(first) + " (" + name + ") has a " + std::to_string(kernels->paramSize) + "-byte parameter block");
        Device& d = s.devices[l.device];
        const Weights& w = LoadWeights(s, d, l.device, l.set);
        memcpy(edited, kernels->pParams, kernels->paramSize);
        const uint64_t address = w.buffer->GetGPUVirtualAddress();
        const uint64_t record = address + e.offset;
        memcpy(edited + at, &record, 8);
        if (e.blend) {
            uint64_t runtimeBlend = 0;
            memcpy(&runtimeBlend, edited + e.blend, 8);
            if (runtimeBlend) { const uint64_t ours = address + w.blendAt; memcpy(edited + e.blend, &ours, 8); }
        }
        theirs = *kernels;
        theirs.pParams = edited;
        haveEdited = true;
        const Plan& p = PlanFor(s, first, kernels->paramSize);
        if (Leave(s.options, first) || (!p.present && s.options.leaveMissing)) {
            const auto status = s.launch(commands, &theirs, 1);
            if (status != NVAPI_OK) throw std::runtime_error("launch " + std::to_string(first) + " (" + name + ", left to NVIDIA) failed: NVAPI status " + std::to_string(status));
            ++l.left; ++s.left;
            return status;
        }
        if (!p.present) throw std::runtime_error("launch " + std::to_string(first) + " (" + name + ") has no Ampere kernel in the manifests");
        if (!p.nvidia) CheckParams(p, first, kernels->paramSize);
        if (p.nvidia) {
            const auto status = s.launch(commands, &theirs, 1);
            if (status != NVAPI_OK) throw std::runtime_error("launch " + std::to_string(first) + " (" + name + ", dry) failed: NVAPI status " + std::to_string(status));
            ++l.dry; ++s.dry;
            return status;
        }
        NVAPI_CU_KERNEL_LAUNCH_PARAMS own = theirs;
        own.hFunction = KernelFor(s, d, l.device, p.file, p.entry);
        own.gridDim = {Apply(p.grid[0], kernels->gridDim.x), Apply(p.grid[1], kernels->gridDim.y), Apply(p.grid[2], kernels->gridDim.z)};
        own.blockDim = {Apply(p.block[0], kernels->blockDim.x), Apply(p.block[1], kernels->blockDim.y), Apply(p.block[2], kernels->blockDim.z)};
        own.dynSharedMemBytes = p.sharedRuntime ? kernels->dynSharedMemBytes : p.shared;
        const uint64_t threads = uint64_t(own.blockDim.x) * own.blockDim.y * own.blockDim.z;
        if (!own.gridDim.x || !own.gridDim.y || !own.gridDim.z || !threads || threads > 1024 || own.dynSharedMemBytes > kMaxShared)
            throw std::runtime_error("launch " + std::to_string(first) + " (" + p.entry + "): launch shape outside the limits");
        const auto status = s.launch(commands, &own, 1);
        if (status != NVAPI_OK) throw std::runtime_error("launch " + std::to_string(first) + " (" + p.entry + " for " + name + ") failed: NVAPI status " + std::to_string(status));
        ++l.ours; ++s.ours;
        return status;
    } catch (const std::exception& error) {
        // Something is still launched in its place: a launch left out would leave the kernels after it waiting for flags
        // that never come. The evaluation is a fault (Finish), and its picture must not be shown.
        l.failed = true;
        l.reason = error.what();
        return haveEdited ? s.launch(commands, &theirs, 1) : s.launch(commands, kernels, count);
    }
}

std::string StatusLocked(State& s, bool counters) {
    std::string status = !s.enabled ? "off" : (s.completed || s.partial) ? "active" : "on, waiting for an evaluation";
    if (s.enabled) {
        status += std::string(", ") + BuildName(s.options.build) + " build, manifests cover " + std::to_string(s.planned) + " launches";
        if (s.plannedNvidia) status += " (" + std::to_string(s.plannedNvidia) + " of them dry: NVIDIA's kernel)";
        if (s.plannedExtra) status += " + " + std::to_string(s.plannedExtra) + " launches without a record";
        if (s.useSet) status += ", fp8 set " + std::to_string(s.useSet);
        if (Development(s.options)) status += ", DEVELOPMENT switches set (evaluations are partial)";
    }
    if (s.faults) status += " (" + s.problem + ")";
    if (!counters) return status;
    return status + " | wholly ours: " + std::to_string(s.completed) + " | partial: " + std::to_string(s.partial) + " | faults: " + std::to_string(s.faults) +
           " | launches ours: " + std::to_string(s.ours) + " | left to NVIDIA: " + std::to_string(s.left) + " | dry: " + std::to_string(s.dry) + " | record-less launches ours: " + std::to_string(s.extra);
}
}   // namespace

void FromEnvironment(Options& options) {
    const auto text = [](const wchar_t* name) {
        const wchar_t* value = _wgetenv(name);
        std::string out;
        for (const wchar_t* c = value; c && *c; ++c) out += char(*c < 128 ? *c : '?');
        return out;
    };
    const std::string arch = text(L"NR_AMP_ARCH");
    if (arch.empty() || arch == "sm120" || arch == "sm_120") options.build = Build::Sm120;
    else if (arch == "sm86" || arch == "sm_86") options.build = Build::Sm86;
    else throw std::runtime_error("NR_AMP_ARCH must be sm120 or sm86, not " + arch);
    options.leave = {};
    options.leaveMissing = false;
    std::string slots = text(L"NR_AMP_SLOTS");
    if (slots == "missing") options.leaveMissing = true;
    else if (!slots.empty()) {
        if (slots.size() > 2 && slots[0] == '0' && (slots[1] == 'x' || slots[1] == 'X')) slots = slots.substr(2);
        unsigned bit = 0;
        for (auto c = slots.rbegin(); c != slots.rend(); ++c) {
            if (*c == '_') continue;
            int v = -1;
            if (*c >= '0' && *c <= '9') v = *c - '0';
            else if (*c >= 'a' && *c <= 'f') v = *c - 'a' + 10;
            else if (*c >= 'A' && *c <= 'F') v = *c - 'A' + 10;
            if (v < 0) throw std::runtime_error("NR_AMP_SLOTS must be a hex number over launch numbers (bit n = launch n) or 'missing'");
            if (bit >= 256 && v) throw std::runtime_error("NR_AMP_SLOTS names a launch above 255");
            for (int b = 0; b < 4; ++b, ++bit) if (((v >> b) & 1) && bit < 256) options.leave[bit / 64] |= uint64_t(1) << (bit % 64);
        }
    }
    const std::string dry = text(L"NR_AMP_DRY");
    options.dry = !dry.empty() && dry != "0";
    const wchar_t* report = _wgetenv(L"NR_AMP_REPORT");
    options.report = report && *report ? fs::path(report) : fs::path();
}

void* WrapNvapi(unsigned id, void* original) {
    if (!original) return original;
    auto& s = S();
    std::lock_guard guard(s.mutex);
    switch (id) {
    case kCreateModule: s.createModule = static_cast<decltype(s.createModule)>(original); return original;     // only for our modules
    case kDestroyModule: s.destroyModule = static_cast<decltype(s.destroyModule)>(original); return original;
    case kCreateFunction: s.createFunction = static_cast<decltype(s.createFunction)>(original); return static_cast<void*>(&CreateFunction);
    case kLaunch: s.launch = static_cast<decltype(s.launch)>(original); return static_cast<void*>(&Launch);
    case kDestroyFunction: s.destroyFunction = static_cast<decltype(s.destroyFunction)>(original); return static_cast<void*>(&DestroyFunction);
    default: return original;
    }
}

void SetLog(void (*log)(const char*)) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.log = log;
}

void Enable(const Options& given) {
    Options options = given;
    if (options.manifests.empty()) options.manifests = options.assets;
    auto& s = S();
    std::lock_guard guard(s.mutex);
    for (auto& [device, d] : s.devices)
        if (!d.modules.empty() || std::any_of(d.sets.begin(), d.sets.end(), [](const Weights& w) { return w.ready; }))
            if (options.records != s.options.records || options.assets != s.options.assets || options.build != s.options.build)
                throw std::runtime_error("the Ampere-set folders and build cannot change while a device holds its records or kernels");
    // Our kernels per launch, from every family's manifest.
    std::array<Plan, kMaxLaunches> plans;
    std::vector<fs::path> manifests;
    std::error_code error;
    for (const auto& item : fs::directory_iterator(options.manifests, error)) {
        const std::string name = item.path().filename().string();
        if (item.is_regular_file() && name.rfind("amp_", 0) == 0 && name.size() > 13 && name.compare(name.size() - 13, 13, "_manifest.txt") == 0)
            manifests.push_back(item.path());
    }
    if (error) throw std::runtime_error("cannot list the Ampere-set manifests in " + options.manifests.string() + ": " + error.message());
    if (manifests.empty()) throw std::runtime_error("no Ampere-set manifest (amp_*_manifest.txt) in " + options.manifests.string());
    std::sort(manifests.begin(), manifests.end());
    std::array<Plan, kMaxLaunches> alts;
    for (const auto& path : manifests) ReadManifest(path, options.dry, plans, alts);
    // The records: every fp8 set in the folder, read as NrFp8 reads them.
    std::array<Set, kMaxSets> sets;
    unsigned found = 0;
    for (unsigned n = 1; n < kMaxSets; ++n) {
        if (!fs::exists(options.records / ("fp8_set" + std::to_string(n) + ".txt"), error)) continue;
        sets[n] = ReadSet(options.records, n);
        ++found;
    }
    if (!found) throw std::runtime_error("no fp8 record set in " + options.records.string() + " (fp8_set<n>.txt / .bin)");
    // Every manifest line must name a weighted launch of every set, and agree with it; every cubin must be there.
    unsigned planned = 0, plannedNvidia = 0, plannedExtra = 0;
    std::set<std::string> missing;
    for (unsigned n = 1; n < kMaxLaunches; ++n)
    for (const Plan* each : {&plans[n], &alts[n]}) {
        const Plan& p = *each;
        if (!p.present) continue;
        const bool second = each == &alts[n];               // counted once per launch, with the first line
        bool weighted = false, unweighted = false;
        for (unsigned k = 1; k < kMaxSets; ++k) {
            if (!sets[k].present) continue;
            const std::string where = p.where + ", launch " + std::to_string(n) + ", fp8 set " + std::to_string(k) + ": ";
            if (n >= sets[k].launches.size()) throw std::runtime_error(where + "past the network's launches");
            if (!sets[k].launches[n].present) {
                // A launch without a record (the two repacks): allowed with our kernel and the runtime kernel's name.
                if (p.nvidia || p.name.empty()) throw std::runtime_error(where + "not a weighted launch (a launch without a record needs 'name <runtime kernel>' and a kernel of ours)");
                unweighted = true;
                continue;
            }
            weighted = true;
            const Entry& e = sets[k].launches[n];
            if (!p.name.empty() && p.name != e.name) throw std::runtime_error(where + "the runtime's kernel is " + e.name + ", not " + p.name);
            if (p.params && std::none_of(e.layouts.begin(), e.layouts.end(), [&](const auto& l) { return l.first == p.params; }))
                throw std::runtime_error(where + "no " + std::to_string(p.params) + "-byte parameter block");
        }
        if (weighted && unweighted) throw std::runtime_error(p.where + ", launch " + std::to_string(n) + ": weighted in one fp8 set and not in another");
        if (second) {
            if (p.nvidia != plans[n].nvidia || plans[n].name != p.name)
                throw std::runtime_error(p.where + ", launch " + std::to_string(n) + ": the second line must be of the same kind and name as " + plans[n].where);
        } else if (weighted) ++planned, plannedNvidia += p.nvidia;
        else ++plannedExtra;
        if (!p.nvidia) {
            const fs::path cubin = options.assets / (p.file + "." + BuildName(options.build) + ".cubin");
            if (!fs::is_regular_file(cubin, error)) missing.insert(cubin.string());
        }
    }
    if (!missing.empty()) {
        std::string list;
        for (const auto& m : missing) list += (list.empty() ? "" : ", ") + m;
        throw std::runtime_error("Ampere-set kernels missing for the " + std::string(BuildName(options.build)) + " build: " + list);
    }
    s.plans = std::move(plans);
    s.alts = std::move(alts);
    s.sets = std::move(sets);
    s.planned = planned;
    s.plannedNvidia = plannedNvidia;
    s.plannedExtra = plannedExtra;
    s.options = options;
    for (auto& [device, d] : s.devices) {
        for (Weights& w : d.sets) w.failure.clear();
        for (auto& [file, m] : d.modules) m.failure.clear();
        for (auto& [key, k] : d.kernels) k.failure.clear();
    }
    s.enabled = true;
    s.notice.clear();
    unsigned weighted = 0;
    for (const Set& set : s.sets) if (set.present) weighted = std::max(weighted, set.weighted);
    std::string slots;
    if (options.leaveMissing) slots = ", NR_AMP_SLOTS=missing (launches without a kernel left to NVIDIA)";
    else if (Development(options) && !options.dry) slots = ", NR_AMP_SLOTS set (chosen launches left to NVIDIA)";
    Report(s, "on, " + std::to_string(manifests.size()) + " manifest(s) covering " + std::to_string(planned) + " of " + std::to_string(weighted) +
              " weighted launches" + (plannedNvidia ? " (" + std::to_string(plannedNvidia) + " dry)" : std::string()) + ", " + BuildName(options.build) +
              " build, kernels " + options.assets.string() + ", records " + options.records.string() + slots +
              (Development(options) ? "; DEVELOPMENT run: no evaluation counts as wholly ours" : ""));
}

void Disable() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.enabled = false;
    s.useSet = 0;
}

bool Enabled() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.enabled;
}

std::string CheckArchitecture(ID3D12Device* device) {
    auto& s = S();
    Build build;
    {
        std::lock_guard guard(s.mutex);
        if (const auto found = s.architecture.find(device); found != s.architecture.end()) return found->second;
        build = s.options.build;
    }
    // (Outside our lock: the NvAPI query below goes through the hooked nvapi_QueryInterface.)
    std::string reason;
    std::string seen = "unknown";
    try {
        if (!device) throw std::runtime_error("no device");
        const HMODULE api = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);     // kept loaded: the runtime uses it too
        if (!api) throw std::runtime_error("the system's nvapi64.dll is not available");
        using Query = void* (__cdecl*)(unsigned);
        const auto query = reinterpret_cast<Query>(GetProcAddress(api, "nvapi_QueryInterface"));
        if (!query) throw std::runtime_error("nvapi_QueryInterface is missing");
        const auto initialize = reinterpret_cast<decltype(&NvAPI_Initialize)>(query(0x0150e828));
        const auto enumerate = reinterpret_cast<decltype(&NvAPI_EnumPhysicalGPUs)>(query(0xe5ac921f));
        const auto logical = reinterpret_cast<decltype(&NvAPI_GetLogicalGPUFromPhysicalGPU)>(query(0xadd604d1));
        const auto info = reinterpret_cast<decltype(&NvAPI_GPU_GetLogicalGpuInfo)>(query(0x842b066e));
        const auto arch = reinterpret_cast<decltype(&NvAPI_GPU_GetArchInfo)>(query(0xd8265d24));
        if (!initialize || !enumerate || !logical || !info || !arch) throw std::runtime_error("the NvAPI architecture query is unavailable");
        NvCheck(initialize(), "NvAPI_Initialize");          // (not unloaded again: NvAPI stays in use by the runtime)
        const LUID selected = device->GetAdapterLuid();
        NvPhysicalGpuHandle handles[NVAPI_MAX_PHYSICAL_GPUS]{};
        NvU32 count = 0;
        NvCheck(enumerate(handles, &count), "NvAPI_EnumPhysicalGPUs");
        if (count > NVAPI_MAX_PHYSICAL_GPUS) throw std::runtime_error("NvAPI reported too many GPUs");
        bool matched = false;
        NV_GPU_ARCH_INFO architecture{};
        for (NvU32 i = 0; i < count; ++i) {
            NvLogicalGpuHandle group = nullptr;
            LUID luid{};
            NV_LOGICAL_GPU_DATA data{};
            data.version = NV_LOGICAL_GPU_DATA_VER;
            data.pOSAdapterId = &luid;
            if (logical(handles[i], &group) != NVAPI_OK || info(group, &data) != NVAPI_OK ||
                luid.LowPart != selected.LowPart || luid.HighPart != selected.HighPart) continue;
            if (data.physicalGpuCount != 1 || matched) throw std::runtime_error("the adapter is a linked group of GPUs");
            matched = true;
            architecture.version = NV_GPU_ARCH_INFO_VER;
            NvCheck(arch(handles[i], &architecture), "NvAPI_GPU_GetArchInfo");
        }
        if (!matched) throw std::runtime_error("no NVIDIA GPU has the D3D12 adapter's LUID");
        char text[96];
        snprintf(text, sizeof text, "architecture 0x%x implementation 0x%x", unsigned(architecture.architecture), unsigned(architecture.implementation));
        seen = text;
        // A cubin runs only on its own major architecture, at the same or a later minor version.
        bool can = false;
        if (build == Build::Sm120) can = architecture.architecture == NV_GPU_ARCHITECTURE_GB200;                            // GB20x: sm_120
        else can = (architecture.architecture == NV_GPU_ARCHITECTURE_GA100 && architecture.implementation != NV_GPU_ARCH_IMPLEMENTATION_GA100) ||
                   architecture.architecture == NV_GPU_ARCHITECTURE_AD100;                                                 // GA10x sm_86, AD10x sm_89
        if (!can)
            reason = std::string("the ") + BuildName(build) + " build of the Ampere set cannot run on this GPU (" + seen + "; " +
                     (build == Build::Sm120 ? "needs an RTX 50 series GPU" : "needs an RTX 30 (GA10x) or RTX 40 series GPU") + "); set NR_AMP_ARCH to the GPU's build";
    } catch (const std::exception& error) {
        reason = std::string("the GPU architecture could not be checked: ") + error.what();
    }
    std::lock_guard guard(s.mutex);
    s.architecture[device] = reason;
    Report(s, reason.empty() ? std::string(BuildName(build)) + " build allowed on this GPU (" + seen + ")" : reason);
    return reason;
}

void UseSet(unsigned set) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.useSet = set < kMaxSets ? set : 0;
    if (set >= kMaxSets) { ++s.faults; s.problem = "fp8 set " + std::to_string(set) + " does not exist"; }
}

bool SetComplete(unsigned set, std::string* missing) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const bool complete = s.enabled && set > 0 && set < kMaxSets && s.sets[set].present;
    if (!complete && missing) *missing = s.enabled ? "fp8_set" + std::to_string(set) + ".txt and .bin in " + s.options.records.string() : "the Ampere set is not enabled";
    return complete;
}

unsigned long long Faults() { auto& s = S(); std::lock_guard guard(s.mutex); return s.faults; }
unsigned long long Completed() { auto& s = S(); std::lock_guard guard(s.mutex); return s.completed; }
unsigned long long Partial() { auto& s = S(); std::lock_guard guard(s.mutex); return s.partial; }
std::string Problem() { auto& s = S(); std::lock_guard guard(s.mutex); return s.problem; }

void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept {
    try {
        auto& s = S();
        std::lock_guard guard(s.mutex);
        const auto found = s.lists.find(commands);
        if (found == s.lists.end()) return;
        Finish(s, found->second);
        s.lists.erase(found);
    } catch (...) {
    }
}

void ReleaseDevice(ID3D12Device* device) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    if (const auto found = s.devices.find(device); found != s.devices.end()) {
        // On a lost device these calls fail, which is fine: the objects die with it.
        for (auto& [key, k] : found->second.kernels) if (s.destroyFunction && k.handle) s.destroyFunction(device, k.handle);
        for (auto& [file, m] : found->second.modules) if (s.destroyModule && m.handle) s.destroyModule(device, m.handle);
        s.devices.erase(found);
    }
    s.architecture.erase(device);
    for (auto i = s.functions.begin(); i != s.functions.end();) i = i->second.device == device ? s.functions.erase(i) : std::next(i);
    std::erase_if(s.lists, [device](const auto& entry) { return entry.second.device == device; });
}

std::string Status(bool counters) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return StatusLocked(s, counters);
}
}
