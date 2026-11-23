// FP8 pass-in-one models: the runtime's kernels with weight records of ours; see NrFp8.h.
#include "NrWeightUpload.h"
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <nvapi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "NrFp8.h"

namespace NrFp8 {
namespace {
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

constexpr unsigned kCreateFunction = 0xe2436e22, kLaunch = 0x24973538, kDestroyFunction = 0xdf295ea6;
constexpr unsigned kMaxSets = 8;            // fp8_set1 .. fp8_set7
constexpr uint16_t kBlendWeight = 0x39eb;   // float16 0.7397: the output block's history blend weight every model of ours expects
constexpr size_t kMaxParams = 512;

void Check(HRESULT result, const char* what) {
    if (FAILED(result)) throw std::runtime_error(std::string(what) + " failed: D3D12 status " + std::to_string(unsigned(result)));
}

std::vector<unsigned char> ReadFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("missing FP8 asset " + path.string());
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (bytes.empty() || !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
        throw std::runtime_error("cannot read FP8 asset " + path.string());
    return bytes;
}

struct Entry {
    bool present = false;
    std::string name;
    std::vector<std::pair<uint32_t, uint32_t>> layouts;     // (parameter block size, offset of the record pointer)
    uint64_t offset = 0, bytes = 0;
    uint32_t blend = 0;                                     // offset of the history blend weight pointer (0: none)
};
struct Set {
    std::string failure;
    bool present = false;
    unsigned weighted = 0;
    std::vector<Entry> launches;                            // one per launch of an evaluation
    fs::path bin;                                           // the records, read when an evaluation first uses the set
    uint64_t binBytes = 0;
};
// A set's records on a device: uploaded when an evaluation first uses the set (only the sets in use take memory).
struct Weights {
    bool ready = false;
    std::string failure;
    ComPtr<ID3D12Resource> buffer;
    uint64_t blendAt = 0;                                   // our copy of the history blend weight, after the records
};
struct Device {
    std::array<Weights, kMaxSets> sets;
};
struct ListState {
    ID3D12Device* device = nullptr;
    unsigned set = 0, index = 0, replaced = 0;
    bool failed = false;
    std::string reason;
};
struct Function { ID3D12Device* device = nullptr; std::string name; unsigned refs = 0; };

struct State {
    std::recursive_mutex mutex;
    bool enabled = false;
    Options options;
    std::array<Set, kMaxSets> sets;
    std::map<NVDX_ObjectHandle, Function> functions;
    std::map<ID3D12Device*, Device> devices;
    std::map<ID3D12GraphicsCommandList*, ListState> lists;
    uint64_t completed = 0, faults = 0, replaced = 0;
    std::string problem, notice;
    void (*log)(const char*) = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
};
State& S() { static State* state = new State; return *state; }

void Report(State& s, const std::string& text) noexcept {
    try {
        const std::string line = "FP8 models: " + text;
        if (s.log) s.log(line.c_str()); else fprintf(stderr, "%s\n", line.c_str());
    } catch (...) {
    }
}

Set ReadSet(const fs::path& folder, unsigned n) {
    const std::string stem = "fp8_set" + std::to_string(n);
    Set set;
    set.bin = folder / (stem + ".bin");
    std::error_code error;
    set.binBytes = fs::file_size(set.bin, error);
    if (error || !set.binBytes) throw std::runtime_error("missing FP8 asset " + set.bin.string());
    std::ifstream file(folder / (stem + ".txt"));
    if (!file) throw std::runtime_error("missing FP8 asset " + (folder / (stem + ".txt")).string());
    unsigned evaluation = 0;
    std::string line;
    unsigned number = 0;
    while (std::getline(file, line)) {
        ++number;
        std::istringstream words(line);
        std::string key;
        if (!(words >> key) || key[0] == '#') continue;
        const auto bad = [&](const char* why) { return std::runtime_error(stem + ".txt line " + std::to_string(number) + ": " + why); };
        if (key == "evaluation") {
            if (!(words >> evaluation) || evaluation == 0 || evaluation > 4096) throw bad("a launch count from 1 to 4096");
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

// The records of set `n` on `device`, read from the file and uploaded the first time an evaluation uses the set.
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
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&w.buffer)), "FP8 weight buffer");
        ComPtr<ID3D12Resource> upload;
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "FP8 upload buffer");
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "FP8 upload map");
        memset(mapped, 0, size_t(bytes));
        memcpy(mapped, blob.data(), blob.size());
        memcpy(static_cast<unsigned char*>(mapped) + w.blendAt, &kBlendWeight, 2);
        upload->Unmap(0, nullptr);
        NrWeightUpload::Copy(device, w.buffer.Get(), upload.Get(), bytes);
        w.ready = true;
        Report(s, "set " + std::to_string(n) + " loaded (" + std::to_string(blob.size() >> 20) + " MB)");
        return w;
    } catch (const std::exception& error) {
        w.buffer.Reset();
        w.failure = error.what();
        throw;
    }
}

// An evaluation with a set ends: counted as whole, or as a fault with its reason.
void Finish(State& s, ListState& l) {
    if (!l.index) return; // Armed on this list, but the runtime never started a model.
    const Set& set = s.sets[l.set];
    if (!l.failed && l.index < set.launches.size())
        l.failed = true, l.reason = "the evaluation ended after " + std::to_string(l.index) + " launches, Model A's network has " + std::to_string(set.launches.size());
    if (!l.failed && l.replaced != set.weighted)
        l.failed = true, l.reason = std::to_string(l.replaced) + " of the " + std::to_string(set.weighted) + " weight records were the set's";
    if (l.failed) {
        ++s.faults;
        s.problem = "FP8 set " + std::to_string(l.set) + ": " + l.reason;
        if (s.notice != s.problem) { s.notice = s.problem; Report(s, s.problem); }
    } else {
        ++s.completed;
    }
}

NvAPI_Status __cdecl CreateFunction(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& s = S();
    const auto result = s.createFunction(device, module, name, out);
    if (result == NVAPI_OK && name && out) {
        std::lock_guard guard(s.mutex);
        auto& f = s.functions[*out];
        if (f.refs && f.device == device && f.name == name) ++f.refs;   // the same kernel handed out again
        else f = {device, name, 1u};                                    // (a handle value can be reused after its function was destroyed)
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
    const auto found = s.lists.find(commands);
    if (found == s.lists.end()) return s.launch(commands, kernels, count);
    ListState& l = found->second;
    if (function && function->name == "cc_cb_clear") {
        Finish(s, l);
        const unsigned set = l.set;
        l = {};
        l.device = function->device;
        l.set = set;
        l.index = 1;
        return s.launch(commands, kernels, count);
    }
    if (!l.index) return s.launch(commands, kernels, count);
    const unsigned first = l.index;
    l.index += kernels ? count : 1;
    if (l.failed) return NVAPI_ERROR;
    const Set& set = s.sets[l.set];
    try {
        if (!kernels || count == 0) throw std::runtime_error("launch " + std::to_string(first) + " has no kernel");
        // Launches after the network (the runtime's own work on the picture) read none of its weight records.
        if (first >= set.launches.size()) return s.launch(commands, kernels, count);
        if (first + count > set.launches.size()) throw std::runtime_error("launch " + std::to_string(first) + " came in a chain past the network's end");
        if (count != 1) {
            for (unsigned k = 0; k < count; ++k)
                if (set.launches[first + k].present) throw std::runtime_error("launch " + std::to_string(first + k) + " came in a chain of " + std::to_string(count));
            return s.launch(commands, kernels, count);
        }
        const Entry& e = set.launches[first];
        if (!e.present) return s.launch(commands, kernels, count);         // a launch that reads no weight record
        const std::string name = function ? function->name : "an unknown kernel";
        // The output block has a second form, which the runtime launches when the pass's intensity is not 1 in the
        // Standard style: the same parameters (weight record, history blend weight) followed by 128 bytes more.
        const bool blendForm = e.blend && name == e.name + "_simple_blend_full_rect";
        if (name != e.name && !blendForm) throw std::runtime_error("launch " + std::to_string(first) + " is " + name + ", Model A's is " + e.name);
        uint32_t at = 0;
        for (const auto& [size, record] : e.layouts) if (kernels->paramSize == size + (blendForm ? 128u : 0u)) at = record;
        if (!at || !kernels->pParams)
            throw std::runtime_error("launch " + std::to_string(first) + " (" + name + ") has a " + std::to_string(kernels->paramSize) + "-byte parameter block");
        const Weights& w = LoadWeights(s, s.devices[l.device], l.device, l.set);
        unsigned char edited[kMaxParams];
        memcpy(edited, kernels->pParams, kernels->paramSize);
        const uint64_t address = w.buffer->GetGPUVirtualAddress();
        const uint64_t record = address + e.offset;
        memcpy(edited + at, &record, 8);
        if (e.blend) {
            uint64_t theirs = 0;
            memcpy(&theirs, edited + e.blend, 8);
            if (theirs) { const uint64_t ours = address + w.blendAt; memcpy(edited + e.blend, &ours, 8); }
        }
        NVAPI_CU_KERNEL_LAUNCH_PARAMS own = *kernels;
        own.pParams = edited;
        const auto status = s.launch(commands, &own, 1);
        if (status != NVAPI_OK) {
            l.failed = true;
            l.reason = "launch " + std::to_string(first) + " (" + name + ") failed: NVAPI status " + std::to_string(status);
            return status;
        }
        ++l.replaced;
        ++s.replaced;
        return status;
    } catch (const std::exception& error) {
        l.failed = true;
        l.reason = error.what();
        return NVAPI_ERROR;
    }
}
}   // namespace

void* WrapNvapi(unsigned id, void* original) {
    if (!original) return original;
    auto& s = S();
    std::lock_guard guard(s.mutex);
    switch (id) {
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

void Enable(const Options& options) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    if (s.enabled && options.assets == s.options.assets) return;
    for (auto& [device, d] : s.devices)
        for (const Weights& w : d.sets)
            if (w.ready && options.assets != s.options.assets) throw std::runtime_error("the FP8 asset folder cannot change while a device holds its weights");
    std::array<Set, kMaxSets> sets;
    unsigned found = 0;
    for (unsigned n = 1; n < kMaxSets; ++n) {
        std::error_code missing;
        if (!fs::exists(options.assets / ("fp8_set" + std::to_string(n) + ".txt"), missing)) continue;
        try { sets[n] = ReadSet(options.assets, n); ++found; }
        catch (const std::exception& e) { sets[n].failure = e.what(); }
    }
    s.sets = std::move(sets);
    s.options = options;
    for (auto& [device, d] : s.devices)
        for (Weights& w : d.sets) w.failure.clear();
    s.enabled = true;
    s.notice.clear();
    Report(s, "on, " + std::to_string(found) + " set(s), assets " + options.assets.string());
}

void Disable() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.enabled = false;
    s.lists.clear();
}

bool Enabled() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.enabled;
}

void UseSet(unsigned set, ID3D12GraphicsCommandList* commands) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    auto found = s.lists.find(commands);
    if (found != s.lists.end()) {
        if (found->second.set == set) return;
        Finish(s, found->second);
        s.lists.erase(found);
    }
    if (set >= kMaxSets) { ++s.faults; s.problem = "FP8 set " + std::to_string(set) + " does not exist"; }
    else if (commands && set && s.enabled && s.sets[set].present) s.lists[commands].set = set;
}

bool SetComplete(unsigned set, std::string* missing) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const bool complete = s.enabled && set > 0 && set < kMaxSets && s.sets[set].present;
    if (!complete && missing) {
        *missing = set < kMaxSets && !s.sets[set].failure.empty() ? s.sets[set].failure
            : "fp8/fp8_set" + std::to_string(set) + ".txt and .bin";
    }
    return complete;
}

unsigned long long Faults() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.faults;
}

std::string Problem() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.problem;
}

unsigned long long Completed() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.completed;
}

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
    s.devices.erase(device);
    for (auto i = s.functions.begin(); i != s.functions.end();) i = i->second.device == device ? s.functions.erase(i) : std::next(i);
    std::erase_if(s.lists, [device](const auto& entry) { return entry.second.device == device; });
}

std::string Status(bool counters) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    std::string status = !s.enabled ? "off" : s.completed ? "active" : "on, waiting for an evaluation";
    if (s.lists.size() == 1) status += ", set " + std::to_string(s.lists.begin()->second.set);
    if (s.faults) status += " (" + s.problem + ")";
    if (!counters) return status;
    return status + " | evaluations on our records: " + std::to_string(s.completed) + " | faults: " + std::to_string(s.faults) +
           " | records replaced: " + std::to_string(s.replaced);
}
}
