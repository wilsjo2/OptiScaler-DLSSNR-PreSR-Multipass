// NVFP4 fused Swin replacement; see NrSwin4.h.
//
// The runtime launches a fused Swin block as one kernel with an 88-byte parameter block (96 bytes at width 32):
//   in, out, weight record, skip, rows and columns, window shift, producer flags, producer shift, own flags, the
//   downsampling layer's second output and its size.
// We launch nr_swin4 instead, with the same grid and block shape, the runtime's own buffers and flags, and our weight
// record. The runtime's neighbours wait for a block's flags without a time limit, so a kernel of ours must publish
// every window's flag exactly as the runtime's does; it does so unconditionally at its end.
#include <windows.h>
#include <bcrypt.h>
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
#include <sstream>
#include <stdexcept>
#include <vector>

#include "NrSwin4.h"

#pragma comment(lib, "bcrypt.lib")

namespace NrSwin4 {
namespace {
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

constexpr unsigned kCreateModule = 0xad1a677d, kCreateFunction = 0xe2436e22, kLaunch = 0x24973538,
                   kDestroyModule = 0x41c65285, kDestroyFunction = 0xdf295ea6;
constexpr unsigned kLayers = 44;            // fused Swin blocks of one Model A evaluation
constexpr size_t kMaxKnown = kLayers * 16;  // weight-record addresses remembered per device
constexpr unsigned kMaxSets = 14;            // weight sets: swin4.bin, swin4_trained.bin, swin4_set2.bin ... swin4_set13.bin
constexpr unsigned kPost = kLayers, kPre = kLayers + 1;   // the output and the input block, as record numbers of a set
// The output block's history blend weight, float16 0.7397 (the value every 4-bit model
// expects). The runtime passes a pointer to its own copy at parameter +104 of the output block; every
// output-block launch of a 4-bit model is pointed at this copy instead, so no value of the runtime's reaches the model.
constexpr uint16_t kBlendWeight = 0x39eb;
constexpr unsigned kBlendParam = 104;
enum Variant : unsigned { Plain, InpView, Downsample, Upsample, OutView };
constexpr const char* kVariantName[5] = {"plain", "inpview", "ds", "upsample", "outview"};

void Check(HRESULT result, const char* what) {
    if (FAILED(result)) throw std::runtime_error(std::string(what) + " failed: D3D12 status " + std::to_string(unsigned(result)));
}
void NvCheck(NvAPI_Status result, const char* what) {
    if (result != NVAPI_OK) throw std::runtime_error(std::string(what) + " failed: NVAPI status " + std::to_string(result));
}

std::vector<unsigned char> ReadAsset(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("missing NVFP4 asset " + path.string());
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (bytes.empty() || !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
        throw std::runtime_error("cannot read NVFP4 asset " + path.string());
    return bytes;
}


// What a kernel of the runtime is, from its name: cc_tinlayout_fused_swin_<n>h_<width>_<n>_[variant_][protocol_]fp8.
struct Kind {
    bool swin = false, clear = false;
    bool post = false, pre = false;         // the output and the input block (only their weight record is ever changed)
    bool probePost = false;                // extended intensity variant: diagnostic only, never a replacement
    unsigned heads = 0;
    Variant variant = Plain;
    bool waits = false, publishes = false;
};
Kind KindOf(const char* name) {
    Kind k;
    if (!strcmp(name, "cc_cb_clear")) { k.clear = true; return k; }
    if (!strcmp(name, "cc_tinlayout_fused_post_block_swin_1h_32_fp8")) { k.post = true; return k; }
    if (!strcmp(name, "cc_tinlayout_fused_post_block_swin_1h_32_fp8_simple_blend_full_rect")) { k.probePost = true; return k; }
    if (!strcmp(name, "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8")) { k.pre = true; return k; }
    std::string rest = name;
    const std::string prefix = "cc_tinlayout_fused_swin_";
    if (rest.rfind(prefix, 0) != 0) return k;
    rest.erase(0, prefix.size());
    const std::pair<const char*, unsigned> widths[] = {{"1h_32_1_", 1}, {"2h_64_2_", 2}, {"4h_128_4_", 4}, {"8h_256_8_", 8}};
    for (const auto& [text, heads] : widths)
        if (rest.rfind(text, 0) == 0) { k.heads = heads; rest.erase(0, strlen(text)); }
    if (!k.heads) return k;
    const std::pair<const char*, Variant> variants[] = {{"inpview_", InpView}, {"ds_", Downsample}, {"upsample_", Upsample}, {"outview_", OutView}};
    for (const auto& [text, variant] : variants)
        if (rest.rfind(text, 0) == 0) { k.variant = variant; rest.erase(0, strlen(text)); }
    if (rest == "tilesync_fp8") k.publishes = true;
    else if (rest == "wait_fp8") k.waits = true;
    else if (rest == "chained_fp8") k.waits = k.publishes = true;
    else if (rest != "fp8") return k;
    k.swin = true;
    return k;
}

struct Buffer {
    ComPtr<ID3D12Resource> resource;
    uint64_t bytes = 0;
    uint64_t Address() const { return resource->GetGPUVirtualAddress(); }
};
ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device, uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (heap == D3D12_HEAP_TYPE_DEFAULT) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = heap;
    ComPtr<ID3D12Resource> resource;
    Check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource)), "NVFP4 buffer allocation");
    return resource;
}

struct Layer { unsigned index = 0, heads = 0, variant = 0; uint64_t offset = 0, bytes = 0; };
struct Manifest {
    bool loaded = false;
    std::string failure, runtimeSha;
    std::array<Layer, kLayers> layers{};
    // `windows <heads> <n> above <count>`: launches of that width with more than <count> windows use the kernels built with
    // WINS n (files swin4_<width>_<variant>_w<n>.cubin: block (32, heads * n), grid (ceil(windows per row / n), rows)).
    std::array<unsigned, 4> windows{1, 1, 1, 1}, wideAbove{0, 0, 0, 0};     // [log2 heads]
    // Optional exact-arithmetic single-window builds for maps with few CTAs.
    // `small <heads> at_most <windows>` selects *_small.cubin; absent means unchanged.
    std::array<unsigned, 4> smallAtMost{};
    bool partial256 = false; // `partial 8`: whole4x4 tiles, partial8x8 windows, padded coarse views.
    std::vector<unsigned char> blob;
    bool trainedIsFirst = false;                // swin4.bin is not there: `blob` holds swin4_trained.bin
    // Sets 1..7 (Options::set). `blob`: the 4-bit records in swin4.bin's layout (empty: the set has none of its own);
    // `skip`: bit n = block n is not launched; `fp8`: records in the runtime's layout for the runtime's own kernels.
    struct Record { bool present = false; uint64_t offset = 0, bytes = 0; };
    struct LevelRecord { bool thin = false; bool present = false; unsigned stages = 0, hidden = 0, blocks = 4; uint64_t offset = 0, bytes = 0; };
    struct Set {
        bool present = false;
        std::vector<unsigned char> blob, fp8;
        uint64_t skip = 0;
        std::array<Record, kLayers + 2> records{};
        std::vector<unsigned char> levelBlob, postBlob, preBlob;
        LevelRecord post, pre;
        std::array<LevelRecord, kLayers> levels{}; // first fused-block number, specified as plan 143/147 in the text
        bool Reshapes() const { if (skip) return true; for (const auto& r : records) if (r.present) return true;
            for (const auto& r : levels) if (r.present) return true; return post.present || pre.present; }
    };
    std::array<Set, kMaxSets> sets;
};
// Kernels and weights of one device. They stay until ReleaseDevice(), because recorded command lists refer to them.
struct Device {
    bool weightsReady = false;
    std::string failure;
    Buffer weights;
    std::array<uint64_t, kMaxSets> setBase{};   // where each set's 4-bit records start in `weights` (0: the first set's)
    std::array<uint64_t, kMaxSets> fp8Base{};   // ... and its FP8 records
    std::array<uint64_t, kMaxSets> levelBase{}, postBase{}, preBase{};
    uint64_t blendAt = 0;                       // our own copy of the output block's history blend weight (float16)
    std::vector<ComPtr<IUnknown>> parked;      // what an upload that did not finish may still be using; freed with the device
    struct Kernel { NVDX_ObjectHandle module = nullptr, function = nullptr; std::string failure; };
    std::array<std::array<Kernel, 5>, 4> kernels{};         // [log2 heads][variant]
    std::array<std::array<Kernel, 5>, 4> wideKernels{};     // ... the multi-window builds
    std::array<std::array<Kernel, 5>, 4> smallKernels{};    // ... one-wave small-map builds
    std::array<std::array<Kernel, 5>, 4> partialKernels{};
    std::map<unsigned, Kernel> levelKernels;              // Level4 / PrePair4 / PostPair4, hidden, stages
    ComPtr<ID3D12Resource> probe, probeReadback;
    ComPtr<ID3D12Resource> probeSamples, probeSamplesReadback, probeProjection;
    uint64_t probeSampleCapacity = 0, probeSampleBytes = 0;
    std::array<unsigned char, 184> probeParams{};
    std::array<unsigned char, 272> probePreParams{};
    bool probeCaptured = false;
    bool probeProjectionDiagnostic = true;
    uint64_t probePreHistory = 0, probePreMotion = 0;
    std::map<uint64_t, unsigned> known;         // address of a runtime weight record -> block number (0..43, kPost, kPre)
};
// What one command list is in the middle of. An evaluation of the model starts with cc_cb_clear.
struct ListState {
    ID3D12Device* device = nullptr;
    unsigned blocks = 0;                        // fused Swin launches since the evaluation's cc_cb_clear
    bool sequence = true;                       // ... all of them in Model A's order so far
    std::array<uint64_t, kLayers> records{};    // their weight-record addresses
    std::array<unsigned char, 272> thinPreParams{};
    bool thinPreCaptured = false, thinPreUsed = false;
    uint64_t pre = 0, post = 0;                 // the weight records of the evaluation's input and output block
    // The last launched block of the level being run: what the next block reads when blocks between them are left out.
    struct Carry { uint64_t out = 0, oflags = 0; int32_t sx = 0, sy = 0; bool dirty = false; } carry;
    // Delay the fused dispatch until OutView supplies this evaluation's actual view destination.
    // The three original launches are retained for replay if the expected sequence is interrupted.
    struct SavedLaunch { NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{}; std::array<unsigned char, 96> params{}; };
    struct PendingLevel {
        unsigned saved = 0, first = 0, heads = 0, set = 0, blocks = 4;
        bool encoder = false;
        uint64_t weights = 0;
        NVDX_ObjectHandle function = nullptr;
        std::array<SavedLaunch, 9> original{};
    } level;
};
struct Target { ID3D12Device* device; NVDX_ObjectHandle module; Kind kind; unsigned refs; };

struct State {
    bool probeEnabled = false;
    struct LaunchTrace {
        struct Row { std::string name; int plan = -1; unsigned grid[3]{}, block[3]{}; };
        bool enabled = false, active = false, pending = false;
        ID3D12GraphicsCommandList* commands = nullptr;
        ComPtr<ID3D12QueryHeap> heap; ComPtr<ID3D12Resource> readback;
        uint64_t frequency = 0;
        std::map<NVDX_ObjectHandle, std::string> names;
        std::vector<Row> rows;
    } trace;
    std::recursive_mutex mutex;
    bool enabled = false;
    bool paused = false;                        // Pause(): this evaluation is the runtime's
    unsigned passSet = kDefaultSet;             // UseSet(): the set of the evaluations that follow
    uint64_t ioPassed = 0;                      // input / output block launches that ran with the runtime's own weights while we were on
    std::vector<unsigned> evaluationSets;       // optional harness overrides, consumed by UseSet
    size_t evaluationSetAt = 0;
    Options options;
    Manifest manifest;
    std::string waiting = "off", notice;        // what Status() says before anything was replaced; why launches were last left alone
    bool active = false;                        // a block was replaced since the last Enable()
    uint64_t replaced = 0, passed = 0, learnt = 0, smallReplaced = 0, partialReplaced = 0;
    uint64_t skipped = 0, rerecorded = 0;       // blocks not launched; launches of the runtime's kernels given a record of the set
    std::array<uint64_t, 4> levelReplaced{}, encoderReplaced{};
    uint64_t postReplaced = 0, preReplaced = 0;
    std::array<uint64_t, 4> perWidth{};
    std::wstring runtimePath;
    bool runtimeMatches = false;
    void (*log)(const char*) = nullptr;
    std::map<NVDX_ObjectHandle, Target> targets;
    std::map<ID3D12Device*, Device> devices;
    std::map<ID3D12GraphicsCommandList*, ListState> lists;
    std::map<ID3D12GraphicsCommandList*, bool> wideOpen;   // the list's last launch was a multi-window block of ours
    unsigned wideBarrier = 0;                   // NR_SWIN4_WIDE_BARRIER: 1 a barrier after a run of multi-window blocks, 2 one before it
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
};
State& S() { static State* state = new State; return *state; }

void Report(State& s, const std::string& text) {
    const std::string line = "NVFP4 fused blocks: " + text;
    if (s.log) s.log(line.c_str()); else fprintf(stderr, "%s\n", line.c_str());
}
// Why launches are being left to the runtime; logged when it changes. Must not throw.
void Notice(State& s, const std::string& text) noexcept {
    try {
        if (s.notice == text) return;
        s.notice = text;
        Report(s, text);
    } catch (...) {
    }
}

unsigned Log2(unsigned heads) { return heads == 1 ? 0 : heads == 2 ? 1 : heads == 4 ? 2 : 3; }

void UavBarrier(ID3D12GraphicsCommandList* commands) noexcept {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    commands->ResourceBarrier(1, &barrier);
}
// A multi-window block takes a whole SM. Work recorded after a run of such blocks may be told to wait for them
// (wideBarrier & 1), so that its small blocks do not sit in the SMs the wide blocks still need.
void LeaveWide(State& s, ID3D12GraphicsCommandList* commands) noexcept {
    try {
        const auto found = s.wideOpen.find(commands);
        if (found == s.wideOpen.end() || !found->second) return;
        found->second = false;
        if (s.wideBarrier & 1) UavBarrier(commands);
    } catch (...) {
    }
}

void LoadManifest(State& s) {
    auto& m = s.manifest;
    if (!m.failure.empty()) throw std::runtime_error(m.failure);
    if (m.loaded) return;
    try {
        // A package may leave the first set out: swin4_trained.bin then stands in its place (set 0 and set 1 are the same records).
        std::error_code noFirst;
        m.trainedIsFirst = !std::filesystem::exists(s.options.assets / "swin4.bin", noFirst);
        m.blob = ReadAsset(s.options.assets / (m.trainedIsFirst ? "swin4_trained.bin" : "swin4.bin"));
        std::ifstream file(s.options.assets / "swin4.txt");
        if (!file) throw std::runtime_error("missing NVFP4 asset " + (s.options.assets / "swin4.txt").string());
        std::array<bool, kLayers> seen{};
        for (std::string line; std::getline(file, line);) {
            std::istringstream in(line);
            std::string key; in >> key;
            if (key == "runtime") { in >> m.runtimeSha; continue; }
            if (key == "partial") {
                unsigned heads=0; std::string tail;
                if (!(in>>heads) || heads!=8 || (in>>tail)) throw std::runtime_error("invalid NVFP4 partial-window line: " + line);
                m.partial256=true; continue;
            }
            if (key == "small") {
                unsigned heads = 0, count = 0; std::string word;
                in >> heads >> word >> count;
                if (!in || word != "at_most" || (heads != 1 && heads != 2 && heads != 4 && heads != 8) || !count)
                    throw std::runtime_error("invalid NVFP4 small-map manifest line: " + line);
                m.smallAtMost[heads == 1 ? 0 : heads == 2 ? 1 : heads == 4 ? 2 : 3] = count;
                continue;
            }
            if (key == "windows") {                 // kernels built with WINS: block (32, heads * n), grid (ceil(windows per row / n), rows)
                unsigned heads = 0, n = 0, above = 0; std::string word;
                in >> heads >> n >> word >> above;
                if (!in || word != "above" || (heads != 1 && heads != 2 && heads != 4 && heads != 8) || n < 1 || n > 4) throw std::runtime_error("invalid NVFP4 manifest line: " + line);
                m.windows[heads == 1 ? 0 : heads == 2 ? 1 : heads == 4 ? 2 : 3] = n;
                m.wideAbove[heads == 1 ? 0 : heads == 2 ? 1 : heads == 4 ? 2 : 3] = above;
                continue;
            }
            if (key != "layer") continue;
            unsigned number; std::string k1, k2, k3, k4, k5; Layer l;
            in >> number >> k1 >> l.index >> k2 >> l.heads >> k3 >> l.variant >> k4 >> l.offset >> k5 >> l.bytes;
            if (!in || number >= kLayers || seen[number] || k1 != "index" || k2 != "heads" || k3 != "variant" || k4 != "offset" || k5 != "bytes" ||
                (l.heads != 1 && l.heads != 2 && l.heads != 4 && l.heads != 8) || l.variant > 4)
                throw std::runtime_error("invalid NVFP4 manifest line: " + line);
            if (l.bytes > m.blob.size() || l.offset > m.blob.size() - l.bytes || l.offset % 256)
                throw std::runtime_error("NVFP4 manifest points outside swin4.bin");
            m.layers[number] = l; seen[number] = true;
        }
        for (bool present : seen) if (!present) throw std::runtime_error("the NVFP4 manifest does not describe 44 fused blocks");
        if (m.runtimeSha.size() != 64) throw std::runtime_error("the NVFP4 manifest does not name the runtime its weights were made from");
        // The further sets are optional. Their 4-bit records share the manifest, so a file has to be swin4.bin's size.
        for (unsigned n = 1; n < kMaxSets; ++n) {
            auto& set = m.sets[n];
            const std::string stem = n == 1 ? "swin4_trained" : "swin4_set" + std::to_string(n);
            std::error_code missing;
            if (n == 1 && m.trainedIsFirst) {
                set.present = true;             // its 4-bit records are `blob` itself
            } else if (std::filesystem::exists(s.options.assets / (stem + ".bin"), missing)) {
                set.blob = ReadAsset(s.options.assets / (stem + ".bin"));
                if (set.blob.size() != m.blob.size()) throw std::runtime_error(stem + ".bin is not the size of swin4.bin");
                set.present = true;
            }
            std::ifstream notes(s.options.assets / (stem + ".txt"));
            if (!notes) continue;
            set.present = true;
            for (std::string line; std::getline(notes, line);) {
                std::istringstream in(line);
                std::string key; in >> key;
                if (key == "skip") {
                    for (unsigned block; in >> block;) {
                        // Only a block inside a level can be left out: its neighbours wait for flags without a time limit.
                        if (block >= kLayers || m.layers[block].variant != Plain) throw std::runtime_error(stem + ".txt: block " + std::to_string(block) + " cannot be left out");
                        set.skip |= 1ull << block;
                    }
                } else if (key == "fp8") {
                    unsigned block = 0; std::string k1, k2; Manifest::Record r;
                    in >> block >> k1 >> r.offset >> k2 >> r.bytes;
                    if (!in || block >= kLayers + 2 || k1 != "offset" || k2 != "bytes" || r.offset % 512 || !r.bytes) throw std::runtime_error("invalid line in " + stem + ".txt: " + line);
                    r.present = true;
                    set.records[block] = r;
                } else if (key == "thin") {
                    std::string end, a, b; Manifest::LevelRecord r;
                    in >> end >> a >> r.offset >> b >> r.bytes;
                    if (!in || (end != "pre" && end != "post") || a != "offset" || b != "bytes" ||
                        r.offset % 256 || r.bytes < 256 || r.bytes % 256)
                        throw std::runtime_error("invalid Thin4 line in " + stem);
                    auto& destination = end == "pre" ? set.pre : set.post;
                    if (destination.present) throw std::runtime_error("duplicate Thin4 end in " + stem);
                    r.present = r.thin = true; r.hidden = 64; r.stages = 0; destination = r;
                } else if (key == "post" || key == "pre") {
                    auto& destination = key == "pre" ? set.pre : set.post;
                    std::string a, b, c, e; Manifest::LevelRecord r;
                    in >> a >> r.stages >> b >> r.hidden >> c >> r.offset >> e >> r.bytes;
                    if (!in || a != "stages" || b != "hidden" || c != "offset" || e != "bytes" ||
                        r.stages < 1 || r.stages > 8 || (r.hidden != 64 && r.hidden != 128 && r.hidden != 256) ||
                        r.offset % 256 || r.bytes < 256 || r.bytes % 256 || destination.present)
                        throw std::runtime_error("invalid Pre/PostPair4 line in " + stem + ".txt: " + line);
                    r.present = true; destination = r;
                } else if (key == "level") {
                    unsigned index = 0; std::string a, b, c, e; Manifest::LevelRecord r;
                    in >> index >> a >> r.stages >> b >> r.hidden >> c >> r.offset >> e >> r.bytes;
                    const auto first = std::find_if(m.layers.begin(), m.layers.end(), [index](const Layer& l) { return l.index == index; });
                    const bool encoder = index == 1 || index == 5 || index == 9 || index == 15;
                    r.blocks=(index==15 || index==129)?8u:(index==9 || index==137)?6u:4u;
                    const unsigned heads=(index==15 || index==129)?8u:(index==9 || index==137)?4u:(index==143 || index==5)?2u:1u;
                    if (!in || (!encoder && index != 129 && index != 137 && index != 143 && index != 147) || a != "stages" || b != "hidden" || c != "offset" || e != "bytes" ||
                        r.stages < 1 || r.stages > 8 || (r.hidden != 64 && r.hidden != 128 && r.hidden != 256) || r.offset % 256 ||
                        r.bytes < 256 || r.bytes % 256 || first == m.layers.end() || first->variant != (encoder ? InpView : Upsample) ||
                        std::distance(first, m.layers.end()) < r.blocks || first->heads != heads ||
                        first[r.blocks-1].variant != (encoder ? Downsample : OutView))
                        throw std::runtime_error("invalid Level4 line in " + stem + ".txt: " + line);
                    for(unsigned j=1;j+1<r.blocks;++j)if(first[j].variant!=Plain || first[j].heads!=heads)
                        throw std::runtime_error("Level4 level has a non-inner block");
                    const size_t block = size_t(first - m.layers.begin());
                    if (set.levels[block].present) throw std::runtime_error("duplicate Level4 record in " + stem);
                    r.present = true; set.levels[block] = r;
                }
            }
            bool records = false;
            for (const auto& r : set.records) records |= r.present;
            if (records) {
                set.fp8 = ReadAsset(s.options.assets / (stem + "_fp8.bin"));
                for (const auto& r : set.records)
                    if (r.present && (r.bytes > set.fp8.size() || r.offset > set.fp8.size() - r.bytes)) throw std::runtime_error(stem + ".txt points outside " + stem + "_fp8.bin");
            }
            if (set.pre.thin && !set.post.thin) throw std::runtime_error(stem + ": Thin4 PRE requires Thin4 POST (no full skip is written)");
            for (bool pre : {false, true}) {
                const auto& r = pre ? set.pre : set.post;
                if (!r.present) continue;
                auto& blob = pre ? set.preBlob : set.postBlob;
                blob = ReadAsset(s.options.assets / (stem + (pre ? "_pre.bin" : "_post.bin")));
                if (r.bytes > blob.size() || r.offset > blob.size() - r.bytes)
                    throw std::runtime_error(stem + ": Pre/PostPair4 record outside its file");
                const auto* bytes = blob.data() + r.offset;
                uint32_t h[64]; memcpy(h, bytes, sizeof h);
                const auto region = [&](uint64_t off, uint64_t size) { return off >= 256 && off % 16 == 0 && size <= r.bytes && off <= r.bytes - size; };
                if (r.thin) {
                    const unsigned inputs = 64;
                    if (h[0] != 0x364e4854u || h[1] != 3 || h[2] != unsigned(!pre) || h[3] != inputs || h[4] != 64 || h[5] != 32)
                        throw std::runtime_error(stem + ": incompatible Thin4 header");
                    const unsigned sizes[3] = {2u * (inputs / 64u) * 1152u, 2304u, 1152u};
                    for (unsigned i=0; i<3; ++i)
                        if (!region(h[8+i],sizes[i]) || !region(h[12+i],i==2?128u:256u))
                            throw std::runtime_error(stem + ": Thin4 region outside record");
                    if (set.records[pre ? kPre : kPost].present) throw std::runtime_error(stem + ": Thin4 conflicts with FP8 record");
                    continue;
                }
                if (h[0] != (pre ? 0x31523450u : 0x31503450u) || h[1] != 1 || h[2] != 32 || h[3] != r.hidden || h[4] != r.stages || !region(h[5], 128))
                    throw std::runtime_error(stem + ": incompatible Pre/PostPair4 header");
                uint32_t io[16]; memcpy(io, bytes + h[5] + 64, sizeof io);
                if (pre) {
                    if (io[12] < 128 || !region(uint64_t(h[5]) + io[12], 1024)) throw std::runtime_error(stem + ": PrePair4 lift outside its file");
                } else {
                    for (unsigned slot : {8u, 11u, 14u})
                        if (io[slot] < 128 || !region(uint64_t(h[5]) + io[slot], slot == 14 ? 512 : 128))
                            throw std::runtime_error(stem + ": PostPair4 I/O record outside its file");
                }
                for (unsigned i = 0; i < r.stages; ++i) {
                    const uint64_t off = h[8 + i];
                    if (off % 256 || !region(off, 128) || h[16 + i] > 5) throw std::runtime_error(stem + ": invalid Pre/PostPair4 stage");
                    uint32_t offsets[5]; memcpy(offsets, bytes + off + 64, sizeof offsets);
                    const uint64_t sizes[5] = {uint64_t(r.hidden / 32) * 1152, 2ull * (r.hidden / 64) * 1152, 4ull * r.hidden, 256, 256};
                    for (unsigned k = 0; k < 5; ++k)
                        if (offsets[k] < 128 || !region(off + offsets[k], sizes[k])) throw std::runtime_error(stem + ": Pre/PostPair4 stage outside its record");
                }
                if (set.records[pre ? kPre : kPost].present) throw std::runtime_error(stem + ": Pre/PostPair4 conflicts with an FP8 record");
            }
            bool levels = false;
            for (const auto& r : set.levels) levels |= r.present;
            if (levels) {
                set.levelBlob = ReadAsset(s.options.assets / (stem + "_level.bin"));
                for (unsigned block = 0; block < kLayers; ++block) {
                    const auto& r = set.levels[block]; if (!r.present) continue;
                    if (r.bytes > set.levelBlob.size() || r.offset > set.levelBlob.size() - r.bytes)
                        throw std::runtime_error(stem + ": Level4 record outside its file");
                    const auto* bytes = set.levelBlob.data() + r.offset;
                    uint32_t h[64]; memcpy(h, bytes, sizeof h);
                    const unsigned C = 32 * m.layers[block].heads;
                    const bool encoder = m.layers[block].variant == InpView;
                    const auto region = [&](uint64_t off, uint64_t size) { return off >= 256 && off % 16 == 0 && size <= r.bytes && off <= r.bytes - size; };
                    if (h[0] != (encoder ? 0x3145344cu : 0x3156344cu) || h[1] != 1 || h[2] != C || h[3] != r.hidden || h[4] != r.stages ||
                        !region(h[5], encoder ? (2 * C / 32) * ((C+63)/64) * 1152 : (C / 32) * (2 * C / 64) * 1152) ||
                        !region(h[6], C * (encoder ? 8 : 4)) || !region(h[7], C * (encoder ? 8 : 4)))
                        throw std::runtime_error(stem + ": incompatible Level4 header");
                    for (unsigned i = 0; i < r.stages; ++i) {
                        const uint64_t off = h[8 + i];
                        if (off % 256 || !region(off, 128) || h[16 + i] > 5)
                            throw std::runtime_error(stem + ": invalid Level4 stage");
                        uint32_t offsets[5]; memcpy(offsets, bytes + off + 64, sizeof offsets);
                        const uint64_t sizes[5] = {uint64_t(r.hidden / 32) * (2 * C / 64) * 1152,
                            uint64_t(2 * C / 32) * (r.hidden / 64) * 1152, 4ull * r.hidden, 8ull * C, 8ull * C};
                        for (unsigned k = 0; k < 5; ++k)
                            if (offsets[k] < 128 || !region(off + offsets[k], sizes[k]))
                                throw std::runtime_error(stem + ": Level4 stage points outside its record");
                    }
                    if (set.skip & (((1ull<<r.blocks)-1) << block)) throw std::runtime_error(stem + ": Level4 conflicts with ordinary skip");
                }
            }
        }
        m.loaded = true;
    } catch (const std::exception& error) {
        m.blob.clear();
        m.failure = error.what();
        throw;
    }
}

// Every weight these layers use is in their own files: any NR runtime is accepted. Whether its launches are the ones
// these kernels replace is still checked launch by launch (kernel names, parameter sizes, shapes).
bool RuntimeMatches(State&) { return GetModuleHandleW(L"nvngx_dlssnr.dll") != nullptr; }

// An output-block parameter block (184 bytes, the runtime's layout) gets our blend weight in place of the runtime's.
// A null pointer (no blend weight given) stays null.
void OwnBlend(const Device& d, unsigned char* post) {
    uint64_t theirs; memcpy(&theirs, post + kBlendParam, 8);
    if (!theirs) return;
    const uint64_t ours = d.weights.Address() + d.blendAt;
    memcpy(post + kBlendParam, &ours, 8);
}

void LoadWeights(State& s, Device& d, ID3D12Device* device) {
    if (!d.failure.empty()) throw std::runtime_error(d.failure);
    if (d.weightsReady) return;
    try {
        const auto& blob = s.manifest.blob;
        // One buffer: the first set, then every further set's 4-bit records and FP8 records (the runtime's kernels read
        // theirs from addresses that are multiples of 512).
        uint64_t cursor = blob.size();
        d.setBase.fill(0); d.fp8Base.fill(0); d.levelBase.fill(0); d.postBase.fill(0); d.preBase.fill(0);
        for (unsigned n = 1; n < kMaxSets; ++n) {
            const auto& set = s.manifest.sets[n];
            if (!set.blob.empty()) { cursor = (cursor + 255) / 256 * 256; d.setBase[n] = cursor; cursor += set.blob.size(); }
            else if (n > 1) d.setBase[n] = d.setBase[1];        // a set without 4-bit records of its own runs the trained ones
            if (!set.fp8.empty()) { cursor = (cursor + 4096 + 511) / 512 * 512; d.fp8Base[n] = cursor; cursor += set.fp8.size(); }
            if (!set.levelBlob.empty()) { cursor = (cursor + 255) / 256 * 256; d.levelBase[n] = cursor; cursor += set.levelBlob.size(); }
            if (!set.postBlob.empty()) { cursor = (cursor + 255) / 256 * 256; d.postBase[n] = cursor; cursor += set.postBlob.size(); }
            if (!set.preBlob.empty()) { cursor = (cursor + 255) / 256 * 256; d.preBase[n] = cursor; cursor += set.preBlob.size(); }
        }
        cursor = (cursor + 511) / 512 * 512; d.blendAt = cursor; cursor += 512;
        const uint64_t bytes = (cursor + 4096 + 255) / 256 * 256;           // a kernel reads 16 bytes at a time
        d.weights = {CreateBuffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), bytes};
        auto upload = CreateBuffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "NVFP4 upload map");
        memset(mapped, 0, size_t(bytes));
        memcpy(mapped, blob.data(), blob.size());
        for (unsigned n = 1; n < kMaxSets; ++n) {
            const auto& set = s.manifest.sets[n];
            if (!set.blob.empty()) memcpy(static_cast<unsigned char*>(mapped) + d.setBase[n], set.blob.data(), set.blob.size());
            if (!set.fp8.empty()) memcpy(static_cast<unsigned char*>(mapped) + d.fp8Base[n], set.fp8.data(), set.fp8.size());
            if (!set.levelBlob.empty()) memcpy(static_cast<unsigned char*>(mapped) + d.levelBase[n], set.levelBlob.data(), set.levelBlob.size());
            if (!set.postBlob.empty()) memcpy(static_cast<unsigned char*>(mapped) + d.postBase[n], set.postBlob.data(), set.postBlob.size());
            if (!set.preBlob.empty()) memcpy(static_cast<unsigned char*>(mapped) + d.preBase[n], set.preBlob.data(), set.preBlob.size());
        }
        memcpy(static_cast<unsigned char*>(mapped) + d.blendAt, &kBlendWeight, 2);
        upload->Unmap(0, nullptr);
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> commands; ComPtr<ID3D12Fence> fence;
        // Once the copy is submitted, a failure below must not free what it may still be using.
        struct Park {
            Device& d; ComPtr<ID3D12Resource>& upload; ComPtr<ID3D12CommandQueue>& queue; ComPtr<ID3D12CommandAllocator>& allocator;
            ComPtr<ID3D12GraphicsCommandList>& commands; bool armed = false;
            ~Park() {
                if (!armed) return;
                try {
                    for (IUnknown* object : {static_cast<IUnknown*>(d.weights.resource.Get()), static_cast<IUnknown*>(upload.Get()),
                                             static_cast<IUnknown*>(queue.Get()), static_cast<IUnknown*>(allocator.Get()),
                                             static_cast<IUnknown*>(commands.Get())})
                        d.parked.emplace_back(object);
                } catch (...) {
                }
            }
        } park{d, upload, queue, allocator, commands};
        Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "NVFP4 upload queue");
        Check(device->CreateCommandAllocator(queueDesc.Type, IID_PPV_ARGS(&allocator)), "NVFP4 upload allocator");
        Check(device->CreateCommandList(0, queueDesc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)), "NVFP4 upload list");
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "NVFP4 upload fence");
        D3D12_RESOURCE_BARRIER transition{};
        transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition = {d.weights.resource.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST};
        commands->ResourceBarrier(1, &transition);
        commands->CopyBufferRegion(d.weights.resource.Get(), 0, upload.Get(), 0, bytes);
        std::swap(transition.Transition.StateBefore, transition.Transition.StateAfter);
        commands->ResourceBarrier(1, &transition);
        Check(commands->Close(), "NVFP4 upload close");
        ID3D12CommandList* lists[] = {commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        park.armed = true;
        Check(queue->Signal(fence.Get(), 1), "NVFP4 upload signal");
        HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!done) throw std::runtime_error("NVFP4 upload event failed");
        const HRESULT armed = fence->SetEventOnCompletion(1, done);
        const DWORD waited = SUCCEEDED(armed) ? WaitForSingleObject(done, 20000) : WAIT_FAILED;
        CloseHandle(done);
        Check(armed, "NVFP4 upload wait");
        if (waited != WAIT_OBJECT_0) throw std::runtime_error("NVFP4 weight upload did not complete");
        park.armed = false;
        d.weightsReady = true;
    } catch (const std::exception& error) {
        d.weights = {};
        d.failure = error.what();
        throw;
    }
}

NVDX_ObjectHandle KernelFor(State& s, Device& d, ID3D12Device* device, unsigned heads, unsigned variant, unsigned windows, bool smallMap, bool partialMap) {
    auto& k = (partialMap ? d.partialKernels : smallMap ? d.smallKernels : windows > 1 ? d.wideKernels : d.kernels)[Log2(heads)][variant];
    if (!k.failure.empty()) throw std::runtime_error(k.failure);
    if (k.function) return k.function;
    try {
        if (!s.createModule || !s.createFunction) throw std::runtime_error("the driver's CUDA entry points were not seen");
        const std::string name = "swin4_" + std::to_string(32 * heads) + "_" + kVariantName[variant] +
            (partialMap ? "_partial" : smallMap ? "_small" : windows > 1 ? "_w" + std::to_string(windows) : "") + ".cubin";
        const auto cubin = ReadAsset(s.options.assets / name);
        // An older GPU refuses this module (it is built for the RTX 50 series' NVFP4 tensor-core instruction).
        NvCheck(s.createModule(device, cubin.data(), unsigned(cubin.size()), &k.module), "NVFP4 kernel module (needs an RTX 50 series GPU)");
        NvCheck(s.createFunction(device, k.module, "nr_swin4", &k.function), "nr_swin4");
        return k.function;
    } catch (const std::exception& error) {
        if (s.destroyFunction && k.function) s.destroyFunction(device, k.function);
        if (s.destroyModule && k.module) s.destroyModule(device, k.module);
        k.function = k.module = nullptr;
        k.failure = error.what();
        throw;
    }
}

// The set in use: Options::set where the folder has it, else the first (0).
unsigned ChosenSet(const State& s) {
    if (s.passSet == 0 || (s.passSet < kMaxSets && s.manifest.sets[s.passSet].present)) return s.passSet;
    return s.options.set < kMaxSets && s.manifest.sets[s.options.set].present ? s.options.set : 0;
}

NVDX_ObjectHandle LevelKernelFor(State& s, Device& d, ID3D12Device* device, unsigned heads, const Manifest::LevelRecord& r) {
    auto& k = d.levelKernels[(heads << 20) | (r.hidden << 8) | r.stages];
    if (!k.failure.empty()) throw std::runtime_error(k.failure);
    if (k.function) return k.function;
    try {
        if (!s.createModule || !s.createFunction) throw std::runtime_error("the driver's CUDA entry points were not seen");
        // heads=0/16: post/pre; 1/2: decoder; 33/34: encoder.
        const bool encoder = heads > 32;
        const auto name = r.thin ? std::string(heads == 16 ? "thin4_pre.cubin" : "thin4_post.cubin") : (encoder ? "encoder4_" + std::to_string(32 * (heads - 32)) : heads == 16 ? "prepair4" : heads ? "level4_" + std::to_string(32 * heads) : "postpair4") + "_h" + std::to_string(r.hidden) + "_s" + std::to_string(r.stages) + ".cubin";
        const char* entry = r.thin ? (heads == 16 ? "nr_thin_pre" : "nr_thin_post") : encoder ? "nr_encoder4" : heads == 16 ? "nr_pre_pair4" : heads ? "nr_level4" : "nr_post_pair4";
        const auto cubin = ReadAsset(s.options.assets / name);
        NvCheck(s.createModule(device, cubin.data(), unsigned(cubin.size()), &k.module), "Level4 module");
        NvCheck(s.createFunction(device, k.module, entry, &k.function), entry);
        return k.function;
    } catch (const std::exception& error) {
        if (s.destroyFunction && k.function) s.destroyFunction(device, k.function);
        if (s.destroyModule && k.module) s.destroyModule(device, k.module);
        k.function = k.module = nullptr; k.failure = error.what(); throw;
    }
}

#pragma pack(push, 8)
struct Params {                     // nvfp4/kernels/swin4.cu
    uint64_t in, out, w, skip, pflags, oflags, dsOut;
    int32_t H, W, sx, sy, psx, psy, d0, d1, e0, e1;
    uint32_t options, reserved;
};
#pragma pack(pop)
static_assert(sizeof(Params) == 104, "nr_swin4 parameter block");

// The runtime's parameter block, either size, in one form.
struct Original {
    uint64_t in = 0, out = 0, w = 0, skip = 0, pflags = 0, oflags = 0, dsOut = 0;
    int32_t H = 0, W = 0, sx = 0, sy = 0, psx = 0, psy = 0, dims[2]{}, skipDims[2]{};
};
Original Read(const NVAPI_CU_KERNEL_LAUNCH_PARAMS& k, unsigned heads) {
    const auto* p = static_cast<const unsigned char*>(k.pParams);
    const auto u64 = [&](unsigned at) { uint64_t v; memcpy(&v, p + at, 8); return v; };
    const auto s32 = [&](unsigned at) { int32_t v; memcpy(&v, p + at, 4); return v; };
    Original o;
    o.in = u64(0); o.out = u64(8); o.w = u64(16);
    if (heads == 1) {
        if (k.paramSize != 96) throw std::runtime_error("the runtime's width-32 parameter block is not 96 bytes");
        o.H = s32(24); o.W = s32(28); o.sx = s32(32); o.sy = s32(36); o.pflags = u64(40); o.psx = s32(48); o.psy = s32(52);
        o.oflags = u64(56); o.dsOut = u64(64); o.dims[0] = s32(72); o.dims[1] = s32(76); o.skip = u64(80);
        o.skipDims[0] = s32(88); o.skipDims[1] = s32(92);
    } else {
        if (k.paramSize != 88) throw std::runtime_error("the runtime's parameter block is not 88 bytes");
        o.skip = u64(24); o.H = s32(32); o.W = s32(36); o.sx = s32(40); o.sy = s32(44); o.pflags = u64(48); o.psx = s32(56);
        o.psy = s32(60); o.oflags = u64(64); o.dsOut = u64(72); o.dims[0] = s32(80); o.dims[1] = s32(84);
    }
    return o;
}

// Our launch for the runtime's launch `k` of block `number`; throws before anything is recorded if it cannot be done.
void Replace(State& s, ID3D12GraphicsCommandList* commands, ID3D12Device* device, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& k, const Kind& kind,
             unsigned number) {
    const Layer& layer = s.manifest.layers[number];
    if (layer.heads != kind.heads || layer.variant != unsigned(kind.variant)) throw std::runtime_error("the launch is not the kernel this block uses in Model A");
    if (!k.pParams) throw std::runtime_error("the launch has no parameter block");
    const Original o = Read(k, kind.heads);
    const auto shift = [](int v) { return v == 0 || v == -4; };
    const bool partialMap = kind.heads == 8 && s.manifest.partial256 && (o.H % 8 || o.W % 8);
    const int multiple = partialMap ? 4 : 8;
    // Whole windows and whole tiles on every level below: the sizes the model uses (frames are padded to multiples of 64).
    if (o.H <= 0 || o.W <= 0 || o.H % multiple || o.W % multiple || o.H > 8192 || o.W > 8192 || !shift(o.sx) || !shift(o.sy) || !shift(o.psx) || !shift(o.psy))
        throw std::runtime_error("unsupported map " + std::to_string(o.H) + "x" + std::to_string(o.W) +
            " shift " + std::to_string(o.sx) + "," + std::to_string(o.sy) + " producer " + std::to_string(o.psx) + "," + std::to_string(o.psy));
    const unsigned gridX = unsigned(o.W - std::min(o.sx, 0) + 7) / 8, gridY = unsigned(o.H - std::min(o.sy, 0) + 7) / 8;
    if (k.gridDim.x != gridX || k.gridDim.y != gridY || k.gridDim.z != 1 || k.blockDim.x != 32 || k.blockDim.y != kind.heads || k.blockDim.z != 1 ||
        k.dynSharedMemBytes)
        throw std::runtime_error("the runtime's launch has an unexpected shape");
    if (!o.in || !o.out || (kind.waits && !o.pflags) || (kind.publishes && !o.oflags)) throw std::runtime_error("a pointer of the launch is null");
    auto& d = s.devices[device];
    LoadWeights(s, d, device);
    Params p{};
    p.in = o.in; p.out = o.out; p.w = d.weights.Address() + d.setBase[ChosenSet(s)] + layer.offset;
    p.pflags = kind.waits ? o.pflags : 0;
    p.oflags = kind.publishes ? o.oflags : 0;
    p.H = o.H; p.W = o.W; p.sx = o.sx; p.sy = o.sy; p.psx = o.psx; p.psy = o.psy;
    p.d0 = o.H; p.d1 = o.W; p.e0 = o.H; p.e1 = o.W;
    p.options = kind.heads == 1 ? s.options.twoTerms : 0;
    switch (kind.variant) {
    case InpView:
    case OutView:
        if (o.dims[0] > 0 && o.dims[1] > 0 && (o.dims[0] != o.H || o.dims[1] != o.W)) throw std::runtime_error("a view of another size than the map");
        break;
    case Downsample:
        // Partial-window builds explicitly zero the padded coarse strips.
        if (!o.dsOut || o.dims[0] != (partialMap ? (o.H/2+3)&~3 : o.H/2) ||
            o.dims[1] != (partialMap ? (o.W/2+3)&~3 : o.W/2)) throw std::runtime_error("unsupported padded downsample output");
        p.dsOut = o.dsOut;
        break;
    case Upsample:
        if (!o.skip) throw std::runtime_error("the upsampling launch has no skip map");
        if (kind.heads == 1 && ((o.skipDims[0] > 0 && o.skipDims[0] != o.H) || (o.skipDims[1] > 0 && o.skipDims[1] != o.W)))
            throw std::runtime_error("a skip map of another size");
        p.skip = o.skip;
        p.d0 = (o.H/2+3)&~3; p.d1 = (o.W/2+3)&~3; // the actual padded coarse-view strides
        break;
    default:
        break;
    }
    NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};
    // A multi-window block takes a whole SM, which pays once a layer has more windows than the SMs hold standard blocks.
    const unsigned level = Log2(kind.heads);
    const unsigned windows = !partialMap && s.manifest.windows[level] > 1 && gridX * gridY > s.manifest.wideAbove[level] ? s.manifest.windows[level] : 1;
    const bool smallMap = windows == 1 && s.manifest.smallAtMost[level] && gridX * gridY <= s.manifest.smallAtMost[level];
    launch.hFunction = KernelFor(s, d, device, kind.heads, unsigned(kind.variant), windows, smallMap, partialMap);
    if (windows > 1) {
        bool& open = s.wideOpen[commands];
        if (!open && (s.wideBarrier & 2)) UavBarrier(commands);
        open = true;
    } else {
        LeaveWide(s, commands);
    }
    launch.gridDim = k.gridDim;
    launch.gridDim.x = (k.gridDim.x + windows - 1) / windows;
    launch.blockDim = k.blockDim;
    launch.blockDim.y = k.blockDim.y * windows;
    launch.pParams = &p;
    launch.paramSize = sizeof p;
    launch.dynSharedMemBytes = 0;
    NvCheck(s.launch(commands, &launch, 1), "NVFP4 kernel launch");
    if (smallMap) ++s.smallReplaced;
    if (partialMap) ++s.partialReplaced;
}

// Where the runtime's parameter block (88 bytes; 96 at width 32) holds what a block after left-out blocks is given.
struct Fields { unsigned sx, sy, pflags, psx, psy, oflags, size; };
Fields FieldsOf(unsigned heads) { return heads == 1 ? Fields{32, 36, 40, 48, 52, 56, 96} : Fields{40, 44, 48, 56, 60, 64, 88}; }

// `edited` = the launch's parameter block reading what the last launched block of the level wrote: its output map, its
// flags and its window shift. A level's blocks write two maps in turn; where this block would now write the map it
// reads, it writes the other one, which the left-out block was to fill. (The level's last block writes a map of its
// own; after an odd number of left-out blocks in an encoder level that map is the one it reads, window by window.)
void Rewire(const NVAPI_CU_KERNEL_LAUNCH_PARAMS& k, const Fields& f, const ListState::Carry& carry, unsigned char* edited) noexcept {
    memcpy(edited, k.pParams, f.size);
    uint64_t in = 0, out = 0;
    memcpy(&in, edited, 8); memcpy(&out, edited + 8, 8);
    uint64_t oflags = 0;
    memcpy(&oflags, edited + f.oflags, 8);
    memcpy(edited, &carry.out, 8);
    if (out == carry.out && oflags) memcpy(edited + 8, &in, 8);     // (a block that publishes flags is not the level's last)
    memcpy(edited + f.pflags, &carry.oflags, 8);
    memcpy(edited + f.psx, &carry.sx, 4);
    memcpy(edited + f.psy, &carry.sy, 4);
}

// After a launch (ours or the runtime's) of a fused block: what the next block of the level may be handed.
void Track(ListState* list, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& k, const Kind& kind, const Fields& f, bool plain) noexcept {
    if (!list) return;
    if (!plain || !kind.publishes) { list->carry = {}; return; }      // a level's last block, or a launch we cannot read
    const auto* p = static_cast<const unsigned char*>(k.pParams);
    ListState::Carry c;
    c.dirty = kind.waits && list->carry.dirty;
    memcpy(&c.out, p + 8, 8); memcpy(&c.oflags, p + f.oflags, 8); memcpy(&c.sx, p + f.sx, 4); memcpy(&c.sy, p + f.sy, 4);
    list->carry = c;
}

// No custom dispatch has been recorded yet. Replay a buffered prefix if recognition or validation fails.
NvAPI_Status FlushLevel(State& s, ID3D12GraphicsCommandList* commands, ListState* list) {
    if (!list || !list->level.saved) return NVAPI_OK;
    auto pending = list->level; list->level = {}; list->carry = {};
    LeaveWide(s, commands); UavBarrier(commands);
    for (unsigned i = 0; i < pending.saved; ++i) {
        auto k = pending.original[i].launch; k.pParams = pending.original[i].params.data();
        const auto status = s.launch(commands, &k, 1);
        if (status != NVAPI_OK) return status;
        UavBarrier(commands); // fallback cannot depend on polling progress of concurrently scheduled CTAs
    }
    return NVAPI_OK;
}

bool LevelLaunch(State& s, ID3D12GraphicsCommandList* commands, ID3D12Device* device, ListState* list,
                 const NVAPI_CU_KERNEL_LAUNCH_PARAMS& k, const Kind& kind, int number, bool plain,
                 const Manifest::Set* set, unsigned chosen, NvAPI_Status& result) {
    if (!list) return false;
    auto& pending = list->level;
    const auto save = [&]() {
        auto& entry = pending.original[pending.saved++]; entry.launch = k;
        memcpy(entry.params.data(), k.pParams, k.paramSize);
    };
    if (pending.saved) {
        const bool expected = plain && set && chosen == pending.set && number == int(pending.first + pending.saved) &&
            kind.heads == pending.heads && kind.variant == (pending.saved+1 == pending.blocks ? (pending.encoder ? Downsample : OutView) : Plain);
        if (!expected) { result = FlushLevel(s, commands, list); return result != NVAPI_OK; }
        auto original = pending.original[0].launch; original.pParams = pending.original[0].params.data();
        const Original first = Read(original, pending.heads), last = Read(k, kind.heads);
        if (last.H != first.H || last.W != first.W) throw std::runtime_error("Level4 sequence changed extent");
        if (pending.saved+1 < pending.blocks) { save(); result = NVAPI_OK; return true; }
        if (!last.out || last.out == first.in || last.out == first.skip) throw std::runtime_error("Level4 final view aliases an input");
        if (pending.encoder && (!last.dsOut || last.dsOut == first.in || last.dsOut == last.out ||
            last.dims[0] < first.H / 2 || last.dims[1] < first.W / 2 ||
            last.dims[0] > (first.H/2+3)/4*4 || last.dims[1] > (first.W/2+3)/4*4))
            throw std::runtime_error("Encoder4 pooled output aliases an input or requires padding");
        Params p{};
        p.in = first.in; p.skip = first.skip; p.out = last.out; p.w = pending.weights;
        p.H = p.e0 = first.H; p.W = p.e1 = first.W; p.d0 = first.H / 2; p.d1 = first.W / 2;
        if (pending.encoder) { p.skip = 0; p.dsOut = last.dsOut; p.d0=last.dims[0];p.d1=last.dims[1]; }
        else { // Upsample does not read the trailing view override; C64+ rounds its coarse view to four.
            p.d0=kind.heads==1 ? first.H/2 : (first.H/2+3)/4*4;
            p.d1=kind.heads==1 ? first.W/2 : (first.W/2+3)/4*4;
        }
        NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};
        launch.hFunction = pending.function; launch.gridDim = {unsigned((p.W+7) / 8), unsigned((p.H+7) / 8), 1};
        launch.blockDim = {32, 2, 1}; launch.pParams = &p; launch.paramSize = sizeof p;
        LeaveWide(s, commands); UavBarrier(commands);
        NvCheck(s.launch(commands, &launch, 1), "Level4 launch");
        UavBarrier(commands); // the level's original boundary barrier also remains in the caller's command list
        ++(pending.encoder ? s.encoderReplaced : s.levelReplaced)[Log2(kind.heads)]; ++s.replaced; ++s.perWidth[Log2(kind.heads)]; s.skipped += pending.blocks-1;
        pending = {}; list->carry = {}; s.active = true; result = NVAPI_OK;
        return true;
    }
    if (!set || number < 0 || !set->levels[number].present) return false;
    const auto& r = set->levels[number];
    const Original first = Read(k, kind.heads);
    const bool encoder = kind.variant == InpView;
    if ((!encoder && kind.variant != Upsample) || kind.waits || !first.in || (!encoder && !first.skip) || !first.out ||
        (kind.heads<4 && (first.sx || first.sy)) || (first.sx!=0 && first.sx!=-4) || (first.sy!=0 && first.sy!=-4) ||
        first.H <= 0 || first.W <= 0 || first.H % 4 || first.W % 4 || first.H > 8192 || first.W > 8192 ||
        k.gridDim.x != unsigned((first.W-std::min(first.sx,0)+7)/8) || k.gridDim.y != unsigned((first.H-std::min(first.sy,0)+7)/8) || k.gridDim.z != 1 ||
        k.blockDim.x != 32 || k.blockDim.y != kind.heads || k.blockDim.z != 1 || k.dynSharedMemBytes ||
        (!encoder && kind.heads == 1 && ((first.skipDims[0] && first.skipDims[0] != first.H) || (first.skipDims[1] && first.skipDims[1] != first.W))) ||
        (encoder && ((first.dims[0] && first.dims[0] != first.H) || (first.dims[1] && first.dims[1] != first.W))))
        throw std::runtime_error("unsupported Level4 input/shape");
    auto& d = s.devices[device]; LoadWeights(s, d, device);
    const auto function = LevelKernelFor(s, d, device, kind.heads + (encoder ? 32 : 0), r); // load before suppressing
    pending.first = unsigned(number); pending.heads = kind.heads; pending.set = chosen;
    pending.blocks = r.blocks;
    pending.encoder = encoder;
    pending.function = function; pending.weights = d.weights.Address() + d.levelBase[chosen] + r.offset;
    save(); list->carry = {}; result = NVAPI_OK; return true;
}

// An evaluation ended (its list starts another, or the host said so): if it was a complete Model A evaluation, its
// weight records are now known by address.
void Learn(State& s, ListState& list) noexcept {
    try {
        if (!list.device || list.blocks != kLayers || !list.sequence) return;
        auto& known = s.devices[list.device].known;
        bool fresh = false;
        for (unsigned n = 0; n < kLayers; ++n) {
            const auto found = known.find(list.records[n]);
            if (found != known.end() && found->second != n) { known.clear(); break; }      // addresses were reused: start again
        }
        if (known.size() + kLayers > kMaxKnown) known.clear();
        for (unsigned n = 0; n < kLayers; ++n) fresh |= known.emplace(list.records[n], n).second;
        if (list.post) known.emplace(list.post, kPost);
        if (list.pre) known.emplace(list.pre, kPre);
        if (fresh) { ++s.learnt; Report(s, "Model A's fused blocks recognised (" + std::to_string(known.size()) + " weight records)"); }
    } catch (...) {
    }
}

NvAPI_Status __cdecl CreateModule(ID3D12Device* device, const void* blob, NvU32 bytes, NVDX_ObjectHandle* out) {
    return S().createModule(device, blob, bytes, out);
}

NvAPI_Status __cdecl CreateFunction(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& s = S();
    const auto result = s.createFunction(device, module, name, out);
    if (result == NVAPI_OK && name && out) {
        const Kind kind = KindOf(name);
        std::lock_guard guard(s.mutex);
        if (s.trace.enabled) s.trace.names[*out] = name;
        const auto found = s.targets.find(*out);
        if (!kind.swin && !kind.clear && !kind.post && !kind.pre && !kind.probePost) {
            if (found != s.targets.end()) s.targets.erase(found);      // a handle value can be reused after its function was destroyed
        } else if (found != s.targets.end() && found->second.module == module && found->second.device == device &&
                   found->second.kind.heads == kind.heads && found->second.kind.variant == kind.variant &&
                   found->second.kind.waits == kind.waits && found->second.kind.publishes == kind.publishes && found->second.kind.clear == kind.clear &&
                   found->second.kind.post == kind.post && found->second.kind.pre == kind.pre && found->second.kind.probePost==kind.probePost) {
            ++found->second.refs;               // the same kernel handed out again, to another feature
        } else {
            s.targets[*out] = {device, module, kind, 1u};
        }
        // A feature of the runtime is being created: its weights may land where another model's were.
        if (kind.swin) {
            const auto d = s.devices.find(device);
            if (d != s.devices.end()) d->second.known.clear();
        }
    }
    return result;
}

void RecordPostProbe(State& s, ID3D12GraphicsCommandList* commands, ID3D12Device* device, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& native) {
    if (!s.probeEnabled || !native.pParams || (native.paramSize != 184 && native.paramSize!=312)) return;
    auto& d=s.devices[device];auto& k=d.levelKernels[0xff040000u];
    if (!k.function) {
        auto cubin=ReadAsset(s.options.assets/"post_probe.cubin");
        NvCheck(s.createModule(device,cubin.data(),unsigned(cubin.size()),&k.module),"post probe module");
        NvCheck(s.createFunction(device,k.module,"nr_post_probe",&k.function),"post probe function");
    }
    if (!d.probe) {
        d.probe=CreateBuffer(device,256,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        d.probeReadback=CreateBuffer(device,256,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    }
    struct alignas(8) P { unsigned char original[184]; uint64_t out; } p{};
    memcpy(p.original,native.pParams,184);memcpy(d.probeParams.data(),native.pParams,184);p.out=d.probe->GetGPUVirtualAddress();
    d.probeProjectionDiagnostic=native.paramSize==184;
    NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};launch.hFunction=k.function;launch.pParams=&p;launch.paramSize=sizeof p;
    launch.gridDim={1,1,1};launch.blockDim={1,1,1};LeaveWide(s,commands);UavBarrier(commands);
    NvCheck(s.launch(commands,&launch,1),"post probe launch");UavBarrier(commands);
    D3D12_RESOURCE_BARRIER transition{};transition.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    transition.Transition={d.probe.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
    commands->ResourceBarrier(1,&transition);commands->CopyBufferRegion(d.probeReadback.Get(),0,d.probe.Get(),0,32);
    std::swap(transition.Transition.StateBefore,transition.Transition.StateAfter);commands->ResourceBarrier(1,&transition);
    d.probeCaptured=true;
    d.probeSampleBytes=0;
    uint64_t history,motion;int32_t clamp,width,height;
    memcpy(&history,p.original+88,8);memcpy(&motion,p.original+96,8);memcpy(&clamp,p.original+52,4);
    memcpy(&width,p.original+172,4);memcpy(&height,p.original+176,4);
    if (!clamp || width<=0 || height<=0 || width>8192 || height>8192) return;
    auto& sample=d.levelKernels[0xff040001u];
    if (!sample.function) {
        auto cubin=ReadAsset(s.options.assets/"post_probe.cubin");
        NvCheck(s.createModule(device,cubin.data(),unsigned(cubin.size()),&sample.module),"post sample module");
        NvCheck(s.createFunction(device,sample.module,"nr_post_sample",&sample.function),"post sample function");
    }
    if (!d.probeProjection) {
        auto bytes=ReadAsset(s.options.assets/"post_probe_projection.bin");
        d.probeProjection=CreateBuffer(device,(bytes.size()+255)/256*256,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        void* dst=nullptr;Check(d.probeProjection->Map(0,nullptr,&dst),"post probe projection upload");memcpy(dst,bytes.data(),bytes.size());d.probeProjection->Unmap(0,nullptr);
    }
    const uint64_t count=uint64_t((width+3)/4)*((height+3)/4),bytes=count*64;
    if (d.probeSampleCapacity<bytes) {
        // Harness drains every frame before another capture; old buffers cannot still be in use.
        d.probeSamples=CreateBuffer(device,(bytes+255)/256*256,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        d.probeSamplesReadback=CreateBuffer(device,(bytes+255)/256*256,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        d.probeSampleCapacity=bytes;
    }
    struct alignas(8) Sample {unsigned char original[184];uint64_t out;uint32_t phase,pad;} sp{};
    memcpy(sp.original,p.original,184);sp.out=d.probeSamples->GetGPUVirtualAddress();
    auto capture=launch;capture.hFunction=sample.function;capture.pParams=&sp;capture.paramSize=sizeof sp;
    capture.gridDim={unsigned((count+255)/256),1,1};capture.blockDim={256,1,1};
    unsigned char rawParams[184];memcpy(rawParams,p.original,184);
    const float one=1.f;const uint64_t zero=0;const uint32_t off=0;
    memcpy(rawParams+48,&one,4);memcpy(rawParams+52,&off,4);memcpy(rawParams+56,&zero,8);
    auto raw=native;raw.pParams=rawParams;
    if(native.paramSize==312) {
        // The extended kernel writes raw temporal history at common surface +16,
        // then intensity-composed output at +192. Observe the actual raw surface;
        // do not reinterpret the extended parameter block or its projection.
        NvCheck(s.launch(commands,&native,1),"post intensity diagnostic replay");UavBarrier(commands);
        for(unsigned phase:{0u,2u}) {
            sp.phase=phase;NvCheck(s.launch(commands,&capture,1),"post intensity history sample");UavBarrier(commands);
        }
    } else for (unsigned phase=0;phase<3;++phase) {
        if (phase==1) {const uint64_t w=d.probeProjection->GetGPUVirtualAddress();memcpy(rawParams+24,&w,8);}
        NvCheck(s.launch(commands,phase==2?&native:&raw,1),"post diagnostic replay");UavBarrier(commands);
        sp.phase=phase;NvCheck(s.launch(commands,&capture,1),"post diagnostic sample");UavBarrier(commands);
    }
    transition.Transition={d.probeSamples.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
    commands->ResourceBarrier(1,&transition);commands->CopyBufferRegion(d.probeSamplesReadback.Get(),0,d.probeSamples.Get(),0,bytes);
    std::swap(transition.Transition.StateBefore,transition.Transition.StateAfter);commands->ResourceBarrier(1,&transition);
    d.probeSampleBytes=bytes;
    // The caller now records the actual POST launch again before any consumer.
}

NvAPI_Status __cdecl Launch(ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const Target* target = nullptr;
    if (kernels && count == 1) {
        const auto found = s.targets.find(kernels->hFunction);
        if (found != s.targets.end()) target = &found->second;
    }
    const auto pendingList = s.lists.find(commands);
    if (pendingList != s.lists.end() && pendingList->second.level.saved && (!target || !target->kind.swin)) {
        const auto status = FlushLevel(s, commands, &pendingList->second);
        if (status != NVAPI_OK) return status;
    }
    if (!target) { LeaveWide(s, commands); return s.launch(commands, kernels, count); }
    if (target->kind.clear) {               // a new evaluation
        auto& list = s.lists[commands];
        Learn(s, list);
        list = ListState{};
        list.device = target->device;
        { LeaveWide(s, commands); return s.launch(commands, kernels, count); }
    }
    const Kind& kind = target->kind;
    const auto known = s.lists.find(commands);
    ListState* list = known != s.lists.end() ? &known->second : nullptr;
    if(kind.probePost) {
        if(list && list->thinPreUsed) { Notice(s,"Thin4 requires the common 184-byte POST; extended intensity variant refused"); return NVAPI_ERROR; }
        if(s.probeEnabled)RecordPostProbe(s,commands,target->device,*kernels);
        LeaveWide(s,commands);
        return s.launch(commands,kernels,count);
    }
    if (kind.post || kind.pre) {
        if (kind.pre && list && kernels->pParams && kernels->paramSize == 272) {
            memcpy(list->thinPreParams.data(),kernels->pParams,272); list->thinPreCaptured = true;
        }
        if (kind.pre && s.probeEnabled && kernels->pParams && kernels->paramSize==272) {
            auto& d=s.devices[target->device];
            memcpy(d.probePreParams.data(),kernels->pParams,272);
            memcpy(&d.probePreHistory,static_cast<const unsigned char*>(kernels->pParams)+8,8);
            memcpy(&d.probePreMotion,static_cast<const unsigned char*>(kernels->pParams)+16,8);
        }
        if (kind.post && s.probeEnabled) {
            try { RecordPostProbe(s,commands,target->device,*kernels); }
            catch (const std::exception& error) { Notice(s,std::string("post probe failed: ")+error.what()); }
        }
        // Each end may have a Pre/PostPair4 record; otherwise either can use its native FP8 record.
        // The input block's parameter block grew by 8 bytes in 310.9.16 (264 -> 272), moving its weight pointer.
        const unsigned at = kind.post ? 24u : kernels->paramSize == 264 ? 224u : 232u, number = kind.post ? kPost : kPre;
        unsigned char edited[272];
        const bool readable = kernels->pParams && kernels->paramSize >= at + 8 && kernels->paramSize <= sizeof edited;
        try {
            uint64_t record = 0;
            if (readable) memcpy(&record, static_cast<const char*>(kernels->pParams) + at, 8);
            bool ours = false;                  // this launch is Model A's
            if (list) {
                (kind.post ? list->post : list->pre) = record;
                if (s.options.byOrder) ours = list->sequence && list->blocks == (kind.post ? kLayers : 0u);
            }
            if (!s.options.byOrder) {
                const auto d = s.devices.find(target->device);
                if (d != s.devices.end()) {
                    const auto found = d->second.known.find(record);
                    ours = found != d->second.known.end() && found->second == number;
                }
            }
            const unsigned chosen = s.manifest.loaded ? ChosenSet(s) : 0;
            if (s.enabled && !s.paused && readable && record && ours && chosen &&
                (kind.pre ? s.manifest.sets[chosen].pre.present : s.manifest.sets[chosen].post.present) && RuntimeMatches(s)) {
                const auto& set = s.manifest.sets[chosen]; const auto& r = kind.pre ? set.pre : set.post;
                const unsigned size = kind.pre ? 272 : 184;
                if (kernels->paramSize != size) throw std::runtime_error("unsupported Pre/PostPair4 parameter size");
                const auto* p = static_cast<const unsigned char*>(kernels->pParams);
                int32_t h, w, sx = 0, sy = 0; memcpy(&h, p + (kind.pre ? 248 : 32), 4); memcpy(&w, p + (kind.pre ? 252 : 36), 4);
                if (!kind.pre) { memcpy(&sx, p + 40, 4); memcpy(&sy, p + 44, 4); }
                if (h <= 0 || w <= 0 || h > 8192 || w > 8192 || h % 8 || w % 8 || (!kind.pre && (sx != -4 || sy != -4)) ||
                    kernels->gridDim.x != unsigned((w - sx + 7) / 8) || kernels->gridDim.y != unsigned((h - sy + 7) / 8) || kernels->gridDim.z != 1 ||
                    kernels->blockDim.x != 32 || kernels->blockDim.y != 1 || kernels->blockDim.z != 1 || kernels->dynSharedMemBytes)
                    throw std::runtime_error("unsupported Pre/PostPair4 shape");
                if (kind.pre) {
                    uint64_t depth, mask; int32_t slot;
                    memcpy(&depth, p + 24, 8); memcpy(&mask, p + 32, 8); memcpy(&slot, p + 196, 4);
                    if (depth || mask || slot != 15) throw std::runtime_error("PrePair4 requires null depth/control mask and feature slot 15");
                }
                auto& d = s.devices[target->device]; LoadWeights(s, d, target->device);
                const auto function = LevelKernelFor(s, d, target->device, kind.pre ? 16 : 0, r);
                const uint64_t bundle = d.weights.Address() + (kind.pre ? d.preBase[chosen] : d.postBase[chosen]) + r.offset;
                if (r.thin) {
                    if (!list || !list->thinPreCaptured) throw std::runtime_error("Thin4 POST has no matching PRE context");
                    if (kind.pre) LevelKernelFor(s,d,target->device,0,set.post); // preflight the consuming kernel before omitting skip writes
                    unsigned char params[464]{};
                    memcpy(params,list->thinPreParams.data(),272);
                    if (!kind.pre) { memcpy(params+272,p,184); OwnBlend(d, params+272); }
                    const unsigned sizeThin = kind.pre ? 280u : 464u;
                    memcpy(params+sizeThin-8,&bundle,8);
                    auto own=*kernels; own.hFunction=function; own.pParams=params; own.paramSize=sizeThin;
                    own.gridDim={unsigned((w/8+3)/4),unsigned(h/8),1}; own.blockDim={32,4,1};
                    LeaveWide(s,commands); UavBarrier(commands);
                    // Checker-only isolated comparison. The enclosing trace frame is the
                    // last measured frame; the ordinary chosen launch below always runs last.
                    // This uses the native process so other GPU work yields during timing.
                    char bench[8]{}; GetEnvironmentVariableA("NR_THIN_ABBA",bench,sizeof bench);
                    if (bench[0]=='1' && s.trace.active && s.trace.commands==commands) {
                        auto& trace=s.trace;
                        for(unsigned pair=0;pair<16;++pair)for(unsigned part=0;part<2;++part) {
                            const bool custom=((pair&1u)^part)!=0;
                            if(trace.rows.size()>=500)throw std::runtime_error("Thin4 benchmark trace capacity");
                            State::LaunchTrace::Row row;row.name=std::string("ThinAB ")+(kind.pre?"PRE ":"POST ")+(custom?"Thin4":"FP8");row.plan=-2;
                            const auto& launch=custom?own:*kernels;
                            row.grid[0]=launch.gridDim.x;row.grid[1]=launch.gridDim.y;row.grid[2]=1;
                            row.block[0]=launch.blockDim.x;row.block[1]=launch.blockDim.y;row.block[2]=1;
                            const unsigned q=unsigned(trace.rows.size());trace.rows.push_back(std::move(row));
                            commands->EndQuery(trace.heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1+2*q);
                            NvCheck(s.launch(commands,&launch,1),"Thin4 benchmark launch");UavBarrier(commands);
                            commands->EndQuery(trace.heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,2+2*q);
                        }
                    }
                    const auto status=s.launch(commands,&own,1); UavBarrier(commands);
                    if (status==NVAPI_OK) { ++(kind.pre?s.preReplaced:s.postReplaced); if(kind.pre)list->thinPreUsed=true; }
                    return status;
                }
                const auto& blob = kind.pre ? set.preBlob : set.postBlob;
                uint32_t io; memcpy(&io, blob.data() + r.offset + 20, 4);
                const uint64_t ioAddress = bundle + io;
                unsigned char params[296]{}; memcpy(params, p, size); memcpy(params + size, &ioAddress, 8); memcpy(params + size + 16, &bundle, 8);
                if (!kind.pre) OwnBlend(d, params);
                auto own = *kernels; own.hFunction = function; own.pParams = params; own.paramSize = size + 24; own.blockDim = {32, 2, 1};
                LeaveWide(s, commands); UavBarrier(commands);
                const auto status = s.launch(commands, &own, 1);
                UavBarrier(commands); if (status == NVAPI_OK) ++(kind.pre ? s.preReplaced : s.postReplaced);
                return status;
            }
            if (s.enabled && !s.paused && readable && record && ours && chosen && s.manifest.sets[chosen].records[number].present && RuntimeMatches(s)) {
                auto& d = s.devices[target->device];
                LoadWeights(s, d, target->device);
                memcpy(edited, kernels->pParams, kernels->paramSize);
                const uint64_t w = d.weights.Address() + d.fp8Base[chosen] + s.manifest.sets[chosen].records[number].offset;
                memcpy(edited + at, &w, 8);
                if (kind.post) OwnBlend(d, edited);
                NVAPI_CU_KERNEL_LAUNCH_PARAMS own = *kernels;
                own.pParams = edited;
                ++s.rerecorded;
                LeaveWide(s, commands);
                return s.launch(commands, &own, 1);
            }
        } catch (const std::exception& error) {
            Notice(s, std::string("the input / output block kept the runtime's weights: ") + error.what());
            if(list && list->thinPreUsed) return NVAPI_ERROR;
        } catch (...) {
        }
        if(list && list->thinPreUsed && kind.post) { Notice(s,"Thin4 POST missing after Thin4 PRE; evaluation refused"); return NVAPI_ERROR; }
        if (s.enabled && !s.paused) ++s.ioPassed;       // the runtime's own weights run this input / output block (Faults())
        LeaveWide(s, commands);
        return s.launch(commands, kernels, count);
    }
    int number = -1;                        // the block of Model A this launch is, when we can tell
    // The launch as it is recorded: the runtime's, or a copy with other pointers in it (a block after left-out blocks,
    // a weight record of the set). Whatever goes wrong after a block was left out, it is this copy that is launched.
    const Fields f = FieldsOf(kind.heads);
    const bool plain = kernels->pParams && kernels->paramSize == f.size;
    unsigned char edited[96];
    NVAPI_CU_KERNEL_LAUNCH_PARAMS own{};
    const NVAPI_CU_KERNEL_LAUNCH_PARAMS* k = kernels;
    const auto runtimes = [&]() {           // the runtime's own kernel for this launch
        LeaveWide(s, commands);
        const auto result = s.launch(commands, k, count);
        Track(list, *k, kind, f, plain);
        return result;
    };
    try {
        uint64_t record = 0;
        if (kernels->pParams && kernels->paramSize >= 24) memcpy(&record, static_cast<const char*>(kernels->pParams) + 16, 8);
        if (list) {
            const unsigned n = list->blocks++;
            if (n < kLayers && s.manifest.loaded) {
                const Layer& expected = s.manifest.layers[n];
                list->sequence &= expected.heads == kind.heads && expected.variant == unsigned(kind.variant) && record != 0;
                list->records[n] = record;
                if (s.options.byOrder && list->sequence) number = int(n);
            } else {
                list->sequence = false;
            }
        }
        if (!s.enabled || s.paused) {
            const auto status = FlushLevel(s, commands, list);
            if (status != NVAPI_OK) return status;
            if (list) list->carry = {};
            LeaveWide(s, commands);
            return s.launch(commands, kernels, count);
        }
        if (!s.options.byOrder) {
            const auto d = s.devices.find(target->device);
            if (d != s.devices.end()) {
                const auto found = d->second.known.find(record);
                if (found != d->second.known.end() && found->second < kLayers) number = int(found->second);
            }
        }
        const unsigned chosen = ChosenSet(s);
        const Manifest::Set* set = chosen && number >= 0 && list && plain && RuntimeMatches(s) ? &s.manifest.sets[chosen] : nullptr;
        NvAPI_Status levelStatus = NVAPI_OK;
        if (LevelLaunch(s, commands, target->device, list, *kernels, kind, number, plain, set, chosen, levelStatus)) return levelStatus;
        // A block the set leaves out: nothing is launched. Only inside a level, after a launched block of the level.
        if (set && ((set->skip >> number) & 1) && kind.variant == Plain && kind.waits && kind.publishes && list->carry.out) {
            list->carry.dirty = true;
            ++s.skipped;
            if (!s.active) {
                s.active = true;
                try { Report(s, "active"); } catch (...) { }
            }
            return NVAPI_OK;
        }
        // The block after left-out ones reads the last launched block's output and waits for that block's flags.
        if (list && list->carry.dirty) {
            if (kind.waits && plain) {
                Rewire(*kernels, f, list->carry, edited);
                own = *kernels; own.pParams = edited; k = &own;
            } else {
                list->carry = {};
            }
        }
        if (number < 0 || !(s.options.widths & (1u << Log2(kind.heads))) || !((s.options.layers >> number) & 1)) {
            // The runtime's kernel; with the set's record for it where there is one.
            if (set && set->records[number].present) {
                auto& d = s.devices[target->device];
                LoadWeights(s, d, target->device);
                if (k == kernels) { memcpy(edited, kernels->pParams, f.size); own = *kernels; own.pParams = edited; k = &own; }
                const uint64_t w = d.weights.Address() + d.fp8Base[chosen] + set->records[number].offset;
                memcpy(edited + 16, &w, 8);
                ++s.rerecorded;
            } else {
                ++s.passed;
            }
            return runtimes();
        }
        if (!RuntimeMatches(s)) {
            Notice(s, "off: no NR runtime is loaded");
            ++s.passed;
            return runtimes();
        }
        Replace(s, commands, target->device, *k, kind, unsigned(number));
        Track(list, *k, kind, f, plain);
        ++s.replaced;
        ++s.perWidth[Log2(kind.heads)];
        if (!s.active) {
            s.active = true;
            try { Report(s, "active"); } catch (...) { }
        }
        return NVAPI_OK;
    } catch (const std::exception& error) {
        // Nothing of ours was recorded for this launch: the runtime's own kernel runs.
        Notice(s, std::string("a block was left to the runtime: ") + error.what());
        const auto status = FlushLevel(s, commands, list);
        if (status != NVAPI_OK) return status;
        ++s.passed;
        return runtimes();
    } catch (...) {
        Notice(s, "a block was left to the runtime: unknown error");
        const auto status = FlushLevel(s, commands, list);
        if (status != NVAPI_OK) return status;
        ++s.passed;
        return runtimes();
    }
}

NvAPI_Status __cdecl TraceLaunch(ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    auto& s=S(); std::lock_guard guard(s.mutex); auto& t=s.trace;
    if (!t.active || commands!=t.commands) return Launch(commands,kernels,count);
    if (!kernels || count!=1 || t.rows.size()>=500) throw std::runtime_error("Checker launch trace capacity/chain mismatch");
    State::LaunchTrace::Row row;
    const auto name=t.names.find(kernels->hFunction);row.name=name==t.names.end()?"unmapped":name->second;
    row.grid[0]=kernels->gridDim.x;row.grid[1]=kernels->gridDim.y;row.grid[2]=kernels->gridDim.z;
    row.block[0]=kernels->blockDim.x;row.block[1]=kernels->blockDim.y;row.block[2]=kernels->blockDim.z;
    const auto target=s.targets.find(kernels->hFunction);
    if (target!=s.targets.end()) {
        if (target->second.kind.pre) row.plan=0;
        else if (target->second.kind.post || target->second.kind.probePost) row.plan=151;
        else if (target->second.kind.swin && kernels->pParams && kernels->paramSize>=24) {
            uint64_t weights=0;memcpy(&weights,static_cast<const unsigned char*>(kernels->pParams)+16,8);
            const auto& known=s.devices[target->second.device].known;const auto found=known.find(weights);
            if(found!=known.end() && found->second<kLayers) row.plan=int(s.manifest.layers[found->second].index);
        }
    }
    const unsigned index=unsigned(t.rows.size());t.rows.push_back(std::move(row));
    commands->EndQuery(t.heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1+2*index);
    const auto result=Launch(commands,kernels,count);
    commands->EndQuery(t.heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,2+2*index);
    return result;
}

NvAPI_Status __cdecl DestroyFunction(ID3D12Device* device, NVDX_ObjectHandle function) {
    auto& s = S();
    const auto result = s.destroyFunction(device, function);
    if (result == NVAPI_OK) {
        std::lock_guard guard(s.mutex);
        const auto found = s.targets.find(function);
        if (found != s.targets.end()) {
            if (found->second.kind.swin) {      // a feature is going away, and its weight records with it
                const auto d = s.devices.find(device);
                if (d != s.devices.end()) d->second.known.clear();
            }
            if (--found->second.refs == 0) s.targets.erase(found);
        }
    }
    return result;
}

NvAPI_Status __cdecl DestroyModule(ID3D12Device* device, NVDX_ObjectHandle module) {
    return S().destroyModule(device, module);
}
}   // namespace

void* WrapNvapi(unsigned id, void* original) {
    if (!original) return original;
    auto& s = S();
    std::lock_guard guard(s.mutex);
    switch (id) {
    case kCreateModule: s.createModule = static_cast<decltype(s.createModule)>(original); return static_cast<void*>(&CreateModule);
    case kCreateFunction: s.createFunction = static_cast<decltype(s.createFunction)>(original); return static_cast<void*>(&CreateFunction);
    case kLaunch: s.launch = static_cast<decltype(s.launch)>(original); return s.trace.enabled ? static_cast<void*>(&TraceLaunch) : static_cast<void*>(&Launch);
    case kDestroyModule: s.destroyModule = static_cast<decltype(s.destroyModule)>(original); return static_cast<void*>(&DestroyModule);
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
    if (options.assets != s.options.assets) {
        // Another asset folder: read it afresh and forget earlier load failures. Weights already on a device stay in
        // use there until ReleaseDevice(), since recorded command lists may refer to them.
        if (s.manifest.loaded) for (auto& [device, d] : s.devices) if (d.weightsReady) throw std::runtime_error("NVFP4 fused-block assets cannot change while a device holds them");
        s.manifest = Manifest{};
        for (auto& [device, d] : s.devices) {
            d.failure.clear();
            for (auto* set : {&d.kernels, &d.wideKernels, &d.smallKernels, &d.partialKernels}) for (auto& width : *set) for (auto& k : width) if (!k.function) k.failure.clear();
            for (auto& [key, k] : d.levelKernels) if (!k.function) k.failure.clear();
        }
    }
    s.options = options;
    if (const char* text = getenv("NR_SWIN4_WIDE_BARRIER")) s.wideBarrier = unsigned(atoi(text)) & 3;
    s.options.widths &= 15;
    s.options.twoTerms &= 15;
    LoadManifest(s);                    // throws: nothing is enabled without readable assets
    // A set that leaves blocks out or brings records for the runtime's kernels is in use at any width selection.
    // (UseSet() may choose any of them later, so one such set in the folder is enough.)
    s.enabled = s.options.widths != 0;
    for (unsigned n = 2; n < kMaxSets; ++n) s.enabled |= s.manifest.sets[n].present && s.manifest.sets[n].Reshapes();
    s.active = false;
    s.notice.clear();
    s.waiting = s.enabled ? (options.byOrder ? "on, waiting for the model's next evaluation" : "on, waiting to recognise Model A's blocks") : "off";
}

void Pause(bool paused) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.paused = paused;
}

unsigned UseSet(unsigned set) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const unsigned overrideSet = s.evaluationSetAt < s.evaluationSets.size() ? s.evaluationSets[s.evaluationSetAt++] : kDefaultSet;
    s.passSet = overrideSet == kDefaultSet ? set : overrideSet;
    return s.passSet;
}

bool SetComplete(unsigned set, std::string* missing) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const unsigned n = set == kDefaultSet ? s.options.set : set;
    bool complete = s.manifest.loaded && n < kMaxSets;
    if (complete && n) {
        const auto& chosen = s.manifest.sets[n];
        complete = chosen.present && !chosen.fp8.empty() && (!chosen.blob.empty() || (n == 1 && s.manifest.trainedIsFirst));
    }
    if (!complete && missing)
        *missing = !s.manifest.loaded ? "swin4/swin4.txt" : n == 1 ? "swin4/swin4_trained.*" : "swin4/swin4_set" + std::to_string(n) + ".*";
    return complete;
}

unsigned long long Faults() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.passed + s.ioPassed;
}

void UseSetsForNextEvaluations(const unsigned* sets, unsigned count) {
    auto& s = S(); std::lock_guard guard(s.mutex);
    if (count && !sets) throw std::runtime_error("null evaluation-set list");
    for (unsigned i = 0; i < count; ++i) if (sets[i] != kDefaultSet && sets[i] >= kMaxSets) throw std::runtime_error("invalid evaluation set");
    s.evaluationSets.clear(); if (count) s.evaluationSets.assign(sets, sets + count); s.evaluationSetAt = 0;
}

void EnablePostProbe(bool enabled) {
    auto& s=S();std::lock_guard guard(s.mutex);s.probeEnabled=enabled;
    if (enabled) for (auto& [device,d] : s.devices) d.probeCaptured=false;
}

void EnableLaunchTrace() { auto& s=S();std::lock_guard guard(s.mutex);s.trace.enabled=true; }
void BeginLaunchTrace(ID3D12GraphicsCommandList* commands, ID3D12CommandQueue* queue) {
    auto& s=S();std::lock_guard guard(s.mutex);auto& t=s.trace;
    if(!t.enabled || t.active || t.pending) throw std::runtime_error("Invalid checker launch trace begin");
    if(!t.heap) {
        ComPtr<ID3D12Device> device;Check(commands->GetDevice(IID_PPV_ARGS(&device)),"trace device");
        D3D12_QUERY_HEAP_DESC desc{D3D12_QUERY_HEAP_TYPE_TIMESTAMP,1024,0};
        Check(device->CreateQueryHeap(&desc,IID_PPV_ARGS(&t.heap)),"trace heap");
        t.readback=CreateBuffer(device.Get(),1024*8,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    }
    Check(queue->GetTimestampFrequency(&t.frequency),"trace frequency");
    t.commands=commands;t.rows.clear();t.active=true;commands->EndQuery(t.heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
}
void EndLaunchTrace(ID3D12GraphicsCommandList* commands) {
    auto& s=S();std::lock_guard guard(s.mutex);auto& t=s.trace;
    if(!t.active || t.commands!=commands) throw std::runtime_error("Invalid checker launch trace end");
    const unsigned count=unsigned(t.rows.size())*2+2;
    commands->EndQuery(t.heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,count-1);
    commands->ResolveQueryData(t.heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,count,t.readback.Get(),0);
    t.active=false;t.pending=true;
}
void WriteLaunchTrace(const std::filesystem::path& path) {
    auto& s=S();std::lock_guard guard(s.mutex);auto& t=s.trace;
    if(!t.pending || t.active) throw std::runtime_error("No completed checker launch trace");
    void* mapped=nullptr;D3D12_RANGE range{0,(2*t.rows.size()+2)*8};
    Check(t.readback->Map(0,&range,&mapped),"trace readback");
    const auto* ticks=static_cast<const uint64_t*>(mapped);std::vector<uint64_t> stamp(ticks,ticks+2*t.rows.size()+2);
    D3D12_RANGE empty{0,0};t.readback->Unmap(0,&empty);t.pending=false;
    std::ofstream out(path,std::ios::trunc);out.precision(12);
    out << "{\"source\":\"NrSwin4 hook around each forwarded CUDA call\",\"timing_warning\":\"per-launch timestamps perturb overlap; suppressed fused calls may contain only timestamps; fused-level work is charged to its last original call\",\"gpu_ms\":"
        << double(stamp.back()-stamp.front())*1000/t.frequency << ",\"launches\":[\n";
    for(size_t i=0;i<t.rows.size();++i) {
        const auto& r=t.rows[i];out << "{\"index\":" << i << ",\"name\":\"" << r.name << "\",\"plan\":" << r.plan
            << ",\"grid\":[" << r.grid[0] << ',' << r.grid[1] << ',' << r.grid[2] << "],\"block\":[" << r.block[0] << ',' << r.block[1] << ',' << r.block[2]
            << "],\"gpu_ms\":" << double(stamp[2*i+2]-stamp[2*i+1])*1000/t.frequency << ",\"start_ms\":" << double(stamp[2*i+1]-stamp[0])*1000/t.frequency << '}';
        out << (i+1<t.rows.size()?",\n":"\n");
    }
    out << "]}\n";if(!out)throw std::runtime_error("Cannot write checker launch trace");
}

void WritePostProbe(ID3D12Device* device, const std::filesystem::path& path) {
    auto& s=S();std::lock_guard guard(s.mutex);const auto found=s.devices.find(device);
    if (found==s.devices.end() || !found->second.probeCaptured) throw std::runtime_error("No native POST diagnostic was captured");
    auto& d=found->second;void* mapped=nullptr;D3D12_RANGE range{0,32};float v[8];
    Check(d.probeReadback->Map(0,&range,&mapped),"post probe readback");memcpy(v,mapped,sizeof v);
    D3D12_RANGE empty{0,0};d.probeReadback->Unmap(0,&empty);
    std::ofstream out(path,std::ios::trunc);out.precision(9);out << "{\n  \"blend_weight\": " << v[0] << ", \"history\": " << v[1]
        << ", \"motion\": " << v[2] << ", \"final_clamp\": " << v[3] << ", \"raw_weight\": \"" << v[4]
        << "\", \"motion_present\": " << v[5] << ", \"weight_pointer\": " << v[6] << ", \"active\": " << v[7]
        << ", \"pre_history\": " << (d.probePreHistory!=0) << ", \"pre_motion\": " << (d.probePreMotion!=0)
        << ", \"sample_count\": " << d.probeSampleBytes/64 << ", \"sample_stride\": 4,\n  \"params_hex\": \"";
    constexpr char hex[]="0123456789abcdef";for (unsigned char b:d.probeParams) out << hex[b>>4] << hex[b&15];
    uint64_t postHistory=0,postMotion=0;int seed=0;
    memcpy(&postHistory,d.probeParams.data()+88,8);memcpy(&postMotion,d.probeParams.data()+96,8);memcpy(&seed,d.probePreParams.data()+208,4);
    out << "\", \"projection_diagnostic\": " << (d.probeProjectionDiagnostic?"true":"false")
        << ", \"pre_seed\": " << seed << ", \"same_history_handle\": " << (d.probePreHistory==postHistory?"true":"false")
        << ", \"same_motion_handle\": " << (d.probePreMotion==postMotion?"true":"false") << ", \"pre_params_hex\": \"";
    for(unsigned char b:d.probePreParams)out << hex[b>>4] << hex[b&15];
    out << "\"\n}\n";if (!out) throw std::runtime_error("Cannot write POST diagnostic");
    if (d.probeSampleBytes) {
        D3D12_RANGE samples{0,SIZE_T(d.probeSampleBytes)};Check(d.probeSamplesReadback->Map(0,&samples,&mapped),"post samples readback");
        auto samplePath=path;samplePath.replace_extension(L".samples.f32");std::ofstream data(samplePath,std::ios::binary|std::ios::trunc);
        data.write(static_cast<const char*>(mapped),std::streamsize(d.probeSampleBytes));d.probeSamplesReadback->Unmap(0,&empty);
        if (!data) throw std::runtime_error("Cannot write POST samples");
    }
}

void Disable() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.enabled = false;
    s.active = false;
    s.waiting = "off";
    s.notice.clear();
}

void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept {
    try {
        auto& s = S();
        std::lock_guard guard(s.mutex);
        const auto found = s.lists.find(commands);
        if (found == s.lists.end()) return;
        if (FlushLevel(s, commands, &found->second) != NVAPI_OK) Notice(s, "Level4 incomplete prefix replay failed");
        Learn(s, found->second);
        s.lists.erase(found);
        s.wideOpen.erase(commands);
    } catch (...) {
    }
}

void ReleaseDevice(ID3D12Device* device) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const auto found = s.devices.find(device);
    if (found != s.devices.end()) {
        // On a lost device these calls fail, which is fine: the objects die with it.
        for (auto* set : {&found->second.kernels, &found->second.wideKernels, &found->second.smallKernels, &found->second.partialKernels})
            for (auto& width : *set)
                for (auto& k : width) {
                    if (s.destroyFunction && k.function) s.destroyFunction(device, k.function);
                    if (s.destroyModule && k.module) s.destroyModule(device, k.module);
                }
        for (auto& [key, k] : found->second.levelKernels) {
            if (s.destroyFunction && k.function) s.destroyFunction(device, k.function);
            if (s.destroyModule && k.module) s.destroyModule(device, k.module);
        }
        s.devices.erase(found);
    }
    for (auto i = s.targets.begin(); i != s.targets.end();) i = i->second.device == device ? s.targets.erase(i) : std::next(i);
    s.lists.clear();                        // command lists of the device go with it; their addresses may be reused
    s.wideOpen.clear();
}

std::string Status(bool counters) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    std::string status = !s.enabled ? "off" : s.active ? "active" : s.waiting;
    if (s.enabled && !s.notice.empty()) status += s.active ? " (" + s.notice + ")" : ": " + s.notice;
    if (s.enabled) {
        status += ", widths";
        for (unsigned bit = 4; bit-- > 0;) if (s.options.widths & (1u << bit)) status += " " + std::to_string(32u << bit);
        if (s.options.set || ChosenSet(s)) {
            const unsigned chosen = ChosenSet(s);
            const std::string name = chosen == 1 ? "trained weights" : "weight set " + std::to_string(chosen ? chosen : s.options.set);
            status += chosen ? ", " + name : ", " + name + " not found (the first set is in use)";
            if (chosen && s.manifest.sets[chosen].skip) {
                unsigned count = 0;
                for (unsigned n = 0; n < kLayers; ++n) count += unsigned((s.manifest.sets[chosen].skip >> n) & 1);
                status += " (" + std::to_string(count) + " blocks left out)";
            }
        }
    }
    if (!counters) return status;
    return status + " | blocks replaced: " + std::to_string(s.replaced) + " (width 32: " + std::to_string(s.perWidth[0]) + ", 64: " +
           std::to_string(s.perWidth[1]) + ", 128: " + std::to_string(s.perWidth[2]) + ", 256: " + std::to_string(s.perWidth[3]) +
           ") | left to the runtime: " + std::to_string(s.passed) + " | left out: " + std::to_string(s.skipped) +
           " | runtime kernels with a record of the set: " + std::to_string(s.rerecorded) + " | small-map blocks: " + std::to_string(s.smallReplaced) +
           " | partial C256 using selected NVFP4 records: " + std::to_string(s.partialReplaced) +
           " | fused levels D32/D64: " + std::to_string(s.levelReplaced[0]) + "/" + std::to_string(s.levelReplaced[1]) +
           " | fused levels E32/E64: " + std::to_string(s.encoderReplaced[0]) + "/" + std::to_string(s.encoderReplaced[1]) +
           " | fused levels E128/D128: " + std::to_string(s.encoderReplaced[2]) + "/" + std::to_string(s.levelReplaced[2]) +
           " | fused levels E256/D256: " + std::to_string(s.encoderReplaced[3]) + "/" + std::to_string(s.levelReplaced[3]) +
           " | PostPair4: " + std::to_string(s.postReplaced) + " | PrePair4: " + std::to_string(s.preReplaced);
}
}   // namespace NrSwin4
