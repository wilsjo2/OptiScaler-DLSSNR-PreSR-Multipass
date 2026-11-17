// NVFP4 replacement of the rest of Model A's pass; see NrNet4.h.
//
// Nothing here knows a kernel's interface. A family's manifest lists, in launch order, the runtime's launches that a
// kernel of ours can stand in for (a "slot"): the runtime kernel's name, block shape and parameter size, where in its
// parameter block the weight-record pointer is, and how our parameter block is made: byte ranges copied from the
// runtime's block, 32-bit constants, the address of our record and an options word. Ours is launched with the
// runtime's grid and block. The runtime's neighbours wait for a kernel's flags without a time limit, so a kernel of
// ours must publish exactly what the runtime's does; that is proven per family before its assets exist.
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

#include "NrNet4.h"
#include "NrFfnV2.h"

#pragma comment(lib, "bcrypt.lib")

namespace NrNet4 {
namespace {
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

constexpr unsigned kCreateModule = 0xad1a677d, kCreateFunction = 0xe2436e22, kLaunch = 0x24973538,
                   kDestroyModule = 0x41c65285, kDestroyFunction = 0xdf295ea6;
constexpr unsigned kMaxParams = 512;        // a parameter block, the runtime's or ours
constexpr unsigned kMaxPartners = 8;
// The output block's history blend weight, float16 0.7397, the value the 4-bit models expect. The runtime
// passes a pointer to its own copy at parameter +104 of the output block; our output-block launches use this copy.
constexpr uint16_t kBlendWeight = 0x39eb;
constexpr unsigned kBlendParam = 104;
constexpr size_t kFeatures = 16;            // evaluations' worth of weight-record addresses remembered per device
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull, kFnvPrime = 0x100000001b3ull;
constexpr const char* kFamilyName[kFamilies] = {"split", "vit", "prepost"};
constexpr const char* kGroupName[kFamilies][kMaxGroups] = {
    {"ffwd", "ffwd_proj", "qkv", "proj", "head", "dec"},
    {"qkv", "attention", "projection"},
    {"pre", "post"}};

// The runtime's kernels of the bottleneck's residual blocks, by what they are to a block (see Depth()).
enum class Role : uint8_t { None, SplitFfwd, SplitFfwdProj, SplitQkv, SplitProj, VitExpandFirst, VitExpand, VitContract, VitQkv, VitAttention, VitProjection };
Role RoleOf(const std::string& name) {
    static const std::pair<const char*, Role> known[] = {
        {"cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8", Role::SplitFfwd}, {"cc_split_swin_16h_ffwd_512_chained_fp8", Role::SplitFfwd},
        {"cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8", Role::SplitFfwdProj}, {"cc_split_swin_16h_ffwd_proj_512_chained_fp8", Role::SplitFfwdProj},
        {"cc_split_swin_16h_qkv_512_chained_fp8", Role::SplitQkv},
        {"cc_split_swin_16h_proj_512_chained_fp8", Role::SplitProj}, {"cc_split_swin_16h_proj_pool_512_chained_fp8", Role::SplitProj},
        {"cc_split_swin_16h_proj_512_outview_wait_fp8", Role::SplitProj},
        {"cc_vit_1d_ffn_expand_publish_fp8", Role::VitExpandFirst}, {"cc_vit_1d_ffn_expand_chained_fp8", Role::VitExpand},
        {"cc_vit_1d_ffn_contract_chained_fp8", Role::VitContract}, {"cc_vit_1d_qkv_chained_fp8", Role::VitQkv},
        {"cc_vit_1d_attention_chained_fp8", Role::VitAttention},
        {"cc_vit_1d_projection_chained_fp8", Role::VitProjection}, {"cc_vit_1d_projection_wait_fp8", Role::VitProjection}};
    for (const auto& [kernel, role] : known) if (name == kernel) return role;
    return Role::None;
}

void Check(HRESULT result, const char* what) {
    if (FAILED(result)) throw std::runtime_error(std::string(what) + " failed: D3D12 status " + std::to_string(unsigned(result)));
}
void NvCheck(NvAPI_Status result, const std::string& what) {
    if (result != NVAPI_OK) throw std::runtime_error(what + " failed: NVAPI status " + std::to_string(result));
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


uint64_t Fnv(uint64_t hash, const std::string& name) {
    for (const unsigned char byte : name) hash = (hash ^ byte) * kFnvPrime;
    return (hash ^ uint64_t('\n')) * kFnvPrime;
}

int GroupOf(unsigned family, const std::string& name) {
    for (unsigned g = 0; g < kMaxGroups; ++g)
        if (kGroupName[family][g] && name == kGroupName[family][g]) return int(g);
    return -1;
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

struct Slot {
    std::string name;                           // the runtime's kernel
    unsigned group = 0, kernel = 0, block[3]{}, launchBlock[3]{}, params = 0, record = 0, size = 0, w = 0, options = 0;
    uint64_t offset = 0, bytes = 0;
    struct Copy { unsigned dst, src, bytes; };
    std::vector<Copy> map;
    std::vector<std::pair<unsigned, uint32_t>> set;         // offset in our block, value
    struct Partner { unsigned slot; uint32_t bits, request; };
    std::vector<Partner> partners;              // bits added when that slot is ours too and both options words hold `request`
    std::vector<std::pair<unsigned, uint32_t>> require;     // offset in the runtime's block, the value it must have
};
struct Manifest {
    bool loaded = false;
    std::string failure, runtimeSha;
    unsigned evaluationLaunches = 0;
    uint64_t evaluationHash = 0;
    std::vector<std::pair<std::string, std::string>> kernels;   // cubin file, entry point
    struct SmallKernel { unsigned kernel=0, dims=0, maxTokens=0, block[3]{}; };
    std::map<unsigned, SmallKernel> smallKernels; // same ABI/grid/block; gate on dimensions in our parameter block
    std::array<uint32_t, kMaxGroups> defaults{};
    std::vector<Slot> slots;
    std::map<std::string, std::pair<unsigned, unsigned>> names; // runtime kernel -> its parameter size and record offset
    std::vector<unsigned char> blob;
    std::vector<unsigned char> trained;         // <family>4_trained.bin: the same records with trained codes; empty without the file
    bool trainedIsFirst = false;                // <family>4.bin is not there: `blob` holds <family>4_trained.bin
};
// Kernels and weights of one device. They stay until ReleaseDevice(), because recorded command lists refer to them.
struct Device {
    // trained: where those records start; 0: none. setAt: the same for a numbered set's records.
    struct Weights { bool ready = false; std::string failure; Buffer buffer; uint64_t trained = 0, blendAt = 0; std::array<uint64_t, kMaxSets> setAt{}; };
    struct Module { NVDX_ObjectHandle module = nullptr; std::string failure; };
    struct Kernel { NVDX_ObjectHandle function = nullptr; std::string failure; };
    std::array<Weights, kFamilies> weights;
    std::array<std::map<std::string, Module>, kFamilies> modules;       // by cubin file
    std::array<std::vector<Kernel>, kFamilies> kernels;                 // by the manifest's kernel number
    std::vector<ComPtr<IUnknown>> parked;       // what an upload that did not finish may still be using; freed with the device
    std::map<uint64_t, std::pair<unsigned, unsigned>> known;            // address of a runtime weight record -> family, slot
    NVDX_ObjectHandle poolModule = nullptr, poolKernel = nullptr, unpoolKernel = nullptr;
    NVDX_ObjectHandle poolCapture = nullptr;
    Buffer poolProbeFull, poolProbe, poolReadback;
    unsigned poolProbeH=0, poolProbeW=0;
    // Stable per-runtime-buffer and pooled-count allocations, retained for recorded command lists.
    std::map<std::pair<uint64_t, unsigned>, Buffer> poolBuffers;
    struct SideNative { Buffer weights; ComPtr<ID3D12Resource> upload; NVDX_ObjectHandle module=nullptr,function=nullptr; };
    std::map<std::pair<unsigned,unsigned>,SideNative> sides;
};
// What one command list is in the middle of. An evaluation of the model starts with cc_cb_clear.
struct FamilyRun {
    unsigned launches = 0;                      // launches of the family's kernels since the evaluation's cc_cb_clear
    bool sequence = true;                       // ... all of them the manifest's, in its order, so far
    std::array<uint64_t, kMaxSlots> records{};  // their weight-record addresses
    std::array<uint8_t, kMaxSlots> state{};     // per slot: 0 not launched yet, 1 left to the runtime, 2 replaced
    std::array<uint8_t, kMaxSlots> promised{};  // per replaced slot: bit i = it counts on its i-th partner being replaced too
};
// Where an evaluation is among the bottleneck's residual blocks, for the blocks a set leaves out.
struct DepthRun {
    bool recognised = false;                    // the last launch that has a slot was recognised as Model A's
    unsigned splitBlocks = 0, vitBlocks = 0;    // blocks begun so far (a block begins with its ffwd / its expand)
    bool dropSplit = false, dropVit = false;    // the block being launched is left out
    bool rewire = false, vitRewire = false;     // the next launched block follows blocks that were left out
    uint64_t out = 0, outFlags = 0;             // the last launched split block's result and the flags of its last kernel
    uint64_t vitFlags = 0;                      // the flags of the last launched transformer block's projection
};
struct ListState {
    ID3D12Device* device = nullptr;
    unsigned launches = 0;                      // kernel launches of the evaluation, cc_cb_clear included
    uint64_t hash = kFnvOffset;                 // ... of their names
    std::array<FamilyRun, kFamilies> family;
    DepthRun depth;
    struct PoolRun { bool active = false; uint64_t full = 0, base = 0; unsigned h = 0, w = 0, tokens = 0; } pool;
    struct SideRun { bool active=false;unsigned side=0,set=0,first=0,seen=0;uint64_t input=0;int h=0,w=0; } side;
};
// A numbered set (NrNet4.h).
struct SetInfo {
    bool present = false;
    uint32_t skipSplit = 0, skipVit = 0;
    unsigned poolVit = 0, poolMin = 0;
    struct Side { unsigned stages=0,hidden=0;std::vector<unsigned char> blob; };
    std::array<Side,2> sides;
    std::array<std::vector<unsigned char>, kFamilies> blob;
};
struct Target {
    ID3D12Device* device = nullptr;
    NVDX_ObjectHandle module = nullptr;
    std::string name;
    unsigned refs = 0;
    bool clear = false;
    bool outside = false;                       // not a launch of the network: left out of an evaluation's count and hash
    Role role = Role::None;
    unsigned generation = 0;                    // of the manifests `family` was looked up in
    int family = -1;
    unsigned params = 0, record = 0;
};

struct State {
    std::recursive_mutex mutex;
    bool enabled = false;
    bool paused = false;                        // Pause(): this evaluation is the runtime's
    Options options;
    std::array<Manifest, kFamilies> manifest;
    unsigned generation = 1;                    // changes when a manifest is loaded or dropped
    std::string notice;                         // why launches were last left alone
    std::array<SetInfo, kMaxSets> sets;
    unsigned passSet = kDefaultSet;             // UseSet()
    uint64_t leftOut = 0;                       // launches of left-out blocks
    std::array<uint64_t,2> sideRuns{};
    std::array<bool, kFamilies> active{};       // a launch of the family was replaced since the last Enable()
    std::array<uint64_t, kFamilies> replaced{}, passed{};
    uint64_t substituted = 0;                   // launches of ours that ran other records than the set asked for (Faults())
    std::array<std::array<uint64_t, kMaxGroups>, kFamilies> perGroup{};
    uint64_t learnt = 0;
    std::wstring runtimePath;
    std::string runtimeSha;
    void (*log)(const char*) = nullptr;
    std::map<NVDX_ObjectHandle, Target> targets;
    std::map<ID3D12Device*, Device> devices;
    std::map<ID3D12GraphicsCommandList*, ListState> lists;
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
};
State& S() { static State* state = new State; return *state; }

void Report(State& s, const std::string& text) {
    const std::string line = "NVFP4 network: " + text;
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

uint64_t Number(const std::string& text, int base = 0) {
    size_t used = 0;
    uint64_t value = 0;
    try { value = std::stoull(text, &used, base); } catch (...) { used = 0; }
    if (text.empty() || text[0] == '-' || used != text.size()) throw std::runtime_error("not a number: " + text);
    return value;
}
std::vector<std::string> Pieces(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::istringstream in(text);
    for (std::string part; std::getline(in, part, separator);) parts.push_back(part);
    return parts;
}

const SetInfo* ChosenSet(const State& s) {
    return s.passSet < kMaxSets && s.sets[s.passSet].present ? &s.sets[s.passSet] : nullptr;
}

// The numbered sets of an asset folder. A file that cannot be understood makes the set absent; it is reported.
void LoadSets(State& s, const fs::path& folder) {
    for (unsigned n = 0; n < kMaxSets; ++n) {
        SetInfo set;
        if (n >= 2) {
            try {
                std::error_code missing;
                const fs::path text = folder / ("net4_set" + std::to_string(n) + ".txt");
                if (fs::exists(text, missing)) {
                    std::ifstream file(text);
                    for (std::string line; std::getline(file, line);) {
                        std::istringstream in(line);
                        std::vector<std::string> t;
                        for (std::string word; in >> word;) t.push_back(word);
                        if (t.empty() || t[0][0] == '#') continue;
                        if (t[0]=="side") {
                            if(t.size()!=9 || t[1]!="split" || (t[2]!="a" && t[2]!="b") || t[3]!="stages" || t[5]!="hidden" || t[7]!="file" ||
                               t[8].find_first_of("/\\:")!=std::string::npos)throw std::runtime_error("bad side line: "+line);
                            auto& side=set.sides[t[2]=="b"];side.stages=unsigned(Number(t[4]));side.hidden=unsigned(Number(t[6]));
                            if(!side.stages || side.stages>8 || (side.hidden!=64 && side.hidden!=128 && side.hidden!=256))throw std::runtime_error("bad side shape");
                            side.blob=ReadAsset(folder/"split4"/t[8]);
                            if(side.blob.size()<256)throw std::runtime_error("short side record");
                            uint32_t header[24];memcpy(header,side.blob.data(),sizeof header);
                            if(header[0]!=0x3156344c || header[1]!=1 || header[2]!=512 || header[3]!=side.hidden || header[4]!=side.stages)throw std::runtime_error("side record header mismatch");
                            for(unsigned j=0;j<side.stages;++j)if(header[16+j]>5 || header[8+j]+128ull>=side.blob.size())throw std::runtime_error("bad side stage offset");
                            continue;
                        }
                        if (t[0] == "pool") {
                            if ((t.size() != 3 && t.size() != 5) || t[1] != "vit" || t[2] != "2" ||
                                (t.size() == 5 && t[3] != "min")) throw std::runtime_error("bad pool line: " + line);
                            set.poolVit = 2;
                            set.poolMin = t.size() == 5 ? unsigned(Number(t[4])) : 0;
                            continue;
                        }
                        if (t.size() < 2 || t[0] != "skip" || (t[1] != "split" && t[1] != "vit")) throw std::runtime_error("unknown line: " + line);
                        for (size_t i = 2; i < t.size(); ++i) {
                            if (t[i][0] == '#') break;
                            const uint64_t block = Number(t[i]);
                            const unsigned count = t[1] == "split" ? 16 : 8, each = 8;
                            if (block >= count || block % each == 0 || block % each == each - 1) throw std::runtime_error("a block that cannot be left out: " + line);
                            (t[1] == "split" ? set.skipSplit : set.skipVit) |= 1u << block;
                        }
                    }
                    set.present = true;
                }
                for (unsigned f = 0; f < kFamilies; ++f) {
                    const std::string stem = std::string(kFamilyName[f]) + "4";
                    const fs::path records = folder / stem / (stem + "_set" + std::to_string(n) + ".bin");
                    if (!fs::exists(records, missing)) continue;
                    set.blob[f] = ReadAsset(records);
                    set.present = true;
                }
            } catch (const std::exception& error) {
                set = SetInfo{};
                try { Report(s, "set " + std::to_string(n) + " is not used: " + error.what()); } catch (...) { }
            }
        }
        s.sets[n] = std::move(set);
    }
}

// Everything the manifest says is checked here, once: Replace() copies bytes by it.
Manifest LoadManifest(const fs::path& folder, unsigned family) {
    Manifest m;
    const std::string stem = std::string(kFamilyName[family]) + "4";
    // A package may leave the first set out: <family>4_trained.bin then stands in its place.
    std::error_code noFirst;
    m.trainedIsFirst = !fs::exists(folder / (stem + ".bin"), noFirst) && fs::exists(folder / (stem + "_trained.bin"), noFirst);
    m.blob = ReadAsset(folder / (stem + (m.trainedIsFirst ? "_trained.bin" : ".bin")));
    // The trained records are optional and share the manifest, so the file has to be the first one's size.
    if (std::error_code missing; !m.trainedIsFirst && fs::exists(folder / (stem + "_trained.bin"), missing)) {
        m.trained = ReadAsset(folder / (stem + "_trained.bin"));
        if (m.trained.size() != m.blob.size()) throw std::runtime_error(stem + "_trained.bin is not the size of " + stem + ".bin");
    }
    std::ifstream file(folder / (stem + ".txt"));
    if (!file) throw std::runtime_error("missing NVFP4 asset " + (folder / (stem + ".txt")).string());
    bool familySeen = false;
    for (std::string line; std::getline(file, line);) {
        std::istringstream in(line);
        std::vector<std::string> t;
        for (std::string word; in >> word;) t.push_back(word);
        if (t.empty() || t[0][0] == '#') continue;
        const auto invalid = [&](const char* why) { return std::runtime_error(std::string("invalid NVFP4 manifest line (") + why + "): " + line); };
        if (t[0] == "runtime" && t.size() == 2) m.runtimeSha = t[1];
        else if (t[0] == "family" && t.size() == 2) { if (t[1] != kFamilyName[family]) throw invalid("another family"); familySeen = true; }
        else if (t[0] == "evaluation" && t.size() == 3) { m.evaluationLaunches = unsigned(Number(t[1])); m.evaluationHash = Number(t[2], 16); }
        else if (t[0] == "kernel" && t.size() == 6 && t[2] == "file" && t[4] == "entry") {
            if (Number(t[1]) != m.kernels.size()) throw invalid("kernels are numbered from 0 in order");
            if (t[3].find_first_of("/\\:") != std::string::npos || t[3].find("..") != std::string::npos) throw invalid("a kernel file is a plain file name");
            m.kernels.emplace_back(t[3], t[5]);
        } else if (t[0] == "small_kernel" && (t.size() == 8 || (t.size() == 10 && t[8] == "block")) && t[2] == "kernel" && t[4] == "dims" && t[6] == "at_most") {
            const unsigned original=unsigned(Number(t[1])), replacement=unsigned(Number(t[3])), dims=unsigned(Number(t[5])), limit=unsigned(Number(t[7]));
            if (original>=m.kernels.size() || replacement>=m.kernels.size() || original==replacement ||
                dims>kMaxParams-8 || (dims&3) || !limit || limit>1048576 || m.smallKernels.contains(original))
                throw invalid("invalid/duplicate small-kernel gate; declare both kernels first");
            Manifest::SmallKernel gate{replacement,dims,limit};
            if (t.size()==10) {
                const auto shape=Pieces(t[9],'x');
                if (shape.size()!=3) throw invalid("small-kernel block shape");
                for (unsigned d=0;d<3;++d) gate.block[d]=unsigned(Number(shape[d]));
                if (!gate.block[0] || !gate.block[1] || !gate.block[2] ||
                    uint64_t(gate.block[0])*gate.block[1]*gate.block[2]>1024) throw invalid("small-kernel block shape");
            }
            m.smallKernels.emplace(original,gate);
        } else if (t[0] == "default" && t.size() == 3) {
            const int group = GroupOf(family, t[1]);
            if (group < 0) throw invalid("unknown group");
            m.defaults[group] = uint32_t(Number(t[2]));
        } else if (t[0] == "slot" && t.size() >= 4 && t.size() % 2 == 0) {
            if (Number(t[1]) != m.slots.size() || m.slots.size() >= kMaxSlots) throw invalid("slots are numbered from 0 in order, at most 128");
            Slot s;
            bool have[13]{};
            for (size_t i = 2; i + 1 < t.size(); i += 2) {
                const std::string& key = t[i]; const std::string& value = t[i + 1];
                if (key == "launch") { Number(value); have[0] = true; }
                else if (key == "name") { s.name = value; have[1] = true; }
                else if (key == "group") { const int g = GroupOf(family, value); if (g < 0) throw invalid("unknown group"); s.group = unsigned(g); have[2] = true; }
                else if (key == "kernel") { s.kernel = unsigned(Number(value)); have[3] = true; }
                else if (key == "block") {
                    const auto parts = Pieces(value, 'x');
                    if (parts.size() != 3) throw invalid("block");
                    for (unsigned d = 0; d < 3; ++d) s.block[d] = unsigned(Number(parts[d]));
                    have[4] = true;
                }
                else if (key == "params") { s.params = unsigned(Number(value)); have[5] = true; }
                else if (key == "record") { s.record = unsigned(Number(value)); have[6] = true; }
                else if (key == "size") { s.size = unsigned(Number(value)); have[7] = true; }
                else if (key == "w") { s.w = unsigned(Number(value)); have[8] = true; }
                else if (key == "options") { s.options = unsigned(Number(value)); have[9] = true; }
                else if (key == "offset") { s.offset = Number(value); have[10] = true; }
                else if (key == "bytes") { s.bytes = Number(value); have[11] = true; }
                else if (key == "launch_block") {
                    const auto parts = Pieces(value, 'x');
                    if (parts.size() != 3) throw invalid("launch_block");
                    for (unsigned d = 0; d < 3; ++d) s.launchBlock[d] = unsigned(Number(parts[d]));
                    if (!s.launchBlock[0] || !s.launchBlock[1] || !s.launchBlock[2] ||
                        uint64_t(s.launchBlock[0]) * s.launchBlock[1] * s.launchBlock[2] > 1024)
                        throw invalid("launch_block shape");
                }
                else if (key == "map") {
                    for (const auto& part : Pieces(value, ',')) {
                        const auto f = Pieces(part, ':');
                        if (f.size() != 3) throw invalid("map");
                        s.map.push_back({unsigned(Number(f[0])), unsigned(Number(f[1])), unsigned(Number(f[2]))});
                    }
                    have[12] = true;
                }
                else if (key == "with") {            // <slot>:<bits>[:<request bits both options words must hold>]
                    for (const auto& part : Pieces(value, ',')) {
                        const auto f = Pieces(part, ':');
                        if (f.size() != 2 && f.size() != 3) throw invalid("with");
                        s.partners.push_back({unsigned(Number(f[0])), uint32_t(Number(f[1])), f.size() == 3 ? uint32_t(Number(f[2])) : 0u});
                    }
                }
                else if (key == "set" || key == "require") {
                    for (const auto& part : Pieces(value, ',')) {
                        const auto f = Pieces(part, ':');
                        if (f.size() != 2) throw invalid("set / require");
                        (key == "set" ? s.set : s.require).emplace_back(unsigned(Number(f[0])), uint32_t(Number(f[1])));
                    }
                }
                else throw invalid("unknown field");
            }
            for (const bool present : have) if (!present) throw invalid("a field is missing");
            if (s.kernel >= m.kernels.size()) throw invalid("unknown kernel");
            if (!s.block[0] || !s.block[1] || !s.block[2] || uint64_t(s.block[0]) * s.block[1] * s.block[2] > 1024) throw invalid("block shape");
            if (s.params > kMaxParams || s.size > kMaxParams || s.size < 16 || s.size % 8) throw invalid("parameter block size");
            if (s.record % 8 || s.record + 8 > s.params || s.w % 8 || s.w + 8 > s.size || s.options % 4 || s.options + 4 > s.size ||
                (s.options < s.w + 8 && s.w < s.options + 4))
                throw invalid("record, w or options");
            if (s.offset % 256 || s.bytes > m.blob.size() || s.offset > m.blob.size() - s.bytes) throw invalid("the record is outside the weight file");
            const auto overlaps = [](unsigned a, unsigned an, unsigned b, unsigned bn) { return a < b + bn && b < a + an; };
            for (const auto& c : s.map)
                if (!c.bytes || c.dst > s.size || c.bytes > s.size - c.dst || c.src > s.params || c.bytes > s.params - c.src ||
                    overlaps(c.dst, c.bytes, s.w, 8) || overlaps(c.dst, c.bytes, s.options, 4))
                    throw invalid("map");
            for (const auto& [at, value] : s.set)
                if (at % 4 || at + 4 > s.size || overlaps(at, 4, s.w, 8) || overlaps(at, 4, s.options, 4)) throw invalid("set");
            if (s.partners.size() > kMaxPartners) throw invalid("too many partners");
            for (const auto& [at, value] : s.require) if (at % 4 || at + 4 > s.params) throw invalid("require");
            const auto known = m.names.find(s.name);
            if (known == m.names.end()) m.names[s.name] = {s.params, s.record};
            else if (known->second != std::pair<unsigned, unsigned>{s.params, s.record}) throw invalid("the same runtime kernel with another parameter block");
            m.slots.push_back(std::move(s));
        } else {
            throw invalid("unknown line");
        }
    }
    if (!familySeen || m.runtimeSha.size() != 64 || !m.evaluationLaunches || m.slots.empty())
        throw std::runtime_error("the NVFP4 manifest of family " + std::string(kFamilyName[family]) + " is incomplete");
    for (size_t n = 0; n < m.slots.size(); ++n)
        for (const auto& partner : m.slots[n].partners)
            if (partner.slot >= m.slots.size() || partner.slot == n || !partner.bits) throw std::runtime_error("the NVFP4 manifest names a partner slot that does not exist");
    m.loaded = true;
    return m;
}

// Every weight these layers use is in their own files: any NR runtime is accepted. Whether its launches are the ones
// these kernels replace is still checked per evaluation (launch count and kernel names) and per launch.
bool RuntimeMatches(State&, unsigned) { return GetModuleHandleW(L"nvngx_dlssnr.dll") != nullptr; }

void LoadWeights(State& s, Device& d, unsigned family, ID3D12Device* device) {
    auto& w = d.weights[family];
    if (!w.failure.empty()) throw std::runtime_error(w.failure);
    if (w.ready) return;
    try {
        const auto& blob = s.manifest[family].blob;
        const auto& trained = s.manifest[family].trained;
        const uint64_t second = (blob.size() + 4096 + 255) / 256 * 256;     // a kernel reads 16 bytes at a time
        uint64_t bytes = trained.empty() ? second : (second + trained.size() + 4096 + 255) / 256 * 256;   // the trained records follow in the same buffer
        w.setAt = {};
        for (unsigned n = 0; n < kMaxSets; ++n) {       // then each numbered set's records (the layout of the first)
            const auto& own = s.sets[n].blob[family];
            if (own.size() != blob.size()) continue;
            w.setAt[n] = bytes;
            bytes = (bytes + own.size() + 4096 + 255) / 256 * 256;
        }
        w.blendAt = (bytes + 511) / 512 * 512; bytes = w.blendAt + 4096;     // our copy of the history blend weight
        w.buffer ={CreateBuffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), bytes};
        w.trained = trained.empty() ? 0 : second;
        auto upload = CreateBuffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "NVFP4 upload map");
        memset(mapped, 0, size_t(bytes));
        memcpy(mapped, blob.data(), blob.size());
        if (!trained.empty()) memcpy(static_cast<unsigned char*>(mapped) + second, trained.data(), trained.size());
        for (unsigned n = 0; n < kMaxSets; ++n)
            if (w.setAt[n]) memcpy(static_cast<unsigned char*>(mapped) + w.setAt[n], s.sets[n].blob[family].data(), blob.size());
        memcpy(static_cast<unsigned char*>(mapped) + w.blendAt, &kBlendWeight, 2);
        upload->Unmap(0, nullptr);
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> commands; ComPtr<ID3D12Fence> fence;
        // Once the copy is submitted, a failure below must not free what it may still be using.
        struct Park {
            Device& d; Buffer& buffer; ComPtr<ID3D12Resource>& upload; ComPtr<ID3D12CommandQueue>& queue; ComPtr<ID3D12CommandAllocator>& allocator;
            ComPtr<ID3D12GraphicsCommandList>& commands; bool armed = false;
            ~Park() {
                if (!armed) return;
                try {
                    for (IUnknown* object : {static_cast<IUnknown*>(buffer.resource.Get()), static_cast<IUnknown*>(upload.Get()),
                                             static_cast<IUnknown*>(queue.Get()), static_cast<IUnknown*>(allocator.Get()),
                                             static_cast<IUnknown*>(commands.Get())})
                        d.parked.emplace_back(object);
                } catch (...) {
                }
            }
        } park{d, w.buffer, upload, queue, allocator, commands};
        Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "NVFP4 upload queue");
        Check(device->CreateCommandAllocator(queueDesc.Type, IID_PPV_ARGS(&allocator)), "NVFP4 upload allocator");
        Check(device->CreateCommandList(0, queueDesc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)), "NVFP4 upload list");
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "NVFP4 upload fence");
        D3D12_RESOURCE_BARRIER transition{};
        transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition = {w.buffer.resource.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST};
        commands->ResourceBarrier(1, &transition);
        commands->CopyBufferRegion(w.buffer.resource.Get(), 0, upload.Get(), 0, bytes);
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
        w.ready = true;
    } catch (const std::exception& error) {
        w.buffer = {};
        w.failure = error.what();
        throw;
    }
}

NVDX_ObjectHandle KernelFor(State& s, Device& d, ID3D12Device* device, unsigned family, unsigned number) {
    const auto& m = s.manifest[family];
    auto& kernels = d.kernels[family];
    if (kernels.size() < m.kernels.size()) kernels.resize(m.kernels.size());
    auto& k = kernels[number];
    if (!k.failure.empty()) throw std::runtime_error(k.failure);
    if (k.function) return k.function;
    try {
        if (!s.createModule || !s.createFunction) throw std::runtime_error("the driver's CUDA entry points were not seen");
        const auto& [file, entry] = m.kernels[number];
        auto& module = d.modules[family][file];
        if (!module.failure.empty()) throw std::runtime_error(module.failure);
        if (!module.module) {
            try {
                const auto cubin = ReadAsset(s.options.assets / (std::string(kFamilyName[family]) + "4") / file);
                // An older GPU refuses this module (it is built for the RTX 50 series' NVFP4 tensor-core instruction).
                NvCheck(s.createModule(device, cubin.data(), unsigned(cubin.size()), &module.module), "NVFP4 kernel module " + file + " (needs an RTX 50 series GPU)");
            } catch (const std::exception& error) {
                module.module = nullptr;
                module.failure = error.what();
                throw;
            }
        }
        NvCheck(s.createFunction(device, module.module, entry.c_str(), &k.function), entry);
        return k.function;
    } catch (const std::exception& error) {
        k.function = nullptr;
        k.failure = error.what();
        throw;
    }
}

// The options word a slot's kernel gets before any partner bits: the host's value for its group, or the manifest's.
uint32_t Word(const State& s, unsigned family, unsigned number) {
    const Manifest& m = s.manifest[family];
    const int64_t given = s.options.family[family].options[m.slots[number].group];
    return given >= 0 ? uint32_t(given) : m.defaults[m.slots[number].group];
}

bool Wanted(const State& s, unsigned family, unsigned number) {
    const auto& options = s.options.family[family];
    return s.enabled && !s.paused && s.manifest[family].loaded && ((options.groups >> s.manifest[family].slots[number].group) & 1) &&
           ((options.slots[number / 64] >> (number % 64)) & 1);
}

// Whether slot `number` will get a kernel of ours when the runtime launches it later in this evaluation: what a
// kernel that hands it data in a format of our own relies on. Makes its kernel and weights now, so that little is
// left to fail then.
bool WillReplace(State& s, ID3D12Device* device, unsigned family, unsigned number) noexcept {
    try {
        if (!Wanted(s, family, number) || !RuntimeMatches(s, family)) return false;
        auto& d = s.devices[device];
        LoadWeights(s, d, family, device);
        const auto kernel=s.manifest[family].slots[number].kernel;
        KernelFor(s, d, device, family, kernel);
        if (const auto gated=s.manifest[family].smallKernels.find(kernel); gated!=s.manifest[family].smallKernels.end())
            KernelFor(s,d,device,family,gated->second.kernel); // preflight before promising a private-format partner
        return true;
    } catch (...) {
        return false;
    }
}

// Our launch for the runtime's launch `k` of a slot; throws before anything is recorded if it cannot be done.
void Replace(State& s, ID3D12GraphicsCommandList* commands, ID3D12Device* device, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& k, unsigned family,
             const Slot& slot, uint32_t options) {
    if (!k.pParams || k.paramSize != slot.params) throw std::runtime_error("the runtime's parameter block has another size");
    if (k.blockDim.x != slot.block[0] || k.blockDim.y != slot.block[1] || k.blockDim.z != slot.block[2] || k.dynSharedMemBytes ||
        !k.gridDim.x || !k.gridDim.y || !k.gridDim.z)
        throw std::runtime_error("the runtime's launch has an unexpected shape");
    const auto* theirs = static_cast<const unsigned char*>(k.pParams);
    for (const auto& [at, value] : slot.require)
        if (memcmp(theirs + at, &value, 4)) throw std::runtime_error("the launch uses an input or mode the 4-bit kernel does not implement (parameter " + std::to_string(at) + ")");
    auto& d = s.devices[device];
    LoadWeights(s, d, family, device);
    alignas(8) unsigned char ours[kMaxParams]{};
    for (const auto& c : slot.map) memcpy(ours + c.dst, theirs + c.src, c.bytes);
    for (const auto& [at, value] : slot.set) memcpy(ours + at, &value, 4);
    if (family == PrePost && slot.params == 184)        // the output block: our history blend weight, not the runtime's
        for (const auto& c : slot.map)
            if (c.src <= kBlendParam && c.src + c.bytes >= kBlendParam + 8) {
                uint64_t theirsAt; memcpy(&theirsAt, theirs + kBlendParam, 8);
                const uint64_t own = d.weights[family].buffer.Address() + d.weights[family].blendAt;
                if (theirsAt) memcpy(ours + c.dst + (kBlendParam - c.src), &own, 8);
            }
    uint64_t records = s.options.family[family].trained ? d.weights[family].trained : 0;
    if (s.passSet < kMaxSets && s.sets[s.passSet].present && d.weights[family].setAt[s.passSet]) records = d.weights[family].setAt[s.passSet];
    else if (family != PrePost && (s.passSet < kMaxSets || (s.options.family[family].trained && !records && !s.manifest[family].trainedIsFirst)))
        ++s.substituted;                        // the set asked for (or the trained records) has no file for this family
    const uint64_t record = d.weights[family].buffer.Address() + records + slot.offset;
    memcpy(ours + slot.w, &record, 8);
    memcpy(ours + slot.options, &options, 4);
    NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};
    unsigned kernel=slot.kernel;
    const Manifest::SmallKernel* selectedGate=nullptr;
    if (const auto gated=s.manifest[family].smallKernels.find(kernel); gated!=s.manifest[family].smallKernels.end()) {
        const auto& gate=gated->second;
        if (gate.dims+8>slot.size) throw std::runtime_error("small-kernel dimensions outside parameter block");
        uint32_t dims[2]{};memcpy(dims,ours+gate.dims,sizeof dims);
        if (dims[0] && dims[1] && uint64_t(dims[0])*dims[1]<=gate.maxTokens) { kernel=gate.kernel; selectedGate=&gate; }
    }
    launch.hFunction = KernelFor(s, d, device, family, kernel);
    launch.gridDim = k.gridDim;
    launch.blockDim = k.blockDim;
    if (slot.launchBlock[0]) launch.blockDim = {slot.launchBlock[0], slot.launchBlock[1], slot.launchBlock[2]};
    if (selectedGate && selectedGate->block[0]) launch.blockDim = {selectedGate->block[0],selectedGate->block[1],selectedGate->block[2]};
    launch.pParams = ours;
    launch.paramSize = slot.size;
    launch.dynSharedMemBytes = 0;
    NvCheck(s.launch(commands, &launch, 1), "NVFP4 kernel launch");
}

// The blocks of the bottleneck that the chosen set leaves out. Nothing of such a block is launched; the next block
// that is launched gets the last launched block's result as its input and that block's flags to wait for (`block`
// then holds the runtime's parameters with those fields changed). Offsets in the runtime's parameter blocks:
//   split  ffwd: in +0, producer flags +32 | ffwd_proj: the block's input +8 | proj: out +16, its flags +48
//   vit    expand: producer flags +48 (every block reads and writes the same stream buffer) | projection: its flags +56
enum class Act { Keep, Drop, Rewired };
Act Depth(State& s, ListState& list, const Target& target, ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& k,
          unsigned char* block) noexcept {
    auto& d = list.depth;
    const auto* p = static_cast<const unsigned char*>(k.pParams);
    const auto field = [&](unsigned at) { uint64_t value = 0; memcpy(&value, p + at, 8); return value; };
    const auto put = [&](unsigned at, uint64_t value) { memcpy(block + at, &value, 8); };
    const SetInfo* set = ChosenSet(s);
    const bool may = set && d.recognised;
    switch (target.role) {
    case Role::SplitFfwd: {
        const unsigned index = d.splitBlocks++, place = index % 8;
        d.dropSplit = may && index < 16 && ((set->skipSplit >> index) & 1) && place != 0 && place != 7 && d.out && d.outFlags;
        if (d.dropSplit) return Act::Drop;
        if (!d.rewire || k.paramSize < 40) { d.rewire = false; return Act::Keep; }
        memcpy(block, p, k.paramSize);
        put(0, d.out); put(32, d.outFlags);
        return Act::Rewired;
    }
    case Role::SplitFfwdProj:
        if (d.dropSplit) return Act::Drop;
        if (!d.rewire) return Act::Keep;
        d.rewire = false;
        memcpy(block, p, k.paramSize);
        put(8, d.out);
        return Act::Rewired;
    case Role::SplitQkv:
        return d.dropSplit ? Act::Drop : Act::Keep;
    case Role::SplitProj:
        if (d.dropSplit) { d.dropSplit = false; d.rewire = true; return Act::Drop; }
        d.out = d.outFlags = 0;
        if (k.paramSize >= 56) { d.out = field(16); d.outFlags = field(48); }
        return Act::Keep;
    case Role::VitExpandFirst:
        d.vitBlocks = 1; d.dropVit = d.vitRewire = false; d.vitFlags = 0;
        return Act::Keep;
    case Role::VitExpand: {
        const unsigned index = d.vitBlocks++;
        d.dropVit = may && index >= 1 && index < 7 && ((set->skipVit >> index) & 1) && d.vitFlags;
        if (d.dropVit) { NrFfnV2::SkipPair(commands); return Act::Drop; }
        if (!d.vitRewire || k.paramSize < 56) { d.vitRewire = false; return Act::Keep; }
        d.vitRewire = false;
        memcpy(block, p, k.paramSize);
        put(48, d.vitFlags);
        return Act::Rewired;
    }
    case Role::VitContract: case Role::VitQkv: case Role::VitAttention:
        return d.dropVit ? Act::Drop : Act::Keep;
    case Role::VitProjection:
        if (d.dropVit) { d.dropVit = false; d.vitRewire = true; return Act::Drop; }
        d.vitFlags = k.paramSize >= 64 ? field(56) : 0;
        return Act::Keep;
    default:
        return Act::Keep;
    }
}

// An evaluation ended (its list starts another, or the host said so): if it was a complete Model A evaluation, its
// weight records are now known by address.
void Learn(State& s, ListState& list) noexcept {
    try {
        if (!list.device) return;
        auto& known = s.devices[list.device].known;
        size_t slots = 0;
        for (unsigned f = 0; f < kFamilies; ++f) slots += s.manifest[f].slots.size();
        bool fresh = false;
        for (unsigned f = 0; f < kFamilies; ++f) {
            const auto& m = s.manifest[f];
            const auto& run = list.family[f];
            if (!m.loaded || list.launches != m.evaluationLaunches || list.hash != m.evaluationHash || run.launches != m.slots.size() || !run.sequence) continue;
            for (unsigned n = 0; n < m.slots.size(); ++n) {
                const auto found = known.find(run.records[n]);
                if (found != known.end() && found->second != std::pair<unsigned, unsigned>{f, n}) { known.clear(); break; }   // addresses were reused: start again
            }
            if (known.size() + m.slots.size() > kFeatures * slots) known.clear();
            for (unsigned n = 0; n < m.slots.size(); ++n) fresh |= known.emplace(run.records[n], std::pair<unsigned, unsigned>{f, n}).second;
        }
        if (fresh) { ++s.learnt; Report(s, "Model A's launches recognised (" + std::to_string(known.size()) + " weight records)"); }
    } catch (...) {
    }
}

void Resolve(State& s, Target& t) {
    if (t.generation == s.generation) return;
    t.generation = s.generation;
    t.family = -1;
    for (unsigned f = 0; f < kFamilies; ++f) {
        if (!s.manifest[f].loaded) continue;
        const auto found = s.manifest[f].names.find(t.name);
        if (found == s.manifest[f].names.end()) continue;
        t.family = int(f); t.params = found->second.first; t.record = found->second.second;
        return;
    }
}

NvAPI_Status __cdecl CreateModule(ID3D12Device* device, const void* blob, NvU32 bytes, NVDX_ObjectHandle* out) {
    return S().createModule(device, blob, bytes, out);
}

NvAPI_Status __cdecl CreateFunction(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& s = S();
    const auto result = s.createFunction(device, module, name, out);
    if (result == NVAPI_OK && name && out) {
        try {
            std::lock_guard guard(s.mutex);
            const auto found = s.targets.find(*out);
            if (found != s.targets.end() && found->second.module == module && found->second.device == device && found->second.name == name) {
                ++found->second.refs;           // the same kernel handed out again, to another feature
            } else {
                Target t;                       // (a handle value can be reused after its function was destroyed)
                t.device = device; t.module = module; t.name = name; t.refs = 1; t.clear = t.name == "cc_cb_clear";
                // With the Natural and Cinematic styles the runtime launches a colour kernel after the network's last
                // launch (Standard has none). It belongs to no layer, and counting it would keep the evaluation from
                // being recognised as Model A's.
                t.outside = t.name == "cg2r_post_process_kernel";
                t.role = RoleOf(t.name);
                s.targets[*out] = std::move(t);
            }
            // A feature of the runtime is being created: its weights may land where another model's were.
            const auto d = s.devices.find(device);
            if (d != s.devices.end() && !strncmp(name, "cc_", 3)) d->second.known.clear();
        } catch (...) {
        }
    }
    return result;
}

void PoolBarrier(ID3D12GraphicsCommandList* commands) {
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    commands->ResourceBarrier(1, &barrier);
}
uint64_t PoolBuffer(Device& d, ID3D12Device* device, uint64_t original, unsigned tokens) {
    if (!original) throw std::runtime_error("null transformer activation");
    auto& b = d.poolBuffers[{original, tokens}];
    if (!b.resource) {
        b.bytes = uint64_t((tokens + 127) / 128) * 128 * 4096;
        b.resource = CreateBuffer(device, b.bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    return b.Address();
}
// 0: unchanged, 1: rewritten block/grid, 2: repack replaced. Counters, scratch and
// flags remain the runtime's (its clear already reset them); they are oversized for
// the pooled count. Weight pointers stay intact so normal record selection works.
int PoolRewrite(State& s, ListState& list, Target& target, ID3D12GraphicsCommandList* commands,
                const NVAPI_CU_KERNEL_LAUNCH_PARAMS& in, NVAPI_CU_KERNEL_LAUNCH_PARAMS& out, unsigned char* data) {
    const bool begin = target.name == "cc_vit_1d_repack_2d_to_1d_fp8";
    const bool end = target.name == "cc_vit_1d_repack_1d_to_2d_fp8";
    auto& p = list.pool;
    auto& d = s.devices[target.device];
    if (begin) {
        const auto set = ChosenSet(s);
        if (!s.enabled || s.paused || !set || !set->poolVit || !list.depth.recognised) return 0;
        if (in.paramSize != 24 || !in.pParams) throw std::runtime_error("unexpected pool repack ABI");
        struct Repack { uint64_t input,output; unsigned h,w; } r;
        memcpy(&r,in.pParams,sizeof r);
        if (!r.h || !r.w || r.h % 4 || r.w % 4 || uint64_t(r.h)*r.w > 16384) throw std::runtime_error("unsupported pool map");
        if (r.h*r.w < set->poolMin) return 0;
        if (!d.poolModule) {
            const auto bytes = ReadAsset(s.options.assets / "vit4" / "vit_pool.cubin");
            NvCheck(s.createModule(target.device,bytes.data(),unsigned(bytes.size()),&d.poolModule),"pool module");
            NvCheck(s.createFunction(target.device,d.poolModule,"nr_vit_pool2",&d.poolKernel),"pool kernel");
            NvCheck(s.createFunction(target.device,d.poolModule,"nr_vit_unpool2",&d.unpoolKernel),"unpool kernel");
        }
        p.h=r.h; p.w=r.w; p.tokens=((r.h+1)/2)*((r.w+1)/2); p.full=r.input;
        p.base=PoolBuffer(d,target.device,r.input,p.tokens);
        struct Params { uint64_t full,stream,base,out; unsigned h,w; } args{p.full,PoolBuffer(d,target.device,r.output,p.tokens),p.base,0,p.h,p.w};
        if (GetEnvironmentVariableA("NR_VIT_POOL_PROBE",nullptr,0)) {
            if (!d.poolProbeFull.resource || d.poolProbeH!=p.h || d.poolProbeW!=p.w) {
                if(d.poolProbeFull.resource)throw std::runtime_error("pool probe supports one size per checker process");
                d.poolProbeH=p.h;d.poolProbeW=p.w;
                d.poolProbeFull.bytes=uint64_t(p.h)*p.w*1024;
                d.poolProbe.bytes=2*d.poolProbeFull.bytes+uint64_t((p.tokens+127)/128)*128*1024*2;
                d.poolProbeFull.resource=CreateBuffer(target.device,d.poolProbeFull.bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                d.poolProbe.resource=CreateBuffer(target.device,d.poolProbe.bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                d.poolReadback.resource=CreateBuffer(target.device,d.poolProbe.bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
                NvCheck(s.createFunction(target.device,d.poolModule,"nr_vit_pool_capture",&d.poolCapture),"pool capture");
            }
            args.out=d.poolProbeFull.Address();
        }
        out=in;out.hFunction=d.poolKernel;out.pParams=&args;out.paramSize=sizeof args;
        out.gridDim={unsigned((p.tokens+127)/128*128*1024/256),1,1};out.blockDim={256,1,1};out.dynSharedMemBytes=0;
        PoolBarrier(commands);NvCheck(s.launch(commands,&out,1),"pooled repack");PoolBarrier(commands);
        p.active=true;
        Notice(s,"ViT pooling " + std::to_string(p.h) + "x" + std::to_string(p.w) + " -> " + std::to_string((p.h+1)/2) + "x" + std::to_string((p.w+1)/2));
        return 2;
    }
    if (!p.active) return 0;
    if (end) {
        if (in.paramSize != 24 || !in.pParams) throw std::runtime_error("unexpected unpool ABI");
        uint64_t ptr[2];memcpy(ptr,in.pParams,16);
        struct Params { uint64_t full,stream,base,out; unsigned h,w; } args{p.full,PoolBuffer(d,target.device,ptr[0],p.tokens),p.base,ptr[1],p.h,p.w};
        out=in;out.hFunction=d.unpoolKernel;out.pParams=&args;out.paramSize=sizeof args;
        out.gridDim={unsigned((uint64_t(p.h)*p.w*1024+255)/256),1,1};out.blockDim={256,1,1};out.dynSharedMemBytes=0;
        PoolBarrier(commands);NvCheck(s.launch(commands,&out,1),"unpooled repack");PoolBarrier(commands);
        if (d.poolProbe.resource) {
            struct Capture { uint64_t full,stream,base,dump;unsigned h,w;uint64_t result; } capture{d.poolProbeFull.Address(),args.stream,p.base,d.poolProbe.Address(),p.h,p.w,ptr[1]};
            auto launch=out;launch.hFunction=d.poolCapture;launch.pParams=&capture;launch.paramSize=sizeof capture;
            NvCheck(s.launch(commands,&launch,1),"pool diagnostic");PoolBarrier(commands);
            D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition={d.poolProbe.resource.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
            commands->ResourceBarrier(1,&b);commands->CopyBufferRegion(d.poolReadback.resource.Get(),0,d.poolProbe.resource.Get(),0,d.poolProbe.bytes);
            std::swap(b.Transition.StateBefore,b.Transition.StateAfter);commands->ResourceBarrier(1,&b);
        }
        p.active=false;return 2;
    }
    unsigned dims=0;
    std::vector<unsigned> pointers;
    const unsigned tiles=(p.tokens+127)/128;
    out=in;
    switch(target.role) {
    case Role::VitExpandFirst: case Role::VitExpand: pointers={0,16};dims=64;out.gridDim={32*tiles,1,1};break;
    case Role::VitContract: pointers={0,8,16};dims=64;out.gridDim={8*tiles,1,4};break;
    case Role::VitQkv: pointers={0,8,16,24};dims=72;out.gridDim={16*tiles,1,2};break;
    case Role::VitAttention: pointers={0,8,16,24};dims=56;out.gridDim={32,(tiles+1)/2,1};break;
    case Role::VitProjection: pointers={0,8,16};dims=64;out.gridDim={8*tiles,1,4};break;
    default: throw std::runtime_error("unexpected launch within pooled transformer: "+target.name);
    }
    if (!in.pParams || in.paramSize != dims+8) throw std::runtime_error("pooled transformer ABI mismatch");
    memcpy(data,in.pParams,in.paramSize);
    for (unsigned offset:pointers) { uint64_t ptr;memcpy(&ptr,data+offset,8);ptr=PoolBuffer(d,target.device,ptr,p.tokens);memcpy(data+offset,&ptr,8); }
    unsigned shape[2]{(p.h+1)/2,(p.w+1)/2};memcpy(data+dims,shape,8);out.pParams=data;
    return 1;
}

bool SideLaunch(State& s,ListState& list,Target& target,ID3D12GraphicsCommandList* commands,
                const NVAPI_CU_KERNEL_LAUNCH_PARAMS& launch,unsigned number) {
    auto& p=list.side;
    const auto set=ChosenSet(s);
    if(!p.active) {
        if(!s.enabled || s.paused || !set || target.role!=Role::SplitFfwd ||
           (list.depth.splitBlocks!=1 && list.depth.splitBlocks!=9))return false;
        const unsigned side=list.depth.splitBlocks==9;
        const auto& record=set->sides[side];if(record.blob.empty())return false;
        if(!RuntimeMatches(s,Split) || launch.paramSize!=56 || !launch.pParams)throw std::runtime_error("Side4 input ABI mismatch");
        const auto raw=static_cast<const unsigned char*>(launch.pParams);uint64_t input;int h,w;
        memcpy(&input,raw,8);memcpy(&h,raw+24,4);memcpy(&w,raw+28,4);
        if(!input || h<=0 || w<=0 || h%4 || w%4 || h>1024 || w>1024)throw std::runtime_error("Side4 input shape mismatch");
        auto& d=s.devices[target.device];auto& native=d.sides[{s.passSet,side}];
        if(!native.function || !native.weights.resource || !native.upload) {
            const auto file=std::string("side4_")+(side?"b":"a")+"_h"+std::to_string(record.hidden)+"_s"+std::to_string(record.stages)+".cubin";
            const auto bytes=ReadAsset(s.options.assets/"split4"/file);
            NvCheck(s.createModule(target.device,bytes.data(),unsigned(bytes.size()),&native.module),"Side4 module");
            NvCheck(s.createFunction(target.device,native.module,"nr_side4",&native.function),"Side4 function");
            native.weights.bytes=(record.blob.size()+4096+255)/256*256;
            native.weights.resource=CreateBuffer(target.device,native.weights.bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
            native.upload=CreateBuffer(target.device,native.weights.bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
            void* dst=nullptr;Check(native.upload->Map(0,nullptr,&dst),"Side4 weights map");memset(dst,0,size_t(native.weights.bytes));memcpy(dst,record.blob.data(),record.blob.size());native.upload->Unmap(0,nullptr);
            commands->CopyBufferRegion(native.weights.resource.Get(),0,native.upload.Get(),0,native.weights.bytes);
            D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition={native.weights.resource.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS};commands->ResourceBarrier(1,&b);
        }
        p={true,side,s.passSet,number,0,input,h,w};
    }
    if(s.passSet!=p.set || number!=p.first+p.seen || launch.paramSize>kMaxParams)throw std::runtime_error("Side4 sequence mismatch after suppression");
    ++p.seen;
    if(p.seen!=32)return true;
    if(target.role!=Role::SplitProj || launch.paramSize!=(p.side?72u:80u))throw std::runtime_error("Side4 last launch mismatch");
    const auto raw=static_cast<const unsigned char*>(launch.pParams);
    uint64_t output,pool=0,flags=0;memcpy(&output,raw+16,8);
    int h,w,dh=(p.h/2+3)/4*4,dw=(p.w/2+3)/4*4;
    if(p.side) {memcpy(&h,raw+32,4);memcpy(&w,raw+36,4);}
    else {memcpy(&pool,raw+24,8);memcpy(&flags,raw+48,8);memcpy(&h,raw+64,4);memcpy(&w,raw+68,4);memcpy(&dh,raw+72,4);memcpy(&dw,raw+76,4);}
    if(h!=p.h || w!=p.w || !output || output==p.input || (!p.side && (!pool || dh!=(p.h/2+3)/4*4 || dw!=(p.w/2+3)/4*4)))throw std::runtime_error("Side4 destination mismatch");
    auto& native=s.devices[target.device].sides.at({p.set,p.side});
    struct Params {uint64_t in,out,weights,skip,pflags,oflags,pool;int h,w,sx,sy,psx,psy,dh,dw,eh,ew;unsigned options,reserved;};
    Params args{p.input,output,native.weights.Address(),0,0,flags,pool,p.h,p.w,0,0,0,0,dh,dw,p.h,p.w,0,0};
    NVAPI_CU_KERNEL_LAUNCH_PARAMS k{};k.hFunction=native.function;k.gridDim={unsigned((p.w+7)/8),unsigned((p.h+7)/8),1};k.blockDim={32,2,1};k.pParams=&args;k.paramSize=sizeof args;
    PoolBarrier(commands);NvCheck(s.launch(commands,&k,1),"Side4 launch");PoolBarrier(commands);
    s.leftOut+=31;++s.sideRuns[p.side];p={};return true;
}

NvAPI_Status __cdecl Launch(ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    Target* target = nullptr;
    if (kernels && count == 1) {
        const auto found = s.targets.find(kernels->hFunction);
        if (found != s.targets.end()) target = &found->second;
    }
    if (target && target->clear) {              // a new evaluation
        try {
            auto& list = s.lists[commands];
            Learn(s, list);
            list = ListState{};
            list.device = target->device;
            list.launches = 1;
            list.hash = Fnv(kFnvOffset, target->name);
        } catch (...) {
        }
        return s.launch(commands, kernels, count);
    }
    if (target && target->outside) return s.launch(commands, kernels, count);
    const auto running = s.lists.find(commands);
    ListState* list = running != s.lists.end() ? &running->second : nullptr;
    if (!target) {                              // not a kernel we saw created, or a chain: this is not an evaluation we know
        if (list) { list->launches += count ? count : 1; list->hash = Fnv(list->hash, "?"); }
        return s.launch(commands, kernels, count);
    }
    if (list) { ++list->launches; list->hash = Fnv(list->hash, target->name); }
    NVAPI_CU_KERNEL_LAUNCH_PARAMS pooled{};
    alignas(8) unsigned char poolData[kMaxParams];
    if (list) {
        try {
            const int action=PoolRewrite(s,*list,*target,commands,*kernels,pooled,poolData);
            if (action==2) return NVAPI_OK;
            if (action==1) kernels=&pooled;
        } catch (const std::exception& error) {
            Notice(s,std::string("pool failure: ")+error.what());
            // Never fall back to full-token launches after a pooled stream started.
            if (list->pool.active) return NVAPI_ERROR;
        }
    }
    // The layers below never see a launch that a layer above them left out: the feed-forward layer recognises the model
    // by the runtime's own count.
    if (list && target->role == Role::VitExpandFirst) NrFfnV2::RuntimeLaunches(commands, list->launches - 1, list->pool.active ? list->pool.h*list->pool.w : 0);
    Act act = Act::Keep;
    NVAPI_CU_KERNEL_LAUNCH_PARAMS rewired{};
    alignas(8) unsigned char block[kMaxParams];
    if (list && target->role != Role::None && s.enabled && !s.paused && kernels->pParams && kernels->paramSize >= 16 && kernels->paramSize <= kMaxParams) {
        act = Depth(s, *list, *target, commands, *kernels, block);
        if (act == Act::Rewired) { rewired = *kernels; rewired.pParams = block; kernels = &rewired; }
        if (act == Act::Drop) ++s.leftOut;
    }
    Resolve(s, *target);
    if (target->family < 0) return act == Act::Drop ? NVAPI_OK : s.launch(commands, kernels, count);
    const unsigned f = unsigned(target->family);
    const Manifest& m = s.manifest[f];
    int number = -1;                            // the slot this launch is, when we can tell
    bool forced = false;                        // a kernel of ours that already ran counts on this one being ours
    try {
        uint64_t record = 0;
        if (kernels->pParams && kernels->paramSize == target->params) memcpy(&record, static_cast<const char*>(kernels->pParams) + target->record, 8);
        if (list) {
            auto& run = list->family[f];
            const unsigned n = run.launches++;
            if (n < m.slots.size() && m.slots[n].name == target->name && record) {
                run.records[n] = record;
                if (s.options.byOrder && run.sequence) number = int(n);
            } else {
                run.sequence = false;
            }
        }
        if (!s.options.byOrder && record) {
            const auto d = s.devices.find(target->device);
            if (d != s.devices.end()) {
                const auto found = d->second.known.find(record);
                if (found != d->second.known.end() && found->second.first == f && m.slots[found->second.second].name == target->name)
                    number = int(found->second.second);
            }
        }
        if (list) list->depth.recognised = number >= 0;
        if (list && f==Split && number>=0 && SideLaunch(s,*list,*target,commands,*kernels,unsigned(number))) {
            if(act==Act::Drop)--s.leftOut; // Side4's 31 suppressed launches are counted once, at its boundary.
            return NVAPI_OK;
        }
        if (act == Act::Drop) return NVAPI_OK;      // a block the set leaves out (its launch was counted above, as the runtime made it)
        if (number < 0) {
            if (s.enabled && !s.paused) ++s.passed[f];
            return s.launch(commands, kernels, count);
        }
        const Slot& slot = m.slots[number];
        const bool wanted = Wanted(s, f, unsigned(number));
        // A private format between two kernels of ours needs both to be ours and both options words to ask for it
        // (`request`): with one side switched off by the host, both keep the runtime's format.
        const uint32_t word = Word(s, f, unsigned(number));
        uint32_t bits = 0;
        uint8_t promise = 0;
        if (list) {
            auto& run = list->family[f];
            for (unsigned i = 0; i < slot.partners.size(); ++i) {
                const auto& partner = slot.partners[i];
                const unsigned other = partner.slot;
                if (other < unsigned(number)) {
                    // It ran already. Did it get a kernel of ours that counts on this one? Then this one follows it,
                    // request bits included, whatever the settings have become since.
                    const auto& theirs = m.slots[other].partners;
                    for (unsigned j = 0; j < theirs.size(); ++j)
                        if (theirs[j].slot == unsigned(number) && run.state[other] == 2 && ((run.promised[other] >> j) & 1)) { bits |= partner.bits | partner.request; forced = true; }
                } else if (wanted && (word & partner.request) == partner.request && (Word(s, f, other) & partner.request) == partner.request &&
                           WillReplace(s, target->device, f, other)) {
                    bits |= partner.bits;
                    promise |= uint8_t(1u << i);
                }
            }
        }
        if (!wanted && !forced) {
            if (list) list->family[f].state[number] = 1;
            if (s.enabled && !s.paused) ++s.passed[f];
            return s.launch(commands, kernels, count);
        }
        if (!RuntimeMatches(s, f)) throw std::runtime_error("no NR runtime is loaded");
        Replace(s, commands, target->device, *kernels, f, slot, word | bits);
        if (list) { list->family[f].state[number] = 2; list->family[f].promised[number] = promise; }
        ++s.replaced[f];
        ++s.perGroup[f][slot.group];
        if (!s.active[f]) {
            s.active[f] = true;
            try { Report(s, std::string(kFamilyName[f]) + ": active"); } catch (...) { }
        }
        return NVAPI_OK;
    } catch (const std::exception& error) {
        // Nothing of ours was recorded for this launch: the runtime's own kernel runs.
        if(list && list->side.active) { Notice(s,std::string("Side4 failed after suppression: ")+error.what());return NVAPI_ERROR; }
        if (list && number >= 0) list->family[f].state[number] = 1;
        Notice(s, std::string(kFamilyName[f]) + (forced ? ": a launch was left to the runtime after a kernel of ours had written its own format for it (this frame's picture is wrong): "
                                                        : ": a launch was left to the runtime: ") + error.what());
        ++s.passed[f];
        return s.launch(commands, kernels, count);
    } catch (...) {
        if(list && list->side.active)return NVAPI_ERROR;
        if (list && number >= 0) list->family[f].state[number] = 1;
        Notice(s, std::string(kFamilyName[f]) + ": a launch was left to the runtime: unknown error");
        ++s.passed[f];
        return s.launch(commands, kernels, count);
    }
}

NvAPI_Status __cdecl DestroyFunction(ID3D12Device* device, NVDX_ObjectHandle function) {
    auto& s = S();
    const auto result = s.destroyFunction(device, function);
    if (result == NVAPI_OK) {
        std::lock_guard guard(s.mutex);
        const auto found = s.targets.find(function);
        if (found != s.targets.end()) {
            const auto d = s.devices.find(device);      // a feature is going away, and its weight records with it
            if (d != s.devices.end() && found->second.name.rfind("cc_", 0) == 0) d->second.known.clear();
            if (--found->second.refs == 0) s.targets.erase(found);
        }
    }
    return result;
}

NvAPI_Status __cdecl DestroyModule(ID3D12Device* device, NVDX_ObjectHandle module) {
    return S().destroyModule(device, module);
}

std::string Groups(unsigned family, unsigned mask) {
    std::string text;
    for (unsigned g = 0; g < kMaxGroups; ++g)
        if (kGroupName[family][g] && ((mask >> g) & 1)) text += (text.empty() ? "" : " ") + std::string(kGroupName[family][g]);
    return text;
}
}   // namespace

void Parse(const std::string& text, Options& options) {
    std::string spaced = text;
    std::replace(spaced.begin(), spaced.end(), ',', ' ');
    std::istringstream in(spaced);
    for (std::string item; in >> item;) {
        const size_t equals = item.find('=');
        if (equals == std::string::npos || !equals || equals + 1 == item.size()) throw std::runtime_error("NVFP4 setting without a value: " + item);
        std::string key = item.substr(0, equals);
        const std::string value = item.substr(equals + 1);
        std::string member;
        const size_t dot = key.find('.');
        if (dot != std::string::npos) { member = key.substr(dot + 1); key.erase(dot); }
        unsigned f = 0;
        while (f < kFamilies && key != kFamilyName[f]) ++f;
        if (f == kFamilies) throw std::runtime_error("unknown NVFP4 family in: " + item);
        auto& family = options.family[f];
        unsigned all = 0;
        for (unsigned g = 0; g < kMaxGroups; ++g) if (kGroupName[f][g]) all |= 1u << g;
        if (member.empty()) {
            if (value == "all") family.groups = all;
            else if (value[0] >= '0' && value[0] <= '9') family.groups = unsigned(Number(value)) & all;     // (no group's name starts with a digit)
            else {
                unsigned mask = 0;
                for (const auto& name : Pieces(value, '+')) {
                    const int g = GroupOf(f, name);
                    if (g < 0) throw std::runtime_error("unknown NVFP4 group in: " + item);
                    mask |= 1u << g;
                }
                family.groups = mask;
            }
        } else if (member == "trained") {
            if (value != "0" && value != "1") throw std::runtime_error("NVFP4 trained is 0 or 1: " + item);
            family.trained = value == "1";
        } else if (member == "slots") {
            if (value.empty() || value.size() > 32 || value.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
                throw std::runtime_error("NVFP4 slots are up to 32 hexadecimal digits: " + item);
            const std::string padded = std::string(32 - value.size(), '0') + value;
            family.slots = {Number(padded.substr(16), 16), Number(padded.substr(0, 16), 16)};
        } else {
            const int g = GroupOf(f, member);
            if (g < 0) throw std::runtime_error("unknown NVFP4 group in: " + item);
            family.options[g] = int64_t(Number(value) & 0xffffffffull);
        }
    }
}

void* WrapNvapi(unsigned id, void* original) {
    if (!original) return original;
    auto& s = S();
    std::lock_guard guard(s.mutex);
    switch (id) {
    case kCreateModule: s.createModule = static_cast<decltype(s.createModule)>(original); return static_cast<void*>(&CreateModule);
    case kCreateFunction: s.createFunction = static_cast<decltype(s.createFunction)>(original); return static_cast<void*>(&CreateFunction);
    case kLaunch: s.launch = static_cast<decltype(s.launch)>(original); return static_cast<void*>(&Launch);
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
        // Another asset folder: read it afresh and forget earlier load failures. What a device already holds stays in
        // use there until ReleaseDevice(), since recorded command lists may refer to it.
        for (auto& [device, d] : s.devices)
            for (unsigned f = 0; f < kFamilies; ++f)
                if (d.weights[f].ready || !d.modules[f].empty()) throw std::runtime_error("NVFP4 assets cannot change while a device holds them");
        for (auto& m : s.manifest) m = Manifest{};
        for (auto& [device, d] : s.devices) { d = Device{}; }
        ++s.generation;
        LoadSets(s, options.assets);
    }
    s.options = options;
    unsigned wanted = 0, loaded = 0;
    std::string failures;
    for (unsigned f = 0; f < kFamilies; ++f) {
        auto& m = s.manifest[f];
        if (!options.family[f].groups) continue;
        ++wanted;
        if (!m.loaded && m.failure.empty()) {
            try {
                m = LoadManifest(options.assets / (std::string(kFamilyName[f]) + "4"), f);
                ++s.generation;
            } catch (const std::exception& error) {
                m = Manifest{};
                m.failure = error.what();
            }
        }
        if (m.loaded) ++loaded;
        else failures += (failures.empty() ? "" : "; ") + std::string(kFamilyName[f]) + ": " + m.failure;
    }
    s.enabled = loaded != 0;
    s.active = {};
    s.notice.clear();
    if (!failures.empty()) {
        if (!loaded) throw std::runtime_error(failures);
        Report(s, "unavailable: " + failures);
    }
}

void Pause(bool paused) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.paused = paused;
}

void UseSet(unsigned set) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.passSet = set;
}

bool SetComplete(unsigned set, std::string* missing) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    for (unsigned f = 0; f < kFamilies; ++f) {
        if (f == PrePost || !s.options.family[f].groups) continue;
        const auto& m = s.manifest[f];
        const std::string stem = std::string(kFamilyName[f]) + "4";
        bool complete = m.loaded;
        std::string file = stem + ".txt";
        if (complete && set == kDefaultSet) {
            complete = !s.options.family[f].trained || !m.trained.empty() || m.trainedIsFirst;
            file = stem + "_trained.bin";
        } else if (complete) {
            complete = set < kMaxSets && !m.blob.empty() && s.sets[set].blob[f].size() == m.blob.size();
            file = stem + "_set" + std::to_string(set) + ".bin";
        }
        if (complete) continue;
        if (missing) *missing = stem + "/" + file;
        return false;
    }
    return true;
}

unsigned long long Faults() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    unsigned long long faults = s.substituted;
    for (const auto passed : s.passed) faults += passed;
    return faults;
}

void Disable() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.enabled = false;
    s.active = {};
    s.notice.clear();
}

void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept {
    try {
        auto& s = S();
        std::lock_guard guard(s.mutex);
        const auto found = s.lists.find(commands);
        if (found == s.lists.end()) return;
        Learn(s, found->second);
        s.lists.erase(found);
    } catch (...) {
    }
}

void ReleaseDevice(ID3D12Device* device) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const auto found = s.devices.find(device);
    if (found != s.devices.end()) {
        // On a lost device these calls fail, which is fine: the objects die with it.
        if (s.destroyFunction && found->second.poolKernel) s.destroyFunction(device,found->second.poolKernel);
        if (s.destroyFunction && found->second.unpoolKernel) s.destroyFunction(device,found->second.unpoolKernel);
        if (s.destroyFunction && found->second.poolCapture) s.destroyFunction(device,found->second.poolCapture);
        if (s.destroyModule && found->second.poolModule) s.destroyModule(device,found->second.poolModule);
        for(auto& [key,side]:found->second.sides) {
            if(s.destroyFunction && side.function)s.destroyFunction(device,side.function);
            if(s.destroyModule && side.module)s.destroyModule(device,side.module);
        }
        for (unsigned f = 0; f < kFamilies; ++f) {
            for (auto& k : found->second.kernels[f]) if (s.destroyFunction && k.function) s.destroyFunction(device, k.function);
            for (auto& [file, module] : found->second.modules[f]) if (s.destroyModule && module.module) s.destroyModule(device, module.module);
        }
        s.devices.erase(found);
    }
    for (auto i = s.targets.begin(); i != s.targets.end();) i = i->second.device == device ? s.targets.erase(i) : std::next(i);
    s.lists.clear();                        // command lists of the device go with it; their addresses may be reused
}

void WritePoolProbe(ID3D12Device* device, const std::filesystem::path& path) {
    auto& s=S();std::lock_guard guard(s.mutex);auto i=s.devices.find(device);
    if(i==s.devices.end() || !i->second.poolReadback.resource)return;
    auto& d=i->second;void* bytes=nullptr;D3D12_RANGE range{0,size_t(d.poolProbe.bytes)};
    Check(d.poolReadback.resource->Map(0,&range,&bytes),"pool probe map");
    std::ofstream out(path,std::ios::binary);const uint32_t header[4]{0x36504c56,d.poolProbeH,d.poolProbeW,0};
    out.write(reinterpret_cast<const char*>(header),sizeof header);out.write(static_cast<const char*>(bytes),std::streamsize(d.poolProbe.bytes));
    D3D12_RANGE empty{0,0};d.poolReadback.resource->Unmap(0,&empty);
    if(!out)throw std::runtime_error("pool probe write failed");
}

std::string Status(bool counters) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    std::string status;
    for (unsigned f = 0; f < kFamilies; ++f) {
        const unsigned groups = s.options.family[f].groups;
        if (!groups) continue;
        const auto& m = s.manifest[f];
        status += (status.empty() ? "" : "; ") + std::string(kFamilyName[f]) + " ";
        if (!m.loaded) status += "unavailable (" + m.failure + ")";
        else if (!s.enabled) status += "off";
        else status += std::string(s.active[f] ? "active" : s.options.byOrder ? "waiting for the next evaluation" : "waiting to recognise Model A") + " (" + Groups(f, groups) + ")";
        if (m.loaded && s.options.family[f].trained) status += m.trained.empty() && !m.trainedIsFirst ? ", trained weights not found (the first set is in use)" : ", trained weights";
        if (counters && m.loaded) {
            status += " replaced " + std::to_string(s.replaced[f]) + " [";
            bool first = true;
            for (unsigned g = 0; g < kMaxGroups; ++g)
                if (kGroupName[f][g] && s.perGroup[f][g]) { status += (first ? "" : ", ") + std::string(kGroupName[f][g]) + " " + std::to_string(s.perGroup[f][g]); first = false; }
            status += "] left " + std::to_string(s.passed[f]);
        }
    }
    if (status.empty()) status = "off";
    if (const SetInfo* set = ChosenSet(s); set && (set->skipSplit || set->skipVit)) {
        unsigned blocks = 0;
        for (unsigned b = 0; b < 16; ++b) blocks += ((set->skipSplit >> b) & 1) + ((set->skipVit >> b) & 1);
        status += "; " + std::to_string(blocks) + " bottleneck blocks left out";
    }
    if (counters && s.leftOut) status += "; launches left out " + std::to_string(s.leftOut);
    if (counters && (s.sideRuns[0] || s.sideRuns[1])) status += "; Side4 a/b " + std::to_string(s.sideRuns[0]) + "/" + std::to_string(s.sideRuns[1]);
    if (s.enabled && !s.notice.empty()) status += " (" + s.notice + ")";
    return status;
}
}   // namespace NrNet4
