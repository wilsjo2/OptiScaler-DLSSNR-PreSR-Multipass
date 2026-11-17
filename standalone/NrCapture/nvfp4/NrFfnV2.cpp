// NVFP4 v2 feed-forward replacement; see NrFfnV2.h.
//
// The runtime launches each transformer block's feed-forward as two kernels with a 72-byte parameter block
// (eight pointers and the token grid's two dimensions):
//   expand   p0 = x, p2 = hidden, p3 = W1 record, p6 = producer flags, p7 = own flags
//   contract p0 = hidden, p1 = x (skip), p2 = output, p3 = W2 record, p4 = merge counter, p5 = scratch,
//            p6 = expand flags, p7 = own flags
// We record  pack(x) -> expand -> contract  on our own buffers instead, write the runtime's output buffer in the
// runtime's own layout, and leave the contract's flags and merge counter the way the runtime's kernel does, so the
// next kernel in its chain proceeds. Our kernels order themselves with flags of their own (or with null UAV barriers).
//
// The runtime's kernels wait for their predecessors without a time limit. So the one thing this layer must never do
// is launch a runtime kernel whose predecessor it swallowed: see the failure handling in Launch().
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

#include <dlssnr/NrPassBorrowTables.h>      // the runtime builds whose kernels and weights have been audited

#include "NrFfnV2.h"

#pragma comment(lib, "bcrypt.lib")

namespace NrFfnV2 {
namespace {
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

constexpr unsigned kCreateModule = 0xad1a677d, kCreateFunction = 0xe2436e22, kLaunch = 0x24973538,
                   kDestroyModule = 0x41c65285, kDestroyFunction = 0xdf295ea6;
constexpr uint64_t kWqBytes = 2359296;      // either weight matrix in the kernels' layout
constexpr unsigned kAqUnit = 576, kWqHalf = 2304;
constexpr unsigned kMaxShared = 101376;     // the driver drops the device on a larger request instead of refusing the launch
constexpr uint64_t kMaxTokens = 1u << 15;   // four times an 8K frame's; our buffers grow with the token count
constexpr size_t kMaxSessions = 8;          // frame sizes per device whose buffers are held (about 50 MB each at 4K), those set aside included
// Kernels of the runtime we recognise by name.
enum Kind : unsigned { ExpandPublish, ExpandChained, Contract, Clear, Other };
// Position of an evaluation's first feed-forward expand, counted in launches from its cc_cb_clear, in every audited
// layout (OptiScaler/dlssnr/NrPassBorrowTables.h): Model A 58, Model B 20.
constexpr unsigned kFirstExpand[2] = {58, 20};
constexpr const wchar_t* kModelFolder[2] = {L"modelA", L"modelB"};
constexpr const char* kModelName[2] = {"Model A", "Model B"};

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
Buffer Allocate(ID3D12Device* device, uint64_t bytes) {
    bytes = (std::max<uint64_t>(bytes, 256) + 255) / 256 * 256;
    return {CreateBuffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS), bytes};
}

struct BlockWeights {
    uint64_t w1 = 0, w2 = 0, gain = 0;
    float tx = 1, out1 = 1, th = 1, out2 = 1;
};
struct ModelWeights {
    bool ready = false;
    std::string failure;
    Buffer weights;
    uint64_t trained = 0;                       // where the second set (ffn_v2_trained.bin) starts in `weights`; 0: there is none
    bool trainedIsFirst = false;                // ffn_v2.bin is not there: the buffer starts with ffn_v2_trained.bin
    std::array<uint64_t, 14> setAt{};            // the same for a numbered set (ffn_v2_set<n>.bin)
    std::vector<ComPtr<IUnknown>> parked;       // what an upload that did not finish may still be using; freed with the device
    std::array<BlockWeights, 8> blocks{};
    unsigned smallSplit = 0, smallMaxTokens = 0, smallMinSet = 0;
    bool smallDirect = false;
    std::string smallFile;
    NVDX_ObjectHandle smallModule = nullptr, partial = nullptr, reduce = nullptr, direct = nullptr;
};
// Kernels and weights of one device. They stay until ReleaseDevice(), because recorded command lists refer to them.
struct Device {
    bool kernels = false;
    std::string failure;
    NVDX_ObjectHandle module = nullptr, pack = nullptr, expand = nullptr, contract = nullptr;
    std::array<ModelWeights, 2> models;
};
// Our buffers for one token count. Each block has its own activations and flags, so nothing of ours is ever
// rewritten while an earlier block's kernels may still be reading it.
struct Session {
    struct Block { Buffer xq, hq, partial; };
    std::array<Block, 8> blocks;
    Buffer flags;                               // per block: [pack: 8 per tile][expand: 64 per tile] ints
    uint64_t used = 0;                          // State::evaluations when an evaluation last recorded with these buffers
    uint64_t FlagAddress(unsigned block, unsigned tiles, bool expand) const {
        return flags.Address() + (uint64_t(block) * 72 + (expand ? 8 : 0)) * tiles * 4;
    }
};
struct Pending {                            // an expand we replaced whose contract has not been seen yet
    bool active = false;
    ID3D12Device* device = nullptr;
    uint64_t x = 0, hidden = 0, flags = 0;
    unsigned dims[2]{}, block = 0;
    int generation = 0;
};
// What one command list is in the middle of. An evaluation of the model starts with cc_cb_clear.
struct ListState {
    bool counting = false;      // a cc_cb_clear was seen: `launches` counts the launches recorded after it
    unsigned launches = 0;
    unsigned told = 0;          // RuntimeLaunches(): the runtime's own count, when a layer above left launches out
    unsigned sourceTokens = 0;  // Before optional ViT pooling: retain the original resolution's family selection.
    bool dropping = false;      // a pair failed half-way: nothing more of this evaluation is launched
    bool replacing = false;     // decided at the evaluation's first expand
    int model = -1;
    unsigned nextBlock = 0;
    Pending pending;
};
// refs: the driver can hand the same function handle to two features of the runtime; each destroy drops one reference.
struct Target { ID3D12Device* device; NVDX_ObjectHandle module; Kind kind; unsigned refs; };

struct State {
    std::recursive_mutex mutex;
    bool enabled = false, broken = false, failed = false;
    bool paused = false;                        // Pause(): this evaluation is the runtime's
    Options options;
    std::array<bool, 2> modelFiles{true, true}; // a model's folder is in the asset folder (a package may leave Model B's out)
    std::string status = "off";
    std::string notice;                         // why an evaluation was last left to the runtime; logged when it changes
    uint64_t evaluations = 1, lastReplaced = 0; // cc_cb_clear launches seen (+1); the evaluation that last had a pair replaced
    uint64_t pairs = 0, passedThrough = 0, smallContracts = 0;
    uint64_t testFailAt = 0;                    // development: fail the contract of this pair (1-based), see Enable()
    int generation = 0;                         // stamps our flags; a new value for every feed-forward pair
    unsigned passSet = kDefaultSet;             // UseSet()
    std::wstring runtimePath;                   // the nvngx_dlssnr.dll last checked ...
    bool runtimeCarries[2] = {false, false};    // ... and whether it carries Model A / Model B as the weights were made from
    void (*log)(const char*) = nullptr;
    std::map<NVDX_ObjectHandle, Target> targets;
    std::map<ID3D12Device*, Device> devices;
    std::map<std::pair<ID3D12Device*, uint64_t>, Session> sessions;
    // Buffers of frame sizes no longer in use, kept until the host says the GPU has finished with them (RetireIdleSizes).
    struct Retired { ID3D12Device* device = nullptr; uint64_t ticket = 0; Session session; };
    std::vector<Retired> retired;
    uint64_t retireTicket = 0;
    uint64_t faults = 0;                        // Faults(): replacements that failed, pairs run with other weights than asked for
    std::array<bool, 14> setFile{};             // ffn_v2_set<n>.bin of Model A is in the asset folder (SetComplete)
    bool trainedFile = false;                   // ... and ffn_v2_trained.bin
    std::map<ID3D12GraphicsCommandList*, ListState> lists;
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
};
State& S() { static State* state = new State; return *state; }

void Report(State& s, const std::string& text) {
    const std::string line = "NVFP4 feed-forward: " + text;
    if (s.log) s.log(line.c_str()); else fprintf(stderr, "%s\n", line.c_str());
}
// Neither of the next two may throw: they run where an exception would be taken for a failed launch.
void Say(State& s, const char* prefix, const char* detail = "") noexcept {
    try {
        s.status = std::string(prefix) + detail;
        s.notice.clear();
        Report(s, s.status);
    } catch (...) {
    }
}
// An evaluation is left to the runtime. The reason is logged when it changes, and shown unless another pass is being
// replaced at the moment: passes of different sizes or models alternate within every frame.
void Notice(State& s, const std::string& text) noexcept {
    try {
        if (s.notice != text) {
            s.notice = text;
            Report(s, text);
        }
        if (!s.lastReplaced || s.evaluations - s.lastReplaced > 16) s.status = text;
    } catch (...) {
    }
}

// The weights are in this layer's own files: any NR runtime is accepted. Which model an evaluation is, and that its
// launches are the expected ones, is still checked per evaluation (ChooseModel) and per launch.
bool RuntimeCarries(State&, int) { return GetModuleHandleW(L"nvngx_dlssnr.dll") != nullptr; }

void Barrier(ID3D12GraphicsCommandList* commands) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    commands->ResourceBarrier(1, &barrier);
}

float FloatFromBits(const std::string& text) {
    const uint32_t bits = uint32_t(std::stoul(text, nullptr, 0));
    float value; memcpy(&value, &bits, 4);
    return value;
}

void LoadKernels(Device& d, ID3D12Device* device) {
    if (!d.failure.empty()) throw std::runtime_error(d.failure);
    if (d.kernels) return;
    try {
        auto& s = S();
        if (!s.createModule || !s.createFunction) throw std::runtime_error("the driver's CUDA entry points were not seen");
        const auto cubin = ReadAsset(s.options.assets / "ffn_v2.cubin");
        // An older GPU refuses this module (it is built for the RTX 50 series' NVFP4 tensor-core instruction).
        NvCheck(s.createModule(device, cubin.data(), unsigned(cubin.size()), &d.module), "NVFP4 kernel module (needs an RTX 50 series GPU)");
        NvCheck(s.createFunction(device, d.module, "nr_pack_x", &d.pack), "nr_pack_x");
        NvCheck(s.createFunction(device, d.module, "nr_ffn_expand", &d.expand), "nr_ffn_expand");
        NvCheck(s.createFunction(device, d.module, "nr_ffn_contract", &d.contract), "nr_ffn_contract");
        d.kernels = true;
    } catch (const std::exception& error) {
        auto& s = S();
        if (s.destroyFunction)
            for (auto* function : {&d.pack, &d.expand, &d.contract}) if (*function) { s.destroyFunction(device, *function); *function = nullptr; }
        if (s.destroyModule && d.module) { s.destroyModule(device, d.module); d.module = nullptr; }
        d.failure = error.what();
        throw;
    }
}

void LoadWeights(ModelWeights& m, ID3D12Device* device, const fs::path& root) {
    if (!m.failure.empty()) throw std::runtime_error(m.failure);
    if (m.ready) return;
    try {
        // A package may leave the first set out: ffn_v2_trained.bin then stands in its place.
        std::error_code noFirst;
        m.trainedIsFirst = !fs::exists(root / "ffn_v2.bin", noFirst) && fs::exists(root / "ffn_v2_trained.bin", noFirst);
        auto blob = ReadAsset(root / (m.trainedIsFirst ? "ffn_v2_trained.bin" : "ffn_v2.bin"));
        const size_t first = blob.size();
        std::ifstream manifest(root / "ffn_v2.txt");
        if (!manifest) throw std::runtime_error("missing NVFP4 asset " + (root / "ffn_v2.txt").string());
        std::array<bool, 8> seen{};
        for (std::string line; std::getline(manifest, line);) {
            std::istringstream in(line);
            std::string key; in >> key;
            if (key == "kernels") {             // the shape ffn_v2.cubin was built with
                std::string k1, k2; unsigned stages = 0, steps = 0;
                in >> k1 >> stages >> k2 >> steps;
                if (!in || k1 != "stages" || k2 != "steps") throw std::runtime_error("invalid NVFP4 manifest line: " + line);
                if (stages != S().options.stages || steps != S().options.stepsPerStage)
                    throw std::runtime_error("the NVFP4 kernels were built with another pipeline shape than this program expects");
                continue;
            }
            if (key == "small_direct") {
                std::string a, b;
                in >> a >> m.smallMaxTokens >> b >> m.smallFile;
                if (!in || a != "max_tokens" || b != "file" || !m.smallMaxTokens || m.smallMaxTokens > 4096 || m.smallSplit)
                    throw std::runtime_error("invalid small_direct manifest line: " + line);
                m.smallDirect = true;
                continue;
            }
            if (key == "small_contract") {
                std::string a, b, c;
                in >> a >> m.smallSplit >> b >> m.smallMaxTokens >> c >> m.smallFile;
                if (!in || a != "split" || b != "max_tokens" || c != "file" ||
                    (m.smallSplit != 2 && m.smallSplit != 4) || m.smallDirect || !m.smallMaxTokens || m.smallMaxTokens > 4096)
                    throw std::runtime_error("invalid small_contract manifest line: " + line);
                std::string tail;
                if (in >> tail) {
                    if (tail != "min_set" || !(in >> m.smallMinSet) || m.smallMinSet >= m.setAt.size())
                        throw std::runtime_error("invalid small_contract min_set: " + line);
                }
                continue;
            }
            if (key != "block") continue;
            unsigned index; std::string k1, k2, k3, k4, k5, k6, k7, tx, out1, th, out2; BlockWeights w;
            in >> index >> k1 >> w.w1 >> k2 >> w.w2 >> k3 >> w.gain >> k4 >> tx >> k5 >> out1 >> k6 >> th >> k7 >> out2;
            if (!in || index > 7 || seen[index] || k1 != "w1" || k2 != "w2" || k3 != "gain" || k4 != "tx" || k5 != "out1" || k6 != "th" || k7 != "out2")
                throw std::runtime_error("invalid NVFP4 manifest line: " + line);
            const auto inside = [&](uint64_t offset, uint64_t bytes) { return bytes <= first && offset <= first - bytes && offset % 16 == 0; };
            if (!inside(w.w1, kWqBytes) || !inside(w.w2, kWqBytes) || !inside(w.gain, 2048))
                throw std::runtime_error("NVFP4 manifest points outside ffn_v2.bin");
            w.tx = FloatFromBits(tx); w.out1 = FloatFromBits(out1); w.th = FloatFromBits(th); w.out2 = FloatFromBits(out2);
            m.blocks[index] = w; seen[index] = true;
        }
        for (bool present : seen) if (!present) throw std::runtime_error("NVFP4 manifest does not describe eight blocks");
        // A second set in the same layout (trained codes) goes behind the first, in the same buffer.
        uint64_t second = 0;
        if (std::error_code ignored; !m.trainedIsFirst && fs::exists(root / "ffn_v2_trained.bin", ignored)) {
            const auto more = ReadAsset(root / "ffn_v2_trained.bin");
            if (more.size() != first) throw std::runtime_error("ffn_v2_trained.bin has another size than ffn_v2.bin");
            second = (first + 255) / 256 * 256;
            blob.resize(second);
            blob.insert(blob.end(), more.begin(), more.end());
        }
        // Numbered sets (a pass-in-one model's own pairs) follow, each in the same layout.
        std::array<uint64_t, 14> setAt{};
        for (unsigned n = 2; n < setAt.size(); ++n) {
            const fs::path file = root / ("ffn_v2_set" + std::to_string(n) + ".bin");
            if (std::error_code ignored; !fs::exists(file, ignored)) continue;
            const auto more = ReadAsset(file);
            if (more.size() != first) throw std::runtime_error(file.filename().string() + " has another size than ffn_v2.bin");
            setAt[n] = (blob.size() + 255) / 256 * 256;
            blob.resize(setAt[n]);
            blob.insert(blob.end(), more.begin(), more.end());
        }

        // Upload the weights on a queue of our own and wait: this runs once, while the model's first pass is recorded.
        m.weights = Allocate(device, blob.size());
        auto upload = CreateBuffer(device, blob.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        Check(upload->Map(0, nullptr, &mapped), "NVFP4 upload map");
        memcpy(mapped, blob.data(), blob.size());
        upload->Unmap(0, nullptr);
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> commands; ComPtr<ID3D12Fence> fence;
        // Once the copy is submitted, a failure below must not free what it may still be using.
        struct Park {
            ModelWeights& m; ComPtr<ID3D12Resource>& upload; ComPtr<ID3D12CommandQueue>& queue; ComPtr<ID3D12CommandAllocator>& allocator;
            ComPtr<ID3D12GraphicsCommandList>& commands; bool armed = false;
            ~Park() {
                if (!armed) return;
                try {
                    for (IUnknown* object : {static_cast<IUnknown*>(m.weights.resource.Get()), static_cast<IUnknown*>(upload.Get()),
                                             static_cast<IUnknown*>(queue.Get()), static_cast<IUnknown*>(allocator.Get()),
                                             static_cast<IUnknown*>(commands.Get())})
                        m.parked.emplace_back(object);
                } catch (...) {
                }
            }
        } park{m, upload, queue, allocator, commands};
        Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "NVFP4 upload queue");
        Check(device->CreateCommandAllocator(queueDesc.Type, IID_PPV_ARGS(&allocator)), "NVFP4 upload allocator");
        Check(device->CreateCommandList(0, queueDesc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)), "NVFP4 upload list");
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "NVFP4 upload fence");
        D3D12_RESOURCE_BARRIER transition{};
        transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition = {m.weights.resource.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST};
        commands->ResourceBarrier(1, &transition);
        commands->CopyBufferRegion(m.weights.resource.Get(), 0, upload.Get(), 0, blob.size());
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
        m.trained = second;
        m.setAt = setAt;
        if (m.smallSplit || m.smallDirect) {
            auto& state = S();
            const auto code = ReadAsset(root / m.smallFile);
            NvCheck(state.createModule(device, code.data(), unsigned(code.size()), &m.smallModule), "small FFN module");
            if (m.smallDirect) {
                NvCheck(state.createFunction(device, m.smallModule, "nr_ffn_contract", &m.direct), "small FFN serial contract");
            } else {
                NvCheck(state.createFunction(device, m.smallModule, "nr_ffn_partial", &m.partial), "small FFN partial");
                NvCheck(state.createFunction(device, m.smallModule, "nr_ffn_reduce", &m.reduce), "small FFN reduce");
            }
        }
        m.ready = true;
    } catch (const std::exception& error) {
        m.weights = {};
        auto& state = S();
        if (state.destroyFunction) for (auto* f : {&m.partial, &m.reduce, &m.direct}) if (*f) { state.destroyFunction(device, *f); *f = nullptr; }
        if (state.destroyModule && m.smallModule) { state.destroyModule(device, m.smallModule); m.smallModule = nullptr; }
        m.failure = error.what();
        throw;
    }
}

Session& SessionFor(ID3D12Device* device, uint64_t tokens) {
    auto& session = S().sessions[{device, tokens}];
    if (!session.flags.resource) {
        try {
            const uint64_t tiles = (tokens + 127) / 128;
            for (auto& block : session.blocks) {
                block.xq = Allocate(device, tiles * 16 * 8 * kAqUnit);   // K = 1024: 16 steps of 64
                block.hq = Allocate(device, tiles * 64 * 8 * kAqUnit);   // K = 4096: 64 steps of 64
            }
            session.flags = Allocate(device, 8 * 72 * tiles * 4);        // created zeroed; generations start at 1
        } catch (...) {
            S().sessions.erase({device, tokens});
            throw;
        }
    }
    session.used = S().evaluations;
    return session;
}

void LaunchKernel(ID3D12GraphicsCommandList* commands, NVDX_ObjectHandle function, const void* params, unsigned bytes,
                  unsigned gridX, unsigned blockY, unsigned shared, unsigned gridY = 1) {
    NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};
    launch.hFunction = function;
    launch.gridDim = {gridX, gridY, 1};
    launch.blockDim = {32, blockY, 1};
    launch.pParams = params;
    launch.paramSize = bytes;
    launch.dynSharedMemBytes = shared;
    NvCheck(S().launch(commands, &launch, 1), "NVFP4 kernel launch");
}

#pragma pack(push, 8)
struct PackParams { uint64_t x, xq, producer, flags; uint32_t tokens; float tensorScale; int32_t generation; };
struct ExpandParams { uint64_t xq, w, hq, flags, ready; uint32_t tokens; float outScale, hiddenScale; uint32_t groups, halves; int32_t generation; };
struct ContractParams { uint64_t hq, w, x, gain, out, flags, counter, ready; uint32_t tokens; float outScale; uint32_t groups, halves; int32_t generation; };
struct PartialParams { ContractParams p; uint64_t partial; };
static_assert(sizeof(ContractParams) == 88 && sizeof(PartialParams) == 96);
#pragma pack(pop)

// Where the records of the set asked for start in the weights of a model. A numbered set without a file, or trained
// records asked for and absent, runs the next best as before, and is counted (Faults()).
uint64_t SetAt(State& s, const ModelWeights& weights, const Options& opt) {
    if (s.passSet < weights.setAt.size() && weights.setAt[s.passSet]) return weights.setAt[s.passSet];
    if (s.passSet < weights.setAt.size() || (opt.trained && !weights.trained && !weights.trainedIsFirst)) ++s.faults;
    return opt.trained ? weights.trained : 0;
}

bool SmallContract(const ModelWeights& weights, unsigned tokens, const Options& opt) {
    return (weights.smallSplit || weights.smallDirect) && tokens <= weights.smallMaxTokens &&
        (!weights.smallSplit || !weights.smallMinSet ||
         (S().passSet >= weights.smallMinSet && S().passSet < weights.setAt.size())) &&
        opt.contractGroups == 8 && opt.contractHalves == 2 && opt.stages == 2 && opt.stepsPerStage == 2;
}

struct Original { uint64_t p[8]; unsigned dims[2]; };

unsigned SharedBytes(unsigned groups, unsigned halves) {
    const auto& o = S().options;
    return 64 + o.stages * o.stepsPerStage * (groups * kAqUnit + halves * kWqHalf);
}

// chainedInput: x comes with the runtime's producer flags (o.p[6]); otherwise x is complete once the work recorded
// before us has run, which a barrier ensures. Everything that can fail here fails before the first launch of ours.
int RecordExpand(ID3D12GraphicsCommandList* commands, ID3D12Device* device, const Original& o, unsigned model, unsigned block, bool chainedInput) {
    auto& s = S();
    auto& d = s.devices[device];
    LoadKernels(d, device);
    auto& weights = d.models[model];
    LoadWeights(weights, device, s.options.modelFolders ? s.options.assets / kModelFolder[model] : s.options.assets);
    const uint32_t tokens = o.dims[0] * o.dims[1], tiles = (tokens + 127) / 128;
    auto& session = SessionFor(device, tokens);
    auto& buffers = session.blocks[block];
    const auto& w = weights.blocks[block];
    const auto& opt = s.options;
    // Allocate before the first launch so allocation failure can safely keep the FP8 pair.
    if (SmallContract(weights, tokens, opt) && weights.smallSplit) {
        const uint64_t bytes = uint64_t(weights.smallSplit) * (8 * tiles) * 256 * 64 * sizeof(float);
        if (!buffers.partial.resource) buffers.partial = Allocate(device, bytes);
        else if (buffers.partial.bytes < bytes) throw std::runtime_error("small FFN scratch shape changed in a live session");
    }
    if (++s.generation <= 0) s.generation = 1;
    const int generation = s.generation;
    const bool chained = opt.chained;
    if ((opt.barriers & 1) || !chained || !chainedInput) Barrier(commands);
    const PackParams pack{o.p[0], buffers.xq.Address(), chained && chainedInput ? o.p[6] : 0, session.FlagAddress(block, tiles, false),
                          tokens, w.tx, generation};
    LaunchKernel(commands, d.pack, &pack, sizeof pack, tiles * 8, 16, 0);
    if ((opt.barriers & 2) || !chained) Barrier(commands);
    const uint64_t set = SetAt(s, weights, opt);
    const ExpandParams expand{buffers.xq.Address(), weights.weights.Address() + set + w.w1, buffers.hq.Address(), session.FlagAddress(block, tiles, true),
                              chained ? session.FlagAddress(block, tiles, false) : 0, tokens, w.out1, w.th,
                              opt.expandGroups, opt.expandHalves, generation};
    LaunchKernel(commands, d.expand, &expand, sizeof expand, 4096 / (64 * opt.expandHalves) * tiles * (8 / opt.expandGroups),
                 opt.expandGroups / 2 * opt.expandHalves, SharedBytes(opt.expandGroups, opt.expandHalves));
    if ((opt.barriers & 4) || !chained) Barrier(commands);
    return generation;
}

void RecordContract(ID3D12GraphicsCommandList* commands, ID3D12Device* device, const Original& o, unsigned model, unsigned block, int generation) {
    auto& s = S();
    auto& d = s.devices.at(device);
    const uint32_t tokens = o.dims[0] * o.dims[1], tiles = (tokens + 127) / 128;
    auto& session = SessionFor(device, tokens);
    const auto& weights = d.models[model];
    const auto& w = weights.blocks[block];
    const auto& opt = s.options;
    const uint64_t set = SetAt(s, weights, opt);
    const ContractParams contract{session.blocks[block].hq.Address(), weights.weights.Address() + set + w.w2, o.p[1], weights.weights.Address() + set + w.gain, o.p[2],
                                  o.p[7], o.p[4], opt.chained ? session.FlagAddress(block, tiles, true) : 0, tokens, w.out2,
                                  opt.contractGroups, opt.contractHalves, generation};
    if (SmallContract(weights, tokens, opt) && weights.smallDirect) {
        // Two pipeline stages, four k64 steps per stage. K accumulation order is unchanged.
        constexpr unsigned serialShared = 64 + 2 * 4 * (8 * kAqUnit + 2 * kWqHalf);
        static_assert(serialShared <= kMaxShared);
        LaunchKernel(commands, weights.direct, &contract, sizeof contract, 8 * tiles, 8, serialShared);
        ++s.smallContracts;
    } else if (SmallContract(weights, tokens, opt)) {
        // Split-K adds consumer CTAs; finish expand first so they cannot occupy
        // the remaining SMs while waiting for its producer under contention.
        Barrier(commands);
        const PartialParams partial{contract, session.blocks[block].partial.Address()};
        LaunchKernel(commands, weights.partial, &partial, sizeof partial, 8 * tiles, 8, SharedBytes(8, 2), weights.smallSplit);
        Barrier(commands); // All partial sums are complete before the fixed-order reduction.
        LaunchKernel(commands, weights.reduce, &partial, sizeof partial, 8 * tiles, 8, 0);
        ++s.smallContracts;
    } else {
    LaunchKernel(commands, d.contract, &contract, sizeof contract, 1024 / (64 * opt.contractHalves) * tiles * (8 / opt.contractGroups),
                 opt.contractGroups / 2 * opt.contractHalves, SharedBytes(opt.contractGroups, opt.contractHalves));
    }
    if ((opt.barriers & 8) || !opt.chained) Barrier(commands);
}

NvAPI_Status __cdecl CreateModule(ID3D12Device* device, const void* blob, NvU32 bytes, NVDX_ObjectHandle* out) {
    return S().createModule(device, blob, bytes, out);
}

NvAPI_Status __cdecl CreateFunction(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& s = S();
    const auto result = s.createFunction(device, module, name, out);
    if (result == NVAPI_OK && name && out) {
        Kind kind = Other;
        if (!strcmp(name, "cc_vit_1d_ffn_expand_publish_fp8")) kind = ExpandPublish;
        else if (!strcmp(name, "cc_vit_1d_ffn_expand_chained_fp8")) kind = ExpandChained;
        else if (!strcmp(name, "cc_vit_1d_ffn_contract_chained_fp8")) kind = Contract;
        else if (!strcmp(name, "cc_cb_clear")) kind = Clear;
        std::lock_guard guard(s.mutex);
        const auto found = s.targets.find(*out);
        if (kind == Other) {
            if (found != s.targets.end()) s.targets.erase(found);      // a handle value can be reused after its function was destroyed
        } else if (found != s.targets.end() && found->second.kind == kind && found->second.module == module && found->second.device == device) {
            ++found->second.refs;               // the same kernel handed out again, to another feature
        } else {
            s.targets[*out] = {device, module, kind, 1u};
        }
    }
    return result;
}

// Which model the evaluation that has just reached its first expand belongs to; -1 leaves it to the runtime.
int ChooseModel(State& s, const ListState& list) {
    if (!s.enabled || s.paused) return -1;
    if (!s.options.modelFolders) return 0;
    for (int model = 0; model < 2; ++model)
        if (list.counting && (list.told ? list.told : list.launches) == kFirstExpand[model]) {
            if (!s.modelFiles[model]) return -1;    // no 4-bit files for this model here: its passes are the runtime's own
            if (RuntimeCarries(s, model)) return model;
            Notice(s, "off: no NR runtime is loaded");
            return -1;
        }
    Notice(s, "off: the model's launch sequence is not a known one (" + std::to_string(list.told ? list.told : list.launches) + " launches up to the transformer's feed-forward)");
    return -1;
}

// Something went wrong while a feed-forward launch was being replaced. The state is settled before anything is
// reported, so that a failure to report cannot change what happens to the launch.
NvAPI_Status Fail(State& s, ListState& list, ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count,
                  const char* what) noexcept {
    const bool half = list.pending.active;
    ++s.faults;
    s.enabled = false;
    list.replacing = false;
    list.pending.active = false;
    if (!half) {
        // Nothing of this pair was replaced (our pack alone is harmless): the runtime's own kernels carry on.
        Say(s, "off, the runtime's own kernels are in use: ", what);
        return s.launch(commands, kernels, count);
    }
    // The runtime's contract would wait for an expand that never ran. Launch nothing more of this evaluation.
    s.failed = true;
    const bool boundary = list.counting;
    if (boundary) list.dropping = true; else s.broken = true;      // broken: no evaluation boundary to resume at
    Say(s, "failed half-way through a pair, switched off: ", what);
    return boundary ? NVAPI_OK : NVAPI_ERROR;
}

NvAPI_Status __cdecl Launch(ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    if (s.broken) return NVAPI_ERROR;
    Kind kind = Other;
    const Target* target = nullptr;
    if (kernels && count == 1) {
        const auto found = s.targets.find(kernels->hFunction);
        if (found != s.targets.end()) { target = &found->second; kind = target->kind; }
    }
    if (kind == Clear) {                    // a new evaluation: whatever the list was in the middle of is over
        auto& list = s.lists[commands];
        list = ListState{};
        list.counting = true;
        ++s.evaluations;
        return s.launch(commands, kernels, count);
    }
    auto known = s.lists.find(commands);
    if (known != s.lists.end()) {
        if (known->second.dropping) return NVAPI_OK;
        ++known->second.launches;
    }
    if (kind == Other) return s.launch(commands, kernels, count);
    auto& list = known != s.lists.end() ? known->second : s.lists[commands];
    auto& pending = list.pending;
    try {
        if (kind == Contract && !pending.active) {      // its expand ran as the runtime's own
            if (list.replacing) ++s.passedThrough;
            return s.launch(commands, kernels, count);
        }
        if (kind != Contract) {
            if (pending.active) throw std::runtime_error("a second expand arrived before the contract of the first");
            if (kind == ExpandPublish) {
                list.nextBlock = 0;
                list.model = ChooseModel(s, list);
                list.replacing = list.model >= 0;
                if (list.replacing && kernels->paramSize == 72 && kernels->pParams) {
                    uint32_t dims[2];
                    memcpy(dims, static_cast<const char*>(kernels->pParams) + 64, 8);
                    const uint64_t size = uint64_t(dims[0]) * dims[1];
                    std::string text;
                    if ((list.sourceTokens ? list.sourceTokens : size) < s.options.minTokens) {
                        // Small frames leave most of the GPU idle either way; the three NVFP4 launches then cost as much as
                        // the two they replace, so the result would change for nothing.
                        char megapixels[32];
                        snprintf(megapixels, sizeof megapixels, "%.1f", s.options.minTokens * 3840.0 / 1e6);
                        text = std::string("off at this resolution: only faster from ") + megapixels + " megapixels up";
                    } else if (!s.sessions.count({target->device, size})) {
                        // Buffers of a size stay until ReleaseDevice() (recorded command lists refer to them), so the
                        // number of sizes is capped instead of freeing any while the model runs.
                        size_t held = 0;
                        for (const auto& entry : s.sessions) held += entry.first.first == target->device;
                        for (const auto& entry : s.retired) held += entry.device == target->device;
                        if (held >= kMaxSessions) text = "off at this resolution: too many frame sizes since the session started";
                    }
                    if (!text.empty()) {
                        list.replacing = false;
                        Notice(s, text);
                    }
                }
            }
            if (!list.replacing) return s.launch(commands, kernels, count);
        }
        if (kernels->paramSize != 72 || !kernels->pParams) throw std::runtime_error("the runtime's feed-forward parameter block is not 72 bytes");
        Original o;
        memcpy(o.p, kernels->pParams, 64);
        memcpy(o.dims, static_cast<const char*>(kernels->pParams) + 64, 8);
        const uint64_t tokens = uint64_t(o.dims[0]) * o.dims[1];
        if (!tokens || tokens > kMaxTokens) throw std::runtime_error("unsupported token count " + std::to_string(tokens));
        const unsigned tiles = unsigned((tokens + 127) / 128);
        if (kind != Contract) {
            if (list.nextBlock > 7) throw std::runtime_error("more than eight feed-forward pairs in one transformer");
            if (kernels->gridDim.x != 32 * tiles || kernels->gridDim.y != 1 || kernels->gridDim.z != 1 || kernels->blockDim.x != 32 ||
                kernels->blockDim.y != 4 || kernels->blockDim.z != 1 || kernels->dynSharedMemBytes)
                throw std::runtime_error("the runtime's expand launch has an unexpected shape");
            const int generation = RecordExpand(commands, target->device, o, unsigned(list.model), list.nextBlock, kind == ExpandChained);
            pending = {true, target->device, o.p[0], o.p[2], o.p[7], {o.dims[0], o.dims[1]}, list.nextBlock, generation};
            return NVAPI_OK;
        }
        if (pending.device != target->device || o.p[0] != pending.hidden || o.p[1] != pending.x || o.p[6] != pending.flags ||
            o.dims[0] != pending.dims[0] || o.dims[1] != pending.dims[1] ||
            kernels->gridDim.x != 8 * tiles || kernels->gridDim.y != 1 || kernels->gridDim.z != 4 || kernels->blockDim.x != 32 ||
            kernels->blockDim.y != 4 || kernels->blockDim.z != 1)
            throw std::runtime_error("the contract launch does not match the expand we replaced");
        if (s.testFailAt && s.pairs + 1 == s.testFailAt) throw std::runtime_error("injected test failure (NR_NVFP4_TEST_FAIL)");
        RecordContract(commands, target->device, o, unsigned(list.model), pending.block, pending.generation);
        // The pair is complete: nothing from here on may throw, or the handler below would take it for unreplaced.
        pending.active = false;
        ++s.pairs;
        s.lastReplaced = s.evaluations;
        if (!list.nextBlock++) {
            try {
                s.status = std::string("active, ") + (s.options.modelFolders ? kModelName[list.model] : "the given model") + ", " +
                           std::to_string(tokens) + " tokens" + (s.options.chained ? "" : ", barriers");
            } catch (...) {
            }
        }
        return NVAPI_OK;
    } catch (const std::exception& error) {
        return Fail(s, list, commands, kernels, count, error.what());
    } catch (...) {
        return Fail(s, list, commands, kernels, count, "unknown error");
    }
}

NvAPI_Status __cdecl DestroyFunction(ID3D12Device* device, NVDX_ObjectHandle function) {
    auto& s = S();
    const auto result = s.destroyFunction(device, function);
    if (result == NVAPI_OK) {
        std::lock_guard guard(s.mutex);
        const auto found = s.targets.find(function);
        if (found != s.targets.end() && --found->second.refs == 0) s.targets.erase(found);
    }
    return result;
}

NvAPI_Status __cdecl DestroyModule(ID3D12Device* device, NVDX_ObjectHandle module) {
    // The functions of a module are forgotten one by one as they are destroyed, not here: the driver may share the
    // module with another feature that is still alive, and a reused handle is renamed by its next CreateCuFunction.
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
    const auto valid = [](unsigned groups, unsigned halves) { return (groups == 2 || groups == 4 || groups == 8) && (halves == 1 || halves == 2); };
    if (!valid(options.expandGroups, options.expandHalves) || !valid(options.contractGroups, options.contractHalves) || !options.stages ||
        !options.stepsPerStage || 64 + options.stages * options.stepsPerStage * (8 * kAqUnit + 2 * kWqHalf) > kMaxShared)
        throw std::runtime_error("Unsupported NVFP4 kernel shape");
    // Several k64 steps per copy are contiguous in the buffers only for the 8 x 2 shape.
    if (options.stepsPerStage > 1 && (options.expandGroups != 8 || options.expandHalves != 2 || options.contractGroups != 8 || options.contractHalves != 2))
        throw std::runtime_error("NVFP4 kernels built with several steps per stage need the 8,2,8,2 shape");
    // Our flags cover a whole 128-token tile, the runtime's a tile x 128 output channels.
    if (options.chained && (options.expandGroups != 8 || options.contractGroups != 8 || options.contractHalves != 2))
        throw std::runtime_error("Flag-chained NVFP4 kernels need 8-group CTAs and a 2-half contract");
    if (options.assets != s.options.assets || options.modelFolders != s.options.modelFolders) {
        // Another asset folder: forget earlier load failures. What is already loaded stays in use.
        for (auto& [device, d] : s.devices) {
            if (!d.kernels) d.failure.clear();
            for (auto& m : d.models) if (!m.ready) m.failure.clear();
        }
    }
    // Development: NR_NVFP4_TEST_FAIL=n fails the n-th replaced pair between its two halves, to exercise the path
    // that drops the rest of an evaluation.
    char text[32]{};
    const DWORD length = GetEnvironmentVariableA("NR_NVFP4_TEST_FAIL", text, sizeof text);
    s.testFailAt = length && length < sizeof text ? strtoull(text, nullptr, 10) : 0;
    s.options = options;
    for (int model = 0; model < 2; ++model) {
        std::error_code ignored;
        s.modelFiles[model] = !options.modelFolders || fs::exists(options.assets / kModelFolder[model] / "ffn_v2.txt", ignored);
    }
    const fs::path modelA = options.modelFolders ? options.assets / kModelFolder[0] : options.assets;
    std::error_code absent;
    s.trainedFile = fs::exists(modelA / "ffn_v2_trained.bin", absent);
    for (unsigned n = 0; n < s.setFile.size(); ++n) s.setFile[n] = n >= 2 && fs::exists(modelA / ("ffn_v2_set" + std::to_string(n) + ".bin"), absent);
    if (const wchar_t* v = _wgetenv(L"NR_FFN_DEBUG_BARRIERS")) s.options.barriers = unsigned(wcstoul(v, nullptr, 0)) & 15u;
    if (const wchar_t* v = _wgetenv(L"NR_FFN_DEBUG_CHAINED")) s.options.chained = *v != L'0';
    s.failed = false;
    s.enabled = !s.broken;
    s.status = s.broken ? "failed; restart to use it again" : "on, waiting for the model's next transformer pass";
    s.notice.clear();
    s.lastReplaced = 0;
}

void Pause(bool paused) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.paused = paused;
}

void Disable() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.enabled = false;
    s.status = "off";
    s.notice.clear();
    s.lastReplaced = 0;
}

void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept {
    try {
        auto& s = S();
        std::lock_guard guard(s.mutex);
        s.lists.erase(commands);
    } catch (...) {
    }
}

void RuntimeLaunches(ID3D12GraphicsCommandList* commands, unsigned launches, unsigned sourceTokens) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const auto found = s.lists.find(commands);
    if (found != s.lists.end() && found->second.counting) { found->second.told = launches; found->second.sourceTokens = sourceTokens; }
}

void UseSet(unsigned set) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.passSet = set;
}

bool SetComplete(unsigned set, std::string* missing) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const bool numbered = set != kDefaultSet;
    const bool complete = numbered ? set < s.setFile.size() && s.setFile[set] : !s.options.trained || s.trainedFile;
    if (!complete && missing)
        *missing = std::string(s.options.modelFolders ? "modelA/" : "") + (numbered ? "ffn_v2_set" + std::to_string(set) + ".bin" : std::string("ffn_v2_trained.bin"));
    return complete;
}

unsigned long long Faults() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.faults + s.passedThrough;
}

uint64_t RetireIdleSizes(ID3D12Device* device, unsigned keep, unsigned recent) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    std::vector<std::pair<uint64_t, uint64_t>> sizes;      // (last used, tokens)
    for (const auto& [key, session] : s.sessions) if (key.first == device) sizes.emplace_back(session.used, key.second);
    if (sizes.size() <= keep) return 0;
    std::sort(sizes.rbegin(), sizes.rend());                // most recently used first
    uint64_t ticket = 0;
    for (size_t i = keep; i < sizes.size(); ++i) {
        if (sizes[i].first + recent >= s.evaluations) continue;
        const auto found = s.sessions.find({device, sizes[i].second});
        if (!ticket) ticket = ++s.retireTicket;
        s.retired.push_back({device, ticket, std::move(found->second)});
        s.sessions.erase(found);
    }
    return ticket;
}

void FreeRetired(ID3D12Device* device, uint64_t ticket) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    std::erase_if(s.retired, [&](const State::Retired& r) { return r.device == device && r.ticket <= ticket; });
}

void SkipPair(ID3D12GraphicsCommandList* commands) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const auto found = s.lists.find(commands);
    if (found != s.lists.end() && !found->second.pending.active) ++found->second.nextBlock;
}

void ReleaseDevice(ID3D12Device* device) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const auto found = s.devices.find(device);
    if (found != s.devices.end()) {
        auto& d = found->second;
        // On a lost device these calls fail, which is fine: the objects die with it.
        if (d.module && (!s.destroyFunction || !s.destroyModule))
            Report(s, "cannot destroy the NVFP4 kernels: the driver's destroy entry points were never asked for");
        if (s.destroyFunction)
            for (auto function : {d.pack, d.expand, d.contract}) if (function) s.destroyFunction(device, function);
        if (s.destroyModule && d.module) s.destroyModule(device, d.module);
        for (auto& m : d.models) {
            if (s.destroyFunction) for (auto f : {m.partial, m.reduce, m.direct}) if (f) s.destroyFunction(device, f);
            if (s.destroyModule && m.smallModule) s.destroyModule(device, m.smallModule);
        }
        s.devices.erase(found);
    }
    for (auto i = s.sessions.begin(); i != s.sessions.end();) i = i->first.first == device ? s.sessions.erase(i) : std::next(i);
    std::erase_if(s.retired, [&](const State::Retired& r) { return r.device == device; });
    for (auto i = s.targets.begin(); i != s.targets.end();) i = i->second.device == device ? s.targets.erase(i) : std::next(i);
    s.lists.clear();                        // command lists of the device go with it; their addresses may be reused
}

std::string Status(bool counters) {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    std::string status = s.status;
    if (s.enabled && s.options.trained) {
        bool loaded = false, found = false;
        for (const auto& [device, d] : s.devices)
            for (const auto& m : d.models) if (m.ready) { loaded = true; found = found || m.trained != 0 || m.trainedIsFirst; }
        if (loaded) status += found ? ", trained weights" : ", trained weights not found (the first set is in use)";
    }
    if (s.smallContracts) status += ", small-map contracts " + std::to_string(s.smallContracts);
    if (!counters) return status;
    return status + " | feed-forward pairs replaced: " + std::to_string(s.pairs) + " | contracts left to the runtime: " + std::to_string(s.passedThrough);
}

bool Failed() {
    auto& s = S();
    std::lock_guard guard(s.mutex);
    return s.failed || s.broken;
}
}   // namespace NrFfnV2
