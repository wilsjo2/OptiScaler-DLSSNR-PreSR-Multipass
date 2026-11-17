// Pass borrowing for multi-pass DLSS-NR: see NrPassBorrow.h. Self-contained (no precompiled header).
#include "NrPassBorrow.h"
#include "NrPassBorrowTest.h"
#include "NrPassBorrowTables.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <d3d12.h>
#include <dxgi.h> // nvapi.h needs IDXGISwapChain
#include <intrin.h>
#include <wrl/client.h>
#include <nvapi.h>
#include <detours/detours.h>
#include "NrPassBorrowCopy.h" // const BYTE NrPassBorrowCopy[]: needs windows.h

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

namespace NrPassBorrow
{
namespace
{
namespace Tb = Tables;
using Microsoft::WRL::ComPtr;

constexpr uint32_t kCreateFunction = 0xe2436e22, kDestroyFunction = 0xdf295ea6, kDestroyModule = 0x41c65285,
                   kLaunch = 0x24973538;
constexpr unsigned kSlots = 4, kCuts = 7, kMaxEvaluations = 16, kMaxLaunches = 1024, kMaxParams = 4096;
constexpr uint64_t kRetireFrames = 8, kMaxBytes = 1ull << 30;
constexpr uint16_t kUnknown = 0xffff;
constexpr size_t kVariantCount = std::size(Tb::kVariants);
static_assert(kVariantCount <= 32);

using CreateFn = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle, const char*, NVDX_ObjectHandle*);
using DestroyFn = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle);
using LaunchFn = NvAPI_Status(__cdecl*)(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
using QueryFn = void*(__stdcall*) (uint32_t);

thread_local bool tArmed = false;

// ---- logging: deduplicated, capped per Reset, delivered after every lock is released ----
std::atomic<LogFn> gLog { nullptr };
struct LogState
{
    std::mutex mutex;
    std::set<std::string> seen;
};
LogState& Logs()
{
    static LogState* logs = new LogState;
    return *logs;
}
struct Pending
{
    int level;
    std::string text;
};
thread_local std::vector<Pending> tPending;
void Log(int level, const std::string& text)
{
    try
    {
        auto& logs = Logs();
        std::lock_guard lock(logs.mutex);
        if (logs.seen.size() >= 64 || !logs.seen.insert(text).second)
            return;
        tPending.push_back({ level, "NrPassBorrow: " + text });
    }
    catch (...)
    {
    }
}
struct FlushOnExit // declare before any lock_guard so the log callback runs unlocked
{
    ~FlushOnExit()
    {
        if (tPending.empty())
            return;
        auto pending = std::move(tPending);
        tPending.clear();
        if (const auto fn = gLog.load())
            for (const auto& entry : pending)
                fn(entry.level, entry.text.c_str());
    }
};

// ---- kernel names from the tables, interned to small ids ----
struct Interned
{
    std::vector<const char*> names; // sorted, unique
    std::vector<uint16_t> ids[kVariantCount];
};
const Interned& Intern()
{
    static const Interned interned = []
    {
        Interned result;
        const auto less = [](const char* a, const char* b) { return std::strcmp(a, b) < 0; };
        for (const auto& variant : Tb::kVariants)
            for (unsigned i = 0; i < variant.count; ++i)
                result.names.push_back(variant.kernels[i].name);
        std::sort(result.names.begin(), result.names.end(), less);
        result.names.erase(std::unique(result.names.begin(), result.names.end(),
                                       [](const char* a, const char* b) { return !std::strcmp(a, b); }),
                           result.names.end());
        for (size_t v = 0; v < kVariantCount; ++v)
            for (unsigned i = 0; i < Tb::kVariants[v].count; ++i)
            {
                const auto found = std::lower_bound(result.names.begin(), result.names.end(),
                                                    Tb::kVariants[v].kernels[i].name, less);
                result.ids[v].push_back(static_cast<uint16_t>(found - result.names.begin()));
            }
        return result;
    }();
    return interned;
}
uint16_t NameId(const char* name)
{
    if (!name)
        return kUnknown;
    const auto& names = Intern().names;
    const auto found = std::lower_bound(names.begin(), names.end(), name,
                                        [](const char* a, const char* b) { return std::strcmp(a, b) < 0; });
    return found != names.end() && !std::strcmp(*found, name) ? static_cast<uint16_t>(found - names.begin())
                                                              : kUnknown;
}
const char* NameOf(uint16_t id) { return id < Intern().names.size() ? Intern().names[id] : "<unknown kernel>"; }

// ---- NvAPI wrappers: one slot per distinct original, so every caller keeps reaching its own original ----
struct Hooks
{
    std::mutex mutex;
    std::atomic<CreateFn> create[kSlots] {};
    std::atomic<DestroyFn> destroyFunction[kSlots] {};
    std::atomic<DestroyFn> destroyModule[kSlots] {};
    std::atomic<LaunchFn> launch[kSlots] {};
    std::atomic<bool> launchWrapped { false }, createWrapped { false };
    QueryFn query = nullptr; // standalone Detours trampoline
    bool detoured = false;
    std::atomic<InnerLayerFn> inner { nullptr }; // standalone: a layer between this module and the driver
};
Hooks& GetHooks()
{
    static Hooks* hooks = new Hooks; // never destroyed: wrappers may run during process exit
    return *hooks;
}

struct Function
{
    NVDX_ObjectHandle module;
    HMODULE creator;
    uint16_t id;
    uint32_t refs = 1;  // the driver can hand the same handle to two features: destroy only drops one reference
    std::string name;   // kept for diagnostics when a kernel is not in the tables
};
struct Names
{
    std::mutex mutex;
    std::unordered_map<NVDX_ObjectHandle, Function> functions;
    std::unordered_map<NVDX_ObjectHandle, std::string> destroyed; // handles whose last reference was destroyed
    uint64_t creates = 0;
};
Names& GetNames()
{
    static Names* names = new Names;
    return *names;
}

// ---- runtime allowlist: SHA-256 of the module that created the NR kernels, hashed once on a background thread ----
enum class GateState : uint8_t
{
    Checking,
    NotAudited,
    Audited,
};
struct GateResult
{
    GateState state = GateState::Checking;
    Tb::Runtime runtime = Tb::Runtime::Unknown;
    const char* name = "";
};
struct RuntimeCheck
{
    std::wstring path;
    std::atomic<GateState> state { GateState::Checking };
    Tb::Runtime runtime = Tb::Runtime::Unknown;
    const char* name = "";
};
struct Runtimes
{
    std::mutex mutex;
    std::map<HMODULE, std::shared_ptr<RuntimeCheck>> modules;
    std::atomic<Test::GateFn> gate { nullptr };
    std::atomic<void*> gateUser { nullptr };
};
Runtimes& GetRuntimes()
{
    static Runtimes* runtimes = new Runtimes;
    return *runtimes;
}

std::wstring ModulePath(HMODULE module)
{
    std::wstring path(32768, L'\0');
    const DWORD length = module ? GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size())) : 0;
    path.resize(length && length < path.size() ? length : 0);
    return path;
}
HMODULE ModuleFrom(void* address)
{
    HMODULE module = nullptr;
    if (!address || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                        static_cast<LPCWSTR>(address), &module))
        module = GetModuleHandleW(L"nvngx_dlssnr.dll");
    return module;
}
bool Sha256File(const std::wstring& path, char (&hex)[65])
{
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return false;
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    bool ok = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0 &&
              BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0;
    std::vector<unsigned char> buffer(1 << 20);
    for (DWORD read = 0; ok;)
    {
        if (!ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr))
            ok = false;
        else if (!read)
            break;
        else
            ok = BCryptHashData(hash, buffer.data(), read, 0) >= 0;
    }
    unsigned char digest[32] {};
    ok = ok && BCryptFinishHash(hash, digest, sizeof digest, 0) >= 0;
    if (hash)
        BCryptDestroyHash(hash);
    if (algorithm)
        BCryptCloseAlgorithmProvider(algorithm, 0);
    CloseHandle(file);
    for (unsigned i = 0; ok && i < 32; ++i)
        std::snprintf(hex + i * 2, 3, "%02x", digest[i]);
    return ok;
}
void HashRuntime(std::shared_ptr<RuntimeCheck> check)
{
    FlushOnExit flush;
    char hex[65] {};
    bool hashed = false;
    try
    {
        hashed = Sha256File(check->path, hex);
    }
    catch (...)
    {
    }
    for (const auto& audited : Tb::kAudited)
        if (hashed && !std::strcmp(hex, audited.sha256))
        {
            check->runtime = audited.runtime;
            check->name = audited.name;
            check->state.store(GateState::Audited, std::memory_order_release);
            Log(0, std::string("runtime ") + audited.name + " is audited for pass borrowing");
            return;
        }
    check->state.store(GateState::NotAudited, std::memory_order_release);
    Log(1, hashed ? std::string("runtime not audited for pass borrowing, SHA-256 ") + hex
                  : std::string("cannot hash the NR runtime; pass borrowing is off"));
}
// Called when a module creates a kernel named in the tables.
void RegisterRuntime(HMODULE module)
{
    if (!module || GetRuntimes().gate.load())
        return;
    auto path = ModulePath(module);
    auto& runtimes = GetRuntimes();
    std::lock_guard lock(runtimes.mutex);
    auto& check = runtimes.modules[module];
    if (check && check->path == path)
        return;
    check = std::make_shared<RuntimeCheck>();
    check->path = std::move(path);
    try
    {
        std::thread(HashRuntime, check).detach();
    }
    catch (...)
    {
        check->state = GateState::NotAudited;
    }
}
GateResult Gate(HMODULE module)
{
    auto& runtimes = GetRuntimes();
    if (const auto gate = runtimes.gate.load())
        switch (gate(runtimes.gateUser.load()))
        {
        case Test::Gate::Checking:
            return {};
        case Test::Gate::NotAudited:
            return { GateState::NotAudited };
        case Test::Gate::R310_8:
            return { GateState::Audited, Tb::Runtime::R310_8, "310.8" };
        case Test::Gate::R310_9_16:
            return { GateState::Audited, Tb::Runtime::R310_9_16, "310.9.16" };
        }
    std::lock_guard lock(runtimes.mutex);
    const auto found = runtimes.modules.find(module);
    if (!module || found == runtimes.modules.end())
        return { GateState::NotAudited };
    const auto state = found->second->state.load(std::memory_order_acquire);
    if (state != GateState::Audited)
        return { state };
    return { state, found->second->runtime, found->second->name };
}

// ---- recorded evaluations ----
struct Record
{
    struct Launch
    {
        uint16_t id;
        uint32_t paramSize, offset, shared;
        NVAPI_DIM3 grid, block;
        HMODULE creator;
    };
    std::vector<Launch> launches;
    std::vector<uint8_t> params;
    void Clear()
    {
        launches.clear();
        params.clear();
    }
    template <class T> bool Read(unsigned launch, unsigned offset, T& value) const
    {
        if (launch >= launches.size() || offset + sizeof(T) > launches[launch].paramSize)
            return false;
        std::memcpy(&value, params.data() + launches[launch].offset + offset, sizeof(T));
        return true;
    }
    uint64_t Q(unsigned launch, unsigned offset) const
    {
        uint64_t value = 0;
        return Read(launch, offset, value) ? value : 0;
    }
    uint32_t U(unsigned launch, unsigned offset) const
    {
        uint32_t value = 0;
        return Read(launch, offset, value) ? value : 0;
    }
    template <class Fn> void Qwords(unsigned launch, Fn&& fn) const
    {
        for (unsigned offset = 0; offset + 8 <= launches[launch].paramSize; offset += 8)
            fn(offset, Q(launch, offset));
    }
};
bool SameDim(const NVAPI_DIM3& a, const NVAPI_DIM3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

struct Snapshot
{
    ComPtr<ID3D12Resource> buffer;
    uint64_t capacity = 0, bytes = 0, frame = 0, source = 0;
    unsigned evaluation = 0;
    bool valid = false;
};
// A GPU object no later recording uses. Recorded command lists may still refer to it.
struct Retired
{
    ComPtr<ID3D12Object> object;
    uint64_t frame;
};
enum class Mode : uint8_t
{
    Off,     // not armed: forward unchanged
    Record,  // forward, record, compare with the template, take snapshots
    Borrow,  // forward before the cut, skip it, copy the snapshot in at the consumer
    Dropped, // failed after skipping: drop every remaining launch
};

struct State
{
    std::mutex mutex;
    // Plan of the current frame.
    bool active = false;
    ID3D12GraphicsCommandList* list = nullptr;
    uint64_t frame = 0;
    unsigned evaluations = 0, next = 0;
    Cut cuts[kMaxEvaluations] {};
    uint8_t source[kMaxEvaluations] {};    // borrowing evaluation -> evaluation that snapshots its cut
    uint8_t snapshots[kMaxEvaluations] {}; // evaluation -> bit mask of cuts it snapshots
    // Template: evaluation 0 of the current frame.
    Record templ;
    bool templateValid = false;
    uint64_t templateFrame = 0;
    int variant = -1;
    // Current evaluation.
    bool inEvaluation = false, matches = true, skipping = false, borrowed = false;
    unsigned index = 0;
    Mode mode = Mode::Off;
    Cut cut = Cut::None;
    const Tb::CutSpec* spec = nullptr;
    uint64_t bytes = 0, arena = 0, arenaEnd = 0;
    uint32_t candidates = 0;
    uint8_t pendingCheck = 0;
    Record current;
    std::vector<uint64_t> before, skippedCounters;
    // Results.
    bool latched = false;
    std::string latchReason, frameReason, status = "Waiting for a borrowing frame";
    uint8_t borrowedMask = 0;
    // GPU.
    Snapshot snapshot[kCuts];
    std::vector<Retired> retired;
    bool hostRetires = false; // TakeRetired() has been called: the host frees retired objects after GPU completion
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    Test::CopyFn copy = nullptr;
    void* copyUser = nullptr;
};
State& GetState()
{
    static State* state = new State; // never destroyed: the GPU may still use snapshot buffers at exit
    return *state;
}

const Tb::CutSpec* FindSpec(int variant, Cut cut)
{
    if (variant < 0 || variant >= static_cast<int>(kVariantCount))
        return nullptr;
    const auto& v = Tb::kVariants[variant];
    for (unsigned i = 0; i < v.cutCount; ++i)
        if (v.cuts[i].cut == static_cast<uint8_t>(cut))
            return &v.cuts[i];
    return nullptr;
}
int UniqueVariant(uint32_t candidates)
{
    return std::popcount(candidates) == 1 ? std::countr_zero(candidates) : -1;
}
std::string VariantName(int variant)
{
    if (variant < 0)
        return "unknown model";
    const auto& v = Tb::kVariants[variant];
    std::string name = std::string("Model ") + v.model;
    if (std::strcmp(v.layout, "Default"))
        name += std::string(", ") + v.layout;
    return name;
}
void Refuse(State& s, const std::string& reason)
{
    if (s.frameReason.empty())
        s.frameReason = reason;
    Log(1, "evaluation " + std::to_string(s.index) + ": " + reason);
}
NvAPI_Status Drop(State& s, const std::string& reason)
{
    s.mode = Mode::Dropped;
    if (!s.latched)
    {
        s.latched = true;
        s.latchReason = reason;
        Log(2, "evaluation " + std::to_string(s.index) + " dropped after skipping started: " + reason +
                   "; pass borrowing is off until reset");
    }
    return NVAPI_OK;
}

// Handoff bytes of `spec` from a record that holds at least the launch `spec.dims`; 0 when invalid.
uint64_t HandoffBytes(const Record& record, const Tb::CutSpec& spec)
{
    const uint64_t rows = record.U(spec.dims, spec.rows), cols = record.U(spec.dims, spec.cols);
    if (!rows || !cols || rows > 65536 || cols > 65536)
        return 0;
    const uint64_t bytes = rows * cols * spec.scale;
    return bytes % 16 || bytes > kMaxBytes ? 0 : bytes;
}
bool ProducerExtentMatches(const Record& record, const Tb::CutSpec& spec)
{
    return record.U(spec.last, spec.checkRows) == record.U(spec.dims, spec.rows) &&
           record.U(spec.last, spec.checkCols) == record.U(spec.dims, spec.cols);
}

// ---- the handoff copy ----
void UavBarrier(ID3D12GraphicsCommandList* list)
{
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    list->ResourceBarrier(1, &barrier);
}
void RetireSnapshot(State& s, Snapshot& snapshot)
{
    if (snapshot.buffer)
        s.retired.push_back({ ComPtr<ID3D12Object>(std::move(snapshot.buffer)), s.frame });
    snapshot.buffer.Reset();
    snapshot.capacity = 0;
    snapshot.valid = false;
}
bool PrepareGpu(State& s, ID3D12GraphicsCommandList* list)
{
    ComPtr<ID3D12Device> device;
    if (FAILED(list->GetDevice(IID_PPV_ARGS(&device))))
        return false;
    if (device == s.device && s.pipeline)
        return true;
    for (auto& snapshot : s.snapshot)
        RetireSnapshot(s, snapshot);
    // Recorded copies on the old device still name its root signature and pipeline.
    if (s.root)
        s.retired.push_back({ ComPtr<ID3D12Object>(std::move(s.root)), s.frame });
    if (s.pipeline)
        s.retired.push_back({ ComPtr<ID3D12Object>(std::move(s.pipeline)), s.frame });
    s.root.Reset();
    s.pipeline.Reset();
    s.device = device;
    // The root signature is embedded in the shader: 0 = UAV u0 (source), 1 = UAV u1 (destination), 2 = 2 constants.
    if (FAILED(device->CreateRootSignature(0, NrPassBorrowCopy, sizeof NrPassBorrowCopy, IID_PPV_ARGS(&s.root))))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC description {};
    description.pRootSignature = s.root.Get();
    description.CS = { NrPassBorrowCopy, sizeof NrPassBorrowCopy };
    if (FAILED(device->CreateComputePipelineState(&description, IID_PPV_ARGS(&s.pipeline))))
    {
        s.root.Reset();
        return false;
    }
    return true;
}
bool RecordCopy(State& s, const Test::Copy& copy)
{
    if (s.copy)
        return s.copy(s.list, copy, s.copyUser);
    auto& snapshot = s.snapshot[static_cast<unsigned>(copy.cut)];
    if (!s.pipeline || !copy.arena || !copy.bytes || copy.bytes % 16 || copy.arena % 16)
        return false;
    if (copy.toSnapshot && snapshot.capacity < copy.bytes)
    {
        RetireSnapshot(s, snapshot);
        D3D12_HEAP_PROPERTIES heap {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC description {};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = (copy.bytes + 0xfffff) & ~0xfffffull;
        description.Height = 1;
        description.DepthOrArraySize = 1;
        description.MipLevels = 1;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        description.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(s.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                     IID_PPV_ARGS(&snapshot.buffer))))
        {
            snapshot.buffer.Reset();
            return false;
        }
        snapshot.capacity = description.Width;
    }
    if (!snapshot.buffer || snapshot.capacity < copy.bytes)
        return false;
    const auto held = snapshot.buffer->GetGPUVirtualAddress();
    const UINT vectors = static_cast<UINT>(copy.bytes / 16), groups = (vectors + 255) / 256;
    const UINT x = (std::min)(groups, 65535u), y = (groups + x - 1) / x;
    const UINT constants[] { vectors, x * 256 };
    UavBarrier(s.list);
    s.list->SetComputeRootSignature(s.root.Get());
    s.list->SetPipelineState(s.pipeline.Get());
    s.list->SetComputeRootUnorderedAccessView(0, copy.toSnapshot ? copy.arena : held);
    s.list->SetComputeRootUnorderedAccessView(1, copy.toSnapshot ? held : copy.arena);
    s.list->SetComputeRoot32BitConstants(2, 2, constants, 0);
    s.list->Dispatch(x, y, 1);
    UavBarrier(s.list);
    return true;
}

// ---- per launch ----
void TakeSnapshots(State& s, unsigned k)
{
    const int variant = UniqueVariant(s.candidates);
    if (!s.snapshots[s.index] || !s.matches || variant < 0 || (s.index && (!s.templateValid || variant != s.variant)))
        return;
    for (unsigned c = 1; c < kCuts; ++c)
    {
        const auto* spec = (s.snapshots[s.index] >> c) & 1 ? FindSpec(variant, static_cast<Cut>(c)) : nullptr;
        if (!spec || spec->last != k)
            continue;
        auto& snapshot = s.snapshot[c];
        snapshot.valid = false;
        const auto gate = Gate(s.current.launches[k].creator);
        if (gate.state != GateState::Audited || gate.runtime != Tb::kVariants[variant].runtime)
            continue;
        const uint64_t bytes = HandoffBytes(s.current, *spec), output = s.current.Q(k, spec->output);
        if (!bytes || !output || output % 16 || !ProducerExtentMatches(s.current, *spec))
        {
            Refuse(s, std::string("Off: unexpected ") + CutName(static_cast<Cut>(c)) + " handoff layout");
            continue;
        }
        if (!RecordCopy(s, { true, static_cast<Cut>(c), output, bytes, s.index }))
        {
            Refuse(s, "Off: snapshot copy failed");
            continue;
        }
        snapshot.valid = true;
        snapshot.bytes = bytes;
        snapshot.frame = s.frame;
        snapshot.source = output;
        snapshot.evaluation = s.index;
        s.pendingCheck |= static_cast<uint8_t>(1u << c);
    }
}
// The consumer of a snapshotted cut must read exactly what the producer wrote.
void CheckSnapshotConsumers(State& s, unsigned k)
{
    const int variant = UniqueVariant(s.candidates);
    for (unsigned c = 1; c < kCuts && s.pendingCheck; ++c)
    {
        const auto* spec = (s.pendingCheck >> c) & 1 ? FindSpec(variant, static_cast<Cut>(c)) : nullptr;
        if (!spec || spec->last + 1u != k)
            continue;
        s.pendingCheck &= static_cast<uint8_t>(~(1u << c));
        if (s.current.Q(k, 0) != s.snapshot[c].source)
        {
            s.snapshot[c].valid = false;
            Refuse(s, std::string("Off: ") + CutName(static_cast<Cut>(c)) + " consumer does not read the producer");
        }
    }
}
bool InSorted(const std::vector<uint64_t>& values, uint64_t value)
{
    return std::binary_search(values.begin(), values.end(), value);
}
void SortUnique(std::vector<uint64_t>& values)
{
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}
bool ReferencesSkippedCounter(const State& s, unsigned k)
{
    bool found = false;
    s.current.Qwords(k, [&](unsigned, uint64_t value) { found = found || InSorted(s.skippedCounters, value); });
    return found;
}
// Start skipping at the first launch of the cut: everything is checked before anything is skipped.
std::string CanStartSkipping(State& s)
{
    const auto& spec = *s.spec;
    const auto& snapshot = s.snapshot[static_cast<unsigned>(s.cut)];
    const uint64_t bytes = HandoffBytes(s.current, spec);
    if (!snapshot.valid || snapshot.frame != s.frame || snapshot.evaluation != s.source[s.index])
        return std::string("Off: no snapshot of ") + CutName(s.cut);
    if (!bytes || bytes != snapshot.bytes || bytes != HandoffBytes(s.templ, spec))
        return "Off: pass sizes differ";
    const auto gate = Gate(s.current.launches[spec.first].creator);
    if (gate.state != GateState::Audited || gate.runtime != Tb::kVariants[s.variant].runtime)
        return gate.state == GateState::Checking ? "Off: checking runtime" : "Off: runtime not audited";
    const auto clear = Tb::kVariants[s.variant].clear;
    s.arena = s.current.Q(clear, 0);
    s.arenaEnd = s.arena + uint64_t(s.current.U(clear, 8)) * 4;
    if (!s.arena || s.arenaEnd <= s.arena)
        return "Off: unexpected counter arena";
    s.before.clear();
    for (unsigned i = 0; i < spec.first; ++i)
        if (i != clear)
            s.current.Qwords(i, [&](unsigned, uint64_t value) { s.before.push_back(value); });
    SortUnique(s.before);
    s.skippedCounters.clear();
    s.bytes = bytes;
    return {};
}
void CollectCounters(State& s, unsigned k)
{
    s.current.Qwords(k,
                     [&](unsigned, uint64_t value)
                     {
                         if (value >= s.arena && value < s.arenaEnd)
                             s.skippedCounters.push_back(value);
                     });
}
// At the consumer: the contract that makes the copy equivalent to the skipped producer.
std::string ConsumerContract(State& s, unsigned k)
{
    const auto& spec = *s.spec;
    SortUnique(s.skippedCounters);
    const uint64_t p0 = s.current.Q(k, 0);
    if (!p0 || p0 % 16 || s.current.Q(spec.last, spec.output) != p0)
        return "producer output is not the consumer input";
    if (!ProducerExtentMatches(s.current, spec))
        return "producer extent differs";
    std::string failure;
    s.current.Qwords(k,
                     [&](unsigned offset, uint64_t value)
                     {
                         if (offset && value >= p0 && value - p0 < s.bytes && InSorted(s.before, value))
                             failure = "handoff region holds a live pointer";
                         if (InSorted(s.skippedCounters, value))
                             failure = "consumer uses a skipped counter";
                         if (value >= s.arena && value < s.arenaEnd && InSorted(s.before, value))
                             failure = "consumer waits on an earlier counter";
                     });
    for (const auto value : s.before)
        if (InSorted(s.skippedCounters, value))
            failure = "a counter is shared across the cut";
    return failure;
}

bool CaptureLaunch(State& s, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count, std::string& why)
{
    Record::Launch launch {};
    launch.id = kUnknown;
    launch.offset = static_cast<uint32_t>(s.current.params.size());
    bool good = count == 1 && kernels && kernels->pParams && kernels->paramSize <= kMaxParams &&
                s.current.launches.size() < kMaxLaunches;
    if (!good)
        why = count != 1 ? "launch call with " + std::to_string(count) + " kernels" : "launch without parameters";
    if (good)
    {
        launch.paramSize = static_cast<uint32_t>(kernels->paramSize);
        launch.grid = kernels->gridDim;
        launch.block = kernels->blockDim;
        launch.shared = kernels->dynSharedMemBytes;
        auto& names = GetNames();
        std::lock_guard lock(names.mutex);
        const auto found = names.functions.find(kernels->hFunction);
        if (found != names.functions.end())
        {
            launch.id = found->second.id;
            launch.creator = found->second.creator;
            if (launch.id == kUnknown)
                why = "kernel '" + found->second.name.substr(0, 60) + "' is not in the borrow tables";
        }
        else if (const auto gone = names.destroyed.find(kernels->hFunction); gone != names.destroyed.end())
            why = "kernel '" + gone->second.substr(0, 60) + "' was launched after it was destroyed";
        else
            why = names.creates ? "a kernel was launched that was not seen being created (" +
                                      std::to_string(names.creates) + " creations seen)"
                                : "NR kernels were not observed being created (restart the NR feature)";
    }
    if (good && launch.id == kUnknown)
        good = false;
    else if (good)
        why.clear();
    if (good)
    {
        const auto* bytes = static_cast<const uint8_t*>(kernels->pParams);
        s.current.params.insert(s.current.params.end(), bytes, bytes + launch.paramSize);
    }
    else
        launch.paramSize = 0;
    s.current.launches.push_back(launch);
    const unsigned k = static_cast<unsigned>(s.current.launches.size() - 1);
    for (uint32_t bits = s.candidates; bits; bits &= bits - 1)
    {
        const int v = std::countr_zero(bits);
        const auto& variant = Tb::kVariants[v];
        if (!good || k >= variant.count || Intern().ids[v][k] != launch.id ||
            variant.kernels[k].paramSize != launch.paramSize)
            s.candidates &= ~(1u << v);
    }
    if (good && s.index && s.templateValid)
    {
        const auto* t = k < s.templ.launches.size() ? &s.templ.launches[k] : nullptr;
        if (!t || t->id != launch.id || t->paramSize != launch.paramSize)
            why = "launch sequence differs";
        else if (!SameDim(t->grid, launch.grid) || !SameDim(t->block, launch.block) || t->shared != launch.shared)
            why = "pass sizes differ";
    }
    else if (good && !s.index && !s.candidates)
        why = "unrecognised NR kernel sequence";
    if (good && !why.empty())
        good = false;
    if (!good && why.empty())
        why = "launch sequence differs";
    return good;
}

NvAPI_Status ArmedLaunch(LaunchFn original, ID3D12GraphicsCommandList* list,
                         const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count)
{
    FlushOnExit flush;
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    if (!s.inEvaluation || list != s.list || s.mode == Mode::Off)
        return original(list, kernels, count);
    if (s.mode == Mode::Dropped)
        return NVAPI_OK;
    std::string why;
    bool same = false;
    unsigned k = 0;
    try
    {
        same = CaptureLaunch(s, kernels, count, why);
        k = static_cast<unsigned>(s.current.launches.size() - 1);
    }
    catch (...)
    {
        if (s.skipping)
            return Drop(s, "out of memory");
        s.mode = Mode::Off;
        return original(list, kernels, count);
    }
    const auto forward = [&]
    {
        const auto result = original(list, kernels, count);
        // The kernel is recorded now: an exception must never reach AfterException, which would launch it again.
        if (result == NVAPI_OK && s.mode != Mode::Dropped)
            try
            {
                CheckSnapshotConsumers(s, k);
                TakeSnapshots(s, k);
            }
            catch (...)
            {
                // Leave the mode alone (Off would bypass the skipped-counter check); later borrows find no snapshot.
                for (auto& snapshot : s.snapshot)
                    snapshot.valid = false;
                s.pendingCheck = 0;
            }
        return result;
    };
    if (s.mode == Mode::Record)
    {
        if (!same && s.matches)
        {
            s.matches = false;
            if (s.snapshots[s.index] || !s.index)
                Refuse(s, "Off: " + why);
        }
        return forward();
    }
    // Borrow.
    const auto& spec = *s.spec;
    if (!s.skipping)
    {
        if (!same)
        {
            s.mode = Mode::Record;
            s.matches = false;
            Refuse(s, "Off: " + why);
            return forward();
        }
        if (k < spec.first)
            return forward();
        auto refusal = CanStartSkipping(s);
        if (!refusal.empty())
        {
            s.mode = Mode::Record;
            Refuse(s, refusal);
            return forward();
        }
        s.skipping = true;
    }
    if (!same)
        return Drop(s, why);
    if (k <= spec.last)
    {
        CollectCounters(s, k);
        return NVAPI_OK;
    }
    if (k == spec.last + 1u)
    {
        auto failure = ConsumerContract(s, k);
        if (!failure.empty())
            return Drop(s, failure);
        if (!RecordCopy(s, { false, s.cut, s.current.Q(k, 0), s.bytes, s.index }))
            return Drop(s, "snapshot copy failed");
        s.borrowed = true;
        return forward();
    }
    if (ReferencesSkippedCounter(s, k))
        return Drop(s, "a later launch uses a skipped counter");
    return forward();
}

// Only reached when building a log or status string throws before the launch was forwarded: never resume launching
// once skipping has started.
NvAPI_Status AfterException(LaunchFn original, ID3D12GraphicsCommandList* list,
                            const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count)
{
    auto& s = GetState();
    {
        std::lock_guard lock(s.mutex);
        if (s.inEvaluation && (s.skipping || s.mode == Mode::Dropped))
        {
            s.mode = Mode::Dropped;
            s.latched = true;
            if (s.latchReason.empty())
                s.latchReason = "internal error";
            return NVAPI_OK;
        }
        s.mode = Mode::Off;
    }
    return original(list, kernels, count);
}
template <unsigned S>
NvAPI_Status __cdecl LaunchSlot(ID3D12GraphicsCommandList* list, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels,
                                NvU32 count)
{
    const auto original = GetHooks().launch[S].load(std::memory_order_acquire);
    if (!tArmed)
        return original(list, kernels, count);
    try
    {
        return ArmedLaunch(original, list, kernels, count);
    }
    catch (...)
    {
    }
    try
    {
        return AfterException(original, list, kernels, count);
    }
    catch (...)
    {
        return NVAPI_ERROR;
    }
}
NvAPI_Status OnCreate(CreateFn original, void* caller, ID3D12Device* device, NVDX_ObjectHandle module,
                      const char* name, NVDX_ObjectHandle* out)
{
    const auto result = original(device, module, name, out);
    if (result != NVAPI_OK || !out || !*out)
        return result;
    FlushOnExit flush;
    try
    {
        const auto id = NameId(name);
        const auto creator = ModuleFrom(caller);
        {
            auto& names = GetNames();
            std::lock_guard lock(names.mutex);
            ++names.creates;
            names.destroyed.erase(*out);
            auto found = names.functions.find(*out);
            if (found != names.functions.end() && found->second.id == id && found->second.creator == creator)
                ++found->second.refs; // the same kernel handed out again (shared by another feature)
            else
                names.functions[*out] = { module, creator, id, 1u, name ? name : "" };
        }
        if (id != kUnknown)
            RegisterRuntime(creator);
    }
    catch (...)
    {
        Log(1, "cannot record an NR kernel name");
    }
    return result;
}
template <unsigned S>
NvAPI_Status __cdecl CreateSlot(ID3D12Device* device, NVDX_ObjectHandle module, const char* name,
                                NVDX_ObjectHandle* out)
{
    return OnCreate(GetHooks().create[S].load(std::memory_order_acquire), _ReturnAddress(), device, module, name,
                    out);
}
template <unsigned S> NvAPI_Status __cdecl DestroyFunctionSlot(ID3D12Device* device, NVDX_ObjectHandle function)
{
    const auto result = GetHooks().destroyFunction[S].load(std::memory_order_acquire)(device, function);
    if (result == NVAPI_OK)
    {
        auto& names = GetNames();
        std::lock_guard lock(names.mutex);
        auto found = names.functions.find(function);
        if (found != names.functions.end() && --found->second.refs == 0)
        {
            if (names.destroyed.size() < 4096)
                names.destroyed[function] = found->second.name;
            names.functions.erase(found);
        }
    }
    return result;
}
template <unsigned S> NvAPI_Status __cdecl DestroyModuleSlot(ID3D12Device* device, NVDX_ObjectHandle module)
{
    const auto result = GetHooks().destroyModule[S].load(std::memory_order_acquire)(device, module);
    // Kernel names are kept when a module is destroyed: the driver may share the module with another live feature, and
    // a reused handle is renamed by its next CreateCuFunction, which every NR creation passes through.
    (void) module;
    return result;
}
constexpr CreateFn kCreateSlots[kSlots] { CreateSlot<0>, CreateSlot<1>, CreateSlot<2>, CreateSlot<3> };
constexpr DestroyFn kDestroyFunctionSlots[kSlots] { DestroyFunctionSlot<0>, DestroyFunctionSlot<1>,
                                                    DestroyFunctionSlot<2>, DestroyFunctionSlot<3> };
constexpr DestroyFn kDestroyModuleSlots[kSlots] { DestroyModuleSlot<0>, DestroyModuleSlot<1>, DestroyModuleSlot<2>,
                                                  DestroyModuleSlot<3> };
constexpr LaunchFn kLaunchSlots[kSlots] { LaunchSlot<0>, LaunchSlot<1>, LaunchSlot<2>, LaunchSlot<3> };

template <class Fn>
void* Wrap(std::atomic<Fn> (&slots)[kSlots], const Fn (&wrappers)[kSlots], void* original, const char* what)
{
    const auto fn = reinterpret_cast<Fn>(original);
    for (const auto wrapper : wrappers)
        if (fn == wrapper)
            return original; // already ours
    auto& hooks = GetHooks();
    std::lock_guard lock(hooks.mutex);
    for (unsigned slot = 0; slot < kSlots; ++slot)
        if (slots[slot].load() == fn)
            return reinterpret_cast<void*>(wrappers[slot]);
    for (unsigned slot = 0; slot < kSlots; ++slot)
        if (!slots[slot].load())
        {
            slots[slot].store(fn, std::memory_order_release);
            return reinterpret_cast<void*>(wrappers[slot]);
        }
    Log(1, std::string("more than four distinct ") + what + " entry points; the extra one is not wrapped");
    return original;
}

void* __stdcall DetouredQueryInterface(uint32_t id)
{
    void* original = GetHooks().query(id);
    if (const auto inner = GetHooks().inner.load())
        original = inner(id, original);
    return WrapInterface(id, original);
}

} // namespace

Cut ChooseCut(bool borrowTransformer, bool borrowBelow, unsigned level)
{
    if (!borrowBelow)
        return borrowTransformer ? Cut::Transformer : Cut::None;
    if (level < 3)
        return Cut::Below2;
    if (level < 6)
        return Cut::Below4;
    if (level < 12)
        return Cut::Below8;
    if (level < 24)
        return Cut::Below16;
    return Cut::Below32;
}

const char* CutName(Cut cut)
{
    switch (cut)
    {
    case Cut::None:
        return "none";
    case Cut::Transformer:
        return "transformer";
    case Cut::Below32:
        return "1/32 and below";
    case Cut::Below16:
        return "1/16 and below";
    case Cut::Below8:
        return "1/8 and below";
    case Cut::Below4:
        return "1/4 and below";
    case Cut::Below2:
        return "1/2 and below";
    }
    return "unknown";
}

void SetLog(LogFn fn) { gLog.store(fn); }

void SetInnerLayer(InnerLayerFn layer) { GetHooks().inner.store(layer); }

bool InstallQueryInterfaceHook(const char** error)
{
    FlushOnExit flush;
    auto& hooks = GetHooks();
    const auto fail = [&](const char* text)
    {
        if (error)
            *error = text;
        Log(2, text);
        return false;
    };
    {
        std::lock_guard lock(hooks.mutex);
        if (hooks.detoured)
            return true;
        const HMODULE nvapi = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!nvapi)
            return fail("cannot load system32 nvapi64.dll");
        hooks.query = reinterpret_cast<QueryFn>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
        if (!hooks.query)
            return fail("nvapi64.dll has no nvapi_QueryInterface");
        LONG result = DetourTransactionBegin();
        if (result == NO_ERROR)
        {
            result = DetourUpdateThread(GetCurrentThread());
            if (result == NO_ERROR)
                result = DetourAttach(reinterpret_cast<PVOID*>(&hooks.query),
                                      reinterpret_cast<PVOID>(&DetouredQueryInterface));
            if (result == NO_ERROR)
                result = DetourTransactionCommit();
            else
                DetourTransactionAbort();
        }
        if (result != NO_ERROR)
            return fail("Detours could not hook nvapi_QueryInterface");
        hooks.detoured = true;
    }
    if (error)
        *error = nullptr;
    return true;
}

void* WrapInterface(uint32_t id, void* original)
{
    if (!original)
        return original;
    FlushOnExit flush;
    auto& hooks = GetHooks();
    switch (id)
    {
    case kCreateFunction:
        hooks.createWrapped = true;
        return Wrap(hooks.create, kCreateSlots, original, "CreateCuFunction");
    case kDestroyFunction:
        return Wrap(hooks.destroyFunction, kDestroyFunctionSlots, original, "DestroyCuFunction");
    case kDestroyModule:
        return Wrap(hooks.destroyModule, kDestroyModuleSlots, original, "DestroyCuModule");
    case kLaunch:
        hooks.launchWrapped = true;
        return Wrap(hooks.launch, kLaunchSlots, original, "LaunchCuKernelChain");
    default:
        return original;
    }
}

unsigned TakeRetired(ID3D12Object** objects, unsigned capacity)
{
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    s.hostRetires = true;
    unsigned count = 0;
    while (objects && count < capacity && !s.retired.empty())
    {
        objects[count++] = s.retired.back().object.Detach();
        s.retired.pop_back();
    }
    return count;
}

void BeginFrame(ID3D12GraphicsCommandList* list, const Cut* cuts, unsigned evaluations)
{
    FlushOnExit flush;
    tArmed = false;
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    try
    {
        ++s.frame;
        // A host that tracks GPU completion collects these (TakeRetired). Otherwise they are kept for a number of
        // frames, which is only safe for a host whose queue finishes each frame before the next is recorded.
        if (!s.hostRetires)
            std::erase_if(s.retired, [&](const Retired& retired) { return s.frame - retired.frame > kRetireFrames; });
        s.active = s.inEvaluation = false;
        s.frameReason.clear();
        s.borrowedMask = 0;
        s.templateValid = false;
        s.index = 0;
        if (!list || !cuts || !evaluations || s.latched)
            return;
        if (!GetHooks().launchWrapped || !GetHooks().createWrapped)
            return Refuse(s, "Off: hook not installed");
        if (evaluations > kMaxEvaluations)
            return Refuse(s, "Off: too many passes");
        bool any = false;
        for (unsigned i = 0; i < evaluations; ++i)
        {
            if (static_cast<unsigned>(cuts[i]) >= kCuts || (!i && cuts[i] != Cut::None))
                return Refuse(s, "Off: invalid pass plan");
            any = any || cuts[i] != Cut::None;
        }
        if (!any)
            return;
        std::fill(std::begin(s.snapshots), std::end(s.snapshots), uint8_t(0));
        for (unsigned j = 0; j < evaluations; ++j)
        {
            s.cuts[j] = cuts[j];
            if (cuts[j] == Cut::None)
                continue;
            unsigned i = j - 1;
            while (cuts[i] >= cuts[j]) // cuts are nested: a cut at least as deep skips this cut's producer
                --i;
            s.source[j] = static_cast<uint8_t>(i);
            s.snapshots[i] |= static_cast<uint8_t>(1u << static_cast<unsigned>(cuts[j]));
        }
        if (!s.copy && !PrepareGpu(s, list))
            return Refuse(s, "Off: copy shader unavailable");
        s.active = true;
        s.list = list;
        s.evaluations = evaluations;
        s.next = 0;
    }
    catch (...)
    {
        s.active = false;
    }
}

void BeginEvaluation(unsigned evaluation)
{
    FlushOnExit flush;
    tArmed = false;
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    s.inEvaluation = false;
    if (!s.active || s.latched)
        return;
    if (evaluation != s.next || evaluation >= s.evaluations)
    {
        s.active = false;
        return Refuse(s, "Off: evaluations out of order");
    }
    s.next = evaluation + 1;
    s.index = evaluation;
    s.cut = s.cuts[evaluation];
    if (evaluation && s.cut == Cut::None && !s.snapshots[evaluation])
        return; // plain native pass: stay out of the way
    s.current.Clear();
    s.matches = true;
    s.skipping = s.borrowed = false;
    s.candidates = static_cast<uint32_t>((uint64_t(1) << kVariantCount) - 1);
    s.pendingCheck = 0;
    s.spec = nullptr;
    s.mode = Mode::Record;
    if (s.cut != Cut::None)
    {
        std::string reason;
        if (!s.templateValid || s.templateFrame != s.frame)
            reason = s.frameReason.empty() ? "Off: first pass not recognised" : s.frameReason;
        else if (!(s.spec = FindSpec(s.variant, s.cut)))
            reason = std::string("Off: ") + CutName(s.cut) + " not supported for " + VariantName(s.variant);
        else
        {
            const auto gate = Gate(s.templ.launches[0].creator);
            if (gate.state == GateState::Checking)
                reason = "Off: checking runtime";
            else if (gate.state != GateState::Audited || gate.runtime != Tb::kVariants[s.variant].runtime)
                reason = "Off: runtime not audited";
            else
            {
                const auto& snapshot = s.snapshot[static_cast<unsigned>(s.cut)];
                if (!snapshot.valid || snapshot.frame != s.frame || snapshot.evaluation != s.source[evaluation])
                    reason = std::string("Off: no snapshot of ") + CutName(s.cut);
            }
        }
        if (reason.empty())
            s.mode = Mode::Borrow;
        else
        {
            s.spec = nullptr;
            Refuse(s, reason);
        }
    }
    s.inEvaluation = true;
    tArmed = true;
}

Outcome EndEvaluation()
{
    FlushOnExit flush;
    tArmed = false;
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    if (!s.inEvaluation)
        return Outcome::Native;
    s.inEvaluation = false;
    const auto count = s.current.launches.size();
    if (!count)
        Refuse(s, "Off: no NR kernel launches seen on this command list");
    if (s.mode == Mode::Borrow && s.skipping && (!s.borrowed || count != Tb::kVariants[s.variant].count))
        Drop(s, "the evaluation ended after " + std::to_string(count) + " launches");
    const bool complete = s.matches && UniqueVariant(s.candidates) >= 0 &&
                          count == Tb::kVariants[UniqueVariant(s.candidates)].count;
    if (!complete || s.mode == Mode::Dropped)
        for (auto& snapshot : s.snapshot)
            if (snapshot.frame == s.frame && snapshot.evaluation == s.index)
                snapshot.valid = false;
    if (!s.index)
    {
        if (complete)
        {
            std::swap(s.templ, s.current);
            s.templateValid = true;
            s.templateFrame = s.frame;
            s.variant = UniqueVariant(s.candidates);
        }
        else if (s.matches)
            Refuse(s, "Off: unrecognised NR kernel sequence");
    }
    const auto mode = s.mode;
    s.mode = Mode::Off;
    if (mode == Mode::Dropped)
        return Outcome::Dropped;
    if (mode == Mode::Borrow && s.borrowed)
    {
        s.borrowedMask |= static_cast<uint8_t>(1u << static_cast<unsigned>(s.cut));
        return Outcome::Borrowed;
    }
    return Outcome::Native;
}

void EndFrame()
{
    FlushOnExit flush;
    tArmed = false;
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    try
    {
        if (s.latched)
            s.status = "Off after failure: " + s.latchReason;
        else if (s.borrowedMask)
        {
            std::string names;
            for (unsigned c = 1; c < kCuts; ++c)
                if ((s.borrowedMask >> c) & 1)
                    names += (names.empty() ? "" : ", ") + std::string(CutName(static_cast<Cut>(c)));
            s.status = "Borrowing: " + names + " (" + VariantName(s.variant) + ", " +
                       Tb::kVariants[s.variant].runtimeName + ")";
        }
        else if (!s.frameReason.empty())
            s.status = s.frameReason;
    }
    catch (...)
    {
    }
    s.active = s.inEvaluation = false;
    s.mode = Mode::Off;
}

void Reset()
{
    auto& s = GetState();
    {
        std::lock_guard lock(s.mutex);
        s.latched = false;
        s.latchReason.clear();
        s.templateValid = false;
        s.templ.Clear();
        s.variant = -1;
        s.active = s.inEvaluation = false;
        s.mode = Mode::Off;
        s.status = "Waiting for a borrowing frame";
        for (auto& snapshot : s.snapshot)
            snapshot.valid = false;
    }
    auto& logs = Logs();
    std::lock_guard lock(logs.mutex);
    logs.seen.clear();
}

void Status(char* buf, size_t size)
{
    if (!buf || !size)
        return;
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    const char* text = !GetHooks().launchWrapped || !GetHooks().createWrapped ? "Off: hook not installed"
                       : s.latched                                            ? nullptr
                                                                              : s.status.c_str();
    if (text)
        std::snprintf(buf, size, "%s", text);
    else
        std::snprintf(buf, size, "Off after failure: %s", s.latchReason.c_str());
}

namespace Test
{
void SetCopy(CopyFn fn, void* user)
{
    auto& s = GetState();
    std::lock_guard lock(s.mutex);
    s.copy = fn;
    s.copyUser = user;
}
void SetRuntimeGate(GateFn fn, void* user)
{
    GetRuntimes().gateUser.store(user);
    GetRuntimes().gate.store(fn);
}
} // namespace Test

} // namespace NrPassBorrow
