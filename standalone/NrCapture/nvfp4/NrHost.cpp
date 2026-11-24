// Own NR pass host; see NrHost.h. Geometry rules: V2/host-spec/geom.py (checked against 19 NVIDIA traces); field
// codes: V2/host_plan.py. Records are uploaded the way NrAmpere.cpp / NrFp8.cpp do it.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <nvapi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "NrHost.h"
#include "NrHostPlan.h"
#include "NrFourBitPlan.h"
#include "NrWeightUpload.h"

namespace NrHost {
namespace {
using Microsoft::WRL::ComPtr;
using namespace NrHostPlan;
namespace fs = std::filesystem;

constexpr unsigned kCreateModule = 0xad1a677d, kCreateFunction = 0xe2436e22, kLaunch = 0x24973538,
                   kMergedTexture = 0x329fe6e0, kIndependentDescriptor = 0x0ddac234;
constexpr unsigned kRing = 64;                    // per-evaluation descriptor slots (evaluations in flight at most)
constexpr unsigned kPerEval = 4;                  // colour SRV, motion SRV, output SRV, output UAV
constexpr unsigned kStatic = 6;                   // history SRV/UAV, netout SRV/UAV, inputcopy SRV/UAV

void (*g_log)(const char*) = nullptr;
std::atomic<void*> g_driverQuery = nullptr;
void Log(const std::string& line) {
    const std::string text = "NR host: " + line;
    if (g_log) g_log(text.c_str()); else fprintf(stderr, "%s\n", text.c_str());
}
void Check(HRESULT r, const char* what) {
    if (FAILED(r)) Fail(std::string(what) + " failed (D3D12 status 0x" + [&] { char b[16]; snprintf(b, sizeof b, "%08x", unsigned(r)); return std::string(b); }() + ")");
}
void NvCheck(NvAPI_Status r, const std::string& what) {
    if (r != NVAPI_OK) Fail(what + " failed (NVAPI status " + std::to_string(int(r)) + ")");
}
std::wstring Env(const wchar_t* key) {
    const wchar_t* v = _wgetenv(key);
    return v ? std::wstring(v) : std::wstring();
}
std::string Narrow(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) s.push_back(c < 128 ? char(c) : '?');
    return s;
}

// ---- NvAPI ----------------------------------------------------------------------------------------------------------
using Query = void* (__cdecl*)(unsigned);
struct Api {
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_GetCudaMergedTextureSamplerObject) mergedTexture = nullptr;
    decltype(&NvAPI_D3D12_GetCudaIndependentDescriptorObject) independent = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
    Query query = nullptr;
};
Api& GetApi() {
    static Api api;
    if (api.query) return api;
    // The driver's own entry (not a hooked one: the app's NvAPI layers must not see the host's launches).
    auto query = reinterpret_cast<Query>(g_driverQuery.load());
    std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(&FreeLibrary)> module(nullptr, &FreeLibrary);
    if (!query) {
        module.reset(LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
        if (!module) Fail("the system's nvapi64.dll is not available (no NVIDIA driver?)");
        query = reinterpret_cast<Query>(GetProcAddress(module.get(), "nvapi_QueryInterface"));
        if (!query) Fail("nvapi_QueryInterface is missing from nvapi64.dll");
    }
    const auto initialize = reinterpret_cast<decltype(&NvAPI_Initialize)>(query(0x0150e828));
    Api a;
    a.createModule = reinterpret_cast<decltype(a.createModule)>(query(kCreateModule));
    a.createFunction = reinterpret_cast<decltype(a.createFunction)>(query(kCreateFunction));
    a.launch = reinterpret_cast<decltype(a.launch)>(query(kLaunch));
    a.mergedTexture = reinterpret_cast<decltype(a.mergedTexture)>(query(kMergedTexture));
    a.independent = reinterpret_cast<decltype(a.independent)>(query(kIndependentDescriptor));
    a.destroyModule = reinterpret_cast<decltype(a.destroyModule)>(query(0x41c65285));
    a.destroyFunction = reinterpret_cast<decltype(a.destroyFunction)>(query(0xdf295ea6));
    if (!a.createModule || !a.createFunction || !a.launch || !a.mergedTexture || !a.independent ||
        !a.destroyModule || !a.destroyFunction)
        Fail("the NVIDIA driver lacks the CUDA-on-D3D12 entry points the host needs (update the driver)");
    if (!initialize || initialize() != NVAPI_OK) Fail("NvAPI_Initialize failed");
    module.release(); // The successful dispatch table is used for the rest of the process.
    a.query = query;
    api = a;
    return api;
}

// GPU architecture of the D3D12 adapter (NV_GPU_ARCH_INFO), as NrAmpere::CheckArchitecture.
std::pair<unsigned, unsigned> Architecture(ID3D12Device* device) {
    Api& api = GetApi();
    const auto enumerate = reinterpret_cast<decltype(&NvAPI_EnumPhysicalGPUs)>(api.query(0xe5ac921f));
    const auto logical = reinterpret_cast<decltype(&NvAPI_GetLogicalGPUFromPhysicalGPU)>(api.query(0xadd604d1));
    const auto info = reinterpret_cast<decltype(&NvAPI_GPU_GetLogicalGpuInfo)>(api.query(0x842b066e));
    const auto arch = reinterpret_cast<decltype(&NvAPI_GPU_GetArchInfo)>(api.query(0xd8265d24));
    if (!enumerate || !logical || !info || !arch) Fail("the NvAPI architecture query is unavailable");
    const LUID selected = device->GetAdapterLuid();
    NvPhysicalGpuHandle handles[NVAPI_MAX_PHYSICAL_GPUS]{};
    NvU32 count = 0;
    NvCheck(enumerate(handles, &count), "NvAPI_EnumPhysicalGPUs");
    for (NvU32 i = 0; i < count && i < NVAPI_MAX_PHYSICAL_GPUS; ++i) {
        NvLogicalGpuHandle group = nullptr;
        LUID luid{};
        NV_LOGICAL_GPU_DATA data{};
        data.version = NV_LOGICAL_GPU_DATA_VER;
        data.pOSAdapterId = &luid;
        if (logical(handles[i], &group) != NVAPI_OK || info(group, &data) != NVAPI_OK || luid.LowPart != selected.LowPart ||
            luid.HighPart != selected.HighPart) continue;
        NV_GPU_ARCH_INFO a{};
        a.version = NV_GPU_ARCH_INFO_VER;
        NvCheck(arch(handles[i], &a), "NvAPI_GPU_GetArchInfo");
        return {unsigned(a.architecture), unsigned(a.implementation)};
    }
    Fail("no NVIDIA GPU has the D3D12 adapter's LUID");
}
bool IsAmpere(std::pair<unsigned, unsigned> a) { return a.first == NV_GPU_ARCHITECTURE_GA100 && a.second != NV_GPU_ARCH_IMPLEMENTATION_GA100; }
bool IsAda(std::pair<unsigned, unsigned> a) { return a.first == NV_GPU_ARCHITECTURE_AD100; }
bool IsBlackwell2(std::pair<unsigned, unsigned> a) { return a.first == NV_GPU_ARCHITECTURE_GB200; }

struct PlanLaunch : LaunchFields {
    NVAPI_CU_KERNEL_LAUNCH_PARAMS dispatch {};
    std::unique_ptr<NrFourBitPlan::Operation> fourBit;
};
using Plan = ModelPlan<PlanLaunch>;

// ---- per-device objects -----------------------------------------------------------------------------------------------
struct Records : RecordLayout { ComPtr<ID3D12Resource> buffer; };
struct DeviceState {
    std::optional<bool> hostFp8;
    std::map<std::string, NVDX_ObjectHandle> modules;                     // by module path
    std::map<std::pair<std::string, std::string>, NVDX_ObjectHandle> functions;
    std::map<std::string, std::shared_ptr<Records>> records;              // by records file
    std::vector<ComPtr<IUnknown>> retired;                                // objects of destroyed hosts (GPU may still use them)
};
std::mutex g_mutex;
std::map<ID3D12Device*, DeviceState>& Devices() { static auto* d = new std::map<ID3D12Device*, DeviceState>; return *d; }

NVDX_ObjectHandle FunctionFor(ID3D12Device* device, const KernelChoice& c) {
    Api& api = GetApi();
    auto& d = Devices()[device];
    const fs::path path = c.folder / ModuleFile(c);
    const std::string key = path.string();
    if (const auto f = d.functions.find({key, c.entry}); f != d.functions.end()) return f->second;
    NVDX_ObjectHandle module = nullptr;
    if (const auto m = d.modules.find(key); m != d.modules.end()) module = m->second;
    else {
        const auto blob = ReadAll(path);
        NvCheck(api.createModule(device, blob.data(), NvU32(blob.size()), &module), "loading kernel module " + path.filename().string() + " (" + c.code + ")");
        d.modules[key] = module;
    }
    NVDX_ObjectHandle function = nullptr;
    NvCheck(api.createFunction(device, module, c.entry.c_str(), &function), "kernel " + c.entry + " in " + path.filename().string());
    d.functions[{key, c.entry}] = function;
    return function;
}

std::shared_ptr<Records> RecordsFor(ID3D12Device* device, const fs::path& folder, const std::string& prefix, unsigned set) {
    const std::string stem = prefix + std::to_string(set);
    const fs::path bin = folder / (stem + ".bin"), txt = folder / (stem + ".txt");
    auto& d = Devices()[device];
    if (const auto found = d.records.find(bin.string()); found != d.records.end()) return found->second;
    auto r = std::make_shared<Records>();
    const auto blob = ReadAll(bin);
    r->Read(txt, bin, blob);
    const uint64_t bytes = (r->blendAt + 512 + 4096 + 255) / 256 * 256;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r->buffer)), "record buffer");
    ComPtr<ID3D12Resource> upload;
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "record upload buffer");
    void* mapped = nullptr;
    Check(upload->Map(0, nullptr, &mapped), "record upload map");
    memset(mapped, 0, size_t(bytes));
    memcpy(mapped, blob.data(), blob.size());
    memcpy(static_cast<unsigned char*>(mapped) + r->blendAt, &kBlendWeight, 2);
    upload->Unmap(0, nullptr);
    NrWeightUpload::Copy(device, r->buffer.Get(), upload.Get(), bytes, D3D12_RESOURCE_STATE_COMMON);
    d.records[bin.string()] = r;
    Log("records " + stem + " loaded (" + std::to_string(blob.size() >> 20) + " MB)");
    return r;
}

// ---- runtime launch trace (lab) ----------------------------------------------------------------------------------------
struct TraceState {
    std::mutex mutex;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    std::map<NVDX_ObjectHandle, std::string> names;
    std::ofstream out;
    bool opened = false;
    unsigned long long seq = 0;
};
TraceState& TS() { static auto* t = new TraceState; return *t; }
thread_local bool t_hostLaunching = false;
std::string Hex(const void* data, size_t size) {
    static const char* digits = "0123456789abcdef";
    std::string s(size * 2, '0');
    const auto* b = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) { s[2 * i] = digits[b[i] >> 4]; s[2 * i + 1] = digits[b[i] & 15]; }
    return s;
}
NvAPI_Status __cdecl TraceCreateFunction(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& t = TS();
    const auto r = t.createFunction(device, module, name, out);
    if (r == NVAPI_OK && name && out) { std::lock_guard g(t.mutex); t.names[*out] = name; }
    return r;
}
NvAPI_Status __cdecl TraceLaunch(ID3D12GraphicsCommandList* list, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* k, NvU32 count) {
    auto& t = TS();
    if (!t_hostLaunching && k) {
        std::lock_guard g(t.mutex);
        if (!t.opened) {
            const std::wstring path = Env(L"NR_HOST_TRACE_RUNTIME");
            if (!path.empty()) t.out.open(fs::path(path), std::ios::trunc);
            t.opened = true;
            if (t.out) t.out << "seq,list,name,grid,block,shared,size,params\n";
        }
        if (t.out) {
            for (NvU32 i = 0; i < count; ++i) {
                const auto found = t.names.find(k[i].hFunction);
                t.out << t.seq++ << "," << static_cast<const void*>(list) << "," << (found == t.names.end() ? "?" : found->second) << "," << k[i].gridDim.x << "x" << k[i].gridDim.y << "x"
                      << k[i].gridDim.z << "," << k[i].blockDim.x << "x" << k[i].blockDim.y << "x" << k[i].blockDim.z << "," << k[i].dynSharedMemBytes << "," << k[i].paramSize << ","
                      << Hex(k[i].pParams, k[i].paramSize) << "\n";
            }
            t.out.flush();
        }
    }
    return t.launch(list, k, count);
}
}  // namespace

// ---- the host -----------------------------------------------------------------------------------------------------------
struct Host::Impl : ArenaLayout {
    ComPtr<ID3D12Device> device;
    Options options;
    unsigned W = 0, H = 0, set = 0;
    Plan plan;
    Geometry geometry{1920, 1080};
    std::vector<PlanLaunch*> order;                 // plan launches for this layout, in idx order
    std::shared_ptr<Records> records;
    // Prepared records ('prepare' in a manifest): one buffer per host, filled by the first evaluation.
    ComPtr<ID3D12Resource> prepared;
    std::map<int, uint64_t> preparedOffset, preparedBytes;
    std::map<int, NVDX_ObjectHandle> prepareFunctions;
    bool preparedDone = false;
    ComPtr<ID3D12Resource> arena, history, netout, inputcopy;
    ComPtr<ID3D12DescriptorHeap> views, samplers;
    UINT viewStride = 0, samplerStride = 0;
    uint64_t historyTex = 0, historySurf = 0, netoutTex = 0, netoutSurf = 0, inputcopyTex = 0, inputcopySurf = 0;
    unsigned long long evaluations = 0;
    uint32_t sinceReset = 0;                              // the noise seed: evaluations since the last reset frame
    bool historyValid = false;
    bool reportedMissingMotion = false;
    std::ofstream trace;
    // NR_HOST_TIMING: a timestamp before the first launch and after each one; read back kTimingSlots evaluations later.
    static constexpr unsigned kTimingSlots = 4;
    ComPtr<ID3D12QueryHeap> timestamps;
    ComPtr<ID3D12Resource> timingReadback;
    uint64_t timestampFrequency = 0;
    unsigned timingStride = 0;                              // queries per slot
    std::array<std::vector<const PlanLaunch*>, kTimingSlots> timedLaunches;
    std::array<unsigned long long, kTimingSlots> timedEvaluation{};
    std::ofstream timing;
    void WriteTiming(unsigned slot) {
        auto& launches = timedLaunches[slot];
        if (launches.empty() || !timing) return;
        const D3D12_RANGE range{size_t(slot) * timingStride * 8, size_t(slot + 1) * timingStride * 8};
        void* mapped = nullptr;
        if (FAILED(timingReadback->Map(0, &range, &mapped))) return;
        const auto* t = reinterpret_cast<const uint64_t*>(static_cast<const unsigned char*>(mapped) + range.Begin);
        const double us = 1e6 / double(timestampFrequency);
        for (size_t k = 0; k < launches.size(); ++k)
            timing << timedEvaluation[slot] << "," << launches[k]->idx << "," << launches[k]->name << "," << double(t[k + 1] - t[k]) * us << '\n';
        timing << timedEvaluation[slot] << ",total,total," << double(t[launches.size()] - t[0]) * us << '\n';
        const D3D12_RANGE none{0, 0};
        timingReadback->Unmap(0, &none);
        timing.flush();
        launches.clear();
    }
    std::pair<unsigned, unsigned> arch{0, 0};
    std::string kindText;

    D3D12_CPU_DESCRIPTOR_HANDLE ViewCpu(UINT i) const { auto h = views->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * viewStride; return h; }
    D3D12_CPU_DESCRIPTOR_HANDLE SamplerCpu(UINT i) const { auto h = samplers->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * samplerStride; return h; }
    uint64_t TexHandle(UINT view, UINT sampler) {
        NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS p{};
        p.structSizeIn = sizeof p;
        p.pDevice = device.Get();
        p.texDesc = ViewCpu(view);
        p.smpDesc = SamplerCpu(sampler);
        NvCheck(GetApi().mergedTexture(&p), "NvAPI_D3D12_GetCudaMergedTextureSamplerObject");
        return p.textureHandle;
    }
    uint64_t SurfHandle(UINT view) {
        NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS p{};
        p.structSizeIn = sizeof p;
        p.pDevice = device.Get();
        p.type = NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SURFACE;
        p.desc = ViewCpu(view);
        NvCheck(GetApi().independent(&p), "NvAPI_D3D12_GetCudaIndependentDescriptorObject");
        return p.handle;
    }
    void Srv(ID3D12Resource* r, UINT at) {
        const auto d = r->GetDesc();
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = d.Format;
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(r, &v, ViewCpu(at));
    }
    void Uav(ID3D12Resource* r, UINT at) {
        const auto d = r->GetDesc();
        D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
        v.Format = d.Format;
        v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(r, nullptr, &v, ViewCpu(at));
    }
    ComPtr<ID3D12Resource> Texture(const char* what) {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = W; d.Height = H; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
        d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        ComPtr<ID3D12Resource> r;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&r)), what);
        return r;
    }
    void EnsureGradingTextures() {
        if (netout) return;
        auto network = Texture("network output texture");
        auto input = Texture("input copy texture");
        Srv(network.Get(), 2); Uav(network.Get(), 3);
        Srv(input.Get(), 4); Uav(input.Get(), 5);
        netoutTex = TexHandle(2, 0); netoutSurf = SurfHandle(3);
        inputcopyTex = TexHandle(4, 0); inputcopySurf = SurfHandle(5);
        netout = std::move(network);
        inputcopy = std::move(input);
    }

};


Host::Host(std::unique_ptr<Impl> i) : impl(std::move(i)) {}
Host::~Host() {
    if (!impl) return;
    try {
        for (unsigned k = 0; k < Impl::kTimingSlots; ++k) impl->WriteTiming(unsigned((impl->evaluations + k) % Impl::kTimingSlots));
        if (impl->options.callerTracksGpuLifetime) return;
        std::lock_guard g(g_mutex);
        auto& d = Devices()[impl->device.Get()];
        for (IUnknown* o : {static_cast<IUnknown*>(impl->arena.Get()), static_cast<IUnknown*>(impl->history.Get()), static_cast<IUnknown*>(impl->netout.Get()),
                            static_cast<IUnknown*>(impl->inputcopy.Get()), static_cast<IUnknown*>(impl->views.Get()), static_cast<IUnknown*>(impl->samplers.Get()),
                            static_cast<IUnknown*>(impl->prepared.Get())})
            if (o) d.retired.emplace_back(o);
    } catch (...) {
    }
}

std::unique_ptr<Host> Host::Create(ID3D12Device* device, const Options& options, unsigned width, unsigned height, unsigned set) {
    std::lock_guard g(g_mutex);
    auto i = std::make_unique<Impl>();
    i->device = device;
    i->options = options;
    i->W = width; i->H = height; i->set = set;
    if (width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION || height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        Fail("frame dimensions exceed D3D12's texture limit");
    if (width < 64 || height < 64) Fail("frame " + std::to_string(width) + "x" + std::to_string(height) + " is too small");
    if (options.fourBit ? (set != 0 && set != 10 && set != 11 && set != 12 && set != 13) : (set < 1 || set > 7)) Fail("set " + std::to_string(set) + " does not exist");
    const std::string setName = options.recordsName + std::to_string(set);
    i->arch = Architecture(device);
    i->plan = ReadPlan<PlanLaunch>(options.plan);
    if (options.kernels.empty()) Fail("no kernel folders (NR_HOST_ASSETS)");
    // Kernel folders and their codes; the architecture rule.
    Choices choices, mixedChoices;
    std::map<int, const KernelChoice*> kernelOf;
    std::set<std::string> codes;
    for (const auto& [folder, code0] : options.kernels) {
        std::string code = code0.empty() ? options.code : code0;
        // (RTX 40: the sm86 cubins run there, compute capability 8.6 code on 8.9, until an sm89 build exists.)
        if (code.empty()) code = IsAmpere(i->arch) || IsAda(i->arch) ? "sm86" : IsBlackwell2(i->arch) ? "sm120" : "";
        if (code.empty()) Fail("this GPU (architecture 0x" + [&] { char b[16]; snprintf(b, sizeof b, "%x", i->arch.first); return std::string(b); }() + ") is not an RTX 30, 40 or 50 series GPU");
        const bool labOnly = code == "sm86ptx" || code == "nvidia";
        if (labOnly && !options.lab) Fail("the " + code + " code is a lab reference only (set NR_HOST_LAB=1)");
        const bool fits = code == "sm86" ? IsAmpere(i->arch) || IsAda(i->arch)
                        : code == "sm89" ? IsAda(i->arch)
                        : code == "sm120" || code == "nvidia" ? IsBlackwell2(i->arch)
                        : code == "sm86ptx" ? (IsAmpere(i->arch) || IsAda(i->arch) || IsBlackwell2(i->arch)) : false;
        if (!fits) Fail("the " + code + " kernels cannot run on this GPU (architecture 0x" + [&] { char b[16]; snprintf(b, sizeof b, "%x", i->arch.first); return std::string(b); }() + ")");
        codes.insert(code);
        ReadFolder(folder, code, choices);
    }
    if (options.mixedOriginal && set == 0)
        ReadFolder(options.plan.parent_path() / "mixed", "sm120", mixedChoices);
    for (const auto& c : codes) i->kindText += (i->kindText.empty() ? "" : "+") + c;
    // The launches of this layout and their kernels.
    const std::string layout = options.layout == 3108 ? "3108" : "3109";
    int swinSlot = 0;
    for (auto& l : i->plan.launches) {
        if (l.layout != "any" && l.layout != layout) continue;
        i->order.push_back(&l);
        if (options.fourBit) {
            auto& api = GetApi();
            const NrFourBitPlan::Driver driver{api.createModule, api.createFunction, api.destroyModule, api.destroyFunction, api.launch};
            auto& op = l.fourBit = std::make_unique<NrFourBitPlan::Operation>();
            op->driver = api.launch;
            const int block = l.name.starts_with("cc_tinlayout_fused_swin_") ? swinSlot++ : -1;
            op->swinSlot = block;
            op->weightSet = set ? set : ~0u;
            NrNet4::BindNative(*op, driver, device, l.name, l.idx);
            NrSwin4::BindNative(*op, driver, device, l.name, set, block);
            NrFfnV2::BindNative(*op, driver, device, l.name, set, block);
        }
        const bool optional = !choices.contains(l.idx) && mixedChoices.contains(l.idx);
        const auto& available = optional ? mixedChoices : choices;
        const auto found = available.find(l.idx);
        if (l.fourBit && found == available.end()) continue; // Selected four-bit layer supplies its kernel.
        if (found == available.end()) Fail("no kernel for launch " + std::to_string(l.idx) + " (" + l.name + ") in the kernel folders");
        const KernelChoice* pick = nullptr;
        for (const auto& c : found->second) if (!c.params || c.params == l.params) { if (pick && pick->params == l.params) continue; pick = &c; if (c.params == l.params) break; }
        if (!pick) Fail("launch " + std::to_string(l.idx) + ": no kernel takes the " + std::to_string(l.params) + "-byte parameter block");
        if (!pick->name.empty() && pick->name != l.name) Fail("launch " + std::to_string(l.idx) + ": the manifest names " + pick->name + ", the plan " + l.name + " (" + pick->where + ")");
        kernelOf[l.idx] = pick;
        if (optional) {
            // The four-bit dispatchers ask for this kernel only when the selected layer uses FP8.
            // Keep its weights out of the GPU allocation until that first request.
            const auto field = std::find_if(l.fields.begin(), l.fields.end(), [](const Field& f) { return f.code == Code::Record; });
            const int recordOffset = field == l.fields.end() ? -1 : int(field->off);
            l.fourBit->firstFallback = [owner = i.get(), &l, kernel = *pick, recordOffset](
                ID3D12GraphicsCommandList* list, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& incoming) {
                auto& op = *l.fourBit;
                std::vector<unsigned char> forwarded;
                auto launch = op.base;
                launch.pParams = incoming.pParams;
                if (recordOffset >= 0 && !op.record) {
                    if (!owner->records)
                        owner->records = RecordsFor(owner->device.Get(), owner->options.plan.parent_path(), "mixed_set", 1);
                    const auto rec = owner->records->offsets.find(l.idx);
                    if (rec == owner->records->offsets.end() || rec->second.second != l.name)
                        Fail("missing trained FP8 equivalent for launch " + std::to_string(l.idx));
                    const uint64_t address = owner->records->buffer->GetGPUVirtualAddress() + rec->second.first;
                    // An advanced layer may have already made a copy with rewired input/output addresses.
                    if (incoming.pParams != l.parameters.data()) {
                        const auto bytes = static_cast<const unsigned char*>(incoming.pParams);
                        forwarded.assign(bytes, bytes + incoming.paramSize);
                        memcpy(forwarded.data() + recordOffset, &address, sizeof address);
                        launch.pParams = forwarded.data();
                    }
                    memcpy(l.parameters.data() + recordOffset, &address, sizeof address);
                    op.record = address;
                }
                launch.hFunction = op.base.hFunction = FunctionFor(owner->device.Get(), kernel);
                return op.driver(list, &launch, 1);
            };
        } else {
            l.dispatch.hFunction = FunctionFor(device, *pick);
        }
        if (!pick->prepare.empty()) {
            KernelChoice prep = *pick;
            prep.entry = pick->prepare;
            i->prepareFunctions[l.idx] = FunctionFor(device, prep);
        }
    }
    std::sort(i->order.begin(), i->order.end(), [](const PlanLaunch* a, const PlanLaunch* b) { return a->idx < b->idx; });
    // Records: every weighted launch must have one, under the plan's kernel name.
    if (!options.fourBit) i->records = RecordsFor(device, options.records, options.recordsName, set);

    for (const auto* l : i->order)
        for (const auto& f : l->fields)
            if (!options.fourBit && f.code == Code::Record) {
                const auto r = i->records->offsets.find(l->idx);
                if (r == i->records->offsets.end()) Fail(setName + ".txt has no record for launch " + std::to_string(l->idx));
                if (r->second.second != l->name) Fail(setName + ".txt launch " + std::to_string(l->idx) + " is " + r->second.second + ", the plan's is " + l->name);
            }
    {   // Prepared records.
        uint64_t total = 0;
        for (const auto& [idx, function] : i->prepareFunctions) {
            const auto bytes = i->records->bytes.find(idx);
            if (bytes == i->records->bytes.end() || !bytes->second) Fail("launch " + std::to_string(idx) + " has a 'prepare' but no record bytes in " + setName + ".txt");
            const uint64_t out = (uint64_t(double(bytes->second) * kernelOf.at(idx)->prepareFactor) + 255) / 256 * 256;
            i->preparedOffset[idx] = total;
            i->preparedBytes[idx] = out;
            total += out;
        }
        if (total) {
            D3D12_RESOURCE_DESC d{};
            d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            d.Width = total; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
            d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&i->prepared)), "prepared record buffer");
            Log("prepared records: " + std::to_string(i->prepareFunctions.size()) + " launches, " + std::to_string(total >> 20) + " MB");
        }
    }
    // Geometry, flag region, arena.
    Geometry& geo = i->geometry;
    geo = Geometry(width, height);
    i->Build(geo, i->plan, i->order);
    // Geometry and kernel selection are fixed for the lifetime of this host.
    for (auto* l : i->order) {
        const KernelChoice base{};
        const auto it = kernelOf.find(l->idx);
        const auto& kernel = it == kernelOf.end() ? base : *it->second;
        const auto [grid, block] = Grid(l->name, geo, l->sx, l->sy);
        auto& dispatch = l->dispatch;
        dispatch.gridDim = {Apply(kernel.grid[0], grid.x), Apply(kernel.grid[1], grid.y), Apply(kernel.grid[2], grid.z)};
        dispatch.blockDim = {Apply(kernel.block[0], block.x), Apply(kernel.block[1], block.y), Apply(kernel.block[2], block.z)};
        dispatch.dynSharedMemBytes = kernel.shared;
        dispatch.paramSize = l->params;
        if (l->fourBit) {
            l->fourBit->base = dispatch;
            dispatch.hFunction = nullptr;
            dispatch.gridDim = {grid.x, grid.y, grid.z};
            dispatch.blockDim = {block.x, block.y, block.z};
            dispatch.dynSharedMemBytes = 0;
        }
    }
    {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = i->arenaBytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&i->arena)), "arena buffer");
    }
    i->history = i->Texture("history texture");
    {
        D3D12_DESCRIPTOR_HEAP_DESC h{};
        h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        h.NumDescriptors = kStatic + kRing * kPerEval;
        h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&i->views)), "descriptor heap");
        h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        h.NumDescriptors = 2;
        Check(device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&i->samplers)), "sampler heap");
        i->viewStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        i->samplerStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        for (UINT k = 0; k < 2; ++k) {
            D3D12_SAMPLER_DESC s{};
            s.Filter = k == 0 ? D3D12_FILTER_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            s.MaxAnisotropy = 1;
            s.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            s.MaxLOD = D3D12_FLOAT32_MAX;
            device->CreateSampler(&s, i->SamplerCpu(k));
        }
        i->Srv(i->history.Get(), 0); i->Uav(i->history.Get(), 1);
        const bool historyLinear = Env(L"NR_HOST_HISTORY_SAMPLER") != L"point";
        i->historyTex = i->TexHandle(0, historyLinear ? 1 : 0); i->historySurf = i->SurfHandle(1);
    }
    // Geometry, tensor addresses, synchronization flags and weights stay fixed until this host is retired.
    const uint64_t arena = i->arena->GetGPUVirtualAddress();
    const uint64_t recordBase = i->records ? i->records->buffer->GetGPUVirtualAddress() : 0;
    const uint64_t preparedBase = i->prepared ? i->prepared->GetGPUVirtualAddress() : 0;
    for (auto* l : i->order) {
        uint64_t record = 0;
        if (l->fourBit) record = l->fourBit->record;
        else if (i->preparedOffset.contains(l->idx)) record = preparedBase + i->preparedOffset.at(l->idx);
        else if (i->records->offsets.contains(l->idx)) record = recordBase + i->records->offsets.at(l->idx).first;
        BindParameters(*l, geo, *i, arena, record, l->fourBit ? l->fourBit->blend : recordBase + i->records->blendAt);
    }
    if (!options.timing.empty()) {
        i->timingStride = unsigned(i->order.size()) + 2;
        D3D12_QUERY_HEAP_DESC q{};
        q.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        q.Count = i->timingStride * Impl::kTimingSlots;
        Check(device->CreateQueryHeap(&q, IID_PPV_ARGS(&i->timestamps)), "timestamp query heap");
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = uint64_t(q.Count) * 8; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&i->timingReadback)), "timestamp readback");
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue;
        Check(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "timestamp frequency queue");
        Check(queue->GetTimestampFrequency(&i->timestampFrequency), "timestamp frequency");
        i->timing.open(options.timing, std::ios::trunc);
        if (i->timing) i->timing << "eval,idx,name,us\n";
    }
    if (!options.trace.empty()) {
        i->trace.open(options.trace, std::ios::trunc);
        if (i->trace) i->trace << "eval,idx,name,grid,block,shared,size,params\n";
    }
    Log("pass host ready: " + std::to_string(width) + "x" + std::to_string(height) + ", " + setName + ", kernels " + i->kindText + ", " +
        std::to_string(i->order.size()) + " launches, arena " + std::to_string(i->arenaBytes >> 20) + " MB (flags " + std::to_string(i->flagBytes) + " B), layout " + layout);
    return std::unique_ptr<Host>(new Host(std::move(i)));
}

void Host::Evaluate(ID3D12GraphicsCommandList* commands, ID3D12Resource* colour, ID3D12Resource* motion, ID3D12Resource* output, const Settings& s, bool reset) {
    std::lock_guard g(g_mutex);
    Impl& m = *impl;
    if (!colour || !output) Fail("no colour or output resource");
    const auto cd = colour->GetDesc(), od = output->GetDesc();
    if (cd.Width != m.W || cd.Height != m.H || od.Width != m.W || od.Height != m.H)
        Fail("the colour/output size differs from the host's " + std::to_string(m.W) + "x" + std::to_string(m.H));
    if (s.style > 2) Fail("style " + std::to_string(s.style));
    const bool resetFrame = reset || !m.historyValid || !motion;
    if (motion) m.reportedMissingMotion = false;
    else if (!reset && !m.reportedMissingMotion) {
        Log("motion vectors unavailable, resetting history until they return");
        m.reportedMissingMotion = true;
    }
    // Blending reads the unfiltered input and the raw model result. Keep the latter as temporal history.
    const bool graded = s.style != 0 || s.intensity != 1.f;
    if (graded) m.EnsureGradingTextures();
    const unsigned slot = unsigned(m.evaluations % kRing);
    const UINT base = kStatic + slot * kPerEval;
    m.Srv(colour, base + 0);
    if (!resetFrame) m.Srv(motion, base + 1);
    if (!graded) m.Srv(output, base + 2);
    m.Uav(output, base + 3);
    const uint64_t colourTex = m.TexHandle(base + 0, 0);
    const uint64_t motionTex = motion && !resetFrame ? m.TexHandle(base + 1, 0) : 0;
    const uint64_t outputTex = graded ? 0 : m.TexHandle(base + 2, 0);
    const uint64_t outputSurf = m.SurfHandle(base + 3);
    ID3D12DescriptorHeap* heaps[] = {m.views.Get(), m.samplers.Get()};
    commands->SetDescriptorHeaps(2, heaps);
    const uint64_t recordBase = m.records ? m.records->buffer->GetGPUVirtualAddress() : 0;
    const Geometry& geo = m.geometry;
    const FrameHandles handles{colourTex, motionTex, outputTex, outputSurf, m.historyTex, m.historySurf,
                               m.netoutTex, m.netoutSurf, m.inputcopyTex, m.inputcopySurf};
    const auto launch = GetApi().launch;
    const bool timed = m.timestamps != nullptr;
    const unsigned timingSlot = unsigned(m.evaluations % Impl::kTimingSlots);
    unsigned query = timingSlot * m.timingStride;
    if (timed) {
        m.WriteTiming(timingSlot);                          // the evaluation kTimingSlots ago (finished by now)
        m.timedEvaluation[timingSlot] = m.evaluations;
        commands->EndQuery(m.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query++);
    }
    struct HostLaunching { HostLaunching() { t_hostLaunching = true; } ~HostLaunching() { t_hostLaunching = false; } } launching;
    const uint64_t preparedBase = m.prepared ? m.prepared->GetGPUVirtualAddress() : 0;
    if (m.prepared && !m.preparedDone) {     // the first evaluation turns the records into the kernels' own layout
        struct PrepareParams { uint64_t in, out; uint32_t inBytes, outBytes; int32_t launch, pad; } pp{};
        for (const auto& [idx, function] : m.prepareFunctions) {
            pp.in = recordBase + m.records->offsets.at(idx).first;
            pp.out = preparedBase + m.preparedOffset.at(idx);
            pp.inBytes = uint32_t(m.records->bytes.at(idx));
            pp.outBytes = uint32_t(m.preparedBytes.at(idx));
            pp.launch = idx;
            NVAPI_CU_KERNEL_LAUNCH_PARAMS p{};
            p.hFunction = function;
            p.gridDim = {std::max(1u, std::min(1024u, (pp.outBytes + 1023) / 1024)), 1, 1};
            p.blockDim = {256, 1, 1};
            p.pParams = &pp;
            p.paramSize = sizeof pp;
            const auto status = launch(commands, &p, 1);
            if (status != NVAPI_OK) NvCheck(status, "preparing the record of launch " + std::to_string(idx));
        }
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        commands->ResourceBarrier(1, &b);
        m.preparedDone = true;
    }
    for (PlanLaunch* l : m.order) {
        if (l->styles == "graded" && !graded) continue;
        const bool barrier = l->barrier || m.options.barriersAll;
        if (barrier) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.UAV.pResource = nullptr;
            commands->ResourceBarrier(1, &b);
        }
        FrameParameters(*l, geo, m.plan.grades, s, handles, resetFrame, graded, m.sinceReset);
        auto& block = l->parameters;
        auto p = l->dispatch;
        p.pParams = block.data();
        if (m.trace)
            m.trace << m.evaluations << "," << l->idx << "," << l->name << "," << p.gridDim.x << "x" << p.gridDim.y << "x" << p.gridDim.z << "," << p.blockDim.x << "x" << p.blockDim.y << "x"
                    << p.blockDim.z << "," << p.dynSharedMemBytes << "," << l->params << "," << Hex(block.data(), l->params) << "\n";
        NvAPI_Status status;
        if (l->fourBit) {
            NrFourBitPlan::Scope scope(*l->fourBit);
            status = NrPassBorrow::SubmitOwned(NrNet4::SubmitNative, l->name.c_str(), commands, &p, 1);
        } else {
            status = launch(commands, &p, 1);
        }
        if (status != NVAPI_OK) {
            const auto detail = l->fourBit && !l->fourBit->failure.empty() ? ": " + l->fourBit->failure : "";
            NvCheck(status, "launch " + std::to_string(l->idx) + " (" + l->name + ")" + detail);
        }
        if (timed) { commands->EndQuery(m.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query++); m.timedLaunches[timingSlot].push_back(l); }
    }
    if (timed)
        commands->ResolveQueryData(m.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timingSlot * m.timingStride, query - timingSlot * m.timingStride,
                                   m.timingReadback.Get(), uint64_t(timingSlot) * m.timingStride * 8);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    commands->ResourceBarrier(1, &b);
    if (m.trace) m.trace.flush();
    m.historyValid = true;
    m.sinceReset = resetFrame ? 1u : m.sinceReset + 1u;
    ++m.evaluations;
    if (!m.options.report.empty()) {
        std::ofstream r(m.options.report, std::ios::trunc);
        r << "own pass host: " << m.kindText << ", " << m.options.recordsName << m.set << ", " << m.W << "x" << m.H << ", evaluations " << m.evaluations
          << ", nvngx_dlssnr.dll loaded: " << (GetModuleHandleW(L"nvngx_dlssnr.dll") ? "yes" : "no")
          << ", NGX dispatcher loaded: " << (GetModuleHandleW(L"_nvngx.dll") ? "yes" : "no") << "\n";
    }
}

unsigned Host::Width() const { return impl->W; }
unsigned Host::Height() const { return impl->H; }
unsigned Host::Set() const { return impl->set; }
unsigned long long Host::Evaluations() const { return impl->evaluations; }
std::string Host::Status() const {
    return "own pass host: " + impl->kindText + ", " + impl->options.recordsName + std::to_string(impl->set) + ", " + std::to_string(impl->W) + "x" + std::to_string(impl->H) + ", " +
           std::to_string(impl->evaluations) + " evaluations";
}

bool Requested(ID3D12Device* device) {
    const std::wstring v = Env(L"NR_HOST");
    if (!v.empty()) return v != L"0";
    if (!device) return false;
    // Unset: on by itself where NVIDIA's runtime refuses the GPU (below the RTX 50 series), so an FP8 model runs there.
    std::lock_guard g(g_mutex);
    auto& cached = Devices()[device].hostFp8;
    if (cached) return *cached;
    bool own = false;
    try { const auto a = Architecture(device); own = IsAmpere(a) || IsAda(a); } catch (const std::exception&) {}
    cached = own;
    return own;
}

Options FromEnvironment(const fs::path& exeDir) {
    Options o;
    o.enabled = true;
    const std::wstring plan = Env(L"NR_HOST_PLAN");
    o.plan = plan.empty() ? exeDir / "nvfp4" / "host" / "modelA_fp8.plan" : fs::path(plan);
    const std::wstring records = Env(L"NR_FP8_ASSETS");
    o.records = records.empty() ? exeDir / "nvfp4" / "fp8" : fs::path(records);
    o.code = Narrow(Env(L"NR_HOST_CODE"));
    if (!o.code.empty() && o.code != "sm86" && o.code != "sm89" && o.code != "sm120" && o.code != "sm86ptx" && o.code != "nvidia")
        Fail("NR_HOST_CODE must be sm86, sm89, sm120, sm86ptx or nvidia");
    const std::wstring assets = Env(L"NR_HOST_ASSETS");
    std::wstring rest = assets.empty() ? (exeDir / "nvfp4" / "host").wstring() : assets;
    size_t start = 0;
    while (start <= rest.size()) {
        const size_t semi = rest.find(L';', start);
        std::wstring part = rest.substr(start, semi == std::wstring::npos ? std::wstring::npos : semi - start);
        if (!part.empty()) {
            std::string code;
            if (const size_t at = part.rfind(L'@'); at != std::wstring::npos) { code = Narrow(part.substr(at + 1)); part.resize(at); }
            o.kernels.push_back({fs::path(part), code});
        }
        if (semi == std::wstring::npos) break;
        start = semi + 1;
    }
    o.lab = Env(L"NR_HOST_LAB") == L"1";
    const std::wstring barriers = Env(L"NR_HOST_BARRIERS");
    if (!barriers.empty() && barriers != L"all" && barriers != L"inferred") Fail("NR_HOST_BARRIERS must be all or inferred");
    o.barriersAll = barriers == L"all";
    const std::wstring layout = Env(L"NR_HOST_LAYOUT");
    if (!layout.empty() && layout != L"3108" && layout != L"3109") Fail("NR_HOST_LAYOUT must be 3108 or 3109");
    o.layout = layout == L"3108" ? 3108 : 3109;
    o.trace = Env(L"NR_HOST_TRACE");
    o.report = Env(L"NR_HOST_REPORT");
    o.timing = Env(L"NR_HOST_TIMING");
    return o;
}

Options FourBitOptions(const fs::path& assets, bool allowMixedOriginal) {
    Options o;
    o.enabled = o.fourBit = true;
    o.mixedOriginal = allowMixedOriginal;
    const auto folder = assets / "host4";
    o.plan = folder / "modelA_fp8.plan";
    o.recordsName = "nvfp4 set ";
    o.kernels.emplace_back(folder, "sm120");
    o.report = Env(L"NR_HOST_REPORT");
    o.trace = Env(L"NR_HOST_TRACE");
    o.timing = Env(L"NR_HOST_TIMING");
    return o;
}

Options Int8Options(const fs::path& folder) {
    Options o;
    o.enabled = true;
    o.plan = folder / "modelA_i8.plan";
    o.records = folder;
    o.recordsName = "int8_set";
    o.kernels.push_back({folder, std::string()});
    o.code = Narrow(Env(L"NR_HOST_CODE"));
    if (!o.code.empty() && o.code != "sm86" && o.code != "sm120") Fail("NR_HOST_CODE must be sm86 or sm120 for the INT8 models");
    o.barriersAll = Env(L"NR_HOST_BARRIERS") == L"all";
    o.trace = Env(L"NR_HOST_TRACE");
    o.report = Env(L"NR_HOST_REPORT");
    o.timing = Env(L"NR_HOST_TIMING");
    return o;
}

void ReleaseDevice(ID3D12Device* device) {
    NrWeightUpload::Collect();
    std::lock_guard g(g_mutex);
    auto& devices = Devices();
    const auto found = devices.find(device);
    if (found == devices.end()) return;
    auto& state = found->second;
    if (!state.modules.empty()) {
        auto& api = GetApi();
        for (const auto& [key, function] : state.functions) api.destroyFunction(device, function);
        for (const auto& [path, module] : state.modules) api.destroyModule(device, module);
    }
    devices.erase(found);
}
void SetLog(void (*log)(const char*)) { g_log = log; }
void SetDriverQuery(void* query) { g_driverQuery = query; }

void* WrapNvapi(unsigned id, void* original) {
    if (!original || Env(L"NR_HOST_TRACE_RUNTIME").empty()) return original;
    auto& t = TS();
    if (id == kCreateFunction) { t.createFunction = reinterpret_cast<decltype(t.createFunction)>(original); return reinterpret_cast<void*>(&TraceCreateFunction); }
    if (id == kLaunch) { t.launch = reinterpret_cast<decltype(t.launch)>(original); return reinterpret_cast<void*>(&TraceLaunch); }
    return original;
}

}  // namespace NrHost
