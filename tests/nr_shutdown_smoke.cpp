// Production shutdown/destructor bodies are extracted by run_nr_shutdown.ps1.
// Only dependencies are counted/mocked; no NGX runtime or GPU is loaded.
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#define NVSDK_NGX_API
#define LOG_INFO(...) ((void) 0)
#define LOG_WARN(...) ((void) 0)
#define NVSDK_NGX_FAILED(r) ((r) != 0)
#define _ReturnAddress() nullptr
using NVSDK_NGX_Result = int;
using PVOID = void*;
constexpr int NVSDK_NGX_Result_Success = 0, NVSDK_NGX_Result_Fail = 1;
struct ID3D12Device
{
};
enum class FGInput
{
    None,
    Upscaler
};
enum class FGNvngxReplacement
{
    None,
    Active
};
enum class Upscaler
{
    None,
    DLSS
};
enum class API
{
    DX12
};
struct FeatureBase
{
    virtual ~FeatureBase() = default;
    virtual Upscaler GetUpscalerType() const { return Upscaler::None; }
    API Api() const { return API::DX12; }
};
unsigned calls0, calls1, fgCalls, rrCalls, otherClients, liveNr, destroyedNr, liveAtShutdown, parentDestructions;
bool ready = true;
int stopResult = 0;
bool suppliedShutdownState = false;
struct FG
{
    void Deactivate() {}
    void DestroyFGContext() {}
};
struct State
{
    bool isShuttingDown = false, nvngxDx12Inited = true, clearCapturedHudlesses = false;
    FeatureBase* currentFeature = nullptr;
    FG* currentFG = nullptr;
    FGInput activeFgInput = FGInput::None;
    FGNvngxReplacement activeFgNvngx = FGNvngxReplacement::None;
    std::vector<void*> modulesToFree;
    static State& Instance()
    {
        static State s;
        return s;
    }
};
int RealShutdown()
{
    ++calls0;
    liveAtShutdown = liveNr;
    return stopResult;
}
int RealShutdown1(ID3D12Device*, unsigned int* shutdownState)
{
    ++calls1;
    suppliedShutdownState = shutdownState && *shutdownState == 0;
    if (shutdownState)
        *shutdownState = 1;
    liveAtShutdown = liveNr;
    return stopResult;
}
using PFN_D3D12_Shutdown = int (*)(void);
using PFN_Nvngx_D3D12_Shutdown1 = int (*)(ID3D12Device*, unsigned int*);
struct NgxModule
{
    PFN_D3D12_Shutdown D3D12_Shutdown;
    PFN_Nvngx_D3D12_Shutdown1 D3D12_Shutdown1;
};
struct CustomNgx
{
};
struct NVNGXProxy
{
    inline static bool _dx12Inited = true;
    inline static NgxModule _module { &RealShutdown, &RealShutdown1 };
    static bool IsDx12Inited() { return _dx12Inited; }
    static void SetDx12Inited(bool b) { _dx12Inited = b; }
    static void ShutdownCustomDx12(CustomNgx&, ID3D12Device*) {}
#include "shutdown_accessors.inc"
};
namespace Nvngx_FG
{
void D3D12_Shutdown() { ++fgCalls; }
void D3D12_Shutdown1(ID3D12Device*) { ++fgCalls; }
} // namespace Nvngx_FG
namespace Nvngx_RR
{
void ShutdownDx12(ID3D12Device*) { ++rrCalls; }
} // namespace Nvngx_RR
// Another module (Streamline, a mod) still holding the NGX instance skips the teardown
struct NVNGXClients
{
    bool Remove(void*)
    {
        if (otherClients == 0)
            return true;
        --otherClients;
        return false;
    }
} _clients;
namespace DlssNr
{
void ClearStatus(void*) {}
bool Shutdown();
} // namespace DlssNr
struct DLSSFeature
{
    static void Shutdown() {}
};
struct DLSSFeatureDx12 : FeatureBase, DLSSFeature
{
    bool _dlssInited = true;
    CustomNgx* _custom = nullptr;
    Upscaler GetUpscalerType() const override { return Upscaler::DLSS; }
    void Shutdown(ID3D12Device*);
};
struct NrState
{
    std::recursive_mutex mutex;
    struct Late
    {
        void Cancel() {}
    } late;
};
struct DlssNr_Dx12
{
    std::unique_ptr<NrState> _state = std::make_unique<NrState>();
    DlssNr_Dx12() { ++liveNr; }
    ~DlssNr_Dx12()
    {
        --liveNr;
        ++destroyedNr;
    }
    static void Retire(std::unique_ptr<DlssNr_Dx12>);
};
std::recursive_mutex nrOwnersMutex;
DlssNr_Dx12* activeNrOwner = nullptr;
auto& RetiredNrOwners()
{
    static std::list<std::unique_ptr<DlssNr_Dx12>> owners;
    return owners;
}
bool DlssNr::Shutdown()
{
    if (!ready)
        return false;
    RetiredNrOwners().clear();
    return true;
}
struct IFeature_Dx12
{
    inline static std::unique_ptr<int> Imgui;
    std::unique_ptr<int> OutputScaler, RCAS, Bias, Magnifier, UpscalerTime;
    std::unique_ptr<DlssNr_Dx12> NeuralRendering = std::make_unique<DlssNr_Dx12>();
    ~IFeature_Dx12();
    void RetireNeuralRendering();
};
struct Parent : IFeature_Dx12
{
    ~Parent() { ++parentDestructions; }
};
struct Entry
{
    std::unique_ptr<Parent> feature;
};
std::map<unsigned, Entry> Dx12Contexts;
struct Registry
{
    void Clear() {}
} HandleToFeature;
ID3D12Device device;
ID3D12Device* D3D12Device = &device;
bool shutdown = false;
unsigned unloaded, logged, closedLogger;
void* skModule = reinterpret_cast<void*>(1);
void* reshadeModule = reinterpret_cast<void*>(2);
std::vector<void*> _asiHandles { reinterpret_cast<void*>(3) };
namespace NtdllProxy
{
void FreeLibrary_Ldr(void*) { ++unloaded; }
} // namespace NtdllProxy
namespace spdlog
{
void info(const char*) { ++logged; }
} // namespace spdlog
void CloseLogger() { ++closedLogger; }
constexpr unsigned DLL_PROCESS_DETACH = 0;

#include "shutdown_production.inc"

void expect(bool ok, const char* message)
{
    if (!ok)
    {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}
void reset()
{
    State::Instance().isShuttingDown = false;
    Dx12Contexts.clear();
    RetiredNrOwners().clear();
    calls0 = calls1 = fgCalls = rrCalls = otherClients = liveNr = destroyedNr = liveAtShutdown = parentDestructions = 0;
    unloaded = logged = closedLogger = 0;
    ready = true;
    stopResult = 0;
    suppliedShutdownState = false;
    shutdown = false;
    State::Instance() = {};
    State::Instance().modulesToFree.push_back(reinterpret_cast<void*>(4));
    NVNGXProxy::_dx12Inited = true;
    NVNGXProxy::_module = { &RealShutdown, &RealShutdown1 };
    D3D12Device = &device;
}
int main()
{
    reset();
    Dx12Contexts[1].feature = std::make_unique<Parent>();
    State::Instance().activeFgNvngx = FGNvngxReplacement::Active;
    DLSSFeatureDx12 dlss;
    State::Instance().currentFeature = &dlss;
    expect(NVSDK_NGX_D3D12_Shutdown() == 0, "legacy shutdown failed");
    expect(calls0 == 1 && calls1 == 0 && fgCalls == 1 && rrCalls == 1 && liveAtShutdown == 0 &&
               parentDestructions == 1,
           "runtime shutdown duplicated or ran before NR/parent release");
    expect(!NVNGXProxy::IsDx12Inited() && !dlss._dlssInited && State::Instance().currentFeature == nullptr,
           "init flags not reset");
    NVSDK_NGX_D3D12_Shutdown();
    expect(calls0 == 1 && fgCalls == 1, "repeated shutdown called a driver/provider again");

    reset();
    NVSDK_NGX_D3D12_Shutdown1(&device);
    expect(calls1 == 1 && calls0 == 0, "device shutdown called both exports");
    expect(suppliedShutdownState, "private shutdown export did not receive initialized output storage");
    reset();
    NVNGXProxy::_module.D3D12_Shutdown = nullptr;
    NVSDK_NGX_D3D12_Shutdown();
    expect(calls1 == 1 && calls0 == 0, "single-export runtime fallback failed");
    expect(suppliedShutdownState, "shutdown fallback did not receive initialized output storage");
    reset();
    NVNGXProxy::_module.D3D12_Shutdown1 = nullptr;
    NVSDK_NGX_D3D12_Shutdown1(&device);
    expect(calls0 == 1 && calls1 == 0, "legacy-only runtime fallback failed");

    reset();
    otherClients = 1;
    Dx12Contexts[1].feature = std::make_unique<Parent>();
    expect(NVSDK_NGX_D3D12_Shutdown() == 0 && calls0 == 0 && liveNr == 1 && NVNGXProxy::IsDx12Inited(),
           "a client's shutdown tore down NGX while another module still uses it");
    NVSDK_NGX_D3D12_Shutdown();
    expect(calls0 == 1 && liveNr == 0, "the last client's shutdown did not tear down");

    reset();
    ready = false;
    Dx12Contexts[1].feature = std::make_unique<Parent>();
    expect(NVSDK_NGX_D3D12_Shutdown() != 0, "unresolved work reported successful shutdown");
    expect(calls0 == 0 && calls1 == 0 && liveNr == 1 && parentDestructions == 0 && NVNGXProxy::IsDx12Inited(),
           "unresolved GPU work lost its runtime or parent owner");
    ready = true;
    NVSDK_NGX_D3D12_Shutdown();
    expect(calls0 == 1 && liveAtShutdown == 0 && parentDestructions == 1, "deferred shutdown could not finish");

    reset();
    stopResult = 7;
    expect(NVSDK_NGX_D3D12_Shutdown() == 7 && NVNGXProxy::IsDx12Inited(), "driver failure was hidden");
    stopResult = 0;
    NVSDK_NGX_D3D12_Shutdown();
    expect(calls0 == 2 && !NVNGXProxy::IsDx12Inited(), "failed shutdown could not be retried");

    reset();
    State::Instance().isShuttingDown = true;
    DlssNr_Dx12* abandoned;
    {
        IFeature_Dx12 feature;
        abandoned = feature.NeuralRendering.get();
    }
    expect(destroyedNr == 0 && RetiredNrOwners().empty(), "detach guard invoked NR destructor/retirement");
    NVSDK_NGX_D3D12_Shutdown();
    NVSDK_NGX_D3D12_Shutdown1(&device);
    expect(calls0 == 0 && calls1 == 0, "detach shutdown entered driver");
    State::Instance().isShuttingDown = false;
    delete abandoned;

    reset();
    Detach(reinterpret_cast<void*>(1));
    expect(State::Instance().isShuttingDown && !unloaded && !logged && !closedLogger,
           "process termination unloaded a DLL or entered logging");
    reset();
    Detach(nullptr);
    expect(unloaded == 4 && closedLogger == 1, "dynamic detach cleanup changed unexpectedly");
    reset();
    std::puts("NR shutdown regressions passed (ordering, idempotence, failure, detach)");
}
