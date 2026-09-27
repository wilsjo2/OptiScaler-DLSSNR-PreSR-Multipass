// tests/fgpresent_owner_routing.cpp - FGPresent incoming-owner routing regression (upstream T8 follow-up).
//
// WHAT THIS PROVES: the ACTUAL FGHooks::FGPresent body (spliced at test-build time from
// OptiScaler/hooks/FG_Hooks.cpp by tests/run_fgpresent_owner_routing.ps1 - this file only
// supplies the stub world below) routes a non-owner incoming XeFG present to the original
// Present/Present1 exactly once with its HRESULT unchanged, without invoking the live
// feature or generic NR. Positive controls prove the registered chain still drives the
// feature once and bypasses generic NR natively while preserving upstream bridge generic routing.
// No GPU, no game, no window: swapchains are distinct fake pointers (FGPresent only
// compares/forwards them), the provider feature is an active IFGFeature double.
//
// The runner also builds the same harness against a guard-removed seed body (must fail
// PIN_FG_NONOWNER) to prove the test detects the defect. A seed that exits 0 means the
// regression no longer guards the boundary.
//
// Usage: fgpresent_owner_routing.exe (no arguments; all cases self-contained).
// Exit codes: 0 = every routing pin held; 1 = a pin failed (PIN_* marker printed).

#include <windows.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <dxgi1_5.h>
#include <d3d11.h>
#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <optional>
#include <shared_mutex>

// --- logging: production LOG_* take fmt strings; discard everything here ---
#define LOG_TRACE(...) ((void) 0)
#define LOG_DEBUG(...) ((void) 0)
#define LOG_INFO(...) ((void) 0)
#define LOG_WARN(...) ((void) 0)
#define LOG_ERROR(...) ((void) 0)

// --- enums used by the real body ---
enum class FGOutput
{
    NoFG = 0,
    DLSSG,
    XeFG
};
enum class FGInput
{
    None = 0,
    FSRFG,
    FSRFG30
};
enum class SwapchainInteropApi
{
    None = 0,
    Dx11wDx12
};
enum class GameEngineType
{
    Unknown = 0,
    Unity
};

namespace sl
{
struct FrameToken
{
};
enum class Result
{
    eOk = 0,
    eErrorReflexAPI
};
enum class PCLMarker
{
    ePresentStart = 0,
    ePresentEnd
};
} // namespace sl

using xefg_swapchain_handle_t = void*;
struct xefg_swapchain_present_status_t
{
    int isFrameGenEnabled = 0;
    unsigned frameGenResult = 0;
    unsigned framesPresented = 0;
};
static constexpr int XEFG_SWAPCHAIN_RESULT_SUCCESS = 0;

// --- Config stub: only the options FGPresent reads ---
template <typename T> struct TestOpt
{
    bool has = false;
    T v {};
    bool has_value() const { return has; }
    T value() const { return v; }
    T value_or_default() const { return v; }
};

struct Config
{
    static Config* Instance()
    {
        static Config instance;
        return &instance;
    }
    TestOpt<bool> FGUseMutexForSwapchain {};
    TestOpt<bool> ForceVsync {};
    TestOpt<unsigned> VsyncInterval {};
    TestOpt<bool> FGDLSSGUseGamesReflexMarkers {};
    TestOpt<bool> SimulateWaitableObject {};
};

// --- IFGFeature double: active/unpaused, counts Present() ---
struct FakeMutex
{
    int getOwner() { return 0; }
    void lock(int) {}
    void unlockThis(int) {}
};

struct IFGFeature
{
    FakeMutex Mutex {};
    virtual ~IFGFeature() = default;
    virtual bool IsActive() = 0;
    virtual bool IsPaused() = 0;
    virtual void Present() = 0;
    virtual uint64_t FrameCount() = 0;
};

struct LiveFeature : IFGFeature
{
    int presentCount = 0;
    bool IsActive() override { return true; }
    bool IsPaused() override { return false; }
    void Present() override { ++presentCount; }
    uint64_t FrameCount() override { return 0; }
};

// --- currentFeature double: upscaler timing answers empty ---
struct UpscalerFeatureStub
{
    std::optional<double> ReadUpscalerTime(ID3D11DeviceContext*) { return std::nullopt; }
    std::optional<double> ReadUpscalerTime(ID3D12CommandQueue*) { return std::nullopt; }
    void ReadDetailedGpuTimes(ID3D11DeviceContext*, int&) {}
    void ReadDetailedGpuTimes(ID3D12CommandQueue*, int&) {}
};

// --- State stub: every member FGPresent touches ---
struct State
{
    static State& Instance()
    {
        static State instance;
        return instance;
    }
    bool isShuttingDown = false;
    FGOutput activeFgOutput = FGOutput::NoFG;
    FGInput activeFgInput = FGInput::None;
    IFGFeature* currentFG = nullptr;
    IDXGISwapChain* currentFGSwapchain = nullptr;
    UpscalerFeatureStub* currentFeature = nullptr;
    ID3D12CommandQueue* currentCommandQueue = nullptr;
    ID3D11Device* currentD3D11Device = nullptr;
    ID3D12Device* currentD3D12Device = nullptr;
    int detailedGpuTimes = 0;
    unsigned dlssgDetectedInterpolationCount = 0;
    uint64_t fgLastFrame = 0;
    double lastFGFrameTime = 0.0;
    uint64_t frameCount = 0;
    SwapchainInteropApi swapchainInteropApi = SwapchainInteropApi::None;
    bool SCAllowTearing = false;
    bool realExclusiveFullscreen = false;
    bool fgPresentIsCalled = false;
    bool reflexLimitsFps = true; // skip the FrameLimit block; not under test
    GameEngineType gameEngine = GameEngineType::Unknown;
    std::mutex frameTimeMutex {};
    std::deque<double> upscaleTimes {};
};

namespace Util
{
inline double MillisecondsNow() { return 0.0; }
inline void GetDeviceRemovedReason(ID3D12Device*) {}
} // namespace Util

namespace ReflexHooks
{
inline bool gameIsSendingMarkers() { return false; }
} // namespace ReflexHooks

namespace StreamlineProxy
{
using MarkerFn = void (*)(sl::PCLMarker, sl::FrameToken&);
using TokenFn = sl::Result (*)(sl::FrameToken*&, const uint32_t*);
using SleepFn = void (*)(sl::FrameToken&);
inline MarkerFn PCLSetMarker() { return nullptr; }
inline TokenFn GetNewFrameToken() { return nullptr; }
inline SleepFn ReflexSleep() { return nullptr; }
} // namespace StreamlineProxy

namespace DlssNr
{
inline int genericApplyCount = 0;
inline void ApplyToFinishedPicture(IDXGISwapChain*, ID3D12CommandQueue*) { ++genericApplyCount; }
} // namespace DlssNr

namespace ResTrack_Dx12
{
inline void ClearPossibleHudless() {}
} // namespace ResTrack_Dx12

namespace Hudfix_Dx12
{
inline void PresentStart() {}
inline void PresentEnd() {}
} // namespace Hudfix_Dx12

namespace IdentifyGpu
{
struct GpuInfo
{
    bool usesDxvk = false;
};
inline GpuInfo getPrimaryGpu() { return {}; }
} // namespace IdentifyGpu

namespace XellHooks
{
inline bool canLimit() { return false; }
} // namespace XellHooks

namespace FrameLimit
{
inline void sleep(bool) {}
} // namespace FrameLimit

namespace XeFGProxy
{
using StatusFn = int (*)(xefg_swapchain_handle_t, xefg_swapchain_present_status_t*);
inline StatusFn GetLastPresentStatus() { return nullptr; }
} // namespace XeFGProxy

inline void ffxPresentCallback() {}
namespace FSR3FG
{
inline void ffxPresentCallback() {}
} // namespace FSR3FG

// --- FGHooks shell: statics the real body touches; FGPresent body is spliced in ---
typedef HRESULT(__stdcall* PFN_Present)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT(__stdcall* PFN_Present1)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);

struct FGHooks
{
    inline static UINT _lastPresentFlags = 0;
    inline static std::shared_mutex _resizeMutex {};
    inline static double _lastFGFrameTime = 0.0;
    inline static void* _semaphore = nullptr;
    inline static PFN_Present o_FGSCPresent = nullptr;
    inline static PFN_Present1 o_FGSCPresent1 = nullptr;

    static uint64_t XeFGHandoffSequence() { return 0; }
    static bool XeFGHandoffSince(uint64_t, uint64_t&, uint64_t&, void*&) { return false; }

    static HRESULT FGPresent(IDXGISwapChain* This, UINT SyncInterval, UINT Flags,
                             const DXGI_PRESENT_PARAMETERS* pPresentParameters);
};

// --- original-Present spies ---
struct PresentSpy
{
    int calls = 0;
    IDXGISwapChain* seenThis = nullptr;
    UINT seenSync = 0;
    UINT seenFlags = 0;
    bool seenParams = false;
    HRESULT result = S_OK;
};

static PresentSpy g_presentSpy;
static PresentSpy g_present1Spy;

static HRESULT __stdcall SpyPresent(IDXGISwapChain* This, UINT SyncInterval, UINT Flags)
{
    ++g_presentSpy.calls;
    g_presentSpy.seenThis = This;
    g_presentSpy.seenSync = SyncInterval;
    g_presentSpy.seenFlags = Flags;
    return g_presentSpy.result;
}

static HRESULT __stdcall SpyPresent1(IDXGISwapChain1* This, UINT SyncInterval, UINT Flags,
                                     const DXGI_PRESENT_PARAMETERS* pPresentParameters)
{
    ++g_present1Spy.calls;
    g_present1Spy.seenThis = This;
    g_present1Spy.seenSync = SyncInterval;
    g_present1Spy.seenFlags = Flags;
    g_present1Spy.seenParams = (pPresentParameters != nullptr);
    return g_present1Spy.result;
}

HRESULT FGHooks::FGPresent(IDXGISwapChain* This, UINT SyncInterval, UINT Flags,
                           const DXGI_PRESENT_PARAMETERS* pPresentParameters)
{
/*__FG_PRESENT_BODY__*/
}

namespace
{
int g_failures = 0;

void Pin(bool held, const char* marker, const char* detail)
{
    if (held)
        return;
    ++g_failures;
    std::printf("%s: %s\n", marker, detail);
}

IDXGISwapChain* ChainA() { return reinterpret_cast<IDXGISwapChain*>(0xA11CE); }
IDXGISwapChain* ChainB() { return reinterpret_cast<IDXGISwapChain*>(0xBEEF); }

void ResetWorld(LiveFeature& live, UpscalerFeatureStub& feat, SwapchainInteropApi interop)
{
    // NOTE: members reset individually; State holds a std::mutex and is not assignable.
    auto& state = State::Instance();
    state.isShuttingDown = false;
    state.activeFgOutput = FGOutput::XeFG;
    state.activeFgInput = FGInput::None;
    state.currentFG = &live;
    state.currentFGSwapchain = ChainB();
    state.currentFeature = &feat;
    state.currentCommandQueue = nullptr;
    state.currentD3D11Device = nullptr;
    state.currentD3D12Device = nullptr;
    state.detailedGpuTimes = 0;
    state.dlssgDetectedInterpolationCount = 0;
    state.fgLastFrame = 0;
    state.lastFGFrameTime = 0.0;
    state.swapchainInteropApi = interop;
    state.SCAllowTearing = false;
    state.realExclusiveFullscreen = false;
    state.fgPresentIsCalled = false;
    state.reflexLimitsFps = true;
    state.gameEngine = GameEngineType::Unknown;
    state.upscaleTimes.clear();
    live.presentCount = 0;
    DlssNr::genericApplyCount = 0;
    g_presentSpy = PresentSpy {};
    g_present1Spy = PresentSpy {};
    g_presentSpy.result = S_OK;
    g_present1Spy.result = (HRESULT) 0x87654321; // non-S_OK proves HRESULT passthrough
    FGHooks::o_FGSCPresent = &SpyPresent;
    FGHooks::o_FGSCPresent1 = &SpyPresent1;
    FGHooks::_semaphore = nullptr;
}
} // namespace

int main()
{
    LiveFeature live;
    UpscalerFeatureStub feat;

    // T1: stale incoming chain A (Present): original gets A once with HRESULT,
    // live feature zero, generic NR zero.
    {
        ResetWorld(live, feat, SwapchainInteropApi::None);
        g_presentSpy.result = (HRESULT) 0x87654320;
        const HRESULT hr = FGHooks::FGPresent(ChainA(), 1, 0, nullptr);
        Pin(g_presentSpy.calls == 1, "PIN_FG_NONOWNER",
            "stale present must forward the original exactly once");
        Pin(g_presentSpy.seenThis == ChainA(), "PIN_FG_NONOWNER",
            "stale present must forward the incoming chain unchanged");
        Pin(g_presentSpy.seenSync == 1 && g_presentSpy.seenFlags == 0, "PIN_FG_NONOWNER",
            "stale present must forward arguments unchanged");
        Pin(hr == g_presentSpy.result && hr == (HRESULT) 0x87654320, "PIN_FG_NONOWNER",
            "stale present must return the original HRESULT unchanged");
        Pin(live.presentCount == 0, "PIN_FG_NONOWNER",
            "stale present must never invoke the live feature");
        Pin(DlssNr::genericApplyCount == 0, "PIN_FG_NONOWNER",
            "stale present must never run generic NR");
        Pin(State::Instance().fgLastFrame == 0 && !State::Instance().fgPresentIsCalled,
            "PIN_FG_NONOWNER", "stale present must not account for an FG frame");
        Pin(g_present1Spy.calls == 0, "PIN_FG_NONOWNER",
            "stale Present must not touch the Present1 path");
    }

    // T2: stale incoming chain A (Present1): same contract on the Present1 path.
    {
        ResetWorld(live, feat, SwapchainInteropApi::None);
        DXGI_PRESENT_PARAMETERS params {};
        const HRESULT hr = FGHooks::FGPresent(ChainA(), 2, 0, &params);
        Pin(g_present1Spy.calls == 1, "PIN_FG_NONOWNER1",
            "stale Present1 must forward the original exactly once");
        Pin(g_present1Spy.seenThis == ChainA() && g_present1Spy.seenParams &&
                g_present1Spy.seenSync == 2 && g_present1Spy.seenFlags == 0,
            "PIN_FG_NONOWNER1",
            "stale Present1 must forward chain and parameters unchanged");
        Pin(hr == g_present1Spy.result && hr == (HRESULT) 0x87654321, "PIN_FG_NONOWNER1",
            "stale Present1 must return the original HRESULT unchanged");
        Pin(live.presentCount == 0, "PIN_FG_NONOWNER1",
            "stale Present1 must never invoke the live feature");
        Pin(DlssNr::genericApplyCount == 0, "PIN_FG_NONOWNER1",
            "stale Present1 must never run generic NR");
        Pin(State::Instance().fgLastFrame == 0 && !State::Instance().fgPresentIsCalled,
            "PIN_FG_NONOWNER1", "stale Present1 must not account for an FG frame");
        Pin(g_presentSpy.calls == 0, "PIN_FG_NONOWNER1",
            "stale Present1 must not touch the Present path");
    }

    // T3: registered chain B, native interop None: feature once, generic bypassed.
    {
        ResetWorld(live, feat, SwapchainInteropApi::None);
        const HRESULT hr = FGHooks::FGPresent(ChainB(), 1, 0, nullptr);
        Pin(g_presentSpy.calls == 1 && g_presentSpy.seenThis == ChainB(), "PIN_FG_OWNER_NATIVE",
            "owner present must forward the original once");
        Pin(hr == g_presentSpy.result, "PIN_FG_OWNER_NATIVE",
            "owner present must return the original HRESULT");
        Pin(live.presentCount == 1, "PIN_FG_OWNER_NATIVE",
            "owner present must invoke the live feature exactly once");
        Pin(DlssNr::genericApplyCount == 0, "PIN_FG_OWNER_NATIVE",
            "owner XeFG present must bypass generic NR");
    }

    // T4: registered chain B under Dx11wDx12 bridging: bridge routing still admitted.
    {
        ResetWorld(live, feat, SwapchainInteropApi::Dx11wDx12);
        const HRESULT hr = FGHooks::FGPresent(ChainB(), 1, 0, nullptr);
        Pin(g_presentSpy.calls == 1 && g_presentSpy.seenThis == ChainB(), "PIN_FG_OWNER_BRIDGE",
            "bridge owner present must forward the original once");
        Pin(hr == g_presentSpy.result, "PIN_FG_OWNER_BRIDGE",
            "bridge owner present must return the original HRESULT");
        Pin(live.presentCount == 1, "PIN_FG_OWNER_BRIDGE",
            "bridge owner present must invoke the live feature exactly once");
        Pin(DlssNr::genericApplyCount == 1, "PIN_FG_OWNER_BRIDGE",
            "bridge owner XeFG present retains upstream generic NR routing");
    }

    if (g_failures == 0)
        std::printf("FGPresent owner routing: all pins held.\n");
    return g_failures == 0 ? 0 : 1;
}
