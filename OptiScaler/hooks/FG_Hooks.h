#pragma once
#include "SysUtils.h"
#include <dxgi1_6.h>
#include <mutex>

#include "Hook_Utils.h"

class FGHooks
{
  public:
    static HRESULT CreateSwapChain(IDXGIFactory* pFactory, IUnknown* pDevice, DXGI_SWAP_CHAIN_DESC* pDesc,
                                   IDXGISwapChain** ppSwapChain);

    static HRESULT CreateSwapChainForHwnd(IDXGIFactory* This, IUnknown* pDevice, HWND hWnd,
                                          DXGI_SWAP_CHAIN_DESC1* pDesc,
                                          DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
                                          IDXGIOutput* pRestrictToOutput, IDXGISwapChain1** ppSwapChain);

    // Registers a real FG swapchain created through FGHooks::CreateSwapChain*().
    static void SetFGSwapchain(IDXGISwapChain* pSwapChain, HWND hWnd);

    // Registers a plain/external DX12 presenting swapchain used by DX11->DX12 interop fallback.
    // This must not mark State::currentFGSwapchain and must not install FG present hooks.
    static void SetDx12InteropPresentSC(IDXGISwapChain* pSwapChain, HWND hWnd);
    static void ClearDx12InteropPresentSC(IUnknown* pSwapChain);
    static bool IsDx12InteropPresentSC(IUnknown* pSwapChain);

    // XeFG app-facing colour-space carrier (NR/XeFG handoff).
    //
    // The XeFG app-facing proxy swapchain is the chain the game calls SetColorSpace1 on, so that call is
    // where the presentation encoding of the application picture becomes visible. hkSetColorSpace1
    // records it here and forwards the request to the original exactly once per distinct request; the NR
    // handoff consumes the record through GetFGColorSpaceCarrier(). The storage and the decision core
    // live in this header so they stay dependency-free and harness-drivable.
    using PFN_SetColorSpace1 = rewrite_signature<decltype(&IDXGISwapChain3::SetColorSpace1)>::type;

    struct FGColorSpaceCarrier
    {
        bool valid = false;                       // an owning-proxy request has been recorded
        HRESULT result = E_FAIL;                  // result of the forwarded original call
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN; // owning proxy buffer format when recorded
        DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        IDXGISwapChain3* proxy = nullptr; // app-facing proxy instance the request was made on (identity only)
        UINT64 epoch = 0;                 // bumped when the proxy is (re)created, released or resized
        UINT64 appliedEpoch = 0;          // epoch the record was applied in
        UINT64 forwarded = 0;             // requests forwarded to the original
        UINT64 absorbed = 0;              // identical repeats that were not forwarded

        // False once the recorded request predates the current proxy state (resize/reset).
        bool current() const { return valid && appliedEpoch == epoch; }
    };

    // Copies the recorded request out; returns false when nothing has been recorded yet.
    static bool GetFGColorSpaceCarrier(FGColorSpaceCarrier& out);

    // Drops the record and bumps the epoch (new or released proxy).
    static void ResetFGColorSpaceCarrier();

    // Bumps the epoch without dropping the record: the next identical request is forwarded again
    // (the owning proxy's buffers were rebuilt by a resize).
    static void InvalidateFGColorSpaceCarrierEpoch();

    // The one entry the detour handler uses: forwards colorSpace through forward exactly once unless it
    // is an identical repeat of the successful request recorded for the owning proxy in the current
    // epoch, or our own forward re-entered this function for the request already in flight. Instances
    // that are not the registered app-facing proxy are forwarded without being recorded. Public so a
    // harness can drive it with a swapchain/forward double.
    static HRESULT CarrySetColorSpace1(IDXGISwapChain3* proxy, DXGI_COLOR_SPACE_TYPE colorSpace,
                                       PFN_SetColorSpace1 forward, bool owningProxy);

    // XeFG owned NR handoff record (todo 10): the identity decided at the end of
    // XeFG_Dx12::Present() for the present in flight, with the XeFG swapchain context
    // that produced it. FGPresent snapshots the sequence before fg->Present() and reads
    // the record after the original proxy Present to emit NR_XEFG_PRESENT; the sequence
    // changes only when a handoff was decided, so a stale record is never reported for a
    // present that had no handoff.
    static void RecordXeFGHandoff(uint64_t generation, uint64_t frame, void* swapchainContext);
    static uint64_t XeFGHandoffSequence();
    static bool XeFGHandoffSince(uint64_t sequence, uint64_t& generation, uint64_t& frame, void*& swapchainContext);

  private:
    using PFN_Present = rewrite_signature<decltype(&IDXGISwapChain::Present)>::type;
    using PFN_Present1 = rewrite_signature<decltype(&IDXGISwapChain1::Present1)>::type;
    using PFN_SetFullscreenState = rewrite_signature<decltype(&IDXGISwapChain::SetFullscreenState)>::type;
    using PFN_GetFullscreenState = rewrite_signature<decltype(&IDXGISwapChain::GetFullscreenState)>::type;
    using PFN_GetFullscreenDesc = rewrite_signature<decltype(&IDXGISwapChain1::GetFullscreenDesc)>::type;
    using PFN_ResizeBuffers = rewrite_signature<decltype(&IDXGISwapChain::ResizeBuffers)>::type;
    using PFN_ResizeBuffers1 = rewrite_signature<decltype(&IDXGISwapChain3::ResizeBuffers1)>::type;
    using PFN_ResizeTarget = rewrite_signature<decltype(&IDXGISwapChain::ResizeTarget)>::type;
    using PFN_GetFrameLatencyWaitableObject =
        rewrite_signature<decltype(&IDXGISwapChain2::GetFrameLatencyWaitableObject)>::type;
    using PFN_Release = rewrite_signature<decltype(&IUnknown::Release)>::type;

    inline static PFN_ResizeBuffers o_FGSCResizeBuffers = nullptr;
    inline static PFN_ResizeTarget o_FGSCResizeTarget = nullptr;
    inline static PFN_ResizeBuffers1 o_FGSCResizeBuffers1 = nullptr;
    inline static PFN_SetFullscreenState o_FGSCSetFullscreenState = nullptr;
    inline static PFN_GetFullscreenState o_FGSCGetFullscreenState = nullptr;
    inline static PFN_GetFullscreenDesc o_FGSCGetFullscreenDesc = nullptr;
    inline static PFN_Present o_FGSCPresent = nullptr;
    inline static PFN_Present1 o_FGSCPresent1 = nullptr;
    inline static PFN_Release o_FGRelease = nullptr;
    inline static PFN_GetFrameLatencyWaitableObject o_FGSCGetFrameLatencyWaitableObject = nullptr;
    inline static PFN_SetColorSpace1 o_FGSCSetColorSpace1 = nullptr;
    inline static HWND _hwnd = nullptr;
    inline static IDXGISwapChain* _dx12InteropPresentSC = nullptr;
    inline static HWND _dx12InteropPresentHwnd = nullptr;
    inline static bool _skipResize = false;
    inline static bool _skipResize1 = false;
    inline static bool _skipPresent = false;
    inline static bool _skipPresent1 = false;
    inline static UINT _lastPresentFlags = 0;
    inline static double _lastFGFrameTime = -1.0;
    inline static std::shared_mutex _resizeMutex;

    inline static FGColorSpaceCarrier _colorSpaceCarrier {};
    inline static UINT64 _colorSpaceCarrierEpoch = 0;
    inline static std::mutex _colorSpaceCarrierMutex;
    inline static thread_local bool _colorSpaceInFlight = false;
    inline static thread_local IDXGISwapChain3* _colorSpaceInFlightProxy = nullptr;
    inline static thread_local DXGI_COLOR_SPACE_TYPE _colorSpaceInFlightColorSpace =
        DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;

    inline static std::mutex _xefgHandoffMutex;
    inline static uint64_t _xefgHandoffSeq = 0;
    inline static uint64_t _xefgHandoffGeneration = 0;
    inline static uint64_t _xefgHandoffFrame = 0;
    inline static void* _xefgHandoffContext = nullptr;

    static void HookFGSwapchain(IDXGISwapChain* pSwapChain);

    static HRESULT hkSetFullscreenState(IDXGISwapChain* This, BOOL Fullscreen, IDXGIOutput* pTarget);
    static HRESULT hkGetFullscreenDesc(IDXGISwapChain1* This, DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pDesc);
    static HRESULT hkGetFullscreenState(IDXGISwapChain* This, BOOL* pFullscreen, IDXGIOutput** ppTarget);
    static HRESULT hkSetColorSpace1(IDXGISwapChain3* This, DXGI_COLOR_SPACE_TYPE ColorSpace);
    static HRESULT hkResizeBuffers(IDXGISwapChain* This, UINT BufferCount, UINT Width, UINT Height,
                                   DXGI_FORMAT NewFormat, UINT SwapChainFlags);
    static HRESULT hkResizeTarget(IDXGISwapChain* This, const DXGI_MODE_DESC* pNewTargetParameters);
    static HRESULT hkResizeBuffers1(IDXGISwapChain3* This, UINT BufferCount, UINT Width, UINT Height,
                                    DXGI_FORMAT Format, UINT SwapChainFlags, const UINT* pCreationNodeMask,
                                    IUnknown* const* ppPresentQueue);
    static ULONG hkFGRelease(IUnknown* This);

    static HRESULT hkFGPresent(IDXGISwapChain* This, UINT SyncInterval, UINT Flags);
    static HRESULT hkFGPresent1(IDXGISwapChain1* This, UINT SyncInterval, UINT Flags,
                                const DXGI_PRESENT_PARAMETERS* pPresentParameters);
    static HRESULT FGPresent(IDXGISwapChain* This, UINT SyncInterval, UINT Flags,
                             const DXGI_PRESENT_PARAMETERS* pPresentParameters);

    static HANDLE hkGetFrameLatencyWaitableObject(IDXGISwapChain2* This);

    VALIDATE_MEMBER_HOOK(hkFGPresent, PFN_Present)
    VALIDATE_MEMBER_HOOK(hkFGPresent1, PFN_Present1)
    VALIDATE_MEMBER_HOOK(hkSetFullscreenState, PFN_SetFullscreenState)
    VALIDATE_MEMBER_HOOK(hkGetFullscreenState, PFN_GetFullscreenState)
    VALIDATE_MEMBER_HOOK(hkGetFullscreenDesc, PFN_GetFullscreenDesc)
    VALIDATE_MEMBER_HOOK(hkResizeBuffers, PFN_ResizeBuffers)
    VALIDATE_MEMBER_HOOK(hkResizeBuffers1, PFN_ResizeBuffers1)
    VALIDATE_MEMBER_HOOK(hkResizeTarget, PFN_ResizeTarget)
    VALIDATE_MEMBER_HOOK(hkGetFrameLatencyWaitableObject, PFN_GetFrameLatencyWaitableObject)
    VALIDATE_MEMBER_HOOK(hkSetColorSpace1, PFN_SetColorSpace1)
    VALIDATE_MEMBER_HOOK(hkFGRelease, PFN_Release)
};

// XeFG app-facing colour-space carrier - defined here (not in the .cpp) so the decision core stays
// dependency-free and harness-drivable; FG_Hooks.cpp only wires the detour into CarrySetColorSpace1().
inline bool FGHooks::GetFGColorSpaceCarrier(FGColorSpaceCarrier& out)
{
    std::lock_guard<std::mutex> lock(_colorSpaceCarrierMutex);
    out = _colorSpaceCarrier;
    return out.valid;
}

inline void FGHooks::ResetFGColorSpaceCarrier()
{
    std::lock_guard<std::mutex> lock(_colorSpaceCarrierMutex);
    _colorSpaceCarrier = FGColorSpaceCarrier {};
    _colorSpaceCarrier.epoch = ++_colorSpaceCarrierEpoch;
}

inline void FGHooks::InvalidateFGColorSpaceCarrierEpoch()
{
    std::lock_guard<std::mutex> lock(_colorSpaceCarrierMutex);
    _colorSpaceCarrier.epoch = ++_colorSpaceCarrierEpoch;
}

inline HRESULT FGHooks::CarrySetColorSpace1(IDXGISwapChain3* proxy, DXGI_COLOR_SPACE_TYPE colorSpace,
                                            PFN_SetColorSpace1 forward, bool owningProxy)
{
    if (forward == nullptr)
        return E_UNEXPECTED;

    // Our own forward can arrive back here: the detoured class code is shared by every instance of the
    // class and the proxy can re-enter while applying the request. The request in flight is being
    // forwarded exactly once; forwarding it again would recurse into the same call.
    if (owningProxy && _colorSpaceInFlight && _colorSpaceInFlightProxy == proxy &&
        _colorSpaceInFlightColorSpace == colorSpace)
    {
        LOG_TRACE("FG colour-space carrier: re-entrant SetColorSpace1({}) on {:X} not forwarded", (UINT) colorSpace,
                  (size_t) proxy);
        return S_OK;
    }

    {
        std::lock_guard<std::mutex> lock(_colorSpaceCarrierMutex);

        // Identical repeat of a request this instance already applied successfully in the current epoch:
        // the swapchain state is already set, so forward nothing and keep the single record. A different
        // colour space, a failed previous attempt, another instance or a bumped epoch all forward again.
        if (owningProxy && _colorSpaceCarrier.valid && SUCCEEDED(_colorSpaceCarrier.result) &&
            _colorSpaceCarrier.proxy == proxy && _colorSpaceCarrier.colorSpace == colorSpace &&
            _colorSpaceCarrier.appliedEpoch == _colorSpaceCarrier.epoch)
        {
            _colorSpaceCarrier.absorbed++;
            LOG_TRACE("FG colour-space carrier: duplicate SetColorSpace1({}) on {:X} absorbed", (UINT) colorSpace,
                      (size_t) proxy);
            return _colorSpaceCarrier.result;
        }
    }

    const bool inFlight = _colorSpaceInFlight;
    IDXGISwapChain3* const inFlightProxy = _colorSpaceInFlightProxy;
    const DXGI_COLOR_SPACE_TYPE inFlightColorSpace = _colorSpaceInFlightColorSpace;

    _colorSpaceInFlight = true;
    _colorSpaceInFlightProxy = proxy;
    _colorSpaceInFlightColorSpace = colorSpace;
    const HRESULT result = forward(proxy, colorSpace);
    _colorSpaceInFlight = inFlight;
    _colorSpaceInFlightProxy = inFlightProxy;
    _colorSpaceInFlightColorSpace = inFlightColorSpace;

    if (owningProxy)
    {
        DXGI_SWAP_CHAIN_DESC desc {};
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        if (proxy->GetDesc(&desc) == S_OK)
            format = desc.BufferDesc.Format;

        {
            std::lock_guard<std::mutex> lock(_colorSpaceCarrierMutex);
            _colorSpaceCarrier.valid = true;
            _colorSpaceCarrier.result = result;
            _colorSpaceCarrier.format = format;
            _colorSpaceCarrier.colorSpace = colorSpace;
            _colorSpaceCarrier.appliedEpoch = _colorSpaceCarrier.epoch;
            _colorSpaceCarrier.forwarded++;
            _colorSpaceCarrier.proxy = proxy;
        }

        LOG_INFO("FG colour-space carrier: proxy {:X}, format {}, colour space {}, result {:X}", (size_t) proxy,
                 (UINT) format, (UINT) colorSpace, (UINT) result);
    }

    return result;
}

// XeFG owned NR handoff record - inline for the same reason as the carrier: the record
// stays dependency-free and harness-drivable; FG_Hooks.cpp only wires it into the
// Present path.
inline void FGHooks::RecordXeFGHandoff(uint64_t generation, uint64_t frame, void* swapchainContext)
{
    std::lock_guard<std::mutex> lock(_xefgHandoffMutex);
    _xefgHandoffGeneration = generation;
    _xefgHandoffFrame = frame;
    _xefgHandoffContext = swapchainContext;
    ++_xefgHandoffSeq;
}

inline uint64_t FGHooks::XeFGHandoffSequence()
{
    std::lock_guard<std::mutex> lock(_xefgHandoffMutex);
    return _xefgHandoffSeq;
}

inline bool FGHooks::XeFGHandoffSince(uint64_t sequence, uint64_t& generation, uint64_t& frame, void*& swapchainContext)
{
    std::lock_guard<std::mutex> lock(_xefgHandoffMutex);
    if (_xefgHandoffSeq == sequence)
        return false; // no handoff was decided since the snapshot
    generation = _xefgHandoffGeneration;
    frame = _xefgHandoffFrame;
    swapchainContext = _xefgHandoffContext;
    return true;
}
