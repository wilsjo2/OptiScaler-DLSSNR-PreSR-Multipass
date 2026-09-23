#pragma once

#include <framegen/IFGFeature_Dx12.h>

#include <proxies/XeLL_Proxy.h>
#include <proxies/XeFG_Proxy.h>

#include <dlssnr/DlssNr_XeFGHandoff.h>
#include <hooks/FG_Hooks.h>

#include "shaders/depth_invert/DI_Dx12.h"

#include <xell.h>
#include <xell_d3d12.h>
#include <xefg_swapchain.h>
#include <xefg_swapchain_d3d12.h>
#include <xefg_swapchain_debug.h>

class XeFG_Dx12 : public virtual IFGFeature_Dx12
{
  private:
    xefg_swapchain_handle_t _swapChainContext = nullptr;
    xefg_swapchain_handle_t _fgContext = nullptr;

    uint32_t _width = 0;
    uint32_t _height = 0;
    bool _infiniteDepth = false;
    std::optional<bool> _haveHudless = std::nullopt;
    bool _uiComposition = false;

    // Creation format of the app-facing proxy, used by the handoff colour gate.
    DXGI_FORMAT _proxyFormat = DXGI_FORMAT_UNKNOWN;

    // Owned NR handoff (nr-xefg-088-release, todo 10): the finished application picture is
    // composed once per accepted application frame at the end of Present() on the
    // XeFG-retained application queue. The decision core is the landed seam core
    // (dlssnr/DlssNr_XeFGHandoff.h): the generation changes on swapchain recreation and FG
    // discontinuity, and the accepted frame id is _lastDispatchedFrame (never GetDispatchIndex).
    DlssNr::XeFGHandoff::Tracker _nrHandoff;
    uint64_t _nrGeneration = 0;
    void OwnedNrHandoff();
    void ResetNrHandoff();

    std::unique_ptr<DI_Dx12> _depthInvert;

    static void xefgLogCallback(const char* message, xefg_swapchain_logging_level_t level, void* userData);

    bool CreateSwapchainContext(ID3D12Device* device);
    bool DestroySwapchainContext();
    xefg_swapchain_d3d12_resource_data_t GetResourceData(FG_ResourceType type, int index = -1);

    bool Dispatch();

  protected:
    void ReleaseObjects() override final;
    void CreateObjects(ID3D12Device* InDevice) override final;

  public:
    // IFGFeature
    const char* Name() override final;
    feature_version Version() override final;
    HWND Hwnd() override final;

    // IFGFeature_Dx12
    bool CreateSwapchain(IDXGIFactory* factory, ID3D12CommandQueue* cmdQueue, DXGI_SWAP_CHAIN_DESC* desc,
                         IDXGISwapChain** swapChain, bool readyToRelease) override final;
    bool CreateSwapchain1(IDXGIFactory* factory, ID3D12CommandQueue* cmdQueue, HWND hwnd, DXGI_SWAP_CHAIN_DESC1* desc,
                          DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc, IDXGISwapChain1** swapChain,
                          bool readyToRelease) override final;

    bool ReleaseSwapchain(HWND hwnd) override final;

    void CreateContext(ID3D12Device* device, FG_Constants& fgConstants) override final;
    void Activate() override final;
    void Deactivate() override final;
    void DestroyFGContext() override final;
    bool Shutdown() override final;

    void EvaluateState(ID3D12Device* device, FG_Constants& fgConstants) override final;

    bool Present() override final;

    bool SetResource(Dx12Resource* inputResource) override final;
    void SetCommandQueue(FG_ResourceType type, ID3D12CommandQueue* queue) override final;

    void* FrameGenerationContext() override final;
    void* SwapchainContext() override final;

    XeFG_Dx12() : IFGFeature_Dx12(), IFGFeature()
    {
        if (XeFGProxy::Module() == nullptr)
            XeFGProxy::InitXeFG();
    }

    ~XeFG_Dx12();

    // Inherited via IFGFeature_Dx12
    bool SetInterpolatedFrameCount(UINT interpolatedFrameCount) override;
};
