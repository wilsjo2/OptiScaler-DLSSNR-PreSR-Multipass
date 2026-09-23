#include "pch.h"
#include "DlssNr_StreamlinePicture.h"
#include "DlssNrFeature_Dx12.h"
#include <Config.h>
#include <State.h>
#include <Util.h>
#include <cstring>

namespace DlssNr::StreamlinePicture
{
namespace
{
using Present = HRESULT (*)(IDXGISwapChain*, UINT, UINT, bool&);
using Present1 = HRESULT (*)(IDXGISwapChain*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*, bool&);
using CreateHwnd = HRESULT (*)(IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                               const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**, bool&);
using Create = HRESULT (*)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**, bool&);

void RegisterQueue(HRESULT result, bool handled, IUnknown* device, IDXGISwapChain* swapchain)
{
    if (State::Instance().isShuttingDown)
        return;
    // Keep the application's render queue, not DLSSG's asynchronous presentation queue.
    if (FAILED(result) || !handled || !device || !swapchain)
        return;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&queue))))
    {
        ID3D12CommandQueue* real = nullptr;
        if (Util::CheckForRealObject(__FUNCTION__, queue.Get(), (IUnknown**) &real))
            queue = real;
        if (SUCCEEDED(swapchain->SetPrivateDataInterface(renderQueueKey, queue.Get())))
            LOG_INFO("DLSS-NR: Streamline finished-picture handoff registered on the game render queue");
    }
}

// The game's and OptiScaler's plugins can coexist. Never call one runtime's hooks on the other's buffers.
template <bool Local> struct Hooks
{
    inline static Present originalPresent = nullptr;
    inline static Present1 originalPresent1 = nullptr;
    inline static CreateHwnd originalCreateHwnd = nullptr;
    inline static Create originalCreate = nullptr;
    inline static GetIndex getIndex = nullptr;
    inline static GetBuffer getBuffer = nullptr;

    static void Apply(IDXGISwapChain* swapchain, UINT flags, bool skip)
    {
        if (State::Instance().isShuttingDown)
            return;
        const auto& cfg = *Config::Instance();
        if (skip || (flags & DXGI_PRESENT_TEST) || !cfg.DlssNrEnabled.value_or_default() ||
            !cfg.DlssNrFinishedPicture.value_or_default())
            return;
        auto queue = RenderQueue(swapchain);
        if (!queue)
            return;
        auto picture = Read(swapchain, getIndex, getBuffer);
        if (!picture)
            return;
        ApplyToStreamlinePicture(swapchain, picture.Get(), queue.Get());
    }

    static HRESULT BeforePresent(IDXGISwapChain* swapchain, UINT interval, UINT flags, bool& skip)
    {
        Apply(swapchain, flags, skip);
        return originalPresent(swapchain, interval, flags, skip);
    }
    static HRESULT BeforePresent1(IDXGISwapChain* swapchain, UINT interval, UINT flags,
                                  const DXGI_PRESENT_PARAMETERS* parameters, bool& skip)
    {
        Apply(swapchain, flags, skip);
        return originalPresent1(swapchain, interval, flags, parameters, skip);
    }
    static HRESULT CreateForHwnd(IDXGIFactory2* factory, IUnknown* device, HWND window,
                                 const DXGI_SWAP_CHAIN_DESC1* desc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
                                 IDXGIOutput* output, IDXGISwapChain1** swapchain, bool& skip)
    {
        const auto result = originalCreateHwnd(factory, device, window, desc, fullscreen, output, swapchain, skip);
        if (SUCCEEDED(result) && skip && swapchain)
            RegisterQueue(result, skip, device, *swapchain);
        return result;
    }
    static HRESULT CreateSwapChain(IDXGIFactory* factory, IUnknown* device, DXGI_SWAP_CHAIN_DESC* desc,
                                   IDXGISwapChain** swapchain, bool& skip)
    {
        const auto result = originalCreate(factory, device, desc, swapchain, skip);
        if (SUCCEEDED(result) && skip && swapchain)
            RegisterQueue(result, skip, device, *swapchain);
        return result;
    }

    static void* Wrap(const char* name, GetFunction getFunction)
    {
        if (!getFunction)
            return nullptr;
        const bool present = std::strcmp(name, "slHookPresent") == 0;
        const bool present1 = std::strcmp(name, "slHookPresent1") == 0;
        const bool createHwnd = std::strcmp(name, "slHookCreateSwapChainForHwnd") == 0;
        const bool create = Local && std::strcmp(name, "slHookCreateSwapChain") == 0;
        if (!present && !present1 && !createHwnd && !create)
            return nullptr;
        auto* function = getFunction(name);
        getIndex = reinterpret_cast<GetIndex>(getFunction("slHookGetCurrentBackBufferIndex"));
        getBuffer = reinterpret_cast<GetBuffer>(getFunction("slHookGetBuffer"));
        if (!function || !getIndex || !getBuffer || !getFunction("slHookPresent") || !getFunction("slHookPresent1"))
            return nullptr;
        if (present)
        {
            originalPresent = reinterpret_cast<Present>(function);
            return &BeforePresent;
        }
        if (present1)
        {
            originalPresent1 = reinterpret_cast<Present1>(function);
            return &BeforePresent1;
        }
        if (createHwnd)
        {
            originalCreateHwnd = reinterpret_cast<CreateHwnd>(function);
            return &CreateForHwnd;
        }
        originalCreate = reinterpret_cast<Create>(function);
        return &CreateSwapChain;
    }
};
} // namespace

void* Wrap(const char* name, GetFunction getFunction, bool local)
{
    if (local && (!State::Instance().gameQuirks[GameQuirk::Kcd2NrBeforeFg] ||
                  State::Instance().activeFgNvngx != FGNvngxReplacement::None))
        return nullptr;
    return local ? Hooks<true>::Wrap(name, getFunction) : Hooks<false>::Wrap(name, getFunction);
}
} // namespace DlssNr::StreamlinePicture
