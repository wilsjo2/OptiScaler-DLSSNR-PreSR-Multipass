// Exercise the production hook wrappers with two fake plugins and real DXGI queue metadata.
#include <windows.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include "../OptiScaler/dlssnr/DlssNr_StreamlinePicture.h"
using Microsoft::WRL::ComPtr;
static void Check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D call failed"); }
static void Expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Setting { bool value = true; bool value_or_default() const { return value; } };
struct Config
{
    Setting DlssNrEnabled, DlssNrFinishedPicture;
    static Config* Instance() { static Config cfg; return &cfg; }
};
enum class GameQuirk { Kcd2NrBeforeFg };
enum class FGNvngxReplacement { None, Other };
struct Quirks { bool enabled = false; bool operator[](GameQuirk) const { return enabled; } };
struct State
{
    bool isShuttingDown = false;
    Quirks gameQuirks;
    FGNvngxReplacement activeFgNvngx = FGNvngxReplacement::None;
    static State& Instance() { static State state; return state; }
};
namespace Util { bool CheckForRealObject(const char*, IUnknown*, IUnknown**) { return false; } }
#define LOG_INFO(...) ((void)0)
static ID3D12Resource* expectedPicture;
static ID3D12CommandQueue* expectedQueue;
static unsigned edits = 0;
static bool expectPq = false;
// DlssNrFeature_Dx12.h's private-data key; avoid pulling NGX/application headers into the harness.
static constexpr GUID colorSpaceKey = { 0x34a31e7b, 0x84c5, 0x44ef, { 0xa7, 0x4d, 0x6b, 0xd3, 0x60, 0x8c, 0xe5, 0x22 } };
namespace DlssNr
{
void ApplyToStreamlinePicture(IDXGISwapChain* swapchain, ID3D12Resource* picture, ID3D12CommandQueue* queue)
{
    Expect(picture == expectedPicture, "Mixed local and game plugin buffers");
    Expect(queue == expectedQueue, "Used the presentation queue instead of the render queue");
    if (expectPq)
    {
        DXGI_COLOR_SPACE_TYPE space {}; UINT size = sizeof(space);
        Check(swapchain->GetPrivateData(colorSpaceKey, &size, &space));
        Expect(space == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, "Lost PQ metadata at handoff");
        Expect(picture->GetDesc().Format == DXGI_FORMAT_R10G10B10A2_UNORM, "Lost 10-bit application buffer");
    }
    ++edits;
}
}
// The runner copies the production body unchanged, replacing only application includes with these seams.
#include "StreamlinePicture.cpp"

template <unsigned Runtime> struct Plugin
{
    inline static ComPtr<IDXGISwapChain1> chain;
    inline static ComPtr<ID3D12Resource> picture;
    inline static unsigned calls = 0, expectedEdits = 0;
    inline static bool handled = true;
    static UINT Index(IDXGISwapChain*, bool& skip) { skip = handled; return Runtime; }
    static HRESULT Buffer(IDXGISwapChain*, UINT index, REFIID iid, void** output, bool& skip)
    {
        Expect(index == Runtime, "Called another plugin's index hook");
        skip = handled;
        return picture->QueryInterface(iid, output);
    }
    static HRESULT Present(IDXGISwapChain*, UINT, UINT, bool&)
    {
        Expect(edits == expectedEdits, "NR did not run before FG consumed the picture");
        ++calls;
        return S_OK;
    }
    static HRESULT Present1(IDXGISwapChain* sc, UINT interval, UINT flags, const DXGI_PRESENT_PARAMETERS*, bool& skip)
    { return Present(sc, interval, flags, skip); }
    static HRESULT Create(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain** out, bool& skip)
    { skip = handled; return chain.CopyTo(out); }
    static HRESULT CreateHwnd(IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                              const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1** out, bool& skip)
    { skip = handled; return chain.CopyTo(out); }
    static void* Get(const char* name)
    {
        if (!strcmp(name, "slHookPresent")) return &Present;
        if (!strcmp(name, "slHookPresent1")) return &Present1;
        if (!strcmp(name, "slHookCreateSwapChain")) return &Create;
        if (!strcmp(name, "slHookCreateSwapChainForHwnd")) return &CreateHwnd;
        if (!strcmp(name, "slHookGetCurrentBackBufferIndex")) return &Index;
        if (!strcmp(name, "slHookGetBuffer")) return &Buffer;
        return nullptr;
    }
};
int main() try
{
    ComPtr<IDXGIFactory4> factory; ComPtr<IDXGIAdapter> adapter; ComPtr<ID3D12Device> device;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
    Check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC q {};
    Check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
    expectedQueue = queue.Get();
    DXGI_SWAP_CHAIN_DESC1 desc {};
    desc.Width = desc.Height = 8; desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1; desc.BufferCount = 2; desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    Check(factory->CreateSwapChainForComposition(queue.Get(), &desc, nullptr, &Plugin<0>::chain));
    desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    Check(factory->CreateSwapChainForComposition(queue.Get(), &desc, nullptr, &Plugin<1>::chain));
    Check(Plugin<0>::chain->GetBuffer(0, IID_PPV_ARGS(&Plugin<0>::picture)));
    Check(Plugin<1>::chain->GetBuffer(1, IID_PPV_ARGS(&Plugin<1>::picture)));

    using namespace DlssNr::StreamlinePicture;
    Expect(!Wrap("slHookPresent", Plugin<1>::Get, true), "Local wrapping enabled without KCD2 quirk");
    auto nativeCreateLegacy = reinterpret_cast<decltype(&Plugin<0>::Create)>(Wrap("slHookCreateSwapChain", Plugin<0>::Get));
    State::Instance().gameQuirks.enabled = true;
    State::Instance().activeFgNvngx = FGNvngxReplacement::Other;
    Expect(!Wrap("slHookPresent", Plugin<1>::Get, true), "Local wrapping enabled for replacement FG");
    State::Instance().activeFgNvngx = FGNvngxReplacement::None;
    auto nativePresent = reinterpret_cast<decltype(&Plugin<0>::Present)>(Wrap("slHookPresent", Plugin<0>::Get));
    auto localPresent = reinterpret_cast<decltype(&Plugin<1>::Present1)>(Wrap("slHookPresent1", Plugin<1>::Get, true));
    auto localPresentLegacy = reinterpret_cast<decltype(&Plugin<1>::Present)>(Wrap("slHookPresent", Plugin<1>::Get, true));
    auto nativeCreate = reinterpret_cast<decltype(&Plugin<0>::CreateHwnd)>(Wrap("slHookCreateSwapChainForHwnd", Plugin<0>::Get));
    auto localCreate = reinterpret_cast<decltype(&Plugin<1>::Create)>(Wrap("slHookCreateSwapChain", Plugin<1>::Get, true));
    Expect(nativePresent && localPresent && localPresentLegacy && nativeCreate && nativeCreateLegacy && localCreate, "Missing presentation/creation wrapper");
    bool skip = false;
    ComPtr<IDXGISwapChain> nativeLegacyChain;
    Plugin<0>::handled = false;
    Check(nativeCreateLegacy(nullptr, queue.Get(), nullptr, &nativeLegacyChain, skip));
    Expect(!RenderQueue(nativeLegacyChain.Get()), "Unhandled legacy create registered a render queue");
    nativeLegacyChain.Reset();
    Plugin<0>::handled = true;
    skip = false;
    Check(nativeCreateLegacy(nullptr, queue.Get(), nullptr, &nativeLegacyChain, skip));
    Expect(RenderQueue(nativeLegacyChain.Get()).Get() == queue.Get(), "Native legacy create lost the render queue");
    expectedPicture = Plugin<0>::picture.Get();
    Plugin<0>::expectedEdits = edits + 1;
    skip = false;
    Check(nativePresent(nativeLegacyChain.Get(), 0, 0, skip));
    Check(nativeLegacyChain->SetPrivateDataInterface(renderQueueKey, nullptr));
    ComPtr<IDXGISwapChain1> nativeChain;
    Check(nativeCreate(nullptr, queue.Get(), nullptr, nullptr, nullptr, nullptr, &nativeChain, skip));
    ComPtr<IDXGISwapChain> localChain;
    skip = false;
    Check(localCreate(nullptr, queue.Get(), nullptr, &localChain, skip));
    Expect(RenderQueue(nativeChain.Get()).Get() == queue.Get(), "Hwnd create lost the render queue");
    Expect(RenderQueue(localChain.Get()).Get() == queue.Get(), "Legacy create lost the render queue");
    const auto pq = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    Check(localChain->SetPrivateData(colorSpaceKey, sizeof(pq), &pq));
    for (unsigned frame = 0; frame < 6; ++frame)
    {
        expectPq = false;
        skip = false; expectedPicture = Plugin<0>::picture.Get(); Plugin<0>::expectedEdits = edits + 1;
        Check(nativePresent(nativeChain.Get(), 0, 0, skip));
        expectPq = true;
        skip = false; expectedPicture = Plugin<1>::picture.Get(); Plugin<1>::expectedEdits = edits + 1;
        if (frame % 2) Check(localPresent(localChain.Get(), 0, 0, nullptr, skip));
        else Check(localPresentLegacy(localChain.Get(), 0, 0, skip));
    }
    Plugin<1>::expectedEdits = edits;
    skip = false; Check(localPresent(localChain.Get(), 0, DXGI_PRESENT_TEST, nullptr, skip));
    skip = true; Check(localPresent(localChain.Get(), 0, 0, nullptr, skip));
    skip = false; Config::Instance()->DlssNrEnabled.value = false;
    Check(localPresent(localChain.Get(), 0, 0, nullptr, skip));
    Config::Instance()->DlssNrEnabled.value = true;
    Config::Instance()->DlssNrFinishedPicture.value = false;
    Check(localPresent(localChain.Get(), 0, 0, nullptr, skip));
    Config::Instance()->DlssNrFinishedPicture.value = true;
    Plugin<1>::handled = false;
    Check(localPresent(localChain.Get(), 0, 0, nullptr, skip));
    Plugin<1>::handled = true;
    State::Instance().isShuttingDown = true;
    Check(localPresent(localChain.Get(), 0, 0, nullptr, skip));
    State::Instance().isShuttingDown = false;
    Expect(Plugin<0>::calls == 7 && Plugin<1>::calls == 12, "Lost or duplicated an original present");
    puts("PASS: KCD2-only local FG gate; local/native legacy + native Hwnd registration; separate plugin buffers; "
         "PQ/10-bit metadata; edit before FG; skip/disabled/shutdown guards");
    return 0;
}
catch (const std::exception& error) { puts(error.what()); return 1; }
