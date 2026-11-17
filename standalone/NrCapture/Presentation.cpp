#include "Presentation.h"
#include "Colour.h"
#include <dwmapi.h>
namespace nr {
namespace {
LRESULT CALLBACK WindowProcedure(HWND w, UINT m, WPARAM p, LPARAM l) {
    if (m == WM_MOUSEACTIVATE) return MA_NOACTIVATE;
    if (m == WM_NCHITTEST) return HTTRANSPARENT;
    if (m == WM_ERASEBKGND) return 1;
    return DefWindowProcW(w, m, p, l);
}
}
Presentation::Presentation(Gpu& g, CaptureSource captureSource, HWND controlWindow, FrameGeneration* frameGeneration, unsigned factor)
    : gpu(g), source(captureSource), controls(controlWindow), fg(frameGeneration), multiplier(factor) {
    WNDCLASSW wc{}; wc.lpfnWndProc = WindowProcedure; wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"OptiScalerNR.Output";
    if (!RegisterClassW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        Check(HRESULT_FROM_WIN32(GetLastError()), "Register output window");
    window = CreateWindowExW(WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_TRANSPARENT,
        wc.lpszClassName, L"Display Filter output", WS_POPUP, 0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr);
    if (!window) Check(HRESULT_FROM_WIN32(GetLastError()), "Output window");
    // Monitor capture must never feed the processed output back into NR.
    if (source.monitor && !SetWindowDisplayAffinity(window, WDA_EXCLUDEFROMCAPTURE)) {
        const auto error = HRESULT_FROM_WIN32(GetLastError()); DestroyWindow(window); window = nullptr;
        Check(error, "Exclude overlay from screen capture");
    }
    Check(SetLayeredWindowAttributes(window, 0, 255, LWA_ALPHA) ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "Click-through output");
    if (fg) { fgOutput = std::make_unique<FrameGenerationOutput>(g); return; }
    Check(DCompositionCreateDevice(nullptr, IID_PPV_ARGS(&composition)), "Composition device");
    Check(composition->CreateTargetForHwnd(window, TRUE, &target), "Composition target");
    Check(composition->CreateVisual(&visual), "Composition visual");
    Check(target->SetRoot(visual.Get()), "Composition root");
}
Presentation::~Presentation() {
    try { Hide(); } catch (const std::exception& e) { Log(e.what()); }
    swapchain.Reset(); if (window) DestroyWindow(window);
}
void Presentation::Hide() {
    if (fg) { fg->Quiesce(swapchain.Get()); fgOutput->Reset(); }
    if (window && IsWindowVisible(window)) ShowWindow(window, SW_HIDE);
}
void Presentation::RefreshVisibility() {
    if (!Active()) { Hide(); return; }
    if (!IsWindowVisible(window)) return;
    RECT bounds{}, previous{}; GetWindowRect(window, &previous);
    if (!source.Bounds(bounds) || (UINT)(bounds.right - bounds.left) != width ||
        (UINT)(bounds.bottom - bounds.top) != height) { Hide(); return; }
    if (bounds.left != previous.left || bounds.top != previous.top) {
        if (fg) fg->Quiesce(swapchain.Get());
        SetWindowPos(window, nullptr, bounds.left, bounds.top, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
    }
    Order();
}
bool Presentation::Active() const {
    // The filter stays on while the user operates our controls; Order keeps them above the output.
    return source.Active() || (ControlsFocused() && source.Shown());
}
void Presentation::Order() {
    // Topmost over a focused source. While our controls have focus the output sits directly above the
    // source window (screen mode: directly below the controls), so whatever covers the source covers it too.
    HWND after = HWND_TOPMOST;
    if (ControlsFocused()) {
        after = source.window ? GetWindow(source.window, GW_HWNDPREV) : controls;
        if (source.window ? after == window : GetWindow(controls, GW_HWNDNEXT) == window) return;
        if (!after) after = HWND_TOP;
    } else if (fg && (GetWindowLongPtrW(window, GWL_EXSTYLE) & WS_EX_TOPMOST)) return;
    // Without FG the output is raised every frame, as before. FG leaves the HWND alone unless it must move.
    if (fg) fg->Quiesce(swapchain.Get());
    SetWindowPos(window, after, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
}
bool Presentation::Hdr() const {
    const HMONITOR monitor = source.Monitor();
    for (UINT i = 0; ; ++i) {
        ComPtr<IDXGIOutput> output; if (gpu.adapter->EnumOutputs(i, &output) == DXGI_ERROR_NOT_FOUND) break;
        ComPtr<IDXGIOutput6> extended; if (FAILED(output.As(&extended))) continue;
        DXGI_OUTPUT_DESC1 d{}; if (SUCCEEDED(extended->GetDesc1(&d)) && d.Monitor == monitor)
            return d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    }
    return false;
}
float Presentation::WhitePoint(bool hdr) {
    if (!hdr) { whiteMonitor = nullptr; return 1.0f; }
    const auto monitor = source.Monitor(); const auto now = GetTickCount64();
    if (monitor != whiteMonitor || now - whiteQueriedAt >= 1000) {
        const auto white = DesktopWhitePoint(monitor);
        desktopWhite = white > 0 ? white : 203.0f / 80.0f;
        whiteMonitor = monitor; whiteQueriedAt = now;
    }
    return desktopWhite;
}
void Presentation::Resize(UINT w, UINT h) {
    const bool hdr = fg && Hdr();
    if (width == w && height == h && (!fg || hdr == outputHdr)) return;
    Hide();
    if (!gpu.Drain()) throw std::runtime_error("GPU timeout while resizing presentation.");
    if (fg) {
        fg->Disable();
        outputHdr = hdr;
        fgOutput->Resize(w, h, hdr);
        if (swapchain) Check(swapchain->ResizeBuffers(3, w, h, fgOutput->Format(), swapFlags), "Resize FG output");
        else {
            BOOL tearing = FALSE; ComPtr<IDXGIFactory5> factory;
            if (SUCCEEDED(gpu.factory.As(&factory))) factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing));
            swapFlags = tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
            DXGI_SWAP_CHAIN_DESC1 desc{}; desc.Width = w; desc.Height = h; desc.Format = fgOutput->Format();
            desc.SampleDesc.Count = 1; desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; desc.BufferCount = 3;
            desc.Scaling = DXGI_SCALING_STRETCH; desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE; desc.Flags = swapFlags;
            // HWND creation is intercepted by Streamline; composition swapchains are not.
            fg->CreateSwapchain(window, desc, &swapchain);
        }
        const auto space = hdr ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        UINT support = 0; Check(swapchain->CheckColorSpaceSupport(space, &support), "FG colour space support");
        if (!(support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) throw std::runtime_error("FG output colour space is unavailable.");
        Check(swapchain->SetColorSpace1(space), "FG output colour space");
        width = w; height = h; return;
    }
    if (swapchain) Check(swapchain->ResizeBuffers(2, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT), "Resize output");
    else {
        DXGI_SWAP_CHAIN_DESC1 d{}; d.Width = w; d.Height = h; d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.SampleDesc.Count = 1; d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; d.BufferCount = 2;
        d.Scaling = DXGI_SCALING_STRETCH; d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        d.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
        d.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        ComPtr<IDXGISwapChain1> chain;
        Check(gpu.factory->CreateSwapChainForComposition(gpu.queue.Get(), &d, nullptr, &chain), "Output swapchain");
        Check(chain.As(&swapchain), "Output swapchain3");
        ComPtr<IDXGISwapChain2> pacing; Check(chain.As(&pacing), "Presentation pacing");
        Check(pacing->SetMaximumFrameLatency(1), "Bound presentation latency");
        Check(visual->SetContent(swapchain.Get()), "Composition content"); Check(composition->Commit(), "Commit output");
    }
    UINT support = 0;
    Check(swapchain->CheckColorSpaceSupport(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, &support), "scRGB support");
    if (!(support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) throw std::runtime_error("scRGB presentation is unavailable.");
    Check(swapchain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709), "scRGB output");
    width = w; height = h;
}
void Presentation::BeginFrame(unsigned targetFps) {
    if (fg) {
        RECT bounds{}, previous{}; GetWindowRect(window, &previous);
        if (!source.Bounds(bounds)) throw std::runtime_error("Output bounds are unavailable.");
        if (!EqualRect(&bounds, &previous)) {
            fg->Quiesce(swapchain.Get());
            Check(SetWindowPos(window, nullptr, bounds.left, bounds.top,
                bounds.right - bounds.left, bounds.bottom - bounds.top, SWP_NOZORDER | SWP_NOACTIVATE) ? S_OK :
                HRESULT_FROM_WIN32(GetLastError()), "Position FG output");
        }
        // Resume with one ordinary present, then show the populated window while
        // FG is still off. Enable interpolation on the following, stable frame.
        if (!fg->Enabled()) fg->Configure(width, height, fgOutput->Format(), multiplier, IsWindowVisible(window) != FALSE);
        fg->BeginFrame(float(targetFps));
    }
}
void Presentation::Copy(ID3D12Resource* output, bool reset, ID3D12Resource* sceneMotion) {
    if (fg) {
        const bool cut = fgOutput->Run(output, reset, sceneMotion);
        output = fgOutput->Colour();
        fg->Tag(output, fgOutput->Depth(), fgOutput->Vectors(), cut);
        Barrier(gpu.commands.Get(), output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    ComPtr<ID3D12Resource> buffer;
    Check(swapchain->GetBuffer(swapchain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&buffer)), "Output backbuffer");
    Barrier(gpu.commands.Get(), buffer.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
    gpu.commands->CopyResource(buffer.Get(), output);
    Barrier(gpu.commands.Get(), buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
    if (fg) Barrier(gpu.commands.Get(), output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}
void Presentation::Present() {
    if (fg) fg->BeforePresent();
    Check(swapchain->Present(0, fg && (swapFlags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) ? DXGI_PRESENT_ALLOW_TEARING : 0), "Present output");
    if (fg) fg->AfterPresent();
    if (!Active()) { Hide(); return; }
    RECT bounds{};
    if (!source.Bounds(bounds) || (UINT)(bounds.right - bounds.left) != width ||
        (UINT)(bounds.bottom - bounds.top) != height) { Hide(); return; }
    if (!fg) {
        Order();
        SetWindowPos(window, nullptr, bounds.left, bounds.top, (int)width, (int)height, SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    } else if (!IsWindowVisible(window)) {
        fg->Quiesce(swapchain.Get());
        Order();
        SetWindowPos(window, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW);
    }
}
}
