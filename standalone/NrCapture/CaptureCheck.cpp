#include "Capture.h"
#include "Processor.h"
#include "Presentation.h"
#include <iostream>
#include <numeric>
#include <vector>
#include <DirectXPackedVector.h>
namespace nr {
namespace {
unsigned phase = 0;
LRESULT CALLBACK Fixture(HWND window, UINT message, WPARAM w, LPARAM l) {
    if (message == WM_PAINT) {
        PAINTSTRUCT paint; HDC dc = BeginPaint(window, &paint); RECT r; GetClientRect(window, &r);
        HBRUSH background = CreateSolidBrush(RGB(24, 56, 90)); FillRect(dc, &r, background); DeleteObject(background);
        for (int i = 0; i < 20; ++i) {
            HBRUSH brush = CreateSolidBrush(RGB((i * 29) % 255, (i * 41) % 255, (i * 53) % 255));
            RECT box{int((phase * 11 + i * 53) % std::max(1L, r.right)), i * 25, 0, i * 25 + 18};
            box.right = box.left + 110; FillRect(dc, &box, brush); DeleteObject(brush);
        }
        SetTextColor(dc, RGB(255, 255, 255)); SetBkMode(dc, TRANSPARENT);
        const auto text = std::format(L"NR capture fixture - moving objects and text - frame {}", phase);
        TextOutW(dc, 20, r.bottom - 60, text.c_str(), (int)text.size()); EndPaint(window, &paint); return 0;
    }
    return DefWindowProcW(window, message, w, l);
}
void Pump() { MSG m; while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&m); DispatchMessageW(&m); } }
struct FixtureWindow {
    HWND value = nullptr;
    ~FixtureWindow() { if (value) DestroyWindow(value); }
};
struct PixelCheck { double difference = 0, maximum = 0, luminance = 0; };
PixelCheck CompareTextures(Gpu& gpu, ID3D12Resource* input, ID3D12Resource* output, UINT margin = 0) {
    auto desc = input->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &bytes);
    auto a = gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK), b = gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    gpu.Begin(); Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    auto copy = [&](ID3D12Resource* source, ID3D12Resource* destination) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{}; src.pResource = source; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource = destination; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = layout;
        gpu.commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    };
    copy(input, a.Get()); copy(output, b.Get());
    Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Capture readback timeout");
    void *pa, *pb; Check(a->Map(0, nullptr, &pa), "Original readback"); Check(b->Map(0, nullptr, &pb), "Output readback");
    PixelCheck result;
    for (UINT y = margin; y < desc.Height - margin; ++y) {
        auto* rowA = (DirectX::PackedVector::HALF*)((BYTE*)pa + layout.Offset + size_t(y) * layout.Footprint.RowPitch);
        auto* rowB = (DirectX::PackedVector::HALF*)((BYTE*)pb + layout.Offset + size_t(y) * layout.Footprint.RowPitch);
        for (UINT64 x = margin; x < desc.Width - margin; ++x) for (UINT c = 0; c < 3; ++c) {
            const float av = DirectX::PackedVector::XMConvertHalfToFloat(rowA[x * 4 + c]);
            const float bv = DirectX::PackedVector::XMConvertHalfToFloat(rowB[x * 4 + c]);
            if (!std::isfinite(av) || !std::isfinite(bv)) throw std::runtime_error("Non-finite captured or processed pixels");
            const double delta = std::abs(av - bv); result.difference += delta; result.maximum = std::max(result.maximum, delta);
            result.luminance += bv;
        }
    }
    a->Unmap(0, nullptr); b->Unmap(0, nullptr);
    const double samples = double(desc.Width - margin * 2) * (desc.Height - margin * 2) * 3;
    result.difference /= samples; result.luminance /= samples;
    return result;
}
}

void MonitorCheck(Settings settings) {
    // Cover the monitor with owned fixtures; no games or private desktop images are saved.
    CaptureSource source(MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY)); RECT bounds{};
    if (!source.Bounds(bounds)) throw std::runtime_error("No monitor available for capture validation.");
    const UINT width = bounds.right - bounds.left, height = bounds.bottom - bounds.top;
    WNDCLASSW wc{}; wc.lpfnWndProc = Fixture; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"NR.MonitorFixture";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); RegisterClassW(&wc);
    FixtureWindow first, second, controls;
    auto makeFixture = [&](const wchar_t* title, DWORD style) {
        HWND result = CreateWindowExW(WS_EX_TOPMOST, wc.lpszClassName, title, style,
            bounds.left, bounds.top, width, height, nullptr, nullptr, wc.hInstance, nullptr);
        if (!result) Check(HRESULT_FROM_WIN32(GetLastError()), "Monitor fixture");
        return result;
    };
    first.value = makeFixture(L"Display Filter monitor test A", WS_POPUP | WS_VISIBLE);
    second.value = makeFixture(L"Display Filter monitor test B", WS_POPUP);
    controls.value = makeFixture(L"Display Filter controls fixture", WS_POPUP);
    Check(SetWindowDisplayAffinity(controls.value, WDA_EXCLUDEFROMCAPTURE) ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "Exclude test controls");
    SetForegroundWindow(first.value); UpdateWindow(first.value); Pump(); Check(DwmFlush(), "Fixture composition");

    Gpu gpu; Processor processor(gpu, settings.runtime);
    Presentation display(gpu, source, controls.value); Capture capture(gpu, source);
    DWORD affinity = 0;
    if (!GetWindowDisplayAffinity(display.Window(), &affinity) || affinity != WDA_EXCLUDEFROMCAPTURE)
        throw std::runtime_error("Monitor output is not excluded from capture.");
    auto next = [&](Capture& target) {
        const double deadline = ClockMs() + 5000;
        while (ClockMs() < deadline) {
            Pump();
            if (target.Next()) {
                if (target.width != width || target.height != height) throw std::runtime_error("Wrong monitor capture size.");
                return;
            }
            WaitForSingleObject(target.Event(), 10);
        }
        throw std::runtime_error("Monitor fixture capture timed out.");
    };
    next(capture); processor.Resize(width, height, settings); display.Resize(width, height);
    auto frameSettings = settings; frameSettings.strength = 0;
    gpu.Begin(); auto* output = processor.Run(capture.Input(), frameSettings, display.Hdr()); gpu.Submit();
    if (!gpu.Drain()) throw std::runtime_error("Monitor bypass timeout.");
    if (CompareTextures(gpu, capture.Input(), output).maximum != 0) throw std::runtime_error("Monitor bypass changed pixels.");
    gpu.Begin(); output = processor.Run(capture.Input(), settings, display.Hdr()); display.Copy(output); gpu.Submit(); display.Present();
    if (!gpu.Drain()) throw std::runtime_error("Monitor NR timeout.");
    auto changed = CompareTextures(gpu, capture.Input(), output);
    if (changed.difference < 0.00001 || !IsWindowVisible(display.Window())) throw std::runtime_error("Monitor NR did not produce visible changed output.");
    std::cout << std::format("Monitor NR {}x{}: RGB delta {:.6f}; zero-strength exact\n", width, height, changed.difference);

    // Show a deliberately different solid output. Screen capture must still match
    // an independent capture of the underlying fixture, even after it changes.
    auto marker = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto description = marker->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &layout, nullptr, nullptr, &bytes);
    auto upload = gpu.Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD); void* mapped = nullptr;
    Check(upload->Map(0, nullptr, &mapped), "Monitor marker upload");
    using namespace DirectX::PackedVector;
    for (UINT y = 0; y < height; ++y) {
        auto* row = (HALF*)((BYTE*)mapped + layout.Offset + size_t(y) * layout.Footprint.RowPitch);
        for (UINT x = 0; x < width; ++x) {
            row[x * 4] = XMConvertFloatToHalf(0.25f); row[x * 4 + 1] = XMConvertFloatToHalf(0.02f);
            row[x * 4 + 2] = XMConvertFloatToHalf(0.35f); row[x * 4 + 3] = XMConvertFloatToHalf(1);
        }
    }
    upload->Unmap(0, nullptr); gpu.Begin();
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{}; src.pResource = upload.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = layout; dst.pResource = marker.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    gpu.commands->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Barrier(gpu.commands.Get(), marker.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    display.Copy(marker.Get()); gpu.Submit(); display.Present();
    if (!gpu.Drain()) throw std::runtime_error("Monitor marker timeout.");
    double maximumDelta = 0;
    for (unsigned iteration = 0; iteration < 4; ++iteration) {
        HWND active = iteration < 2 ? first.value : second.value;
        if (iteration == 2) ShowWindow(second.value, SW_SHOW);
        phase += 17; InvalidateRect(active, nullptr, FALSE); UpdateWindow(active);
        SetForegroundWindow(active); Pump(); display.Present(); Check(DwmFlush(), "Changed fixture composition");
        if (!display.Active() || !IsWindowVisible(display.Window()) || GetForegroundWindow() != active ||
            WindowFromPoint({bounds.left + 15, bounds.top + 15}) != active)
            throw std::runtime_error(std::format("Monitor switching check: active={}, visible={}, foreground={}, expected={}, hit={}",
                display.Active(),IsWindowVisible(display.Window()),(void*)GetForegroundWindow(),(void*)active,
                (void*)WindowFromPoint({bounds.left+15,bounds.top+15})));
        Capture reference(gpu, active); next(reference);
        auto expected = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
        gpu.Begin(); Barrier(gpu.commands.Get(), reference.Input(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        gpu.commands->CopyResource(expected.Get(), reference.Input());
        Barrier(gpu.commands.Get(), reference.Input(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        Barrier(gpu.commands.Get(), expected.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Monitor reference timeout.");
        bool matches = false; double delta = 0;
        const double deadline = ClockMs() + 5000;
        do {
            // The OS capture border is not part of the fixture's client image.
            next(capture); auto result = CompareTextures(gpu, capture.Input(), expected.Get(), 8); delta = result.difference;
            matches = delta < 0.001;
        } while (!matches && ClockMs() < deadline);
        std::cout << std::format("Monitor exclusion frame {}: underlying-window RGB delta {:.6f}\n", iteration, delta);
        if (!matches) throw std::runtime_error("Screen capture includes the overlay or fails to reveal the underlying window.");
        maximumDelta = std::max(maximumDelta, delta);
    }
    // The filter stays on while the controls have focus, directly beneath them.
    ShowWindow(controls.value, SW_SHOW); SetForegroundWindow(controls.value); Pump(); display.RefreshVisibility();
    if (!display.Active() || !IsWindowVisible(display.Window()) || GetWindow(controls.value, GW_HWNDNEXT) != display.Window())
        throw std::runtime_error("Overlay is not directly beneath its focused controls.");
    ShowWindow(controls.value, SW_HIDE); SetForegroundWindow(second.value); Pump(); display.Present();
    if (!display.Active() || !IsWindowVisible(display.Window()) || !(GetWindowLongPtrW(display.Window(), GWL_EXSTYLE) & WS_EX_TOPMOST))
        throw std::runtime_error("Monitor overlay failed to return to topmost.");
    display.Hide(); if (!gpu.Drain()) throw std::runtime_error("Monitor stop timeout."); capture.Stop();
    { Capture restarted(gpu, source); next(restarted); if (!gpu.Drain()) throw std::runtime_error("Monitor restart timeout."); }
    std::cout << std::format("PASS monitor capture: NR, identity, exclusion across four changed frames (max mean delta {:.6f}), window switching, click-through, output beneath focused controls and restart\n", maximumDelta);
}
void CaptureCheck(Settings settings, UINT width, UINT height, unsigned requested) {
    Gpu gpu; Processor processor(gpu, settings.runtime);
    WNDCLASSW wc{}; wc.lpfnWndProc = Fixture; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"NR.IndependentFixture";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); RegisterClassW(&wc);
    FixtureWindow fixture;
    fixture.value = CreateWindowW(wc.lpszClassName, L"Independent NR validation - no game", WS_POPUP | WS_VISIBLE,
        30, 30, width, height, nullptr, nullptr, wc.hInstance, nullptr);
    if (!fixture.value) Check(HRESULT_FROM_WIN32(GetLastError()), "Fixture window");
    SetForegroundWindow(fixture.value); UpdateWindow(fixture.value); Pump();
    const auto focusDeadline = ClockMs() + 30000;
    while (GetForegroundWindow() != fixture.value && ClockMs() < focusDeadline) { Pump(); Sleep(10); }
    if (GetForegroundWindow() != fixture.value) throw std::runtime_error("Focus the independent validation window to run its presentation checks.");
    FixtureWindow panel; // Stands in for the application's controls; shown later.
    panel.value = CreateWindowW(wc.lpszClassName, L"Controls fixture", WS_OVERLAPPEDWINDOW, 60, 60, 320, 200, nullptr, nullptr, wc.hInstance, nullptr);
    Presentation display(gpu, fixture.value, panel.value);
    Capture capture(gpu, fixture.value);
    double start = ClockMs(), lastCapture = start;
    std::vector<double> latency, completion;
    double steadyStart = 0;
    bool identity = false, changed = false, focus = false, clickThrough = false;
    bool presentationPixels = false;
    unsigned rendered = 0;
    while (rendered < requested && ClockMs() - start < 45000) {
        Pump(); if (!gpu.Complete()) { Sleep(1); continue; }
        if (!capture.Next()) { WaitForSingleObject(capture.Event(), 10); continue; }
        lastCapture = ClockMs();
        processor.Resize(capture.width, capture.height, settings); display.Resize(capture.width, capture.height);
        Settings frameSettings = settings; frameSettings.strength = rendered == 0 ? 0.0f : 1.0f;
        gpu.Begin(); auto* output = processor.Run(capture.Input(), frameSettings, display.Hdr());
        display.Copy(output); gpu.Submit(); display.Present();
        const double delay = ClockMs() - capture.timestamp;
        if (!gpu.Drain()) throw std::runtime_error("Capture processing timeout");
        if (rendered == 5) steadyStart = ClockMs();
        if (rendered > 5) completion.push_back(ClockMs() - capture.timestamp);
        if (rendered == 0 || rendered == 1 || (settings.estimateMotion && rendered % 15 == 0)) {
            auto checked = CompareTextures(gpu, capture.Input(), output);
            if (rendered == 0) identity = checked.maximum == 0;
            else changed = checked.difference > 0.00001 && checked.luminance > 0.001;
            if (rendered > 0 && !changed) throw std::runtime_error("Continuous motion capture lost the NR effect.");
            std::cout << std::format("Capture frame {}: RGB delta {:.6f}, max {:.6f}, output mean {:.6f}\n",
                rendered, checked.difference, checked.maximum, checked.luminance);
        }
        if (rendered > 2) latency.push_back(delay);
        if (rendered == 3) {
            focus = GetForegroundWindow() == fixture.value;
            RECT r{}; GetWindowRect(fixture.value, &r);
            clickThrough = WindowFromPoint({r.left + 15, r.top + 15}) == fixture.value;
            if (!IsWindowVisible(display.Window())) throw std::runtime_error("Output window was not shown.");
            // Independently capture our own composed output: catches a visible-but-black
            // layered window, alpha mistakes, and a swapchain that never reaches DWM.
            Capture presented(gpu, display.Window());
            const double deadline = ClockMs() + 3000;
            while (ClockMs() < deadline) {
                Pump();
                if (presented.Next()) {
                    auto pixels = CompareTextures(gpu, presented.Input(), output);
                    std::cout << std::format("Presented output: RGB delta {:.6f}; mean {:.6f}\n", pixels.difference, pixels.luminance);
                    presentationPixels = pixels.difference < 0.015 && pixels.luminance > 0.001;
                    break;
                }
                WaitForSingleObject(presented.Event(), 10);
            }
            if (!gpu.Drain()) throw std::runtime_error("Presented capture drain timeout");
        }
        ++rendered; ++phase; InvalidateRect(fixture.value, nullptr, FALSE); UpdateWindow(fixture.value);
    }
    display.Hide(); if (!gpu.Drain()) throw std::runtime_error("Final capture drain timeout");
    if (rendered != requested || !identity || !changed || !focus || !clickThrough || !presentationPixels)
        throw std::runtime_error(std::format("Capture check failed: frames={}/{}, identity={}, NR={}, focus={}, clickThrough={}, presentation={}, last frame age={:.0f} ms",
            rendered, requested, identity, changed, focus, clickThrough, presentationPixels, ClockMs() - lastCapture));
    std::sort(latency.begin(), latency.end());
    const auto finished = ClockMs();
    std::sort(completion.begin(), completion.end());
    DXGI_QUERY_VIDEO_MEMORY_INFO memory{}; gpu.adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory);
    auto result = std::format("PASS WGC {}x{}: {} frames, {:.1f} steady fps; {:.1f} ms mean / {:.1f} ms p95 acquire-to-Present return; {:.1f} ms p95 acquire-to-GPU-complete; {:.0f} MiB app VRAM; identity, NR pixels, focus, click-through and composed output",
        width, height, rendered, (rendered - 6) * 1000.0 / (finished - steadyStart),
        std::accumulate(latency.begin(), latency.end(), 0.0) / latency.size(), latency[size_t(latency.size() * 0.95)],
        completion[size_t(completion.size() * 0.95)], memory.CurrentUsage / 1048576.0);
    std::cout << result << std::endl; Log(result);
    // Resize to a decorated window, then stop/restart capture without restarting NGX.
    SetWindowLongPtrW(fixture.value, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
    RECT wanted{0, 0, 640, 360}; AdjustWindowRectExForDpi(&wanted, WS_OVERLAPPEDWINDOW, FALSE, 0, GetDpiForWindow(fixture.value));
    SetWindowPos(fixture.value, nullptr, 40, 40, wanted.right - wanted.left, wanted.bottom - wanted.top, SWP_FRAMECHANGED | SWP_NOZORDER);
    ++phase; InvalidateRect(fixture.value, nullptr, FALSE); UpdateWindow(fixture.value);
    auto presentFresh = [&](Capture& current) {
        const double deadline = ClockMs() + 5000;
        while (ClockMs() < deadline) {
            InvalidateRect(fixture.value, nullptr, FALSE); UpdateWindow(fixture.value);
            Pump();
            if (current.Next()) {
                if (current.width != 640 || current.height != 360) {
                    gpu.Begin(); gpu.Submit(); gpu.Drain(); continue;
                }
                processor.Resize(current.width, current.height, settings); display.Resize(current.width, current.height);
                gpu.Begin(); auto* edited = processor.Run(current.Input(), settings, display.Hdr());
                display.Copy(edited); gpu.Submit(); display.Present();
                if (!gpu.Drain()) throw std::runtime_error("Lifecycle fence timeout");
                return;
            }
            WaitForSingleObject(current.Event(), 10);
        }
        throw std::runtime_error("Windowed client crop/resize or capture restart failed.");
    };
    Log("Capture check: resizing to decorated 640x360 client.");
    presentFresh(capture); capture.Stop(); Log("Capture check: restarting.");
    {
        Capture restarted(gpu, fixture.value); presentFresh(restarted);
    }
    ShowWindow(fixture.value, SW_MINIMIZE); Pump();
    if (display.Active()) throw std::runtime_error("Minimized window remained active.");
    display.Hide(); if (IsWindowVisible(display.Window())) throw std::runtime_error("Output did not hide.");
    ShowWindow(fixture.value, SW_RESTORE); SetForegroundWindow(fixture.value); Pump();
    {
        Capture restarted(gpu, fixture.value); presentFresh(restarted);
    }
    // The filter stays on while the controls have focus: directly above the source, so beneath the controls.
    ShowWindow(panel.value, SW_SHOW); SetForegroundWindow(panel.value); Pump();
    {
        Capture restarted(gpu, fixture.value); presentFresh(restarted); display.RefreshVisibility();
        if (GetForegroundWindow() != panel.value) throw std::runtime_error("The controls fixture did not take focus.");
        if (!display.Active() || !IsWindowVisible(display.Window()) || GetWindow(fixture.value, GW_HWNDPREV) != display.Window() ||
            (GetWindowLongPtrW(display.Window(), GWL_EXSTYLE) & WS_EX_TOPMOST))
            throw std::runtime_error("Output is not directly above the source while the controls have focus.");
        SetForegroundWindow(fixture.value); Pump(); presentFresh(restarted);
        if (!display.Active() || !IsWindowVisible(display.Window()) || !(GetWindowLongPtrW(display.Window(), GWL_EXSTYLE) & WS_EX_TOPMOST))
            throw std::runtime_error("Output did not return to topmost with the source.");
    }
    ShowWindow(panel.value, SW_HIDE);
    FixtureWindow other;
    other.value = CreateWindowW(wc.lpszClassName, L"Focus loss fixture", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        750, 100, 400, 250, nullptr, nullptr, wc.hInstance, nullptr);
    SetForegroundWindow(other.value); Pump();
    if (display.Active()) throw std::runtime_error("Focus loss was not detected.");
    display.Hide(); if (IsWindowVisible(display.Window())) throw std::runtime_error("Focus loss left output visible.");
    std::cout << "PASS decorated-window crop, resize, capture restart, minimize/restore, output beneath focused controls and focus loss\n";
}

void WindowCheck(Settings settings, const std::wstring& title, unsigned requested) {
    HWND source=FindWindowW(nullptr,title.c_str());
    if (!source) throw std::runtime_error("The requested capture window is not open.");
    Gpu gpu; Processor processor(gpu,settings.runtime); Presentation display(gpu,source); Capture capture(gpu,source);
    unsigned frames=0; bool identity=false,changed=false,presented=false;
    double start=ClockMs();
    while (frames<requested && ClockMs()-start<30000) {
        Pump();
        if (!capture.Next()) { WaitForSingleObject(capture.Event(),10); continue; }
        processor.Resize(capture.width,capture.height,settings); display.Resize(capture.width,capture.height);
        auto current=settings; if (!frames) current.strength=0;
        gpu.Begin(); auto* output=processor.Run(capture.Input(),current,display.Hdr());
        display.Copy(output); gpu.Submit(); display.Present();
        if (!gpu.Drain()) throw std::runtime_error("Application capture completion timeout.");
        if (frames<2) {
            auto pixels=CompareTextures(gpu,capture.Input(),output);
            if (!frames) identity=pixels.maximum==0;
            else changed=pixels.difference>0.00001;
            std::cout<<std::format("Application frame {}: NR delta {:.6f}, output mean {:.6f}\n",frames,pixels.difference,pixels.luminance);
        }
        if (frames==3 && IsWindowVisible(display.Window())) {
            Capture observer(gpu,display.Window()); const auto deadline=ClockMs()+3000;
            while (ClockMs()<deadline) {
                Pump();
                if (observer.Next()) {
                    auto pixels=CompareTextures(gpu,observer.Input(),output);
                    presented=pixels.difference<0.015 && pixels.luminance>0.001;
                    std::cout<<std::format("Application displayed output: RGB delta {:.6f}\n",pixels.difference);
                    break;
                }
                WaitForSingleObject(observer.Event(),10);
            }
            if (!gpu.Drain()) throw std::runtime_error("Application output capture timeout.");
        }
        ++frames;
    }
    display.Hide(); if (!gpu.Drain()) throw std::runtime_error("Application capture stop timeout.");
    if (frames!=requested || !identity || !changed || !presented)
        throw std::runtime_error(std::format("Application capture check: frames={}/{}, identity={}, NR={}, displayed={}; keep the selected window focused",
            frames,requested,identity,changed,presented));
    std::cout<<std::format("PASS {} frames from {}: exact bypass, changed NR pixels, composed output, {} motion\n",
        frames,Narrow(title),settings.estimateMotion ? "estimated" : "zero");
}
}
