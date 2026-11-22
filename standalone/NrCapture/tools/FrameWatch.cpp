// frame_watch.exe: what reaches the screen, read back from the screen itself (no administrator rights, no hooks).
// For every screen update it finds pace_source.exe's bar in one row of pixels and writes the update's time (QPC
// milliseconds, when Windows composed that desktop image) and the bar's position. With pace_source's own log (when each
// frame was presented, the bar at frame x 12) this gives:
//   - which source frame each screen update shows, so the delay from the game's present to the screen;
//   - the spacing of the updates (pacing);
//   - frames made in between (DLSS FG, Smooth Motion): the bar half way between two source positions.
//   frame_watch.exe --out file.csv [--seconds 20] [--x 960] [--y 560] [--width 1920] [--wgc 0|1]
// x, width: the pixels of the row to read (the source window's left edge and width); y: the row (in the black strip).
// It reads the primary display with DXGI Desktop Duplication. When DXGI lists no display output (a TV in standby leaves
// Windows a placeholder display) or with --wgc 1, it reads the primary screen with Windows Graphics Capture instead:
// then the time is the composition time Windows gives the captured frame, and "accumulated" is always 1.
// --window "title" reads one window with Windows Graphics Capture instead (x, y relative to it): e.g. the Display
// Filter's output window "Display Filter output", or "Pace source" itself, while no display is connected.
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Metadata.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Security.Authorization.AppCapabilityAccess.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;
namespace {
double ToMs(LONGLONG qpc) {
    static const double frequency = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return double(f.QuadPart); }();
    return qpc * 1000.0 / frequency;
}
double Clock() { LARGE_INTEGER now; QueryPerformanceCounter(&now); return ToMs(now.QuadPart); }
float Half(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    uint32_t bits;
    if (exponent == 0) { if (!mantissa) bits = sign; else { float f = std::ldexp(float(mantissa), -24); return sign ? -f : f; } }
    else if (exponent == 31) bits = sign | 0x7f800000u | (mantissa << 13);
    else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}
void Check(HRESULT result, const char* what) {
    if (FAILED(result)) { std::fprintf(stderr, "%s failed: 0x%08lX\n", what, unsigned long(result)); ExitProcess(1); }
}
// Copies one row of the screen image and finds the bar: the centre of the widest run above half the brightest pixel.
struct RowReader {
    ID3D11DeviceContext* context; ID3D11Texture2D* row; int x0, y, width; bool half;
    std::vector<float> light;
    void Read(ID3D11Texture2D* screen, double& bar, float& peak) {
        const D3D11_BOX box{UINT(x0), UINT(y), 0, UINT(x0 + width), UINT(y + 1), 1};
        context->CopySubresourceRegion(row, 0, 0, 0, 0, screen, 0, &box);
        D3D11_MAPPED_SUBRESOURCE mapped{}; Check(context->Map(row, 0, D3D11_MAP_READ, 0, &mapped), "Map row");
        light.resize(size_t(width)); peak = 0;
        for (int i = 0; i < width; ++i) {
            float r, g, b;
            if (half) { const auto* p = static_cast<const uint16_t*>(mapped.pData) + size_t(i) * 4; r = Half(p[0]); g = Half(p[1]); b = Half(p[2]); }
            else { const auto* p = static_cast<const uint8_t*>(mapped.pData) + size_t(i) * 4; b = p[0] / 255.f; g = p[1] / 255.f; r = p[2] / 255.f; }
            light[size_t(i)] = 0.2126f * r + 0.7152f * g + 0.0722f * b; peak = std::max(peak, light[size_t(i)]);
        }
        context->Unmap(row, 0);
        bar = -1; int bestRun = 0;
        for (int i = 0; i < width;) {
            if (light[size_t(i)] < peak * .5f) { ++i; continue; }
            int j = i; double weighted = 0, sum = 0;
            while (j < width && light[size_t(j)] >= peak * .5f) { weighted += j * double(light[size_t(j)]); sum += light[size_t(j)]; ++j; }
            if (j - i > bestRun) { bestRun = j - i; bar = weighted / sum; }
            i = j;
        }
        if (bestRun < 6) bar = -1;
    }
};
ComPtr<ID3D11Texture2D> RowTexture(ID3D11Device* device, int width, DXGI_FORMAT format) {
    D3D11_TEXTURE2D_DESC staging{}; staging.Width = UINT(width); staging.Height = 1; staging.MipLevels = 1; staging.ArraySize = 1;
    staging.Format = format; staging.SampleDesc.Count = 1; staging.Usage = D3D11_USAGE_STAGING; staging.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> row; Check(device->CreateTexture2D(&staging, nullptr, &row), "Staging row");
    return row;
}
// Windows Graphics Capture of the primary screen, or of one window (`window`; the row is then relative to it).
int WatchWithWgc(std::FILE* file, double seconds, int x0, int y, int width, HWND window) {
    using namespace winrt::Windows::Graphics::Capture;
    using winrt::Windows::Graphics::DirectX::DirectXPixelFormat;
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context;
    Check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
        D3D11_SDK_VERSION, &device, nullptr, &context), "D3D11 device");
    auto interop = winrt::get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    GraphicsCaptureItem item{nullptr};
    if (window) Check(interop->CreateForWindow(window, winrt::guid_of<GraphicsCaptureItem>(), winrt::put_abi(item)), "Capture window");
    else Check(interop->CreateForMonitor(MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY), winrt::guid_of<GraphicsCaptureItem>(),
        winrt::put_abi(item)), "Capture screen");
    ComPtr<IDXGIDevice> dxgi; Check(device.As(&dxgi), "DXGI device");
    winrt::com_ptr<IInspectable> inspect; Check(CreateDirect3D11DeviceFromDXGIDevice(dxgi.Get(), inspect.put()), "WinRT device");
    const auto captureDevice = inspect.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
    const auto size = item.Size();
    std::fprintf(stderr, "Windows Graphics Capture of %s, %dx%d\n", window ? "the window" : "the primary screen", size.Width, size.Height);
    if (x0 < 0 || y < 0 || x0 + width > size.Width || y >= size.Height) { std::fprintf(stderr, "The row lies outside the screen.\n"); return 1; }
    HANDLE arrived = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    auto pool = Direct3D11CaptureFramePool::CreateFreeThreaded(captureDevice, DirectXPixelFormat::R16G16B16A16Float, 3, size);
    const auto token = pool.FrameArrived([arrived](auto&&, auto&&) { SetEvent(arrived); });
    auto session = pool.CreateCaptureSession(item); session.IsCursorCaptureEnabled(false);
    using winrt::Windows::Foundation::Metadata::ApiInformation;
    if (ApiInformation::IsPropertyPresent(L"Windows.Graphics.Capture.GraphicsCaptureSession", L"MinUpdateInterval"))
        try { session.MinUpdateInterval(std::chrono::milliseconds(1)); } catch (const winrt::hresult_error&) {}
    if (!window && ApiInformation::IsPropertyPresent(L"Windows.Graphics.Capture.GraphicsCaptureSession", L"IsBorderRequired"))
        try { GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless).get(); session.IsBorderRequired(false); }
        catch (const winrt::hresult_error&) {}
    const auto row = RowTexture(device.Get(), width, DXGI_FORMAT_R16G16B16A16_FLOAT);
    RowReader reader{context.Get(), row.Get(), x0, y, width, true, {}};
    session.StartCapture();
    const double start = Clock(); unsigned long long updates = 0;
    while (Clock() - start < seconds * 1000) {
        WaitForSingleObject(arrived, 100);
        while (auto frame = pool.TryGetNextFrame()) {
            const auto access = frame.Surface().as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
            ComPtr<ID3D11Texture2D> screen; Check(access->GetInterface(IID_PPV_ARGS(&screen)), "Captured screen");
            double bar; float peak; reader.Read(screen.Get(), bar, peak);
            std::fprintf(file, "%.3f,1,%.2f,%.3f,%.3f\n", frame.SystemRelativeTime().count() / 10000.0, bar, peak, Clock());
            frame.Close(); ++updates;
        }
    }
    session.Close(); pool.FrameArrived(token); pool.Close(); CloseHandle(arrived);
    std::printf("%llu screen updates read (Windows Graphics Capture)\n", updates);
    return 0;
}
}
int wmain(int argc, wchar_t** argv) {
    double seconds = 20; int x0 = 960, y = 560, width = 1920; bool wgc = false; std::wstring out, title;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::wstring key = argv[i], value = argv[i + 1];
        if (key == L"--seconds") seconds = _wtof(value.c_str());
        else if (key == L"--window") title = value;
        else if (key == L"--x") x0 = _wtoi(value.c_str());
        else if (key == L"--y") y = _wtoi(value.c_str());
        else if (key == L"--width") width = _wtoi(value.c_str());
        else if (key == L"--wgc") wgc = _wtoi(value.c_str()) != 0;
        else if (key == L"--out") out = value;
        else { std::fprintf(stderr, "Unknown argument.\n"); return 1; }
    }
    if (out.empty() || width <= 0) { std::fprintf(stderr, "Use: frame_watch --out file.csv [--seconds s --x px --y px --width px --wgc 0|1 --window title]\n"); return 1; }
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    if (!title.empty()) {
        const HWND window = FindWindowW(nullptr, title.c_str());
        if (!window) { std::fprintf(stderr, "No window has that title.\n"); return 1; }
        std::FILE* file = nullptr;
        if (_wfopen_s(&file, out.c_str(), L"w") != 0 || !file) { std::fprintf(stderr, "Cannot write the output file.\n"); return 1; }
        std::fputs("shown,accumulated,bar,peak,read\n", file);
        const int result = WatchWithWgc(file, seconds, x0, y, width, window);
        std::fclose(file); return result;
    }
    ComPtr<IDXGIFactory1> factory; Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
    // The output that holds the primary display (its desktop starts at 0,0), on whichever adapter drives it.
    ComPtr<IDXGIAdapter1> adapter; ComPtr<IDXGIOutput> output;
    for (UINT a = 0; !output && !wgc; ++a) {
        ComPtr<IDXGIAdapter1> candidate; if (factory->EnumAdapters1(a, &candidate) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 ad{}; candidate->GetDesc1(&ad);
        std::fwprintf(stderr, L"adapter %u: %ls\n", a, ad.Description);
        for (UINT o = 0;; ++o) {
            ComPtr<IDXGIOutput> each; const HRESULT found = candidate->EnumOutputs(o, &each);
            if (FAILED(found)) { if (found != DXGI_ERROR_NOT_FOUND) std::fprintf(stderr, "  outputs: 0x%08lX\n", unsigned long(found)); break; }
            DXGI_OUTPUT_DESC d{}; each->GetDesc(&d);
            std::fwprintf(stderr, L"  output %u: %ls attached %d at %ld,%ld\n", o, d.DeviceName, d.AttachedToDesktop, d.DesktopCoordinates.left, d.DesktopCoordinates.top);
            if (d.AttachedToDesktop && d.DesktopCoordinates.left == 0 && d.DesktopCoordinates.top == 0) { adapter = candidate; output = each; break; }
        }
    }
    std::FILE* file = nullptr;
    if (_wfopen_s(&file, out.c_str(), L"w") != 0 || !file) { std::fprintf(stderr, "Cannot write the output file.\n"); return 1; }
    std::fputs("shown,accumulated,bar,peak,read\n", file);
    if (!output) {
        if (!wgc) std::fprintf(stderr, "DXGI lists no output for the primary display: using Windows Graphics Capture.\n");
        const int result = WatchWithWgc(file, seconds, x0, y, width, nullptr);
        std::fclose(file); return result;
    }
    ComPtr<IDXGIOutput5> output5; Check(output.As(&output5), "Output5");
    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context;
    Check(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &context), "D3D11 device");
    const DXGI_FORMAT formats[]{DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_B8G8R8A8_UNORM};
    ComPtr<IDXGIOutputDuplication> duplication; Check(output5->DuplicateOutput1(device.Get(), 0, 2, formats, &duplication), "Duplicate output");
    DXGI_OUTDUPL_DESC described{}; duplication->GetDesc(&described);
    const auto row = RowTexture(device.Get(), width, described.ModeDesc.Format);
    RowReader reader{context.Get(), row.Get(), x0, y, width, described.ModeDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT, {}};
    const double start = Clock();
    unsigned long long updates = 0, missed = 0;
    while (Clock() - start < seconds * 1000) {
        DXGI_OUTDUPL_FRAME_INFO info{}; ComPtr<IDXGIResource> resource;
        const HRESULT acquired = duplication->AcquireNextFrame(100, &info, &resource);
        if (acquired == DXGI_ERROR_WAIT_TIMEOUT) continue;
        if (acquired == DXGI_ERROR_ACCESS_LOST) { std::fprintf(stderr, "Desktop duplication access lost.\n"); break; }
        Check(acquired, "Acquire frame");
        if (info.LastPresentTime.QuadPart) {
            ComPtr<ID3D11Texture2D> desktop; Check(resource.As(&desktop), "Desktop texture");
            double bar; float peak; reader.Read(desktop.Get(), bar, peak);
            std::fprintf(file, "%.3f,%u,%.2f,%.3f,%.3f\n", ToMs(info.LastPresentTime.QuadPart), info.AccumulatedFrames, bar, peak, Clock());
            ++updates; missed += info.AccumulatedFrames > 1 ? info.AccumulatedFrames - 1 : 0;
        }
        duplication->ReleaseFrame();
    }
    std::fclose(file);
    std::printf("%llu screen updates read, %llu more were merged into them (missed)\n", updates, missed);
    return 0;
}
