// pace_source.exe: a stand-in game window for pacing and latency tests of the Display Filter. A borderless window
// (title "Pace source") presents an HDR picture at an exact rate: a window of a raw frame (RGBA16F, linear scRGB) that
// slides sideways, and a white bar that moves across it, so every frame differs and motion is easy to follow.
//   pace_source.exe [--fps 60] [--x 0] [--y 0] [--width 1920] [--height 1080] [--seconds 60] [--pan 4]
//                   [--image frame.rgba16f --image-width 3840 --image-height 2160] [--log presents.csv]
// --log writes one line per frame: its number and the QPC time (milliseconds) right before and after Present.
// Without --image the picture is a grey ramp. Esc or closing the window ends it.
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;
namespace {
double Clock() {
    static const double frequency = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return double(f.QuadPart); }();
    LARGE_INTEGER now; QueryPerformanceCounter(&now); return now.QuadPart * 1000.0 / frequency;
}
void Check(HRESULT result, const char* what) {
    if (FAILED(result)) { std::fprintf(stderr, "%s failed: 0x%08lX\n", what, unsigned long(result)); ExitProcess(1); }
}
bool running = true;
LRESULT CALLBACK Procedure(HWND window, UINT message, WPARAM w, LPARAM l) {
    if (message == WM_DESTROY || (message == WM_KEYDOWN && w == VK_ESCAPE)) { running = false; PostQuitMessage(0); return 0; }
    return DefWindowProcW(window, message, w, l);
}
}
int wmain(int argc, wchar_t** argv) {
    double fps = 60, seconds = 60; int x = 0, y = 0, width = 1920, height = 1080, pan = 4, imageWidth = 3840, imageHeight = 2160;
    std::wstring image, log;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::wstring key = argv[i], value = argv[i + 1];
        if (key == L"--fps") fps = _wtof(value.c_str());
        else if (key == L"--seconds") seconds = _wtof(value.c_str());
        else if (key == L"--x") x = _wtoi(value.c_str());
        else if (key == L"--y") y = _wtoi(value.c_str());
        else if (key == L"--width") width = _wtoi(value.c_str());
        else if (key == L"--height") height = _wtoi(value.c_str());
        else if (key == L"--pan") pan = _wtoi(value.c_str());
        else if (key == L"--image") image = value;
        else if (key == L"--image-width") imageWidth = _wtoi(value.c_str());
        else if (key == L"--image-height") imageHeight = _wtoi(value.c_str());
        else if (key == L"--log") log = value;
        else { std::fprintf(stderr, "Unknown argument.\n"); return 1; }
    }
    if (fps <= 0 || width <= 0 || height <= 0) return 1;
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    WNDCLASSW windowClass{}; windowClass.lpfnWndProc = Procedure; windowClass.hInstance = GetModuleHandleW(nullptr);
    windowClass.lpszClassName = L"PaceSource"; windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassW(&windowClass);
    HWND window = CreateWindowExW(0, windowClass.lpszClassName, L"Pace source", WS_POPUP | WS_VISIBLE, x, y, width, height,
        nullptr, nullptr, windowClass.hInstance, nullptr);
    if (!window) return 1;
    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context;
    Check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &context), "D3D11 device");
    ComPtr<ID3D11DeviceContext1> context1; Check(context.As(&context1), "D3D11.1 context");
    ComPtr<IDXGIDevice> dxgiDevice; Check(device.As(&dxgiDevice), "DXGI device");
    ComPtr<IDXGIAdapter> adapter; Check(dxgiDevice->GetAdapter(&adapter), "Adapter");
    ComPtr<IDXGIFactory2> factory; Check(adapter->GetParent(IID_PPV_ARGS(&factory)), "Factory");
    DXGI_SWAP_CHAIN_DESC1 description{}; description.Width = UINT(width); description.Height = UINT(height);
    description.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; description.SampleDesc.Count = 1;
    description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; description.BufferCount = 2;
    description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGISwapChain1> chain; Check(factory->CreateSwapChainForHwnd(device.Get(), window, &description, nullptr, nullptr, &chain), "Swap chain");
    factory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
    ComPtr<IDXGISwapChain3> chain3;
    if (SUCCEEDED(chain.As(&chain3))) chain3->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709);
    // The picture: a raw frame, or a ramp. It is wider than the window so it can slide.
    int pictureWidth = width + 1024, pictureHeight = height;
    std::vector<uint16_t> pixels;
    if (!image.empty()) {
        pixels.resize(size_t(imageWidth) * imageHeight * 4);
        std::ifstream file(image, std::ios::binary | std::ios::ate);
        if (!file || size_t(file.tellg()) != pixels.size() * 2) { std::fprintf(stderr, "The image is not RGBA16F at its size.\n"); return 1; }
        file.seekg(0); file.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * 2);
        pictureWidth = imageWidth; pictureHeight = imageHeight;
    } else {
        pixels.resize(size_t(pictureWidth) * pictureHeight * 4);
        for (int py = 0; py < pictureHeight; ++py) for (int px = 0; px < pictureWidth; ++px) {
            const float v = 0.05f + 1.5f * float((px / 64 + py / 64) % 8) / 7.f; // scRGB, up to 1.55 (124 nits)
            const uint16_t half = [](float f) { uint32_t b; memcpy(&b, &f, 4); return uint16_t(((b >> 16) & 0x8000) | ((((b & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((b >> 13) & 0x03ff)); }(v);
            for (int c = 0; c < 3; ++c) pixels[(size_t(py) * pictureWidth + px) * 4 + c] = half;
            pixels[(size_t(py) * pictureWidth + px) * 4 + 3] = 0x3c00;
        }
    }
    D3D11_TEXTURE2D_DESC textureDescription{}; textureDescription.Width = UINT(pictureWidth); textureDescription.Height = UINT(pictureHeight);
    textureDescription.MipLevels = 1; textureDescription.ArraySize = 1; textureDescription.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    textureDescription.SampleDesc.Count = 1; textureDescription.Usage = D3D11_USAGE_IMMUTABLE; textureDescription.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA data{pixels.data(), UINT(pictureWidth) * 8, 0};
    ComPtr<ID3D11Texture2D> picture; Check(device->CreateTexture2D(&textureDescription, &data, &picture), "Picture");
    pixels.clear(); pixels.shrink_to_fit();
    const int slideRoom = std::max(0, pictureWidth - width), riseRoom = std::max(0, pictureHeight - height);
    std::FILE* presents = log.empty() ? nullptr : _wfopen(log.c_str(), L"w");
    if (presents) std::fputs("frame,beforePresent,afterPresent\n", presents);
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const double period = 1000.0 / fps, start = Clock();
    double next = start;
    for (unsigned long long frame = 0; running && Clock() - start < seconds * 1000; ++frame) {
        MSG message; while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&message); DispatchMessageW(&message); }
        if (!running) break;
        next += period;
        if (Clock() - next > period) next = Clock(); // (a stall restarts the grid instead of catching up)
        for (double wait; (wait = next - Clock()) > 0;) {
            if (wait > 1.5) {
                LARGE_INTEGER due; due.QuadPart = -LONGLONG((wait - 1.0) * 10000);
                SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE); WaitForSingleObject(timer, 100);
            } else YieldProcessor();
        }
        ComPtr<ID3D11Texture2D> buffer; Check(chain->GetBuffer(0, IID_PPV_ARGS(&buffer)), "Back buffer");
        // Slide back and forth over the picture's room; ping-pong so the motion has no jump.
        int offset = 0;
        if (slideRoom > 0) { const int cycle = 2 * slideRoom; const int at = int((frame * pan) % cycle); offset = at < slideRoom ? at : cycle - at; }
        D3D11_BOX box{UINT(offset), UINT(riseRoom / 2), 0, UINT(offset + std::min(width, pictureWidth)), UINT(riseRoom / 2 + std::min(height, pictureHeight)), 1};
        context->CopySubresourceRegion(buffer.Get(), 0, 0, 0, 0, picture.Get(), 0, &box);
        ComPtr<ID3D11RenderTargetView> target; Check(device->CreateRenderTargetView(buffer.Get(), nullptr, &target), "Target");
        // A black strip at the top with only the bar in it: frame_watch.exe finds the bar there on the screen, and so
        // which frame is shown (the bar moves 12 pixels a frame; a frame made in between shows it half way).
        const float white[4]{2.5f, 2.5f, 2.5f, 1}, black[4]{0, 0, 0, 1}; // 200 nits
        const D3D11_RECT strip{0, 0, width, std::min(48, height)};
        context1->ClearView(target.Get(), black, &strip, 1);
        const LONG barX = LONG((frame * 12) % UINT(width));
        const D3D11_RECT bar{barX, 0, std::min<LONG>(barX + 24, width), height};
        context1->ClearView(target.Get(), white, &bar, 1);
        const double before = Clock();
        Check(chain->Present(0, 0), "Present");
        if (presents) std::fprintf(presents, "%llu,%.3f,%.3f\n", frame, before, Clock());
    }
    if (presents) std::fclose(presents);
    CloseHandle(timer);
    DestroyWindow(window);
    return 0;
}
