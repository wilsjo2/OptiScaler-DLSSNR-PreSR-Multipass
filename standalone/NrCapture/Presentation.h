#pragma once
#include "Gpu.h"
#include "CaptureSource.h"
#include <dcomp.h>
#include "FrameGeneration.h"
#include "FrameGenerationOutput.h"
namespace nr {
class Presentation {
public:
    // presentMode: Settings::presentMode (DLSS FG off only): 0 overlay surface (DirectComposition), 1 flip window.
    Presentation(Gpu& gpu, CaptureSource source, HWND controls = nullptr, FrameGeneration* fg = nullptr, unsigned multiplier = 1,
                 unsigned presentMode = 0);
    ~Presentation();
    void Resize(UINT width, UINT height);
    void BeginFrame(unsigned targetFps = 0);
    void Copy(ID3D12Resource* output, bool reset = false, ID3D12Resource* sceneMotion = nullptr);
    void Present();
    void RefreshVisibility();
    bool Active() const;
    bool Hdr() const;
    float WhitePoint(bool hdr);
    void Hide();
    HWND Window() const { return window; }
    bool FlipWindow() const { return flipWindow; } // the flip-model swapchain of Settings::presentMode 1 (DLSS FG off)
    // Diagnostics: the time Present was called and returned (ClockMs) and the swapchain's present count, for the last frame.
    double presentCalled = 0, presentReturned = 0;
    UINT lastPresentCount = 0;
private:
    bool ControlsFocused() const { return controls && GetForegroundWindow() == controls; }
    void Order();
    bool flipWindow = false;
    HANDLE latencyWait = nullptr; // the flip window's frame-latency object (owned by the swapchain)
    Gpu& gpu;
    CaptureSource source;
    HWND controls, window = nullptr;
    UINT width = 0, height = 0;
    HMONITOR whiteMonitor = nullptr;
    ULONGLONG whiteQueriedAt = 0;
    float desktopWhite = 203.0f / 80.0f;
    ComPtr<IDXGISwapChain3> swapchain;
    ComPtr<IDCompositionDevice> composition;
    ComPtr<IDCompositionTarget> target;
    ComPtr<IDCompositionVisual> visual;
    FrameGeneration* fg = nullptr;
    std::unique_ptr<FrameGenerationOutput> fgOutput;
    unsigned multiplier = 1;
    bool outputHdr = false;
    UINT swapFlags = 0;
};
}
