#pragma once
#include "Gpu.h"
#include "CaptureSource.h"
#include <dcomp.h>
#include "FrameGeneration.h"
#include "FrameGenerationOutput.h"
namespace nr {
class Presentation {
public:
    Presentation(Gpu& gpu, CaptureSource source, HWND controls = nullptr, FrameGeneration* fg = nullptr, unsigned multiplier = 1);
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
private:
    bool ControlsFocused() const { return controls && GetForegroundWindow() == controls; }
    void Order();
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
