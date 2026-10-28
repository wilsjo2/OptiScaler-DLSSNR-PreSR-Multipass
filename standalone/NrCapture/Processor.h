#pragma once
#include "Gpu.h"
#include "Runtime.h"
#include "Motion.h"
#include "Scaling.h"
#include <shaders/dlssnr/DlssNr_Common.h>
namespace nr {
class Processor {
public:
    Processor(Gpu& gpu, const std::filesystem::path& runtime, bool unlockAdaMfg = false);
    ~Processor();
    void Resize(UINT width, UINT height, const Settings& settings);
    // Input is linear scRGB (also for SDR capture), arrives/leaves COMMON.
    // Records into the host's open list. Returned texture is COPY_SOURCE.
    // Estimated motion first submits a GPU flow pass to read its scene-cut flag.
    ID3D12Resource* Run(ID3D12Resource* input, const Settings& settings, bool hdr, bool resetHistory = false,
                       float desktopWhitePoint = 0, float frameTimeMs = 16.67f, int processedFps = -1, int outputFps = -1);
    UINT width = 0, height = 0;
    UINT modelWidth = 0, modelHeight = 0;
    bool historyReset = true; // Actual reset sent to NR for the last processed frame.
    // Current-to-previous full-resolution guides, valid through frame completion.
    ID3D12Resource* FullResolutionMotion() const { return fullResolutionMotion; }
    ID3D12Resource* ModelInput() const { return proxy.Get(); } // Diagnostic readback; UAV after Run completes.
private:
    friend struct ResidualProbe;
    friend struct QualityProbe; // Headless stage/guide comparisons; no live capture or UI.
    explicit Processor(Gpu& gpu); // Shader-only construction for the friend fixture; no NGX runtime.
    Gpu& gpu;
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<Motion> opticalFlow;
    std::shared_ptr<Scaling> scaling;
    ComPtr<ID3D12Resource> nativeAnswer, filteredAnswer;
    std::array<DlssNr::ModelSettings, Settings::MaxPasses> activeModels{};
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> shader, captureColour;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> constants, proxy, original, answer, output, depth, motion, passInput;
    ComPtr<ID3D12Resource> fullProxy, flowInput;
    ComPtr<ID3D12Resource> resizeInput, enlarged, boundedDlss, exposure;
    ComPtr<ID3D12Resource> sdrInput, sdrOutput;
    ComPtr<ID3D12Resource> heldInput;
    ComPtr<ID3D12Resource> labelled;
    bool labelsReadable = false;
    ID3D12Resource* fullResolutionMotion = nullptr;
    bool heldValid = false;
    bool privateDlss = false;
    UINT stride = 0;
    bool outputReadable = false;
    unsigned lastTone = 0;
    unsigned lastTransfer = 1;
    unsigned activePasses = 0;
    bool lastHdr = false;
    bool lastSdr = false;
    float lastWhitePoint = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(UINT index) const;
    void Dispatch(UINT offset, const DlssNrConstants& settings, ID3D12Resource* source,
        ID3D12Resource* model, ID3D12Resource* clean, ID3D12Resource* target, ID3D12Resource* keep);
};
}
