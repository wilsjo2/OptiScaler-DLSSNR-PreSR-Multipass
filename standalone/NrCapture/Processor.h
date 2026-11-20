#pragma once
#include "Gpu.h"
#include "Runtime.h"
#include "Motion.h"
#include "Scaling.h"
#include "Spatial.h"
#include <vector>
#include <shaders/dlssnr/DlssNr_Common.h>
#include <dlssnr/NrPassBorrow.h>
namespace nr {
// Cuts per network evaluation of a frame (re-apply passes are not evaluations; cuts[0] is None).
// Returns false when nothing borrows, so the caller leaves NrPassBorrow out of the frame.
bool BorrowCuts(const Settings& settings, std::array<NrPassBorrow::Cut, Settings::MaxPasses>& cuts, unsigned& evaluations);
// Install NrPassBorrow's NvAPI hook and route its messages to capture.log. Call once, before any Runtime exists.
// False (logged) when unavailable; borrowing then stays off and nothing else changes.
bool InstallPassBorrow();
// The experimental NVFP4 feed-forward kernels (nvfp4/NrFfnV2.h). InstallPassBorrow() puts them in place, switched off;
// Processor::Run follows Settings::nvfp4FeedForward. Assets: the nvfp4 folder beside the executable.
void ApplyNvfp4(bool wanted, bool trained = false, bool everySize = false); // trained: ffn_v2_trained.bin where the model's folder has it;
                                                    // everySize: no least frame size (a pass-in-one model's own feed-forward records must run at any size)
std::string Nvfp4Status();
// The experimental NVFP4 fused-block kernels (nvfp4/NrSwin4.h), put in place the same way. Arguments: Settings::nvfp4Swin,
// nvfp4SwinLayers, nvfp4SwinTwo, nvfp4SwinTrained. Assets: the folder nvfp4/swin4 beside the executable.
// trained: the weight set (NrSwin4::Options::set): 0 the first, 1 the trained one, 2.. a pass-in-one model's.
void ApplyNvfp4Swin(unsigned widths, unsigned long long layers, unsigned twoTerms, unsigned trained, bool sets = false); // sets: a pass may ask for a further set
std::string Nvfp4SwinStatus(bool counters = false);
// The experimental NVFP4 kernels for the rest of Model A's pass (nvfp4/NrNet4.h), put in place the same way.
// Arguments: Settings::nvfp4Split, nvfp4Vit, nvfp4PrePost, nvfp4Net. Assets: the folders nvfp4/split4, nvfp4/vit4 and
// nvfp4/prepost4 beside the executable.
// trained: bit 0 the split blocks, bit 1 the transformer use their trained weight records (Settings::nvfp4SplitTrained,
// nvfp4VitTrained) where the asset folder has them.
void ApplyNvfp4Net(unsigned split, unsigned vit, unsigned prepost, const std::string& text, unsigned trained);
std::string Nvfp4NetStatus(bool counters = false);
// A NrNet4::Parse text into settings: its group masks into nvfp4Split, nvfp4Vit and nvfp4PrePost, all of it into nvfp4Net.
void SetNvfp4Net(Settings& settings, const std::string& text);
void ReleaseNvfp4(ID3D12Device* device); // Only once the device's queue has drained, or the device is lost.
// The FP8 pass-in-one models (nvfp4/NrFp8.h), put in place by InstallPassBorrow() the same way: on while a pass has
// one (Settings::Fp8Pass). Throws when they cannot be loaded. Assets: the folder nvfp4/fp8 beside the executable.
void ApplyFp8(bool wanted);
std::string Fp8Status(bool counters = false);
class Processor {
public:
    Processor(Gpu& gpu, const std::filesystem::path& runtime, bool unlockAdaMfg = false);
    ~Processor();
    void Resize(UINT width, UINT height, const Settings& settings);
    // Input is linear scRGB (also for SDR capture), arrives/leaves COMMON.
    // Records into the host's open list. Returned texture is COPY_SOURCE.
    // Estimated motion records into the same list; a detected scene cut resets history on the next frame.
    ID3D12Resource* Run(ID3D12Resource* input, const Settings& settings, bool hdr, bool resetHistory = false,
                       float desktopWhitePoint = 0, float frameTimeMs = 16.67f, int processedFps = -1, int outputFps = -1);
    UINT width = 0, height = 0;
    UINT modelWidth = 0, modelHeight = 0;
    bool historyReset = true; // Actual reset sent to NR for the last processed frame.
    // Checks: without Settings::estimateMotion the passes keep their history (zero vectors) unless
    // the caller resets it, instead of every frame being a first frame. The application leaves it false.
    bool keepHistoryWithoutMotion = false;
    bool fieldReused = false; // Settings::editInterval: the last frame ran no model, the kept edit was added to it.
    // Current-to-previous scene motion used by NR, valid through frame completion: the whole picture at the
    // first pass's model size, in that size's pixel units. Null when NR estimated none this frame.
    ID3D12Resource* SceneMotion() const { return sceneMotion; }
    ID3D12Resource* ModelInput() const { return advanced ? chainInput.Get() : proxy.Get(); } // Diagnostic readback; UAV after Run completes.
    std::pair<UINT, UINT> PassModelSize(unsigned pass) const;
    // Diagnostic: NrPassBorrow outcome of each network evaluation of the last frame; 0 evaluations when it did not borrow.
    std::array<NrPassBorrow::Outcome, Settings::MaxPasses> borrowOutcomes{};
    unsigned borrowEvaluations = 0;
private:
    friend struct ResidualProbe;
    friend struct ReapplyProbe; // Shader-only re-apply fixture.
    friend struct QualityProbe; // Headless stage/guide comparisons; no live capture or UI.
    friend struct DepthProbe; // Controlled raw-NR depth sensitivity check.
    friend struct PassResolutionProbe; // Headless per-stage history diagnostics.
    explicit Processor(Gpu& gpu); // Shader-only construction for the friend fixture; no NGX runtime.
    Gpu& gpu;
    bool modelStage = false, composeOnly = false, advanced = false;
    bool stageResetPending = true;
    bool clampStageInput = false;
    std::vector<std::shared_ptr<Processor>> stages;
    std::vector<Settings> stageSettings;
    ID3D12Resource* externalAnswer = nullptr;
    ID3D12Resource* preparedProxy = nullptr;
    ID3D12Resource* chainMotionInput = nullptr;
    ID3D12Resource* sharedMotionVectors = nullptr;
    ID3D12Resource* usedMotionVectors = nullptr;
    bool reuseChainMotion = true; // Test probe can compare the independent-estimation reference.
    ComPtr<ID3D12Resource> chainInput;
    std::shared_ptr<Spatial> spatialShader;
    DlssNr::Spatial::Layout spatialLayout{};
    ComPtr<ID3D12Resource> packedProxy, packedAnswer, packedDepth, packedMotion, packedPassInput;
    ComPtr<ID3D12Resource> gridOutput;
    bool gridReadable = false;
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<Motion> opticalFlow;
    std::shared_ptr<Scaling> scaling;
    ComPtr<ID3D12Resource> nativeAnswer, filteredAnswer;
    ComPtr<ID3D12Resource> reapplied, packedReapplied, chainReapplied; // Re-apply results; created on first use.
    std::array<DlssNr::ModelSettings, Settings::MaxPasses> activeModels{};
    unsigned activeReapply = 0; // Settings::ReapplyMask of the created features.
    std::array<NrPassBorrow::Cut, Settings::MaxPasses> activeCuts{};
    int borrowEvaluation = -1; // Next evaluation index while a frame borrows; -1 leaves NGX calls untouched.
    struct BorrowFrame { // Ends a borrowing frame and stops wrapping evaluations, also when recording throws.
        bool active; int& evaluation;
        ~BorrowFrame() { evaluation = -1; if (active) NrPassBorrow::EndFrame(); }
    };
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> shader, captureColour;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> constants, proxy, original, answer, output, depth, motion, passInput;
    ComPtr<ID3D12Resource> fullProxy, flowInput;
    ComPtr<ID3D12Resource> resizeInput, enlarged, boundedDlss, exposure;
    // Settings::editInterval: resizeInput holds the edit of the last evaluated frame; fieldAge counts the frames since.
    bool fieldValid = false; unsigned fieldAge = 0;
    bool activeFullField = false; // a 100 % pass that keeps its edit: resizeInput and nativeAnswer exist at the full size
    ComPtr<ID3D12Resource> sdrInput, sdrOutput;
    ComPtr<ID3D12Resource> heldInput;
    ComPtr<ID3D12Resource> labelled;
    bool labelsReadable = false;
    ID3D12Resource* sceneMotion = nullptr;
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
    void ResizeLegacy(UINT width, UINT height, const Settings& settings);
    ID3D12Resource* RunLegacy(ID3D12Resource* input, const Settings& settings, bool hdr, bool resetHistory,
                             float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps);
    ID3D12Resource* RunChain(ID3D12Resource* input, const Settings& settings, bool hdr, bool resetHistory,
                            float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps);
    ID3D12Resource* Preview(ID3D12Resource* input, const Settings& settings);
    D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(UINT index) const;
    void Dispatch(UINT offset, const DlssNrConstants& settings, ID3D12Resource* source,
        ID3D12Resource* model, ID3D12Resource* clean, ID3D12Resource* target, ID3D12Resource* keep, ID3D12Resource* fifth = nullptr);
    void Evaluate(ID3D12Resource* colour, ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
        UINT width, UINT height, bool reset, unsigned pass);
    // Descriptor/constant slots: 0..98 step 7 as before; re-apply run ending at pass p uses ReapplySlot(p) and the
    // clamp from a re-apply result uses ReapplyClampSlot, so no slot is rewritten with other inputs in one list.
    static constexpr UINT ReapplySlot(unsigned pass) { return 105 + 7 * (pass - 1); }
    static constexpr UINT ReapplyClampSlot = 105 + 7 * (Settings::MaxPasses - 1), SlotCount = ReapplyClampSlot + 7;
};
}
