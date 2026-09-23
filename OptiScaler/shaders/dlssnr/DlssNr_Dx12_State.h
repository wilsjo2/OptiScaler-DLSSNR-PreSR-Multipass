#pragma once
#include "DlssNr_Dx12_ModelState.h"
#include <dlssnr/DlssNr_Placement.h>
#include <dlssnr/DlssNr_FinishedReady.h>
#include <dlssnr/PassProfiles.h>

#include <dlssnr/DlssNr_Exposure.h>
#include <gpu_time/Vitals.h>
#include <set>
#include <wrl/client.h>
#include <resource_tracking/ResTrack_Dx12.h>
#include <dlssnr/DlssNr_FinishedPictureBridge_Dx11.h>
#include <dlssnr/DlssNr_HoldParameters_Dx12.h>
#include <upscalers/ShaderPipeline_Dx12.h>

#include <dlssnr/DlssNr.h>

#include <dlssnr/DlssNr_Capture.h>
#include <dlssnr/DlssNr_PipelineCapture.h>
#include <dlssnr/DlssNr_Proxy.h>
#include <dlssnr/DlssNr_GpuLifetime.h>

#include "DlssNr_Dx12.h"
#include "DlssNr_ActiveColor.h"
#include "DlssNr_Upscaler_Dx12.h"
#include <dlssnr/DlssNr_Pipeline_Dx12.h>
#include "DlssNr_Guides.h"
#include "DlssNr_SeamClock.h"

#include <Config.h>
#include <State.h>
#include <Util.h>

#include <proxies/NVNGX_Proxy.h>
#include <hooks/D3D12_Hooks.h>
#include <gpu_time/GpuTime_Dx12.h>
#include "DlssNr_GpuTime.h"

#include <mutex>
#include <algorithm>
#include <cstring>
#include "DlssNr_ResidualPair.h"
#include "../output_scaling/OS_Dx12.h"

using DlssNr::Profiles::PassSettings;

struct DlssNr_Dx12::State
{

    // NGX result names for diagnostics.
    const char* NgxResultName(unsigned int r);

    using NrState = DlssNr::Detail::ModelStateDx12;
    NrState nr;
    DlssNr_Dx12& shader;
    struct Enlarger
    {
        template <typename T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        std::unique_ptr<DlssNr::PrivateUpscalerDx12> dlss;
        DlssNr::GpuLifetime lifetime;
        ComPtr<ID3D12Resource> input, output, depth, motion, exposure;
        ComPtr<ID3D12CommandQueue> queue;
        ID3D12CommandList* creation = nullptr;
        unsigned w = 0, h = 0, outW = 0, outH = 0;
        uint64_t lastFrame = 0;
        bool submitted = false, failed = false, depthInverted = false, readable = false, reset = true;
        bool structural = false;
        ~Enlarger() { dlss.reset(); } // Release NGX before its borrowed input/output resources.
    };
    std::unique_ptr<Enlarger> enlarger;
    std::vector<std::unique_ptr<Enlarger>> retiredEnlargers;
    bool collectingEnlargers = false;
    std::string enlargementStatus;
    void ReleaseEnlarger();
    void CollectEnlargers();
    ID3D12Resource* EnlargeMatchedResidual(ID3D12GraphicsCommandList* cmd, ID3D12Device* device, ID3D12Resource* proxy,
                                           ID3D12Resource* answer, ID3D12Resource* depth, ID3D12Resource* motion,
                                           const DlssNrFrameInfo& frame, const DlssNrConstants& resolve, bool reset,
                                           ID3D12CommandQueue* queue);

    // What the pass costs on the GPU, for the breakdown in the overlay.
    std::unique_ptr<DlssNrGpuTime> gpuTime;

    // Model-only timing separates NGX cost from encoding, copies and composition.
    std::unique_ptr<DlssNrGpuTime> ngxTime;
    std::optional<double> lastNgxTime;
    std::optional<double> lastGpuTime;

    // Writes matched before/after frames on request, so comparisons stop depending on video.
    capture::FrameCapture captureFrames;
    DlssNr::PipelineCaptureFrame* pipelineCapture = nullptr; // Owned by lifetime retirement after End.
    std::filesystem::path pipelineCaptureDirectory;
    unsigned pipelineCaptureRemaining = 0;

    unsigned long long frames = 0;

    // Logical frame identity for deferred pairing; feature readiness keeps the raw submission counter.
    DlssNrSeamClock seamClock;

    struct InputHold
    {
        struct Texture
        {
            const char* name;
            const char* alias;
            ID3D12Resource* frozen = nullptr;
            D3D12_RESOURCE_DESC desc {};
        };
        std::array<Texture, 4> textures { { { NVSDK_NGX_Parameter_Color, "DLSSD.Color" },
                                            { NVSDK_NGX_Parameter_Depth, "DLSSD.Depth" },
                                            { NVSDK_NGX_Parameter_MotionVectors, "DLSSD.MotionVectors" },
                                            { NVSDK_NGX_Parameter_ExposureTexture, "DLSSD.ExposureTexture" } } };
        NrHoldParameters_Dx12 parameters;
        D3D12_RESOURCE_DESC outputDesc {};
        bool active = false;
        unsigned route = 0;
        uint64_t generation = 0;
        ID3D12CommandList* captureCommands = nullptr;
    } inputHold;

    static bool SameHoldShape(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b);

    void ReleaseInputHold();

    void BeginInputHold(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* params,
                        const D3D12_RESOURCE_STATES* states);

    // Poll dlssnr-capture.trigger beside OptiScaler every 60 evaluations.
    void CheckCaptureTrigger();

    DlssNr::GpuLifetime lifetime;

    void ParkNrResource(ID3D12Resource*& resource);

    void ReleaseSurfacesIfFormatChanged(DXGI_FORMAT modelFormat, DXGI_FORMAT nativeFormat);
    bool PrepareSpatialResources(ID3D12Device* device, const DlssNr::Spatial::Layout& layout);
    void ReleaseSpatialResources();
    void ReleaseSupersamplers();

    ID3D12Resource* CreateScratch(ID3D12Device* device, DXGI_FORMAT format, unsigned int width, unsigned int height);

    void Barrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* res, D3D12_RESOURCE_STATES from,
                 D3D12_RESOURCE_STATES to);

    // A typeless resource cannot be viewed, and NGX builds its own views with nothing to tell it which
    // format to use. Depth is very often declared typeless, so the typed member of the same family is
    // substituted; CopyResource accepts that as a destination for the typeless original.
    DXGI_FORMAT TypedGuideFormat(DXGI_FORMAT f);

    bool IsTypeless(DXGI_FORMAT f);

    // Creates a typed twin of a guide buffer, matching everything but the format.
    ID3D12Resource* CreateGuideClone(ID3D12Device* device, ID3D12Resource* source);

    // Typeless guides are copied into a typed resource for NGX. The clone rests in COPY_DEST.
    // Return the original typed resource when no conversion is required.
    ID3D12Resource* ReadableGuide(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source,
                                  ID3D12Resource** clone);

    // Try both SR and ray-reconstruction parameter names; absent values return null.

    ID3D12Resource* GetResource(NVSDK_NGX_Parameter* params, const char* a, const char* b);

    bool TuningMatchesFeature(const Config& cfg, unsigned int requestedPasses);

    // Serialize rendering, presentation and submission callbacks; destruction can re-enter hooks.
    std::recursive_mutex mutex;

    // Restore the caller's compute bindings on every exit, without capturing NR's own state.
    struct ScopedNrStateEnvelope
    {
        ID3D12GraphicsCommandList* cmd;
        ScopedSkipHeapCapture skipHeap;

        bool tracking;
        explicit ScopedNrStateEnvelope(ID3D12GraphicsCommandList* c)
            : cmd(c), tracking(D3D12Hooks::IsRootSignatureTrackingEnabled())
        {
            D3D12Hooks::SetRootSignatureTracking(false);
        }

        ~ScopedNrStateEnvelope()
        {
            if (tracking)
                D3D12Hooks::RestoreRoot(cmd);
            D3D12Hooks::SetRootSignatureTracking(tracking);
        }
    };

    // Report each missing-input or readiness reason once per owner.
    void ReportSkipOnce(const char* reason);

    struct DeferredSrContext
    {
        State& owner;
        explicit DeferredSrContext(State& state) : owner(state) {}

        static constexpr unsigned MarkerCount = 16;
        struct Generation
        {
            ID3D12Device* device = nullptr;
            ID3D12CommandQueue* queue = nullptr; // identity/reference only; no private submissions
            unsigned w = 0, h = 0, outW = 0, outH = 0, flags = 0;
            DXGI_FORMAT inputFormat {}, outputFormat {};
            ID3D12Resource *edited = nullptr, *residualInput = nullptr, *residualOutput = nullptr, *clean = nullptr,
                           *composed = nullptr, *exposure = nullptr, *readback = nullptr;
            ID3D12QueryHeap* queries = nullptr;
            volatile UINT64* completed = nullptr;
            bool occupied[MarkerCount] {};
            unsigned nextMarker = 0;
            ID3D12Resource* accumulatedEdit[2] {};
            unsigned accumulatedIndex = 0;
            bool accumulationReadable = false, accumulationValid = false;
            bool smallReadable = false, reset = true, failed = false;
            bool rayReconstruction = false, finishedPicture = false, privateRr = false;
            DlssNr::PrivateUpscaler backend = DlssNr::PrivateUpscaler::DLSS;
            std::unique_ptr<DlssNr::PrivateUpscalerDx12> upscaler;
            DlssNr::PrivateUpscalerFrameDx12 frame;
            unsigned long long createEpoch = 0;
            unsigned long long lastBeginEpoch = 0;
            bool began = false;
            std::unique_ptr<DlssNr_Dx12> codec;
            ~Generation()
            {
                DlssNr_Dx12::Retire(std::move(codec));
                upscaler.reset(); // Completion protects all four backend histories.
                if (readback && completed)
                    readback->Unmap(0, nullptr);
                for (auto* r : { edited, residualInput, residualOutput, clean, composed, exposure, readback })
                    if (r)
                        r->Release();
                for (auto* r : accumulatedEdit)
                    if (r)
                        r->Release();
                if (queries)
                    queries->Release();
                if (queue)
                    queue->Release();
                if (device)
                    device->Release();
            }
        };

        // Preserve v0.8.4's recording limit as well as GPU-safe generation retirement.
        struct Use
        {
            Generation& g;
            ID3D12GraphicsCommandList* cmd;
            unsigned slot;
            bool valid;
            Use(Generation& gen, ID3D12GraphicsCommandList* commands) : g(gen), cmd(commands), slot(g.nextMarker)
            {
                valid = !g.occupied[slot] || g.completed[slot] != 0;
                if (!valid)
                    return;
                g.completed[slot] = 0;
                g.occupied[slot] = true;
                g.nextMarker = (slot + 1) % MarkerCount;
            }
            ~Use()
            {
                if (!valid)
                    return;
                cmd->EndQuery(g.queries, D3D12_QUERY_TYPE_TIMESTAMP, slot);
                cmd->ResolveQueryData(g.queries, D3D12_QUERY_TYPE_TIMESTAMP, slot, 1, g.readback,
                                      slot * sizeof(UINT64));
            }
        };

        std::unique_ptr<Generation> current;
        unsigned retiredCount = 0;
        DlssNr::GpuLifetime lifetime;
        std::string status = "not started";
        struct Pending
        {
            ID3D12GraphicsCommandList* cmd = nullptr;
            NVSDK_NGX_Parameter* caller = nullptr;
            ID3D12Resource* output = nullptr;
            float scale = 1;
        } pending;

        void Say(const std::string& text);
        void Cancel();
        void RetireCurrent();

        unsigned UInt(NVSDK_NGX_Parameter* p, const char* key, unsigned fallback = 0);
        float Float(NVSDK_NGX_Parameter* p, const char* key, float fallback);

        bool Allocate(Generation& g);

        void Before(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source, unsigned long long epoch,
                    unsigned long long submittedEpoch, ID3D12CommandQueue* queue, bool interop, bool rayReconstruction);

        void After(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source, unsigned long long epoch);

        void ReleaseResources();
    };
    DeferredSrContext deferredSr { *this };

    struct LateContext
    {
        State& owner;
        explicit LateContext(State& state) : owner(state) {}

        template <typename T> using ComPtr = Microsoft::WRL::ComPtr<T>;
        struct Slot
        {
            ComPtr<ID3D12Resource> depth, motion, linear, encoded, residual, cleanScene;
            ComPtr<ID3D12Resource> response[2]; // per-slot ping-pong; reused only after its GPU fence
            unsigned responseIndex = 0, responseWidth = 0, responseHeight = 0;
            uint64_t responseEpoch = 0;
            float responseExposure = 1.0f;
            DXGI_COLOR_SPACE_TYPE responseSpace = DXGI_COLOR_SPACE_CUSTOM;
            bool cleanSceneValid = false, responseValid = false;
            ComPtr<ID3D12Fence> fence;
            ComPtr<ID3D12CommandQueue> producerQueue;
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> commands;
            ID3D12CommandList* producer = nullptr; // identity only; never dereferenced
            DlssNrFrameInfo frame {};
            uint64_t ready = 0, done = 0, serial = 0;
            bool pending = false, submitted = false, residualOnly = false, sceneLinear = true;
        };
        std::array<Slot, 4> slots;
        ComPtr<ID3D12Device> device;
        ComPtr<ID3D12CommandQueue> producerQueue;
        Dx11FinishedPictureBridge dx11;
        // One clean presentation snapshot for the owner, never one per rotating backbuffer/slot.
        ComPtr<ID3D12Resource> heldFinished;
        ComPtr<ID3D12Fence> heldFence;
        ComPtr<ID3D12CommandQueue> heldQueue;
        uint64_t heldReady = 0, heldGeneration = 0;
        D3D12_RESOURCE_STATES heldState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        DXGI_COLOR_SPACE_TYPE heldSpace = DXGI_COLOR_SPACE_CUSTOM;
        bool heldValid = false, heldFailed = false;
        bool reportedQueueDelay = false;
        Slot* heldSlot = nullptr;
        uint64_t heldSlotSerial = 0;
        uint64_t serial = 0, successes = 0;
        std::string status = "Waiting for a finished picture.";
        bool reset = true;
        std::atomic<bool> tracking { false };

        void Say(const char* message);

        bool Finished(const Slot& slot);

        void Cancel();

        bool Clone(ComPtr<ID3D12Resource>& copy, ID3D12Resource* source);

        Slot* Acquire(ID3D12GraphicsCommandList* cmd);

        void Arm(Slot& slot, ID3D12GraphicsCommandList* cmd);

        void Capture(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* params, bool rr);

        bool CaptureResidual(ID3D12GraphicsCommandList* cmd, ID3D12Resource* clean, ID3D12Resource* residual,
                             float scale, bool sceneLinear, bool reset);
    };
    LateContext late { *this };

    void FinishedPictureResetCommandList(ID3D12CommandList* cmd);

    bool WaitForFinishedPicture();

    std::string FinishedPictureStatus();

    void FinishedPictureSubmitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists);

    DXGI_COLOR_SPACE_TYPE FinishedColorSpace(IDXGISwapChain* swapchain, DXGI_FORMAT format);

    void ApplyToFinishedPictureDx11(IDXGISwapChain* swapchain);

    bool ApplyFinishedColor(ID3D12Resource* color, ID3D12CommandQueue* queue, DXGI_COLOR_SPACE_TYPE colorSpace,
                            bool gameFrameHandoff = false);

    bool PendingFinishedCapture(ID3D12Resource* color, ID3D12CommandQueue* queue, DXGI_COLOR_SPACE_TYPE colorSpace,
                                DlssNr::XeFGCapture& facts);
    void CloseFinishedCaptures(uint64_t throughSerial);

    bool PrepareRunModels(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, const DlssNrFrameInfo& frame,
                          const D3D12_RESOURCE_DESC& desc, DlssNr::ColorExtent native, DlssNr::ColorExtent work,
                          float workScale, unsigned int requestedPasses, bool spatial);
    struct EncodeContext
    {
        ID3D12GraphicsCommandList* cmdList;
        ID3D12Device* device;
        ID3D12Resource* target;
        D3D12_RESOURCE_STATES targetState;
        const DlssNrFrameInfo& frame;
        float workScale;
        bool targetSupportsUav;
        bool spatial = false;
        bool encodeSucceeded = false;
        float whitePoint = 1.0f;
        ID3D12Resource* modelInput = nullptr;
        ID3D12Resource* exposure = nullptr;
        DlssNrConstants exposureConstants {};
    };
    void EncodeInput(EncodeContext& context);
    DlssNrConstants MakeResolveConstants(const EncodeContext& context, unsigned int effectivePasses);
    OptiScaler::RollingVitals vitals;
    void EndGpuTiming(ID3D12GraphicsCommandList* cmdList);

    void Run(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* colour, ID3D12Resource* depth, ID3D12Resource* motion,
             ID3D12Resource* output, const DlssNrFrameInfo& frame, ID3D12CommandQueue* timingQueue);

    std::string DeferredDlssStatus();

    void RetryAfterFailure();

    // Adapt the game's NGX parameters into the explicit NR frame contract.
    void EvaluateInternal(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params, bool beforeUpscale,
                          ID3D12CommandQueue* timingQueue, bool rayReconstruction, unsigned long long submissionEpoch,
                          bool interop);

    void ReleaseResources();

    struct GuideReport
    {
        bool valid;
        bool depthInverted;
        float mvScaleX;
        float mvScaleY;
        unsigned int guideW;
        unsigned int guideH;
        unsigned int frameW;
        unsigned int frameH;
        bool operator==(const GuideReport&) const = default;
    };
    GuideReport loggedGuides {};
    struct ComposeReport
    {
        bool valid;
        float whitePoint;
        float transfer;
        float colour;
        float maxRatio;
        unsigned int passthrough;
        unsigned int debugView;
        unsigned int compareMode;
        unsigned int residual;
        unsigned int workW;
        unsigned int workH;
        unsigned int passes;
        bool operator==(const ComposeReport&) const = default;
    };
    ComposeReport loggedCompose {};

    std::set<std::string> seen;
    unsigned long long resets = 0;
    bool reportedHdr = false;
    bool reportedHdrValue = false;
    bool reportedBefore = false;
    bool warnedSuper = false;
    unsigned int loggedConfigured = 0;
    unsigned int loggedEffective = 0;
    unsigned int lastSuper = 0;
    unsigned long long lastSplitLog = 0;
    unsigned lastFinishedMode = 0;

    bool modelRunning = false;
    ID3D12Resource* buffer = nullptr;
    D3D12_RESOURCE_STATES bufferState = D3D12_RESOURCE_STATE_COMMON;
    uint32_t featureFlags = 0;
    DlssNr::ControlRequests controls = DlssNr::ReadControlRequests();
    explicit State(DlssNr_Dx12& owner) : shader(owner) {}
    void ConsumeControls();
    void Publish();
    ~State()
    {
        WaitForFinishedPicture();
        ReleaseInputHold();
        ReleaseResources();
        SAFE_RELEASE(buffer);
    }
};
