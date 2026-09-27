#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <string>

#include <shaders/dlssnr/DlssNr_Common.h>
#include <nvsdk_ngx.h>

// DLSS 5 Neural Rendering, run over the upscaler's output.
//
// Neural Rendering is a post-process, not an upscaler and not a denoiser: it takes a finished frame plus
// depth and motion vectors and synthesises detail. NVIDIA ships no public integration for it, so it is
// driven directly through nvngx_dlssnr.dll as feature 18.
//
// OptiScaler is the right host for it because of one thing it knows that an external hook cannot: which
// NGX evaluate belongs to the upscaler and which to frame generation. Both are handed depth and motion
// vectors, so anything guessing from the parameter block alone attaches to both and runs the model twice
// per rendered frame. Here it is a lookup on the feature handle.
class Config;

namespace DlssNr
{
inline constexpr unsigned int MaxPassCount = 30;
inline constexpr unsigned int DefaultMaxPassCount = 3;

// The model runs immediately after the game's upscaler, before the interface is drawn. It is shown a
// display-referred proxy of that frame -- the sort of picture it was trained on -- and its answer is
// composed back over the untouched original.
// Runs the model over Output on the same command list, immediately after the upscaler has written it.
// Called only for upscaler evaluates -- never for frame generation, which is the whole point.
//
// Safe to call every frame; it builds what it needs on first use and disables itself for the session if
// anything fails, rather than retrying into a crash.
// timingQueue is the queue this command list will be executed on, when the caller knows it.
// State::currentCommandQueue only exists once a D3D12 swapchain has been created, which a Vulkan
// game never does -- so without this the pass runs and never reports what it cost.
// rayReconstruction identifies the feature for history reset and the SR-only deferred experiment.
// Placement and model cost controls are shared by SR and RR+SR.
void EvaluateAfterUpscale(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params,
                          ID3D12CommandQueue* timingQueue = nullptr, bool rayReconstruction = false,
                          unsigned long long submissionEpoch = 0);

// Runs the same pass over Color immediately before SR or RR+SR consumes it. The call is a no-op
// unless RunBeforeSR is enabled. Color is returned in its original readable state.
void EvaluateBeforeUpscale(ID3D12GraphicsCommandList* cmdList, NVSDK_NGX_Parameter* params,
                           ID3D12CommandQueue* timingQueue = nullptr, unsigned long long submissionEpoch = 0,
                           bool rayReconstruction = false);

// Frame generation titles tag their UI layer through Streamline; a copy of it makes the HUD mask
// exact at the finished frame. Called at tag time.

// The settings panel, drawn inside OptiScaler's menu.
void RenderMenu(::Config* config, float menuResScale);

// Clears the session failure latch, so a failure caused by transient thrash does not cost a restart.
void RetryAfterFailure();

std::string FinishedPictureStatus();
bool WaitForFinishedPicture();
void FinishedPictureResetCommandList(ID3D12CommandList* cmd);
void FinishedPictureSubmitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists);
void ApplyToFinishedPicture(IDXGISwapChain* swapchain, ID3D12CommandQueue* queue);
void FinishedPictureColorSpace(IDXGISwapChain* swapchain, DXGI_COLOR_SPACE_TYPE colorSpace);

// Finished-picture colour-space record shared by the generic route and the XeFG owned
// handoff. Same GUID as Late::colorSpaceKey (shaders/dlssnr/DlssNr_Late.inl): one record,
// read by both paths.
inline constexpr GUID FinishedColorSpaceKey = {
    0x34a31e7b, 0x84c5, 0x44ef, { 0xa7, 0x4d, 0x6b, 0xd3, 0x60, 0x8c, 0xe5, 0x22 }
};

// XeFG owned application-frame handoff. The Present-site wiring in
// framegen/xefg/XeFG_Dx12.cpp drives these on the XeFG-retained application queue (never
// State.currentCommandQueue): the capture facts are queried first, the composition runs only
// after the handoff core (dlssnr/DlssNr_XeFGHandoff.h) accepted the frame.
// Swapchain/buffer/colour-space queries must happen before these calls - they take the NR
// locks (see ApplyToFinishedPicture for the ordering rule).
struct XeFGCapture
{
    bool exists = false;    // a pending capture matches this finished picture
    bool submitted = false; // its producer command list was executed
    bool ready = false;     // same application queue, or its readiness fence completed
    bool sameQueue = false; // the capture's producer queue is the application queue
    uint64_t serial = 0;    // the pending capture's slot serial
};
bool XeFGPendingCapture(ID3D12Resource* picture, ID3D12CommandQueue* queue, DXGI_COLOR_SPACE_TYPE space,
                        XeFGCapture& facts);
bool ApplyXeFGPicture(ID3D12Resource* picture, ID3D12CommandQueue* queue, DXGI_COLOR_SPACE_TYPE space);
void XeFGCloseCaptures(uint64_t throughSerial);

// Asks the model whether it will work on Direct3D 11 at all, once, and logs the answer.
//
// The bridge exists because of a claim nobody tested: "the model refuses on DX11, it answers
// FeatureNotSupported". Nothing in this project has ever called the snippet's own D3D11 entry points
// -- it exports ten of them, implemented in ngx_d3d11.cpp and sharing CreateFeatureCommon and
// EvaluateFeatureCommon with the D3D12 path. Nothing is created and nothing changes; it resolves the
// entry points and initialises on the game's own device, which is where a refusal would appear.
void ProbeD3D11(void* d3d11Device);

// What scale this game's buffer is on, measured from the untouched copy of each frame.
//
// A suggestion only. Nothing applies it: the menu shows it and the user takes it or does not, which
// keeps the number visible and adjustable rather than a value that moved on its own. Confidence is
// how settled recent readings are -- 1 means they agree, 0 means the scene is changing under the
// measurement and no single value would serve.
struct CalibrationReading
{
    float suggestion = 0.0f;

    // How much recent readings agree. This is steadiness, not correctness: a frozen frame agrees with
    // itself perfectly, so a loading screen scores full marks for a number that means nothing. Read it
    // together with usable.
    float steadiness = 0.0f;

    unsigned long long samples = 0;

    // Whether the scene is worth measuring at all. False when the frame is already tone mapped -- the
    // divisor does nothing there and the reading would be a meaningless 0.9 -- or when too little of
    // the picture is lit to say where the top of the range is. A dark cave gives a small number very
    // steadily, which is the trap this exists to close.
    bool usable = false;
    const char* why = "";
};

CalibrationReading Calibration();

// Whether the model is loaded and running, for the overlay.
bool IsRunning();
// Private residual-upscaler status; separate from the NR model's own running status/time.
std::string DeferredDlssStatus();

// Why it is not, if it is not. Empty while it is running or has not been tried yet.
const char* FailureReason();

// What the game offers by way of exposure. Observed every frame whether or not the setting is on, so
// the menu can say whether turning it on would do anything here.
struct ExposureStatus
{
    unsigned long long seenFrames = 0; // evaluates observed; 0 means nothing has run yet
    bool offeredNow = false;           // a texture on the most recent frame
    bool everOffered = false;          // a texture on any frame so far
    float exposure = 0.0f;             // last value read back, 0 if none
    float preExposure = 1.0f;
};

ExposureStatus GameExposureStatus();

// The white point the exposure meter has settled on, or 0 if it has not taken a reading yet. For the
// overlay, so the number in use is visible rather than inferred.

// What the pass last cost on the GPU, in milliseconds, or nothing if it has not been measured yet.
std::optional<double> LastGpuTime();

// What the white point meter last settled on, or 0 when it is not running. For the menu.

// Writes a run of consecutive frames, each as the upscaler produced it and again after the model's edit.
// The pair is a control: same frames, same run, one variable.
void RequestCapture(unsigned int frames);
bool CaptureInProgress();

void Shutdown();
} // namespace DlssNr
