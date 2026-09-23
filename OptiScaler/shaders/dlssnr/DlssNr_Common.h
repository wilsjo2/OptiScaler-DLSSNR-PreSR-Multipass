#pragma once

// Everything about the Neural Rendering composition pass that is not Direct3D 12.
//
// Two kinds of thing live here. The constants the shader reads, so a Vulkan implementation can share
// the struct rather than redefine it and drift; and the parameter names the model is driven by, which
// are the model's own and identical whatever API is calling it.
//
// The Direct3D 12 side is DlssNr_Dx12, which implements Shader_Dx12 the way RCAS and Output Scaling
// do. The model itself is separate again: creating and evaluating an NGX feature is not a dispatch,
// so it does not belong in a shader class.

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Which of the passes a dispatch is. One shader, because they read and write the same set of
// resources and differ only in what they compute.
enum DlssNrMode : uint32_t
{
    DlssNrMode_Encode = 0,         // the frame -> a tone-mapped proxy, plus an untouched copy
    DlssNrMode_Resolve = 1,        // proxy + the model's answer + the untouched copy -> the edited frame
    DlssNrMode_Downsample = 2,     // the proxy -> a smaller proxy, when the model works below full size
    DlssNrMode_EncodeResidual = 5, // NR-composed minus original; signed difference encoded around 0.5
    DlssNrMode_ApplyResidual = 6,  // decode private DLSS result and add to clean SR output
    DlssNrMode_UnitExposure = 7,   // constant exposure for the private DLSS feature
    DlssNrMode_ClampProxy = 8,     // restore the encoded RGB range between model passes
    DlssNrMode_EncodeProxyResidual = 9,
    DlssNrMode_Meter = 3,
    DlssNrMode_AutoExposure = 11,
    DlssNrMode_ResizePrivateGuides = 10,
    DlssNrMode_EncodeResizeField = 12
};

inline bool DlssNrUsesDlssEnlargement(uint32_t transfer) { return transfer == 2 || transfer == 4; }

// The spatial mode is used until the caller has supplied an enlarged DLSS carrier.
inline uint32_t DlssNrSpatialTransfer(uint32_t transfer)
{
    return DlssNrUsesDlssEnlargement(transfer) ? transfer - 1 : std::min(transfer, 3u);
}

// Finished-colour shader's exposure-normalised brightness response, -12..12 stops.
constexpr uint32_t kDlssNrHdrCurveBins = 48;

// What the composition shader reads.
//
// The model does not replace the frame. It is shown a tone-mapped proxy of the picture, and its
// answer is transferred back onto the real frame -- so most of these describe how much of that answer
// to take, not what the model should do.
// Aligned to 256 because a constant buffer view's size must be a multiple of it. Without this the
// buffer is created at the struct's natural size, the view is invalid, and the device is removed a
// few milliseconds later -- with nothing in any log to say why. Every other shader here does the
// same thing; it is not optional.
// What the caller knows about the frame, and the pass cannot work out for itself.
//
// Everything here is a property of how the game encodes its buffers, not a setting: the user's
// choices -- preset, intensity, strengths, paper white -- stay in Config, so a caller placing this
// pass in a new pipeline does not have to plumb a dozen sliders through it.
//
// Allocation sizes come from the resource descriptors. Before SR, the reported render subrect
// also determines the active colour size; padded colour is copied through a compact work texture.
struct DlssNrFrameInfo
{
    uint32_t Width = 0, Height = 0, GuideWidth = 0, GuideHeight = 0;
    bool PipelineManagedStates = false;
    // Which way round depth runs. The game states this when it creates its own upscaler.
    bool DepthInverted = false;

    // How the game encodes its motion vectors, as the game itself reports it. Passed through: every
    // resource already carries a subrect saying how big it is, so scaling by the resolution ratio on
    // top of that counts it twice.
    float MvScaleX = 1.0f;
    float MvScaleY = 1.0f;

    // Throw away the model's history. Set it on a cut, a teleport, or the first frame of a feature.
    bool Reset = false;

    // Whether the colour buffer holds linear, open-ended light or a frame that has already been
    // through a tonemapper. Getting this wrong encodes an encoded frame a second time, which looks
    // washed out and banded.
    bool ColourIsLinearHdr = true;

    // The SR colour input arrives readable, whereas a completed upscaler output normally arrives as
    // a UAV. The DX12 pass uses this to preserve the caller's state and to fall back through a copy
    // when a pre-SR colour resource was not created with UAV support.
    bool BeforeUpscale = false;
    // Owned copy, not the game's Color: always arrives/returns NON_PIXEL_SHADER_RESOURCE.
    bool PrivateColorCopy = false;
    bool FinishedPicture = false;
    uint32_t OutputArrivalState = 0;
    float WhitePointOverride = 0.0f;
    bool IndependentCommands = false; // owned command list, no game root signature to restore
    // Reset temporal history when switching between ordinary SR and Ray Reconstruction.
    bool RayReconstruction = false;

    // Submission epoch supplied by the caller. Native DX12 uses the wrapped swapchain Present count;
    // the DX11/Vulkan bridges use their successfully submitted frame counter. A feature created in an
    // epoch is never evaluated until this value changes.
    unsigned long long SubmissionEpoch = 0;
    float FrameTimeMs = 16.67f;

    // The scale the game multiplied its buffer by for float precision, which DLSS is told so it can
    // undo it. Usually 1. Divided out before the exposure is applied, exactly as FSR's PrepareRgb does.
    float PreExposure = 1.0f;
    void* ExposureTexture = nullptr; // Borrowed for this synchronous evaluation only.
    uint32_t ExposureState = 0;

    // How much of the depth and motion vector textures the game actually rendered into.
    // Before SR this also selects the origin-zero active colour rectangle, not its allocation.
    //
    // Not the same thing as how big those textures are, and the difference is the whole point. A game
    // with dynamic resolution allocates its guides once at the largest size it will ever need and
    // then renders into the top-left corner of them, telling the upscaler how much is real through
    // DLSS.Render.Subrect.Dimensions. Sizing the guides from the resource instead means handing the
    // model whatever was left in the margin -- stale depth and stale vectors -- and calling it scene.
    //
    // Zero means the game did not say, in which case the resource's own size is the best answer
    // available and is what gets used.
    unsigned int RenderSubrectWidth = 0;
    unsigned int RenderSubrectHeight = 0;

    unsigned int DepthSubrectBaseX = 0;
    unsigned int DepthSubrectBaseY = 0;
    unsigned int MotionSubrectBaseX = 0;
    unsigned int MotionSubrectBaseY = 0;

    // DLSS permits motion vectors at either render or output resolution. This flag comes from the
    // feature-create flags and decides which valid-region dimensions apply to the motion texture.
    bool MotionVectorsLowResolution = false;
    // DLSS output extent, independent of the NR injection point or working scale.
    unsigned int OutputWidth = 0;
    unsigned int OutputHeight = 0;
};

struct alignas(256) DlssNrConstants
{
    uint32_t Mode;
    float WhitePoint;

    uint32_t Width;
    uint32_t Height;

    // How much of the model's edit lands, and how much of it is allowed to be colour rather than
    // luminance. Separating the two is what keeps saturated highlights from shifting hue.
    float TransferStrength;
    float ColourStrength;

    uint32_t DebugView;

    // A ceiling on how far a pixel may be brightened. The transfer is a ratio, and a ratio against a
    // near-black proxy pixel is unbounded without one.
    float MaxRatio;

    // Set when the game's buffer is already tone-mapped, in which case there is nothing to convert
    // and the transfer is the identity.
    uint32_t Passthrough;

    float MvScaleX;
    float MvScaleY;

    // Depth and motion vectors come from the upscaler's inputs and so may be at render resolution
    // while colour and output are at display resolution.
    uint32_t GuideWidth;
    uint32_t GuideHeight;

    // Showing the pass against itself. 0 off, 1 side by side, 2 a wipe.
    //
    // Both are drawn by the resolve rather than by a pass of their own, because the resolve is the
    // one place that already holds the frame as the upscaler produced it and the frame the model
    // edited. Comparing them anywhere else would mean keeping a second copy of one of them.
    uint32_t CompareMode;
    float CompareSplit;

    // How much of the frame side by side shows. 1 fits the whole thing at its right shape and
    // letterboxes what is left over; 2 fills the half and crops to the middle instead.
    float CompareZoom;

    // Which side the edited frame is on. Swapping matters because the eye is not even-handed about
    // left and right, so a difference can look like an improvement purely from where it sits.
    uint32_t CompareSwap;

    // Below-size reconstruction: 0 classic, 1/2 matched residual spatial/DLSS,
    // 3/4 relative lighting and chromaticity spatial/DLSS. Native size bypasses resizing.
    //
    // Classic composes the model's own low-resolution picture against the full-resolution frame, so
    // the two disagree by the blur the downsample introduced as well as by the edit -- and the
    // composition reads that disagreement as headroom the frame has and the model never saw. Matched
    // residual takes only the model's *difference* from low resolution and lays it on the frame's own
    // full-resolution proxy, so the two pictures being compared are at the same scale and the only
    // thing carried up from small is the edit itself.
    //
    // The idea and the cube-scaled residual are hhkbble's, from the multi-pass PR against this fork.
    uint32_t Transfer;

    // What the debug views are multiplied by on their way out.
    //
    // They have to be scaled into the frame's units or the game's tonemapper shows them wrong, but
    // scaling them by the live white point makes the instrument move with the thing being measured:
    // two captures at different exposures then differ by the exposure, whatever the edit did. This
    // is the user's own multiplier, independent of the image's brightness.
    float DebugScale;

    // The reversible-proxy mode. 0 soft knee + our composition (default), 1 unclipped Neutwo proxy +
    // our composition, 2 Neutwo proxy + pure-inverse replace (model's answer straight back, no
    // composition). Trailing field, mirroring the shader's cbuffer, so the layout stays a flat run of
    // 4-byte scalars that C++ and HLSL agree on.
    uint32_t ReversibleMode;

    // 0 = output the clean upscaler frame (the pass still runs, so Hold frame keeps a frozen frame
    // to A/B against), 1 = apply the model's edit. Trailing scalar, mirrored in the shader cbuffer.
    uint32_t ApplyModel;

    uint32_t Reserved;   // Preserve the shared constant-buffer layout.
    float ResidualScale; // Scene pre-exposure used to encode/decode the private residual carrier.
    // Optional colour-based final-composition mask. Not the runtime's semantic mask.
    uint32_t SkinProtection;
    uint32_t ShowSkinMask;
    float SkinDetail;
    float SkinColour;
    float EnvironmentDetail;
    float EnvironmentColour;

    // ResidualAcrossRR v2 only (dlssnr_residual.hlsl). History blend rate for the MV-reprojected
    // accumulator, 0..1. Read only by that separate shader; dlssnr.hlsl never declares it. Appended
    // here rather than in a new struct so DispatchResidualPass reuses the existing constant upload --
    // it lands inside the 256-byte alignas padding, so sizeof(DlssNrConstants) is unchanged.
    float ResidualBlend;
    uint32_t ResidualHistoryValid;
    uint32_t ResidualMotionBaseX;
    uint32_t ResidualMotionBaseY;
    float ReplaceDetailStrength;
    float ModelWorkScale;
    float ResidualConfidenceSensitivity;
    uint32_t ExposureMode;
    float PreExposure;
    float ExposureTrim;
    float ExposureProtection;
    uint32_t ExposureAnchorCount;
    uint32_t ExposureSourceWidth;
    uint32_t ExposureSourceHeight;
    uint32_t ExposurePadding;
    float ExposureAnchors[16]; // Eight float2 pairs, packed as four float4s in HLSL.
};
static_assert(sizeof(DlssNrConstants) == 256);
static_assert(offsetof(DlssNrConstants, ReplaceDetailStrength) == 132);
static_assert(offsetof(DlssNrConstants, ExposureAnchors) == 176);

// Local mode numbering for dlssnr_residual.hlsl (a separate blob / PSO from the DlssNrMode shader).
enum DlssNrResidualMode : uint32_t
{
    DlssNrResidualMode_Accumulate = 0, // (edited - original) blended into the reprojected history
    DlssNrResidualMode_Apply = 1,      // base + delta * TransferStrength, after RR+SR
};

class DlssNr_Common
{
  public:
    // Keep the controls consumed by the shared shader identical across graphics APIs.
    template <typename ConfigType>
    static DlssNrConstants MakeConstants(DlssNrMode mode, uint32_t width, uint32_t height, float whitePoint,
                                         bool linearHdr, const ConfigType& config)
    {
        DlssNrConstants constants {};
        constants.Mode = mode;
        constants.Width = width;
        constants.Height = height;
        constants.WhitePoint = whitePoint;
        constants.Passthrough = linearHdr ? 0u : 1u;
        constants.TransferStrength = config.DlssNrTransferStrength.value_or_default();
        constants.ColourStrength = config.DlssNrColourStrength.value_or_default();
        constants.DebugView = config.DlssNrDebugView.value_or_default();
        constants.MaxRatio = config.DlssNrMaxRatio.value_or_default();
        constants.Transfer = DlssNrSpatialTransfer(config.DlssNrTransfer.value_or_default());
        constants.DebugScale = config.DlssNrWhitePointScale.value_or_default();
        constants.CompareMode = config.DlssNrCompare.value_or_default();
        constants.CompareSplit = config.DlssNrCompareSplit.value_or_default();
        constants.CompareZoom = std::max(1.0f, config.DlssNrCompareZoom.value_or_default());
        constants.CompareSwap = config.DlssNrCompareSwap.value_or_default() ? 1u : 0u;
        constants.ReversibleMode = config.DlssNrReversibleMode.value_or_default();
        constants.ApplyModel = config.DlssNrApplyModel.value_or_default() ? 1u : 0u;
        constants.SkinProtection = config.DlssNrSkinProtection.value_or_default() ? 1u : 0u;
        constants.ShowSkinMask = config.DlssNrShowSkinMask.value_or_default() ? 1u : 0u;
        constants.SkinDetail = config.DlssNrSkinDetail.value_or_default();
        constants.SkinColour = config.DlssNrSkinColour.value_or_default();
        constants.EnvironmentDetail = config.DlssNrEnvironmentDetail.value_or_default();
        constants.EnvironmentColour = config.DlssNrEnvironmentColour.value_or_default();
        return constants;
    }
};
