#pragma once
#include <dlssnr/DlssNr_Proxy.h>
#include <dlssnr/PassProfiles.h>
#include <dlssnr/DlssNrFeature_Dx12.h>
#include <shaders/output_scaling/OS_Dx12.h>
#include "DlssNr_Spatial.h"

namespace DlssNr::Detail
{
struct ModelStateDx12
{
    unsigned long long successfulDispatches = 0;
    // Each model pass owns its NGX feature, parameters and temporal history.
    DlssNr::Proxy::Context models[DlssNr::MaxPassCount];
    bool passCreateFailed[DlssNr::MaxPassCount] = {};

    // The model cannot read and write one resource, so the frame is staged through these.
    ID3D12Resource* colorCopy = nullptr;
    ID3D12Resource* output = nullptr;

    // The second half of the model-output ping-pong. The base proxy stays immutable: pass 0 writes
    // output (A), pass 1 writes this (B), and pass 2 writes A again. Only the final answer is composed.
    ID3D12Resource* passScratch = nullptr;
    bool passScratchFailed = false;
    ID3D12Resource* passClamp = nullptr; // bounded input for the next model pass

    // The frame as the upscaler wrote it. The resolve adds the model's edit to this rather than
    // reconstructing it by inverting the tone curve, which is what turned every light in the frame into
    // a string of coloured cells.
    ID3D12Resource* hdrCopy = nullptr;
    ID3D12Resource* exposureMeter = nullptr;
    ID3D12Resource* exposure = nullptr;
    bool exposureReadable = false;
    unsigned exposureSource = 0;
    float exposurePreExposure = 1;

    // Compact origin-zero pre-SR image, only needed when Color has allocation padding. All codec,
    // hold and capture paths then see the real raster. UAV at rest, retired with the scratch set.
    ID3D12Resource* activeColor = nullptr;

    // The frame shrunk for the model, when it is working below full resolution.
    ID3D12Resource* colorSmall = nullptr;

    // Peripheral compression keeps a packed model pair and packed guides separate from the
    // ordinary uniform-scale pair consumed by composition and optional DLSS enlargement.
    ID3D12Resource* spatialColor = nullptr;
    ID3D12Resource* spatialDepth = nullptr;
    ID3D12Resource* spatialMotion = nullptr;
    ID3D12Resource* spatialProxy = nullptr;
    ID3D12Resource* spatialAnswer = nullptr;
    ID3D12Resource* spatialProxyNative = nullptr;
    ID3D12Resource* spatialAnswerNative = nullptr;
    Spatial::Layout spatialLayout {};
    bool spatialSignatureValid = false;
    DXGI_FORMAT spatialColorFormat = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT spatialDepthFormat = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT spatialMotionFormat = DXGI_FORMAT_UNKNOWN;
    unsigned spatialDepthW = 0, spatialDepthH = 0, spatialMotionW = 0, spatialMotionH = 0;
    bool spatialFallback = false;
    const char* spatialFallbackReason = "";
    bool spatialActive = false;

    // Supersampling filters are allocated lazily; dispatch dimensions come from the resources.
    OS_Dx12* superUp = nullptr;

    // The down-leg returns the model answer to native size before composition.
    ID3D12Resource* outputNative = nullptr;
    OS_Dx12* superDown = nullptr;
    OS_Dx12* spatialProxyDown = nullptr;
    Scaler nrScaler = Scaler::Count;

    // Frame hold (design/frame-hold.md): a persistent copy of the output taken on hold-on and restored
    // over the live output before the encode reads it while held, so a setting change re-renders the
    // same frame. heldWhitePoint preserves the encode scale for the comparison.
    ID3D12Resource* heldColor = nullptr;
    bool heldActive = false;
    unsigned int heldWidth = 0;
    unsigned int heldHeight = 0;
    DXGI_FORMAT heldFormat = DXGI_FORMAT_UNKNOWN;
    float heldWhitePoint = 1.0f;

    unsigned int workWidth = 0;
    unsigned int workHeight = 0;

    // Cloned unconditionally when running at present, and only for typeless formats otherwise.
    ID3D12Resource* depthClone = nullptr;
    ID3D12Resource* motionClone = nullptr;

    unsigned int width = 0;
    unsigned int height = 0;
    bool beforeUpscale = false;
    bool rayReconstruction = false;
    bool reset = true;

    // The preset, style and strengths each live feature was created with.
    ModelSettings builtSettings[DlssNr::MaxPassCount] {};

    // Latch failures until an explicit retry rather than recording failing GPU work every frame.
    bool failed = false;
    const char* reason = "";
};
} // namespace DlssNr::Detail
