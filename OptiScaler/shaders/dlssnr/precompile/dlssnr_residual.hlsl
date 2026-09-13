// ResidualAcrossRR v2 -- the MV-reprojected temporal accumulator for the pre-SR NR residual.
//
// Deliberately a SEPARATE shader from dlssnr.hlsl. Regenerating dlssnr.hlsl's blob with a current
// dxc produces materially different DXIL from the committed one (older compiler), and that shader
// carries every NR path -- post-SR, pre-SR, RR, DeferredDLSS. These two experimental
// modes get their own tiny blob and a second compute PSO instead, so the battle-tested one is
// never touched. The cbuffer and bindings mirror dlssnr.hlsl exactly so DlssNr_Dx12's existing
// root signature and descriptor table are reused as-is; only gResidualBlend is appended, and it
// fits inside DlssNrConstants' existing 256-byte alignment with no size change.
//
//   gMode == 0  Accumulate: (edited - original) blended into the MV-reprojected history layer.
//               history_t = lerp( reproject(history_{t-1}), edited - original, blend )
//               The per-frame ray-trace noise term of (edited - original) is temporally
//               uncorrelated and averages to zero; the enhancement term follows geometry and
//               persists. Invalid reprojection (off-screen / bad MV) -> the
//               history is treated as zero at that pixel and rebuilds over the next frames.
//
//               `blend` is not the single gResidualBlend scalar: it is gResidualBlend widened
//               toward 1 by a per-pixel confidence gate, in proportion to how much this frame's
//               freshly-computed edit disagrees with what the reprojected history predicted.
//               Full agreement keeps gResidualBlend's stable floor; a disocclusion the binary
//               valid/invalid test missed, a lighting change, or a genuinely invalid reprojection
//               (history forced to 0 above) all show up as disagreement and blend in faster
//               instead of fading in over several frames at the same fixed rate regardless of
//               cause. This is an adaptation, not a port, of NRD's history-confidence concept
//               (research/nvidia/nrd/README.md, "HISTORY CONFIDENCE": "An application should not
//               rely solely on the anti-lag provided by REBLUR/RELAX") to what this pass actually
//               has: no re-traced radiance to diff a stored value against, only its own
//               reprojected accumulation and this frame's edit, so "disagreement" here is their
//               local difference rather than a separate gradient pass over ground-truth radiance.
//   gMode == 1  Apply: base + delta * gTransferStrength, clamped non-negative. Run after RR+SR
//               with the upscaled history layer as the delta.
//   gMode == 2  ReprojectOnly: carry gOriginal (the last real model answer) forward through this
//               frame's motion vectors, with no new evaluation this frame -- for NR evaluation-
//               cadence decoupling (DlssNr_Dx12_Run.cpp): a skipped frame reprojects last frame's
//               raw model output instead of paying for a fresh NGX evaluate. Same reprojection/
//               validity test as gMode==0's history term; invalid (off-screen / bad MV / caller-
//               declared invalid) drops to zero rather than showing a stale answer from the wrong
//               place on screen.

#ifdef VK_MODE
[[vk::binding(0, 0)]]
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    uint  gMode;
    float gWhitePoint;
    uint  gWidth;
    uint  gHeight;
    float gTransferStrength;
    float gColourStrength;
    uint  gDebugView;
    float gMaxRatio;
    uint  gPassthrough;
    float gMvScaleX;
    float gMvScaleY;
    uint  gGuideWidth;
    uint  gGuideHeight;
    uint  gCompareMode;
    float gCompareSplit;
    float gCompareZoom;
    uint  gCompareSwap;
    uint  gTransfer;
    float gDebugScale;
    uint  gReversibleMode;
    uint  gApplyModel;
    uint  gUseGameExposure;
    float gExposurePreMul;
    uint  gSkinProtection;
    uint  gShowSkinMask;
    float gSkinDetail;
    float gSkinColour;
    float gEnvironmentDetail;
    float gEnvironmentColour;
    float gResidualBlend;   // v2 only: history blend rate, 0..1. 1 == no accumulation (== v1).
    uint gResidualHistoryValid;
    uint gResidualMotionBaseX;
    uint gResidualMotionBaseY;
    float gResidualConfidenceSensitivity;  // v2 confidence gate; see file header comment.
};

// Same registers and the same SPIR-V binding numbers as dlssnr.hlsl, including the slots these
// modes do not read (gExposure t4, gKeep u1) -- DispatchResidualPass binds a stand-in into them
// exactly as DispatchPass does, and a future Vulkan host path needs the numbering to line up.
#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float4>   gSource   : register(t0);  // accumulate: the untouched pre-SR frame. apply: the RR+SR output.
#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float4>   gModel    : register(t1);  // accumulate: the NR-edited frame. apply: the upscaled delta layer.
#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
Texture2D<float4>   gOriginal : register(t2);  // accumulate: the previous history layer.
#ifdef VK_MODE
[[vk::binding(4, 0)]]
#endif
Texture2D<float4>   gMotion   : register(t3);  // raw game motion; active size, offsets and scale come from the host.
#ifndef VK_MODE
Texture2D<float4>   gExposure : register(t4);  // unused here; bound for descriptor-table parity.
#endif
#ifdef VK_MODE
[[vk::binding(5, 0)]]
#endif
RWTexture2D<float4> gTarget   : register(u0);  // accumulate: the new history layer. apply: the composed frame.
#ifdef VK_MODE
[[vk::binding(6, 0)]]
#endif
RWTexture2D<float4> gKeep     : register(u1);  // unused here; bound for descriptor-table parity.
#ifdef VK_MODE
[[vk::binding(7, 0)]]
#endif
SamplerState        gLinear   : register(s0);  // history is sampled at the reprojected coordinate.

float  SanitizeFinite(float v, float fallback)   { return isfinite(v) ? v : fallback; }
float3 SanitizeFinite3(float3 v, float3 fallback)
{
    return float3(SanitizeFinite(v.x, fallback.x), SanitizeFinite(v.y, fallback.y),
                  SanitizeFinite(v.z, fallback.z));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight)
        return;

    if (gMode == 0)
    {
        float3 delta = SanitizeFinite3(gModel.Load(int3(id.xy, 0)).rgb -
                                       gSource.Load(int3(id.xy, 0)).rgb, float3(0.0, 0.0, 0.0));

        float2 uv = (float2(id.xy) + 0.5) / float2(gWidth, gHeight);
        uint2 guideSize = uint2(gGuideWidth, gGuideHeight);
        uint2 guidePos = min(uint2(uv * guideSize), guideSize - 1) +
                         uint2(gResidualMotionBaseX, gResidualMotionBaseY);
        float2 motion = gMotion.Load(int3(guidePos, 0)).xy * float2(gMvScaleX, gMvScaleY);
        float2 prevUV = uv + motion;

        bool valid = gResidualHistoryValid != 0 && all(isfinite(motion)) && all(abs(motion) < 2.0) &&
                     all(prevUV >= 0.0) && all(prevUV <= 1.0);

        float3 history = valid ? gOriginal.SampleLevel(gLinear, prevUV, 0).rgb : float3(0.0, 0.0, 0.0);
        history = SanitizeFinite3(history, float3(0.0, 0.0, 0.0));

        // Confidence gate: see the file header comment. Disagreement in [0,1], 0 = history still
        // predicts this frame's edit, 1 = fully stale (including the invalid-reprojection case,
        // where history is 0 and any non-trivial delta disagrees maximally by construction).
        float sensitivity = max(gResidualConfidenceSensitivity, 1e-4);
        float disagreement = saturate(length(delta - history) / sensitivity);
        float a = lerp(clamp(gResidualBlend, 0.0, 1.0), 1.0, disagreement);

        gTarget[id.xy] = float4(lerp(history, delta, a), 1.0);
        return;
    }

    if (gMode == 1)
    {
        float4 base  = gSource.Load(int3(id.xy, 0));
        float2 uv = (float2(id.xy) + 0.5) / float2(gWidth, gHeight);
        float3 delta = SanitizeFinite3(gModel.SampleLevel(gLinear, uv, 0).rgb, float3(0.0, 0.0, 0.0));

        gTarget[id.xy] = float4(max(base.rgb + delta * gTransferStrength, 0.0), base.a);
        return;
    }

    if (gMode == 2)
    {
        float2 uv = (float2(id.xy) + 0.5) / float2(gWidth, gHeight);
        uint2 guideSize = uint2(gGuideWidth, gGuideHeight);
        uint2 guidePos = min(uint2(uv * guideSize), guideSize - 1) +
                         uint2(gResidualMotionBaseX, gResidualMotionBaseY);
        float2 motion = gMotion.Load(int3(guidePos, 0)).xy * float2(gMvScaleX, gMvScaleY);
        float2 prevUV = uv + motion;

        bool valid = gResidualHistoryValid != 0 && all(isfinite(motion)) && all(abs(motion) < 2.0) &&
                     all(prevUV >= 0.0) && all(prevUV <= 1.0);

        float3 reprojected = valid ? gOriginal.SampleLevel(gLinear, prevUV, 0).rgb : float3(0.0, 0.0, 0.0);
        reprojected = SanitizeFinite3(reprojected, float3(0.0, 0.0, 0.0));

        gTarget[id.xy] = float4(reprojected, 1.0);
        return;
    }

    gTarget[id.xy] = gSource.Load(int3(id.xy, 0));
}
