// MV-reprojected temporal accumulation of the pre-SR NR residual.
// Mode 0 blends (edited - original) into history; invalid reprojection fades in from zero.
// Mode 1 applies the signed residual after upscaling.
// Bindings match the main NR shader; appended history fields fit its existing constant-buffer allocation.

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
    uint  gReserved;
    float gResidualScale;
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
    float gReplaceDetailUnused, gModelWorkScaleUnused, gResidualConfidenceSensitivity;
    uint gExposureModeUnused;
    float gHistoryExposureRatio, gExposureTrimUnused, gExposureProtectionUnused;
    uint gExposureAnchorCountUnused, gExposureSourceWidthUnused, gExposureSourceHeightUnused, gExposurePaddingUnused;
    float4 gExposureAnchorsUnused[4];
    float gHaloUnused, gHaloBrightUnused, gHaloRadiusUnused;
    uint gHistoryRejection;
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

float InputLight(int2 pixel)
{
    float3 colour = max(SanitizeFinite3(gSource.Load(int3(pixel, 0)).rgb, 0), 0);
    float exposure = isfinite(gResidualScale) && gResidualScale > 1e-6 ? gResidualScale : 1.0;
    return log2(max(dot(colour, float3(0.2126, 0.7152, 0.0722)) / exposure, 1.0 / 1024.0));
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

        float4 old = valid ? gOriginal.SampleLevel(gLinear, prevUV, 0) : float4(0, 0, 0, 0);
        float3 history = old.rgb;
        history = SanitizeFinite3(history, float3(0.0, 0.0, 0.0));

        // Invalid reprojection: history is 0, so the pixel fades in from no edit at the normal blend
        // rate over the next frames. A cold start/cut also fades in, without sampling uninitialized history.
        float a = clamp(gResidualBlend, 0.0, 1.0);
        float inputLight = 1.0;
        if (gHistoryRejection != 0)
        {
            // Adapted from Lecram's look_compose: clip the edit to today's
            // neighbours and reject history whose underlying input changed.
            inputLight = InputLight(int2(id.xy));
            float loI = inputLight, hiI = inputLight;
            float3 lo = delta, hi = delta;
            [unroll] for (int y = -1; y <= 1; ++y) [unroll] for (int x = -1; x <= 1; ++x)
            {
                int2 q = clamp(int2(id.xy) + int2(x, y), int2(0, 0), int2(gWidth - 1, gHeight - 1));
                float3 edit = SanitizeFinite3(gModel.Load(int3(q, 0)).rgb - gSource.Load(int3(q, 0)).rgb, delta);
                lo = min(lo, edit); hi = max(hi, edit);
                float light = InputLight(q);
                loI = min(loI, light); hiI = max(hiI, light);
            }
            valid = valid && all(isfinite(old));
            // Match the history to this frame's pre-exposed colour units before clipping.
            valid = valid && isfinite(gHistoryExposureRatio) && gHistoryExposureRatio > 0;
            history *= valid ? gHistoryExposureRatio : 1.0;
            float changed = valid ? max(max(loI - old.a, old.a - hiI), 0.0) : 1.0;
            a = max(a, smoothstep(0.1, 0.3, changed));
            history = valid ? clamp(history, lo, hi) : delta;
        }
        if (gResidualConfidenceSensitivity > 0.0)
            a = lerp(a, 1.0, saturate(length(delta - history) / gResidualConfidenceSensitivity));

        // Alpha is unused by composition. Carry input light there without another image.
        gTarget[id.xy] = float4(lerp(history, delta, a), inputLight);
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

    gTarget[id.xy] = gSource.Load(int3(id.xy, 0));
}
