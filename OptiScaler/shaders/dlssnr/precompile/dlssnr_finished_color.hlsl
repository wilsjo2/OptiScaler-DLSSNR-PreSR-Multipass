// HDR10 <-> linear Rec.709 in scRGB units (1 = 80 nits).
// Separate from the existing NR shader so SDR and ordinary NR keep their compiled code.
cbuffer Params : register(b0)
{
    uint mode; float exposureScale; uint width; uint height;
    float sceneIsLinear; float unusedColour; uint unusedDebug; float maxRatio;
    uint unusedPassthrough;
    float unusedMvScaleX;
    float unusedMvScaleY;
    uint unusedGuideWidth;
    uint unusedGuideHeight;
    uint unusedCompareMode;
    float unusedCompareSplit;
    float unusedCompareZoom;
    uint unusedCompareSwap;
    uint unusedTransfer;
    float unusedDebugScale;
    uint unusedReversibleMode;
    uint unusedApplyModel;
    uint unusedUseGameExposure;
    float unusedExposurePreMul;
    uint unusedSkinProtection;
    uint unusedShowSkinMask;
    float unusedSkinDetail;
    float unusedSkinColour;
    float unusedEnvironmentDetail;
    float unusedEnvironmentColour;
    float unusedResidualBlend;
    uint unusedResidualHistoryValid;
    uint unusedResidualMotionBaseX;
    uint unusedResidualMotionBaseY;
    float shadowFloor;
};
Texture2D<float4> source : register(t0);
Texture2D<float4> reference : register(t1);
Texture2D<float4> original : register(t2);
RWTexture2D<float4> target : register(u0);

float3 DecodePQ(float3 code)
{
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    float3 p = pow(saturate(code), 1.0 / m2);
    return pow(max(p - c1, 0.0) / max(c2 - c3 * p, 1e-6), 1.0 / m1) * 125.0;
}
float3 EncodePQ(float3 light)
{
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    float3 p = pow(saturate(light / 125.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    float4 pixel = source.Load(int3(id.xy, 0));
    const float3x3 to709 = {
         1.6604910, -0.5876411, -0.0728499,
        -0.1245505,  1.1328999, -0.0083494,
        -0.0181508, -0.1005789,  1.1187297 };
    const float3x3 to2020 = {
        0.6274039, 0.3292830, 0.0433131,
        0.0690973, 0.9195404, 0.0113623,
        0.0163914, 0.0880133, 0.8955953 };
    if (mode == 5)
    {
        // Carry a bounded relative change through DLSS. An absolute signed residual
        // around 0.5 loses small dark-scene edits when stored in FP16; dividing that
        // quantization error by the dark SR reference produces large colour noise.
        float3 base = max(pixel.rgb, 0.0);
        float3 edited = max(reference.Load(int3(id.xy, 0)).rgb, 0.0);
        if (!all(isfinite(base)) || !all(isfinite(edited)))
        { target[id.xy] = float4(0.5, 0.5, 0.5, 1.0); return; }
        if (sceneIsLinear < 0.5)
        { base = pow(base, 2.2); edited = pow(edited, 2.2); }
        float floorValue = max(max(base.r, max(base.g, base.b)) * 0.02,
                               max(exposureScale, 1e-4) * 1e-4);
        float limit = clamp(maxRatio, 1.0, 8.0);
        float3 gain = clamp(1.0 + (edited - base) / max(base, floorValue), 1.0 / limit, limit);
        if (shadowFloor > 0.0)
        {
            const float lowerGain = clamp(shadowFloor, 1.0 / limit, 1.0);
            gain = clamp(1.0 + (edited - base) / max(base, floorValue), lowerGain, limit);
        }
        target[id.xy] = float4(0.5 + log2(gain) / 8.0, 1.0);
        return;
    }
    if (mode >= 2)
    {
        // t0 = finished picture, t2 = log-gain carrier upscaled by private DLSS.
        // Apply bounded relative changes in linear light. This approximates the game's
        // unknown tonemapper and colour grading; no scene-linear delta is added to display code.
        float3 carrier = original.Load(int3(id.xy, 0)).rgb;
        if (!all(isfinite(carrier)) || all(carrier == 0.5))
        { target[id.xy] = pixel; return; }
        float limit = clamp(maxRatio, 1.0, 8.0);
        float3 gain = exp2(clamp((carrier - 0.5) * 8.0, -log2(limit), log2(limit)));
        if (shadowFloor > 0.0)
        {
            const float lowerGain = clamp(shadowFloor, 1.0 / limit, 1.0);
            gain = exp2(clamp((carrier - 0.5) * 8.0, log2(lowerGain), log2(limit)));
        }
        float3 light = mode == 2 ? pow(max(pixel.rgb, 0.0), 2.2) :
                       mode == 4 ? mul(to709, DecodePQ(pixel.rgb)) : pixel.rgb;
        light *= gain;
        float3 result = mode == 2 ? pow(saturate(light), 1.0 / 2.2) :
                        mode == 4 ? EncodePQ(mul(to2020, light)) : light;
        target[id.xy] = float4(all(isfinite(result)) ? result : pixel.rgb, pixel.a);
        return;
    }
    if (mode == 0)
    {
        // Negative components carry wide-gamut colours; retain them in FP16.
        target[id.xy] = float4(mul(to709, DecodePQ(pixel.rgb)), pixel.a);
    }
    else
    {
        float4 base = original.Load(int3(id.xy, 0));
        float3 result = EncodePQ(mul(to2020, pixel.rgb));
        target[id.xy] = float4(all(isfinite(result)) ? result : base.rgb, base.a);
    }
}
