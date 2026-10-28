// Denoise-first, second step 2: put NR's edit back onto the raw jittered render.
//
// NR ran on a clean, un-jittered 1:1 image. Its edit (NR minus clean, or NR over clean) therefore
// lives on the stable pixel grid, while the raw render sampled the scene a sub-pixel jitter away.
// The edit is resampled at the jittered position so it lines up with the raw samples, then added to
// (or multiplied onto) the raw colour. The game's own upscale then receives real jittered samples
// carrying a stable edit, which is the only arrangement that gives it genuine sub-pixel information.
//
// Bindings and the constant block match dlssnr_residual.hlsl so DispatchCompute's descriptor table
// is reused unchanged. Several fields are repurposed, see the host side (DenoiseFirst context):
//   gMvScaleX / gMvScaleY  signed jitter shift in pixels (0 when the shift is off)
//   gDebugView             kernel: 0 bilinear, 1 Catmull-Rom, 2 Lanczos 2
//   gCompareMode           edit mode: 0 difference (added), 1 ratio (multiplied)
//   gCompareSwap           1 = clamp the resampled edit to its 2x2 neighbourhood (anti-ringing)
//   gPassthrough           1 = ratio mode does not multiply fireflies (samples far above the clean value)
//   gTransferStrength      edit strength, 1 = as the model produced it
//   gMaxRatio              ceiling for the ratio mode
cbuffer Params : register(b0)
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
    float gResidualBlend;
    uint  gResidualHistoryValid;
    uint  gResidualMotionBaseX;
    uint  gResidualMotionBaseY;
    float gReplaceDetailUnused, gModelWorkScaleUnused, gResidualConfidenceSensitivity;
};

Texture2D<float4>   gSource   : register(t0); // the raw, jittered, noisy render (active region, origin zero)
Texture2D<float4>   gModel    : register(t1); // NR's output on the clean 1:1 image
Texture2D<float4>   gOriginal : register(t2); // the clean 1:1 image NR was shown
Texture2D<float4>   gMotion   : register(t3); // unused; bound for descriptor-table parity
Texture2D<float4>   gExposure : register(t4); // unused; bound for descriptor-table parity
RWTexture2D<float4> gTarget   : register(u0); // raw render carrying the edit, handed to the game's upscale
RWTexture2D<float4> gKeep     : register(u1); // unused; bound for descriptor-table parity
SamplerState        gLinear   : register(s0); // unused; the kernels gather their own taps

static const float kRatioFloor = 1e-3; // scene-linear; keeps the ratio finite on black pixels

float  Finite(float v, float fallback)   { return isfinite(v) ? v : fallback; }
float3 Finite3(float3 v, float3 fallback)
{
    return float3(Finite(v.x, fallback.x), Finite(v.y, fallback.y), Finite(v.z, fallback.z));
}

// The edit at one integer pixel of the clean grid.
float3 EditAt(int2 p)
{
    p = clamp(p, int2(0, 0), int2(gWidth, gHeight) - 1);
    const float3 model = Finite3(gModel.Load(int3(p, 0)).rgb, float3(0.0, 0.0, 0.0));
    const float3 clean = Finite3(gOriginal.Load(int3(p, 0)).rgb, float3(0.0, 0.0, 0.0));
    if (gCompareMode == 1)
        return clamp((max(model, 0.0) + kRatioFloor) / (max(clean, 0.0) + kRatioFloor), 0.0, max(gMaxRatio, 1.0));
    return model - clean;
}

float CatmullRom(float x)
{
    x = abs(x);
    if (x < 1.0)
        return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    if (x < 2.0)
        return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    return 0.0;
}

float Sinc(float x)
{
    const float px = 3.14159265 * x;
    return abs(x) < 1e-4 ? 1.0 : sin(px) / px;
}

float Lanczos2(float x)
{
    x = abs(x);
    return x < 2.0 ? Sinc(x) * Sinc(x * 0.5) : 0.0;
}

float Weight(float x)
{
    if (gDebugView == 1)
        return CatmullRom(x);
    if (gDebugView == 2)
        return Lanczos2(x);
    return max(1.0 - abs(x), 0.0); // bilinear
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight)
        return;

    const float4 raw = gSource.Load(int3(id.xy, 0));

    // Where this raw pixel's sample sits on the clean grid, in clean pixel units.
    const float2 shift = float2(Finite(gMvScaleX, 0.0), Finite(gMvScaleY, 0.0));
    const float2 sample = float2(id.xy) + 0.5 + shift;
    const float2 base = floor(sample - 0.5);
    const float2 frac = sample - 0.5 - base;

    // 4x4 gather with the separable kernel; bilinear simply has zero weight on the outer ring.
    float3 edit = float3(0.0, 0.0, 0.0);
    float  total = 0.0;
    float3 lo = float3(1e30, 1e30, 1e30), hi = float3(-1e30, -1e30, -1e30);
    [unroll]
    for (int ky = -1; ky <= 2; ++ky)
    {
        const float wy = Weight(frac.y - ky);
        [unroll]
        for (int kx = -1; kx <= 2; ++kx)
        {
            const float w = wy * Weight(frac.x - kx);
            const float3 tap = EditAt(int2(base) + int2(kx, ky));
            edit += tap * w;
            total += w;
            if (kx >= 0 && kx <= 1 && ky >= 0 && ky <= 1)
            {
                lo = min(lo, tap);
                hi = max(hi, tap);
            }
        }
    }
    edit = total > 1e-6 ? edit / total : EditAt(int2(id.xy));
    if (gCompareSwap != 0)
        edit = clamp(edit, lo, hi);
    edit = Finite3(edit, gCompareMode == 1 ? float3(1.0, 1.0, 1.0) : float3(0.0, 0.0, 0.0));

    float3 result;
    if (gCompareMode == 1)
    {
        // A path-traced sample far brighter than the clean value is a firefly, not a lit pixel. A
        // multiplicative gain sized for the clean value would boost it (measured 1.3x to 1.6x on a
        // dark floor, and the game's RR then kept some as sparkles). So the gain is applied in
        // proportion to how far the sample is at or below the clean value: ordinary samples get the
        // full ratio, a firefly gets the same absolute change the clean pixel would have received.
        const float cleanLum = dot(Finite3(gOriginal.Load(int3(id.xy, 0)).rgb, float3(0.0, 0.0, 0.0)),
                                   float3(0.2126, 0.7152, 0.0722));
        const float rawLum = dot(max(raw.rgb, 0.0), float3(0.2126, 0.7152, 0.0722));
        // gPassthrough carries the firefly-guard switch: 0 = plain ratio on every sample.
        const float ordinary = gPassthrough != 0 ? saturate(max(cleanLum, 0.0) / max(rawLum, 1e-6)) : 1.0;
        const float3 gain = lerp(float3(1.0, 1.0, 1.0), edit, gTransferStrength);
        result = raw.rgb * (1.0 + (gain - 1.0) * ordinary);
    }
    else
        result = raw.rgb + edit * gTransferStrength;

    // Radiance never goes negative into the upscaler, whatever the edit asked for.
    gTarget[id.xy] = float4(max(Finite3(result, raw.rgb), 0.0), raw.a);
}
