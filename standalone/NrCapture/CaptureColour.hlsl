// Reuse the exact transfer functions and bindings, without changing the in-game shader.
#include "../../OptiScaler/shaders/dlssnr/precompile/dlssnr.hlsl"
#include "ModelResize.hlsli"

float4 FpsLabel(uint2 pixel, float4 base)
{
    // Small 5x7 glyphs at 2x scale, with no background.
    // Mode 13: DebugView 1 = single rate, 2 = input/output; CompareZoom and
    // TransferStrength carry the two rates independently of model settings.
    int2 p = int2(pixel) - int2(12, 12);
    if (gDebugView == 0 || any(p < 0)) return base;
    bool dual = gDebugView == 2;
    uint labelRow = (uint)p.y / 18;
    if (labelRow >= (dual ? 2u : 1u)) return base;
    p.y %= 18;
    if (p.y >= 14) return base;
    uint value = min(9999u, (uint)max(0.0, labelRow == 0 ? gCompareZoom : gTransferStrength));
    uint digits = value >= 1000 ? 4u : value >= 100 ? 3u : value >= 10 ? 2u : 1u;
    uint column = (uint)p.x / 12;
    if (p.x % 12 >= 10) return base;
    const uint rows[126] = {
        14,17,19,21,25,17,14, 4,12,4,4,4,4,14, 14,17,1,2,4,8,31,
        30,1,1,14,1,1,30, 2,6,10,18,31,2,2, 31,16,16,30,1,1,30,
        14,16,16,30,17,17,14, 31,1,2,4,8,8,8, 14,17,17,14,17,17,14,
        14,17,17,15,1,1,14, 31,16,16,30,16,16,16, 30,17,17,30,16,16,16,
        15,16,16,14,1,1,30, 31,4,4,4,4,4,31, 17,25,25,21,19,19,17,
        14,17,17,17,17,17,14, 17,17,17,17,17,17,14, 31,4,4,4,4,4,4}; // 0-9, F P S I N O U T
    uint prefix = dual ? (labelRow == 0 ? 2u : 3u) : 0u;
    uint glyph;
    if (column < prefix) glyph = (labelRow == 0 ? 13u : 15u) + column;
    else {
        if (prefix > 0 && column == prefix) return base;
        column -= prefix > 0 ? prefix + 1 : 0;
        if (column >= digits + 4 || column == digits) return base;
        glyph = 10 + column - digits - 1;
        if (column < digits) {
            const uint divisors[4] = {1, 10, 100, 1000};
            glyph = (value / divisors[digits - column - 1]) % 10;
        }
    }
    bool ink = (rows[glyph * 7 + (uint)p.y / 2] & (1u << (4 - (uint)p.x % 12 / 2))) != 0;
    return ink ? float4(lerp(base.rgb, WhitePoint().xxx, 0.5), base.a) : base;
}

float4 ComparisonLabel(uint2 pixel, float4 base)
{
    if (gApplyModel == 0 || gCompareMode == 0) return base;
    const uint letters[8] = {0, 1, 2, 3, 2, 4, 5, 6}; // ORIGINAL
    const uint rows[49] = {
        14,17,17,17,17,17,14, 30,17,17,30,20,18,17, 31,4,4,4,4,4,31,
        14,17,16,23,17,17,15, 17,25,25,21,19,19,17, 14,17,17,31,17,17,17, 16,16,16,16,16,16,31};
    float boundary = gCompareMode == 1 ? 0.5 : gCompareSplit;
    bool left = pixel.x < boundary * gWidth;
    bool original = left != (gCompareSwap != 0);
    float lo = left ? 0 : boundary * gWidth;
    float hi = left ? boundary * gWidth : gWidth;
    int scale = max(1, (int)round(gDebugScale * 2));
    int count = original ? 8 : 2;
    int2 size = int2((count * 6 + 4) * scale, 11 * scale);
    int2 position = int2((lo + hi - size.x) * 0.5, 12);
    int2 p = int2(pixel) - position;
    if (hi - lo < size.x || any(p < 0) || any(p >= size)) return base;
    int2 cell = p / scale - 2;
    bool ink = false;
    if (all(cell >= 0) && cell.y < 7 && cell.x < count * 6 && cell.x % 6 < 5) {
        uint glyph = original ? letters[cell.x / 6] : cell.x / 6 == 0 ? 4 : 1;
        ink = (rows[glyph * 7 + cell.y] & (1u << (4 - cell.x % 6))) != 0;
    }
    return float4(ink ? WhitePoint().xxx : 0.0.xxx, 1);
}

float4 HdrOutput(uint2 pixel, float4 sample)
{
    // Mode 13 reuses MaxRatio/ColourStrength for output peak/black in scRGB
    // units, Passthrough for bypass and MvScaleX for comparison zoom.
    if (gPassthrough != 0) return sample;
    float2 uv = (pixel + 0.5) / float2(gWidth, gHeight);
    if (gCompareMode != 0) {
        float split = gCompareMode == 1 ? 0.5 : gCompareSplit;
        if ((uv.x < split) != (gCompareSwap != 0) || abs(uv.x - split) < 1.0 / gWidth) return sample;
        // Keep the shared resolve's original half, divider and letterbox intact.
        if (gCompareMode == 1 && abs(uv.y - 0.5) * 2 > 0.5 * gMvScaleX) return sample;
    }
    float3 colour = sample.rgb;
    if (gColourStrength != 0) {
        float luminance = max(dot(colour, float3(0.2126, 0.7152, 0.0722)), 0);
        float weight = saturate(1 - luminance / WhitePoint());
        float offset = gColourStrength * weight * weight;
        // Lift by a neutral offset; darken by scaling, without clipping individual
        // colour channels. Neither adjustment touches reference white/highlights.
        if (offset >= 0) colour += offset;
        else colour *= max(0, luminance + offset) / max(luminance, 1e-6);
    }
    if (gMaxRatio > 0) {
        float brightest = max(colour.r, max(colour.g, colour.b));
        float knee = gMaxRatio * 0.75;
        if (brightest > knee) {
            float room = gMaxRatio - knee;
            float mapped = knee + room * (1 - room / (room + brightest - knee));
            colour *= mapped / brightest; // Preserve RGB ratios through the shoulder.
        }
    }
    return float4(colour, sample.a);
}

float4 BoundPrivateUpscale(uint2 pixel)
{
    uint w, h; gSource.GetDimensions(w, h);
    float2 uv = (pixel + 0.5) / float2(gWidth, gHeight);
    int2 base = int2(floor(uv * float2(w, h) - 0.5));
    float3 lo = 1.0, hi = 0.0;
    [unroll] for (int y = 0; y < 2; ++y) [unroll] for (int x = 0; x < 2; ++x) {
        float3 tap = SanitizeFinite3(gSource.Load(int3(clamp(base + int2(x,y), int2(0,0), int2(w-1,h-1)),0)).rgb, 0.5);
        lo = min(lo, tap); hi = max(hi, tap);
    }
    float3 reference = clamp(SanitizeFinite3(gSource.SampleLevel(gLinear, uv, 0).rgb, 0.5), lo, hi);
    float3 candidate = SanitizeFinite3(gModel.Load(int3(pixel,0)).rgb, reference);
    // Clip DLSS excursions outside the current colour or residual footprint,
    // along the deviation from its bilinear value.
    // This range guard leaves in-range output unchanged.
    if (any(candidate < lo) || any(candidate > hi)) {
        float3 delta = candidate - reference;
        float amount = 1;
        [unroll] for (int c = 0; c < 3; ++c) {
            if (delta[c] > 0) amount = min(amount, (hi[c] - reference[c]) / delta[c]);
            else if (delta[c] < 0) amount = min(amount, (lo[c] - reference[c]) / delta[c]);
        }
        candidate = clamp(reference + saturate(amount) * delta, lo, hi);
    }

    return float4(candidate, 1);
}

[numthreads(8, 8, 1)]
void CaptureColour(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) return;
    if (gMode == 20) {
        // Raw encoded stages have already reconstructed to the capture size.
        // Their replace resolve only keeps the model's nonnegative RGB and the
        // input alpha; final scene/skin/HDR composition belongs to the host.
        float2 uv = (id.xy + 0.5) / float2(gWidth, gHeight);
        float3 model = gModel.SampleLevel(gLinear, uv, 0).rgb;
        gTarget[id.xy] = float4(max(model, 0.0), gOriginal.Load(int3(id.xy, 0)).a);
        return;
    }
    if (gMode == 21) {
        float4 sample = gSource.Load(int3(id.xy, 0));
        float4 encoded = float4(LinearToSrgb(max(sample.rgb / WhitePoint(), 0.0)), sample.a);
        // Keep the exact same FP16 carrier that the ordinary SDR encode stores,
        // so final composition can reuse this initial conversion.
        gTarget[id.xy] = encoded; gKeep[id.xy] = encoded;
        return;
    }
    if (gMode == 19) {
        // Independent stages retain the legacy inter-pass bounded input.
        float4 sample = gSource.Load(int3(id.xy, 0));
        float4 bounded = float4(saturate(SanitizeFinite3(sample.rgb, 0.5)), sample.a);
        gTarget[id.xy] = bounded; gKeep[id.xy] = bounded;
        return;
    }
    // Use OptiScaler's shared filter; private mode 13 is already the label pass.
    if (gMode == 18) { gTarget[id.xy] = NrSuppressHalo(int2(id.xy)); return; }
    if (gMode == 14 || gMode == 15) { gTarget[id.xy] = BoundPrivateUpscale(id.xy); return; }
    if (gMode == 16) { gTarget[id.xy] = PrepareModelResize(id.xy); return; }
    if (gMode == 17) { gTarget[id.xy] = RestoreModelResize(id.xy); return; }
    float4 sample = gSource.Load(int3(id.xy, 0));
    if (gMode == 13) gTarget[id.xy] = FpsLabel(id.xy, ComparisonLabel(id.xy, HdrOutput(id.xy, sample)));
    else if (gMode == 11)
    {
        // WGC stores finished SDR in linear scRGB; NR's in-game SDR path receives sRGB.
        gTarget[id.xy] = float4(LinearToSrgb(max(sample.rgb / WhitePoint(), 0.0)), sample.a);
    }
    else
    {
        float4 original = gModel.Load(int3(id.xy, 0));
        float3 encodedOriginal = gOriginal.Load(int3(id.xy, 0)).rgb;
        // Preserve protected/unmodified pixels exactly, including FP16 round-trip error.
        float3 restored = all(sample.rgb == encodedOriginal) ? original.rgb : SrgbToLinear(sample.rgb) * WhitePoint();
        gTarget[id.xy] = float4(restored, original.a);
    }
}
