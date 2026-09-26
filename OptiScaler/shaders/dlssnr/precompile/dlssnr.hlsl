
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
    float gMvScaleX;     // motion vector units -> pixels of this dispatch
    float gMvScaleY;
    uint  gGuideWidth;   // the motion texture's valid region
    uint  gGuideHeight;
    uint  gCompareMode;  // 0 off, 1 side by side, 2 wipe
    float gCompareSplit; // where the wipe cuts, 0..1
    float gCompareZoom;  // side by side: 1 fits the frame, 2 fills the half
    uint  gCompareSwap;  // put the edited frame on the other side
    uint  gTransfer;     // 0 classic, 1 matched residual -- how a below-size model comes back
    float gDebugScale;   // what the debug views are scaled by, held still while the meter moves
    uint  gReversibleMode; // 0 knee, 1 Neutwo+composed, 2 Neutwo+replace, 3 hybrid+composed, 4 hybrid+replace
    uint  gApplyModel;     // 0 output the clean frame (pass still runs), 1 apply the model's edit
    float gMaxDarkening;    // percent: 100 = uncapped, 0 = no darkening
    float gResidualScale;
    uint  gSkinProtection;
    uint  gShowSkinMask;
    float gSkinDetail;
    float gSkinColour;
    float gEnvironmentDetail;
    float gEnvironmentColour;
    float gResidualBlendUnused;
    uint gResidualHistoryValidUnused, gResidualMotionBaseXUnused, gResidualMotionBaseYUnused;
    float gReplaceDetailStrength, gModelWorkScale, gResidualConfidenceUnused;
    uint gExposureMode;
    float gPreExposure, gExposureTrim, gExposureProtection;
    uint gExposureAnchorCount, gExposureSourceWidth, gExposureSourceHeight, gExposurePadding;
    float4 gExposureAnchors[4];
};

// Hue-preserving gamut compression toward the D65 neutral axis.
// Adapted from clshortfuse/RenoDX (https://github.com/clshortfuse/renodx).
// See Licenses/RenoDX_ATTRIBUTION.txt.

float SanitizeFinite(float v, float fallback) { return isfinite(v) ? v : fallback; }

// Approximate skin-colour selection, not a face/skin segmentation network. Warm
// materials may be selected and coloured lighting can hide skin. The preview is
// deliberately exposed so users can check this before relying on protection.
float SkinColourWeight(float3 rgb)
{
    rgb = saturate(rgb);
    float y = dot(rgb, float3(0.299, 0.587, 0.114));
    float cb = (rgb.b - y) * 0.564 + 0.5;
    float cr = (rgb.r - y) * 0.713 + 0.5;
    float2 distance = (float2(cb, cr) - float2(0.405, 0.600)) / float2(0.090, 0.110);
    float chroma = max(rgb.r, max(rgb.g, rgb.b)) - min(rgb.r, min(rgb.g, rgb.b));
    return (1.0 - smoothstep(0.55, 1.35, length(distance))) * smoothstep(0.02, 0.10, chroma);
}

float3 SanitizeFinite3(float3 v, float3 fallback)
{
    return float3(SanitizeFinite(v.x, fallback.x), SanitizeFinite(v.y, fallback.y),
                  SanitizeFinite(v.z, fallback.z));
}

float SafeDivide(float numerator, float denominator, float fallback)
{
    return abs(denominator) > 1e-8 ? numerator / denominator : fallback;
}

// Hunt-Pointer-Estevez LMS over linear BT.709, carrying the fixed D65 adaptation state the
// compression is defined against. The signal itself never leaves BT.709.
float3 LMSToBT709(float3 color)
{
    const float3x3 m = { 5.62059812, -4.57145756, 0.15577924,
                         -1.15555585, 2.25800438, -0.15415806,
                         0.03059913, -0.19018011, 1.06820532 };
    return mul(m, color);
}

float3 BT709ToLMS(float3 color)
{
    const float3x3 m = { 0.30569589, 0.62271286, 0.04528636,
                         0.15776262, 0.76968599, 0.08807030,
                         0.01933082, 0.11919478, 0.95053215 };
    return mul(m, color);
}

// The neutral colour of the same luminance as what is being compressed -- the point everything is
// pulled toward, so that pulling changes saturation and not hue.
float3 D65NeutralBT709(float3 adaptiveStateLms, float luminance)
{
    float3 d65 = LMSToBT709(max(adaptiveStateLms, 1e-8));
    float d65Y = max(dot(d65, float3(0.2126, 0.7152, 0.0722)), 1e-8);
    return d65 * (luminance / d65Y);
}

// The largest scale toward the neutral axis that leaves no channel negative. One for a colour that
// was already representable, which is why this is safe to run on every pixel.
float GamutCompressionScale(float3 color, float3 adaptiveStateLms)
{
    color = SanitizeFinite3(color, float3(0.0, 0.0, 0.0));

    const float y = dot(color, float3(0.2126, 0.7152, 0.0722));

    if (!(y > 1e-8))
        return 1.0;

    const float3 neutral = D65NeutralBT709(adaptiveStateLms, y);
    float scale = 1.0;

    if (color.r < 0.0 && neutral.r > color.r)
        scale = min(scale, SafeDivide(neutral.r, neutral.r - color.r, 1.0));

    if (color.g < 0.0 && neutral.g > color.g)
        scale = min(scale, SafeDivide(neutral.g, neutral.g - color.g, 1.0));

    if (color.b < 0.0 && neutral.b > color.b)
        scale = min(scale, SafeDivide(neutral.b, neutral.b - color.b, 1.0));

    return saturate(SanitizeFinite(scale, 1.0));
}

float3 ClampAp1(float3 color)
{
    const float3 adaptiveStateLms = BT709ToLMS(float3(0.18, 0.18, 0.18));
    const float scale = GamutCompressionScale(color, adaptiveStateLms);

    // Nothing was out of gamut. Leave the colour exactly as it arrived.
    if (scale >= 1.0)
        return color;

    const float y = dot(color, float3(0.2126, 0.7152, 0.0722));
    const float3 neutral = D65NeutralBT709(adaptiveStateLms, y);

    return SanitizeFinite3(neutral + (color - neutral) * scale, max(neutral, 0.0));
}

// ---------------------------------------------------------------------------------------------
// The composition below (UpgradeToneMap's two-branch ratio, the OkLab hue correction, and the blend
// between a luminance-only result and the model's own colour) is taken from RenoDX's DLSS 5 addon by
// clshortfuse -- https://github.com/clshortfuse/renodx. It is their design, not ours; see
// Licenses/RenoDX_LICENSE.txt. The OkLab matrices are Bjorn Ottosson's published constants and the
// AP1, sRGB and PQ transforms are standard colour science.
// ---------------------------------------------------------------------------------------------

// Use OkLab to retain the model's hue while adjusting chroma magnitude.
float3 CbrtSigned(float3 v) { return sign(v) * pow(abs(v), 1.0 / 3.0); }

float3 ToOkLab(float3 color)
{
    const float3x3 rgb_to_lms = { 0.4122214708, 0.5363325363, 0.0514459929,
                                  0.2119034982, 0.6806995451, 0.1073969566,
                                  0.0883024619, 0.2817188376, 0.6299787005 };
    const float3x3 lms_to_lab = { 0.2104542553, 0.7936177850, -0.0040720468,
                                  1.9779984951, -2.4285922050, 0.4505937099,
                                  0.0259040371, 0.7827717662, -0.8086757660 };
    return mul(lms_to_lab, CbrtSigned(mul(rgb_to_lms, color)));
}

float3 FromOkLab(float3 lab)
{
    const float3x3 lab_to_lms = { 1.0, 0.3963377774, 0.2158037573,
                                  1.0, -0.1055613458, -0.0638541728,
                                  1.0, -0.0894841775, -1.2914855480 };
    const float3x3 lms_to_rgb = { 4.0767416621, -3.3077115913, 0.2309699292,
                                  -1.2684380046, 2.6097574011, -0.3413193965,
                                  -0.0041960863, -0.7034186147, 1.7076147010 };
    float3 lms = mul(lab_to_lms, lab);
    return mul(lms_to_rgb, lms * lms * lms);
}

// Take hue direction from correct and chroma magnitude from incorrect.
float3 HueOkLab(float3 incorrect, float3 correct)
{
    float3 incorrectLab = ToOkLab(incorrect);
    const float3 correctLab = ToOkLab(correct);
    const float incorrectChroma = length(incorrectLab.yz);
    const float correctChroma = length(correctLab.yz);

    // Normalize hue direction before scaling; near-grey chroma must not amplify numerical noise.
    const float2 hueDirection = correctChroma > 1e-5 ? correctLab.yz / correctChroma : float2(0.0, 0.0);

    incorrectLab.yz = hueDirection * incorrectChroma;

    return ClampAp1(FromOkLab(incorrectLab));
}

// Bindings are stated for SPIR-V rather than inferred. D3D keeps b, t, u and s in separate register
// files, so b0 and t0 do not collide; Vulkan has one number line per descriptor set, and dxc's default
// mapping would put both at binding 0. The numbers below are the order the pass binds them in, and
// DlssNr_Vk's descriptor set layout has to agree with them entry for entry.
#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float4>   gSource   : register(t0);  // encode: the frame. resolve: the proxy.
#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float4>   gModel    : register(t1);  // resolve: what the model returned.
#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
Texture2D<float4>   gOriginal : register(t2);  // resolve: the untouched frame.
#ifdef VK_MODE
[[vk::binding(4, 0)]]
#endif
Texture2D<float4>   gMotion   : register(t3);  // resolve, accumulating: the game's motion vectors.

#ifdef VK_MODE
[[vk::binding(5, 0)]]
#endif
RWTexture2D<float4> gTarget   : register(u0);  // encode: the proxy. resolve: the frame.
#ifdef VK_MODE
[[vk::binding(6, 0)]]
#endif
RWTexture2D<float4> gKeep     : register(u1);  // encode: the untouched copy. unused by the resolve.
#ifdef VK_MODE
[[vk::binding(7, 0)]]
#endif
SamplerState        gLinear   : register(s0);  // so the edit can be read at a different size

float2 ExposureAnchor(uint i)
{
    return (i & 1u) ? gExposureAnchors[min(i / 2u, 3u)].zw : gExposureAnchors[min(i / 2u, 3u)].xy;
}
float WhitePoint()
{
    float base = max(gWhitePoint, 1e-4);
    if (gExposureMode == 1 || gExposureMode == 3)
    {
        float measured = gMotion.Load(int3(0, 0, 0)).r;
        if (isfinite(measured) && measured > 1e-8)
        {
        base = gExposureMode == 1 ? gPreExposure / measured : measured;
        float trim = gExposureTrim;
        uint count = min(gExposureAnchorCount, 8u);
        if (count > 0)
        {
            trim = ExposureAnchor(0).y;
            [unroll] for (uint i = 1; i < 8; ++i)
            {
                float2 previous = ExposureAnchor(i - 1);
                float2 next = ExposureAnchor(i);
                if (i < count && base > previous.x)
                {
                    float t = saturate(log2(max(base, 1e-8) / previous.x) / log2(next.x / previous.x));
                    trim = exp2(lerp(log2(previous.y), log2(next.y), t));
                }
            }
        }
        base *= trim;
        }
    }
    return max(SanitizeFinite(base, gWhitePoint), 1e-4);
}

static const float3 kLuma = float3(0.2126, 0.7152, 0.0722);

// sRGB rather than a plain 2.2 power: it is what an SDR game buffer actually carries, and the model was
// trained on those.
float3 LinearToSrgb(float3 v)
{
    v = saturate(v);
    return lerp(v * 12.92, 1.055 * pow(max(v, 1e-8), 1.0 / 2.4) - 0.055, step(0.0031308, v));
}

float3 SrgbToLinear(float3 v)
{
    v = saturate(v);
    return lerp(v / 12.92, pow((v + 0.055) / 1.055, 2.4), step(0.04045, v));
}

// The edit at an arbitrary position, exactly as the resolve computes its own.
float3 EditAt(float2 uvq)
{
    float3 p = gSource.SampleLevel(gLinear, uvq, 0).rgb;
    float3 m = gModel.SampleLevel(gLinear, uvq, 0).rgb;

    if (gPassthrough == 0)
    {
        p = SrgbToLinear(p);
        m = SrgbToLinear(m);
    }

    return m - p;
}


// Soft-knee proxy shared by encoding and matched-residual reconstruction.
float3 SoftKnee(float3 display)
{
    if (gPassthrough != 0)
        return display;

    float displayLuma = dot(display, kLuma);

    if (displayLuma > 0.75)
    {
        float rolled = 0.75 + 0.25 * (1.0 - exp(-(displayLuma - 0.75) / 0.25));
        display *= rolled / displayLuma;
    }

    // Per-channel headroom, with the hue kept.
    //
    // The roll-off above is on luminance, and luminance is a weighted sum in which blue counts for
    // seven percent. A saturated blue can therefore sit at B = 2 with a luminance of 0.14, pass the
    // knee untouched, and be clipped per channel by the saturate in LinearToSrgb -- and clipping one
    // channel of a triple is a hue rotation, so blue arrives as cyan. That was the green cast over
    // every blue thing in GTA V at colour strength 1: the sky, the denim, the minimap. The model was
    // shown a cyan proxy, answered in cyan, and at colour strength 1 its hue is the frame's hue.
    //
    // One scalar on the whole triple cannot move hue, so the peak channel is brought to 1 that way.
    // Only pixels that were already being clipped are touched, so everything else is bit-identical
    // to before, and the resolve's reconstruction of this proxy stays exact because it goes through
    // this same function.
    float peak = max(display.r, max(display.g, display.b));

    if (peak > 1.0)
        display /= peak;

    return display;
}

// Unclipped, hue-preserving Neutwo proxy adapted from clshortfuse/RenoDX.
// One scalar maps the peak channel to [0,1), preserving RGB ratios.
// Negative input channels are clamped before encoding. See Licenses/RenoDX_LICENSE.txt.
float Neutwo(float x) { return x * rsqrt(x * x + 1.0); } // [0, inf) -> [0, 1), no clip point

float3 NeutwoEncode(float3 v)
{
    v = max(v, 0.0);
    float m = max(v.r, max(v.g, v.b));

    if (m <= 1e-6)
        return v;

    // One scalar taken from the peak channel keeps the hue; the peak lands at Neutwo(m) < 1, so no
    // channel clips and LinearToSrgb's saturate never fires -- the proxy is fully invertible.
    return v * (Neutwo(m) / m);
}

// Replace-mode inverse of NeutwoEncode. Clamp below its pole at 1 to keep highlights finite.
float3 NeutwoDecode(float3 y)
{
    y = max(y, 0.0);
    float m = max(y.r, max(y.g, y.b));
    m = min(m, 0.999999);

    if (m <= 1e-6)
        return y;

    float x = m * rsqrt(max(1.0 - m * m, 1e-8)); // Neutwo^-1 of the peak
    return y * (x / m);
}

// Hybrid proxy: identity below the knee, a C1-continuous Neutwo rolloff above it.
float HybridCurve(float m)
{
    const float k = 0.75; // knee point: identity below, gentle unclipped roll above

    if (m <= k)
        return m;

    const float e = (m - k) / (1.0 - k);         // excess above the knee, [0, inf)
    return k + (1.0 - k) * (e * rsqrt(e * e + 1.0)); // Neutwo(e) scaled into [k, 1); -> 1, never clips
}

float3 HybridEncode(float3 v)
{
    v = max(v, 0.0);
    float m = max(v.r, max(v.g, v.b));

    if (m <= 1e-6)
        return v;

    // One scalar on the peak channel, hue preserved. Below the knee the scalar is 1 (identity); above
    // it the peak lands at HybridCurve(m) < 1, so no channel clips.
    return v * (HybridCurve(m) / m);
}

// The exact inverse of the hybrid curve, for the hybrid REPLACE decode (mode 4). Because it is IDENTITY
// below the knee, the steep expansion is confined to genuine highlights: midtone model wobble is not
// amplified, so hybrid-replace flashes far less than Neutwo-replace while keeping the raw model detail.
float HybridCurveInv(float y)
{
    const float k = 0.75;

    if (y <= k)
        return y;

    float u = (y - k) / (1.0 - k);                  // Neutwo(e), in [0,1)
    u = min(u, 0.999999);                           // the inverse diverges at 1
    const float e = u * rsqrt(max(1.0 - u * u, 1e-8)); // Neutwo^-1 of the excess
    return k + (1.0 - k) * e;
}

float3 HybridDecode(float3 y)
{
    y = max(y, 0.0);
    float m = max(y.r, max(y.g, y.b));

    if (m <= 1e-6)
        return y;

    return y * (HybridCurveInv(m) / m);
}

// Scale a residual so the result cannot leave the unit cube, without changing its direction.
//
// The model's edit is carried up from a smaller raster and laid on the frame's own proxy, so nothing
// guarantees the sum is still a colour. Clamping per channel would bend the hue -- the channel that
// hits the wall first decides the colour of the rest -- so the whole residual is scaled by the
// largest factor that keeps every channel inside, and the direction survives.
//
// hhkbble's, from the multi-pass PR against this fork.
float3 CubeScaleResidual(float3 P, float3 T)
{
    if (gPassthrough != 0)
        return T;

    float3 d = T - P;
    float alpha = 1.0;

    [unroll] for (int c = 0; c < 3; ++c)
    {
        if (d[c] > 1e-6)
            alpha = min(alpha, (1.0 - P[c]) / d[c]);
        else if (d[c] < -1e-6)
            alpha = min(alpha, (0.0 - P[c]) / d[c]);
    }

    return P + saturate(alpha) * d;
}

groupshared float4 gExposureReduce[64];

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.y * 8u + groupThreadId.x;

    if (gMode == 3)
    {
        if (groupId.x >= gWidth || groupId.y >= gHeight)
            return;

        uint fullW, fullH;
        gSource.GetDimensions(fullW, fullH);

        const uint tx0 = (groupId.x * fullW) / gWidth;
        const uint tx1 = ((groupId.x + 1u) * fullW) / gWidth;
        const uint ty0 = (groupId.y * fullH) / gHeight;
        const uint ty1 = ((groupId.y + 1u) * fullH) / gHeight;
        const uint endX = max(tx1, tx0 + 1u);
        const uint endY = max(ty1, ty0 + 1u);

        float localSum = 0.0;
        [loop] for (uint ty = ty0 + groupThreadId.y; ty < endY; ty += 8u)
        {
            [loop] for (uint tx = tx0 + groupThreadId.x; tx < endX; tx += 8u)
            {
                const float3 c = max(gSource.Load(int3(min(tx, fullW - 1u), min(ty, fullH - 1u), 0)).rgb, 0.0);
                const float luma = dot(c, kLuma);
                localSum += isfinite(luma) ? max(luma, 0.0) : 0.0;
            }
        }

        gExposureReduce[lane] = float4(localSum, 0.0, 0.0, 0.0);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint stride = 32u; stride > 0u; stride >>= 1u)
        {
            if (lane < stride)
                gExposureReduce[lane].x += gExposureReduce[lane + stride].x;
            GroupMemoryBarrierWithGroupSync();
        }

        if (lane == 0u)
        {
            const uint taken = (endX - tx0) * (endY - ty0);
            gTarget[groupId.xy] = float4(taken > 0u ? gExposureReduce[0].x / (float) taken : 0.0,
                                         0.0, 0.0, 1.0);
        }
        return;
    }

    if (gMode == 11)
    {
        const uint srcW = max(gExposureSourceWidth, 1u);
        const uint srcH = max(gExposureSourceHeight, 1u);
        const float preExposure =
            (isfinite(gPreExposure) && gPreExposure > 1e-6) ? gPreExposure : 1.0;
        const float protection = saturate(gExposureProtection * 0.01);

        float weightedBufferLuma = 0.0;
        float weightedSceneLogLuma = 0.0;
        float totalPixels = 0.0;

        [loop] for (uint index = lane; index < 4096u; index += 64u)
        {
            const uint tx = index & 63u;
            const uint ty = index >> 6u;
            const uint x0 = (tx * srcW) / 64u;
            const uint x1 = ((tx + 1u) * srcW) / 64u;
            const uint y0 = (ty * srcH) / 64u;
            const uint y1 = ((ty + 1u) * srcH) / 64u;
            const uint tileW = max(x1 - x0, 1u);
            const uint tileH = max(y1 - y0, 1u);
            const float pixels = (float) tileW * (float) tileH;
            const float tileMean = max(SanitizeFinite(gSource.Load(int3(tx, ty, 0)).r, 0.0), 0.0);

            weightedBufferLuma += tileMean * pixels;
            totalPixels += pixels;
            if (protection > 0.0)
            {
                const float sceneLuma = max(tileMean / preExposure, 1e-8);
                weightedSceneLogLuma += clamp(log2(sceneLuma), -24.0, 24.0) * pixels;
            }
        }

        gExposureReduce[lane] = float4(weightedBufferLuma, weightedSceneLogLuma, totalPixels, 0.0);
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint stride = 32u; stride > 0u; stride >>= 1u)
        {
            if (lane < stride)
                gExposureReduce[lane].xyz += gExposureReduce[lane + stride].xyz;
            GroupMemoryBarrierWithGroupSync();
        }

        const float allPixels = gExposureReduce[0].z;
        const float averageBufferLuma =
            allPixels > 0.0 ? gExposureReduce[0].x / allPixels : 0.0;
        float meteredSceneLuma = averageBufferLuma / preExposure;

        if (protection > 0.0 && allPixels > 0.0)
        {
            const float referenceLogLuma = gExposureReduce[0].y / allPixels;
            const float highlightKneeEv = lerp(3.0, 1.0, protection);
            const float highlightCompressionSlope = lerp(1.0, 0.35, protection);
            float protectedLinearSum = 0.0;

            [loop] for (uint index2 = lane; index2 < 4096u; index2 += 64u)
            {
                const uint tx2 = index2 & 63u;
                const uint ty2 = index2 >> 6u;
                const uint x0 = (tx2 * srcW) / 64u;
                const uint x1 = ((tx2 + 1u) * srcW) / 64u;
                const uint y0 = (ty2 * srcH) / 64u;
                const uint y1 = ((ty2 + 1u) * srcH) / 64u;
                const uint tileW = max(x1 - x0, 1u);
                const uint tileH = max(y1 - y0, 1u);
                const float pixels = (float) tileW * (float) tileH;
                const float tileMean = max(SanitizeFinite(gSource.Load(int3(tx2, ty2, 0)).r, 0.0), 0.0);
                const float sceneLuma = max(tileMean / preExposure, 1e-8);
                const float logLuma = clamp(log2(sceneLuma), -24.0, 24.0);
                const float deltaEv = logLuma - referenceLogLuma;
                float compressedLogLuma = logLuma;
                if (deltaEv > highlightKneeEv)
                    compressedLogLuma = referenceLogLuma + highlightKneeEv +
                                        (deltaEv - highlightKneeEv) * highlightCompressionSlope;
                protectedLinearSum += exp2(clamp(compressedLogLuma, -24.0, 24.0)) * pixels;
            }

            gExposureReduce[lane].w = protectedLinearSum;
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint stride2 = 32u; stride2 > 0u; stride2 >>= 1u)
            {
                if (lane < stride2)
                    gExposureReduce[lane].w += gExposureReduce[lane + stride2].w;
                GroupMemoryBarrierWithGroupSync();
            }

            const float protectedAverage = gExposureReduce[0].w / allPixels;
            if (isfinite(protectedAverage) && protectedAverage > 1e-8)
                meteredSceneLuma = protectedAverage;
        }

        if (lane == 0u)
        {
            float white = preExposure * meteredSceneLuma * (0.82 / 0.18);
            gTarget[uint2(0, 0)] = float4(isfinite(white) && white > 1e-8 ? white : 1.0, 0, 0, 1);
        }
        return;
    }

    if (id.x >= gWidth || id.y >= gHeight)
        return;

    // Normalised, so the source may be any size relative to this dispatch.
    float2 uv = (float2(id.xy) + 0.5) / float2(gWidth, gHeight);

    // Experimental private-DLSS carrier, not an ordinary colour image. Neutral 0.5 encodes zero;
    // values below it carry darkening. A reversible signed compression avoids clipping negative
    // edits at the DLSS input. Scale small linear-light edits up before storing them in FP16;
    // at unit scale, a dark scene's edits round to neutral before DLSS even sees them.
    if (gMode == 9)
    {
        float3 source = gSource.Load(int3(id.xy, 0)).rgb;
        float3 answer = gModel.Load(int3(id.xy, 0)).rgb;
        if (gPassthrough == 0) { source = SrgbToLinear(source); answer = SrgbToLinear(answer); }
        float3 d = SanitizeFinite3(answer - source, 0.0);
        gTarget[id.xy] = float4(0.5 + 0.5 * d / (1.0 / 64.0 + abs(d)), 1.0);
        return;
    }
    if (gMode == 10)
    {
        // Mode-local fields: guide active sizes/origins, and motion-to-working-pixel scale.
        uint2 dp = min(uint2(uv * uint2(gGuideWidth, gGuideHeight)),
                       uint2(gGuideWidth, gGuideHeight) - 1) + uint2(gDebugView, gCompareMode);
        uint2 size = uint2(gTransferStrength, gColourStrength);
        uint2 mp = min(uint2(uv * size), size - 1) + uint2(gCompareSwap, gTransfer);
        float z = gSource.Load(int3(dp, 0)).r;
        float2 mv = gModel.Load(int3(mp, 0)).xy * float2(gMvScaleX, gMvScaleY);
        gTarget[id.xy] = float4(isfinite(z) ? z : 0.0, 0, 0, 1);
        gKeep[id.xy] = float4(all(isfinite(mv)) ? mv : float2(0, 0), 0, 1);
        return;
    }
    if (gMode == 5)
    {
        float3 difference = SanitizeFinite3(gModel.Load(int3(id.xy, 0)).rgb -
                                            gSource.Load(int3(id.xy, 0)).rgb, 0.0);
        float3 d = difference / max(gResidualScale, 1e-4);
        gTarget[id.xy] = float4(0.5 + 0.5 * d / (1.0 + abs(d)), 1.0);
        return;
    }
    if (gMode == 6)
    {
        float4 base = gSource.Load(int3(id.xy, 0));
        float3 encoded = SanitizeFinite3(gModel.Load(int3(id.xy, 0)).rgb, 0.5);
        // Limit the inverse near its poles: DLSS can ring outside the carrier's [0,1] range.
        float3 signedEdit = clamp(2.0 * encoded - 1.0, -0.999, 0.999);
        float3 edit = signedEdit / (1.0 - abs(signedEdit)) * max(gResidualScale, 1e-4);
        gTarget[id.xy] = float4(max(SanitizeFinite3(base.rgb + edit, base.rgb), 0.0), base.a);
        return;
    }
    if (gMode == 7)
    {
        gTarget[id.xy] = 1.0;
        return;
    }

    if (gMode == 8)
    {
        // Already encoded: restore the input range without applying the tone curve again.
        float4 raw = gSource.Load(int3(id.xy, 0));
        gTarget[id.xy] = float4(saturate(SanitizeFinite3(raw.rgb, 0.5)), raw.a);
        return;
    }

    if (gMode == 2)
    {
        uint srcW, srcH;
        gSource.GetDimensions(srcW, srcH);

        // Nothing to do when the sizes already agree.
        if (srcW == gWidth && srcH == gHeight)
        {
            gTarget[id.xy] = gSource.Load(int3(id.xy, 0));
            return;
        }

        // Exact area-weighted downsampling avoids the aliasing of a single bilinear tap.
        // Adapted from hhkbble's multi-pass contribution.
        const float x0 = ((float) id.x * (float) srcW) / (float) gWidth;
        const float x1 = ((float) (id.x + 1) * (float) srcW) / (float) gWidth;
        const float y0 = ((float) id.y * (float) srcH) / (float) gHeight;
        const float y1 = ((float) (id.y + 1) * (float) srcH) / (float) gHeight;
        const float area = (x1 - x0) * (y1 - y0);

        const int i0 = (int) floor(x0);
        const int i1 = (int) ceil(x1) - 1;
        const int j0 = (int) floor(y0);
        const int j1 = (int) ceil(y1) - 1;

        float3 acc = 0.0;

        for (int j = j0; j <= j1; ++j)
        {
            const int jj = clamp(j, 0, (int) srcH - 1);
            const float aY = max(y0, (float) j);
            const float bY = min(y1, (float) j + 1.0);
            const float wy = max(bY - aY, 0.0);

            for (int i = i0; i <= i1; ++i)
            {
                const int ii = clamp(i, 0, (int) srcW - 1);
                const float aX = max(x0, (float) i);
                const float bX = min(x1, (float) i + 1.0);
                acc += gSource.Load(int3(ii, jj, 0)).rgb * (max(bX - aX, 0.0) * wy);
            }
        }

        const int acx = clamp((int) floor(((float) id.x + 0.5) * (float) srcW / (float) gWidth), 0, (int) srcW - 1);
        const int acy = clamp((int) floor(((float) id.y + 0.5) * (float) srcH / (float) gHeight), 0, (int) srcH - 1);

        gTarget[id.xy] = float4(acc / area, gSource.Load(int3(acx, acy, 0)).a);
        return;
    }

    if (gMode == 0)
    {
        float4 source = gSource.Load(int3(id.xy, 0));
        float3 frame = max(source.rgb, float3(0.0, 0.0, 0.0));

        // Kept so the resolve has the frame as it was, rather than having to reconstruct it.
        gKeep[id.xy] = float4(frame, source.a);

        // Some games hand DLSS a frame that has already been through their tonemapper. The game says
        // which in its own DLSS creation flags, and converting one that needs no conversion is pure
        // damage, so it goes through untouched.
        if (gPassthrough != 0)
        {
            gTarget[id.xy] = float4(frame, source.a);
            return;
        }

        // What the model is shown. Mode 2 -- the default -- scales the frame and encodes it, and that
        // is all: the game is going to tone map this picture later, so tone mapping it here as well
        // shows the model a doubly compressed image. Measured against Cyberpunk's own numbers, the
        // Reinhard proxy handed the model a scene value of 1.0 as 0.55 and 1.5 as 0.64 -- flat, dark,
        // and nothing like the finished frame it was trained on. The model then synthesised weakly,
        // judged tone on a picture that does not exist, and its answer had to be un-crushed on the way
        // back. Mode 0 keeps that old curve, mode 1 the fitted one.
        // A soft knee instead of a hard ceiling. Anything above 0.75 is rolled off rather than
        // clipped, so the model is never shown a field of flat white whose blown pixels flip between
        // frames -- unstable input is unstable output, and this is where a bright scene would produce
        // it. The resolve reproduces this exactly, so the two agree on what the frame's own proxy is.
        // The classic soft knee, or -- when the reversible proxy is on -- the unclipped Neutwo encode
        // that shows the model highlight gradation the knee throws away. Reached only when the frame
        // is not passthrough (handled and returned above), so NeutwoEncode never sees a tone-mapped
        // frame. Both are undone by the resolve: the knee approximately, Neutwo exactly.
        float3 normalized = frame / WhitePoint();
        float3 display;
        if (gReversibleMode == 0)
            display = SoftKnee(normalized);        // soft knee
        else if (gReversibleMode >= 3)
            display = HybridEncode(normalized);    // 3 hybrid composed, 4 hybrid replace -- same curve
        else
            display = NeutwoEncode(normalized);    // 1 composed, 2 replace -- both the full Neutwo proxy

        // The reversible proxy forces opaque alpha -- feature 18 expects an opaque colour input, and
        // the frame's own alpha is not part of what the model reads. The knee path keeps the frame's
        // alpha, so the default stays byte-identical.
        float alpha = gReversibleMode != 0 ? 1.0 : source.a;

        gTarget[id.xy] = float4(LinearToSrgb(display), alpha);
        return;
    }

    // Comparison, decided before anything is read, because side by side changes which part of the
    // frame this pixel is showing rather than just which version of it.
    //
    //   1  side by side  each half carries the whole frame, so both are squeezed horizontally
    //   2  wipe          one frame cut at the split, nothing resampled
    //
    // Neither needs the menu open to stay up. The wipe's split is a setting like any other; the menu
    // is only how you drag it.
    float2 cmpUv = uv;
    bool showOriginal = false;
    bool onDivider = false;
    bool outsideFrame = false;

    if (gCompareMode == 1)
    {
        showOriginal = (uv.x < 0.5) != (gCompareSwap != 0);

        // Each half is half as wide as the frame and just as tall, so the frame cannot fill it and
        // keep its shape. Stretching it to fit is what made both sides look squashed. Fitting it
        // properly leaves the halves letterboxed, which is the honest way round: a comparison that
        // changes the shape of what it is comparing is not showing you the picture.
        //
        // Zoom decides which is given up. At 1 the whole frame is there at its right proportions
        // with bars above and below; at 2 the half is filled and the sides are cropped away.
        float2 half2 = float2(uv.x < 0.5 ? uv.x * 2.0 : (uv.x - 0.5) * 2.0, uv.y) - 0.5;
        cmpUv = float2(0.5 + half2.x / gCompareZoom, 0.5 + half2.y * 2.0 / gCompareZoom);

        outsideFrame = cmpUv.x < 0.0 || cmpUv.x > 1.0 || cmpUv.y < 0.0 || cmpUv.y > 1.0;
        onDivider = abs(uv.x - 0.5) < (1.0 / max(gWidth, 1u));
    }
    else if (gCompareMode == 2)
    {
        showOriginal = (uv.x < gCompareSplit) != (gCompareSwap != 0);
        onDivider = abs(uv.x - gCompareSplit) < (1.0 / max(gWidth, 1u));
    }

    // Sampled rather than loaded: when the model ran at a reduced resolution these are smaller than the
    // frame, and its edit is enlarged here while the frame underneath stays untouched.
    float4 proxySample = gSource.SampleLevel(gLinear, cmpUv, 0);
    float4 modelSample = gModel.SampleLevel(gLinear, cmpUv, 0);

    // Nothing was encoded on the way in, so nothing is decoded here either.
    float3 proxy = gPassthrough != 0 ? proxySample.rgb : SrgbToLinear(proxySample.rgb);
    float3 model = gPassthrough != 0 ? modelSample.rgb : SrgbToLinear(modelSample.rgb);

    // The model's own answer, kept before the matched-residual block below can rewrite `model`, so the
    // replace decode uses what the model returned rather than the residual reconstruction.
    float3 modelDirect = model;
    float4 originalSample = gCompareMode == 1 ? gOriginal.SampleLevel(gLinear, cmpUv, 0)
                                              : gOriginal.Load(int3(id.xy, 0));

    // All three pictures have to share a scale before their luminances can be compared. The proxy and
    // the model come back from an sRGB decode, so they sit in 0..1 where 1 is the white point; the
    // frame is raw linear and runs well past that. Comparing them unnormalised is a real bug and it
    // reads exactly like the model has stopped adding detail: with the frame several times larger,
    // the shadow branch never fires, every pixel takes the highlight branch, and the clamp flattens
    // the result to a near-constant scale. Colour still moves, because that comes from the model's
    // own hue, which is what makes the failure so confusing to look at.
    const float normScale = gPassthrough != 0 ? 1.0 : WhitePoint();
    float3 original = originalSample.rgb / normScale;

    float originalLuma = dot(original, kLuma);
    float proxyLuma = dot(proxy, kLuma);

    // Apply the model. Off outputs the frame as the upscaler produced it (clean) while the pass keeps
    // running -- so with Hold frame you can freeze a frame and toggle this to A/B the same frozen frame
    // with and without Neural Rendering. In passthrough the frame is already display-referred.
    if (gApplyModel == 0)
    {
        gTarget[id.xy] = float4(max(originalSample.rgb, 0.0), originalSample.a);
        return;
    }

    if (gDebugView == 1)
    {
        gTarget[id.xy] = float4(proxy * gDebugScale, originalSample.a);
        return;
    }

    if (gDebugView == 2)
    {
        gTarget[id.xy] = float4(model * gDebugScale, originalSample.a);
        return;
    }

    float3 edit = model - proxy;
    if (gTransfer == 2)
    {
        float3 carrier = clamp(2.0 * SanitizeFinite3(modelSample.rgb, 0.5) - 1.0, -0.999, 0.999);
        edit = (1.0 / 64.0) * carrier / (1.0 - abs(carrier));
    }

    // Coring was tried here and removed: the per-frame churn's amplitude overlaps the real detail's,
    // so an amplitude threshold cannot separate them -- it only relocated the noise to the threshold.

    if (gDebugView == 3)
    {
        // Amplified and centred on grey, so both directions of the edit are visible at once.
        float3 shown = saturate(0.5 + edit * 20.0);
        gTarget[id.xy] = float4(SrgbToLinear(shown) * gDebugScale, originalSample.a);
        return;
    }

    // Composition uses the current model answer; temporal residual accumulation is a separate pass.

    // Rebuild the full-resolution proxy and add only the upsampled model difference.
    // Skip ordinary matched residual at native resolution to preserve Classic's exact arithmetic.
    // Residual transfer and cube scaling are adapted from hhkbble's multi-pass contribution.
    uint proxyW, proxyH;
    gSource.GetDimensions(proxyW, proxyH);
    const bool modelRanSmall = proxyW != gWidth || proxyH != gHeight;

    if ((gTransfer == 1 && modelRanSmall) || gTransfer == 2)
    {
        // Match the encode's curve, passthrough and saturation before cube-scaling the residual.
        // An out-of-range reconstructed proxy would collapse the residual scale to zero.
        float3 fullProxy = gPassthrough != 0
                               ? saturate(original)
                               : (gReversibleMode == 0   ? saturate(SoftKnee(original))
                                  : gReversibleMode >= 3 ? HybridEncode(original)
                                                         : NeutwoEncode(original));
        proxy = fullProxy;
        proxyLuma = dot(proxy, kLuma);

        // At the same rate there is no residual to carry: the model's own picture is already at the
        // frame's resolution, and P + (m - p) collapses to m exactly.
        model = CubeScaleResidual(fullProxy, fullProxy + edit);
        if (gTransfer == 2) modelDirect = model;
    }

    // Rescale the model answer to the original luminance and restore headroom lost by the proxy.
    float modelLuma = dot(model, kLuma);
    float3 upgraded;

    if (modelLuma <= 1e-5)
    {
        // The model can return an empty frame for an input it cannot read. Rescaling that collapses
        // the picture to black, so the frame is handed back untouched instead.
        upgraded = original;
    }
    else
    {
        float ratio;

        if (originalLuma < proxyLuma)
        {
            // Below what the proxy showed: the frame's own luminance is the target.
            ratio = originalLuma / max(proxyLuma, 1e-6);
        }
        else
        {
            // Above it, the difference is headroom the proxy could not represent -- brightness the
            // frame really has and the model never saw. It is handed back on top of the model's own
            // answer rather than scaled away, which is what kept highlights from being muted.
            ratio = (modelLuma + max(0.0, originalLuma - proxyLuma)) / modelLuma;
        }

        // Keep the RGB blend within [0,1]; strength above 1 amplifies the bounded luminance ratio below.
        upgraded = lerp(original, HueOkLab(model * ratio, model), saturate(gTransferStrength));
    }

    // Detail strength decides how much of the model's picture is reached at all; colour strength
    // decides whether its colour comes with it. At 0 the frame keeps the game's own hue exactly and
    // only its light carries the model's verdict; at 1 the model's colour arrives as well.
    float upgradedLuma = dot(upgraded, kLuma);

    // A common luminance floor suppresses unstable ratios in near-black pixels.
    const float kRatioFloor = 1.0 / 512.0;
    float lumaRatio = (upgradedLuma + kRatioFloor) / (originalLuma + kRatioFloor);

    // Amplify detail through a luminance-ratio power while preserving neutral edits.
    const float amplified = pow(max(lumaRatio, 1e-6), 1.0 + max(gTransferStrength - 1.0, 0.0));

    // Independent one-sided guards. Highlight guard only limits brightening. Darkening guard only
    // raises the lower luminance-ratio bound; at 100% its floor is zero, so darkening is uncapped.
    const float highlightGuard = max(gMaxRatio, 1.0);
    const float darkeningFloor = 1.0 - saturate(gMaxDarkening / 100.0);
    float boundedRatio = clamp(amplified, darkeningFloor, highlightGuard);

    // Exactly one while the ratio is already inside the guard, so a frame that never needed bounding
    // is untouched rather than rounded, and strength zero stays bit-identical.
    upgraded *= boundedRatio / max(lumaRatio, 1e-6);

    // Both blend endpoints obey the luminance guard. Above colour strength 1, boost OkLab chroma
    // while preserving lightness/hue, then compress out-of-gamut colours toward neutral.
    float3 result = lerp(original * boundedRatio, upgraded, min(gColourStrength, 1.0));

    if (gColourStrength > 1.0)
        result = ClampAp1(FromOkLab(float3(1.0, gColourStrength, gColourStrength) * ToOkLab(max(result, 0.0))));

    // Replace modes decode the model answer directly; passthrough colour needs no inverse transform.
    if (gReversibleMode == 2)
        result = gPassthrough != 0 ? modelDirect : NeutwoDecode(modelDirect);
    else if (gReversibleMode == 4)
        result = gPassthrough != 0 ? modelDirect : HybridDecode(modelDirect);

    // Restore native luminance detail with a bounded, positive ratio (including near-black edges).
    if ((gReversibleMode == 2 || gReversibleMode == 4) && gModelWorkScale > 0.0 &&
        gModelWorkScale < 0.999 && gReplaceDetailStrength > 0.0)
    {
        float2 tap = clamp(round(1.0 / gModelWorkScale), 1.0, 4.0) / float2(gWidth, gHeight);
        float3 neighbours = gOriginal.SampleLevel(gLinear, cmpUv + float2(tap.x, 0), 0).rgb +
                            gOriginal.SampleLevel(gLinear, cmpUv - float2(tap.x, 0), 0).rgb +
                            gOriginal.SampleLevel(gLinear, cmpUv + float2(0, tap.y), 0).rgb +
                            gOriginal.SampleLevel(gLinear, cmpUv - float2(0, tap.y), 0).rgb;
        float blur = dot(original + neighbours / normScale, kLuma) / 5.0;
        float contrast = (originalLuma - blur) / (max(originalLuma, blur) + kRatioFloor);
        result *= exp2(clamp(gReplaceDetailStrength, 0.0, 2.0) * clamp(contrast, -1.0, 1.0));
    }

    // Back out of the normalised space the composition worked in.
    result *= normScale;

    if (gSkinProtection != 0)
    {
        // Classify the untouched frame, never NR's recoloured output. Controls
        // attenuate the final edit, including replace mode and all model passes.
        float3 displayRgb = gPassthrough != 0 ? original : LinearToSrgb(saturate(original));
        float mask = SkinColourWeight(displayRgb);
        float detail = lerp(gEnvironmentDetail, gSkinDetail, mask);
        float colour = lerp(gEnvironmentColour, gSkinColour, mask);
        float baseY = dot(max(originalSample.rgb, 0.0), kLuma);
        float editedY = dot(max(result, 0.0), kLuma);
        float wantedY = lerp(baseY, editedY, detail);
        float3 baseChroma = originalSample.rgb / max(baseY, 1e-6);
        float3 editedChroma = result / max(editedY, 1e-6);
        // Exact endpoints avoid changing the default image or fully protected pixels.
        if (detail == 0.0 && colour == 0.0)
            result = originalSample.rgb;
        else if (detail != 1.0 || colour != 1.0)
            result = ClampAp1(lerp(baseChroma, editedChroma, colour) * wantedY);
        if (gShowSkinMask != 0)
            result = mask.xxx * normScale;
    }

    // The stabilized ratio above deliberately adds kRatioFloor, which is useful for model composition
    // but can hide large real percentage losses in very dark pixels. Enforce the darkening control
    // once more on the final edited luminance without that stabilization term. This makes 0% mean
    // exactly "no luminance darkening" and N% cap the final reduction to N%, including deep shadows,
    // replace modes and colour/skin composition. 100% remains a no-op.
    const float finalDarkeningFloor = 1.0 - saturate(gMaxDarkening / 100.0);
    if (finalDarkeningFloor > 0.0 && gShowSkinMask == 0)
    {
        const float baseY = dot(max(originalSample.rgb, 0.0), kLuma);
        const float editedY = dot(max(result, 0.0), kLuma);
        const float minimumY = baseY * finalDarkeningFloor;
        if (editedY < minimumY)
        {
            if (editedY > 1e-6)
                result *= minimumY / editedY;
            else if (baseY > 1e-6)
                result = max(originalSample.rgb, 0.0) * finalDarkeningFloor;
        }
    }

    // The side being shown untouched takes the frame as it arrived, past every step above.
    if (showOriginal)
        result = originalSample.rgb;

    // The letterbox. The sampler clamps rather than wrapping, so without this the bars would be the
    // frame's edge row smeared down the screen.
    if (outsideFrame)
        result = float3(0.0, 0.0, 0.0);

    // A hairline so the two sides are never mistaken for one picture.
    if (onDivider)
        result = float3(WhitePoint(), WhitePoint(), WhitePoint());

    gTarget[id.xy] = float4(max(result, float3(0.0, 0.0, 0.0)), originalSample.a);
}
