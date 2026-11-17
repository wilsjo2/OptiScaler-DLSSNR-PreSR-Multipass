// Reconstruct NR at the capture's resolution before the shared final composition.
// Absolute edits cannot be moved between differently filtered baselines: even a
// uniform darkening leaves F + U((gain-1)*L) brighter than gain*F at a sharp edge.
// Carry relative colour/edit instead. Both upscalers operate
// on that relative field; final strength, HDR and skin controls still run once.
Texture2D<uint> gEditCuts : register(t4);   // mode 17 with a kept edit: the motion estimator's cut counter
float3 EncodeResizeField(float3 field)
{
    return 0.5 + 0.5 * field / (1 + abs(field));
}

float3 DecodeResizeField(float3 carrier)
{
    float3 d = clamp(2 * SanitizeFinite3(carrier, 0.5) - 1, -0.999, 0.999);
    return d / (1 - abs(d));
}

float3 SampleResizeField(Texture2D<float4> image, float2 uv)
{
    uint w, h; image.GetDimensions(w, h);
    float2 p = uv * float2(w, h) - 0.5;
    int2 base = int2(floor(p));
    float2 f = frac(p);
    float3 value = 0;
    // Decode before bilinear interpolation: the bounded DLSS carrier is nonlinear.
    [unroll] for (int y = 0; y < 2; ++y) [unroll] for (int x = 0; x < 2; ++x) {
        int2 q = clamp(base + int2(x,y), int2(0,0), int2(w-1,h-1));
        value += DecodeResizeField(image.Load(int3(q,0)).rgb) *
            (x ? f.x : 1-f.x) * (y ? f.y : 1-f.y);
    }
    return value;
}

float2 ResizeChroma(float3 colour)
{
    float y = dot(colour, kLuma);
    return y > 1e-6 ? colour.rb / y : float2(1,1);
}

float4 PrepareModelResize(uint2 pixel)
{
    float3 source = gSource.Load(int3(pixel,0)).rgb;
    float3 model = gModel.Load(int3(pixel,0)).rgb;
    // Transfer 3 ("difference"): the edit as the plain difference of the values the model was given and answered
    // with. For a model made for a reduced-size pass: its picture's difference, enlarged, is the full-size edit
    // to within 44 to 52 dB on clean frames
    // (measured; the relative field below cannot come nearer than 30 to 42 dB on saturated game frames).
    if (gTransfer == 3) return float4(EncodeResizeField(SanitizeFinite3(model - source, 0)), 1);
    // Transfer 4: the same difference for a 100 % pass that keeps its edit over frames (Settings::editInterval),
    // stored as it is: nothing enlarges it, so it needs no bounded carrier and loses nothing to one.
    if (gTransfer == 4) return float4(SanitizeFinite3(model - source, 0), 1);
    if (gPassthrough == 0) { source = SrgbToLinear(source); model = SrgbToLinear(model); }
    float sourceY = dot(source, kLuma), modelY = dot(model, kLuma);
    if (sourceY < 1.0 / 1024.0) {
        // Black contains no measurable lighting gain. Infer it from paired
        // neighbouring samples so an undefined gain does not dilute a valid
        // edit when an upscaler interpolates across a black/bright boundary.
        sourceY = modelY = 0;
        [unroll] for (int y = -1; y <= 1; ++y) [unroll] for (int x = -1; x <= 1; ++x) {
            int2 q = clamp(int2(pixel) + int2(x,y), int2(0,0), int2(gWidth-1,gHeight-1));
            float3 p = gSource.Load(int3(q,0)).rgb, m = gModel.Load(int3(q,0)).rgb;
            if (gPassthrough == 0) { p = SrgbToLinear(p); m = SrgbToLinear(m); }
            sourceY += dot(p,kLuma); modelY += dot(m,kLuma);
        }
    }
    float lighting = (modelY - sourceY) / max(sourceY, 1.0 / 1024.0);
    float2 chroma = ResizeChroma(model);
    if (gTransfer == 1) chroma -= ResizeChroma(source);
    // Keep lighting separate from chroma: RGB resampling/gamut limits must not
    // turn a colour boundary into an extra luminance change.
    return float4(EncodeResizeField(SanitizeFinite3(float3(lighting, chroma), 0)), 1);
}

float3 RestoreResizeField(float3 base, float3 field)
{
    float baseY = dot(base, kLuma);
    float y = max(0, baseY * (1 + field.x));
    float2 chroma = field.yz + (gTransfer == 1 ? ResizeChroma(base) : float2(0,0));
    float r = chroma.x * y, b = chroma.y * y;
    return float3(r, (y - kLuma.r*r - kLuma.b*b) / kLuma.g, b);
}

float4 RestoreModelResize(uint2 pixel)
{
    float2 uv = (pixel + 0.5) / float2(gWidth, gHeight);
    float4 encodedBase = gModel.Load(int3(pixel,0));
    float3 base = gPassthrough != 0 ? encodedBase.rgb : SrgbToLinear(encodedBase.rgb);
    uint w, h; gSource.GetDimensions(w,h);
    // The kept edit of an earlier frame (Settings::editInterval) follows the picture: gOriginal then holds this
    // frame's motion (current to previous, in that texture's pixels) and gDebugView the frames since the edit was made.
    const bool moved = gTransfer >= 3 && gDebugView != 0;
    if (moved) {
        // A new scene keeps nothing of the old one's edit: the motion estimator's count of coarse cells without a
        // match in the frame before, against the count the host allows (gCompareMode; 0: not looked at).
        if (gCompareMode != 0 && gEditCuts.Load(int3(0,0,0)) > gCompareMode) return encodedBase;
        uint mw, mh; gOriginal.GetDimensions(mw, mh);
        uv += gOriginal.SampleLevel(gLinear, uv, 0).xy * float(gDebugView) / float2(mw, mh);
    }
    if (gTransfer == 4) {
        float3 plain = moved || w != gWidth || h != gHeight ? gSource.SampleLevel(gLinear, uv, 0).rgb : gSource.Load(int3(pixel,0)).rgb;
        return float4(saturate(encodedBase.rgb + plain), encodedBase.a);
    }
    float3 field = w == gWidth && h == gHeight && !moved ? DecodeResizeField(gSource.Load(int3(pixel,0)).rgb)
                                                         : SampleResizeField(gSource, uv);
    if (gTransfer == 3) return float4(saturate(encodedBase.rgb + field), encodedBase.a);
    if (gTransfer == 1 && all(field == 0)) return encodedBase;
    float3 result = RestoreResizeField(base, field);
    if (gPassthrough == 0 && (gReversibleMode == 2 || gReversibleMode == 4)) {
        float3 reference = RestoreResizeField(base, SampleResizeField(gOriginal, uv));
        // A DLSS-only crossing of the HDR inverse's white pole uses the stable
        // bilinear field. Genuine spatial highlights retain their existing range.
        if (any(result >= 1) && all(reference < 1)) result = reference;
    }
    // Fit chroma into the proxy gamut around the requested luminance. Scaling a
    // whole RGB residual toward the original would undo lighting at colour edges.
    float y = saturate(dot(result, kLuma));
    float3 delta = result - y;
    float amount = 1;
    [unroll] for (int ch = 0; ch < 3; ++ch) {
        if (delta[ch] > 0) amount = min(amount, (1-y) / delta[ch]);
        else if (delta[ch] < 0) amount = min(amount, -y / delta[ch]);
    }
    result = saturate(y + saturate(amount) * delta);
    return float4(gPassthrough != 0 ? result : LinearToSrgb(result), encodedBase.a);
}
