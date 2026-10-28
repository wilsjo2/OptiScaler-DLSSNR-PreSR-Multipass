// Adapted from Lecram's RenoDX DLSS5 look shaders (MIT).
// See Licenses/RenoDX_ATTRIBUTION.txt for the source revision and licence.
static const float kNrEditFloor = 1.0 / 1024.0;

float3 NrLogEdit(int2 pixel, out float inputLight)
{
    float3 proxy = max(SanitizeFinite3(SrgbToLinear(gSource.Load(int3(pixel, 0)).rgb), 0), 0);
    float3 model = max(SanitizeFinite3(SrgbToLinear(gModel.Load(int3(pixel, 0)).rgb), proxy), 0);
    inputLight = log2(max(dot(proxy, kLuma), kNrEditFloor));
    return log2(max(model, kNrEditFloor)) - log2(max(proxy, kNrEditFloor));
}

// Fit around luminance rather than clipping each colour channel independently.
float3 NrFitQualityModel(float3 value)
{
    float y = saturate(dot(value, kLuma));
    float3 d = value - y;
    float amount = 1;
    [unroll] for (int c = 0; c < 3; ++c)
    {
        if (d[c] > 0) amount = min(amount, (1 - y) / d[c]);
        else if (d[c] < 0) amount = min(amount, -y / d[c]);
    }
    return saturate(y + saturate(amount) * d);
}

// Enlarge only the model's logarithmic RGB gain. The original pixel guides
// the filter, so samples across a brightness edge contribute less.
float3 NrEdgeAwareModel(float3 fullProxy, float2 uv)
{
    uint w, h; gSource.GetDimensions(w, h);
    float2 position = uv * float2(w, h) - 0.5;
    int2 base = int2(floor(position));
    float3 fullLinear = gPassthrough != 0 ? SrgbToLinear(fullProxy) : fullProxy;
    float guide = log2(max(dot(max(fullLinear, 0), kLuma), kNrEditFloor));
    float3 joint = 0, spatial = 0;
    float jointWeight = 0, spatialWeight = 0;
    [unroll] for (int y = -1; y <= 2; ++y) [unroll] for (int x = -1; x <= 2; ++x)
    {
        int2 tap = base + int2(x, y);
        float inputLight;
        float3 edit = NrLogEdit(clamp(tap, int2(0, 0), int2(w - 1, h - 1)), inputLight);
        float2 distance = float2(tap) - position;
        float ws = exp(-2.0 * dot(distance, distance)); // sigma = half an NR pixel
        float difference = guide - inputLight;
        float weight = ws * exp(-2.0 * difference * difference); // sigma = half a stop
        spatial += ws * edit; spatialWeight += ws;
        joint += weight * edit; jointWeight += weight;
    }
    float3 edit = jointWeight > 1e-6 ? joint / jointWeight : spatial / max(spatialWeight, 1e-6);
    float3 answer = NrFitQualityModel(max(fullLinear, 0) * exp2(clamp(edit, -10, 10)));
    return gPassthrough != 0 ? LinearToSrgb(answer) : answer;
}

// Two small, brightness-guided neighbourhoods split the edit into fine,
// medium and broad lighting. Only negative fine/medium bands can be lifted.
// This runs once on the NR grid, before any enlargement or HDR inversion.
float4 NrSuppressHalo(int2 pixel)
{
    float4 answer = gModel.Load(int3(pixel, 0));
    if (gHaloSuppression <= 0 && gHaloBrightSuppression <= 0) return answer;
    float inputLight;
    float3 edit = NrLogEdit(pixel, inputLight);
    float a = dot(edit, kLuma);
    float narrow = 0, wide = 0, narrowWeight = 0, wideWeight = 0;
    float peak = inputLight, trough = inputLight;
    int radius = int(clamp(round(gHaloRadius), 1, 4));
    [unroll] for (int y = -1; y <= 1; ++y) [unroll] for (int x = -1; x <= 1; ++x)
    {
        float spatialWeight = (x == 0 ? 2.0 : 1.0) * (y == 0 ? 2.0 : 1.0);
        [unroll] for (int band = 0; band < 2; ++band)
        {
            int2 tap = clamp(pixel + int2(x, y) * radius * (band == 0 ? 1 : 3),
                             int2(0, 0), int2(gWidth - 1, gHeight - 1));
            float light;
            float value = dot(NrLogEdit(tap, light), kLuma);
            peak = max(peak, light); trough = min(trough, light);
            float delta = light - inputLight;
            float weight = spatialWeight * exp(-2.0 * delta * delta);
            if (band == 0) { narrow += weight * value; narrowWeight += weight; }
            else { wide += weight * value; wideWeight += weight; }
        }
    }
    narrow /= max(narrowWeight, 1e-6); wide /= max(wideWeight, 1e-6);
    float fine = a - narrow, mid = narrow - wide;
    float wmax = smoothstep(1.0, 3.0, peak - inputLight);
    float wmin = smoothstep(1.0, 3.0, inputLight - trough);
    float shapedFine = fine - saturate(gHaloSuppression) * wmax * min(fine, 0);
    shapedFine -= saturate(gHaloBrightSuppression) * max(wmax, wmin) * min(shapedFine, 0);
    float lift = shapedFine - fine - saturate(gHaloBrightSuppression) * wmin * min(mid, 0);
    if (lift <= 1e-6) return answer; // Preserve broad shading and the disabled result exactly.
    float3 corrected = NrFitQualityModel(max(SrgbToLinear(answer.rgb), 0) * exp2(min(lift, 2.0)));
    return float4(LinearToSrgb(corrected), answer.a);
}
