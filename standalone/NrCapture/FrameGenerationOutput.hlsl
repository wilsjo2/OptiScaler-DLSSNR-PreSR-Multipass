Texture2D<float4> Input : register(t0);
Texture2D<float2> SceneMotion : register(t1);
RWTexture2D<float4> Colour : register(u0);
RWTexture2D<float> Depth : register(u1);
RWTexture2D<float2> Vectors : register(u2);
SamplerState LinearClamp : register(s0);
cbuffer Params : register(b0) { uint Width, Height, Hdr, MotionWidth, MotionHeight; };
float3 Pq(float3 value) {
    float3 p = pow(saturate(value), 2610.0 / 16384.0);
    return pow((3424.0 / 4096.0 + (2413.0 / 128.0) * p) / (1.0 + (2392.0 / 128.0) * p), 2523.0 / 32.0);
}
[numthreads(8,8,1)]
void Encode(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    float3 value = Input.Load(int3(id.xy, 0)).rgb;
    if (Hdr) {
        // scRGB is linear BT.709 with 1 = 80 nits; HDR10 is BT.2020/PQ.
        value = mul(float3x3(0.627404,0.329283,0.043313, 0.069097,0.919540,0.011362,
                            0.016391,0.088013,0.895595), value);
        value = Pq(value * (80.0 / 10000.0));
    } else {
        value = saturate(value);
        float3 high = 1.055 * pow(value, 1.0 / 2.4) - 0.055;
        value = float3(value.r <= .0031308 ? value.r * 12.92 : high.r,
                       value.g <= .0031308 ? value.g * 12.92 : high.g,
                       value.b <= .0031308 ? value.b * 12.92 : high.b);
    }
    Colour[id.xy] = float4(value, 1);
    Depth[id.xy] = 0.5; // Explicitly synthetic planar depth; no engine/camera data is available.
    // NR's scene motion at another size: resample it and convert its pixel units to output pixels.
    if (MotionWidth)
        Vectors[id.xy] = SceneMotion.SampleLevel(LinearClamp, (id.xy + 0.5) / float2(Width, Height), 0) *
            (float2(Width, Height) / float2(MotionWidth, MotionHeight));
}
