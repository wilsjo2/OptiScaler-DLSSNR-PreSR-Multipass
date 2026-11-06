// Reconstruct a raw encoded proxy for the next independently sized NR pass.
// Kept separate so the existing encode/composition shader remains byte-identical.
#ifdef VK_MODE
[[vk::binding(0, 0)]]
#endif
cbuffer Params : register(b0)
{
    uint gMode;
    float gWhitePoint;
    uint gWidth;
    uint gHeight;
};
#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float4> gSource : register(t0);
#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float4> gModel : register(t1);
#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
Texture2D<float4> gOriginal : register(t2);
#ifdef VK_MODE
[[vk::binding(5, 0)]]
[[vk::image_format("rgba16f")]]
#endif
RWTexture2D<float4> gTarget : register(u0);
#ifdef VK_MODE
[[vk::binding(7, 0)]]
#endif
SamplerState gLinear : register(s0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(gWidth, gHeight);
    float4 original = gOriginal.Load(int3(id.xy, 0));
    float3 edit = gModel.SampleLevel(gLinear, uv, 0).rgb - gSource.SampleLevel(gLinear, uv, 0).rgb;
    // Match passthrough matched-residual reconstruction, without final scene composition.
    gTarget[id.xy] = float4(max(0.0, saturate(original.rgb) + edit), original.a);
}
