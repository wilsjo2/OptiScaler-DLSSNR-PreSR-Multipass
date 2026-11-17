// Re-apply the last network pass's change instead of running the network again.
// One dispatch covers a run of r re-applying passes: gGain = r * ReapplyGain, so
// out = saturate(out_m + gGain * (out_m - in_m)), alpha from out_m. Because each channel's delta keeps
// its sign, this equals applying saturate(prev + ReapplyGain * (out_m - in_m)) r times.
// Kept separate so the existing encode/composition shader remains byte-identical.
#ifdef VK_MODE
[[vk::binding(0, 0)]]
#endif
cbuffer Params : register(b0)
{
    uint gMode;
    float gGain;
    uint gWidth;
    uint gHeight;
};
#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float4> gSource : register(t0); // in_m: what the last network pass was shown
#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float4> gModel : register(t1); // out_m: what it returned
#ifdef VK_MODE
[[vk::binding(5, 0)]]
[[vk::image_format("rgba16f")]]
#endif
RWTexture2D<float4> gTarget : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) return;
    float4 answer = gModel.Load(int3(id.xy, 0));
    float3 shown = gSource.Load(int3(id.xy, 0)).rgb;
    float3 result = answer.rgb + gGain * (answer.rgb - shown);
    // A non-finite pixel falls back to mid-grey, as the between-pass clamp does.
    result = float3(isfinite(result.x) ? result.x : 0.5, isfinite(result.y) ? result.y : 0.5,
                    isfinite(result.z) ? result.z : 0.5);
    gTarget[id.xy] = float4(saturate(result), answer.a);
}
