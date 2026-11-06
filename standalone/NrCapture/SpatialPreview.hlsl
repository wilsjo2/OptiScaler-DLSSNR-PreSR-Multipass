// Keep this layout identical to DlssNr::Spatial::Constants and dlssnr_spatial.hlsl.
cbuffer Params : register(b0)
{
    uint gMode; float gGridSpacing; uint gWidth, gHeight;
    float4 gNativeWorkSize;
    float4 gCenterWorkFractions;
    float4 gCompressionEdgeSlope;
    uint4 gOptions;
    float4 gBandCenter;
    float4 gSideNeg;
    float4 gSidePos;
    float4 gSideWorkNeg;
    float4 gSideWorkPos;
    float4 gSideEdgeSlope;
    float4 gWorkScale;
    float4 gDepthRect;
    float4 gMotionRect;
    float2 gMotionScale;
    float2 gPadding0;
    float4 gPadding1;
};
Texture2D<float4> gSource : register(t0);
RWTexture2D<float4> gOutput : register(u0);

#include "../../OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_warp.hlsli"

float2 GridPosition(float2 pixel)
{
    return gMode == 103 ? PwPack(pixel) : pixel * gNativeWorkSize.zw / gNativeWorkSize.xy;
}

[numthreads(8, 8, 1)]
void Preview(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) return;
    const float2 pixel = float2(id.xy) + 0.5;
    const float2 packed = GridPosition(pixel);
    // The mapping is separable by axis: spacing, rather than line curvature,
    // exposes how many model samples each part of the original image receives.
    const float2 derivative = max(abs(GridPosition(pixel + 0.5) - GridPosition(pixel - 0.5)), 1e-4);
    const float2 cell = packed / max(gGridSpacing, 1.0);
    const float2 distance = abs(frac(cell + 0.5) - 0.5) * gGridSpacing / derivative;
    const float coverage = 1.0 - smoothstep(0.35, 1.15, min(distance.x, distance.y));
    float4 source = gSource.Load(int3(id.xy, 0));
    const float density = saturate(min(derivative.x, derivative.y) / max(min(gWorkScale.x, gWorkScale.y), 1e-4));
    const float3 ink = lerp(float3(1.0, 0.65, 0.02), float3(0.02, 1.0, 0.3), gMode == 103 ? density : 1.0);
    gOutput[id.xy] = float4(lerp(source.rgb, ink, coverage * 0.8), source.a);
}
