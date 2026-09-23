// NR peripheral Pack/Unpack. Mapping and filter adapted from PeripheralWarp
// 64902dd6a02460e5f6b778504ec2a4005faf4d9c (MIT; see external/peripheral_warp/LICENSE).

#ifdef VK_MODE
[[vk::binding(0, 0)]]
cbuffer Params : register(b0, space0)
#else
cbuffer Params : register(b0)
#endif
{
    uint gMode; float gUnused; uint gWidth, gHeight;
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

#ifdef VK_MODE
[[vk::binding(1, 0)]]
#endif
Texture2D<float4> gSource0 : register(t0); // native encoded colour / immutable packed proxy
#ifdef VK_MODE
[[vk::binding(2, 0)]]
#endif
Texture2D<float4> gSource1 : register(t1); // source depth / packed model answer
#ifdef VK_MODE
[[vk::binding(3, 0)]]
#endif
Texture2D<float4> gSource2 : register(t2); // source motion
#ifdef VK_MODE
[[vk::binding(7, 0)]]
#endif
SamplerState gLinearClamp : register(s0);

#ifdef SPATIAL_GUIDES
#ifdef VK_MODE
[[vk::binding(5, 0)]]
[[vk::image_format("r32f")]]
#endif
RWTexture2D<float> gTarget0 : register(u0); // R32F depth
#ifdef VK_MODE
[[vk::binding(6, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> gTarget1 : register(u1); // RGBA32F motion (xy used by NR)
#else
RWTexture2D<float2> gTarget1 : register(u1); // RG32F motion
#endif
#else
#ifdef VK_MODE
[[vk::binding(5, 0)]]
[[vk::image_format("rgba16f")]]
#endif
RWTexture2D<float4> gTarget0 : register(u0); // RGBA16F packed colour / unpacked proxy
#ifdef VK_MODE
[[vk::binding(6, 0)]]
[[vk::image_format("rgba16f")]]
#endif
RWTexture2D<float4> gTarget1 : register(u1); // RGBA16F unpacked model answer
#endif

#include "dlssnr_spatial_warp.hlsli"

float4 SamplePackedColour(float2 nativePixel, float2 workPixel)
{
    float2 nativeUv = nativePixel / gNativeWorkSize.xy;
    float2 footprint = PwFootprint(workPixel);
    float spread = 0.375 * saturate((max(footprint.x, footprint.y) - 1.0) / 0.3);
    if (spread <= 0.0) return gSource0.SampleLevel(gLinearClamp, nativeUv, 0);
    float2 delta = spread * footprint / gNativeWorkSize.xy;
    return 0.2 * (gSource0.SampleLevel(gLinearClamp, nativeUv, 0) +
                  gSource0.SampleLevel(gLinearClamp, nativeUv + float2(-delta.x, -delta.y), 0) +
                  gSource0.SampleLevel(gLinearClamp, nativeUv + float2( delta.x, -delta.y), 0) +
                  gSource0.SampleLevel(gLinearClamp, nativeUv + float2(-delta.x,  delta.y), 0) +
                  gSource0.SampleLevel(gLinearClamp, nativeUv + float2( delta.x,  delta.y), 0));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) return;
    float2 pixel = float2(id.xy) + 0.5;
#ifdef SPATIAL_GUIDES
    if (gMode != 101) return;
    float2 nativePixel = PwUnpack(pixel);
    float2 nativeUv = nativePixel / gNativeWorkSize.xy;
    int2 depthLo = int2(gDepthRect.xy), motionLo = int2(gMotionRect.xy);
    int2 depthHi = int2(gDepthRect.xy + gDepthRect.zw) - 1;
    int2 motionHi = int2(gMotionRect.xy + gMotionRect.zw) - 1;
    int2 depthPixel = clamp(int2(gDepthRect.xy + nativeUv * gDepthRect.zw), depthLo, depthHi);
    int2 motionPixel = clamp(int2(gMotionRect.xy + nativeUv * gMotionRect.zw), motionLo, motionHi);
    float depth = gSource1.Load(int3(depthPixel, 0)).x;
    float2 motionNative = gSource2.Load(int3(motionPixel, 0)).xy * gMotionScale;
    // Transform endpoints; the curve extends beyond frame edges for motion there.
    float2 motionPacked = PwPack(nativePixel + motionNative) - PwPack(nativePixel);
    gTarget0[id.xy] = depth;
#ifdef VK_MODE
    gTarget1[id.xy] = float4(motionPacked, 0.0, 0.0);
#else
    gTarget1[id.xy] = motionPacked;
#endif
#else
    if (gMode == 100)
    {
        float2 nativePixel = PwUnpack(pixel);
        gTarget0[id.xy] = SamplePackedColour(nativePixel, pixel);
    }
    else if (gMode == 102)
    {
        // The existing enlargement pass consumes this ordinary uniform grid.
        float2 nativePixel = pixel * gNativeWorkSize.xy / float2(gWidth, gHeight);
        float2 packedUv = PwPack(nativePixel) / gNativeWorkSize.zw;
        gTarget0[id.xy] = gSource0.SampleLevel(gLinearClamp, packedUv, 0);
        gTarget1[id.xy] = gSource1.SampleLevel(gLinearClamp, packedUv, 0);
    }
#endif
}
