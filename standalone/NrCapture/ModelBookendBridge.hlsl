// Probe-only, approximate RGB-bookend bridge. The learned pre/post Swin bodies
// remain in the native model. This updates an already FP16-rounded input lift;
// it cannot recover the original accumulator. Native projection MMA also uses
// FP16 accumulation, whereas this bounded prototype uses scalar FP32 products.
// Contract: reset/no history, unchanged noise/conditioning, intensity 1, SDR.
// Source and anchor buffers must not alias: padded threads read reflected source.
ByteAddressBuffer OriginalPreLift : register(t0); // Padded NHWC, 32 half channels.
ByteAddressBuffer PostWide : register(t1);        // Padded NHWC, 32 half channels.
ByteAddressBuffer OriginalRGBA : register(t2);    // Valid image, packed RGBA16F.
ByteAddressBuffer BridgeWeights : register(t3);   // P[3,32] then C[32,3], float32.
RWByteAddressBuffer NextPreLift : register(u0);   // Padded NHWC, 32 half channels.
RWByteAddressBuffer FirstAnchor : register(u1);   // Valid image, packed RGBA16F.

cbuffer BridgeParams : register(b0)
{
    uint Width, Height, PaddedWidth, PaddedHeight;
    uint Mode, Reserved0, Reserved1, Reserved2; // 0: colour delta; 1: affine diagnostic.
};

float3 Half3(float3 value)
{
    return f16tof32(f32tof16(value));
}

uint PackHalf2(float a, float b)
{
    return (f32tof16(a) & 0xffffu) | (f32tof16(b) << 16);
}

uint MirrorIndex(uint index, uint extent)
{
    // Native/reference padding reflects once, then clamps; it is not periodic.
    const int reflected = index < extent ? int(index) : 2 * int(extent) - int(index) - 2;
    return uint(clamp(reflected, 0, int(extent) - 1));
}

float3 CentredColour(float3 value)
{
    return Half3(Half3(Half3(value) - 0.5) * 0.125);
}

float3 ProjectionWeights(uint channel)
{
    return float3(asfloat(BridgeWeights.Load(channel * 4)),
                  asfloat(BridgeWeights.Load((32 + channel) * 4)),
                  asfloat(BridgeWeights.Load((64 + channel) * 4)));
}

float3 ColourWeights(uint channel)
{
    return asfloat(BridgeWeights.Load3(384 + channel * 12));
}

[numthreads(8, 8, 1)]
void BridgeBookends(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= PaddedWidth || id.y >= PaddedHeight) return;

    const uint2 pixel = uint2(MirrorIndex(id.x, Width), MirrorIndex(id.y, Height));
    const uint validIndex = pixel.y * Width + pixel.x;
    const uint postOffset = (pixel.y * PaddedWidth + pixel.x) * 64;
    const uint liftOffset = (id.y * PaddedWidth + id.x) * 64;
    const uint2 rgba = OriginalRGBA.Load2(validIndex * 8);
    const float3 original = float3(f16tof32(rgba.x & 0xffffu),
                                   f16tof32(rgba.x >> 16), f16tof32(rgba.y & 0xffffu));

    precise float3 residual = 0;
    [unroll]
    for (uint pair = 0; pair < 16; ++pair)
    {
        const uint packed = PostWide.Load(postOffset + pair * 4);
        residual += ProjectionWeights(pair * 2) * f16tof32(packed & 0xffffu);
        residual += ProjectionWeights(pair * 2 + 1) * f16tof32(packed >> 16);
    }
    const float3 anchor = Half3(saturate(original + residual * 0.25));
    const float3 colourDelta = Mode == 1 ? residual * (1.0 / 32.0)
                                         : CentredColour(anchor) - CentredColour(original);

    [unroll]
    for (uint liftPair = 0; liftPair < 16; ++liftPair)
    {
        // Read the original lift at the PADDED coordinate: its noise features
        // belong to this position even when the image colour is reflected.
        const uint packed = OriginalPreLift.Load(liftOffset + liftPair * 4);
        precise float low = f16tof32(packed & 0xffffu) + dot(ColourWeights(liftPair * 2), colourDelta);
        precise float high = f16tof32(packed >> 16) + dot(ColourWeights(liftPair * 2 + 1), colourDelta);
        NextPreLift.Store(liftOffset + liftPair * 4, PackHalf2(low, high));
    }

    if (id.x < Width && id.y < Height)
    {
        // Preserve the first result as the second final residual anchor. Keeping
        // this small carry is necessary even though the next input is latent32.
        FirstAnchor.Store2(validIndex * 8, uint2(PackHalf2(anchor.r, anchor.g),
                                               PackHalf2(anchor.b, 1.0)));
    }
}
