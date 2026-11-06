RWByteAddressBuffer Saved : register(u0);
RWByteAddressBuffer Current : register(u1);
cbuffer MixSize : register(b0) { uint WordCount; uint RowStride; float Alpha; };

float DecodeE4M3(uint bits)
{
    uint exponent = (bits >> 3) & 15;
    uint mantissa = bits & 7;
    float value = exponent == 0 ? float(mantissa) * (1.0 / 512.0)
        : asfloat(((exponent + 120) << 23) | (mantissa << 20));
    if (exponent == 15 && mantissa == 7) value = asfloat(0x7fc00000);
    return (bits & 128) ? -value : value;
}

uint EncodeE4M3(float value)
{
    uint sign = (asuint(value) >> 24) & 128;
    if (isnan(value)) return sign | 127;
    float magnitude = min(abs(value), 448.0);
    if (magnitude < (1.0 / 64.0))
        return sign | uint(round(magnitude * 512.0));
    int exponent = int((asuint(magnitude) >> 23) & 255) - 127;
    float scale = asfloat(uint(127 + 3 - exponent) << 23);
    uint rounded = uint(round(magnitude * scale));
    return sign | (uint(exponent + 7) * 8 + rounded - 8);
}

[numthreads(256, 1, 1)]
void MixLatent(uint3 thread : SV_DispatchThreadID)
{
    uint index = thread.x + thread.y * RowStride;
    if (index >= WordCount) return;
    uint original = Saved.Load(index * 4);
    // Preserve even signed zero / NaN payloads for the identity control.
    if (Alpha == 0.0) { Current.Store(index * 4, original); return; }
    uint repeated = Current.Load(index * 4), output = 0;
    [unroll] for (uint lane = 0; lane < 4; ++lane)
    {
        float z = DecodeE4M3((original >> (8 * lane)) & 255);
        float z2 = DecodeE4M3((repeated >> (8 * lane)) & 255);
        float mixed = f16tof32(f32tof16(z + Alpha * (z2 - z)));
        output |= EncodeE4M3(mixed) << (8 * lane);
    }
    Current.Store(index * 4, output);
}
