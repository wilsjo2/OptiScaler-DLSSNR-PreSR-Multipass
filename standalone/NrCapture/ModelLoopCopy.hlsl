RWByteAddressBuffer Source : register(u0);
RWByteAddressBuffer Destination : register(u1);
cbuffer CopySize : register(b0) { uint VectorCount; uint RowStride; };
[numthreads(256, 1, 1)]
void CopyLatent(uint3 thread : SV_DispatchThreadID)
{
    const uint index = thread.x + thread.y * RowStride;
    if (index < VectorCount)
        Destination.Store4(index * 16, Source.Load4(index * 16));
}
