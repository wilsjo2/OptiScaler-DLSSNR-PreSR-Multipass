// NrPassBorrow handoff copy: 16 bytes per thread between two root UAVs (raw GPU virtual addresses inside the NR
// runtime's buffer pool, or this module's snapshot buffers). Same as standalone/NrCapture/ModelLoopCopy.hlsl, plus an
// embedded root signature (root parameters 0 = u0, 1 = u1, 2 = two constants) so no serializer is needed.
// Rebuild NrPassBorrowCopy.h from OptiScaler/dlssnr (Windows SDK 10.0.26100 fxc):
//   fxc /nologo /T cs_5_0 /E CopyHandoff /O3 /Vn NrPassBorrowCopy /Fh NrPassBorrowCopy.h NrPassBorrowCopy.hlsl
// then prepend the two comment lines at the top of NrPassBorrowCopy.h.
RWByteAddressBuffer Source : register(u0);
RWByteAddressBuffer Destination : register(u1);
cbuffer CopySize : register(b0) { uint VectorCount; uint RowStride; };
[RootSignature("UAV(u0), UAV(u1), RootConstants(num32BitConstants = 2, b0)")]
[numthreads(256, 1, 1)]
void CopyHandoff(uint3 thread : SV_DispatchThreadID)
{
    const uint index = thread.x + thread.y * RowStride;
    if (index < VectorCount)
        Destination.Store4(index * 16, Source.Load4(index * 16));
}
