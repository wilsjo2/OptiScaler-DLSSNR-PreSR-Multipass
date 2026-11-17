// Conversion probe: which packed float conversions exist on sm_120a and how they pack their two results.
struct CvtParams { const float* in; unsigned* out; int count; };

extern "C" __global__ void nr_cvt_probe(CvtParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count) return;
    const float a = p.in[2 * i], b = p.in[2 * i + 1];
    unsigned short e4 = 0, h2e = 0; unsigned e2 = 0, f16pair = 0, back = 0;
    // two f32 -> two e4m3 in 16 bits
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4) : "f"(a), "f"(b));
    // two f32 -> two e2m1 nibbles in one byte
    asm volatile("{\n .reg .b8 nr_t;\n cvt.rn.satfinite.e2m1x2.f32 nr_t, %1, %2;\n cvt.u32.u8 %0, nr_t;\n}"
                 : "=r"(e2) : "f"(a), "f"(b));
    // two f32 -> f16x2, then f16x2 -> e4m3x2, and e4m3x2 -> f16x2
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(f16pair) : "f"(a), "f"(b));
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h2e) : "r"(f16pair));
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(back) : "h"(e4));
    p.out[4 * i + 0] = e4 | (e2 << 16);
    p.out[4 * i + 1] = f16pair;
    p.out[4 * i + 2] = h2e;
    p.out[4 * i + 3] = back;
}
