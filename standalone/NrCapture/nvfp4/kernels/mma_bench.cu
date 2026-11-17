// Raw tensor-core throughput: 16 independent accumulator chains per warp, operands held in registers.
typedef unsigned int u32;
typedef unsigned short u16;
struct BenchParams { float* out; u32 iterations; u32 seed; };

#define CHAINS 16

extern "C" __global__ void nr_bench_fp8(BenchParams p)
{
    u32 a[4], b[2], acc[CHAINS][2];
    for (int i = 0; i < 4; ++i) a[i] = 0x38383838u ^ (p.seed * (i + 1) & 0x03030303u);
    for (int i = 0; i < 2; ++i) b[i] = 0x30303030u ^ (threadIdx.x & 0x01010101u);
    for (int c = 0; c < CHAINS; ++c) acc[c][0] = acc[c][1] = 0u;
    for (u32 it = 0; it < p.iterations; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) {
            u32 d0, d1;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
                         : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(acc[c][0]), "r"(acc[c][1]));
            acc[c][0] = d0; acc[c][1] = d1;
        }
    }
    u32 sum = 0u;
    for (int c = 0; c < CHAINS; ++c) sum ^= acc[c][0] ^ acc[c][1];
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) p.out[0] = (float)sum;
}

extern "C" __global__ void nr_bench_fp4(BenchParams p)
{
    u32 a[4], b[2]; float acc[CHAINS][4];
    for (int i = 0; i < 4; ++i) a[i] = 0x22222222u ^ (p.seed * (i + 1) & 0x11111111u);
    for (int i = 0; i < 2; ++i) b[i] = 0x22222222u ^ (threadIdx.x & 0x11111111u);
    const u32 sfa = 0x38383838u, sfb = 0x30303030u; const u16 z = 0;
    for (int c = 0; c < CHAINS; ++c) for (int k = 0; k < 4; ++k) acc[c][k] = 0.0f;
    for (u32 it = 0; it < p.iterations; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) {
            float d0, d1, d2, d3;
            asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                         "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%16, %16}, {%15}, {%16, %16};"
                         : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                           "f"(acc[c][0]), "f"(acc[c][1]), "f"(acc[c][2]), "f"(acc[c][3]), "r"(sfa), "r"(sfb), "h"(z));
            acc[c][0] = d0; acc[c][1] = d1; acc[c][2] = d2; acc[c][3] = d3;
        }
    }
    float sum = 0.0f;
    for (int c = 0; c < CHAINS; ++c) sum += acc[c][0] + acc[c][3];
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) p.out[0] = sum;
}

// f8f6f4 form: FP8 activations (e4m3) times FP4 weights (e2m1), k32, no block scale
extern "C" __global__ void nr_bench_fp8x4(BenchParams p)
{
    u32 a[4], b[2]; float acc[CHAINS][4];
    for (int i = 0; i < 4; ++i) a[i] = 0x38383838u ^ (p.seed * (i + 1) & 0x03030303u);
    for (int i = 0; i < 2; ++i) b[i] = 0x08080808u ^ (threadIdx.x & 0x04040404u);
    for (int c = 0; c < CHAINS; ++c) for (int k = 0; k < 4; ++k) acc[c][k] = 0.0f;
    for (u32 it = 0; it < p.iterations; ++it) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) {
            float d0, d1, d2, d3;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
                         "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                         : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                           "f"(acc[c][0]), "f"(acc[c][1]), "f"(acc[c][2]), "f"(acc[c][3]));
            acc[c][0] = d0; acc[c][1] = d1; acc[c][2] = d2; acc[c][3] = d3;
        }
    }
    float sum = 0.0f;
    for (int c = 0; c < CHAINS; ++c) sum += acc[c][0] + acc[c][3];
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) p.out[0] = sum;
}
