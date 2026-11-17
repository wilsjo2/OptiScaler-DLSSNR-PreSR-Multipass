// What does each part of the GEMM inner loop cost? The same 16 NVFP4 mma per warp and step as ffn_v2.cu, with the
// surroundings switched on one at a time. MODE bits: 1 = operands loaded from shared memory every step (else held in
// registers), 2 = __syncthreads() every step, 4 = one mbarrier try_wait every step, 8 = stage slot varies per step.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };
struct BenchParams { float* out; u32 iterations; u32 seed; };
#define NR_INLINE static __forceinline__ __device__

NR_INLINE void mma4(float* c, const V4& a, u32 b0, u32 b1, u32 sfa, u32 sfb)
{
    float d0, d1, d2, d3;
    const u16 z = 0;
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%16, %16}, {%15}, {%16, %16};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
                   "r"(sfa), "r"(sfb), "h"(z));
    c[0] = d0; c[1] = d1; c[2] = d2; c[3] = d3;
}
NR_INLINE u32 smem_addr(const void* p)
{
    u64 a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"((u64)p));
    return (u32)a;
}

#define STAGE 9216u
#define BODY(MODE)                                                                                                    \
    extern __shared__ __align__(512) u8 smem[];                                                                       \
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.y;                                                            \
    const u32 smem0 = smem_addr(smem);                                                                                \
    if ((threadIdx.x | threadIdx.y) == 0) {                                                                           \
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem0), "r"(1u) : "memory");                           \
        u64 state;                                                                                                    \
        asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(smem0) : "memory");                \
    }                                                                                                                 \
    for (u32 i = 64u + threadIdx.y * blockDim.x + threadIdx.x; i < 64u + 4u * STAGE; i += blockDim.x * blockDim.y) smem[i] = (u8)(0x22u + (i & 1u)); \
    __syncthreads();                                                                                                  \
    float acc[2][8][4];                                                                                               \
    for (u32 i = 0; i < 2; ++i) for (u32 j = 0; j < 8; ++j) for (u32 k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;           \
    V4 a0 = {0x22222222u ^ p.seed, 0x22222222u, 0x33333333u, 0x22222222u}, a1 = a0, n0 = a0, n1 = a0, n2 = a0, n3 = a0, s0 = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u}, s1 = s0; \
    u32 sa0 = 0x30303030u, sa1 = 0x30303030u;                                                                         \
    for (u32 it = 0; it < p.iterations; ++it) {                                                                       \
        if ((MODE) & 4) {                                                                                             \
            u32 ok;                                                                                                   \
            asm volatile("{\n .reg .pred nr_p;\n mbarrier.try_wait.parity.shared::cta.b64 nr_p, [%1], %2;\n selp.u32 %0, 1, 0, nr_p;\n}" : "=r"(ok) : "r"(smem0), "r"(0u) : "memory"); \
            if (!ok) break;                                                                                           \
        }                                                                                                             \
        if ((MODE) & 1) {                                                                                             \
            const u8* base = smem + 64u + (((MODE) & 8) ? (it & 3u) * STAGE : 0u);                                     \
            const u8* units = base + (warp >> 1) * 1152u;                                                             \
            const u8* block = base + 4608u + (warp & 1u) * 2304u;                                                     \
            a0 = *(const V4*)(units + lane * 16u); a1 = *(const V4*)(units + 576u + lane * 16u);                       \
            sa0 = *(const u32*)(units + 512u + (lane >> 2) * 8u + (lane & 1u) * 4u);                                   \
            sa1 = *(const u32*)(units + 576u + 512u + (lane >> 2) * 8u + (lane & 1u) * 4u);                            \
            n0 = *(const V4*)(block + lane * 16u); n1 = *(const V4*)(block + 512u + lane * 16u);                       \
            n2 = *(const V4*)(block + 1024u + lane * 16u); n3 = *(const V4*)(block + 1536u + lane * 16u);              \
            s0 = *(const V4*)(block + 2048u + (lane >> 2) * 32u); s1 = *(const V4*)(block + 2064u + (lane >> 2) * 32u); \
        }                                                                                                             \
        mma4(acc[0][0], a0, n0.x, n0.y, sa0, s0.x); mma4(acc[1][0], a1, n0.x, n0.y, sa1, s0.x);                       \
        mma4(acc[0][1], a0, n0.z, n0.w, sa0, s0.y); mma4(acc[1][1], a1, n0.z, n0.w, sa1, s0.y);                       \
        mma4(acc[0][2], a0, n1.x, n1.y, sa0, s0.z); mma4(acc[1][2], a1, n1.x, n1.y, sa1, s0.z);                       \
        mma4(acc[0][3], a0, n1.z, n1.w, sa0, s0.w); mma4(acc[1][3], a1, n1.z, n1.w, sa1, s0.w);                       \
        mma4(acc[0][4], a0, n2.x, n2.y, sa0, s1.x); mma4(acc[1][4], a1, n2.x, n2.y, sa1, s1.x);                       \
        mma4(acc[0][5], a0, n2.z, n2.w, sa0, s1.y); mma4(acc[1][5], a1, n2.z, n2.w, sa1, s1.y);                       \
        mma4(acc[0][6], a0, n3.x, n3.y, sa0, s1.z); mma4(acc[1][6], a1, n3.x, n3.y, sa1, s1.z);                       \
        mma4(acc[0][7], a0, n3.z, n3.w, sa0, s1.w); mma4(acc[1][7], a1, n3.z, n3.w, sa1, s1.w);                       \
        if ((MODE) & 2) __syncthreads();                                                                              \
    }                                                                                                                 \
    float sum = 0.0f;                                                                                                 \
    for (u32 i = 0; i < 2; ++i) for (u32 j = 0; j < 8; ++j) sum += acc[i][j][0] + acc[i][j][3];                        \
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) p.out[0] = sum;

extern "C" __global__ void nr_loop_0(BenchParams p) { BODY(0) }
extern "C" __global__ void nr_loop_1(BenchParams p) { BODY(1) }
extern "C" __global__ void nr_loop_9(BenchParams p) { BODY(9) }
extern "C" __global__ void nr_loop_2(BenchParams p) { BODY(2) }
extern "C" __global__ void nr_loop_4(BenchParams p) { BODY(4) }
extern "C" __global__ void nr_loop_11(BenchParams p) { BODY(11) }
extern "C" __global__ void nr_loop_15(BenchParams p) { BODY(15) }
