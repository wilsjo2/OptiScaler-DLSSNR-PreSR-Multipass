// Do bulk copies and tensor-core work in the same CTA get in each other's way? Thread 0 of warp `copier` keeps bulk
// copies of `stage` bytes in flight (`copies` in total, two at a time); every warp runs `iterations` steps of 16
// register-only NVFP4 mma. Either part can be switched off (copies = 0 or iterations = 0).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
#define NR_INLINE static __forceinline__ __device__
#define WAIT_LIMIT (1 << 22)

struct MixParams { const u8* src; float* out; u32 footprint; u32 stage; u32 copies; u32 iterations; u32 copier; };

NR_INLINE u32 smem_addr(const void* p)
{
    u64 a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"((u64)p));
    return (u32)a;
}

extern "C" __global__ void nr_mix_bench(MixParams p)
{
    extern __shared__ __align__(512) u8 smem[];
    const u32 smem0 = smem_addr(smem);
    float sum = 0.0f;
    if (threadIdx.x == 0 && threadIdx.y == p.copier && p.copies) {
        for (u32 s = 0; s < 2u; ++s) asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem0 + 8u * s), "r"(1u) : "memory");
        u32 offset = (u32)(((u64)blockIdx.x * 2654435761ull) % (p.footprint / p.stage)) * p.stage;
        for (u32 step = 0; step < p.copies + 2u; ++step) {
            const u32 slot = step & 1u, bar = smem0 + 8u * slot, dst = smem0 + 64u + slot * p.stage;
            if (step >= 2u) {
                const u32 parity = ((step >> 1) - 1u) & 1u;
                for (int i = 0; i < WAIT_LIMIT; ++i) {
                    u32 ok;
                    asm volatile("{\n .reg .pred nr_p;\n mbarrier.try_wait.parity.shared::cta.b64 nr_p, [%1], %2;\n selp.u32 %0, 1, 0, nr_p;\n}"
                                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
                    if (ok) break;
                }
            }
            if (step < p.copies) {
                u64 state;
                asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(p.stage) : "memory");
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             :: "r"(dst), "l"((u64)(p.src + offset)), "r"(p.stage), "r"(bar) : "memory");
                asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
                offset += p.stage;
                if (offset + p.stage > p.footprint) offset = 0u;
            }
        }
    } else if (p.iterations && !(threadIdx.y == p.copier && p.copies)) {
        u32 a[4], b[2]; float acc[16][4];
        for (int i = 0; i < 4; ++i) a[i] = 0x22222222u ^ (threadIdx.x & 0x11111111u);
        for (int i = 0; i < 2; ++i) b[i] = 0x22222222u ^ (threadIdx.y & 0x11111111u);
        const u32 sfa = 0x38383838u, sfb = 0x30303030u; const u16 z = 0;
        for (int c = 0; c < 16; ++c) for (int k = 0; k < 4; ++k) acc[c][k] = 0.0f;
        for (u32 it = 0; it < p.iterations; ++it) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                float d0, d1, d2, d3;
                asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                             "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%16, %16}, {%15}, {%16, %16};"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                               "f"(acc[c][0]), "f"(acc[c][1]), "f"(acc[c][2]), "f"(acc[c][3]), "r"(sfa), "r"(sfb), "h"(z));
                acc[c][0] = d0; acc[c][1] = d1; acc[c][2] = d2; acc[c][3] = d3;
            }
        }
        for (int c = 0; c < 16; ++c) sum += acc[c][0];
    }
    if (blockIdx.x == 0 && threadIdx.x == 1 && threadIdx.y == 0) p.out[0] = sum;
}
