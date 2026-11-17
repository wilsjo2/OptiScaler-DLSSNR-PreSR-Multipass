// How fast can data reach shared memory? One thread per CTA runs the same bulk-copy pipeline as ffn_v2.cu with no
// arithmetic at all: `depth` copies of `stage` bytes in flight, `steps` copies in total, each CTA reading its own
// stretch of a source of `footprint` bytes (a small footprint stays in the L2 cache, as the weights do).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;
#define NR_INLINE static __forceinline__ __device__
#define WAIT_LIMIT (1 << 22)

struct CopyParams { const u8* src; u32* out; u32 footprint; u32 stage; u32 steps; u32 depth; };

NR_INLINE u32 smem_addr(const void* p)
{
    u64 a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"((u64)p));
    return (u32)a;
}

extern "C" __global__ void nr_copy_bench(CopyParams p)
{
    extern __shared__ __align__(512) u8 smem[];
    if (threadIdx.x | threadIdx.y) return;
    const u32 smem0 = smem_addr(smem);
    for (u32 s = 0; s < p.depth; ++s) asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem0 + 8u * s), "r"(1u) : "memory");
    u32 offset = (u32)(((u64)blockIdx.x * 2654435761ull) % (p.footprint / p.stage)) * p.stage;
    u32 sum = 0u;
    for (u32 step = 0; step < p.steps + p.depth; ++step) {
        const u32 slot = step % p.depth, bar = smem0 + 8u * slot, dst = smem0 + 64u + slot * p.stage;
        if (step >= p.depth) {                                            // the copy issued `depth` steps ago
            const u32 parity = ((step / p.depth) - 1u) & 1u;
            for (int i = 0; i < WAIT_LIMIT; ++i) {
                u32 ok;
                asm volatile("{\n .reg .pred nr_p;\n mbarrier.try_wait.parity.shared::cta.b64 nr_p, [%1], %2;\n selp.u32 %0, 1, 0, nr_p;\n}"
                             : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
                if (ok) break;
            }
            sum += smem[64u + slot * p.stage];
        }
        if (step < p.steps) {
            u64 state;
            asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(p.stage) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst), "l"((u64)(p.src + offset)), "r"(p.stage), "r"(bar) : "memory");
            asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
            offset += p.stage;
            if (offset + p.stage > p.footprint) offset = 0u;
        }
    }
    if (blockIdx.x == 0) p.out[0] = sum;
}
