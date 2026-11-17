// Throughput of ordinary global loads (through the L1 cache) with no arithmetic, alone and next to a bulk-copy
// stream: is it a second path into the SM, independent of the bulk-copy engine?
// Every warp reads `lines` consecutive 512-byte lines per step (each lane one 16-byte vector per line), as the GEMM
// reads weights. mode bit 1: thread 0 also keeps bulk copies of `stage` bytes in flight (depth 2).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };
#define NR_INLINE static __forceinline__ __device__
#define WAIT_LIMIT (1 << 22)

struct LoadParams { const u8* src; u32* out; u32 footprint; u32 lines; u32 steps; u32 mode; u32 stage; };

NR_INLINE u32 smem_addr(const void* p)
{
    u64 a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"((u64)p));
    return (u32)a;
}

extern "C" __global__ void nr_load_bench(LoadParams p)
{
    extern __shared__ __align__(512) u8 smem[];
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.y, warps = blockDim.y;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const u32 smem0 = smem_addr(smem);
    const bool copies = (p.mode & 1u) != 0u;
    if (leader && copies)
        for (u32 s = 0; s < 2u; ++s) asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem0 + 8u * s), "r"(1u) : "memory");
    const u32 chunk = p.lines * 512u, per_step = chunk * warps;
    u32 offset = (u32)(((u64)blockIdx.x * 2654435761ull) % (p.footprint / per_step)) * per_step;
    u32 copy_offset = (u32)(((u64)(blockIdx.x + 7u) * 40503ull) % (p.footprint / p.stage)) * p.stage;
    u32 sum = 0u;
    for (u32 step = 0; step < p.steps; ++step) {
        const u8* base = p.src + offset + warp * chunk + lane * 16u;
#pragma unroll 1
        for (u32 line = 0; line < p.lines; line += 4u) {
            const V4 a = *(const V4*)(base + line * 512u), b = *(const V4*)(base + line * 512u + 512u);
            const V4 c = *(const V4*)(base + line * 512u + 1024u), d = *(const V4*)(base + line * 512u + 1536u);
            sum += a.x ^ b.y ^ c.z ^ d.w;
        }
        offset += per_step;
        if (offset + per_step > p.footprint) offset = 0u;
        if (leader && copies) {
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
            u64 state;
            asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(p.stage) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(dst), "l"((u64)(p.src + copy_offset)), "r"(p.stage), "r"(bar) : "memory");
            asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
            copy_offset += p.stage;
            if (copy_offset + p.stage > p.footprint) copy_offset = 0u;
        }
    }
    if (leader && copies) {                                               // let the last two copies land before the CTA exits
        for (u32 slot = 0; slot < 2u; ++slot) {
            const u32 last = p.steps - 1u - ((p.steps - 1u - slot) & 1u);
            const u32 parity = (last >> 1) & 1u;
            for (int i = 0; i < WAIT_LIMIT; ++i) {
                u32 ok;
                asm volatile("{\n .reg .pred nr_p;\n mbarrier.try_wait.parity.shared::cta.b64 nr_p, [%1], %2;\n selp.u32 %0, 1, 0, nr_p;\n}"
                             : "=r"(ok) : "r"(smem0 + 8u * slot), "r"(parity) : "memory");
                if (ok) break;
            }
        }
    }
    if (blockIdx.x == 0 && lane == 0 && warp == 0) p.out[0] = sum;
}
