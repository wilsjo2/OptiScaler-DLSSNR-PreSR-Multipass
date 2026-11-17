// Does a global load issued early hide its latency behind the work that follows it?
// One CTA per SM; every warp has its own stream (no line is ever in L1). A step: issue the loads of step + LEAD (LINES
// 512-byte lines, each lane one 16-byte vector per line), then consume the values of this step (loaded LEAD steps ago)
// and run `work` dependent float operations. If a warp only stalls when it needs a value that has not arrived, a step
// takes max(work, latency / LEAD-ish); if it stalls at the load itself, work + latency whatever LEAD is.
// MODE 1: the loads are cp.async into shared memory (ring of LEAD + 1 slots per warp), read back with ld.shared.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };
#ifndef LEAD
#define LEAD 0
#endif
#ifndef LINES
#define LINES 1
#endif
#ifndef MODE
#define MODE 0
#endif
#define NR_INLINE __forceinline__ __device__

struct LeadParams { const u8* src; u32* out; u32 footprint; u32 steps; u32 work; u32 pad; };

NR_INLINE V4 ld16(const u8* q)
{
    V4 v;
#if MODE == 2                                                   // MODE 2: ld.volatile (may the compiler still move it?)
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"((u64)q) : "memory");
#else
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"((u64)q));
#endif
    return v;
}
NR_INLINE void cp16(u8* dst, const u8* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((u32)__cvta_generic_to_shared(dst)), "l"((u64)src) : "memory");
}
NR_INLINE void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> NR_INLINE void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

extern "C" __global__ void nr_lead_bench(LeadParams p)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.y;
    const u32 chunk = LINES * 512u, wrap = p.footprint / chunk;
    const u8* const region = p.src + (u64)((blockIdx.x * blockDim.y + warp) % 256u) * p.footprint + lane * 16u;
    float x = 1.0f;
    u32 sum = 0u;
#if MODE == 1
    extern __shared__ __align__(16) u8 sm[];
    u8* const mine = sm + warp * ((LEAD + 1u) * chunk) + lane * 16u;
#pragma unroll
    for (u32 s = 0; s < LEAD; ++s) {
#pragma unroll
        for (u32 l = 0; l < LINES; ++l) cp16(mine + s * chunk + l * 512u, region + (u64)(s % wrap) * chunk + l * 512u);
        cp_commit();
    }
    for (u32 step = 0; step < p.steps; ++step) {
        const u32 slot = (step + LEAD) % (LEAD + 1u);
#pragma unroll
        for (u32 l = 0; l < LINES; ++l) cp16(mine + slot * chunk + l * 512u, region + (u64)((step + LEAD) % wrap) * chunk + l * 512u);
        cp_commit();
        cp_wait<LEAD>();                                        // the group of this step has landed
        u32 v = 0u;
        const u32 now = step % (LEAD + 1u);
#pragma unroll
        for (u32 l = 0; l < LINES; ++l) {
            const V4 a = *(const V4*)(mine + now * chunk + l * 512u);
            v ^= a.x ^ a.y ^ a.z ^ a.w;
        }
        x += (float)(v & 255u);
        for (u32 k = 0; k < p.work; ++k) x = x * 0.999f + 1.0f;
        sum += (u32)x;
    }
#else
    V4 ring[LEAD + 1][LINES];
#pragma unroll
    for (u32 s = 0; s < LEAD; ++s)
#pragma unroll
        for (u32 l = 0; l < LINES; ++l) ring[s][l] = ld16(region + (u64)(s % wrap) * chunk + l * 512u);
    for (u32 step = 0; step < p.steps; ++step) {
#pragma unroll
        for (u32 l = 0; l < LINES; ++l) ring[LEAD][l] = ld16(region + (u64)((step + LEAD) % wrap) * chunk + l * 512u);
        u32 v = 0u;
#pragma unroll
        for (u32 l = 0; l < LINES; ++l) v ^= ring[0][l].x ^ ring[0][l].y ^ ring[0][l].z ^ ring[0][l].w;
        x += (float)(v & 255u);
        for (u32 k = 0; k < p.work; ++k) x = x * 0.999f + 1.0f;
        sum += (u32)x;
#pragma unroll
        for (u32 s = 0; s < LEAD; ++s)
#pragma unroll
            for (u32 l = 0; l < LINES; ++l) ring[s][l] = ring[s + 1][l];
    }
#endif
    if (blockIdx.x == 0 && lane == 0 && warp == 0) p.out[0] = sum + (u32)x;
}
