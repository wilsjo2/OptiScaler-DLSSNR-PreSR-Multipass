// Does a second warp that reads the same global memory as another warp of its SM pay for it again?
// Every warp reads `lines` consecutive 512-byte lines per step (each lane one 16-byte vector per line), as the GEMM
// kernels read weights. Warps are grouped `share` to a stream: the warps of a group read the same addresses, warp i of
// the group `lag` steps behind warp i - 1 (lag 0: at the same time). The stream wraps inside `footprint` bytes per
// group, far more than L1 holds, so a line is in L1 only if a warp of this SM read it a short while ago.
// mode bit 0: a barrier after every step (the warps of the CTA cannot drift apart).
// Dynamic shared memory is only reserved by the launch (it sets how much of the SM's L1 / shared memory block is left
// to L1); the kernel never touches it. No waits, no flags.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };

struct ShareParams { const u8* src; u32* out; u32 footprint; u32 lines; u32 steps; u32 mode; u32 share; u32 lag; };

extern "C" __global__ void nr_share_bench(ShareParams p)
{
    const u32 lane = threadIdx.x & 31u, warp = threadIdx.y;
    const u32 group = warp / p.share, member = warp - group * p.share, groups = (blockDim.y + p.share - 1u) / p.share;
    const u32 chunk = p.lines * 512u, wrap = p.footprint / chunk;               // steps before a group's stream repeats
    // each (SM, group) has its own region; CTAs that share an SM read different regions (blockIdx)
    const u8* const region = p.src + (u64)((blockIdx.x * groups + group) % 64u) * p.footprint;
    const u32 lead = (p.share - 1u - member) * p.lag;                           // member 0 is the furthest ahead
    u32 sum = 0u;
    for (u32 step = 0; step < p.steps; ++step) {
        const u8* base = region + (u64)((step + lead) % wrap) * chunk + lane * 16u;
#pragma unroll 1
        for (u32 line = 0; line < p.lines; line += 4u) {
            const V4 a = *(const V4*)(base + line * 512u), b = *(const V4*)(base + line * 512u + 512u);
            const V4 c = *(const V4*)(base + line * 512u + 1024u), d = *(const V4*)(base + line * 512u + 1536u);
            sum += a.x ^ b.y ^ c.z ^ d.w;
        }
        if (p.mode & 1u) __syncthreads();
    }
    if (blockIdx.x == 0 && lane == 0 && warp == 0) p.out[0] = sum;
}
