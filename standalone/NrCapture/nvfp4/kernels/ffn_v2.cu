// NVFP4 v2 feed-forward kernels for the DLSS-NR vision transformer. Target sm_120a, built with NVRTC
// (--fmad=false: every multiply-add below that is meant to be fused is written as fmaf).
//
//   nr_pack_x      : the runtime's FP8 activations x [tokens,1024]  -> NVFP4 A operand ("AQ")
//   nr_ffn_expand  : AQ(x) * W1 (NVFP4) -> cubic SiLU -> NVFP4 A operand of the hidden layer
//   nr_ffn_contract: AQ(hidden) * W2 (NVFP4) + gain * x -> the runtime's FP8 output, flags published
//
// Layouts (all multi-byte loads are 16-byte aligned):
//   runtime FP8 activations  [group16][kblock32][lane = 4*(token&7) + ((c>>1)&3)][16 B]   (see kernel-spec/ffn-fp8-spec.md)
//   AQ  [tile128][ksub64][group 0..7] units of 576 B:
//         512 B  [lane = 4*(token&7) + ((k&31)>>3)][4 words: ((k>>5)&1)*2 + ((token>>3)&1)], nibble k&7
//          64 B  [token&7][(token>>3)&1] words, byte j scales k = 16j..16j+15 of the k64 step
//   WQ  [out128][ksub64][half 0..1] blocks of 2304 B:
//        2048 B  [n16 0..3][lane = 4*(n&7) + ((k&31)>>3)][4 words: ((n>>3)&1)*2 + ((k>>5)&1)], nibble k&7
//         256 B  [n&7][ntile 0..7] words, byte j as above
//
// A GEMM CTA covers gt 16-token groups (2, 4 or 8) and nh 64-output halves (1 or 2); one warp is 32 tokens x 64
// outputs = 16 mma per k64 step. Launch with block (32, gt/2 * nh) and dynamic shared memory NR_SMEM(gt, nh).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };

#define NR_INLINE static __forceinline__ __device__
#define AQ_UNIT 576u
#define WQ_HALF 2304u
// A bulk copy costs about 100 ns before its first byte, so a copy must be big to use the SM's data port well: 4608
// bytes arrive at 29 GB/s, 9216 or more at the port's 43 GB/s. A pipeline stage is therefore KSTEPS k64 steps, fetched
// with one copy of activations and one of weights; STAGES of them are in flight. KSTEPS > 1 needs the 8-group,
// 2-half CTA shape, where consecutive k64 steps are contiguous in both buffers.
#ifndef STAGES
#define STAGES 2u
#endif
#ifndef KSTEPS
#define KSTEPS 2u
#endif
#define BAR_BYTES 64u
// Every wait of a thread for another kernel's flags draws on one budget of nominal sleep time. A kernel whose producer
// never shows up therefore ends with a wrong result instead of holding the GPU until Windows resets the driver (after
// two seconds). A polling step really takes 0.2 to 0.5 us, several times its nominal 64 ns (test_timeout.py). Waits
// that are in order last microseconds.
#ifndef WAIT_BUDGET_NS
#define WAIT_BUDGET_NS (1 << 23)                    // measured: 30 to 70 ms before a kernel gives up
#endif
// 1: only lane 0 of each warp tests the copy barrier, the other lanes wait for it at a __syncwarp.
#ifndef WARP_WAIT
#define WARP_WAIT 0
#endif
#ifndef WAIT_NAP
#define WAIT_NAP 1
#endif
#ifndef DBG_SKIP_EPILOGUE
#define DBG_SKIP_EPILOGUE 0
#endif

// ---------------------------------------------------------------------------------------------------------------
NR_INLINE u32 smem_addr(const void* p)
{
    u64 a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"((u64)p));
    return (u32)a;
}
NR_INLINE void bar_init(u32 bar, u32 count) { asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory"); }
NR_INLINE void bar_expect(u32 bar, u32 bytes) { asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
NR_INLINE void bar_arrive(u32 bar)
{
    u64 state;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(bar) : "memory");
}
NR_INLINE void bulk_copy(u32 dst, const void* src, u32 bytes, u32 bar)
{
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"((u64)src), "r"(bytes), "r"(bar) : "memory");
}
NR_INLINE u32 bar_try(u32 bar, u32 parity)
{
    u32 ok;
    asm volatile("{\n .reg .pred nr_p;\n mbarrier.try_wait.parity.shared::cta.b64 nr_p, [%1], %2;\n selp.u32 %0, 1, 0, nr_p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}
// Waiting for a stage's copies. This wait must outlast everything that can delay the leader before it issues them
// (its flag waits, bounded by WAIT_BUDGET_NS): a thread that gave up here would let the leader reuse a ring slot
// while a copy into it is still in flight. The bound is only there so that a copy that never lands cannot hang the
// GPU for ever; it is far beyond the flag budget and is not reached in practice.
#define BAR_WAIT_LIMIT (1 << 23)
NR_INLINE void bar_wait(u32 bar, u32 parity)
{
    for (int i = 0; i < BAR_WAIT_LIMIT; ++i) {
        if (bar_try(bar, parity)) return;
#if WAIT_NAP
        asm volatile("nanosleep.u32 32;" ::: "memory");      // leave the SM to warps that have work
#endif
    }
}

// Cross-launch ordering without barriers, the way the runtime's own kernels do it: a producer publishes an int with
// a release store once a region is completely written, a consumer polls it with an acquire load and a short sleep.
// Every wait is bounded (WAIT_BUDGET_NS), so a missing producer costs a wrong frame, never a hung GPU.
NR_INLINE int flag_load(const int* flag)
{
    int value;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"((u64)flag) : "memory");
    return value;
}
NR_INLINE void flag_publish(int* flag, int value) { asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"((u64)flag), "r"(value) : "memory"); }
NR_INLINE void nap() { asm volatile("nanosleep.u32 64;" ::: "memory"); }
NR_INLINE void wait_equal(const int* flag, int wanted, int& budget)
{
    for (; budget > 0 && flag_load(flag) != wanted; budget -= 64) nap();
}
NR_INLINE void wait_not_negative(const int* flag, int& budget)
{
    for (; budget > 0 && flag_load(flag) < 0; budget -= 64) nap();
}
// A flag load takes about 0.3 us even when the flag is set, and a thread's loads queue behind each other only if one
// depends on another. So flags are read two per load (neighbours, 8-byte aligned), several loads at a time, and a
// kernel that polls once per pipeline stage reads the next stage's flags a stage early (see issue_stage).
NR_INLINE u64 flag_pair_load(const int* flags)
{
    u64 value;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"((u64)flags) : "memory");
    return value;
}
NR_INLINE u64 flag_pair(int wanted) { return ((u64)(u32)wanted << 32) | (u64)(u32)wanted; }
// Eight consecutive flags, 8-byte aligned.
NR_INLINE void wait_equal8(const int* flags, int wanted, int& budget)
{
    const u64 both = flag_pair(wanted);
    for (;;) {
        const u64 a = flag_pair_load(flags), b = flag_pair_load(flags + 2), c = flag_pair_load(flags + 4), d = flag_pair_load(flags + 6);
        if ((a == both && b == both && c == both && d == both) || budget <= 0) return;
        nap();
        budget -= 64;
    }
}

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

// f32 pair -> one byte of two e2m1 nibbles (hi in the high nibble)
NR_INLINE u32 e2m1x2(float hi, float lo)
{
    u32 packed;
    asm volatile("{\n .reg .b8 nr_t;\n cvt.rn.satfinite.e2m1x2.f32 nr_t, %1, %2;\n cvt.u32.u8 %0, nr_t;\n}" : "=r"(packed) : "f"(hi), "f"(lo));
    return packed;
}
// f32 -> e4m3 code
NR_INLINE u32 e4m3_code(float v)
{
    u16 pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(pair) : "f"(v));
    return pair & 0xFFu;
}
// two e4m3 codes (16 bits) -> two floats
NR_INLINE void e4m3x2_to_f32(u32 pair16, float& lo, float& hi)
{
    u32 halves; u16 in = (u16)pair16, l16, h16;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(in));
    l16 = (u16)halves; h16 = (u16)(halves >> 16);
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"(l16));
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"(h16));
}
NR_INLINE float f16_to_f32(u32 half)
{
    float out; u16 in = (u16)half;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(in));
    return out;
}
// two floats -> float16 -> two e4m3 codes (lo in the low byte)
NR_INLINE u32 f32x2_to_e4m3x2(float lo, float hi)
{
    u32 halves; u16 out;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(halves) : "f"(hi), "f"(lo));
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(out) : "r"(halves));
    return out;
}
// block scale: e4m3 code of clamp(amax / 6, 2^-9, 448) and the float32 reciprocal of the stored scale
NR_INLINE u32 block_scale(float amax, float& reciprocal)
{
    const float raw = fminf(fmaxf(amax * 0.16666667163372040f, 0.001953125f), 448.0f);
    const u32 code = e4m3_code(raw);
    float lo, hi;
    e4m3x2_to_f32(code, lo, hi);
    reciprocal = 1.0f / lo;
    return code;
}

// ---------------------------------------------------------------------------------------------------------------
// Pack: one CTA per 16-token group, 16 warps = the 16 k64 steps of K = 1024; a thread owns one AQ lane slot.
// producer: the runtime's flags of x, [tile * 8 + 128-channel chunk] >= 0 when ready (null: x is already complete).
// flags: ours, [16-token group] = gen once the group is packed (null: not published).
struct PackParams { const u8* x; u8* xq; const int* producer; int* flags; u32 tokens; float tensor_scale; int gen; };

extern "C" __global__ void nr_pack_x(PackParams p)
{
    const u32 group = blockIdx.x, ksub = threadIdx.y, lane = threadIdx.x & 31u;
    int budget = WAIT_BUDGET_NS;
    if (p.producer) wait_not_negative(p.producer + (group >> 3) * 8u + (ksub >> 1), budget);
    const u32 r = lane >> 2, hq = lane & 3u, h = hq >> 1, q = hq & 1u;
    const u32 groups = (((p.tokens - 1u) & ~31u) + 32u) >> 4;          // groups that exist in the runtime's buffer
    const bool live = group < groups;
    V4 words = {0u, 0u, 0u, 0u};
    u32 scale_part[2] = {0u, 0u};                                        // per u: bytes j = h and 2 + h
#pragma unroll
    for (u32 kkhi = 0; kkhi < 2; ++kkhi) {
        const u8* block = p.x + (u64)group * 16384u + (2u * ksub + kkhi) * 512u + r * 64u + h * 8u;
#pragma unroll
        for (u32 u = 0; u < 2; ++u) {
            float v[8];
            float amax = 0.0f;
#pragma unroll
            for (u32 t8 = 0; t8 < 4; ++t8) {
                const u32 word = live ? *(const u32*)(block + t8 * 16u + u * 4u) : 0u;
                float lo, hi;
                e4m3x2_to_f32((word >> (16u * q)) & 0xFFFFu, lo, hi);
                v[2 * t8] = lo * p.tensor_scale; v[2 * t8 + 1] = hi * p.tensor_scale;
                amax = fmaxf(amax, fmaxf(fabsf(v[2 * t8]), fabsf(v[2 * t8 + 1])));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));   // the other 8 channels of this block of 16
            float reciprocal;
            const u32 code = block_scale(amax, reciprocal);
            u32 out = 0u;
#pragma unroll
            for (u32 t8 = 0; t8 < 4; ++t8)
                out |= e2m1x2(v[2 * t8 + 1] * reciprocal, v[2 * t8] * reciprocal) << (8u * t8);
            const u32 index = (kkhi << 1) | u;
            if (index == 0) words.x = out; else if (index == 1) words.y = out; else if (index == 2) words.z = out; else words.w = out;
            scale_part[u] |= code << (8u * (kkhi * 2u + h));
        }
    }
    u8* unit = p.xq + ((u64)((group >> 3) * 16u + ksub) * 8u + (group & 7u)) * AQ_UNIT;
    *(V4*)(unit + lane * 16u) = words;
#pragma unroll
    for (u32 u = 0; u < 2; ++u) {
        const u32 full = scale_part[u] | __shfl_xor_sync(0xffffffffu, scale_part[u], 2);
        if (hq == u) *(u32*)(unit + 512u + r * 8u + u * 4u) = full;
    }
    if (p.flags) {
        __syncthreads();
        if ((threadIdx.x | threadIdx.y) == 0) flag_publish(p.flags + group, p.gen);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM core. Activations and weights both arrive through a STAGES-deep bulk-copy pipeline, one k64 step per stage:
// stage = [gt units of A][nh blocks of W].
struct Geometry { u32 gt, nh, tile, gi0, wt, o64, half0, stage_bytes; };

NR_INLINE Geometry geometry(u32 tokens, u32 gt, u32 nh)
{
    Geometry g;
    const u32 tiles = (tokens + 127u) >> 7, parts = tiles * (8u / gt);
    const u32 part = blockIdx.x % parts, oc = blockIdx.x / parts;
    g.gt = gt; g.nh = nh;
    g.tile = part / (8u / gt); g.gi0 = (part % (8u / gt)) * gt;       // first group of this CTA in its tile
    g.wt = threadIdx.y / nh;                                          // warp's pair of groups
    g.o64 = oc * nh + threadIdx.y % nh;                               // warp's 64-output index
    g.half0 = (oc * nh) & 1u;                                         // first half of the CTA within its 128 outputs
    g.stage_bytes = KSTEPS * (gt * AQ_UNIT + nh * WQ_HALF);
    return g;
}

// Stage `stage` = k64 steps stage * KSTEPS ... into ring slot `slot`.
// chain: flags of the activations' producer, one per k64 step (null: the activations are complete).
// Stages are issued in order. peek: with KSTEPS == 2, this stage's two flags as they were when the stage before was
// issued (the caller loads stage 0's); on return it holds the next stage's. Reading them a stage early keeps the load
// out of the way: nothing waits for it unless the producer really is late.
NR_INLINE void issue_stage(u32 smem0, const Geometry& g, const u8* aq_cta, const u8* wq_cta, u32 stage, u32 slot, const int* chain, int gen,
                           int& budget, u64& peek, u32 nstages)
{
    const u32 ksub = stage * KSTEPS;
    if (chain) {
#if KSTEPS == 2
        const u64 both = flag_pair(gen);
        for (; budget > 0 && peek != both; budget -= 64) {
            nap();
            peek = flag_pair_load(chain + ksub);
        }
        if (stage + 1u < nstages) peek = flag_pair_load(chain + ksub + KSTEPS);
#else
        for (u32 j = 0; j < KSTEPS; ++j) wait_equal(chain + ksub + j, gen, budget);
#endif
    }
    const u32 bar = smem0 + 8u * slot, dst = smem0 + BAR_BYTES + slot * g.stage_bytes;
    bar_expect(bar, g.stage_bytes);
    bulk_copy(dst, aq_cta + (u64)ksub * (8u * AQ_UNIT), KSTEPS * g.gt * AQ_UNIT, bar);
    bulk_copy(dst + KSTEPS * g.gt * AQ_UNIT, wq_cta + (u64)ksub * (2u * WQ_HALF), KSTEPS * g.nh * WQ_HALF, bar);
    bar_arrive(bar);
}

// One k64 step of a warp: two 16-token groups (units) against 64 outputs (block), 16 mma.
#define NR_STEP(acc, units, block, lane_)                                                                               \
    {                                                                                                                   \
        const V4 a0 = *(const V4*)((units) + (lane_) * 16u);                                                            \
        const V4 a1 = *(const V4*)((units) + AQ_UNIT + (lane_) * 16u);                                                  \
        const u32 sa0 = *(const u32*)((units) + 512u + ((lane_) >> 2) * 8u + ((lane_) & 1u) * 4u);                      \
        const u32 sa1 = *(const u32*)((units) + AQ_UNIT + 512u + ((lane_) >> 2) * 8u + ((lane_) & 1u) * 4u);            \
        const V4 n0 = *(const V4*)((block) + (lane_) * 16u), n1 = *(const V4*)((block) + 512u + (lane_) * 16u);         \
        const V4 n2 = *(const V4*)((block) + 1024u + (lane_) * 16u), n3 = *(const V4*)((block) + 1536u + (lane_) * 16u); \
        const V4 s0 = *(const V4*)((block) + 2048u + ((lane_) >> 2) * 32u), s1 = *(const V4*)((block) + 2064u + ((lane_) >> 2) * 32u); \
        mma4(acc[0][0], a0, n0.x, n0.y, sa0, s0.x); mma4(acc[1][0], a1, n0.x, n0.y, sa1, s0.x);                         \
        mma4(acc[0][1], a0, n0.z, n0.w, sa0, s0.y); mma4(acc[1][1], a1, n0.z, n0.w, sa1, s0.y);                         \
        mma4(acc[0][2], a0, n1.x, n1.y, sa0, s0.z); mma4(acc[1][2], a1, n1.x, n1.y, sa1, s0.z);                         \
        mma4(acc[0][3], a0, n1.z, n1.w, sa0, s0.w); mma4(acc[1][3], a1, n1.z, n1.w, sa1, s0.w);                         \
        mma4(acc[0][4], a0, n2.x, n2.y, sa0, s1.x); mma4(acc[1][4], a1, n2.x, n2.y, sa1, s1.x);                         \
        mma4(acc[0][5], a0, n2.z, n2.w, sa0, s1.y); mma4(acc[1][5], a1, n2.z, n2.w, sa1, s1.y);                         \
        mma4(acc[0][6], a0, n3.x, n3.y, sa0, s1.z); mma4(acc[1][6], a1, n3.x, n3.y, sa1, s1.z);                         \
        mma4(acc[0][7], a0, n3.z, n3.w, sa0, s1.w); mma4(acc[1][7], a1, n3.z, n3.w, sa1, s1.w);                         \
    }

// aq_cta: AQ unit of (tile, ksub 0, gi0); wq_cta: WQ block of (out128, ksub 0, half0); nsub: K / 64.
// Uses the kernel's `int budget` (the thread's wait budget).
#define NR_GEMM_CORE(acc, G, smem, aq_cta, wq_cta, nsub, chain, gen)                                                     \
    {                                                                                                                   \
        const u32 smem0 = smem_addr(smem), nstages = (nsub) / KSTEPS;                                                   \
        const bool leader = (threadIdx.x | threadIdx.y) == 0;                                                           \
        u64 peek = 0;                                                                                                   \
        if (leader) {                                                                                                   \
            for (u32 s = 0; s < STAGES; ++s) bar_init(smem0 + 8u * s, 1u);                                               \
            if ((chain) && KSTEPS == 2u) peek = flag_pair_load(chain);                                                  \
        }                                                                                                               \
        __syncthreads();                                                                                                \
        if (leader) {                                                                                                   \
            for (u32 s = 0; s < STAGES && s < nstages; ++s) issue_stage(smem0, G, aq_cta, wq_cta, s, s, chain, gen, budget, peek, nstages);\
        }                                                                                                               \
        const u32 lane_ = threadIdx.x & 31u;                                                                            \
        for (u32 ksub = 0; ksub < (nsub); ++ksub) {                                                                     \
            const u32 stage = ksub / KSTEPS, sub = ksub % KSTEPS, slot = stage % STAGES;                                \
            if (sub == 0u) {                                                                                            \
                if (!WARP_WAIT || lane_ == 0u) bar_wait(smem0 + 8u * slot, (stage / STAGES) & 1u);                      \
                if (WARP_WAIT) __syncwarp();                                                                            \
            }                                                                                                           \
            const u8* base = (smem) + BAR_BYTES + slot * G.stage_bytes;                                                 \
            const u8* units = base + sub * (G.gt * AQ_UNIT) + G.wt * (2u * AQ_UNIT);                                    \
            const u8* block = base + KSTEPS * (G.gt * AQ_UNIT) + sub * (G.nh * WQ_HALF) + (threadIdx.y % G.nh) * WQ_HALF; \
            NR_STEP(acc, units, block, lane_)                                                                           \
            if (sub == KSTEPS - 1u) {                                                                                   \
                __syncthreads();                                                                                        \
                if (leader && stage + STAGES < nstages) issue_stage(smem0, G, aq_cta, wq_cta, stage + STAGES, slot, chain, gen, budget, peek, nstages);\
            }                                                                                                           \
        }                                                                                                               \
        __syncthreads();                                                                                                \
    }

#define NR_ZERO_ACC(acc)                                         \
    float acc[2][8][4];                                          \
    for (u32 i_ = 0; i_ < 2; ++i_)                               \
        for (u32 j_ = 0; j_ < 8; ++j_)                           \
            for (u32 k_ = 0; k_ < 4; ++k_) acc[i_][j_][k_] = 0.0f;

// Expand epilogue of one warp: activation, NVFP4 with one scale per 16 hidden channels, written as two units of the
// contraction's A operand. `mine` is 2 * AQ_UNIT bytes of shared memory private to the warp; `out` the two units in
// global memory.
NR_INLINE void expand_store(float acc[2][8][4], u8* mine, u8* out, u32 lane, float out_scale, float hidden_scale)
{
    const u32 g = lane >> 2, t = lane & 3u;
    const float c1 = -0.055908203125f * hidden_scale, c2 = 0.447265625f * hidden_scale, c3 = 0.89453125f * hidden_scale;
#pragma unroll
    for (u32 ga = 0; ga < 2; ++ga) {
        u8* unit = mine + ga * AQ_UNIT;
#pragma unroll
        for (u32 u = 0; u < 2; ++u) {
            float hs[8][2];
#pragma unroll
            for (u32 b = 0; b < 8; ++b)
#pragma unroll
                for (u32 c = 0; c < 2; ++c) {
                    const float v = acc[ga][b][u * 2u + c] * out_scale;
                    const float clipped = fminf(fmaxf(v, -4.0f), 4.0f);
                    hs[b][c] = DBG_SKIP_EPILOGUE ? v : v * fmaf(clipped, fmaf(fabsf(clipped), c1, c2), c3);
                }
            u32 scale_word = 0u;
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                float reciprocal = 1.0f;
                if (!DBG_SKIP_EPILOGUE) {
                    float amax = fmaxf(fmaxf(fabsf(hs[2 * j][0]), fabsf(hs[2 * j][1])), fmaxf(fabsf(hs[2 * j + 1][0]), fabsf(hs[2 * j + 1][1])));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                    scale_word |= block_scale(amax, reciprocal) << (8u * j);
                }
#pragma unroll
                for (u32 b = 2 * j; b < 2 * j + 2; ++b)
                    unit[(4u * g + (b & 3u)) * 16u + ((b >> 2) * 2u + u) * 4u + t] = (u8)e2m1x2(hs[b][1] * reciprocal, hs[b][0] * reciprocal);
            }
            if (t == u) *(u32*)(unit + 512u + g * 8u + u * 4u) = scale_word;
        }
    }
    __syncwarp();
#pragma unroll
    for (u32 ga = 0; ga < 2; ++ga) {
        *(V4*)(out + ga * AQ_UNIT + lane * 16u) = *(const V4*)(mine + ga * AQ_UNIT + lane * 16u);
        if (lane < 16u) *(u32*)(out + ga * AQ_UNIT + 512u + lane * 4u) = *(const u32*)(mine + ga * AQ_UNIT + 512u + lane * 4u);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Expand: grid.x = 4096/(64*nh) * token parts (token part fastest).
// ready: the pack kernel's flags, [16-token group] == gen (null: xq is complete).
// flags: ours, [tile * 64 + 64-channel half of the hidden layer] = gen; only meaningful with gt = 8, where a CTA covers
// whole tiles.
struct ExpandParams { const u8* xq; const u8* w; u8* hq; int* flags; const int* ready; u32 tokens; float out_scale; float hidden_scale; u32 gt; u32 nh; int gen; };

extern "C" __global__ void nr_ffn_expand(ExpandParams p)
{
    extern __shared__ __align__(512) u8 smem[];
    int budget = WAIT_BUDGET_NS;
    const u32 lane = threadIdx.x & 31u;
    const Geometry G = geometry(p.tokens, p.gt, p.nh);
    const u8* aq_cta = p.xq + ((u64)G.tile * 16u * 8u + G.gi0) * AQ_UNIT;                  // K = 1024: 16 k64 steps per tile
    const u8* wq_cta = p.w + ((u64)(G.o64 >> 1) * 16u * 2u + G.half0) * WQ_HALF;           // 16 k64 steps per 128 outputs
    if (p.ready && (threadIdx.x | threadIdx.y) == 0) {
        if (p.gt == 8u) wait_equal8(p.ready + G.tile * 8u, p.gen, budget);
        else
            for (u32 g = 0; g < p.gt; ++g) wait_equal(p.ready + G.tile * 8u + G.gi0 + g, p.gen, budget);
    }
    NR_ZERO_ACC(acc)
    NR_GEMM_CORE(acc, G, smem, aq_cta, wq_cta, 16u, (const int*)0, 0)

    expand_store(acc, smem + BAR_BYTES + threadIdx.y * (2u * AQ_UNIT),
                 p.hq + (((u64)G.tile * 64u + G.o64) * 8u + G.gi0 + G.wt * 2u) * AQ_UNIT, lane, p.out_scale, p.hidden_scale);
    __syncthreads();
    if ((threadIdx.x | threadIdx.y) == 0 && p.flags)
        for (u32 h = 0; h < p.nh; ++h) flag_publish(p.flags + G.tile * 64u + G.o64 + h, p.gen);
}

// ---------------------------------------------------------------------------------------------------------------
// Expand with the activations resident in shared memory.
//
// Data reaches an SM at no more than about 43 GB/s on this GPU, bulk copies and ordinary loads together, and the plain
// GEMM above needs 72 bytes per mma: more than that rate can feed. Here a CTA owns one 128-token tile and a contiguous
// range of 128-output tiles. It loads the tile's activations once, during its first round, and afterwards streams only
// weights (36 bytes per mma), so the tensor cores set the pace.
//   grid.x = tiles * per_tile (part fastest), block (32, 8), dynamic shared memory 92224 bytes:
//   [barriers 64][activations 73728][weight ring 2 * 9216]
// A stage is two k64 steps of both halves' weights (one 9216-byte copy) plus, in the first round, the activations of
// those steps (another). At the end of a round the slot just consumed serves as the epilogue's scratch.
// A CTA may have at most 101376 bytes; one byte more and the driver drops the device instead of refusing the launch.
#define RES_A_BYTES (16u * 8u * AQ_UNIT)
#define RES_STAGE (2u * 2u * WQ_HALF)

struct ExpandResidentParams { const u8* xq; const u8* w; u8* hq; int* flags; const int* ready; u32 tokens; float out_scale; float hidden_scale; u32 per_tile; int gen; };

// Global pair gp = round * 8 + (k64 step / 2); o0 = the CTA's first 128-output tile.
NR_INLINE void resident_issue(u32 smem0, const ExpandResidentParams& p, const u8* aq_tile, u32 o0, u32 gp)
{
    const u32 round = gp >> 3, k = (gp & 7u) * 2u, slot = gp & 1u, bar = smem0 + 8u * slot;
    bar_expect(bar, (round ? 0u : 2u * 8u * AQ_UNIT) + RES_STAGE);
    if (!round) bulk_copy(smem0 + BAR_BYTES + k * (8u * AQ_UNIT), aq_tile + (u64)k * (8u * AQ_UNIT), 2u * 8u * AQ_UNIT, bar);
    bulk_copy(smem0 + BAR_BYTES + RES_A_BYTES + slot * RES_STAGE, p.w + ((u64)(o0 + round) * 16u + k) * (2u * WQ_HALF), RES_STAGE, bar);
    bar_arrive(bar);
}

extern "C" __global__ void nr_ffn_expand_resident(ExpandResidentParams p)
{
    extern __shared__ __align__(512) u8 smem[];
    int budget = WAIT_BUDGET_NS;
    const u32 lane = threadIdx.x & 31u, wt = threadIdx.y >> 1, hsel = threadIdx.y & 1u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const u32 tile = blockIdx.x / p.per_tile, part = blockIdx.x % p.per_tile;
    const u32 o0 = part * 32u / p.per_tile, o1 = (part + 1u) * 32u / p.per_tile;          // this CTA's 128-output tiles [o0, o1)
    const u32 pairs = (o1 - o0) * 8u;
    const u32 smem0 = smem_addr(smem);
    const u8* aq_tile = p.xq + (u64)tile * (16u * 8u * AQ_UNIT);
    if (leader) {
        if (p.ready)
            wait_equal8(p.ready + tile * 8u, p.gen, budget);
        bar_init(smem0, 1u); bar_init(smem0 + 8u, 1u);
    }
    __syncthreads();
    if (leader)
        for (u32 gp = 0; gp < 2u && gp < pairs; ++gp) resident_issue(smem0, p, aq_tile, o0, gp);
    NR_ZERO_ACC(acc)
    for (u32 gp = 0; gp < pairs; ++gp) {
        const u32 slot = gp & 1u, k = (gp & 7u) * 2u;
        u8* stage = smem + BAR_BYTES + RES_A_BYTES + slot * RES_STAGE;
        bar_wait(smem0 + 8u * slot, (gp >> 1) & 1u);
#pragma unroll
        for (u32 sub = 0; sub < 2u; ++sub) {
            const u8* units = smem + BAR_BYTES + (k + sub) * (8u * AQ_UNIT) + wt * (2u * AQ_UNIT);
            const u8* block = stage + sub * (2u * WQ_HALF) + hsel * WQ_HALF;
            NR_STEP(acc, units, block, lane)
        }
        __syncthreads();
        if ((gp & 7u) == 7u) {
            const u32 half = (o0 + (gp >> 3)) * 2u + hsel;
            expand_store(acc, stage + threadIdx.y * (2u * AQ_UNIT), p.hq + (((u64)tile * 64u + half) * 8u + wt * 2u) * AQ_UNIT,
                         lane, p.out_scale, p.hidden_scale);
            __syncthreads();
            if (leader && p.flags) {
                flag_publish(p.flags + tile * 64u + (o0 + (gp >> 3)) * 2u, p.gen);
                flag_publish(p.flags + tile * 64u + (o0 + (gp >> 3)) * 2u + 1u, p.gen);
            }
            for (u32 i = 0; i < 2; ++i)
                for (u32 j = 0; j < 8; ++j)
                    for (u32 c = 0; c < 4; ++c) acc[i][j][c] = 0.0f;
        }
        if (leader && gp + 2u < pairs) resident_issue(smem0, p, aq_tile, o0, gp + 2u);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Contract: grid.x = 1024/(64*nh) * token parts. Writes the runtime's FP8 layout.
// ready: the expand kernel's flags, [tile * 64 + 64-channel half] == gen (null: hq is complete).
// flags and counter: the runtime's, [tile * 8 + 128-channel chunk of the output]; left as its own contract leaves
// them (flag 0, merge counter 3). Only meaningful with gt = 8, nh = 2.
struct ContractParams { const u8* hq; const u8* w; const u8* x; const u16* gain; u8* out; int* flags; int* counter; const int* ready; u32 tokens; float out_scale; u32 gt; u32 nh; int gen; };

extern "C" __global__ void nr_ffn_contract(ContractParams p)
{
    extern __shared__ __align__(512) u8 smem[];
    int budget = WAIT_BUDGET_NS;
    const u32 lane = threadIdx.x & 31u;
    const Geometry G = geometry(p.tokens, p.gt, p.nh);
    const u8* aq_cta = p.hq + ((u64)G.tile * 64u * 8u + G.gi0) * AQ_UNIT;                  // K = 4096: 64 k64 steps per tile
    const u8* wq_cta = p.w + ((u64)(G.o64 >> 1) * 64u * 2u + G.half0) * WQ_HALF;           // 64 k64 steps per 128 outputs
    const int* chain = p.ready ? p.ready + G.tile * 64u : (const int*)0;
    NR_ZERO_ACC(acc)
    NR_GEMM_CORE(acc, G, smem, aq_cta, wq_cta, 64u, chain, p.gen)

    // epilogue: y = acc * out_scale + gain * x, float16, e4m3, in the runtime's layout (a lane owns whole 16-byte slots)
    const u32 t = lane & 3u;
    const u32 groups = (((p.tokens - 1u) & ~31u) + 32u) >> 4;
#pragma unroll
    for (u32 ga = 0; ga < 2; ++ga) {
        const u32 group = G.tile * 8u + G.gi0 + G.wt * 2u + ga;
        if (group < groups) {
#pragma unroll
            for (u32 kbl = 0; kbl < 2; ++kbl) {
                const u64 offset = (u64)group * 16384u + (G.o64 * 2u + kbl) * 512u + lane * 16u;
                const V4 skip = *(const V4*)(p.x + offset);
                const u32 skip_words[4] = {skip.x, skip.y, skip.z, skip.w};
                u32 out_words[4];
#pragma unroll
                for (u32 h = 0; h < 2; ++h)
#pragma unroll
                    for (u32 u = 0; u < 2; ++u) {
                        u32 word = 0u;
#pragma unroll
                        for (u32 q = 0; q < 2; ++q) {
                            const u32 b = kbl * 4u + h * 2u + q;
                            const u32 gains = *(const u32*)(p.gain + G.o64 * 64u + b * 8u + t * 2u);
                            float x0, x1;
                            e4m3x2_to_f32((skip_words[h * 2u + u] >> (16u * q)) & 0xFFFFu, x0, x1);
                            const float y0 = acc[ga][b][u * 2u] * p.out_scale + f16_to_f32(gains & 0xFFFFu) * x0;
                            const float y1 = acc[ga][b][u * 2u + 1u] * p.out_scale + f16_to_f32(gains >> 16) * x1;
                            word |= f32x2_to_e4m3x2(y0, y1) << (16u * q);
                        }
                        out_words[h * 2u + u] = word;
                    }
                V4 packed = {out_words[0], out_words[1], out_words[2], out_words[3]};
                *(V4*)(p.out + offset) = packed;
            }
        }
    }
    __syncthreads();
    if ((threadIdx.x | threadIdx.y) == 0) {
        if (p.counter) flag_publish(p.counter + G.tile * 8u + (G.o64 >> 1), 3);
        if (p.flags) flag_publish(p.flags + G.tile * 8u + (G.o64 >> 1), 0);
    }
}
