// amp_vit.cu: Ampere-legal (sm_86) FP16 replacements of the DLSS-NR transformer launches (runtime 310.9.16, Model A):
//   cc_vit_1d_ffn_expand_publish_fp8   -> amp_vit_ffn_expand_publish     grid (32 T, 1, 1)  block (32, 4)  72 B params
//   cc_vit_1d_ffn_expand_chained_fp8   -> amp_vit_ffn_expand_chained     grid (32 T, 1, 1)  block (32, 4)  72 B
//   cc_vit_1d_ffn_contract_chained_fp8 -> amp_vit_ffn_contract_chained   grid (8 T, 1, 4)   block (32, 4)  72 B
//   cc_vit_1d_qkv_chained_fp8          -> amp_vit_qkv_chained            grid (16 T, 1, 2)  block (32, 4)  80 B
//   cc_vit_1d_attention_chained_fp8    -> amp_vit_attention_chained      grid (32, ceil(T/2), 1) block (32, 4)  64 B
//   cc_vit_1d_projection_chained_fp8   -> amp_vit_projection_chained     grid (8 T, 1, 4)   block (32, 4)  72 B
//   cc_vit_1d_projection_wait_fp8      -> amp_vit_projection_wait        grid (8 T, 1, 4)   block (32, 4)  72 B
//   cc_vit_1d_repack_2d_to_1d_fp8      -> amp_vit_repack_2d_to_1d        grid-stride, any grid (runtime: W / 256 x 256)
//   cc_vit_1d_repack_1d_to_2d_fp8      -> amp_vit_repack_1d_to_2d        grid-stride, any grid (runtime: N x 256)
// T = 128-token tiles = (N - 1) / 128 + 1, N = dims[0] * dims[1].
//
// Each entry takes EXACTLY the runtime kernel's parameter block (same size, same fields; the weight record pointer is
// NVIDIA's own e4m3 record, as fp8_set<n>.bin holds it), reads and writes the same buffers in the same layouts
// (vit-spec/abi.md 1-5, kernel-spec/ffn-fp8-spec.md 1-4), honours and publishes the same flags and counters, and needs
// no dynamic shared memory (static shared <= 32 KB). Extensions that never trigger with the runtime's blocks: a null
// producer-flag pointer skips that kernel's waits; a null own-flag pointer skips the publish. No wait gives up silently
// (over AMP_WAIT_LIMIT_NS one traps: a visible GPU fault).
//
// Arithmetic (phase A target): NVIDIA's FP8 kernels' order, with each e4m3 m16n8k32 f16-accumulate instruction
// replaced by two m16n8k16 f16.f16.f16.f16 instructions on fragments decoded exactly in registers (amp_common.cuh:
// e4m3 register 0 -> f16 registers 0 and 2, register 1 -> 1 and 3; accumulator seeding unchanged), stores rounded to
// e4m3 in integer code exactly as cvt.rn.satfinite.e4m3x2.f16x2. Everything else (cubic SiLU, the Q/K norms,
// rsqrt/rcp.approx, the attention exponential and its fixed f16 sum order, the K-split merges) in NVIDIA's order.
//   -DAMP_VIT_F32ACC=1 (variant f32acc, NOT the phase-A target): each k32 step accumulates its two k16 steps in f32
//   and rounds once to f16, as NVIDIA's k32 instruction does (half tensor rate on GeForce Ampere).
//
// amp-variant: base
// amp-variant: f32acc -DAMP_VIT_F32ACC=1
// amp-variant: ns2 -DGEMM_NS=2
#include "amp_common.cuh"

#ifndef AMP_VIT_F32ACC
#define AMP_VIT_F32ACC 0
#endif
// attention: 64-key groups per shared-memory stage (two stages of 4 KB x ATT_SG)
#ifndef ATT_SG
#define ATT_SG 4
#endif
#ifndef GEMM_NS
#define GEMM_NS 3
#endif                                  // GEMM kernels: shared-memory stages of 8 groups x 2 K blocks (8 KB each)
#define STAGE_BYTES 8192u
#define FLOOR_BITS 0x388205FFu          // NVIDIA's norm / denominator floor (f32 bits), narrowed to f16 = 1040 * 2^-24

struct FfnParams {          // expand / contract / projection: the runtime's 72 bytes
    const u8* p0; const u8* p1; u8* p2; const u8* p3; int* p4; u8* p5; const int* p6; int* p7; int d0, d1;
};
struct QkvParams {          // the runtime's 80 bytes
    const u8* x; u8* q; u8* k; u8* v; const u8* w; int* counter; u8* scratch; const int* pflags; int* oflags; int d0, d1;
};
struct AttParams {          // the runtime's 64 bytes (+32, the 2-byte record, is not read)
    const u8* q; const u8* k; const u8* v; u8* out; const u8* rec; const int* pflags; int* oflags; int d0, d1;
};
struct RepackParams {       // the runtime's 24 bytes
    const u32* in; u32* out; int d0, d1;
};

// AMP_PREPARED=1 (NrHost 'prepare amp_prepare_vit 2'): the kernel's record is the host's prepared copy: every piece at
// twice its record offset (WS), each 512-byte e4m3 weight slice as 1024 bytes of f16 B fragments (lane L's 16 bytes:
// the two k16 fragments of its first k32 word pair at +16 L, of its second at +512 + 16 L). Same layout as amp_prepost.
#ifndef AMP_PREPARED
#define AMP_PREPARED 0
#endif
#define WS (1u + AMP_PREPARED)
#ifndef GEMM_MINB                       // lab: minimum CTAs per SM for the expand / contract / projection kernels
#define GEMM_MINB 3
#endif

AMP_INLINE int tiles_of(int n) { return (n - 1) / 128 + 1; }
AMP_INLINE int groups_of(int n) { return (n + 31) / 32 * 2; }
AMP_INLINE u32 h2_dup(float x) { return amp_h2_pack(x, x); }
AMP_INLINE u32 h2_swap(u32 h) { return amp_prmt(h, h, 0x1032u); }
AMP_INLINE float sqrt_approx(float x) { float d; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
AMP_INLINE float f16_round(float x) { float lo, hi; amp_h2_unpack(amp_h2_pack(x, x), lo, hi); return lo; }
AMP_INLINE void st_v4(u8* p, u32 a, u32 b, u32 c, u32 d) { amp_v4 v = {a, b, c, d}; *(amp_v4*)p = v; }
// two float16 pairs (rows r, r + 8 of one fragment, or two fragments of one row) -> one e4m3 word (lo pair -> bytes 0, 1)
// AMP_VIT_OLD_CODEC=1: the first integer forms (identical results except NaN inputs; BRIEF-11 timing comparisons).
#ifndef AMP_VIT_OLD_CODEC
#define AMP_VIT_OLD_CODEC 0
#endif
#if AMP_VIT_OLD_CODEC
AMP_INLINE u32 e8w(u32 lo, u32 hi) { return amp_h2x2_to_e4m3x4(lo, hi); }
#else
AMP_INLINE u32 e8w(u32 lo, u32 hi) { return amp_h2x2_to_e4m3x4_fast(lo, hi); }
#endif
// Lab timing switches (WRONG numbers, never shipped): AMP_TIME_NOADEC (activation A fragments), AMP_TIME_NOWDEC (weight
// B fragments of the GEMMs), AMP_TIME_NOKVDEC (attention K / V fragments): the e4m3 decode replaced by byte moves.
#ifndef AMP_TIME_NOADEC
#define AMP_TIME_NOADEC 0
#endif
#ifndef AMP_TIME_NOWDEC
#define AMP_TIME_NOWDEC 0
#endif
#ifndef AMP_TIME_NOKVDEC
#define AMP_TIME_NOKVDEC 0
#endif
AMP_INLINE void fake_b(const u32* b8, u32* k0, u32* k1)
{
    k0[0] = amp_prmt(b8[0], 0u, 0x1404u); k0[1] = amp_prmt(b8[0], 0u, 0x3424u);
    k1[0] = amp_prmt(b8[1], 0u, 0x1404u); k1[1] = amp_prmt(b8[1], 0u, 0x3424u);
}
AMP_INLINE void frag_a(const u32* a8, u32* k0, u32* k1)
{
#if AMP_TIME_NOADEC
    fake_b(a8, k0, k0 + 2); fake_b(a8 + 2, k1, k1 + 2);
#else
    amp_frag_a_e4m3(a8, k0, k1);
#endif
}
AMP_INLINE void frag_w(const u32* b8, u32* k0, u32* k1)
{
#if AMP_TIME_NOWDEC
    fake_b(b8, k0, k1);
#else
    amp_frag_b_e4m3(b8, k0, k1);
#endif
}
AMP_INLINE void frag_kv(const u32* b8, u32* k0, u32* k1)
{
#if AMP_TIME_NOKVDEC
    fake_b(b8, k0, k1);
#else
    amp_frag_b_e4m3(b8, k0, k1);
#endif
}

// f16(rsqrt.approx(f32(h))) per half, written as NVIDIA's QKV writes it: per-half cvt.f32.f16, rsqrt.approx.ftz.f32,
// per-half cvt.rn.f16.f32 in one block (ptxas may map this pattern to a different hardware sequence than a packed
// cvt.rn.f16x2.f32; the per-half form is the one that reproduces NVIDIA's codes)
AMP_INLINE u32 rsqrt_h2(u32 h)
{
    u32 d;
    asm("{\n .reg .b16 hl, hu;\n .reg .b32 fl, fu;\n mov.b32 {hl, hu}, %1;\n cvt.f32.f16 fl, hl;\n cvt.f32.f16 fu, hu;\n"
        " rsqrt.approx.ftz.f32 fl, fl;\n rsqrt.approx.ftz.f32 fu, fu;\n cvt.rn.f16.f32 hl, fl;\n cvt.rn.f16.f32 hu, fu;\n"
        " mov.b32 %0, {hl, hu};\n}" : "=r"(d) : "r"(h));
    return d;
}

// e4m3 rounding of a non-negative finite float16 pair, kept as float16 (= decode(encode(x)), x < 448): the attention's
// P operand. Normal results (x >= 2^-6): round the 7 dropped mantissa bits to even; below: x + 2 rounds to 2^-9 steps.
AMP_INLINE u32 snap_e4m3_pos(u32 h)
{
#if !AMP_VIT_OLD_CODEC
    return amp_h2_q8_pos(h);
#endif
    const u32 normal = (h + 0x003F003Fu + ((h >> 7) & 0x00010001u)) & 0xFF80FF80u;
    const u32 sub = amp_h2_sub(amp_h2_add(h, 0x40004000u), 0x40004000u);
    const u32 big = (((h + 0x5C005C00u) >> 15) & 0x00010001u) * 0xFFFFu;   // halves with h >= 0x2400
    return (normal & big) | (sub & ~big);
}

// one NVIDIA k32 step on decoded fragments: D = A(K 16..31) B + (A(K 0..15) B + C) in NVIDIA's K numbering (K 0..15 first;
// the other order measured no closer to NVIDIA)
AMP_INLINE void step32(u32* c, const u32* a0, const u32* a1, const u32* b0, const u32* b1)
{
#if AMP_VIT_F32ACC
    float f[4];
    amp_h2_unpack(c[0], f[0], f[1]);
    amp_h2_unpack(c[1], f[2], f[3]);
    amp_mma16_f32(f, a0, b0);
    amp_mma16_f32(f, a1, b1);
    c[0] = amp_h2_pack(f[0], f[1]);
    c[1] = amp_h2_pack(f[2], f[3]);
#else
    amp_mma16_f16(c, a0, b0);
    amp_mma16_f16(c, a1, b1);
#endif
}

// =====================================================================================================================
// GEMM core of expand / contract / projection / QKV (NVIDIA's tiling, ffn-fp8-spec.md 3 and 5): CTA = 128 tokens (the
// tile's 8 groups of 16) x 128 (QKV: 192) outputs; warp w = tokens 64 (w >> 1) .. + 63 (4 M fragments) x NP pairs of
// 8-output fragments. A (activations, act layout = A-fragment order) through shared memory (cp.async, GEMM_NS stages of
// two 32-channel K blocks); B (weights, weight layout = B-fragment order) straight from global, 16 B per lane = two
// fragments.

// stage s of the input: K blocks kb0, kb0 + 1 of the tile's 8 groups -> [group][kb][512] (groups >= G: zeros)
AMP_INLINE void stage_issue(u8* st, const u8* src, u32 gstride, u32 tile, int G, u32 kb0, u32 tid)
{
#pragma unroll
    for (u32 r = 0; r < 4; ++r) {
        const u32 c = tid + 128u * r, gi = c >> 6, part = (c >> 5) & 1u, off = (c & 31u) * 16u;
        const u32 grp = tile * 8u + gi;
        const u32 dst = amp_smem(st + gi * 1024u + part * 512u + off);
        if ((int)grp < G) amp_cp_async16(dst, src + (u64)grp * gstride + (kb0 + part) * 512u + off);
        else amp_cp_async16_zfill(dst, src, 0u);
    }
    amp_cp_async_commit();
}

// the products of one stage: K blocks kbl, kbl + 1 (weight rows at wb + kb * kstride + np * 512, wb includes lane * 16)
template <int NP>
AMP_INLINE void stage_mma(u32 (&acc)[4][2 * NP][2], const u8* st, u32 th, u32 lane, const u8* wb, u32 kstride, u32 kbl)
{
#if NR_NATIVE_FP8
#pragma unroll
    for (unsigned part = 0; part < 2; ++part) {
        amp_v4 input[4];
#pragma unroll
        for (unsigned row = 0; row < 4; ++row)
            input[row] = *(const amp_v4*)(st + (th * 4u + row) * 1024u + part * 512u + lane * 16u);
        const u8* weights = wb + (u64)(kbl + part) * kstride;
#pragma unroll
        for (unsigned n = 0; n < NP; ++n) {
            const amp_v4 w = amp_load_nc(weights + n * 512u);
#pragma unroll
            for (unsigned row = 0; row < 4; ++row) {
                const amp_v4 a = input[row];
                u32* left = acc[row][2 * n];
                u32* right = acc[row][2 * n + 1];
                asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                    : "+r"(left[0]), "+r"(left[1]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(w.x), "r"(w.y));
                asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                    : "+r"(right[0]), "+r"(right[1]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(w.z), "r"(w.w));
            }
        }
    }
#else
#pragma unroll
    for (u32 part = 0; part < 2; ++part) {
        u32 A0[4][4], A1[4][4];
#pragma unroll
        for (u32 m = 0; m < 4; ++m) {
            const amp_v4 a = *(const amp_v4*)(st + (th * 4u + m) * 1024u + part * 512u + lane * 16u);
            const u32 a8[4] = {a.x, a.y, a.z, a.w};
            frag_a(a8, A0[m], A1[m]);
        }
        const u8* wk = wb + (u64)(kbl + part) * kstride;        // (kstride and wb already scaled by WS)
#pragma unroll
        for (u32 np = 0; np < (u32)NP; ++np) {
#if AMP_PREPARED
            const amp_v4 b = amp_load_nc(wk + np * 1024u), c = amp_load_nc(wk + np * 1024u + 512u);
            const u32 B0[2] = {b.x, b.y}, B1[2] = {b.z, b.w}, C0[2] = {c.x, c.y}, C1[2] = {c.z, c.w};
#else
            const amp_v4 b = amp_load_nc(wk + np * 512u);
            const u32 bl[2] = {b.x, b.y}, bh[2] = {b.z, b.w};
            u32 B0[2], B1[2], C0[2], C1[2];
            frag_w(bl, B0, B1);
            frag_w(bh, C0, C1);
#endif
#pragma unroll
            for (u32 m = 0; m < 4; ++m) {
                step32(acc[m][2u * np], A0[m], A1[m], B0, B1);
                step32(acc[m][2u * np + 1u], A0[m], A1[m], C0, C1);
            }
        }
    }
#endif
}

// the whole K loop over `stages` stages. wait(s) runs on the leader thread only, before stage s is issued (a CTA
// barrier follows before anybody issues it): the producer-flag waits.
template <int NP, class Wait>
AMP_INLINE void gemm_loop(u32 (&acc)[4][2 * NP][2], u8* sm, const u8* src, u32 gstride, u32 tile, int G, int stages,
                          const u8* wb, u32 kstride, Wait wait, u32 tid, u32 th, u32 lane, bool leader)
{
#pragma unroll
    for (int s = 0; s < GEMM_NS - 1; ++s) {
        if (s < stages) {
            if (leader) wait(s);
            __syncthreads();
            stage_issue(sm + s * STAGE_BYTES, src, gstride, tile, G, 2u * (u32)s, tid);
        } else {
            amp_cp_async_commit();
        }
    }
#pragma unroll 1
    for (int k = 0; k < stages; ++k) {
        const int s = k + GEMM_NS - 1;
        if (leader && s < stages) wait(s);
        amp_cp_async_wait<GEMM_NS - 2>();
        __syncthreads();                                        // stage k landed for all; stage k - 1 consumed by all
        if (s < stages) stage_issue(sm + (s % GEMM_NS) * STAGE_BYTES, src, gstride, tile, G, 2u * (u32)s, tid);
        else amp_cp_async_commit();
        stage_mma<NP>(acc, sm + (k % GEMM_NS) * STAGE_BYTES, th, lane, wb, kstride, 2u * (u32)k);
    }
}

AMP_INLINE void wait_flag(const int* f, int& budget)
{
    amp_wait_not_negative(f, budget);
    __threadfence();
}

// act-layout word set of one 32-channel block from the f16 accumulators of 4 fragments n0..n0+3 (row r: [0], r+8: [1])
AMP_INLINE void store_act(u8* dst, const u32 (*c)[2])
{
    st_v4(dst, e8w(c[0][0], c[1][0]), e8w(c[0][1], c[1][1]), e8w(c[2][0], c[3][0]), e8w(c[2][1], c[3][1]));
}

// =====================================================================================================================
// FFN expand: hidden = e4m3(cubic SiLU(x W1^T)), 1024 -> 4096 (ffn-fp8-spec.md 5.3)
// NPW = B fragment pairs per warp: 4 (NVIDIA's 128-output CTA) or 2 (BRIEF-11 '64': 64 outputs per CTA, grid x twice
// the runtime's, so a launch fills the GPU in fewer, fuller waves; the two CTAs of a 128-channel group each add 1 to its
// flag, so the contraction that reads it must wait for >= 1 instead of >= 0: amp_vit_ffn_contract_chained_after64).
template <bool CHAINED, int NPW = 4>
AMP_INLINE void ffn_expand(const FfnParams& p, u8* sm)
{
    constexpr u32 CW = 32u * NPW;                               // outputs per CTA
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, tid = w * 32u + lane;
    const bool leader = tid == 0u;
    const int N = p.d0 * p.d1, T = tiles_of(N), G = groups_of(N);
    const u32 cg = blockIdx.x / (u32)T, tile = blockIdx.x - cg * (u32)T, th = w >> 1;
    const u32 out0 = cg * CW + (CW / 2u) * (w & 1u);
    u32 acc[4][2 * NPW][2];
#pragma unroll
    for (u32 m = 0; m < 4; ++m)
#pragma unroll
        for (u32 n = 0; n < 2u * NPW; ++n) acc[m][n][0] = acc[m][n][1] = 0u;
    int budget = AMP_WAIT_BUDGET_NS;
    const int* pf = CHAINED ? p.p6 : nullptr;
    auto wait = [&](int s) {                                    // x chunk of 128 channels = 2 stages
        if (pf && (s & 1) == 0) wait_flag(pf + tile * 8u + (u32)(s >> 1), budget);
    };
    gemm_loop<NPW>(acc, sm, p.p0, 16384u, tile, G, 16, p.p3 + WS * out0 * 32u + lane * 16u, WS * 131072u, wait, tid, th, lane, leader);

    // cubic SiLU on the f16 accumulator: t = clamp(v, -4, 4); v * fma(t, fma(-0.0559, |t|, 0.4473), 0.8945)
    const u32 c4 = h2_dup(4.0f), cm4 = h2_dup(-4.0f), ca = h2_dup(0.89453125f), cb = h2_dup(0.447265625f),
              cc = h2_dup(-0.055908203125f);
#pragma unroll
    for (u32 m = 0; m < 4; ++m)
#pragma unroll
        for (u32 n = 0; n < 2u * NPW; ++n)
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const u32 v = acc[m][n][r];
                const u32 t = amp_h2_max(amp_h2_min(v, c4), cm4);
                const u32 u = amp_h2_fma(cc, amp_h2_abs(t), cb);
                acc[m][n][r] = amp_h2_mul(v, amp_h2_fma(t, u, ca));
            }
#pragma unroll
    for (u32 m = 0; m < 4; ++m) {
        const u32 grp = tile * 8u + th * 4u + m;
        if ((int)grp >= G) continue;
#pragma unroll
        for (u32 blk = 0; blk < NPW / 2u; ++blk)
            store_act(p.p2 + (u64)grp * 65536u + ((out0 >> 5) + blk) * 512u + lane * 16u, acc[m] + 4u * blk);
    }
    if (p.p7) {
        __threadfence();
        __syncthreads();
        if (leader) {
            if (NPW == 4) amp_flag_publish(p.p7 + tile * 32u + cg, 0);
            else asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"((u64)(p.p7 + tile * 32u + (cg >> 1))) : "memory");
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, GEMM_MINB) amp_vit_ffn_expand_publish(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    ffn_expand<false>(p, sm);
}
extern "C" __global__ void __launch_bounds__(128, GEMM_MINB) amp_vit_ffn_expand_chained(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    ffn_expand<true>(p, sm);
}
extern "C" __global__ void __launch_bounds__(128, 4) amp_vit_ffn_expand64_publish(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    ffn_expand<false, 2>(p, sm);
}
extern "C" __global__ void __launch_bounds__(128, 4) amp_vit_ffn_expand64_chained(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    ffn_expand<true, 2>(p, sm);
}

// =====================================================================================================================
// FFN contract (KBQ = 32, input C = 4096) and attention projection (KBQ = 8, input C = 1024): K split over z quarters,
// out = e4m3(f16(f16(f16(a0 + a1) + a2) + a3)), a0 seeded with f16(f16(skip) * gain) (ffn-fp8-spec.md 4.4, 5.3;
// abi.md 5.3, 7.3). Same 72-byte block: p0 input, p1 skip, p2 out, p3 record (matrix, gain), p4 counter, p5 scratch,
// p6 producer flags, p7 own flags.
template <bool PROJ, bool PUBLISH, int EXPAND_READY = 0>
AMP_INLINE void merge_body(const FfnParams& p, u8* sm)
{
    constexpr u32 KBQ = PROJ ? 8u : 32u;
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, tid = w * 32u + lane, t = lane & 3u;
    const bool leader = tid == 0u;
    const int N = p.d0 * p.d1, T = tiles_of(N), G = groups_of(N);
    const u32 cg = blockIdx.x / (u32)T, tile = blockIdx.x - cg * (u32)T, z = blockIdx.z, th = w >> 1;
    const u32 out0 = cg * 128u + 64u * (w & 1u);
    const u8* gain = p.p3 + WS * (PROJ ? 1048576u : 4194304u);
    u32 acc[4][8][2];
#pragma unroll
    for (u32 m = 0; m < 4; ++m)
#pragma unroll
        for (u32 n = 0; n < 8; ++n) acc[m][n][0] = acc[m][n][1] = 0u;
    int budget = AMP_WAIT_BUDGET_NS;
    const int* pf = p.p6;
    auto wait = [&](int s) {
        if (!pf) return;
        if (PROJ) {                                             // attention flags, one per input K block (head)
            wait_flag(pf + tile * 32u + 8u * z + 2u * (u32)s, budget);
            wait_flag(pf + tile * 32u + 8u * z + 2u * (u32)s + 1u, budget);
        } else if ((s & 1) == 0) {                              // expand flags, one per 128 hidden channels
            if (EXPAND_READY) { amp_wait_at_least(pf + tile * 32u + 8u * z + (u32)(s >> 1), EXPAND_READY, budget); __threadfence(); }
            else wait_flag(pf + tile * 32u + 8u * z + (u32)(s >> 1), budget);
        }
    };
    // the first stage (and its producer wait) before the skip is read: run the prologue by hand
    const u8* src = p.p0 + z * KBQ * 512u;
    const u32 gstride = PROJ ? 16384u : 65536u;
    const u8* wb = p.p3 + WS * ((u64)z * KBQ * 32768u + out0 * 32u) + lane * 16u;
    const int stages = (int)KBQ / 2;
#pragma unroll
    for (int s = 0; s < GEMM_NS - 1; ++s) {
        if (s < stages) {
            if (leader) wait(s);
            __syncthreads();
            stage_issue(sm + s * STAGE_BYTES, src, gstride, tile, G, 2u * (u32)s, tid);
        } else {
            amp_cp_async_commit();
        }
    }
    if (z == 0u) {                                              // seed: f16(skip) * gain pairs, NVIDIA's mul.f16x2
#pragma unroll
        for (u32 m = 0; m < 4; ++m) {
            const u32 grp = tile * 8u + th * 4u + m;
            if ((int)grp >= G) continue;
#pragma unroll
            for (u32 blk = 0; blk < 2; ++blk) {
                const amp_v4 sk = amp_load_cg(p.p1 + (u64)grp * 16384u + ((out0 >> 5) + blk) * 512u + lane * 16u);
                u32 r0[2], r1[2], r2[2], r3[2];                 // [frag pair half]: words x (row r), y (r+8), z, w
                amp_e4m3x4_to_h2x2_finite(sk.x, r0[0], r0[1]);
                amp_e4m3x4_to_h2x2_finite(sk.y, r1[0], r1[1]);
                amp_e4m3x4_to_h2x2_finite(sk.z, r2[0], r2[1]);
                amp_e4m3x4_to_h2x2_finite(sk.w, r3[0], r3[1]);
                const u32 n0 = 4u * blk;
                u32 gp[4];
#pragma unroll
                for (u32 j = 0; j < 4; ++j) gp[j] = *(const u32*)(gain + 2u * (out0 + 8u * (n0 + j) + 2u * t));
                acc[m][n0 + 0u][0] = amp_h2_mul(r0[0], gp[0]); acc[m][n0 + 0u][1] = amp_h2_mul(r1[0], gp[0]);
                acc[m][n0 + 1u][0] = amp_h2_mul(r0[1], gp[1]); acc[m][n0 + 1u][1] = amp_h2_mul(r1[1], gp[1]);
                acc[m][n0 + 2u][0] = amp_h2_mul(r2[0], gp[2]); acc[m][n0 + 2u][1] = amp_h2_mul(r3[0], gp[2]);
                acc[m][n0 + 3u][0] = amp_h2_mul(r2[1], gp[3]); acc[m][n0 + 3u][1] = amp_h2_mul(r3[1], gp[3]);
            }
        }
    }
#pragma unroll 1
    for (int k = 0; k < stages; ++k) {
        const int s = k + GEMM_NS - 1;
        if (leader && s < stages) wait(s);
        amp_cp_async_wait<GEMM_NS - 2>();
        __syncthreads();
        if (s < stages) stage_issue(sm + (s % GEMM_NS) * STAGE_BYTES, src, gstride, tile, G, 2u * (u32)s, tid);
        else amp_cp_async_commit();
        stage_mma<4>(acc, sm + (k % GEMM_NS) * STAGE_BYTES, th, lane, wb, WS * 32768u, 2u * (u32)k);
    }

    // merge in z order behind the counter: 0 stores, 1 and 2 add (scalar f16x2 reductions), 3 adds and stores e4m3
    const u32 idx = tile * 8u + cg;
    if (z > 0u) {
        if (leader) {
            amp_wait_at_least(p.p4 + idx, (int)z - 1, budget);
            __threadfence();
        }
        __syncthreads();
    }
#pragma unroll
    for (u32 m = 0; m < 4; ++m) {
        const u32 grp = tile * 8u + th * 4u + m;
        if ((int)grp >= G) continue;
        u8* region = p.p5 + (u64)grp * 32768u + (out0 >> 4) * 512u + lane * 16u;    // 16 outputs per 512 B
#pragma unroll
        for (u32 f2 = 0; f2 < 4; ++f2) {
            u32* cell = (u32*)(region + f2 * 512u);
            u32* a = &acc[m][2u * f2][0];
            u32* b = &acc[m][2u * f2 + 1u][0];
            if (z == 0u) {
                st_v4((u8*)cell, a[0], a[1], b[0], b[1]);
            } else if (z < 3u) {
                amp_red_add_h2(cell + 0, a[0]); amp_red_add_h2(cell + 1, a[1]);
                amp_red_add_h2(cell + 2, b[0]); amp_red_add_h2(cell + 3, b[1]);
            } else {
                const amp_v4 v = amp_load_cg(cell);
                a[0] = amp_h2_add(v.x, a[0]); a[1] = amp_h2_add(v.y, a[1]);
                b[0] = amp_h2_add(v.z, b[0]); b[1] = amp_h2_add(v.w, b[1]);
            }
        }
        if (z == 3u) {
#pragma unroll
            for (u32 blk = 0; blk < 2; ++blk)
                store_act(p.p2 + (u64)grp * 16384u + ((out0 >> 5) + blk) * 512u + lane * 16u, acc[m] + 4u * blk);
        }
    }
    __threadfence();
    __syncthreads();
    if (leader) {
        amp_flag_publish(p.p4 + idx, (int)z);
        if (PUBLISH && z == 3u && p.p7) amp_flag_publish(p.p7 + idx, 0);
    }
}

extern "C" __global__ void __launch_bounds__(128, GEMM_MINB) amp_vit_ffn_contract_chained(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    merge_body<false, true>(p, sm);
}
extern "C" __global__ void __launch_bounds__(128, GEMM_MINB) amp_vit_ffn_contract_chained_after64(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    merge_body<false, true, 1>(p, sm);
}
extern "C" __global__ void __launch_bounds__(128, GEMM_MINB) amp_vit_projection_chained(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    merge_body<true, true>(p, sm);
}
extern "C" __global__ void __launch_bounds__(128, GEMM_MINB) amp_vit_projection_wait(const FfnParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    merge_body<true, false>(p, sm);
}

// =====================================================================================================================
// QKV (abi.md 5.1, 7.1): CTA (tile, cg = heads 2 cg, 2 cg + 1, z = input K half); warp (head 2 cg + (w & 1), tokens
// 64 (w >> 1) ..) x 96 outputs (q, k, v of the head: fragments 0-3, 4-7, 8-11). z = 0 stores its f16 partials to the
// scratch, z = 1 adds them and normalises: K = e4m3(k n(k)), Q = e4m3(((q n(q)) sqrt32) temperature), V = e4m3(v).
extern "C" __global__ void __launch_bounds__(128, 2) amp_vit_qkv_chained(const QkvParams p)
{
    __shared__ __align__(128) u8 sm[GEMM_NS * STAGE_BYTES];
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, tid = w * 32u + lane;
    const bool leader = tid == 0u;
    const int N = p.d0 * p.d1, T = tiles_of(N), G = groups_of(N);
    const u32 cg = blockIdx.x / (u32)T, tile = blockIdx.x - cg * (u32)T, z = blockIdx.z, th = w >> 1;
    const u32 head = 2u * cg + (w & 1u);
    u32 acc[4][12][2];
#pragma unroll
    for (u32 m = 0; m < 4; ++m)
#pragma unroll
        for (u32 n = 0; n < 12; ++n) acc[m][n][0] = acc[m][n][1] = 0u;
    int budget = AMP_WAIT_BUDGET_NS;
    const int* pf = p.pflags;
    auto wait = [&](int s) {                                    // contract flags, one per 128 input channels
        if (pf && (s & 1) == 0) wait_flag(pf + tile * 8u + 4u * z + (u32)(s >> 1), budget);
    };
    const u8* wb = p.w + WS * (128u + (u64)(16u * z) * 98304u + (6u * head) * 512u) + lane * 16u;
    gemm_loop<6>(acc, sm, p.x + z * 16u * 512u, 16384u, tile, G, 8, wb, WS * 98304u, wait, tid, th, lane, leader);

    const u32 idx = tile * 16u + cg;
    const u64 region = (u64)G * 32768u;
    if (z == 0u) {
#pragma unroll
        for (u32 m = 0; m < 4; ++m) {
            const u32 grp = tile * 8u + th * 4u + m;
            if ((int)grp >= G) continue;
#pragma unroll
            for (u32 which = 0; which < 3; ++which)
#pragma unroll
                for (u32 f2 = 0; f2 < 2; ++f2) {
                    const u32* a = acc[m][4u * which + 2u * f2];
                    const u32* b = acc[m][4u * which + 2u * f2 + 1u];
                    st_v4(p.scratch + which * region + (u64)grp * 32768u + (2u * head + f2) * 512u + lane * 16u,
                          a[0], a[1], b[0], b[1]);
                }
        }
    } else {
        if (leader) {
            amp_wait_at_least(p.counter + idx, 0, budget);
            __threadfence();
        }
        __syncthreads();
        const u32 floor_h = h2_dup(__uint_as_float(FLOOR_BITS));
        const u32 sq32 = h2_dup(sqrt_approx(32.0f));
        const u32 temp = h2_dup(((const float*)p.w)[head]);
#pragma unroll
        for (u32 pass = 0; pass < 2; ++pass) {                  // 32-token blocks of the value layout: m = 2 pass, + 1
            u32 T8[2][2][4];
#pragma unroll
            for (u32 mm = 0; mm < 2; ++mm) {
                const u32 m = 2u * pass + mm;
                const u32 grp = tile * 8u + th * 4u + m;
                const bool here = (int)grp < G;
                if (here) {
#pragma unroll
                    for (u32 which = 0; which < 3; ++which)
#pragma unroll
                        for (u32 f2 = 0; f2 < 2; ++f2) {
                            u32* a = acc[m][4u * which + 2u * f2];
                            u32* b = acc[m][4u * which + 2u * f2 + 1u];
                            const amp_v4 v = amp_load_cg(p.scratch + which * region + (u64)grp * 32768u
                                                         + (2u * head + f2) * 512u + lane * 16u);
                            a[0] = amp_h2_add(v.x, a[0]); a[1] = amp_h2_add(v.y, a[1]);
                            b[0] = amp_h2_add(v.z, b[0]); b[1] = amp_h2_add(v.w, b[1]);
                        }
                    // norms of q (fragments 0-3) and k (4-7) per row. PTX order ((c1^2 + c3^2) + (c0^2 + c2^2)), bfly 2,
                    // bfly 1, lo + hi; NVIDIA's compiled kernel contracts c1^2 + . and c0^2 + . into fma (one rounding),
                    // which is what the runtime computes (found by bit-matching its K/Q codes on real data)
                    u32 nrm[2][2];
#pragma unroll
                    for (u32 qk = 0; qk < 2; ++qk)
#pragma unroll
                        for (u32 r = 0; r < 2; ++r) {
                            const u32 c0 = acc[m][4u * qk][r], c1 = acc[m][4u * qk + 1u][r];
                            const u32 c2 = acc[m][4u * qk + 2u][r], c3 = acc[m][4u * qk + 3u][r];
                            u32 S = amp_h2_add(amp_h2_fma(c1, c1, amp_h2_mul(c3, c3)), amp_h2_fma(c0, c0, amp_h2_mul(c2, c2)));
                            S = amp_h2_add(S, __shfl_xor_sync(0xffffffffu, S, 2));
                            S = amp_h2_add(S, __shfl_xor_sync(0xffffffffu, S, 1));
                            S = amp_h2_add(S, h2_swap(S));
                            nrm[qk][r] = rsqrt_h2(amp_h2_max(S, floor_h));
                        }
                    u32 Q[4][2], K[4][2];
#pragma unroll
                    for (u32 n = 0; n < 4; ++n)
#pragma unroll
                        for (u32 r = 0; r < 2; ++r) {
                            Q[n][r] = amp_h2_mul(amp_h2_mul(amp_h2_mul(acc[m][n][r], nrm[0][r]), sq32), temp);
                            K[n][r] = amp_h2_mul(acc[m][4u + n][r], nrm[1][r]);
                        }
                    // Q: act layout; K: key layout (words: {N0 r, N1 r}, {N2 r, N3 r}, {N0 r+8, N1 r+8}, {N2 r+8, N3 r+8})
                    st_v4(p.q + (u64)grp * 16384u + head * 512u + lane * 16u,
                          e8w(Q[0][0], Q[1][0]), e8w(Q[0][1], Q[1][1]), e8w(Q[2][0], Q[3][0]), e8w(Q[2][1], Q[3][1]));
                    st_v4(p.k + (u64)grp * 16384u + head * 512u + lane * 16u,
                          e8w(K[0][0], K[1][0]), e8w(K[2][0], K[3][0]), e8w(K[0][1], K[1][1]), e8w(K[2][1], K[3][1]));
                }
                // V transposed per 8 x 8 (warp-wide: executed whether or not this group exists; G is even, so the pair
                // of groups of one 32-token block is either complete or absent)
#pragma unroll
                for (u32 r = 0; r < 2; ++r)
#pragma unroll
                    for (u32 f = 0; f < 4; ++f) T8[mm][r][f] = amp_transpose8(acc[m][8u + f][r]);
            }
            const u32 grp0 = tile * 8u + th * 4u + 2u * pass;
            if ((int)grp0 < G) {
#pragma unroll
                for (u32 sl = 0; sl < 2; ++sl) {
                    u32 word[4];
#pragma unroll
                    for (u32 d = 0; d < 2; ++d)
#pragma unroll
                        for (u32 mm = 0; mm < 2; ++mm)
                            word[d * 2u + mm] = e8w(T8[mm][0][2u * sl + d], T8[mm][1][2u * sl + d]);
                    st_v4(p.v + (u64)(tile * 4u + th * 2u + pass) * 32768u + (2u * head + sl) * 512u + lane * 16u,
                          word[0], word[1], word[2], word[3]);
                }
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (leader) {
        amp_flag_publish(p.counter + idx, (int)z);
        if (z == 1u && p.oflags) amp_flag_publish(p.oflags + idx, 0);
    }
}

// =====================================================================================================================
// Attention (abi.md 5.2, 7.2): CTA (head, 256 queries = tiles 2 y, 2 y + 1), warp 64 queries; key groups of 64
// double-buffered in shared memory (cp.async), polled tile by tile behind the QKV flags. Per group: S = Q K^T (one
// k32 step from zero), the network's exponential in f16, the fixed f16 group-sum order into the denominator,
// P = e4m3(p) (kept as f16 values on the e4m3 grid), acc += P V (f16 accumulator over all keys); then the padding
// keys out of the denominator, f16(rcp.approx), out = e4m3(acc * inv), rows >= N zero.
// MW = 16-query groups per warp: 4 (NVIDIA's CTA: two 128-query tiles, grid (32, ceil(T / 2))) or 2 (BRIEF-11 '1t': one
// tile per CTA, grid (32, T): twice the CTAs, so small token counts fill the GPU; the K / V of a head is read twice as often).
template <u32 MW>
AMP_INLINE void att_body(const AttParams& p, u8 (*sm_kv)[4096 * ATT_SG], const u32 y)
{
    constexpr u32 TPC = MW / 2u;                                // 128-query tiles per CTA
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, g = lane >> 2, t = lane & 3u;
    const u32 tid = w * 32u + lane;
    const bool leader = tid == 0u;
    const int N = p.d0 * p.d1, T = tiles_of(N), G = groups_of(N), KG = (N - 1) / 64 + 1;
    const u32 head = blockIdx.x;
    int budget = AMP_WAIT_BUDGET_NS;
    if ((int)(TPC * y) >= T) return;                            // grid y rounded up past the last tile (whole CTA)

    if (p.pflags) {                                             // Q of this CTA's tiles, K and V of stages 0 and 1
        if (leader) {
            for (int tt = 0; tt < min(ATT_SG, T); ++tt) amp_wait_not_negative(p.pflags + tt * 16 + (head >> 1), budget);
            for (int tt = TPC * (int)y; tt < min(TPC * (int)y + TPC, T); ++tt) amp_wait_not_negative(p.pflags + tt * 16 + (head >> 1), budget);
            __threadfence();
        }
        __syncthreads();
    }
    // Q: the f16 A fragments (two k16 steps) of the warp's four 16-query groups
    u32 QA0[MW][4], QA1[MW][4];
#pragma unroll
    for (u32 m = 0; m < MW; ++m) {
        const u32 grp = (4u * MW) * y + MW * w + m;
        amp_v4 v = {0u, 0u, 0u, 0u};
        if ((int)grp < G) v = amp_load_cg(p.q + (u64)grp * 16384u + head * 512u + lane * 16u);
        const u32 a8[4] = {v.x, v.y, v.z, v.w};
        frag_a(a8, QA0[m], QA1[m]);
    }

    const int NS = (KG + ATT_SG - 1) / ATT_SG;
    auto issue = [&](int s, u32 stage) {
#pragma unroll
        for (u32 r = 0; r < 2u * ATT_SG; ++r) {
            const u32 c = tid + 128u * r, u = c >> 8, part = (c >> 5) & 7u, off = (c & 31u) * 16u;
            const u32 i = (u32)s * ATT_SG + u;
            const u32 dst = amp_smem(&sm_kv[stage][u * 4096u + part * 512u + off]);
            const u32 kgrp = 4u * i + part;                     // K: 16-key groups 4 i .. 4 i + 3
            const u32 vblk = 2u * i + ((part - 4u) >> 1), vsl = 2u * head + ((part - 4u) & 1u);
            const bool inside = (int)i < KG && (part < 4u ? (int)kgrp < G : (int)(2u * vblk) < G);
            if (inside) {
                const u8* src = part < 4u ? p.k + (u64)kgrp * 16384u + head * 512u + off
                                          : p.v + (u64)vblk * 32768u + vsl * 512u + off;
                amp_cp_async16(dst, src);
            } else {
                amp_cp_async16_zfill(dst, p.k, 0u);
            }
        }
        amp_cp_async_commit();
    };
    issue(0, 0u);
    if (NS > 1) issue(1, 1u);

    u32 den[MW / 2];                                                 // f16 pair (both halves equal) of row (set, t)
#pragma unroll
    for (u32 i = 0; i < MW / 2; ++i) den[i] = 0u;
    u32 acc[MW][4][2];                                           // f16 accumulators [m][dims 8 f][row half]
#pragma unroll
    for (u32 m = 0; m < MW; ++m)
#pragma unroll
        for (u32 f = 0; f < 4; ++f) acc[m][f][0] = acc[m][f][1] = 0u;
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const u32 stage = (u32)s & 1u;
        if (s + 1 < NS) amp_cp_async_wait<1>(); else amp_cp_async_wait<0>();
        __syncthreads();
#pragma unroll 1
        for (int u = 0; u < ATT_SG; ++u) {
            if (s * ATT_SG + u >= KG) break;
            const u8* ks_ = sm_kv[stage] + u * 4096;
            const u8* vs_ = ks_ + 2048u;
            u32 S[MW][8][2];                                     // logits [m][key fragment n][row half], f16 pairs
#pragma unroll
            for (u32 jj = 0; jj < 4; ++jj) {
                const amp_v4 kv = *(const amp_v4*)(ks_ + jj * 512u + lane * 16u);
                const u32 kl[2] = {kv.x, kv.y}, kh[2] = {kv.z, kv.w};
                u32 B0[2], B1[2], C0[2], C1[2];
                frag_kv(kl, B0, B1);
                frag_kv(kh, C0, C1);
#pragma unroll
                for (u32 m = 0; m < MW; ++m) {
                    S[m][2u * jj][0] = S[m][2u * jj][1] = 0u;
                    S[m][2u * jj + 1u][0] = S[m][2u * jj + 1u][1] = 0u;
                    step32(S[m][2u * jj], QA0[m], QA1[m], B0, B1);
                    step32(S[m][2u * jj + 1u], QA0[m], QA1[m], C0, C1);
                }
            }
            // the network's exponential: fma, clamp, mantissa << 4 read as a float16
            u32 X[2 * MW];                                           // row partial sums [m * 2 + r] (keys 8 n + 2 t + e)
#pragma unroll
            for (u32 m = 0; m < MW; ++m)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
#pragma unroll
                    for (u32 n = 0; n < 8; ++n) {
                        u32 x = amp_h2_fma(S[m][n][r], 0x2DBB2DBBu, 0x3ED63ED6u);
                        x = amp_h2_min(amp_h2_max(x, 0x3DC23DC2u), 0x3FE93FE9u);
                        S[m][n][r] = (x << 4) + 0x3FFC4000u;
                    }
                    X[m * 2u + r] = amp_h2_add(amp_h2_add(amp_h2_add(amp_h2_add(S[m][0][r], S[m][1][r]),
                                                                     amp_h2_add(S[m][2][r], S[m][3][r])),
                                                          amp_h2_add(S[m][4][r], S[m][5][r])),
                                               amp_h2_add(S[m][6][r], S[m][7][r]));
                }
            // the four lanes t in order ((t0 + t1) + t2) + t3: lane t ends with row index t of each set of four
#pragma unroll
            for (u32 set = 0; set < MW / 2; ++set) {
                const u32* x = X + 4u * set;
                u32 term[4];
#pragma unroll
                for (u32 k = 0; k < 4; ++k) {
                    const u32 send = (t ^ k) == 0u ? x[0] : (t ^ k) == 1u ? x[1] : (t ^ k) == 2u ? x[2] : x[3];
                    term[k] = k ? __shfl_xor_sync(0xffffffffu, send, k) : send;
                }
                u32 by[4];
#pragma unroll
                for (u32 s2 = 0; s2 < 4; ++s2) {
                    const u32 k = s2 ^ t;
                    by[s2] = k == 0u ? term[0] : k == 1u ? term[1] : k == 2u ? term[2] : term[3];
                }
                const u32 sum = amp_h2_add(amp_h2_add(amp_h2_add(by[0], by[1]), by[2]), by[3]);
                den[set] = amp_h2_add(den[set], amp_h2_add(sum, h2_swap(sum)));
            }
            // P (on the e4m3 grid, as f16) x V. NVIDIA's k32 A word {e4m3 S[n] row r, e4m3 S[n+1] row r} decodes to the
            // k16 steps {P[n][0], P[n][1], P[n+1][0], P[n+1][1]} and {P[n+2][0], P[n+2][1], P[n+3][0], P[n+3][1]}.
#pragma unroll
            for (u32 m = 0; m < MW; ++m)
#pragma unroll
                for (u32 n = 0; n < 8; ++n)
#pragma unroll
                    for (u32 r = 0; r < 2; ++r) S[m][n][r] = snap_e4m3_pos(S[m][n][r]);
#pragma unroll
            for (u32 kh = 0; kh < 2; ++kh) {
                u32 VB[4][2][2];                                // [dims fragment f][k16 step][2]
#pragma unroll
                for (u32 sl = 0; sl < 2; ++sl) {
                    const amp_v4 vv = *(const amp_v4*)(vs_ + (kh * 2u + sl) * 512u + lane * 16u);
                    const u32 vl[2] = {vv.x, vv.y}, vh[2] = {vv.z, vv.w};
                    frag_kv(vl, VB[2u * sl][0], VB[2u * sl][1]);
                    frag_kv(vh, VB[2u * sl + 1u][0], VB[2u * sl + 1u][1]);
                }
#pragma unroll
                for (u32 m = 0; m < MW; ++m) {
                    const u32 n = 4u * kh;
                    const u32 P0[4] = {S[m][n][0], S[m][n][1], S[m][n + 1u][0], S[m][n + 1u][1]};
                    const u32 P1[4] = {S[m][n + 2u][0], S[m][n + 2u][1], S[m][n + 3u][0], S[m][n + 3u][1]};
#pragma unroll
                    for (u32 f = 0; f < 4; ++f) step32(acc[m][f], P0, P1, VB[f][0], VB[f][1]);
                }
            }
        }
        __syncthreads();                                        // everybody is done with this stage
        if (s + 2 < NS) {
            if (p.pflags) {
                if (leader) {
                    const int i0 = (s + 2) * ATT_SG, i1 = min(KG, i0 + ATT_SG);
                    for (int tt = i0 >> 1; tt <= (i1 - 1) >> 1; ++tt) amp_wait_not_negative(p.pflags + tt * 16 + (head >> 1), budget);
                    __threadfence();
                }
                __syncthreads();
            }
            issue(s + 2, stage);
        }
    }
    // denominators: padding keys out, reciprocal, to the lanes that hold each row
    const float pad = f16_round((float)(64 * KG - N) * 0.083984375f);
    float inv_row[MW / 2];
#pragma unroll
    for (u32 set = 0; set < MW / 2; ++set) {
        float lo, hi;
        amp_h2_unpack(den[set], lo, hi);
        const float d = f16_round(lo - pad);
        inv_row[set] = f16_round(amp_rcp_approx(fmaxf(d, __uint_as_float(0x38820000u))));
    }
    float inv[MW][2];
#pragma unroll
    for (u32 m = 0; m < MW; ++m)
#pragma unroll
        for (u32 r = 0; r < 2; ++r)
            inv[m][r] = __shfl_sync(0xffffffffu, inv_row[m >> 1], (lane & ~3u) | (((m & 1u) << 1) | r));
#pragma unroll
    for (u32 m = 0; m < MW; ++m) {
        const u32 grp = (4u * MW) * y + MW * w + m;
        if ((int)grp >= G) continue;
        u32 word[2][2];                                         // [dims 16 half][row half]
#pragma unroll
        for (u32 r = 0; r < 2; ++r) {
            const bool valid = (int)(grp * 16u + g + 8u * r) < N;
            const u32 iv = h2_dup(inv[m][r]);
#pragma unroll
            for (u32 fh = 0; fh < 2; ++fh) {
                const u32 wv = e8w(amp_h2_mul(acc[m][2u * fh][r], iv), amp_h2_mul(acc[m][2u * fh + 1u][r], iv));
                word[fh][r] = valid ? wv : 0u;
            }
        }
        st_v4(p.out + (u64)grp * 16384u + head * 512u + lane * 16u, word[0][0], word[0][1], word[1][0], word[1][1]);
    }
    __threadfence();
    __syncthreads();
    if (leader && p.oflags) {
#pragma unroll
        for (u32 i = 0; i < TPC; ++i)
            if ((int)(TPC * y + i) < T) amp_flag_publish(p.oflags + (TPC * y + i) * 32u + head, 0);
    }
}
extern "C" __global__ void __launch_bounds__(128, 2) amp_vit_attention_chained(const AttParams p)
{
    __shared__ __align__(128) u8 sm_kv[2][4096 * ATT_SG];      // [stage][group][K 4 x 512 | V 4 x 512]
    att_body<4>(p, sm_kv, blockIdx.y);
}
extern "C" __global__ void __launch_bounds__(128, 3) amp_vit_attention1t_chained(const AttParams p)
{
    __shared__ __align__(128) u8 sm_kv[2][4096 * ATT_SG];
    att_body<2>(p, sm_kv, blockIdx.y);
}
// Either split, chosen on the GPU from the SM count: grid (32, 2 ceil(T / 2)) (manifest 'grid = *2 ='). One tile per CTA
// when those 32 T CTAs fit in one wave of 2 per SM (1440p and below on the RTX 5090), else NVIDIA's two tiles per CTA
// (the odd rows of the grid exit at once). Same arithmetic per query either way.
extern "C" __global__ void __launch_bounds__(128, 2) amp_vit_attention_auto_chained(const AttParams p)
{
    __shared__ __align__(128) u8 sm_kv[2][4096 * ATT_SG];
    u32 sms;
    asm("mov.u32 %0, %%nsmid;" : "=r"(sms));
    const int T = tiles_of(p.d0 * p.d1);
    if (32u * (u32)T <= 2u * sms) {
        att_body<2>(p, sm_kv, blockIdx.y);
    } else {
        if (blockIdx.y & 1u) return;
        att_body<4>(p, sm_kv, blockIdx.y >> 1);
    }
}

#if AMP_PREPARED
// NrHost prepare ABI (NrHost.cpp PrepareParams). Records: expand 4 MB matrix + 16 B; contract 4 MB + 2 KB gain;
// projection 1 MB + 2 KB gain; QKV 128 B (head temperatures) + 3 MB matrix. Matrices -> f16 slices, the rest copied.
struct PrepareParams { const u8* in; u8* out; u32 inBytes, outBytes; int launch, pad; };
static_assert(sizeof(PrepareParams) == 32, "NrHost prepare ABI");
AMP_INLINE void prep_matrix(const PrepareParams& p, u32 off, u32 bytes)
{
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 16u; i += blockDim.x * gridDim.x) {
        const amp_v4 v = amp_load_nc(p.in + off + 16u * i);
        u8* const to = p.out + 2u * off + (i / 32u) * 1024u + (i & 31u) * 16u;
        amp_v4 a, b;
        amp_e4m3x4_to_h2x2_finite(v.x, a.x, a.y);
        amp_e4m3x4_to_h2x2_finite(v.y, a.z, a.w);
        amp_e4m3x4_to_h2x2_finite(v.z, b.x, b.y);
        amp_e4m3x4_to_h2x2_finite(v.w, b.z, b.w);
        *(amp_v4*)to = a;
        *(amp_v4*)(to + 512u) = b;
    }
}
AMP_INLINE void prep_raw(const PrepareParams& p, u32 off, u32 bytes)
{
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 4u; i += blockDim.x * gridDim.x)
        *(u32*)(p.out + 2u * off + 4u * i) = *(const u32*)(p.in + off + 4u * i);
}
extern "C" __global__ void amp_prepare_vit(const PrepareParams p)
{
    if (p.outBytes < 2u * p.inBytes) __trap();
    if (p.inBytes == 3145856u) {                                // QKV
        prep_raw(p, 0u, 128u);
        prep_matrix(p, 128u, 3145728u);
        return;
    }
    const u32 m = p.inBytes >= 4194304u ? 4194304u : 1048576u;  // expand / contract : projection
    if (p.inBytes != m + 16u && p.inBytes != m + 2048u) __trap();
    prep_matrix(p, 0u, m);
    prep_raw(p, m, p.inBytes - m);
}
#endif

// =====================================================================================================================
// Repacks (weightless word shuffles, vit-spec/work/repack.txt): 4-byte words, 256 per token (1024 channels). The 1D
// (act) layout groups 16 consecutive tokens; the 2D layout groups 4 x 4 token patches of the map (token = y d1 + x).
// Same index arithmetic as the runtime's kernels for every valid (d0, d1 > 0); grid-stride loops over all words.
AMP_INLINE int word_in_group(int row16, int j)                 // row16 = token index inside its group of 16
{
    const int c = 4 * j, cc = c & 31, h = cc >> 4, c16 = cc & 15, kb = j >> 3;
    return kb * 128 + (row16 >> 3) + 2 * h + (((row16 & 7) * 16 + c16) >> 2) * 4;
}

extern "C" __global__ void amp_vit_repack_2d_to_1d(const RepackParams p)
{
    const int n = p.d0 * p.d1;
    const int words = n == 0 ? 0 : ((n - 1) / 32) * 8192 + 8192;     // tokens rounded up to 32, rows >= n get zeros
    const int stride = (int)(gridDim.x * blockDim.x);
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < words; i += stride) {
        const int j = i & 255, tok = i >> 8;
        u32 value = 0u;
        if (tok < n) {
            const int yy = tok / p.d1, xx = tok - yy * p.d1;
            const int py = (p.d0 & ~3) == 4 ? 0 : yy >> 2, px = (p.d1 & ~3) == 4 ? 0 : xx >> 2;
            const int pos = (yy & 3) * 4 + (xx & 3);
            value = p.in[(py * (p.d1 / 4) + px) * 4096 + word_in_group(pos, j)];
        }
        p.out[(i & ~4095) + word_in_group(tok & 15, j)] = value;
    }
}

extern "C" __global__ void amp_vit_repack_1d_to_2d(const RepackParams p)
{
    const int words = p.d0 * p.d1 * 256;
    const int stride = (int)(gridDim.x * blockDim.x);
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < words; i += stride) {
        const int j = i & 255, tok = i >> 8;
        const int yy = tok / p.d1, xx = tok - yy * p.d1;
        const int pos = (yy & 3) * 4 + (xx & 3);
        const u32 value = p.in[(i & ~4095) + word_in_group(tok & 15, j)];
        p.out[((yy >> 2) * (p.d1 / 4) + (xx >> 2)) * 4096 + word_in_group(pos, j)] = value;
    }
}
