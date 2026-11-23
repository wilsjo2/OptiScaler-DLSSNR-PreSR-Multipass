// Ampere-legal float16 replacement of the DLSS-NR fused Swin blocks cc_tinlayout_fused_swin_{1h_32,2h_64,4h_128,8h_256}_*
// (the five runtime variants inpview_tilesync, chained, ds_wait, upsample_tilesync, outview_wait at each width).
//
// Takes exactly the runtime's own launch: NVIDIA's parameter block (96 bytes at width 32, 88 bytes at widths 64/128/256),
// NVIDIA's e4m3 weight record unchanged, the same activation layouts, block (32, HEADS), one CTA per 8x8-token window,
// grid ceil((W - min(sx, 0)) / 8) x ceil((H - min(sy, 0)) / 8), no dynamic shared memory, the same flag protocol.
//
// Arithmetic: NVIDIA's order, with each FP8 tensor-core step (mma.m16n8k32 e4m3, float16 accumulate) replaced by two
// float16 steps (mma.m16n8k16.f16.f16.f16.f16) on operands decoded exactly from the e4m3 codes, and every e4m3 store
// done by the software round-to-nearest-even satfinite of amp_common.cuh. Nothing else differs, so the outputs agree with
// NVIDIA's kernel except where the k16 pair rounds its float16 accumulator differently from the k32 instruction.
//
// The sm_120 NVFP4 host builds its 32- and 64-channel E4M3 kernels with NR_NATIVE_FP8=1
// and a 168-register limit.
// AMP_HW_REF selects the same k32 arithmetic for individual reference builds.
//
// Switches the hardware reference settles (defaults = what it showed):
//   AMP_UP_FMA       upsample blend skip * gain + projection: the runtime's PTX has mul.f16x2 then add.f16x2 (no .rn) and
//                    ptxas fuses them: 1 one fma.rn (shown 100 % at all four widths), 0 two roundings
//   AMP_UP_ZERO_OUT  upsample: the blend of tiles outside the map is zeroed (1) or kept (0); the runtime's multi-head PTX
//                    zeroes it (per pixel against H, W), the single-head PTX does not (default HEADS > 1; no lab input
//                    has shown a difference: those tiles' blend is already zero there)
//   AMP_NORM_FMA     Q/K norm: the runtime's PTX squares and adds (mul.f16x2, add.f16x2 without .rn) and ptxas fuses
//                    the first square of each pair into the add: 1 fma(q0, q0, q2 * q2) (shown 100 %), 2 the other, 0 none
//
// One warp is one head; lane (g = lane >> 2, t = lane & 3) holds tokens g and g + 8 of each 4x4-token tile. Within
// a channel block of 32, channel 16a + 8b + 2t + e sits at K = 16a + 4t + 2b + e, so an accumulator's n-fragments j
// (channels 8j + 2t + e) pack into the A operand of the next product without leaving the lane:
//   word 0 = (nf0, nf1) of row g, word 1 = (nf0, nf1) of row g + 8, word 2 = (nf2, nf3) of row g, word 3 = row g + 8.
//
// Variants (name = <width>_<runtime variant>):
// amp-variant: 32_inpview_tilesync  -DHEADS=1 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 --maxrregcount=128
// amp-variant: 32_chained           -DHEADS=1 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 --maxrregcount=128
// amp-variant: 32_ds_wait           -DHEADS=1 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 --maxrregcount=128
// amp-variant: 32_upsample_tilesync -DHEADS=1 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 --maxrregcount=128
// amp-variant: 32_outview_wait      -DHEADS=1 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 --maxrregcount=128
// amp-variant: 64_inpview_tilesync  -DHEADS=2 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 --maxrregcount=168
// amp-variant: 64_chained           -DHEADS=2 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 --maxrregcount=168
// amp-variant: 64_ds_wait           -DHEADS=2 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 --maxrregcount=168
// amp-variant: 64_upsample_tilesync -DHEADS=2 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 --maxrregcount=168
// amp-variant: 64_outview_wait      -DHEADS=2 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 --maxrregcount=168
// amp-variant: 128_inpview_tilesync  -DHEADS=4 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 --maxrregcount=168
// amp-variant: 128_chained           -DHEADS=4 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 --maxrregcount=168
// amp-variant: 128_ds_wait           -DHEADS=4 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 --maxrregcount=168
// amp-variant: 128_upsample_tilesync -DHEADS=4 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 --maxrregcount=168
// amp-variant: 128_outview_wait      -DHEADS=4 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 --maxrregcount=168
// amp-variant: 256_inpview_tilesync  -DHEADS=8 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 --maxrregcount=128
// amp-variant: 256_chained           -DHEADS=8 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 --maxrregcount=128
// amp-variant: 256_ds_wait           -DHEADS=8 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 --maxrregcount=128
// amp-variant: 256_upsample_tilesync -DHEADS=8 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 --maxrregcount=128
// amp-variant: 256_outview_wait      -DHEADS=8 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 --maxrregcount=128
//
// _exact builds (AMP_F32ACC=1, still Ampere-legal; not the phase-A target, same entries, blocks and parameters): each
// k32 step as two m16n8k16 steps with an f32 accumulator between them, rounded once to float16 per k32 step.
// amp-variant: 32_inpview_tilesync_exact  -DHEADS=1 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 32_chained_exact           -DHEADS=1 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 32_ds_wait_exact           -DHEADS=1 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 32_upsample_tilesync_exact -DHEADS=1 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 32_outview_wait_exact      -DHEADS=1 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 64_inpview_tilesync_exact  -DHEADS=2 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 64_chained_exact           -DHEADS=2 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 64_ds_wait_exact           -DHEADS=2 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 64_upsample_tilesync_exact -DHEADS=2 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 64_outview_wait_exact      -DHEADS=2 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 128_inpview_tilesync_exact -DHEADS=4 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 128_chained_exact          -DHEADS=4 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 128_ds_wait_exact          -DHEADS=4 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 128_upsample_tilesync_exact -DHEADS=4 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 128_outview_wait_exact     -DHEADS=4 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=168
// amp-variant: 256_inpview_tilesync_exact -DHEADS=8 -DVARIANT=1 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 256_chained_exact          -DHEADS=8 -DVARIANT=0 -DWAIT=1 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 256_ds_wait_exact          -DHEADS=8 -DVARIANT=2 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 256_upsample_tilesync_exact -DHEADS=8 -DVARIANT=3 -DWAIT=0 -DPUBLISH=1 -DAMP_F32ACC=1 --maxrregcount=128
// amp-variant: 256_outview_wait_exact     -DHEADS=8 -DVARIANT=4 -DWAIT=1 -DPUBLISH=0 -DAMP_F32ACC=1 --maxrregcount=128
#include "amp_common.cuh"

#ifndef HEADS
#define HEADS 8
#endif
// 0: tile-layout input and output (chained); 1: view-layout input (inpview); 2: plus 2x2 mean and C -> 2C projection
// (ds); 3: 2C -> C projection of the coarser level, 2x replication, blend with the skip (upsample); 4: view-layout
// output (outview)
#ifndef VARIANT
#define VARIANT 0
#endif
#ifndef WAIT
#define WAIT 0
#endif
#ifndef PUBLISH
#define PUBLISH 0
#endif
#ifndef AMP_HW_REF
#define AMP_HW_REF NR_NATIVE_FP8
#endif
#ifndef AMP_F32ACC
#define AMP_F32ACC 0
#endif
#ifndef AMP_UP_FMA
#define AMP_UP_FMA 1
#endif
#ifndef AMP_NORM_FMA
#define AMP_NORM_FMA 1
#endif
#ifndef AMP_UP_ZERO_OUT
#define AMP_UP_ZERO_OUT (HEADS > 1)
#endif
#define C (32u * HEADS)
#define NB ((u32)HEADS)                         // K blocks of 32 channels over C

// ---- NVIDIA's parameter block ---------------------------------------------------------------------------------------
#if HEADS == 1
struct Params {                                 // 96 bytes
    const u8* in; u8* out; const u8* rec;
    int H, W, sx, sy;
    const int* pflags; int psx, psy;
    int* oflags;
    u8* ds_out; int d0, d1;                     // ds padded dims / view dims override
    const u8* skip; int e0, e1;                 // upsample: skip map and its (rows, cols) if > 0
};
static_assert(sizeof(Params) == 96, "parameter block");
#else
struct Params {                                 // 88 bytes
    const u8* in; u8* out; const u8* rec; const u8* skip;
    int H, W, sx, sy;
    const int* pflags; int psx, psy;
    int* oflags;
    u8* ds_out; int d0, d1;
};
static_assert(sizeof(Params) == 88, "parameter block");
#endif

// ---- NVIDIA's record offsets ----------------------------------------------------------------------------------------
#if HEADS == 1
#define R_EXPAND 0u
#define R_CONTRACT 4096u
#if VARIANT == 3
#define R_RESAMPLE 8192u
#define R_RS 10256u
#define R_GAIN 10336u
#define R_QKV 10400u
#define R_BIAS 13472u
#define R_SCAL 21664u
#define R_OUT 21680u
#define R_ARS 22704u
#else
#define R_RS 8208u
#define R_QKV 8288u
#define R_BIAS 11360u
#define R_SCAL 19552u
#define R_OUT 19568u
#define R_ARS 20592u
#define R_RESAMPLE 20656u
#endif
#elif HEADS == 2
#define R_EXPAND 0u
#define R_MID 16384u
#define R_CONTRACT 24576u
#if VARIANT == 3
#define R_RESAMPLE 28672u
#define R_RS 36864u
#define R_GAIN 36992u
#define R_QKV 37120u
#define R_BIAS 49408u
#define R_SCAL 65792u
#define R_OUT 65808u
#define R_ARS 69904u
#else
#define R_RS 28688u
#define R_QKV 28832u
#define R_BIAS 41120u
#define R_SCAL 57504u
#define R_OUT 57520u
#define R_ARS 61616u
#define R_RESAMPLE 61744u
#endif
#elif HEADS == 4
#define R_EXPAND 0u
#define R_MID 65536u
#define R_CONTRACT 81920u
#if VARIANT == 3
#define R_RESAMPLE 98304u
#define R_RS 131072u
#define R_GAIN 131328u
#define R_QKV 131584u
#define R_BIAS 180736u
#define R_SCAL 213504u
#define R_OUT 213520u
#define R_ARS 229904u
#else
#define R_RS 98320u
#define R_QKV 98592u
#define R_BIAS 147744u
#define R_SCAL 180512u
#define R_OUT 180528u
#define R_ARS 196912u
#define R_RESAMPLE 197168u
#endif
#elif HEADS == 8
#define R_EXPAND 0u
#define R_MID 262144u
#define R_CONTRACT 294912u
#if VARIANT == 3
#define R_RESAMPLE 360448u
#define R_RS 491520u
#define R_GAIN 492032u
#define R_QKV 492544u
#define R_BIAS 689152u
#define R_SCAL 754688u
#define R_OUT 754720u
#define R_ARS 820256u
#else
#define R_RS 360464u
#define R_QKV 360992u
#define R_BIAS 557600u
#define R_SCAL 623136u
#define R_OUT 623168u
#define R_ARS 688704u
#define R_RESAMPLE 689216u
#endif
#else
#error HEADS must be 1, 2, 4 or 8
#endif

// float16 constants exactly as the runtime makes them (cvt.rn.f16.f32 of these float bit patterns), both halves
#define K_ACT_HI 1082130432u        //  4.0
#define K_ACT_LO 3229614080u        // -4.0
#define K_ACT_A 3177512960u         // -0.055908203125
#define K_ACT_B 1055195136u         //  0.447265625
#define K_ACT_C 1063583744u         //  0.89453125
#define K_FLOOR 948045311u          // norm and softmax-denominator floor
#define K_EXP_A 1027077105u
#define K_EXP_B 1067877303u
#define K_EXP_LO 1065615360u
#define K_EXP_HI 1070129152u
AMP_INLINE u32 hconst(u32 bits) { const float f = __uint_as_float(bits); return amp_h2_pack(f, f); }

// ---- the product: one k32 step of NVIDIA's FP8 instruction ----------------------------------------------------------
#if AMP_HW_REF
struct AF { u32 r[4]; };
struct BF { u32 r[2]; };
AMP_INLINE AF afrag(u32 a0, u32 a1, u32 a2, u32 a3) { AF f; f.r[0] = a0; f.r[1] = a1; f.r[2] = a2; f.r[3] = a3; return f; }
AMP_INLINE BF bfrag(u32 b0, u32 b1) { BF f; f.r[0] = b0; f.r[1] = b1; return f; }
AMP_INLINE void mma(u32* c, const AF& a, const BF& b)
{
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(c[0]), "+r"(c[1]) : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]), "r"(b.r[0]), "r"(b.r[1]));
}
#else
struct AF { u32 k0[4], k1[4]; };
struct BF { u32 k0[2], k1[2]; };
AMP_INLINE AF afrag(u32 a0, u32 a1, u32 a2, u32 a3)
{
    AF f;
    amp_e4m3x4_to_h2x2_finite(a0, f.k0[0], f.k0[2]);
    amp_e4m3x4_to_h2x2_finite(a1, f.k0[1], f.k0[3]);
    amp_e4m3x4_to_h2x2_finite(a2, f.k1[0], f.k1[2]);
    amp_e4m3x4_to_h2x2_finite(a3, f.k1[1], f.k1[3]);
    return f;
}
AMP_INLINE BF bfrag(u32 b0, u32 b1)
{
    BF f;
    amp_e4m3x4_to_h2x2_finite(b0, f.k0[0], f.k0[1]);
    amp_e4m3x4_to_h2x2_finite(b1, f.k1[0], f.k1[1]);
    return f;
}
AMP_INLINE void mma16(u32* c, const u32* a, const u32* b)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
AMP_INLINE void mma(u32* c, const AF& a, const BF& b)
{
#if AMP_F32ACC
    // the _exact builds: an f32 accumulator between the two k16 steps, one rounding to float16 per k32 step (as the
    // k32 instruction rounds once); half the tensor rate on GeForce Ampere
    float f[4];
    amp_h2_unpack(c[0], f[0], f[1]);
    amp_h2_unpack(c[1], f[2], f[3]);
    amp_mma16_f32(f, a.k0, b.k0);
    amp_mma16_f32(f, a.k1, b.k1);
    c[0] = amp_h2_pack(f[0], f[1]);
    c[1] = amp_h2_pack(f[2], f[3]);
#else
    mma16(c, a.k0, b.k0);
    mma16(c, a.k1, b.k1);
#endif
}
#endif
AMP_INLINE AF afrag4(const u32* w) { return afrag(w[0], w[1], w[2], w[3]); }
AMP_INLINE AF afrag_v(const amp_v4& v) { return afrag(v.x, v.y, v.z, v.w); }

// ---- small helpers ---------------------------------------------------------------------------------------------------
AMP_INLINE u32 e2(u32 h) { return amp_h2_to_e4m3x2(h); }                    // float16 pair -> two codes (low 16 bits)
AMP_INLINE u32 e4(u32 lo, u32 hi)
{
#if NR_NATIVE_FP8
    return amp_native_e4m3x4(lo, hi);
#else
    return e2(lo) | (e2(hi) << 16);
#endif
}
// accumulator (4 n-fragments x rows g, g + 8) -> the 4 A-operand words
AMP_INLINE void pack_a(const u32 (*acc)[2], u32* w)
{
    w[0] = e4(acc[0][0], acc[1][0]);
    w[1] = e4(acc[0][1], acc[1][1]);
    w[2] = e4(acc[2][0], acc[3][0]);
    w[3] = e4(acc[2][1], acc[3][1]);
}
// float16 pair of n-fragment j, row r (0: g, 1: g + 8) from the 4 A-operand words (exact decode)
AMP_INLINE u32 pair_h(const u32* w, u32 j, u32 r) { return amp_e4m3x2_to_h2((w[(j >> 1) * 2u + r] >> (16u * (j & 1u))) & 0xFFFFu); }
AMP_INLINE u32 swap_h(u32 x) { return amp_prmt(x, x, 0x1032u); }
AMP_INLINE u32 shfl(u32 v, int src) { return __shfl_sync(0xffffffffu, v, src); }
AMP_INLINE u32 shfl_x(u32 v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
AMP_INLINE amp_v4 ldw(const u8* p) { return amp_load_nc(p); }               // weights: read-only for the launch
AMP_INLINE amp_v4 lds(const u8* p) { return *(const amp_v4*)p; }
AMP_INLINE void sts(u8* p, const u32* w) { amp_v4 v; v.x = w[0]; v.y = w[1]; v.z = w[2]; v.w = w[3]; *(amp_v4*)p = v; }
// the four n-fragment B operands of two consecutive 512-byte weight slices (16 outputs each)
AMP_INLINE void bfrag2(const u8* slice0, u32 lane, BF* b)
{
    const amp_v4 s0 = ldw(slice0 + lane * 16u), s1 = ldw(slice0 + 512u + lane * 16u);
    b[0] = bfrag(s0.x, s0.y); b[1] = bfrag(s0.z, s0.w);
    b[2] = bfrag(s1.x, s1.y); b[3] = bfrag(s1.z, s1.w);
}
#define ZERO4x2(a) { for (u32 _j = 0; _j < 4; ++_j) { (a)[_j][0] = 0u; (a)[_j][1] = 0u; } }

// The runtime's activation on a float16 pair: x * (0.89453125 + c * (0.447265625 - 0.055908203125 |c|)), c = clamp(x, 4)
AMP_INLINE u32 act(u32 x)
{
    const u32 c = amp_h2_max(amp_h2_min(x, hconst(K_ACT_HI)), hconst(K_ACT_LO));
    const u32 inner = amp_h2_fma(hconst(K_ACT_A), amp_h2_abs(c), hconst(K_ACT_B));
    const u32 outer = amp_h2_fma(c, inner, hconst(K_ACT_C));
    return amp_h2_mul(x, outer);
}

// Row norm of Q or K over the head's 32 dimensions (rows g and g + 8 of one tile), times `scale` (float16 pair) when
// `scaled`, to e4m3 words
AMP_INLINE void norm_rows(u32 (*q)[2], bool scaled, u32 scale, u32* words)
{
#pragma unroll
    for (u32 r = 0; r < 2; ++r) {
#if AMP_NORM_FMA == 1
        const u32 s02 = amp_h2_fma(q[0][r], q[0][r], amp_h2_mul(q[2][r], q[2][r]));
        const u32 s13 = amp_h2_fma(q[1][r], q[1][r], amp_h2_mul(q[3][r], q[3][r]));
#elif AMP_NORM_FMA == 2
        const u32 s02 = amp_h2_fma(q[2][r], q[2][r], amp_h2_mul(q[0][r], q[0][r]));
        const u32 s13 = amp_h2_fma(q[3][r], q[3][r], amp_h2_mul(q[1][r], q[1][r]));
#else
        const u32 s02 = amp_h2_add(amp_h2_mul(q[0][r], q[0][r]), amp_h2_mul(q[2][r], q[2][r]));
        const u32 s13 = amp_h2_add(amp_h2_mul(q[1][r], q[1][r]), amp_h2_mul(q[3][r], q[3][r]));
#endif
        u32 s = amp_h2_add(s13, s02);
        s = amp_h2_add(s, shfl_x(s, 2));
        s = amp_h2_add(s, shfl_x(s, 1));
        s = amp_h2_add(s, swap_h(s));
        s = amp_h2_max(s, hconst(K_FLOOR));
#if NR_NATIVE_FP8
        // The preceding half-pair reduction duplicated the sum.
        const float inv = amp_rsqrt_approx(amp_h_to_f32(u16(s)));
        const u32 rr = amp_h2_pack(inv, inv);
#else
        float lo, hi;
        amp_h2_unpack(s, lo, hi);
        const u32 rr = amp_h2_pack(amp_rsqrt_approx(lo), amp_rsqrt_approx(hi));
#endif
#pragma unroll
        for (u32 j = 0; j < 4; ++j) {
            q[j][r] = amp_h2_mul(q[j][r], rr);
            if (scaled) q[j][r] = amp_h2_mul(q[j][r], scale);
        }
    }
    pack_a(q, words);
}

// NVIDIA's transposed reduction of the softmax denominators of two query tiles (A, B = rows g, g + 8 of the first;
// Cc, D of the second): lane L ends with the sum over t (in the order t = 0, 1, 2, 3) for query tile L >> 4, row L & 15,
// then low half + high half in both halves
AMP_INLINE u32 den_reduce(u32 A, u32 B, u32 Cc, u32 D, u32 L)
{
    const bool p52 = (L & 1u) != 0u;
    const u32 r78 = p52 ? B : A, r79 = p52 ? A : B, r80 = p52 ? D : Cc, r81 = p52 ? Cc : D;
    const bool p53 = (L & 2u) == 0u;
    const u32 r82 = p53 ? r78 : r80, r83 = p53 ? r80 : r78, r84 = p53 ? r79 : r81, r85 = p53 ? r81 : r79;
    const int src = (int)(((L << 2) & 28u) + (L >> 3));
    const u32 v0 = shfl(r82, src), v1 = shfl(r84, src ^ 1), v2 = shfl(r83, src ^ 2), v3 = shfl(r85, src ^ 3);
    const bool p58 = (L & 8u) == 0u;
    const u32 r97 = p58 ? v0 : v1, r98 = p58 ? v1 : v0, r99 = p58 ? v2 : v3, r00 = p58 ? v3 : v2;
    const bool p59 = (L & 16u) == 0u;
    const u32 a = p59 ? r97 : r99, c = p59 ? r99 : r97, b = p59 ? r98 : r00, d = p59 ? r00 : r98;
    const u32 sum = amp_h2_add(amp_h2_add(amp_h2_add(a, b), c), d);
    return amp_h2_add(sum, swap_h(sum));
}

// ---- shared memory -----------------------------------------------------------------------------------------------------
// WIN: the window as A operands, [tile 4][K block NB][32 lanes x 16 B]: x, then the mid output, then y.
// CTX: the attention contexts in the same layout (upsample: first the coarse pixels, 2 NB blocks).
// POOL (ds): the pooled 4x4 tokens, [K block NB][512 B].
#define BLK(s, kb) (((s) * NB + (kb)) * 512u)

// A single warp owns the complete narrow window. Only wait/publish and global padding
// stores need synchronization when intermediate values stay in the owning lane.
#if NR_NATIVE_FP8 && HEADS == 1
#define NR_SINGLE_WARP 1
#define NR_TILE_BARRIER() ((void)0)
#define NR_MLP_LOOP _Pragma("unroll")
#else
#define NR_SINGLE_WARP 0
#define NR_TILE_BARRIER() __syncthreads()
#define NR_MLP_LOOP _Pragma("unroll 1")
#endif
#if NR_NATIVE_FP8 && HEADS <= 2
#define NR_CHANNEL_LOOP _Pragma("unroll")
#else
#define NR_CHANNEL_LOOP _Pragma("unroll 1")
#endif
extern "C" __global__ void amp_fused_swin(Params p)
{
    __shared__ __align__(16) u8 WIN[4u * NB * 512u];
#if NR_NATIVE_FP8 && HEADS <= 2
    // QKV consumes the window before attention writes its context.
    u8* const CTX = WIN;
#else
    __shared__ __align__(16) u8 CTX[4u * NB * 512u];
#endif
#if VARIANT == 2
#if NR_NATIVE_FP8 && HEADS == 1
    u8* const POOL = WIN;
#else
    __shared__ __align__(16) u8 POOL[NB * 512u];
#endif
#endif
#if HEADS == 1 && VARIANT == 3
#if NR_NATIVE_FP8
    amp_v4 residual[4][2];                      // each lane reads back its own unrounded blend
#else
    __shared__ __align__(16) u8 XF[4u * 2u * 32u * 16u];
#endif
#endif
#if NR_NATIVE_FP8 && HEADS == 1
    // A lane owns these four tiles throughout the single-head block.
    amp_v4 localWin[4];
#define LOAD_WIN(tile, k) localWin[tile]
#define STORE_WIN(tile, k, v) localWin[tile] = amp_v4{v[0], v[1], v[2], v[3]}
#else
#define LOAD_WIN(tile, k) lds(WIN + BLK(tile, k) + lane * 16u)
#define STORE_WIN(tile, k, v) sts(WIN + BLK(tile, k) + lane * 16u, v)
#endif
#if NR_NATIVE_FP8 && HEADS == 1
    amp_v4 localContext[4];
#define LOAD_CTX(tile, k) localContext[tile]
#define STORE_CTX(tile, k, v) localContext[tile] = amp_v4{v[0], v[1], v[2], v[3]}
#else
#define LOAD_CTX(tile, k) lds(CTX + BLK(tile, k) + lane * 16u)
#define STORE_CTX(tile, k, v) sts(CTX + BLK(tile, k) + lane * 16u, v)
#endif
    const u32 lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    const u32 w = NR_SINGLE_WARP ? 0u : threadIdx.y;
    const bool leader = (threadIdx.x | threadIdx.y) == 0u;
    const int x0 = p.sx + 8 * (int)blockIdx.x, y0 = p.sy + 8 * (int)blockIdx.y;
    const int TW = p.W >> 2, TH = p.H >> 2, tx0 = x0 >> 2, ty0 = y0 >> 2;
    const u8* const rec = p.rec;

    bool inside[4];
    u64 tile_at[4];
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        inside[s] = ty >= 0 && ty < TH && tx >= 0 && tx < TW;
        tile_at[s] = (u64)(u32)(ty * TW + tx) * (16u * C) + w * 512u + lane * 16u;
    }

    // ---- wait for the producer's windows under this one --------------------------------------------------------------
#if WAIT
    if (p.pflags) {
        if (leader) {
            const int gx = (p.W - min(p.psx, 0) + 7) >> 3, gy = (p.H - min(p.psy, 0) + 7) >> 3;
            const int dy = y0 - p.psy, dx = x0 - p.psx;
            const int r0 = max(dy, 0) >> 3, r1 = min(dy + 8 > 0 ? ((dy + 7) >> 3) + 1 : 0, gy);
            const int c0 = max(dx, 0) >> 3, c1 = min(dx + 8 > 0 ? ((dx + 7) >> 3) + 1 : 0, gx);
            int budget = AMP_WAIT_BUDGET_NS;
            for (int r = r0; r < r1; ++r)
                for (int c = c0; c < c1; ++c) amp_wait_not_negative(p.pflags + r * gx + c, budget);
        }
        __syncthreads();
    }
#endif

    // ---- the block's input x: e4m3 A words of channel block w of the four tiles, into WIN ---------------------------
#if VARIANT == 3
    {
        // the 4x4 coarse pixels under the window (2C channels, view layout) -> CTX as 2 NB K blocks; warp w loads 2w, 2w+1
#if HEADS == 1
        const int cr = p.H / 2, cc = p.W / 2;
#else
        const int cr = (((p.H + 1) >> 1) + 3) & ~3, cc = (((p.W + 1) >> 1) + 3) & ~3;
#endif
#if NR_SINGLE_WARP
        amp_v4 coarse[2];
#pragma unroll
#endif
        for (u32 kb = w; kb < 2u * NB; kb += NB) {
            u32 a[4];
#pragma unroll
            for (u32 i = 0; i < 4; ++i) {           // a[0]: group 2kb row g, a[1]: group 2kb row g+8, a[2], a[3]: group 2kb+1
                const u32 m = g + 8u * (i & 1u), grp = 2u * kb + (i >> 1);
                const int cy = y0 / 2 + (int)(m >> 2), cx = x0 / 2 + (int)(m & 3u);
                const bool in = cy >= 0 && cy < cr && cx >= 0 && cx < cc;
                a[i] = in ? amp_load_cg32(p.in + ((u64)(grp * (u32)cr + (u32)cy) * (u32)cc + (u32)cx) * 16u + t * 4u) : 0u;
            }
#if NR_SINGLE_WARP
            coarse[kb] = {a[0], a[1], a[2], a[3]};
#else
            sts(CTX + kb * 512u + lane * 16u, a);
#endif
        }
        NR_TILE_BARRIER();
        // 2C -> C projection: warp w's 32 channels, K blocks in order, accumulator from 0
        u32 pr[4][2];
        ZERO4x2(pr)
        for (u32 kb = 0; kb < 2u * NB; ++kb) {
            BF b[4];
            bfrag2(rec + R_RESAMPLE + kb * (32u * C) + w * 1024u, lane, b);
#if NR_SINGLE_WARP
            const AF a = afrag_v(coarse[kb]);
#else
            const AF a = afrag_v(lds(CTX + kb * 512u + lane * 16u));
#endif
#pragma unroll
            for (u32 j = 0; j < 4; ++j) mma(pr[j], a, b[j]);
        }
#if NR_NATIVE_FP8 && HEADS == 2
        NR_TILE_BARRIER(); // Every warp has read the coarse tiles before WIN reuses them.
#endif
        // skip (tile layout) * gain + the replicated projection
        u32 gain[4];
#pragma unroll
        for (u32 j = 0; j < 4; ++j) gain[j] = *(const u32*)(rec + R_GAIN + 4u * (16u * w + 4u * j + t));
#if HEADS == 1
        const int er = p.e0 > 0 ? p.e0 : p.H, ec = p.e1 > 0 ? p.e1 : p.W;
        const int STH = er >> 2, STW = ec >> 2;
#else
        const int STH = TH, STW = TW;
#endif
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const int sty = ty0 + (int)(s >> 1), stx = tx0 + (int)(s & 1u);
            const bool sin = sty >= 0 && sty < STH && stx >= 0 && stx < STW;
            amp_v4 sk = {0u, 0u, 0u, 0u};
            if (sin) sk = amp_load_cg(p.skip + (u64)(u32)(sty * STW + stx) * (16u * C) + w * 512u + lane * 16u);
            const u32 skw[4] = {sk.x, sk.y, sk.z, sk.w};
            u32 bl[4][2];
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const u32 mc7 = 4u * r + 2u * (s & 1u) + ((g & 3u) >> 1);    // coarse token & 7 (its >> 3 is s >> 1)
                const int src = (int)(4u * mc7 + t);
#pragma unroll
                for (u32 j = 0; j < 4; ++j) {
                    const u32 proj = shfl(pr[j][s >> 1], src);
                    const u32 skh = pair_h(skw, j, r);
#if AMP_UP_FMA
                    u32 v = amp_h2_fma(skh, gain[j], proj);
#else
                    u32 v = amp_h2_add(amp_h2_mul(skh, gain[j]), proj);
#endif
#if AMP_UP_ZERO_OUT
                    if (!inside[s]) v = 0u;
#endif
                    bl[j][r] = v;
                }
            }
            u32 xw[4];
            pack_a(bl, xw);
            STORE_WIN(s, w, xw);
#if HEADS == 1
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const u32 v[4] = {bl[0][r], bl[1][r], bl[2][r], bl[3][r]};
#if NR_SINGLE_WARP
                residual[s][r] = {v[0], v[1], v[2], v[3]};
#else
                sts(XF + ((s * 2u + r) * 32u + lane) * 16u, v);
#endif
            }
#endif
        }
    }
#else
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        u32 xw[4] = {0u, 0u, 0u, 0u};
#if VARIANT == 1
        const int vr = p.d0 > 0 ? p.d0 : p.H, vc = p.d1 > 0 ? p.d1 : p.W;
#pragma unroll
        for (u32 r = 0; r < 2; ++r) {
            const u32 m = g + 8u * r;
            const int py = (ty0 + (int)(s >> 1)) * 4 + (int)(m >> 2), px = (tx0 + (int)(s & 1u)) * 4 + (int)(m & 3u);
            const bool in = py >= 0 && py < vr && px >= 0 && px < vc && inside[s];
#pragma unroll
            for (u32 a = 0; a < 2; ++a)
                if (in) xw[2u * a + r] = amp_load_cg32(p.in + ((u64)(u32)((int)(2u * w + a) * vr + py) * (u32)vc + (u32)px) * 16u + t * 4u);
        }
#else
        if (inside[s]) {
            const amp_v4 v = amp_load_cg(p.in + tile_at[s]);
            xw[0] = v.x; xw[1] = v.y; xw[2] = v.z; xw[3] = v.w;
        }
#endif
        STORE_WIN(s, w, xw);
    }
#endif
    NR_TILE_BARRIER();

    // ---- MLP ----------------------------------------------------------------------------------------------------------
    // macc: the mid projection's accumulator (multi-head), or the contraction's (single head, seeded with the residual)
    u32 macc[4][4][2];
    u32 rsw[4];
#pragma unroll
    for (u32 j = 0; j < 4; ++j) rsw[j] = *(const u32*)(rec + R_RS + 4u * (16u * w + 4u * j + t));
#if HEADS == 1
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
#if VARIANT == 3
#pragma unroll
        for (u32 r = 0; r < 2; ++r) {
#if NR_SINGLE_WARP
            const amp_v4 v = residual[s][r];
#else
            const amp_v4 v = lds(XF + ((s * 2u + r) * 32u + lane) * 16u);
#endif
            const u32 bl[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (u32 j = 0; j < 4; ++j) macc[s][j][r] = amp_h2_mul(bl[j], rsw[j]);
        }
#else
        const amp_v4 v = LOAD_WIN(s, 0u);
        const u32 xw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (u32 j = 0; j < 4; ++j)
#pragma unroll
            for (u32 r = 0; r < 2; ++r) macc[s][j][r] = amp_h2_mul(pair_h(xw, j, r), rsw[j]);
#endif
    }
#else
#pragma unroll
    for (u32 s = 0; s < 4; ++s) ZERO4x2(macc[s])
#endif
NR_MLP_LOOP
    for (u32 pass = 0; pass < 4; ++pass) {
        u32 eacc[4][4][2];
#pragma unroll
        for (u32 s = 0; s < 4; ++s) ZERO4x2(eacc[s])
        const u8* const we = rec + R_EXPAND + w * (NB * 4096u) + pass * 1024u;
NR_CHANNEL_LOOP
        for (u32 kb = 0; kb < NB; ++kb) {
            BF b[4];
            bfrag2(we + kb * 4096u, lane, b);
#pragma unroll
            for (u32 s = 0; s < 4; ++s) {
                const AF a = afrag_v(LOAD_WIN(s, kb));
#pragma unroll
                for (u32 j = 0; j < 4; ++j) mma(eacc[s][j], a, b[j]);
            }
        }
        BF b[4];
#if HEADS == 1
        bfrag2(rec + R_CONTRACT + pass * 1024u, lane, b);
#else
        bfrag2(rec + R_MID + w * 4096u + pass * 1024u, lane, b);
#endif
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
#pragma unroll
            for (u32 j = 0; j < 4; ++j) { eacc[s][j][0] = act(eacc[s][j][0]); eacc[s][j][1] = act(eacc[s][j][1]); }
            u32 hw[4];
            pack_a(eacc[s], hw);
            const AF a = afrag4(hw);
#pragma unroll
            for (u32 j = 0; j < 4; ++j) mma(macc[s][j], a, b[j]);
        }
    }

    // ---- contraction (multi-head): residual seed from x, mid output through WIN -------------------------------------
    u32 yacc[4][4][2];                          // y as float16 (the unrounded contraction result)
#if HEADS == 1
#pragma unroll
    for (u32 s = 0; s < 4; ++s)
#pragma unroll
        for (u32 j = 0; j < 4; ++j) { yacc[s][j][0] = macc[s][j][0]; yacc[s][j][1] = macc[s][j][1]; }
#else
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        const amp_v4 v = LOAD_WIN(s, w);
        const u32 xw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (u32 j = 0; j < 4; ++j)
#pragma unroll
            for (u32 r = 0; r < 2; ++r) yacc[s][j][r] = amp_h2_mul(pair_h(xw, j, r), rsw[j]);
    }
    NR_TILE_BARRIER();                            // everyone is done with x
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        u32 mw[4];
        pack_a(macc[s], mw);
        STORE_WIN(s, w, mw);
    }
    NR_TILE_BARRIER();
NR_CHANNEL_LOOP
    for (u32 kb = 0; kb < NB; ++kb) {
        BF b[4];
        bfrag2(rec + R_CONTRACT + kb * (NB * 1024u) + w * 1024u, lane, b);
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const AF a = afrag_v(LOAD_WIN(s, kb));
#pragma unroll
            for (u32 j = 0; j < 4; ++j) mma(yacc[s][j], a, b[j]);
        }
    }
    NR_TILE_BARRIER();                            // everyone is done with the mid output
#endif
    // y -> e4m3 -> WIN (the QKV input; the multi-head attention residual reads it back)
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        u32 yw[4];
        pack_a(yacc[s], yw);
        STORE_WIN(s, w, yw);
    }
#if HEADS == 1
    // the single-head attention residual: float16 y * ars, kept as the output projection's seed
    u32 ars[4];
#pragma unroll
    for (u32 j = 0; j < 4; ++j) ars[j] = *(const u32*)(rec + R_ARS + 4u * (16u * w + 4u * j + t));
#pragma unroll
    for (u32 s = 0; s < 4; ++s)
#pragma unroll
        for (u32 j = 0; j < 4; ++j) { yacc[s][j][0] = amp_h2_mul(yacc[s][j][0], ars[j]); yacc[s][j][1] = amp_h2_mul(yacc[s][j][1], ars[j]); }
#endif
    NR_TILE_BARRIER();

    // ---- QKV of head w: K, V, then Q, each over all K blocks of y ----------------------------------------------------
    u32 kw[4][4], vw[4][4], qw[4][4];           // e4m3 words: K and Q as A operands, V transposed (B words per tile, nf)
    const u8* const wqkv = rec + R_QKV + w * 3072u;
NR_CHANNEL_LOOP
    for (u32 m = 0; m < 3; ++m) {               // m 0: K (slices 2, 3), 1: V (4, 5), 2: Q (0, 1)
        const u32 slice = m == 0 ? 2u : (m == 1 ? 4u : 0u);
        u32 acc[4][4][2];
#pragma unroll
        for (u32 s = 0; s < 4; ++s) ZERO4x2(acc[s])
NR_CHANNEL_LOOP
        for (u32 kb = 0; kb < NB; ++kb) {
            BF b[4];
            bfrag2(wqkv + kb * (NB * 3072u) + slice * 512u, lane, b);
#pragma unroll
            for (u32 s = 0; s < 4; ++s) {
                const AF a = afrag_v(LOAD_WIN(s, kb));
#pragma unroll
                for (u32 j = 0; j < 4; ++j) mma(acc[s][j], a, b[j]);
            }
        }
        if (m == 0) {
#pragma unroll
            for (u32 s = 0; s < 4; ++s) norm_rows(acc[s], false, 0u, kw[s]);
        } else if (m == 1) {
#pragma unroll
            for (u32 s = 0; s < 4; ++s)
#pragma unroll
                for (u32 j = 0; j < 4; ++j)
                    vw[s][j] = e4(amp_transpose8(acc[s][j][0]), amp_transpose8(acc[s][j][1]));
        } else {
            const float hs = *(const float*)(rec + R_SCAL + 4u * w);
            const u32 hs2 = amp_h2_pack(hs, hs);
#pragma unroll
            for (u32 s = 0; s < 4; ++s) norm_rows(acc[s], true, hs2, qw[s]);
        }
    }

#if NR_NATIVE_FP8 && HEADS == 2
    amp_v4 residualY[4];
#pragma unroll
    for (u32 tile = 0; tile < 4; ++tile) residualY[tile] = LOAD_WIN(tile, w);
    NR_TILE_BARRIER(); // Finish QKV and save the residual before CTX overwrites WIN.
#endif
    // ---- attention, two query tiles at a time ---------------------------------------------------------------------------
    const u8* const wbias = rec + R_BIAS + w * 8192u;
#if NR_NATIVE_FP8 && HEADS == 1
#pragma unroll
#else
NR_CHANNEL_LOOP
#endif
    for (u32 gp = 0; gp < 2; ++gp) {
        u32 n[2][8][2];                         // logits -> numerators, [query tile][key fragment][row]
#pragma unroll
        for (u32 qq = 0; qq < 2; ++qq) {
            const u32 qt = 2u * gp + qq;
            u32 qsel[4];                        // (selects: a run-time index would put qw in local memory)
#pragma unroll
            for (u32 i = 0; i < 4; ++i) qsel[i] = gp ? qw[2 + qq][i] : qw[qq][i];
            const AF qa = afrag4(qsel);
#pragma unroll
            for (u32 kt = 0; kt < 4; ++kt) {
                const amp_v4 bias = ldw(wbias + (qt * 4u + kt) * 512u + lane * 16u);
                n[qq][2 * kt][0] = bias.x; n[qq][2 * kt][1] = bias.y;
                n[qq][2 * kt + 1][0] = bias.z; n[qq][2 * kt + 1][1] = bias.w;
                mma(n[qq][2 * kt], qa, bfrag(kw[kt][0], kw[kt][2]));
                mma(n[qq][2 * kt + 1], qa, bfrag(kw[kt][1], kw[kt][3]));
            }
        }
        u32 den[2][2];
#pragma unroll
        for (u32 qq = 0; qq < 2; ++qq)
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
#pragma unroll
                for (u32 k = 0; k < 8; ++k) {
                    u32 x = amp_h2_fma(n[qq][k][r], hconst(K_EXP_A), hconst(K_EXP_B));
                    x = amp_h2_min(amp_h2_max(x, hconst(K_EXP_LO)), hconst(K_EXP_HI));
                    n[qq][k][r] = (x << 5) + 0x7FF88000u;
                }
                den[qq][r] = amp_h2_add(amp_h2_add(amp_h2_add(amp_h2_add(n[qq][0][r], n[qq][1][r]), amp_h2_add(n[qq][2][r], n[qq][3][r])),
                                                   amp_h2_add(n[qq][4][r], n[qq][5][r])), amp_h2_add(n[qq][6][r], n[qq][7][r]));
            }
        u32 rc = den_reduce(den[0][0], den[0][1], den[1][0], den[1][1], lane);
        rc = amp_h2_max(rc, hconst(K_FLOOR));
        {
#if NR_NATIVE_FP8
            const float inv = amp_rcp_approx(amp_h_to_f32(u16(rc)));
            rc = amp_h2_pack(inv, inv);
#else
            float lo, hi;
            amp_h2_unpack(rc, lo, hi);
            rc = amp_h2_pack(amp_rcp_approx(lo), amp_rcp_approx(hi));
#endif
        }
#pragma unroll
        for (u32 qq = 0; qq < 2; ++qq) {
            const u32 qt = 2u * gp + qq;
            const u32 rq0 = shfl(rc, (int)(16u * qq + g)), rq1 = shfl(rc, (int)(16u * qq + g + 8u));
            u32 pa[2][4];                       // P as A operands of the two k32 steps
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) {
                u32 pv[4][2];
#pragma unroll
                for (u32 j = 0; j < 4; ++j) {
                    pv[j][0] = amp_h2_mul(n[qq][4 * ks + j][0], rq0);
                    pv[j][1] = amp_h2_mul(n[qq][4 * ks + j][1], rq1);
                }
                pack_a(pv, pa[ks]);
            }
            u32 ctx[4][2];
            ZERO4x2(ctx)
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) {
                const AF a = afrag4(pa[ks]);
#pragma unroll
                for (u32 j = 0; j < 4; ++j) mma(ctx[j], a, bfrag(vw[2 * ks][j], vw[2 * ks + 1][j]));
            }
            u32 cw[4];
            pack_a(ctx, cw);
            STORE_CTX(qt, w, cw);
        }
    }

    // ---- output projection with the attention residual -------------------------------------------------------------------
    u32 o[4][4][2];
#if HEADS == 1
#pragma unroll
    for (u32 s = 0; s < 4; ++s)
#pragma unroll
        for (u32 j = 0; j < 4; ++j) { o[s][j][0] = yacc[s][j][0]; o[s][j][1] = yacc[s][j][1]; }
#else
    {
        u32 ars[4];
#pragma unroll
        for (u32 j = 0; j < 4; ++j) ars[j] = *(const u32*)(rec + R_ARS + 4u * (16u * w + 4u * j + t));
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
#if NR_NATIVE_FP8 && HEADS == 2
            const amp_v4 v = residualY[s];
#else
            const amp_v4 v = LOAD_WIN(s, w);
#endif
            const u32 yw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (u32 j = 0; j < 4; ++j)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) o[s][j][r] = amp_h2_mul(pair_h(yw, j, r), ars[j]);
        }
    }
#endif
    NR_TILE_BARRIER();                            // all contexts are in CTX
NR_CHANNEL_LOOP
    for (u32 kb = 0; kb < NB; ++kb) {
        BF b[4];
        bfrag2(rec + R_OUT + kb * (NB * 1024u) + w * 1024u, lane, b);
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const AF a = afrag_v(LOAD_CTX(s, kb));
#pragma unroll
            for (u32 j = 0; j < 4; ++j) mma(o[s][j], a, b[j]);
        }
    }

    // ---- stores ---------------------------------------------------------------------------------------------------------
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        u32 ow[4];
        pack_a(o[s], ow);
#if VARIANT == 4
        const int vr = p.d0 > 0 ? p.d0 : p.H, vc = p.d1 > 0 ? p.d1 : p.W;
#pragma unroll
        for (u32 r = 0; r < 2; ++r) {
            const u32 m = g + 8u * r;
            const int py = (ty0 + (int)(s >> 1)) * 4 + (int)(m >> 2), px = (tx0 + (int)(s & 1u)) * 4 + (int)(m & 3u);
            if (inside[s] && py >= 0 && py < vr && px >= 0 && px < vc) {
#pragma unroll
                for (u32 a = 0; a < 2; ++a)
                    *(u32*)(p.out + ((u64)(u32)((int)(2u * w + a) * vr + py) * (u32)vc + (u32)px) * 16u + t * 4u) = ow[2u * a + r];
            }
        }
#else
        if (inside[s]) {
            amp_v4 v; v.x = ow[0]; v.y = ow[1]; v.z = ow[2]; v.w = ow[3];
            *(amp_v4*)(p.out + tile_at[s]) = v;
        }
#endif
    }

#if VARIANT == 2
    // ---- 2x2 mean of the unrounded output, then the C -> 2C projection of the 16 pooled tokens ------------------------
    // Lane group g takes pooled tokens (tile ci, row r, column (g >> 1) & 1) as rows g and g + 8 of one M tile; the two
    // butterfly steps add (v + v^4) + (v^16 + v^20), the runtime's additions (float16 addition commutes exactly).
    {
        const u32 ci = (g & 1u) + 2u * (g >> 2);
        const bool b2 = (g & 1u) != 0u, b4 = (g >> 2) != 0u;
        u32 pool[4][2];
#pragma unroll
        for (u32 r = 0; r < 2; ++r)
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                u32 k0 = b2 ? o[1][j][r] : o[0][j][r];
                u32 k1 = b2 ? o[3][j][r] : o[2][j][r];
                const u32 s0 = b2 ? o[0][j][r] : o[1][j][r];
                const u32 s1 = b2 ? o[2][j][r] : o[3][j][r];
                k0 = amp_h2_add(k0, shfl_x(s0, 4));
                k1 = amp_h2_add(k1, shfl_x(s1, 4));
                const u32 a = amp_h2_add(b4 ? k1 : k0, shfl_x(b4 ? k0 : k1, 16));
                pool[j][r] = amp_h2_mul(a, 0x34003400u);
            }
        u32 pw[4];
        pack_a(pool, pw);
#if !NR_SINGLE_WARP
        sts(POOL + w * 512u + lane * 16u, pw);
#endif
        NR_TILE_BARRIER();
        const int hH = (p.H + 1) >> 1, hW = (p.W + 1) >> 1, vr = (hH + 3) & ~3, vc = (hW + 3) & ~3;
        const int px = (x0 >> 1) + (int)((ci & 1u) * 2u + ((g >> 1) & 1u));
NR_CHANNEL_LOOP
        for (u32 half = 0; half < 2; ++half) {
            const u32 sl0 = half ? 2u * NB + 2u * w : 2u * w;
            u32 racc[4][2];
            ZERO4x2(racc)
NR_CHANNEL_LOOP
            for (u32 kb = 0; kb < NB; ++kb) {
                BF b[4];
                bfrag2(rec + R_RESAMPLE + kb * (64u * C) + sl0 * 512u, lane, b);
#if NR_SINGLE_WARP
                const AF a = afrag4(pw);
#else
                const AF a = afrag_v(lds(POOL + kb * 512u + lane * 16u));
#endif
#pragma unroll
                for (u32 j = 0; j < 4; ++j) mma(racc[j], a, b[j]);
            }
            if (p.ds_out) {
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    const int py = (y0 >> 1) + (int)((ci >> 1) * 2u + r);
                    if (py >= 0 && py < vr && px >= 0 && px < vc) {
#pragma unroll
                        for (u32 k = 0; k < 2; ++k)
                            *(u32*)(p.ds_out + ((u64)(u32)((int)(sl0 + k) * vr + py) * (u32)vc + (u32)px) * 16u + t * 4u) =
                                e4(racc[2 * k][r], racc[2 * k + 1][r]);
                    }
                }
            }
        }
        // zeros in the padding of the pooled view when the runtime's dims exceed the half size: each window clears the
        // padding pixels of its own 4x4 pooled block after its stores; window (0, 0) also the pixels no window covers
        if (p.ds_out && (p.d0 > hH || p.d1 > hW)) {
#if NR_SINGLE_WARP
            __syncwarp();
#else
            __syncthreads();
#endif
            const int R = p.d0, Cn = p.d1;
            const u32 groups = 2u * C / 16u, tid = threadIdx.y * 32u + threadIdx.x, nt = 32u * NB;
            for (u32 i = tid; i < 16u * groups * 4u; i += nt) {
                const u32 word = i & 3u, grp = (i >> 2) % groups, px16 = (i >> 2) / groups;
                const int py = (y0 >> 1) + (int)(px16 >> 2), pxx = (x0 >> 1) + (int)(px16 & 3u);
                if (py >= 0 && py < R && pxx >= 0 && pxx < Cn && (py >= hH || pxx >= hW))
                    *(u32*)(p.ds_out + ((u64)(grp * (u32)R + (u32)py) * (u32)Cn + (u32)pxx) * 16u + word * 4u) = 0u;
            }
            if (blockIdx.x == 0 && blockIdx.y == 0) {
                const int ry = (p.sy >> 1) + 4 * (int)gridDim.y, rx = (p.sx >> 1) + 4 * (int)gridDim.x;
                for (int py = 0; py < R; ++py) {
                    const int c0 = py >= ry ? 0 : max(rx, 0);
                    for (u32 i = tid; i < (u32)max(Cn - c0, 0) * groups * 4u; i += nt) {
                        const u32 word = i & 3u, grp = (i >> 2) % groups;
                        const int pxx = c0 + (int)((i >> 2) / groups);
                        if (py >= hH || pxx >= hW)
                            *(u32*)(p.ds_out + ((u64)(grp * (u32)R + (u32)py) * (u32)Cn + (u32)pxx) * 16u + word * 4u) = 0u;
                    }
                }
            }
        }
    }
#endif

#if PUBLISH
    if (p.oflags) {
        __syncthreads();
        if (leader) amp_flag_publish(p.oflags + blockIdx.y * gridDim.x + blockIdx.x, 0);
    }
#endif
}

#undef NR_SINGLE_WARP
#undef NR_TILE_BARRIER
#undef NR_MLP_LOOP
#undef NR_CHANNEL_LOOP
#undef LOAD_WIN
#undef STORE_WIN
#undef LOAD_CTX
#undef STORE_CTX
