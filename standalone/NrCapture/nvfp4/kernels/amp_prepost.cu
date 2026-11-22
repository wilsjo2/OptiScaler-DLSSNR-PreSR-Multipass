// amp_prepost.cu: Ampere-legal (sm_86) FP16 replacements of the DLSS-NR Model A input and output blocks (Phase A).
//
//   variant pre     amp_pre       for cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8, runtime 310.9.16 (272-byte block,
//                                 weight record pointer at +232)
//   variant pre3108 amp_pre_3108  the same kernel for runtime 310.8 (264-byte block, record pointer at +224)
//   variant post    amp_post      for cc_tinlayout_fused_post_block_swin_1h_32_fp8 (184-byte block in both runtimes,
//                                 record pointer at +24, pointer to the f16 history blend weight at +104)
//
// Drop-in: each kernel takes EXACTLY NVIDIA's parameter block and NVIDIA's own e4m3 weight record (the fp8_set<n>.bin
// records of the shipped FP8 models, unchanged), is launched with the runtime's grid and block (32 x 1 x 1, one warp
// per 8x8 window, no dynamic shared memory), reads and writes the same buffers in the same layouts. No flags (neither
// of NVIDIA's kernels has an ordering protocol). Texture / surface handles are 64-bit: build with
// --ptxas-options=--set-texmode-raw=true (every variant line below carries it).
//
// Arithmetic: NVIDIA's operation order, read from the 310.9.16 PTX (fatbins/310.9.16/fatbin00-sm120-entry0.ptx, pre
// 14458-29961, post 29963-46228) and its SASS operation counts:
//   * every e4m3 matrix product (NVIDIA: mma.m16n8k32 .f16.e4m3.e4m3.f16) is two mma.m16n8k16 .f16.f16.f16.f16 on the
//     same registers after an exact software e4m3 -> f16 decode (amp_common.cuh); the f16 accumulator is rounded
//     after each 16 products instead of each 32 (the only intended difference from NVIDIA);
//   * every store to e4m3 (cvt.rn.satfinite.e4m3x2.f16x2) in software integer code, bit-identical to the hardware;
//     where NVIDIA keeps e4m3 codes in registers only as the next product's operand (x, the MLP hidden units, y, K, V,
//     Q, the softmax weights, the attention context), q8() rounds the f16 values to e4m3 directly (the same bits as
//     encoding and decoding, all 65536 f16 values checked; prepost_lab.py amp-unit): about 2.5x faster than the codes;
//   * input lift (pre) and output projection (post) are NVIDIA's own f16 m16n8k16 products (already Ampere);
//   * the cubic SiLU, Q/K norms (squares fused into the first additions as NVIDIA's ptxas did: 32 HFMA2 more and 32
//     HMUL2 fewer than its PTX), softmax by the exponent-field trick and NVIDIA's transposed lane reduction of the
//     denominator, rcp/rsqrt.approx narrowed to f16, the residual seeds as starting accumulators, the f16 attention
//     residual (one head: the unrounded y);
//   * input and output stages copied from prepost4.cu (proven bit for bit with NVIDIA's identity-body kernels), with
//     the hardware e4m3 conversions replaced by amp_common.cuh's.
// *_exact variants (F32ACC=31, still Ampere-legal): each k32 step as two m16n8k16 with an f32 accumulator between the
// halves (f16 in, f16 out, one rounding per k32 like NVIDIA's instruction). Measured bit for bit equal to NVIDIA's
// kernels (prepost_lab.py amp --exact: 1080p / 1440p / 4K, both runtimes, with and without history, conditioning
// branches); the plain variants are the f16-accumulate target (one more f16 rounding per k32 step). F32ACC bits pick
// the product groups (1 MLP, 2 QKV, 4 attention logits, 8 context, 16 output projection).
// Lab-only switch AMP_HW_K32=1 (sm_120 only, NOT Ampere-legal, never shipped): NVIDIA's e4m3 k32 instruction instead
// of the k16 pair; everything else unchanged. It isolates layout / order bugs: it should equal NVIDIA's kernel bit for
// bit (it does: prepost_lab.py amp --hw). NORM_FUSE picks which square ptxas folded into the first addition of each
// norm (1: the first operand, 2: the second, 0: none); the default is the one the lab measured. Lab-only switches that
// change no bit: Q8_FORM (0: the first q8 form), Y_KEEP, MLP_ROLL; AMP_TIME_NOCODEC=1 is timing only (wrong numbers).
//
// Not implemented (the host must not give these launches to this kernel): pre-block depth texture (+24 != 0) and
// control-mask texture (+32 != 0). Everything else of the traced blocks is: history / motion (pre +8 / +16, post +88 /
// +96 / +104 / +112), automatic mask on and off (+200), the slot selector (+196), skin / other strengths below zero.
//
// amp-variant: pre -DBLOCK=1 --ftz=true --ptxas-options=--set-texmode-raw=true
// amp-variant: pre3108 -DBLOCK=3 --ftz=true --ptxas-options=--set-texmode-raw=true
// amp-variant: post -DBLOCK=2 --ftz=true --ptxas-options=--set-texmode-raw=true
// amp-variant: pre_exact -DBLOCK=1 -DF32ACC=31 --ftz=true --ptxas-options=--set-texmode-raw=true
// amp-variant: pre3108_exact -DBLOCK=3 -DF32ACC=31 --ftz=true --ptxas-options=--set-texmode-raw=true
// amp-variant: post_exact -DBLOCK=2 -DF32ACC=31 --ftz=true --ptxas-options=--set-texmode-raw=true
// Own pass host only (NrHost 'prepare amp_prepare_prepost 2': f16 weights converted once at load; NrAmpere cannot run these):
// amp-variant: pre_p -DBLOCK=1 -DAMP_PREPARED=1 --ftz=true --ptxas-options=--set-texmode-raw=true
// amp-variant: pre3108_p -DBLOCK=3 -DAMP_PREPARED=1 --ftz=true --ptxas-options=--set-texmode-raw=true
// amp-variant: post_p -DBLOCK=2 -DAMP_PREPARED=1 --ftz=true --ptxas-options=--set-texmode-raw=true
#include "amp_common.cuh"

#ifndef BLOCK
#define BLOCK 1
#endif
#ifndef AMP_HW_K32
#define AMP_HW_K32 0
#endif
#ifndef NORM_FUSE
#define NORM_FUSE 1
#endif
#ifndef AMP_ACT_EXACT          // activations decoded with the NaN-exact decode (1) or the finite one (0)
#define AMP_ACT_EXACT 1
#endif
#ifndef Y_KEEP                 // e4m3(y) operands kept for K, V and Q (1) or made three times (0); same bits either way
#define Y_KEEP 1
#endif
#ifndef MLP_ROLL
#define MLP_ROLL 0
#endif
#ifndef DBG_STAGE              // 0: the block; 2: Swin body skipped (o = x): compare with NVIDIA on an identity-body record
#define DBG_STAGE 0
#endif

#define NR_INLINE static __forceinline__ __device__
typedef amp_v4 V4;

// ---- parameter blocks -------------------------------------------------------------------------------------------
#if BLOCK == 1
struct __align__(8) Params { u8 b[272]; };
#define OFS(o) (o)
#elif BLOCK == 3
struct __align__(8) Params { u8 b[264]; };
// 310.8 lacks two 32-bit fields of 310.9.16: +40 (mask combine mode) and +196 (feature slot selector). Offsets are
// written as 310.9.16 offsets everywhere below and moved here (checked against the traced 310.8 and 310.9.16 4K blocks).
#define OFS(o) ((o) < 40 ? (o) : ((o) <= 192 ? (o) - 4 : (o) - 8))
#else
struct __align__(8) Params { u8 b[184]; };
#define OFS(o) (o)
#endif
#define P_U64(o) (*(const u64*)(P.b + OFS(o)))
#define P_F32(o) (*(const float*)(P.b + OFS(o)))
#define P_S32(o) (*(const int*)(P.b + OFS(o)))

// ---- NVIDIA record offsets ----------------------------------------------------------------------------------------
struct Rec { u32 exp, con, rs, qkv, bias, hs, proj, ars; };
#if BLOCK == 2
#define R_MGAIN 8272u
#define R_SGAIN 8336u
#define R_OUTP 20784u
__device__ __constant__ const Rec kRec = {0u, 4096u, 8208u, 8400u, 11472u, 19664u, 19680u, 20704u};
#else
#define R_LIFT 8208u
__device__ __constant__ const Rec kRec = {0u, 4096u, 9232u, 9312u, 12384u, 20576u, 20592u, 21616u};
#endif

// ---- small helpers ----------------------------------------------------------------------------------------------
struct F4 { float x, y, z, w; };
NR_INLINE F4 tex_rgba(u64 tex, float u, float v)
{
    F4 r;
    asm volatile("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(tex), "f"(u), "f"(v));
    return r;
}
NR_INLINE float rcp_approx(float x) { return amp_rcp_approx(x); }
NR_INLINE u16 f16_of(float x) { u16 d; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(d) : "f"(x)); return d; }
NR_INLINE u32 h2_fma(u32 a, u32 b, u32 c) { return amp_h2_fma(a, b, c); }
NR_INLINE u32 h2_mul(u32 a, u32 b) { return amp_h2_mul(a, b); }
NR_INLINE u32 h2_add(u32 a, u32 b) { return amp_h2_add(a, b); }
NR_INLINE u32 h2_max(u32 a, u32 b) { return amp_h2_max(a, b); }
NR_INLINE u32 h2_min(u32 a, u32 b) { return amp_h2_min(a, b); }
#ifndef AMP_TIME_NOCODEC        // lab timing only, WRONG numerics: activation encode / decode replaced by byte moves
#define AMP_TIME_NOCODEC 0
#endif
#if AMP_TIME_NOCODEC
NR_INLINE u32 enc(u32 lo, u32 hi) { return amp_prmt(lo, hi, 0x6420u); }
#else
#ifndef ENC_FAST                 // 1: amp_common.cuh's fast encoder (identical codes for every non-NaN value)
#define ENC_FAST 1
#endif
#if ENC_FAST
NR_INLINE u32 enc(u32 lo, u32 hi) { return amp_h2x2_to_e4m3x4_fast(lo, hi); }   // two f16 pairs -> 4 e4m3 codes
#else
NR_INLINE u32 enc(u32 lo, u32 hi) { return amp_h2x2_to_e4m3x4(lo, hi); }        // two f16 pairs -> 4 e4m3 codes
#endif
#endif
NR_INLINE V4 ldw(const u8* p) { return *(const V4*)p; }                             // weights: ld.global.v4 (L1)
NR_INLINE u32 ldw32(const u8* p) { return *(const u32*)p; }
// f16 pair whose two halves are f16(f(x.lo)), f16(f(x.hi)) for an approx f32 function f
NR_INLINE u32 h2_rsqrt(u32 x)
{
    float lo, hi;
    amp_h2_unpack(x, lo, hi);
    return amp_h2_pack(amp_rsqrt_approx(lo), amp_rsqrt_approx(hi));
}
NR_INLINE u32 swap_halves(u32 x) { return amp_prmt(x, x, 0x1032u); }
NR_INLINE u32 low_twice(u32 x) { return amp_prmt(x, x, 0x5410u); }
NR_INLINE u32 high_twice(u32 x) { return amp_prmt(x, x, 0x7632u); }

// ---- e4m3 matrix products ---------------------------------------------------------------------------------------
// A operand of one k32 step: 4 words of e4m3 codes in NVIDIA's layout; B: 2 words. Decoded once and reused.
#if AMP_HW_K32
struct FA { u32 w[4]; };
struct FB { u32 w[2]; };
NR_INLINE FA fa(u32 a0, u32 a1, u32 a2, u32 a3) { FA r = {{a0, a1, a2, a3}}; return r; }
NR_INLINE FB fb(u32 b0, u32 b1) { FB r = {{b0, b1}}; return r; }
NR_INLINE void mma(u32* c, const FA& a, const FB& b)
{
    u32 d0, d1;
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
        : "=r"(d0), "=r"(d1) : "r"(a.w[0]), "r"(a.w[1]), "r"(a.w[2]), "r"(a.w[3]), "r"(b.w[0]), "r"(b.w[1]), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}
#else
struct FA { u32 k0[4], k1[4]; };
struct FB { u32 k0[2], k1[2]; };
NR_INLINE void dec_act(u32 w, u32& lo, u32& hi)
{
#if AMP_TIME_NOCODEC
    lo = amp_prmt(w, 0u, 0x1404u); hi = amp_prmt(w, 0u, 0x3424u);
#elif AMP_ACT_EXACT
    amp_e4m3x4_to_h2x2(w, lo, hi);
#else
    amp_e4m3x4_to_h2x2_finite(w, lo, hi);
#endif
}
// e4m3 A word j holds K 4t..4t+3 (j = 0, 1) / 4t+16..19 (j = 2, 3): bytes 0,1 -> k16 slot 2t, bytes 2,3 -> slot 2t+8
NR_INLINE FA fa(u32 a0, u32 a1, u32 a2, u32 a3)
{
    FA r;
    dec_act(a0, r.k0[0], r.k0[2]);
    dec_act(a1, r.k0[1], r.k0[3]);
    dec_act(a2, r.k1[0], r.k1[2]);
    dec_act(a3, r.k1[1], r.k1[3]);
    return r;
}
// weights (never NaN codes in the records) and attention operands built from activations
#ifndef AMP_TIME_NOWDEC          // lab timing only, WRONG numerics: weight decode replaced by byte moves
#define AMP_TIME_NOWDEC 0
#endif
NR_INLINE FB fb(u32 b0, u32 b1)
{
    FB r;
#if AMP_TIME_NOWDEC
    r.k0[0] = amp_prmt(b0, 0u, 0x1404u); r.k0[1] = amp_prmt(b0, 0u, 0x3424u);
    r.k1[0] = amp_prmt(b1, 0u, 0x1404u); r.k1[1] = amp_prmt(b1, 0u, 0x3424u);
#else
    amp_e4m3x4_to_h2x2_finite(b0, r.k0[0], r.k0[1]);
    amp_e4m3x4_to_h2x2_finite(b1, r.k1[0], r.k1[1]);
#endif
    return r;
}
NR_INLINE FB fb_act(u32 b0, u32 b1)
{
    FB r;
    dec_act(b0, r.k0[0], r.k0[1]);
    dec_act(b1, r.k1[0], r.k1[1]);
    return r;
}
NR_INLINE void mma16(u32* c, const u32* a, const u32* b)
{
    u32 d0, d1;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
        : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}
// NVIDIA's k32 step: K 0..15 first, then K 16..31, on one f16 accumulator
NR_INLINE void mma(u32* c, const FA& a, const FB& b)
{
    mma16(c, a.k0, b.k0);
    mma16(c, a.k1, b.k1);
}
// The same k32 step with an f32 accumulator between the two halves, rounded to f16 once at the end (as the k32
// instruction rounds once): chosen per product group by F32ACC (bit 0 MLP, 1 QKV, 2 attention logits, 3 context,
// 4 attention output projection). Default 0: the f16-accumulate target.
NR_INLINE void mma_f32(u32* c, const FA& a, const FB& b)
{
    float f0, f1, f2, f3;
    amp_h2_unpack(c[0], f0, f1);
    amp_h2_unpack(c[1], f2, f3);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(f0), "+f"(f1), "+f"(f2), "+f"(f3) : "r"(a.k0[0]), "r"(a.k0[1]), "r"(a.k0[2]), "r"(a.k0[3]), "r"(b.k0[0]), "r"(b.k0[1]));
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(f0), "+f"(f1), "+f"(f2), "+f"(f3) : "r"(a.k1[0]), "r"(a.k1[1]), "r"(a.k1[2]), "r"(a.k1[3]), "r"(b.k1[0]), "r"(b.k1[1]));
    c[0] = amp_h2_pack(f0, f1);
    c[1] = amp_h2_pack(f2, f3);
}
#endif
#ifndef F32ACC
#define F32ACC 0
#endif
#if AMP_HW_K32
#define MMA_G(bit, c, a, b) mma(c, a, b)
#else
#define MMA_G(bit, c, a, b) do { if (F32ACC & (bit)) mma_f32(c, a, b); else mma(c, a, b); } while (0)
#endif
#if AMP_HW_K32
#define fb_act fb
#endif
// plain f16 m16n8k16 (NVIDIA's lift / output projection: already f16 operands)
NR_INLINE void mma16_h(u32* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1)
{
    u32 d0, d1;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
        : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}

// ---- prepared records (AMP_PREPARED=1; NrHost 'prepare amp_prepare_prepost 2') -----------------------------------
// The host runs amp_prepare_prepost once on the fp8 record and gives the kernel the copy instead: every piece at twice
// its record offset (RO), the four e4m3 matrices (MLP expansion / contraction, QKV, attention projection) as f16 B
// fragments in the order the kernel loads them (each 512-byte slice -> 1024 bytes: lane L's 16 bytes of fb(x, y) at
// +16 L, of fb(z, w) at +512 + 16 L), everything else copied unchanged. Same layout as collab/astra/amp11 fused/split.
#ifndef AMP_PREPARED
#define AMP_PREPARED 0
#endif
#if AMP_PREPARED && AMP_HW_K32
#error "AMP_PREPARED needs the f16 operand path"
#endif
#define RO(o) ((o) * (1u + AMP_PREPARED))
// B operands of a 1024-byte matrix block at record offset off (two 512-byte slices): b[0..3]
NR_INLINE void fb4(const u8* rec, u32 off, u32 lane, FB (&b)[4])
{
#if AMP_PREPARED
#pragma unroll
    for (u32 j = 0; j < 4; ++j) {
        const V4 v = ldw(rec + 2u * off + j * 512u + lane * 16u);
        b[j].k0[0] = v.x; b[j].k0[1] = v.y; b[j].k1[0] = v.z; b[j].k1[1] = v.w;
    }
#else
    const V4 e0 = ldw(rec + off + lane * 16u), e1 = ldw(rec + off + 512u + lane * 16u);
    b[0] = fb(e0.x, e0.y); b[1] = fb(e0.z, e0.w); b[2] = fb(e1.x, e1.y); b[3] = fb(e1.z, e1.w);
#endif
}

// ---- e4m3 rounding without the codes -----------------------------------------------------------------------------
// q8(h): the f16 pair of the e4m3 rounding of an f16 pair, bit for bit the same as decoding (dec_act) the codes that
// enc() stores (checked for all 65536 f16 values in both halves: prepost_lab.py amp-unit). Used wherever NVIDIA stores
// e4m3 codes in registers only to feed them to its next e4m3 product: the codes themselves are never needed there, so
// the integer encode and the decode collapse into one rounding on the f16 bit pattern (about 13 instead of about 30
// operations per pair). |x| is clamped to 448 first (satfinite), then rounded by (|x| + c) - c in f16 (round to
// nearest even) with c = 2^(e + 7) for |x| in [2^e, 2^(e+1)): the sum's ulp is then 2^(e-3), e4m3's step for that
// binade; c >= 2 (ulp 2^-9) gives e4m3's subnormal step below 2^-6; nothing overflows (448 + 2^15 < 65504). NaN -> what
// the decode gives for the NaN code (0x7FFF exact, 480 = 0x5F80 with the finite decode); signs kept (-0 included).
// Q8_FORM 0: the first form (integer rounding of the f16 bits for normals, (|x| + 2) - 2 for subnormals, a select).
#if AMP_ACT_EXACT
#define AMP_Q8_NAN 0x7FFF7FFFu
#else
#define AMP_Q8_NAN 0x5F805F80u
#endif
#ifndef Q8_FORM                // 2 (BRIEF-11, 6 operations); 1: the 21 Nov form (10 operations); 0: the first form
#define Q8_FORM 2
#endif
#ifndef Q8_SIGN                // 1: form 2 keeps -0 for negative values that round to zero (one more operation)
#define Q8_SIGN 0
#endif
#ifndef AMP_F16_ACT              // 1: activations stay f16 between products (no e4m3 rounding in registers); stores keep e4m3
#define AMP_F16_ACT 0
#endif
NR_INLINE u32 q8(u32 h)
{
#if AMP_TIME_NOCODEC || AMP_F16_ACT
    return h;
#elif Q8_FORM == 2
    // 6 operations (7 with Q8_SIGN): hc = h clamped to +-448 keeping its sign (min.xorsign.abs, sm_86); cs = +-2^e of hc
    // (sign and exponent bits) times 2^7, at least +-2 (max.xorsign.abs keeps cs's sign); (hc + cs) - cs rounds |hc| to
    // e4m3's step for its binade, ties to even, symmetric in the sign. Same as form 1 for every finite value except
    // that a negative value rounding to zero gives +0 without Q8_SIGN; a NaN saturates to +-448 (form 1: AMP_Q8_NAN).
    const u32 hc = amp_h2_xamin(h, 0x5F005F00u);
    const u32 cs = amp_h2_xamax(h2_mul(hc & 0xFC00FC00u, 0x58005800u), 0x40004000u);
    const u32 r = amp_h2_sub(h2_add(hc, cs), cs);
#if Q8_SIGN
    return r | (h & 0x80008000u);
#else
    return r;
#endif
#elif Q8_FORM == 1
    const u32 m = h2_min(h & 0x7FFF7FFFu, 0x5F005F00u);
    const u32 c = h2_max((m & 0x7C007C00u) + 0x1C001C00u, 0x40004000u);
    const u32 r = amp_h2_sub(h2_add(m, c), c) | (h & 0x80008000u);
    u32 nan;
    asm("set.nan.u32.f16x2 %0, %1, %1;" : "=r"(nan) : "r"(h));
    return (r & ~nan) | (nan & AMP_Q8_NAN);
#else
    const u32 m = h2_min(h & 0x7FFF7FFFu, 0x5F005F00u);
    const u32 rn = (m + 0x003F003Fu + ((m >> 7) & 0x00010001u)) & 0x7F807F80u;
    const u32 sub = amp_h2_sub(h2_add(m, 0x40004000u), 0x40004000u);
    u32 big, nan;
    asm("set.ge.u32.f16x2 %0, %1, %2;" : "=r"(big) : "r"(m), "r"(0x24002400u));
    asm("set.nan.u32.f16x2 %0, %1, %1;" : "=r"(nan) : "r"(h));
    const u32 r = ((rn & big) | (sub & ~big)) | (h & 0x80008000u);
    return (r & ~nan) | (nan & AMP_Q8_NAN);
#endif
}

// The A operand words of a 16-token tile from an accumulator-layout value v[j][r] (frag j = channels 8j+2t+e, row
// r = token g / g+8): {e4m3(f0 g, f1 g), e4m3(f0 g+8, f1 g+8), e4m3(f2 g, f3 g), e4m3(f2 g+8, f3 g+8)} (NVIDIA's
// packing everywhere in these kernels: K = 16 (j >> 1) + 4 t + 2 (j & 1) + e).
NR_INLINE void pack_a(const u32 (&v)[4][2], u32 (&w)[4])
{
    w[0] = enc(v[0][0], v[1][0]);
    w[1] = enc(v[0][1], v[1][1]);
    w[2] = enc(v[2][0], v[3][0]);
    w[3] = enc(v[2][1], v[3][1]);
}
// The operand NVIDIA's e4m3 product reads from those codes, made directly (q8: no codes). fa_q: A operand of an
// accumulator-layout value as pack_a + fa would give it; fb_q: B operand of the four f16 pairs p0..p3 that NVIDIA
// packs as {e4m3(p0, p1), e4m3(p2, p3)}; fb_q_half: half (0 / 1) of such a B operand from two pairs. The HW k32 lab
// build keeps the codes (its instruction reads them).
NR_INLINE FA fa_q(const u32 (&v)[4][2])
{
#if AMP_HW_K32
    u32 w[4];
    pack_a(v, w);
    return fa(w[0], w[1], w[2], w[3]);
#else
    FA r;
    r.k0[0] = q8(v[0][0]); r.k0[1] = q8(v[0][1]); r.k0[2] = q8(v[1][0]); r.k0[3] = q8(v[1][1]);
    r.k1[0] = q8(v[2][0]); r.k1[1] = q8(v[2][1]); r.k1[2] = q8(v[3][0]); r.k1[3] = q8(v[3][1]);
    return r;
#endif
}
NR_INLINE void fb_q_half(FB& b, u32 half, u32 p0, u32 p1)
{
#if AMP_HW_K32
    b.w[half] = enc(p0, p1);
#else
    if (half == 0u) { b.k0[0] = q8(p0); b.k0[1] = q8(p1); }
    else            { b.k1[0] = q8(p0); b.k1[1] = q8(p1); }
#endif
}

// ---- the Swin body (one head, width 32, NVIDIA's arithmetic) ------------------------------------------------------
// The window's 4 tiles: x[s][r][j] (f16 pairs, the block's input) -> o[s][r][j] (f16 pairs, the body output).
// Cubic SiLU on f16 pairs (constants of the PTX: clamp +-4, -0.055908203125, 0.447265625, 0.89453125)
NR_INLINE u32 silu3(u32 x)
{
    const u32 c = h2_max(h2_min(x, 0x44004400u), 0xC400C400u);
    const u32 u = h2_fma(0xAB28AB28u, amp_h2_abs(c), 0x37283728u);
    const u32 v = h2_fma(c, u, 0x3B283B28u);
    return h2_mul(x, v);
}
// squared norm of a token row over its 32 dims (8 here, 24 in the lanes t ^ 1, 2, 3): NVIDIA's f16 order, the
// squares of frags 0 / 1 (or 2 / 3, NORM_FUSE) folded into the first additions as its ptxas did
NR_INLINE u32 sq_norm(u32 f0, u32 f1, u32 f2, u32 f3)
{
#if NORM_FUSE == 1
    const u32 p02 = h2_fma(f0, f0, h2_mul(f2, f2));
    const u32 p13 = h2_fma(f1, f1, h2_mul(f3, f3));
#elif NORM_FUSE == 2
    const u32 p02 = h2_fma(f2, f2, h2_mul(f0, f0));
    const u32 p13 = h2_fma(f3, f3, h2_mul(f1, f1));
#else
    const u32 p02 = h2_add(h2_mul(f0, f0), h2_mul(f2, f2));
    const u32 p13 = h2_add(h2_mul(f1, f1), h2_mul(f3, f3));
#endif
    u32 s = h2_add(p13, p02);
    s = h2_add(s, __shfl_xor_sync(0xffffffffu, s, 2));
    s = h2_add(s, __shfl_xor_sync(0xffffffffu, s, 1));
    s = h2_add(s, swap_halves(s));                                  // lo + hi in both halves
    return h2_rsqrt(h2_max(s, 0x04100410u));                        // f16(6.2e-5) floor, rsqrt.approx -> f16
}

NR_INLINE void body(const u8* __restrict__ rec, u32 lane, const u32 (&x)[4][2][4], u32 (&o)[4][2][4])
{
    const u32 t = lane & 3u;
    const Rec R = kRec;
    // ---- MLP: expansion 32 -> 128 in four passes of 32 hidden units, cubic SiLU, contraction 128 -> 32 seeded with
    //      f16(x) * residual_scale; pass p of the expansion feeds K block p of the contraction (NVIDIA's order)
    u32 y[4][4][2];                                                 // [tile][frag][row]
    FA XQ[4];                                                       // e4m3(x) as A operands (q8: no codes)
    {
        u32 rs[4];
#pragma unroll
        for (u32 j = 0; j < 4; ++j) rs[j] = ldw32(rec + RO(R.rs) + 4u * (t + 4u * j));
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            u32 v[4][2];
#pragma unroll
            for (u32 j = 0; j < 4; ++j)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    y[s][j][r] = h2_mul(x[s][r][j], rs[j]);
                    v[j][r] = x[s][r][j];
                }
            XQ[s] = fa_q(v);
        }
    }
#if MLP_ROLL                   // the four MLP passes as a loop (4x less code; nothing is indexed by the pass)
#pragma unroll 1
#else
#pragma unroll
#endif
    for (u32 pass = 0; pass < 4; ++pass) {
        FB be[4], bc[4];
        fb4(rec, R.exp + pass * 1024u, lane, be);
        fb4(rec, R.con + pass * 1024u, lane, bc);
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const FA& a = XQ[s];
            u32 h[4][2];
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                h[j][0] = h[j][1] = 0u;
                MMA_G(1, h[j], a, be[j]);
                h[j][0] = silu3(h[j][0]);
                h[j][1] = silu3(h[j][1]);
            }
            const FA ah = fa_q(h);
#pragma unroll
            for (u32 j = 0; j < 4; ++j) MMA_G(1, y[s][j], ah, bc[j]);
        }
    }
    // ---- QKV on e4m3(y); K and V of the whole window, then Q
    // e4m3(y) as A operands: kept (Y_KEEP=1) or made again from y (q8) for each of K, V and Q (Y_KEEP=0)
#if Y_KEEP
    FA YQ[4];
#pragma unroll
    for (u32 s = 0; s < 4; ++s) YQ[s] = fa_q(y[s]);
#define Y_OPERAND(s) YQ[s]
#else
#define Y_OPERAND(s) fa_q(y[s])
#endif
    FB KQ[8];                                                       // K: B operand per 8 keys (tile 2 s + half)
    FB VQ[4][2];                                                    // V: B operand [dim frag][k32 step (tile pair)]
    {
        FB bk[4];
        fb4(rec, R.qkv + 1024u, lane, bk);
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const FA a = Y_OPERAND(s);
            u32 k[4][2];
#pragma unroll
            for (u32 j = 0; j < 4; ++j) { k[j][0] = k[j][1] = 0u; MMA_G(2, k[j], a, bk[j]); }
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const u32 n = sq_norm(k[0][r], k[1][r], k[2][r], k[3][r]);
                fb_q_half(KQ[2u * s + r], 0u, h2_mul(k[0][r], n), h2_mul(k[1][r], n));
                fb_q_half(KQ[2u * s + r], 1u, h2_mul(k[2][r], n), h2_mul(k[3][r], n));
            }
        }
    }
    {
        FB bv[4];
        fb4(rec, R.qkv + 2048u, lane, bv);
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const FA a = Y_OPERAND(s);
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                u32 v[2] = {0u, 0u};
                MMA_G(2, v, a, bv[j]);
                // 8x8 b16 transposes: lane holds dim 8 j + g of tokens 2t, 2t+1 (and 8+2t, 9+2t)
                fb_q_half(VQ[j][s >> 1], s & 1u, amp_transpose8(v[0]), amp_transpose8(v[1]));
            }
        }
    }
    FA QQ[4];
    {
        FB bq[4];
        fb4(rec, R.qkv, lane, bq);
        const float hsf = *(const float*)(rec + RO(R.hs));
        const u32 hs = (u32)f16_of(hsf) * 0x10001u;
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const FA a = Y_OPERAND(s);
            u32 q[4][2];
#pragma unroll
            for (u32 j = 0; j < 4; ++j) { q[j][0] = q[j][1] = 0u; MMA_G(2, q[j], a, bq[j]); }
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const u32 n = sq_norm(q[0][r], q[1][r], q[2][r], q[3][r]);
#pragma unroll
                for (u32 j = 0; j < 4; ++j) q[j][r] = h2_mul(h2_mul(q[j][r], n), hs);
            }
            QQ[s] = fa_q(q);
        }
    }
    // ---- attention in two halves of two query tiles (the softmax denominators of a half are reduced together)
    u32 ars[4];
#pragma unroll
    for (u32 j = 0; j < 4; ++j) ars[j] = ldw32(rec + RO(R.ars) + 4u * (t + 4u * j));
#pragma unroll
    for (u32 half = 0; half < 2; ++half) {
        u32 E[2][8][2];                                             // [query tile of the half][key group][row]
        u32 part[2][2];                                             // partial denominators [query tile][row]
#pragma unroll
        for (u32 d = 0; d < 2; ++d) {
            const u32 qt = 2u * half + d;
            const FA& a = QQ[qt];
#pragma unroll
            for (u32 kt = 0; kt < 4; ++kt) {
                const V4 b = ldw(rec + RO(R.bias) + (qt * 4u + kt) * 512u + lane * 16u);
                E[d][2u * kt][0] = b.x; E[d][2u * kt][1] = b.y; E[d][2u * kt + 1u][0] = b.z; E[d][2u * kt + 1u][1] = b.w;
            }
#pragma unroll
            for (u32 nt = 0; nt < 8; ++nt) MMA_G(4, E[d][nt], a, KQ[nt]);
#pragma unroll
            for (u32 nt = 0; nt < 8; ++nt)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    u32 z = h2_fma(E[d][nt][r], 0x29C029C0u, 0x3D343D34u);
                    z = h2_min(h2_max(z, 0x3C203C20u), 0x3E473E47u);
                    E[d][nt][r] = (z << 5) + 0x7FF88000u;           // mantissa << 5 in each half: the exponential
                }
#pragma unroll
            for (u32 r = 0; r < 2; ++r)
                part[d][r] = h2_add(h2_add(h2_add(h2_add(E[d][0][r], E[d][1][r]), h2_add(E[d][2][r], E[d][3][r])),
                                           h2_add(E[d][4][r], E[d][5][r])), h2_add(E[d][6][r], E[d][7][r]));
        }
        // NVIDIA's transposed reduction: lane L ends with the total of (query tile L >> 4, row L & 15) over the four
        // lanes of its quad, summed in the order below (F0 26114-26166)
        u32 recip;
        {
            const u32 L = lane;
            const u32 A = part[0][0], B = part[0][1], Cc = part[1][0], D = part[1][1];
            const bool odd = (L & 1u) != 0u, p49 = (L & 2u) == 0u, p54 = (L & 8u) == 0u, p55 = (L & 16u) == 0u;
            const u32 v37 = odd ? B : A, v38 = odd ? A : B, v39 = odd ? D : Cc, v40 = odd ? Cc : D;
            const u32 v41 = p49 ? v37 : v39, v42 = p49 ? v39 : v37, v43 = p49 ? v38 : v40, v44 = p49 ? v40 : v38;
            const u32 s0 = ((L << 2) & 28u) + (L >> 3);
            const u32 v49 = __shfl_sync(0xffffffffu, v41, s0), v51 = __shfl_sync(0xffffffffu, v43, s0 ^ 1u);
            const u32 v53 = __shfl_sync(0xffffffffu, v42, s0 ^ 2u), v55 = __shfl_sync(0xffffffffu, v44, s0 ^ 3u);
            const u32 v56 = p54 ? v49 : v51, v57 = p54 ? v51 : v49, v58 = p54 ? v53 : v55, v59 = p54 ? v55 : v53;
            const u32 a72 = p55 ? v56 : v58, a75 = p55 ? v58 : v56, a73 = p55 ? v57 : v59, a78 = p55 ? v59 : v57;
            const u32 sum = h2_add(h2_add(h2_add(a72, a73), a75), a78);
            const u32 tot = low_twice(h2_add(low_twice(sum), high_twice(sum)));
            const u32 m = h2_max(tot, 0x04100410u);
            float lo, hi;
            amp_h2_unpack(m, lo, hi);
            recip = amp_h2_pack(rcp_approx(lo), rcp_approx(hi));
        }
#pragma unroll
        for (u32 d = 0; d < 2; ++d) {
            const u32 qt = 2u * half + d;
            const u32 g = lane >> 2;
            const u32 r0 = low_twice(__shfl_sync(0xffffffffu, recip, (g + 16u * d) & 31u));
            const u32 r1 = low_twice(__shfl_sync(0xffffffffu, recip, (g + 16u * d + 8u) & 31u));
            u32 P[8][2];
#pragma unroll
            for (u32 nt = 0; nt < 8; ++nt) { P[nt][0] = h2_mul(E[d][nt][0], r0); P[nt][1] = h2_mul(E[d][nt][1], r1); }
            const FA a0 = fa_q(*reinterpret_cast<const u32 (*)[4][2]>(&P[0][0]));
            const FA a1 = fa_q(*reinterpret_cast<const u32 (*)[4][2]>(&P[4][0]));
            u32 c[4][2];
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                c[j][0] = c[j][1] = 0u;
                MMA_G(8, c[j], a0, VQ[j][0]);
                MMA_G(8, c[j], a1, VQ[j][1]);
            }
            const FA ac = fa_q(c);
            FB bp[4];
            fb4(rec, R.proj, lane, bp);
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                u32 acc[2] = {h2_mul(y[qt][j][0], ars[j]), h2_mul(y[qt][j][1], ars[j])};   // attention residual seed
                MMA_G(16, acc, ac, bp[j]);
                o[qt][0][j] = acc[0];
                o[qt][1][j] = acc[1];
            }
        }
    }
}

// ---- the reprojected history (copied from prepost4.cu, bit for bit with NVIDIA's machine code there) --------------
NR_INLINE void history5(u64 tex, float qx, float qy, float fxs, float fys, float cm, float rm, float rc, float rr,
                        float su, float ou, float qu, float sv, float ov, float qv, F4& sum, float& r)
{
    const float cx = floorf(fmaf(qx, fxs, -0.5f)) + 0.5f, cy = floorf(fmaf(qy, fys, -0.5f)) + 0.5f;
    const float fx = fminf(fmaxf(fmaf(qx, fxs, -cx), 0.0f), 1.0f), fy = fminf(fmaxf(fmaf(qy, fys, -cy), 0.0f), 1.0f);
    const float x2 = fx * fx, y2 = fy * fy, x3 = fx * x2, y3 = fy * y2;
    const float w0x = fmaf(fx + x3, -0.5f, x2), w0y = fmaf(fy + y3, -0.5f, y2);
    const float w1x = fmaf(x3, 1.5f, -(x2 * 2.5f)) + 1.0f, w1y = fmaf(y3, 1.5f, -(y2 * 2.5f)) + 1.0f;
    const float w3x = (x3 - x2) * 0.5f, w3y = (y3 - y2) * 0.5f;
    const float w2x = ((1.0f - w0x) - w1x) - w3x, w2y = ((1.0f - w0y) - w1y) - w3y;
    const float wx = w1x + w2x, wy = w1y + w2y;
    const float t0x = fminf(fmaxf(cx - 1.0f, 0.5f), cm), t0y = fminf(fmaxf(cy - 1.0f, 0.5f), rm);
    const float t1x = fminf(fmaxf(fmaf(w2x, rcp_approx(wx), cx), 0.5f), cm), t1y = fminf(fmaxf(fmaf(w2y, rcp_approx(wy), cy), 0.5f), rm);
    const float t3x = fminf(fmaxf(cx + 2.0f, 0.5f), cm), t3y = fminf(fmaxf(cy + 2.0f, 0.5f), rm);
    const float u0 = qu * fmaf(su, rc * t0x, ou), u1 = qu * fmaf(su, rc * t1x, ou), u3 = qu * fmaf(su, rc * t3x, ou);
    const float v0 = qv * fmaf(sv, rr * t0y, ov), v1 = qv * fmaf(sv, rr * t1y, ov), v3 = qv * fmaf(sv, rr * t3y, ov);
    const F4 A = tex_rgba(tex, u0, v1), B = tex_rgba(tex, u1, v0), Cc = tex_rgba(tex, u1, v1), D = tex_rgba(tex, u1, v3);
    const F4 E = tex_rgba(tex, u3, v1);
    const float wA = w0x * wy, wB = w0y * wx, wC = wx * wy, wD = w3y * wx, wE = w3x * wy;
    r = rcp_approx((((wB + wA) + wC) + wD) + wE);
    sum.x = fmaf(wE, E.x, fmaf(wD, D.x, fmaf(wC, Cc.x, fmaf(wA, A.x, wB * B.x))));
    sum.y = fmaf(wE, E.y, fmaf(wD, D.y, fmaf(wC, Cc.y, fmaf(wA, A.y, wB * B.y))));
    sum.z = fmaf(wE, E.z, fmaf(wD, D.z, fmaf(wC, Cc.z, fmaf(wA, A.z, wB * B.z))));
    sum.w = 0.0f;
}

#if BLOCK != 2
// ---- pre-block: noise, colour and conditioning features (prepost4.cu, with NVIDIA's full conditioning logic) ------
NR_INLINE u32 pcg(u32 v) { v = (v >> ((v >> 28) + 4u)) ^ v; return v * 0x108EF2D9u; }
NR_INLINE float unit24(u32 s) { const u32 t = pcg(s); return (float)(((t >> 30) ^ (t >> 8)) + 1u) * 5.9604644775390625e-08f; }
NR_INLINE float lg2a(float x) { float d; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE float sqrta(float x) { float d; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE float sina(float x) { float d; asm("sin.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE float cosa(float x) { float d; asm("cos.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE u16 centred(float c, u16 scale)                   // f16(f16(f16(c) - 0.5) * f16(2 * +204))
{
    u16 d;
    asm("{\n .reg .b16 amp_a, amp_h;\n cvt.rn.f16.f32 amp_a, %1;\n mov.b16 amp_h, 0x3800;\n sub.rn.f16 amp_a, amp_a, amp_h;\n mul.rn.f16 %0, amp_a, %2;\n}"
        : "=h"(d) : "f"(c), "h"(scale));
    return d;
}
#endif

#if BLOCK == 1
extern "C" __global__ void __launch_bounds__(32) amp_pre(Params P)
#elif BLOCK == 3
extern "C" __global__ void __launch_bounds__(32) amp_pre_3108(Params P)
#else
extern "C" __global__ void __launch_bounds__(32) amp_post(Params P)
#endif
{
    const u32 lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
#if BLOCK != 2
    __shared__ __align__(16) u8 sm_f[64 * 32];                     // the 64 pixels' feature rows (16 x f16 each)
    const u8* const rec = (const u8*)P_U64(232);
    const int H = P_S32(248), W = P_S32(252), sx = 0, sy = 0;
#else
    const u8* const rec = (const u8*)P_U64(24);
    const int H = P_S32(32), W = P_S32(36), sx = P_S32(40), sy = P_S32(44);
#endif
    const int x0 = sx + 8 * (int)blockIdx.x, y0 = sy + 8 * (int)blockIdx.y;
    const int TW = W >> 2, TH = H >> 2, tx0 = x0 >> 2, ty0 = y0 >> 2;
    bool inside[4];
    u64 tile_at[4];                                             // byte offset of this lane's 16 bytes of tile s (tile layout)
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        inside[s] = ty >= 0 && ty < TH && tx >= 0 && tx < TW;
        tile_at[s] = (u64)(u32)(ty * TW + tx) * 512u + lane * 16u;
    }

    u32 x[4][2][4];                                             // the block's input x: [tile][row g / g+8][frag]
#if BLOCK != 2
    // ---- features of the window's 64 pixels: row (16 x f16) of token (tile s, m) at sm_f + (16 s + m) * 32 --------
    {
        const u64 tex = P_U64(0);
        const int frows = P_S32(216), fcols = P_S32(220);
        const float fr = (float)frows, fc = (float)fcols;
        const u64 hist = P_U64(8), mot = P_U64(16);
        const float rcfc = rcp_approx(fc), rcfr = rcp_approx(fr);
        const u32 seedmix = (u32)P_S32(208) * 0x9E3779B9u;
        const u16 cscale = f16_of(P_F32(204) + P_F32(204));
        // conditioning (F0 14536-14560, 15020-15065; no control-mask texture): features 10-15
        const float tone = P_F32(176), structure = P_F32(180), style = P_F32(184), skinp = P_F32(188), otherp = P_F32(192);
#if BLOCK == 1
        const bool swap = P_S32(196) > 15;
#else
        const bool swap = false;                                // 310.8 has no slot selector (its fixed layout = 15)
#endif
        const bool automask = P_S32(200) != 0;
        const bool p1 = fmaxf(skinp, otherp) >= 0.0f;
        u16 f12, f13, f14;
        const u16 f11 = f16_of(1.0f * tone);
        if (!automask) {
            f12 = f16_of(1.0f * structure);
            f13 = f14 = 0;
        } else {
            f12 = f16_of(p1 ? 1.0f : 1.0f * structure);
            f13 = f16_of(p1 ? (skinp < 0.0f ? structure : skinp) : -1.0f);
            f14 = f16_of(p1 ? (otherp < 0.0f ? structure : otherp) : -1.0f);
        }
        const u32 cond0 = (u32)f16_of(style) | ((u32)f11 << 16);                     // style, tone
        const u32 cond1 = (u32)f12 | ((u32)(swap ? (u16)0 : f13) << 16);             // structure, skin
        const u32 cond2 = (u32)(swap ? f13 : f14) | ((u32)(swap ? f14 : (u16)0) << 16);  // other, 0
#pragma unroll
        for (u32 i = 0; i < 2; ++i) {
            const u32 idx = lane + 32u * i, r = idx >> 3, c = idx & 7u;
            const int y = y0 + (int)r, xx = x0 + (int)c;
            const int ym = y < frows ? y : 2 * frows - y - 2, xm = xx < fcols ? xx : 2 * fcols - xx - 2;
            const float xn = __fdividef((float)xm + 0.5f, fc), yn = __fdividef((float)ym + 0.5f, fr);
            const float u = P_F32(156) * fmaf(xn, P_F32(148), P_F32(140));
            const float v = P_F32(160) * fmaf(yn, P_F32(152), P_F32(144));
            const F4 col = tex_rgba(tex, u, v);
            const u32 cr = centred(col.x, cscale), cg = centred(col.y, cscale), cb = centred(col.z, cscale);
            u32 hr = cr, hg = cg, hb = cb;                      // "previous colour": the colour itself without history
            if (hist != 0ull && mot != 0ull) {                  // depth (+24) must be null (not implemented)
                const F4 mv = tex_rgba(mot, P_F32(84) * fmaf(xn, P_F32(76), P_F32(68)), P_F32(88) * fmaf(yn, P_F32(80), P_F32(72)));
                F4 hs; float hn;
                history5(hist, fmaf(mv.x, P_F32(164), xn), fmaf(mv.y, P_F32(168), yn), fc, fr, fc - 0.5f, fr - 0.5f, rcfc, rcfr,
                         P_F32(52), P_F32(44), P_F32(60), P_F32(56), P_F32(48), P_F32(64), hs, hn);
                hr = centred(hn * hs.x, cscale); hg = centred(hn * hs.y, cscale); hb = centred(hn * hs.z, cscale);
            }
            u32 n01, n2;
            {
                const u32 base = ((u32)xx * 0x8DA6B343u) ^ seedmix ^ ((u32)y * 0xD8163841u) ^ 0x243F6A88u;
                const u32 t0 = pcg(base), hh = (t0 >> 22) ^ t0;
                const float ua = unit24(hh * 0x2C9277B5u + 0xAC564B05u), ub = unit24(hh * 0xFA6DC5F9u + 0x4712A88Eu);
                const float uc = unit24(hh * 0xCAA5B80Du + 0x21DD796Bu), ud = unit24(hh * 0x83232C31u + 0x3463E0ACu);
                const float ra = sqrta(lg2a(ua) * 0.693147182464599609f * -2.0f), rc = sqrta(lg2a(uc) * 0.693147182464599609f * -2.0f);
                const float aa = ub * 6.28318548202514648f, ac = ud * 6.28318548202514648f;
                n01 = (u32)f16_of(ra * cosa(aa)) | ((u32)f16_of(ra * sina(aa)) << 16);
                n2 = (u32)f16_of(rc * cosa(ac)) | (0x3C00u << 16);                       // third noise, constant 1
            }
            const u32 s = (r >> 2) * 2u + (c >> 2), m = (r & 3u) * 4u + (c & 3u);
            const V4 lo = {n01, n2, cr | (cg << 16), cb | (hr << 16)};                   // n0 n1 n2 1 R G B R'
            const V4 hi = {hg | (hb << 16), cond0, cond1, cond2};                        // G' B' style tone structure skin other 0
            *(V4*)(sm_f + (16u * s + m) * 32u) = lo;
            *(V4*)(sm_f + (16u * s + m) * 32u + 16u) = hi;
        }
        __syncwarp();
    }
    // ---- input lift 16 -> 32: NVIDIA's f16 m16n8k16 product, f16 accumulator from zero (F0 15467-15575)
    {
        const V4 l0 = ldw(rec + RO(R_LIFT) + lane * 16u), l1 = ldw(rec + RO(R_LIFT) + 512u + lane * 16u);
        const u32 lb[4][2] = {{l0.x, l0.y}, {l0.z, l0.w}, {l1.x, l1.y}, {l1.z, l1.w}};
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const u8* const frow0 = sm_f + (16u * s + g) * 32u, * const frow1 = frow0 + 8u * 32u;
            const u32 a0 = *(const u32*)(frow0 + 4u * t), a1 = *(const u32*)(frow1 + 4u * t);
            const u32 a2 = *(const u32*)(frow0 + 16u + 4u * t), a3 = *(const u32*)(frow1 + 16u + 4u * t);
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                u32 c[2] = {0u, 0u};
                mma16_h(c, a0, a1, a2, a3, lb[j][0], lb[j][1]);
                x[s][0][j] = c[0];
                x[s][1][j] = c[1];
            }
        }
    }
#else
    // ---- post-block input: up = the half-size main input replicated 2x (view layout), skip = encoder skip (tile
    //      layout); blend = f16(skip * skip_gain + f16(up * main_gain)) (fma form, LAB-verified in prepost4.cu)
    {
        const u8* const xin = (const u8*)P_U64(0);
        const u8* const skin = (const u8*)P_U64(8);
        const int hH = (H + 1) >> 1, hW = (W + 1) >> 1, vr = (hH + 3) & ~3, vc = (hW + 3) & ~3;
        u32 mg[4], sg[4];
#pragma unroll
        for (u32 j = 0; j < 4; ++j) { mg[j] = ldw32(rec + RO(R_MGAIN) + 4u * (t + 4u * j)); sg[j] = ldw32(rec + RO(R_SGAIN) + 4u * (t + 4u * j)); }
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            u32 up[2][4], sk[2][4];
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                const u32 m = g + 8u * rh;
                const int py = (ty0 + (int)(s >> 1)) * 4 + (int)(m >> 2), px = (tx0 + (int)(s & 1u)) * 4 + (int)(m & 3u);
                const int cy = min(py >> 1, hH - 1), cx = min(px >> 1, hW - 1);
                const bool in = inside[s] && py >= 0 && px >= 0;
#pragma unroll
                for (u32 a = 0; a < 2; ++a) {
                    const u64 at = ((u64)(u32)((int)a * vr + (in ? cy : 0)) * (u32)vc + (u32)(in ? cx : 0)) * 16u + t * 4u;
                    const u32 word = in ? amp_load_cg32(xin + at) : 0u;
                    amp_e4m3x4_to_h2x2(word, up[rh][2 * a], up[rh][2 * a + 1]);
                }
            }
            const V4 skw = inside[s] ? amp_load_cg(skin + tile_at[s]) : V4{0u, 0u, 0u, 0u};
            amp_e4m3x4_to_h2x2(skw.x, sk[0][0], sk[0][1]); amp_e4m3x4_to_h2x2(skw.y, sk[1][0], sk[1][1]);
            amp_e4m3x4_to_h2x2(skw.z, sk[0][2], sk[0][3]); amp_e4m3x4_to_h2x2(skw.w, sk[1][2], sk[1][3]);
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
                for (u32 j = 0; j < 4; ++j)
                    x[s][rh][j] = inside[s] ? h2_fma(sk[rh][j], sg[j], h2_mul(up[rh][j], mg[j])) : 0u;
        }
    }
#endif

    // ---- the body ------------------------------------------------------------------------------------------------
    u32 o[4][2][4];
#if DBG_STAGE == 2
#pragma unroll
    for (u32 s = 0; s < 4; ++s)
#pragma unroll
        for (u32 r = 0; r < 2; ++r)
#pragma unroll
            for (u32 j = 0; j < 4; ++j) o[s][r][j] = x[s][r][j];
#else
    body(rec, lane, x, o);
#endif

#if BLOCK != 2
    // ---- pre-block outputs: e4m3(o) in tile layout (+224); 2x2 mean of o in view layout (+256) ------------------
    {
        u8* const skip_out = (u8*)P_U64(224);
#pragma unroll
        for (u32 s = 0; s < 4; ++s)
            if (inside[s]) {
                const V4 packed = {enc(o[s][0][0], o[s][0][1]), enc(o[s][1][0], o[s][1][1]), enc(o[s][0][2], o[s][0][3]), enc(o[s][1][2], o[s][1][3])};
                asm volatile("st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4};"
                             :: "l"((u64)(skip_out + tile_at[s])), "r"(packed.x), "r"(packed.y), "r"(packed.z), "r"(packed.w) : "memory");
            }
        // column pairs (lane ^ 4), then row pairs (lane ^ 16), f16 additions, times 0.25 (prepost4.cu, bit-exact)
        const u32 ci = (g & 1u) + 2u * (g >> 2);
        const bool b2 = (g & 1u) != 0u, b4 = (g >> 2) != 0u;
        u32 pool[2][4];
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                u32 k0 = b2 ? o[1][rh][j] : o[0][rh][j];
                u32 k1 = b2 ? o[3][rh][j] : o[2][rh][j];
                const u32 s0 = b2 ? o[0][rh][j] : o[1][rh][j];
                const u32 s1 = b2 ? o[2][rh][j] : o[3][rh][j];
                k0 = h2_add(k0, __shfl_xor_sync(0xffffffffu, s0, 4));
                k1 = h2_add(k1, __shfl_xor_sync(0xffffffffu, s1, 4));
                const u32 a = h2_add(b4 ? k1 : k0, __shfl_xor_sync(0xffffffffu, b4 ? k0 : k1, 16));
                pool[rh][j] = h2_mul(a, 0x34003400u);
            }
        u8* const pooled = (u8*)P_U64(256);
        const int pr = P_S32(264), pc = P_S32(268), hH = H >> 1, hW = W >> 1;
        const int px = (x0 >> 1) + (int)((ci & 1u) * 2u + ((g >> 1) & 1u));
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh) {
            const int py = (y0 >> 1) + (int)((ci >> 1) * 2u + rh);
            if (py >= 0 && py < hH && px >= 0 && px < hW) {
#pragma unroll
                for (u32 a = 0; a < 2; ++a)
                    *(u32*)(pooled + ((u64)(u32)((int)a * pr + py) * (u32)pc + (u32)px) * 16u + t * 4u) = enc(pool[rh][2 * a], pool[rh][2 * a + 1]);
            }
        }
        // zeros in the padded part of the pooled view (rows hH .. pr - 1, columns hW .. pc - 1), as NVIDIA's kernel:
        // the last column of windows fills the extra columns of its rows, the last row of windows the extra rows
        const bool lastx = blockIdx.x == gridDim.x - 1u, lasty = blockIdx.y == gridDim.y - 1u;
        if (lastx && pc > hW) {
            for (int e = (int)lane; e < 2 * 4 * 4 * (pc - hW); e += 32) {        // groups x rows x words x columns
                const int a = e & 1, rr = (e >> 1) & 3, wd = (e >> 3) & 3, cc = hW + (e >> 5);
                const int py = (y0 >> 1) + rr;
                if (py < pr) *(u32*)(pooled + ((u64)(u32)(a * pr + py) * (u32)pc + (u32)cc) * 16u + (u32)wd * 4u) = 0u;
            }
        }
        if (lasty && pr > hH) {
            const int c0 = x0 >> 1, c1 = lastx ? pc : min(c0 + 4, pc);
            for (int e = (int)lane; e < 2 * 4 * (pr - hH) * (c1 - c0); e += 32) {
                const int a = e & 1, wd = (e >> 1) & 3, rest = e >> 3, cc = c0 + rest % (c1 - c0), py = hH + rest / (c1 - c0);
                *(u32*)(pooled + ((u64)(u32)(a * pr + py) * (u32)pc + (u32)cc) * 16u + (u32)wd * 4u) = 0u;
            }
        }
    }
#else
    // ---- post-block outputs: projection 32 -> 4 (outputs 0-3), compose with the frame, temporal blend, surface store
    {
        const u64 surf = P_U64(16), tex = P_U64(56);
        const int fcols = P_S32(172), frows = P_S32(176);
        const float rw = P_F32(48);
        const bool final_clamp = P_S32(52) != 0;
        const u64 hist = P_U64(88), mot = P_U64(96), bscale = P_U64(104);
        // NVIDIA's out_project record: two k16 slices (K 0-15, 16-31) of 16 outputs; outputs 0-7 in words x, y
        const V4 w0 = ldw(rec + RO(R_OUTP) + lane * 16u), w1 = ldw(rec + RO(R_OUTP) + 512u + lane * 16u);
        float RR[4][4];                                         // per tile: this lane's pixel's outputs 0-3 (lanes t < 2)
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            float rg[2], rg8[2];                                // this lane's outputs 2t, 2t + 1 of rows g and g + 8
            u32 c[2] = {0u, 0u};
            mma16_h(c, o[s][0][0], o[s][1][0], o[s][0][1], o[s][1][1], w0.x, w0.y);
            mma16_h(c, o[s][0][2], o[s][1][2], o[s][0][3], o[s][1][3], w1.x, w1.y);
            amp_h2_unpack(c[0], rg[0], rg[1]);
            amp_h2_unpack(c[1], rg8[0], rg8[1]);
            // lane t = 0 finishes row g (gets outputs 2, 3 from lane 1), lane t = 1 row g + 8 (outputs 0, 1 from lane 0)
            const bool odd = (t & 1u) != 0u;
            const float send0 = odd ? rg[0] : rg8[0], send1 = odd ? rg[1] : rg8[1];
            const float got0 = __shfl_xor_sync(0xffffffffu, send0, 1), got1 = __shfl_xor_sync(0xffffffffu, send1, 1);
            RR[s][0] = odd ? got0 : rg[0]; RR[s][1] = odd ? got1 : rg[1];
            RR[s][2] = odd ? rg8[0] : got0; RR[s][3] = odd ? rg8[1] : got1;
        }
        // compose (+ temporal) per pixel: two tiles per pass over all 32 lanes (lanes t = 2, 3 take tile s + 1)
#pragma unroll
        for (u32 pass = 0; pass < 2; ++pass) {
            const u32 sa = 2u * pass, sb = 2u * pass + 1u;
            float r0 = RR[sa][0], r1 = RR[sa][1], r2 = RR[sa][2], r3 = RR[sa][3];
            {
                const float q0 = __shfl_xor_sync(0xffffffffu, RR[sb][0], 2), q1 = __shfl_xor_sync(0xffffffffu, RR[sb][1], 2);
                const float q2 = __shfl_xor_sync(0xffffffffu, RR[sb][2], 2), q3 = __shfl_xor_sync(0xffffffffu, RR[sb][3], 2);
                if (t >= 2u) { r0 = q0; r1 = q1; r2 = q2; r3 = q3; }
            }
            const u32 s = t >= 2u ? sb : sa;
            const u32 m = g + 8u * (t & 1u);
            const int py = (ty0 + (int)(s >> 1)) * 4 + (int)(m >> 2), px = (tx0 + (int)(s & 1u)) * 4 + (int)(m & 3u);
            const bool in_s = s == 0u ? inside[0] : s == 1u ? inside[1] : s == 2u ? inside[2] : inside[3];
            if (in_s && px >= 0 && py >= 0 && px < W && py < H && px < fcols && py < frows) {
                float v0, v1, v2, alpha = 0.0f;
                const float xn = __fdividef((float)px + 0.5f, (float)fcols), yn = __fdividef((float)py + 0.5f, (float)frows);
                if (tex != 0ull) {
                    const F4 f = tex_rgba(tex, P_F32(80) * fmaf(P_F32(72), xn, P_F32(64)), P_F32(84) * fmaf(P_F32(76), yn, P_F32(68)));
                    v0 = fmaf(rw, r0, fmaf(f.x, 0.125f, -0.0625f));
                    v1 = fmaf(rw, r1, fmaf(f.y, 0.125f, -0.0625f));
                    v2 = fmaf(rw, r2, fmaf(f.z, 0.125f, -0.0625f));
                } else {
                    v0 = rw * r0; v1 = rw * r1; v2 = rw * r2;
                }
                if (final_clamp) {
                    v0 = fminf(1.0f, fmaxf(0.0f, fmaf(v0, 8.0f, 0.5f)));
                    v1 = fminf(1.0f, fmaxf(0.0f, fmaf(v1, 8.0f, 0.5f)));
                    v2 = fminf(1.0f, fmaxf(0.0f, fmaf(v2, 8.0f, 0.5f)));
                    alpha = 1.0f;
                }
                // temporal accumulation (F0 43905-44084): blend weight from +104 (f16, inf -> 0, clamped to [0, 1]),
                // only after the final clamp, with history (+88) and motion (+96); +112: motion vectors present
                float bw = 1.0f;
                if (bscale != 0ull) {
                    const float b = amp_h_to_f32(*(const u16*)bscale);
                    bw = fabsf(b) == __int_as_float(0x7F800000) ? 0.0f : fminf(1.0f, fmaxf(0.0f, b));
                }
                if (final_clamp && hist != 0ull && bw > 0.0f && mot != 0ull) {
                    const F4 mv = tex_rgba(mot, P_F32(156) * fmaf(P_F32(148), xn, P_F32(140)), P_F32(160) * fmaf(P_F32(152), yn, P_F32(144)));
                    const bool have = P_S32(112) != 0;
                    const float qx = have ? fmaf(P_F32(164), mv.x, xn) : xn, qy = have ? fmaf(P_F32(168), mv.y, yn) : yn;
                    const float fcf = (float)fcols, frf = (float)frows;
                    F4 hs; float hn;
                    history5(hist, qx, qy, fcf, frf, fcf - 0.5f, frf - 0.5f, rcp_approx(fcf), rcp_approx(frf),
                             P_F32(124), P_F32(116), P_F32(132), P_F32(128), P_F32(120), P_F32(136), hs, hn);
                    const float a = fminf(1.0f, fmaxf(0.0f, rcp_approx(amp_ex2_approx(r3 * -1.44269502162933350f) + 1.0f) * bw));
                    v0 = fmaf(a, fmaf(hs.x, hn, -v0), v0);              // (s * r - v): ptxas folded the mul (SASS: FFMA)
                    v1 = fmaf(a, fmaf(hs.y, hn, -v1), v1);
                    v2 = fmaf(a, fmaf(hs.z, hn, -v2), v2);
                }
                asm volatile("sust.p.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                             :: "l"(surf), "r"(px), "r"(py), "f"(v0), "f"(v1), "f"(v2), "f"(alpha) : "memory");
            }
        }
    }
#endif
}

#if AMP_PREPARED
// NrHost prepare ABI (NrHost.cpp PrepareParams): the fp8 record in, the prepared copy (2x the record bytes) out
struct PrepareParams { const u8* in; u8* out; u32 inBytes, outBytes; int launch, pad; };
static_assert(sizeof(PrepareParams) == 32, "NrHost prepare ABI");
NR_INLINE void prepare_matrix(const PrepareParams& p, u32 off, u32 bytes)
{
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 16u; i += blockDim.x * gridDim.x) {
        const V4 v = *(const V4*)(p.in + off + 16u * i);
        u8* const to = p.out + 2u * off + (i / 32u) * 1024u + (i & 31u) * 16u;
        V4 a, b;
        amp_e4m3x4_to_h2x2_finite(v.x, a.x, a.y);
        amp_e4m3x4_to_h2x2_finite(v.y, a.z, a.w);
        amp_e4m3x4_to_h2x2_finite(v.z, b.x, b.y);
        amp_e4m3x4_to_h2x2_finite(v.w, b.z, b.w);
        *(V4*)to = a;
        *(V4*)(to + 512u) = b;
    }
}
// a piece the kernel reads unchanged (f16 / f32 values): bytes [off, off + bytes) -> [2 off, 2 off + bytes)
NR_INLINE void prepare_raw(const PrepareParams& p, u32 off, u32 bytes)
{
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 4u; i += blockDim.x * gridDim.x)
        *(u32*)(p.out + 2u * off + 4u * i) = *(const u32*)(p.in + off + 4u * i);
}
extern "C" __global__ void amp_prepare_prepost(const PrepareParams p)
{
    const Rec R = kRec;
    if (p.outBytes < 2u * p.inBytes) __trap();
    prepare_matrix(p, R.exp, 4096u);
    prepare_matrix(p, R.con, 4096u);
    prepare_matrix(p, R.qkv, 3072u);
    prepare_matrix(p, R.proj, 1024u);
    prepare_raw(p, R.rs, 64u);
    prepare_raw(p, R.ars, 64u);
    prepare_raw(p, R.hs, 16u);
    prepare_raw(p, R.bias, 8192u);
#if BLOCK == 2
    prepare_raw(p, R_MGAIN, 64u);
    prepare_raw(p, R_SGAIN, 64u);
    prepare_raw(p, R_OUTP, 1024u);
#else
    prepare_raw(p, R_LIFT, 1024u);
#endif
}
#endif

#if AMP_UNIT_TEST
// lab-only (prepost_lab.py amp-unit): for every f16 pair word i in [0, 65536): ref = dec_act of enc, got = q8
extern "C" __global__ void amp_unit_q8(const u32* __restrict__ in, u32* __restrict__ ref, u32* __restrict__ got, u32 n)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u32 h = in[i];
    u32 lo, hi;
    dec_act(amp_h2x2_to_e4m3x4(h, h ^ 0x80008000u), lo, hi);
    ref[2u * i] = lo; ref[2u * i + 1u] = hi;
    got[2u * i] = q8(h); got[2u * i + 1u] = q8(h ^ 0x80008000u);
}
// lab-only: the fast encoder and the non-negative q8 of amp_common.cuh against the exact ones (ref / got per word:
// [0] codes of h, [1] codes of -h, [2] q8_pos of |h| (exact form: finite decode of the codes))
extern "C" __global__ void amp_unit_fast(const u32* __restrict__ in, u32* __restrict__ ref, u32* __restrict__ got, u32 n)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u32 h = in[i], m = h & 0x7FFF7FFFu;
    ref[3u * i] = amp_h2_to_e4m3x2(h);
    ref[3u * i + 1u] = amp_h2_to_e4m3x2(h ^ 0x80008000u);
    got[3u * i] = amp_prmt(amp_h2_to_e4m3x2_fast(h), 0u, 0x4420u);
    got[3u * i + 1u] = amp_prmt(amp_h2_to_e4m3x2_fast(h ^ 0x80008000u), 0u, 0x4420u);
    const u32 mc = amp_h2_min(m, 0x5F005F00u);                       // the positive form needs x < 448 (and no NaN)
    ref[3u * i + 2u] = amp_e4m3x2_to_h2_finite(amp_h2_to_e4m3x2(mc));
    got[3u * i + 2u] = amp_h2_q8_pos(mc);
}
#endif
