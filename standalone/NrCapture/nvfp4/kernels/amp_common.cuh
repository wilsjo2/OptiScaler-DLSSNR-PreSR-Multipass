// amp_common.cuh: shared helpers of the Ampere (sm_86) FP16 replacement kernels of the DLSS-NR FP8 network (amp_*.cu).
//
// By default the helpers use only instructions available on sm_86. The NVFP4 host builds its FP8 layers
// with NR_NATIVE_FP8=1 on sm_120 to use hardware E4M3 conversion and matrix multiplication. Include with
//     #include "amp_common.cuh"
// and compile with --include-path=<nvfp4/kernels> (amp_legal.py and amp_common_test.py pass it), --fmad=false.
// The same source is built for sm_86 (the legality proof and the RTX 30 build) and for sm_120 (not sm_120a) to run on
// RTX 50 hardware.
//
// Phase A numeric target: reproduce NVIDIA's FP8 kernels' arithmetic as closely as Ampere allows.
//   * e4m3 -> f16 decode is exact for every code, in integer code plus one f16 multiply by 2^8 (no .ftz, so e4m3
//     subnormals m * 2^-9 come out exact). amp_e4m3x2_to_h2 matches cvt.rn.f16x2.e4m3x2 for all 256 codes, the two
//     NaN codes 0x7F / 0xFF included; the _finite forms skip the NaN fix-up and give +-480 for those two codes
//     (no record of the shipped sets holds them; a satfinite store never writes them from a finite value). The
//     hardware decodes both NaN codes to 0x7FFF (no sign).
//   * f16 / f32 -> e4m3 rounding matches cvt.rn.satfinite.e4m3x2.{f16x2,f32}: round to nearest even, saturate finite
//     values and infinities at +-448, NaN -> 0x7F (with the sign bit as the hardware gives it, see AMP_NAN_E4M3),
//     results below 2^-6 on the subnormal grid of 2^-9. The f32 form rounds once, directly from f32. A NaN input gives
//     0x7F whatever its sign (as the hardware does).
//   * One NVIDIA  mma.m16n8k32.row.col.f16.e4m3.e4m3.f16  becomes two  mma.m16n8k16.row.col.f16.f16.f16.f16  on the
//     SAME registers, no shuffles: decode the e4m3 fragments in place.
//       A (k32, e4m3)  a[0]=(row g, K 4t..4t+3)  a[1]=(g+8, 4t..4t+3)  a[2]=(g, 4t+16..19)  a[3]=(g+8, 4t+16..19)
//       A (k16, f16)   r[0]=(g, 2t..2t+1)        r[1]=(g+8, 2t..2t+1)  r[2]=(g, 2t+8..9)     r[3]=(g+8, 2t+8..9)
//       B (k32, e4m3)  b[0]=(K 4t..4t+3, col g)  b[1]=(K 4t+16..19, col g)
//       B (k16, f16)   q[0]=(K 2t..2t+1, col g)  q[1]=(K 2t+8..9, col g)
//     First k16 step:  r[0] = bytes 0,1 of a[0], r[2] = bytes 2,3 of a[0], r[1] / r[3] = bytes 0,1 / 2,3 of a[1];
//                      q[0] = bytes 0,1 of b[0], q[1] = bytes 2,3 of b[0].
//     Second k16 step: the same with a[2], a[3] and b[1].
//     So k16 slot 2t+j holds K 4t+j and slot 2t+8+j holds K 4t+2+j (j = 0, 1), identically on A and B: a bijection
//     of K, hence the same sum. The accumulator (C/D) layout of the two instructions is identical:
//       c[0] = (row g, cols 2t, 2t+1), c[1] = (row g+8, cols 2t, 2t+1), low half first.
//     Order: K 0..15 of the k32 step first (a[0], a[1], b[0]), then K 16..31. The f16 accumulator is rounded after
//     each k16 step (NVIDIA's single k32 instruction rounds once per 32 products): results are not bit-identical
//     in general; amp_common_test.py measures how far they are (identical %, within one f16 step %, worst).
//     Accumulator seeding (residual as the starting accumulator) is unchanged: pass it as c.
//   * If operands are first decoded to f16 into shared memory in natural K order and read with ldmatrix, the plain
//     m16n8k16 fragments apply and no relabelling is needed; the relabelling matters only when e4m3 fragments that
//     were loaded in NVIDIA's k32 layout are decoded in registers.
#pragma once

#ifndef NR_NATIVE_FP8
#define NR_NATIVE_FP8 0
#endif

typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;

#define AMP_INLINE static __forceinline__ __device__

struct __align__(16) amp_v4 { u32 x, y, z, w; };

// ---- integer helpers -------------------------------------------------------------------------------------------
AMP_INLINE u32 amp_prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
AMP_INLINE void amp_split(u32 w, u16& lo, u16& hi) { asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(w)); }
AMP_INLINE u32 amp_join(u16 lo, u16 hi) { u32 d; asm("mov.b32 %0, {%1, %2};" : "=r"(d) : "h"(lo), "h"(hi)); return d; }

// ---- float16 pairs packed in a word (low half first); all round to nearest even, none flushes subnormals ---------
AMP_INLINE u32 amp_h2_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
AMP_INLINE u32 amp_h2_mul(u32 a, u32 b) { u32 d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
AMP_INLINE u32 amp_h2_add(u32 a, u32 b) { u32 d; asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
AMP_INLINE u32 amp_h2_sub(u32 a, u32 b) { u32 d; asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
AMP_INLINE u32 amp_h2_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
AMP_INLINE u32 amp_h2_min(u32 a, u32 b) { u32 d; asm("min.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
AMP_INLINE u32 amp_h2_abs(u32 a) { u32 d; asm("abs.f16x2 %0, %1;" : "=r"(d) : "r"(a)); return d; }
// max/min of |a|, |b| with the sign of a XOR b (sm_86 and up)
AMP_INLINE u32 amp_h2_xamax(u32 a, u32 b) { u32 d; asm("max.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
AMP_INLINE u32 amp_h2_xamin(u32 a, u32 b) { u32 d; asm("min.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
AMP_INLINE float amp_xamaxf(float a, float b) { float d; asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
// two floats -> float16 pair, round to nearest even (lo in the low half)
AMP_INLINE u32 amp_h2_pack(float lo, float hi) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
AMP_INLINE void amp_h2_unpack(u32 h, float& lo, float& hi)
{
    asm("{\n .reg .b16 amp_l, amp_u;\n mov.b32 {amp_l, amp_u}, %2;\n cvt.f32.f16 %0, amp_l;\n cvt.f32.f16 %1, amp_u;\n}"
        : "=f"(lo), "=f"(hi) : "r"(h));
}
AMP_INLINE float amp_h_to_f32(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
AMP_INLINE float amp_rsqrt_approx(float x) { float d; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
AMP_INLINE float amp_rcp_approx(float x) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
AMP_INLINE float amp_ex2_approx(float x) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
AMP_INLINE float amp_add_rn(float a, float b) { float d; asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
AMP_INLINE float amp_mul_rn(float a, float b) { float d; asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
AMP_INLINE float amp_fma_rn(float a, float b, float c) { float d; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
// 8x8 transpose of b16 elements across the warp (sm_75 and up)
AMP_INLINE u32 amp_transpose8(u32 h) { u32 d; asm volatile("movmatrix.sync.trans.aligned.m8n8.b16 %0, %1;" : "=r"(d) : "r"(h)); return d; }

// ---- e4m3 -> float16, exact ------------------------------------------------------------------------------------
// The f16 bit pattern ((b & 0x7F) << 7) | ((b & 0x80) << 8) is the e4m3 value times 2^-8 (normals and subnormals),
// so one exact f16 multiply by 256 (0x5C00) finishes the decode.
#define AMP_H2_256 0x5C005C00u
// the f16 that cvt.rn.f16x2.e4m3x2 gives for both NaN codes 0x7F and 0xFF (measured on the RTX 5090 by
// amp_common_test.py: 0x7FFF, the sign is dropped)
#define AMP_NAN_H 0x7FFFu

// two codes (byte 0 -> low half, byte 1 -> high half) -> float16 pair, times 2^-8 (no multiply; exact)
AMP_INLINE u32 amp_e4m3x2_to_h2_scaled(u32 pair)
{
    const u32 y = amp_prmt(pair, 0u, 0x1404u);                      // b1 << 24 | b0 << 8
    return (y & 0x80008000u) | ((y >> 1) & 0x3F803F80u);
}
// two codes -> float16 pair, exact for every code except the NaN codes (they give +-480)
AMP_INLINE u32 amp_e4m3x2_to_h2_finite(u32 pair) {
#if NR_NATIVE_FP8
    u32 h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(u16(pair)));
    return h;
#else
 return amp_h2_mul(amp_e4m3x2_to_h2_scaled(pair), AMP_H2_256);
#endif
}
// two codes -> float16 pair, identical to cvt.rn.f16x2.e4m3x2 for all 65536 code pairs
AMP_INLINE u32 amp_e4m3x2_to_h2(u32 pair)
{
#if NR_NATIVE_FP8
    return amp_e4m3x2_to_h2_finite(pair);
#else
    const u32 y = amp_prmt(pair, 0u, 0x1404u);
    const u32 h = amp_h2_mul((y & 0x80008000u) | ((y >> 1) & 0x3F803F80u), AMP_H2_256);
    const u32 n = (((y & 0x7F007F00u) + 0x01000100u) >> 15) & 0x00010001u;   // 1 in a half whose code is 0x7F / 0xFF
    const u32 nan = n * 0xFFFFu;
    return (h & ~nan) | (n * AMP_NAN_H);
#endif
}
// four codes (byte 0 first) -> two float16 pairs: lo = bytes 0, 1; hi = bytes 2, 3
AMP_INLINE void amp_e4m3x4_to_h2x2_finite(u32 w, u32& lo, u32& hi)
{
#if NR_NATIVE_FP8
    lo = amp_e4m3x2_to_h2_finite(w);
    hi = amp_e4m3x2_to_h2_finite(w >> 16);
#else
    lo = amp_h2_mul((amp_prmt(w, 0u, 0x1404u) & 0x80008000u) | ((amp_prmt(w, 0u, 0x1404u) >> 1) & 0x3F803F80u), AMP_H2_256);
    hi = amp_h2_mul((amp_prmt(w, 0u, 0x3424u) & 0x80008000u) | ((amp_prmt(w, 0u, 0x3424u) >> 1) & 0x3F803F80u), AMP_H2_256);
#endif
}
AMP_INLINE void amp_e4m3x4_to_h2x2(u32 w, u32& lo, u32& hi)
{
    lo = amp_e4m3x2_to_h2(w & 0xFFFFu);
    hi = amp_e4m3x2_to_h2(w >> 16);
}
// one code -> float16 bits
AMP_INLINE u16 amp_e4m3_to_h(u32 b) { return (u16)(amp_e4m3x2_to_h2(b & 0xFFu) & 0xFFFFu); }
// four codes -> four floats (byte 0 first), exact
AMP_INLINE void amp_e4m3x4_to_f32(u32 w, float* v)
{
    u32 lo, hi;
    amp_e4m3x4_to_h2x2(w, lo, hi);
    amp_h2_unpack(lo, v[0], v[1]);
    amp_h2_unpack(hi, v[2], v[3]);
}

// ---- float16 / float32 -> e4m3 (cvt.rn.satfinite.e4m3x2 in integer code) ------------------------------------------
// e4m3 code of a NaN input, whatever its sign (measured: the hardware drops the sign)
#define AMP_NAN_E4M3 0x7Fu

// float16 pair -> two codes in the low 16 bits (low half -> byte 0), as cvt.rn.satfinite.e4m3x2.f16x2.
AMP_INLINE u32 amp_h2_to_e4m3x2(u32 h)
{
#if NR_NATIVE_FP8
    u16 codes;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(codes) : "r"(h));
    return codes;
#else
    const u32 mag = h & 0x7FFF7FFFu;
    // |x| >= 464 (and inf) saturates to 448, and 448 <= |x| < 464 rounds to 448: clamping |x| to 448 first is the same.
    // min.f16x2 returns the other operand for a NaN half; NaN halves are fixed up below.
    const u32 m = amp_h2_min(mag, 0x5F005F00u);
    // normal results (|x| >= 2^-6, f16 bits >= 0x2400): round the 7 dropped bits to even, then rebias the exponent by
    // -8 (code - 0x40, done as + 0xC0 mod 256 so no half borrows from the other; garbage, masked, below 0x2400)
    const u32 q = ((m + 0x003F003Fu + ((m >> 7) & 0x00010001u)) >> 7) & 0x01FF01FFu;
    const u32 normal = (q + 0x00C000C0u) & 0x00FF00FFu;
    // subnormal results: |x| + 2 in f16 rounds |x| to a multiple of 2^-9 (ulp of [2, 4)), ties to even
    const u32 sub = amp_h2_add(m, 0x40004000u) - 0x40004000u;        // 0..8 per half when |x| < 2^-6
    const u32 big = ((m + 0x5C005C00u) >> 15) & 0x00010001u;          // 1 where |x| >= 2^-6
    u32 code = (normal & (big * 0xFFFFu)) | (sub & ~(big * 0xFFFFu));
    const u32 nan = ((mag + 0x03FF03FFu) >> 15) & 0x00010001u;        // 1 where |x| > inf (mag >= 0x7C01)
    code |= nan * AMP_NAN_E4M3;
    code |= (h >> 8) & 0x00800080u & ~(nan * 0x80u);                  // sign (none on NaN)
    return amp_prmt(code, 0u, 0x4420u);                               // bytes 0 and 2 -> bytes 0 and 1
#endif
}
// ---- faster forms (BRIEF-11), sm_86 min/max.xorsign.abs -------------------------------------------------------------
// amp_h2_q8: the f16 pair of the e4m3 rounding of an f16 pair (= the finite decode of amp_h2_to_e4m3x2), 6 operations:
//   hc = h clamped to +-448 keeping its sign; cs = +-2^e of hc times 2^7, at least +-2; (hc + cs) - cs rounds |hc| to
//   e4m3's step for its binade (2^(e-3); 2^-9 below 2^-6), ties to even. The sum stays in cs's binade (|hc| < |cs| / 64)
//   and below 2^15 + 448. Identical for every finite value except that a negative value rounding to zero gives +0
//   (AMP_Q8_SIGN=1 keeps -0: one more operation); a NaN saturates to +-448 (the finite decode of the NaN code: 480).
#ifndef AMP_Q8_SIGN
#define AMP_Q8_SIGN 0
#endif
AMP_INLINE u32 amp_h2_q8_cs(u32 hc) { return amp_h2_xamax(amp_h2_mul(hc & 0xFC00FC00u, 0x58005800u), 0x40004000u); }
AMP_INLINE u32 amp_h2_q8(u32 h)
{
    const u32 hc = amp_h2_xamin(h, 0x5F005F00u);
    const u32 cs = amp_h2_q8_cs(hc);
    const u32 r = amp_h2_sub(amp_h2_add(hc, cs), cs);
#if AMP_Q8_SIGN
    return r | (h & 0x80008000u);
#else
    return r;
#endif
}
// the same for a non-negative pair below 448 (softmax weights): 5 operations, exact (no sign, no clamp needed)
AMP_INLINE u32 amp_h2_q8_pos(u32 h)
{
    const u32 cs = amp_h2_max(amp_h2_mul(h & 0x7C007C00u, 0x58005800u), 0x40004000u);
    return amp_h2_sub(amp_h2_add(h, cs), cs);
}
// amp_h2_to_e4m3x2_fast: two codes in bytes 0 and 2 (bits 8..15 / 24..31 are NOT clean: pack with amp_e4m3_pack),
// about 10 operations a pair. With sum = hc + cs as above, |sum| = 2^(E+7) + k 2^(E-3), E = max(e, -6), k = 0..16: its
// mantissa field is k and its exponent field E + 22, so the code magnitude 8 E + 48 + k = k + ((sum >> 7) & 0x7F).
// Identical to amp_h2_to_e4m3x2 for every non-NaN f16 value (signed zeros and infinities included); NaN -> +-0x7E.
AMP_INLINE u32 amp_h2_to_e4m3x2_fast(u32 h)
{
    const u32 hc = amp_h2_xamin(h, 0x5F005F00u);
    const u32 sum = amp_h2_add(hc, amp_h2_q8_cs(hc));
    const u32 mag = sum + ((sum >> 7) & 0x007F007Fu);
    return mag | ((h >> 8) & 0x00800080u);
}
// bytes 0 and 2 of a (codes of pair a) and of b -> one word: a.byte0, a.byte2, b.byte0, b.byte2
AMP_INLINE u32 amp_e4m3_pack(u32 a, u32 b) { return amp_prmt(a, b, 0x6420u); }
// two f16 pairs -> four codes (lo pair -> bytes 0, 1), the fast form of amp_h2x2_to_e4m3x4
AMP_INLINE u32 amp_h2x2_to_e4m3x4_fast(u32 lo, u32 hi)
{
#if NR_NATIVE_FP8
    return amp_join(u16(amp_h2_to_e4m3x2(lo)), u16(amp_h2_to_e4m3x2(hi)));
#else
    return amp_e4m3_pack(amp_h2_to_e4m3x2_fast(lo), amp_h2_to_e4m3x2_fast(hi));
#endif
}

// float32 -> one code (in the low 8 bits), rounded once, as cvt.rn.satfinite.e4m3x2.f32
AMP_INLINE u32 amp_f32_to_e4m3(float x)
{
    const u32 u = __float_as_uint(x);
    const u32 mag = u & 0x7FFFFFFFu;
    u32 code;
    if (mag > 0x7F800000u) {
        return AMP_NAN_E4M3;
    } else if (mag >= 0x3C800000u) {                                  // |x| >= 2^-6
        const u32 m = min(mag, 0x43E00000u);                          // clamp to 448 (inf included)
        const u32 r = m - 0x3C000000u;                                // exponent rebias 127 -> 7
        code = (r + 0x7FFFFu + ((r >> 20) & 1u)) >> 20;
    } else {                                                          // |x| < 2^-6: |x| + 2^14 rounds to 2^-9 steps
        code = __float_as_uint(amp_add_rn(__uint_as_float(mag), 16384.0f)) - 0x46800000u;
    }
    return code | ((u >> 24) & 0x80u);
}
// two floats -> two codes (lo -> byte 0), as  cvt.rn.satfinite.e4m3x2.f32 d, hi, lo
AMP_INLINE u32 amp_f32x2_to_e4m3x2(float lo, float hi) {
#if NR_NATIVE_FP8
    u16 codes;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(codes) : "f"(hi), "f"(lo));
    return codes;
#else
 return amp_f32_to_e4m3(lo) | (amp_f32_to_e4m3(hi) << 8);
#endif
}
// four floats -> four codes (a -> byte 0), each rounded once
AMP_INLINE u32 amp_f32x4_to_e4m3x4(float a, float b, float c, float d)
{
    return amp_f32x2_to_e4m3x2(a, b) | (amp_f32x2_to_e4m3x2(c, d) << 16);
}
// two float16 pairs -> four codes (lo pair -> bytes 0, 1)
AMP_INLINE u32 amp_h2x2_to_e4m3x4(u32 lo, u32 hi) { return amp_h2_to_e4m3x2(lo) | (amp_h2_to_e4m3x2(hi) << 16); }
// four floats -> float16 (round to nearest even) -> four codes: the double rounding of NVIDIA's stored results
// (cvt.rn.f16x2.f32 then cvt.rn.satfinite.e4m3x2.f16x2), as f32x4_to_e4m3x4 in swin4.cu
AMP_INLINE u32 amp_f32x4_to_e4m3x4_via_f16(float a, float b, float c, float d)
{
    return amp_h2x2_to_e4m3x4(amp_h2_pack(a, b), amp_h2_pack(c, d));
}

// ---- tensor-core products --------------------------------------------------------------------------------------
// D = A * B + C, m16n8k16, f16 operands, f16 accumulator (two packed pairs), full rate on GeForce Ampere
AMP_INLINE void amp_mma16_f16(u32* c, const u32* a, const u32* b)
{
    u32 d0, d1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
                 : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}
// the same with an f32 accumulator (c[0..3] = (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)); half rate on GeForce Ampere
AMP_INLINE void amp_mma16_f32(float* c, const u32* a, const u32* b)
{
    float d0, d1, d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    c[0] = d0; c[1] = d1; c[2] = d2; c[3] = d3;
}
// e4m3 A fragment of one k32 step (NVIDIA's layout, 4 words) -> the f16 A fragments of its two k16 steps
AMP_INLINE void amp_frag_a_e4m3(const u32* a8, u32* k0, u32* k1)
{
    amp_e4m3x4_to_h2x2_finite(a8[0], k0[0], k0[2]);
    amp_e4m3x4_to_h2x2_finite(a8[1], k0[1], k0[3]);
    amp_e4m3x4_to_h2x2_finite(a8[2], k1[0], k1[2]);
    amp_e4m3x4_to_h2x2_finite(a8[3], k1[1], k1[3]);
}
// e4m3 B fragment of one k32 step (2 words) -> the f16 B fragments of its two k16 steps
AMP_INLINE void amp_frag_b_e4m3(const u32* b8, u32* k0, u32* k1)
{
    amp_e4m3x4_to_h2x2_finite(b8[0], k0[0], k0[1]);
    amp_e4m3x4_to_h2x2_finite(b8[1], k1[0], k1[1]);
}
// NVIDIA's  mma.m16n8k32.row.col.f16.e4m3.e4m3.f16  as two f16-accumulate k16 steps on already-decoded fragments
AMP_INLINE void amp_mma32_f16(u32* c, const u32* a_k0, const u32* a_k1, const u32* b_k0, const u32* b_k1)
{
    amp_mma16_f16(c, a_k0, b_k0);
    amp_mma16_f16(c, a_k1, b_k1);
}
// ... and straight from the e4m3 fragments (NVIDIA's registers, unchanged)
AMP_INLINE void amp_mma_e4m3_k32(u32* c, const u32* a8, const u32* b8)
{
    u32 a0[4], a1[4], b0[2], b1[2];
    amp_frag_a_e4m3(a8, a0, a1);
    amp_frag_b_e4m3(b8, b0, b1);
    amp_mma32_f16(c, a0, a1, b0, b1);
}

// NOT the Phase A target (that is the f16-accumulate pair above); kept for comparison. The same k32 step with an f32
// accumulator: C (f16 pairs) widened exactly, two f32-accumulate k16 steps, one rounding to f16 at the end. Half the
// tensor rate on GeForce Ampere, twice the accumulator registers; on the RTX 5090 it matches NVIDIA's k32 instruction
// far more often than the f16 pair (amp_common_test.py prints both).
AMP_INLINE void amp_mma_e4m3_k32_f32acc(u32* c, const u32* a8, const u32* b8)
{
    u32 a0[4], a1[4], b0[2], b1[2];
    amp_frag_a_e4m3(a8, a0, a1);
    amp_frag_b_e4m3(b8, b0, b1);
    float f[4];
    amp_h2_unpack(c[0], f[0], f[1]);
    amp_h2_unpack(c[1], f[2], f[3]);
    amp_mma16_f32(f, a0, b0);
    amp_mma16_f32(f, a1, b1);
    c[0] = amp_h2_pack(f[0], f[1]);
    c[1] = amp_h2_pack(f[2], f[3]);
}

// ---- shared memory staging (sm_80) -----------------------------------------------------------------------------
AMP_INLINE u32 amp_smem(const void* p) { return (u32)__cvta_generic_to_shared(p); }
// 16 bytes global -> shared, asynchronous, bypassing L1 (both addresses 16-byte aligned)
AMP_INLINE void amp_cp_async16(u32 smem_addr, const void* global)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"((u64)global) : "memory");
}
// the same, but only `bytes` (0..16) are read and the rest of the 16 are zero-filled
AMP_INLINE void amp_cp_async16_zfill(u32 smem_addr, const void* global, u32 bytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_addr), "l"((u64)global), "r"(bytes) : "memory");
}
AMP_INLINE void amp_cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> AMP_INLINE void amp_cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
AMP_INLINE void amp_cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
// four 8x8 b16 matrices from shared memory (lanes 8i..8i+7 give the row addresses of matrix i)
AMP_INLINE void amp_ldmatrix_x4(u32* r, u32 smem_addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr) : "memory");
}
AMP_INLINE void amp_ldmatrix_x4_trans(u32* r, u32 smem_addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr) : "memory");
}
AMP_INLINE void amp_ldmatrix_x2(u32* r, u32 smem_addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_addr) : "memory");
}
AMP_INLINE void amp_ldmatrix_x2_trans(u32* r, u32 smem_addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_addr) : "memory");
}

// ---- flags between kernels of one evaluation (NVIDIA's protocol; ld.relaxed / st.release / nanosleep: sm_70) -----
// A wait never gives up silently (no silent fallbacks): it polls like NVIDIA's unbounded loop, and if one wait lasts
// longer than AMP_WAIT_LIMIT_NS of wall time (%globaltimer) the kernel executes `trap`. The trap is a visible GPU fault
// (the app reports the device as removed) instead of a frame computed from unready data. The limit is far above any
// real producer and below Windows' 2 s reset. The `budget` arguments are kept only so the call sites stay unchanged.
#ifndef AMP_WAIT_BUDGET_NS
#define AMP_WAIT_BUDGET_NS (1 << 23)
#endif
#ifndef AMP_WAIT_LIMIT_NS
#define AMP_WAIT_LIMIT_NS 500000000ull
#endif
AMP_INLINE int amp_flag_load(const int* flag)
{
    int value;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"((u64)flag) : "memory");
    return value;
}
AMP_INLINE int amp_flag_acquire(const int* flag)
{
    int value;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"((u64)flag) : "memory");
    return value;
}
AMP_INLINE void amp_flag_publish(int* flag, int value) { asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"((u64)flag), "r"(value) : "memory"); }
AMP_INLINE void amp_nap() { asm volatile("nanosleep.u32 64;" ::: "memory"); }
AMP_INLINE void amp_fence_acq_rel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
AMP_INLINE u64 amp_now_ns()
{
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}
AMP_INLINE void amp_wait_failed() { asm volatile("trap;" ::: "memory"); }
// wait while *flag < 0 (the runtime clears flags to -1); a wait longer than AMP_WAIT_LIMIT_NS traps
AMP_INLINE void amp_wait_not_negative(const int* flag, int& budget)
{
    (void)budget;
    if (amp_flag_load(flag) >= 0) return;
    const u64 start = amp_now_ns();
    do {
        amp_nap();
        if (amp_now_ns() - start > AMP_WAIT_LIMIT_NS) amp_wait_failed();
    } while (amp_flag_load(flag) < 0);
}
// wait while *flag < target; a wait longer than AMP_WAIT_LIMIT_NS traps
AMP_INLINE void amp_wait_at_least(const int* flag, int target, int& budget)
{
    (void)budget;
    if (amp_flag_load(flag) >= target) return;
    const u64 start = amp_now_ns();
    do {
        amp_nap();
        if (amp_now_ns() - start > AMP_WAIT_LIMIT_NS) amp_wait_failed();
    } while (amp_flag_load(flag) < target);
}
// loads of data another kernel of this evaluation wrote: not from this SM's L1
AMP_INLINE amp_v4 amp_load_cg(const void* p)
{
    amp_v4 v;
    asm volatile("ld.weak.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"((u64)p) : "memory");
    return v;
}
AMP_INLINE u32 amp_load_cg32(const void* p)
{
    u32 v;
    asm volatile("ld.weak.global.cg.u32 %0, [%1];" : "=r"(v) : "l"((u64)p) : "memory");
    return v;
}
AMP_INLINE amp_v4 amp_load_nc(const void* p)
{
    amp_v4 v;
    asm("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"((u64)p));
    return v;
}
// split-K merge in f16 (scalar form; the .v2/.v4 vector forms are sm_90)
AMP_INLINE void amp_red_add_h2(u32* p, u32 h2) { asm volatile("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;" :: "l"((u64)p), "r"(h2) : "memory"); }
