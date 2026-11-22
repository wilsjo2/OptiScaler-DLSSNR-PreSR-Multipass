// Test kernels for amp_common.cuh (driven by v2/amp_common_test.py through NrKernelLab).
//
// Default build (what amp_legal.py compiles for sm_86 and sm_120): only our software paths; the "hardware" outputs
// are zero. With -DAMP_HW_REF=1 (sm_120 only, never shipped) the kernels also run the RTX 50 instructions our code
// replaces (cvt.rn.satfinite.e4m3x2.{f16x2,f32}, cvt.rn.f16x2.e4m3x2, mma.m16n8k32 e4m3 f16-accumulate) so the
// test can compare bit for bit.
// amp-variant: base
#include "amp_common.cuh"

#ifndef AMP_HW_REF
#define AMP_HW_REF 0
#endif

// in: float16 pairs; out[2i] = ours (two codes, low half -> byte 0), out[2i+1] = hardware
struct CvtHParams { const u32* in; u32* out; int count; int pad; };
extern "C" __global__ void amp_t_cvt_h(CvtHParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count) return;
    const u32 h = p.in[i];
    u32 hw = 0;
#if AMP_HW_REF
    u16 d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(h));
    hw = d;
#endif
    p.out[2 * i] = amp_h2_to_e4m3x2(h);
    p.out[2 * i + 1] = hw;
}

// in: float pairs (lo, hi); out[4i] = ours single rounding, [4i+1] = hardware .f32 form (d, hi, lo),
// [4i+2] = ours through float16 (cvt.rn.f16x2.f32 then e4m3), [4i+3] = hardware through float16
struct CvtFParams { const float* in; u32* out; int count; int pad; };
extern "C" __global__ void amp_t_cvt_f(CvtFParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count) return;
    const float lo = p.in[2 * i], hi = p.in[2 * i + 1];
    u32 hw = 0, hw16 = 0;
#if AMP_HW_REF
    u16 d, e;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
    asm volatile("{\n .reg .b32 t;\n cvt.rn.f16x2.f32 t, %1, %2;\n cvt.rn.satfinite.e4m3x2.f16x2 %0, t;\n}" : "=h"(e) : "f"(hi), "f"(lo));
    hw = d;
    hw16 = e;
#endif
    p.out[4 * i] = amp_f32x2_to_e4m3x2(lo, hi);
    p.out[4 * i + 1] = hw;
    p.out[4 * i + 2] = amp_h2_to_e4m3x2(amp_h2_pack(lo, hi));
    p.out[4 * i + 3] = hw16;
}

// in: code pairs (low 16 bits); out[4i] = ours exact, [4i+1] = ours _finite, [4i+2] = hardware, [4i+3] = ours x4 form
struct DecParams { const u32* in; u32* out; int count; int pad; };
extern "C" __global__ void amp_t_dec(DecParams p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count) return;
    const u32 pair = p.in[i] & 0xFFFFu;
    u32 hw = 0;
#if AMP_HW_REF
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hw) : "h"((u16)pair));
#endif
    u32 lo, hi;
    amp_e4m3x4_to_h2x2(pair | (pair << 16), lo, hi);
    p.out[4 * i] = amp_e4m3x2_to_h2(pair);
    p.out[4 * i + 1] = amp_e4m3x2_to_h2_finite(pair);
    p.out[4 * i + 2] = hw;
    p.out[4 * i + 3] = lo == hi ? lo : 0xDEADBEEFu;
}

// One warp per experiment: A (k32 e4m3 fragment, 4 words per lane), B (2 words), C (2 words, f16 pairs).
// out per lane (8 words): [0,1] ours, K 0..15 first; [2,3] ours, K 16..31 first; [4,5] hardware k32;
// [6,7] ours with the NaN-exact decode (must equal [0,1] for finite codes); [8,9] the f32-accumulate alternative
struct MmaParams { const u32* a; const u32* b; const u32* c; u32* out; int count; int pad; };
extern "C" __global__ void amp_t_mma(MmaParams p)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= p.count) return;
    const u32* a8 = p.a + (warp * 32 + lane) * 4;
    const u32* b8 = p.b + (warp * 32 + lane) * 2;
    const u32* c0 = p.c + (warp * 32 + lane) * 2;
    u32 A[4] = {a8[0], a8[1], a8[2], a8[3]}, B[2] = {b8[0], b8[1]};
    u32 d[2] = {c0[0], c0[1]};
    amp_mma_e4m3_k32(d, A, B);
    u32 ak0[4], ak1[4], bk0[2], bk1[2];
    amp_frag_a_e4m3(A, ak0, ak1);
    amp_frag_b_e4m3(B, bk0, bk1);
    u32 r[2] = {c0[0], c0[1]};
    amp_mma32_f16(r, ak1, ak0, bk1, bk0);
    // exact-decode path
    u32 ek0[4], ek1[4], fk0[2], fk1[2];
    amp_e4m3x4_to_h2x2(A[0], ek0[0], ek0[2]); amp_e4m3x4_to_h2x2(A[1], ek0[1], ek0[3]);
    amp_e4m3x4_to_h2x2(A[2], ek1[0], ek1[2]); amp_e4m3x4_to_h2x2(A[3], ek1[1], ek1[3]);
    amp_e4m3x4_to_h2x2(B[0], fk0[0], fk0[1]); amp_e4m3x4_to_h2x2(B[1], fk1[0], fk1[1]);
    u32 e[2] = {c0[0], c0[1]};
    amp_mma32_f16(e, ek0, ek1, fk0, fk1);
    u32 hw[2] = {0, 0};
#if AMP_HW_REF
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
                 : "=r"(hw[0]), "=r"(hw[1]) : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                   "r"(c0[0]), "r"(c0[1]));
#endif
    u32 g[2] = {c0[0], c0[1]};
    amp_mma_e4m3_k32_f32acc(g, A, B);
    u32* o = p.out + (warp * 32 + lane) * 10;
    o[0] = d[0]; o[1] = d[1]; o[2] = r[0]; o[3] = r[1]; o[4] = hw[0]; o[5] = hw[1]; o[6] = e[0]; o[7] = e[1];
    o[8] = g[0]; o[9] = g[1];
}

// Smoke test of the remaining helpers (one CTA of 32 threads): cp.async + ldmatrix (plain and transposed), the f32
// accumulate mma, flags, the f16x2 reduction, movmatrix and the packed-f16 helpers. out: 32 lanes x 16 words.
struct MiscParams { const u32* in; u32* out; int* flags; u32* red; };
extern "C" __global__ void amp_t_misc(MiscParams p)
{
    __shared__ __align__(16) u32 tile[256];                  // 32 rows of 16 bytes = 16x16 b16 = four 8x8 matrices
    const int lane = threadIdx.x;
    for (int j = 0; j < 2; ++j) amp_cp_async16(amp_smem(tile + (lane * 2 + j) * 4), p.in + (lane * 2 + j) * 4);
    amp_cp_async16_zfill(amp_smem(tile + 128 + lane * 4), p.in + 128 + lane * 4, 16);
    amp_cp_async_commit();
    amp_cp_async_wait<0>();
    amp_cp_async_wait_all();
    __syncwarp();
    u32* o = p.out + lane * 16;
    u32 r[4], t[4], s[2], u[2];
    // matrix i = rows 8i..8i+7 of the first 16 rows (8 b16 = 16 bytes each)
    amp_ldmatrix_x4(r, amp_smem(tile + lane * 4));
    amp_ldmatrix_x4_trans(t, amp_smem(tile + lane * 4));
    amp_ldmatrix_x2(s, amp_smem(tile + (lane & 15) * 4));
    amp_ldmatrix_x2_trans(u, amp_smem(tile + (lane & 15) * 4));
    for (int j = 0; j < 4; ++j) { o[j] = r[j]; o[4 + j] = t[j]; }
    o[8] = s[0]; o[9] = s[1]; o[10] = u[0]; o[11] = u[1];
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const u32 bb[2] = {r[0], r[1]};
    amp_mma16_f32(acc, r, bb);
    o[12] = __float_as_uint(acc[0]) ^ __float_as_uint(acc[3]);
    o[13] = amp_transpose8(r[0]);
    int budget = AMP_WAIT_BUDGET_NS;
    if (lane == 0) amp_flag_publish(p.flags, 7);
    amp_wait_not_negative(p.flags, budget);
    amp_wait_at_least(p.flags + 1, 0, budget);
    amp_fence_acq_rel();
    const amp_v4 v = amp_load_cg(p.in + lane * 4);
    const amp_v4 n = amp_load_nc(p.in + lane * 4);
    o[14] = (u32)amp_flag_acquire(p.flags) + amp_load_cg32(p.in + lane) + (v.x ^ n.y);
    amp_red_add_h2(p.red, 0x3C003C00u);                      // +1.0 in both halves, 32 times
    float lo, hi;
    amp_h2_unpack(amp_h2_fma(amp_h2_mul(r[0], r[1]), amp_h2_sub(amp_h2_add(r[2], r[3]), r[1]), amp_h2_abs(r[0])), lo, hi);
    const u32 m = amp_h2_xamax(amp_h2_max(r[0], r[1]), amp_h2_xamin(amp_h2_min(r[2], r[3]), r[0]));
    o[15] = amp_h2_pack(amp_rsqrt_approx(fabsf(lo) + 1.f) + amp_rcp_approx(fabsf(hi) + 1.f) + amp_ex2_approx(-1.f),
                        amp_fma_rn(amp_mul_rn(lo, hi), amp_add_rn(lo, 1.f), amp_xamaxf(lo, hi)))
            ^ m ^ (u32)amp_e4m3_to_h(lane) ^ amp_join(1, 2) ^ amp_f32x4_to_e4m3x4(lo, hi, 1.f, 2.f)
            ^ amp_f32x4_to_e4m3x4_via_f16(lo, hi, 3.f, 4.f) ^ amp_h2x2_to_e4m3x4(r[0], r[1]) ^ (u32)amp_h_to_f32(1);
    float f4[4];
    amp_e4m3x4_to_f32(r[3], f4);
    u16 a16, b16;
    amp_split(amp_e4m3x2_to_h2_scaled(r[2]), a16, b16);
    o[15] ^= __float_as_uint(f4[0] + f4[1] + f4[2] + f4[3]) ^ a16 ^ b16;
}
