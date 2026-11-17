// NVFP4 (4-bit) replacement of the DLSS-NR 512-wide split Swin kernels (cc_split_swin_16h_*_512_*_fp8) and of the
// decoder-input projection (cc_dec_input_upsample_1024_512_*_fp8), Model A, runtime 310.9.16.
// Target sm_120a, built with NVRTC (--fmad=false) through lab.compile_kernel; one cubin per kind:
//   -DKIND=1  FFWD   grouped feed-forward             grid (ceil(W/8), ceil(H/8), 2), block (32, 8); -DVIEW_IN=1: block (32, 4),
//                                                      view-layout input (ffwd_inpview)
//   -DKIND=2  PROJ   1x1 projection + gained residual  grid (2 ceil(W/8), ceil(H/8)), block (32, 4) or (32, 8);
//                                                      -DVIEW_RES=1 residual in view layout (ffwd_proj_inpview),
//                                                      -DVIEW_OUT=1 output in view layout (proj_outview)
//   -DKIND=3  POOL   PROJ plus the 2x2 mean, pooled map in tile layout (proj_pool)
//   -DKIND=4  HEAD   512 -> 1024 projection of the pooled map, no residual (final_head), grid (4 ceil(W/8), ceil(H/8))
//   -DKIND=5  DEC    1024 -> 512 projection of the coarse map, 2x replication, + gain * skip (dec_input_upsample),
//                    grid (2 ceil(Wc/4), ceil(Hc/4), 4), block (32, 2)
//   -DKIND=6  QKV    window attention (qkv), grid (ceil((W - min(sx,0))/8), ceil((H - min(sy,0))/8), 4), block (32, 4)
// Launch each with the runtime's own grid and block (the 4K trace), no dynamic shared memory. Buffers, flags and the
// handoff layouts are the runtime's (split-spec/abi.md); the parameter block is ours (Params below, abi.md section O),
// every field taken from the runtime's block plus one pointer to our weight record (sim/export_split4.py).
//
// Every product with a weight matrix is NVFP4 on both operands (mma m16n8k64 kind::mxf4nvf4, block scale ue4m3 per 16
// inputs, float32 accumulator): weights packed offline, activations packed here per token at the tensor scale of the
// record (4). The attention's Q.K and P.V are the runtime's FP8 arithmetic (copied from swin4.cu). The bit-level
// model is sim/split4_model.py.
//
// Lane (g = lane >> 2, t = lane & 3) of a 16-token tile: tokens g and g + 8, of every 32 channels the eight
// 8 j + 2 t + e, exactly the lane's 16 bytes of each 512-byte channel block of the tile layout; channel
// 32 h + 8 j + 2 t + e sits at k64 step h >> 1, K = 32 (h & 1) + 8 t + 2 j + e of the 4-bit instruction, so packing,
// accumulators and stores stay in the lane (no shuffles except the block maxima and scale bytes).
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };

#define NR_INLINE static __forceinline__ __device__

#define KIND_FFWD 1
#define KIND_PROJ 2
#define KIND_POOL 3
#define KIND_HEAD 4
#define KIND_DEC 5
#define KIND_QKV 6
#ifndef KIND
#define KIND KIND_PROJ
#endif
#ifndef VIEW_IN
#define VIEW_IN 0
#endif
#ifndef VIEW_RES
#define VIEW_RES 0
#endif
#ifndef VIEW_OUT
#define VIEW_OUT 0
#endif
#ifndef PARTS
#define PARTS 4             // PROJ / POOL: channel parts per producer window (qkv 4; ffwd 2 for ffwd_proj)
#endif
#define SLAB 1152u                                // 32 outputs x one k64 step: 2 x 512 B of nibbles, 128 B of scales
#define UNIT 576u                                   // one packed A operand (16 tokens x 64 inputs): 512 B nibbles, 64 B scales
// Every wait for another kernel's flags draws on one budget of nominal sleep time (as swin4.cu): a missing producer
// costs a wrong frame, never a hung GPU.
#ifndef WAIT_BUDGET_NS
#define WAIT_BUDGET_NS (1 << 23)
#endif

#define OPT_ATT4 1u         // QKV: Q.K and P.V as NVFP4 products too (sim/split4_model.py --att4)

struct Params {
    const u8* in;           // input map (tile layout; FFWD VIEW_IN: view layout; DEC: coarse map, 1024 channels)
    const u8* res;          // residual (PROJ, POOL; VIEW_RES: view layout) or skip (DEC)
    u8* out;                // output map (tile layout; VIEW_OUT: view layout; HEAD: 1024 channels)
    u8* out2;               // POOL: pooled map, tile layout of d0 x d1 tokens
    const u8* w;            // our weight record
    const int* pflags;      // producer's flags or null (no wait)
    int* oflags;            // own flags or null (no publish)
    int* counter;           // DEC: completion counter per (coarse tile, half), -1 before the launch
    int H, W;               // token rows and columns of the map this kernel works on (HEAD: the pooled map)
    int sx, sy;             // QKV: own window shift (0 or -4)
    int psx, psy;           // the producer's window shift
    int d0, d1;             // POOL: pooled rows, columns; DEC: coarse rows, columns
    u32 options;
    u32 reserved;
};

// record: 16 floats, 16 offsets
enum { F_T1, F_S1, F_T2, F_S2, F_T3, F_S3 };
enum { O_M1, O_M2, O_M3, O_GAIN, O_BIAS, O_HEADSCALE };

// ---- copied from swin4.cu ---------------------------------------------------------------------------------------
NR_INLINE int flag_load(const int* flag)
{
    int value;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"((u64)flag) : "memory");
    return value;
}
NR_INLINE void flag_publish(int* flag, int value) { asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"((u64)flag), "r"(value) : "memory"); }
NR_INLINE void nap() { asm volatile("nanosleep.u32 64;" ::: "memory"); }
NR_INLINE void wait_not_negative(const int* flag, int& budget)
{
    for (; budget > 0 && flag_load(flag) < 0; budget -= 64) nap();
}
NR_INLINE V4 load_cg(const u8* p)
{
    V4 v;
    asm volatile("ld.weak.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"((u64)p) : "memory");
    return v;
}
NR_INLINE u32 load_cg32(const u8* p)
{
    u32 v;
    asm volatile("ld.weak.global.cg.u32 %0, [%1];" : "=r"(v) : "l"((u64)p) : "memory");
    return v;
}
NR_INLINE void mma4(float* c, const V4& a, u32 b0, u32 b1, u32 sfa, u32 sfb)
{
    float d0, d1, d2, d3;
    const u16 z = 0;
    asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%16, %16}, {%15}, {%16, %16};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "r"(sfb), "h"(z));
    c[0] = d0; c[1] = d1; c[2] = d2; c[3] = d3;
}
NR_INLINE void mma8(u32* c, const u32* a, u32 b0, u32 b1)
{
    u32 d0, d1;
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
        : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}
NR_INLINE u32 h2_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
NR_INLINE u32 h2_mul(u32 a, u32 b) { u32 d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_add(u32 a, u32 b) { u32 d; asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_min(u32 a, u32 b) { u32 d; asm("min.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_abs(u32 a) { u32 d; asm("abs.f16x2 %0, %1;" : "=r"(d) : "r"(a)); return d; }
NR_INLINE u32 h2_pack(float lo, float hi) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
NR_INLINE void h2_unpack(u32 h, float& lo, float& hi)
{
    asm("{\n .reg .b16 nr_l, nr_u;\n mov.b32 {nr_l, nr_u}, %2;\n cvt.f32.f16 %0, nr_l;\n cvt.f32.f16 %1, nr_u;\n}" : "=f"(lo), "=f"(hi) : "r"(h));
}
NR_INLINE float f16_round(float x)
{
    float d;
    asm("{\n .reg .b16 nr_h;\n cvt.rn.f16.f32 nr_h, %1;\n cvt.f32.f16 %0, nr_h;\n}" : "=f"(d) : "f"(x));
    return d;
}
NR_INLINE float rsqrt_approx(float x) { float d; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE float rcp_approx(float x) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE void e4m3x4_to_f32(u32 word, float* v)
{
    float a, b, c, d;
    asm("{\n .reg .b16 nr_l, nr_u, nr_a, nr_b;\n .reg .b32 nr_p, nr_q;\n mov.b32 {nr_l, nr_u}, %4;\n"
        " cvt.rn.f16x2.e4m3x2 nr_p, nr_l;\n cvt.rn.f16x2.e4m3x2 nr_q, nr_u;\n"
        " mov.b32 {nr_a, nr_b}, nr_p;\n cvt.f32.f16 %0, nr_a;\n cvt.f32.f16 %1, nr_b;\n"
        " mov.b32 {nr_a, nr_b}, nr_q;\n cvt.f32.f16 %2, nr_a;\n cvt.f32.f16 %3, nr_b;\n}"
        : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "r"(word));
    v[0] = a; v[1] = b; v[2] = c; v[3] = d;
}
NR_INLINE u32 f32x4_to_e4m3x4(float a, float b, float c, float d)
{
    u32 word;
    asm("{\n .reg .b16 nr_l, nr_u;\n .reg .b32 nr_p, nr_q;\n cvt.rn.f16x2.f32 nr_p, %2, %1;\n cvt.rn.f16x2.f32 nr_q, %4, %3;\n"
        " cvt.rn.satfinite.e4m3x2.f16x2 nr_l, nr_p;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_u, nr_q;\n mov.b32 %0, {nr_l, nr_u};\n}"
        : "=r"(word) : "f"(a), "f"(b), "f"(c), "f"(d));
    return word;
}
NR_INLINE u32 h2_to_e4m3x2(u32 h) { u16 d; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(h)); return d; }
NR_INLINE u32 m_xamax(u32 a, u32 b) { u32 d; asm("max.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 m_prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
NR_INLINE void m_split(u32 w, u16& lo, u16& hi) { asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(w)); }
NR_INLINE void m_e4m3x4_to_h2(u32 word, u32& a, u32& b)
{
    asm("{\n .reg .b16 nr_l, nr_u;\n mov.b32 {nr_l, nr_u}, %2;\n cvt.rn.f16x2.e4m3x2 %0, nr_l;\n cvt.rn.f16x2.e4m3x2 %1, nr_u;\n}"
        : "=r"(a), "=r"(b) : "r"(word));
}
NR_INLINE u32 m_h2x2_to_e4m3x4(u32 a, u32 b)
{
    u32 word;
    asm("{\n .reg .b16 nr_l, nr_u;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_l, %1;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_u, %2;\n mov.b32 %0, {nr_l, nr_u};\n}"
        : "=r"(word) : "r"(a), "r"(b));
    return word;
}
NR_INLINE u32 m_e2m1x8_h(const u32* h, u16 r)
{
    u32 word;
    asm("{\n .reg .b16 nr_a<8>;\n .reg .f32 nr_f<8>;\n .reg .b8 nr_n<4>;\n"
        " mov.b32 {nr_a0, nr_a1}, %1;\n mov.b32 {nr_a2, nr_a3}, %2;\n mov.b32 {nr_a4, nr_a5}, %3;\n mov.b32 {nr_a6, nr_a7}, %4;\n"
        " fma.rn.f32.f16 nr_f0, nr_a0, %5, 0f80000000;\n fma.rn.f32.f16 nr_f1, nr_a1, %5, 0f80000000;\n"
        " fma.rn.f32.f16 nr_f2, nr_a2, %5, 0f80000000;\n fma.rn.f32.f16 nr_f3, nr_a3, %5, 0f80000000;\n"
        " fma.rn.f32.f16 nr_f4, nr_a4, %5, 0f80000000;\n fma.rn.f32.f16 nr_f5, nr_a5, %5, 0f80000000;\n"
        " fma.rn.f32.f16 nr_f6, nr_a6, %5, 0f80000000;\n fma.rn.f32.f16 nr_f7, nr_a7, %5, 0f80000000;\n"
        " cvt.rn.satfinite.e2m1x2.f32 nr_n0, nr_f1, nr_f0;\n cvt.rn.satfinite.e2m1x2.f32 nr_n1, nr_f3, nr_f2;\n"
        " cvt.rn.satfinite.e2m1x2.f32 nr_n2, nr_f5, nr_f4;\n cvt.rn.satfinite.e2m1x2.f32 nr_n3, nr_f7, nr_f6;\n"
        " mov.b32 %0, {nr_n0, nr_n1, nr_n2, nr_n3};\n}"
        : "=r"(word) : "r"(h[0]), "r"(h[1]), "r"(h[2]), "r"(h[3]), "h"(r));
    return word;
}
NR_INLINE u32 m_scale2(float a0, float a1, float& s0, float& s1, float& r0, float& r1)
{
    const float raw0 = fmaxf(a0 * 0.16666667163372040f, 0.001953125f), raw1 = fmaxf(a1 * 0.16666667163372040f, 0.001953125f);
    u16 c16;
    u32 sv;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(c16) : "f"(raw0), "f"(raw1));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sv) : "h"(c16));
    h2_unpack(sv, s0, s1);
    r0 = rcp_approx(s0);
    r1 = rcp_approx(s1);
    return (u32)c16;
}
NR_INLINE void m_amax2_h(const u32* h0, const u32* h1, float& a0, float& a1)
{
    const u32 m0 = m_xamax(m_xamax(h0[0], h0[1]), m_xamax(h0[2], h0[3]));
    const u32 m1 = m_xamax(m_xamax(h1[0], h1[1]), m_xamax(h1[2], h1[3]));
    u32 mm = m_xamax(m_prmt(m0, m1, 0x5410u), m_prmt(m0, m1, 0x7632u));
    mm = m_xamax(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    h2_unpack(h2_abs(mm), a0, a1);
}
// Two rows (tokens g, g + 8) of float16 values -> two words of nibbles; returns the scale code pair (byte 0: row g).
NR_INLINE u32 m_pack2_h(const u32* h0, const u32* h1, float pre, u32& w0, u32& w1)
{
    float a0, a1, s0, s1, r0, r1;
    m_amax2_h(h0, h1, a0, a1);
    const u32 codes = m_scale2(a0 * pre, a1 * pre, s0, s1, r0, r1);
    u16 q0, q1;
    m_split(h2_pack(r0 * pre, r1 * pre), q0, q1);
    w0 = m_e2m1x8_h(h0, q0);
    w1 = m_e2m1x8_h(h1, q1);
    return codes;
}
NR_INLINE void m_put_scale(u8* words, u32 g, u32 t, u32 khalf, u32 pair)
{
    words[g * 8u + (t & 1u) * 4u + khalf * 2u + (t >> 1)] = (u8)(pair >> ((t & 1u) * 8u));
}
NR_INLINE u32 att_e4m3x4(float a, float b, float c, float d)
{
    u32 word;
    asm("{\n .reg .b16 nr_l, nr_u;\n cvt.rn.satfinite.e4m3x2.f32 nr_l, %2, %1;\n cvt.rn.satfinite.e4m3x2.f32 nr_u, %4, %3;\n"
        " mov.b32 %0, {nr_l, nr_u};\n}"
        : "=r"(word) : "f"(a), "f"(b), "f"(c), "f"(d));
    return word;
}
NR_INLINE u32 att_swap_mid_bytes(u32 a) { u32 d; asm("prmt.b32 %0, %1, 0, 0x3120;" : "=r"(d) : "r"(a)); return d; }
NR_INLINE u32 att2_transpose8(u32 h)
{
    u32 d;
    asm("movmatrix.sync.trans.aligned.m8n8.b16 %0, %1;" : "=r"(d) : "r"(h));
    return d;
}
NR_INLINE void att2_qk_rows(const float (&a)[4][4], float sq2, float scale, bool odd, u32& lo0, u32& hi0, u32& lo1, u32& hi1)
{
    float sum[2];
#pragma unroll
    for (u32 rh = 0; rh < 2; ++rh) {
        float x = a[0][rh * 2u] * a[0][rh * 2u];
        x = fmaf(a[0][rh * 2u + 1u], a[0][rh * 2u + 1u], x);
#pragma unroll
        for (u32 j = 1; j < 4; ++j)
#pragma unroll
            for (u32 e = 0; e < 2; ++e) x = fmaf(a[j][rh * 2u + e], a[j][rh * 2u + e], x);
        sum[rh] = x;
    }
    float x = (odd ? sum[1] : sum[0]) + __shfl_xor_sync(0xffffffffu, odd ? sum[0] : sum[1], 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    x *= sq2;
    const float r = rsqrt_approx(fmaxf(x, 6.19888305664e-05f));
    const float mine = (x >= 65520.0f ? 0.0f : r) * scale;
    const float other = __shfl_xor_sync(0xffffffffu, mine, 1);
    const float f0 = odd ? other : mine, f1 = odd ? mine : other;
    lo0 = att_e4m3x4(a[0][0] * f0, a[0][1] * f0, a[1][0] * f0, a[1][1] * f0);
    hi0 = att_e4m3x4(a[2][0] * f0, a[2][1] * f0, a[3][0] * f0, a[3][1] * f0);
    lo1 = att_e4m3x4(a[0][2] * f1, a[0][3] * f1, a[1][2] * f1, a[1][3] * f1);
    hi1 = att_e4m3x4(a[2][2] * f1, a[2][3] * f1, a[3][2] * f1, a[3][3] * f1);
}
NR_INLINE void att2_v_rows(const float (&a)[4][4], float sqkv, u32& v0, u32& v1, u32& v2, u32& v3)
{
    u32 v[4];
#pragma unroll
    for (u32 j = 0; j < 4; ++j)
        v[j] = att_swap_mid_bytes(att2_transpose8(att_e4m3x4(a[j][0] * sqkv, a[j][2] * sqkv, a[j][1] * sqkv, a[j][3] * sqkv)));
    v0 = v[0]; v1 = v[1]; v2 = v[2]; v3 = v[3];
}

// ---- split4 helpers ---------------------------------------------------------------------------------------------
// Wait (one thread) for every part of every producer window under the region [y0, y0 + rows) x [x0, x0 + cols) of the
// map; producer windows of 2^wl tokens on a grid shifted by (psx, psy), index [(window row * parts + part) * gx + col].
// Coarser than the runtime's own pipelined waits (it waits for all parts before starting): safe.
NR_INLINE void wait_region(const int* f, int parts, int wl, int H, int W, int psx, int psy, int y0, int x0, int rows, int cols)
{
    const int win = 1 << wl;
    const int gx = (W - min(psx, 0) + win - 1) >> wl, gy = (H - min(psy, 0) + win - 1) >> wl;
    const int dy = y0 - psy, dx = x0 - psx;
    const int r0 = max(dy, 0) >> wl, r1 = min(dy + rows > 0 ? ((dy + rows - 1) >> wl) + 1 : 0, gy);
    const int c0 = max(dx, 0) >> wl, c1 = min(dx + cols > 0 ? ((dx + cols - 1) >> wl) + 1 : 0, gx);
    int budget = WAIT_BUDGET_NS;
    for (int r = r0; r < r1; ++r)
        for (int q = 0; q < parts; ++q)
            for (int c = c0; c < c1; ++c) wait_not_negative(f + (r * parts + q) * gx + c, budget);
}
// The lane's 16 bytes of channel block b of a tile from a view-layout map [C/16][rows][cols][16] (as swin4.cu VARIANT 1)
NR_INLINE V4 view_load(const u8* map, int rows, int cols, int ty, int tx, u32 b, u32 g, u32 t)
{
    u32 words[4];
#pragma unroll
    for (u32 rh = 0; rh < 2; ++rh) {
        const u32 m = g + 8u * rh;
        const int py = ty * 4 + (int)(m >> 2), px = tx * 4 + (int)(m & 3u);
        const bool in = py >= 0 && py < rows && px >= 0 && px < cols;
#pragma unroll
        for (u32 a = 0; a < 2; ++a) {
            const u64 at = ((u64)(u32)((int)(2u * b + a) * rows + (in ? py : 0)) * (u32)cols + (u32)(in ? px : 0)) * 16u + t * 4u;
            words[2u * a + rh] = in ? load_cg32(map + at) : 0u;
        }
    }
    return V4{words[0], words[1], words[2], words[3]};
}
NR_INLINE void view_store(u8* map, int rows, int cols, int ty, int tx, u32 b, u32 g, u32 t, const u32* words)
{
#pragma unroll
    for (u32 rh = 0; rh < 2; ++rh) {
        const u32 m = g + 8u * rh;
        const int py = ty * 4 + (int)(m >> 2), px = tx * 4 + (int)(m & 3u);
        if (py >= 0 && py < rows && px >= 0 && px < cols) {
#pragma unroll
            for (u32 a = 0; a < 2; ++a)
                *(u32*)(map + ((u64)(u32)((int)(2u * b + a) * rows + py) * (u32)cols + (u32)px) * 16u + t * 4u) = words[2u * a + rh];
        }
    }
}
// e4m3 words of channel blocks 2 ks and 2 ks + 1 of a tile (rows g, g + 8) -> one packed unit at the tensor scale `pre`
NR_INLINE void pack_unit(u8* unit, const V4& lo, const V4& hi, float pre, u32 lane, u32 g, u32 t)
{
    u32 n[4];
#pragma unroll
    for (u32 kh = 0; kh < 2; ++kh) {
        const V4& x = kh ? hi : lo;
        u32 h0[4], h1[4];
        m_e4m3x4_to_h2(x.x, h0[0], h0[1]); m_e4m3x4_to_h2(x.z, h0[2], h0[3]);
        m_e4m3x4_to_h2(x.y, h1[0], h1[1]); m_e4m3x4_to_h2(x.w, h1[2], h1[3]);
        const u32 codes = m_pack2_h(h0, h1, pre, n[2u * kh], n[2u * kh + 1u]);
        m_put_scale(unit + 512u, g, t, kh, codes);
    }
    *(V4*)(unit + lane * 16u) = V4{n[0], n[1], n[2], n[3]};
}
// The A scale word of a lane built from the code pairs of the two k32 halves (each lane holds the pair of its own
// block: byte 0 row g, byte 1 row g + 8), without shared memory: m_put_scale's layout through four shuffles.
NR_INLINE u32 scale_word(u32 c0, u32 c1, u32 lane, u32 t)
{
    const u32 base = lane & ~3u, sel = (t & 1u) * 8u;
    const u32 b00 = (__shfl_sync(0xffffffffu, c0, base) >> sel) & 0xFFu, b01 = (__shfl_sync(0xffffffffu, c0, base + 2u) >> sel) & 0xFFu;
    const u32 b10 = (__shfl_sync(0xffffffffu, c1, base) >> sel) & 0xFFu, b11 = (__shfl_sync(0xffffffffu, c1, base + 2u) >> sel) & 0xFFu;
    return b00 | (b01 << 8) | (b10 << 16) | (b11 << 24);
}
// Accumulators of two 32-output slabs (outputs 8 j + 2 t + e of rows g, g + 8) times `s`, as float16, packed at the
// tensor scale `pre` into the A operand of the next product (slab 0 = K 0..31, slab 1 = K 32..63): lane-local
NR_INLINE void acc_to_operand(const float (&a0)[4][4], const float (&a1)[4][4], float s, float pre, u32 lane, u32 t, V4& A, u32& sa)
{
    u32 n[4], c[2];
#pragma unroll
    for (u32 kh = 0; kh < 2; ++kh) {
        const float (&a)[4][4] = kh ? a1 : a0;
        u32 h0[4], h1[4];
#pragma unroll
        for (u32 j = 0; j < 4; ++j) { h0[j] = h2_pack(a[j][0] * s, a[j][1] * s); h1[j] = h2_pack(a[j][2] * s, a[j][3] * s); }
        c[kh] = m_pack2_h(h0, h1, pre, n[2u * kh], n[2u * kh + 1u]);
    }
    A = V4{n[0], n[1], n[2], n[3]};
    sa = scale_word(c[0], c[1], lane, t);
}
// one k64 step of one 32-output slab against an A operand
NR_INLINE void slab_mma(float (&acc)[4][4], const u8* sl, const V4& a, u32 sa, u32 lane, u32 g)
{
    const V4 n01 = *(const V4*)(sl + lane * 16u), n23 = *(const V4*)(sl + 512u + lane * 16u);
    const V4 sw = *(const V4*)(sl + 1024u + g * 16u);
    mma4(acc[0], a, n01.x, n01.y, sa, sw.x); mma4(acc[1], a, n01.z, n01.w, sa, sw.y);
    mma4(acc[2], a, n23.x, n23.y, sa, sw.z); mma4(acc[3], a, n23.z, n23.w, sa, sw.w);
}
NR_INLINE void zero4(float (&acc)[4][4])
{
#pragma unroll
    for (u32 j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
}
NR_INLINE V4 unit_a(const u8* unit, u32 lane) { return *(const V4*)(unit + lane * 16u); }
NR_INLINE u32 unit_s(const u8* unit, u32 g, u32 t) { return *(const u32*)(unit + 512u + g * 8u + (t & 1u) * 4u); }

// ================================================================================================================
#if KIND == KIND_FFWD
// CTA = one unshifted 8x8 window x four of the eight groups (g = .. + 4 z), as the runtime. Block (32, 8): warp = (tile
// row w >> 2, group (w & 3) + 4 z); VIEW_IN block (32, 4): warp = group w + 4 z for both tile rows.
// Per warp and tile row: reduce acc1 = X . A_g (K 512) -> f16(acc1 s1) packed (t2) = one k64 operand; per pair of hidden
// chunks: acc2 = r . Q0 (K 64), x = f16(acc2 s2), the runtime's f16 cubic SiLU, packed (t3) = one k64 operand of
// acc3 += h . Q2 (K 256 over four steps); out = e4m3(f16(acc3 s3)).
extern "C" __global__ void nr_split4(Params p)
{
    __shared__ __align__(16) u8 sm_x[4u * 8u * UNIT];
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, nw = blockDim.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int TW = p.W >> 2, TH = p.H >> 2, ty0 = 2 * (int)blockIdx.y, tx0 = 2 * (int)blockIdx.x;
    const float* const F = (const float*)p.w;
    const u32* const O = (const u32*)(p.w + 64);
    if (p.pflags) {
        if (leader) wait_region(p.pflags, 2, 3, p.H, p.W, p.psx, p.psy, 8 * (int)blockIdx.y, 8 * (int)blockIdx.x, 8, 8);
        __syncthreads();
    }
    for (u32 u = w; u < 32u; u += nw) {                         // unit = (tile s, k64 step ks)
        const u32 s = u >> 3, ks = u & 7u;
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        const bool in = ty < TH && tx < TW;
#if VIEW_IN
        const V4 lo = view_load(p.in, p.H, p.W, ty, tx, 2u * ks, g, t), hi = view_load(p.in, p.H, p.W, ty, tx, 2u * ks + 1u, g, t);
#else
        const u8* src = p.in + (u64)(u32)(ty * TW + tx) * 8192u + ks * 1024u + lane * 16u;
        const V4 lo = in ? load_cg(src) : V4{0u, 0u, 0u, 0u}, hi = in ? load_cg(src + 512u) : V4{0u, 0u, 0u, 0u};
#endif
        pack_unit(sm_x + u * UNIT, lo, hi, F[F_T1], lane, g, t);
    }
    __syncthreads();
    const u32 grp = (nw == 8u ? (w & 3u) : w) + 4u * blockIdx.z;
    const u8* const wa = p.w + O[O_M1], * const wq0 = p.w + O[O_M2], * const wq2 = p.w + O[O_M3];
    const float s1 = F[F_S1], t2 = F[F_T2], s2 = F[F_S2], t3 = F[F_T3], s3 = F[F_S3];
    for (u32 tr = (nw == 8u ? (w >> 2) : 0u); tr < 2u; tr += (nw == 8u ? 2u : 1u)) {
        V4 RA[2];
        u32 RS[2];
        {
            float acc[2][2][4][4];
#pragma unroll
            for (u32 d = 0; d < 2; ++d) { zero4(acc[d][0]); zero4(acc[d][1]); }
#pragma unroll 2
            for (u32 ks = 0; ks < 8u; ++ks)
#pragma unroll
                for (u32 sb = 0; sb < 2; ++sb) {
                    const u8* sl = wa + ((2u * grp + sb) * 8u + ks) * SLAB;
#pragma unroll
                    for (u32 d = 0; d < 2; ++d) {
                        const u8* unit = sm_x + ((tr * 2u + d) * 8u + ks) * UNIT;
                        slab_mma(acc[d][sb], sl, unit_a(unit, lane), unit_s(unit, g, t), lane, g);
                    }
                }
#pragma unroll
            for (u32 d = 0; d < 2; ++d) acc_to_operand(acc[d][0], acc[d][1], s1, t2, lane, t, RA[d], RS[d]);
        }
        float out[2][2][4][4];
#pragma unroll
        for (u32 d = 0; d < 2; ++d) { zero4(out[d][0]); zero4(out[d][1]); }
#pragma unroll 1
        for (u32 c = 0; c < 4u; ++c) {
            float acc[2][2][4][4];
#pragma unroll
            for (u32 d = 0; d < 2; ++d) { zero4(acc[d][0]); zero4(acc[d][1]); }
#pragma unroll
            for (u32 sb = 0; sb < 2; ++sb) {
                const u8* sl = wq0 + (grp * 8u + 2u * c + sb) * SLAB;
#pragma unroll
                for (u32 d = 0; d < 2; ++d) slab_mma(acc[d][sb], sl, RA[d], RS[d], lane, g);
            }
#pragma unroll
            for (u32 d = 0; d < 2; ++d) {
                u32 n[4], cc[2];
#pragma unroll
                for (u32 kh = 0; kh < 2; ++kh) {
                    u32 h[2][4];
#pragma unroll
                    for (u32 j = 0; j < 4; ++j)
#pragma unroll
                        for (u32 rh = 0; rh < 2; ++rh) {
                            const u32 x = h2_pack(acc[d][kh][j][2u * rh] * s2, acc[d][kh][j][2u * rh + 1u] * s2);
                            const u32 cl = h2_min(h2_max(x, 0xC400C400u), 0x44004400u);
                            const u32 v = h2_fma(cl, h2_fma(h2_abs(cl), 0xAB28AB28u, 0x37283728u), 0x3B283B28u);
                            h[rh][j] = h2_mul(x, v);
                        }
                    cc[kh] = m_pack2_h(h[0], h[1], t3, n[2u * kh], n[2u * kh + 1u]);
                }
                const V4 HA = {n[0], n[1], n[2], n[3]};
                const u32 HS = scale_word(cc[0], cc[1], lane, t);
#pragma unroll
                for (u32 sb = 0; sb < 2; ++sb) slab_mma(out[d][sb], wq2 + ((2u * grp + sb) * 4u + c) * SLAB, HA, HS, lane, g);
            }
        }
#pragma unroll
        for (u32 d = 0; d < 2; ++d) {
            const int ty = ty0 + (int)tr, tx = tx0 + (int)d;
            if (ty < TH && tx < TW) {
#pragma unroll
                for (u32 sb = 0; sb < 2; ++sb) {
                    const float (&a)[4][4] = out[d][sb];
                    const V4 o = {f32x4_to_e4m3x4(a[0][0] * s3, a[0][1] * s3, a[1][0] * s3, a[1][1] * s3),
                                  f32x4_to_e4m3x4(a[0][2] * s3, a[0][3] * s3, a[1][2] * s3, a[1][3] * s3),
                                  f32x4_to_e4m3x4(a[2][0] * s3, a[2][1] * s3, a[3][0] * s3, a[3][1] * s3),
                                  f32x4_to_e4m3x4(a[2][2] * s3, a[2][3] * s3, a[3][2] * s3, a[3][3] * s3)};
                    *(V4*)(p.out + (u64)(u32)(ty * TW + tx) * 8192u + (2u * grp + sb) * 512u + lane * 16u) = o;
                }
            }
        }
    }
    if (p.oflags) {
        __syncthreads();
        if (leader) flag_publish(p.oflags + (2 * blockIdx.y + blockIdx.z) * gridDim.x + blockIdx.x, 0);
    }
}

// ================================================================================================================
#elif KIND == KIND_PROJ || KIND == KIND_POOL || KIND == KIND_HEAD
// CTA = one unshifted 8x8 window (2x2 tiles) x 256 outputs (h = blockIdx.x / Gx), as the runtime; the window's 4-bit
// operand (4 tiles x 8 k64 steps) is packed once into shared memory, then warp w takes 32-output slabs w, w + nw, ..
// of the CTA's eight, all four tiles. v = fma(acc, s1, f16(f16(gain) R)), out = e4m3(f16(v)) (HEAD: acc s1).
#if KIND == KIND_HEAD
#define COUT 1024u
#else
#define COUT 512u
#endif
extern "C" __global__ void nr_split4(Params p)
{
    __shared__ __align__(16) u8 sm_x[4u * 8u * UNIT];
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, nw = blockDim.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int Gx = (p.W + 7) >> 3;
    const int col = (int)blockIdx.x % Gx, h = (int)blockIdx.x / Gx, cy = (int)blockIdx.y;
    const int TW = p.W >> 2, TH = p.H >> 2, ty0 = 2 * cy, tx0 = 2 * col;
    const float* const F = (const float*)p.w;
    const u32* const O = (const u32*)(p.w + 64);
    if (p.pflags) {
        if (leader) {
#if KIND == KIND_HEAD
            wait_region(p.pflags, 2, 2, p.H, p.W, 0, 0, 8 * cy, 8 * col, 8, 8);         // pooled tiles, both halves
#elif VIEW_RES
            wait_region(p.pflags, 2, 3, p.H, p.W, 0, 0, 8 * cy, 8 * col, 8, 8);         // ffwd_inpview, both halves
#else
            wait_region(p.pflags, PARTS, 3, p.H, p.W, p.psx, p.psy, 8 * cy, 8 * col, 8, 8);    // qkv (4) or ffwd (2)
#endif
        }
        __syncthreads();
    }
    for (u32 u = w; u < 32u; u += nw) {
        const u32 s = u >> 3, ks = u & 7u;
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        const bool in = ty < TH && tx < TW;
        const u8* src = p.in + (u64)(u32)(ty * TW + tx) * 8192u + ks * 1024u + lane * 16u;
        const V4 lo = in ? load_cg(src) : V4{0u, 0u, 0u, 0u}, hi = in ? load_cg(src + 512u) : V4{0u, 0u, 0u, 0u};
        pack_unit(sm_x + u * UNIT, lo, hi, F[F_T1], lane, g, t);
    }
    __syncthreads();
    const float s1 = F[F_S1];
#pragma unroll 1
    for (u32 sb = w; sb < 8u; sb += nw) {
        const u32 ob = 8u * (u32)h + sb;
        float acc[4][4][4];
#pragma unroll
        for (u32 s = 0; s < 4; ++s) zero4(acc[s]);
        const u8* wm = p.w + O[O_M1] + (u64)(ob * 8u) * SLAB;
#pragma unroll 2
        for (u32 ks = 0; ks < 8u; ++ks)
#pragma unroll
            for (u32 s = 0; s < 4; ++s) {
                const u8* unit = sm_x + (s * 8u + ks) * UNIT;
                slab_mma(acc[s], wm + ks * SLAB, unit_a(unit, lane), unit_s(unit, g, t), lane, g);
            }
#if KIND != KIND_HEAD
        const float* const gain = (const float*)(p.w + O[O_GAIN] + (ob * 4u + t) * 32u);
#endif
#if KIND == KIND_POOL
        u32 o16[4][2][4];
#endif
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
            const bool in = ty < TH && tx < TW;
            u32 ow[4];
#if KIND != KIND_HEAD
#if VIEW_RES
            const V4 rv = view_load(p.res, p.H, p.W, ty, tx, ob, g, t);
#else
            const V4 rv = in ? load_cg(p.res + (u64)(u32)(ty * TW + tx) * 8192u + ob * 512u + lane * 16u) : V4{0u, 0u, 0u, 0u};
#endif
            const u32 rw[4] = {rv.x, rv.y, rv.z, rv.w};
#endif
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                float o[8];
#if KIND == KIND_HEAD
#pragma unroll
                for (u32 i = 0; i < 8; ++i) o[i] = acc[s][i >> 1][rh * 2u + (i & 1u)] * s1;
#else
                float r[8];
                e4m3x4_to_f32(rw[rh], r);
                e4m3x4_to_f32(rw[2u + rh], r + 4);
#pragma unroll
                for (u32 i = 0; i < 8; ++i) {
                    const float c = f16_round(gain[i] * r[i]);                          // f16(f16(gain) R): exact product, one rounding
                    o[i] = fmaf(acc[s][i >> 1][rh * 2u + (i & 1u)], s1, c);
                }
#endif
#if KIND == KIND_POOL
#pragma unroll
                for (u32 j = 0; j < 4; ++j) o16[s][rh][j] = h2_pack(o[2 * j], o[2 * j + 1]);
#endif
                ow[rh] = f32x4_to_e4m3x4(o[0], o[1], o[2], o[3]);
                ow[2u + rh] = f32x4_to_e4m3x4(o[4], o[5], o[6], o[7]);
            }
#if VIEW_OUT
            if (in) view_store(p.out, p.H, p.W, ty, tx, ob, g, t, ow);
#else
            if (in) *(V4*)(p.out + (u64)(u32)(ty * TW + tx) * (16u * COUT) + ob * 512u + lane * 16u) = V4{ow[0], ow[1], ow[2], ow[3]};
#endif
        }
#if KIND == KIND_POOL
        // 2x2 mean of the float16 values: column pairs (token g ^ 1 = lane ^ 4), then row pairs (token g ^ 4 = lane ^ 16),
        // * 0.25, e4m3; lanes with g in {0, 2} hold pooled token (row rh of the tile's 2x2 block, column g >> 1)
        const int PTW = p.d1 >> 2;
        u8* const ptile = p.out2 + (u64)(u32)(cy * PTW + col) * 8192u + ob * 512u;
#pragma unroll
        for (u32 s = 0; s < 4; ++s)
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
                for (u32 j = 0; j < 4; ++j) {
                    const u32 a = h2_add(o16[s][rh][j], __shfl_xor_sync(0xffffffffu, o16[s][rh][j], 4));
                    const u32 b = h2_mul(h2_add(a, __shfl_xor_sync(0xffffffffu, a, 16)), 0x34003400u);
                    if ((g & 5u) == 0u && cy < (p.d0 >> 2) && col < PTW) {
                        const u32 py = (s >> 1) * 2u + rh, px = (s & 1u) * 2u + (g >> 1);
                        const u32 m = py * 4u + px;
                        const u32 at = ((m & 7u) * 4u + t) * 16u + ((j >> 1) * 2u + (m >> 3)) * 4u + (j & 1u) * 2u;
                        *(u16*)(ptile + at) = (u16)h2_to_e4m3x2(b);
                    }
                }
#endif
    }
    if (p.oflags) {
        __syncthreads();
        if (leader) flag_publish(p.oflags + blockIdx.y * gridDim.x + blockIdx.x, 0);
    }
}

// ================================================================================================================
#elif KIND == KIND_DEC
// CTA = one coarse 4x4 tile (ty, tx) x 64 outputs (half * 256 + z * 64; the runtime's z is a K quarter, ours an output
// quarter, so every CTA finishes its own outputs and no partial sums are merged) x full K = 1024; warp = 32 outputs.
// p = f16(acc s1) of the 16 coarse tokens, then for the 2x2 output tiles out = e4m3(f16(p + f16(skip gain))).
// Completion: each CTA counts itself on counter[idx] (-1 before the launch); the fourth (old value 2) publishes
// flags[idx] = 0 after a fence. No waits.
extern "C" __global__ void nr_split4(Params p)
{
    __shared__ __align__(16) u8 sm_x[16u * UNIT];
    __shared__ __align__(16) u32 sm_p[16u * 32u];               // [coarse token][64 outputs as float16 pairs]
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, nw = blockDim.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int Tc = (p.d1 + 3) >> 2;
    const int half = (int)blockIdx.x / Tc, tx = (int)blockIdx.x % Tc, ty = (int)blockIdx.y, z = (int)blockIdx.z;
    const int CTW = p.d1 >> 2, CTH = p.d0 >> 2;
    const bool cin = ty < CTH && tx < CTW;
    const float* const F = (const float*)p.w;
    const u32* const O = (const u32*)(p.w + 64);
    for (u32 ks = w; ks < 16u; ks += nw) {
        const u8* src = p.in + (u64)(u32)(ty * CTW + tx) * 16384u + ks * 1024u + lane * 16u;
        const V4 lo = cin ? load_cg(src) : V4{0u, 0u, 0u, 0u}, hi = cin ? load_cg(src + 512u) : V4{0u, 0u, 0u, 0u};
        pack_unit(sm_x + ks * UNIT, lo, hi, F[F_T1], lane, g, t);
    }
    __syncthreads();
    const float s1 = F[F_S1];
    for (u32 q = w; q < 2u; q += nw) {
        const u32 ob = 8u * (u32)half + 2u * (u32)z + q;
        float acc[4][4];
        zero4(acc);
        const u8* wm = p.w + O[O_M1] + (u64)(ob * 16u) * SLAB;
#pragma unroll 4
        for (u32 ks = 0; ks < 16u; ++ks) slab_mma(acc, wm + ks * SLAB, unit_a(sm_x + ks * UNIT, lane), unit_s(sm_x + ks * UNIT, g, t), lane, g);
#pragma unroll
        for (u32 j = 0; j < 4; ++j) {
            sm_p[g * 32u + q * 16u + 4u * j + t] = h2_pack(acc[j][0] * s1, acc[j][1] * s1);
            sm_p[(g + 8u) * 32u + q * 16u + 4u * j + t] = h2_pack(acc[j][2] * s1, acc[j][3] * s1);
        }
    }
    __syncthreads();
    const int TW = p.W >> 2, TH = p.H >> 2;
    for (u32 q = w; q < 2u; q += nw) {
        const u32 ob = 8u * (u32)half + 2u * (u32)z + q;
        const float* const gain = (const float*)(p.w + O[O_GAIN] + (ob * 4u + t) * 32u);
        const u32 g16[4] = {h2_pack(gain[0], gain[1]), h2_pack(gain[2], gain[3]), h2_pack(gain[4], gain[5]), h2_pack(gain[6], gain[7])};
#pragma unroll
        for (u32 f = 0; f < 4; ++f) {
            const int fty = 2 * ty + (int)(f >> 1), ftx = 2 * tx + (int)(f & 1u);
            if (fty >= TH || ftx >= TW) continue;
            u8* const tile = p.out + (u64)(u32)(fty * TW + ftx) * 8192u + ob * 512u + lane * 16u;
            const V4 sv = load_cg(p.res + (u64)(u32)(fty * TW + ftx) * 8192u + ob * 512u + lane * 16u);
            const u32 sw[4] = {sv.x, sv.y, sv.z, sv.w};
            u32 ow[4];
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                const u32 m = g + 8u * rh;
                const u32 fy = (f >> 1) * 4u + (m >> 2), fx = (f & 1u) * 4u + (m & 3u);
                const u32 cm = (fy >> 1) * 4u + (fx >> 1);
                u32 sk[4];
                m_e4m3x4_to_h2(sw[rh], sk[0], sk[1]);
                m_e4m3x4_to_h2(sw[2u + rh], sk[2], sk[3]);
                u32 v[4];
#pragma unroll
                for (u32 j = 0; j < 4; ++j) v[j] = h2_add(sm_p[cm * 32u + q * 16u + 4u * j + t], h2_mul(sk[j], g16[j]));
                ow[rh] = m_h2x2_to_e4m3x4(v[0], v[1]);
                ow[2u + rh] = m_h2x2_to_e4m3x4(v[2], v[3]);
            }
            *(V4*)tile = V4{ow[0], ow[1], ow[2], ow[3]};
        }
    }
    if (p.counter) {
        __threadfence();
        __syncthreads();
        if (leader) {
            const int idx = (int)(blockIdx.y * gridDim.x + blockIdx.x);
            const int old = atomicAdd(p.counter + idx, 1);
            if (old == 2 && p.oflags) {
                __threadfence();
                flag_publish(p.oflags + idx, 0);
            }
        }
    }
}

// ================================================================================================================
#elif KIND == KIND_QKV
// CTA = one (shifted) 8x8 window x four heads (z), warp = one head, as the runtime. QKV = X . Wqkv (4-bit, K 512) for
// the window's four tiles, then swin4.cu's attention (norms on the accumulators times s1^2, FP8 Q.K on the position
// bias in a float16 accumulator, the exponent ramp, sums, reciprocal, P e4m3, P.V in two k32 steps); the context
// (float16) is stored as e4m3 in channel block `head` of the tiles inside the map.
extern "C" __global__ void nr_split4(Params p)
{
    __shared__ __align__(16) u8 sm_x[4u * 8u * UNIT];
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, nw = blockDim.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int x0 = p.sx + 8 * (int)blockIdx.x, y0 = p.sy + 8 * (int)blockIdx.y;
    const int TW = p.W >> 2, TH = p.H >> 2, tx0 = x0 >> 2, ty0 = y0 >> 2;
    const float* const F = (const float*)p.w;
    const u32* const O = (const u32*)(p.w + 64);
    if (p.pflags) {
        if (leader) wait_region(p.pflags, 2, 3, p.H, p.W, p.psx, p.psy, y0, x0, 8, 8);
        __syncthreads();
    }
    bool inside[4];
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        inside[s] = ty >= 0 && ty < TH && tx >= 0 && tx < TW;
    }
    for (u32 u = w; u < 32u; u += nw) {
        const u32 s = u >> 3, ks = u & 7u;
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        const bool in = ty >= 0 && ty < TH && tx >= 0 && tx < TW;
        const u8* src = p.in + (u64)(u32)(in ? ty * TW + tx : 0) * 8192u + ks * 1024u + lane * 16u;
        const V4 lo = in ? load_cg(src) : V4{0u, 0u, 0u, 0u}, hi = in ? load_cg(src + 512u) : V4{0u, 0u, 0u, 0u};
        pack_unit(sm_x + u * UNIT, lo, hi, F[F_T1], lane, g, t);
    }
    __syncthreads();
    const u32 head = 4u * blockIdx.z + w;
    const bool odd = (t & 1u) != 0u;
    const float sqkv = F[F_S1], sq2 = sqkv * sqkv;
    const u8* const wqkv = p.w + O[O_M1];
    u32 KB[8][2];
    u32 VB[4][2][2];
#pragma unroll
    for (u32 which = 1; which < 3; ++which) {
        const u8* wq = wqkv + (u64)((head * 3u + which) * 8u) * SLAB;
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            float acc[4][4];
            zero4(acc);
#pragma unroll 2
            for (u32 ks = 0; ks < 8u; ++ks) {
                const u8* unit = sm_x + (s * 8u + ks) * UNIT;
                slab_mma(acc, wq + ks * SLAB, unit_a(unit, lane), unit_s(unit, g, t), lane, g);
            }
            if (which == 1u) att2_qk_rows(acc, sq2, sqkv, odd, KB[s * 2u][0], KB[s * 2u][1], KB[s * 2u + 1u][0], KB[s * 2u + 1u][1]);
            else att2_v_rows(acc, sqkv, VB[0][s >> 1][s & 1u], VB[1][s >> 1][s & 1u], VB[2][s >> 1][s & 1u], VB[3][s >> 1][s & 1u]);
        }
    }
    // OPT_ATT4: Q.K and P.V as NVFP4 products, every operand rebuilt in the lane from the FP8 operand bytes above (the
    // head dimensions and the keys keep the FP8 instruction's lane split, so A and B agree on the K order): blocks of 16
    // = this lane and lane ^ 1, tensor scale 1, K 32..63 of Q.K zero. B scale bytes = K blocks 0..3 of column g.
    const bool att4 = (p.options & OPT_ATT4) != 0u;
    const u32 base = lane & ~3u;
    u32 KB4[8], KS4[8], VB4[4][2], VS4[4];
    if (att4) {
#pragma unroll
        for (u32 nt = 0; nt < 8; nt += 2) {
            u32 h0[4], h1[4];
            m_e4m3x4_to_h2(KB[nt][0], h0[0], h0[1]); m_e4m3x4_to_h2(KB[nt][1], h0[2], h0[3]);
            m_e4m3x4_to_h2(KB[nt + 1][0], h1[0], h1[1]); m_e4m3x4_to_h2(KB[nt + 1][1], h1[2], h1[3]);
            const u32 c = m_pack2_h(h0, h1, 1.0f, KB4[nt], KB4[nt + 1]);
            const u32 c0 = __shfl_sync(0xffffffffu, c, base), c2 = __shfl_sync(0xffffffffu, c, base + 2u);
            KS4[nt] = (c0 & 0xFFu) | ((c2 & 0xFFu) << 8);
            KS4[nt + 1] = ((c0 >> 8) & 0xFFu) | (((c2 >> 8) & 0xFFu) << 8);
        }
#pragma unroll
        for (u32 j = 0; j < 4; ++j) {
            u32 h0[4], h1[4];
            m_e4m3x4_to_h2(VB[j][0][0], h0[0], h0[1]); m_e4m3x4_to_h2(VB[j][0][1], h0[2], h0[3]);
            m_e4m3x4_to_h2(VB[j][1][0], h1[0], h1[1]); m_e4m3x4_to_h2(VB[j][1][1], h1[2], h1[3]);
            const u32 c = m_pack2_h(h0, h1, 1.0f, VB4[j][0], VB4[j][1]);
            const u32 c0 = __shfl_sync(0xffffffffu, c, base), c2 = __shfl_sync(0xffffffffu, c, base + 2u);
            VS4[j] = (c0 & 0xFFu) | ((c2 & 0xFFu) << 8) | (((c0 >> 8) & 0xFFu) << 16) | (((c2 >> 8) & 0xFFu) << 24);
        }
    }
    const float fq = sqkv * ((const float*)(p.w + O[O_HEADSCALE]))[head];
    const u8* const wqq = wqkv + (u64)((head * 3u) * 8u) * SLAB;
    const u8* const bias = p.w + O[O_BIAS] + (u64)head * 8192u + lane * 16u;
#pragma unroll 1
    for (u32 qs = 0; qs < 4u; ++qs) {
        u32 QA[4];
        {
            float acc[4][4];
            zero4(acc);
#pragma unroll 2
            for (u32 ks = 0; ks < 8u; ++ks) {
                const u8* unit = sm_x + (qs * 8u + ks) * UNIT;
                slab_mma(acc, wqq + ks * SLAB, unit_a(unit, lane), unit_s(unit, g, t), lane, g);
            }
            att2_qk_rows(acc, sq2, fq, odd, QA[0], QA[2], QA[1], QA[3]);
        }
        u32 lacc[8][2];
#pragma unroll
        for (u32 kt = 0; kt < 4; ++kt) {
            const V4 b = *(const V4*)(bias + (qs * 4u + kt) * 512u);
            lacc[kt * 2u][0] = b.x; lacc[kt * 2u][1] = b.y; lacc[kt * 2u + 1u][0] = b.z; lacc[kt * 2u + 1u][1] = b.w;
        }
        if (att4) {
            u32 h0[4], h1[4], n0, n1;
            m_e4m3x4_to_h2(QA[0], h0[0], h0[1]); m_e4m3x4_to_h2(QA[2], h0[2], h0[3]);
            m_e4m3x4_to_h2(QA[1], h1[0], h1[1]); m_e4m3x4_to_h2(QA[3], h1[2], h1[3]);
            const u32 cq = m_pack2_h(h0, h1, 1.0f, n0, n1);
            const V4 qa4 = {n0, n1, 0u, 0u};
            const u32 sq = scale_word(cq, 0u, lane, t);
#pragma unroll
            for (u32 nt = 0; nt < 8; ++nt) {                    // float32 accumulator on the bias, then float16
                float c[4];
                h2_unpack(lacc[nt][0], c[0], c[1]);
                h2_unpack(lacc[nt][1], c[2], c[3]);
                mma4(c, qa4, KB4[nt], 0u, sq, KS4[nt]);
                lacc[nt][0] = h2_pack(c[0], c[1]);
                lacc[nt][1] = h2_pack(c[2], c[3]);
            }
        } else {
#pragma unroll
            for (u32 nt = 0; nt < 8; ++nt) mma8(lacc[nt], QA, KB[nt][0], KB[nt][1]);
        }
        u32 sums[2];
#pragma unroll
        for (u32 r = 0; r < 2; ++r) {
#pragma unroll
            for (u32 nt = 0; nt < 8; ++nt) {
                u32 x = h2_fma(lacc[nt][r], 0x29C029C0u, 0x3D343D34u);
                x = h2_min(h2_max(x, 0x3C203C20u), 0x3E473E47u);
                lacc[nt][r] = (x << 5) + 0x7FF88000u;
            }
            sums[r] = h2_add(h2_add(h2_add(h2_add(lacc[0][r], lacc[1][r]), h2_add(lacc[2][r], lacc[3][r])), h2_add(lacc[4][r], lacc[5][r])),
                             h2_add(lacc[6][r], lacc[7][r]));
        }
        u32 total[2];
        {
            u32 sum = h2_add(odd ? sums[1] : sums[0], __shfl_xor_sync(0xffffffffu, odd ? sums[0] : sums[1], 1));
            sum = h2_add(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
            float lo, hi;
            h2_unpack(sum, lo, hi);
            const float reciprocal = rcp_approx(fmaxf(lo + hi, 6.19888305664e-05f));
            const u32 mine = h2_pack(reciprocal, reciprocal);
            const u32 other = __shfl_xor_sync(0xffffffffu, mine, 1);
            total[0] = odd ? other : mine;
            total[1] = odd ? mine : other;
        }
        u32 PA[2][4];
#pragma unroll
        for (u32 ks = 0; ks < 2; ++ks)
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                PA[ks][r] = m_h2x2_to_e4m3x4(h2_mul(lacc[4 * ks][r], total[r]), h2_mul(lacc[4 * ks + 1][r], total[r]));
                PA[ks][2u + r] = m_h2x2_to_e4m3x4(h2_mul(lacc[4 * ks + 2][r], total[r]), h2_mul(lacc[4 * ks + 3][r], total[r]));
            }
        if (att4) {                                             // P (e4m3) as the 4-bit A operand: tiles 0, 1 = K 0..31
            V4 pa4;
            u32 cp[2];
#pragma unroll
            for (u32 kh = 0; kh < 2; ++kh) {
                u32 h0[4], h1[4];
                m_e4m3x4_to_h2(PA[kh][0], h0[0], h0[1]); m_e4m3x4_to_h2(PA[kh][2], h0[2], h0[3]);
                m_e4m3x4_to_h2(PA[kh][1], h1[0], h1[1]); m_e4m3x4_to_h2(PA[kh][3], h1[2], h1[3]);
                u32 n0, n1;
                cp[kh] = m_pack2_h(h0, h1, 1.0f, n0, n1);
                if (kh) { pa4.z = n0; pa4.w = n1; } else { pa4.x = n0; pa4.y = n1; }
            }
            const u32 sp = scale_word(cp[0], cp[1], lane, t);
            float cf[4][4];
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                cf[j][0] = cf[j][1] = cf[j][2] = cf[j][3] = 0.0f;
                mma4(cf[j], pa4, VB4[j][0], VB4[j][1], sp, VS4[j]);
            }
            if (inside[qs]) {
                const int ty = ty0 + (int)(qs >> 1), tx = tx0 + (int)(qs & 1u);
                const V4 o = {f32x4_to_e4m3x4(cf[0][0], cf[0][1], cf[1][0], cf[1][1]), f32x4_to_e4m3x4(cf[0][2], cf[0][3], cf[1][2], cf[1][3]),
                              f32x4_to_e4m3x4(cf[2][0], cf[2][1], cf[3][0], cf[3][1]), f32x4_to_e4m3x4(cf[2][2], cf[2][3], cf[3][2], cf[3][3])};
                *(V4*)(p.out + (u64)(u32)(ty * TW + tx) * 8192u + head * 512u + lane * 16u) = o;
            }
            continue;
        }
        u32 cacc[4][2];
#pragma unroll
        for (u32 j = 0; j < 4; ++j) {
            cacc[j][0] = cacc[j][1] = 0u;
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) mma8(cacc[j], PA[ks], VB[j][ks][0], VB[j][ks][1]);
        }
        if (inside[qs]) {
            const int ty = ty0 + (int)(qs >> 1), tx = tx0 + (int)(qs & 1u);
            const V4 o = {m_h2x2_to_e4m3x4(cacc[0][0], cacc[1][0]), m_h2x2_to_e4m3x4(cacc[0][1], cacc[1][1]),
                          m_h2x2_to_e4m3x4(cacc[2][0], cacc[3][0]), m_h2x2_to_e4m3x4(cacc[2][1], cacc[3][1])};
            *(V4*)(p.out + (u64)(u32)(ty * TW + tx) * 8192u + head * 512u + lane * 16u) = o;
        }
    }
    if (p.oflags) {
        __syncthreads();
        if (leader) flag_publish(p.oflags + (4 * blockIdx.y + blockIdx.z) * gridDim.x + blockIdx.x, 0);
    }
}
#endif
