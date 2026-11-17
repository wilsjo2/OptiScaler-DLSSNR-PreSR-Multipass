// The first build of swin4.cu, kept for comparisons (model: v2/sim/swin4_model_first.py, assets: v2/assets/swin4-first).
// NVFP4 (4-bit) replacement of the DLSS-NR fused Swin blocks: cc_tinlayout_fused_swin_{1h_32,2h_64,4h_128,8h_256}_*.
// Target sm_120a, built with NVRTC (--fmad=false), one cubin per (HEADS, VARIANT):
//   -DHEADS=1|2|4|8      channels C = 32 * HEADS
//   -DVARIANT=0          tile-layout input, tile-layout output       (the runtime's plain / tilesync / wait / chained)
//            =1          view-layout input                           (inpview*)
//            =2          ... plus 2x2 mean and C -> 2C projection    (ds*)
//            =3          2C -> C projection of the coarser level, 2x replication, blend with the skip   (upsample*)
//            =4          view-layout output                          (outview*)
// Launch exactly as the runtime launches its own kernel: block (32, HEADS), one CTA per 8x8-token window, grid
// ceil((W - min(sx, 0)) / 8) x ceil((H - min(sy, 0)) / 8), no dynamic shared memory.
//
// What is 4-bit: every product with a weight matrix (expand, mid, contract, QKV, output projection, resample), both
// operands. Activations are packed in the kernel, one ue4m3 scale per 16 channels of a token. The attention itself
// (Q.K, softmax, P.V) is the runtime's own FP8 / float16 arithmetic.
//
// One warp is one head (32 channels of all 64 tokens of the window). A lane (g = lane >> 2, t = lane & 3) holds tokens
// m = g and g + 8 of each of the four 4x4-token tiles, and of every 32 channels the eight with ((c >> 1) & 3) == t.
// Channel c = 32 h + 8 j + 2 t + e sits at k64 step h >> 1, K = 32 (h & 1) + 8 t + 2 j + e, so a lane packs its own
// values into its own operand words and only the block scale needs the lane next to it.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };

#define NR_INLINE static __forceinline__ __device__

#ifndef HEADS
#define HEADS 8
#endif
#ifndef VARIANT
#define VARIANT 0
#endif
// 0: the block. 1: stop after the MLP and write its output. 2: write the input back (layouts and flags only).
#ifndef DBG_STAGE
#define DBG_STAGE 0
#endif
// Tiles of the window taken per pass over a weight matrix: 4, or 2 (the accumulators in registers are halved and every
// weight block is loaded twice).
#ifndef TPP
#define TPP 4u
#endif
#define C (32u * HEADS)
#define KS ((HEADS + 1u) / 2u)                      // k64 steps over C channels (C = 32: one, half used)
#define SLAB 1152u                                  // 32 outputs x one k64 step: 2 x 512 B of nibbles, 128 B of scales
#define QBYTES (4u * KS * 512u)
#define SBYTES (4u * KS * 64u)
#if HEADS > 1
#define XBYTES (4u * HEADS * 512u)                  // the window as e4m3 codes
#else
#define XBYTES 4096u                                // the window as float16 (the single-head block keeps wide values)
#endif
// Every wait for another kernel's flags draws on one budget of nominal sleep time, so a missing producer costs a
// wrong frame, never a hung GPU (see ffn_v2.cu).
#ifndef WAIT_BUDGET_NS
#define WAIT_BUDGET_NS (1 << 23)
#endif

// options
#define OPT_TWO_X 1u        // HEADS == 1: second 4-bit term for x (expand input)
#define OPT_TWO_Y 2u        //             for y (QKV input)
#define OPT_TWO_C 4u        //             for the attention context (projection input)
#define OPT_TWO_R 8u        //             for the pooled map (ds projection input)

struct Params {
    const u8* in;           // input map: tile layout; VARIANT 1: view layout; VARIANT 3: coarse map, 2C channels, view layout
    u8* out;                // output map: tile layout; VARIANT 4: view layout
    const u8* w;            // our record (see export_swin4.py)
    const u8* skip;         // VARIANT 3: encoder skip, tile layout
    const int* pflags;      // producer's flags or null
    int* oflags;            // our flags or null
    u8* ds_out;             // VARIANT 2: pooled and projected output, view layout
    int H, W;               // token rows and columns
    int sx, sy;             // window shift, 0 or -4
    int psx, psy;           // the producer's
    int d0, d1;             // rows, columns of the view (VARIANT 1, 4) or of the coarse map (VARIANT 3)
    int e0, e1;             // VARIANT 3: rows, columns of the skip map
    u32 options;
    u32 reserved;
};

// record: 16 floats, 16 offsets, then the tables
enum { F_TX, F_HC1, F_HC2, F_HC3, F_HLIM, F_SMID, F_SCON, F_TY, F_SQKV, F_TC, F_SPROJ, F_TR, F_SRES, F_TM };
enum { O_EXPAND, O_MID, O_CONTRACT, O_QKV, O_PROJ, O_RES, O_RS, O_ARS, O_GAIN, O_BIAS, O_HEADSCALE };

// ---------------------------------------------------------------------------------------------------------------
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
// loads of data another kernel of this evaluation wrote: not from this SM's L1
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
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%16, %16}, {%15}, {%16, %16};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
                   "r"(sfa), "r"(sfb), "h"(z));
    c[0] = d0; c[1] = d1; c[2] = d2; c[3] = d3;
}
// the runtime's own FP8 instruction: e4m3 operands, float16 accumulator (two packed pairs)
NR_INLINE void mma8(u32* c, const u32* a, u32 b0, u32 b1)
{
    u32 d0, d1;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
                 : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}

// float16 pairs packed in a word (low half first)
NR_INLINE u32 h2_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
NR_INLINE u32 h2_mul(u32 a, u32 b) { u32 d; asm("mul.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_add(u32 a, u32 b) { u32 d; asm("add.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_min(u32 a, u32 b) { u32 d; asm("min.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_pack(float lo, float hi) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
NR_INLINE void h2_unpack(u32 h, float& lo, float& hi)
{
    asm("{\n .reg .b16 nr_l, nr_u;\n mov.b32 {nr_l, nr_u}, %2;\n cvt.f32.f16 %0, nr_l;\n cvt.f32.f16 %1, nr_u;\n}" : "=f"(lo), "=f"(hi) : "r"(h));
}
NR_INLINE u32 h2_to_e4m3x2(u32 h) { u16 d; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(h)); return d; }
NR_INLINE u32 transpose8(u32 h)
{
    u32 d;
    asm volatile("movmatrix.sync.trans.aligned.m8n8.b16 %0, %1;" : "=r"(d) : "r"(h));
    return d;
}
NR_INLINE float rsqrt_approx(float x) { float d; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE float rcp_approx(float x) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }

// four e4m3 codes -> four floats (byte 0 first)
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
// four floats -> float16 -> four e4m3 codes (the double rounding of a stored result)
NR_INLINE u32 f32x4_to_e4m3x4(float a, float b, float c, float d)
{
    u32 word;
    asm("{\n .reg .b16 nr_l, nr_u;\n .reg .b32 nr_p, nr_q;\n cvt.rn.f16x2.f32 nr_p, %2, %1;\n cvt.rn.f16x2.f32 nr_q, %4, %3;\n"
        " cvt.rn.satfinite.e4m3x2.f16x2 nr_l, nr_p;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_u, nr_q;\n mov.b32 %0, {nr_l, nr_u};\n}"
        : "=r"(word) : "f"(a), "f"(b), "f"(c), "f"(d));
    return word;
}
// eight floats -> eight e2m1 nibbles (v[0] in the lowest)
NR_INLINE u32 e2m1x8(const float* v)
{
    u32 word;
    asm("{\n .reg .b8 nr_a, nr_b, nr_c, nr_d;\n cvt.rn.satfinite.e2m1x2.f32 nr_a, %2, %1;\n cvt.rn.satfinite.e2m1x2.f32 nr_b, %4, %3;\n"
        " cvt.rn.satfinite.e2m1x2.f32 nr_c, %6, %5;\n cvt.rn.satfinite.e2m1x2.f32 nr_d, %8, %7;\n mov.b32 %0, {nr_a, nr_b, nr_c, nr_d};\n}"
        : "=r"(word) : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return word;
}
// block scale: ue4m3 code of clamp(amax / 6, 2^-9, 448); `scale` is the value the code stands for, `reciprocal` 1 / scale
NR_INLINE u32 block_scale(float amax, float& scale, float& reciprocal)
{
    const float raw = fminf(fmaxf(amax * 0.16666667163372040f, 0.001953125f), 448.0f);
    u32 code;
    float value;
    asm("{\n .reg .b16 nr_c, nr_l, nr_u;\n .reg .b32 nr_p;\n cvt.rn.satfinite.e4m3x2.f32 nr_c, %2, %2;\n cvt.rn.f16x2.e4m3x2 nr_p, nr_c;\n"
        " mov.b32 {nr_l, nr_u}, nr_p;\n cvt.f32.f16 %1, nr_l;\n cvt.u32.u16 %0, nr_c;\n}"
        : "=r"(code), "=f"(value) : "f"(raw));
    scale = value;
    reciprocal = rcp_approx(value);
    return code & 0xFFu;
}
NR_INLINE float amax8(const float* v)
{
    return fmaxf(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))),
                 fmaxf(fmaxf(fabsf(v[4]), fabsf(v[5])), fmaxf(fabsf(v[6]), fabsf(v[7]))));
}
// Eight values of one token (already times the tensor scale) -> eight nibbles; the lane next to this one (lane ^ 1)
// holds the other eight of the block of 16 and arrives at the same scale.
NR_INLINE u32 pack_row(const float* v, u32& code)
{
    float amax = amax8(v);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    float scale, r;
    code = block_scale(amax, scale, r);
    const float q[8] = {v[0] * r, v[1] * r, v[2] * r, v[3] * r, v[4] * r, v[5] * r, v[6] * r, v[7] * r};
    return e2m1x8(q);
}
#if HEADS == 1
// The nearest e2m1 value (0, .5, 1, 1.5, 2, 3, 4, 6; ties to the even code), as cvt.rn.satfinite.e2m1x2 rounds.
NR_INLINE float e2m1_round(float u)
{
    const float a = fminf(fabsf(u), 6.0f);
    const float q = a < 2.0f ? rintf(a * 2.0f) * 0.5f : (a < 4.0f ? rintf(a) : (a <= 5.0f ? 4.0f : 6.0f));
    return copysignf(q, u);
}
// Two 4-bit terms: the second holds what the first left over, with its own scale, and meets the same weights in the
// other half of the k64 instruction. `two` false: the second term is zero.
NR_INLINE u32 pack_row2(const float* v, u32& code, u32& word2, u32& code2, bool two)
{
    float amax = amax8(v);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    float scale, r;
    code = block_scale(amax, scale, r);
    float q[8], rest[8];
#pragma unroll
    for (u32 i = 0; i < 8; ++i) {
        const float u = v[i] * r;
        q[i] = e2m1_round(u);
        rest[i] = (u - q[i]) * scale;
    }
    float amax2 = amax8(rest);
    amax2 = fmaxf(amax2, __shfl_xor_sync(0xffffffffu, amax2, 1));
    float scale2, r2;
    const u32 c2 = block_scale(amax2, scale2, r2);
#pragma unroll
    for (u32 i = 0; i < 8; ++i) rest[i] *= r2;
    const u32 w2 = e2m1x8(rest);
    word2 = two ? w2 : 0u;
    code2 = two ? c2 : 0u;
    return e2m1x8(q);
}
#endif
// Scale byte of a block of 16: byte 2 * khalf + (t >> 1) of the token's word; lane t & 1 == 0 owns the word of token g,
// t & 1 == 1 that of token g + 8 (the lanes the instruction reads its A scales from).
NR_INLINE void put_scale(u8* words, u32 g, u32 t, u32 khalf, u32 code0, u32 code1)
{
    words[g * 8u + (t & 1u) * 4u + khalf * 2u + (t >> 1)] = (u8)((t & 1u) ? code1 : code0);
}

#define ZERO_ACC(acc, n0, n1)                                   \
    for (u32 i_ = 0; i_ < (n0); ++i_)                           \
        for (u32 j_ = 0; j_ < (n1); ++j_)                       \
            for (u32 k_ = 0; k_ < 4; ++k_) acc[i_][j_][k_] = 0.0f;

// 32 outputs of tiles s0 .. s0 + TPP - 1 against one k64 step: operands of tile s at unit (s * KS + ks) of a window buffer
#define STEP(acc, qbuf, ks_, slab_, s0_)                                                                                \
    {                                                                                                                   \
        const u8* sl_ = (slab_);                                                                                        \
        const V4 n01 = *(const V4*)(sl_ + lane * 16u), n23 = *(const V4*)(sl_ + 512u + lane * 16u);                     \
        const V4 sw = *(const V4*)(sl_ + 1024u + g * 16u);                                                              \
        _Pragma("unroll")                                                                                               \
        for (u32 d_ = 0; d_ < TPP; ++d_) {                                                                              \
            const u32 u_ = ((s0_) + d_) * KS + (ks_);                                                                   \
            const V4 a_ = *(const V4*)((qbuf) + u_ * 512u + lane * 16u);                                                \
            const u32 sa_ = *(const u32*)((qbuf) + QBYTES + u_ * 64u + g * 8u + (t & 1u) * 4u);                         \
            mma4(acc[d_][0], a_, n01.x, n01.y, sa_, sw.x); mma4(acc[d_][1], a_, n01.z, n01.w, sa_, sw.y);               \
            mma4(acc[d_][2], a_, n23.x, n23.y, sa_, sw.z); mma4(acc[d_][3], a_, n23.z, n23.w, sa_, sw.w);               \
        }                                                                                                               \
    }
// the same against the hidden layer held in registers
#define STEP_HIDDEN(acc, ks_, slab_, s0_)                                                                               \
    {                                                                                                                   \
        const u8* sl_ = (slab_);                                                                                        \
        const V4 n01 = *(const V4*)(sl_ + lane * 16u), n23 = *(const V4*)(sl_ + 512u + lane * 16u);                     \
        const V4 sw = *(const V4*)(sl_ + 1024u + g * 16u);                                                              \
        _Pragma("unroll")                                                                                               \
        for (u32 d_ = 0; d_ < TPP; ++d_) {                                                                              \
            const u32 s_ = (s0_) + d_;                                                                                  \
            const V4 a_ = {MA[ks_][s_][0], MA[ks_][s_][1], MA[ks_][s_][2], MA[ks_][s_][3]};                             \
            mma4(acc[d_][0], a_, n01.x, n01.y, MS[ks_][s_], sw.x); mma4(acc[d_][1], a_, n01.z, n01.w, MS[ks_][s_], sw.y); \
            mma4(acc[d_][2], a_, n23.x, n23.y, MS[ks_][s_], sw.z); mma4(acc[d_][3], a_, n23.z, n23.w, MS[ks_][s_], sw.w); \
        }                                                                                                               \
    }

extern "C" __global__ void nr_swin4(Params p)
{
    __shared__ __align__(16) u8 sm_x[XBYTES];
    __shared__ __align__(16) u8 sm_qa[QBYTES + SBYTES];
    __shared__ __align__(16) u8 sm_qb[QBYTES + SBYTES];
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int x0 = p.sx + 8 * (int)blockIdx.x, y0 = p.sy + 8 * (int)blockIdx.y;
    const int TW = p.W >> 2, TH = p.H >> 2, tx0 = x0 >> 2, ty0 = y0 >> 2;
    const float* const F = (const float*)p.w;
    const u32* const O = (const u32*)(p.w + 64);

    bool inside[4];
    u64 tile_at[4];                                             // byte offset of this lane's 16 bytes of tile s in a tile-layout map
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        inside[s] = ty >= 0 && ty < TH && tx >= 0 && tx < TW;
        tile_at[s] = (u64)(u32)(ty * TW + tx) * (16u * C) + w * 512u + lane * 16u;
    }

    // ---- wait for the producer's windows under this one --------------------------------------------------------
    if (p.pflags) {
        if (leader) {
            const int gx = (p.W - min(p.psx, 0) + 7) >> 3, gy = (p.H - min(p.psy, 0) + 7) >> 3;
            const int dy = y0 - p.psy, dx = x0 - p.psx;
            const int r0 = max(dy, 0) >> 3, r1 = min(dy + 8 > 0 ? ((dy + 7) >> 3) + 1 : 0, gy);
            const int c0 = max(dx, 0) >> 3, c1 = min(dx + 8 > 0 ? ((dx + 7) >> 3) + 1 : 0, gx);
            int budget = WAIT_BUDGET_NS;
            for (int r = r0; r < r1; ++r)
                for (int c = c0; c < c1; ++c) wait_not_negative(p.pflags + r * gx + c, budget);
        }
        __syncthreads();
    }

    // ---- the block's input: values, the residual's copy, the 4-bit operand of the expansion ---------------------
#if VARIANT == 3
    u32 pr[2][4];                                               // projected coarse pixels of this lane, float16 pairs
    {
        const u32 ci = (g & 1u) + 2u * (g >> 2), cxl = (g >> 1) & 1u;
        const int cx = (x0 >> 1) + (int)((ci & 1u) * 2u + cxl);
#pragma unroll
        for (u32 hb = 0; hb < 2; ++hb) {                        // this warp's 64 of the 2C channels: one k64 step
            u32 codes[2];
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                const int cy = (y0 >> 1) + (int)((ci >> 1) * 2u + rh);
                const bool in = cy >= 0 && cy < p.d0 && cx >= 0 && cx < p.d1;
                float v[8];
#pragma unroll
                for (u32 a = 0; a < 2; ++a) {
                    const u64 at = ((u64)(u32)((int)(4u * w + 2u * hb + a) * p.d0 + (in ? cy : 0)) * (u32)p.d1 + (u32)(in ? cx : 0)) * 16u + t * 4u;
                    const u32 word = in ? load_cg32(p.in + at) : 0u;
                    e4m3x4_to_f32(word, v + 4 * a);
                }
#pragma unroll
                for (u32 i = 0; i < 8; ++i) v[i] *= F[F_TR];
                const u32 nib = pack_row(v, codes[rh]);
                *(u32*)(sm_qa + w * 512u + lane * 16u + (hb * 2u + rh) * 4u) = nib;
            }
            put_scale(sm_qa + QBYTES + w * 64u, g, t, hb, codes[0], codes[1]);
        }
        __syncthreads();
        float pacc[1][4][4];
        ZERO_ACC(pacc, 1, 4)
        const u8* wres = p.w + O[O_RES] + (u64)(w * HEADS) * SLAB;
        for (u32 ks = 0; ks < HEADS; ++ks) {
            const u8* sl = wres + ks * SLAB;
            const V4 n01 = *(const V4*)(sl + lane * 16u), n23 = *(const V4*)(sl + 512u + lane * 16u);
            const V4 sw = *(const V4*)(sl + 1024u + g * 16u);
            const V4 a = *(const V4*)(sm_qa + ks * 512u + lane * 16u);
            const u32 sa = *(const u32*)(sm_qa + QBYTES + ks * 64u + g * 8u + (t & 1u) * 4u);
            mma4(pacc[0][0], a, n01.x, n01.y, sa, sw.x); mma4(pacc[0][1], a, n01.z, n01.w, sa, sw.y);
            mma4(pacc[0][2], a, n23.x, n23.y, sa, sw.z); mma4(pacc[0][3], a, n23.z, n23.w, sa, sw.w);
        }
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
            for (u32 j = 0; j < 4; ++j) pr[rh][j] = h2_pack(pacc[0][j][rh * 2u] * F[F_SRES], pacc[0][j][rh * 2u + 1u] * F[F_SRES]);
        __syncthreads();                                        // sm_qa is reused for x below
    }
    const float* const gain = (const float*)(p.w + O[O_GAIN] + (w * 4u + t) * 32u);
    const int STW = p.e1 >> 2, STH = p.e0 >> 2;
#endif
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        u32 codes[2], codes2[2];
        V4 xw;                                                  // the lane's 16 values of this tile as e4m3 codes
#if VARIANT == 3
        const int sty = ty0 + (int)(s >> 1), stx = tx0 + (int)(s & 1u);
        const bool sin = sty >= 0 && sty < STH && stx >= 0 && stx < STW;
        const V4 skw = sin ? load_cg(p.skip + (u64)(u32)(sty * STW + stx) * (16u * C) + w * 512u + lane * 16u) : V4{0u, 0u, 0u, 0u};
        const u32 skwords[4] = {skw.x, skw.y, skw.z, skw.w};
        const u32 src = 4u * ((s & 1u) + 2u * ((g >> 1) & 1u) + 4u * (s >> 1)) + t;
#elif VARIANT == 1
        const int vr = p.d0, vc = p.d1;
#else
        xw = inside[s] ? load_cg(p.in + tile_at[s]) : V4{0u, 0u, 0u, 0u};
#endif
        u32 xwords[4];
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh) {
            float v[8];
#if VARIANT == 3
            float sv[8];
            e4m3x4_to_f32(skwords[rh], sv);
            e4m3x4_to_f32(skwords[2u + rh], sv + 4);
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                float lo, hi;
                h2_unpack(__shfl_sync(0xffffffffu, pr[rh][j], src), lo, hi);
                // float16(skip * gain + projection): one rounding
                float b0, b1;
                h2_unpack(h2_pack(fmaf(sv[2 * j], gain[2 * j], lo), fmaf(sv[2 * j + 1], gain[2 * j + 1], hi)), b0, b1);
                v[2 * j] = inside[s] ? b0 : 0.0f;
                v[2 * j + 1] = inside[s] ? b1 : 0.0f;
            }
            xwords[rh] = f32x4_to_e4m3x4(v[0], v[1], v[2], v[3]);
            xwords[2u + rh] = f32x4_to_e4m3x4(v[4], v[5], v[6], v[7]);
#elif VARIANT == 1
            {
                const u32 m = g + 8u * rh;
                const int py = (ty0 + (int)(s >> 1)) * 4 + (int)(m >> 2), px = (tx0 + (int)(s & 1u)) * 4 + (int)(m & 3u);
                const bool in = py >= 0 && py < vr && px >= 0 && px < vc && inside[s];
#pragma unroll
                for (u32 a = 0; a < 2; ++a) {
                    const u64 at = ((u64)(u32)((int)(2u * w + a) * vr + (in ? py : 0)) * (u32)vc + (u32)(in ? px : 0)) * 16u + t * 4u;
                    xwords[2u * a + rh] = in ? load_cg32(p.in + at) : 0u;
                }
            }
            e4m3x4_to_f32(xwords[rh], v);
            e4m3x4_to_f32(xwords[2u + rh], v + 4);
#else
            xwords[0] = xw.x; xwords[1] = xw.y; xwords[2] = xw.z; xwords[3] = xw.w;
            e4m3x4_to_f32(xwords[rh], v);
            e4m3x4_to_f32(xwords[2u + rh], v + 4);
#endif
#if HEADS == 1
            {   // the residual reads float16 values
                const V4 wide = {h2_pack(v[0], v[1]), h2_pack(v[2], v[3]), h2_pack(v[4], v[5]), h2_pack(v[6], v[7])};
                *(V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u) = wide;
            }
#endif
#pragma unroll
            for (u32 i = 0; i < 8; ++i) v[i] *= F[F_TX];
#if HEADS == 1
            u32 word2;
            const u32 nib = pack_row2(v, codes[rh], word2, codes2[rh], (p.options & OPT_TWO_X) != 0u);
            *(u32*)(sm_qa + s * 512u + lane * 16u + rh * 4u) = nib;
            *(u32*)(sm_qa + s * 512u + lane * 16u + (2u + rh) * 4u) = word2;
#else
            const u32 nib = pack_row(v, codes[rh]);
            *(u32*)(sm_qa + (s * KS + (w >> 1)) * 512u + lane * 16u + ((w & 1u) * 2u + rh) * 4u) = nib;
#endif
        }
#if HEADS == 1
        put_scale(sm_qa + QBYTES + s * 64u, g, t, 0u, codes[0], codes[1]);
        put_scale(sm_qa + QBYTES + s * 64u, g, t, 1u, codes2[0], codes2[1]);
#else
        put_scale(sm_qa + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes[0], codes[1]);
        {
            const V4 keep = {xwords[0], xwords[1], xwords[2], xwords[3]};
            *(V4*)(sm_x + (s * HEADS + w) * 512u + lane * 16u) = keep;
        }
#endif
    }
    __syncthreads();

#if DBG_STAGE != 2
    // ---- MLP: expansion (four passes of 32 hidden units), activation, the hidden layer as a 4-bit operand -------
    u32 MA[2][4][4];                                            // [k64 step of the 128 hidden][tile][word]
    u32 MS[2][4];                                               // the lane's scale word of each
#pragma unroll
    for (u32 i = 0; i < 2; ++i)
#pragma unroll
        for (u32 s = 0; s < 4; ++s) MS[i][s] = 0u;
    {
        const float hc1 = F[F_HC1], hc2 = F[F_HC2], hc3 = F[F_HC3], hlim = F[F_HLIM];
#pragma unroll
        for (u32 pass = 0; pass < 4; ++pass) {
            const u8* wexp = p.w + O[O_EXPAND] + (u64)((w * 4u + pass) * KS) * SLAB;
#pragma unroll
            for (u32 s0 = 0; s0 < 4; s0 += TPP) {
                float acc[TPP][4][4];
                ZERO_ACC(acc, TPP, 4)
#pragma unroll
                for (u32 ks = 0; ks < KS; ++ks) STEP(acc, sm_qa, ks, wexp + ks * SLAB, s0)
#pragma unroll
                for (u32 d = 0; d < TPP; ++d) {
                    const u32 s = s0 + d;
                    u32 codes[2], words[2];
#pragma unroll
                    for (u32 rh = 0; rh < 2; ++rh) {
                        float v[8];
#pragma unroll
                        for (u32 j = 0; j < 4; ++j)
#pragma unroll
                            for (u32 e = 0; e < 2; ++e) {
                                const float x = acc[d][j][rh * 2u + e];
                                const float c = fminf(fmaxf(x, -hlim), hlim);
                                v[2 * j + e] = x * fmaf(c, fmaf(fabsf(c), hc1, hc2), hc3);
                            }
                        words[rh] = pack_row(v, codes[rh]);
                    }
                    const u32 mine = (t & 1u) ? codes[1] : codes[0];
                    const u32 other = __shfl_xor_sync(0xffffffffu, mine, 2);
                    const u32 both = (t & 2u) ? (other | (mine << 8)) : (mine | (other << 8));
                    MA[pass >> 1][s][(pass & 1u) * 2u] = words[0];             // pass = 2 * (k64 step) + half
                    MA[pass >> 1][s][(pass & 1u) * 2u + 1u] = words[1];
                    MS[pass >> 1][s] |= both << (16u * (pass & 1u));
                }
            }
        }
    }

    // ---- the 128 hidden units of the head -> 32: the grouped middle projection (several heads) or the contraction
#if HEADS > 1
    {
        const u8* wmid = p.w + O[O_MID] + (u64)(w * 2u) * SLAB;
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += TPP) {
            float acc[TPP][4][4];
            ZERO_ACC(acc, TPP, 4)
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) STEP_HIDDEN(acc, ks, wmid + ks * SLAB, s0)
#pragma unroll
            for (u32 d = 0; d < TPP; ++d) {
                const u32 s = s0 + d;
                u32 codes[2];
#pragma unroll
                for (u32 rh = 0; rh < 2; ++rh) {
                    float v[8];
#pragma unroll
                    for (u32 j = 0; j < 4; ++j)
#pragma unroll
                        for (u32 e = 0; e < 2; ++e) v[2 * j + e] = acc[d][j][rh * 2u + e] * F[F_SMID];
                    const u32 nib = pack_row(v, codes[rh]);
                    *(u32*)(sm_qb + (s * KS + (w >> 1)) * 512u + lane * 16u + ((w & 1u) * 2u + rh) * 4u) = nib;
                }
                put_scale(sm_qb + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes[0], codes[1]);
            }
        }
    }
    __syncthreads();
#endif
    {
        // y = acc * s + rs * x; stored for the attention's residual and packed for QKV
        const float* const rs = (const float*)(p.w + O[O_RS] + (w * 4u + t) * 32u);
#if HEADS > 1
        const u8* wcon = p.w + O[O_CONTRACT] + (u64)(w * KS) * SLAB;
#else
        const u8* wcon = p.w + O[O_CONTRACT];
#endif
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += TPP) {
            float acc[TPP][4][4];
            ZERO_ACC(acc, TPP, 4)
#if HEADS > 1
#pragma unroll
            for (u32 ks = 0; ks < KS; ++ks) STEP(acc, sm_qb, ks, wcon + ks * SLAB, s0)
#else
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) STEP_HIDDEN(acc, ks, wcon + ks * SLAB, s0)
#endif
#pragma unroll
            for (u32 d = 0; d < TPP; ++d) {
                const u32 s = s0 + d;
                u32 codes[2], codes2[2];
#if HEADS > 1
                const V4 xk = *(const V4*)(sm_x + (s * HEADS + w) * 512u + lane * 16u);
                const u32 xwords[4] = {xk.x, xk.y, xk.z, xk.w};
                u32 ywords[4];
#endif
#pragma unroll
                for (u32 rh = 0; rh < 2; ++rh) {
                    float xv[8], y[8];
#if HEADS > 1
                    e4m3x4_to_f32(xwords[rh], xv);
                    e4m3x4_to_f32(xwords[2u + rh], xv + 4);
#else
                    const V4 wide = *(const V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u);
                    h2_unpack(wide.x, xv[0], xv[1]); h2_unpack(wide.y, xv[2], xv[3]);
                    h2_unpack(wide.z, xv[4], xv[5]); h2_unpack(wide.w, xv[6], xv[7]);
#endif
#pragma unroll
                    for (u32 j = 0; j < 4; ++j)
#pragma unroll
                        for (u32 e = 0; e < 2; ++e) y[2 * j + e] = fmaf(acc[d][j][rh * 2u + e], F[F_SCON], rs[2 * j + e] * xv[2 * j + e]);
#if HEADS > 1
                    ywords[rh] = f32x4_to_e4m3x4(y[0], y[1], y[2], y[3]);
                    ywords[2u + rh] = f32x4_to_e4m3x4(y[4], y[5], y[6], y[7]);
#else
                    {   // one head: the attention's residual is the unrounded (float16) y
                        const V4 wide16 = {h2_pack(y[0], y[1]), h2_pack(y[2], y[3]), h2_pack(y[4], y[5]), h2_pack(y[6], y[7])};
                        *(V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u) = wide16;
                    }
#endif
#pragma unroll
                    for (u32 i = 0; i < 8; ++i) y[i] *= F[F_TY];
#if HEADS == 1
                    u32 word2;
                    const u32 nib = pack_row2(y, codes[rh], word2, codes2[rh], (p.options & OPT_TWO_Y) != 0u);
                    *(u32*)(sm_qa + s * 512u + lane * 16u + rh * 4u) = nib;
                    *(u32*)(sm_qa + s * 512u + lane * 16u + (2u + rh) * 4u) = word2;
#else
                    const u32 nib = pack_row(y, codes[rh]);
                    *(u32*)(sm_qa + (s * KS + (w >> 1)) * 512u + lane * 16u + ((w & 1u) * 2u + rh) * 4u) = nib;
#endif
                }
#if HEADS == 1
                put_scale(sm_qa + QBYTES + s * 64u, g, t, 0u, codes[0], codes[1]);
                put_scale(sm_qa + QBYTES + s * 64u, g, t, 1u, codes2[0], codes2[1]);
#else
                put_scale(sm_qa + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes[0], codes[1]);
                {
                    const V4 keep = {ywords[0], ywords[1], ywords[2], ywords[3]};
                    *(V4*)(sm_x + (s * HEADS + w) * 512u + lane * 16u) = keep;
                }
#endif
            }
        }
    }
    __syncthreads();
#endif  // DBG_STAGE != 2

#if DBG_STAGE == 0
    // ---- attention ---------------------------------------------------------------------------------------------
    // Head dimension d = 8 j + 2 t + e sits at K = 16 (j >> 1) + 4 t + 2 (j & 1) + e of the FP8 instruction, key token
    // (tile, m) at k32 step tile >> 1, K = 16 (tile & 1) + 4 ((m & 7) >> 1) + 2 (m >> 3) + (m & 1): again every lane
    // builds its own operand bytes; only V has to be transposed.
    u32 QA[4][4];                                               // Q, A operand per query tile
    u32 KB[8][2];                                               // K, B operand per 8 key tokens (tile * 2 + half)
    u32 VB[4][2][2];                                            // V, B operand [8 dims][k32 step]
#pragma unroll
    for (u32 j = 0; j < 4; ++j)
#pragma unroll
        for (u32 k = 0; k < 2; ++k) VB[j][k][0] = VB[j][k][1] = 0u;
    {
        const float head_scale = ((const float*)(p.w + O[O_HEADSCALE]))[w];
#pragma unroll
        for (u32 which = 0; which < 3; ++which) {
            const u8* wq = p.w + O[O_QKV] + (u64)((w * 3u + which) * KS) * SLAB;
#pragma unroll
            for (u32 s0 = 0; s0 < 4; s0 += TPP) {
                float acc[TPP][4][4];
                ZERO_ACC(acc, TPP, 4)
#pragma unroll
                for (u32 ks = 0; ks < KS; ++ks) STEP(acc, sm_qa, ks, wq + ks * SLAB, s0)
#pragma unroll
                for (u32 d = 0; d < TPP; ++d)
#pragma unroll
                    for (u32 rh = 0; rh < 2; ++rh) {
                        const u32 s = s0 + d;
                        float v[8];
                        float sum = 0.0f;
#pragma unroll
                        for (u32 j = 0; j < 4; ++j)
#pragma unroll
                            for (u32 e = 0; e < 2; ++e) {
                                v[2 * j + e] = acc[d][j][rh * 2u + e] * F[F_SQKV];
                                sum = fmaf(v[2 * j + e], v[2 * j + e], sum);
                            }
                        if (which < 2u) {
                            // the squared norm over the head's 32 dimensions: the other 24 are in the three lanes next to this one
                            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
                            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
                            // the runtime sums in float16: a sum beyond its range gives a zero scale
                            float scale = sum >= 65520.0f ? 0.0f : rsqrt_approx(fmaxf(sum, 6.19888305664e-05f));
                            scale = which == 0u ? scale * head_scale : scale;
                            const u32 lo = f32x4_to_e4m3x4(v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale);
                            const u32 hi = f32x4_to_e4m3x4(v[4] * scale, v[5] * scale, v[6] * scale, v[7] * scale);
                            if (which == 0u) { QA[s][rh] = lo; QA[s][2u + rh] = hi; }
                            else { KB[s * 2u + rh][0] = lo; KB[s * 2u + rh][1] = hi; }
                        } else {
                            // V: float16, transposed (8 tokens x 8 dims -> a lane holds two tokens of one dim), e4m3
                            const u32 group = s * 2u + rh;
#pragma unroll
                            for (u32 j = 0; j < 4; ++j) {
                                const u32 bytes = h2_to_e4m3x2(transpose8(h2_pack(v[2 * j], v[2 * j + 1])));
                                VB[j][group >> 2][(group >> 1) & 1u] |= bytes << (16u * (group & 1u));
                            }
                        }
                    }
            }
        }
    }
    {
        const u8* const bias = p.w + O[O_BIAS] + (u64)w * 8192u + lane * 16u;
#pragma unroll
        for (u32 qs = 0; qs < 4; ++qs) {
            u32 lacc[8][2];
#pragma unroll
            for (u32 kt = 0; kt < 4; ++kt) {
                const V4 b = *(const V4*)(bias + (qs * 4u + kt) * 512u);
                lacc[kt * 2u][0] = b.x; lacc[kt * 2u][1] = b.y; lacc[kt * 2u + 1u][0] = b.z; lacc[kt * 2u + 1u][1] = b.w;
            }
#pragma unroll
            for (u32 nt = 0; nt < 8; ++nt) mma8(lacc[nt], QA[qs], KB[nt][0], KB[nt][1]);
            // exp as the runtime builds it: an affine ramp, clamped, whose mantissa becomes exponent and mantissa
            u32 total[2];
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
#pragma unroll
                for (u32 nt = 0; nt < 8; ++nt) {
                    u32 x = h2_fma(lacc[nt][r], 0x29C029C0u, 0x3D343D34u);
                    x = h2_min(h2_max(x, 0x3C203C20u), 0x3E473E47u);
                    lacc[nt][r] = (x << 5) + 0x7FF88000u;
                }
                u32 sum = h2_add(h2_add(h2_add(h2_add(lacc[0][r], lacc[1][r]), h2_add(lacc[2][r], lacc[3][r])), h2_add(lacc[4][r], lacc[5][r])),
                                 h2_add(lacc[6][r], lacc[7][r]));
                sum = h2_add(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
                sum = h2_add(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
                float lo, hi;
                h2_unpack(sum, lo, hi);
                const float reciprocal = rcp_approx(fmaxf(lo + hi, 6.19888305664e-05f));
                total[r] = h2_pack(reciprocal, reciprocal);
            }
            u32 PA[2][4];
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    PA[ks][r] = h2_to_e4m3x2(h2_mul(lacc[4 * ks][r], total[r])) | (h2_to_e4m3x2(h2_mul(lacc[4 * ks + 1][r], total[r])) << 16);
                    PA[ks][2u + r] = h2_to_e4m3x2(h2_mul(lacc[4 * ks + 2][r], total[r])) | (h2_to_e4m3x2(h2_mul(lacc[4 * ks + 3][r], total[r])) << 16);
                }
            u32 cacc[4][2];
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                cacc[j][0] = cacc[j][1] = 0u;
#pragma unroll
                for (u32 ks = 0; ks < 2; ++ks) mma8(cacc[j], PA[ks], VB[j][ks][0], VB[j][ks][1]);
            }
            // the context of tile qs as the 4-bit operand of the output projection
            u32 codes[2], codes2[2];
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                float v[8];
#pragma unroll
                for (u32 j = 0; j < 4; ++j) {
                    h2_unpack(cacc[j][rh], v[2 * j], v[2 * j + 1]);
                    v[2 * j] *= F[F_TC]; v[2 * j + 1] *= F[F_TC];
                }
#if HEADS == 1
                u32 word2;
                const u32 nib = pack_row2(v, codes[rh], word2, codes2[rh], (p.options & OPT_TWO_C) != 0u);
                *(u32*)(sm_qb + qs * 512u + lane * 16u + rh * 4u) = nib;
                *(u32*)(sm_qb + qs * 512u + lane * 16u + (2u + rh) * 4u) = word2;
#else
                const u32 nib = pack_row(v, codes[rh]);
                *(u32*)(sm_qb + (qs * KS + (w >> 1)) * 512u + lane * 16u + ((w & 1u) * 2u + rh) * 4u) = nib;
#endif
            }
#if HEADS == 1
            put_scale(sm_qb + QBYTES + qs * 64u, g, t, 0u, codes[0], codes[1]);
            put_scale(sm_qb + QBYTES + qs * 64u, g, t, 1u, codes2[0], codes2[1]);
#else
            put_scale(sm_qb + QBYTES + (qs * KS + (w >> 1)) * 64u, g, t, w & 1u, codes[0], codes[1]);
#endif
        }
    }
    __syncthreads();
#endif  // DBG_STAGE == 0

    // ---- output projection, residual, stores ------------------------------------------------------------------
    {
#if VARIANT == 2
        u32 o16[4][2][4];                                       // the unrounded output as float16 pairs, for the pooling
#endif
#if DBG_STAGE == 0
        const u8* wproj = p.w + O[O_PROJ] + (u64)(w * KS) * SLAB;
        const float* const ars = (const float*)(p.w + O[O_ARS] + (w * 4u + t) * 32u);
#endif
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += TPP) {
#if DBG_STAGE == 0
            float acc[TPP][4][4];
            ZERO_ACC(acc, TPP, 4)
#pragma unroll
            for (u32 ks = 0; ks < KS; ++ks) STEP(acc, sm_qb, ks, wproj + ks * SLAB, s0)
#endif
#pragma unroll
            for (u32 d = 0; d < TPP; ++d) {
                const u32 s = s0 + d;
                u32 owords[4];
#if HEADS > 1
                const V4 yk = *(const V4*)(sm_x + (s * HEADS + w) * 512u + lane * 16u);
                const u32 ywords[4] = {yk.x, yk.y, yk.z, yk.w};
#endif
#pragma unroll
                for (u32 rh = 0; rh < 2; ++rh) {
                    float yv[8], o[8];
#if HEADS > 1
                    e4m3x4_to_f32(ywords[rh], yv);
                    e4m3x4_to_f32(ywords[2u + rh], yv + 4);
#else
                    const V4 wide = *(const V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u);
                    h2_unpack(wide.x, yv[0], yv[1]); h2_unpack(wide.y, yv[2], yv[3]);
                    h2_unpack(wide.z, yv[4], yv[5]); h2_unpack(wide.w, yv[6], yv[7]);
#endif
#pragma unroll
                    for (u32 i = 0; i < 8; ++i) {
#if DBG_STAGE == 0
                        o[i] = fmaf(acc[d][i >> 1][rh * 2u + (i & 1u)], F[F_SPROJ], ars[i] * yv[i]);
#else
                        o[i] = yv[i];
#endif
                    }
                    owords[rh] = f32x4_to_e4m3x4(o[0], o[1], o[2], o[3]);
                    owords[2u + rh] = f32x4_to_e4m3x4(o[4], o[5], o[6], o[7]);
#if VARIANT == 2
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) o16[s][rh][j] = h2_pack(o[2 * j], o[2 * j + 1]);
#endif
#if VARIANT == 4
                    {
                        const u32 m = g + 8u * rh;
                        const int py = (ty0 + (int)(s >> 1)) * 4 + (int)(m >> 2), px = (tx0 + (int)(s & 1u)) * 4 + (int)(m & 3u);
                        if (inside[s] && py >= 0 && py < p.d0 && px >= 0 && px < p.d1) {
#pragma unroll
                            for (u32 a = 0; a < 2; ++a)
                                *(u32*)(p.out + ((u64)(u32)((int)(2u * w + a) * p.d0 + py) * (u32)p.d1 + (u32)px) * 16u + t * 4u) = owords[2u * a + rh];
                        }
                    }
#endif
                }
#if VARIANT != 4
                if (inside[s]) {
                    const V4 packed = {owords[0], owords[1], owords[2], owords[3]};
                    *(V4*)(p.out + tile_at[s]) = packed;
                }
#endif
            }
        }
#if VARIANT == 2 && DBG_STAGE == 0
        // ---- 2x2 mean of the unrounded output, then the C -> 2C projection of the 16 pooled tokens of the window ----
        // Lane group g holds pooled tokens (tile ci, row half, column (g >> 1) & 1) as rows g and g + 8 of one M tile.
        const u32 ci = (g & 1u) + 2u * (g >> 2);
        u32 pool[2][4];
#pragma unroll
        for (u32 s = 0; s < 4; ++s)
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
                for (u32 j = 0; j < 4; ++j) {
                    u32 a = o16[s][rh][j];
                    a = h2_add(a, __shfl_xor_sync(0xffffffffu, a, 4));
                    a = h2_add(a, __shfl_xor_sync(0xffffffffu, a, 16));
                    a = h2_mul(a, 0x34003400u);                 // * 0.25
                    pool[rh][j] = (s == ci || s == 0u) ? a : pool[rh][j];
                }
        {
            u32 codes[2], codes2[2];
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                float v[8];
#pragma unroll
                for (u32 j = 0; j < 4; ++j) {
                    h2_unpack(pool[rh][j], v[2 * j], v[2 * j + 1]);
                    v[2 * j] *= F[F_TR]; v[2 * j + 1] *= F[F_TR];
                }
#if HEADS == 1
                u32 word2;
                const u32 nib = pack_row2(v, codes[rh], word2, codes2[rh], (p.options & OPT_TWO_R) != 0u);
                *(u32*)(sm_qa + lane * 16u + rh * 4u) = nib;
                *(u32*)(sm_qa + lane * 16u + (2u + rh) * 4u) = word2;
#else
                const u32 nib = pack_row(v, codes[rh]);
                *(u32*)(sm_qa + (w >> 1) * 512u + lane * 16u + ((w & 1u) * 2u + rh) * 4u) = nib;
#endif
            }
#if HEADS == 1
            put_scale(sm_qa + QBYTES, g, t, 0u, codes[0], codes[1]);
            put_scale(sm_qa + QBYTES, g, t, 1u, codes2[0], codes2[1]);
#else
            put_scale(sm_qa + QBYTES + (w >> 1) * 64u, g, t, w & 1u, codes[0], codes[1]);
#endif
        }
        __syncthreads();
        float racc[2][4][4];
        ZERO_ACC(racc, 2, 4)
        const u8* wres = p.w + O[O_RES] + (u64)(w * 2u * KS) * SLAB;
#pragma unroll
        for (u32 ks = 0; ks < KS; ++ks) {
            const V4 a = *(const V4*)(sm_qa + ks * 512u + lane * 16u);
            const u32 sa = *(const u32*)(sm_qa + QBYTES + ks * 64u + g * 8u + (t & 1u) * 4u);
#pragma unroll
            for (u32 half = 0; half < 2; ++half) {
                const u8* sl = wres + (half * KS + ks) * SLAB;
                const V4 n01 = *(const V4*)(sl + lane * 16u), n23 = *(const V4*)(sl + 512u + lane * 16u);
                const V4 sw = *(const V4*)(sl + 1024u + g * 16u);
                mma4(racc[half][0], a, n01.x, n01.y, sa, sw.x); mma4(racc[half][1], a, n01.z, n01.w, sa, sw.y);
                mma4(racc[half][2], a, n23.x, n23.y, sa, sw.z); mma4(racc[half][3], a, n23.z, n23.w, sa, sw.w);
            }
        }
        {
            const int hH = (p.H + 1) >> 1, hW = (p.W + 1) >> 1, vr = (hH + 3) & ~3, vc = (hW + 3) & ~3;
            const int px = (x0 >> 1) + (int)((ci & 1u) * 2u + ((g >> 1) & 1u));
            const float sres = F[F_SRES];
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                const int py = (y0 >> 1) + (int)((ci >> 1) * 2u + rh);
                if (py >= 0 && py < hH && px >= 0 && px < hW) {
#pragma unroll
                    for (u32 k = 0; k < 4; ++k) {
                        const float* lo = racc[k >> 1][(k & 1u) * 2u];
                        const float* hi = racc[k >> 1][(k & 1u) * 2u + 1u];
                        const u32 word = f32x4_to_e4m3x4(lo[rh * 2u] * sres, lo[rh * 2u + 1u] * sres, hi[rh * 2u] * sres, hi[rh * 2u + 1u] * sres);
                        *(u32*)(p.ds_out + ((u64)(u32)((int)(4u * w + k) * vr + py) * (u32)vc + (u32)px) * 16u + t * 4u) = word;
                    }
                }
            }
        }
#endif
    }

    // ---- tell the next layer this window is in memory ----------------------------------------------------------
    if (p.oflags) {
        __syncthreads();
        if (leader) flag_publish(p.oflags + blockIdx.y * gridDim.x + blockIdx.x, 0);
    }
}
