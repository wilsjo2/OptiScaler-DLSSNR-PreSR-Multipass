// NVFP4 (4-bit) replacement of the DLSS-NR fused Swin blocks: cc_tinlayout_fused_swin_{1h_32,2h_64,4h_128,8h_256}_*.
// Target sm_120a, built with NVRTC (--fmad=false), one cubin per (HEADS, VARIANT):
//   -DHEADS=1|2|4|8      channels C = 32 * HEADS
//   -DVARIANT=0          tile-layout input, tile-layout output       (the runtime's plain / tilesync / wait / chained)
//            =1          view-layout input                           (inpview*)
//            =2          ... plus 2x2 mean and C -> 2C projection    (ds*)
//            =3          2C -> C projection of the coarser level, 2x replication, blend with the skip   (upsample*)
//            =4          view-layout output                          (outview*)
// Register limits (--maxrregcount): 128 at width 256 (two blocks share an SM; the order of the work is chosen for it,
// see REG_MLP), 168 at width 128 (three blocks), none at widths 64 and 32. swin4_first.cu is the first build: the same
// block computed value by value in float32, about a fifth more instructions, one width-256 block per SM.
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
// Order of the work in the MLP (no arithmetic depends on it). REG_MLP 1: MT tiles at a time go through expansion,
// activation and the middle projection, so far fewer values are held at once: the width-256 block then fits 128
// registers and two blocks share an SM (measured on the RTX 5090, one block alone: 63 us against 85 us with one block
// per SM; the runtime's own block 78 us). MT 1 is much slower (95 us: too few independent accumulators per warp).
// The other widths are fastest in the plain order (width 128 at 168 registers: 60 us against 65-67 us at 128).
#ifndef REG_MLP
#define REG_MLP (HEADS == 8)
#endif
#ifndef MT
#define MT 2u
#endif
#ifndef TPP
#define TPP 4u
#endif
// ---- swin4_once.cu: swin4.cu plus switches that change when and how often weights are read (all off: the code of
// swin4.cu, byte for byte: coordinator/once_build.py hash) -------------------------------------------------------
// What they are for: a width-256 window reads 710 KB of weights and tables from L2 (measured ceiling: about 43 GB/s
// per SM): 602 KB, of which 147 KB are the second read of the expansion slabs (once per tile pair). Neither switch
// changes a product, an accumulation order or a rounding. (Q's slabs are already read once: the compiler merges the
// four per-tile reads.)
// EXP_HALF 1 (REG_MLP builds): the expansion over all four tiles of the window at once, two of a pass's four 8-output
//           fragments at a time (the accumulators stay 32 registers), so every expansion slab is read once per window.
//           The first half's activations of a tile (4 words per lane) wait in shared memory for the second half
//           (tiles 0, 1 in sm_qb, which is idle until the middle projection, tiles 2, 3 in sm_h); the tile's operand
//           words are re-read from shared memory for every half pass.
// WINS n:   n windows per CTA (block (32, HEADS * n), grid ceil(windows per row / n) x rows): the n windows run the
//           same code in step, so head h of each reads the same slabs at about the same time. Whether the second
//           read is then served from L1 is what v2/bench_share.py measures. Flags stay one per window.
// ONCE_WIDE 1: for an export of all four widths with one option: PIPE at widths 128 and 256, with EXP_HALF at 256;
//           widths 32 and 64 stay the release code.
#if defined(ONCE_WIDE) && ONCE_WIDE && HEADS >= 4
#define PIPE 1
#if HEADS == 8
#define EXP_HALF 1
#endif
#endif
// ONCE_W2H 1: likewise, the build measured fastest in the chain (RTX 5090, 4K): at width 256 two windows per block (WINS 2)
//           with EXP_HALF; the other widths stay the release code. The launch of a width-256 block is then block
//           (32, 16), grid (ceil(windows per row / 2), rows): `windows 8 2` in swin4.txt tells NrSwin4.
#if defined(ONCE_W2H) && ONCE_W2H && HEADS == 8
#define EXP_HALF 1
#define WINS 2
#endif
#ifndef EXP_HALF
#define EXP_HALF 0
#endif
#ifndef WINS
#define WINS 1
#endif
// WIN_BAR 1 (with WINS > 1): once the block's flags are in, its windows no longer wait for one another: every barrier
//           after the one behind the wait is a barrier of the window's own HEADS warps (barrier number win + 1).
#ifndef WIN_BAR
#define WIN_BAR 0
#endif
#if WINS > 1 && WIN_BAR
#define NR_BARRIER() asm volatile("bar.sync %0, %1;" :: "r"(win + 1u), "r"(32u * HEADS) : "memory")
#else
#define NR_BARRIER() __syncthreads()
#endif
// AHEAD n:  every weight slab is asked for n k64 steps before its operands are loaded (prefetch.global.L1 of the bytes
//           the step will load), and the attention bias one query tile early, so the fetch from L2 runs under the
//           matrix instructions of the steps before it instead of stalling the warp at the load. Why: the measured
//           block times are the sum of the instruction time and the time the bytes take (one alone 85 us, two per SM
//           63 us), not the larger of the two -- today the tensor cores idle while a step's weights arrive. Nothing
//           else changes. Widths 128 and 256 in their default orders (TPP 4, ATT2_KVT 4).
//           AHEAD_UNIT: bytes a lane asks for per prefetch instruction (16: its own operand bytes; 32, 64, 128: fewer
//           instructions when one touch brings a whole unit of that size).
#ifndef AHEAD
#define AHEAD 0
#endif
#ifndef AHEAD_UNIT
#define AHEAD_UNIT 32u
#endif
// PIPE 1:   the same aim as AHEAD with the real loads (ptxas turns prefetch.global.L1 into a 4-byte load into a scratch
//           register, which may hold the warp when that register is written next): the operand words of every k64 step,
//           and the attention bias of a query tile, are loaded one step before the step that multiplies them and wait
//           in registers (12 per slab, 6 with EXP_HALF), as the runtime's own split-512 kernels do. No byte more than
//           the release and 130 to 180 instructions per thread for the addresses. ptxas may still move a load: at
//           the 128-register limit of width 256 it keeps the nibble loads a step ahead and sinks the scale words to
//           mid-step (coordinator/kseq.py prints the order), and without EXP_HALF it spills (130 stores per thread),
//           so at width 256 use it with EXP_HALF. Not combined with AHEAD.
#ifndef PIPE
#define PIPE 0
#endif
// ONE_SLAB 1 (timing only, the output is wrong): every k64 step of every head multiplies the layer's first expansion
//           slab and every query tile adds head 0's first bias tile (3.2 KB in all), so after the first step no weight
//           comes from L2. Every load and every matrix instruction of the release is still there: the addresses are
//           folded onto that slab with a factor the compiler cannot know is zero (ONE_AT). The time against the release
//           is what the block spends waiting for weight bytes: the most any of the switches above can win.
#ifndef ONE_SLAB
#define ONE_SLAB 0
#endif
// STAGE 1:  the aim of PIPE with a lead the compiler cannot take back (ptxas sinks an early register load to its first
//           use, kernels/lead_bench.cu): the operand words of a k64 step are copied into shared memory two steps before
//           the step multiplies them (cp.async, a ring of three one-slab slots per warp) and the step reads them
//           there. The copy is asynchronous, so the L2 round trip (about 0.29 us) runs under two steps of matrix
//           instructions. Same bytes, same products, same order as the release. Width 128 (3456 B of shared memory per
//           warp: 32256 B per block, still three blocks per SM). The attention bias is read as in the release.
//           Measured (RTX 5090, 4K, layers 10 and 11): byte-identical output, 4 to 5 us per block SLOWER than the
//           release (66.8 against 61.9 us). What a block pays for its weights is a cost per line fetched from L2
//           (about 0.5 ns of SM time per 64 bytes), not latency a schedule could hide: only fewer fetched bytes help
//           (EXP_HALF, WINS). Kept as a record; not for an export.
#ifndef STAGE
#define STAGE 0
#endif
#define C (32u * HEADS)
#define KS ((HEADS + 1u) / 2u)                      // k64 steps over C channels (C = 32: one, half used)
#define SLAB 1152u                                  // 32 outputs x one k64 step: 2 x 512 B of nibbles, 128 B of scales
#define QBYTES (4u * KS * 512u)
#define SBYTES (4u * KS * 64u)
#define HBYTES (2u * HEADS * 512u)                  // sm_h: 16 bytes per lane and head, two tiles
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

// shared memory the compiler may neither merge nor forward (a value parked there must leave its registers)
NR_INLINE V4 once_lds128(const u8* q)
{
    V4 v;
    asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"((u32)__cvta_generic_to_shared(q)) : "memory");
    return v;
}
NR_INLINE u32 once_lds32(const u8* q)
{
    u32 v;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"((u32)__cvta_generic_to_shared(q)) : "memory");
    return v;
}
NR_INLINE void once_sts128(u8* q, u32 a, u32 b, u32 c, u32 d)
{
    asm volatile("st.volatile.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"((u32)__cvta_generic_to_shared(q)), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
// a weight operand load the compiler may not merge with an identical one (keeps re-read slabs out of registers)
NR_INLINE V4 wload(const u8* p)
{
#if REG_WLOAD
    V4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"((u64)p));
    return v;
#else
    return *(const V4*)p;
#endif
}
#if REG_MMA_NV
#define NR_MMA_ASM asm
#else
#define NR_MMA_ASM asm volatile
#endif
NR_INLINE void mma4(float* c, const V4& a, u32 b0, u32 b1, u32 sfa, u32 sfb)
{
    float d0, d1, d2, d3;
    const u16 z = 0;
    NR_MMA_ASM("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
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
    NR_MMA_ASM("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
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

// ---- the MLP half's packing on float16 pairs ----------------------------------------------------------------------
// Values are held as float16 pairs (low half first). A row is the lane's 8 values of one token (4 words); rows are
// packed two at a time (the tokens g and g + 8 of a tile), so the block maximum takes one shuffle and the two scale
// codes one conversion. The 4-bit operand: u = value * float16(pre / scale) as an exact float32 product (the mixed
// fma.rn.f32.f16), then cvt.rn.satfinite.e2m1x2.f32 as before.
NR_INLINE u32 m_xamax(u32 a, u32 b) { u32 d; asm("max.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 m_clamp(u32 x, u32 lim) { u32 d; asm("min.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(d) : "r"(x), "r"(lim)); return d; }
NR_INLINE u32 m_abs(u32 a) { u32 d; asm("abs.f16x2 %0, %1;" : "=r"(d) : "r"(a)); return d; }
NR_INLINE u32 m_mul(u32 a, u32 b) { u32 d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 m_prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
NR_INLINE float m_xamaxf(float a, float b) { float d; asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
NR_INLINE void m_split(u32 w, u16& lo, u16& hi) { asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(w)); }
// float32 maximum of |v| over 8 values (sign of the result undefined)
NR_INLINE float m_amax8x(const float* v)
{
    return m_xamaxf(m_xamaxf(m_xamaxf(v[0], v[1]), m_xamaxf(v[2], v[3])), m_xamaxf(m_xamaxf(v[4], v[5]), m_xamaxf(v[6], v[7])));
}
// four e4m3 codes -> two float16 pairs (exact)
NR_INLINE void m_e4m3x4_to_h2(u32 word, u32& a, u32& b)
{
    asm("{\n .reg .b16 nr_l, nr_u;\n mov.b32 {nr_l, nr_u}, %2;\n cvt.rn.f16x2.e4m3x2 %0, nr_l;\n cvt.rn.f16x2.e4m3x2 %1, nr_u;\n}"
        : "=r"(a), "=r"(b) : "r"(word));
}
// two float16 pairs -> four e4m3 codes (as f32x4_to_e4m3x4 of the same values)
NR_INLINE u32 m_h2x2_to_e4m3x4(u32 a, u32 b)
{
    u32 word;
    asm("{\n .reg .b16 nr_l, nr_u;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_l, %1;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_u, %2;\n mov.b32 %0, {nr_l, nr_u};\n}"
        : "=r"(word) : "r"(a), "r"(b));
    return word;
}
// eight float16 values (four pairs) times r (float16) -> eight e2m1 nibbles; the products are exact in float32
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
#if HEADS == 1
// the same, and what each value leaves over: d = u - q (float32, q the e2m1 value the nibble stands for)
NR_INLINE u32 m_e2m1x8_h_rest(const u32* h, u16 r, u16 minus_one, float* d)
{
    u32 word;
    asm("{\n .reg .b16 nr_a<8>, nr_q<8>;\n .reg .f32 nr_f<8>;\n .reg .b8 nr_n<4>;\n .reg .b32 nr_p<4>;\n"
        " mov.b32 {nr_a0, nr_a1}, %9;\n mov.b32 {nr_a2, nr_a3}, %10;\n mov.b32 {nr_a4, nr_a5}, %11;\n mov.b32 {nr_a6, nr_a7}, %12;\n"
        " fma.rn.f32.f16 nr_f0, nr_a0, %13, 0f80000000;\n fma.rn.f32.f16 nr_f1, nr_a1, %13, 0f80000000;\n"
        " fma.rn.f32.f16 nr_f2, nr_a2, %13, 0f80000000;\n fma.rn.f32.f16 nr_f3, nr_a3, %13, 0f80000000;\n"
        " fma.rn.f32.f16 nr_f4, nr_a4, %13, 0f80000000;\n fma.rn.f32.f16 nr_f5, nr_a5, %13, 0f80000000;\n"
        " fma.rn.f32.f16 nr_f6, nr_a6, %13, 0f80000000;\n fma.rn.f32.f16 nr_f7, nr_a7, %13, 0f80000000;\n"
        " cvt.rn.satfinite.e2m1x2.f32 nr_n0, nr_f1, nr_f0;\n cvt.rn.satfinite.e2m1x2.f32 nr_n1, nr_f3, nr_f2;\n"
        " cvt.rn.satfinite.e2m1x2.f32 nr_n2, nr_f5, nr_f4;\n cvt.rn.satfinite.e2m1x2.f32 nr_n3, nr_f7, nr_f6;\n"
        " mov.b32 %0, {nr_n0, nr_n1, nr_n2, nr_n3};\n"
        " cvt.rn.f16x2.e2m1x2 nr_p0, nr_n0;\n cvt.rn.f16x2.e2m1x2 nr_p1, nr_n1;\n"
        " cvt.rn.f16x2.e2m1x2 nr_p2, nr_n2;\n cvt.rn.f16x2.e2m1x2 nr_p3, nr_n3;\n"
        " mov.b32 {nr_q0, nr_q1}, nr_p0;\n mov.b32 {nr_q2, nr_q3}, nr_p1;\n mov.b32 {nr_q4, nr_q5}, nr_p2;\n mov.b32 {nr_q6, nr_q7}, nr_p3;\n"
        " fma.rn.f32.f16 %1, nr_q0, %14, nr_f0;\n fma.rn.f32.f16 %2, nr_q1, %14, nr_f1;\n"
        " fma.rn.f32.f16 %3, nr_q2, %14, nr_f2;\n fma.rn.f32.f16 %4, nr_q3, %14, nr_f3;\n"
        " fma.rn.f32.f16 %5, nr_q4, %14, nr_f4;\n fma.rn.f32.f16 %6, nr_q5, %14, nr_f5;\n"
        " fma.rn.f32.f16 %7, nr_q6, %14, nr_f6;\n fma.rn.f32.f16 %8, nr_q7, %14, nr_f7;\n}"
        : "=r"(word), "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(h[0]), "r"(h[1]), "r"(h[2]), "r"(h[3]), "h"(r), "h"(minus_one));
    return word;
}
#endif
// Scale codes of two rows from their maxima (float32, times the tensor scale): code pair (byte 0: row 0), the values
// the codes stand for and their reciprocals. Same codes as block_scale (satfinite does its clamp at 448).
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
// maxima of |value| of two rows of float16 pairs over the block of 16 (this lane and lane ^ 1), as float32
NR_INLINE void m_amax2_h(const u32* h0, const u32* h1, float& a0, float& a1)
{
    const u32 m0 = m_xamax(m_xamax(h0[0], h0[1]), m_xamax(h0[2], h0[3]));
    const u32 m1 = m_xamax(m_xamax(h1[0], h1[1]), m_xamax(h1[2], h1[3]));
    u32 mm = m_xamax(m_prmt(m0, m1, 0x5410u), m_prmt(m0, m1, 0x7632u));         // (row 0, row 1), signs undefined
    mm = m_xamax(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    h2_unpack(m_abs(mm), a0, a1);
}
// Two rows of float16 values -> two words of nibbles; returns the scale code pair. `pre`: the tensor scale the values
// are packed at (the operand stands for value * pre).
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
#if HEADS == 1
// The same with the second 4-bit term (pack_row2's arithmetic from u on): word2 / codes2 zero when `two` is false.
NR_INLINE u32 m_pack2_h_two(const u32* h0, const u32* h1, float pre, u32& w0, u32& w1, u32& x0, u32& x1, u32& codes2, bool two)
{
    float a0, a1, s0, s1, r0, r1;
    m_amax2_h(h0, h1, a0, a1);
    const u32 codes = m_scale2(a0 * pre, a1 * pre, s0, s1, r0, r1);
    u16 q0, q1;
    m_split(h2_pack(r0 * pre, r1 * pre), q0, q1);
    x0 = x1 = codes2 = 0u;
#ifdef MLP_FORCE_ONE_TERM                                       // measurement builds only: the executed path with the option off
    two = false;
#endif
    if (!two) {                                                 // the option is the same for the whole grid
        w0 = m_e2m1x8_h(h0, q0);
        w1 = m_e2m1x8_h(h1, q1);
        return codes;
    }
    float d0[8], d1[8];
    w0 = m_e2m1x8_h_rest(h0, q0, (u16)0xBC00u, d0);
    w1 = m_e2m1x8_h_rest(h1, q1, (u16)0xBC00u, d1);
#pragma unroll
    for (u32 i = 0; i < 8; ++i) { d0[i] *= s0; d1[i] *= s1; }
    float b0 = m_amax8x(d0), b1 = m_amax8x(d1);
    b0 = fabsf(m_xamaxf(b0, __shfl_xor_sync(0xffffffffu, b0, 1)));
    b1 = fabsf(m_xamaxf(b1, __shfl_xor_sync(0xffffffffu, b1, 1)));
    float t0, t1, p0, p1;
    codes2 = m_scale2(b0, b1, t0, t1, p0, p1);
#pragma unroll
    for (u32 i = 0; i < 8; ++i) { d0[i] *= p0; d1[i] *= p1; }
    x0 = e2m1x8(d0);
    x1 = e2m1x8(d1);
    return codes;
}
#endif
// x (four float16 pairs) times r (eight float16 values): exact float32 products
NR_INLINE void m_mix8(const u32* x, const u16* r, float* d)
{
    asm("{\n .reg .b16 nr_a<8>;\n"
        " mov.b32 {nr_a0, nr_a1}, %8;\n mov.b32 {nr_a2, nr_a3}, %9;\n mov.b32 {nr_a4, nr_a5}, %10;\n mov.b32 {nr_a6, nr_a7}, %11;\n"
        " fma.rn.f32.f16 %0, nr_a0, %12, 0f80000000;\n fma.rn.f32.f16 %1, nr_a1, %13, 0f80000000;\n"
        " fma.rn.f32.f16 %2, nr_a2, %14, 0f80000000;\n fma.rn.f32.f16 %3, nr_a3, %15, 0f80000000;\n"
        " fma.rn.f32.f16 %4, nr_a4, %16, 0f80000000;\n fma.rn.f32.f16 %5, nr_a5, %17, 0f80000000;\n"
        " fma.rn.f32.f16 %6, nr_a6, %18, 0f80000000;\n fma.rn.f32.f16 %7, nr_a7, %19, 0f80000000;\n}"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "h"(r[0]), "h"(r[1]), "h"(r[2]), "h"(r[3]), "h"(r[4]), "h"(r[5]), "h"(r[6]), "h"(r[7]));
}
// put_scale from a code pair (byte 0: token g, byte 1: token g + 8)
NR_INLINE void m_put_scale(u8* words, u32 g, u32 t, u32 khalf, u32 pair)
{
    words[g * 8u + (t & 1u) * 4u + khalf * 2u + (t >> 1)] = (u8)(pair >> ((t & 1u) * 8u));
}

#define ZERO_ACC(acc, n0, n1)                                   \
    for (u32 i_ = 0; i_ < (n0); ++i_)                           \
        for (u32 j_ = 0; j_ < (n1); ++j_)                       \
            for (u32 k_ = 0; k_ < 4; ++k_) acc[i_][j_][k_] = 0.0f;

// 32 outputs of tiles s0 .. s0 + TPP - 1 against one k64 step: operands of tile s at unit (s * KS + ks) of a window buffer
#define STEP(acc, qbuf, ks_, slab_, s0_)                                                                                \
    {                                                                                                                   \
        const u8* sl_ = (slab_);                                                                                        \
        STEP_OPS(sl_)                                                                                                   \
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
        STEP_OPS(sl_)                                                                                                   \
        _Pragma("unroll")                                                                                               \
        for (u32 d_ = 0; d_ < TPP; ++d_) {                                                                              \
            const u32 s_ = (s0_) + d_;                                                                                  \
            const V4 a_ = {MA[ks_][s_][0], MA[ks_][s_][1], MA[ks_][s_][2], MA[ks_][s_][3]};                             \
            mma4(acc[d_][0], a_, n01.x, n01.y, MS[ks_][s_], sw.x); mma4(acc[d_][1], a_, n01.z, n01.w, MS[ks_][s_], sw.y); \
            mma4(acc[d_][2], a_, n23.x, n23.y, MS[ks_][s_], sw.z); mma4(acc[d_][3], a_, n23.z, n23.w, MS[ks_][s_], sw.w); \
        }                                                                                                               \
    }

// ---- round 2 of the attention half (att2_*) ----------------------------------------------------------------------
// The tensor-core and movmatrix instructions without `volatile`: their results are used, so nothing is dropped, and
// ptxas may schedule them freely (the volatile forms pin their order and leave NOPs behind).
NR_INLINE void att2_mma4(float* c, const V4& a, u32 b0, u32 b1, u32 sfa, u32 sfb)
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
NR_INLINE void att2_mma8(u32* c, const u32* a, u32 b0, u32 b1)
{
    u32 d0, d1;
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
        : "=r"(d0), "=r"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}
NR_INLINE u32 att2_transpose8(u32 h)
{
    u32 d;
    asm("movmatrix.sync.trans.aligned.m8n8.b16 %0, %1;" : "=r"(d) : "r"(h));
    return d;
}
// STEP with att2_mma4
#define ATT2_STEP(acc, qbuf, ks_, slab_, s0_)                                                                           \
    {                                                                                                                   \
        const u8* sl_ = (slab_);                                                                                        \
        STEP_OPS(sl_)                                                                                                   \
        _Pragma("unroll")                                                                                               \
        for (u32 d_ = 0; d_ < TPP; ++d_) {                                                                              \
            const u32 u_ = ((s0_) + d_) * KS + (ks_);                                                                   \
            const V4 a_ = *(const V4*)((qbuf) + u_ * 512u + lane * 16u);                                                \
            const u32 sa_ = *(const u32*)((qbuf) + QBYTES + u_ * 64u + g * 8u + (t & 1u) * 4u);                         \
            att2_mma4(acc[d_][0], a_, n01.x, n01.y, sa_, sw.x); att2_mma4(acc[d_][1], a_, n01.z, n01.w, sa_, sw.y);     \
            att2_mma4(acc[d_][2], a_, n23.x, n23.y, sa_, sw.z); att2_mma4(acc[d_][3], a_, n23.z, n23.w, sa_, sw.w);     \
        }                                                                                                               \
    }

// ATT2_STEP over n_ tiles instead of TPP
#define ATT2_STEPN(acc, qbuf, ks_, slab_, s0_, n_)                                                                      \
    {                                                                                                                   \
        const u8* sl_ = (slab_);                                                                                        \
        STEP_OPS(sl_)                                                                                                   \
        _Pragma("unroll")                                                                                               \
        for (u32 d_ = 0; d_ < (n_); ++d_) {                                                                             \
            const u32 u_ = ((s0_) + d_) * KS + (ks_);                                                                   \
            const V4 a_ = ATT2_LDA((qbuf) + u_ * 512u + lane * 16u);                                                    \
            const u32 sa_ = ATT2_LDSA((qbuf) + QBYTES + u_ * 64u + g * 8u + (t & 1u) * 4u);                             \
            att2_mma4(acc[d_][0], a_, n01.x, n01.y, sa_, sw.x); att2_mma4(acc[d_][1], a_, n01.z, n01.w, sa_, sw.y);     \
            att2_mma4(acc[d_][2], a_, n23.x, n23.y, sa_, sw.z); att2_mma4(acc[d_][3], a_, n23.z, n23.w, sa_, sw.w);     \
        }                                                                                                               \
    }
#define ATT2_LDW(q) (*(const V4*)(q))
// ATT2_RELOAD 1: the A operands are read from shared memory again on every pass (volatile loads) instead of being kept
// in registers between passes, which the compiler otherwise does and spills under a 128-register limit
#ifndef ATT2_RELOAD
#define ATT2_RELOAD (HEADS == 8)
#endif
#if ATT2_RELOAD
NR_INLINE V4 att2_lds128(const u8* q)
{
    V4 v;
    asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"((u32)__cvta_generic_to_shared(q)));
    return v;
}
NR_INLINE u32 att2_lds32(const u8* q)
{
    u32 v;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"((u32)__cvta_generic_to_shared(q)));
    return v;
}
#define ATT2_LDA(q) att2_lds128(q)
#define ATT2_LDSA(q) att2_lds32(q)
#else
#define ATT2_LDA(q) (*(const V4*)(q))
#define ATT2_LDSA(q) (*(const u32*)(q))
#endif
// ATT2_OUT_SINGLE 1 (measurement only, not mirrored in the model): the stored output rounded to e4m3 once instead of
// float32 -> float16 -> e4m3
#ifndef ATT2_OUT_SINGLE
#define ATT2_OUT_SINGLE 0
#endif
// tiles per pass over the K and V matrices, and query tiles per Q pass (each followed by its attention)
#ifndef ATT2_KVT
#define ATT2_KVT TPP
#endif
#ifndef ATT2_QT
#define ATT2_QT (HEADS == 8 ? 1u : 2u)              // width 256: one query tile at a time keeps it within 128 registers
#endif

// ---- helpers of the attention half (att_*) ---------------------------------------------------------------------
// four floats -> four e4m3 codes with ONE rounding (a in the lowest byte); f32x4_to_e4m3x4 rounds twice
NR_INLINE u32 att_e4m3x4(float a, float b, float c, float d)
{
    u32 word;
    asm("{\n .reg .b16 nr_l, nr_u;\n cvt.rn.satfinite.e4m3x2.f32 nr_l, %2, %1;\n cvt.rn.satfinite.e4m3x2.f32 nr_u, %4, %3;\n"
        " mov.b32 %0, {nr_l, nr_u};\n}"
        : "=r"(word) : "f"(a), "f"(b), "f"(c), "f"(d));
    return word;
}
// two float16 pairs -> four e4m3 codes (h0 in the low half)
NR_INLINE u32 att_h2x2_to_e4m3x4(u32 h0, u32 h1)
{
    u32 word;
    asm("{\n .reg .b16 nr_l, nr_u;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_l, %1;\n cvt.rn.satfinite.e4m3x2.f16x2 nr_u, %2;\n"
        " mov.b32 %0, {nr_l, nr_u};\n}"
        : "=r"(word) : "r"(h0), "r"(h1));
    return word;
}
// bytes 0, 2, 1, 3 of a word
NR_INLINE u32 att_swap_mid_bytes(u32 a) { u32 d; asm("prmt.b32 %0, %1, 0, 0x3120;" : "=r"(d) : "r"(a)); return d; }
// |result| = max(|a|, |b|) per half (the sign of the result is meaningless)
NR_INLINE u32 att_h2_absmax(u32 a, u32 b) { u32 d; asm("max.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
// largest magnitude of eight float16 values (four pairs) of this lane and of lane ^ 1, as a float
NR_INLINE float att_amax_h2x4_pair(u32 h0, u32 h1, u32 h2, u32 h3)
{
    u32 m = att_h2_absmax(att_h2_absmax(h0, h1), att_h2_absmax(h2, h3));
    m = att_h2_absmax(m, __shfl_xor_sync(0xffffffffu, m, 1));
    float lo, hi;
    h2_unpack(m, lo, hi);
    return fmaxf(fabsf(lo), fabsf(hi));
}
// block_scale without the upper clamp: cvt.rn.satfinite already saturates at 448 (same code, same value)
NR_INLINE u32 att_block_scale(float amax, float& scale, float& reciprocal)
{
    const float raw = fmaxf(amax * 0.16666667163372040f, 0.001953125f);
    u32 code;
    float value;
    asm("{\n .reg .b16 nr_c, nr_l, nr_u;\n .reg .b32 nr_p;\n cvt.rn.satfinite.e4m3x2.f32 nr_c, %2, %2;\n cvt.rn.f16x2.e4m3x2 nr_p, nr_c;\n"
        " mov.b32 {nr_l, nr_u}, nr_p;\n cvt.f32.f16 %1, nr_l;\n cvt.u32.u16 %0, nr_c;\n}"
        : "=r"(code), "=f"(value) : "f"(raw));
    scale = value;
    reciprocal = rcp_approx(value);
    return code & 0xFFu;
}
// Eight raw values c of a token -> eight nibbles of c * ts, the tensor scale ts folded into the reciprocal of the block
// scale: u = c * (ts * r). `amax` is max |c| over this lane and lane ^ 1 (times |ts| inside).
NR_INLINE u32 att_pack_row_ts(const float* c, float amax, float ts, float ts_abs, u32& code)
{
    float scale, r;
    code = att_block_scale(amax * ts_abs, scale, r);
    const float f = ts * r;
    const float u[8] = {c[0] * f, c[1] * f, c[2] * f, c[3] * f, c[4] * f, c[5] * f, c[6] * f, c[7] * f};
    return e2m1x8(u);
}
// Q or K of one tile: squared norms of rows g and g + 8 (pair reduced together), the reciprocal root times `factor`,
// e4m3 words; lo[rh] / hi[rh] = dimensions 0-15 / 16-31 of row rh (the arithmetic of the original loop body)
NR_INLINE void att_qk_rows(const float (*acc)[4], float sq2, float factor, bool odd, u32* lo, u32* hi)
{
    float sum[2];
#pragma unroll
    for (u32 rh = 0; rh < 2; ++rh) {
        float a = acc[0][rh * 2u] * acc[0][rh * 2u];
        a = fmaf(acc[0][rh * 2u + 1u], acc[0][rh * 2u + 1u], a);
#pragma unroll
        for (u32 j = 1; j < 4; ++j)
#pragma unroll
            for (u32 e = 0; e < 2; ++e) a = fmaf(acc[j][rh * 2u + e], acc[j][rh * 2u + e], a);
        sum[rh] = a;
    }
    float x = (odd ? sum[1] : sum[0]) + __shfl_xor_sync(0xffffffffu, odd ? sum[0] : sum[1], 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    x *= sq2;
    const float r = rsqrt_approx(fmaxf(x, 6.19888305664e-05f));
    const float mine = (x >= 65520.0f ? 0.0f : r) * factor;
    const float other = __shfl_xor_sync(0xffffffffu, mine, 1);
#pragma unroll
    for (u32 rh = 0; rh < 2; ++rh) {
        const float f = (odd == (rh != 0u)) ? mine : other;
        const float* a0 = acc[0] + rh * 2u;
        const float* a1 = acc[1] + rh * 2u;
        const float* a2 = acc[2] + rh * 2u;
        const float* a3 = acc[3] + rh * 2u;
        lo[rh] = att_e4m3x4(a0[0] * f, a0[1] * f, a1[0] * f, a1[1] * f);
        hi[rh] = att_e4m3x4(a2[0] * f, a2[1] * f, a3[0] * f, a3[1] * f);
    }
}
#if HEADS == 1
// eight e2m1 nibbles -> the eight values they stand for (nibble 0 first)
NR_INLINE void att_e2m1x8_to_f32(u32 word, float* q)
{
    asm("{\n .reg .b8 nr_a, nr_b, nr_c, nr_d;\n .reg .b16 nr_l, nr_u;\n .reg .b32 nr_p;\n mov.b32 {nr_a, nr_b, nr_c, nr_d}, %8;\n"
        " cvt.rn.f16x2.e2m1x2 nr_p, nr_a;\n mov.b32 {nr_l, nr_u}, nr_p;\n cvt.f32.f16 %0, nr_l;\n cvt.f32.f16 %1, nr_u;\n"
        " cvt.rn.f16x2.e2m1x2 nr_p, nr_b;\n mov.b32 {nr_l, nr_u}, nr_p;\n cvt.f32.f16 %2, nr_l;\n cvt.f32.f16 %3, nr_u;\n"
        " cvt.rn.f16x2.e2m1x2 nr_p, nr_c;\n mov.b32 {nr_l, nr_u}, nr_p;\n cvt.f32.f16 %4, nr_l;\n cvt.f32.f16 %5, nr_u;\n"
        " cvt.rn.f16x2.e2m1x2 nr_p, nr_d;\n mov.b32 {nr_l, nr_u}, nr_p;\n cvt.f32.f16 %6, nr_l;\n cvt.f32.f16 %7, nr_u;\n}"
        : "=f"(q[0]), "=f"(q[1]), "=f"(q[2]), "=f"(q[3]), "=f"(q[4]), "=f"(q[5]), "=f"(q[6]), "=f"(q[7]) : "r"(word));
}
// pack_row2 with the tensor scale folded as in att_pack_row_ts; the first term's rounding is read back from the
// hardware conversion (cvt.rn.satfinite.e2m1x2 rounds as e2m1_round does). `two` must be uniform over the warp.
NR_INLINE u32 att_pack_row2_ts(const float* c, float amax, float ts, float ts_abs, u32& code, u32& word2, u32& code2, bool two)
{
    float scale, r;
    code = att_block_scale(amax * ts_abs, scale, r);
    const float f = ts * r;
    float u[8];
#pragma unroll
    for (u32 i = 0; i < 8; ++i) u[i] = c[i] * f;
    const u32 w1 = e2m1x8(u);
    word2 = 0u;
    code2 = 0u;
    if (two) {
        float q[8], rest[8];
        att_e2m1x8_to_f32(w1, q);
#pragma unroll
        for (u32 i = 0; i < 8; ++i) rest[i] = (u[i] - q[i]) * scale;
        float amax2 = amax8(rest);
        amax2 = fmaxf(amax2, __shfl_xor_sync(0xffffffffu, amax2, 1));
        float scale2, r2;
        code2 = att_block_scale(amax2, scale2, r2);
#pragma unroll
        for (u32 i = 0; i < 8; ++i) rest[i] *= r2;
        word2 = e2m1x8(rest);
    }
    return w1;
}
#endif

// Q or K of one 16-token tile from its float32 accumulators (unscaled): squared norms of rows g and g + 8 over the
// head's 32 dimensions (24 of them in the three lanes next to this one), reduced as a pair (the even lane finishes
// row 0, the odd row 1), factor = rsqrt(sum * sq2) * `scale`, then e4m3 with one rounding. Outputs: the low / high
// operand words of row g, then of row g + 8.
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
    // the runtime sums in float16: a sum beyond its range gives a zero scale
    const float r = rsqrt_approx(fmaxf(x, 6.19888305664e-05f));
    const float mine = (x >= 65520.0f ? 0.0f : r) * scale;
    const float other = __shfl_xor_sync(0xffffffffu, mine, 1);
    const float f0 = odd ? other : mine, f1 = odd ? mine : other;
    lo0 = att_e4m3x4(a[0][0] * f0, a[0][1] * f0, a[1][0] * f0, a[1][1] * f0);
    hi0 = att_e4m3x4(a[2][0] * f0, a[2][1] * f0, a[3][0] * f0, a[3][1] * f0);
    lo1 = att_e4m3x4(a[0][2] * f1, a[0][3] * f1, a[1][2] * f1, a[1][3] * f1);
    hi1 = att_e4m3x4(a[2][2] * f1, a[2][3] * f1, a[3][2] * f1, a[3][3] * f1);
}
// V of one tile: e4m3 bytes of (token g, token g + 8) per dimension, transposed as b16 elements (8 tokens x 8 dims ->
// a lane holds one dim of tokens 2t, 2t + 8, 2t + 1, 2t + 9), bytes reordered; v_j: the B word of dims 8 j ..
NR_INLINE void att2_v_rows(const float (&a)[4][4], float sqkv, u32& v0, u32& v1, u32& v2, u32& v3)
{
    u32 v[4];
#pragma unroll
    for (u32 j = 0; j < 4; ++j)
        v[j] = att_swap_mid_bytes(att2_transpose8(att_e4m3x4(a[j][0] * sqkv, a[j][2] * sqkv, a[j][1] * sqkv, a[j][3] * sqkv)));
    v0 = v[0]; v1 = v[1]; v2 = v[2]; v3 = v[3];
}
// ---- AHEAD, PIPE, ONE_SLAB ---------------------------------------------------------------------------------------
#if PIPE
// a weight load written where it is to be issued
NR_INLINE V4 pipe_ld(const u8* q)
{
    V4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"((u64)q));
    return v;
}
NR_INLINE void pipe_ld2(const u8* q, u32& a, u32& b) { asm volatile("ld.global.v2.u32 {%0, %1}, [%2];" : "=r"(a), "=r"(b) : "l"((u64)q)); }
#endif
#if AHEAD && PIPE
#error AHEAD and PIPE are two forms of the same thing: one at a time
#endif
#if ONE_SLAB && (AHEAD || PIPE || HEADS < 2)
#error ONE_SLAB is a timing build of the plain order (widths 64 to 256, with or without EXP_HALF)
#endif
#if (AHEAD || PIPE) && (HEADS < 4 || TPP != 4u || ATT2_KVT != 4u)
#error AHEAD and PIPE are written for widths 128 and 256 in their default orders (TPP 4, ATT2_KVT 4)
#endif
#if STAGE && (AHEAD || PIPE || ONE_SLAB || WINS > 1 || HEADS < 2 || (EXP_HALF && REG_MLP))
#error STAGE is a form of its own (not with AHEAD, PIPE, ONE_SLAB, WINS or EXP_HALF; widths 64 to 256)
#endif
#if STAGE
#define ST_RING 3u                                              // slots per warp: the step being read and the two after it
NR_INLINE void st_cp16(u32 dst, const u8* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"((u64)src) : "memory"); }
NR_INLINE void st_cp4(u32 dst, const u8* src) { asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"((u64)src) : "memory"); }
NR_INLINE void st_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
NR_INLINE void st_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
NR_INLINE void st_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
struct W3 { V4 n01, n23, sw; };                                 // the operand words of one k64 step
#endif
#if AHEAD
NR_INLINE void ahead_touch(const u8* q) { asm volatile("prefetch.global.L1 [%0];" :: "l"((u64)q)); }
// bytes_ at q_: AHEAD_UNIT per lane and instruction (lanes past the end ask for the last byte again)
#define AHEAD_RANGE(q_, bytes_)                                                                                         \
    {                                                                                                                   \
        _Pragma("unroll")                                                                                               \
        for (u32 o_ = 0; o_ < (u32)(bytes_); o_ += 32u * AHEAD_UNIT)                                                    \
            ahead_touch((q_) + o_ + min(lane * AHEAD_UNIT, (u32)(bytes_) - o_ - 1u));                                   \
    }
// Step i_ of a run of n_ slabs at run_: the slab AHEAD steps on; past the end of the run, the slab that many steps into
// the run that follows (next_, m_ slabs; m_ 0: none).
#define AHEAD_STEP(run_, i_, n_, next_, m_)                                                                             \
    {                                                                                                                   \
        if ((i_) + AHEAD < (n_)) AHEAD_RANGE((run_) + (u64)((i_) + AHEAD) * SLAB, SLAB)                                 \
        else if ((i_) + AHEAD - (n_) < (m_)) AHEAD_RANGE((next_) + (u64)((i_) + AHEAD - (n_)) * SLAB, SLAB)             \
    }
// the first slabs of a run, before anything of it is loaded
#define AHEAD_FIRST(run_, n_)                                                                                           \
    {                                                                                                                   \
        _Pragma("unroll")                                                                                               \
        for (u32 f_ = 0; f_ < AHEAD && f_ < (n_); ++f_) AHEAD_RANGE((run_) + (u64)f_ * SLAB, SLAB)                      \
    }
#elif PIPE
// PIPE_CHECK 1 (a compile-only proof, coordinator/once_build.py pipecheck; never a build to run): every pipelined load
// carries the address it was loaded from, and every step compares it with the address the release loads there. The
// compiler folds a comparison away when the two are the same expression; one it cannot fold leaves a
// `nanosleep <source line>` in the PTX.
#ifndef PIPE_CHECK
#define PIPE_CHECK 0
#endif
#if PIPE_CHECK
struct W3 { V4 n01, n23, sw; const u8* at; };
#define PIPE_STR2(x_) #x_
#define PIPE_STR(x_) PIPE_STR2(x_)
// (as pointers, or as 32-bit offsets into the record: there a sum formed before and after widening to 64 bits is one
// expression; either form the compiler can fold proves the site)
#define PIPE_SAME(a_, b_)                                                                                                   {                                                                                                                           if ((const u8*)(a_) != (const u8*)(b_) && (u32)((u64)(const u8*)(a_) - (u64)p.w) != (u32)((u64)(const u8*)(b_) - (u64)p.w))             asm volatile("nanosleep.u32 " PIPE_STR(__LINE__) ";" ::: "memory");                                             }
#define PIPE_SAMEQ(a_, b_) { if ((a_) != (b_)) asm volatile("nanosleep.u32 " PIPE_STR(__LINE__) ";" ::: "memory"); }
#define PIPE_AT(dst_, sl_) (dst_) = (sl_);
#else
#define PIPE_SAMEQ(a_, b_)
struct W3 { V4 n01, n23, sw; };                                 // the operand words of one k64 step: nibbles of outputs 0-15, 16-31, scales
#define PIPE_SAME(a_, b_)
#define PIPE_AT(dst_, sl_)
#endif
#define PIPE_LOAD(dst_, sl_)                                                                                            \
    {                                                                                                                   \
        (dst_).n01 = pipe_ld((sl_) + lane * 16u);                                                                       \
        (dst_).n23 = pipe_ld((sl_) + 512u + lane * 16u);                                                                \
        (dst_).sw = pipe_ld((sl_) + 1024u + g * 16u);                                                                   \
        PIPE_AT((dst_).at, (sl_))                                                                                       \
    }
// EXP_HALF: the operands of half step q_ of the expansion (nibbles of two fragments, their two scale words)
#define PIPE_HALF(q_)                                                                                                   \
    {                                                                                                                   \
        const u32 o_ = ((q_) / (2u * KS) * KS + (q_) % KS) * SLAB, h_ = ((q_) / KS) & 1u;                               \
        pipe_hn = pipe_ld(pipe_en + o_ + h_ * 512u);                                                                    \
        pipe_ld2(pipe_es + o_ + h_ * 8u, pipe_hs0, pipe_hs1);                                                           \
        PIPE_AT(pipe_hq, (q_))                                                                                          \
    }
#define AHEAD_RANGE(q_, bytes_) {}
// Step i_ of a run of n_ slabs at run_: the operands loaded a step ago become this step's, and the next step's are
// loaded: the next slab of the run; after the run's last step the first slab of the run that follows (m_ 0: none).
#define AHEAD_STEP(run_, i_, n_, next_, m_)                                                                             \
    {                                                                                                                   \
        pipe_cur = pipe_nxt;                                                                                            \
        if ((i_) + 1u < (n_)) PIPE_LOAD(pipe_nxt, (run_) + (u64)((i_) + 1u) * SLAB)                                     \
        else if ((m_) > 0u) PIPE_LOAD(pipe_nxt, (next_))                                                                \
    }
#define AHEAD_FIRST(run_, n_) PIPE_LOAD(pipe_nxt, (run_))
#elif STAGE
// One commit group per slab: the warp copies the slab's 1152 bytes (each lane its 16 bytes of both nibble halves and 4
// bytes of the scale words) into slot r_ of its ring.
#define ST_ISSUE(sl_, r_)                                                                                               \
    {                                                                                                                   \
        const u8* const st_s_ = (sl_);                                                                                  \
        const u32 st_d_ = (u32)__cvta_generic_to_shared(st_ring + ((r_) % ST_RING) * SLAB);                             \
        st_cp16(st_d_ + lane * 16u, st_s_ + lane * 16u);                                                                \
        st_cp16(st_d_ + 512u + lane * 16u, st_s_ + 512u + lane * 16u);                                                  \
        st_cp4(st_d_ + 1024u + lane * 4u, st_s_ + 1024u + lane * 4u);                                                   \
        st_commit();                                                                                                    \
    }
#define AHEAD_RANGE(q_, bytes_) {}
// Step i_ of a run of n_ slabs at run_ (followed by m_ slabs at next_; m_ 0: nothing known): notes the slabs of the next
// two steps; STEP_OPS issues those not yet under way once it has read this step's words. All of the state (st_n: how
// many of the coming steps are under way, st_k: the step count that picks the slot) is known when compiling.
#define AHEAD_STEP(run_, i_, n_, next_, m_)                                                                             \
    {                                                                                                                   \
        st_has1 = ((i_) + 1u < (n_)) || ((i_) + 1u - (n_) < (m_));                                                      \
        st_has2 = st_has1 && (((i_) + 2u < (n_)) || ((i_) + 2u - (n_) < (m_)));                                         \
        st_a1 = ((i_) + 1u < (n_)) ? (run_) + (u64)((i_) + 1u) * SLAB : (next_) + (u64)((i_) + 1u - (n_)) * SLAB;       \
        st_a2 = ((i_) + 2u < (n_)) ? (run_) + (u64)((i_) + 2u) * SLAB : (next_) + (u64)((i_) + 2u - (n_)) * SLAB;       \
        st_armed = true;                                                                                                \
    }
// the first two slabs of a run nothing leads into, as the next two steps'
#define AHEAD_FIRST(run_, n_)                                                                                           \
    {                                                                                                                   \
        if (st_n == 0u) {                                                                                               \
            ST_ISSUE((run_), st_k)                                                                                      \
            if ((n_) > 1u) ST_ISSUE((run_) + (u64)SLAB, st_k + 1u)                                                      \
            st_n = (n_) > 1u ? 2u : 1u;                                                                                 \
        }                                                                                                               \
    }
#else
#define AHEAD_RANGE(q_, bytes_) {}
#define AHEAD_STEP(run_, i_, n_, next_, m_) {}
#define AHEAD_FIRST(run_, n_) {}
#endif
// the operand words of the k64 step at slab sl_
#if PIPE
#define STEP_OPS(sl_) PIPE_SAME(pipe_cur.at, sl_) const V4 n01 = pipe_cur.n01, n23 = pipe_cur.n23, sw = pipe_cur.sw;
#elif ONE_SLAB
#define ONE_AT(q_) (one_slab + (u64)(one_zero * (u32)((q_) - one_slab)))
#define STEP_OPS(sl_)                                                                                                   \
    const u8* const one_o_ = ONE_AT(sl_);                                                                               \
    const V4 n01 = *(const V4*)(one_o_ + lane * 16u), n23 = *(const V4*)(one_o_ + 512u + lane * 16u);                   \
    const V4 sw = *(const V4*)(one_o_ + 1024u + g * 16u);
#elif STAGE
// The step's words from its slot (after the copy issued two steps ago has landed: wait_group 1 leaves the next step's
// copy in flight), or straight from the record where nothing was staged; then the copies of the steps after it.
// A step without AHEAD_STEP before it is the release's load.
#define STEP_OPS(sl_)                                                                                                   \
    V4 n01, n23, sw;                                                                                                    \
    if (st_qmode == 2u) { n01 = st_q[st_qi].n01; n23 = st_q[st_qi].n23; sw = st_q[st_qi].sw; }                          \
    else {                                                                                                              \
        if (st_armed && st_n > 0u) {                                                                                    \
            if (st_n > 1u) st_wait1(); else st_wait0();                                                                 \
            const u8* const st_r_ = st_ring + (st_k % ST_RING) * SLAB;                                                  \
            n01 = once_lds128(st_r_ + lane * 16u); n23 = once_lds128(st_r_ + 512u + lane * 16u);                        \
            sw = once_lds128(st_r_ + 1024u + g * 16u);                                                                  \
        } else {                                                                                                        \
            n01 = *(const V4*)((sl_) + lane * 16u); n23 = *(const V4*)((sl_) + 512u + lane * 16u);                      \
            sw = *(const V4*)((sl_) + 1024u + g * 16u);                                                                 \
            if (!st_armed) st_n = 0u;                                                                                   \
        }                                                                                                               \
        if (st_armed) {                                                                                                 \
            if (st_has1 && st_n < 2u) ST_ISSUE(st_a1, st_k + 1u)                                                        \
            if (st_has2) ST_ISSUE(st_a2, st_k + 2u)                                                                     \
            st_n = (st_has1 ? 1u : 0u) + (st_has2 ? 1u : 0u);                                                           \
            st_armed = false;                                                                                           \
        }                                                                                                               \
        ++st_k;                                                                                                         \
        if (st_qmode == 1u) { st_q[st_qi].n01 = n01; st_q[st_qi].n23 = n23; st_q[st_qi].sw = sw; }                      \
    }
#else
#define STEP_OPS(sl_) const V4 n01 = *(const V4*)((sl_) + lane * 16u), n23 = *(const V4*)((sl_) + 512u + lane * 16u); const V4 sw = *(const V4*)((sl_) + 1024u + g * 16u);
#endif
NR_INLINE int chain_cta(u32 axis); NR_INLINE u32 chain_flag(); NR_INLINE bool chain_wait_par(); NR_INLINE void chain_wait_parallel(const Params& p, int x0, int y0); NR_INLINE void chain_prefetch(const Params& p, u32 w, u32 lane); NR_INLINE void chain_trace(const Params& p, u32 slot);  // chain switches: end of file
extern "C" __global__ void nr_swin4(Params p)
{
#if WINS > 1
#if CTA_ORD || WAIT_PAR || PRE_WAIT || NR_TRACE
#error WINS > 1 is not combined with the chain switches
#endif
    __shared__ __align__(16) u8 sm_x_[WINS * XBYTES];
    __shared__ __align__(16) u8 sm_qa_[WINS * (QBYTES + SBYTES)];
    __shared__ __align__(16) u8 sm_qb_[WINS * (QBYTES + SBYTES)];
#if EXP_HALF && REG_MLP && HEADS > 1
    __shared__ __align__(16) u8 sm_h_[WINS * HBYTES];
#endif
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y % HEADS, win = threadIdx.y / HEADS, g = lane >> 2, t = lane & 3u;
    u8* const sm_x = sm_x_ + win * XBYTES;
    u8* const sm_qa = sm_qa_ + win * (QBYTES + SBYTES);
    u8* const sm_qb = sm_qb_ + win * (QBYTES + SBYTES);
#if EXP_HALF && REG_MLP && HEADS > 1
    u8* const sm_h = sm_h_ + win * HBYTES;
#endif
    const int win_x = (int)(blockIdx.x * WINS + win);           // this thread's window in its row
    const bool win_live = win_x < ((p.W - min(p.sx, 0) + 7) >> 3);   // the last CTA of a row may hold a window past the map
    const bool leader = threadIdx.x == 0u && w == 0u && win_live;
    const int x0 = p.sx + 8 * win_x, y0 = p.sy + 8 * (int)blockIdx.y;
#else
    __shared__ __align__(16) u8 sm_x[XBYTES];
    __shared__ __align__(16) u8 sm_qa[QBYTES + SBYTES];
    __shared__ __align__(16) u8 sm_qb[QBYTES + SBYTES];
#if EXP_HALF && REG_MLP && HEADS > 1
    __shared__ __align__(16) u8 sm_h[HBYTES];
#endif
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int x0 = p.sx + 8 * chain_cta(0u), y0 = p.sy + 8 * chain_cta(1u);       // CTA_ORD 0: blockIdx.x, blockIdx.y
#endif
    const int TW = p.W >> 2, TH = p.H >> 2, tx0 = x0 >> 2, ty0 = y0 >> 2;
    const float* const F = (const float*)p.w;
#if PIPE && PIPE_CHECK                                          // the offsets as values, so that two uses of one are one expression
    u32 O[11];
#pragma unroll
    for (u32 i_ = 0; i_ < 11; ++i_) O[i_] = ((const u32*)(p.w + 64))[i_];
#else
    const u32* const O = (const u32*)(p.w + 64);
#endif
#if ONE_SLAB
    const u8* const one_slab = p.w + O[O_EXPAND];
    const u32 one_zero = threadIdx.x >> 5;                      // 0: the block is 32 threads wide (not known when compiling)
#endif
#if PIPE
    W3 pipe_cur, pipe_nxt;                                      // operand words of the step being multiplied / of the step after it
#if EXP_HALF && REG_MLP && HEADS > 1
    V4 pipe_hn;                                                 // EXP_HALF expansion: the next half step's nibbles ...
    u32 pipe_hs0, pipe_hs1;                                     // ... and its two scale words
#if PIPE_CHECK
    u32 pipe_hq;                                                // which half step the waiting operands belong to
#endif
    const u8* const pipe_en = p.w + O[O_EXPAND] + (u64)(w * 4u * KS) * SLAB + lane * 16u;        // this lane's nibbles in the warp's first
    const u8* const pipe_es = p.w + O[O_EXPAND] + (u64)(w * 4u * KS) * SLAB + 1024u + g * 16u;   // expansion slab, and its scale words
#endif
#endif
#if STAGE
    __shared__ __align__(16) u8 sm_st[HEADS * ST_RING * SLAB];
    u8* const st_ring = sm_st + w * (ST_RING * SLAB);           // this warp's three slots
    u32 st_n = 0u, st_k = 0u, st_qmode = 0u, st_qi = 0u;        // (known when compiling: the kernel is straight-line)
    bool st_armed = false, st_has1 = false, st_has2 = false;
    const u8 *st_a1 = p.w, *st_a2 = p.w;
    W3 st_q[KS];                                                // Q's operand words, kept for the later query tiles
#endif

    bool inside[4];
#if REG_TA32
    u32 tile_at[4];                                             // (32 bits: maps below 4 GiB) byte offset
#else
    u64 tile_at[4];                                             // byte offset of this lane's 16 bytes of tile s in a tile-layout map
#endif
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        const int ty = ty0 + (int)(s >> 1), tx = tx0 + (int)(s & 1u);
        inside[s] = ty >= 0 && ty < TH && tx >= 0 && tx < TW;
#if REG_TA32
        tile_at[s] = (u32)(ty * TW + tx) * (16u * C) + w * 512u + lane * 16u;
#else
        tile_at[s] = (u64)(u32)(ty * TW + tx) * (16u * C) + w * 512u + lane * 16u;
#endif
    }

#if AHEAD || PIPE || STAGE                                      // the first slabs the block reads, asked for before the wait
#if VARIANT == 3
    AHEAD_FIRST(p.w + O[O_RES] + (u64)(w * HEADS) * SLAB, HEADS)
#elif PIPE && EXP_HALF && REG_MLP && HEADS > 1
    PIPE_HALF(0u)
#else
    AHEAD_FIRST(p.w + O[O_EXPAND] + (u64)(w * 4u * KS) * SLAB, 4u * KS)
#endif
#endif
    chain_trace(p, 0u);                                         // ---- wait for the producer's windows under this one --------
    if (p.pflags) { chain_prefetch(p, w, lane);
        if (leader && !chain_wait_par()) {
            const int gx = (p.W - min(p.psx, 0) + 7) >> 3, gy = (p.H - min(p.psy, 0) + 7) >> 3;
            const int dy = y0 - p.psy, dx = x0 - p.psx;
            const int r0 = max(dy, 0) >> 3, r1 = min(dy + 8 > 0 ? ((dy + 7) >> 3) + 1 : 0, gy);
            const int c0 = max(dx, 0) >> 3, c1 = min(dx + 8 > 0 ? ((dx + 7) >> 3) + 1 : 0, gx);
            int budget = WAIT_BUDGET_NS;
            for (int r = r0; r < r1; ++r)
                for (int c = c0; c < c1; ++c) wait_not_negative(p.pflags + r * gx + c, budget);
        } else if (chain_wait_par()) chain_wait_parallel(p, x0, y0);
        __syncthreads();
    } chain_trace(p, 1u);

    // ---- the block's input: values, the residual's copy, the 4-bit operand of the expansion ---------------------
#if VARIANT == 3
    u32 pr[2][4];                                               // projected coarse pixels of this lane, float16 pairs
    {
        const u32 ci = (g & 1u) + 2u * (g >> 2), cxl = (g >> 1) & 1u;
        const int cx = (x0 >> 1) + (int)((ci & 1u) * 2u + cxl);
#pragma unroll
        for (u32 hb = 0; hb < 2; ++hb) {                        // this warp's 64 of the 2C channels: one k64 step
            u32 hv[2][4];                                       // the two rows as float16 pairs (exact)
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                const int cy = (y0 >> 1) + (int)((ci >> 1) * 2u + rh);
                const bool in = cy >= 0 && cy < p.d0 && cx >= 0 && cx < p.d1;
#pragma unroll
                for (u32 a = 0; a < 2; ++a) {
                    const u64 at = ((u64)(u32)((int)(4u * w + 2u * hb + a) * p.d0 + (in ? cy : 0)) * (u32)p.d1 + (u32)(in ? cx : 0)) * 16u + t * 4u;
                    const u32 word = in ? load_cg32(p.in + at) : 0u;
                    m_e4m3x4_to_h2(word, hv[rh][2 * a], hv[rh][2 * a + 1]);
                }
            }
            u32 n0, n1;
            const u32 codes = m_pack2_h(hv[0], hv[1], F[F_TR], n0, n1);
            *(u64*)(sm_qa + w * 512u + lane * 16u + hb * 8u) = (u64)n0 | ((u64)n1 << 32);
            m_put_scale(sm_qa + QBYTES + w * 64u, g, t, hb, codes);
        }
        NR_BARRIER();
        float pacc[1][4][4];
        ZERO_ACC(pacc, 1, 4)
        const u8* wres = p.w + O[O_RES] + (u64)(w * HEADS) * SLAB;
#if PIPE || STAGE
#pragma unroll
#endif
        for (u32 ks = 0; ks < HEADS; ++ks) {
            AHEAD_STEP(wres, ks, HEADS, p.w + O[O_EXPAND] + (u64)(w * 4u * KS) * SLAB, (PIPE && EXP_HALF && REG_MLP && HEADS > 1) ? 0u : 4u * KS)
            const u8* sl = wres + ks * SLAB;
            STEP_OPS(sl)
            const V4 a = *(const V4*)(sm_qa + ks * 512u + lane * 16u);
            const u32 sa = *(const u32*)(sm_qa + QBYTES + ks * 64u + g * 8u + (t & 1u) * 4u);
            mma4(pacc[0][0], a, n01.x, n01.y, sa, sw.x); mma4(pacc[0][1], a, n01.z, n01.w, sa, sw.y);
            mma4(pacc[0][2], a, n23.x, n23.y, sa, sw.z); mma4(pacc[0][3], a, n23.z, n23.w, sa, sw.w);
        }
#if PIPE && EXP_HALF && REG_MLP && HEADS > 1
        PIPE_HALF(0u)                                           // the expansion's first half step
#endif
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
            for (u32 j = 0; j < 4; ++j) pr[rh][j] = h2_pack(pacc[0][j][rh * 2u] * F[F_SRES], pacc[0][j][rh * 2u + 1u] * F[F_SRES]);
        NR_BARRIER();                                        // sm_qa is reused for x below
    }
    const float* const gain = (const float*)(p.w + O[O_GAIN] + (w * 4u + t) * 32u);
    const u32 gain16[4] = {h2_pack(gain[0], gain[1]), h2_pack(gain[2], gain[3]), h2_pack(gain[4], gain[5]), h2_pack(gain[6], gain[7])};
    const int STW = p.e1 >> 2, STH = p.e0 >> 2;
#endif
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        u32 xwords[4];                                          // the lane's 16 values of this tile as e4m3 codes
        u32 hv[2][4];                                           // the two rows the expansion packs, float16 pairs
#if VARIANT == 3
        const int sty = ty0 + (int)(s >> 1), stx = tx0 + (int)(s & 1u);
        const bool sin = sty >= 0 && sty < STH && stx >= 0 && stx < STW;
        const V4 skw = sin ? load_cg(p.skip + (u64)(u32)(sty * STW + stx) * (16u * C) + w * 512u + lane * 16u) : V4{0u, 0u, 0u, 0u};
        const u32 skwords[4] = {skw.x, skw.y, skw.z, skw.w};
        const u32 src = 4u * ((s & 1u) + 2u * ((g >> 1) & 1u) + 4u * (s >> 1)) + t;
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh) {
            u32 sk[4];
            m_e4m3x4_to_h2(skwords[rh], sk[0], sk[1]);
            m_e4m3x4_to_h2(skwords[2u + rh], sk[2], sk[3]);
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                // float16(skip * gain + projection): one rounding (fma.rn.f16x2)
                const u32 b = h2_fma(sk[j], gain16[j], __shfl_sync(0xffffffffu, pr[rh][j], src));
                hv[rh][j] = inside[s] ? b : 0u;
            }
            xwords[rh] = m_h2x2_to_e4m3x4(hv[rh][0], hv[rh][1]);
            xwords[2u + rh] = m_h2x2_to_e4m3x4(hv[rh][2], hv[rh][3]);
        }
#else
#if VARIANT == 1
        {
            const int vr = p.d0, vc = p.d1;
#pragma unroll
            for (u32 rh = 0; rh < 2; ++rh) {
                const u32 m = g + 8u * rh;
                const int py = (ty0 + (int)(s >> 1)) * 4 + (int)(m >> 2), px = (tx0 + (int)(s & 1u)) * 4 + (int)(m & 3u);
                const bool in = py >= 0 && py < vr && px >= 0 && px < vc && inside[s];
#pragma unroll
                for (u32 a = 0; a < 2; ++a) {
                    const u64 at = ((u64)(u32)((int)(2u * w + a) * vr + (in ? py : 0)) * (u32)vc + (u32)(in ? px : 0)) * 16u + t * 4u;
                    xwords[2u * a + rh] = in ? load_cg32(p.in + at) : 0u;
                }
            }
        }
#else
        {
            const V4 xw = inside[s] ? load_cg(p.in + tile_at[s]) : V4{0u, 0u, 0u, 0u};
            xwords[0] = xw.x; xwords[1] = xw.y; xwords[2] = xw.z; xwords[3] = xw.w;
        }
#endif
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh) {
            m_e4m3x4_to_h2(xwords[rh], hv[rh][0], hv[rh][1]);
            m_e4m3x4_to_h2(xwords[2u + rh], hv[rh][2], hv[rh][3]);
        }
#endif
#if HEADS == 1
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh) {                        // the residual reads float16 values
            const V4 wide = {hv[rh][0], hv[rh][1], hv[rh][2], hv[rh][3]};
            *(V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u) = wide;
        }
        {
            u32 n0, n1, e0, e1, codes2;
            const u32 codes = m_pack2_h_two(hv[0], hv[1], F[F_TX], n0, n1, e0, e1, codes2, (p.options & OPT_TWO_X) != 0u);
            const V4 q = {n0, n1, e0, e1};
            *(V4*)(sm_qa + s * 512u + lane * 16u) = q;
            m_put_scale(sm_qa + QBYTES + s * 64u, g, t, 0u, codes);
            m_put_scale(sm_qa + QBYTES + s * 64u, g, t, 1u, codes2);
        }
#else
        {
            u32 n0, n1;
            const u32 codes = m_pack2_h(hv[0], hv[1], F[F_TX], n0, n1);
            *(u64*)(sm_qa + (s * KS + (w >> 1)) * 512u + lane * 16u + (w & 1u) * 8u) = (u64)n0 | ((u64)n1 << 32);
            m_put_scale(sm_qa + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes);
            const V4 keep = {xwords[0], xwords[1], xwords[2], xwords[3]};
            *(V4*)(sm_x + (s * HEADS + w) * 512u + lane * 16u) = keep;
        }
#endif
    }
    NR_BARRIER();

#if DBG_STAGE != 2
#if EXP_HALF && REG_MLP && HEADS > 1
    // ---- MLP with every expansion slab read once per window (EXP_HALF): all four tiles per pass, two of the pass's
    // four 8-output fragments at a time. Each accumulator sees the same products in the same order as below.
    {
        const float hlim = F[F_HLIM];
        const u32 eb = (__float_as_uint(hlim) >> 23) & 0xFFu;
        const float kk = __uint_as_float((256u - eb) << 23), ik = __uint_as_float((eb - 2u) << 23);
        const u32 G1 = h2_pack(F[F_HC1] * ik * ik * ik, F[F_HC1] * ik * ik * ik), G2 = h2_pack(F[F_HC2] * ik * ik, F[F_HC2] * ik * ik);
        const u32 G3 = h2_pack(F[F_HC3] * ik, F[F_HC3] * ik), HL = h2_pack(hlim * kk, hlim * kk);
        const u32 i1 = t & 1u;
        const u32 sel_ms = (t & 2u) ? ((4u + i1) | (i1 << 4) | ((6u + i1) << 8) | ((2u + i1) << 12))
                                    : (i1 | ((4u + i1) << 4) | ((2u + i1) << 8) | ((6u + i1) << 12));
        const float smid = F[F_SMID];
        u32 MAs[4][2][4], MSs[4][2], even_s[4];
#pragma unroll
        for (u32 pass = 0; pass < 4; ++pass) {
            const u8* wexp = p.w + O[O_EXPAND] + (u64)((w * 4u + pass) * KS) * SLAB;
#pragma unroll
            for (u32 half = 0; half < 2; ++half) {
                float acc[4][2][4];
                ZERO_ACC(acc, 4, 2)
#pragma unroll
                for (u32 ks = 0; ks < KS; ++ks) {
#if AHEAD
                    {   // the half slab AHEAD half-steps on (its scale words come with the first half); past the last, the middle's
                        const u32 q2_ = (pass * 2u + half) * KS + ks + AHEAD;
                        if (q2_ < 8u * KS) {
                            const u8* const s2_ = p.w + O[O_EXPAND] + (u64)((w * 4u + q2_ / (2u * KS)) * KS + q2_ % KS) * SLAB;
                            AHEAD_RANGE(s2_ + ((q2_ / KS) & 1u) * 512u, 512u)
                            if (((q2_ / KS) & 1u) == 0u) AHEAD_RANGE(s2_ + 1024u, 128u)
                        } else if (q2_ - 8u * KS < 2u) {
                            AHEAD_RANGE(p.w + O[O_MID] + (u64)(w * 2u + q2_ - 8u * KS) * SLAB, SLAB)
                        }
                    }
#endif
#if PIPE
                    const V4 n = pipe_hn;                                                   // loaded one half step ago
                    const u32 sw0 = pipe_hs0, sw1 = pipe_hs1;
                    // half step q reads slab (q / (2 KS)) KS + q % KS = pass KS + ks of the warp, half (q / KS) & 1 = half:
                    // pipe_en + o_ + h_ 512 is wexp + ks SLAB + half 512 + lane 16, pipe_es + o_ + h_ 8 the scale words below
                    PIPE_SAMEQ(pipe_hq, (pass * 2u + half) * KS + ks)
                    if ((pass * 2u + half) * KS + ks + 1u < 8u * KS) PIPE_HALF((pass * 2u + half) * KS + ks + 1u)
                    else PIPE_LOAD(pipe_nxt, p.w + O[O_MID] + (u64)(w * 2u) * SLAB)       // after the last: the middle projection's first slab
#elif ONE_SLAB
                    const u8* sl_ = wexp + ks * SLAB;
                    const V4 n = *(const V4*)(ONE_AT(sl_ + half * 512u) + lane * 16u);
                    const u64 sw = *(const u64*)(ONE_AT(sl_ + half * 8u) + 1024u + g * 16u);
                    const u32 sw0 = (u32)sw, sw1 = (u32)(sw >> 32);
#else
                    const u8* sl_ = wexp + ks * SLAB;
                    const V4 n = *(const V4*)(sl_ + half * 512u + lane * 16u);              // fragments 2 half, 2 half + 1
                    const u64 sw = *(const u64*)(sl_ + 1024u + g * 16u + half * 8u);        // their two scale words
                    const u32 sw0 = (u32)sw, sw1 = (u32)(sw >> 32);
#endif
#pragma unroll
                    for (u32 d = 0; d < 4; ++d) {
                        const u32 u_ = d * KS + ks;
                        const V4 a_ = once_lds128(sm_qa + u_ * 512u + lane * 16u);
                        const u32 sa_ = once_lds32(sm_qa + QBYTES + u_ * 64u + g * 8u + (t & 1u) * 4u);
                        mma4(acc[d][0], a_, n.x, n.y, sa_, sw0); mma4(acc[d][1], a_, n.z, n.w, sa_, sw1);
                    }
                }
#pragma unroll
                for (u32 d = 0; d < 4; ++d) {
                    u32 hw[2][2];
#pragma unroll
                    for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
                        for (u32 j = 0; j < 2; ++j) {
                            const u32 x = h2_pack(acc[d][j][rh * 2u] * kk, acc[d][j][rh * 2u + 1u] * kk);
                            const u32 c = m_clamp(x, HL);
                            hw[rh][j] = m_mul(x, h2_fma(c, h2_fma(m_abs(c), G1, G2), G3));
                        }
                    u8* const park = (d < 2u ? sm_qb + (d * HEADS + w) * 512u : sm_h + ((d - 2u) * HEADS + w) * 512u) + lane * 16u;
                    if (half == 0u) {
                        once_sts128(park, hw[0][0], hw[0][1], hw[1][0], hw[1][1]);
                    } else {
                        const V4 f = once_lds128(park);
                        const u32 h0[4] = {f.x, f.y, hw[0][0], hw[0][1]}, h1[4] = {f.z, f.w, hw[1][0], hw[1][1]};
                        u32 w0, w1;
                        const u32 codes = m_pack2_h(h0, h1, 1.0f, w0, w1);
                        MAs[d][pass >> 1][(pass & 1u) * 2u] = w0;
                        MAs[d][pass >> 1][(pass & 1u) * 2u + 1u] = w1;
                        if (pass & 1u) {
                            const u32 mine = m_prmt(even_s[d], codes, 0x5410u);
                            MSs[d][pass >> 1] = m_prmt(mine, __shfl_xor_sync(0xffffffffu, mine, 2), sel_ms);
                        } else {
                            even_s[d] = codes;
                        }
                    }
                }
            }
        }
        NR_BARRIER();                                        // every warp is done with its words in sm_qb: the middle
                                                                // projection writes its output there
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += 2u) {                    // the middle projection, two tiles at a time (as REG_MLP, MT 2)
#if PIPE && PIPE_CHECK
            const u64 zs = 0u;                                  // (zero in every launch; written out so the check can compare addresses)
#else
            const u64 zs = (u64)(threadIdx.z * s0);
#endif
            const u8* wmid = p.w + zs + O[O_MID] + (u64)(w * 2u) * SLAB;
            float acc[2][4][4];
            ZERO_ACC(acc, 2, 4)
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) {
#if PIPE
                AHEAD_STEP(wmid, ks, 2u, s0 == 0u ? wmid : p.w + O[O_CONTRACT] + (u64)(w * KS) * SLAB, s0 == 0u ? 2u : KS)      // (the second tile pair reads the same two slabs)
#else
                AHEAD_STEP(wmid, ks, 2u, p.w + O[O_CONTRACT] + (u64)(w * KS) * SLAB, s0 == 0u ? 0u : KS)
#endif
                const u8* sl_ = wmid + ks * SLAB;
                STEP_OPS(sl_)
#pragma unroll
                for (u32 d = 0; d < 2; ++d) {
                    const V4 a_ = {MAs[s0 + d][ks][0], MAs[s0 + d][ks][1], MAs[s0 + d][ks][2], MAs[s0 + d][ks][3]};
                    mma4(acc[d][0], a_, n01.x, n01.y, MSs[s0 + d][ks], sw.x); mma4(acc[d][1], a_, n01.z, n01.w, MSs[s0 + d][ks], sw.y);
                    mma4(acc[d][2], a_, n23.x, n23.y, MSs[s0 + d][ks], sw.z); mma4(acc[d][3], a_, n23.z, n23.w, MSs[s0 + d][ks], sw.w);
                }
            }
#pragma unroll
            for (u32 d = 0; d < 2; ++d) {
                const u32 s = s0 + d;
                float v[2][8], a[2];
#pragma unroll
                for (u32 rh = 0; rh < 2; ++rh) {
#pragma unroll
                    for (u32 j = 0; j < 4; ++j)
#pragma unroll
                        for (u32 e = 0; e < 2; ++e) v[rh][2 * j + e] = acc[d][j][rh * 2u + e];
                    a[rh] = m_amax8x(v[rh]);
                    a[rh] = fabsf(m_xamaxf(a[rh], __shfl_xor_sync(0xffffffffu, a[rh], 1))) * smid;
                }
                float s0v, s1v, r0, r1;
                const u32 codes = m_scale2(a[0], a[1], s0v, s1v, r0, r1);
                r0 *= smid;
                r1 *= smid;
#pragma unroll
                for (u32 i = 0; i < 8; ++i) { v[0][i] *= r0; v[1][i] *= r1; }
                const u32 n0 = e2m1x8(v[0]), n1 = e2m1x8(v[1]);
                *(u64*)(sm_qb + (s * KS + (w >> 1)) * 512u + lane * 16u + (w & 1u) * 8u) = (u64)n0 | ((u64)n1 << 32);
                m_put_scale(sm_qb + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes);
            }
        }
    }
    NR_BARRIER();
#elif REG_MLP && HEADS > 1
    // ---- MLP, MT tiles at a time (MT = 1 or 2): the four expansion passes of those tiles, their hidden layer (10
    // registers per tile), then their grouped middle projection. Same products in the same order as the pass-outer
    // form; the weight slabs are read once per tile group instead of once per window (from L1).
    {
        const float hlim = F[F_HLIM];
        const u32 eb = (__float_as_uint(hlim) >> 23) & 0xFFu;
        const float kk = __uint_as_float((256u - eb) << 23), ik = __uint_as_float((eb - 2u) << 23);
        const u32 G1 = h2_pack(F[F_HC1] * ik * ik * ik, F[F_HC1] * ik * ik * ik), G2 = h2_pack(F[F_HC2] * ik * ik, F[F_HC2] * ik * ik);
        const u32 G3 = h2_pack(F[F_HC3] * ik, F[F_HC3] * ik), HL = h2_pack(hlim * kk, hlim * kk);
        const u32 i1 = t & 1u;
        const u32 sel_ms = (t & 2u) ? ((4u + i1) | (i1 << 4) | ((6u + i1) << 8) | ((2u + i1) << 12))
                                    : (i1 | ((4u + i1) << 4) | ((2u + i1) << 8) | ((6u + i1) << 12));
        const float smid = F[F_SMID];
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += MT) {
            // threadIdx.z is 0 in every launch (block (32, HEADS, 1)), so this offset changes no address; it only keeps
            // ptxas from proving that this group reads the same slabs as the previous one (it would keep them all live)
            const u64 zs = (u64)(threadIdx.z * s0);
            const u8* const wbase = p.w + zs;
            u32 MAs[MT][2][4], MSs[MT][2], even_s[MT];
#pragma unroll
            for (u32 pass = 0; pass < 4; ++pass) {
                const u8* wexp = wbase + O[O_EXPAND] + (u64)((w * 4u + pass) * KS) * SLAB;
                float acc[MT][4][4];
                ZERO_ACC(acc, MT, 4)
#pragma unroll
                for (u32 ks = 0; ks < KS; ++ks) {
                    AHEAD_STEP(wbase + O[O_EXPAND] + (u64)(w * 4u * KS) * SLAB, pass * KS + ks, 4u * KS, wbase + O[O_MID] + (u64)(w * 2u) * SLAB, 2u)
                    const u8* sl_ = wexp + ks * SLAB;
                    STEP_OPS(sl_)
#pragma unroll
                    for (u32 d = 0; d < MT; ++d) {
                        const u32 u_ = (s0 + d) * KS + ks;
                        const V4 a_ = *(const V4*)(sm_qa + u_ * 512u + lane * 16u);
                        const u32 sa_ = *(const u32*)(sm_qa + QBYTES + u_ * 64u + g * 8u + (t & 1u) * 4u);
                        mma4(acc[d][0], a_, n01.x, n01.y, sa_, sw.x); mma4(acc[d][1], a_, n01.z, n01.w, sa_, sw.y);
                        mma4(acc[d][2], a_, n23.x, n23.y, sa_, sw.z); mma4(acc[d][3], a_, n23.z, n23.w, sa_, sw.w);
                    }
                }
#pragma unroll
                for (u32 d = 0; d < MT; ++d) {
                    u32 hv[2][4];
#pragma unroll
                    for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
                        for (u32 j = 0; j < 4; ++j) {
                            const u32 x = h2_pack(acc[d][j][rh * 2u] * kk, acc[d][j][rh * 2u + 1u] * kk);
                            const u32 c = m_clamp(x, HL);
                            hv[rh][j] = m_mul(x, h2_fma(c, h2_fma(m_abs(c), G1, G2), G3));
                        }
                    u32 w0, w1;
                    const u32 codes = m_pack2_h(hv[0], hv[1], 1.0f, w0, w1);
                    MAs[d][pass >> 1][(pass & 1u) * 2u] = w0;
                    MAs[d][pass >> 1][(pass & 1u) * 2u + 1u] = w1;
                    if (pass & 1u) {
                        const u32 mine = m_prmt(even_s[d], codes, 0x5410u);
                        MSs[d][pass >> 1] = m_prmt(mine, __shfl_xor_sync(0xffffffffu, mine, 2), sel_ms);
                    } else {
                        even_s[d] = codes;
                    }
                }
            }
            // the middle projection of these tiles
            const u8* wmid = wbase + O[O_MID] + (u64)(w * 2u) * SLAB;
            float acc[MT][4][4];
            ZERO_ACC(acc, MT, 4)
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) {
                // (the next tile group reads the expansion slabs again; after the last group comes the contraction)
                AHEAD_STEP(wmid, ks, 2u, s0 + MT < 4u ? p.w + O[O_EXPAND] + (u64)(w * 4u * KS) * SLAB : p.w + O[O_CONTRACT] + (u64)(w * KS) * SLAB, s0 + MT < 4u ? 4u * KS : KS)
                const u8* sl_ = wmid + ks * SLAB;
                STEP_OPS(sl_)
#pragma unroll
                for (u32 d = 0; d < MT; ++d) {
                    const V4 a_ = {MAs[d][ks][0], MAs[d][ks][1], MAs[d][ks][2], MAs[d][ks][3]};
                    mma4(acc[d][0], a_, n01.x, n01.y, MSs[d][ks], sw.x); mma4(acc[d][1], a_, n01.z, n01.w, MSs[d][ks], sw.y);
                    mma4(acc[d][2], a_, n23.x, n23.y, MSs[d][ks], sw.z); mma4(acc[d][3], a_, n23.z, n23.w, MSs[d][ks], sw.w);
                }
            }
#pragma unroll
            for (u32 d = 0; d < MT; ++d) {
                const u32 s = s0 + d;
                float v[2][8], a[2];
#pragma unroll
                for (u32 rh = 0; rh < 2; ++rh) {
#pragma unroll
                    for (u32 j = 0; j < 4; ++j)
#pragma unroll
                        for (u32 e = 0; e < 2; ++e) v[rh][2 * j + e] = acc[d][j][rh * 2u + e];
                    a[rh] = m_amax8x(v[rh]);
                    a[rh] = fabsf(m_xamaxf(a[rh], __shfl_xor_sync(0xffffffffu, a[rh], 1))) * smid;
                }
                float s0v, s1v, r0, r1;
                const u32 codes = m_scale2(a[0], a[1], s0v, s1v, r0, r1);
                r0 *= smid;
                r1 *= smid;
#pragma unroll
                for (u32 i = 0; i < 8; ++i) { v[0][i] *= r0; v[1][i] *= r1; }
                const u32 n0 = e2m1x8(v[0]), n1 = e2m1x8(v[1]);
                *(u64*)(sm_qb + (s * KS + (w >> 1)) * 512u + lane * 16u + (w & 1u) * 8u) = (u64)n0 | ((u64)n1 << 32);
                m_put_scale(sm_qb + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes);
            }
        }
    }
    NR_BARRIER();
#else
    // ---- MLP: expansion (four passes of 32 hidden units), activation, the hidden layer as a 4-bit operand -------
    u32 MA[2][4][4];                                            // [k64 step of the 128 hidden][tile][word]
    u32 MS[2][4];                                               // the lane's scale word of each
    {
        // The activation on float16 pairs. The accumulator is first brought into float16 range by k = 2^(2 - floor(log2
        // hlim)) (exact), and k is folded into the clamp and the coefficients: x' = x k, c' = clamp(x', +-hlim k),
        // value = x' (c' (|c'| hc1 / k^3 + hc2 / k^2) + hc3 / k), each operation rounded to float16.
        const float hlim = F[F_HLIM];
        const u32 eb = (__float_as_uint(hlim) >> 23) & 0xFFu;
        const float kk = __uint_as_float((256u - eb) << 23), ik = __uint_as_float((eb - 2u) << 23);
        const u32 G1 = h2_pack(F[F_HC1] * ik * ik * ik, F[F_HC1] * ik * ik * ik), G2 = h2_pack(F[F_HC2] * ik * ik, F[F_HC2] * ik * ik);
        const u32 G3 = h2_pack(F[F_HC3] * ik, F[F_HC3] * ik), HL = h2_pack(hlim * kk, hlim * kk);
        // the scale word of a tile: bytes (pass 0 | 2 of the k64 step) x (block t >> 1 = 0, 1) of token g + 8 (t & 1)
        const u32 i1 = t & 1u;
        const u32 sel_ms = (t & 2u) ? ((4u + i1) | (i1 << 4) | ((6u + i1) << 8) | ((2u + i1) << 12))
                                    : (i1 | ((4u + i1) << 4) | ((2u + i1) << 8) | ((6u + i1) << 12));
        u32 even[4];                                            // code pairs of the first half of each k64 step
#pragma unroll
        for (u32 pass = 0; pass < 4; ++pass) {
            const u8* wexp = p.w + O[O_EXPAND] + (u64)((w * 4u + pass) * KS) * SLAB;
#pragma unroll
            for (u32 s0 = 0; s0 < 4; s0 += TPP) {
                float acc[TPP][4][4];
                ZERO_ACC(acc, TPP, 4)
#pragma unroll
                for (u32 ks = 0; ks < KS; ++ks) {
                    AHEAD_STEP(p.w + O[O_EXPAND] + (u64)(w * 4u * KS) * SLAB, pass * KS + ks, 4u * KS, p.w + O[O_MID] + (u64)(w * 2u) * SLAB, HEADS > 1 ? 2u : 0u)
                    STEP(acc, sm_qa, ks, wexp + ks * SLAB, s0)
                }
#pragma unroll
                for (u32 d = 0; d < TPP; ++d) {
                    const u32 s = s0 + d;
                    u32 hv[2][4];
#pragma unroll
                    for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
                        for (u32 j = 0; j < 4; ++j) {
                            const u32 x = h2_pack(acc[d][j][rh * 2u] * kk, acc[d][j][rh * 2u + 1u] * kk);
                            const u32 c = m_clamp(x, HL);
                            hv[rh][j] = m_mul(x, h2_fma(c, h2_fma(m_abs(c), G1, G2), G3));
                        }
                    u32 w0, w1;
                    const u32 codes = m_pack2_h(hv[0], hv[1], 1.0f, w0, w1);
                    MA[pass >> 1][s][(pass & 1u) * 2u] = w0;                    // pass = 2 * (k64 step) + half
                    MA[pass >> 1][s][(pass & 1u) * 2u + 1u] = w1;
                    if (pass & 1u) {
                        const u32 mine = m_prmt(even[s], codes, 0x5410u);       // (pass even: g, g + 8, pass odd: g, g + 8)
                        MS[pass >> 1][s] = m_prmt(mine, __shfl_xor_sync(0xffffffffu, mine, 2), sel_ms);
                    } else {
                        even[s] = codes;
                    }
                }
            }
        }
    }

    // ---- the 128 hidden units of the head -> 32: the grouped middle projection (several heads) or the contraction
#if HEADS > 1
    {
        const u8* wmid = p.w + O[O_MID] + (u64)(w * 2u) * SLAB;
        const float smid = F[F_SMID];
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += TPP) {
            float acc[TPP][4][4];
            ZERO_ACC(acc, TPP, 4)
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) {
                AHEAD_STEP(wmid, ks, 2u, p.w + O[O_CONTRACT] + (u64)(w * KS) * SLAB, KS)
                STEP_HIDDEN(acc, ks, wmid + ks * SLAB, s0)
            }
#pragma unroll
            for (u32 d = 0; d < TPP; ++d) {
                const u32 s = s0 + d;
                // the tensor scale smid goes into the row factor: u = acc * (smid / scale), the maximum is that of
                // |acc| times smid (the same as the maximum of the products)
                float v[2][8], a[2];
#pragma unroll
                for (u32 rh = 0; rh < 2; ++rh) {
#pragma unroll
                    for (u32 j = 0; j < 4; ++j)
#pragma unroll
                        for (u32 e = 0; e < 2; ++e) v[rh][2 * j + e] = acc[d][j][rh * 2u + e];
                    a[rh] = m_amax8x(v[rh]);
                    a[rh] = fabsf(m_xamaxf(a[rh], __shfl_xor_sync(0xffffffffu, a[rh], 1))) * smid;
                }
                float s0v, s1v, r0, r1;
                const u32 codes = m_scale2(a[0], a[1], s0v, s1v, r0, r1);
                r0 *= smid;
                r1 *= smid;
#pragma unroll
                for (u32 i = 0; i < 8; ++i) { v[0][i] *= r0; v[1][i] *= r1; }
                const u32 n0 = e2m1x8(v[0]), n1 = e2m1x8(v[1]);
                *(u64*)(sm_qb + (s * KS + (w >> 1)) * 512u + lane * 16u + (w & 1u) * 8u) = (u64)n0 | ((u64)n1 << 32);
                m_put_scale(sm_qb + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes);
            }
        }
    }
    NR_BARRIER();
#endif
#endif  // REG_MLP
    {
        // y = acc * s + rs * x; stored for the attention's residual and packed for QKV
        const float* const rs = (const float*)(p.w + O[O_RS] + (w * 4u + t) * 32u);
        u16 rsh[8];                                             // rs as float16 (the table is float16-exact)
        m_split(h2_pack(rs[0], rs[1]), rsh[0], rsh[1]); m_split(h2_pack(rs[2], rs[3]), rsh[2], rsh[3]);
        m_split(h2_pack(rs[4], rs[5]), rsh[4], rsh[5]); m_split(h2_pack(rs[6], rs[7]), rsh[6], rsh[7]);
        const float scon = F[F_SCON];
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
            for (u32 ks = 0; ks < KS; ++ks) {
                AHEAD_STEP(wcon, ks, KS, p.w + O[O_QKV] + (u64)((w * 3u + 1u) * KS) * SLAB, 2u * KS)
                STEP(acc, sm_qb, ks, wcon + ks * SLAB, s0)
            }
#else
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) STEP_HIDDEN(acc, ks, wcon + ks * SLAB, s0)
#endif
#pragma unroll
            for (u32 d = 0; d < TPP; ++d) {
                const u32 s = s0 + d;
                u32 yh[2][4];                                   // y as float16 pairs
#if HEADS > 1
                const V4 xk = *(const V4*)(sm_x + (s * HEADS + w) * 512u + lane * 16u);
                const u32 xwords[4] = {xk.x, xk.y, xk.z, xk.w};
                u32 ywords[4];
#endif
#pragma unroll
                for (u32 rh = 0; rh < 2; ++rh) {
                    u32 xh[4];
#if HEADS > 1
                    m_e4m3x4_to_h2(xwords[rh], xh[0], xh[1]);
                    m_e4m3x4_to_h2(xwords[2u + rh], xh[2], xh[3]);
#else
                    const V4 wide = *(const V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u);
                    xh[0] = wide.x; xh[1] = wide.y; xh[2] = wide.z; xh[3] = wide.w;
#endif
                    float rx[8];                                // rs * x, exact in float32
                    m_mix8(xh, rsh, rx);
#pragma unroll
                    for (u32 j = 0; j < 4; ++j)
                        yh[rh][j] = h2_pack(fmaf(acc[d][j][rh * 2u], scon, rx[2 * j]), fmaf(acc[d][j][rh * 2u + 1u], scon, rx[2 * j + 1]));
#if HEADS > 1
                    ywords[rh] = m_h2x2_to_e4m3x4(yh[rh][0], yh[rh][1]);
                    ywords[2u + rh] = m_h2x2_to_e4m3x4(yh[rh][2], yh[rh][3]);
#else
                    {   // one head: the attention's residual is the unrounded (float16) y
                        const V4 wide16 = {yh[rh][0], yh[rh][1], yh[rh][2], yh[rh][3]};
                        *(V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u) = wide16;
                    }
#endif
                }
                // QKV's operand packs float16(y) at the tensor scale ty
#if HEADS == 1
                {
                    u32 n0, n1, e0, e1, codes2;
                    const u32 codes = m_pack2_h_two(yh[0], yh[1], F[F_TY], n0, n1, e0, e1, codes2, (p.options & OPT_TWO_Y) != 0u);
                    const V4 q = {n0, n1, e0, e1};
                    *(V4*)(sm_qa + s * 512u + lane * 16u) = q;
                    m_put_scale(sm_qa + QBYTES + s * 64u, g, t, 0u, codes);
                    m_put_scale(sm_qa + QBYTES + s * 64u, g, t, 1u, codes2);
                }
#else
                {
                    u32 n0, n1;
                    const u32 codes = m_pack2_h(yh[0], yh[1], F[F_TY], n0, n1);
                    *(u64*)(sm_qa + (s * KS + (w >> 1)) * 512u + lane * 16u + (w & 1u) * 8u) = (u64)n0 | ((u64)n1 << 32);
                    m_put_scale(sm_qa + QBYTES + (s * KS + (w >> 1)) * 64u, g, t, w & 1u, codes);
                    const V4 keep = {ywords[0], ywords[1], ywords[2], ywords[3]};
                    *(V4*)(sm_x + (s * HEADS + w) * 512u + lane * 16u) = keep;
                }
#endif
            }
        }
    }
    NR_BARRIER();
#endif  // DBG_STAGE != 2

#if DBG_STAGE == 0
    // ---- attention ---------------------------------------------------------------------------------------------
    // Head dimension d = 8 j + 2 t + e sits at K = 16 (j >> 1) + 4 t + 2 (j & 1) + e of the FP8 instruction, key token
    // (tile, m) at k32 step tile >> 1, K = 16 (tile & 1) + 4 ((m & 7) >> 1) + 2 (m >> 3) + (m & 1): again every lane
    // builds its own operand bytes; only V has to be transposed.
    // Order: K and V of the whole window first, then Q ATT2_QT query tiles at a time, each followed at once by its
    // logits, softmax, P.V and context (Q of a tile is used by that tile only, so it never has to be held for all four).
    const bool odd = (t & 1u) != 0u;                            // in the lane-pair reductions: this lane finishes row 1
    const float sqkv = F[F_SQKV];
    const float sq2 = sqkv * sqkv;                              // the squared norm is taken on the raw accumulators
    u32 KB[8][2];                                               // K, B operand per 8 key tokens (tile * 2 + half)
    u32 VB[4][2][2];                                            // V, B operand [8 dims][k32 step]
#pragma unroll
    for (u32 which = 1; which < 3; ++which) {
        const u8* wq = p.w + O[O_QKV] + (u64)((w * 3u + which) * KS) * SLAB;
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += ATT2_KVT) {
            float acc[ATT2_KVT][4][4];
            ZERO_ACC(acc, ATT2_KVT, 4)
#pragma unroll
            for (u32 ks = 0; ks < KS; ++ks) {
                AHEAD_STEP(p.w + O[O_QKV] + (u64)((w * 3u + 1u) * KS) * SLAB, (which - 1u) * KS + ks, 2u * KS, p.w + O[O_QKV] + (u64)((w * 3u) * KS) * SLAB, KS)
                ATT2_STEPN(acc, sm_qa, ks, wq + ks * SLAB, s0, ATT2_KVT)
            }
#pragma unroll
            for (u32 d = 0; d < ATT2_KVT; ++d) {
                const u32 s = s0 + d;
                if (which == 1u) att2_qk_rows(acc[d], sq2, sqkv, odd, KB[s * 2u][0], KB[s * 2u][1], KB[s * 2u + 1u][0], KB[s * 2u + 1u][1]);
                else att2_v_rows(acc[d], sqkv, VB[0][s >> 1][s & 1u], VB[1][s >> 1][s & 1u], VB[2][s >> 1][s & 1u], VB[3][s >> 1][s & 1u]);
            }
        }
    }
    {
        const float fq = sqkv * ((const float*)(p.w + O[O_HEADSCALE]))[w];
        const u8* const wqq = p.w + O[O_QKV] + (u64)((w * 3u) * KS) * SLAB;
        const u8* const bias = p.w + O[O_BIAS] + (u64)w * 8192u + lane * 16u;
        const float tc = F[F_TC];
        AHEAD_RANGE(p.w + O[O_BIAS] + (u64)w * 8192u, 2048u * ATT2_QT)                    // the bias of the first query tiles
#if PIPE
        W3 pipe_q[KS];                                          // Q's operand words, kept for the later query tiles
#endif
#pragma unroll
        for (u32 q0 = 0; q0 < 4; q0 += ATT2_QT) {
            u32 QA[ATT2_QT][4];                                 // Q, A operand of query tiles q0 ..
#if PIPE
            V4 pipe_b[ATT2_QT][4];                              // the bias words of these query tiles, loaded before their Q
#if PIPE_CHECK
            const u8* pipe_ba[ATT2_QT][4];
#endif
#pragma unroll
            for (u32 dq = 0; dq < ATT2_QT; ++dq)
#pragma unroll
                for (u32 kt = 0; kt < 4; ++kt) {
                    pipe_b[dq][kt] = pipe_ld(bias + ((q0 + dq) * 4u + kt) * 512u);
                    PIPE_AT(pipe_ba[dq][kt], bias + ((q0 + dq) * 4u + kt) * 512u)
                }
#endif
#if AHEAD                                                       // the next tiles' bias; before the last tiles, the projection's first slabs
            if (q0 + ATT2_QT < 4u) AHEAD_RANGE(p.w + O[O_BIAS] + (u64)w * 8192u + (q0 + ATT2_QT) * 2048u, 2048u * ATT2_QT)
            else AHEAD_FIRST(p.w + O[O_PROJ] + (u64)(w * KS) * SLAB, KS)
#endif
            {
                float acc[ATT2_QT][4][4];
                ZERO_ACC(acc, ATT2_QT, 4)
#pragma unroll
                for (u32 ks = 0; ks < KS; ++ks) {
#if PIPE
                    if (q0 == 0u) { AHEAD_STEP(wqq, ks, KS, wqq, 0u) pipe_q[ks] = pipe_cur; }
                    else pipe_cur = pipe_q[ks];
#elif STAGE
                    if (q0 == 0u) { AHEAD_STEP(wqq, ks, KS, wqq, 0u) st_qmode = 1u; }
                    else st_qmode = 2u;
                    st_qi = ks;
#else
                    if (q0 == 0u) AHEAD_STEP(wqq, ks, KS, wqq, 0u)      // (the later tiles' Q weights are not loaded again)
#endif
                    ATT2_STEPN(acc, sm_qa, ks, wqq + ks * SLAB, q0, ATT2_QT)
#if STAGE
                    st_qmode = 0u;
#endif
                }
#pragma unroll
                for (u32 d = 0; d < ATT2_QT; ++d) att2_qk_rows(acc[d], sq2, fq, odd, QA[d][0], QA[d][2], QA[d][1], QA[d][3]);
            }
#pragma unroll
            for (u32 dq = 0; dq < ATT2_QT; ++dq) {
                const u32 qs = q0 + dq;
#if STAGE
                if (qs == 3u) AHEAD_FIRST(p.w + O[O_PROJ] + (u64)(w * KS) * SLAB, KS)       // the output projection's first slabs, a query tile early
#endif
                u32 lacc[8][2];
#pragma unroll
                for (u32 kt = 0; kt < 4; ++kt) {
#if PIPE
                    const V4 b = pipe_b[dq][kt];
                    PIPE_SAME(pipe_ba[dq][kt], bias + (qs * 4u + kt) * 512u)
#elif ONE_SLAB
                    const V4 b = *(const V4*)(p.w + O[O_BIAS] + lane * 16u + kt * 512u + (u64)(one_zero * (w * 8192u + qs * 2048u)));
#else
                    const V4 b = *(const V4*)(bias + (qs * 4u + kt) * 512u);
#endif
                    lacc[kt * 2u][0] = b.x; lacc[kt * 2u][1] = b.y; lacc[kt * 2u + 1u][0] = b.z; lacc[kt * 2u + 1u][1] = b.w;
                }
#pragma unroll
                for (u32 nt = 0; nt < 8; ++nt) att2_mma8(lacc[nt], QA[dq], KB[nt][0], KB[nt][1]);
                // exp as the runtime builds it: an affine ramp, clamped, whose mantissa becomes exponent and mantissa
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
                // the four lanes t, rows g and g + 8 together (the same additions in the same order as one row at a time)
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
                        PA[ks][r] = att_h2x2_to_e4m3x4(h2_mul(lacc[4 * ks][r], total[r]), h2_mul(lacc[4 * ks + 1][r], total[r]));
                        PA[ks][2u + r] = att_h2x2_to_e4m3x4(h2_mul(lacc[4 * ks + 2][r], total[r]), h2_mul(lacc[4 * ks + 3][r], total[r]));
                    }
                u32 cacc[4][2];
#pragma unroll
                for (u32 j = 0; j < 4; ++j) {
                    cacc[j][0] = cacc[j][1] = 0u;
#pragma unroll
                    for (u32 ks = 0; ks < 2; ++ks) att2_mma8(cacc[j], PA[ks], VB[j][ks][0], VB[j][ks][1]);
                }
                // the context of tile qs (float16 pairs) as the 4-bit operand of the output projection at the tensor
                // scale tc: both rows at once, u = c * float16(tc / scale) (the MLP half's m_pack2_h)
                {
#if PIPE
                    if (qs == 3u) PIPE_LOAD(pipe_nxt, p.w + O[O_PROJ] + (u64)(w * KS) * SLAB)         // the output projection's first slab
#endif
                    const u32 h0[4] = {cacc[0][0], cacc[1][0], cacc[2][0], cacc[3][0]};
                    const u32 h1[4] = {cacc[0][1], cacc[1][1], cacc[2][1], cacc[3][1]};
#if HEADS == 1
                    u32 n0, n1, e0, e1, codes2;
                    const u32 codes = m_pack2_h_two(h0, h1, tc, n0, n1, e0, e1, codes2, (p.options & OPT_TWO_C) != 0u);
                    const V4 q = {n0, n1, e0, e1};
                    *(V4*)(sm_qb + qs * 512u + lane * 16u) = q;
                    m_put_scale(sm_qb + QBYTES + qs * 64u, g, t, 0u, codes);
                    m_put_scale(sm_qb + QBYTES + qs * 64u, g, t, 1u, codes2);
#else
                    u32 n0, n1;
                    const u32 codes = m_pack2_h(h0, h1, tc, n0, n1);
                    *(u64*)(sm_qb + (qs * KS + (w >> 1)) * 512u + lane * 16u + (w & 1u) * 8u) = (u64)n0 | ((u64)n1 << 32);
                    m_put_scale(sm_qb + QBYTES + (qs * KS + (w >> 1)) * 64u, g, t, w & 1u, codes);
#endif
                }
            }
        }
    }
    NR_BARRIER();
#endif  // DBG_STAGE == 0

    // ---- output projection, residual, stores ------------------------------------------------------------------
    {
#if VARIANT == 2
        u32 o16[4][2][4];                                       // the unrounded output as float16 pairs, for the pooling
#endif
#if DBG_STAGE == 0
        const u8* wproj = p.w + O[O_PROJ] + (u64)(w * KS) * SLAB;
        const float* const ars = (const float*)(p.w + O[O_ARS] + (w * 4u + t) * 32u);
        u16 arh[8];                                             // ars as float16 (the table is float16-exact)
        m_split(h2_pack(ars[0], ars[1]), arh[0], arh[1]); m_split(h2_pack(ars[2], ars[3]), arh[2], arh[3]);
        m_split(h2_pack(ars[4], ars[5]), arh[4], arh[5]); m_split(h2_pack(ars[6], ars[7]), arh[6], arh[7]);
        const float sproj = F[F_SPROJ];
#endif
#pragma unroll
        for (u32 s0 = 0; s0 < 4; s0 += TPP) {
#if DBG_STAGE == 0
            float acc[TPP][4][4];
            ZERO_ACC(acc, TPP, 4)
#pragma unroll
            for (u32 ks = 0; ks < KS; ++ks) {
                AHEAD_STEP(wproj, ks, KS, wproj, 0u)
                ATT2_STEP(acc, sm_qb, ks, wproj + ks * SLAB, s0)
            }
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
                    float o[8];
                    u32 yh[4];                                  // y as float16 pairs (exact)
#if HEADS > 1
                    m_e4m3x4_to_h2(ywords[rh], yh[0], yh[1]);
                    m_e4m3x4_to_h2(ywords[2u + rh], yh[2], yh[3]);
#else
                    {
                        const V4 wide = *(const V4*)(sm_x + ((s * 2u + rh) * 32u + lane) * 16u);
                        yh[0] = wide.x; yh[1] = wide.y; yh[2] = wide.z; yh[3] = wide.w;
                    }
#endif
#if DBG_STAGE == 0
                    float ry[8];                                // ars * y, exact in float32
                    m_mix8(yh, arh, ry);
#pragma unroll
                    for (u32 i = 0; i < 8; ++i) o[i] = fmaf(acc[d][i >> 1][rh * 2u + (i & 1u)], sproj, ry[i]);
#else
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) h2_unpack(yh[j], o[2 * j], o[2 * j + 1]);
#endif
#if VARIANT == 2
                    // the float16 pairs serve both the stored codes and the pooling
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) o16[s][rh][j] = h2_pack(o[2 * j], o[2 * j + 1]);
                    owords[rh] = att_h2x2_to_e4m3x4(o16[s][rh][0], o16[s][rh][1]);
                    owords[2u + rh] = att_h2x2_to_e4m3x4(o16[s][rh][2], o16[s][rh][3]);
#elif ATT2_OUT_SINGLE
                    owords[rh] = att_e4m3x4(o[0], o[1], o[2], o[3]);
                    owords[2u + rh] = att_e4m3x4(o[4], o[5], o[6], o[7]);
#else
                    owords[rh] = f32x4_to_e4m3x4(o[0], o[1], o[2], o[3]);
                    owords[2u + rh] = f32x4_to_e4m3x4(o[4], o[5], o[6], o[7]);
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
        // ci = bit 2 of the lane + 2 * bit 4, the same bits the two butterfly steps cross: each step keeps the tiles of
        // this lane's side and sends the others (the same additions in the same order as summing every tile).
        const u32 ci = (g & 1u) + 2u * (g >> 2);
        const bool b2 = (g & 1u) != 0u, b4 = (g >> 2) != 0u;
        u32 pool[2][4];
#pragma unroll
        for (u32 rh = 0; rh < 2; ++rh)
#pragma unroll
            for (u32 j = 0; j < 4; ++j) {
                u32 k0 = b2 ? o16[1][rh][j] : o16[0][rh][j];   // tile b2
                u32 k1 = b2 ? o16[3][rh][j] : o16[2][rh][j];   // tile 2 + b2
                const u32 s0 = b2 ? o16[0][rh][j] : o16[1][rh][j];
                const u32 s1 = b2 ? o16[2][rh][j] : o16[3][rh][j];
                k0 = h2_add(k0, __shfl_xor_sync(0xffffffffu, s0, 4));
                k1 = h2_add(k1, __shfl_xor_sync(0xffffffffu, s1, 4));
                u32 a = h2_add(b4 ? k1 : k0, __shfl_xor_sync(0xffffffffu, b4 ? k0 : k1, 16));
                pool[rh][j] = h2_mul(a, 0x34003400u);           // * 0.25
            }
        {   // the pooled map (float16 pairs) at the tensor scale tr, both rows at once (m_pack2_h)
            const float tr = F[F_TR];
#if HEADS == 1
            u32 n0, n1, e0, e1, codes2;
            const u32 codes = m_pack2_h_two(pool[0], pool[1], tr, n0, n1, e0, e1, codes2, (p.options & OPT_TWO_R) != 0u);
            const V4 q = {n0, n1, e0, e1};
            *(V4*)(sm_qa + lane * 16u) = q;
            m_put_scale(sm_qa + QBYTES, g, t, 0u, codes);
            m_put_scale(sm_qa + QBYTES, g, t, 1u, codes2);
#else
            u32 n0, n1;
            const u32 codes = m_pack2_h(pool[0], pool[1], tr, n0, n1);
            *(u64*)(sm_qa + (w >> 1) * 512u + lane * 16u + (w & 1u) * 8u) = (u64)n0 | ((u64)n1 << 32);
            m_put_scale(sm_qa + QBYTES + (w >> 1) * 64u, g, t, w & 1u, codes);
#endif
        }
#if PIPE
        PIPE_LOAD(pipe_nxt, p.w + O[O_RES] + (u64)(w * 2u * KS) * SLAB)
#elif STAGE
        if (st_n == 0u) {                                       // the first two steps: (step 0, half 0), (step 0, half 1)
            ST_ISSUE(p.w + O[O_RES] + (u64)(w * 2u * KS) * SLAB, st_k)
            ST_ISSUE(p.w + O[O_RES] + (u64)(w * 2u * KS + KS) * SLAB, st_k + 1u)
            st_n = 2u;
        }
#else
        AHEAD_FIRST(p.w + O[O_RES] + (u64)(w * 2u * KS) * SLAB, KS)
        AHEAD_FIRST(p.w + O[O_RES] + (u64)(w * 2u * KS + KS) * SLAB, KS)
#endif
        NR_BARRIER();
        float racc[2][4][4];
        ZERO_ACC(racc, 2, 4)
        const u8* wres = p.w + O[O_RES] + (u64)(w * 2u * KS) * SLAB;
#pragma unroll
        for (u32 ks = 0; ks < KS; ++ks) {
#if !PIPE && !STAGE
            AHEAD_STEP(wres, ks, KS, wres, 0u)
            AHEAD_STEP(wres + (u64)KS * SLAB, ks, KS, wres, 0u)
#endif
            const V4 a = *(const V4*)(sm_qa + ks * 512u + lane * 16u);
            const u32 sa = *(const u32*)(sm_qa + QBYTES + ks * 64u + g * 8u + (t & 1u) * 4u);
#pragma unroll
            for (u32 half = 0; half < 2; ++half) {
                const u8* sl = wres + (half * KS + ks) * SLAB;
#if PIPE
                pipe_cur = pipe_nxt;                            // the slabs are read in the order (step, half): step 0 half 0, step 0 half 1, ...
                if (ks * 2u + half + 1u < 2u * KS)
                    PIPE_LOAD(pipe_nxt, wres + (u64)(((ks * 2u + half + 1u) & 1u) * KS + ((ks * 2u + half + 1u) >> 1)) * SLAB)
#elif STAGE
                {
                    const u32 st_j_ = ks * 2u + half;           // the slabs are read in the order (step, half)
                    st_has1 = st_j_ + 1u < 2u * KS; st_has2 = st_j_ + 2u < 2u * KS;
                    st_a1 = wres + (u64)(((st_j_ + 1u) & 1u) * KS + ((st_j_ + 1u) >> 1)) * SLAB;
                    st_a2 = wres + (u64)(((st_j_ + 2u) & 1u) * KS + ((st_j_ + 2u) >> 1)) * SLAB;
                    st_armed = true;
                }
#endif
                STEP_OPS(sl)
                att2_mma4(racc[half][0], a, n01.x, n01.y, sa, sw.x); att2_mma4(racc[half][1], a, n01.z, n01.w, sa, sw.y);
                att2_mma4(racc[half][2], a, n23.x, n23.y, sa, sw.z); att2_mma4(racc[half][3], a, n23.z, n23.w, sa, sw.w);
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
        NR_BARRIER();
#if WINS > 1                                                    // one flag per window, at the runtime's index (its grid is one CTA per window)
        if (leader) flag_publish(p.oflags + blockIdx.y * (u32)((p.W - min(p.sx, 0) + 7) >> 3) + (u32)win_x, 0);
#elif defined(CTA_ORD) && CTA_ORD                               // (the index through chain_flag() changes the release code)
        if (leader) flag_publish(p.oflags + chain_flag(), 0);
#else
        if (leader) flag_publish(p.oflags + blockIdx.y * gridDim.x + blockIdx.x, 0);
#endif
    }
    chain_trace(p, 2u);
}

// =====================================================================================================================
// Chain experiments (v2/swin-spec/CHAIN.md). Every switch defaults to the release kernel: with all of them unset (or at
// the values below)
// the 20 cubins are byte-identical to v2/assets/swin4 (checked by `swin_chain.py hash`). None changes the arithmetic or
// the flags a window publishes; they change which window a CTA index takes, how the producer flags are polled, what a
// CTA does while it waits, and (lab builds only) a per-CTA timeline. Defined after the kernel (declared on the line
// before it) so that the lines prepost_make_cu.py copies keep their numbers and prepost4.cu its text.
//
// CTA_ORD   which 8x8 window CTA index (blockIdx.x, blockIdx.y) takes. The hardware hands out CTAs in linear order
//           l = blockIdx.y * gridDim.x + blockIdx.x; the window and its flag index follow the remapped (bx, by), so the
//           set of flags and each window's arithmetic are the release kernel's, only the time order differs.
//           0: (bx, by) = blockIdx (row-major, the runtime's order)   1: column-major (y fastest)
//           2: reversed row-major (last window first)                 3: bands of two window rows, column-major inside
//           Only 0 is safe next to the runtime's row-major kernels in the time sense: 1 and 2 make a row-major consumer's
//           first CTAs wait for this kernel's last ones (they are the negative controls of the chain timing).
// WAIT_PAR  1: the producer flags under the window (1, 2 or 4) are polled at once, one per lane of warp 0, each lane on
//           its own wait budget, instead of one after another by thread 0 (each read of a set flag costs ~0.3 us).
// WAIT_NAP  with WAIT_PAR: the sleep between two polls of an unset flag in ns (64 as nap(); 0 = poll back to back).
// PRE_WAIT  1: before the wait, each warp prefetches its first expansion slabs (pass 0, all k64 steps) into L1 with
//           prefetch.global.L1, so the first pass after the wait does not start on L2 latency. No effect on results.
// NR_TRACE  1 (lab builds only, never exported): per CTA four u64 at out + map bytes + 65536 + 32 * l, l the hardware
//           linear index: [0] globaltimer at the start, [1] after the producer wait and its barrier (= start when the
//           kernel does not wait), [2] after the stores and the flag publish, [3] SM id | (window index << 32). Map
//           bytes = C * H * W (VARIANT 4: C * d0 * d1). The lab allocates the room (swin_chain.py ctrace).
#ifndef CTA_ORD
#define CTA_ORD 0
#endif
#ifndef WAIT_PAR
#define WAIT_PAR 0
#endif
#ifndef WAIT_NAP
#define WAIT_NAP 64
#endif
#ifndef PRE_WAIT
#define PRE_WAIT 0
#endif
#ifndef NR_TRACE
#define NR_TRACE 0
#endif
#if CTA_ORD < 0 || CTA_ORD > 3
#error CTA_ORD must be 0..3
#endif

// window of this CTA along `axis` (0: x, 1: y)
NR_INLINE int chain_cta(u32 axis)
{
#if CTA_ORD == 0
    return axis ? (int)blockIdx.y : (int)blockIdx.x;
#else
    const u32 X = gridDim.x, Y = gridDim.y, l = blockIdx.y * X + blockIdx.x;
    u32 bx, by;
#if CTA_ORD == 1
    bx = l / Y; by = l - bx * Y;
#elif CTA_ORD == 2
    const u32 r = X * Y - 1u - l;
    by = r / X; bx = r - by * X;
#else
    const u32 band = l / (2u * X), first = 2u * band, rows = min(2u, Y - first), in = l - band * 2u * X;
    bx = in / rows; by = first + (in - bx * rows);
#endif
    return axis ? (int)by : (int)bx;
#endif
}

// index of this CTA's window in the flag array (the runtime's row-major order with the runtime's grid)
NR_INLINE u32 chain_flag()
{
#if CTA_ORD == 0
    return blockIdx.y * gridDim.x + blockIdx.x;
#else
    return (u32)chain_cta(1u) * gridDim.x + (u32)chain_cta(0u);
#endif
}

NR_INLINE bool chain_wait_par() { return WAIT_PAR != 0; }

// WAIT_PAR: the producer windows under this one, flag i of the (rows x columns) block polled by lane i of warp 0
NR_INLINE void chain_wait_parallel(const Params& p, int x0, int y0)
{
#if WAIT_PAR
    if (threadIdx.y != 0u) return;
    const int gx = (p.W - min(p.psx, 0) + 7) >> 3, gy = (p.H - min(p.psy, 0) + 7) >> 3;
    const int dy = y0 - p.psy, dx = x0 - p.psx;
    const int r0 = max(dy, 0) >> 3, r1 = min(dy + 8 > 0 ? ((dy + 7) >> 3) + 1 : 0, gy);
    const int c0 = max(dx, 0) >> 3, c1 = min(dx + 8 > 0 ? ((dx + 7) >> 3) + 1 : 0, gx);
    // an 8-token window overlaps at most two 8-token producer windows per axis, so at most 4 flags: lane i takes row
    // r0 + (i >> 1), column c0 + (i & 1) of the 2 x 2 block, if that one is in the range
    const int i = (int)threadIdx.x, r = r0 + (i >> 1), c = c0 + (i & 1);
    if (i < 4 && r < r1 && c < c1) {
        const int* const flag = p.pflags + r * gx + c;
#if WAIT_NAP == 64
        int budget = WAIT_BUDGET_NS;
        wait_not_negative(flag, budget);
#else                                                           // the budget still falls by 64 per poll (a poll takes >= 0.2 us)
        for (int budget = WAIT_BUDGET_NS; budget > 0 && flag_load(flag) < 0; budget -= 64)
            if (WAIT_NAP > 0) asm volatile("nanosleep.u32 %0;" :: "n"(WAIT_NAP) : "memory");
#endif
    }
#endif
}

// PRE_WAIT: this warp's expansion slabs of pass 0 (KS x SLAB bytes, the first weights the block reads) into L1
NR_INLINE void chain_prefetch(const Params& p, u32 w, u32 lane)
{
#if PRE_WAIT
    const u32* const O = (const u32*)(p.w + 64);
    const u8* const first = p.w + O[O_EXPAND] + (u64)((w * 4u) * KS) * SLAB;
    for (u32 at = lane * 128u; at < KS * SLAB; at += 32u * 128u)
        asm volatile("prefetch.global.L1 [%0];" :: "l"((u64)(first + at)));
#endif
}

// NR_TRACE: see above
NR_INLINE void chain_trace(const Params& p, u32 slot)
{
#if NR_TRACE
    if (slot == 2u && !p.oflags) __syncthreads();               // the end of the whole CTA (with flags: the publish's barrier)
    if ((threadIdx.x | threadIdx.y) != 0u) return;
    u64 now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
#if VARIANT == 4
    const u64 map = (u64)C * (u64)(u32)p.d0 * (u64)(u32)p.d1;
#else
    const u64 map = (u64)C * (u64)(u32)p.H * (u64)(u32)p.W;
#endif
    u64* const rec = (u64*)(p.out + map + 65536u) + 4u * (blockIdx.y * gridDim.x + blockIdx.x);
    rec[slot] = now;
    if (slot == 0u) {
        u32 sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        rec[3] = (u64)sm | ((u64)chain_flag() << 32);
    }
#endif
}
