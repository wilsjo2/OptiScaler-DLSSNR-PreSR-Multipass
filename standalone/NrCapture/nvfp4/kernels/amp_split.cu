// amp_split.cu: Ampere-legal (sm_86) FP16 drop-in replacements of NVIDIA's 512-wide split Swin kernels
// (cc_split_swin_16h_*_512_*_fp8) and of cc_dec_input_upsample_1024_512_tilesync_fp8, DLSS-NR Model A, runtime 310.9.16.
//
// Phase A: every entry takes EXACTLY the runtime kernel's parameter block (same size and offsets, split-spec/abi.md
// F.1, P.1, Q.1, D.1), NVIDIA's own e4m3 weight record (unchanged bytes) and the runtime's buffers in the runtime's
// layouts (tile layout, view layout, flags, merge counter, f16 scratch). Launch each with the runtime's own grid and
// block; no dynamic shared memory. One entry per runtime kernel type:
//
//   amp_split_ffwd               cc_split_swin_16h_ffwd_512_chained_fp8              block (32,8,1)  params 56
//   amp_split_ffwd_inpview       cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8     block (32,4,1)  params 56
//   amp_split_ffwd_proj          cc_split_swin_16h_ffwd_proj_512_chained_fp8         block (32,4,1)  params 72
//   amp_split_ffwd_proj_inpview  cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8 block (32,4,1)  params 72
//   amp_split_qkv                cc_split_swin_16h_qkv_512_chained_fp8               block (32,4,1)  params 56
//   amp_split_proj               cc_split_swin_16h_proj_512_chained_fp8              block (32,8,1)  params 72
//   amp_split_proj_pool          cc_split_swin_16h_proj_pool_512_chained_fp8         block (32,4,1)  params 80
//   amp_split_final_head         cc_split_swin_16h_final_head_512_wait_fp8           block (32,8,1)  params 40
//   amp_split_proj_outview       cc_split_swin_16h_proj_512_outview_wait_fp8         block (32,4,1)  params 72
//   amp_dec_input_upsample       cc_dec_input_upsample_1024_512_tilesync_fp8         block (32,2,1)  params 80
//
// Flags: a null producer-flag pointer means "do not wait" and a null own-flag pointer "do not publish", so the same
// entry also stands in for the runtime's plain twins (the lab's plain runs pass 0 there). The runtime's chained
// launches always pass both. Waits use NVIDIA's protocol (ld.relaxed.gpu + nanosleep 64, flags reset to -1, ready >= 0)
// pipelined per producer channel half (channels 0-255, then 256-511; for a qkv producer = quarters 0,1 then 2,3), over
// the whole 8x8 window (NVIDIA waits per tile row in the 8-warp kernels: ours is a superset, never earlier), each wait
// unbounded as NVIDIA's, except that a wait over AMP_WAIT_LIMIT_NS traps (a visible GPU fault). Publishing:
// bar.sync, then thread 0 st.release.gpu 0 at NVIDIA's index. Every read of another kernel's data comes after the
// first wait, including the projections' residual (the producer's producer output, not flagged itself): the runtime's
// chained launches overlap, and a residual read at kernel start took stale data from the previous evaluation (found
// in the whole-pass check: launch 27 alone 27 dB; after the fix 54 dB, hwref build identical to NVIDIA).
//
// Arithmetic, in NVIDIA's order (abi.md F.4, P.6, Q.6, D.6):
//   * Every product: e4m3 operands decoded exactly to f16 in registers (amp_common.cuh), each k32 step of NVIDIA's
//     mma.m16n8k32.f16.e4m3.e4m3.f16 as two mma.m16n8k16.f16.f16.f16.f16 on the same registers (f16 accumulator, so the
//     running sum is rounded after every 16 inputs instead of every 32: the only intended difference). K blocks in
//     NVIDIA's order; accumulator seeds (residual * gain, position bias) as NVIDIA's.
//   * Every e4m3 result (stored maps, reduced / hidden values, Q, K, V, P, context) rounded like
//     cvt.rn.satfinite.e4m3x2.f16x2 in integer code (amp_h2_to_e4m3x2).
//   * Cubic SiLU, residual seed f16(f16(R) * gain), Q/K norms, the exp2 bit trick, the softmax denominator in NVIDIA's
//     fixed order, rcp/rsqrt.approx narrowed to f16, the 2x2 mean, the 2x upsample, the split-K merge order of dec.
//   * Where NVIDIA's PTX has an f16 mul feeding an add, NVIDIA's ptxas (default --fmad=true) fuses them; we build with
//     --fmad=false, so such places are written as explicit fma: dec's blend round(up + f16(skip) * gain) and the
//     qkv Q/K squared norms fma(n1, n1, f16(n3^2)) + fma(n0, n0, f16(n2^2)) (both found with -DAMP_HW_REF=1).
//
// dec split-K merge: NVIDIA's protocol kept as is (z = 0 stores its f16 partial, z = 1, 2 add into the scratch with
// f16x2 reductions, z = 3 adds the scratch to its own partial; each z waits for counter >= z - 1, so the merges are
// strictly ordered ((p0 + p1) + p2) + p3 and the result is deterministic). The only change: NVIDIA's
// red.global.v4.f16x2.add.noftz (sm_90) becomes four scalar red.global.add.noftz.f16x2 (sm_80); each word is still
// added exactly once, by one CTA, at its turn, so ordering and rounding are unchanged. The counter must be -1 before
// every launch (the runtime's cc_cb_clear), exactly as for NVIDIA's kernel.
//
// Test build only (never for sm_86, never shipped): -DAMP_HW_REF=1 replaces each k16 pair by NVIDIA's own k32 e4m3
// instruction on the undecoded operand words (sm_89+). With it every product is bit-identical to NVIDIA's, so any
// remaining difference against NVIDIA's kernel is a layout or non-matrix order bug (split_lab.py amp --hwref).
//
// Builds (same entries, same parameter blocks, same launch shapes):
//   base   the decided target: f16 accumulator, rounded after every 16 inputs (full tensor rate on GeForce Ampere)
//   exact  -DAMP_F32ACC=1: each k32 step as two k16 steps on an f32 accumulator, rounded to f16 once per k32 step as
//          NVIDIA's instruction does (half rate on GeForce Ampere); still Ampere-legal
// amp-variant: base
// amp-variant: exact -DAMP_F32ACC=1
#include "amp_common.cuh"

#ifndef AMP_HW_REF
#define AMP_HW_REF 0
#endif
#ifndef AMP_F32ACC
#define AMP_F32ACC 0
#endif
// Q/K squared-norm pairing of the four 8-channel fragments of a head when NORM_FMA = -1: 0 = (n1^2 + n3^2) +
// (n0^2 + n2^2) (abi.md Q.6); 1 = (n0^2 + n1^2) + (n2^2 + n3^2); 2 = (n0^2 + n3^2) + (n1^2 + n2^2). Experiments only.
#ifndef NORM_PAIRS
#define NORM_PAIRS 0
#endif
// The pairs as NVIDIA's SASS computes them (its ptxas contracts the PTX's mul.f16x2 + add.f16x2): default 0 =
// fma(n1, n1, f16(n3^2)) + fma(n0, n0, f16(n2^2)), measured bit-exact against NVIDIA's kernel on all 16 qkv layers
// with -DAMP_HW_REF=1 (the two-rounding form of abi.md Q.6 differs in up to 0.6 % of the codes, up to 179 steps).
// -1 = no contraction (then NORM_PAIRS applies); 1..3 = the other choices of which square stays a mul.
#ifndef NORM_FMA
#define NORM_FMA 0
#endif
// dec merge of slices 1, 2 (experiments only): 0 = red.global.add.noftz.f16x2 (NVIDIA's operation, scalar form);
// 1 = load, add.f16x2, store (the slices are serialised by the counter, so this is race-free too)
#ifndef DEC_MERGE
#define DEC_MERGE 0
#endif

// ---- parameter blocks (byte-exact copies of the runtime's) ---------------------------------------------------------
struct FfwdP {                      // 56 bytes: ffwd_512_chained, ffwd_inpview_512_tilesync
    const u8* in;                   // +0  input X (tile layout 512; inpview: view layout [32][H][W][16])
    u8* out;                        // +8  output F (tile layout 512)
    const u8* w;                    // +16 weight record (524288 B)
    int H, W;                       // +24 token rows, columns
    const int* pflags;              // +32 producer flags (proj / dec), 2 parts per unshifted window; inpview: unused
    int psx, psy;                   // +40 producer shift (0)
    int* oflags;                    // +48 own flags
};
struct ProjP {                      // 72 bytes: ffwd_proj (+inpview), proj, proj_outview
    const u8* in;                   // +0  X (tile layout 512)
    const u8* res;                  // +8  residual R (tile layout; ffwd_proj_inpview: view layout)
    u8* out;                        // +16 Y (tile layout; outview: view layout)
    const u8* w;                    // +24 record: e4m3 [512,512] slices, f16 gain[512] at +262144
    int H, W;                       // +32
    int unused0, unused1;           // +40 never read
    int* oflags;                    // +48 own flags (outview: not read)
    const int* pflags;              // +56 producer flags
    int psx, psy;                   // +64 producer shift
};
struct PoolP {                      // 80 bytes: proj_pool
    const u8* in;                   // +0
    const u8* res;                  // +8
    u8* out;                        // +16 full-resolution Y
    u8* pooled;                     // +24 pooled map, tile layout of prow x pcol tokens
    const u8* w;                    // +32
    const int* pflags;              // +40 qkv flags
    int* oflags;                    // +48
    int psx, psy;                   // +56
    int H, W;                       // +64
    int prow, pcol;                 // +72 roundup4(ceil(H/2)), roundup4(ceil(W/2))
};
struct HeadP {                      // 40 bytes: final_head
    const u8* in;                   // +0 pooled map (tile layout 512)
    u8* out;                        // +8 tile layout 1024
    const u8* w;                    // +16 e4m3 [1024,512] slices
    const int* pflags;              // +24 proj_pool flags
    int H, W;                       // +32 pooled rows, columns
};
struct QkvP {                       // 56 bytes: qkv_512_chained
    const u8* in;                   // +0
    u8* out;                        // +8
    const u8* w;                    // +16 qkv [1536,512] slices, bias f16 [16][4096] at 786432, f32 head scale[16] at 917504
    int H, W;                       // +24
    int sx, sy;                     // +32 own shift
    const int* pflags;              // +40 ffwd_proj flags (unshifted, 2 parts)
    int* oflags;                    // +48 own flags (4 parts per shifted window)
};
struct DecP {                       // 80 bytes: dec_input_upsample_1024_512_tilesync
    const u8* in;                   // +0  coarse map, tile layout 1024
    const u8* skip;                 // +8  tile layout 512, H x W
    u8* out;                        // +16 tile layout 512, H x W
    const u8* unused;               // +24 never read
    int* counter;                   // +32 merge counter per (coarse tile, half); -1 before the launch
    int* oflags;                    // +40 own flags
    u8* scratch;                    // +48 f16 partials [coarse tile][32][512]
    const u8* w;                    // +56 e4m3 [512,1024] slices, f16 gain[512] at 524288
    int Hc, Wc;                     // +64 coarse rows, columns
    int H, W;                       // +72 output rows, columns
};
static_assert(sizeof(FfwdP) == 56 && sizeof(ProjP) == 72 && sizeof(PoolP) == 80 && sizeof(HeadP) == 40 &&
              sizeof(QkvP) == 56 && sizeof(DecP) == 80, "parameter blocks must match the runtime's");

// ---- fragments of one NVIDIA k32 step --------------------------------------------------------------------------
#if AMP_HW_REF
struct FA { u32 r[4]; };
struct FB { u32 r[2]; };
AMP_INLINE FA make_a(u32 w0, u32 w1, u32 w2, u32 w3) { FA a; a.r[0] = w0; a.r[1] = w1; a.r[2] = w2; a.r[3] = w3; return a; }
AMP_INLINE FB make_b(u32 w0, u32 w1) { FB b; b.r[0] = w0; b.r[1] = w1; return b; }
AMP_INLINE void mma(u32* c, const FA& a, const FB& b)
{
    u32 d0, d1;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
                 : "=r"(d0), "=r"(d1) : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]), "r"(b.r[0]), "r"(b.r[1]), "r"(c[0]), "r"(c[1]));
    c[0] = d0; c[1] = d1;
}
#else
// A: k16 step 0 in r[0..3], step 1 in r[4..7]; B: step 0 in r[0..1], step 1 in r[2..3] (amp_common.cuh mapping)
struct FA { u32 r[8]; };
struct FB { u32 r[4]; };
AMP_INLINE FA make_a(u32 w0, u32 w1, u32 w2, u32 w3)
{
    FA a;
    amp_e4m3x4_to_h2x2_finite(w0, a.r[0], a.r[2]);
    amp_e4m3x4_to_h2x2_finite(w1, a.r[1], a.r[3]);
    amp_e4m3x4_to_h2x2_finite(w2, a.r[4], a.r[6]);
    amp_e4m3x4_to_h2x2_finite(w3, a.r[5], a.r[7]);
    return a;
}
AMP_INLINE FB make_b(u32 w0, u32 w1)
{
    FB b;
    amp_e4m3x4_to_h2x2_finite(w0, b.r[0], b.r[1]);
    amp_e4m3x4_to_h2x2_finite(w1, b.r[2], b.r[3]);
    return b;
}
AMP_INLINE void mma(u32* c, const FA& a, const FB& b)
{
#if AMP_F32ACC
    // 'exact' build: the two k16 halves with an f32 accumulator, rounded to f16 once per k32 step like NVIDIA's
    // instruction (half the tensor rate on GeForce Ampere)
    float f[4];
    amp_h2_unpack(c[0], f[0], f[1]);
    amp_h2_unpack(c[1], f[2], f[3]);
    amp_mma16_f32(f, a.r, b.r);
    amp_mma16_f32(f, a.r + 4, b.r + 2);
    c[0] = amp_h2_pack(f[0], f[1]);
    c[1] = amp_h2_pack(f[2], f[3]);
#else
    amp_mma16_f16(c, a.r, b.r);
    amp_mma16_f16(c, a.r + 4, b.r + 2);
#endif
}
#endif
AMP_INLINE FA make_a(const amp_v4& v) { return make_a(v.x, v.y, v.z, v.w); }

// two f16 pairs -> four e4m3 codes (lo -> bytes 0, 1)
AMP_INLINE u32 pack4(u32 lo, u32 hi) { return amp_h2_to_e4m3x2(lo) | (amp_h2_to_e4m3x2(hi) << 16); }
// four 8-output accumulators (one 32-channel block, rows g / g + 8) -> the lane's 16 bytes of that block in the tile
// layout (abi.md 3.1: word 0 = row g of fragments 0, 1; word 1 = row g + 8; words 2, 3 the same for fragments 2, 3),
// which is also NVIDIA's A operand of the next product
AMP_INLINE amp_v4 block_words(const u32 (*c)[2])
{
    amp_v4 v;
    v.x = pack4(c[0][0], c[1][0]); v.y = pack4(c[0][1], c[1][1]);
    v.z = pack4(c[2][0], c[3][0]); v.w = pack4(c[2][1], c[3][1]);
    return v;
}
AMP_INLINE FA acc_to_a(const u32 (*c)[2]) { return make_a(block_words(c)); }

AMP_INLINE amp_v4 lds16(const u8* p) { return *(const amp_v4*)p; }
AMP_INLINE void sts16_zero(u8* p) { *(amp_v4*)p = amp_v4{0u, 0u, 0u, 0u}; }
AMP_INLINE void stg16(u8* p, const amp_v4& v) { *(amp_v4*)p = v; }
AMP_INLINE u32 ldg32(const u8* p) { return __ldg((const unsigned int*)p); }
// the f16 pair of channels (2 h, 2 h + 1) of a word of four e4m3 codes, exactly as cvt.rn.f16x2.e4m3x2
AMP_INLINE u32 word_h2(u32 word, u32 h) { return amp_e4m3x2_to_h2((word >> (16u * h)) & 0xFFFFu); }
AMP_INLINE u32 word_of(const amp_v4& v, u32 i) { return i == 0u ? v.x : i == 1u ? v.y : i == 2u ? v.z : v.w; }

// ---- flags -----------------------------------------------------------------------------------------------------
// One thread waits for parts [q0, q1) of every producer window under [y0, y0 + rows) x [x0, x0 + cols); producer
// windows of 2^wl tokens on a grid shifted by (psx, psy), index (window row * parts + part) * gx + window column, gx/gy
// recomputed as NVIDIA does (abi.md 0.1, P.5, Q.5, F.3).
AMP_INLINE void wait_parts(const int* f, int parts, int q0, int q1, int wl, int H, int W, int psx, int psy, int y0, int x0,
                           int rows, int cols, int& budget)
{
    const int win = 1 << wl;
    const int gx = (W - min(psx, 0) + win - 1) >> wl, gy = (H - min(psy, 0) + win - 1) >> wl;
    const int dy = y0 - psy, dx = x0 - psx;
    const int r0 = max(dy, 0) >> wl, r1 = min(dy + rows > 0 ? ((dy + rows - 1) >> wl) + 1 : 0, gy);
    const int c0 = max(dx, 0) >> wl, c1 = min(dx + cols > 0 ? ((dx + cols - 1) >> wl) + 1 : 0, gx);
    for (int r = r0; r < r1; ++r)
        for (int q = q0; q < q1; ++q)
            for (int c = c0; c < c1; ++c) amp_wait_not_negative(f + (r * parts + q) * gx + c, budget);
}

// ---- staging -----------------------------------------------------------------------------------------------------
// Channel blocks [b0, b0 + 8) of the four tiles (ty0 + s / 2, tx0 + s % 2) of a window, tile layout with `blocks`
// 512-byte blocks per tile, into shared memory [slot][blocks][512] with cp.async (16 B, L2 only); tiles outside
// [0, TH) x [0, TW) are e4m3 zeros. Commits one group.
template <u32 NT>
AMP_INLINE void stage_window(u8* sm, const u8* map, int TH, int TW, int ty0, int tx0, u32 blocks, u32 b0, u32 tid)
{
    constexpr u32 N = 4u * 8u * 32u;                        // 16-byte chunks
#pragma unroll
    for (u32 i = tid; i < N; i += NT) {
        const u32 slot = i >> 8, blk = b0 + ((i >> 5) & 7u), l = i & 31u;
        const int ty = ty0 + (int)(slot >> 1), tx = tx0 + (int)(slot & 1u);
        u8* const dst = sm + (slot * blocks + blk) * 512u + l * 16u;
        if (ty >= 0 && ty < TH && tx >= 0 && tx < TW)
            amp_cp_async16(amp_smem(dst), map + ((u64)(u32)(ty * TW + tx) * blocks + blk) * 512u + l * 16u);
        else
            sts16_zero(dst);
    }
    amp_cp_async_commit();
}
// The same from a view-layout map [C/16][H][W][16] (inpview; the runtime orders this launch after its producer):
// word j of lane l of block b of tile (ty, tx) = view word t = l & 3 of pixel (4 ty + m / 4, 4 tx + m % 4),
// m = l / 4 + 8 (j & 1), group 2 b + j / 2 (abi.md 3.5 / P.8); pixels outside [0,H) x [0,W) are zero.
template <u32 NT>
AMP_INLINE void stage_window_view(u8* sm, const u8* map, int H, int W, int ty0, int tx0, u32 b0, u32 tid)
{
    constexpr u32 N = 4u * 8u * 32u * 4u;                   // words
#pragma unroll 4
    for (u32 i = tid; i < N; i += NT) {
        const u32 j = i & 3u, l = (i >> 2) & 31u, blk = b0 + ((i >> 7) & 7u), slot = i >> 10;
        const u32 m = (l >> 2) + 8u * (j & 1u);
        const int y = 4 * (ty0 + (int)(slot >> 1)) + (int)(m >> 2), x = 4 * (tx0 + (int)(slot & 1u)) + (int)(m & 3u);
        const u32 group = 2u * blk + (j >> 1);
        u32 v = 0u;
        if (y >= 0 && y < H && x >= 0 && x < W)
            v = ldg32(map + ((u64)(group * (u32)H + (u32)y) * (u32)W + (u32)x) * 16u + (l & 3u) * 4u);
        *(u32*)(sm + (slot * 16u + blk) * 512u + l * 16u + j * 4u) = v;
    }
}
// The lane's 16 bytes of block b of tile (ty, tx) from a view-layout map (zero outside), as stage_window_view
AMP_INLINE amp_v4 view_load(const u8* map, int H, int W, int ty, int tx, u32 b, u32 lane)
{
    u32 words[4];
#pragma unroll
    for (u32 j = 0; j < 4; ++j) {
        const u32 m = (lane >> 2) + 8u * (j & 1u);
        const int y = 4 * ty + (int)(m >> 2), x = 4 * tx + (int)(m & 3u);
        const u32 group = 2u * b + (j >> 1);
        words[j] = (y >= 0 && y < H && x >= 0 && x < W)
                       ? amp_load_cg32(map + ((u64)(group * (u32)H + (u32)y) * (u32)W + (u32)x) * 16u + (lane & 3u) * 4u) : 0u;
    }
    return amp_v4{words[0], words[1], words[2], words[3]};
}
// ... and the store (outview: st of 4 bytes per word, pixel-level bounds)
AMP_INLINE void view_store(u8* map, int H, int W, int ty, int tx, u32 b, u32 lane, const amp_v4& v)
{
#pragma unroll
    for (u32 j = 0; j < 4; ++j) {
        const u32 m = (lane >> 2) + 8u * (j & 1u);
        const int y = 4 * ty + (int)(m >> 2), x = 4 * tx + (int)(m & 3u);
        const u32 group = 2u * b + (j >> 1);
        if (y >= 0 && y < H && x >= 0 && x < W)
            *(u32*)(map + ((u64)(group * (u32)H + (u32)y) * (u32)W + (u32)x) * 16u + (lane & 3u) * 4u) = word_of(v, j);
    }
}

// one K block (32 inputs) of NT_ tiles x 8 output fragments: weights = four 512-byte slices at wk (lane's 16 bytes:
// words 0-1 fragment 2 s, words 2-3 fragment 2 s + 1), A = the lane's 16 bytes of the block of each tile in shared memory
template <u32 NT_>
AMP_INLINE void kblock8(u32 (&acc)[NT_][8][2], const u8* wk, const u8* const (&a_at)[NT_], u32 lane)
{
    FB b[8];
#pragma unroll
    for (u32 s = 0; s < 4; ++s) {
        const amp_v4 v = amp_load_nc(wk + s * 512u + lane * 16u);
        b[2 * s] = make_b(v.x, v.y);
        b[2 * s + 1] = make_b(v.z, v.w);
    }
#pragma unroll
    for (u32 t = 0; t < NT_; ++t) {
        const FA a = make_a(lds16(a_at[t] + lane * 16u));
#pragma unroll
        for (u32 nf = 0; nf < 8; ++nf) mma(acc[t][nf], a, b[nf]);
    }
}

// =====================================================================================================================
// Grouped feed-forward (abi.md F; inpview P.8). CTA = one unshifted 8x8 window x groups 4 z .. 4 z + 3.
// NT = 2: block (32,8), warp = (tile row w >> 2, group (w & 3) + 4 z); NT = 4 (inpview): block (32,4), warp = group
// w + 4 z for all four tiles.
template <u32 NT, bool VIEW>
AMP_INLINE void ffwd_body(const FfwdP& p)
{
    __shared__ __align__(128) u8 sm[4u * 16u * 512u];
    constexpr u32 NTHREADS = NT == 2u ? 256u : 128u;
    const u32 lane = threadIdx.x, w = threadIdx.y, tid = w * 32u + lane;
    const bool leader = tid == 0u;
    const int cx = (int)blockIdx.x, cy = (int)blockIdx.y, z = (int)blockIdx.z;
    const int TH = p.H >> 2, TW = p.W >> 2, ty0 = 2 * cy, tx0 = 2 * cx;
    const u32 grp = (NT == 2u ? (w & 3u) : w) + 4u * (u32)z;
    const u32 slot0 = NT == 2u ? 2u * (w >> 2) : 0u;
    const u8* a_at[NT];
    u32 acc[NT][8][2];
#pragma unroll
    for (u32 t = 0; t < NT; ++t)
#pragma unroll
        for (u32 nf = 0; nf < 8; ++nf) acc[t][nf][0] = acc[t][nf][1] = 0u;
    int budget = AMP_WAIT_BUDGET_NS;
    // reduce: A_g [64, 512] (record 0: [K block 16][slice 32][512], group g = slices 4g..4g+3), 16 K blocks ascending
#pragma unroll 1
    for (u32 half = 0; half < 2; ++half) {
        if (!VIEW && p.pflags) {
            if (leader) {
                wait_parts(p.pflags, 2, (int)half, (int)half + 1, 3, p.H, p.W, p.psx, p.psy, 8 * cy, 8 * cx, 8, 8, budget);
                amp_fence_acq_rel();
            }
            __syncthreads();
        }
        if (VIEW) {
            stage_window_view<NTHREADS>(sm, p.in, p.H, p.W, ty0, tx0, 8u * half, tid);
        } else {
            stage_window<NTHREADS>(sm, p.in, TH, TW, ty0, tx0, 16u, 8u * half, tid);
            amp_cp_async_wait_all();
        }
        __syncthreads();
#pragma unroll 2
        for (u32 kb = 8u * half; kb < 8u * half + 8u; ++kb) {
#pragma unroll
            for (u32 t = 0; t < NT; ++t) a_at[t] = sm + ((slot0 + t) * 16u + kb) * 512u;
            kblock8<NT>(acc, p.w + kb * 16384u + grp * 4u * 512u, a_at, lane);
        }
    }
    // the 64 reduced values -> e4m3 -> A operands of the expansion (two K blocks of 32)
    FA ra[NT][2];
#pragma unroll
    for (u32 t = 0; t < NT; ++t) {
        ra[t][0] = acc_to_a(&acc[t][0]);
        ra[t][1] = acc_to_a(&acc[t][4]);
    }
    u32 o[NT][8][2];
#pragma unroll
    for (u32 t = 0; t < NT; ++t)
#pragma unroll
        for (u32 nf = 0; nf < 8; ++nf) o[t][nf][0] = o[t][nf][1] = 0u;
    const u8* const q0 = p.w + 262144u + grp * 16384u;
    const u8* const q2 = p.w + 393216u + grp * 16384u;
#pragma unroll 1
    for (u32 c = 0; c < 8; ++c) {
        // expand: Q0_g chunk c (32 hidden), K = 64 (K block 0 at +1024 c, K block 1 at +8192)
        u32 e[NT][4][2];
#pragma unroll
        for (u32 t = 0; t < NT; ++t)
#pragma unroll
            for (u32 nf = 0; nf < 4; ++nf) e[t][nf][0] = e[t][nf][1] = 0u;
#pragma unroll
        for (u32 kb = 0; kb < 2; ++kb) {
            FB b[4];
#pragma unroll
            for (u32 s = 0; s < 2; ++s) {
                const amp_v4 v = amp_load_nc(q0 + kb * 8192u + c * 1024u + s * 512u + lane * 16u);
                b[2 * s] = make_b(v.x, v.y);
                b[2 * s + 1] = make_b(v.z, v.w);
            }
#pragma unroll
            for (u32 t = 0; t < NT; ++t)
#pragma unroll
                for (u32 nf = 0; nf < 4; ++nf) mma(e[t][nf], ra[t][kb], b[nf]);
        }
        // cubic SiLU in f16x2: t = max(min(x, 4), -4); u = fma(-0.055908203125, |t|, 0.447265625);
        // v = fma(t, u, 0.89453125); y = x * v  (F4 17299-17319)
#pragma unroll
        for (u32 t = 0; t < NT; ++t)
#pragma unroll
            for (u32 nf = 0; nf < 4; ++nf)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    const u32 x = e[t][nf][r];
                    const u32 tt = amp_h2_max(amp_h2_min(x, 0x44004400u), 0xC400C400u);
                    const u32 u = amp_h2_fma(0xAB28AB28u, amp_h2_abs(tt), 0x37283728u);
                    const u32 v = amp_h2_fma(tt, u, 0x3B283B28u);
                    e[t][nf][r] = amp_h2_mul(x, v);
                }
        // contract: Q2_g chunk c (one K block of 32 hidden) onto the running 64-output accumulator
        FB b2[8];
#pragma unroll
        for (u32 s = 0; s < 4; ++s) {
            const amp_v4 v = amp_load_nc(q2 + c * 2048u + s * 512u + lane * 16u);
            b2[2 * s] = make_b(v.x, v.y);
            b2[2 * s + 1] = make_b(v.z, v.w);
        }
#pragma unroll
        for (u32 t = 0; t < NT; ++t) {
            const FA h = acc_to_a(&e[t][0]);
#pragma unroll
            for (u32 nf = 0; nf < 8; ++nf) mma(o[t][nf], h, b2[nf]);
        }
    }
    // store channel blocks 2g, 2g+1 of the tiles inside the map
#pragma unroll
    for (u32 t = 0; t < NT; ++t) {
        const int ty = ty0 + (int)((slot0 + t) >> 1), tx = tx0 + (int)((slot0 + t) & 1u);
        if (ty < TH && tx < TW) {
            u8* const dst = p.out + (u64)(u32)(ty * TW + tx) * 8192u + grp * 1024u + lane * 16u;
            stg16(dst, block_words(&o[t][0]));
            stg16(dst + 512u, block_words(&o[t][4]));
        }
    }
    if (p.oflags) {
        __syncthreads();
        if (leader) amp_flag_publish(p.oflags + (2 * cy + z) * ((p.W + 7) >> 3) + cx, 0);
    }
}

extern "C" __global__ void __launch_bounds__(256) amp_split_ffwd(const FfwdP p) { ffwd_body<2u, false>(p); }
extern "C" __global__ void __launch_bounds__(128) amp_split_ffwd_inpview(const FfwdP p) { ffwd_body<4u, true>(p); }

// =====================================================================================================================
// 1x1 projections (abi.md P): CTA = one unshifted 8x8 window x 256 output channels (h = ctaid.x / Gx).
// NW = 4: warp = outputs 256 h + 64 w for all four tiles; NW = 8: outputs 256 h + 64 (w & 3) for tile row w >> 2.
// acc = f16(f16(R) * gain) (RES) or 0, K = 512 in 16 blocks ascending, store e4m3 (tile layout, or view layout with
// VOUT), POOL: plus the 2x2 mean of the unrounded f16 result. WL / PARTS: the producer's flag geometry.
struct ProjArgs {
    const u8* in; const u8* res; u8* out; u8* pooled; const u8* w; const int* pflags; int* oflags;
    int psx, psy, H, W, prow, pcol;
};
// accumulator seed of a projection: f16(f16(R) * gain) of the window's tiles (zero outside the map), or 0
template <u32 NT, bool RES, bool VRES, typename PA>
AMP_INLINE void seed_acc(u32 (&acc)[NT][8][2], const PA& p, u32 n0, u32 slot0, int ty0, int tx0, int TH, int TW, u32 ft, u32 lane)
{
    if (RES) {
        u32 gain[8];
#pragma unroll
        for (u32 nf = 0; nf < 8; ++nf) gain[nf] = ldg32(p.w + 262144u + 2u * (n0 + 8u * nf + 2u * ft));
#pragma unroll
        for (u32 t = 0; t < NT; ++t) {
            const int ty = ty0 + (int)((slot0 + t) >> 1), tx = tx0 + (int)((slot0 + t) & 1u);
            const bool inside = ty < TH && tx < TW;
#pragma unroll
            for (u32 jb = 0; jb < 2; ++jb) {
                const u32 b = n0 / 32u + jb;
                amp_v4 r = amp_v4{0u, 0u, 0u, 0u};
                if (VRES) r = view_load(p.res, p.H, p.W, ty, tx, b, lane);
                else if (inside) r = amp_load_cg(p.res + (u64)(u32)(ty * TW + tx) * 8192u + b * 512u + lane * 16u);
#pragma unroll
                for (u32 f = 0; f < 4; ++f) {
                    const u32 nf = 4u * jb + f;
                    acc[t][nf][0] = amp_h2_mul(word_h2(word_of(r, (f >> 1) * 2u), f & 1u), gain[nf]);
                    acc[t][nf][1] = amp_h2_mul(word_h2(word_of(r, (f >> 1) * 2u + 1u), f & 1u), gain[nf]);
                }
            }
        }
    } else {
#pragma unroll
        for (u32 t = 0; t < NT; ++t)
#pragma unroll
            for (u32 nf = 0; nf < 8; ++nf) acc[t][nf][0] = acc[t][nf][1] = 0u;
    }
}
template <u32 NW, bool RES, bool VRES, bool VOUT, bool POOL, u32 COUT, int PARTS, int WL>
AMP_INLINE void proj_body(const ProjArgs& p)
{
    __shared__ __align__(128) u8 sm[4u * 16u * 512u];
    constexpr u32 NT = NW == 4u ? 4u : 2u;
    constexpr u32 KBSTRIDE = COUT * 32u;                     // bytes per K block of the matrix
    constexpr u32 TILE = COUT * 16u;                         // output tile bytes
    const u32 lane = threadIdx.x, w = threadIdx.y, tid = w * 32u + lane, ft = lane & 3u;
    const bool leader = tid == 0u;
    const int Gx = (p.W + 7) >> 3;
    const int h = (int)blockIdx.x / Gx, col = (int)blockIdx.x - h * Gx, cy = (int)blockIdx.y;
    const int TH = p.H >> 2, TW = p.W >> 2, ty0 = 2 * cy, tx0 = 2 * col;
    const u32 n0 = 256u * (u32)h + 64u * (NW == 4u ? w : (w & 3u));
    const u32 slot0 = NW == 4u ? 0u : 2u * (w >> 2);
    u32 acc[NT][8][2];
    int budget = AMP_WAIT_BUDGET_NS;
    const u8* a_at[NT];
#pragma unroll 1
    for (u32 half = 0; half < 2; ++half) {
        if (p.pflags) {
            if (leader) {
                wait_parts(p.pflags, PARTS, (int)half * PARTS / 2, ((int)half + 1) * PARTS / 2, WL, p.H, p.W, p.psx, p.psy,
                           8 * cy, 8 * col, 8, 8, budget);
                amp_fence_acq_rel();
            }
            __syncthreads();
        }
        stage_window<NW * 32u>(sm, p.in, TH, TW, ty0, tx0, 16u, 8u * half, tid);
        if (half == 0u) {
            // accumulator seed, read only now: the residual is the output of the producer's producer (proj: the qkv
            // input; ffwd_proj: the ffwd input), which the runtime's chained launches may still be writing when this
            // kernel starts; once a producer flag of this window is set, its producer had published this window too
            // (abi.md 0.1 / P.5; NVIDIA's PTX also loads the residual after its flag poll, F4 63612 vs 63966)
            seed_acc<NT, RES, VRES>(acc, p, n0, slot0, ty0, tx0, TH, TW, ft, lane);
        }
        amp_cp_async_wait_all();
        __syncthreads();
#pragma unroll 2
        for (u32 kb = 8u * half; kb < 8u * half + 8u; ++kb) {
#pragma unroll
            for (u32 t = 0; t < NT; ++t) a_at[t] = sm + ((slot0 + t) * 16u + kb) * 512u;
            kblock8<NT>(acc, p.w + kb * KBSTRIDE + (n0 / 16u) * 512u, a_at, lane);
        }
    }
    // stores
#pragma unroll
    for (u32 t = 0; t < NT; ++t) {
        const int ty = ty0 + (int)((slot0 + t) >> 1), tx = tx0 + (int)((slot0 + t) & 1u);
#pragma unroll
        for (u32 jb = 0; jb < 2; ++jb) {
            const amp_v4 v = block_words(&acc[t][4u * jb]);
            if (VOUT) view_store(p.out, p.H, p.W, ty, tx, n0 / 32u + jb, lane, v);
            else if (ty < TH && tx < TW) stg16(p.out + (u64)(u32)(ty * TW + tx) * TILE + (n0 / 32u + jb) * 512u + lane * 16u, v);
        }
    }
    if (POOL) {
        // 2x2 mean of the unrounded f16 accumulators (abi.md P.6 item 4): pooled token (py, px) of the window's
        // pooled tile = lane L's word 0 (py = L>>4 & 1, px = L>>2 & 3; tile row 0) and word 1 (py + 2; tile row 1).
        // Sources: lanes s, s+4 (columns 2px, 2px+1 of row 2py), s+16, s+20 (row 2py+1), s = (L<<1 & 8) | (L & 3),
        // register = tile column px >> 1, row half py & 1. mean = ((a + b) + (c + d)) * f16(0.25), then e4m3.
        const u32 s = ((lane << 1) & 8u) | ft;
        const u32 tc = (lane >> 3) & 1u, rh = (lane >> 4) & 1u;
        u32 pv[2][8];
#pragma unroll
        for (u32 tr = 0; tr < 2; ++tr)
#pragma unroll
            for (u32 nf = 0; nf < 8; ++nf) {
                u32 q[4][4];
#pragma unroll
                for (u32 cand = 0; cand < 4; ++cand) {
                    const u32 src = acc[2u * tr + (cand >> 1)][nf][cand & 1u];
                    q[cand][0] = __shfl_sync(0xffffffffu, src, s);
                    q[cand][1] = __shfl_sync(0xffffffffu, src, s + 4u);
                    q[cand][2] = __shfl_sync(0xffffffffu, src, s + 16u);
                    q[cand][3] = __shfl_sync(0xffffffffu, src, s + 20u);
                }
                const u32 cand = tc * 2u + rh;
                u32 a = q[0][0], b = q[0][1], c = q[0][2], d = q[0][3];
#pragma unroll
                for (u32 k = 1; k < 4; ++k)
                    if (cand == k) { a = q[k][0]; b = q[k][1]; c = q[k][2]; d = q[k][3]; }
                pv[tr][nf] = amp_h2_mul(amp_h2_add(amp_h2_add(a, b), amp_h2_add(c, d)), 0x34003400u);
            }
        if (cy < p.prow / 4 && col < p.pcol / 4) {
            u8* const dst = p.pooled + (u64)(u32)(cy * (p.pcol / 4) + col) * 8192u + (n0 / 32u) * 512u + lane * 16u;
#pragma unroll
            for (u32 jb = 0; jb < 2; ++jb) {
                amp_v4 v;
                v.x = pack4(pv[0][4u * jb], pv[0][4u * jb + 1u]); v.y = pack4(pv[1][4u * jb], pv[1][4u * jb + 1u]);
                v.z = pack4(pv[0][4u * jb + 2u], pv[0][4u * jb + 3u]); v.w = pack4(pv[1][4u * jb + 2u], pv[1][4u * jb + 3u]);
                stg16(dst + jb * 512u, v);
            }
        }
    }
    if (p.oflags) {
        __syncthreads();
        if (leader) amp_flag_publish(p.oflags + 2 * cy * Gx + (int)blockIdx.x, 0);
    }
}
AMP_INLINE ProjArgs args_of(const ProjP& q, bool publish)
{
    return ProjArgs{q.in, q.res, q.out, nullptr, q.w, q.pflags, publish ? q.oflags : nullptr, q.psx, q.psy, q.H, q.W, 0, 0};
}
extern "C" __global__ void __launch_bounds__(128) amp_split_ffwd_proj(const ProjP p)
{ proj_body<4u, true, false, false, false, 512u, 2, 3>(args_of(p, true)); }
extern "C" __global__ void __launch_bounds__(128) amp_split_ffwd_proj_inpview(const ProjP p)
{ proj_body<4u, true, true, false, false, 512u, 2, 3>(args_of(p, true)); }
extern "C" __global__ void __launch_bounds__(256) amp_split_proj(const ProjP p)
{ proj_body<8u, true, false, false, false, 512u, 4, 3>(args_of(p, true)); }
extern "C" __global__ void __launch_bounds__(128) amp_split_proj_outview(const ProjP p)
{ proj_body<4u, true, false, true, false, 512u, 4, 3>(args_of(p, false)); }
extern "C" __global__ void __launch_bounds__(128) amp_split_proj_pool(const PoolP p)
{
    proj_body<4u, true, false, false, true, 512u, 4, 3>(
        ProjArgs{p.in, p.res, p.out, p.pooled, p.w, p.pflags, p.oflags, p.psx, p.psy, p.H, p.W, p.prow, p.pcol});
}
// final head: pooled map 512 -> 1024, no residual; waits on proj_pool's flags per pooled tile (4-token windows,
// index (2 trow + half) * ceil(W/4) + tcol, rows below ceil(H/4) only), publishes nothing
extern "C" __global__ void __launch_bounds__(256) amp_split_final_head(const HeadP p)
{
    proj_body<8u, false, false, false, false, 1024u, 2, 2>(
        ProjArgs{p.in, nullptr, p.out, nullptr, p.w, p.pflags, nullptr, 0, 0, p.H, p.W, 0, 0});
}

// =====================================================================================================================
// Shifted-window attention (abi.md Q). CTA = one (shifted) 8x8 window x heads 4 z .. 4 z + 3; warp = one head for all
// 64 tokens. K and V of the four tiles in one pass, then per query half (tiles 0,1 then 2,3) the Q pass and the
// attention. (NVIDIA computes Q, K, V in one pass; every value is the same product in the same K order.)
//
// Q / K norm of a row (lane: 8 of the head's 32 dims, in four f16x2 fragments n0..n3): squares and pair sums as
// NORM_FMA (NVIDIA's contracted form by default), then across the quad by xor 2 then xor 1, then the two halves (f16), max f16(0.000062),
// rsqrt.approx.ftz.f32, narrowed to f16.
AMP_INLINE u32 norm_rsqrt(const u32 (&d)[4])
{
    const u32 s0 = amp_h2_mul(d[0], d[0]), s1 = amp_h2_mul(d[1], d[1]), s2 = amp_h2_mul(d[2], d[2]), s3 = amp_h2_mul(d[3], d[3]);
#if NORM_FMA >= 0
    // the pairs as fma (ptxas contraction of mul + add): bit 0 picks which square of (n1, n3) stays a mul, bit 1 of (n0, n2)
    const u32 a13 = (NORM_FMA & 1) ? amp_h2_fma(d[3], d[3], s1) : amp_h2_fma(d[1], d[1], s3);
    const u32 a02 = (NORM_FMA & 2) ? amp_h2_fma(d[2], d[2], s0) : amp_h2_fma(d[0], d[0], s2);
    u32 s = amp_h2_add(a13, a02);
#elif NORM_PAIRS == 0
    u32 s = amp_h2_add(amp_h2_add(s1, s3), amp_h2_add(s0, s2));
#elif NORM_PAIRS == 1
    u32 s = amp_h2_add(amp_h2_add(s0, s1), amp_h2_add(s2, s3));
#else
    u32 s = amp_h2_add(amp_h2_add(s0, s3), amp_h2_add(s1, s2));
#endif
    s = amp_h2_add(s, __shfl_xor_sync(0xffffffffu, s, 2));
    s = amp_h2_add(s, __shfl_xor_sync(0xffffffffu, s, 1));
    const u32 total = amp_h2_add(amp_prmt(s, 0u, 0x1010u), amp_prmt(s, 0u, 0x3232u));   // {lo,lo} + {hi,hi}
    const u32 floored = amp_h2_max(total, 0x04100410u);                                // f16(0.000062)
    float lo, hi;
    amp_h2_unpack(floored, lo, hi);
    const float r = amp_rsqrt_approx(lo);
    return amp_h2_pack(r, r);
}
// the rows of one tile's Q or K (acc[n][r], n = the head's four 8-dim fragments) times their rsqrt (and `scale`) -> the
// lane's operand words: lo / hi of row g in w[0] / w[2], of row g + 8 in w[1] / w[3] (tile-layout block order)
AMP_INLINE amp_v4 qk_words(const u32 (*acc)[2], bool scaled, u32 scale)
{
    u32 o[4][2];
#pragma unroll
    for (u32 r = 0; r < 2; ++r) {
        const u32 d[4] = {acc[0][r], acc[1][r], acc[2][r], acc[3][r]};
        const u32 rs = norm_rsqrt(d);
#pragma unroll
        for (u32 n = 0; n < 4; ++n) {
            u32 v = amp_h2_mul(acc[n][r], rs);
            if (scaled) v = amp_h2_mul(v, scale);
            o[n][r] = v;
        }
    }
    return block_words(o);
}

extern "C" __global__ void __launch_bounds__(128) amp_split_qkv(const QkvP p)
{
    __shared__ __align__(128) u8 sm[4u * 16u * 512u];
    const u32 lane = threadIdx.x, w = threadIdx.y, tid = w * 32u + lane;
    const bool leader = tid == 0u;
    const int x0 = p.sx + 8 * (int)blockIdx.x, y0 = p.sy + 8 * (int)blockIdx.y;
    const int TH = p.H >> 2, TW = p.W >> 2, tx0 = x0 >> 2, ty0 = y0 >> 2;
    const u32 head = 4u * blockIdx.z + w;
    const u8* const wq = p.w + head * 6u * 512u + lane * 16u;
    int budget = AMP_WAIT_BUDGET_NS;
    // stage the window; producer = ffwd_proj (unshifted, 2 parts per window; no producer shift field)
#pragma unroll 1
    for (u32 half = 0; half < 2; ++half) {
        if (p.pflags) {
            if (leader) {
                wait_parts(p.pflags, 2, (int)half, (int)half + 1, 3, p.H, p.W, 0, 0, y0, x0, 8, 8, budget);
                amp_fence_acq_rel();
            }
            __syncthreads();
        }
        stage_window<128u>(sm, p.in, TH, TW, ty0, tx0, 16u, 8u * half, tid);
    }
    amp_cp_async_wait_all();
    __syncthreads();
    // K (slices j = 2, 3) and V (j = 4, 5) of the four tiles: 16 K blocks ascending, accumulators start at f16 0
    u32 kv[4][8][2];
#pragma unroll
    for (u32 t = 0; t < 4; ++t)
#pragma unroll
        for (u32 nf = 0; nf < 8; ++nf) kv[t][nf][0] = kv[t][nf][1] = 0u;
#pragma unroll 2
    for (u32 kb = 0; kb < 16; ++kb) {
        FB b[8];
#pragma unroll
        for (u32 j = 0; j < 4; ++j) {
            const amp_v4 v = amp_load_nc(wq + kb * 49152u + (2u + j) * 512u);
            b[2 * j] = make_b(v.x, v.y);
            b[2 * j + 1] = make_b(v.z, v.w);
        }
#pragma unroll
        for (u32 t = 0; t < 4; ++t) {
            const FA a = make_a(lds16(sm + (t * 16u + kb) * 512u + lane * 16u));
#pragma unroll
            for (u32 nf = 0; nf < 8; ++nf) mma(kv[t][nf], a, b[nf]);
        }
    }
    // K: normalised, e4m3 -> B operands of the logits (key fragment 2 t + r = tile t rows 0-7 / 8-15)
    FB kb8[8];
#pragma unroll
    for (u32 t = 0; t < 4; ++t) {
        const amp_v4 k = qk_words(&kv[t][0], false, 0u);
        kb8[2 * t] = make_b(k.x, k.z);
        kb8[2 * t + 1] = make_b(k.y, k.w);
    }
    // V: f16 -> movmatrix.trans -> e4m3 -> B operands of the context: step ks = key tiles 2 ks, 2 ks + 1, fragment j =
    // dims 8 j..8 j + 7; word 0 = tile 2 ks (keys 2t, 2t+1 | 8+2t, 9+2t), word 1 = tile 2 ks + 1
    FB vb[2][4];
#pragma unroll
    for (u32 ks = 0; ks < 2; ++ks)
#pragma unroll
        for (u32 j = 0; j < 4; ++j) {
            const u32 w0 = pack4(amp_transpose8(kv[2 * ks][4 + j][0]), amp_transpose8(kv[2 * ks][4 + j][1]));
            const u32 w1 = pack4(amp_transpose8(kv[2 * ks + 1][4 + j][0]), amp_transpose8(kv[2 * ks + 1][4 + j][1]));
            vb[ks][j] = make_b(w0, w1);
        }
    const u8* const bias = p.w + 786432u + head * 8192u + lane * 16u;
    const u32 base = lane & ~3u;
#pragma unroll 1
    for (u32 qh = 0; qh < 2; ++qh) {
        float sf = __ldg((const float*)(p.w + 917504u) + head);
        const u32 scale = amp_h2_pack(sf, sf);               // cvt.rn.f16.f32 of the head scale
        // Q of query tiles 2 qh, 2 qh + 1 (slices j = 0, 1)
        u32 q[2][4][2];
#pragma unroll
        for (u32 i = 0; i < 2; ++i)
#pragma unroll
            for (u32 nf = 0; nf < 4; ++nf) q[i][nf][0] = q[i][nf][1] = 0u;
#pragma unroll 2
        for (u32 kb = 0; kb < 16; ++kb) {
            FB b[4];
#pragma unroll
            for (u32 j = 0; j < 2; ++j) {
                const amp_v4 v = amp_load_nc(wq + kb * 49152u + j * 512u);
                b[2 * j] = make_b(v.x, v.y);
                b[2 * j + 1] = make_b(v.z, v.w);
            }
#pragma unroll
            for (u32 i = 0; i < 2; ++i) {
                const FA a = make_a(lds16(sm + ((2u * qh + i) * 16u + kb) * 512u + lane * 16u));
#pragma unroll
                for (u32 nf = 0; nf < 4; ++nf) mma(q[i][nf], a, b[nf]);
            }
        }
        // logits on the raw f16 position bias, numerators
        u32 num[2][8][2];
#pragma unroll
        for (u32 i = 0; i < 2; ++i) {
            const FA qa = make_a(qk_words(q[i], true, scale));
            const u32 qt = 2u * qh + i;
#pragma unroll
            for (u32 kt = 0; kt < 4; ++kt) {
                const amp_v4 b = amp_load_nc(bias + (qt * 4u + kt) * 512u);
                num[i][2 * kt][0] = b.x; num[i][2 * kt][1] = b.y; num[i][2 * kt + 1][0] = b.z; num[i][2 * kt + 1][1] = b.w;
            }
#pragma unroll
            for (u32 kf = 0; kf < 8; ++kf) mma(num[i][kf], qa, kb8[kf]);
#pragma unroll
            for (u32 kf = 0; kf < 8; ++kf)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    u32 x = amp_h2_fma(num[i][kf][r], 0x29C029C0u, 0x3D343D34u);
                    x = amp_h2_min(amp_h2_max(x, 0x3C203C20u), 0x3E473E47u);
                    num[i][kf][r] = (x << 5) + 0x7FF88000u;
                }
        }
        // denominators: per lane (((f0+f1)+(f2+f3))+(f4+f5))+(f6+f7), then over the quad ((P0+P1)+P2)+P3 in lane order,
        // then even + odd key (f16), max f16(0.000062), rcp.approx.ftz.f32, narrowed to f16 (F4 53838-54030)
        u32 rec[2][2];
#pragma unroll
        for (u32 i = 0; i < 2; ++i)
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const u32 (*f)[2] = num[i];
                const u32 part = amp_h2_add(amp_h2_add(amp_h2_add(amp_h2_add(f[0][r], f[1][r]), amp_h2_add(f[2][r], f[3][r])),
                                                       amp_h2_add(f[4][r], f[5][r])), amp_h2_add(f[6][r], f[7][r]));
                const u32 p0 = __shfl_sync(0xffffffffu, part, base), p1 = __shfl_sync(0xffffffffu, part, base + 1u);
                const u32 p2 = __shfl_sync(0xffffffffu, part, base + 2u), p3 = __shfl_sync(0xffffffffu, part, base + 3u);
                const u32 s = amp_h2_add(amp_h2_add(amp_h2_add(p0, p1), p2), p3);
                const u32 total = amp_h2_max(amp_h2_add(amp_prmt(s, 0u, 0x1010u), amp_prmt(s, 0u, 0x3232u)), 0x04100410u);
                float lo, hi;
                amp_h2_unpack(total, lo, hi);
                const float rc = amp_rcp_approx(lo);
                rec[i][r] = amp_h2_pack(rc, rc);
            }
        // probabilities -> e4m3 A operands (keys 0-31 = fragments 0..3, 32-63 = 4..7); context = P.V, two k32 steps
#pragma unroll
        for (u32 i = 0; i < 2; ++i) {
            u32 cacc[4][2];
#pragma unroll
            for (u32 j = 0; j < 4; ++j) cacc[j][0] = cacc[j][1] = 0u;
#pragma unroll
            for (u32 ks = 0; ks < 2; ++ks) {
                u32 pr[4][2];
#pragma unroll
                for (u32 f = 0; f < 4; ++f)
#pragma unroll
                    for (u32 r = 0; r < 2; ++r) pr[f][r] = amp_h2_mul(num[i][4 * ks + f][r], rec[i][r]);
                const FA pa = acc_to_a(pr);
#pragma unroll
                for (u32 j = 0; j < 4; ++j) mma(cacc[j], pa, vb[ks][j]);
            }
            const u32 qt = 2u * qh + i;
            const int ty = ty0 + (int)(qt >> 1), tx = tx0 + (int)(qt & 1u);
            if (ty >= 0 && ty < TH && tx >= 0 && tx < TW)
                stg16(p.out + (u64)(u32)(ty * TW + tx) * 8192u + head * 512u + lane * 16u, block_words(cacc));
        }
    }
    if (p.oflags) {
        __syncthreads();
        const int gx_own = (p.W - min(p.sx, 0) + 7) >> 3;
        if (leader) amp_flag_publish(p.oflags + (4 * (int)blockIdx.y + (int)blockIdx.z) * gx_own + (int)blockIdx.x, 0);
    }
}

// =====================================================================================================================
// Decoder input (abi.md D): CTA = coarse 4x4 tile (ty, tx) x 256 outputs (half) x K quarter z; warp = 128 outputs.
// Partial p_z (K = 256, 8 K blocks ascending, f16 accumulator from 0), merged in NVIDIA's order through the f16
// scratch under the counter; z = 3 adds, upsamples 2x (nearest), adds f16(f16(skip) * gain), stores e4m3, publishes.
extern "C" __global__ void __launch_bounds__(64) amp_dec_input_upsample(const DecP p)
{
    __shared__ __align__(128) u8 sm[8u * 512u];
    const u32 lane = threadIdx.x, w = threadIdx.y, tid = w * 32u + lane, ft = lane & 3u;
    const bool leader = tid == 0u;
    const int Tc = (p.Wc + 3) >> 2;
    const int half = (int)blockIdx.x / Tc, tx = (int)blockIdx.x - half * Tc, ty = (int)blockIdx.y, z = (int)blockIdx.z;
    const int TCH = p.Hc >> 2, TCW = p.Wc >> 2;
    const bool in_range = ty < TCH && tx < TCW;
    const u32 oc0 = 256u * (u32)half + 128u * w;
    const int idx = (2 * ty + half) * Tc + tx;
    // the coarse tile's K quarter: channel blocks 8z .. 8z+7 of a 1024-channel tile (zeros outside)
    for (u32 i = tid; i < 8u * 32u; i += 64u) {
        u8* const dst = sm + i * 16u;
        if (in_range) amp_cp_async16(amp_smem(dst), p.in + (u64)(u32)(ty * TCW + tx) * 16384u + (8u * (u32)z) * 512u + i * 16u);
        else sts16_zero(dst);
    }
    amp_cp_async_commit();
    amp_cp_async_wait_all();
    __syncthreads();
    u32 acc[16][2];
#pragma unroll
    for (u32 nf = 0; nf < 16; ++nf) acc[nf][0] = acc[nf][1] = 0u;
#pragma unroll 1
    for (u32 i = 0; i < 8; ++i) {
        const u32 kb = 8u * (u32)z + i;
        const FA a = make_a(lds16(sm + i * 512u + lane * 16u));
#pragma unroll
        for (u32 s = 0; s < 8; ++s) {
            const amp_v4 v = amp_load_nc(p.w + kb * 16384u + (oc0 / 16u + s) * 512u + lane * 16u);
            mma(acc[2 * s], a, make_b(v.x, v.y));
            mma(acc[2 * s + 1], a, make_b(v.z, v.w));
        }
    }
    // scratch word layout: [coarse tile ty*(Wc/4)+tx][group of 16 outputs][lane][acc nf0 rows r, r+8 | nf1 rows r, r+8]
    u8* const scr = p.scratch + (u64)(u32)(ty * TCW + tx) * 16384u + (oc0 / 16u) * 512u + lane * 16u;
    if (z != 0) {
        int budget = AMP_WAIT_BUDGET_NS;
        if (leader) {
            amp_wait_at_least(p.counter + idx, z - 1, budget);
            amp_fence_acq_rel();
        }
        __syncthreads();
    }
    if (z == 0) {
        if (in_range)
#pragma unroll
            for (u32 s = 0; s < 8; ++s) stg16(scr + s * 512u, amp_v4{acc[2 * s][0], acc[2 * s][1], acc[2 * s + 1][0], acc[2 * s + 1][1]});
    } else if (z < 3) {
        if (in_range)
#pragma unroll
            for (u32 s = 0; s < 8; ++s) {
                u32* const q = (u32*)(scr + s * 512u);
#if DEC_MERGE == 1
                const amp_v4 v = amp_load_cg(q);
                stg16((u8*)q, amp_v4{amp_h2_add(v.x, acc[2 * s][0]), amp_h2_add(v.y, acc[2 * s][1]),
                                     amp_h2_add(v.z, acc[2 * s + 1][0]), amp_h2_add(v.w, acc[2 * s + 1][1])});
#else
                amp_red_add_h2(q + 0, acc[2 * s][0]);
                amp_red_add_h2(q + 1, acc[2 * s][1]);
                amp_red_add_h2(q + 2, acc[2 * s + 1][0]);
                amp_red_add_h2(q + 3, acc[2 * s + 1][1]);
#endif
            }
    } else if (in_range) {
#pragma unroll
        for (u32 s = 0; s < 8; ++s) {
            const amp_v4 v = amp_load_cg(scr + s * 512u);
            acc[2 * s][0] = amp_h2_add(v.x, acc[2 * s][0]);
            acc[2 * s][1] = amp_h2_add(v.y, acc[2 * s][1]);
            acc[2 * s + 1][0] = amp_h2_add(v.z, acc[2 * s + 1][0]);
            acc[2 * s + 1][1] = amp_h2_add(v.w, acc[2 * s + 1][1]);
        }
        u32 gain[16];
#pragma unroll
        for (u32 nf = 0; nf < 16; ++nf) gain[nf] = ldg32(p.w + 524288u + 2u * (oc0 + 8u * nf + 2u * ft));
        const int TH = p.H >> 2, TW = p.W >> 2;
        const u32 src0 = ft | ((lane >> 1) & 4u);
#pragma unroll
        for (u32 oy = 0; oy < 2; ++oy)
#pragma unroll
            for (u32 ox = 0; ox < 2; ++ox) {
                const int oty = 2 * ty + (int)oy, otx = 2 * tx + (int)ox;
                const bool inside = oty < TH && otx < TW;
#pragma unroll
                for (u32 jb = 0; jb < 4; ++jb) {
                    const u32 b = oc0 / 32u + jb;
                    const amp_v4 sk = inside ? amp_load_cg(p.skip + (u64)(u32)(oty * TW + otx) * 8192u + b * 512u + lane * 16u)
                                             : amp_v4{0u, 0u, 0u, 0u};
                    u32 v[4][2];
#pragma unroll
                    for (u32 f = 0; f < 4; ++f) {
                        const u32 nf = 4u * jb + f;
                        const u32 up0 = __shfl_sync(0xffffffffu, acc[nf][oy], src0 | (8u * ox));
                        const u32 up1 = __shfl_sync(0xffffffffu, acc[nf][oy], src0 | (8u * ox) | 16u);
                        // NVIDIA's PTX has mul.f16x2 then add.f16x2, which its ptxas (default --fmad=true) contracts
                        // into one fma: v = round(up + skip * gain), measured (split_lab.py amp --build hwref: the
                        // two-rounding form differs from NVIDIA's kernel in 0.2 % of the codes, the fma in none)
                        v[f][0] = amp_h2_fma(word_h2(word_of(sk, (f >> 1) * 2u), f & 1u), gain[nf], up0);
                        v[f][1] = amp_h2_fma(word_h2(word_of(sk, (f >> 1) * 2u + 1u), f & 1u), gain[nf], up1);
                    }
                    if (inside) stg16(p.out + (u64)(u32)(oty * TW + otx) * 8192u + b * 512u + lane * 16u, block_words(v));
                }
            }
    }
    __syncthreads();
    if (leader) {
        amp_flag_publish(p.counter + idx, z);
        if (z == 3 && p.oflags) amp_flag_publish(p.oflags + idx, 0);
    }
}
