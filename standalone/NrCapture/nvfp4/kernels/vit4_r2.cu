// NVFP4 (4-bit) replacements of the DLSS-NR ViT kernels cc_vit_1d_{qkv_chained, attention_chained, projection_chained,
// projection_wait}_fp8 (runtime 310.9.16). Target sm_120a, built with NVRTC (--fmad=false). One module, three entries:
//   nr_vit4_qkv         grid (16 T, 1, 2), block (32, 4)     T = 128-token tiles
//   nr_vit4_attention   grid (32, ceil(T / 2), 1), block (32, 4)
//   nr_vit4_projection  grid (8 T, 1, 4), block (32, 4)      (oflags 0: the runtime's projection_wait)
// Exactly the runtime's launch geometry, no dynamic shared memory; buffers, flags and the formats handed to the next
// kernel are the runtime's (vit-spec/abi.md 1-5), so any one of the three can be the runtime's own kernel.
// Parameters: the runtime's block with its weight record pointer replaced by ours, plus an options word (abi.md 10).
// The arithmetic is defined by out-model-map/nvfp4-20261113/v2/sim/vit4_model.py.
//
// QKV and projection: both operands NVFP4 (weights packed offline, activations packed here per token, one ue4m3 scale
// per 16 channels: of each 32 the 16 with the same bit 2), float32 accumulators. The K split of the grid's z (halves /
// quarters) is kept; partial sums meet in the runtime's float16 scratch in z order behind its merge counter.
// Attention: the runtime's FP8 / float16 arithmetic (default); options bit 1 = P and V NVFP4, bit 2 = Q and K NVFP4.
//
// Switches: -DWAIT_BUDGET_NS=<n>  nominal sleep time every wait may spend (default 1 << 23, as swin4.cu)
//           -DATT_SG=<n>          attention: 64-key groups per shared-memory stage (default 4; 1 = a barrier per group)
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
struct __align__(16) V4 { u32 x, y, z, w; };

#define NR_INLINE static __forceinline__ __device__
#ifndef WAIT_BUDGET_NS
#define WAIT_BUDGET_NS (1 << 23)
#endif
// Attention: key groups of 64 per shared-memory stage (two stages, 2 x 4 KB x ATT_SG; one CTA barrier per stage)
#ifndef ATT_SG
#define ATT_SG 4
#endif
// Attention round 2 switches: ATT_SEL 1 = the row-sum transposition by selects on the bits of t (0 = round 1 chains,
// which ptxas compiled as branches); ATT_PVORD 1 = P.V issued key-half outer, query group inner (accumulator reuse
// distance 16 instead of 4)
#ifndef ATT_SEL
#define ATT_SEL 1
#endif
#ifndef ATT_PVORD
#define ATT_PVORD 1
#endif
// Timing builds only (outputs meaningless): DBG_STAGE 0 = no key loop, 1 = + K and V shared reads, 2 = + Q.K and row
// sums, 3 = + the exponential, 4 and up = everything (the release build is 9)
#ifndef DBG_STAGE
#define DBG_STAGE 9
#endif
// Projection timing builds: PDBG_STAGE 1 = input stored raw (no NVFP4 packing), 2 = no mma; 0 = release
#ifndef PDBG_STAGE
#define PDBG_STAGE 0
#endif
#define SLAB 1152u
#define FLOOR 6.198883056640625e-05f
#define OPT_PV4 1u
#define OPT_QK4 2u

struct QkvParams {          // the runtime's 80 bytes (+32: our record), then options
    const u8* x; u8* q; u8* k; u8* v; const u8* w; int* counter; u8* scratch; const int* pflags; int* oflags;
    int d0, d1; u32 options; u32 reserved;
};
struct AttParams {          // the runtime's 64 bytes (+32: our record), then options
    const u8* q; const u8* k; const u8* v; u8* out; const u8* w; const int* pflags; int* oflags;
    int d0, d1; u32 options; u32 reserved;
};
struct ProjParams {         // the runtime's 72 bytes (+24: our record), then options
    const u8* x; const u8* skip; u8* out; const u8* w; int* counter; u8* scratch; const int* pflags; int* oflags;
    int d0, d1; u32 options; u32 reserved;
};

// ---- flags (bounded waits: a missing producer costs a wrong frame, never a hung GPU) -------------------------------
NR_INLINE int flag_load(const int* flag)
{
    int value;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"((u64)flag) : "memory");
    return value;
}
NR_INLINE void flag_publish(int* flag, int value) { asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"((u64)flag), "r"(value) : "memory"); }
NR_INLINE void nap() { asm volatile("nanosleep.u32 64;" ::: "memory"); }
NR_INLINE void wait_at_least(const int* flag, int value, int& budget)
{
    for (; budget > 0 && flag_load(flag) < value; budget -= 64) nap();
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
NR_INLINE u32 smem_addr(const void* p)
{
    u32 a;
    asm("{\n .reg .u64 nr_t;\n cvta.to.shared.u64 nr_t, %1;\n cvt.u32.u64 %0, nr_t;\n}" : "=r"(a) : "l"((u64)p));
    return a;
}
NR_INLINE void cp_async16(void* dst, const u8* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr(dst)), "l"((u64)src) : "memory");
}
NR_INLINE void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
NR_INLINE void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
NR_INLINE void cp_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// ---- tensor-core instructions --------------------------------------------------------------------------------------
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

// ---- float16 pairs (low half first) ---------------------------------------------------------------------------------
NR_INLINE u32 h2_fma(u32 a, u32 b, u32 c) { u32 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
NR_INLINE u32 h2_mul(u32 a, u32 b) { u32 d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_add(u32 a, u32 b) { u32 d; asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_max(u32 a, u32 b) { u32 d; asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_min(u32 a, u32 b) { u32 d; asm("min.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 h2_pack(float lo, float hi) { u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
NR_INLINE void h2_unpack(u32 h, float& lo, float& hi)
{
    asm("{\n .reg .b16 nr_l, nr_u;\n mov.b32 {nr_l, nr_u}, %2;\n cvt.f32.f16 %0, nr_l;\n cvt.f32.f16 %1, nr_u;\n}" : "=f"(lo), "=f"(hi) : "r"(h));
}
NR_INLINE u32 h2_to_e4m3x2(u32 h) { u16 d; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(h)); return d; }
NR_INLINE u32 transpose8(u32 h) { u32 d; asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(h)); return d; }
NR_INLINE float rsqrt_approx(float x) { float d; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE float rcp_approx(float x) { float d; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }
NR_INLINE float f16_round(float x) { float lo, hi; h2_unpack(h2_pack(x, x), lo, hi); return lo; }
NR_INLINE u32 prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
NR_INLINE void e4m3x4_to_h2(u32 word, u32& a, u32& b)
{
    asm("{\n .reg .b16 nr_l, nr_u;\n mov.b32 {nr_l, nr_u}, %2;\n cvt.rn.f16x2.e4m3x2 %0, nr_l;\n cvt.rn.f16x2.e4m3x2 %1, nr_u;\n}"
        : "=r"(a), "=r"(b) : "r"(word));
}
NR_INLINE u32 h2x2_to_e4m3x4(u32 a, u32 b) { return h2_to_e4m3x2(a) | (h2_to_e4m3x2(b) << 16); }
// four floats -> float16 -> e4m3 codes (byte 0 first)
NR_INLINE u32 f32x4_to_e4m3x4(float a, float b, float c, float d) { return h2x2_to_e4m3x4(h2_pack(a, b), h2_pack(c, d)); }

// ---- NVFP4 packing (swin4.cu m_pack2_h) -------------------------------------------------------------------------------
NR_INLINE u32 xamax(u32 a, u32 b) { u32 d; asm("max.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
NR_INLINE u32 habs(u32 a) { u32 d; asm("abs.f16x2 %0, %1;" : "=r"(d) : "r"(a)); return d; }
NR_INLINE void split16(u32 w, u16& lo, u16& hi) { asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(w)); }
NR_INLINE u32 e2m1x8_h(const u32* h, u16 r)
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
// two rows of eight float16 values (this lane's half of a block of 16; lane ^ 1 holds the other half) -> two words of
// nibbles; returns the scale codes (byte 0: row 0, byte 1: row 1). The operand stands for value * pre.
NR_INLINE u32 pack2(const u32* h0, const u32* h1, float pre, u32& w0, u32& w1)
{
    const u32 m0 = xamax(xamax(h0[0], h0[1]), xamax(h0[2], h0[3]));
    const u32 m1 = xamax(xamax(h1[0], h1[1]), xamax(h1[2], h1[3]));
    u32 mm = xamax(prmt(m0, m1, 0x5410u), prmt(m0, m1, 0x7632u));
    mm = xamax(mm, __shfl_xor_sync(0xffffffffu, mm, 1));
    float a0, a1;
    h2_unpack(habs(mm), a0, a1);
    const float raw0 = fmaxf(a0 * pre * 0.16666667163372040f, 0.001953125f), raw1 = fmaxf(a1 * pre * 0.16666667163372040f, 0.001953125f);
    u16 c16;
    u32 sv;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(c16) : "f"(raw0), "f"(raw1));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sv) : "h"(c16));
    float s0, s1;
    h2_unpack(sv, s0, s1);
    u16 q0, q1;
    split16(h2_pack(rcp_approx(s0) * pre, rcp_approx(s1) * pre), q0, q1);
    w0 = e2m1x8_h(h0, q0);
    w1 = e2m1x8_h(h1, q1);
    return (u32)c16;
}
// The scale word of an A operand (lane t even: row g, odd: row g + 8) or of a B operand (row = column `sel`), from
// two code pairs this lane holds for blocks (half 0, its tb) and (half 1, its tb): bytes [b = 2 half + tb].
NR_INLINE u32 scale_word(u32 pair_h0, u32 pair_h1, u32 t, u32 sel)
{
    const u32 mine = (pair_h0 & 0xFFFFu) | (pair_h1 << 16);
    const u32 other = __shfl_xor_sync(0xffffffffu, mine, 2);
    const u32 lo = (t >> 1) ? other : mine, hi = (t >> 1) ? mine : other;
    return prmt(lo, hi, sel | ((4u + sel) << 4) | ((2u + sel) << 8) | ((6u + sel) << 12));
}
// one 16-byte act-layout load of a 32-channel block -> float16 rows (row g: words x, z; row g + 8: words y, w)
NR_INLINE void act_rows(const V4& v, u32* h0, u32* h1)
{
    e4m3x4_to_h2(v.x, h0[0], h0[1]); e4m3x4_to_h2(v.z, h0[2], h0[3]);
    e4m3x4_to_h2(v.y, h1[0], h1[1]); e4m3x4_to_h2(v.w, h1[2], h1[3]);
}
// shared-memory unit = 16 tokens x one k64 step: 512 B of A words (lane order) + 64 B of scale words
NR_INLINE void put_scale(u8* words, u32 g, u32 t, u32 khalf, u32 pair)
{
    words[g * 8u + (t & 1u) * 4u + khalf * 2u + (t >> 1)] = (u8)(pair >> ((t & 1u) * 8u));
}
NR_INLINE void pack_unit(const V4& xa, const V4& xb, u8* nib, u8* sc, float pre, u32 lane, u32 g, u32 t)
{
    u32 h0[4], h1[4], n0, n1, n2, n3;
    act_rows(xa, h0, h1);
    const u32 ca = pack2(h0, h1, pre, n0, n1);
    act_rows(xb, h0, h1);
    const u32 cb = pack2(h0, h1, pre, n2, n3);
    const V4 a = {n0, n1, n2, n3};
    *(V4*)(nib + lane * 16u) = a;
    put_scale(sc, g, t, 0u, ca);
    put_scale(sc, g, t, 1u, cb);
}

// ---- the NVFP4 handoff format (options bits 4 and 8): a 512-byte FP8 chunk (32 lanes x 16 bytes) of the runtime's
// layouts is replaced in place by what each lane's pack2 makes of its 16 values: nibbles at lane * 8 (2 words), the two
// scale codes at 256 + lane * 2. Bytes 320..511 of the chunk are not written.
NR_INLINE void put_packed(u8* chunk, u32 lane, u32 n0, u32 n1, u32 c16)
{
    *(uint2*)(chunk + lane * 8u) = make_uint2(n0, n1);
    *(u16*)(chunk + 256u + lane * 2u) = (u16)c16;
}
NR_INLINE void get_packed_cg(const u8* chunk, u32 lane, u32& n0, u32& n1, u32& c)
{
    u16 c16;
    asm volatile("ld.weak.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(n0), "=r"(n1) : "l"((u64)(chunk + lane * 8u)) : "memory");
    asm volatile("ld.weak.global.cg.u16 %0, [%1];" : "=h"(c16) : "l"((u64)(chunk + 256u + lane * 2u)) : "memory");
    c = c16;
}
NR_INLINE void get_packed_sm(const u8* chunk, u32 lane, u32& n0, u32& n1, u32& c)
{
    const uint2 n = *(const uint2*)(chunk + lane * 8u);
    n0 = n.x; n1 = n.y;
    c = *(const u16*)(chunk + 256u + lane * 2u);
}
// A handoff is used only when its request bit (set by the host on both kernels of the pair) and its partner-present
// bit (added by the native layer when the other kernel of the pair is ours in this evaluation) are both set; otherwise
// every kernel reads and writes the runtime's FP8 layouts, so each stays swappable alone.
#define OPT_HQ 4u                                               // request: QKV -> attention, Q, K, V in the handoff format
#define OPT_HA 8u                                               // request: attention -> projection, output likewise
#define OPT_PQ 0x100u                                           // present: the QKV / attention partner is ours
#define OPT_PA 0x200u                                           // present: the attention / projection partner is ours
NR_INLINE u32 effective_options(u32 o)
{
    if (!(o & OPT_PQ)) o &= ~OPT_HQ;
    if (!(o & OPT_PA)) o &= ~OPT_HA;
    return o;
}

NR_INLINE int tiles_of(int n) { return (n - 1) / 128 + 1; }
NR_INLINE int groups_of(int n) { return (n + 31) / 32 * 2; }

// =====================================================================================================================
// QKV: CTA (tile, cg = heads 2 cg, 2 cg + 1, z = input channels 512 z ..); warp (head 2 cg + (w & 1), tokens 64 (w >> 1)
// .. + 63 in two passes of 32). z = 0 leaves float16(acc s) in the scratch, z = 1 adds, normalises and stores.
// record: F[0] s, F[1] tx; +128 c_h[32]; +256 slabs (3 h + which) * 16 + ks
#ifndef QKV_MINB
#define QKV_MINB 2
#endif
#ifndef QKV_PRE                                                 // 1: one wait for the four input flags, loads batched
#define QKV_PRE 1
#endif
extern "C" __global__ void __launch_bounds__(128, QKV_MINB) nr_vit4_qkv(QkvParams p)
{
    __shared__ __align__(16) u8 sm_a[64u * 512u];               // 8 groups x 8 k64 steps
    __shared__ __align__(16) u8 sm_s[64u * 64u];
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int N = p.d0 * p.d1, T = tiles_of(N), G = groups_of(N);
    const u32 cg = blockIdx.x / (u32)T, tile = blockIdx.x - cg * (u32)T, z = blockIdx.z;
    const float* F = (const float*)p.w;
    const float tx = F[1], s = F[0];
    const u32 head = 2u * cg + (w & 1u), th = w >> 1;

    // ---- the input's 512 channels of this z, packed (each warp two groups), chunk by chunk behind the producer's flags
#if QKV_PRE                                                     // all four flags, then 16 loads in flight per half
    if (p.pflags) {
        if (leader) {
            int budget = WAIT_BUDGET_NS;
            for (u32 c = 0; c < 4; ++c) wait_at_least(p.pflags + tile * 8u + 4u * z + c, 0, budget);
            __threadfence();
        }
        __syncthreads();
    }
#pragma unroll
    for (u32 half = 0; half < 2; ++half) {
        V4 xin[2][4][2];
#pragma unroll
        for (u32 q = 0; q < 2; ++q) {
            const u32 grp = tile * 8u + 2u * w + q;
#pragma unroll
            for (u32 k4 = 0; k4 < 4; ++k4)
#pragma unroll
                for (u32 h = 0; h < 2; ++h) {
                    const u32 kb = 16u * z + 2u * (4u * half + k4) + h;
                    xin[q][k4][h] = {0u, 0u, 0u, 0u};
                    if ((int)grp < G) xin[q][k4][h] = load_cg(p.x + (u64)grp * 16384u + kb * 512u + lane * 16u);
                }
        }
#pragma unroll
        for (u32 q = 0; q < 2; ++q)
#pragma unroll
            for (u32 k4 = 0; k4 < 4; ++k4) {
                const u32 gi = 2u * w + q, ks = 4u * half + k4;
                pack_unit(xin[q][k4][0], xin[q][k4][1], sm_a + (gi * 8u + ks) * 512u, sm_s + (gi * 8u + ks) * 64u, tx, lane, g, t);
            }
    }
#else
    for (u32 c = 0; c < 4; ++c) {
        if (p.pflags) {
            if (leader) {
                int budget = WAIT_BUDGET_NS;
                wait_at_least(p.pflags + tile * 8u + 4u * z + c, 0, budget);
                __threadfence();
            }
            __syncthreads();
        }
#pragma unroll
        for (u32 gi = 2u * w; gi < 2u * w + 2u; ++gi) {
            const u32 grp = tile * 8u + gi;
#pragma unroll
            for (u32 k2 = 0; k2 < 2; ++k2) {
                const u32 ks = 2u * c + k2, kb = 16u * z + 2u * ks;
                V4 xa = {0u, 0u, 0u, 0u}, xb = {0u, 0u, 0u, 0u};
                if ((int)grp < G) {
                    xa = load_cg(p.x + (u64)grp * 16384u + kb * 512u + lane * 16u);
                    xb = load_cg(p.x + (u64)grp * 16384u + (kb + 1u) * 512u + lane * 16u);
                }
                pack_unit(xa, xb, sm_a + (gi * 8u + ks) * 512u, sm_s + (gi * 8u + ks) * 64u, tx, lane, g, t);
            }
        }
    }
#endif
    __syncthreads();

    const u8* wslab = p.w + 256u + (u64)(head * 3u) * 16u * SLAB;
    const float ch = ((const float*)(p.w + 128u))[head];
    bool merged = false;
#pragma unroll 1
    for (u32 pass = 0; pass < 2; ++pass) {
        float acc[2][12][4];
#pragma unroll
        for (u32 m = 0; m < 2; ++m)
#pragma unroll
            for (u32 f = 0; f < 12; ++f) acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
#pragma unroll 2
        for (u32 ks = 0; ks < 8; ++ks) {
            V4 a[2];
            u32 sa[2];
#pragma unroll
            for (u32 m = 0; m < 2; ++m) {
                const u32 unit = (4u * th + 2u * pass + m) * 8u + ks;
                a[m] = *(const V4*)(sm_a + unit * 512u + lane * 16u);
                sa[m] = *(const u32*)(sm_s + unit * 64u + g * 8u + (t & 1u) * 4u);
            }
#pragma unroll
            for (u32 which = 0; which < 3; ++which) {
                const u8* sl = wslab + (which * 16u + 8u * z + ks) * SLAB;
                const V4 n01 = *(const V4*)(sl + lane * 16u), n23 = *(const V4*)(sl + 512u + lane * 16u);
                const V4 sw = *(const V4*)(sl + 1024u + g * 16u);
#pragma unroll
                for (u32 m = 0; m < 2; ++m) {
                    mma4(acc[m][which * 4u + 0u], a[m], n01.x, n01.y, sa[m], sw.x);
                    mma4(acc[m][which * 4u + 1u], a[m], n01.z, n01.w, sa[m], sw.y);
                    mma4(acc[m][which * 4u + 2u], a[m], n23.x, n23.y, sa[m], sw.z);
                    mma4(acc[m][which * 4u + 3u], a[m], n23.z, n23.w, sa[m], sw.w);
                }
            }
        }
        // scratch: 6144 B per (32-token pair, head): G / 2 x 32 x 6144 = 3 G x 32768 bytes, the runtime's size
        const u32 grp0 = tile * 8u + 4u * th + 2u * pass;
        const bool store = (int)grp0 < G;                       // groups come in pairs; G is even
        u8* region = p.scratch + (u64)((grp0 >> 1) * 32u + head) * 6144u;
        if (z == 0u) {
            if (!store) continue;
#pragma unroll
            for (u32 m = 0; m < 2; ++m)
#pragma unroll
                for (u32 f2 = 0; f2 < 6; ++f2) {
                    const float* a0 = acc[m][2u * f2];
                    const float* a1 = acc[m][2u * f2 + 1u];
                    const V4 v = {h2_pack(a0[0] * s, a0[1] * s), h2_pack(a0[2] * s, a0[3] * s),
                                  h2_pack(a1[0] * s, a1[1] * s), h2_pack(a1[2] * s, a1[3] * s)};
                    *(V4*)(region + ((m * 6u + f2) * 32u + lane) * 16u) = v;
                }
            continue;
        }
        if (!merged) {                                          // z = 1: the z = 0 CTA of this (tile, cg) is done
            if (leader) {
                int budget = WAIT_BUDGET_NS;
                wait_at_least(p.counter + tile * 16u + cg, 0, budget);
                __threadfence();
            }
            __syncthreads();
            merged = true;
        }
        if (!store) continue;
#pragma unroll
        for (u32 m = 0; m < 2; ++m)
#pragma unroll
            for (u32 f2 = 0; f2 < 6; ++f2) {
                const V4 v = load_cg(region + ((m * 6u + f2) * 32u + lane) * 16u);
                float lo, hi;
                float* a0 = acc[m][2u * f2];
                float* a1 = acc[m][2u * f2 + 1u];
                h2_unpack(v.x, lo, hi); a0[0] = fmaf(a0[0], s, lo); a0[1] = fmaf(a0[1], s, hi);
                h2_unpack(v.y, lo, hi); a0[2] = fmaf(a0[2], s, lo); a0[3] = fmaf(a0[3], s, hi);
                h2_unpack(v.z, lo, hi); a1[0] = fmaf(a1[0], s, lo); a1[1] = fmaf(a1[1], s, hi);
                h2_unpack(v.w, lo, hi); a1[2] = fmaf(a1[2], s, lo); a1[3] = fmaf(a1[3], s, hi);
            }
        // ---- normalise and store Q (act layout), K (key layout), V (value layout) ----------------------------------
        u32 T8[2][2][4];                                       // V as float16 pairs before / after the transposition
#pragma unroll
        for (u32 m = 0; m < 2; ++m) {
            const u32 grp = grp0 + m;
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const bool valid = (int)(grp * 16u + g + 8u * r) < N;
                if (!valid)
#pragma unroll
                    for (u32 f = 0; f < 12; ++f) acc[m][f][2u * r] = acc[m][f][2u * r + 1u] = 0.0f;
            }
            float inv[2][2];                                    // [q, k][row half]
#pragma unroll
            for (u32 qk = 0; qk < 2; ++qk)
#pragma unroll
                for (u32 r = 0; r < 2; ++r) {
                    const float* c0 = acc[m][4u * qk + 0u];
                    const float* c1 = acc[m][4u * qk + 1u];
                    const float* c2 = acc[m][4u * qk + 2u];
                    const float* c3 = acc[m][4u * qk + 3u];
                    const u32 e = 2u * r;
                    float sum = ((c0[e] * c0[e] + c0[e + 1] * c0[e + 1]) + (c1[e] * c1[e] + c1[e + 1] * c1[e + 1])) +
                                ((c2[e] * c2[e] + c2[e + 1] * c2[e + 1]) + (c3[e] * c3[e] + c3[e + 1] * c3[e + 1]));
                    sum = sum + __shfl_xor_sync(0xffffffffu, sum, 1);
                    sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
                    inv[qk][r] = rsqrt_approx(fmaxf(sum, FLOOR));
                }
            const float fq0 = inv[0][0] * ch, fq1 = inv[0][1] * ch;
            if (store && (effective_options(p.options) & OPT_HQ)) {                // the handoff: float16 values packed, no e4m3 step
                const float k0 = inv[1][0], k1 = inv[1][1];
                u32 h0[4], h1[4], n0, n1;
#pragma unroll
                for (u32 f = 0; f < 4; ++f) {
                    h0[f] = h2_pack(acc[m][f][0] * fq0, acc[m][f][1] * fq0);
                    h1[f] = h2_pack(acc[m][f][2] * fq1, acc[m][f][3] * fq1);
                }
                u32 c = pack2(h0, h1, F[3], n0, n1);
                put_packed(p.q + (u64)grp * 16384u + head * 512u, lane, n0, n1, c);
#pragma unroll
                for (u32 f = 0; f < 4; ++f) {
                    h0[f] = h2_pack(acc[m][4u + f][0] * k0, acc[m][4u + f][1] * k0);
                    h1[f] = h2_pack(acc[m][4u + f][2] * k1, acc[m][4u + f][3] * k1);
                }
                c = pack2(h0, h1, F[4], n0, n1);
                put_packed(p.k + (u64)grp * 16384u + head * 512u, lane, n0, n1, c);
            } else if (store) {
                const float* q = acc[m][0];
                const V4 qv = {f32x4_to_e4m3x4(q[0] * fq0, q[1] * fq0, acc[m][1][0] * fq0, acc[m][1][1] * fq0),
                               f32x4_to_e4m3x4(q[2] * fq1, q[3] * fq1, acc[m][1][2] * fq1, acc[m][1][3] * fq1),
                               f32x4_to_e4m3x4(acc[m][2][0] * fq0, acc[m][2][1] * fq0, acc[m][3][0] * fq0, acc[m][3][1] * fq0),
                               f32x4_to_e4m3x4(acc[m][2][2] * fq1, acc[m][2][3] * fq1, acc[m][3][2] * fq1, acc[m][3][3] * fq1)};
                *(V4*)(p.q + (u64)grp * 16384u + head * 512u + lane * 16u) = qv;
                const float k0 = inv[1][0], k1 = inv[1][1];
                const V4 kv = {f32x4_to_e4m3x4(acc[m][4][0] * k0, acc[m][4][1] * k0, acc[m][5][0] * k0, acc[m][5][1] * k0),
                               f32x4_to_e4m3x4(acc[m][6][0] * k0, acc[m][6][1] * k0, acc[m][7][0] * k0, acc[m][7][1] * k0),
                               f32x4_to_e4m3x4(acc[m][4][2] * k1, acc[m][4][3] * k1, acc[m][5][2] * k1, acc[m][5][3] * k1),
                               f32x4_to_e4m3x4(acc[m][6][2] * k1, acc[m][6][3] * k1, acc[m][7][2] * k1, acc[m][7][3] * k1)};
                *(V4*)(p.k + (u64)grp * 16384u + head * 512u + lane * 16u) = kv;
            }
#pragma unroll
            for (u32 r = 0; r < 2; ++r)
#pragma unroll
                for (u32 f = 0; f < 4; ++f)
                    T8[m][r][f] = transpose8(h2_pack(acc[m][8u + f][2u * r], acc[m][8u + f][2u * r + 1u]));
        }
        if (store && (effective_options(p.options) & OPT_HQ)) {
#pragma unroll
            for (u32 sl = 0; sl < 2; ++sl) {
                const u32 h0[4] = {T8[0][0][2u * sl], T8[0][1][2u * sl], T8[1][0][2u * sl], T8[1][1][2u * sl]};
                const u32 h1[4] = {T8[0][0][2u * sl + 1u], T8[0][1][2u * sl + 1u], T8[1][0][2u * sl + 1u], T8[1][1][2u * sl + 1u]};
                u32 n0, n1;
                const u32 c = pack2(h0, h1, F[5], n0, n1);
                put_packed(p.v + (u64)(tile * 4u + th * 2u + pass) * 32768u + (2u * head + sl) * 512u, lane, n0, n1, c);
            }
        } else if (store) {
#pragma unroll
            for (u32 sl = 0; sl < 2; ++sl) {
                u32 word[4];
#pragma unroll
                for (u32 d = 0; d < 2; ++d)
#pragma unroll
                    for (u32 m = 0; m < 2; ++m)
                        word[d * 2u + m] = h2_to_e4m3x2(T8[m][0][2u * sl + d]) | (h2_to_e4m3x2(T8[m][1][2u * sl + d]) << 16);
                const V4 vv = {word[0], word[1], word[2], word[3]};
                *(V4*)(p.v + (u64)(tile * 4u + th * 2u + pass) * 32768u + (2u * head + sl) * 512u + lane * 16u) = vv;
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (leader) {
        flag_publish(p.counter + tile * 16u + cg, (int)z);
        if (z == 1u && p.oflags) flag_publish(p.oflags + tile * 16u + cg, 0);
    }
}

// =====================================================================================================================
// Attention: CTA (head, 256 queries = tiles 2 y, 2 y + 1), warp 64 queries; key groups of 64 double-buffered in shared
// memory (cp.async), polled tile by tile behind the QKV flags.
// record (options only): F[0] tp, F[1] tv, F[2] tq, F[3] tk, F[4] 1 / (tp tv), F[5] 1 / (tq tk)
template <u32 OPTS>
NR_INLINE void attention_body(const AttParams& p, u8 (*sm_kv)[4096 * ATT_SG])
{
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, g = lane >> 2, t = lane & 3u;
    const u32 tid = w * 32u + lane;
    const bool leader = tid == 0u;
    const int N = p.d0 * p.d1, T = tiles_of(N), G = groups_of(N), KG = (N - 1) / 64 + 1;
    const u32 head = blockIdx.x, y = blockIdx.y;
    constexpr bool hq = (OPTS & OPT_HQ) != 0u, ha = (OPTS & OPT_HA) != 0u;   // handoff in / out
    constexpr bool pv4 = (OPTS & OPT_PV4) != 0u || hq, qk4 = (OPTS & OPT_QK4) != 0u || hq;
    const float* F = (const float*)p.w;

    if (p.pflags) {                                             // Q of this CTA's two tiles, K and V of stages 0 and 1
        if (leader) {
            int budget = WAIT_BUDGET_NS;
            for (int tt = 0; tt < min(ATT_SG, T); ++tt) wait_at_least(p.pflags + tt * 16 + (head >> 1), 0, budget);
            for (int tt = 2 * (int)y; tt < min(2 * (int)y + 2, T); ++tt) wait_at_least(p.pflags + tt * 16 + (head >> 1), 0, budget);
            __threadfence();
        }
        __syncthreads();
    }
    // Q: A operands of the warp's four 16-query groups
    u32 QA[4][4];
    u32 QS[4];                                                  // QK4: scale words
#pragma unroll
    for (u32 m = 0; m < 4; ++m) {
        const u32 grp = 16u * y + 4u * w + m;
        if (hq) {                                               // already packed by our QKV
            u32 n0 = 0u, n1 = 0u, c = 0u;
            if ((int)grp < G) get_packed_cg(p.q + (u64)grp * 16384u + head * 512u, lane, n0, n1, c);
            QA[m][0] = n0; QA[m][1] = n1; QA[m][2] = 0u; QA[m][3] = 0u;
            QS[m] = scale_word(c, 0u, t, t & 1u) & 0xFFFFu;
            continue;
        }
        V4 v = {0u, 0u, 0u, 0u};
        if ((int)grp < G) v = load_cg(p.q + (u64)grp * 16384u + head * 512u + lane * 16u);
        QA[m][0] = v.x; QA[m][1] = v.y; QA[m][2] = v.z; QA[m][3] = v.w;
        QS[m] = 0u;
        if (qk4) {
            u32 h0[4], h1[4], n0, n1;
            act_rows(v, h0, h1);
            const u32 c = pack2(h0, h1, F[2], n0, n1);
            QA[m][0] = n0; QA[m][1] = n1; QA[m][2] = 0u; QA[m][3] = 0u;
            QS[m] = scale_word(c, 0u, t, t & 1u) & 0xFFFFu;
        }
    }

    const int NS = (KG + ATT_SG - 1) / ATT_SG;                  // stages of ATT_SG key groups
    auto issue = [&](int s, u32 stage) {
#pragma unroll
        for (u32 r = 0; r < 2u * ATT_SG; ++r) {
            const u32 c = tid + 128u * r, u = c >> 8, part = (c >> 5) & 7u, off = (c & 31u) * 16u;
            const u32 i = (u32)s * ATT_SG + u;
            u8* dst = &sm_kv[stage][u * 4096u + part * 512u + off];
            const u32 kgrp = 4u * i + part;                     // K: 16-key groups 4 i .. 4 i + 3
            const u32 vblk = 2u * i + ((part - 4u) >> 1), vsl = 2u * head + ((part - 4u) & 1u);
            const bool inside = (int)i < KG && (part < 4u ? (int)kgrp < G : (int)(2u * vblk) < G);
            if (inside) {
                const u8* src = part < 4u ? p.k + (u64)kgrp * 16384u + head * 512u + off
                                          : p.v + (u64)vblk * 32768u + vsl * 512u + off;
                cp_async16(dst, src);
            } else {
                const V4 zero = {0u, 0u, 0u, 0u};
                *(V4*)dst = zero;
            }
        }
        cp_commit();
    };
    issue(0, 0u);
    if (NS > 1) issue(1, 1u);

    u32 den[2];                                                 // f16 pair (both halves equal) of row (set, t)
    den[0] = den[1] = 0u;
    u32 acc8[4][4][2];                                          // FP8: float16 accumulators [m][dims 8 f][row half]
    float acc4[4][4][4];                                        // PV4: float32 accumulators
#pragma unroll
    for (u32 m = 0; m < 4; ++m)
#pragma unroll
        for (u32 f = 0; f < 4; ++f) {
            acc8[m][f][0] = acc8[m][f][1] = 0u;
            acc4[m][f][0] = acc4[m][f][1] = acc4[m][f][2] = acc4[m][f][3] = 0.0f;
        }
    const float rqk = F[5];
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const u32 stage = (u32)s & 1u;
        if (s + 1 < NS) cp_wait1(); else cp_wait0();
        __syncthreads();
#pragma unroll 1
        for (int u = 0; u < ATT_SG; ++u) {
        if (s * ATT_SG + u >= KG) break;
        const u8* ks_ = sm_kv[stage] + u * 4096;
        const u8* vs_ = ks_ + 2048u;
        // logits [m][key fragment n][row half] as float16 pairs
        u32 S[4][8][2];
#if DBG_STAGE < 2                                               // timing stage 1: the K and V reads only
        if (DBG_STAGE == 1) {
#pragma unroll
            for (u32 jj = 0; jj < 8; ++jj) {
                const V4 kv = *(const V4*)(ks_ + jj * 512u + lane * 16u);
                den[jj & 1u] ^= kv.x ^ kv.y ^ kv.z ^ kv.w;
            }
        }
        continue;
#endif
        if (!qk4) {
#pragma unroll
            for (u32 jj = 0; jj < 4; ++jj) {
                const V4 kv = *(const V4*)(ks_ + jj * 512u + lane * 16u);
#pragma unroll
                for (u32 m = 0; m < 4; ++m) {
                    S[m][2u * jj][0] = S[m][2u * jj][1] = 0u;
                    S[m][2u * jj + 1u][0] = S[m][2u * jj + 1u][1] = 0u;
                    mma8(S[m][2u * jj], QA[m], kv.x, kv.y);
                    mma8(S[m][2u * jj + 1u], QA[m], kv.z, kv.w);
                }
            }
        } else {
#pragma unroll
            for (u32 jj = 0; jj < 4; ++jj) {
                u32 nk0, nk1, c;
                if (hq) {
                    get_packed_sm(ks_ + jj * 512u, lane, nk0, nk1, c);
                } else {
                    const V4 kv = *(const V4*)(ks_ + jj * 512u + lane * 16u);
                    u32 h0[4], h1[4];
                    e4m3x4_to_h2(kv.x, h0[0], h0[1]); e4m3x4_to_h2(kv.y, h0[2], h0[3]);  // key g: dims j 0, 1 | 2, 3
                    e4m3x4_to_h2(kv.z, h1[0], h1[1]); e4m3x4_to_h2(kv.w, h1[2], h1[3]);  // key g + 8
                    c = pack2(h0, h1, F[3], nk0, nk1);
                }
                const u32 sb0 = scale_word(c, 0u, t, 0u) & 0xFFFFu, sb1 = scale_word(c, 0u, t, 1u) & 0xFFFFu;
#pragma unroll
                for (u32 m = 0; m < 4; ++m) {
                    const V4 a = {QA[m][0], QA[m][1], 0u, 0u};
                    float l0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, l1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma4(l0, a, nk0, 0u, QS[m], sb0);
                    mma4(l1, a, nk1, 0u, QS[m], sb1);
                    S[m][2u * jj][0] = h2_pack(l0[0] * rqk, l0[1] * rqk);
                    S[m][2u * jj][1] = h2_pack(l0[2] * rqk, l0[3] * rqk);
                    S[m][2u * jj + 1u][0] = h2_pack(l1[0] * rqk, l1[1] * rqk);
                    S[m][2u * jj + 1u][1] = h2_pack(l1[2] * rqk, l1[3] * rqk);
                }
            }
        }
        // the network's exponential: fma, clamp, mantissa << 4 read as a float16
        u32 X[8];                                               // row partial sums [m * 2 + r] (keys 8 n + 2 t + e)
#pragma unroll
        for (u32 m = 0; m < 4; ++m)
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
#pragma unroll
                for (u32 n = 0; n < 8; ++n) {
                    if (DBG_STAGE == 2) continue;                // timing stage 2: logits summed raw, no exponential
                    u32 x = h2_fma(S[m][n][r], 0x2DBB2DBBu, 0x3ED63ED6u);
                    x = h2_min(h2_max(x, 0x3DC23DC2u), 0x3FE93FE9u);
                    S[m][n][r] = (x << 4) + 0x3FFC4000u;
                }
                X[m * 2u + r] = h2_add(h2_add(h2_add(h2_add(S[m][0][r], S[m][1][r]), h2_add(S[m][2][r], S[m][3][r])),
                                              h2_add(S[m][4][r], S[m][5][r])), h2_add(S[m][6][r], S[m][7][r]));
            }
        // the four lanes t in order ((t0 + t1) + t2) + t3: lane t ends with row index t of each set of four
#if ATT_SEL                                                     // two-level selects on the bits of t (no branches)
        const bool b1 = (t & 1u) != 0u, b2 = (t & 2u) != 0u;
#pragma unroll
        for (u32 set = 0; set < 2; ++set) {
            const u32* x = X + 4u * set;
            u32 term[4];
#pragma unroll
            for (u32 k = 0; k < 4; ++k) {                       // send x[t ^ k]
                const bool s1 = b1 != ((k & 1u) != 0u), s2 = b2 != ((k & 2u) != 0u);
                const u32 lo = s1 ? x[1] : x[0], hi = s1 ? x[3] : x[2];
                const u32 send = s2 ? hi : lo;
                term[k] = k ? __shfl_xor_sync(0xffffffffu, send, k) : send;     // = value t of lane t ^ k
            }
            u32 by[4];                                          // by source lane s2 = t ^ k: term[s2 ^ t]
#pragma unroll
            for (u32 s = 0; s < 4; ++s) {
                const bool s1 = b1 != ((s & 1u) != 0u), s2 = b2 != ((s & 2u) != 0u);
                const u32 lo = s1 ? term[1] : term[0], hi = s1 ? term[3] : term[2];
                by[s] = s2 ? hi : lo;
            }
            const u32 sum = h2_add(h2_add(h2_add(by[0], by[1]), by[2]), by[3]);
            den[set] = h2_add(den[set], h2_add(sum, prmt(sum, sum, 0x1032u)));
        }
#else
#pragma unroll
        for (u32 set = 0; set < 2; ++set) {
            const u32* x = X + 4u * set;
            u32 term[4];
#pragma unroll
            for (u32 k = 0; k < 4; ++k) {
                const u32 send = (t ^ k) == 0u ? x[0] : (t ^ k) == 1u ? x[1] : (t ^ k) == 2u ? x[2] : x[3];
                term[k] = k ? __shfl_xor_sync(0xffffffffu, send, k) : send;     // = value t of lane t ^ k
            }
            u32 by[4];                                          // by source lane s = t ^ k
#pragma unroll
            for (u32 s2 = 0; s2 < 4; ++s2) {
                const u32 k = s2 ^ t;
                by[s2] = k == 0u ? term[0] : k == 1u ? term[1] : k == 2u ? term[2] : term[3];
            }
            const u32 sum = h2_add(h2_add(h2_add(by[0], by[1]), by[2]), by[3]);
            den[set] = h2_add(den[set], h2_add(sum, prmt(sum, sum, 0x1032u)));
        }
#endif
        if (DBG_STAGE < 4) continue;                            // timing stage 3: no P.V
        if (!pv4 && ATT_PVORD) {
#pragma unroll
            for (u32 kh = 0; kh < 2; ++kh) {
                V4 vv[2];
#pragma unroll
                for (u32 sl = 0; sl < 2; ++sl) vv[sl] = *(const V4*)(vs_ + (kh * 2u + sl) * 512u + lane * 16u);
#pragma unroll
                for (u32 m = 0; m < 4; ++m) {
                    const u32 n = 4u * kh;
                    const u32 PA[4] = {h2x2_to_e4m3x4(S[m][n][0], S[m][n + 1u][0]), h2x2_to_e4m3x4(S[m][n][1], S[m][n + 1u][1]),
                                       h2x2_to_e4m3x4(S[m][n + 2u][0], S[m][n + 3u][0]), h2x2_to_e4m3x4(S[m][n + 2u][1], S[m][n + 3u][1])};
#pragma unroll
                    for (u32 sl = 0; sl < 2; ++sl) {
                        mma8(acc8[m][2u * sl], PA, vv[sl].x, vv[sl].y);
                        mma8(acc8[m][2u * sl + 1u], PA, vv[sl].z, vv[sl].w);
                    }
                }
            }
        } else if (!pv4) {
#pragma unroll
            for (u32 m = 0; m < 4; ++m)
#pragma unroll
                for (u32 kh = 0; kh < 2; ++kh) {
                    const u32 n = 4u * kh;
                    const u32 PA[4] = {h2x2_to_e4m3x4(S[m][n][0], S[m][n + 1u][0]), h2x2_to_e4m3x4(S[m][n][1], S[m][n + 1u][1]),
                                       h2x2_to_e4m3x4(S[m][n + 2u][0], S[m][n + 3u][0]), h2x2_to_e4m3x4(S[m][n + 2u][1], S[m][n + 3u][1])};
#pragma unroll
                    for (u32 sl = 0; sl < 2; ++sl) {
                        const V4 vv = *(const V4*)(vs_ + (kh * 2u + sl) * 512u + lane * 16u);
                        mma8(acc8[m][2u * sl], PA, vv.x, vv.y);
                        mma8(acc8[m][2u * sl + 1u], PA, vv.z, vv.w);
                    }
                }
        } else {
            // V: B operands per dims 8 f + g, keys 32 hh + 8 j + 2 t + e (this lane's own bytes)
            u32 VB[4][2], SB[4];
#pragma unroll
            for (u32 sl = 0; sl < 2; ++sl) {
                u32 pair[2];
#pragma unroll
                for (u32 hh = 0; hh < 2; ++hh) {
                    if (hq) {
                        get_packed_sm(vs_ + (hh * 2u + sl) * 512u, lane, VB[2u * sl][hh], VB[2u * sl + 1u][hh], pair[hh]);
                        continue;
                    }
                    const V4 vv = *(const V4*)(vs_ + (hh * 2u + sl) * 512u + lane * 16u);
                    u32 h0[4], h1[4];
                    e4m3x4_to_h2(vv.x, h0[0], h0[1]); e4m3x4_to_h2(vv.y, h0[2], h0[3]);
                    e4m3x4_to_h2(vv.z, h1[0], h1[1]); e4m3x4_to_h2(vv.w, h1[2], h1[3]);
                    pair[hh] = pack2(h0, h1, F[1], VB[2u * sl][hh], VB[2u * sl + 1u][hh]);
                }
                SB[2u * sl] = scale_word(pair[0], pair[1], t, 0u);
                SB[2u * sl + 1u] = scale_word(pair[0], pair[1], t, 1u);
            }
#pragma unroll
            for (u32 m = 0; m < 4; ++m) {
                u32 n0a, n1a, n0b, n1b;
                const u32 h0a[4] = {S[m][0][0], S[m][1][0], S[m][2][0], S[m][3][0]}, h1a[4] = {S[m][0][1], S[m][1][1], S[m][2][1], S[m][3][1]};
                const u32 ca = pack2(h0a, h1a, F[0], n0a, n1a);
                const u32 h0b[4] = {S[m][4][0], S[m][5][0], S[m][6][0], S[m][7][0]}, h1b[4] = {S[m][4][1], S[m][5][1], S[m][6][1], S[m][7][1]};
                const u32 cb = pack2(h0b, h1b, F[0], n0b, n1b);
                const V4 a = {n0a, n1a, n0b, n1b};
                const u32 sa = scale_word(ca, cb, t, t & 1u);
#pragma unroll
                for (u32 f = 0; f < 4; ++f) mma4(acc4[m][f], a, VB[f][0], VB[f][1], sa, SB[f]);
            }
        }
        }                                                       // the groups of this stage
        __syncthreads();                                        // everybody is done with this stage
        if (s + 2 < NS) {
            if (p.pflags) {
                if (leader) {                                   // the tiles of stage s + 2
                    int budget = WAIT_BUDGET_NS;
                    const int i0 = (s + 2) * ATT_SG, i1 = min(KG, i0 + ATT_SG);
                    for (int tt = i0 >> 1; tt <= (i1 - 1) >> 1; ++tt) wait_at_least(p.pflags + tt * 16 + (head >> 1), 0, budget);
                    __threadfence();
                }
                __syncthreads();
            }
            issue(s + 2, stage);
        }
    }
    // ---- denominators: padding keys out, reciprocal, to the lanes that hold each row ---------------------------------
    const float pad = f16_round((float)(64 * KG - N) * 0.083984375f);
    float inv_row[2];
#pragma unroll
    for (u32 set = 0; set < 2; ++set) {
        float lo, hi;
        h2_unpack(den[set], lo, hi);
        const float d = f16_round(lo - pad);
        inv_row[set] = f16_round(rcp_approx(fmaxf(d, FLOOR)));
    }
    float inv[4][2];                                            // [m][row half]
#pragma unroll
    for (u32 m = 0; m < 4; ++m)
#pragma unroll
        for (u32 r = 0; r < 2; ++r)
            inv[m][r] = __shfl_sync(0xffffffffu, inv_row[m >> 1], (lane & ~3u) | (((m & 1u) << 1) | r));
    const float rpv = F[4];
#pragma unroll
    for (u32 m = 0; m < 4; ++m) {
        const u32 grp = 16u * y + 4u * w + m;
        if ((int)grp >= G) continue;
        if (ha) {                                               // the handoff: the float16 output packed for our projection
            u32 h[2][4];                                        // [row half][dims 8 f + 2 t + e: f]
#pragma unroll
            for (u32 r = 0; r < 2; ++r) {
                const bool valid = (int)(grp * 16u + g + 8u * r) < N;
#pragma unroll
                for (u32 f = 0; f < 4; ++f) {
                    u32 hv;
                    if (!pv4) {
                        hv = h2_mul(acc8[m][f][r], h2_pack(inv[m][r], inv[m][r]));
                    } else {
                        const float fct = inv[m][r] * rpv;
                        hv = h2_pack(acc4[m][f][2u * r] * fct, acc4[m][f][2u * r + 1u] * fct);
                    }
                    h[r][f] = valid ? hv : 0u;
                }
            }
            u32 n0, n1;
            const u32 c = pack2(h[0], h[1], F[6], n0, n1);
            put_packed(p.out + (u64)grp * 16384u + head * 512u, lane, n0, n1, c);
            continue;
        }
        u32 word[2][2][2];                                      // [dims 16 half][row half][pair of fragments]
#pragma unroll
        for (u32 r = 0; r < 2; ++r) {
            const bool valid = (int)(grp * 16u + g + 8u * r) < N;
#pragma unroll
            for (u32 fh = 0; fh < 2; ++fh) {
                u32 wv;
                if (!pv4) {
                    const u32 iv = h2_pack(inv[m][r], inv[m][r]);
                    wv = h2x2_to_e4m3x4(h2_mul(acc8[m][2u * fh][r], iv), h2_mul(acc8[m][2u * fh + 1u][r], iv));
                } else {
                    const float fct = inv[m][r] * rpv;
                    const float* a0 = acc4[m][2u * fh];
                    const float* a1 = acc4[m][2u * fh + 1u];
                    wv = f32x4_to_e4m3x4(a0[2u * r] * fct, a0[2u * r + 1u] * fct, a1[2u * r] * fct, a1[2u * r + 1u] * fct);
                }
                word[fh][r][0] = valid ? wv : 0u;
            }
        }
        const V4 ov = {word[0][0][0], word[0][1][0], word[1][0][0], word[1][1][0]};
        *(V4*)(p.out + (u64)grp * 16384u + head * 512u + lane * 16u) = ov;
    }
    __threadfence();
    __syncthreads();
    if (leader && p.oflags) {
        if ((int)(2u * y) < T) flag_publish(p.oflags + (2u * y) * 32u + head, 0);
        if ((int)(2u * y + 1u) < T) flag_publish(p.oflags + (2u * y + 1u) * 32u + head, 0);
    }
}

#ifndef ATT_MINB                                                // CTAs per SM promised to ptxas (register limit 65536 / 128 / n)
#define ATT_MINB 2                                              // 3 is faster alone, slower in the chain (R2.5)
#endif
extern "C" __global__ void __launch_bounds__(128, ATT_MINB) nr_vit4_attention(AttParams p)
{
    __shared__ __align__(16) u8 sm_kv[2][4096 * ATT_SG];        // [stage][group][K 4 x 512 | V 4 x 512]
#ifdef ATT_OPTS                                                 // analysis builds: one option set only
    attention_body<ATT_OPTS>(p, sm_kv);
#else
    // bits 1, 2: NVFP4 arithmetic; 4: Q, K, V arrive packed (implies both NVFP4 products); 8: output leaves packed
    u32 o = effective_options(p.options) & 15u;
    if (o & OPT_HQ) o = (o & OPT_HA) | 7u;
    switch (o) {
    case 0u: attention_body<0u>(p, sm_kv); break;
    case 1u: attention_body<1u>(p, sm_kv); break;
    case 2u: attention_body<2u>(p, sm_kv); break;
    case 3u: attention_body<3u>(p, sm_kv); break;
    case 7u: attention_body<7u>(p, sm_kv); break;
    case 8u: attention_body<8u>(p, sm_kv); break;
    case 9u: attention_body<9u>(p, sm_kv); break;
    case 10u: attention_body<10u>(p, sm_kv); break;
    case 11u: attention_body<11u>(p, sm_kv); break;
    default: attention_body<15u>(p, sm_kv); break;
    }
#endif
}

// =====================================================================================================================
// Projection: CTA (tile, cg = outputs 128 cg .., z = input channels 256 z .. = heads 8 z ..); warp (outputs 64 (w & 1)
// .. + 63, tokens 64 (w >> 1) .. in two passes of 32). Merge in z order through the float16 scratch:
// v0 = f16(fma(acc0, s, gain skip)), v_z = f16(fma(acc_z, s, v_{z-1})), out = e4m3(v3).
// record: F[0] s, F[1] ta; +128 gain f16[1024]; +2304 slabs (output / 32) * 16 + ks
#ifndef PROJ_MINB
#define PROJ_MINB 2
#endif
#ifndef PROJ_ZS                                                 // CTAs per (tile, cg) that work: 4 (each one slice) or 2
#define PROJ_ZS 4
#endif
#ifndef PROJ_WSM                                                // 1: the CTA's weight slabs staged in shared memory (cp.async)
#define PROJ_WSM 1
#endif
#ifndef PROJ_PRE                                                // 1: one wait for all input flags, all input loads in flight
#define PROJ_PRE 1
#endif
extern "C" __global__ void __launch_bounds__(128, PROJ_MINB) nr_vit4_projection(ProjParams p)
{
    __shared__ __align__(16) u8 sm_a[32u * 512u];               // 8 groups x 4 k64 steps
    __shared__ __align__(16) u8 sm_s[32u * 64u];
    const u32 lane = threadIdx.x & 31u, w = threadIdx.y, g = lane >> 2, t = lane & 3u;
    const bool leader = (threadIdx.x | threadIdx.y) == 0;
    const int N = p.d0 * p.d1, T = tiles_of(N), G = groups_of(N);
    const u32 cg = blockIdx.x / (u32)T, tile = blockIdx.x - cg * (u32)T, z = blockIdx.z;
    const float* F = (const float*)p.w;
    const float s = F[0], ta = F[1];
    const u32 ob = cg * 128u + 64u * (w & 1u), th = w >> 1;

#if PROJ_ZS == 2
    // Two CTAs per (tile, cg) do the work, each two 256-channel slices (z = 0: slices 0, 1; z = 1: slices 2, 3); the
    // CTAs z = 2, 3 leave at once. Per pass of 32 tokens: both slices packed (this pass's four groups), both slices'
    // accumulators computed, then the runtime's merge order in registers: v = f16(fma(acc_slice, s, v_previous)),
    // exactly the four-CTA rounding. Counter: CTA 0 publishes 0, CTA 1 publishes 3 (the value the runtime leaves).
    if (z >= 2u) return;
    const u8* wslab = p.w + 2304u + (u64)(ob >> 5) * 16u * SLAB;
    const u8* gain = p.w + 128u;
    const bool last = z == 1u;
#pragma unroll 1
    for (u32 pq = 0; pq < 2; ++pq) {
        if (pq) __syncthreads();                                // pass 0's units are consumed
        // pack: warp w -> local group w (gi = 4 (w >> 1) + 2 pq + (w & 1)), slices c, steps ks: unit (c * 4 + w) * 4 + ks
#pragma unroll
        for (u32 c = 0; c < 2; ++c) {
            const u32 zz = 2u * z + c;
            for (u32 ks = 0; ks < 4; ++ks) {
                if (p.pflags && pq == 0u) {
                    if (leader) {
                        int budget = WAIT_BUDGET_NS;
                        wait_at_least(p.pflags + tile * 32u + 8u * zz + 2u * ks, 0, budget);
                        wait_at_least(p.pflags + tile * 32u + 8u * zz + 2u * ks + 1u, 0, budget);
                        __threadfence();
                    }
                    __syncthreads();
                }
                const u32 gi = 4u * (w >> 1) + 2u * pq + (w & 1u);
                const u32 grp = tile * 8u + gi, kb = 8u * zz + 2u * ks;
                V4 xa = {0u, 0u, 0u, 0u}, xb = {0u, 0u, 0u, 0u};
                if ((int)grp < G) {
                    xa = load_cg(p.x + (u64)grp * 16384u + kb * 512u + lane * 16u);
                    xb = load_cg(p.x + (u64)grp * 16384u + (kb + 1u) * 512u + lane * 16u);
                }
                const u32 unit = (c * 4u + w) * 4u + ks;
                pack_unit(xa, xb, sm_a + unit * 512u, sm_s + unit * 64u, ta, lane, g, t);
            }
        }
        __syncthreads();
        float acc[2][2][8][4];                                  // [slice][m][f][e]
#pragma unroll
        for (u32 c = 0; c < 2; ++c)
#pragma unroll
            for (u32 m = 0; m < 2; ++m)
#pragma unroll
                for (u32 f = 0; f < 8; ++f) acc[c][m][f][0] = acc[c][m][f][1] = acc[c][m][f][2] = acc[c][m][f][3] = 0.0f;
#pragma unroll
        for (u32 c = 0; c < 2; ++c)
#pragma unroll
            for (u32 ks = 0; ks < 4; ++ks) {
                V4 a[2];
                u32 sa[2];
#pragma unroll
                for (u32 m = 0; m < 2; ++m) {
                    const u32 unit = (c * 4u + 2u * th + m) * 4u + ks;
                    a[m] = *(const V4*)(sm_a + unit * 512u + lane * 16u);
                    sa[m] = *(const u32*)(sm_s + unit * 64u + g * 8u + (t & 1u) * 4u);
                }
#pragma unroll
                for (u32 o2 = 0; o2 < 2; ++o2) {
                    const u8* sl = wslab + (o2 * 16u + 4u * (2u * z + c) + ks) * SLAB;
                    const V4 n01 = *(const V4*)(sl + lane * 16u), n23 = *(const V4*)(sl + 512u + lane * 16u);
                    const V4 sw = *(const V4*)(sl + 1024u + g * 16u);
#pragma unroll
                    for (u32 m = 0; m < 2; ++m) {
                        mma4(acc[c][m][o2 * 4u + 0u], a[m], n01.x, n01.y, sa[m], sw.x);
                        mma4(acc[c][m][o2 * 4u + 1u], a[m], n01.z, n01.w, sa[m], sw.y);
                        mma4(acc[c][m][o2 * 4u + 2u], a[m], n23.x, n23.y, sa[m], sw.z);
                        mma4(acc[c][m][o2 * 4u + 3u], a[m], n23.z, n23.w, sa[m], sw.w);
                    }
                }
            }
        if (z == 1u && pq == 0u) {                              // CTA 0 has merged slices 0 and 1
            if (leader) {
                int budget = WAIT_BUDGET_NS;
                wait_at_least(p.counter + tile * 8u + cg, 0, budget);
                __threadfence();
            }
            __syncthreads();
        }
        const u32 grp0 = tile * 8u + 4u * th + 2u * pq;
        u8* region = p.scratch + (u64)((grp0 >> 1) * 16u + (ob >> 6)) * 4096u;
#pragma unroll
        for (u32 m = 0; m < 2; ++m) {
            const u32 grp = grp0 + m;
            if ((int)grp >= G) continue;
            float prev[8][4];
            if (z == 0u) {
#pragma unroll
                for (u32 fq = 0; fq < 2; ++fq) {
                    const V4 sk = load_cg(p.skip + (u64)grp * 16384u + ((ob >> 5) + fq) * 512u + lane * 16u);
                    u32 h0[4], h1[4];
                    act_rows(sk, h0, h1);
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) {
                        const u32 f = 4u * fq + j;
                        float g0, g1, a0, a1, b0, b1;
                        h2_unpack(load_cg32(gain + (ob + 8u * f + 2u * t) * 2u), g0, g1);
                        h2_unpack(h0[j], a0, a1);
                        h2_unpack(h1[j], b0, b1);
                        prev[f][0] = g0 * a0; prev[f][1] = g1 * a1; prev[f][2] = g0 * b0; prev[f][3] = g1 * b1;
                    }
                }
            } else {
#pragma unroll
                for (u32 f2 = 0; f2 < 4; ++f2) {
                    const V4 v = load_cg(region + ((m * 4u + f2) * 32u + lane) * 16u);
                    h2_unpack(v.x, prev[2u * f2][0], prev[2u * f2][1]); h2_unpack(v.y, prev[2u * f2][2], prev[2u * f2][3]);
                    h2_unpack(v.z, prev[2u * f2 + 1u][0], prev[2u * f2 + 1u][1]); h2_unpack(v.w, prev[2u * f2 + 1u][2], prev[2u * f2 + 1u][3]);
                }
            }
            float val[8][4];
#pragma unroll
            for (u32 f = 0; f < 8; ++f)
#pragma unroll
                for (u32 e = 0; e < 4; ++e) val[f][e] = f16_round(fmaf(acc[0][m][f][e], s, prev[f][e]));
#pragma unroll
            for (u32 f = 0; f < 8; ++f)
#pragma unroll
                for (u32 e = 0; e < 4; ++e) val[f][e] = fmaf(acc[1][m][f][e], s, val[f][e]);
            if (!last) {
#pragma unroll
                for (u32 f2 = 0; f2 < 4; ++f2) {
                    const V4 v = {h2_pack(val[2u * f2][0], val[2u * f2][1]), h2_pack(val[2u * f2][2], val[2u * f2][3]),
                                  h2_pack(val[2u * f2 + 1u][0], val[2u * f2 + 1u][1]), h2_pack(val[2u * f2 + 1u][2], val[2u * f2 + 1u][3])};
                    *(V4*)(region + ((m * 4u + f2) * 32u + lane) * 16u) = v;
                }
            } else {
                const bool v0 = (int)(grp * 16u + g) < N, v1 = (int)(grp * 16u + g + 8u) < N;
#pragma unroll
                for (u32 fq = 0; fq < 2; ++fq) {
                    const u32 f = 4u * fq;
                    const u32 w0 = f32x4_to_e4m3x4(val[f][0], val[f][1], val[f + 1u][0], val[f + 1u][1]);
                    const u32 w1 = f32x4_to_e4m3x4(val[f][2], val[f][3], val[f + 1u][2], val[f + 1u][3]);
                    const u32 w2 = f32x4_to_e4m3x4(val[f + 2u][0], val[f + 2u][1], val[f + 3u][0], val[f + 3u][1]);
                    const u32 w3 = f32x4_to_e4m3x4(val[f + 2u][2], val[f + 2u][3], val[f + 3u][2], val[f + 3u][3]);
                    const V4 ov = {v0 ? w0 : 0u, v1 ? w1 : 0u, v0 ? w2 : 0u, v1 ? w3 : 0u};
                    *(V4*)(p.out + (u64)grp * 16384u + ((ob >> 5) + fq) * 512u + lane * 16u) = ov;
                }
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (leader) {
        flag_publish(p.counter + tile * 8u + cg, last ? 3 : 0);
        if (last && p.oflags) flag_publish(p.oflags + tile * 8u + cg, 0);
    }
#else
#if PROJ_WSM
    // this CTA's 16 weight slabs (outputs 32 (4 cg + j) .., steps 4 z + ks) to shared memory first: they do not depend
    // on the producer, so the copy overlaps the flag waits and the packing. Local slab (j * 4 + ks), j = 2 (w & 1) + o2
    __shared__ __align__(16) u8 sm_w[16u * SLAB];
#pragma unroll
    for (u32 r = 0; r < 9u; ++r) {
        const u32 c = threadIdx.y * 32u + lane + 128u * r;      // 16-byte chunk 0 .. 1151
        const u32 slab = c / (SLAB / 16u), off = (c - slab * (SLAB / 16u)) * 16u;
        const u32 j = slab >> 2, ks = slab & 3u;
        cp_async16(sm_w + slab * SLAB + off, p.w + 2304u + (u64)((cg * 4u + j) * 16u + 4u * z + ks) * SLAB + off);
    }
    cp_commit();
#endif

#if PROJ_PRE
    // all eight producer flags of this slice first, then every input load of the warp in flight at once (16 x 16 B),
    // then the packing (round 1 had two loads in flight between barriers)
    if (p.pflags) {
        if (leader) {
            int budget = WAIT_BUDGET_NS;
            for (u32 i = 0; i < 8u; ++i) wait_at_least(p.pflags + tile * 32u + 8u * z + i, 0, budget);
            __threadfence();
        }
        __syncthreads();
    }
    if (effective_options(p.options) & OPT_HA) {                // the attention output arrives packed: copy to the units
        u32 nin[2][4][2][3];
#pragma unroll
        for (u32 q = 0; q < 2; ++q) {
            const u32 grp = tile * 8u + 2u * w + q;
#pragma unroll
            for (u32 ks = 0; ks < 4; ++ks)
#pragma unroll
                for (u32 h = 0; h < 2; ++h) {
                    nin[q][ks][h][0] = nin[q][ks][h][1] = nin[q][ks][h][2] = 0u;
                    if ((int)grp < G)
                        get_packed_cg(p.x + (u64)grp * 16384u + (8u * z + 2u * ks + h) * 512u, lane,
                                      nin[q][ks][h][0], nin[q][ks][h][1], nin[q][ks][h][2]);
                }
        }
#pragma unroll
        for (u32 q = 0; q < 2; ++q)
#pragma unroll
            for (u32 ks = 0; ks < 4; ++ks) {
                const u32 unit = (2u * w + q) * 4u + ks;
                const V4 a = {nin[q][ks][0][0], nin[q][ks][0][1], nin[q][ks][1][0], nin[q][ks][1][1]};
                *(V4*)(sm_a + unit * 512u + lane * 16u) = a;
                put_scale(sm_s + unit * 64u, g, t, 0u, nin[q][ks][0][2]);
                put_scale(sm_s + unit * 64u, g, t, 1u, nin[q][ks][1][2]);
            }
    } else {
        V4 xin[2][4][2];
#pragma unroll
        for (u32 q = 0; q < 2; ++q) {
            const u32 grp = tile * 8u + 2u * w + q;
#pragma unroll
            for (u32 ks = 0; ks < 4; ++ks)
#pragma unroll
                for (u32 h = 0; h < 2; ++h) {
                    xin[q][ks][h] = {0u, 0u, 0u, 0u};
                    if ((int)grp < G) xin[q][ks][h] = load_cg(p.x + (u64)grp * 16384u + (8u * z + 2u * ks + h) * 512u + lane * 16u);
                }
        }
#pragma unroll
        for (u32 q = 0; q < 2; ++q)
#pragma unroll
            for (u32 ks = 0; ks < 4; ++ks) {
                const u32 gi = 2u * w + q;
                pack_unit(xin[q][ks][0], xin[q][ks][1], sm_a + (gi * 4u + ks) * 512u, sm_s + (gi * 4u + ks) * 64u, ta, lane, g, t);
            }
    }
#else
    for (u32 ks = 0; ks < 4; ++ks) {
        if (p.pflags) {
            if (leader) {
                int budget = WAIT_BUDGET_NS;
                wait_at_least(p.pflags + tile * 32u + 8u * z + 2u * ks, 0, budget);
                wait_at_least(p.pflags + tile * 32u + 8u * z + 2u * ks + 1u, 0, budget);
                __threadfence();
            }
            __syncthreads();
        }
#pragma unroll
        for (u32 gi = 2u * w; gi < 2u * w + 2u; ++gi) {
            const u32 grp = tile * 8u + gi, kb = 8u * z + 2u * ks;
            V4 xa = {0u, 0u, 0u, 0u}, xb = {0u, 0u, 0u, 0u};
            if ((int)grp < G) {
                xa = load_cg(p.x + (u64)grp * 16384u + kb * 512u + lane * 16u);
                xb = load_cg(p.x + (u64)grp * 16384u + (kb + 1u) * 512u + lane * 16u);
            }
#if PDBG_STAGE == 1                                             // timing: the input stored raw, not packed
            *(V4*)(sm_a + (gi * 4u + ks) * 512u + lane * 16u) = {xa.x ^ xb.x, xa.y ^ xb.y, xa.z ^ xb.z, xa.w ^ xb.w};
#else
            pack_unit(xa, xb, sm_a + (gi * 4u + ks) * 512u, sm_s + (gi * 4u + ks) * 64u, ta, lane, g, t);
#endif
        }
    }
#endif
#if PROJ_WSM
    cp_wait0();
#endif
    __syncthreads();

#if PROJ_WSM
    const u8* wslab = sm_w + (2u * (w & 1u)) * 4u * SLAB - 4u * z * SLAB;   // + (o2 * 16 + 4 z + ks) SLAB below:
#define PROJ_SLAB_AT(o2, ks) (wslab + ((o2) * 4u + 4u * z + (ks)) * SLAB)   //   local (2 (w & 1) + o2) * 4 + ks
#else
    const u8* wslab = p.w + 2304u + (u64)(ob >> 5) * 16u * SLAB;
#define PROJ_SLAB_AT(o2, ks) (wslab + ((o2) * 16u + 4u * z + (ks)) * SLAB)
#endif
    const u8* gain = p.w + 128u;
    bool merged = z == 0u;
#pragma unroll 1
    for (u32 pass = 0; pass < 2; ++pass) {
        float acc[2][8][4];
#pragma unroll
        for (u32 m = 0; m < 2; ++m)
#pragma unroll
            for (u32 f = 0; f < 8; ++f) acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
#pragma unroll
        for (u32 ks = 0; ks < 4; ++ks) {
            V4 a[2];
            u32 sa[2];
#pragma unroll
            for (u32 m = 0; m < 2; ++m) {
                const u32 unit = (4u * th + 2u * pass + m) * 4u + ks;
                a[m] = *(const V4*)(sm_a + unit * 512u + lane * 16u);
                sa[m] = *(const u32*)(sm_s + unit * 64u + g * 8u + (t & 1u) * 4u);
            }
#pragma unroll
            for (u32 o2 = 0; o2 < 2; ++o2) {
                const u8* sl = PROJ_SLAB_AT(o2, ks);
#if PDBG_STAGE == 4                                             // timing: weights not loaded (one slab for all)
                const u8* sl0 = wslab;
#else
                const u8* sl0 = sl;
#endif
                const V4 n01 = *(const V4*)(sl0 + lane * 16u), n23 = *(const V4*)(sl0 + 512u + lane * 16u);
                const V4 sw = *(const V4*)(sl0 + 1024u + g * 16u);
#pragma unroll
                for (u32 m = 0; m < 2; ++m) {
                    if (PDBG_STAGE == 2) {                      // timing: no mma, operands folded into the accumulators
                        acc[m][o2 * 4u][0] += __uint_as_float((a[m].x ^ n01.x ^ n23.y ^ sw.x) & 0x3FFFFFFFu);
                        acc[m][o2 * 4u + 1u][1] += __uint_as_float((a[m].y ^ n01.y ^ n23.z ^ sw.y ^ sa[m]) & 0x3FFFFFFFu);
                        acc[m][o2 * 4u + 2u][2] += __uint_as_float((a[m].z ^ n01.z ^ n23.w ^ sw.z) & 0x3FFFFFFFu);
                        acc[m][o2 * 4u + 3u][3] += __uint_as_float((a[m].w ^ n01.w ^ n23.x ^ sw.w) & 0x3FFFFFFFu);
                        continue;
                    }
                    mma4(acc[m][o2 * 4u + 0u], a[m], n01.x, n01.y, sa[m], sw.x);
                    mma4(acc[m][o2 * 4u + 1u], a[m], n01.z, n01.w, sa[m], sw.y);
                    mma4(acc[m][o2 * 4u + 2u], a[m], n23.x, n23.y, sa[m], sw.z);
                    mma4(acc[m][o2 * 4u + 3u], a[m], n23.z, n23.w, sa[m], sw.w);
                }
            }
        }
        if (!merged && PDBG_STAGE != 3) {                       // z > 0: the CTA of slice z - 1 has merged its part
            if (leader) {                                       // (timing stage 3: no wait, wrong sums)
                int budget = WAIT_BUDGET_NS;
                wait_at_least(p.counter + tile * 8u + cg, (int)z - 1, budget);
                __threadfence();
            }
            __syncthreads();
            merged = true;
        }
        // scratch: 4096 B per (32-token pair, 64 outputs): G / 2 x 16 x 4096 = G x 32768 bytes, the runtime's size
        const u32 grp0 = tile * 8u + 4u * th + 2u * pass;
        u8* region = p.scratch + (u64)((grp0 >> 1) * 16u + (ob >> 6)) * 4096u;
#pragma unroll
        for (u32 m = 0; m < 2; ++m) {
            const u32 grp = grp0 + m;
            if ((int)grp >= G) continue;
            // the previous value: z = 0 the gain-scaled skip (exact in float32), else the scratch
            float prev[8][4];
            if (z == 0u) {
#pragma unroll
                for (u32 fq = 0; fq < 2; ++fq) {
                    const V4 sk = load_cg(p.skip + (u64)grp * 16384u + ((ob >> 5) + fq) * 512u + lane * 16u);
                    u32 h0[4], h1[4];
                    act_rows(sk, h0, h1);                       // row g: channels f 4 fq + 0..3 pairs; row g + 8
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) {
                        const u32 f = 4u * fq + j;
                        float g0, g1, a0, a1, b0, b1;
                        h2_unpack(load_cg32(gain + (ob + 8u * f + 2u * t) * 2u), g0, g1);
                        h2_unpack(h0[j], a0, a1);
                        h2_unpack(h1[j], b0, b1);
                        prev[f][0] = g0 * a0; prev[f][1] = g1 * a1; prev[f][2] = g0 * b0; prev[f][3] = g1 * b1;
                    }
                }
            } else {
#pragma unroll
                for (u32 f2 = 0; f2 < 4; ++f2) {
                    const V4 v = load_cg(region + ((m * 4u + f2) * 32u + lane) * 16u);
                    h2_unpack(v.x, prev[2u * f2][0], prev[2u * f2][1]); h2_unpack(v.y, prev[2u * f2][2], prev[2u * f2][3]);
                    h2_unpack(v.z, prev[2u * f2 + 1u][0], prev[2u * f2 + 1u][1]); h2_unpack(v.w, prev[2u * f2 + 1u][2], prev[2u * f2 + 1u][3]);
                }
            }
            float val[8][4];
#pragma unroll
            for (u32 f = 0; f < 8; ++f)
#pragma unroll
                for (u32 e = 0; e < 4; ++e) val[f][e] = fmaf(acc[m][f][e], s, prev[f][e]);
            if (z < 3u) {
#pragma unroll
                for (u32 f2 = 0; f2 < 4; ++f2) {
                    const V4 v = {h2_pack(val[2u * f2][0], val[2u * f2][1]), h2_pack(val[2u * f2][2], val[2u * f2][3]),
                                  h2_pack(val[2u * f2 + 1u][0], val[2u * f2 + 1u][1]), h2_pack(val[2u * f2 + 1u][2], val[2u * f2 + 1u][3])};
                    *(V4*)(region + ((m * 4u + f2) * 32u + lane) * 16u) = v;
                }
            } else {
                const bool v0 = (int)(grp * 16u + g) < N, v1 = (int)(grp * 16u + g + 8u) < N;
#pragma unroll
                for (u32 fq = 0; fq < 2; ++fq) {
                    const u32 f = 4u * fq;
                    const u32 w0 = f32x4_to_e4m3x4(val[f][0], val[f][1], val[f + 1u][0], val[f + 1u][1]);
                    const u32 w1 = f32x4_to_e4m3x4(val[f][2], val[f][3], val[f + 1u][2], val[f + 1u][3]);
                    const u32 w2 = f32x4_to_e4m3x4(val[f + 2u][0], val[f + 2u][1], val[f + 3u][0], val[f + 3u][1]);
                    const u32 w3 = f32x4_to_e4m3x4(val[f + 2u][2], val[f + 2u][3], val[f + 3u][2], val[f + 3u][3]);
                    const V4 ov = {v0 ? w0 : 0u, v1 ? w1 : 0u, v0 ? w2 : 0u, v1 ? w3 : 0u};
                    *(V4*)(p.out + (u64)grp * 16384u + ((ob >> 5) + fq) * 512u + lane * 16u) = ov;
                }
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (leader) {
        flag_publish(p.counter + tile * 8u + cg, (int)z);
        if (z == 3u && p.oflags) flag_publish(p.oflags + tile * 8u + cg, 0);
    }
#endif
}
