// The own NR pass host's helper launches (nvfp4/NrHost.cpp), Ampere-legal, our own code:
//   amp_clear  launch 0      sets the flag region to -1 (int32 words)
//   amp_copy   launch -1     the colour input into the input-copy texture (Natural / Cinematic)
//              launch 155    the network output (or, for Standard, the output) into the history texture
//   amp_grade  launch 156    the colour step of Natural / Cinematic: the network output graded, then blended with the
//                            input copy by the intensity, into the output
// Parameter blocks are the plan's (assets/host-plan/modelA_fp8.plan): the same byte layout the runtime used for its
// helpers, so the host fills them unchanged. Texture handles are SRV | sampler << 32 (--set-texmode-raw), surfaces UAV
// indices; coordinates are normalised (pixel centre / size) and every value goes through a formatted texture load and
// a formatted surface store (RGBA16F).
//
// amp_grade: the grade is fourteen numbers k0..k13 (+316..+368) the host computes per style (NrHost.cpp, style
// "gradeN"): levels (k0 black, k1 white), a warm/cool tint (k7) and a green/magenta tint (k8) at full saturation and the
// pixel's own lightness, exposure (2^k2), an S-curve contrast (k4), a five-band tone curve (weights k9..k13 at 0, 1/4,
// 1/2, 3/4, 1: x^(2^-g)), gamma (x^(2^-k3)), saturation (s * (1 + k5)) and a saturation curve (s^(2^-k6)) in HSL. The
// optional mask / transfer inputs (+96..+287, +300..+308) are always null in the plan: not implemented here.
// The arithmetic order follows the runtime's results to the bit where it can (approximate division, ex2, lg2, FTZ).
// amp-variant: base --ftz=true --ptxas-options=--set-texmode-raw=true
#include "amp_common.cuh"

#define HI __device__ __forceinline__

struct CopyParams {                 // 72 bytes
    unsigned long long tex, surf;   // +0 source texture, +8 target surface
    float offX, offY;               // +16
    float scaleX, scaleY;           // +24
    float rcpX, rcpY;               // +32
    float dstX, dstY;               // +40
    float unused[4];                // +48 (the target's size; not read)
    int w, h;                       // +64
};
struct GradeParams {                // 376 bytes
    unsigned long long surf;        // +0 output surface
    float dstX, dstY;               // +8
    float outSize[4];               // +16 (not read)
    unsigned long long net;         // +32 network output texture
    float netOff[2], netScale[2], netRcp[2];   // +40, +48, +56
    unsigned long long input;       // +64 input-copy texture
    float inOff[2], inScale[2], inRcp[2];      // +72, +80, +88
    unsigned char optional[192];    // +96 .. +287 masks (null)
    float intensity;                // +288
    int w, h;                       // +292
    int transfer[3];                // +300 (0)
    int pad;                        // +312
    float k[14];                    // +316 .. +368
    int tail;                       // +372
};
static_assert(sizeof(CopyParams) == 72, "copy block");
static_assert(sizeof(GradeParams) == 376, "grade block");

HI float divx(float a, float b) { float r; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
HI float ex2x(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
HI float lg2x(float a) { float r; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
HI float sat(float a) { return __saturatef(a); }
HI float4 fetch(unsigned long long tex, float u, float v) {
    float4 r;
    asm volatile("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(tex), "f"(u), "f"(v));
    return r;
}
HI void store(unsigned long long surf, int x, int y, float a, float b, float c, float d) {
    asm volatile("sust.p.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};" :: "l"(surf), "r"(x), "r"(y), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

extern "C" __global__ void amp_clear(int* base, int count) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) base[i] = -1;
}

extern "C" __global__ void amp_copy(const CopyParams p) {
    const int x = int(blockIdx.x * blockDim.x + threadIdx.x), y = int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.w || y >= p.h) return;
    const float u = divx(float(x) + 0.5f, float(p.w)), v = divx(float(y) + 0.5f, float(p.h));
    const float4 c = fetch(p.tex, fmaf(p.scaleX, u, p.offX) * p.rcpX, fmaf(p.scaleY, v, p.offY) * p.rcpY);
    store(p.surf, int(p.dstX) + x, int(p.dstY) + y, c.x, c.y, c.z, c.w);
}

// Lightness and a full-saturation colour of one fixed hue at that lightness: (q, p) of the HSL conversion.
HI void full(float l, float& q, float& p) {
    const float twice = l + l;
    q = l < 0.5f ? twice : (l + 1.f) - l;
    p = twice - q;
}
HI float lightness(float r, float g, float b) { return (fmaxf(fmaxf(r, g), b) + fminf(fminf(r, g), b)) * 0.5f; }
HI float smooth(float t) { return (t * t) * (3.f - (t + t)); }
// The five-band tone curve of one channel.
HI float tone(float x, const float* k) {
    const float m25 = x + -0.25f, m50 = x + -0.5f, m75 = x + -0.75f, m100 = x + -1.f;
    const float w0 = smooth(sat(divx(m25, -0.25f)));
    const float w1 = smooth(sat(divx(x, 0.25f))) * smooth(sat(divx(m50, -0.25f)));
    const float w2 = smooth(sat(divx(m25, 0.25f))) * smooth(sat(divx(m75, -0.25f)));
    const float w3 = smooth(sat(divx(m50, 0.25f))) * smooth(sat(divx(m100, -0.25f)));
    const float w4 = smooth(sat(divx(m75, 0.25f)));
    const float g = fmaf(w4, k[13], fmaf(w3, k[12], fmaf(w2, k[11], fmaf(w0, k[9], w1 * k[10]))));
    return ex2x(lg2x(x) * ex2x(-g));
}
HI void toHsl(float r, float g, float b, float& h, float& s, float& l) {
    const float mx = fmaxf(fmaxf(r, g), b), mn = fminf(fminf(r, g), b), sum = mx + mn;
    l = sum * 0.5f;
    h = 0.f; s = 0.f;
    if (!(mx > mn)) return;
    const float d = mx - mn;
    s = l > 0.5f ? divx(d, (2.f - mx) - mn) : divx(d, sum);
    if (mx == r) h = divx((g < b ? 6.f : 0.f) + divx(g - b, d), 6.f);
    else if (mx == g) h = divx(divx(b - r, d) + 2.f, 6.f);
    else h = divx(divx(r - g, d) + 4.f, 6.f);
}
HI float hue(float t, float q, float p) {
    t = t < 0.f ? t + 1.f : t;
    t = t > 1.f ? t + -1.f : t;
    if (t < 0.166666672f) return fmaf(t, (q - p) * 6.f, p);
    if (t < 0.5f) return q;
    if (t < 0.666666687f) return fmaf((0.666666687f - t) * (q - p), 6.f, p);
    return p;
}
HI void fromHsl(float h, float s, float l, float& r, float& g, float& b) {
    if (s <= 0.f) { r = g = b = l; return; }
    const float q = l < 0.5f ? l * (s + 1.f) : (l + s) - l * s;
    const float p = (l + l) - q;
    r = hue(h + 0.333333343f, q, p);
    g = hue(h, q, p);
    b = hue(h + -0.333333343f, q, p);
}

extern "C" __global__ void amp_grade(const GradeParams p) {
    const int x = int(blockIdx.x * blockDim.x + threadIdx.x), y = int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= p.w || y >= p.h) return;
    const float u = divx(float(x) + 0.5f, float(p.w)), v = divx(float(y) + 0.5f, float(p.h));
    const float4 net = fetch(p.net, fmaf(u, p.netScale[0], p.netOff[0]) * p.netRcp[0], fmaf(v, p.netScale[1], p.netOff[1]) * p.netRcp[1]);
    const float4 in = fetch(p.input, fmaf(u, p.inScale[0], p.inOff[0]) * p.inRcp[0], fmaf(v, p.inScale[1], p.inOff[1]) * p.inRcp[1]);
    const float* k = p.k;
    // Levels.
    const float range = (k[1] - k[0]) + 1e-10f;
    float r = sat(divx(sat(net.x) - k[0], range)), g = sat(divx(sat(net.y) - k[0], range)), b = sat(divx(sat(net.z) - k[0], range));
    // Warm / cool tint (k7).
    if (!(fabsf(k[7]) < 1e-6f)) {
        const float a = fabsf(k[7]), l = lightness(r, g, b);
        float q, pp, tr, tg, tb;
        full(l, q, pp);
        if (k[7] > 0.f) { tr = q; tb = pp; tg = fmaf((q - pp) * 6.f, 0.0611100011f, pp); }
        else { tb = q; tr = pp; tg = fmaf((q - pp) * 0.105556667f, 6.f, pp); }
        r = sat(fmaf(a, tr - r, r)); g = sat(fmaf(a, tg - g, g)); b = sat(fmaf(a, tb - b, b));
    }
    // Green / magenta tint (k8).
    if (!(fabsf(k[8]) < 1e-6f)) {
        const float a = fabsf(k[8]), l = lightness(r, g, b);
        float q, pp, tr, tg, tb;
        full(l, q, pp);
        if (k[8] > 0.f) { tg = q; tb = pp; tr = fmaf((q - pp) * 0.0222233534f, 6.f, pp); }
        else { tb = q; tg = pp; tr = fmaf((q - pp) * 6.f, 0.144443393f, pp); }
        r = sat(fmaf(a, tr - r, r)); g = sat(fmaf(a, tg - g, g)); b = sat(fmaf(a, tb - b, b));
    }
    // Exposure, contrast, tone curve.
    const float e = ex2x(k[2]);
    float c[3] = {r * e, g * e, b * e};
    for (int i = 0; i < 3; ++i) {
        const float s = sat(c[i]);
        c[i] = tone(sat(fmaf(k[4], smooth(s) - s, s)), k);
    }
    // Gamma.
    const float gamma = ex2x(-k[3]);
    for (int i = 0; i < 3; ++i) c[i] = ex2x(gamma * lg2x(fmaxf(0.f, c[i])));
    // Saturation, then the saturation curve.
    float h, s, l;
    toHsl(c[0], c[1], c[2], h, s, l);
    fromHsl(h, sat(s * (k[5] + 1.f)), l, c[0], c[1], c[2]);
    toHsl(c[0], c[1], c[2], h, s, l);
    fromHsl(h, sat(ex2x(ex2x(-k[6]) * lg2x(s))), l, c[0], c[1], c[2]);
    // Blend with the input by the intensity.
    const float a = sat(p.intensity);
    store(p.surf, int(p.dstX) + x, int(p.dstY) + y, sat(fmaf(a, sat(c[0]) - in.x, in.x)), sat(fmaf(a, sat(c[1]) - in.y, in.y)),
          sat(fmaf(a, sat(c[2]) - in.z, in.z)), 1.f);
}
