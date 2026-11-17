// Explicit diagnostic only. Reads native POST scalar controls, never alters a tensor.
typedef unsigned char u8; typedef unsigned short u16; typedef unsigned long long u64;
struct Probe { u8 native[184]; float* out; };
extern "C" __global__ void nr_post_probe(Probe p) {
    const u64 history=*(const u64*)(p.native+88),motion=*(const u64*)(p.native+96),scale=*(const u64*)(p.native+104);
    float raw=1.f;
    if(scale) {u16 h=*(const u16*)scale;asm("cvt.f32.f16 %0,%1;":"=f"(raw):"h"(h));}
    float bw=isinf(raw)?0.f:fminf(1.f,fmaxf(0.f,raw));
    p.out[0]=bw;p.out[1]=history!=0;p.out[2]=motion!=0;p.out[3]=*(const int*)(p.native+52)!=0;
    p.out[4]=raw;p.out[5]=*(const int*)(p.native+112)!=0;p.out[6]=scale!=0;
    p.out[7]=p.out[1]&&p.out[2]&&p.out[3]&&bw>0;
}


// Sampled POST diagnostic: one pixel per 4x4 cell, 16 float values each.
typedef unsigned int u32;
#define NR_INLINE static __forceinline__ __device__
#define W1_FORM 0
NR_INLINE float rcp_approx(float x){float d;asm("rcp.approx.ftz.f32 %0,%1;":"=f"(d):"f"(x));return d;}
struct F4 { float x, y, z, w; };
#define P_U64(o) (*(const u64*)(P.b + (o)))
#define P_F32(o) (*(const float*)(P.b + (o)))
#define P_S32(o) (*(const int*)(P.b + (o)))

NR_INLINE F4 tex_rgba(u64 tex, float u, float v)
{
    F4 r;
#if TEX_FREE
    asm("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(tex), "f"(u), "f"(v));
#else
    asm volatile("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(tex), "f"(u), "f"(v));
#endif
    return r;
}
NR_INLINE float h16_to_f32(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
NR_INLINE float ex2a(float x) { float d; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(x)); return d; }

NR_INLINE void history5(u64 tex, float qx, float qy, float fxs, float fys, float cm, float rm, float rc, float rr,
                        float su, float ou, float qu, float sv, float ov, float qv, F4& sum, float& r)
{
    const float cx = floorf(fmaf(qx, fxs, -0.5f)) + 0.5f, cy = floorf(fmaf(qy, fys, -0.5f)) + 0.5f;
    const float fx = fminf(fmaxf(fmaf(qx, fxs, -cx), 0.0f), 1.0f), fy = fminf(fmaxf(fmaf(qy, fys, -cy), 0.0f), 1.0f);
    const float x2 = fx * fx, y2 = fy * fy, x3 = fx * x2, y3 = fy * y2;
    const float w0x = fmaf(fx + x3, -0.5f, x2), w0y = fmaf(fy + y3, -0.5f, y2);
#if W1_FORM == 0
    const float w1x = fmaf(x3, 1.5f, -(x2 * 2.5f)) + 1.0f, w1y = fmaf(y3, 1.5f, -(y2 * 2.5f)) + 1.0f;
#else
    const float w1x = fmaf(x2, -2.5f, x3 * 1.5f) + 1.0f, w1y = fmaf(y2, -2.5f, y3 * 1.5f) + 1.0f;
#endif
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

struct Sample {u8 b[184];float* out;u32 phase,pad;};
extern "C" __global__ void nr_post_sample(Sample P) {
    const int width=P_S32(172),height=P_S32(176),cols=(width+3)/4,rows=(height+3)/4;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cols*rows)return;
    const int x=min(4*(i%cols)+2,width-1),y=min(4*(i/cols)+2,height-1);
    u16 a,b,c,d;const u64 surf=P_U64(16);
    asm volatile("suld.b.2d.v4.b16.zero {%0,%1,%2,%3}, [%4, {%5,%6}];"
                 :"=h"(a),"=h"(b),"=h"(c),"=h"(d):"l"(surf),"r"(x*8),"r"(y):"memory");
    float* out=P.out+16*i;
    if(P.phase==0) {
        out[0]=h16_to_f32(a);out[1]=h16_to_f32(b);out[2]=h16_to_f32(c);
        const float xn=__fdividef(float(x)+.5f,float(width)),yn=__fdividef(float(y)+.5f,float(height));
        const F4 f=tex_rgba(P_U64(56),P_F32(80)*fmaf(P_F32(72),xn,P_F32(64)),P_F32(84)*fmaf(P_F32(76),yn,P_F32(68)));
        out[9]=f.x;out[10]=f.y;out[11]=f.z;
        F4 hs={};float hn=1;
        if(P_U64(88)&&P_U64(96)) {
            const F4 mv=tex_rgba(P_U64(96),P_F32(156)*fmaf(P_F32(148),xn,P_F32(140)),P_F32(160)*fmaf(P_F32(152),yn,P_F32(144)));
            const bool have=P_S32(112)!=0;const float qx=have?fmaf(P_F32(164),mv.x,xn):xn,qy=have?fmaf(P_F32(168),mv.y,yn):yn;
            history5(P_U64(88),qx,qy,float(width),float(height),float(width)-.5f,float(height)-.5f,rcp_approx(float(width)),rcp_approx(float(height)),
                P_F32(124),P_F32(116),P_F32(132),P_F32(128),P_F32(120),P_F32(136),hs,hn);
        }
        out[5]=hs.x;out[6]=hs.y;out[7]=hs.z;out[8]=hn;
    } else if(P.phase==1) {
        const float r3=h16_to_f32(a);float bw=1.f;
        if(P_U64(104)){float v=h16_to_f32(*(const u16*)P_U64(104));bw=isinf(v)?0.f:fminf(1.f,fmaxf(0.f,v));}
        out[3]=r3;out[4]=(P_U64(88)&&P_U64(96)) ? fminf(1.f,fmaxf(0.f,rcp_approx(ex2a(r3*-1.4426950216293335f)+1.f)*bw)) : 0.f;
    } else {out[12]=h16_to_f32(a);out[13]=h16_to_f32(b);out[14]=h16_to_f32(c);out[15]=1.f;}
}
