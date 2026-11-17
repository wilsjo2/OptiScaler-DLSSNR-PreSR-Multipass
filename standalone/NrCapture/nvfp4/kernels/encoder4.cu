// Paired-pixel residual MLP, Astra round 1. Compile ONLY through lab.compile_kernel.
// sm_120a, --fmad=false, -DC=32|64 -DHIDDEN=64|128|256 -DVIEW_OUT=0|1.
// Two warps / original 8x8 window. Same external E4M3 layouts and window flags as Swin.
// This is a DIFFERENT learned block; original Swin records cannot be used.
// Equation: Q8(Q4(ReLU_clip(Q4(pair(x))*W1 + b1))*W2 + b2 + gain*pair(x)).
// Q4 uses power-of-two UE4M3 activation scales per 16 K values, no approximate reciprocal.
// Pairing (options): bit 0 axis (0 x, 1 y), bits 1..2 coordinate bit (0..2).
// Adjacent layers may choose different pairing distances/axes. Both pixels stay in the window.
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
struct __align__(16) V4 { u32 x, y, z, w; };
#define FI static __forceinline__ __device__
#ifndef C
#define C 32
#endif
#ifndef HIDDEN
#define HIDDEN 128
#endif
#ifndef VIEW_OUT
#define VIEW_OUT 0
#endif
#ifndef STAGED
#define STAGED 1
#endif
#ifndef SHARED_REUSE
#define SHARED_REUSE 0
#endif
#if (C != 32 && C != 64 && C != 128 && C != 256) || (HIDDEN % 64 != 0)
#error Unsupported dimensions
#endif
#define D (2*C)
#define IK (D/64)
#define HK (HIDDEN/64)
#define SLAB 1152

// Same normalised 104-byte Params as nr_swin4; unused fields must be zero/valid dims.
struct Params {
    const u8* in; u8* out; const u8* w; const u8* skip;
    const int* pflags; int* oflags; u8* ds_out;
    int H,W,sx,sy,psx,psy,d0,d1,e0,e1;
    u32 options,reserved;
};
// record: float[16] (0 hidden clip), uint32[16] offsets at 64:
// 0 W1, 1 W2, 2 b1[HIDDEN], 3 b2[D], 4 gain[D]. Matrices use swin4's 1152-byte slabs.

FI u32 cg32(const u8* p) {
    u32 v; asm volatile("ld.weak.global.cg.u32 %0, [%1];" : "=r"(v) : "l"((u64)p) : "memory"); return v;
}
FI int flag(const int* p) {
    int v; asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"((u64)p) : "memory"); return v;
}
FI void publish(int* p) { asm volatile("st.release.gpu.global.s32 [%0], 0;" :: "l"((u64)p) : "memory"); }
FI void mma(float* c, V4 a, u32 b0,u32 b1,u32 sa,u32 sb) {
    float d0,d1,d2,d3; const u16 z=0;
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%16,%16}, {%15}, {%16,%16};"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b0),"r"(b1),
          "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),"r"(sa),"r"(sb),"h"(z));
    c[0]=d0;c[1]=d1;c[2]=d2;c[3]=d3;
}
FI void decode(u32 x,float* v) {
    asm("{ .reg .b16 l,h,a,b; .reg .b32 p,q; mov.b32 {l,h},%4;"
        "cvt.rn.f16x2.e4m3x2 p,l; cvt.rn.f16x2.e4m3x2 q,h;"
        "mov.b32 {a,b},p; cvt.f32.f16 %0,a; cvt.f32.f16 %1,b;"
        "mov.b32 {a,b},q; cvt.f32.f16 %2,a; cvt.f32.f16 %3,b; }"
        : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "r"(x));
}
FI u32 store8(float a,float b,float c,float d) {
    u32 v;
    asm("{ .reg .b16 l,h; .reg .b32 p,q; cvt.rn.f16x2.f32 p,%2,%1; cvt.rn.f16x2.f32 q,%4,%3;"
        "cvt.rn.satfinite.e4m3x2.f16x2 l,p; cvt.rn.satfinite.e4m3x2.f16x2 h,q; mov.b32 %0,{l,h}; }"
        : "=r"(v) : "f"(a),"f"(b),"f"(c),"f"(d)); return v;
}
FI u32 nib8(const float* v) {
    u32 word;
    asm("{ .reg .b8 a,b,c,d; cvt.rn.satfinite.e2m1x2.f32 a,%2,%1; cvt.rn.satfinite.e2m1x2.f32 b,%4,%3;"
        "cvt.rn.satfinite.e2m1x2.f32 c,%6,%5; cvt.rn.satfinite.e2m1x2.f32 d,%8,%7; mov.b32 %0,{a,b,c,d}; }"
        : "=r"(word) : "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),"f"(v[4]),"f"(v[5]),"f"(v[6]),"f"(v[7]));
    return word;
}
FI u32 quant(float* v,u32& code) {
    float m=0;
#pragma unroll
    for(int j=0;j<8;j++) m=fmaxf(m,fabsf(v[j]));
    m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,1));
    const u32 bits=__float_as_uint(fmaxf(m*(1.0f/6.0f),0.001953125f));
    int e=(bits>>23)+((bits&0x7fffffu)!=0);
    e=min(max(e,118),135);
    const float r=__uint_as_float((254-e)<<23);
    code=e<121 ? (1u<<(e-118)) : (u32)((e-120)<<3);
#pragma unroll
    for(int j=0;j<8;j++) v[j]*=r;
    return nib8(v);
}
FI u32 scales(u32 lo0,u32 lo1,u32 hi0,u32 hi1,u32 t) {
    u32 s=(t&1u) ? (lo1|(hi1<<16)) : (lo0|(hi0<<16));
    u32 b=__shfl_xor_sync(0xffffffffu,s,2);
    return (t&2u) ? (b|(s<<8)) : (s|(b<<8));
}
FI u64 at(int y,int x,int h,u32 t,int W) {
    const int m=4*(y&3)+(x&3);
    return ((u64)(y>>2)*(W/4)+(x>>2))*16*C+h*512+(4*(m&7)+t)*16+((m>>3)&1)*4;
}
FI void coords(int pair,int which,int mode,int x0,int y0,int& x,int& y) {
    const int axis=mode&1, bit=(mode>>1)&3;
    const int a=pair&3, b=pair>>2, mask=1<<bit;
    const int v=(a&(mask-1))|((a&~(mask-1))<<1)|(which<<bit);
    x=x0+(axis ? b : v); y=y0+(axis ? v : b);
}
FI void read8(const Params& p,const u8* window,int pair,int h,u32 t,int x0,int y0,float* v) {
#if STAGED
    int x,y;coords(pair,h/(C/32),p.options,0,0,x,y);
    u64 o=at(y,x,h%(C/32),t,8);
    decode(*(const u32*)(window+o),v);decode(*(const u32*)(window+o+8),v+4);
#else
    int x,y; coords(pair,h/(C/32),p.options,x0,y0,x,y);
    u32 a=0,b=0;
    if(x>=0 && x<p.W && y>=0 && y<p.H) {
        u64 o=at(y,x,h%(C/32),t,p.W); a=cg32(p.in+o);b=cg32(p.in+o+8);
    }
    decode(a,v);decode(b,v+4);
#endif
}
FI void put8(const Params& p,u8* window,int pair,int h,u32 t,int x0,int y0,const float* v) {
#if STAGED
    int x,y;coords(pair,h/(C/32),p.options,0,0,x,y);
    u64 o=at(y,x,h%(C/32),t,8);
    *(u32*)(window+o)=store8(v[0],v[1],v[2],v[3]);
    *(u32*)(window+o+8)=store8(v[4],v[5],v[6],v[7]);
#else
    int x,y;coords(pair,h/(C/32),p.options,x0,y0,x,y);
    if(x<0 || x>=p.W || y<0 || y>=p.H) return;
    const u32 a=store8(v[0],v[1],v[2],v[3]),b=store8(v[4],v[5],v[6],v[7]);
#if VIEW_OUT
    if(x>=p.d1 || y>=p.d0) return;
    u64 o=((u64)(2*(h%(C/32)))*p.d0*p.d1+(u64)y*p.d1+x)*16+t*4;
    *(u32*)(p.out+o)=a;*(u32*)(p.out+o+(u64)p.d0*p.d1*16)=b;
#else
    u64 o=at(y,x,h%(C/32),t,p.W);*(u32*)(p.out+o)=a;*(u32*)(p.out+o+8)=b;
#endif
#endif
}

// Level4-v1: coarse view(2C) + skip tiles(C) -> output view(C).
// One CTA per unshifted 8x8 output window; all body stages stay in shared memory.
// Params: H,W fine shape; d0,d1 coarse shape. sx=sy=0. pflags must be null.
// Optional oflags publishes one entry per fine window. Host orders both inputs.
#ifndef STAGES
#define STAGES 4
#endif
#ifndef COPY_ONLY
#define COPY_ONLY 0
#endif
FI void body_stage(const Params& p,u8* input) {
    const u32 lane=threadIdx.x,t=lane&3,g=lane>>2,warp=threadIdx.y;
    const int x0=0,y0=0;
    u8* output=input;
    const u32* O=(const u32*)(p.w+64);
    const float clip=*(const float*)p.w;
    const float* b1=(const float*)(p.w+O[2]);
    const float* b2=(const float*)(p.w+O[3]);
    const float* gain=(const float*)(p.w+O[4]);
    V4 a[IK],h[HK];u32 sa[IK],sh[HK];
#pragma unroll
    for(int k=0;k<IK;k++) {
        float v[8];u32 c00,c01,c10,c11;
        read8(p,input,16*warp+g,2*k,t,x0,y0,v);a[k].x=quant(v,c00);
        read8(p,input,16*warp+g+8,2*k,t,x0,y0,v);a[k].y=quant(v,c01);
        read8(p,input,16*warp+g,2*k+1,t,x0,y0,v);a[k].z=quant(v,c10);
        read8(p,input,16*warp+g+8,2*k+1,t,x0,y0,v);a[k].w=quant(v,c11);
        sa[k]=scales(c00,c01,c10,c11,t);
    }
    u32 hc[HIDDEN/32][2];
#pragma unroll
    for(int ob=0;ob<HIDDEN/32;ob++) {
        float ac[4][4]={};
#pragma unroll
        for(int k=0;k<IK;k++) {
            const u8* w=p.w+O[0]+(ob*IK+k)*SLAB;
            const V4 v0=*(const V4*)(w+lane*16),v1=*(const V4*)(w+512+lane*16);
            const V4 s=*(const V4*)(w+1024+g*16);
            mma(ac[0],a[k],v0.x,v0.y,sa[k],s.x);mma(ac[1],a[k],v0.z,v0.w,sa[k],s.y);
            mma(ac[2],a[k],v1.x,v1.y,sa[k],s.z);mma(ac[3],a[k],v1.z,v1.w,sa[k],s.w);
        }
#pragma unroll
        for(int rh=0;rh<2;rh++) {
            float v[8];
#pragma unroll
            for(int j=0;j<4;j++) {
                v[2*j]=fminf(clip,fmaxf(0.0f,ac[j][2*rh]+b1[32*ob+8*j+2*t]));
                v[2*j+1]=fminf(clip,fmaxf(0.0f,ac[j][2*rh+1]+b1[32*ob+8*j+2*t+1]));
            }
            u32 q=quant(v,hc[ob][rh]);
            if((ob&1)==0) { if(rh==0) h[ob/2].x=q;else h[ob/2].y=q; }
            else { if(rh==0) h[ob/2].z=q;else h[ob/2].w=q; }
        }
    }
#pragma unroll
    for(int k=0;k<HK;k++) sh[k]=scales(hc[2*k][0],hc[2*k][1],hc[2*k+1][0],hc[2*k+1][1],t);
#pragma unroll
    for(int ob=0;ob<D/32;ob++) {
        float ac[4][4]={};
#pragma unroll
        for(int k=0;k<HK;k++) {
            const u8* w=p.w+O[1]+(ob*HK+k)*SLAB;
            const V4 v0=*(const V4*)(w+lane*16),v1=*(const V4*)(w+512+lane*16);
            const V4 s=*(const V4*)(w+1024+g*16);
            mma(ac[0],h[k],v0.x,v0.y,sh[k],s.x);mma(ac[1],h[k],v0.z,v0.w,sh[k],s.y);
            mma(ac[2],h[k],v1.x,v1.y,sh[k],s.z);mma(ac[3],h[k],v1.z,v1.w,sh[k],s.w);
        }
#pragma unroll
        for(int rh=0;rh<2;rh++) {
            float v[8];read8(p,input,16*warp+g+8*rh,ob,t,x0,y0,v);
#pragma unroll
            for(int j=0;j<4;j++) {
                const int c=32*ob+8*j+2*t;
                v[2*j]=(ac[j][2*rh]+b2[c])+gain[c]*v[2*j];
                v[2*j+1]=(ac[j][2*rh+1]+b2[c+1])+gain[c+1]*v[2*j+1];
            }
            put8(p,output,16*warp+g+8*rh,ob,t,x0,y0,v);
        }
    }

}

// Encoder4-v1. in: C-channel view; out: C-channel tiled skip; ds_out: 2C view.
// H,W fine size; d0,d1 padded pooled destination size. No flags/waits at boundaries.
FI void pool_read(const u8* window,int row,int head,u32 t,float* v) {
    if(head>=C/32) {
#pragma unroll
        for(int j=0;j<8;j++)v[j]=0;
        return;
    }
    int y=2*(row>>2),x=2*(row&3);float a[8],b[8],c[8],d[8];
    const u64 o0=at(y,x,head,t,8),o1=at(y,x+1,head,t,8),o2=at(y+1,x,head,t,8),o3=at(y+1,x+1,head,t,8);
    decode(*(u32*)(window+o0),a);decode(*(u32*)(window+o0+8),a+4);
    decode(*(u32*)(window+o1),b);decode(*(u32*)(window+o1+8),b+4);
    decode(*(u32*)(window+o2),c);decode(*(u32*)(window+o2+8),c+4);
    decode(*(u32*)(window+o3),d);decode(*(u32*)(window+o3+8),d+4);
#pragma unroll
    for(int j=0;j<8;j++)v[j]=((a[j]+b[j])+(c[j]+d[j]))*.25f;
}
extern "C" __global__ void nr_encoder4(Params p) {
    const u32 lane=threadIdx.x,t=lane&3,g=lane>>2,warp=threadIdx.y;
    __shared__ __align__(16) u8 window[64*C];
    const u32* U=(const u32*)p.w;
    const int px=lane&7,py=4*warp+(lane>>3),x=8*blockIdx.x+px,y=8*blockIdx.y+py;
    // One coalesced 16-byte view fetch per pixel/channel group, then tile scatter.
#pragma unroll
    for(int cg=0;cg<C/16;cg++) {
        const V4 a=(x<p.W && y<p.H)?*(const V4*)(p.in+((u64)cg*p.H*p.W+(u64)y*p.W+x)*16):V4{0,0,0,0};
        const u64 o=at(py,px,cg/2,0,8)+(cg&1)*8;
        *(u32*)(window+o)=a.x;*(u32*)(window+o+16)=a.y;*(u32*)(window+o+32)=a.z;*(u32*)(window+o+48)=a.w;
    }
    __syncthreads();
#if !COPY_ONLY
#pragma unroll 1
    for(int stage=0;stage<STAGES;stage++) {
        Params q=p;q.w=p.w+U[8+stage];q.options=U[16+stage];body_stage(q,window);__syncthreads();
    }
#endif
    // Save the level's skip in the runtime's persistent tile layout.
#pragma unroll
    for(int unit=warp;unit<4*(C/32);unit+=2) {
        const int tile=unit/(C/32),head=unit%(C/32),tx=2*blockIdx.x+(tile&1),ty=2*blockIdx.y+(tile>>1);
        if(tx<p.W/4 && ty<p.H/4) *(V4*)(p.out+((u64)ty*(p.W/4)+tx)*16*C+head*512+lane*16)=*(V4*)(window+unit*512+lane*16);
    }
#if COPY_ONLY
    // Same three map byte counts; pool output is duplicated top-left samples.
    if(lane<16) {
        int cy=lane>>2,cx=lane&3;
#pragma unroll
        for(int cg=warp;cg<2*C/16;cg+=2) {
            const u64 o=at(2*cy,2*cx,(cg%(C/16))/2,0,8)+(cg&1)*8;
            V4 a={*(u32*)(window+o),*(u32*)(window+o+16),*(u32*)(window+o+32),*(u32*)(window+o+48)};
            int oy=4*blockIdx.y+cy,ox=4*blockIdx.x+cx;
            if(oy>=p.H/2 || ox>=p.W/2)a=V4{0,0,0,0};
            if(oy<p.d0 && ox<p.d1) *(V4*)(p.ds_out+((u64)cg*p.d0*p.d1+oy*p.d1+ox)*16)=a;
        }
    }
#else
    // C32 pads K to 64; wider levels accumulate consecutive k64 slabs.
    constexpr int PK=(C+63)/64;
    V4 a[PK];u32 sa[PK];
#pragma unroll
    for(int k=0;k<PK;k++) {
        u32 c00,c01,c10,c11;float v[8];
        pool_read(window,g,2*k,t,v);a[k].x=quant(v,c00);pool_read(window,g+8,2*k,t,v);a[k].y=quant(v,c01);
        pool_read(window,g,2*k+1,t,v);a[k].z=quant(v,c10);pool_read(window,g+8,2*k+1,t,v);a[k].w=quant(v,c11);
        sa[k]=scales(c00,c01,c10,c11,t);
    }
#pragma unroll
    for(int ob=warp;ob<2*C/32;ob+=2) {
        float ac[4][4]={};
#pragma unroll
        for(int k=0;k<PK;k++) {
            const u8* w=p.w+U[5]+(ob*PK+k)*SLAB;
            const V4 v0=*(const V4*)(w+lane*16),v1=*(const V4*)(w+512+lane*16),sc=*(const V4*)(w+1024+g*16);
            mma(ac[0],a[k],v0.x,v0.y,sa[k],sc.x);mma(ac[1],a[k],v0.z,v0.w,sa[k],sc.y);
            mma(ac[2],a[k],v1.x,v1.y,sa[k],sc.z);mma(ac[3],a[k],v1.z,v1.w,sa[k],sc.w);
        }
        const float* bias=(const float*)(p.w+U[6]);
#pragma unroll
        for(int rh=0;rh<2;rh++) {
#pragma unroll
            for(int j=0;j<4;j++) {ac[j][rh*2]+=bias[ob*32+j*8+2*t];ac[j][rh*2+1]+=bias[ob*32+j*8+2*t+1];}
            const int row=g+8*rh,oy=4*blockIdx.y+(row>>2),ox=4*blockIdx.x+(row&3);
            if(oy>=p.d0 || ox>=p.d1)continue;
            if(oy>=p.H/2 || ox>=p.W/2) {
#pragma unroll
                for(int j=0;j<4;j++){ac[j][rh*2]=0;ac[j][rh*2+1]=0;}
            }
            const u64 o=((u64)(2*ob)*p.d0*p.d1+(u64)oy*p.d1+ox)*16+4*t;
            *(u32*)(p.ds_out+o)=store8(ac[0][rh*2],ac[0][rh*2+1],ac[1][rh*2],ac[1][rh*2+1]);
            *(u32*)(p.ds_out+o+(u64)p.d0*p.d1*16)=store8(ac[2][rh*2],ac[2][rh*2+1],ac[3][rh*2],ac[3][rh*2+1]);
        }
    }
#endif
    if(p.oflags) {__syncthreads();if(lane==0&&warp==0)publish(p.oflags+blockIdx.y*gridDim.x+blockIdx.x);}
}
