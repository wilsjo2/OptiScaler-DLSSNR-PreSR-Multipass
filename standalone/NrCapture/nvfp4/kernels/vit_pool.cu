// Astra: replace the runtime's two ViT repacks. Compile only with lab.compile_kernel.
typedef unsigned char u8;
typedef unsigned int u32;
struct Params { const u8* full; u8* stream; u8* base; u8* out; u32 H,W; };
#define FI static __forceinline__ __device__
FI unsigned act(unsigned m,unsigned c) {
    return (m>>4)*16384+(c>>5)*512+(4*(m&7)+((c>>1)&3))*16+((c>>4)&1)*8+((m>>3)&1)*4+((c>>3)&1)*2+(c&1);
}
FI unsigned view(unsigned y,unsigned x,unsigned c,unsigned W) {
    return ((y>>2)*((W+3)/4)+(x>>2))*16384+act((y&3)*4+(x&3),c);
}
FI float decode(u8 x) {
    unsigned short h;
    asm("{ .reg .b16 a; .reg .b32 b; mov.b16 a,%1; cvt.rn.f16x2.e4m3x2 b,a; mov.b32 {%0,a},b; }" : "=h"(h):"h"((unsigned short)x));
    float f; asm("cvt.f32.f16 %0,%1;":"=f"(f):"h"(h)); return f;
}
FI u8 q8(float x) {
    unsigned short code;
    asm("{ .reg .b32 h; cvt.rn.f16x2.f32 h,%1,%1; cvt.rn.satfinite.e4m3x2.f16x2 %0,h; }":"=h"(code):"f"(x));
    return (u8)code;
}
extern "C" __global__ void nr_vit_pool2(Params p) {
    unsigned PH=(p.H+1)/2,PW=(p.W+1)/2,N=PH*PW, padded=(N+127)/128*128;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<padded*1024;i+=blockDim.x*gridDim.x) {
        unsigned m=i/1024,c=i%1024;u8 value=0;
        if(m<N) {
            unsigned y=(m/PW)*2,x=(m%PW)*2;
            float a=decode(p.full[view(y,x,c,p.W)]);
            float b=x+1<p.W?decode(p.full[view(y,x+1,c,p.W)]):0;
            float d=y+1<p.H?decode(p.full[view(y+1,x,c,p.W)]):0;
            float e=(y+1<p.H&&x+1<p.W)?decode(p.full[view(y+1,x+1,c,p.W)]):0;
            unsigned count=(1+(x+1<p.W))*(1+(y+1<p.H));
            value=q8(((a+b)+(d+e))*(1.0f/count));
        }
        p.stream[act(m,c)]=value;p.base[act(m,c)]=value;
    }
    // Optional native diagnostic, disabled in ordinary/timing runs.
    if(p.out) for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<p.H*p.W*1024;i+=blockDim.x*gridDim.x) p.out[i]=p.full[i];
}
struct Capture { const u8* full; const u8* stream; const u8* base; u8* dump; u32 H,W; const u8* result; };
extern "C" __global__ void nr_vit_pool_capture(Capture p) {
    unsigned f=p.H*p.W*1024,n=((p.H+1)/2)*((p.W+1)/2),a=(n+127)/128*128*1024;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<2*f+2*a;i+=blockDim.x*gridDim.x) {
        p.dump[i]=i<f?p.full[i]:i<f+a?p.base[i-f]:i<f+2*a?p.stream[i-f-a]:p.result[i-f-2*a];
    }
}
extern "C" __global__ void nr_vit_unpool2(Params p) {
    unsigned PW=(p.W+1)/2;
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<p.H*p.W*1024;i+=blockDim.x*gridDim.x) {
        unsigned m=i/1024,c=i%1024,y=m/p.W,x=m%p.W,src=act((y/2)*PW+x/2,c),dst=view(y,x,c,p.W);
        float delta=decode(p.stream[src])-decode(p.base[src]);
        p.out[dst]=q8(decode(p.full[dst])+delta);
    }
}
