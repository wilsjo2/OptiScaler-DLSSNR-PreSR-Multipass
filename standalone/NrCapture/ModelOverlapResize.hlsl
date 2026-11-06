// Probe-only interpass preparation. The loop/order below follows shared
// dlssnr.hlsl mode 2 (area downsample), with mode 8's clamp applied per source
// texel. Original-frame alpha matches the standalone's raw stage resolve.
cbuffer Params : register(b0) { uint SourceW, SourceH, TargetW, TargetH; };
Texture2D<float4> Answer : register(t0);
Texture2D<float4> Original : register(t1);
RWTexture2D<float4> Target : register(u0);
float3 Bounded(int2 p) {
    float3 value = Answer.Load(int3(p, 0)).rgb;
    return saturate(float3(isfinite(value.x) ? value.x : 0.5,
                          isfinite(value.y) ? value.y : 0.5,
                          isfinite(value.z) ? value.z : 0.5));
}
[numthreads(8,8,1)]
void Prepare(uint3 id : SV_DispatchThreadID) {
    if (id.x >= TargetW || id.y >= TargetH) return;
    const float x0 = ((float)id.x * (float)SourceW) / (float)TargetW;
    const float x1 = ((float)(id.x + 1) * (float)SourceW) / (float)TargetW;
    const float y0 = ((float)id.y * (float)SourceH) / (float)TargetH;
    const float y1 = ((float)(id.y + 1) * (float)SourceH) / (float)TargetH;
    const float area = (x1-x0)*(y1-y0);
    const int i0 = (int)floor(x0), i1 = (int)ceil(x1)-1;
    const int j0 = (int)floor(y0), j1 = (int)ceil(y1)-1;
    float3 acc = 0;
    for (int j=j0; j<=j1; ++j) {
        const int jj = clamp(j,0,(int)SourceH-1);
        const float aY = max(y0,(float)j), bY = min(y1,(float)j+1.0);
        const float wy = max(bY-aY,0.0);
        for (int i=i0; i<=i1; ++i) {
            const int ii = clamp(i,0,(int)SourceW-1);
            const float aX = max(x0,(float)i), bX = min(x1,(float)i+1.0);
            acc += Bounded(int2(ii,jj))*(max(bX-aX,0.0)*wy);
        }
    }
    const int acx = clamp((int)floor(((float)id.x+0.5)*(float)SourceW/(float)TargetW),0,(int)SourceW-1);
    const int acy = clamp((int)floor(((float)id.y+0.5)*(float)SourceH/(float)TargetH),0,(int)SourceH-1);
    Target[id.xy] = float4(acc/area,Original.Load(int3(acx,acy,0)).a);
}
