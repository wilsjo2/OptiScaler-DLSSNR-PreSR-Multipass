cbuffer Params : register(b0) {
    uint Width, Height, InputWidth, InputHeight;
    uint Reset, HasCoarse, Step, CutLimit;
    // Settings::motionZeroBias (the finest level only; 0 elsewhere): zero displacement is kept where its patch error is
    // within this share of the best displacement's, so a still area does not take noise-driven vectors.
    float ZeroBias;
};
Texture2D<float4> Current : register(t0);
Texture2D<float4> Previous : register(t1);
Texture2D<float4> Coarse : register(t2);
SamplerState LinearClamp : register(s0);
RWTexture2D<uint> Cuts : register(u1);

#ifdef RESOLVE
RWTexture2D<float2> Result : register(u0);
[numthreads(8,8,1)]
void Resolve(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    if (Reset) { Result[id.xy] = 0; return; }
    float4 estimate = Coarse.SampleLevel(LinearClamp, (id.xy + 0.5) / float2(Width, Height), 0);
    // Low-confidence regions and scene cuts use zero vectors. The host resets NR
    // history for the same scene-cut frame after reading the tiny cut counter.
    Result[id.xy] = (Cuts[uint2(0,0)] > CutLimit || estimate.w < 0.5) ? 0 : estimate.xy * 4;
}
#else
RWTexture2D<float4> Result : register(u0);
[numthreads(8,8,1)]
void Luma(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    float sum = 0;
    for (uint y=0; y<Step; ++y) for (uint x=0; x<Step; ++x) {
        uint2 p = min(id.xy * Step + uint2(x,y), uint2(InputWidth-1,InputHeight-1));
        float4 value = Current.Load(int3(p,0));
        sum += HasCoarse ? value.x : log2(1 + max(dot(value.rgb,float3(0.2126,0.7152,0.0722)),0));
    }
    Result[id.xy] = float4(sum / (Step*Step),0,0,0);
}
// An 8x8 group reuses the current image's 5x5 patches. Cache the 12x12 halo
// once instead of reloading the same texels for every displacement candidate.
groupshared float CurrentTile[144];
groupshared float PreviousZeroTile[144];
float Cost(int2 p, float2 offset, float rejectAbove, bool cachedZero) {
    float error = 0;
    int2 local = p % 8 + 2;
    [unroll] for (int y=-2; y<=2; ++y) {
        [unroll] for (int x=-2; x<=2; ++x) {
            int2 q = clamp(p + int2(x,y), int2(0,0), int2(Width-1,Height-1));
            float a = CurrentTile[(local.y+y)*12+local.x+x];
            float b = cachedZero ? PreviousZeroTile[(local.y+y)*12+local.x+x]
                : Previous.SampleLevel(LinearClamp, (q + offset + 0.5) / float2(Width,Height), 0).x;
            error += abs(a-b);
        }
        // Remaining terms and the displacement penalty are non-negative. A
        // candidate already worse than the best score cannot become the winner.
        if (error / 25 > rejectAbove) return 1e30;
    }
    return error / 25;
}
[numthreads(8,8,1)]
void Estimate(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID) {
    for (uint i=lane.y*8+lane.x; i<144; i+=64) {
        int2 q = int2(group.xy*8) + int2(i%12,i/12) - 2;
        q = clamp(q,int2(0,0),int2(Width-1,Height-1));
        CurrentTile[i] = Current.Load(int3(q,0)).x;
        // Keep the same normalized SampleLevel coordinates as Cost(p, 0).
        // Texture.Load would change filtering/coordinate rounding at some sizes.
        PreviousZeroTile[i] = Previous.SampleLevel(LinearClamp, (float2(q) + 0.5) / float2(Width,Height), 0).x;
    }
    GroupMemoryBarrierWithGroupSync();
    if (id.x >= Width || id.y >= Height) return;
    if (Reset) { Result[id.xy] = 0; return; }
    int2 p = id.xy;
    // Always consider zero displacement: static UI and flat areas must not inherit
    // a nearby object's motion merely because the coarser estimate moved.
    float2 best = 0;
    float bestCost = Cost(p,0,1e30,true), bestScore = bestCost;
    const float zeroCost = bestCost;
    // No candidate can beat a perfect zero-displacement match (ties retain zero).
    if (bestCost == 0) { Result[id.xy] = float4(0,0,0,1); return; }
    if (HasCoarse) {
        uint cw,ch; Coarse.GetDimensions(cw,ch);
        [unroll] for (int y=-1; y<=1; ++y) [unroll] for (int x=-1; x<=1; ++x) {
            int2 q=clamp(p/2+int2(x,y),int2(0,0),int2(cw-1,ch-1));
            float2 candidate=round(Coarse.Load(int3(q,0)).xy*2);
            // Zero and the current best were already scored. Nonnegative patch
            // error also cannot rescue a penalty that already loses the tie.
            if (all(candidate == 0) || all(candidate == best)) continue;
            float penalty=0.00002*dot(candidate,candidate);
            if (penalty >= bestScore) continue;
            float cost=Cost(p,candidate,bestScore,false), score=cost+penalty;
            if (score<bestScore) { best=candidate; bestCost=cost; bestScore=score; }
        }
    }
    float2 centre = best;
    int radius = HasCoarse ? 2 : 4;
    [loop] for (int y=-radius; y<=radius; ++y) [loop] for (int x=-radius; x<=radius; ++x) {
        float2 candidate = centre + float2(x,y);
        if (all(candidate == 0) || all(candidate == best)) continue;
        float penalty = 0.00002 * dot(candidate,candidate);
        if (penalty >= bestScore) continue;
        float cost = Cost(p,candidate,bestScore,false);
        float score = cost + penalty;
        if (score < bestScore) { best = candidate; bestCost = cost; bestScore = score; }
    }
    // Refine to half a pyramid pixel (two full-resolution pixels at the finest level).
    centre = best;
    [unroll] for (int y=-1; y<=1; ++y) [unroll] for (int x=-1; x<=1; ++x) {
        float2 candidate = centre + float2(x,y)*0.5;
        if (all(candidate == 0) || all(candidate == best)) continue;
        float penalty = 0.00002 * dot(candidate,candidate);
        if (penalty >= bestScore) continue;
        float cost = Cost(p,candidate,bestScore,false);
        float score = cost + penalty;
        if (score < bestScore) { best = candidate; bestCost = cost; bestScore = score; }
    }
    // (a small absolute floor as well, so a near-perfect zero match is not lost to a candidate that is only a hair better)
    if (ZeroBias > 0 && any(best != 0) && zeroCost <= bestCost * (1 + ZeroBias) + 0.002 * ZeroBias) { best = 0; bestCost = zeroCost; }
    if (!HasCoarse && bestCost > 0.12) InterlockedAdd(Cuts[uint2(0,0)],1);
    bool inside = all(p + best >= 0) && all(p + best < float2(Width,Height));
    Result[id.xy] = float4(best, bestCost, inside && bestCost < 0.08 ? 1 : 0);
}
#endif
