#include "Motion.h"
#include <MotionLuma.h>
#include <MotionEstimate.h>
#include <MotionResolve.h>
namespace nr {
Motion::Motion(Gpu& g) : gpu(g) {
    D3D12_DESCRIPTOR_RANGE ranges[]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,3,0,0,0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV,2,0,0,3}};
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[0].DescriptorTable = {2,ranges};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[1].Constants = {0,0,9};
    D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP; sampler.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_ROOT_SIGNATURE_DESC signature{2,parameters,1,&sampler,D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, errors;
    Check(D3D12SerializeRootSignature(&signature,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors), "Motion root signature");
    Check(gpu.device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)), "Motion root");
    const D3D12_SHADER_BYTECODE bytecode[]{{MotionLuma,sizeof(MotionLuma)},
        {MotionEstimate,sizeof(MotionEstimate)},{MotionResolve,sizeof(MotionResolve)}};
    for (UINT i=0; i<3; ++i) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature = root.Get();
        pipeline.CS = bytecode[i];
        Check(gpu.device->CreateComputePipelineState(&pipeline,IID_PPV_ARGS(&shaders[i])),"Motion pipeline");
    }
    D3D12_DESCRIPTOR_HEAP_DESC h{}; h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    h.NumDescriptors = DescriptorCount; h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gpu.device->CreateDescriptorHeap(&h,IID_PPV_ARGS(&heap)),"Motion descriptors");
    stride = gpu.device->GetDescriptorHandleIncrementSize(h.Type);
    cuts = gpu.Texture(1,1,DXGI_FORMAT_R32_UINT); readback = gpu.Buffer(256,D3D12_HEAP_TYPE_READBACK);
}
D3D12_CPU_DESCRIPTOR_HANDLE Motion::Cpu(UINT i) const { auto h=heap->GetCPUDescriptorHandleForHeapStart(); h.ptr+=SIZE_T(i)*stride; return h; }
D3D12_GPU_DESCRIPTOR_HANDLE Motion::GpuHandle(UINT i) const { auto h=heap->GetGPUDescriptorHandleForHeapStart(); h.ptr+=UINT64(i)*stride; return h; }
void Motion::Resize(UINT w, UINT h) {
    if (width==w && height==h) return;
    if (!gpu.Drain()) throw std::runtime_error("GPU timeout resizing optical flow.");
    width=w; height=h; history=false; pendingCut=false; frame=0;
    UINT lw=(w+3)/4, lh=(h+3)/4;
    for (UINT i=0; i<Levels; ++i) {
        for (auto& pyramid:luma) pyramid[i]=gpu.Texture(lw,lh,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        flow[i]=gpu.Texture(lw,lh,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        lw=(lw+1)/2; lh=(lh+1)/2;
    }
    vectors=gpu.Texture(w,h,DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}
void Motion::Dispatch(UINT slot, UINT shader, const Constants& constants, ID3D12Resource* current,
    ID3D12Resource* previous, ID3D12Resource* coarse, ID3D12Resource* output) {
    const UINT offset=slot*5; ID3D12Resource* sources[]{current,previous,coarse};
    for (UINT i=0; i<3; ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format=sources[i]->GetDesc().Format;
        view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D; view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture2D.MipLevels=1; gpu.device->CreateShaderResourceView(sources[i],&view,Cpu(offset+i));
    }
    gpu.device->CreateUnorderedAccessView(output,nullptr,nullptr,Cpu(offset+3));
    gpu.device->CreateUnorderedAccessView(cuts.Get(),nullptr,nullptr,Cpu(offset+4));
    auto* c=gpu.commands.Get(); ID3D12DescriptorHeap* heaps[]{heap.Get()}; c->SetDescriptorHeaps(1,heaps);
    c->SetComputeRootSignature(root.Get()); c->SetPipelineState(shaders[shader].Get());
    c->SetComputeRootDescriptorTable(0,GpuHandle(offset)); c->SetComputeRoot32BitConstants(1,9,&constants,0);
    Barrier(c,output,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c->Dispatch((constants.width+7)/8,(constants.height+7)/8,1);
    Barrier(c,output,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}
bool Motion::Estimate(ID3D12Resource* input, bool reset, float zeroBias) {
    if (pendingCut) {
        // The previous frame's list has completed: the host opens a new list only after that.
        void* data=nullptr; D3D12_RANGE range{0,sizeof(UINT)}; Check(readback->Map(0,&range,&data),"Read scene-cut flag");
        lastCutCount = *static_cast<UINT*>(data); lastCutLimit = pendingCutLimit;
        reset = reset || lastCutCount>pendingCutLimit; D3D12_RANGE written{0,0}; readback->Unmap(0,&written);
        pendingCut=false;
    }
    reset = reset || !history;
    auto& current=luma[frame]; auto& previous=luma[1-frame];
    auto coarseSize=flow.back()->GetDesc(); const UINT cutLimit=UINT(coarseSize.Width)*coarseSize.Height/2;
    ID3D12DescriptorHeap* heaps[]{heap.Get()}; gpu.commands->SetDescriptorHeaps(1,heaps);
    gpu.device->CreateUnorderedAccessView(cuts.Get(),nullptr,nullptr,Cpu(4));
    const UINT zero[4]{}; gpu.commands->ClearUnorderedAccessViewUint(GpuHandle(4),Cpu(4),cuts.Get(),zero,0,nullptr);
    D3D12_RESOURCE_BARRIER order{}; order.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; order.UAV.pResource=cuts.Get();
    gpu.commands->ResourceBarrier(1,&order);
    for (UINT i=0; i<Levels; ++i) {
        auto* source=i ? current[i-1].Get() : input; auto from=source->GetDesc(), to=current[i]->GetDesc();
        Constants p{UINT(to.Width),to.Height,UINT(from.Width),from.Height,UINT(reset),UINT(i!=0),i ? 2u : 4u,cutLimit};
        Dispatch(i,0,p,source,source,source,current[i].Get());
    }
    if (reset) {
        // The cut decision is already known. Seed the next frame's luma history
        // and record zero vectors, without running searches. Flow levels are all
        // recomputed before they are consumed on the next normal frame.
        Constants p{width,height,0,0,1,0,0,cutLimit};
        Dispatch(Levels*2,2,p,current[0].Get(),previous[0].Get(),flow[0].Get(),vectors.Get());
        history=true; frame=1-frame;
        return true;
    }
    for (int i=Levels-1; i>=0; --i) {
        auto size=flow[i]->GetDesc(); auto* coarser=i+1<int(Levels) ? flow[i+1].Get() : current[i].Get();
        Constants p{UINT(size.Width),size.Height,0,0,UINT(reset),UINT(i+1<int(Levels)),0,cutLimit, i == 0 ? zeroBias : 0.f};
        Dispatch(Levels+UINT(i),1,p,current[i].Get(),previous[i].Get(),coarser,flow[i].Get());
    }
    gpu.commands->ResourceBarrier(1,&order);
    Constants p{width,height,0,0,UINT(reset),0,0,cutLimit};
    Dispatch(Levels*2,2,p,current[0].Get(),previous[0].Get(),flow[0].Get(),vectors.Get());
    Barrier(gpu.commands.Get(),cuts.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{},dst{}; src.pResource=cuts.Get(); src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource=readback.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint={DXGI_FORMAT_R32_UINT,1,1,1,256};
    gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    Barrier(gpu.commands.Get(),cuts.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // No submit-and-wait: the count is collected at the start of the next estimate.
    pendingCut=true; pendingCutLimit=cutLimit;
    history=true; frame=1-frame; return false;
}
}
