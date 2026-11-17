#include "FrameGenerationOutput.h"
#include <FrameGenerationEncode.h>
namespace nr {
FrameGenerationOutput::FrameGenerationOutput(Gpu& g) : gpu(g), motion(g) {
    D3D12_DESCRIPTOR_RANGE ranges[]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,2,0,0,0}, {D3D12_DESCRIPTOR_RANGE_TYPE_UAV,3,0,0,2}};
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[0].DescriptorTable = {2,ranges};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[1].Constants = {0,0,5};
    D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX; sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC signature{2,parameters,1,&sampler}; ComPtr<ID3DBlob> blob, error;
    Check(D3D12SerializeRootSignature(&signature,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error), "FG output signature");
    Check(g.device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)), "FG output root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC p{}; p.pRootSignature = root.Get(); p.CS = {FrameGenerationEncode,sizeof(FrameGenerationEncode)};
    Check(g.device->CreateComputePipelineState(&p,IID_PPV_ARGS(&shader)), "FG output pipeline");
    D3D12_DESCRIPTOR_HEAP_DESC desc{}; desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 5; desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(g.device->CreateDescriptorHeap(&desc,IID_PPV_ARGS(&heap)), "FG output descriptors");
}
void FrameGenerationOutput::Resize(UINT w, UINT h, bool isHdr) {
    if (w == width && h == height && hdr == isHdr) return;
    if (!gpu.Drain()) throw std::runtime_error("GPU timeout resizing frame generation inputs.");
    width = w; height = h; hdr = isHdr; motion.Resize(w,h); motion.Reset();
    sharedMotion.Reset(); resampledMotion.Reset(); resampled = false;
    colour = gpu.Texture(w,h,Format(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    depth = gpu.Texture(w,h,DXGI_FORMAT_R32_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}
bool FrameGenerationOutput::Run(ID3D12Resource* input, bool reset, ID3D12Resource* sceneMotion) {
    gpu.Time(GpuStage::FgMotion);
    Barrier(gpu.commands.Get(),input,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    bool cut = reset;
    UINT motionWidth = 0, motionHeight = 0; // Non-zero: the encode pass resamples sceneMotion from this size.
    sharedMotion.Reset(); resampled = false;
    if (sceneMotion) {
        const auto desc = sceneMotion->GetDesc();
        if (desc.Format != DXGI_FORMAT_R16G16_FLOAT) throw std::runtime_error("FG shared motion must be R16G16F vectors.");
        // As with engine-provided guides, reuse the scene motion already used by
        // NR. Its completion is covered by the same FG input-consumption fence.
        if (desc.Width == width && desc.Height == height) sharedMotion = sceneMotion;
        else {
            motionWidth = UINT(desc.Width); motionHeight = desc.Height; resampled = true;
            if (!resampledMotion) resampledMotion = gpu.Texture(width,height,DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        motion.Reset();
    } else cut = motion.Estimate(input,reset);
    gpu.Time(GpuStage::FgMotion, true);
    gpu.Time(GpuStage::FgEncode);
    auto* c = gpu.commands.Get();
    Barrier(c,colour.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Barrier(c,depth.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (resampled) Barrier(c,resampledMotion.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto handle = heap->GetCPUDescriptorHandleForHeapStart(); const auto stride = gpu.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{}; srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; srv.Texture2D.MipLevels = 1; srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    gpu.device->CreateShaderResourceView(input,&srv,handle); handle.ptr += stride;
    // Unused motion slots hold null views: the shader touches them only while resampling.
    srv.Format = DXGI_FORMAT_R16G16_FLOAT;
    gpu.device->CreateShaderResourceView(resampled ? sceneMotion : nullptr,&srv,handle); handle.ptr += stride;
    gpu.device->CreateUnorderedAccessView(colour.Get(),nullptr,nullptr,handle); handle.ptr += stride;
    gpu.device->CreateUnorderedAccessView(depth.Get(),nullptr,nullptr,handle); handle.ptr += stride;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.Format = DXGI_FORMAT_R16G16_FLOAT; uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(resampled ? resampledMotion.Get() : nullptr,nullptr,&uav,handle);
    ID3D12DescriptorHeap* heaps[]{heap.Get()}; c->SetDescriptorHeaps(1,heaps);
    c->SetComputeRootSignature(root.Get()); c->SetPipelineState(shader.Get());
    c->SetComputeRootDescriptorTable(0,heap->GetGPUDescriptorHandleForHeapStart());
    const UINT values[]{width,height,UINT(hdr),motionWidth,motionHeight}; c->SetComputeRoot32BitConstants(1,5,values,0);
    c->Dispatch((width+7)/8,(height+7)/8,1);
    Barrier(c,input,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(c,colour.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(c,depth.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (resampled) Barrier(c,resampledMotion.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Time(GpuStage::FgEncode, true);
    return cut;
}
}
