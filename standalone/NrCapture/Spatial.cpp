#include "Spatial.h"
#include <shaders/dlssnr/precompile/dlssnr_spatial_Shader.h>
#include <shaders/dlssnr/precompile/dlssnr_spatial_guides_Shader.h>
#include <SpatialPreview.h>

namespace nr {
Spatial::Spatial(Gpu& g) : gpu(g) {
    D3D12_DESCRIPTOR_RANGE ranges[]{
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 3}};
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {2, ranges};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_ROOT_SIGNATURE_DESC signature{2, parameters, 1, &sampler};
    ComPtr<ID3DBlob> blob, error;
    Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), "Spatial root signature");
    Check(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Spatial root");
    auto create = [&](const void* code, size_t size, ComPtr<ID3D12PipelineState>& target) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature = root.Get(); desc.CS = {code, size};
        Check(gpu.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&target)), "Spatial pipeline");
    };
    create(dlssnr_spatial_cso, sizeof(dlssnr_spatial_cso), colour);
    create(dlssnr_spatial_guides_cso, sizeof(dlssnr_spatial_guides_cso), guides);
    create(SpatialPreview, sizeof(SpatialPreview), preview);
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 4 * 5; desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gpu.device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)), "Spatial descriptors");
    stride = gpu.device->GetDescriptorHandleIncrementSize(desc.Type);
    constants = gpu.Buffer(4 * sizeof(DlssNr::Spatial::Constants), D3D12_HEAP_TYPE_UPLOAD);
}
Spatial::~Spatial() {
    gpu.lifetime.Retire([rs = std::move(root), a = std::move(colour), b = std::move(guides),
        c = std::move(preview), descriptors = std::move(heap), cb = std::move(constants)] {});
    gpu.lifetime.Collect();
}
void Spatial::Record(unsigned slot, const DlssNr::Spatial::Constants& values, ID3D12PipelineState* pipeline,
    ID3D12Resource* source0, ID3D12Resource* source1, ID3D12Resource* source2,
    ID3D12Resource* target0, ID3D12Resource* target1) {
    if (!source0 || !target0 || !values.width || !values.height)
        throw std::runtime_error("Spatial dispatch needs valid colour, target and dimensions.");
    ID3D12Resource* inputs[]{source0, source1 ? source1 : source0, source2 ? source2 : source0};
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart(); cpu.ptr += SIZE_T(slot) * 5 * stride;
    auto descriptor = heap->GetGPUDescriptorHandleForHeapStart(); descriptor.ptr += UINT64(slot) * 5 * stride;
    for (auto* input : inputs) {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = input->GetDesc().Format;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; view.Texture2D.MipLevels = 1;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        gpu.device->CreateShaderResourceView(input, &view, cpu); cpu.ptr += stride;
    }
    gpu.device->CreateUnorderedAccessView(target0, nullptr, nullptr, cpu); cpu.ptr += stride;
    gpu.device->CreateUnorderedAccessView(target1 ? target1 : target0, nullptr, nullptr, cpu);
    void* mapped = nullptr; Check(constants->Map(0, nullptr, &mapped), "Spatial constants");
    const UINT offset = slot * UINT(sizeof(DlssNr::Spatial::Constants));
    memcpy(static_cast<BYTE*>(mapped) + offset, &values, sizeof(values)); constants->Unmap(0, nullptr);
    auto* commands = gpu.commands.Get(); ID3D12DescriptorHeap* heaps[]{heap.Get()};
    commands->SetDescriptorHeaps(1, heaps); commands->SetComputeRootSignature(root.Get());
    commands->SetPipelineState(pipeline); commands->SetComputeRootDescriptorTable(0, descriptor);
    commands->SetComputeRootConstantBufferView(1, constants->GetGPUVirtualAddress() + offset);
    commands->Dispatch((values.width + 7) / 8, (values.height + 7) / 8, 1);
}
void Spatial::Dispatch(unsigned slot, const DlssNr::Spatial::Constants& values,
    ID3D12Resource* source0, ID3D12Resource* source1, ID3D12Resource* source2,
    ID3D12Resource* target0, ID3D12Resource* target1) {
    if (slot >= 3 || values.mode < 100 || values.mode > 102)
        throw std::runtime_error("Invalid spatial dispatch slot or mode.");
    if ((values.mode == 101 && (!source1 || !source2 || !target1)) ||
        (values.mode == 102 && (!source1 || !target1)))
        throw std::runtime_error("Spatial guide/unpack dispatch is missing an input or output.");
    Record(slot, values, values.mode == 101 ? guides.Get() : colour.Get(),
        source0, source1, source2, target0, target1);
}
void Spatial::Preview(const DlssNr::Spatial::Layout& layout, ID3D12Resource* input, ID3D12Resource* output) {
    DlssNr::Spatial::Constants values{};
    values.mode = layout.active ? 103 : 104;
    values.width = layout.nativeW; values.height = layout.nativeH;
    values.unused = std::max(8.0f, 24.0f * layout.globalScale);
    if (layout.active) values.warp = pw::MakeShaderConstants(layout.warp);
    else {
        values.warp.nativeWidth = float(layout.nativeW); values.warp.nativeHeight = float(layout.nativeH);
        values.warp.workWidth = float(layout.ordinaryW); values.warp.workHeight = float(layout.ordinaryH);
    }
    Record(3, values, preview.Get(), input, nullptr, nullptr, output, nullptr);
}
}
