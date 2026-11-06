#pragma once
#include "Gpu.h"
#include <ModelOverlapResize.h>

namespace nr {
// Probe-only interpass preparation: clamp before area filtering and retain
// original-frame alpha. All resources enter/leave COMMON. The caller must
// complete the prior submission before changing this helper's descriptors.
struct ModelProbePrepare {
    Gpu& gpu;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> heap;
    explicit ModelProbePrepare(Gpu& gpu) : gpu(gpu) {
        D3D12_DESCRIPTOR_RANGE ranges[]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0},
            {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 2}};
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable = {2, ranges};
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants = {0, 0, 4};
        D3D12_ROOT_SIGNATURE_DESC signature{2, parameters}; ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "Overlap handoff root signature");
        Check(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Overlap handoff root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC description{}; description.pRootSignature = root.Get();
        description.CS = {ModelOverlapResize, sizeof ModelOverlapResize};
        Check(gpu.device->CreateComputePipelineState(&description, IID_PPV_ARGS(&pipeline)), "Overlap handoff pipeline");
        D3D12_DESCRIPTOR_HEAP_DESC descriptors{}; descriptors.NumDescriptors = 3;
        descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; descriptors.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(gpu.device->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(&heap)), "Overlap handoff descriptors");
    }
    void Run(ID3D12GraphicsCommandList* commands, ID3D12Resource* answer, ID3D12Resource* original, ID3D12Resource* target) {
        auto handle = heap->GetCPUDescriptorHandleForHeapStart();
        const auto stride = gpu.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        for (auto* source : {answer, original}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = source->GetDesc().Format;
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; view.Texture2D.MipLevels = 1;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            gpu.device->CreateShaderResourceView(source, &view, handle); handle.ptr += stride;
            Barrier(commands, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        gpu.device->CreateUnorderedAccessView(target, nullptr, nullptr, handle);
        Barrier(commands, target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[]{heap.Get()}; commands->SetDescriptorHeaps(1, heaps);
        commands->SetComputeRootSignature(root.Get()); commands->SetPipelineState(pipeline.Get());
        commands->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        const UINT constants[]{UINT(answer->GetDesc().Width), answer->GetDesc().Height, UINT(target->GetDesc().Width), target->GetDesc().Height};
        commands->SetComputeRoot32BitConstants(1, 4, constants, 0);
        commands->Dispatch((constants[2] + 7) / 8, (constants[3] + 7) / 8, 1);
        Barrier(commands, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        for (auto* source : {answer, original})
            Barrier(commands, source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
};
}
