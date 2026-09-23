#pragma once

#include <upscalers/ShaderPipeline_Dx12.h>

class DlssNr_Dx12;

namespace DlssNr
{
inline bool FormatCanHoldLinearHdr(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return true;
    default:
        return false;
    }
}

struct InputStates_Dx12
{
    D3D12_RESOURCE_STATES color;
    D3D12_RESOURCE_STATES depth;
    D3D12_RESOURCE_STATES motion;
    D3D12_RESOURCE_STATES exposure;
};

// Shared arrival-state policy for NR input copies and private upscaler guides.
InputStates_Dx12 ResolveInputStates_Dx12(bool interop);
bool CanRunBeforeUpscale_Dx12(NVSDK_NGX_Parameter* parameters);
} // namespace DlssNr

// These adapters only translate NGX inputs and resource states. The shader owns all NR resources/history.
ShaderPass_Dx12 MakeDlssNrPass(DlssNr_Dx12& shader, ID3D12Device* device, ID3D12GraphicsCommandList* commandList,
                               NVSDK_NGX_Parameter* parameters, bool beforeUpscale, unsigned int featureFlags,
                               ID3D12CommandQueue* timingQueue = nullptr, bool interop = false,
                               bool rayReconstruction = false, uint64_t submissionEpoch = 0);
ID3D12Resource* PrepareDlssNrInput(DlssNr_Dx12& shader, ID3D12Device* device, ID3D12GraphicsCommandList* commandList,
                                   NVSDK_NGX_Parameter* parameters, unsigned int featureFlags,
                                   ID3D12CommandQueue* timingQueue = nullptr, bool interop = false,
                                   bool rayReconstruction = false, uint64_t submissionEpoch = 0);
