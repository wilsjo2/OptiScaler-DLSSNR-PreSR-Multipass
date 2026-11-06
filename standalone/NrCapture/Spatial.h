#pragma once
#include "Gpu.h"
#include <shaders/dlssnr/DlssNr_Spatial.h>

namespace nr {
// The standalone uses the same packing and guide transformation as the addon.
// Inputs must be NON_PIXEL_SHADER_RESOURCE and outputs UNORDERED_ACCESS.
// Each slot may be recorded once before the host completes the submission.
class Spatial {
public:
    explicit Spatial(Gpu& gpu);
    ~Spatial();
    void Dispatch(unsigned slot, const DlssNr::Spatial::Constants& constants,
                  ID3D12Resource* source0, ID3D12Resource* source1, ID3D12Resource* source2,
                  ID3D12Resource* target0, ID3D12Resource* target1 = nullptr);
    // One separate dispatch slot; draws the actual sampling grid over linear colour.
    void Preview(const DlssNr::Spatial::Layout& layout, ID3D12Resource* input, ID3D12Resource* output);
private:
    Gpu& gpu;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> colour, guides, preview;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> constants;
    UINT stride = 0;
    void Record(unsigned slot, const DlssNr::Spatial::Constants& values, ID3D12PipelineState* pipeline,
                ID3D12Resource* source0, ID3D12Resource* source1, ID3D12Resource* source2,
                ID3D12Resource* target0, ID3D12Resource* target1);
};
}
