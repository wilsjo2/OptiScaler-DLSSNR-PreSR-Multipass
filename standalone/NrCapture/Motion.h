#pragma once
#include "Gpu.h"
#include <array>
namespace nr {
// Image-only optical flow, in full-resolution current-to-previous pixel units.
class Motion {
public:
    explicit Motion(Gpu& gpu);
    void Resize(UINT width, UINT height);
    void Reset() { history = false; }
    // Input must be NON_PIXEL_SHADER_RESOURCE. Normal frames submit and wait
    // for four bytes of scene-cut metadata, then open a new list. Known-reset
    // frames only record luma history and zero vectors into the current list.
    bool Estimate(ID3D12Resource* input, bool reset);
    ID3D12Resource* Vectors() const { return vectors.Get(); }
private:
    static constexpr UINT Levels = 4, DescriptorCount = (Levels * 2 + 1) * 5;
    Gpu& gpu;
    UINT width = 0, height = 0, frame = 0, stride = 0;
    bool history = false;
    std::array<std::array<ComPtr<ID3D12Resource>, Levels>, 2> luma;
    std::array<ComPtr<ID3D12Resource>, Levels> flow;
    ComPtr<ID3D12Resource> vectors, cuts, readback;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12DescriptorHeap> heap;
    std::array<ComPtr<ID3D12PipelineState>, 3> shaders;
    struct Constants { UINT width, height, inputWidth, inputHeight, reset, coarse, step, cutLimit; };
    D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(UINT index) const;
    void Dispatch(UINT slot, UINT shader, const Constants& constants,
        ID3D12Resource* current, ID3D12Resource* previous, ID3D12Resource* coarse, ID3D12Resource* output);
};
}
