#pragma once
#include "Gpu.h"
#include <array>
namespace nr {
// Image-only optical flow, in full-resolution current-to-previous pixel units.
class Motion {
public:
    explicit Motion(Gpu& gpu);
    void Resize(UINT width, UINT height);
    void Reset() { history = false; pendingCut = false; }
    // Input must be NON_PIXEL_SHADER_RESOURCE. Everything is recorded into the
    // host's open list: nothing is submitted or waited for here. The shader
    // zeroes the vectors of a scene-cut frame itself; the four-byte cut count
    // is read when the next frame is estimated, so the returned history reset
    // follows a detected cut by one frame. Call once per submitted list.
    bool Estimate(ID3D12Resource* input, bool reset);
    ID3D12Resource* Vectors() const { return vectors.Get(); }
    // The cut counter of the frame just estimated (one R32_UINT texel, UNORDERED_ACCESS): coarse cells without a match.
    ID3D12Resource* Cuts() const { return cuts.Get(); }
    // The count read at the last Estimate (it belongs to the frame before that call) and the count a cut needs.
    UINT lastCutCount = 0, lastCutLimit = 0;
private:
    static constexpr UINT Levels = 4, DescriptorCount = (Levels * 2 + 1) * 5;
    Gpu& gpu;
    UINT width = 0, height = 0, frame = 0, stride = 0;
    bool history = false, pendingCut = false;
    UINT pendingCutLimit = 0;
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
