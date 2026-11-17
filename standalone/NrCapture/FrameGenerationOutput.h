#pragma once
#include "Motion.h"
namespace nr {
// Converts our linear scRGB output to an FG-supported swap-chain colour space.
class FrameGenerationOutput {
public:
    explicit FrameGenerationOutput(Gpu& gpu);
    void Resize(UINT width, UINT height, bool hdr);
    // sceneMotion: NR's R16G16F current-to-previous vectors over the whole picture, in its own pixel units.
    // At the output size it is handed to FG as it is; at any other size the encode pass resamples it.
    // Without it FG estimates its own motion from the output picture.
    bool Run(ID3D12Resource* input, bool reset, ID3D12Resource* sceneMotion = nullptr);
    void Reset() { motion.Reset(); }
    ID3D12Resource* Colour() const { return colour.Get(); }
    ID3D12Resource* Depth() const { return depth.Get(); }
    ID3D12Resource* Vectors() const { return sharedMotion ? sharedMotion.Get() : resampled ? resampledMotion.Get() : motion.Vectors(); }
    DXGI_FORMAT Format() const { return hdr ? DXGI_FORMAT_R10G10B10A2_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM; }
private:
    Gpu& gpu;
    Motion motion;
    UINT width = 0, height = 0;
    bool hdr = false;
    ComPtr<ID3D12Resource> colour, depth;
    ComPtr<ID3D12Resource> sharedMotion, resampledMotion;
    bool resampled = false; // This frame's guide is resampledMotion.
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> shader;
    ComPtr<ID3D12DescriptorHeap> heap;
};
}
