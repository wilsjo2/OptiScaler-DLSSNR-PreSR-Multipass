#pragma once
#include "Gpu.h"
namespace nr {
// Puts two finished pictures of the same frame into one for a model comparison (Settings::ComparingModels): the
// picture's own passes on one side, the same passes run with another model on the other. Wipe: the pictures meet at
// Settings::compareSplit. Side by side: the middle half of each at full size. The other model is on the left
// (Settings::compareSwap: on the right), where an ordinary comparison has the original; a thin line marks the border.
class ModelCompare {
public:
    explicit ModelCompare(Gpu& gpu) : gpu(gpu) {}
    // Both pictures are COPY_SOURCE, of one size and format, and stay so. Records into the open list; the returned
    // picture is COPY_SOURCE and valid until the next call.
    ID3D12Resource* Compose(ID3D12Resource* picture, ID3D12Resource* other, const Settings& settings);
private:
    Gpu& gpu;
    ComPtr<ID3D12Resource> target;
    ComPtr<ID3D12DescriptorHeap> view;
};
}
