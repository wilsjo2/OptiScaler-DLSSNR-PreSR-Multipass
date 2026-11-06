#pragma once
#include "Processor.h"
#include <vector>

namespace nr {
using DepthPreviewWriter = void (*)(const std::filesystem::path&, const std::vector<uint16_t>&, UINT, UINT);
void DepthCheck(Gpu& gpu, Processor& processor, ID3D12Resource* input, Settings settings, unsigned frames,
                const std::filesystem::path& directory, DepthPreviewWriter savePreview);
}
