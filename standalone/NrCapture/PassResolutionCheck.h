#pragma once
#include "Processor.h"
#include <functional>
#include <vector>

namespace nr {
using PassResolutionPreviewWriter = void (*)(const std::filesystem::path&, const std::vector<uint16_t>&, UINT, UINT);
void PassResolutionCheck(Processor& processor, const Settings& settings, std::vector<uint16_t>& pixels,
    const std::function<void()>& upload,
    const std::function<std::vector<uint16_t>(Settings, bool, float, bool)>& read,
    const std::filesystem::path& directory, PassResolutionPreviewWriter savePreview);
}
