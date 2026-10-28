#pragma once
#include "Processor.h"
#include <functional>
#include <vector>
namespace nr {
void RuntimePathCheck();
void FeatureChecks(Processor& processor, const Settings& settings, std::vector<uint16_t>& pixels,
    const std::function<void()>& upload,
    const std::function<std::vector<uint16_t>(Settings, bool, float, bool)>& read);
void HdrOutputCheck(Processor& processor, const Settings& settings, std::vector<uint16_t>& pixels,
    const std::function<void()>& upload,
    const std::function<std::vector<uint16_t>(Settings, bool, float, bool)>& read);
}
