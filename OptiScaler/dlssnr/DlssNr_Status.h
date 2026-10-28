#pragma once

#include <cstdint>
#include <optional>
#include <string>

class Config;

namespace DlssNr
{
enum class Backend
{
    Dx12,
    Vulkan
};

// Menu telemetry contains values only. GPU resources remain owned by the shader instance.
struct StatusSnapshot
{
    bool running = false;
    std::string failureReason;
    std::optional<double> gpuTime;
    unsigned long long frames = 0;
    std::string spatialStatus;
    bool spatialActive = false;
};

struct ControlRequests
{
    uint64_t retryGeneration = 0;
    uint64_t captureGeneration = 0;
    unsigned int captureFrames = 0;
};

void PublishStatus(const void* owner, Backend backend, const StatusSnapshot& status);
void ClearStatus(const void* owner);
StatusSnapshot ReadStatus(Backend backend);
ControlRequests ReadControlRequests();

void RenderMenu(::Config* config, float menuResScale);
void RetryAfterFailure();
std::optional<double> LastGpuTime();
void RequestCapture(unsigned int frames);
} // namespace DlssNr
