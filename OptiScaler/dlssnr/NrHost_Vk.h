#pragma once

#include <vulkan/vulkan.h>
#include <filesystem>
#include <memory>
#include "DlssNr_ModelParameters.h"

namespace DlssNr
{
// A buffer with a device address, owned by `owner`, mapped when asked. Throws when it cannot be allocated.
struct VulkanMemory
{
    std::shared_ptr<void> owner;
    uint64_t address = 0;
    void* mapped = nullptr;
};
VulkanMemory AllocateVulkanMemory(VkPhysicalDevice physical, VkDevice device, uint64_t bytes, bool mapped);

// Records one model on the caller's Vulkan command buffer. The owner retires its GPU work before destruction.
class VulkanHost
{
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit VulkanHost(std::unique_ptr<Impl> state);

  public:
    static std::unique_ptr<VulkanHost> Create(VkPhysicalDevice physical, VkDevice device,
                                              const std::filesystem::path& assets, unsigned preset, unsigned width,
                                              unsigned height);
    ~VulkanHost();
    void Evaluate(VkCommandBuffer cmd, VkImageView colour, VkImageView motion, VkImageView output,
                  const ModelSettings& settings, float mvX, float mvY, bool reset);
};
} // namespace DlssNr
