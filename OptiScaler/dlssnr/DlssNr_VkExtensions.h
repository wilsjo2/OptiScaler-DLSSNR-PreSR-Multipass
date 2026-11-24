#pragma once

// NR's device extensions must be requested at vkCreateDevice, before model initialization.

#include <vulkan/vulkan.h>

#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <mutex>
#include <unordered_set>

namespace DlssNr::VkExt
{

// Names taken from nvngx_dlssnr.dll. Append only supported extensions the game has not enabled.
inline const char* const kDevice[] = {
    "VK_NVX_binary_import",
    "VK_NVX_image_view_handle",
    VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
};

inline std::mutex hostDeviceMutex;
inline std::unordered_set<VkDevice> hostDevices;
inline void RecordHostDevice(VkDevice device, bool ready)
{
    std::lock_guard lock(hostDeviceMutex);
    if (ready) hostDevices.insert(device);
    else hostDevices.erase(device);
}
inline bool HostDeviceReady(VkDevice device)
{
    std::lock_guard lock(hostDeviceMutex);
    return hostDevices.contains(device);
}

// Own the pointer array until vkCreateDevice returns; the names themselves are borrowed.
struct Merged
{
    std::vector<const char*> names;
};

inline std::vector<VkExtensionProperties> SupportedDeviceExtensions(PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                                                          VkInstance instance, VkPhysicalDevice physicalDevice)
{

    if (getInstanceProcAddr == nullptr || physicalDevice == VK_NULL_HANDLE)
        return {};

    auto enumerate = (PFN_vkEnumerateDeviceExtensionProperties) getInstanceProcAddr(
        instance, "vkEnumerateDeviceExtensionProperties");

    if (enumerate == nullptr)
        return {};

    uint32_t count = 0;

    if (enumerate(physicalDevice, nullptr, &count, nullptr) != VK_SUCCESS || count == 0)
        return {};

    std::vector<VkExtensionProperties> props(count);

    if (enumerate(physicalDevice, nullptr, &count, props.data()) != VK_SUCCESS)
        return {};

    props.resize(count);
    return props;
}

inline bool Contains(const std::vector<VkExtensionProperties>& extensions, std::string_view name, uint32_t revision = 1)
{
    return std::any_of(extensions.begin(), extensions.end(), [&](const auto& ext) {
        return name == ext.extensionName && ext.specVersion >= revision;
    });
}

inline bool ListHas(const char* const* list, uint32_t count, const char* needle)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        if (list[i] != nullptr && std::string_view(list[i]) == needle)
            return true;
    }

    return false;
}

} // namespace DlssNr::VkExt
