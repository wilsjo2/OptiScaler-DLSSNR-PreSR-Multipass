// Headless smoke for the production pass host. Takes the packaged nvfp4 folder and a preset.
#include "../OptiScaler/dlssnr/NrHost_Vk.h"
#include "../OptiScaler/dlssnr/DlssNr_Image_Vk.h"
#include <DirectXPackedVector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

static void Check(VkResult result)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error("VkResult " + std::to_string(result));
}
int main(int argc, char** argv)
try
{
    if (argc < 3)
        throw std::runtime_error("Pass nvfp4 asset folder and preset number");
    VkApplicationInfo app { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VkInstance instance {};
    Check(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t count = 0;
    Check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    if (!count)
        throw std::runtime_error("No Vulkan device");
    std::vector<VkPhysicalDevice> physicals(count);
    Check(vkEnumeratePhysicalDevices(instance, &count, physicals.data()));
    VkPhysicalDevice physical = physicals.front();
    for (auto candidate : physicals)
    {
        VkPhysicalDeviceProperties p {};
        vkGetPhysicalDeviceProperties(candidate, &p);
        if (p.vendorID == 0x10de)
            physical = candidate;
    }
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(physical, &props);
    std::cout << props.deviceName << '\n';
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    uint32_t family = 0;
    while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        ++family;
    if (family == count)
        throw std::runtime_error("No Vulkan compute queue");
    float priority = 1;
    VkDeviceQueueCreateInfo qci { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    const char* extensions[] = { VK_NVX_BINARY_IMPORT_EXTENSION_NAME, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME,
                                 VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME };
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = extensions;
    VkPhysicalDeviceBufferDeviceAddressFeatures bda {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES
    };
    bda.bufferDeviceAddress = true;
    dci.pNext = &bda;
    VkDevice device {};
    Check(vkCreateDevice(physical, &dci, nullptr, &device));
    VkQueue queue {};
    vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory {};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    auto allocate = [&](VkMemoryRequirements req, VkMemoryPropertyFlags flags)
    {
        VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = req.size;
        for (; ai.memoryTypeIndex < memory.memoryTypeCount; ++ai.memoryTypeIndex)
            if ((req.memoryTypeBits & (1u << ai.memoryTypeIndex)) &&
                (memory.memoryTypes[ai.memoryTypeIndex].propertyFlags & flags) == flags)
                break;
        if (ai.memoryTypeIndex == memory.memoryTypeCount)
            throw std::runtime_error("No compatible memory");
        VkDeviceMemory m {};
        Check(vkAllocateMemory(device, &ai, nullptr, &m));
        return m;
    };
    struct Buffer
    {
        VkBuffer handle {};
        VkDeviceMemory memory {};
        void* mapped {};
    };
    auto makeBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage)
    {
        Buffer b;
        VkBufferCreateInfo ci { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        ci.size = size;
        ci.usage = usage;
        Check(vkCreateBuffer(device, &ci, nullptr, &b.handle));
        VkMemoryRequirements req {};
        vkGetBufferMemoryRequirements(device, b.handle, &req);
        b.memory = allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        Check(vkBindBufferMemory(device, b.handle, b.memory, 0));
        Check(vkMapMemory(device, b.memory, 0, size, 0, &b.mapped));
        return b;
    };

    constexpr unsigned width = 321, height = 193;
    constexpr size_t samples = width * height * 4, bytes = samples * sizeof(uint16_t);
    auto pixels = makeBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::array<DlssNr::ImageVk, 3> images {};
    for (auto& image : images)
    {
        if (!image.Ensure(device, physical, width, height, VK_FORMAT_R16G16B16A16_SFLOAT))
            throw std::runtime_error("image allocation failed");
    }
    auto model = DlssNr::VulkanHost::Create(physical, device, argv[1], std::stoul(argv[2]), width, height);
    VkCommandPoolCreateInfo pci { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = family;
    VkCommandPool pool {};
    Check(vkCreateCommandPool(device, &pci, nullptr, &pool));
    VkCommandBufferAllocateInfo cai { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = pool;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd {};
    Check(vkAllocateCommandBuffers(device, &cai, &cmd));
    using namespace DirectX::PackedVector;
    for (unsigned frame = 0; frame < 3; ++frame)
    {
        std::vector<uint16_t> source(samples);
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
            {
                auto* p = source.data() + 4 * (y * width + x);
                p[0] = XMConvertFloatToHalf(float(x + frame) / (width + 3));
                p[1] = XMConvertFloatToHalf(float(y) / height);
                p[2] = XMConvertFloatToHalf(.15f + .25f * ((x / 16 + y / 8) % 2));
                p[3] = XMConvertFloatToHalf(1.f);
            }
        memcpy(pixels.mapped, source.data(), bytes);
        VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        Check(vkBeginCommandBuffer(cmd, &begin));
        auto barrier = [&](DlssNr::ImageVk& image, VkImageLayout layout)
        {
            VkImageMemoryBarrier b { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            b.srcAccessMask =
                image.layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.oldLayout = image.layout;
            b.newLayout = layout;
            b.image = image.info.Image;
            b.subresourceRange = image.info.SubresourceRange;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b);
            image.layout = layout;
        };
        VkBufferImageCopy copy {};
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent = { width, height, 1 };
        barrier(images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cmd, pixels.handle, images[0].info.Image, images[0].layout, 1, &copy);
        barrier(images[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        barrier(images[1], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue zero {};
        vkCmdClearColorImage(cmd, images[1].info.Image, images[1].layout, &zero, 1, &images[1].info.SubresourceRange);
        barrier(images[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        barrier(images[2], VK_IMAGE_LAYOUT_GENERAL);
        DlssNr::ModelSettings settings;
        settings.style = frame == 0 ? 0 : 2;
        settings.localTone = 1;
        model->Evaluate(cmd, images[0].info.ImageView, images[1].info.ImageView, images[2].info.ImageView, settings, 1,
                        1, frame == 0);
        barrier(images[2], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vkCmdCopyImageToBuffer(cmd, images[2].info.Image, images[2].layout, pixels.handle, 1, &copy);
        Check(vkEndCommandBuffer(cmd));
        VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        Check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        Check(vkQueueWaitIdle(queue));
        auto* result = static_cast<uint16_t*>(pixels.mapped);
        double total = 0, error = 0;
        for (size_t i = 0; i < samples; ++i)
            if (i % 4 != 3)
            {
                const float value = XMConvertHalfToFloat(result[i]);
                if (!std::isfinite(value))
                    throw std::runtime_error("nonfinite output");
                total += value;
                error += std::abs(value - XMConvertHalfToFloat(source[i]));
            }
        const double mean = total / (width * height * 3.), delta = error / (width * height * 3.);
        std::cout << "preset " << argv[2] << " frame " << frame << " mean " << mean << " mean absolute edit " << delta
                  << std::endl;
        if (mean <= .01 || mean >= 2 || delta < .00001)
            throw std::runtime_error("blank, invalid or unedited frame");
        if (argc == 4)
        {
            std::ofstream dump(std::string(argv[3]) + std::to_string(frame) + ".rgba16f", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(result), bytes);
        }
        Check(vkResetCommandPool(device, pool, 0));
    }
    model.reset();
    for (auto& image : images)
        image.Destroy(device);
    vkUnmapMemory(device, pixels.memory);
    vkDestroyBuffer(device, pixels.handle, nullptr);
    vkFreeMemory(device, pixels.memory, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return 1;
}
