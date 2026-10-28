// Real Vulkan dispatch of both peripheral spatial shaders; no NGX runtime required.
#include <vulkan/vulkan.h>
#include "../OptiScaler/dlssnr/DlssNr_Image_Vk.h"
#include "../OptiScaler/shaders/dlssnr/DlssNr_Spatial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Check(VkResult r) { if (r != VK_SUCCESS) throw std::runtime_error("Vulkan result " + std::to_string(r)); }
static void Near(float got, float expected, float tolerance = .003f) {
    if (!std::isfinite(got) || std::abs(got - expected) > tolerance)
        throw std::runtime_error("Pixel mismatch: got " + std::to_string(got) + ", expected " + std::to_string(expected));
}
static std::vector<uint32_t> ReadSpv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() <= 0 || (static_cast<size_t>(f.tellg()) % 4) != 0) throw std::runtime_error("Bad SPIR-V path");
    std::vector<uint32_t> out(static_cast<size_t>(f.tellg()) / 4);
    f.seekg(0); f.read(reinterpret_cast<char*>(out.data()), out.size() * 4); return out;
}
static float Half(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
    uint32_t exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    uint32_t full = 0;
    if (exponent == 0) {
        if (mantissa) { exponent = 113; while ((mantissa & 1024) == 0) { mantissa <<= 1; --exponent; }
                        full = sign | (exponent << 23) | ((mantissa & 1023) << 13); }
        else full = sign;
    } else if (exponent == 31) full = sign | 0x7f800000 | (mantissa << 13);
    else full = sign | ((exponent + 112) << 23) | (mantissa << 13);
    float value; std::memcpy(&value, &full, 4); return value;
}

int main(int argc, char** argv) try {
    using DlssNr::Spatial::Settings;
    using DlssNr::Spatial::Build;
    using DlssNr::Spatial::MakeConstants;
    if (argc != 3) throw std::runtime_error("Pass colour and guide SPIR-V paths");
    const auto colourCode = ReadSpv(argv[1]), guideCode = ReadSpv(argv[2]);
    Settings settings{}; settings.enabled = true;
    const auto layout = Build(settings, 64, 64, 1.0f);
    if (!layout.active || layout.modelW != 58) throw std::runtime_error("Unexpected fixture layout");

    VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance instance {}; Check(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t count = 0; Check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    if (!count) throw std::runtime_error("No Vulkan device");
    std::vector<VkPhysicalDevice> physicals(count); Check(vkEnumeratePhysicalDevices(instance, &count, physicals.data()));
    VkPhysicalDevice physical = physicals.front();
    for (auto candidate : physicals) { VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(candidate, &p);
        if (p.vendorID == 0x10de) physical = candidate; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(physical, &props);
    std::cout << props.deviceName << '\n';
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count); vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    uint32_t family = 0; while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
    if (family == count) throw std::runtime_error("No Vulkan compute queue");
    float priority = 1;
    VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = family;
    qci.queueCount = 1; qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci; // Deliberately no optional storage-image format features.
    VkDevice device {}; Check(vkCreateDevice(physical, &dci, nullptr, &device));
    VkQueue queue {}; vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory{}; vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    auto allocate = [&](VkMemoryRequirements req, VkMemoryPropertyFlags flags) {
        VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size;
        for (; ai.memoryTypeIndex < memory.memoryTypeCount; ++ai.memoryTypeIndex)
            if ((req.memoryTypeBits & (1u << ai.memoryTypeIndex)) &&
                (memory.memoryTypes[ai.memoryTypeIndex].propertyFlags & flags) == flags) break;
        if (ai.memoryTypeIndex == memory.memoryTypeCount) throw std::runtime_error("No compatible memory");
        VkDeviceMemory m{}; Check(vkAllocateMemory(device, &ai, nullptr, &m)); return m;
    };
    struct Buffer { VkBuffer handle{}; VkDeviceMemory memory{}; void* mapped{}; };
    auto makeBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage) {
        Buffer b; VkBufferCreateInfo ci {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = size; ci.usage = usage;
        Check(vkCreateBuffer(device, &ci, nullptr, &b.handle));
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(device, b.handle, &req);
        b.memory = allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        Check(vkBindBufferMemory(device, b.handle, b.memory, 0));
        Check(vkMapMemory(device, b.memory, 0, size, 0, &b.mapped)); return b;
    };
    enum { Colour, Depth, Motion, PackedColour, PackedDepth, PackedMotion, UnpackedProxy, UnpackedAnswer, ImageCount };
    std::array<DlssNr::ImageVk, ImageCount> images{};
    const VkFormat formats[] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
    const uint32_t widths[] = {64, 68, 70, 58, 58, 58, 64, 64};
    const uint32_t heights[] = {64, 68, 70, 58, 58, 58, 64, 64};
    for (int i = 0; i < ImageCount; ++i)
        if (!images[i].Ensure(device, physical, widths[i], heights[i], formats[i]))
            throw std::runtime_error("Image creation failed");

    const DlssNr::GuideRegions regions{{2, 3, 64, 64}, {4, 5, 64, 64}};
    std::array<Buffer, 3> uniform{};
    for (int i = 0; i < 3; ++i) {
        uniform[i] = makeBuffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        const auto c = MakeConstants(layout, i == 0 ? 100 : i == 1 ? 101 : 102,
                                     regions, 1.25f, -.75f, 64, 64);
        std::memcpy(uniform[i].mapped, &c, sizeof(c));
    }
    Buffer depthUpload = makeBuffer(68 * 68 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buffer motionUpload = makeBuffer(70 * 70 * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto depthData = static_cast<float*>(depthUpload.mapped);
    auto motionData = static_cast<float*>(motionUpload.mapped);
    for (int y = 0; y < 68; ++y) for (int x = 0; x < 68; ++x)
        depthData[y * 68 + x] = (x + 100 * y) / 10000.0f;
    for (int y = 0; y < 70; ++y) for (int x = 0; x < 70; ++x) {
        motionData[(y * 70 + x) * 2] = (x - 4) * .125f;
        motionData[(y * 70 + x) * 2 + 1] = -(y - 5) * .0625f;
    }
    Buffer colourRead = makeBuffer(58 * 58 * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer depthRead = makeBuffer(58 * 58 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer motionRead = makeBuffer(58 * 58 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer unpackRead = makeBuffer(64 * 64 * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer answerRead = makeBuffer(64 * 64 * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
    for (uint32_t i = 0; i < 8; ++i)
        bindings[i] = {i, i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : i < 5 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                       i < 7 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
                       1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo slci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 8; slci.pBindings = bindings.data();
    VkDescriptorSetLayout setLayout{}; Check(vkCreateDescriptorSetLayout(device, &slci, nullptr, &setLayout));
    VkPipelineLayoutCreateInfo plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &setLayout;
    VkPipelineLayout pipelineLayout{}; Check(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout));
    VkSamplerCreateInfo sci {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler{}; Check(vkCreateSampler(device, &sci, nullptr, &sampler));
    VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 3}};
    VkDescriptorPoolCreateInfo dpci {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 3; dpci.poolSizeCount = 4; dpci.pPoolSizes = poolSizes;
    VkDescriptorPool pool{}; Check(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));
    VkDescriptorSetLayout layouts[] = {setLayout, setLayout, setLayout}; VkDescriptorSet sets[3]{};
    VkDescriptorSetAllocateInfo dsai {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool; dsai.descriptorSetCount = 3; dsai.pSetLayouts = layouts;
    Check(vkAllocateDescriptorSets(device, &dsai, sets));
    const int source[][4] = {{Colour, Depth, Motion, Colour}, {Colour, Depth, Motion, Colour},
                             {PackedColour, PackedColour, Motion, Colour}};
    const int target[][2] = {{PackedColour, PackedColour}, {PackedDepth, PackedMotion},
                             {UnpackedProxy, UnpackedAnswer}};
    for (int pass = 0; pass < 3; ++pass) {
        VkDescriptorBufferInfo bi {uniform[pass].handle, 0, 256};
        std::array<VkDescriptorImageInfo, 7> ii{};
        std::array<VkWriteDescriptorSet, 8> writes{};
        for (uint32_t binding = 0; binding < 8; ++binding) {
            auto& w = writes[binding]; w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = sets[pass]; w.dstBinding = binding; w.descriptorCount = 1;
            w.descriptorType = bindings[binding].descriptorType;
            if (!binding) w.pBufferInfo = &bi;
            else {
                const int image = binding <= 4 ? source[pass][binding - 1] :
                                  binding <= 6 ? target[pass][binding - 5] : Colour;
                ii[binding - 1] = {sampler, images[image].info.ImageView, VK_IMAGE_LAYOUT_GENERAL};
                w.pImageInfo = &ii[binding - 1];
            }
        }
        vkUpdateDescriptorSets(device, 8, writes.data(), 0, nullptr);
    }
    VkShaderModule modules[2]{}; VkPipeline pipelines[2]{};
    const std::vector<uint32_t>* codes[] = {&colourCode, &guideCode};
    for (int i = 0; i < 2; ++i) {
        VkShaderModuleCreateInfo ci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = codes[i]->size() * 4; ci.pCode = codes[i]->data();
        Check(vkCreateShaderModule(device, &ci, nullptr, &modules[i]));
        VkComputePipelineCreateInfo pi {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pi.layout = pipelineLayout;
        pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                    VK_SHADER_STAGE_COMPUTE_BIT, modules[i], "CSMain", nullptr};
        Check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipelines[i]));
    }
    VkCommandPoolCreateInfo cpci {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = family;
    VkCommandPool commandPool{}; Check(vkCreateCommandPool(device, &cpci, nullptr, &commandPool));
    VkCommandBufferAllocateInfo cai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = commandPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer command{}; Check(vkAllocateCommandBuffers(device, &cai, &command));
    VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; Check(vkBeginCommandBuffer(command, &begin));
    const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkClearColorValue colour {{.25f, .5f, .75f, 1.0f}}, sentinel {{7, 7, 7, 7}};
    for (int i = 0; i < ImageCount; ++i) {
        VkImageMemoryBarrier b {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = images[i].info.Image; b.subresourceRange = range;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        vkCmdClearColorImage(command, images[i].info.Image, VK_IMAGE_LAYOUT_GENERAL,
                             i == Colour ? &colour : &sentinel, 1, &range);
    }
    auto uploadImage = [&](const Buffer& b, int i) {
        VkBufferImageCopy c{}; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {widths[i], heights[i], 1};
        vkCmdCopyBufferToImage(command, b.handle, images[i].info.Image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    };
    uploadImage(depthUpload, Depth); uploadImage(motionUpload, Motion);
    auto barrier = [&](VkAccessFlags before, VkAccessFlags after) {
        VkMemoryBarrier b {VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask = before; b.dstAccessMask = after;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &b, 0, nullptr, 0, nullptr);
    };
    barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    for (int pass = 0; pass < 3; ++pass) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[pass == 1 ? 1 : 0]);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &sets[pass], 0, nullptr);
        const uint32_t w = pass == 2 ? 64 : 58;
        vkCmdDispatch(command, (w + 7) / 8, (w + 7) / 8, 1);
        barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    auto copyImage = [&](int i, const Buffer& b) {
        VkBufferImageCopy c{}; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {widths[i], heights[i], 1};
        vkCmdCopyImageToBuffer(command, images[i].info.Image, VK_IMAGE_LAYOUT_GENERAL, b.handle, 1, &c);
    };
    copyImage(PackedColour, colourRead); copyImage(PackedDepth, depthRead);
    copyImage(PackedMotion, motionRead); copyImage(UnpackedProxy, unpackRead);
    copyImage(UnpackedAnswer, answerRead);
    barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    Check(vkEndCommandBuffer(command));
    VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    Check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE)); Check(vkQueueWaitIdle(queue));

    const auto packedColour = static_cast<const uint16_t*>(colourRead.mapped);
    const auto packedDepth = static_cast<const float*>(depthRead.mapped);
    const auto packedMotion = static_cast<const float*>(motionRead.mapped);
    const auto unpacked = static_cast<const uint16_t*>(unpackRead.mapped);
    const auto answer = static_cast<const uint16_t*>(answerRead.mapped);
    for (int y : {0, 29, 57}) for (int x : {0, 29, 57}) {
        const size_t i = static_cast<size_t>(y) * 58 + x;
        Near(Half(packedColour[i * 4]), .25f);
        Near(Half(packedColour[i * 4 + 1]), .5f);
        const float nx = pw::Unpack(x + .5f, layout.warp.x);
        const float ny = pw::Unpack(y + .5f, layout.warp.y);
        const int dx = std::clamp(static_cast<int>(regions.depth.x + nx), 2, 65);
        const int dy = std::clamp(static_cast<int>(regions.depth.y + ny), 3, 66);
        Near(packedDepth[i], (dx + 100 * dy) / 10000.0f);
        const int mx = std::clamp(static_cast<int>(regions.motion.x + nx), 4, 67);
        const int my = std::clamp(static_cast<int>(regions.motion.y + ny), 5, 68);
        const float motionX = (mx - 4) * .125f * 1.25f;
        const float motionY = -(my - 5) * .0625f * -.75f;
        Near(packedMotion[i * 4], pw::Pack(nx + motionX, layout.warp.x) - pw::Pack(nx, layout.warp.x), .01f);
        Near(packedMotion[i * 4 + 1], pw::Pack(ny + motionY, layout.warp.y) - pw::Pack(ny, layout.warp.y), .01f);
        Near(packedMotion[i * 4 + 2], 0); Near(packedMotion[i * 4 + 3], 0);
    }
    for (int y : {0, 31, 63}) for (int x : {0, 31, 63}) {
        const size_t i = static_cast<size_t>(y) * 64 + x;
        Near(Half(unpacked[i * 4]), .25f); Near(Half(unpacked[i * 4 + 1]), .5f);
        Near(Half(answer[i * 4]), .25f); Near(Half(answer[i * 4 + 1]), .5f);
    }
    std::cout << "PASS: Vulkan spatial colour, typed depth/motion, guide rects, motion endpoints, unpack\n";
    for (auto& image : images) image.Destroy(device);
    for (auto* b : {&depthUpload, &motionUpload, &colourRead, &depthRead, &motionRead, &unpackRead, &answerRead,
                    &uniform[0], &uniform[1], &uniform[2]}) {
        vkUnmapMemory(device, b->memory); vkDestroyBuffer(device, b->handle, nullptr); vkFreeMemory(device, b->memory, nullptr);
    }
    vkDestroyCommandPool(device, commandPool, nullptr);
    for (int i = 0; i < 2; ++i) { vkDestroyPipeline(device, pipelines[i], nullptr); vkDestroyShaderModule(device, modules[i], nullptr); }
    vkDestroyDescriptorPool(device, pool, nullptr); vkDestroySampler(device, sampler, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr); vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    return 0;
} catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
