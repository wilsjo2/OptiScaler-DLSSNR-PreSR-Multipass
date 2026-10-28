// Runs the production SPIR-V encode/resolve on a real Vulkan device. No game or NR DLL needed.
// cl /std:c++20 /EHsc /IOptiScaler /Iexternal/vulkan/include /Iexternal/nvngx_dlss_sdk
//    /Iexternal/spdlog/include tests/nr_vulkan_shader_smoke.cpp
//    /link OptiScaler/library/vulkan/vulkan-1.lib
#include <vulkan/vulkan.h>
#include "../OptiScaler/dlssnr/DlssNr_Image_Vk.h"
#include "../OptiScaler/shaders/dlssnr/DlssNr_Common.h"
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cmath>

static void check(VkResult r) { if (r != VK_SUCCESS) throw std::runtime_error("Vulkan result " + std::to_string(r)); }
int main(int argc, char** argv)
try
{
    if (argc != 2) throw std::runtime_error("Pass the production DlssNr_Shader_Vk.spv path");
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open SPIR-V");
    const auto bytes = (size_t) file.tellg();
    if (!bytes || bytes % 4) throw std::runtime_error("Invalid SPIR-V length");
    std::vector<uint32_t> code(bytes / 4);
    file.seekg(0); file.read((char*) code.data(), bytes);
    VkApplicationInfo app { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "NR Vulkan shader smoke"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO }; ici.pApplicationInfo = &app;
    VkInstance instance {}; check(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t count = 0; check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    if (!count) throw std::runtime_error("No Vulkan device");
    VkPhysicalDevice pd = devices.front();
    for (auto d : devices) { VkPhysicalDeviceProperties p {}; vkGetPhysicalDeviceProperties(d, &p); if (p.vendorID == 0x10de) pd = d; }
    VkPhysicalDeviceProperties props {}; vkGetPhysicalDeviceProperties(pd, &props);
    std::cout << props.deviceName << '\n';
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count); vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, families.data());
    uint32_t family = 0;
    while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
    if (family == count) throw std::runtime_error("No compute queue");
    float priority = 1;
    VkDeviceQueueCreateInfo qci { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures available {}; vkGetPhysicalDeviceFeatures(pd, &available);
    VkPhysicalDeviceFeatures enabled {};
    enabled.shaderStorageImageReadWithoutFormat = available.shaderStorageImageReadWithoutFormat;
    enabled.shaderStorageImageWriteWithoutFormat = available.shaderStorageImageWriteWithoutFormat;
    VkDeviceCreateInfo dci { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pEnabledFeatures = &enabled;
    VkDevice device {}; check(vkCreateDevice(pd, &dci, nullptr, &device));
    VkQueue queue {}; vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory {}; vkGetPhysicalDeviceMemoryProperties(pd, &memory);
    auto allocate = [&](VkMemoryRequirements req, VkMemoryPropertyFlags flags) {
        VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; ai.allocationSize = req.size;
        for (ai.memoryTypeIndex = 0; ai.memoryTypeIndex < memory.memoryTypeCount; ++ai.memoryTypeIndex)
            if ((req.memoryTypeBits & (1u << ai.memoryTypeIndex)) &&
                (memory.memoryTypes[ai.memoryTypeIndex].propertyFlags & flags) == flags) break;
        if (ai.memoryTypeIndex == memory.memoryTypeCount) throw std::runtime_error("No suitable memory type");
        VkDeviceMemory m {}; check(vkAllocateMemory(device, &ai, nullptr, &m)); return m;
    };
    // The active rectangle is an odd-sized Balanced input inside a padded allocation.
    constexpr uint32_t width = 1507, height = 847, allocationWidth = 1536, allocationHeight = 864;
    std::array<DlssNr::ImageVk, 6> images {}; // source, proxy, keep, model, result, dummy
    for (auto& image : images)
    {
        if (!image.Ensure(device, pd, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT))
            throw std::runtime_error("NR placeholder allocation failed");
        for (auto format : { VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_FORMAT_R32G32B32A32_SFLOAT })
        {
            if (!image.Ensure(device, pd, allocationWidth, allocationHeight, format) ||
                image.info.Width != allocationWidth || image.info.Height != allocationHeight ||
                image.info.Format != format || image.layout != VK_IMAGE_LAYOUT_UNDEFINED)
                throw std::runtime_error("NR image allocation/format replacement failed");
            const auto handle = image.info.Image;
            if (!image.Ensure(device, pd, allocationWidth, allocationHeight, format) || image.info.Image != handle)
                throw std::runtime_error("Unchanged NR image was reallocated");
        }
    }
    struct Buffer { VkBuffer buffer {}; VkDeviceMemory memory {}; void* mapped {}; };
    auto buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage) {
        Buffer b; VkBufferCreateInfo ci { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO }; ci.size = size; ci.usage = usage;
        check(vkCreateBuffer(device, &ci, nullptr, &b.buffer));
        VkMemoryRequirements req {}; vkGetBufferMemoryRequirements(device, b.buffer, &req);
        b.memory = allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, b.buffer, b.memory, 0));
        check(vkMapMemory(device, b.memory, 0, size, 0, &b.mapped)); return b;
    };
    auto readback = buffer((VkDeviceSize) allocationWidth * allocationHeight * 16,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::array<Buffer, 2> uniforms { buffer(sizeof(DlssNrConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
                                   buffer(sizeof(DlssNrConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) };
    DlssNrConstants constants {}; constants.Width = width; constants.Height = height;
    constants.Passthrough = 1; constants.ApplyModel = 0; constants.WhitePoint = 1;
    std::memcpy(uniforms[0].mapped, &constants, sizeof(constants));
    constants.Mode = DlssNrMode_Resolve; std::memcpy(uniforms[1].mapped, &constants, sizeof(constants));
    std::array<VkDescriptorSetLayoutBinding, 8> bindings {};
    for (uint32_t i = 0; i < 8; ++i)
        bindings[i] = { i, i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER :
                          i < 5 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                          i < 7 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
                        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo lci { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 8; lci.pBindings = bindings.data();
    VkDescriptorSetLayout layout {}; check(vkCreateDescriptorSetLayout(device, &lci, nullptr, &layout));
    VkPipelineLayoutCreateInfo plci { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &layout;
    VkPipelineLayout pipelineLayout {}; check(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout));
    VkShaderModuleCreateInfo sci { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO }; sci.codeSize = bytes; sci.pCode = code.data();
    VkShaderModule shader {}; check(vkCreateShaderModule(device, &sci, nullptr, &shader));
    VkComputePipelineCreateInfo pci { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO }; pci.layout = pipelineLayout;
    pci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "CSMain", nullptr };
    VkPipeline pipeline {}; check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));
    VkSamplerCreateInfo si { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO }; si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler {}; check(vkCreateSampler(device, &si, nullptr, &sampler));
    VkDescriptorPoolSize sizes[] = { {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2} };
    VkDescriptorPoolCreateInfo dpci { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO }; dpci.maxSets = 2; dpci.poolSizeCount = 4; dpci.pPoolSizes = sizes;
    VkDescriptorPool pool {}; check(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));
    VkDescriptorSetLayout layouts[] = { layout, layout }; VkDescriptorSet sets[2] {};
    VkDescriptorSetAllocateInfo dai { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO }; dai.descriptorPool = pool; dai.descriptorSetCount = 2; dai.pSetLayouts = layouts;
    check(vkAllocateDescriptorSets(device, &dai, sets));
    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        // Production binding order: Source, Model, Original, Motion, Target, Keep, Sampler.
        const uint32_t encode[] = { 0, 3, 2, 5, 1, 2, 5 }, resolve[] = { 1, 3, 2, 5, 4, 5, 5 };
        VkDescriptorBufferInfo bi { uniforms[pass].buffer, 0, sizeof(DlssNrConstants) };
        std::array<VkDescriptorImageInfo, 7> ii {}; std::array<VkWriteDescriptorSet, 8> writes {};
        for (uint32_t b = 0; b < 8; ++b)
        {
            writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; writes[b].dstSet = sets[pass];
            writes[b].dstBinding = b; writes[b].descriptorCount = 1; writes[b].descriptorType = bindings[b].descriptorType;
            if (!b) writes[b].pBufferInfo = &bi;
            else { ii[b - 1] = { sampler, images[(pass ? resolve : encode)[b - 1]].info.ImageView, VK_IMAGE_LAYOUT_GENERAL }; writes[b].pImageInfo = &ii[b - 1]; }
        }
        vkUpdateDescriptorSets(device, 8, writes.data(), 0, nullptr);
    }
    VkCommandPoolCreateInfo cpci { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO }; cpci.queueFamilyIndex = family;
    VkCommandPool commandPool {}; check(vkCreateCommandPool(device, &cpci, nullptr, &commandPool));
    VkCommandBufferAllocateInfo cai { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = commandPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd {}; check(vkAllocateCommandBuffers(device, &cai, &cmd));
    VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; check(vkBeginCommandBuffer(cmd, &begin));
    const VkImageSubresourceRange range { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    const VkClearColorValue color { { 0.25f, 0.5f, 0.75f, 0.375f } }, sentinel { { 7, 7, 7, 7 } };
    for (uint32_t i = 0; i < images.size(); ++i)
    {
        VkImageMemoryBarrier barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL; barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = images[i].info.Image; barrier.subresourceRange = range;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        vkCmdClearColorImage(cmd, images[i].info.Image, VK_IMAGE_LAYOUT_GENERAL, i == 0 ? &color : &sentinel, 1, &range);
    }
    auto memoryBarrier = [&](VkAccessFlags src, VkAccessFlags dst) {
        VkMemoryBarrier b { VK_STRUCTURE_TYPE_MEMORY_BARRIER }; b.srcAccessMask = src; b.dstAccessMask = dst;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    };
    memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    for (auto set : sets)
    {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
        memoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    }
    VkBufferImageCopy copy {}; copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { allocationWidth, allocationHeight, 1 };
    vkCmdCopyImageToBuffer(cmd, images[4].info.Image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &copy);
    memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    check(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit { VK_STRUCTURE_TYPE_SUBMIT_INFO }; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
    check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE)); check(vkQueueWaitIdle(queue)); // standalone test only
    auto pixels = (const float*) readback.mapped;
    for (uint32_t y = 0; y < allocationHeight; ++y)
        for (uint32_t x = 0; x < allocationWidth; ++x)
            for (uint32_t c = 0; c < 4; ++c)
            {
                const float expected = x < width && y < height ? color.float32[c] : sentinel.float32[c];
                const float got = pixels[((size_t) y * allocationWidth + x) * 4 + c];
                if (!std::isfinite(got) || std::abs(got - expected) > 0.002f)
                    throw std::runtime_error("Pixel mismatch at " + std::to_string(x) + "," + std::to_string(y) + " got " + std::to_string(got));
            }
    std::cout << "PASS: production Vulkan encode/resolve, 1507x847 active inside 1536x864, RGBA preserved, padding untouched\n";
    // Reuse two unchanged bindings for a 30-pass clamp chain, as the production model loop does.
    constants.Mode = DlssNrMode_ClampProxy;
    for (auto& uniform : uniforms) std::memcpy(uniform.mapped, &constants, sizeof(constants));
    for (uint32_t pass=0; pass<2; ++pass)
    {
        VkDescriptorImageInfo source {sampler, images[pass ? 1 : 4].info.ImageView, VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo target {sampler, images[pass ? 4 : 1].info.ImageView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[2] {};
        for (auto& write : writes) { write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; write.dstSet=sets[pass]; write.descriptorCount=1; }
        writes[0].dstBinding=1; writes[0].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[0].pImageInfo=&source;
        writes[1].dstBinding=5; writes[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo=&target;
        vkUpdateDescriptorSets(device,2,writes,0,nullptr);
    }
    const VkClearColorValue inputs[] = {{{-0.2f,1.2f,0.3f,0.25f}}, {{INFINITY,-INFINITY,NAN,0.75f}}};
    const VkClearColorValue expected[] = {{{0,1,0.3f,0.25f}}, {{0.5f,0.5f,0.5f,0.75f}}};
    for (uint32_t sample=0; sample<2; ++sample)
    {
        check(vkResetCommandPool(device,commandPool,0));
        check(vkBeginCommandBuffer(cmd,&begin));
        vkCmdClearColorImage(cmd,images[4].info.Image,VK_IMAGE_LAYOUT_GENERAL,&inputs[sample],1,&range);
        memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
        for (uint32_t pass=0; pass<30; ++pass)
        {
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipelineLayout,0,1,&sets[pass%2],0,nullptr);
            vkCmdDispatch(cmd,(width+7)/8,(height+7)/8,1);
            memoryBarrier(VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        }
        vkCmdCopyImageToBuffer(cmd,images[4].info.Image,VK_IMAGE_LAYOUT_GENERAL,readback.buffer,1,&copy);
        memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
        check(vkEndCommandBuffer(cmd));
        check(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE)); check(vkQueueWaitIdle(queue));
        for (uint32_t y=0; y<height; ++y)
            for (uint32_t x=0; x<width; ++x)
                for (uint32_t c=0; c<4; ++c)
                {
                    const float got=pixels[((size_t)y*allocationWidth+x)*4+c];
                    if (!std::isfinite(got) || std::abs(got-expected[sample].float32[c])>0.0001f)
                        throw std::runtime_error("Vulkan interpass clamp mismatch");
                }
    }
    std::cout << "PASS: production Vulkan 30-pass clamp chain, finite RGB, bounded range, identity and alpha\n";
    // Execute the automatic exposure path with the production SPIR-V and its actual binding layout.
    constants = {};
    constants.Mode = DlssNrMode_Meter;
    constants.Width = constants.Height = 64;
    std::memcpy(uniforms[0].mapped, &constants, sizeof(constants));
    constants.Mode = DlssNrMode_AutoExposure;
    constants.Width = constants.Height = 1;
    constants.PreExposure = 8;
    constants.ExposureSourceWidth = allocationWidth;
    constants.ExposureSourceHeight = allocationHeight;
    std::memcpy(uniforms[1].mapped, &constants, sizeof(constants));
    for (unsigned pass = 0; pass < 2; ++pass)
    {
        VkDescriptorImageInfo source { sampler, images[pass ? 1 : 0].info.ImageView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo target { sampler, images[pass ? 4 : 1].info.ImageView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet writes[2] {};
        for (unsigned n = 0; n < 2; ++n)
        {
            writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[n].dstSet = sets[pass];
            writes[n].dstBinding = n ? 5 : 1;
            writes[n].descriptorCount = 1;
            writes[n].descriptorType = n ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[n].pImageInfo = n ? &target : &source;
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }
    check(vkResetCommandPool(device, commandPool, 0));
    check(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    for (unsigned pass = 0; pass < 2; ++pass)
    {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &sets[pass], 0, nullptr);
        vkCmdDispatch(cmd, pass ? 1 : 64, pass ? 1 : 64, 1);
        memoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    }
    vkCmdCopyImageToBuffer(cmd, images[4].info.Image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer, 1, &copy);
    memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    check(vkEndCommandBuffer(cmd));
    check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(queue));
    const float white = (0.25f * 0.2126f + 0.5f * 0.7152f + 0.75f * 0.0722f) * 0.82f / 0.18f;
    if (!std::isfinite(pixels[0]) || std::abs(pixels[0] - white) > 0.001f)
        throw std::runtime_error("Vulkan automatic exposure mismatch");
    std::cout << "PASS: production Vulkan automatic exposure and pre-exposure normalization\n";
    // A dark model rim beside a bright pixel exercises the new mode and trailing constants.
    constants={}; constants.Mode=DlssNrMode_SuppressHalo; constants.Width=32; constants.Height=1;
    constants.HaloSuppression=constants.HaloBrightSuppression=1; constants.HaloRadius=1;
    std::memcpy(uniforms[1].mapped,&constants,sizeof(constants));
    VkDescriptorImageInfo haloSource {sampler,images[0].info.ImageView,VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet haloWrite {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    haloWrite.dstSet=sets[1]; haloWrite.dstBinding=1; haloWrite.descriptorCount=1;
    haloWrite.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; haloWrite.pImageInfo=&haloSource;
    vkUpdateDescriptorSets(device,1,&haloWrite,0,nullptr);
    const float upload[12]={.8f,.8f,.8f,.4f, .7f,.7f,.7f,.4f, .02f,.02f,.02f,.4f};
    std::memcpy(readback.mapped,upload,sizeof(upload));
    const VkClearColorValue haloBase {{.04f,.04f,.04f,.4f}}, haloBroad {{.032f,.032f,.032f,.4f}};
    check(vkResetCommandPool(device,commandPool,0)); check(vkBeginCommandBuffer(cmd,&begin));
    vkCmdClearColorImage(cmd,images[0].info.Image,VK_IMAGE_LAYOUT_GENERAL,&haloBase,1,&range);
    vkCmdClearColorImage(cmd,images[3].info.Image,VK_IMAGE_LAYOUT_GENERAL,&haloBroad,1,&range);
    vkCmdClearColorImage(cmd,images[4].info.Image,VK_IMAGE_LAYOUT_GENERAL,&sentinel,1,&range);
    memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
    for (unsigned i=0; i<3; ++i)
    {
        VkBufferImageCopy point {}; point.bufferOffset=i*16;
        point.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
        point.imageOffset={i==2 ? 15 : 16,0,0}; point.imageExtent={1,1,1};
        vkCmdCopyBufferToImage(cmd,readback.buffer,images[i==0 ? 0 : 3].info.Image,VK_IMAGE_LAYOUT_GENERAL,1,&point);
    }
    memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipelineLayout,0,1,&sets[1],0,nullptr);
    vkCmdDispatch(cmd,4,1,1);
    memoryBarrier(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdCopyImageToBuffer(cmd,images[4].info.Image,VK_IMAGE_LAYOUT_GENERAL,readback.buffer,1,&copy);
    memoryBarrier(VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
    check(vkEndCommandBuffer(cmd)); check(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE)); check(vkQueueWaitIdle(queue));
    const float rim=pixels[15*4];
    if (!std::isfinite(rim) || rim<=.0205f || rim>.0321f ||
        std::abs(pixels[15*4+3]-.4f)>.0001f || std::abs(pixels[0]-.032f)>.0001f || pixels[32*4]!=7)
        throw std::runtime_error("Vulkan halo correction, alpha or bounds mismatch");
    std::cout << "PASS: production Vulkan halo correction lifts the rim and preserves broad shading, alpha and bounds\n";
    vkDestroyCommandPool(device, commandPool, nullptr); vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroySampler(device, sampler, nullptr); vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shader, nullptr); vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
    for (auto& image : images)
    {
        image.Destroy(device);
        if (image.Valid() || image.info.Image || image.memory || image.layout != VK_IMAGE_LAYOUT_UNDEFINED)
            throw std::runtime_error("NR image destruction left stale metadata");
    }
    for (auto b : { uniforms[0], uniforms[1], readback }) { vkUnmapMemory(device, b.memory); vkDestroyBuffer(device, b.buffer, nullptr); vkFreeMemory(device, b.memory, nullptr); }
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    return 0;
}
catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
