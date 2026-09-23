#include "pch.h"

#include "DlssNr_Vk.h"
#include <dlssnr/DlssNrFeature_Vk.h>

#include "precompile/DlssNr_Shader_Vk.h"
#include "precompile/dlssnr_finished_color_Shader_Vk.h"
#include "precompile/dlssnr_spatial_Shader_Vk.h"
#include "precompile/dlssnr_spatial_guides_Shader_Vk.h"
#include <dlssnr/DlssNrFinished_Vk.h>

#include <algorithm>
#include <cstring>

DlssNr_Vk::DlssNr_Vk(std::string InName, VkDevice InDevice, VkPhysicalDevice InPhysicalDevice)
    : Shader_Vk(InName, InDevice, InPhysicalDevice)
{
    if (InDevice == VK_NULL_HANDLE || InPhysicalDevice == VK_NULL_HANDLE)
    {
        LOG_ERROR("DLSS-NR Vulkan pass: no device");
        _init = false;
        return;
    }

    _maxFramesInFlight = kFramesInFlight;

    // Linear, because the resolve reads the model's answer at a different size than it writes -- the
    // model may have run at a reduced resolution and the edit has to be stretched back over the frame.
    // The D3D12 pass uses a linear sampler for the same reason and the two must agree.
    CreateSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // The constant ring. A uniform buffer binding can be offset into, but only to a multiple of the
    // device's own alignment, so the stride is the struct rounded up rather than the struct itself.
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(_physicalDevice, &props);

    const VkDeviceSize alignment = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 1);
    _slotStride = ((sizeof(DlssNrConstants) + alignment - 1) / alignment) * alignment;

    if (!CreateBufferResource(_device, _physicalDevice, &_constantBuffer, &_constantBufferMemory, _slotStride * kSlots,
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
    {
        LOG_ERROR("DLSS-NR Vulkan pass: could not allocate the constant ring");
        _init = false;
        return;
    }

    if (vkMapMemory(_device, _constantBufferMemory, 0, _slotStride * kSlots, 0, &_mappedConstantBuffer) != VK_SUCCESS)
    {
        LOG_ERROR("DLSS-NR Vulkan pass: could not map the constant ring");
        _init = false;
        return;
    }

    // The layout mirrors the [[vk::binding]] numbers in dlssnr.hlsl, entry for entry. Combined image
    // samplers for the reads: the shader declares its sampler separately, and a combined descriptor
    // satisfies a separately declared sampled image with the sampler half simply unused.
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        CreateBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),         // Params
        CreateBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gSource
        CreateBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gModel
        CreateBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gOriginal
        CreateBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), // gMotion
        CreateBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),          // gTarget
        CreateBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),          // gKeep
        CreateBinding(7, VK_DESCRIPTOR_TYPE_SAMPLER),                // gLinear
    };

    CreateLayouts(bindings);

    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSlots },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 * kSlots },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kSlots },
        { VK_DESCRIPTOR_TYPE_SAMPLER, kSlots },
    };

    CreateDescriptorPool(poolSizes, kSlots);

    // One set per slot rather than per frame: two dispatches in the same frame need two sets, or the
    // second overwrites bindings the first has not consumed yet.
    _maxFramesInFlight = kSlots;
    CreateDescriptorSets(_descriptorSetLayout, _descriptorPool, _descriptorSets);
    _maxFramesInFlight = kFramesInFlight;

    if (_descriptorSets.size() < kSlots)
    {
        LOG_ERROR("DLSS-NR Vulkan pass: expected {} descriptor sets, got {}", kSlots, _descriptorSets.size());
        _init = false;
        return;
    }

    std::vector<char> shaderCode(dlssnr_spv, dlssnr_spv + sizeof(dlssnr_spv));

    if (!CreateComputePipeline(_device, _pipelineLayout, &_pipeline, shaderCode))
    {
        LOG_ERROR("DLSS-NR Vulkan pass: could not create the compute pipeline");
        _init = false;
        return;
    }

    std::vector<char> finishedCode(dlssnr_finished_color_spv,
                                   dlssnr_finished_color_spv + sizeof(dlssnr_finished_color_spv));
    CreateComputePipeline(_device, _pipelineLayout, &_finishedPipeline, finishedCode);
    std::vector<char> spatialCode(dlssnr_spatial_spv, dlssnr_spatial_spv + sizeof(dlssnr_spatial_spv));
    std::vector<char> spatialGuidesCode(dlssnr_spatial_guides_spv,
                                        dlssnr_spatial_guides_spv + sizeof(dlssnr_spatial_guides_spv));
    if (!CreateComputePipeline(_device, _pipelineLayout, &_spatialPipeline, spatialCode) ||
        !CreateComputePipeline(_device, _pipelineLayout, &_spatialGuidesPipeline, spatialGuidesCode))
        LOG_WARN("DLSS-NR Vulkan: spatial compression shaders unavailable; ordinary NR remains available");
    _init = true;
    LOG_INFO("DLSS-NR Vulkan pass up: {} constant slots, stride {}", kSlots, (uint64_t) _slotStride);
}

DlssNr_Vk::~DlssNr_Vk()
{
    if (State::Instance().isShuttingDown)
    {
        _finished.release();
        _model.release();
        return;
    }
    _finished.reset();
    _model.reset();
    if (_finishedPipeline)
        vkDestroyPipeline(_device, _finishedPipeline, nullptr);
    if (_spatialPipeline)
        vkDestroyPipeline(_device, _spatialPipeline, nullptr);
    if (_spatialGuidesPipeline)
        vkDestroyPipeline(_device, _spatialGuidesPipeline, nullptr);
    _dummy.Destroy(_device);
}

// All declared descriptors must be valid, even when the selected mode does not read them.
bool DlssNr_Vk::CreateDummy(VkCommandBuffer cmdList)
{
    if (_dummy.layout == VK_IMAGE_LAYOUT_GENERAL)
        return true;
    if (!_dummy.Ensure(_device, _physicalDevice, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT))
    {
        LOG_ERROR("DLSS-NR Vulkan pass: could not allocate the placeholder image");
        return false;
    }
    SetImageLayout(cmdList, _dummy.info.Image, _dummy.layout, VK_IMAGE_LAYOUT_GENERAL, _dummy.info.SubresourceRange);
    _dummy.layout = VK_IMAGE_LAYOUT_GENERAL;
    return true;
}

void DlssNr_Vk::WriteDescriptors(VkDescriptorSet set, VkDeviceSize constantOffset, VkImageView source,
                                 VkImageView model, VkImageView original, VkImageView motion, VkImageView target,
                                 VkImageView keep, VkImageLayout sourceLayout, VkImageLayout motionLayout,
                                 VkImageLayout modelLayout, VkImageLayout originalLayout)
{
    VkDescriptorBufferInfo bufferInfo { _constantBuffer, constantOffset, sizeof(DlssNrConstants) };

    // A read slot standing in for nothing is bound in GENERAL, which is the layout the placeholder is
    // left in. A real read is bound in the layout the caller says its image is in.
    const auto readInfo = [&](VkImageView v, VkImageLayout layout)
    {
        return VkDescriptorImageInfo { _textureSampler, v != VK_NULL_HANDLE ? v : _dummy.info.ImageView,
                                       v != VK_NULL_HANDLE ? layout : VK_IMAGE_LAYOUT_GENERAL };
    };

    const auto writeInfo = [&](VkImageView v)
    {
        return VkDescriptorImageInfo { VK_NULL_HANDLE, v != VK_NULL_HANDLE ? v : _dummy.info.ImageView,
                                       VK_IMAGE_LAYOUT_GENERAL };
    };

    VkDescriptorImageInfo sourceInfo = readInfo(source, sourceLayout);
    VkDescriptorImageInfo modelInfo = readInfo(model, modelLayout);
    VkDescriptorImageInfo originalInfo = readInfo(original, originalLayout);
    VkDescriptorImageInfo motionInfo = readInfo(motion, motionLayout);
    VkDescriptorImageInfo targetInfo = writeInfo(target);
    VkDescriptorImageInfo keepInfo = writeInfo(keep);
    VkDescriptorImageInfo samplerInfo { _textureSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };

    const VkWriteDescriptorSet writes[] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr,
          &bufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          &sourceInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          &modelInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          &originalInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          &motionInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &targetInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &keepInfo,
          nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 7, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &samplerInfo,
          nullptr, nullptr },
    };

    vkUpdateDescriptorSets(_device, (uint32_t) (sizeof(writes) / sizeof(writes[0])), writes, 0, nullptr);
}

bool DlssNr_Vk::Dispatch(VkCommandBuffer InCmdList, const DlssNrConstants& InConstants, uint32_t InThreadsX,
                         uint32_t InThreadsY, VkImageView InSource, VkImageView InModel, VkImageView InOriginal,
                         VkImageView InMotion, VkImageView InTarget, VkImageView InKeep, VkImageLayout InSourceLayout,
                         VkImageLayout InMotionLayout, bool finishedColor, uint32_t* immutableSlot)
{
    if (!CanRender() || InCmdList == VK_NULL_HANDLE || (finishedColor && !_finishedPipeline))
        return false;

    if (InTarget == VK_NULL_HANDLE)
    {
        LOG_ERROR("DLSS-NR Vulkan pass: a dispatch with nothing to write");
        return false;
    }

    if (!CreateDummy(InCmdList))
        return false;

    const bool reuse = immutableSlot && *immutableSlot != UINT32_MAX;
    const uint32_t slot = reuse ? *immutableSlot : _slot;
    if (!reuse)
    {
        _slot = (_slot + 1) % kSlots;
        const VkDeviceSize offset = _slotStride * slot;
        std::memcpy((char*) _mappedConstantBuffer + offset, &InConstants, sizeof(DlssNrConstants));

        WriteDescriptors(_descriptorSets[slot], offset, InSource, InModel, InOriginal, InMotion, InTarget, InKeep,
                         InSourceLayout, InMotionLayout);
        if (immutableSlot)
            *immutableSlot = slot;
    }

    vkCmdBindPipeline(InCmdList, VK_PIPELINE_BIND_POINT_COMPUTE, finishedColor ? _finishedPipeline : _pipeline);
    vkCmdBindDescriptorSets(InCmdList, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &_descriptorSets[slot], 0,
                            nullptr);

    // The shader's thread group is 8x8, the same as the D3D12 path.
    const uint32_t groupsX = InConstants.Mode == DlssNrMode_Meter ? InThreadsX : (InThreadsX + 7) / 8;
    const uint32_t groupsY = InConstants.Mode == DlssNrMode_Meter ? InThreadsY : (InThreadsY + 7) / 8;

    vkCmdDispatch(InCmdList, groupsX, groupsY, 1);

    // What this dispatch wrote, the next one reads. Left to the caller and it is a race that shows up
    // as a frame of stale detail rather than as an error, which is the worst kind to chase.
    VkMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(InCmdList, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);

    return true;
}

bool DlssNr_Vk::DispatchSpatial(VkCommandBuffer cmd, const DlssNr::Spatial::Constants& constants, VkImageView source,
                                VkImageView second, VkImageView third, VkImageView target, VkImageView keep,
                                VkImageLayout sourceLayout, VkImageLayout secondLayout, VkImageLayout thirdLayout)
{
    static_assert(sizeof(DlssNr::Spatial::Constants) == sizeof(DlssNrConstants));
    if (!SpatialReady() || cmd == VK_NULL_HANDLE || target == VK_NULL_HANDLE || !CreateDummy(cmd))
        return false;

    const uint32_t slot = _slot;
    _slot = (_slot + 1) % kSlots;
    const VkDeviceSize offset = _slotStride * slot;
    std::memcpy((char*) _mappedConstantBuffer + offset, &constants, sizeof(constants));
    WriteDescriptors(_descriptorSets[slot], offset, source, second, third, VK_NULL_HANDLE, target, keep, sourceLayout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, secondLayout, thirdLayout);

    const VkPipeline pipeline = constants.mode == 101 ? _spatialGuidesPipeline : _spatialPipeline;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &_descriptorSets[slot], 0,
                            nullptr);
    vkCmdDispatch(cmd, (constants.width + 7) / 8, (constants.height + 7) / 8, 1);

    VkMemoryBarrier barrier { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
    return true;
}

VkImageInfo DlssNr_Vk::PrepareInput(VkCommandBuffer cmd, const VkImageInfo& nextOutput)
{
    if (!nextOutput.Image || !nextOutput.Width || !nextOutput.Height)
        return {};
    const bool resized = nextOutput.Width != _width || nextOutput.Height != _height || nextOutput.Format != _format;
    if (resized && GetImage() != VK_NULL_HANDLE && vkDeviceWaitIdle(_device) != VK_SUCCESS)
    {
        LOG_ERROR("DLSS-NR Vulkan: device did not retire the previous intermediate");
        return {};
    }
    if (!CreateImageResource(nextOutput.Width, nextOutput.Height, nextOutput.Format,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return {};
    if (resized)
        _intermediateLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageInfo image { GetImageView(),    GetImage(),       { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                        nextOutput.Format, nextOutput.Width, nextOutput.Height };
    SetImageLayout(cmd, image.Image, _intermediateLayout, VK_IMAGE_LAYOUT_GENERAL, image.SubresourceRange);
    _intermediateLayout = VK_IMAGE_LAYOUT_GENERAL;
    return image;
}

bool DlssNr_Vk::Dispatch(VkCommandBuffer cmd, const VkImageInfo& colour, const VkImageInfo& depth,
                         const VkImageInfo& motion, const VkImageInfo& output, const DlssNrFrameInfo_Vk& frame,
                         VkInstance instance, VkImageLayout inputLayout, bool* modelRan)
{
    if (!CanRender() || !colour.Image || !output.Image || colour.Image == output.Image)
        return false;

    // Always produce the unmodified frame first. A missing model, retryable failure, or missing
    // guides must leave the downstream pipeline a valid image. Compute copy needs only SAMPLED
    // usage on the game's input, unlike a transfer copy.
    SetImageLayout(cmd, colour.Image, inputLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, colour.SubresourceRange);
    DlssNrConstants copy {};
    copy.Mode = DlssNrMode_Downsample;
    copy.Width = output.Width;
    copy.Height = output.Height;
    const bool copied = Dispatch(cmd, copy, output.Width, output.Height, colour.ImageView, VK_NULL_HANDLE,
                                 VK_NULL_HANDLE, VK_NULL_HANDLE, output.ImageView, VK_NULL_HANDLE);
    SetImageLayout(cmd, colour.Image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, inputLayout, colour.SubresourceRange);
    if (!copied)
        return false;
    if (!_model)
        _model = std::make_unique<DlssNr::ModelVk>(*this);
    const bool ran =
        _model->Evaluate(cmd, colour, depth, motion, output, frame, instance, _physicalDevice, _device, inputLayout);
    if (modelRan)
        *modelRan = ran;
    return true;
}

void DlssNr_Vk::CaptureFinished(VkCommandBuffer cmd, const VkImageInfo& depth, const VkImageInfo& motion,
                                const DlssNrFrameInfo_Vk& frame, VkInstance instance)
{
    if (!CanRender())
        return;
    if (!_finished)
        _finished = std::make_unique<DlssNr::FinishedVk>(*this, _device, _physicalDevice);
    _finished->Capture(cmd, depth, motion, frame, instance);
}
