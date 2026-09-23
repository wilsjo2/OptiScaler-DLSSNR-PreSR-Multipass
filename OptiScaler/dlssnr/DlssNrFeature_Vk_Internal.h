#pragma once

#include "DlssNrFeature_Vk.h"
#include "DlssNr_Image_Vk.h"
#include "DlssNrFeature_Dx12.h"
#include <shaders/dlssnr/DlssNr_Guides.h>
#include <shaders/dlssnr/DlssNr_Spatial.h>
#include "PassProfiles.h"
#include <nvsdk_ngx_vk.h>
#include <shaders/output_scaling/OS_Vk.h>
#include <mutex>

namespace DlssNr
{

// Each layer has its own driver-created feature and capability parameter map.
// Sharing either would couple temporal histories when two passes use the same profile.
struct NgxPassVk
{
    NVSDK_NGX_Handle* feature = nullptr;
    NVSDK_NGX_Parameter* parameters = nullptr;
};

struct VkState
{
    bool failed = false;
    const char* reason = "";

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    bool ngxInitialised = false;
    NgxPassVk models[DlssNr::MaxPassCount] {};
    ModelSettings builtSettings[DlssNr::MaxPassCount] {};
    unsigned int activePasses = 0;
    VkEvent creationReady = VK_NULL_HANDLE;
    bool creationPending = false;

    // What the model writes, the proxy it is shown, and the frame as the upscaler left it.
    ImageVk output;
    ImageVk scratch;
    ImageVk passClamp;
    ImageVk proxy;
    ImageVk keep;
    ImageVk exposureMeter;
    ImageVk exposure;
    bool beforeSr = false;
    bool rayReconstruction = false;

    // The proxy at the model's working size, when that is below the frame. The model -- 98% of the
    // cost -- then runs on this instead of the full proxy, which is the whole point of the working
    // scale slider. Unused (and never created) at scale 1, so the default path is unchanged.
    ImageVk proxySmall;

    // Spatial compression has its own packed model input and typed packed guides. Vulkan
    // stores motion in RGBA32F to avoid requiring shaderStorageImageExtendedFormats for
    // RG32F storage writes; NGX reads its xy channels. The two ordinary-size images hold
    // the unpacked proxy/answer for the existing resolve shader.
    ImageVk spatialProxy;
    ImageVk spatialDepth;
    ImageVk spatialMotion;
    ImageVk spatialProxyUnpacked;
    ImageVk spatialAnswerUnpacked;
    ImageVk spatialProxyNative;
    Spatial::Layout spatialLayout;
    Spatial::Layout spatialAttemptLayout;
    VkFormat spatialColourFormat = VK_FORMAT_UNDEFINED;
    VkFormat spatialDepthFormat = VK_FORMAT_UNDEFINED;
    VkFormat spatialMotionFormat = VK_FORMAT_UNDEFINED;
    uint32_t spatialDepthWidth = 0, spatialDepthHeight = 0;
    uint32_t spatialMotionWidth = 0, spatialMotionHeight = 0;
    bool spatialDisabled = false;
    bool spatialRan = false;
    std::string spatialStatus;

    // Supersampling (working scale > 1): the model runs above native, superUp enlarges the proxy to
    // that size and superDown averages the answer (output) back into outputNative at native for a 1:1
    // composite. nrScaler is the filter both were built with, so a changed DlssNrScalingDownscaler
    // rebuilds them. Unused and never created at scale <= 1.
    ImageVk outputNative;
    std::unique_ptr<OS_Vk> superUp;
    std::unique_ptr<OS_Vk> superDown;
    std::unique_ptr<OS_Vk> spatialDownProxy;
    Scaler nrScaler = Scaler::Count;

    DlssNr_Vk* pass = nullptr;

    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t workWidth = 0;
    uint32_t workHeight = 0;
    bool reset = true;
    unsigned long long frames = 0;

    // Timing. A pair of timestamps per frame across a ring, read back three frames later: a query
    // read the frame it was written stalls the CPU on the GPU, which would cost more than the pass
    // it is measuring. Vulkan reports ticks, and timestampPeriod is how many nanoseconds a tick is.
    VkQueryPool queryPool = VK_NULL_HANDLE;
    float timestampPeriod = 0.0f;
    unsigned long long timedFrames = 0;
    std::optional<double> lastGpuTime;
};

// Four frames of pairs. Three would do, four keeps the modulo cheap and the slot being written well
// clear of the slot being read.
constexpr uint32_t kTimingSlots = 4;

struct ModelVk::Impl
{
    VkState state;
    bool reported = false;
    bool warnedVkSuper = false;
    bool saidEncoding = false;
    bool warnedDeferred = false;
    std::mutex mutex;
    uint64_t retryGeneration = ReadControlRequests().retryGeneration;
    Impl(DlssNr_Vk& shader) { state.pass = &shader; }
    ~Impl()
    {
        Shutdown();
        ClearStatus(this);
    }

    void Fail(const char* why);
    bool CreateImage(ImageVk& img, uint32_t width, uint32_t height, VkFormat format);
    void Transition(VkCommandBuffer cmd, ImageVk& img, VkImageLayout to);
    void TransitionForeign(VkCommandBuffer cmd, VkImage image, VkImageSubresourceRange range, VkImageLayout from,
                           VkImageLayout to);
    bool InitDriver(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
    void ReleaseModels();
    bool CreateModel(VkCommandBuffer commandBuffer, unsigned int passIndex, unsigned int width, unsigned int height,
                     const Config& config);
    NVSDK_NGX_Result EvaluateModel(VkCommandBuffer commandBuffer, unsigned int passIndex, NVSDK_NGX_Resource_VK* colour,
                                   NVSDK_NGX_Resource_VK* depth, NVSDK_NGX_Resource_VK* motion,
                                   NVSDK_NGX_Resource_VK* output, unsigned int width, unsigned int height,
                                   const GuideRegions& guides, bool depthInverted, float mvX, float mvY,
                                   const Config& config);
    bool FormatCanHoldLinearHdr(VkFormat format);
    bool PrepareModels(VkCommandBuffer cmdBuffer, const DlssNrFrameInfo_Vk& frame, uint32_t width, uint32_t height,
                       uint32_t workWidth, uint32_t workHeight, float workScale, unsigned int passes,
                       const Spatial::Layout& spatial);
    bool Evaluate(VkCommandBuffer cmdBuffer, const VkImageInfo& colourInfo, const VkImageInfo& depthInfo,
                  const VkImageInfo& motionInfo, const VkImageInfo& target, const DlssNrFrameInfo_Vk& frame,
                  VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, VkImageLayout inputLayout);
    void Shutdown();
};
} // namespace DlssNr
