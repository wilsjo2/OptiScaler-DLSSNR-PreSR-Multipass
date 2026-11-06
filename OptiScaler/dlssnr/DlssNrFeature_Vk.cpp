#include "pch.h"

#include "DlssNrFeature_Vk_Internal.h"
#include "DlssNrFeature_Dx12.h"
#include "DlssNr_Status.h"
#include "DlssNr_Placement.h"
#include "DlssNrPipeline_Vk.h"
#include <nvsdk_ngx_vk.h>
#include "PassProfiles.h"
#include "DlssNr_Exposure.h"

#include <Config.h>
#include <State.h>
#include <proxies/NVNGX_Proxy.h>

#include <shaders/dlssnr/DlssNr_Vk.h>
#include <shaders/dlssnr/DlssNr_Guides.h>
#include <shaders/output_scaling/OS_Vk.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <string>

namespace DlssNr
{

void ModelVk::Impl::ReleaseIndependent()
{
    // Callers have retired GPU work before any stage or descriptor ring is freed.
    for (auto& pass : independentPasses)
        pass.output.Destroy(state.device);
    independentPasses.clear();
    independentWidth = independentHeight = 0;
    independentStatus = {};
}

bool ModelVk::Impl::PrepareIndependent(uint32_t width, uint32_t height, unsigned int passes,
                                        const DlssNrFrameInfo_Vk& frame, const Config& cfg)
{
    bool changed = independentPasses.size() != passes || independentWidth != width || independentHeight != height ||
                   independentBeforeSr != frame.BeforeUpscale || independentRayReconstruction != frame.RayReconstruction;
    for (unsigned int i = 0; i < passes; ++i)
    {
        const auto resolution = Profiles::PassResolutionSettings(cfg, i);
        if (resolution.workingScale < 1.0f && DlssNrUsesDlssEnlargement(resolution.transfer))
        {
            independentStatus.failureReason =
                std::format("Pass {}: DLSS enlargement requires the DX12 processing path.", i + 1);
            for (auto& pass : independentPasses)
                pass.model->state.reset = true;
            state.reset = true;
            return false;
        }
        if (!changed && (independentPasses[i].resolution != resolution ||
                         independentPasses[i].settings != Profiles::PassSettings(cfg, i)))
            changed = true;
    }
    if (!changed)
        return true;

    if (vkDeviceWaitIdle(state.device) != VK_SUCCESS)
    {
        independentStatus.failureReason = "The Vulkan device could not retire the previous pass chain.";
        return false;
    }
    ReleaseIndependent();
    ReleaseModels();
    state.activePasses = 0;
    state.creationPending = false;
    // Native-size encoded colour between stages avoids compounding tone mapping or composition.
    constexpr auto format = VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!state.proxy.Ensure(state.device, state.physicalDevice, width, height, format) ||
        !state.keep.Ensure(state.device, state.physicalDevice, width, height, format) ||
        !state.passClamp.Ensure(state.device, state.physicalDevice, width, height, format))
    {
        independentStatus.failureReason = "Could not allocate the independent pass composition images.";
        return false;
    }
    independentPasses.reserve(passes);
    for (unsigned int i = 0; i < passes; ++i)
    {
        auto& pass = independentPasses.emplace_back();
        pass.shader = std::make_unique<DlssNr_Vk>(std::format("DLSS-NR VK pass {}", i + 1), state.device,
                                                state.physicalDevice);
        pass.model = std::make_unique<Impl>(*pass.shader);
        pass.settings = Profiles::PassSettings(cfg, i);
        pass.resolution = Profiles::PassResolutionSettings(cfg, i);
        if (!pass.shader->IsInit() ||
            !pass.output.Ensure(state.device, state.physicalDevice, width, height, format))
        {
            independentStatus.failureReason = std::format("Could not allocate Vulkan pass {}.", i + 1);
            return false;
        }
    }
    independentWidth = width;
    independentHeight = height;
    independentBeforeSr = frame.BeforeUpscale;
    independentRayReconstruction = frame.RayReconstruction;
    state.reset = true;
    return true;
}

bool ModelVk::Impl::EvaluateIndependent(VkCommandBuffer cmdBuffer, const VkImageInfo& depth,
                                         const VkImageInfo& motion, const VkImageInfo& target,
                                         const DlssNrFrameInfo_Vk& frame, const Config& cfg,
                                         const DlssNrConstants& encode, VkImageView exposure)
{
    Transition(cmdBuffer, state.proxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cmdBuffer, state.keep, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ImageVk* input = &state.proxy;
    bool ready = true;
    double gpuTime = 0;
    bool haveTiming = true;
    Config passConfig = cfg;
    auto passFrame = frame;
    passFrame.ColourIsLinearHdr = false; // input is the already encoded proxy
    passFrame.WhitePointOverride = 1.0f;
    passFrame.Reset = frame.Reset || state.reset;
    struct ResetSkippedHistory
    {
        Impl& owner;
        bool completed = false;
        ~ResetSkippedHistory()
        {
            if (!completed)
            {
                owner.state.reset = true;
                for (auto& pass : owner.independentPasses)
                    pass.model->state.reset = true;
            }
        }
    } history { *this };
    for (unsigned int i = 0; i < independentPasses.size(); ++i)
    {
        auto& pass = independentPasses[i];
        Profiles::ApplyPassConfig(passConfig, cfg, i);
        // Return only a reconstructed encoded answer. The parent's resolve applies all
        // user composition controls once against the original frame.
        passConfig.DlssNrReversibleMode = 2u;
        passConfig.DlssNrApplyModel = true;
        passConfig.DlssNrCompare = 0u;
        passConfig.DlssNrDebugView = 0u;
        passConfig.DlssNrSkinProtection = false;
        passConfig.DlssNrShowSkinMask = false;
        passConfig.DlssNrReplaceDetailStrength = 0.0f;
        passConfig.DlssNrHaloSuppression = 0.0f;
        passConfig.DlssNrHaloBrightSuppression = 0.0f;
        passConfig.DlssNrHoldFrame = false;
        Transition(cmdBuffer, pass.output, VK_IMAGE_LAYOUT_GENERAL);
        const bool ran = pass.model->Evaluate(cmdBuffer, input->info, depth, motion, pass.output.info, passFrame,
                                              state.instance, state.physicalDevice, state.device,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, &passConfig);
        auto*& feature = pass.model->state.models[0].feature;
        // Some NGX versions cache identical create parameters; every stage must still
        // have its own temporal state, including two stages with identical settings.
        for (unsigned int earlier = 0; feature && earlier < i; ++earlier)
        {
            const auto* other = independentPasses[earlier].model->state.models[0].feature;
            if (other && (feature == other || feature->Id == other->Id))
            {
                feature = nullptr; // another stage owns it
                pass.model->Fail("the NVIDIA NGX driver reused a feature for independent NR passes");
            }
        }
        if (!ran || pass.model->state.failed)
        {
            if (independentStatus.failureReason.empty())
                independentStatus.failureReason = std::format("Pass {}: {}", i + 1,
                    pass.model->state.failed ? pass.model->state.reason : "waiting for model creation");
            ready = false;
        }
        else if (ready)
        {
            input = &pass.output;
            Transition(cmdBuffer, *input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (i + 1 < independentPasses.size())
            {
                // Match ordinary layered NR: only intermediate answers are clamped.
                // This stage's encode proxy is no longer needed and has native dimensions.
                auto& clamped = pass.model->state.proxy;
                Transition(cmdBuffer, clamped, VK_IMAGE_LAYOUT_GENERAL);
                DlssNrConstants clamp {};
                clamp.Mode = DlssNrMode_ClampProxy;
                clamp.Width = encode.Width;
                clamp.Height = encode.Height;
                if (!pass.shader->Dispatch(cmdBuffer, clamp, encode.Width, encode.Height, input->info.ImageView,
                                             VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                             clamped.info.ImageView, VK_NULL_HANDLE))
                {
                    independentStatus.failureReason = std::format("Pass {}: proxy clamp failed", i + 1);
                    ready = false;
                }
                else
                {
                    input = &clamped;
                    Transition(cmdBuffer, *input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                }
            }
        }
        haveTiming = haveTiming && pass.model->state.lastGpuTime.has_value();
        gpuTime += pass.model->state.lastGpuTime.value_or(0);
        independentStatus.spatialActive |= pass.model->state.spatialRan;
        if (!pass.model->state.spatialStatus.empty())
        {
            if (!independentStatus.spatialStatus.empty())
                independentStatus.spatialStatus += "; ";
            independentStatus.spatialStatus += std::format("Pass {}: {}", i + 1, pass.model->state.spatialStatus);
        }
    }
    // Warm every stage even when one is creating. Never expose a partial chain or keep
    // history from a frame that did not reach the final output.
    if (!ready)
        return false;

    auto resolve = encode;
    resolve.Mode = DlssNrMode_Resolve;
    resolve.Transfer = 0; // every stage has already reconstructed to native size
    resolve.ModelWorkScale = 1.0f;
    resolve.ReplaceDetailStrength = cfg.DlssNrReplaceDetailStrength.value_or_default();
    if (DlssNr_Common::HaloEnabled(cfg) && resolve.DebugView != 2)
    {
        auto halo = resolve;
        halo.Mode = DlssNrMode_SuppressHalo;
        Transition(cmdBuffer, state.passClamp, VK_IMAGE_LAYOUT_GENERAL);
        if (!state.pass->Dispatch(cmdBuffer, halo, encode.Width, encode.Height, state.proxy.info.ImageView,
                                  input->info.ImageView, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                  state.passClamp.info.ImageView, VK_NULL_HANDLE))
        {
            independentStatus.failureReason = "The independent pass halo dispatch failed.";
            return false;
        }
        input = &state.passClamp;
        Transition(cmdBuffer, *input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    // The packed-input diagnostic refers to the final model's own working image.
    auto* resolveProxy = &state.proxy;
    if (resolve.DebugView == 4)
    {
        auto& last = independentPasses.back().model->state;
        resolveProxy = last.spatialRan ? &last.spatialProxy : last.proxySmall.Valid() ? &last.proxySmall : &last.proxy;
        Transition(cmdBuffer, *resolveProxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        resolve.DebugView = 1;
    }
    if (!state.pass->Dispatch(cmdBuffer, resolve, encode.Width, encode.Height, resolveProxy->info.ImageView,
                              input->info.ImageView, state.keep.info.ImageView, exposure, target.ImageView,
                              VK_NULL_HANDLE))
    {
        independentStatus.failureReason = "The independent pass composition dispatch failed.";
        return false;
    }
    history.completed = true;
    state.reset = false;
    state.frames++;
    independentStatus.running = true;
    independentStatus.frames = state.frames;
    if (haveTiming)
        independentStatus.gpuTime = gpuTime;
    return true;
}

bool ModelVk::Impl::Evaluate(VkCommandBuffer cmdBuffer, const VkImageInfo& colourInfo, const VkImageInfo& depthInfo,
                             const VkImageInfo& motionInfo, const VkImageInfo& target, const DlssNrFrameInfo_Vk& frame,
                             VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                             VkImageLayout inputLayout, const Config* passConfig)
{
    const bool beforeSr = frame.BeforeUpscale;
    const bool rayReconstruction = frame.RayReconstruction;
    const auto& cfg = passConfig ? *passConfig : *Config::Instance();
    const unsigned int passes =
        std::clamp(cfg.DlssNrPasses.value_or_default(), 1u,
                   cfg.DlssNrUnlockPasses.value_or_default() ? DlssNr::MaxPassCount : DlssNr::DefaultMaxPassCount);
    const bool independent = !passConfig && Profiles::HasPassResolutionOverrides(cfg, passes);

    if (ResolvePlacement(cfg.DlssNrRunBeforeSr.value_or_default(), cfg.DlssNrDeferredDlss.value_or_default(),
                         cfg.DlssNrResidualAcrossRr.value_or_default(), cfg.DlssNrFinishedPicture.value_or_default())
            .deferred &&
        !frame.FinishedPicture)
    {

        if (!warnedDeferred)
        {
            LOG_WARN("DLSS-NR DeferredDLSS requires the D3D12 path or a D3D12 bridge; "
                     "native Vulkan leaves the clean SR frame unchanged");
            warnedDeferred = true;
        }
        return false;
    }

    if (cfg.DlssNrFinishedPicture.value_or_default() && !frame.FinishedPicture)
        return false; // the presentation stage owns this mode

    if (!cfg.DlssNrEnabled.value_or_default())
        return false;

    if (cmdBuffer == VK_NULL_HANDLE || device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE)
        return false;

    std::lock_guard<std::mutex> lock(mutex);
    state.spatialRan = false;
    if (!independent && DlssNrUsesDlssEnlargement(cfg.DlssNrTransfer.value_or_default()) &&
        cfg.DlssNrWorkingScale.value_or_default() < 1.0f)
    {
        if (!passConfig)
            PublishStatus(this, Backend::Vulkan, { false, "DLSS enlargement requires the DX12 processing path." });
        return false;
    }

    struct ReportStatus
    {
        Impl* owner;
        bool enabled;
        bool independent;
        ~ReportStatus()
        {
            if (!enabled)
                return;
            if (independent)
            {
                if (owner->state.failed)
                {
                    owner->independentStatus.running = false;
                    owner->independentStatus.failureReason = owner->state.reason;
                }
                PublishStatus(owner, Backend::Vulkan, owner->independentStatus);
                return;
            }
            const auto& state = owner->state;
            StatusSnapshot snapshot { state.models[0].feature != nullptr && !state.failed, state.reason,
                                      state.lastGpuTime, state.frames };
            snapshot.spatialStatus = state.spatialStatus;
            snapshot.spatialActive = state.spatialRan;
            PublishStatus(owner, Backend::Vulkan, snapshot);
        }
    } report { this, !passConfig, independent };
    if (independent)
        independentStatus = {};
    const auto requests = ReadControlRequests();
    if (requests.retryGeneration != retryGeneration)
    {
        retryGeneration = requests.retryGeneration;
        if (state.device && vkDeviceWaitIdle(state.device) != VK_SUCCESS)
        {
            Fail("the Vulkan device could not retire work for retry");
            return false;
        }
        Shutdown();
        state.failed = false;
        state.reason = "";
    }
    if (state.failed)
        return false;

    auto colourResource = WrapImage(colourInfo, false);
    auto depthResource = WrapImage(depthInfo, frame.DepthReadWrite);
    auto motionResource = WrapImage(motionInfo, frame.MotionReadWrite);
    auto* colour = &colourResource;
    auto* depth = &depthResource;
    auto* motion = &motionResource;

    if (colour->Resource.ImageViewInfo.ImageView == VK_NULL_HANDLE ||
        depth->Resource.ImageViewInfo.ImageView == VK_NULL_HANDLE ||
        motion->Resource.ImageViewInfo.ImageView == VK_NULL_HANDLE)
        return false;

    uint32_t width = colour->Resource.ImageViewInfo.Width;
    uint32_t height = colour->Resource.ImageViewInfo.Height;
    const auto renderWidth = frame.RenderSubrectWidth, renderHeight = frame.RenderSubrectHeight;
    const auto baseX = frame.ColorSubrectBaseX, baseY = frame.ColorSubrectBaseY;
    if (beforeSr)
    {
        // Origin-zero padded inputs are common with dynamic resolution. Never use a preset table.
        if (baseX || baseY || ((renderWidth == 0) != (renderHeight == 0)) || renderWidth > width ||
            renderHeight > height)
            return false; // caller falls back to post-SR, without editing the input
        if (renderWidth && renderHeight)
        {
            width = renderWidth;
            height = renderHeight;
        }
    }
    const auto depthX = frame.DepthSubrectBaseX, depthY = frame.DepthSubrectBaseY;
    const auto motionX = frame.MotionSubrectBaseX, motionY = frame.MotionSubrectBaseY;
    const auto outputWidth = frame.OutputWidth ? frame.OutputWidth : target.Width;
    const auto outputHeight = frame.OutputHeight ? frame.OutputHeight : target.Height;
    auto guides = ResolveGuideRegions({ depth->Resource.ImageViewInfo.Width, depth->Resource.ImageViewInfo.Height },
                                      { motion->Resource.ImageViewInfo.Width, motion->Resource.ImageViewInfo.Height },
                                      { renderWidth, renderHeight }, { outputWidth, outputHeight },
                                      frame.MotionVectorsLowResolution, depthX, depthY, motionX, motionY);
    if (!guides.depth.valid() || !guides.motion.valid())
        return false;
    const auto guideWidth = guides.depth.width, guideHeight = guides.depth.height;

    if (!width || !height || width > target.Width || height > target.Height)
        return false;

    // The model uses working resolution; source and composition remain native.
    float workScale = cfg.DlssNrWorkingScale.value_or_default();
    workScale = std::isfinite(workScale) ? std::clamp(workScale, 0.25f, 2.0f) : 1.0f;
    auto spatial = Spatial::Build(Spatial::ReadSettings(cfg), width, height, workScale);
    if (state.spatialDisabled &&
        (spatial != state.spatialAttemptLayout || colourInfo.Format != state.spatialColourFormat ||
         depthInfo.Format != state.spatialDepthFormat || motionInfo.Format != state.spatialMotionFormat ||
         depthInfo.Width != state.spatialDepthWidth || depthInfo.Height != state.spatialDepthHeight ||
         motionInfo.Width != state.spatialMotionWidth || motionInfo.Height != state.spatialMotionHeight))
    {
        state.spatialDisabled = false;
        state.spatialStatus.clear();
    }
    state.spatialAttemptLayout = spatial;
    state.spatialColourFormat = colourInfo.Format;
    state.spatialDepthFormat = depthInfo.Format;
    state.spatialMotionFormat = motionInfo.Format;
    state.spatialDepthWidth = depthInfo.Width;
    state.spatialDepthHeight = depthInfo.Height;
    state.spatialMotionWidth = motionInfo.Width;
    state.spatialMotionHeight = motionInfo.Height;
    if (spatial.active && !state.pass->SpatialReady())
    {
        state.spatialDisabled = true;
        state.spatialStatus = "Spatial compression unavailable: Vulkan shader pipeline failed";
    }
    if (state.spatialDisabled)
        spatial.active = false;
    if (!state.spatialDisabled)
    {
        if (spatial.active)
            state.spatialStatus = std::format("Spatial compression {}x{} → {}x{} ({:.1f}% pixels)", spatial.ordinaryW,
                                              spatial.ordinaryH, spatial.modelW, spatial.modelH,
                                              100.0 * double(spatial.modelW) * spatial.modelH /
                                                  (double(spatial.ordinaryW) * spatial.ordinaryH));
        else if (spatial.requested)
            state.spatialStatus = spatial.reason;
        else
            state.spatialStatus.clear();
    }
    const uint32_t workWidth = spatial.active ? spatial.modelW : spatial.ordinaryW;
    const uint32_t workHeight = spatial.active ? spatial.modelH : spatial.ordinaryH;
    const bool reduced = !spatial.active && (workWidth != width || workHeight != height);

    state.instance = instance;
    state.physicalDevice = physicalDevice;

    // The shader belongs to one device. Replacement features own replacement models.
    if (state.device != VK_NULL_HANDLE && state.device != device)
    {
        Fail("a Vulkan model was dispatched on a different device");
        return false;
    }
    state.device = device;

    if (independent)
    {
        if (!PrepareIndependent(width, height, passes, frame, cfg))
            return false;
    }
    else
    {
        if (!independentPasses.empty())
        {
            if (vkDeviceWaitIdle(device) != VK_SUCCESS)
            {
                Fail("the Vulkan device could not retire independently sized passes");
                return false;
            }
            ReleaseIndependent();
            state.reset = true;
        }
        if (!PrepareModels(cmdBuffer, frame, width, height, workWidth, workHeight, workScale, passes, spatial, cfg))
            return false;
    }

    if (!independent && spatial.active && workScale > 1.0f)
    {
        const Scaler wantScaler = cfg.DlssNrScalingDownscaler.value_or_default();
        if (state.nrScaler != wantScaler)
        {
            if (vkDeviceWaitIdle(state.device) != VK_SUCCESS)
            {
                Fail("the Vulkan device could not retire the supersampling filter");
                return false;
            }
            state.superDown.reset();
            state.spatialDownProxy.reset();
            state.nrScaler = wantScaler;
        }
        if (!state.superDown)
            state.superDown =
                std::make_unique<OS_Vk>("DLSS-NR VK spatial downsample", device, physicalDevice, false, wantScaler);
        if (!state.spatialDownProxy)
            state.spatialDownProxy = std::make_unique<OS_Vk>("DLSS-NR VK spatial proxy downsample", device,
                                                             physicalDevice, false, wantScaler);
    }

    // -----------------------------------------------------------------------------------------
    // Encode: the frame the upscaler wrote -> a display-referred proxy, plus an untouched copy
    // -----------------------------------------------------------------------------------------

    const bool gameSaysHdr = frame.ColourIsLinearHdr;
    const bool depthInverted = frame.DepthInverted;

    if (frame.Reset)
        state.reset = true;

    // Both have to agree. A game can set the HDR flag on a buffer that cannot hold open-ended light,
    // and encoding an already tone-mapped frame a second time looks washed out and banded.
    const bool linearHdr = gameSaysHdr && FormatCanHoldLinearHdr(colour->Resource.ImageViewInfo.Format);

    float whitePoint = cfg.DlssNrWhitePointScale.value_or_default();

    if (!saidEncoding)
    {
        saidEncoding = true;
        LOG_INFO("DLSS-NR Vulkan: the game's buffer is {} (flag {}, format {}), depth {}",
                 linearHdr ? "linear HDR" : "already tone-mapped", gameSaysHdr ? "set" : "clear",
                 (int) colour->Resource.ImageViewInfo.Format, depthInverted ? "inverted" : "normal");
    }

    if (frame.WhitePointOverride > 0.0f)
        whitePoint = frame.WhitePointOverride;

    auto encode = DlssNr_Common::MakeConstants(DlssNrMode_Encode, width, height, whitePoint, linearHdr, cfg);
    encode.GuideWidth = guideWidth;
    encode.GuideHeight = guideHeight;

    // Open the measurement. Reset immediately before writing: a query pool slot must be reset before
    // it is written again, and doing it here rather than at the end keeps the two in one place.
    const uint32_t timingSlot = (uint32_t) (state.timedFrames % kTimingSlots);

    if (!independent && state.queryPool != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmdBuffer, state.queryPool, timingSlot * 2, 2);
        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, state.queryPool, timingSlot * 2);
    }

    // The game's colour is read here and written at the end. Its layout on arrival is GENERAL, which
    // is what NGX requires of a resource it is handed, so it is left alone.
    Transition(cmdBuffer, state.proxy, VK_IMAGE_LAYOUT_GENERAL);
    Transition(cmdBuffer, state.keep, VK_IMAGE_LAYOUT_GENERAL);

    VkImageView exposureView = VK_NULL_HANDLE;
    const auto exposureSource = cfg.DlssNrWhitePointSource.value_or_default();
    const bool gameExposure = frame.Exposure.ImageView && frame.Exposure.Width == 1 && frame.Exposure.Height == 1;
    if (linearHdr && frame.WhitePointOverride <= 0 && (exposureSource == 3 || (exposureSource == 1 && gameExposure)))
    {
        const auto format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (state.exposure.Ensure(device, physicalDevice, 1, 1, format) &&
            (exposureSource != 3 || state.exposureMeter.Ensure(device, physicalDevice, 64, 64, format)))
        {
            DlssNrConstants meter {};
            ExposureConstants(meter, cfg, exposureSource, frame.PreExposure);
            meter.ExposureSourceWidth = width;
            meter.ExposureSourceHeight = height;
            Transition(cmdBuffer, state.exposure, VK_IMAGE_LAYOUT_GENERAL);
            bool ready;
            if (exposureSource == 3)
            {
                Transition(cmdBuffer, state.exposureMeter, VK_IMAGE_LAYOUT_GENERAL);
                meter.Mode = DlssNrMode_Meter;
                meter.Width = meter.Height = 64;
                ready = state.pass->Dispatch(cmdBuffer, meter, 64, 64, colour->Resource.ImageViewInfo.ImageView,
                                             VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                             state.exposureMeter.info.ImageView, VK_NULL_HANDLE, inputLayout);
                Transition(cmdBuffer, state.exposureMeter, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                meter.Mode = DlssNrMode_AutoExposure;
                meter.Width = meter.Height = 1;
                ready = ready && state.pass->Dispatch(cmdBuffer, meter, 1, 1, state.exposureMeter.info.ImageView,
                                                      VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                      state.exposure.info.ImageView, VK_NULL_HANDLE);
            }
            else
            {
                meter.Mode = DlssNrMode_Downsample;
                meter.Width = meter.Height = 1;
                ready = state.pass->Dispatch(cmdBuffer, meter, 1, 1, frame.Exposure.ImageView, VK_NULL_HANDLE,
                                             VK_NULL_HANDLE, VK_NULL_HANDLE, state.exposure.info.ImageView,
                                             VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
            }
            Transition(cmdBuffer, state.exposure, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (ready)
            {
                ExposureConstants(encode, cfg, exposureSource, frame.PreExposure);
                exposureView = state.exposure.info.ImageView;
            }
        }
    }

    // Read the caller's actual input layout; the resolve restores it after writing.
    if (!state.pass->Dispatch(cmdBuffer, encode, width, height, colour->Resource.ImageViewInfo.ImageView,
                              VK_NULL_HANDLE, VK_NULL_HANDLE, exposureView, state.proxy.info.ImageView,
                              state.keep.info.ImageView, inputLayout))
    {
        Fail("the encode dispatch failed");
        return false;
    }

    if (independent)
        return EvaluateIndependent(cmdBuffer, depthInfo, motionInfo, target, frame, cfg, encode, exposureView);

    // The model's input: the full proxy, or a downsampled copy of it when the working scale is below
    // the frame. Mirrors the D3D12 path -- the encode always writes a full proxy, and a separate
    // downsample makes the small one the model actually reads.
    ImageVk* modelInput = &state.proxy;

    if (spatial.active)
    {
        Transition(cmdBuffer, state.proxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, state.spatialProxy, VK_IMAGE_LAYOUT_GENERAL);
        auto constants = Spatial::MakeConstants(spatial, 100, guides, frame.MvScaleX, frame.MvScaleY, width, height);
        if (!state.pass->DispatchSpatial(cmdBuffer, constants, state.proxy.info.ImageView, VK_NULL_HANDLE,
                                         VK_NULL_HANDLE, state.spatialProxy.info.ImageView, VK_NULL_HANDLE))
        {
            state.spatialDisabled = true;
            state.spatialStatus = "Spatial compression unavailable: colour packing failed";
            return false;
        }

        Transition(cmdBuffer, state.spatialDepth, VK_IMAGE_LAYOUT_GENERAL);
        Transition(cmdBuffer, state.spatialMotion, VK_IMAGE_LAYOUT_GENERAL);
        constants = Spatial::MakeConstants(spatial, 101, guides, frame.MvScaleX, frame.MvScaleY, width, height);
        if (!state.pass->DispatchSpatial(
                cmdBuffer, constants, VK_NULL_HANDLE, depthInfo.ImageView, motionInfo.ImageView,
                state.spatialDepth.info.ImageView, state.spatialMotion.info.ImageView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                frame.DepthReadWrite ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                frame.MotionReadWrite ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
        {
            state.spatialDisabled = true;
            state.spatialStatus = "Spatial compression unavailable: guide packing failed";
            return false;
        }
        depthResource = WrapImage(state.spatialDepth.info, true);
        motionResource = WrapImage(state.spatialMotion.info, true);
        guides = { { 0, 0, workWidth, workHeight }, { 0, 0, workWidth, workHeight } };
        modelInput = &state.spatialProxy;
    }

    if (reduced && state.proxySmall.Valid())
    {
        bool built = false;

        if (workScale > 1.0f)
        {
            // Supersample: upscale the proxy to the super-native working size with the chosen filter so
            // the model sees a clean input. Rebuild both scalers when the NR downscaler changed (baked
            // at construction). proxy -> SHADER_READ_ONLY (sampled), proxySmall -> GENERAL (storage).
            const Scaler wantScaler = cfg.DlssNrScalingDownscaler.value_or_default();
            if (state.nrScaler != wantScaler)
            {
                // Drain submitted work before replacing filter pipelines and descriptor resources.
                if (state.device != VK_NULL_HANDLE && vkDeviceWaitIdle(state.device) != VK_SUCCESS)
                {
                    Fail("the Vulkan device could not retire the supersampling filters");
                    return false;
                }
                state.superUp.reset();
                state.superDown.reset();
                state.spatialDownProxy.reset();
                state.nrScaler = wantScaler;
            }
            if (!state.superUp)
                state.superUp =
                    std::make_unique<OS_Vk>("DLSS-NR VK supersample up", device, physicalDevice, true, wantScaler);
            if (!state.superDown)
                state.superDown =
                    std::make_unique<OS_Vk>("DLSS-NR VK supersample down", device, physicalDevice, false, wantScaler);

            Transition(cmdBuffer, state.proxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            Transition(cmdBuffer, state.proxySmall, VK_IMAGE_LAYOUT_GENERAL);

            VkImageInfo upin = state.proxy.info;
            VkImageInfo upout = state.proxySmall.info;

            if (state.superUp && state.superUp->IsInit() && state.superUp->DispatchResources(cmdBuffer, upin, upout))
                built = true;
            else
            {

                if (!warnedVkSuper)
                {
                    warnedVkSuper = true;
                    LOG_WARN("DLSS-NR Vulkan supersample: upscaler unavailable, falling back to box enlarge.");
                }
            }
        }

        if (!built)
        {
            DlssNrConstants down = encode;
            down.Mode = DlssNrMode_Downsample;
            down.Width = workWidth;
            down.Height = workHeight;

            Transition(cmdBuffer, state.proxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            Transition(cmdBuffer, state.proxySmall, VK_IMAGE_LAYOUT_GENERAL);

            if (!state.pass->Dispatch(cmdBuffer, down, workWidth, workHeight, state.proxy.info.ImageView,
                                      VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, state.proxySmall.info.ImageView,
                                      VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
            {
                Fail("the downsample dispatch failed");
                return false;
            }
        }

        modelInput = &state.proxySmall;
    }

    // Below native (spatial compression packs its own guides), the model was handed a colour at
    // the working size and the game's depth and motion at their own size, with only the vector
    // magnitudes rescaled -- the contract the D3D12 path had before MatchGuides, and it flickered at
    // every scale below 100% there. Same fix, same shader mode: resample both to the working size
    // and describe them as a full zero-origin region. The vectors keep the game's units; the scale
    // below converts them. Mirrors DlssNr_Dx12_Run.cpp.
    bool matchedGuides = false;
    if (reduced && workWidth < width && cfg.DlssNrMatchGuides.value_or_default() && state.depthSmall.Valid() &&
        state.motionSmall.Valid())
    {
        DlssNrConstants resize {};
        resize.Mode = DlssNrMode_ResizePrivateGuides;
        resize.Width = workWidth;
        resize.Height = workHeight;
        resize.GuideWidth = guides.depth.width;
        resize.GuideHeight = guides.depth.height;
        resize.DebugView = guides.depth.x;
        resize.CompareMode = guides.depth.y;
        resize.TransferStrength = float(guides.motion.width);
        resize.ColourStrength = float(guides.motion.height);
        resize.CompareSwap = guides.motion.x;
        resize.Transfer = guides.motion.y;
        resize.MvScaleX = resize.MvScaleY = 1.0f;
        Transition(cmdBuffer, state.depthSmall, VK_IMAGE_LAYOUT_GENERAL);
        Transition(cmdBuffer, state.motionSmall, VK_IMAGE_LAYOUT_GENERAL);
        // Mode 10 reads depth through the source slot and motion through the model slot; both are
        // the game's images, so both layouts are stated.
        if (state.pass->Dispatch(
                cmdBuffer, resize, workWidth, workHeight, depthInfo.ImageView, motionInfo.ImageView, VK_NULL_HANDLE,
                VK_NULL_HANDLE, state.depthSmall.info.ImageView, state.motionSmall.info.ImageView,
                frame.DepthReadWrite ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false, nullptr,
                frame.MotionReadWrite ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
        {
            depthResource = WrapImage(state.depthSmall.info, true);
            motionResource = WrapImage(state.motionSmall.info, true);
            guides = { { 0, 0, workWidth, workHeight }, { 0, 0, workWidth, workHeight } };
            matchedGuides = true;
        }
        else if (!warnedMatchGuides)
        {
            warnedMatchGuides = true;
            LOG_WARN("DLSS-NR Vulkan: guides could not be matched to the working size; the model keeps the game's "
                     "{}x{} guides for its {}x{} colour",
                     guideWidth, guideHeight, workWidth, workHeight);
        }
    }

    // -----------------------------------------------------------------------------------------
    // The model
    // -----------------------------------------------------------------------------------------

    // The game's scale turns its vectors into pixels of the size they are measured in: the render
    // size for low-resolution vectors, the output size otherwise. The model reads the scale in
    // pixels of the motion texture it is handed, so the conversion is "texture the model reads /
    // size the vectors are measured in": the matched resample when there is one, the game's own
    // region otherwise (which makes that case the game's own scale). Same rule as the D3D12 path;
    // RenderMotionScale=false keeps the old working size / frame size conversion for an A/B.
    float mvX = frame.MvScaleX, mvY = frame.MvScaleY;
    const bool renderMotionScale = cfg.DlssNrRenderMotionScale.value_or_default();
    const uint32_t mvGuideW = !renderMotionScale ? workWidth : matchedGuides ? workWidth : guides.motion.width;
    const uint32_t mvGuideH = !renderMotionScale ? workHeight : matchedGuides ? workHeight : guides.motion.height;
    const uint32_t mvRefW = !renderMotionScale ? width : frame.MotionVectorsLowResolution ? renderWidth : outputWidth;
    const uint32_t mvRefH = !renderMotionScale                 ? height
                            : frame.MotionVectorsLowResolution ? renderHeight
                                                               : outputHeight;
    if (spatial.active)
        mvX = mvY = 1.0f; // packed vectors already carry displacement in packed pixels
    else
    {
        mvX *= mvRefW ? (float) mvGuideW / mvRefW : 1.0f;
        mvY *= mvRefH ? (float) mvGuideH / mvRefH : 1.0f;
    }
    ImageVk* answer = &state.output;
    ImageVk* input = modelInput;
    bool clampFailed = false;
    uint32_t clampSlots[2] = { UINT32_MAX, UINT32_MAX };
    NVSDK_NGX_Result evaluated = NVSDK_NGX_Result_Success;
    for (unsigned int pass = 0; pass < passes; ++pass)
    {
        Transition(cmdBuffer, *input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, *answer, VK_IMAGE_LAYOUT_GENERAL);
        auto inputResource = WrapImage(input->info, true);
        auto answerResource = WrapImage(answer->info, true);
        evaluated = EvaluateModel(cmdBuffer, pass, &inputResource, depth, motion, &answerResource, workWidth,
                                  workHeight, guides, depthInverted, mvX, mvY, cfg);
        if (evaluated != NVSDK_NGX_Result_Success)
            break;
        if (pass + 1 < passes)
        {
            Transition(cmdBuffer, *answer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            Transition(cmdBuffer, state.passClamp, VK_IMAGE_LAYOUT_GENERAL);
            DlssNrConstants clamp {};
            clamp.Mode = DlssNrMode_ClampProxy;
            clamp.Width = workWidth;
            clamp.Height = workHeight;
            if (!state.pass->Dispatch(cmdBuffer, clamp, workWidth, workHeight, answer->info.ImageView, VK_NULL_HANDLE,
                                      VK_NULL_HANDLE, VK_NULL_HANDLE, state.passClamp.info.ImageView, VK_NULL_HANDLE,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false, &clampSlots[pass % 2]))
            {
                clampFailed = true;
                break; // Resolve the last valid answer and reset skipped histories next frame.
            }
            input = &state.passClamp;
            answer = answer == &state.output ? &state.scratch : &state.output;
        }
    }

    state.reset = clampFailed;
    state.frames++;

    if (evaluated != NVSDK_NGX_Result_Success)
    {
        LOG_ERROR("DLSS-NR Vulkan: evaluate returned 0x{:X}", static_cast<unsigned int>(evaluated));
        if (spatial.active)
        {
            state.spatialDisabled = true;
            state.spatialStatus = "Spatial compression unavailable: the driver rejected the packed evaluation";
            state.reset = true;
            return false;
        }
        Fail("the model refused to evaluate");
        return false;
    }

    // -----------------------------------------------------------------------------------------
    // Resolve: proxy + the model's answer + the untouched copy -> the frame
    // -----------------------------------------------------------------------------------------

    DlssNrConstants resolve = encode;
    resolve.Mode = DlssNrMode_Resolve;
    resolve.ReplaceDetailStrength = cfg.DlssNrReplaceDetailStrength.value_or_default();
    resolve.ModelWorkScale = workScale;

    if (DlssNr_Common::HaloEnabled(cfg) && cfg.DlssNrDebugView.value_or_default() != 2)
    {
        Transition(cmdBuffer, *modelInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, *answer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, state.passClamp, VK_IMAGE_LAYOUT_GENERAL);
        auto halo = resolve;
        halo.Mode = DlssNrMode_SuppressHalo;
        halo.Width = workWidth;
        halo.Height = workHeight;
        if (!state.pass->Dispatch(cmdBuffer, halo, workWidth, workHeight, modelInput->info.ImageView,
                                  answer->info.ImageView, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                  state.passClamp.info.ImageView, VK_NULL_HANDLE,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
            return false;
        answer = &state.passClamp;
    }

    // Downsample the model answer to native before composition.
    ImageVk* resolveProxy = modelInput;
    ImageVk* resolveAnswer = answer;

    if (spatial.active)
    {
        Transition(cmdBuffer, *modelInput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, *answer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, state.spatialProxyUnpacked, VK_IMAGE_LAYOUT_GENERAL);
        Transition(cmdBuffer, state.spatialAnswerUnpacked, VK_IMAGE_LAYOUT_GENERAL);
        const auto constants =
            Spatial::MakeConstants(spatial, 102, guides, frame.MvScaleX, frame.MvScaleY, width, height);
        if (!state.pass->DispatchSpatial(cmdBuffer, constants, modelInput->info.ImageView, answer->info.ImageView,
                                         VK_NULL_HANDLE, state.spatialProxyUnpacked.info.ImageView,
                                         state.spatialAnswerUnpacked.info.ImageView))
        {
            state.spatialDisabled = true;
            state.spatialStatus = "Spatial compression unavailable: answer unpacking failed";
            return false;
        }
        resolveProxy = &state.spatialProxyUnpacked;
        resolveAnswer = &state.spatialAnswerUnpacked;
    }

    if (spatial.active && workScale > 1.0f)
    {
        if (!state.superDown || !state.superDown->IsInit() || !state.spatialDownProxy ||
            !state.spatialDownProxy->IsInit())
        {
            state.spatialDisabled = true;
            state.spatialStatus = "Spatial compression unavailable: downsampling filter failed";
            return false;
        }
        Transition(cmdBuffer, *resolveProxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, *resolveAnswer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, state.spatialProxyNative, VK_IMAGE_LAYOUT_GENERAL);
        Transition(cmdBuffer, state.outputNative, VK_IMAGE_LAYOUT_GENERAL);
        if (!state.spatialDownProxy->DispatchResources(cmdBuffer, resolveProxy->info, state.spatialProxyNative.info) ||
            !state.superDown->DispatchResources(cmdBuffer, resolveAnswer->info, state.outputNative.info))
        {
            state.spatialDisabled = true;
            state.spatialStatus = "Spatial compression unavailable: matched downsampling failed";
            return false;
        }
        resolveProxy = &state.spatialProxyNative;
        resolveAnswer = &state.outputNative;
    }
    else if (workScale > 1.0f && state.superDown && state.superDown->IsInit() && state.outputNative.Valid())
    {
        Transition(cmdBuffer, *resolveAnswer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Transition(cmdBuffer, state.outputNative, VK_IMAGE_LAYOUT_GENERAL);

        VkImageInfo dsin = resolveAnswer->info;
        VkImageInfo dsout = state.outputNative.info;

        if (state.superDown->DispatchResources(cmdBuffer, dsin, dsout))
        {
            resolveProxy = &state.proxy;
            resolveAnswer = &state.outputNative;
        }
    }

    // Match DX12: inspect the immutable packed model input using the existing proxy display.
    if (resolve.DebugView == 4)
    {
        resolveProxy = modelInput;
        resolve.DebugView = 1;
    }
    // The child outputs encoded proxy colour. Matched residual reconstruction needs the
    // full-size proxy here, without the public codec's final composition controls.
    if (passConfig && resolve.Transfer == 1 && workScale < 1.0f)
        resolve.Mode = DlssNrMode_RawResidual;
    Transition(cmdBuffer, *resolveProxy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cmdBuffer, *resolveAnswer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(cmdBuffer, state.keep, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (!state.pass->Dispatch(cmdBuffer, resolve, width, height, resolveProxy->info.ImageView,
                              resolveAnswer->info.ImageView, state.keep.info.ImageView, exposureView, target.ImageView,
                              VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
    {
        Fail("the resolve dispatch failed");
        return false;
    }

    // Close it, and read the pair from three frames ago -- retired by now, so the read does not wait.
    if (state.queryPool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, state.queryPool, timingSlot * 2 + 1);
        state.timedFrames++;

        if (state.timedFrames > kTimingSlots)
        {
            const uint32_t readSlot = (uint32_t) (state.timedFrames % kTimingSlots);
            uint64_t ticks[2] = {};

            // Without WAIT: a slot this old is retired, and if it somehow is not, NOT_READY is the
            // right answer rather than a stall.
            if (vkGetQueryPoolResults(device, state.queryPool, readSlot * 2, 2, sizeof(ticks), ticks, sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
                ticks[1] > ticks[0])
            {
                const double ms = (double) (ticks[1] - ticks[0]) * (double) state.timestampPeriod / 1e6;

                // A pass that appears to have taken over a second did not; the queue was reset under
                // it or the pair straddled a device change.
                if (ms > 0.0 && ms < 1000.0)
                    state.lastGpuTime = ms;
            }
        }
    }

    if (!reported && state.frames > 2)
    {
        reported = true;
        LOG_INFO("DLSS-NR Vulkan: running {} SR at {}x{}, guides {}x{}", beforeSr ? "before" : "after", width, height,
                 guideWidth, guideHeight);
        if (!spatial.active)
            LOG_INFO("DLSS-NR Vulkan model motion scale {:.1f} x {:.1f}: game scale {} x {} measured against {}x{} "
                     "({}), motion texture {}x{} ({}), model {}x{}",
                     mvX, mvY, frame.MvScaleX, frame.MvScaleY, mvRefW, mvRefH,
                     !renderMotionScale                 ? "frame size, RenderMotionScale=false"
                     : frame.MotionVectorsLowResolution ? "render size, low-resolution vectors"
                                                        : "output size",
                     mvGuideW, mvGuideH, matchedGuides ? "matched to the working size" : "the game's region", workWidth,
                     workHeight);
    }
    state.spatialRan = spatial.active;
    return true;
}

ModelVk::ModelVk(DlssNr_Vk& shader) : _impl(std::make_unique<Impl>(shader)) {}
ModelVk::~ModelVk() = default;
bool ModelVk::Evaluate(VkCommandBuffer cmd, const VkImageInfo& colour, const VkImageInfo& depth,
                       const VkImageInfo& motion, const VkImageInfo& output, const DlssNrFrameInfo_Vk& frame,
                       VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, VkImageLayout inputLayout)
{
    return _impl->Evaluate(cmd, colour, depth, motion, output, frame, instance, physicalDevice, device, inputLayout);
}
} // namespace DlssNr
