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

bool ModelVk::Impl::Evaluate(VkCommandBuffer cmdBuffer, const VkImageInfo& colourInfo, const VkImageInfo& depthInfo,
                             const VkImageInfo& motionInfo, const VkImageInfo& target, const DlssNrFrameInfo_Vk& frame,
                             VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                             VkImageLayout inputLayout)
{
    const bool beforeSr = frame.BeforeUpscale;
    const bool rayReconstruction = frame.RayReconstruction;
    auto& cfg = *Config::Instance();

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
    if (DlssNrUsesDlssEnlargement(cfg.DlssNrTransfer.value_or_default()) &&
        cfg.DlssNrWorkingScale.value_or_default() < 1.0f)
    {
        PublishStatus(this, Backend::Vulkan, { false, "DLSS enlargement requires the DX12 processing path." });
        return false;
    }

    struct ReportStatus
    {
        Impl* owner;
        ~ReportStatus()
        {
            const auto& state = owner->state;
            StatusSnapshot snapshot { state.models[0].feature != nullptr && !state.failed, state.reason,
                                      state.lastGpuTime, state.frames };
            snapshot.spatialStatus = state.spatialStatus;
            snapshot.spatialActive = state.spatialRan;
            PublishStatus(owner, Backend::Vulkan, snapshot);
        }
    } report { this };
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
    const unsigned int passes =
        std::clamp(cfg.DlssNrPasses.value_or_default(), 1u,
                   cfg.DlssNrUnlockPasses.value_or_default() ? DlssNr::MaxPassCount : DlssNr::DefaultMaxPassCount);

    state.instance = instance;
    state.physicalDevice = physicalDevice;

    // The shader belongs to one device. Replacement features own replacement models.
    if (state.device != VK_NULL_HANDLE && state.device != device)
    {
        Fail("a Vulkan model was dispatched on a different device");
        return false;
    }
    state.device = device;

    if (!PrepareModels(cmdBuffer, frame, width, height, workWidth, workHeight, workScale, passes, spatial))
        return false;

    if (spatial.active && workScale > 1.0f)
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

    if (state.queryPool != VK_NULL_HANDLE)
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

    // -----------------------------------------------------------------------------------------
    // The model
    // -----------------------------------------------------------------------------------------

    float mvX = frame.MvScaleX, mvY = frame.MvScaleY;
    // Match D3D12: preserve the game's vector encoding, then adjust only for the NR working scale.
    if (spatial.active)
        mvX = mvY = 1.0f; // packed vectors already carry displacement in packed pixels
    else
    {
        mvX *= (float) workWidth / width;
        mvY *= (float) workHeight / height;
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
