#include "pch.h"
#include "DlssNr_Dx12_State.h"
#include <dlssnr/NrPassBorrow.h>
#include <dlssnr/NrNvfp4.h>
#include <dlssnr/PassProfiles.h>
#include <Util.h>

void DlssNr_Dx12::State::ReleaseIndependentPasses()
{
    for (auto& pass : independentPasses)
        DlssNr_Dx12::Retire(std::move(pass));
    independentSizes.clear();
    independentModels.clear();
    independentStatus.clear();
    independentSpatialStatus.clear();
    independentActive = false;
    nr.reset = true;
}

bool DlssNr_Dx12::State::RunIndependentPasses(EncodeContext& encoded, ID3D12Resource* depth,
                                              ID3D12Resource* motion, unsigned int passes,
                                              ID3D12CommandQueue* queue)
{
    const auto& cfg = RunConfig();
    auto* cmd = encoded.cmdList;
    std::vector<DlssNr::Profiles::ResolutionSettings> sizes;
    std::vector<DlssNr::ModelSettings> models;
    std::vector<bool> reapplies;
    for (unsigned pass = 0; pass < passes; ++pass)
    {
        // A re-applying pass has no child: it repeats the previous child's change in one dispatch here.
        reapplies.push_back(DlssNr::Profiles::PassReapplies(cfg, pass));
        if (reapplies.back())
            continue;
        sizes.push_back(DlssNr::Profiles::PassResolutionSettings(cfg, pass));
        models.push_back(PassSettings(cfg, pass));
    }
    if (reapplies != independentReapply)
        LOG_INFO("DLSS-NR independent passes: configured {}, {} re-applied (gain {:.3f})", passes,
                 std::count(reapplies.begin(), reapplies.end(), true), DlssNr::Profiles::ReapplyGain(cfg));
    const bool reset = nr.reset || sizes != independentSizes || models != independentModels ||
                       reapplies != independentReapply;
    independentSizes = std::move(sizes);
    independentModels = std::move(models);
    independentReapply = std::move(reapplies);
    for (unsigned pass = 0; pass < independentPasses.size(); ++pass)
        if (pass >= passes || independentReapply[pass])
            DlssNr_Dx12::Retire(std::move(independentPasses[pass]));

    // Every child owns its NGX history, reconstruction surfaces and descriptor ring. Only the
    // encoded proxy crosses a stage boundary; the immutable scene remains in this owner's hdrCopy.
    auto frame = encoded.frame;
    frame.ColourIsLinearHdr = false;
    frame.ExposureTexture = nullptr;
    frame.WhitePointOverride = 1.0f;
    frame.Reset = reset;
    if (nr.heldActive)
        frame.MvScaleX = frame.MvScaleY = 0;
    ID3D12Resource* source = nr.colorCopy;
    ID3D12Resource* answer = nullptr;
    bool ok = nr.passScratch && nr.passClamp;
    bool clampReadable = false;
    independentStatus.clear();
    independentSpatialStatus.clear();
    nr.spatialActive = false;
    unsigned lastNetwork = 0;
    ngxTime->Start(cmd);
    for (unsigned pass = 0; ok && pass < passes;)
    {
        if (const auto run = pass == 0 ? 0 : DlssNr::Profiles::ReapplyRun(cfg, pass, passes))
        {
            // source and answer are the last child's input and answer; write the other ping-pong texture.
            auto* reapplied = answer == nr.output ? nr.passScratch : nr.output;
            if (source == nr.passClamp)
                Barrier(cmd, nr.passClamp, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            DlssNrConstants reapply {};
            reapply.Mode = DlssNrMode_Reapply;
            reapply.WhitePoint = float(run) * DlssNr::Profiles::ReapplyGain(cfg);
            reapply.Width = nr.width;
            reapply.Height = nr.height;
            const bool dispatched =
                shader.DispatchPass(cmd, reapply, source, answer, nullptr, nullptr, nullptr, reapplied, nullptr);
            if (source == nr.passClamp)
                Barrier(cmd, nr.passClamp, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!dispatched)
            {
                ReportSkipOnce("the re-apply dispatch failed; using the last model answer");
                break;
            }
            Barrier(cmd, answer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(cmd, reapplied, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            answer = reapplied;
            for (unsigned i = 0; i < run; ++i)
                independentSpatialStatus += "; Pass " + std::to_string(pass + i + 1) + ": re-applied";
            pass += run;
        }
        else
        {
            if (!independentPasses[pass])
                independentPasses[pass] = std::make_unique<DlssNr_Dx12>("DLSS-NR independent pass " +
                                                                           std::to_string(pass + 1), encoded.device);
            Config profile = cfg;
            DlssNr::Profiles::ApplyPassConfig(profile, cfg, pass);
            profile.DlssNrReversibleMode = 2u; // Encoded proxy in, reconstructed proxy out; no scene composition.
            profile.DlssNrApplyModel = true;
            profile.DlssNrCompare = 0u;
            profile.DlssNrDebugView = 0u;
            profile.DlssNrSkinProtection = false;
            profile.DlssNrShowSkinMask = false;
            profile.DlssNrHoldFrame = false;
            profile.DlssNrReplaceDetailStrength = 0.0f;
            profile.DlssNrHaloSuppression = 0.0f;
            profile.DlssNrHaloBrightSuppression = 0.0f;
            profile.DlssNrSpatialShowCenter = false;
            profile.DlssNrSpatialShowWork = false;
            auto* target = pass % 2 ? nr.passScratch : nr.output;
            ok = independentPasses[pass]->DispatchConfigured(cmd, source, depth, motion, target, frame, queue,
                                                             &profile);
            if (clampReadable)
            {
                Barrier(cmd, nr.passClamp, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                clampReadable = false;
            }
            if (!ok)
            {
                const auto& child = *independentPasses[pass]->_state;
                const std::string reason = child.nr.failed ? child.nr.reason
                                           : !child.enlargementStatus.empty() ? child.enlargementStatus
                                           : child.nr.spatialFallback ? child.nr.spatialFallbackReason
                                                                      : "waiting for model initialization";
                independentStatus = "Pass " + std::to_string(pass + 1) + ": " + reason;
                break;
            }
            const auto& child = *independentPasses[pass]->_state;
            if (!independentSpatialStatus.empty())
                independentSpatialStatus += "; ";
            independentSpatialStatus += "Pass " + std::to_string(pass + 1) + ": " +
                                        std::to_string(child.nr.workWidth) + "x" + std::to_string(child.nr.workHeight);
            if (child.nr.spatialLayout.requested)
            {
                if (child.nr.spatialFallback)
                    independentSpatialStatus +=
                        std::string(" (spatial fallback: ") + child.nr.spatialFallbackReason + ")";
                else if (!child.nr.spatialLayout.active)
                    independentSpatialStatus += " (spatial unavailable: " + child.nr.spatialLayout.reason + ")";
                else
                    independentSpatialStatus += " (peripheral compression)";
            }
            nr.spatialActive = nr.spatialActive || child.nr.spatialActive;
            Barrier(cmd, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            answer = target;
            lastNetwork = pass++;
        }
        if (pass < passes && !DlssNr::Profiles::PassReapplies(cfg, pass))
        {
            DlssNrConstants clamp {};
            clamp.Mode = DlssNrMode_ClampProxy;
            clamp.Width = nr.width;
            clamp.Height = nr.height;
            ok = shader.DispatchPass(cmd, clamp, answer, nullptr, nullptr, nullptr, nullptr, nr.passClamp, nullptr);
            Barrier(cmd, answer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            answer = nullptr;
            Barrier(cmd, nr.passClamp, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            clampReadable = true;
            source = nr.passClamp;
        }
    }
    ngxTime->End(cmd);

    if (clampReadable)
        Barrier(cmd, nr.passClamp, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (ok && answer && DlssNr_Common::HaloEnabled(cfg) && cfg.DlssNrDebugView.value_or_default() != 2)
    {
        DlssNrConstants halo {};
        halo.Mode = DlssNrMode_SuppressHalo;
        halo.Width = nr.width;
        halo.Height = nr.height;
        DlssNr_Common::SetQualityConstants(halo, cfg);
        if (shader.DispatchPass(cmd, halo, nr.colorCopy, answer, nullptr, nullptr, nullptr, nr.passClamp, nullptr))
        {
            Barrier(cmd, answer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            answer = nr.passClamp;
            Barrier(cmd, answer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }

    bool composed = false;
    if (ok && answer)
    {
        auto resolve = MakeResolveConstants(encoded, passes);
        resolve.Transfer = 0; // Each stage has already reconstructed its answer at native size.
        resolve.ModelWorkScale = 1.0f;
        ID3D12Resource* resolveProxy = nr.colorCopy;
        bool childProxyReadable = false;
        if (resolve.DebugView == 4)
        {
            // Inspect the last pass's actual model input, including its peripheral warp.
            const auto& last = independentPasses[lastNetwork]->_state->nr;
            resolveProxy = last.spatialActive ? last.spatialColor
                           : last.colorSmall ? last.colorSmall : last.colorCopy;
            Barrier(cmd, resolveProxy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            childProxyReadable = true;
            resolve.DebugView = 1;
        }
        if (encoded.targetSupportsUav)
        {
            Barrier(cmd, encoded.target, encoded.targetState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            encoded.targetState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        else
            Barrier(cmd, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        composed = shader.DispatchPass(cmd, resolve, resolveProxy, answer,
                                        encoded.targetSupportsUav ? nr.hdrCopy : encoded.target, encoded.exposure,
                                        nullptr, encoded.targetSupportsUav ? encoded.target : nr.hdrCopy, nullptr);
        if (childProxyReadable)
            Barrier(cmd, resolveProxy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!encoded.targetSupportsUav)
        {
            if (composed)
            {
                Barrier(cmd, nr.hdrCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                Barrier(cmd, encoded.target, encoded.targetState, D3D12_RESOURCE_STATE_COPY_DEST);
                cmd->CopyResource(encoded.target, nr.hdrCopy);
                Barrier(cmd, encoded.target, D3D12_RESOURCE_STATE_COPY_DEST, encoded.targetState);
                Barrier(cmd, nr.hdrCopy, D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            else
                Barrier(cmd, nr.hdrCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        if (composed && captureFrames.isActive())
            captureFrames.record(cmd, encoded.device, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 encoded.target, encoded.targetState);
        if (composed && !passConfig)
            DlssNr::Calibrate::Composed();
    }
    if (answer)
        Barrier(cmd, answer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    nr.reset = !composed;
    modelRunning = composed;
    if (!composed)
    {
        if (independentStatus.empty())
            independentStatus = "Independent pass reconstruction could not complete";
        for (auto& pass : independentPasses)
            if (pass)
                pass->_state->nr.reset = true;
        ReportSkipOnce(independentStatus.c_str());
    }
    return composed;
}

auto DlssNr_Dx12::State::Run(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* colour, ID3D12Resource* depth,
                             ID3D12Resource* motion, ID3D12Resource* output, const DlssNrFrameInfo& frame,
                             ID3D12CommandQueue* timingQueue) -> void
{
    std::lock_guard<std::recursive_mutex> nrLock(mutex);
    // The slider calibrator changes the settings between frames, here, before this frame reads them.
    if (!passConfig)
        DlssNr::Calibrate::Tick(*Config::Instance(), captureFrames);
    const Config& cfg = RunConfig();
    nr.spatialActive = false;

    if (nr.failed || cmdList == nullptr || colour == nullptr || depth == nullptr || motion == nullptr ||
        output == nullptr)
    {
        ReportSkipOnce(nr.failed ? "it already failed this session" : "a resource was missing");
        return;
    }

    ID3D12Resource* target = output;

    // Guard creation and dispatch together: either can record GPU work and alter compute bindings.
    const bool restoreRequired =
        cfg.RestoreComputeSignature.value_or_default() || cfg.RestoreGraphicSignature.value_or_default();
    if (restoreRequired && !frame.IndependentCommands && !D3D12Hooks::CanRestoreRootSignature(cmdList))
    {
        ReportSkipOnce("the upscaler could not restore state this frame");
        return;
    }
    lifetime.Record(cmdList);
    ScopedNrStateEnvelope stateEnvelope(cmdList);

    // A completed upscaler output normally arrives as a UAV. The pre-SR colour input instead arrives
    // readable. Track every transition so both paths return the resource exactly as their caller gave
    // it to us; a pre-SR resource without UAV support is written through a scratch-and-copy fallback.
    const D3D12_RESOURCE_STATES outputArrival =
        frame.PipelineManagedStates ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        : frame.FinishedPicture     ? (D3D12_RESOURCE_STATES) frame.OutputArrivalState
        : frame.BeforeUpscale       ? (!frame.PrivateColorCopy && Config::Instance()->ColorResourceBarrier.has_value()
                                           ? (D3D12_RESOURCE_STATES) Config::Instance()->ColorResourceBarrier.value()
                                           : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        : Config::Instance()->OutputResourceBarrier.has_value()
            ? (D3D12_RESOURCE_STATES) Config::Instance()->OutputResourceBarrier.value()
            : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES targetState = outputArrival;
    const auto TransitionTarget = [&](D3D12_RESOURCE_STATES to)
    {
        Barrier(cmdList, target, targetState, to);
        targetState = to;
    };

    Microsoft::WRL::ComPtr<ID3D12Device> deviceRef;

    if (FAILED(target->GetDevice(IID_PPV_ARGS(&deviceRef))))
    {
        ReportSkipOnce("the output texture belongs to no D3D12 device");
        return;
    }

    auto* device = deviceRef.Get();
    const D3D12_RESOURCE_DESC desc = target->GetDesc();
    const auto active =
        frame.BeforeUpscale
            ? DlssNr::PreSrColorExtent(desc, frame.RenderSubrectWidth, frame.RenderSubrectHeight)
            : std::optional<DlssNr::ColorExtent> { DlssNr::ColorExtent { (unsigned int) desc.Width, desc.Height } };
    if (!active)
    {
        ReportSkipOnce("the pre-SR active colour size is invalid");
        return;
    }
    const auto width = active->width;
    const auto height = active->height;
    const bool cropColor = frame.BeforeUpscale && (width != desc.Width || height != desc.Height);
    const bool targetSupportsUav = cropColor || (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;

    const auto guideDesc = depth->GetDesc();
    const auto motionDesc = motion->GetDesc();
    const auto guides = DlssNr::ResolveGuideRegions(
        { (unsigned int) guideDesc.Width, guideDesc.Height }, { (unsigned int) motionDesc.Width, motionDesc.Height },
        { frame.RenderSubrectWidth, frame.RenderSubrectHeight }, { frame.OutputWidth, frame.OutputHeight },
        frame.MotionVectorsLowResolution, frame.DepthSubrectBaseX, frame.DepthSubrectBaseY, frame.MotionSubrectBaseX,
        frame.MotionSubrectBaseY);
    if (!guides.depth.valid() || !guides.motion.valid())
    {
        ReportSkipOnce("depth or motion-vector subrect is empty");
        return;
    }
    const auto guideWidth = guides.depth.width, guideHeight = guides.depth.height;

    if (frame.Reset)
    {
        nr.reset = true;

        ++resets;

        if (resets <= 3 || resets % 100 == 0)
            LOG_INFO("DLSS-NR: the game asked for a history reset ({} so far)", resets);
    }

    // Guide dimensions can change without rebuilding the model; report changes as they occur.

    const GuideReport guidesNow { true,       frame.DepthInverted, frame.MvScaleX, frame.MvScaleY,
                                  guideWidth, guideHeight,         width,          (unsigned int) height };

    if (loggedGuides != guidesNow)
    {
        loggedGuides = guidesNow;
        LOG_INFO("DLSS-NR guides: depth {}, motion vector scale {} x {}, guides {}x{} for a {}x{} frame",
                 frame.DepthInverted ? "inverted" : "not inverted", frame.MvScaleX, frame.MvScaleY, guideWidth,
                 guideHeight, width, height);
    }

    const unsigned int requestedPasses =
        std::clamp(cfg.DlssNrPasses.value_or_default(), 1u,
                   cfg.DlssNrUnlockPasses.value_or_default() ? DlssNr::MaxPassCount : DlssNr::DefaultMaxPassCount);
    const bool independent = !passConfig && DlssNr::Profiles::HasPassResolutionOverrides(cfg, requestedPasses);
    if (!independent && independentActive)
        ReleaseIndependentPasses();
    independentActive = independent;
    for (auto& model : nr.models)
        model.Collect();
    // (The INT8 models run on the own pass host and need no NR runtime.)
    if (DlssNr::Profiles::NeedsRuntime(cfg, requestedPasses) &&
        ((!NVNGXProxy::IsDx12Inited() && !NVNGXProxy::InitDx12(device)) || !DlssNr::Proxy::Context::Available()))
    {
        nr.failed = true;
        nr.reason = "the NVIDIA NGX driver does not provide Neural Rendering";
        LOG_ERROR("DLSS-NR unavailable: {}", nr.reason);
        return;
    }

    // Only the model runs at working resolution; source and composition remain at native size.
    float workScale = independent ? 1.0f : cfg.DlssNrWorkingScale.value_or_default();
    if (!std::isfinite(workScale))
        workScale = 1.0f;
    workScale = std::clamp(workScale, 0.25f, 2.0f);
    const auto workWidth = (unsigned int) (width * workScale + 0.5f);
    const auto workHeight = (unsigned int) (height * workScale + 0.5f);
    const bool reduced = workWidth != width || workHeight != height;

    auto spatialSettings = DlssNr::Spatial::ReadSettings(cfg);
    if (independent)
        spatialSettings.enabled = false;
    const auto spatialLayout = DlssNr::Spatial::Build(spatialSettings, width, height, workScale);
    const bool spatialSignatureChanged =
        nr.spatialSignatureValid &&
        (nr.spatialLayout != spatialLayout || nr.spatialColorFormat != desc.Format ||
         nr.spatialDepthFormat != guideDesc.Format || nr.spatialMotionFormat != motionDesc.Format ||
         nr.spatialDepthW != guideDesc.Width || nr.spatialDepthH != guideDesc.Height ||
         nr.spatialMotionW != motionDesc.Width || nr.spatialMotionH != motionDesc.Height);
    if (spatialSignatureChanged)
    {
        if (nr.spatialLayout.requested || spatialLayout.requested)
            nr.reset = true;
        nr.spatialFallback = false;
        nr.spatialFallbackReason = "";
    }
    nr.spatialLayout = spatialLayout;
    nr.spatialSignatureValid = true;
    nr.spatialColorFormat = desc.Format;
    nr.spatialDepthFormat = guideDesc.Format;
    nr.spatialMotionFormat = motionDesc.Format;
    nr.spatialDepthW = (unsigned) guideDesc.Width;
    nr.spatialDepthH = guideDesc.Height;
    nr.spatialMotionW = (unsigned) motionDesc.Width;
    nr.spatialMotionH = motionDesc.Height;
    const bool spatial = spatialLayout.active && !nr.spatialFallback;
    if (!spatial && nr.spatialColor)
        ReleaseSpatialResources();
    if (spatial && (!shader.SpatialReady() || !PrepareSpatialResources(device, spatialLayout)))
    {
        nr.spatialFallback = true;
        nr.spatialFallbackReason = "the spatial shader or textures could not be created";
        nr.reset = true;
        modelRunning = false;
        ReportSkipOnce(nr.spatialFallbackReason);
        return;
    }
    const unsigned modelWidth = spatial ? spatialLayout.modelW : workWidth;
    const unsigned modelHeight = spatial ? spatialLayout.modelH : workHeight;
    if (!PrepareRunModels(cmdList, device, frame, desc, { width, height }, { modelWidth, modelHeight }, workScale,
                          requestedPasses, spatial, independent))
        return;
    // The parameter adapter already combined the HDR flag with the active color format.
    const bool isHdrBuffer = frame.ColourIsLinearHdr;

    if (!reportedHdr || reportedHdrValue != isHdrBuffer || reportedBefore != frame.BeforeUpscale)
    {
        reportedHdr = true;
        reportedHdrValue = isHdrBuffer;
        reportedBefore = frame.BeforeUpscale;
        LOG_INFO("DLSS-NR {} SR: the game's DLSS colour space is {} so the colour transform is {}",
                 frame.BeforeUpscale ? "before" : "after", isHdrBuffer ? "linear HDR" : "already tone-mapped",
                 isHdrBuffer ? "on" : "off");
    }

    if (!shader.IsInit())
    {
        nr.failed = true;
        nr.reason = "the colour codec would not compile";
        LOG_ERROR("DLSS-NR unavailable: {}", nr.reason);
        return;
    }

    // Advance capture scheduling only once the codec and models are ready.
    ++frames;
    lifetime.Collect();
    CheckCaptureTrigger();

    if (captureFrames.isActive())
    {
        const auto captureDir = Util::DllPath().remove_filename() / "dlssnr-capture";
        const auto written = captureFrames.write(captureDir);

        if (!written.empty())
            LOG_INFO("DLSS-NR wrote matched before/after frames to {}", written);
    }

    ResTrack_Dx12::HookLateNrQueue(device);
    if (gpuTime == nullptr)
        gpuTime = std::make_unique<DlssNrGpuTime>(device);

    if (ngxTime == nullptr)
        ngxTime = std::make_unique<DlssNrGpuTime>(device);

    gpuTime->Start(cmdList);

    // Copy just the live image, not the stale right/bottom margins. Do this only after model
    // creation/pending-submission early returns, and inside the measured GPU interval. The compact
    // texture lets every existing codec/compare/hold/capture path use unmodified pixel coordinates.
    ID3D12Resource* const gameColor = target;
    if (cropColor)
    {
        TransitionTarget(D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(cmdList, nr.activeColor, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        DlssNr::CopyActiveColor(cmdList, nr.activeColor, gameColor, *active);
        TransitionTarget(outputArrival);
        Barrier(cmdList, nr.activeColor, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        target = nr.activeColor;
        targetState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }

    const auto FinishColor = [&](bool copyBack)
    {
        if (cropColor)
        {
            if (copyBack)
            {
                TransitionTarget(D3D12_RESOURCE_STATE_COPY_SOURCE);
                Barrier(cmdList, gameColor, outputArrival, D3D12_RESOURCE_STATE_COPY_DEST);
                DlssNr::CopyActiveColor(cmdList, gameColor, target, *active);
                Barrier(cmdList, gameColor, D3D12_RESOURCE_STATE_COPY_DEST, outputArrival);
            }
            TransitionTarget(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else
        {
            TransitionTarget(outputArrival);
        }
    };

    EncodeContext encoded { cmdList, device, target, targetState, frame, workScale, targetSupportsUav, spatial };
    EncodeInput(encoded);
    targetState = encoded.targetState;
    if (spatial && !encoded.encodeSucceeded)
    {
        nr.spatialFallback = true;
        nr.spatialFallbackReason = "the spatial frame's colour encode dispatch failed";
        nr.reset = true;
        modelRunning = false;
        Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cmdList, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        FinishColor(false);
        EndGpuTiming(cmdList);
        return;
    }
    auto* modelInput = encoded.modelInput;

    if (independent)
    {
        const bool composed = encoded.encodeSucceeded &&
                              RunIndependentPasses(encoded, depth, motion, requestedPasses, timingQueue);
        targetState = encoded.targetState;
        Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cmdList, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        FinishColor(composed);
        if (composed)
            ++nr.successfulDispatches;
        EndGpuTiming(cmdList);
        return;
    }

    ID3D12Resource* depthIn = ReadableGuide(device, cmdList, depth, &nr.depthClone);
    ID3D12Resource* motionIn = ReadableGuide(device, cmdList, motion, &nr.motionClone);

    if (depthIn == nullptr || motionIn == nullptr)
    {
        nr.reset = true;
        ReportSkipOnce("the game's depth or motion vectors could not be made readable this frame");
        Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cmdList, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        FinishColor(false);
        EndGpuTiming(cmdList);
        if (depthIn == nr.depthClone)
            Barrier(cmdList, nr.depthClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST);
        if (motionIn == nr.motionClone)
            Barrier(cmdList, nr.motionClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST);
        return;
    }
    ID3D12Resource* const originalDepthIn = depthIn;
    ID3D12Resource* const originalMotionIn = motionIn;

    if (spatial)
    {
        const auto colorConstants =
            DlssNr::Spatial::MakeConstants(spatialLayout, 100, guides, frame.MvScaleX, frame.MvScaleY, width, height);
        const auto guideConstants =
            DlssNr::Spatial::MakeConstants(spatialLayout, 101, guides, frame.MvScaleX, frame.MvScaleY, width, height);
        const bool packedColor =
            shader.DispatchSpatial(cmdList, colorConstants, nr.colorCopy, nullptr, nullptr, nr.spatialColor);
        const bool packedGuides = packedColor && shader.DispatchSpatial(cmdList, guideConstants, nr.colorCopy, depthIn,
                                                                        motionIn, nr.spatialDepth, nr.spatialMotion);
        if (!packedGuides)
        {
            nr.spatialFallback = true;
            nr.spatialFallbackReason = "a spatial packing dispatch failed";
            nr.reset = true;
            modelRunning = false;
            ReportSkipOnce(nr.spatialFallbackReason);
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(cmdList, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            FinishColor(false);
            EndGpuTiming(cmdList);
            if (originalDepthIn == nr.depthClone)
                Barrier(cmdList, nr.depthClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST);
            if (originalMotionIn == nr.motionClone)
                Barrier(cmdList, nr.motionClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST);
            return;
        }
        for (auto* packed : { nr.spatialColor, nr.spatialDepth, nr.spatialMotion })
            Barrier(cmdList, packed, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        modelInput = nr.spatialColor;
        depthIn = nr.spatialDepth;
        motionIn = nr.spatialMotion;
    }

    // Below native (and not packed by spatial compression, which brings its own guides), the model
    // was handed a colour at the working size and depth and motion at the frame's size, with only
    // the vector magnitudes rescaled. Nothing says the model resamples a guide that is larger than
    // its colour, and the picture said it does not: every scale below 100% flickered and settled for
    // frames after the camera stopped, while the same model at 100% of a frame the game had already
    // shrunk was steady. So give it guides at its own size -- the same point resample the DLSS
    // enlargement path already builds for its private upscaler -- and describe them as a full,
    // zero-origin region. The vectors keep the game's units; the scale below converts them.
    bool matchedGuides = false;
    if (reduced && !spatial && workWidth < width && cfg.DlssNrMatchGuides.value_or_default())
    {
        if (nr.depthSmall == nullptr)
            nr.depthSmall = CreateScratch(device, DXGI_FORMAT_R32_FLOAT, workWidth, workHeight);
        if (nr.motionSmall == nullptr)
            nr.motionSmall = CreateScratch(device, DXGI_FORMAT_R32G32_FLOAT, workWidth, workHeight);
        if (nr.depthSmall != nullptr && nr.motionSmall != nullptr)
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
            if (shader.DispatchPass(cmdList, resize, depthIn, motionIn, nullptr, nullptr, nullptr, nr.depthSmall,
                                    nr.motionSmall))
            {
                Barrier(cmdList, nr.depthSmall, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(cmdList, nr.motionSmall, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                depthIn = nr.depthSmall;
                motionIn = nr.motionSmall;
                matchedGuides = true;
            }
        }
        static bool loggedMatch = false;
        if (!loggedMatch)
        {
            loggedMatch = true;
            if (matchedGuides)
                LOG_INFO("DLSS-NR guides matched to the working size: depth and motion {}x{} for a {}x{} model "
                         "(the frame's guides are {}x{})",
                         workWidth, workHeight, workWidth, workHeight, guides.depth.width, guides.depth.height);
            else
                LOG_WARN("DLSS-NR guides could not be matched to the working size; the model keeps the frame's "
                         "{}x{} guides for its {}x{} colour",
                         guides.depth.width, guides.depth.height, workWidth, workHeight);
        }
    }

    // The game's scale turns its vectors into pixels of the size they are measured in: the render
    // size for low-resolution vectors, the output size otherwise. The model reads the scale it is
    // given in pixels of the motion texture it is handed, so the conversion is "texture the model
    // reads / size the vectors are measured in" -- the DLSS-enlargement path's formula, whose
    // texture is always its own resample. Here the texture is the matched working-size resample
    // when there is one and the game's own region otherwise. Measuring against the working size
    // in both cases (v0.8.92) was right only with matched guides: at 100% the model kept the
    // game's render-size vectors and got a scale for a frame-size texture -- twice too large in
    // a game upscaling 2x with low-resolution vectors (Onimusha: 3840 where 1920 was right), and
    // 100% flickered where it had been steady. Before v0.8.92 the reference was the frame size,
    // which halved every vector below 100% in the same game; RenderMotionScale=false keeps that
    // for an A/B.
    const bool renderMotionScale = cfg.DlssNrRenderMotionScale.value_or_default();
    const unsigned int mvGuideW = !renderMotionScale ? workWidth : matchedGuides ? workWidth : guides.motion.width;
    const unsigned int mvGuideH = !renderMotionScale ? workHeight : matchedGuides ? workHeight : guides.motion.height;
    const unsigned int mvRefW = !renderMotionScale                 ? width
                                : frame.MotionVectorsLowResolution ? frame.RenderSubrectWidth
                                                                   : frame.OutputWidth;
    const unsigned int mvRefH = !renderMotionScale                 ? height
                                : frame.MotionVectorsLowResolution ? frame.RenderSubrectHeight
                                                                   : frame.OutputHeight;
    const float mvToWorkX = mvRefW != 0 ? (float) mvGuideW / (float) mvRefW : 1.0f;
    const float mvToWorkY = mvRefH != 0 ? (float) mvGuideH / (float) mvRefH : 1.0f;
    {
        static unsigned int loggedRefW = 0, loggedGuideW = 0, loggedWorkW = 0;
        if (loggedRefW != mvRefW || loggedGuideW != mvGuideW || loggedWorkW != workWidth)
        {
            loggedRefW = mvRefW;
            loggedGuideW = mvGuideW;
            loggedWorkW = workWidth;
            LOG_INFO("DLSS-NR model motion scale {:.1f} x {:.1f}: game scale {} x {} measured against {}x{} ({}), "
                     "motion texture {}x{} ({}), model {}x{}",
                     frame.MvScaleX * mvToWorkX, frame.MvScaleY * mvToWorkY, frame.MvScaleX, frame.MvScaleY, mvRefW,
                     mvRefH,
                     !renderMotionScale                 ? "frame size, RenderMotionScale=false"
                     : frame.MotionVectorsLowResolution ? "render size, low-resolution vectors"
                                                        : "output size",
                     mvGuideW, mvGuideH, matchedGuides ? "matched to the working size" : "the game's region", workWidth,
                     workHeight);
        }
    }

    ngxTime->Start(cmdList);

    // Count only a contiguous set of ready, separate feature histories. A failed extra creation never
    // falls back to reusing the main feature: that tells one temporal model several frames elapsed in
    // one game frame and makes its history fight the later layers.
    // A re-applying pass needs no feature: it repeats the last network pass's change in one dispatch.
    unsigned int effectivePasses = 1;
    unsigned int reappliedPasses = 0;
    if (nr.passScratch != nullptr)
    {
        for (unsigned int pass = 1; pass < requestedPasses; ++pass)
        {
            const bool reapply = DlssNr::Profiles::PassReapplies(cfg, pass);
            if (!reapply && !nr.models[pass].Ready(frame.SubmissionEpoch))
                break;
            ++effectivePasses;
            reappliedPasses += reapply;
        }
    }

    if (loggedConfigured != requestedPasses || loggedEffective != effectivePasses ||
        loggedReapplied != reappliedPasses)
    {
        loggedConfigured = requestedPasses;
        loggedEffective = effectivePasses;
        loggedReapplied = reappliedPasses;
        LOG_INFO("DLSS-NR model passes: configured {}, effective {} ({} re-applied, gain {:.3f})", requestedPasses,
                 effectivePasses, reappliedPasses, DlssNr::Profiles::ReapplyGain(cfg));
    }

    // Borrowing (shared resolution, D3D12 only): one cut per network evaluation; the first never borrows.
    NrPassBorrow::Cut cuts[DlssNr::MaxPassCount] {};
    unsigned int evaluations = 0;
    bool borrowing = false;
    for (unsigned int pass = 0; pass < effectivePasses; ++pass)
    {
        if (DlssNr::Profiles::PassReapplies(cfg, pass))
            continue;
        const auto choice = DlssNr::Profiles::PassBorrow(cfg, pass);
        const auto cut = evaluations == 0 ? NrPassBorrow::Cut::None
                                          : NrPassBorrow::ChooseCut(choice.transformer, choice.below, choice.level);
        cuts[evaluations++] = cut;
        borrowing = borrowing || cut != NrPassBorrow::Cut::None;
    }
    static_assert(sizeof(cuts) == sizeof(borrowCuts));
    if (std::memcmp(borrowCuts, cuts, sizeof(cuts)) != 0)
    {
        std::memcpy(borrowCuts, cuts, sizeof(cuts));
        NrPassBorrow::Reset();
    }
    if (borrowing)
        NrPassBorrow::BeginFrame(cmdList, cuts, evaluations);
    unsigned int evaluation = 0;

    // The 4-bit kernels and pass-in-one weight sets (model choices 4..6): loaded when a pass first asks for them.
    // No fallback: a pass that asked for one and was not run by our kernels is not shown (nvfp4Rejected).
    bool nvfp4Rejected = false;
    {
        const bool nvfp4 = DlssNr::Profiles::WantsNvfp4(cfg, effectivePasses);
        DlssNr::Nvfp4::Configure(nvfp4, Util::DllPath().parent_path() / L"OptiScaler" / L"nvfp4");
        static unsigned int nvfp4Frames = 0;
        if (nvfp4 && nvfp4Frames++ % 600 == 30)
            LOG_INFO("DLSS-NR NVFP4 status: {}", DlssNr::Nvfp4::Status());
    }

    // Keep the encoded base immutable; ping-pong model outputs and compose the final delta once.
    ID3D12Resource* passInput = modelInput;
    ID3D12Resource* passOutput = nr.output;
    ID3D12Resource* finalAnswer = nullptr;
    bool outputReadable = false;
    bool scratchReadable = false;
    bool clampReadable = false;
    bool clampFailed = false;
    uint32_t clampSlots[2] = { UINT32_MAX, UINT32_MAX };

    const auto MakeModelReadable = [&](ID3D12Resource* resource)
    {
        bool& readable = resource == nr.output      ? outputReadable
                         : resource == nr.passClamp ? clampReadable
                                                    : scratchReadable;
        if (!readable)
        {
            Barrier(cmdList, resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            readable = true;
        }
    };

    const auto MakeModelWritable = [&](ID3D12Resource* resource)
    {
        bool& readable = resource == nr.output      ? outputReadable
                         : resource == nr.passClamp ? clampReadable
                                                    : scratchReadable;
        if (readable)
        {
            Barrier(cmdList, resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            readable = false;
        }
    };

    int result = NVSDK_NGX_Result_Success;
    const bool enlargementReset = nr.reset;
    bool compositionSucceeded = false;

    DlssNr::Proxy::Frame modelFrame {};
    modelFrame.depth = depthIn;
    modelFrame.motion = motionIn;
    modelFrame.size = { modelWidth, modelHeight };
    modelFrame.guides = (spatial || matchedGuides) ? DlssNr::GuideRegions { { 0, 0, modelWidth, modelHeight },
                                                                            { 0, 0, modelWidth, modelHeight } }
                                                   : guides;
    modelFrame.depthInverted = frame.DepthInverted;
    modelFrame.reset = nr.reset;
    modelFrame.mvScaleX = spatial ? 1.0f : frame.MvScaleX * mvToWorkX;
    modelFrame.mvScaleY = spatial ? 1.0f : frame.MvScaleY * mvToWorkY;

    for (unsigned int pass = 0; pass < effectivePasses && result == NVSDK_NGX_Result_Success;)
    {
        if (const auto run = pass == 0 ? 0 : DlssNr::Profiles::ReapplyRun(cfg, pass, effectivePasses))
        {
            // passInput and finalAnswer are the last network pass's input and answer, both still intact; the
            // target is the other ping-pong texture, so the dispatch never writes what it reads.
            auto* reapplied = finalAnswer == nr.output ? nr.passScratch : nr.output;
            MakeModelWritable(reapplied);
            DlssNrConstants reapply {};
            reapply.Mode = DlssNrMode_Reapply;
            reapply.WhitePoint = float(run) * DlssNr::Profiles::ReapplyGain(cfg);
            reapply.Width = modelWidth;
            reapply.Height = modelHeight;
            if (!shader.DispatchPass(cmdList, reapply, passInput, finalAnswer, nullptr, nullptr, nullptr, reapplied,
                                     nullptr))
            {
                ReportSkipOnce("the re-apply dispatch failed; using the last model answer");
                effectivePasses = pass;
                break;
            }
            finalAnswer = reapplied;
            MakeModelReadable(finalAnswer);
            pass += run;
        }
        else
        {
            MakeModelWritable(passOutput);
            bool evaluated = false;
            modelFrame.color = passInput;
            modelFrame.output = passOutput;
            if (borrowing)
                NrPassBorrow::BeginEvaluation(evaluation);
            // Which kernels and weights this pass runs; told where the evaluation ends whatever its outcome.
            DlssNr::Nvfp4::BeginEvaluation(DlssNr::Profiles::RequestedPreset(cfg, pass));
            result = static_cast<int>(nr.models[pass].Run(cmdList, device, modelFrame, PassSettings(cfg, pass),
                                                          frame.SubmissionEpoch, &evaluated));
            DlssNr::Nvfp4::EndEvaluation(cmdList);
            // A borrowed part of an evaluation runs no kernels of its own.
            const bool wholeEvaluation =
                !borrowing || evaluation >= evaluations || cuts[evaluation] == NrPassBorrow::Cut::None;
            const auto outcome = borrowing ? NrPassBorrow::EndEvaluation() : NrPassBorrow::Outcome::Native;
            ++evaluation;
            modelRunning = evaluated && result == NVSDK_NGX_Result_Success;
            if (!evaluated || result != NVSDK_NGX_Result_Success)
                break;
            if (!DlssNr::Nvfp4::OursRan(wholeEvaluation))
                nvfp4Rejected = true;
            if (outcome == NrPassBorrow::Outcome::Dropped)
            {
                // The rest of this evaluation's kernels never ran: passOutput is stale and this feature's history is
                // partial. Keep the previous answer (the other ping-pong texture; never evaluation 0, whose cut is
                // None) and reset every history next frame. Borrowing stays latched off, so that frame runs natively.
                ReportSkipOnce("pass borrowing failed mid-evaluation; using the previous pass's answer");
                clampFailed = true;
                effectivePasses = pass;
                break;
            }

            finalAnswer = passOutput;
            MakeModelReadable(finalAnswer);
            ++pass;
        }

        // The next network pass reads the clamped answer; a following re-apply reads this pass's input instead.
        if (pass < effectivePasses && !DlssNr::Profiles::PassReapplies(cfg, pass))
        {
            MakeModelWritable(nr.passClamp);
            DlssNrConstants clamp {};
            clamp.Mode = DlssNrMode_ClampProxy;
            clamp.Width = modelWidth;
            clamp.Height = modelHeight;
            // One cached slot per source: re-applied passes break the old pass-parity alternation.
            if (!shader.DispatchPass(cmdList, clamp, finalAnswer, nullptr, nullptr, nullptr, nullptr, nr.passClamp,
                                     nullptr, &clampSlots[finalAnswer == nr.output ? 0 : 1]))
            {
                // Keep this frame's last valid answer; later histories skipped a frame.
                clampFailed = true;
                effectivePasses = pass;
                break;
            }
            MakeModelReadable(nr.passClamp);
            passInput = nr.passClamp;
            passOutput = finalAnswer == nr.output ? nr.passScratch : nr.output;
        }
    }
    if (borrowing)
    {
        NrPassBorrow::EndFrame();
        // Log the borrow status whenever it changes, so a log shows whether borrowing engaged.
        static std::string lastBorrowStatus;
        char status[160] {};
        NrPassBorrow::Status(status, sizeof(status));
        if (lastBorrowStatus != status)
        {
            lastBorrowStatus = status;
            LOG_INFO("DLSS-NR pass borrowing: {}", lastBorrowStatus);
        }
    }
    // A model comparison (Compare with a model as the other side): one pass of the other model on the same input, with
    // pass 1's settings. Its own feature and history; the picture is put together after the resolve below.
    const unsigned int compareModel = passConfig ? 0u : DlssNr::Profiles::CompareModel(cfg);
    auto& compareContext = nr.models[DlssNr::MaxPassCount];
    ID3D12Resource* compareAnswer = nullptr;
    const char* compareNote = "";
    if (compareModel == 0)
    {
        compareContext.RetryAfterFailure();
        ParkNrResource(nr.compareAnswer);
        ParkNrResource(nr.compareColor);
    }
    else if (spatial || borrowing)
        compareNote = "not with peripheral compression or pass borrowing";
    else if (result != NVSDK_NGX_Result_Success || finalAnswer == nullptr)
        compareNote = "the picture's own passes did not run";
    else
    {
        if (nr.compareAnswer == nullptr)
            nr.compareAnswer = CreateScratch(device, nr.output->GetDesc().Format, modelWidth, modelHeight);
        if (nr.compareAnswer == nullptr)
            compareNote = "its texture could not be allocated";
        else
        {
            const bool hadFeature = compareContext.HasFeature();
            bool evaluated = false;
            modelFrame.color = modelInput;
            modelFrame.output = nr.compareAnswer;
            DlssNr::Nvfp4::BeginEvaluation(compareModel);
            const auto compared =
                compareContext.Run(cmdList, device, modelFrame, DlssNr::Profiles::CompareSettings(cfg),
                                   frame.SubmissionEpoch, &evaluated);
            DlssNr::Nvfp4::EndEvaluation(cmdList);
            if (!hadFeature && compareContext.HasFeature())
                NrPassBorrow::Reset();
            if (compared != NVSDK_NGX_Result_Success)
                compareNote = "the other model could not be set up or run (Retry tries again)";
            else if (!evaluated)
                compareNote = "the other model is being set up";
            else if (!DlssNr::Nvfp4::OursRan(true))
                compareNote = "the other model did not run";
            else
            {
                ++evaluation;
                compareAnswer = nr.compareAnswer;
                Barrier(cmdList, compareAnswer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
        }
    }
    unsigned int compareShown = 0;

    // Buffers pass borrowing no longer uses (a resolution change replaces them) and the 4-bit feed-forward buffers of
    // picture sizes no longer in use: this frame's command list and earlier ones may still refer to them, so they are
    // freed only once the GPU has finished everything recorded so far.
    {
        ID3D12Object* retired[16];
        for (unsigned count; (count = NrPassBorrow::TakeRetired(retired, 16)) != 0;)
            for (unsigned i = 0; i < count; ++i)
                lifetime.Retire([object = retired[i]] { object->Release(); });
        if (const auto ticket = DlssNr::Nvfp4::RetireIdle(device, evaluation))
            lifetime.Retire([device, ticket] { DlssNr::Nvfp4::FreeRetired(device, ticket); });
    }

    ngxTime->End(cmdList);

    // The last model has finished reading passClamp. Reuse it for the final
    // chain's halo correction; the base proxy and all model histories stay intact.
    if (result == NVSDK_NGX_Result_Success && finalAnswer && nr.passClamp && DlssNr_Common::HaloEnabled(cfg) &&
        cfg.DlssNrDebugView.value_or_default() != 2)
    {
        MakeModelWritable(nr.passClamp);
        DlssNrConstants halo {};
        halo.Mode = DlssNrMode_SuppressHalo;
        halo.Width = modelWidth;
        halo.Height = modelHeight;
        DlssNr_Common::SetQualityConstants(halo, cfg);
        if (shader.DispatchPass(cmdList, halo, modelInput, finalAnswer, nullptr, nullptr, nullptr, nr.passClamp,
                                nullptr))
        {
            finalAnswer = nr.passClamp;
            MakeModelReadable(finalAnswer);
        }
        else
            ReportSkipOnce("halo filter dispatch failed; using the unfiltered model answer");
    }

    nr.reset = clampFailed || finalAnswer == nullptr;
    if (nvfp4Rejected)
    {
        // The answer is, at least in part, the runtime's own model's. It is not shown: the picture stays unprocessed
        // and the histories start again, so the first frame our kernels do run starts clean.
        ReportSkipOnce("a pass asked for a 4-bit model that did not run; there is no fallback to the original model");
        finalAnswer = nullptr;
        nr.reset = true;
        modelRunning = false;
    }
    bool spatialUnpacked = false;
    ID3D12Resource* ordinaryProxy = modelInput;
    ID3D12Resource* ordinaryAnswer = finalAnswer;
    if (spatial && result == NVSDK_NGX_Result_Success && finalAnswer)
    {
        const auto unpackConstants =
            DlssNr::Spatial::MakeConstants(spatialLayout, 102, guides, frame.MvScaleX, frame.MvScaleY, width, height);
        spatialUnpacked = shader.DispatchSpatial(cmdList, unpackConstants, modelInput, finalAnswer, nullptr,
                                                 nr.spatialProxy, nr.spatialAnswer);
        if (spatialUnpacked)
        {
            for (auto* unpacked : { nr.spatialProxy, nr.spatialAnswer })
                Barrier(cmdList, unpacked, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ordinaryProxy = nr.spatialProxy;
            ordinaryAnswer = nr.spatialAnswer;
        }
        else
        {
            nr.spatialFallback = true;
            nr.spatialFallbackReason = "a spatial unpacking dispatch failed";
            nr.reset = true;
            modelRunning = false;
            ReportSkipOnce(nr.spatialFallbackReason);
            finalAnswer = nullptr;
        }
    }

    // Supersampling probe: report the model working ABOVE native so a test log tells us whether NGX even
    // accepts a super-native evaluate and what it returns. Once per working-size change, or on any error.
    if (modelWidth > width || modelHeight > height)
    {

        if (lastSuper != modelWidth || result != 1)
        {
            lastSuper = modelWidth;
            LOG_INFO("DLSS-NR SUPERSAMPLE: model at {}x{} = {:.2f}x native {}x{}, evaluate result {} ({})", modelWidth,
                     modelHeight, (float) modelWidth / (float) width, width, height, result,
                     NgxResultName((unsigned int) result));
        }
    }

    if (result == NVSDK_NGX_Result_Success && finalAnswer != nullptr)
    {
        // Resolve takes the difference between what the model returned and what it was shown, and adds
        // that back to the frame. At strength zero the result is what the upscaler produced, exactly, and
        // anything the model left alone is untouched rather than round-tripped through the curve.
        auto resolveParams = MakeResolveConstants(encoded, effectivePasses);

        // For spatial supersampling, both halves of the pair use the same filter. A failed paired
        // downsample skips composition this frame so a mismatched proxy cannot create an edit.
        bool superDownOk = false;
        bool spatialDownFailed = false;
        if (spatial && workScale > 1.0f)
        {
            const Scaler scaler = cfg.DlssNrScalingDownscaler.value_or_default();
            if (nr.nrScaler != scaler)
            {
                ReleaseSupersamplers();
                nr.nrScaler = scaler;
            }
            if (nr.superDown == nullptr)
                nr.superDown = new OS_Dx12("DLSS-NR supersample down", device, false, scaler);
            if (nr.spatialProxyDown == nullptr)
                nr.spatialProxyDown = new OS_Dx12("DLSS-NR spatial proxy down", device, false, scaler);
        }
        if (spatial && workScale > 1.0f)
        {
            const bool proxyDown =
                nr.superDown && nr.spatialProxyDown && nr.spatialProxyNative && nr.spatialAnswerNative &&
                nr.spatialProxyDown->DispatchResources(cmdList, ordinaryProxy, nr.spatialProxyNative);
            const bool answerDown =
                proxyDown && nr.superDown->DispatchResources(cmdList, ordinaryAnswer, nr.spatialAnswerNative);
            if (answerDown)
            {
                for (auto* nativePair : { nr.spatialProxyNative, nr.spatialAnswerNative })
                    Barrier(cmdList, nativePair, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                superDownOk = true;
            }
            else
            {
                spatialDownFailed = true;
                nr.spatialFallback = true;
                nr.spatialFallbackReason = "a spatial supersample downsampling dispatch failed";
                nr.reset = true;
                modelRunning = false;
                ReportSkipOnce(nr.spatialFallbackReason);
            }
        }
        else if (!spatial && workScale > 1.0f && nr.superDown != nullptr && nr.outputNative != nullptr &&
                 nr.superDown->DispatchResources(cmdList, ordinaryAnswer, nr.outputNative))
        {
            Barrier(cmdList, nr.outputNative, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            superDownOk = true;
        }

        ID3D12Resource* resolveProxy = superDownOk ? (spatial ? nr.spatialProxyNative : nr.colorCopy) : ordinaryProxy;
        ID3D12Resource* resolveAnswer =
            superDownOk ? (spatial ? nr.spatialAnswerNative : nr.outputNative) : ordinaryAnswer;
        bool enlargementReady = !spatialDownFailed;
        const auto transfer = cfg.DlssNrTransfer.value_or_default();
        bool resizeFieldReadable = false;
        if (resolveParams.DebugView != 4 && !spatialDownFailed && DlssNrUsesDlssEnlargement(transfer) && reduced &&
            (transfer == 2 || workScale < 1.0f))
        {
            auto* enlarged =
                EnlargeMatchedResidual(cmdList, device, ordinaryProxy, ordinaryAnswer, originalDepthIn,
                                       originalMotionIn, frame, resolveParams, enlargementReset, timingQueue);
            enlargementReady = enlarged != nullptr;
            if (enlarged)
            {
                resolveAnswer = enlarged;
                resolveParams.Transfer = transfer;
            }
            if (enlarged && transfer == 4)
            {
                // Retain the spatial field for the inverse-HDR range guard.
                resolveProxy = enlarger->input.Get();
                Barrier(cmdList, resolveProxy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                resizeFieldReadable = true;
            }
            if (enlarged && (resolveParams.DebugView == 2 || (transfer == 4 && resolveParams.DebugView == 1)))
            {
                resolveProxy = ordinaryProxy;
                resolveAnswer = ordinaryAnswer;
                resolveParams.Transfer = DlssNrSpatialTransfer(transfer); // Inspect the actual model pair.
            }
        }
        else
        {
            ReleaseEnlarger();
            enlargementStatus.clear();
        }

        // Reuse proxy display with the immutable input actually passed to NR, before unpacking.
        if (resolveParams.DebugView == 4)
        {
            resolveProxy = modelInput;
            resolveParams.DebugView = 1;
        }

        // Resolve pre-SR inputs without UAV support through an owned scratch and copy-back.
        if (passConfig && resolveParams.Transfer == 1 && !superDownOk && workScale < 1.0f)
            resolveParams.Mode = DlssNrMode_RawResidual;
        ID3D12Resource* resolveOriginal = targetSupportsUav ? nr.hdrCopy : target;
        ID3D12Resource* resolveTarget = targetSupportsUav ? target : nr.hdrCopy;

        if (targetSupportsUav)
        {
            TransitionTarget(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        else
        {
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // The comparison's picture: the same resolve from the other model's answer with the sides exchanged, so the
        // other model's picture lies where the main resolve shows the original one. That part is copied over below.
        bool compareResolved = false, compareDifference = false;
        if (compareAnswer != nullptr && resolveParams.DebugView == 3)
        {
            // The Difference view between the two models: the view shows (answer - model input) x 20 around grey, so
            // with the other model's answer in the input's place it shows this model's answer minus the other's.
            if (!enlargementReady || superDownOk || resolveProxy != ordinaryProxy || resolveAnswer != ordinaryAnswer)
                compareNote = "not with supersampling or DLSS enlargement";
            else
            {
                resolveProxy = compareAnswer;
                resolveParams.CompareMode = 0;
                resolveParams.Transfer = 1;
                compareDifference = true;
            }
        }
        else if (compareAnswer != nullptr)
        {
            if (!enlargementReady || superDownOk || resolveProxy != ordinaryProxy || resolveAnswer != ordinaryAnswer ||
                resolveParams.DebugView != 0 || resolveParams.CompareMode == 0)
                compareNote = "not with this debug view, supersampling or DLSS enlargement";
            else
            {
                if (nr.compareColor == nullptr)
                    nr.compareColor = CreateScratch(device, desc.Format, width, height);
                auto compareParams = resolveParams;
                compareParams.CompareSwap ^= 1u;
                compareResolved = nr.compareColor != nullptr &&
                                  shader.DispatchPass(cmdList, compareParams, resolveProxy, compareAnswer,
                                                      resolveOriginal, encoded.exposure, nullptr, nr.compareColor,
                                                      nullptr);
                if (!compareResolved)
                    compareNote = "its picture could not be made";
            }
        }

        const bool resolved =
            enlargementReady && shader.DispatchPass(cmdList, resolveParams, resolveProxy, resolveAnswer,
                                                    resolveOriginal, encoded.exposure, nullptr, resolveTarget, nullptr);
        compositionSucceeded = resolved;
        if (compareDifference && resolved)
            compareShown = compareModel;
        if (resizeFieldReadable)
            Barrier(cmdList, enlarger->input.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        if (resolved && !targetSupportsUav)
        {
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            const D3D12_RESOURCE_STATES priorTargetState = targetState;
            TransitionTarget(D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->CopyResource(target, nr.hdrCopy);
            TransitionTarget(priorTargetState);
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        else if (!targetSupportsUav)
        {
            // The resolve target was made writable even while private DLSS was
            // warming up. Restore it before the common end-of-frame transition.
            Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        if (compareResolved && resolved)
        {
            // The shader's own rule: the original picture is where (x + 0.5) / width < split, unless swapped.
            const float split =
                resolveParams.CompareMode == 1 ? 0.5f : std::clamp(resolveParams.CompareSplit, 0.0f, 1.0f);
            const auto boundary =
                (unsigned int) std::clamp(std::ceil(split * (float) width - 0.5f), 0.0f, (float) width);
            const bool otherOnLeft = resolveParams.CompareSwap == 0;
            D3D12_BOX box { otherOnLeft ? 0u : boundary, 0u, 0u, otherOnLeft ? boundary : (unsigned int) width,
                            (unsigned int) height, 1u };
            if (box.right > box.left)
            {
                D3D12_TEXTURE_COPY_LOCATION from {}, to {};
                from.pResource = nr.compareColor;
                from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                to.pResource = target;
                to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                Barrier(cmdList, nr.compareColor, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE);
                const D3D12_RESOURCE_STATES priorTargetState = targetState;
                TransitionTarget(D3D12_RESOURCE_STATE_COPY_DEST);
                cmdList->CopyTextureRegion(&to, box.left, 0, 0, &from, &box);
                TransitionTarget(priorTargetState);
                Barrier(cmdList, nr.compareColor, D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            compareShown = compareModel;
        }

        if (superDownOk)
        {
            if (spatial)
            {
                for (auto* nativePair : { nr.spatialProxyNative, nr.spatialAnswerNative })
                    Barrier(cmdList, nativePair, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            else
                Barrier(cmdList, nr.outputNative, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        // Schedule matched proxy/output capture for delayed readback.
        if (captureFrames.isActive())
        {
            captureFrames.record(cmdList, device, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, target,
                                 targetState);
        }
        if (!passConfig)
            DlssNr::Calibrate::Composed();
    }
    else if (result != NVSDK_NGX_Result_Success)
    {
        if (spatial)
        {
            nr.spatialFallback = true;
            nr.spatialFallbackReason = "NGX rejected the packed model input";
            nr.reset = true;
            modelRunning = false;
            for (auto& model : nr.models)
                model.RetryAfterFailure();
            LOG_WARN("DLSS-NR spatial evaluate returned 0x{:X} ({}); trying ordinary NR next frame", (uint32_t) result,
                     NgxResultName((unsigned int) result));
        }
        else
        {
            nr.failed = true;
            nr.reason = "the model refused to run";
            LOG_ERROR("DLSS-NR evaluate returned 0x{:X} ({}); use Retry to recreate the model", (uint32_t) result,
                      NgxResultName((unsigned int) result));
        }
    }

    if (!passConfig)
    {
        const char* note = compareModel != 0 && compareShown == 0 ? compareNote : "";
        if (compareShown != DlssNr::Nvfp4::ComparisonShown() || DlssNr::Nvfp4::ComparisonNote() != note)
        {
            if (compareShown != 0)
                LOG_INFO("DLSS-NR model comparison: the other side shows {}", DlssNr::Nvfp4::ModelName(compareShown));
            else if (compareModel != 0)
                LOG_INFO("DLSS-NR model comparison: {} is not shown: {}", DlssNr::Nvfp4::ModelName(compareModel), note);
            else
                LOG_INFO("DLSS-NR model comparison: off");
        }
        DlssNr::Nvfp4::SetComparison(compareShown, note);
    }
    if (compareAnswer != nullptr)
        Barrier(cmdList, compareAnswer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Restore all intermediate surfaces to the UAV state expected by the next frame.
    MakeModelWritable(nr.output);
    if (nr.passScratch != nullptr)
        MakeModelWritable(nr.passScratch);

    Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    if (nr.passClamp != nullptr)
        MakeModelWritable(nr.passClamp);

    if (spatial)
    {
        for (auto* packed : { nr.spatialColor, nr.spatialDepth, nr.spatialMotion })
            Barrier(cmdList, packed, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (spatialUnpacked)
            for (auto* unpacked : { nr.spatialProxy, nr.spatialAnswer })
                Barrier(cmdList, unpacked, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    // The matched guides were written this frame and read by the model; back to UAV for the next.
    if (matchedGuides)
    {
        Barrier(cmdList, nr.depthSmall, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cmdList, nr.motionSmall, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    // Failed evaluations leave the game's original image intact. A successful copy-back writes
    // only the active rectangle and restores both resources before DLSS consumes the image.
    FinishColor(compositionSucceeded);
    if (compositionSucceeded)
        ++nr.successfulDispatches;
    nr.spatialActive = spatial && compositionSucceeded;

    EndGpuTiming(cmdList);

    // Restore guide clones to COPY_DEST for the next frame's refresh.
    if (originalDepthIn == nr.depthClone)
        Barrier(cmdList, nr.depthClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);

    if (originalMotionIn == nr.motionClone)
        Barrier(cmdList, nr.motionClone, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST);

    if (reduced && !spatial && nr.colorSmall != nullptr)
        Barrier(cmdList, nr.colorSmall, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Leave the staging copy as the next frame expects to find it.
    Barrier(cmdList, nr.colorCopy, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
