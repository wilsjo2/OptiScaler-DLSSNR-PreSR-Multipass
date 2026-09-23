#include "pch.h"
#include "DlssNr_Dx12_State.h"

void DlssNr_Dx12::State::EncodeInput(EncodeContext& context)
{
    const auto& cfg = *Config::Instance();
    const auto& frame = context.frame;
    auto* cmdList = context.cmdList;
    auto* device = context.device;
    auto* target = context.target;
    auto& targetState = context.targetState;
    auto& whitePoint = context.whitePoint;
    auto& modelInput = context.modelInput;
    const auto width = nr.width, height = nr.height;
    const auto workWidth = nr.workWidth, workHeight = nr.workHeight;
    const auto workScale = context.workScale;
    const bool targetSupportsUav = context.targetSupportsUav;
    const bool reduced = workWidth != width || workHeight != height;
    const bool isHdrBuffer = frame.ColourIsLinearHdr;
    const auto TransitionTarget = [&](D3D12_RESOURCE_STATES to)
    {
        Barrier(cmdList, target, targetState, to);
        targetState = to;
    };
    whitePoint =
        frame.WhitePointOverride > 0.0f ? frame.WhitePointOverride : cfg.DlssNrWhitePointScale.value_or_default();

    const bool wasHeld = nr.heldActive;
    // Frame hold. Freeze the encode's input so a live setting change re-renders the same frame. This
    // is self-contained on purpose: it copies the output aside on hold-on and copies it BACK over the
    // live output before the encode reads it while held, so the encode's own path and barriers below
    // are untouched and the default (hold off) is byte-identical. See design/frame-hold.md.
    //
    // `target` is UAV here (normalised at entry). The held copy is
    // left in COPY_SOURCE after capture and stays there for every restore.
    {
        const bool hold = cfg.DlssNrHoldFrame.value_or_default();

        if (hold)
        {
            const D3D12_RESOURCE_DESC td = target->GetDesc();
            const bool needCapture = !nr.heldActive || nr.heldColor == nullptr ||
                                     (unsigned int) td.Width != nr.heldWidth || td.Height != nr.heldHeight ||
                                     td.Format != nr.heldFormat;

            if (needCapture)
            {
                // Hold-on (or the output changed shape under a hold): capture THIS frame, do not
                // restore -- target already holds the frame to freeze, and the pass runs on it.
                if (nr.heldColor != nullptr)
                    ParkNrResource(nr.heldColor);

                nr.heldColor = CreateScratch(device, td.Format, (unsigned int) td.Width, td.Height);

                if (nr.heldColor != nullptr)
                {
                    const D3D12_RESOURCE_STATES priorTargetState = targetState;
                    TransitionTarget(D3D12_RESOURCE_STATE_COPY_SOURCE);
                    Barrier(cmdList, nr.heldColor, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_DEST);
                    cmdList->CopyResource(nr.heldColor, target);
                    Barrier(cmdList, nr.heldColor, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    TransitionTarget(priorTargetState);

                    nr.heldActive = true;
                    nr.heldWidth = (unsigned int) td.Width;
                    nr.heldHeight = td.Height;
                    nr.heldFormat = td.Format;
                    nr.heldWhitePoint = whitePoint;
                }
            }
            else
            {
                // Held: restore the frozen frame onto the live output before the encode reads it.
                const D3D12_RESOURCE_STATES priorTargetState = targetState;
                TransitionTarget(D3D12_RESOURCE_STATE_COPY_DEST);
                cmdList->CopyResource(target, nr.heldColor);
                TransitionTarget(priorTargetState);
            }

            // Keep the captured white point for the held comparison.
            if (nr.heldActive)
            {
                whitePoint = nr.heldWhitePoint;
            }
        }
        else if (nr.heldActive)
        {
            // Released: let go of the frozen frame and resume live input next frame.
            if (nr.heldColor != nullptr)
                ParkNrResource(nr.heldColor);
            nr.heldActive = false;
        }
    }

    const auto source = cfg.DlssNrWhitePointSource.value_or_default();
    auto* gameExposure = static_cast<ID3D12Resource*>(frame.ExposureTexture);
    const bool gameValid = gameExposure && gameExposure->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                           gameExposure->GetDesc().Width == 1 && gameExposure->GetDesc().Height == 1 &&
                           gameExposure->GetDesc().DepthOrArraySize == 1 &&
                           gameExposure->GetDesc().SampleDesc.Count == 1;
    const bool exposureHeld = wasHeld && nr.heldActive && nr.exposureReadable && nr.exposureSource == source;
    if (exposureHeld)
    {
        context.exposure = nr.exposure;
        DlssNr::ExposureConstants(context.exposureConstants, cfg, source, nr.exposurePreExposure);
    }
    if (!exposureHeld && isHdrBuffer && frame.WhitePointOverride <= 0 && (source == 3 || (source == 1 && gameValid)))
    {
        if (!nr.exposure)
            nr.exposure = CreateScratch(device, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 1);
        if (source == 3 && !nr.exposureMeter)
            nr.exposureMeter = CreateScratch(device, DXGI_FORMAT_R32G32B32A32_FLOAT, 64, 64);
        if (nr.exposure && (source != 3 || nr.exposureMeter))
        {
            auto& meter = context.exposureConstants;
            DlssNr::ExposureConstants(meter, cfg, source, frame.PreExposure);
            meter.ExposureSourceWidth = width;
            meter.ExposureSourceHeight = height;
            if (nr.exposureReadable)
                Barrier(cmdList, nr.exposure, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            bool ready = false;
            if (source == 3)
            {
                const auto previous = targetState;
                TransitionTarget(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                meter.Mode = DlssNrMode_Meter;
                meter.Width = meter.Height = 64;
                ready = shader.DispatchPass(cmdList, meter, target, nullptr, nullptr, nullptr, nullptr,
                                            nr.exposureMeter, nullptr);
                TransitionTarget(previous);
                Barrier(cmdList, nr.exposureMeter, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                meter.Mode = DlssNrMode_AutoExposure;
                meter.Width = meter.Height = 1;
                ready = ready && shader.DispatchPass(cmdList, meter, nr.exposureMeter, nullptr, nullptr, nullptr,
                                                     nullptr, nr.exposure, nullptr);
                Barrier(cmdList, nr.exposureMeter, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            else
            {
                const auto prior = static_cast<D3D12_RESOURCE_STATES>(frame.ExposureState);
                Barrier(cmdList, gameExposure, prior, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                meter.Mode = DlssNrMode_Downsample;
                meter.Width = meter.Height = 1;
                ready = shader.DispatchPass(cmdList, meter, gameExposure, nullptr, nullptr, nullptr, nullptr,
                                            nr.exposure, nullptr);
                Barrier(cmdList, gameExposure, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, prior);
            }
            Barrier(cmdList, nr.exposure, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            nr.exposureReadable = true;
            if (ready)
            {
                context.exposure = nr.exposure;
                nr.exposureSource = source;
                nr.exposurePreExposure = meter.PreExposure;
            }
        }
    }

    DlssNrConstants encodeParams = context.exposure ? context.exposureConstants : DlssNrConstants {};
    encodeParams.Mode = DlssNrMode_Encode;
    // A frame that is already display-referred is handed over untouched: the encode becomes a copy and
    // the resolve adds the model's edit back at full scale.
    encodeParams.Passthrough = isHdrBuffer ? 0u : 1u;
    encodeParams.WhitePoint = whitePoint;
    encodeParams.ReversibleMode = cfg.DlssNrReversibleMode.value_or_default();
    encodeParams.Width = width;
    encodeParams.Height = height;

    TransitionTarget(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    context.encodeSucceeded = shader.DispatchPass(cmdList, encodeParams, target, nullptr, nullptr, context.exposure,
                                                  nullptr, nr.colorCopy, nr.hdrCopy);

    if (targetSupportsUav)
        TransitionTarget(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // The transitions double as the wait for the encode's writes.
    Barrier(cmdList, nr.colorCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(cmdList, nr.hdrCopy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Below full resolution the model is shown a filtered shrink of the proxy; the edit it returns is
    // enlarged during the resolve while the frame underneath stays full size and untouched.
    modelInput = nr.colorCopy;

    if (reduced && !context.spatial && nr.colorSmall != nullptr)
    {
        bool built = false;

        if (workScale > 1.0f)
        {
            // Supersample: enlarge the proxy to the larger working size with a real upscaling filter
            // (the Output Scaling upsampler) so the model sees a clean super-native input, rather than
            // the box minifier which only makes sense going down. colorCopy is NON_PIXEL_SHADER_RESOURCE
            // from the encode (SRV-ready); colorSmall is UNORDERED_ACCESS from last frame's resolve.
            // (Re)build the supersample scalers when missing or when the NR downscaler changed (the
            // filter is baked at construction). Both use NR's own DlssNrScalingDownscaler, independent
            // of Output Scaling, so the two can run different filters at once. superDown is built here
            // and used after the model (the down-leg below).
            const Scaler nrScaler = cfg.DlssNrScalingDownscaler.value_or_default();
            if (nr.nrScaler != nrScaler)
            {
                ReleaseSupersamplers();
                nr.nrScaler = nrScaler;
            }
            if (nr.superUp == nullptr)
                nr.superUp = new OS_Dx12("DLSS-NR supersample up", device, true, nrScaler);
            if (nr.superDown == nullptr)
                nr.superDown = new OS_Dx12("DLSS-NR supersample down", device, false, nrScaler);

            if (nr.superUp != nullptr && nr.superUp->DispatchResources(cmdList, nr.colorCopy, nr.colorSmall))
            {
                Barrier(cmdList, nr.colorSmall, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                built = true;
            }
        }

        if (!built)
        {
            if (workScale > 1.0f)
            {
                // Wanted to supersample but the upscaler was not available -- warn once; the box path
                // below can only enlarge blockily, so the user should know the clean path is off.

                if (!warnedSuper)
                {
                    warnedSuper = true;
                    LOG_WARN("DLSS-NR supersample: upscaler unavailable, falling back to a blocky enlarge.");
                }
            }

            // Sub-native (or the upsampler could not be built): box-resample the proxy to the work size.
            DlssNrConstants down {};
            down.Mode = DlssNrMode_Downsample;
            down.Width = workWidth;
            down.Height = workHeight;
            shader.DispatchPass(cmdList, down, modelInput, nullptr, nullptr, nullptr, nullptr, nr.colorSmall, nullptr);
            Barrier(cmdList, nr.colorSmall, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        modelInput = nr.colorSmall;
    }
}

DlssNrConstants DlssNr_Dx12::State::MakeResolveConstants(const EncodeContext& context, unsigned int effectivePasses)
{
    const auto& cfg = *Config::Instance();
    const auto whitePoint = context.whitePoint;
    const auto width = nr.width, height = nr.height;
    const bool isHdrBuffer = context.frame.ColourIsLinearHdr;
    DlssNrConstants resolveParams = context.exposure ? context.exposureConstants : DlssNrConstants {};
    resolveParams.Mode = DlssNrMode_Resolve;
    resolveParams.WhitePoint = whitePoint;
    resolveParams.Width = width;
    resolveParams.Height = height;
    resolveParams.TransferStrength = cfg.DlssNrTransferStrength.value_or_default();
    const auto strength = [](float v) { return std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 1.0f; };
    resolveParams.SkinProtection = cfg.DlssNrSkinProtection.value_or_default();
    resolveParams.ShowSkinMask = cfg.DlssNrShowSkinMask.value_or_default();
    resolveParams.SkinDetail = strength(cfg.DlssNrSkinDetail.value_or_default());
    resolveParams.SkinColour = strength(cfg.DlssNrSkinColour.value_or_default());
    resolveParams.EnvironmentDetail = strength(cfg.DlssNrEnvironmentDetail.value_or_default());
    resolveParams.EnvironmentColour = strength(cfg.DlssNrEnvironmentColour.value_or_default());
    resolveParams.ColourStrength = cfg.DlssNrColourStrength.value_or_default();
    resolveParams.DebugView = cfg.DlssNrDebugView.value_or_default();
    resolveParams.MaxRatio = cfg.DlssNrMaxRatio.value_or_default();
    resolveParams.Transfer = DlssNrSpatialTransfer(cfg.DlssNrTransfer.value_or_default());
    resolveParams.DebugScale = cfg.DlssNrWhitePointScale.value_or_default();
    resolveParams.Passthrough = isHdrBuffer ? 0u : 1u;
    resolveParams.ReversibleMode = cfg.DlssNrReversibleMode.value_or_default();
    resolveParams.ApplyModel = cfg.DlssNrApplyModel.value_or_default() ? 1u : 0u;
    resolveParams.CompareMode = cfg.DlssNrCompare.value_or_default();
    resolveParams.CompareSplit = cfg.DlssNrCompareSplit.value_or_default();
    resolveParams.CompareZoom = std::max(1.0f, cfg.DlssNrCompareZoom.value_or_default());
    resolveParams.CompareSwap = cfg.DlssNrCompareSwap.value_or_default() ? 1u : 0u;

    resolveParams.ReplaceDetailStrength = cfg.DlssNrReplaceDetailStrength.value_or_default();
    resolveParams.ModelWorkScale = context.workScale;

    // Report the effective composition settings when they change.

    // Quantised to the precision it is printed at. Comparing raw floats logged 2376 lines in one
    // Enshrouded session, because a measured white point drifts continuously and every drift was a
    // change. A line per meaningful change is the point; a line per frame is a different problem.
    const ComposeReport composeNow { true,
                                     std::round(resolveParams.WhitePoint * 100.0f) / 100.0f,
                                     resolveParams.TransferStrength,
                                     resolveParams.ColourStrength,
                                     resolveParams.MaxRatio,
                                     resolveParams.Passthrough,
                                     resolveParams.DebugView,
                                     resolveParams.CompareMode,
                                     resolveParams.Transfer,
                                     nr.workWidth,
                                     nr.workHeight,
                                     effectivePasses };

    if (loggedCompose != composeNow)
    {
        loggedCompose = composeNow;
        LOG_INFO("DLSS-NR composition: paper white {:.2f}x, detail {:.2f}, colour {:.2f}, guard "
                 "{:.1f}x, colour transform {}, transfer {}, model {}x{}, passes {}, debug view {}, compare {}",
                 composeNow.whitePoint, composeNow.transfer, composeNow.colour, composeNow.maxRatio,
                 composeNow.passthrough != 0 ? "off (frame already tone mapped)" : "on (linear HDR)",
                 composeNow.residual == 3   ? "lighting + colour"
                 : composeNow.residual == 1 ? "matched residual"
                                            : "classic",
                 composeNow.workW, composeNow.workH, composeNow.passes, composeNow.debugView, composeNow.compareMode);
    }

    return resolveParams;
}
