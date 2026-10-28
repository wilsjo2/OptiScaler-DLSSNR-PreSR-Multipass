#include "pch.h"
#include "DlssNr_Dx12_State.h"

// Denoise first.
//
// Measured on The Witcher 3 with path tracing (2026-10-04): shown the noisy, jittered pre-RR frame,
// NR keeps only a coarse tone and colour grade and its fine detail is tiny and changes every frame,
// because the model re-decides detail on every new noise pattern. Ray Reconstruction passes the
// coarse edit through untouched, so the denoiser is not the problem; the input is. This context
// gives the model a clean, steady image at render resolution by running the game's own upscaler
// a second time at a 1:1 ratio, and then offers three ways to carry the result to the screen.

namespace
{
unsigned UInt(NVSDK_NGX_Parameter* p, const char* key, unsigned fallback = 0)
{
    unsigned value = fallback;
    p->Get(key, &value);
    return value;
}

float Float(NVSDK_NGX_Parameter* p, const char* key, float fallback)
{
    float value = fallback;
    p->Get(key, &value);
    return std::isfinite(value) ? value : fallback;
}

constexpr unsigned kRebuildFlags = NVSDK_NGX_DLSS_Feature_Flags_DepthInverted | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                   NVSDK_NGX_DLSS_Feature_Flags_MVJittered | NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                                   NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

const char* StepName(int step)
{
    switch (step)
    {
    case 0:
        return "NR'd clean image -> game upscaler (jitter 0)";
    case 1:
        return "NR'd clean image -> private DLSS SR (jitter 0)";
    default:
        return "edit onto raw render -> game upscaler";
    }
}

const char* KernelName(int kernel)
{
    switch (kernel)
    {
    case 0:
        return "bilinear";
    case 2:
        return "Lanczos 2";
    default:
        return "Catmull-Rom";
    }
}

} // namespace

auto DlssNr_Dx12::State::DenoiseFirstContext::Say(const std::string& text) -> void
{
    if (status == text)
        return;
    status = text;
    LOG_INFO("DLSS-NR denoise first: {}", text);
}

auto DlssNr_Dx12::State::DenoiseFirstContext::Cancel() -> void
{
    pending = {};
    if (current)
        current->reset = true;
}

auto DlssNr_Dx12::State::DenoiseFirstContext::RetireCurrent() -> void
{
    if (!current)
        return;
    auto* retired = current.release();
    ++retiredCount;
    lifetime.Retire(
        [this, retired]
        {
            delete retired;
            --retiredCount;
        });
    lifetime.BeginGeneration();
}

auto DlssNr_Dx12::State::DenoiseFirstContext::Idle(bool active) -> void
{
    lifetime.Collect(); // retired generations are freed here once the mode is off
    if (active || !current)
        return;
    Cancel();
    RetireCurrent();
    lastDenoiseTime.reset();
    lastEnlargeTime.reset();
    Say("off");
}

auto DlssNr_Dx12::State::DenoiseFirstContext::ReleaseResources() -> void
{
    Cancel();
    RetireCurrent();
    lifetime.Collect();
}

namespace
{
// A scratch texture that can stand in for the game's Color: the game's own flags plus UAV, so it can
// be transitioned to whatever state the game declares for its colour buffer. Created in UAV like
// State::CreateScratch, so the first-use transition below treats both kinds alike.
ID3D12Resource* CreateColorStandIn(ID3D12Device* device, DXGI_FORMAT format, unsigned w, unsigned h,
                                   unsigned colorFlags)
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = (D3D12_RESOURCE_FLAGS) ((colorFlags | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) &
                                         ~D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
    ID3D12Resource* res = nullptr;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                    IID_PPV_ARGS(&res));
    return res;
}
} // namespace

auto DlssNr_Dx12::State::DenoiseFirstContext::Allocate(Generation& g) -> bool
{
    // The step-specific textures (composite, upscaled) and step 1's upscaler are made on demand, so
    // switching steps in the menu keeps the 1:1 feature and its history.
    g.clean = owner.CreateScratch(g.device, g.inputFormat, g.w, g.h);
    g.edited = CreateColorStandIn(g.device, g.inputFormat, g.w, g.h, g.colorFlags);
    g.exposure = owner.CreateScratch(g.device, DXGI_FORMAT_R32_FLOAT, 1, 1);
    if (!g.clean || !g.edited || !g.exposure)
        return false;
    if (timerDevice != g.device)
    {
        denoiseTime.reset();
        enlargeTime.reset();
        timerDevice = g.device;
    }
    if (!denoiseTime)
        denoiseTime = std::make_unique<DlssNrGpuTime>(g.device);
    if (!enlargeTime)
        enlargeTime = std::make_unique<DlssNrGpuTime>(g.device);
    return true;
}

auto DlssNr_Dx12::State::DenoiseFirstContext::Before(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source,
                                                     uint32_t featureFlags, unsigned long long submittedEpoch,
                                                     ID3D12CommandQueue* queue, bool rayReconstruction)
    -> DenoiseFirstHandoff
{
    DenoiseFirstHandoff handoff {};
    if (pending.cmd && current)
        current->reset = true; // the previous After never ran: the game's upscale was abandoned
    pending = {};
    struct ResetOnGap
    {
        DenoiseFirstContext& state;
        ~ResetOnGap()
        {
            if (!state.pending.cmd && state.current)
                state.current->reset = true;
        }
    } resetOnGap { *this };
    lifetime.Collect();
    const auto& cfg = *Config::Instance();
    if (cfg.DlssNrDebugView.value_or_default() != 0 || cfg.DlssNrCompare.value_or_default() != 0 ||
        cfg.DlssNrShowSkinMask.value_or_default())
    {
        Say("Disable the Debug view, Compare and Show skin mask first.");
        return handoff;
    }
    if (cmd->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT ||
        ((cfg.RestoreComputeSignature.value_or_default() || cfg.RestoreGraphicSignature.value_or_default()) &&
         !D3D12Hooks::CanRestoreRootSignature(cmd)))
    {
        Say("inactive: requires a direct command list with restorable game state");
        return handoff;
    }
    auto* color = owner.GetResource(source, NVSDK_NGX_Parameter_Color, "DLSSD.Color");
    auto* output = owner.GetResource(source, NVSDK_NGX_Parameter_Output, "DLSSD.Output");
    auto* depth = owner.GetResource(source, NVSDK_NGX_Parameter_Depth, "DLSSD.Depth");
    auto* motion = owner.GetResource(source, NVSDK_NGX_Parameter_MotionVectors, "DLSSD.MotionVectors");
    if (!color || !output || !depth || !motion || color == output)
    {
        Say("inactive: distinct Color/Output, depth and motion are required");
        return handoff;
    }
    for (const char* key :
         { NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y,
           NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y,
           NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y,
           NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y })
        if (UInt(source, key) != 0)
        {
            Say("inactive: non-zero colour/guide/output offsets");
            return handoff;
        }
    const auto inDesc = color->GetDesc(), outDesc = output->GetDesc();
    const auto active =
        DlssNr::PreSrColorExtent(inDesc, UInt(source, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width),
                                 UInt(source, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height));
    if (!active || !DlssNr::PreSrColorExtent(outDesc, 0, 0) || inDesc.MipLevels != 1 || active->width > outDesc.Width ||
        active->height > outDesc.Height)
    {
        Say("inactive: unsupported active input/output dimensions");
        return handoff;
    }
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))))
        return handoff;

    const unsigned flags = (featureFlags ? featureFlags : UInt(source, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags));
    const unsigned key = flags & kRebuildFlags;
    if (!(flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes))
    {
        // The private frame has no motion-vector extent of its own; a 1:1 feature expects render-sized
        // motion, and display-sized vectors would be read misaligned.
        device->Release();
        Say("inactive: this game supplies display-resolution motion vectors; the 1:1 pass needs render-resolution "
            "ones");
        return handoff;
    }
    const int step = std::clamp(cfg.DlssNrDenoiseFirstStep.value_or_default(), 0, 2);
    auto rrInputs = rayReconstruction ? DlssNr::PrivateUpscalerDx12::ReadRrInputs(source, active->width, active->height)
                                      : DlssNr::PrivateRrInputsDx12 {};
    const bool privateRr = rrInputs.valid;
    const int quality = (int) UInt(source, NVSDK_NGX_Parameter_PerfQualityValue, NVSDK_NGX_PerfQuality_Value_MaxPerf);
    if (current &&
        (current->rayReconstruction != rayReconstruction || current->privateRr != privateRr ||
         (privateRr && (current->rr.roughnessMode != rrInputs.roughnessMode ||
                        current->rr.hardwareDepth != rrInputs.hardwareDepth)) ||
         current->device != device || current->w != active->width || current->h != active->height ||
         current->outW != outDesc.Width || current->outH != outDesc.Height || current->inputFormat != inDesc.Format ||
         current->outputFormat != outDesc.Format || current->flags != key ||
         current->colorFlags != (unsigned) inDesc.Flags || current->quality != quality))
        RetireCurrent();
    if (!current)
    {
        if (retiredCount >= 4)
        {
            device->Release();
            Say("waiting for retired GPU work");
            return handoff;
        }
        current = std::make_unique<Generation>();
        current->device = device; // takes the GetDevice reference
        current->w = active->width;
        current->h = active->height;
        current->outW = (unsigned) outDesc.Width;
        current->outH = outDesc.Height;
        current->inputFormat = inDesc.Format;
        current->outputFormat = outDesc.Format;
        current->flags = key;
        current->colorFlags = (unsigned) inDesc.Flags;
        current->step = step;
        current->quality = quality;
        current->rayReconstruction = rayReconstruction;
        current->privateRr = privateRr;
        if (!Allocate(*current))
        {
            current->failed = true;
            Say("allocation failed");
            return handoff;
        }
        LOG_INFO("DLSS-NR denoise first: generation {}x{} -> {}x{}, 1:1 pass {}, step {}", current->w, current->h,
                 current->outW, current->outH, privateRr ? "DLSS RR" : "DLSS SR", StepName(step));
        if (rayReconstruction && !privateRr)
            LOG_WARN("DLSS-NR denoise first: the game runs RR but its guide buffers are not readable here; the 1:1 "
                     "pass falls back to DLSS SR, which does not denoise path tracing");
    }
    else
        device->Release();
    auto& g = *current;
    g.rr = rrInputs; // own the matrix values before the game's evaluate can rewrite its table
    const auto inputStates = DlssNr::ResolveInputStates_Dx12(false);
    const auto arrival = inputStates.color;
    for (auto& guide : g.rr.guides)
    {
        if (guide.resource == color)
            guide.state = arrival;
        else if (guide.resource == depth)
            guide.state = inputStates.depth;
        else if (guide.resource == motion)
            guide.state = inputStates.motion;
    }
    if (g.failed)
        return handoff;
    owner.lifetime.Record(cmd);
    lifetime.Record(cmd);

    const bool hdr = (flags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0;
    const bool autoExposure = (flags & NVSDK_NGX_DLSS_Feature_Flags_AutoExposure) != 0;
    if (!g.denoiser)
    {
        ScopedNrStateEnvelope envelope(cmd);
        DlssNr::PrivateUpscalerCreateDx12 info {};
        info.width = g.w;
        info.height = g.h;
        info.outputWidth = g.w;
        info.outputHeight = g.h;
        info.quality = NVSDK_NGX_PerfQuality_Value_DLAA;
        info.depthInverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
        info.jitteredMotion = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered) != 0;
        info.lowResolutionMotion = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) != 0;
        info.rayReconstruction = g.privateRr;
        info.roughnessMode = g.rr.roughnessMode;
        info.hardwareDepth = g.rr.hardwareDepth;
        info.hdr = hdr;
        info.autoExposure = autoExposure;
        g.denoiser = std::make_unique<DlssNr::PrivateUpscalerDx12>(DlssNr::PrivateUpscaler::DLSS);
        LOG_INFO("DLSS-NR denoise first: creating 1:1 {} {}x{}, inverted {}, jittered MV {}, low-res MV {}, HDR {}, "
                 "auto exposure {}",
                 g.privateRr ? "DLSS RR" : "DLSS SR", g.w, g.h, info.depthInverted, info.jitteredMotion,
                 info.lowResolutionMotion, hdr, autoExposure);
        if (!g.denoiser->Init(g.device, cmd, info))
        {
            g.failed = true;
            Say(std::string("1:1 ") + (g.privateRr ? "RR" : "DLSS") + " creation failed: " + g.denoiser->Error());
            return handoff;
        }
        DlssNrConstants unit {};
        unit.Mode = DlssNrMode_UnitExposure;
        unit.Width = unit.Height = 1;
        if (!owner.shader.DispatchPass(cmd, unit, g.clean, nullptr, nullptr, nullptr, nullptr, g.exposure, nullptr))
        {
            g.failed = true;
            Say("private exposure initialization failed");
            return handoff;
        }
        // Everything private rests readable between uses; the first frame moves it out of its creation state.
        for (auto* r : { g.clean, g.edited, g.composite, g.upscaled, g.exposure })
            if (r)
                owner.Barrier(cmd, r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        g.readable = true;
        g.createEpoch = submittedEpoch;
        Say(std::string("1:1 ") + (g.privateRr ? "RR" : "DLSS") + " created; waiting for a later submission epoch");
        return handoff;
    }
    if (submittedEpoch == g.createEpoch)
        return handoff; // the creation command list has not provably been submitted yet

    // Step-specific resources, made the first time a step asks for them. A step change keeps the
    // 1:1 feature and its history; only the new step's own textures and upscaler are added.
    g.step = step;
    if (step == EditOntoRaw && !g.composite)
    {
        g.composite = CreateColorStandIn(g.device, g.inputFormat, g.w, g.h, g.colorFlags);
        if (!g.composite)
        {
            g.failed = true;
            Say("allocation failed");
            return handoff;
        }
        owner.Barrier(cmd, g.composite, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if (step == PrivateSr && !g.upscaled)
    {
        g.upscaled = owner.CreateScratch(g.device, g.outputFormat, g.outW, g.outH);
        if (!g.upscaled)
        {
            g.failed = true;
            Say("allocation failed");
            return handoff;
        }
        owner.Barrier(cmd, g.upscaled, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if (step == PrivateSr && !g.enlarger)
    {
        ScopedNrStateEnvelope envelope(cmd);
        DlssNr::PrivateUpscalerCreateDx12 up {};
        up.width = g.w;
        up.height = g.h;
        up.outputWidth = g.outW;
        up.outputHeight = g.outH;
        up.quality = g.quality;
        up.depthInverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
        up.jitteredMotion = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered) != 0;
        up.lowResolutionMotion = true;
        up.hdr = hdr;
        up.autoExposure = autoExposure;
        g.enlarger = std::make_unique<DlssNr::PrivateUpscalerDx12>(DlssNr::PrivateUpscaler::DLSS);
        if (!g.enlarger->Init(g.device, cmd, up))
        {
            g.failed = true;
            Say("private DLSS SR creation failed: " + g.enlarger->Error());
            return handoff;
        }
        g.enlargerEpoch = submittedEpoch;
        Say("private DLSS SR created; waiting for a later submission epoch");
        return handoff;
    }
    if (step == PrivateSr && submittedEpoch == g.enlargerEpoch)
        return handoff;

    // Frame parameters shared by the private passes and NR.
    auto* exposureTexture = GetUpscalerResource_Dx12(source, NVSDK_NGX_Parameter_ExposureTexture);
    const auto exposureState =
        (D3D12_RESOURCE_STATES) cfg.ExposureResourceBarrier.value_or(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const float preExposure = std::max(Float(source, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1), 1e-4f);
    const float exposureScale = Float(source, NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1);
    const float jitterX = Float(source, NVSDK_NGX_Parameter_Jitter_Offset_X, 0);
    const float jitterY = Float(source, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0);
    float mvScaleX = Float(source, NVSDK_NGX_Parameter_MV_Scale_X, 1);
    float mvScaleY = Float(source, NVSDK_NGX_Parameter_MV_Scale_Y, 1);
    if (mvScaleX == 0)
        mvScaleX = 1;
    if (mvScaleY == 0)
        mvScaleY = 1;
    const bool reset = UInt(source, NVSDK_NGX_Parameter_Reset) != 0 || g.reset;
    const float frameTime = Float(source, NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, 16.67f);

    DlssNr::PrivateUpscalerFrameDx12 f {};
    f.color = { color, arrival };
    f.depth = { depth, depth == color ? arrival : inputStates.depth };
    f.motion = { motion, motion == color ? arrival : inputStates.motion };
    if (exposureTexture)
        f.exposure = { exposureTexture, exposureState };
    else if (!autoExposure)
        f.exposure = { g.exposure, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE };
    f.output = { g.clean, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE };
    f.width = g.w;
    f.height = g.h;
    f.outputWidth = g.w;
    f.outputHeight = g.h;
    f.reset = reset;
    f.jitterX = jitterX; // the raw render is jittered, and the 1:1 pass must know by how much
    f.jitterY = jitterY;
    f.motionScaleX = mvScaleX;
    f.motionScaleY = mvScaleY;
    f.frameTimeMs = frameTime;
    f.preExposure = preExposure;
    f.exposureScale = exposureScale;
    f.rr = g.rr;

    // Step one: the game's own upscaler at 1:1. Jittered noisy samples in, clean steady image out.
    {
        ScopedNrStateEnvelope envelope(cmd);
        denoiseTime->Start(cmd);
        const bool ok = g.denoiser->Evaluate(cmd, f);
        denoiseTime->End(cmd);
        if (auto ms = denoiseTime->ReadGpuTime())
            lastDenoiseTime = ms;
        if (!ok)
        {
            g.failed = true;
            Say(std::string("1:1 ") + (g.privateRr ? "RR" : "DLSS") + " evaluation failed: " + g.denoiser->Error());
            return handoff;
        }
    }

    // Step two: NR on the clean image. It edits a copy in place, so the clean image survives for
    // the difference in step three and for the capture.
    owner.Barrier(cmd, g.clean, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    owner.Barrier(cmd, g.edited, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    DlssNr::CopyActiveColor(cmd, g.edited, g.clean, { g.w, g.h });
    owner.Barrier(cmd, g.clean, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    owner.Barrier(cmd, g.edited, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    DlssNrFrameInfo frame {};
    frame.BeforeUpscale = frame.PrivateColorCopy = true;
    frame.RayReconstruction = rayReconstruction;
    frame.SubmissionEpoch = submittedEpoch;
    frame.RenderSubrectWidth = g.w;
    frame.RenderSubrectHeight = g.h;
    frame.OutputWidth = g.outW;
    frame.OutputHeight = g.outH;
    frame.MotionVectorsLowResolution = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) != 0;
    frame.DepthInverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
    frame.ColourIsLinearHdr = hdr && DlssNr::FormatCanHoldLinearHdr(inDesc.Format);
    frame.Reset = reset;
    frame.MvScaleX = mvScaleX;
    frame.MvScaleY = mvScaleY;
    frame.ExposureTexture = exposureTexture;
    frame.ExposureState = exposureState;
    frame.PreExposure = preExposure;
    const auto before = owner.nr.successfulDispatches;
    {
        struct RestoreGuides
        {
            State& owner;
            ID3D12GraphicsCommandList* cmd;
            std::vector<std::pair<ID3D12Resource*, D3D12_RESOURCE_STATES>> resources;
            void Read(ID3D12Resource* resource, D3D12_RESOURCE_STATES state)
            {
                if (!resource || std::any_of(resources.begin(), resources.end(),
                                             [resource](const auto& entry) { return entry.first == resource; }))
                    return;
                resources.emplace_back(resource, state);
                owner.Barrier(cmd, resource, state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            ~RestoreGuides()
            {
                for (auto it = resources.rbegin(); it != resources.rend(); ++it)
                    owner.Barrier(cmd, it->first, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, it->second);
            }
        } restore { owner, cmd };
        restore.Read(depth, depth == color ? arrival : inputStates.depth);
        restore.Read(motion, motion == color ? arrival : inputStates.motion);
        owner.Run(cmd, g.edited, depth, motion, g.edited, frame, queue);
    }
    if (owner.nr.successfulDispatches == before)
    {
        g.reset = true;
        Say("waiting for NR evaluation; the game's upscale keeps its raw input");
        return handoff;
    }

    if (owner.pipelineCapture)
    {
        owner.pipelineCapture->metadata << "denoise_first step " << g.step << " one_to_one "
                                        << (g.privateRr ? "RR" : "SR") << " kernel "
                                        << cfg.DlssNrDenoiseFirstKernel.value_or_default() << " edit "
                                        << cfg.DlssNrDenoiseFirstEdit.value_or_default() << " shift "
                                        << cfg.DlssNrDenoiseFirstShift.value_or_default() << " flip "
                                        << cfg.DlssNrDenoiseFirstFlipJitter.value_or_default() << " clamp "
                                        << cfg.DlssNrDenoiseFirstNeighbourhoodClamp.value_or_default() << '\n';
        owner.pipelineCapture->Copy(cmd, g.device, "clean", g.clean, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        owner.pipelineCapture->Copy(cmd, g.device, "nr_clean", g.edited,
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    // Step three.
    std::string detail;
    if (g.step == GameUpscaler)
    {
        // Lent in the state the game declares for its own colour, since the upscaler backends and
        // the capture barrier from that state; After() takes it back to readable.
        owner.Barrier(cmd, g.edited, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, arrival);
        pending.handedOff = g.edited;
        pending.handedState = arrival;
        handoff.color = g.edited;
        handoff.zeroJitter = true;
    }
    else if (g.step == PrivateSr)
    {
        ScopedNrStateEnvelope envelope(cmd);
        auto up = f;
        up.color = { g.edited, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE };
        up.output = { g.upscaled, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE };
        up.outputWidth = g.outW;
        up.outputHeight = g.outH;
        up.jitterX = up.jitterY = 0; // the clean image carries no jitter
        up.rr = DlssNr::PrivateRrInputsDx12 {};
        enlargeTime->Start(cmd);
        const bool ok = g.enlarger->Evaluate(cmd, up);
        enlargeTime->End(cmd);
        if (auto ms = enlargeTime->ReadGpuTime())
            lastEnlargeTime = ms;
        if (!ok)
        {
            g.failed = true;
            Say("private DLSS SR evaluation failed: " + g.enlarger->Error());
            return handoff;
        }
        handoff.replaceOutput = true;
    }
    else
    {
        // Measured on The Witcher 3 (REDengine) 2026-10-05: the raw render correlates with the clean
        // image shifted by MINUS the NGX jitter offset on every captured frame, and plus is worse
        // than no shift at all. So the default direction is negative; "flip" selects positive.
        const bool shift = cfg.DlssNrDenoiseFirstShift.value_or_default();
        const float sign = cfg.DlssNrDenoiseFirstFlipJitter.value_or_default() ? 1.0f : -1.0f;
        DlssNrConstants c {};
        c.Mode = 0;
        c.Width = g.w;
        c.Height = g.h;
        c.TransferStrength = 1.0f;
        c.DebugView = (uint32_t) std::clamp(cfg.DlssNrDenoiseFirstKernel.value_or_default(), 0, 2);
        c.MaxRatio = std::clamp(cfg.DlssNrMaxRatio.value_or_default(), 1.0f, 8.0f);
        c.MvScaleX = shift ? sign * jitterX : 0.0f;
        c.MvScaleY = shift ? sign * jitterY : 0.0f;
        c.CompareMode = cfg.DlssNrDenoiseFirstEdit.value_or_default() == 1 ? 1u : 0u;
        c.CompareSwap = cfg.DlssNrDenoiseFirstNeighbourhoodClamp.value_or_default() ? 1u : 0u;
        c.Passthrough = cfg.DlssNrDenoiseFirstFireflyGuard.value_or_default() ? 1u : 0u;
        ScopedNrStateEnvelope envelope(cmd);
        owner.Barrier(cmd, color, arrival, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        owner.Barrier(cmd, g.composite, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const bool ok = owner.shader.DispatchDenoiseFirstPass(cmd, c, color, g.edited, g.clean, g.composite);
        owner.Barrier(cmd, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, arrival);
        if (!ok)
        {
            owner.Barrier(cmd, g.composite, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            g.reset = true;
            Say("edit composition failed; the game's upscale keeps its raw input");
            return handoff;
        }
        // Lent in the game's declared colour state; After() takes it back to readable.
        owner.Barrier(cmd, g.composite, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, arrival);
        if (owner.pipelineCapture)
            owner.pipelineCapture->Copy(cmd, g.device, "composite", g.composite, arrival);
        pending.handedOff = g.composite;
        pending.handedState = arrival;
        handoff.color = g.composite;
        // No per-frame numbers here: the status is logged whenever it changes.
        detail = std::string(" (") + KernelName((int) c.DebugView) + ", " + (c.CompareMode ? "ratio" : "difference") +
                 ", jitter shift " + (shift ? (sign > 0 ? "on, flipped" : "on") : "off") +
                 (c.CompareSwap ? ", clamped" : "") + ")";
    }
    pending.cmd = cmd;
    pending.caller = source;
    pending.replaceOutput = handoff.replaceOutput;
    g.reset = false;
    // No timings here: Say() logs on every change, and DenoiseStatus() appends the live numbers.
    Say(std::string("running: ") + (g.privateRr ? "RR" : "DLSS") + " 1:1 -> NR -> " + StepName(g.step) + detail);
    return handoff;
}

auto DlssNr_Dx12::State::DenoiseFirstContext::After(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source,
                                                    ID3D12Resource* output, bool upscaled) -> void
{
    const auto pair = pending;
    pending = {};
    if (!current || pair.cmd != cmd || pair.caller != source)
    {
        if (current)
            current->reset = true;
        return;
    }
    auto& g = *current;
    owner.lifetime.Record(cmd);
    lifetime.Record(cmd);
    if (pair.handedOff)
        owner.Barrier(cmd, pair.handedOff, pair.handedState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!pair.replaceOutput)
        return;
    if (!upscaled || !output || g.failed)
    {
        g.reset = true;
        return;
    }
    const auto& cfg = *Config::Instance();
    if ((cfg.RestoreComputeSignature.value_or_default() || cfg.RestoreGraphicSignature.value_or_default()) &&
        !D3D12Hooks::CanRestoreRootSignature(cmd))
    {
        g.reset = true;
        Say("inactive: game state cannot be restored after the upscale");
        return;
    }
    const auto desc = output->GetDesc();
    if (desc.Width < g.outW || desc.Height < g.outH || desc.Format != g.outputFormat)
    {
        g.reset = true;
        Say("inactive: the upscaler output changed shape under the private upscale");
        return;
    }
    const auto arrival = cfg.OutputResourceBarrier.has_value()
                             ? (D3D12_RESOURCE_STATES) cfg.OutputResourceBarrier.value()
                             : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    owner.Barrier(cmd, g.upscaled, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    owner.Barrier(cmd, output, arrival, D3D12_RESOURCE_STATE_COPY_DEST);
    DlssNr::CopyActiveColor(cmd, output, g.upscaled, { g.outW, g.outH });
    owner.Barrier(cmd, output, D3D12_RESOURCE_STATE_COPY_DEST, arrival);
    owner.Barrier(cmd, g.upscaled, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}
