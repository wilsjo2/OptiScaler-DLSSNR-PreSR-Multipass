#include "pch.h"
#include "DlssNr_Proxy.h"
#include "DlssNr_GpuLifetime.h"
#include "DlssNr_NgxDiagnostics.h"
#include "DlssNr_CompatibilityRuntime.h"

#include "NrNvfp4.h"

#include <Logger.h>
#include <Util.h>
#include <proxies/NVNGX_Proxy.h>
#include <vector>

namespace
{
struct ProxyState
{
    NVSDK_NGX_Handle* feature = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    std::shared_ptr<DlssNr::CompatibilityRuntime> compatibility;

    DlssNr::ModelSettings settings {};
    unsigned int width = 0, height = 0;
    uint64_t creationEpoch = 0;
    std::function<DlssNr::GpuLifetime::Recorded()> creation; // what became of the command list CreateFeature recorded on
    bool creationReady = false;
    unsigned discards = 0; // creations in a row whose command list was thrown away before it reached a queue
    ID3D12Device* device = nullptr;
    bool failed = false;
    bool reset = true;
    // Native passes use a host in place of an NGX feature. hostFailed: it could not be made or could not run;
    // the pass then reports through DlssNr::Nvfp4::OursRan and nothing is shown.
    std::shared_ptr<void> host;
    bool hostModel = false, hostFailed = false, hostMotionTold = false;
    // The game's motion vectors when they are a region of a larger texture: the host samples whole textures, so the
    // region is copied into this one each frame.
    std::shared_ptr<ID3D12Resource> hostMotion;
};

void DestroyState(ProxyState& state)
{
    if (state.feature != nullptr)
    {
        if (state.compatibility)
            state.compatibility->Release(state.feature);
        else if (NVNGXProxy::D3D12_ReleaseFeature() != nullptr)
            NVNGXProxy::D3D12_ReleaseFeature()(state.feature);
    }

    if (state.params != nullptr && NVNGXProxy::D3D12_DestroyParameters() != nullptr)
        NVNGXProxy::D3D12_DestroyParameters()(state.params);

    state = {};
}

// Everything the model reads when the feature is built.
//
// These have to be set before create, not at evaluate. The model reads its tuning once, while
// building the feature; values written only at evaluate are ignored, which is why several of these
// controls appeared to do nothing for a long time.
void SetCreationParameters(NVSDK_NGX_Parameter* params, const DlssNr::ModelSettings& settings, unsigned int width,
                           unsigned int height)
{
    params->Set("DLSSNR.Enabled", 1u);
    params->Set("DLSSNR.Width", width);
    params->Set("DLSSNR.Height", height);
    params->Set("CreationNodeMask", 1u);
    params->Set("VisibilityNodeMask", 1u);

    // Set the default preset explicitly, too.
    params->Set("DLSSNR.Hint.Render.Preset", (unsigned int) settings.preset);

    DlssNr::SetModelTuning(params, settings);

    // UI correction at the model's own default: with no UI layer fed to it there is nothing to
    // correct.
    params->Set("DLSSNR.UICorrection", 1u);
    params->Set("DLSSNR.ControlMask", static_cast<ID3D12Resource*>(nullptr));
    params->Set("DLSSNR.UI", static_cast<ID3D12Resource*>(nullptr));
    params->Set("DLSSNR.UIAlpha", static_cast<ID3D12Resource*>(nullptr));
    params->Set("DLSSNR.Backbuffer", static_cast<ID3D12Resource*>(nullptr));
    for (const char* key :
         { "DLSSNR.UISubrectBaseX", "DLSSNR.UISubrectBaseY", "DLSSNR.UISubrectWidth", "DLSSNR.UISubrectHeight",
           "DLSSNR.UIAlphaSubrectBaseX", "DLSSNR.UIAlphaSubrectBaseY", "DLSSNR.UIAlphaSubrectWidth",
           "DLSSNR.UIAlphaSubrectHeight", "DLSSNR.BackbufferSubrectBaseX", "DLSSNR.BackbufferSubrectBaseY",
           "DLSSNR.BackbufferSubrectWidth", "DLSSNR.BackbufferSubrectHeight" })
        params->Set(key, 0u);
}
} // namespace

namespace DlssNr
{
namespace Proxy
{
struct Context::Impl
{
    ProxyState state;
    DlssNr::GpuLifetime lifetime;
    // CreateFeature records the feature's setup on the caller's command list. The feature may be evaluated once the
    // GPU has run that setup, or once the list has been given to a queue and a later submission epoch has begun
    // (whatever is recorded now is queued behind it). A list that is still open, or that was reset or released
    // without being executed, has not set anything up: evaluating then would run the model on unwritten GPU state.
    // Without submission notifications (Untracked) the epoch is all there is to go by.
    bool CreationReady(uint64_t epoch)
    {
        if (state.creationReady)
            return true;
        using Recorded = DlssNr::GpuLifetime::Recorded;
        const auto recorded = state.creation ? state.creation() : Recorded::Untracked;
        const bool ready = recorded == Recorded::Finished ||
                           (epoch != state.creationEpoch &&
                            (recorded == Recorded::Submitted || recorded == Recorded::Untracked));
        if (ready)
            state.discards = 0;
        return state.creationReady = ready;
    }
    bool CreationDiscarded() const
    {
        return state.feature && !state.creationReady && state.creation &&
               state.creation() == DlssNr::GpuLifetime::Recorded::Discarded;
    }
    void RetireState();
    bool UpdateTuning(const ModelSettings& settings);
    unsigned int Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                         unsigned int height, const ModelSettings& settings, uint64_t submissionEpoch, bool* ready);
    void Release();
};

void Context::Impl::RetireState()
{
    if (state.feature != nullptr || state.params != nullptr || state.host || state.hostMotion)
        lifetime.Retire([retired = state]() mutable { DestroyState(retired); });
    state = {};
    lifetime.BeginGeneration();
}

bool Context::Available()
{
    return NVNGXProxy::IsDx12Inited() && NVNGXProxy::D3D12_GetCapabilityParameters() != nullptr &&
           NVNGXProxy::D3D12_DestroyParameters() != nullptr && NVNGXProxy::D3D12_ReleaseFeature() != nullptr &&
           NVNGXProxy::D3D12_CreateFeature() != nullptr && NVNGXProxy::D3D12_EvaluateFeature() != nullptr;
}

void Context::Impl::Release()
{
    RetireState();
    lifetime.Collect();
}

void Context::RetryAfterFailure() { _impl->RetireState(); }

bool Context::Impl::UpdateTuning(const ModelSettings& settings)
{
    if ((!state.feature && !state.hostModel && !state.failed) || state.settings == settings)
        return false;
    if (state.hostModel && !state.hostFailed && state.settings.requested == settings.requested)
    {
        state.settings = settings;
        state.reset = true;
        // (An NGX feature logs its tuning when it is created; the host keeps running, so say it here.)
        LOG_INFO("DLSS-NR: native model {} tuning: style={}, intensity={:.2f}, structure={:.2f}, tone={:.2f}",
                 settings.requested, settings.style, settings.intensity, settings.localStructure, settings.localTone);
    }
    else
        RetireState();
    return true;
}

bool Context::UpdateTuning(const ModelSettings& settings) { return _impl->UpdateTuning(settings); }

unsigned int Context::Impl::Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                                    unsigned int height, const ModelSettings& settings, uint64_t submissionEpoch,
                                    bool* ready)
{
    *ready = false;
    lifetime.Collect();
    if (!cmdList || !device || !width || !height)
        return 0;
    if ((state.feature || state.hostModel || state.failed) &&
        (state.device != device || state.width != width || state.height != height))
        RetireState();
    else
        UpdateTuning(settings);
    if (state.failed)
        return 0;
    const auto rememberFailure = [&] {
        state.failed = true;
        state.settings = settings;
        state.device = device;
        state.width = width;
        state.height = height;
    };
    if (DlssNr::Nvfp4::UsesNativeHost(settings.requested))
    {
        // No NGX feature: the whole pass runs on the own pass host, which is ready as soon as it is made.
        if (!state.hostModel)
        {
            state.host = DlssNr::Nvfp4::CreateHost(device, settings.requested, width, height,
                                                   Util::DllPath().parent_path() / L"OptiScaler" / L"nvfp4");
            state.hostModel = true;
            state.hostFailed = !state.host;
            state.settings = settings;
            state.device = device;
            state.width = width;
            state.height = height;
            state.creationEpoch = submissionEpoch;
            state.creationReady = true;
            if (state.host)
                LOG_INFO("DLSS-NR: native model {} set up at {}x{} on the own pass host: style={}, intensity={:.2f}, "
                         "structure={:.2f}, tone={:.2f}",
                         settings.requested, width, height, settings.style, settings.intensity,
                         settings.localStructure, settings.localTone);
        }
        *ready = true;
        return (unsigned int) NVSDK_NGX_Result_Success;
    }
    if ((!NVNGXProxy::IsDx12Inited() && !NVNGXProxy::InitDx12(device)) || !Context::Available())
        return 0;
    if (CreationDiscarded())
    {
        // The command list that carried this feature's setup was reset or released without being executed. The
        // feature is retired unused and built again on this command list.
        constexpr unsigned kMaxDiscards = 3;
        const unsigned discards = state.discards + 1;
        LOG_WARN("DLSS-NR: the command list that carried the feature's setup was discarded before it reached the GPU "
                 "({} in a row); creating the feature again", discards);
        RetireState();
        if (discards >= kMaxDiscards)
        {
            rememberFailure();
            LOG_ERROR("DLSS-NR: the feature's setup was discarded {} times in a row; NR is off until Retry or a "
                      "settings change", discards);
            return (unsigned int) NVSDK_NGX_Result_Fail;
        }
        state.discards = discards;
    }
    if (state.params == nullptr)
    {
        // A dedicated parameter map populated with NGX capabilities. Unlike the deprecated
        // GetParameters API, GetCapabilityParameters transfers ownership to the caller.
        NgxDiagnostics::Scope nrCapabilityTrace;
        const auto allocated = NVNGXProxy::D3D12_GetCapabilityParameters()(&state.params);
        LOG_INFO("NR diagnostic capability parameters: result=0x{:08X} params={}", (unsigned) allocated,
                 (void*) state.params);
        if (allocated != NVSDK_NGX_Result_Success || state.params == nullptr)
        {
            DestroyState(state);
            rememberFailure();
            LOG_ERROR("DLSS-NR (driver): the NGX core refused its capability parameters");
            return (unsigned int) (allocated == NVSDK_NGX_Result_Success ? NVSDK_NGX_Result_Fail : allocated);
        }
    }

    if (state.feature == nullptr)
    {
        NgxDiagnostics::Scope nrCreateTrace;
        NgxDiagnostics::RuntimeReport(cmdList, device, "before CreateFeature(18)");
        LOG_INFO("NR diagnostic creation: {}x{}, preset={}, style={}, intensity={}, structure={}, tone={}, "
                 "skin={}, autoMask={}, node masks=1/1, UI correction=1, UI/control/backbuffer=null, epoch={}",
                 width, height, settings.preset, settings.style, settings.intensity, settings.localStructure,
                 settings.localTone, settings.skinStructure, settings.autoMask, submissionEpoch);
        SetCreationParameters(state.params, settings, width, height);

        lifetime.Record(cmdList);
        auto created = NVNGXProxy::D3D12_CreateFeature()(cmdList, (NVSDK_NGX_Feature) 18, state.params, &state.feature);
        LOG_INFO("NR diagnostic CreateFeature(18): result=0x{:08X} handle={}", (unsigned) created,
                 (void*) state.feature);
        if (NVSDK_NGX_FAILED(created) && !state.feature)
        {
            state.compatibility = CompatibilityRuntime::TryOpen(device);
            if (state.compatibility)
            {
                SetCreationParameters(state.params, settings, width, height);
                created = state.compatibility->Create(cmdList, state.params, &state.feature);
                LOG_INFO("NR compatibility: CreateFeature(18) result=0x{:08X} handle={}", (unsigned) created,
                         (void*) state.feature);
            }
        }
        NgxDiagnostics::RuntimeReport(cmdList, device, "after CreateFeature(18)");

        if (created != NVSDK_NGX_Result_Success || state.feature == nullptr)
        {
            RetireState();
            rememberFailure();
            LOG_ERROR("DLSS-NR: CreateFeature(18) failed 0x{:X}", (unsigned int) created);
            return (unsigned int) (created == NVSDK_NGX_Result_Success ? NVSDK_NGX_Result_Fail : created);
        }

        state.settings = settings;
        state.device = device;
        state.width = width;
        state.height = height;
        state.creationEpoch = submissionEpoch;
        state.creation = lifetime.RecordingProbe(cmdList);
        LOG_INFO("DLSS-NR: feature created at {}x{} through {}", width, height,
                 state.compatibility ? "direct compatibility runtime" : "NVIDIA NGX driver");

        // Creation must reach the GPU before any evaluation is recorded.
        return (unsigned int) NVSDK_NGX_Result_Success;
    }

    *ready = CreationReady(submissionEpoch);
    return (unsigned int) NVSDK_NGX_Result_Success;
}

unsigned int Context::Run(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, const Frame& frame,
                          const ModelSettings& settings, uint64_t submissionEpoch, bool* evaluated)
{
    auto& state = _impl->state;
    auto& lifetime = _impl->lifetime;
    if (evaluated)
        *evaluated = false;
    if (!frame.color || !frame.depth || !frame.motion || !frame.output || !frame.guides.depth.valid() ||
        !frame.guides.motion.valid())
        return 0;
    bool ready = false;
    const auto prepared =
        Prepare(cmdList, device, frame.size.width, frame.size.height, settings, submissionEpoch, &ready);
    if (prepared != NVSDK_NGX_Result_Success || !ready)
        return prepared;

    if (state.hostModel)
    {
        // The host samples the whole motion texture. When the game's vectors are a region of a larger one (dynamic
        // resolution, an atlas), that region is copied into a texture of its own first.
        ID3D12Resource* motion = frame.motion;
        const auto motionDesc = motion->GetDesc();
        const auto& region = frame.guides.motion;
        if (region.x != 0 || region.y != 0 || region.width != motionDesc.Width || region.height != motionDesc.Height)
        {
            const bool inside = region.width != 0 && region.height != 0 &&
                                uint64_t(region.x) + region.width <= motionDesc.Width &&
                                uint64_t(region.y) + region.height <= motionDesc.Height &&
                                motionDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                                motionDesc.SampleDesc.Count == 1;
            if (inside && state.hostMotion)
            {
                const auto held = state.hostMotion->GetDesc();
                if (held.Width != region.width || held.Height != region.height || held.Format != motionDesc.Format)
                {
                    // Earlier command lists may still read it: freed once the GPU has finished them.
                    lifetime.Retire([retired = std::move(state.hostMotion)]() mutable { retired.reset(); });
                    state.hostMotion.reset();
                }
            }
            if (inside && !state.hostMotion)
            {
                D3D12_HEAP_PROPERTIES heap {};
                heap.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC desc {};
                desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                desc.Width = region.width;
                desc.Height = region.height;
                desc.DepthOrArraySize = 1;
                desc.MipLevels = 1;
                desc.Format = motionDesc.Format;
                desc.SampleDesc.Count = 1;
                ID3D12Resource* made = nullptr;
                if (SUCCEEDED(state.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                                    nullptr, IID_PPV_ARGS(&made))))
                    state.hostMotion.reset(made, [](ID3D12Resource* resource) { resource->Release(); });
            }
            if (inside && state.hostMotion)
            {
                const auto transition = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES from,
                                            D3D12_RESOURCE_STATES to)
                {
                    D3D12_RESOURCE_BARRIER barrier {};
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barrier.Transition.pResource = resource;
                    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    barrier.Transition.StateBefore = from;
                    barrier.Transition.StateAfter = to;
                    cmdList->ResourceBarrier(1, &barrier);
                };
                // (The model's guides arrive readable: NON_PIXEL_SHADER_RESOURCE.)
                transition(motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
                transition(state.hostMotion.get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_TEXTURE_COPY_LOCATION to {}, from {};
                to.pResource = state.hostMotion.get();
                to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                from.pResource = motion;
                from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                const D3D12_BOX box { region.x, region.y, 0, region.x + region.width, region.y + region.height, 1 };
                cmdList->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
                transition(state.hostMotion.get(), D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(motion, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                motion = state.hostMotion.get();
                if (!state.hostMotionTold)
                    LOG_INFO("DLSS-NR native model: the motion vectors are a {}x{} region at {},{} of a {}x{} texture; "
                             "the region is copied for the own pass host each frame",
                             region.width, region.height, region.x, region.y, (unsigned) motionDesc.Width,
                             motionDesc.Height);
            }
            else
            {
                motion = nullptr;
                if (!state.hostMotionTold)
                    LOG_WARN("DLSS-NR native model: the motion vectors' {}x{} region at {},{} of a {}x{} texture could "
                             "not be copied; every frame runs without history",
                             region.width, region.height, region.x, region.y, (unsigned) motionDesc.Width,
                             motionDesc.Height);
            }
            state.hostMotionTold = true;
        }
        DlssNr::Nvfp4::HostPass pass;
        pass.colour = frame.color;
        pass.motion = motion;
        pass.output = frame.output;
        pass.style = settings.style;
        pass.intensity = settings.intensity;
        pass.structure = settings.localStructure;
        pass.tone = settings.localTone;
        pass.skin = settings.skinStructure;
        pass.autoMask = settings.autoMask;
        // The scale turns the game's vectors into texels of the motion texture (what the NGX model is given). The
        // host works in pixels of the model's picture: with the model after the upscaler the motion texture is the
        // smaller render size, and the vectors would come out too short by that ratio.
        pass.mvScaleX = frame.mvScaleX * float(frame.size.width) / float(region.width);
        pass.mvScaleY = frame.mvScaleY * float(frame.size.height) / float(region.height);
        pass.reset = frame.reset || state.reset;
        lifetime.Record(cmdList);
        if (!state.hostFailed && DlssNr::Nvfp4::RunHost(state.host, cmdList, pass))
            state.reset = false;
        else
            state.hostFailed = true; // until Retry or a settings change
        // Whether it ran is DlssNr::Nvfp4::OursRan's to say: a failed pass is reported there and nothing is shown.
        if (evaluated != nullptr)
            *evaluated = true;
        return (unsigned int) NVSDK_NGX_Result_Success;
    }

    NVSDK_NGX_Parameter* params = state.params;

    params->Set("DLSSNR.Color", frame.color);
    params->Set("DLSSNR.Depth", frame.depth);
    params->Set("DLSSNR.MVec", frame.motion);
    params->Set("DLSSNR.Output", frame.output);

    params->Set("DLSSNR.Enabled", 1u);
    params->Set("DLSSNR.Width", frame.size.width);
    params->Set("DLSSNR.Height", frame.size.height);
    params->Set("DLSSNR.DepthInverted", frame.depthInverted ? 1u : 0u);
    params->Set("DLSSNR.Reset", (frame.reset || state.reset) ? 1u : 0u);

    // Colour and output are display resolution; depth and motion come from the game's own DLSS
    // evaluation and may be render resolution, so each resource carries its own subrect.
    SetModelRegions(params, frame.size, frame.guides);

    // The game's own encoding, passed through. Deriving this from the resolutions was a guess, and
    // at native resolution it came out as exactly 1.0 -- so a game using normalised vectors was
    // telling the model that almost nothing had moved.
    params->Set("DLSSNR.MVecScaleX", frame.mvScaleX);
    params->Set("DLSSNR.MVecScaleY", frame.mvScaleY);

    DlssNr::SetModelTuning(params, settings);

    lifetime.Record(cmdList);
    const auto result = state.compatibility
                            ? state.compatibility->Evaluate(cmdList, state.feature, params)
                            : NVNGXProxy::D3D12_EvaluateFeature()(cmdList, state.feature, params, nullptr);

    if (result == NVSDK_NGX_Result_Success)
    {
        state.reset = false;
        if (evaluated != nullptr)
            *evaluated = true;
    }
    else
    {
        state.failed = true;
    }

    return (unsigned int) result;
}
Context::Context() : _impl(std::make_unique<Impl>()) {}
Context::~Context() { _impl->Release(); }
void Context::Release() { _impl->Release(); }
void Context::Submitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists)
{
    _impl->lifetime.Submitted(queue, count, lists);
}
void Context::ResetRecording(ID3D12CommandList* commands) { _impl->lifetime.ResetRecording(commands); }
bool Context::Idle() { return _impl->lifetime.Idle(); }
void Context::FinishSubmitted() { _impl->lifetime.FinishSubmitted(); }

unsigned int Context::Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                              unsigned int height, const ModelSettings& settings, uint64_t submissionEpoch, bool* ready)
{
    return _impl->Prepare(cmdList, device, width, height, settings, submissionEpoch, ready);
}
bool Context::HasFeature() const { return _impl->state.feature != nullptr || _impl->state.hostModel; }
bool Context::Ready(uint64_t epoch) const
{
    return HasFeature() && !_impl->state.failed && _impl->CreationReady(epoch);
}
void Context::Collect() { _impl->lifetime.Collect(); }

} // namespace Proxy
} // namespace DlssNr
