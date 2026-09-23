#include "pch.h"
#include "DlssNr_Proxy.h"
#include "DlssNr_GpuLifetime.h"
#include "DlssNr_NgxDiagnostics.h"
#include "DlssNr_CompatibilityRuntime.h"

#include <Logger.h>
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
    std::function<bool()> creationComplete;
    bool creationReady = false;
    ID3D12Device* device = nullptr;
    bool failed = false;
    bool reset = true;
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
    bool CreationReady(uint64_t epoch)
    {
        return state.creationReady = state.creationReady || epoch != state.creationEpoch ||
                                     (state.creationComplete && state.creationComplete());
    }
    void RetireState();
    unsigned int Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                         unsigned int height, const ModelSettings& settings, uint64_t submissionEpoch, bool* ready);
    void Release();
};

void Context::Impl::RetireState()
{
    if (state.feature != nullptr || state.params != nullptr)
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

unsigned int Context::Impl::Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                                    unsigned int height, const ModelSettings& settings, uint64_t submissionEpoch,
                                    bool* ready)
{
    *ready = false;
    lifetime.Collect();
    if (state.failed || !cmdList || !device || !width || !height)
        return 0;
    if ((!NVNGXProxy::IsDx12Inited() && !NVNGXProxy::InitDx12(device)) || !Context::Available())
        return 0;
    if (state.feature &&
        (state.settings != settings || state.device != device || state.width != width || state.height != height))
        RetireState();
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
            state.failed = true;
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
            state.failed = true;
            LOG_ERROR("DLSS-NR: CreateFeature(18) failed 0x{:X}", (unsigned int) created);
            return (unsigned int) (created == NVSDK_NGX_Result_Success ? NVSDK_NGX_Result_Fail : created);
        }

        state.settings = settings;
        state.device = device;
        state.width = width;
        state.height = height;
        state.creationEpoch = submissionEpoch;
        state.creationComplete = lifetime.CompletionProbe(cmdList);
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
bool Context::HasFeature() const { return _impl->state.feature != nullptr; }
bool Context::Ready(uint64_t epoch) const
{
    return HasFeature() && !_impl->state.failed && _impl->CreationReady(epoch);
}
void Context::Collect() { _impl->lifetime.Collect(); }

} // namespace Proxy
} // namespace DlssNr
