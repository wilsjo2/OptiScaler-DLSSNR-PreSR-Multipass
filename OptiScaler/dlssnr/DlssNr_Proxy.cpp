#include "pch.h"
#include "DlssNr_Proxy.h"
#include "DlssNr_GpuLifetime.h"
#include "DlssNr_NgxDiagnostics.h"
#include "DlssNr_CompatibilityRuntime.h"

#include <Config.h>
#include <Logger.h>
#include <proxies/NVNGX_Proxy.h>
#include <vector>

namespace
{
// Use the SDK interface so the compiler selects the correct overloaded virtual method.
// Declaration order is not vtable order under the MSVC ABI.
void SetUInt(NVSDK_NGX_Parameter* params, const char* name, unsigned int value) { params->Set(name, value); }

void SetResource(NVSDK_NGX_Parameter* params, const char* name, ID3D12Resource* value) { params->Set(name, value); }

void SetFloat(NVSDK_NGX_Parameter* params, const char* name, float value) { params->Set(name, value); }

struct ProxyState
{
    NVSDK_NGX_Handle* feature = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    std::shared_ptr<DlssNr::CompatibilityRuntime> compatibility;

    DlssNr::Proxy::Settings settings {};
    unsigned int width = 0, height = 0;
    uint64_t creationEpoch = 0;
    ID3D12Device* device = nullptr;
    bool failed = false;
    bool reset = true;
};

void DestroyState(ProxyState& state)
{
    if (state.feature != nullptr)
    {
        if (state.compatibility) state.compatibility->Release(state.feature);
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
void SetCreationParameters(NVSDK_NGX_Parameter* params, const DlssNr::Proxy::Settings& settings, unsigned int width,
                           unsigned int height)
{
    SetUInt(params, "DLSSNR.Enabled", 1u);
    SetUInt(params, "DLSSNR.Width", width);
    SetUInt(params, "DLSSNR.Height", height);
    SetUInt(params, "CreationNodeMask", 1u);
    SetUInt(params, "VisibilityNodeMask", 1u);

    // Set the default preset explicitly, too.
    SetUInt(params, "DLSSNR.Hint.Render.Preset", (unsigned int) settings.preset);

    SetFloat(params, "DLSSNR.Intensity", settings.intensity);
    SetUInt(params, "DLSSNR.Style", (unsigned int) settings.style);
    SetFloat(params, "DLSSNR.LocalStructureStrength", settings.localStructure);
    SetFloat(params, "DLSSNR.LocalToneStrength", settings.localTone);
    SetFloat(params, "DLSSNR.SkinStructureStrength", settings.skinStructure);
    SetUInt(params, "DLSSNR.UseAutoMask", settings.autoMask ? 1u : 0u);

    // UI correction at the model's own default: with no UI layer fed to it there is nothing to
    // correct.
    SetUInt(params, "DLSSNR.UICorrection", 1u);
    SetResource(params, "DLSSNR.ControlMask", nullptr);
    SetResource(params, "DLSSNR.UI", nullptr);
    SetResource(params, "DLSSNR.UIAlpha", nullptr);
    SetResource(params, "DLSSNR.Backbuffer", nullptr);
    for (const char* key :
         { "DLSSNR.UISubrectBaseX", "DLSSNR.UISubrectBaseY", "DLSSNR.UISubrectWidth", "DLSSNR.UISubrectHeight",
           "DLSSNR.UIAlphaSubrectBaseX", "DLSSNR.UIAlphaSubrectBaseY", "DLSSNR.UIAlphaSubrectWidth",
           "DLSSNR.UIAlphaSubrectHeight", "DLSSNR.BackbufferSubrectBaseX", "DLSSNR.BackbufferSubrectBaseY",
           "DLSSNR.BackbufferSubrectWidth", "DLSSNR.BackbufferSubrectHeight" })
        SetUInt(params, key, 0u);
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
    void RetireState();
    void TickRetired(uint64_t epoch);
    unsigned int Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                         unsigned int height, const Settings& settings, uint64_t submissionEpoch, bool* ready);
    void Release();
    unsigned int Run(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, ID3D12Resource* color,
                     ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output, unsigned int width,
                     unsigned int height, unsigned int guideWidth, unsigned int guideHeight, unsigned int motionWidth,
                     unsigned int motionHeight, unsigned int depthBaseX, unsigned int depthBaseY,
                     unsigned int motionBaseX, unsigned int motionBaseY, bool depthInverted, bool reset, float mvScaleX,
                     float mvScaleY, const Settings& settings, uint64_t submissionEpoch, bool* evaluated);
};

void Context::Impl::RetireState()
{
    if (state.feature != nullptr || state.params != nullptr)
        lifetime.Retire([retired = state]() mutable { DestroyState(retired); });
    state = {};
}

void Context::Impl::TickRetired([[maybe_unused]] uint64_t epoch) { lifetime.Collect(); }

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
                                    unsigned int height, const Settings& settings, uint64_t submissionEpoch,
                                    bool* ready)
{
    *ready = false;
    TickRetired(submissionEpoch);
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
        LOG_INFO("NR diagnostic capability parameters: result=0x{:08X} params={}",
                 (unsigned)allocated, (void*)state.params);
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
        auto created =
            NVNGXProxy::D3D12_CreateFeature()(cmdList, (NVSDK_NGX_Feature) 18, state.params, &state.feature);
        LOG_INFO("NR diagnostic CreateFeature(18): result=0x{:08X} handle={}", (unsigned)created, (void*)state.feature);
        if (NVSDK_NGX_FAILED(created) && !state.feature)
        {
            state.compatibility = CompatibilityRuntime::TryOpen(device);
            if (state.compatibility)
            {
                SetCreationParameters(state.params, settings, width, height);
                created = state.compatibility->Create(cmdList, state.params, &state.feature);
                LOG_INFO("NR compatibility: CreateFeature(18) result=0x{:08X} handle={}",
                         (unsigned)created, (void*)state.feature);
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
        LOG_INFO("DLSS-NR: feature created at {}x{} through {}", width, height,
                 state.compatibility ? "direct compatibility runtime" : "NVIDIA NGX driver");

        // Creation must reach the GPU before any evaluation is recorded.
        return (unsigned int) NVSDK_NGX_Result_Success;
    }

    *ready = submissionEpoch != state.creationEpoch;
    return (unsigned int) NVSDK_NGX_Result_Success;
}

unsigned int Context::Impl::Run(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, ID3D12Resource* color,
                                ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                                unsigned int width, unsigned int height, unsigned int guideWidth,
                                unsigned int guideHeight, unsigned int motionWidth, unsigned int motionHeight,
                                unsigned int depthBaseX, unsigned int depthBaseY, unsigned int motionBaseX,
                                unsigned int motionBaseY, bool depthInverted, bool reset, float mvScaleX,
                                float mvScaleY, const Settings& settings, uint64_t submissionEpoch, bool* evaluated)
{
    if (evaluated)
        *evaluated = false;
    if (!color || !depth || !motion || !output || !guideWidth || !guideHeight || !motionWidth || !motionHeight)
        return 0;
    bool ready = false;
    const auto prepared = Prepare(cmdList, device, width, height, settings, submissionEpoch, &ready);
    if (prepared != NVSDK_NGX_Result_Success || !ready)
        return prepared;

    NVSDK_NGX_Parameter* params = state.params;

    SetResource(params, "DLSSNR.Color", color);
    SetResource(params, "DLSSNR.Depth", depth);
    SetResource(params, "DLSSNR.MVec", motion);
    SetResource(params, "DLSSNR.Output", output);

    // TEMPORARY EXPERIMENT (ADR-013 control-mask investigation, remove after use). DLSSNR.ControlMask
    // is a real, currently-unused optional input -- confirmed via direct strings/objdump analysis of
    // the deployed nvngx_dlssnr.dll, with full subrect addressing exactly like Color/Depth/MVec/Output.
    // Its actual semantics are undocumented anywhere reachable from source (Streamline's own header
    // says only "optional 4-channel control mask", nothing about format or population strategy).
    // Cheapest possible experiment for the one question that matters first: does Feature 18 respond to
    // *anything* bound here at all? Round 1 (motion as probe): EvaluateFeature result=0x1, no visible
    // artifact -- inconclusive, motion vectors are likely near-uniform/low-magnitude in this scene,
    // which would mask a real effect even if consumed. Round 2: reuse the colour buffer instead --
    // high dynamic range, strong per-pixel structure, so any real consumption should be unmistakable.
    // Deliberately the wrong format/content either way; a visible artifact or timing/behavior change
    // versus the null baseline is itself the finding, independent of whether the content is
    // semantically meaningful. Not a claim about what the mask should actually contain.
    const auto controlMaskTestPattern = Config::Instance()->DlssNrControlMaskTestPattern.value_or_default();
    ID3D12Resource* controlMaskProbe =
        controlMaskTestPattern == 1 ? motion : controlMaskTestPattern == 2 ? color : nullptr;
    const char* controlMaskProbeName =
        controlMaskTestPattern == 1 ? "motion-as-mask" : controlMaskTestPattern == 2 ? "color-as-mask" : "null";
    SetResource(params, "DLSSNR.ControlMask", controlMaskProbe);
    SetUInt(params, "DLSSNR.ControlMaskSubrectBaseX", 0u);
    SetUInt(params, "DLSSNR.ControlMaskSubrectBaseY", 0u);
    SetUInt(params, "DLSSNR.ControlMaskSubrectWidth",
            controlMaskProbe == nullptr ? 0u : controlMaskTestPattern == 1 ? motionWidth : width);
    SetUInt(params, "DLSSNR.ControlMaskSubrectHeight",
            controlMaskProbe == nullptr ? 0u : controlMaskTestPattern == 1 ? motionHeight : height);

    // TEMPORARY EXPERIMENT (Phase 5 unexplored-inputs audit, remove after use). DLSSNR.
    // BidirectionalDistortionField is a real, currently-unused optional input -- confirmed via direct
    // strings/objdump analysis of the deployed nvngx_dlssnr.dll, with full subrect addressing exactly
    // like Color/Depth/MVec/ControlMask. Never referenced anywhere in this codebase before now, and no
    // source or public documentation explains its semantics -- the name suggests some form of per-pixel
    // UV/lens correction, but that is a guess, not a finding. Same cheapest-possible test as the
    // ControlMask experiment: does Feature 18 respond to anything bound here at all. Colour as probe
    // (high dynamic range, strong per-pixel structure) rather than motion, since motion-as-probe was
    // inconclusive for ControlMask.
    const auto bidirDistortionTestPattern = Config::Instance()->DlssNrBidirDistortionTestPattern.value_or_default();
    ID3D12Resource* bidirDistortionProbe = bidirDistortionTestPattern == 1 ? color : nullptr;
    SetResource(params, "DLSSNR.BidirectionalDistortionField", bidirDistortionProbe);
    SetUInt(params, "DLSSNR.BidirectionalDistortionFieldSubrectBaseX", 0u);
    SetUInt(params, "DLSSNR.BidirectionalDistortionFieldSubrectBaseY", 0u);
    SetUInt(params, "DLSSNR.BidirectionalDistortionFieldSubrectWidth", bidirDistortionProbe == nullptr ? 0u : width);
    SetUInt(params, "DLSSNR.BidirectionalDistortionFieldSubrectHeight", bidirDistortionProbe == nullptr ? 0u : height);

    SetUInt(params, "DLSSNR.Enabled", 1u);
    SetUInt(params, "DLSSNR.Width", width);
    SetUInt(params, "DLSSNR.Height", height);
    SetUInt(params, "DLSSNR.DepthInverted", depthInverted ? 1u : 0u);
    SetUInt(params, "DLSSNR.Reset", (reset || state.reset) ? 1u : 0u);

    // Colour and output are display resolution; depth and motion come from the game's own DLSS
    // evaluation and may be render resolution, so each resource carries its own subrect.
    SetUInt(params, "DLSSNR.ColorSubrectBaseX", 0u);
    SetUInt(params, "DLSSNR.ColorSubrectBaseY", 0u);
    SetUInt(params, "DLSSNR.ColorSubrectWidth", width);
    SetUInt(params, "DLSSNR.ColorSubrectHeight", height);
    SetUInt(params, "DLSSNR.OutputSubrectBaseX", 0u);
    SetUInt(params, "DLSSNR.OutputSubrectBaseY", 0u);
    SetUInt(params, "DLSSNR.OutputSubrectWidth", width);
    SetUInt(params, "DLSSNR.OutputSubrectHeight", height);
    SetUInt(params, "DLSSNR.DepthSubrectBaseX", depthBaseX);
    SetUInt(params, "DLSSNR.DepthSubrectBaseY", depthBaseY);
    SetUInt(params, "DLSSNR.DepthSubrectWidth", guideWidth);
    SetUInt(params, "DLSSNR.DepthSubrectHeight", guideHeight);
    SetUInt(params, "DLSSNR.MVecSubrectBaseX", motionBaseX);
    SetUInt(params, "DLSSNR.MVecSubrectBaseY", motionBaseY);
    SetUInt(params, "DLSSNR.MVecSubrectWidth", motionWidth);
    SetUInt(params, "DLSSNR.MVecSubrectHeight", motionHeight);

    // The game's own encoding, passed through. Deriving this from the resolutions was a guess, and
    // at native resolution it came out as exactly 1.0 -- so a game using normalised vectors was
    // telling the model that almost nothing had moved.
    SetFloat(params, "DLSSNR.MVecScaleX", mvScaleX);
    SetFloat(params, "DLSSNR.MVecScaleY", mvScaleY);

    SetFloat(params, "DLSSNR.Intensity", settings.intensity);
    SetUInt(params, "DLSSNR.Style", (unsigned int) settings.style);
    SetFloat(params, "DLSSNR.LocalStructureStrength", settings.localStructure);
    SetFloat(params, "DLSSNR.LocalToneStrength", settings.localTone);
    SetFloat(params, "DLSSNR.SkinStructureStrength", settings.skinStructure);
    SetUInt(params, "DLSSNR.UseAutoMask", settings.autoMask ? 1u : 0u);

    lifetime.Record(cmdList);
    const auto result = state.compatibility ? state.compatibility->Evaluate(cmdList, state.feature, params)
                                           : NVNGXProxy::D3D12_EvaluateFeature()(cmdList, state.feature, params, nullptr);

    // TEMPORARY EXPERIMENT (ADR-013, remove after use): one-shot log of whether binding something to
    // ControlMask changes the EvaluateFeature result code at all versus the established null baseline.
    if (controlMaskTestPattern != 0)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("DLSS-NR ControlMask experiment: probe={} EvaluateFeature result=0x{:X}",
                     controlMaskProbeName, (unsigned int) result);
        }
    }

    if (bidirDistortionTestPattern != 0)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            LOG_INFO("DLSS-NR BidirectionalDistortionField experiment: probe={} EvaluateFeature result=0x{:X}",
                     bidirDistortionProbe == nullptr ? "null" : "color-as-probe", (unsigned int) result);
        }
    }

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

unsigned int Context::Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                              unsigned int height, const Settings& settings, uint64_t submissionEpoch, bool* ready)
{
    return _impl->Prepare(cmdList, device, width, height, settings, submissionEpoch, ready);
}
bool Context::HasFeature() const { return _impl->state.feature != nullptr; }
bool Context::Ready(uint64_t epoch) const
{
    return HasFeature() && !_impl->state.failed && epoch != _impl->state.creationEpoch;
}
void Context::AdvanceEpoch(uint64_t epoch) { _impl->TickRetired(epoch); }

unsigned int Context::Run(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, ID3D12Resource* color,
                          ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output, unsigned int width,
                          unsigned int height, unsigned int guideWidth, unsigned int guideHeight,
                          unsigned int motionWidth, unsigned int motionHeight, unsigned int depthBaseX,
                          unsigned int depthBaseY, unsigned int motionBaseX, unsigned int motionBaseY,
                          bool depthInverted, bool reset, float mvScaleX, float mvScaleY, const Settings& settings,
                          uint64_t submissionEpoch, bool* evaluated)
{
    return _impl->Run(cmdList, device, color, depth, motion, output, width, height, guideWidth, guideHeight,
                      motionWidth, motionHeight, depthBaseX, depthBaseY, motionBaseX, motionBaseY, depthInverted, reset,
                      mvScaleX, mvScaleY, settings, submissionEpoch, evaluated);
}
} // namespace Proxy
} // namespace DlssNr
