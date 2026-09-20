// Compile the production backend with mock NGX entry points and the real SDK parameter interface.
#include "../../OptiScaler/dlssnr/DlssNr_Proxy.cpp"
#include "../../OptiScaler/dlssnr/DlssNr_Status.cpp"
#include "../../OptiScaler/upscalers/ShaderPipeline_Dx12.h"
#include "../../OptiScaler/dlssnr/DlssNr_HoldParameters_Dx12.h"

namespace DlssNr::NgxDiagnostics
{
Scope::Scope() {}
Scope::~Scope() {}
void RuntimeReport(ID3D12GraphicsCommandList*, ID3D12Device*, const char*) {}
}

// Routing seam: hardware tests separately exercise the real compatibility loader.
namespace CompatibilityMock {
bool available=false;
unsigned opens=0,creates=0,evaluates=0,releases=0,destroyed=0;
NVSDK_NGX_Result createResult=NVSDK_NGX_Result_Success;
}
namespace DlssNr {
std::shared_ptr<CompatibilityRuntime> CompatibilityRuntime::TryOpen(ID3D12Device*) {
    ++CompatibilityMock::opens;
    return CompatibilityMock::available ? std::shared_ptr<CompatibilityRuntime>(new CompatibilityRuntime()) : nullptr;
}
CompatibilityRuntime::~CompatibilityRuntime() { ++CompatibilityMock::destroyed; }
NVSDK_NGX_Result CompatibilityRuntime::Create(ID3D12GraphicsCommandList* c,NVSDK_NGX_Parameter* p,NVSDK_NGX_Handle** h) {
    ++CompatibilityMock::creates;
    const auto saved=Mock::createResult;Mock::createResult=CompatibilityMock::createResult;
    const auto result=Mock::Create(c,(NVSDK_NGX_Feature)18,p,h);Mock::createResult=saved;return result;
}
NVSDK_NGX_Result CompatibilityRuntime::Evaluate(ID3D12GraphicsCommandList* c,const NVSDK_NGX_Handle* h,NVSDK_NGX_Parameter* p) {
    ++CompatibilityMock::evaluates;return Mock::Evaluate(c,h,p,nullptr);
}
NVSDK_NGX_Result CompatibilityRuntime::Release(NVSDK_NGX_Handle* h) {
    ++CompatibilityMock::releases;return Mock::Release(h);
}
}

namespace CompletionMock { bool complete = false; }
// NGX tests substitute completion only; nr_gpu_lifetime_smoke exercises real D3D12 fences.
struct DlssNr::GpuLifetime::Impl
{
    bool pending = false;
    std::vector<std::function<void()>> retired;
};
DlssNr::GpuLifetime::GpuLifetime() : impl(std::make_unique<Impl>()) {}
DlssNr::GpuLifetime::~GpuLifetime() { Collect(); }
void DlssNr::GpuLifetime::Record(ID3D12GraphicsCommandList*) { impl->pending = true; }
std::function<bool()> DlssNr::GpuLifetime::CompletionProbe(ID3D12GraphicsCommandList*)
{ return [] { return CompletionMock::complete; }; }
void DlssNr::GpuLifetime::Submitted(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*) {}
void DlssNr::GpuLifetime::ResetRecording(ID3D12CommandList*) { impl->pending = false; Collect(); }
void DlssNr::GpuLifetime::Retire(std::function<void()> destroy)
{
    impl->retired.push_back(std::move(destroy));
    Collect();
}
void DlssNr::GpuLifetime::Collect()
{
    if (impl->pending) return;
    for (auto& destroy : impl->retired) destroy();
    impl->retired.clear();
}
bool DlssNr::GpuLifetime::Idle() { return !impl->pending; }

// NVIDIA's DX11 table accepts bridge resources through void*, but ignores DX12 setters.
struct Dx11Parameters : Mock::Params
{
    using Mock::Params::Get;
    using Mock::Params::Set;
    void Set(const char*, ID3D12Resource*) override {}
    NVSDK_NGX_Result Get(const char*, ID3D12Resource**) const override { return NVSDK_NGX_Result_Fail; }
};

int main()
{
    DlssNr::Proxy::Context proxy;
    ID3D12Device device;
    ID3D12GraphicsCommandList commands;
    ID3D12Resource color, depth, motion, output;
    bool evaluated = true;
    uint64_t epoch = 0;
    DlssNr::Proxy::Settings settings { 0, 0, 0.5f, 1.0f, 0.0f, -1.0f, true };
    auto run = [&](bool advance = true)
    {
        if (advance)
            ++epoch;
        return proxy.Run(&commands, &device, &color, &depth, &motion, &output, 1920, 1080, 1280, 720, 1920, 1080, 12,
                         24, 32, 48, true, false, 0.5f, -0.25f, settings, epoch, &evaluated);
    };
    auto value = []<typename T>(const char* key)
    {
        T result {};
        assert(Mock::latest->Get(key, &result) == NVSDK_NGX_Result_Success);
        return result;
    };

    // Creation has its own frame, and setters must preserve SDK types (not raw vtable offsets).
    assert(run() == NVSDK_NGX_Result_Success && !evaluated);
    assert(Mock::creations == 1 && Mock::evaluations == 0);
    assert(run(false) == NVSDK_NGX_Result_Success && !evaluated);
    assert(Mock::creations == 1 && Mock::evaluations == 0);
    assert(value.operator()<unsigned int>("DLSSNR.Width") == 1920);
    assert(value.operator()<unsigned int>("DLSSNR.Hint.Render.Preset") == 0);
    assert(value.operator()<float>("DLSSNR.Intensity") == 0.5f);
    assert(!proxy.Ready(epoch));
    CompletionMock::complete = true;
    assert(proxy.Ready(epoch));
    CompletionMock::complete = false;
    assert(run(false) == NVSDK_NGX_Result_Success && evaluated);
    assert(value.operator()<ID3D12Resource*>("DLSSNR.Color") == &color);
    assert(value.operator()<ID3D12Resource*>("DLSSNR.Output") == &output);
    // High-resolution motion and guide offsets must survive the typed NGX dispatch independently.
    assert(value.operator()<unsigned int>("DLSSNR.DepthSubrectWidth") == 1280);
    assert(value.operator()<unsigned int>("DLSSNR.DepthSubrectHeight") == 720);
    assert(value.operator()<unsigned int>("DLSSNR.DepthSubrectBaseX") == 12);
    assert(value.operator()<unsigned int>("DLSSNR.DepthSubrectBaseY") == 24);
    assert(value.operator()<unsigned int>("DLSSNR.MVecSubrectWidth") == 1920);
    assert(value.operator()<unsigned int>("DLSSNR.MVecSubrectHeight") == 1080);
    assert(value.operator()<unsigned int>("DLSSNR.MVecSubrectBaseX") == 32);
    assert(value.operator()<unsigned int>("DLSSNR.MVecSubrectBaseY") == 48);
    assert(value.operator()<float>("DLSSNR.MVecScaleY") == -0.25f);
    assert(value.operator()<unsigned int>("DLSSNR.Reset") == 1);
    assert(run() == NVSDK_NGX_Result_Success && evaluated);
    assert(value.operator()<unsigned int>("DLSSNR.Reset") == 0);

    // Creation-time tuning edits rebuild; the previous GPU feature/map survive the retirement window.
    settings.preset = 2;
    assert(run() == NVSDK_NGX_Result_Success && !evaluated);
    assert(Mock::creations == 2 && Mock::releases == 0 && Mock::destructions == 0);
    assert(value.operator()<unsigned int>("DLSSNR.Hint.Render.Preset") == 2);
    for (int call = 0; call < 40; ++call)
        assert(run(false) == NVSDK_NGX_Result_Success && !evaluated);
    assert(Mock::releases == 0 && Mock::destructions == 0);
    for (int frame = 0; frame < 31; ++frame)
        assert(run() == NVSDK_NGX_Result_Success && evaluated);
    assert(Mock::releases == 0 && Mock::destructions == 0);
    assert(run() == NVSDK_NGX_Result_Success && evaluated);
    assert(Mock::releases == 0 && Mock::destructions == 0); // CPU epochs are not completion.
    proxy.ResetRecording(&commands);
    assert(Mock::releases == 1 && Mock::destructions == 1);

    // A real NGX failure reaches the caller and stays latched until an explicit retry.
    Mock::evaluateResult = NVSDK_NGX_Result_FAIL_UnableToInitializeFeature;
    assert(run() == (unsigned int) Mock::evaluateResult && !evaluated);
    auto evaluationsBefore = Mock::evaluations;
    assert(run() == 0 && !evaluated && Mock::evaluations == evaluationsBefore);
    proxy.RetryAfterFailure();
    Mock::evaluateResult = NVSDK_NGX_Result_Success;
    Mock::createResult = NVSDK_NGX_Result_FAIL_UnableToInitializeFeature;
    assert(run() == (unsigned int) Mock::createResult && !evaluated);
    auto creationsBefore = Mock::creations;
    assert(run() == 0 && Mock::creations == creationsBefore);
    proxy.RetryAfterFailure();
    Mock::createResult = NVSDK_NGX_Result_Success;
    assert(run() == NVSDK_NGX_Result_Success && !evaluated);
    assert(run() == NVSDK_NGX_Result_Success && evaluated);

    proxy.ResetRecording(&commands);
    proxy.Release();
    assert(Mock::handles.empty());
    assert(Mock::allocations == Mock::destructions);
    proxy.ResetRecording(&commands);
    proxy.Release(); // Idempotent shutdown.
    assert(Mock::allocations == Mock::destructions);

    // Alternating upscalers own independent features, parameter maps, failure latches and teardown.
    {
        DlssNr::Proxy::Context other;
        assert(run() == NVSDK_NGX_Result_Success && !evaluated);
        auto* firstParams = Mock::latest;
        DlssNr::Proxy::Settings otherSettings { 3, 2, 0.75f, 0.4f, 0.3f, 0.2f, false };
        auto runOther = [&]
        {
            return other.Run(&commands, &device, &color, &depth, &motion, &output, 1280, 720, 1280, 720, 1280, 720, 0,
                             0, 0, 0, false, false, 1.0f, 1.0f, otherSettings, ++epoch, &evaluated);
        };
        assert(runOther() == NVSDK_NGX_Result_Success && !evaluated);
        auto* secondParams = Mock::latest;
        assert(value.operator()<unsigned int>("DLSSNR.Hint.Render.Preset") == 3);
        assert(value.operator()<unsigned int>("DLSSNR.Style") == 2);
        assert(value.operator()<float>("DLSSNR.LocalToneStrength") == 0.3f);
        assert(value.operator()<unsigned int>("DLSSNR.UseAutoMask") == 0);
        assert(firstParams != secondParams && Mock::handles.size() == 2);
        auto creations = Mock::creations;
        assert(run() == NVSDK_NGX_Result_Success && evaluated);
        assert(runOther() == NVSDK_NGX_Result_Success && evaluated);
        assert(Mock::creations == creations);
        Mock::evaluateResult = NVSDK_NGX_Result_FAIL_UnableToInitializeFeature;
        assert(run() == (unsigned int) Mock::evaluateResult && !evaluated);
        Mock::evaluateResult = NVSDK_NGX_Result_Success;
        assert(runOther() == NVSDK_NGX_Result_Success && evaluated);
        proxy.ResetRecording(&commands);
        proxy.Release();
        assert(Mock::handles.size() == 1);
        assert(runOther() == NVSDK_NGX_Result_Success && evaluated);
        other.ResetRecording(&commands);
    }
    assert(Mock::handles.empty() && Mock::allocations == Mock::destructions);

    // Release is safe even when called while creation commands remain unsubmitted.
    assert(run() == NVSDK_NGX_Result_Success && !evaluated);
    proxy.Release();
    assert(Mock::handles.size() == 1);
    for (int frame = 0; frame < 100; ++frame) proxy.AdvanceEpoch(++epoch);
    assert(Mock::handles.size() == 1);
    proxy.ResetRecording(&commands);
    assert(Mock::handles.empty() && Mock::allocations == Mock::destructions);

    // Only the earlier driver initialization rejection tried compatibility loading.
    assert(CompatibilityMock::opens == 1);
    Mock::createResult = NVSDK_NGX_Result_FAIL_UnableToInitializeFeature;
    proxy.RetryAfterFailure();
    assert(run() == NVSDK_NGX_Result_FAIL_UnableToInitializeFeature && !evaluated);
    proxy.ResetRecording(&commands);
    assert(CompatibilityMock::opens == 2 && Mock::handles.empty());

    CompatibilityMock::available = true;
    proxy.RetryAfterFailure();
    assert(run() == NVSDK_NGX_Result_Success && !evaluated);
    proxy.ResetRecording(&commands);
    assert(run() == NVSDK_NGX_Result_Success && evaluated);
    assert(CompatibilityMock::creates == 1 && CompatibilityMock::evaluates == 1);
    proxy.Release();
    assert(CompatibilityMock::releases == 0 && CompatibilityMock::destroyed == 0);
    proxy.ResetRecording(&commands);
    assert(CompatibilityMock::releases == 1 && CompatibilityMock::destroyed == 1);
    assert(Mock::handles.empty() && Mock::allocations == Mock::destructions);

    // Direct creation failures also retain backend ownership until recording is retired.
    CompatibilityMock::createResult = NVSDK_NGX_Result_Fail;
    proxy.RetryAfterFailure();
    assert(run() == NVSDK_NGX_Result_Fail && !evaluated);
    assert(CompatibilityMock::destroyed == 1);
    proxy.ResetRecording(&commands);
    assert(CompatibilityMock::destroyed == 2 && Mock::allocations == Mock::destructions);
    const auto attempts = CompatibilityMock::opens;
    CompatibilityMock::createResult = NVSDK_NGX_Result_Success;
    Mock::createResult = NVSDK_NGX_Result_Fail;
    proxy.RetryAfterFailure();
    assert(run() == NVSDK_NGX_Result_Success && !evaluated);
    proxy.Release();
    proxy.ResetRecording(&commands);
    assert(CompatibilityMock::opens == attempts + 1 && Mock::allocations == Mock::destructions);
    Mock::createResult = NVSDK_NGX_Result_Success;

    // Clearing an older owner must not erase the current shader's menu snapshot.
    DlssNr::StatusSnapshot status;
    status.running = true;
    status.frames = 5;
    DlssNr::PublishStatus(&color, DlssNr::Backend::Dx12, status);
    DlssNr::PublishStatus(&output, DlssNr::Backend::Dx12, status);
    DlssNr::ClearStatus(&color);
    assert(DlssNr::IsRunning());
    DlssNr::ClearStatus(&output);
    assert(!DlssNr::IsRunning());
    const auto requestsBefore = DlssNr::ReadControlRequests();
    DlssNr::RetryAfterFailure();
    DlssNr::RequestCapture(8);
    const auto requestsAfter = DlssNr::ReadControlRequests();
    assert(requestsAfter.retryGeneration == requestsBefore.retryGeneration + 1);
    assert(requestsAfter.captureGeneration == requestsBefore.captureGeneration + 1);
    assert(requestsAfter.captureFrames == 8);

    // The shared runner routes resources backwards, then executes stages forwards.
    ID3D12Resource intermediate;
    std::string order;
    ShaderPipeline_Dx12 pipeline { { [&](ID3D12Resource* target)
                                     {
                                         assert(target == &intermediate);
                                         order += 'a';
                                         return &color;
                                     },
                                     [&](ID3D12Resource* input, ID3D12Resource* target)
                                     {
                                         assert(input == &color && target == &intermediate);
                                         order += 'A';
                                         return true;
                                     } },
                                   { [&](ID3D12Resource* target)
                                     {
                                         assert(target == &output);
                                         order += 'b';
                                         return &intermediate;
                                     },
                                     [&](ID3D12Resource* input, ID3D12Resource* target)
                                     {
                                         assert(input == &intermediate && target == &output);
                                         order += 'B';
                                         return false;
                                     } } };
    assert(SetupShaderPipeline(pipeline, &output) == &color);
    assert(!DispatchShaderPipeline(pipeline));
    assert(order == "baAB");

    // Bridge parameter maps may store void*; restoration preserves both pointer and SDK type.
    Mock::Params parameters;
    parameters.Set(NVSDK_NGX_Parameter_Color, static_cast<void*>(&color));
    parameters.Set(NVSDK_NGX_Parameter_Output, &output);
    {
        RestoreUpscalerResources_Dx12 restore(&parameters);
        parameters.Set(NVSDK_NGX_Parameter_Color, &intermediate);
        parameters.Set(NVSDK_NGX_Parameter_Output, &intermediate);
        assert(!DispatchShaderPipeline(pipeline));
    }
    void* restoredColor = nullptr;
    ID3D12Resource* restoredOutput = nullptr;
    assert(parameters.Get(NVSDK_NGX_Parameter_Color, &restoredColor) == NVSDK_NGX_Result_Success);
    assert(parameters.Get(NVSDK_NGX_Parameter_Output, &restoredOutput) == NVSDK_NGX_Result_Success);
    assert(restoredColor == &color && restoredOutput == &output);

    Dx11Parameters bridgeParameters;
    bridgeParameters.Set(NVSDK_NGX_Parameter_Color, static_cast<void*>(&color));
    bridgeParameters.Set(NVSDK_NGX_Parameter_Output, static_cast<void*>(&output));
    // Reproduce both failed substitutions: unchanged colour bypasses pre-SR NR; unchanged
    // output leaves the post-SR NR input unwritten even though upscaling reports success.
    bridgeParameters.Set(NVSDK_NGX_Parameter_Color, &intermediate);
    bridgeParameters.Set(NVSDK_NGX_Parameter_Output, &intermediate);
    assert(GetUpscalerResource_Dx12(&bridgeParameters, NVSDK_NGX_Parameter_Color) == &color);
    assert(GetUpscalerResource_Dx12(&bridgeParameters, NVSDK_NGX_Parameter_Output) == &output);

    for (NVSDK_NGX_Parameter* table : { static_cast<NVSDK_NGX_Parameter*>(&parameters),
                                       static_cast<NVSDK_NGX_Parameter*>(&bridgeParameters) })
    {
        NrHoldParameters_Dx12 held;
        table->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.25f);
        table->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 2.0f);
        table->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, 1024u);
        table->Set(NVSDK_NGX_Parameter_Reset, 0u);
        held.Capture(table);
        table->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, -0.5f);
        table->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 4.0f);
        table->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, 1280u);
        held.Apply(table);
        float jitter = 0, exposure = 0;
        unsigned width = 0, reset = 0;
        table->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitter);
        table->Get(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, &exposure);
        table->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, &width);
        table->Get(NVSDK_NGX_Parameter_Reset, &reset);
        assert(jitter == 0.25f && exposure == 2.0f && width == 1024u && reset == 1u);
        assert(!DispatchShaderPipeline(pipeline)); // restoration must also work after a failed evaluate
        held.Restore(table);
        table->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitter);
        table->Get(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, &exposure);
        table->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, &width);
        table->Get(NVSDK_NGX_Parameter_Reset, &reset);
        assert(jitter == -0.5f && exposure == 4.0f && width == 1280u && reset == 0u);
        held.Apply(table, false); // first live frame resets history, without restoring held metadata
        table->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitter);
        assert(jitter == -0.5f);
        held.Restore(table);
        {
            RestoreUpscalerResources_Dx12 restore(table);
            SetUpscalerResource_Dx12(table, NVSDK_NGX_Parameter_Color, &intermediate);
            SetUpscalerResource_Dx12(table, NVSDK_NGX_Parameter_Output, &intermediate);
            assert(GetUpscalerResource_Dx12(table, NVSDK_NGX_Parameter_Color) == &intermediate);
            assert(GetUpscalerResource_Dx12(table, NVSDK_NGX_Parameter_Output) == &intermediate);
        }
        assert(GetUpscalerResource_Dx12(table, NVSDK_NGX_Parameter_Color) == &color);
        assert(GetUpscalerResource_Dx12(table, NVSDK_NGX_Parameter_Output) == &output);
    }
}
