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

namespace DlssNr::Nvfp4
{
unsigned hostsCreated = 0, hostsDestroyed = 0;
HostPass lastPass;
std::shared_ptr<void> CreateHost(ID3D12Device*, unsigned, unsigned, unsigned, const std::filesystem::path&)
{
    ++hostsCreated;
    return std::shared_ptr<void>(new int{}, [](int* host) { delete host; ++hostsDestroyed; });
}
bool RunHost(const std::shared_ptr<void>&, ID3D12GraphicsCommandList*, const HostPass& pass)
{
    lastPass = pass;
    return true;
}
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

// What the tracker says became of the command list a creation was recorded on. Untracked (no submission
// notifications) leaves the submission epoch to decide, as the tests of creation timing below expect.
namespace CompletionMock
{
bool complete = false;
auto recorded = DlssNr::GpuLifetime::Recorded::Untracked;
}
// NGX tests substitute completion only; nr_gpu_lifetime_smoke exercises real D3D12 fences.
struct DlssNr::GpuLifetime::Impl
{
    bool pending = false;
    std::vector<std::function<void()>> retired;
};
DlssNr::GpuLifetime::GpuLifetime() : impl(std::make_unique<Impl>()) {}
DlssNr::GpuLifetime::~GpuLifetime() { Collect(); }
void DlssNr::GpuLifetime::Record(ID3D12GraphicsCommandList*) { impl->pending = true; }
std::function<DlssNr::GpuLifetime::Recorded()> DlssNr::GpuLifetime::RecordingProbe(ID3D12GraphicsCommandList*)
{ return [] { return CompletionMock::complete ? Recorded::Finished : CompletionMock::recorded; }; }
void DlssNr::GpuLifetime::Submitted(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*) {}
void DlssNr::GpuLifetime::ResetRecording(ID3D12CommandList*) { impl->pending = false; Collect(); }
void DlssNr::GpuLifetime::BeginGeneration() {} // Recording/fence generations are tested with the real helper.
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
void DlssNr::GpuLifetime::FinishSubmitted() { Collect(); }

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
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    ComPtr<ID3D12Device> warpDevice;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    assert(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))));
    assert(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))));
    assert(SUCCEEDED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warpDevice))));
    auto& device = *warpDevice.Get();
    assert(SUCCEEDED(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    assert(SUCCEEDED(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))));
    auto& commands = *list.Get();
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 64;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R16G16_FLOAT;
    ComPtr<ID3D12Resource> textures[5];
    for (auto& texture : textures)
        assert(SUCCEEDED(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&texture))));
    auto& color = *textures[0].Get();
    auto& depth = *textures[1].Get();
    auto& motion = *textures[2].Get();
    auto& output = *textures[3].Get();
    bool evaluated = true;
    uint64_t epoch = 0;
    DlssNr::ModelSettings settings { 0, 0, 0.5f, 1.0f, 0.0f, -1.0f, true };
    auto run = [&](bool advance = true)
    {
        if (advance)
            ++epoch;
        const DlssNr::Proxy::Frame frame { &color, &depth, &motion, &output, { 1920, 1080 },
            { { 12, 24, 1280, 720 }, { 32, 48, 1920, 1080 } }, true, false, 0.5f, -0.25f };
        return proxy.Run(&commands, &device, frame, settings, epoch, &evaluated);
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
        DlssNr::ModelSettings otherSettings { 3, 2, 0.75f, 0.4f, 0.3f, 0.2f, false };
        auto runOther = [&]
        {
            const DlssNr::Proxy::Frame frame { &color, &depth, &motion, &output, { 1280, 720 },
                { { 0, 0, 1280, 720 }, { 0, 0, 1280, 720 } } };
            return other.Run(&commands, &device, frame, otherSettings, ++epoch, &evaluated);
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
    for (int frame = 0; frame < 100; ++frame) proxy.Collect();
    assert(Mock::handles.size() == 1);
    proxy.ResetRecording(&commands);
    assert(Mock::handles.empty() && Mock::allocations == Mock::destructions);

    // A feature whose setup never reached a queue is never evaluated. While its command list is still open it waits,
    // whatever the epoch; once the list is reset or released unexecuted the feature is retired and built again.
    {
        using Recorded = DlssNr::GpuLifetime::Recorded;
        proxy.RetryAfterFailure();
        proxy.ResetRecording(&commands);
        auto creations = Mock::creations, evaluations = Mock::evaluations;
        CompletionMock::recorded = Recorded::Open;
        assert(run() == NVSDK_NGX_Result_Success && !evaluated && Mock::creations == creations + 1);
        for (int frame = 0; frame < 5; ++frame)
            assert(run() == NVSDK_NGX_Result_Success && !evaluated);
        assert(Mock::creations == creations + 1 && Mock::evaluations == evaluations);
        CompletionMock::recorded = Recorded::Discarded;
        assert(run() == NVSDK_NGX_Result_Success && !evaluated);
        assert(Mock::creations == creations + 2 && Mock::evaluations == evaluations); // built again, not evaluated
        proxy.ResetRecording(&commands);
        assert(Mock::handles.size() == 1); // the discarded feature was released unused
        // Given to a queue: ready in the next submission epoch, not in the one it was recorded in.
        CompletionMock::recorded = Recorded::Submitted;
        assert(run(false) == NVSDK_NGX_Result_Success && !evaluated);
        assert(run() == NVSDK_NGX_Result_Success && evaluated && Mock::creations == creations + 2);
        // Once ready it stays ready: later discards of other recordings do not rebuild a working feature.
        CompletionMock::recorded = Recorded::Discarded;
        assert(run() == NVSDK_NGX_Result_Success && evaluated && Mock::creations == creations + 2);

        // Three creations discarded in a row latch a failure instead of creating for ever; Retry clears it.
        proxy.RetryAfterFailure();
        proxy.ResetRecording(&commands);
        creations = Mock::creations;
        evaluations = Mock::evaluations;
        assert(run() == NVSDK_NGX_Result_Success && !evaluated); // created; its list is then discarded
        assert(run() == NVSDK_NGX_Result_Success && !evaluated); // discard 1: created again
        assert(run() == NVSDK_NGX_Result_Success && !evaluated); // discard 2: created again
        assert(Mock::creations == creations + 3);
        assert(run() == NVSDK_NGX_Result_Fail && !evaluated);    // discard 3: latched
        assert(run() == 0 && !evaluated && Mock::creations == creations + 3 && Mock::evaluations == evaluations);
        proxy.ResetRecording(&commands);
        assert(Mock::handles.empty() && Mock::allocations == Mock::destructions);
        CompletionMock::recorded = Recorded::Untracked;
        proxy.RetryAfterFailure();
        assert(run() == NVSDK_NGX_Result_Success && !evaluated);
        assert(run() == NVSDK_NGX_Result_Success && evaluated);
        proxy.ResetRecording(&commands);
        proxy.Release();
        proxy.ResetRecording(&commands);
        assert(Mock::handles.empty() && Mock::allocations == Mock::destructions);
    }

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

    // Calibration retunes a live INT8 host. Both the pipeline preflight and direct Run callers use this path.
    {
        using namespace DlssNr::Nvfp4;
        DlssNr::Proxy::Context int8;
        DlssNr::ModelSettings tuning {};
        tuning.requested = ModelInt8;
        DlssNr::Proxy::Frame frame { &color, &depth, &motion, &output, { 64, 64 },
                                     { { 0, 0, 64, 64 }, { 0, 0, 64, 64 } } };
        auto evaluate = [&] { assert(int8.Run(&commands, &device, frame, tuning, ++epoch, &evaluated) ==
                                     NVSDK_NGX_Result_Success && evaluated); };
        for (unsigned trial = 1; trial <= 64; ++trial)
        {
            tuning.intensity = float(trial) / 64;
            if (trial % 2 == 0)
                assert(int8.UpdateTuning(tuning));
            evaluate();
            assert(hostsCreated == 1 && hostsDestroyed == 0 && lastPass.intensity == tuning.intensity && lastPass.reset);
            evaluate();
            assert(!lastPass.reset);
        }
        // Moving an active region within the motion atlas preserves history and the game's vector units.
        frame.guides.motion = { 4, 8, 48, 40 };
        frame.mvScaleX = -64.0f;
        frame.mvScaleY = 32.0f;
        evaluate();
        assert(lastPass.motion && lastPass.motion != &motion && !lastPass.reset);
        assert(lastPass.motion->GetDesc().Width == 48 && lastPass.motion->GetDesc().Height == 40);
        assert(lastPass.mvScaleX == -64.0f && lastPass.mvScaleY == 32.0f);
        const auto cropped = lastPass.motion;
        frame.guides.motion.x = 8;
        evaluate();
        assert(lastPass.motion == cropped && !lastPass.reset);
        tuning.requested = ModelInt8Double;
        evaluate();
        assert(hostsCreated == 2 && hostsDestroyed == 0 && lastPass.reset);
        int8.Release();
        assert(hostsDestroyed == 0);
        int8.ResetRecording(&commands);
        assert(hostsDestroyed == 2);
    }

    // Clearing an older owner must not erase the current shader's menu snapshot.
    DlssNr::StatusSnapshot status;
    status.running = true;
    status.frames = 5;
    DlssNr::PublishStatus(&color, DlssNr::Backend::Dx12, status);
    DlssNr::PublishStatus(&output, DlssNr::Backend::Dx12, status);
    DlssNr::ClearStatus(&color);
    assert(DlssNr::ReadStatus(DlssNr::Backend::Dx12).running);
    DlssNr::ClearStatus(&output);
    assert(!DlssNr::ReadStatus(DlssNr::Backend::Dx12).running);
    const auto requestsBefore = DlssNr::ReadControlRequests();
    DlssNr::RetryAfterFailure();
    DlssNr::RequestCapture(8);
    const auto requestsAfter = DlssNr::ReadControlRequests();
    assert(requestsAfter.retryGeneration == requestsBefore.retryGeneration + 1);
    assert(requestsAfter.captureGeneration == requestsBefore.captureGeneration + 1);
    assert(requestsAfter.captureFrames == 8);

    // The shared runner routes resources backwards, then executes stages forwards.
    auto& intermediate = *textures[4].Get();
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
