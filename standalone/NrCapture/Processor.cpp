#include "Processor.h"
#include "nvfp4/NrFfnV2.h"
#include "nvfp4/NrSwin4.h"
#include "nvfp4/NrNet4.h"
#include "nvfp4/NrFp8.h"
#include "nvfp4/NrAmpere.h"
#include "nvfp4/NrPassRuntime.h"
#include <atomic>
#include <mutex>
#include <shaders/dlssnr/precompile/DlssNr_Shader.h>
#include <CaptureColour.h>
#include <array>
namespace nr {
Processor::Processor(Gpu& g, const std::filesystem::path& path, bool unlockAdaMfg) : Processor(g) {
    runtime = std::make_shared<Runtime>(g.device.Get(), path, unlockAdaMfg);
}
Processor::Processor(Gpu& g, const Processor& sharing) : Processor(g) {
    runtime = std::make_shared<Runtime>(sharing.runtime);
}
Processor::Processor(Gpu& g) : gpu(g) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 5};
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {2, ranges};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[1].Descriptor.ShaderRegister = 0;
    D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_ROOT_SIGNATURE_DESC signature{2, parameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, error;
    Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), "NR root signature");
    Check(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "NR root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature = root.Get();
    pipeline.CS = {DlssNr_cso, sizeof(DlssNr_cso)};
    Check(gpu.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&shader)), "NR colour shader");
    pipeline.CS = {CaptureColour, sizeof(CaptureColour)};
    Check(gpu.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&captureColour)), "Capture colour conversion");
    D3D12_DESCRIPTOR_HEAP_DESC h{}; h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    h.NumDescriptors = SlotCount; h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gpu.device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&heap)), "NR descriptors");
    stride = gpu.device->GetDescriptorHandleIncrementSize(h.Type);
    constants = gpu.Buffer(SlotCount / 7 * 256, D3D12_HEAP_TYPE_UPLOAD);
}
Processor::~Processor() {
    stages.clear();
    // Share the same fence-backed retention used by the in-game NR implementation.
    gpu.lifetime.Retire([model = std::move(runtime), nativeHosts = std::move(hosts), flow = std::move(opticalFlow), a = std::move(proxy), b = std::move(original),
        c = std::move(answer), d = std::move(output), e = std::move(depth), f = std::move(motion),
        full = std::move(fullProxy), flowSource = std::move(flowInput), filtered = std::move(filteredAnswer),
        srInput = std::move(resizeInput), upscaled = std::move(enlarged), bounded = std::move(boundedDlss), unit = std::move(exposure),
        encoded = std::move(sdrInput), composed = std::move(sdrOutput), held = std::move(heldInput), conversion = std::move(captureColour),
        labels = std::move(labelled), filters = std::move(scaling), native = std::move(nativeAnswer), next = std::move(passInput),
        spatial = std::move(spatialShader), packed = std::move(packedProxy), packedResult = std::move(packedAnswer),
        packedZ = std::move(packedDepth), packedMv = std::move(packedMotion), packedNext = std::move(packedPassInput),
        grid = std::move(gridOutput), chain = std::move(chainInput),
        steady = std::move(steadyAnswer), keptA = std::move(keptEdit[0]), keptB = std::move(keptEdit[1]),
        extended = std::move(reapplied), packedExtended = std::move(packedReapplied), chainExtended = std::move(chainReapplied),
        cb = std::move(constants), descriptors = std::move(heap), rs = std::move(root), pso = std::move(shader)] {});
    gpu.lifetime.Collect();
}
D3D12_CPU_DESCRIPTOR_HANDLE Processor::Cpu(UINT i) const {
    auto h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * stride; return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE Processor::GpuHandle(UINT i) const {
    auto h = heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * stride; return h;
}
// Settings::editInterval and editLate, or what a check forces: NR_EDIT_INTERVAL=n; NR_EDIT_LATE=0/1, or "shift": a
// variant with an interval n > 1 runs late with the interval n - 1, and one without stays as it is (so that one run
// of a checker holds the plain reference beside "late, every frame" as its /2).
struct EditPlan { unsigned interval; bool late; };
static EditPlan PlanEdit(const Settings& s) {
    static const unsigned forced = [] { const wchar_t* v = _wgetenv(L"NR_EDIT_INTERVAL"); return v ? (unsigned)std::clamp(_wtoi(v), 0, 4) : 0u; }();
    static const int lateForced = [] { const wchar_t* v = _wgetenv(L"NR_EDIT_LATE"); return !v ? -1 : !_wcsicmp(v, L"shift") ? 2 : int(_wtoi(v) != 0); }();
    EditPlan plan{forced ? forced : s.editInterval, lateForced < 0 ? s.editLate : lateForced == 1};
    if (lateForced == 2 && plan.interval > 1) { plan.late = true; --plan.interval; }
    return plan;
}
void Processor::ResizeLegacy(UINT w, UINT h, const Settings& s) {
    if (!w || !h || w > 7680 || h > 4320) throw std::runtime_error("Capture extent is outside the prototype's 8K limit.");
    if (s.passes < 1 || s.passes > Settings::MaxPasses) throw std::runtime_error("Choose 1-10 NR passes.");
    if (s.modelScale < 25 || s.modelScale > 200) throw std::runtime_error("Model resolution must be 25-200%.");
    if (s.upscaler > 1 || s.transfer > 3)
        throw std::runtime_error("Unknown upscaler or edit transfer.");
    const UINT mw = std::max(1u, (w * s.modelScale + 50) / 100);
    const UINT mh = std::max(1u, (h * s.modelScale + 50) / 100);
    auto nextSpatial = DlssNr::Spatial::Build(s.spatial, w, h, s.modelScale * .01f);
    // Preserve the legacy integer rounding for ordinary resources. The shared
    // float-scale builder can round half pixels differently for odd extents.
    nextSpatial.ordinaryW = mw; nextSpatial.ordinaryH = mh;
    if (nextSpatial.requested && !nextSpatial.active && nextSpatial.reason != "work is 100% on both axes")
        throw std::runtime_error("Invalid foveation layout: " + nextSpatial.reason);
    const bool spatialChanged = spatialLayout != nextSpatial;
    const bool resized = width != w || height != h || modelWidth != mw || modelHeight != mh || spatialChanged ||
        (!composeOnly && !proxy);
    const bool useDlss = s.upscaler == 1 && s.transfer != 2 && (mw < w || mh < h);
    // A 100 % pass with an edit interval keeps its edit the same way as a reduced one: the difference of answer and
    // input, here at the full size (transfer 4 in the shader: stored as it is, no carrier, no enlargement).
    const EditPlan editPlan = PlanEdit(s);
    const bool fullField = mw == w && mh == h && !nextSpatial.active && !composeOnly && (editPlan.interval > 1 || editPlan.late);
    if (s.modelScale > 100 && !scaling) scaling = std::make_shared<Scaling>(gpu);
    if (resized || (!s.estimateMotion && flowInput) || (s.estimateMotion && (mw != w || mh != h) && !flowInput)) {
        if (!gpu.Drain()) throw std::runtime_error("GPU timeout resizing model inputs.");
        flowInput.Reset();
        if (s.estimateMotion && (mw != w || mh != h)) flowInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    if (s.estimateMotion) {
        if (!opticalFlow) {
            if (!gpu.Drain()) throw std::runtime_error("GPU timeout enabling optical flow.");
            opticalFlow = std::make_shared<Motion>(gpu);
        }
        opticalFlow->Resize(mw, mh);
    }
    if (!resized && activeModels == s.Models() && activePasses == s.passes && activeReapply == s.ReapplyMask() &&
        privateDlss == useDlss && activeFullField == fullField) return;
    if (opticalFlow) opticalFlow->Reset();
    if (!gpu.Drain()) throw std::runtime_error("GPU timeout while rebuilding NR.");
    for (size_t pass = 0; pass < hosts.size(); ++pass) {
        if (hosts[pass] && (pass >= s.passes || hostModels[pass] != s.PassModel(unsigned(pass)).preset || resized))
            hosts[pass].reset();
    }
    spatialLayout = nextSpatial;
    runtime->ReleaseDlss(); privateDlss = false;
    resizeInput.Reset(); enlarged.Reset(); boundedDlss.Reset(); exposure.Reset();
    if (mw < w || mh < h || fullField) resizeInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    fieldValid = false; fieldAge = 0; activeFullField = fullField;
    keptValid = false; // (Settings::editSmoothing: the kept edit belonged to the features just released)
    if (s.passes == 1) passInput.Reset();
    else if (!passInput || resized) passInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (resized) {
        packedProxy.Reset(); packedAnswer.Reset(); packedDepth.Reset(); packedMotion.Reset(); packedPassInput.Reset();
        gridOutput.Reset(); gridReadable = false;
        filteredAnswer.Reset(); reapplied.Reset(); packedReapplied.Reset();
        steadyAnswer.Reset(); keptEdit[0].Reset(); keptEdit[1].Reset(); keptValid = false;
        labelled.Reset(); labelsReadable = false;
        nativeAnswer.Reset();
        if (mw != w || mh != h) nativeAnswer = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (width != w || height != h) { heldInput.Reset(); heldValid = false; }
        sdrInput.Reset(); sdrOutput.Reset();
        if (!composeOnly) proxy = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        fullProxy.Reset();
        if (mw != w || mh != h || spatialLayout.active) fullProxy = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        original = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (!composeOnly) answer = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        output = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (!composeOnly) {
            depth = gpu.Texture(mw, mh, DXGI_FORMAT_R32_FLOAT);
            motion = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16_FLOAT);
        }
        outputReadable = false; width = w; height = h;
        modelWidth = mw; modelHeight = mh;
        if (!composeOnly) {
            gpu.Begin();
            ID3D12DescriptorHeap* heaps[]{heap.Get()}; gpu.commands->SetDescriptorHeaps(1, heaps);
            const float one[]{1, 1, 1, 1}, zero[]{0, 0, 0, 0};
            gpu.device->CreateUnorderedAccessView(depth.Get(), nullptr, nullptr, Cpu(63));
            gpu.device->CreateUnorderedAccessView(motion.Get(), nullptr, nullptr, Cpu(64));
            gpu.commands->ClearUnorderedAccessViewFloat(GpuHandle(63), Cpu(63), depth.Get(), one, 0, nullptr);
            gpu.commands->ClearUnorderedAccessViewFloat(GpuHandle(64), Cpu(64), motion.Get(), zero, 0, nullptr);
            Barrier(gpu.commands.Get(), depth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(gpu.commands.Get(), motion.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Guide initialization timeout.");
        }
    }
    if (mw == w && mh == h) {
        if (!fullField) nativeAnswer.Reset();
        else if (!nativeAnswer) nativeAnswer = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    if (composeOnly) {
        // The host only composes the children's answer against preparedProxy.
        // Do not retain a second unused native NR input/output and guide set.
        proxy.Reset(); answer.Reset(); depth.Reset(); motion.Reset(); sdrInput.Reset();
    }
    if (spatialLayout.active) {
        if (!spatialShader) spatialShader = std::make_shared<Spatial>(gpu);
        const auto sw = spatialLayout.modelW, sh = spatialLayout.modelH;
        if (!packedProxy) {
            packedProxy = gpu.Texture(sw, sh, DXGI_FORMAT_R16G16B16A16_FLOAT);
            packedAnswer = gpu.Texture(sw, sh, DXGI_FORMAT_R16G16B16A16_FLOAT);
            packedDepth = gpu.Texture(sw, sh, DXGI_FORMAT_R32_FLOAT);
            packedMotion = gpu.Texture(sw, sh, DXGI_FORMAT_R32G32_FLOAT);
        }
        if (s.passes > 1 && !packedPassInput) packedPassInput = gpu.Texture(sw, sh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    if (composeOnly) runtime->Release();
    else {
        for (unsigned pass = 0; pass < s.passes; ++pass) {
            if (s.Shortcut(pass).reapply) continue;
            const unsigned preset = s.PassModel(pass).preset;
            const unsigned set = Settings::Fp8Set(preset);
            if (!set || NrHost::Requested(gpu.device.Get())) continue;
            ApplyFp8(true);
            std::string missing;
            if (!NrFp8::SetComplete(set, &missing))
                throw std::runtime_error(std::format("NR pass {}: FP8 model {} unavailable: {}", pass + 1, preset, missing));
        }
        gpu.Begin();
        const auto created = runtime->Create(gpu.commands.Get(), spatialLayout.active ? spatialLayout.modelW : mw,
            spatialLayout.active ? spatialLayout.modelH : mh, s.model, s.passes, s.passOverrides, s.ReapplyMask());
        // Even failed creation may record commands: finish its submission before releasing ownership.
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("NR creation timeout.");
        NrPassBorrow::Reset(); // The recorded template and snapshots belonged to the released features.
        Runtime::CheckResult(created, "NR feature creation");
    }
    activeModels = s.Models(); activePasses = s.passes; activeReapply = s.ReapplyMask();
    if (useDlss) {
        enlarged = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        boundedDlss = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        exposure = gpu.Texture(1, 1, DXGI_FORMAT_R32_FLOAT);
        gpu.Begin();
        DlssNrConstants unit{}; unit.Mode = DlssNrMode_UnitExposure; unit.Width = unit.Height = 1;
        Dispatch(42, unit, proxy.Get(), nullptr, nullptr, exposure.Get(), nullptr);
        Barrier(gpu.commands.Get(), exposure.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const auto result = runtime->CreateDlss(gpu.commands.Get(), mw, mh, w, h);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Private DLSS creation timeout.");
        Runtime::CheckResult(result, "Private DLSS creation"); privateDlss = true;
    }
}
void Processor::Dispatch(UINT offset, const DlssNrConstants& s, ID3D12Resource* source,
    ID3D12Resource* model, ID3D12Resource* clean, ID3D12Resource* target, ID3D12Resource* keep, ID3D12Resource* fifth,
    ID3D12Resource* fourth) {
    ID3D12Resource* inputs[]{source, model ? model : source, clean ? clean : source, fourth ? fourth : source, fifth ? fifth : source};
    for (UINT i = 0; i < 5; ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = inputs[i]->GetDesc().Format;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
        gpu.device->CreateShaderResourceView(inputs[i], &view, Cpu(offset + i));
    }
    gpu.device->CreateUnorderedAccessView(target, nullptr, nullptr, Cpu(offset + 5));
    gpu.device->CreateUnorderedAccessView(keep ? keep : target, nullptr, nullptr, Cpu(offset + 6));
    void* data = nullptr; Check(constants->Map(0, nullptr, &data), "Map constants");
    const UINT cbOffset = (offset / 7) * 256; // Each dispatch needs its own descriptors and constants until submission completes.
    memcpy(static_cast<char*>(data) + cbOffset, &s, sizeof(s)); constants->Unmap(0, nullptr);
    auto* c = gpu.commands.Get(); ID3D12DescriptorHeap* heaps[]{heap.Get()};
    c->SetDescriptorHeaps(1, heaps); c->SetComputeRootSignature(root.Get());
    c->SetPipelineState(s.Mode >= 11 && s.Mode <= 23 ? captureColour.Get() : shader.Get());
    c->SetComputeRootDescriptorTable(0, GpuHandle(offset));
    c->SetComputeRootConstantBufferView(1, constants->GetGPUVirtualAddress() + cbOffset);
    c->Dispatch((s.Width + 7) / 8, (s.Height + 7) / 8, 1);
}
bool InstallPassBorrow() {
    NrPassBorrow::SetLog([](int level, const char* message) {
        try { Log(std::string(level >= 2 ? "Pass borrowing error: " : level ? "Pass borrowing warning: " : "Pass borrowing: ") + (message ? message : "")); }
        catch (...) { /* Logging must not escape into the NvAPI hook. */ }
    });
    // The NVFP4 feed-forward layer sits between pass borrowing and the driver; it changes nothing until enabled.
    NrFfnV2::SetLog([](const char* line) { try { Log(line); } catch (...) { /* Logging must not escape into the NvAPI hook. */ } });
    NrSwin4::SetLog([](const char* line) { try { Log(line); } catch (...) { /* As above. */ } });
    NrNet4::SetLog([](const char* line) { try { Log(line); } catch (...) { /* As above. */ } });
    NrFp8::SetLog([](const char* line) { try { Log(line); } catch (...) { /* As above. */ } });
    NrAmpere::SetLog([](const char* line) { try { Log(line); } catch (...) { /* As above. */ } });
    NrHost::SetLog([](const char* line) { try { Log(line); } catch (...) { /* As above. */ } });
    // The FP8 models outermost (they count the runtime's own launches of an evaluation and give its kernels our
    // records; the 4-bit layers are paused in their passes), then the rest-of-the-network layer (it recognises Model A
    // by the runtime's own sequence of launches), then the fused blocks: each of their launches replaces one of the
    // runtime's, so the feed-forward layer below still counts the launches of an evaluation as the runtime made them.
    // The Ampere set (NrAmpere, development: NR_AMP=1) outside them all: it replaces every weighted launch's kernel.
    NrPassBorrow::SetInnerLayer([](unsigned id, void* original) { return NrAmpere::WrapNvapi(id, NrFp8::WrapNvapi(id, NrNet4::WrapNvapi(id, NrSwin4::WrapNvapi(id, NrFfnV2::WrapNvapi(id, NrHost::WrapNvapi(id, original)))))); });
    const char* error = nullptr;
    if (NrPassBorrow::InstallQueryInterfaceHook(&error)) {
        NrHost::SetDriverQuery(NrPassBorrow::DriverQueryInterface()); // the own pass host launches past every layer
        // The runtime asks NvAPI for an entry point when it first needs it. Ask for the two destroy calls now, through
        // the hook, so the NVFP4 layer can destroy its own kernels even before the runtime has destroyed anything.
        if (const HMODULE nvapi = GetModuleHandleW(L"nvapi64.dll"))
            if (const auto query = reinterpret_cast<void* (*)(unsigned)>(GetProcAddress(nvapi, "nvapi_QueryInterface")))
                for (const unsigned id : {0x41c65285u, 0xdf295ea6u}) query(id);     // DestroyCuModule, DestroyCuFunction
        return true;
    }
    Log(std::string("Pass borrowing unavailable: ") + (error ? error : "NvAPI hook installation failed"));
    return false;
}
namespace {
std::atomic<int> nvfp4Applied{-1};
}
void ApplyNvfp4(bool wanted, bool trained, bool everySize) {
    if (const wchar_t* text = _wgetenv(L"NR_FFN_V2_TRAINED"); text && *text) trained = *text != L'0'; // Development override.
    const int state = wanted ? (trained ? 2 : 1) + (everySize ? 2 : 0) : 0;
    if (nvfp4Applied.exchange(state) == state) return;
    if (!wanted) { NrFfnV2::Disable(); Log("NVFP4 feed-forward: off"); return; }
    NrFfnV2::Options options; options.modelFolders = true; options.trained = trained;
    const wchar_t* folder = _wgetenv(L"NR_NVFP4_ASSETS"); // Development override.
    options.assets = folder && *folder ? std::filesystem::path(folder) : ExecutableDirectory() / L"nvfp4";
    // NVIDIA Model B retains its measured size threshold. Native NVFP4 uses the selected precision at every size.
    const wchar_t* least = _wgetenv(L"NR_NVFP4_MIN_TOKENS"); // Development override.
    // A pass-in-one model's feed-forward records are its own: the runtime's kernels would run NVIDIA's weights in their place.
    options.minTokens = least && *least ? unsigned(wcstoul(least, nullptr, 10)) : everySize ? 0 : 2000;
    options.nativeMinTokens = least && *least ? options.minTokens : 0;
    try { NrFfnV2::Enable(options); Log(std::string("NVFP4 feed-forward: on, ") + (trained ? "trained" : "first") + " weights, assets " + Narrow(options.assets.wstring())); }
    catch (const std::exception& e) { NrFfnV2::Disable(); Log(std::string("NVFP4 feed-forward unavailable: ") + e.what()); }
}
std::string Nvfp4Status() { return NrFfnV2::Status(); }
void ReleaseNvfp4(ID3D12Device* device) { NrFfnV2::ReleaseDevice(device); NrSwin4::ReleaseDevice(device); NrNet4::ReleaseDevice(device); NrFp8::ReleaseDevice(device); NrAmpere::ReleaseDevice(device); NrHost::ReleaseDevice(device); }
namespace {
// NR_AMP=1 (development, no preset yet): the FP8 presets' record sets run on the Ampere set's kernels (NrAmpere).
bool AmpereRequested() { const wchar_t* text = _wgetenv(L"NR_AMP"); return text && *text && *text != L'0'; }
void ApplyAmpere(bool wanted) {
    if (!wanted) { if (NrAmpere::Enabled()) { NrAmpere::Disable(); Log("Ampere set: off"); } return; }
    if (NrAmpere::Enabled()) return;
    NrAmpere::Options options;
    const wchar_t* records = _wgetenv(L"NR_FP8_ASSETS"); // The FP8 models' record sets, used unchanged.
    options.records = records && *records ? std::filesystem::path(records) : ExecutableDirectory() / L"nvfp4" / L"fp8";
    const wchar_t* folder = _wgetenv(L"NR_AMP_ASSETS"); // Development override: the cubins.
    options.assets = folder && *folder ? std::filesystem::path(folder) : ExecutableDirectory() / L"nvfp4" / L"amp";
    const wchar_t* manifests = _wgetenv(L"NR_AMP_MANIFESTS"); // Development override: amp_*_manifest.txt (default: the cubins' folder).
    if (manifests && *manifests) options.manifests = manifests;
    try { NrAmpere::FromEnvironment(options); NrAmpere::Enable(options); }
    catch (const std::exception& e) { throw std::runtime_error(std::string("The Ampere set (NR_AMP=1) cannot be loaded: ") + e.what()); }
}
}
void ApplyFp8(bool wanted) {
    ApplyAmpere(wanted && AmpereRequested());
    if (!wanted) { if (NrFp8::Enabled()) { NrFp8::Disable(); Log("FP8 models: off"); } return; }
    if (NrFp8::Enabled()) return;
    NrFp8::Options options;
    const wchar_t* folder = _wgetenv(L"NR_FP8_ASSETS"); // Development override.
    options.assets = folder && *folder ? std::filesystem::path(folder) : ExecutableDirectory() / L"nvfp4" / L"fp8";
    try { NrFp8::Enable(options); }
    catch (const std::exception& e) { throw std::runtime_error(std::string("The FP8 pass-in-one models cannot be loaded: ") + e.what()); }
}
std::string Fp8Status(bool counters) { return NrFp8::Status(counters); }
namespace {
std::mutex nvfp4SwinMutex;
bool nvfp4SwinSet = false;
unsigned nvfp4SwinWidths = 0, nvfp4SwinTwo = 0;
unsigned long long nvfp4SwinLayers = 0;
unsigned nvfp4SwinTrained = 0;
bool nvfp4SwinSets = false;
}
void ApplyNvfp4Swin(unsigned widths, unsigned long long layers, unsigned twoTerms, unsigned trained, bool sets) {
    std::lock_guard guard(nvfp4SwinMutex);
    if (const wchar_t* text = _wgetenv(L"NR_SWIN4_TRAINED"); text && *text) trained = *text != L'0'; // Development override.
    if (const wchar_t* text = _wgetenv(L"NR_SWIN4_SET"); text && *text) trained = unsigned(_wtoi(text)); // Development override: the weight set.
    if (nvfp4SwinSet && widths == nvfp4SwinWidths && layers == nvfp4SwinLayers && twoTerms == nvfp4SwinTwo && trained == nvfp4SwinTrained && sets == nvfp4SwinSets) return;
    nvfp4SwinSet = true; nvfp4SwinWidths = widths; nvfp4SwinLayers = layers; nvfp4SwinTwo = twoTerms; nvfp4SwinTrained = trained; nvfp4SwinSets = sets;
    // (A set from 2 on may leave blocks out or bring records for the runtime's own kernels: it is in use at any width selection.)
    if (!widths && trained < 2 && !sets) { NrSwin4::Disable(); Log("NVFP4 fused blocks: off"); return; }
    NrSwin4::Options options;
    const wchar_t* folder = _wgetenv(L"NR_SWIN4_ASSETS"); // Development override.
    options.assets = folder && *folder ? std::filesystem::path(folder) : ExecutableDirectory() / L"nvfp4" / L"swin4";
    options.widths = widths; options.layers = layers; options.twoTerms = twoTerms; options.set = trained;
    try { NrSwin4::Enable(options); Log(std::format("NVFP4 fused blocks: on, widths mask {}, blocks {:x}, {} weights, assets {}", widths, layers, trained == 0 ? std::string("first") : trained == 1 ? std::string("trained") : "set " + std::to_string(trained), Narrow(options.assets.wstring()))); }
    catch (const std::exception& e) { NrSwin4::Disable(); Log(std::string("NVFP4 fused blocks unavailable: ") + e.what()); }
}
std::string Nvfp4SwinStatus(bool counters) { return NrSwin4::Status(counters); }
namespace {
std::mutex nvfp4NetMutex;
bool nvfp4NetSet = false;
unsigned nvfp4NetMasks[4]{};
std::string nvfp4NetText;
}
void ApplyNvfp4Net(unsigned split, unsigned vit, unsigned prepost, const std::string& text, unsigned trained) {
    std::lock_guard guard(nvfp4NetMutex);
    if (const wchar_t* forced = _wgetenv(L"NR_NET4_TRAINED"); forced && *forced) trained = unsigned(_wtoi(forced)); // Development override.
    if (nvfp4NetSet && split == nvfp4NetMasks[0] && vit == nvfp4NetMasks[1] && prepost == nvfp4NetMasks[2] && trained == nvfp4NetMasks[3] && text == nvfp4NetText) return;
    nvfp4NetSet = true; nvfp4NetMasks[0] = split; nvfp4NetMasks[1] = vit; nvfp4NetMasks[2] = prepost; nvfp4NetMasks[3] = trained; nvfp4NetText = text;
    NrNet4::Options options;
    try { NrNet4::Parse(text, options); }
    catch (const std::exception& e) { options = {}; Log(std::string("NVFP4 network: the Nvfp4Net setting was ignored: ") + e.what()); }
    // The three masks decide which groups run; the text only adds what has no control (options words, single launches).
    options.family[NrNet4::Split].groups = split & 63; options.family[NrNet4::Vit].groups = vit & 7; options.family[NrNet4::PrePost].groups = prepost & 3;
    options.family[NrNet4::Split].trained = (trained & 1) != 0; options.family[NrNet4::Vit].trained = (trained & 2) != 0;
    if (!options.Any()) { NrNet4::Disable(); Log("NVFP4 network: off"); return; }
    const wchar_t* folder = _wgetenv(L"NR_NET4_ASSETS"); // Development override.
    options.assets = folder && *folder ? std::filesystem::path(folder) : ExecutableDirectory() / L"nvfp4";
    try { NrNet4::Enable(options); Log(std::format("NVFP4 network: on, split {} transformer {} input/output {}, trained weights {}, settings \"{}\", assets {}", split, vit, prepost, trained, text, Narrow(options.assets.wstring()))); }
    catch (const std::exception& e) { NrNet4::Disable(); Log(std::string("NVFP4 network unavailable: ") + e.what()); }
}
std::string Nvfp4NetStatus(bool counters) { return NrNet4::Status(counters); }
void SetNvfp4Net(Settings& settings, const std::string& text) {
    NrNet4::Options options;
    NrNet4::Parse(text, options);
    settings.nvfp4Split = options.family[NrNet4::Split].groups; settings.nvfp4Vit = options.family[NrNet4::Vit].groups;
    settings.nvfp4PrePost = options.family[NrNet4::PrePost].groups; settings.nvfp4Net = text;
}
bool BorrowCuts(const Settings& s, std::array<NrPassBorrow::Cut, Settings::MaxPasses>& cuts, unsigned& evaluations) {
    cuts.fill(NrPassBorrow::Cut::None); evaluations = 1; bool any = false;
    for (unsigned pass = 1; pass < std::min(s.passes, Settings::MaxPasses); ++pass) {
        const auto shortcut = s.Shortcut(pass);
        if (shortcut.reapply) continue; // Re-apply wins over borrowing and evaluates nothing.
        cuts[evaluations] = NrPassBorrow::ChooseCut(shortcut.borrowTransformer, shortcut.borrowBelow, shortcut.borrowLevel);
        any |= cuts[evaluations++] != NrPassBorrow::Cut::None;
    }
    return any;
}
void Processor::Evaluate(ID3D12Resource* colour, ID3D12Resource* depthGuide, ID3D12Resource* motionGuide,
    ID3D12Resource* result, UINT w, UINT h, bool reset, unsigned pass) {
    auto* c = gpu.commands.Get();
    struct Boundary {
        ID3D12GraphicsCommandList* list;
        ~Boundary() {
            NrPassRuntime::End(list);
            NrAmpere::EndEvaluation(list);
            NrAmpere::UseSet(0);
        }
    } boundary{c};
    const unsigned chosen = runtime->RequestedModel(pass);
    const unsigned modelFp8Set = Settings::Fp8Set(chosen);
    const auto model = NrPassRuntime::Select(chosen, modelFp8Set && NrHost::Requested(gpu.device.Get()));
    const unsigned selected = NrPassRuntime::BindFourBitWeights(model);
    if (model.backend == NrPassRuntime::Backend::Nvfp4) {
        std::string missing;
        bool complete = NrSwin4::SetComplete(selected, &missing);
        {
            std::lock_guard guard(nvfp4NetMutex);
            if (complete && (nvfp4NetMasks[0] || nvfp4NetMasks[1] || nvfp4NetMasks[2]))
                complete = NrNet4::SetComplete(selected, &missing);
        }
        if (complete && nvfp4Applied.load() > 0) complete = NrFfnV2::SetComplete(selected, &missing);
        if (!complete) throw std::runtime_error(std::format("NR pass {}: model {} unavailable: {}.", pass + 1, chosen, missing));
    }
    NrPassRuntime::Activate(model, c, chosen == 2);
    const auto evaluate = [&](auto&& run) {
        if (borrowEvaluation < 0) { run(); return; }
        const unsigned evaluation = unsigned(borrowEvaluation++);
        NrPassBorrow::BeginEvaluation(evaluation);
        try { run(); }
        catch (...) { NrPassBorrow::EndEvaluation(); throw; }
        const auto outcome = NrPassBorrow::EndEvaluation();
        if (evaluation < borrowOutcomes.size()) borrowOutcomes[evaluation] = outcome;
    };
    const bool isInt8 = Settings::IsInt8(chosen);
    if (NrPassRuntime::UsesHost(model)) {
        if (borrowEvaluation >= 0 && model.backend != NrPassRuntime::Backend::Nvfp4) throw std::runtime_error(std::format("NR pass {}: pass borrowing does not work with {}. Nothing was shown.", pass + 1, isInt8 ? "an INT8 model" : "the own pass host (NR_HOST=1)"));
        NrAmpere::UseSet(0);
        const size_t slot = std::min<size_t>(pass, hosts.size() - 1);
        auto& host = hosts[slot];
        const unsigned hostSet = model.weights;
        if (!host || host->Width() != w || host->Height() != h || host->Set() != hostSet || hostModels[slot] != chosen) {
            if (host) gpu.lifetime.Retire([previous = std::move(host)] {});
            try {
                const wchar_t* folder = _wgetenv(L"NR_INT8_ASSETS"); // Development override.
                const wchar_t* fourBitFolder = _wgetenv(L"NR_NVFP4_ASSETS");
                auto options = model.backend == NrPassRuntime::Backend::Nvfp4 ?
                    NrHost::FourBitOptions(fourBitFolder && *fourBitFolder ? std::filesystem::path(fourBitFolder) : ExecutableDirectory() / L"nvfp4", true) :
                    isInt8 ? NrHost::Int8Options(folder && *folder ? std::filesystem::path(folder) : ExecutableDirectory() / L"nvfp4" / L"int8")
                                                  : NrHost::FromEnvironment(ExecutableDirectory());
                host = NrPassRuntime::CreateTrackedHost(gpu.device.Get(), model, std::move(options), w, h);
                hostModels[slot] = chosen;
            }
            catch (const std::exception& e) { throw std::runtime_error(std::format("NR pass {}: {}", pass + 1, e.what())); }
        }
        const auto& tuning = runtime->PassSettings(pass);
        NrHost::Settings hs;
        hs.style = tuning.style; hs.intensity = tuning.intensity; hs.structure = tuning.localStructure; hs.tone = tuning.localTone;
        hs.skin = tuning.skinStructure; hs.autoMask = tuning.autoMask; hs.mvScaleX = hs.mvScaleY = runtime->motionScale;
        try { evaluate([&] { host->Evaluate(c, colour, motionGuide, result, hs, reset); }); }
        catch (const std::exception& e) { throw std::runtime_error(std::format("NR pass {}: {}", pass + 1, e.what())); }
        return;
    }
    const unsigned ampereSet = modelFp8Set && AmpereRequested() ? modelFp8Set : 0;
    if (ampereSet) {
        std::string missing;
        if (!NrAmpere::SetComplete(ampereSet, &missing))
            throw std::runtime_error(std::format("NR pass {}: the Ampere set cannot run fp8 set {} ({}).", pass + 1, ampereSet, missing));
        if (const std::string refused = NrAmpere::CheckArchitecture(gpu.device.Get()); !refused.empty())
            throw std::runtime_error(std::format("NR pass {}: {}. Nothing was shown.", pass + 1, refused));
    }
    NrAmpere::UseSet(ampereSet);
    const auto ampereFaults = NrAmpere::Faults(), ampereRan = NrAmpere::Completed() + NrAmpere::Partial();
    const unsigned fp8Set = ampereSet ? 0 : modelFp8Set;
    std::string missing;
    if (fp8Set && !NrFp8::SetComplete(fp8Set, &missing))
        throw std::runtime_error(std::format("NR pass {}: FP8 model {} unavailable: {}", pass + 1, chosen, missing));
    NrFp8::UseSet(fp8Set, c);
    const auto fp8Faults = NrFp8::Faults(), fp8Completed = NrFp8::Completed();
    const auto checkFp8 = [&] {
        if (ampereSet) {
            NrAmpere::EndEvaluation(c);
            if (NrAmpere::Faults() != ampereFaults)
                throw std::runtime_error(std::format("NR pass {}: the Ampere set could not run ({}). Nothing was shown.", pass + 1, NrAmpere::Problem()));
            if (NrAmpere::Completed() + NrAmpere::Partial() == ampereRan)
                throw std::runtime_error(std::format("NR pass {}: the Ampere set saw no evaluation of the model. Nothing was shown.", pass + 1));
        }
        if (!fp8Set) return;
        NrFp8::EndEvaluation(c);
        if (NrFp8::Faults() != fp8Faults)
            throw std::runtime_error(std::format("NR pass {}: the FP8 model could not run ({}). Nothing was shown.", pass + 1, NrFp8::Problem()));
        if (NrFp8::Completed() == fp8Completed)
            throw std::runtime_error(std::format("NR pass {}: the FP8 model recorded no evaluation on the selected command list.", pass + 1));
    };
    evaluate([&] { runtime->Evaluate(c, colour, depthGuide, motionGuide, result, w, h, reset, pass); });
    checkFp8();
}
ID3D12Resource* Processor::RunLegacy(ID3D12Resource* input, const Settings& s, bool hdr, bool resetHistory, float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps) {
    // Colour and signed edits are different DLSS inputs; never reuse one history for the other.
    resetHistory |= privateDlss && lastTransfer != s.transfer;
    lastTransfer = s.transfer;
    sceneMotion = nullptr; lastAnswer = nullptr;
    usedMotionVectors = nullptr;
    steadyApplied = false;
    if (s.editSmoothing <= 0) keptValid = false;
    if (!s.holdFrame) { resetHistory |= heldValid; heldValid = false; }
    else {
        if (!heldInput) heldInput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        if (!heldValid) {
            auto* c = gpu.commands.Get();
            Barrier(c, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Barrier(c, heldInput.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            c->CopyResource(heldInput.Get(), input);
            Barrier(c, input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            Barrier(c, heldInput.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            heldValid = true; resetHistory = true;
        }
        input = heldInput.Get();
    }
    const bool sdr = !modelStage && (s.inputColour == 1 || (s.inputColour == 0 && !hdr));
    if (sdr && !preparedProxy && !sdrInput) sdrInput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (sdr && !sdrOutput) sdrOutput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    const float whitePoint = s.referenceWhiteNits > 0 ? s.referenceWhiteNits / 80.0f :
        desktopWhitePoint > 0 ? desktopWhitePoint : hdr ? 203.0f / 80.0f : 1.0f;
    ID3D12Resource* vectors = motion.Get(); bool reset = true;
    bool ownFlow = false; // this call estimated the motion itself: opticalFlow's cut counter is this frame's
    gpu.Time(GpuStage::InputMotion);
    if (!composeOnly && s.estimateMotion && s.enabled && s.strength != 0 && s.compare != 1) {
        if (sharedMotionVectors) {
            // Stages with the same ordinary input geometry/filter estimate the
            // same original capture. Reuse that result, while matching the zero
            // vectors of a fresh local history when this pass needs a reset.
            reset = resetHistory || lastTone != s.tone || lastHdr != hdr || lastSdr != sdr || lastWhitePoint != whitePoint;
            vectors = reset ? motion.Get() : sharedMotionVectors;
        } else {
            auto* motionInput = chainMotionInput ? chainMotionInput : input;
            Barrier(gpu.commands.Get(), motionInput, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (flowInput) {
                DlssNrConstants down{}; down.Mode = DlssNrMode_Downsample; down.Width = modelWidth; down.Height = modelHeight;
                if (s.modelScale > 100) scaling->Run(2, motionInput, flowInput.Get(), s.downscaler, true);
                else Dispatch(28, down, motionInput, nullptr, nullptr, flowInput.Get(), nullptr);
                Barrier(gpu.commands.Get(), flowInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            // Estimate in model pixels so both passes receive correctly scaled vectors.
            reset = opticalFlow->Estimate(flowInput ? flowInput.Get() : motionInput,
                resetHistory || lastTone != s.tone || lastHdr != hdr || lastSdr != sdr || lastWhitePoint != whitePoint,
                s.motionZeroBias);
            if (flowInput) Barrier(gpu.commands.Get(), flowInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(gpu.commands.Get(), motionInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            vectors = opticalFlow->Vectors(); ownFlow = true;
            static const bool logCuts = _wgetenv(L"NR_EDIT_LOG") != nullptr;
            if (logCuts) fprintf(stderr, "cut count of the frame before: %u (a cut needs more than %u)\n", opticalFlow->lastCutCount, opticalFlow->lastCutLimit);
        }
        usedMotionVectors = vectors;
        sceneMotion = vectors;
    } else {
        if (opticalFlow) opticalFlow->Reset();
        if (keepHistoryWithoutMotion && !s.estimateMotion)
            reset = resetHistory || lastTone != s.tone || lastHdr != hdr || lastSdr != sdr || lastWhitePoint != whitePoint;
    }
    lastTone = s.tone; lastHdr = hdr; lastSdr = sdr; lastWhitePoint = whitePoint;
    // Settings::editInterval: the edit of a reduced-size pass with the "Difference" transfer is a smooth field that
    // hardly changes from one frame to the next (one frame late: 47-48 dB against 49 on time, WoW animation). On
    // the frames between two evaluations the model does not run: the kept field is added to the new frame. A cut
    // (reset) always evaluates. NR_EDIT_INTERVAL overrides the setting (checks).
    const EditPlan editPlan = PlanEdit(s);
    const unsigned editInterval = editPlan.interval;
    // (ResizeLegacy: a 100 % pass with an interval has the field at the full size, whatever transfer is chosen)
    const bool fullField = resizeInput && modelWidth == width && modelHeight == height;
    const unsigned transfer = fullField ? 4u : s.transfer;
    const bool fieldPass = resizeInput && transfer >= 3 && !externalAnswer && !spatialLayout.active && !privateDlss &&
        s.debugView == 0 && s.enabled && s.strength != 0 && s.compare != 1;
    // (without motion estimation every evaluation is a first frame for the model; a cut is then what the caller says)
    const bool cut = s.estimateMotion ? reset : resetHistory;
    // A new scene that the estimator's own limit (half of the coarse cells without a match) does not call a cut
    // still must not keep the old scene's edit: a sixteenth of that limit (measured: a cut between two game scenes
    // 930 of 8160 cells, a 48-pixel pan at most 38). The count is known one frame late here; the shader reads this
    // frame's count itself and drops the kept edit (kEditCutDivisor below).
    constexpr UINT kEditCutDivisor = 16;
    const bool newScene = ownFlow && opticalFlow->lastCutLimit && opticalFlow->lastCutCount > opticalFlow->lastCutLimit / kEditCutDivisor;
    // Settings::editLate: every frame takes the kept edit; the model evaluates the frame after its picture is made
    // (the end of this function). Not with borrowing between passes, nor in a chain of stages.
    std::array<NrPassBorrow::Cut, Settings::MaxPasses> lateCuts{}; unsigned lateEvaluations = 0;
    const bool lateEdit = editPlan.late && fieldPass && !modelStage && !composeOnly && !BorrowCuts(s, lateCuts, lateEvaluations);
    const bool reuseField = fieldPass && fieldValid && !cut && !newScene && !composeOnly &&
        (lateEdit || (editInterval > 1 && fieldAge + 1 < editInterval));
    const bool lateDue = lateEdit && reuseField && fieldAge + 1 >= editInterval; // (the model's input is then needed)
    if (!fieldPass) fieldValid = false;
    fieldReused = reuseField;
    historyReset = reset;
    gpu.Time(GpuStage::InputMotion, true);
    auto* c = gpu.commands.Get();
    if (outputReadable) Barrier(c, output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!s.enabled || s.compare == 1 || s.strength == 0) {
        Barrier(c, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        c->CopyResource(output.Get(), input);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    } else {
        gpu.Time(GpuStage::Encode);
        Barrier(c, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        DlssNrConstants p{}; p.Mode = DlssNrMode_Encode; p.Width = width; p.Height = height;
        p.WhitePoint = whitePoint;
        p.ReversibleMode = modelStage ? 2u : s.HdrMode(); // Raw stage replace avoids applying final controls between passes.
        if (sdr && !preparedProxy) {
            p.Mode = 11;
            Dispatch(49, p, input, nullptr, nullptr, sdrInput.Get(), nullptr);
            Barrier(c, sdrInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        p.Mode = clampStageInput ? 19u : DlssNrMode_Encode; p.Passthrough = sdr || modelStage ? 1u : 0u;
        if (!preparedProxy)
            Dispatch(0, p, sdr ? sdrInput.Get() : input, nullptr, nullptr, fullProxy ? fullProxy.Get() : proxy.Get(), original.Get());
        p.Width = modelWidth; p.Height = modelHeight;
        const DlssNrConstants modelConstants = p;
        if (fullProxy && !spatialLayout.active) {
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            p.Mode = DlssNrMode_Downsample;
            if (s.modelScale > 100) scaling->Run(0, fullProxy.Get(), proxy.Get(), s.downscaler, true);
            else if (!reuseField || lateDue) Dispatch(21, p, fullProxy.Get(), nullptr, nullptr, proxy.Get(), nullptr);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (!spatialLayout.active && !preparedProxy)
            Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(c, original.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Time(GpuStage::Encode, true);
        gpu.Time(GpuStage::NR);
        std::array<NrPassBorrow::Cut, Settings::MaxPasses> cuts{}; unsigned evaluations = 0;
        const bool borrowing = !externalAnswer && !modelStage && !reuseField && BorrowCuts(s, cuts, evaluations);
        if (borrowing) {
            bool native = false;
            for (unsigned pass = 0; pass < std::min(s.passes, Settings::MaxPasses); ++pass)
                native |= NrPassRuntime::Select(runtime->RequestedModel(pass)).backend == NrPassRuntime::Backend::Nvfp4;
            NrPassBorrow::BeginFrame(c, cuts.data(), evaluations, native);
            borrowEvaluation = 0;
            borrowEvaluations = evaluations;
        }
        const BorrowFrame borrowFrame{borrowing, borrowEvaluation};
        // Pass 1 evaluates first into result; later passes evaluate the clamped previous result from next.
        // A run of r re-apply passes evaluates nothing: one dispatch into extended extends the last evaluated
        // pass's change, saturate(out + r * gain * (out - in)). Returns the final answer and whether extended was written.
        const auto runPasses = [&](ID3D12Resource* first, ID3D12Resource* next, ID3D12Resource* result,
            ComPtr<ID3D12Resource>& extended, ID3D12Resource* depthGuide, ID3D12Resource* motionGuide, UINT pw, UINT ph) {
            Evaluate(first, depthGuide, motionGuide, result, pw, ph, reset, 0);
            Barrier(c, result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ID3D12Resource* evaluated = first; ID3D12Resource* current = result;
            unsigned run = 0; bool written = false;
            const auto extend = [&](unsigned pass) {
                if (!extended) extended = gpu.Texture(pw, ph, DXGI_FORMAT_R16G16B16A16_FLOAT);
                else if (written) Barrier(c, extended.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                if (evaluated == next) Barrier(c, next, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                auto k = p; k.Mode = 22; k.Width = pw; k.Height = ph; k.TransferStrength = float(run) * s.reapplyGain;
                Dispatch(ReapplySlot(pass), k, result, evaluated, nullptr, extended.Get(), nullptr);
                Barrier(c, extended.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (evaluated == next) Barrier(c, next, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                current = extended.Get(); written = true; run = 0;
            };
            for (unsigned pass = 1; pass < s.passes; ++pass) {
                if (s.Shortcut(pass).reapply) { ++run; continue; }
                if (run) extend(pass - 1);
                auto clamp = p; clamp.Mode = DlssNrMode_ClampProxy; clamp.Width = pw; clamp.Height = ph;
                Dispatch(current == result ? 14 : ReapplyClampSlot, clamp, current, nullptr, nullptr, next, nullptr);
                Barrier(c, next, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(c, result, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                Evaluate(next, depthGuide, motionGuide, result, pw, ph, reset, pass);
                Barrier(c, result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(c, next, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                evaluated = next; current = result;
            }
            if (run) extend(s.passes - 1);
            return std::pair{current, written};
        };
        ID3D12Resource* evaluatedAnswer = answer.Get();
        bool reappliedWritten = false, steadyUsed = false;
        // Settings::editSmoothing ("Steady edit"): the edit of all passes together (answer - input) blended with the
        // previous frame's, moved along this frame's estimated vectors. Needs real vectors at the model's size; not in
        // a chain of stages. With Picture first it runs on the late evaluation (each frame's, so only with an edit
        // interval of 1: the vectors span one frame).
        // kSteadyTolerance: the difference between the two edits (per colour channel, in the model's encoded
        // values, 0..1) at which the kept edit is no longer used: a real change (motion the vectors missed,
        // light, an animation) takes the new edit at once; flicker smaller than that is smoothed.
        // kSteadyInputTolerance: the change of the input's brightness (moved along the motion) at which it is no
        // longer used: there the input itself changed (film grain, noise) and so may the edit. Without it, film
        // grain flickered more than with no steady edit (the kept edit carried the last frame's grain); 0.001..0.03
        // measured with --stability-check (WoW sequence, pan, grain 1.5 % and 4 %): 0.002 was best overall.
        // NR_STEADY_TOLERANCE and NR_STEADY_INPUT_TOLERANCE override them (checks).
        static const float kSteadyTolerance = [] { const wchar_t* v = _wgetenv(L"NR_STEADY_TOLERANCE"); return v && *v ? std::clamp(float(_wtof(v)), 1e-3f, 1.f) : .04f; }();
        static const float kSteadyInputTolerance = [] { const wchar_t* v = _wgetenv(L"NR_STEADY_INPUT_TOLERANCE"); return v && *v ? std::clamp(float(_wtof(v)), 1e-4f, 10.f) : .002f; }();
        const bool steadyPossible = s.editSmoothing > 0 && usedMotionVectors && usedMotionVectors != motion.Get() && !modelStage &&
            !preparedProxy && (!lateEdit || editInterval == 1);
        // The evaluated answer (non-pixel-shader readable) in, the steady answer (readable) out. Uses c and p as they are.
        const auto applySteady = [&](ID3D12Resource* evaluated) -> ID3D12Resource* {
            if (!steadyAnswer) {
                steadyAnswer = gpu.Texture(modelWidth, modelHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
                for (auto& kept : keptEdit) kept = gpu.Texture(modelWidth, modelHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
                keptValid = false;
            }
            auto steady = p; steady.Mode = 23; steady.Width = modelWidth; steady.Height = modelHeight;
            // A history reset (cut, start, setting change) takes the new edit only, and keeps it for the next frame.
            steady.TransferStrength = reset || !keptValid ? 0.f : std::clamp(s.editSmoothing, 0.f, .9f);
            steady.ColourStrength = kSteadyTolerance; steady.MaxRatio = kSteadyInputTolerance;
            ID3D12Resource* previous = keptEdit[keptIndex].Get(); ID3D12Resource* next = keptEdit[1 - keptIndex].Get();
            Barrier(c, previous, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Dispatch(SteadySlot, steady, proxy.Get(), evaluated, previous, steadyAnswer.Get(), next, nullptr, vectors);
            Barrier(c, previous, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, steadyAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            keptIndex = 1 - keptIndex; keptValid = true;
            return steadyAnswer.Get();
        };
        if (!externalAnswer && spatialLayout.active) {
            DlssNr::GuideRegions regions{};
            regions.depth = {0, 0, modelWidth, modelHeight};
            regions.motion = {0, 0, modelWidth, modelHeight};
            const auto make = [&](unsigned mode) {
                return DlssNr::Spatial::MakeConstants(spatialLayout, mode, regions,
                    float(width) / modelWidth, float(height) / modelHeight, width, height);
            };
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            spatialShader->Dispatch(0, make(100), fullProxy.Get(), nullptr, nullptr, packedProxy.Get());
            spatialShader->Dispatch(1, make(101), fullProxy.Get(), depth.Get(), vectors, packedDepth.Get(), packedMotion.Get());
            for (auto* r : {packedProxy.Get(), packedDepth.Get(), packedMotion.Get()})
                Barrier(c, r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            const UINT sw = spatialLayout.modelW, sh = spatialLayout.modelH;
            const auto [packedFinal, packedWritten] = runPasses(packedProxy.Get(), packedPassInput.Get(), packedAnswer.Get(),
                packedReapplied, packedDepth.Get(), packedMotion.Get(), sw, sh);
            spatialShader->Dispatch(2, make(102), packedProxy.Get(), packedFinal, nullptr, proxy.Get(), answer.Get());
            for (auto* r : {packedProxy.Get(), packedDepth.Get(), packedMotion.Get(), packedAnswer.Get(), fullProxy.Get()})
                Barrier(c, r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (packedWritten) Barrier(c, packedReapplied.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        } else if (reuseField) {
            // No model on this frame. (The answer's state as an evaluation leaves it, for the barriers below.)
            Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ++fieldAge;
            // (the steady edit's kept edit would be older than the vectors say; Picture first on every frame keeps it:
            // its evaluation at the end of this function is this frame's and takes it further)
            if (!lateDue || editInterval != 1) keptValid = false;
        } else if (!externalAnswer) {
            // The vectors are this frame's against the one before; the history is fieldAge + 1 frames old.
            runtime->motionScale = fieldPass && fieldValid && !cut ? float(fieldAge + 1) : 1.0f;
            std::tie(evaluatedAnswer, reappliedWritten) = runPasses(proxy.Get(), passInput.Get(), answer.Get(),
                reapplied, depth.Get(), vectors, modelWidth, modelHeight);
            runtime->motionScale = 1.0f;
            fieldAge = 0; fieldValid = fieldPass;
            if (steadyPossible) { evaluatedAnswer = applySteady(evaluatedAnswer); steadyUsed = true; }
            else keptValid = false;
        }
        gpu.Time(GpuStage::NR, true);
        ID3D12Resource* modelAnswer = externalAnswer ? externalAnswer : evaluatedAnswer;
        ID3D12Resource* resolveProxy = preparedProxy ? preparedProxy : proxy.Get();
        const bool halo = (s.haloSuppression > 0 || s.haloBrightSuppression > 0) && !reuseField; // (the kept field has it)
        if (halo) {
            if (!filteredAnswer) filteredAnswer = gpu.Texture(modelWidth, modelHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
            auto filter = p; filter.Mode = 18;
            filter.HaloSuppression = std::clamp(s.haloSuppression, 0.f, 1.f);
            filter.HaloBrightSuppression = std::clamp(s.haloBrightSuppression, 0.f, 1.f);
            filter.HaloRadius = std::clamp(s.haloRadius, 1.f, 4.f);
            Dispatch(91, filter, resolveProxy, modelAnswer, nullptr, filteredAnswer.Get(), nullptr);
            Barrier(c, filteredAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            modelAnswer = filteredAnswer.Get();
        }
        const bool edgeAware = transfer == 2 && resizeInput;
        if (!externalAnswer && nativeAnswer && !resizeInput) {
            scaling->Run(1, modelAnswer, nativeAnswer.Get(), s.downscaler, false);
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            modelAnswer = nativeAnswer.Get(); resolveProxy = fullProxy.Get();
        }
        if (!externalAnswer && resizeInput && !edgeAware) {
            // Measure lighting against the input at the same resolution, before
            // enlargement. Reconstruct a native-size model image for the resolve.
            // (the two passes of a model pass below 100 %, timed as "Edit transfer": the NR stage does not hold them)
            gpu.Time(GpuStage::Transfer);
            p.Mode = 16; p.Transfer = transfer;
            if (!reuseField) Dispatch(35, p, proxy.Get(), modelAnswer, nullptr, resizeInput.Get(), nullptr);
            Barrier(c, resizeInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Time(GpuStage::Transfer, true);
            ID3D12Resource* field = resizeInput.Get();
            if (privateDlss) {
                gpu.Time(GpuStage::PrivateDlss);
                runtime->EvaluateDlss(c, field, depth.Get(), vectors, exposure.Get(), enlarged.Get(), reset, frameTimeMs);
                Barrier(c, enlarged.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                gpu.Time(GpuStage::PrivateDlss, true);
                gpu.Time(GpuStage::EdgeCorrection);
                auto bound = p; bound.Mode = 14; bound.Width = width; bound.Height = height;
                Dispatch(77, bound, field, enlarged.Get(), nullptr, boundedDlss.Get(), nullptr);
                Barrier(c, boundedDlss.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                field = boundedDlss.Get();
                gpu.Time(GpuStage::EdgeCorrection, true);
            }
            gpu.Time(GpuStage::Transfer);
            // (a 100 % pass has no separate full-size input: the model's own is it, and is readable already)
            ID3D12Resource* base = fullProxy ? fullProxy.Get() : proxy.Get();
            if (fullProxy) Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            auto restore = p; restore.Mode = 17; restore.Width = width; restore.Height = height;
            // A kept edit follows the picture: this frame's motion vectors, times the frames since the edit was made
            // (ModelResize.hlsli reads them from the third input when DebugView is not 0). NR_EDIT_FOLLOW=0: off.
            static const bool follow = [] { const wchar_t* v = _wgetenv(L"NR_EDIT_FOLLOW"); return !v || _wtoi(v) != 0; }();
            const bool followMotion = follow && reuseField && s.estimateMotion && vectors && vectors != motion.Get();
            restore.DebugView = followMotion ? fieldAge : 0;
            // ... and is dropped for this frame where the estimator finds a new scene (its counter, read by the shader).
            ID3D12Resource* cutCount = followMotion && ownFlow && opticalFlow->lastCutLimit ? opticalFlow->Cuts() : nullptr;
            restore.CompareMode = cutCount ? opticalFlow->lastCutLimit / kEditCutDivisor : 0;
            if (cutCount) Barrier(c, cutCount, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Dispatch(84, restore, field, base, followMotion ? vectors : resizeInput.Get(), nativeAnswer.Get(), nullptr, cutCount);
            if (cutCount) Barrier(c, cutCount, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Time(GpuStage::Transfer, true);
            Barrier(c, resizeInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            modelAnswer = nativeAnswer.Get(); resolveProxy = base;
        }
        gpu.Time(GpuStage::Resolve);
        p.Mode = DlssNrMode_Resolve; p.ApplyModel = s.applyModel; p.TransferStrength = s.strength; p.ColourStrength = s.colourStrength;
        // Development override (the lab's checkers have no control for it): the Colour strength slider's value.
        static const float forcedColour = [] { const wchar_t* v = _wgetenv(L"NR_COLOUR_STRENGTH"); return v && *v ? std::clamp(float(_wtof(v)), 0.f, 4.f) : -1.f; }();
        if (forcedColour >= 0) p.ColourStrength = forcedColour;
        p.DebugView = s.debugView; p.DebugScale = sdr ? 1.0f : whitePoint;
        p.SkinProtection = s.separateSkin; p.ShowSkinMask = s.previewSkin;
        p.SkinDetail = s.skinDetail; p.SkinColour = s.skinColour;
        p.EnvironmentDetail = s.environmentDetail; p.EnvironmentColour = s.environmentColour;
        p.Width = width; p.Height = height;
        p.MaxRatio = s.highlightGuard; p.Transfer = edgeAware ? 5 : nativeAnswer && !fullField ? 0 : std::min(s.transfer, 1u);
        p.CompareMode = s.compare == 3 ? 1 : s.compare == 2 ? 2 : 0;
        p.CompareSplit = s.compareSplit; p.CompareZoom = s.compareZoom; p.CompareSwap = s.compareSwap;
        if (s.debugView != 0) { modelAnswer = externalAnswer ? externalAnswer : evaluatedAnswer; resolveProxy = preparedProxy ? preparedProxy : proxy.Get(); p.Transfer = 1; }
        if (modelStage && !edgeAware) p.Mode = 20;
        lastAnswer = externalAnswer ? nullptr : evaluatedAnswer;
        // The difference of two models: the view shows (answer - model input) x 20, so the other model's answer
        // takes the input's place.
        ID3D12Resource* against = nullptr;
        if (s.debugView == 3 && differenceAgainst && differenceAgainst != modelAnswer) {
            const auto mine = modelAnswer->GetDesc(), theirs = differenceAgainst->GetDesc();
            if (mine.Width == theirs.Width && mine.Height == theirs.Height) against = differenceAgainst;
        }
        if (against) {
            Barrier(c, against, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resolveProxy = against; p.CompareMode = 0;
        }
        Dispatch(7, p, resolveProxy, modelAnswer, original.Get(), sdr ? sdrOutput.Get() : output.Get(), nullptr);
        if (against) Barrier(c, against, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (sdr) {
            Barrier(c, sdrOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            p.Mode = 12;
            Dispatch(56, p, sdrOutput.Get(), input, preparedProxy ? preparedProxy : sdrInput.Get(), output.Get(), nullptr);
            Barrier(c, sdrOutput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!preparedProxy) Barrier(c, sdrInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (privateDlss) {
            Barrier(c, enlarged.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, boundedDlss.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (nativeAnswer && !edgeAware) {
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (fullProxy) Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (!preparedProxy) Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, original.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!externalAnswer) Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (reappliedWritten) Barrier(c, reapplied.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (steadyUsed) Barrier(c, steadyAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        steadyApplied = steadyUsed;
        if (halo) Barrier(c, filteredAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        gpu.Time(GpuStage::Resolve, true);
        if (lateDue) {
            // The picture above is complete. Now the model's evaluation of this frame: its edit serves the frames
            // after. With a host that presents (Gpu::lateSplit) this goes into a list of its own, sent to the GPU
            // after the frame is presented; otherwise it follows in the frame's list.
            const bool split = gpu.BeginLate();
            c = gpu.commands.Get();
            p = modelConstants;
            gpu.Time(GpuStage::NR);
            Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            runtime->motionScale = float(fieldAge); // the frames since the model's last evaluation
            const auto [lateAnswer, lateWritten] = runPasses(proxy.Get(), passInput.Get(), answer.Get(),
                reapplied, depth.Get(), vectors, modelWidth, modelHeight);
            runtime->motionScale = 1.0f;
            // Steady edit on this frame's late evaluation (steadyPossible: an edit interval of 1 here).
            const bool lateSteady = steadyPossible;
            ID3D12Resource* lateModel = lateSteady ? applySteady(lateAnswer) : lateAnswer;
            if (!lateSteady) keptValid = false;
            gpu.Time(GpuStage::NR, true);
            const bool lateHalo = s.haloSuppression > 0 || s.haloBrightSuppression > 0;
            if (lateHalo) {
                if (!filteredAnswer) filteredAnswer = gpu.Texture(modelWidth, modelHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
                auto filter = p; filter.Mode = 18;
                filter.HaloSuppression = std::clamp(s.haloSuppression, 0.f, 1.f);
                filter.HaloBrightSuppression = std::clamp(s.haloBrightSuppression, 0.f, 1.f);
                filter.HaloRadius = std::clamp(s.haloRadius, 1.f, 4.f);
                Dispatch(91, filter, proxy.Get(), lateModel, nullptr, filteredAnswer.Get(), nullptr);
                Barrier(c, filteredAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                lateModel = filteredAnswer.Get();
            }
            gpu.Time(GpuStage::Transfer);
            auto make = p; make.Mode = 16; make.Transfer = transfer;
            Dispatch(35, make, proxy.Get(), lateModel, nullptr, resizeInput.Get(), nullptr);
            gpu.Time(GpuStage::Transfer, true);
            Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (lateWritten) Barrier(c, reapplied.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (lateHalo) Barrier(c, filteredAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (lateSteady) { Barrier(c, steadyAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); steadyApplied = true; }
            fieldAge = 0; // (counted from the frame the edit was made for)
            if (split) { gpu.EndLate(); c = gpu.commands.Get(); }
        }
    }
    Barrier(c, input, (!s.enabled || s.compare == 1 || s.strength == 0) ? D3D12_RESOURCE_STATE_COPY_SOURCE :
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    outputReadable = true;
    const bool compareLabels = s.compare >= 2 && s.compareLabels && s.debugView == 0 && s.enabled && s.strength != 0;
    const bool tuneHdr = hdr && s.enabled && s.strength != 0 && s.applyModel && !s.previewSkin && s.debugView == 0 &&
        s.compare != 1 && (s.peakBrightnessNits > 0 || s.blackLevelNits != 0);
    if (compareLabels || processedFps >= 0 || tuneHdr) {
        gpu.Time(GpuStage::HdrLabels);
        if (!labelled) labelled = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (labelsReadable) Barrier(c, labelled.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        DlssNrConstants label{}; label.Mode = 13; label.Width = width; label.Height = height;
        label.WhitePoint = whitePoint; label.DebugScale = s.labelScale;
        // Mode 13: one FPS value, or separate real-input / FG-output values.
        label.DebugView = processedFps < 0 ? 0 : outputFps >= 0 ? 2 : 1;
        label.CompareZoom = static_cast<float>(processedFps); label.TransferStrength = static_cast<float>(outputFps);
        label.CompareMode = s.compare == 3 ? 1 : s.compare == 2 ? 2 : 0;
        label.ApplyModel = compareLabels; label.CompareSplit = s.compareSplit; label.CompareSwap = s.compareSwap;
        label.Passthrough = !tuneHdr; label.MaxRatio = s.peakBrightnessNits / 80; label.ColourStrength = s.blackLevelNits / 80;
        label.MvScaleX = s.compareZoom;
        Dispatch(70, label, output.Get(), nullptr, nullptr, labelled.Get(), nullptr);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(c, labelled.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        gpu.Time(GpuStage::HdrLabels, true);
        labelsReadable = true; return labelled.Get();
    }
    return output.Get();
}
}
