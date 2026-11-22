#include "Runtime.h"
#include "AdaMfgUnlock.h"
#include "nvfp4/NrHost.h"
#include <wintrust.h>
#include <softpub.h>
#include <wincrypt.h>
namespace nr {
void Runtime::VerifyNvidia(const std::filesystem::path& path) {
    WINTRUST_FILE_INFO file{sizeof(file)}; file.pcwszFilePath = path.c_str();
    WINTRUST_DATA trust{sizeof(trust)}; trust.dwUIChoice = WTD_UI_NONE;
    trust.dwUnionChoice = WTD_CHOICE_FILE; trust.pFile = &file;
    trust.dwStateAction = WTD_STATEACTION_VERIFY; trust.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID policy = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    LONG result = WinVerifyTrust(nullptr, &policy, &trust);
    auto* provider = WTHelperProvDataFromStateData(trust.hWVTStateData);
    auto* signer = provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
    wchar_t subject[512]{};
    if (signer && signer->csCertChain)
        CertGetNameStringW(signer->pasCertChain[0].pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0,
            nullptr, subject, (DWORD)std::size(subject));
    trust.dwStateAction = WTD_STATEACTION_CLOSE; WinVerifyTrust(nullptr, &policy, &trust);
    if (result != ERROR_SUCCESS || std::wstring(subject).find(L"NVIDIA") == std::wstring::npos)
        throw std::runtime_error("A valid NVIDIA-signed DLL is required: " + Narrow(path.wstring()) +
            std::format(" (signature status 0x{:08X})", (unsigned)result));
}
namespace {
void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) noexcept {
    try { if (message) Log(std::string("NGX: ") + message); }
    catch (...) { /* A logging failure must not escape into the vendor runtime. */ }
}
template<class T> T Export(HMODULE module, const char* name) {
    auto address = GetProcAddress(module, name);
    if (!address) throw std::runtime_error(std::string("NGX driver is missing ") + name);
    return reinterpret_cast<T>(address);
}
}
void Runtime::CheckResult(NVSDK_NGX_Result result, const char* operation) {
    if (NVSDK_NGX_FAILED(result)) {
        auto message = std::format("{} failed: NGX 0x{:08X}. See capture.log for the selected runtime and loader diagnostics.", operation, (unsigned)result);
        Log(message); throw std::runtime_error(message);
    }
}
Runtime::Runtime(ID3D12Device* d, const std::filesystem::path& modelPath, bool unlockAdaMfg) : device(d) {
    if (modelPath.filename() != L"nvngx_dlssnr.dll")
        throw std::runtime_error("Select nvngx_dlssnr.dll.");
    const auto selected = modelPath.is_absolute() ? modelPath : ExecutableDirectory() / modelPath;
    std::error_code pathError;
    if (!std::filesystem::is_regular_file(selected, pathError))
        throw std::runtime_error("NR DLL not found: " + Narrow(selected.wstring()) +
            ". Put nvngx_dlssnr.dll in the runtime folder beside display_filter.exe, or use Browse.");
    const auto model = std::filesystem::canonical(selected);
    this->modelPath = model;
    wchar_t folder[32768]{}; DWORD bytes = sizeof(folder);
    LSTATUS read = RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore",
        L"FullPath", RRF_RT_REG_SZ, nullptr, folder, &bytes);
    Check(HRESULT_FROM_WIN32(read), "NGX driver location");
    const auto driver = std::filesystem::canonical(std::filesystem::path(folder) / L"_nvngx.dll");
    VerifyNvidia(driver);
    module = LoadLibraryExW(driver.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) Check(HRESULT_FROM_WIN32(GetLastError()), "Load NGX driver");
    // NGX can retain asynchronous driver callbacks after Shutdown. Keep its code
    // resident until process exit, including when feature initialization fails.
    HMODULE pinned = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(module), &pinned)) {
        const auto error = HRESULT_FROM_WIN32(GetLastError()); FreeLibrary(module); module = nullptr;
        Check(error, "Retain NGX dispatcher");
    }
    bool initialized = false;
    try {
        using Init = NVSDK_NGX_Result (*)(const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*,
            ID3D12Device*, NVSDK_NGX_Version, const NVSDK_NGX_FeatureCommonInfo*);
        auto init = Export<Init>(module, "NVSDK_NGX_D3D12_Init_ProjectID");
        shutdown = Export<decltype(shutdown)>(module, "NVSDK_NGX_D3D12_Shutdown");
        create = Export<decltype(create)>(module, "NVSDK_NGX_D3D12_CreateFeature");
        evaluate = Export<decltype(evaluate)>(module, "NVSDK_NGX_D3D12_EvaluateFeature");
        release = Export<decltype(release)>(module, "NVSDK_NGX_D3D12_ReleaseFeature");
        destroy = Export<decltype(destroy)>(module, "NVSDK_NGX_D3D12_DestroyParameters");
        allocate = Export<decltype(allocate)>(module, "NVSDK_NGX_D3D12_AllocateParameters");
        capabilities = Export<decltype(capabilities)>(module,
            "NVSDK_NGX_D3D12_GetCapabilityParameters");
        const auto bundled = (ExecutableDirectory() / L"runtime").wstring();
        dlssPath = model.parent_path() / L"nvngx_dlss.dll";
        if (!std::filesystem::exists(dlssPath)) dlssPath = std::filesystem::path(bundled) / L"nvngx_dlss.dll";
        const auto directory = model.parent_path().wstring();
        const auto streamline = (std::filesystem::path(bundled) / L"streamline").wstring();
        const wchar_t* paths[]{directory.c_str(), bundled.c_str(), streamline.c_str()};
        NVSDK_NGX_FeatureCommonInfo common{}; common.PathListInfo.Path = paths; common.PathListInfo.Length = UINT(std::size(paths));
        // Initialize NGX before Streamline with a process-lifetime callback. Some
        // snippets retain the first callback across Shutdown/Init; sl.common unloads.
        common.LoggingInfo.LoggingCallback = NgxLog;
        common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
        common.LoggingInfo.DisableOtherLoggingSinks = true;
        CheckResult(init("969c9e2d-5cbf-4d4a-9693-c2ac4f8f52de", NVSDK_NGX_ENGINE_TYPE_CUSTOM,
            "OptiScalerNR-1", DataDirectory().c_str(), d, NVSDK_NGX_Version_API, &common), "NGX initialization");
        initialized = true;
        // NGX has selected its provider, but no Streamline FG feature exists yet.
        AdaMfgUnlock::ResetSession();
        if (unlockAdaMfg) AdaMfgUnlock::PrepareProvider(d);
        Log("NR runtime requested: " + Narrow(model.wstring()));
        Log("Unmodified NGX dispatcher: " + Narrow(driver.wstring()));
    } catch (...) {
        if (initialized) shutdown();
        FreeLibrary(module); module = nullptr; throw;
    }
}
Runtime::Runtime(std::shared_ptr<Runtime> parent) : owner(std::move(parent)) {
    if (!owner) throw std::runtime_error("Missing NR runtime owner.");
    device = owner->device; dlssPath = owner->dlssPath; modelPath = owner->modelPath;
    compatibility = owner->compatibility;
    create = owner->create; evaluate = owner->evaluate; release = owner->release;
    destroy = owner->destroy; allocate = owner->allocate; capabilities = owner->capabilities;
}
Runtime::~Runtime() {
    ReleaseDlss();
    Release();
    compatibility.reset();
    if (shutdown) shutdown(); if (module) FreeLibrary(module);
}
void Runtime::Release() {
    for (size_t pass = 0; pass < features.size(); ++pass) {
        if (features[pass]) {
            if (compatibilityFeatures[pass]) compatibility->Release(features[pass]);
            else release(features[pass]);
            features[pass] = nullptr;
        }
        compatibilityFeatures[pass] = false;
    }
    for (auto& params : parameters) if (params) { destroy(params); params = nullptr; }
}
NVSDK_NGX_Result Runtime::Create(ID3D12GraphicsCommandList* commands, UINT w, UINT h, const DlssNr::ModelSettings& model, unsigned passes,
    const std::array<std::optional<DlssNr::ModelSettings>, Settings::MaxPasses - 1>& overrides, unsigned skipMask) {
    if (passes < 1 || passes > features.size()) throw std::runtime_error("Choose 1-10 NR passes.");
    Release();
    if (owner && owner->compatibility) compatibility = owner->compatibility;
    for (unsigned pass = 0; pass < passes; ++pass) {
        if (skipMask >> pass & 1) continue; // Re-apply passes evaluate nothing; Evaluate refuses them.
        auto& tuning = settings[pass]; tuning = model;
        if (pass) { tuning.localTone = 0; if (overrides[pass - 1]) tuning = *overrides[pass - 1]; }
        requested[pass] = tuning.preset;
        if (!Settings::IsModel(tuning.preset)) // A removed model saved in the settings: never run another model in its place.
            throw std::runtime_error("NR pass " + std::to_string(pass + 1) + " asks for model " + std::to_string(tuning.preset) +
                ", which was removed from this build. Choose another model in the Model list.");
        // NR_HOST=1: an FP8 model runs on the own pass host (Processor::Evaluate); NVIDIA's runtime makes no feature for it
        // (it refuses GPUs below the RTX 50 series).
        // An INT8 model always runs there: its kernels are ours on every GPU.
        const bool ownHost = Settings::IsInt8(tuning.preset) || (Settings::IsFp8(tuning.preset) && NrHost::Requested(device.Get()));
        if (Settings::IsNvfp4(tuning.preset) || Settings::IsFp8(tuning.preset) || Settings::IsInt8(tuning.preset)) tuning.preset = 1; // Model A; Processor::Evaluate lets the 4-bit kernels or the FP8 records in.
        tuning.preset = std::min(tuning.preset, 3u); tuning.style = std::min(tuning.style, 2u);
        tuning.intensity = std::clamp(tuning.intensity, 0.0f, 2.0f);
        tuning.localStructure = std::clamp(tuning.localStructure, 0.0f, 2.0f);
        tuning.localTone = std::clamp(tuning.localTone, 0.0f, 2.0f);
        tuning.skinStructure = std::clamp(tuning.skinStructure, -1.0f, 2.0f);
        if (ownHost) {
            Log(std::format("NR pass {} {}x{}: model {} on the own pass host (no NVIDIA feature); intensity {}, structure {}, tone {}, skin {}, style {}",
                pass + 1, w, h, requested[pass], tuning.intensity, tuning.localStructure, tuning.localTone, tuning.skinStructure, tuning.style));
            continue;
        }
        CheckResult(capabilities(&parameters[pass]), "NR pass parameters");
        auto* p = parameters[pass]; if (!p) throw std::runtime_error("NGX returned no parameter map.");
        // Match OptiScaler's default pass profiles: only the first pass changes broad lighting.
        p->Set("DLSSNR.Enabled", 1u); p->Set("DLSSNR.Width", w); p->Set("DLSSNR.Height", h);
        p->Set("CreationNodeMask", 1u); p->Set("VisibilityNodeMask", 1u);
        p->Set("DLSSNR.Hint.Render.Preset", tuning.preset);
        DlssNr::SetModelTuning(p, tuning); p->Set("DLSSNR.UICorrection", 1u);
        for (auto name : {"DLSSNR.UI", "DLSSNR.UIAlpha", "DLSSNR.ControlMask", "DLSSNR.Backbuffer"})
            p->Set(name, static_cast<ID3D12Resource*>(nullptr));
        for (const char* layer : {"UI", "UIAlpha", "Backbuffer"})
            for (const char* field : {"SubrectBaseX", "SubrectBaseY", "SubrectWidth", "SubrectHeight"})
                p->Set((std::string("DLSSNR.") + layer + field).c_str(), 0u);
        auto result = compatibility ? compatibility->Create(commands, p, &features[pass]) :
            create(commands, static_cast<NVSDK_NGX_Feature>(18), p, &features[pass]);
        if (!compatibility && NVSDK_NGX_FAILED(result) && !features[pass]) {
            Log(std::format("NR driver creation failed 0x{:08X}; trying the selected NR compatibility runtime: {}",
                (unsigned)result, Narrow(modelPath.wstring())));
            auto& sharedProvider = owner ? owner->compatibility : compatibility;
            if (!sharedProvider)
                sharedProvider = DlssNr::CompatibilityRuntime::Open(modelPath, device.Get(), allocate, destroy, DataDirectory());
            compatibility = sharedProvider;
            if (compatibility) result = compatibility->Create(commands, p, &features[pass]);
        }
        compatibilityFeatures[pass] = compatibility != nullptr;
        if (NVSDK_NGX_SUCCEED(result) && !features[pass]) result = NVSDK_NGX_Result_Fail;
        Log(std::format("CreateFeature(18) pass {} {}x{}: 0x{:08X}; intensity {}, structure {}, tone {}, skin {}, style {}, preset {}, auto mask {}",
            pass + 1, w, h, (unsigned)result, tuning.intensity, tuning.localStructure, tuning.localTone, tuning.skinStructure,
            tuning.style, tuning.preset, tuning.autoMask));
        if (NVSDK_NGX_FAILED(result)) return result;
        if (!loggedRuntime) {
            wchar_t loaded[32768]{}; const auto library = GetModuleHandleW(L"nvngx_dlssnr.dll");
            if (library && GetModuleFileNameW(library, loaded, (DWORD)std::size(loaded)))
                Log("NR runtime actually loaded: " + Narrow(loaded));
            else Log("NR loaded-runtime path unavailable from the process module list.");
            loggedRuntime = true;
        }
    }
    return NVSDK_NGX_Result_Success;
}
void Runtime::Evaluate(ID3D12GraphicsCommandList* commands, ID3D12Resource* color, ID3D12Resource* depth,
    ID3D12Resource* motion, ID3D12Resource* output, UINT w, UINT h, bool reset, unsigned pass) {
    if (pass >= features.size() || !features[pass]) throw std::runtime_error("NR has no valid feature.");
    auto* parameters = this->parameters[pass];
    parameters->Set("DLSSNR.Enabled", 1u); parameters->Set("DLSSNR.Width", w); parameters->Set("DLSSNR.Height", h);
    parameters->Set("DLSSNR.Color", color); parameters->Set("DLSSNR.Depth", depth);
    parameters->Set("DLSSNR.MVec", motion); parameters->Set("DLSSNR.Output", output);
    parameters->Set("DLSSNR.Reset", reset ? 1u : 0u); parameters->Set("DLSSNR.DepthInverted", 0u);
    parameters->Set("DLSSNR.MVecScaleX", motionScale); parameters->Set("DLSSNR.MVecScaleY", motionScale);
    DlssNr::GuideRegions guides{}; guides.depth = {0, 0, w, h}; guides.motion = {0, 0, w, h};
    DlssNr::SetModelRegions(parameters, {w, h}, guides);
    DlssNr::SetModelTuning(parameters, settings[pass]);
    CheckResult(compatibilityFeatures[pass] ? compatibility->Evaluate(commands, features[pass], parameters) :
        evaluate(commands, features[pass], parameters, nullptr), "NR evaluation");
}
}
