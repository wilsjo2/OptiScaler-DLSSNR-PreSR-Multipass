#pragma once
#include "Common.h"
#include <nvsdk_ngx.h>
#include <array>
#include <mutex>
#include <dlssnr/DlssNr_CompatibilityRuntime.h>
namespace nr {
class Runtime {
public:
    Runtime(ID3D12Device* device, const std::filesystem::path& modelPath, bool unlockAdaMfg = false);
    // Independent features sharing the owning runtime's initialized driver/provider.
    explicit Runtime(std::shared_ptr<Runtime> owner);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    NVSDK_NGX_Result Create(ID3D12GraphicsCommandList* commands, UINT width, UINT height,
                           const DlssNr::ModelSettings& settings, unsigned passes = 1,
                           const std::array<std::optional<DlssNr::ModelSettings>, Settings::MaxPasses - 1>& overrides = {},
                           unsigned skipMask = 0); // Bit p set: create no feature for pass p (never pass 0).
    void Evaluate(ID3D12GraphicsCommandList* commands, ID3D12Resource* color, ID3D12Resource* depth,
                  ID3D12Resource* motion, ID3D12Resource* output, UINT width, UINT height, bool reset = true, unsigned pass = 0);
    // The motion vectors are one frame's; a caller that evaluates on every n-th frame only sets n here for that
    // evaluation (the history is then n frames old).
    float motionScale = 1.0f;
    // The model the pass was created with, as chosen (Settings::ModelANvfp4 included; the runtime itself got preset 1).
    unsigned RequestedModel(unsigned pass) const { return pass < requested.size() ? requested[pass] : 0; }
    // The pass's tuning as given to the runtime (clamped; the own pass host builds its parameter blocks from it).
    const DlssNr::ModelSettings& PassSettings(unsigned pass) const { return settings[std::min<size_t>(pass, settings.size() - 1)]; }
    void InitializeDriver(bool unlockAdaMfg = false);
    void Release(); // Only after the host's fence completes, or recording is discarded.
    NVSDK_NGX_Result CreateDlss(ID3D12GraphicsCommandList* commands, UINT width, UINT height, UINT outWidth, UINT outHeight);
    void EvaluateDlss(ID3D12GraphicsCommandList* commands, ID3D12Resource* colour, ID3D12Resource* depth,
                      ID3D12Resource* motion, ID3D12Resource* exposure, ID3D12Resource* output, bool reset, float frameTimeMs);
    void ReleaseDlss();
    static void CheckResult(NVSDK_NGX_Result result, const char* operation);
private:
    std::shared_ptr<Runtime> owner;
    std::mutex driverMutex;
    bool driverReady = false;
    HMODULE module = nullptr;
    ComPtr<ID3D12Device> device;
    std::array<NVSDK_NGX_Parameter*, Settings::MaxPasses> parameters{};
    NVSDK_NGX_Parameter* dlssParameters = nullptr;
    NVSDK_NGX_Handle* dlssFeature = nullptr;
    std::filesystem::path dlssPath;
    std::filesystem::path modelPath;
    std::shared_ptr<DlssNr::CompatibilityRuntime> compatibility;
    std::array<bool, Settings::MaxPasses> compatibilityFeatures{};
    UINT dlssWidth = 0, dlssHeight = 0;
    bool loggedRuntime = false;
    std::array<NVSDK_NGX_Handle*, Settings::MaxPasses> features{};
    std::array<DlssNr::ModelSettings, Settings::MaxPasses> settings{};
    std::array<unsigned, Settings::MaxPasses> requested{};
    decltype(&NVSDK_NGX_D3D12_CreateFeature) create = nullptr;
    decltype(&NVSDK_NGX_D3D12_EvaluateFeature) evaluate = nullptr;
    decltype(&NVSDK_NGX_D3D12_ReleaseFeature) release = nullptr;
    decltype(&NVSDK_NGX_D3D12_DestroyParameters) destroy = nullptr;
    decltype(&NVSDK_NGX_D3D12_AllocateParameters) allocate = nullptr;
    decltype(&NVSDK_NGX_D3D12_GetCapabilityParameters) capabilities = nullptr;
    static void VerifyNvidia(const std::filesystem::path& path);
    NVSDK_NGX_Result (*shutdown)() = nullptr;
};
}
