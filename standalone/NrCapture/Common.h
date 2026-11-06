#pragma once
#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <filesystem>
#include <format>
#include <string>
#include <stdexcept>
#include <array>
#include <optional>
#include <dlssnr/DlssNr_ModelParameters.h>
#include <shaders/dlssnr/DlssNr_Spatial.h>
namespace nr {
template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
void Check(HRESULT result, const char* operation);
void Log(const std::string& message);
inline constexpr size_t MaxLogBytes = 2 * 1024 * 1024;
// The caller serializes writes; an oversized old log retains only its newest bytes.
void AppendBoundedLog(const std::filesystem::path& file, const std::string& entry);
std::filesystem::path DataDirectory();
std::filesystem::path ExecutableDirectory();
std::filesystem::path ResolveRuntimePath(const std::filesystem::path& saved, const std::filesystem::path& directory);
std::string Narrow(const std::wstring& value);
std::wstring Wide(const std::string& value);
double ClockMs();
struct Handle {
    HANDLE value = nullptr;
    ~Handle() { if (value) CloseHandle(value); }
    Handle() = default;
    explicit Handle(HANDLE h) : value(h) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
struct PassResolutionSettings {
    unsigned modelScale = 100, downscaler = 4, transfer = 1, upscaler = 0;
    DlssNr::Spatial::Settings spatial{};
    bool operator==(const PassResolutionSettings&) const = default;
};
struct Settings {
    static constexpr unsigned MaxPasses = 10;
    std::filesystem::path runtime;
    DlssNr::ModelSettings model{0, 0, 1, 1, 1, -1, true}; // Match OptiScaler's first-pass defaults.
    std::array<std::optional<DlssNr::ModelSettings>, MaxPasses - 1> passOverrides{};
    std::array<std::optional<PassResolutionSettings>, MaxPasses - 1> passResolutionOverrides{};
    bool unlockPasses = false;
    float strength = 1.0f;
    float colourStrength = 1.0f;
    float skinDetail = 1.0f, skinColour = 1.0f;
    float environmentDetail = 1.0f, environmentColour = 1.0f;
    bool separateSkin = false, previewSkin = false;
    float referenceWhiteNits = 0; // 0 follows the desktop; native HDR can use an explicit reference.
    float peakBrightnessNits = 0; // 0 leaves output highlights unchanged.
    float blackLevelNits = 0; // Shadow offset, tapered to zero at reference white.
    unsigned passes = 1;
    unsigned modelScale = 100; // Percentage of capture width and height, 25..200.
    unsigned downscaler = 4; // Same order as OptiScaler: FSR1, Bicubic, Catmull, Lanczos2/3, Kaiser2/3, MAGIC.
    float highlightGuard = 2;
    float haloSuppression = 0, haloBrightSuppression = 0, haloRadius = 2;
    float compareSplit = 0.5f, compareZoom = 1, labelScale = 1;
    bool compareSwap = false, compareLabels = true;
    unsigned debugView = 0;
    bool holdFrame = false, applyModel = true;
    unsigned transfer = 1; // 0 classic composition, 1 matched residuals, 2 edge-aware (spatial only).
    unsigned upscaler = 0; // 0 bilinear, 1 private DLSS SR below native size.
    DlssNr::Spatial::Settings spatial{}; // Pass 1's peripheral compression; later passes can override it.
    bool showFoveation = false; // Session-only wireframe preview of one pass's actual mapping.
    unsigned foveationPreviewPass = 0;
    unsigned compare = 0; // 0 processed, 1 original, 2 wipe, 3 side by side (same captured frame).
    bool enabled = true;
    bool estimateMotion = false;
    unsigned tone = 2; // UI order preserves old INI values: Hybrid, Neutwo, soft knee, Hybrid replace, Neutwo replace.
    unsigned inputColour = 0; // 0 follows display HDR, 1 finished SDR, 2 linear HDR.
    unsigned maxFps = 0; // Processing cap; 0 follows available capture/GPU cadence.
    unsigned captureBackend = 0; // 0 WGC, 1 DXGI Desktop Duplication (screens only).
    unsigned frameGeneration = 1; // Total multiplier: 1 = off; 2..4 require runtime/hardware support.
    double ProcessingFps() const { return double(maxFps) / std::max(1u, frameGeneration); }
    unsigned HdrMode() const { constexpr unsigned modes[]{3, 1, 0, 4, 2}; return modes[std::min(tone, 4u)]; }
    DlssNr::ModelSettings PassModel(unsigned pass) const {
        if (pass && passOverrides.at(pass - 1)) return *passOverrides[pass - 1];
        auto value = model; if (pass) value.localTone = 0; return value;
    }
    PassResolutionSettings PassResolution(unsigned pass) const {
        if (pass && passResolutionOverrides.at(pass - 1)) return *passResolutionOverrides[pass - 1];
        return {modelScale, downscaler, transfer, upscaler, spatial};
    }
    bool HasPassResolutionOverrides() const {
        for (unsigned pass = 1; pass < std::min(passes, MaxPasses); ++pass)
            if (passResolutionOverrides[pass - 1]) return true;
        return false;
    }
    auto Models() const {
        std::array<DlssNr::ModelSettings, MaxPasses> result{};
        for (unsigned i = 0; i < std::min(passes, MaxPasses); ++i) result[i] = PassModel(i);
        return result;
    }
    bool operator==(const Settings&) const = default;
};
Settings LoadSettings(const std::filesystem::path& file = {});
void SaveSettings(const Settings& settings, const std::filesystem::path& file = {});
}
