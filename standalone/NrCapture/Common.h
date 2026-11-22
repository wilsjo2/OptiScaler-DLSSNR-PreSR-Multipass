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
// `chosen` is the DLL the user explicitly picked: it keeps another package's bundled runtime from being replaced.
std::filesystem::path ResolveRuntimePath(const std::filesystem::path& saved, const std::filesystem::path& directory,
                                         const std::filesystem::path& chosen = {});
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
// Pass 2+ shortcuts. Re-apply evaluates nothing: one dispatch extends the last evaluated pass's change.
// Borrowing evaluates but skips the network blocks an earlier pass already ran (below wins over transformer).
struct PassShortcut {
    bool reapply = false, borrowTransformer = false, borrowBelow = false;
    unsigned borrowLevel = 32; // 2, 4, 8, 16 or 32: borrow every block at 1/level size and below.
    bool operator==(const PassShortcut&) const = default;
};
struct Settings {
    static constexpr unsigned MaxPasses = 10;
    // A pass's model (ModelSettings::preset): the runtime's presets 0 automatic, 1 Model A, 2 Model B, 3 legacy, and
    // one of ours: Model A (the runtime is given preset 1) with the parts ticked on the NVFP4 tab as 4-bit kernels.
    // The 4-bit kernels run in passes with this choice only; a Model B pass still follows the feed-forward box.
    static constexpr unsigned ModelANvfp4 = 4;
    // Pass-in-one models: the 4-bit kernels with another set of weights (nvfp4/swin4/swin4_set<n>.*). Such a pass
    // runs all of its own 4-bit parts, so while a pass has one, the NVFP4 tab's boxes are not what is in use. Each
    // is run as ONE Cinematic pass at intensity, tone and structure 1.
    // NVFP4 3xPass-in-one (set 12): the look of three passes. NVFP4 2xPass-in-one (set 13): the look of two.
    // The numbers 5..14 belonged to models that were removed: a saved one is refused (Runtime::Create), never run
    // as another model.
    static constexpr unsigned ModelANvfp4TwoTriple = 15, ModelANvfp4TwoDouble = 16;
    // FP8 pass-in-one models: NVIDIA's own FP8 kernels of Model A with weights of ours (nvfp4/NrFp8.h; RTX 40 and 50
    // series). FP8 2xPass-in-one (set 1): the look of two passes; FP8 3xPass-in-one (set 2): of three; their v2 weights
    // (sets 3 and 4). No 4-bit kernel runs in such a pass. Run as ONE Cinematic pass at intensity, tone and structure 1,
    // like the 4-bit ones.
    static constexpr unsigned ModelAFp8TwoDouble = 17, ModelAFp8TwoTriple = 18, ModelAFp8TwoDouble2 = 19,
                              ModelAFp8TwoTriple2 = 20;
    // The same four looks with their 4K weights: NVFP4 2xPass-in-one 4K (set 10), NVFP4 3xPass-in-one 4K (set 11), FP8
    // 2xPass-in-one 4K (FP8 set 5), FP8 3xPass-in-one 4K (FP8 set 6).
    static constexpr unsigned ModelANvfp4TwoDouble4k = 21, ModelANvfp4TwoTriple4k = 22, ModelAFp8TwoDouble4k = 23,
                              ModelAFp8TwoTriple4k = 24;
    // INT8 models: Model A with INT8 weights of its own and INT8 kernels of ours, run wholly on the own pass host
    // (nvfp4/NrHost.h; RTX 30, 40 and 50 series, no NVIDIA NR runtime launch). Original - INT8 (set 1): the look of
    // Model A. INT8 2xPass-in-one (set 2): one Cinematic pass at intensity, tone and structure 1 with the look of
    // two. Files: nvfp4/int8.
    static constexpr unsigned ModelAInt8 = 25, ModelAInt8TwoDouble = 26, ModelLast = 26;
    // One version of each model. The numbers 15..20 were earlier weights of the four pass-in-one models: a saved one
    // means the model as it is now (21..24).
    static constexpr unsigned CurrentModel(unsigned preset)
    {
        return preset == ModelANvfp4TwoDouble ? ModelANvfp4TwoDouble4k : preset == ModelANvfp4TwoTriple ? ModelANvfp4TwoTriple4k
            : preset == ModelAFp8TwoDouble || preset == ModelAFp8TwoDouble2 ? ModelAFp8TwoDouble4k
            : preset == ModelAFp8TwoTriple || preset == ModelAFp8TwoTriple2 ? ModelAFp8TwoTriple4k : preset;
    }
    static constexpr bool IsPassInOne(unsigned preset)
    {
        return preset == ModelANvfp4TwoTriple || preset == ModelANvfp4TwoDouble || preset == ModelANvfp4TwoDouble4k ||
               preset == ModelANvfp4TwoTriple4k;
    }
    static constexpr bool IsNvfp4(unsigned preset) { return preset == ModelANvfp4 || IsPassInOne(preset); }
    static constexpr bool IsFp8(unsigned preset)
    {
        return (preset >= ModelAFp8TwoDouble && preset <= ModelAFp8TwoTriple2) || preset == ModelAFp8TwoDouble4k ||
               preset == ModelAFp8TwoTriple4k;
    }
    static constexpr bool IsInt8(unsigned preset) { return preset >= ModelAInt8 && preset <= ModelAInt8TwoDouble; }
    static constexpr unsigned Int8Set(unsigned preset) { return IsInt8(preset) ? preset - (ModelAInt8 - 1) : 0; } // nvfp4/int8/int8_set<n>
    static constexpr bool IsModel(unsigned preset) { return preset <= 3 || IsNvfp4(preset) || IsFp8(preset) || IsInt8(preset); }
    static constexpr unsigned Nvfp4Set(unsigned preset) // 0: the tab's choice
    {
        return preset == ModelANvfp4TwoDouble4k ? 10 : preset == ModelANvfp4TwoTriple4k ? 11 : IsPassInOne(preset) ? preset - 3 : 0;
    }
    static constexpr unsigned Fp8Set(unsigned preset) // nvfp4/fp8/fp8_set<n>
    {
        return preset == ModelAFp8TwoDouble4k ? 5 : preset == ModelAFp8TwoTriple4k ? 6 : IsFp8(preset) ? preset - 16 : 0;
    }
    std::filesystem::path runtime;
    std::filesystem::path runtimeChosen; // The DLL last picked or typed in the box: kept even in another package.
    DlssNr::ModelSettings model{0, 0, 1, 1, 1, -1, true}; // Match OptiScaler's first-pass defaults.
    std::array<std::optional<DlssNr::ModelSettings>, MaxPasses - 1> passOverrides{};
    std::array<std::optional<PassResolutionSettings>, MaxPasses - 1> passResolutionOverrides{};
    std::array<PassShortcut, MaxPasses - 1> passShortcuts{}; // Saved for inactive passes too.
    float reapplyGain = 0.81f; // 0..2; a run of r re-apply passes applies r * gain.
    // 1..4: a 100 % pass, or one below 100 % with the "Difference" transfer, runs its model on every n-th frame only; on the frames
    // between, the last edit (answer - input at the model's size) is added to the new frame. The picture is never late,
    // the edit is by up to n - 1 frames.
    unsigned editInterval = 1;
    // The picture does not wait for the model: every frame takes the kept edit, and the model then evaluates the
    // frame for the ones after it (every editInterval-th frame). The edit is one frame older than with the interval
    // alone; the model's time is no longer in the picture's way to the display. A cut is still evaluated at once.
    bool editLate = false;
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
    unsigned transfer = 1; // 0 classic composition, 1 matched residuals, 2 edge-aware (spatial only), 3 difference.
    unsigned upscaler = 0; // 0 bilinear, 1 private DLSS SR below native size.
    DlssNr::Spatial::Settings spatial{}; // Pass 1's peripheral compression; later passes can override it.
    bool showFoveation = false; // Session-only wireframe preview of one pass's actual mapping.
    unsigned foveationPreviewPass = 0;
    unsigned compare = 0; // 0 processed, 1 original, 2 wipe, 3 side by side (same captured frame).
    // Wipe and side by side: -1 puts the original picture on the other side; a model (a ModelSettings::preset value)
    // puts there the same passes run with that model in place of each pass's own (ComparisonSide).
    int compareModel = -1;
    // Not saved. The comparison side's settings carry which kernel layers the picture's own passes need, so that both
    // sides ask for the same layers (bit 0: a 4-bit model, 1: a pass-in-one 4-bit model, 2: an FP8 model).
    unsigned keepLayers = 0;
    bool enabled = true;
    bool estimateMotion = false;
    // Experimental: the transformer's feed-forward layers run as NVFP4 kernels of ours (RTX 50 series; nvfp4/NrFfnV2.h).
    bool nvfp4FeedForward = false;
    // Experimental: Model A's fused Swin blocks run as NVFP4 kernels of ours (nvfp4/NrSwin4.h). nvfp4Swin: 1 = width 32,
    // 2 = 64, 4 = 128, 8 = 256. nvfp4SwinLayers (settings file only): bit n clear keeps the n-th block (0..43, in launch
    // order) on the runtime's kernel. nvfp4SwinTwo (settings file only): NrSwin4::Options::twoTerms.
    unsigned nvfp4Swin = 0, nvfp4SwinTwo = 7;
    unsigned long long nvfp4SwinLayers = ~0ull;
    // The fused blocks' 4-bit weights: true = swin4_trained.bin (what the application always asks for); false = the
    // first set, swin4.bin, where a folder still has one.
    bool nvfp4SwinTrained = true;
    // Experimental: the rest of Model A's pass as NVFP4 kernels of ours (nvfp4/NrNet4.h), a bit per group.
    // nvfp4Split (width-512 split blocks): 1 feed-forward, 2 its projection, 4 QKV and attention, 8 attention projection,
    // 16 bottleneck head, 32 decoder input. nvfp4Vit (transformer): 1 QKV, 2 attention, 4 projection.
    // nvfp4PrePost: 1 input block, 2 output block. nvfp4Net (the "Kernel options" field): further NrNet4::Parse items, such as
    // a group's options word ("vit.attention=1") or the launches of a family to leave alone ("split.slots=<hex>").
    unsigned nvfp4Split = 0, nvfp4Vit = 0, nvfp4PrePost = 0;
    // The trained weight records of those families (<family>4_trained.bin beside the first set) where the folder has them.
    bool nvfp4SplitTrained = true, nvfp4VitTrained = true, nvfp4FfnTrained = true;
    std::string nvfp4Net;
    unsigned tone = 2; // UI order preserves old INI values: Hybrid, Neutwo, soft knee, Hybrid replace, Neutwo replace.
    unsigned inputColour = 0; // 0 follows display HDR, 1 finished SDR, 2 linear HDR.
    unsigned maxFps = 0; // Processing cap; 0 follows available capture/GPU cadence.
    unsigned captureBackend = 0; // 0 WGC, 1 DXGI Desktop Duplication (screens only).
    unsigned frameGeneration = 1; // Total multiplier: 1 = off; 2..4 require runtime/hardware support.
    // "Pacing and latency" tab.
    // How the output reaches the screen with DLSS FG off: 0 = overlay surface (DirectComposition; the original way),
    // 1 = flip window: a flip-model swapchain on the output window, which NVIDIA's driver Smooth Motion can take.
    // DLSS FG always uses its own flip-model swapchain. Takes effect when capture starts.
    unsigned presentMode = 0;
    // 0 = standard pacing (the original: the capture and the processing each keep their own timer at the target FPS);
    // 1 = low latency: each new source frame is copied and processed as soon as it arrives (the target FPS still
    // caps the average, with 10 % room, without holding back frames that arrive unevenly: CapturePacer), and the newest
    // frame is taken again right before recording.
    unsigned latencyMode = 0;
    // 0..0.9, "Steady edit": the model's edit (its answer minus its input) is blended with the previous frame's edit,
    // moved along the estimated motion; where the two differ a lot (movement the vectors miss, a new scene) the new edit
    // is kept. 0 = off (the original). Needs "Estimate motion"; not with foveation, with passes that have their own
    // resolution settings, nor with an edit interval above 1 (the tab's note says so). With "Picture first" it runs on
    // the late evaluation.
    float editSmoothing = 0;
    // Motion estimation keeps zero motion where it matches nearly as well as the best displacement (by this share of
    // the patch error). 0 = the original rule (zero wins only ties). Reduces vectors jittering on a still picture.
    float motionZeroBias = 0;
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
    // What a pass runs with (the same as what the controls hold and the INI saves).
    PassResolutionSettings EffectiveResolution(unsigned pass) const { return PassResolution(pass); }
    PassShortcut Shortcut(unsigned pass) const { return pass && pass < std::min(passes, MaxPasses) ? passShortcuts[pass - 1] : PassShortcut{}; }
    unsigned ReapplyMask() const {
        unsigned mask = 0;
        for (unsigned pass = 1; pass < std::min(passes, MaxPasses); ++pass) if (Shortcut(pass).reapply) mask |= 1u << pass;
        return mask;
    }
    bool HasPassResolutionOverrides() const {
        for (unsigned pass = 1; pass < std::min(passes, MaxPasses); ++pass) {
            if (Shortcut(pass).reapply) continue; // A re-apply pass runs no model.
            if (passResolutionOverrides[pass - 1]) return true;
        }
        return false;
    }
    auto Models() const {
        std::array<DlssNr::ModelSettings, MaxPasses> result{};
        for (unsigned i = 0; i < std::min(passes, MaxPasses); ++i) result[i] = PassModel(i);
        return result;
    }
    bool Nvfp4Parts() const { return nvfp4FeedForward || nvfp4Swin || nvfp4Split || nvfp4Vit || nvfp4PrePost; }
    // An active pass that runs the network has the model "Model A - NVFP4".
    bool Nvfp4Pass() const {
        for (unsigned pass = 0; pass < std::min(passes, MaxPasses); ++pass)
            if (IsNvfp4(PassModel(pass).preset) && !Shortcut(pass).reapply) return true;
        return (keepLayers & 1) || (ComparingModels() && IsNvfp4(unsigned(compareModel)));
    }
    // An active pass that runs the network has an FP8 pass-in-one model.
    bool Fp8Pass() const {
        for (unsigned pass = 0; pass < std::min(passes, MaxPasses); ++pass)
            if (IsFp8(PassModel(pass).preset) && !Shortcut(pass).reapply) return true;
        return (keepLayers & 4) || (ComparingModels() && IsFp8(unsigned(compareModel)));
    }
    // An active pass that runs the network has a pass-in-one model.
    bool Nvfp4PassInOne() const {
        for (unsigned pass = 0; pass < std::min(passes, MaxPasses); ++pass)
            if (Nvfp4Set(PassModel(pass).preset) && !Shortcut(pass).reapply) return true;
        return (keepLayers & 2) || (ComparingModels() && Nvfp4Set(unsigned(compareModel)));
    }
    // An active pass that runs the network has an INT8 model.
    bool Int8Pass() const {
        for (unsigned pass = 0; pass < std::min(passes, MaxPasses); ++pass)
            if (IsInt8(PassModel(pass).preset) && !Shortcut(pass).reapply) return true;
        return false;
    }
    // Wipe or side by side with a model on the other side (not while the edit is hidden or the original is shown).
    bool ComparingModels() const { return compareModel >= 0 && IsModel(unsigned(compareModel)) && (compare == 2 || compare == 3) && applyModel && enabled && !DifferenceOfModels(); }
    // The "Difference (20x)" debug view with a model chosen as the other side: the whole picture shows this model's
    // answer minus the other model's (same passes), in place of the answer minus the model's input.
    bool DifferenceOfModels() const { return compareModel >= 0 && IsModel(unsigned(compareModel)) && debugView == 3 && applyModel && enabled; }
    // The other side of a model comparison: these settings with every pass's model replaced, shown whole.
    Settings ComparisonSide() const {
        Settings side = *this;
        side.keepLayers = (Nvfp4Pass() ? 1u : 0u) | (Nvfp4PassInOne() ? 2u : 0u) | (Fp8Pass() ? 4u : 0u);
        side.model.preset = unsigned(compareModel);
        for (auto& pass : side.passOverrides) if (pass) pass->preset = unsigned(compareModel);
        side.compare = 0; side.compareModel = -1; side.showFoveation = false; side.debugView = 0;
        return side;
    }
    // The picture's own side of a model comparison: shown whole; the two are put together afterwards.
    Settings PictureSide() const {
        Settings side = *this;
        const unsigned other = unsigned(compareModel);
        side.keepLayers = (IsNvfp4(other) ? 1u : 0u) | (Nvfp4Set(other) ? 2u : 0u) | (IsFp8(other) ? 4u : 0u);
        side.compare = 0; side.compareModel = -1;
        return side;
    }
    bool operator==(const Settings&) const = default;
};
Settings LoadSettings(const std::filesystem::path& file = {});
void SaveSettings(const Settings& settings, const std::filesystem::path& file = {});
}
