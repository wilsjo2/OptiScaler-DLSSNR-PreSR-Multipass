#include "pch.h"
#include "DlssNrFeature_Dx12.h"
#include "DlssNr_MenuSections.h"
#include "DlssNr_Placement.h"
#include "DlssNr_Status.h"
#include "PassProfiles.h"
#include "NrPassBorrow.h"
#include <shaders/dlssnr/DlssNr_Spatial.h>
#include <Config.h>
#include <menu/menu_common.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>

namespace DlssNr::MenuSections
{
template <typename Option> static void Checkbox(const char* label, Option& option)
{
    bool value = option.value_or_default();
    if (ImGui::Checkbox(label, &value))
        option = value;
}

template <typename Option>
static void Slider(const char* label, Option& option, float minimum, float maximum, const char* format = "%.2f",
                   std::optional<float> reset = {}, ImGuiSliderFlags flags = 0)
{
    float value = option.value_or_default();
    if (ImGui::SliderFloat(label, &value, minimum, maximum, format, flags))
        option = value;
    if (reset)
    {
        ImGui::SameLine();
        ImGui::PushID(label);
        if (ImGui::SmallButton("Reset"))
            option = *reset;
        ImGui::PopID();
    }
}

static void StoreSpatial(Config* config, const Spatial::Settings& value)
{
    config->DlssNrSpatialCenterX = value.centerX;
    config->DlssNrSpatialCenterY = value.centerY;
    config->DlssNrSpatialWorkX = value.workX;
    config->DlssNrSpatialWorkY = value.workY;
    config->DlssNrSpatialOffsetX = value.offsetX;
    config->DlssNrSpatialOffsetY = value.offsetY;
    config->DlssNrSpatialShiftX = value.shiftX;
    config->DlssNrSpatialShiftY = value.shiftY;
}

// Used only for user edits. Invalid INI values remain visible as a runtime fallback.
static void ConstrainSpatialControls(Spatial::Settings& value, float scale)
{
    const float minimumWork = Spatial::MinimumWorkPercent(scale);
    auto axis = [&](float& center, float& work, float& offset)
    {
        center = std::clamp(std::isfinite(center) ? center : 80.0f, 1.0f, 99.5f);
        work = std::clamp(std::isfinite(work) ? work : 90.0f, std::max(minimumWork, center + 0.5f), 100.0f);
        const float limit = Spatial::MaxCenterOffset(center);
        offset = std::clamp(std::isfinite(offset) ? offset : 0.0f, -limit, limit);
    };
    axis(value.centerX, value.workX, value.offsetX);
    axis(value.centerY, value.workY, value.offsetY);
    const auto x = Spatial::WorkShiftLimits(value, false);
    const auto y = Spatial::WorkShiftLimits(value, true);
    value.shiftX = std::clamp(std::isfinite(value.shiftX) ? value.shiftX : 0.0f, x.first, x.second);
    value.shiftY = std::clamp(std::isfinite(value.shiftY) ? value.shiftY : 0.0f, y.first, y.second);
}

static void RenderSpatial(Config* config, bool presentationControls = true)
{
    Checkbox("Peripheral compression", config->DlssNrSpatialCompression);
    HelpMarker(
        "Keep more model detail in the centre and compress the edges. Model resolution still scales the whole image.");
    if (presentationControls)
    {
        bool preview = config->DlssNrDebugView.value_or_default() == 4;
        if (ImGui::Checkbox("Preview", &preview))
            config->DlssNrDebugView = preview ? 4u : 0u;
        HelpMarker("Show the packed model input before spatial unpacking, scaled to fill the screen. "
                   "This is the same as the Compressed model input debug view. "
                   "Without active compression, shows the ordinary model input. Apply model must be enabled.");
    }
    if (!config->DlssNrSpatialCompression.value_or_default())
        return;

    if (presentationControls)
    {
        const auto feature = State::Instance().currentFeature;
        const bool nativeVk = feature && feature->Api() == API::Vulkan && !feature->IsWithDx12();
        const auto status = ReadStatus(nativeVk ? Backend::Vulkan : Backend::Dx12);
        if (!status.spatialStatus.empty())
            ImGui::TextWrapped("%s", status.spatialStatus.c_str());
    }

    struct SpatialEdit
    {
        Spatial::Settings pending;
        bool editing = false;
    };
    static std::unordered_map<ImGuiID, SpatialEdit> edits;
    auto& edit = edits[ImGui::GetID("Spatial layout")];
    auto& pending = edit.pending;
    auto& editing = edit.editing;
    if (!editing)
        pending = Spatial::ReadSettings(*config);
    const float scale = config->DlssNrWorkingScale.value_or_default();
    ConstrainSpatialControls(pending, scale);
    bool commit = false;
    auto slider = [&](const char* label, float& value, float lo, float hi)
    {
        if (ImGui::SliderFloat(label, &value, lo, hi, "%.1f%%"))
            editing = true;
        if (ImGui::IsItemDeactivatedAfterEdit())
            commit = true;
    };
    slider("Centre width", pending.centerX, 1.0f, pending.workX - 0.5f);
    slider("Centre height", pending.centerY, 1.0f, pending.workY - 0.5f);
    const float minimumWork = Spatial::MinimumWorkPercent(scale);
    slider("Working width", pending.workX, std::max(minimumWork, pending.centerX + 0.5f), 100.0f);
    slider("Working height", pending.workY, std::max(minimumWork, pending.centerY + 0.5f), 100.0f);
    const float xLimit = Spatial::MaxCenterOffset(pending.centerX);
    const float yLimit = Spatial::MaxCenterOffset(pending.centerY);
    slider("Centre horizontal offset", pending.offsetX, -xLimit, xLimit);
    slider("Centre vertical offset", pending.offsetY, -yLimit, yLimit);
    const auto xShift = Spatial::WorkShiftLimits(pending, false);
    const auto yShift = Spatial::WorkShiftLimits(pending, true);
    slider("Working region horizontal shift", pending.shiftX, xShift.first, xShift.second);
    slider("Working region vertical shift", pending.shiftY, yShift.first, yShift.second);
    HelpMarker("Extreme shifts can leave an edge with less than one working pixel. Compression then falls back to "
               "ordinary NR; the status above explains why.");
    if (ImGui::SmallButton("Reset compression layout"))
    {
        pending = Spatial::Settings {};
        commit = true;
    }
    if (commit)
    {
        ConstrainSpatialControls(pending, scale);
        StoreSpatial(config, pending);
        editing = false;
    }
    if (presentationControls)
    {
        Checkbox("Show centre outline", config->DlssNrSpatialShowCenter);
        Checkbox("Show working region outline", config->DlssNrSpatialShowWork);
    }
    ImGui::TextWrapped("Centre detail follows Model resolution. Strong edge compression can soften detail or shimmer "
                       "during movement.");
}

static void RenderResolution(Config* config, bool presentationControls)
{
    // Resolution changes rebuild model resources; commit only after releasing the slider.
    static std::unordered_map<ImGuiID, int> pendingScales;
    const ImGuiID scaleId = ImGui::GetID("Model resolution");
    const auto pending = pendingScales.find(scaleId);

    int scalePercent = pending != pendingScales.end()
                           ? pending->second
                           : (int) lroundf(Profiles::PassResolutionSettings(*config, 0).workingScale * 100.0f);

    if (ImGui::SliderInt("Model resolution", &scalePercent, 25, 200, "%d%%"))
        pendingScales[scaleId] = scalePercent;

    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        config->DlssNrWorkingScale = std::clamp(scalePercent, 25, 200) / 100.0f;
        if (config->DlssNrSpatialCompression.value_or_default())
        {
            auto spatial = Spatial::ReadSettings(*config);
            ConstrainSpatialControls(spatial, config->DlssNrWorkingScale.value_or_default());
            StoreSpatial(config, spatial);
        }
        pendingScales.erase(scaleId);
    }

    HelpMarker("50% halves width and height. 100% uses the full input size.");
    RenderSpatial(config, presentationControls);

    if (scalePercent > 100)
    {
        static const char* dsNames[] = { "FSR1",     "Bicubic", "Catmull-Rom", "Lanczos2",
                                         "Lanczos3", "Kaiser2", "Kaiser3",     "MAGIC" };
        int ds = (int) config->DlssNrScalingDownscaler.value_or_default();
        if (ds < 0 || ds >= IM_ARRAYSIZE(dsNames))
            ds = (int) Scaler::Lanczos3;

        if (ImGui::Combo("Downscaler (NR)", &ds, dsNames, IM_ARRAYSIZE(dsNames)))
            config->DlssNrScalingDownscaler = (Scaler) ds;

        HelpMarker("Downsampling filter for resolutions above 100%.");
    }
    {
        const bool reduced = config->DlssNrWorkingScale.value_or_default() < 0.999f;

        ImGui::BeginDisabled(!reduced);

        static const char* enlargeNames[] = {
            "Classic",           "Matched residual",         "Matched residual + private upscaler",
            "Lighting + colour", "Lighting + colour + private upscaler", "Edge-aware"
        };
        int enlarge = (int) std::min(config->DlssNrTransfer.value_or_default(), 5u);

        if (ImGui::Combo("Enlargement", &enlarge, enlargeNames, IM_ARRAYSIZE(enlargeNames)))
            config->DlssNrTransfer = (uint32_t) enlarge;

        HelpMarker("Below 100%: Lighting + colour resizes lighting gain and colour changes separately, then applies "
                   "them at full resolution. This can reduce resize halos; fully black pixels remain black. "
                   "Edge-aware uses the original picture's brightness to keep edits on the correct side of edges. "
                   "Private upscaler modes require post-upscale DX12 processing.");

        if (enlarge == 2 || enlarge == 4)
        {
            static const char* backends[] = { "DLSS SR", "FSR 2.2", "FidelityFX", "XeSS" };
            int backend = Profiles::PassResolutionSettings(*config, 0).privateUpscaler;
            if (ImGui::Combo("Private upscaler", &backend, backends, IM_ARRAYSIZE(backends)))
                config->DlssNrPrivateUpscaler = backend;
            HelpMarker("Upscaler used by this pass for temporal enlargement. Requires its installed runtime. "
                       "Unavailable on native Vulkan: select a spatial enlargement mode there.");
        }
        ImGui::EndDisabled();
    }
}

void RenderInput(Config* config)
{
    RenderResolution(config, true);
    static const char* reversibleNames[] = { "Off (soft knee)", "Neutwo proxy + composed", "Neutwo proxy + replace",
                                             "Hybrid proxy + composed", "Hybrid proxy + replace" };
    int reversible = (int) config->DlssNrReversibleMode.value_or_default();
    if (reversible < 0 || reversible > 4)
        reversible = 0;
    if (ImGui::Combo("HDR mapping (experimental)", &reversible, reversibleNames, IM_ARRAYSIZE(reversibleNames)))
        config->DlssNrReversibleMode = (uint32_t) reversible;

    HelpMarker("HDR mapping curve. Replace bypasses strength and highlight controls.");

    if (reversible == 2 || reversible == 4)
        Slider("Restore sharpness", config->DlssNrReplaceDetailStrength, 0.0f, 2.0f, "%.2f", 0.0f);

    const char* exposureNames[] = { "Manual", "Game exposure", "Automatic HDR exposure" };
    const auto source = config->DlssNrWhitePointSource.value_or_default();
    int selected = source == 3 ? 2 : source == 1 ? 1 : 0;
    if (ImGui::Combo("White point source", &selected, exposureNames, 3))
        config->DlssNrWhitePointSource = selected == 2 ? 3u : static_cast<unsigned>(selected);
    if (selected)
    {
        auto& trim = selected == 2 ? config->DlssNrAutoExposureTrim : config->DlssNrWhitePointTrim;
        Slider("Exposure trim", trim, 0.001f, 1000.0f, "%.3fx", selected == 2 ? 5.0f : 1.0f,
               ImGuiSliderFlags_Logarithmic);
        if (selected == 2)
            Slider("Highlight protection", config->DlssNrAutoExposureHighlightProtection, 0.0f, 100.0f, "%.0f%%", 0.0f);
        auto& curve = selected == 2 ? config->DlssNrAutoExposureTrimAnchors : config->DlssNrExposureTrimAnchors;
        char text[512] {};
        const auto value = curve.value_or_default();
        std::memcpy(text, value.data(), std::min(value.size(), sizeof(text) - 1));
        if (ImGui::InputText("Trim anchors", text, sizeof(text)))
            curve = std::string(text);
        HelpMarker("Up to eight base-white-point:trim pairs, e.g. 1:5 100:2. Trim interpolates logarithmically. Clear "
                   "for a fixed trim.");
    }
    Slider("Paper white", config->DlssNrWhitePointScale, 0.25f, 2000.0f, "%.2fx", {}, ImGuiSliderFlags_Logarithmic);
    HelpMarker("Higher values darken the NR input; lower values brighten it.");
}

// Model tuning rebuilds the feature; commit slider changes only on release.
template <typename Option>
static void DeferredSlider(const char* label, Option* opt, float mn, float mx, float def, bool inheritReset = false)
{
    static std::unordered_map<ImGuiID, float> pending;
    const ImGuiID id = ImGui::GetID(label);

    auto it = pending.find(id);
    float value = it != pending.end() ? it->second : (opt->value_or(def));

    if (ImGui::SliderFloat(label, &value, mn, mx, "%.2f"))
        pending[id] = value;

    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        *opt = std::clamp(value, mn, mx);
        pending.erase(id);
    }

    ImGui::SameLine();

    const std::string resetId = std::string("Reset##") + label;
    if (ImGui::SmallButton(resetId.c_str()))
    {
        if (inheritReset)
            *opt = std::optional<float> {};
        else
            *opt = def;
        pending.erase(id);
    }

    if (std::strcmp(label, "Intensity") == 0)
        HelpMarker("Enhancement strength. 1 = default.");
    else if (std::strcmp(label, "Local structure") == 0)
        HelpMarker("Fine detail and local contrast. 1 = default.");
    else if (std::strcmp(label, "Local tone") == 0)
        HelpMarker("Broad lighting changes. Later passes default to 0.");
    else if (std::strcmp(label, "Skin structure") == 0)
        HelpMarker("Skin detail. -1 follows Local structure.");
}

template <typename Option> static void NrModelCombo(Option* opt, bool allowInherit)
{
    // DlssNr::Nvfp4::kModels, sorted by the precision of the weights: FP8 (the runtime's presets 1 and 2 and the FP8
    // pass-in-one models), INT8, NVFP4 (4-bit).
    const bool inherited = allowInherit && !opt->has_value();
    const uint32_t preset = opt->has_value() ? opt->value() : 0u;
    const char* preview = inherited ? "Inherit pass 1" : DlssNr::Nvfp4::ModelName(preset);

    if (ImGui::BeginCombo("NR model", preview))
    {
        if (allowInherit && ImGui::Selectable("Inherit pass 1", inherited))
            *opt = std::optional<uint32_t> {};
        for (const auto& model : DlssNr::Nvfp4::kModels)
        {
            const bool selected = !inherited && preset == model.preset;
            if (ImGui::Selectable(model.name, selected))
                *opt = (uint32_t) model.preset;
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    HelpMarker("The list is sorted by the precision of the weights: FP8, INT8, NVFP4.\n"
               "Automatic: use the installed NR runtime's default model.\n"
               "FP8 Original: the original model (preset 1).\n"
               "FP8 Optimized: the smaller FP8 model (preset 2), included in NR 310.9.16.0.\n"
               "Older runtimes fall back to their default when a model is unavailable.\n"
               "FP8 2xPass-in-one, FP8 3xPass-in-one: the look of two or three passes in one, as FP8 models for\n"
               "RTX 40 and 50 series GPUs (NVIDIA's own FP8 kernels with weights of ours; files in\n"
               "OptiScaler/nvfp4/fp8). The v2 models are the same two with other weights.\n"
               "INT8 Original, INT8 2xPass-in-one: INT8 models for RTX 30, 40 and 50 series GPUs (D3D12 path;\n"
               "files in OptiScaler/nvfp4/int8). They run wholly on kernels of ours and need no NR runtime.\n"
               "NVFP4 Original, NVFP4 2xPass-in-one, NVFP4 3xPass-in-one: 4-bit models for RTX 50 series GPUs\n"
               "(D3D12 path; files in OptiScaler/nvfp4). They carry all their own weights and use none of the\n"
               "runtime's, so any NR runtime that offers the original model (preset 1) works, such as 310.8.0\n"
               "or 310.9.16.0.\n"
               "The 4K models (NVFP4 and FP8): the same models with weights made for 4K output.\n"
               "There is no fallback: if a model of ours cannot run, NR is not applied and the reason is shown\n"
               "here in red.\n"
               "Run a pass-in-one model as ONE pass: Cinematic, intensity, tone and structure at 1.\n"
               "Changing the model rebuilds this pass and resets its history.");
    if (!inherited && DlssNr::Nvfp4::IsNvfp4(preset))
    {
        const char* kind = DlssNr::Nvfp4::IsInt8(preset) ? "INT8" : DlssNr::Nvfp4::IsFp8(preset) ? "FP8" : "4-bit";
        if (const auto problem = DlssNr::Nvfp4::Problem(); !problem.empty())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.35f, 0.3f, 1.0f));
            ImGui::TextWrapped("%s model is NOT running, NR is not applied: %s", kind, problem.c_str());
            ImGui::PopStyleColor();
        }
        else if (DlssNr::Nvfp4::Running())
            ImGui::TextColored(ImVec4(0.45f, 0.9f, 0.45f, 1.0f), "%s model running", kind);
        else
            ImGui::TextDisabled("%s model starting...", kind);
    }
}

static void RenderPassResolution(Config* config, unsigned int selectedPass)
{
    if (!ImGui::TreeNode("Resolution and upscaling"))
        return;

    if (selectedPass == 0)
    {
        ImGui::TextDisabled("Pass 1 also supplies defaults for later passes.");
        RenderResolution(config, false);
    }
    else
    {
        auto& extra = config->DlssNrPassOverrides[selectedPass - 1];
        bool independent = Profiles::HasPassResolutionOverride(extra);
        if (ImGui::Checkbox("Independent resolution and upscaling", &independent))
        {
            if (independent)
                Profiles::SetPassResolutionSettings(extra, Profiles::PassResolutionSettings(*config, selectedPass));
            else
                Profiles::ClearPassResolutionSettings(extra);
        }
        HelpMarker("Give this pass its own model resolution, enlargement, downsampling and peripheral compression. "
                   "Each pass reconstructs the full input size before the next pass. "
                   "Turning this off restores pass 1's sizing settings and keeps this pass's model tuning.");

        if (independent)
        {
            const auto before = Profiles::PassResolutionSettings(*config, selectedPass);
            Config values = *config;
            Profiles::ApplyPassConfig(values, *config, selectedPass);
            RenderResolution(&values, false);
            const auto after = Profiles::PassResolutionSettings(values, 0);
            if (!(before == after))
                Profiles::SetPassResolutionSettings(extra, after);
        }
        else
        {
            const auto inherited = Profiles::PassResolutionSettings(*config, selectedPass);
            ImGui::TextDisabled("Inherits pass 1: %.0f%% model resolution, peripheral compression %s.",
                                inherited.workingScale * 100.0f, inherited.spatial.enabled ? "on" : "off");
        }
    }
    ImGui::TreePop();
}

// Later passes can skip the model (re-apply) or skip part of it (borrow from the frame's first model pass).
// Quality figures: Model A, 4K median PSNR against two real passes; one pass alone is 32.1 dB.
static void RenderPassReuse(Config* config, unsigned int selectedPass)
{
    auto& pass = config->DlssNrPassOverrides[selectedPass - 1];
    bool reapply = pass.reapply.value_or(false);
    if (ImGui::Checkbox("Re-apply previous change", &reapply))
        pass.reapply = reapply;
    HelpMarker("Skip the model and repeat the previous model pass's change in one tiny dispatch "
               "(41.5 dB at gain 0.81, 39.2 dB at gain 1).");
    if (reapply)
    {
        Slider("Re-apply gain", config->DlssNrReapplyGain, 0.0f, 2.0f, "%.2f", 0.81f);
        HelpMarker("Shared by every re-applying pass: 1 repeats the change exactly, 0.81 measured best.");
    }

    const auto feature = State::Instance().currentFeature;
    const bool nativeVk = feature ? feature->Api() == API::Vulkan && !feature->IsWithDx12()
                                  : State::Instance().swapchainApi == API::Vulkan;
    // The same pass count the run uses, so the menu and the chain choice agree.
    const unsigned runPasses = std::clamp(config->DlssNrPasses.value_or_default(), 1u,
                                          config->DlssNrUnlockPasses.value_or_default() ? DlssNr::MaxPassCount
                                                                                        : DlssNr::DefaultMaxPassCount);
    const bool independent =
        Profiles::HasPassResolutionOverride(pass) || Profiles::HasPassResolutionOverrides(*config, runPasses);
    const char* unavailable = reapply       ? "Unavailable while this pass re-applies the previous change."
                              : nativeVk    ? "Unavailable on native Vulkan: borrowing intercepts the D3D12 NR "
                                              "runtime's kernel launches."
                              : independent ? "Unavailable with independent pass resolution: every pass must run "
                                              "the same model at the same size."
                                            : nullptr;
    ImGui::BeginDisabled(unavailable != nullptr);
    bool transformer = pass.borrowTransformer.value_or(false);
    if (ImGui::Checkbox("Borrow pass 1's transformer", &transformer))
        pass.borrowTransformer = transformer;
    ImGui::EndDisabled();
    HelpMarker(unavailable ? unavailable
                           : "Run the model but take pass 1's transformer blocks instead of recomputing them; saves "
                             "GPU time (43.6 dB).");
    ImGui::BeginDisabled(unavailable != nullptr);
    bool below = pass.borrowBelow.value_or(false);
    if (ImGui::Checkbox("Borrow everything at 1/x size and below", &below))
        pass.borrowBelow = below;
    ImGui::EndDisabled();
    HelpMarker(unavailable ? unavailable
                           : "Run only the outer, larger blocks and take everything at 1/x of the frame size and "
                             "smaller from pass 1; overrides the transformer box (1/32 43.1, 1/16 42.1, 1/8 40.7, "
                             "1/4 40.7, 1/2 40.0 dB).");
    static const unsigned levels[] = { 2, 4, 8, 16, 32 };
    static const char* levelNames[] = { "1/2", "1/4", "1/8", "1/16", "1/32" };
    // Show the level the run will use: ChooseCut snaps an INI value to the nearest (Below2..Below32 = 6..2).
    int index = 6 - int(NrPassBorrow::ChooseCut(false, true, pass.borrowLevel.value_or(32u)));
    ImGui::BeginDisabled(unavailable != nullptr || !below);
    ImGui::Indent();
    if (ImGui::Combo("Borrow at 1/x and below", &index, levelNames, IM_ARRAYSIZE(levelNames)))
        pass.borrowLevel = (uint32_t) levels[index];
    ImGui::Unindent();
    ImGui::EndDisabled();
    char status[192];
    NrPassBorrow::Status(status, sizeof(status));
    ImGui::TextDisabled("%s", status);
}

void RenderModel(Config* config)
{
    bool unlockPasses = config->DlssNrUnlockPasses.value_or_default();
    const int menuPassLimit = unlockPasses ? 10 : 2;
    static int passes = 1;
    static bool editingPasses = false;
    if (!editingPasses)
        passes = (int) std::clamp(config->DlssNrPasses.value_or_default(), 1u, (unsigned) menuPassLimit);
    ImGui::SliderInt("Model passes", &passes, 1, menuPassLimit, "%d", ImGuiSliderFlags_AlwaysClamp);
    editingPasses = ImGui::IsItemActive();
    if (ImGui::IsItemDeactivatedAfterEdit())
        config->DlssNrPasses = (uint32_t) std::clamp(passes, 1, menuPassLimit);
    if (ImGui::Checkbox("Unlock up to 10 passes", &unlockPasses))
    {
        config->DlssNrUnlockPasses = unlockPasses;
        config->DlssNrPasses = std::clamp(config->DlssNrPasses.value_or_default(), 1u, unlockPasses ? 10u : 2u);
    }

    // Pass shortcuts sit right under the pass count so they are visible without picking a pass to edit.
    const unsigned activePasses = std::clamp(config->DlssNrPasses.value_or_default(), 1u, (unsigned) menuPassLimit);
    if (activePasses >= 2)
    {
        ImGui::SeparatorText("Later-pass shortcuts (re-apply / borrow)");
        for (unsigned shortcutPass = 1; shortcutPass < activePasses; ++shortcutPass)
        {
            ImGui::PushID(1000 + (int) shortcutPass);
            if (activePasses > 2)
                ImGui::TextDisabled("Pass %u", shortcutPass + 1);
            RenderPassReuse(config, shortcutPass);
            ImGui::PopID();
        }
        ImGui::Separator();
    }
    else
    {
        ImGui::TextDisabled("Set 2 or more passes for the re-apply and borrow options.");
    }

    unsigned& selectedPass = EditPass();
    const auto selectedLabel = std::format("Pass {}", selectedPass + 1);
    if (ImGui::BeginCombo("Edit pass", selectedLabel.c_str()))
    {
        for (unsigned pass = 0; pass <= std::size(config->DlssNrPassOverrides); ++pass)
        {
            const auto label = std::format("Pass {}", pass + 1);
            if (ImGui::Selectable(label.c_str(), selectedPass == pass))
                selectedPass = pass;
            if (selectedPass == pass)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if (selectedPass >= config->DlssNrPasses.value_or_default())
        ImGui::TextDisabled("Inactive pass. Settings apply when this pass is enabled.");

    // Distinct widget IDs keep uncommitted slider edits with their selected pass.
    ImGui::PushID((int) selectedPass);
    const bool inherited = selectedPass != 0;
    const auto tuning = [&](auto& intensity, auto& structure, auto& tone, auto& skin, auto& autoMask)
    {
        DeferredSlider("Intensity", &intensity, 0.0f, 2.0f,
                       inherited ? config->DlssNrIntensity.value_or_default() : 1.0f, inherited);
        DeferredSlider("Local structure", &structure, 0.0f, 2.0f,
                       inherited ? config->DlssNrLocalStructure.value_or_default() : 1.0f, inherited);
        DeferredSlider("Local tone", &tone, 0.0f, 2.0f, inherited ? 0.0f : 1.0f, inherited);
        DeferredSlider("Skin structure", &skin, -1.0f, 2.0f,
                       inherited ? config->DlssNrSkinStructure.value_or_default() : -1.0f, inherited);
        bool mask = autoMask.value_or(inherited ? config->DlssNrAutoMask.value_or_default() : true);
        if (ImGui::Checkbox("Auto skin mask", &mask))
            autoMask = mask;
        if (inherited)
        {
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##mask"))
                autoMask = std::optional<bool> {};
        }
        HelpMarker("Model-based skin selection.");
    };
    static const char* styles[] = { "Standard", "Natural", "Cinematic" };
    static const char* inheritedStyles[] = { "Auto", "Standard", "Natural", "Cinematic" };
    if (selectedPass == 0)
    {
        NrModelCombo(&config->DlssNrPreset, false);
        int style = (int) std::min(config->DlssNrStyle.value_or_default(), 2u);
        if (ImGui::Combo("Style", &style, styles, IM_ARRAYSIZE(styles)))
            config->DlssNrStyle = (uint32_t) style;
        tuning(config->DlssNrIntensity, config->DlssNrLocalStructure, config->DlssNrLocalTone,
               config->DlssNrSkinStructure, config->DlssNrAutoMask);
    }
    else
    {
        auto& pass = config->DlssNrPassOverrides[selectedPass - 1];
        if (selectedPass <= 2)
            NrModelCombo(&pass.preset, true);
        else
            ImGui::TextDisabled("NR model: inherits pass 1");
        int style = pass.style.has_value() ? std::clamp((int) pass.style.value(), 0, 2) + 1 : 0;
        if (ImGui::Combo("Style", &style, inheritedStyles, IM_ARRAYSIZE(inheritedStyles)))
            pass.style = style ? std::optional<uint32_t>(style - 1) : std::nullopt;
        tuning(pass.intensity, pass.structure, pass.tone, pass.skin, pass.autoMask);
    }
    RenderPassResolution(config, selectedPass);
    ImGui::PopID();
}

void RenderBlend(Config* config)
{
    if (config->DlssNrResidualAcrossRr.value_or_default())
    {
        Checkbox("Reject stale history", config->DlssNrHistoryRejection);
        HelpMarker("RR residual path: reject old edits when the input changes and limit them to nearby current edits. "
                   "Can reduce trails. Off keeps the original history filter.");
        Slider("History confidence threshold", config->DlssNrResidualConfidenceSensitivity, 0.0f, 2.0f, "%.3f", 0.0f);
    }
    if (ImGui::TreeNode("Halo suppression (experimental)"))
    {
        Slider("Dark rims near bright objects", config->DlssNrHaloSuppression, 0.0f, 1.0f, "%.2f", 0.0f);
        Slider("Dark rims near dark objects", config->DlssNrHaloBrightSuppression, 0.0f, 1.0f, "%.2f", 0.0f);
        Slider("Halo radius (NR pixels)", config->DlssNrHaloRadius, 1.0f, 4.0f, "%.0f", 2.0f);
        HelpMarker("Lifts fine dark rims while keeping broad lighting. Both strengths at zero disable the pass. "
                   "May lighten real shadows; compare the result before keeping it on.");
        ImGui::TreePop();
    }
    if (config->DlssNrFinishedPicture.value_or_default() &&
        (config->DlssNrRunBeforeSr.value_or_default() || config->DlssNrDeferredDlss.value_or_default()))
    {
        const auto feature = State::Instance().currentFeature;
        ImGui::BeginDisabled(State::Instance().swapchainApi == API::Vulkan ||
                             (feature && feature->GetUpscalerType() == Upscaler::DLSSD));
        Checkbox("Match HDR brightness response (experimental)", config->DlssNrHdrTransfer);
        ImGui::EndDisabled();
        HelpMarker(
            "Match early NR brightness changes to the finished HDR image. Adds GPU work; unreliable fits fall back.");
    }
    Slider("Detail strength", config->DlssNrTransferStrength, 0.0f, 2.0f, "%.2f", 1.0f);
    HelpMarker("0 = no detail change. 1 = normal.");

    Slider("Colour strength", config->DlssNrColourStrength, 0.0f, 4.0f, "%.2f", 1.0f);
    HelpMarker("0 = game colours. 1 = model colours. Above 1 boosts saturation.");

    if (ImGui::TreeNode("Skin and environment (final edit)"))
    {
        Checkbox("Separate skin / environment controls", config->DlssNrSkinProtection);
        ImGui::BeginDisabled(!config->DlssNrSkinProtection.value_or_default());
        const auto slider = [](const char* label, auto& option)
        {
            Slider(label, option, 0.0f, 1.0f);
            HelpMarker("0 = unchanged. 1 = full effect.");
        };
        slider("Skin detail / lighting", config->DlssNrSkinDetail);
        slider("Skin colour", config->DlssNrSkinColour);
        slider("Environment detail / lighting", config->DlssNrEnvironmentDetail);
        slider("Environment colour", config->DlssNrEnvironmentColour);
        Checkbox("Preview colour-based mask", config->DlssNrShowSkinMask);
        ImGui::EndDisabled();
        ImGui::TreePop();
    }

    Slider("Highlight guard", config->DlssNrMaxRatio, 1.0f, 8.0f, "%.1fx", 2.0f);
    HelpMarker("Limit pixel brightening and darkening.");
}

void RenderInspect(Config* config)
{
    const auto placement = DlssNr::ResolvePlacement(
        config->DlssNrRunBeforeSr.value_or_default(), config->DlssNrDeferredDlss.value_or_default(),
        config->DlssNrResidualAcrossRr.value_or_default(), config->DlssNrFinishedPicture.value_or_default());
    if (placement.deferred && (config->DlssNrCompare.value_or_default() || config->DlssNrDebugView.value_or_default() ||
                               config->DlssNrShowSkinMask.value_or_default()))
        ImGui::TextWrapped("Compare, debug view and skin-mask inspection suspend the separate edit-upscale path.");
    Checkbox("Hold frame", config->DlssNrHoldFrame);
    HelpMarker(
        "Freeze a frame for NR tuning. Later game effects may update; temporal behaviour is not representative.");

    static const char* compareNames[] = { "Off", "Side by side", "Wipe" };
    int compare = (int) config->DlssNrCompare.value_or_default();
    if (ImGui::Combo("Compare", &compare, compareNames, IM_ARRAYSIZE(compareNames)))
        config->DlssNrCompare = (uint32_t) compare;

    HelpMarker("Compare the NR output with the original picture, or with another NR model.");

    if (compare != 0)
    {
        // What lies beside the NR picture: the original picture, or one pass of another model.
        const uint32_t other = config->DlssNrCompareModel.value_or_default();
        if (ImGui::BeginCombo("Other side", other == 0 ? "Original picture" : DlssNr::Nvfp4::ModelName(other)))
        {
            if (ImGui::Selectable("Original picture", other == 0))
                config->DlssNrCompareModel = 0u;
            for (const auto& model : DlssNr::Nvfp4::kModels)
                if (model.preset != 0 && ImGui::Selectable(model.name, other == model.preset))
                    config->DlssNrCompareModel = (uint32_t) model.preset;
            ImGui::EndCombo();
        }
        HelpMarker("Compare two models directly: the NR picture on one side, and on the other the same frame\n"
                   "through ONE pass of the model chosen here, with pass 1's style and strengths.\n"
                   "Both models are loaded and run every frame, so the frame costs more while this is on.\n"
                   "Not available with a debug view, peripheral compression, pass borrowing, supersampling or\n"
                   "DLSS enlargement.");
        if (other != 0)
        {
            if (DlssNr::Nvfp4::ComparisonShown() == other)
                ImGui::TextColored(ImVec4(0.45f, 0.9f, 0.45f, 1.0f), "Comparing with %s",
                                   DlssNr::Nvfp4::ModelName(other));
            else if (const auto note = DlssNr::Nvfp4::ComparisonNote(); !note.empty())
            {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.35f, 0.3f, 1.0f));
                ImGui::TextWrapped("The other model is NOT shown (that side is the original picture): %s",
                                   note.c_str());
                ImGui::PopStyleColor();
            }
        }

        Checkbox("Swap sides", config->DlssNrCompareSwap);
        Checkbox("Label the sides", config->DlssNrCompareTags);
        if (config->DlssNrCompareTags.value_or_default())
            Slider("Label size", config->DlssNrTagScale, 0.5f, 5.0f, "%.1fx", {}, ImGuiSliderFlags_AlwaysClamp);
    }

    if (compare == 1)
    {
        Slider("Zoom", config->DlssNrCompareZoom, 1.0f, 2.0f, "%.2f", {}, ImGuiSliderFlags_AlwaysClamp);
        HelpMarker("1 = fit. 2 = crop and enlarge.");
    }

    if (compare == 2)
    {
        Slider("Split", config->DlssNrCompareSplit, 0.0f, 1.0f, "%.2f", {}, ImGuiSliderFlags_AlwaysClamp);
        HelpMarker("Move the comparison boundary.");
    }

    static const char* debugNames[] = { "Off", "Proxy (what the model sees)", "Model output (raw)",
                                        "Difference (amplified)", "Compressed model input" };
    int debugView = (int) config->DlssNrDebugView.value_or_default();
    if (ImGui::Combo("Debug view", &debugView, debugNames, IM_ARRAYSIZE(debugNames)))
        config->DlssNrDebugView = (uint32_t) debugView;

    HelpMarker("Difference is amplified 20x. Grey means unchanged. Proxy and raw model output use unpacked geometry. "
               "Compressed model input shows the input before unpacking, scaled to fill the screen.");
}
} // namespace DlssNr::MenuSections
