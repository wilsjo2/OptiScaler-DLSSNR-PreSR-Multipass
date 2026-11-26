#include "pch.h"

#include "DlssNr.h"
#include "DlssNrFeature_Vk.h"
#include "DlssNrFinished_Vk.h"
#include "DlssNr_Upscaler.h"
#include "DlssNr_MenuSections.h"
#include "DlssNr_Placement.h"
#include <Config.h>
#include <menu/menu_common.h>
#include <algorithm>
#include <cmath>
#include <format>

// The Neural Rendering menu. From the top: the two switches and what NR is doing, a row to load a saved preset, and
// the look (passes, model, tuning and strengths, all in one list). Everything set once per game or used now and then
// sits in collapsed sections below: where NR runs, HDR and exposure, artifact fixes, compare and inspect, presets
// and calibration.
namespace DlssNr::MenuSections
{
namespace
{
// Where NR runs is one choice. The settings behind it are four switches that depend on one another.
enum class Where
{
    AfterUpscale,
    BeforeUpscale,
    BeforeEditAfter,
    Finished,
    BeforeEditFinished,
    DenoiseFirst,
    Count
};
struct WhereInfo
{
    const char* name;
    const char* brief; // for the section's header
    const char* what;
};
constexpr WhereInfo kWhere[(int) Where::Count] = {
    { "After the upscaler", "after the upscaler", "NR edits the upscaled picture, before the game's effects and HUD." },
    { "Before the upscaler", "before the upscaler", "NR edits the game's render before SR or Ray Reconstruction upscales it." },
    { "Before the upscaler, edit added after it", "before the upscaler, edit added after",
      "The game upscales its own clean render. NR's edit is upscaled by a separate upscaler and added afterwards." },
    { "On the finished picture", "finished picture", "NR edits the final picture, after the game's effects and HUD." },
    { "Before the upscaler, edit added to the finished picture", "before the upscaler, edit added at the end",
      "NR runs on the game's render. Its edit is upscaled separately and added to the final picture, after effects "
      "and HUD." },
    { "Denoise first, then NR, then upscale", "denoise first",
      "The game's upscaler runs once more at render resolution, so NR is shown a clean, steady picture instead of the "
      "noisy one. Native DirectX 12 only; costs one extra upscaler pass per frame." },
};

Placement Resolved(Config* config)
{
    return ResolvePlacement(config->DlssNrRunBeforeSr.value_or_default(), config->DlssNrDeferredDlss.value_or_default(),
                            config->DlssNrResidualAcrossRr.value_or_default(),
                            config->DlssNrFinishedPicture.value_or_default());
}

Where CurrentWhere(Config* config)
{
    const auto placement = Resolved(config);
    if (placement.finished)
        return placement.deferred ? Where::BeforeEditFinished : Where::Finished;
    if (placement.deferred)
        return Where::BeforeEditAfter;
    // (Denoise first is ignored by the run while a separate edit or the finished picture is chosen.)
    if (config->DlssNrDenoiseFirst.value_or_default())
        return Where::DenoiseFirst;
    return placement.beforeUpscale ? Where::BeforeUpscale : Where::AfterUpscale;
}

void SetWhere(Config* config, Where where)
{
    const bool finishedBefore = config->DlssNrFinishedPicture.value_or_default();
    const bool finished = where == Where::Finished || where == Where::BeforeEditFinished;
    const bool deferred = where == Where::BeforeEditAfter || where == Where::BeforeEditFinished;
    config->DlssNrFinishedPicture = finished;
    config->DlssNrDeferredDlss = deferred;
    config->DlssNrResidualAcrossRr = false; // the old name of the separate-edit route
    config->DlssNrDenoiseFirst = where == Where::DenoiseFirst;
    // Denoise first decides the placement itself and leaves this switch as it is.
    if (where != Where::DenoiseFirst)
        config->DlssNrRunBeforeSr = deferred || where == Where::BeforeUpscale;
    if (finished != finishedBefore)
        DlssNr::RetryAfterFailure();
}

void RenderDenoiseFirst(Config* config)
{
    int step = std::clamp(config->DlssNrDenoiseFirstStep.value_or_default(), 0, 2);
    if (ImGui::Combo("Second step", &step,
                     "Game upscaler again, jitter zeroed\0Private DLSS SR, jitter zeroed\0Edit onto the raw "
                     "render, then the game upscaler\0"))
        config->DlssNrDenoiseFirstStep = step;
    HelpMarker("Game upscaler again: the game's upscaler receives the NR'd clean image. Simple, but the second pass "
               "gets no real sub-pixel samples, so it can look softer than native.\n\n"
               "Private DLSS SR: the same, but a separate plain DLSS pass does the upscale and replaces the output. "
               "Avoids running Ray Reconstruction twice.\n\n"
               "Edit onto the raw render: only NR's change is taken from the clean image, lined up with the raw "
               "render's jitter and added onto it. The game's upscaler then sees real samples carrying the edit. "
               "Sharpest in theory; Ray Reconstruction may soften part of the edit.");
    if (step == 2)
    {
        int edit = config->DlssNrDenoiseFirstEdit.value_or_default() == 1 ? 1 : 0;
        if (ImGui::Combo("Edit as", &edit, "Difference (added)\0Ratio (multiplied)\0"))
            config->DlssNrDenoiseFirstEdit = edit;
        HelpMarker("Ratio multiplies each raw pixel by NR / clean, so every noisy sample keeps its own hue "
                   "and only its brightness and tint move. Recommended.\nDifference adds NR - clean as an "
                   "absolute amount. On samples that came back dark that amount outweighs the sample, which "
                   "recolours them and roughly doubles the colour noise the upscaler receives.");
        if (edit == 1)
        {
            bool fireflies = config->DlssNrDenoiseFirstFireflyGuard.value_or_default();
            if (ImGui::Checkbox("Do not multiply fireflies", &fireflies))
                config->DlssNrDenoiseFirstFireflyGuard = fireflies;
            HelpMarker("A raw sample far brighter than the clean value is a path-tracing firefly, not a lit "
                       "pixel. Multiplying it by the ratio makes it brighter, and when the camera is still "
                       "the game's Ray Reconstruction can keep such sparkles in shadows. With this on, those "
                       "samples get the absolute change their pixel would have had instead. Turn off to "
                       "compare.");
        }

        bool shift = config->DlssNrDenoiseFirstShift.value_or_default();
        if (ImGui::Checkbox("Shift the edit by the frame's jitter", &shift))
            config->DlssNrDenoiseFirstShift = shift;
        HelpMarker("The clean image sits on a steady pixel grid; the raw render is offset by a sub-pixel "
                   "jitter that changes every frame. Shifting the edit by that jitter lines the two up. "
                   "Turn off only to compare.");
        if (shift)
        {
            bool flip = config->DlssNrDenoiseFirstFlipJitter.value_or_default();
            if (ImGui::Checkbox("Flip the jitter direction", &flip))
                config->DlssNrDenoiseFirstFlipJitter = flip;
            HelpMarker("Engines disagree on which way the jitter value points. The wrong direction doubles "
                       "the misalignment instead of removing it. If edges look worse with the shift on than "
                       "off, flip this.");
            int kernel = std::clamp(config->DlssNrDenoiseFirstKernel.value_or_default(), 0, 2);
            if (ImGui::Combo("Edit resampling", &kernel, "Bilinear\0Catmull-Rom\0Lanczos 2\0"))
                config->DlssNrDenoiseFirstKernel = kernel;
            HelpMarker("How the edit is interpolated at the shifted position. Bilinear is soft and never "
                       "overshoots. Catmull-Rom is sharp with little overshoot. Lanczos 2 is the sharpest "
                       "and can ring around hard edges.");
            bool clampEdit = config->DlssNrDenoiseFirstNeighbourhoodClamp.value_or_default();
            if (ImGui::Checkbox("Clamp the edit to its neighbours", &clampEdit))
                config->DlssNrDenoiseFirstNeighbourhoodClamp = clampEdit;
            HelpMarker("Limits the interpolated edit to the range of the four nearest source pixels. Removes "
                       "ringing from the sharper kernels at the cost of a little sharpness.");
        }
    }
    ImGui::TextWrapped("Denoise first: %s", DlssNr::DenoiseFirstStatus().c_str());
}

void RenderPlacement(Config* config, Where where)
{
    int chosen = (int) where;
    if (ImGui::BeginCombo("NR runs", kWhere[chosen].name))
    {
        for (int i = 0; i < (int) Where::Count; ++i)
        {
            if (ImGui::Selectable(kWhere[i].name, i == chosen) && i != chosen)
                SetWhere(config, (Where) i);
            if (ImGui::IsItemHovered())
            {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
                ImGui::TextUnformatted(kWhere[i].what);
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            if (i == chosen)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    where = CurrentWhere(config);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", kWhere[(int) where].what);
    ImGui::PopStyleColor();

    const bool enabled = config->DlssNrEnabled.value_or_default();
    const auto placement = Resolved(config);
    const auto feature = State::Instance().currentFeature;
    if (where == Where::DenoiseFirst)
        RenderDenoiseFirst(config);

    const bool nativeVk = feature && feature->Api() == API::Vulkan && !feature->IsWithDx12();
    if (placement.deferred && !nativeVk)
    {
        int backend = (int) GetPrivateUpscaler(config->DlssNrPrivateUpscaler.value_or_default());
        if (ImGui::Combo("Upscaler for the edit", &backend, "DLSS\0FSR 2.2\0FSR (FidelityFX)\0XeSS\0"))
            config->DlssNrPrivateUpscaler = backend;
        HelpMarker("Upscales only NR's edit, with or without Ray Reconstruction in the game. FSR (FidelityFX) and "
                   "XeSS need their runtimes.");
    }
    if (placement.finished && enabled)
    {
        if (State::Instance().swapchainApi == API::Vulkan)
            ImGui::TextWrapped("%s", DlssNr::FinishedVkStatus().c_str());
        else if (feature && (feature->Api() != API::DX12 ||
                             (feature->IsWithDx12() && State::Instance().swapchainApi != API::DX11 &&
                              State::Instance().swapchainInteropApi != SwapchainInteropApi::Dx11wDx12)))
            ImGui::TextWrapped("This option needs DirectX 12 or a DirectX 11 upscaler marked w/Dx12.");
        else
            ImGui::TextWrapped("%s", DlssNr::FinishedPictureStatus().c_str());
    }
    if (placement.deferred)
        ImGui::TextWrapped("Separate upscale: %s", DlssNr::DeferredDlssStatus().c_str());
}

// What NR is doing, in a line or two.
void RenderStatus(Config* config)
{
    const bool enabled = config->DlssNrEnabled.value_or_default();
    const auto dx12 = ReadStatus(Backend::Dx12);
    const auto vk = ReadStatus(Backend::Vulkan);
    const bool vulkan = vk.running;

    // An existing model handle does not mean NR is enabled this frame.
    if (!enabled)
    {
        ImGui::TextDisabled("NR off.");
        return;
    }
    if (!dx12.running && !vulkan)
    {
        const auto feature = State::Instance().currentFeature;
        const bool nativeVk = feature && feature->Api() == API::Vulkan && !feature->IsWithDx12();
        const auto& reason = nativeVk ? vk.failureReason : dx12.failureReason;

        if (!reason.empty())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.35f, 0.3f, 1.0f));
            ImGui::TextWrapped("%s", reason.c_str());
            ImGui::PopStyleColor();
            if (nativeVk)
                ImGui::TextUnformatted("Restart the game to retry native Vulkan NR.");
            else if (ImGui::SmallButton("Retry"))
                DlssNr::RetryAfterFailure();
        }
        else if (feature && feature->Api() == API::DX11 && !feature->IsWithDx12())
            ImGui::TextWrapped("NR needs the D3D12 bridge on D3D11. Choose an upscaler marked w/Dx12 and restart.");
        else if (nativeVk && Resolved(config).deferred)
            ImGui::TextWrapped("A separately upscaled edit needs DirectX 12 or its bridge. Choose another place for NR "
                               "to run (under \"Where NR runs\") to use native Vulkan NR.");
        else
            ImGui::TextDisabled("Waiting for the upscaler to run.");
        return;
    }

    const auto ms = vulkan ? vk.gpuTime : dx12.gpuTime;
    const auto& state = State::Instance();
    // The FG swapchain interval is between real game frames, not interpolated presents.
    // Native Vulkan has no DXGI timing, so use the existing overlay frame interval there.
    const double frameMs = state.swapchainApi == API::Vulkan ? (state.frameTimes.empty() ? 0.0 : state.frameTimes.back())
                           : state.currentFG                 ? state.lastFGFrameTime
                                                             : state.presentFrameTime;
    const bool timed = ms.has_value() && std::isfinite(ms.value()) && ms.value() >= 0.0;
    const bool framed = timed && std::isfinite(frameMs) && frameMs > 0.0;
    std::string text = vulkan ? "Running natively on Vulkan" : "Running";
    if (framed)
        text += std::format(": NR {:.2f} ms of a {:.2f} ms frame", ms.value(), frameMs);
    else if (timed)
        text += std::format(": NR {:.2f} ms", ms.value());
    // Hiding the edit keeps model evaluation running.
    if (!config->DlssNrApplyModel.value_or_default())
        text += " (edit hidden, model still runs)";

    // Keep the running indicator green, using the theme's HDR-adjusted text brightness.
    const auto textColor = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    ImGui::PushStyleColor(ImGuiCol_Text,
                          ImVec4(textColor.x * 0.55f, textColor.y * 0.80f, textColor.z * 0.55f, textColor.w));
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Time between the start and end of NR on the GPU, including delays while other work "
                          "runs,\nagainst the time between rendered frames. The two can overlap: compare FPS to "
                          "check the effect on game performance.");
    if (framed)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              ImVec4(textColor.x * 0.20f, textColor.y * 0.55f, textColor.z * 0.25f, textColor.w));
        ImGui::ProgressBar(float(ms.value() / std::max(frameMs, ms.value())),
                           ImVec2(-1.0f, std::max(ImGui::GetFontSize() * 0.3f, 3.0f)), "");
        ImGui::PopStyleColor();
    }
}

} // namespace
} // namespace DlssNr::MenuSections

namespace DlssNr
{

void RenderMenu(Config* config, float menuResScale)
{
    using namespace MenuSections;
    ImGui::Spacing();
    if (auto header = ScopedCollapsingHeader("DLSS Neural Rendering"); header.IsHeaderOpen())
    {
        ScopedIndent indent {};
        bool enabled = config->DlssNrEnabled.value_or_default();
        if (ImGui::Checkbox("Enable Neural Rendering", &enabled))
            config->DlssNrEnabled = enabled;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Run NR. Off stops its GPU cost.");
        ImGui::SameLine();
        bool applyModel = config->DlssNrApplyModel.value_or_default();
        if (ImGui::Checkbox("Apply model", &applyModel))
            config->DlssNrApplyModel = applyModel;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Show or hide the NR effect, to compare by eye. The model still runs when hidden.");
        RenderStatus(config);

        const float itemWidth = std::min(220.0f * menuResScale, ImGui::GetContentRegionAvail().x * 0.42f);
        ImGui::PushItemWidth(itemWidth);
        RenderPresetBar(config);

        ImGui::SeparatorText("Look");
        RenderLook(config);
        ImGui::Spacing();

        // The header names the place in use, so it need not be opened to see it.
        const auto where = CurrentWhere(config);
        if (ImGui::CollapsingHeader(std::format("Where NR runs: {}###where", kWhere[(int) where].brief).c_str()))
            RenderPlacement(config, where);
        if (ImGui::CollapsingHeader("HDR and exposure"))
            RenderHdr(config);
        if (ImGui::CollapsingHeader("Artifact fixes (experimental)"))
            RenderFixes(config);
        if (ImGui::CollapsingHeader("Compare and inspect"))
            RenderInspect(config);
        if (ImGui::CollapsingHeader("Presets and calibration"))
            RenderPresets(config);
        ImGui::PopItemWidth();
    }
}

} // namespace DlssNr
