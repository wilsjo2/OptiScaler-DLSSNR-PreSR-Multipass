#include "pch.h"

#include "DlssNr_MenuOverlay.h"
#include "DlssNr_Status.h"
#include <shaders/dlssnr/DlssNr_Spatial.h>
#include <Config.h>
#include <imgui/imgui.h>
#include <algorithm>
#include <cfloat>

namespace DlssNr
{
static void RenderSpatialOutlines()
{
    const auto& config = *Config::Instance();
    if (!config.DlssNrEnabled.value_or_default() || !config.DlssNrSpatialCompression.value_or_default() ||
        (!config.DlssNrSpatialShowCenter.value_or_default() && !config.DlssNrSpatialShowWork.value_or_default()))
        return;
    const auto feature = State::Instance().currentFeature;
    const bool nativeVk = feature && feature->Api() == API::Vulkan && !feature->IsWithDx12();
    if (!ReadStatus(nativeVk ? Backend::Vulkan : Backend::Dx12).spatialActive)
        return;
    const auto screen = ImGui::GetIO().DisplaySize;
    if (screen.x < 2 || screen.y < 2)
        return;
    const auto layout = Spatial::Build(Spatial::ReadSettings(config), static_cast<unsigned>(screen.x),
                                       static_cast<unsigned>(screen.y), config.DlssNrWorkingScale.value_or_default());
    if (!layout.active)
        return;
    auto* draw = ImGui::GetForegroundDrawList();
    const auto rectangle = [&](const Spatial::Rect& bounds, ImU32 color)
    {
        const ImVec2 lo { bounds.left * screen.x, bounds.top * screen.y };
        const ImVec2 hi { bounds.right * screen.x, bounds.bottom * screen.y };
        draw->AddRect(lo, hi, IM_COL32(0, 0, 0, 220), 0, 0, 4.0f);
        draw->AddRect(lo, hi, color, 0, 0, 2.0f);
    };
    if (config.DlssNrSpatialShowWork.value_or_default())
        rectangle(layout.workBounds, IM_COL32(255, 115, 0, 255));
    if (config.DlssNrSpatialShowCenter.value_or_default())
        rectangle(layout.centerBounds, IM_COL32(0, 210, 255, 255));
}

void RenderNrCompareTags()
{
    RenderSpatialOutlines();
    auto* config = Config::Instance();

    const uint32_t mode = config->DlssNrCompare.value_or_default();

    if (mode == 0 || !config->DlssNrCompareTags.value_or_default())
        return;

    const ImVec2 screen = ImGui::GetIO().DisplaySize;

    if (screen.x < 1.0f || screen.y < 1.0f)
        return;

    const bool swap = config->DlssNrCompareSwap.value_or_default();
    const float split = mode == 1 ? 0.5f : std::clamp(config->DlssNrCompareSplit.value_or_default(), 0.0f, 1.0f);
    const float splitX = split * screen.x;

    const float scale = std::clamp(config->DlssNrTagScale.value_or_default(), 0.5f, 5.0f);

    // The left side is the untouched frame unless swapped -- matching the shader's
    // showOriginal = (uv.x < split) != swap.
    const char* leftText = swap ? "DLSS NR : ON" : "DLSS NR : OFF";
    const char* rightText = swap ? "DLSS NR : OFF" : "DLSS NR : ON";

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImFont* font = ImGui::GetFont();
    const float fontSize = ImGui::GetFontSize() * scale;
    const float margin = 10.0f * scale;

    // Both labels flank the divider along the top: the left picture's label is right-aligned just
    // left of the split, the right picture's is left-aligned just right of it. Each is clipped to its
    // own side, so in the wipe the split reveals and hides them along with the images.
    auto drawTag = [&](const char* text, float x, ImVec2 clipMin, ImVec2 clipMax)
    {
        const ImVec2 size = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text);

        // Never let a label run off the visible frame as it grows.
        x = std::min(std::max(x, 0.0f), screen.x - size.x);
        float y = std::min(margin, screen.y - size.y - margin);
        y = std::max(y, 0.0f);

        dl->PushClipRect(clipMin, clipMax, true);
        dl->AddText(font, fontSize, ImVec2(x + 2.0f, y + 2.0f), IM_COL32(0, 0, 0, 210), text);
        dl->AddText(font, fontSize, ImVec2(x, y), IM_COL32(255, 255, 255, 255), text);
        dl->PopClipRect();
    };

    const ImVec2 leftSize = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, leftText);

    // Left picture's label: right edge a margin in from the split. Right picture's: left edge a margin
    // out from the split.
    drawTag(leftText, splitX - margin - leftSize.x, ImVec2(0.0f, 0.0f), ImVec2(splitX, screen.y));
    drawTag(rightText, splitX + margin, ImVec2(splitX, 0.0f), ImVec2(screen.x, screen.y));
}

} // namespace DlssNr
