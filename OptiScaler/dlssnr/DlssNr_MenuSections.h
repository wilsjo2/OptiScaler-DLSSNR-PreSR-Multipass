#pragma once

#include <imgui/imgui.h>

class Config;

namespace DlssNr::MenuSections
{
// Compact contextual help, matching the rest of the menu.
inline void HelpMarker(const char* tip)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");

    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 40.0f);
        ImGui::TextUnformatted(tip);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

// The pass chosen in the pass tabs (0 = pass 1): the per-pass controls edit it, the calibrator adjusts it.
inline unsigned& EditPass()
{
    static unsigned pass = 0;
    return pass;
}

// Always shown: passes, each pass's model and tuning, the strengths of the final edit, skin, resolution.
void RenderLook(Config* config);
// Collapsed sections.
void RenderHdr(Config* config);
void RenderFixes(Config* config);
void RenderInspect(Config* config);
void RenderPresets(Config* config);
// One row at the top: choose a saved preset and load it.
void RenderPresetBar(Config* config);
} // namespace DlssNr::MenuSections
