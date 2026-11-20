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

// The pass chosen under "Edit pass" (0 = pass 1): the Model section edits it, the calibrator adjusts it.
inline unsigned& EditPass()
{
    static unsigned pass = 0;
    return pass;
}

void RenderInput(Config* config);
void RenderModel(Config* config);
void RenderBlend(Config* config);
void RenderInspect(Config* config);
void RenderPresets(Config* config);
} // namespace DlssNr::MenuSections
