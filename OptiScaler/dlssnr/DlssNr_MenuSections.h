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

void RenderInput(Config* config);
void RenderModel(Config* config);
void RenderBlend(Config* config);
void RenderInspect(Config* config);
} // namespace DlssNr::MenuSections
