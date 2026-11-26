#include "pch.h"
#include "DlssNr_Calibrate.h"
#include "DlssNr_MenuSections.h"
#include "DlssNr_Presets.h"
#include "DlssNr_Status.h"
#include <Config.h>
#include <State.h>
#include <menu/menu_common.h>
#include <algorithm>
#include <format>
#include <string>
#include <vector>

namespace DlssNr::MenuSections
{
namespace
{
// Shared by the row at the top and the section at the bottom: the saved presets, the one chosen, and what the last
// action said.
struct PresetState
{
    std::vector<std::string> names = Presets::List();
    std::string selected, goal, message;
    double listedAt = 0, saidAt = -100;
    char typed[65] {};
};
PresetState& Shared()
{
    static PresetState state;
    return state;
}

// Files may be added or removed outside the menu: look again every two seconds while the menu shows them.
void Refresh(PresetState& state)
{
    if (const double now = ImGui::GetTime(); now - state.listedAt > 2.0)
    {
        state.names = Presets::List();
        state.listedAt = now;
    }
}

// Keeps the action's message; true when it succeeded.
bool Done(PresetState& state, const std::string& problem, std::string success)
{
    state.message = problem.empty() ? std::move(success) : problem;
    state.saidAt = ImGui::GetTime();
    return problem.empty();
}

// A list of the presets with one chosen; the choice follows its name when the list changes.
bool PresetCombo(const char* label, const std::vector<std::string>& names, std::string& chosen)
{
    if (std::ranges::find(names, chosen) == names.end())
        chosen = names.empty() ? std::string {} : names.front();
    bool changed = false;
    if (ImGui::BeginCombo(label, chosen.empty() ? "(none saved)" : chosen.c_str()))
    {
        for (const auto& name : names)
        {
            if (ImGui::Selectable(name.c_str(), name == chosen))
            {
                chosen = name;
                changed = true;
            }
            if (name == chosen)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}
} // namespace

void RenderPresetBar(Config* config)
{
    auto& state = Shared();
    Refresh(state);
    const bool calibrating = Calibrate::Read().active;
    ImGui::PushID("preset bar");
    PresetCombo("Preset", state.names, state.selected);
    ImGui::SameLine();
    ImGui::BeginDisabled(state.selected.empty() || calibrating);
    if (ImGui::Button("Load"))
        Done(state, Presets::Load(state.selected, *config), std::format("Loaded preset \"{}\".", state.selected));
    ImGui::EndDisabled();
    HelpMarker("A preset is the NR settings saved under a name: where NR runs, the passes and their models, resolution, "
               "strengths, skin, fixes and white point. It leaves alone Enable, the toggle key, Apply model, Hold "
               "frame, Compare and the debug views. Save, rename and delete presets under \"Presets and "
               "calibration\" below.");
    ImGui::PopID();
    // What the last preset action said, for a few seconds (it stays in the section below).
    if (!state.message.empty() && ImGui::GetTime() - state.saidAt < 6.0)
        ImGui::TextWrapped("%s", state.message.c_str());
    if (const auto report = Calibrate::Read(); report.active)
        ImGui::TextWrapped("%s", report.text.c_str());
}

void RenderPresets(Config* config)
{
    auto& state = Shared();
    Refresh(state);
    auto& names = state.names;
    auto& selected = state.selected;
    const auto relist = [&](const std::string& choose)
    {
        names = Presets::List();
        if (!choose.empty())
            selected = choose;
    };
    const bool calibrating = Calibrate::Read().active;

    ImGui::SeparatorText("Save and manage");
    PresetCombo("Preset", names, selected);
    HelpMarker("The same choice as the row at the top. Files: the nr-presets folder beside OptiScaler.");
    ImGui::BeginDisabled(selected.empty() || calibrating);
    if (ImGui::Button("Load"))
        Done(state, Presets::Load(selected, *config), std::format("Loaded preset \"{}\".", selected));
    ImGui::SameLine();
    if (ImGui::Button("Overwrite"))
        Done(state, Presets::Save(selected, *config),
             std::format("Preset \"{}\" now holds the current settings.", selected));
    ImGui::SameLine();
    if (ImGui::Button("Delete"))
        ImGui::OpenPopup("Delete NR preset");
    ImGui::EndDisabled();
    if (ImGui::BeginPopup("Delete NR preset"))
    {
        ImGui::Text("Delete the preset \"%s\"?", selected.c_str());
        if (ImGui::Button("Delete it"))
        {
            const auto gone = selected;
            if (Done(state, Presets::Delete(gone), std::format("Deleted preset \"{}\".", gone)))
                relist({});
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Keep it"))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::InputText("Name", state.typed, sizeof(state.typed));
    ImGui::BeginDisabled(state.typed[0] == '\0' || calibrating);
    if (ImGui::Button("Save as new preset"))
    {
        const std::string name = Presets::Cleaned(state.typed);
        if (Presets::Exists(name))
            Done(state, "A preset with that name exists already. Select it and use Overwrite, or type another name.",
                 {});
        else if (Done(state, Presets::Save(name, *config),
                      std::format("Saved the current settings as preset \"{}\".", name)))
        {
            relist(name);
            state.typed[0] = '\0';
        }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(selected.empty());
    if (ImGui::Button("Rename selected"))
    {
        const std::string name = Presets::Cleaned(state.typed), old = selected;
        if (Done(state, Presets::Rename(old, name), std::format("Renamed preset \"{}\" to \"{}\".", old, name)))
        {
            relist(name);
            if (state.goal == old)
                state.goal = name;
            state.typed[0] = '\0';
        }
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (!state.message.empty())
        ImGui::TextWrapped("%s", state.message.c_str());

    ImGui::SeparatorText("Calibrate to a preset");
    PresetCombo("Preset to match", names, state.goal);
    const unsigned pass = EditPass();
    HelpMarker("Makes the current settings look like a saved preset, for example one pass of a pass-in-one model like "
               "a two-pass preset.\n\n"
               "It holds the current frame and renders the preset to match on it (all of its passes). Then it tries "
               "values of Intensity, Local structure and Local tone of the pass chosen in the pass tabs, and of Detail "
               "strength and Colour strength, until NR's picture is as close to that preset's as it gets, and leaves "
               "the best values in the sliders. Other passes that are enabled keep running as they are set.\n\n"
               "Takes a minute or two; the picture changes while it runs. Direct3D 12 only.");
    if (std::max(config->DlssNrPasses.value_or_default(), 1u) > 1)
        ImGui::Text("Adjusts pass %u (chosen in the pass tabs).", pass + 1);
    static bool skin = false;
    ImGui::BeginDisabled(calibrating);
    ImGui::Checkbox("Also set skin, environment and highlight guard", &skin);
    ImGui::EndDisabled();
    HelpMarker("Also tries Skin detail, Skin colour, Environment detail and Environment colour (\"Separate skin / "
               "environment controls\" is switched on) and Highlight guard.\n\n"
               "Skin and everything else are then compared separately, so a face counts as much as the rest of the "
               "picture however small it is. Hold a frame with a face in view: without skin in the frame the four "
               "sliders are left as they are.");
    const bool dx12 = ReadStatus(Backend::Dx12).running;
    if (!calibrating)
    {
        ImGui::BeginDisabled(state.goal.empty() || !dx12);
        if (ImGui::Button("Calibrate"))
            Calibrate::Start(state.goal, pass, skin);
        ImGui::EndDisabled();
        if (!dx12)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("NR must be running with Direct3D 12.");
        }
    }
    else if (ImGui::Button("Cancel calibration"))
        Calibrate::Cancel();
    if (const auto report = Calibrate::Read(); !report.text.empty())
        ImGui::TextWrapped("%s", report.text.c_str());
}
} // namespace DlssNr::MenuSections
