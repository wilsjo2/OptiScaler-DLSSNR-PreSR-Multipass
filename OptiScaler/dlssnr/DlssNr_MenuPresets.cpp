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

void RenderPresets(Config* config)
{
    static std::vector<std::string> names = Presets::List();
    static std::string selected, goal, message;
    static char typed[65] {};
    static double listedAt = 0;
    // Files may be added or removed outside the menu: look again every two seconds while this section is open.
    if (const double now = ImGui::GetTime(); now - listedAt > 2.0)
    {
        names = Presets::List();
        listedAt = now;
    }
    const auto relist = [&](const std::string& choose)
    {
        names = Presets::List();
        if (!choose.empty())
            selected = choose;
    };
    const auto done = [&](const std::string& problem, std::string success)
    {
        message = problem.empty() ? std::move(success) : problem;
        return problem.empty();
    };
    const bool calibrating = Calibrate::Read().active;

    ImGui::SeparatorText("Presets");
    PresetCombo("Preset", names, selected);
    HelpMarker("A preset is the NR settings saved under a name: where NR runs, the passes and their models, resolution, "
               "composition, halo suppression and white point. It leaves alone Enable, the toggle key, Apply model, "
               "Hold frame, Compare and the debug views. Files: the nr-presets folder beside OptiScaler.");
    ImGui::BeginDisabled(selected.empty() || calibrating);
    if (ImGui::Button("Load"))
        done(Presets::Load(selected, *config), std::format("Loaded preset \"{}\".", selected));
    ImGui::SameLine();
    if (ImGui::Button("Overwrite"))
        done(Presets::Save(selected, *config), std::format("Preset \"{}\" now holds the current settings.", selected));
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
            if (done(Presets::Delete(gone), std::format("Deleted preset \"{}\".", gone)))
                relist({});
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Keep it"))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::InputText("Name", typed, sizeof(typed));
    ImGui::BeginDisabled(typed[0] == '\0' || calibrating);
    if (ImGui::Button("Save as new preset"))
    {
        const std::string name = Presets::Cleaned(typed);
        if (Presets::Exists(name))
            message = "A preset with that name exists already. Select it and use Overwrite, or type another name.";
        else if (done(Presets::Save(name, *config), std::format("Saved the current settings as preset \"{}\".", name)))
        {
            relist(name);
            typed[0] = '\0';
        }
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(selected.empty());
    if (ImGui::Button("Rename selected"))
    {
        const std::string name = Presets::Cleaned(typed), old = selected;
        if (done(Presets::Rename(old, name), std::format("Renamed preset \"{}\" to \"{}\".", old, name)))
        {
            relist(name);
            if (goal == old)
                goal = name;
            typed[0] = '\0';
        }
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (!message.empty())
        ImGui::TextWrapped("%s", message.c_str());

    ImGui::SeparatorText("Calibrate the edit pass to a preset");
    PresetCombo("Preset to match", names, goal);
    HelpMarker("Holds the current frame and renders the preset to match on it (all of its passes). Then it tries values "
               "of Intensity, Local structure and Local tone of the pass chosen under \"Edit pass\" (Model section), and "
               "of Detail strength and Colour strength, until NR's picture is as close to that preset's as it gets, "
               "and leaves the best values in the sliders. Other passes that are enabled keep running as they are "
               "set. Takes about a minute; the picture changes while it runs. Direct3D 12 only.");
    const unsigned pass = EditPass();
    ImGui::Text("Edit pass: pass %u", pass + 1);
    static bool skin = false;
    ImGui::BeginDisabled(calibrating);
    ImGui::Checkbox("Enable skin settings optimisation", &skin);
    ImGui::EndDisabled();
    HelpMarker("Also tries Skin detail, Skin colour, Environment detail and Environment colour (Composition; "
               "\"Separate skin / environment controls\" is switched on) and Highlight guard.");
    const bool dx12 = ReadStatus(Backend::Dx12).running;
    if (!calibrating)
    {
        ImGui::BeginDisabled(goal.empty() || !dx12);
        if (ImGui::Button("Calibrate"))
            Calibrate::Start(goal, pass, skin);
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
