#pragma once
#include "Common.h"
#include <vector>
namespace nr {
// Named presets: the settings saved under a name, one file each (<data folder>/presets/<name>.ini, the layout of
// settings.ini). A preset holds the look and the model set-up. It does not hold what belongs to the machine or the
// session: the runtime's path, the capture backend, frame generation, "Enable NR", and the choices of the "Inspect and
// compare" tab; those stay as they are when a preset is loaded.
std::filesystem::path PresetDirectory();
std::vector<std::wstring> ListPresets(); // names in alphabetical order
// Why `name` cannot be a preset's name; empty when it can.
std::wstring PresetNameError(const std::wstring& name);
bool PresetExists(const std::wstring& name);
void SavePreset(const std::wstring& name, const Settings& settings);
Settings LoadPreset(const std::wstring& name, const Settings& current);
void RenamePreset(const std::wstring& from, const std::wstring& to);
void DeletePreset(const std::wstring& name);
}
