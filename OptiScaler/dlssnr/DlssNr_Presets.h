#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

class Config;

// Named presets of the Neural Rendering settings: one file each, <OptiScaler folder>/nr-presets/<name>.ini, with the
// option names of OptiScaler.ini's [DlssNr] section. A preset holds where NR runs, the passes and their models,
// resolution, composition, halo suppression, white point and the rest of the look. It does not hold what belongs to
// the session: Enable, the toggle key, Apply model, Hold frame, Compare and its options, the debug view and the
// preview overlays; those stay as they are when a preset is loaded.
namespace DlssNr::Presets
{
// Option name -> value as text. An option left at "auto" (unset) is absent.
using Values = std::map<std::string, std::string>;

Values Capture(Config& config);
// Every option a preset holds is set from `values`; one that is absent goes back to unset.
void Apply(Config& config, const Values& values);

std::filesystem::path Directory();
std::vector<std::string> List(); // names in alphabetical order
// Why `name` cannot be a preset's name; empty when it can.
std::string NameError(const std::string& name);
bool Exists(const std::string& name);
// The name as its file has it: without spaces at the ends.
std::string Cleaned(const std::string& name);

// Each returns an empty string, or what went wrong.
std::string Save(const std::string& name, Config& config);
std::string Read(const std::string& name, Values& values);
std::string Load(const std::string& name, Config& config);
std::string Rename(const std::string& from, const std::string& to);
std::string Delete(const std::string& name);
} // namespace DlssNr::Presets
