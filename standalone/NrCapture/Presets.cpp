#include "Presets.h"
#include <algorithm>
namespace nr {
namespace {
std::wstring Trimmed(std::wstring name) {
    while (!name.empty() && (name.back() == L' ' || name.back() == L'.')) name.pop_back();
    while (!name.empty() && name.front() == L' ') name.erase(name.begin());
    return name;
}
std::filesystem::path File(const std::wstring& name) { return PresetDirectory() / (Trimmed(name) + L".ini"); }
void Require(const std::wstring& name) {
    if (const auto error = PresetNameError(name); !error.empty()) throw std::runtime_error(Narrow(error));
}
}
std::filesystem::path PresetDirectory() {
    const auto path = DataDirectory() / L"presets";
    std::filesystem::create_directories(path);
    return path;
}
std::vector<std::wstring> ListPresets() {
    std::vector<std::wstring> names;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(PresetDirectory(), error))
        if (entry.is_regular_file(error) && !_wcsicmp(entry.path().extension().c_str(), L".ini"))
            names.push_back(entry.path().stem().wstring());
    std::sort(names.begin(), names.end(), [](const std::wstring& a, const std::wstring& b) { return _wcsicmp(a.c_str(), b.c_str()) < 0; });
    return names;
}
std::wstring PresetNameError(const std::wstring& raw) {
    const auto name = Trimmed(raw);
    if (name.empty()) return L"Type a name for the preset.";
    if (name.size() > 64) return L"A preset name can have 64 characters at most.";
    for (const wchar_t c : name)
        if (c < 32 || wcschr(L"\\/:*?\"<>|", c)) return L"A preset name cannot contain \\ / : * ? \" < > |";
    // Names Windows reserves for devices cannot be file names.
    for (const wchar_t* reserved : {L"CON", L"PRN", L"AUX", L"NUL", L"COM1", L"COM2", L"COM3", L"COM4", L"COM5", L"COM6", L"COM7",
         L"COM8", L"COM9", L"LPT1", L"LPT2", L"LPT3", L"LPT4", L"LPT5", L"LPT6", L"LPT7", L"LPT8", L"LPT9"})
        if (!_wcsicmp(name.c_str(), reserved)) return L"Windows reserves that name; choose another.";
    return {};
}
bool PresetExists(const std::wstring& name) {
    std::error_code error;
    return PresetNameError(name).empty() && std::filesystem::is_regular_file(File(name), error);
}
void SavePreset(const std::wstring& name, const Settings& settings) {
    Require(name);
    const auto file = File(name);
    auto saved = settings; saved.runtime.clear(); saved.runtimeChosen.clear(); // (the runtime's path is the machine's, not the preset's)
    SaveSettings(saved, file);
    if (!PresetExists(name)) throw std::runtime_error("The preset could not be written to " + Narrow(file.wstring()));
}
Settings LoadPreset(const std::wstring& name, const Settings& current) {
    Require(name);
    if (!PresetExists(name)) throw std::runtime_error("There is no preset named \"" + Narrow(Trimmed(name)) + "\".");
    auto loaded = LoadSettings(File(name));
    loaded.runtime = current.runtime; loaded.runtimeChosen = current.runtimeChosen;
    loaded.captureBackend = current.captureBackend; loaded.frameGeneration = current.frameGeneration;
    loaded.presentMode = current.presentMode; loaded.latencyMode = current.latencyMode; // (how frames reach the screen, not a look)
    loaded.enabled = current.enabled;
    loaded.holdFrame = current.holdFrame; loaded.applyModel = current.applyModel; loaded.debugView = current.debugView;
    loaded.compare = current.compare; loaded.compareSplit = current.compareSplit; loaded.compareZoom = current.compareZoom;
    loaded.compareSwap = current.compareSwap; loaded.compareLabels = current.compareLabels; loaded.labelScale = current.labelScale;
    loaded.previewSkin = current.previewSkin;
    loaded.showFoveation = current.showFoveation; loaded.foveationPreviewPass = current.foveationPreviewPass;
    return loaded;
}
void RenamePreset(const std::wstring& from, const std::wstring& to) {
    Require(from); Require(to);
    if (!PresetExists(from)) throw std::runtime_error("There is no preset named \"" + Narrow(Trimmed(from)) + "\".");
    // A change of letter case only is the same file to Windows: that rename is allowed.
    if (PresetExists(to) && _wcsicmp(Trimmed(from).c_str(), Trimmed(to).c_str()))
        throw std::runtime_error("A preset named \"" + Narrow(Trimmed(to)) + "\" exists already.");
    std::error_code error;
    std::filesystem::rename(File(from), File(to), error);
    if (error) throw std::runtime_error("The preset could not be renamed: " + error.message());
}
void DeletePreset(const std::wstring& name) {
    Require(name);
    std::error_code error;
    if (!std::filesystem::remove(File(name), error) || error)
        throw std::runtime_error("The preset \"" + Narrow(Trimmed(name)) + "\" could not be deleted.");
}
}
