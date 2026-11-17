#include "Common.h"
#include <shlobj.h>
#include <fstream>
#include <mutex>
#include <cmath>
namespace nr {
void Check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        auto message = std::format("{} failed: 0x{:08X}", operation, (unsigned)result);
        throw std::runtime_error(message);
    }
}
std::filesystem::path DataDirectory() {
    PWSTR raw = nullptr;
    Check(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &raw), "LocalAppData");
    std::filesystem::path path = std::filesystem::path(raw) / L"OptiScalerNR";
    CoTaskMemFree(raw);
    std::filesystem::create_directories(path);
    return path;
}
void AppendBoundedLog(const std::filesystem::path& file, const std::string& entry) {
    const auto line = entry.substr(0, MaxLogBytes - 1) + '\n';
    std::error_code error;
    const auto size = std::filesystem::file_size(file, error);
    if (error && error != std::errc::no_such_file_or_directory) return;
    bool rotate = !error && size > MaxLogBytes - line.size();
    if (rotate) {
        std::ifstream previous(file, std::ios::binary);
        const auto count = std::min<uintmax_t>(size, MaxLogBytes);
        previous.seekg(-static_cast<std::streamoff>(count), std::ios::end);
        std::string tail(static_cast<size_t>(count), '\0');
        if (!previous.read(tail.data(), static_cast<std::streamsize>(count))) return;
        previous.close();
        auto backup = file; backup += L".1";
        std::ofstream saved(backup, std::ios::binary | std::ios::trunc);
        saved.write(tail.data(), static_cast<std::streamsize>(tail.size())); saved.close();
        if (!saved) return; // Logging failure must not stop rendering or keep growing the old file.
    }
    std::ofstream output(file, std::ios::binary | (rotate ? std::ios::trunc : std::ios::app));
    output.write(line.data(), static_cast<std::streamsize>(line.size()));
}
void Log(const std::string& message) {
    static std::mutex mutex;
    std::lock_guard guard(mutex);
    OutputDebugStringA((message + "\n").c_str());
    SYSTEMTIME time; GetLocalTime(&time);
    AppendBoundedLog(DataDirectory() / L"capture.log",
        std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02} {}", time.wYear, time.wMonth,
            time.wDay, time.wHour, time.wMinute, time.wSecond, message));
}
std::string Narrow(const std::wstring& value) {
    int size = WideCharToMultiByte(CP_UTF8, 0, value.data(), (int)value.size(), nullptr, 0, nullptr, nullptr);
    std::string result(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), (int)value.size(), result.data(), size, nullptr, nullptr);
    return result;
}
std::wstring Wide(const std::string& value) {
    int size = MultiByteToWideChar(CP_UTF8, 0, value.data(), (int)value.size(), nullptr, 0);
    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.data(), (int)value.size(), result.data(), size);
    return result;
}
double ClockMs() {
    LARGE_INTEGER now, frequency; QueryPerformanceCounter(&now); QueryPerformanceFrequency(&frequency);
    return 1000.0 * now.QuadPart / frequency.QuadPart;
}
std::filesystem::path ExecutableDirectory() {
    wchar_t executable[32768]{};
    const DWORD length = GetModuleFileNameW(nullptr, executable, (DWORD)std::size(executable));
    if (!length) Check(HRESULT_FROM_WIN32(GetLastError()), "Find application directory");
    if (length >= std::size(executable)) throw std::runtime_error("Application path is too long.");
    return std::filesystem::path(executable).parent_path();
}
std::filesystem::path ResolveRuntimePath(const std::filesystem::path& saved, const std::filesystem::path& directory,
                                         const std::filesystem::path& chosen) {
    const auto exists = [](const std::filesystem::path& path) {
        std::error_code error;
        return std::filesystem::is_regular_file(path, error);
    };
    const auto selected = (saved.is_absolute() ? saved : directory / saved).lexically_normal();
    const auto bundled = directory / L"runtime" / L"nvngx_dlssnr.dll";
    const auto adjacent = directory / L"nvngx_dlssnr.dll";
    // Another package's bundled runtime (<package>/runtime/nvngx_dlssnr.dll beside its own display_filter.exe)
    // was saved by that package, not chosen: it must not shadow this package's newer bundled runtime. Unless the
    // user picked exactly that DLL here (an older package that rewrites Runtime leaves RuntimeChosen different).
    const auto package = selected.parent_path().parent_path();
    std::error_code same, picked;
    const bool foreignBundle = !_wcsicmp(selected.parent_path().filename().c_str(), L"runtime") && exists(package / L"display_filter.exe") &&
        !std::filesystem::equivalent(package, directory, same) && package != directory.lexically_normal() &&
        (chosen.empty() || !std::filesystem::equivalent(selected, chosen.is_absolute() ? chosen : directory / chosen, picked));
    if (!saved.empty() && exists(selected) && !(foreignBundle && (exists(bundled) || exists(adjacent)))) return selected;
    // A moved package must not remain tied to an obsolete saved path or the
    // launcher's working directory. Prefer runtime/, then a DLL beside the EXE.
    if (exists(bundled)) return bundled;
    if (exists(adjacent)) return adjacent;
    return saved.empty() ? bundled : selected;
}
Settings LoadSettings(const std::filesystem::path& settingsFile) {
    Settings s;
    const auto file = settingsFile.empty() ? DataDirectory() / L"settings.ini" : settingsFile;
    wchar_t path[32768]{};
    GetPrivateProfileStringW(L"NR", L"RuntimeChosen", L"", path, (DWORD)std::size(path), file.c_str());
    s.runtimeChosen = path;
    GetPrivateProfileStringW(L"NR", L"Runtime", L"", path, (DWORD)std::size(path), file.c_str());
    s.runtime = ResolveRuntimePath(path, ExecutableDirectory(), s.runtimeChosen);
    auto number = [&](const wchar_t* key, float fallback, float low, float high) {
        wchar_t text[64]{}; GetPrivateProfileStringW(L"NR", key, L"", text, 64, file.c_str());
        wchar_t* end = nullptr; float v = wcstof(text, &end);
        return end == text || *end || !std::isfinite(v) ? fallback : std::clamp(v, low, high);
    };
    s.strength = number(L"Strength", 1, 0, 2);
    s.colourStrength = number(L"ColourStrength", 1, 0, 4);
    s.model.skinStructure = number(L"SkinStructure", -1, -1, 2);
    s.skinDetail = number(L"SkinDetail", 1, 0, 1);
    s.skinColour = number(L"SkinColour", 1, 0, 1);
    s.environmentDetail = number(L"EnvironmentDetail", 1, 0, 1);
    s.environmentColour = number(L"EnvironmentColour", 1, 0, 1);
    s.separateSkin = number(L"SeparateSkin", 0, 0, 1) != 0;
    s.referenceWhiteNits = number(L"ReferenceWhiteNits", 0, 0, 10000);
    s.peakBrightnessNits = number(L"PeakBrightnessNits", 0, 0, 10000);
    s.blackLevelNits = number(L"BlackLevelNits", 0, -5, 5);
    s.unlockPasses = number(L"UnlockPasses", 0, 0, 1) != 0;
    s.passes = (unsigned)number(L"Passes", 1, 1, s.unlockPasses ? float(Settings::MaxPasses) : 2.0f);
    s.modelScale = (unsigned)number(L"ModelScale", 100, 25, 200);
    s.downscaler = (unsigned)number(L"Downscaler", 4, 0, 7);
    s.highlightGuard = number(L"HighlightGuard", 2, 1, 8);
    s.haloSuppression = number(L"HaloSuppression", 0, 0, 1);
    s.haloBrightSuppression = number(L"HaloBrightSuppression", 0, 0, 1);
    s.haloRadius = number(L"HaloRadius", 2, 1, 4);
    s.compareSplit = number(L"CompareSplit", .5f, 0, 1);
    s.compareZoom = number(L"CompareZoom", 1, 1, 2);
    s.compareSwap = number(L"CompareSwap", 0, 0, 1) != 0;
    s.compareLabels = number(L"CompareLabels", 1, 0, 1) != 0;
    s.labelScale = number(L"LabelScale", 1, .5f, 5);
    s.model.style = (unsigned)number(L"Style", 0, 0, 2);
    s.model.preset = (unsigned)number(L"Preset", 0, 0, Settings::ModelLast);
    s.model.autoMask = number(L"AutoMask", 1, 0, 1) != 0;
    s.transfer = (unsigned)number(L"Transfer", 1, 0, 3);
    s.upscaler = (unsigned)number(L"Upscaler", 0, 0, 1);
    s.model.intensity = number(L"Intensity", 1, 0, 2);
    s.model.localStructure = number(L"Structure", 1, 0, 2);
    s.model.localTone = number(L"LocalTone", 1, 0, 2);
    s.tone = (unsigned)number(L"Codec", 2, 0, 4);
    s.inputColour = (unsigned)number(L"InputColour", 0, 0, 2);
    s.maxFps = (unsigned)number(L"MaxFps", 0, 0, 360);
    s.captureBackend = (unsigned)number(L"CaptureBackend", 0, 0, 1);
    s.frameGeneration = (unsigned)number(L"FrameGeneration", 1, 1, 4);
    s.estimateMotion = number(L"EstimateMotion", 0, 0, 1) != 0;
    s.nvfp4FeedForward = number(L"Nvfp4FeedForward", 0, 0, 1) != 0;
    s.nvfp4Swin = (unsigned)number(L"Nvfp4Swin", 0, 0, 15);
    s.nvfp4SwinTwo = (unsigned)number(L"Nvfp4SwinTwo", 7, 0, 15);
    s.nvfp4SwinTrained = number(L"Nvfp4SwinTrained", 1, 0, 1) != 0;
    {
        wchar_t text[64]{}; GetPrivateProfileStringW(L"NR", L"Nvfp4SwinLayers", L"", text, 64, file.c_str());
        wchar_t* end = nullptr; const unsigned long long mask = wcstoull(text, &end, 16);
        s.nvfp4SwinLayers = end == text || *end ? ~0ull : mask;
    }
    s.nvfp4Split = (unsigned)number(L"Nvfp4Split", 0, 0, 63);
    s.nvfp4Vit = (unsigned)number(L"Nvfp4Vit", 0, 0, 7);
    s.nvfp4PrePost = (unsigned)number(L"Nvfp4PrePost", 0, 0, 3);
    s.nvfp4SplitTrained = number(L"Nvfp4SplitTrained", 1, 0, 1) != 0;
    s.nvfp4VitTrained = number(L"Nvfp4VitTrained", 1, 0, 1) != 0;
    s.nvfp4FfnTrained = number(L"Nvfp4FfnTrained", 1, 0, 1) != 0;
    {
        wchar_t text[512]{}; GetPrivateProfileStringW(L"NR", L"Nvfp4Net", L"", text, 512, file.c_str());
        s.nvfp4Net = Narrow(text);
    }
    s.reapplyGain = number(L"ReapplyGain", .81f, 0, 2);
    s.editInterval = (unsigned)number(L"EditInterval", 1, 1, 4);
    s.editLate = number(L"EditLate", 0, 0, 1) != 0;
    const auto readSpatial = [&](const std::wstring& prefix, DlssNr::Spatial::Settings spatial) {
        const auto value = [&](const wchar_t* key, float fallback, float low, float high) {
            return number((prefix + key).c_str(), fallback, low, high);
        };
        spatial.enabled = value(L"SpatialCompression", spatial.enabled ? 1.f : 0.f, 0, 1) != 0;
        spatial.centerX = value(L"SpatialCenterX", spatial.centerX, 1, 99);
        spatial.centerY = value(L"SpatialCenterY", spatial.centerY, 1, 99);
        spatial.workX = value(L"SpatialWorkX", spatial.workX, 25, 100);
        spatial.workY = value(L"SpatialWorkY", spatial.workY, 25, 100);
        spatial.offsetX = value(L"SpatialOffsetX", spatial.offsetX, -49, 49);
        spatial.offsetY = value(L"SpatialOffsetY", spatial.offsetY, -49, 49);
        spatial.shiftX = value(L"SpatialShiftX", spatial.shiftX, -49, 49);
        spatial.shiftY = value(L"SpatialShiftY", spatial.shiftY, -49, 49);
        return spatial;
    };
    s.spatial = readSpatial(L"", s.spatial);
    for (unsigned pass = 1; pass < Settings::MaxPasses; ++pass) {
        const auto key = [&](const wchar_t* field) { return std::format(L"Pass{}.{}", pass + 1, field); };
        if (number(key(L"ResolutionOverride").c_str(), 0, 0, 1) != 0) {
            auto resolution = s.PassResolution(pass);
            resolution.modelScale = (unsigned)number(key(L"ModelScale").c_str(), float(resolution.modelScale), 25, 200);
            resolution.downscaler = (unsigned)number(key(L"Downscaler").c_str(), float(resolution.downscaler), 0, 7);
            resolution.transfer = (unsigned)number(key(L"Transfer").c_str(), float(resolution.transfer), 0, 3);
            resolution.upscaler = (unsigned)number(key(L"Upscaler").c_str(), float(resolution.upscaler), 0, 1);
            resolution.spatial = readSpatial(std::format(L"Pass{}.", pass + 1), resolution.spatial);
            s.passResolutionOverrides[pass - 1] = resolution;
        }
        auto& shortcut = s.passShortcuts[pass - 1];
        shortcut.reapply = number(key(L"Reapply").c_str(), 0, 0, 1) != 0;
        shortcut.borrowTransformer = number(key(L"BorrowTransformer").c_str(), 0, 0, 1) != 0;
        shortcut.borrowBelow = number(key(L"BorrowBelow").c_str(), 0, 0, 1) != 0;
        // Levels are 1/2..1/32 of the frame size; other saved values snap to the nearest power of two.
        shortcut.borrowLevel = 1u << std::clamp((int)std::lround(std::log2(number(key(L"BorrowLevel").c_str(), 32, 2, 32))), 1, 5);
        if (number(key(L"Override").c_str(), 0, 0, 1) == 0) continue;
        auto model = s.PassModel(pass);
        model.intensity = number(key(L"Intensity").c_str(), model.intensity, 0, 2);
        model.localStructure = number(key(L"Structure").c_str(), model.localStructure, 0, 2);
        model.localTone = number(key(L"LocalTone").c_str(), 0, 0, 2);
        model.skinStructure = number(key(L"SkinStructure").c_str(), model.skinStructure, -1, 2);
        model.style = (unsigned)number(key(L"Style").c_str(), (float)model.style, 0, 2);
        model.preset = (unsigned)number(key(L"Preset").c_str(), (float)model.preset, 0, Settings::ModelLast);
        model.autoMask = number(key(L"AutoMask").c_str(), model.autoMask ? 1.f : 0.f, 0, 1) != 0;
        s.passOverrides[pass - 1] = model;
    }
    return s;
}
void SaveSettings(const Settings& s, const std::filesystem::path& settingsFile) {
    const auto file = settingsFile.empty() ? DataDirectory() / L"settings.ini" : settingsFile;
    auto write = [&](const wchar_t* key, const std::wstring& value) {
        if (!WritePrivateProfileStringW(L"NR", key, value.c_str(), file.c_str()))
            Check(HRESULT_FROM_WIN32(GetLastError()), "Save settings");
    };
    write(L"Runtime", s.runtime.wstring()); write(L"RuntimeChosen", s.runtimeChosen.wstring());
    write(L"Strength", std::to_wstring(s.strength));
    write(L"ColourStrength", std::to_wstring(s.colourStrength));
    write(L"SkinStructure", std::to_wstring(s.model.skinStructure));
    write(L"SkinDetail", std::to_wstring(s.skinDetail));
    write(L"SkinColour", std::to_wstring(s.skinColour));
    write(L"EnvironmentDetail", std::to_wstring(s.environmentDetail));
    write(L"EnvironmentColour", std::to_wstring(s.environmentColour));
    write(L"SeparateSkin", s.separateSkin ? L"1" : L"0");
    write(L"ReferenceWhiteNits", std::to_wstring(s.referenceWhiteNits));
    write(L"PeakBrightnessNits", std::to_wstring(s.peakBrightnessNits));
    write(L"BlackLevelNits", std::to_wstring(s.blackLevelNits));
    write(L"Passes", std::to_wstring(s.passes));
    write(L"ModelScale", std::to_wstring(s.modelScale));
    write(L"Transfer", std::to_wstring(s.transfer));
    write(L"Upscaler", std::to_wstring(s.upscaler));
    write(L"Intensity", std::to_wstring(s.model.intensity));
    write(L"Structure", std::to_wstring(s.model.localStructure));
    write(L"LocalTone", std::to_wstring(s.model.localTone)); write(L"Codec", std::to_wstring(s.tone));
    write(L"EstimateMotion", s.estimateMotion ? L"1" : L"0");
    write(L"Nvfp4FeedForward", s.nvfp4FeedForward ? L"1" : L"0");
    write(L"Nvfp4Swin", std::to_wstring(s.nvfp4Swin));
    write(L"Nvfp4SwinTwo", std::to_wstring(s.nvfp4SwinTwo));
    write(L"Nvfp4SwinTrained", s.nvfp4SwinTrained ? L"1" : L"0");
    write(L"Nvfp4SwinLayers", std::format(L"{:x}", s.nvfp4SwinLayers));
    write(L"Nvfp4Split", std::to_wstring(s.nvfp4Split));
    write(L"Nvfp4Vit", std::to_wstring(s.nvfp4Vit));
    write(L"Nvfp4PrePost", std::to_wstring(s.nvfp4PrePost));
    write(L"Nvfp4SplitTrained", s.nvfp4SplitTrained ? L"1" : L"0");
    write(L"Nvfp4VitTrained", s.nvfp4VitTrained ? L"1" : L"0");
    write(L"Nvfp4FfnTrained", s.nvfp4FfnTrained ? L"1" : L"0");
    write(L"Nvfp4Net", Wide(s.nvfp4Net));
    write(L"InputColour", std::to_wstring(s.inputColour));
    write(L"MaxFps", std::to_wstring(s.maxFps));
    write(L"CaptureBackend", std::to_wstring(s.captureBackend));
    write(L"FrameGeneration", std::to_wstring(s.frameGeneration));
    write(L"UnlockPasses", s.unlockPasses ? L"1" : L"0");
    write(L"Downscaler", std::to_wstring(s.downscaler));
    write(L"HighlightGuard", std::to_wstring(s.highlightGuard));
    write(L"HaloSuppression", std::to_wstring(s.haloSuppression));
    write(L"HaloBrightSuppression", std::to_wstring(s.haloBrightSuppression));
    write(L"HaloRadius", std::to_wstring(s.haloRadius));
    write(L"CompareSplit", std::to_wstring(s.compareSplit));
    write(L"CompareZoom", std::to_wstring(s.compareZoom));
    write(L"CompareSwap", s.compareSwap ? L"1" : L"0");
    write(L"CompareLabels", s.compareLabels ? L"1" : L"0");
    write(L"LabelScale", std::to_wstring(s.labelScale));
    write(L"Style", std::to_wstring(s.model.style));
    write(L"Preset", std::to_wstring(s.model.preset));
    write(L"AutoMask", s.model.autoMask ? L"1" : L"0");
    write(L"ReapplyGain", std::to_wstring(s.reapplyGain));
    write(L"EditInterval", std::to_wstring(s.editInterval));
    write(L"EditLate", s.editLate ? L"1" : L"0");
    const auto writeSpatial = [&](const std::wstring& prefix, const DlssNr::Spatial::Settings& spatial) {
        write((prefix + L"SpatialCompression").c_str(), spatial.enabled ? L"1" : L"0");
        write((prefix + L"SpatialCenterX").c_str(), std::to_wstring(spatial.centerX));
        write((prefix + L"SpatialCenterY").c_str(), std::to_wstring(spatial.centerY));
        write((prefix + L"SpatialWorkX").c_str(), std::to_wstring(spatial.workX));
        write((prefix + L"SpatialWorkY").c_str(), std::to_wstring(spatial.workY));
        write((prefix + L"SpatialOffsetX").c_str(), std::to_wstring(spatial.offsetX));
        write((prefix + L"SpatialOffsetY").c_str(), std::to_wstring(spatial.offsetY));
        write((prefix + L"SpatialShiftX").c_str(), std::to_wstring(spatial.shiftX));
        write((prefix + L"SpatialShiftY").c_str(), std::to_wstring(spatial.shiftY));
    };
    writeSpatial(L"", s.spatial);
    for (unsigned pass = 1; pass < Settings::MaxPasses; ++pass) {
        const auto key = [&](const wchar_t* field) { return std::format(L"Pass{}.{}", pass + 1, field); };
        write(key(L"ResolutionOverride").c_str(), s.passResolutionOverrides[pass - 1] ? L"1" : L"0");
        if (s.passResolutionOverrides[pass - 1]) {
            const auto resolution = s.PassResolution(pass);
            write(key(L"ModelScale").c_str(), std::to_wstring(resolution.modelScale));
            write(key(L"Downscaler").c_str(), std::to_wstring(resolution.downscaler));
            write(key(L"Transfer").c_str(), std::to_wstring(resolution.transfer));
            write(key(L"Upscaler").c_str(), std::to_wstring(resolution.upscaler));
            writeSpatial(std::format(L"Pass{}.", pass + 1), resolution.spatial);
        }
        const auto& shortcut = s.passShortcuts[pass - 1];
        write(key(L"Reapply").c_str(), shortcut.reapply ? L"1" : L"0");
        write(key(L"BorrowTransformer").c_str(), shortcut.borrowTransformer ? L"1" : L"0");
        write(key(L"BorrowBelow").c_str(), shortcut.borrowBelow ? L"1" : L"0");
        write(key(L"BorrowLevel").c_str(), std::to_wstring(shortcut.borrowLevel));
        write(key(L"Override").c_str(), s.passOverrides[pass - 1] ? L"1" : L"0");
        if (!s.passOverrides[pass - 1]) continue;
        const auto model = s.PassModel(pass);
        write(key(L"Intensity").c_str(), std::to_wstring(model.intensity));
        write(key(L"Structure").c_str(), std::to_wstring(model.localStructure));
        write(key(L"LocalTone").c_str(), std::to_wstring(model.localTone));
        write(key(L"SkinStructure").c_str(), std::to_wstring(model.skinStructure));
        write(key(L"Style").c_str(), std::to_wstring(model.style));
        write(key(L"Preset").c_str(), std::to_wstring(model.preset));
        write(key(L"AutoMask").c_str(), model.autoMask ? L"1" : L"0");
    }
}
}
