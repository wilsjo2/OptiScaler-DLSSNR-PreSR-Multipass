#include "FeatureChecks.h"
#include "Presets.h"
#include <DirectXPackedVector.h>
#include <iostream>
#include <fstream>
namespace nr {
void RuntimePathCheck() {
    const auto root = std::filesystem::temp_directory_path() /
        std::format(L"nr-path-{}-{}-\u6d4b\u8bd5", GetCurrentProcessId(), GetTickCount64());
    const auto app = root / L"moved app";
    const auto bundled = app / L"runtime" / L"nvngx_dlssnr.dll";
    const auto adjacent = app / L"nvngx_dlssnr.dll";
    const auto custom = root / L"custom" / L"nvngx_dlssnr.dll";
    std::filesystem::create_directories(bundled.parent_path());
    std::filesystem::create_directories(custom.parent_path());
    struct Cleanup { std::filesystem::path root; ~Cleanup() { std::error_code error; std::filesystem::remove_all(root, error); } } cleanup{root};
    std::ofstream(bundled).put('x'); std::ofstream(adjacent).put('x'); std::ofstream(custom).put('x');
    const auto expect = [&](const std::filesystem::path& saved, const std::filesystem::path& wanted) {
        if (ResolveRuntimePath(saved, app) != wanted) throw std::runtime_error("NR runtime path lookup failed.");
    };
    expect({}, bundled);
    expect(root / L"old package/runtime/nvngx_dlssnr.dll", bundled);
    expect(L"runtime/nvngx_dlssnr.dll", bundled);
    expect(custom, custom);
    expect(app / L"runtime", bundled); // A directory is not a DLL.
    // Another package's bundled runtime (the older package this one replaces) must not shadow ours.
    const auto old = root / L"old package";
    std::filesystem::create_directories(old / L"runtime");
    std::ofstream(old / L"display_filter.exe").put('x'); std::ofstream(old / L"runtime" / L"nvngx_dlssnr.dll").put('x');
    expect(old / L"runtime" / L"nvngx_dlssnr.dll", bundled);
    expect(old / L"Runtime" / L"nvngx_dlssnr.dll", bundled);
    // ...unless the user picked exactly that DLL here; a different choice (an older package rewrote Runtime) does not count.
    const auto chosen = [&](const std::filesystem::path& saved, const std::filesystem::path& pick, const std::filesystem::path& wanted) {
        if (ResolveRuntimePath(saved, app, pick) != wanted) throw std::runtime_error("Chosen NR runtime path lookup failed.");
    };
    chosen(old / L"runtime" / L"nvngx_dlssnr.dll", old / L"runtime" / L"nvngx_dlssnr.dll", old / L"runtime" / L"nvngx_dlssnr.dll");
    chosen(old / L"runtime" / L"nvngx_dlssnr.dll", old / L"Runtime" / L"NVNGX_DLSSNR.dll", old / L"runtime" / L"nvngx_dlssnr.dll");
    chosen(old / L"runtime" / L"nvngx_dlssnr.dll", custom, bundled);
    chosen(old / L"runtime" / L"nvngx_dlssnr.dll", root / L"missing" / L"nvngx_dlssnr.dll", bundled);
    const auto missing = root / L"missing" / L"nvngx_dlssnr.dll";
    chosen(missing, missing, missing); // Keep an explicit selection even with a valid bundled model.
    // A user-chosen DLL in a runtime folder without a package executable still wins.
    const auto tools = root / L"tools" / L"runtime" / L"nvngx_dlssnr.dll";
    std::filesystem::create_directories(tools.parent_path()); std::ofstream(tools).put('x');
    expect(tools, tools);
    expect(bundled, bundled); // This package's own bundled runtime, saved earlier.
    std::filesystem::remove(bundled); expect({}, adjacent);
    expect(old / L"runtime" / L"nvngx_dlssnr.dll", adjacent);
    std::filesystem::remove(adjacent); expect({}, bundled);
    // With no runtime in this package, the other package's DLL is still better than none.
    expect(old / L"runtime" / L"nvngx_dlssnr.dll", old / L"runtime" / L"nvngx_dlssnr.dll");
    std::cout << "PASS runtime lookup: moved package, missing saved path, executable-relative path, custom DLL, adjacent DLL, other package's bundled runtime (unless chosen) and Unicode folders\n";
}
// Presets are files in the data folder: this uses names no user would choose and removes them again.
static void PresetsCheck(const Settings& saved) {
    const auto tag = std::format(L"nr-check-{}-{}", GetCurrentProcessId(), GetTickCount64());
    const std::wstring first = tag + L" one", second = tag + L" two", capitals = tag + L" TWO";
    struct Cleanup { std::vector<std::wstring> names; ~Cleanup() {
        for (const auto& name : names) { std::error_code error; std::filesystem::remove(PresetDirectory() / (name + L".ini"), error); } } }
        cleanup{{first, second}};
    const auto fails = [](const auto& action) { try { action(); } catch (const std::runtime_error&) { return true; } return false; };
    const auto listed = [](const std::wstring& name) { const auto names = ListPresets(); return std::find(names.begin(), names.end(), name) != names.end(); };
    for (const wchar_t* bad : {L"", L"   ", L"a/b", L"a\\b", L"a:b", L"what?", L"con", L"LPT1", L"tab\there"})
        if (PresetNameError(bad).empty()) throw std::runtime_error("A name that cannot be a file name was accepted for a preset.");
    if (PresetNameError(std::wstring(65, L'x')).empty() || !PresetNameError(std::wstring(64, L'x')).empty() ||
        !PresetNameError(L"  2x Cinematic, strong.  ").empty())
        throw std::runtime_error("Preset name rules changed.");
    if (PresetExists(first) || listed(first)) throw std::runtime_error("The preset check's name is in use.");
    SavePreset(L"  " + first + L" ", saved); // (spaces at the ends are not part of the name)
    if (!PresetExists(first) || !listed(first)) throw std::runtime_error("A saved preset is not listed.");
    // What belongs to the machine and the session stays as it is when a preset is loaded; the rest is the preset's.
    Settings current;
    current.runtime = L"X:/this machine/nvngx_dlssnr.dll"; current.runtimeChosen = L"X:/picked/nvngx_dlssnr.dll";
    current.captureBackend = 1; current.frameGeneration = 2; current.enabled = false; current.holdFrame = true;
    current.applyModel = false; current.debugView = 2; current.compare = 3; current.compareSplit = .75f; current.compareZoom = 2.5f;
    current.compareSwap = !saved.compareSwap; current.compareLabels = !saved.compareLabels; current.labelScale = 3;
    current.previewSkin = true; current.showFoveation = true; current.foveationPreviewPass = 2;
    current.presentMode = 0; current.latencyMode = 0; // (saved has 1 and 1: the session's presentation and pacing stay)
    auto expected = saved;
    expected.presentMode = current.presentMode; expected.latencyMode = current.latencyMode;
    expected.runtime = current.runtime; expected.runtimeChosen = current.runtimeChosen;
    expected.captureBackend = current.captureBackend; expected.frameGeneration = current.frameGeneration;
    expected.enabled = current.enabled; expected.holdFrame = current.holdFrame; expected.applyModel = current.applyModel;
    expected.debugView = current.debugView; expected.compare = current.compare; expected.compareSplit = current.compareSplit;
    expected.compareZoom = current.compareZoom; expected.compareSwap = current.compareSwap; expected.compareLabels = current.compareLabels;
    expected.labelScale = current.labelScale; expected.previewSkin = current.previewSkin;
    expected.showFoveation = current.showFoveation; expected.foveationPreviewPass = current.foveationPreviewPass;
    if (LoadPreset(first, current) != expected) throw std::runtime_error("A preset did not come back as it was saved.");
    {
        std::ifstream file(PresetDirectory() / (first + L".ini"), std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::string plain; for (const char c : text) if (c) plain += c; // (the file may be UTF-16)
        if (plain.find("chosen/runtime") != std::string::npos || plain.find("nvngx_dlssnr") != std::string::npos)
            throw std::runtime_error("A preset file holds the runtime's path.");
    }
    auto changed = saved; changed.strength = .35f; changed.colourStrength = 3.5f; changed.model.intensity = .6f;
    SavePreset(first, changed); expected.strength = .35f; expected.colourStrength = 3.5f; expected.model.intensity = .6f;
    if (LoadPreset(first, current) != expected) throw std::runtime_error("Overwriting a preset did not replace it.");
    RenamePreset(first, second);
    if (PresetExists(first) || !listed(second) || LoadPreset(second, current) != expected) throw std::runtime_error("Renaming a preset failed.");
    RenamePreset(second, capitals); // a change of capitals only
    if (!listed(capitals) || listed(second)) throw std::runtime_error("Renaming a preset to other capitals failed.");
    SavePreset(first, saved);
    if (!fails([&] { RenamePreset(first, second); }) || !PresetExists(first) || LoadPreset(capitals, current) != expected)
        throw std::runtime_error("A rename onto an existing preset was not refused.");
    if (!fails([&] { LoadPreset(tag + L" none", current); }) || !fails([&] { DeletePreset(tag + L" none"); }) ||
        !fails([&] { RenamePreset(tag + L" none", tag + L" other"); }) || !fails([&] { SavePreset(L"a/b", saved); }))
        throw std::runtime_error("A missing preset or a bad name was not refused.");
    DeletePreset(first); DeletePreset(capitals);
    if (PresetExists(first) || PresetExists(second) || listed(first) || listed(capitals)) throw std::runtime_error("Deleting a preset failed.");
    std::cout << "PASS presets: name rules, save, list, load (session and machine settings kept), overwrite, rename, capitals, refusals, delete\n";
}
void SettingsCheck(const Settings& settings) {
    Settings saved = settings;
    if (saved.runtime.empty()) saved.runtime = ResolveRuntimePath({}, ExecutableDirectory()); // What loading resolves it to.
    saved.unlockPasses = true; saved.passes = 10; saved.modelScale = 175; saved.downscaler = 6;
    saved.model.style = 2; saved.model.preset = Settings::ModelANvfp4; saved.model.autoMask = false; // The selector's own value, above the runtime's presets.
    saved.highlightGuard = 3; saved.compareSplit = .25f; saved.compareZoom = 1.5f;
    saved.compareSwap = true; saved.compareLabels = false; saved.labelScale = 2;
    saved.peakBrightnessNits = 1000; saved.blackLevelNits = -.05f;
    saved.haloSuppression = .75f; saved.haloBrightSuppression = .25f; saved.haloRadius = 3;
    saved.upscaler = 1; saved.transfer = 0; // DLSS must not silently force matched transfer on load.
    saved.passOverrides[8] = saved.PassModel(9); saved.passOverrides[8]->localTone = .5f; saved.passOverrides[8]->preset = 1;
    saved.reapplyGain = 1.25f;
    saved.runtimeChosen = L"C:/chosen/runtime/nvngx_dlssnr.dll"; // Stored as picked, never resolved.
    saved.passShortcuts[0] = {true, true, false, 8}; // Saved as ticked even though re-apply wins.
    saved.passShortcuts[3] = {false, true, true, 2};
    saved.passShortcuts[8] = {false, false, true, 16}; // Pass 10, without a model override.
    saved.presentMode = 1; saved.latencyMode = 1; saved.editSmoothing = .5f; saved.motionZeroBias = .25f; // Pacing and latency tab.
    const auto file = std::filesystem::temp_directory_path() / std::format(L"nr-feature-settings-{}-{}.ini", GetCurrentProcessId(), GetTickCount64());
    SaveSettings(saved, file);
    const auto restored = LoadSettings(file);
    if (restored != saved) { std::filesystem::remove(file); throw std::runtime_error("Feature settings failed to round trip."); }
    // Out-of-range saved values are clamped: gain to 0..2, borrow levels to the nearest of 2/4/8/16/32.
    WritePrivateProfileStringW(L"NR", L"ReapplyGain", L"7", file.c_str());
    WritePrivateProfileStringW(L"NR", L"Pass2.BorrowLevel", L"5", file.c_str());
    WritePrivateProfileStringW(L"NR", L"Pass3.BorrowLevel", L"1000", file.c_str());
    WritePrivateProfileStringW(L"NR", L"Pass4.BorrowLevel", L"x", file.c_str());
    WritePrivateProfileStringW(L"NR", L"Pass5.Reapply", L"3", file.c_str());
    WritePrivateProfileStringW(L"NR", L"PresentMode", L"9", file.c_str());
    WritePrivateProfileStringW(L"NR", L"LatencyMode", L"9", file.c_str());
    WritePrivateProfileStringW(L"NR", L"EditSmoothing", L"5", file.c_str());
    WritePrivateProfileStringW(L"NR", L"MotionZeroBias", L"-1", file.c_str());
    const auto clamped = LoadSettings(file); std::filesystem::remove(file);
    if (clamped.reapplyGain != 2 || clamped.passShortcuts[0].borrowLevel != 4 || clamped.passShortcuts[1].borrowLevel != 32 ||
        clamped.passShortcuts[2].borrowLevel != 32 || !clamped.passShortcuts[3].reapply)
        throw std::runtime_error("Saved re-apply/borrow values were not validated.");
    if (clamped.presentMode != 1 || clamped.latencyMode != 1 || clamped.editSmoothing != .9f || clamped.motionZeroBias != 0)
        throw std::runtime_error("Saved pacing/latency/steady values were not validated.");
    const auto missing = LoadSettings(std::filesystem::temp_directory_path() / L"nr-feature-settings-missing.ini");
    if (missing.reapplyGain != .81f || missing.passShortcuts != decltype(missing.passShortcuts){})
        throw std::runtime_error("Re-apply/borrow defaults changed.");
    if (missing.presentMode != 0 || missing.latencyMode != 0 || missing.editSmoothing != 0 || missing.motionZeroBias != 0)
        throw std::runtime_error("Pacing/latency/steady defaults are not the original behaviour.");
    // A resolution override left on a re-apply pass is dead: it must not select the independent chain.
    Settings dead; dead.passes = 2; dead.passResolutionOverrides[0] = dead.PassResolution(1);
    if (!dead.HasPassResolutionOverrides()) throw std::runtime_error("Pass resolution override not seen.");
    dead.passShortcuts[0].reapply = true;
    if (dead.HasPassResolutionOverrides()) throw std::runtime_error("A re-apply pass's resolution override selected the chain.");
    PresetsCheck(saved);
    std::cout << "PASS per-pass, re-apply/borrow, supersampling, comparison and composition settings round trip and validation (isolated INI)\n";
}
void FeatureChecks(Processor& processor, const Settings& settings, std::vector<uint16_t>& pixels,
    const std::function<void()>& upload,
    const std::function<std::vector<uint16_t>(Settings, bool, float, bool)>& read) {
    using DirectX::PackedVector::XMConvertHalfToFloat;
    using DirectX::PackedVector::XMConvertFloatToHalf;
    const auto original = pixels;
    SettingsCheck(settings);
    auto mode = settings; mode.estimateMotion = false; mode.upscaler = 0; mode.modelScale = 100; mode.compareLabels = false;
    mode.passes = 1;
    const auto one = read(mode, false, 1, true);
    mode.passes = 10;
    const auto ten = read(mode, false, 1, true);
    if (ten == one) throw std::runtime_error("Ten passes had no additional effect.");
    mode.passes = 1;
    if (read(mode, false, 1, true) != one) throw std::runtime_error("Pass count round trip changed pass 1.");
    mode.passes = 10;
    if (read(mode, false, 1, true) != ten) throw std::runtime_error("Ten-pass round trip changed output.");
    mode.passOverrides[8] = mode.PassModel(9); mode.passOverrides[8]->intensity = .25f;
    if (read(mode, false, 1, true) == ten) throw std::runtime_error("Pass 10 override did not affect output.");
    mode.passes = 1;
    if (read(mode, false, 1, true) != one) throw std::runtime_error("Inactive pass override changed active output.");
    mode.model.style = 1; mode.model.autoMask = false;
    if (read(mode, false, 1, true) == one) throw std::runtime_error("Style/mask tuning did not change output.");
    std::cout << "PASS independent 1/10-pass switching, pass-10 override, inactive profile isolation and style/mask tuning\n";
    for (const bool chain : {false, true}) {
        // Re-apply: no network for pass 2+; r consecutive re-apply passes equal one pass with r times the gain.
        auto shortcut = settings; shortcut.estimateMotion = false; shortcut.upscaler = 0; shortcut.modelScale = 100;
        shortcut.compareLabels = false; shortcut.unlockPasses = true; shortcut.debugView = 2; // Raw model answer.
        if (chain) for (unsigned pass = 1; pass < 3; ++pass) shortcut.passResolutionOverrides[pass - 1] = shortcut.PassResolution(pass);
        const char* path = chain ? "independent-pass chain" : "shared-feature";
        shortcut.passes = 1; const auto single = read(shortcut, false, 1, true);
        shortcut.passes = 2; const auto twice = read(shortcut, false, 1, true);
        shortcut.passShortcuts[0].reapply = true; shortcut.reapplyGain = .81f;
        const auto reapplied = read(shortcut, false, 1, true);
        if (reapplied == single || reapplied == twice) throw std::runtime_error(std::format("{}: re-apply matched one or two network passes.", path));
        if (read(shortcut, false, 1, true) != reapplied) throw std::runtime_error(std::format("{}: re-apply was not deterministic.", path));
        shortcut.reapplyGain = 1;
        if (read(shortcut, false, 1, true) == reapplied) throw std::runtime_error(std::format("{}: re-apply gain had no effect.", path));
        shortcut.reapplyGain = 2 * .81f; const auto doubled = read(shortcut, false, 1, true);
        shortcut.reapplyGain = .81f; shortcut.passes = 3; shortcut.passShortcuts[1].reapply = true;
        if (read(shortcut, false, 1, true) != doubled) throw std::runtime_error(std::format("{}: two re-apply passes differ from one with twice the gain.", path));
        shortcut.passShortcuts[1].reapply = false; // Real, re-apply, real: the third pass sees the clamped re-apply result.
        const auto between = read(shortcut, false, 1, true);
        if (between == reapplied || between == twice) throw std::runtime_error(std::format("{}: a network pass after re-apply had no effect.", path));
        shortcut.passes = 1;
        if (read(shortcut, false, 1, true) != single) throw std::runtime_error(std::format("{}: inactive re-apply passes changed active output.", path));
        std::cout << std::format("PASS {} re-apply: no evaluation, gain, run of two = one dispatch at twice the gain, later network pass, inactive isolation\n", path);
    }
    mode = settings; mode.estimateMotion = false; mode.passes = 1; mode.upscaler = 0; mode.modelScale = 100; mode.compareLabels = false;
    mode.holdFrame = true;
    const auto held = read(mode, false, 1, true);
    for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
        pixels[i] = XMConvertFloatToHalf(XMConvertHalfToFloat(pixels[i]) * .25f);
    upload();
    if (read(mode, false, 1, true) != held) throw std::runtime_error("Held frame followed the live source.");
    mode.modelScale = 125; mode.applyModel = false;
    if (read(mode, false, 1, true) != original) throw std::runtime_error("Hold/model resize or hidden edit lost original pixels.");
    mode.holdFrame = false;
    if (read(mode, false, 1, true) != pixels) throw std::runtime_error("Unholding did not restore live input.");
    pixels = original; upload(); mode.applyModel = true;
    std::cout << "PASS held-frame identity across source changes/model resize; hidden edit; live resume\n";
    for (unsigned filter = 0; filter < 8; ++filter) {
        mode.downscaler = filter;
        auto result = read(mode, false, 1, true);
        if (result == original) throw std::runtime_error("Supersampling filter returned the original.");
    }
    mode.modelScale = 200; read(mode, false, 1, true);
    auto temporal = mode; temporal.modelScale = 125; temporal.passes = 3; temporal.estimateMotion = true;
    temporal.passOverrides[1] = temporal.PassModel(2); temporal.passOverrides[1]->localTone = .5f;
    for (unsigned frame = 0; frame < 3; ++frame) {
        read(temporal, true, 1, frame == 0);
        if (processor.historyReset != (frame == 0)) throw std::runtime_error("Supersampled HDR history reset unexpectedly.");
    }
    temporal.holdFrame = true; read(temporal, true, 1, false);
    temporal.holdFrame = false; read(temporal, true, 1, false);
    if (!processor.historyReset) throw std::runtime_error("Leaving hold did not reset temporal history.");
    mode.modelScale = 100;
    if (read(mode, false, 1, true) != one) throw std::runtime_error("Supersampling round trip changed native output.");
    std::cout << "PASS all eight supersampling filters, 200% model, supersampled HDR/three-pass motion, hold reset and exact native round trip\n";
    mode.highlightGuard = 1; mode.colourStrength = 0;
    const auto protectedLight = read(mode, true, 1, true);
    double error = 0;
    for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
        error += std::abs(XMConvertHalfToFloat(protectedLight[i]) - XMConvertHalfToFloat(original[i]));
    if (error / (pixels.size() / 4 * 3) > .0001) throw std::runtime_error("Highlight guard 1 did not preserve luminance.");
    mode.highlightGuard = 2; mode.colourStrength = 1;
    for (unsigned debug = 1; debug <= 3; ++debug) {
        mode.debugView = debug;
        if (read(mode, false, 1, true) == one) throw std::runtime_error("Debug view did not change the output.");
    }
    mode.debugView = 0; mode.compare = 2; mode.compareSplit = .3f;
    const auto wipe = read(mode, false, 1, true);
    mode.compareSwap = true;
    const auto swapped = read(mode, false, 1, true);
    if (wipe == swapped) throw std::runtime_error("Comparison side swap had no effect.");
    const auto width = processor.width, height = processor.height;
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x + 2 < unsigned(width * .3f); ++x)
        for (unsigned c = 0; c < 3; ++c) {
            const auto i = (size_t(y) * width + x) * 4 + c;
            if (wipe[i] != original[i] || swapped[i] != one[i]) throw std::runtime_error("Wipe halves are not the matched original/processed frame.");
        }
    mode.compare = 3;
    const auto fitted = read(mode, false, 1, true);
    mode.compareZoom = 2;
    if (read(mode, false, 1, true) == fitted) throw std::runtime_error("Comparison zoom had no effect.");
    mode.compareLabels = true;
    const auto labelled = read(mode, false, 1, true);
    mode.labelScale = 2;
    if (read(mode, false, 1, true) == labelled) throw std::runtime_error("Comparison label size had no effect.");
    std::cout << "PASS highlight guard, three debug views, matched wipe sides, side-by-side zoom and labels\n";
    mode = settings; mode.passes = 2; mode.modelScale = 50; mode.upscaler = 1; mode.transfer = 0;
    mode.estimateMotion = false; mode.inputColour = 1; mode.compareLabels = false;
    mode.tone = 0;
    // Recreate the private feature before each pixel comparison: Reset alone can
    // vary slightly with the DLSS feature's age on identical input.
    auto freshDlss = [&](Settings value, bool hdr) {
        auto native = value; native.modelScale = 100;
        processor.Resize(processor.width, processor.height, native);
        return read(value, hdr, 1, true);
    };
    const auto classicDlss = freshDlss(mode, false);
    if (classicDlss == original) throw std::runtime_error("Classic DLSS did not apply NR.");
    mode.upscaler = 0;
    if (read(mode, false, 1, true) == classicDlss) throw std::runtime_error("Classic DLSS fell back to bilinear.");
    mode.upscaler = 1; mode.transfer = 1;
    if (freshDlss(mode, false) == classicDlss) throw std::runtime_error("Classic DLSS was coerced to matched residuals.");
    mode.transfer = 0;
    if (freshDlss(mode, false) != classicDlss) throw std::runtime_error("Classic DLSS changed after switching transfer.");
    mode.applyModel = false;
    if (read(mode, false, 1, true) != original) throw std::runtime_error("Classic DLSS hidden edit changed the original.");
    mode.applyModel = true; mode.strength = 0;
    if (read(mode, true, 1, true) != original) throw std::runtime_error("Classic DLSS zero strength changed the original.");
    mode.strength = 1; mode.estimateMotion = true;
    for (unsigned transfer : {0u, 1u, 0u}) {
        mode.transfer = transfer;
        read(mode, false, 1, false);
        if (!processor.historyReset) throw std::runtime_error("DLSS transfer switch reused incompatible history.");
        read(mode, false, 1, false);
        if (processor.historyReset) throw std::runtime_error("Classic/matched DLSS history failed to resume.");
    }
    mode.modelScale = 100;
    const auto nativeDlss = read(mode, false, 1, true);
    mode.upscaler = 0;
    if (read(mode, false, 1, true) != nativeDlss) throw std::runtime_error("Classic DLSS selection changed native-size output.");
    mode.modelScale = 50; mode.upscaler = 1; mode.inputColour = 2;
    for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
        pixels[i] = XMConvertFloatToHalf(XMConvertHalfToFloat(original[i]) * 4);
    upload();
    for (unsigned tone = 0; tone < 5; ++tone) {
        mode.tone = tone;
        if (read(mode, true, 1, true) == pixels) throw std::runtime_error("Classic DLSS HDR output is unchanged.");
    }
    pixels = original; upload();
    std::cout << "PASS Classic DLSS: distinct colour reconstruction, bilinear/matched switching, exact bypass/native identity, transfer history reset/resume, two-pass motion and all five HDR mappings\n";
    mode.estimateMotion = false; mode.inputColour = 2; mode.tone = 2;
    for (unsigned scale : {99u,100u,101u}) for (unsigned transfer : {0u,1u}) {
        mode.modelScale = scale; mode.transfer = transfer; mode.upscaler = 0;
        const auto spatial = read(mode, true, 1, true);
        mode.upscaler = 1;
        const auto dlss = read(mode, true, 1, true);
        for (auto value : dlss) if (!std::isfinite(XMConvertHalfToFloat(value)))
            throw std::runtime_error("Nonfinite output at the 99/100/101% boundary.");
        if (scale >= 100 && spatial != dlss) throw std::runtime_error("Upscaler selection changed native/supersampled output.");
    }
    std::cout << "PASS real NR/DLSS at 99/100/101%, both transfers, finite output and exact upscaler bypass at/above native\n";
    mode = settings; mode.estimateMotion = false; mode.passes = 2; mode.upscaler = 0;
    mode.compareLabels = false; mode.inputColour = 1;
    // A dark/bright boundary gives the filters an edit spanning a strong brightness edge.
    for (UINT y = 0; y < processor.height; ++y) for (UINT x = 0; x < processor.width; ++x) {
        const float v = ((x / 31 + y / 47) % 2) ? .8f : .006f;
        for (unsigned ch = 0; ch < 3; ++ch) pixels[(size_t(y) * processor.width + x) * 4 + ch] = XMConvertFloatToHalf(v);
    }
    upload();
    bool haloChanged = false;
    for (unsigned scale : {50u, 100u, 125u}) {
        mode.modelScale = scale; mode.transfer = 1;
        const auto baseline = read(mode, false, 1, true);
        mode.transfer = 2;
        const auto edge = read(mode, false, 1, true);
        if ((scale < 100) == (edge == baseline)) throw std::runtime_error("Edge-aware activation boundary is incorrect.");
        mode.upscaler = 1;
        if (read(mode, false, 1, true) != edge) throw std::runtime_error("Edge-aware unexpectedly used private DLSS.");
        mode.upscaler = 0;
        for (unsigned side : {0u, 1u}) {
            mode.haloSuppression = side == 0 ? 1.f : 0.f;
            mode.haloBrightSuppression = side == 1 ? 1.f : 0.f;
            for (float radius : {1.f, 4.f}) {
                mode.haloRadius = radius;
                const auto filtered = read(mode, false, 1, true);
                haloChanged |= filtered != edge;
            }
        }
        mode.applyModel = false;
        if (read(mode, false, 1, true) != pixels) throw std::runtime_error("Halo/edge filters changed hidden-edit output.");
        mode.applyModel = true; mode.haloSuppression = mode.haloBrightSuppression = 0;
        if (read(mode, false, 1, true) != edge) throw std::runtime_error("Disabling halos did not restore identical output.");
        mode.transfer = 1;
        if (read(mode, false, 1, true) != baseline) throw std::runtime_error("Disabling edge-aware did not restore identical output.");
    }
    if (!haloChanged) throw std::runtime_error("Halo controls did not change any output.");
    mode.modelScale = 50; mode.transfer = 2; mode.inputColour = 2;
    mode.haloSuppression = mode.haloBrightSuppression = 1;
    for (unsigned tone = 0; tone < 5; ++tone) { mode.tone = tone; read(mode, true, 1, true); }
    auto saved = settings; saved.transfer = 2;
    const auto file = std::filesystem::temp_directory_path() / std::format(L"nr-edge-settings-{}-{}.ini", GetCurrentProcessId(), GetTickCount64());
    SaveSettings(saved, file); const bool restored = LoadSettings(file) == saved; std::filesystem::remove(file);
    if (!restored) throw std::runtime_error("Edge-aware settings failed to round trip.");
    pixels = original; upload();
    std::cout << "PASS edge-aware and both halo controls: actual pixel changes, 50/100/125%, two passes, SDR/five HDR mappings, exact disable/hidden-edit output and saved settings\n";
    HdrOutputCheck(processor, settings, pixels, upload, read);
}
}
