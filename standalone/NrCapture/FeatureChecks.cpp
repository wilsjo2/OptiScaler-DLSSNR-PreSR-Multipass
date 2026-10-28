#include "FeatureChecks.h"
#include <DirectXPackedVector.h>
#include <iostream>
namespace nr {
void FeatureChecks(Processor& processor, const Settings& settings, std::vector<uint16_t>& pixels,
    const std::function<void()>& upload,
    const std::function<std::vector<uint16_t>(Settings, bool, float, bool)>& read) {
    using DirectX::PackedVector::XMConvertHalfToFloat;
    using DirectX::PackedVector::XMConvertFloatToHalf;
    const auto original = pixels;
    Settings saved = settings;
    saved.unlockPasses = true; saved.passes = 10; saved.modelScale = 175; saved.downscaler = 6;
    saved.model.style = 2; saved.model.preset = 1; saved.model.autoMask = false;
    saved.highlightGuard = 3; saved.compareSplit = .25f; saved.compareZoom = 1.5f;
    saved.compareSwap = true; saved.compareLabels = false; saved.labelScale = 2;
    saved.peakBrightnessNits = 1000; saved.blackLevelNits = -.05f;
    saved.haloSuppression = .75f; saved.haloBrightSuppression = .25f; saved.haloRadius = 3;
    saved.upscaler = 1; saved.transfer = 0; // DLSS must not silently force matched transfer on load.
    saved.passOverrides[8] = saved.PassModel(9); saved.passOverrides[8]->localTone = .5f;
    const auto file = std::filesystem::temp_directory_path() / std::format(L"nr-feature-settings-{}-{}.ini", GetCurrentProcessId(), GetTickCount64());
    SaveSettings(saved, file);
    const auto restored = LoadSettings(file); std::filesystem::remove(file);
    if (restored != saved) throw std::runtime_error("Feature settings failed to round trip.");
    std::cout << "PASS per-pass, supersampling, comparison and composition settings round trip (isolated INI)\n";
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
    saved.transfer = 2; SaveSettings(saved, file);
    if (LoadSettings(file) != saved) throw std::runtime_error("Edge-aware settings failed to round trip.");
    std::filesystem::remove(file); pixels = original; upload();
    std::cout << "PASS edge-aware and both halo controls: actual pixel changes, 50/100/125%, two passes, SDR/five HDR mappings, exact disable/hidden-edit output and saved settings\n";
    HdrOutputCheck(processor, settings, pixels, upload, read);
}
}
