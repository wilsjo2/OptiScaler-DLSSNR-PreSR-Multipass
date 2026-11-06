#include "PassResolutionCheck.h"
#include <DirectXPackedVector.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace nr {
struct PassResolutionProbe {
    static bool HistoryReset(const Processor& processor, unsigned pass) {
        return processor.stages.at(pass)->historyReset;
    }
    static void ReuseMotion(Processor& processor, bool enabled) { processor.reuseChainMotion = enabled; }
    static bool SharesMotion(const Processor& processor) {
        return processor.stages.at(0)->usedMotionVectors == processor.stages.at(1)->usedMotionVectors;
    }
};
namespace {
using DirectX::PackedVector::XMConvertFloatToHalf;
using DirectX::PackedVector::XMConvertHalfToFloat;
using Pixels = std::vector<uint16_t>;
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
DlssNr::Spatial::Settings Foveation(float centerX, float centerY, float workX, float workY,
                                  float offsetX, float offsetY) {
    DlssNr::Spatial::Settings result;
    result.enabled = true; result.centerX = centerX; result.centerY = centerY;
    result.workX = workX; result.workY = workY; result.offsetX = offsetX; result.offsetY = offsetY;
    return result;
}
void CheckExtents(const Processor& processor, const Settings& settings) {
    for (unsigned pass = 0; pass < settings.passes; ++pass) {
        const auto resolution = settings.PassResolution(pass);
        const auto layout = DlssNr::Spatial::Build(resolution.spatial, processor.width, processor.height,
                                                 float(resolution.modelScale) / 100);
        Require(!resolution.spatial.enabled || layout.active, "Test selected an invalid foveation layout.");
        const auto actual = processor.PassModelSize(pass);
        Require(actual.first == layout.modelW && actual.second == layout.modelH,
                "A pass evaluated at the wrong model dimensions.");
        if (layout.active) Require(uint64_t(actual.first) * actual.second < uint64_t(layout.ordinaryW) * layout.ordinaryH,
                                  "Foveation did not reduce the actual model pixel count.");
        std::cout << std::format("  Pass {}: model {}x{}, ordinary {}x{}, foveation {}\n",
            pass + 1, actual.first, actual.second, layout.ordinaryW, layout.ordinaryH, layout.active);
    }
}
void SettingsCheck(const Settings& baseline) {
    Settings saved = baseline;
    saved.unlockPasses = true; saved.passes = 3;
    saved.modelScale = 75; saved.transfer = 0; saved.upscaler = 1; saved.downscaler = 6;
    saved.spatial = Foveation(40, 50, 75, 85, 5, -5);
    saved.passResolutionOverrides[0] = saved.PassResolution(1);
    saved.passResolutionOverrides[0]->modelScale = 125;
    saved.passResolutionOverrides[0]->upscaler = 0;
    saved.passResolutionOverrides[0]->transfer = 2;
    saved.passResolutionOverrides[0]->downscaler = 2;
    saved.passResolutionOverrides[0]->spatial = Foveation(25, 35, 60, 70, -8, 8);
    saved.passResolutionOverrides[8] = saved.PassResolution(9);
    saved.passResolutionOverrides[8]->modelScale = 150;
    saved.passResolutionOverrides[8]->spatial.shiftX = 1;
    saved.passResolutionOverrides[8]->spatial.shiftY = -1;
    const auto file = std::filesystem::temp_directory_path() /
        std::format(L"nr-pass-resolution-{}-{}.ini", GetCurrentProcessId(), GetTickCount64());
    struct Cleanup {
        std::filesystem::path file;
        ~Cleanup() { std::error_code error; std::filesystem::remove(file, error); }
    } cleanup{file};
    SaveSettings(saved, file);
    Require(LoadSettings(file) == saved, "Per-pass resolution/foveation settings failed to round trip.");
    saved.showFoveation = true; saved.foveationPreviewPass = 9;
    SaveSettings(saved, file);
    const auto transient = LoadSettings(file);
    saved.showFoveation = false; saved.foveationPreviewPass = 0;
    Require(transient == saved, "Wireframe preview leaked into persistent settings.");
    saved.passResolutionOverrides[0].reset();
    SaveSettings(saved, file);
    const auto inherited = LoadSettings(file);
    Require(inherited == saved && inherited.PassResolution(1) == inherited.PassResolution(0),
            "Cleared pass override did not restore inheritance after reload.");
    Require(!inherited.HasPassResolutionOverrides(), "Inactive pass-10 override activated the independent pipeline.");
    saved.modelScale = 100;
    Require(saved.PassResolution(1).modelScale == 100 && saved.PassResolution(9).modelScale == 150,
            "Pass-1 edits did not propagate exclusively to inherited resolution profiles.");
    std::cout << "PASS resolution/foveation persistence, inactive profiles, independent inheritance and session-only wireframe\n";
}
}

void PassResolutionCheck(Processor& processor, const Settings& settings, std::vector<uint16_t>& pixels,
    const std::function<void()>& upload,
    const std::function<std::vector<uint16_t>(Settings, bool, float, bool)>& read,
    const std::filesystem::path& directory, PassResolutionPreviewWriter savePreview) {
    // Vary alpha independently of colour so every readback verifies source-alpha preservation.
    for (UINT y = 0; y < processor.height; ++y) for (UINT x = 0; x < processor.width; ++x)
        pixels[(size_t(y) * processor.width + x) * 4 + 3] =
            XMConvertFloatToHalf(.25f + .25f * float((x / 17 + y / 13) % 4));
    upload();
    const auto original = pixels;
    Settings base; base.runtime = settings.runtime; base.model = settings.model;
    base.passes = 2; base.inputColour = 1; base.compareLabels = false;
    SettingsCheck(base);
    std::filesystem::create_directories(directory);
    const auto save = [&](const char* name, const Pixels& values) {
        savePreview(directory / name, values, processor.width, processor.height);
    };
    save("source.png", original);
    const auto legacy = read(base, false, 1, true);
    Require(legacy != original, "NR is inactive in the pass-resolution fixture.");
    auto inactive = base;
    inactive.passResolutionOverrides[8] = inactive.PassResolution(9);
    inactive.passResolutionOverrides[8]->modelScale = 50;
    inactive.passResolutionOverrides[8]->spatial = Foveation(30, 40, 70, 80, -5, 5);
    Require(read(inactive, false, 1, true) == legacy, "An inactive resolution/foveation override changed output.");

    auto mixed = base;
    mixed.modelScale = 50; mixed.transfer = 0;
    mixed.passResolutionOverrides[0] = mixed.PassResolution(1);
    mixed.passResolutionOverrides[0]->modelScale = 125;
    mixed.passResolutionOverrides[0]->downscaler = 6;
    mixed.passResolutionOverrides[0]->transfer = 1;
    const auto differentSizes = read(mixed, false, 1, true);
    Require(differentSizes != legacy, "Independent model sizes made no change.");
    CheckExtents(processor, mixed); save("mixed-resolution.png", differentSizes);
    auto variant = mixed; variant.passResolutionOverrides[0]->modelScale = 100;
    Require(read(variant, false, 1, true) != differentSizes, "Pass-2 resolution override had no effect.");
    variant = mixed; variant.passResolutionOverrides[0]->downscaler = 1;
    Require(read(variant, false, 1, true) != differentSizes, "Pass-2 downsampling filter override had no effect.");
    Require(read(base, false, 1, true) == legacy, "Disabling resolution overrides changed legacy output.");
    std::cout << "PASS actual independent 50/125% dimensions, pass-2 scale/filter effects, inactive isolation and exact legacy restoration\n";

    mixed.passes = 3; mixed.unlockPasses = true;
    mixed.passResolutionOverrides[1] = mixed.PassResolution(2);
    mixed.passResolutionOverrides[1]->modelScale = 67;
    mixed.passResolutionOverrides[1]->transfer = 1;
    mixed.passResolutionOverrides[1]->upscaler = 1;
    const auto privateDlss = read(mixed, false, 1, true);
    Require(privateDlss != original, "Mixed-pass private DLSS returned the unmodified source.");
    CheckExtents(processor, mixed);
    variant = mixed; variant.passResolutionOverrides[1]->upscaler = 0;
    const auto bilinear = read(variant, false, 1, true);
    Require(bilinear != privateDlss, "Pass-3 private DLSS selection had no effect.");
    variant.passResolutionOverrides[1]->transfer = 2;
    const auto edge = read(variant, false, 1, true);
    Require(edge != bilinear, "Pass-3 edge-aware transfer selection had no effect.");
    variant.passResolutionOverrides[1]->upscaler = 1;
    Require(read(variant, false, 1, true) == edge, "Edge-aware pass unexpectedly used private DLSS.");
    std::cout << "PASS three-pass 50/125/67% chain, independent private DLSS/bilinear/edge-aware reconstruction and finite alpha-preserving output\n";

    auto shared = base;
    shared.spatial = Foveation(40, 50, 70, 75, 5, -5);
    const auto sharedWarp = read(shared, false, 1, true);
    Require(sharedWarp != legacy, "Shared-profile foveation had no effect.");
    CheckExtents(processor, shared); save("shared-foveation.png", sharedWarp);
    variant = shared; variant.strength = 0;
    Require(read(variant, false, 1, true) == original, "Shared-profile foveation changed the zero-strength source.");
    variant = shared; variant.enabled = false;
    Require(read(variant, false, 1, true) == original, "Disabled shared-profile foveation changed the source.");
    variant = shared; variant.showFoveation = true; variant.foveationPreviewPass = 9;
    variant.passResolutionOverrides[8] = variant.PassResolution(9);
    variant.passResolutionOverrides[8]->spatial = Foveation(25, 35, 60, 70, -8, 8);
    const auto inactiveGrid = read(variant, false, 1, true);
    Require(inactiveGrid != sharedWarp, "Selecting an inactive pass did not preview its wireframe.");
    save("wireframe-inactive-pass-10.png", inactiveGrid);
    variant.showFoveation = false;
    Require(read(variant, false, 1, true) == sharedWarp, "Previewing an inactive pass changed active processing.");
    std::cout << "PASS shared-profile foveation dimensions/effect, exact disabled/zero bypass and inactive-pass preview\n";

    auto foveated = shared;
    foveated.passResolutionOverrides[0] = foveated.PassResolution(1);
    foveated.passResolutionOverrides[0]->modelScale = 125;
    foveated.passResolutionOverrides[0]->spatial = Foveation(25, 35, 60, 70, -8, 8);
    const auto warped = read(foveated, false, 1, true);
    CheckExtents(processor, foveated); save("foveated.png", warped);
    variant = foveated; variant.passResolutionOverrides[0]->spatial.enabled = false;
    Require(read(variant, false, 1, true) != warped, "Pass-2 foveation toggle had no effect.");
    CheckExtents(processor, variant);
    variant = foveated; variant.passResolutionOverrides[0]->spatial.offsetX = 10;
    Require(read(variant, false, 1, true) != warped, "Pass-2 foveation centre had no effect.");
    Require(read(foveated, false, 1, true) == warped, "Restoring per-pass foveation did not restore the reset-frame output.");
    variant = foveated; variant.showFoveation = true; variant.foveationPreviewPass = 0;
    const auto grid0 = read(variant, false, 1, true);
    Require(grid0 != warped, "Show area did not draw its wireframe.");
    variant.foveationPreviewPass = 1;
    const auto grid1 = read(variant, false, 1, true);
    Require(grid1 != grid0 && grid1 != warped, "Show area did not follow the selected pass's foveation.");
    save("wireframe-pass-1.png", grid0); save("wireframe-pass-2.png", grid1);
    variant.showFoveation = false;
    Require(read(variant, false, 1, true) == warped, "Hiding the wireframe changed the processed image.");
    Require(read(base, false, 1, true) == legacy, "Leaving foveation did not restore legacy output.");
    std::cout << "PASS separate foveation layouts, reduced actual model work, centre movement, selected-pass wireframe and exact preview/legacy restoration\n";

    variant = foveated; variant.estimateMotion = true;
    for (unsigned frame = 0; frame < 3; ++frame) {
        read(variant, false, 1, frame == 0);
        Require(processor.historyReset == (frame == 0), "Per-pass foveated motion history reset unexpectedly.");
    }
    variant.passResolutionOverrides[0]->spatial.offsetX = 10;
    read(variant, false, 1, false);
    Require(processor.historyReset, "Moving a pass's foveation centre did not reset temporal history.");
    read(variant, false, 1, false);
    Require(!processor.historyReset, "Changed foveation layout did not resume temporal accumulation.");
    Require(!PassResolutionProbe::HistoryReset(processor, 1), "Pass 2 history did not settle before the upstream-change check.");
    // Pass 2 has an explicit profile: only pass 1's feature/layout changes here.
    variant.spatial.offsetY = 10;
    read(variant, false, 1, false);
    Require(PassResolutionProbe::HistoryReset(processor, 0), "Changing pass 1's layout did not reset its history.");
    Require(PassResolutionProbe::HistoryReset(processor, 1), "Pass 1's layout reset did not propagate to pass 2.");
    read(variant, false, 1, false);
    Require(!processor.historyReset, "The pass chain did not resume temporal accumulation after the upstream reset.");
    std::cout << "PASS per-pass foveated motion history, local/upstream layout resets and resumed accumulation\n";

    // Run the same moving sequence with independent estimates and exact reuse.
    // Distinct foveation layouts share ordinary motion geometry, but retain their
    // own model history and warped guides. A local reset still consumes zeros.
    std::vector<Pixels> independentFrames;
    for (bool reuse : {false, true}) {
        pixels = original; upload(); read(base, false, 1, true);
        PassResolutionProbe::ReuseMotion(processor, reuse);
        variant = foveated; variant.estimateMotion = true; variant.modelScale = 50;
        variant.passResolutionOverrides[0]->modelScale = 50;
        for (unsigned frame = 0; frame < 6; ++frame) {
            for (UINT y = 0; y < processor.height; ++y) for (UINT x = 0; x < processor.width; ++x) {
                const size_t target = (size_t(y) * processor.width + x) * 4;
                const size_t source = (size_t((y + frame * 2) % processor.height) * processor.width +
                                       (x + frame * 3) % processor.width) * 4;
                for (unsigned channel = 0; channel < 3; ++channel) pixels[target + channel] = original[source + channel];
            }
            upload();
            if (frame == 3) variant.passResolutionOverrides[0]->spatial.offsetX = 10;
            const auto result = read(variant, false, 1, frame == 0);
            if (!reuse) independentFrames.push_back(result);
            else Require(result == independentFrames.at(frame), "Shared capture motion changed the per-pass temporal output.");
            if (frame == 3) Require(!PassResolutionProbe::HistoryReset(processor, 0) &&
                PassResolutionProbe::HistoryReset(processor, 1), "A local foveation reset changed the upstream history.");
            else if (frame > 0 && reuse)
                Require(PassResolutionProbe::SharesMotion(processor), "Equal geometry did not reuse its capture motion estimate.");
        }
    }
    PassResolutionProbe::ReuseMotion(processor, true);
    pixels = original; upload(); read(base, false, 1, true);
    std::cout << "PASS exact moving-image output with shared motion, independent foveation, local reset and resumed history\n";

    for (unsigned bypass = 0; bypass < 4; ++bypass) {
        variant = foveated;
        if (bypass == 0) variant.strength = 0;
        if (bypass == 1) variant.enabled = false;
        if (bypass == 2) variant.applyModel = false;
        if (bypass == 3) {
            variant.separateSkin = true;
            variant.skinDetail = variant.skinColour = variant.environmentDetail = variant.environmentColour = 0;
        }
        Require(read(variant, false, 1, true) == original, "Per-pass/foveated final composition bypass changed the source.");
    }
    variant = foveated; variant.holdFrame = true;
    const auto held = read(variant, false, 1, true);
    for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
        pixels[i] = XMConvertFloatToHalf(XMConvertHalfToFloat(original[i]) * .25f);
    upload();
    Require(read(variant, false, 1, true) == held, "Foveated hold followed the changing source.");
    variant.passResolutionOverrides[0]->modelScale = 150;
    variant.applyModel = false;
    Require(read(variant, false, 1, true) == original, "Changing pass resolution discarded the held original.");
    variant.holdFrame = false;
    Require(read(variant, false, 1, true) == pixels, "Leaving hold did not restore the live source.");
    pixels = original; upload();
    std::cout << "PASS exact final strength/enable/visibility/skin bypasses and held-source preservation across per-pass resizing\n";

    for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
        pixels[i] = XMConvertFloatToHalf(XMConvertHalfToFloat(original[i]) * 4);
    pixels[0] = pixels[1] = pixels[2] = XMConvertFloatToHalf(4);
    upload();
    variant = foveated; variant.inputColour = 2;
    for (unsigned tone : {2u, 3u}) {
        variant.tone = tone;
        const auto hdr = read(variant, true, 1, true);
        Require(hdr != pixels, "Per-pass foveated HDR did not apply NR.");
        float maximum = 0;
        for (size_t i = 0; i < hdr.size(); ++i) if (i % 4 != 3)
            maximum = std::max(maximum, XMConvertHalfToFloat(hdr[i]));
        Require(maximum > 1, "Per-pass foveation clipped HDR to SDR.");
        variant.strength = 0;
        Require(read(variant, true, 1, true) == pixels, "Per-pass foveated HDR bypass changed the source.");
        variant.strength = 1;
    }
    pixels = original; upload();
    read(base, false, 1, true);
    std::cout << "PASS per-pass foveation in SDR/HDR, soft-knee/replace mapping, finite output, alpha and exact HDR bypass\n"
        << "PASS per-pass resolution and foveation checks; previews: " << directory.string() << '\n';
}
}
