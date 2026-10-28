#include "FeatureChecks.h"
#include <DirectXPackedVector.h>
#include <iostream>
namespace nr {
void HdrOutputCheck(Processor& processor, const Settings& settings, std::vector<uint16_t>& pixels,
    const std::function<void()>& upload,
    const std::function<std::vector<uint16_t>(Settings, bool, float, bool)>& read) {
    using DirectX::PackedVector::XMConvertHalfToFloat;
    using DirectX::PackedVector::XMConvertFloatToHalf;
    const auto original = pixels;
    const unsigned width = processor.width, height = processor.height;
    constexpr float levels[]{0, .001f, .01f, .1f, .5f, 1, 4, 8, 16, 64};
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const float level = levels[std::min(9u, x * 10 / width)];
        for (unsigned c = 0; c < 3; ++c)
            pixels[(size_t(y) * width + x) * 4 + c] = XMConvertFloatToHalf(level * (y < height / 2 ? 1.f : 1.f / (c + 1)));
    }
    upload();
    auto mode = settings; mode.modelScale = 100; mode.upscaler = 0; mode.estimateMotion = false; mode.passes = 1;
    mode.inputColour = 2; mode.referenceWhiteNits = 320; mode.compareLabels = false;
    auto sample = [&](Settings value, bool hdr = true) { return read(value, hdr, 4, true); };
    for (unsigned tone = 0; tone < 5; ++tone) {
        mode.tone = tone; mode.peakBrightnessNits = mode.blackLevelNits = 0;
        const auto baseline = sample(mode);
        float sourcePeak = 0;
        for (size_t i = 0; i < baseline.size(); ++i) if (i % 4 != 3)
            sourcePeak = std::max(sourcePeak, XMConvertHalfToFloat(baseline[i]));
        if (sourcePeak <= 0) throw std::runtime_error("HDR fixture has no positive highlight.");
        // Different NR runtimes can keep the whole image below 1000 nits.
        // Place the test ceiling below this actual output so roll-off is exercised.
        mode.peakBrightnessNits = std::min(1000.f, sourcePeak * 80.f * .8f);
        const float ceiling = mode.peakBrightnessNits / 80.f;
        std::cout << std::format("HDR mode {}: source peak {:.1f} nits, test ceiling {:.1f} nits\n", tone, sourcePeak * 80, mode.peakBrightnessNits);
        const auto peak = sample(mode);
        bool rolled = false;
        for (size_t i = 0; i < peak.size(); i += 4) {
            float brightest = 0;
            for (unsigned c = 0; c < 3; ++c) brightest = std::max(brightest, XMConvertHalfToFloat(baseline[i + c]));
            for (unsigned c = 0; c < 3; ++c) {
                const float a = XMConvertHalfToFloat(baseline[i + c]), b = XMConvertHalfToFloat(peak[i + c]);
                if (b > ceiling || b > a) throw std::runtime_error("HDR peak control exceeded its ceiling or brightened a pixel.");
                if (brightest <= ceiling * .75f && peak[i + c] != baseline[i + c]) throw std::runtime_error("HDR peak changed pixels below its shoulder.");
                rolled |= b < a;
            }
        }
        if (!rolled) throw std::runtime_error("HDR peak control did not affect highlights.");
        mode.peakBrightnessNits = 0; mode.blackLevelNits = .5f;
        const auto lifted = sample(mode);
        mode.blackLevelNits = -.5f;
        const auto darkened = sample(mode);
        bool liftChanged = false, darkChanged = false;
        for (size_t i = 0; i < baseline.size(); i += 4) {
            const float luminance = .2126f * XMConvertHalfToFloat(baseline[i]) + .7152f * XMConvertHalfToFloat(baseline[i + 1]) + .0722f * XMConvertHalfToFloat(baseline[i + 2]);
            for (unsigned c = 0; c < 3; ++c) {
                const float a = XMConvertHalfToFloat(baseline[i + c]), b = XMConvertHalfToFloat(lifted[i + c]), d = XMConvertHalfToFloat(darkened[i + c]);
                if (b < a || d > a || d < 0) throw std::runtime_error("HDR black control moved shadows in the wrong direction.");
                if (luminance >= 4 && (lifted[i + c] != baseline[i + c] || darkened[i + c] != baseline[i + c]))
                    throw std::runtime_error("Black adjustment changed reference white/highlights.");
                liftChanged |= b > a; darkChanged |= d < a;
            }
        }
        if (!liftChanged || !darkChanged) throw std::runtime_error("HDR black control had no shadow effect.");
        mode.blackLevelNits = 0;
        if (sample(mode) != baseline) throw std::runtime_error("Neutral HDR settings did not restore exact pixels.");
    }
    std::cout << "PASS all five HDR modes: peak roll-off, unchanged lower tones, shadow lift/darken, reference white and neutral round trip\n";

    mode.tone = 0;
    for (unsigned comparison : {2u, 3u}) for (bool swap : {false, true}) {
        mode.compare = comparison; mode.compareSwap = swap; mode.compareSplit = .3f;
        const auto baseline = sample(mode);
        mode.peakBrightnessNits = 1000; mode.blackLevelNits = .5f;
        const auto tuned = sample(mode);
        if (tuned == baseline) throw std::runtime_error("HDR adjustment did not reach the processed comparison side.");
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
            const float split = comparison == 3 ? .5f : .3f, uvx = (x + .5f) / width, uvy = (y + .5f) / height;
            const bool untouched = ((uvx < split) != swap) || std::abs(uvx - split) < 1.f / width ||
                (comparison == 3 && (uvy < .25f || uvy > .75f));
            if (untouched) for (unsigned c = 0; c < 4; ++c) {
                const auto i = (size_t(y) * width + x) * 4 + c;
                if (tuned[i] != baseline[i]) throw std::runtime_error("HDR adjustment changed comparison originals, bars or divider.");
            }
        }
        mode.peakBrightnessNits = mode.blackLevelNits = 0;
    }
    mode.compare = 0; mode.compareSwap = false;
    for (unsigned bypass = 0; bypass < 7; ++bypass) {
        auto check = mode;
        if (bypass == 0) check.enabled = false;
        if (bypass == 1) check.strength = 0;
        if (bypass == 2) check.compare = 1;
        if (bypass == 3) check.applyModel = false;
        if (bypass == 4) check.debugView = 1;
        if (bypass == 5) { check.separateSkin = true; check.previewSkin = true; }
        const auto baseline = sample(check, bypass != 6);
        check.peakBrightnessNits = 200; check.blackLevelNits = 2;
        if (sample(check, bypass != 6) != baseline) throw std::runtime_error("HDR controls changed a bypass/debug/SDR-display view.");
    }
    mode.tone = 3; mode.modelScale = 70; mode.upscaler = 1; mode.transfer = 1; mode.passes = 2;
    mode.estimateMotion = true; mode.peakBrightnessNits = 1000; mode.blackLevelNits = .05f;
    for (unsigned frame = 0; frame < 3; ++frame) {
        const auto result = read(mode, true, 4, frame == 0);
        for (size_t i = 0; i < result.size(); ++i) if (i % 4 != 3 && XMConvertHalfToFloat(result[i]) > 12.5f)
            throw std::runtime_error("Private DLSS bypassed the HDR output peak.");
    }
    pixels = original; upload();
    std::cout << "PASS HDR controls preserve original comparison, swapped sides, letterbox/divider, bypass, debug and SDR-display pixels\n";
    std::cout << "PASS HDR output controls with two NR passes, estimated motion, 70% private DLSS and Hybrid replace\n";
}
}
