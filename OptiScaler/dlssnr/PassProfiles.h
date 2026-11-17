#pragma once
#include <Config.h>
#include "DlssNr_ModelParameters.h"
#include "NrNvfp4.h"
#include <shaders/dlssnr/DlssNr_Spatial.h>
#include <algorithm>
#include <cmath>

namespace DlssNr::Profiles
{
// The model a pass asks for: the runtime's presets 0..3, or one of DlssNr::Nvfp4's (4..6: Model A with the 4-bit
// kernels, or a pass-in-one model's weight set). PassSettings gives the runtime preset 1 for those.
inline unsigned int RequestedPreset(const Config& cfg, unsigned int pass)
{
    unsigned int preset = cfg.DlssNrPreset.value_or_default();
    if (pass > 0 && pass <= 2 && pass <= std::size(cfg.DlssNrPassOverrides))
        preset = cfg.DlssNrPassOverrides[pass - 1].preset.value_or(preset);
    return preset;
}

// Whether any of the passes asks for one of DlssNr::Nvfp4's models.
inline bool WantsNvfp4(const Config& cfg, unsigned int passes)
{
    for (unsigned int pass = 0; pass < passes; ++pass)
        if (Nvfp4::IsNvfp4(RequestedPreset(cfg, pass)))
            return true;
    return false;
}

inline ModelSettings PassSettings(const Config& cfg, unsigned int pass)
{
    ModelSettings result { cfg.DlssNrPreset.value_or_default(),
                           cfg.DlssNrStyle.value_or_default(),
                           cfg.DlssNrIntensity.value_or_default(),
                           cfg.DlssNrLocalStructure.value_or_default(),
                           pass == 0 ? cfg.DlssNrLocalTone.value_or_default() : 0.0f,
                           cfg.DlssNrSkinStructure.value_or_default(),
                           cfg.DlssNrAutoMask.value_or_default() };
    if (pass > 0 && pass <= std::size(cfg.DlssNrPassOverrides))
    {
        const auto& extra = cfg.DlssNrPassOverrides[pass - 1];
        if (pass <= 2) // Only the legacy pass 2/3 keys have a preset override.
            result.preset = extra.preset.value_or(result.preset);
        result.style = extra.style.value_or(result.style);
        result.intensity = extra.intensity.value_or(result.intensity);
        result.localStructure = extra.structure.value_or(result.localStructure);
        result.localTone = extra.tone.value_or(result.localTone);
        result.skinStructure = extra.skin.value_or(result.skinStructure);
        result.autoMask = extra.autoMask.value_or(result.autoMask);
    }
    result.requested = Nvfp4::IsNvfp4(result.preset) ? result.preset : 0u;
    result.preset = result.requested ? 1u : std::min(result.preset, 3u);
    result.style = std::min(result.style, 2u);
    const auto bounded = [](float value, float fallback, float minimum)
    { return std::isfinite(value) ? std::clamp(value, minimum, 2.0f) : fallback; };
    result.intensity = bounded(result.intensity, 1.0f, 0.0f);
    result.localStructure = bounded(result.localStructure, 1.0f, 0.0f);
    result.localTone = bounded(result.localTone, pass == 0 ? 1.0f : 0.0f, 0.0f);
    result.skinStructure = bounded(result.skinStructure, -1.0f, -1.0f);
    return result;
}

struct ResolutionSettings
{
    float workingScale = 1.0f;
    Scaler scalingDownscaler = Scaler::Lanczos3;
    uint32_t transfer = 1;
    int privateUpscaler = 0;
    Spatial::Settings spatial;

    bool operator==(const ResolutionSettings&) const = default;
};

inline bool HasPassResolutionOverride(const Config::NrPassOverrides& extra)
{
    return extra.workingScale.has_value() || extra.scalingDownscaler.has_value() || extra.transfer.has_value() ||
           extra.privateUpscaler.has_value() || extra.spatialCompression.has_value() ||
           extra.spatialCenterX.has_value() || extra.spatialCenterY.has_value() || extra.spatialWorkX.has_value() ||
           extra.spatialWorkY.has_value() || extra.spatialOffsetX.has_value() || extra.spatialOffsetY.has_value() ||
           extra.spatialShiftX.has_value() || extra.spatialShiftY.has_value();
}

// A re-applying pass skips the network and repeats the previous network pass's change; pass 1 always runs it.
inline bool PassReapplies(const Config& cfg, unsigned int pass)
{
    return pass > 0 && pass <= std::size(cfg.DlssNrPassOverrides) &&
           cfg.DlssNrPassOverrides[pass - 1].reapply.value_or(false);
}

inline float ReapplyGain(const Config& cfg)
{
    const float gain = cfg.DlssNrReapplyGain.value_or_default();
    return std::isfinite(gain) ? std::clamp(gain, 0.0f, 2.0f) : 0.81f;
}

// Consecutive re-applying passes from `pass` (inclusive) up to `end` (exclusive).
inline unsigned int ReapplyRun(const Config& cfg, unsigned int pass, unsigned int end)
{
    unsigned int run = 0;
    while (pass + run < end && PassReapplies(cfg, pass + run))
        ++run;
    return run;
}

// The borrow tick-boxes of a network pass. NrPassBorrow::ChooseCut turns them into a cut; re-apply wins over both.
struct BorrowChoice
{
    bool transformer = false, below = false;
    unsigned int level = 32;
};

inline BorrowChoice PassBorrow(const Config& cfg, unsigned int pass)
{
    if (pass == 0 || pass > std::size(cfg.DlssNrPassOverrides) || PassReapplies(cfg, pass))
        return {};
    const auto& extra = cfg.DlssNrPassOverrides[pass - 1];
    return { extra.borrowTransformer.value_or(false), extra.borrowBelow.value_or(false),
             extra.borrowLevel.value_or(32u) };
}

// A re-applying pass runs no network, so its resolution profile is ignored.
inline bool HasPassResolutionOverrides(const Config& cfg, unsigned int requestedPasses)
{
    const auto count = std::min(requestedPasses, (unsigned int) std::size(cfg.DlssNrPassOverrides) + 1);
    for (unsigned int pass = 1; pass < count; ++pass)
        if (!PassReapplies(cfg, pass) && HasPassResolutionOverride(cfg.DlssNrPassOverrides[pass - 1]))
            return true;
    return false;
}

// Every absent field inherits pass 1, independently of the model-tuning overrides.
inline ResolutionSettings PassResolutionSettings(const Config& cfg, unsigned int pass)
{
    ResolutionSettings result { cfg.DlssNrWorkingScale.value_or_default(),
                                cfg.DlssNrScalingDownscaler.value_or_default(),
                                cfg.DlssNrTransfer.value_or_default(),
                                cfg.DlssNrPrivateUpscaler.value_or_default(), Spatial::ReadSettings(cfg) };
    if (pass > 0 && pass <= std::size(cfg.DlssNrPassOverrides))
    {
        const auto& extra = cfg.DlssNrPassOverrides[pass - 1];
        result.workingScale = extra.workingScale.value_or(result.workingScale);
        result.scalingDownscaler = extra.scalingDownscaler.value_or(result.scalingDownscaler);
        result.transfer = extra.transfer.value_or(result.transfer);
        result.privateUpscaler = extra.privateUpscaler.value_or(result.privateUpscaler);
        result.spatial.enabled = extra.spatialCompression.value_or(result.spatial.enabled);
        result.spatial.centerX = extra.spatialCenterX.value_or(result.spatial.centerX);
        result.spatial.centerY = extra.spatialCenterY.value_or(result.spatial.centerY);
        result.spatial.workX = extra.spatialWorkX.value_or(result.spatial.workX);
        result.spatial.workY = extra.spatialWorkY.value_or(result.spatial.workY);
        result.spatial.offsetX = extra.spatialOffsetX.value_or(result.spatial.offsetX);
        result.spatial.offsetY = extra.spatialOffsetY.value_or(result.spatial.offsetY);
        result.spatial.shiftX = extra.spatialShiftX.value_or(result.spatial.shiftX);
        result.spatial.shiftY = extra.spatialShiftY.value_or(result.spatial.shiftY);
    }
    result.workingScale = std::isfinite(result.workingScale) ? std::clamp(result.workingScale, 0.25f, 2.0f) : 1.0f;
    if (result.scalingDownscaler >= Scaler::Count)
        result.scalingDownscaler = Scaler::Lanczos3;
    result.transfer = std::min(result.transfer, 5u);
    if (result.privateUpscaler < 0 || result.privateUpscaler > 3)
        result.privateUpscaler = 0;
    // Spatial::Build reports invalid INI geometry and uses the ordinary NR path.
    return result;
}

inline void SetPassResolutionSettings(Config::NrPassOverrides& extra, const ResolutionSettings& value)
{
    extra.workingScale = value.workingScale;
    extra.scalingDownscaler = value.scalingDownscaler;
    extra.transfer = value.transfer;
    extra.privateUpscaler = value.privateUpscaler;
    extra.spatialCompression = value.spatial.enabled;
    extra.spatialCenterX = value.spatial.centerX;
    extra.spatialCenterY = value.spatial.centerY;
    extra.spatialWorkX = value.spatial.workX;
    extra.spatialWorkY = value.spatial.workY;
    extra.spatialOffsetX = value.spatial.offsetX;
    extra.spatialOffsetY = value.spatial.offsetY;
    extra.spatialShiftX = value.spatial.shiftX;
    extra.spatialShiftY = value.spatial.shiftY;
}

inline void ClearPassResolutionSettings(Config::NrPassOverrides& extra)
{
    extra.workingScale.reset();
    extra.scalingDownscaler.reset();
    extra.transfer.reset();
    extra.privateUpscaler.reset();
    extra.spatialCompression.reset();
    extra.spatialCenterX.reset();
    extra.spatialCenterY.reset();
    extra.spatialWorkX.reset();
    extra.spatialWorkY.reset();
    extra.spatialOffsetX.reset();
    extra.spatialOffsetY.reset();
    extra.spatialShiftX.reset();
    extra.spatialShiftY.reset();
}

// The caller owns this snapshot; resolving a pass never mutates the shared configuration.
inline void ApplyPassConfig(Config& dest, const Config& source, unsigned int pass)
{
    const auto model = PassSettings(source, pass);
    const auto resolution = PassResolutionSettings(source, pass);
    dest = source;
    dest.DlssNrPreset = model.requested ? model.requested : model.preset; // a 4-bit model stays what was asked
    dest.DlssNrStyle = model.style;
    dest.DlssNrIntensity = model.intensity;
    dest.DlssNrLocalStructure = model.localStructure;
    dest.DlssNrLocalTone = model.localTone;
    dest.DlssNrSkinStructure = model.skinStructure;
    dest.DlssNrAutoMask = model.autoMask;
    dest.DlssNrWorkingScale = resolution.workingScale;
    dest.DlssNrScalingDownscaler = resolution.scalingDownscaler;
    dest.DlssNrTransfer = resolution.transfer;
    dest.DlssNrPrivateUpscaler = resolution.privateUpscaler;
    dest.DlssNrSpatialCompression = resolution.spatial.enabled;
    dest.DlssNrSpatialCenterX = resolution.spatial.centerX;
    dest.DlssNrSpatialCenterY = resolution.spatial.centerY;
    dest.DlssNrSpatialWorkX = resolution.spatial.workX;
    dest.DlssNrSpatialWorkY = resolution.spatial.workY;
    dest.DlssNrSpatialOffsetX = resolution.spatial.offsetX;
    dest.DlssNrSpatialOffsetY = resolution.spatial.offsetY;
    dest.DlssNrSpatialShiftX = resolution.spatial.shiftX;
    dest.DlssNrSpatialShiftY = resolution.spatial.shiftY;
    dest.DlssNrPasses = 1u;
    for (auto& extra : dest.DlssNrPassOverrides)
        extra = Config::NrPassOverrides {};
}
} // namespace DlssNr::Profiles
