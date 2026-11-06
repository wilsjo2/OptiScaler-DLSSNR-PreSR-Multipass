#include "pch.h"
#include <dlssnr/PassProfiles.h>

#include <cassert>
#include <cstdio>
#include <limits>

// Exercise the real Config fields without reading an installation's INI file.
Config::Config() {}

using namespace DlssNr::Profiles;

int main()
{
    Config config;
    config.DlssNrPasses = 3u;
    config.DlssNrWorkingScale = 0.75f;
    config.DlssNrTransfer = 4u;
    config.DlssNrPrivateUpscaler = 1;
    config.DlssNrScalingDownscaler = Scaler::Kaiser3;
    config.DlssNrSpatialCompression = true;
    config.DlssNrSpatialCenterX = 60.0f;
    config.DlssNrSpatialCenterY = 65.0f;
    config.DlssNrSpatialWorkX = 85.0f;
    config.DlssNrSpatialWorkY = 90.0f;
    config.DlssNrSpatialOffsetX = 2.0f;
    config.DlssNrSpatialOffsetY = -3.0f;
    config.DlssNrSpatialShiftX = 1.0f;
    config.DlssNrSpatialShiftY = -1.0f;

    const auto inherited = PassResolutionSettings(config, 0);
    assert(PassResolutionSettings(config, 1) == inherited);
    assert(PassResolutionSettings(config, 29) == inherited);
    assert(!HasPassResolutionOverrides(config, 30));

    // Model tuning alone preserves the legacy shared-resolution pipeline.
    auto& pass2 = config.DlssNrPassOverrides[0];
    pass2.intensity = 0.4f;
    pass2.style = 2u;
    pass2.tone = 0.3f;
    assert(!HasPassResolutionOverrides(config, 30));
    assert(PassResolutionSettings(config, 1) == inherited);

    // Inactive profiles must not change the active pipeline.
    config.DlssNrPassOverrides[2].workingScale = 1.5f;
    assert(!HasPassResolutionOverrides(config, 3));
    assert(HasPassResolutionOverrides(config, 4));
    assert(HasPassResolutionOverrides(config, 100));
    assert(!HasPassResolutionOverrides(config, 0));

    // Partial overrides retain their own inheritance for each field.
    pass2.workingScale = 0.5f;
    pass2.privateUpscaler = 3;
    pass2.spatialCompression = false;
    auto pass2Settings = PassResolutionSettings(config, 1);
    assert(pass2Settings.workingScale == 0.5f && pass2Settings.privateUpscaler == 3);
    assert(pass2Settings.transfer == inherited.transfer);
    assert(pass2Settings.scalingDownscaler == inherited.scalingDownscaler);
    assert(!pass2Settings.spatial.enabled && pass2Settings.spatial.centerX == 60.0f);
    assert(HasPassResolutionOverrides(config, 2));

    ResolutionSettings pass3Settings { 1.75f, Scaler::CatmullRom, 5u, 2,
                                      { true, 45.0f, 50.0f, 70.0f, 80.0f, -4.0f, 5.0f, 2.0f, -2.0f } };
    SetPassResolutionSettings(config.DlssNrPassOverrides[1], pass3Settings);
    assert(PassResolutionSettings(config, 2) == pass3Settings);
    assert(PassResolutionSettings(config, 1) == pass2Settings);
    assert(PassResolutionSettings(config, 0) == inherited);

    // A child owns a single-pass snapshot, including its model tuning and spatial layout.
    Config child;
    ApplyPassConfig(child, config, 1);
    assert(PassResolutionSettings(child, 0) == pass2Settings);
    assert(PassSettings(child, 0) == PassSettings(config, 1));
    assert(child.DlssNrPasses.value_or_default() == 1);
    assert(!HasPassResolutionOverrides(child, 30));
    for (const auto& extra : child.DlssNrPassOverrides)
        assert(!extra.intensity.has_value() && !extra.style.has_value() && !extra.tone.has_value());
    child.DlssNrWorkingScale = 1.9f;
    assert(config.DlssNrPasses.value_or_default() == 3);
    assert(PassResolutionSettings(config, 0) == inherited);
    assert(PassResolutionSettings(config, 1) == pass2Settings);
    assert(config.DlssNrPassOverrides[0].intensity.value() == 0.4f);

    // Alias-safe application resolves before clearing child overrides.
    Config aliased = config;
    ApplyPassConfig(aliased, aliased, 2);
    assert(PassResolutionSettings(aliased, 0) == pass3Settings);
    assert(aliased.DlssNrLocalTone.value_or_default() == 0.0f);
    assert(!HasPassResolutionOverrides(aliased, 30));

    // Turning inheritance back on clears sizing only, preserving model edits.
    ClearPassResolutionSettings(pass2);
    assert(!HasPassResolutionOverride(pass2));
    assert(PassResolutionSettings(config, 1) == inherited);
    assert(pass2.intensity.value() == 0.4f && pass2.style.value() == 2u && pass2.tone.value() == 0.3f);

    // Malformed numeric settings cannot request invalid resource scales or backends.
    pass2.workingScale = std::numeric_limits<float>::quiet_NaN();
    pass2.scalingDownscaler = static_cast<Scaler>(999);
    pass2.transfer = 999u;
    pass2.privateUpscaler = -1;
    auto sanitized = PassResolutionSettings(config, 1);
    assert(sanitized.workingScale == 1.0f && sanitized.scalingDownscaler == Scaler::Lanczos3);
    assert(sanitized.transfer == 5 && sanitized.privateUpscaler == 0);
    pass2.workingScale = 0.01f;
    assert(PassResolutionSettings(config, 1).workingScale == 0.25f);
    pass2.workingScale = 9.0f;
    assert(PassResolutionSettings(config, 1).workingScale == 2.0f);

    std::puts("NR pass profiles: inheritance, independent sizing/spatial settings, isolated snapshots and bounds passed.");
}
