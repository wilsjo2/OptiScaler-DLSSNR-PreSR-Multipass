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

    // Re-apply and borrow options: pass 1 never has them, re-apply wins over borrowing and over a resolution
    // profile, runs of re-applying passes are counted, and child snapshots never inherit them.
    Config reuse;
    reuse.DlssNrPasses = 5u;
    assert(reuse.DlssNrReapplyGain.value_or_default() == 0.81f && ReapplyGain(reuse) == 0.81f);
    reuse.DlssNrReapplyGain = 3.0f;
    assert(ReapplyGain(reuse) == 2.0f);
    reuse.DlssNrReapplyGain = -1.0f;
    assert(ReapplyGain(reuse) == 0.0f);
    reuse.DlssNrReapplyGain = std::numeric_limits<float>::quiet_NaN();
    assert(ReapplyGain(reuse) == 0.81f);
    for (unsigned pass = 0; pass < 31; ++pass)
        assert(!PassReapplies(reuse, pass) && !PassBorrow(reuse, pass).transformer && !PassBorrow(reuse, pass).below);
    assert(PassBorrow(reuse, 1).level == 32);
    auto& second = reuse.DlssNrPassOverrides[0];
    auto& third = reuse.DlssNrPassOverrides[1];
    auto& fourth = reuse.DlssNrPassOverrides[2];
    second.borrowTransformer = true;
    third.borrowBelow = true;
    third.borrowLevel = 8u;
    assert(PassBorrow(reuse, 1).transformer && !PassBorrow(reuse, 1).below);
    assert(PassBorrow(reuse, 2).below && PassBorrow(reuse, 2).level == 8);
    second.reapply = true;
    third.reapply = true;
    fourth.reapply = false;
    reuse.DlssNrPassOverrides[3].reapply = true;
    assert(!PassReapplies(reuse, 0) && PassReapplies(reuse, 1) && PassReapplies(reuse, 2));
    assert(!PassReapplies(reuse, 3) && PassReapplies(reuse, 4) && !PassReapplies(reuse, 30));
    assert(!PassBorrow(reuse, 1).transformer && !PassBorrow(reuse, 2).below); // re-apply wins
    assert(ReapplyRun(reuse, 1, 5) == 2 && ReapplyRun(reuse, 2, 5) == 1 && ReapplyRun(reuse, 3, 5) == 0);
    assert(ReapplyRun(reuse, 4, 5) == 1 && ReapplyRun(reuse, 1, 2) == 1 && ReapplyRun(reuse, 4, 4) == 0);
    // A re-applying pass runs no network, so its resolution profile does not select the independent chain.
    second.workingScale = 0.5f;
    assert(!HasPassResolutionOverrides(reuse, 5));
    second.reapply = false;
    assert(HasPassResolutionOverrides(reuse, 5));
    second.workingScale.reset();
    Config reuseChild;
    ApplyPassConfig(reuseChild, reuse, 3);
    for (unsigned pass = 0; pass < 31; ++pass)
        assert(!PassReapplies(reuseChild, pass) && !PassBorrow(reuseChild, pass).below);

    std::puts("NR pass profiles: inheritance, independent sizing/spatial settings, isolated snapshots, bounds and "
              "re-apply/borrow options passed.");
}
