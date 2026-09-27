#pragma once

// tests/xefg_unlock/stubs/Config.h - MINIMAL Config stub for the T7 unlock/pacing tests.
//
// The REAL OptiScaler/Config.h is the global settings surface (CustomOptional,
// ini load/save, hundreds of options). The headers under test only touch three
// opt-in XeFG options plus the shared sanity bound:
//
//   Config::Instance()->FGXeFGUnlockEnabled.value_or_default()      (default false)
//   Config::Instance()->FGXeFGMaxInterpolatedFrames.value_or_default() (default 5)
//   Config::Instance()->FGXeFGExtraPacing.value_or_default()        (default true)
//   Config::XeFGMaxInterpolations                                   (31)
//
// The stub mirrors those production defaults so OFF-by-default behaviour is what
// the test drives, but every functional case below sets the values explicitly.
// The REAL production defaults are asserted separately by source-text pins
// (PIN_DEFAULT_*), never trusted from this stub. Test-only; production code is
// untouched.

#include <cstdint>

struct XeFGUnlockTestBool
{
    bool v;
    bool value_or_default() const { return v; }
};

struct XeFGUnlockTestInt
{
    int v = 5;
    int value_or_default() const { return v; }
};

class Config
{
  public:
    static Config* Instance()
    {
        static Config instance;
        return &instance;
    }

    static constexpr int32_t XeFGMaxInterpolations = 31;

    XeFGUnlockTestBool FGXeFGUnlockEnabled { false };      // { false } in production
    XeFGUnlockTestInt FGXeFGMaxInterpolatedFrames; // { 5 } in production
    XeFGUnlockTestBool FGXeFGExtraPacing { true }; // { true } in production
};
