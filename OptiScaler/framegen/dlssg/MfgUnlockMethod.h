#pragma once

// Which temporal fix the configuration selects for the built-in RTX 40 unlock. Header-only and free of
// OptiScaler headers so tests/mfg_method_smoke.cpp can compile it alone.

#include <optional>
#include <string>

namespace MfgUnlock
{
// How generated frames are given their own time between the two real ones. Above 2X the Ada build of the
// interpolation kernel puts every one of them at the midpoint.
//   Retarget: the module's Blackwell image is relabelled to answer for Ada (RewriteBlackwellKernels).
//   Ptx:      the Ada kernel's PTX is rewritten to blend at each frame's own time (MfgUnlockPtx.h).
// One per session: both edit the same fatbin. None means no attempt was made.
enum class TemporalMethod
{
    None,
    Retarget,
    Ptx,
};

// `fix` is [DLSSG] AdaTemporalFix as read ("Auto", "Retarget" or "Ptx"). A named method wins; "Auto", or
// nothing, is Retarget, the method the unlock shipped with.
inline TemporalMethod ResolveTemporalMethod(const std::optional<std::string>& fix)
{
    if (fix.has_value() && *fix == "Ptx")
        return TemporalMethod::Ptx;

    return TemporalMethod::Retarget;
}
} // namespace MfgUnlock
