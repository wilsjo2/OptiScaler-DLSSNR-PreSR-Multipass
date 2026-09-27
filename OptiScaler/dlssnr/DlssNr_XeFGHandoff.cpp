// ============================================================================
// DlssNr_XeFGHandoff.cpp - compile anchor for the XeFG handoff core.
//
// The handoff tracker is header-only (DlssNr_XeFGHandoff.h) and dependency-free:
// tests/run_xefg_handoff_smoke.ps1 compiles its probe against the header alone,
// so every definition must stay inline. This translation unit exists so the real
// project (MSBuild Release x64) compiles the unit before the Present-site wiring
// of todo 10 includes it in XeFG_Dx12.cpp, and it pins the small ABI/type
// contract the seam relies on. It deliberately includes no pch.h (the project
// marks it PrecompiledHeader=NotUsing) so the seam runner can compile it
// standalone exactly as the project does.
// ============================================================================

#include "DlssNr_XeFGHandoff.h"

#include <type_traits>

namespace
{
using DlssNr::XeFGHandoff::Identity;
using DlssNr::XeFGHandoff::Outcome;
using DlssNr::XeFGHandoff::SkipReason;

// The seam passes Identity by value and copies Outcome; both must stay plain
// aggregates so the Present site can fill them without a constructor.
static_assert(std::is_trivially_copyable_v<Identity>, "handoff Identity must stay trivially copyable");
static_assert(std::is_trivially_copyable_v<Outcome>, "handoff Outcome must stay trivially copyable");

// NR_XEFG_SKIP logs reason=<token> from the enum; None is the "no reason" value
// and 0 is never a valid capture serial, so both must stay 0.
static_assert(static_cast<unsigned>(SkipReason::None) == 0, "SkipReason::None must stay 0");
} // namespace
