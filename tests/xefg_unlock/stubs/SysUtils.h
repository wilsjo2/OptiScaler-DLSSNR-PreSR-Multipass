#pragma once

// tests/xefg_unlock/stubs/SysUtils.h - MINIMAL compile stub for the T7 unlock/pacing tests.
//
// The REAL OptiScaler/SysUtils.h pulls the precompiled header, spdlog, Config and the
// whole util surface. The two headers under test (proxies/XeFGUnlock.h,
// proxies/XeFGPacing.h) only need the Windows API surface plus the C/C++ standard
// library, so this stub provides exactly that. It exists ONLY so the test TU can
// compile the real headers; it is never used by production code.
//
// Include order in the test TU: <windows.h> and the CRT headers are pre-included
// before the `private->public` test latch (see UnlockPacingTests.cpp), so by the
// time this file is reached every include below is already loaded.

#include <windows.h>

#include <cstdint>
#include <cstring>
#include <string>
