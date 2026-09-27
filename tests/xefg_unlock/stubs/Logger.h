#pragma once

// tests/xefg_unlock/stubs/Logger.h - MINIMAL logging stub for the T7 unlock/pacing tests.
//
// Production Logger.h forwards to spdlog with a file backend. The headers under test
// only emit LOG_INFO/LOG_WARN/LOG_ERROR/LOG_DEBUG statements, so this stub records
// the WARN/ERROR first-argument literals (used to prove the bad-build warning path)
// and discards the rest. Test-only; never used by production code.

#include <cstdarg>
#include <string>
#include <vector>

namespace XeFGUnlockTestLog
{
inline std::vector<std::string> g_warns;
inline std::vector<std::string> g_errors;

// First argument of every LOG_* call in the headers under test is a string literal;
// the remaining fmt arguments are intentionally dropped (passing them through
// C-style ... would trip C4840 on the std::string temporaries and they are never
// asserted on).
inline void NoteWarn(const char* message)
{
    g_warns.emplace_back(message == nullptr ? "" : message);
}

inline void NoteError(const char* message)
{
    g_errors.emplace_back(message == nullptr ? "" : message);
}

inline void Clear()
{
    g_warns.clear();
    g_errors.clear();
}
} // namespace XeFGUnlockTestLog

#define LOG_INFO(msg, ...) ((void) 0)
#define LOG_DEBUG(msg, ...) ((void) 0)
#define LOG_WARN(msg, ...) ::XeFGUnlockTestLog::NoteWarn(msg)
#define LOG_ERROR(msg, ...) ::XeFGUnlockTestLog::NoteError(msg)
