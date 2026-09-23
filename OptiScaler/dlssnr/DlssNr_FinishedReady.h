#pragma once
#include <cstdint>

namespace DlssNr
{
// A producer's queued signal can itself depend on presentation completing first.
// Same-queue submission order is sufficient; cross-queue input must already be complete.
inline bool FinishedInputReady(bool sameQueue, uint64_t completed, uint64_t required)
{
    return completed != UINT64_MAX && (sameQueue || completed >= required);
}
} // namespace DlssNr
