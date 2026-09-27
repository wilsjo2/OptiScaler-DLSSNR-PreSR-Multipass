#pragma once

namespace DlssNr
{
// Queue-identity normalization shared by the finished-picture Late slots.
//
// The submission hook observes queues through a detour boundary: it asks the observed
// queue for the Streamline interface (Util::CheckForRealObject) and records the
// unwrapped (real) object when that query succeeds. A native queue answers a foreign
// IID with E_NOINTERFACE and clears the out pointer, so a failed query's output must
// never replace the observed pointer: two distinct native queues would otherwise both
// record null, compare equal, and admit incomplete cross-queue work as same-queue.
// Returns the unwrapped identity on success, the observed pointer otherwise.
inline void* UnwrapOrPreserveQueue(void* observed, bool unwrapped, void* out) { return unwrapped ? out : observed; }

// The production slot comparison shape: a null record never matches, so a failed
// unwrap that slipped through still cannot compare equal to anything.
inline bool SameRecordedQueue(const void* stored, const void* observed)
{
    return stored != nullptr && stored == observed;
}
} // namespace DlssNr
