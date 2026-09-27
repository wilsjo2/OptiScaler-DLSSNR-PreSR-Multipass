#pragma once
// ============================================================================
// DlssNr_FinishedConsumer.h - who consumes a finished-picture capture.
//
// WHY THIS EXISTS
//   The finished-picture capture interval is armed from the NR evaluation seam.
//   Under XeFG it is consumed by the owned handoff in XeFG_Dx12::Present(), which
//   only runs while XeFG is active and this feature's app-facing proxy is the
//   registered FG swapchain. During the pre-activation warmup (and after a
//   deactivation) captures would be armed with no consumer: their game recordings
//   stay unresolved, the slot stays owned, and the four-slot pool starves -
//   capture-20260926-1255-run-end-pid87148 is exactly that state (four arms before
//   XeFG activated at 12:48:33.11, then 11835 no_current_capture skips / 0 apply).
//
//   The XeFG lifecycle publishes this state (ResetNrHandoff, Activate,
//   Deactivate, swapchain release) so the arm path can refuse that warmup arming.
//   Every other provider - including the generic finished-picture route - stays
//   Generic and is untouched.
//
// CONTRACT
//   C1 Generic      no XeFG-owned handoff is in play; arming proceeds as before.
//   C2 XeFGWaiting  an XeFG app-facing proxy is the route and its owned handoff
//                   does not consume captures yet; arming must be refused.
//   C3 XeFGAvailable the owning XeFG proxy is active; arming proceeds and the
//                   next Present consumes the capture.
//   C4 the state is a single process-wide value: the XeFG feature that owns the
//      registered proxy publishes it, and NR reads it under its own lock.
//   C5 only the registered owner publishes; a nested teardown or an aborted release must
//      not leave the state stuck (XeFG release ordering).
// ============================================================================
#include <atomic>

namespace DlssNr
{
enum class FinishedConsumer : unsigned
{
    Generic = 0,
    XeFGWaiting,
    XeFGAvailable
};

inline std::atomic<unsigned>& FinishedConsumerState()
{
    static std::atomic<unsigned> state { (unsigned) FinishedConsumer::Generic };
    return state;
}

inline void PublishFinishedConsumer(FinishedConsumer consumer)
{
    FinishedConsumerState().store((unsigned) consumer, std::memory_order_release);
}

inline FinishedConsumer CurrentFinishedConsumer()
{
    return (FinishedConsumer) FinishedConsumerState().load(std::memory_order_acquire);
}

// True when the capture path must not arm: the route is an XeFG app-facing proxy
// whose owned handoff has no consumer for this generation.
inline bool FinishedConsumerUnavailable() { return CurrentFinishedConsumer() == FinishedConsumer::XeFGWaiting; }

// The pool's capture admission rule. Both capture entry points (LateContext::Capture and
// LateContext::CaptureResidual) call exactly this, so the deferred residual route cannot
// fill the pool while the consumer is unavailable.
inline bool FinishedConsumerAdmitsCapture() { return !FinishedConsumerUnavailable(); }

// Publication policy (pure, so the XeFG lifecycle wiring is testable): only the
// instance whose app-facing proxy is the registered FG swapchain may publish, and
// during proxy creation it may claim only while no other owner exists.
inline bool FinishedConsumerMayPublish(const void* instanceSwapchain, const void* registeredSwapchain)
{
    return instanceSwapchain != nullptr && (instanceSwapchain == registeredSwapchain || registeredSwapchain == nullptr);
}

inline bool FinishedConsumerIsOwner(const void* instanceSwapchain, const void* registeredSwapchain)
{
    return instanceSwapchain != nullptr && instanceSwapchain == registeredSwapchain;
}

// A release publishes Generic only when it actually tore the owning proxy down: an
// aborted/reentrant release, or a non-owner's release, must leave the consumer state
// to the registered owner (a nested teardown inside the release publishes XeFGWaiting
// on the way, so this must be applied after it).
inline bool ReleasePublishesGeneric(bool wasOwner, bool aborted) { return wasOwner && !aborted; }
} // namespace DlssNr
