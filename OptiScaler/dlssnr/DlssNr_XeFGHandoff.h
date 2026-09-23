#pragma once
// ============================================================================
// DlssNr_XeFGHandoff.h - handoff identity and capture-interval core for NR on
// the XeFG application path (nr-xefg-088-release, todo 9).
//
// WHY THIS EXISTS
//   Under XeFG the finished application picture has to be handed to NR from the
//   app-facing proxy's Present path once per accepted application frame, while
//   the SDK still displays six generated pictures per accepted frame. The
//   capture path feeds this core the lifecycle of every captured frame; the
//   Present side calls Handoff() once per accepted application frame and gets a
//   single decision back: apply that frame's capture now, or skip it with a
//   named reason.
//
//   This unit is the dependency-free decision core of the contract pinned by
//   tests/xefg_handoff_smoke.cpp (todo 6): no DXGI/D3D12/Windows headers, no GPU
//   resource, no I/O, no locks, no waits. The Present-site wiring (todo 10)
//   includes it in XeFG_Dx12.cpp and emits the diagnostics from the outcome;
//   DlssNr_XeFGHandoff.cpp exists only so the unit is compiled by the real
//   project before that wiring lands.
//
// THREADING AND OWNERSHIP
//   All methods are called under the caller's NR lock on the single present path
//   (the seam's contract). The tracker has no internal synchronization and owns
//   no GPU resource: Reset() drops bookkeeping only, never an in-flight slot or
//   fence. The caller keeps its existing serial/pending ownership
//   (shaders/dlssnr/DlssNr_Dx12_Late.cpp Arm/Cancel) and the fence-preserving
//   skipped-slot path (DlssNr_Dx12_FinishedQueue.cpp) unchanged.
//
// NORMATIVE INVARIANTS (asserted by the seam)
//   R1 identity    Identity is (swapchain generation, accepted application frame
//                  id). At most ONE applied handoff per identity; a duplicate is
//                  refused with duplicate_frame and consumes nothing.
//   R2 interval    Handoff closes every open capture interval of the handed-off
//                  frame and of older frames, in EVERY outcome including a skip:
//                  a stale slot never "catches up" as a later frame's NR input.
//                  A generation-mismatched handoff is refused before closure; it
//                  cannot consume current-generation state.
//   R3 outcome     First match wins:
//                    1. generation mismatch                      -> no_current_capture
//                    2. identity already applied                 -> duplicate_frame
//                    3. no interval matching the frame           -> no_current_capture
//                    4. more than one match, or a match while
//                       more than one interval is open
//                       (pipelined frames)                       -> ambiguous_capture
//                    5. interval cancelled                       -> no_current_capture
//                    6. producer not submitted                   -> producer_not_submitted
//                    7. producer not ready                       -> producer_not_ready
//                    8. otherwise applied; Outcome::slot is the consumed serial.
//                  applied == false always means reason != None.
//   R4 facts       Submitted/Ready/Cancelled for a closed or unknown serial are
//                  ignored; they never revive an interval.
//   R5 reset       Reset establishes a new generation and clears identity and
//                  interval state: nothing is applied and nothing is released.
//                  The same frame id in a new generation is a fresh identity, and
//                  a stale old-generation handoff replays nothing.
//   R6 no wait     Handoff never waits: it applies or skips.
//   R7 present     One owning application proxy Present calls Handoff exactly
//                  once; DXGI_PRESENT_TEST accepts no frame, calls no handoff and
//                  consumes no interval.
//
// WIRING NOTES
//   - frameId is the id the Present site already holds (_lastDispatchedFrame),
//     treated as opaque except that it must be monotonically non-decreasing
//     within a generation, so "older frames" is well defined. Use it to close
//     the capture interval of the accepted frame and of every older one.
//   - Serials returned by OpenInterval() are opaque and never reused for the
//     tracker's lifetime (also not across Reset), so a late fact for an old
//     serial can never land on a newer interval. 0 is never a valid serial.
//   - A wiring-level refusal that must not consume the frame's capture (the
//     route refuses unsupported_color_space / missing_color_space before the
//     handoff) closes that interval before the next present. The wiring does
//     this via XeFGCloseCaptures (max-or-serial: max for a colour refusal that
//     persists until the proxy state changes, the offered capture's serial for
//     a tracker refusal) at the call site, never via Cancelled(serial), and
//     emits NR_XEFG_SKIP with the refusal reason. The observable outcome equals
//     a Cancelled() mark: nothing is applied. This is defense-in-depth: with
//     the wiring's single open/close per present a leftover interval cannot
//     arise in production, and the ambiguous_capture branch (R3.4) is
//     seam/probe-only (safe degeneration). The two refusal reasons live in the
//     shared vocabulary but are never returned by Handoff(), which only judges
//     capture-lifecycle facts.
//   - Diagnostics from an outcome: applied ->
//       NR_XEFG_APPLY generation=<G> frame=<F> slot=<S>
//     otherwise ->
//       NR_XEFG_SKIP generation=<G> frame=<F> reason=<SkipReasonName(reason)>
//
// CAPACITY
//   The tracker keeps a fixed, allocation-free list of at most Capacity open
//   intervals. When it is exhausted a further capture is not tracked and its
//   frame's handoff skips with no_current_capture; older tracked intervals are
//   never evicted, because a stale slot must not become a later frame's input.
// ============================================================================

#include <cstdint>

namespace DlssNr::XeFGHandoff
{
// Reason vocabulary for NR_XEFG_SKIP (tokens are the contract; see the handoff
// doc "Proposed new diagnostics"). UnsupportedColorSpace and MissingColorSpace
// are wiring-level refusals: Handoff() never returns them (see WIRING NOTES).
enum class SkipReason : unsigned
{
    None = 0,
    DuplicateFrame,
    NoCurrentCapture,
    AmbiguousCapture,
    ProducerNotSubmitted,
    ProducerNotReady,
    UnsupportedColorSpace,
    MissingColorSpace
};

inline const char* SkipReasonName(SkipReason reason)
{
    switch (reason)
    {
    case SkipReason::None:
        return "none";
    case SkipReason::DuplicateFrame:
        return "duplicate_frame";
    case SkipReason::NoCurrentCapture:
        return "no_current_capture";
    case SkipReason::AmbiguousCapture:
        return "ambiguous_capture";
    case SkipReason::ProducerNotSubmitted:
        return "producer_not_submitted";
    case SkipReason::ProducerNotReady:
        return "producer_not_ready";
    case SkipReason::UnsupportedColorSpace:
        return "unsupported_color_space";
    case SkipReason::MissingColorSpace:
        return "missing_color_space";
    }
    return "unknown";
}

// The accepted handoff identity: (swapchain generation, accepted application
// frame id). Aggregates so the Present site can fill them without a constructor.
struct Identity
{
    uint64_t generation = 0;
    uint64_t frameId = 0;
};

struct Outcome
{
    bool applied = false;
    SkipReason reason = SkipReason::None;
    uint64_t slot = 0; // consumed capture serial when applied
};

class Tracker
{
  public:
    // Maximum number of capture intervals tracked concurrently. The capture path
    // opens one interval per captured application frame and Handoff() closes the
    // accepted frame's interval plus every older one, so the steady state is a
    // single open interval; deeper pipelining is already pathological. Seam-only:
    // exhaustion is unreachable via OwnedNrHandoff's single open/close (safe degeneration).
    static constexpr uint32_t Capacity = 8;

    // Capture side: one call per captured application frame. Returns the opaque
    // serial (slot id) used by the producer facts and consumed by Handoff().
    uint64_t OpenInterval(uint64_t frameId)
    {
        const uint64_t opened = ++serialCounter;
        if (open == Capacity)
            return opened; // capacity exhausted: never tracked, never applied (see CAPACITY)

        Interval& interval = intervals[open++];
        interval = Interval {};
        interval.serial = opened;
        interval.frameId = frameId;
        return opened;
    }

    // Producer facts (submission hook / readiness fence). Facts for a closed or
    // unknown serial are ignored and never revive an interval.
    void Submitted(uint64_t serial) { Set(serial, &Interval::submitted); }
    void Ready(uint64_t serial) { Set(serial, &Interval::ready); }
    void Cancelled(uint64_t serial) { Set(serial, &Interval::cancelled); }

    // Present side: one call per accepted application frame. Applies at most one
    // capture, closes the interval, or refuses with a named reason (R1-R3).
    Outcome Handoff(Identity accepted)
    {
        // R3.1: a stale generation is refused before closure; it must not consume
        // current-generation state. R2: only current-generation handoffs close.
        if (accepted.generation != generation)
            return { false, SkipReason::NoCurrentCapture, 0 };

        // R3.2: at most one application per accepted identity.
        if (applied && accepted.frameId == appliedFrame)
        {
            CloseThrough(accepted.frameId);
            return { false, SkipReason::DuplicateFrame, 0 };
        }

        uint32_t matches = 0;
        for (uint32_t i = 0; i < open; ++i)
            if (intervals[i].frameId == accepted.frameId)
                ++matches;

        // R3.3: no capture for this accepted frame (closed, never opened or
        // already consumed): the interval still closes for everything older.
        if (matches == 0)
        {
            CloseThrough(accepted.frameId);
            return { false, SkipReason::NoCurrentCapture, 0 };
        }

        // R3.4: pipelined frames are refused, never guessed. A match while any
        // other frame's interval is still open is ambiguous. Seam/probe-only:
        // unreachable via OwnedNrHandoff's single open/close (safe degeneration).
        if (matches > 1 || open > 1)
        {
            CloseThrough(accepted.frameId);
            return { false, SkipReason::AmbiguousCapture, 0 };
        }

        // matches == 1 && open == 1: this interval is the frame's only capture.
        Interval& interval = intervals[0];
        if (interval.cancelled)
        {
            CloseThrough(accepted.frameId);
            return { false, SkipReason::NoCurrentCapture, 0 };
        }
        if (!interval.submitted) // R3.6 (after R3.5): producer never executed
        {
            CloseThrough(accepted.frameId);
            return { false, SkipReason::ProducerNotSubmitted, 0 };
        }
        if (!interval.ready) // R3.7: cross-queue input not fenced complete: skip, never wait
        {
            CloseThrough(accepted.frameId);
            return { false, SkipReason::ProducerNotReady, 0 };
        }

        const uint64_t slot = interval.serial;
        CloseThrough(accepted.frameId);
        applied = true;
        appliedFrame = accepted.frameId;
        ++applications;
        return { true, SkipReason::None, slot };
    }

    // Swapchain recreation / discontinuity / placement change: a new generation,
    // cleared identity and intervals. Applies nothing and releases nothing
    // (in-flight GPU ownership stays with the caller). Applications() survives
    // on purpose: it is the diagnostic count of applied handoffs so far.
    void Reset(uint64_t newGeneration)
    {
        generation = newGeneration;
        open = 0;
        applied = false;
        appliedFrame = 0;
    }

    // Applied handoffs so far (diagnostic; Reset does not clear it).
    uint64_t Applications() const { return applications; }

  private:
    struct Interval
    {
        uint64_t serial = 0;
        uint64_t frameId = 0;
        bool submitted = false;
        bool ready = false;
        bool cancelled = false;
    };

    void Set(uint64_t serial, bool Interval::* field)
    {
        for (uint32_t i = 0; i < open; ++i)
            if (intervals[i].serial == serial)
                intervals[i].*field = true;
    }

    // R2: close the handed-off frame's interval and every older one in place
    // (allocation-free); newer pipelined intervals stay open for their own handoff.
    void CloseThrough(uint64_t frameId)
    {
        uint32_t kept = 0;
        for (uint32_t i = 0; i < open; ++i)
            if (intervals[i].frameId > frameId)
                intervals[kept++] = intervals[i];
        open = kept;
    }

    Interval intervals[Capacity] {};
    uint32_t open = 0;
    uint64_t serialCounter = 0; // monotonic for the tracker's lifetime; never reused
    uint64_t generation = 0;
    uint64_t appliedFrame = 0;
    uint64_t applications = 0;
    bool applied = false;
};
} // namespace DlssNr::XeFGHandoff
