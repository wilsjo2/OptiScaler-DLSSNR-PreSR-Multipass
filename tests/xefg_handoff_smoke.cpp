// ============================================================================
// tests/xefg_handoff_smoke.cpp - XeFG application-frame handoff test seam (TODO 6).
//
// Standalone cl.exe. No GPU, no game, no NR runtime: the XeFG proxy Present path
// and the SDK present status are doubles (this file); the only production surface
// the seam drives is the handoff core named below. Zero sleeps, zero waits: the
// state machine is driven by explicit capture facts and bounded by the runner.
//
// WHY THIS EXISTS
// ---------------
// The handoff of the finished application picture from the XeFG-owned proxy into
// NR is synchronization-sensitive (HANDOFF-nr-xefg-placement.md, "Mandatory
// invariants"). The seam pins the contract before the implementation exists:
// rows 9/10 implement the core below and call it at the end of
// XeFG_Dx12::Present() (after a successful Dispatch() preparation, for the
// owning application proxy only). This file then runs green. Until the core
// exists the runner exits 3 with the named HANDOFF_NOT_IMPLEMENTED marker.
//
// EXPECTED PRODUCTION SURFACE (the contract; rows 7-10 own the implementation)
// ---------------------------------------------------------------------------
//   OptiScaler/dlssnr/DlssNr_XeFGHandoff.h   (dependency-free; included here as
//                                             <dlssnr/DlssNr_XeFGHandoff.h>)
//
//   namespace DlssNr::XeFGHandoff
//     enum class SkipReason { None, DuplicateFrame, NoCurrentCapture,
//                             AmbiguousCapture, ProducerNotSubmitted,
//                             ProducerNotReady, UnsupportedColorSpace,
//                             MissingColorSpace };
//     const char* SkipReasonName(SkipReason);            // NR_XEFG_SKIP reason token
//     struct Identity { uint64_t generation; uint64_t frameId; };  // aggregate
//     struct Outcome  { bool applied; SkipReason reason; uint64_t slot; };
//     class Tracker {
//       uint64_t OpenInterval(uint64_t frameId);  // capture side: one per app frame
//       void Submitted(uint64_t serial);          // producer lists executed
//       void Ready(uint64_t serial);              // readiness fence complete / same queue
//       void Cancelled(uint64_t serial);          // recording discarded before submission
//       Outcome Handoff(Identity accepted);       // present side: one per app frame
//       void Reset(uint64_t generation);          // recreation/discontinuity/placement
//       uint64_t Applications() const;            // applied handoffs so far (diagnostic;
//                                                 // Reset does NOT clear it - the
//                                                 // recreation case asserts it stays)
//     };
//   All methods are called under the caller's NR lock on the single present path;
//   the tracker owns no GPU resource and performs no I/O.
//
// NORMATIVE RULES (asserted by the cases below)
//   R1 Identity = (swapchain generation, accepted application frame id). At most
//      ONE applied handoff per identity; a duplicate is refused with
//      duplicate_frame and consumes nothing.
//   R2 Handoff closes every open capture interval of the handed-off frame and of
//      older frames, in EVERY outcome including a skip: a stale slot never
//      "catches up" as a later frame's NR input. Generation-mismatched handoffs
//      are refused before closure (they cannot consume current-generation state).
//   R3 Handoff outcome, first match wins:
//        1. identity.generation != current generation -> no_current_capture
//        2. identity already applied                  -> duplicate_frame
//        3. matching interval count == 0              -> no_current_capture
//           matching interval count > 1, or a match exists while more than one
//           interval is open (pipelined frames)      -> ambiguous_capture
//        4. slot cancelled                           -> no_current_capture
//        5. producer not submitted                   -> producer_not_submitted
//        6. producer not ready                       -> producer_not_ready
//        7. otherwise applied; Outcome.slot is the consumed capture serial.
//      applied == false always means reason != None.
//   R4 Submitted/Ready/Cancelled for a closed or unknown serial are ignored; they
//      never revive an interval.
//   R5 Reset establishes a new generation and clears identity/interval state:
//      nothing is applied and nothing is released (in-flight GPU ownership stays
//      with the caller). The same frame id in a new generation is a fresh
//      identity, and a stale old-generation handoff replays nothing.
//   R6 The Present path never waits. Handoff is called after the wiring has
//      queried readiness; it applies or skips, and the seam bounds the call.
//   R7 One owning application proxy Present forwards the original Present exactly
//      once and calls Handoff exactly once. DXGI_PRESENT_TEST accepts no frame,
//      calls no handoff and consumes no interval. The XeFG 6x output displays six
//      pictures per accepted application frame - one NR application per
//      application frame, never six, never zero.
//
// EXIT CODES
//   0  GREEN - every scenario passed against the production core
//   1  production core present but a scenario failed (contract violated)
//   3  RED - production core absent; HANDOFF_NOT_IMPLEMENTED is printed
// ============================================================================

#include <windows.h>
#include <dxgi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NR_XEFG_HANDOFF_PRESENT
#define NR_XEFG_HANDOFF_PRESENT 0
#endif

#if NR_XEFG_HANDOFF_PRESENT
#include <dlssnr/DlssNr_XeFGHandoff.h>
#else
// ---------------------------------------------------------------------------
// Contract mirror used ONLY to compile and run this seam red before the
// production core exists. The signatures are the contract; Handoff deliberately
// reports "not applied" for every call (HANDOFF_NOT_IMPLEMENTED).
// ---------------------------------------------------------------------------
namespace DlssNr::XeFGHandoff
{
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
        case SkipReason::None: return "none";
        case SkipReason::DuplicateFrame: return "duplicate_frame";
        case SkipReason::NoCurrentCapture: return "no_current_capture";
        case SkipReason::AmbiguousCapture: return "ambiguous_capture";
        case SkipReason::ProducerNotSubmitted: return "producer_not_submitted";
        case SkipReason::ProducerNotReady: return "producer_not_ready";
        case SkipReason::UnsupportedColorSpace: return "unsupported_color_space";
        case SkipReason::MissingColorSpace: return "missing_color_space";
    }
    return "unknown";
}

struct Identity
{
    uint64_t generation = 0;
    uint64_t frameId = 0;
};

struct Outcome
{
    bool applied = false;
    SkipReason reason = SkipReason::None;
    uint64_t slot = 0;
};

class Tracker
{
  public:
    uint64_t OpenInterval(uint64_t) { return ++serial; }
    void Submitted(uint64_t) {}
    void Ready(uint64_t) {}
    void Cancelled(uint64_t) {}
    Outcome Handoff(Identity) { return {}; } // not implemented: this is the red state
    void Reset(uint64_t) {}
    uint64_t Applications() const { return 0; }

  private:
    uint64_t serial = 0;
};
} // namespace DlssNr::XeFGHandoff
#endif

namespace seam = DlssNr::XeFGHandoff;

using seam::Identity;
using seam::Outcome;
using seam::SkipReason;
using seam::Tracker;

namespace
{
int checks = 0;

void Check(bool condition, const std::string& what)
{
    ++checks;
    if (!condition)
        throw std::runtime_error(what);
}

// ---------------------------------------------------------------------------
// Test fixture: the game's accepted application-frame counter plus the capture
// interval lifecycle forwarded to the core, exactly as rows 9/10 must wire them.
// ---------------------------------------------------------------------------
struct Fixture
{
    Tracker tracker;
    uint64_t generation = 0;
    uint64_t frameId = 0;
    std::vector<uint64_t> applied; // slots consumed by applied handoffs
    std::vector<Outcome> outcomes;

    void Cold(uint64_t gen)
    {
        generation = gen;
        tracker.Reset(gen);
    }

    uint64_t BeginFrame() { return ++frameId; }

    uint64_t OpenCapture(uint64_t frame) { return tracker.OpenInterval(frame); }

    Outcome HandoffIdentity(uint64_t gen, uint64_t frame)
    {
        Identity identity {};
        identity.generation = gen;
        identity.frameId = frame;
        const Outcome outcome = tracker.Handoff(identity);
        if (outcome.applied)
            applied.push_back(outcome.slot);
        outcomes.push_back(outcome);
        return outcome;
    }

    Outcome Handoff(uint64_t frame) { return HandoffIdentity(generation, frame); }
};

// ---------------------------------------------------------------------------
// XeFG proxy double. Models only the SDK contract that matters here: rotating
// proxy backbuffers, one handoff per accepted application frame, six displayed
// pictures per application frame (XeFG 6x), the original Present forwarded
// exactly once, and DXGI_PRESENT_TEST accepted as a no-op.
// ---------------------------------------------------------------------------
struct ProxyDouble
{
    static constexpr uint32_t backBuffers = 3;
    static constexpr uint32_t picturesPerFrame = 6; // XeFG 6x

    struct PresentRecord
    {
        uint64_t generation = 0;
        uint64_t acceptedFrame = 0;
        uint32_t flags = 0;
        uint32_t bufferIndex = 0;
        bool handoffCalled = false;
        uint64_t displayedPictures = 0;
        Outcome outcome {};
    };

    uint64_t forwards = 0; // original proxy Present invocations
    uint64_t displayedPictures = 0;
    uint32_t bufferIndex = 0;
    bool nestedForward = false;
    bool inPresent = false;
    std::vector<PresentRecord> presents;

    Outcome Present(Fixture& fixture, uint32_t flags)
    {
        if (inPresent)
        {
            nestedForward = true; // the original Present must never be forwarded twice
            return {};
        }
        struct Guard
        {
            bool& flag;
            ~Guard() { flag = false; }
        } guard { inPresent };
        inPresent = true;

        ++forwards;                                    // the original proxy Present, forwarded once
        bufferIndex = (bufferIndex + 1) % backBuffers; // rotating proxy buffers

        PresentRecord record {};
        record.generation = fixture.generation;
        record.acceptedFrame = fixture.frameId;
        record.flags = flags;
        record.bufferIndex = bufferIndex;

        if (flags & DXGI_PRESENT_TEST)
        {
            presents.push_back(record); // test present: no frame accepted, no handoff
            return {};
        }

        record.handoffCalled = true;
        record.outcome = fixture.Handoff(fixture.frameId);
        record.displayedPictures = picturesPerFrame;
        displayedPictures += picturesPerFrame;
        presents.push_back(record);
        return record.outcome;
    }
};

// ---------------------------------------------------------------------------
// Cases. Required by plan TODO 6; every case name is printed by main().
// ---------------------------------------------------------------------------

// Rotating proxy buffers: three application frames, one application each,
// regardless of which proxy backbuffer the SDK presents.
void RotatingProxyBuffers()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    std::vector<uint32_t> buffers;
    for (int frame = 1; frame <= 3; ++frame)
    {
        fixture.BeginFrame();
        const uint64_t serial = fixture.OpenCapture(fixture.frameId);
        fixture.tracker.Submitted(serial);
        fixture.tracker.Ready(serial);
        const Outcome outcome = proxy.Present(fixture, 0);
        Check(outcome.applied, "rotating buffers: frame " + std::to_string(frame) + " did not apply");
        Check(outcome.slot == serial, "rotating buffers: frame " + std::to_string(frame) + " applied the wrong slot");
        buffers.push_back(proxy.presents.back().bufferIndex);
    }

    Check(proxy.forwards == 3, "rotating buffers: original Present not forwarded exactly once per present");
    Check(!proxy.nestedForward, "rotating buffers: the original Present was forwarded twice");
    Check(buffers == std::vector<uint32_t>({ 1, 2, 0 }), "rotating buffers: proxy buffers did not rotate");
    Check(fixture.applied == std::vector<uint64_t>({ 1, 2, 3 }),
          "rotating buffers: not exactly one application per accepted identity");
    Check(fixture.tracker.Applications() == 3, "rotating buffers: application count is not three");
    Check(proxy.displayedPictures == 18, "rotating buffers: three frames did not display six pictures each");
}

// Duplicate (swapchain generation, accepted frame id) handoffs.
void DuplicateHandoffs()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame();
    const uint64_t serial = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(serial);
    fixture.tracker.Ready(serial);
    const Outcome first = proxy.Present(fixture, 0);
    Check(first.applied, "duplicate: the first handoff did not apply");

    // Present1 after Present, a retry, or any second handoff for the same accepted
    // identity must not dispatch NR again.
    const Outcome duplicate = fixture.Handoff(fixture.frameId);
    Check(!duplicate.applied, "duplicate: the duplicate handoff applied NR again");
    Check(duplicate.reason == SkipReason::DuplicateFrame, "duplicate: wrong skip reason");
    Check(duplicate.slot == 0, "duplicate: the duplicate reported a consumed slot");
    Check(fixture.tracker.Applications() == 1, "duplicate: more than one application for one accepted identity");
    Check(fixture.applied == std::vector<uint64_t>({ serial }), "duplicate: the input slot was consumed twice");

    // A duplicate must not poison the next accepted frame.
    fixture.BeginFrame();
    const uint64_t next = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(next);
    fixture.tracker.Ready(next);
    const Outcome applied = proxy.Present(fixture, 0);
    Check(applied.applied && applied.slot == next, "duplicate: the next frame did not apply its own capture");
    Check(proxy.forwards == 2, "duplicate: original Present forwarding count");
}

// Six displayed pictures per application frame: one NR application, not six.
void SixDisplayedPictures()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame();
    const uint64_t serial = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(serial);
    fixture.tracker.Ready(serial);
    const Outcome outcome = proxy.Present(fixture, 0);

    Check(proxy.displayedPictures == ProxyDouble::picturesPerFrame,
          "six pictures: XeFG 6x did not display six pictures");
    Check(proxy.presents.size() == 1 && proxy.presents[0].handoffCalled,
          "six pictures: handoff not called exactly once for the application frame");
    Check(outcome.applied, "six pictures: NR was not applied for the application frame");
    Check(fixture.tracker.Applications() == 1,
          "six pictures: six displayed pictures caused more than one NR application");
    Check(proxy.forwards == 1, "six pictures: original Present not forwarded exactly once");
    Check(!proxy.nestedForward, "six pictures: original Present forwarded twice");
}

// DXGI_PRESENT_TEST: no accepted frame, no handoff, no consumed interval.
void PresentTest()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame();
    const uint64_t serial = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(serial);
    fixture.tracker.Ready(serial);

    proxy.Present(fixture, DXGI_PRESENT_TEST);
    Check(proxy.presents.size() == 1 && !proxy.presents[0].handoffCalled,
          "present_test: handoff ran for DXGI_PRESENT_TEST");
    Check(proxy.displayedPictures == 0, "present_test: a test present displayed pictures");
    Check(fixture.outcomes.empty(), "present_test: the handoff ran for a test present");
    Check(fixture.tracker.Applications() == 0, "present_test: a test present accepted an application frame");

    // The test present must not consume the capture interval: the following real
    // present applies that same input exactly once.
    const Outcome real = proxy.Present(fixture, 0);
    Check(real.applied && real.slot == serial, "present_test: the real present after a test present lost its input");
    Check(proxy.displayedPictures == ProxyDouble::picturesPerFrame,
          "present_test: the real present did not display six pictures");
    Check(fixture.tracker.Applications() == 1, "present_test: unexpected application count");
    Check(proxy.forwards == 2, "present_test: original Present not forwarded exactly once per present call");
}

// Unsubmitted input: producer never executed, so no application, no catch-up.
void UnsubmittedInput()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame();
    const uint64_t stale = fixture.OpenCapture(fixture.frameId); // captured, never submitted
    const Outcome skipped = proxy.Present(fixture, 0);
    Check(!skipped.applied, "unsubmitted: NR applied without a submitted producer");
    Check(skipped.reason == SkipReason::ProducerNotSubmitted, "unsubmitted: wrong skip reason");
    Check(fixture.tracker.Applications() == 0, "unsubmitted: application count");

    // The rejected slot is submitted/ready only after its handoff: never a later
    // frame's NR input.
    fixture.tracker.Submitted(stale);
    fixture.tracker.Ready(stale);

    fixture.BeginFrame();
    const uint64_t fresh = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(fresh);
    fixture.tracker.Ready(fresh);
    const Outcome applied = proxy.Present(fixture, 0);
    Check(applied.applied && applied.slot == fresh, "unsubmitted: the later frame did not apply its own capture");
    Check(fixture.applied == std::vector<uint64_t>({ fresh }),
          "unsubmitted: the rejected slot became a later frame's NR input");
}

// Cross-queue input that is not ready: skip without waiting; late readiness does
// not become a later frame's input either.
void CrossQueueInput()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame();
    const uint64_t serial = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(serial); // executed on a queue other than the application queue

    const auto start = std::chrono::steady_clock::now();
    const Outcome skipped = proxy.Present(fixture, 0); // its readiness fence has not completed
    const long long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();

    Check(!skipped.applied, "cross-queue: NR applied on an unready cross-queue producer");
    Check(skipped.reason == SkipReason::ProducerNotReady, "cross-queue: wrong skip reason");
    Check(elapsedMs < 1000, "cross-queue: the handoff waited on the producer (Present-side wait)");
    Check(fixture.tracker.Applications() == 0, "cross-queue: application count");

    fixture.tracker.Ready(serial); // the fence completes after the handoff: too late
    fixture.BeginFrame();
    const uint64_t fresh = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(fresh);
    fixture.tracker.Ready(fresh);
    const Outcome applied = proxy.Present(fixture, 0);
    Check(applied.applied && applied.slot == fresh, "cross-queue: the later frame did not apply its own capture");
    Check(fixture.applied == std::vector<uint64_t>({ fresh }),
          "cross-queue: late cross-queue input became a later frame's NR input");
}

// Cancellation: a discarded recording is never applied and never replayed.
void Cancellation()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame();
    const uint64_t serial = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Cancelled(serial); // recording discarded before submission
    fixture.tracker.Submitted(serial); // late queue hook: must not revive the interval
    fixture.tracker.Ready(serial);
    const Outcome skipped = proxy.Present(fixture, 0);
    Check(!skipped.applied, "cancellation: a cancelled capture was applied");
    Check(skipped.reason == SkipReason::NoCurrentCapture, "cancellation: wrong skip reason");
    Check(fixture.tracker.Applications() == 0, "cancellation: application count");

    fixture.BeginFrame();
    const uint64_t fresh = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(fresh);
    fixture.tracker.Ready(fresh);
    const Outcome applied = proxy.Present(fixture, 0);
    Check(applied.applied && applied.slot == fresh, "cancellation: the later frame did not apply its own capture");
    Check(fixture.applied == std::vector<uint64_t>({ fresh }), "cancellation: the cancelled slot was replayed later");
}

// Capture ambiguity: pipelined frames are refused, never guessed.
void CaptureAmbiguity()
{
    Fixture fixture;
    fixture.Cold(1);

    fixture.BeginFrame(); // frame 1
    const uint64_t first = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(first);
    fixture.tracker.Ready(first);
    fixture.BeginFrame(); // frame 2 captured before frame 1's present: pipelined
    const uint64_t second = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(second);
    fixture.tracker.Ready(second);

    const Outcome ambiguous = fixture.Handoff(1);
    Check(!ambiguous.applied, "ambiguity: an arbitrary pipelined slot was applied");
    Check(ambiguous.reason == SkipReason::AmbiguousCapture, "ambiguity: wrong skip reason");
    Check(fixture.tracker.Applications() == 0, "ambiguity: application count");

    // The pipelined frame's own handoff still applies its own capture exactly once.
    const Outcome owned = fixture.Handoff(2);
    Check(owned.applied && owned.slot == second, "ambiguity: the pipelined frame did not apply its own capture");
    Check(fixture.applied == std::vector<uint64_t>({ second }), "ambiguity: the ambiguous slot caught up later");
}

// The capture interval is closed at every handoff, even when composition is
// skipped; old slots never catch up.
void CaptureIntervalClosedOnSkip()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame();
    const uint64_t stale = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(stale); // submitted but not ready: composition is skipped
    const Outcome skipped = proxy.Present(fixture, 0);
    Check(!skipped.applied && skipped.reason == SkipReason::ProducerNotReady,
          "interval: expected a producer_not_ready skip");

    fixture.tracker.Ready(stale); // the old slot becomes ready after the handoff
    fixture.BeginFrame();         // no capture this frame
    const Outcome empty = proxy.Present(fixture, 0);
    Check(!empty.applied, "interval: a closed slot was applied");
    Check(empty.reason == SkipReason::NoCurrentCapture, "interval: wrong skip reason after a closed interval");
    Check(fixture.applied.empty(), "interval: a closed slot caught up as NR input");
}

// Recreation reset: new generation, no stale replay, no release by the tracker.
void RecreationReset()
{
    Fixture fixture;
    fixture.Cold(1);
    ProxyDouble proxy;

    fixture.BeginFrame(); // frame 1 of generation 1
    const uint64_t first = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(first);
    fixture.tracker.Ready(first);
    const Outcome applied = proxy.Present(fixture, 0);
    Check(applied.applied, "recreation: the first application is missing");

    fixture.BeginFrame(); // frame 2 captured, never handed off
    const uint64_t inflight = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(inflight);

    fixture.Cold(2); // swapchain recreation: reset, nothing applied, nothing released
    Check(fixture.tracker.Applications() == 1, "recreation: reset changed the application count");

    fixture.frameId = 0; // the new swapchain accepts frame ids from scratch
    fixture.BeginFrame();
    const uint64_t fresh = fixture.OpenCapture(fixture.frameId);
    fixture.tracker.Submitted(fresh);
    fixture.tracker.Ready(fresh);

    // A stale handoff from the old generation is refused and must not consume the
    // fresh interval.
    const Outcome stale = fixture.HandoffIdentity(1, 1);
    Check(!stale.applied, "recreation: a stale-generation handoff applied");
    Check(stale.reason == SkipReason::NoCurrentCapture, "recreation: wrong skip reason for a stale generation");
    Check(stale.slot == 0, "recreation: a stale-generation handoff consumed a slot");

    // The same frame id in the new generation is a fresh identity, not a duplicate.
    const Outcome reapplied = fixture.HandoffIdentity(2, 1);
    Check(reapplied.applied && reapplied.slot == fresh, "recreation: the new generation identity was not applied");
    Check(fixture.applied == std::vector<uint64_t>({ first, fresh }),
          "recreation: a stale slot replayed across recreation");
    Check(fixture.tracker.Applications() == 2, "recreation: application count after recreation");
}

// The NR_XEFG_SKIP reason vocabulary is part of the contract (handoff doc,
// "Proposed new diagnostics").
void ReasonVocabulary()
{
    Check(std::string(seam::SkipReasonName(SkipReason::DuplicateFrame)) == "duplicate_frame",
          "vocabulary: duplicate_frame token");
    Check(std::string(seam::SkipReasonName(SkipReason::NoCurrentCapture)) == "no_current_capture",
          "vocabulary: no_current_capture token");
    Check(std::string(seam::SkipReasonName(SkipReason::AmbiguousCapture)) == "ambiguous_capture",
          "vocabulary: ambiguous_capture token");
    Check(std::string(seam::SkipReasonName(SkipReason::ProducerNotSubmitted)) == "producer_not_submitted",
          "vocabulary: producer_not_submitted token");
    Check(std::string(seam::SkipReasonName(SkipReason::ProducerNotReady)) == "producer_not_ready",
          "vocabulary: producer_not_ready token");
    Check(std::string(seam::SkipReasonName(SkipReason::UnsupportedColorSpace)) == "unsupported_color_space",
          "vocabulary: unsupported_color_space token");
    Check(std::string(seam::SkipReasonName(SkipReason::MissingColorSpace)) == "missing_color_space",
          "vocabulary: missing_color_space token");
}
} // namespace

int main()
{
    struct Case
    {
        const char* name;
        void (*run)();
    };

    const Case cases[] = {
        { "rotating_proxy_buffers", RotatingProxyBuffers },
        { "duplicate_handoffs", DuplicateHandoffs },
        { "six_displayed_pictures", SixDisplayedPictures },
        { "present_test", PresentTest },
        { "unsubmitted_input", UnsubmittedInput },
        { "cross_queue_input", CrossQueueInput },
        { "cancellation", Cancellation },
        { "capture_ambiguity", CaptureAmbiguity },
        { "capture_interval_closed_on_skip", CaptureIntervalClosedOnSkip },
        { "recreation_reset", RecreationReset },
        { "reason_vocabulary", ReasonVocabulary },
    };

    std::printf("XEFG_HANDOFF_SEAM contract=dlssnr/DlssNr_XeFGHandoff.h production=%d cases=%u\n",
                NR_XEFG_HANDOFF_PRESENT ? 1 : 0, (unsigned) std::size(cases));

    unsigned failed = 0;
    for (const Case& test : cases)
    {
        checks = 0;
        try
        {
            test.run();
            if (checks == 0)
                throw std::runtime_error("scenario exercised no contract assertions");
            std::printf("CASE %-34s PASS (%d checks)\n", test.name, checks);
        }
        catch (const std::exception& error)
        {
            ++failed;
            std::printf("CASE %-34s FAIL %s\n", test.name, error.what());
        }
    }

    std::printf("XEFG_HANDOFF_SEAM cases=%u failed=%u\n", (unsigned) std::size(cases), failed);

#if !NR_XEFG_HANDOFF_PRESENT
    std::printf("RED: HANDOFF_NOT_IMPLEMENTED expected production unit "
                "<dlssnr/DlssNr_XeFGHandoff.h> (DlssNr::XeFGHandoff::Tracker); "
                "rows 9/10 own the implementation\n");
    return 3;
#else
    return failed == 0 ? 0 : 1;
#endif
}
