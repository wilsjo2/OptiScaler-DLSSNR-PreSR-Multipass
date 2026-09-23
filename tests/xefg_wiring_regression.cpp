// ============================================================================
// xefg_wiring_regression.cpp - nr-xefg-followups T3, wiring-double proof harness.
//
// WHAT THIS IS
//   The frozen seam (tests/xefg_handoff_smoke.cpp, 088-release todo 6) drives the
//   production decision core (DlssNr::XeFGHandoff::Tracker) only; it cannot catch
//   wiring-level defects (which capture the compose consumes, whether a refused
//   capture is ever closed, whether the skip site closes the untracked-facts gap,
//   which frame id the identity uses). This harness is that wiring-level proof:
//
//   - it compiles the REAL production decision core
//     (OptiScaler/dlssnr/DlssNr_XeFGHandoff.h), and
//   - drives a faithful mirror of XeFG_Dx12::OwnedNrHandoff plus the NR-store
//     edges it calls, with the D3D/DlssNr/FGHooks edges stubbed.
//
//   THE MIRROR CONTRACT - RE-MIRROR ON PRODUCTION CHANGE
//   The mirror below copies the production step order and each site's semantics,
//   with inherited Susemi source line references below (not wil checkout line numbers).
//   Symbols are authoritative: changes to their semantics require re-mirroring.
//   The wil port separates invalid presentation extents from pending recording
//   ownership and exercises FinishedPictureSubmitted/Reset gating explicitly.
//   Inherited source map:
//     XeFG_Dx12.cpp:1528      OwnedNrHandoff entry / gate order
//     XeFG_Dx12.cpp:1549-1556 buffer acquisition failure -> one named SKIP line
//     XeFG_Dx12.cpp:1573-1582 wiring-level colour refusals (close-max path)
//     XeFG_Dx12.cpp:1590-1591 accepted frame id = the id the Present site
//                            already holds (_lastDispatchedFrame), NOT GetDispatchIndex
//     XeFG_Dx12.cpp:1602-1611 XeFGPendingCapture facts feed the tracker
//     XeFG_Dx12.cpp:1613-1615 identity + Handoff, once per accepted frame
//     XeFG_Dx12.cpp:1617-1632 applied -> ApplyXeFGPicture -> APPLY line -> record
//     XeFG_Dx12.cpp:1644-1652 the single skip site (closeAllCaptures: colour
//                            refusal OR no_current_capture close max; H1)
//     XeFG_Dx12.cpp:1653-1657 SKIP line + RecordXeFGHandoff (decided present)
//     XeFG_Dx12.cpp:1512-1519 ResetNrHandoff (Cold)
//     DlssNr_Dx12.cpp:612-661 PendingFinishedCapture (newest PENDING match)
//     DlssNr_Dx12.cpp:665-673 CloseFinishedCaptures (through-serial close)
//     DlssNr_Dx12.cpp:756-787  XeFGPendingCapture/ApplyXeFGPicture/XeFGCloseCaptures
//     DlssNr_Dx12_FinishedCompose.cpp:36-49 compose selection (epoch rule off
//                            for gameFrameHandoff; newest pending+submitted match)
//     DlssNr_Dx12_Late.cpp:55-84 + DlssNr_Dx12_State.h:327 the 4-slot pending pool
//     FG_Hooks.h:261-289 + FG_Hooks.cpp:1358-1377 RecordXeFGHandoff / present report
//
// SCOPE: links the Tracker header plus stubs ONLY. Never XeFG_Dx12.cpp, never
//   FG_Hooks. No GPU, no game, no NR runtime, no sleeps, no waits (the only
//   timing is the elapsed<1000ms no-wait assert, smoke :451-459 pattern).
//
// THE TWO FAILURE SEEDS (failure-provable; a seed run that exits 0 means the
// harness no longer detects its defect):
//   SEED_SKIPS_CLOSE     the skip site emits its SKIP line but never closes: the
//                        refused capture stays pending (the reverted-close defect
//                        T1's H1 fixed; also the pre-fix no-capture gap, where the
//                        untracked facts' serial 0 closed nothing - F2 review N1/N6).
//   SEED_STALE_COMPOSE   the compose ignores the close/consume gating and the
//                        recency rule: the OLDEST matching slot composes, closed or
//                        consumed (the "stale compose" the close gating prevents).
//
// The seed's Report() dead loop from the 10/seed double is fixed here: every
// case emits its own wiring log lines to stdout, PASS or FAIL.
// ============================================================================

#include <dlssnr/DlssNr_XeFGHandoff.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr uint32_t kPresentTest = 0x00000001; // DXGI_PRESENT_TEST
constexpr uint64_t kMaxSerial = std::numeric_limits<uint64_t>::max();
constexpr uint32_t kBackBuffers = 3;   // rotating proxy backbuffers (smoke ProxyDouble)
constexpr uint32_t kCaptureSlots = 4;   // LateContext pending pool (DlssNr_Dx12_State.h:327)

// ---------------------------------------------------------------------------
// NR-store double: LateContext slots (DlssNr_Dx12_Late.cpp:55-84 Acquire; selection
// DlssNr_Dx12.cpp:612-661; close DlssNr_Dx12.cpp:665-673; compose selection
// DlssNr_Dx12_FinishedCompose.cpp:36-49). "pending" gates composability exactly
// like the production slot; serials are NR LateContext serials - a DIFFERENT
// space from the tracker interval serial (F2 review N8).
// ---------------------------------------------------------------------------
struct Wiring
{
    DlssNr::XeFGHandoff::Tracker tracker; // the REAL production decision core

    uint64_t generation = 0;
    uint64_t heldFrame = 0;   // _lastDispatchedFrame: the id the Present site holds
    uint64_t displayIndex = 0; // GetDispatchIndex stand-in: advances on EVERY present
    uint32_t backBuffer = 0;

    struct Slot
    {
        uint64_t serial = 0;   // NR LateContext serial
        uint64_t frameId = 0;  // the application frame whose picture was captured
        uint32_t picture = 0; // the captured backbuffer (extent stand-in)
        bool pending = false;
        bool submitted = false;
        bool ready = false;
        bool validExtent = true; // OutputWidth != 0; separate from recording ownership
    };
    std::vector<Slot> slots;
    uint64_t serialCounter = 0;

    // DlssNr::XeFGCapture facts (DlssNr_Dx12.cpp:612-661).
    struct Facts
    {
        bool exists = false;
        uint64_t serial = 0;
        bool submitted = false;
        bool ready = false;
    };

    std::vector<uint64_t> composed; // serials consumed by ApplyXeFGPicture, in order
    std::vector<std::string> log;
    uint64_t recordSeq = 0; // FGHooks::XeFGHandoffSequence (FG_Hooks.h:261-289)
    uint64_t forwards = 0;

    // ResetNrHandoff mirror (XeFG_Dx12.cpp:1512-1519): new generation, tracker reset,
    // every pending capture closed (nothing released; fences still protect reuse).
    void Cold(uint64_t newGeneration)
    {
        generation = newGeneration;
        tracker.Reset(newGeneration);
        CloseCaptures(kMaxSerial);
    }

    // Dispatch tags a new accepted application frame: the held id advances, the
    // proxy backbuffer rotates (the display counter is separate and advances per
    // present, incl. test presents - IFGFeature.cpp:135-161).
    void BeginFrame()
    {
        ++heldFrame;
        backBuffer = (backBuffer + 1) % kBackBuffers;
    }

    uint32_t Picture() const { return backBuffer; }

    // Late.cpp Acquire mirror (DlssNr_Dx12_Late.cpp:55-84): a full pending pool
    // cannot arm this frame's capture (F2 review N1's pool precondition).
    bool Arm(uint64_t frame, uint32_t picture, bool submitted, bool ready)
    {
        uint32_t pendingCount = 0;
        for (const auto& slot : slots)
            if (slot.pending)
                ++pendingCount;
        if (pendingCount == kCaptureSlots)
            return false;
        Slot slot {};
        slot.serial = ++serialCounter;
        slot.frameId = frame;
        slot.picture = picture;
        slot.pending = true;
        slot.submitted = submitted;
        slot.ready = ready;
        slots.push_back(slot);
        return true;
    }

    bool PendingSlot(uint64_t serial) const
    {
        for (const auto& slot : slots)
            if (slot.serial == serial)
                return slot.pending;
        return false;
    }

    uint64_t SerialFrame(uint64_t serial) const
    {
        for (const auto& slot : slots)
            if (slot.serial == serial)
                return slot.frameId;
        return 0;
    }

    // DlssNr::XeFGPendingCapture mirror (DlssNr_Dx12.cpp:756-764 -> :612-661):
    // the NEWEST pending slot matching this finished picture (extent match,
    // :642-656 `slot.serial > match->serial`); no match -> untracked facts.
    bool PendingCapture(uint32_t picture, Facts& facts) const
    {
        const Slot* match = nullptr;
        for (const auto& slot : slots)
        {
            if (!slot.pending || !slot.validExtent || slot.picture != picture)
                continue;
            if (match == nullptr || slot.serial > match->serial)
                match = &slot;
        }
        if (match == nullptr)
            return false;
        facts.exists = true;
        facts.serial = match->serial;
        facts.submitted = match->submitted;
        facts.ready = match->submitted && match->ready;
        return true;
    }

    // DlssNr::ApplyXeFGPicture mirror (DlssNr_Dx12.cpp:767-777 ->
    // DlssNr_Dx12_FinishedCompose.cpp:36-49): the NEWEST pending+submitted+ready
    // slot matching the picture composes and is consumed; the epoch rule is off
    // for gameFrameHandoff (`!gameFrameHandoff &&`), so the close/consume gating
    // is the ONLY staleness defense. Returns false when nothing is composable.
    bool Apply(uint32_t picture)
    {
#if defined(SEED_STALE_COMPOSE)
        // SEED (stale compose): the gating is ignored and the OLDEST match wins -
        // a closed or already-consumed slot composes.
        for (auto& slot : slots)
            if (slot.picture == picture)
            {
                composed.push_back(slot.serial);
                return true;
            }
        return false;
#else
        Slot* match = nullptr;
        for (auto& slot : slots)
        {
            if (!slot.pending || !slot.validExtent || !slot.submitted || !slot.ready || slot.picture != picture)
                continue;
            if (match == nullptr || slot.serial > match->serial)
                match = &slot;
        }
        if (match == nullptr)
            return false;
        match->pending = false; // consumed: never selected again
        composed.push_back(match->serial);
        return true;
#endif
    }

    // CloseFinishedCaptures: submitted work closes pending; an unsubmitted recording
    // loses its extent but stays visible to wil's FinishedPictureSubmitted/Reset hooks.
    void CloseCaptures(uint64_t throughSerial)
    {
        for (auto& slot : slots)
            if (slot.pending && slot.serial <= throughSerial)
            {
                if (slot.submitted)
                    slot.pending = false;
                else
                    slot.validExtent = false;
            }
    }

    // FinishedQueue.cpp hook gating: only pending, unsubmitted recordings are observed.
    // Serial stands in for command-list identity; completion is delivered explicitly.
    void Submit(uint64_t serial)
    {
        for (auto& slot : slots)
            if (slot.serial == serial && slot.pending && !slot.submitted)
                slot.submitted = true;
    }

    void ResetRecording(uint64_t serial)
    {
        for (auto& slot : slots)
            if (slot.serial == serial && slot.pending && !slot.submitted)
            {
                slot.pending = false;
                slot.ready = true; // discarded recording: no signal was promised
            }
    }

    size_t CountToken(const char* token) const
    {
        size_t count = 0;
        for (const auto& line : log)
            if (line.find(token) == 0)
                ++count;
        return count;
    }

    size_t CountReason(const char* reason) const
    {
        const std::string needle = std::string("reason=") + reason;
        size_t count = 0;
        for (const auto& line : log)
            if (line.find(needle) != std::string::npos)
                ++count;
        return count;
    }
};

// ---------------------------------------------------------------------------
// Mirror of XeFG_Dx12::OwnedNrHandoff (XeFG_Dx12.cpp:1528-1659). Same step order
// as production; every step carries its production line cross-ref. The seeds
// flip only the marked steps.
// ---------------------------------------------------------------------------
void OwnedHandoffMirror(Wiring& w)
{
    // The accepted frame id is the id this Present site already holds - NOT
    // GetDispatchIndex, which reads the display counter (XeFG_Dx12.cpp:1590-1591).
    const uint64_t frame = w.heldFrame;

    // Wiring-level colour refusals (XeFG_Dx12.cpp:1573-1582) keep their close-max
    // path through the shared close site below; no case drives them, so the
    // mirror carries the flag as the constant the cases exercise (declared at the
    // close site because that is its only use - the seed removes both together).

    Wiring::Facts facts {};
    DlssNr::XeFGHandoff::SkipReason reason = DlssNr::XeFGHandoff::SkipReason::None;

    // Capture-lifecycle facts feed the decision core, then the tracker is called
    // exactly once for this accepted application frame (XeFG_Dx12.cpp:1602-1615).
    const bool tracked = w.PendingCapture(w.Picture(), facts); // DlssNr::XeFGPendingCapture
    if (tracked)
    {
        const uint64_t serial = w.tracker.OpenInterval(frame);
        if (facts.submitted)
            w.tracker.Submitted(serial);
        if (facts.ready)
            w.tracker.Ready(serial);
    }

    DlssNr::XeFGHandoff::Identity identity {};
    identity.generation = w.generation;
    identity.frameId = frame;
    const auto outcome = w.tracker.Handoff(identity);

    if (outcome.applied)
    {
        // Compose with real-game-frame semantics on the XeFG-retained application
        // queue (XeFG_Dx12.cpp:1617-1620); the queue identity itself is pinned by
        // the 10/seed double, not re-proven here.
        if (w.Apply(w.Picture()))
        {
            // interval= is the tracker capture-interval serial (XeFG_Dx12.cpp:1622-1625).
            w.log.push_back("NR_XEFG_APPLY generation=" + std::to_string(w.generation) +
                            " frame=" + std::to_string(frame) + " interval=" +
                            std::to_string(outcome.slot) + " same_queue=1 submitted=1");
            ++w.recordSeq; // FGHooks::RecordXeFGHandoff (XeFG_Dx12.cpp:1627)
            return;
        }
        // The facts were a snapshot and composition re-validated to nothing
        // runnable (XeFG_Dx12.cpp:1629-1632).
        reason = DlssNr::XeFGHandoff::SkipReason::NoCurrentCapture;
    }
    else
    {
        reason = outcome.reason; // XeFG_Dx12.cpp:1634-1640
    }

    // The single skip site (XeFG_Dx12.cpp:1644-1652): a colour refusal persists
    // until the proxy state changes, and a no_current_capture skip cannot name
    // the capture that stalled - the frame's facts may be untracked (serial 0)
    // while older NR-store leftovers are still pending (todo H1) - so both close
    // every pending capture; any other tracker refusal names exactly the capture
    // that was offered.
#if defined(SEED_SKIPS_CLOSE)
    // SEED (reverted close / the pre-fix gap): the skip site never closes; the
    // refused capture stays pending in the NR store. The closeAllCaptures
    // computation is the unseeded step this seed removes.
#else
    const bool colourRefusal = false; // wiring-level colour refusals (XeFG_Dx12.cpp:1573-1582)
    const bool closeAllCaptures =
        colourRefusal || reason == DlssNr::XeFGHandoff::SkipReason::NoCurrentCapture;
    w.CloseCaptures(closeAllCaptures ? kMaxSerial : facts.serial);
#endif
    w.log.push_back("NR_XEFG_SKIP generation=" + std::to_string(w.generation) + " frame=" +
                    std::to_string(frame) + " reason=" +
                    DlssNr::XeFGHandoff::SkipReasonName(reason)); // XeFG_Dx12.cpp:1653
    ++w.recordSeq; // a skip still decided this present (XeFG_Dx12.cpp:1657)
}

// ---------------------------------------------------------------------------
// Mirror of the FGPresent mechanics (FG_Hooks.cpp:1358-1377, FG_Hooks.h:261-289):
// DXGI_PRESENT_TEST runs no handoff (R7), the original proxy Present is
// forwarded exactly once, and NR_XEFG_PRESENT reports only when the owned
// handoff decided this present.
// ---------------------------------------------------------------------------
void PresentMirror(Wiring& w, uint32_t flags)
{
    ++w.displayIndex; // the display counter advances on every present (test or not)
    const uint64_t seqBefore = w.recordSeq; // FGHooks::XeFGHandoffSequence()

    if ((flags & kPresentTest) == 0) // willPresent: a test present runs no handoff
        OwnedHandoffMirror(w);

    ++w.forwards; // the original proxy Present, forwarded exactly once

    if (w.recordSeq != seqBefore)
        w.log.push_back("NR_XEFG_PRESENT generation=" + std::to_string(w.generation) +
                        " frame=" + std::to_string(w.heldFrame) +
                        " enabled=1 framegen_result=0 frames_presented=6");
}

int checks = 0;

void Check(bool condition, const std::string& what)
{
    ++checks;
    if (!condition)
        throw std::runtime_error(what);
}

// The 10/seed double's Report() dead loop (it iterated a temporary's empty log
// and printed nothing) is fixed here: every case emits its own wiring log lines,
// PASS or FAIL, so the raw logs carry the actual SKIP/APPLY/PRESENT evidence.
void Emit(const Wiring& w)
{
    for (const auto& line : w.log)
        std::printf("  wiring: %s\n", line.c_str());
}

// ---------------------------------------------------------------------------
// Cases. Every case is isolated: fresh Wiring, fresh tracker, exceptions are
// caught per case in main(), so one failing case never masks another (the
// smoke :641-656 pattern).
// ---------------------------------------------------------------------------

// The H1 gap: a capture whose handoff never ran (a test present) stays pending in
// the NR store; later no_current_capture skips have untracked facts (serial 0),
// so only the close-max at the skip site can close it. When the gap is open, the
// backbuffer rotates back to the stale capture's picture and a later frame's
// APPLY composes the STALE capture (F2 review N1's harmful variant).
void NoCaptureCloseGap()
{
    Wiring w;
    w.Cold(1);

    w.BeginFrame();                       // held frame 1, picture 1
    Check(w.Arm(1, w.Picture(), true, true), "no_capture_close_gap: the frame-1 capture could not be armed");
    PresentMirror(w, kPresentTest);     // R7: no handoff - the capture stays pending

    const auto start = std::chrono::steady_clock::now();
    w.BeginFrame();                       // held frame 2, picture 2
    PresentMirror(w, 0);                // untracked facts -> no_current_capture skip
    w.BeginFrame();                       // held frame 3, picture 0
    PresentMirror(w, 0);                // same
    w.BeginFrame();                       // held frame 4, picture 1: backbuffer rotation
    PresentMirror(w, 0);                // the stale capture's picture comes around again
    const long long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();

    Emit(w);

    Check(w.composed.empty(),
          "no_capture_close_gap: a stale capture composed for a later frame (serials=" +
              std::to_string(w.composed.size()) + ")");
    Check(w.tracker.Applications() == 0,
          "no_capture_close_gap: a stale capture was applied as a later frame's handoff");
    Check(w.CountReason("no_current_capture") == 3,
          "no_capture_close_gap: the three no-capture presents were not all no_current_capture skips");
    Check(w.composed.empty() && !w.PendingSlot(1) && w.slots.size() == 1 && !w.slots[0].pending,
          "no_capture_close_gap: the untracked-facts skip left a capture pending in the NR store");
    Check(elapsedMs < 1000,
          "no_capture_close_gap: the handoff path waited (Present-side wait)");
}

// An applied handoff consumes its capture; a later compose of the same (closed or
// consumed) capture must be rejected - the close/consume gating is the only
// staleness defense for this path (DlssNr_Dx12_FinishedCompose.cpp:36-49).
void AppliedThenComposeRejected()
{
    Wiring w;
    w.Cold(1);

    w.BeginFrame();                       // held frame 1, picture 1
    Check(w.Arm(1, w.Picture(), true, true), "applied_then_compose: the frame-1 capture could not be armed");
    PresentMirror(w, 0);                // applied: composed exactly once
    w.BeginFrame();                       // held frame 2, picture 2
    PresentMirror(w, 0);                // nothing pending -> no_current_capture skip
    w.BeginFrame();                       // held frame 3, picture 0
    PresentMirror(w, 0);
    w.BeginFrame();                       // held frame 4, picture 1: the consumed picture
    PresentMirror(w, 0);                // comes around again - nothing may compose

    Emit(w);

    Check(w.composed.size() == 1,
          "applied_then_compose: the frame-1 application did not compose exactly once");
    Check(w.CountToken("NR_XEFG_APPLY") == 1,
          "applied_then_compose: the APPLY token count is not one");
    Check(w.tracker.Applications() == 1,
          "applied_then_compose: the application count changed");
    Check(!w.Apply(w.Picture()),
          "applied_then_compose: a compose of the consumed capture was not rejected");
    Check(w.composed.size() == 1,
          "applied_then_compose: the rejected compose still consumed a capture");
}

// An unsubmitted producer is skipped with the named reason and its capture is
// closed at the single skip site; it must not survive to compose as a later
// frame's NR input (R2).
void UnsubmittedSlotSurvivesCompose()
{
    Wiring w;
    w.Cold(1);

    w.BeginFrame();                       // held frame 1, picture 1
    Check(w.Arm(1, w.Picture(), false, false),
          "unsubmitted: the frame-1 capture could not be armed");
    PresentMirror(w, 0);                // producer_not_submitted -> close(facts.serial)
    w.BeginFrame();                       // held frame 2, picture 2
    PresentMirror(w, 0);
    w.BeginFrame();                       // held frame 3, picture 0
    PresentMirror(w, 0);
    w.BeginFrame();                       // held frame 4, picture 1: rotation
    Check(w.Arm(4, w.Picture(), true, true),
          "unsubmitted: the frame-4 capture could not be armed");
    PresentMirror(w, 0);                // the fresh capture applies

    Emit(w);

    Check(w.CountReason("producer_not_submitted") == 1,
          "unsubmitted: the refusal was not logged with reason=producer_not_submitted");
    Check(w.PendingSlot(1) && !w.slots[0].validExtent,
          "unsubmitted: close must invalidate presentation without dropping recording ownership");
    Check(w.composed.size() == 1 && w.composed[0] == 2,
          "unsubmitted: the later frame did not compose exactly its own capture");
    Check(w.tracker.Applications() == 1,
          "unsubmitted: the application count is wrong");
}

// H1 close-max must also invalidate unmatched, unsubmitted records without making
// their future submission/reset invisible. Otherwise their promised fence never
// retires and repeated skips exhaust the real four-slot pool.
void ClosedRecordingLifecycle()
{
    Wiring w;
    w.Cold(1);
    w.BeginFrame();
    Check(w.Arm(1, w.Picture(), false, false), "lifecycle: arm submitted-later record");
    Check(w.Arm(1, w.Picture(), false, false), "lifecycle: arm discarded-later record");
    PresentMirror(w, kPresentTest);
    w.BeginFrame(); // no matching capture; H1 must close max, not serial zero
    PresentMirror(w, 0);
    Check(w.PendingSlot(1) && w.PendingSlot(2), "lifecycle: recording ownership lost at close");
    Check(!w.slots[0].validExtent && !w.slots[1].validExtent, "lifecycle: H1 left an eligible capture");
    w.Submit(1);
    Check(w.slots[0].submitted, "lifecycle: wil submission hook could not observe closed recording");
    w.slots[0].ready = true; // exact fence-completion event, no timing or polling
    Check(!w.Apply(w.slots[0].picture), "lifecycle: late submission revived presentation");
    w.ResetRecording(2);
    Check(!w.PendingSlot(2) && w.slots[1].ready, "lifecycle: wil reset hook could not retire recording");
    w.CloseCaptures(kMaxSerial);
    Check(!w.PendingSlot(1) && w.slots[0].ready, "lifecycle: submitted slot did not become reusable");
    Emit(w);
}

// The identity is (generation, the held frame id), not the display counter: test
// presents advance the display counter but not the held id, a re-present of the
// same held id is refused as duplicate_frame, and every APPLY composes the
// capture armed for that held id.
void FrameIdRegression()
{
    Wiring w;
    w.Cold(1);

    w.BeginFrame();                       // held frame 1, picture 1
    Check(w.Arm(1, w.Picture(), true, true), "frame_id: the frame-1 capture could not be armed");
    PresentMirror(w, 0);                // applied for the held id 1
    PresentMirror(w, kPresentTest);     // the display counter advances, the id does not
    PresentMirror(w, kPresentTest);
    PresentMirror(w, 0);                // re-present of the SAME held id: duplicate_frame

    w.BeginFrame();                       // held frame 2, picture 2
    Check(w.Arm(2, w.Picture(), true, true), "frame_id: the frame-2 capture could not be armed");
    PresentMirror(w, 0);
    w.BeginFrame();                       // held frame 3, picture 0
    Check(w.Arm(3, w.Picture(), true, true), "frame_id: the frame-3 capture could not be armed");
    PresentMirror(w, 0);
    w.BeginFrame();                       // held frame 4, picture 1: rotation
    Check(w.Arm(4, w.Picture(), true, true), "frame_id: the frame-4 capture could not be armed");
    PresentMirror(w, 0);

    Emit(w);

    Check(w.displayIndex > w.heldFrame,
          "frame_id: the display counter did not diverge from the held frame id");
    Check(w.CountReason("duplicate_frame") == 1,
          "frame_id: the repeated held-id present was not refused as duplicate_frame");
    Check(w.CountToken("NR_XEFG_APPLY") == 4,
          "frame_id: the APPLY token count is not one per applied held id");
    Check(w.tracker.Applications() == 4,
          "frame_id: the application count is not one per applied held id");
    Check(w.composed.size() == 4,
          "frame_id: not every applied handoff composed exactly one capture");
    for (uint64_t applied = 1; applied <= 4; ++applied)
    {
        Check(w.SerialFrame(w.composed[applied - 1]) == applied,
              "frame_id: APPLY composed a capture that was not the accepted frame's (composed frame=" +
                  std::to_string(w.SerialFrame(w.composed[applied - 1])) + ")");
        Check(w.CountToken(("NR_XEFG_APPLY generation=1 frame=" + std::to_string(applied) + " ").c_str()) == 1,
              "frame_id: the APPLY line does not carry the held frame id " + std::to_string(applied));
    }
    Check(w.forwards == 7,
          "frame_id: the original proxy Present was not forwarded exactly once per present call");
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
        { "no_capture_close_gap", NoCaptureCloseGap },
        { "applied_then_compose_rejected", AppliedThenComposeRejected },
        { "unsubmitted_slot_survives_compose", UnsubmittedSlotSurvivesCompose },
        { "frame_id_regression", FrameIdRegression },
        { "closed_recording_lifecycle", ClosedRecordingLifecycle },
    };

    std::printf("XEFG_WIRING_REGRESSION production=dlssnr/DlssNr_XeFGHandoff.h seed_skips_close=%d "
                "seed_stale_compose=%d cases=%u\n",
#if defined(SEED_SKIPS_CLOSE)
                1,
#else
                0,
#endif
#if defined(SEED_STALE_COMPOSE)
                1,
#else
                0,
#endif
                (unsigned) std::size(cases));

    unsigned failed = 0;
    for (const Case& test : cases)
    {
        checks = 0;
        try
        {
            test.run();
            if (checks == 0)
                throw std::runtime_error("scenario exercised no wiring assertions");
            std::printf("CASE %-34s PASS (%d checks)\n", test.name, checks);
        }
        catch (const std::exception& error)
        {
            ++failed;
            std::printf("CASE %-34s FAIL %s\n", test.name, error.what());
        }
    }

    std::printf("XEFG_WIRING_REGRESSION cases=%u failed=%u\n", (unsigned) std::size(cases), failed);
    return failed == 0 ? 0 : 1;
}