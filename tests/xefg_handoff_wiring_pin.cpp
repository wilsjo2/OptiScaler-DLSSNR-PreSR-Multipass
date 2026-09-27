// tests/xefg_handoff_wiring_pin.cpp - production-route pin for the XeFG finished-picture NR handoff.
//
// WHAT THIS PROVES (and what it does not mirror):
//   - Sections A-C drive the ACTUAL decision headers (dlssnr/DlssNr_XeFGHandoff.h,
//     dlssnr/DlssNr_FinishedConsumer.h, dlssnr/DlssNr_FinishedReady.h,
//     dlssnr/DlssNr_QueueIdentity.h) with fixed contracts: duplicate/stale/unsubmitted/
//     not-ready/cancelled/ambiguous inputs must refuse with their named reasons, and any
//     applied violation fails named. Queue identities are COM-faithful doubles, never
//     preselected booleans: two distinct fake queues whose foreign-IID QueryInterface
//     clears the out pointer must not collapse to one identity, and incomplete
//     cross-queue input derived from them must be refused. Nothing is re-derived from
//     the code under test: every expectation is a literal written below.
//   - Sections D-F read the ACTUAL production sources given on the command line
//     (default: the repo's OptiScaler/framegen/xefg/XeFG_Dx12.cpp,
//     OptiScaler/hooks/FG_Hooks.cpp and OptiScaler/shaders/dlssnr/DlssNr_Late.inl) and
//     assert the wiring INSIDE the real function bodies: XeFG_Dx12::Present must publish
//     PublishFinishedConsumerState(consumes) before calling OwnedNrHandoff() under the same
//     consumes predicate, FGHooks::FGPresent must bypass the generic ApplyToFinishedPicture
//     on the owned route and report NR_XEFG_PRESENT, and both Late unwrap sites must
//     preserve the observed queue when the de-proxy query fails (PIN_B1_QUEUE).
//   - The runner (tests/run_xefg_production_route_smoke.ps1) re-runs this binary against
//     disposable copies with the call site removed (must fail PIN_XEFG_CALLSITE), the
//     bypass removed (must fail PIN_FG_BYPASS), and the Late unwrap preserve removed
//     (must fail PIN_B1_QUEUE). A seed that exits 0 means this pin no longer detects
//     its defect.
//
// Usage: xefg_handoff_wiring_pin.exe [XeFG_Dx12.cpp] [FG_Hooks.cpp] [DlssNr_Late.inl]
// Exit codes: 0 = every pin held; 1 = a pin failed (PIN_* marker printed); 2 = environment broken.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include <dlssnr/DlssNr_XeFGHandoff.h>
#include <dlssnr/DlssNr_FinishedConsumer.h>
#include <dlssnr/DlssNr_FinishedReady.h>
#include <dlssnr/DlssNr_QueueIdentity.h>

namespace
{
int g_failures = 0;

void Pin(bool held, const char* marker, const char* detail)
{
    if (held)
        return;
    ++g_failures;
    std::printf("%s: %s\n", marker, detail);
}

std::string ReadFile(const char* path, bool& ok)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        ok = false;
        return {};
    }
    std::ostringstream text;
    text << in.rdbuf();
    ok = true;
    return text.str();
}

// Extracts the body (without the outer braces) of the function whose definition starts at
// `signature`, e.g. "XeFG_Dx12::Present()". Strings, character literals and comments are
// skipped so braces inside them cannot corrupt the match. Empty when not found.
std::string FunctionBody(const std::string& source, const std::string& signature)
{
    const size_t at = source.find(signature);
    if (at == std::string::npos)
        return {};
    size_t open = source.find('{', at);
    if (open == std::string::npos)
        return {};
    size_t i = open + 1;
    int depth = 1;
    bool lineComment = false, blockComment = false, str = false, ch = false;
    for (; i < source.size(); ++i)
    {
        const char c = source[i];
        const char next = i + 1 < source.size() ? source[i + 1] : '\0';
        if (lineComment)
        {
            if (c == '\n')
                lineComment = false;
            continue;
        }
        if (blockComment)
        {
            if (c == '*' && next == '/')
            {
                blockComment = false;
                ++i;
            }
            continue;
        }
        if (str)
        {
            if (c == '\\')
                ++i;
            else if (c == '"')
                str = false;
            continue;
        }
        if (ch)
        {
            if (c == '\\')
                ++i;
            else if (c == '\'')
                ch = false;
            continue;
        }
        if (c == '/' && next == '/')
            lineComment = true;
        else if (c == '/' && next == '*')
            blockComment = true;
        else if (c == '"')
            str = true;
        else if (c == '\'')
            ch = true;
        else if (c == '{')
            ++depth;
        else if (c == '}')
        {
            if (--depth == 0)
                return source.substr(open + 1, i - open - 1);
        }
    }
    return {};
}
} // namespace

int main(int argc, char** argv)
{
    const char* xefgPath = argc > 1 ? argv[1] : "OptiScaler/framegen/xefg/XeFG_Dx12.cpp";
    const char* fgHooksPath = argc > 2 ? argv[2] : "OptiScaler/hooks/FG_Hooks.cpp";
    const char* latePath = argc > 3 ? argv[3] : "OptiScaler/shaders/dlssnr/DlssNr_Late.inl";

    using DlssNr::XeFGHandoff::Identity;
    using DlssNr::XeFGHandoff::Outcome;
    using DlssNr::XeFGHandoff::SkipReason;
    using DlssNr::XeFGHandoff::Tracker;

    // A. Tracker decision core (actual header): fixed refuse/apply contracts.
    {
        Tracker tracker;
        tracker.Reset(1);
        const uint64_t serial = tracker.OpenInterval(7);
        tracker.Submitted(serial);
        tracker.Ready(serial);
        Identity id {};
        id.generation = 1;
        id.frameId = 7;
        const Outcome happy = tracker.Handoff(id);
        Pin(happy.applied && happy.reason == SkipReason::None && happy.slot == serial, "PIN_TRACKER_HAPPY",
            "open/submit/ready must apply and consume the interval serial");

        const Outcome duplicate = tracker.Handoff(id);
        Pin(!duplicate.applied && duplicate.reason == SkipReason::DuplicateFrame, "PIN_TRACKER_DUPLICATE",
            "second handoff of the same identity must refuse duplicate_frame");
    }
    {
        Tracker tracker;
        tracker.Reset(1);
        const uint64_t serial = tracker.OpenInterval(7);
        tracker.Submitted(serial);
        tracker.Ready(serial);
        tracker.Reset(2); // swapchain recreation / discontinuity: old generation is stale
        Identity stale {};
        stale.generation = 1;
        stale.frameId = 7;
        const Outcome outcome = tracker.Handoff(stale);
        Pin(!outcome.applied && outcome.reason == SkipReason::NoCurrentCapture, "PIN_TRACKER_STALE",
            "old-generation handoff after reset must refuse no_current_capture and consume nothing");
    }
    {
        Tracker tracker;
        tracker.Reset(1);
        tracker.OpenInterval(9); // never submitted
        Identity id {};
        id.generation = 1;
        id.frameId = 9;
        const Outcome outcome = tracker.Handoff(id);
        Pin(!outcome.applied && outcome.reason == SkipReason::ProducerNotSubmitted, "PIN_TRACKER_UNSUBMITTED",
            "unsubmitted capture must refuse producer_not_submitted, never wait");
    }
    {
        Tracker tracker;
        tracker.Reset(1);
        const uint64_t serial = tracker.OpenInterval(9);
        tracker.Submitted(serial); // submitted but cross-queue input not fenced complete
        Identity id {};
        id.generation = 1;
        id.frameId = 9;
        const Outcome outcome = tracker.Handoff(id);
        Pin(!outcome.applied && outcome.reason == SkipReason::ProducerNotReady, "PIN_TRACKER_NOTREADY",
            "submitted-but-not-ready capture must refuse producer_not_ready, never wait");
    }
    {
        Tracker tracker;
        tracker.Reset(1);
        const uint64_t serial = tracker.OpenInterval(11);
        tracker.Cancelled(serial);
        Identity id {};
        id.generation = 1;
        id.frameId = 11;
        const Outcome outcome = tracker.Handoff(id);
        Pin(!outcome.applied && outcome.reason == SkipReason::NoCurrentCapture, "PIN_TRACKER_CANCELLED",
            "cancelled capture must refuse no_current_capture");
    }
    {
        Tracker tracker;
        tracker.Reset(1);
        tracker.OpenInterval(13);
        tracker.OpenInterval(13); // pipelined duplicate: never guessed
        Identity id {};
        id.generation = 1;
        id.frameId = 13;
        const Outcome outcome = tracker.Handoff(id);
        Pin(!outcome.applied && outcome.reason == SkipReason::AmbiguousCapture, "PIN_TRACKER_AMBIGUOUS",
            "pipelined frames must refuse ambiguous_capture");
    }

    // B. Producer readiness + queue identity (actual headers, COM-faithful doubles).
    // Two distinct fake queues answer a foreign IID the way a native D3D12 queue answers the
    // Streamline IID: E_NOINTERFACE with the out pointer cleared. The production unwrap sites
    // must preserve the observed pointer on that failure (DlssNr::UnwrapOrPreserveQueue), so
    // the identities below are derived through the REAL helper, never preselected booleans.
    struct FakeQueue
    {
        struct VTable
        {
            long (*QueryInterface)(void* self, const unsigned char iid[16], void** out);
        };
        const VTable* vtable;
        int tag;
    };
    static constexpr unsigned char kIID_IUnknown[16] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                      0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46 };
    static constexpr unsigned char kIID_Foreign[16] = { 0xE2, 0x44, 0xEC, 0xAD, 0xF0, 0x61, 0x45, 0xC3,
                                                     0xAD, 0x9F, 0x1B, 0x37, 0x37, 0x92, 0x84, 0xFF };
    auto fakeQI = [](void* self, const unsigned char iid[16], void** out) -> long
    {
        if (std::memcmp(iid, kIID_IUnknown, 16) == 0)
        {
            *out = self;
            return 0; // S_OK
        }
        *out = nullptr; // COM-faithful failure: the out pointer is cleared
        return static_cast<long>(0x80004002L); // E_NOINTERFACE
    };
    const FakeQueue::VTable kFakeVTable { fakeQI };
    FakeQueue submitQ { &kFakeVTable, 0 };
    FakeQueue otherQ { &kFakeVTable, 1 };
    // Mirror of Util::CheckForRealObject's success contract (S_OK plus non-null out).
    auto unwrap = [&](FakeQueue* observed, const unsigned char iid[16]) -> void*
    {
        void* out = nullptr;
        const long hr = observed->vtable->QueryInterface(observed, iid, &out);
        const bool unwrapped = (hr == 0 && out != nullptr);
        return DlssNr::UnwrapOrPreserveQueue(observed, unwrapped, out);
    };
    void* const storedSubmit = unwrap(&submitQ, kIID_Foreign); // submit site, native queue
    void* const storedOther = unwrap(&otherQ, kIID_Foreign);   // facts site, another native queue
    void* const storedKnown = unwrap(&submitQ, kIID_IUnknown); // known IID unwraps to self
    Pin(storedSubmit == &submitQ, "PIN_B1_QUEUE_PRESERVE",
        "failed unwrap must preserve the observed queue pointer");
    Pin(storedOther == &otherQ, "PIN_B1_QUEUE_PRESERVE_OTHER",
        "failed unwrap must preserve the observed queue pointer on a second queue");
    Pin(storedKnown == &submitQ, "PIN_B1_QUEUE_UNWRAP",
        "successful unwrap must yield the object identity");
    Pin(storedSubmit != storedOther, "PIN_B1_QUEUE_IDENTITY",
        "two distinct native queues must not collapse to one identity");
    const bool sameSubmitOther =
        DlssNr::SameRecordedQueue(storedSubmit, storedOther); // production comparison shape
    const bool sameSubmitSubmit = DlssNr::SameRecordedQueue(storedSubmit, storedSubmit);
    Pin(!sameSubmitOther, "PIN_B1_QUEUE_DISTINCT",
        "distinct recorded queues must compare unequal");
    Pin(sameSubmitSubmit, "PIN_B1_QUEUE_SAME",
        "identical recorded queues must compare equal");
    Pin(!DlssNr::SameRecordedQueue(nullptr, storedOther) && !DlssNr::SameRecordedQueue(storedSubmit, nullptr),
        "PIN_B1_NULL_NEVER_MATCHES", "a null record must never compare equal");
    Pin(DlssNr::FinishedInputReady(sameSubmitSubmit, 5, 10), "PIN_READY_SAME_QUEUE",
        "same-queue submission order must be sufficient");
    Pin(!DlssNr::FinishedInputReady(sameSubmitOther, 5, 10), "PIN_READY_CROSS_QUEUE_PENDING",
        "cross-queue incomplete input must not be ready (skip, never wait)");
    Pin(DlssNr::FinishedInputReady(sameSubmitOther, 10, 10), "PIN_READY_CROSS_QUEUE_DONE",
        "cross-queue completed input must be ready");
    Pin(!DlssNr::FinishedInputReady(sameSubmitOther, UINT64_MAX, 1), "PIN_READY_REMOVED_DEVICE",
        "UINT64_MAX completion must never be ready");
    Pin(!DlssNr::FinishedInputReady(sameSubmitSubmit, UINT64_MAX, 1), "PIN_READY_REMOVED_DEVICE_SAME_QUEUE",
        "UINT64_MAX completion must never be ready even same-queue");

    // C. Consumer publication policy (actual header): only the registered owner publishes, an
    // aborted release publishes nothing, and the default route admits captures.
    {
        const void* owner = reinterpret_cast<const void*>(0x1234);
        const void* other = reinterpret_cast<const void*>(0x5678);
        Pin(DlssNr::FinishedConsumerMayPublish(owner, owner), "PIN_CONSUMER_CLAIM_OWNER",
            "registered owner must be allowed to publish");
        Pin(!DlssNr::FinishedConsumerMayPublish(other, owner), "PIN_CONSUMER_CLAIM_NONOWNER",
            "non-owner must never publish");
        Pin(DlssNr::FinishedConsumerMayPublish(owner, nullptr), "PIN_CONSUMER_CLAIM_UNREGISTERED",
            "claim allowed while no owner is registered");
        Pin(DlssNr::FinishedConsumerIsOwner(owner, owner), "PIN_CONSUMER_IS_OWNER",
            "owner identity must hold");
        Pin(!DlssNr::FinishedConsumerIsOwner(other, owner), "PIN_CONSUMER_IS_NOT_OWNER",
            "stale instance must not own the consumer state");
        Pin(!DlssNr::FinishedConsumerIsOwner(nullptr, owner), "PIN_CONSUMER_NULL_NEVER_OWNS",
            "null instance must never own the consumer state");
        Pin(DlssNr::ReleasePublishesGeneric(true, false), "PIN_CONSUMER_RELEASE_PUBLISHES",
            "completed owner release must return the route to generic");
        Pin(!DlssNr::ReleasePublishesGeneric(true, true), "PIN_CONSUMER_RELEASE_ABORTED",
            "aborted reentrant release must publish nothing");
        Pin(!DlssNr::ReleasePublishesGeneric(false, false), "PIN_CONSUMER_RELEASE_NONOWNER",
            "non-owner release must publish nothing");
        Pin(DlssNr::FinishedConsumerAdmitsCapture(), "PIN_CONSUMER_DEFAULT_ADMITS",
            "default generic route must admit captures");
    }

    // D. Actual XeFG_Dx12::Present wiring, function-bounded.
    {
        bool ok = false;
        const std::string source = ReadFile(xefgPath, ok);
        if (!ok)
        {
            std::printf("PIN_ENV: cannot read %s\n", xefgPath);
            return 2;
        }
        const std::string present = FunctionBody(source, "XeFG_Dx12::Present()");
        if (present.empty())
        {
            std::printf("PIN_ENV: XeFG_Dx12::Present body not found in %s\n", xefgPath);
            return 2;
        }
        const size_t publish = present.find("PublishFinishedConsumerState(consumes)");
        const size_t handoff = present.find("OwnedNrHandoff()");
        const size_t gate = present.find("if (consumes)");
        Pin(publish != std::string::npos, "PIN_XEFG_CALLSITE",
            "Present must publish PublishFinishedConsumerState(consumes)");
        Pin(handoff != std::string::npos, "PIN_XEFG_CALLSITE",
            "Present must call OwnedNrHandoff()");
        Pin(publish != std::string::npos && handoff != std::string::npos && publish < handoff,
            "PIN_XEFG_CALLSITE", "publication must precede the handoff call");
        Pin(gate != std::string::npos, "PIN_XEFG_CALLSITE",
            "OwnedNrHandoff must run under the consumes predicate");
        const size_t consumes = present.find("consumes");
        const bool derivation =
            consumes != std::string::npos && present.find("dispatched", consumes) != std::string::npos &&
            present.find("IsActive()", consumes) != std::string::npos &&
            present.find("IsPaused()", consumes) != std::string::npos;
        Pin(derivation, "PIN_XEFG_CALLSITE",
            "consumes must derive from dispatched, active and unpaused in Present");
    }

    // E. Actual FGHooks::FGPresent wiring, function-bounded: generic NR bypass + report.
    {
        bool ok = false;
        const std::string source = ReadFile(fgHooksPath, ok);
        if (!ok)
        {
            std::printf("PIN_ENV: cannot read %s\n", fgHooksPath);
            return 2;
        }
        const std::string fgPresent = FunctionBody(source, "FGHooks::FGPresent(");
        if (fgPresent.empty())
        {
            std::printf("PIN_ENV: FGHooks::FGPresent body not found in %s\n", fgHooksPath);
            return 2;
        }
        const size_t apply = fgPresent.find("ApplyToFinishedPicture(This, state.currentCommandQueue)");
        Pin(apply != std::string::npos, "PIN_FG_BYPASS", "generic ApplyToFinishedPicture call must exist");
        bool guarded = false;
        if (apply != std::string::npos)
        {
            const size_t window = apply >= 120 ? 120 : apply;
            guarded = fgPresent.substr(apply - window, window).find("!xefgOwnedHandoff") != std::string::npos;
        }
        Pin(guarded, "PIN_FG_BYPASS",
            "generic ApplyToFinishedPicture must be bypassed on the owned handoff route");
        Pin(fgPresent.find("XeFGHandoffSince") != std::string::npos, "PIN_FG_REPORT",
            "FGPresent must read the decided handoff record via XeFGHandoffSince");
        Pin(fgPresent.find("NR_XEFG_PRESENT") != std::string::npos, "PIN_FG_REPORT",
            "FGPresent must report NR_XEFG_PRESENT after the original proxy Present");
    }

    // F. Actual Late unwrap sites, function-bounded (B1 queue-identity gate). Both the submit
    // site and the facts site must preserve the observed queue when the de-proxy query fails;
    // a native failed QueryInterface clears the out pointer, so keeping it would collapse two
    // distinct queues to null == null and admit incomplete cross-queue work as same-queue.
    {
        bool ok = false;
        const std::string source = ReadFile(latePath, ok);
        if (!ok)
        {
            std::printf("PIN_ENV: cannot read %s\n", latePath);
            return 2;
        }
        const std::string facts = FunctionBody(source, "PendingCaptureFacts(ID3D12Resource* color");
        if (facts.empty())
        {
            std::printf("PIN_ENV: PendingCaptureFacts body not found in %s\n", latePath);
            return 2;
        }
        const std::string submitted = FunctionBody(source, "FinishedPictureSubmitted(ID3D12CommandQueue* queue");
        if (submitted.empty())
        {
            std::printf("PIN_ENV: FinishedPictureSubmitted body not found in %s\n", latePath);
            return 2;
        }
        Pin(facts.find("UnwrapOrPreserveQueue(queue, unwrapped, realQueue)") != std::string::npos,
            "PIN_B1_QUEUE",
            "PendingCaptureFacts must preserve the queue when the de-proxy query fails");
        Pin(submitted.find("UnwrapOrPreserveQueue(queue, unwrapped, realQueue)") != std::string::npos,
            "PIN_B1_QUEUE",
            "FinishedPictureSubmitted must preserve the queue when the de-proxy query fails");
    }

    if (g_failures == 0)
        std::printf("XEFG production route pin: all pins held.\n");
    return g_failures == 0 ? 0 : 1;
}
