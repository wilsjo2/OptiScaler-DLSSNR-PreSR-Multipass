// Vendored from optiscaler/OptiScaler @ bb1619ec1eac76e8109d3adf9c795d6d37ba1ba4 (OptiScaler/proxies/XeFGPacing.h).
// Upstream authorship (GPL-3.0). Do not edit in place - re-fetch the pinned blob instead.
// sha256 of the vendored body (from line 4 down): 453270a4cd3f7944b4f1259f54727f76d5d256c81675b92d1059ab14cc92a507

#pragma once

// Spreads Intel's generated frames out in time, so that multi frame generation
// above 2X stops arriving as one clump followed by a gap.
//
// The provider already knows what the spacing should be. Once per burst it
// works out how long a real frame took and divides it by the number of frames
// the burst will deliver:
//
//     0x220254  mov  r9, [r8+8]        ; count = generated frames in this burst
//     0x220258  lea  rcx, [r9+1]       ; count + 1 = the multiplier
//     0x22025E  mov  rax, [rbp-0x41]   ; duration of the real frame
//     0x220262  div  rcx
//     0x220265  mov  r12, rax          ; -> duration / multiplier, per frame
//
// but then the present loop that follows hands every generated frame straight
// to the swapchain back to back:
//
//     0x220280  ...                    ; loop over rbx = 1 .. count-1
//     0x2202E8  call 0x1800025c0       ; -> jmp 0x18021f730, present one frame
//     0x2202ED  ...
//     0x22030A  jb   0x180220280
//
// and `r12` is not consumed until the loop has finished, by the FPS limiter
// block at 0x220317 - which is itself only reached when count > 2 and two other
// conditions hold. So the burst goes out as fast as the swapchain accepts it and
// the limiter then holds the *last* frame of the burst back, which is what reads
// as frames being bunched up and out of order above 2X.
//
// The last frame is not presented from the loop at all. It goes out from the
// call at 0x220462, after the loop and after the provider has submitted the
// burst, so it needs pacing of its own:
//
//     0x220411  call 0x180003100       ; -> jmp 0x18021ee30, submit the burst
//     0x220445  mov  byte [rsp+0x30], 0
//     0x220462  call 0x1800025c0       ; present the last generated frame
//
// Fixing it means pacing each generated frame, which means getting in between
// the loop and the present. Every one of those presents goes through the 5 byte
// thunk at 0x25C0, and that thunk is followed by 11 bytes of int3 padding, so
// it is 16 bytes that contain nothing but a jump. That is room for a 14 byte
// `jmp qword ptr [rip+0]`, so the thunk can be redirected anywhere in the
// address space without relocating a single instruction and without a code cave.
//
// The thunk has two other callers (0x220A52, 0x220DAD), so the detour checks
// _ReturnAddress() and paces only the two sites above; every other present is
// forwarded untouched.
//
// The spacing is not invented here. The provider has a frame scheduler of its
// own - it just never calls it for the frames the loop presents, only for the
// burst's last one. So each of those presents is handed to 0x21EE30 with the
// arguments the present path already has, and the provider works out the
// presentation time itself. A wall clock does the same job when the scheduler
// cannot be reached; waiting on the provider's D3D12 fence as well was tried and
// then removed, see 08-PACING.md §7.7 for the measurements that killed it.

#include <intrin.h>
#include "SysUtils.h"
#include "Logger.h"
#include "Config.h"

namespace XeFGPacing
{
// --- provider internals, libxess_fg.dll 1.3.1.78 ------------------------

constexpr uint32_t PresentThunkRva = 0x25C0;
constexpr uint32_t NativePresentRva = 0x21F730;

// Return address of the `call` in the burst loop - i.e. the exact identity
// of the call site we want to pace.
constexpr uint32_t PacedCallerRva = 0x2202ED;

// The burst's last generated frame is not presented from the loop. It goes
// out after the loop and after the provider's own submit, from the call at
// 0x220462, with the same burst in arg5 but a different arg6 and arg7 = 0
// instead of 1:
//
//     0x220445  mov  byte ptr [rsp+0x30], 0   ; arg7 = 0
//     0x220462  call 0x25c0
//
// Left alone, every burst ends with two frames carrying the same timestamp:
// at 4X the delivered pattern is 0, i, 2i, 2i instead of 0, i, 2i, 3i.
constexpr uint32_t LastFrameCallerRva = 0x220467;

// The provider does space that last frame itself - but only from the limiter
// block at 0x220317, and only when all three of its conditions hold:
//
//     0x220317  cmp  byte  ptr [rsi + 0x340], 0   ; enabled
//     0x220324  cmp  qword ptr [r8 + 8], 2        ; count > 2, i.e. 4X up
//     0x22032F  cmp  dword ptr [r8 + 0x28], 0     ; per-burst field is 0
//
// Mirroring the condition keeps exactly one of us waiting on that frame
// rather than two waits stacking into one long frame.
constexpr uint32_t LimiterEnabledOffset = 0x340;
constexpr uint32_t BurstLimiterField = 0x28;

// There is a D3D12 fence in the swapchain context (ctx+0x328, event at
// ctx+0x330) and the provider knows how to wait on it properly - see
// 0x21EEF2..0x21EF2D, and the fence value it waits for is looked up out of a
// 9 entry ring at ctx+0x168. Waiting on that same fence from here was tried
// and **removed again**: `WaitForSingleObject` rounds a short timeout up to
// the system timer tick, so every wait that expired overshot by ~7 ms, over
// half of them expired, and the measured gap between generated frames ended
// up 30-40% above the target the pacing was aiming for. The full numbers and
// the reasoning are in 08-PACING.md §4.6 and §7.7.
//
// The wall clock is now only the fallback, used when the scheduler below
// cannot be reached. Its real frame period is measured across two frames
// whose spacing is set by the scheduler's own timebase instead of being
// imposed, so it is reported rather than used to decide anything.

// --- the provider's own frame scheduler --------------------------------

// 0x3100 is another entry in the same thunk table as 0x25C0, and it is not
// what it looks like from inside 0x21EE30. That function is not "wait for
// the fence" - the fence wait is a short block it can skip. What it really
// does is schedule the frame: look the frame's record up in the ring at
// ctx+0x168 by burst->tag, stamp it with the current counter, then have
// 0x224B30 (via 0x3430) compute a presentation time from
//
//     f * arg5 / (count + 1) * K   clamped against  arg5 * *(arg4+8) / (count+1)
//     ... +/- corrections read out of the record
//
// and hand that time to 0x7A30. Its arg5 is the frame's *index* in the
// burst, and the provider's own call site (0x220411) passes burst->count -
// the index of the last generated frame, which is the only one that site is
// responsible for. So above 2X every intermediate frame goes out with no
// presentation time computed for it at all, which is a much better
// explanation of the ordering trouble than the missing fence wait was.
//
// All five of its arguments are reachable from the present's own arguments:
// arg2 is the burst (present arg5 - 0x38), arg5 is the index the loop is
// already walking, arg3 is burst[0xC0] & 1, and arg4 is what 0x4DA0 writes
// when given the ring:
//
//     0x22023D  lea  r14, [rsi + 0x168]   ; ring = ctx + 0x168
//     0x220244  lea  rdx, [rbp - 0x49]
//     0x220248  mov  rcx, r14
//     0x22024B  call 0x4da0               ; -> arg4
//
// This is what dashdogy's patch does - one schedule call per generated
// frame. He rebuilds those arguments through a 1500 byte helper (0x3F570)
// because he hooks without the present's arguments in hand; from inside the
// present they are free.
//
// A run with only a forwarding detour on the thunk confirmed all of it
// before any of this was written: index came back equal to the burst count
// every time, the gate byte equalled burst[0xC0] & 1 every time, and the
// arg4 block was the same non-null pointer with a fresh median each frame.
//
// ScheduleFrame is the call. It is made for frames 1..count-1 only - the
// provider's own site keeps the last one.
constexpr uint32_t SchedThunkRva = 0x3100;
constexpr uint32_t SchedFnRva = 0x21EE30;

// arg2 of the scheduler is present arg5 - 0x38, so the gate byte the
// present path reads into r8 at 0x22040A sits at 0xC0 of it.
constexpr uint32_t BurstGateOffset = 0xC0;

// The scheduler is not always on. Its entry gate is 0xdbb0 -> 0x21ed40, and
// the whole of it is two bytes of the context:
//
//     0x21ED40  cmp   byte ptr [rcx + 0x340], 0
//     0x21ED47  je    0x21ed4c
//     0x21ED49  xor   al, al                    ; [ctx+0x340] != 0 -> false
//     0x21ED4B  ret
//     0x21ED4C  movzx eax, byte ptr [rcx + 0x341] ; else return [ctx+0x341]
//
// 0x21EE30 then does `test al, al; je 0x21efa7`, i.e. it bails unless that
// second byte is set too. And 0x340 is the same byte the provider's FPS
// limiter is gated on (0x220317 `cmp byte ptr [rsi + 0x340], 0`), so the two
// are mutually exclusive by construction: when the limiter owns the pacing,
// the scheduler is off. Calling into it anyway is a silent no-op, which is
// exactly what a build that trusts the hook alone would do.
constexpr uint32_t SchedLimiterOffset = 0x340;
constexpr uint32_t SchedEnableOffset = 0x341;

// arg4 comes out of 0x4DA0, which is a thunk for 0x224CF0. That function
// walks the nine slot ring inside the container at ctx+0x168 and hands back
// the median duration of its slots. Disassembled in full (318 bytes,
// 0x224CF0..0x224E2E) it has exactly three stores through the out pointer:
//
//     0x224DD1  mov     [r14], rbx      ; number of usable samples
//     0x224DD4  mov     [r14 + 8], rax  ; the median itself
//     0x224DE6  movups  [r14], xmm6     ; ...or 16 zero bytes, when none
//
// so the block is 16 bytes and nothing more. 0x224B30 only ever reads +8 off
// it (`imul rax, [r14 + 8]` at 0x224B7B) - the +0x14 and +0x20 it also
// touches are fields of the ring record, not of this. That settles the one
// thing that had to be known before calling it, and it matches what the
// observation run showed: +0 unread, +8 a plausible duration.
constexpr uint32_t RingSnapshotFnRva = 0x224CF0;
constexpr uint32_t RingOffset = 0x168;

// Every frame that gets a deadline ends up waiting in 0x7A30 -> 0x225070,
// and the value it waits on is produced by 0x3430 -> 0x224B30. That function
// is the whole of the provider's timing model:
//
//     deadline = base
//              + min( index * median / (count + 1),          ; nanoseconds
//                     f * index / (count + 1) * 1000000 )   ; ms -> ns
//              - recv * 1000000
//
// where `median` is the ring half of arg4 (`[r9 + 8]`), `recv` is
// `(float)[lookup + 0x14]`, f is `clamp(ring[0x1B8], 0.125, 500.0)` read as
// a float in milliseconds, and `base` is a nanosecond timestamp written in
// exactly one place: the fence block, which only runs when
// `gate != 0 && index == 1`.
//
// Two consequences, both of them visible in a run:
//
//   * `base` is fresh only for index 1. For every other index it is whatever
//     that stack slot happened to hold, so the provider's own call - the one
//     that carries index == count, which above 2X is the only call it makes -
//     computes its deadline from garbage. 2X is the one multiplier where its
//     own call has index 1, and it is the one multiplier that ever worked.
//   * the min() clamps the step down to f, and f is smaller than a real frame
//     interval. Measured at 4X: 8.5 ms against a 20.6 ms frame, so the step
//     came out at 2.13 ms where the burst needed 5.03 ms and its frames were
//     pushed out in a clump instead of spread.
//
// Both are repaired at one choke point, because 0x3430 is called once per
// scheduled frame - ours and the provider's alike - and it hands back a
// writable out pointer. See TsDetour.
constexpr uint32_t TimestampThunkRva = 0x3430;
constexpr uint32_t TimestampFnRva = 0x224B30;

// The float 0x224B30 clamps, at ring + 0x1B8. Read here only to work out how
// much the min() removed; the value itself is never written.
constexpr uint32_t RingMeasuredOffset = 0x1B8;

inline const uint8_t TimestampThunkExpected[16] = {
    0xE9, 0xFB, 0x16, 0x22, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
};

inline const uint8_t SchedThunkExpected[16] = {
    0xE9, 0x2B, 0xBD, 0x21, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
};

// The provider's own jmp plus the int3 padding that follows it.
constexpr uint32_t PresentThunkSize = 16;

inline const uint8_t PresentThunkExpected[PresentThunkSize] = {
    0xE9, 0x6B, 0xD1, 0x21, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
};

// int64_t present(void* ctx, uint32_t a2, uint32_t a3, uint64_t a4,
//                 void* arg5, void* arg6, uint64_t arg7)
//
// The argument names are the register/stack slots the burst loop fills in at
// 0x2202C3..0x2202D5; only arg5/arg6/arg7 are looked at (see TryPace).
using PresentFn = int64_t (*)(void*, uint32_t, uint32_t, uint64_t, void*, void*, uint64_t);

// How many real frame periods to keep for the median.
constexpr int32_t SampleCount = 15;

inline uint8_t* g_base = nullptr;
inline PresentFn g_native = nullptr;
inline bool g_enabled = false;

// Written and read only by the swapchain present thread; the install path
// runs before the provider is ever entered.
inline LARGE_INTEGER g_freq {};
inline int64_t g_lastBurstQpc = 0;
inline int64_t g_periodNs = 0;
inline int64_t g_intervalQpc = 0;
inline int64_t g_targetQpc = 0;
inline int64_t g_samples[SampleCount] {};
inline int32_t g_sampleCount = 0;
inline int32_t g_samplePos = 0;
inline int64_t g_pacedFrames = 0;
inline int64_t g_pacedBursts = 0;
inline bool g_loggedFirstBurst = false;

// bool sched(void* ctx, void* burst, uint8_t gate, void* timing, uint32_t index)
//
// and the block 0x4DA0 fills, which is a 16 byte {sample count, median} pair.
using SchedFn = bool (*)(void*, void*, uint8_t, void*, uint32_t);
using RingSnapshotFn = void* (*) (void*, void*);

inline SchedFn g_schedNative = nullptr;
inline RingSnapshotFn g_ringSnapshot = nullptr;

// Logging is bounded on purpose: three lines per multiplier is enough to see
// the schedule go out, and this runs on the present thread.
inline int32_t g_schedLogged = 0;
inline uint64_t g_schedLastIndex = ~0ull;
inline int64_t g_schedCalls = 0;

// Whether the call was refused (its gate said no) and how long it spent
// inside. The refusals are the whole point: a refused call and a completed
// one are indistinguishable from the outside, and pretending otherwise is
// what made the previous build quietly do nothing.
inline int64_t g_schedRefused = 0;
inline int64_t g_schedWaitQpc = 0;
inline int64_t g_lastMultiplier = 0;

// Blocking the current burst has cost so far, and what is left of the period
// once that is taken back out of it. See RenderTimeMs.
inline int64_t g_burstBlockQpc = 0;
inline int64_t g_renderTimeNs = 0;

// What the provider was handed as `frameRenderTime`, in milliseconds.
// XeFG_Dx12.cpp is the only caller. Recorded rather than returned because the
// pacing report is where it needs to be visible, next to the period it came
// out of - that pairing is the entire point of the change.
inline double g_fedFrameTimeMs = 0.0;

constexpr int32_t SchedLogLimit = 12;

// void* timestamp(void* a1, int64_t* out, void* lookup, void* timing,
//                 uint32_t index, uint32_t countPlus1)
//
// The fifth and sixth arguments arrive on the stack and are the two the
// formula divides by, which is what makes this hook able to identify a frame
// without any of the burst's own state.
using TimestampFn = void* (*) (void*, int64_t*, void*, void*, uint32_t, uint32_t);

inline TimestampFn g_tsNative = nullptr;

// ctx + 0x168, captured whenever a frame is scheduled. Until the first of
// those the deadline hook leaves the provider's own arithmetic alone.
inline uint8_t* g_ring = nullptr;

// The schedule TsDetour hands out, in nanoseconds. `g_burstStepNs` is the
// step the burst wants - median / (count + 1) - and is held fixed for the
// whole burst so its frames cannot drift apart from each other.
inline int64_t g_nextDeadlineNs = 0;
inline int64_t g_burstStepNs = 0;
inline uint32_t g_lastTsIndex = 0;
inline uint32_t g_lastTsCountPlus1 = 0;

inline int64_t g_tsCalls = 0;
inline int64_t g_tsRebased = 0;
inline int64_t g_tsClamped = 0;

// The provider's own gate, transcribed. Nothing is handed to the scheduler
// unless it would actually run.
inline bool SchedulerUsable(void* ctx)
{
    const auto* ctxBytes = static_cast<const uint8_t*>(ctx);

    return ctxBytes[SchedLimiterOffset] == 0 && ctxBytes[SchedEnableOffset] != 0;
}

// Measured spacing between two consecutive paced frames, and the window the
// numbers are reported over. The gap is the only direct evidence that the
// frames come out evenly - everything else here is a claim about intent.
inline int64_t g_lastPacedQpc = 0;
inline int64_t g_gapSumQpc = 0;
inline int64_t g_gapMinQpc = 0;
inline int64_t g_gapMaxQpc = 0;
inline int32_t g_gapCount = 0;
inline int64_t g_statsDeadline = 0;

constexpr int64_t StatsWindowNs = 5000000000LL; // 5 s

inline int64_t NsFromQpc(int64_t delta) { return g_freq.QuadPart > 0 ? (delta * 1000000000LL) / g_freq.QuadPart : 0; }

inline int64_t QpcFromNs(int64_t ns) { return g_freq.QuadPart > 0 ? (ns * g_freq.QuadPart) / 1000000000LL : 0; }

inline double MsFromQpc(int64_t qpc) { return g_freq.QuadPart > 0 ? (qpc * 1000.0) / g_freq.QuadPart : 0.0; }

inline void ResetStats(int64_t nowQpc)
{
    g_gapSumQpc = 0;
    g_gapMinQpc = 0;
    g_gapMaxQpc = 0;
    g_gapCount = 0;
    g_statsDeadline = nowQpc + QpcFromNs(StatsWindowNs);
}

// The burst's spacing only exists on screen if the frames actually arrive
// apart, so report what was measured rather than what was intended. One line
// every few seconds: if `avg` tracks `target` and max stays near it, the
// pacing is real; if max is a multiple of target, frames are still clumping.
inline void ReportStats(int64_t multiplier)
{
    if (g_gapCount <= 0)
        return;

    // The real frame period is here because it decides whether even perfect
    // pacing can be shown: a present rate above the panel's refresh rate
    // cannot be displayed evenly no matter how well the frames are spaced.
    //
    // `refused` is the number that says whether the scheduler was really
    // doing anything. If it equals the call count, every call was a no-op
    // and the wall clock should have been the one pacing.
    // The pairing below is load bearing, and it fails silently: fmt ignores
    // surplus arguments, so an over-supplied call looks fine, but a `{:.2f}`
    // landing on one of the counters throws fmt::format_error at runtime.
    // LOG_INFO has no catch around it (see SysUtils.h) and this runs on the
    // present thread, so that throw unwinds straight out of the hook and the
    // pacing stops for the rest of the session - with nothing in the log to
    // say so, because the line that would have said it is the one throwing.
    // Count the placeholders before touching this. check_reportstats_args.py
    // in _analysis does it, and prints the pairing as compiled.
    LOG_INFO("XeFG pacing: {}X, real frame {:.2f} ms ({:.1f} fps), target {:.2f} ms/frame; "
             "gap {:.2f} avg / {:.2f} min / {:.2f} max ms over {} frames; "
             "scheduler {} calls, {} refused, {:.2f} ms avg inside; "
             "deadlines {} calls, {} rebased, {} clamped; render-est {:.2f} ms, fed {:.2f} ms",
             multiplier, g_periodNs / 1000000.0, g_periodNs > 0 ? 1e9 / g_periodNs : 0.0, MsFromQpc(g_intervalQpc),
             MsFromQpc(g_gapSumQpc / g_gapCount), MsFromQpc(g_gapMinQpc), MsFromQpc(g_gapMaxQpc), g_gapCount,
             g_schedCalls, g_schedRefused, g_schedCalls > 0 ? MsFromQpc(g_schedWaitQpc / g_schedCalls) : 0.0, g_tsCalls,
             g_tsRebased, g_tsClamped, g_renderTimeNs / 1000000.0, g_fedFrameTimeMs);
}

// What the frames actually did, as opposed to what the pacing asked them to
// do. Shared by both pacing paths so the numbers stay comparable between
// builds - the scheduler path is not exempt from being measured.
inline void RecordGap(int64_t endQpc)
{
    if (g_lastPacedQpc != 0)
    {
        const int64_t gap = endQpc - g_lastPacedQpc;

        g_gapSumQpc += gap;

        if (g_gapCount == 0 || gap < g_gapMinQpc)
            g_gapMinQpc = gap;

        if (gap > g_gapMaxQpc)
            g_gapMaxQpc = gap;

        g_gapCount++;
    }

    g_lastPacedQpc = endQpc;

    if (g_statsDeadline == 0)
        ResetStats(endQpc);
    else if (endQpc >= g_statsDeadline)
    {
        ReportStats(g_lastMultiplier);
        ResetStats(endQpc);
    }
}

// Median of the recent real frame periods. A median rather than a mean so
// that one hitch does not stretch the pacing for the next few seconds.
inline void PushPeriod(int64_t ns)
{
    if (ns <= 0)
        return;

    g_samples[g_samplePos] = ns;
    g_samplePos = (g_samplePos + 1) % SampleCount;

    if (g_sampleCount < SampleCount)
        g_sampleCount++;

    int64_t sorted[SampleCount];
    memcpy(sorted, g_samples, sizeof(int64_t) * g_sampleCount);

    for (int32_t i = 1; i < g_sampleCount; i++)
    {
        int64_t key = sorted[i];
        int32_t j = i - 1;

        while (j >= 0 && sorted[j] > key)
        {
            sorted[j + 1] = sorted[j];
            j--;
        }

        sorted[j + 1] = key;
    }

    g_periodNs = sorted[g_sampleCount / 2];
}

// Burst bookkeeping, shared by both pacing paths: the gap between two burst
// starts is one real frame, and the median of recent ones divided by the
// multiplier is the interval they are both aiming at.
inline void NoteFrame(uint64_t index, uint64_t count, int64_t nowQpc)
{
    if (index > 1)
        return;

    if (g_lastBurstQpc != 0)
        PushPeriod(NsFromQpc(nowQpc - g_lastBurstQpc));

    // The period just measured is taken at the present, so it contains the
    // blocking the pacing itself did. Handing that back to the provider is
    // what locks the real frame rate: it is asked to fill a period that only
    // exists because it was asked to fill it, and the fixed point is
    //
    //     period = renderTime + period * count / (count + 1)
    //
    // so the higher the multiplier, the longer the "real" frame gets. Taking
    // the block back out leaves the part of the frame the game actually got
    // to spend rendering - the quantity the provider means by
    // `frameRenderTime` and sizes the generated interval from.
    //
    // It is an upper bound on that quantity, not an equality: the provider's
    // own wait on the burst's last frame happens inside this window too and
    // is not measured here. Under-reporting would be the dangerous direction
    // (it would ask for a burst shorter than the frames can be produced in),
    // over-reporting only leaves some of the lock in place.
    if (g_periodNs > 0)
    {
        const int64_t blockNs = NsFromQpc(g_burstBlockQpc);

        g_renderTimeNs = g_periodNs > blockNs ? g_periodNs - blockNs : 0;
    }

    g_burstBlockQpc = 0;
    g_lastBurstQpc = nowQpc;
    g_pacedBursts++;

    if (g_periodNs > 0 && g_freq.QuadPart > 0)
        g_intervalQpc = QpcFromNs(g_periodNs / (static_cast<int64_t>(count) + 1));
}

// Milliseconds of frame the game actually got to render, i.e. the measured
// period with this burst's blocking taken back out. Zero until a burst has
// been measured, which is the caller's signal to keep its own fallback.
inline double RenderTimeMs() { return g_renderTimeNs > 0 ? g_renderTimeNs / 1000000.0 : 0.0; }

inline void NoteFedFrameTime(double ms) { g_fedFrameTimeMs = ms; }

// The provider runs this on the swapchain present thread, so the wait has to
// be cheap and must not overshoot by much. Sleep(1) is deliberately avoided:
// its granularity is a full timer tick, which is coarser than the interval
// we are trying to hit.
inline void WaitUntil(int64_t targetQpc)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);

    const int64_t yieldBelow = QpcFromNs(200000); // 200 us

    while (now.QuadPart < targetQpc)
    {
        if ((targetQpc - now.QuadPart) > yieldBelow)
            Sleep(0);
        else
            YieldProcessor();

        QueryPerformanceCounter(&now);
    }
}

// `index` is the frame's position in the burst (0 is the real frame, and the
// loop presents 1..count-1) and `count` is the number of generated frames,
// so a burst delivers count + 1 frames in total.
inline void PaceFrame(uint64_t index, uint64_t count)
{
    const int64_t multiplier = static_cast<int64_t>(count) + 1;

    // 2X is what the provider already schedules correctly, and it is the
    // only setting where this block is bypassed anyway.
    if (multiplier <= 2)
        return;

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);

    NoteFrame(index, count, now.QuadPart);

    if (index <= 1)
    {
        // Nothing to pace against until a whole real frame has been seen.
        if (g_periodNs <= 0 || g_intervalQpc <= 0)
            return;

        g_targetQpc = now.QuadPart + g_intervalQpc;

        if (!g_loggedFirstBurst)
        {
            g_loggedFirstBurst = true;
            LOG_INFO("XeFG pacing: {}X, real frame {:.3f} ms -> {:.3f} ms per generated frame", multiplier,
                     g_periodNs / 1000000.0, (g_periodNs / multiplier) / 1000000.0);
        }
    }
    else
    {
        if (g_intervalQpc <= 0)
            return;

        g_targetQpc += g_intervalQpc;
    }

    // If a hitch has already put us a whole frame behind, give up on this
    // burst rather than stalling the present thread trying to catch up.
    const int64_t latest = now.QuadPart + QpcFromNs(g_periodNs);

    if (g_targetQpc > latest)
        g_targetQpc = latest;

    // Hold the wall clock line the provider already worked out for this
    // burst. Nothing else - see the note at the top of the file for why the
    // GPU fence is not consulted here.
    WaitUntil(g_targetQpc);

    g_pacedFrames++;

    LARGE_INTEGER end;
    QueryPerformanceCounter(&end);

    RecordGap(end.QuadPart);
}

// Hands one generated frame to the provider's own scheduler.
//
// The provider calls 0x21EE30 once per burst and only for the frame the
// post-loop site at 0x220462 presents, so at 4X and up the frames the loop
// presents never get a presentation time computed for them at all. This is
// that missing call - one per frame, which is what dashdogy's patch does.
//
// Every argument is in hand here. The provider's own order is snapshot the
// ring first (0x22024B) and schedule later (0x220411), even though the two
// are a whole loop apart, so this keeps the same order.
//
// 2X never reaches here: count is 1, the loop runs 1..0 and does not
// execute, and the caller leaves the last frame to the provider.
inline void ScheduleFrame(void* ctx, uint8_t* burst, uint64_t index)
{
    if (g_schedNative == nullptr || g_ringSnapshot == nullptr)
        return;

    g_ring = reinterpret_cast<uint8_t*>(ctx) + RingOffset;

    // 0x224CF0 writes +0 and +8, and reads nothing back, so a fresh block per
    // call is correct - the provider's own is a reused stack slot, and its
    // median does change from frame to frame.
    alignas(16) uint8_t timing[0x20] {};

    g_ringSnapshot(reinterpret_cast<uint8_t*>(ctx) + RingOffset, timing);

    const uint8_t gate = burst[BurstGateOffset] & 1;
    const uint64_t count = *reinterpret_cast<uint64_t*>(burst + 8);

    LARGE_INTEGER before;
    QueryPerformanceCounter(&before);

    NoteFrame(index, count, before.QuadPart);

    const bool scheduled = g_schedNative(ctx, burst, gate, timing, static_cast<uint32_t>(index));

    LARGE_INTEGER after;
    QueryPerformanceCounter(&after);

    g_schedCalls++;

    if (!scheduled)
        g_schedRefused++;

    g_schedWaitQpc += after.QuadPart - before.QuadPart;
    g_burstBlockQpc += after.QuadPart - before.QuadPart;

    if (g_schedLogged < SchedLogLimit && index != g_schedLastIndex)
    {
        g_schedLastIndex = index;
        g_schedLogged++;

        // "refused" is the case that matters: the provider's gate said no, so
        // no presentation time was computed and nothing was waited for. It
        // costs almost nothing and is invisible without this word.
        LOG_INFO("XeFG pacing: frame {}/{} -> {} in {:.3f} ms "
                 "(ctx+0x340 {} ctx+0x341 {}, ring {} samples, median {})",
                 index, count + 1, scheduled ? "scheduled" : "REFUSED", MsFromQpc(after.QuadPart - before.QuadPart),
                 reinterpret_cast<uint8_t*>(ctx)[SchedLimiterOffset],
                 reinterpret_cast<uint8_t*>(ctx)[SchedEnableOffset], *reinterpret_cast<uint64_t*>(timing),
                 *reinterpret_cast<uint64_t*>(timing + 8));
    }

    RecordGap(after.QuadPart);
}

// The provider's limiter already holds the burst's last frame back under
// these conditions, so we must not hold it back a second time.
inline bool ProviderPacesLastFrame(void* ctx, const uint8_t* burst, uint64_t count)
{
    if (count <= 2)
        return false;

    const auto* ctxBytes = reinterpret_cast<const uint8_t*>(ctx);

    return ctxBytes[LimiterEnabledOffset] != 0 && *reinterpret_cast<const uint32_t*>(burst + BurstLimiterField) == 0;
}

// At the call site the burst struct is at [rbp+0x67], and the loop sets up
// the call as
//
//     arg4 = burst->tag, also the key into the provider's frame ring
//     arg5 = burst + 0x38
//     arg6 = burst->frames + index * 8
//     arg7 = 1
//
// so the burst and the frame's position in it are both recoverable from the
// call's own arguments, without touching a caller register.
//
// `isLast` marks the post-loop present of the burst's final generated frame.
// It carries the same burst in arg5, but arg6 is not a frame array entry and
// arg7 is 0, so the position has to come from the burst instead:
// the loop covers 1..count-1, which leaves exactly count for this one.
inline void TryPace(void* ctx, void* arg5, void* arg6, uint64_t arg7, bool isLast)
{
    if (arg5 == nullptr)
        return;

    // arg7 is 1 for the loop's presents and 0 for the last one; anything
    // else is not a call this hook should be touching.
    //
    // The provider writes only the low byte of that stack slot
    // (`mov byte ptr [rsp+0x30], 1`), and the present itself reads it back
    // as a byte, so the upper seven bytes are leftovers - compare the byte.
    const uint8_t flag = static_cast<uint8_t>(arg7);

    if ((flag == 1) == isLast)
        return;

    auto* burst = reinterpret_cast<uint8_t*>(arg5) - 0x38;
    const uint64_t count = *reinterpret_cast<uint64_t*>(burst + 8);

    // The provider caps the multiplier at 6X, so count = 5 is the ceiling.
    if (count < 1 || count > 5)
        return;

    g_lastMultiplier = static_cast<int64_t>(count) + 1;

    uint64_t index = count;

    if (!isLast)
    {
        auto* frames = *reinterpret_cast<uint8_t**>(burst);

        if (frames == nullptr || reinterpret_cast<uint8_t*>(arg6) < frames)
            return;

        const uint64_t offset = reinterpret_cast<uint8_t*>(arg6) - frames;

        if ((offset % 8) != 0)
            return;

        index = offset / 8;

        // The loop presents frames 1..count-1.
        if (index < 1 || index >= count)
            return;
    }

    // Above 2X the provider only ever schedules the burst's last frame. The
    // ones the loop presents are the ones nobody stamps, and they are
    // exactly what this hook sees, so hand them over. The last frame is left
    // alone: the provider's own call at 0x220411 is already doing it, and
    // doing it twice would put two timestamps on one frame.
    // But the scheduler can be off. Its gate is `ctx[0x340] == 0 &&
    // ctx[0x341] != 0`, and 0x340 is the limiter's own byte - when the
    // limiter owns the pacing the scheduler is deliberately dormant and
    // every call into it is a silent no-op. Ask the same two bytes first; if
    // it would refuse, the wall clock paces instead of us paying for nothing.
    if (g_schedNative != nullptr && SchedulerUsable(ctx))
    {
        if (!isLast)
            ScheduleFrame(ctx, burst, index);

        return;
    }

    // Wall clock. The provider holds the last frame back under its own
    // conditions, so do not wait on that one twice.
    if (isLast && ProviderPacesLastFrame(ctx, burst, count))
        return;

    PaceFrame(index, count);
}

inline int64_t Detour(void* ctx, uint32_t a2, uint32_t a3, uint64_t a4, void* arg5, void* arg6, uint64_t arg7)
{
    if (g_enabled)
    {
        auto* caller = reinterpret_cast<uint8_t*>(_ReturnAddress());

        if (caller == g_base + PacedCallerRva)
            TryPace(ctx, arg5, arg6, arg7, false);
        else if (caller == g_base + LastFrameCallerRva)
            TryPace(ctx, arg5, arg6, arg7, true);
    }

    return g_native(ctx, a2, a3, a4, arg5, arg6, arg7);
}

// The provider's own call for the burst's last frame still enters through
// this thunk, so it has to keep working. Nothing to add: go on through.
inline bool SchedForwarder(void* ctx, void* burst, uint8_t gate, void* timing, uint32_t index)
{
    return g_schedNative != nullptr && g_schedNative(ctx, burst, gate, timing, index);
}

// The provider's deadline is `base + min(step, clamped) - recv`, and both of
// the faults in it are faults of that one expression. Rather than reproduce
// the expression, the out pointer is rewritten:
//
//   * the part the min() clamped away is added back, which leaves the first
//     frame of a burst - the only one whose `base` is real - anchored at
//     `base + index * median / (count + 1)`, the spacing it should have had;
//   * every later frame of the burst is then stepped from that anchor rather
//     than from the native result, so the stale `base` can no longer matter.
//
// The native is still called and its return value still passed back. Its only
// effects are two ring reads, so calling it is not required, but keeping it
// means the caller's contract does not differ between the two paths.
inline void* TsDetour(void* a1, int64_t* out, void* lookup, void* timing, uint32_t index, uint32_t countPlus1)
{
    void* const result = g_tsNative(a1, out, lookup, timing, index, countPlus1);

    // index 0 is the real frame and never reaches the provider's tail; the
    // provider caps the multiplier at 6X, so index 5 is the ceiling.
    if (out == nullptr || timing == nullptr || index == 0 || index > 5 || countPlus1 < 2)
        return result;

    const int64_t median = *reinterpret_cast<const int64_t*>(reinterpret_cast<const uint8_t*>(timing) + 8);
    const int64_t unit = median / static_cast<int64_t>(countPlus1);

    if (unit <= 0)
        return result;

    g_tsCalls++;

    // A burst begins at index 1. The other two clauses catch anything that
    // does not look like a fresh burst - a first call that is not index 1,
    // a multiplier change, a repeated index - and re-anchor instead of
    // stepping from a stale schedule.
    const bool fresh = index == 1 || index <= g_lastTsIndex || countPlus1 != g_lastTsCountPlus1;

    if (fresh)
    {
        // How much the min() removed, recomputed the way the provider does
        // it: f clamped, scaled to nanoseconds, truncated, divided.
        int64_t nativeUnit = unit;

        if (g_ring != nullptr)
        {
            float f = *reinterpret_cast<const float*>(g_ring + RingMeasuredOffset);

            if (!(f >= 0.125f) || f > 500.0f)
                f = 500.0f;

            const int64_t fNs = static_cast<int64_t>(f * 1000000.0f);
            const int64_t clampedUnit = fNs / static_cast<int64_t>(countPlus1);

            if (clampedUnit < nativeUnit)
            {
                nativeUnit = clampedUnit;
                g_tsClamped++;
            }
        }

        g_nextDeadlineNs = *out + static_cast<int64_t>(index) * (unit - nativeUnit);
        g_burstStepNs = unit;
        g_tsRebased++;
    }
    else
    {
        g_nextDeadlineNs += g_burstStepNs;
    }

    g_lastTsIndex = index;
    g_lastTsCountPlus1 = countPlus1;

    *out = g_nextDeadlineNs;

    return result;
}

inline bool WriteVerified(uint8_t* dst, const uint8_t* bytes, uint32_t size)
{
    DWORD oldProtect = 0;

    if (!VirtualProtect(dst, size, PAGE_EXECUTE_READWRITE, &oldProtect))
        return false;

    memcpy(dst, bytes, size);
    FlushInstructionCache(GetCurrentProcess(), dst, size);

    DWORD ignored = 0;
    VirtualProtect(dst, size, oldProtect, &ignored);

    return memcmp(dst, bytes, size) == 0;
}

// Redirects one 16 byte thunk slot to `detour` and reports the provider
// function it used to reach. Every entry of the table has the same shape -
// a 5 byte E9 rel32 and 11 bytes of int3 - so a 14 byte `jmp qword ptr
// [rip+0]` fits it exactly, with no instruction to relocate and no code
// cave. Verified before writing: the slot must still hold what we expect.
inline bool HookThunk(uint8_t* base, uint32_t rva, uint32_t targetRva, const uint8_t* expected, void* detour,
                      void** nativeOut)
{
    uint8_t* thunk = base + rva;

    if (memcmp(thunk, expected, PresentThunkSize) != 0)
    {
        LOG_WARN("XeFG pacing: thunk at {:#x} has unexpected bytes, not hooking", rva);
        return false;
    }

    uint8_t replacement[PresentThunkSize] = {
        0xFF, 0x25, 0x00, 0x00, 0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC,
    };

    // The rel32 is zero, so the jmp reads the 8 byte absolute address
    // immediately after it.
    auto target = reinterpret_cast<uint64_t>(detour);
    memcpy(replacement + 6, &target, sizeof(target));

    if (!WriteVerified(thunk, replacement, PresentThunkSize))
    {
        LOG_WARN("XeFG pacing: failed to write the thunk at {:#x}", rva);
        return false;
    }

    *nativeOut = base + targetRva;

    return true;
}

// Redirects the present thunk at 0x25C0 through Detour. Safe to call more
// than once, and a failure just leaves the provider presenting the way it
// always did.
inline bool Install(uint8_t* base)
{
    if (g_enabled)
        return true;

    if (base == nullptr)
        return false;

    if (!Config::Instance()->FGXeFGExtraPacing.value_or_default())
    {
        LOG_INFO("XeFG pacing: disabled by config (XeFG\\ExtraPacing)");
        return false;
    }

    QueryPerformanceFrequency(&g_freq);

    if (g_freq.QuadPart <= 0)
    {
        LOG_WARN("XeFG pacing: no performance counter, not hooking");
        return false;
    }

    g_base = base;
    g_lastBurstQpc = 0;
    g_periodNs = 0;
    g_intervalQpc = 0;
    g_targetQpc = 0;
    g_sampleCount = 0;
    g_samplePos = 0;
    g_pacedFrames = 0;
    g_pacedBursts = 0;
    g_loggedFirstBurst = false;
    g_lastPacedQpc = 0;
    g_schedLogged = 0;
    g_schedLastIndex = ~0ull;
    g_schedCalls = 0;
    g_schedRefused = 0;
    g_schedWaitQpc = 0;
    g_lastMultiplier = 0;
    g_burstBlockQpc = 0;
    g_renderTimeNs = 0;
    g_fedFrameTimeMs = 0.0;
    g_ring = nullptr;
    g_nextDeadlineNs = 0;
    g_burstStepNs = 0;
    g_lastTsIndex = 0;
    g_lastTsCountPlus1 = 0;
    g_tsCalls = 0;
    g_tsRebased = 0;
    g_tsClamped = 0;
    ResetStats(0);

    void* native = nullptr;

    if (!HookThunk(base, PresentThunkRva, NativePresentRva, PresentThunkExpected, &Detour, &native))
        return false;

    g_native = reinterpret_cast<PresentFn>(native);
    g_ringSnapshot = reinterpret_cast<RingSnapshotFn>(base + RingSnapshotFnRva);

    g_enabled = true;

    LOG_INFO("XeFG pacing: generated frames are paced above 2X (thunk {:#x} -> {:#x})", PresentThunkRva,
             reinterpret_cast<uint64_t>(&Detour));

    // The scheduler is reached through its thunk rather than called at
    // 0x21EE30 directly, so that the provider's own call for the burst's
    // last frame flows through the same path. If the thunk is not what it
    // should be, the pacing falls back to the wall clock - non-fatal.
    //
    // The native pointer is set from the RVA before the thunk is written,
    // so that the forwarder can never observe the thunk live and the
    // pointer still null.
    g_schedNative = reinterpret_cast<SchedFn>(base + SchedFnRva);

    if (HookThunk(base, SchedThunkRva, SchedFnRva, SchedThunkExpected, &SchedForwarder, &native))
        LOG_INFO("XeFG pacing: scheduling every generated frame through the provider's own scheduler "
                 "(thunk {:#x})",
                 SchedThunkRva);
    else
    {
        g_schedNative = nullptr;
        LOG_WARN("XeFG pacing: no scheduler, falling back to the wall clock");
    }

    // Last, because it is the one that changes the arithmetic rather than
    // the calls: without a scheduler there is nothing computing deadlines to
    // correct, so this is only worth arming if the scheduler is live.
    g_tsNative = reinterpret_cast<TimestampFn>(base + TimestampFnRva);

    if (g_schedNative != nullptr &&
        HookThunk(base, TimestampThunkRva, TimestampFnRva, TimestampThunkExpected, &TsDetour, &native))
        LOG_INFO("XeFG pacing: deadlines come from the burst's own frame interval (thunk {:#x})", TimestampThunkRva);
    else
        g_tsNative = nullptr;

    return true;
}
} // namespace XeFGPacing
