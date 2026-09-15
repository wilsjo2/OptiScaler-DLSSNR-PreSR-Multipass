// Headless sequence regression; no game or NVIDIA runtime is loaded.
#include "../OptiScaler/shaders/dlssnr/DlssNr_SeamClock.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

// Same continuity/alternation conditions used by DeferredSr::PrepareHalfRate.
// This isolates scheduling, not NR quality, GPU execution or display pacing.
struct Cadence
{
    unsigned long long last = 0;
    bool havePrevious = false, previousWasAnchor = false;
    unsigned anchors = 0, skipped = 0;
    void Evaluate(unsigned long long epoch, bool reset = false)
    {
        if (reset || epoch != last + 1) havePrevious = previousWasAnchor = false;
        last = epoch;
        const bool skip = havePrevious && previousWasAnchor;
        if (skip) ++skipped; else ++anchors;
        havePrevious = true;
        previousWasAnchor = !skip;
    }
};

int main()
{
#ifdef _WIN32
    _set_error_mode(_OUT_TO_STDERR);
#endif
    DlssNrSeamClock clock;
    assert(clock.AtSeam(true, false, 10) == 10);
    // Present ticks between the paired calls: keep identity, then advance once next frame.
    assert(clock.AtSeam(false, false, 11) == 10);
    assert(clock.AtSeam(true, false, 11) == 11);
    assert(clock.AtSeam(false, false, 11) == 11);
    // A stalled Present counter must not look like a duplicate native evaluate.
    assert(clock.AtSeam(true, false, 11) == 12);
    assert(clock.AtSeam(false, false, 12) == 12);
    assert(clock.AtSeam(true, false, 12) == 13);
    // Present jumps/resets are not rendered-frame gaps. FG can present several
    // times per upscale, or Present can run on a different thread.
    assert(clock.AtSeam(true, false, 20) == 14);
    assert(clock.AtSeam(false, false, 21) == 14);
    assert(clock.AtSeam(true, false, 0) == 15);
    // Bridges retain their own epoch, including duplicate evaluations.
    assert(clock.AtSeam(true, true, 7) == 7);
    assert(clock.AtSeam(true, true, 7) == 7);
    assert(clock.AtSeam(false, true, 7) == 7);
    assert(clock.AtSeam(true, true, 8) == 8);
    assert(clock.AtSeam(true, true, 20) == 20); // a real bridge-submission gap survives
    assert(clock.AtSeam(true, false, 3) == 16); // bridge never advances native identity

    // Requiem log 2026-09-14 01:02:49-01:02:57: 313 native upscales,
    // raw Present epochs 27537,27539,... Previously 313 anchors / 0 skips.
    for (unsigned presentsPerRender : {1u, 2u, 3u, 4u, 6u})
    {
        DlssNrSeamClock replay;
        Cadence cadence;
        for (unsigned i = 0; i < 313; ++i)
        {
            const auto raw = 27537ull + i * presentsPerRender;
            const auto render = replay.AtSeam(true, false, raw);
            assert(replay.AtSeam(false, false, raw + presentsPerRender) == render);
            cadence.Evaluate(render);
        }
        assert(cadence.anchors == 157 && cadence.skipped == 156);
        std::printf("PASS: %u Present(s)/render -> 157 NR anchors / 156 skips\n", presentsPerRender);
    }
    DlssNrSeamClock varying;
    Cadence cadence;
    const unsigned long long raw[] = {100,102,105,105,109,0,6,6,7,1000};
    for (unsigned i = 0; i < 10; ++i)
    {
        auto render = varying.AtSeam(true, false, raw[i]);
        assert(render == 100 + i);
        // Explicit game reset still forces an anchor even on a would-be skip.
        cadence.Evaluate(render, i == 5);
    }
    assert(cadence.anchors == 6 && cadence.skipped == 4);
    Cadence bridge;
    DlssNrSeamClock bridgeClock;
    bridge.Evaluate(bridgeClock.AtSeam(true, true, 10));
    bridge.Evaluate(bridgeClock.AtSeam(true, true, 12));
    assert(bridge.anchors == 2 && bridge.skipped == 0);
    std::puts("PASS: native pairing, variable Present rate, explicit reset and bridge gap");
}
