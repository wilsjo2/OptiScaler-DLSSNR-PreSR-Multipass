#pragma once
#include "Common.h"
#include <cmath>
namespace nr {
class FramePacer {
    Handle timer{CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)};
    double next = 0, rate = 0;
public:
    FramePacer() { if (!timer.value) Check(HRESULT_FROM_WIN32(GetLastError()), "Frame pacing timer"); }
    bool Due(double fps, double now) const { return !fps || now >= next; }
    // Deadlines stay on a fixed grid, so a late timer wake does not delay every later frame: the cap is
    // reached on average. A frame later than the slack moves the grid instead of being caught up in a burst.
    void Submitted(double fps, double started) {
        if (!fps) { next = rate = 0; return; }
        const double period = 1000.0 / fps, slack = std::min(2.0, period / 4);
        // A new rate, the first frame or a stall of a whole period restarts the grid from this frame.
        next = fps != rate || !next || started - next > period ? started + period :
            std::max(next + period, started + period - slack);
        rate = fps;
    }
    void Wait(HANDLE stop, HANDLE capture = nullptr) { WaitUntil(next, stop, capture); }
    // Until `deadline` (ClockMs), an event, or at most 20 ms.
    void WaitUntil(double deadline, HANDLE stop, HANDLE capture = nullptr) {
        LARGE_INTEGER due{};
        due.QuadPart = -std::max<LONGLONG>(1, (LONGLONG)std::ceil((deadline - ClockMs()) * 10000));
        Check(SetWaitableTimer(timer.value, &due, 0, nullptr, nullptr, FALSE) ? S_OK : HRESULT_FROM_WIN32(GetLastError()),
            "Set frame pacing timer");
        HANDLE events[]{stop, timer.value, capture};
        if (WaitForMultipleObjects(capture ? 3 : 2, events, FALSE, 20) == WAIT_FAILED) // Keep focus/visibility changes responsive at low caps.
            Check(HRESULT_FROM_WIN32(GetLastError()), "Wait for next capture/frame");
    }
};
}
