#pragma once
#include "FramePacer.h"
#include <algorithm>
#include <atomic>
#include <stop_token>
namespace nr {
// Pace acquisition, not just NR. Keep the last DXGI frame owned during the wait
// so Windows can accumulate updates instead of copying every game present.
// Standard: acquisitions keep a timer grid at the cap (the original behaviour).
// Low latency (Settings::latencyMode): a frame is acquired as soon as the backend has one. The cap limits the average
// rate only (to 110 % of it), with a little credit kept from longer gaps, so frames that arrive unevenly are not held
// back: Windows hands a 60 fps game's frames to the capture at its screen refreshes, which on a 164 Hz screen makes
// them 18.3, 18.3 and 12.2 ms apart. With at most 1.5 frames of credit, acquisitions start at least 45 % of the cap's
// period apart. (Measured with --pacing-simulation. A fixed least spacing of 90 % of the period held every third frame of that
// pattern about 3 ms; counting the spacing from the end of the copy made it longer than the period at 120 fps, so
// frames waited longer and longer and some were dropped.)
class CapturePacer {
    FramePacer timer;
    Handle changed{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    std::atomic<double> target;
    std::atomic<bool> eager{false};
    double applied = -1, lastStarted = 0;
    // Low latency: the credit (in frames) left after the last acquisition; one frame's worth is spent per acquisition.
    double credit = 0;
    bool appliedEager = false;
public:
    // Low latency: the share above the cap the average may reach, and the most credit kept (in frames).
    static constexpr double Room = 1.1, MostCredit = 1.5;
    explicit CapturePacer(double fps = 0) : target(fps) {
        if (!changed.value) Check(HRESULT_FROM_WIN32(GetLastError()), "Capture pacing event");
    }
    void SetRate(double fps) {
        if (target.exchange(fps) != fps) SetEvent(changed.value);
    }
    void SetLowLatency(bool on) {
        if (eager.exchange(on) != on) SetEvent(changed.value);
    }
    // The least spacing of low-latency acquisitions at this cap (0: none): a frame's worth of credit less the most kept.
    static double MinimumSpacing(double fps) { return fps > 0 ? (2.0 - MostCredit) * 1000.0 / (Room * fps) : 0; }
    // Low latency: the earliest start of the next acquisition (credit is at most MostCredit - 1 after one).
    double Due(double fps) const { return fps > 0 ? lastStarted + (1.0 - credit) * 1000.0 / (Room * fps) : lastStarted; }
    bool Wait(std::stop_token stop) {
        std::stop_callback wake(stop, [this] { SetEvent(changed.value); });
        while (!stop.stop_requested()) {
            const double fps = target.load(); const bool low = eager.load();
            if (fps != applied || low != appliedEager) {
                applied = fps; appliedEager = low; timer.Submitted(fps, lastStarted);
                credit = 0; // (a new cap or mode starts with no credit: the next frame is due one period / 110 % on)
            }
            if (appliedEager) {
                const double due = Due(fps);
                if (ClockMs() >= due) return true;
                timer.WaitUntil(due, changed.value);
                continue;
            }
            if (timer.Due(fps, ClockMs())) return true;
            timer.Wait(changed.value);
        }
        return false;
    }
    void Submitted(double started) {
        if (appliedEager && applied > 0 && lastStarted > 0)
            credit = std::max(0.0, std::min(MostCredit, credit + (started - lastStarted) * Room * applied / 1000.0) - 1.0);
        else credit = 0;
        lastStarted = started;
        timer.Submitted(applied, started);
    }
};
}
