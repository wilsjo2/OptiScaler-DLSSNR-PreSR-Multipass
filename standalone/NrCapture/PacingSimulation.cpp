// --pacing-simulation <seconds>: the delay the app's own pacing adds, measured with its real pacers (CapturePacer on the
// capture thread, FramePacer on the processing loop) and real timers, but a stand-in game and no GPU. A thread presents
// "game frames" at a set rate with a little jitter; a capture thread takes the newest one the way the capture thread does
// (Settings::latencyMode: standard or low latency); the processing loop takes the newest captured frame the way
// App::Run does and "processes" it for a fixed time (the app waits for its GPU work before the next frame).
// Reported per case: game present -> processing start (ms), game frames never processed, spacing of processing starts
// (its sd is how evenly the app presents: what NVIDIA Smooth Motion has to work with).
// Cases "on a <n> Hz screen" also stand in for Windows composing: a game frame reaches the capture only at the next
// refresh, so arrivals come unevenly (60 fps on 164 Hz: 18.3, 18.3, 12.2 ms apart); the wait for that refresh is in
// their delay. Not included: the GPU work, and Windows composing the app's output.
#include "CapturePacer.h"
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>
namespace nr {
namespace {
void SpinUntil(double deadline, HANDLE timer) {
    const double wait = deadline - ClockMs() - 1.0;
    if (wait > 0) {
        LARGE_INTEGER due{}; due.QuadPart = -LONGLONG(wait * 10000);
        SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE); WaitForSingleObject(timer, 1000);
    }
    while (ClockMs() < deadline) YieldProcessor();
}
double Percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1, size_t(std::lround(q * double(values.size() - 1))))];
}
struct Case { const char* name; double gameFps, jitterMs, cap, screenHz = 0; };
struct Outcome { std::vector<double> delays, captureDelays, spacing; uint64_t gameFrames = 0, processedFrames = 0; };
Outcome Run(const Case& c, bool lowLatency, double seconds, double processingMs) {
    std::mutex sourceMutex, queueMutex;
    uint64_t gameSequence = 0, publishedSequence = 0; double gameTime = 0, publishedTime = 0, publishedAt = 0;
    Handle gameEvent{CreateEventW(nullptr, FALSE, FALSE, nullptr)}, arrived{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    Handle stopEvent{CreateEventW(nullptr, TRUE, FALSE, nullptr)};
    CapturePacer capturePacer(c.cap); capturePacer.SetLowLatency(lowLatency);
    std::stop_source stop;
    std::jthread game([&] {
        Handle timer{CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)};
        std::mt19937 random(11); std::normal_distribution<double> jitter(0, c.jitterMs);
        double next = ClockMs(); const double refreshes = next, refresh = c.screenHz > 0 ? 1000.0 / c.screenHz : 0;
        while (!stop.get_token().stop_requested()) {
            next += 1000.0 / c.gameFps;
            SpinUntil(next + std::clamp(jitter(random), -3 * c.jitterMs, 3 * c.jitterMs), timer.value);
            const double presented = ClockMs();
            // (the capture sees the frame once Windows has composed it, at the next refresh; 0.3 ms for the composition)
            if (refresh > 0) SpinUntil(refreshes + std::ceil((presented - refreshes) / refresh) * refresh + 0.3, timer.value);
            { std::lock_guard lock(sourceMutex); ++gameSequence; gameTime = presented; }
            SetEvent(gameEvent.value);
        }
    });
    std::jthread capture([&] {
        Handle timer{CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)};
        uint64_t taken = 0; const auto token = stop.get_token();
        while (!token.stop_requested()) {
            if (!capturePacer.Wait(token)) break;
            const double started = ClockMs();
            uint64_t sequence; double time;
            { std::lock_guard lock(sourceMutex); sequence = gameSequence; time = gameTime; }
            // As WgcCapture::Next: nothing new waits up to 10 ms for an arrival and reports no frame.
            if (sequence == taken) { WaitForSingleObject(gameEvent.value, 10); continue; }
            taken = sequence;
            SpinUntil(ClockMs() + 0.7, timer.value); // the copy into the capture queue
            { std::lock_guard lock(queueMutex); publishedSequence = sequence; publishedTime = time; publishedAt = ClockMs(); }
            capturePacer.Submitted(started);
            SetEvent(arrived.value);
        }
    });
    Outcome outcome;
    Handle timer{CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)};
    FramePacer pacer; uint64_t consumed = 0, frameSequence = 0, firstGame = 0; double frameTime = 0, frameCaptured = 0, lastStart = 0;
    bool pending = false, processed = false; std::set<uint64_t> shown;
    const double begin = ClockMs() + 500, end = begin + seconds * 1000; // (half a second to settle)
    while (ClockMs() < end) {
        bool acquired = false;
        { std::lock_guard lock(queueMutex); if (publishedSequence > consumed) { consumed = frameSequence = publishedSequence; frameTime = publishedTime; frameCaptured = publishedAt; acquired = true; } }
        pending |= acquired;
        if (!pending) { HANDLE waits[]{stopEvent.value, arrived.value}; WaitForMultipleObjects(2, waits, FALSE, 20); continue; }
        // As App::Run: standard pacing waits for its own slot; low latency processes a new frame at once.
        if (!lowLatency && processed && !pacer.Due(c.cap, ClockMs())) { pacer.Wait(stopEvent.value, arrived.value); continue; }
        const double start = ClockMs();
        if (lowLatency) { // the newest frame again, right before recording
            std::lock_guard lock(queueMutex);
            if (publishedSequence > consumed) { consumed = frameSequence = publishedSequence; frameTime = publishedTime; frameCaptured = publishedAt; }
        }
        if (start >= begin) {
            if (!firstGame) firstGame = frameSequence;
            outcome.delays.push_back(start - frameTime); outcome.captureDelays.push_back(frameCaptured - frameTime); shown.insert(frameSequence);
            if (lastStart >= begin) outcome.spacing.push_back(start - lastStart);
        }
        lastStart = start;
        SpinUntil(start + processingMs, timer.value); // recording and the GPU work (the app waits for it)
        pacer.Submitted(c.cap, start); pending = false; processed = true;
    }
    stop.request_stop(); SetEvent(stopEvent.value);
    capture.join(); game.join();
    outcome.processedFrames = shown.size();
    outcome.gameFrames = !shown.empty() ? *shown.rbegin() - firstGame + 1 : 0;
    return outcome;
}
}
void PacingSimulation(double seconds) {
    const double processingMs = 4;
    const Case cases[]{{"game 60 fps, Target FPS 60", 60, 0.8, 60}, {"game 60 fps, Target FPS 0 (uncapped)", 60, 0.8, 0},
                       {"game 55 fps, Target FPS 60", 55, 0.8, 60}, {"game 120 fps, Target FPS 120", 120, 0.5, 120},
                       {"game 60 fps on a 164 Hz screen, Target FPS 60", 60, 0.8, 60, 164},
                       {"game 60 fps on a 164 Hz screen, Target FPS 0", 60, 0.8, 0, 164},
                       {"game 60 fps on a 120 Hz screen, Target FPS 60", 60, 0.8, 60, 120}};
    std::cout << std::format("Pacing simulation: 5 runs of {:.0f} s per case, processing {:.0f} ms per frame. Delay = game present -> processing start.\n", seconds, processingMs);
    // The standard pacers' timers settle at a different phase to the game every run, so each case runs five times and
    // the figures pool all five (the range of the five runs' means is shown too).
    constexpr int kRuns = 5;
    for (const auto& c : cases) {
        for (const bool low : {false, true}) {
            Outcome o; double lowestMean = 1e9, highestMean = 0;
            for (int run = 0; run < kRuns; ++run) {
                const auto one = Run(c, low, seconds, processingMs);
                double m = 0; for (double d : one.delays) m += d; m /= std::max<size_t>(1, one.delays.size());
                lowestMean = std::min(lowestMean, m); highestMean = std::max(highestMean, m);
                o.delays.insert(o.delays.end(), one.delays.begin(), one.delays.end());
                o.captureDelays.insert(o.captureDelays.end(), one.captureDelays.begin(), one.captureDelays.end());
                o.spacing.insert(o.spacing.end(), one.spacing.begin(), one.spacing.end());
                o.gameFrames += one.gameFrames; o.processedFrames += one.processedFrames;
            }
            double mean = 0; for (double d : o.delays) mean += d; mean /= std::max<size_t>(1, o.delays.size());
            double captureMean = 0; for (double d : o.captureDelays) captureMean += d; captureMean /= std::max<size_t>(1, o.captureDelays.size());
            double spacingMean = 0, spacingSd = 0;
            for (double d : o.spacing) spacingMean += d; spacingMean /= std::max<size_t>(1, o.spacing.size());
            for (double d : o.spacing) spacingSd += (d - spacingMean) * (d - spacingMean); spacingSd = std::sqrt(spacingSd / std::max<size_t>(1, o.spacing.size()));
            std::cout << std::format("  {:<38} {:<12} delay mean {:5.2f} ms (= {:.2f} game frames; runs {:.2f}..{:.2f}; capture part {:5.2f}), p50 {:5.2f}, p99 {:5.2f}, max {:5.2f} | "
                                     "{} of {} game frames processed | spacing {:5.2f} ms, sd {:4.2f}\n",
                c.name, low ? "low latency" : "standard", mean, mean * c.gameFps / 1000.0, lowestMean, highestMean, captureMean,
                Percentile(o.delays, .5), Percentile(o.delays, .99), Percentile(o.delays, 1), o.processedFrames, o.gameFrames, spacingMean, spacingSd);
        }
    }
}
}
