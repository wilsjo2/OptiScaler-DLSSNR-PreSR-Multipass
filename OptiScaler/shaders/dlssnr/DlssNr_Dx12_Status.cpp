#include "pch.h"
#include "DlssNr_Dx12_State.h"

auto DlssNr_Dx12::State::ReportSkipOnce(const char* reason) -> void
{

    if (seen.insert(reason).second)
        LOG_INFO("DLSS-NR did not run: {}", reason);
}

auto DlssNr_Dx12::State::SynchronousDeferredDlssStatus() -> std::string
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return deferredSr.status;
}

auto DlssNr_Dx12::State::DeferredDlssStatus() -> std::string
{ return SynchronousDeferredDlssStatus(); }

auto DlssNr_Dx12::State::RetryAfterFailure() -> void
{
    ReleaseEnlarger();
    enlargementStatus.clear();
    nr.failed = false;
    nr.reason = "";
    nr.reset = true;
}

auto DlssNr_Dx12::State::ConsumeControls() -> void
{
    const auto& cfg = *Config::Instance();
    if (!cfg.DlssNrEnabled.value_or_default() || cfg.DlssNrTransfer.value_or_default() != 2 ||
        cfg.DlssNrWorkingScale.value_or_default() >= 1.0f)
    {
        ReleaseEnlarger();
        enlargementStatus.clear();
    }
    const auto requested = DlssNr::ReadControlRequests();
    if (requested.retryGeneration != controls.retryGeneration)
    {
        for (auto& model : nr.models)
            model.RetryAfterFailure();
        std::fill(std::begin(nr.passCreateFailed), std::end(nr.passCreateFailed), false);
        modelRunning = false;
        RetryAfterFailure();
    }
    if (requested.captureGeneration != controls.captureGeneration)
        captureFrames.request(requested.captureFrames);
    controls = requested;
}

auto DlssNr_Dx12::State::Publish() -> void
{
    DlssNr::PublishStatus(
        &shader, DlssNr::Backend::Dx12,
        { !nr.failed && modelRunning && enlargementStatus.empty(),
          nr.failed ? nr.reason : enlargementStatus,
          lastGpuTime,
          frames,
          { nr.exposureFrames, nr.exposureOfferedNow, nr.exposureEverOffered, nr.gameExposure, nr.gamePreExposure },
          captureFrames.isActive() });
}

void DlssNr_Dx12::State::EndGpuTiming(ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* timingQueue)
{
    if (gpuTime != nullptr)
    {
        gpuTime->End(cmdList);

        // Use the explicit bridge queue when supplied; otherwise use the upscaler's queue.
        auto* queue =
            timingQueue != nullptr ? timingQueue : (ID3D12CommandQueue*) ::State::Instance().currentCommandQueue;

        if (queue != nullptr)
        {
            if (auto ms = gpuTime->ReadGpuTime(queue); ms.has_value())
                lastGpuTime = ms;

            if (ngxTime != nullptr)
            {
                if (auto ngx = ngxTime->ReadGpuTime(queue); ngx.has_value())
                    lastNgxTime = ngx;
            }

            // The split, once every few hundred frames. What is worth reading is not the total but the
            // remainder: the model's cost is NVIDIA's to set, and everything else is ours.

            if (lastGpuTime.has_value() && lastNgxTime.has_value())
                vitals.Push(lastGpuTime.value(), lastNgxTime.value());

            if (lastGpuTime.has_value() && lastNgxTime.has_value() && frames - lastSplitLog > 600)
            {
                // Captured before lastSplitLog is overwritten below - this is the actual window size
                // the cadence counters below need, not a value that's already been zeroed against itself.
                const unsigned long long windowFrames = frames - lastSplitLog;
                lastSplitLog = frames;
                const double total = lastGpuTime.value();
                const double ngx = lastNgxTime.value();
                LOG_INFO("DLSS-NR elapsed: {:.2f} ms total, {:.2f} ms model, {:.2f} ms surrounding work ({:.0f}%; "
                         "intervals may include other GPU work)",
                         total, ngx, total - ngx, total > 0.0 ? 100.0 * (total - ngx) / total : 0.0);

                // Vitals: the window this split-log sample came from, not just that one sample - a
                // single frame landing on the 600th tick can miss every real spike in between.
                // Human-readable line plus a pipe-delimited one, greppable by any external tool or
                // by hand, without needing MangoHud or a separate post-processing script for NR's
                // own GPU cost specifically (see optiscaler-deploy/mangohud-fps.py for the whole-
                // frame equivalent this project already relies on).
                const auto summary = vitals.Compute();
                if (summary.sampleCount > 0)
                {
                    LOG_INFO("DLSS-NR vitals: last {} frames, total mean {:.2f} ms p99 {:.2f} ms, "
                             "model mean {:.2f} ms p99 {:.2f} ms, resets {}, effective passes {}",
                             summary.sampleCount, summary.totalMean, summary.totalP99, summary.innerMean,
                             summary.innerP99, resets, loggedEffective);
                    LOG_INFO("DLSS-NR-VITALS|frames={}|samples={}|total_mean_ms={:.2f}|total_p99_ms={:.2f}"
                             "|model_mean_ms={:.2f}|model_p99_ms={:.2f}|resets={}|effective_passes={}",
                             frames, summary.sampleCount, summary.totalMean, summary.totalP99, summary.innerMean,
                             summary.innerP99, resets, loggedEffective);

                    // ADR-014/017: cadence-decoupling counters, reported as this window's deltas
                    // (never a running total with no denominator) so the line is self-checkable:
                    // window_frames == evaluates + skips + skip_blocked_by_reset +
                    // skip_blocked_by_invalid_history must hold, or the counting itself is broken.
                    // Only printed once cadence has actually been configured >1 at least once.
                    if (cadenceEvaluates || cadenceSkips || cadenceSkipBlockedByReset ||
                        cadenceSkipBlockedByInvalidHistory || cadenceCarryForwardFailed)
                    {
                        const unsigned long long windowResets = resets - lastResetsAtLog;
                        const unsigned long long windowEvaluates = cadenceEvaluates - lastCadenceEvaluatesAtLog;
                        const unsigned long long windowSkips = cadenceSkips - lastCadenceSkipsAtLog;
                        const unsigned long long windowBlockedByReset =
                            cadenceSkipBlockedByReset - lastCadenceSkipBlockedByResetAtLog;
                        const unsigned long long windowBlockedByInvalidHistory =
                            cadenceSkipBlockedByInvalidHistory - lastCadenceSkipBlockedByInvalidHistoryAtLog;
                        const unsigned long long windowCarryForwardFailed =
                            cadenceCarryForwardFailed - lastCadenceCarryForwardFailedAtLog;

                        LOG_INFO("DLSS-NR-CADENCE|window_frames={}|window_resets={}|evaluates={}|skips={}"
                                 "|skip_blocked_by_reset={}|skip_blocked_by_invalid_history={}"
                                 "|carry_forward_dispatch_failed={}",
                                 windowFrames, windowResets, windowEvaluates, windowSkips, windowBlockedByReset,
                                 windowBlockedByInvalidHistory, windowCarryForwardFailed);

                        lastResetsAtLog = resets;
                        lastCadenceEvaluatesAtLog = cadenceEvaluates;
                        lastCadenceSkipsAtLog = cadenceSkips;
                        lastCadenceSkipBlockedByResetAtLog = cadenceSkipBlockedByReset;
                        lastCadenceSkipBlockedByInvalidHistoryAtLog = cadenceSkipBlockedByInvalidHistory;
                        lastCadenceCarryForwardFailedAtLog = cadenceCarryForwardFailed;
                    }
                }
            }
        }
    }

}
