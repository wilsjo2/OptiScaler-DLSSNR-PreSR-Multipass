#include "App.h"
#include "ModelCompare.h"
namespace nr {
struct Session {
    Gpu gpu;
    std::unique_ptr<FrameGeneration> fg;
    std::unique_ptr<Processor> processor;
    // A model comparison (Settings::ComparingModels): the same passes with the other model, and the two put together.
    std::unique_ptr<Processor> comparison;
    std::unique_ptr<ModelCompare> composer;
    std::unique_ptr<Capture> capture;
    std::unique_ptr<Presentation> display;
};
void App::Run(CaptureSource source, Settings initial) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    Calibration calibration; // (declared before the session: its readback buffer is released after the GPU has drained)
    std::unique_ptr<Session> session;
    try {
        Log(source.monitor ? "Starting monitor capture." : "Starting window capture.");
        session = std::make_unique<Session>();
        auto& g = session->gpu;
        session->processor = std::make_unique<Processor>(g, initial.runtime, initial.frameGeneration > 2);
        if (initial.frameGeneration > 1) session->fg = std::make_unique<FrameGeneration>(g, std::filesystem::path{}, initial.runtime);
        session->display = std::make_unique<Presentation>(g, source, window, session->fg.get(), initial.frameGeneration, initial.presentMode);
        session->capture = std::make_unique<Capture>(g, source, initial.captureBackend, initial.ProcessingFps());
        session->capture->SetLowLatency(initial.latencyMode == 1);
        Log(std::format("Presentation: {}; pacing: {}.", session->fg ? "DLSS FG swapchain" : session->display->FlipWindow() ?
            "flip window (Smooth Motion can take it)" : "overlay surface", initial.latencyMode == 1 ? "low latency" : "standard"));
        // Diagnostics: NR_FRAME_LOG=<file.csv> writes one line per processed frame (times in ClockMs, i.e. QPC milliseconds).
        std::FILE* frameLog = nullptr;
        if (wchar_t path[MAX_PATH]{}; GetEnvironmentVariableW(L"NR_FRAME_LOG", path, MAX_PATH) && *path) {
            if (_wfopen_s(&frameLog, path, L"w") != 0) frameLog = nullptr;
            if (frameLog) std::fputs("frame,capture,acquired,started,reacquired,submitted,presentCalled,presentReturned,presentCount,lowLatency,flipWindow,fg\n", frameLog);
        }
        struct LogCloser { std::FILE*& file; ~LogCloser() { if (file) std::fclose(file); } } frameLogCloser{frameLog};
        uint64_t frameIndex = 0; double acquiredAt = 0;
        Say(source.monitor ? L"Screen capture ready. Ctrl+Alt+End stops." :
            L"Capture ready. The filter shows while the selected window or these controls have focus. Ctrl+Alt+End stops.");
        double lastFrame = ClockMs(), reportAt = ClockMs(), latencySum = 0;
        unsigned frames = 0, resets = 0, fpsFrames = 0;
        unsigned timingFrame = 0;
        wchar_t timingOption[2]{};
        const bool diagnostics = GetEnvironmentVariableW(L"NR_CAPTURE_TIMINGS", timingOption, 2) == 1 && timingOption[0] == L'1';
        double lastLog = 0;
        struct Timings {
            double gpuWait = 0, capture = 0, idle = 0, pacing = 0, record = 0, present = 0;
            double reflex = 0, nr = 0, fgPrepare = 0;
            std::array<double, unsigned(GpuStage::Count)> stages{};
            unsigned stageFrames = 0;
        } timings;
        uint64_t reportedCaptures = 0;
        uint64_t reportedPublished = 0; double reportedCopyMs = 0;
        uint64_t reportedPresented = 0;
        double fpsSince = ClockMs(); int displayedFps = 0;
        int displayedOutputFps = session->fg ? 0 : -1;
        uint64_t fpsPresented = 0;
        Settings applied; bool processed = false, historyReset = true;
        FramePacer pacer; bool pendingCapture = false;
        double previousCaptureTime = 0; HWND previousForeground = nullptr;
        std::wstring pausedReason;
        auto pause = [&](const wchar_t* reason) {
            session->display->Hide(); historyReset = true;
            fpsSince = ClockMs(); fpsFrames = 0; displayedFps = 0;
            displayedOutputFps = session->fg ? 0 : -1;
            fpsPresented = session->fg ? session->fg->PresentedFrames() : 0;
            if (pausedReason != reason) { pausedReason = reason; Say(pausedReason); Log(Narrow(pausedReason)); }
        };
        capturing = true;
        std::wstring calibrationStatus;
        while (WaitForSingleObject(stop.value, 0) != WAIT_OBJECT_0 && source.Exists() && !session->capture->Closed()) {
            MSG message{}; while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&message); DispatchMessageW(&message);
            }
            session->display->RefreshVisibility();
            if (!g.Complete()) {
                const auto start = ClockMs(); g.WaitForCompletion(stop.value);
                timings.gpuWait += ClockMs() - start; continue;
            }
            std::array<double, unsigned(GpuStage::Count)> stages{};
            if (g.TakeTimings(stages)) {
                for (unsigned i = 0; i < stages.size(); ++i) timings.stages[i] += stages[i];
                ++timings.stageFrames;
            }
            const auto captureStart = ClockMs();
            const bool acquired = session->capture->Next();
            timings.capture += ClockMs() - captureStart;
            if (acquired) acquiredAt = ClockMs();
            pendingCapture |= acquired;
            const bool fresh = pendingCapture;
            calibration.Completed(); // the frame it copied has completed: read it and choose the next values
            Settings current; std::optional<Calibration::Request> wanted; bool cancel = false;
            { std::lock_guard lock(mutex); current = settings; wanted = std::move(calibrationRequest); calibrationRequest.reset(); cancel = calibrationCancel; calibrationCancel = false; }
            if (cancel) calibration.Cancel();
            if (wanted) {
                try { calibration.Start(*wanted, current); }
                catch (const std::exception& e) { std::lock_guard lock(mutex); calibrationResult = Calibration::Result{false, wanted->pass, {}, Wide(e.what()), wanted->skin}; PostMessageW(window, CalibrationMessage, 0, 0); }
            }
            if (auto finished = calibration.TakeResult()) {
                Say(finished->text); calibrationStatus.clear(); historyReset = true;
                { std::lock_guard lock(mutex); calibrationResult = std::move(finished); }
                PostMessageW(window, CalibrationMessage, 0, 0);
            }
            const bool calibrating = calibration.Active();
            if (calibrating) {
                current = calibration.Frame();
                if (auto text = calibration.Status(); text != calibrationStatus) { calibrationStatus = text; Say(text); }
            }
            session->capture->SetFrameRate(current.ProcessingFps());
            const bool lowLatency = current.latencyMode == 1;
            session->capture->SetLowLatency(lowLatency);
            const bool repaint = calibrating || !processed || current != applied || !IsWindowVisible(session->display->Window());
            RECT bounds{};
            const bool usable = source.Bounds(bounds) && session->capture->Input() &&
                session->capture->width == (UINT)(bounds.right - bounds.left) &&
                session->capture->height == (UINT)(bounds.bottom - bounds.top);
            if (!usable) pause(L"Waiting for a capture matching the source size. The original picture is visible.");
            if ((!fresh && !repaint) || !usable) {
                if (!usable) pendingCapture = false;
                const auto start = ClockMs();
                HANDLE waits[]{stop.value, session->capture->Event()}; WaitForMultipleObjects(2, waits, FALSE, 20);
                timings.idle += ClockMs() - start; continue;
            }
            if (!session->display->Active() || !current.enabled) {
                pause(!current.enabled ? L"NR is disabled. The original picture is visible." :
                    GetForegroundWindow() == window ? L"Paused: the selected source window is minimised or hidden." :
                    L"Paused: focus the selected source window or these controls to display the filter.");
                // Capture publishes only completed copies; Next queues no GPU work.
                pendingCapture = false;
                if (!acquired) WaitForSingleObject(stop.value, 20);
                continue;
            }
            // Low latency: the capture producer already keeps the cap, so a fresh frame is processed at once.
            if (!lowLatency && !calibrating && processed && current == applied && !pacer.Due(current.ProcessingFps(), ClockMs())) {
                // Pin the newest completed capture while capped. There is no GPU
                // work to submit until we actually process it.
                const auto start = ClockMs(); pacer.Wait(stop.value, session->capture->Event());
                timings.pacing += ClockMs() - start; continue;
            }
            const double processingStarted = ClockMs();
            auto& capture = *session->capture;
            auto& processor = *session->processor;
            if (!pausedReason.empty()) {
                pausedReason.clear(); reportAt = ClockMs(); frames = resets = 0; latencySum = 0;
                timings = {}; reportedCaptures = capture.receivedFrames;
                reportedPublished = capture.publishedFrames; reportedCopyMs = capture.copyMilliseconds;
                reportedPresented = session->fg ? session->fg->PresentedFrames() : 0;
                Say(L"Capture resumed."); Log("Capture resumed.");
            }
            if (current.estimateMotion && (!processed || !applied.estimateMotion)) {
                Say(L"Preparing motion estimation..."); Log("Preparing motion estimation.");
            }
            // A model comparison runs the passes twice: the picture's own (shown whole) and the other model's.
            const bool differing = !calibrating && current.DifferenceOfModels();
            const bool comparing = !calibrating && (current.ComparingModels() || differing);
            const Settings pictureSide = comparing ? current.PictureSide() : current;
            const Settings otherSide = comparing ? current.ComparisonSide() : Settings{};
            if (comparing && !session->comparison) {
                session->comparison = std::make_unique<Processor>(g, processor);
                session->composer = std::make_unique<ModelCompare>(g);
                Log(std::format("Model comparison: the other side runs model {}.", current.compareModel));
            } else if (!comparing && session->comparison) {
                // (Every earlier list has completed: the loop waits for the GPU before a frame is recorded.)
                session->composer.reset(); session->comparison.reset();
                Log("Model comparison: off.");
            }
            processor.Resize(capture.width, capture.height, pictureSide);
            if (comparing) session->comparison->Resize(capture.width, capture.height, otherSide);
            session->display->Resize(capture.width, capture.height);
            const bool hdr = session->display->Hdr();
            const float desktopWhite = session->display->WhitePoint(hdr);
            const HWND foreground = GetForegroundWindow();
            // While calibrating the frame is held: history restarts only when the slider values change.
            const bool reset = calibrating ? calibration.ResetHistory() : historyReset || !fresh || capture.timestamp - previousCaptureTime > 250 ||
                (source.monitor && foreground != previousForeground);
            float frameTime = previousCaptureTime > 0 ? (float)std::clamp(capture.timestamp - previousCaptureTime, 1.0, 1000.0) : 16.67f;
            if (processingStarted - fpsSince >= 500) {
                const auto presented = session->fg ? session->fg->PresentedFrames() : 0;
                const auto rate = 1000.0 / (processingStarted - fpsSince);
                displayedFps = static_cast<int>(fpsFrames * rate + 0.5);
                displayedOutputFps = session->fg ? static_cast<int>((presented - fpsPresented) * rate + 0.5) : -1;
                fpsPresented = presented;
                fpsSince = processingStarted; fpsFrames = 0;
            }
            const auto reflexStart = ClockMs();
            session->display->BeginFrame(current.maxFps); timings.reflex += ClockMs() - reflexStart;
            // Low latency: a newer frame of the same size may have arrived while the frame was prepared (FG's Reflex
            // sleep, resizing): take it now, right before recording. (Not while calibrating: its frame is held.)
            bool reacquired = false;
            // (Only while the GPU is idle: resizing may have submitted work that reads the current input.)
            if (lowLatency && !calibrating && !current.holdFrame && g.Complete() && capture.NextOfSize(capture.width, capture.height)) {
                reacquired = true; acquiredAt = ClockMs();
                if (previousCaptureTime > 0) frameTime = (float)std::clamp(capture.timestamp - previousCaptureTime, 1.0, 1000.0);
            }
            const auto nrStart = ClockMs();
            g.lateSplit = true; // (Settings::editLate: the model's list goes to the GPU after the frame is presented)
            g.Begin();
            if (diagnostics && (timingFrame++ % 32) == 0) g.BeginTimings();
            // (No FPS figures are drawn into a picture the calibrator measures.)
            // The difference of two models: the other model runs first, and the picture's own resolve subtracts its answer.
            processor.differenceAgainst = nullptr;
            if (differing) {
                session->comparison->Run(capture.Input(), otherSide, hdr, reset || (processed && current != applied), desktopWhite, frameTime, -1, -1);
                processor.differenceAgainst = session->comparison->LastAnswer();
            }
            auto* output = processor.Run(capture.Input(), pictureSide, hdr, reset, desktopWhite, frameTime, calibrating ? -1 : displayedFps, calibrating ? -1 : displayedOutputFps);
            if (comparing && !differing) {
                auto* other = session->comparison->Run(capture.Input(), otherSide, hdr, reset || (processed && current != applied), desktopWhite, frameTime, -1, -1);
                output = session->composer->Compose(output, other, current);
            }
            if (calibrating) calibration.Record(g, output);
            timings.nr += ClockMs() - nrStart;
            const auto fgStart = ClockMs();
            auto* sharedMotion = processor.SceneMotion();
            session->display->Copy(output, reset || (processed && current != applied) || (sharedMotion && processor.historyReset), sharedMotion);
            g.ResolveTimings(); g.Submit();
            timings.fgPrepare += ClockMs() - fgStart;
            const auto presentStart = ClockMs(); timings.record += presentStart - processingStarted;
            session->display->Present(); timings.present += ClockMs() - presentStart;
            g.SubmitLate();
            if (frameLog)
                std::fprintf(frameLog, "%llu,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%u,%d,%d,%u\n", (unsigned long long)frameIndex++,
                    capture.timestamp, acquiredAt, processingStarted, reacquired ? 1 : 0, presentStart,
                    session->display->presentCalled, session->display->presentReturned, session->display->lastPresentCount,
                    lowLatency ? 1 : 0, session->display->FlipWindow() ? 1 : 0, session->fg ? current.frameGeneration : 1u);
            pacer.Submitted(current.ProcessingFps(), processingStarted); pendingCapture = false;
            historyReset = false; previousCaptureTime = capture.timestamp; previousForeground = foreground;
            applied = current; processed = true;
            lastFrame = ClockMs();
            ++fpsFrames;
            if (fresh) { latencySum += lastFrame - capture.timestamp; ++frames; resets += processor.historyReset ? 1 : 0; }
            if (calibrating) { reportAt = lastFrame; frames = resets = 0; latencySum = 0; } // its status line stays up
            if (lastFrame - reportAt >= 2000 && frames) {
                DXGI_QUERY_VIDEO_MEMORY_INFO memory{}; g.adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory);
                std::string passSizes;
                for (unsigned pass = 0; pass < current.passes; ++pass) {
                    const auto resolution = current.EffectiveResolution(pass);
                    const auto layout = DlssNr::Spatial::Build(resolution.spatial, capture.width, capture.height, resolution.modelScale * .01f);
                    const auto [modelWidth, modelHeight] = processor.PassModelSize(pass);
                    const char* reconstruction = resolution.modelScale > 100 ? "supersampled" :
                        resolution.modelScale == 100 ? "native" : resolution.transfer == 2 ? "edge-aware" :
                        resolution.upscaler == 1 ? "DLSS SR" : "bilinear";
                    if (!passSizes.empty()) passSizes += ", ";
                    if (current.Shortcut(pass).reapply) { passSizes += std::format("P{} re-apply", pass + 1); continue; }
                    passSizes += std::format("P{} {}x{} {}{}", pass + 1, modelWidth, modelHeight,
                        reconstruction, layout.active ? "/foveated" : "");
                }
                auto text = std::format("{}x{} | {} | {} | {:.1f} processed fps | {:.1f} ms source-frame age at Present return | {:.0f} MiB app VRAM | {} motion | {} pass(es) | {:.0f}-nit reference white",
                    capture.width, capture.height, passSizes,
                    current.inputColour == 1 || (current.inputColour == 0 && !hdr) ? "finished SDR" : "HDR proxy",
                    frames * 1000.0 / (lastFrame - reportAt),
                    latencySum / frames, memory.CurrentUsage / 1048576.0, current.estimateMotion ? "estimated" : "zero", current.passes,
                    current.referenceWhiteNits > 0 ? current.referenceWhiteNits : desktopWhite * 80);
                text += current.maxFps ? std::format(" | target {} / {}x = {:.2f} processing cap", current.maxFps, current.frameGeneration, current.ProcessingFps()) : " | uncapped";
                text += session->fg ? (initial.presentMode == 1 ? " | DLSS FG swapchain (the flip window applies with DLSS FG off)" : "") :
                    session->display->FlipWindow() ? " | flip window" : " | overlay surface";
                if (current.presentMode != initial.presentMode) text += " (restart capture to change the presentation)";
                text += lowLatency ? " | low latency" : " | standard pacing";
                if (session->fg) {
                    const auto presented = session->fg->PresentedFrames();
                    text += std::format(" | DLSS FG {:.1f} presented fps (synthetic guides)", (presented - reportedPresented) * 1000.0 / (lastFrame - reportAt));
                    text += std::format(" | requested {}x, observed {:.2f}x (driver overrides may apply)", current.frameGeneration, double(presented - reportedPresented) / frames);
                    text += sharedMotion ? " | FG reuses NR motion" : " | FG estimates output motion";
                    reportedPresented = presented;
                }
                text += std::format(" | NR resets {}/{}", resets, frames);
                if (processor.borrowEvaluations) {
                    char borrow[256]{}; NrPassBorrow::Status(borrow, sizeof(borrow)); text += std::string(" | ") + borrow;
                }
                // The fused blocks and the rest of the network run 4-bit only in a "Model A - NVFP4" pass.
                const bool nvfp4Model = current.Nvfp4Pass();
                const bool passInOne = current.Nvfp4PassInOne(); // (such a model runs all of its own parts)
                if (current.nvfp4FeedForward || passInOne) text += " | NVFP4 feed-forward " + Nvfp4Status();
                if ((current.nvfp4Swin || passInOne) && nvfp4Model) text += " | NVFP4 fused blocks " + Nvfp4SwinStatus();
                if ((current.nvfp4Split || current.nvfp4Vit || current.nvfp4PrePost || passInOne) && nvfp4Model) text += " | NVFP4 network " + Nvfp4NetStatus();
                if (current.Fp8Pass()) text += " | FP8 model " + Fp8Status();
                if (current.Int8Pass()) text += " | INT8 model on the own pass host";
                if (comparing) text += std::format(" | comparing with model {} ({})", current.compareModel, current.compareSwap ? "right" : "left");
                const auto acquiredFps = (capture.receivedFrames - reportedCaptures) * 1000.0 / (lastFrame - reportAt);
                text += std::format(" | {} capture {:.1f} fps | GPU wait {:.1f} ms | Present {:.1f} ms",
                    initial.captureBackend ? "DXGI" : "WGC", acquiredFps, timings.gpuWait / frames, timings.present / frames);
                if (current.holdFrame) text += " | held frame";
                if (!current.applyModel) text += " | edit hidden";
                if (current.showFoveation) text += std::format(" | wireframe: pass {}", current.foveationPreviewPass + 1);
                Say(Wide(text));
                if (diagnostics || lastFrame - lastLog >= 30000) { Log(text); lastLog = lastFrame; }
                if (diagnostics) {
                    Log(std::format("Pipeline: {:.1f} acquired fps | ms per processed frame: capture {:.2f}, waiting for capture {:.2f}, FPS cap {:.2f}, GPU wait {:.2f}, record/prepare {:.2f}, Present {:.2f}",
                        (capture.receivedFrames - reportedCaptures) * 1000.0 / (lastFrame - reportAt),
                        timings.capture / frames, timings.idle / frames, timings.pacing / frames,
                        timings.gpuWait / frames, timings.record / frames, timings.present / frames));
                    Log(std::format("Preparation CPU elapsed (includes internal GPU waits): Reflex {:.2f} ms | NR record/motion {:.2f} ms | FG guides/encode/copy {:.2f} ms",
                        timings.reflex / frames, timings.nr / frames, timings.fgPrepare / frames));
                    if (timings.stageFrames) {
                        std::string stageReport = "GPU stage elapsed (sampled every 32 frames; includes queue scheduling):";
                        for (unsigned i = 0; i < timings.stages.size(); ++i)
                            stageReport += std::format(" {} {:.2f} ms |", GpuStageNames[i], timings.stages[i] / timings.stageFrames);
                        Log(stageReport);
                    }
                    Log(std::format("Capture producer: {:.1f} published fps | {:.2f} ms GPU copy/convert+wait | frame age at Present {:.2f} ms",
                        (capture.publishedFrames - reportedPublished) * 1000.0 / (lastFrame - reportAt),
                        (capture.copyMilliseconds - reportedCopyMs) / std::max<uint64_t>(1, capture.publishedFrames - reportedPublished), latencySum / frames));
                }
                reportedCaptures = capture.receivedFrames; timings = {};
                reportedPublished = capture.publishedFrames; reportedCopyMs = capture.copyMilliseconds;
                reportAt = lastFrame; frames = resets = 0; latencySum = 0;
            }
        }
        Say(L"Stopped. The overlay is hidden.");
    } catch (const winrt::hresult_error& e) { Say(e.message().c_str()); Log(Narrow(e.message().c_str())); }
      catch (const std::exception& e) { Say(Wide(e.what())); Log(e.what()); }
    capturing = false;
    calibration.Cancel(L"Calibration stopped: capture ended.");
    if (auto finished = calibration.TakeResult()) {
        { std::lock_guard lock(mutex); calibrationResult = std::move(finished); }
        PostMessageW(window, CalibrationMessage, 0, 0);
    }
    if (session) {
        if (session->display) {
            try { session->display->Hide(); } catch (const std::exception& e) { Log(e.what()); }
        }
        if (session->capture) session->capture->StopProducer();
        try {
            if (!session->gpu.Drain()) {
                // An unresolved submission retains all ownership until process exit.
                session.release(); restartRequired = true; calibration.Abandon();
                Say(L"GPU completion timed out. Capture stopped; restart this application before retrying.");
            }
        } catch (...) { /* Device removal: releasing the dead device's resources is safe. */ }
        // The NVFP4 kernels and buffers live outside the session; unreleased they would keep its device alive.
        if (session) ReleaseNvfp4(session->gpu.device.Get());
    }
    session.reset(); PostMessageW(window, StatusMessage, 1, 0);
    winrt::uninit_apartment();
}
}
