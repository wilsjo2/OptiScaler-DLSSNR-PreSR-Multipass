#include "Processor.h"
#include <algorithm>

namespace nr {
namespace {
Settings StageSettings(const Settings& s, unsigned pass) {
    const auto resolution = s.EffectiveResolution(pass);
    Settings stage;
    stage.runtime = s.runtime; stage.model = s.PassModel(pass);
    stage.modelScale = resolution.modelScale; stage.downscaler = resolution.downscaler;
    stage.upscaler = resolution.upscaler; stage.transfer = resolution.transfer; stage.spatial = resolution.spatial;
    stage.estimateMotion = s.estimateMotion; stage.inputColour = 1;
    // Raw encoded proxy stages. Reversible replace + passthrough returns the
    // reconstructed model image; user appearance controls run only in the host.
    stage.tone = 4; stage.highlightGuard = 8;
    return stage;
}
// The settings of the path where every pass has one resolution (the controls' own; no model fixes its own any more).
const Settings& Single(const Settings& s, Settings&) { return s; }
}

void Processor::Resize(UINT w, UINT h, const Settings& s) {
    if (!modelStage) {
        // A different set of cuts (borrow options, or re-apply moving the evaluations) restarts the borrow template.
        std::array<NrPassBorrow::Cut, Settings::MaxPasses> cuts{}; unsigned evaluations = 0;
        BorrowCuts(s, cuts, evaluations);
        if (cuts != activeCuts) { NrPassBorrow::Reset(); activeCuts = cuts; }
    }
    const bool nextAdvanced = !modelStage && s.HasPassResolutionOverrides();
    if (nextAdvanced != advanced) {
        if (!gpu.Drain()) throw std::runtime_error("GPU timeout changing NR pass layout.");
        activePasses = 0; advanced = nextAdvanced;
    }
    composeOnly = advanced;
    if (!advanced) {
        if (!stages.empty()) {
            if (!gpu.Drain()) throw std::runtime_error("GPU timeout releasing independent NR passes.");
            stages.clear(); stageSettings.clear(); chainInput.Reset();
        }
        Settings scratch;
        ResizeLegacy(w, h, Single(s, scratch));
        return;
    }
    Settings host = s;
    host.modelScale = 100; host.upscaler = 0; host.transfer = 1; host.spatial.enabled = false;
    host.passes = 1; host.estimateMotion = false;
    ResizeLegacy(w, h, host);
    if (!chainInput || chainInput->GetDesc().Width != w || chainInput->GetDesc().Height != h) {
        chainInput = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT); chainReapplied.Reset();
    }
    if (stages.size() != s.passes) {
        if (!gpu.Drain()) throw std::runtime_error("GPU timeout changing NR pass count.");
        stages.resize(s.passes); stageSettings.resize(s.passes);
    }
    for (unsigned pass = 0, evaluated = 0; pass < s.passes; ++pass) {
        if (s.Shortcut(pass).reapply) {
            // A foveated stage's answer is the unpacked packed answer; above 100% it is also downscaled, so no
            // capture-size input of the same detail exists to take the change against (see RunChain's extend).
            const auto& last = *stages[evaluated];
            if (last.spatialLayout.active && !last.resizeInput && (last.modelWidth != w || last.modelHeight != h))
                throw std::runtime_error(std::format("Pass {} cannot re-apply pass {}'s change: it uses foveation above 100% "
                                                     "model resolution. Set that pass to 100% or below, or untick Re-apply.",
                                                     pass + 1, evaluated + 1));
            // Re-apply passes evaluate nothing: release their independent feature like a removed pass.
            if (stages[pass]) {
                if (!gpu.Drain()) throw std::runtime_error("GPU timeout releasing a re-apply pass.");
                stages[pass].reset();
            }
            stageSettings[pass] = StageSettings(s, pass);
            continue;
        }
        if (!stages[pass]) {
            stages[pass] = std::shared_ptr<Processor>(new Processor(gpu));
            stages[pass]->runtime = std::make_shared<Runtime>(runtime);
            stages[pass]->modelStage = true;
            stages[pass]->clampStageInput = pass != 0;
        }
        const auto next = StageSettings(s, pass);
        stages[pass]->stageResetPending |= stageSettings[pass] != next;
        stageSettings[pass] = next;
        stages[pass]->Resize(w, h, stageSettings[pass]);
        evaluated = pass;
    }
}

std::pair<UINT, UINT> Processor::PassModelSize(unsigned pass) const {
    if (advanced) return pass < stages.size() && stages[pass] ? stages[pass]->PassModelSize(0) : std::pair<UINT, UINT>{0, 0};
    if (pass >= activePasses) return {0, 0};
    return spatialLayout.active ? std::pair{spatialLayout.modelW, spatialLayout.modelH} : std::pair{modelWidth, modelHeight};
}

ID3D12Resource* Processor::Run(ID3D12Resource* input, const Settings& s, bool hdr, bool resetHistory,
    float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps) {
    if (!modelStage) {
        borrowEvaluations = 0;
        // A pass-in-one model is made of these parts as 4-bit kernels with its own weights: it runs with them.
        if (s.Nvfp4PassInOne()) { ApplyNvfp4(true, true, true); ApplyNvfp4Swin(12, ~0ull, s.nvfp4SwinTwo, 1, true); ApplyNvfp4Net(63, 7, 0, s.nvfp4Net, 3); }
        else { ApplyNvfp4(s.nvfp4FeedForward, s.nvfp4FfnTrained); ApplyNvfp4Swin(s.nvfp4Swin, s.nvfp4SwinLayers, s.nvfp4SwinTwo, s.nvfp4SwinTrained); ApplyNvfp4Net(s.nvfp4Split, s.nvfp4Vit, s.nvfp4PrePost, s.nvfp4Net, (s.nvfp4SplitTrained ? 1u : 0u) | (s.nvfp4VitTrained ? 2u : 0u)); }
        ApplyFp8(s.Fp8Pass()); // (throws when a pass asks for an FP8 model that cannot be loaded)
    }
    Settings scratch;
    auto* result = advanced ? RunChain(input, s, hdr, resetHistory, desktopWhitePoint, frameTimeMs, processedFps, outputFps)
                           : RunLegacy(input, Single(s, scratch), hdr, resetHistory, desktopWhitePoint, frameTimeMs, processedFps, outputFps);
    return !modelStage && s.showFoveation ? Preview(result, s) : result;
}

ID3D12Resource* Processor::RunChain(ID3D12Resource* input, const Settings& s, bool hdr, bool resetHistory,
    float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps) {
    // Keep the held source common to all passes and final composition.
    if (!s.holdFrame) { resetHistory |= heldValid; heldValid = false; }
    else {
        if (!heldInput) heldInput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        if (!heldValid) {
            Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Barrier(gpu.commands.Get(), heldInput.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.commands->CopyResource(heldInput.Get(), input);
            Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            Barrier(gpu.commands.Get(), heldInput.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            heldValid = true; resetHistory = true;
        }
        input = heldInput.Get();
    }
    if (!s.enabled || s.compare == 1 || s.strength == 0) {
        for (auto& stage : stages) if (stage) stage->stageResetPending = true;
        return RunLegacy(input, s, hdr, true, desktopWhitePoint, frameTimeMs, processedFps, outputFps);
    }
    const bool sdr = s.inputColour == 1 || (s.inputColour == 0 && !hdr);
    const float white = s.referenceWhiteNits > 0 ? s.referenceWhiteNits / 80.0f :
        desktopWhitePoint > 0 ? desktopWhitePoint : hdr ? 203.0f / 80.0f : 1;
    resetHistory |= lastTone != s.tone || lastHdr != hdr || lastSdr != sdr || lastWhitePoint != white;
    DlssNrConstants encode{}; encode.Mode = sdr ? 21 : DlssNrMode_Encode;
    encode.Width = width; encode.Height = height; encode.WhitePoint = white; encode.ReversibleMode = s.HdrMode();
    Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Dispatch(98, encode, input, nullptr, nullptr, chainInput.Get(), original.Get());
    Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    Barrier(gpu.commands.Get(), chainInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    ID3D12Resource* current = chainInput.Get();
    ID3D12Resource* stageMotion = nullptr;
    bool reset = resetHistory;
    auto* c = gpu.commands.Get();
    std::array<NrPassBorrow::Cut, Settings::MaxPasses> cuts{}; unsigned evaluations = 0, evaluation = 0;
    const bool borrowing = BorrowCuts(s, cuts, evaluations);
    if (borrowing) { NrPassBorrow::BeginFrame(c, cuts.data(), evaluations); borrowEvaluations = evaluations; }
    const BorrowFrame borrowFrame{borrowing, borrowEvaluation};
    // A run of r re-apply stages (null) is one host dispatch at capture size: the last evaluated stage's output
    // extended by r * gain times its change from that stage's bounded input (kept in its original texture). A
    // foveated stage's output is its unpacked packed answer, softer in the periphery than original: take the change
    // from its unpacked proxy instead (same grid and filter; Resize refuses the sizes where the two differ).
    unsigned evaluated = 0, run = 0;
    const auto extend = [&](unsigned pass) {
        const auto& stage = *stages[evaluated];
        auto* before = (stage.spatialLayout.active && !stage.resizeInput ? stage.proxy : stage.original).Get();
        if (!chainReapplied) chainReapplied = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        Barrier(c, chainReapplied.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, current, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(c, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        DlssNrConstants k{}; k.Mode = 22; k.Width = width; k.Height = height; k.TransferStrength = float(run) * s.reapplyGain;
        Dispatch(ReapplySlot(pass), k, current, before, nullptr, chainReapplied.Get(), nullptr);
        Barrier(c, before, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, current, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        Barrier(c, chainReapplied.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        current = chainReapplied.Get(); run = 0;
    };
    for (unsigned pass = 0; pass < stages.size(); ++pass) {
        if (!stages[pass]) { ++run; continue; }
        if (run) extend(pass - 1);
        auto& stage = *stages[pass];
        stage.chainMotionInput = input;
        if (reuseChainMotion && s.estimateMotion) {
            // Foveation transforms these ordinary vectors later; different
            // foveation layouts can still share the same capture estimate.
            const auto filter = [&](unsigned index) {
                return stageSettings[index].modelScale > 100 ? stageSettings[index].downscaler : 0u;
            };
            for (unsigned previous = 0; previous < pass; ++previous) {
                if (!stages[previous]) continue;
                const auto& candidate = *stages[previous];
                if (candidate.modelWidth == stage.modelWidth && candidate.modelHeight == stage.modelHeight &&
                    filter(previous) == filter(pass) && candidate.usedMotionVectors) {
                    stage.sharedMotionVectors = candidate.usedMotionVectors;
                    break;
                }
            }
        }
        if (borrowing) stage.borrowEvaluation = int(evaluation);
        current = stage.RunLegacy(current, stageSettings[pass], false, reset || stage.stageResetPending, 1, frameTimeMs, -1, -1);
        if (borrowing) { borrowOutcomes[evaluation] = stage.borrowOutcomes[evaluation]; stage.borrowEvaluation = -1; }
        ++evaluation; evaluated = pass;
        stage.chainMotionInput = nullptr;
        stage.sharedMotionVectors = nullptr;
        stage.stageResetPending = false;
        reset |= stage.historyReset;
        if (!stageMotion) stageMotion = stage.SceneMotion();
        Barrier(gpu.commands.Get(), current, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    if (run) extend(unsigned(stages.size()) - 1);
    Barrier(gpu.commands.Get(), current, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(gpu.commands.Get(), chainInput.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    externalAnswer = current;
    preparedProxy = chainInput.Get();
    auto* result = RunLegacy(input, s, hdr, reset, desktopWhitePoint, frameTimeMs, processedFps, outputFps);
    externalAnswer = nullptr;
    preparedProxy = nullptr;
    Barrier(gpu.commands.Get(), current, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    for (auto& stage : stages)
        if (stage) Barrier(gpu.commands.Get(), stage->output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(gpu.commands.Get(), chainInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    historyReset = reset; sceneMotion = stageMotion;
    return result;
}

ID3D12Resource* Processor::Preview(ID3D12Resource* input, const Settings& s) {
    const auto selected = s.PassResolution(std::min(s.foveationPreviewPass, Settings::MaxPasses - 1));
    auto layout = DlssNr::Spatial::Build(selected.spatial, width, height, selected.modelScale * .01f);
    layout.ordinaryW = std::max(1u, (width * selected.modelScale + 50) / 100);
    layout.ordinaryH = std::max(1u, (height * selected.modelScale + 50) / 100);
    if (selected.spatial.enabled && !layout.active && layout.reason != "work is 100% on both axes")
        throw std::runtime_error("Invalid foveation preview: " + layout.reason);
    if (!spatialShader) spatialShader = std::make_shared<Spatial>(gpu);
    if (!gridOutput) gridOutput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    auto* c = gpu.commands.Get();
    if (gridReadable) Barrier(c, gridOutput.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Barrier(c, input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    spatialShader->Preview(layout, input, gridOutput.Get());
    Barrier(c, input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(c, gridOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    gridReadable = true;
    return gridOutput.Get();
}
}
