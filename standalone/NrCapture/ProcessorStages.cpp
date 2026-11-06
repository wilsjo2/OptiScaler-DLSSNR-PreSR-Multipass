#include "Processor.h"
#include <algorithm>

namespace nr {
namespace {
Settings StageSettings(const Settings& s, unsigned pass) {
    const auto resolution = s.PassResolution(pass);
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
}

void Processor::Resize(UINT w, UINT h, const Settings& s) {
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
        ResizeLegacy(w, h, s);
        return;
    }
    Settings host = s;
    host.modelScale = 100; host.upscaler = 0; host.transfer = 1; host.spatial.enabled = false;
    host.passes = 1; host.estimateMotion = false;
    ResizeLegacy(w, h, host);
    if (!chainInput || chainInput->GetDesc().Width != w || chainInput->GetDesc().Height != h)
        chainInput = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (stages.size() != s.passes) {
        if (!gpu.Drain()) throw std::runtime_error("GPU timeout changing NR pass count.");
        stages.resize(s.passes); stageSettings.resize(s.passes);
    }
    for (unsigned pass = 0; pass < s.passes; ++pass) {
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
    }
}

std::pair<UINT, UINT> Processor::PassModelSize(unsigned pass) const {
    if (advanced) return pass < stages.size() ? stages[pass]->PassModelSize(0) : std::pair<UINT, UINT>{0, 0};
    if (pass >= activePasses) return {0, 0};
    return spatialLayout.active ? std::pair{spatialLayout.modelW, spatialLayout.modelH} : std::pair{modelWidth, modelHeight};
}

ID3D12Resource* Processor::Run(ID3D12Resource* input, const Settings& s, bool hdr, bool resetHistory,
    float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps) {
    auto* result = advanced ? RunChain(input, s, hdr, resetHistory, desktopWhitePoint, frameTimeMs, processedFps, outputFps)
                           : RunLegacy(input, s, hdr, resetHistory, desktopWhitePoint, frameTimeMs, processedFps, outputFps);
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
        for (auto& stage : stages) stage->stageResetPending = true;
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
    ID3D12Resource* fullMotion = nullptr;
    bool reset = resetHistory;
    for (unsigned pass = 0; pass < stages.size(); ++pass) {
        auto& stage = *stages[pass];
        stage.chainMotionInput = input;
        if (reuseChainMotion && s.estimateMotion) {
            // Foveation transforms these ordinary vectors later; different
            // foveation layouts can still share the same capture estimate.
            const auto filter = [&](unsigned index) {
                return stageSettings[index].modelScale > 100 ? stageSettings[index].downscaler : 0u;
            };
            for (unsigned previous = 0; previous < pass; ++previous) {
                const auto& candidate = *stages[previous];
                if (candidate.modelWidth == stage.modelWidth && candidate.modelHeight == stage.modelHeight &&
                    filter(previous) == filter(pass) && candidate.usedMotionVectors) {
                    stage.sharedMotionVectors = candidate.usedMotionVectors;
                    break;
                }
            }
        }
        current = stage.RunLegacy(current, stageSettings[pass], false, reset || stage.stageResetPending, 1, frameTimeMs, -1, -1);
        stage.chainMotionInput = nullptr;
        stage.sharedMotionVectors = nullptr;
        stage.stageResetPending = false;
        reset |= stage.historyReset;
        if (!fullMotion) fullMotion = stage.FullResolutionMotion();
        Barrier(gpu.commands.Get(), current, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    Barrier(gpu.commands.Get(), current, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(gpu.commands.Get(), chainInput.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    externalAnswer = current;
    preparedProxy = chainInput.Get();
    auto* result = RunLegacy(input, s, hdr, reset, desktopWhitePoint, frameTimeMs, processedFps, outputFps);
    externalAnswer = nullptr;
    preparedProxy = nullptr;
    Barrier(gpu.commands.Get(), current, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    for (auto& stage : stages)
        Barrier(gpu.commands.Get(), stage->output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(gpu.commands.Get(), chainInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    historyReset = reset; fullResolutionMotion = fullMotion;
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
