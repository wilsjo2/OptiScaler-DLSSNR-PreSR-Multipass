#include "Processor.h"
#include <shaders/dlssnr/precompile/DlssNr_Shader.h>
#include <CaptureColour.h>
#include <array>
namespace nr {
Processor::Processor(Gpu& g, const std::filesystem::path& path, bool unlockAdaMfg) : Processor(g) {
    runtime = std::make_shared<Runtime>(g.device.Get(), path, unlockAdaMfg);
}
Processor::Processor(Gpu& g) : gpu(g) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 5};
    D3D12_ROOT_PARAMETER parameters[2]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {2, ranges};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[1].Descriptor.ShaderRegister = 0;
    D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_ROOT_SIGNATURE_DESC signature{2, parameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, error;
    Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), "NR root signature");
    Check(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "NR root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature = root.Get();
    pipeline.CS = {DlssNr_cso, sizeof(DlssNr_cso)};
    Check(gpu.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&shader)), "NR colour shader");
    pipeline.CS = {CaptureColour, sizeof(CaptureColour)};
    Check(gpu.device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&captureColour)), "Capture colour conversion");
    D3D12_DESCRIPTOR_HEAP_DESC h{}; h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    h.NumDescriptors = 105; h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gpu.device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&heap)), "NR descriptors");
    stride = gpu.device->GetDescriptorHandleIncrementSize(h.Type);
    constants = gpu.Buffer(15 * 256, D3D12_HEAP_TYPE_UPLOAD);
}
Processor::~Processor() {
    stages.clear();
    // Share the same fence-backed retention used by the in-game NR implementation.
    gpu.lifetime.Retire([model = std::move(runtime), flow = std::move(opticalFlow), a = std::move(proxy), b = std::move(original),
        c = std::move(answer), d = std::move(output), e = std::move(depth), f = std::move(motion),
        full = std::move(fullProxy), flowSource = std::move(flowInput), filtered = std::move(filteredAnswer),
        srInput = std::move(resizeInput), upscaled = std::move(enlarged), bounded = std::move(boundedDlss), unit = std::move(exposure),
        encoded = std::move(sdrInput), composed = std::move(sdrOutput), held = std::move(heldInput), conversion = std::move(captureColour),
        labels = std::move(labelled), filters = std::move(scaling), native = std::move(nativeAnswer), next = std::move(passInput),
        spatial = std::move(spatialShader), packed = std::move(packedProxy), packedResult = std::move(packedAnswer),
        packedZ = std::move(packedDepth), packedMv = std::move(packedMotion), packedNext = std::move(packedPassInput),
        grid = std::move(gridOutput), chain = std::move(chainInput),
        cb = std::move(constants), descriptors = std::move(heap), rs = std::move(root), pso = std::move(shader)] {});
    gpu.lifetime.Collect();
}
D3D12_CPU_DESCRIPTOR_HANDLE Processor::Cpu(UINT i) const {
    auto h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * stride; return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE Processor::GpuHandle(UINT i) const {
    auto h = heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * stride; return h;
}
void Processor::ResizeLegacy(UINT w, UINT h, const Settings& s) {
    if (!w || !h || w > 7680 || h > 4320) throw std::runtime_error("Capture extent is outside the prototype's 8K limit.");
    if (s.passes < 1 || s.passes > Settings::MaxPasses) throw std::runtime_error("Choose 1-10 NR passes.");
    if (s.modelScale < 25 || s.modelScale > 200) throw std::runtime_error("Model resolution must be 25-200%.");
    if (s.upscaler > 1 || s.transfer > 2)
        throw std::runtime_error("Unknown upscaler or edit transfer.");
    const UINT mw = std::max(1u, (w * s.modelScale + 50) / 100);
    const UINT mh = std::max(1u, (h * s.modelScale + 50) / 100);
    auto nextSpatial = DlssNr::Spatial::Build(s.spatial, w, h, s.modelScale * .01f);
    // Preserve the legacy integer rounding for ordinary resources. The shared
    // float-scale builder can round half pixels differently for odd extents.
    nextSpatial.ordinaryW = mw; nextSpatial.ordinaryH = mh;
    if (nextSpatial.requested && !nextSpatial.active && nextSpatial.reason != "work is 100% on both axes")
        throw std::runtime_error("Invalid foveation layout: " + nextSpatial.reason);
    const bool spatialChanged = spatialLayout != nextSpatial;
    const bool resized = width != w || height != h || modelWidth != mw || modelHeight != mh || spatialChanged ||
        (!composeOnly && !proxy);
    const bool useDlss = s.upscaler == 1 && s.transfer != 2 && (mw < w || mh < h);
    if (s.modelScale > 100 && !scaling) scaling = std::make_shared<Scaling>(gpu);
    if (resized || (!s.estimateMotion && flowInput) || (s.estimateMotion && (mw != w || mh != h) && !flowInput)) {
        if (!gpu.Drain()) throw std::runtime_error("GPU timeout resizing model inputs.");
        flowInput.Reset();
        if (s.estimateMotion && (mw != w || mh != h)) flowInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    if (s.estimateMotion) {
        if (!opticalFlow) {
            if (!gpu.Drain()) throw std::runtime_error("GPU timeout enabling optical flow.");
            opticalFlow = std::make_shared<Motion>(gpu);
        }
        opticalFlow->Resize(mw, mh);
    }
    if (!resized && activeModels == s.Models() && activePasses == s.passes && privateDlss == useDlss) return;
    if (opticalFlow) opticalFlow->Reset();
    if (!gpu.Drain()) throw std::runtime_error("GPU timeout while rebuilding NR.");
    spatialLayout = nextSpatial;
    runtime->ReleaseDlss(); privateDlss = false;
    resizeInput.Reset(); enlarged.Reset(); boundedDlss.Reset(); exposure.Reset();
    if (mw < w || mh < h) resizeInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (s.passes == 1) passInput.Reset();
    else if (!passInput || resized) passInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (resized) {
        packedProxy.Reset(); packedAnswer.Reset(); packedDepth.Reset(); packedMotion.Reset(); packedPassInput.Reset();
        gridOutput.Reset(); gridReadable = false;
        filteredAnswer.Reset();
        labelled.Reset(); labelsReadable = false;
        nativeAnswer.Reset();
        if (mw != w || mh != h) nativeAnswer = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (width != w || height != h) { heldInput.Reset(); heldValid = false; }
        sdrInput.Reset(); sdrOutput.Reset();
        if (!composeOnly) proxy = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        fullProxy.Reset();
        if (mw != w || mh != h || spatialLayout.active) fullProxy = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        original = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (!composeOnly) answer = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        output = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (!composeOnly) {
            depth = gpu.Texture(mw, mh, DXGI_FORMAT_R32_FLOAT);
            motion = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16_FLOAT);
        }
        outputReadable = false; width = w; height = h;
        modelWidth = mw; modelHeight = mh;
        if (!composeOnly) {
            gpu.Begin();
            ID3D12DescriptorHeap* heaps[]{heap.Get()}; gpu.commands->SetDescriptorHeaps(1, heaps);
            const float one[]{1, 1, 1, 1}, zero[]{0, 0, 0, 0};
            gpu.device->CreateUnorderedAccessView(depth.Get(), nullptr, nullptr, Cpu(63));
            gpu.device->CreateUnorderedAccessView(motion.Get(), nullptr, nullptr, Cpu(64));
            gpu.commands->ClearUnorderedAccessViewFloat(GpuHandle(63), Cpu(63), depth.Get(), one, 0, nullptr);
            gpu.commands->ClearUnorderedAccessViewFloat(GpuHandle(64), Cpu(64), motion.Get(), zero, 0, nullptr);
            Barrier(gpu.commands.Get(), depth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(gpu.commands.Get(), motion.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Guide initialization timeout.");
        }
    }
    if (composeOnly) {
        // The host only composes the children's answer against preparedProxy.
        // Do not retain a second unused native NR input/output and guide set.
        proxy.Reset(); answer.Reset(); depth.Reset(); motion.Reset(); sdrInput.Reset();
    }
    if (spatialLayout.active) {
        if (!spatialShader) spatialShader = std::make_shared<Spatial>(gpu);
        const auto sw = spatialLayout.modelW, sh = spatialLayout.modelH;
        if (!packedProxy) {
            packedProxy = gpu.Texture(sw, sh, DXGI_FORMAT_R16G16B16A16_FLOAT);
            packedAnswer = gpu.Texture(sw, sh, DXGI_FORMAT_R16G16B16A16_FLOAT);
            packedDepth = gpu.Texture(sw, sh, DXGI_FORMAT_R32_FLOAT);
            packedMotion = gpu.Texture(sw, sh, DXGI_FORMAT_R32G32_FLOAT);
        }
        if (s.passes > 1 && !packedPassInput) packedPassInput = gpu.Texture(sw, sh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    if (composeOnly) runtime->Release();
    else {
        gpu.Begin();
        const auto created = runtime->Create(gpu.commands.Get(), spatialLayout.active ? spatialLayout.modelW : mw,
            spatialLayout.active ? spatialLayout.modelH : mh, s.model, s.passes, s.passOverrides);
        // Even failed creation may record commands: finish its submission before releasing ownership.
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("NR creation timeout.");
        Runtime::CheckResult(created, "NR feature creation");
    }
    activeModels = s.Models(); activePasses = s.passes;
    if (useDlss) {
        enlarged = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        boundedDlss = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        exposure = gpu.Texture(1, 1, DXGI_FORMAT_R32_FLOAT);
        gpu.Begin();
        DlssNrConstants unit{}; unit.Mode = DlssNrMode_UnitExposure; unit.Width = unit.Height = 1;
        Dispatch(42, unit, proxy.Get(), nullptr, nullptr, exposure.Get(), nullptr);
        Barrier(gpu.commands.Get(), exposure.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const auto result = runtime->CreateDlss(gpu.commands.Get(), mw, mh, w, h);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Private DLSS creation timeout.");
        Runtime::CheckResult(result, "Private DLSS creation"); privateDlss = true;
    }
}
void Processor::Dispatch(UINT offset, const DlssNrConstants& s, ID3D12Resource* source,
    ID3D12Resource* model, ID3D12Resource* clean, ID3D12Resource* target, ID3D12Resource* keep) {
    ID3D12Resource* inputs[]{source, model ? model : source, clean ? clean : source, source, source};
    for (UINT i = 0; i < 5; ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = inputs[i]->GetDesc().Format;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
        gpu.device->CreateShaderResourceView(inputs[i], &view, Cpu(offset + i));
    }
    gpu.device->CreateUnorderedAccessView(target, nullptr, nullptr, Cpu(offset + 5));
    gpu.device->CreateUnorderedAccessView(keep ? keep : target, nullptr, nullptr, Cpu(offset + 6));
    void* data = nullptr; Check(constants->Map(0, nullptr, &data), "Map constants");
    const UINT cbOffset = (offset / 7) * 256; // Each dispatch needs its own descriptors and constants until submission completes.
    memcpy(static_cast<char*>(data) + cbOffset, &s, sizeof(s)); constants->Unmap(0, nullptr);
    auto* c = gpu.commands.Get(); ID3D12DescriptorHeap* heaps[]{heap.Get()};
    c->SetDescriptorHeaps(1, heaps); c->SetComputeRootSignature(root.Get());
    c->SetPipelineState(s.Mode >= 11 && s.Mode <= 21 ? captureColour.Get() : shader.Get());
    c->SetComputeRootDescriptorTable(0, GpuHandle(offset));
    c->SetComputeRootConstantBufferView(1, constants->GetGPUVirtualAddress() + cbOffset);
    c->Dispatch((s.Width + 7) / 8, (s.Height + 7) / 8, 1);
}
ID3D12Resource* Processor::RunLegacy(ID3D12Resource* input, const Settings& s, bool hdr, bool resetHistory, float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps) {
    // Colour and signed edits are different DLSS inputs; never reuse one history for the other.
    resetHistory |= privateDlss && lastTransfer != s.transfer;
    lastTransfer = s.transfer;
    fullResolutionMotion = nullptr;
    usedMotionVectors = nullptr;
    if (!s.holdFrame) { resetHistory |= heldValid; heldValid = false; }
    else {
        if (!heldInput) heldInput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        if (!heldValid) {
            auto* c = gpu.commands.Get();
            Barrier(c, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Barrier(c, heldInput.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            c->CopyResource(heldInput.Get(), input);
            Barrier(c, input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            Barrier(c, heldInput.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            heldValid = true; resetHistory = true;
        }
        input = heldInput.Get();
    }
    const bool sdr = !modelStage && (s.inputColour == 1 || (s.inputColour == 0 && !hdr));
    if (sdr && !preparedProxy && !sdrInput) sdrInput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (sdr && !sdrOutput) sdrOutput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    const float whitePoint = s.referenceWhiteNits > 0 ? s.referenceWhiteNits / 80.0f :
        desktopWhitePoint > 0 ? desktopWhitePoint : hdr ? 203.0f / 80.0f : 1.0f;
    ID3D12Resource* vectors = motion.Get(); bool reset = true;
    gpu.Time(GpuStage::InputMotion);
    if (!composeOnly && s.estimateMotion && s.enabled && s.strength != 0 && s.compare != 1) {
        if (sharedMotionVectors) {
            // Stages with the same ordinary input geometry/filter estimate the
            // same original capture. Reuse that result, while matching the zero
            // vectors of a fresh local history when this pass needs a reset.
            reset = resetHistory || lastTone != s.tone || lastHdr != hdr || lastSdr != sdr || lastWhitePoint != whitePoint;
            vectors = reset ? motion.Get() : sharedMotionVectors;
        } else {
            auto* motionInput = chainMotionInput ? chainMotionInput : input;
            Barrier(gpu.commands.Get(), motionInput, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (flowInput) {
                DlssNrConstants down{}; down.Mode = DlssNrMode_Downsample; down.Width = modelWidth; down.Height = modelHeight;
                if (s.modelScale > 100) scaling->Run(2, motionInput, flowInput.Get(), s.downscaler, true);
                else Dispatch(28, down, motionInput, nullptr, nullptr, flowInput.Get(), nullptr);
                Barrier(gpu.commands.Get(), flowInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            // Estimate in model pixels so both passes receive correctly scaled vectors.
            reset = opticalFlow->Estimate(flowInput ? flowInput.Get() : motionInput,
                resetHistory || lastTone != s.tone || lastHdr != hdr || lastSdr != sdr || lastWhitePoint != whitePoint);
            if (flowInput) Barrier(gpu.commands.Get(), flowInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(gpu.commands.Get(), motionInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            vectors = opticalFlow->Vectors();
        }
        usedMotionVectors = vectors;
        if (modelWidth == width && modelHeight == height) fullResolutionMotion = vectors;
    } else if (opticalFlow) opticalFlow->Reset();
    lastTone = s.tone; lastHdr = hdr; lastSdr = sdr; lastWhitePoint = whitePoint;
    historyReset = reset;
    gpu.Time(GpuStage::InputMotion, true);
    auto* c = gpu.commands.Get();
    if (outputReadable) Barrier(c, output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!s.enabled || s.compare == 1 || s.strength == 0) {
        Barrier(c, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        c->CopyResource(output.Get(), input);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    } else {
        gpu.Time(GpuStage::Encode);
        Barrier(c, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        DlssNrConstants p{}; p.Mode = DlssNrMode_Encode; p.Width = width; p.Height = height;
        p.WhitePoint = whitePoint;
        p.ReversibleMode = modelStage ? 2u : s.HdrMode(); // Raw stage replace avoids applying final controls between passes.
        if (sdr && !preparedProxy) {
            p.Mode = 11;
            Dispatch(49, p, input, nullptr, nullptr, sdrInput.Get(), nullptr);
            Barrier(c, sdrInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        p.Mode = clampStageInput ? 19u : DlssNrMode_Encode; p.Passthrough = sdr || modelStage ? 1u : 0u;
        if (!preparedProxy)
            Dispatch(0, p, sdr ? sdrInput.Get() : input, nullptr, nullptr, fullProxy ? fullProxy.Get() : proxy.Get(), original.Get());
        p.Width = modelWidth; p.Height = modelHeight;
        if (fullProxy && !spatialLayout.active) {
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            p.Mode = DlssNrMode_Downsample;
            if (s.modelScale > 100) scaling->Run(0, fullProxy.Get(), proxy.Get(), s.downscaler, true);
            else Dispatch(21, p, fullProxy.Get(), nullptr, nullptr, proxy.Get(), nullptr);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (!spatialLayout.active && !preparedProxy)
            Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(c, original.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Time(GpuStage::Encode, true);
        gpu.Time(GpuStage::NR);
        if (!externalAnswer && spatialLayout.active) {
            DlssNr::GuideRegions regions{};
            regions.depth = {0, 0, modelWidth, modelHeight};
            regions.motion = {0, 0, modelWidth, modelHeight};
            const auto make = [&](unsigned mode) {
                return DlssNr::Spatial::MakeConstants(spatialLayout, mode, regions,
                    float(width) / modelWidth, float(height) / modelHeight, width, height);
            };
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            spatialShader->Dispatch(0, make(100), fullProxy.Get(), nullptr, nullptr, packedProxy.Get());
            spatialShader->Dispatch(1, make(101), fullProxy.Get(), depth.Get(), vectors, packedDepth.Get(), packedMotion.Get());
            for (auto* r : {packedProxy.Get(), packedDepth.Get(), packedMotion.Get()})
                Barrier(c, r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            const UINT sw = spatialLayout.modelW, sh = spatialLayout.modelH;
            runtime->Evaluate(c, packedProxy.Get(), packedDepth.Get(), packedMotion.Get(), packedAnswer.Get(), sw, sh, reset);
            Barrier(c, packedAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            for (unsigned pass = 1; pass < s.passes; ++pass) {
                auto clamp = p; clamp.Mode = DlssNrMode_ClampProxy; clamp.Width = sw; clamp.Height = sh;
                Dispatch(14, clamp, packedAnswer.Get(), nullptr, nullptr, packedPassInput.Get(), nullptr);
                Barrier(c, packedPassInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(c, packedAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                runtime->Evaluate(c, packedPassInput.Get(), packedDepth.Get(), packedMotion.Get(), packedAnswer.Get(), sw, sh, reset, pass);
                Barrier(c, packedAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(c, packedPassInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            spatialShader->Dispatch(2, make(102), packedProxy.Get(), packedAnswer.Get(), nullptr, proxy.Get(), answer.Get());
            for (auto* r : {packedProxy.Get(), packedDepth.Get(), packedMotion.Get(), packedAnswer.Get(), fullProxy.Get()})
                Barrier(c, r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        } else if (!externalAnswer) {
            runtime->Evaluate(c, proxy.Get(), depth.Get(), vectors, answer.Get(), modelWidth, modelHeight, reset);
            Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            for (unsigned pass = 1; pass < s.passes; ++pass) {
                p.Mode = DlssNrMode_ClampProxy;
                Dispatch(14, p, answer.Get(), nullptr, nullptr, passInput.Get(), nullptr);
                Barrier(c, passInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                runtime->Evaluate(c, passInput.Get(), depth.Get(), vectors, answer.Get(), modelWidth, modelHeight, reset, pass);
                Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(c, passInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
        }
        gpu.Time(GpuStage::NR, true);
        ID3D12Resource* modelAnswer = externalAnswer ? externalAnswer : answer.Get();
        ID3D12Resource* resolveProxy = preparedProxy ? preparedProxy : proxy.Get();
        const bool halo = s.haloSuppression > 0 || s.haloBrightSuppression > 0;
        if (halo) {
            if (!filteredAnswer) filteredAnswer = gpu.Texture(modelWidth, modelHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
            auto filter = p; filter.Mode = 18;
            filter.HaloSuppression = std::clamp(s.haloSuppression, 0.f, 1.f);
            filter.HaloBrightSuppression = std::clamp(s.haloBrightSuppression, 0.f, 1.f);
            filter.HaloRadius = std::clamp(s.haloRadius, 1.f, 4.f);
            Dispatch(91, filter, resolveProxy, modelAnswer, nullptr, filteredAnswer.Get(), nullptr);
            Barrier(c, filteredAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            modelAnswer = filteredAnswer.Get();
        }
        const bool edgeAware = s.transfer == 2 && resizeInput;
        if (!externalAnswer && nativeAnswer && !resizeInput) {
            scaling->Run(1, modelAnswer, nativeAnswer.Get(), s.downscaler, false);
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            modelAnswer = nativeAnswer.Get(); resolveProxy = fullProxy.Get();
        }
        if (!externalAnswer && resizeInput && !edgeAware) {
            // Measure lighting against the input at the same resolution, before
            // enlargement. Reconstruct a native-size model image for the resolve.
            p.Mode = 16; p.Transfer = s.transfer;
            Dispatch(35, p, proxy.Get(), modelAnswer, nullptr, resizeInput.Get(), nullptr);
            Barrier(c, resizeInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ID3D12Resource* field = resizeInput.Get();
            if (privateDlss) {
                gpu.Time(GpuStage::PrivateDlss);
                runtime->EvaluateDlss(c, field, depth.Get(), vectors, exposure.Get(), enlarged.Get(), reset, frameTimeMs);
                Barrier(c, enlarged.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                gpu.Time(GpuStage::PrivateDlss, true);
                gpu.Time(GpuStage::EdgeCorrection);
                auto bound = p; bound.Mode = 14; bound.Width = width; bound.Height = height;
                Dispatch(77, bound, field, enlarged.Get(), nullptr, boundedDlss.Get(), nullptr);
                Barrier(c, boundedDlss.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                field = boundedDlss.Get();
                gpu.Time(GpuStage::EdgeCorrection, true);
            }
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            auto restore = p; restore.Mode = 17; restore.Width = width; restore.Height = height;
            Dispatch(84, restore, field, fullProxy.Get(), resizeInput.Get(), nativeAnswer.Get(), nullptr);
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(c, resizeInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            modelAnswer = nativeAnswer.Get(); resolveProxy = fullProxy.Get();
        }
        gpu.Time(GpuStage::Resolve);
        p.Mode = DlssNrMode_Resolve; p.ApplyModel = s.applyModel; p.TransferStrength = s.strength; p.ColourStrength = s.colourStrength;
        p.DebugView = s.debugView; p.DebugScale = sdr ? 1.0f : whitePoint;
        p.SkinProtection = s.separateSkin; p.ShowSkinMask = s.previewSkin;
        p.SkinDetail = s.skinDetail; p.SkinColour = s.skinColour;
        p.EnvironmentDetail = s.environmentDetail; p.EnvironmentColour = s.environmentColour;
        p.Width = width; p.Height = height;
        p.MaxRatio = s.highlightGuard; p.Transfer = edgeAware ? 5 : nativeAnswer ? 0 : std::min(s.transfer, 1u);
        p.CompareMode = s.compare == 3 ? 1 : s.compare == 2 ? 2 : 0;
        p.CompareSplit = s.compareSplit; p.CompareZoom = s.compareZoom; p.CompareSwap = s.compareSwap;
        if (s.debugView != 0) { modelAnswer = externalAnswer ? externalAnswer : answer.Get(); resolveProxy = preparedProxy ? preparedProxy : proxy.Get(); p.Transfer = 1; }
        if (modelStage && !edgeAware) p.Mode = 20;
        Dispatch(7, p, resolveProxy, modelAnswer, original.Get(), sdr ? sdrOutput.Get() : output.Get(), nullptr);
        if (sdr) {
            Barrier(c, sdrOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            p.Mode = 12;
            Dispatch(56, p, sdrOutput.Get(), input, preparedProxy ? preparedProxy : sdrInput.Get(), output.Get(), nullptr);
            Barrier(c, sdrOutput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!preparedProxy) Barrier(c, sdrInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (privateDlss) {
            Barrier(c, enlarged.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, boundedDlss.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (nativeAnswer && !edgeAware) {
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (!preparedProxy) Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, original.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!externalAnswer) Barrier(c, answer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (halo) Barrier(c, filteredAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        gpu.Time(GpuStage::Resolve, true);
    }
    Barrier(c, input, (!s.enabled || s.compare == 1 || s.strength == 0) ? D3D12_RESOURCE_STATE_COPY_SOURCE :
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    outputReadable = true;
    const bool compareLabels = s.compare >= 2 && s.compareLabels && s.debugView == 0 && s.enabled && s.strength != 0;
    const bool tuneHdr = hdr && s.enabled && s.strength != 0 && s.applyModel && !s.previewSkin && s.debugView == 0 &&
        s.compare != 1 && (s.peakBrightnessNits > 0 || s.blackLevelNits != 0);
    if (compareLabels || processedFps >= 0 || tuneHdr) {
        gpu.Time(GpuStage::HdrLabels);
        if (!labelled) labelled = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (labelsReadable) Barrier(c, labelled.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        DlssNrConstants label{}; label.Mode = 13; label.Width = width; label.Height = height;
        label.WhitePoint = whitePoint; label.DebugScale = s.labelScale;
        // Mode 13: one FPS value, or separate real-input / FG-output values.
        label.DebugView = processedFps < 0 ? 0 : outputFps >= 0 ? 2 : 1;
        label.CompareZoom = static_cast<float>(processedFps); label.TransferStrength = static_cast<float>(outputFps);
        label.CompareMode = s.compare == 3 ? 1 : s.compare == 2 ? 2 : 0;
        label.ApplyModel = compareLabels; label.CompareSplit = s.compareSplit; label.CompareSwap = s.compareSwap;
        label.Passthrough = !tuneHdr; label.MaxRatio = s.peakBrightnessNits / 80; label.ColourStrength = s.blackLevelNits / 80;
        label.MvScaleX = s.compareZoom;
        Dispatch(70, label, output.Get(), nullptr, nullptr, labelled.Get(), nullptr);
        Barrier(c, output.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(c, labelled.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        gpu.Time(GpuStage::HdrLabels, true);
        labelsReadable = true; return labelled.Get();
    }
    return output.Get();
}
}
