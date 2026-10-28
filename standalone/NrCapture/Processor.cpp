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
    h.NumDescriptors = 98; h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(gpu.device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&heap)), "NR descriptors");
    stride = gpu.device->GetDescriptorHandleIncrementSize(h.Type);
    constants = gpu.Buffer(14 * 256, D3D12_HEAP_TYPE_UPLOAD);
}
Processor::~Processor() {
    // Share the same fence-backed retention used by the in-game NR implementation.
    gpu.lifetime.Retire([model = std::move(runtime), flow = std::move(opticalFlow), a = std::move(proxy), b = std::move(original),
        c = std::move(answer), d = std::move(output), e = std::move(depth), f = std::move(motion),
        full = std::move(fullProxy), flowSource = std::move(flowInput), filtered = std::move(filteredAnswer),
        srInput = std::move(resizeInput), upscaled = std::move(enlarged), bounded = std::move(boundedDlss), unit = std::move(exposure),
        encoded = std::move(sdrInput), composed = std::move(sdrOutput), held = std::move(heldInput), conversion = std::move(captureColour),
        labels = std::move(labelled), filters = std::move(scaling), native = std::move(nativeAnswer), next = std::move(passInput), cb = std::move(constants), descriptors = std::move(heap), rs = std::move(root), pso = std::move(shader)] {});
    gpu.lifetime.Collect();
}
D3D12_CPU_DESCRIPTOR_HANDLE Processor::Cpu(UINT i) const {
    auto h = heap->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * stride; return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE Processor::GpuHandle(UINT i) const {
    auto h = heap->GetGPUDescriptorHandleForHeapStart(); h.ptr += UINT64(i) * stride; return h;
}
void Processor::Resize(UINT w, UINT h, const Settings& s) {
    if (!w || !h || w > 7680 || h > 4320) throw std::runtime_error("Capture extent is outside the prototype's 8K limit.");
    if (s.passes < 1 || s.passes > Settings::MaxPasses) throw std::runtime_error("Choose 1-10 NR passes.");
    if (s.modelScale < 25 || s.modelScale > 200) throw std::runtime_error("Model resolution must be 25-200%.");
    if (s.upscaler > 1 || s.transfer > 2)
        throw std::runtime_error("Unknown upscaler or edit transfer.");
    const UINT mw = std::max(1u, (w * s.modelScale + 50) / 100);
    const UINT mh = std::max(1u, (h * s.modelScale + 50) / 100);
    const bool resized = width != w || height != h || modelWidth != mw || modelHeight != mh;
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
    runtime->ReleaseDlss(); privateDlss = false;
    resizeInput.Reset(); enlarged.Reset(); boundedDlss.Reset(); exposure.Reset();
    if (mw < w || mh < h) resizeInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (s.passes == 1) passInput.Reset();
    else if (!passInput || resized) passInput = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (resized) {
        filteredAnswer.Reset();
        labelled.Reset(); labelsReadable = false;
        nativeAnswer.Reset();
        if (mw != w || mh != h) nativeAnswer = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (width != w || height != h) { heldInput.Reset(); heldValid = false; }
        sdrInput.Reset(); sdrOutput.Reset();
        proxy = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        fullProxy.Reset();
        if (mw != w || mh != h) fullProxy = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        original = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        answer = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16B16A16_FLOAT);
        output = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        depth = gpu.Texture(mw, mh, DXGI_FORMAT_R32_FLOAT);
        motion = gpu.Texture(mw, mh, DXGI_FORMAT_R16G16_FLOAT);
        outputReadable = false; width = w; height = h;
        modelWidth = mw; modelHeight = mh;
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
    gpu.Begin();
    const auto created = runtime->Create(gpu.commands.Get(), mw, mh, s.model, s.passes, s.passOverrides);
    // Even failed creation may record commands: finish its submission before releasing ownership.
    gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("NR creation timeout.");
    Runtime::CheckResult(created, "NR feature creation"); activeModels = s.Models(); activePasses = s.passes;
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
    c->SetPipelineState(s.Mode >= 11 && s.Mode <= 18 ? captureColour.Get() : shader.Get());
    c->SetComputeRootDescriptorTable(0, GpuHandle(offset));
    c->SetComputeRootConstantBufferView(1, constants->GetGPUVirtualAddress() + cbOffset);
    c->Dispatch((s.Width + 7) / 8, (s.Height + 7) / 8, 1);
}
ID3D12Resource* Processor::Run(ID3D12Resource* input, const Settings& s, bool hdr, bool resetHistory, float desktopWhitePoint, float frameTimeMs, int processedFps, int outputFps) {
    // Colour and signed edits are different DLSS inputs; never reuse one history for the other.
    resetHistory |= privateDlss && lastTransfer != s.transfer;
    lastTransfer = s.transfer;
    fullResolutionMotion = nullptr;
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
    const bool sdr = s.inputColour == 1 || (s.inputColour == 0 && !hdr);
    if (sdr && !sdrInput) {
        sdrInput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
        sdrOutput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    const float whitePoint = s.referenceWhiteNits > 0 ? s.referenceWhiteNits / 80.0f :
        desktopWhitePoint > 0 ? desktopWhitePoint : hdr ? 203.0f / 80.0f : 1.0f;
    ID3D12Resource* vectors = motion.Get(); bool reset = true;
    gpu.Time(GpuStage::InputMotion);
    if (s.estimateMotion && s.enabled && s.strength != 0 && s.compare != 1) {
        Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (flowInput) {
            DlssNrConstants down{}; down.Mode = DlssNrMode_Downsample; down.Width = modelWidth; down.Height = modelHeight;
            if (s.modelScale > 100) scaling->Run(2, input, flowInput.Get(), s.downscaler, true);
            else Dispatch(28, down, input, nullptr, nullptr, flowInput.Get(), nullptr);
            Barrier(gpu.commands.Get(), flowInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        // Estimate in model pixels so both passes receive correctly scaled vectors.
        reset = opticalFlow->Estimate(flowInput ? flowInput.Get() : input,
            resetHistory || lastTone != s.tone || lastHdr != hdr || lastSdr != sdr || lastWhitePoint != whitePoint);
        if (flowInput) Barrier(gpu.commands.Get(), flowInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        vectors = opticalFlow->Vectors();
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
        p.ReversibleMode = s.HdrMode(); // Passthrough skips HDR encoding; replace/composed semantics still match OptiScaler.
        if (sdr) {
            p.Mode = 11;
            Dispatch(49, p, input, nullptr, nullptr, sdrInput.Get(), nullptr);
            Barrier(c, sdrInput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        p.Mode = DlssNrMode_Encode; p.Passthrough = sdr ? 1u : 0u;
        Dispatch(0, p, sdr ? sdrInput.Get() : input, nullptr, nullptr, fullProxy ? fullProxy.Get() : proxy.Get(), original.Get());
        p.Width = modelWidth; p.Height = modelHeight;
        if (fullProxy) {
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            p.Mode = DlssNrMode_Downsample;
            if (s.modelScale > 100) scaling->Run(0, fullProxy.Get(), proxy.Get(), s.downscaler, true);
            else Dispatch(21, p, fullProxy.Get(), nullptr, nullptr, proxy.Get(), nullptr);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        Barrier(c, proxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(c, original.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Time(GpuStage::Encode, true);
        gpu.Time(GpuStage::NR);
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
        gpu.Time(GpuStage::NR, true);
        ID3D12Resource* modelAnswer = answer.Get();
        ID3D12Resource* resolveProxy = proxy.Get();
        const bool halo = s.haloSuppression > 0 || s.haloBrightSuppression > 0;
        if (halo) {
            if (!filteredAnswer) filteredAnswer = gpu.Texture(modelWidth, modelHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
            auto filter = p; filter.Mode = 18;
            filter.HaloSuppression = std::clamp(s.haloSuppression, 0.f, 1.f);
            filter.HaloBrightSuppression = std::clamp(s.haloBrightSuppression, 0.f, 1.f);
            filter.HaloRadius = std::clamp(s.haloRadius, 1.f, 4.f);
            Dispatch(91, filter, proxy.Get(), answer.Get(), nullptr, filteredAnswer.Get(), nullptr);
            Barrier(c, filteredAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            modelAnswer = filteredAnswer.Get();
        }
        const bool edgeAware = s.transfer == 2 && resizeInput;
        if (nativeAnswer && !resizeInput) {
            scaling->Run(1, modelAnswer, nativeAnswer.Get(), s.downscaler, false);
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            modelAnswer = nativeAnswer.Get(); resolveProxy = fullProxy.Get();
        }
        if (resizeInput && !edgeAware) {
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
        if (s.debugView != 0) { modelAnswer = answer.Get(); resolveProxy = proxy.Get(); p.Transfer = 1; }
        Dispatch(7, p, resolveProxy, modelAnswer, original.Get(), sdr ? sdrOutput.Get() : output.Get(), nullptr);
        if (sdr) {
            Barrier(c, sdrOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            p.Mode = 12;
            Dispatch(56, p, sdrOutput.Get(), input, sdrInput.Get(), output.Get(), nullptr);
            Barrier(c, sdrOutput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, sdrInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (privateDlss) {
            Barrier(c, enlarged.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, boundedDlss.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        if (nativeAnswer && !edgeAware) {
            Barrier(c, nativeAnswer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(c, fullProxy.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        for (auto* r : {proxy.Get(), original.Get(), answer.Get()})
            Barrier(c, r, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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
