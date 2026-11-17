#include "Gpu.h"
#include "Runtime.h"
#include "ModelKernelTrace.h"
#include "nvfp4/DlssNrNative.h"
#include "nvfp4/NrFfnV2.h"
#include "nvfp4/NrSwin4.h"
#include "nvfp4/NrNet4.h"
#include "ModelOverlap.h"
#include "ModelCapture.h"
#include "ModelVitSequence.h"
#include "ModelProbePrepare.h"
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Research harness: feed model-space RGBA16F directly to NR, without the
// standalone's colour encoding, residual composition, scaling or motion search.
// Output is the first reset evaluation; reset probes also save a final parity sample.
namespace {
using namespace nr;
struct Transfer {
    Gpu& gpu;
    ID3D12Resource* texture;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    ComPtr<ID3D12Resource> buffer;
    size_t rowBytes;
    UINT height;
    bool upload;
    Transfer(Gpu& g, ID3D12Resource* t, size_t pixelBytes, bool up)
        : gpu(g), texture(t), upload(up) {
        const auto desc = t->GetDesc(); UINT64 bytes = 0;
        height = desc.Height; rowBytes = size_t(desc.Width) * pixelBytes;
        gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        buffer = gpu.Buffer(bytes, up ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK);
    }
    void Copy() {
        D3D12_TEXTURE_COPY_LOCATION tex{}, buf{};
        tex.pResource = texture; tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        buf.pResource = buffer.Get(); buf.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        buf.PlacedFootprint = footprint;
        const auto state = upload ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        Barrier(gpu.commands.Get(), texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, state);
        gpu.commands->CopyTextureRegion(upload ? &tex : &buf, 0, 0, 0, upload ? &buf : &tex, nullptr);
        Barrier(gpu.commands.Get(), texture, state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    void Upload(const std::vector<char>& data) {
        if (data.size() != rowBytes * height) throw std::runtime_error("Input byte count does not match extent.");
        void* mapped = nullptr; Check(buffer->Map(0, nullptr, &mapped), "Map input");
        for (UINT y = 0; y < height; ++y)
            memcpy(static_cast<BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
                   data.data() + size_t(y) * rowBytes, rowBytes);
        buffer->Unmap(0, nullptr); Copy();
    }
    void Save(const std::filesystem::path& path) {
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot create output file.");
        void* mapped = nullptr; Check(buffer->Map(0, nullptr, &mapped), "Map output");
        for (UINT y = 0; y < height; ++y)
            file.write(reinterpret_cast<char*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
                       std::streamsize(rowBytes));
        buffer->Unmap(0, nullptr);
        if (!file) throw std::runtime_error("Output write failed.");
    }
};
void Complete(Gpu& gpu) {
    gpu.Submit(); if (!gpu.Drain(30000)) throw std::runtime_error("Model probe GPU timeout.");
    ComPtr<ID3D12InfoQueue> diagnostics;
    if (SUCCEEDED(gpu.device.As(&diagnostics))) {
        bool failed = false;
        for (UINT64 i = 0; i < diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
            SIZE_T bytes = 0; diagnostics->GetMessage(i, nullptr, &bytes);
            std::vector<BYTE> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            Check(diagnostics->GetMessage(i, message, &bytes), "Read validation message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::cerr << message->pDescription << '\n'; failed = true;
            }
        }
        diagnostics->ClearStoredMessages();
        if (failed) throw std::runtime_error("D3D12 validation failed.");
    }
}
// Alternate one pass, negative ViT feedback and two complete passes on the same
// DIRECT queue/device. This reduces cross-process GPU-load drift in cost tests.
int VitCost(int argc, wchar_t** argv) {
    if (argc < 7 || argc > 8) throw std::runtime_error("Usage: NrModelProbe --vit-cost <runtime> <input.rgba16f> <width> <height> <output-prefix> [alpha=-1]");
    const UINT width = std::stoul(argv[4]), height = std::stoul(argv[5]);
    if (!width || !height || width > 8192 || height > 8192) throw std::runtime_error("Invalid cost-probe extent");
    const float alpha = argc > 7 ? std::stof(argv[7]) : -1;
    const std::filesystem::path prefix = argv[6];
    if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
    const auto path = [&](const std::wstring& suffix) { return std::filesystem::path(prefix.wstring() + suffix); };
    std::ifstream file(std::filesystem::path(argv[3]), std::ios::binary | std::ios::ate);
    const size_t bytes = size_t(width) * height * 8;
    if (!file || size_t(file.tellg()) != bytes) throw std::runtime_error("Expected tightly packed RGBA16F cost input");
    std::vector<char> pixels(bytes); file.seekg(0); file.read(pixels.data(), std::streamsize(bytes));
    if (!file) throw std::runtime_error("Cannot read cost input");
    ModelKernelTrace::Install();
    Gpu gpu; Runtime runtime(gpu.device.Get(), std::filesystem::absolute(argv[2]));
    DlssNr::ModelSettings settings; settings.preset = 1; settings.localTone = 1;
    std::array<std::optional<DlssNr::ModelSettings>, Settings::MaxPasses - 1> overrides{}; overrides[0] = settings;
    auto colour = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    auto depth = gpu.Texture(width, height, DXGI_FORMAT_R32_FLOAT);
    auto motion = gpu.Texture(width, height, DXGI_FORMAT_R16G16_FLOAT);
    auto intermediate = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    auto secondInput = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON);
    auto output = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    ModelProbePrepare prepare(gpu);
    Transfer inputTransfer(gpu, colour.Get(), 8, true), depthTransfer(gpu, depth.Get(), 4, true), motionTransfer(gpu, motion.Get(), 4, true),
        firstReadback(gpu, intermediate.Get(), 8, false), secondReadback(gpu, output.Get(), 8, false);
    gpu.Begin(); Runtime::CheckResult(runtime.Create(gpu.commands.Get(), width, height, settings, 2, overrides), "Create cost features"); Complete(gpu);
    gpu.Begin(); inputTransfer.Upload(pixels); depthTransfer.Upload(std::vector<char>(size_t(width) * height * 4));
    motionTransfer.Upload(std::vector<char>(size_t(width) * height * 4)); Complete(gpu);
    const char* names[]{"one", "vit", "two"}; const wchar_t* wideNames[]{L"one", L"vit", L"two"};
    std::array<std::vector<double>, 3> times, cpuTimes;
    std::ofstream csv(path(L".csv")); csv << "frame,mode,warmup,gpu_ms,cpu_ms\n" << std::setprecision(12);
    for (unsigned frame = 0; frame < 69; ++frame) {
        const unsigned mode = (frame / 3) % 2 ? 2 - frame % 3 : frame % 3;
        gpu.Begin(); gpu.BeginTimings();
        Barrier(gpu.commands.Get(), colour.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Time(GpuStage::NR);
        if (mode == 1) ModelKernelTrace::BeginVitLoop(gpu.commands.Get(), gpu.device.Get(), alpha);
        const double start = ClockMs();
        try {
            runtime.Evaluate(gpu.commands.Get(), colour.Get(), depth.Get(), motion.Get(), intermediate.Get(), width, height, true, 0);
            if (mode == 1) ModelKernelTrace::EndVitLoop();
            if (mode == 2) {
                Barrier(gpu.commands.Get(), intermediate.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
                Barrier(gpu.commands.Get(), colour.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
                prepare.Run(gpu.commands.Get(), intermediate.Get(), colour.Get(), secondInput.Get());
                Barrier(gpu.commands.Get(), secondInput.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                runtime.Evaluate(gpu.commands.Get(), secondInput.Get(), depth.Get(), motion.Get(), output.Get(), width, height, true, 1);
                Barrier(gpu.commands.Get(), secondInput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
                Barrier(gpu.commands.Get(), intermediate.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                Barrier(gpu.commands.Get(), colour.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
        } catch (...) { ModelKernelTrace::CancelVitLoop(); throw; }
        const double elapsed = ClockMs() - start;
        gpu.Time(GpuStage::NR, true);
        Barrier(gpu.commands.Get(), colour.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.ResolveTimings();
        auto& readback = mode == 2 ? secondReadback : firstReadback;
        const bool capture = frame < 3 || frame >= 66;
        if (capture) readback.Copy(); Complete(gpu);
        std::array<double, unsigned(GpuStage::Count)> values{};
        if (!gpu.TakeTimings(values)) throw std::runtime_error("Cost timestamps unavailable");
        const double ms = values[unsigned(GpuStage::NR)];
        csv << frame << ',' << names[mode] << ',' << (frame < 9) << ',' << ms << ',' << elapsed << '\n';
        if (frame >= 9) { times[mode].push_back(ms); cpuTimes[mode].push_back(elapsed); }
        if (capture) readback.Save(path(std::wstring(L".") + wideNames[mode] + (frame >= 66 ? L".last.rgba16f" : L".rgba16f")));
    }
    std::ofstream json(path(L".json"));
    json << std::setprecision(12) << "{\"schema\":2,\"experiment\":\"interleaved-cost\",\"width\":" << width << ",\"height\":" << height
         << ",\"alpha\":" << alpha << ",\"reset\":true,\"tone_both\":1,\"interpass\":\"clamp-original-alpha\",\"queue\":\"DIRECT\",\"modes\":[";
    for (unsigned mode = 0; mode < 3; ++mode) {
        if (times[mode].size() != 20) throw std::runtime_error("Incomplete interleaved cost sample");
        auto sorted = times[mode]; std::sort(sorted.begin(), sorted.end());
        const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        const double median = (sorted[9] + sorted[10]) / 2;
        const double p10 = sorted[1] + .9 * (sorted[2] - sorted[1]);
        if (mode) json << ',';
        json << "{\"name\":\"" << names[mode] << "\",\"samples\":20,\"gpu_mean_ms\":" << mean << ",\"gpu_median_ms\":" << median
             << ",\"gpu_minimum_ms\":" << sorted.front() << ",\"gpu_p10_ms\":" << p10
             << ",\"cpu_mean_ms\":" << std::accumulate(cpuTimes[mode].begin(), cpuTimes[mode].end(), 0.0) / cpuTimes[mode].size() << '}';
        std::cout << std::format("Interleaved {} {}x{}: {:.4f} ms minimum, {:.4f} ms p10, {:.4f} ms mean GPU; 20 samples.\n", names[mode], width, height, sorted.front(), p10, mean);
    }
    json << "],\"percentile_method\":\"linear interpolation at p*(N-1)\",\"note\":\"One, ViT repeat and two complete passes interleaved on one DIRECT queue; order reverses every triplet; nine warmup evaluations. Two-pass timing includes clamped interpass preparation with original-frame alpha. Native repeat includes conservative replay barriers, snapshot, repacks and mixing. Capture/readback excluded.\"}\n";
    if (!csv || !json) throw std::runtime_error("Cannot write interleaved cost results");
    runtime.Release(); return 0;
}
}
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc > 1 && std::wstring_view(argv[1]) == L"--capture-sequence") return RunModelCapture(argc, argv);
        if (argc > 1 && std::wstring_view(argv[1]) == L"--vit-sequence") return RunModelVitSequence(argc, argv);
        if (argc > 1 && std::wstring_view(argv[1]) == L"--vit-cost") return VitCost(argc, argv);
        if (argc > 1 && std::wstring_view(argv[1]) == L"--overlap") {
            ModelKernelTrace::Install();
            return RunModelOverlap(argc, argv);
        }
        unsigned preset = 1, style = 0, hybrid = 0;
        bool explicitSettings = false;
        if (argc > 3 && std::wstring_view(argv[1]) == L"--hybrid") {
            // The recovered v0.7.2 NVFP4 feed-forward replacement: 2 = original hybrid, 4 = its later candidate.
            hybrid = std::stoul(argv[2]);
            // 5 = the NVFP4 v2 feed-forward kernels (nvfp4/NrFfnV2.cpp); assets from NR_FFN_V2_ASSETS.
            // 6 = the NVFP4 fused Swin kernels (nvfp4/NrSwin4.cpp), Model A only: assets from NR_SWIN4_ASSETS, widths from
            // NR_SWIN4_WIDTHS (bits: 1 = 32, 2 = 64, 4 = 128, 8 = 256; default 15), NR_SWIN4_LAYERS (hex mask of the 44
            // blocks, default all), NR_SWIN4_TWO (second 4-bit terms at width 32, default 7). With NR_FFN_V2_ASSETS set
            // as well, the feed-forward kernels of 5 run too. With NR_NET4 set (NrNet4::Parse items, for example
            // "split=all vit=all prepost=3") the NVFP4 kernels of the rest of the network run as well (nvfp4/NrNet4.cpp):
            // assets from NR_NET4_ASSETS, the folder that holds split4, vit4 and prepost4.
            if (hybrid != 2 && hybrid != 4 && hybrid != 5 && hybrid != 6)
                throw std::runtime_error("--hybrid requires 2 (v0.7.2 hybrid), 4 (candidate), 5 (NVFP4 v2 feed-forward) or 6 (NVFP4 fused blocks)");
            argc -= 2; argv += 2;
        }
        while (argc > 1 && (std::wstring_view(argv[1]) == L"--preset" || std::wstring_view(argv[1]) == L"--style")) {
            const bool isStyle = std::wstring_view(argv[1]) == L"--style";
            if (argc < 4 || std::wstring_view(argv[2]).size() != 1 || argv[2][0] < L'0' || argv[2][0] > (isStyle ? L'2' : L'3'))
                throw std::runtime_error(isStyle ? "--style requires 0 (Standard), 1 (Natural), or 2 (Cinematic)" :
                    "--preset requires 0 (automatic), 1 (A), 2 (B), or 3 before the normal probe arguments");
            (isStyle ? style : preset) = unsigned(argv[2][0] - L'0');
            explicitSettings = true; argc -= 2; argv += 2;
        }
        const bool vitRepeat = argc > 1 && std::wstring_view(argv[1]) == L"--vit-repeat";
        if (explicitSettings && vitRepeat) throw std::runtime_error("Preset/style selection is for unmodified forward passes only");
        if (vitRepeat) { --argc; ++argv; }
        if (argc < 6 || argc > 15)
            throw std::runtime_error("Usage: NrModelProbe <runtime.dll> <input.rgba16f> <width> <height> <output.rgba16f> [intensity=1 autoMask=1 structure=1 tone=1 trace=off|census|timed body-loops=1|2|3 bookend.fatbin bookend-mode=control|export|import|anchor|bridge|affine bridge-weights.bin]");
        const UINT width = std::stoul(argv[3]), height = std::stoul(argv[4]);
        if (!width || !height || width > 8192 || height > 8192) throw std::runtime_error("Unsupported extent.");
        DlssNr::ModelSettings settings; settings.preset = preset; settings.style = style; settings.localTone = 1;
        if (argc > 6) settings.intensity = std::stof(argv[6]);
        if (argc > 7) settings.autoMask = std::stoul(argv[7]) != 0;
        if (argc > 8) settings.localStructure = std::stof(argv[8]);
        if (argc > 9) settings.localTone = std::stof(argv[9]);
        const std::wstring trace = argc > 10 ? argv[10] : L"off";
        if (trace != L"off" && trace != L"census" && trace != L"timed") throw std::runtime_error("Unknown trace mode.");
        const float vitAlpha = vitRepeat && argc > 11 ? std::stof(argv[11]) : -1;
        const unsigned bodyLoops = !vitRepeat && argc > 11 ? std::stoul(argv[11]) : 1;
        if (vitRepeat && (argc > 12 || trace != L"off")) throw std::runtime_error("ViT repetition uses trace off, followed by alpha (default -1), with no bookend arguments");
        if (bodyLoops < 1 || bodyLoops > 3) throw std::runtime_error("Choose one to three complete body traversals.");
        if (bodyLoops > 1 && trace != L"off") throw std::runtime_error("Run kernel tracing and feature feedback separately.");
        const std::filesystem::path bookendControl = argc > 12 ? argv[12] : L"";
        if (explicitSettings && (bodyLoops != 1 || !bookendControl.empty()))
            throw std::runtime_error("Preset/style selection is for unmodified forward passes only");
        const std::wstring bookendMode = argc > 13 ? argv[13] : L"control";
        const bool bridge = bookendMode == L"bridge" || bookendMode == L"affine";
        if (bookendMode != L"control" && bookendMode != L"export" && bookendMode != L"import" && bookendMode != L"anchor" && !bridge)
            throw std::runtime_error("Unknown bookend mode");
        if (bridge && (argc != 15 || trace != L"off" || settings.intensity != 1 || settings.localTone != 1 ||
            settings.localStructure != 1 || !settings.autoMask)) throw std::runtime_error("Bridge requires weights, no trace, reset default controls and intensity one");
        if (!bookendControl.empty() && bodyLoops > 1) throw std::runtime_error("Run JIT bookend control and body feedback separately.");
        if (hybrid) {
            if (bodyLoops > 1 || !bookendControl.empty() || vitRepeat) throw std::runtime_error("Run the NVFP4 hybrid on unmodified forward passes only");
            if (hybrid == 6) {
                NrSwin4::Options options;
                const wchar_t* assets = _wgetenv(L"NR_SWIN4_ASSETS");
                if (!assets) throw std::runtime_error("Set NR_SWIN4_ASSETS to the folder with swin4.txt, swin4.bin and the swin4 kernels");
                options.assets = assets;
                options.byOrder = true;         // the probe evaluates one model, chosen with --preset
                options.widths = 15;
                if (const wchar_t* widths = _wgetenv(L"NR_SWIN4_WIDTHS")) options.widths = std::stoul(widths);
                if (const wchar_t* layers = _wgetenv(L"NR_SWIN4_LAYERS")) options.layers = std::stoull(layers, nullptr, 16);
                if (const wchar_t* two = _wgetenv(L"NR_SWIN4_TWO")) options.twoTerms = std::stoul(two);
                if (const wchar_t* trained = _wgetenv(L"NR_SWIN4_TRAINED")) options.set = *trained && *trained != L'0' ? 1 : 0;
                if (const wchar_t* set = _wgetenv(L"NR_SWIN4_SET")) options.set = std::stoul(set);
                if (preset != 1) throw std::runtime_error("The NVFP4 fused blocks are Model A's: use --preset 1");
                NrSwin4::Enable(options);
                if (const wchar_t* items = _wgetenv(L"NR_NET4")) {
                    NrNet4::Options net;
                    std::string text;
                    for (const wchar_t* c = items; *c; ++c) text += *c < 128 ? char(*c) : '?';
                    NrNet4::Parse(text, net);
                    const wchar_t* folder = _wgetenv(L"NR_NET4_ASSETS");
                    if (!folder) throw std::runtime_error("Set NR_NET4_ASSETS to the folder that holds split4, vit4 and prepost4");
                    net.assets = folder;
                    net.byOrder = true;
                    if (net.Any()) NrNet4::Enable(net);
                }
            }
            if (hybrid == 5 || (hybrid == 6 && _wgetenv(L"NR_FFN_V2_ASSETS"))) {
                NrFfnV2::Options options;
                const wchar_t* assets = _wgetenv(L"NR_FFN_V2_ASSETS");
                if (!assets) throw std::runtime_error("Set NR_FFN_V2_ASSETS to the folder with ffn_v2.cubin, ffn_v2.bin and ffn_v2.txt");
                options.assets = assets;
                if (const wchar_t* shape = _wgetenv(L"NR_FFN_V2_SHAPE"))
                    if (swscanf_s(shape, L"%u,%u,%u,%u", &options.expandGroups, &options.expandHalves, &options.contractGroups, &options.contractHalves) != 4)
                        throw std::runtime_error("NR_FFN_V2_SHAPE is expandGroups,expandHalves,contractGroups,contractHalves");
                if (const wchar_t* barriers = _wgetenv(L"NR_FFN_V2_BARRIERS")) options.barriers = std::stoul(barriers);
                if (const wchar_t* chained = _wgetenv(L"NR_FFN_V2_CHAINED")) options.chained = std::stoul(chained) != 0;
                if (const wchar_t* stages = _wgetenv(L"NR_FFN_V2_STAGES")) options.stages = std::stoul(stages);
                if (const wchar_t* steps = _wgetenv(L"NR_FFN_V2_KSTEPS")) options.stepsPerStage = std::stoul(steps);
                if (const wchar_t* trained = _wgetenv(L"NR_FFN_V2_TRAINED")) options.trained = *trained && *trained != L'0';  // ffn_v2_trained.bin
                NrFfnV2::Enable(options);
            }
            // A numbered set (NR_SWIN4_SET >= 2): its bottleneck lines (net4_set<n>.txt) and feed-forward records as well.
            if (hybrid == 6) if (const wchar_t* set = _wgetenv(L"NR_SWIN4_SET")) { NrNet4::UseSet(std::stoul(set)); NrFfnV2::UseSet(std::stoul(set)); }
            if (hybrid == 5 || hybrid == 6) {
                ModelKernelTrace::SetDriverLayer([](unsigned id, void* original) { return NrNet4::WrapNvapi(id, NrSwin4::WrapNvapi(id, NrFfnV2::WrapNvapi(id, original))); });
            } else {
                ModelKernelTrace::SetDriverLayer(&DlssNrNative::WrapNvapi);
                DlssNrNative::SetPrecision(hybrid);
            }
        }
        if (hybrid || trace != L"off" || bodyLoops > 1 || !bookendControl.empty() || vitRepeat) ModelKernelTrace::Install();
        std::ifstream file(std::filesystem::path(argv[2]), std::ios::binary | std::ios::ate);
        const size_t bytes = size_t(width) * height * 8;
        if (!file || size_t(file.tellg()) != bytes) throw std::runtime_error("Expected tightly packed RGBA16F input.");
        std::vector<char> pixels(bytes); file.seekg(0); file.read(pixels.data(), std::streamsize(bytes));
        if (!file) throw std::runtime_error("Input read failed.");
        Gpu gpu; Runtime runtime(gpu.device.Get(), std::filesystem::absolute(argv[1]));
        auto colour = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
        auto depth = gpu.Texture(width, height, DXGI_FORMAT_R32_FLOAT);
        auto motion = gpu.Texture(width, height, DXGI_FORMAT_R16G16_FLOAT);
        auto output = gpu.Texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT);
        Transfer inputTransfer(gpu, colour.Get(), 8, true), depthTransfer(gpu, depth.Get(), 4, true),
            motionTransfer(gpu, motion.Get(), 4, true), outputTransfer(gpu, output.Get(), 8, false);
        gpu.Begin();
        Runtime::CheckResult(runtime.Create(gpu.commands.Get(), width, height, settings), "Create probe model");
        Complete(gpu);
        if (!bookendControl.empty()) {
            if (bookendMode == L"control") ModelKernelTrace::LoadBookendControl(gpu.device.Get(), bookendControl);
            else ModelKernelTrace::LoadBookendFeatures(gpu.device.Get(), bookendControl, width, height, bookendMode == L"import");
            if (bookendMode == L"anchor" || bridge) ModelKernelTrace::ConfigureBookendBridge(gpu.device.Get(), pixels,
                argc > 14 ? argv[14] : L"", bookendMode == L"anchor" ? ModelKernelTrace::BookendBridge::AnchorControl :
                bookendMode == L"affine" ? ModelKernelTrace::BookendBridge::Affine : ModelKernelTrace::BookendBridge::ColourDelta);
        }
        gpu.Begin(); inputTransfer.Upload(pixels);
        depthTransfer.Upload(std::vector<char>(size_t(width) * height * 4));
        {
            // NR_PROBE_MOTION=1: a smooth, non-uniform, sub-pixel motion field instead of zero motion, so that with
            // NR_PROBE_RESET=0 the model reprojects its history through fractional taps (near the borders the
            // vectors point outside the frame). The same field for every kernel set, so outputs stay comparable.
            std::vector<char> field(size_t(width) * height * 4);
            wchar_t text[8]{};
            if (GetEnvironmentVariableW(L"NR_PROBE_MOTION", text, 8) && text[0] == L'1') {
                auto* half = reinterpret_cast<DirectX::PackedVector::HALF*>(field.data());
                for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
                    const float u = float(x) / float(width), v = float(y) / float(height);
                    half[(size_t(y) * width + x) * 2] = DirectX::PackedVector::XMConvertFloatToHalf(3.3f * std::sin(9.42478f * v) + .37f);
                    half[(size_t(y) * width + x) * 2 + 1] = DirectX::PackedVector::XMConvertFloatToHalf(2.6f * std::cos(12.56637f * u) - .21f);
                }
            }
            motionTransfer.Upload(field);
        }
        Complete(gpu);
        std::vector<double> timings, cpuTimes;
        // 5 warm-up and 20 timed evaluations unless the environment says otherwise (longer runs let the GPU's clocks settle)
        const auto countFromEnvironment = [](const wchar_t* name, unsigned fallback, unsigned least) {
            wchar_t text[16]{};
            const DWORD length = GetEnvironmentVariableW(name, text, 16);
            const unsigned value = length && length < 16 ? unsigned(std::wcstoul(text, nullptr, 10)) : fallback;
            return value >= least && value <= 100000 ? value : fallback;
        };
        const unsigned warmup = countFromEnvironment(L"NR_PROBE_WARMUP", 5, 1), timed = countFromEnvironment(L"NR_PROBE_TIMED", 20, 20);
        const unsigned lastFrame = warmup + timed;
        // NR_PROBE_RESET=0: only the first evaluation resets the model's history, as an application's frames do
        // (the default resets every evaluation, so each one is the same work on the same input).
        const bool resetEvery = countFromEnvironment(L"NR_PROBE_RESET", 1, 0) != 0;
        // The final validation/trace evaluation is excluded from the mean. Both
        // modes save that same frame so the tracing adapter can be checked.
        for (unsigned frame = 0; frame <= lastFrame; ++frame) {
            gpu.Begin(); gpu.BeginTimings();
            Barrier(gpu.commands.Get(), colour.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Time(GpuStage::NR);
            if (frame == lastFrame && trace != L"off") ModelKernelTrace::Begin(gpu.commands.Get(), gpu.queue.Get(), trace == L"timed");
            if (bodyLoops > 1) ModelKernelTrace::BeginLoop(gpu.commands.Get(), gpu.device.Get(), bodyLoops);
            if (vitRepeat) ModelKernelTrace::BeginVitLoop(gpu.commands.Get(), gpu.device.Get(), vitAlpha);
            const double cpuStart = ClockMs();
            try {
                runtime.Evaluate(gpu.commands.Get(), colour.Get(), depth.Get(), motion.Get(), output.Get(), width, height, resetEvery || !frame);
            } catch (...) {
                if (vitRepeat) {
                    try { ModelKernelTrace::EndVitLoop(); }
                    catch (const std::exception& detail) { std::cerr << detail.what() << '\n'; }
                    ModelKernelTrace::CancelVitLoop();
                }
                if (!bookendControl.empty()) {
                    try { ModelKernelTrace::CheckBookendControl(frame + 1); }
                    catch (const std::exception& detail) { std::cerr << detail.what() << '\n'; }
                }
                if (bodyLoops > 1) {
                    try { ModelKernelTrace::EndLoop(); }
                    catch (const std::exception& detail) { std::cerr << detail.what() << '\n'; }
                }
                ModelKernelTrace::CancelLoop(); throw;
            }
            const double cpuEnd = ClockMs();
            if (vitRepeat) {
                const auto report = ModelKernelTrace::EndVitLoop();
                if (!frame) std::cout << std::format("ViT feedback: blocks 31..38 repeated, alpha {}, {} replayed kernels including repacks, {} latent bytes; one encoder/decoder.\n",
                    vitAlpha, report.replayedKernels, report.latentBytes);
            }
            if (!bookendControl.empty()) ModelKernelTrace::CheckBookendControl(frame + 1);
            if (bodyLoops > 1) {
                const auto report = ModelKernelTrace::EndLoop();
                if (!frame) std::cout << std::format("Feature feedback: {} complete body traversals, {} additional body kernels plus a counter clear and latent copy per repeat, {} latent bytes; pre/post blocks once.\n",
                    report.bodies, report.replayedKernels, report.latentBytes);
            }
            if (frame == lastFrame && trace != L"off") ModelKernelTrace::End(gpu.commands.Get());
            gpu.Time(GpuStage::NR, true);
            Barrier(gpu.commands.Get(), colour.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            gpu.ResolveTimings();
            if (!frame || frame == lastFrame) outputTransfer.Copy();
            if (!frame && bookendMode != L"control" && bookendMode != L"anchor") ModelKernelTrace::CaptureBookendFeatures(gpu.commands.Get());
            Complete(gpu);
            std::array<double, unsigned(GpuStage::Count)> values{};
            if (!gpu.TakeTimings(values)) throw std::runtime_error("Model timestamps unavailable.");
            if (frame >= warmup && frame < lastFrame) {
                timings.push_back(values[unsigned(GpuStage::NR)]);
                cpuTimes.push_back(cpuEnd - cpuStart);
            }
            if (!frame) outputTransfer.Save(argv[5]);
            if (!frame && bookendMode != L"control" && bookendMode != L"anchor") ModelKernelTrace::SaveBookendFeatures(argv[5]);
            if (frame == lastFrame) outputTransfer.Save(std::filesystem::path(argv[5]).wstring() + L".last");
            if (frame == lastFrame && trace != L"off") {
                const auto report = ModelKernelTrace::Collect(std::filesystem::path(argv[5]).wstring() + L".trace");
                std::cout << std::format("Trace: {} launch calls, {} kernels, {:.4f} ms GPU, {:.4f} ms CPU Evaluate, {:.4f} ms inside native launch calls; per-launch timing {}.\n",
                    report.launchCalls, report.kernels, report.gpuMs, report.cpuEvaluateMs, report.cpuLaunchMs, report.perLaunchTiming);
            }
        }
        const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        std::ofstream samples(std::filesystem::path(std::filesystem::path(argv[5]).wstring() + L".timings.csv"));
        samples << "sample,preset,style,gpu_ms,cpu_evaluate_ms\n" << std::setprecision(12);
        for (size_t i = 0; i < timings.size(); ++i) samples << i << ',' << preset << ',' << style << ',' << timings[i] << ',' << cpuTimes[i] << '\n';
        if (!samples) throw std::runtime_error("Cannot write probe timing samples");
        std::sort(timings.begin(), timings.end());
        const auto percentile = [&](double p) {                    // linear interpolation at p * (N - 1)
            const double at = p * double(timings.size() - 1);
            const size_t below = size_t(at);
            return below + 1 < timings.size() ? timings[below] + (at - double(below)) * (timings[below + 1] - timings[below]) : timings.back();
        };
        const double p10 = percentile(0.10), p95 = percentile(0.95);
        std::cout << std::format("NR {}x{} preset {} style {}: minimum {:.4f} ms, mean {:.4f} ms, median {:.4f} ms, p10 {:.4f} ms, p95 {:.4f} ms; {} reset evaluations after {} warmup.\n",
            width, height, preset, style, timings.front(), mean, percentile(0.5), p10, p95, timed, warmup);
        std::cout << std::format("CPU Evaluate mean {:.4f} ms (command recording, not GPU execution).\n",
            std::accumulate(cpuTimes.begin(), cpuTimes.end(), 0.0) / cpuTimes.size());
        std::cout << "Saved first reset output as model-space RGBA16F; timing excludes transfers.\n";
        if (hybrid) std::cout << "NVFP4 hybrid " << hybrid << ": " << (hybrid == 5 || hybrid == 6 ? NrFfnV2::Status(true) : DlssNrNative::Status()) << '\n';
        if (hybrid == 6) std::cout << "NVFP4 fused blocks: " << NrSwin4::Status(true) << '\n';
        if (hybrid == 6) std::cout << "NVFP4 network: " << NrNet4::Status(true) << '\n';
        if (hybrid == 5 && NrFfnV2::Failed()) throw std::runtime_error("The NVFP4 feed-forward replacement failed during the run");
        if (!bookendControl.empty()) std::cout << "Native bookend probe: replaced exactly one pre/post pair in each of 26 evaluations.\n";
        runtime.Release();
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
