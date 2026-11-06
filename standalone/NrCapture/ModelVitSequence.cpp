#include "ModelVitSequence.h"
#include "Gpu.h"
#include "Runtime.h"
#include "ModelKernelTrace.h"
#include "ModelProbePrepare.h"
#include "../../external/nlohmann/json.hpp"
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
using namespace nr;
using Pixels = std::vector<char>;
using Json = nlohmann::json;

[[noreturn]] void InFlightFailure() {
    std::cerr << "Cannot establish GPU completion; terminating the ViT sequence probe before releasing submitted resources.\n";
    std::cerr.flush(); TerminateProcess(GetCurrentProcess(), 2); std::abort();
}
void Drain(Gpu& gpu) {
    try { if (!gpu.Drain(60000)) InFlightFailure(); }
    catch (...) { InFlightFailure(); }
}
struct DrainOnExit {
    Gpu& gpu;
    ~DrainOnExit() { ModelKernelTrace::CancelVitLoop(); Drain(gpu); }
};
void Complete(Gpu& gpu) {
    try { gpu.Submit(); } catch (...) { Drain(gpu); throw; }
    Drain(gpu);
    ComPtr<ID3D12InfoQueue> diagnostics;
    if (FAILED(gpu.device.As(&diagnostics))) return;
    bool failed = false;
    for (UINT64 i = 0; i < diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T bytes = 0; diagnostics->GetMessage(i, nullptr, &bytes);
        std::vector<BYTE> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        Check(diagnostics->GetMessage(i, message, &bytes), "Read ViT sequence validation message");
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            std::cerr << message->pDescription << '\n'; failed = true;
        }
    }
    diagnostics->ClearStoredMessages();
    if (failed) throw std::runtime_error("D3D12 validation failed in ViT sequence probe.");
}
Pixels Load(const std::filesystem::path& path, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || size_t(file.tellg()) != bytes)
        throw std::runtime_error("Expected tightly packed model-space RGBA16F: " + Narrow(path.wstring()));
    Pixels pixels(bytes); file.seekg(0); file.read(pixels.data(), std::streamsize(bytes));
    if (!file) throw std::runtime_error("Reading sequence frame failed.");
    return pixels;
}
void Save(const std::filesystem::path& path, const Pixels& pixels) {
    std::ofstream file(path, std::ios::binary);
    file.write(pixels.data(), std::streamsize(pixels.size())); file.close();
    if (!file) throw std::runtime_error("Writing sequence output failed.");
}
std::string Hash(const Pixels& pixels) {
    UINT64 hash = 14695981039346656037ull;
    for (const auto byte : pixels) { hash ^= static_cast<unsigned char>(byte); hash *= 1099511628211ull; }
    return std::format("{:016x}", hash);
}
void RequireFinite(const Pixels& pixels) {
    for (size_t offset = 0; offset < pixels.size(); offset += 2) {
        uint16_t value; memcpy(&value, pixels.data() + offset, 2);
        if ((value & 0x7c00) == 0x7c00) throw std::runtime_error("Non-finite FP16 sequence output.");
    }
}
Json Statistics(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("No measured sequence frames.");
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    const double position = .1 * (values.size() - 1);
    const size_t lo = size_t(position), hi = std::min(lo + 1, values.size() - 1);
    return {{"minimum_ms", values.front()}, {"p10_ms", values[lo] + (values[hi] - values[lo]) * (position - lo)},
        {"mean_ms", mean}, {"median_ms", .5 * (values[(values.size()-1)/2] + values[values.size()/2])}};
}

// All copies use COMMON handoff, and callers fence before touching mapped data
// or reusing the upload allocation. Transfers are outside model timestamps.
struct Transfer {
    ID3D12Resource* texture;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    ComPtr<ID3D12Resource> buffer;
    size_t rowBytes;
    UINT height;
    bool upload;
    Transfer(Gpu& gpu, ID3D12Resource* texture, size_t pixelBytes, bool upload)
        : texture(texture), upload(upload) {
        const auto desc = texture->GetDesc(); UINT64 bytes = 0;
        rowBytes = size_t(desc.Width) * pixelBytes; height = desc.Height;
        gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        buffer = gpu.Buffer(bytes, upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK);
    }
    void Copy(ID3D12GraphicsCommandList* commands) {
        D3D12_TEXTURE_COPY_LOCATION tex{}, buf{};
        tex.pResource = texture; tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        buf.pResource = buffer.Get(); buf.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; buf.PlacedFootprint = footprint;
        const auto state = upload ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        Barrier(commands, texture, D3D12_RESOURCE_STATE_COMMON, state);
        commands->CopyTextureRegion(upload ? &tex : &buf, 0, 0, 0, upload ? &buf : &tex, nullptr);
        Barrier(commands, texture, state, D3D12_RESOURCE_STATE_COMMON);
    }
    void Upload(ID3D12GraphicsCommandList* commands, const Pixels& pixels) {
        if (pixels.size() != rowBytes * height) throw std::runtime_error("Wrong sequence upload size.");
        void* mapped = nullptr; const D3D12_RANGE none{0, 0};
        Check(buffer->Map(0, &none, &mapped), "Map sequence upload");
        for (UINT y = 0; y < height; ++y)
            memcpy(static_cast<char*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
                pixels.data() + size_t(y) * rowBytes, rowBytes);
        buffer->Unmap(0, nullptr); Copy(commands);
    }
    Pixels Read() {
        Pixels pixels(rowBytes * height); void* mapped = nullptr;
        const D3D12_RANGE range{SIZE_T(footprint.Offset), SIZE_T(footprint.Offset + UINT64(footprint.Footprint.RowPitch) * (height - 1) + rowBytes)};
        Check(buffer->Map(0, &range, &mapped), "Map sequence readback");
        for (UINT y = 0; y < height; ++y)
            memcpy(pixels.data() + size_t(y) * rowBytes,
                static_cast<const char*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, rowBytes);
        const D3D12_RANGE none{0, 0}; buffer->Unmap(0, &none); return pixels;
    }
};
}

int RunModelVitSequence(int argc, wchar_t** argv) {
    if (argc < 7)
        throw std::runtime_error("Usage: NrModelProbe --vit-sequence <runtime> <input.rgba16f|directory> <width> <height> <output-prefix> [frames=48] [warmups=6] [alpha=-1] [--alpha value]");
    const UINT width = std::stoul(argv[4]), height = std::stoul(argv[5]);
    unsigned frames = 48, warmups = 6, positional = 0;
    float alpha = -1; bool alphaSpecified = false;
    for (int i = 7; i < argc; ++i) {
        if (std::wstring_view(argv[i]) == L"--alpha") {
            if (alphaSpecified || ++i == argc) throw std::runtime_error("Supply alpha once, followed by a number.");
            alpha = std::stof(argv[i]); alphaSpecified = true;
        } else if (positional == 0) { frames = std::stoul(argv[i]); ++positional; }
        else if (positional == 1) { warmups = std::stoul(argv[i]); ++positional; }
        else if (positional == 2 && !alphaSpecified) { alpha = std::stof(argv[i]); alphaSpecified = true; ++positional; }
        else throw std::runtime_error("Unexpected ViT sequence argument.");
    }
    if (width < 8 || height < 4 || width > 8192 || height > 8192 || !frames || frames > 240 || warmups >= frames ||
        !std::isfinite(alpha) || alpha < -1 || alpha > 1)
        throw std::runtime_error("Invalid sequence extent, frame count, warmup count or alpha.");
    const std::filesystem::path inputPath = argv[3], prefix = argv[6];
    const bool directory = std::filesystem::is_directory(inputPath);
    const size_t pixelCount = size_t(width) * height, bytes = pixelCount * 8;
    const auto path = [&](const std::wstring& suffix) { return std::filesystem::path(prefix.wstring() + suffix); };
    const auto outputPath = [&](const wchar_t* mode, unsigned frame) {
        return path(std::format(L".{}.frame-{:04}.rgba16f", mode, frame));
    };
    const auto inputFrame = [&](unsigned frame) { return inputPath / std::format(L"frame-{:04}.rgba16f", frame); };
    if (directory) for (unsigned frame = 0; frame < frames; ++frame)
        if (!std::filesystem::is_regular_file(inputFrame(frame)) || std::filesystem::file_size(inputFrame(frame)) != bytes)
            throw std::runtime_error("Missing/incomplete sequence frame: " + Narrow(inputFrame(frame).wstring()));
    const Pixels base = directory ? Pixels{} : Load(inputPath, bytes);
    const auto pixelsFor = [&](unsigned frame) {
        if (directory) return Load(inputFrame(frame), bytes);
        const UINT dx = (frame * 8) % width, dy = (frame * 4) % height;
        Pixels result(bytes);
        for (UINT y = 0; y < height; ++y) {
            const char* source = base.data() + size_t((y + height - dy) % height) * width * 8;
            char* target = result.data() + size_t(y) * width * 8;
            memcpy(target, source + size_t(width - dx) * 8, size_t(dx) * 8);
            memcpy(target + size_t(dx) * 8, source, size_t(width - dx) * 8);
        }
        return result;
    };
    Pixels zeroGuides(pixelCount * 4), movingGuides = zeroGuides;
    if (!directory) for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) {
        const uint16_t motion[]{DirectX::PackedVector::XMConvertFloatToHalf(x < 8 ? float(width) - 8 : -8),
            DirectX::PackedVector::XMConvertFloatToHalf(y < 4 ? float(height) - 4 : -4)};
        memcpy(movingGuides.data() + (size_t(y) * width + x) * 4, motion, 4);
    }
    const std::string guideMode = directory ? "zero-depth-zero-motion-unavailable" : "zero-depth-known-backward-wrapped-translation";
    const std::string zeroMotionHash = Hash(zeroGuides), movingMotionHash = Hash(movingGuides);
    if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
    Json report{{"schema", 1}, {"experiment", "vit-temporal-sequence"}, {"status", "running"},
        {"width", width}, {"height", height}, {"frames_per_mode", frames}, {"warmup_frames_per_mode", warmups},
        {"measured_frames_per_mode", frames - warmups}, {"alpha", alpha}, {"mode_order", {"one", "vit", "two"}},
        {"source", directory ? "capture-directory" : "synthetic-progressive-pan"}, {"source_path", Narrow(std::filesystem::absolute(inputPath).wstring())},
        {"guide_mode", guideMode}, {"pan_step_x", directory ? Json(nullptr) : Json(8)}, {"pan_step_y", directory ? Json(nullptr) : Json(4)},
        {"reset_policy", "frame zero only; new features before every mode"}, {"maximum_live_nr_features", 2},
        {"tone_first", 1}, {"tone_second", 1}, {"two_pass_handoff", "source-clamp-original-alpha; shared ModelProbePrepare shader at equal extent"},
        {"output_format", "little-endian tightly packed model-space RGBA16F"},
        {"hash_algorithm", "FNV-1a 64-bit; alpha-zero parity uses full bytes"},
        {"native_validation", "Every ViT frame retains the audited DLL hash, 156 original-launch ABI/sequence and counter-isolation guards; exactly 42 replayed kernels required before submission"},
        {"alpha_zero_parity", alpha == 0 ? "pending-all-frames" : "not-applicable"},
        {"percentile_method", "linear interpolation at p*(N-1), warmup frames excluded"},
        {"note", "One complete chronological clip per mode, sequential DIRECT execution; timings exclude feature creation, transfers, validation, hashing and disk writes. GPU total includes interpass preparation and state transitions. ViT includes conservative replay barriers. CPU recording includes guards; CPU Evaluate is reported separately. Capture directories have unavailable geometry guides and use zero depth/MV, not a claim of zero scene motion. No standalone spatial reconstruction, residual composition or SR. Separate-mode timing may be affected by clock or background-load drift."},
        {"modes", Json::object()}};
    const auto saveReport = [&] {
        std::ofstream file(path(L".json")); file << report.dump(2) << '\n'; file.close();
        if (!file) throw std::runtime_error("Writing sequence metadata failed.");
    };
    saveReport();
    std::ofstream csv(path(L".csv"));
    csv << std::setprecision(12) << "mode,frame,warmup,reset,alpha,gpu_total_ms,gpu_first_nr_ms,gpu_handoff_ms,gpu_second_nr_ms,cpu_record_ms,cpu_evaluate_ms,guard_replayed_kernels,latent_bytes,input_hash,motion_hash,output_hash,alpha_zero_exact\n";
    if (!csv) throw std::runtime_error("Cannot create sequence CSV.");

    ModelKernelTrace::Install();
    Gpu gpu; Runtime runtime(gpu.device.Get(), std::filesystem::absolute(argv[2]));
    const auto texture = [&](DXGI_FORMAT format) { return gpu.Texture(width, height, format, D3D12_RESOURCE_STATE_COMMON); };
    auto colour = texture(DXGI_FORMAT_R16G16B16A16_FLOAT), intermediate = texture(DXGI_FORMAT_R16G16B16A16_FLOAT),
        prepared = texture(DXGI_FORMAT_R16G16B16A16_FLOAT), output = texture(DXGI_FORMAT_R16G16B16A16_FLOAT);
    auto depth = texture(DXGI_FORMAT_R32_FLOAT), motion = texture(DXGI_FORMAT_R16G16_FLOAT);
    Transfer inputUpload(gpu, colour.Get(), 8, true), depthUpload(gpu, depth.Get(), 4, true), motionUpload(gpu, motion.Get(), 4, true);
    Transfer firstReadback(gpu, intermediate.Get(), 8, false), secondReadback(gpu, output.Get(), 8, false);
    ModelProbePrepare prepare(gpu);
    DrainOnExit drain{gpu};
    DlssNr::ModelSettings settings; settings.preset = 1; settings.localTone = 1;
    std::array<std::optional<DlssNr::ModelSettings>, Settings::MaxPasses - 1> overrides{}; overrides[0] = settings;
    std::vector<std::string> inputHashes(frames);
    const wchar_t* modeNames[]{L"one", L"vit", L"two"};
    for (unsigned mode = 0; mode < 3; ++mode) {
        const std::string name = Narrow(modeNames[mode]);
        // The prior clip is fully drained. Release all its parameters/feature
        // arenas before creating the next mode, which owns at most two features.
        runtime.Release(); gpu.Begin();
        Runtime::CheckResult(runtime.Create(gpu.commands.Get(), width, height, settings, mode == 2 ? 2u : 1u, overrides), "Create sequence features");
        Complete(gpu);
        gpu.Begin(); depthUpload.Upload(gpu.commands.Get(), zeroGuides); Complete(gpu);
        std::vector<double> totalTimes, firstTimes, handoffTimes, secondTimes, recordTimes, evaluateTimes;
        unsigned checkedZeroFrames = 0;
        std::cout << std::format("ViT sequence {}: {} frames at {}x{}, alpha {}, guides {}.\n", name, frames, width, height, alpha, guideMode) << std::flush;
        for (unsigned frame = 0; frame < frames; ++frame) {
            const auto pixels = pixelsFor(frame); const auto inputHash = Hash(pixels);
            if (!mode) inputHashes[frame] = inputHash;
            else if (inputHash != inputHashes[frame]) throw std::runtime_error("Input changed between sequence modes.");
            const bool moving = frame && !directory;
            gpu.Begin(); inputUpload.Upload(gpu.commands.Get(), pixels);
            motionUpload.Upload(gpu.commands.Get(), moving ? movingGuides : zeroGuides); Complete(gpu);
            gpu.Begin(); gpu.BeginTimings();
            ModelKernelTrace::LoopReport loop{}; double cpuEvaluate = 0;
            const double began = ClockMs();
            gpu.Time(GpuStage::NR);
            const auto evaluate = [&](ID3D12Resource* input, ID3D12Resource* target, unsigned pass, GpuStage stage) {
                Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(gpu.commands.Get(), target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                Barrier(gpu.commands.Get(), depth.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                Barrier(gpu.commands.Get(), motion.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                gpu.Time(stage);
                if (mode == 1) ModelKernelTrace::BeginVitLoop(gpu.commands.Get(), gpu.device.Get(), alpha);
                const double start = ClockMs();
                runtime.Evaluate(gpu.commands.Get(), input, depth.Get(), motion.Get(), target, width, height, frame == 0, pass);
                cpuEvaluate += ClockMs() - start;
                if (mode == 1) {
                    loop = ModelKernelTrace::EndVitLoop();
                    if (loop.bodies != 2 || loop.replayedKernels != 42 || !loop.latentBytes)
                        throw std::runtime_error("ViT sequence did not replay exactly 42 native kernels.");
                }
                gpu.Time(stage, true);
                Barrier(gpu.commands.Get(), target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
                Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
                Barrier(gpu.commands.Get(), depth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
                Barrier(gpu.commands.Get(), motion.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            };
            try {
                evaluate(colour.Get(), intermediate.Get(), 0, GpuStage::InputMotion);
                if (mode == 2) {
                    gpu.Time(GpuStage::Encode); prepare.Run(gpu.commands.Get(), intermediate.Get(), colour.Get(), prepared.Get()); gpu.Time(GpuStage::Encode, true);
                    evaluate(prepared.Get(), output.Get(), 1, GpuStage::PrivateDlss);
                }
            } catch (...) {
                // Discard this unsubmitted list. Never continue an already
                // advanced host history after a failed evaluation/guard.
                ModelKernelTrace::CancelVitLoop(); throw;
            }
            gpu.Time(GpuStage::NR, true);
            const double cpuRecord = ClockMs() - began;
            gpu.ResolveTimings(); Complete(gpu);
            std::array<double, unsigned(GpuStage::Count)> times{};
            if (!gpu.TakeTimings(times)) throw std::runtime_error("Sequence timing readback unavailable.");
            const double total = times[unsigned(GpuStage::NR)], first = times[unsigned(GpuStage::InputMotion)],
                handoff = times[unsigned(GpuStage::Encode)], second = times[unsigned(GpuStage::PrivateDlss)];
            auto& readback = mode == 2 ? secondReadback : firstReadback;
            gpu.Begin(); readback.Copy(gpu.commands.Get()); Complete(gpu);
            auto answer = readback.Read();
            Save(outputPath(modeNames[mode], frame), answer); RequireFinite(answer);
            bool alphaZeroExact = false;
            if (mode == 1 && alpha == 0) {
                const auto baseline = Load(outputPath(L"one", frame), bytes);
                alphaZeroExact = answer == baseline;
                if (alphaZeroExact) ++checkedZeroFrames;
            }
            csv << name << ',' << frame << ',' << (frame < warmups) << ',' << (frame == 0) << ',' << (mode == 1 ? alpha : 0) << ','
                << total << ',' << first << ',' << handoff << ',' << second << ',' << cpuRecord << ',' << cpuEvaluate << ','
                << loop.replayedKernels << ',' << loop.latentBytes << ',' << inputHash << ',' << (moving ? movingMotionHash : zeroMotionHash)
                << ',' << Hash(answer) << ',' << (mode == 1 && alpha == 0 ? (alphaZeroExact ? "true" : "false") : "not-applicable") << '\n';
            csv.flush(); if (!csv) throw std::runtime_error("Writing sequence CSV failed.");
            if (mode == 1 && alpha == 0 && !alphaZeroExact)
                throw std::runtime_error(std::format("Alpha-zero ViT differs from baseline at chronological frame {}.", frame));
            if (frame >= warmups) {
                totalTimes.push_back(total); firstTimes.push_back(first); handoffTimes.push_back(handoff); secondTimes.push_back(second);
                recordTimes.push_back(cpuRecord); evaluateTimes.push_back(cpuEvaluate);
            }
        }
        runtime.Release();
        report["modes"][name] = {{"frames_saved", frames}, {"statistics", {
            {"gpu_total", Statistics(totalTimes)}, {"gpu_first_nr", Statistics(firstTimes)},
            {"gpu_handoff", Statistics(handoffTimes)}, {"gpu_second_nr", Statistics(secondTimes)},
            {"cpu_record", Statistics(recordTimes)}, {"cpu_evaluate", Statistics(evaluateTimes)}}}};
        if (mode == 1 && alpha == 0) {
            if (checkedZeroFrames != frames) throw std::runtime_error("Incomplete alpha-zero parity control.");
            report["alpha_zero_parity"] = "bit-exact-every-frame";
        }
        saveReport();
        std::cout << std::format("{} GPU minimum {:.4f}, p10 {:.4f}, mean {:.4f} ms ({} measured chronological frames).\n", name,
            report["modes"][name]["statistics"]["gpu_total"]["minimum_ms"].get<double>(),
            report["modes"][name]["statistics"]["gpu_total"]["p10_ms"].get<double>(),
            report["modes"][name]["statistics"]["gpu_total"]["mean_ms"].get<double>(), frames - warmups);
    }
    report["status"] = "complete"; report["all_outputs_finite"] = true;
    report["changed_inputs"] = std::any_of(inputHashes.begin() + 1, inputHashes.end(), [&](const auto& hash) { return hash != inputHashes[0]; });
    saveReport();
    return 0;
}
