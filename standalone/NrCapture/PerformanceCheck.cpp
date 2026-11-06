#include "Processor.h"
#include "Colour.h"
#include "FrameGenerationOutput.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace nr {
namespace {
void MoveFixture(Gpu& gpu, ID3D12Resource* input, ID3D12Resource* target, unsigned frame) {
    const auto size = input->GetDesc();
    const UINT width = UINT(size.Width), height = size.Height;
    const UINT shiftX = UINT(uint64_t(frame) * 7 % width), shiftY = UINT(uint64_t(frame) * 3 % height);
    gpu.Begin();
    Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(gpu.commands.Get(), target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
    source.pResource = input; source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.pResource = target; destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    const UINT sourceX[]{shiftX, 0}, sourceY[]{shiftY, 0};
    const UINT destinationX[]{0, width - shiftX}, destinationY[]{0, height - shiftY};
    const UINT widths[]{width - shiftX, shiftX}, heights[]{height - shiftY, shiftY};
    for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x) {
        if (!widths[x] || !heights[y]) continue;
        const D3D12_BOX region{sourceX[x], sourceY[y], 0, sourceX[x] + widths[x], sourceY[y] + heights[y], 1};
        gpu.commands->CopyTextureRegion(&destination, destinationX[x], destinationY[y], 0, &source, &region);
    }
    Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    Barrier(gpu.commands.Get(), target, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    gpu.Submit();
    if (!gpu.Drain()) throw std::runtime_error("Moving performance fixture preparation timed out.");
}

uint64_t OutputHash(Gpu& gpu, ID3D12Resource* output) {
    const auto desc = output->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0; UINT64 rowBytes = 0, bytes = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowBytes, &bytes);
    auto readback = gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
    source.pResource = output; source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.pResource = readback.Get(); destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;
    gpu.Begin();
    gpu.commands->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    gpu.Submit();
    if (!gpu.Drain()) throw std::runtime_error("Performance output readback timed out.");
    void* mapped = nullptr; D3D12_RANGE range{0, SIZE_T(bytes)};
    Check(readback->Map(0, &range, &mapped), "Performance output readback");
    // Hash only actual RGBA bytes, excluding undefined row-pitch padding.
    uint64_t hash = 14695981039346656037ull;
    for (UINT y = 0; y < rows; ++y) {
        const auto* row = static_cast<const BYTE*>(mapped) + footprint.Offset + SIZE_T(y) * footprint.Footprint.RowPitch;
        for (UINT64 x = 0; x < rowBytes; ++x) { hash ^= row[x]; hash *= 1099511628211ull; }
    }
    D3D12_RANGE written{0, 0}; readback->Unmap(0, &written);
    return hash;
}
}

void PerformanceCheck(Gpu& gpu, Processor& processor, ID3D12Resource* input, Settings settings, unsigned frames,
    unsigned modeMask, bool moving) {
    if (!frames || !modeMask || (modeMask & ~0xfu)) throw std::runtime_error("Invalid performance frames or mode mask.");
    processor.Resize(processor.width, processor.height, settings);
    std::cout << std::format("Capture {}x{}, {} measured frames after 5 warmup frames, estimated motion {}\n",
        processor.width, processor.height, frames, settings.estimateMotion);
    std::cout << (moving ? "Fixture: wrapped translation, 7px horizontal/3px vertical per frame; preparation excluded\n"
                         : "Fixture: unchanged captured frame\n");
    for (unsigned pass = 0; pass < settings.passes; ++pass) {
        const auto profile = settings.PassResolution(pass);
        const auto [width, height] = processor.PassModelSize(pass);
        std::cout << std::format("Pass {}: {}x{} model, {}% scale, transfer {}, upscaler {}, downscaler {}, foveation {}\n",
            pass + 1, width, height, profile.modelScale, profile.transfer, profile.upscaler,
            profile.downscaler, profile.spatial.enabled);
    }
    const auto monitor = MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY);
    auto report = [](const char* name, std::vector<double> values) {
        const auto mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        std::sort(values.begin(), values.end());
        std::cout << std::format("{}: {:.3f} ms mean, {:.3f} ms p95\n", name, mean,
            values[std::min(values.size() - 1, values.size() * 95 / 100)]) << std::flush;
    };
    // Mirror the live per-frame HDR query without creating any windows/capture sessions.
    std::vector<double> hdrTimes, whiteTimes;
    for (unsigned frame = 0; frame < 20; ++frame) {
        const auto start = ClockMs();
        for (UINT i = 0; ; ++i) {
            ComPtr<IDXGIOutput> output;
            if (gpu.adapter->EnumOutputs(i, &output) == DXGI_ERROR_NOT_FOUND) break;
            ComPtr<IDXGIOutput6> extended;
            if (FAILED(output.As(&extended))) continue;
            DXGI_OUTPUT_DESC1 desc{};
            if (SUCCEEDED(extended->GetDesc1(&desc)) && desc.Monitor == monitor) break;
        }
        hdrTimes.push_back(ClockMs() - start);
        const auto whiteStart = ClockMs(); DesktopWhitePoint(monitor);
        whiteTimes.push_back(ClockMs() - whiteStart);
    }
    report("Live HDR enumeration/query", hdrTimes);
    report("Desktop white query (normally once/second)", whiteTimes);
    ComPtr<ID3D12QueryHeap> queries;
    D3D12_QUERY_HEAP_DESC queryDesc{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
    Check(gpu.device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queries)), "Performance timestamps");
    auto readback = gpu.Buffer(16, D3D12_HEAP_TYPE_READBACK);
    UINT64 frequency = 0; Check(gpu.queue->GetTimestampFrequency(&frequency), "Timestamp frequency");
    const bool hdr = settings.inputColour == 2;
    ComPtr<ID3D12Resource> movedInput;
    if (moving) movedInput = gpu.Texture(processor.width, processor.height, input->GetDesc().Format,
        D3D12_RESOURCE_STATE_COMMON, (input->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0);
    std::unique_ptr<FrameGenerationOutput> fgOutput;
    if (modeMask & 0xcu) {
        fgOutput = std::make_unique<FrameGenerationOutput>(gpu);
        fgOutput->Resize(processor.width, processor.height, hdr);
    }
    for (const int mode : {0, 1, 2, 3}) {
        if (!(modeMask & (1u << mode))) continue;
        const int fps = mode == 0 ? -1 : 120;
        std::vector<double> resizeTimes, cpuTimes, completeTimes, gpuTimes;
        std::array<std::vector<double>, unsigned(GpuStage::Count)> stageTimes;
        ID3D12Resource* finalOutput = nullptr;
        for (unsigned frame = 0; frame < frames + 5; ++frame) {
            if (moving) MoveFixture(gpu, input, movedInput.Get(), frame);
            // The live loop checks effective settings/resources every frame.
            // Include this cost, especially when profiling independent passes.
            const auto resizeStart = ClockMs();
            processor.Resize(processor.width, processor.height, settings);
            const auto start = ClockMs();
            gpu.Begin(); gpu.BeginTimings();
            gpu.commands->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            auto* output = processor.Run(moving ? movedInput.Get() : input, settings, hdr, false, 1, 8.33f, fps);
            finalOutput = output;
            auto* sharedMotion = mode == 3 ? processor.FullResolutionMotion() : nullptr;
            // Match live capture: zero-motion NR resets its own history every
            // frame, but independent FG motion still retains its history.
            if (mode >= 2) fgOutput->Run(output, frame == 0 || (sharedMotion && processor.historyReset), sharedMotion);
            gpu.commands->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            gpu.commands->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback.Get(), 0);
            gpu.ResolveTimings();
            gpu.Submit();
            const auto submitted = ClockMs();
            if (!gpu.Drain()) throw std::runtime_error("Performance check GPU timeout.");
            const auto completed = ClockMs();
            std::array<double, unsigned(GpuStage::Count)> stages{};
            if (!gpu.TakeTimings(stages)) throw std::runtime_error("GPU stage timings were not ready after completion.");
            void* data = nullptr; Check(readback->Map(0, nullptr, &data), "Performance readback");
            const auto* stamps = static_cast<const UINT64*>(data);
            const auto elapsed = (stamps[1] - stamps[0]) * 1000.0 / frequency;
            readback->Unmap(0, nullptr);
            if (frame >= 5) {
                resizeTimes.push_back(start - resizeStart);
                cpuTimes.push_back(submitted - start); completeTimes.push_back(completed - resizeStart); gpuTimes.push_back(elapsed);
                for (unsigned i = 0; i < stages.size(); ++i) stageTimes[i].push_back(stages[i]);
            }
        }
        std::cout << (mode == 0 ? "NR only\n" : mode == 1 ? "NR + FPS label\n" : mode == 2 ?
            "NR + FPS label + independent FG motion/encoding (no interpolation/capture/presentation)\n" :
            "NR + FPS label + shared FG motion/encoding (no interpolation/capture/presentation)\n");
        report("CPU per-frame Resize/settings check", resizeTimes);
        report("CPU record including internal GPU waits", cpuTimes);
        report("Processing through GPU completion (including Resize)", completeTimes);
        report("GPU timestamp span", gpuTimes);
        for (unsigned i = 0; i < stageTimes.size(); ++i) report(GpuStageNames[i], stageTimes[i]);
        std::cout << std::format("Final output FNV-1a-64: {:016X} (readback excluded from timings)\n", OutputHash(gpu, finalOutput))
            << std::flush;
    }
}
void PerformanceCheck(Gpu& gpu, Processor& processor, ID3D12Resource* input, Settings settings, unsigned frames) {
    PerformanceCheck(gpu, processor, input, settings, frames, 0xfu, false);
}
}
