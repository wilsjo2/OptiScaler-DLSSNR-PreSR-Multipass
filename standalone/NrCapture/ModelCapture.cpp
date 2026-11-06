#include "ModelCapture.h"
#include "Capture.h"
#include "Colour.h"
#include "../../external/nlohmann/json.hpp"
#include <winrt/base.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
using namespace nr;
using Json = nlohmann::json;

Json DisplayMetadata(Gpu& gpu, HMONITOR monitor) {
    Json result{{"hdr_enabled", nullptr}, {"desktop_sdr_white_scrgb", nullptr},
                {"desktop_sdr_white_nits", nullptr}};
    const float white = DesktopWhitePoint(monitor);
    if (white > 0) {
        result["desktop_sdr_white_scrgb"] = white;
        result["desktop_sdr_white_nits"] = white * 80;
    }
    MONITORINFOEXW info{}; info.cbSize = sizeof(info);
    if (GetMonitorInfoW(monitor, &info)) result["display_name"] = Narrow(info.szDevice);
    for (UINT index = 0; ; ++index) {
        ComPtr<IDXGIOutput> output;
        const auto status = gpu.adapter->EnumOutputs(index, &output);
        if (status == DXGI_ERROR_NOT_FOUND) break;
        Check(status, "Enumerate capture display");
        ComPtr<IDXGIOutput6> extended;
        if (FAILED(output.As(&extended))) continue;
        DXGI_OUTPUT_DESC1 desc{};
        if (FAILED(extended->GetDesc1(&desc)) || desc.Monitor != monitor) continue;
        result["hdr_enabled"] = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
        result["dxgi_color_space"] = unsigned(desc.ColorSpace);
        result["bits_per_color"] = desc.BitsPerColor;
        result["minimum_luminance_nits"] = desc.MinLuminance;
        result["maximum_luminance_nits"] = desc.MaxLuminance;
        result["maximum_full_frame_luminance_nits"] = desc.MaxFullFrameLuminance;
        break;
    }
    return result;
}

struct Apartment {
    Apartment() { winrt::init_apartment(winrt::apartment_type::multi_threaded); }
    ~Apartment() { winrt::uninit_apartment(); }
};

// The capture reader slot and readback allocation must outlive the submitted copy.
struct StopAndDrain {
    Capture& capture;
    Gpu& gpu;
    ID3D12Resource* readback;
    ~StopAndDrain() {
        capture.StopProducer();
        bool complete = false;
        try { complete = gpu.Drain(30000); } catch (...) {}
        if (!complete) {
            // A timed-out queue may still use this shared source, its readback
            // destination and allocator. Intentionally retain until process exit.
            if (auto* source = capture.Input()) source->AddRef();
            readback->AddRef(); gpu.queue->AddRef(); gpu.allocator->AddRef();
            gpu.commands->AddRef(); gpu.fence->AddRef();
        }
    }
};
}

int RunModelCapture(int argc, wchar_t** argv) try {
    if (argc < 4 || argc > 6)
        throw std::runtime_error("Usage: NrModelProbe --capture-sequence <exact-window-title> <output-directory> [frames=48] [fps=24]");
    const unsigned requested = argc > 4 ? std::stoul(argv[4]) : 48;
    const double fps = argc > 5 ? std::stod(argv[5]) : 24;
    if (!requested || requested > 240 || !std::isfinite(fps) || fps < 1 || fps > 120)
        throw std::runtime_error("Capture requires 1..240 frames and 1..120 fps.");
    const std::filesystem::path directory = std::filesystem::absolute(argv[3]);
    if (std::filesystem::exists(directory) && !std::filesystem::is_empty(directory))
        throw std::runtime_error("Capture output directory must be new or empty.");
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    Apartment apartment;
    const HWND window = FindWindowW(nullptr, argv[2]);
    if (!window || !IsWindowVisible(window) || IsIconic(window))
        throw std::runtime_error("The requested capture window must already be open, visible and not minimized.");
    CaptureSource source(window);
    RECT bounds{};
    if (!source.Bounds(bounds)) throw std::runtime_error("Cannot read capture client bounds.");
    const UINT width = bounds.right - bounds.left, height = bounds.bottom - bounds.top;
    if (width > 8192 || height > 8192) throw std::runtime_error("Capture extent exceeds 8192 pixels.");
    const size_t rowBytes = size_t(width) * 8, frameBytes = rowBytes * height;
    const UINT64 totalBytes = UINT64(frameBytes) * requested;
    MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
    if (totalBytes > 8ull * 1024 * 1024 * 1024 ||
        (GlobalMemoryStatusEx(&memory) && totalBytes + 256ull * 1024 * 1024 > memory.ullAvailPhys))
        throw std::runtime_error("Insufficient RAM for this buffered capture; request fewer frames.");

    // Allocate and touch all frame storage before acquisition, avoiding allocation
    // or disk stalls during the short sequence. At 4K, 48 frames consume ~3 GiB.
    std::vector<std::vector<char>> frames;
    frames.reserve(requested);
    for (unsigned i = 0; i < requested; ++i) frames.emplace_back(frameBytes);
    Gpu gpu;
    const HMONITOR monitor = source.Monitor();
    const auto displayStart = DisplayMetadata(gpu, monitor);
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 readbackBytes = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &readbackBytes);
    auto readback = gpu.Buffer(readbackBytes, D3D12_HEAP_TYPE_READBACK);
    std::filesystem::create_directories(directory / L"raw");
    SYSTEMTIME utc{}; GetSystemTime(&utc);
    Json metadata{{"schema", 1}, {"capture", "Windows Graphics Capture window client"},
        {"window_title", Narrow(argv[2])}, {"width", width}, {"height", height},
        {"format", "little-endian tightly packed RGBA16F"}, {"row_bytes", rowBytes},
        {"colour_space", "linear scRGB"}, {"scrgb_unit_nits", 80},
        {"model_encoded", false}, {"requested_frames", requested}, {"requested_fps", fps},
        {"timestamp_unit", "milliseconds"}, {"timestamp_clock", "WGC SystemRelativeTime (QPC)"},
        {"utc_started", std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z", utc.wYear,
            utc.wMonth, utc.wDay, utc.wHour, utc.wMinute, utc.wSecond, utc.wMilliseconds)},
        {"display_start", displayStart}, {"frames", Json::array()},
        {"notes", "No NR, presentation, resizing, input synthesis, focus changes, depth or motion vectors. Captures newest frames at a nominal rate; actual timestamps are recorded. HDR is display status, not proof of the game's render colour space."}};
    std::vector<double> timestamps(requested), acquired(requested), copyMs(requested);
    std::vector<uint64_t> received(requested), published(requested);
    std::cout << std::format("Capturing {} frames at {} fps now.\n", requested, fps) << std::flush;
    Capture capture(gpu, source, 0, fps);
    StopAndDrain cleanup{capture, gpu, readback.Get()};
    const double start = ClockMs(), deadline = start + 15000 + requested * 1000.0 / fps * 3;
    unsigned count = 0;
    while (count < requested) {
        if (ClockMs() > deadline) throw std::runtime_error("Timed out waiting for fresh capture frames; no sequence was written.");
        if (!source.Exists() || IsIconic(window) || source.Monitor() != monitor)
            throw std::runtime_error("Capture window closed, minimized or changed displays; no sequence was written.");
        if (!capture.Next()) {
            if (capture.Closed()) throw std::runtime_error("Capture source closed before the sequence completed.");
            WaitForSingleObject(capture.Event(), 10);
            continue;
        }
        if (capture.width != width || capture.height != height)
            throw std::runtime_error("Capture window resized; no sequence was written.");
        if (count && capture.timestamp <= timestamps[count - 1]) continue;
        auto* input = capture.Input();
        const auto inputDesc = input->GetDesc();
        if (inputDesc.Format != desc.Format || inputDesc.Width != width || inputDesc.Height != height)
            throw std::runtime_error("Unexpected capture texture format or dimensions.");
        const double began = ClockMs();
        timestamps[count] = capture.timestamp; acquired[count] = began;
        received[count] = capture.receivedFrames; published[count] = capture.publishedFrames;
        gpu.Begin();
        Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = input; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.pResource = readback.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = footprint;
        gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        Barrier(gpu.commands.Get(), input, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        gpu.Submit();
        if (!gpu.Drain(30000)) throw std::runtime_error("Capture readback GPU timeout.");
        void* mapped = nullptr;
        const D3D12_RANGE readRange{0, size_t(readbackBytes)};
        Check(readback->Map(0, &readRange, &mapped), "Map capture readback");
        for (UINT y = 0; y < height; ++y)
            memcpy(frames[count].data() + size_t(y) * rowBytes,
                static_cast<const char*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, rowBytes);
        const D3D12_RANGE noWrites{0, 0}; readback->Unmap(0, &noWrites);
        copyMs[count] = ClockMs() - began;
        ++count;
    }
    capture.StopProducer();
    if (!gpu.Drain(30000)) throw std::runtime_error("Capture shutdown GPU timeout.");
    capture.Stop();
    const auto displayEnd = DisplayMetadata(gpu, monitor);
    metadata["display_end"] = displayEnd;
    metadata["display_metadata_changed"] = displayStart != displayEnd;
    metadata["captured_frames"] = count;
    metadata["span_ms"] = timestamps.back() - timestamps.front();
    for (unsigned i = 0; i < count; ++i) {
        const auto name = std::format("raw/frame-{:04}.rgba16f", i);
        std::ofstream file(directory / name, std::ios::binary);
        file.write(frames[i].data(), std::streamsize(frameBytes)); file.close();
        if (!file) throw std::runtime_error("Writing captured frame failed.");
        metadata["frames"].push_back({{"index", i}, {"file", name},
            {"timestamp_ms", timestamps[i]}, {"relative_ms", timestamps[i] - timestamps.front()},
            {"delta_ms", i ? timestamps[i] - timestamps[i - 1] : 0},
            {"readback_started_qpc_ms", acquired[i]}, {"readback_cpu_ms", copyMs[i]},
            {"received_frames", received[i]}, {"published_frames", published[i]}});
    }
    std::ofstream manifest(directory / L"capture.json"); manifest << metadata.dump(2) << '\n'; manifest.close();
    if (!manifest) throw std::runtime_error("Writing capture metadata failed.");
    std::cout << std::format("Saved {} raw scRGB frames at {}x{}, spanning {:.1f} ms, to {}\n",
        count, width, height, timestamps.back() - timestamps.front(), Narrow(directory.wstring()));
    return 0;
} catch (const winrt::hresult_error& error) {
    throw std::runtime_error("Window capture: " + Narrow(error.message().c_str()));
}
