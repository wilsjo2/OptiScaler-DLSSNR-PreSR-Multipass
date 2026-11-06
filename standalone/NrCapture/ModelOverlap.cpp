#include "ModelOverlap.h"
#include "Gpu.h"
#include "Runtime.h"
#include "ModelKernelTrace.h"
#include <ModelOverlapResize.h>
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <vector>

namespace {
using namespace nr;
using Pixels = std::vector<char>;

[[noreturn]] void InFlightFailure() {
    // This is a disposable research process. If completion is unknown, normal
    // unwinding would free feature arenas and textures still in GPU use.
    std::cerr << "Cannot establish GPU completion; terminating the overlap probe before releasing submitted resources.\n";
    std::cerr.flush(); TerminateProcess(GetCurrentProcess(), 2); std::abort();
}

void Validate(ID3D12Device* device) {
    ComPtr<ID3D12InfoQueue> diagnostics;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&diagnostics)))) return;
    bool failed = false;
    for (UINT64 i = 0; i < diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
        SIZE_T bytes = 0; diagnostics->GetMessage(i, nullptr, &bytes);
        std::vector<BYTE> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        Check(diagnostics->GetMessage(i, message, &bytes), "Read overlap validation message");
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            std::cerr << message->pDescription << '\n'; failed = true;
        }
    }
    diagnostics->ClearStoredMessages();
    if (failed) throw std::runtime_error("D3D12 validation failed in model overlap probe.");
}
void Complete(Gpu& gpu) {
    try {
        gpu.Submit();
        if (!gpu.Drain(60000)) InFlightFailure();
    } catch (...) {
        try { if (!gpu.Drain(60000)) InFlightFailure(); }
        catch (...) { InFlightFailure(); }
        throw;
    }
    Validate(gpu.device.Get());
}
std::string Hash(const Pixels& pixels) {
    UINT64 hash = 14695981039346656037ull;
    for (const auto byte : pixels) { hash ^= static_cast<unsigned char>(byte); hash *= 1099511628211ull; }
    return std::format("{:016x}", hash);
}
void Save(const std::filesystem::path& path, const Pixels& pixels) {
    std::ofstream file(path, std::ios::binary);
    file.write(pixels.data(), std::streamsize(pixels.size()));
    if (!file) throw std::runtime_error("Cannot write overlap output.");
}
void RequireEqual(const Pixels& actual, const Pixels& expected, const char* label) {
    if (actual == expected) return;
    size_t differences = 0;
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) differences += actual[i] != expected[i];
    throw std::runtime_error(std::format("{} differs from DIRECT reference: {} byte differences, actual {}, expected {}.",
        label, differences, Hash(actual), Hash(expected)));
}

// Copies are deliberately outside the NR timestamps and outside each measured
// paired wave. All textures hand off between queues in COMMON state.
struct Transfer {
    ID3D12Resource* texture;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    ComPtr<ID3D12Resource> buffer;
    size_t rowBytes;
    UINT height;
    bool upload;
    Transfer(Gpu& gpu, ID3D12Resource* texture, size_t pixelBytes, bool upload)
        : texture(texture), upload(upload) {
        const auto description = texture->GetDesc(); UINT64 bytes = 0;
        height = description.Height; rowBytes = size_t(description.Width) * pixelBytes;
        gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        buffer = gpu.Buffer(bytes, upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK);
    }
    void Copy(ID3D12GraphicsCommandList* commands) {
        D3D12_TEXTURE_COPY_LOCATION tex{}, buf{};
        tex.pResource = texture; tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        buf.pResource = buffer.Get(); buf.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        buf.PlacedFootprint = footprint;
        const auto state = upload ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        Barrier(commands, texture, D3D12_RESOURCE_STATE_COMMON, state);
        commands->CopyTextureRegion(upload ? &tex : &buf, 0, 0, 0, upload ? &buf : &tex, nullptr);
        Barrier(commands, texture, state, D3D12_RESOURCE_STATE_COMMON);
    }
    void Upload(ID3D12GraphicsCommandList* commands, const Pixels& pixels) {
        if (pixels.size() != rowBytes * height) throw std::runtime_error("Invalid overlap upload size.");
        void* mapped = nullptr; D3D12_RANGE none{};
        Check(buffer->Map(0, &none, &mapped), "Map overlap upload");
        for (UINT y = 0; y < height; ++y)
            memcpy(static_cast<BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
                pixels.data() + size_t(y) * rowBytes, rowBytes);
        buffer->Unmap(0, nullptr); Copy(commands);
    }
    Pixels Read() {
        Pixels pixels(rowBytes * height); void* mapped = nullptr;
        D3D12_RANGE range{SIZE_T(footprint.Offset), SIZE_T(footprint.Offset + UINT64(footprint.Footprint.RowPitch) * height)};
        Check(buffer->Map(0, &range, &mapped), "Map overlap output");
        for (UINT y = 0; y < height; ++y)
            memcpy(pixels.data() + size_t(y) * rowBytes,
                static_cast<BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, rowBytes);
        D3D12_RANGE none{}; buffer->Unmap(0, &none); return pixels;
    }
};

struct Interval { double start = 0, end = 0; double Duration() const { return end - start; } };
struct Times { Interval handoff, nr; };
struct Compute {
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> readback;
    Handle event{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    UINT64 serial = 0;
    explicit Compute(Gpu& gpu) {
        if (!event.value) Check(HRESULT_FROM_WIN32(GetLastError()), "Overlap fence event");
        D3D12_COMMAND_QUEUE_DESC description{}; description.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        Check(gpu.device->CreateCommandQueue(&description, IID_PPV_ARGS(&queue)), "Overlap compute queue");
        Check(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator)), "Overlap allocator");
        Check(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(), nullptr,
            IID_PPV_ARGS(&commands)), "Overlap compute list");
        Check(commands->Close(), "Close initial overlap list");
        Check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Overlap fence");
        D3D12_QUERY_HEAP_DESC heap{}; heap.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; heap.Count = 4;
        Check(gpu.device->CreateQueryHeap(&heap, IID_PPV_ARGS(&queries)), "Overlap timestamps");
        readback = gpu.Buffer(4 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK);
    }
    void Begin() {
        Wait();
        Check(allocator->Reset(), "Reset overlap allocator");
        Check(commands->Reset(allocator.Get(), nullptr), "Reset overlap list");
    }
    void Close() { Check(commands->Close(), "Close overlap list"); }
    UINT64 Submit(ID3D12CommandQueue* destination) {
        ID3D12CommandList* lists[]{commands.Get()}; destination->ExecuteCommandLists(1, lists);
        Check(destination->Signal(fence.Get(), ++serial), "Signal overlap completion"); return serial;
    }
    void Wait() {
        const auto completed = fence->GetCompletedValue();
        if (completed == UINT64_MAX) throw std::runtime_error("Device removed during overlap probe.");
        if (completed >= serial) return;
        Check(fence->SetEventOnCompletion(serial, event.value), "Overlap completion event");
        if (WaitForSingleObject(event.value, 60000) != WAIT_OBJECT_0) throw std::runtime_error("Overlap compute timeout.");
    }
    Times ReadTimes(ID3D12CommandQueue* actualQueue, UINT64 qpcOrigin, UINT64 qpcFrequency) {
        UINT64 frequency = 0, calibratedGpu = 0, calibratedCpu = 0;
        Check(actualQueue->GetTimestampFrequency(&frequency), "Overlap timestamp frequency");
        // Calibration happens after execution. Subtract as double, never UINT64:
        // the recorded timestamps normally precede this calibration point.
        Check(actualQueue->GetClockCalibration(&calibratedGpu, &calibratedCpu), "Overlap clock calibration");
        UINT64* mapped = nullptr; D3D12_RANGE range{0, 4 * sizeof(UINT64)};
        Check(readback->Map(0, &range, reinterpret_cast<void**>(&mapped)), "Map overlap timestamps");
        std::array<UINT64, 4> ticks{}; std::copy_n(mapped, 4, ticks.begin());
        D3D12_RANGE none{}; readback->Unmap(0, &none);
        if (!frequency || !ticks[0] || !std::is_sorted(ticks.begin(), ticks.end())) throw std::runtime_error("Invalid overlap timestamps.");
        const auto convert = [&](UINT64 ticks) {
            return (double(calibratedCpu) - double(qpcOrigin)) * 1000.0 / double(qpcFrequency)
                + (double(ticks) - double(calibratedGpu)) * 1000.0 / double(frequency);
        };
        return {{convert(ticks[0]), convert(ticks[1])}, {convert(ticks[2]), convert(ticks[3])}};
    }
};
struct DrainOnExit {
    Compute& a; Compute& b;
    ~DrainOnExit() { try { a.Wait(); b.Wait(); } catch (...) { InFlightFailure(); } }
};
struct Row {
    unsigned wave = 0, aFrame = 0, bFrame = 0;
    bool warmup = false;
    Times a, b;
    double overlap = 0, makespan = 0, nrMakespan = 0, cpuRecord = 0, cpuEvaluateA = 0, cpuEvaluateB = 0, wall = 0;
    std::string aHash, bHash;
};

// The B list owns this one dispatch and its descriptors. Its prior submission
// always completes before the next recording changes descriptors/constants.
struct PrepareB {
    Gpu& gpu;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> heap;
    explicit PrepareB(Gpu& gpu) : gpu(gpu) {
        D3D12_DESCRIPTOR_RANGE ranges[]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0},
            {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 2}};
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable = {2, ranges};
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants = {0, 0, 4};
        D3D12_ROOT_SIGNATURE_DESC signature{2, parameters}; ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "Overlap handoff root signature");
        Check(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Overlap handoff root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC description{}; description.pRootSignature = root.Get();
        description.CS = {ModelOverlapResize, sizeof ModelOverlapResize};
        Check(gpu.device->CreateComputePipelineState(&description, IID_PPV_ARGS(&pipeline)), "Overlap handoff pipeline");
        D3D12_DESCRIPTOR_HEAP_DESC descriptors{}; descriptors.NumDescriptors = 3;
        descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; descriptors.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(gpu.device->CreateDescriptorHeap(&descriptors, IID_PPV_ARGS(&heap)), "Overlap handoff descriptors");
    }
    void Run(ID3D12GraphicsCommandList* commands, ID3D12Resource* answer, ID3D12Resource* original, ID3D12Resource* target) {
        auto handle = heap->GetCPUDescriptorHandleForHeapStart();
        const auto stride = gpu.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        for (auto* source : {answer, original}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = source->GetDesc().Format;
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; view.Texture2D.MipLevels = 1;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            gpu.device->CreateShaderResourceView(source, &view, handle); handle.ptr += stride;
            Barrier(commands, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        gpu.device->CreateUnorderedAccessView(target, nullptr, nullptr, handle);
        Barrier(commands, target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[]{heap.Get()}; commands->SetDescriptorHeaps(1, heaps);
        commands->SetComputeRootSignature(root.Get()); commands->SetPipelineState(pipeline.Get());
        commands->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        const UINT constants[]{UINT(answer->GetDesc().Width), answer->GetDesc().Height, UINT(target->GetDesc().Width), target->GetDesc().Height};
        commands->SetComputeRoot32BitConstants(1, 4, constants, 0);
        commands->Dispatch((constants[2] + 7) / 8, (constants[3] + 7) / 8, 1);
        Barrier(commands, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        for (auto* source : {answer, original})
            Barrier(commands, source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
};

struct Stats {
    double minimum = 0, p10 = 0, mean = 0, median = 0;
    explicit Stats(std::vector<double> values) {
        if (values.empty()) throw std::runtime_error("No measured overlap samples.");
        mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        std::sort(values.begin(), values.end()); minimum = values.front();
        // Linear interpolation at p*(N-1), explicitly recorded in the report.
        const double index = .1 * (values.size() - 1); const size_t lo = size_t(index), hi = std::min(lo + 1, values.size() - 1);
        p10 = values[lo] + (values[hi] - values[lo]) * (index - lo);
        median = (values[(values.size()-1)/2] + values[values.size()/2]) * .5;
    }
};

Pixels Load(const std::filesystem::path& path, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || size_t(file.tellg()) != bytes) throw std::runtime_error("Expected tightly packed model-space RGBA16F: " + Narrow(path.wstring()));
    Pixels pixels(bytes); file.seekg(0); file.read(pixels.data(), std::streamsize(bytes));
    if (!file) throw std::runtime_error("Reading overlap input failed."); return pixels;
}

// Long 4K sequences use bounded RAM. Keep the cache after failure for diagnosis;
// remove only this probe-created cache after every byte has been compared.
struct References {
    bool sequence;
    size_t aBytes, bBytes;
    std::filesystem::path path;
    std::fstream cache;
    std::array<std::array<Pixels, 2>, 2> resetFrames;
    References(bool sequence, size_t aBytes, size_t bBytes, const std::filesystem::path& path)
        : sequence(sequence), aBytes(aBytes), bBytes(bBytes), path(path) {
        if (sequence) {
            cache.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
            if (!cache) throw std::runtime_error("Cannot create chronological reference cache.");
        }
    }
    void Store(unsigned frame, std::array<Pixels, 2> values) {
        if (!sequence) { resetFrames.at(frame) = std::move(values); return; }
        cache.seekp(std::streamoff(frame) * std::streamoff(aBytes + bBytes));
        for (const auto& value : values) cache.write(value.data(), std::streamsize(value.size()));
        if (!cache) throw std::runtime_error("Writing chronological reference cache failed.");
    }
    Pixels Get(unsigned frame, unsigned feature) {
        if (!sequence) return resetFrames.at(frame % 2).at(feature);
        const size_t bytes = feature ? bBytes : aBytes; Pixels result(bytes);
        cache.seekg(std::streamoff(frame) * std::streamoff(aBytes + bBytes) + (feature ? std::streamoff(aBytes) : 0));
        cache.read(result.data(), std::streamsize(bytes));
        if (!cache) throw std::runtime_error("Reading chronological reference cache failed."); return result;
    }
    void Finish() { if (sequence) { cache.close(); std::filesystem::remove(path); } }
};
}

int RunModelOverlap(int argc, wchar_t** argv) {
    try {
        if (argc != 8 && argc != 9 && argc != 11 && argc != 12)
            throw std::runtime_error("Usage: NrModelProbe --overlap <runtime> <input.rgba16f|sequence-directory> <a-width> <a-height> <output-prefix> <single|serial|parallel> [samples=40] [b-width b-height] [reset|sequence=reset]");
        const UINT width = std::stoul(argv[4]), height = std::stoul(argv[5]);
        const UINT bWidth = argc > 9 ? std::stoul(argv[9]) : width, bHeight = argc > 10 ? std::stoul(argv[10]) : height;
        const std::filesystem::path prefix = argv[6], inputPath = argv[3];
        const std::wstring mode = argv[7], history = argc > 11 ? argv[11] : L"reset";
        const unsigned samples = argc > 8 ? std::stoul(argv[8]) : 40;
        constexpr unsigned warmups = 6;
        const unsigned waves = warmups + samples;
        const bool sequence = history == L"sequence", mixed = bWidth != width || bHeight != height;
        const bool directory = std::filesystem::is_directory(inputPath);
        if (!width || !height || width > 8192 || height > 8192 || !bWidth || !bHeight || bWidth > width || bHeight > height ||
            !samples || samples > 10000) throw std::runtime_error("Unsupported overlap extent/sample count; B must be no larger than A.");
        if (mode != L"single" && mode != L"serial" && mode != L"parallel") throw std::runtime_error("Unknown overlap mode.");
        if (history != L"reset" && history != L"sequence") throw std::runtime_error("Unknown overlap history mode.");
        if (directory && !sequence) throw std::runtime_error("A capture directory requires sequence mode.");
        if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
        const auto path = [&](const wchar_t* suffix) { return std::filesystem::path(prefix.wstring() + suffix); };
        const auto framePath = [&](unsigned frame) { return inputPath / std::format(L"frame-{:04}.rgba16f", frame); };
        const size_t pixelCount = size_t(width) * height, bPixelCount = size_t(bWidth) * bHeight;
        const size_t bytes = pixelCount * 8, bBytes = bPixelCount * 8;
        // Fail missing/incomplete captures before initializing or exercising NR.
        if (directory) for (unsigned frame = 0; frame <= waves; ++frame)
            if (!std::filesystem::is_regular_file(framePath(frame)) || std::filesystem::file_size(framePath(frame)) != bytes)
                throw std::runtime_error("Missing/incomplete sequence frame: " + Narrow(framePath(frame).wstring()));
        const Pixels base = directory ? Pixels{} : Load(inputPath, bytes);
        const auto pixelsFor = [&](unsigned frame) {
            if (directory) return Load(framePath(frame), bytes);
            const UINT dx = (sequence ? frame * 8 : (frame % 2) * 7) % width;
            const UINT dy = (sequence ? frame * 4 : (frame % 2) * 3) % height;
            Pixels result(bytes);
            for (UINT y = 0; y < height; ++y) {
                const char* source = base.data() + size_t((y + height - dy) % height) * width * 8;
                char* target = result.data() + size_t(y) * width * 8;
                memcpy(target, source + size_t(width-dx)*8, size_t(dx)*8);
                memcpy(target + size_t(dx)*8, source, size_t(width-dx)*8);
            }
            return result;
        };
        const std::array<UINT, 2> widths{width, bWidth}, heights{height, bHeight};
        std::array<Pixels, 2> zeroGuides{Pixels(pixelCount * 4), Pixels(bPixelCount * 4)}, movingGuides = zeroGuides;
        // Known backward vectors for the synthetic wrapped pan. Capture playback
        // has no geometric guides, so it explicitly uses zero depth and zero MV.
        if (sequence && !directory) for (unsigned feature = 0; feature < 2; ++feature) {
            const float dx = 8.0f * widths[feature] / width, dy = 4.0f * heights[feature] / height;
            for (UINT y = 0; y < heights[feature]; ++y) for (UINT x = 0; x < widths[feature]; ++x) {
                const uint16_t vector[]{DirectX::PackedVector::XMConvertFloatToHalf(x < dx ? widths[feature]-dx : -dx),
                    DirectX::PackedVector::XMConvertFloatToHalf(y < dy ? heights[feature]-dy : -dy)};
                memcpy(movingGuides[feature].data() + (size_t(y)*widths[feature]+x)*4, vector, 4);
            }
        }
        const auto motionFor = [&](unsigned feature, unsigned frame) -> const Pixels& {
            return sequence && frame && !directory ? movingGuides[feature] : zeroGuides[feature];
        };
        const auto resetFor = [&](unsigned frame) { return !sequence || !frame; };
        const std::string guideMode = directory ? "zero-depth-zero-motion-unavailable" :
            sequence ? "zero-depth-known-backward-wrapped-translation" : "zero-depth-zero-motion-reset";
        std::cout << std::format("Overlap model-only {}: A {}x{}, B {}x{}, {}, {} frames; guides {}.\n",
            Narrow(mode), width, height, bWidth, bHeight, Narrow(history), waves + 1, guideMode);

        Gpu gpu;
        auto runtimeA = std::make_shared<Runtime>(gpu.device.Get(), std::filesystem::absolute(argv[2]));
        Runtime runtimeB(runtimeA);
        DlssNr::ModelSettings settings; settings.preset = 1; settings.localTone = 1;
        const auto createFeatures = [&] {
            // Both are pass zero in independent feature owners, so tone one is
            // explicit for both and the later-pass default cannot change B.
            gpu.Begin();
            Runtime::CheckResult(runtimeA->Create(gpu.commands.Get(), width, height, settings), "Create A feature");
            Runtime::CheckResult(runtimeB.Create(gpu.commands.Get(), bWidth, bHeight, settings), "Create B feature");
            Complete(gpu);
        };
        createFeatures();
        const auto texture = [&](UINT w, UINT h, DXGI_FORMAT format) { return gpu.Texture(w, h, format, D3D12_RESOURCE_STATE_COMMON); };
        std::array<ComPtr<ID3D12Resource>, 2> colours{texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT), texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT)};
        std::array<ComPtr<ID3D12Resource>, 2> intermediate{texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT), texture(width, height, DXGI_FORMAT_R16G16B16A16_FLOAT)};
        std::array<ComPtr<ID3D12Resource>, 2> depths{texture(width, height, DXGI_FORMAT_R32_FLOAT), texture(bWidth, bHeight, DXGI_FORMAT_R32_FLOAT)};
        std::array<ComPtr<ID3D12Resource>, 2> motions{texture(width, height, DXGI_FORMAT_R16G16_FLOAT), texture(bWidth, bHeight, DXGI_FORMAT_R16G16_FLOAT)};
        auto output = texture(bWidth, bHeight, DXGI_FORMAT_R16G16B16A16_FLOAT);
        auto bInput = mixed ? texture(bWidth, bHeight, DXGI_FORMAT_R16G16B16A16_FLOAT) : ComPtr<ID3D12Resource>{};
        std::unique_ptr<PrepareB> prepare;
        if (mixed) prepare = std::make_unique<PrepareB>(gpu);
        Transfer input0(gpu, colours[0].Get(), 8, true), input1(gpu, colours[1].Get(), 8, true);
        Transfer depth0(gpu, depths[0].Get(), 4, true), depth1(gpu, depths[1].Get(), 4, true);
        Transfer motion0(gpu, motions[0].Get(), 4, true), motion1(gpu, motions[1].Get(), 4, true);
        Transfer readA0(gpu, intermediate[0].Get(), 8, false), readA1(gpu, intermediate[1].Get(), 8, false), readB(gpu, output.Get(), 8, false);
        std::array<Transfer*, 2> inputs{&input0, &input1}, readA{&readA0, &readA1};
        std::array<Transfer*, 2> depthUploads{&depth0, &depth1}, motionUploads{&motion0, &motion1};
        Compute a(gpu), b(gpu);
        DrainOnExit drain{a, b};
        ID3D12CommandQueue* bQueue = mode == L"single" ? a.queue.Get() : b.queue.Get();
        const auto waitDirect = [&] {
            Check(gpu.queue->Wait(a.fence.Get(), a.serial), "Direct transfer waits for A");
            Check(gpu.queue->Wait(b.fence.Get(), b.serial), "Direct transfer waits for B");
        };
        const auto uploadFrame = [&](unsigned aFrame, unsigned bFrame) {
            // CPU uploads occur only after the previous wave drained; these
            // three snapshots stay immutable until both feature fences finish.
            const auto pixels = pixelsFor(aFrame);
            waitDirect(); gpu.Begin(); inputs[aFrame % 2]->Upload(gpu.commands.Get(), pixels);
            for (unsigned feature = 0; feature < 2; ++feature) {
                depthUploads[feature]->Upload(gpu.commands.Get(), zeroGuides[feature]);
                motionUploads[feature]->Upload(gpu.commands.Get(), motionFor(feature, feature ? bFrame : aFrame));
            }
            Complete(gpu); return Hash(pixels);
        };
        const auto evaluate = [&](ID3D12GraphicsCommandList* commands, unsigned feature, unsigned frame,
                                  ID3D12Resource* input, ID3D12Resource* target, Compute* timing, ID3D12CommandQueue* traceQueue) {
            Barrier(commands, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(commands, target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(commands, depths[feature].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(commands, motions[feature].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (timing) commands->EndQuery(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
            if (traceQueue) ModelKernelTrace::Begin(commands, traceQueue, false);
            const double start = ClockMs();
            (feature ? runtimeB : *runtimeA).Evaluate(commands, input, depths[feature].Get(), motions[feature].Get(),
                target, widths[feature], heights[feature], resetFor(frame));
            const double elapsed = ClockMs() - start;
            if (traceQueue) ModelKernelTrace::End(commands);
            if (timing) {
                commands->EndQuery(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
                commands->ResolveQueryData(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, timing->readback.Get(), 0);
            }
            Barrier(commands, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            Barrier(commands, input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            Barrier(commands, depths[feature].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            Barrier(commands, motions[feature].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            return elapsed;
        };
        const auto record = [&](ID3D12GraphicsCommandList* commands, unsigned feature, unsigned frame,
                                Compute* timing = nullptr, ID3D12CommandQueue* traceQueue = nullptr) {
            const unsigned slot = frame % 2;
            if (timing) commands->EndQuery(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if (feature && mixed) prepare->Run(commands, intermediate[slot].Get(), colours[slot].Get(), bInput.Get());
            if (timing) commands->EndQuery(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            return evaluate(commands, feature, frame,
                feature ? (mixed ? bInput.Get() : intermediate[slot].Get()) : colours[slot].Get(),
                feature ? output.Get() : intermediate[slot].Get(), timing, traceQueue);
        };
        const auto readPair = [&](unsigned slot) {
            waitDirect(); gpu.Begin(); readA[slot]->Copy(gpu.commands.Get()); readB.Copy(gpu.commands.Get()); Complete(gpu);
            return std::array<Pixels, 2>{readA[slot]->Read(), readB.Read()};
        };
        References references(sequence, bytes, bBytes, path(L".sequence-reference.tmp"));
        std::ofstream referenceCsv(path(L".references.csv"));
        referenceCsv << "frame,input_hash,a_motion_hash,b_motion_hash,a_hash,b_hash,reset\n";
        const unsigned referenceCount = sequence ? waves + 1 : 2;
        std::vector<std::string> inputHashes(referenceCount);
        bool changedInput = false;
        for (unsigned frame = 0; frame < referenceCount; ++frame) {
            inputHashes[frame] = uploadFrame(frame, frame);
            if (frame && inputHashes[frame] != inputHashes[0]) changedInput = true;
            gpu.Begin(); record(gpu.commands.Get(), 0, frame); Complete(gpu);
            gpu.Begin(); record(gpu.commands.Get(), 1, frame); Complete(gpu);
            auto answers = readPair(frame % 2);
            if (frame < 2) {
                Save(path(frame ? L".direct.A1.rgba16f" : L".direct.A0.rgba16f"), answers[0]);
                Save(path(frame ? L".direct.B1.rgba16f" : L".direct.B0.rgba16f"), answers[1]);
            }
            if (sequence && frame == waves) Save(path(L".direct.last.A.rgba16f"), answers[0]);
            if (sequence && frame + 1 == waves) Save(path(L".direct.last.B.rgba16f"), answers[1]);
            referenceCsv << frame << ',' << inputHashes[frame] << ',' << Hash(motionFor(0, frame)) << ',' << Hash(motionFor(1, frame))
                << ',' << Hash(answers[0]) << ',' << Hash(answers[1]) << ',' << resetFor(frame) << '\n';
            references.Store(frame, std::move(answers));
        }
        referenceCsv.flush();
        if (!referenceCsv) throw std::runtime_error("Writing reference manifest failed.");
        if (!changedInput) throw std::runtime_error("All overlap inputs are identical; provide a moving/spatially detailed fixture.");
        if (sequence) references.cache.flush();

        // Sequential references must not seed the histories used by the paired
        // run. Destroy both finished features, then create a completely fresh pair.
        runtimeB.Release(); runtimeA->Release(); createFeatures();
        uploadFrame(0, 0);
        a.Begin(); record(a.commands.Get(), 0, 0, nullptr, a.queue.Get()); a.Close();
        Check(a.queue->Wait(gpu.fence.Get(), gpu.serial), "Compute A waits for initial upload");
        a.Submit(a.queue.Get()); a.Wait(); Validate(gpu.device.Get());
        const auto traceA = ModelKernelTrace::Collect(path(L".A.trace"));
        waitDirect(); gpu.Begin(); readA[0]->Copy(gpu.commands.Get()); Complete(gpu);
        auto firstA = readA[0]->Read(); Save(path(L".A0.rgba16f"), firstA);
        RequireEqual(firstA, references.Get(0, 0), "Initial COMPUTE A0");
        ModelKernelTrace::Report traceB{};
        if (!sequence) {
            b.Begin(); record(b.commands.Get(), 1, 0, nullptr, bQueue); b.Close();
            Check(bQueue->Wait(a.fence.Get(), a.serial), "Wait for priming A output");
            Check(bQueue->Wait(gpu.fence.Get(), gpu.serial), "Priming B waits for readback");
            b.Submit(bQueue); b.Wait(); Validate(gpu.device.Get());
            traceB = ModelKernelTrace::Collect(path(L".B.trace"));
            auto priming = readPair(0); Save(path(L".B0.rgba16f"), priming[1]);
            RequireEqual(priming[1], references.Get(0, 1), "Initial COMPUTE B0");
        }
        // Sequence mode intentionally has no extra B0 priming: B must see each
        // frame exactly once. B0 is traced in the first (excluded warmup) wave.
        LARGE_INTEGER qpcOrigin{}, qpcFrequency{}; QueryPerformanceCounter(&qpcOrigin); QueryPerformanceFrequency(&qpcFrequency);
        std::array<UINT64, 2> produced{a.serial, 0}, consumed{b.serial, 0};
        std::vector<Row> rows; rows.reserve(waves);
        std::ofstream csv(path(L".csv"));
        if (!csv) throw std::runtime_error("Cannot create overlap CSV.");
        csv << std::setprecision(12)
            << "wave,warmup,a_frame,b_frame,a_reset,b_reset,a_start_ms,a_end_ms,b_handoff_start_ms,b_handoff_end_ms,b_start_ms,b_end_ms,a_ms,b_ms,handoff_ms,interval_overlap_ms,nr_pair_makespan_ms,pair_makespan_ms,cpu_record_ms,cpu_evaluate_a_ms,cpu_evaluate_b_ms,record_submit_wait_ms,a_hash,b_hash,exact_direct_parity\n";
        for (unsigned wave = 0; wave < waves; ++wave) {
            Row row; row.wave = wave; row.warmup = wave < warmups; row.aFrame = wave + 1; row.bFrame = wave;
            const unsigned next = row.aFrame % 2, previous = row.bFrame % 2;
            const auto inputHash = uploadFrame(row.aFrame, row.bFrame);
            if (inputHash != inputHashes[sequence ? row.aFrame : next])
                throw std::runtime_error("Input changed between reference and COMPUTE replay.");
            const double wallStart = ClockMs(); a.Begin(); b.Begin();
            const double recordStart = ClockMs();
            row.cpuEvaluateA = record(a.commands.Get(), 0, row.aFrame, &a);
            row.cpuEvaluateB = record(b.commands.Get(), 1, row.bFrame, &b, sequence && !wave ? bQueue : nullptr);
            a.Close(); b.Close(); row.cpuRecord = ClockMs() - recordStart;
            ComPtr<ID3D12Fence> gate;
            Check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)), "Create paired-wave gate");
            try {
                Check(a.queue->Wait(gate.Get(), 1), "Gate feature A queue");
                Check(a.queue->Wait(gpu.fence.Get(), gpu.serial), "A waits for frame snapshots");
                if (consumed[next]) Check(a.queue->Wait(b.fence.Get(), consumed[next]), "Wait for reusable A slot");
                const UINT64 aDone = a.Submit(a.queue.Get());
                Check(bQueue->Wait(gate.Get(), 1), "Gate feature B queue");
                Check(bQueue->Wait(gpu.fence.Get(), gpu.serial), "B waits for frame snapshots");
                Check(bQueue->Wait(a.fence.Get(), produced[previous]), "Wait for B frame producer");
                if (mode == L"serial") Check(bQueue->Wait(a.fence.Get(), aDone), "Serialize current A before B");
                const UINT64 bDone = b.Submit(bQueue);
                Check(gate->Signal(1), "Release paired-wave gate");
                a.Wait(); b.Wait(); produced[next] = aDone; consumed[previous] = bDone;
            } catch (...) {
                gate->Signal(1);
                try { a.Wait(); b.Wait(); } catch (...) { InFlightFailure(); }
                throw;
            }
            row.wall = ClockMs() - wallStart; Validate(gpu.device.Get());
            row.a = a.ReadTimes(a.queue.Get(), UINT64(qpcOrigin.QuadPart), UINT64(qpcFrequency.QuadPart));
            row.b = b.ReadTimes(bQueue, UINT64(qpcOrigin.QuadPart), UINT64(qpcFrequency.QuadPart));
            if (sequence && !wave) traceB = ModelKernelTrace::Collect(path(L".B.trace"));
            row.overlap = std::max(0.0, std::min(row.a.nr.end, row.b.nr.end) - std::max(row.a.nr.start, row.b.nr.start));
            row.nrMakespan = std::max(row.a.nr.end, row.b.nr.end) - std::min(row.a.nr.start, row.b.nr.start);
            row.makespan = std::max(row.a.nr.end, row.b.nr.end) - std::min(row.a.nr.start, mixed ? row.b.handoff.start : row.b.nr.start);
            const auto actual = readPair(next);
            row.aHash = Hash(actual[0]); row.bHash = Hash(actual[1]);
            if (!wave) {
                Save(path(L".first.A.rgba16f"), actual[0]); Save(path(L".first.B.rgba16f"), actual[1]);
                if (sequence) Save(path(L".B0.rgba16f"), actual[1]);
            }
            if (wave + 1 == waves) { Save(path(L".last.A.rgba16f"), actual[0]); Save(path(L".last.B.rgba16f"), actual[1]); }
            const auto expectedA = references.Get(row.aFrame, 0), expectedB = references.Get(row.bFrame, 1);
            const bool equal = actual[0] == expectedA && actual[1] == expectedB;
            csv << row.wave << ',' << row.warmup << ',' << row.aFrame << ',' << row.bFrame << ',' << resetFor(row.aFrame) << ',' << resetFor(row.bFrame) << ','
                << row.a.nr.start << ',' << row.a.nr.end << ',' << row.b.handoff.start << ',' << row.b.handoff.end << ',' << row.b.nr.start << ',' << row.b.nr.end << ','
                << row.a.nr.Duration() << ',' << row.b.nr.Duration() << ',' << (mixed ? row.b.handoff.Duration() : 0) << ',' << row.overlap << ',' << row.nrMakespan << ',' << row.makespan << ','
                << row.cpuRecord << ',' << row.cpuEvaluateA << ',' << row.cpuEvaluateB << ',' << row.wall << ',' << row.aHash << ',' << row.bHash << ',' << (equal ? "true" : "false") << '\n';
            csv.flush(); if (!csv) throw std::runtime_error("Writing overlap CSV failed.");
            if (!equal) {
                Save(path(L".failed.A.rgba16f"), actual[0]); Save(path(L".failed.B.rgba16f"), actual[1]);
                RequireEqual(actual[0], expectedA, "Paired COMPUTE A(next frame)");
                RequireEqual(actual[1], expectedB, "Paired COMPUTE B(previous frame)");
            }
            rows.push_back(row);
        }
        const auto stats = [&](auto value) {
            std::vector<double> values; for (const auto& row : rows) if (!row.warmup) values.push_back(value(row)); return Stats(std::move(values));
        };
        const auto aStats = stats([](const Row& r) { return r.a.nr.Duration(); });
        const auto bStats = stats([](const Row& r) { return r.b.nr.Duration(); });
        const auto handoffStats = stats([&](const Row& r) { return mixed ? r.b.handoff.Duration() : 0; });
        const auto fullStats = stats([](const Row& r) { return r.makespan; });
        const auto nrStats = stats([](const Row& r) { return r.nrMakespan; });
        const auto overlapStats = stats([](const Row& r) { return r.overlap; });
        const auto cpuStats = stats([](const Row& r) { return r.cpuRecord; });
        const auto wallStats = stats([](const Row& r) { return r.wall; });
        std::ofstream json(path(L".json")); json << std::setprecision(12);
        json << "{\n\"schema\":2,\"experiment\":\"paired-wave\",\"mode\":\"" << Narrow(mode)
            << "\",\"scope\":\"model-only NR plus interpass preparation; no native reconstruction, final composition or SR\""
            << ",\"width\":" << width << ",\"height\":" << height << ",\"b_width\":" << bWidth << ",\"b_height\":" << bHeight
            << ",\"history\":\"" << Narrow(history) << "\",\"source\":\"" << (directory ? "capture-directory" : sequence ? "synthetic-progressive-pan" : "alternating-pan") << "\""
            << ",\"guide_mode\":\"" << guideMode << "\",\"warmup_waves\":" << warmups << ",\"samples\":" << samples
            << ",\"independent_features\":2,\"fresh_features_after_reference\":true,\"compute_queues_used\":" << (mode == L"single" ? 1 : 2)
            << ",\"reset_each_evaluation\":" << (sequence ? "false" : "true") << ",\"tone_a\":1,\"tone_b\":1,\"direct_parity\":\"bit-exact-all-frames\""
            << ",\"interpass_filter\":\"" << (mixed ? "source-clamp-area-average-original-alpha" : "legacy-unmodified-model-output") << "\""
            << ",\"pair_makespan_mean_ms\":" << fullStats.mean << ",\"pair_makespan_median_ms\":" << fullStats.median
            << ",\"interval_overlap_mean_ms\":" << overlapStats.mean << ",\"a_mean_ms\":" << aStats.mean << ",\"b_mean_ms\":" << bStats.mean
            << ",\"cpu_record_mean_ms\":" << cpuStats.mean << ",\"record_submit_wait_mean_ms\":" << wallStats.mean
            << ",\"cpu_evaluate_a_mean_ms\":" << stats([](const Row& r) { return r.cpuEvaluateA; }).mean
            << ",\"cpu_evaluate_b_mean_ms\":" << stats([](const Row& r) { return r.cpuEvaluateB; }).mean
            << ",\"initial_trace_a_kernels\":" << traceA.kernels << ",\"initial_trace_b_kernels\":" << traceB.kernels
            << ",\"last_a_frame\":" << rows.back().aFrame << ",\"last_b_frame\":" << rows.back().bFrame
            << ",\"statistics\":{";
        bool first = true;
        const auto writeStats = [&](const char* name, const Stats& s) {
            if (!first) json << ','; first = false;
            json << '\"' << name << "\":{\"minimum_ms\":" << s.minimum << ",\"p10_ms\":" << s.p10 << ",\"mean_ms\":" << s.mean << ",\"median_ms\":" << s.median << '}';
        };
        writeStats("a_nr", aStats); writeStats("b_nr", bStats); writeStats("b_handoff", handoffStats);
        writeStats("full_pair", fullStats); writeStats("nr_pair", nrStats); writeStats("nr_interval_overlap", overlapStats);
        writeStats("cpu_record", cpuStats); writeStats("record_submit_wait", wallStats);
        json << "},\"percentile_method\":\"linear interpolation at p*(N-1), measured waves only\""
            << ",\"note\":\"B(frame N) and A(frame N+1), fresh gated submissions, both queues drained per wave. Handoff runs on B queue before its NR timestamp and is included in full_pair. Uploads, reference playback, readbacks and byte comparisons are excluded from paired-wave timings. This is latency, not steady throughput. Interval overlap does not prove physical kernel concurrency; compare full_pair against serial controls. Capture playback has no inferred geometry and explicitly uses zero guides.\"\n}\n";
        if (!json) throw std::runtime_error("Writing overlap JSON failed.");
        references.Finish();
        std::cout << std::format("Full pair minimum {:.4f}, p10 {:.4f}, mean {:.4f} ms; NR A mean {:.4f}, B mean {:.4f}, handoff mean {:.4f} ms.\n",
            fullStats.minimum, fullStats.p10, fullStats.mean, aStats.mean, bStats.mean, handoffStats.mean);
        std::cout << "All chronological A/B outputs bit-exact against fresh DIRECT references; paired-wave measurements exclude transfers.\n";
        runtimeB.Release(); runtimeA->Release(); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
