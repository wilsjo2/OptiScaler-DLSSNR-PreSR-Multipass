#include "ModelOverlap.h"
#include "Gpu.h"
#include "Runtime.h"
#include "ModelKernelTrace.h"
#include <d3d12sdklayers.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
using namespace nr;
using Pixels = std::vector<char>;

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
    gpu.Submit();
    if (!gpu.Drain(60000)) throw std::runtime_error("Direct setup/readback timeout.");
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

struct Interval { double start = 0, end = 0; };
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
        D3D12_QUERY_HEAP_DESC heap{}; heap.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; heap.Count = 2;
        Check(gpu.device->CreateQueryHeap(&heap, IID_PPV_ARGS(&queries)), "Overlap timestamps");
        readback = gpu.Buffer(2 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK);
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
    Interval ReadTimes(ID3D12CommandQueue* actualQueue, UINT64 qpcOrigin, UINT64 qpcFrequency) {
        UINT64 frequency = 0, calibratedGpu = 0, calibratedCpu = 0;
        Check(actualQueue->GetTimestampFrequency(&frequency), "Overlap timestamp frequency");
        // Calibration happens after execution. Subtract as double, never UINT64:
        // the recorded timestamps normally precede this calibration point.
        Check(actualQueue->GetClockCalibration(&calibratedGpu, &calibratedCpu), "Overlap clock calibration");
        UINT64* mapped = nullptr; D3D12_RANGE range{0, 2 * sizeof(UINT64)};
        Check(readback->Map(0, &range, reinterpret_cast<void**>(&mapped)), "Map overlap timestamps");
        const UINT64 begin = mapped[0], end = mapped[1]; D3D12_RANGE none{}; readback->Unmap(0, &none);
        if (!frequency || !begin || end < begin) throw std::runtime_error("Invalid overlap timestamps.");
        const auto convert = [&](UINT64 ticks) {
            return (double(calibratedCpu) - double(qpcOrigin)) * 1000.0 / double(qpcFrequency)
                + (double(ticks) - double(calibratedGpu)) * 1000.0 / double(frequency);
        };
        return {convert(begin), convert(end)};
    }
};
struct DrainOnExit {
    Compute& a; Compute& b;
    ~DrainOnExit() { try { a.Wait(); b.Wait(); } catch (...) {} }
};
struct Row {
    unsigned wave = 0, aFrame = 0, bFrame = 0;
    bool warmup = false;
    Interval a, b;
    double overlap = 0, makespan = 0, cpuRecord = 0, cpuEvaluateA = 0, cpuEvaluateB = 0, wall = 0;
    std::string aHash, bHash;
};
}

int RunModelOverlap(int argc, wchar_t** argv) {
    try {
        if (argc < 8 || argc > 9)
            throw std::runtime_error("Usage: NrModelProbe --overlap <runtime> <input.rgba16f> <width> <height> <output-prefix> <single|serial|parallel> [samples=40]");
        // argv is the complete process argument array, including --overlap.
        const UINT width = std::stoul(argv[4]), height = std::stoul(argv[5]);
        const std::filesystem::path prefix = argv[6];
        const std::wstring mode = argv[7];
        const unsigned samples = argc > 8 ? std::stoul(argv[8]) : 40;
        if (!width || !height || width > 8192 || height > 8192 || !samples || samples > 10000)
            throw std::runtime_error("Unsupported overlap extent/sample count.");
        if (mode != L"single" && mode != L"serial" && mode != L"parallel") throw std::runtime_error("Unknown overlap mode.");
        if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
        const auto path = [&](const wchar_t* suffix) { return std::filesystem::path(prefix.wstring() + suffix); };
        const size_t pixelCount = size_t(width) * height, bytes = pixelCount * 8;
        std::ifstream file(std::filesystem::path(argv[3]), std::ios::binary | std::ios::ate);
        if (!file || size_t(file.tellg()) != bytes) throw std::runtime_error("Expected tightly packed RGBA16F overlap input.");
        std::array<Pixels, 2> inputs{Pixels(bytes), Pixels(bytes)};
        file.seekg(0); file.read(inputs[0].data(), std::streamsize(bytes));
        if (!file) throw std::runtime_error("Overlap input read failed.");
        for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x)
            memcpy(inputs[1].data() + (size_t(y) * width + x) * 8,
                inputs[0].data() + (size_t((y + height - (3 % height)) % height) * width + (x + width - (7 % width)) % width) * 8, 8);
        if (inputs[0] == inputs[1]) throw std::runtime_error("Overlap fixture is unchanged by a (7,3) roll; use an image with spatial detail.");

        Gpu gpu;
        Runtime runtime(gpu.device.Get(), std::filesystem::absolute(argv[2]));
        DlssNr::ModelSettings settings; settings.preset = 1; settings.localTone = 1;
        std::array<std::optional<DlssNr::ModelSettings>, Settings::MaxPasses - 1> overrides{};
        overrides[0] = settings; // Later passes otherwise silently default to tone zero.
        gpu.Begin();
        Runtime::CheckResult(runtime.Create(gpu.commands.Get(), width, height, settings, 2, overrides), "Create independent overlap features");
        Complete(gpu);

        const auto texture = [&](DXGI_FORMAT format) { return gpu.Texture(width, height, format, D3D12_RESOURCE_STATE_COMMON); };
        std::array<ComPtr<ID3D12Resource>, 2> colours{texture(DXGI_FORMAT_R16G16B16A16_FLOAT), texture(DXGI_FORMAT_R16G16B16A16_FLOAT)};
        std::array<ComPtr<ID3D12Resource>, 2> intermediate{texture(DXGI_FORMAT_R16G16B16A16_FLOAT), texture(DXGI_FORMAT_R16G16B16A16_FLOAT)};
        std::array<ComPtr<ID3D12Resource>, 2> depths{texture(DXGI_FORMAT_R32_FLOAT), texture(DXGI_FORMAT_R32_FLOAT)};
        std::array<ComPtr<ID3D12Resource>, 2> motions{texture(DXGI_FORMAT_R16G16_FLOAT), texture(DXGI_FORMAT_R16G16_FLOAT)};
        auto output = texture(DXGI_FORMAT_R16G16B16A16_FLOAT);
        Transfer input0(gpu, colours[0].Get(), 8, true), input1(gpu, colours[1].Get(), 8, true);
        Transfer depth0(gpu, depths[0].Get(), 4, true), depth1(gpu, depths[1].Get(), 4, true);
        Transfer motion0(gpu, motions[0].Get(), 4, true), motion1(gpu, motions[1].Get(), 4, true);
        Transfer readA0(gpu, intermediate[0].Get(), 8, false), readA1(gpu, intermediate[1].Get(), 8, false), readB(gpu, output.Get(), 8, false);
        std::array<Transfer*, 2> readA{&readA0, &readA1};
        gpu.Begin(); input0.Upload(gpu.commands.Get(), inputs[0]); input1.Upload(gpu.commands.Get(), inputs[1]);
        const Pixels zeros(pixelCount * 4);
        for (auto* transfer : {&depth0, &depth1, &motion0, &motion1}) transfer->Upload(gpu.commands.Get(), zeros);
        Complete(gpu);
        Compute a(gpu), b(gpu);
        DrainOnExit drain{a, b};
        ID3D12CommandQueue* bQueue = mode == L"single" ? a.queue.Get() : b.queue.Get();
        const auto evaluate = [&](ID3D12GraphicsCommandList* commands, unsigned feature, ID3D12Resource* input, ID3D12Resource* target,
                                  Compute* timing = nullptr, ID3D12CommandQueue* traceQueue = nullptr) {
            Barrier(commands, input, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(commands, target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(commands, depths[feature].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Barrier(commands, motions[feature].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (timing) commands->EndQuery(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if (traceQueue) ModelKernelTrace::Begin(commands, traceQueue, false);
            const double start = ClockMs();
            runtime.Evaluate(commands, input, depths[feature].Get(), motions[feature].Get(), target, width, height, true, feature);
            const double elapsed = ClockMs() - start;
            if (traceQueue) ModelKernelTrace::End(commands);
            if (timing) {
                commands->EndQuery(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                commands->ResolveQueryData(timing->queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, timing->readback.Get(), 0);
            }
            Barrier(commands, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            Barrier(commands, input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            Barrier(commands, depths[feature].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            Barrier(commands, motions[feature].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            return elapsed;
        };
        const auto readPair = [&](unsigned slot) {
            Check(gpu.queue->Wait(a.fence.Get(), a.serial), "Direct readback waits for A");
            Check(gpu.queue->Wait(b.fence.Get(), b.serial), "Direct readback waits for B");
            gpu.Begin(); readA[slot]->Copy(gpu.commands.Get()); readB.Copy(gpu.commands.Get()); Complete(gpu);
            return std::array<Pixels, 2>{readA[slot]->Read(), readB.Read()};
        };

        // Serial DIRECT references test COMPUTE-list compatibility and distinguish
        // the two frames. They are outside all reported performance measurements.
        std::array<std::array<Pixels, 2>, 2> reference;
        for (unsigned variant = 0; variant < 2; ++variant) {
            gpu.Begin(); evaluate(gpu.commands.Get(), 0, colours[variant].Get(), intermediate[variant].Get()); Complete(gpu);
            gpu.Begin(); evaluate(gpu.commands.Get(), 1, intermediate[variant].Get(), output.Get()); Complete(gpu);
            reference[variant] = readPair(variant);
            Save(path(variant ? L".direct.A1.rgba16f" : L".direct.A0.rgba16f"), reference[variant][0]);
            Save(path(variant ? L".direct.B1.rgba16f" : L".direct.B0.rgba16f"), reference[variant][1]);
        }
        if (reference[0][0] == reference[1][0] || reference[0][1] == reference[1][1])
            throw std::runtime_error("Translated fixture produces indistinguishable NR outputs; frame-order validation is ineffective.");

        // Trace one evaluation from each independent feature, collecting each
        // only after its fence completes. The parameter captures expose distinct
        // writable counter/tensor arenas without tracing the timed paired waves.
        a.Begin(); evaluate(a.commands.Get(), 0, colours[0].Get(), intermediate[0].Get(), nullptr, a.queue.Get());
        a.Close(); Check(a.queue->Wait(gpu.fence.Get(), gpu.serial), "Compute A waits for DIRECT controls");
        a.Submit(a.queue.Get()); a.Wait(); Validate(gpu.device.Get());
        const auto traceA = ModelKernelTrace::Collect(path(L".A.trace"));
        b.Begin(); evaluate(b.commands.Get(), 1, intermediate[0].Get(), output.Get(), nullptr, bQueue);
        b.Close(); Check(bQueue->Wait(a.fence.Get(), a.serial), "Wait for priming A output");
        b.Submit(bQueue); b.Wait(); Validate(gpu.device.Get());
        const auto traceB = ModelKernelTrace::Collect(path(L".B.trace"));
        auto priming = readPair(0);
        Save(path(L".A0.rgba16f"), priming[0]); Save(path(L".B0.rgba16f"), priming[1]);
        RequireEqual(priming[0], reference[0][0], "Initial COMPUTE A0");
        RequireEqual(priming[1], reference[0][1], "Initial COMPUTE B0");

        LARGE_INTEGER qpcOrigin{}, qpcFrequency{}; QueryPerformanceCounter(&qpcOrigin); QueryPerformanceFrequency(&qpcFrequency);
        std::array<UINT64, 2> produced{a.serial, 0}, consumed{b.serial, 0};
        constexpr unsigned warmups = 6;
        std::vector<Row> rows; rows.reserve(warmups + samples);
        std::ofstream csv(path(L".csv"));
        if (!csv) throw std::runtime_error("Cannot create overlap CSV.");
        csv << std::setprecision(12)
            << "wave,warmup,a_frame,b_frame,a_start_ms,a_end_ms,b_start_ms,b_end_ms,a_ms,b_ms,interval_overlap_ms,pair_makespan_ms,cpu_record_ms,cpu_evaluate_a_ms,cpu_evaluate_b_ms,record_submit_wait_ms,a_hash,b_hash,exact_direct_parity\n";
        for (unsigned wave = 0; wave < warmups + samples; ++wave) {
            Row row; row.wave = wave; row.warmup = wave < warmups; row.aFrame = wave + 1; row.bFrame = wave;
            const unsigned next = row.aFrame % 2, previous = row.bFrame % 2;
            const double wallStart = ClockMs();
            a.Begin(); b.Begin();
            const double recordStart = ClockMs();
            row.cpuEvaluateA = evaluate(a.commands.Get(), 0, colours[next].Get(), intermediate[next].Get(), &a);
            row.cpuEvaluateB = evaluate(b.commands.Get(), 1, intermediate[previous].Get(), output.Get(), &b);
            // Finish all fallible command recording before queues can wait on a
            // CPU gate. A recording exception therefore cannot strand a queue.
            a.Close(); b.Close(); row.cpuRecord = ClockMs() - recordStart;
            ComPtr<ID3D12Fence> gate;
            Check(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)), "Create paired-wave gate");
            try {
                Check(a.queue->Wait(gate.Get(), 1), "Gate feature A queue");
                Check(a.queue->Wait(gpu.fence.Get(), gpu.serial), "A waits for completed readbacks");
                if (consumed[next]) Check(a.queue->Wait(b.fence.Get(), consumed[next]), "Wait for reusable A slot");
                const UINT64 aDone = a.Submit(a.queue.Get());
                Check(bQueue->Wait(gate.Get(), 1), "Gate feature B queue");
                Check(bQueue->Wait(gpu.fence.Get(), gpu.serial), "B waits for completed readbacks");
                Check(bQueue->Wait(a.fence.Get(), produced[previous]), "Wait for B frame producer");
                if (mode == L"serial") Check(bQueue->Wait(a.fence.Get(), aDone), "Serialize current A before B");
                const UINT64 bDone = b.Submit(bQueue);
                // Both real ExecuteCommandLists calls have now reached queues.
                Check(gate->Signal(1), "Release paired-wave gate");
                a.Wait(); b.Wait();
                produced[next] = aDone; consumed[previous] = bDone;
            } catch (...) {
                gate->Signal(1); // Also release a partially submitted pair.
                try { a.Wait(); b.Wait(); } catch (...) {}
                throw;
            }
            row.wall = ClockMs() - wallStart;
            Validate(gpu.device.Get());
            row.a = a.ReadTimes(a.queue.Get(), UINT64(qpcOrigin.QuadPart), UINT64(qpcFrequency.QuadPart));
            row.b = b.ReadTimes(bQueue, UINT64(qpcOrigin.QuadPart), UINT64(qpcFrequency.QuadPart));
            row.overlap = std::max(0.0, std::min(row.a.end, row.b.end) - std::max(row.a.start, row.b.start));
            row.makespan = std::max(row.a.end, row.b.end) - std::min(row.a.start, row.b.start);
            const auto actual = readPair(next);
            row.aHash = Hash(actual[0]); row.bHash = Hash(actual[1]);
            // Save first/final answers before checking, so a failed experiment
            // leaves concrete outputs for diagnosis rather than just a hash.
            if (!wave) { Save(path(L".first.A.rgba16f"), actual[0]); Save(path(L".first.B.rgba16f"), actual[1]); }
            if (wave + 1 == warmups + samples) { Save(path(L".last.A.rgba16f"), actual[0]); Save(path(L".last.B.rgba16f"), actual[1]); }
            RequireEqual(actual[0], reference[next][0], "Paired COMPUTE A(next frame)");
            RequireEqual(actual[1], reference[previous][1], "Paired COMPUTE B(previous frame)");
            csv << row.wave << ',' << row.warmup << ',' << row.aFrame << ',' << row.bFrame << ','
                << row.a.start << ',' << row.a.end << ',' << row.b.start << ',' << row.b.end << ','
                << row.a.end - row.a.start << ',' << row.b.end - row.b.start << ',' << row.overlap << ',' << row.makespan << ','
                << row.cpuRecord << ',' << row.cpuEvaluateA << ',' << row.cpuEvaluateB << ',' << row.wall << ','
                << row.aHash << ',' << row.bHash << ",true\n";
            csv.flush(); if (!csv) throw std::runtime_error("Writing overlap CSV failed.");
            rows.push_back(row);
        }
        const auto mean = [&](auto value) {
            double sum = 0; for (const auto& row : rows) if (!row.warmup) sum += value(row); return sum / samples;
        };
        std::vector<double> spans; for (const auto& row : rows) if (!row.warmup) spans.push_back(row.makespan);
        std::sort(spans.begin(), spans.end());
        const double median = (spans[(spans.size() - 1) / 2] + spans[spans.size() / 2]) * .5;
        const double pairMean = mean([](const Row& r) { return r.makespan; });
        const double overlapMean = mean([](const Row& r) { return r.overlap; });
        std::ofstream json(path(L".json"));
        json << std::setprecision(12) << "{\n\"schema\":1,\"experiment\":\"paired-wave\",\"mode\":\"" << Narrow(mode)
            << "\",\"width\":" << width << ",\"height\":" << height << ",\"warmup_waves\":" << warmups << ",\"samples\":" << samples
            << ",\"independent_features\":2,\"compute_queues_used\":" << (mode == L"single" ? 1 : 2)
            << ",\"reset_each_evaluation\":true,\"tone_a\":1,\"tone_b\":1,\"translation\":[7,3],\"direct_parity\":\"bit-exact-all-frames\""
            << ",\"a_mean_ms\":" << mean([](const Row& r) { return r.a.end - r.a.start; })
            << ",\"b_mean_ms\":" << mean([](const Row& r) { return r.b.end - r.b.start; })
            << ",\"interval_overlap_mean_ms\":" << overlapMean << ",\"pair_makespan_mean_ms\":" << pairMean
            << ",\"pair_makespan_median_ms\":" << median
            << ",\"cpu_record_mean_ms\":" << mean([](const Row& r) { return r.cpuRecord; })
            << ",\"cpu_evaluate_a_mean_ms\":" << mean([](const Row& r) { return r.cpuEvaluateA; })
            << ",\"cpu_evaluate_b_mean_ms\":" << mean([](const Row& r) { return r.cpuEvaluateB; })
            << ",\"record_submit_wait_mean_ms\":" << mean([](const Row& r) { return r.wall; })
            << ",\"initial_trace_a_kernels\":" << traceA.kernels << ",\"initial_trace_b_kernels\":" << traceB.kernels
            << ",\"last_a_frame\":" << rows.back().aFrame << ",\"last_b_frame\":" << rows.back().bFrame
            << ",\"input_hashes\":[\"" << Hash(inputs[0]) << "\",\"" << Hash(inputs[1]) << "\"],\"direct_a_hashes\":[\""
            << Hash(reference[0][0]) << "\",\"" << Hash(reference[1][0]) << "\"],\"direct_b_hashes\":[\""
            << Hash(reference[0][1]) << "\",\"" << Hash(reference[1][1]) << "\"]"
            << ",\"note\":\"B(frame N) and A(frame N+1); both lists submitted behind a fresh CPU-released gate. Every wave drains both queues before rerecording. Transfers and parity checks are outside NR timestamps and record/submit/wait duration. This is paired-wave latency, not steady-state throughput. Timestamp interval overlap does not prove physical kernel concurrency; compare pair makespan against serial controls.\"\n}\n";
        if (!json) throw std::runtime_error("Writing overlap JSON failed.");
        std::cout << std::format("NR paired-wave {}: {:.4f} ms mean / {:.4f} ms median makespan, {:.4f} ms timestamp-interval overlap; {} measured waves after {} warmup.\n",
            Narrow(mode), pairMean, median, overlapMean, samples, warmups);
        std::cout << "All A/B outputs bit-exact against alternating-frame DIRECT references. Timestamp overlap alone does not prove concurrent kernels; compare serial makespan.\n";
        runtime.Release(); return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
