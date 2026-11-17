#include "Gpu.h"
namespace nr {
void Gpu::BeginTimings() {
    if (pendingTimes) throw std::runtime_error("Read completed GPU timings before starting another frame.");
    if (recordingTimes) throw std::runtime_error("GPU timings are already recording.");
    if (!timeQueries) {
        D3D12_QUERY_HEAP_DESC desc{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, unsigned(GpuStage::Count) * MaxTimingIntervals * 2, 0};
        Check(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&timeQueries)), "GPU stage timestamps");
        timeReadback = Buffer(desc.Count * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK);
        Check(queue->GetTimestampFrequency(&timeFrequency), "GPU timestamp frequency");
    }
    timed.fill(0); timingActive.fill(false); recordingTimes = true;
}
void Gpu::Time(GpuStage stage, bool end) {
    if (!recordingTimes) return;
    const auto index = unsigned(stage);
    if (index >= timed.size() || timed[index] >= MaxTimingIntervals)
        throw std::runtime_error("GPU stage timing capacity exceeded.");
    if (timingActive[index] != end)
        throw std::runtime_error("GPU stage timings must be paired and cannot nest the same stage.");
    const auto query = (index * MaxTimingIntervals + timed[index]) * 2 + unsigned(end);
    commands->EndQuery(timeQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query);
    timingActive[index] = !end;
    if (end) ++timed[index];
}
void Gpu::ResolveTimings() {
    if (!recordingTimes) return;
    // (late work holds timestamps too: they are read at the end of its list)
    auto* list = lateOpen && !lateActive ? lateCommands.Get() : commands.Get();
    for (unsigned i = 0; i < timed.size(); ++i) {
        if (timingActive[i]) throw std::runtime_error("GPU stage timing ended with an open interval.");
        if (timed[i]) {
            const auto query = i * MaxTimingIntervals * 2;
            list->ResolveQueryData(timeQueries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query, timed[i] * 2,
                timeReadback.Get(), query * sizeof(UINT64));
        }
    }
    recordingTimes = false; pendingTimes = true;
}
bool Gpu::TakeTimings(std::array<double, unsigned(GpuStage::Count)>& milliseconds) {
    if (!pendingTimes || lateOpen || !Complete()) return false;
    void* mapped = nullptr; D3D12_RANGE range{0, timed.size() * MaxTimingIntervals * 2 * sizeof(UINT64)};
    Check(timeReadback->Map(0, &range, &mapped), "GPU stage timing readback");
    const auto* stamps = static_cast<const UINT64*>(mapped);
    milliseconds.fill(0);
    for (unsigned i = 0; i < timed.size(); ++i) for (unsigned sample = 0; sample < timed[i]; ++sample) {
        const auto query = (i * MaxTimingIntervals + sample) * 2;
        milliseconds[i] += double(stamps[query + 1] - stamps[query]) * 1000.0 / timeFrequency;
    }
    D3D12_RANGE written{0, 0}; timeReadback->Unmap(0, &written); pendingTimes = false;
    return true;
}
}
