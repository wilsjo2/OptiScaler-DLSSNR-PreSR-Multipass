#pragma once
#include <algorithm>

// Rolling window of paired GPU-timing samples (an outer/total measurement and an inner/"the
// interesting part" measurement) for a periodic vitals summary - mean and p99 (worst 1%, reported
// as a time, not converted to fps) over the whole window, not one arbitrary snapshot sample.
//
// Lives alongside GpuTime_Dx12/Dx11 (this same directory) rather than under any one feature's own
// folder: any subsystem that already measures its own GPU cost with GpuTime can drop its two
// per-frame numbers in here and get a real rolling summary for free, instead of every subsystem
// re-deriving "is this frame representative" on its own. First user: the DLSS-NR pass
// (DlssNr_Dx12_Status.cpp) - not the only intended one.
//
// Adapted from a lock-free ring-buffer/percentile pattern proven in
// ~/Dev/greenboost_all/greenboost_gaming's Vulkan layer (frame-time P99 -> 1% low FPS); this
// version reports whatever unit the caller already measures in (this project reports GPU cost in
// milliseconds, not a converted framerate) and uses std::sort since STL is available here - 256
// elements sorted once every few hundred frames is negligible, unlike the plain-C insertion sort
// that layer needed without one.
namespace OptiScaler
{
class RollingVitals
{
  public:
    static constexpr unsigned int kSize = 256;

    void Push(double totalMs, double innerMs)
    {
        const auto i = static_cast<unsigned int>(_head % kSize);
        _total[i] = totalMs;
        _inner[i] = innerMs;
        ++_head;
    }

    unsigned int Count() const { return _head < kSize ? static_cast<unsigned int>(_head) : kSize; }

    struct Summary
    {
        double totalMean = 0.0;
        double totalP99 = 0.0;
        double innerMean = 0.0;
        double innerP99 = 0.0;
        unsigned int sampleCount = 0;
    };

    Summary Compute() const
    {
        Summary s {};
        s.sampleCount = Count();
        if (s.sampleCount == 0)
            return s;

        double totalSorted[kSize];
        double innerSorted[kSize];
        double totalSum = 0.0, innerSum = 0.0;
        for (unsigned int i = 0; i < s.sampleCount; ++i)
        {
            totalSorted[i] = _total[i];
            innerSorted[i] = _inner[i];
            totalSum += _total[i];
            innerSum += _inner[i];
        }
        std::sort(totalSorted, totalSorted + s.sampleCount);
        std::sort(innerSorted, innerSorted + s.sampleCount);

        const auto idx99 = std::min(s.sampleCount - 1, s.sampleCount * 99u / 100u);
        s.totalMean = totalSum / s.sampleCount;
        s.innerMean = innerSum / s.sampleCount;
        s.totalP99 = totalSorted[idx99];
        s.innerP99 = innerSorted[idx99];
        return s;
    }

  private:
    double _total[kSize] {};
    double _inner[kSize] {};
    unsigned long long _head = 0;
};
}
