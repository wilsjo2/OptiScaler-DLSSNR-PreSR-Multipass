#pragma once
#include "Common.h"
#include <dlssnr/DlssNr_GpuLifetime.h>
namespace nr {
enum class GpuStage : unsigned { InputMotion, Encode, NR, PrivateDlss, EdgeCorrection, Resolve, HdrLabels, FgMotion, FgEncode, Count };
inline constexpr std::array<const char*, unsigned(GpuStage::Count)> GpuStageNames{
    "input motion", "encode", "NR", "private DLSS", "edge correction", "resolve", "HDR/labels", "FG motion", "FG encode"};
void Barrier(ID3D12GraphicsCommandList* commands, ID3D12Resource* resource,
             D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
class Gpu {
public:
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter4> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    DlssNr::GpuLifetime lifetime;
    UINT64 serial = 0;
    Gpu();
    bool Complete() const;
    void WaitForCompletion(HANDLE stop);
    bool Drain(DWORD timeout = 5000);
    void Begin();
    void Submit();
    void BeginTimings();
    void Time(GpuStage stage, bool end = false);
    void ResolveTimings();
    bool TakeTimings(std::array<double, unsigned(GpuStage::Count)>& milliseconds);
    ComPtr<ID3D12Resource> Texture(UINT width, UINT height, DXGI_FORMAT format,
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bool shared = false);
    ComPtr<ID3D12Resource> Buffer(UINT64 bytes, D3D12_HEAP_TYPE type);
private:
    Handle event;
    Handle frameEvent;
    UINT64 frameEventSerial = 0;
    ComPtr<ID3D12QueryHeap> timeQueries;
    ComPtr<ID3D12Resource> timeReadback;
    UINT64 timeFrequency = 0;
    // Independent passes repeat the same stages. Keep each interval instead of
    // overwriting the first pass's timestamps with the final composition pass.
    static constexpr unsigned MaxTimingIntervals = Settings::MaxPasses + 2;
    std::array<unsigned, unsigned(GpuStage::Count)> timed{};
    std::array<bool, unsigned(GpuStage::Count)> timingActive{};
    bool recordingTimes = false, pendingTimes = false;
};
}
