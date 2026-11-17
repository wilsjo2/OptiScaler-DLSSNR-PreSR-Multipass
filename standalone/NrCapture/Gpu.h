#pragma once
#include "Common.h"
#include <dlssnr/DlssNr_GpuLifetime.h>
namespace nr {
enum class GpuStage : unsigned { InputMotion, Encode, NR, PrivateDlss, EdgeCorrection, Resolve, HdrLabels, FgMotion, FgEncode, Transfer, Count };
inline constexpr std::array<const char*, unsigned(GpuStage::Count)> GpuStageNames{
    "input motion", "encode", "NR", "private DLSS", "edge correction", "resolve", "HDR/labels", "FG motion", "FG encode", "edit transfer"};
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
    explicit Gpu(bool warp = false); // WARP: shader-only fixtures without the NVIDIA driver or NGX.
    bool Complete() const;
    void WaitForCompletion(HANDLE stop);
    bool Drain(DWORD timeout = 5000);
    void Begin();
    void Submit();
    // A second list for work the frame's picture does not wait for (Processor, Settings::editLate). A host that
    // presents opts in with lateSplit and calls SubmitLate after presenting. Without it BeginLate answers false and
    // the caller records the work into the frame's own list, after the picture.
    bool lateSplit = false;
    bool BeginLate();
    void EndLate();
    void SubmitLate();
    void BeginTimings();
    void Time(GpuStage stage, bool end = false);
    void ResolveTimings();
    bool TakeTimings(std::array<double, unsigned(GpuStage::Count)>& milliseconds);
    ComPtr<ID3D12Resource> Texture(UINT width, UINT height, DXGI_FORMAT format,
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bool shared = false);
    ComPtr<ID3D12Resource> Buffer(UINT64 bytes, D3D12_HEAP_TYPE type);
private:
    ComPtr<ID3D12CommandAllocator> lateAllocator;
    ComPtr<ID3D12GraphicsCommandList> lateCommands; // (swapped with commands while late work is recorded)
    bool lateOpen = false, lateActive = false;
    bool mainSubmitted = false; // late work depends on this frame's main list
    Handle event;
    Handle frameEvent;
    UINT64 frameEventSerial = 0;
    ComPtr<ID3D12QueryHeap> timeQueries;
    ComPtr<ID3D12Resource> timeReadback;
    UINT64 timeFrequency = 0;
    // Independent passes repeat the same stages. Keep each interval instead of
    // overwriting the first pass's timestamps with the final composition pass.
    static constexpr unsigned MaxTimingIntervals = 2 * Settings::MaxPasses + 2; // (the edit transfer: two intervals a pass around private DLSS)
    std::array<unsigned, unsigned(GpuStage::Count)> timed{};
    std::array<bool, unsigned(GpuStage::Count)> timingActive{};
    bool recordingTimes = false, pendingTimes = false;
};
}
