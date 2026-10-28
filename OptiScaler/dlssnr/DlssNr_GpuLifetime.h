#pragma once

#include <d3d12.h>
#include <functional>
#include <memory>

namespace DlssNr
{
// NR-local lifetime tracking. Record/submit/reset/retire/collect calls are serialized internally.
// Owners must still guard their resources and outlive all calls. Lock order when owned by NR:
// owner registry -> NR state -> tracker; retirement callbacks may re-enter on the calling thread.
class GpuLifetime
{
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    GpuLifetime();
    ~GpuLifetime();
    GpuLifetime(const GpuLifetime&) = delete;
    GpuLifetime& operator=(const GpuLifetime&) = delete;
    void Record(ID3D12GraphicsCommandList* commands);
    // Tracker must outlive the probe; discarded work never satisfies it.
    std::function<bool()> CompletionProbe(ID3D12GraphicsCommandList* commands);
    // One reusable monotonic fence per queue; aliases are normalized at every notification.
    // Only call after the real ExecuteCommandLists or a successful command-list Reset.
    void Submitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists);
    void ResetRecording(ID3D12CommandList* commands);
    // Start tracking a replacement resource set. Older recordings still receive submit/reset
    // notifications, but only recordings used again belong to the new generation.
    void BeginGeneration();
    // Unresolved callbacks, including their captured ownership, are retained at destruction.
    void Retire(std::function<void()> destroy);
    void Collect();
    bool Idle();
    // Retired owners only: completed submissions can no longer be replayed by this owner.
    // Unsubmitted recordings and failed/removed-device fences remain unresolved.
    void FinishSubmitted();
};
} // namespace DlssNr
