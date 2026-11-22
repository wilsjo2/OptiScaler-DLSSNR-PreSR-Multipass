#pragma once
#include "CaptureBackend.h"
#include "Gpu.h"
#include <array>
#include <mutex>
namespace nr {
// Three owned GPU surfaces: one pinned by processing, two replaceable by capture.
// Only the newest published surface is consumed; a slow consumer cannot grow the queue.
class CaptureQueue {
public:
    explicit CaptureQueue(Gpu& gpu);
    ID3D11Device5* Device() const { return device.Get(); }
    void Publish(const CaptureFrame& frame); // Capture thread only; waits for its GPU copy, never NR.
    bool Next(); // Processing thread, only after its previous GPU submission is complete.
    bool Next(UINT width, UINT height); // As Next, taking the newest frame only if it has this size.
    ID3D12Resource* Input() const;
    UINT width = 0, height = 0;
    double timestamp = 0;
    double LastCopyMs() const { return copyMs; } // Producer only.
    bool CopiesComplete() const { return copied12->GetCompletedValue() >= copySerial; }
private:
    struct Slot {
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<ID3D12Resource> resource;
        ComPtr<ID3D11UnorderedAccessView> uav;
        uint64_t sequence = 0;
        double timestamp = 0;
        UINT width = 0, height = 0;
        bool writing = false;
    };
    Gpu& gpu;
    ComPtr<ID3D11Device5> device;
    ComPtr<ID3D11DeviceContext4> context;
    ComPtr<ID3D11Fence> copied11;
    ComPtr<ID3D12Fence> copied12;
    ComPtr<ID3D11ComputeShader> convert;
    ComPtr<ID3D11Buffer> constants;
    ComPtr<ID3D11Texture2D> readable;
    ComPtr<ID3D11ShaderResourceView> readableView;
    Handle copiedEvent;
    std::array<Slot, 3> slots;
    mutable std::mutex mutex;
    int reading = -1;
    uint64_t sequence = 0, consumed = 0, copySerial = 0;
    double copyMs = 0;
    void Allocate(Slot& slot, UINT width, UINT height);
};
}
