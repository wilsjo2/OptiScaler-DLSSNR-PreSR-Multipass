#include "Gpu.h"
namespace nr {
void Barrier(ID3D12GraphicsCommandList* c, ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    c->ResourceBarrier(1, &b);
}
Gpu::Gpu(bool warp) : event(CreateEventW(nullptr, FALSE, FALSE, nullptr)),
    frameEvent(CreateEventW(nullptr, TRUE, FALSE, nullptr)) {
    if (GetEnvironmentVariableW(L"NR_CAPTURE_DEBUG", nullptr, 0)) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    }
    if (!event.value || !frameEvent.value) Check(HRESULT_FROM_WIN32(GetLastError()), "Fence event");
    Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "DXGI factory");
    if (warp) {
        Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "WARP adapter");
        Check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "WARP device");
    }
    for (UINT index = 0; !warp; ++index) {
        ComPtr<IDXGIAdapter4> candidate;
        if (factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC3 description{}; candidate->GetDesc3(&description);
        if (description.VendorId != 0x10de || (description.Flags & DXGI_ADAPTER_FLAG3_SOFTWARE)) continue;
        if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
            adapter = candidate; Log("GPU: " + Narrow(description.Description)); break;
        }
    }
    if (!device) throw std::runtime_error("An NVIDIA D3D12 adapter is required.");
    D3D12_COMMAND_QUEUE_DESC q{}; Check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "Command queue");
    Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Fence");
    Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "Allocator");
    Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
        IID_PPV_ARGS(&commands)), "Command list");
    Check(commands->Close(), "Close initial list");
}
bool Gpu::Complete() const {
    const auto value = fence->GetCompletedValue();
    if (value == UINT64_MAX) Check(device->GetDeviceRemovedReason(), "GPU removed");
    return value >= serial;
}
void Gpu::WaitForCompletion(HANDLE stop) {
    if (Complete()) return;
    if (frameEventSerial != serial) {
        Check(ResetEvent(frameEvent.value) ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "Reset frame event");
        Check(fence->SetEventOnCompletion(serial, frameEvent.value), "Frame completion event");
        frameEventSerial = serial;
    }
    // GPU completion wakes us immediately; the timeout only bounds visibility checks.
    // Keep this event separate from Drain so a stopped frame cannot wake a later drain.
    HANDLE waits[]{stop, frameEvent.value};
    if (MsgWaitForMultipleObjectsEx(2, waits, 20, QS_ALLINPUT, MWMO_INPUTAVAILABLE) == WAIT_FAILED)
        Check(HRESULT_FROM_WIN32(GetLastError()), "Wait for frame completion");
}
bool Gpu::BeginLate() {
    if (!lateSplit) return false;
    if (lateActive) throw std::runtime_error("Late GPU work is already being recorded.");
    if (!lateCommands) {
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&lateAllocator)), "Late allocator");
        Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, lateAllocator.Get(), nullptr,
            IID_PPV_ARGS(&lateCommands)), "Late command list");
        Check(lateCommands->Close(), "Close initial late list");
    }
    if (!lateOpen) {
        // Begin has seen every submission complete, the last late list among them.
        Check(lateAllocator->Reset(), "Reset late allocator"); Check(lateCommands->Reset(lateAllocator.Get(), nullptr), "Reset late list");
        lifetime.ResetRecording(lateCommands.Get()); lifetime.Record(lateCommands.Get());
        lateOpen = true;
    }
    commands.Swap(lateCommands); lateActive = true;
    return true;
}
void Gpu::EndLate() {
    if (lateActive) { commands.Swap(lateCommands); lateActive = false; }
}
void Gpu::SubmitLate() {
    if (!lateOpen) return;
    if (!mainSubmitted) throw std::runtime_error("Cannot submit late work before its main list.");
    EndLate();
    Check(lateCommands->Close(), "Close late list"); ID3D12CommandList* lists[]{lateCommands.Get()};
    queue->ExecuteCommandLists(1, lists); lifetime.Submitted(queue.Get(), 1, lists);
    Check(queue->Signal(fence.Get(), ++serial), "Signal late submission");
    lateOpen = false;
}
bool Gpu::Drain(DWORD timeout) {
    // An exception may leave BOTH lists merely recorded. Never execute the late
    // half without its encode, motion, field-read and resource-state transitions.
    if (lateOpen && !mainSubmitted) {
        EndLate(); lateCommands->Close(); lifetime.ResetRecording(lateCommands.Get());
        lateOpen = false; recordingTimes = pendingTimes = false; timingActive.fill(false);
    } else SubmitLate(); // a submitted main list may still need its late half
    // Covers an external capture-copy wait even if no processing list followed it.
    Check(queue->Signal(fence.Get(), ++serial), "Drain queue signal");
    if (!Complete()) {
        Check(fence->SetEventOnCompletion(serial, event.value), "Fence completion event");
        if (WaitForSingleObject(event.value, timeout) != WAIT_OBJECT_0) return false;
    }
    lifetime.ResetRecording(commands.Get()); lifetime.Collect();
    return true;
}
void Gpu::Begin() {
    if (!Complete()) throw std::runtime_error("Cannot reuse unfinished frame resources.");
    if (lateOpen) throw std::runtime_error("Submit or drain pending late work before beginning another frame.");
    EndLate(); mainSubmitted = false;
    Check(allocator->Reset(), "Reset allocator"); Check(commands->Reset(allocator.Get(), nullptr), "Reset list");
    lifetime.ResetRecording(commands.Get()); lifetime.Collect(); lifetime.Record(commands.Get());
}
void Gpu::Submit() {
    Check(commands->Close(), "Close list"); ID3D12CommandList* lists[]{commands.Get()};
    queue->ExecuteCommandLists(1, lists); mainSubmitted = true; lifetime.Submitted(queue.Get(), 1, lists);
    Check(queue->Signal(fence.Get(), ++serial), "Signal submission");
}
ComPtr<ID3D12Resource> Gpu::Texture(UINT w, UINT h, DXGI_FORMAT format, D3D12_RESOURCE_STATES state, bool shared) {
    D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = w; d.Height = h;
    d.DepthOrArraySize = d.MipLevels = 1; d.Format = format; d.SampleDesc.Count = 1;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (shared) d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> resource;
    Check(device->CreateCommittedResource(&heap, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
        &d, state, nullptr, IID_PPV_ARGS(&resource)), "Texture"); return resource;
}
ComPtr<ID3D12Resource> Gpu::Buffer(UINT64 bytes, D3D12_HEAP_TYPE type) {
    D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = bytes;
    d.Height = d.DepthOrArraySize = d.MipLevels = 1; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = type; ComPtr<ID3D12Resource> resource;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
        type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&resource)), "Buffer"); return resource;
}
}
