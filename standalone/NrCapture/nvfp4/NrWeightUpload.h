#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace NrWeightUpload
{
namespace detail
{
using Microsoft::WRL::ComPtr;
struct Transfer
{
    ID3D12Device* device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> source, target;
    std::unique_ptr<void, decltype(&CloseHandle)> event { nullptr, &CloseHandle };
    std::atomic<bool> failed = false;
    bool needsSignal = false;
};

struct Pending
{
    std::mutex mutex;
    std::list<std::shared_ptr<Transfer>> transfers;
};

inline Pending& Registry()
{
    // Unresolved GPU work must survive DLL teardown. Completed transfers leave the registry immediately.
    static auto* pending = new Pending;
    return *pending;
}

inline void Check(HRESULT result, const char* operation)
{
    if (FAILED(result)) throw std::runtime_error(std::string("NR weight upload: ") + operation + " failed (" +
                                                 std::to_string(static_cast<unsigned>(result)) + ")");
}
} // namespace detail

// A private upload queue can still be using the weights after all of the application's frame fences are done.
// Keep the source, destination and submission objects together until its own fence signals or the device is removed.
inline void Collect()
{
    std::list<std::shared_ptr<detail::Transfer>> completed;
    auto& pending = detail::Registry();
    std::lock_guard guard(pending.mutex);
    for (auto it = pending.transfers.begin(); it != pending.transfers.end();)
    {
        auto current = it++;
        if ((*current)->fence->GetCompletedValue() != 0)
            completed.splice(completed.end(), pending.transfers, current);
    }
    // Release the queue and resources after unlocking, since COM teardown can re-enter application code.
}

inline void Copy(ID3D12Device* device, ID3D12Resource* target, ID3D12Resource* source, UINT64 bytes,
                 D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
{
    auto& pending = detail::Registry();
    std::shared_ptr<detail::Transfer> previous;
    {
        std::lock_guard guard(pending.mutex);
        for (const auto& transfer : pending.transfers)
            if (transfer->device == device && transfer->failed && transfer->fence->GetCompletedValue() == 0)
            {
                previous = transfer;
                break;
            }
    }
    if (previous)
    {
        if (previous->needsSignal && SUCCEEDED(previous->queue->Signal(previous->fence.Get(), 1)))
            WaitForSingleObject(previous->event.get(), 20000);
        if (previous->fence->GetCompletedValue() == 0) throw std::runtime_error("The previous NR weight upload is still pending.");
    }
    Collect();
    auto transfer = std::make_shared<detail::Transfer>();
    transfer->device = device;
    transfer->source = source;
    transfer->target = target;
    D3D12_COMMAND_QUEUE_DESC queueDesc {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    detail::Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&transfer->queue)), "queue creation");
    detail::Check(device->CreateCommandAllocator(queueDesc.Type, IID_PPV_ARGS(&transfer->allocator)), "allocator creation");
    detail::Check(device->CreateCommandList(0, queueDesc.Type, transfer->allocator.Get(), nullptr, IID_PPV_ARGS(&transfer->list)), "list creation");
    detail::Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&transfer->fence)), "fence creation");
    transfer->event.reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!transfer->event) throw std::runtime_error("NR weight upload event creation failed.");
    auto* list = transfer->list.Get();
    D3D12_RESOURCE_BARRIER barrier {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = { target, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state, D3D12_RESOURCE_STATE_COPY_DEST };
    // COMMON buffers promote to COPY_DEST and decay at the end of the submission.
    const bool transition = state != D3D12_RESOURCE_STATE_COMMON && state != D3D12_RESOURCE_STATE_COPY_DEST;
    if (transition) list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(target, 0, source, 0, bytes);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    if (transition) list->ResourceBarrier(1, &barrier);
    detail::Check(list->Close(), "closing the command list");
    detail::Check(transfer->fence->SetEventOnCompletion(1, transfer->event.get()), "arming the fence");
    {
        std::lock_guard guard(pending.mutex);
        pending.transfers.push_back(transfer);
    }
    ID3D12CommandList* lists[] = { list };
    transfer->queue->ExecuteCommandLists(1, lists);
    transfer->needsSignal = FAILED(transfer->queue->Signal(transfer->fence.Get(), 1));
    if (!transfer->needsSignal) WaitForSingleObject(transfer->event.get(), 20000);
    const auto completed = transfer->fence->GetCompletedValue();
    if (!completed)
    {
        transfer->failed = true;
        throw std::runtime_error("The NR weight upload did not complete.");
    }
    Collect();
    if (completed == UINT64_MAX) throw std::runtime_error("Device removed during NR weight upload.");
}
} // namespace NrWeightUpload
