// No FG presentation path may wait for render work gated by that same present.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include "../OptiScaler/dlssnr/DlssNr_FinishedReady.h"

using Microsoft::WRL::ComPtr;
static void Check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D12 call failed"); }
static void Expect(bool yes, const char* why) { if (!yes) throw std::runtime_error(why); }

static ComPtr<ID3D12Resource> Buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = sizeof(UINT);
    desc.Height = desc.SampleDesc.Count = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> resource;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                          IID_PPV_ARGS(&resource)));
    return resource;
}

int main() try
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
    Check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    D3D12_COMMAND_QUEUE_DESC desc {};
    ComPtr<ID3D12CommandQueue> render, present;
    Check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&render)));
    Check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&present)));
    ComPtr<ID3D12Fence> presentGate, inputReady;
    Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&presentGate)));
    Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&inputReady)));
    ComPtr<ID3D12Fence> composed;
    Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&composed)));
    auto upload = Buffer(device.Get(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto savedNr = Buffer(device.Get(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto gamePicture = Buffer(device.Get(), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComPtr<ID3D12CommandAllocator> producerAllocator, composeAllocator;
    ComPtr<ID3D12GraphicsCommandList> producerCommands, composeCommands;
    Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&producerAllocator)));
    Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&composeAllocator)));
    Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, producerAllocator.Get(), nullptr,
                                    IID_PPV_ARGS(&producerCommands)));
    Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, composeAllocator.Get(), nullptr,
                                    IID_PPV_ARGS(&composeCommands)));
    Check(producerCommands->Close());
    Check(composeCommands->Close());
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Expect(event != nullptr, "Could not create completion event");
    for (UINT64 frame = 1; frame <= 16; ++frame)
    {
        UINT* value = nullptr;
        Check(upload->Map(0, nullptr, reinterpret_cast<void**>(&value)));
        *value = static_cast<UINT>(frame);
        upload->Unmap(0, nullptr);
        Check(producerAllocator->Reset());
        Check(composeAllocator->Reset());
        Check(producerCommands->Reset(producerAllocator.Get(), nullptr));
        Check(composeCommands->Reset(composeAllocator.Get(), nullptr));
        producerCommands->CopyResource(savedNr.Get(), upload.Get());
        Check(producerCommands->Close());
        D3D12_RESOURCE_BARRIER barrier {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = { savedNr.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                               D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE };
        composeCommands->ResourceBarrier(1, &barrier);
        composeCommands->CopyResource(gamePicture.Get(), savedNr.Get());
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        composeCommands->ResourceBarrier(1, &barrier);
        Check(composeCommands->Close());
        // A submitted producer signal can sit behind a wait for this presentation.
        Check(render->Wait(presentGate.Get(), frame));
        ID3D12CommandList* producerLists[] = { producerCommands.Get() };
        render->ExecuteCommandLists(1, producerLists);
        Check(render->Signal(inputReady.Get(), frame));
        const auto completed = inputReady->GetCompletedValue();
        const bool crossQueueReady = DlssNr::FinishedInputReady(false, completed, frame);
        const bool sameQueueReady = DlssNr::FinishedInputReady(true, completed, frame);
        const bool olderInputReady = DlssNr::FinishedInputReady(false, completed, frame - 1);
        // The XeFG app-frame handoff uses the render queue. NR is eligible even
        // while the GPU is behind, and is consumed before the provider's Present.
        if (sameQueueReady)
        {
            ID3D12CommandList* composeLists[] = { composeCommands.Get() };
            render->ExecuteCommandLists(1, composeLists);
        }
        Check(render->Signal(composed.Get(), frame));
        // Do not enqueue present->Wait(inputReady, frame): that would deadlock these queues.
        Check(present->Signal(presentGate.Get(), frame));
        Check(composed->SetEventOnCompletion(frame, event));
        const auto result = WaitForSingleObject(event, 5000);
        if (result != WAIT_OBJECT_0) presentGate->Signal(frame); // cleanup even on regression failure
        Expect(result == WAIT_OBJECT_0, "Present/render queues stalled");
        Expect(!crossQueueReady, "Accepted future render input on the presentation queue");
        Expect(sameQueueReady, "Rejected ordered same-queue input");
        Check(gamePicture->Map(0, nullptr, reinterpret_cast<void**>(&value)));
        const UINT displayed = *value;
        gamePicture->Unmap(0, nullptr);
        Expect(displayed == frame, "The saved NR result did not reach the game picture");
        Expect(olderInputReady, "Rejected the previous completed input");
        Expect(DlssNr::FinishedInputReady(false, inputReady->GetCompletedValue(), frame),
               "Completed cross-queue input remained unavailable");
    }
    CloseHandle(event);
    Expect(!DlssNr::FinishedInputReady(false, UINT64_MAX, 1), "Accepted a removed device");
    Expect(!DlssNr::FinishedInputReady(true, UINT64_MAX, 1), "Same-queue bypassed device removal");
    puts("PASS finished-picture queue readiness: presentation dependency, same-queue game-picture transfer, completed input, device removal");
    return 0;
}
catch (const std::exception& e) { puts(e.what()); return 1; }
