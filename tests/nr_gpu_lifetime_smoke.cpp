// Exercises production NR retirement on WARP, including delayed and replayed command lists.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdio>
#include <stdexcept>
#include <atomic>
#include <barrier>
#include <thread>
#include <vector>
#include <Util.h>
#include "../OptiScaler/dlssnr/DlssNr_GpuLifetime.h"
#include "../standalone/NrCapture/nvfp4/NrWeightUpload.h"
using Microsoft::WRL::ComPtr;
static void check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D12 call failed"); }
static void expect(bool yes, const char* why) { if (!yes) throw std::runtime_error(why); }
int main()
try
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
    check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC desc {};
    check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));
    check(commands->Close());
    ID3D12CommandList* lists[] { commands.Get() };
    ComPtr<ID3D12Fence> gate, drain;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)));
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&drain)));
    UINT64 serial = 0;
    auto waitOn = [&](ID3D12CommandQueue* submittedQueue)
    {
        check(submittedQueue->Signal(drain.Get(), ++serial));
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        check(drain->SetEventOnCompletion(serial, event));
        const auto result = WaitForSingleObject(event, 5000);
        CloseHandle(event);
        expect(result == WAIT_OBJECT_0, "GPU did not drain");
    };
    auto wait = [&] { waitOn(queue.Get()); };
    // Both native INT8 (COMMON) and the NVFP4/FP8 layers (UAV) read weights uploaded on a separate queue.
    {
        const auto buffer = [&](D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES state) {
            D3D12_HEAP_PROPERTIES heap {};
            heap.Type = heapType;
            D3D12_RESOURCE_DESC resource {};
            resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            resource.Width = 4;
            resource.Height = 1;
            resource.DepthOrArraySize = 1;
            resource.MipLevels = 1;
            resource.SampleDesc.Count = 1;
            resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (heapType == D3D12_HEAP_TYPE_DEFAULT) resource.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            ComPtr<ID3D12Resource> result;
            check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &resource, state, nullptr, IID_PPV_ARGS(&result)));
            return result;
        };
        auto source = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        unsigned marker = 0;
        for (auto state : { D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS }) {
            auto target = buffer(D3D12_HEAP_TYPE_DEFAULT, state);
            void* data = nullptr;
            check(source->Map(0, nullptr, &data));
            *static_cast<unsigned*>(data) = ++marker;
            source->Unmap(0, nullptr);
            NrWeightUpload::Copy(device.Get(), target.Get(), source.Get(), sizeof(marker), state);
            ComPtr<ID3D12GraphicsCommandList> copy;
            check(device->CreateCommandList(0, desc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&copy)));
            if (state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                D3D12_RESOURCE_BARRIER barrier {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition = { target.Get(), 0, state, D3D12_RESOURCE_STATE_COPY_SOURCE };
                copy->ResourceBarrier(1, &barrier);
            }
            copy->CopyBufferRegion(readback.Get(), 0, target.Get(), 0, sizeof(marker));
            check(copy->Close());
            ID3D12CommandList* copies[] { copy.Get() };
            queue->ExecuteCommandLists(1, copies);
            wait();
            check(readback->Map(0, nullptr, &data));
            expect(*static_cast<unsigned*>(data) == marker, "NR weight upload returned before the copy completed");
            readback->Unmap(0, nullptr);
        }
    }
    // Readiness must survive collection, but never accept discarded or unsubmitted work; and once submission
    // notifications are known to arrive, a recording that closed without one is reported as discarded.
    {
        using Recorded = DlssNr::GpuLifetime::Recorded;
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        auto completed = life.RecordingProbe(commands.Get());
        expect(completed() == Recorded::Untracked, "nothing can be told before the first submission notification");
        life.ResetRecording(commands.Get());
        expect(completed() == Recorded::Untracked, "a discard was claimed before any submission notification");
        life.Record(commands.Get());
        auto submitted = life.RecordingProbe(commands.Get());
        check(queue->Wait(gate.Get(), 100));
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        expect(submitted() == Recorded::Submitted, "pending GPU creation became ready");
        check(gate->Signal(100));
        wait();
        expect(submitted() == Recorded::Finished, "completed creation stayed blocked at fixed epoch");
        life.ResetRecording(commands.Get());
        life.Collect();
        expect(submitted() == Recorded::Finished, "collection lost completed creation proof");
        expect(completed() == Recorded::Discarded, "unrelated submission revived discarded creation");
        // Still being recorded, then reset without being executed, then released without being executed.
        life.Record(commands.Get());
        auto open = life.RecordingProbe(commands.Get());
        expect(open() == Recorded::Open, "an open recording was not reported as open");
        life.ResetRecording(commands.Get());
        life.Collect();
        expect(open() == Recorded::Discarded, "a reset, unexecuted recording was not reported as discarded");
        {
            ComPtr<ID3D12CommandAllocator> otherAllocator;
            ComPtr<ID3D12GraphicsCommandList> dropped;
            check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&otherAllocator)));
            check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, otherAllocator.Get(), nullptr,
                                            IID_PPV_ARGS(&dropped)));
            life.Record(dropped.Get());
            auto released = life.RecordingProbe(dropped.Get());
            expect(released() == Recorded::Open, "a new recording was not reported as open");
            check(dropped->Close());
            dropped.Reset();
            expect(released() == Recorded::Discarded, "a released, unexecuted recording was not reported as discarded");
        }
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)));
    }
    int released = 0;
    {
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        life.Retire([&] { ++released; });
        for (int epoch = 0; epoch < 100; ++epoch) life.Collect();
        expect(released == 0 && !life.Idle(), "unsubmitted work released by CPU progress");
        life.ResetRecording(commands.Get());
        expect(released == 1 && life.Idle(), "discarded recording not released");
    }
    {
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        check(queue->Wait(gate.Get(), 1));
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        life.Retire([&] { ++released; });
        life.ResetRecording(commands.Get());
        expect(released == 1 && !life.Idle(), "reset released submitted GPU work");
        check(gate->Signal(1));
        wait();
        life.Collect();
        expect(released == 2 && life.Idle(), "completed submitted work not released");
    }
    {
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        wait();
        life.Retire([&] { ++released; });
        expect(released == 2, "replayable closed list released after first submission");
        check(queue->Wait(gate.Get(), 2));
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        life.ResetRecording(commands.Get());
        expect(released == 2, "second execution lost its completion dependency");
        check(gate->Signal(2));
        wait();
        life.Collect();
        expect(released == 3, "replayed recording not released after completion");
    }
    {
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        life.Retire([&] { ++released; });
    }
    expect(released == 3, "destructor freed unsubmitted ownership");
    {
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        life.Submitted(nullptr, 1, lists); // failed completion proof
        life.ResetRecording(commands.Get());
        life.Retire([&] { ++released; });
        expect(!life.Idle(), "failed signal treated as completion");
    }
    expect(released == 3, "failed-signal ownership not abandoned");
    {
        // Alias mapping substitutes only the Streamline unwrap seam; fences/queues remain real.
        ComPtr<ID3D12GraphicsCommandList> wrapped;
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&wrapped)));
        check(wrapped->Close());
        TestUtil::wrapped = wrapped.Get();
        TestUtil::real = commands.Get();
        DlssNr::GpuLifetime life;
        life.Record(wrapped.Get());
        life.Record(commands.Get()); // same recording through the unwrapped identity
        life.Retire([&] { ++released; });
        queue->ExecuteCommandLists(1, lists);
        ID3D12CommandList* wrappedLists[] { wrapped.Get() };
        life.Submitted(queue.Get(), 1, wrappedLists);
        life.ResetRecording(wrapped.Get());
        wait();
        life.Collect();
        expect(released == 4 && life.Idle(), "wrapped submit/reset identity did not match");
        life.Record(wrapped.Get());
        life.Retire([&] { ++released; });
        life.ResetRecording(commands.Get());
        expect(released == 5 && life.Idle(), "unwrapped reset did not discard wrapped recording");
        TestUtil::wrapped = TestUtil::real = nullptr;
    }
    {
        ComPtr<ID3D12CommandQueue> secondQueue;
        check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&secondQueue)));
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        // Repeated submissions share one queue timeline; another queue needs its own dependency.
        for (int replay = 0; replay < 16; ++replay)
        {
            queue->ExecuteCommandLists(1, lists);
            life.Submitted(queue.Get(), 1, lists);
            wait();
        }
        check(secondQueue->Wait(gate.Get(), 3));
        secondQueue->ExecuteCommandLists(1, lists);
        life.Submitted(secondQueue.Get(), 1, lists);
        life.Retire([&] { ++released; });
        life.ResetRecording(commands.Get());
        expect(released == 5 && !life.Idle(), "first queue completion released second queue work");
        check(gate->Signal(3));
        waitOn(secondQueue.Get());
        life.Collect();
        expect(released == 6 && life.Idle(), "multiple queue dependencies did not complete");
    }
    {
        // Some engines dispose of command lists instead of resetting them. Destruction
        // must close recording ownership, while the submission fence still protects GPU use.
        DlssNr::GpuLifetime life;
        for (int cycle = 0; cycle < 64; ++cycle)
        {
            ComPtr<ID3D12GraphicsCommandList> temporary;
            check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&temporary)));
            check(temporary->Close());
            life.Record(temporary.Get());
            life.Retire([&] { ++released; });
            temporary.Reset();
            life.Collect();
            expect(life.Idle() && released == 7 + cycle, "destroyed unsubmitted list retained ownership");
        }
        ComPtr<ID3D12GraphicsCommandList> temporary;
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&temporary)));
        check(temporary->Close());
        life.Record(temporary.Get());
        check(queue->Wait(gate.Get(), 4));
        ID3D12CommandList* pending[] { temporary.Get() };
        queue->ExecuteCommandLists(1, pending);
        life.Submitted(queue.Get(), 1, pending);
        life.Retire([&] { ++released; });
        temporary.Reset();
        life.Collect();
        expect(released == 70 && !life.Idle(), "destroyed list released in-flight work");
        check(gate->Signal(4));
        wait();
        life.Collect();
        expect(released == 71 && life.Idle(), "destroyed submitted list retained completed work");
    }
    {
        DlssNr::GpuLifetime life;
        unsigned outer = 0, nested = 0;
        life.Retire([&]
        {
            expect(++outer == 1, "retirement callback re-entered itself");
            // NGX feature destruction can re-enter queue/reset hooks and retire more resources.
            // Force the retired vector to grow while the original destruction callback is active.
            for (unsigned i = 0; i < 64; ++i) life.Retire([&] { ++nested; });
            life.Collect();
            expect(!life.Idle(), "collector reported idle inside a destruction callback");
        });
        expect(outer == 1 && nested == 64 && life.Idle(), "reentrant retirement did not drain exactly once");
    }
    {
        // An unrelated open NR recording must not pin a retired private DLSS context.
        DlssNr::GpuLifetime common, privateDlss;
        ComPtr<ID3D12CommandAllocator> localAllocator;
        ComPtr<ID3D12GraphicsCommandList> unrelated, work;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&localAllocator)));
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, localAllocator.Get(), nullptr,
                                        IID_PPV_ARGS(&unrelated)));
        check(unrelated->Close());
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, localAllocator.Get(), nullptr,
                                        IID_PPV_ARGS(&work)));
        check(work->Close());
        common.Record(unrelated.Get()); common.Record(work.Get()); privateDlss.Record(work.Get());
        bool commonReleased = false, privateReleased = false;
        common.Retire([&] { commonReleased = true; });
        privateDlss.Retire([&] { privateReleased = true; });
        ID3D12CommandList* submitted[] { work.Get() };
        queue->ExecuteCommandLists(1, submitted);
        common.Submitted(queue.Get(), 1, submitted); privateDlss.Submitted(queue.Get(), 1, submitted);
        wait(); work.Reset(); common.Collect(); privateDlss.Collect();
        expect(privateReleased && privateDlss.Idle() && !commonReleased && !common.Idle(),
               "private DLSS retirement depends on unrelated NR recordings");
        unrelated.Reset(); common.Collect(); expect(commonReleased, "unrelated recording did not retire");
    }
    {
        // Starfield resets lists on worker threads while the NR render path records/collects.
        // Each worker owns a distinct list; only the production lifetime tracker is shared.
        constexpr unsigned workers = 4, cycles = 2000;
        DlssNr::GpuLifetime life;
        std::vector<ComPtr<ID3D12GraphicsCommandList>> work(workers);
        for (auto& list : work)
        {
            check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&list)));
            check(list->Close());
        }
        std::barrier start(workers + 1);
        std::atomic_uint callbacks { 0 }, finished { 0 };
        std::vector<std::jthread> threads;
        for (unsigned worker = 0; worker < workers; ++worker)
            threads.emplace_back([&, worker]
            {
                start.arrive_and_wait();
                for (unsigned cycle = 0; cycle < cycles; ++cycle)
                {
                    life.Record(work[worker].Get());
                    life.Retire([&] { ++callbacks; });
                    life.ResetRecording(work[worker].Get());
                }
                ++finished;
            });
        start.arrive_and_wait();
        while (finished != workers)
        {
            life.Collect();
            std::this_thread::yield();
        }
        threads.clear(); // join before checking final ownership
        life.Collect();
        expect(life.Idle() && callbacks == workers * cycles,
               "concurrent record/reset lost, duplicated or retained ownership");
    }
    {
        // Leaving a presentation path must not make its dormant list a dependency of
        // every later model rebuild. Old generations must still be safe to replay.
        DlssNr::GpuLifetime life;
        bool oldReleased = false;
        unsigned replacementsReleased = 0;
        life.Record(commands.Get());
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        wait();
        life.Retire([&] { oldReleased = true; });
        life.BeginGeneration();
        ComPtr<ID3D12CommandAllocator> localAllocator;
        ComPtr<ID3D12GraphicsCommandList> work;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&localAllocator)));
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, localAllocator.Get(), nullptr,
                                        IID_PPV_ARGS(&work)));
        check(work->Close());
        ID3D12CommandList* replacementLists[] { work.Get() };
        for (unsigned cycle = 0; cycle < 12; ++cycle)
        {
            check(localAllocator->Reset());
            check(work->Reset(localAllocator.Get(), nullptr));
            life.Record(work.Get());
            check(work->Close());
            queue->ExecuteCommandLists(1, replacementLists);
            life.Submitted(queue.Get(), 1, replacementLists);
            life.Retire([&] { ++replacementsReleased; });
            life.Retire([&] { ++replacementsReleased; }); // multiple resources in one generation
            life.BeginGeneration();
            wait();
            expect(replacementsReleased == cycle * 2, "replayable replacement freed prematurely");
            life.ResetRecording(work.Get());
            expect(replacementsReleased == (cycle + 1) * 2 && !oldReleased,
                   "dormant old generation pinned unrelated replacements");
        }
        check(queue->Wait(gate.Get(), 5));
        queue->ExecuteCommandLists(1, lists); // replay the original generation after newer ones retired
        life.Submitted(queue.Get(), 1, lists);
        life.ResetRecording(commands.Get()); // cancelled owned list, GPU work still pending
        expect(!oldReleased, "generation boundary lost the old submission fence");
        check(gate->Signal(5));
        wait(); life.Collect();
        expect(oldReleased && life.Idle(), "cancelled old generation did not drain");
    }
    {
        // One recording may genuinely reference both old and new resources.
        DlssNr::GpuLifetime life;
        unsigned generationsReleased = 0;
        life.Record(commands.Get());
        life.Retire([&] { ++generationsReleased; });
        life.BeginGeneration();
        life.Record(commands.Get());
        life.Retire([&] { ++generationsReleased; });
        life.BeginGeneration();
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        wait(); life.Collect();
        expect(generationsReleased == 0, "shared recording lost a generation dependency");
        life.ResetRecording(commands.Get());
        expect(generationsReleased == 2 && life.Idle(), "shared generations did not retire");
    }
    {
        // Abandoned retirements may own a compatibility runtime as well as raw GPU handles.
        // Destroying the tracker must retain those captures when completion is still unknown.
        ComPtr<ID3D12CommandAllocator> localAllocator;
        ComPtr<ID3D12GraphicsCommandList> abandoned;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&localAllocator)));
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, localAllocator.Get(), nullptr,
                                        IID_PPV_ARGS(&abandoned)));
        check(abandoned->Close());
        std::weak_ptr<int> captured;
        {
            auto owner = std::make_shared<int>(1);
            captured = owner;
            DlssNr::GpuLifetime life;
            life.Record(abandoned.Get());
            life.Retire([owner] {});
        }
        expect(!captured.expired(), "unfinished retirement destroyed its captured owner at teardown");
        abandoned.Reset();
        expect(!captured.expired(), "abandoned callback ownership was not retained");
        DlssNr::GpuLifetime completed;
        auto owner = std::make_shared<int>(2);
        captured = owner;
        completed.Retire([owner = std::move(owner)] {});
        expect(captured.expired(), "completed retirement retained its captured owner");
    }
    {
        // Shutdown may seal completed submitted lists, but not missing submissions or live GPU work.
        DlssNr::GpuLifetime life;
        bool destroyed = false;
        life.Record(commands.Get());
        life.Retire([&] { destroyed = true; });
        life.FinishSubmitted();
        expect(!destroyed && !life.Idle(), "shutdown discarded an unsubmitted recording");
        ComPtr<ID3D12Fence> blocked;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&blocked)));
        check(queue->Wait(blocked.Get(), 1));
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        life.FinishSubmitted();
        expect(!destroyed && !life.Idle(), "shutdown freed GPU work before completion");
        check(blocked->Signal(1));
        wait();
        expect(!life.Idle(), "ordinary collection lost replay protection");
        life.FinishSubmitted();
        expect(destroyed && life.Idle(), "shutdown could not reclaim the final completed recording");
    }
    {
        DlssNr::GpuLifetime life;
        life.Record(commands.Get());
        queue->ExecuteCommandLists(1, lists);
        life.Submitted(queue.Get(), 1, lists);
        wait();
        ComPtr<ID3D12Device5> removable;
        check(device.As(&removable));
        removable->RemoveDevice();
        life.FinishSubmitted();
        expect(!life.Idle(), "shutdown accepted a removed-device fence as GPU completion");
    }
    std::puts("NR GPU lifetime smoke passed (generations, shutdown, delayed/unsubmitted work, device removal)");
    return 0;
}
catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
