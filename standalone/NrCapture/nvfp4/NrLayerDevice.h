#pragma once
#include "NrWeightUpload.h"
#include <memory>
#include <stdexcept>

// The three NVFP4 layers share parameter layouts and scheduling across graphics APIs.
// Vulkan supplies its own allocations, command recording, and isolated layer state for a pass.
namespace NrLayer
{
struct Memory
{
    std::shared_ptr<void> owner;
    uint64_t address = 0;
    void* mapped = nullptr;
};
struct Context
{
    std::shared_ptr<void> net, swin, ffn;
    virtual Memory AllocateMemory(uint64_t bytes, bool upload) = 0;
    virtual void Barrier() = 0;

  protected:
    ~Context() = default;
};
inline thread_local Context* active = nullptr;
struct Scope
{
    Context* previous = active;
    explicit Scope(Context* context) { active = context; }
    ~Scope() { active = previous; }
};
template <class T> T& State(std::shared_ptr<void>& storage)
{
    if (!storage)
        storage = std::make_shared<T>();
    return *static_cast<T*>(storage.get());
}
struct Buffer
{
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    uint64_t bytes = 0;
    Memory memory;
    uint64_t Address() const { return memory.owner ? memory.address : resource->GetGPUVirtualAddress(); }
    explicit operator bool() const { return memory.owner || resource; }
};

template <class Create, class Fill> Buffer Upload(ID3D12Device* device, uint64_t bytes, Create create, Fill fill)
{
    Buffer buffer;
    buffer.bytes = bytes;
    if (active)
    {
        buffer.memory = active->AllocateMemory(bytes, true);
        fill(buffer.memory.mapped);
    }
    else
    {
        buffer.resource = create(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto upload = create(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        if (FAILED(upload->Map(0, nullptr, &mapped)))
            throw std::runtime_error("NVFP4 weight upload map failed");
        fill(mapped);
        upload->Unmap(0, nullptr);
        NrWeightUpload::Copy(device, buffer.resource.Get(), upload.Get(), bytes);
    }
    return buffer;
}
} // namespace NrLayer
