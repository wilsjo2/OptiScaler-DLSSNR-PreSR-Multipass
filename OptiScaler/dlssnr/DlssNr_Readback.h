#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <utility>

namespace DlssNr
{
inline bool CreateReadbackBuffer(ID3D12Device* device, UINT64 bytes, ID3D12Resource** result)
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(result)));
}

inline DXGI_FORMAT TypedReadbackFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    default:
        return format;
    }
}

// Owns one texture copy. The caller proves GPU completion before reading or destruction.
struct ReadbackImage
{
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout {};
    UINT64 bytes = 0;

    bool Allocate(ID3D12Device* device, D3D12_RESOURCE_DESC desc, UINT64 budget = UINT64_MAX)
    {
        desc.Format = TypedReadbackFormat(desc.Format);
        device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &bytes);
        return bytes && bytes <= budget && CreateReadbackBuffer(device, bytes, &readback);
    }

    void Copy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_BARRIER barrier {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = { source, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state,
                               D3D12_RESOURCE_STATE_COPY_SOURCE };
        if (state != D3D12_RESOURCE_STATE_COPY_SOURCE)
            cmd->ResourceBarrier(1, &barrier);
        D3D12_TEXTURE_COPY_LOCATION from {};
        from.pResource = source;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION to {};
        to.pResource = readback.Get();
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = layout;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        if (state != D3D12_RESOURCE_STATE_COPY_SOURCE)
            cmd->ResourceBarrier(1, &barrier);
    }

    bool Write(const std::filesystem::path& path) const
    {
        void* data = nullptr;
        D3D12_RANGE range { 0, SIZE_T(bytes) }, empty {};
        if (FAILED(readback->Map(0, &range, &data)))
            return false;
        std::ofstream file(path, std::ios::binary);
        file.write(static_cast<const char*>(data), std::streamsize(bytes));
        file.close();
        readback->Unmap(0, &empty);
        return !file.fail();
    }
};
} // namespace DlssNr
