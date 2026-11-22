#include "ModelCompare.h"
#include <algorithm>
#include <cmath>
namespace nr {
ID3D12Resource* ModelCompare::Compose(ID3D12Resource* picture, ID3D12Resource* other, const Settings& s) {
    const auto size = picture->GetDesc(), otherSize = other->GetDesc();
    if (size.Width != otherSize.Width || size.Height != otherSize.Height || size.Format != otherSize.Format)
        throw std::runtime_error("Model comparison: the two pictures differ in size or format.");
    const UINT width = UINT(size.Width), height = size.Height;
    if (target) {
        const auto held = target->GetDesc();
        if (held.Width != size.Width || held.Height != size.Height || held.Format != size.Format) {
            // The GPU has finished every earlier list by the time a frame is recorded (the caller waits for it).
            target.Reset(); view.Reset();
        }
    }
    if (!target) {
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = width; desc.Height = height;
        desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.Format = size.Format; desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        Check(gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
            IID_PPV_ARGS(&target)), "Model comparison picture");
        D3D12_DESCRIPTOR_HEAP_DESC views{}; views.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; views.NumDescriptors = 1;
        Check(gpu.device->CreateDescriptorHeap(&views, IID_PPV_ARGS(&view)), "Model comparison view");
        gpu.device->CreateRenderTargetView(target.Get(), nullptr, view->GetCPUDescriptorHandleForHeapStart());
    }
    auto* c = gpu.commands.Get();
    Barrier(c, target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto* left = s.compareSwap ? picture : other;
    auto* right = s.compareSwap ? other : picture;
    const auto copy = [&](ID3D12Resource* source, UINT toX, UINT fromX, UINT count) {
        if (!count) return;
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = target.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from.pResource = source; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        const D3D12_BOX box{fromX, 0, 0, fromX + count, height, 1};
        c->CopyTextureRegion(&to, toX, 0, 0, &from, &box);
    };
    UINT border;
    if (s.compare == 3) { // Side by side: the same middle part of both, pixel for pixel.
        border = width / 2;
        const UINT rightCount = width - border, start = (width - rightCount) / 2;
        copy(left, 0, start, border);
        copy(right, border, start, rightCount);
    } else {              // Wipe.
        border = UINT(std::clamp(std::lround(double(s.compareSplit) * width), 0l, long(width)));
        copy(left, 0, 0, border);
        copy(right, border, border, width - border);
    }
    Barrier(c, target.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET);
    if (border > 0 && border < width) {
        const float grey[4]{0.5f, 0.5f, 0.5f, 1.0f};
        const D3D12_RECT line{LONG(border) - 1, 0, LONG(std::min(width, border + 1)), LONG(height)};
        c->ClearRenderTargetView(view->GetCPUDescriptorHandleForHeapStart(), grey, 1, &line);
    }
    Barrier(c, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    return target.Get();
}
}
