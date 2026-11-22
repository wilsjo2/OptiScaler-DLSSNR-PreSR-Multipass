#include "CaptureQueue.h"
#include <CaptureConvert.h>
namespace nr {
CaptureQueue::CaptureQueue(Gpu& g) : gpu(g), copiedEvent(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {
    if (!copiedEvent.value) Check(HRESULT_FROM_WIN32(GetLastError()), "Capture copy event");
    ComPtr<ID3D11Device> base; ComPtr<ID3D11DeviceContext> immediate;
    Check(D3D11CreateDevice(g.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr, 0, D3D11_SDK_VERSION, &base, nullptr, &immediate), "Capture D3D11 device");
    Check(base.As(&device), "Capture device5"); Check(immediate.As(&context), "Capture context4");
    ComPtr<ID3D11Multithread> threading; Check(context.As(&threading), "Capture threading");
    threading->SetMultithreadProtected(TRUE);
    Check(g.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&copied12)), "Capture copy fence");
    Handle sharedFence;
    Check(g.device->CreateSharedHandle(copied12.Get(), nullptr, GENERIC_ALL, nullptr, &sharedFence.value), "Share copy fence");
    Check(device->OpenSharedFence(sharedFence.value, IID_PPV_ARGS(&copied11)), "Open copy fence");
    Check(device->CreateComputeShader(CaptureConvert, sizeof(CaptureConvert), nullptr, &convert), "Capture conversion shader");
    D3D11_BUFFER_DESC cb{}; cb.ByteWidth = 32; cb.Usage = D3D11_USAGE_DEFAULT; cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    Check(device->CreateBuffer(&cb, nullptr, &constants), "Capture conversion constants");
}
void CaptureQueue::Allocate(Slot& slot, UINT w, UINT h) {
    if (slot.width == w && slot.height == h) return;
    slot.uav.Reset(); slot.texture.Reset(); slot.resource.Reset();
    D3D11_TEXTURE2D_DESC desc{}; desc.Width = w; desc.Height = h;
    desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    Check(device->CreateTexture2D(&desc, nullptr, &slot.texture), "Capture queue surface");
    Check(device->CreateUnorderedAccessView(slot.texture.Get(), nullptr, &slot.uav), "Capture queue UAV");
    ComPtr<IDXGIResource1> resource; Check(slot.texture.As(&resource), "Shared capture resource");
    Handle shared;
    Check(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr, &shared.value), "Share capture surface");
    Check(gpu.device->OpenSharedHandle(shared.value, IID_PPV_ARGS(&slot.resource)), "Open capture surface");
    slot.width = w; slot.height = h;
}
void CaptureQueue::Publish(const CaptureFrame& frame) {
    const auto start = ClockMs();
    int index = -1;
    {
        std::lock_guard lock(mutex);
        for (int i = 0; i < int(slots.size()); ++i)
            if (i != reading && (index < 0 || slots[i].sequence < slots[index].sequence)) index = i;
        slots[index].writing = true; slots[index].sequence = 0;
    }
    auto& slot = slots[index];
    UINT w = frame.crop.right - frame.crop.left, h = frame.crop.bottom - frame.crop.top;
    if (frame.rotation == DXGI_MODE_ROTATION_ROTATE90 || frame.rotation == DXGI_MODE_ROTATION_ROTATE270) std::swap(w, h);
    Allocate(slot, w, h);
    D3D11_TEXTURE2D_DESC sourceDesc{}; frame.texture->GetDesc(&sourceDesc);
    ++copySerial;
    const bool fp16 = sourceDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (fp16 && frame.rotation == DXGI_MODE_ROTATION_IDENTITY) {
        context->CopySubresourceRegion(slot.texture.Get(), 0, 0, 0, 0, frame.texture.Get(), 0, &frame.crop);
    } else {
        if (!fp16 && sourceDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM && sourceDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM)
            throw std::runtime_error("Unsupported DXGI capture colour format; use Windows Graphics Capture.");
        // Duplication surfaces need not be shader-readable. Copy to an SRV when required.
        D3D11_TEXTURE2D_DESC current{}; if (readable) readable->GetDesc(&current);
        if (!readable || current.Width != sourceDesc.Width || current.Height != sourceDesc.Height || current.Format != sourceDesc.Format) {
            readableView.Reset(); readable.Reset();
            sourceDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE; sourceDesc.MiscFlags = 0;
            sourceDesc.Usage = D3D11_USAGE_DEFAULT; sourceDesc.CPUAccessFlags = 0;
            Check(device->CreateTexture2D(&sourceDesc, nullptr, &readable), "Readable duplication surface");
            Check(device->CreateShaderResourceView(readable.Get(), nullptr, &readableView), "Capture input SRV");
        }
        context->CopyResource(readable.Get(), frame.texture.Get());
        const UINT params[]{w, h, frame.crop.left, frame.crop.top, UINT(frame.rotation), UINT(!fp16), 0, 0};
        context->UpdateSubresource(constants.Get(), 0, nullptr, params, 0, 0);
        context->CSSetShader(convert.Get(), nullptr, 0); context->CSSetConstantBuffers(0, 1, constants.GetAddressOf());
        context->CSSetShaderResources(0, 1, readableView.GetAddressOf()); context->CSSetUnorderedAccessViews(0, 1, slot.uav.GetAddressOf(), nullptr);
        context->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        ID3D11ShaderResourceView* noSrv = nullptr; ID3D11UnorderedAccessView* noUav = nullptr;
        context->CSSetShaderResources(0, 1, &noSrv); context->CSSetUnorderedAccessViews(0, 1, &noUav, nullptr);
    }
    Check(context->Signal(copied11.Get(), copySerial), "Signal capture copy"); context->Flush();
    Check(copied12->SetEventOnCompletion(copySerial, copiedEvent.value), "Capture copy completion");
    if (WaitForSingleObject(copiedEvent.value, 5000) != WAIT_OBJECT_0)
        throw std::runtime_error("Capture GPU copy timed out.");
    Check(device->GetDeviceRemovedReason(), "Capture device removed");
    {
        std::lock_guard lock(mutex);
        slot.timestamp = frame.timestamp; slot.sequence = ++sequence; slot.writing = false;
    }
    copyMs = ClockMs() - start;
}
bool CaptureQueue::Next() { return Next(0, 0); }
bool CaptureQueue::Next(UINT w, UINT h) {
    if (!gpu.Complete()) throw std::runtime_error("Cannot replace capture input while processing uses it.");
    std::lock_guard lock(mutex);
    int newest = -1;
    for (int i = 0; i < int(slots.size()); ++i)
        if (!slots[i].writing && slots[i].sequence > consumed &&
            (newest < 0 || slots[i].sequence > slots[newest].sequence)) newest = i;
    if (newest < 0) return false;
    if (w && (slots[newest].width != w || slots[newest].height != h)) return false;
    reading = newest; consumed = slots[newest].sequence;
    width = slots[newest].width; height = slots[newest].height; timestamp = slots[newest].timestamp;
    return true;
}
ID3D12Resource* CaptureQueue::Input() const { return reading >= 0 ? slots[reading].resource.Get() : nullptr; }
}
