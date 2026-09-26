#include "pch.h"
#include "DlssNr_Dx12_State.h"

auto DlssNr_Dx12::State::ParkNrResource(ID3D12Resource*& resource) -> void
{
    if (!resource)
        return;
    auto* retired = resource;
    resource = nullptr;
    lifetime.Retire([retired] { retired->Release(); });
}

auto DlssNr_Dx12::State::ReleaseSurfacesIfFormatChanged(DXGI_FORMAT modelFormat, DXGI_FORMAT nativeFormat) -> void
{
    if (nr.output == nullptr ||
        (nr.output->GetDesc().Format == modelFormat && nr.colorCopy && nr.hdrCopy &&
         nr.colorCopy->GetDesc().Format == nativeFormat && nr.hdrCopy->GetDesc().Format == nativeFormat))
        return;

    LOG_INFO("DLSS-NR rebuilding surfaces: model format {} -> {}, frame format {}", (int) nr.output->GetDesc().Format,
             (int) modelFormat, (int) nativeFormat);

    for (auto& model : nr.models)
        model.RetryAfterFailure();
    std::fill(std::begin(nr.passCreateFailed), std::end(nr.passCreateFailed), false);
    modelRunning = false;

    for (ID3D12Resource** r : { &nr.output, &nr.passScratch, &nr.passClamp, &nr.colorCopy, &nr.hdrCopy, &nr.colorSmall, &nr.depthSmall, &nr.motionSmall,
                                &nr.outputNative, &nr.activeColor })
        ParkNrResource(*r);

    nr.passScratchFailed = false;

    nr.reset = true;
}

void DlssNr_Dx12::State::ReleaseSpatialResources()
{
    for (auto** resource : { &nr.spatialColor, &nr.spatialDepth, &nr.spatialMotion, &nr.spatialProxy, &nr.spatialAnswer,
                             &nr.spatialProxyNative, &nr.spatialAnswerNative })
        ParkNrResource(*resource);
}

bool DlssNr_Dx12::State::PrepareSpatialResources(ID3D12Device* device, const DlssNr::Spatial::Layout& layout)
{
    const auto matches = [](ID3D12Resource* resource, DXGI_FORMAT format, unsigned w, unsigned h)
    {
        if (!resource)
            return false;
        const auto desc = resource->GetDesc();
        return desc.Format == format && desc.Width == w && desc.Height == h;
    };
    if (!matches(nr.spatialColor, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.modelW, layout.modelH) ||
        !matches(nr.spatialDepth, DXGI_FORMAT_R32_FLOAT, layout.modelW, layout.modelH) ||
        !matches(nr.spatialMotion, DXGI_FORMAT_R32G32_FLOAT, layout.modelW, layout.modelH) ||
        !matches(nr.spatialProxy, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ordinaryW, layout.ordinaryH) ||
        !matches(nr.spatialAnswer, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ordinaryW, layout.ordinaryH) ||
        (layout.globalScale > 1.0f &&
         (!matches(nr.spatialProxyNative, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.nativeW, layout.nativeH) ||
          !matches(nr.spatialAnswerNative, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.nativeW, layout.nativeH))))
    {
        ReleaseSpatialResources();
        nr.spatialColor = CreateScratch(device, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.modelW, layout.modelH);
        nr.spatialDepth = CreateScratch(device, DXGI_FORMAT_R32_FLOAT, layout.modelW, layout.modelH);
        nr.spatialMotion = CreateScratch(device, DXGI_FORMAT_R32G32_FLOAT, layout.modelW, layout.modelH);
        nr.spatialProxy = CreateScratch(device, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ordinaryW, layout.ordinaryH);
        nr.spatialAnswer = CreateScratch(device, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.ordinaryW, layout.ordinaryH);
        if (layout.globalScale > 1.0f)
        {
            nr.spatialProxyNative =
                CreateScratch(device, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.nativeW, layout.nativeH);
            nr.spatialAnswerNative =
                CreateScratch(device, DXGI_FORMAT_R16G16B16A16_FLOAT, layout.nativeW, layout.nativeH);
        }
    }
    const bool ready = nr.spatialColor && nr.spatialDepth && nr.spatialMotion && nr.spatialProxy && nr.spatialAnswer &&
                       (layout.globalScale <= 1.0f || (nr.spatialProxyNative && nr.spatialAnswerNative));
    if (!ready)
        ReleaseSpatialResources();
    return ready;
}

auto DlssNr_Dx12::State::CreateScratch(ID3D12Device* device, DXGI_FORMAT format, unsigned int width,
                                       unsigned int height) -> ID3D12Resource*
{
    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    // The model writes its result, so the destination has to be a UAV.
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* res = nullptr;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                    IID_PPV_ARGS(&res));
    return res;
}

auto DlssNr_Dx12::State::Barrier(ID3D12GraphicsCommandList* cmdList, ID3D12Resource* res, D3D12_RESOURCE_STATES from,
                                 D3D12_RESOURCE_STATES to) -> void
{
    if (from == to)
        return;
    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    cmdList->ResourceBarrier(1, &b);
}

auto DlssNr_Dx12::State::TypedGuideFormat(DXGI_FORMAT f) -> DXGI_FORMAT
{
    switch (f)
    {
    case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R24G8_TYPELESS:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default:
        return f;
    }
}

auto DlssNr_Dx12::State::IsTypeless(DXGI_FORMAT f) -> bool { return TypedGuideFormat(f) != f; }

auto DlssNr_Dx12::State::CreateGuideClone(ID3D12Device* device, ID3D12Resource* source) -> ID3D12Resource*
{
    D3D12_RESOURCE_DESC desc = source->GetDesc();
    desc.Format = TypedGuideFormat(desc.Format);
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    ID3D12Resource* res = nullptr;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                    IID_PPV_ARGS(&res));
    return res;
}

auto DlssNr_Dx12::State::ReadableGuide(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12Resource* source,
                                       ID3D12Resource** clone) -> ID3D12Resource*
{
    if (source == nullptr || !IsTypeless(source->GetDesc().Format))
        return source;

    // A dynamic-resolution game reallocates its depth and motion vectors as the render size moves, so
    // the clone made for the old size no longer matches -- and CopyResource demands identical
    // dimensions. Copying a 1970x1108 source into a 984x554 clone is undefined and removes the device,
    // which is the DRS crash. Rebuild the clone whenever the source's shape has changed under it.
    if (*clone != nullptr)
    {
        const D3D12_RESOURCE_DESC have = (*clone)->GetDesc();
        const D3D12_RESOURCE_DESC want = source->GetDesc();

        if (have.Width != want.Width || have.Height != want.Height || have.Format != TypedGuideFormat(want.Format))
        {
            // Retired, not released: the previous copy may still be in flight on the game's queue.
            ParkNrResource(*clone);
        }
    }

    if (*clone == nullptr)
    {
        *clone = CreateGuideClone(device, source);

        if (*clone == nullptr)
            return nullptr;

        LOG_DEBUG("DLSS-NR cloned a typeless guide as format {}", (int) TypedGuideFormat(source->GetDesc().Format));
    }

    Barrier(cmdList, source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmdList->CopyResource(*clone, source);
    Barrier(cmdList, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(cmdList, *clone, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return *clone;
}

auto DlssNr_Dx12::State::GetResource(NVSDK_NGX_Parameter* params, const char* a, const char* b) -> ID3D12Resource*
{
    // Preserve typed-key precedence; DX11/Vulkan bridges can supply untyped resources.
    for (const char* name : { a, b })
    {
        ID3D12Resource* resource = nullptr;
        if (params->Get(name, &resource) == NVSDK_NGX_Result_Success && resource)
            return resource;
    }
    for (const char* name : { a, b })
    {
        void* resource = nullptr;
        if (params->Get(name, &resource) == NVSDK_NGX_Result_Success && resource)
            return static_cast<ID3D12Resource*>(resource);
    }
    return nullptr;
}

void DlssNr_Dx12::State::ReleaseSupersamplers()
{
    for (auto** scaler : { &nr.superUp, &nr.superDown, &nr.spatialProxyDown })
        if (auto* retired = std::exchange(*scaler, nullptr))
            lifetime.Retire([retired] { delete retired; });
}

auto DlssNr_Dx12::State::ReleaseResources() -> void
{
    std::lock_guard<std::recursive_mutex> nrLock(mutex);
    ReleaseEnlarger();
    enlargementStatus.clear();
    deferredSr.ReleaseResources();

    lifetime.Collect();

    for (auto& model : nr.models)
        model.Release();
    std::fill(std::begin(nr.passCreateFailed), std::end(nr.passCreateFailed), false);
    modelRunning = false;

    for (auto** resource :
         { &nr.output, &nr.passScratch, &nr.passClamp, &nr.colorCopy, &nr.hdrCopy, &nr.activeColor, &nr.colorSmall, &nr.depthSmall, &nr.motionSmall })
        ParkNrResource(*resource);
    ReleaseSpatialResources();
    nr.spatialSignatureValid = false;
    nr.spatialFallback = false;
    nr.spatialActive = false;
    nr.passScratchFailed = false;

    ReleaseSupersamplers();

    ParkNrResource(nr.outputNative);

    ParkNrResource(nr.exposureMeter);
    ParkNrResource(nr.exposure);
    nr.exposureReadable = false;
    ParkNrResource(nr.heldColor);
    nr.heldActive = false;

    ParkNrResource(nr.depthClone);

    ParkNrResource(nr.motionClone);

    captureFrames.release();
    if (auto* timer = gpuTime.release())
        lifetime.Retire([timer] { delete timer; });
    if (auto* timer = ngxTime.release())
        lifetime.Retire([timer] { delete timer; });
    lastNgxTime.reset();
    lastGpuTime.reset();
}
