// Executes the NR re-apply shader on WARP with nonconstant 2D images and checks it against a CPU reference
// for runs of one and two re-applied passes: the HLSL source on D3D11 and the shipped DXIL blob on D3D12.
// No game, NGX model or GPU capture is needed.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using Pixel = std::array<float, 4>;
struct Image
{
    unsigned width, height;
    std::vector<Pixel> pixels;
    Image(unsigned w, unsigned h) : width(w), height(h), pixels(w * h) {}
};
static void Check(HRESULT hr)
{
    if (FAILED(hr)) throw std::runtime_error("D3D call failed: " + std::to_string(hr));
}
static float Saturate(float v) { return std::clamp(v, 0.0f, 1.0f); }

#include "../OptiScaler/shaders/dlssnr/precompile/dlssnr_reapply_Shader.h"

// Runs dlssnr_reapply_cso (what OptiScaler's DX12 path binds) with t0 = shown, t1 = answer, u0 = target.
static Image RunDx12(const Image& shown, const Image& answer, float gain)
{
    ComPtr<IDXGIFactory4> factory;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter> warp;
    Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));
    ComPtr<ID3D12Device> device;
    Check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

    D3D12_DESCRIPTOR_RANGE srvRange { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0 };
    D3D12_DESCRIPTOR_RANGE uavRange { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0 };
    D3D12_ROOT_PARAMETER params[3] {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[1].ParameterType = params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = { 1, &srvRange };
    params[2].DescriptorTable = { 1, &uavRange };
    D3D12_ROOT_SIGNATURE_DESC rootDesc { 3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };
    ComPtr<ID3DBlob> serialized, errors;
    Check(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors));
    ComPtr<ID3D12RootSignature> root;
    Check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                      IID_PPV_ARGS(&root)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc {};
    psoDesc.pRootSignature = root.Get();
    psoDesc.CS = { dlssnr_reapply_cso, sizeof(dlssnr_reapply_cso) };
    ComPtr<ID3D12PipelineState> pso;
    Check(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)));

    const D3D12_HEAP_PROPERTIES defaultHeap { D3D12_HEAP_TYPE_DEFAULT }, uploadHeap { D3D12_HEAP_TYPE_UPLOAD },
        readbackHeap { D3D12_HEAP_TYPE_READBACK };
    auto texture = [&](bool uav, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC d {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = answer.width;
        d.Height = answer.height;
        d.DepthOrArraySize = d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        d.SampleDesc.Count = 1;
        d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> resource;
        Check(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                              IID_PPV_ARGS(&resource)));
        return resource;
    };
    auto buffer = [&](const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC d {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = d.DepthOrArraySize = d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> resource;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                              IID_PPV_ARGS(&resource)));
        return resource;
    };
    const auto targetDesc = texture(true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint {};
    UINT64 bytes = 0;
    const auto desc = targetDesc->GetDesc();
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);

    ComPtr<ID3D12CommandAllocator> allocator;
    Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list;
    Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pso.Get(),
                                    IID_PPV_ARGS(&list)));
    auto barrier = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER b {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = { resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to };
        list->ResourceBarrier(1, &b);
    };

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3,
                                          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    ComPtr<ID3D12DescriptorHeap> heap;
    Check(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)));
    const auto step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    auto gpu = heap->GetGPUDescriptorHandleForHeapStart();

    std::array<ComPtr<ID3D12Resource>, 2> inputs, uploads;
    const Image* images[] = { &shown, &answer };
    for (unsigned i = 0; i < 2; ++i)
    {
        inputs[i] = texture(false, D3D12_RESOURCE_STATE_COPY_DEST);
        uploads[i] = buffer(uploadHeap, bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
        char* mapped = nullptr;
        Check(uploads[i]->Map(0, nullptr, reinterpret_cast<void**>(&mapped)));
        for (unsigned y = 0; y < answer.height; ++y)
            std::memcpy(mapped + footprint.Offset + y * footprint.Footprint.RowPitch,
                        images[i]->pixels.data() + y * answer.width, answer.width * sizeof(Pixel));
        uploads[i]->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst { inputs[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {} };
        D3D12_TEXTURE_COPY_LOCATION src { uploads[i].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
        src.PlacedFootprint = footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        barrier(inputs[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        device->CreateShaderResourceView(inputs[i].Get(), nullptr, { cpu.ptr + i * step });
    }
    device->CreateUnorderedAccessView(targetDesc.Get(), nullptr, nullptr, { cpu.ptr + 2 * step });

    struct alignas(256) Constants { unsigned mode; float gain; unsigned width, height; } constants {
        131, gain, answer.width, answer.height };
    const auto cbv = buffer(uploadHeap, sizeof(constants), D3D12_RESOURCE_STATE_GENERIC_READ);
    void* cbvData = nullptr;
    Check(cbv->Map(0, nullptr, &cbvData));
    std::memcpy(cbvData, &constants, sizeof(constants));
    cbv->Unmap(0, nullptr);

    ID3D12DescriptorHeap* heaps[] = { heap.Get() };
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(root.Get());
    list->SetComputeRootConstantBufferView(0, cbv->GetGPUVirtualAddress());
    list->SetComputeRootDescriptorTable(1, gpu);
    list->SetComputeRootDescriptorTable(2, { gpu.ptr + 2 * step });
    list->Dispatch((answer.width + 7) / 8, (answer.height + 7) / 8, 1);
    barrier(targetDesc.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto readback = buffer(readbackHeap, bytes, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst { readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {} };
    dst.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION src { targetDesc.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {} };
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Check(list->Close());

    D3D12_COMMAND_QUEUE_DESC queueDesc { D3D12_COMMAND_LIST_TYPE_DIRECT };
    ComPtr<ID3D12CommandQueue> queue;
    Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
    ID3D12CommandList* lists[] = { list.Get() };
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    Check(queue->Signal(fence.Get(), 1));
    const HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Check(fence->SetEventOnCompletion(1, done));
    WaitForSingleObject(done, INFINITE);
    CloseHandle(done);
    Check(device->GetDeviceRemovedReason());

    Image result(answer.width, answer.height);
    char* mapped = nullptr;
    Check(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)));
    for (unsigned y = 0; y < answer.height; ++y)
        std::memcpy(result.pixels.data() + y * answer.width,
                    mapped + footprint.Offset + y * footprint.Footprint.RowPitch, answer.width * sizeof(Pixel));
    readback->Unmap(0, nullptr);
    return result;
}

int main() try
{
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    Check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                           &device, nullptr, &context));
    ComPtr<ID3DBlob> code, errors;
    const HRESULT compiled = D3DCompileFromFile(
        L"OptiScaler/shaders/dlssnr/precompile/dlssnr_reapply.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS, 0, &code, &errors);
    if (errors) std::fputs(static_cast<const char*>(errors->GetBufferPointer()), stderr);
    Check(compiled);
    ComPtr<ID3D11ComputeShader> shader;
    Check(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader));

    auto run = [&](const Image& shown, const Image& answer, float gain)
    {
        std::array<ComPtr<ID3D11Texture2D>, 2> textures;
        std::array<ComPtr<ID3D11ShaderResourceView>, 2> views;
        const Image* inputs[] = {&shown, &answer};
        for (unsigned i = 0; i < 2; ++i)
        {
            D3D11_TEXTURE2D_DESC d{};
            d.Width = inputs[i]->width; d.Height = inputs[i]->height;
            d.MipLevels = d.ArraySize = d.SampleDesc.Count = 1;
            d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            D3D11_SUBRESOURCE_DATA initial{inputs[i]->pixels.data(), inputs[i]->width * sizeof(Pixel), 0};
            Check(device->CreateTexture2D(&d, &initial, &textures[i]));
            Check(device->CreateShaderResourceView(textures[i].Get(), nullptr, &views[i]));
        }
        D3D11_TEXTURE2D_DESC d{};
        d.Width = answer.width; d.Height = answer.height;
        d.MipLevels = d.ArraySize = d.SampleDesc.Count = 1;
        d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        ComPtr<ID3D11Texture2D> target, readback;
        Check(device->CreateTexture2D(&d, nullptr, &target));
        ComPtr<ID3D11UnorderedAccessView> targetView;
        Check(device->CreateUnorderedAccessView(target.Get(), nullptr, &targetView));
        d.BindFlags = 0; d.Usage = D3D11_USAGE_STAGING; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        Check(device->CreateTexture2D(&d, nullptr, &readback));
        // DlssNrMode_Reapply; the gain travels in the WhitePoint slot of DlssNrConstants.
        struct Constants { unsigned mode; float gain; unsigned width, height; } constants{
            131, gain, answer.width, answer.height};
        D3D11_BUFFER_DESC bufferDesc{};
        bufferDesc.ByteWidth = sizeof(constants); bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial{&constants, 0, 0};
        ComPtr<ID3D11Buffer> buffer;
        Check(device->CreateBuffer(&bufferDesc, &initial, &buffer));
        ID3D11ShaderResourceView* srvs[]{views[0].Get(), views[1].Get()};
        context->CSSetShader(shader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 2, srvs);
        context->CSSetUnorderedAccessViews(0, 1, targetView.GetAddressOf(), nullptr);
        context->CSSetConstantBuffers(0, 1, buffer.GetAddressOf());
        context->Dispatch((answer.width + 7) / 8, (answer.height + 7) / 8, 1);
        context->ClearState();
        context->CopyResource(readback.Get(), target.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(context->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped));
        Image result(answer.width, answer.height);
        for (unsigned y = 0; y < answer.height; ++y)
            std::memcpy(result.pixels.data() + y * answer.width,
                        static_cast<const char*>(mapped.pData) + y * mapped.RowPitch, answer.width * sizeof(Pixel));
        context->Unmap(readback.Get(), 0);
        return result;
    };

    // Odd, non-square dimensions exercise partial thread groups. The answer stays in the encoded [0, 1] range
    // (where r single re-applies equal one dispatch of gain r*g); edits are large enough to clip both ends.
    Image shown(19, 13), answer(19, 13);
    for (unsigned y = 0; y < shown.height; ++y) for (unsigned x = 0; x < shown.width; ++x)
    {
        const unsigned at = y * shown.width + x;
        auto& s = shown.pixels[at];
        auto& m = answer.pixels[at];
        s = {float(x) / (shown.width - 1), float(y) / (shown.height - 1), ((x / 2 + y / 2) % 2) ? .9f : .05f, 1};
        const bool disk = (int(x) - 9) * (int(x) - 9) + (int(y) - 6) * (int(y) - 6) <= 16;
        m = {Saturate(s[0] + (disk ? .3f : -.2f)), Saturate(s[1] + .25f * std::sin(float(x + 2 * y))),
             Saturate(s[2] + (((x + y) % 2) ? .35f : -.35f)), float(x + 2 * y) / 50};
    }
    // A non-finite input pixel must come back mid-grey, as the between-pass clamp does.
    shown.pixels[5 * shown.width + 4][0] = std::numeric_limits<float>::quiet_NaN();
    shown.pixels[7 * shown.width + 11][2] = std::numeric_limits<float>::infinity();

    const float g = 0.81f;
    float worst = 0;
    unsigned floored = 0, ceiled = 0, nonfinite = 0;
    for (unsigned check = 0; check < 4; ++check)
    {
        // r = 1, 2 from the HLSL source on D3D11, then r = 1, 2 from the shipped DXIL blob on D3D12.
        const unsigned r = 1 + check % 2;
        const auto got = check < 2 ? run(shown, answer, float(r) * g) : RunDx12(shown, answer, float(r) * g);
        for (unsigned at = 0; at < answer.pixels.size(); ++at)
        {
            const auto& s = shown.pixels[at];
            const auto& m = answer.pixels[at];
            for (unsigned c = 0; c < 4; ++c)
            {
                const float value = got.pixels[at][c];
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite output");
                float expected = m[c];
                if (c < 3)
                {
                    const float raw = m[c] + float(r) * g * (m[c] - s[c]);
                    expected = std::isfinite(raw) ? Saturate(raw) : .5f;
                    nonfinite += !std::isfinite(raw);
                    if (std::isfinite(s[c]))
                    {
                        // The closed form equals r sequential single re-applies.
                        float iterated = m[c];
                        for (unsigned i = 0; i < r; ++i)
                            iterated = Saturate(iterated + g * (m[c] - s[c]));
                        if (std::abs(iterated - expected) > 2e-6f)
                            throw std::runtime_error("Closed form disagrees with iterated re-apply");
                    }
                    floored += expected == 0;
                    ceiled += expected == 1;
                }
                const float error = std::abs(value - expected);
                worst = std::max(worst, error);
                if (error > 2e-6f)
                {
                    std::fprintf(stderr, "%s r=%u pixel %u channel %u: got %.8f expected %.8f\n",
                                 check < 2 ? "D3D11 source" : "D3D12 blob", r, at, c, value, expected);
                    throw std::runtime_error("Re-apply disagrees with the CPU reference");
                }
            }
        }
    }
    if (!floored || !ceiled || nonfinite != 8) throw std::runtime_error("Fixtures missed clip or non-finite cases");
    std::printf("Re-apply WARP smoke passed: HLSL source (D3D11) and shipped DXIL (D3D12), r=1 and r=2 at gain "
                "%.2f, clipping both ends, non-finite fallback; max error %.8f\n", g, worst);
    return 0;
}
catch (const std::exception& error)
{
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
