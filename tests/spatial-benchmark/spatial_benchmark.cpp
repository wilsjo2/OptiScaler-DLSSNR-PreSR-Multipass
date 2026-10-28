// Offscreen, real-NGX 4K comparison. No game process or install is touched.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <nvsdk_ngx.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "../../OptiScaler/dlssnr/DlssNr_CompatibilityRuntime.h"
#include "../../OptiScaler/shaders/dlssnr/DlssNr_Spatial.h"
#include "../../OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_Shader.h"
#include "../../OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_guides_Shader.h"

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
constexpr UINT W = 3840, H = 2160;
void check(HRESULT hr, const char* text) { if (FAILED(hr)) { std::fprintf(stderr, "%s: %08X\n", text, unsigned(hr)); throw std::runtime_error(text); } }
void ngx(NVSDK_NGX_Result r, const char* text) { if (r != NVSDK_NGX_Result_Success) { std::fprintf(stderr, "%s: %08X\n", text, unsigned(r)); throw std::runtime_error(text); } }
void require(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
template<class T> T proc(HMODULE m, const char* name) { auto* p = GetProcAddress(m, name); require(p != nullptr, name); return reinterpret_cast<T>(p); }

void transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after}; cmd->ResourceBarrier(1, &b);
}
void uavBarrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = resource; cmd->ResourceBarrier(1, &b);
}
struct Gpu {
    ComPtr<IDXGIFactory1> factory; ComPtr<IDXGIAdapter1> adapter; ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> cmd; ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr; UINT64 serial = 0, frequency = 0;
    Gpu() {
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d{}; adapter->GetDesc1(&d);
            if (d.VendorId == 0x10de && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
                std::printf("GPU: %ls\n", d.Description); break;
            }
            adapter.Reset();
        }
        require(device != nullptr, "NVIDIA D3D12 adapter missing");
        D3D12_COMMAND_QUEUE_DESC q{}; check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)), "queue");
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&cmd)), "command list");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
        check(queue->GetTimestampFrequency(&frequency), "timestamp frequency");
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr); require(event != nullptr, "fence event");
    }
    ~Gpu() { if (event) CloseHandle(event); }
    void submit() {
        check(cmd->Close(), "close"); ID3D12CommandList* lists[] = {cmd.Get()}; queue->ExecuteCommandLists(1, lists);
        check(queue->Signal(fence.Get(), ++serial), "signal");
        check(fence->SetEventOnCompletion(serial, event), "fence completion");
        require(WaitForSingleObject(event, 30000) == WAIT_OBJECT_0, "GPU fence timeout");
        check(allocator->Reset(), "allocator reset"); check(cmd->Reset(allocator.Get(), nullptr), "list reset");
    }
    ComPtr<ID3D12Resource> texture(DXGI_FORMAT format, UINT w, UINT h) {
        D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w; d.Height = h; d.DepthOrArraySize = d.MipLevels = 1; d.Format = format;
        d.SampleDesc.Count = 1; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES p{}; p.Type = D3D12_HEAP_TYPE_DEFAULT; ComPtr<ID3D12Resource> r;
        check(device->CreateCommittedResource(&p, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                              nullptr, IID_PPV_ARGS(&r)), "texture"); return r;
    }
    ComPtr<ID3D12Resource> buffer(UINT64 bytes, D3D12_HEAP_TYPE type) {
        D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = bytes; d.Height = 1;
        d.DepthOrArraySize = d.MipLevels = 1; d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES p{}; p.Type = type; ComPtr<ID3D12Resource> r;
        auto state = type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ;
        check(device->CreateCommittedResource(&p, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)), "buffer"); return r;
    }
};

struct Shader {
    Gpu& gpu; ComPtr<ID3D12RootSignature> root; ComPtr<ID3D12PipelineState> color, guides;
    UINT stride;
    explicit Shader(Gpu& g) : gpu(g) {
        D3D12_DESCRIPTOR_RANGE ranges[3]{};
        ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0, 0, 0};
        ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 5};
        ranges[2] = {D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, 7};
        D3D12_ROOT_PARAMETER param{}; param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        param.DescriptorTable = {3, ranges}; param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxLOD = D3D12_FLOAT32_MAX; sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters = 1; desc.pParameters = &param;
        desc.NumStaticSamplers = 1; desc.pStaticSamplers = &sampler;
        ComPtr<ID3DBlob> blob, error;
        check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), "root serialization");
        check(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC p{}; p.pRootSignature = root.Get();
        p.CS = {dlssnr_spatial_cso, sizeof(dlssnr_spatial_cso)};
        check(gpu.device->CreateComputePipelineState(&p, IID_PPV_ARGS(&color)), "spatial color PSO");
        p.CS = {dlssnr_spatial_guides_cso, sizeof(dlssnr_spatial_guides_cso)};
        check(gpu.device->CreateComputePipelineState(&p, IID_PPV_ARGS(&guides)), "spatial guide PSO");
        stride = gpu.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    struct Binding {
        ComPtr<ID3D12DescriptorHeap> heap; ComPtr<ID3D12Resource> constants;
        D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
    };
    Binding bind(const DlssNr::Spatial::Constants& constants,
                 ID3D12Resource* t0, ID3D12Resource* t1, ID3D12Resource* t2,
                 ID3D12Resource* u0, ID3D12Resource* u1) {
        Binding b{}; D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.NumDescriptors = 8;
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&b.heap)), "spatial descriptors");
        auto base = b.heap->GetCPUDescriptorHandleForHeapStart(); b.gpu = b.heap->GetGPUDescriptorHandleForHeapStart();
        ID3D12Resource* srv[] = {t0, t1, t2, t0, t0};
        for (UINT i = 0; i < 5; ++i) { auto h = base; h.ptr += SIZE_T(i) * stride; gpu.device->CreateShaderResourceView(srv[i], nullptr, h); }
        ID3D12Resource* uav[] = {u0, u1 ? u1 : u0};
        for (UINT i = 0; i < 2; ++i) { auto h = base; h.ptr += SIZE_T(i + 5) * stride; gpu.device->CreateUnorderedAccessView(uav[i], nullptr, nullptr, h); }
        b.constants = gpu.buffer(256, D3D12_HEAP_TYPE_UPLOAD); void* mapped = nullptr;
        check(b.constants->Map(0, nullptr, &mapped), "map constants"); std::memcpy(mapped, &constants, sizeof(constants)); b.constants->Unmap(0, nullptr);
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{}; cbv.BufferLocation = b.constants->GetGPUVirtualAddress(); cbv.SizeInBytes = 256;
        auto h = base; h.ptr += SIZE_T(7) * stride; gpu.device->CreateConstantBufferView(&cbv, h); return b;
    }
    void dispatch(ID3D12GraphicsCommandList* cmd, const Binding& b, bool guide, UINT w, UINT h) {
        ID3D12DescriptorHeap* heaps[] = {b.heap.Get()}; cmd->SetDescriptorHeaps(1, heaps);
        cmd->SetComputeRootSignature(root.Get()); cmd->SetPipelineState(guide ? guides.Get() : color.Get());
        cmd->SetComputeRootDescriptorTable(0, b.gpu); cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
};

struct Scene {
    Gpu& gpu; std::array<ComPtr<ID3D12Resource>, 4> color;
    ComPtr<ID3D12Resource> depth, motion, exposure;
    explicit Scene(Gpu& g) : gpu(g) {
        for (auto& r : color) r = gpu.texture(DXGI_FORMAT_R16G16B16A16_FLOAT, W, H);
        depth = gpu.texture(DXGI_FORMAT_R32_FLOAT, W, H);
        motion = gpu.texture(DXGI_FORMAT_R16G16_FLOAT, W, H);
        exposure = gpu.texture(DXGI_FORMAT_R32_FLOAT, 1, 1);
        D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.NumDescriptors = 7; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; ComPtr<ID3D12DescriptorHeap> heap;
        check(gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "scene descriptors");
        const UINT stride = gpu.device->GetDescriptorHandleIncrementSize(hd.Type);
        std::array<ID3D12Resource*, 7> inputs{color[0].Get(), color[1].Get(), color[2].Get(), color[3].Get(), depth.Get(), motion.Get(), exposure.Get()};
        for (UINT i = 0; i < inputs.size(); ++i) { auto cpu = heap->GetCPUDescriptorHandleForHeapStart(); cpu.ptr += SIZE_T(i) * stride; gpu.device->CreateUnorderedAccessView(inputs[i], nullptr, nullptr, cpu); }
        ID3D12DescriptorHeap* heaps[] = {heap.Get()}; gpu.cmd->SetDescriptorHeaps(1, heaps);
        auto clear = [&](UINT index, float value, const D3D12_RECT* rect = nullptr) {
            auto cpu = heap->GetCPUDescriptorHandleForHeapStart(); cpu.ptr += SIZE_T(index) * stride;
            auto handle = heap->GetGPUDescriptorHandleForHeapStart(); handle.ptr += UINT64(index) * stride;
            const float v[]{value, value, value, 1};
            gpu.cmd->ClearUnorderedAccessViewFloat(handle, cpu, inputs[index], v, rect ? 1 : 0, rect);
        };
        for (UINT frame = 0; frame < color.size(); ++frame) {
            clear(frame, .36f);
            uavBarrier(gpu.cmd.Get(), color[frame].Get());
            // Repeated edges at many spatial frequencies plus a moving block across the centre.
            for (UINT y = 0; y < 12; ++y) for (UINT x = 0; x < 24; ++x) {
                D3D12_RECT r{LONG(x * W / 24), LONG(y * H / 12), LONG((x + 1) * W / 24), LONG((y + 1) * H / 12)};
                const float wave = .14f * std::sin(float(x) * .66f) + .11f * std::cos(float(y) * .91f);
                const float grid = ((x + y) & 1) ? .06f : -.06f;
                clear(frame, std::clamp(.47f + wave + grid, .08f, .9f), &r);
            }
            uavBarrier(gpu.cmd.Get(), color[frame].Get());
            D3D12_RECT moving{LONG(W / 2 - 210 + frame * 110), LONG(H / 2 - 170),
                              LONG(W / 2 + 130 + frame * 110), LONG(H / 2 + 170)};
            clear(frame, .91f, &moving);
            uavBarrier(gpu.cmd.Get(), color[frame].Get());
            transition(gpu.cmd.Get(), color[frame].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        clear(4, .5f); clear(5, 0); clear(6, 1);
        for (auto* r : {depth.Get(), motion.Get(), exposure.Get()})
            transition(gpu.cmd.Get(), r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.submit();
    }
};

std::array<float, 2> readFloatPixel(Gpu& gpu, ID3D12Resource* image, UINT x, UINT y) {
    const auto d = image->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &bytes);
    auto readback = gpu.buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    transition(gpu.cmd.Get(), image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{}; src.pResource = image; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = fp;
    gpu.cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(gpu.cmd.Get(), image, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); gpu.submit();
    unsigned char* mapped = nullptr; check(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "guide map");
    const UINT channels = d.Format == DXGI_FORMAT_R32G32_FLOAT ? 2 : 1;
    const auto* pixel = reinterpret_cast<const float*>(mapped + fp.Offset + size_t(y) * fp.Footprint.RowPitch + size_t(x) * channels * 4);
    std::array<float, 2> values{pixel[0], channels == 2 ? pixel[1] : 0}; readback->Unmap(0, nullptr); return values;
}

void verifyGuides(Gpu& gpu, Shader& shader) {
    DlssNr::Spatial::Settings settings{}; settings.enabled = true;
    const auto layout = DlssNr::Spatial::Build(settings, 64, 32, 1.0f);
    require(layout.active, "guide fixture layout");
    auto depth = gpu.texture(DXGI_FORMAT_R32_FLOAT, 80, 40);
    auto motion = gpu.texture(DXGI_FORMAT_R32G32_FLOAT, 80, 40);
    auto outDepth = gpu.texture(DXGI_FORMAT_R32_FLOAT, layout.modelW, layout.modelH);
    auto outMotion = gpu.texture(DXGI_FORMAT_R32G32_FLOAT, layout.modelW, layout.modelH);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.NumDescriptors = 2; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; ComPtr<ID3D12DescriptorHeap> heap;
    check(gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "guide source heap");
    const UINT stride = gpu.device->GetDescriptorHandleIncrementSize(hd.Type);
    for (UINT i = 0; i < 2; ++i) {
        auto cpu = heap->GetCPUDescriptorHandleForHeapStart(); cpu.ptr += SIZE_T(i) * stride;
        gpu.device->CreateUnorderedAccessView(i ? motion.Get() : depth.Get(), nullptr, nullptr, cpu);
    }
    ID3D12DescriptorHeap* heaps[]{heap.Get()}; gpu.cmd->SetDescriptorHeaps(1, heaps);
    auto clear = [&](UINT i, const float* value, const D3D12_RECT* rect) {
        auto cpu = heap->GetCPUDescriptorHandleForHeapStart(); cpu.ptr += SIZE_T(i) * stride;
        auto handle = heap->GetGPUDescriptorHandleForHeapStart(); handle.ptr += UINT64(i) * stride;
        gpu.cmd->ClearUnorderedAccessViewFloat(handle, cpu, i ? motion.Get() : depth.Get(), value, rect ? 1 : 0, rect);
    };
    const float depthBase[]{.1f, 0, 0, 0}, depthRegion[]{.72f, 0, 0, 0};
    const float zero[]{0, 0, 0, 0}, motionRegion[]{2, -1, 0, 0};
    const D3D12_RECT depthRect{7, 5, 71, 37}, motionRect{11, 3, 75, 35};
    clear(0, depthBase, nullptr); uavBarrier(gpu.cmd.Get(), depth.Get()); clear(0, depthRegion, &depthRect);
    clear(1, zero, nullptr); uavBarrier(gpu.cmd.Get(), motion.Get()); clear(1, motionRegion, &motionRect);
    transition(gpu.cmd.Get(), depth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(gpu.cmd.Get(), motion.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    DlssNr::GuideRegions regions{{7, 5, 64, 32}, {11, 3, 64, 32}};
    auto binding = shader.bind(DlssNr::Spatial::MakeConstants(layout, 101, regions, 1.25f, .75f, 64, 32),
                               depth.Get(), depth.Get(), motion.Get(), outDepth.Get(), outMotion.Get());
    shader.dispatch(gpu.cmd.Get(), binding, true, layout.modelW, layout.modelH); gpu.submit();
    const std::array<std::array<UINT, 2>, 3> points{{{0, 0}, {layout.modelW / 2, layout.modelH / 2}, {layout.modelW - 1, layout.modelH - 1}}};
    for (auto [x, y] : points) {
        const auto d = readFloatPixel(gpu, outDepth.Get(), x, y);
        const auto m = readFloatPixel(gpu, outMotion.Get(), x, y);
        const float nativeX = pw::Unpack(float(x) + .5f, layout.warp.x);
        const float nativeY = pw::Unpack(float(y) + .5f, layout.warp.y);
        const float expectedX = pw::Pack(nativeX + 2.5f, layout.warp.x) - pw::Pack(nativeX, layout.warp.x);
        const float expectedY = pw::Pack(nativeY - .75f, layout.warp.y) - pw::Pack(nativeY, layout.warp.y);
        if (std::abs(d[0] - .72f) > 1e-4f || std::abs(m[0] - expectedX) > .005f || std::abs(m[1] - expectedY) > .005f) {
            std::fprintf(stderr, "guide (%u,%u): depth %.5f motion %.5f %.5f expected %.5f %.5f\n",
                         x, y, d[0], m[0], m[1], expectedX, expectedY);
            throw std::runtime_error("typed spatial guide mismatch");
        }
    }
    std::puts("PASS: typed DX12 guide shader, offset subrects, scaled motion, boundary extrapolation");
}

struct Case {
    std::string name; float global; DlssNr::Spatial::Settings settings;
    DlssNr::Spatial::Layout layout; bool spatial;
    ComPtr<ID3D12Resource> packedColor, packedDepth, packedMotion, modelOutput, ordinaryProxy, ordinaryAnswer;
    Shader::Binding packColor, packGuides, unpack;
    ComPtr<ID3D12DescriptorHeap> clearHeap;
};

void dump(Gpu& gpu, ID3D12Resource* image, const fs::path& path) {
    const auto d = image->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &bytes);
    auto readback = gpu.buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    transition(gpu.cmd.Get(), image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{}; src.pResource = image; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = fp;
    gpu.cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(gpu.cmd.Get(), image, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); gpu.submit();
    unsigned char* mapped = nullptr; check(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "readback map");
    std::ofstream out(path, std::ios::binary); require(bool(out), "output file");
    const size_t rowBytes = size_t(d.Width) * 8;
    for (UINT y = 0; y < d.Height; ++y) out.write(reinterpret_cast<const char*>(mapped + fp.Offset + size_t(y) * fp.Footprint.RowPitch), rowBytes);
    readback->Unmap(0, nullptr); require(bool(out), "output write");
}

int wmain(int argc, wchar_t** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(argc == 5 || (argc == 6 && std::wstring(argv[5]) == L"--reverse"),
            "Usage: spatial_benchmark <nvngx.dll> <NR runtime directory> <output directory> <frames> [--reverse]");
    const fs::path outputDir = argv[3]; fs::create_directories(outputDir);
    const int frames = std::max(5, _wtoi(argv[4]));
    Gpu gpu; Scene scene(gpu); Shader shader(gpu); verifyGuides(gpu, shader);
    HMODULE driver = LoadLibraryW(argv[1]); require(driver != nullptr, "driver nvngx.dll load");
    using Init = NVSDK_NGX_Result(*)(unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version, const NVSDK_NGX_FeatureCommonInfo*);
    auto init = proc<Init>(driver, "NVSDK_NGX_D3D12_Init_Ext");
    auto allocate = proc<decltype(&NVSDK_NGX_D3D12_GetCapabilityParameters)>(driver, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    auto destroy = proc<decltype(&NVSDK_NGX_D3D12_DestroyParameters)>(driver, "NVSDK_NGX_D3D12_DestroyParameters");
    const wchar_t* paths[]{argv[2]}; NVSDK_NGX_FeatureCommonInfo common{}; common.PathListInfo.Path = paths; common.PathListInfo.Length = 1;
    ngx(init(0x24480451, L".", gpu.device.Get(), NVSDK_NGX_Version_API, &common), "NGX init");
    auto runtime = DlssNr::CompatibilityRuntime::Open(fs::path(argv[2]) / L"nvngx_dlssnr.dll", gpu.device.Get(), allocate, destroy);
    require(runtime != nullptr, "real NR runtime open");
    ComPtr<ID3D12QueryHeap> queries; D3D12_QUERY_HEAP_DESC qd{}; qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qd.Count = 4;
    check(gpu.device->CreateQueryHeap(&qd, IID_PPV_ARGS(&queries)), "timestamp heap");
    auto queryReadback = gpu.buffer(4 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK);
    std::ofstream csv(outputDir / "timings.csv");
    csv << "case,frame,pack_ms,nr_ms,unpack_ms,total_ms,model_w,model_h,ordinary_w,ordinary_h\n";
    DlssNr::Spatial::Settings spatial{}; spatial.enabled = true;
    DlssNr::Spatial::Settings uniform = spatial; uniform.workX = uniform.workY = 100;
    std::array<std::tuple<std::string, float, DlssNr::Spatial::Settings>, 4> cases{{
        {"native100", 1.0f, {}}, {"spatial100", 1.0f, spatial},
        {"uniform85", .85f, uniform}, {"spatial85", .85f, spatial}}};
    if (argc == 6) std::reverse(cases.begin(), cases.end());
    for (const auto& [name, global, settings] : cases) {
        auto layout = DlssNr::Spatial::Build(settings, W, H, global);
        // Production bypasses spatial mapping when work is 100%. The uniform comparator
        // forces the same shader's linear-map branch to perform its usual 85% downsample.
        if (name == "uniform85" && !layout.active) {
            layout.warp.x = pw::BuildAxis(W, W, layout.ordinaryW, .8f, 0, 0);
            layout.warp.y = pw::BuildAxis(H, H, layout.ordinaryH, .8f, 0, 0);
            layout.modelW = layout.ordinaryW; layout.modelH = layout.ordinaryH;
            layout.active = true; layout.reason = "benchmark linear resample";
        }
        if (settings.enabled && !layout.active) {
            std::fprintf(stderr, "layout %s rejected: %s\n", name.c_str(), layout.reason.c_str());
            throw std::runtime_error("invalid spatial layout");
        }
        const bool compressed = settings.enabled;
        const bool unpackNeeded = name == "spatial100" || name == "spatial85";
        const UINT mw = layout.modelW, mh = layout.modelH, ow = layout.ordinaryW, oh = layout.ordinaryH;
        std::printf("case %s: model %ux%u, ordinary %ux%u, spatial %d\n", name.c_str(), mw, mh, ow, oh, compressed);
        auto modelOutput = gpu.texture(DXGI_FORMAT_R16G16B16A16_FLOAT, mw, mh);
        auto packedColor = compressed ? gpu.texture(DXGI_FORMAT_R16G16B16A16_FLOAT, mw, mh) : ComPtr<ID3D12Resource>{};
        auto packedDepth = compressed ? gpu.texture(DXGI_FORMAT_R32_FLOAT, mw, mh) : ComPtr<ID3D12Resource>{};
        auto packedMotion = compressed ? gpu.texture(DXGI_FORMAT_R32G32_FLOAT, mw, mh) : ComPtr<ID3D12Resource>{};
        auto ordinaryProxy = unpackNeeded ? gpu.texture(DXGI_FORMAT_R16G16B16A16_FLOAT, ow, oh) : ComPtr<ID3D12Resource>{};
        auto ordinaryAnswer = unpackNeeded ? gpu.texture(DXGI_FORMAT_R16G16B16A16_FLOAT, ow, oh) : ComPtr<ID3D12Resource>{};
        DlssNr::GuideRegions regions{{0, 0, W, H}, {0, 0, W, H}};
        auto guides = compressed ? shader.bind(DlssNr::Spatial::MakeConstants(layout, 101, regions, 1, 1, W, H),
                                               scene.color[0].Get(), scene.depth.Get(), scene.motion.Get(), packedDepth.Get(), packedMotion.Get()) : Shader::Binding{};
        auto unpack = unpackNeeded ? shader.bind(DlssNr::Spatial::MakeConstants(layout, 102, regions, 1, 1, W, H),
                                               packedColor.Get(), modelOutput.Get(), scene.motion.Get(), ordinaryProxy.Get(), ordinaryAnswer.Get()) : Shader::Binding{};
        std::array<Shader::Binding, 4> packs;
        if (compressed) for (UINT i = 0; i < 4; ++i) packs[i] = shader.bind(DlssNr::Spatial::MakeConstants(layout, 100, regions, 1, 1, W, H),
                scene.color[i].Get(), scene.depth.Get(), scene.motion.Get(), packedColor.Get(), nullptr);
        NVSDK_NGX_Parameter* params = nullptr; ngx(allocate(&params), "parameter allocation");
        params->Set("DLSSNR.Enabled", 1u); params->Set("DLSSNR.Width", mw); params->Set("DLSSNR.Height", mh);
        params->Set("CreationNodeMask", 1u); params->Set("VisibilityNodeMask", 1u);
        params->Set("DLSSNR.Hint.Render.Preset", 0u); params->Set("DLSSNR.Intensity", 1.0f);
        params->Set("DLSSNR.Style", 0u); params->Set("DLSSNR.LocalStructureStrength", 1.0f);
        params->Set("DLSSNR.LocalToneStrength", 0.0f); params->Set("DLSSNR.SkinStructureStrength", -1.0f);
        params->Set("DLSSNR.UseAutoMask", 1u); params->Set("DLSSNR.UICorrection", 1u);
        NVSDK_NGX_Handle* feature = nullptr; ngx(runtime->Create(gpu.cmd.Get(), params, &feature), "real NR feature create"); gpu.submit();
        auto bindNr = [&](UINT index) {
            params->Set("DLSSNR.Color", compressed ? packedColor.Get() : scene.color[index].Get());
            params->Set("DLSSNR.Depth", compressed ? packedDepth.Get() : scene.depth.Get());
            params->Set("DLSSNR.MVec", compressed ? packedMotion.Get() : scene.motion.Get());
            params->Set("DLSSNR.Output", modelOutput.Get()); params->Set("DLSSNR.DepthInverted", 0u);
            params->Set("DLSSNR.MVecScaleX", 1.0f); params->Set("DLSSNR.MVecScaleY", 1.0f);
            for (const char* key : {"Color", "Output", "Depth", "MVec"}) {
                char k[80]; sprintf_s(k, "DLSSNR.%sSubrectBaseX", key); params->Set(k, 0u);
                sprintf_s(k, "DLSSNR.%sSubrectBaseY", key); params->Set(k, 0u);
                sprintf_s(k, "DLSSNR.%sSubrectWidth", key); params->Set(k, mw);
                sprintf_s(k, "DLSSNR.%sSubrectHeight", key); params->Set(k, mh);
            }
        };
        for (int frame = 0; frame < frames; ++frame) {
            const UINT index = UINT(frame % 4); bindNr(index); params->Set("DLSSNR.Reset", frame == 0 ? 1u : 0u);
            auto* cmd = gpu.cmd.Get(); cmd->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if (compressed) {
                shader.dispatch(cmd, packs[index], false, mw, mh); uavBarrier(cmd, packedColor.Get());
                shader.dispatch(cmd, guides, true, mw, mh);
                transition(cmd, packedColor.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(cmd, packedDepth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                transition(cmd, packedMotion.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            cmd->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            ngx(runtime->Evaluate(cmd, feature, params), "real NR evaluate");
            cmd->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
            if (unpackNeeded) {
                transition(cmd, modelOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                shader.dispatch(cmd, unpack, false, ow, oh);
                transition(cmd, modelOutput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                uavBarrier(cmd, ordinaryAnswer.Get());
            } else uavBarrier(cmd, modelOutput.Get());
            if (compressed) {
                transition(cmd, packedColor.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                transition(cmd, packedDepth.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                transition(cmd, packedMotion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            cmd->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
            cmd->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, queryReadback.Get(), 0);
            gpu.submit();
            UINT64* ticks = nullptr; check(queryReadback->Map(0, nullptr, reinterpret_cast<void**>(&ticks)), "timestamp map");
            const auto ms = [&](int a, int b) { return double(ticks[b] - ticks[a]) * 1000.0 / double(gpu.frequency); };
            csv << name << ',' << frame << ',' << ms(0, 1) << ',' << ms(1, 2) << ',' << ms(2, 3) << ',' << ms(0, 3)
                << ',' << mw << ',' << mh << ',' << ow << ',' << oh << '\n';
            queryReadback->Unmap(0, nullptr);
        }
        csv.flush();
        dump(gpu, unpackNeeded ? ordinaryAnswer.Get() : modelOutput.Get(), outputDir / (name + ".rgba16f"));
        ngx(runtime->Release(feature), "NR feature release"); ngx(destroy(params), "parameter destroy");
    }
    std::puts("PASS: four cases used real NVIDIA NR feature 18, spatial shaders, timestamps, and full image readback.");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
