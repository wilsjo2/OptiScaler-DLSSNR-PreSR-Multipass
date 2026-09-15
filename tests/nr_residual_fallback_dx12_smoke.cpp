// Executes the exact embedded production DXIL on an NVIDIA GPU (no game/NGX loaded).
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "../OptiScaler/shaders/dlssnr/DlssNr_Common.h"
#include "../OptiScaler/shaders/dlssnr/precompile/DlssNr_Shader.h"
using Microsoft::WRL::ComPtr;
struct Pixel { float r,g,b,a; };
void check(HRESULT hr) { if (FAILED(hr)) { std::fprintf(stderr,"HRESULT %08X\n",(unsigned)hr); throw std::runtime_error("D3D12 failure"); } }
void expect(bool b,const char* text) { if (!b) throw std::runtime_error(text); }
bool same(Pixel a,Pixel b) { return std::abs(a.r-b.r)<1e-4f && std::abs(a.g-b.g)<1e-4f && std::abs(a.b-b.b)<1e-4f && a.a==b.a; }
int main() try {
    static_assert(sizeof(DlssNrConstants)==256);
    ComPtr<IDXGIFactory1> factory; check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<ID3D12Device> device; ComPtr<IDXGIAdapter1> adapter;
    for (unsigned i=0; factory->EnumAdapters1(i,&adapter)!=DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d {}; check(adapter->GetDesc1(&d));
        if (d.VendorId==0x10de && SUCCEEDED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)))) {
            std::wprintf(L"GPU: %ls\n",d.Description); break;
        }
        adapter.Reset();
    }
    expect(device!=nullptr,"No NVIDIA D3D12 adapter");
    ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC q {};
    check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> cmd;
    check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&cmd)));
    ComPtr<ID3D12Fence> fence; check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr); expect(event!=nullptr,"Fence event failed");
    UINT64 serial=0;
    auto submit=[&] {
        check(cmd->Close()); ID3D12CommandList* lists[]={cmd.Get()}; queue->ExecuteCommandLists(1,lists);
        check(queue->Signal(fence.Get(),++serial)); check(fence->SetEventOnCompletion(serial,event));
        expect(WaitForSingleObject(event,15000)==WAIT_OBJECT_0,"GPU completion timeout");
        check(allocator->Reset()); check(cmd->Reset(allocator.Get(),nullptr));
    };
    auto barrier=[&](ID3D12Resource* r,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b {}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; cmd->ResourceBarrier(1,&b);
    };
    auto buffer=[&](UINT64 bytes,D3D12_HEAP_TYPE type,D3D12_RESOURCE_STATES state) {
        D3D12_RESOURCE_DESC d {}; d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=bytes;
        d.Height=d.DepthOrArraySize=d.MipLevels=1; d.SampleDesc.Count=1; d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES h {}; h.Type=type; ComPtr<ID3D12Resource> r;
        check(device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,state,nullptr,IID_PPV_ARGS(&r))); return r;
    };
    auto texture=[&](unsigned width,DXGI_FORMAT format,bool output) {
        D3D12_RESOURCE_DESC d {}; d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width=width;
        d.Height=d.DepthOrArraySize=d.MipLevels=1; d.SampleDesc.Count=1; d.Format=format;
        if (output) d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES h {}; h.Type=D3D12_HEAP_TYPE_DEFAULT; ComPtr<ID3D12Resource> r;
        check(device->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,output?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
              D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&r))); return r;
    };
    D3D12_DESCRIPTOR_RANGE ranges[2] {};
    ranges[0]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,5,0,0,0}; ranges[1]={D3D12_DESCRIPTOR_RANGE_TYPE_UAV,2,0,0,0};
    D3D12_ROOT_PARAMETER parameters[3] {};
    parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameters[0].Descriptor={0,0};
    for (unsigned i=0;i<2;++i) { parameters[i+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; parameters[i+1].DescriptorTable={1,&ranges[i]}; }
    D3D12_STATIC_SAMPLER_DESC sampler {}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP; sampler.MaxLOD=D3D12_FLOAT32_MAX;
    D3D12_ROOT_SIGNATURE_DESC rootDesc {3,parameters,1,&sampler,D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> rootBytes,errors; check(D3D12SerializeRootSignature(&rootDesc,D3D_ROOT_SIGNATURE_VERSION_1,&rootBytes,&errors));
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0,rootBytes->GetBufferPointer(),rootBytes->GetBufferSize(),IID_PPV_ARGS(&root)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline {}; pipeline.pRootSignature=root.Get(); pipeline.CS={DlssNr_cso,sizeof(DlssNr_cso)};
    ComPtr<ID3D12PipelineState> pso; check(device->CreateComputePipelineState(&pipeline,IID_PPV_ARGS(&pso)));
    ComPtr<ID3D12DescriptorHeap> heap; D3D12_DESCRIPTOR_HEAP_DESC hd {};
    hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=7; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));
    const unsigned stride=device->GetDescriptorHandleIncrementSize(hd.Type);
    std::array<ComPtr<ID3D12Resource>,5> inputs,uploads;
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT,5> footprints {};
    for (unsigned i=0;i<5;++i) {
        inputs[i]=texture(i==4?1:2,i==4?DXGI_FORMAT_R8_UNORM:DXGI_FORMAT_R32G32B32A32_FLOAT,false);
        auto d=inputs[i]->GetDesc(); UINT64 bytes=0; device->GetCopyableFootprints(&d,0,1,0,&footprints[i],nullptr,nullptr,&bytes);
        uploads[i]=buffer(bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        auto cpu=heap->GetCPUDescriptorHandleForHeapStart(); cpu.ptr+=i*stride;
        D3D12_SHADER_RESOURCE_VIEW_DESC view {}; view.Format=d.Format; view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels=1;
        device->CreateShaderResourceView(inputs[i].Get(),&view,cpu);
    }
    auto output=texture(2,DXGI_FORMAT_R32G32B32A32_FLOAT,true);
    for (unsigned i=5;i<7;++i) { auto cpu=heap->GetCPUDescriptorHandleForHeapStart(); cpu.ptr+=i*stride; device->CreateUnorderedAccessView(output.Get(),nullptr,nullptr,cpu); }
    auto constants=buffer(256,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    DlssNrConstants* mappedConstants=nullptr; check(constants->Map(0,nullptr,(void**)&mappedConstants));
    *mappedConstants={}; mappedConstants->Mode=DlssNrMode_ApplyResidualFgFallback;
    mappedConstants->Width=2; mappedConstants->Height=1; mappedConstants->ExposurePreMul=1;
    auto outputDesc=output->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT outputFootprint {}; UINT64 readBytes=0;
    device->GetCopyableFootprints(&outputDesc,0,1,0,&outputFootprint,nullptr,nullptr,&readBytes);
    auto readback=buffer(readBytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    bool first=true;
    const std::array<Pixel,2> base {{{0.4f,0.3f,0.2f,0.25f},{0.6f,0.5f,0.4f,0.75f}}};
    const std::array<Pixel,2> carrier {{{0.6f,0.5f,0.4f,1},{0.5f,0.6f,0.5f,1}}};
    std::array<Pixel,2> motion {{{0,0,0,1},{0,0,0,1}}};
    auto run=[&](unsigned char rejected) {
        const void* data[]={base.data(),carrier.data(),carrier.data(),motion.data(),&rejected};
        for (unsigned i=0;i<5;++i) {
            void* mapped=nullptr; check(uploads[i]->Map(0,nullptr,&mapped)); std::memcpy(mapped,data[i],i==4?1:sizeof(base)); uploads[i]->Unmap(0,nullptr);
            if (!first) barrier(inputs[i].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION from {},to {}; from.pResource=uploads[i].Get(); from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint=footprints[i];
            to.pResource=inputs[i].Get(); to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            cmd->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            barrier(inputs[i].Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        first=false; ID3D12DescriptorHeap* heaps[]={heap.Get()}; cmd->SetDescriptorHeaps(1,heaps);
        cmd->SetComputeRootSignature(root.Get()); cmd->SetPipelineState(pso.Get());
        cmd->SetComputeRootConstantBufferView(0,constants->GetGPUVirtualAddress());
        auto gpu=heap->GetGPUDescriptorHandleForHeapStart(); cmd->SetComputeRootDescriptorTable(1,gpu); gpu.ptr+=5*stride; cmd->SetComputeRootDescriptorTable(2,gpu);
        cmd->Dispatch(1,1,1); barrier(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from {},to {}; from.pResource=output.Get(); from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.pResource=readback.Get(); to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint=outputFootprint;
        cmd->CopyTextureRegion(&to,0,0,0,&from,nullptr); barrier(output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        submit(); void* mapped=nullptr; check(readback->Map(0,nullptr,&mapped));
        std::array<Pixel,2> result; std::memcpy(result.data(),mapped,sizeof(result)); readback->Unmap(0,nullptr); return result;
    };
    const std::array<Pixel,2> expected {{{0.65f,0.3f,0,0.25f},{0.6f,0.75f,0.4f,0.75f}}};
    for (unsigned i=0;i<16;++i) { const auto result=run((unsigned char)(i%2)); expect(same(result[0],expected[0]) && same(result[1],expected[1]),"Static FG rejection pulses NR"); }
    motion={{{0.5f,0,0,1},{0.5f,0,0,1}}}; auto result=run(1);
    expect(same(result[0],{0.4f,0.55f,0.2f,0.25f}) && same(result[1],base[1]),"Midpoint motion/offscreen fallback failed");
    motion={{{NAN,0,0,1},{0,0,0,0}}}; result=run(1);
    expect(same(result[0],base[0]) && same(result[1],base[1]),"Invalid motion was reused");
    result=run(0); expect(same(result[0],expected[0]) && same(result[1],expected[1]),"Valid FG path changed");
    mappedConstants->Mode=DlssNrMode_ApplyReprojectedResidual;
    motion={{{0.5f,0,0,1},{0.5f,0,0,1}}}; result=run(0);
    expect(same(result[0],{0.4f,0.55f,0.2f,0.25f}) && same(result[1],base[1]),"Current-raster reprojection failed");
    motion={{{NAN,0,0,1},{0,0,0,0}}}; result=run(0);
    expect(same(result[0],base[0]) && same(result[1],base[1]),"Current-raster reprojection trusted invalid motion");
    constants->Unmap(0,nullptr); CloseHandle(event);
    std::puts("PASS: production DXIL, suppression fallback plus current-raster reprojection, motion/offscreen/invalid handling");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
