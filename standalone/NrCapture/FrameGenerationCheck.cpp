#include "FrameGeneration.h"
#include "FrameGenerationOutput.h"
#include "FramePacer.h"
#include "Runtime.h"
#include "Processor.h"
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <iostream>
#include <vector>
#include <cmath>
namespace nr {
void FrameGenerationPresentCheck(Gpu& gpu, FrameGeneration& fg, Processor* processor);
void FrameGenerationCheck(const std::filesystem::path& plugins, const std::filesystem::path& nrRuntime, bool relocatePlugin) {
    struct PluginRelocation {
        void* base = nullptr; SIZE_T size = 0;
        ~PluginRelocation() {
            // Keep a just-unloaded plugin's address range unavailable until process exit.
            // This exposes stale callbacks even when the loader would reuse its old base.
            if (base && VirtualAlloc(base, size, MEM_RESERVE, PAGE_NOACCESS))
                std::cout << "Reserved unloaded Streamline plugin range for restart stress\n" << std::flush;
        }
    } relocation;
    Settings settings; settings.maxFps = 120;
    for (unsigned multiplier = 1; multiplier <= 4; ++multiplier) {
        settings.frameGeneration = multiplier;
        if (std::abs(settings.ProcessingFps() - 120.0 / multiplier) > 1e-9) throw std::runtime_error("FG limiter division failed.");
    }
    settings.maxFps = 100; settings.frameGeneration = 3;
    FramePacer pacer; pacer.Submitted(settings.ProcessingFps(), 100);
    if (pacer.Due(settings.ProcessingFps(),129.99) || !pacer.Due(settings.ProcessingFps(),130.01))
        throw std::runtime_error("Fractional FG processing rate was rounded.");
    settings.maxFps = 0;
    if (settings.ProcessingFps() != 0 || !pacer.Due(settings.ProcessingFps(),0)) throw std::runtime_error("FG uncapped mode failed.");
    const auto ini = std::filesystem::temp_directory_path() / std::format(L"fg-check-{}.ini",GetCurrentProcessId());
    SaveSettings(settings,ini); const auto loaded = LoadSettings(ini); std::filesystem::remove(ini);
    if (loaded.frameGeneration != 3) throw std::runtime_error("FG setting persistence failed.");
    std::cout << "PASS output-target division: 120/2=60, 120/3=40, 120/4=30; fractional rates, uncapped and persistence\n" << std::flush;
    Gpu gpu;
    // Match the application's NGX-first initialization and reverse teardown order.
    std::unique_ptr<FrameGeneration> frameGeneration;
    std::unique_ptr<Processor> processor;
    if (!nrRuntime.empty()) processor = std::make_unique<Processor>(gpu,nrRuntime,true);
    frameGeneration = std::make_unique<FrameGeneration>(gpu,plugins,nrRuntime);
    auto& fg = *frameGeneration;
    if (relocatePlugin) {
        auto getFunction = reinterpret_cast<PFun_slGetFeatureFunction*>(GetProcAddress(GetModuleHandleW(L"sl.interposer.dll"), "slGetFeatureFunction"));
        void* address = nullptr; HMODULE common = nullptr;
        if (!getFunction || getFunction(sl::kFeatureCommon, "slSetTagForFrame", address) != sl::Result::eOk ||
            !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(address), &common))
            throw std::runtime_error("Cannot locate the active Streamline plugin for restart stress.");
        auto* base = reinterpret_cast<BYTE*>(common);
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
        relocation.base = base;
        relocation.size = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew)->OptionalHeader.SizeOfImage;
    }
    std::cout << std::format("PASS Streamline/Reflex initialization; native runtime reports maximum {}x\n",fg.MaximumMultiplier()) << std::flush;
    if (!nrRuntime.empty()) {
        processor->Resize(1280,720,settings);
        std::cout << "PASS NR feature creation alongside Streamline\n" << std::flush;
    }
    FrameGenerationOutput output(gpu);
    constexpr UINT w = 64, h = 64;
    auto input = gpu.Texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
    const auto desc = input->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 size = 0;
    gpu.device->GetCopyableFootprints(&desc,0,1,0,&footprint,nullptr,nullptr,&size);
    auto upload = gpu.Buffer(size,D3D12_HEAP_TYPE_UPLOAD);
    using namespace DirectX::PackedVector;
    void* mapped = nullptr; Check(upload->Map(0,nullptr,&mapped),"FG test upload");
    for (UINT y=0;y<h;++y) {
        auto* row = reinterpret_cast<HALF*>(static_cast<BYTE*>(mapped)+footprint.Offset+size_t(y)*footprint.Footprint.RowPitch);
        for (UINT x=0;x<w;++x) { for (UINT c=0;c<3;++c) row[x*4+c]=XMConvertFloatToHalf(4); row[x*4+3]=XMConvertFloatToHalf(1); }
    }
    upload->Unmap(0,nullptr);
    gpu.Begin(); D3D12_TEXTURE_COPY_LOCATION src{},dst{};
    src.pResource=upload.Get(); src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint=footprint;
    dst.pResource=input.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
    gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("FG fixture upload timeout.");
    for (bool hdr : {false,true}) {
        output.Resize(w,h,hdr);
        gpu.Begin(); if (!output.Run(input.Get(),true)) throw std::runtime_error("FG history reset missing.");
        auto colourDesc=output.Colour()->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT readLayout{};
        gpu.device->GetCopyableFootprints(&colourDesc,0,1,0,&readLayout,nullptr,nullptr,&size);
        auto readback=gpu.Buffer(size,D3D12_HEAP_TYPE_READBACK);
        Barrier(gpu.commands.Get(),output.Colour(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        src={}; src.pResource=output.Colour(); src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst={}; dst.pResource=readback.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint=readLayout;
        gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        Barrier(gpu.commands.Get(),output.Colour(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("FG colour timeout.");
        Check(readback->Map(0,nullptr,&mapped),"FG colour readback");
        const auto pixel=*reinterpret_cast<const uint32_t*>(static_cast<const BYTE*>(mapped)+readLayout.Offset);
        readback->Unmap(0,nullptr);
        if (!hdr && pixel!=0xffffffffu) throw std::runtime_error("SDR output white mismatch.");
        const double p=std::pow(320.0/10000,2610.0/16384);
        const double expected=std::pow((3424.0/4096+(2413.0/128)*p)/(1+(2392.0/128)*p),2523.0/32);
        if (hdr) for (unsigned shift : {0u,10u,20u})
            if (std::abs(double((pixel>>shift)&1023)/1023-expected)>2.0/1023) throw std::runtime_error("HDR10/PQ brightness mismatch.");
    }
    {
        // NR motion at half the output size: the encode pass resamples it and rescales its pixel units.
        constexpr UINT mw = w / 2, mh = h / 2;
        auto guide = gpu.Texture(mw,mh,DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        const auto guideDesc = guide->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT guideLayout{};
        gpu.device->GetCopyableFootprints(&guideDesc,0,1,0,&guideLayout,nullptr,nullptr,&size);
        auto guideUpload = gpu.Buffer(size,D3D12_HEAP_TYPE_UPLOAD);
        Check(guideUpload->Map(0,nullptr,&mapped),"FG guide upload");
        for (UINT y=0;y<mh;++y) {
            auto* row = reinterpret_cast<HALF*>(static_cast<BYTE*>(mapped)+guideLayout.Offset+size_t(y)*guideLayout.Footprint.RowPitch);
            for (UINT x=0;x<mw;++x) { row[x*2]=XMConvertFloatToHalf(3); row[x*2+1]=XMConvertFloatToHalf(-2); }
        }
        guideUpload->Unmap(0,nullptr);
        gpu.Begin();
        src={}; src.pResource=guideUpload.Get(); src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint=guideLayout;
        dst={}; dst.pResource=guide.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        Barrier(gpu.commands.Get(),guide.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (output.Run(input.Get(),false,guide.Get()) || output.Vectors()==guide.Get())
            throw std::runtime_error("FG did not resample a smaller motion guide.");
        const auto vectorDesc=output.Vectors()->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT vectorLayout{};
        gpu.device->GetCopyableFootprints(&vectorDesc,0,1,0,&vectorLayout,nullptr,nullptr,&size);
        auto readback=gpu.Buffer(size,D3D12_HEAP_TYPE_READBACK);
        Barrier(gpu.commands.Get(),output.Vectors(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        src={}; src.pResource=output.Vectors(); src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst={}; dst.pResource=readback.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint=vectorLayout;
        gpu.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        Barrier(gpu.commands.Get(),output.Vectors(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("FG resampled motion timeout.");
        Check(readback->Map(0,nullptr,&mapped),"FG motion readback");
        for (UINT y : {0u,h/2,h-1}) for (UINT x : {0u,w/2,w-1}) {
            const auto* v=reinterpret_cast<const HALF*>(static_cast<const BYTE*>(mapped)+vectorLayout.Offset+size_t(y)*vectorLayout.Footprint.RowPitch)+x*2;
            if (std::abs(XMConvertHalfToFloat(v[0])-6)>0.01f || std::abs(XMConvertHalfToFloat(v[1])+4)>0.01f) {
                readback->Unmap(0,nullptr);
                throw std::runtime_error(std::format("Resampled FG motion is ({}, {}) at {},{}; expected (6, -4).",
                    XMConvertHalfToFloat(v[0]),XMConvertHalfToFloat(v[1]),x,y));
            }
        }
        readback->Unmap(0,nullptr);
        // The next frame without a guide returns to FG's own estimate.
        gpu.Begin(); output.Run(input.Get(),true);
        if (output.Vectors()==guide.Get()) throw std::runtime_error("FG kept a stale shared guide.");
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("FG fallback motion timeout.");
        std::cout << "PASS half-size NR motion resampled to output pixels for FG\n" << std::flush;
    }
    FrameGenerationPresentCheck(gpu,fg,processor.get());
    ComPtr<ID3D12InfoQueue> diagnostics; gpu.device.As(&diagnostics);
    if (diagnostics) for (UINT64 i=0;i<diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter();++i) {
        SIZE_T bytes=0; diagnostics->GetMessage(i,nullptr,&bytes); std::vector<BYTE> storage(bytes);
        auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data()); diagnostics->GetMessage(i,message,&bytes);
        if (message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR) throw std::runtime_error(message->pDescription);
    }
    std::cout << "PASS SDR encoding, 320-nit HDR10/PQ preservation, synthetic-guide reset and D3D12 validation\n" << std::flush;
    std::cout << "NOT VERIFIED: visible generated-frame quality, HDR display and gameplay latency.\n";
}
}
