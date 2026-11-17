#include "FrameGeneration.h"
#include "FrameGenerationOutput.h"
#include "Processor.h"
#include <iostream>
namespace nr {
void FrameGenerationPresentCheck(Gpu& gpu, FrameGeneration& fg, Processor* processor) {
    // Never shown, activated, or used to capture anything. Exercises our own swapchain only.
    WNDCLASSW wc{}; wc.lpfnWndProc=DefWindowProcW; wc.hInstance=GetModuleHandleW(nullptr);
    wc.lpszClassName=L"DisplayFilter.HiddenFgCheck";
    if (!RegisterClassW(&wc) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) Check(HRESULT_FROM_WIN32(GetLastError()),"FG test window class");
    HWND window=CreateWindowExW(WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW|WS_EX_LAYERED|WS_EX_TRANSPARENT,
        wc.lpszClassName,L"",WS_POPUP,0,0,1280,720,nullptr,nullptr,wc.hInstance,nullptr);
    if (!window) Check(HRESULT_FROM_WIN32(GetLastError()),"FG test window");
    struct WindowCleanup { HWND window; ~WindowCleanup(){DestroyWindow(window);} } cleanup{window};
    Check(SetLayeredWindowAttributes(window,0,255,LWA_ALPHA)?S_OK:HRESULT_FROM_WIN32(GetLastError()),"FG test layering");
    FrameGenerationOutput output(gpu); output.Resize(1280,720,false);
    DXGI_SWAP_CHAIN_DESC1 desc{}; desc.Width=1280; desc.Height=720; desc.Format=output.Format();
    desc.SampleDesc.Count=1; desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT; desc.BufferCount=3;
    desc.Scaling=DXGI_SCALING_STRETCH; desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD; desc.AlphaMode=DXGI_ALPHA_MODE_IGNORE;
    ComPtr<IDXGISwapChain3> chain; fg.CreateSwapchain(window,desc,&chain);
    Check(chain->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),"FG test colour space");
    auto input=gpu.Texture(1280,720,DXGI_FORMAT_R16G16B16A16_FLOAT);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors=1; hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap; Check(gpu.device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)),"FG fixture descriptors");
    gpu.device->CreateUnorderedAccessView(input.Get(),nullptr,nullptr,heap->GetCPUDescriptorHandleForHeapStart());
    bool readable=false;
    unsigned submitted=0;
    const auto presentsBefore=fg.PresentedFrames();
    try {
        for (unsigned multiplier=2;multiplier<=std::min(4u,fg.MaximumMultiplier());++multiplier) {
            fg.Configure(1280,720,output.Format(),multiplier,false);
            auto nextFrame=ClockMs();
            const auto modePresents=fg.PresentedFrames();
            for (unsigned frame=0;frame<60;++frame) {
                if (frame==1) fg.Configure(1280,720,output.Format(),multiplier);
                const auto remaining=nextFrame-ClockMs();
                if (remaining>0) Sleep(DWORD(remaining));
                nextFrame=ClockMs()+1000.0*multiplier/120.0;
                Settings settings;
                settings.passes=2; settings.estimateMotion=true; settings.transfer=1;
                if (frame>=15 && frame<30) settings.transfer=0; // Classic colour SR with FG, then back to residuals.
                settings.upscaler=frame<45 ? 1 : 0;
                settings.modelScale=frame<15 ? 60 : frame>=30 && frame<45 ? 100 : 80;
                settings.model.style=frame<30 ? 0 : 2;
                if (frame==15 || frame==30 || frame==45) {
                    // Consume FG-off on the present thread before a visibility or
                    // settings transition; retained resources survive the pause.
                    fg.Quiesce(chain.Get()); output.Reset(); Sleep(120);
                    Check(SetWindowPos(window,nullptr,0,0,1280,720,SWP_NOACTIVATE|SWP_NOZORDER) ? S_OK :
                        HRESULT_FROM_WIN32(GetLastError()), "Reposition hidden FG fixture while disabled");
                    fg.Configure(1280,720,output.Format(),multiplier);
                }
                if (processor) processor->Resize(1280,720,settings);
                fg.BeginFrame(120); gpu.Begin(); gpu.BeginTimings();
                if (readable) Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                ID3D12DescriptorHeap* heaps[]{heap.Get()}; gpu.commands->SetDescriptorHeaps(1,heaps);
                const float colour[]{.15f+frame*.01f,.2f,.3f,1};
                gpu.commands->ClearUnorderedAccessViewFloat(heap->GetGPUDescriptorHandleForHeapStart(),heap->GetCPUDescriptorHandleForHeapStart(),input.Get(),colour,0,nullptr);
                Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); readable=true;
                ID3D12Resource* picture=input.Get();
                if (processor) {
                    Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
                    picture=processor->Run(input.Get(),settings,false,frame==0 || frame%15==0,1,1000.f*multiplier/120);
                    Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
                }
                auto* guide=processor ? processor->SceneMotion() : nullptr;
                const auto serialBefore=gpu.serial;
                const bool reset=output.Run(picture,frame==0 || (guide && processor->historyReset),guide);
                // An output-size guide is passed through; any other size is resampled by the encode pass.
                const bool sameSize=guide && guide->GetDesc().Width==output.Colour()->GetDesc().Width &&
                    guide->GetDesc().Height==output.Colour()->GetDesc().Height;
                if (guide && ((output.Vectors()==guide)!=sameSize || gpu.serial!=serialBefore))
                    throw std::runtime_error("Shared FG motion must reuse the guide without a second submission/wait.");
                fg.Tag(output.Colour(),output.Depth(),output.Vectors(),reset);
                ComPtr<ID3D12Resource> back; Check(chain->GetBuffer(chain->GetCurrentBackBufferIndex(),IID_PPV_ARGS(&back)),"FG test backbuffer");
                Barrier(gpu.commands.Get(),output.Colour(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
                Barrier(gpu.commands.Get(),back.Get(),D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_COPY_DEST);
                gpu.commands->CopyResource(back.Get(),output.Colour());
                Barrier(gpu.commands.Get(),back.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PRESENT);
                Barrier(gpu.commands.Get(),output.Colour(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                gpu.ResolveTimings(); gpu.Submit(); fg.BeforePresent(); Check(chain->Present(0,0),"Hidden FG present"); fg.AfterPresent(); ++submitted;
                if (!gpu.Drain()) throw std::runtime_error("Hidden FG completion timeout.");
                std::array<double, unsigned(GpuStage::Count)> stages{};
                if (!gpu.TakeTimings(stages)) throw std::runtime_error("FG frame stage timings were not ready.");
                for (const auto value : stages) if (!std::isfinite(value) || value < 0)
                    throw std::runtime_error("Invalid GPU stage duration.");
                if (GetForegroundWindow()==window || IsWindowVisible(window))
                    throw std::runtime_error("The headless FG check must never show or activate its window.");
            }
            std::cout<<std::format("Requested {}x, 120 FPS target: 60 real frames, {} reported presents\n",multiplier,fg.PresentedFrames()-modePresents)<<std::flush;
            fg.Quiesce(chain.Get());
        }
        const auto presented=fg.PresentedFrames()-presentsBefore;
        std::cout<<std::format("PASS hidden swapchain lifecycle; {} real submissions, {} reported presents\n",submitted,presented);
        if (presented<=submitted)
            throw std::runtime_error("Frame generation failed: no additional presents observed. Inspect the runtime log for suppression reasons.");
        std::cout<<"NOT VERIFIED physical scanout or generated-frame image quality\n"<<std::flush;
        if (processor) std::cout<<"PASS two NR passes + private DLSS + FG, shared/fallback motion, 60/80/100% model resize, style rebuild, pause/resume and bilinear switch\n"<<std::flush;
    } catch (...) {
        try { fg.Disable(); gpu.Drain(); } catch (...) {}
        throw;
    }
}
}
