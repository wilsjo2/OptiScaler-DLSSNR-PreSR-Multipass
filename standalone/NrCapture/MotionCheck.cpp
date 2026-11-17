#include "Motion.h"
#include <DirectXPackedVector.h>
#include <vector>
#include <iostream>
#include <cmath>
namespace nr {
void MotionCheck() {
    using namespace DirectX::PackedVector;
    constexpr UINT width=512,height=320;
    Gpu gpu; const auto start = ClockMs(); Motion motion(gpu); motion.Resize(width,height);
    std::cout << std::format("Motion initialization: {:.1f} ms\n", ClockMs() - start);
    auto input=gpu.Texture(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_COMMON);
    auto desc=input->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT uploadLayout{},vectorLayout{}; UINT64 bytes=0;
    gpu.device->GetCopyableFootprints(&desc,0,1,0,&uploadLayout,nullptr,nullptr,&bytes);
    auto upload=gpu.Buffer(bytes,D3D12_HEAP_TYPE_UPLOAD);
    desc=motion.Vectors()->GetDesc(); gpu.device->GetCopyableFootprints(&desc,0,1,0,&vectorLayout,nullptr,nullptr,&bytes);
    auto readback=gpu.Buffer(bytes,D3D12_HEAP_TYPE_READBACK);
    auto run = [&](int shiftX,int shiftY,bool reset,bool cut) {
        void* data=nullptr; Check(upload->Map(0,nullptr,&data),"Motion fixture upload");
        for (UINT y=0; y<height; ++y) {
            auto* row=(HALF*)((BYTE*)data+uploadLayout.Offset+size_t(y)*uploadLayout.Footprint.RowPitch);
            for (UINT x=0; x<width; ++x) {
                // An irregular textured surface with a known translation in both axes.
                const int px=int(x)-shiftX,py=int(y)-shiftY;
                uint32_t hash=uint32_t((px+4096)/8)*0x9e3779b9u ^ uint32_t((py+4096)/8)*0x85ebca6bu;
                hash^=hash>>16; hash*=0x7feb352du; hash^=hash>>15;
                float value=cut ? 32.0f : 0.05f+0.8f*float(hash&65535)/65535;
                row[x*4]=row[x*4+1]=row[x*4+2]=XMConvertFloatToHalf(value); row[x*4+3]=XMConvertFloatToHalf(1);
            }
        }
        upload->Unmap(0,nullptr); gpu.Begin();
        Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION from{},to{};
        from.pResource=upload.Get(); from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint=uploadLayout;
        to.pResource=input.Get(); to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        gpu.commands->CopyTextureRegion(&to,0,0,0,&from,nullptr);
        Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const bool historyReset=motion.Estimate(input.Get(),reset);
        Barrier(gpu.commands.Get(),input.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        Barrier(gpu.commands.Get(),motion.Vectors(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        from={}; from.pResource=motion.Vectors(); from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to={}; to.pResource=readback.Get(); to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint=vectorLayout;
        gpu.commands->CopyTextureRegion(&to,0,0,0,&from,nullptr);
        Barrier(gpu.commands.Get(),motion.Vectors(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Motion check completion timeout.");
        Check(readback->Map(0,nullptr,&data),"Motion readback"); std::vector<float> values;
        for (UINT y=48; y<height-48; ++y) {
            auto* row=(HALF*)((BYTE*)data+vectorLayout.Offset+size_t(y)*vectorLayout.Footprint.RowPitch);
            for (UINT x=48; x<width-48; ++x) for (UINT c=0;c<2;++c) {
                float v=XMConvertHalfToFloat(row[x*2+c]); if (!std::isfinite(v)) throw std::runtime_error("Non-finite motion.");
                values.push_back(v);
            }
        }
        readback->Unmap(0,nullptr);
        uint64_t fingerprint=14695981039346656037ull;
        for (auto value : values) { uint32_t bits; memcpy(&bits,&value,sizeof(bits)); fingerprint=(fingerprint^bits)*1099511628211ull; }
        std::cout<<std::format("Vector fixture ({},{}), reset={}, cut={}: {:016X}\n",shiftX,shiftY,reset,cut,fingerprint);
        return std::pair{historyReset,values};
    };
    auto zero = [](const auto& result) {
        return std::all_of(result.second.begin(),result.second.end(),[](float v){return v==0;});
    };
    auto first=run(0,0,false,false);
    if (!first.first || !zero(first)) throw std::runtime_error("First optical-flow frame must reset with zero vectors.");
    auto shifted=run(12,-8,false,false);
    if (shifted.first) throw std::runtime_error("Translation incorrectly detected as a scene cut.");
    std::vector<float> error; unsigned accurate=0;
    for (size_t i=0;i<shifted.second.size();i+=2) {
        float e=std::hypot(shifted.second[i]+12,shifted.second[i+1]-8); error.push_back(e);
        if (e<2.5f) ++accurate;
    }
    std::sort(error.begin(),error.end());
    const float ratio=float(accurate)/float(error.size()),median=error[error.size()/2];
    std::cout<<std::format("Known translation (+12,-8): expected vectors (-12,+8), median error {:.3f}px, {:.1f}% within 2.5px\n",median,ratio*100);
    if (median>1.5f || ratio<0.70f) throw std::runtime_error("Motion direction, scale or accuracy failed.");
    auto still=run(12,-8,false,false);
    if (still.first || !zero(still)) throw std::runtime_error("Static frame generated motion or reset history.");
    auto discontinuity=run(-40,28,true,false);
    if (!discontinuity.first || !zero(discontinuity)) throw std::runtime_error("Explicit reset retained old motion.");
    // The cut frame's vectors are zeroed on the GPU; the history reset follows on the next frame.
    auto cut=run(0,0,false,true);
    if (cut.first || !zero(cut)) throw std::runtime_error("Scene cut did not zero motion within its own frame.");
    auto after=run(0,0,false,false);
    if (!after.first || !zero(after)) throw std::runtime_error("The frame after a scene cut did not reset history.");
    auto resumed=run(6,4,false,false);
    if (resumed.first || zero(resumed)) throw std::runtime_error("Motion did not resume after a scene-cut reset.");
    std::cout<<"PASS optical-flow direction/scale, finite vectors, static identity, first frame, discontinuity and scene cut\n";
}
}
