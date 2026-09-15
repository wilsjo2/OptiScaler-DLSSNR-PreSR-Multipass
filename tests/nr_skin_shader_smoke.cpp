// Headless shader test: Windows D3D11 WARP executes the shared HLSL, no game or NR DLL.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <cstddef>
#include "../OptiScaler/shaders/dlssnr/DlssNr_Common.h"
using Microsoft::WRL::ComPtr;
struct Pixel { float r, g, b, a; };
static void check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D call failed"); }
static void expect(bool ok, const char* label) { if (!ok) throw std::runtime_error(label); }
static bool same(Pixel a, Pixel b) {
    return std::abs(a.r-b.r)<0.0001f && std::abs(a.g-b.g)<0.0001f && std::abs(a.b-b.b)<0.0001f && a.a==b.a;
}
int wmain(int argc, wchar_t** argv) try {
    if (argc != 2) throw std::runtime_error("Pass the dlssnr.hlsl path");
    static_assert(offsetof(DlssNrConstants, SkinProtection) == 92);
    static_assert(offsetof(DlssNrConstants, EnvironmentColour) == 112);
    ComPtr<ID3DBlob> code, errors;
    HRESULT compiled = D3DCompileFromFile(argv[1], nullptr, nullptr, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (errors) std::fprintf(stderr, "%s", (char*)errors->GetBufferPointer());
    check(compiled);
    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> ctx;
    check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &ctx));
    ComPtr<ID3D11ComputeShader> shader;
    check(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader));
    const std::array<Pixel, 2> base {{{0.75f,0.50f,0.40f,1}, {0.20f,0.40f,0.80f,1}}};
    const std::array<Pixel, 2> edited {{{0.50f,0.75f,0.40f,1}, {0.65f,0.25f,0.40f,1}}};
    D3D11_TEXTURE2D_DESC desc {};
    desc.Width=2; desc.Height=1; desc.MipLevels=1; desc.ArraySize=1;
    desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; desc.SampleDesc.Count=1;
    desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    ComPtr<ID3D11Texture2D> original, model, output, keep, readback;
    D3D11_SUBRESOURCE_DATA data {base.data(), sizeof(base), 0};
    check(device->CreateTexture2D(&desc,&data,&original));
    data.pSysMem=edited.data(); check(device->CreateTexture2D(&desc,&data,&model));
    ComPtr<ID3D11ShaderResourceView> originalSrv, modelSrv;
    check(device->CreateShaderResourceView(original.Get(),nullptr,&originalSrv));
    check(device->CreateShaderResourceView(model.Get(),nullptr,&modelSrv));
    desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
    check(device->CreateTexture2D(&desc,nullptr,&output)); check(device->CreateTexture2D(&desc,nullptr,&keep));
    ComPtr<ID3D11UnorderedAccessView> outputUav, keepUav;
    check(device->CreateUnorderedAccessView(output.Get(),nullptr,&outputUav));
    check(device->CreateUnorderedAccessView(keep.Get(),nullptr,&keepUav));
    desc.BindFlags=0; desc.Usage=D3D11_USAGE_STAGING; desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    check(device->CreateTexture2D(&desc,nullptr,&readback));
    D3D11_BUFFER_DESC buffer {}; buffer.ByteWidth=sizeof(DlssNrConstants); buffer.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    ComPtr<ID3D11Buffer> constants; check(device->CreateBuffer(&buffer,nullptr,&constants));
    D3D11_SAMPLER_DESC sampling {}; sampling.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampling.AddressU=sampling.AddressV=sampling.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP; sampling.MaxLOD=D3D11_FLOAT32_MAX;
    ComPtr<ID3D11SamplerState> sampler; check(device->CreateSamplerState(&sampling,&sampler));
    ID3D11ShaderResourceView* srvs[]={originalSrv.Get(),modelSrv.Get(),originalSrv.Get(),originalSrv.Get(),originalSrv.Get()};
    ID3D11UnorderedAccessView* uavs[]={outputUav.Get(),keepUav.Get()};
    ctx->CSSetShader(shader.Get(),nullptr,0); ctx->CSSetShaderResources(0,5,srvs);
    ctx->CSSetUnorderedAccessViews(0,2,uavs,nullptr); ctx->CSSetConstantBuffers(0,1,constants.GetAddressOf());
    ctx->CSSetSamplers(0,1,sampler.GetAddressOf());
    DlssNrConstants settings {}; settings.Mode=DlssNrMode_Resolve; settings.Width=2; settings.Height=1;
    settings.WhitePoint=1; settings.Passthrough=1; settings.ApplyModel=1; settings.ReversibleMode=2;
    settings.TransferStrength=settings.ColourStrength=1; settings.MaxRatio=2;
    settings.SkinDetail=settings.SkinColour=settings.EnvironmentDetail=settings.EnvironmentColour=1;
    auto run=[&]() {
        ctx->UpdateSubresource(constants.Get(),0,nullptr,&settings,0,0); ctx->Dispatch(1,1,1);
        ctx->CopyResource(readback.Get(),output.Get()); D3D11_MAPPED_SUBRESOURCE mapped {};
        check(ctx->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped));
        std::array<Pixel,2> result; memcpy(result.data(),mapped.pData,sizeof(result)); ctx->Unmap(readback.Get(),0); return result;
    };
    auto result=run(); expect(same(result[0],edited[0]) && same(result[1],edited[1]), "Disabled filter changed output");
    settings.SkinProtection=1; result=run(); expect(same(result[0],edited[0]) && same(result[1],edited[1]), "Unity settings changed output");
    settings.SkinDetail=settings.SkinColour=settings.EnvironmentDetail=settings.EnvironmentColour=0;
    result=run(); expect(same(result[0],base[0]) && same(result[1],base[1]), "Zero strengths did not restore input");
    settings.EnvironmentDetail=settings.EnvironmentColour=1;
    result=run(); expect(same(result[0],base[0]) && same(result[1],edited[1]), "Skin/scene separation failed");
    settings.SkinDetail=1; result=run();
    expect(std::abs(result[0].r/result[0].g - base[0].r/base[0].g)<0.0001f &&
           std::abs(result[0].b/result[0].g - base[0].b/base[0].g)<0.0001f && same(result[1],edited[1]),
           "Skin colour suppression did not preserve chroma / affected environment");
    settings.ShowSkinMask=1; result=run(); expect(result[0].r>0.99f && result[1].r<0.01f, "Preview mismatch");
    settings.ShowSkinMask=0; settings.ApplyModel=0; result=run();
    expect(same(result[0],base[0]) && same(result[1],base[1]), "Model bypass did not preserve input");
    std::puts("PASS: disabled/unity identity, zero restore, skin/scene separation, preview, model bypass (WARP HLSL)");

    // Exercise the exact new carrier shader independently of proprietary DLSS/NR. Both positive
    // and negative RGB changes must survive encoding; neutral must leave the clean raster intact.
    const std::array<Pixel,2> carrierBase {{{2.0f,0.25f,0.75f,0.25f}, {0.10f,1.50f,0.30f,0.75f}}};
    const std::array<Pixel,2> carrierEdit {{{1.0f,0.50f,0.70f,0.25f}, {0.20f,0.25f,0.40f,0.75f}}};
    ctx->UpdateSubresource(original.Get(),0,nullptr,carrierBase.data(),sizeof(carrierBase),0);
    ctx->UpdateSubresource(model.Get(),0,nullptr,carrierEdit.data(),sizeof(carrierEdit),0);
    settings.Mode=DlssNrMode_EncodeResidual; settings.ExposurePreMul=2.0f;
    auto carrier=run();
    expect(carrier[0].r<0.5f && carrier[0].g>0.5f && carrier[1].g<0.5f,
           "Signed residual lost shadow/brightening information");
    ctx->UpdateSubresource(model.Get(),0,nullptr,carrier.data(),sizeof(carrier),0);
    settings.Mode=DlssNrMode_ApplyResidual; result=run();
    expect(same(result[0],carrierEdit[0]) && same(result[1],carrierEdit[1]),
           "Residual roundtrip or original alpha preservation failed");
    const std::array<Pixel,2> neutral {{{0.5f,0.5f,0.5f,1}, {0.5f,0.5f,0.5f,1}}};
    ctx->UpdateSubresource(model.Get(),0,nullptr,neutral.data(),sizeof(neutral),0);
    result=run(); expect(same(result[0],carrierBase[0]) && same(result[1],carrierBase[1]),
                         "Neutral carrier altered the clean raster");
    const std::array<Pixel,2> overshoot {{{-1,2,0.5f,1}, {INFINITY,NAN,0.5f,1}}};
    ctx->UpdateSubresource(model.Get(),0,nullptr,overshoot.data(),sizeof(overshoot),0);
    result=run();
    for (const auto pixel : result)
        expect(std::isfinite(pixel.r) && std::isfinite(pixel.g) && std::isfinite(pixel.b) &&
               pixel.r>=0 && pixel.g>=0 && pixel.b>=0, "Carrier overshoot produced invalid output");
    settings.Mode=DlssNrMode_UnitExposure; result=run();
    expect(result[0].r==1 && result[1].r==1, "Private DLSS exposure is not fixed at one");
    std::puts("PASS: signed residual, shadow/brightening roundtrip, neutral identity, alpha, overshoot, unit exposure");
    const std::array<Pixel,2> currentMotion {{{1,0,0,1}, {INFINITY,0,0,1}}};
    ctx->UpdateSubresource(original.Get(),0,nullptr,currentMotion.data(),sizeof(currentMotion),0);
    settings.Mode=DlssNrMode_NormalizeMotion; settings.MvScaleX=0.5f; settings.MvScaleY=1;
    result=run();
    expect(result[0].r==0.5f && result[0].a==1 && result[1].r==65504 && result[1].a==0,
           "Motion normalization / invalid sentinel failed");
    const std::array<Pixel,2> normalized {{{0.5f,0,0,1}, {0.5f,0,0,1}}};
    const std::array<Pixel,2> previousMotion {{{0.2f,0,0,1}, {0.1f,0,0,1}}};
    ctx->UpdateSubresource(original.Get(),0,nullptr,normalized.data(),sizeof(normalized),0);
    ctx->UpdateSubresource(model.Get(),0,nullptr,previousMotion.data(),sizeof(previousMotion),0);
    settings.Mode=DlssNrMode_ComposeMotion; result=run();
    expect(std::abs(result[0].r-0.6f)<0.0001f && result[0].a==1,
           "Two-frame motion did not sample the previous field at the displaced position");
    expect(result[1].r==65504 && result[1].a==0, "Offscreen history was silently clamped");
    std::puts("PASS: motion normalization, displaced two-frame composition, invalid and offscreen history");
    ctx->UpdateSubresource(original.Get(),0,nullptr,carrierBase.data(),sizeof(carrierBase),0);
    ctx->UpdateSubresource(model.Get(),0,nullptr,carrier.data(),sizeof(carrier),0);
    settings.Mode=DlssNrMode_ApplyInterpolatedResidual; settings.ExposurePreMul=2;
    D3D11_TEXTURE2D_DESC flagDesc {}; flagDesc.Width=flagDesc.Height=flagDesc.MipLevels=flagDesc.ArraySize=1;
    flagDesc.Format=DXGI_FORMAT_R8_UNORM; flagDesc.SampleDesc.Count=1;
    flagDesc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    unsigned char flag=1; D3D11_SUBRESOURCE_DATA flagData {&flag,1,0};
    ComPtr<ID3D11Texture2D> flagTexture; ComPtr<ID3D11ShaderResourceView> flagSrv;
    check(device->CreateTexture2D(&flagDesc,&flagData,&flagTexture));
    check(device->CreateShaderResourceView(flagTexture.Get(),nullptr,&flagSrv));
    ctx->CSSetShaderResources(4,1,flagSrv.GetAddressOf());
    result=run(); expect(same(result[0],carrierBase[0]) && same(result[1],carrierBase[1]),
                         "FG suppression did not preserve matching clean frame");
    flag=0; ctx->UpdateSubresource(flagTexture.Get(),0,nullptr,&flag,1,0);
    result=run(); expect(same(result[0],carrierEdit[0]) && same(result[1],carrierEdit[1]),
                         "Allowed FG residual was not composed");
    std::puts("PASS: rejected FG output preserves the clean frame");
    // A rejected midpoint must not alternate NR-on / NR-off in a static scene.
    // t2 is the PREVIOUS NR anchor; t3 is midpoint -> previous-anchor motion,
    // not current-anchor motion and not the composed two-frame field.
    ComPtr<ID3D11Texture2D> fallbackTexture, fallbackMotion;
    ComPtr<ID3D11ShaderResourceView> fallbackSrv, fallbackMotionSrv;
    desc.BindFlags=D3D11_BIND_SHADER_RESOURCE; desc.Usage=D3D11_USAGE_DEFAULT; desc.CPUAccessFlags=0;
    data.pSysMem=carrier.data(); data.SysMemPitch=sizeof(carrier);
    check(device->CreateTexture2D(&desc,&data,&fallbackTexture));
    check(device->CreateShaderResourceView(fallbackTexture.Get(),nullptr,&fallbackSrv));
    const std::array<Pixel,2> stillMotion {{{0,0,0,1},{0,0,0,1}}};
    data.pSysMem=stillMotion.data();
    check(device->CreateTexture2D(&desc,&data,&fallbackMotion));
    check(device->CreateShaderResourceView(fallbackMotion.Get(),nullptr,&fallbackMotionSrv));
    ctx->CSSetShaderResources(2,1,fallbackSrv.GetAddressOf());
    ctx->CSSetShaderResources(3,1,fallbackMotionSrv.GetAddressOf());
    settings.Mode=DlssNrMode_ApplyResidualFgFallback;
    for (unsigned i=0;i<16;++i) {
        flag=(unsigned char)(i%2); ctx->UpdateSubresource(flagTexture.Get(),0,nullptr,&flag,1,0);
        result=run();
        expect(same(result[0],carrierEdit[0]) && same(result[1],carrierEdit[1]),
               "FG rejection toggled the NR edit off in a static scene");
    }
    const std::array<Pixel,2> displacedMotion {{{0.5f,0,0,1},{0.5f,0,0,1}}};
    ctx->UpdateSubresource(fallbackMotion.Get(),0,nullptr,displacedMotion.data(),sizeof(displacedMotion),0);
    result=run();
    const Pixel translated {carrierBase[0].r+carrierEdit[1].r-carrierBase[1].r,
                           std::max(0.0f,carrierBase[0].g+carrierEdit[1].g-carrierBase[1].g),
                           carrierBase[0].b+carrierEdit[1].b-carrierBase[1].b,carrierBase[0].a};
    expect(same(result[0],translated) && same(result[1],carrierBase[1]),
           "Rejected FG fallback used the wrong motion interval or clamped offscreen history");
    for (float invalid : {NAN,INFINITY,65504.0f}) {
        const std::array<Pixel,2> badMotion {{{invalid,0,0,1},{0,0,0,0}}};
        ctx->UpdateSubresource(fallbackMotion.Get(),0,nullptr,badMotion.data(),sizeof(badMotion),0);
        result=run();
        expect(same(result[0],carrierBase[0]) && same(result[1],carrierBase[1]),
               "Rejected FG fallback trusted invalid motion");
    }
    flag=0; ctx->UpdateSubresource(flagTexture.Get(),0,nullptr,&flag,1,0);
    result=run();
    expect(same(result[0],carrierEdit[0]) && same(result[1],carrierEdit[1]),
           "Valid NVIDIA interpolation was replaced by the rejection fallback");
    std::puts("PASS: FG rejection without static pulsing, midpoint reprojection, invalid/offscreen rejection, valid FG unchanged");
    ctx->UpdateSubresource(fallbackMotion.Get(),0,nullptr,displacedMotion.data(),sizeof(displacedMotion),0);
    settings.Mode=DlssNrMode_ApplyReprojectedResidual;
    result=run();
    expect(same(result[0],translated) && same(result[1],carrierBase[1]),
           "Current-raster half-rate did not reproject the preceding NR edit");
    const std::array<Pixel,2> invalidCurrentMotion {{{NAN,0,0,1},{0,0,0,0}}};
    ctx->UpdateSubresource(fallbackMotion.Get(),0,nullptr,invalidCurrentMotion.data(),sizeof(invalidCurrentMotion),0);
    result=run();
    expect(same(result[0],carrierBase[0]) && same(result[1],carrierBase[1]),
           "Current-raster half-rate trusted invalid motion");
    std::puts("PASS: current-raster half-rate reprojection and invalid/offscreen clean fallback");
    DlssNrResidualHold hold;
    expect(!hold.CanReuse(0), "Uninitialized hold was reused");
    hold.SampleSucceeded(0);
    expect(!hold.CanReuse(0) && hold.CanReuse(1) && !hold.CanReuse(2), "Hold exceeded two-frame cadence");
    hold.Reset(); expect(!hold.CanReuse(1), "Cut/failure did not invalidate held residual");
    hold.SampleSucceeded(5); expect(!hold.CanReuse(7), "Frame gap reused stale residual");
    auto nextBase=carrierBase, nextExpected=carrierEdit;
    for (unsigned i=0;i<2;++i) {
        nextBase[i].r+=0.3f; nextBase[i].g+=0.3f; nextBase[i].b+=0.3f;
        nextExpected[i].r+=0.3f; nextExpected[i].g+=0.3f; nextExpected[i].b+=0.3f;
    }
    ctx->UpdateSubresource(original.Get(),0,nullptr,nextBase.data(),sizeof(nextBase),0);
    settings.Mode=DlssNrMode_ApplyResidual; result=run();
    expect(same(result[0],nextExpected[0]) && same(result[1],nextExpected[1]),
           "Held residual froze the raster instead of editing the next current frame");
    settings.Mode=DlssNrMode_ZeroMotion; result=run();
    expect(result[0].r==0 && result[0].g==0 && result[1].r==0 && result[1].g==0,
           "Private reset-only motion guide was not zero");
    std::puts("PASS: two-frame hold cadence, cut/gap invalidation, current-raster composition, zero guide");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
