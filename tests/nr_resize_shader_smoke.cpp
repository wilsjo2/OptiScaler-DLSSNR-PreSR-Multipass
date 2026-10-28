// Execute the shipped shader on WARP; no game, capture or NVIDIA runtime is loaded.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "../OptiScaler/shaders/dlssnr/DlssNr_Common.h"
#include "../OptiScaler/shaders/dlssnr/precompile/DlssNr_Shader.h"
using Microsoft::WRL::ComPtr;
struct Pixel { float r, g, b, a; };
struct Image { unsigned w, h; std::vector<Pixel> pixels; };
void check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D11 call failed"); }
void expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
float linear(float v) { return v <= .04045f ? v / 12.92f : std::pow((v + .055f) / 1.055f, 2.4f); }
float srgb(float v) { return v <= .0031308f ? v * 12.92f : 1.055f * std::pow(v, 1 / 2.4f) - .055f; }
float difference(const Image& a, const Image& b)
{
    float error = 0;
    for (size_t i = 0; i < a.pixels.size(); ++i)
    {
        const auto x = a.pixels[i], y = b.pixels[i];
        expect(std::isfinite(x.r) && std::isfinite(x.g) && std::isfinite(x.b), "Non-finite reconstruction");
        error = std::max({error, std::abs(x.r-y.r), std::abs(x.g-y.g), std::abs(x.b-y.b), std::abs(x.a-y.a)});
    }
    return error;
}
Image modelAnswer(Image image, bool hdr, float gain, float colour)
{
    for (auto& p : image.pixels)
    {
        if (hdr) { p.r=linear(p.r); p.g=linear(p.g); p.b=linear(p.b); }
        float y = .2126f*p.r + .7152f*p.g + .0722f*p.b;
        p.r = gain * (p.r + colour*y); p.b = gain * (p.b - colour*y);
        p.g = (gain*y - .2126f*p.r - .0722f*p.b) / .7152f;
        if (hdr) { p.r=srgb(p.r); p.g=srgb(p.g); p.b=srgb(p.b); }
    }
    return image;
}
int main()
try
{
    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> ctx;
    check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0,
                           D3D11_SDK_VERSION, &device, nullptr, &ctx));
    ComPtr<ID3D11ComputeShader> shader;
    check(device->CreateComputeShader(DlssNr_cso, sizeof(DlssNr_cso), nullptr, &shader));
    D3D11_BUFFER_DESC bd {}; bd.ByteWidth=sizeof(DlssNrConstants); bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
    ComPtr<ID3D11Buffer> cb; check(device->CreateBuffer(&bd, nullptr, &cb));
    D3D11_SAMPLER_DESC sd {}; sd.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU=sd.AddressV=sd.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP; sd.MaxLOD=D3D11_FLOAT32_MAX;
    ComPtr<ID3D11SamplerState> sampler; check(device->CreateSamplerState(&sd, &sampler));
    auto run = [&](const DlssNrConstants& c, const Image& source, const Image& model, const Image& original)
    {
        ComPtr<ID3D11Texture2D> textures[4], target, keep, staging;
        ComPtr<ID3D11ShaderResourceView> views[4];
        const Image* inputs[] = { &source, &model, &original, &original };
        D3D11_TEXTURE2D_DESC td {}; td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;
        td.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; td.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        for (unsigned i=0; i<4; ++i)
        {
            td.Width=inputs[i]->w; td.Height=inputs[i]->h;
            D3D11_SUBRESOURCE_DATA data { inputs[i]->pixels.data(), td.Width*sizeof(Pixel), 0 };
            check(device->CreateTexture2D(&td, &data, &textures[i]));
            check(device->CreateShaderResourceView(textures[i].Get(), nullptr, &views[i]));
        }
        td.Width=c.Width; td.Height=c.Height; td.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
        check(device->CreateTexture2D(&td, nullptr, &target)); check(device->CreateTexture2D(&td, nullptr, &keep));
        ComPtr<ID3D11UnorderedAccessView> uav, keepUav;
        check(device->CreateUnorderedAccessView(target.Get(), nullptr, &uav));
        check(device->CreateUnorderedAccessView(keep.Get(), nullptr, &keepUav));
        td.Usage=D3D11_USAGE_STAGING; td.BindFlags=0; td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        check(device->CreateTexture2D(&td, nullptr, &staging));
        ID3D11ShaderResourceView* srv[] = {views[0].Get(), views[1].Get(), views[2].Get(), views[3].Get()};
        ID3D11UnorderedAccessView* outputs[] = {uav.Get(), keepUav.Get()};
        ctx->UpdateSubresource(cb.Get(),0,nullptr,&c,0,0);
        ctx->CSSetShader(shader.Get(),nullptr,0); ctx->CSSetConstantBuffers(0,1,cb.GetAddressOf());
        ctx->CSSetSamplers(0,1,sampler.GetAddressOf()); ctx->CSSetShaderResources(0,4,srv);
        ctx->CSSetUnorderedAccessViews(0,2,outputs,nullptr); ctx->Dispatch((c.Width+7)/8,(c.Height+7)/8,1);
        ctx->CopyResource(staging.Get(),target.Get()); D3D11_MAPPED_SUBRESOURCE mapped {};
        check(ctx->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped));
        Image result {c.Width,c.Height,std::vector<Pixel>(size_t(c.Width)*c.Height)};
        for (unsigned y=0; y<c.Height; ++y)
            memcpy(result.pixels.data()+y*c.Width, static_cast<char*>(mapped.pData)+y*mapped.RowPitch, c.Width*sizeof(Pixel));
        ctx->Unmap(staging.Get(),0); ctx->ClearState(); return result;
    };
    Image original {100,8,std::vector<Pixel>(800)};
    for (unsigned y=0; y<8; ++y) for (unsigned x=0; x<100; ++x)
        original.pixels[y*100+x] = x<51 ? Pixel{.024f,.018f,.012f,.4f} : Pixel{.65f,.36f,.24f,.8f};
    unsigned cases=0; float worst=0, legacyHalo=0;
    for (unsigned hdr : {0u,1u}) for (unsigned curve=0; curve<5; ++curve)
    {
        DlssNrConstants c {}; c.Width=100; c.Height=8; c.Mode=DlssNrMode_Encode;
        c.Passthrough=!hdr; c.WhitePoint=1; c.ReversibleMode=curve; c.MaxRatio=8;
        c.TransferStrength=c.ColourStrength=1; c.ApplyModel=1; c.CompareZoom=1;
        const auto full = run(c,original,original,original);
        for (float gain : {.75f,1.0f,1.15f}) for (float colour : {0.0f,.05f})
        {
            const auto nativeAnswer=modelAnswer(full,hdr!=0,gain,colour);
            c.Mode=DlssNrMode_Resolve; c.Transfer=0;
            const auto reference=run(c,full,nativeAnswer,original);
            for (unsigned percent : {25u,65u,99u,100u,101u})
            {
                auto work=c; work.Width=percent; work.Height=std::max(2u,8*percent/100);
                work.Mode=DlssNrMode_Downsample;
                const auto proxy=run(work,full,full,original);
                const auto answer=modelAnswer(proxy,hdr!=0,gain,colour);
                c.Transfer=3;
                auto reconstructed=run(c,proxy,answer,original);
                if (percent<100)
                {
                    float error=difference(reconstructed,reference); worst=std::max(worst,error);
                    if (error>.0004f) std::printf("hdr=%u curve=%u gain=%g colour=%g scale=%u error=%g\n",hdr,curve,gain,colour,percent,error);
                    expect(error<.0004f,"Separated resize changed a uniform lighting/chroma edit at the hair/skin edge");
                    c.Transfer=1; const auto legacy=run(c,proxy,answer,original);
                    legacyHalo=std::max(legacyHalo,difference(legacy,reference));
                    c.Transfer=3; c.ApplyModel=0;
                    expect(difference(run(c,proxy,answer,original),original)<1e-6f,"Bypass changed the source frame");
                    c.ApplyModel=1;
                    // Carrier encoding and full-size carrier consumption, without proprietary DLSS.
                    work.Mode=DlssNrMode_EncodeResizeField;
                    const auto field=run(work,proxy,answer,original);
                    Image enlarged {100,8,std::vector<Pixel>(800,field.pixels[field.pixels.size()/2])};
                    c.Transfer=4;
                    expect(difference(run(c,field,enlarged,original),reference)<.0005f,"Private carrier reconstruction mismatch");
                    std::fill(enlarged.pixels.begin(),enlarged.pixels.end(),Pixel{2,-1,3,1});
                    expect(difference(run(c,field,enlarged,original),reference)<.0005f,"Private carrier range guard failed");
                }
                else
                {
                    c.Transfer=0;
                    expect(difference(reconstructed,run(c,proxy,answer,original))<1e-6f,"Native/supersampled path changed");
                }
                if (hdr && colour==0 && percent<100)
                {
                    c.Transfer=5;
                    expect(difference(run(c,proxy,answer,original),reference)<.0005f,
                           "Edge-aware enlargement changed a uniform HDR lighting edit");
                }
                ++cases;
            }
        }
    }
    expect(legacyHalo>.01f,"Fixture did not reproduce the old resize mismatch");
    DlssNrConstants black {}; black.Mode=DlssNrMode_Resolve; black.Width=100; black.Height=8;
    black.Passthrough=black.ApplyModel=1; black.ReversibleMode=2; black.Transfer=3;
    Image blackFrame {100,8,std::vector<Pixel>(800,Pixel{0,0,0,1})};
    Image blackProxy {25,2,std::vector<Pixel>(50,Pixel{0,0,0,1})};
    Image brightAnswer {25,2,std::vector<Pixel>(50,Pixel{.3f,.2f,.1f,1})};
    expect(difference(run(black,blackProxy,brightAnswer,blackFrame),blackFrame)==0,
           "Relative reconstruction created light at true black");
    black.Transfer=5;
    expect(difference(run(black,blackProxy,brightAnswer,blackFrame),blackFrame)==0,
           "Edge-aware enlargement created light at true black");

    // A dark-side edit next to an unchanged bright surface must not spill across the edge.
    Image edge {32,8,std::vector<Pixel>(256)}, proxy {8,2,std::vector<Pixel>(16)};
    for (unsigned y=0; y<8; ++y) for (unsigned x=0; x<32; ++x)
        edge.pixels[y*32+x] = x<16 ? Pixel{.1f,.1f,.1f,.4f} : Pixel{.8f,.8f,.8f,.8f};
    for (unsigned y=0; y<2; ++y) for (unsigned x=0; x<8; ++x)
        proxy.pixels[y*8+x] = edge.pixels[x*4];
    auto answer=proxy, expected=edge;
    for (auto* img : {&answer,&expected}) for (unsigned y=0; y<img->h; ++y)
        for (unsigned x=0; x<img->w/2; ++x)
        {
            auto& p=img->pixels[y*img->w+x];
            p.r=p.g=p.b=srgb(linear(p.r)*.5f);
        }
    auto ec=black; ec.Width=32; ec.Height=8; ec.Transfer=3;
    const float smoothError=difference(run(ec,proxy,answer,edge),expected);
    ec.Transfer=5;
    const float edgeError=difference(run(ec,proxy,answer,edge),expected);
    expect(smoothError>.01f && edgeError<.001f, "Edge-aware filter did not stop cross-edge edit leakage");
    expect(difference(run(ec,proxy,proxy,edge),edge)<1e-6f,"Edge-aware no-edit reconstruction changed the frame");
    expect(difference(run(ec,edge,expected,edge),expected)<1e-6f,"Edge-aware native-size path changed");

    // Halo correction must lift narrow dark rims while retaining broad shading and alpha.
    Image haloProxy {64,8,std::vector<Pixel>(512)};
    for (unsigned y=0; y<8; ++y) for (unsigned x=0; x<64; ++x)
        haloProxy.pixels[y*64+x]=x<32 ? Pixel{.04f,.04f,.04f,.3f} : Pixel{.7f,.7f,.7f,.9f};
    const auto broad=modelAnswer(haloProxy,true,.8f,0);
    auto rims=broad;
    for (unsigned y=0; y<8; ++y) for (unsigned x=31; x<=34; ++x)
    {
        auto& p=rims.pixels[y*64+x];
        p.r=p.g=p.b=srgb(linear(p.r)*.6f);
    }
    DlssNrConstants hc {}; hc.Mode=DlssNrMode_SuppressHalo; hc.Width=64; hc.Height=8; hc.HaloRadius=2;
    expect(difference(run(hc,haloProxy,rims,haloProxy),rims)==0,"Disabled halo filter changed output");
    hc.HaloSuppression=hc.HaloBrightSuppression=1;
    expect(difference(run(hc,haloProxy,broad,haloProxy),broad)<1e-6f,"Halo filter removed broad shading");
    expect(difference(run(hc,haloProxy,haloProxy,haloProxy),haloProxy)<1e-6f,"Halo filter changed an unedited image");
    const auto corrected=run(hc,haloProxy,rims,haloProxy);
    for (unsigned x : {31u,32u})
    {
        expect(corrected.pixels[x].r>rims.pixels[x].r+.001f,"Halo filter did not lift a dark rim");
        expect(corrected.pixels[x].r<=broad.pixels[x].r+.0001f,"Halo filter overshot broad lighting");
    }
    for (size_t i=0; i<rims.pixels.size(); ++i)
        expect(std::isfinite(corrected.pixels[i].r) && corrected.pixels[i].a==rims.pixels[i].a,
               "Halo filter lost alpha or produced invalid output");
    std::printf("PASS: %u resize fixtures; max error %.7f; edge leakage %.7f -> %.7f; halo preservation\n",
                cases,worst,smoothError,edgeError);
    expect(DlssNrSpatialTransfer(2)==1 && DlssNrSpatialTransfer(4)==3 && DlssNrSpatialTransfer(0)==0 &&
           DlssNrSpatialTransfer(5)==5,
           "Transfer routing changed");
    return 0;
}
catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
