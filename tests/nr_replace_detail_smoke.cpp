// Headless shader test for Restore Sharpness (ReplaceDetailStrength): Windows D3D11 WARP executes the
// shared HLSL and the production blob, no game or NR DLL. The frame is 9x9 with one vertical luminance edge
// (left of x=5 dark, right bright) and the model's answer is a flat colour, so the only structure in the
// output can come from the injection.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "../OptiScaler/shaders/dlssnr/DlssNr_Common.h"
#include "../OptiScaler/shaders/dlssnr/precompile/DlssNr_Shader.h"
using Microsoft::WRL::ComPtr;

struct Pixel { float r, g, b, a; };
constexpr int kSize = 9;

static void check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D call failed"); }
static void expect(bool ok, const char* label) { if (!ok) throw std::runtime_error(label); }
static bool closeTo(float a, float b, float tolerance) { return std::abs(a - b) <= tolerance; }
static float luma(const Pixel& p) { return 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b; }

int wmain(int argc, wchar_t** argv) try {
    if (argc != 2) throw std::runtime_error("Pass the dlssnr.hlsl path");

    // The new fields must sit after the residual block, so nothing before them moves, and inside the 256
    // byte constant buffer. The shader declares four placeholders for the residual block; if these offsets
    // drift the shader reads the wrong floats and the cases below fail rather than pass by accident.
    static_assert(offsetof(DlssNrConstants, EnvironmentColour) == 112);
    static_assert(offsetof(DlssNrConstants, ResidualMotionBaseY) == 128);
    static_assert(offsetof(DlssNrConstants, ReplaceDetailStrength) == 132);
    static_assert(offsetof(DlssNrConstants, ModelWorkScale) == 136);
    static_assert(sizeof(DlssNrConstants) == 256);

    ComPtr<ID3DBlob> code, errors;
    HRESULT compiled = D3DCompileFromFile(argv[1], nullptr, nullptr, "CSMain", "cs_5_0",
                                          D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (errors) std::fprintf(stderr, "%s", (char*) errors->GetBufferPointer());
    check(compiled);

    ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> ctx;
    check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device,
                            nullptr, &ctx));
    ComPtr<ID3D11ComputeShader> fromSource, production;
    check(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &fromSource));
    check(device->CreateComputeShader(DlssNr_cso, sizeof(DlssNr_cso), nullptr, &production));

    // Frame: luminance 0.2 for x <= 4, 0.8 for x >= 5, grey. Model answer: flat (0.6, 0.4, 0.2).
    std::vector<Pixel> original(kSize * kSize), model(kSize * kSize);
    for (int y = 0; y < kSize; ++y)
        for (int x = 0; x < kSize; ++x)
        {
            const float v = x <= 4 ? 0.2f : 0.8f;
            original[y * kSize + x] = { v, v, v, 1.0f };
            model[y * kSize + x] = { 0.6f, 0.4f, 0.2f, 1.0f };
        }

    D3D11_TEXTURE2D_DESC desc {};
    desc.Width = kSize; desc.Height = kSize; desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    ComPtr<ID3D11Texture2D> originalTex, modelTex, outputTex, keepTex, readback;
    D3D11_SUBRESOURCE_DATA data { original.data(), sizeof(Pixel) * kSize, 0 };
    check(device->CreateTexture2D(&desc, &data, &originalTex));
    data.pSysMem = model.data(); check(device->CreateTexture2D(&desc, &data, &modelTex));
    ComPtr<ID3D11ShaderResourceView> originalSrv, modelSrv;
    check(device->CreateShaderResourceView(originalTex.Get(), nullptr, &originalSrv));
    check(device->CreateShaderResourceView(modelTex.Get(), nullptr, &modelSrv));
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    check(device->CreateTexture2D(&desc, nullptr, &outputTex)); check(device->CreateTexture2D(&desc, nullptr, &keepTex));
    ComPtr<ID3D11UnorderedAccessView> outputUav, keepUav;
    check(device->CreateUnorderedAccessView(outputTex.Get(), nullptr, &outputUav));
    check(device->CreateUnorderedAccessView(keepTex.Get(), nullptr, &keepUav));
    desc.BindFlags = 0; desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    check(device->CreateTexture2D(&desc, nullptr, &readback));
    D3D11_BUFFER_DESC buffer {}; buffer.ByteWidth = sizeof(DlssNrConstants); buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ComPtr<ID3D11Buffer> constants; check(device->CreateBuffer(&buffer, nullptr, &constants));
    D3D11_SAMPLER_DESC sampling {}; sampling.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampling.AddressU = sampling.AddressV = sampling.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampling.MaxLOD = D3D11_FLOAT32_MAX;
    ComPtr<ID3D11SamplerState> sampler; check(device->CreateSamplerState(&sampling, &sampler));

    ID3D11ShaderResourceView* srvs[] = { originalSrv.Get(), modelSrv.Get(), originalSrv.Get(), originalSrv.Get(),
                                         originalSrv.Get() };
    ID3D11UnorderedAccessView* uavs[] = { outputUav.Get(), keepUav.Get() };
    ctx->CSSetShaderResources(0, 5, srvs);
    ctx->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
    ctx->CSSetConstantBuffers(0, 1, constants.GetAddressOf());
    ctx->CSSetSamplers(0, 1, sampler.GetAddressOf());

    // Passthrough frame (already display-referred, normalisation is 1), Replace by default.
    const auto makeSettings = [](uint32_t reversible, float strength, float workScale)
    {
        DlssNrConstants s {};
        s.Mode = DlssNrMode_Resolve; s.Width = kSize; s.Height = kSize; s.WhitePoint = 1; s.Passthrough = 1;
        s.ApplyModel = 1; s.ReversibleMode = reversible; s.TransferStrength = s.ColourStrength = 1; s.MaxRatio = 2;
        s.SkinDetail = s.SkinColour = s.EnvironmentDetail = s.EnvironmentColour = 1;
        s.ReplaceDetailStrength = strength; s.ModelWorkScale = workScale;
        return s;
    };
    const auto run = [&](ID3D11ComputeShader* shader, const DlssNrConstants& s)
    {
        ctx->CSSetShader(shader, nullptr, 0);
        ctx->UpdateSubresource(constants.Get(), 0, nullptr, &s, 0, 0);
        ctx->Dispatch((kSize + 7) / 8, (kSize + 7) / 8, 1);
        ctx->CopyResource(readback.Get(), outputTex.Get());
        D3D11_MAPPED_SUBRESOURCE mapped {};
        check(ctx->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped));
        std::vector<Pixel> result(kSize * kSize);
        for (int y = 0; y < kSize; ++y)
            std::memcpy(&result[y * kSize], (const char*) mapped.pData + y * mapped.RowPitch, sizeof(Pixel) * kSize);
        ctx->Unmap(readback.Get(), 0);
        return result;
    };
    const auto at = [](const std::vector<Pixel>& image, int x, int y) { return image[y * kSize + x]; };
    const auto identicalToModel = [&](const std::vector<Pixel>& image)
    {
        for (const Pixel& p : image)
            if (!closeTo(p.r, 0.6f, 1e-4f) || !closeTo(p.g, 0.4f, 1e-4f) || !closeTo(p.b, 0.2f, 1e-4f)) return false;
        return true;
    };
    const auto identical = [&](const std::vector<Pixel>& a, const std::vector<Pixel>& b)
    {
        for (size_t i = 0; i < a.size(); ++i)
            if (!closeTo(a[i].r, b[i].r, 1e-5f) || !closeTo(a[i].g, b[i].g, 1e-5f) || !closeTo(a[i].b, b[i].b, 1e-5f)) return false;
        return true;
    };

    struct Target { const char* name; ID3D11ComputeShader* shader; };
    for (const Target& target : { Target { "HLSL source", fromSource.Get() }, Target { "production blob", production.Get() } })
    {
        // Off: strength 0, a full-size model, a supersampled model, and a constant that was never set (the
        // zero every other dispatch carries) all leave the model's answer untouched.
        expect(identicalToModel(run(target.shader, makeSettings(2, 0.0f, 0.5f))), "strength 0 changed Replace output");
        expect(identicalToModel(run(target.shader, makeSettings(2, 1.0f, 1.0f))), "full-size model changed Replace output");
        expect(identicalToModel(run(target.shader, makeSettings(2, 1.0f, 2.0f))), "supersampled model changed Replace output");
        expect(identicalToModel(run(target.shader, makeSettings(2, 1.0f, 0.0f))), "unset ModelWorkScale changed Replace output");

        // Composed modes never reach the injection: strength must not matter.
        for (uint32_t composed : { 0u, 1u, 3u })
            expect(identical(run(target.shader, makeSettings(composed, 0.0f, 0.5f)),
                             run(target.shader, makeSettings(composed, 2.0f, 0.5f))),
                   "a Composed mode reached the detail injection");

        // On: model ran at half size (radius 2). Flat area, dark side of the edge, bright side of the edge.
        const auto on = run(target.shader, makeSettings(2, 1.0f, 0.5f));
        for (const Pixel& p : on)
            expect(std::isfinite(p.r) && std::isfinite(p.g) && std::isfinite(p.b) && p.r >= 0 && p.g >= 0 && p.b >= 0,
                   "injection produced a non-finite or negative value");

        const Pixel flat = at(on, 2, 4);      // every tap lands in the flat dark area
        expect(closeTo(flat.r, 0.6f, 1e-4f) && closeTo(flat.g, 0.4f, 1e-4f) && closeTo(flat.b, 0.2f, 1e-4f),
               "flat area was changed");

        // Flat pixels on the frame's own border: taps that fall outside the frame must repeat the edge, not
        // read black, or a ring of pixels around the whole frame would be pushed brighter.
        for (const Pixel& border : { at(on, 0, 4), at(on, 2, 0), at(on, 2, 8), at(on, 0, 0) })
            expect(closeTo(border.r, 0.6f, 1e-4f) && closeTo(border.g, 0.4f, 1e-4f) && closeTo(border.b, 0.2f, 1e-4f),
                   "flat pixel on the frame border was changed");

        // Dark side: original 0.2, taps average to 0.32, so the ratio is (0.2 - 0.12 + f) / (0.2 + f) with the
        // shader's floor f = 1/512. The model's colour is scaled by it; its hue must survive.
        const Pixel dark = at(on, 4, 4);
        const float darkRatio = (0.2f - 0.12f + 1.0f / 512) / (0.2f + 1.0f / 512);
        expect(closeTo(dark.r, 0.6f * darkRatio, 0.003f), "dark side of the edge did not get the expected ratio");
        expect(closeTo(dark.r / dark.g, 1.5f, 0.001f) && closeTo(dark.b / dark.g, 0.5f, 0.001f),
               "injection shifted hue on the dark side");

        // Bright side: original 0.8, taps average to 0.68, so the edge is pushed up.
        const Pixel bright = at(on, 5, 4);
        const float brightRatio = (0.8f + 0.12f + 1.0f / 512) / (0.8f + 1.0f / 512);
        expect(closeTo(bright.r, 0.6f * brightRatio, 0.003f), "bright side of the edge did not get the expected ratio");
        expect(closeTo(bright.r / bright.g, 1.5f, 0.001f), "injection shifted hue on the bright side");
        expect(luma(dark) < luma(Pixel { 0.6f, 0.4f, 0.2f, 1 }) && luma(bright) > luma(Pixel { 0.6f, 0.4f, 0.2f, 1 }),
               "edge contrast was not restored");

        // Monotonic in strength, and the hybrid Replace choice takes the same path.
        const auto stronger = run(target.shader, makeSettings(2, 2.0f, 0.5f));
        expect(luma(at(stronger, 4, 4)) < luma(dark) && luma(at(stronger, 5, 4)) > luma(bright),
               "a higher strength did not restore more");
        const auto hybrid = run(target.shader, makeSettings(4, 1.0f, 0.5f));
        expect(closeTo(at(hybrid, 4, 4).r, dark.r, 1e-4f) && closeTo(at(hybrid, 5, 4).r, bright.r, 1e-4f),
               "Hybrid Replace did not take the injection");
        std::printf("PASS (%s): off cases identical, Composed unreachable, edge restored hue-preserving and finite\n",
                    target.name);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
