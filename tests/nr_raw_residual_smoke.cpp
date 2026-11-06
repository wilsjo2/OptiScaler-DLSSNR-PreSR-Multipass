// Executes raw inter-pass reconstruction on WARP with nonconstant 2D images.
// No game, NGX model or GPU capture is needed.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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
static float Sample(const Image& image, float u, float v, unsigned channel)
{
    const float px = u * image.width - .5f, py = v * image.height - .5f;
    const int x = static_cast<int>(std::floor(px)), y = static_cast<int>(std::floor(py));
    const float fx = px - x, fy = py - y;
    float result = 0;
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i)
    {
        const int sx = std::clamp(x + i, 0, static_cast<int>(image.width) - 1);
        const int sy = std::clamp(y + j, 0, static_cast<int>(image.height) - 1);
        result += image.pixels[sy * image.width + sx][channel] * (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
    }
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
        L"OptiScaler/shaders/dlssnr/precompile/dlssnr_raw_residual.hlsl", nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE, "CSMain", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0, &code, &errors);
    if (errors) std::fputs(static_cast<const char*>(errors->GetBufferPointer()), stderr);
    Check(compiled);
    ComPtr<ID3D11ComputeShader> shader;
    Check(device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader));

    auto run = [&](const Image& proxy, const Image& answer, const Image& native)
    {
        std::array<ComPtr<ID3D11Texture2D>, 3> textures;
        std::array<ComPtr<ID3D11ShaderResourceView>, 3> views;
        const Image* inputs[] = {&proxy, &answer, &native};
        for (unsigned i = 0; i < 3; ++i)
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
        d.Width = native.width; d.Height = native.height;
        d.MipLevels = d.ArraySize = d.SampleDesc.Count = 1;
        d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        d.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        ComPtr<ID3D11Texture2D> target, readback;
        Check(device->CreateTexture2D(&d, nullptr, &target));
        ComPtr<ID3D11UnorderedAccessView> targetView;
        Check(device->CreateUnorderedAccessView(target.Get(), nullptr, &targetView));
        d.BindFlags = 0; d.Usage = D3D11_USAGE_STAGING; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        Check(device->CreateTexture2D(&d, nullptr, &readback));
        struct Constants { unsigned mode; float whitePoint; unsigned width, height; } constants{
            130, 1, native.width, native.height};
        D3D11_BUFFER_DESC bufferDesc{};
        bufferDesc.ByteWidth = sizeof(constants); bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA initial{&constants, 0, 0};
        ComPtr<ID3D11Buffer> buffer;
        Check(device->CreateBuffer(&bufferDesc, &initial, &buffer));
        D3D11_SAMPLER_DESC samplerDesc{};
        samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
        ComPtr<ID3D11SamplerState> sampler;
        Check(device->CreateSamplerState(&samplerDesc, &sampler));
        ID3D11ShaderResourceView* srvs[]{views[0].Get(), views[1].Get(), views[2].Get()};
        context->CSSetShader(shader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 3, srvs);
        context->CSSetUnorderedAccessViews(0, 1, targetView.GetAddressOf(), nullptr);
        context->CSSetConstantBuffers(0, 1, buffer.GetAddressOf());
        context->CSSetSamplers(0, 1, sampler.GetAddressOf());
        context->Dispatch((native.width + 7) / 8, (native.height + 7) / 8, 1);
        context->ClearState();
        context->CopyResource(readback.Get(), target.Get());
        D3D11_MAPPED_SUBRESOURCE mapped{};
        Check(context->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped));
        Image result(native.width, native.height);
        for (unsigned y = 0; y < native.height; ++y)
            std::memcpy(result.pixels.data() + y * native.width,
                        static_cast<const char*>(mapped.pData) + y * mapped.RowPitch, native.width * sizeof(Pixel));
        context->Unmap(readback.Get(), 0);
        return result;
    };

    // Odd, non-square dimensions exercise both bilinear edges and partial thread groups.
    Image native(19, 13), proxy(7, 5), answer(7, 5);
    for (unsigned y = 0; y < native.height; ++y) for (unsigned x = 0; x < native.width; ++x)
        native.pixels[y * native.width + x] = {
            float(x) / (native.width - 1), float(y) / (native.height - 1),
            ((x / 2 + y / 2) % 2) ? .9f : .03f, float(x + 2 * y) / 50};
    for (unsigned y = 0; y < proxy.height; ++y) for (unsigned x = 0; x < proxy.width; ++x)
        proxy.pixels[y * proxy.width + x] = {.1f + .1f * x, .15f + .12f * y, .4f, 1};
    float worst = 0;
    unsigned floored = 0, aboveOne = 0;
    for (unsigned fixture = 0; fixture < 3; ++fixture)
    {
        answer = proxy;
        for (unsigned y = 0; y < proxy.height; ++y) for (unsigned x = 0; x < proxy.width; ++x)
        {
            auto& p = answer.pixels[y * proxy.width + x];
            if (fixture == 1) { p[0] += .2f; p[1] -= .25f; p[2] += .1f; }
            if (fixture == 2)
            {
                const bool disk = (int(x) - 3) * (int(x) - 3) + (int(y) - 2) * (int(y) - 2) <= 4;
                p[0] += disk ? .35f : -.2f;
                p[1] += .15f * std::sin(float(x + 2 * y));
                p[2] += ((x + y) % 2) ? .3f : -.3f;
            }
        }
        const auto got = run(proxy, answer, native);
        for (unsigned y = 0; y < native.height; ++y) for (unsigned x = 0; x < native.width; ++x)
        {
            const unsigned at = y * native.width + x;
            const float u = (x + .5f) / native.width, v = (y + .5f) / native.height;
            for (unsigned c = 0; c < 4; ++c)
            {
                float expected = native.pixels[at][c];
                if (c < 3)
                    expected = std::max(0.0f, expected + Sample(answer, u, v, c) - Sample(proxy, u, v, c));
                if (!std::isfinite(got.pixels[at][c])) throw std::runtime_error("Nonfinite output");
                const float error = std::abs(got.pixels[at][c] - expected);
                worst = std::max(worst, error);
                // Texture filtering permits finite sub-texel precision; these large signed
                // checker edits expose that rounding more strongly than ordinary images.
                const float tolerance = fixture == 2 && c < 3 ? .0025f : 2e-6f;
                if (error > tolerance) throw std::runtime_error("Raw reconstruction disagrees with CPU bilinear reference");
                if (c < 3 && fixture) { floored += expected == 0; aboveOne += expected > 1; }
            }
        }
    }
    if (!floored || !aboveOne) throw std::runtime_error("Fixtures missed floor/headroom behavior");
    std::printf("Raw residual WARP smoke passed: identity, signed residual, gradient/disk/checker patterns; max error %.8f\n", worst);
    return 0;
}
catch (const std::exception& error)
{
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
