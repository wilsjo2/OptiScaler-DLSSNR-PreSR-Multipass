#include "Processor.h"
#include "Capture.h"
#include "Presentation.h"
#include "CheckCrash.h"
#include "Colour.h"
#include "FramePacer.h"
#include "FeatureChecks.h"
#include "DepthCheck.h"
#include "PassResolutionCheck.h"
#include <DirectXPackedVector.h>
#include <wincodec.h>
#include <fstream>
#include <vector>
#include <numeric>
#include <iostream>
#include <d3d12sdklayers.h>
#include <winrt/base.h>
namespace nr {
void ResidualCheck(const std::filesystem::path& runtime);
void ResizeCheck();
void LoggingCheck();
void FrameGenerationCheck(const std::filesystem::path& plugins, const std::filesystem::path& nrRuntime, bool relocatePlugin);
void CaptureQueueCheck();
void CaptureCheck(Settings settings, UINT width, UINT height, unsigned frames);
void MonitorCheck(Settings settings);
void MotionCheck();
void WindowCheck(Settings settings, const std::wstring& title, unsigned frames);
void PerformanceCheck(Gpu& gpu, Processor& processor, ID3D12Resource* input, Settings settings, unsigned frames, unsigned modeMask, bool moving);
void QualityCheck(Gpu& gpu, Processor& processor, ID3D12Resource* input, Settings settings);
void ReapplyCheck();
void BorrowCheck(const std::filesystem::path& fixture, UINT width, UINT height, const std::filesystem::path& report,
    unsigned preset, const std::filesystem::path& runtime, unsigned frames, bool sameTuning, unsigned style = 0, bool nvfp4 = false, unsigned swin = 0, const std::string& net = {});
}
namespace {
using namespace nr;
using DirectX::PackedVector::HALF;
using DirectX::PackedVector::XMConvertFloatToHalf;
using DirectX::PackedVector::XMConvertHalfToFloat;
std::vector<HALF> Pixels(UINT w, UINT h, const std::filesystem::path& image) {
    if (image.extension() == L".raw") {
        std::vector<HALF> values(size_t(w) * h * 4);
        std::ifstream file(image, std::ios::binary | std::ios::ate);
        if (!file || (size_t)file.tellg() != values.size() * sizeof(HALF))
            throw std::runtime_error("Raw fixture must be tightly packed RGBA16F at the requested size.");
        file.seekg(0); file.read((char*)values.data(), values.size() * sizeof(HALF));
        for (size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(XMConvertHalfToFloat(values[i]))) throw std::runtime_error("Non-finite raw fixture.");
            if (i % 4 == 3) values[i] = XMConvertFloatToHalf(1);
        }
        return values;
    }
    std::vector<BYTE> rgba;
    if (!image.empty()) {
        ComPtr<IWICImagingFactory> factory; Check(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
            CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "WIC factory");
        ComPtr<IWICBitmapDecoder> decoder; Check(factory->CreateDecoderFromFilename(image.c_str(), nullptr,
            GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder), "Read fixture image");
        ComPtr<IWICBitmapFrameDecode> frame; Check(decoder->GetFrame(0, &frame), "Fixture frame");
        ComPtr<IWICBitmapScaler> scaled; Check(factory->CreateBitmapScaler(&scaled), "Fixture scaler");
        Check(scaled->Initialize(frame.Get(), w, h, WICBitmapInterpolationModeFant), "Scale fixture");
        ComPtr<IWICFormatConverter> converted; Check(factory->CreateFormatConverter(&converted), "Fixture converter");
        Check(converted->Initialize(scaled.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone,
            nullptr, 0, WICBitmapPaletteTypeCustom), "Convert fixture");
        rgba.resize(size_t(w) * h * 4); Check(converted->CopyPixels(nullptr, w * 4, (UINT)rgba.size(), rgba.data()), "Read fixture pixels");
    }
    auto linear = [](float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); };
    std::vector<HALF> values(size_t(w) * h * 4);
    for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
        size_t i = (size_t(y) * w + x) * 4;
        for (UINT c = 0; c < 3; ++c) {
            float v = rgba.empty() ? (0.05f + 0.8f * ((x / 24 + y / 24 + c) % 7) / 6.0f) : linear(rgba[i + c] / 255.0f);
            values[i + c] = XMConvertFloatToHalf(v);
        }
        values[i + 3] = XMConvertFloatToHalf(1.0f);
    }
    return values;
}
void SavePng(const std::filesystem::path& file, const std::vector<HALF>& values, UINT w, UINT h) {
    std::vector<BYTE> rgb(size_t(w) * h * 4);
    for (size_t i = 0; i < rgb.size(); ++i) {
        float v = XMConvertHalfToFloat(values[i]);
        if (i % 4 == 3) v = 1;
        else v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(std::max(0.0f, v), 1 / 2.4f) - 0.055f;
        rgb[i] = (BYTE)std::clamp(v * 255, 0.0f, 255.0f);
    }
    ComPtr<IWICImagingFactory> factory; Check(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
        CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "PNG factory");
    ComPtr<IWICStream> stream; Check(factory->CreateStream(&stream), "PNG stream");
    Check(stream->InitializeFromFilename(file.c_str(), GENERIC_WRITE), "PNG file");
    ComPtr<IWICBitmapEncoder> encoder; Check(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder), "PNG encoder");
    Check(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "PNG initialize");
    ComPtr<IWICBitmapFrameEncode> frame; Check(encoder->CreateNewFrame(&frame, nullptr), "PNG frame");
    Check(frame->Initialize(nullptr), "PNG frame init"); Check(frame->SetSize(w, h), "PNG size");
    for (size_t i = 0; i < rgb.size(); i += 4) std::swap(rgb[i], rgb[i + 2]);
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA; Check(frame->SetPixelFormat(&format), "PNG format");
    if (format != GUID_WICPixelFormat32bppBGRA) throw std::runtime_error("PNG encoder rejected BGRA.");
    Check(frame->WritePixels(h, w * 4, (UINT)rgb.size(), rgb.data()), "PNG pixels");
    Check(frame->Commit(), "PNG frame commit"); Check(encoder->Commit(), "PNG commit");
}
void ModelCheck(const Settings& settings, UINT w, UINT h, unsigned count, const std::filesystem::path& image, bool appearanceCheck, bool alignmentCheck, bool historyCheck, bool qualityCheck, bool featureCheck, bool performanceCheck, bool depthCheck, const std::filesystem::path& depthOutput, bool passResolutionCheck, const std::filesystem::path& passResolutionOutput, unsigned performanceModes, bool performanceMoving) {
    InstallPassBorrow(); // As in the app: before the runtime resolves NvAPI, transparent unless a pass borrows.
    Gpu gpu; Processor processor(gpu, settings.runtime); processor.Resize(w, h, settings);
    ComPtr<ID3D12InfoQueue> diagnostics; gpu.device.As(&diagnostics);
    Log("Check: NR creation submitted and completed.");
    // Match live capture's shared, simultaneous-access texture and its state decay.
    auto input = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON, true);
    auto pixels = Pixels(w, h, image); auto desc = input->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    auto upload = gpu.Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD), readback = gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    void* mapped = nullptr; D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    auto uploadPixels = [&] {
        Check(upload->Map(0, nullptr, &mapped), "Upload map");
        for (UINT y = 0; y < h; ++y) memcpy((BYTE*)mapped + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
            pixels.data() + size_t(y) * w * 4, size_t(w) * 8);
        upload->Unmap(0, nullptr);
        gpu.Begin(); Barrier(gpu.commands.Get(), input.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        from = {}; from.pResource = upload.Get(); from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
        to = {}; to.pResource = input.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        Barrier(gpu.commands.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Upload timeout");
    };
    uploadPixels();
    if (depthCheck) {
        std::cout << "Depth fixture: " << (image.empty() ? "generated colour tiles" : image.string()) << '\n';
        nr::DepthCheck(gpu, processor, input.Get(), settings, count,
            depthOutput.empty() ? DataDirectory() / "validation" / "depth-check" : depthOutput, SavePng);
        return;
    }
    if (performanceCheck) { nr::PerformanceCheck(gpu, processor, input.Get(), settings, count, performanceModes, performanceMoving); return; }
    if (qualityCheck) { nr::QualityCheck(gpu, processor, input.Get(), settings); return; }
    auto read = [&](Settings s, bool hdr, float desktopWhite = 0, bool reset = false) {
        processor.Resize(w, h, s);
        gpu.Begin(); auto* result = processor.Run(input.Get(), s, hdr, reset, desktopWhite);
        from = {}; from.pResource = result; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to = {}; to.pResource = readback.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = footprint;
        gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr); gpu.Submit();
        if (!gpu.Drain(30000)) throw std::runtime_error("Readback timeout");
        if (diagnostics) {
            bool failed = false;
            for (UINT64 i = 0; i < diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
                SIZE_T size = 0; diagnostics->GetMessage(i, nullptr, &size); std::vector<BYTE> bytes(size);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
                Check(diagnostics->GetMessage(i, message, &size), "GPU diagnostic");
                if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                    std::cerr << message->pDescription << '\n'; failed = true;
                }
            }
            diagnostics->ClearStoredMessages();
            if (failed) throw std::runtime_error("D3D12 validation failed during processing.");
        }
        std::vector<HALF> values(pixels.size()); Check(readback->Map(0, nullptr, &mapped), "Readback map");
        for (UINT y = 0; y < h; ++y) memcpy(values.data() + size_t(y) * w * 4,
            (BYTE*)mapped + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, size_t(w) * 8);
        readback->Unmap(0, nullptr);
        for (size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(XMConvertHalfToFloat(values[i]))) throw std::runtime_error("Non-finite output pixel.");
            if (i % 4 == 3 && values[i] != pixels[i]) throw std::runtime_error("Composition changed original alpha.");
        }
        return values;
    };
    auto readReset = [&](Settings mode, bool hdr, float white) {
        // DLSS can vary on identical inputs even with Reset=1. Compare colour controls
        // at the same feature age, without weakening exact-pixel endpoint assertions.
        if (mode.upscaler == 1 && mode.modelScale < 100) {
            auto native = mode; native.modelScale = 100; processor.Resize(w, h, native);
        }
        return read(mode, hdr, white, true);
    };
    if (passResolutionCheck) {
        nr::PassResolutionCheck(processor, settings, pixels, uploadPixels, read,
            passResolutionOutput.empty() ? DataDirectory() / "validation" / "pass-resolution" : passResolutionOutput, SavePng);
        return;
    }
    if (featureCheck) { nr::FeatureChecks(processor, settings, pixels, uploadPixels, read); return; }
    Log("Check: fixture uploaded; testing bypass.");
    auto bypass = settings; bypass.strength = 0;
    if (read(bypass, false) != pixels || read(bypass, true) != pixels) throw std::runtime_error("Zero strength changed pixels.");
    Log("Check: bypass passed; evaluating NR.");
    auto result = read(settings, false); double difference = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        const float v = XMConvertHalfToFloat(result[i]); if (!std::isfinite(v)) throw std::runtime_error("Non-finite NR output");
        if (i % 4 != 3) difference += std::abs(v - XMConvertHalfToFloat(pixels[i]));
    }
    difference /= double(w) * h * 3;
    if (difference < 0.00001) throw std::runtime_error("NR returned effectively unchanged pixels.");
    auto directory = DataDirectory() / L"validation"; std::filesystem::create_directories(directory);
    SavePng(directory / std::format(L"before-{}x{}.png", w, h), pixels, w, h);
    SavePng(directory / std::format(L"after-{}x{}.png", w, h), result, w, h);
    if (settings.passes == 2) {
        auto mode = settings; mode.estimateMotion = false; mode.passes = 1;
        auto single = read(mode, false, 0, true); mode.passes = 2;
        auto two = read(mode, false, 0, true); double delta = 0;
        for (size_t i = 0; i < two.size(); ++i) if (i % 4 != 3)
            delta += std::abs(XMConvertHalfToFloat(two[i]) - XMConvertHalfToFloat(single[i]));
        delta /= double(w) * h * 3;
        if (delta < 0.00001) throw std::runtime_error("Second NR pass made no measurable change.");
        std::cout << std::format("PASS 2 -> 1 -> 2 pass switching; second-pass RGB delta {:.6f}\n", delta);
    }
    if (settings.modelScale < 100) {
        auto mode = settings; mode.estimateMotion = false; mode.transfer = 1; mode.upscaler = 0;
        const auto matched = read(mode, false, 0, true);
        const auto mw = processor.modelWidth, mh = processor.modelHeight;
        mode.transfer = 0; const auto classic = read(mode, false, 0, true); double delta = 0;
        for (size_t i = 0; i < matched.size(); ++i) if (i % 4 != 3)
            delta += std::abs(XMConvertHalfToFloat(matched[i]) - XMConvertHalfToFloat(classic[i]));
        delta /= double(w) * h * 3;
        if (delta < 0.000001) throw std::runtime_error("Reduced-resolution transfer modes produced the same picture.");
        mode.modelScale = 100; const auto native = read(mode, false, 0, true);
        mode.transfer = 1;
        if (read(mode, false, 0, true) != native) throw std::runtime_error("Matched transfer changed native-resolution output.");
        mode.modelScale = settings.modelScale;
        if (read(mode, false, 0, true) != matched) throw std::runtime_error("Resolution switching changed reset-frame output.");
        mode.enabled = false;
        if (read(mode, false) != pixels) throw std::runtime_error("Reduced-resolution disable changed original pixels.");
        std::cout << std::format("PASS {}% model ({}x{}), classic/matched RGB delta {:.6f}, native identity, resize round trip and exact disable\n",
            settings.modelScale, mw, mh, delta);
    }
    const auto basePixels = pixels;
    if (historyCheck) {
        auto mode = settings; mode.estimateMotion = false;
        const auto resetFrame = read(mode, false, 0, true);
        mode.estimateMotion = true; unsigned resetCount = 0;
        std::vector<HALF> accumulated;
        for (unsigned frame = 0; frame < 60; ++frame) {
            accumulated = read(mode, false, 0, frame == 0); resetCount += processor.historyReset ? 1u : 0u;
        }
        double resetDelta = 0, historyDelta = 0, difference = 0;
        for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3) {
            const auto base = XMConvertHalfToFloat(pixels[i]);
            resetDelta += std::abs(XMConvertHalfToFloat(resetFrame[i]) - base);
            historyDelta += std::abs(XMConvertHalfToFloat(accumulated[i]) - base);
            difference += std::abs(XMConvertHalfToFloat(accumulated[i]) - XMConvertHalfToFloat(resetFrame[i]));
        }
        if (resetCount != 1) throw std::runtime_error("Static motion history reset unexpectedly.");
        const auto n = double(w) * h * 3;
        std::cout << std::format("History comparison, identical static input/settings: reset edit {:.7f}, frame-60 edit {:.7f}, difference {:.7f}, NR resets {}/60\n",
            resetDelta / n, historyDelta / n, difference / n, resetCount);
        read(settings, false, 0, true);
    }
    if (alignmentCheck) {
        auto mode = settings; mode.modelScale = 100; mode.upscaler = 0; mode.passes = 1;
        mode.estimateMotion = false; mode.referenceWhiteNits = 80; mode.inputColour = 1;
        const auto sdr = readReset(mode, false, 1);
        auto* modelInput = processor.ModelInput();
        gpu.Begin(); Barrier(gpu.commands.Get(), modelInput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        from = {}; from.pResource = modelInput; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to = {}; to.pResource = readback.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = footprint;
        gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        Barrier(gpu.commands.Get(), modelInput, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.Submit(); if (!gpu.Drain()) throw std::runtime_error("Model input readback timeout.");
        Check(readback->Map(0, nullptr, &mapped), "Model input map"); double inputError = 0, sdrDelta = 0;
        for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) for (UINT c = 0; c < 3; ++c) {
            const size_t i = (size_t(y) * w + x) * 4 + c;
            const float v = XMConvertHalfToFloat(pixels[i]);
            const float encoded = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1 / 2.4f) - 0.055f;
            const auto* row = (HALF*)((BYTE*)mapped + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch);
            inputError += std::abs(encoded - XMConvertHalfToFloat(row[x * 4 + c]));
            sdrDelta += std::abs(XMConvertHalfToFloat(sdr[i]) - v);
        }
        readback->Unmap(0, nullptr); const double samples = double(w) * h * 3;
        if (inputError / samples > 0.00025) throw std::runtime_error("Capture SDR input differs from the in-game sRGB input.");
        std::cout << std::format("PASS SDR model input vs original sRGB: {:.7f} mean error; NR edit {:.7f}\n", inputError / samples, sdrDelta / samples);
        for (unsigned tone = 0; tone < 2; ++tone) {
            mode.inputColour = 2; mode.tone = tone; const auto old = readReset(mode, false, 1); double delta = 0;
            for (size_t i = 0; i < old.size(); ++i) if (i % 4 != 3)
                delta += std::abs(XMConvertHalfToFloat(old[i]) - XMConvertHalfToFloat(pixels[i]));
            std::cout << std::format("Same SDR fixture with extra {} curve: edit {:.7f}; corrected/old {:.2f}x\n",
                tone ? "Neutwo" : "Hybrid", delta / samples, sdrDelta / std::max(delta, 1e-12));
        }
        mode = settings; mode.inputColour = 1; mode.referenceWhiteNits = 0; mode.estimateMotion = false;
        const auto sdr80 = readReset(mode, true, 1);
        for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
            pixels[i] = XMConvertFloatToHalf(XMConvertHalfToFloat(pixels[i]) * 4);
        uploadPixels(); const auto sdr320 = readReset(mode, true, 4);
        double sdrWhiteError = 0, sdrWhiteMax = 0;
        for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3) {
            const double error = std::abs(XMConvertHalfToFloat(sdr320[i]) / 4 - XMConvertHalfToFloat(sdr80[i]));
            sdrWhiteError += error; sdrWhiteMax = std::max(sdrWhiteMax, error);
        }
        std::cout << std::format("SDR-on-HDR white invariance: mean {:.9f}, max {:.9f}\n", sdrWhiteError / samples, sdrWhiteMax);
        if (sdrWhiteMax > 0.00000006) throw std::runtime_error("SDR-on-HDR brightness normalization changed the effect.");
        pixels = basePixels; uploadPixels(); read(settings, false, 0, true);
        std::cout << "PASS explicit SDR on HDR desktop: normalized 80/320-nit output within one FP16 subnormal step\n";
    }
    if (settings.upscaler == 1 && settings.modelScale < 100) {
        auto mode = settings; mode.estimateMotion = false;
        const auto dlss = read(mode, false, 0, true);
        mode.upscaler = 0; const auto bilinear = read(mode, false, 0, true);
        if (dlss == bilinear) throw std::runtime_error("Private DLSS output is identical to bilinear.");
        mode.upscaler = 1;
        if (read(mode, false, 0, true) != dlss) throw std::runtime_error("Private DLSS reset output changed after switching backends.");
        std::cout << "PASS private DLSS produces distinct pixels; DLSS -> bilinear -> DLSS reset output identical\n";
    }
    if (appearanceCheck) {
        auto mode = settings; mode.estimateMotion = false; mode.referenceWhiteNits = 80;
        const auto normal = readReset(mode, false, 1);
        mode.colourStrength = 0; const auto originalColour = readReset(mode, false, 1);
        mode.colourStrength = 4; const auto boosted = readReset(mode, false, 1);
        if (normal == originalColour || normal == boosted) throw std::runtime_error("Colour strength did not change output.");
        mode.colourStrength = 1; mode.separateSkin = true;
        if (readReset(mode, false, 1) != normal) throw std::runtime_error("Unity skin controls changed the default image.");
        mode.skinDetail = mode.skinColour = mode.environmentDetail = mode.environmentColour = 0;
        if (readReset(mode, false, 1) != pixels) throw std::runtime_error("Full protection did not preserve original pixels.");
        // Known warm and blue patches exercise selection independently of the model recognizing a face.
        const float warm[]{0.603827f, 0.263273f, 0.132868f}, blue[]{0.033105f, 0.132868f, 0.603827f};
        for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) for (UINT c = 0; c < 3; ++c)
            pixels[(size_t(y) * w + x) * 4 + c] = XMConvertFloatToHalf(x < w / 2 ? warm[c] : blue[c]);
        uploadPixels(); mode.previewSkin = true;
        const auto mask = readReset(mode, false, 1);
        for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
            const size_t i = (size_t(y) * w + x) * 4; const float value = XMConvertHalfToFloat(mask[i]);
            if (mask[i] != mask[i + 1] || mask[i] != mask[i + 2] || (x < w / 2 ? value < 0.99f : value > 0.001f))
                throw std::runtime_error("Skin mask preview did not select the expected colours.");
        }
        mode.previewSkin = false; mode.separateSkin = false;
        const auto patchResult = readReset(mode, false, 1);
        mode.separateSkin = true; mode.environmentDetail = mode.environmentColour = 1;
        const auto protectedSkin = readReset(mode, false, 1);
        mode.skinDetail = mode.skinColour = 1; mode.environmentDetail = mode.environmentColour = 0;
        const auto protectedEnvironment = readReset(mode, false, 1);
        for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) for (UINT c = 0; c < 3; ++c) {
            const size_t i = (size_t(y) * w + x) * 4 + c;
            if (protectedSkin[i] != (x < w / 2 ? pixels[i] : patchResult[i]) ||
                protectedEnvironment[i] != (x < w / 2 ? patchResult[i] : pixels[i]))
                throw std::runtime_error("Skin/environment protection affected the wrong region.");
        }
        mode.model.skinStructure = 2; readReset(mode, false, 1);
        for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
            pixels[i] = XMConvertFloatToHalf(XMConvertHalfToFloat(pixels[i]) * 4);
        uploadPixels(); mode.referenceWhiteNits = 320;
        mode.skinDetail = mode.skinColour = mode.environmentDetail = mode.environmentColour = 0;
        for (unsigned tone = 0; tone < 2; ++tone) {
            mode.tone = tone;
            if (readReset(mode, true, 4) != pixels) throw std::runtime_error("Skin protection changed HDR originals.");
        }
        pixels = basePixels; uploadPixels(); read(settings, false, 0, true);
        std::cout << "PASS colour 0/1/4, unity skin identity, exact SDR/HDR protection, skin mask and independent region protection; model skin structure accepted\n";
    }
    for (unsigned tone = 0; tone < 2; ++tone) {
        auto mode = settings; mode.tone = tone; mode.estimateMotion = false; mode.referenceWhiteNits = 0;
        pixels = basePixels; uploadPixels();
        const auto reference = readReset(mode, true, 1.0f);
        if (settings.upscaler == 1) {
            const auto repeated = read(mode, true, 1.0f, true); double repeatError = 0;
            for (size_t i = 0; i < repeated.size(); ++i) if (i % 4 != 3)
                repeatError += std::abs(XMConvertHalfToFloat(repeated[i]) - XMConvertHalfToFloat(reference[i]));
            std::cout << std::format("Private DLSS identical-input reset-frame variation: {:.7f}\n", repeatError / (double(w) * h * 3));
        }
        // The same SDR image at 80 and 320 nits must produce the same relative edit.
        for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
            pixels[i] = XMConvertFloatToHalf(XMConvertHalfToFloat(pixels[i]) * 4);
        uploadPixels(); const auto adjusted = readReset(mode, true, 4.0f);
        const auto legacy = readReset(mode, true, 203.0f / 80.0f);
        double error = 0, legacyError = 0;
        for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3) {
            const auto expected = XMConvertHalfToFloat(reference[i]);
            error += std::abs(XMConvertHalfToFloat(adjusted[i]) / 4 - expected);
            legacyError += std::abs(XMConvertHalfToFloat(legacy[i]) / 4 - expected);
        }
        error /= double(w) * h * 3; legacyError /= double(w) * h * 3;
        std::cout << std::format("{} SDR-white invariance: corrected {:.7f}, fixed-203-nit {:.7f} mean RGB error\n",
            tone ? "Neutwo" : "Hybrid", error, legacyError);
        if (!std::isfinite(error) || error > 0.00005) throw std::runtime_error("Desktop white level changed the relative NR effect.");
        mode.referenceWhiteNits = 320;
        if (readReset(mode, true, 1.0f) != adjusted) throw std::runtime_error("Explicit reference white was not applied.");
    }
    pixels = basePixels; uploadPixels();
    if (settings.estimateMotion) {
        // First-frame resets alone do not verify continuous temporal evaluation.
        const auto originalPixels = pixels;
        for (unsigned frame = 1; frame <= 12; ++frame) {
            for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) for (UINT c = 0; c < 4; ++c)
                pixels[(size_t(y) * w + x) * 4 + c] = originalPixels[(size_t((y + frame * 8) % h) * w + (x + frame * 12) % w) * 4 + c];
            uploadPixels();
            auto mode = settings; mode.estimateMotion = frame > 3;
            auto continuous = read(mode, false); double delta = 0;
            for (size_t i = 0; i < continuous.size(); ++i) if (i % 4 != 3) {
                const auto value = XMConvertHalfToFloat(continuous[i]);
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite temporal output.");
                delta += std::abs(value - XMConvertHalfToFloat(pixels[i]));
            }
            delta /= double(w) * h * 3;
            std::cout << std::format("Moving frame {} ({} motion): mean RGB delta {:.6f}\n", frame, mode.estimateMotion ? "estimated" : "zero", delta);
            if (delta < 0.00001) throw std::runtime_error("Temporal NR stopped changing pixels.");
        }
    }
    std::vector<double> timings;
    for (unsigned i = 0; i < count; ++i) {
        const auto start = ClockMs(); gpu.Begin(); processor.Run(input.Get(), settings, false); gpu.Submit();
        if (!gpu.Drain(30000)) throw std::runtime_error("Benchmark timeout");
        timings.push_back(ClockMs() - start);
    }
    std::sort(timings.begin(), timings.end()); const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / count;
    DXGI_QUERY_VIDEO_MEMORY_INFO memory{}; gpu.adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory);
    auto report = std::format("PASS {}x{}: mean RGB delta {:.6f}; zero-strength SDR/HDR exact; {:.2f} ms mean, {:.2f} ms p95, {:.1f} serial fps; {:.0f} MiB app VRAM",
        w, h, difference, mean, timings[std::min<size_t>(count - 1, size_t(count * 0.95))], 1000 / mean, memory.CurrentUsage / 1048576.0);
    std::cout << report << std::endl; Log(report);
    // HDR highlights must survive both codecs. This is numerical validation;
    // it does not claim that a physical HDR monitor has been visually calibrated.
    for (size_t i = 0; i < pixels.size(); ++i) if (i % 4 != 3)
        pixels[i] = XMConvertFloatToHalf(std::max(0.0f, XMConvertHalfToFloat(pixels[i])) * 8);
    pixels[0] = pixels[1] = pixels[2] = XMConvertFloatToHalf(8);
    uploadPixels();
    if (read(bypass, true) != pixels) throw std::runtime_error("HDR bypass changed highlights.");
    for (unsigned tone = 0; tone < 2; ++tone) for (unsigned frame = 0; frame < 3; ++frame) {
        auto hdrSettings = settings; hdrSettings.tone = tone; auto hdr = read(hdrSettings, true); float maximum = 0;
        double delta = 0;
        for (size_t i = 0; i < hdr.size(); ++i) if (i % 4 != 3) {
            const float value = XMConvertHalfToFloat(hdr[i]);
            if (!std::isfinite(value)) throw std::runtime_error("Non-finite HDR output."); maximum = std::max(maximum, value);
            delta += std::abs(value - XMConvertHalfToFloat(pixels[i]));
        }
        if (maximum <= 1) throw std::runtime_error("HDR highlights were clipped to SDR.");
        if (delta / (double(w) * h * 3) < 0.00001) throw std::runtime_error("HDR temporal NR stopped changing pixels.");
    }
    std::cout << "PASS HDR highlight preservation, finite output, hybrid/Neutwo and exact HDR bypass\n";
}
void RuntimeLifecycleCheck(const Settings& settings) {
    // A focus/capture failure may end a session before its first model evaluation.
    // Keep the process alive after shutdown so delayed driver callbacks can run.
    for (unsigned i = 0; i < 5; ++i) {
        { Gpu gpu; Runtime runtime(gpu.device.Get(), settings.runtime); }
        Sleep(200);
        if (!GetModuleHandleW(L"_nvngx.dll")) throw std::runtime_error("NGX dispatcher unloaded after session shutdown.");
    }
    std::cout << "PASS five NGX initialize/shutdown cycles; dispatcher remains resident\n";
}
void PacingCheck() {
    FramePacer pacer; Handle stop{CreateEventW(nullptr, TRUE, FALSE, nullptr)}, capture{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    for (unsigned fps : {30u, 120u, 240u}) {
        double first = 0, previous = 0;
        for (unsigned frame = 0; frame < 12; ++frame) {
            while (!pacer.Due(fps, ClockMs())) pacer.Wait(stop.value, capture.value);
            const auto now = ClockMs(); if (!frame) first = now;
            // Frame k may not start before its slot on the grid the first frame set; a late frame is not made up beyond the slack.
            if (frame && now - first < frame * 1000.0 / fps - 0.05) throw std::runtime_error("FPS cap submitted too early.");
            if (frame && now - previous < 1000.0 / fps - std::min(2.0, 250.0 / fps) - 0.05) throw std::runtime_error("FPS cap caught up in a burst.");
            pacer.Submitted(fps, now); previous = now;
        }
        const double measured = 11000.0 / (previous - first);
        if (measured < fps * 0.99) throw std::runtime_error(std::format("FPS cap {} only reached {:.1f} fps.", fps, measured));
        std::cout << std::format("PASS timer cap {} fps: {:.1f} fps measured (no capture/presentation)\n", fps, measured);
    }
    pacer.Submitted(1, ClockMs());
    if (!pacer.Due(0, ClockMs())) throw std::runtime_error("Uncapped mode waited for a deadline.");
    SetEvent(capture.value); pacer.Wait(stop.value, capture.value);
    if (pacer.Due(1, ClockMs())) throw std::runtime_error("New capture bypassed the FPS cap.");
    SetEvent(stop.value); const auto start = ClockMs(); pacer.Wait(stop.value, capture.value);
    if (ClockMs() - start > 50) throw std::runtime_error("Stop did not interrupt frame pacing.");
    std::cout << "PASS uncapped, new capture wake and stop interrupt\n";
}
}
void GpuWaitCheck();
#ifdef NR_LAB_COMMANDS
namespace nr { bool LabCommand(int argc, wchar_t** argv); }
#endif

int wmain(int argc, wchar_t** argv) {
    SetUnhandledExceptionFilter(CheckCrash);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try {
        nr::Settings settings; UINT width = 1920, height = 1080, frames = 20; std::filesystem::path image, fgPlugins, depthOutput, passResolutionOutput; std::wstring window;
        bool depthCheck = false, passResolutionCheck = false, performanceMoving = false;
        unsigned performanceModes = 15;
        bool capture = false, monitor = false, lifecycle = false, motion = false, displayInfo = false, appearance = false, alignment = false, history = false, pacing = false, quality = false, features = false, performance = false, captureQueue = false, residual = false, logging = false, resizeCheck = false;
        bool reapplyCheck = false, settingsCheck = false;
#ifdef NR_LAB_COMMANDS
        if (nr::LabCommand(argc, argv)) return 0;      // further commands of a local build
#endif
        if (argc >= 6 && std::wstring(argv[1]) == L"--borrow-check") {
            // GPU/NR harness for manual runs: --borrow-check <fixture.rgba16f> <width> <height> <out.json> [--preset N] [--runtime dll] [--frames N] [--same-tuning 1] [--style 0|1|2]
            unsigned preset = 0, count = 8, style = 0, swin = 0; bool sameTuning = false, nvfp4 = false; std::filesystem::path runtime; std::string net;
            for (int i = 6; i < argc; i += 2) {
                const std::wstring key = argv[i]; if (i + 1 >= argc) throw std::runtime_error("Missing argument value.");
                if (key == L"--preset") preset = std::stoul(argv[i + 1]);
                else if (key == L"--runtime") runtime = argv[i + 1];
                else if (key == L"--frames") count = std::stoul(argv[i + 1]);
                else if (key == L"--same-tuning") sameTuning = std::stoul(argv[i + 1]) != 0;
                else if (key == L"--style") style = std::stoul(argv[i + 1]);
                else if (key == L"--nvfp4") nvfp4 = std::stoul(argv[i + 1]) != 0;
                else if (key == L"--nvfp4swin") swin = std::stoul(argv[i + 1]) & 15u;
                else if (key == L"--nvfp4net") net = nr::Narrow(argv[i + 1]);
                else throw std::runtime_error("Unknown borrow-check argument.");
            }
            nr::BorrowCheck(argv[2], std::stoul(argv[3]), std::stoul(argv[4]), argv[5], preset, runtime, count, sameTuning, style, nvfp4, swin, net);
            return 0;
        }
        for (int i = 1; i < argc; ++i) {
            std::wstring key = argv[i]; if (i + 1 >= argc) throw std::runtime_error("Missing argument value.");
            if (key == L"--runtime-path-check") { nr::RuntimePathCheck(); return 0; }
            if (key == L"--runtime") settings.runtime = argv[++i];
            else if (key == L"--settings") settings = nr::LoadSettings(argv[++i]);
            else if (key == L"--width") width = std::stoul(argv[++i]);
            else if (key == L"--height") height = std::stoul(argv[++i]);
            else if (key == L"--frames") frames = std::stoul(argv[++i]);
            else if (key == L"--image") image = argv[++i];
            else if (key == L"--capture") capture = std::stoul(argv[++i]) != 0;
            else if (key == L"--monitor") monitor = std::stoul(argv[++i]) != 0;
            else if (key == L"--lifecycle") lifecycle = std::stoul(argv[++i]) != 0;
            else if (key == L"--motion-check") motion = std::stoul(argv[++i]) != 0;
            else if (key == L"--motion") settings.estimateMotion = std::stoul(argv[++i]) != 0;
            else if (key == L"--passes") settings.passes = std::stoul(argv[++i]);
            else if (key == L"--scale") settings.modelScale = std::stoul(argv[++i]);
            else if (key == L"--upscaler") settings.upscaler = std::stoul(argv[++i]);
            else if (key == L"--appearance-check") appearance = std::stoul(argv[++i]) != 0;
            else if (key == L"--alignment-check") alignment = std::stoul(argv[++i]) != 0;
            else if (key == L"--history-check") history = std::stoul(argv[++i]) != 0;
            else if (key == L"--quality-check") quality = std::stoul(argv[++i]) != 0;
            else if (key == L"--depth-check") depthCheck = std::stoul(argv[++i]) != 0;
            else if (key == L"--depth-output") depthOutput = argv[++i];
            else if (key == L"--pass-resolution-check") passResolutionCheck = std::stoul(argv[++i]) != 0;
            else if (key == L"--pass-resolution-output") passResolutionOutput = argv[++i];
            else if (key == L"--fg-check") fgPlugins = argv[++i];
            else if (key == L"--capture-queue-check") captureQueue = std::stoul(argv[++i]) != 0;
            else if (key == L"--performance-check") performance = std::stoul(argv[++i]) != 0;
            else if (key == L"--performance-modes") performanceModes = std::stoul(argv[++i]);
            else if (key == L"--performance-moving") performanceMoving = std::stoul(argv[++i]) != 0;
            else if (key == L"--residual-check") residual = std::stoul(argv[++i]) != 0;
            else if (key == L"--resize-check") resizeCheck = std::stoul(argv[++i]) != 0;
            else if (key == L"--logging-check") logging = std::stoul(argv[++i]) != 0;
            else if (key == L"--reapply-check") reapplyCheck = std::stoul(argv[++i]) != 0;
            else if (key == L"--settings-check") settingsCheck = std::stoul(argv[++i]) != 0;
            else if (key == L"--feature-check") features = std::stoul(argv[++i]) != 0;
            else if (key == L"--pacing-check") pacing = std::stoul(argv[++i]) != 0;
            else if (key == L"--input-colour") settings.inputColour = std::stoul(argv[++i]);
            else if (key == L"--codec") settings.tone = std::stoul(argv[++i]);
            else if (key == L"--intensity") settings.model.intensity = std::stof(argv[++i]);
            else if (key == L"--local-tone") settings.model.localTone = std::stof(argv[++i]);
            else if (key == L"--transfer") settings.transfer = std::stoul(argv[++i]);
            else if (key == L"--display-info") displayInfo = std::stoul(argv[++i]) != 0;
            else if (key == L"--window") window = argv[++i];
            else throw std::runtime_error("Unknown argument.");
        }
        if (logging) { nr::LoggingCheck(); return 0; }
        if (settingsCheck) { nr::SettingsCheck(); return 0; }
        if (reapplyCheck) { nr::ReapplyCheck(); return 0; }
        if (resizeCheck) { nr::ResizeCheck(); return 0; }
        if (displayInfo) {
            const auto white = nr::DesktopWhitePoint(MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY));
            std::cout << std::format("Primary display SDR reference white: {:.1f} nits ({:.4f} scRGB; zero means unavailable)\n", white * 80, white);
            return 0;
        }
        if (!fgPlugins.empty()) {
            if (lifecycle && settings.runtime.empty())
                throw std::runtime_error("FG restart checks require --runtime to exercise the application's NGX initialization order.");
            // Restart in the same process: NGX may retain callbacks across shutdown.
            for (unsigned cycle = 0; cycle < (lifecycle ? 3u : 1u); ++cycle) {
                std::cout << "FG session " << cycle + 1 << '\n' << std::flush;
                nr::FrameGenerationCheck(fgPlugins, settings.runtime, lifecycle);
            }
            return 0;
        }
        if (captureQueue) { nr::CaptureQueueCheck(); return 0; }
        if (pacing) { PacingCheck(); GpuWaitCheck(); return 0; }
        if (!frames || frames > 10000 || settings.runtime.empty()) throw std::runtime_error("Use --runtime <NR DLL> [--width N --height N --frames N --image fixture.png].");
        if (residual) { nr::ResidualCheck(settings.runtime); return 0; }
        if (!window.empty()) nr::WindowCheck(settings,window,frames);
        else if (motion) nr::MotionCheck();
        else if (lifecycle) RuntimeLifecycleCheck(settings);
        else if (monitor) nr::MonitorCheck(settings);
        else if (capture) { if (frames < 10) throw std::runtime_error("Capture checks require at least 10 frames."); nr::CaptureCheck(settings, width, height, frames); }
        else ModelCheck(settings, width, height, frames, image, appearance, alignment, history, quality, features, performance, depthCheck, depthOutput, passResolutionCheck, passResolutionOutput, performanceModes, performanceMoving);
        return 0;
    } catch (const winrt::hresult_error& e) { std::cerr << nr::Narrow(e.message().c_str()) << '\n'; return 1; }
      catch (const std::exception& e) { std::cerr << e.what() << '\n'; nr::Log(e.what()); return 1; }
}
