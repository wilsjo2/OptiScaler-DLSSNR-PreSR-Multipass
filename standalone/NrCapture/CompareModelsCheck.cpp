#include "Processor.h"
#include "ModelCompare.h"
#include <DirectXPackedVector.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
// GPU harness for manual use (needs the NVIDIA GPU and the NR runtime):
//   NrCaptureCheck --compare-models-check <fixture.rgba16f> <width> <height> <model> <other model> [--runtime dll] [--frames N]
// One fixture frame through the model comparison as the application runs it: the picture's own passes with <model>, the
// same passes with <other model> on a second processor, and the two put together as a wipe at 0.3 and side by side.
// Passes when every pixel of each composed picture is the pixel of the side it belongs to (the two-pixel border line
// aside) and the two sides differ. <model> values are ModelSettings::preset (Settings::ModelAInt8 = 25 and so on).
namespace nr {
namespace {
using DirectX::PackedVector::HALF;
using DirectX::PackedVector::XMConvertFloatToHalf;
using DirectX::PackedVector::XMConvertHalfToFloat;
float Decode(float v) { v = std::max(v, 0.f); return v <= .04045f ? v / 12.92f : std::pow((v + .055f) / 1.055f, 2.4f); }
}
void CompareModelsCheck(const std::filesystem::path& fixture, UINT w, UINT h, unsigned model, unsigned otherModel,
    const std::filesystem::path& runtimePath, unsigned frames) {
    if (!w || !h || w > 7680 || h > 4320 || !Settings::IsModel(model) || !Settings::IsModel(otherModel) || frames < 1)
        throw std::runtime_error("Compare-models check: invalid extent, model or frame count.");
    InstallPassBorrow();
    std::ifstream file(fixture, std::ios::binary | std::ios::ate);
    std::vector<HALF> pixels(size_t(w) * h * 4);
    if (!file || size_t(file.tellg()) != pixels.size() * sizeof(HALF)) throw std::runtime_error("Fixture must be tightly packed RGBA16F at the given size.");
    file.seekg(0); file.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(pixels.size() * sizeof(HALF)));
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = XMConvertFloatToHalf(i % 4 == 3 ? 1.f : Decode(XMConvertHalfToFloat(pixels[i])));
    const auto runtime = runtimePath.empty() ? ResolveRuntimePath({}, ExecutableDirectory()) : std::filesystem::absolute(runtimePath);
    Gpu gpu; Processor processor(gpu, runtime); Processor comparison(gpu, processor); ModelCompare composer(gpu);
    auto input = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON, true);
    auto desc = input->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    auto upload = gpu.Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD);
    ComPtr<ID3D12Resource> readback[3]{gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK), gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK), gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK)};
    void* mapped = nullptr; Check(upload->Map(0, nullptr, &mapped), "Fixture upload");
    for (UINT y = 0; y < h; ++y) memcpy(static_cast<BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
        pixels.data() + size_t(y) * w * 4, size_t(w) * 8);
    upload->Unmap(0, nullptr);
    gpu.Begin(); Barrier(gpu.commands.Get(), input.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = upload.Get(); from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
    to.pResource = input.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    Barrier(gpu.commands.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    gpu.Submit(); if (!gpu.Drain(30000)) throw std::runtime_error("Fixture upload timeout.");

    const auto read = [&](ID3D12Resource* buffer) {
        std::vector<HALF> values(size_t(w) * h * 4);
        void* data = nullptr; Check(buffer->Map(0, nullptr, &data), "Readback");
        for (UINT y = 0; y < h; ++y) memcpy(values.data() + size_t(y) * w * 4,
            static_cast<const BYTE*>(data) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, size_t(w) * 8);
        buffer->Unmap(0, nullptr);
        return values;
    };
    bool failed = false;
    for (unsigned mode : {2u, 3u}) for (bool swap : {false, true}) {
        Settings s; s.runtime = runtime; s.model.preset = model; s.inputColour = 1; s.estimateMotion = false;
        s.compare = mode; s.compareSplit = .3f; s.compareSwap = swap; s.compareModel = int(otherModel); s.compareLabels = false;
        if (!s.ComparingModels()) throw std::runtime_error("The settings do not ask for a model comparison.");
        const auto pictureSide = s.PictureSide(), otherSide = s.ComparisonSide();
        processor.Resize(w, h, pictureSide); comparison.Resize(w, h, otherSide);
        for (unsigned frame = 0; frame < frames; ++frame) {
            gpu.Begin();
            auto* picture = processor.Run(input.Get(), pictureSide, false, frame == 0, 1);
            auto* other = comparison.Run(input.Get(), otherSide, false, frame == 0, 1);
            auto* composed = composer.Compose(picture, other, s);
            if (frame + 1 == frames) {
                ID3D12Resource* sources[3]{picture, other, composed};
                for (unsigned i = 0; i < 3; ++i) {
                    from = {}; from.pResource = sources[i]; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    to = {}; to.pResource = readback[i].Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = footprint;
                    gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                }
            }
            gpu.Submit();
            if (!gpu.Drain(60000)) throw std::runtime_error("Compare-models check frame timeout.");
        }
        const auto picture = read(readback[0].Get()), other = read(readback[1].Get()), composed = read(readback[2].Get());
        const UINT border = mode == 3 ? w / 2 : UINT(std::lround(double(s.compareSplit) * w));
        const UINT start = mode == 3 ? (w - (w - border)) / 2 : 0;
        size_t wrong = 0, differing = 0;
        for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
            if (x + 1 == border || x == border) continue; // the border line
            const bool left = x < border;
            const auto& side = left != swap ? other : picture;
            const UINT sourceX = mode == 3 ? start + (left ? x : x - border) : x;
            const HALF* got = &composed[(size_t(y) * w + x) * 4];
            const HALF* want = &side[(size_t(y) * w + sourceX) * 4];
            if (got[0] != want[0] || got[1] != want[1] || got[2] != want[2]) ++wrong;
            const HALF* a = &picture[(size_t(y) * w + x) * 4]; const HALF* b = &other[(size_t(y) * w + x) * 4];
            if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2]) ++differing;
        }
        std::cout << std::format("{} {}: model {} against {}: {} wrong pixels; the two sides differ in {:.1f} % of the picture\n",
            mode == 3 ? "side by side" : "wipe", swap ? "swapped" : "as set", model, otherModel, wrong, 100.0 * differing / (double(w) * h));
        failed |= wrong != 0 || (model != otherModel && differing == 0);
    }
    if (!gpu.Drain(30000)) throw std::runtime_error("Compare-models check: final drain timeout.");
    ReleaseNvfp4(gpu.device.Get());
    if (failed) throw std::runtime_error("Compare-models check FAILED.");
    std::cout << "PASS model comparison: both sides are the models asked for, in wipe and side by side, swapped and not\n";
}
}
