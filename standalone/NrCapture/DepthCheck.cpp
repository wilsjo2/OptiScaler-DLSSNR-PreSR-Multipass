#include "DepthCheck.h"
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace nr {
namespace {
using DirectX::PackedVector::XMConvertFloatToHalf;
using DirectX::PackedVector::XMConvertHalfToFloat;
using Pixels = std::vector<uint16_t>;
constexpr const char* Patterns[]{"flat-far", "flat-mid", "horizontal", "vertical", "radial",
    "shapes", "shapes-reversed", "checkerboard", "moving-step", "noise", "image-luma"};
constexpr const char* Modes[]{"reset-every-frame", "history-zero-motion", "history-known-motion"};

// Test-only transfers. Call Upload/Download inside a list; fence before mapping
// downloads or reusing uploads. All readbacks preserve the texture's input state.
struct Transfer {
    Gpu& gpu;
    ID3D12Resource* texture;
    D3D12_RESOURCE_STATES state;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    ComPtr<ID3D12Resource> upload, readback;
    size_t rowBytes;
    UINT height;
    Transfer(Gpu& g, ID3D12Resource* t, D3D12_RESOURCE_STATES s, size_t texelBytes)
        : gpu(g), texture(t), state(s) {
        const auto desc = t->GetDesc(); UINT64 bytes = 0;
        height = desc.Height; rowBytes = size_t(desc.Width) * texelBytes;
        gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &bytes);
        upload = gpu.Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD);
        readback = gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    }
    void Copy(ID3D12Resource* buffer, bool toTexture) {
        D3D12_TEXTURE_COPY_LOCATION tex{}, buf{};
        tex.pResource = texture; tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        buf.pResource = buffer; buf.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; buf.PlacedFootprint = layout;
        const auto copyState = toTexture ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        Barrier(gpu.commands.Get(), texture, state, copyState);
        gpu.commands->CopyTextureRegion(toTexture ? &tex : &buf, 0, 0, 0, toTexture ? &buf : &tex, nullptr);
        Barrier(gpu.commands.Get(), texture, copyState, state);
    }
    template<class T> void Upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != rowBytes * height) throw std::runtime_error("Depth probe upload size mismatch.");
        void* mapped = nullptr; Check(upload->Map(0, nullptr, &mapped), "Depth probe upload map");
        for (UINT y = 0; y < height; ++y)
            memcpy(static_cast<BYTE*>(mapped) + layout.Offset + size_t(y) * layout.Footprint.RowPitch,
                reinterpret_cast<const BYTE*>(values.data()) + size_t(y) * rowBytes, rowBytes);
        upload->Unmap(0, nullptr); Copy(upload.Get(), true);
    }
    void Download() { Copy(readback.Get(), false); }
    template<class T> std::vector<T> Read() {
        std::vector<T> values(rowBytes * height / sizeof(T));
        void* mapped = nullptr; Check(readback->Map(0, nullptr, &mapped), "Depth probe readback map");
        for (UINT y = 0; y < height; ++y)
            memcpy(reinterpret_cast<BYTE*>(values.data()) + size_t(y) * rowBytes,
                static_cast<const BYTE*>(mapped) + layout.Offset + size_t(y) * layout.Footprint.RowPitch, rowBytes);
        readback->Unmap(0, nullptr); return values;
    }
};
void Complete(Gpu& gpu) {
    gpu.Submit();
    if (!gpu.Drain(30000)) throw std::runtime_error("Depth probe GPU timeout.");
    ComPtr<ID3D12InfoQueue> diagnostics;
    if (SUCCEEDED(gpu.device.As(&diagnostics))) {
        bool failed = false;
        for (UINT64 i = 0; i < diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
            SIZE_T size = 0; diagnostics->GetMessage(i, nullptr, &size); std::vector<BYTE> data(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(data.data());
            Check(diagnostics->GetMessage(i, message, &size), "Depth probe GPU diagnostic");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::cerr << message->pDescription << '\n'; failed = true;
            }
        }
        diagnostics->ClearStoredMessages();
        if (failed) throw std::runtime_error("D3D12 validation failed during depth probe.");
    }
}
struct Difference {
    double mean = 0;
    float maximum = 0;
    size_t changedPixels = 0;
};
Difference Compare(const Pixels& a, const Pixels& b) {
    if (a.size() != b.size() || a.empty()) throw std::runtime_error("Depth probe comparison size mismatch.");
    Difference d;
    for (size_t i = 0; i < a.size(); i += 4) {
        bool changed = false;
        for (size_t c = 0; c < 4; ++c) {
            const float av = XMConvertHalfToFloat(a[i + c]), bv = XMConvertHalfToFloat(b[i + c]);
            if (!std::isfinite(av) || !std::isfinite(bv)) throw std::runtime_error("Nonfinite depth probe output.");
            if (c == 3) continue;
            const float delta = std::abs(av - bv);
            d.mean += delta; d.maximum = std::max(d.maximum, delta); changed |= delta != 0;
        }
        d.changedPixels += changed;
    }
    d.mean /= double(a.size() / 4 * 3); return d;
}
int Object(int x, int y, unsigned frame, UINT w, UINT h) {
    const float cx = float(w) * 0.28f + 4 * frame, cy = float(h) * 0.62f;
    if (std::hypot(float(x) - cx, float(y) - cy) < float(h) * 0.18f) return 2;
    const int rx = x + 2 * int(frame), ry = y - int(frame);
    return rx >= int(w * 0.64f) && rx < int(w * 0.88f) && ry >= int(h * 0.18f) && ry < int(h * 0.48f) ? 1 : 0;
}
struct Frame { Pixels colour, motion; std::vector<int> objects; };
Frame Alter(const Pixels& base, UINT w, UINT h, unsigned frame) {
    Frame f{Pixels(base.size()), Pixels(size_t(w) * h * 2), std::vector<int>(size_t(w) * h)};
    const float brightness = 1 + 0.06f * std::sin(float(frame) * 0.45f);
    for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const int object = Object(int(x), int(y), frame, w, h); f.objects[i] = object;
        const int dx = object == 2 ? 4 : object == 1 ? -2 : 2, dy = object == 2 ? 0 : 1;
        // Pan the original and move two textured, tinted occluders independently.
        const int sx = std::clamp(int(x) - dx * int(frame) + (object == 2 ? int(w / 6) : 0), 0, int(w) - 1);
        const int sy = std::clamp(int(y) - dy * int(frame), 0, int(h) - 1);
        for (size_t c = 0; c < 3; ++c) {
            float value = XMConvertHalfToFloat(base[(size_t(sy) * w + sx) * 4 + c]);
            if (object) value = value * 0.65f + ((object == 2 && c == 0) || (object == 1 && c == 2) ? 0.25f : 0.04f);
            f.colour[i * 4 + c] = XMConvertFloatToHalf(std::clamp(value * brightness, 0.0f, 1.0f));
        }
        f.colour[i * 4 + 3] = XMConvertFloatToHalf(1);
        const int px = int(x) - dx, py = int(y) - dy;
        const bool valid = frame && px >= 0 && py >= 0 && px < int(w) && py < int(h) &&
            Object(px, py, frame - 1, w, h) == object;
        // Current-to-previous pixel units. Newly exposed pixels have no match.
        f.motion[i * 2] = XMConvertFloatToHalf(valid ? -float(dx) : 0);
        f.motion[i * 2 + 1] = XMConvertFloatToHalf(valid ? -float(dy) : 0);
    }
    return f;
}
std::vector<float> Depth(unsigned pattern, const Frame& f, UINT w, UINT h, unsigned frame, unsigned count) {
    std::vector<float> values(size_t(w) * h);
    for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const float u = float(x) / float(w - 1), v = float(y) / float(h - 1);
        float z = 1;
        switch (pattern) {
        case 1: z = 0.5f; break;
        case 2: z = 0.05f + 0.9f * u; break;
        case 3: z = 0.05f + 0.9f * v; break;
        case 4: z = 0.05f + 0.9f * std::min(1.0f, 2 * std::hypot(u - 0.5f, v - 0.5f)); break;
        case 5: case 6:
            z = f.objects[i] == 2 ? 0.08f : f.objects[i] == 1 ? 0.35f : 0.85f + 0.1f * v;
            if (pattern == 6) z = 1 - z;
            break;
        case 7: z = ((x / 16 + y / 16) & 1) ? 0.95f : 0.05f; break;
        case 8: z = u < 0.15f + 0.7f * float(frame) / float(count - 1) ? 0.05f : 0.95f; break;
        case 9: {
            uint32_t hash = x * 0x9e3779b9u ^ y * 0x85ebca6bu;
            hash ^= hash >> 16; hash *= 0x7feb352du; hash ^= hash >> 15;
            z = 0.05f + 0.9f * float(hash & 65535) / 65535; break;
        }
        case 10:
            z = 0.05f + 0.9f * (0.2126f * XMConvertHalfToFloat(f.colour[i * 4]) +
                0.7152f * XMConvertHalfToFloat(f.colour[i * 4 + 1]) + 0.0722f * XMConvertHalfToFloat(f.colour[i * 4 + 2]));
            break;
        }
        values[i] = z;
    }
    return values;
}
// Existing PNG writer expects linear RGB; model output here is encoded SDR.
Pixels Preview(const Pixels& encoded) {
    auto linear = encoded;
    for (size_t i = 0; i < linear.size(); ++i) {
        float v = XMConvertHalfToFloat(encoded[i]);
        v = i % 4 == 3 ? 1 : v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
        linear[i] = XMConvertFloatToHalf(v);
    }
    return linear;
}
}

struct DepthProbe {
    static void Run(Gpu& gpu, Processor& p, ID3D12Resource* input, Settings s, unsigned frames,
                    const std::filesystem::path& directory, DepthPreviewWriter save) {
        if (frames < 4 || frames > 120) throw std::runtime_error("Depth probe requires 4-120 frames.");
        if (s.inputColour == 2) throw std::runtime_error("Depth probe uses an SDR fixture; use --input-colour 1.");
        // Isolate raw, single-pass NR. Composition, private SR, FG and estimated
        // motion must not confound the comparison.
        s = Settings{.runtime = s.runtime, .model = s.model};
        s.inputColour = 1; p.Resize(p.width, p.height, s);
        const UINT w = p.modelWidth, h = p.modelHeight;
        if (w < 64 || h < 64) throw std::runtime_error("Depth probe requires at least 64x64 pixels.");
        std::filesystem::create_directories(directory);
        std::ofstream csv(directory / "frames.csv"), summary(directory / "summary.csv"), metadata(directory / "metadata.txt");
        if (!csv || !summary || !metadata) throw std::runtime_error("Cannot create depth probe reports.");
        csv << std::setprecision(12) << "mode,pattern,frame,reset,depth_min,depth_max,depth_changed_pixels,mean_abs_rgb,max_abs_rgb,changed_rgb_pixels,model_edit_mean\n";
        summary << std::setprecision(12) << "mode,pattern,mean_abs_rgb,max_abs_rgb,changed_frames,repeat_max_abs_rgb,above_repeat_floor\n";
        metadata << "Requested runtime: " << s.runtime.string() << "\nModel size: " << w << 'x' << h << "\nFrames: " << frames
            << "\nPreset/style/intensity/structure/tone/skin/auto-mask: " << s.model.preset << '/' << s.model.style << '/'
            << s.model.intensity << '/' << s.model.localStructure << '/' << s.model.localTone << '/' << s.model.skinStructure << '/' << s.model.autoMask
            << "\nSingle-pass raw RGBA16F NR; standalone SDR input encoding; R32_FLOAT depth in [0,1]; DepthInverted=0."
            << "\nSame pan, independently moving tinted circle/rectangle and brightness pulse in every trial."
            << "\nAnalytic motion is current-to-previous pixels; disocclusions use zero. Synthetic geometry is not engine depth."
            << "\nFresh feature per trial; history modes reset only frame 0. Flat replay measures repeatability."
            << "\nAll uploaded colour/depth/motion textures are GPU-readback verified each frame."
            << "\nPNG previews show final-frame inputs/results and absolute RGB differences amplified 32x."
            << "\nMetrics compare decoded FP16 RGB before PNG conversion; zero depth effect is a valid result.\n";
        gpu.Begin(); p.Run(input, s, false, true, 1); Complete(gpu);
        Transfer seed(gpu, p.ModelInput(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 8);
        gpu.Begin(); seed.Download(); Complete(gpu);
        const auto base = seed.Read<uint16_t>();
        save(directory / "source.png", Preview(base), w, h);
        auto colour = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto depth = gpu.Texture(w, h, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto motion = gpu.Texture(w, h, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto answer = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        Transfer colours(gpu, colour.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 8),
            depths(gpu, depth.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 4),
            motions(gpu, motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 4),
            answers(gpu, answer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 8);
        std::vector<Pixels> resetBaseline, zeroMotionBaseline;
        for (unsigned mode = 0; mode < std::size(Modes); ++mode) {
            const auto folder = directory / Modes[mode]; std::filesystem::create_directories(folder);
            std::vector<Pixels> baseline;
            double repeatMax = 0;
            // Baseline, identical baseline replay, then each alternative.
            for (unsigned trial = 0; trial <= std::size(Patterns); ++trial) {
                const unsigned pattern = trial < 2 ? 0 : trial - 1;
                const std::string name = trial == 1 ? "flat-repeat" : Patterns[pattern];
                gpu.Begin(); const auto created = p.runtime->Create(gpu.commands.Get(), w, h, s.model);
                Complete(gpu); Runtime::CheckResult(created, "Depth probe feature creation");
                double totalMean = 0, maxDifference = 0, modelEdit = 0, historyDifference = 0, motionDifference = 0;
                unsigned changedFrames = 0;
                Pixels previousInput, previousOutput;
                for (unsigned frame = 0; frame < frames; ++frame) {
                    auto altered = Alter(base, w, h, frame);
                    if (mode != 2) std::fill(altered.motion.begin(), altered.motion.end(), uint16_t{0});
                    const auto guide = Depth(pattern, altered, w, h, frame, frames);
                    const auto [low, high] = std::minmax_element(guide.begin(), guide.end());
                    if (!std::isfinite(*low) || !std::isfinite(*high) || *low < 0 || *high > 1 ||
                        (pattern >= 2 && *low == *high)) throw std::runtime_error("Invalid/non-varying depth pattern.");
                    const bool reset = mode == 0 || frame == 0;
                    gpu.Begin(); colours.Upload(altered.colour); depths.Upload(guide); motions.Upload(altered.motion);
                    p.runtime->Evaluate(gpu.commands.Get(), colour.Get(), depth.Get(), motion.Get(), answer.Get(), w, h, reset);
                    colours.Download(); depths.Download(); motions.Download(); answers.Download(); Complete(gpu);
                    if (colours.Read<uint16_t>() != altered.colour || depths.Read<float>() != guide || motions.Read<uint16_t>() != altered.motion)
                        throw std::runtime_error("Depth probe GPU input readback mismatch.");
                    const auto output = answers.Read<uint16_t>();
                    if (!trial) baseline.push_back(output);
                    const auto difference = Compare(output, baseline.at(frame));
                    const auto edit = Compare(output, altered.colour); modelEdit += edit.mean;
                    if (frame && Compare(altered.colour, previousInput).maximum == 0)
                        throw std::runtime_error("Depth probe fixture did not change between frames.");
                    if (frame && !trial && Compare(output, previousOutput).maximum == 0)
                        throw std::runtime_error("NR did not respond to the altered image.");
                    previousInput = altered.colour; previousOutput = output;
                    if (mode && !trial) historyDifference = std::max(historyDifference, double(Compare(output, resetBaseline.at(frame)).maximum));
                    if (mode == 2 && !trial) motionDifference = std::max(motionDifference, double(Compare(output, zeroMotionBaseline.at(frame)).maximum));
                    totalMean += difference.mean; maxDifference = std::max(maxDifference, double(difference.maximum));
                    changedFrames += difference.changedPixels != 0;
                    const auto changedDepth = std::count_if(guide.begin(), guide.end(), [](float v) { return v != 1; });
                    csv << Modes[mode] << ',' << name << ',' << frame << ',' << reset << ',' << *low << ',' << *high << ',' << changedDepth << ','
                        << difference.mean << ',' << difference.maximum << ',' << difference.changedPixels << ',' << edit.mean << '\n';
                    if (mode == 0 && !trial && (frame == 0 || frame == frames / 2 || frame == frames - 1))
                        save(directory / std::format("input-{:03}.png", frame), Preview(altered.colour), w, h);
                    if (frame == frames - 1) {
                        save(folder / (name + ".png"), Preview(output), w, h);
                        if (trial > 1) {
                            Pixels diff(output.size()), preview(output.size());
                            for (size_t i = 0; i < output.size(); ++i) {
                                diff[i] = XMConvertFloatToHalf(i % 4 == 3 ? 1.0f :
                                    std::min(1.0f, 32 * std::abs(XMConvertHalfToFloat(output[i]) - XMConvertHalfToFloat(baseline[frame][i]))));
                                preview[i] = XMConvertFloatToHalf(i % 4 == 3 ? 1.0f : guide[i / 4]);
                            }
                            save(folder / (name + "-diff-x32.png"), Preview(diff), w, h);
                            if (mode == 0) save(directory / (name + "-depth.png"), Preview(preview), w, h);
                        }
                    }
                }
                if (modelEdit / frames < 0.000001) throw std::runtime_error("NR was effectively inactive on the fixture.");
                if (trial == 1) repeatMax = maxDifference;
                summary << Modes[mode] << ',' << name << ',' << totalMean / frames << ',' << maxDifference << ','
                    << changedFrames << ',' << repeatMax << ',' << (maxDifference > repeatMax) << '\n';
                std::cout << std::format("{} / {}: mean RGB {:.9g}, max {:.9g}, changed frames {}/{}, repeat max {:.9g}\n",
                    Modes[mode], name, totalMean / frames, maxDifference, changedFrames, frames, repeatMax) << std::flush;
                if (mode && !trial) std::cout << std::format("  History control vs reset-every-frame: max RGB {:.9g}\n", historyDifference);
                if (!mode && !trial) resetBaseline = baseline;
                if (mode == 1 && !trial) zeroMotionBaseline = baseline;
                if (!trial) metadata << Modes[mode] << " controls: mean model edit=" << modelEdit / frames
                    << ", max difference vs reset=" << historyDifference << ", max difference vs zero motion=" << motionDifference << '\n';
                if (mode == 2 && !trial) std::cout << std::format("  Motion control vs history-zero-motion: max RGB {:.9g}\n", motionDifference);
                csv.flush(); summary.flush();
            }
        }
        wchar_t loaded[32768]{}; const auto library = GetModuleHandleW(L"nvngx_dlssnr.dll");
        if (library && GetModuleFileNameW(library, loaded, DWORD(std::size(loaded))))
            metadata << "Loaded runtime: " << Narrow(loaded) << '\n';
        if (!csv || !summary || !metadata) throw std::runtime_error("Failed writing depth probe reports.");
        std::cout << "PASS depth probe: active NR, altered image, verified GPU inputs, finite output, completed comparisons.\n"
            << "Reports: " << directory.string() << '\n';
    }
};
void DepthCheck(Gpu& gpu, Processor& processor, ID3D12Resource* input, Settings settings, unsigned frames,
                const std::filesystem::path& directory, DepthPreviewWriter savePreview) {
    DepthProbe::Run(gpu, processor, input, settings, frames, directory, savePreview);
}
}
