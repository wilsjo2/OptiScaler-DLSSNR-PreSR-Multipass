// --stability-check <folder>: how much the output changes from one frame to the next where the picture does not, for
// Settings::editSmoothing ("Steady edit") and Settings::motionZeroBias ("Keep still areas still"), on a captured
// sequence (frame-0000.rgba16f, ... tightly packed RGBA16F linear scRGB at the source size) and on one of its frames
// shown again and again. Every value is in steps of 1/255 of the HDR display signal (PQ, SMPTE ST 2084; scRGB 1 =
// 80 nits), so a step is about as visible in a dark area as in a bright one.
//   still:  the same frame shown 30 times, history kept: the mean and 99.9th percentile of the change between the last
//           frames (any change is the processing's own flicker).
//   static: in the sequence, the pixels whose (grain-free) input changed by less than a quarter step: their output's
//           change, and the length of the motion vectors estimated there (0 is right).
//   moving: the pixels whose input changed by more than two steps: the mean difference from the first setting's output
//           (the original processing). Large values there would mean trails or lag.
// --pan P: instead of the sequence, a window of the output size slides P pixels a frame across the still frame at its
//   own resolution (camera pan). The ideal output moves with it exactly: "along motion" is the change between a pixel
//   and the same point of the picture in the frame before.
// --grain g: film grain on every input frame (the same random value for the three channels, g x the value).
// --stability-images <folder>: the last frame of every setting (and the input) as PNG.
#include "Processor.h"
#include <DirectXPackedVector.h>
#include <wincodec.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
namespace nr {
namespace {
using DirectX::PackedVector::HALF;
using DirectX::PackedVector::XMConvertFloatToHalf;
using DirectX::PackedVector::XMConvertHalfToFloat;
struct Encoded { // PQ signal x 255 for every half-float bit pattern
    std::vector<float> table = std::vector<float>(65536);
    Encoded() {
        for (unsigned bits = 0; bits < 65536; ++bits) {
            const float v = XMConvertHalfToFloat(HALF(bits));
            const double y = std::isfinite(v) ? std::clamp(double(v) * 80.0 / 10000.0, 0.0, 1.0) : 0.0;
            const double m1 = 2610.0 / 16384, m2 = 2523.0 / 4096 * 128, c1 = 3424.0 / 4096, c2 = 2413.0 / 4096 * 32, c3 = 2392.0 / 4096 * 32;
            const double p = std::pow(y, m1);
            table[bits] = float(255.0 * std::pow((c1 + c2 * p) / (1 + c3 * p), m2));
        }
    }
    float operator()(HALF h) const { return table[h]; }
};
std::vector<HALF> ReadRaw(const std::filesystem::path& file, UINT sw, UINT sh) {
    std::vector<HALF> source(size_t(sw) * sh * 4);
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in || size_t(in.tellg()) != source.size() * sizeof(HALF)) throw std::runtime_error("Frame " + file.string() + " is not RGBA16F at the source size.");
    in.seekg(0); in.read(reinterpret_cast<char*>(source.data()), source.size() * sizeof(HALF));
    for (size_t i = 3; i < source.size(); i += 4) source[i] = XMConvertFloatToHalf(1);
    return source;
}
std::vector<HALF> Resample(const std::vector<HALF>& source, UINT sw, UINT sh, UINT w, UINT h) {
    if (sw == w && sh == h) return source;
    // Area average (box) for whole-number factors, otherwise bilinear at pixel centres.
    std::vector<HALF> out(size_t(w) * h * 4);
    const bool box = sw % w == 0 && sh % h == 0;
    const UINT fx = sw / w, fy = sh / h;
    for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
        float sum[3]{};
        if (box) {
            for (UINT j = 0; j < fy; ++j) for (UINT i = 0; i < fx; ++i) {
                const size_t s = (size_t(y * fy + j) * sw + x * fx + i) * 4;
                for (int c = 0; c < 3; ++c) sum[c] += XMConvertHalfToFloat(source[s + c]);
            }
            for (float& v : sum) v /= float(fx * fy);
        } else {
            const float u = (x + .5f) * sw / w - .5f, v = (y + .5f) * sh / h - .5f;
            const int x0 = std::clamp(int(std::floor(u)), 0, int(sw) - 1), y0 = std::clamp(int(std::floor(v)), 0, int(sh) - 1);
            const int x1 = std::min(x0 + 1, int(sw) - 1), y1 = std::min(y0 + 1, int(sh) - 1);
            const float ax = std::clamp(u - x0, 0.f, 1.f), ay = std::clamp(v - y0, 0.f, 1.f);
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int px, int py) { return XMConvertHalfToFloat(source[(size_t(py) * sw + px) * 4 + c]); };
                sum[c] = (at(x0, y0) * (1 - ax) + at(x1, y0) * ax) * (1 - ay) + (at(x0, y1) * (1 - ax) + at(x1, y1) * ax) * ay;
            }
        }
        const size_t d = (size_t(y) * w + x) * 4;
        for (int c = 0; c < 3; ++c) out[d + c] = XMConvertFloatToHalf(sum[c]);
        out[d + 3] = XMConvertFloatToHalf(1);
    }
    return out;
}
std::vector<HALF> Crop(const std::vector<HALF>& source, UINT sw, UINT x0, UINT y0, UINT w, UINT h) {
    std::vector<HALF> out(size_t(w) * h * 4);
    for (UINT y = 0; y < h; ++y) memcpy(out.data() + size_t(y) * w * 4, source.data() + (size_t(y0 + y) * sw + x0) * 4, size_t(w) * 8);
    return out;
}
float Gaussian(uint32_t seed) { // deterministic N(0, 1)
    const auto hash = [](uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; };
    const float u1 = (hash(seed) + 1.f) / 4294967297.f, u2 = hash(seed ^ 0x9e3779b9U) / 4294967296.f;
    return std::sqrt(-2 * std::log(u1)) * std::cos(6.2831853f * u2);
}
std::vector<HALF> Grain(const std::vector<HALF>& clean, float amount, unsigned frame) {
    if (amount <= 0) return clean;
    auto out = clean;
    for (size_t p = 0; p < out.size() / 4; ++p) {
        const float n = 1 + amount * Gaussian(uint32_t(p * 2654435761U) ^ (frame * 0x51ED27U + 0x1234567U));
        for (int c = 0; c < 3; ++c) out[p * 4 + c] = XMConvertFloatToHalf(std::max(0.f, XMConvertHalfToFloat(clean[p * 4 + c]) * n));
    }
    return out;
}
void SaveDisplayPng(const std::filesystem::path& file, const std::vector<HALF>& values, UINT w, UINT h) {
    // scRGB to an SDR picture: the desktop's white (1.75) is white, a soft shoulder above it.
    std::vector<BYTE> bgra(size_t(w) * h * 4);
    for (size_t p = 0; p < size_t(w) * h; ++p) {
        for (int c = 0; c < 3; ++c) {
            float v = std::max(0.f, XMConvertHalfToFloat(values[p * 4 + c])) / 1.75f;
            v = v / (1 + v * .15f) * 1.15f;
            v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1 / 2.4f) - 0.055f;
            bgra[p * 4 + 2 - c] = BYTE(std::clamp(v * 255 + .5f, 0.f, 255.f));
        }
        bgra[p * 4 + 3] = 255;
    }
    ComPtr<IWICImagingFactory> factory; Check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "PNG factory");
    ComPtr<IWICStream> stream; Check(factory->CreateStream(&stream), "PNG stream");
    Check(stream->InitializeFromFilename(file.c_str(), GENERIC_WRITE), "PNG file");
    ComPtr<IWICBitmapEncoder> encoder; Check(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder), "PNG encoder");
    Check(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "PNG initialize");
    ComPtr<IWICBitmapFrameEncode> frame; Check(encoder->CreateNewFrame(&frame, nullptr), "PNG frame");
    Check(frame->Initialize(nullptr), "PNG frame init"); Check(frame->SetSize(w, h), "PNG size");
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA; Check(frame->SetPixelFormat(&format), "PNG format");
    Check(frame->WritePixels(h, w * 4, UINT(bgra.size()), bgra.data()), "PNG pixels");
    Check(frame->Commit(), "PNG frame commit"); Check(encoder->Commit(), "PNG commit");
}
struct Histogram { // steps of 1/64 of a display step, up to 64 steps
    std::vector<uint64_t> bins = std::vector<uint64_t>(64 * 64 + 1);
    double sum = 0; uint64_t count = 0;
    void Add(float v) { sum += v; ++count; ++bins[std::min<size_t>(bins.size() - 1, size_t(v * 64))]; }
    double Mean() const { return count ? sum / count : 0; }
    double Quantile(double q) const {
        if (!count) return 0;
        const uint64_t want = uint64_t(std::ceil(q * count)); uint64_t seen = 0;
        for (size_t i = 0; i < bins.size(); ++i) if ((seen += bins[i]) >= want) return (i + 1) / 64.0;
        return 64;
    }
};
}
void StabilityCheck(Settings settings, const std::filesystem::path& folder, UINT sw, UINT sh, UINT w, UINT h, unsigned frames,
                    unsigned still, const std::vector<std::pair<float, float>>& variants, const std::filesystem::path& csv) {
    const auto number = [](const wchar_t* name, double fallback) { const wchar_t* v = _wgetenv(name); return v && *v ? _wtof(v) : fallback; };
    // (these three come from the command line through the environment: Check.cpp sets them)
    const UINT pan = UINT(number(L"NR_STABILITY_PAN", 0));
    const float grain = float(number(L"NR_STABILITY_GRAIN", 0));
    const wchar_t* imagesValue = _wgetenv(L"NR_STABILITY_IMAGES");
    const std::filesystem::path images = imagesValue && *imagesValue ? imagesValue : L"";
    if (!images.empty()) std::filesystem::create_directories(images);
    InstallPassBorrow();
    Gpu gpu; Processor processor(gpu, settings.runtime); processor.Resize(w, h, settings);
    const Encoded pq;
    // clean: what the mask and the "along motion" figure are made from; the processing gets it with grain.
    std::vector<std::vector<HALF>> clean;
    if (pan) {
        if (sw < w + pan * (frames - 1) || sh < h) throw std::runtime_error("The pan does not fit in the source frame.");
        const auto source = ReadRaw(folder / std::format("frame-{:04}.rgba16f", still), sw, sh);
        for (unsigned t = 0; t < frames; ++t) clean.push_back(Crop(source, sw, t * pan, (sh - h) / 2, w, h));
        std::cout << std::format("Pan: frame {} of {} at its own resolution, a {}x{} window moving {} px a frame\n", still, folder.string(), w, h, pan);
    } else {
        std::cout << std::format("Loading {} frames of {} ({}x{} -> {}x{})\n", frames, folder.string(), sw, sh, w, h) << std::flush;
        for (unsigned i = 0; i < frames; ++i) clean.push_back(Resample(ReadRaw(folder / std::format("frame-{:04}.rgba16f", i), sw, sh), sw, sh, w, h));
        if (still >= frames) throw std::runtime_error("The still frame is not in the sequence.");
    }
    if (grain > 0) std::cout << std::format("Film grain on the input: {:.3f} of the value\n", grain);
    auto input = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON, true);
    auto desc = input->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    auto upload = gpu.Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD), readback = gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    ComPtr<ID3D12Resource> vectorReadback; D3D12_PLACED_SUBRESOURCE_FOOTPRINT vectorFootprint{}; UINT vectorW = 0, vectorH = 0;
    std::vector<float> vectorLength; // per model pixel, the last frame's estimated motion in pixels; empty without
    const auto run = [&](const std::vector<HALF>& pixels, const Settings& s, bool reset, double& nrMs) {
        void* mapped = nullptr; Check(upload->Map(0, nullptr, &mapped), "Upload map");
        for (UINT y = 0; y < h; ++y) memcpy(static_cast<BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
            pixels.data() + size_t(y) * w * 4, size_t(w) * 8);
        upload->Unmap(0, nullptr);
        processor.Resize(w, h, s);
        gpu.Begin(); gpu.BeginTimings();
        auto* c = gpu.commands.Get();
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        Barrier(c, input.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        from.pResource = upload.Get(); from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
        to.pResource = input.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        c->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        Barrier(c, input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        auto* result = processor.Run(input.Get(), s, true, reset, 1.75f, 16.67f);
        c = gpu.commands.Get();
        from = {}; from.pResource = result; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to = {}; to.pResource = readback.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = footprint;
        c->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        ID3D12Resource* vectors = processor.SceneMotion();
        if (vectors) {
            const auto vd = vectors->GetDesc();
            if (!vectorReadback || vd.Width != vectorW || vd.Height != vectorH) {
                UINT64 vectorBytes = 0; vectorW = UINT(vd.Width); vectorH = vd.Height;
                gpu.device->GetCopyableFootprints(&vd, 0, 1, 0, &vectorFootprint, nullptr, nullptr, &vectorBytes);
                vectorReadback = gpu.Buffer(vectorBytes, D3D12_HEAP_TYPE_READBACK);
            }
            Barrier(c, vectors, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            from = {}; from.pResource = vectors; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            to = {}; to.pResource = vectorReadback.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = vectorFootprint;
            c->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            Barrier(c, vectors, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        gpu.ResolveTimings(); gpu.Submit();
        if (!gpu.Drain(30000)) throw std::runtime_error("Stability check GPU timeout.");
        std::array<double, unsigned(GpuStage::Count)> stages{};
        nrMs = gpu.TakeTimings(stages) ? stages[unsigned(GpuStage::NR)] : 0;
        std::vector<HALF> values(size_t(w) * h * 4); Check(readback->Map(0, nullptr, &mapped), "Readback map");
        for (UINT y = 0; y < h; ++y) memcpy(values.data() + size_t(y) * w * 4,
            static_cast<BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch, size_t(w) * 8);
        readback->Unmap(0, nullptr);
        vectorLength.clear();
        if (vectors) {
            Check(vectorReadback->Map(0, nullptr, &mapped), "Vector readback map");
            vectorLength.resize(size_t(vectorW) * vectorH);
            for (UINT y = 0; y < vectorH; ++y) {
                const auto* row = reinterpret_cast<const HALF*>(static_cast<const BYTE*>(mapped) + vectorFootprint.Offset + size_t(y) * vectorFootprint.Footprint.RowPitch);
                for (UINT x = 0; x < vectorW; ++x) vectorLength[size_t(y) * vectorW + x] = std::hypot(XMConvertHalfToFloat(row[x * 2]), XMConvertHalfToFloat(row[x * 2 + 1]));
            }
            vectorReadback->Unmap(0, nullptr);
        }
        return values;
    };
    // The first setting's outputs (the reference for "moving"), every other pixel per dimension above 2560 wide.
    const UINT stride = w > 2560 ? 2 : 1;
    std::vector<std::vector<uint16_t>> reference;
    const auto sampleIndex = [&](UINT x, UINT y) { return (size_t(y / stride) * ((w + stride - 1) / stride) + x / stride) * 3; };
    std::ofstream report;
    if (!csv.empty()) {
        report.open(csv);
        report << "steady,zeroBias,pan,grain,stillMean,stillP999,staticMean,staticP999,staticShare,inputStaticMean,staticVectorPx,staticVectorShare,"
                  "alongMean,alongP999,movingDiff,movingP99,movingShare,allDiff,nrMs,steadyFrames\n";
    }
    std::cout << "Steps of 1/255 of the PQ display signal; frames 3.. of the sequence.\n";
    if (pan) std::cout << "steady zero |  still mean  p99.9 | along motion mean  p99.9 (input) | vs first: moving  p99 | NR ms\n";
    else std::cout << "steady zero |  still mean  p99.9 | static mean  p99.9  (share, input) vectors px (moving) | vs first: moving  p99 (share) all | NR ms\n";
    for (size_t v = 0; v < variants.size(); ++v) {
        auto s = settings; s.editSmoothing = variants[v].first; s.motionZeroBias = variants[v].second;
        double nrMs = 0; std::vector<double> nrTimes;
        // Still: the frame shown 30 times; the change over the last 6 pairs. (grain changes every time: then it is not still)
        Histogram stillChange; std::vector<HALF> previous;
        const auto& stillFrame = pan ? clean.front() : clean[still];
        for (unsigned n = 0; n < 30 && grain <= 0; ++n) {
            auto out = run(stillFrame, s, n == 0, nrMs);
            if (n >= 24) for (size_t i = 0; i < out.size(); ++i) if (i % 4 != 3) stillChange.Add(std::abs(pq(out[i]) - pq(previous[i])));
            previous = std::move(out);
        }
        Histogram staticChange, inputStatic, moving, all, along, inputAlong, staticVector;
        uint64_t staticPixels = 0, movingPixels = 0, counted = 0, staticMoved = 0; unsigned steadyFrames = 0;
        previous.clear(); std::vector<HALF> lastInput;
        for (unsigned t = 0; t < frames; ++t) {
            const auto fed = Grain(clean[t], grain, t);
            auto out = run(fed, s, t == 0, nrMs);
            if (t >= 3) nrTimes.push_back(nrMs);
            steadyFrames += processor.steadyApplied ? 1 : 0;
            if (v == 0) {
                std::vector<uint16_t> kept(size_t((w + stride - 1) / stride) * ((h + stride - 1) / stride) * 3);
                for (UINT y = 0; y < h; y += stride) for (UINT x = 0; x < w; x += stride)
                    for (int c = 0; c < 3; ++c) kept[sampleIndex(x, y) + c] = uint16_t(std::lround(pq(out[(size_t(y) * w + x) * 4 + c]) * 256));
                reference.push_back(std::move(kept));
            }
            if (t >= 3) {
                const auto& in = clean[t]; const auto& before = clean[t - 1];
                for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) {
                    const size_t i = (size_t(y) * w + x) * 4;
                    float inputChange = 0;
                    if (pan) {
                        // The picture moved left by `pan`: this pixel was at x + pan in the frame before.
                        if (x + pan < w) {
                            const size_t j = (size_t(y) * w + x + pan) * 4;
                            for (int c = 0; c < 3; ++c) { along.Add(std::abs(pq(out[i + c]) - pq(previous[j + c]))); inputAlong.Add(std::abs(pq(in[i + c]) - pq(before[j + c]))); }
                        }
                        inputChange = 1e9f; // every pixel moves
                    } else {
                        for (int c = 0; c < 3; ++c) inputChange = std::max(inputChange, std::abs(pq(in[i + c]) - pq(before[i + c])));
                        ++counted;
                        if (inputChange < .25f) {
                            ++staticPixels;
                            for (int c = 0; c < 3; ++c) { staticChange.Add(std::abs(pq(out[i + c]) - pq(previous[i + c]))); inputStatic.Add(std::abs(pq(fed[i + c]) - pq(lastInput[i + c]))); }
                            if (!vectorLength.empty()) {
                                const float length = vectorLength[size_t(y * vectorH / h) * vectorW + x * vectorW / w];
                                staticVector.Add(length); staticMoved += length > .01f;
                            }
                        }
                    }
                    if (x % stride || y % stride) continue;
                    const auto& base = reference[t];
                    float difference = 0;
                    for (int c = 0; c < 3; ++c) difference += std::abs(pq(out[i + c]) - base[sampleIndex(x, y) + c] / 256.f);
                    difference /= 3; all.Add(difference);
                    if (inputChange > 2) { ++movingPixels; moving.Add(difference); }
                }
            }
            if (t + 1 == frames && !images.empty()) {
                if (v == 0) SaveDisplayPng(images / L"input.png", fed, w, h);
                SaveDisplayPng(images / std::format(L"steady{:.2f}-zero{:.2f}.png", s.editSmoothing, s.motionZeroBias), out, w, h);
            }
            previous = std::move(out); lastInput = fed;
        }
        std::sort(nrTimes.begin(), nrTimes.end());
        const double nrMedian = nrTimes.empty() ? 0 : nrTimes[nrTimes.size() / 2];
        const double staticShare = counted ? double(staticPixels) / counted : 0;
        const double movingShare = all.count ? double(movingPixels) / all.count : 0;
        const double vectorShare = staticVector.count ? double(staticMoved) / staticVector.count : 0;
        const std::string steadyNote = s.editSmoothing > 0 ? std::format("  (steady edit on {} of {} frames)", steadyFrames, frames) : std::string();
        if (pan) std::cout << std::format("{:6.2f} {:4.2f} | {:6.3f} {:6.2f} | {:6.3f} {:6.2f} ({:5.3f}) | {:6.3f} {:6.2f} | {:5.2f}{}\n",
            s.editSmoothing, s.motionZeroBias, stillChange.Mean(), stillChange.Quantile(.999), along.Mean(), along.Quantile(.999), inputAlong.Mean(),
            moving.Mean(), moving.Quantile(.99), nrMedian, steadyNote) << std::flush;
        else std::cout << std::format("{:6.2f} {:4.2f} | {:6.3f} {:6.2f} | {:6.3f} {:6.2f} ({:4.1f} %, {:5.3f}) {:5.3f} ({:4.1f} %) | {:6.3f} {:6.2f} ({:4.1f} %) {:6.3f} | {:5.2f}{}\n",
            s.editSmoothing, s.motionZeroBias, stillChange.Mean(), stillChange.Quantile(.999), staticChange.Mean(), staticChange.Quantile(.999),
            100 * staticShare, inputStatic.Mean(), staticVector.Mean(), 100 * vectorShare, moving.Mean(), moving.Quantile(.99), 100 * movingShare, all.Mean(),
            nrMedian, steadyNote) << std::flush;
        if (report) report << std::format("{},{},{},{},{:.4f},{:.3f},{:.4f},{:.3f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.3f},{:.4f},{:.3f},{:.4f},{:.4f},{:.3f},{}\n",
            s.editSmoothing, s.motionZeroBias, pan, grain, stillChange.Mean(), stillChange.Quantile(.999), staticChange.Mean(), staticChange.Quantile(.999),
            staticShare, inputStatic.Mean(), staticVector.Mean(), vectorShare, along.Mean(), along.Quantile(.999), moving.Mean(), moving.Quantile(.99),
            movingShare, all.Mean(), nrMedian, steadyFrames);
        if (s.editSmoothing > 0 && s.estimateMotion && !steadyFrames) throw std::runtime_error("Steady edit was asked for and never applied.");
        if (s.editSmoothing <= 0 && steadyFrames) throw std::runtime_error("Steady edit ran while switched off.");
    }
}
}
