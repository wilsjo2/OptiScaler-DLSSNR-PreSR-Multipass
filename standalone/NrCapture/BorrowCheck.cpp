#include "Processor.h"
#include <DirectXPackedVector.h>
#include <d3d12sdklayers.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
// GPU harness for manual use (needs the NVIDIA GPU and NR runtime; no automated check runs it):
//   NrCaptureCheck --borrow-check <fixture.rgba16f> <width> <height> <out.json> [--preset 0-3] [--runtime dll] [--frames N] [--nvfp4 1] [--nvfp4swin widths] [--nvfp4net "items"]
// One fixture frame through the real Processor path, history reset every frame, for one/two/three-pass variants with
// re-apply and borrowing. Writes RGB PSNR of each against its all-real reference, NrPassBorrow outcomes per network
// evaluation, and the median NR GPU stage time. --nvfp4 1 runs the NVFP4 feed-forward option instead: one, two and
// three passes and a transformer borrow, each against the same passes with the runtime's own kernels.
// --nvfp4swin <1 = 32, 2 = 64, 4 = 128, 8 = 256, summed> does the same for the NVFP4 fused blocks, with a below-level
// borrow added (a pass that stops part-way through the fused blocks). --nvfp4net "<NrNet4::Parse items>" adds the
// NVFP4 kernels of the rest of the network (for example "split=all vit=all prepost=3") to the same set of variants.
namespace nr {
namespace {
using DirectX::PackedVector::HALF;
using DirectX::PackedVector::XMConvertFloatToHalf;
using DirectX::PackedVector::XMConvertHalfToFloat;
float Decode(float v) { v = std::max(v, 0.f); return v <= .04045f ? v / 12.92f : std::pow((v + .055f) / 1.055f, 2.4f); }
float Encode(float v) { v = std::max(v, 0.f); return v <= .0031308f ? 12.92f * v : 1.055f * std::pow(v, 1 / 2.4f) - .055f; }
std::string Quote(const std::string& text) {
    std::string result = "\"";
    for (char c : text) { if (c == '\\' || c == '"') result += '\\'; result += c; }
    return result + '"';
}
struct Variant {
    std::string name, reference;
    unsigned passes = 1;
    std::array<PassShortcut, Settings::MaxPasses - 1> shortcuts{};
    float gain = .81f;
    bool nvfp4 = false;
    unsigned swin = 0;
    std::string net;
};
std::string Describe(const Variant& v) {
    std::string text = "real";
    for (unsigned pass = 1; pass < v.passes; ++pass) {
        const auto& s = v.shortcuts[pass - 1];
        text += s.reapply ? ",reapply" : std::string(",") +
            (s.borrowBelow || s.borrowTransformer ? "borrow " + std::string(NrPassBorrow::CutName(NrPassBorrow::ChooseCut(s.borrowTransformer, s.borrowBelow, s.borrowLevel))) : "real");
    }
    return text;
}
}
void BorrowCheck(const std::filesystem::path& fixture, UINT w, UINT h, const std::filesystem::path& report,
    unsigned preset, const std::filesystem::path& runtimePath, unsigned frames, bool sameTuning, unsigned style, bool nvfp4, unsigned swin, const std::string& net) {
    if (!w || !h || w > 7680 || h > 4320 || preset > Settings::ModelANvfp4 || style > 2 || frames < 2) throw std::runtime_error("Borrow check: invalid extent, preset or frame count.");
    if (!InstallPassBorrow()) std::cout << "Pass borrowing hook unavailable; borrow variants will run natively.\n";
    std::ifstream file(fixture, std::ios::binary | std::ios::ate);
    std::vector<HALF> pixels(size_t(w) * h * 4);
    if (!file || size_t(file.tellg()) != pixels.size() * sizeof(HALF)) throw std::runtime_error("Fixture must be tightly packed RGBA16F at the given size.");
    file.seekg(0); file.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(pixels.size() * sizeof(HALF)));
    // Fixtures hold the encoded model input (sRGB-like, 0..1). Present them as finished SDR in linear scRGB, so the
    // app's SDR encode gives the network these values again (white point 1).
    for (size_t i = 0; i < pixels.size(); ++i) {
        const float v = XMConvertHalfToFloat(pixels[i]);
        if (!std::isfinite(v)) throw std::runtime_error("Non-finite fixture pixel.");
        pixels[i] = XMConvertFloatToHalf(i % 4 == 3 ? 1.f : Decode(v));
    }
    const auto runtime = runtimePath.empty() ? ResolveRuntimePath({}, ExecutableDirectory()) : std::filesystem::absolute(runtimePath);
    Gpu gpu; Processor processor(gpu, runtime);
    ComPtr<ID3D12InfoQueue> diagnostics; gpu.device.As(&diagnostics);
    auto input = gpu.Texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON, true);
    auto desc = input->GetDesc(); D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    auto upload = gpu.Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD), readback = gpu.Buffer(bytes, D3D12_HEAP_TYPE_READBACK);
    void* mapped = nullptr; Check(upload->Map(0, nullptr, &mapped), "Borrow fixture upload");
    for (UINT y = 0; y < h; ++y) memcpy(static_cast<BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
        pixels.data() + size_t(y) * w * 4, size_t(w) * 8);
    upload->Unmap(0, nullptr);
    gpu.Begin(); Barrier(gpu.commands.Get(), input.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = upload.Get(); from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; from.PlacedFootprint = footprint;
    to.pResource = input.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    Barrier(gpu.commands.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    gpu.Submit(); if (!gpu.Drain(30000)) throw std::runtime_error("Borrow fixture upload timeout.");

    std::vector<Variant> variants;
    const auto add = [&](std::string name, std::string reference, unsigned passes, std::initializer_list<PassShortcut> shortcuts, float gain = .81f) {
        Variant v{std::move(name), std::move(reference), passes, {}, gain};
        std::copy(shortcuts.begin(), shortcuts.end(), v.shortcuts.begin()); variants.push_back(v);
    };
    PassShortcut reapply{}; reapply.reapply = true;
    PassShortcut transformer{}; transformer.borrowTransformer = true;
    if (swin || !net.empty()) {
        PassShortcut below{}; below.borrowBelow = true; below.borrowLevel = 8;
        const auto pair = [&](const char* name, unsigned passes, std::initializer_list<PassShortcut> shortcuts) {
            add(name, name, passes, shortcuts);
            add(std::string(name) + "-swin4", name, passes, shortcuts); variants.back().swin = swin; variants.back().nvfp4 = nvfp4; variants.back().net = net;
        };
        pair("one", 1, {}); pair("two", 2, {}); pair("three", 3, {});
        pair("two-borrow-transformer", 2, {transformer}); pair("two-borrow-below-8", 2, {below});
        add("one-again", "one", 1, {}); // The runtime's kernels once more: its difference from "one" is the timing noise.
    } else if (nvfp4) {
        for (auto [name, passes] : {std::pair{"one", 1u}, {"two", 2u}, {"three", 3u}}) {
            add(name, name, passes, {});
            add(std::string(name) + "-nvfp4", name, passes, {}); variants.back().nvfp4 = true;
        }
        add("two-borrow-transformer", "two", 2, {transformer});
        add("two-borrow-transformer-nvfp4", "two-borrow-transformer", 2, {transformer}); variants.back().nvfp4 = true;
        add("one-again", "one", 1, {}); // The runtime's kernels once more: its difference from "one" is the timing noise.
    } else {
    add("two", "two", 2, {}); add("one", "two", 1, {});
    add("two-reapply-0.81", "two", 2, {reapply}, .81f); add("two-reapply-1", "two", 2, {reapply}, 1.f);
    add("two-borrow-transformer", "two", 2, {transformer});
    for (unsigned level : {32u, 16u, 8u, 4u, 2u}) {
        PassShortcut below{}; below.borrowBelow = true; below.borrowLevel = level;
        add(std::format("two-borrow-below-{}", level), "two", 2, {below});
    }
    add("three", "three", 3, {}); add("three-borrow-transformer-x2", "three", 3, {transformer, transformer});
    add("three-reapply-last", "three", 3, {PassShortcut{}, reapply});
    }

    struct Result { std::vector<float> model, composed; double nrMs = 0, totalMs = 0; std::vector<NrPassBorrow::Outcome> outcomes; std::string status, nvfp4; };
    std::vector<Result> results;
    for (const auto& v : variants) {
        Result result;
        for (unsigned view : {2u, 0u}) { // Raw model answer (model space), then the composed output (linear).
            Settings s; s.runtime = runtime; s.unlockPasses = true; s.passes = v.passes; s.passShortcuts = v.shortcuts;
            s.nvfp4FeedForward = v.nvfp4; s.nvfp4Swin = v.swin; SetNvfp4Net(s, v.net);
            // A variant with 4-bit parts is the selector's "Model A - NVFP4"; plain Model A never runs them.
            // NR_CHECK_MODEL_CHOICE (for checking the selector itself): 1 keeps "Model A (Original)" with the parts ticked,
            // which must leave every launch to the runtime; 2 gives pass 1 the 4-bit model and the later passes the original.
            const wchar_t* choiceText = _wgetenv(L"NR_CHECK_MODEL_CHOICE"); const int choice = choiceText ? _wtoi(choiceText) : 0;
            s.reapplyGain = v.gain; s.model.preset = preset == 1 && s.Nvfp4Parts() && choice != 1 ? Settings::ModelANvfp4 : preset; s.model.style = style; s.inputColour = 1; s.estimateMotion = false; s.debugView = view;
            if (sameTuning) for (auto& o : s.passOverrides) o = s.model; // every pass uses pass 1's controls (as the research did)
            if (choice == 2) for (unsigned pass = 1; pass < Settings::MaxPasses; ++pass) { auto later = s.PassModel(pass); later.preset = 1; s.passOverrides[pass - 1] = later; }
            processor.Resize(w, h, s);
            std::vector<double> nr, total;
            for (unsigned frame = 0; frame < frames; ++frame) {
                gpu.Begin(); gpu.BeginTimings();
                auto* output = processor.Run(input.Get(), s, false, true, 1);
                if (frame + 1 == frames) {
                    from = {}; from.pResource = output; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    to = {}; to.pResource = readback.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint = footprint;
                    gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                }
                gpu.ResolveTimings(); gpu.Submit();
                if (!gpu.Drain(30000)) throw std::runtime_error("Borrow check frame timeout.");
                std::array<double, unsigned(GpuStage::Count)> stages{};
                if (gpu.TakeTimings(stages) && frame) {
                    nr.push_back(stages[unsigned(GpuStage::NR)]);
                    double sum = 0; for (double value : stages) sum += value; total.push_back(sum);
                }
            }
            if (diagnostics) {
                for (UINT64 i = 0; i < diagnostics->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
                    SIZE_T size = 0; diagnostics->GetMessage(i, nullptr, &size); std::vector<BYTE> message(size);
                    auto* m = reinterpret_cast<D3D12_MESSAGE*>(message.data()); diagnostics->GetMessage(i, m, &size);
                    if (m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) throw std::runtime_error(std::string("D3D12: ") + m->pDescription);
                }
                diagnostics->ClearStoredMessages();
            }
            std::vector<float> values(size_t(w) * h * 3);
            Check(readback->Map(0, nullptr, &mapped), "Borrow readback");
            for (UINT y = 0; y < h; ++y) {
                const auto* row = reinterpret_cast<const HALF*>(static_cast<const BYTE*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch);
                for (UINT x = 0; x < w; ++x) for (UINT c = 0; c < 3; ++c) {
                    const float value = XMConvertHalfToFloat(row[x * 4 + c]);
                    values[(size_t(y) * w + x) * 3 + c] = view == 2 ? Encode(value) : value;
                }
            }
            readback->Unmap(0, nullptr);
            (view == 2 ? result.model : result.composed) = std::move(values);
            if (view == 2) {
                std::sort(nr.begin(), nr.end()); std::sort(total.begin(), total.end());
                if (!nr.empty()) { result.nrMs = nr[nr.size() / 2]; result.totalMs = total[total.size() / 2]; }
                result.outcomes.assign(processor.borrowOutcomes.begin(), processor.borrowOutcomes.begin() + processor.borrowEvaluations);
                char status[256]{}; NrPassBorrow::Status(status, sizeof(status)); result.status = status;
                result.nvfp4 = swin ? Nvfp4SwinStatus(true) : Nvfp4Status();
                if (!net.empty()) result.nvfp4 += " | network: " + Nvfp4NetStatus(true);
            }
        }
        std::cout << std::format("{}: {} | NR {:.3f} ms | {} | NVFP4 {}\n", v.name, Describe(v), result.nrMs, result.status, result.nvfp4) << std::flush;
        results.push_back(std::move(result));
    }
    const auto psnr = [](const std::vector<float>& a, const std::vector<float>& b) {
        double error = 0;
        for (size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - b[i]; error += d * d; }
        error /= double(a.size());
        return error > 0 ? std::format("{:.3f}", -10 * std::log10(error)) : std::string("null");
    };
    const auto index = [&](const std::string& name) {
        for (size_t i = 0; i < variants.size(); ++i) if (variants[i].name == name) return i;
        throw std::runtime_error("Unknown reference variant.");
    };
    std::ofstream json(report, std::ios::binary | std::ios::trunc);
    json << "{\n  \"fixture\": " << Quote(Narrow(fixture.wstring())) << ",\n  \"width\": " << w << ", \"height\": " << h
         << ", \"preset\": " << preset << ", \"style\": " << style << ", \"same_tuning\": " << (sameTuning ? "true" : "false") << ", \"frames\": " << frames << ",\n  \"runtime\": " << Quote(Narrow(runtime.wstring()))
         << ",\n  \"input\": \"fixture values decoded with the sRGB EOTF and processed as finished SDR at white point 1; history reset every frame\""
         << ",\n  \"metric\": \"RGB PSNR, peak 1, aggregate MSE over all pixels and channels, against the reference variant. psnr_model_db: raw model answer (debug view 2) re-encoded to model space; psnr_output_db: final composed linear output. null = identical\""
         << ",\n  \"timing\": \"median GPU NR stage (and all stages) over frames 2..N of the raw-answer run; re-apply/clamp dispatches are inside the NR stage\""
         << ",\n  \"variants\": [\n";
    for (size_t i = 0; i < variants.size(); ++i) {
        const auto& v = variants[i]; const auto& r = results[i]; const auto& reference = results[index(v.reference)];
        std::string outcomes;
        for (auto o : r.outcomes) outcomes += std::string(outcomes.empty() ? "" : ", ") +
            (o == NrPassBorrow::Outcome::Borrowed ? "\"borrowed\"" : o == NrPassBorrow::Outcome::Dropped ? "\"dropped\"" : "\"native\"");
        json << "    {\"name\": " << Quote(v.name) << ", \"passes\": " << Quote(Describe(v)) << ", \"reapply_gain\": " << v.gain
             << ", \"reference\": " << Quote(v.reference) << ", \"psnr_model_db\": " << psnr(r.model, reference.model)
             << ", \"psnr_output_db\": " << psnr(r.composed, reference.composed)
             << std::format(", \"nr_gpu_ms\": {:.4f}, \"total_gpu_ms\": {:.4f}", r.nrMs, r.totalMs)
             << ", \"borrow_outcomes\": [" << outcomes << "], \"borrow_status\": " << Quote(r.status)
             << ", \"nvfp4\": " << (v.nvfp4 ? "true" : "false") << ", \"nvfp4_status\": " << Quote(r.nvfp4) << '}'
             << (i + 1 < variants.size() ? ",\n" : "\n");
    }
    json << "  ]\n}\n";
    if (!json) throw std::runtime_error("Cannot write the borrow check report.");
    std::cout << "Wrote " << Narrow(report.wstring()) << '\n';
    // As the app does when a session ends: every frame above was drained, so nothing can still be using the kernels.
    if (nvfp4 || swin || !net.empty()) { ReleaseNvfp4(gpu.device.Get()); std::cout << "Released the NVFP4 kernels and buffers.\n"; }
}
}
