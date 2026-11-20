// Checks the slider calibrator's shared parts without a game or the NVIDIA runtime:
//   the picture reader and the comparison of two pictures,
//   the solver, on a made-up picture model whose right answer is known,
//   the in-game frame capture handing its pictures to a reader, on the software GPU (WARP).
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <DirectXPackedVector.h>
#include <Util.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../OptiScaler/dlssnr/NrCalibrate.h"
#include "../../OptiScaler/dlssnr/DlssNr_Capture.h"

using Microsoft::WRL::ComPtr;
using namespace NrCalibrate;

namespace
{
void expect(bool yes, const std::string& why)
{
    if (!yes)
        throw std::runtime_error(why);
}
void check(HRESULT hr)
{
    if (FAILED(hr))
        throw std::runtime_error("A Direct3D 12 call failed.");
}
bool about(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }
uint16_t Half(float value) { return DirectX::PackedVector::XMConvertFloatToHalf(value); }

// ---- Pictures -------------------------------------------------------------------------------------------------
void PictureChecks()
{
    Picture a;
    a.width = 2;
    a.height = 1;
    a.values = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    Picture sum;
    expect(sum.Empty() && sum.Add(a) && sum.values == a.values && sum.width == 2, "adding to an empty picture");
    expect(sum.Add(a), "adding a picture of the same size");
    sum.Scale(0.5f);
    expect(Difference(sum, a) < 1e-14, "the mean of two equal pictures is that picture");
    Picture other = a;
    other.width = 1;
    other.height = 2;
    expect(!sum.Add(other) && Difference(a, other) < 0, "pictures of different sizes were combined");
    expect(Difference(Picture {}, Picture {}) < 0, "empty pictures were compared");
    Picture shifted = a;
    for (auto& value : shifted.values)
        value += 0.1f;
    expect(about(Difference(a, shifted), 0.01, 1e-7), "difference of a constant offset");
    expect(about(Decibels(0.01), 20.0, 1e-9) && about(Decibels(0.0), 120.0, 1e-9), "decibel scale");
    std::puts("PASS pictures: add, average, difference, sizes that do not match, decibels");
}

// ---- Reader ---------------------------------------------------------------------------------------------------
void ReaderChecks()
{
    const float infinity = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    // scRGB values and what PQ makes of them (reference figures of the standard: 100 nits 0.5081, 1000 nits 0.7518).
    const float light[8] { 1.25f, 12.5f, 0.0f, -1.0f, nan, infinity, 125.0f, 250.0f };
    const double encoded[8] { 0.5081, 0.7518, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    // 4 x 2 pixels, rows padded with garbage that must not be read.
    const unsigned width = 4, height = 2, pitch16 = 40, pitch32 = 80;
    std::vector<uint16_t> halves(pitch16 / 2 * height, 0x7C00);
    std::vector<float> floats(pitch32 / 4 * height, infinity);
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
            for (unsigned c = 0; c < 4; ++c)
            {
                const float value = c == 3 ? 1.0f : light[y * 4 + x];
                halves[y * (pitch16 / 2) + x * 4 + c] = Half(value);
                floats[y * (pitch32 / 4) + x * 4 + c] = value;
            }
    Reader reader;
    Picture p16, p32;
    expect(reader.Read(p16, DXGI_FORMAT_R16G16B16A16_FLOAT, halves.data(), width, height, pitch16), "16-bit float");
    expect(reader.Read(p32, DXGI_FORMAT_R32G32B32A32_FLOAT, floats.data(), width, height, pitch32), "32-bit float");
    expect(p16.width == 4 && p16.height == 2 && p16.values.size() == 24, "size of a small picture");
    for (unsigned i = 0; i < 8; ++i)
        for (unsigned c = 0; c < 3; ++c)
        {
            expect(about(p16.values[i * 3 + c], encoded[i], 0.0005), "PQ value of a 16-bit float pixel " + std::to_string(i));
            expect(about(p32.values[i * 3 + c], p16.values[i * 3 + c], 1e-6), "32-bit and 16-bit floats disagree");
        }

    // R11G11B10: 1.0 (exponent 15), 0.5 (exponent 14), infinity and a not-a-number.
    const uint32_t packed[4] { 0x3C0u | 0x3C0u << 11 | 0x1E0u << 22, 0x380u | 0x380u << 11 | 0x1C0u << 22,
                              0x7C0u | 0x7C0u << 11 | 0x3E0u << 22, 0x7C1u | 0x7C1u << 11 | 0x3E1u << 22 };
    const float same[16] { 1, 1, 1, 1, 0.5f, 0.5f, 0.5f, 1, infinity, infinity, infinity, 1, nan, nan, nan, 1 };
    Picture p11, expected;
    expect(reader.Read(p11, DXGI_FORMAT_R11G11B10_FLOAT, packed, 4, 1, 16), "R11G11B10");
    expect(reader.Read(expected, DXGI_FORMAT_R32G32B32A32_FLOAT, same, 4, 1, 64), "32-bit float row");
    expect(Difference(p11, expected) < 1e-12, "R11G11B10 values");

    // Integer formats are taken as they are; BGRA has red and blue the other way round.
    const uint32_t ten = 1023u | 512u << 10 | 0u << 20 | 3u << 30;
    Picture p10;
    expect(reader.Read(p10, DXGI_FORMAT_R10G10B10A2_UNORM, &ten, 1, 1, 4), "R10G10B10A2");
    expect(about(p10.values[0], 1.0, 1e-6) && about(p10.values[1], 512.0 / 1023.0, 1e-6) && p10.values[2] == 0, "10-bit values");
    const uint8_t bytes[4] { 255, 128, 0, 7 };
    Picture rgba, bgra;
    expect(reader.Read(rgba, DXGI_FORMAT_R8G8B8A8_UNORM, bytes, 1, 1, 4) && reader.Read(bgra, DXGI_FORMAT_B8G8R8A8_UNORM, bytes, 1, 1, 4),
           "8-bit formats");
    expect(rgba.values[0] == 1.0f && about(rgba.values[1], 128.0 / 255.0, 1e-6) && rgba.values[2] == 0, "RGBA order");
    expect(bgra.values[0] == 0 && bgra.values[2] == 1.0f, "BGRA order");
    expect(!reader.Read(rgba, DXGI_FORMAT_R8_UNORM, bytes, 1, 1, 4) && !Reader::Readable(DXGI_FORMAT_R8_UNORM), "an unknown format was read");
    expect(!reader.Read(rgba, DXGI_FORMAT_R8G8B8A8_UNORM, nullptr, 1, 1, 4), "a missing buffer was read");

    // A large picture is thinned to about 600,000 pixels: every second pixel of 1601 x 901.
    {
        const unsigned w = 1601, h = 901, pitch = w * 4 + 12;
        std::vector<uint8_t> big(size_t(pitch) * h, 99);
        for (unsigned y = 0; y < h; ++y)
            for (unsigned x = 0; x < w; ++x)
            {
                uint8_t* p = &big[size_t(y) * pitch + x * 4];
                p[0] = uint8_t(x % 251);
                p[1] = uint8_t(y % 241);
                p[2] = uint8_t((x + y) % 239);
            }
        Picture thin;
        expect(reader.Read(thin, DXGI_FORMAT_R8G8B8A8_UNORM, big.data(), w, h, pitch), "large picture");
        expect(thin.width == 801 && thin.height == 451 && thin.values.size() == size_t(801) * 451 * 3, "thinned size");
        for (const auto [x, y] : { std::pair { 0u, 0u }, std::pair { 1u, 1u }, std::pair { 800u, 450u }, std::pair { 333u, 207u } })
        {
            const float* p = &thin.values[(size_t(y) * thin.width + x) * 3];
            expect(about(p[0], (x * 2 % 251) / 255.0, 1e-6) && about(p[1], (y * 2 % 241) / 255.0, 1e-6) &&
                       about(p[2], ((x + y) * 2 % 239) / 255.0, 1e-6),
                   "a thinned picture took the wrong pixels");
        }
    }

    // A game's picture before tone mapping: the first picture sets the exposure, later ones keep it.
    {
        Reader scene;
        scene.sceneLight = true;
        std::vector<float> grey(8 * 8 * 4, 4.0f), brighter(8 * 8 * 4, 8.0f);
        grey[0] = -50.0f; // negative: counts as black in the mean
        Picture first, second, again;
        expect(scene.Read(first, DXGI_FORMAT_R32G32B32A32_FLOAT, grey.data(), 8, 8, 128), "scene light");
        const double mean = 4.0 * 191.0 / 192.0, scale = 0.18 / mean;
        const auto curve = [&](double value) { const double x = value * scale; return std::pow(x / (1 + x), 1 / 2.2); };
        expect(first.values[0] == 0 && about(first.values[5], curve(4.0), 1e-5), "exposure from the first picture");
        expect(scene.Read(second, DXGI_FORMAT_R32G32B32A32_FLOAT, brighter.data(), 8, 8, 128), "second picture");
        expect(about(second.values[5], curve(8.0), 1e-5), "a later picture did not keep the first one's exposure");
        scene.Reset();
        expect(scene.Read(again, DXGI_FORMAT_R32G32B32A32_FLOAT, brighter.data(), 8, 8, 128), "after a reset");
        expect(about(again.values[5], std::pow(0.18 / 1.18, 1 / 2.2), 1e-5), "a reset did not set a new exposure");
        // The 16-bit path goes through a table: it must agree with the direct one.
        std::vector<uint16_t> half(8 * 8 * 4, Half(8.0f));
        Picture table;
        expect(scene.Read(table, DXGI_FORMAT_R16G16B16A16_FLOAT, half.data(), 8, 8, 64), "scene light, 16-bit");
        expect(Difference(table, again) < 1e-12, "the 16-bit table disagrees");
    }
    std::puts("PASS reader: every format, row padding, PQ reference values, thinning, scene exposure, bad input");
}

// ---- Solver ---------------------------------------------------------------------------------------------------
// A made-up renderer with the properties of the real one that matter to the search: Intensity stops having an effect
// above 1, the model sliders change nothing while Detail strength is 0, and the five values have one right answer.
// The skin / environment sliders scale the edit by how much a pixel counts as skin, and Highlight guard limits how
// far a pixel may brighten.
Point Five(float intensity, float structure, float tone, float detail, float colour)
{
    return { intensity, structure, tone, detail, colour, 1, 1, 1, 1, 8 };
}
struct Model
{
    unsigned width = 24, height = 24;
    std::vector<float> base, skin, p[5];
    explicit Model(unsigned seed)
    {
        std::mt19937 random(seed);
        std::uniform_real_distribution<float> level(0.2f, 0.8f), pattern(-1.0f, 1.0f);
        const size_t count = size_t(width) * height * 3;
        base.resize(count);
        skin.resize(count);
        for (auto& value : base)
            value = level(random);
        for (auto& value : skin)
            value = std::clamp(pattern(random) * 1.5f, 0.0f, 1.0f); // half the picture is no skin at all
        for (auto& values : p)
        {
            values.resize(count);
            for (auto& value : values)
                value = pattern(random);
        }
    }
    Picture Render(const Point& v) const
    {
        const float i = std::min(v[Intensity], 1.0f), s = v[Structure], t = v[Tone];
        Picture out;
        out.width = width;
        out.height = height;
        out.values.resize(base.size());
        for (size_t n = 0; n < base.size(); ++n)
        {
            const float detail = v[EnvironmentDetail] + (v[SkinDetail] - v[EnvironmentDetail]) * skin[n];
            const float colour = v[EnvironmentColour] + (v[SkinColour] - v[EnvironmentColour]) * skin[n];
            const float edited =
                base[n] +
                detail * v[Detail] * 0.1f * (i * (s * p[0][n] + 0.3f * s * s * p[1][n]) + t * p[2][n] + 0.2f * t * t * p[3][n]) +
                colour * v[Colour] * 0.05f * i * p[4][n];
            out.values[n] = std::min(edited, base[n] * v[Guard]);
        }
        return out;
    }
};

struct Outcome
{
    Point best {};
    double first = 0, loss = 0;
    unsigned trials = 0;
};

template <class Measure> Outcome Solve(const Point& start, const Solver::Options& options, Measure&& measure)
{
    Solver solver;
    solver.Start(start, options);
    double lowest = std::numeric_limits<double>::max();
    unsigned count = 0;
    while (!solver.Done())
    {
        expect(++count <= options.budget, "the solver ran past its budget");
        const Point trial = solver.Trial();
        for (unsigned i = 0; i < Count; ++i)
        {
            expect(trial[i] >= options.low[i] - 1e-6f && trial[i] <= options.high[i] + 1e-6f, "a trial value left its range");
            expect(about(trial[i] * 100.0, std::round(trial[i] * 100.0), 1e-3), "a trial value has more than two decimals");
        }
        if (count == 1)
            for (unsigned i = 0; i < Count; ++i)
                expect(about(trial[i], std::clamp(start[i], options.low[i], options.high[i]), 0.005), "the first trial is not the start");
        const double loss = measure(trial);
        lowest = std::min(lowest, loss);
        solver.Report(loss);
    }
    expect(solver.Trials() == count, "trial count");
    expect(solver.BestLoss() == lowest, "the best loss is not the lowest one reported");
    expect(measure(solver.Best()) == lowest, "the best values are not the ones that gave the lowest loss");
    return { solver.Best(), solver.FirstLoss(), solver.BestLoss(), count };
}

void SolverChecks()
{
    const Model model(7);
    const Point target = Five(0.8f, 1.2f, 0.6f, 1.3f, 1.5f);
    const Picture goal = model.Render(target);
    const auto measure = [&](const Point& values) { return Difference(model.Render(values), goal); };
    const Solver::Options options;
    struct Start
    {
        const char* name;
        Point values;
    };
    const Start starts[] { { "defaults", Five(1, 1, 1, 1, 1) },
                           { "low model values", Five(0.5f, 0.5f, 0.5f, 0.5f, 2) },
                           { "Detail strength 0", Five(1, 1, 1, 0, 1) },
                           { "everything 0", Five(0, 0, 0, 0, 0) },
                           { "everything at its top", Five(2, 2, 2, 2, 4) },
                           { "out of range", Five(-3, 9, 0.333f, 5, -1) } };
    for (const auto& start : starts)
    {
        const auto out = Solve(start.values, options, measure);
        std::printf("  from %-22s %5.1f dB -> %5.1f dB in %4u measurements: %.2f %.2f %.2f %.2f %.2f\n", start.name,
                    Decibels(out.first), Decibels(out.loss), out.trials, out.best[0], out.best[1], out.best[2], out.best[3], out.best[4]);
        expect(out.loss <= out.first, std::string("worse than the start, from ") + start.name);
        // (Several sliders can stand in for one another, so the values need not be the goal's own; the picture must be.)
        expect(Decibels(out.loss) >= 50.0, std::string("the goal's picture was not approached, from ") + start.name);
    }

    // Already there: nothing better exists, the start is kept.
    {
        const auto out = Solve(target, options, measure);
        expect(out.loss == 0 && out.first == 0 && out.best == target, "a perfect start was not kept");
    }
    // A goal this model cannot reach: never worse than the start, and it ends.
    {
        const Model stranger(99);
        const Picture unlike = stranger.Render(Five(1, 1, 1, 1, 1));
        const auto out = Solve(Five(1, 1, 1, 1, 1), options, [&](const Point& v) { return Difference(model.Render(v), unlike); });
        expect(out.loss <= out.first, "an unreachable goal made it worse");
    }
    // A picture that does not answer the sliders at all: it ends, with the start.
    {
        const Point start = Five(0.7f, 0.7f, 0.7f, 0.7f, 0.7f);
        const auto out = Solve(start, options, [](const Point&) { return 0.25; });
        expect(out.best == start && out.trials < options.budget, "a flat response did not end at the start");
    }
    // A small budget is kept to, and the result is still the best measured.
    {
        Solver::Options few;
        few.budget = 40;
        const auto out = Solve(Five(1, 1, 1, 1, 1), few, measure);
        expect(out.trials == 40, "a small budget was not used up exactly");
    }
    // A slider whose range is one value stays there.
    {
        Solver::Options fixed;
        fixed.low[Intensity] = fixed.high[Intensity] = 1.0f;
        fixed.low[Colour] = fixed.high[Colour] = 1.0f;
        const auto out = Solve(Five(0.3f, 1, 1, 1, 3), fixed, measure);
        expect(out.best[Intensity] == 1.0f && out.best[Colour] == 1.0f && out.loss <= out.first, "a fixed slider moved");
    }
    // Skin settings optimisation. Without it the five extra sliders are never touched; with it a goal that uses
    // them is matched better than without.
    {
        const Point skinTarget { 0.8f, 1.2f, 0.6f, 1.3f, 1.5f, 0.6f, 0.8f, 0.9f, 0.5f, 1.15f };
        const Picture skinGoal = model.Render(skinTarget);
        const auto skinMeasure = [&](const Point& values) { return Difference(model.Render(values), skinGoal); };
        const Point starts5[] { { 1, 1, 1, 1, 1, 1, 1, 1, 1, 2 }, { 0.5f, 0.5f, 0.5f, 0.5f, 2, 0.3f, 1, 0.2f, 0.4f, 6 } };
        for (const auto& start : starts5)
        {
            Solver solver;
            solver.Start(start, options);
            while (!solver.Done())
            {
                for (unsigned i = BasicCount; i < Count; ++i)
                    expect(solver.Trial()[i] == start[i], "a skin slider was tried without skin settings optimisation");
                solver.Report(skinMeasure(solver.Trial()));
            }
            Solver::Options with;
            with.skin = true;
            const auto out = Solve(start, with, skinMeasure);
            std::printf("  skin settings: %5.1f dB -> %5.1f dB without, %5.1f dB with (%u measurements): skin %.2f/%.2f environment "
                        "%.2f/%.2f guard %.2f\n",
                        Decibels(out.first), Decibels(solver.BestLoss()), Decibels(out.loss), out.trials, out.best[SkinDetail],
                        out.best[SkinColour], out.best[EnvironmentDetail], out.best[EnvironmentColour], out.best[Guard]);
            expect(out.loss < solver.BestLoss(), "skin settings optimisation did not get closer than the five sliders alone");
            expect(Decibels(out.loss) >= Decibels(solver.BestLoss()) + 6.0, "skin settings optimisation gained less than 6 dB");
        }
    }
    // The plain search leaves alone the sliders outside its mask.
    {
        Search search;
        Search::Options only;
        only.mask = 1u << Detail | 1u << Colour;
        const Point start = Five(0.4f, 0.6f, 0.8f, 0.2f, 3.0f);
        search.Start(start, only);
        unsigned count = 0;
        while (!search.Done())
        {
            expect(++count <= only.budget, "the search ran past its budget");
            const Point trial = search.Trial();
            expect(trial[Intensity] == start[Intensity] && trial[Structure] == start[Structure] && trial[Tone] == start[Tone],
                   "the search moved a slider outside its mask");
            search.Report(measure(trial));
        }
        expect(search.BestLoss() <= search.FirstLoss(), "the search made it worse");
    }
    std::puts("PASS solver: a goal it can reach, from six starts (Detail 0 and out-of-range among them), perfect start, unreachable "
              "goal, flat response, small budget, fixed sliders, skin settings, masked search");
}

// ---- Frame capture to a reader --------------------------------------------------------------------------------
void CaptureChecks()
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
    check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queueDesc {};
    check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));
    check(commands->Close());
    ID3D12CommandList* lists[] { commands.Get() };
    ComPtr<ID3D12Fence> gate, drain;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)));
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&drain)));
    UINT64 serial = 0;

    capture::FrameCapture frames;
    // As the game path does it: the capture hears of every command-list reset and every submission.
    const auto begin = [&]
    {
        check(allocator->Reset());
        check(commands->Reset(allocator.Get(), nullptr));
        frames.ResetRecording(commands.Get());
    };
    const auto submit = [&]
    {
        check(commands->Close());
        queue->ExecuteCommandLists(1, lists);
        frames.Submitted(queue.Get(), 1, lists);
    };
    const auto wait = [&]
    {
        check(queue->Signal(drain.Get(), ++serial));
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        check(drain->SetEventOnCompletion(serial, event));
        const auto result = WaitForSingleObject(event, 10000);
        CloseHandle(event);
        expect(result == WAIT_OBJECT_0, "the GPU did not finish");
    };

    struct Target
    {
        ComPtr<ID3D12Resource> texture;
        D3D12_RESOURCE_STATES state;
    };
    const auto create = [&](unsigned width, unsigned height, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES heap {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.SampleDesc.Count = 1;
        Target target { nullptr, state };
        check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&target.texture)));
        return target;
    };
    // Every pixel gets (value, value / 2, value / 4).
    const auto fill = [&](Target& target, float value)
    {
        const auto desc = target.texture->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout {};
        UINT64 bytes = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &layout, nullptr, nullptr, &bytes);
        D3D12_HEAP_PROPERTIES heap {};
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC buffer {};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = bytes;
        buffer.Height = 1;
        buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> upload;
        check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                              IID_PPV_ARGS(&upload)));
        void* mapped = nullptr;
        check(upload->Map(0, nullptr, &mapped));
        for (unsigned y = 0; y < layout.Footprint.Height; ++y)
        {
            auto* row = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(mapped) + layout.Offset + size_t(y) * layout.Footprint.RowPitch);
            for (unsigned x = 0; x < layout.Footprint.Width; ++x)
            {
                row[x * 4] = Half(value);
                row[x * 4 + 1] = Half(value / 2);
                row[x * 4 + 2] = Half(value / 4);
                row[x * 4 + 3] = Half(1.0f);
            }
        }
        upload->Unmap(0, nullptr);
        begin();
        D3D12_RESOURCE_BARRIER barrier {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = { target.texture.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, target.state, D3D12_RESOURCE_STATE_COPY_DEST };
        commands->ResourceBarrier(1, &barrier);
        D3D12_TEXTURE_COPY_LOCATION to {}, from {};
        to.pResource = target.texture.Get();
        to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from.pResource = upload.Get();
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint = layout;
        commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        commands->ResourceBarrier(1, &barrier);
        submit();
        wait();
    };
    // One NR frame: the picture is copied (when a capture wants it), the list runs, and the next frame begins.
    const auto frame = [&](Target& before, Target& after)
    {
        begin();
        frames.record(commands.Get(), device.Get(), before.texture.Get(), before.state, after.texture.Get(), after.state);
        submit();
    };

    auto before = create(64, 32, D3D12_RESOURCE_STATE_COPY_SOURCE);
    auto after = create(64, 32, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    fill(before, 0.25f);
    fill(after, 1.25f);

    Reader reader;
    struct Seen
    {
        Picture picture;
        ID3D12Resource* buffer = nullptr;
    };
    std::vector<Seen> seen;
    const auto take = [&](const DlssNr::ReadbackImage& image)
    {
        Seen one;
        one.buffer = image.readback.Get();
        expect(image.Read([&](const void* data, const D3D12_SUBRESOURCE_FOOTPRINT& footprint)
                          { expect(reader.Read(one.picture, footprint.Format, data, footprint.Width, footprint.Height, footprint.RowPitch),
                                   "the reader could not read the capture's picture"); }),
               "the capture's buffer could not be mapped");
        seen.push_back(std::move(one));
    };
    // What a picture filled with `value` must read as.
    const auto holds = [&](const Picture& picture, float value, unsigned width, unsigned height)
    {
        std::vector<uint16_t> pixels(size_t(width) * height * 4);
        for (size_t i = 0; i < pixels.size(); i += 4)
        {
            pixels[i] = Half(value);
            pixels[i + 1] = Half(value / 2);
            pixels[i + 2] = Half(value / 4);
            pixels[i + 3] = Half(1.0f);
        }
        Picture wanted;
        Reader same;
        same.Read(wanted, DXGI_FORMAT_R16G16B16A16_FLOAT, pixels.data(), width, height, width * 8);
        return Difference(picture, wanted) == 0.0;
    };
    const std::filesystem::path nowhere;

    // 1. One picture; nothing is handed over while the GPU still works on it or the list could still be replayed.
    expect(frames.request(1, take) && frames.isActive(), "a capture could not be asked for");
    expect(!frames.request(1, take), "a second capture was accepted while one was under way");
    check(queue->Wait(gate.Get(), 1));
    frame(before, after);
    expect(frames.write(nowhere).empty() && seen.empty(), "a picture was handed over before the GPU finished it");
    check(gate->Signal(1));
    wait();
    expect(frames.write(nowhere).empty() && seen.empty(), "a picture was handed over while its command list was still open");
    begin();
    submit();
    wait();
    expect(frames.write(nowhere).empty() && seen.size() == 1, "the finished picture was not handed to the reader");
    expect(holds(seen[0].picture, 1.25f, 64, 32), "the reader got the wrong picture");
    expect(!frames.isActive(), "the capture stayed active after its picture was read");
    expect(frames.write(nowhere).empty() && seen.size() == 1, "a picture was handed over twice");

    // 2. The next capture uses the same buffer again and delivers the new picture.
    fill(after, 2.5f);
    expect(frames.request(1, take), "the next capture could not be asked for");
    frame(before, after);
    wait();
    begin();
    submit();
    wait();
    frames.write(nowhere);
    expect(seen.size() == 2 && holds(seen[1].picture, 2.5f, 64, 32), "the second capture's picture is wrong");
    expect(seen[1].buffer == seen[0].buffer, "the second capture did not use the first one's buffer again");

    // 3. The reader may ask for the next capture itself.
    bool asked = false;
    fill(after, 5.0f);
    expect(frames.request(1, [&](const DlssNr::ReadbackImage& image)
                          {
                              take(image);
                              asked = frames.request(1, take);
                          }),
           "third capture");
    frame(before, after);
    wait();
    begin();
    submit();
    wait();
    frames.write(nowhere);
    expect(seen.size() == 3 && asked && frames.isActive() && holds(seen[2].picture, 5.0f, 64, 32), "asking from inside the reader");
    fill(after, 0.5f);
    frame(before, after);
    wait();
    begin();
    submit();
    wait();
    frames.write(nowhere);
    expect(seen.size() == 4 && holds(seen[3].picture, 0.5f, 64, 32), "the capture asked for from inside the reader");

    // 4. A command list that is thrown away: the picture is taken again on a later frame, never read unfinished.
    fill(after, 3.0f);
    expect(frames.request(1, take), "fifth capture");
    begin();
    frames.record(commands.Get(), device.Get(), before.texture.Get(), before.state, after.texture.Get(), after.state);
    check(commands->Close()); // closed, never executed
    begin();
    submit();
    wait();
    expect(frames.write(nowhere).empty() && seen.size() == 4, "a picture from a discarded command list was handed over");
    expect(frames.isActive(), "the capture gave up after a discarded command list");
    for (unsigned n = 0; n < 3 && seen.size() == 4; ++n)
    {
        frame(before, after);
        wait();
        begin();
        submit();
        wait();
        frames.write(nowhere);
    }
    expect(seen.size() == 5 && holds(seen[4].picture, 3.0f, 64, 32), "no picture after a discarded command list");

    // 5. The picture changes size: new buffers, and the picture of the new size arrives.
    auto smallBefore = create(32, 16, D3D12_RESOURCE_STATE_COPY_SOURCE);
    auto smallAfter = create(32, 16, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    fill(smallBefore, 0.25f);
    fill(smallAfter, 7.0f);
    expect(frames.request(1, take), "sixth capture");
    for (unsigned n = 0; n < 4 && seen.size() == 5; ++n)
    {
        frame(smallBefore, smallAfter);
        wait();
        begin();
        submit();
        wait();
        frames.write(nowhere);
    }
    expect(seen.size() == 6 && seen[5].picture.width == 32 && seen[5].picture.height == 16 && holds(seen[5].picture, 7.0f, 32, 16),
           "the picture of the new size did not arrive");

    // 6. Two frames in one capture arrive in order.
    expect(frames.request(2, take), "two-frame capture");
    fill(smallAfter, 1.0f);
    frame(smallBefore, smallAfter);
    wait();
    fill(smallAfter, 2.0f);
    frame(smallBefore, smallAfter);
    wait();
    begin();
    submit();
    wait();
    frames.write(nowhere);
    expect(seen.size() == 8 && holds(seen[6].picture, 1.0f, 32, 16) && holds(seen[7].picture, 2.0f, 32, 16), "two frames in order");

    // 7. A capture to files (the menu's "capture frames") still works after captures to a reader.
    const auto folder = std::filesystem::temp_directory_path() / ("nr-calibrate-capture-" + std::to_string(GetCurrentProcessId()));
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    expect(frames.request(2), "file capture");
    std::string written;
    for (unsigned n = 0; n < 5 && written.empty(); ++n)
    {
        frame(smallBefore, smallAfter);
        wait();
        begin();
        submit();
        wait();
        written = frames.write(folder);
    }
    const bool files = !written.empty() && std::filesystem::file_size(folder / "after_01.raw", ignored) >= 32 * 16 * 8 &&
                       std::filesystem::exists(folder / "before_00.raw") && std::filesystem::exists(folder / "manifest.txt");
    std::filesystem::remove_all(folder, ignored);
    expect(files && seen.size() == 8 && !frames.isActive(), "a capture to files after captures to a reader");

    frames.release();
    wait();
    frames.FinishSubmitted();
    std::puts("PASS capture to a reader (WARP): waits for the GPU and the list, buffer reuse, asking from the reader, "
              "discarded list, size change, two frames, files afterwards");
}
} // namespace

int main()
try
{
    PictureChecks();
    ReaderChecks();
    SolverChecks();
    CaptureChecks();
    std::puts("NR calibrator checks passed.");
    return 0;
}
catch (const std::exception& error)
{
    std::fprintf(stderr, "FAILED: %s\n", error.what());
    return 1;
}
