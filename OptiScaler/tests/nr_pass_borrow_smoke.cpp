// NrPassBorrow smoke test: drives the NvAPI wrappers with fake originals and kernel sequences synthesised from the
// generated tables, then runs the real D3D12 handoff copy on WARP. No NR runtime, NGX or GPU model is used.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <nvapi.h>

#include "NrPassBorrow.h"
#include "NrPassBorrowTables.h"
#include "NrPassBorrowTest.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace NrPassBorrow;
namespace Tb = NrPassBorrow::Tables;
using Microsoft::WRL::ComPtr;
using LaunchFn = NvAPI_Status(__cdecl*)(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
using CreateFn = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle, const char*, NVDX_ObjectHandle*);
using DestroyFn = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle);

static int failures = 0;
#define CHECK(condition, ...)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            ++failures;                                                                                                \
            std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #condition);                                          \
            std::printf(__VA_ARGS__);                                                                                  \
            std::printf("\n");                                                                                         \
        }                                                                                                              \
    } while (0)

// ---- fakes ----
struct Event
{
    bool copy;
    const void* params; // launch: pParams of the forwarded call
    NvU32 count;
    Test::Copy what;    // copy
};
static std::vector<Event> events;
static unsigned secondOriginalCalls = 0;
static NvAPI_Status __cdecl FakeLaunch(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels,
                                       NvU32 count)
{
    events.push_back({ false, kernels ? kernels->pParams : nullptr, count, {} });
    return NVAPI_OK;
}
static NvAPI_Status __cdecl SecondLaunch(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32)
{
    ++secondOriginalCalls;
    return NVAPI_OK;
}
static uintptr_t nextHandle = 0x1000;
static NVDX_ObjectHandle sharedHandle = nullptr; // set: the driver hands out an existing handle again
static NvAPI_Status __cdecl FakeCreate(ID3D12Device*, NVDX_ObjectHandle, const char*, NVDX_ObjectHandle* out)
{
    *out = sharedHandle ? sharedHandle : reinterpret_cast<NVDX_ObjectHandle>(nextHandle += 16);
    return NVAPI_OK;
}
static unsigned destroyed = 0;
static NvAPI_Status __cdecl FakeDestroy(ID3D12Device*, NVDX_ObjectHandle)
{
    ++destroyed;
    return NVAPI_OK;
}
static bool failCopies = false;
static bool RecordCopy(ID3D12GraphicsCommandList*, const Test::Copy& copy, void*)
{
    events.push_back({ true, nullptr, 0, copy });
    return !failCopies;
}
static Test::Gate gate = Test::Gate::R310_9_16;
static Test::Gate GateFn(void*) { return gate; }
static std::vector<std::string> logs;
static void CaptureLog(int, const char* message) { logs.push_back(message); }

static LaunchFn launch;
static CreateFn create;
static DestroyFn destroyFunction;
static std::map<std::string, NVDX_ObjectHandle> handles;
static ID3D12GraphicsCommandList* const list = reinterpret_cast<ID3D12GraphicsCommandList*>(0x5000);

// ---- sequences synthesised from the tables ----
constexpr uint64_t kSpacing = 0x8000000;
struct Sequence
{
    int variant;
    uint64_t base;
    std::vector<std::vector<uint8_t>> params;
    std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS> launches;
    uint64_t Handoff(unsigned cut) const { return base + kSpacing * cut; }
    template <class T> void Put(unsigned k, unsigned offset, T value)
    {
        std::memcpy(params[k].data() + offset, &value, sizeof value);
    }
    uint64_t Counter(unsigned k) const { return base + 64ull * k; }
};
// 1080p extents per cut (rows, cols), as in the traces.
static const unsigned kExtent[7][2] { {}, { 20, 32 }, { 36, 60 }, { 72, 120 }, { 144, 240 }, { 288, 480 }, { 576, 960 } };
// Model A/B handoff bytes at 1080p from the trace audit (map-kernels.md): independent of the tables' scale constants.
static const uint64_t kBytes1080[7] { 0, 655360, 1105920, 2211840, 4423680, 8847360, 17694720 };

static std::set<unsigned> UsedSlots(int variant, unsigned k)
{
    std::set<unsigned> used;
    const auto& v = Tb::kVariants[variant];
    for (unsigned i = 0; i < v.cutCount; ++i)
    {
        const auto& c = v.cuts[i];
        if (k == c.dims)
            used.insert({ c.rows / 8u * 8u, c.cols / 8u * 8u });
        if (k == c.last)
            used.insert({ unsigned(c.output), c.checkRows / 8u * 8u, c.checkCols / 8u * 8u });
        if (k == c.last + 1u)
            used.insert(0);
    }
    return used;
}
static Sequence Make(int variant, uint64_t base)
{
    const auto& v = Tb::kVariants[variant];
    Sequence s { variant, base };
    s.params.resize(v.count);
    s.launches.resize(v.count);
    for (unsigned k = 0; k < v.count; ++k)
    {
        s.params[k].assign(v.kernels[k].paramSize, 0);
        auto& l = s.launches[k];
        l.hFunction = handles.at(v.kernels[k].name);
        l.gridDim = { k + 1, 2, 1 };
        l.blockDim = { 32, 1, 1 };
        l.dynSharedMemBytes = 0;
        l.paramSize = v.kernels[k].paramSize;
        if (k == v.clear)
        {
            s.Put<uint64_t>(k, 0, base);
            s.Put<uint32_t>(k, 8, 4096); // 16 KiB of counters
            continue;
        }
        const auto used = UsedSlots(variant, k);
        for (unsigned offset = (v.kernels[k].paramSize - 8) / 8 * 8; offset >= 8; offset -= 8)
            if (!used.count(offset))
            {
                s.Put<uint64_t>(k, offset, s.Counter(k)); // one private completion counter per launch
                break;
            }
    }
    for (unsigned i = 0; i < v.cutCount; ++i)
    {
        const auto& c = v.cuts[i];
        s.Put<uint32_t>(c.dims, c.rows, kExtent[c.cut][0]);
        s.Put<uint32_t>(c.dims, c.cols, kExtent[c.cut][1]);
        s.Put<uint32_t>(c.last, c.checkRows, kExtent[c.cut][0]);
        s.Put<uint32_t>(c.last, c.checkCols, kExtent[c.cut][1]);
        s.Put<uint64_t>(c.last, c.output, s.Handoff(c.cut));
        s.Put<uint64_t>(c.last + 1u, 0, s.Handoff(c.cut));
    }
    for (unsigned k = 0; k < v.count; ++k)
        s.launches[k].pParams = s.params[k].data();
    return s;
}
static const Tb::CutSpec* Spec(int variant, Cut cut)
{
    const auto& v = Tb::kVariants[variant];
    for (unsigned i = 0; i < v.cutCount; ++i)
        if (v.cuts[i].cut == static_cast<uint8_t>(cut))
            return &v.cuts[i];
    return nullptr;
}
static void Run(Sequence& s, unsigned upTo = ~0u)
{
    for (unsigned k = 0; k < s.launches.size() && k < upTo; ++k)
        launch(list, &s.launches[k], 1);
}
static std::string StatusText()
{
    char buffer[256];
    Status(buffer, sizeof buffer);
    return buffer;
}
// Indices of `s` whose launch was forwarded, in order, from events[from..to).
static std::vector<unsigned> Forwarded(const Sequence& s, size_t from, size_t to)
{
    std::vector<unsigned> result;
    for (size_t e = from; e < to; ++e)
        for (unsigned k = 0; k < s.params.size(); ++k)
            if (!events[e].copy && events[e].params == s.params[k].data())
                result.push_back(k);
    return result;
}
static std::vector<Test::Copy> Copies(size_t from, size_t to)
{
    std::vector<Test::Copy> result;
    for (size_t e = from; e < to; ++e)
        if (events[e].copy)
            result.push_back(events[e].what);
    return result;
}
static std::vector<unsigned> Range(unsigned begin, unsigned end)
{
    std::vector<unsigned> result;
    for (unsigned k = begin; k < end; ++k)
        result.push_back(k);
    return result;
}
static Test::Gate GateOf(int variant)
{
    return Tb::kVariants[variant].runtime == Tb::Runtime::R310_8 ? Test::Gate::R310_8 : Test::Gate::R310_9_16;
}
static std::string Expected(int variant, Cut cut)
{
    const auto& v = Tb::kVariants[variant];
    return std::string("Borrowing: ") + CutName(cut) + " (Model " + v.model +
           (std::strcmp(v.layout, "Default") ? std::string(", ") + v.layout : std::string()) + ", " + v.runtimeName +
           ")";
}

// Evaluation results of one frame.
struct Frame
{
    std::vector<Outcome> outcomes;
    std::vector<std::pair<size_t, size_t>> spans; // events of each evaluation
};
static Frame RunFrame(std::vector<Sequence>& passes, const std::vector<Cut>& cuts,
                      void (*during)(unsigned evaluation, Sequence& s) = nullptr)
{
    Frame frame;
    BeginFrame(list, cuts.data(), static_cast<unsigned>(cuts.size()));
    for (unsigned i = 0; i < passes.size(); ++i)
    {
        const size_t from = events.size();
        BeginEvaluation(i);
        if (during)
            during(i, passes[i]);
        else
            Run(passes[i]);
        frame.outcomes.push_back(EndEvaluation());
        frame.spans.push_back({ from, events.size() });
    }
    EndFrame();
    return frame;
}
static std::vector<Sequence> Passes(int variant, unsigned count)
{
    std::vector<Sequence> passes;
    for (unsigned i = 0; i < count; ++i)
        passes.push_back(Make(variant, 0x100000000ull * (i + 1)));
    return passes;
}

static void TestHelpers()
{
    CHECK(ChooseCut(false, false, 32) == Cut::None, "neither box");
    CHECK(ChooseCut(true, false, 2) == Cut::Transformer, "transformer");
    CHECK(ChooseCut(true, true, 16) == Cut::Below16, "below wins");
    CHECK(ChooseCut(false, true, 2) == Cut::Below2 && ChooseCut(false, true, 4) == Cut::Below4 &&
              ChooseCut(false, true, 8) == Cut::Below8 && ChooseCut(false, true, 32) == Cut::Below32 &&
              ChooseCut(false, true, 1000) == Cut::Below32 && ChooseCut(false, true, 0) == Cut::Below2,
          "levels");
    CHECK(!std::strcmp(CutName(Cut::Below32), "1/32 and below") && !std::strcmp(CutName(Cut::Transformer), "transformer"),
          "names");
}

static void TestWrappers()
{
    CHECK(StatusText() == "Off: hook not installed", "%s", StatusText().c_str());
    CHECK(WrapInterface(0x24973538, nullptr) == nullptr, "null original");
    CHECK(WrapInterface(0x12345678, reinterpret_cast<void*>(&FakeLaunch)) == reinterpret_cast<void*>(&FakeLaunch),
          "other ids pass through");
    launch = reinterpret_cast<LaunchFn>(WrapInterface(0x24973538, reinterpret_cast<void*>(&FakeLaunch)));
    create = reinterpret_cast<CreateFn>(WrapInterface(0xe2436e22, reinterpret_cast<void*>(&FakeCreate)));
    destroyFunction = reinterpret_cast<DestroyFn>(WrapInterface(0xdf295ea6, reinterpret_cast<void*>(&FakeDestroy)));
    const auto destroyModule = reinterpret_cast<DestroyFn>(WrapInterface(0x41c65285, reinterpret_cast<void*>(&FakeDestroy)));
    CHECK(launch && launch != &FakeLaunch && create != &FakeCreate && destroyFunction != &FakeDestroy &&
              destroyModule != &FakeDestroy,
          "wrapped");
    CHECK(WrapInterface(0x24973538, reinterpret_cast<void*>(&FakeLaunch)) == reinterpret_cast<void*>(launch),
          "same original, same wrapper");
    CHECK(WrapInterface(0x24973538, reinterpret_cast<void*>(launch)) == reinterpret_cast<void*>(launch),
          "own wrapper is not wrapped again");
    const auto second = reinterpret_cast<LaunchFn>(WrapInterface(0x24973538, reinterpret_cast<void*>(&SecondLaunch)));
    CHECK(second && second != launch && second != &SecondLaunch, "second original gets its own wrapper");
    NVAPI_CU_KERNEL_LAUNCH_PARAMS kernel {};
    second(list, &kernel, 1);
    CHECK(secondOriginalCalls == 1 && events.empty(), "second wrapper forwards to the second original");

    // Kernel handles for every table name, created through the wrapper (the gate seam is set: no hashing).
    for (const auto& v : Tb::kVariants)
        for (unsigned k = 0; k < v.count; ++k)
            if (!handles.count(v.kernels[k].name))
            {
                NVDX_ObjectHandle handle = nullptr;
                CHECK(create(nullptr, reinterpret_cast<NVDX_ObjectHandle>(0x77), v.kernels[k].name, &handle) ==
                          NVAPI_OK,
                      "create");
                handles[v.kernels[k].name] = handle;
            }
    NVDX_ObjectHandle temporary = nullptr;
    create(nullptr, reinterpret_cast<NVDX_ObjectHandle>(0x78), "cc_cb_clear", &temporary);
    CHECK(destroyFunction(nullptr, temporary) == NVAPI_OK && destroyed == 1, "destroy forwards");
    CHECK(destroyModule(nullptr, reinterpret_cast<NVDX_ObjectHandle>(0x79)) == NVAPI_OK && destroyed == 2,
          "destroy module forwards");

    // A second feature is handed the same kernel handles, then destroyed (a recreated pass): the first feature's
    // kernels must stay known. Every later test borrows through these handles.
    for (const auto& [name, handle] : handles)
    {
        NVDX_ObjectHandle again = nullptr;
        sharedHandle = handle;
        create(nullptr, reinterpret_cast<NVDX_ObjectHandle>(0x77), name.c_str(), &again);
        sharedHandle = nullptr;
        CHECK(again == handle, "shared handle");
        destroyFunction(nullptr, handle);
    }
    CHECK(destroyModule(nullptr, reinterpret_cast<NVDX_ObjectHandle>(0x77)) == NVAPI_OK,
          "destroying a shared module keeps the live feature's kernels");
    destroyed = 0;
}

static void TestTransparent()
{
    // Not armed: forwarded with the caller's exact pointer and count, including count != 1.
    auto s = Make(1, 0x100000000ull);
    events.clear();
    NVAPI_CU_KERNEL_LAUNCH_PARAMS two[2] { s.launches[0], s.launches[1] };
    CHECK(launch(list, two, 2) == NVAPI_OK, "forward");
    Run(s);
    CHECK(events.size() == s.launches.size() + 1 && events[0].count == 2 && events[0].params == two[0].pParams,
          "unarmed calls are forwarded unchanged");
    // A plan without cuts: transparent, no copies.
    std::vector<Sequence> passes = Passes(1, 2);
    events.clear();
    const auto frame = RunFrame(passes, { Cut::None, Cut::None });
    CHECK(frame.outcomes[0] == Outcome::Native && frame.outcomes[1] == Outcome::Native, "native");
    CHECK(Copies(0, events.size()).empty() && events.size() == 2 * passes[0].launches.size(), "plan without cuts");
    // Armed, but another command list: forwarded.
    passes = Passes(1, 2);
    events.clear();
    Cut cuts[] { Cut::None, Cut::Transformer };
    BeginFrame(list, cuts, 2);
    BeginEvaluation(0);
    for (auto& l : passes[0].launches)
        launch(reinterpret_cast<ID3D12GraphicsCommandList*>(0x6000), &l, 1);
    EndEvaluation();
    EndFrame();
    CHECK(events.size() == passes[0].launches.size() && Copies(0, events.size()).empty(), "other list forwarded");
}

static void TestEveryCut()
{
    for (int variant = 0; variant < static_cast<int>(std::size(Tb::kVariants)); ++variant)
    {
        const auto& v = Tb::kVariants[variant];
        gate = GateOf(variant);
        for (unsigned i = 0; i < v.cutCount; ++i)
        {
            const auto& spec = v.cuts[i];
            const Cut cut = static_cast<Cut>(spec.cut);
            auto passes = Passes(variant, 2);
            events.clear();
            Reset();
            const auto frame = RunFrame(passes, { Cut::None, cut });
            const auto [a0, a1] = frame.spans[0];
            const auto [b0, b1] = frame.spans[1];
            const char* label = CutName(cut);
            CHECK(frame.outcomes[0] == Outcome::Native && frame.outcomes[1] == Outcome::Borrowed, "%s %s %s: outcomes",
                  v.model, v.layout, label);
            CHECK(Forwarded(passes[0], a0, a1) == Range(0, v.count), "%s %s %s: template forwards everything", v.model,
                  v.layout, label);
            auto expected = Range(0, spec.first);
            for (unsigned k = spec.last + 1u; k < v.count; ++k)
                expected.push_back(k);
            CHECK(Forwarded(passes[1], b0, b1) == expected, "%s %s %s: borrow forwards exactly outside %u..%u",
                  v.model, v.layout, label, spec.first, spec.last);
            const auto in = Copies(a0, a1), out = Copies(b0, b1);
            CHECK(in.size() == 1 && in[0].toSnapshot && in[0].cut == cut && in[0].arena == passes[0].Handoff(spec.cut) &&
                      in[0].bytes == kBytes1080[spec.cut] && in[0].evaluation == 0,
                  "%s %s %s: snapshot", v.model, v.layout, label);
            CHECK(out.size() == 1 && !out[0].toSnapshot && out[0].cut == cut &&
                      out[0].arena == passes[1].Handoff(spec.cut) && out[0].bytes == kBytes1080[spec.cut] &&
                      out[0].evaluation == 1,
                  "%s %s %s: copy into the consumer", v.model, v.layout, label);
            // Ordering: snapshot right after the producer, copy-in right before the consumer.
            const auto at = [&](const Sequence& s, unsigned k, size_t from, size_t to)
            {
                for (size_t e = from; e < to; ++e)
                    if (!events[e].copy && events[e].params == s.params[k].data())
                        return e;
                return size_t(-1);
            };
            CHECK(events[at(passes[0], spec.last, a0, a1) + 1].copy, "%s %s %s: snapshot after the producer",
                  v.model, v.layout, label);
            const auto consumer = at(passes[1], spec.last + 1u, b0, b1);
            CHECK(consumer != size_t(-1) && consumer > b0 && events[consumer - 1].copy,
                  "%s %s %s: copy before the consumer", v.model, v.layout, label);
            CHECK(StatusText() == Expected(variant, cut), "%s", StatusText().c_str());
        }
    }
    gate = Test::Gate::R310_9_16;
}

static void TestMismatches()
{
    const int variant = 1; // Model A, 310.9.16
    const auto* spec = Spec(variant, Cut::Transformer);
    // Mismatch before the cut: a different kernel at launch 3 -> Native, everything forwarded, not latched.
    {
        auto passes = Passes(variant, 2);
        passes[1].launches[3].hFunction = handles.at("cc_cb_clear");
        events.clear();
        Reset();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Native, "mismatch before the cut is native");
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == Range(0, 156), "all forwarded");
        CHECK(Copies(frame.spans[1].first, frame.spans[1].second).empty(), "no copy-in");
        CHECK(StatusText() == "Off: launch sequence differs", "%s", StatusText().c_str());
        auto again = Passes(variant, 2);
        CHECK(RunFrame(again, { Cut::None, Cut::Transformer }).outcomes[1] == Outcome::Borrowed, "not latched");
    }
    // Grid mismatch (different extent) before the cut -> Native.
    {
        auto passes = Passes(variant, 2);
        passes[1].launches[10].gridDim.x += 1;
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Native, "grid mismatch is native");
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == Range(0, 156), "all forwarded");
        CHECK(StatusText() == "Off: pass sizes differ", "%s", StatusText().c_str());
    }
    // Different handoff extent at the first skipped launch -> Native (refused before skipping).
    {
        auto passes = Passes(variant, 2);
        passes[1].Put<uint32_t>(spec->dims, spec->rows, 24);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Native, "handoff size mismatch is native");
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == Range(0, 156), "all forwarded");
        CHECK(StatusText() == "Off: pass sizes differ", "%s", StatusText().c_str());
    }
    // count != 1 before the cut -> forwarded unchanged, Native.
    {
        auto passes = Passes(variant, 2);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer },
                                    [](unsigned evaluation, Sequence& s)
                                    {
                                        for (unsigned k = 0; k < s.launches.size(); ++k)
                                            if (evaluation == 1 && k == 4)
                                                launch(list, &s.launches[k], 2), ++k;
                                            else
                                                launch(list, &s.launches[k], 1);
                                    });
        CHECK(frame.outcomes[1] == Outcome::Native, "count 2 is native");
        bool sawTwo = false;
        for (size_t e = frame.spans[1].first; e < frame.spans[1].second; ++e)
            sawTwo = sawTwo || (events[e].count == 2 && events[e].params == passes[1].params[4].data());
        CHECK(sawTwo && Copies(frame.spans[1].first, frame.spans[1].second).empty(), "count 2 forwarded unchanged");
        CHECK(StatusText() == "Off: launch call with 2 kernels", "%s", StatusText().c_str());
    }
    // Mismatch inside the cut -> Dropped, nothing launched after it, latched until Reset.
    {
        auto passes = Passes(variant, 2);
        passes[1].launches[spec->first + 3].hFunction = handles.at("cc_cb_clear");
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Dropped, "mismatch inside the cut drops");
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == Range(0, spec->first),
              "nothing after skipping started was launched");
        CHECK(StatusText().rfind("Off after failure: ", 0) == 0, "%s", StatusText().c_str());
        auto next = Passes(variant, 2);
        events.clear();
        const auto latched = RunFrame(next, { Cut::None, Cut::Transformer });
        CHECK(latched.outcomes[1] == Outcome::Native && events.size() == 2 * 156 && Copies(0, events.size()).empty(),
              "latched off: fully transparent");
        Reset();
        auto after = Passes(variant, 2);
        CHECK(RunFrame(after, { Cut::None, Cut::Transformer }).outcomes[1] == Outcome::Borrowed, "Reset clears it");
    }
    // count != 1 inside the cut -> Dropped.
    {
        auto passes = Passes(variant, 2);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer },
                                    [](unsigned evaluation, Sequence& s)
                                    {
                                        for (unsigned k = 0; k < s.launches.size(); ++k)
                                            launch(list, &s.launches[k], evaluation == 1 && k == 60 ? 3 : 1);
                                    });
        CHECK(frame.outcomes[1] == Outcome::Dropped, "count 3 inside the cut drops");
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == Range(0, spec->first),
              "nothing launched after skipping");
        Reset();
    }
    // Consumer does not read the producer -> Dropped, consumer not launched.
    {
        auto passes = Passes(variant, 2);
        passes[1].Put<uint64_t>(spec->last + 1u, 0, passes[1].Handoff(1) + 4096);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Dropped, "consumer mismatch drops");
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == Range(0, spec->first),
              "consumer not launched");
        CHECK(Copies(frame.spans[1].first, frame.spans[1].second).empty(), "no copy-in");
        Reset();
    }
    // A pointer carried from before the cut lies inside the handoff region -> Dropped.
    {
        auto passes = Passes(variant, 2);
        const uint64_t live = passes[1].Handoff(1) + 512;
        passes[1].Put<uint64_t>(5, 8, live);
        passes[1].Put<uint64_t>(spec->last + 1u, 72, live);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Dropped, "live pointer inside the handoff drops");
        CHECK(StatusText() == "Off after failure: handoff region holds a live pointer", "%s", StatusText().c_str());
        Reset();
    }
    // A later launch waits on a counter of a skipped launch -> Dropped after the consumer ran.
    {
        auto passes = Passes(variant, 2);
        uint64_t skipped = 0;
        for (unsigned offset = 0; offset + 8 <= passes[1].params[spec->first + 2].size(); offset += 8)
        {
            uint64_t value;
            std::memcpy(&value, passes[1].params[spec->first + 2].data() + offset, 8);
            if (value == passes[1].Counter(spec->first + 2))
                skipped = value;
        }
        CHECK(skipped != 0, "synthesised counter");
        passes[1].Put<uint64_t>(spec->last + 5, 8, skipped);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Dropped, "skipped counter use drops");
        auto expected = Range(0, spec->first);
        for (unsigned k = spec->last + 1u; k < spec->last + 5u; ++k)
            expected.push_back(k);
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == expected,
              "launches up to the offending one");
        Reset();
    }
    // Short borrowing evaluation -> Dropped.
    {
        auto passes = Passes(variant, 2);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer },
                                    [](unsigned evaluation, Sequence& s) { Run(s, evaluation ? 120 : ~0u); });
        CHECK(frame.outcomes[1] == Outcome::Dropped, "short evaluation drops");
        Reset();
    }
    // Snapshot copy failure -> Native (refused before skipping); copy-in failure -> Dropped.
    {
        auto passes = Passes(variant, 2);
        failCopies = true;
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        failCopies = false;
        CHECK(frame.outcomes[1] == Outcome::Native, "failed snapshot refuses");
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == Range(0, 156), "all forwarded");
        Reset();
    }
    // Unknown kernel handle -> mismatch.
    {
        auto passes = Passes(variant, 2);
        passes[1].launches[0].hFunction = reinterpret_cast<NVDX_ObjectHandle>(0xdead0);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Native, "unknown handle is native");
        Reset();
    }
    // Runtime gate.
    {
        const Test::Gate gates[] { Test::Gate::Checking, Test::Gate::NotAudited, Test::Gate::R310_8 };
        const char* texts[] { "Off: checking runtime", "Off: runtime not audited", "Off: runtime not audited" };
        for (unsigned g = 0; g < 3; ++g)
        {
            gate = gates[g];
            auto passes = Passes(variant, 2);
            events.clear();
            const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
            CHECK(frame.outcomes[1] == Outcome::Native && Copies(0, events.size()).empty() &&
                      events.size() == 2 * 156u,
                  "gate %u refuses transparently", g);
            CHECK(StatusText() == texts[g], "%s", StatusText().c_str());
        }
        gate = Test::Gate::R310_9_16;
        Reset();
    }
    // Model A 310.8 sequence on a 310.9.16 runtime is refused too.
    {
        auto passes = Passes(0, 2);
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Native, "310.8 sequence needs the 310.8 runtime");
        Reset();
    }
}

static void TestThreePasses()
{
    const int variant = 1;
    const auto* t = Spec(variant, Cut::Transformer);
    const auto* b16 = Spec(variant, Cut::Below16);
    {
        auto passes = Passes(variant, 3);
        events.clear();
        Reset();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Borrowed && frame.outcomes[2] == Outcome::Borrowed, "both borrow");
        const auto c0 = Copies(frame.spans[0].first, frame.spans[0].second);
        const auto c1 = Copies(frame.spans[1].first, frame.spans[1].second);
        const auto c2 = Copies(frame.spans[2].first, frame.spans[2].second);
        CHECK(c0.size() == 1 && c0[0].toSnapshot && c0[0].arena == passes[0].Handoff(1), "one snapshot by pass 1");
        CHECK(c1.size() == 1 && !c1[0].toSnapshot && c1[0].arena == passes[1].Handoff(1), "pass 2 copy-in");
        CHECK(c2.size() == 1 && !c2[0].toSnapshot && c2[0].arena == passes[2].Handoff(1), "pass 3 copy-in");
        auto expected = Range(0, t->first);
        for (unsigned k = t->last + 1u; k < 156; ++k)
            expected.push_back(k);
        CHECK(Forwarded(passes[2], frame.spans[2].first, frame.spans[2].second) == expected, "pass 3 skips");
    }
    {
        auto passes = Passes(variant, 3);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Below16, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Borrowed && frame.outcomes[2] == Outcome::Borrowed, "both borrow");
        const auto c0 = Copies(frame.spans[0].first, frame.spans[0].second);
        const auto c1 = Copies(frame.spans[1].first, frame.spans[1].second);
        const auto c2 = Copies(frame.spans[2].first, frame.spans[2].second);
        CHECK(c0.size() == 2 && c0[0].cut == Cut::Transformer && c0[1].cut == Cut::Below16 && c0[0].toSnapshot &&
                  c0[1].toSnapshot && c0[1].bytes == kBytes1080[3],
              "pass 1 snapshots the transformer, then 1/16");
        CHECK(c1.size() == 1 && c1[0].cut == Cut::Below16 && !c1[0].toSnapshot, "pass 2 borrows 1/16");
        CHECK(c2.size() == 1 && c2[0].cut == Cut::Transformer && !c2[0].toSnapshot, "pass 3 borrows the transformer");
        auto expected = Range(0, b16->first);
        for (unsigned k = b16->last + 1u; k < 156; ++k)
            expected.push_back(k);
        CHECK(Forwarded(passes[1], frame.spans[1].first, frame.spans[1].second) == expected, "pass 2 skips 1/16");
        CHECK(StatusText() == "Borrowing: transformer, 1/16 and below (Model A, 310.9.16)", "%s", StatusText().c_str());
    }
    {
        // The latest pass that ran a cut's producer provides it: pass 2 (transformer borrowed) snapshots 1/16.
        auto passes = Passes(variant, 3);
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer, Cut::Below16 });
        CHECK(frame.outcomes[1] == Outcome::Borrowed && frame.outcomes[2] == Outcome::Borrowed, "both borrow");
        const auto c0 = Copies(frame.spans[0].first, frame.spans[0].second);
        const auto c1 = Copies(frame.spans[1].first, frame.spans[1].second);
        const auto c2 = Copies(frame.spans[2].first, frame.spans[2].second);
        CHECK(c0.size() == 1 && c0[0].cut == Cut::Transformer, "pass 1 snapshots the transformer only");
        CHECK(c1.size() == 2 && !c1[0].toSnapshot && c1[0].cut == Cut::Transformer && c1[1].toSnapshot &&
                  c1[1].cut == Cut::Below16 && c1[1].arena == passes[1].Handoff(3),
              "pass 2 borrows the transformer and snapshots 1/16");
        CHECK(c2.size() == 1 && !c2[0].toSnapshot && c2[0].cut == Cut::Below16 && c2[0].arena == passes[2].Handoff(3),
              "pass 3 borrows pass 2's 1/16");
    }
    {
        // A failed snapshot source leaves the borrower native.
        auto passes = Passes(variant, 3);
        passes[0].launches[2].gridDim.y = 7; // pass 1 still records; pass 2/3 differ from it
        events.clear();
        const auto frame = RunFrame(passes, { Cut::None, Cut::Transformer, Cut::Transformer });
        CHECK(frame.outcomes[1] == Outcome::Native && frame.outcomes[2] == Outcome::Native, "sizes differ: native");
    }
    Reset();
}

// The real D3D12 copy path on WARP: the template's handoff reaches the borrowing pass's consumer input.
static void TestWarp()
{
    ComPtr<ID3D12Debug> debug;
    const bool debugLayer = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    if (debugLayer)
        debug->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
    {
        CHECK(false, "WARP device unavailable");
        return;
    }
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queueDesc { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> allocator;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    ComPtr<ID3D12GraphicsCommandList> commands;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands));
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    UINT64 fenceValue = 0;
    const auto submit = [&]
    {
        commands->Close();
        ID3D12CommandList* lists[] { commands.Get() };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), ++fenceValue);
        while (fence->GetCompletedValue() < fenceValue)
            Sleep(1);
        allocator->Reset();
        commands->Reset(allocator.Get(), nullptr);
    };
    const auto buffer = [&](UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES heap { type };
        D3D12_RESOURCE_DESC desc {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;
        ComPtr<ID3D12Resource> resource;
        device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource));
        return resource;
    };
    // Transformer handoff of a 4 x 4 token grid: 4 * 4 * 1024 = 16 KiB. Both "arenas" are 32 KiB.
    constexpr UINT64 bytes = 16384, size = 32768;
    auto arenaA = buffer(size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON);
    auto arenaB = buffer(size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON);
    auto upload = buffer(2 * size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = buffer(size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!arenaA || !arenaB || !upload || !readback)
    {
        CHECK(false, "WARP buffers");
        return;
    }
    uint32_t* mapped = nullptr;
    upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    for (UINT64 i = 0; i < size / 4; ++i)
    {
        mapped[i] = 0x9e3779b9u * static_cast<uint32_t>(i + 1); // arena A: pattern
        mapped[size / 4 + i] = 0xa5a5a5a5u;                    // arena B: sentinel
    }
    upload->Unmap(0, nullptr);
    commands->CopyBufferRegion(arenaA.Get(), 0, upload.Get(), 0, size);
    commands->CopyBufferRegion(arenaB.Get(), 0, upload.Get(), size, size);
    submit();

    const int variant = 1;
    const auto* spec = Spec(variant, Cut::Transformer);
    // Synthetic counters and pointers far from WARP's own virtual addresses (which start at 0x100000000).
    std::vector<Sequence> passes; // moved, never copied: launches point into each sequence's parameter buffers
    passes.push_back(Make(variant, 0x700000000000ull));
    passes.push_back(Make(variant, 0x710000000000ull));
    const UINT64 va[] { arenaA->GetGPUVirtualAddress(), arenaB->GetGPUVirtualAddress() };
    std::printf("WARP arenas at %#llx and %#llx\n", static_cast<unsigned long long>(va[0]),
                static_cast<unsigned long long>(va[1]));
    for (unsigned i = 0; i < 2; ++i)
    {
        passes[i].Put<uint32_t>(spec->dims, spec->rows, 4);
        passes[i].Put<uint32_t>(spec->dims, spec->cols, 4);
        passes[i].Put<uint32_t>(spec->last, spec->checkRows, 4);
        passes[i].Put<uint32_t>(spec->last, spec->checkCols, 4);
        passes[i].Put<uint64_t>(spec->last, spec->output, va[i]);
        passes[i].Put<uint64_t>(spec->last + 1u, 0, va[i]);
    }
    Test::SetCopy(nullptr, nullptr); // the real D3D12 path
    Reset();
    Cut cuts[] { Cut::None, Cut::Transformer };
    Outcome outcomes[2] {};
    BeginFrame(commands.Get(), cuts, 2);
    for (unsigned i = 0; i < 2; ++i)
    {
        BeginEvaluation(i);
        for (auto& l : passes[i].launches)
            launch(commands.Get(), &l, 1);
        outcomes[i] = EndEvaluation();
    }
    EndFrame();
    Test::SetCopy(&RecordCopy, nullptr);
    CHECK(outcomes[0] == Outcome::Native && outcomes[1] == Outcome::Borrowed, "WARP borrow outcome (%s)",
          StatusText().c_str());
    submit();
    commands->CopyBufferRegion(readback.Get(), 0, arenaB.Get(), 0, size);
    submit();
    void* data = nullptr;
    readback->Map(0, nullptr, &data);
    const auto* result = static_cast<const uint32_t*>(data);
    unsigned wrong = 0;
    for (UINT64 i = 0; i < size / 4; ++i)
        wrong += result[i] != (i < bytes / 4 ? 0x9e3779b9u * static_cast<uint32_t>(i + 1) : 0xa5a5a5a5u);
    readback->Unmap(0, nullptr);
    CHECK(!wrong, "WARP: %u words differ (handoff copied, rest untouched)", wrong);
    CHECK(device->GetDeviceRemovedReason() == S_OK, "device removed");
    ComPtr<ID3D12InfoQueue> info;
    if (debugLayer && SUCCEEDED(device.As(&info)))
    {
        UINT64 errors = 0;
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i)
        {
            SIZE_T length = 0;
            info->GetMessage(i, nullptr, &length);
            std::vector<char> storage(length);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            info->GetMessage(i, message, &length);
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
            {
                ++errors;
                std::printf("D3D12: %s\n", message->pDescription);
            }
        }
        CHECK(!errors, "D3D12 debug layer errors");
    }
    std::printf("WARP copy checked (%s debug layer)\n", debugLayer ? "with" : "without");
    Reset();
}

int main()
{
    SetLog(&CaptureLog);
    Test::SetRuntimeGate(&GateFn, nullptr);
    Test::SetCopy(&RecordCopy, nullptr);
    TestHelpers();
    TestWrappers();
    TestTransparent();
    TestEveryCut();
    TestMismatches();
    TestThreePasses();
    TestWarp();
    unsigned cuts = 0;
    for (const auto& v : Tb::kVariants)
        cuts += v.cutCount;
    std::printf("%zu variants, %u cuts, %zu log lines\n", std::size(Tb::kVariants), cuts, logs.size());
    for (const auto& line : logs)
        std::printf("  log: %s\n", line.c_str());
    if (failures)
    {
        std::printf("nr_pass_borrow_smoke: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("nr_pass_borrow_smoke: passed\n");
    return 0;
}
