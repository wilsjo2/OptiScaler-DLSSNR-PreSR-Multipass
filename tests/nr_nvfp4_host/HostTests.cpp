// The in-game host of the 4-bit NR models (OptiScaler/dlssnr/NrNvfp4.cpp) with the three kernel layers replaced by
// counters: which evaluations it accepts as the model that was asked for, and which it refuses.
// No GPU, NR runtime or game is used. Run with tests/nr_nvfp4_host/run.ps1.
#include <d3d12.h>

#include "../../OptiScaler/dlssnr/NrNvfp4.h"
#include "../../standalone/NrCapture/nvfp4/NrFfnV2.h"
#include "../../standalone/NrCapture/nvfp4/NrFp8.h"
#include "../../standalone/NrCapture/nvfp4/NrHost.h"
#include "../../standalone/NrCapture/nvfp4/NrNet4.h"
#include "../../standalone/NrCapture/nvfp4/NrSwin4.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(x) \
    do \
    { \
        if (!(x)) \
        { \
            std::printf("FAIL line %d: %s\n", __LINE__, #x); \
            std::exit(1); \
        } \
    } while (0)

namespace Mock
{
struct Layer
{
    bool enabled = false, paused = false;
    unsigned set = ~0u;
    unsigned long long work = 0, faults = 0;
    unsigned incomplete = 0;  // a set whose files are not all there (0: none)
    bool defaultIncomplete = false;
    std::string missing;
    unsigned released = 0;
    bool Complete(unsigned asked, std::string* file) const
    {
        const bool complete = asked == ~0u ? !defaultIncomplete : asked != incomplete;
        if (!complete && file)
            *file = missing;
        return complete;
    }
};
Layer swin, net, ffn;
// The FP8 models: NVIDIA's kernels with our records, counted per evaluation.
struct Fp8
{
    bool enabled = false, failEnable = false;
    unsigned set = 0, used = 0, missingSet = 0;
    unsigned long long completed = 0, faults = 0;
    bool faultNext = false; // the next evaluation with a set does not match
    unsigned released = 0;
} fp8;
std::vector<std::string> log;
unsigned long long retiredTicket = 0, freedTicket = 0;
unsigned retireKeep = 0, retireRecent = 0;
// One evaluation of the model as the runtime would launch it: a layer that is on and not paused does its part.
void Evaluate(unsigned long long swinFaults = 0, unsigned long long netFaults = 0, unsigned long long ffnFaults = 0)
{
    for (Layer* layer : { &swin, &net, &ffn })
        if (layer->enabled && !layer->paused)
            layer->work += 10;
    if (swin.enabled && !swin.paused)
        swin.faults += swinFaults;
    if (net.enabled && !net.paused)
        net.faults += netFaults;
    if (ffn.enabled && !ffn.paused)
        ffn.faults += ffnFaults;
    if (fp8.enabled && fp8.set)
    {
        fp8.used = fp8.set;
        if (fp8.faultNext)
            ++fp8.faults, fp8.faultNext = false;
        else
            ++fp8.completed;
    }
}
} // namespace Mock

namespace NrSwin4
{
void* WrapNvapi(unsigned, void* original) { return original; }
void SetLog(void (*)(const char*)) {}
void Enable(const Options&) { Mock::swin.enabled = true; }
void Disable() { Mock::swin.enabled = false; }
void Pause(bool paused) { Mock::swin.paused = paused; }
unsigned UseSet(unsigned set) { return Mock::swin.set = set; }
bool SetComplete(unsigned set, std::string* missing) { return Mock::swin.Complete(set, missing); }
unsigned long long Faults() { return Mock::swin.faults; }
void EndEvaluation(ID3D12GraphicsCommandList*) noexcept {}
void ReleaseDevice(ID3D12Device*) { ++Mock::swin.released; }
std::string Status(bool counters)
{
    return std::string("active, widths 256 128") +
           (counters ? " | blocks replaced: " + std::to_string(Mock::swin.work) + " (width 32: 0) | left to the runtime: 7" : "");
}
} // namespace NrSwin4
namespace NrNet4
{
void* WrapNvapi(unsigned, void* original) { return original; }
void SetLog(void (*)(const char*)) {}
void Enable(const Options&) { Mock::net.enabled = true; }
void Disable() { Mock::net.enabled = false; }
void Pause(bool paused) { Mock::net.paused = paused; }
void UseSet(unsigned set) { Mock::net.set = set; }
bool SetComplete(unsigned set, std::string* missing) { return Mock::net.Complete(set, missing); }
unsigned long long Faults() { return Mock::net.faults; }
void EndEvaluation(ID3D12GraphicsCommandList*) noexcept {}
void ReleaseDevice(ID3D12Device*) { ++Mock::net.released; }
std::string Status(bool counters)
{
    return std::string("split active") + (counters ? " replaced " + std::to_string(Mock::net.work) + " [ffwd 1] left 0" : "");
}
} // namespace NrNet4
namespace NrFfnV2
{
void* WrapNvapi(unsigned, void* original) { return original; }
void SetLog(void (*)(const char*)) {}
void Enable(const Options&) { Mock::ffn.enabled = true; }
void Pause(bool paused) { Mock::ffn.paused = paused; }
void UseSet(unsigned set) { Mock::ffn.set = set; }
bool SetComplete(unsigned set, std::string* missing) { return Mock::ffn.Complete(set, missing); }
unsigned long long Faults() { return Mock::ffn.faults; }
bool Failed() { return false; }
void EndEvaluation(ID3D12GraphicsCommandList*) noexcept {}
void ReleaseDevice(ID3D12Device*) { ++Mock::ffn.released; }
uint64_t RetireIdleSizes(ID3D12Device*, unsigned keep, unsigned recent)
{
    Mock::retireKeep = keep;
    Mock::retireRecent = recent;
    return ++Mock::retiredTicket;
}
void FreeRetired(ID3D12Device*, uint64_t ticket) { Mock::freedTicket = ticket; }
std::string Status(bool counters)
{
    return std::string("active, Model A") +
           (counters ? " | feed-forward pairs replaced: " + std::to_string(Mock::ffn.work) + " | contracts left to the runtime: 0" : "");
}
} // namespace NrFfnV2
namespace NrFp8
{
void* WrapNvapi(unsigned, void* original) { return original; }
void SetLog(void (*)(const char*)) {}
void Enable(const Options&)
{
    if (Mock::fp8.failEnable)
        throw std::runtime_error("no FP8 model in assets/fp8");
    Mock::fp8.enabled = true;
}
void Disable() { Mock::fp8.enabled = false; }
bool Enabled() { return Mock::fp8.enabled; }
void UseSet(unsigned set) { Mock::fp8.set = set; }
bool SetComplete(unsigned set, std::string* missing)
{
    const bool complete = Mock::fp8.enabled && set && set != Mock::fp8.missingSet;
    if (!complete && missing)
        *missing = "fp8/fp8_set" + std::to_string(set) + ".txt and .bin";
    return complete;
}
unsigned long long Faults() { return Mock::fp8.faults; }
std::string Problem() { return "FP8 set 1: launch 57 is cc_repack, Model A's is cc_vit_1d_qkv_chained_fp8"; }
unsigned long long Completed() { return Mock::fp8.completed; }
void EndEvaluation(ID3D12GraphicsCommandList*) noexcept {}
void ReleaseDevice(ID3D12Device*) { ++Mock::fp8.released; }
std::string Status(bool) { return "active"; }
} // namespace NrFp8

// The own pass host (the INT8 models), as counters.
namespace Mock
{
struct HostMock
{
    bool failCreate = false, failRun = false;
    unsigned created = 0, evaluated = 0, released = 0, set = 0;
    std::filesystem::path folder;
    std::string recordsName;
    void* query = nullptr;
} host;
} // namespace Mock
namespace NrHost
{
struct Host::Impl
{
    unsigned width, height, set;
};
Host::Host(std::unique_ptr<Impl> made) : impl(std::move(made)) {}
Host::~Host() = default;
std::unique_ptr<Host> Host::Create(ID3D12Device*, const Options& options, unsigned width, unsigned height, unsigned set)
{
    if (Mock::host.failCreate)
        throw std::runtime_error("int8_set" + std::to_string(set) + ".bin is missing. Nothing was shown.");
    ++Mock::host.created;
    Mock::host.set = set;
    Mock::host.folder = options.records;
    Mock::host.recordsName = options.recordsName;
    return std::unique_ptr<Host>(new Host(std::make_unique<Impl>(Impl { width, height, set })));
}
void Host::Evaluate(ID3D12GraphicsCommandList*, ID3D12Resource*, ID3D12Resource*, ID3D12Resource*, const Settings&, bool)
{
    if (Mock::host.failRun)
        throw std::runtime_error("a kernel could not be launched. Nothing was shown.");
    ++Mock::host.evaluated;
}
Options Int8Options(const std::filesystem::path& folder)
{
    Options options;
    options.enabled = true;
    options.records = folder;
    options.recordsName = "int8_set";
    return options;
}
void ReleaseDevice(ID3D12Device*) { ++Mock::host.released; }
void SetLog(void (*)(const char*)) {}
void SetDriverQuery(void* query) { Mock::host.query = query; }
} // namespace NrHost

namespace
{
using namespace DlssNr::Nvfp4;
// A pass as DlssNr_Dx12_Run.cpp runs it; returns whether its picture may be shown.
bool Pass(unsigned preset, unsigned long long swinFaults = 0, unsigned long long netFaults = 0,
          unsigned long long ffnFaults = 0, bool whole = true)
{
    BeginEvaluation(preset);
    Mock::Evaluate(swinFaults, netFaults, ffnFaults);
    EndEvaluation(nullptr);
    return OursRan(whole);
}
bool AllPaused() { return Mock::swin.paused && Mock::net.paused && Mock::ffn.paused; }
bool Contains(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }
} // namespace

int main()
{
    SetLog([](const char* line) { Mock::log.emplace_back(line); });
    auto* device = reinterpret_cast<ID3D12Device*>(static_cast<size_t>(0x1000));

    // Not loaded: a pass that asks for a 4-bit model is refused; any other pass is none of this module's business.
    CHECK(Pass(1));
    CHECK(!Pass(ModelANvfp4));
    Configure(true, L"assets");
    CHECK(Mock::swin.enabled && Mock::net.enabled && Mock::ffn.enabled && AllPaused());

    // The three models run with their own weight set in every layer, and the layers are idle between evaluations.
    CHECK(Pass(ModelANvfp4) && Running() && Problem().empty());
    CHECK(Mock::swin.set == ~0u && Mock::net.set == ~0u && Mock::ffn.set == ~0u && AllPaused());
    CHECK(Pass(ModelTwoDouble) && Mock::swin.set == 13 && Mock::net.set == 13 && Mock::ffn.set == 13);
    CHECK(Pass(ModelTwoTriple) && Mock::swin.set == 12 && Mock::net.set == 12 && Mock::ffn.set == 12);
    CHECK(Pass(ModelTwoDouble4k) && Mock::swin.set == 10 && Mock::net.set == 10 && Mock::ffn.set == 10);
    CHECK(Pass(ModelTwoTriple4k) && Mock::swin.set == 11 && Mock::net.set == 11 && Mock::ffn.set == 11);
    CHECK(Pass(ModelTwoDouble) && Mock::swin.set == 13);
    // A pass of the runtime's own model in between leaves our kernels out of it.
    {
        const auto work = Mock::swin.work + Mock::net.work + Mock::ffn.work;
        CHECK(Pass(1) && Mock::swin.work + Mock::net.work + Mock::ffn.work == work);
    }
    // A borrowed part of an evaluation runs no split, transformer or feed-forward kernels of its own.
    BeginEvaluation(ModelANvfp4);
    Mock::swin.work += 10;
    EndEvaluation(nullptr);
    CHECK(OursRan(false));

    // Finding 4: a weight file of the model missing in any one layer. Nothing is run in its place (every layer is
    // paused for that evaluation), the pass is refused at once, and the reason names the file.
    for (Mock::Layer* layer : { &Mock::swin, &Mock::net, &Mock::ffn })
    {
        layer->incomplete = 13;
        layer->missing = layer == &Mock::swin ? "swin4/swin4_set13.*"
                         : layer == &Mock::net ? "vit4/vit4_set13.bin"
                                               : "modelA/ffn_v2_set13.bin";
        const auto work = Mock::swin.work + Mock::net.work + Mock::ffn.work;
        BeginEvaluation(ModelTwoDouble);
        CHECK(AllPaused());
        Mock::Evaluate();
        EndEvaluation(nullptr);
        CHECK(!OursRan(true) && !Running());
        CHECK(Mock::swin.work + Mock::net.work + Mock::ffn.work == work);
        CHECK(Contains(Problem(), layer->missing.c_str()) && Contains(Problem(), "missing or incomplete"));
        // The other models are complete and still run; so does this one once its file is back.
        CHECK(Pass(ModelTwoTriple) && Pass(ModelANvfp4));
        layer->incomplete = 0;
        CHECK(Pass(ModelTwoDouble) && Problem().empty());
    }
    // ... and the same for the base model's own (trained) weights.
    Mock::ffn.defaultIncomplete = true;
    Mock::ffn.missing = "modelA/ffn_v2_trained.bin";
    CHECK(!Pass(ModelANvfp4) && Contains(Problem(), "modelA/ffn_v2_trained.bin"));
    CHECK(Pass(ModelTwoDouble));
    Mock::ffn.defaultIncomplete = false;
    CHECK(Pass(ModelANvfp4) && Problem().empty());

    // Finding 5: our kernels ran (every layer's work went up) but a launch was left to the runtime's kernel and
    // weights, or ran with another set's weights. The evaluation is a mix of two models and is refused.
    CHECK(!Pass(ModelTwoDouble, 1, 0, 0));
    CHECK(!Pass(ModelTwoDouble, 0, 1, 0));
    CHECK(!Pass(ModelTwoDouble, 0, 0, 1));
    CHECK(!Pass(ModelANvfp4, 0, 3, 0, false));
    CHECK(!Running());
    // The reason reaches the menu once the failure has lasted (the first evaluations after loading may miss).
    for (int i = 0; i < 40; ++i)
        CHECK(!Pass(ModelTwoDouble, 1, 0, 0));
    CHECK(Contains(Problem(), "runtime's own weights"));
    for (int i = 0; i < 40; ++i)
        CHECK(!Pass(ModelTwoDouble, 0, 1, 0));
    CHECK(Contains(Problem(), "without our kernels or weights"));
    // Faults of an evaluation that did not ask for our model are not held against the next one that does.
    Mock::swin.faults += 5;
    CHECK(Pass(ModelTwoDouble) && Running() && Problem().empty());

    // The FP8 models: their own set each, no 4-bit kernel in their passes, the set released after the evaluation.
    CHECK(Mock::fp8.enabled);
    {
        const auto work = Mock::swin.work + Mock::net.work + Mock::ffn.work;
        CHECK(Pass(ModelFp8Double) && Mock::fp8.used == 1 && Running() && Problem().empty());
        CHECK(Pass(ModelFp8Triple) && Mock::fp8.used == 2);
        CHECK(Pass(ModelFp8Double2) && Mock::fp8.used == 3);
        CHECK(Pass(ModelFp8Double4k) && Mock::fp8.used == 5);
        CHECK(Pass(ModelFp8Triple4k) && Mock::fp8.used == 6);
        CHECK(Pass(ModelFp8Triple2) && Mock::fp8.used == 4);
        CHECK(Mock::swin.work + Mock::net.work + Mock::ffn.work == work && AllPaused() && Mock::fp8.set == 0);
    }
    // ... and a 4-bit pass after them runs its kernels without the FP8 records.
    CHECK(Pass(ModelTwoDouble) && Mock::fp8.used == 4 && Mock::swin.set == 13);
    // Numbers between and beyond the models are refused, never run as another model.
    CHECK(!Pass(14) && !Pass(27) && IsKnown(24) && !IsKnown(27));
    // An FP8 evaluation that did not run wholly on our records is refused at once, with its reason.
    Mock::fp8.faultNext = true;
    CHECK(!Pass(ModelFp8Double) && !Running() && Contains(Problem(), "launch 57"));
    CHECK(Pass(ModelFp8Double) && Problem().empty());
    // An evaluation the FP8 layer never saw (no cc_cb_clear reached it) is refused too.
    BeginEvaluation(ModelFp8Triple);
    EndEvaluation(nullptr);
    CHECK(!OursRan(true) && Contains(Problem(), "did not see"));
    // A missing set: nothing of ours is asked for, the pass is refused, the reason names the files.
    Mock::fp8.missingSet = 2;
    CHECK(!Pass(ModelFp8Triple) && Contains(Problem(), "fp8/fp8_set2"));
    CHECK(Pass(ModelFp8Double));
    Mock::fp8.missingSet = 0;

    // A model number this build does not have is refused, never run as another model.
    CHECK(!Pass(9) && Contains(Problem(), "removed from this build"));

    // Finding 3: the size reclaim and the device release reach the layers.
    const auto ticket = RetireIdle(device, 4);
    CHECK(ticket == Mock::retiredTicket && Mock::retireKeep == 3 && Mock::retireRecent == 4);
    FreeRetired(device, ticket);
    CHECK(Mock::freedTicket == ticket);
    Release(device);
    CHECK(Mock::swin.released == 1 && Mock::net.released == 1 && Mock::ffn.released == 1 && Mock::fp8.released == 1 && !Running());
    CHECK(Pass(ModelANvfp4)); // loads again on demand: still switched on

    // FP8 sets that cannot be loaded: the FP8 passes are refused, the 4-bit ones still run (a new asset folder reloads).
    Mock::fp8.failEnable = true;
    Mock::fp8.enabled = false;
    Configure(true, L"other-assets");
    CHECK(!Pass(ModelFp8Double) && Contains(Problem(), "FP8 models"));
    CHECK(Pass(ModelTwoTriple));

    // The INT8 models run on the own pass host: no layer takes part, and the pass is judged by whether the host ran.
    {
        CHECK(IsKnown(ModelInt8) && IsNvfp4(ModelInt8Double) && !IsInt8(24) && !IsInt8(27));
        CHECK(Int8Set(ModelInt8) == 1 && Int8Set(ModelInt8Double) == 2);
        SetDriverQuery(device);
        CHECK(Mock::host.query == device);
        const auto host = CreateInt8(device, ModelInt8Double, 1920, 1080, L"assets");
        CHECK(host && Mock::host.created == 1 && Mock::host.set == 2 && Mock::host.recordsName == "int8_set");
        CHECK(Mock::host.folder == std::filesystem::path(L"assets") / L"int8");
        const Int8Pass pass;
        const auto work = Mock::swin.work + Mock::net.work + Mock::ffn.work;
        BeginEvaluation(ModelInt8Double);
        CHECK(AllPaused() && Mock::fp8.set == 0);
        CHECK(RunInt8(host, nullptr, pass) && Mock::host.evaluated == 1);
        EndEvaluation(nullptr);
        CHECK(OursRan(true) && Running() && Problem().empty());
        CHECK(Mock::swin.work + Mock::net.work + Mock::ffn.work == work);
        // A pass of the host that did not run is refused at once, with the reason; the next one that runs is shown.
        Mock::host.failRun = true;
        BeginEvaluation(ModelInt8Double);
        CHECK(!RunInt8(host, nullptr, pass));
        EndEvaluation(nullptr);
        CHECK(!OursRan(true) && !Running() && Contains(Problem(), "INT8") && Contains(Problem(), "launched"));
        Mock::host.failRun = false;
        BeginEvaluation(ModelInt8Double);
        CHECK(RunInt8(host, nullptr, pass));
        EndEvaluation(nullptr);
        CHECK(OursRan(true) && Problem().empty());
        // A host that cannot be made: the pass is recorded without it and refused, and the reason names the file.
        Mock::host.failCreate = true;
        const auto none = CreateInt8(device, ModelInt8, 1920, 1080, L"assets");
        CHECK(!none);
        BeginEvaluation(ModelInt8);
        CHECK(!RunInt8(none, nullptr, pass));
        EndEvaluation(nullptr);
        CHECK(!OursRan(true) && Contains(Problem(), "int8_set1"));
        Mock::host.failCreate = false;
        // An INT8 pass leaves the other models as they were.
        CHECK(Pass(ModelTwoTriple) && Pass(1));
        // The comparison's status is what the frame last said.
        SetComparison(ModelInt8, "");
        CHECK(ComparisonShown() == ModelInt8 && ComparisonNote().empty());
        SetComparison(0, "the other model is being set up");
        CHECK(ComparisonShown() == 0 && Contains(ComparisonNote(), "set up"));
        // The menu's list: every model once, sorted by precision (FP8, INT8, NVFP4).
        int rank = 0;
        for (const auto& model : kModels)
        {
            const std::string name = model.name;
            const int now = name.rfind("FP8", 0) == 0 ? 1 : name.rfind("INT8", 0) == 0 ? 2 : name.rfind("NVFP4", 0) == 0 ? 3 : 0;
            CHECK(now >= rank && (model.preset <= 2 || IsKnown(model.preset)));
            CHECK(std::string(ModelName(model.preset)) == name);
            rank = now;
        }
        CHECK(rank == 3 && std::string(ModelName(9)) == "Unavailable preset");
        // One version of each pass-in-one model: an earlier number means the model as it is now.
        CHECK(Current(ModelTwoDouble) == ModelTwoDouble4k && Current(ModelTwoTriple) == ModelTwoTriple4k);
        CHECK(Current(ModelFp8Double) == ModelFp8Double4k && Current(ModelFp8Triple2) == ModelFp8Triple4k);
        CHECK(Current(ModelInt8Double) == ModelInt8Double && std::string(ModelName(ModelFp8Double)) == "FP8 2xPass-in-one");
        Release(device);
        CHECK(Mock::host.released >= 2);
    }

    bool failed = false, running = false;
    for (const auto& line : Mock::log)
    {
        failed = failed || Contains(line, "FAILED");
        running = running || Contains(line, "running");
    }
    CHECK(failed && running);
    std::printf("NR 4-bit host tests passed (%zu log lines)\n", Mock::log.size());
    return 0;
}
