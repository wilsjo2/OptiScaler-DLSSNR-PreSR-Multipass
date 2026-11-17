// The in-game host of the 4-bit NR models (OptiScaler/dlssnr/NrNvfp4.cpp) with the three kernel layers replaced by
// counters: which evaluations it accepts as the model that was asked for, and which it refuses.
// No GPU, NR runtime or game is used. Run with tests/nr_nvfp4_host/run.ps1.
#include <d3d12.h>

#include "../../OptiScaler/dlssnr/NrNvfp4.h"
#include "../../standalone/NrCapture/nvfp4/NrFfnV2.h"
#include "../../standalone/NrCapture/nvfp4/NrNet4.h"
#include "../../standalone/NrCapture/nvfp4/NrSwin4.h"

#include <cstdio>
#include <cstdlib>
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

    // A model number this build does not have is refused, never run as another model.
    CHECK(!Pass(9) && Contains(Problem(), "removed from this build"));

    // Finding 3: the size reclaim and the device release reach the layers.
    const auto ticket = RetireIdle(device, 4);
    CHECK(ticket == Mock::retiredTicket && Mock::retireKeep == 3 && Mock::retireRecent == 4);
    FreeRetired(device, ticket);
    CHECK(Mock::freedTicket == ticket);
    Release(device);
    CHECK(Mock::swin.released == 1 && Mock::net.released == 1 && Mock::ffn.released == 1 && !Running());
    CHECK(Pass(ModelANvfp4)); // loads again on demand: still switched on

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
