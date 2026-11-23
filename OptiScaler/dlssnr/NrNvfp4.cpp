// See NrNvfp4.h. Self-contained (no precompiled header). The settings are the ones the Display Filter application
// gives the same three layers (standalone/NrCapture/Processor.cpp: ApplyNvfp4, ApplyNvfp4Swin, ApplyNvfp4Net,
// Processor::Evaluate).
#include "NrNvfp4.h"

#include <windows.h>
#include <d3d12.h>

#include "../../standalone/NrCapture/nvfp4/NrFfnV2.h"
#include "../../standalone/NrCapture/nvfp4/NrFp8.h"
#include "../../standalone/NrCapture/nvfp4/NrHost.h"
#include "../../standalone/NrCapture/nvfp4/NrPassRuntime.h"
#include "../../standalone/NrCapture/nvfp4/NrNet4.h"
#include "../../standalone/NrCapture/nvfp4/NrSwin4.h"

#include <atomic>
#include <exception>
#include <mutex>

namespace DlssNr::Nvfp4
{
namespace
{
struct Work
{
    bool swinActive = false;
    unsigned long long swin = 0, net = 0, ffn = 0;
    // Launches that did not run as asked (each layer: Faults()): any during an evaluation and it is not our model.
    unsigned long long swinFaults = 0, netFaults = 0, ffnFaults = 0;
};

void (*gLog)(const char*) = nullptr;
std::mutex gMutex;
bool gOn = false;
std::filesystem::path gAssets;
bool gAskedDestroy = false;
std::string gEnableProblem; // a layer that could not be switched on: nothing can run
std::string gRefused;       // a path that cannot run these models at all
bool gOurs = false;         // the evaluation in progress asked for a 4-bit model
std::string gSetMissing;     // the model asked for lacks a weight file: nothing else is run in its place
Work gBefore;
std::string gWhy;     // why the last of them did not
bool gRunning = false, gToldFailed = false;
// The FP8 pass-in-one models (NrFp8): loaded beside the 4-bit layers, independent of them (they run on RTX 40 too).
std::string gFp8Problem;    // the FP8 sets could not be loaded
bool gFp8 = false;          // the evaluation in progress asked for an FP8 model
unsigned long long gFp8Faults = 0, gFp8Completed = 0;
// The INT8 models (NrHost): the whole pass is ours, on the own pass host.
bool gInt8 = false;       // the evaluation in progress asked for an INT8 model
bool gHostRan = false;    // ... and the host recorded it
bool gLastInt8 = false;   // the last evaluation judged (OursRan) asked for an INT8 model
unsigned gHostProblemPreset = 0, gEvaluatingPreset = 0;
std::string gHostProblem; // why the host could not be made or could not run
std::atomic<unsigned> gComparisonShown { 0 };
std::string gComparisonNote;

void Say(const std::string& line)
{
    if (gLog)
        gLog(line.c_str());
}

Work Read()
{
    Work work;
    work.swin = NrSwin4::CompletedWork(work.swinActive);
    work.net = NrNet4::CompletedWork();
    work.ffn = NrFfnV2::CompletedWork();
    work.swinFaults = NrSwin4::Faults();
    work.netFaults = NrNet4::Faults();
    work.ffnFaults = NrFfnV2::Faults();
    return work;
}

} // namespace

void SetLog(void (*log)(const char* line))
{
    gLog = log;
    NrFfnV2::SetLog(log);
    NrSwin4::SetLog(log);
    NrNet4::SetLog(log);
    NrFp8::SetLog(log);
    NrHost::SetLog(log);
}

void SetDriverQuery(void* query) { NrHost::SetDriverQuery(query); }

std::shared_ptr<void> CreateHost(ID3D12Device* device, unsigned preset, unsigned width, unsigned height,
                                 const std::filesystem::path& assets)
{
    try
    {
        auto options = IsInt8(preset) ? NrHost::Int8Options(assets / L"int8") : NrHost::FourBitOptions(assets, false);
        return NrPassRuntime::CreateTrackedHost(device, NrPassRuntime::Select(preset), std::move(options), width,
                                                height);
    }
    catch (const std::exception& e)
    {
        std::lock_guard guard(gMutex);
        gHostProblemPreset = preset;
        gHostProblem = std::string(IsInt8(preset) ? "INT8 model: " : "NVFP4 model: ") + e.what();
        Say("Could not create native NR model: " + std::string(e.what()));
        return {};
    }
}

bool RunHost(const std::shared_ptr<void>& host, ID3D12GraphicsCommandList* list, const HostPass& pass)
{
    if (!host)
        return false;
    {
        std::lock_guard guard(gMutex);
        if (gOurs && !gInt8 && (!gOn || !gEnableProblem.empty() || !gSetMissing.empty()))
            return false;
    }
    try
    {
        NrHost::Settings settings;
        settings.style = pass.style;
        settings.intensity = pass.intensity;
        settings.structure = pass.structure;
        settings.tone = pass.tone;
        settings.skin = pass.skin;
        settings.autoMask = pass.autoMask;
        settings.mvScaleX = pass.mvScaleX;
        settings.mvScaleY = pass.mvScaleY;
        static_cast<NrHost::Host*>(host.get())->Evaluate(list, pass.colour, pass.motion, pass.output, settings,
                                                         pass.reset || !pass.motion);
        std::lock_guard guard(gMutex);
        gHostRan = true;
        gHostProblem.clear();
        return true;
    }
    catch (const std::exception& e)
    {
        std::lock_guard guard(gMutex);
        gHostProblemPreset = gEvaluatingPreset;
        gHostProblem = std::string(gInt8 ? "INT8 model: " : "NVFP4 model: ") + e.what();
        return false;
    }
}

void SetComparison(unsigned shownModel, const char* note)
{
    gComparisonShown = shownModel;
    std::lock_guard guard(gMutex);
    gComparisonNote = note ? note : "";
}
unsigned ComparisonShown() { return gComparisonShown; }
std::string ComparisonNote()
{
    std::lock_guard guard(gMutex);
    return gComparisonNote;
}

void* WrapNvapi(unsigned id, void* original)
{
    if (!original)
        return original;
    // The rest-of-the-network layer outermost (it recognises Model A by the runtime's own sequence of launches), then
    // the fused blocks: each of their launches replaces one of the runtime's, so the feed-forward layer below still
    // counts the launches of an evaluation as the runtime made them.
    // The FP8 models outside them all: they count the runtime's own launches of an evaluation (the 4-bit layers are
    // paused in their passes).
    return NrFp8::WrapNvapi(id, NrNet4::WrapNvapi(id, NrSwin4::WrapNvapi(id, NrFfnV2::WrapNvapi(id, original))));
}

void Configure(bool wanted, const std::filesystem::path& assets)
{
    std::lock_guard guard(gMutex);
    if (!wanted || (gOn && assets == gAssets))
        return;
    gOn = true;
    gAssets = assets;
    gEnableProblem.clear();
    if (!gAskedDestroy)
    {
        // The runtime asks NvAPI for an entry point when it first needs it. Ask for the two destroy calls now, through
        // the hook, so the layers can destroy their own kernels even before the runtime has destroyed anything.
        gAskedDestroy = true;
        if (const HMODULE nvapi = GetModuleHandleW(L"nvapi64.dll"))
            if (const auto query = reinterpret_cast<void* (*) (unsigned)>(GetProcAddress(nvapi, "nvapi_QueryInterface")))
                for (const unsigned id : { 0x41c65285u, 0xdf295ea6u }) // DestroyCuModule, DestroyCuFunction
                    query(id);
    }
    const auto failed = [](const char* what, const std::exception& e)
    {
        gEnableProblem += (gEnableProblem.empty() ? "" : "; ") + std::string(what) + ": " + e.what();
    };
    try
    {
        NrFfnV2::Options options;
        options.modelFolders = true;
        options.trained = true;
        options.assets = assets;
        options.minTokens = 0; // every size: the 4-bit model is the 4-bit feed-forward too, whatever the picture size
        NrFfnV2::Enable(options);
    }
    catch (const std::exception& e)
    {
        failed("feed-forward", e);
    }
    try
    {
        NrSwin4::Options options;
        options.assets = assets / L"swin4";
        options.widths = 12; // 256 and 128: the fastest mix at 4K
        options.set = 1;     // the trained codes; a pass-in-one model's own set is chosen per evaluation
        NrSwin4::Enable(options);
    }
    catch (const std::exception& e)
    {
        NrSwin4::Disable();
        failed("fused blocks", e);
    }
    try
    {
        NrNet4::Options options;
        options.assets = assets;
        options.family[NrNet4::Split].groups = 63;
        options.family[NrNet4::Vit].groups = 7;
        options.family[NrNet4::Split].trained = true;
        options.family[NrNet4::Vit].trained = true;
        NrNet4::Enable(options);
    }
    catch (const std::exception& e)
    {
        NrNet4::Disable();
        failed("split blocks and transformer", e);
    }
    gFp8Problem.clear();
    try
    {
        NrFp8::Options options;
        options.assets = assets / L"fp8";
        NrFp8::Enable(options);
    }
    catch (const std::exception& e)
    {
        gFp8Problem = std::string("FP8 models: ") + e.what();
        Say("FAILED to load the FP8 models, so a pass that asks for one is not applied: " + std::string(e.what()));
    }
    // Idle until an evaluation asks: other users of the same NvAPI calls (DLSS itself) pass straight through.
    NrPassRuntime::EnableFourBit(false);
    gWhy.clear();
    gRunning = gToldFailed = false;
    if (gEnableProblem.empty())
        Say("loaded, assets " + assets.string());
    else
        Say("FAILED to load, so a pass that asks for a 4-bit model is not applied: " + gEnableProblem);
}

namespace {
std::string Fp8AssetProblem(unsigned set)
{
    if (!gOn) return "the FP8 models were not loaded";
    if (!gFp8Problem.empty()) return gFp8Problem;
    std::string missing;
    if (!NrFp8::SetComplete(set, &missing)) return "FP8 model unavailable: " + missing;
    return {};
}
}

std::string Fp8SetupProblem(unsigned preset)
{
    std::lock_guard guard(gMutex);
    return Fp8AssetProblem(Fp8Set(preset));
}

bool BeginEvaluation(unsigned requestedPreset, ID3D12GraphicsCommandList* commands)
{
    std::lock_guard guard(gMutex);
    gEvaluatingPreset = requestedPreset;
    const auto model = NrPassRuntime::Select(requestedPreset);
    gOurs = model.backend != NrPassRuntime::Backend::Nvidia;
    gFp8 = model.backend == NrPassRuntime::Backend::Fp8;
    gInt8 = model.backend == NrPassRuntime::Backend::Host;
    gHostRan = false;
    if (gInt8)
    {
        // The whole pass runs on the own pass host: none of the layers has anything to do.
        NrPassRuntime::Activate(model, commands);
        gSetMissing.clear();
        return true;
    }
    if (gFp8)
    {
        // NVIDIA's kernels with our records: no 4-bit kernel runs.
        gSetMissing = Fp8AssetProblem(model.weights);
        if (!gSetMissing.empty()) return false;
        NrPassRuntime::Activate(model, commands);
        gFp8Faults = NrFp8::Faults();
        gFp8Completed = NrFp8::Completed();
        return true;
    }
    const auto taken = NrPassRuntime::BindFourBitWeights(model);
    gSetMissing.clear();
    if (gOurs && !IsKnown(requestedPreset))
        gSetMissing = "model " + std::to_string(requestedPreset) + " was removed from this build; choose another NR model";
    else if (gOurs && gOn && gEnableProblem.empty())
    {
        // Every part of the model: a layer whose file of this set is absent would run another set's weights instead.
        std::string file;
        if (!NrSwin4::SetComplete(taken, &file) || !NrNet4::SetComplete(taken, &file) ||
            !NrFfnV2::SetComplete(taken, &file))
            gSetMissing = "a weight file of this model is missing or incomplete (OptiScaler/nvfp4/" + file + ")";
    }
    if (gOurs && (!gOn || !gSetMissing.empty() || !gEnableProblem.empty())) return false;
    NrPassRuntime::Activate(model, commands);
    if (gOurs) gBefore = Read();
    return true;
}

void EndEvaluation(ID3D12GraphicsCommandList* list) { NrPassRuntime::End(list); }

bool OursRan(bool whole)
{
    std::lock_guard guard(gMutex);
    if (!gOurs)
        return true;
    gRefused.clear();
    gLastInt8 = gInt8;
    std::string why;
    if (gInt8)
    {
        if (!gHostRan)
            why = gHostProblem.empty() ? "the INT8 model's pass was not run" : gHostProblem;
    }
    else if (gFp8)
    {
        if (!gOn)
            why = "the FP8 models were not loaded";
        else if (!gFp8Problem.empty())
            why = gFp8Problem;
        else if (!gSetMissing.empty())
            why = gSetMissing;
        else if (NrFp8::Faults() != gFp8Faults)
            why = NrFp8::Problem();
        else if (NrFp8::Completed() == gFp8Completed)
            why = "the FP8 model did not see this pass's evaluation (" + NrFp8::Status(false) + ")";
    }
    else if (!gOn)
        why = "the 4-bit kernels were not loaded";
    else if (!gEnableProblem.empty())
        why = gEnableProblem;
    else if (!gHostRan && gHostProblemPreset == gEvaluatingPreset && !gHostProblem.empty())
        why = gHostProblem;
    else if (!gSetMissing.empty())
        why = gSetMissing;
    else if (!gHostRan)
        why = "the NVFP4 host did not record this pass";
    else
    {
        const Work now = Read();
        if (!now.swinActive || now.swin <= gBefore.swin)
            why = "fused blocks: " + NrSwin4::Status(false);
        else if (whole && now.net <= gBefore.net)
            why = "split blocks and transformer: " + NrNet4::Status(false);
        else if (whole && now.ffn <= gBefore.ffn)
            why = "feed-forward: " + NrFfnV2::Status(false);
        // Our kernels ran, but not all of the pass: a launch that was left to the runtime's kernel and weights, or run
        // with another weight set, makes the picture a mix of two models.
        else if (now.swinFaults > gBefore.swinFaults)
            why = "part of this pass ran with the runtime's own weights (fused blocks: " + NrSwin4::Status(false) + ")";
        else if (now.netFaults > gBefore.netFaults)
            why = "part of this pass ran without our kernels or weights (split blocks and transformer: " +
                  NrNet4::Status(false) + ")";
        else if (now.ffnFaults > gBefore.ffnFaults)
            why = "part of this pass ran without our kernels or weights (feed-forward: " + NrFfnV2::Status(false) + ")";
    }
    if (why.empty())
    {
        if (!gRunning)
            Say(gInt8 ? std::string("running: this pass is an INT8 model of ours on the own pass host")
                : gFp8 ? "running: this pass is an FP8 model of ours (NVIDIA's kernels with our weights) | " + NrFp8::Status(false)
                       : "running: this pass is our 4-bit kernels | " + NrSwin4::Status(false) + " | " + NrNet4::Status(false));
        gRunning = true;
        gToldFailed = false;
        gWhy.clear();
        return true;
    }
    gRunning = false;
    gWhy = std::move(why);
    if (!gToldFailed)
    {
        gToldFailed = true;
        Say(std::string("FAILED: the pass asked for ") + (gInt8 ? "an INT8" : gFp8 ? "an FP8" : "a 4-bit") +
            " model and it did not run, so the pass is not applied (no fallback to the original model): " + gWhy);
    }
    return false;
}

void Refuse(const char* reason)
{
    std::lock_guard guard(gMutex);
    const std::string text = reason ? reason : "";
    if (gRefused != text)
        Say("FAILED: " + text + "; the pass is not applied (no fallback to the original model)");
    gRefused = text;
    gRunning = false;
}

std::string Problem()
{
    std::lock_guard guard(gMutex);
    if (!gRefused.empty())
        return gRefused;
    // (An INT8 pass uses none of the layers: a layer that could not be switched on is not its problem.)
    if (!gEnableProblem.empty() && !gLastInt8)
        return gEnableProblem;
    return gWhy;
}

bool Running()
{
    std::lock_guard guard(gMutex);
    return gRunning;
}

unsigned long long RetireIdle(ID3D12Device* device, unsigned evaluations)
{
    // Three sizes stay ready (a game that steps its render resolution moves between a few), and every size a pass of
    // this frame used, so that a frame with more than three pass sizes does not allocate them again each frame.
    return NrFfnV2::RetireIdleSizes(device, 3, evaluations);
}

void FreeRetired(ID3D12Device* device, unsigned long long ticket) { NrFfnV2::FreeRetired(device, ticket); }

void Release(ID3D12Device* device)
{
    NrHost::ReleaseDevice(device);
    {
        std::lock_guard guard(gMutex);
        if (!gOn)
            return;
        gRunning = false;
    }
    NrFfnV2::ReleaseDevice(device);
    NrSwin4::ReleaseDevice(device);
    NrNet4::ReleaseDevice(device);
    NrFp8::ReleaseDevice(device);
    Say("released the 4-bit kernels, the FP8 records, weights and buffers of a device");
}

std::string Status()
{
    return "feed-forward: " + NrFfnV2::Status(true) + " | fused blocks: " + NrSwin4::Status(true) + " | network: " +
           NrNet4::Status(true) + " | FP8 models: " + NrFp8::Status(true);
}
} // namespace DlssNr::Nvfp4
