// Adapted from y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG, tag v4 (7b7220bb), GPL-3.0.
#pragma once

#if defined(OPTISCALER_RTX40_MFG)

#include "MfgUnlockMethod.h"
#include "MfgUnlockProvider.h"

#include <SysUtils.h>

#include <atomic>

#include <string>

// Multi Frame Generation on Ada.
//
// nvngx_dlssg.dll gates MFG on the architecture id reported by the driver: 0x1b0 is Blackwell, Ada
// is below it. Two sites decide what a card is allowed to do, and both compare against that constant.
//
//   Advertise, the function that publishes DLSSG.MultiFrameCountMax:
//       mov   ebx, 0x1
//       mov   r8d, 0x3            the Blackwell count
//       cmp   edi, 0x1b0
//       cmovl r8d, ebx            below Blackwell the count becomes 1
//
//   Validate, the function that accepts or rejects a requested count:
//       cmp   eax, 0x1b0
//       jl    ada                 Ada takes this branch and accepts only 1
//       cmp   ebx, 0x3
//       jbe   accept
//
// Patched: the count immediates become 5, the cmovl becomes a nop, and the jl becomes two nops. The
// result is a maximum of five generated frames -- 6X -- on supported Ada GPUs.
//
// Memory only. The file on disk carries an Authenticode signature and is left alone.
//
// A Streamline wrapper between the game and the snippet can carry a lower ceiling of its own. That
// one is raised where the count crosses slDLSSGGetState. Advertise and validate have no such
// boundary: nothing stands between sl.dlss_g.dll and nvngx_dlssg.dll to intercept.
//
// Ada also runs a different interpolation kernel: Kernel_EstimateIntermMvecsScatter reads three f32
// fields of its parameter block on sm_120 and one on sm_89, so every generated frame lands at the
// same point between the two real ones. The Blackwell image is retargeted in place to answer for Ada.
namespace MfgUnlock
{
// What the last attempt found. The signatures are version specific by construction -- they carry the
// shape of the code they patch -- so a module this does not recognise is the expected outcome on a
// version nobody has looked at yet, not a fault. The menu reports this so a report comes back with a
// version number attached rather than "it does not work".
struct Status
{
    bool ModuleFound = false; // nvngx_dlssg.dll was loaded
    bool AdvertiseMatched = false;
    bool ValidateMatched = false;
    unsigned int KernelsRewritten = 0; // kernel groups relabelled, or descriptors redirected to the PTX rebuild
    TemporalMethod TemporalAttempted = TemporalMethod::None; // what the configuration asked for at load
    std::string TemporalDetail;                              // why the attempt did not land, or what it did
    std::string SnippetVersion; // file version of nvngx_dlssg.dll, empty if it could not be read

    // The Streamline DLSS-G plugin's own frame-count clamp. A string literal, empty until a plugin has
    // been seen, so the overlay can read it while a hook thread writes it.
    const char* PluginCeiling = "";

    // Software frame pacing (the flip-metering patch). A string literal like PluginCeiling: empty until a
    // plugin has been seen, "patched", or the reason it was not.
    const char* FlipMetering = "";
    unsigned int FlipSites = 0;
    bool FlipRequested = false; // [DLSSG] AdaFlipMeteringPatch at load, for the restart line
};

Status LastStatus();
bool EnabledForSession();

// The method [DLSSG] AdaTemporalFix selects right now: Retarget unless it names Ptx. The overlay compares
// it with Status::TemporalAttempted to show that a change needs a restart.
TemporalMethod ConfiguredTemporalMethod();

// Applies the patches once per process. Silent and harmless when the config option is off, when
// nvngx_dlssg.dll is not loaded, or when a signature does not match exactly once.
void TryApply(HMODULE module = nullptr);
bool Pending();

// The generated frame ceiling the patches opened, or 0 when they did not land.
unsigned int UnlockedMax();

// What the game's own DLSS-G is doing, for the overlay. Written on the game's threads and read by the
// overlay, so plain atomics. A checkbox proves nothing about MFG: these are the counts Streamline itself
// reported. Not filled while OptiScaler's own frame generation stands in for DLSS-G.
struct Telemetry
{
    std::atomic_bool optionsSeen { false };
    std::atomic_bool active { false };         // the last slDLSSGSetOptions sent left DLSS-G on
    std::atomic<unsigned int> requested { 1 }; // generated frames the game asked for (1 = 2X)
    std::atomic<unsigned int> sent { 1 };      // generated frames sent on, after any override
    std::atomic<unsigned int> result { 0 };    // sl::Result of that call (0 = ok)
    std::atomic_bool stateSeen { false };
    std::atomic<unsigned int> presented { 0 }; // numFramesActuallyPresented at the last slDLSSGGetState
    std::atomic<unsigned int> maxPresented { 0 };
};

const Telemetry& GetTelemetry();
void RecordSetOptions(unsigned int requested, unsigned int sent, bool active, unsigned int result);
void RecordState(unsigned int presented);

// A Streamline DLSS-G plugin (sl.dlss_g) was loaded, from wherever the game or the driver's OTA store
// put it. Its own frame-count clamp is neutralised once the snippet unlock has landed, so a wrapper
// that cached 1 cannot lower the ceiling again. Ordinary threads and the load hook; never scans.
void OnStreamlinePluginLoaded(HMODULE plugin);

// Whether the Streamline plugin was put on software frame pacing.
bool SoftwarePacing();
} // namespace MfgUnlock

#endif
