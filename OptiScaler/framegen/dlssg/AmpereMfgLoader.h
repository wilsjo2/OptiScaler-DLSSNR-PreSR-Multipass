// AmpereMfgLoader - the RTX 20/30 (SM75/SM86) frame-generation sidecar.
//
// Ported from wilsjo2/main:OptiScaler/framegen/dlssg/AmpereMfgLoader.{h,cpp} and adapted to the payload this
// build actually bundles: sdli1995/dlssg_for_sm86 v0.3.5 (pinned in vendor/dlssg_sm86/PIN.json, contract in
// docs/rtx2030-payload-contract.md). Upstream emits "Native 0.2.3" content for a payload generation that no
// longer exists; the contract document's R1 verdict lists what had to change. Three consequences shape this
// header:
//
//   * the module file name is not upstream's dlssg_sm86.dll hardcoded in four search branches: it is one
//     constant, and tools/check_payload_pin.py --expect-name-from-header reads it back to prove it still
//     matches the pin manifest;
//   * the companion INI is the 0.3.5 proxy schema, not the 0.2.4 native layout, and every value written is
//     clamped into the schema - the payload logs a warning and keeps its own default for anything else;
//   * the payload resolves the kernel family from the physical GPU itself in this build, so upstream's
//     hardware routing helpers (ResolveRouter / ResolveAutoKernelImage) are not ported: Router stays Auto
//     unless the user pins it, and KernelImage is left out unless pinned.
//
// This header is the pure half and includes no OptiScaler header, so tests/ampere_mfg_ini_smoke.cpp compiles
// it alone. Discovery, the companion-INI write, the status and the single Arm() entry point are in
// AmpereMfgLoader.cpp.
#pragma once

#if defined(OPTISCALER_RTX40_MFG)

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace AmpereMfgLoader
{
// The module file name the release ships, exactly as written in vendor/dlssg_sm86/PIN.json bundled_name (the
// renamed variant; todo 2's explicit-load probe measured that it loads cleanly and takes the active role).
// This is the only place the name is spelled - the payload is discovered by this constant, and
//     python tools/check_payload_pin.py --expect-name-from-header OptiScaler/framegen/dlssg/AmpereMfgLoader.h
// fails when it stops matching the pin manifest.
inline constexpr wchar_t kPayloadModuleName[] = L"dlssg_sm86.dll";

// The payload's identity, byte for byte: the size and the SHA-256 vendor/dlssg_sm86/PIN.json pins (files[0],
// the bundled_name entry). Two constants, like the module name above them: the arming ladder compares the
// module it found against these and never reads the manifest at runtime, and its refusal is the named
// PayloadDigestMismatch. tools/check_payload_pin.py --expect-digest-from-header reads BOTH constants back out
// of this header and fails when either stops matching PIN.json:
//     python tools/check_payload_pin.py --expect-digest-from-header OptiScaler/framegen/dlssg/AmpereMfgLoader.h
inline constexpr unsigned long long kPayloadExpectedBytes = 30021920ULL;
inline constexpr char kPayloadExpectedSha256[] = "c3934a09399f022504227c72df0bf8c0de55f9a08880dddde898c5262cefa838";

// The pin the ladder enforces. In the shipped build this is the constant pair above, and nothing can change it
// at runtime. The smoke harnesses compile THIS production file against stub modules that deliberately are not
// the shipped 30 MB proxy (which is why tests/ampere_mfg_eligibility_mocks.h replaces the host sources), so
// they define AMPERE_MFG_PAYLOAD_PIN_SEAM and hand in the pin of the bytes their case treats as the pinned
// payload (tests/ampere_mfg_payload_pin_seam.h, included before this file is compiled). The ladder calls the
// same two accessors either way; the production build never defines the macro, so no test seam exists in the
// shipped binary.
#if defined(AMPERE_MFG_PAYLOAD_PIN_SEAM)
// Provided by tests/ampere_mfg_payload_pin_seam.h.
#else
inline unsigned long long ExpectedPayloadBytes() { return kPayloadExpectedBytes; }

inline std::string ExpectedPayloadSha256() { return kPayloadExpectedSha256; }
#endif

// The companion INI. The name is fixed by the payload - it reads the file from the module's own directory and
// offers no setting for it - so it is not a knob and never gets a second spelling.
inline constexpr wchar_t kPayloadIniName[] = L"dlssg_sm86.ini";

// Where the module is searched, relative to the OptiScaler package root (the directory that holds
// OptiScaler.dll). [0] is the shipped release layout - the zip carries OptiScaler/dlssg_sm86/; [1] covers a
// flat install that keeps the folder beside the DLL. Search order is this order.
inline constexpr std::array<const wchar_t*, 2> kPayloadFolderCandidates { L"OptiScaler\\dlssg_sm86", L"dlssg_sm86" };

// The generated companion INI. The payload reads dlssg_sm86.ini from the folder the module was loaded from, so
// the loader writes one there before the module is loaded. The generated file carries the four keys this loader
// owns plus the factory values of [General] and [Logging]; every other key of the 0.3.5 schema is left out on
// purpose so the payload's own default applies (docs/rtx2030-payload-contract.md, "INI schema keys").
//
// Clamping is strict and total - FormatIniContent always emits a value the payload accepts:
//
//   MaxGeneratedFrames  0..5, saturated. 0 is not "unset": it means the runtime's own value. 5 is the shipped
//                       310.9 runtime's ceiling (6X); the 310.1 runtime clamps 5 back to 3 itself.
//   Optimized           0..3; anything out of range falls back to 1, the payload's own documented rule
//                       ("Out-of-range falls back to 1, not 0") and the factory value.
//   Router              Auto|SM75|SM86; anything else falls back to Auto, and the payload then resolves the
//                       family from the physical GPU.
//   KernelImage         PTX and Cubin are written as asked; Auto, and anything unrecognised, leaves the key
//                       out entirely. The factory INI has no KernelImage key.
//   Mode                Bundled|Auto|Pinned; anything else falls back to Bundled - the runtime embedded in the
//                       payload, which is what this build ships.
//   LogLevel            0..3, saturated; 1 is the factory value. 2 is the level that records the install
//                       signals (backend_install.status / install.active) the receipts read.
struct IniSettings
{
    int MaxGeneratedFrames = 3;       // [FrameGeneration] 1..5 = 2X..6X, 3 = 4X (factory), 0 = runtime's own value
    int Optimized = 1;                // [FrameGeneration] consistency tier; 1 = bit-identical to the official runtime
    std::string Router = "Auto";      // [Compatibility] kernel family; Auto resolves from the physical GPU
    std::string KernelImage = "Auto"; // [Compatibility] PTX|Cubin override; Auto omits the key
    std::string Mode = "Bundled";     // [Runtime] Bundled = the embedded runtime
    int LogLevel = 1;                 // [Logging] 0 off, 1 errors, 2 capability/install records, 3 kernel traces
};

// MaxGeneratedFrames, saturated into the schema range 0..5.
inline int ClampMaxGeneratedFrames(int value)
{
    if (value < 0)
        return 0;

    if (value > 5)
        return 5;

    return value;
}

// Optimized: in range (0..3) as asked, otherwise the payload's own fallback, 1.
inline int ClampOptimized(int value) { return value >= 0 && value <= 3 ? value : 1; }

// Log level, saturated into 0..3.
inline int ClampLogLevel(int value)
{
    if (value < 0)
        return 0;

    if (value > 3)
        return 3;

    return value;
}

// Router: one of the three schema values, otherwise Auto.
inline std::string ClampRouter(const std::string& value)
{
    return value == "Auto" || value == "SM75" || value == "SM86" ? value : "Auto";
}

// Mode: one of the three schema values, otherwise Bundled.
inline std::string ClampMode(const std::string& value)
{
    return value == "Bundled" || value == "Auto" || value == "Pinned" ? value : "Bundled";
}

// KernelImage: an empty result means the key is not written. PTX and Cubin are the recognised overrides.
inline std::string ClampKernelImage(const std::string& value)
{
    return value == "PTX" || value == "Cubin" ? value : "";
}

// The exact bytes of the companion INI for these settings. Deterministic, ASCII, LF line endings.
inline std::string FormatIniContent(const IniSettings& settings)
{
    const int maxFrames = ClampMaxGeneratedFrames(settings.MaxGeneratedFrames);
    const int optimized = ClampOptimized(settings.Optimized);
    const int logLevel = ClampLogLevel(settings.LogLevel);
    const std::string router = ClampRouter(settings.Router);
    const std::string kernelImage = ClampKernelImage(settings.KernelImage);
    const std::string mode = ClampMode(settings.Mode);

    std::string ini;
    ini += "; dlssg_sm86.ini - written by OptiScaler from your settings before the payload is loaded.\n";
    ini += "; The other keys of the 0.3.5 schema keep their defaults; restart the game after a change.\n";
    ini += "\n[General]\n";
    ini += "Enabled=1\n";
    ini += "\n[FrameGeneration]\n";
    ini += "Optimized=" + std::to_string(optimized) + "\n";
    ini += "MaxGeneratedFrames=" + std::to_string(maxFrames) + "\n";
    ini += "\n[Compatibility]\n";
    ini += "Router=" + router + "\n";

    if (!kernelImage.empty())
        ini += "KernelImage=" + kernelImage + "\n";

    ini += "\n[Logging]\n";
    ini += "Level=" + std::to_string(logLevel) + "\n";
    ini += "Directory=dlssg_sm86\\logs\n";
    ini += "\n[Runtime]\n";
    ini += "Mode=" + mode + "\n";

    return ini;
}

// Returns the module path under the first candidate folder that holds it, or an empty path. `tried` collects
// every candidate that was checked, in order, for the status detail. Filesystem only - the smoke drives it
// against a scratch tree.
inline std::filesystem::path FindPayloadModule(const std::filesystem::path& packageRoot,
                                               std::vector<std::filesystem::path>& tried)
{
    for (const wchar_t* relative : kPayloadFolderCandidates)
    {
        const std::filesystem::path candidate = packageRoot / relative / kPayloadModuleName;
        tried.push_back(candidate);

        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error))
            return candidate;
    }

    return {};
}

// What the last Arm() found. Same shape as MfgUnlock::Status: plain fields read through LastStatus(), a short
// human-readable detail, and the shared state vocabulary the receipts quote. A state is only ever set from
// evidence (C5): a handle is load evidence, and a stale GetLastError after a successful load is not - that was
// measured in todo 2. The pipeline is
//     Disabled / Ineligible / Conflict -> PayloadValidated -> Loaded -> BackendInstalled -> FeatureCreated ->
//     Evaluating -> Presenting
// and this revision reaches Loaded: Arm() validates the payload, applies the companion INI and loads the module
// by explicit absolute path at the Streamline init boundary (StreamlineHooks::hkslInit arms before it forwards
// to the game's slInit). BackendInstalled and the states after it are fed by the payload's OWN records
// (`backend_install.status=0` + `install.active=true`, then kernel create/evaluate in
// `dlssg_sm86\logs\*_<pid>.jsonl`) - LoadLibrary returning a handle never claims them (C5).
enum class State
{
    Disabled,              // the setting is off, or nothing has run yet - the default
    Ineligible,            // the physical adapter is neither SM75 nor SM86
    Conflict,              // another frame-generation owner is active
    PayloadMissing,        // no module under any candidate folder
    PayloadIncomplete,     // the module is there but is not a plausible PE image, or is not the pinned size
    PayloadDigestMismatch, // the module is the pinned size but not the pinned bytes (PIN.json)
    PayloadLoadFailed,     // LoadLibrary returned no handle
    PayloadStandby,        // the module loaded as a standby proxy, not as the active one
    IniWriteFailed,        // the companion INI could not be written
    PayloadValidated,      // module found and the companion INI applied
    Loaded,                // a module handle is held
    BackendInstalled,      // the payload's own install signal: backend_install.status=0 and install.active=true
    FeatureCreated,        // a kernel was created
    Evaluating,            // evaluation records were seen
    Presenting,            // generated frames were actually presented
};

// The name as the receipts and the overlay print it.
constexpr const char* StateName(State state)
{
    switch (state)
    {
    case State::Disabled:
        return "Disabled";
    case State::Ineligible:
        return "Ineligible";
    case State::Conflict:
        return "Conflict";
    case State::PayloadMissing:
        return "PayloadMissing";
    case State::PayloadIncomplete:
        return "PayloadIncomplete";
    case State::PayloadDigestMismatch:
        return "PayloadDigestMismatch";
    case State::PayloadLoadFailed:
        return "PayloadLoadFailed";
    case State::PayloadStandby:
        return "PayloadStandby";
    case State::IniWriteFailed:
        return "IniWriteFailed";
    case State::PayloadValidated:
        return "PayloadValidated";
    case State::Loaded:
        return "Loaded";
    case State::BackendInstalled:
        return "BackendInstalled";
    case State::FeatureCreated:
        return "FeatureCreated";
    case State::Evaluating:
        return "Evaluating";
    case State::Presenting:
        return "Presenting";
    }

    return "Unknown";
}

struct Status
{
    State state = State::Disabled;
    std::wstring ModulePath; // where the module was found, empty when it was not
    std::wstring IniPath;    // the companion INI the loader wrote, empty when it did not
    std::string Detail;      // why the loader stopped there, or what it did

    const char* StateText() const { return StateName(state); }
};

// The last Arm() result, safe to read from any thread.
Status LastStatus();

// Everything Arm() needs. Enabled defaults to false: calling Arm() with no options reports Disabled and
// touches nothing, so the loader cannot enable itself before the setting is read and the hkslInit call site
// exists.
struct ArmOptions
{
    bool Enabled = false;
    IniSettings Ini {};
    std::filesystem::path PackageRoot; // empty = the directory that holds OptiScaler.dll
};

// ---------------------------------------------------------------------------
// The gate: physical eligibility and frame-generation ownership
// ---------------------------------------------------------------------------
// Two questions decide whether the payload may run at all, and they are answered here, before any file work:
// is this machine physically one of the two parts the payload supports, and is it this unlocker that owns the
// DLSSG output. Both answers are data - Arm() hands the gate what it read and does nothing else when the
// verdict is not Eligible - so the eligibility matrix drives the same code with fixture facts and no 20/30
// card present.
//
// Physical means physical: the part comes from the adapter the driver reports (IdentifyGpu, the enumeration
// the Ada patcher trusts: a spoofed description is skipped and the architecture id comes from NVAPI on the
// physical GPU handle behind the adapter's LUID). Nothing here reads the setting, the menu or the game, and
// nothing here can be talked into a part by a spoof (the architect advisory: physical identification precedes
// any spoofing).
struct AdapterFacts
{
    int VendorId = 0;                       // PCI vendor id; 0x10DE is NVIDIA
    int ArchitectureId = 0;                 // NV_GPU_ARCHITECTURE_* of the physical GPU, 0 when unknown
    int ImplementationId = 0;               // NV_GPU_ARCH_IMPLEMENTATION_* inside that architecture
    std::string Name;                       // adapter description, for the status text only
    int PhysicalAdapterCount = 0;           // hardware adapters the OS reports; software adapters do not count
    bool CorrelatedToProcessDevice = false; // the classified adapter is the one this process renders on
};

// The architecture ids this build classifies, and the implementations inside Turing that separate RTX 20 from
// GTX 16. The names on the right are the ones in external/nvapi/nvapi.h; AmpereMfgLoader.cpp asserts these
// constants against that header at compile time, so a drift on either side fails the build rather than
// silently reclassifying a card.
inline constexpr int kVendorNvidia = 0x10DE;
inline constexpr int kArchTuring = 0x160;    // NV_GPU_ARCHITECTURE_TU100
inline constexpr int kArchAmpere = 0x170;    // NV_GPU_ARCHITECTURE_GA100
inline constexpr int kArchAda = 0x190;       // NV_GPU_ARCHITECTURE_AD100
inline constexpr int kArchBlackwell = 0x1B0; // NV_GPU_ARCHITECTURE_GB200
inline constexpr int kImplTu102 = 2;         // RTX 20
inline constexpr int kImplTu104 = 4;         // RTX 20
inline constexpr int kImplTu106 = 6;         // RTX 20
inline constexpr int kImplTu117 = 7;         // GTX 16: SM75 without the tensor units
inline constexpr int kImplTu116 = 8;         // GTX 16: SM75 without the tensor units
inline constexpr int kImplGa100 = 0;         // SM80 (datacenter; every other Ampere implementation is GA10x)

enum class PhysicalClass
{
    Unknown, // NVIDIA vendor, but the architecture id is missing
    NotNvidia,
    TuringRtx20,  // SM75 with the tensor units (TU102/TU104/TU106)
    TuringGtx16,  // SM75 without them (TU116/TU117): no DLSS at all
    AmpereGa10x,  // SM86 (GA10x)
    AmpereGa100,  // SM80 (GA100)
    Ada,          // SM89
    Blackwell,    // SM100/SM120
    PreTuring,    // anything before Turing
    Unrecognised, // an NVIDIA architecture this build does not know
};

// The class as the status text names it.
constexpr const char* PhysicalClassName(PhysicalClass part)
{
    switch (part)
    {
    case PhysicalClass::Unknown:
        return "an NVIDIA part whose architecture id is missing";
    case PhysicalClass::NotNvidia:
        return "not an NVIDIA adapter";
    case PhysicalClass::TuringRtx20:
        return "SM75 (RTX 20)";
    case PhysicalClass::TuringGtx16:
        return "a Turing GTX 16 part: SM75 without the RTX 20 tensor units";
    case PhysicalClass::AmpereGa10x:
        return "SM86 (RTX 30)";
    case PhysicalClass::AmpereGa100:
        return "SM80 (GA100)";
    case PhysicalClass::Ada:
        return "SM89 (Ada)";
    case PhysicalClass::Blackwell:
        return "a Blackwell part (SM100 or SM120)";
    case PhysicalClass::PreTuring:
        return "a pre-Turing NVIDIA part";
    case PhysicalClass::Unrecognised:
        return "an NVIDIA architecture this build does not recognise";
    }

    return "an unknown part";
}

// Which part the facts describe. Turing separates cleanly: TU102/TU104/TU106 are the RTX 20 parts, TU116 and
// TU117 the GTX 16 ones (same SM75, no tensor units). Ampere separates GA100 (SM80, datacenter) from every
// other implementation of that architecture (GA10x, SM86). Anything else is named by its family and refused.
inline PhysicalClass Classify(const AdapterFacts& adapter)
{
    if (adapter.VendorId != kVendorNvidia)
        return PhysicalClass::NotNvidia;

    if (adapter.ArchitectureId == 0)
        return PhysicalClass::Unknown;

    if (adapter.ArchitectureId < kArchTuring)
        return PhysicalClass::PreTuring;

    if (adapter.ArchitectureId == kArchTuring)
    {
        switch (adapter.ImplementationId)
        {
        case kImplTu102:
        case kImplTu104:
        case kImplTu106:
            return PhysicalClass::TuringRtx20;
        case kImplTu116:
        case kImplTu117:
            return PhysicalClass::TuringGtx16;
        default:
            return PhysicalClass::Unrecognised;
        }
    }

    if (adapter.ArchitectureId == kArchAmpere)
        return adapter.ImplementationId == kImplGa100 ? PhysicalClass::AmpereGa100 : PhysicalClass::AmpereGa10x;

    if (adapter.ArchitectureId == kArchAda)
        return PhysicalClass::Ada;

    if (adapter.ArchitectureId == kArchBlackwell)
        return PhysicalClass::Blackwell;

    return PhysicalClass::Unrecognised;
}

// The two parts the payload supports. Eligibility is exactly this predicate: it is never widened, and a card
// that is not one of these is a named Ineligible rather than a hopeful attempt.
constexpr bool IsSupportedPart(PhysicalClass part)
{
    return part == PhysicalClass::TuringRtx20 || part == PhysicalClass::AmpereGa10x;
}

inline constexpr const char* kSupportedPartsText = "SM75 (RTX 20) or SM86 (RTX 30)";

// Who owns frame generation right now. One owner: the game's Streamline owns the generated frames and the
// payload only stands in for its runtime; OptiScaler owns upscaling and NR. Every field is read from the
// process by Arm(), or handed in by a fixture.
struct OwnershipFacts
{
    bool ExternalFrameGeneration = false;    // External enabled and no active OptiScaler FG selections
    bool AdaUnlockActive = false;            // the RTX 40 (Ada) unlock is on for this session
    bool OtherOwnerPresent = false;          // another unlocker already holds the DLSSG output
    bool OptiScalerOwnedDlssgOutput = false; // OptiScaler's own DLSSG output path is active
};

// The bundled payload hands the generated frames to the game's Streamline DLSS-G plugin and owns only its own
// runtime and kernels (docs/rtx2030-payload-contract.md, "one FG owner"), so the game must own frame
// generation: [FrameGen] External has to be on. That is the payload contract, not a preference - which is why
// an External=false state is refused and reported instead of being rewritten behind the user's back (C3). A
// plain External=true, exactly what the restored config surface stores when the 20/30 unlock is enabled, is
// the SUPPORTED configuration.
inline constexpr bool kPayloadRequiresGameStreamlineOwnership = true;

// What the gate decided. The three refusals are spelt like the pipeline states they become in LastStatus(), so
// a receipt can quote one vocabulary; Eligible is the gate's "may proceed", and the file work that follows
// decides between PayloadValidated and the payload failures.
enum class Verdict
{
    Disabled,   // the setting is off - nothing runs and nothing is read
    Ineligible, // the physical adapter is neither SM75 (RTX 20) nor SM86 (RTX 30)
    Conflict,   // another frame-generation owner is active
    Eligible,   // may proceed to the payload
};

constexpr const char* VerdictName(Verdict verdict)
{
    switch (verdict)
    {
    case Verdict::Disabled:
        return "Disabled";
    case Verdict::Ineligible:
        return "Ineligible";
    case Verdict::Conflict:
        return "Conflict";
    case Verdict::Eligible:
        return "Eligible";
    }

    return "Unknown";
}

struct Decision
{
    Verdict verdict = Verdict::Disabled;
    std::string Detail; // why the gate stopped there, or what it found

    const char* VerdictText() const { return VerdictName(verdict); }
};

inline std::string AdapterText(const AdapterFacts& adapter)
{
    return adapter.Name.empty() ? std::string("the physical adapter") : "adapter '" + adapter.Name + "'";
}

// The gate itself, and the only place eligibility and ownership are decided. Pure: no file I/O, no GPU access,
// no state - the verdict is a function of what it is handed. Order: off, then the physical part, then the
// ownership conflict; the physical question is answered first, so a machine that cannot run the payload is
// Ineligible even when an unlocker is also active (which is the status this build reports on Ada hosts).
inline Decision Evaluate(const ArmOptions& options, const AdapterFacts& adapter, const OwnershipFacts& ownership)
{
    if (!options.Enabled)
        return { Verdict::Disabled, "the 20/30 unlock is off" };

    // Correlation first: with more than one physical adapter and no correlation to this process's render
    // device, the adapter above is not necessarily the one the game draws on, and the loader will not guess
    // (C1). Fail closed, named.
    if (!adapter.CorrelatedToProcessDevice)
    {
        const std::string count = adapter.PhysicalAdapterCount == 0
                                      ? std::string("no physical adapter")
                                      : std::to_string(adapter.PhysicalAdapterCount) + " physical adapter" +
                                            (adapter.PhysicalAdapterCount == 1 ? "" : "s");
        return { Verdict::Ineligible, "the host reports " + count + " and " + AdapterText(adapter) +
                                          " is not correlated to this process's render device; the payload runs on " +
                                          kSupportedPartsText + " only" };
    }

    const PhysicalClass part = Classify(adapter);

    if (part == PhysicalClass::NotNvidia)
    {
        const std::string cause = adapter.VendorId == 0 ? "the host reports no NVIDIA physical adapter"
                                                        : AdapterText(adapter) + " is not an NVIDIA adapter";
        return { Verdict::Ineligible, cause + "; the payload runs on " + kSupportedPartsText + " only" };
    }

    if (part == PhysicalClass::Ada)
        return { Verdict::Ineligible, AdapterText(adapter) + " is SM89 (Ada); the 40 series path is the Ada "
                                                             "unlock, not this payload" };

    if (!IsSupportedPart(part))
        return { Verdict::Ineligible, AdapterText(adapter) + " is " + PhysicalClassName(part) +
                                          "; the payload runs on " + kSupportedPartsText + " only" };

    // Ownership. Genuine conflicts only - a plain External=true is the supported configuration.
    if (ownership.AdaUnlockActive)
        return { Verdict::Conflict,
                 "the Ada (RTX 40) unlock is active for this session; one frame-generation owner only" };

    if (ownership.OtherOwnerPresent)
        return { Verdict::Conflict,
                 "another unlocker or frame-generation owner is already present: the payload module is loaded "
                 "without this loader having loaded it" };

    if (ownership.OptiScalerOwnedDlssgOutput)
        return { Verdict::Conflict,
                 "OptiScaler's own DLSSG output path is active; it owns the generated frames for this session" };

    if (kPayloadRequiresGameStreamlineOwnership && !ownership.ExternalFrameGeneration)
        return { Verdict::Conflict,
                 "External frame generation does not own this session; enable [FrameGen] External and restart "
                 "with no active OptiScaler FG selections so the game's Streamline owns frame generation" };

    return { Verdict::Eligible, AdapterText(adapter) + " is " + PhysicalClassName(part) +
                                    "; External frame generation is on and no other owner holds the DLSSG "
                                    "output" };
}

// How many times this process entered the payload load path. Arm() counts exactly once, immediately after the
// gate returned Eligible and before any file work, so a refused row leaves the count at zero by construction.
// The eligibility matrix asserts the zero on every refusal row instead of inferring "nothing was loaded" from
// the status string, and todo 11's report quotes it for this host.
int LoadAttempts();

// The entry point. The first call does the file work - discover the payload, write the clamped companion INI,
// load the module by explicit absolute path - and latches its result; every later call returns the latched
// status unchanged (C4: one pass per process, a setting change needs a restart). It is called from the
// Streamline init path (StreamlineHooks::hkslInit) before forwarding to the game's slInit, so the payload and
// its INI are in place before the capability decision (C2), and arming is idempotent by the latch.
//
// Arm() reads the host itself: the settings from the restored config surface, the adapter through IdentifyGpu,
// and the ownership state from the process. The explicit form is what the harnesses and the eligibility matrix
// use to hand in fixture facts.
Status Arm();
Status Arm(const ArmOptions& options);
Status Arm(const ArmOptions& options, const AdapterFacts& adapter, const OwnershipFacts& ownership);
} // namespace AmpereMfgLoader

#endif
