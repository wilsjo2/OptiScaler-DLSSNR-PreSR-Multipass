// AmpereMfgLoader.cpp - the process half of the RTX 20/30 (SM75/SM86) sidecar: the host facts the gate
// decides on, the payload discovery, the companion-INI write and the single Arm() entry point. Compiled into
// the RTX40-MFG flavour only, next to MfgUnlock.cpp.
//
// The clamp, the INI text, the discovery candidates and the gate (AdapterFacts/OwnershipFacts/Evaluate) are in
// the header; this file owns the host reads, the file I/O, the process-latched status, the load-attempt count
// and the logging. Arm() is the single arming entry point and it LOADS the payload: validate (a plausible PE
// image, then the PIN.json size and digest) -> companion INI -> LoadLibrary by explicit absolute path -> Loaded
// (or the payload's own standby role as a named refusal). The call site is StreamlineHooks::hkslInit, before it
// forwards to the game's slInit (C2); the module handle is the only load evidence, and BackendInstalled is left
// to the payload's own install records (C5).
#include "pch.h"

#if defined(OPTISCALER_RTX40_MFG)

#include "AmpereMfgLoader.h"

#include <Config.h>
#include <Util.h>
#include <misc/IdentifyGpu.h>
#include <framegen/dlssg/MfgUnlock.h>
#include <nvapi.h>
#include <State.h>

#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

// The classification table in the header is NVAPI's names as numbers. A driver that reports an id this file
// misremembers would silently reclassify a card, so the mapping is checked at compile time against nvapi.h
// itself rather than trusted to a comment.
static_assert(AmpereMfgLoader::kVendorNvidia == static_cast<int>(VendorId::Nvidia), "vendor id must be NVIDIA's");
static_assert(AmpereMfgLoader::kArchTuring == NV_GPU_ARCHITECTURE_TU100, "Turing arch id");
static_assert(AmpereMfgLoader::kArchAmpere == NV_GPU_ARCHITECTURE_GA100, "Ampere arch id");
static_assert(AmpereMfgLoader::kArchAda == NV_GPU_ARCHITECTURE_AD100, "Ada arch id");
static_assert(AmpereMfgLoader::kArchBlackwell == NV_GPU_ARCHITECTURE_GB200, "Blackwell arch id");
static_assert(AmpereMfgLoader::kImplTu102 == NV_GPU_ARCH_IMPLEMENTATION_TU102, "TU102 implementation id");
static_assert(AmpereMfgLoader::kImplTu104 == NV_GPU_ARCH_IMPLEMENTATION_TU104, "TU104 implementation id");
static_assert(AmpereMfgLoader::kImplTu106 == NV_GPU_ARCH_IMPLEMENTATION_TU106, "TU106 implementation id");
static_assert(AmpereMfgLoader::kImplTu116 == NV_GPU_ARCH_IMPLEMENTATION_TU116, "TU116 implementation id");
static_assert(AmpereMfgLoader::kImplTu117 == NV_GPU_ARCH_IMPLEMENTATION_TU117, "TU117 implementation id");
static_assert(AmpereMfgLoader::kImplGa100 == NV_GPU_ARCH_IMPLEMENTATION_GA100, "GA100 implementation id");

namespace AmpereMfgLoader
{
namespace
{
std::recursive_mutex g_mutex;
Status g_status {};
bool g_attempted = false;
int g_loadAttempts = 0;         // written under g_mutex; see LoadAttempts()
constexpr int kStandbyRole = 2; // the payload's own DlssgProxy_Role value for a standby proxy

// The physical adapter, as IdentifyGpu reports it: the DXGI enumeration with the spoofed description skipped,
// the architecture id read from NVAPI on the physical GPU behind the adapter's LUID. Nothing here looks at the
// setting or the menu, and nothing here can qualify a card by name - a spoof cannot make this Ineligible host
// eligible (it would have to make the driver report another architecture, which is a different programme
// altogether).
AdapterFacts HostAdapterFacts()
{
    AdapterFacts adapter;

    for (const auto& gpu : ::IdentifyGpu::getAllGpus())
    {
        if (!gpu.softwareAdapter)
            ++adapter.PhysicalAdapterCount;
    }

    const auto& primary = ::IdentifyGpu::getPrimaryGpu();

    adapter.VendorId = static_cast<int>(primary.vendorId);
    adapter.ArchitectureId = static_cast<int>(primary.nvidiaArchInfo.architecture_id);
    adapter.ImplementationId = static_cast<int>(primary.nvidiaArchInfo.implementation);
    adapter.Name = primary.name;

    // Correlation. One physical adapter is correlated by construction. With more than one, this loader has no
    // device LUID of its own at the arming boundary and the preferred adapter is not necessarily the one the game
    // draws on - a mixed host is a named Ineligible rather than a guess, and an enumeration that reports nothing
    // is refused the same way (C1).
    //
    // The LUID question, answered for good (todo 6): the arming boundary is StreamlineHooks::hkslInit, and
    // sl::Preferences carries no adapter or device handle (external/streamline/sl_core_types.h: Preferences has
    // renderAPI/engine/pathsToPlugins only) - Streamline 2.x makes its capability decision BEFORE a D3D device
    // exists, so no render LUID is available at the arm site. Passing one is therefore not "trivially
    // available": mixed-GPU hosts stay Ineligible, and the receipts say so instead of guessing.
    adapter.CorrelatedToProcessDevice = adapter.PhysicalAdapterCount == 1;

    return adapter;
}

// Who owns frame generation right now, read from the process.
OwnershipFacts HostOwnershipFacts()
{
    OwnershipFacts ownership;
    auto* config = ::Config::Instance();

    const auto& state = ::State::Instance();
    ownership.ExternalFrameGeneration =
        config->ExternalFrameGeneration.value_or_default() && state.activeFgInput == ::FGInput::NoFG &&
        state.activeFgOutput == ::FGOutput::NoFG && state.activeFgNvngx == ::FGNvngxReplacement::None;

    // "Active" is the session-latched Ada unlock being on, read the same way MfgUnlock::EnabledForSession
    // reads it. It is a conflict because the DLSSG output has exactly one owner and the Ada route is the other
    // claimant for it: upstream's own UI locks the Ampere checkbox while the Ada unlock is on, so a machine
    // with both ticks is a configuration the loader refuses and leaves alone rather than choosing for the user.
    // Whether the Ada patcher would engage here is deliberately not the test - on a 20/30 part it would not,
    // and a ticked box that could still be turned on is not what "one owner only" means.
    ownership.AdaUnlockActive = ::MfgUnlock::EnabledForSession();

    // The payload under its shipped name, already in this process without this loader having put it there: the
    // distribution also ships proxy-named copies (vendor/dlssg_sm86/PIN.json, alternatives/), so whoever loaded
    // it owns the DLSSG output.
    ownership.OtherOwnerPresent = GetModuleHandleW(kPayloadModuleName) != nullptr;

    // A native DLSSG output owns the route even without an nvngx replacement.
    ownership.OptiScalerOwnedDlssgOutput = state.activeFgOutput == ::FGOutput::DLSSG;

    return ownership;
}

// The settings the restored config surface carries, read once. Arm() latches the result for the process, so this
// only runs on the first call: a change in the INI takes effect on the next launch (C4).
ArmOptions HostOptions()
{
    auto* config = ::Config::Instance();

    ArmOptions options;
    options.Enabled = config->FGDLSSGAmpereMfgUnlock.value_or_default();
    options.Ini.MaxGeneratedFrames = config->FGDLSSGAmpereMfgMaxFrames.value_or_default();
    options.Ini.KernelImage = config->FGDLSSGAmpereMfgKernelImage.value_for_config_or("Auto");

    // Router/Optimized/Mode/LogLevel keep the loader's defaults: 0.3.5 resolves the kernel family from the
    // physical GPU itself. PackageRoot stays empty in production - the OptiScaler.dll directory.
    return options;
}

// The gate's refusal as the pipeline state the receipts quote.
constexpr State RefusalState(Verdict verdict)
{
    switch (verdict)
    {
    case Verdict::Ineligible:
        return State::Ineligible;
    case Verdict::Conflict:
        return State::Conflict;
    case Verdict::Disabled:
    case Verdict::Eligible:
        break; // Eligible does the file work instead of stopping
    }

    return State::Disabled;
}

// A module that is not empty and starts with an MZ header: enough to tell "the payload is deployed" from "a
// stray file with the right name". This is the first half of the ladder's payload check, and it is deliberately
// weak: the second half is CheckPinnedPayload below, which answers the different question PIN.json asks - is
// this the payload this build ships, byte for byte. On success `detail` carries the size for the receipt.
bool LooksLikeModule(const std::filesystem::path& module, std::string& detail)
{
    std::error_code error;
    const auto bytes = std::filesystem::file_size(module, error);

    if (error || bytes < 2)
    {
        detail = "the file is smaller than a PE header";
        return false;
    }

    std::ifstream file(module, std::ios::binary);
    char magic[2] {};

    if (!file.read(magic, 2) || magic[0] != 'M' || magic[1] != 'Z')
    {
        detail = "the file does not start with an MZ header";
        return false;
    }

    detail = std::to_string(bytes) + " bytes";
    return true;
}

// The SHA-256 of a file, lowercase hex, streamed: the pinned payload is 30 MB and the arming path must not
// hold a copy of it in memory. False means the file could not be read or the provider failed; the caller turns
// that into the same named refusal a wrong size gets, so an unreadable module can never fall through to the
// load.
bool Sha256File(const std::filesystem::path& file, std::string& digest)
{
    std::ifstream stream(file, std::ios::binary);

    if (!stream.is_open())
        return false;

    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;

    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        return false;

    if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0)
    {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        return false;
    }

    unsigned char buffer[64 * 1024] {};
    bool ok = true;

    while (stream.read(reinterpret_cast<char*>(buffer), sizeof(buffer)) || stream.gcount() > 0)
    {
        if (BCryptHashData(hash, buffer, static_cast<ULONG>(stream.gcount()), 0) < 0)
        {
            ok = false;
            break;
        }
    }

    unsigned char raw[32] {};
    ok = ok && !stream.bad() && BCryptFinishHash(hash, raw, sizeof(raw), 0) >= 0;

    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);

    if (!ok)
        return false;

    static constexpr char kHex[] = "0123456789abcdef";
    digest.clear();
    digest.reserve(sizeof(raw) * 2);

    for (const unsigned char byte : raw)
    {
        digest.push_back(kHex[byte >> 4]);
        digest.push_back(kHex[byte & 0x0F]);
    }

    return true;
}

// The pin rung (registered from todo 11's GAP-SEAM-1). LooksLikeModule answers "is this a plausible PE
// image"; this answers the other question vendor/dlssg_sm86/PIN.json asks - is it THIS payload, byte for
// byte. It runs before the companion INI is written and before LoadLibrary, so a module that is not the pinned
// payload leaves no INI behind and is never mapped. Size first, digest second: a module that is not even the
// pinned size is an incomplete deployment of this payload (PayloadIncomplete, the status the other size
// failures already use), and a module of exactly the pinned size whose digest differs is the named
// PayloadDigestMismatch. The expected size and digest are the header's compile-time constants; nothing here
// reads the manifest at runtime.
struct PinCheck
{
    bool matched = false;
    State refusal = State::PayloadIncomplete; // only read when the module did not match
    std::string detail;                       // what was found, or why the refusal fired
};

PinCheck CheckPinnedPayload(const std::filesystem::path& module)
{
    PinCheck check;
    const unsigned long long expectedBytes = ExpectedPayloadBytes();
    std::error_code error;
    const auto bytes = std::filesystem::file_size(module, error);

    if (error)
    {
        check.detail = "the module's size could not be read: " + error.message();
        return check;
    }

    if (bytes != expectedBytes)
    {
        check.refusal = State::PayloadIncomplete;
        check.detail = "the module is " + std::to_string(bytes) + " bytes and the pinned payload is " +
                       std::to_string(expectedBytes) + " bytes";
        return check;
    }

    std::string digest;

    if (!Sha256File(module, digest))
    {
        check.refusal = State::PayloadIncomplete;
        check.detail = "the module's sha256 could not be computed";
        return check;
    }

    const std::string expectedDigest = ExpectedPayloadSha256();

    if (digest != expectedDigest)
    {
        check.refusal = State::PayloadDigestMismatch;
        check.detail = "the module is the pinned " + std::to_string(bytes) + " bytes but its sha256 is " + digest +
                       " and the pinned digest is " + expectedDigest;
        return check;
    }

    check.matched = true;
    check.detail = std::to_string(bytes) + " bytes, sha256 " + digest + " (pinned)";
    return check;
}

std::string Narrow(const std::wstring& text) { return wstring_to_string(text); }

std::string Tried(const std::vector<std::filesystem::path>& paths)
{
    std::string text;

    for (const auto& path : paths)
        text += (text.empty() ? "" : ", ") + Narrow(path.wstring());

    return text;
}

bool WriteCompanionIni(const std::filesystem::path& iniPath, const std::string& content, std::string& why)
{
    std::ofstream file(iniPath, std::ios::binary | std::ios::trunc);

    if (!file.is_open())
    {
        why = "cannot open " + Narrow(iniPath.wstring()) + " for writing";
        return false;
    }

    file << content;
    file.close();

    if (!file)
    {
        why = "could not finish writing " + Narrow(iniPath.wstring());
        return false;
    }

    return true;
}

// The payload's own proxy identity, read from the module that was just loaded. The 0.3.5 proxy exposes
// DlssgProxy_Name (the name it took) and DlssgProxy_Role (1 = active: it installed its LoadLibrary hook, read
// the INI and arms the backend; 2 = standby: another proxy of the family is already active, so this one only
// forwards its exports - the contract calls that a named refusal, never "installed"). A module without the
// exports is not a proxy of this family; it is reported as unknown and left at Loaded.
struct ModuleIdentity
{
    bool hasRole = false;
    int role = -1;
    std::string name;
};

ModuleIdentity ReadModuleIdentity(HMODULE module)
{
    ModuleIdentity identity;

    using NameFn = const wchar_t*(__cdecl*) (void);
    using RoleFn = int(__cdecl*)(void);

    if (auto nameFn = reinterpret_cast<NameFn>(::GetProcAddress(module, "DlssgProxy_Name")))
    {
        const wchar_t* name = nameFn();
        if (name != nullptr)
            identity.name = wstring_to_string(name);
    }

    if (auto roleFn = reinterpret_cast<RoleFn>(::GetProcAddress(module, "DlssgProxy_Role")))
    {
        identity.hasRole = true;
        identity.role = roleFn();
    }

    return identity;
}
} // namespace

Status LastStatus()
{
    std::lock_guard lock(g_mutex);
    return g_status;
}

int LoadAttempts()
{
    std::lock_guard lock(g_mutex);
    return g_loadAttempts;
}

Status Arm()
{
    // The latch is checked before the host reads: the Streamline init path calls this once per process, and a
    // repeated call (the host can call slInit more than once) must not re-read the host or re-do any file work.
    {
        std::lock_guard lock(g_mutex);

        if (g_attempted)
            return g_status;
    }

    const ArmOptions options = HostOptions();

    // The unlock is off by default. With it off nothing is read from the host at all: the GPU discovery stays in
    // the existing getGpuInfo worker, and the arm site only consumes its cached result when the feature is on -
    // so the disabled path cannot trigger an adapter enumeration on the game's slInit thread either.
    if (!options.Enabled)
        return Arm(options, AdapterFacts {}, OwnershipFacts {});

    return Arm(options, HostAdapterFacts(), HostOwnershipFacts());
}

Status Arm(const ArmOptions& options)
{
    const AdapterFacts adapter = HostAdapterFacts();
    return Arm(options, adapter, HostOwnershipFacts());
}

Status Arm(const ArmOptions& options, const AdapterFacts& adapter, const OwnershipFacts& ownership)
{
    std::lock_guard lock(g_mutex);

    // One pass per process (C4). A setting change needs a restart, so a later call returns the latched result
    // instead of re-reading and re-writing anything.
    if (g_attempted)
    {
        LOG_INFO("AmpereMfgLoader: Arm() already ran this process; keeping {} ({})", StateName(g_status.state),
                 g_status.Detail);
        return g_status;
    }

    g_attempted = true;
    g_status = {};

    // The gate decides everything that is not file work. A refusal latches and returns without touching the
    // payload folder, the companion INI or the load-attempt count; a plain External=true is not a refusal.
    const Decision decision = Evaluate(options, adapter, ownership);

    if (decision.verdict != Verdict::Eligible)
    {
        g_status.state = RefusalState(decision.verdict);
        g_status.Detail = decision.Detail;

        if (decision.verdict == Verdict::Conflict)
            LOG_WARN("AmpereMfgLoader: {}: {}", VerdictName(decision.verdict), g_status.Detail);
        else
            LOG_INFO("AmpereMfgLoader: {}: {}", VerdictName(decision.verdict), g_status.Detail);

        return g_status;
    }

    // Eligible. This is the load attempt: counted before the file work, so a payload that turns out to be
    // missing is still an attempt, and the refusals above can never reach this line.
    ++g_loadAttempts;

    const std::filesystem::path root =
        options.PackageRoot.empty() ? std::filesystem::path(Util::DllPath()).parent_path() : options.PackageRoot;

    std::vector<std::filesystem::path> tried;
    const std::filesystem::path modulePath = FindPayloadModule(root, tried);

    if (modulePath.empty())
    {
        g_status.state = State::PayloadMissing;
        g_status.Detail = "the payload module " + Narrow(kPayloadModuleName) + " was not found under " +
                          Narrow(root.wstring()) + " (checked: " + Tried(tried) + ")";
        LOG_ERROR("AmpereMfgLoader: {}", g_status.Detail);
        return g_status;
    }

    std::string detail;

    if (!LooksLikeModule(modulePath, detail))
    {
        g_status.state = State::PayloadIncomplete;
        g_status.Detail = Narrow(modulePath.wstring()) + " is not a usable module: " + detail;
        LOG_ERROR("AmpereMfgLoader: {}", g_status.Detail);
        return g_status;
    }

    // The pin rung, before the companion INI and before the load: a structurally valid PE is not the payload
    // this build ships. A refusal here leaves no INI behind and never reaches LoadLibrary, which is what the
    // failure matrix asserts on the wrong-size and wrong-digest fixtures.
    const PinCheck pin = CheckPinnedPayload(modulePath);

    if (!pin.matched)
    {
        g_status.state = pin.refusal;
        g_status.Detail = Narrow(modulePath.wstring()) + " is not the pinned payload: " + pin.detail +
                          " (vendor/dlssg_sm86/PIN.json)";
        LOG_ERROR("AmpereMfgLoader: {}", g_status.Detail);
        return g_status;
    }

    // The Loaded text carries the pinned identity (size and digest), not just the byte count LooksLikeModule
    // measured.
    detail = pin.detail;

    // The companion INI goes next to the module - the payload reads it from its own directory, and the name is
    // fixed. Clamped content only: FormatIniContent emits nothing the payload would have to reject.
    const std::filesystem::path iniPath = modulePath.parent_path() / kPayloadIniName;
    const std::string ini = FormatIniContent(options.Ini);

    if (!WriteCompanionIni(iniPath, ini, detail))
    {
        g_status.state = State::IniWriteFailed;
        g_status.Detail = detail;
        LOG_ERROR("AmpereMfgLoader: {}", g_status.Detail);
        return g_status;
    }

    // The load. Explicit absolute path, never name resolution: the payload is a proxy that forwards to the
    // system DLL of its own name, so what gets loaded must be the file this loader found.
    const HMODULE module = ::LoadLibraryW(modulePath.wstring().c_str());

    if (module == nullptr)
    {
        const unsigned long error = ::GetLastError();

        g_status.state = State::PayloadLoadFailed;
        g_status.ModulePath = modulePath.wstring();
        g_status.IniPath = iniPath.wstring();
        g_status.Detail =
            "LoadLibraryW failed for " + Narrow(modulePath.wstring()) + " with error " + std::to_string(error);
        LOG_ERROR("AmpereMfgLoader: {}", g_status.Detail);
        return g_status;
    }

    const ModuleIdentity identity = ReadModuleIdentity(module);
    const std::string roleText = identity.hasRole ? std::to_string(identity.role) : std::string("n/a");
    const std::string nameText = identity.name.empty() ? std::string("unnamed") : identity.name;

    g_status.ModulePath = modulePath.wstring();
    g_status.IniPath = iniPath.wstring();

    // Loaded but standby is a named refusal: the module forwards its exports and installs nothing, so treating
    // it as armed would claim a backend that is not there (C5, payload contract "several proxies at once").
    if (identity.hasRole && identity.role == kStandbyRole)
    {
        g_status.state = State::PayloadStandby;
        g_status.Detail = "the payload loaded as a standby proxy (DlssgProxy_Role=2, name '" + nameText +
                          "'): another proxy of the family is already active in this process, so this one "
                          "installs nothing";
        LOG_ERROR("AmpereMfgLoader: {}", g_status.Detail);
        return g_status;
    }

    g_status.state = State::Loaded;
    g_status.Detail = "payload loaded from " + Narrow(modulePath.wstring()) + " (" + detail + ", proxy name '" +
                      nameText + "', role " + roleText +
                      ") and the companion INI applied; the backend's own install record "
                      "(backend_install.status=0, install.active=true) is written by the payload, not by "
                      "LoadLibrary, so no backend state is claimed here (C5)";
    LOG_INFO("AmpereMfgLoader: {} -> {} ({})", StateName(g_status.state), g_status.Detail, ini.size());

    return g_status;
}
} // namespace AmpereMfgLoader

#endif
