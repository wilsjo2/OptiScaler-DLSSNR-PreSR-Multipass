// Adapted from y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG, tag v4 (7b7220bb), GPL-3.0.
#include "pch.h"

#if defined(OPTISCALER_RTX40_MFG)

#include "MfgUnlock.h"

#include <Config.h>
#include <Util.h>
#include <scanner/scanner.h>
#include <misc/IdentifyGpu.h>

#include "MfgUnlockFlip.h"
#include "MfgUnlockPlugin.h"
#include "MfgUnlockPtx.h"

#include <mutex>
#include <tlhelp32.h>

namespace
{
// mov ebx,1 / mov r8d,3 / cmp edi,0x1b0 / cmovl r8d,ebx. The two counts and the architecture
// constant together identify the legacy capability gate.
constexpr std::string_view kAdvertisePattern = "BB 01 00 00 00 41 B8 03 00 00 00 81 FF B0 01 00 00 44 0F 4C C3";

// cmp eax,0x1b0 / jl / cmp ebx,3 / jbe. The only comparison against the architecture constant that
// is followed by a signed branch and a count test.
constexpr std::string_view kValidatePattern = "3D B0 01 00 00 7C ? 83 FB 03 76";

// Five generated frames, the count both patched sites carry.
constexpr uint8_t kMaxGeneratedFrames = 5;

// 310.9 restructured both gates. The count is no longer an immediate next to the comparison: the
// Blackwell branch starts at five and reads a configured value, and anything below Blackwell is sent
// to a branch that publishes one.
//     cmp ebp, 0x1b0
//     jl  ada          <- neutralised, so every card takes the Blackwell branch
//     mov edi, 0x5
constexpr std::string_view kAdvertisePattern309 = "81 FD B0 01 00 00 0F 8C ? ? ? ? BF 05 00 00 00";

// The capability flag in the same build is a setae rather than a branch.
//     cmp   eax, 0x1b0
//     setae al
constexpr std::string_view kValidatePattern309 = "3D B0 01 00 00 0F 93 C0";

MfgUnlock::Status g_status {};
std::recursive_mutex g_mutex;

uintptr_t UniqueAddress(HMODULE module, std::string_view pattern)
{
    const auto first = scanner::GetAddress(module, pattern);
    return first && !scanner::GetAddress(module, pattern, 0, first + 1) ? first : 0;
}

// The module's own file version, for the report. A signature that does not match is expected on a
// version nobody has looked at, and the version is the one thing that makes such a report actionable.
std::string ModuleVersion(HMODULE module)
{
    wchar_t path[MAX_PATH] {};

    if (GetModuleFileNameW(module, path, MAX_PATH) == 0)
        return {};

    version_t file {};
    version_t product {};

    if (!Util::GetFileVersion(path, &file, &product))
        return {};

    return std::format("{}.{}.{}", file.major, file.minor, file.patch);
}

// The DLSS-G provider, if it is mapped. The file name finds the game's own copy. The driver's OTA copy
// (models\dlssg\versions\<n>\files\<hash>.bin) and a renamed snippet are found by walking the loaded
// modules, which is only safe from an ordinary thread: a snapshot taken under the loader lock
// deadlocks. The load hook hands TryApply its module directly and never comes here.
//
// TryApply runs on every Streamline call until the snippet is found, so the walk is rate limited. A
// provider that has not appeared after these few walks is not going to appear through this route.
HMODULE FindProvider()
{
    if (auto module = GetModuleHandleW(L"nvngx_dlssg.dll"))
        return module;

    static uint64_t nextWalk = 0;
    static unsigned walks = 0;
    constexpr unsigned kMaxWalks = 8;
    constexpr uint64_t kWalkIntervalMs = 2000;

    const uint64_t now = GetTickCount64();

    if (walks >= kMaxWalks || now < nextWalk)
        return nullptr;

    nextWalk = now + kWalkIntervalMs;
    ++walks;

    // The marker string is a literal in this DLL, so it has to be excluded from the walk.
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&FindProvider), &self);

    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());

    if (snapshot == INVALID_HANDLE_VALUE)
        return nullptr;

    HMODULE found = nullptr;
    MODULEENTRY32W entry {};
    entry.dwSize = sizeof(entry);

    for (bool more = Module32FirstW(snapshot, &entry); more && found == nullptr; more = Module32NextW(snapshot, &entry))
    {
        if (entry.hModule == self)
            continue;

        if (MfgUnlock::Provider::IsProviderPath(entry.szExePath))
        {
            found = entry.hModule;
            continue;
        }

        // A renamed or relocated snippet: only look inside modules that expose an NGX entry point.
        if (GetProcAddress(entry.hModule, "NVSDK_NGX_D3D12_PopulateDeviceParameters_Impl") == nullptr &&
            GetProcAddress(entry.hModule, "NVSDK_NGX_VULKAN_PopulateDeviceParameters_Impl") == nullptr)
            continue;

        if (MfgUnlock::Provider::ImageContains(entry.hModule, MfgUnlock::Provider::kMarker))
            found = entry.hModule;
    }

    CloseHandle(snapshot);

    return found;
}

// The Streamline DLSS-G plugins seen so far, and the ones already tried. A plugin is tried once: the
// answer does not change, and a repeat would only repeat the log line.
// Status, plugin discovery and patch application share g_mutex.
std::vector<HMODULE> g_plugins;
std::vector<HMODULE> g_pluginsTried;

// Same guards as TryApply: the option on for this session, an Ada GPU.
bool AdaUnlockWanted()
{
    if (!MfgUnlock::EnabledForSession())
        return false;

    const auto& gpu = IdentifyGpu::getPrimaryGpu();
    return gpu.vendorId == VendorId::Nvidia && gpu.nvidiaArchInfo.architecture_id == NV_GPU_ARCHITECTURE_AD100;
}

// Software frame pacing: pin the plugin's flip-metering state to its own software fallback. Applied when
// the plugin is loaded, before Streamline uses it, because rewriting a register store is a seven byte
// write into code that no other thread should be executing yet. Caller holds g_mutex.
void PatchFlipMetering(HMODULE plugin)
{
    g_status.FlipRequested = Config::Instance()->FGDLSSGAdaFlipMeteringPatch.value_or_default();

    if (std::string_view(g_status.FlipMetering) == "patched")
        return;

    // A plugin has been seen; "off" tells the overlay that, and that nothing was asked of it.
    if (!g_status.FlipRequested)
    {
        g_status.FlipMetering = "off";
        return;
    }

    wchar_t path[MAX_PATH] {};
    GetModuleFileNameW(plugin, path, MAX_PATH);
    const auto pluginPath = wstring_to_string(path);

    MfgUnlock::Flip::Plan plan;
    const auto found = MfgUnlock::Flip::FindPlan(plugin, plan);

    if (found != MfgUnlock::Flip::FindResult::Found)
    {
        g_status.FlipMetering = MfgUnlock::Flip::Describe(found);
        LOG_WARN("MFG unlock: {}: software frame pacing not applied: {}", pluginPath, g_status.FlipMetering);
        return;
    }

    switch (MfgUnlock::Flip::Apply(plan))
    {
    case MfgUnlock::Flip::ApplyResult::Patched:
        g_status.FlipMetering = "patched";
        g_status.FlipSites = static_cast<unsigned int>(plan.sites.size());
        LOG_INFO("MFG unlock: {}: flip-metering state +0x{:X} pinned to {} at {} site(s); multi-frame should pace in "
                 "software",
                 pluginPath, plan.field, plan.value, plan.sites.size());
        break;
    case MfgUnlock::Flip::ApplyResult::Mismatch:
        g_status.FlipMetering = "the plugin changed while it was being patched";
        LOG_WARN("MFG unlock: {}: software frame pacing not applied: {}", pluginPath, g_status.FlipMetering);
        break;
    case MfgUnlock::Flip::ApplyResult::ProtectFailed:
        g_status.FlipMetering = "its memory could not be made writable";
        LOG_WARN("MFG unlock: {}: software frame pacing not applied: {}", pluginPath, g_status.FlipMetering);
        break;
    }
}

// Only once the snippet unlock has landed. Raising the plugin's ceiling while the snippet still answers
// Ada's 1 would only move the rejection from the plugin to the snippet.
void PatchPluginCeilings()
{
    std::lock_guard lock(g_mutex);
    if (MfgUnlock::UnlockedMax() == 0)
        return;

    for (HMODULE plugin : g_plugins)
    {
        if (std::find(g_pluginsTried.begin(), g_pluginsTried.end(), plugin) != g_pluginsTried.end())
            continue;

        g_pluginsTried.push_back(plugin);

        wchar_t path[MAX_PATH] {};
        GetModuleFileNameW(plugin, path, MAX_PATH);
        const auto pluginPath = wstring_to_string(path);

        // A plugin that was patched stays patched; a second one that cannot be does not change that.
        const bool alreadyPatched = std::string_view(g_status.PluginCeiling) == "patched";

        MfgUnlock::Plugin::CeilingSite site;
        const char* result = "not matched";

        switch (MfgUnlock::Plugin::FindCeilingSite(plugin, site))
        {
        case MfgUnlock::Plugin::FindResult::Found:
            switch (MfgUnlock::Plugin::ApplyCeilingPatch(site))
            {
            case MfgUnlock::Plugin::ApplyResult::Patched:
                result = "patched";
                LOG_INFO("MFG unlock: {}: frame-count clamp neutralised, compiled maximum {} generated frame(s)",
                         pluginPath, site.compiled);
                break;
            case MfgUnlock::Plugin::ApplyResult::ProtectFailed:
                result = "not writable";
                LOG_WARN("MFG unlock: {}: frame-count clamp found but its page could not be made writable", pluginPath);
                break;
            case MfgUnlock::Plugin::ApplyResult::Mismatch:
                LOG_WARN("MFG unlock: {}: frame-count clamp changed under us; left unchanged", pluginPath);
                break;
            }
            break;
        case MfgUnlock::Plugin::FindResult::Ambiguous:
            result = "ambiguous";
            LOG_WARN("MFG unlock: {}: more than one frame-count clamp; left unchanged", pluginPath);
            break;
        case MfgUnlock::Plugin::FindResult::None:
        case MfgUnlock::Plugin::FindResult::BadImage:
            LOG_WARN("MFG unlock: {}: no frame-count clamp of the known shape; left unchanged", pluginPath);
            break;
        }

        if (!alreadyPatched)
            g_status.PluginCeiling = result;
    }
}

bool WriteBytes(uintptr_t address, const uint8_t* bytes, size_t count)
{
    DWORD oldProtect = 0;

    if (!VirtualProtect((LPVOID) address, count, PAGE_EXECUTE_READWRITE, &oldProtect))
    {
        LOG_WARN("VirtualProtect failed at {:X}", address);
        return false;
    }

    std::memcpy((void*) address, bytes, count);

    DWORD ignored = 0;
    VirtualProtect((LPVOID) address, count, oldProtect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), (LPCVOID) address, count);

    return true;
}

std::string Hex(const uint8_t* bytes, size_t count)
{
    std::string out;

    for (size_t i = 0; i < count; ++i)
        out += std::format("{}{:02X}", i == 0 ? "" : " ", bytes[i]);

    return out;
}

// Rewrites count and neutralises the architecture clamp, so MultiFrameCountMax is published as five.
bool PatchAdvertise(HMODULE module)
{
    if (const auto at309 = UniqueAddress(module, kAdvertisePattern309); at309 != 0)
    {
        // The jl is a rel32, six bytes.
        const auto branchAt = at309 + 6;
        const uint8_t nop[] = { 0x0F, 0x1F, 0x44, 0x00, 0x00, 0x90 };

        LOG_INFO("MFG unlock: advertise (310.9) at {:X}, jl {} -> {}", at309,
                 Hex((const uint8_t*) branchAt, sizeof(nop)), Hex(nop, sizeof(nop)));

        return WriteBytes(branchAt, nop, sizeof(nop));
    }

    const auto address = UniqueAddress(module, kAdvertisePattern);

    if (address == 0)
    {
        LOG_WARN("MFG unlock: the advertise signature did not match, nvngx_dlssg.dll left alone");
        return false;
    }

    // Offsets within the matched sequence: the r8d immediate, and the cmovl.
    const auto countAt = address + 7;
    const auto cmovAt = address + 17;

    const uint8_t count[] = { kMaxGeneratedFrames };
    const uint8_t nop[] = { 0x0F, 0x1F, 0x40, 0x00 };

    LOG_INFO("MFG unlock: advertise at {:X}, count {} -> {}, cmovl {} -> {}", address, *(const uint8_t*) countAt,
             kMaxGeneratedFrames, Hex((const uint8_t*) cmovAt, sizeof(nop)), Hex(nop, sizeof(nop)));

    return WriteBytes(countAt, count, sizeof(count)) && WriteBytes(cmovAt, nop, sizeof(nop));
}

// Drops the Ada branch and raises the accepted count, so a request for five is not rejected.
bool PatchValidate(HMODULE module)
{
    if (const auto at309 = UniqueAddress(module, kValidatePattern309); at309 != 0)
    {
        // setae al -> mov al, 1, so the flag is set whatever the architecture reports.
        const auto setAt = at309 + 5;
        const uint8_t always[] = { 0xB0, 0x01, 0x90 };

        LOG_INFO("MFG unlock: validate (310.9) at {:X}, setae {} -> {}", at309,
                 Hex((const uint8_t*) setAt, sizeof(always)), Hex(always, sizeof(always)));

        return WriteBytes(setAt, always, sizeof(always));
    }

    const auto address = UniqueAddress(module, kValidatePattern);

    if (address == 0)
    {
        LOG_WARN("MFG unlock: the validate signature did not match, nvngx_dlssg.dll left alone");
        return false;
    }

    // Offsets within the matched sequence: the jl, and the immediate of the count test behind it.
    const auto branchAt = address + 5;
    const auto countAt = address + 9;

    const uint8_t nop[] = { 0x90, 0x90 };
    const uint8_t count[] = { kMaxGeneratedFrames };

    LOG_INFO("MFG unlock: validate at {:X}, jl {} -> {}, count {} -> {}", address,
             Hex((const uint8_t*) branchAt, sizeof(nop)), Hex(nop, sizeof(nop)), *(const uint8_t*) countAt,
             kMaxGeneratedFrames);

    return WriteBytes(branchAt, nop, sizeof(nop)) && WriteBytes(countAt, count, sizeof(count));
}

// Gives Ada the Blackwell kernels the module already carries.
//
// nvngx_dlssg.dll ships two builds of the interpolation kernels. Kernel_EstimateIntermMvecsScatter
// reads three f32 fields of its parameter block on sm_120 and one on sm_89, so on Ada every generated
// frame is placed at the same point between the two real ones: the world does not advance between
// them while the interface, composited once per present, does. At 2X there is one frame and nothing
// to distinguish; above it that is the whole symptom.
//
// The sm_120 module uses no instruction Ada lacks. So per container: the Blackwell PTX image is
// relabelled sm_89, its .target directive is rewritten in place (".target sm_120" and
// ".target sm_89 " are both fourteen bytes, and the directive sits in the literal run at the head of
// the LZ4 stream), and the images that were sm_89 -- the Ada PTX and its SASS -- are relabelled to an
// architecture that does not exist so the driver cannot select them. The driver then JITs Blackwell's
// kernel when it asks for Ada's.
//
// Nothing is copied in and no payload changes length. A container without both images is left alone.
constexpr uint32_t kArchAda = 89;
constexpr uint32_t kArchBlackwell = 120;

// No such shader model. Parks an image where nothing will ask for it.
constexpr uint32_t kArchParked = 122;

// Offsets inside a fatbin image header: payload length, and the architecture the image answers for.
constexpr size_t kImagePayloadSize = 8;
constexpr size_t kImageArch = 28;

unsigned int RewriteBlackwellKernels(HMODULE module)
{
    auto base = reinterpret_cast<uint8_t*>(module);
    auto dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    auto section = IMAGE_FIRST_SECTION(nt);

    const uint8_t magic[] = { 0x50, 0xED, 0x55, 0xBA };
    unsigned int rewritten = 0;

    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i)
    {
        const auto& s = section[i];

        if (s.Characteristics & IMAGE_SCN_MEM_EXECUTE)
            continue;

        uint8_t* start = base + s.VirtualAddress;
        uint8_t* end = start + s.Misc.VirtualSize;

        for (uint8_t* c = std::search(start, end, magic, magic + sizeof(magic)); c < end;
             c = std::search(c + 1, end, magic, magic + sizeof(magic)))
        {
            if (end - c < 16)
                break;

            const auto headerSize = *reinterpret_cast<const uint16_t*>(c + 6);
            const auto fatSize = *reinterpret_cast<const uint64_t*>(c + 8);

            if (headerSize != 0x10 || fatSize == 0 || fatSize > (uint64_t) (end - c - 16))
                continue;

            uint8_t* blackwell = nullptr;
            size_t blackwellHeader = 0;
            size_t blackwellPayload = 0;
            std::vector<uint8_t*> ada;
            bool valid = true;

            for (uint8_t* image = c + 16; image < c + 16 + fatSize;)
            {
                const auto remaining = (uint64_t) (c + 16 + fatSize - image);
                if (remaining < kImageArch + sizeof(uint32_t))
                {
                    valid = false;
                    break;
                }
                const auto kind = *reinterpret_cast<const uint16_t*>(image);
                const auto imageHeader = *reinterpret_cast<const uint32_t*>(image + 4);
                const auto payload = *reinterpret_cast<const uint64_t*>(image + kImagePayloadSize);
                const auto arch = *reinterpret_cast<const uint32_t*>(image + kImageArch);

                if (imageHeader < kImageArch + sizeof(uint32_t) || imageHeader > remaining || payload == 0 ||
                    payload > remaining - imageHeader)
                {
                    valid = false;
                    break;
                }

                // kind 1 is PTX, 2 is a cubin. Only the PTX can be retargeted; the cubin is parked.
                if (kind == 1 && arch == kArchBlackwell)
                {
                    blackwell = image;
                    blackwellHeader = imageHeader;
                    blackwellPayload = payload;
                }
                else if (arch == kArchAda)
                {
                    ada.push_back(image);
                }

                image += imageHeader + payload;
            }

            if (!valid || blackwell == nullptr || ada.empty())
                continue;

            const char from[] = ".target sm_120";
            const char to[] = ".target sm_89 ";
            static_assert(sizeof(from) == sizeof(to), "the directive rewrite must not change length");

            uint8_t* body = blackwell + blackwellHeader;
            uint8_t* bodyEnd = body + blackwellPayload;
            auto at = std::search(body, bodyEnd, from, from + sizeof(from) - 1);

            if (at == bodyEnd)
                continue;

            const uint32_t ada89 = kArchAda;
            const uint32_t parked = kArchParked;
            // Prepare one complete container first. A failed protection change must not leave
            // its PTX target and architecture headers disagreeing, or count a partial rewrite.
            std::vector<uint8_t> patched(c, c + 16 + fatSize);
            std::memcpy(patched.data() + (at - c), to, sizeof(to) - 1);
            std::memcpy(patched.data() + (blackwell + kImageArch - c), &ada89, sizeof(ada89));
            for (uint8_t* image : ada)
                std::memcpy(patched.data() + (image + kImageArch - c), &parked, sizeof(parked));
            if (WriteBytes(reinterpret_cast<uintptr_t>(c), patched.data(), patched.size()))
                ++rewritten;
        }
    }

    LOG_INFO("MFG unlock: {} kernel containers answer Ada with the Blackwell image", rewritten);

    return rewritten;
}
} // namespace

void MfgUnlock::TryApply(HMODULE requestedModule)
{
    if (!AdaUnlockWanted())
        return;

    std::lock_guard lock(g_mutex);

    // Latches once nvngx_dlssg.dll is present; before that every call rescans for it.
    static bool snippetDone = false;

    if (!snippetDone)
    {
        if (auto module = requestedModule ? requestedModule : FindProvider(); module != nullptr)
        {
            snippetDone = true;
            g_status.ModuleFound = true;
            g_status.SnippetVersion = ModuleVersion(module);

            wchar_t modulePath[MAX_PATH] {};
            GetModuleFileNameW(module, modulePath, MAX_PATH);
            LOG_INFO("MFG unlock: DLSS-G provider {} at {}", g_status.SnippetVersion, wstring_to_string(modulePath));

            // Validate both gates before touching either. Ambiguous/unknown versions remain unmodified.
            const bool knownGates =
                (UniqueAddress(module, kAdvertisePattern309) && UniqueAddress(module, kValidatePattern309)) ||
                (UniqueAddress(module, kAdvertisePattern) && UniqueAddress(module, kValidatePattern));
            if (!knownGates)
            {
                LOG_WARN("MFG unlock: unsupported or ambiguous DLSSG {} signatures; left unchanged",
                         g_status.SnippetVersion);
                return;
            }

            // Retargeting and both gates form one feature; a count-only unlock repeats frames. One temporal
            // method per session, since both edit the same fatbin.
            const auto method = ConfiguredTemporalMethod();
            g_status.TemporalAttempted = method;

            if (method == TemporalMethod::Retarget)
            {
                g_status.KernelsRewritten = RewriteBlackwellKernels(module);

                g_status.TemporalDetail = g_status.KernelsRewritten > 0
                                              ? "reused the Blackwell interpolation kernel"
                                              : "no compatible Blackwell interpolation kernel image";
            }
            else
            {
                Ptx::Result ptx;

                Ptx::Apply(module, ptx);
                g_status.KernelsRewritten = static_cast<unsigned int>(ptx.redirected);
                g_status.TemporalDetail = ptx.detail;

                if (ptx.redirected > 0)
                    LOG_INFO("MFG unlock: PTX temporal fix: {}", ptx.detail);
                else
                    LOG_WARN("MFG unlock: PTX temporal fix not applied: {}", ptx.detail);
            }

            if (g_status.KernelsRewritten == 0)
            {
                LOG_WARN("MFG unlock: no compatible interpolation kernels; frame-count gates left unchanged");
                return;
            }
            const bool advertise = PatchAdvertise(module);
            const bool validate = PatchValidate(module);
            g_status.AdvertiseMatched = advertise;
            g_status.ValidateMatched = validate;

            if (advertise && validate)
                LOG_INFO("MFG unlock: nvngx_dlssg.dll patched for {} generated frames", kMaxGeneratedFrames);
            else
                LOG_WARN("MFG unlock: nvngx_dlssg.dll incomplete, advertise {}, validate {}", advertise, validate);

            // A plugin that was loaded first has been waiting for this.
            PatchPluginCeilings();
        }
    }
}

namespace
{
MfgUnlock::Telemetry g_telemetry;
}

const MfgUnlock::Telemetry& MfgUnlock::GetTelemetry() { return g_telemetry; }

bool MfgUnlock::SoftwarePacing() { return std::string_view(LastStatus().FlipMetering) == "patched"; }

void MfgUnlock::RecordSetOptions(unsigned int requested, unsigned int sent, bool active, unsigned int result)
{
    // Above 2X with hardware flip metering still on can freeze presentation. Say so once, and change
    // nothing: whether it freezes depends on the game and the plugin build.
    static std::atomic_bool warned { false };

    if (active && sent > 1 && UnlockedMax() > 0 && !SoftwarePacing() &&
        !Config::Instance()->DisableFlipMetering.value_or(false) && !warned.exchange(true))
        LOG_WARN("MFG unlock: {}X requested with hardware flip metering still on. If presentation freezes, try "
                 "[NvApi] DisableFlipMetering=true, then [DLSSG] AdaFlipMeteringPatch=true.",
                 sent + 1);

    g_telemetry.requested.store(requested, std::memory_order_relaxed);
    g_telemetry.sent.store(sent, std::memory_order_relaxed);
    g_telemetry.result.store(result, std::memory_order_relaxed);
    g_telemetry.active.store(active, std::memory_order_relaxed);
    g_telemetry.optionsSeen.store(true, std::memory_order_release);
}

void MfgUnlock::RecordState(unsigned int presented)
{
    g_telemetry.presented.store(presented, std::memory_order_relaxed);

    auto seenMax = g_telemetry.maxPresented.load(std::memory_order_relaxed);
    while (presented > seenMax &&
           !g_telemetry.maxPresented.compare_exchange_weak(seenMax, presented, std::memory_order_relaxed))
    {
    }

    g_telemetry.stateSeen.store(true, std::memory_order_release);
}

void MfgUnlock::OnStreamlinePluginLoaded(HMODULE plugin)
{
    if (plugin == nullptr || !AdaUnlockWanted())
        return;

    {
        std::lock_guard lock(g_mutex);

        if (std::find(g_plugins.begin(), g_plugins.end(), plugin) == g_plugins.end())
        {
            g_plugins.push_back(plugin);

            // At load, ahead of any use of the plugin.
            PatchFlipMetering(plugin);
        }
    }

    // A no-op until the snippet unlock has landed; TryApply calls it again then.
    PatchPluginCeilings();
}

unsigned int MfgUnlock::UnlockedMax()
{
    const auto& status = LastStatus();

    return status.AdvertiseMatched && status.ValidateMatched && status.KernelsRewritten > 0 ? kMaxGeneratedFrames : 0;
}

bool MfgUnlock::Pending()
{
    if (!EnabledForSession() || LastStatus().ModuleFound)
        return false;
    const auto& gpu = IdentifyGpu::getPrimaryGpu();
    return gpu.vendorId == VendorId::Nvidia && gpu.nvidiaArchInfo.architecture_id == NV_GPU_ARCHITECTURE_AD100;
}

MfgUnlock::Status MfgUnlock::LastStatus()
{
    std::lock_guard lock(g_mutex);
    return g_status;
}

bool MfgUnlock::EnabledForSession()
{
    // Latch before the first FG load/query. UI changes take effect on the next launch.
    static const bool enabled = Config::Instance()->FGDLSSGAdaMfgUnlock.value_or_default();
    return enabled;
}

MfgUnlock::TemporalMethod MfgUnlock::ConfiguredTemporalMethod()
{
    const auto* config = Config::Instance();

    std::optional<std::string> fix;
    if (config->FGDLSSGAdaTemporalFix.has_value())
        fix = config->FGDLSSGAdaTemporalFix.value_or("Auto");

    return ResolveTemporalMethod(fix);
}

#endif
