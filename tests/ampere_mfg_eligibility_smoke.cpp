// ============================================================================
// tests/ampere_mfg_eligibility_smoke.cpp
// Todo 5 of .omo/plans/rtx2030-mfg-integration.md: the eligibility + FG-ownership matrix.
//
// What this harness is
// --------------------
// The production loader FILE is the code under test. tests/ampere_mfg_eligibility_mocks.h replaces the four
// sources AmpereMfgLoader.cpp reads the host from - the restored config surface, IdentifyGpu's adapter list,
// MfgUnlock's session latch and State's FG output - and the empty headers in
// tests/ampere_mfg_eligibility_seams/ stop the real ones from being pulled in; everything else is shipped code,
// compiled here with /DOPTISCALER_RTX40_MFG:
//
//   * the matrix drives AmpereMfgLoader::Evaluate() through the production host readers
//     (HostAdapterFacts / HostOwnershipFacts / HostOptions) with fixture facts - no 20/30 card, no game;
//   * every row also reads AmpereMfgLoader::LoadAttempts() around the call: the gate is a decision and must
//     never attempt a load, so the delta has to be 0 on every row, refused and eligible alike;
//   * the arm cases run the real AmpereMfgLoader::Arm() in a CHILD PROCESS (the loader latches once per
//     process, so one case per process is the only way to observe it) against a scratch package root, and
//     assert the exact status, the load-attempt count, whether the companion INI was written at all, and - for
//     the eligible case - that a second Arm() neither re-loads nor changes the status (C4);
//   * arm-host fills the mocked IdentifyGpu with THIS machine's real adapters, read with DXGI and NVAPI the way
//     IdentifyGpu::checkGpuInfo/queryNvapi read them, and runs the production Arm() against them: the shipped
//     behaviour on an Ada host must be Ineligible with zero load attempts.
//
// Nothing here executes GPU code, loads the pinned payload, launches a game or touches a live install. The
// payload file staged into the scratch trees is the todo-2 stub DLL.
//
// Cases
// -----
//   matrix rows   el1..el4 eligible (SM75/SM86 with External=true), in1..in10 ineligible (SM80, GTX 16 x2,
//                 mixed host, Ada, Blackwell, pre-Turing, non-NVIDIA, no architecture id, unknown
//                 architecture), co1..co4 conflicts (Ada unlock active, another owner, External off,
//                 OptiScaler-owned DLSSG output), di1..di3 off
//   arm cases     arm-disabled, arm-eligible-sm86, arm-eligible-sm75, arm-eligible-missing-payload,
//                 arm-latch, arm-ineligible-sm80, arm-ineligible-gtx16, arm-ineligible-mixed,
//                 arm-ineligible-ada, arm-conflict-ada, arm-conflict-other-owner, arm-conflict-external-off,
//                 arm-conflict-optifg, arm-host
//
// Exit codes
// ----------
//   0 = every matrix row and every arm case behaved as specified
//   1 = at least one row or case failed
//   2 = usage/IO error, or --expect-red was given and the permissive RED revision passed (a blind matrix)
//   3 = --expect-red: the refusals are missing exactly as a gate-less revision must show, and only refusal
//       expectations failed (the RED state this harness exists to catch)
//   4 = --expect-red: failures appeared on rows the RED revision should not affect
//
// Usage
// -----
//   ampere_mfg_eligibility_smoke.exe --evidence <dir> --scratch <dir> --stub <stub dll> [--expect-red]
//   ampere_mfg_eligibility_smoke.exe --child <case> --root <scratch case dir> --stub <dll> --out <file>
//                                    --host-out <file>
// ============================================================================

#include "ampere_mfg_eligibility_mocks.h"

#include <dxgi1_6.h>
#include <nvapi.h>

// The production loader, verbatim. Its own "AmpereMfgLoader.h" resolves next to it, so a RED run that points
// /I at an evidence copy compiles that copy's header with it (see the runner's notes).
#include "AmpereMfgLoader.cpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace
{
// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
std::string Narrow(const std::wstring& text)
{
    if (text.empty())
        return {};

    const int needed = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0,
                                           nullptr, nullptr);
    std::string out(static_cast<size_t>(needed < 0 ? 0 : needed), '\0');

    if (needed > 0)
        WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(), needed, nullptr,
                            nullptr);

    return out;
}

std::wstring Wide(const std::string& text)
{
    if (text.empty())
        return {};

    const int needed = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(needed < 0 ? 0 : needed), L'\0');

    if (needed > 0)
        MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(), needed);

    return out;
}

std::string Quote(const std::string& text)
{
    return "\"" + text + "\"";
}

std::string JsonEscape(const std::string& text)
{
    std::string escaped;

    for (const char c : text)
    {
        switch (c)
        {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default: escaped += c; break;
        }
    }

    return escaped;
}

std::string Hex(unsigned value)
{
    char buffer[16] {};
    std::snprintf(buffer, sizeof(buffer), "0x%X", value);
    return buffer;
}

std::string ReadText(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    return stream ? std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>())
                  : std::string();
}

void WriteText(const fs::path& path, const std::string& text)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << text;
}

std::vector<std::string> Lines(const std::string& text)
{
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;

    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        lines.push_back(line);
    }

    return lines;
}

// The child receipt: key=value lines, read back by the parent. Only the first '=' separates, so details that
// contain one ("[FrameGen] External=false") survive intact.
std::vector<std::pair<std::string, std::string>> ReadReceipt(const fs::path& path)
{
    std::vector<std::pair<std::string, std::string>> entries;

    for (const auto& line : Lines(ReadText(path)))
    {
        const auto separator = line.find('=');

        if (separator != std::string::npos)
            entries.emplace_back(line.substr(0, separator), line.substr(separator + 1));
    }

    return entries;
}

std::string ValueOf(const std::vector<std::pair<std::string, std::string>>& receipt, const std::string& key)
{
    for (const auto& [name, value] : receipt)
    {
        if (name == key)
            return value;
    }

    return "<missing>";
}

fs::path OwnExecutablePath()
{
    wchar_t path[MAX_PATH] {};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    return fs::path(path);
}

// ---------------------------------------------------------------------------
// The host probe (test side): what the real machine reports
// ---------------------------------------------------------------------------
// IdentifyGpu::checkGpuInfo/queryNvapi cannot be linked outside the game (they sit on the DXGI/Vulkan proxies
// and the hook layer), so the probe repeats the two reads they make, through the same APIs:
//   * DXGI for the adapters the OS reports (EnumAdapterByGpuPreference + DXGI_ADAPTER_DESC1);
//   * NVAPI for the architecture of the physical GPU behind each adapter LUID
//     (NvAPI_SYS_GetLogicalGPUs -> NvAPI_GPU_GetLogicalGpuInfo -> NvAPI_GPU_GetArchInfo), the chain
//     IdentifyGpu::queryNvapi walks.
// The interface ids are the ones in external/nvapi/nvapi_interface.h, the table NvApiTypes resolves names
// through.
struct ProbeEntry
{
    std::string name;
    unsigned vendorId = 0;
    unsigned deviceId = 0;
    unsigned archId = 0;
    unsigned implId = 0;
    bool software = false;
    bool archRead = false;
    LUID luid {};
};

struct HostProbe
{
    bool dxgiOk = false;
    bool nvapiOk = false;
    int hardwareAdapterCount = 0;
    std::string notes;
    std::vector<ProbeEntry> entries;
};

HostProbe ProbeHostAdapter()
{
    HostProbe probe;

    IDXGIFactory6* factory = nullptr;

    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory6), reinterpret_cast<void**>(&factory))) || factory == nullptr)
    {
        probe.notes = "the DXGI factory could not be created";
        return probe;
    }

    probe.dxgiOk = true;

    for (UINT index = 0;; ++index)
    {
        IDXGIAdapter1* adapter = nullptr;

        if (factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_UNSPECIFIED, __uuidof(IDXGIAdapter1),
                                                reinterpret_cast<void**>(&adapter)) != S_OK ||
            adapter == nullptr)
            break;

        DXGI_ADAPTER_DESC1 desc {};

        if (adapter->GetDesc1(&desc) == S_OK)
        {
            ProbeEntry entry;
            entry.name = Narrow(desc.Description);
            entry.vendorId = desc.VendorId;
            entry.deviceId = desc.DeviceId;
            entry.software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
            entry.luid = desc.AdapterLuid;

            if (!entry.software)
                ++probe.hardwareAdapterCount;

            probe.entries.push_back(entry);
        }

        adapter->Release();
    }

    factory->Release();

    HMODULE nvapi = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);

    if (nvapi == nullptr)
    {
        probe.notes = "nvapi64.dll is not present";
        return probe;
    }

    auto query = reinterpret_cast<void* (__stdcall*) (unsigned int)>(
        reinterpret_cast<void*> (GetProcAddress(nvapi, "nvapi_QueryInterface")));

    if (query == nullptr)
    {
        probe.notes = "nvapi_QueryInterface is missing";
        FreeLibrary(nvapi);
        return probe;
    }

    auto initialize = reinterpret_cast<decltype(&NvAPI_Initialize)>(query(0x0150e828));
    auto unload = reinterpret_cast<decltype(&NvAPI_Unload)>(query(0xd22bdd7e));
    auto getLogicalGpus = reinterpret_cast<decltype(&NvAPI_SYS_GetLogicalGPUs)>(query(0xccfffc10));
    auto getLogicalGpuInfo = reinterpret_cast<decltype(&NvAPI_GPU_GetLogicalGpuInfo)>(query(0x842b066e));
    auto getArchInfo = reinterpret_cast<decltype(&NvAPI_GPU_GetArchInfo)>(query(0xd8265d24));

    if (initialize == nullptr || initialize() != NVAPI_OK)
    {
        probe.notes = "NvAPI_Initialize failed";
        FreeLibrary(nvapi);
        return probe;
    }

    probe.nvapiOk = true;

    NV_LOGICAL_GPUS logicals {};
    logicals.version = NV_LOGICAL_GPUS_VER;

    if (getLogicalGpus != nullptr && getLogicalGpus(&logicals) == NVAPI_OK)
    {
        for (uint32_t i = 0; i < logicals.gpuHandleCount; ++i)
        {
            LUID luid {};
            NV_LOGICAL_GPU_DATA data {};
            data.version = NV_LOGICAL_GPU_DATA_VER;
            data.pOSAdapterId = &luid;

            if (getLogicalGpuInfo == nullptr ||
                getLogicalGpuInfo(logicals.gpuHandleData[i].hLogicalGpu, &data) != NVAPI_OK ||
                data.physicalGpuCount == 0)
                continue;

            NV_GPU_ARCH_INFO arch {};
            arch.version = NV_GPU_ARCH_INFO_VER;

            if (getArchInfo == nullptr || getArchInfo(data.physicalGpuHandles[0], &arch) != NVAPI_OK)
                continue;

            for (auto& entry : probe.entries)
            {
                if (entry.luid.HighPart == luid.HighPart && entry.luid.LowPart == luid.LowPart)
                {
                    entry.archId = arch.architecture_id;
                    entry.implId = arch.implementation;
                    entry.archRead = true;
                }
            }
        }
    }
    else
    {
        probe.notes = "NvAPI_SYS_GetLogicalGPUs failed";
    }

    if (unload != nullptr)
        unload();

    FreeLibrary(nvapi);
    return probe;
}

std::string ProbeJson(const HostProbe& probe)
{
    std::string text;
    text += "{\n";
    text += " \"schema\": 1,\n";
    text += " \"tool\": \"tests/ampere_mfg_eligibility_smoke.cpp (DXGI + NVAPI)\",\n";
    text += " \"dxgiOk\": " + std::string(probe.dxgiOk ? "true" : "false") + ",\n";
    text += " \"nvapiOk\": " + std::string(probe.nvapiOk ? "true" : "false") + ",\n";
    text += " \"hardwareAdapterCount\": " + std::to_string(probe.hardwareAdapterCount) + ",\n";
    text += " \"notes\": \"" + JsonEscape(probe.notes) + "\",\n";
    text += " \"adapters\": [\n";

    for (size_t i = 0; i < probe.entries.size(); ++i)
    {
        const auto& entry = probe.entries[i];

        text += "  {\"index\": " + std::to_string(i) + ", \"name\": \"" + JsonEscape(entry.name) +
                "\", \"vendorId\": \"" + Hex(entry.vendorId) + "\", \"deviceId\": \"" + Hex(entry.deviceId) +
                "\", \"archId\": \"" + (entry.archRead ? Hex(entry.archId) : std::string("unread")) +
                "\", \"implementation\": " + std::to_string(entry.implId) +
                ", \"software\": " + std::string(entry.software ? "true" : "false") +
                ", \"archRead\": " + std::string(entry.archRead ? "true" : "false") + "}" +
                (i + 1 == probe.entries.size() ? "" : ",") + "\n";
    }

    text += " ]\n";
    text += "}\n";
    return text;
}

std::string ProbeLines(const HostProbe& probe)
{
    std::string text;
    text += "dxgiOk=" + std::string(probe.dxgiOk ? "true" : "false") + "\n";
    text += "nvapiOk=" + std::string(probe.nvapiOk ? "true" : "false") + "\n";
    text += "hardwareAdapterCount=" + std::to_string(probe.hardwareAdapterCount) + "\n";
    text += "notes=" + probe.notes + "\n";

    for (size_t i = 0; i < probe.entries.size(); ++i)
    {
        const auto& entry = probe.entries[i];

        text += "adapter[" + std::to_string(i) + "] name='" + entry.name + "' vendor=" + Hex(entry.vendorId) +
                " device=" + Hex(entry.deviceId) + " arch=" + (entry.archRead ? Hex(entry.archId) : "unread") +
                " impl=" + std::to_string(entry.implId) +
                " software=" + std::string(entry.software ? "true" : "false") + "\n";
    }

    return text;
}

// ---------------------------------------------------------------------------
// The case table
// ---------------------------------------------------------------------------
struct Case
{
    std::string id;
    std::string name;
    std::string child; // arm-case id; empty = matrix row only

    std::vector<GpuInformation> adapters;

    bool enabled = true;
    bool externalFg = true;
    bool adaUnlock = false;
    bool optiFgOutput = false;
    bool otherOwner = false; // the stub module is loaded into the child under the shipped name

    std::string expectedVerdict;
    std::string expectedDetail;

    // Arm-case expectations (child cases only). The gate detail is the matrix's; a status has its own.
    std::string expectedStatus;       // StateText() of the latched status
    std::string expectedArmDetail;    // exact Status.Detail, when non-empty
    std::string expectedArmDetailPrefix; // prefix of Status.Detail (file-work rows, whose byte count is a build fact)
    int expectedAttempts = 0;
    bool expectedIni = false;
    bool missPayload = false;
    bool secondArm = false;
    bool hostProbe = false;
};

GpuInformation Adapter(const std::string& name, unsigned vendor, unsigned arch, unsigned impl, bool software = false)
{
    GpuInformation gpu;
    gpu.name = name;
    gpu.vendorId = static_cast<VendorId::Value>(vendor);
    gpu.softwareAdapter = software;
    gpu.nvidiaArchInfo.architecture = arch;
    gpu.nvidiaArchInfo.architecture_id = arch;
    gpu.nvidiaArchInfo.implementation = impl;
    return gpu;
}

GpuInformation NvidiaRtx(const std::string& name, unsigned arch, unsigned impl)
{
    return Adapter(name, VendorId::Nvidia, arch, impl);
}

constexpr unsigned kArchIntelIgpu = 0x0; // the iGPU carries no NVAPI architecture; only its presence counts

std::vector<Case> Cases()
{
    const std::string supported = "; the payload runs on SM75 (RTX 20) or SM86 (RTX 30) only";

    const GpuInformation sm86 = NvidiaRtx("NVIDIA GeForce RTX 3060", AmpereMfgLoader::kArchAmpere, 6);
    const GpuInformation sm86Ga104 = NvidiaRtx("NVIDIA GeForce RTX 3070", AmpereMfgLoader::kArchAmpere, 4);
    const GpuInformation sm75 = NvidiaRtx("NVIDIA GeForce RTX 2080 Ti", AmpereMfgLoader::kArchTuring,
                                           AmpereMfgLoader::kImplTu102);
    const GpuInformation sm75Tu106 = NvidiaRtx("NVIDIA GeForce RTX 2060", AmpereMfgLoader::kArchTuring,
                                               AmpereMfgLoader::kImplTu106);
    const GpuInformation intelIgpu = Adapter("Intel(R) UHD Graphics 630", VendorId::Intel, kArchIntelIgpu, 0);
    const GpuInformation basicRender = Adapter("Microsoft Basic Render Driver", VendorId::Microsoft, kArchIntelIgpu, 0,
                                                true);

    const std::string sm86Detail = "adapter 'NVIDIA GeForce RTX 3060' is SM86 (RTX 30); External frame generation "
                                   "is on and no other owner holds the DLSSG output";
    const std::string sm75Detail = "adapter 'NVIDIA GeForce RTX 2080 Ti' is SM75 (RTX 20); External frame "
                                   "generation is on and no other owner holds the DLSSG output";

    std::vector<Case> cases;

    auto addMatrix = [&cases](const std::string& id, const std::string& name, std::vector<GpuInformation> adapters,
                              const std::string& verdict, const std::string& detail)
    {
        Case c;
        c.id = id;
        c.name = name;
        c.adapters = std::move(adapters);
        c.expectedVerdict = verdict;
        c.expectedDetail = detail;
        cases.push_back(std::move(c));
    };

    // Eligible rows: SM75 (RTX 20) and SM86 (RTX 30) with External=true - the supported configuration, which
    // must NOT be refused.
    addMatrix("el1", "RTX 3070 SM86 (GA104) External=true", { sm86Ga104 }, "Eligible",
              "adapter 'NVIDIA GeForce RTX 3070' is SM86 (RTX 30); External frame generation is on and no other "
              "owner holds the DLSSG output");
    addMatrix("el2", "RTX 3060 SM86 (GA106) External=true", { sm86 }, "Eligible", sm86Detail);
    addMatrix("el3", "RTX 2060 SM75 (TU106) External=true", { sm75Tu106 }, "Eligible",
              "adapter 'NVIDIA GeForce RTX 2060' is SM75 (RTX 20); External frame generation is on and no other "
              "owner holds the DLSSG output");
    addMatrix("el4", "RTX 2080 Ti SM75 (TU102) External=true", { sm75 }, "Eligible", sm75Detail);

    // Ineligible rows: every shape of "not a 20/30 part", each with a named status.
    addMatrix("in1", "A100 SM80 (GA100)", { NvidiaRtx("NVIDIA A100-SXM4-40GB", AmpereMfgLoader::kArchAmpere, 0) },
              "Ineligible", "adapter 'NVIDIA A100-SXM4-40GB' is SM80 (GA100)" + supported);
    addMatrix("in2", "GTX 1660 SUPER (TU116, SM75 without tensor units)",
              { NvidiaRtx("NVIDIA GeForce GTX 1660 SUPER", AmpereMfgLoader::kArchTuring, AmpereMfgLoader::kImplTu116) },
              "Ineligible",
              "adapter 'NVIDIA GeForce GTX 1660 SUPER' is a Turing GTX 16 part: SM75 without the RTX 20 tensor "
              "units" +
                  supported);
    addMatrix("in3", "GTX 1650 (TU117, SM75 without tensor units)",
              { NvidiaRtx("NVIDIA GeForce GTX 1650", AmpereMfgLoader::kArchTuring, AmpereMfgLoader::kImplTu117) },
              "Ineligible",
              "adapter 'NVIDIA GeForce GTX 1650' is a Turing GTX 16 part: SM75 without the RTX 20 tensor units" +
                  supported);
    addMatrix("in4", "mixed host: two adapters, no correlation", { sm86, intelIgpu }, "Ineligible",
              "the host reports 2 physical adapters and adapter 'NVIDIA GeForce RTX 3060' is not correlated to "
              "this process's render device" +
                  supported);
    addMatrix("in5", "RTX 4090 SM89 (Ada)", { NvidiaRtx("NVIDIA GeForce RTX 4090", AmpereMfgLoader::kArchAda, 2) },
              "Ineligible",
              "adapter 'NVIDIA GeForce RTX 4090' is SM89 (Ada); the 40 series path is the Ada unlock, not this "
              "payload");
    addMatrix("in6", "RTX 5080 (Blackwell)", { NvidiaRtx("NVIDIA GeForce RTX 5080", AmpereMfgLoader::kArchBlackwell, 2) },
              "Ineligible", "adapter 'NVIDIA GeForce RTX 5080' is a Blackwell part (SM100 or SM120)" + supported);
    addMatrix("in7", "GTX 1080 (pre-Turing)", { NvidiaRtx("NVIDIA GeForce GTX 1080", 0x130, 0) }, "Ineligible",
              "adapter 'NVIDIA GeForce GTX 1080' is a pre-Turing NVIDIA part" + supported);
    addMatrix("in8", "Intel iGPU", { intelIgpu }, "Ineligible",
              "adapter 'Intel(R) UHD Graphics 630' is not an NVIDIA adapter" + supported);
    addMatrix("in9", "NVIDIA adapter with no architecture id", { NvidiaRtx("NVIDIA GPU", 0, 0) }, "Ineligible",
              "adapter 'NVIDIA GPU' is an NVIDIA part whose architecture id is missing" + supported);
    addMatrix("in10", "unknown NVIDIA architecture (0x1C0)", { NvidiaRtx("NVIDIA GPU", 0x1C0, 0) }, "Ineligible",
              "adapter 'NVIDIA GPU' is an NVIDIA architecture this build does not recognise" + supported);
    addMatrix("in11", "a software adapter is all the host reports", { basicRender }, "Ineligible",
              "the host reports no physical adapter and adapter 'Microsoft Basic Render Driver' is not correlated "
              "to this process's render device" +
                  supported);

    // Conflict rows: genuine ownership conflicts. All of them sit on an eligible part with External=true, so
    // the conflict is the only cause - except co3, which is exactly the External=false state.
    addMatrix("co1", "Ada unlock on for the session", { sm86 }, "Conflict",
              "the Ada (RTX 40) unlock is active for this session; one frame-generation owner only");
    addMatrix("co2", "another unlocker/FG owner present", { sm86 }, "Conflict",
              "another unlocker or frame-generation owner is already present: the payload module is loaded "
              "without this loader having loaded it");
    addMatrix("co3", "External off while the payload needs game-Streamline ownership", { sm86 }, "Conflict",
              "External frame generation does not own this session; enable [FrameGen] External and restart "
              "with no active OptiScaler FG selections so the game's Streamline owns frame generation");
    addMatrix("co4", "OptiScaler-owned DLSSG output active", { sm86 }, "Conflict",
              "OptiScaler's own DLSSG output path is active; it owns the generated frames for this session");

    // Off first: the setting is answered before anything else is read.
    addMatrix("di1", "off on an eligible part", { sm86 }, "Disabled", "the 20/30 unlock is off");
    addMatrix("di2", "off on an Ada part", { NvidiaRtx("NVIDIA GeForce RTX 4090", AmpereMfgLoader::kArchAda, 2) },
              "Disabled", "the 20/30 unlock is off");
    addMatrix("di3", "off on a mixed host", { sm86, intelIgpu }, "Disabled", "the 20/30 unlock is off");

    for (auto& c : cases)
    {
        if (c.id == "co1")
            c.adaUnlock = true;

        if (c.id == "co2")
            c.otherOwner = true;

        if (c.id == "co3")
            c.externalFg = false;

        if (c.id == "co4")
            c.optiFgOutput = true;

        if (c.id == "di1" || c.id == "di2" || c.id == "di3")
            c.enabled = false;
    }

    // ------------------------------------------------------------------
    // Arm cases: copies of the rows above, run through the real Arm()
    // ------------------------------------------------------------------
    const auto findRow = [&cases](const std::string& id) -> Case
    {
        for (const auto& c : cases)
        {
            if (c.id == id)
                return c;
        }

        return {};
    };

    std::vector<Case> arms;

    auto addArm = [&arms](const std::string& child, const std::string& name, Case c)
    {
        c.id = child;
        c.child = child;
        c.name = name;
        arms.push_back(std::move(c));
    };

    {
        Case eligible = findRow("el2");
        // Todo 6 wired the production arming path to LOAD the payload at the Streamline init boundary, so the
        // eligible rows now report the state after the load (the plan's ladder: PayloadValidated -> Loaded). The
        // assertions below are unchanged otherwise: one load attempt, the companion INI written.
        eligible.expectedStatus = "Loaded";
        eligible.expectedArmDetail.clear(); // the file work has its own detail; the gate detail is the matrix's
        eligible.expectedArmDetailPrefix = "payload loaded from ";
        eligible.expectedAttempts = 1;
        eligible.expectedIni = true;

        Case latch = eligible;
        latch.secondArm = true;

        Case missing = eligible;
        missing.missPayload = true;
        missing.expectedStatus = "PayloadMissing";
        missing.expectedArmDetailPrefix = "the payload module dlssg_sm86.dll was not found under";
        missing.expectedIni = false;

        Case disabled = eligible;
        disabled.enabled = false;
        disabled.expectedVerdict = "Disabled";
        disabled.expectedDetail = "the 20/30 unlock is off";
        disabled.expectedStatus = "Disabled";
        disabled.expectedArmDetail = "the 20/30 unlock is off";
        disabled.expectedArmDetailPrefix.clear();
        disabled.expectedAttempts = 0;
        disabled.expectedIni = false;

        addArm("arm-disabled", "Arm() with the setting off", disabled);
        addArm("arm-eligible-sm86", "Arm() on an eligible SM86 host with External=true", eligible);
        addArm("arm-latch", "Arm() twice: one pass, one load attempt", latch);
        addArm("arm-eligible-missing-payload", "Arm() eligible with no payload staged", missing);
    }

    {
        Case sm75Eligible = findRow("el4");
        sm75Eligible.expectedStatus = "Loaded";
        sm75Eligible.expectedArmDetail.clear();
        sm75Eligible.expectedArmDetailPrefix = "payload loaded from ";
        sm75Eligible.expectedAttempts = 1;
        sm75Eligible.expectedIni = true;
        addArm("arm-eligible-sm75", "Arm() on an eligible SM75 (RTX 20) host", sm75Eligible);
    }

    const std::pair<const char*, const char*> refusals[] = {
        { "in1", "arm-ineligible-sm80" },
        { "in2", "arm-ineligible-gtx16" },
        { "in4", "arm-ineligible-mixed" },
        { "in5", "arm-ineligible-ada" },
        { "in11", "arm-ineligible-software-adapter" },
        { "co1", "arm-conflict-ada" },
        { "co2", "arm-conflict-other-owner" },
        { "co3", "arm-conflict-external-off" },
        { "co4", "arm-conflict-optifg" },
    };

    for (const auto& [rowId, child] : refusals)
    {
        Case c = findRow(rowId);
        c.expectedStatus = c.expectedVerdict; // Arm() reports the refusal verbatim, detail included
        c.expectedArmDetail = c.expectedDetail;
        c.expectedArmDetailPrefix.clear();
        c.expectedAttempts = 0;
        c.expectedIni = false;

        // The name is built before the move: reading c after std::move(c) is not defined to see the row.
        const std::string name = "Arm() refused: " + std::string(c.expectedVerdict) + " (" + rowId + ")";
        addArm(child, name, std::move(c));
    }

    {
        Case host;
        host.child = "arm-host";
        host.name = "Arm() against this machine's real adapters";
        host.hostProbe = true;
        host.expectedVerdict = "Ineligible";
        host.expectedDetail = "<this host>";
        host.expectedStatus = "Ineligible";
        host.expectedAttempts = 0;
        host.expectedIni = false;
        arms.push_back(std::move(host));
    }

    cases.insert(cases.end(), arms.begin(), arms.end());
    return cases;
}

// ---------------------------------------------------------------------------
// Fixture application and the receipt
// ---------------------------------------------------------------------------
struct Row
{
    std::string caseId;
    std::string name;
    std::string expected;
    std::string observed;
    bool ok = false;
};

std::vector<Row> g_rows;
std::vector<std::string> g_matrixJson;
std::vector<std::string> g_armJson;
std::vector<std::string> g_loadCounts;
fs::path g_evidence;
fs::path g_scratch;
fs::path g_stub;
int g_failures = 0;
int g_refusalFailures = 0;
int g_allowedFailures = 0;
bool g_refusalRow = false;
bool g_expectRed = false;

void Check(const std::string& caseId, const std::string& name, const std::string& expected, const std::string& observed)
{
    const bool ok = expected == observed;
    g_rows.push_back({ caseId, name, expected, observed, ok });

    if (ok)
    {
        std::printf("pass %-28s %-54s = %s\n", caseId.c_str(), name.c_str(), observed.c_str());
    }
    else
    {
        ++g_failures;

        if (g_refusalRow)
            ++g_refusalFailures;
        else
            ++g_allowedFailures;

        std::printf("FAIL %-28s %-54s expected [%s] observed [%s]\n", caseId.c_str(), name.c_str(), expected.c_str(),
                    observed.c_str());
    }
}

void ApplyFacts(const Case& c)
{
    IdentifyGpu::SetAdapters(c.adapters);

    auto* config = Config::Instance();
    config->ExternalFrameGeneration.stored = c.externalFg;
    config->FGDLSSGAmpereMfgUnlock.stored = c.enabled;
    config->FGDLSSGAmpereMfgMaxFrames.stored = 3;
    config->FGDLSSGAmpereMfgKernelImage.stored = std::nullopt;

    MfgUnlock::g_enabledForSession = c.adaUnlock;

    auto& state = State::Instance();
    state.activeFgInput = FGInput::NoFG;
    state.activeFgOutput = c.optiFgOutput ? FGOutput::DLSSG : FGOutput::NoFG;
    // A native DLSSG output conflicts even with no nvngx replacement (B2).
    state.activeFgNvngx = FGNvngxReplacement::None;
}

void ApplyStartupSelections()
{
#include "production-fg-startup.inc"
}

// Compile the real hook's ownership suppression block, not a copy of its condition.
namespace sl
{
enum class DLSSGMode { eOff, eOn };
}

sl::DLSSGMode ForwardGameMode()
{
    auto& state = State::Instance();
    struct { sl::DLSSGMode mode; } newOptions { sl::DLSSGMode::eOn };
    const int viewport = 0;
    const auto o_slDLSSGSetOptions = [](int, const auto& options) { return options.mode; };
#include "production-fg-suppression.inc"
    return newOptions.mode;
}

std::string BoolText(bool value) { return value ? "true" : "false"; }

void CheckStartupOwnership()
{
    auto* config = Config::Instance();
    auto& state = State::Instance();
    for (bool external : { false, true })
    {
        for (auto input : { FGInput::Upscaler, FGInput::DLSSG, FGInput::NvngxFG })
        {
            config->ExternalFrameGeneration.stored = external;
            config->FGInput.stored = input;
            config->FGOutput.stored = FGOutput::DLSSG;
            config->FGNvngxReplacement.stored = FGNvngxReplacement::Nukems;
            ApplyStartupSelections();
            const auto expectedOutput = external || input == FGInput::NvngxFG ? FGOutput::NoFG : FGOutput::DLSSG;
            Check("startup", "active input", "true", BoolText(state.activeFgInput == (external ? FGInput::NoFG : input)));
            Check("startup", "active output", "true", BoolText(state.activeFgOutput == expectedOutput));
            Check("startup", "active replacement", "true", BoolText(state.activeFgNvngx ==
                  (external ? FGNvngxReplacement::None : FGNvngxReplacement::Nukems)));
            Check("startup", "host ownership", BoolText(external),
                  BoolText(AmpereMfgLoader::HostOwnershipFacts().ExternalFrameGeneration));
            const bool suppress = !external && input == FGInput::Upscaler;
            Check("startup", "game mode forwarded", BoolText(!suppress),
                  BoolText(ForwardGameMode() == sl::DLSSGMode::eOn));
        }
    }

    // Turning the stored flag on mid-session cannot grant ownership while any selection remains active.
    config->ExternalFrameGeneration.stored = true;
    for (int selection = 0; selection < 3; ++selection)
    {
        state.activeFgInput = selection == 0 ? FGInput::DLSSG : FGInput::NoFG;
        state.activeFgOutput = selection == 1 ? FGOutput::FSRFG : FGOutput::NoFG;
        state.activeFgNvngx = selection == 2 ? FGNvngxReplacement::Nukems : FGNvngxReplacement::None;
        Check("startup", "stored flag cannot grant ownership", "false",
              BoolText(AmpereMfgLoader::HostOwnershipFacts().ExternalFrameGeneration));
    }
    ApplyStartupSelections();
    config->ExternalFrameGeneration.stored = false;
    Check("startup", "menu change does not adopt selections", "true",
          BoolText(state.activeFgInput == FGInput::NoFG && state.activeFgOutput == FGOutput::NoFG &&
                   state.activeFgNvngx == FGNvngxReplacement::None));
}

// ---------------------------------------------------------------------------
// Child process
// ---------------------------------------------------------------------------
int RunChildProcess(const fs::path& exe, const std::string& arguments, const fs::path& stdioFile)
{
    std::error_code error;
    fs::create_directories(stdioFile.parent_path(), error);

    SECURITY_ATTRIBUTES attributes = {};
    attributes.nLength = sizeof(attributes);
    attributes.bInheritHandle = TRUE;

    HANDLE stdio = CreateFileW(stdioFile.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ, &attributes, CREATE_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL, nullptr);

    STARTUPINFOW startup = {};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = nullptr;
    startup.hStdOutput = stdio;
    startup.hStdError = stdio;

    PROCESS_INFORMATION process = {};
    std::wstring commandLine = L"\"" + exe.wstring() + L"\" " + Wide(arguments);

    const BOOL created = CreateProcessW(exe.wstring().c_str(), commandLine.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);

    if (stdio != INVALID_HANDLE_VALUE)
        CloseHandle(stdio);

    if (!created)
    {
        std::printf("child process could not be created: error %lu\n", GetLastError());
        return -1;
    }

    const DWORD wait = WaitForSingleObject(process.hProcess, 30000);
    DWORD exitCode = 0;

    if (wait == WAIT_TIMEOUT)
    {
        TerminateProcess(process.hProcess, 0xDEAD);
        WaitForSingleObject(process.hProcess, 5000);
        exitCode = 0xDEAD;
    }
    else
    {
        GetExitCodeProcess(process.hProcess, &exitCode);
    }

    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exitCode);
}

// ---------------------------------------------------------------------------
// Child mode
// ---------------------------------------------------------------------------
const Case* FindChild(const std::vector<Case>& cases, const std::string& child)
{
    for (const auto& c : cases)
    {
        if (c.child == child)
            return &c;
    }

    return nullptr;
}

int RunChild(const std::string& child, const fs::path& root, const fs::path& stub, const fs::path& out,
             const fs::path& hostOut)
{
    const auto cases = Cases();
    const Case* found = FindChild(cases, child);

    if (found == nullptr)
    {
        std::printf("unknown child case: %s\n", child.c_str());
        return 1;
    }

    Case c = *found;
    std::string probeText;

    if (c.hostProbe)
    {
        // This machine, as DXGI and NVAPI report it: every adapter becomes a mock IdentifyGpu entry, the
        // preferred one first, so the production HostAdapterFacts sees this host's real shape.
        const HostProbe probe = ProbeHostAdapter();
        probeText = ProbeLines(probe);

        std::vector<GpuInformation> adapters;

        for (const auto& entry : probe.entries)
            adapters.push_back(Adapter(entry.name, entry.vendorId, entry.archRead ? entry.archId : 0,
                                       entry.archRead ? entry.implId : 0, entry.software));

        if (adapters.empty())
            adapters.push_back(GpuInformation {});

        c.adapters = adapters;

        if (!hostOut.empty())
            WriteText(hostOut, ProbeJson(probe));
    }

    ApplyFacts(c);

    if (c.otherOwner)
    {
        // A real module under the shipped name, loaded before Arm(): this is what "another owner already holds
        // the DLSSG output" looks like from inside the process, and the production
        // GetModuleHandleW(kPayloadModuleName) check detects it - no fixture flag involved.
        if (stub.empty() || LoadLibraryW(stub.wstring().c_str()) == nullptr)
        {
            std::printf("the stub module could not be loaded from %s\n", stub.string().c_str());
            return 1;
        }
    }

    // Todo 13's pin seam: the production ladder compares the module it found against the pinned payload, and
    // this harness stages the stub as that payload - so the pin it hands the ladder is the stub's own size and
    // digest, not the shipped 30 MB pair (tests/ampere_mfg_payload_pin_seam.h).
    if (!stub.empty())
        AmpereMfgPinSeam::PinTo(stub);

    AmpereMfgLoader::ArmOptions options = AmpereMfgLoader::HostOptions();
    options.PackageRoot = root;

    const auto status = AmpereMfgLoader::Arm(options);

    std::string secondStatus;
    int attemptsAfterSecond = AmpereMfgLoader::LoadAttempts();

    if (c.secondArm)
    {
        const auto second = AmpereMfgLoader::Arm(options);
        secondStatus = second.StateText();
        attemptsAfterSecond = AmpereMfgLoader::LoadAttempts();
    }

    const auto adapter = AmpereMfgLoader::HostAdapterFacts();
    const auto ownership = AmpereMfgLoader::HostOwnershipFacts();
    const auto decision = AmpereMfgLoader::Evaluate(options, adapter, ownership);

    std::string receipt;
    receipt += "case=" + child + "\n";
    receipt += "verdict=" + std::string(AmpereMfgLoader::VerdictName(decision.verdict)) + "\n";
    receipt += "gate_detail=" + decision.Detail + "\n";
    receipt += "adapter_count=" + std::to_string(adapter.PhysicalAdapterCount) + "\n";
    receipt += "adapter_name=" + adapter.Name + "\n";
    receipt += "status=" + std::string(status.StateText()) + "\n";
    receipt += "detail=" + status.Detail + "\n";
    receipt += "module_path=" + (status.ModulePath.empty() ? std::string("<empty>") : Narrow(status.ModulePath)) + "\n";
    receipt += "ini_path=" + (status.IniPath.empty() ? std::string("<empty>") : Narrow(status.IniPath)) + "\n";
    receipt += "ini_written=" + std::string(!status.IniPath.empty() && fs::exists(status.IniPath) ? "true" : "false") +
               "\n";
    receipt += "load_attempts=" + std::to_string(AmpereMfgLoader::LoadAttempts()) + "\n";
    receipt += "second_status=" + (secondStatus.empty() ? std::string("<none>") : secondStatus) + "\n";
    receipt += "load_attempts_after_second=" + std::to_string(attemptsAfterSecond) + "\n";
    receipt += "host_probe=" + std::string(c.hostProbe ? "true" : "false") + "\n";

    for (const auto& line : Lines(probeText))
        receipt += "probe." + line + "\n";

    WriteText(out, receipt);
    std::printf("%s", receipt.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// Parent mode
// ---------------------------------------------------------------------------
std::string StagePayload(const fs::path& caseRoot, bool stage)
{
    if (!stage)
        return "<not staged>";

    const fs::path folder = caseRoot / "OptiScaler" / "dlssg_sm86";
    std::error_code error;
    fs::create_directories(folder, error);

    const fs::path target = folder / AmpereMfgLoader::kPayloadModuleName;
    fs::copy_file(g_stub, target, fs::copy_options::overwrite_existing, error);

    return error ? "<copy failed: " + error.message() + ">" : Narrow(target.wstring());
}
} // namespace

int main(int argc, char** argv)
{
    fs::path evidence;
    fs::path scratch;
    fs::path stub;
    std::string child;
    fs::path childRoot;
    fs::path childOut;
    fs::path childHostOut;

    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string
        {
            return i + 1 < argc ? argv[++i] : std::string();
        };

        if (argument == "--evidence")
            evidence = next();
        else if (argument == "--scratch")
            scratch = next();
        else if (argument == "--stub")
            stub = next();
        else if (argument == "--child")
            child = next();
        else if (argument == "--root")
            childRoot = next();
        else if (argument == "--out")
            childOut = next();
        else if (argument == "--host-out")
            childHostOut = next();
        else if (argument == "--expect-red")
            g_expectRed = true;
        else
        {
            std::printf("unknown argument: %s\n", argument.c_str());
            return 2;
        }
    }

    if (!child.empty())
        return RunChild(child, childRoot, stub, childOut, childHostOut);

    if (evidence.empty() || scratch.empty() || stub.empty())
    {
        std::printf("--evidence <dir> --scratch <dir> --stub <stub dll> are required\n");
        return 2;
    }

    if (!fs::exists(stub))
    {
        std::printf("the stub payload %s does not exist\n", stub.string().c_str());
        return 2;
    }

    g_evidence = evidence;
    g_scratch = scratch;
    g_stub = stub;

    std::error_code error;
    fs::create_directories(g_evidence, error);
    fs::create_directories(g_scratch, error);

    const auto cases = Cases();
    CheckStartupOwnership();

    // ------------------------------------------------------------------
    // 1. The matrix: the gate with fixture facts, and the load counter
    // ------------------------------------------------------------------
    std::printf("=== eligibility matrix (production gate, fixture facts)\n");

    for (const auto& c : cases)
    {
        // Arm cases exercise the same facts through Arm() below; the matrix is the gate on its own.
        if (!c.child.empty())
            continue;

        g_refusalRow = c.expectedVerdict == "Ineligible" || c.expectedVerdict == "Conflict";
        ApplyFacts(c);

        // "Another owner" is a fact about the process, not about the fixture: the row loads the stub under the
        // shipped name so the production GetModuleHandleW(kPayloadModuleName) check sees a real module, and
        // releases it again before the next row.
        HMODULE ownerModule = nullptr;

        if (c.otherOwner)
        {
            ownerModule = LoadLibraryW(g_stub.wstring().c_str());
            Check(c.id, c.name + " owner module loaded", "true", ownerModule != nullptr ? "true" : "false");
        }

        const auto adapter = AmpereMfgLoader::HostAdapterFacts();
        const auto ownership = AmpereMfgLoader::HostOwnershipFacts();
        const auto options = AmpereMfgLoader::HostOptions();

        const int before = AmpereMfgLoader::LoadAttempts();
        const auto decision = AmpereMfgLoader::Evaluate(options, adapter, ownership);
        const int delta = AmpereMfgLoader::LoadAttempts() - before;

        if (ownerModule != nullptr)
            FreeLibrary(ownerModule);

        Check(c.id, c.name + " verdict", c.expectedVerdict, AmpereMfgLoader::VerdictName(decision.verdict));
        Check(c.id, c.name + " detail", c.expectedDetail, decision.Detail);
        Check(c.id, c.name + " load attempt delta", "0", std::to_string(delta));

        g_matrixJson.push_back("  {\"case\": \"" + JsonEscape(c.id) + "\", \"name\": \"" + JsonEscape(c.name) +
                               "\", \"verdict\": \"" + JsonEscape(AmpereMfgLoader::VerdictName(decision.verdict)) +
                               "\", \"expectedVerdict\": \"" + JsonEscape(c.expectedVerdict) + "\", \"detail\": \"" +
                               JsonEscape(decision.Detail) + "\", \"loadAttemptDelta\": " + std::to_string(delta) + "}");
        g_loadCounts.push_back(c.id + " gate load_attempt_delta=" + std::to_string(delta) + " expected=0");
    }

    // ------------------------------------------------------------------
    // 2. The arm cases: the real Arm(), one child process each
    // ------------------------------------------------------------------
    std::printf("\n=== arm cases (production Arm(), one child process each)\n");

    for (const auto& c : cases)
    {
        if (c.child.empty())
            continue;

        g_refusalRow = c.expectedVerdict == "Ineligible" || c.expectedVerdict == "Conflict";

        const fs::path caseRoot = g_scratch / c.child;
        fs::remove_all(caseRoot, error);
        fs::create_directories(caseRoot, error);

        // The eligible and refused rows both get the staged payload: a refusal that writes the companion INI
        // (or loads anything) is exactly what the load-count and ini_written checks exist to catch.
        const bool stage = !c.missPayload && !c.hostProbe && c.enabled;
        const std::string staged = StagePayload(caseRoot, stage);
        const fs::path receipt = g_evidence / "arm" / (c.child + ".kv");
        const fs::path hostReceipt = g_evidence / "host-adapter.json";
        const fs::path log = g_evidence / "arm" / (c.child + ".log");

        const std::string arguments = "--child " + c.child + " --root " + Quote(Narrow(caseRoot.wstring())) +
                                      " --stub " + Quote(Narrow(g_stub.wstring())) + " --out " +
                                      Quote(Narrow(receipt.wstring())) + " --host-out " +
                                      Quote(Narrow(hostReceipt.wstring()));

        const int exitCode = RunChildProcess(OwnExecutablePath(), arguments, log);
        const auto values = ReadReceipt(receipt);

        const std::string status = ValueOf(values, "status");
        const std::string detail = ValueOf(values, "detail");
        const std::string attempts = ValueOf(values, "load_attempts");
        const std::string iniWritten = ValueOf(values, "ini_written");
        const std::string secondStatus = ValueOf(values, "second_status");
        const std::string attemptsAfterSecond = ValueOf(values, "load_attempts_after_second");

        if (c.hostProbe)
        {
            const std::string probe = ReadText(hostReceipt);
            const std::string adapterName = ValueOf(values, "adapter_name");

            Check(c.child, "child ran", "0", std::to_string(exitCode));
            Check(c.child, "status on this host", "Ineligible", status);
            Check(c.child, "detail names an SM89 (Ada) or uncorrelated class", "true",
                  detail.find("SM89 (Ada)") != std::string::npos || detail.find("not correlated") != std::string::npos
                      ? "true"
                      : "false");
            Check(c.child, "load attempts", "0", attempts);
            Check(c.child, "companion INI written", "false", iniWritten);
            Check(c.child, "real adapter was enumerated", "true",
                  adapterName.empty() ? "false" : "true");
            Check(c.child, "probe read a DXGI factory", "true",
                  probe.find("\"dxgiOk\": true") != std::string::npos ? "true" : "false");
            Check(c.child, "probe read an NVAPI architecture", "true",
                  probe.find("\"nvapiOk\": true") != std::string::npos ? "true" : "false");

            g_armJson.push_back("  {\"case\": \"" + JsonEscape(c.child) + "\", \"name\": \"" + JsonEscape(c.name) +
                                "\", \"exitCode\": " + std::to_string(exitCode) + ", \"status\": \"" +
                                JsonEscape(status) + "\", \"expectedStatus\": \"Ineligible\", \"detail\": \"" +
                                JsonEscape(detail) + "\", \"loadAttempts\": " +
                                (attempts == "<missing>" ? "-1" : attempts) +
                                ", \"expectedLoadAttempts\": 0, \"iniWritten\": " +
                                (iniWritten == "true" ? "true" : "false") + ", \"adapterName\": \"" +
                                JsonEscape(adapterName) + "\"}");
            g_loadCounts.push_back(c.child + " load_attempts=" + attempts + " expected=0");
            continue;
        }

        Check(c.child, "child ran", "0", std::to_string(exitCode));
        Check(c.child, "status", c.expectedStatus, status);

        if (!c.expectedArmDetail.empty())
            Check(c.child, "detail", c.expectedArmDetail, detail);
        else
            Check(c.child, "detail shape", c.expectedArmDetailPrefix + "...",
                  detail.rfind(c.expectedArmDetailPrefix, 0) == 0 ? c.expectedArmDetailPrefix + "..."
                                                                  : detail.substr(0, 80));

        Check(c.child, "load attempts", std::to_string(c.expectedAttempts), attempts);
        Check(c.child, "companion INI written", c.expectedIni ? "true" : "false", iniWritten);

        if (c.secondArm)
        {
            Check(c.child, "second Arm() keeps the status", c.expectedStatus, secondStatus);
            Check(c.child, "second Arm() does not attempt another load", std::to_string(c.expectedAttempts),
                  attemptsAfterSecond);
        }

        g_armJson.push_back("  {\"case\": \"" + JsonEscape(c.child) + "\", \"name\": \"" + JsonEscape(c.name) +
                            "\", \"exitCode\": " + std::to_string(exitCode) + ", \"status\": \"" + JsonEscape(status) +
                            "\", \"expectedStatus\": \"" + JsonEscape(c.expectedStatus) + "\", \"detail\": \"" +
                            JsonEscape(detail) + "\", \"expectedDetail\": \"" +
                            JsonEscape(!c.expectedArmDetail.empty() ? c.expectedArmDetail
                                                                    : c.expectedArmDetailPrefix + "...") +
                            "\", \"loadAttempts\": " + (attempts == "<missing>" ? "-1" : attempts) +
                            ", \"expectedLoadAttempts\": " + std::to_string(c.expectedAttempts) + ", \"iniWritten\": " +
                            (iniWritten == "true" ? "true" : "false") + ", \"stagedPayload\": \"" + JsonEscape(staged) +
                            "\"}");
        g_loadCounts.push_back(c.child + " load_attempts=" + attempts +
                               " expected=" + std::to_string(c.expectedAttempts));
    }

    // ------------------------------------------------------------------
    // Receipts
    // ------------------------------------------------------------------
    {
        std::string json = "{\n \"schema\": 1,\n \"tool\": \"tests/ampere_mfg_eligibility_smoke.cpp\",\n";
        json += " \"failures\": " + std::to_string(g_failures) + ",\n";
        json += " \"rows\": [\n";

        for (size_t i = 0; i < g_matrixJson.size(); ++i)
            json += g_matrixJson[i] + (i + 1 == g_matrixJson.size() ? "" : ",") + "\n";

        json += " ]\n}\n";
        WriteText(g_evidence / "eligibility-matrix.json", json);

        std::string arms = "{\n \"schema\": 1,\n \"tool\": \"tests/ampere_mfg_eligibility_smoke.cpp\",\n";
        arms += " \"failures\": " + std::to_string(g_failures) + ",\n";
        arms += " \"cases\": [\n";

        for (size_t i = 0; i < g_armJson.size(); ++i)
            arms += g_armJson[i] + (i + 1 == g_armJson.size() ? "" : ",") + "\n";

        arms += " ]\n}\n";
        WriteText(g_evidence / "eligibility-arm-cases.json", arms);

        std::string counts = "load-attempt probe: every refused row must leave the count at zero, and the gate\n";
        counts += "itself must never attempt a load\n";

        for (const auto& line : g_loadCounts)
            counts += line + "\n";

        WriteText(g_evidence / "load-counts.txt", counts);

        std::string rows;

        for (const auto& row : g_rows)
        {
            rows += std::string(row.ok ? "pass " : "FAIL ") + row.caseId + " " + row.name + " expected[" + row.expected +
                    "] observed[" + row.observed + "]\n";
        }

        rows += "\nchecks=" + std::to_string(g_rows.size()) + " failures=" + std::to_string(g_failures) +
                " refusalRowFailures=" + std::to_string(g_refusalFailures) +
                " allowedRowFailures=" + std::to_string(g_allowedFailures) + "\n";
        WriteText(g_evidence / "eligibility-rows.txt", rows);
    }

    std::printf("\nchecks=%d failures=%d refusalRowFailures=%d allowedRowFailures=%d\n", (int) g_rows.size(), g_failures,
                g_refusalFailures, g_allowedFailures);

    if (g_expectRed)
    {
        // The RED revision is the gate surface with the refusal branches removed: the harness must fail on the
        // refusal expectations and nowhere else. Exit 3 says exactly that.
        if (g_failures == 0)
        {
            std::printf("RED NOT CONFIRMED: the permissive gate passed every row - the matrix is blind\n");
            return 2;
        }

        if (g_allowedFailures == 0)
        {
            std::printf("RED CONFIRMED: %d failure(s), all on refusal expectations\n", g_failures);
            return 3;
        }

        std::printf("RED CONFIRMED BUT TOO BROAD: %d failure(s) on rows the RED revision should not affect\n",
                    g_allowedFailures);
        return 4;
    }

    return g_failures == 0 ? 0 : 1;
}
