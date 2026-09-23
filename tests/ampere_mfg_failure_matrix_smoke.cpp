// ============================================================================
// tests/ampere_mfg_failure_matrix_smoke.cpp
// Todo 11 of .omo/plans/rtx2030-mfg-integration.md: the loader failure matrix and the idempotence tests.
//
// What this harness is
// --------------------
// The production loader FILE is the code under test. tests/ampere_mfg_eligibility_mocks.h replaces the four
// sources AmpereMfgLoader.cpp reads the host from - the restored config surface, IdentifyGpu's adapter list,
// MfgUnlock's session latch and State's FG output - and the empty headers in
// tests/ampere_mfg_eligibility_seams/ stop the real ones from being pulled in; everything else is shipped code,
// compiled here with /DOPTISCALER_RTX40_MFG. This is the todo-5 seam pattern, reused verbatim.
//
// Every case runs the real AmpereMfgLoader::Arm() in its own CHILD PROCESS (the loader latches once per
// process, so one case per process is the only way to observe a first arm), against a scratch package root the
// parent stages. Nothing here executes GPU code, loads the pinned 30 MB payload, launches a game or touches a
// live install; the module files are the todo-2 stub DLLs (active role 1, tests/ampere_mfg_stub_payload.cpp;
// standby role 2, tests/ampere_mfg_stub_payload_standby.cpp) plus deliberately corrupted derivatives of them.
//
// Cases (one child process each)
// ------------------------------
//   payload/file work (eligible SM86 host, External frame generation on)
//     m01-missing-payload          no module under either candidate folder
//     m02-tiny-module              1-byte file where the module goes
//     m03-not-a-pe                 a text file named dlssg_sm86.dll
//     m04-wrong-size-truncated     the active stub cut to 60%: the wrong size, refused by the pin rung
//     m05-wrong-hash-same-size     the active stub zero-padded to the PIN byte count: the pinned size, a
//                                  different digest, refused by the pin rung
//     m06-standby-role             the role-2 stub: PayloadStandby, a named refusal
//     m07-flat-layout              module under the second candidate folder (<root>\dlssg_sm86\)
//     m08-ini-unwritable           a DIRECTORY occupies dlssg_sm86.ini: the companion INI cannot be written
//     m09-ini-truncated-preexisting a 45-byte truncated INI is already there: the loader must replace it whole
//     m10-ini-foreign-schema       a foreign 0.2.4-native INI is already there: same, and no foreign key survives
//     m11-valid-control            the control row: Loaded, the INI equals the generated schema, module untouched
//   kernel-image selector (through the production HostOptions -> Arm() -> INI path)
//     k01-kernel-auto              unset: the key is left out
//     k02-kernel-ptx               KernelImage=PTX is written
//     k03-kernel-cubin             KernelImage=Cubin is written
//     k04-kernel-foreign-value     an unrecognised value falls back to Auto: the key is left out
//     k05-kernel-lowercase         "ptx" is not "PTX": the selector is case-sensitive, the key is left out
//   conflicting owners (eligible SM86 host)
//     c01-conflict-ada-unlock, c02-conflict-other-owner, c03-conflict-external-off, c04-conflict-optiscaler-dlssg
//   unsupported adapters
//     u01-ineligible-sm80, u02-ineligible-gtx16, u03-ineligible-arch-missing, u04-ineligible-mixed,
//     u05-ineligible-unknown-arch
//   idempotence and the restart latch
//     i01-repeat-init-idempotent   INI tampered between two Arm() calls: the latch neither reloads nor rewrites
//     i02-restart-latch-options    the second Arm() asks for different settings: they are ignored until restart
//   unit-level header checks (no child process): the kernel-image emission rule of ClampKernelImage/FormatIniContent
//
// The pin rung (todo 13) is what the wrong-size and wrong-hash rows exercise: the production ladder validates
// "a plausible PE image" (AmpereMfgLoader.cpp, LooksLikeModule) AND the PIN.json size and digest
// (CheckPinnedPayload), and refuses a module that is not the pinned payload with PayloadIncomplete (wrong size)
// or PayloadDigestMismatch (the pinned size, different bytes) before the companion INI is written and before
// LoadLibrary. The two fixtures keep their role as the corrupted modules; what changed is the ladder's answer.
// The rows are pinned to the bytes their case treats as the pinned payload through
// tests/ampere_mfg_payload_pin_seam.h: the pristine stub for the stub fixtures, and the REAL
// vendor/dlssg_sm86/dlssg_sm86.dll for m05, which is what keeps that fixture "the pinned size, a different
// digest" (its size comes from the same file). The stub the shipped constants describe cannot load in a test
// process, so the harness hashes the pristine module it pins to instead; tools/check_payload_pin.py
// --expect-digest-from-header proves the constants themselves equal PIN.json.
//
// Exit codes (the parent)
// -----------------------
//   0 = every case behaved as specified
//   1 = at least one case failed
//   2 = usage/IO error, or --broken-fixture was given and every case still passed (a blind matrix)
//   3 = --broken-fixture: the deliberately broken fixture failed as designed, and nothing else failed
//   4 = --broken-fixture: failures appeared on cases the broken fixture cannot explain
//
// Exit codes (a child process)
// ----------------------------
//   0 = Arm() reached Disabled/Ineligible/Conflict/Loaded (the non-failure ladder)
//   3 = Arm() reached a named failure state (PayloadMissing/PayloadIncomplete/PayloadDigestMismatch/
//       PayloadLoadFailed/PayloadStandby/IniWriteFailed) - the corrupted fixtures are designed to produce
//       exactly this nonzero
//   1 = the child could not run the case at all
//
// Usage
// -----
//   ampere_mfg_failure_matrix_smoke.exe --evidence <dir> --scratch <dir> --stub <active stub dll>
//       --stub-standby <standby stub dll> --pinned-module <vendor payload dll> [--broken-fixture]
//   ampere_mfg_failure_matrix_smoke.exe --child <case> --root <scratch case dir> --out <receipt kv>
// ============================================================================

#include "ampere_mfg_eligibility_mocks.h"

#include <bcrypt.h>

// The production loader, verbatim (resolved through /I "%LOADER_DIR%").
#include "AmpereMfgLoader.cpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

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
        default: escaped.push_back(c); break;
        }
    }

    return escaped;
}

std::string ReadBytes(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open())
        return {};

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string ReadText(const fs::path& path)
{
    return ReadBytes(path);
}

void WriteBytes(const fs::path& path, const std::string& bytes)
{
    std::error_code error;
    fs::create_directories(path.parent_path(), error);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteText(const fs::path& path, const std::string& text)
{
    WriteBytes(path, text);
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

std::vector<std::pair<std::string, std::string>> ReadReceipt(const fs::path& path)
{
    std::vector<std::pair<std::string, std::string>> values;

    for (const auto& line : Lines(ReadText(path)))
    {
        const size_t separator = line.find('=');

        if (separator == std::string::npos)
            continue;

        values.emplace_back(line.substr(0, separator), line.substr(separator + 1));
    }

    return values;
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
    wchar_t path[MAX_PATH * 4] = {};
    const DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH * 4);
    return fs::path(std::wstring(path, length));
}

// SHA-256 through CNG. The child hashes the companion INI it is looking at; the parent hashes the same file
// itself, and the runner cross-checks the digest with PowerShell's Get-FileHash, so the "identical INI
// sha256" claim never rests on one implementation.
std::string Sha256Hex(const std::string& data, bool& ok)
{
    ok = false;

    BCRYPT_ALG_HANDLE algorithm = nullptr;

    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0)
        return {};

    BCRYPT_HASH_HANDLE hash = nullptr;

    if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) == 0)
    {
        bool hashed = true;

        if (!data.empty())
            hashed = BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(data.data())),
                                    static_cast<ULONG>(data.size()), 0) == 0;

        if (hashed)
        {
            unsigned char digest[32] = {};
            bool finished = BCryptFinishHash(hash, digest, sizeof(digest), 0) == 0;
            ok = finished;

            static const char* kHex = "0123456789abcdef";
            std::string hex;
            hex.reserve(64);

            for (const unsigned char byte : digest)
            {
                hex.push_back(kHex[byte >> 4]);
                hex.push_back(kHex[byte & 0x0F]);
            }

            BCryptDestroyHash(hash);
            BCryptCloseAlgorithmProvider(algorithm, 0);
            return hex;
        }

        BCryptDestroyHash(hash);
    }

    BCryptCloseAlgorithmProvider(algorithm, 0);
    return {};
}

std::string Sha256OfFile(const fs::path& path, bool& ok)
{
    ok = false;

    if (!fs::exists(path))
        return {};

    return Sha256Hex(ReadBytes(path), ok);
}

// The line that carries `key=` in an INI, or "absent".
std::string IniLine(const std::string& content, const std::string& key)
{
    for (const auto& line : Lines(content))
    {
        if (line.rfind(key, 0) == 0)
            return line;
    }

    return "absent";
}

// A named failure state: the ladder rungs the corrupted fixtures are designed to reach. Everything else
// (Disabled/Ineligible/Conflict/Loaded) is reported with a zero child exit.
bool IsFailureStatus(const std::string& name)
{
    return name == "PayloadMissing" || name == "PayloadIncomplete" || name == "PayloadDigestMismatch" ||
           name == "PayloadLoadFailed" || name == "PayloadStandby" || name == "IniWriteFailed";
}

bool IsFailureStatus(AmpereMfgLoader::State state)
{
    return IsFailureStatus(std::string(AmpereMfgLoader::StateName(state)));
}

constexpr const char* kBrokenCase = "m11-valid-control";

// ---------------------------------------------------------------------------
// The case table: one entry per child process
// ---------------------------------------------------------------------------
enum class ModuleKind
{
    Active,    // the todo-2 stub, DlssgProxy_Role=1
    Standby,   // the role-2 stub fixture
    Tiny,      // 1 byte
    Text,      // not a PE
    Truncated, // the active stub cut to 60%: MZ header, no valid image
    Padded,    // the active stub zero-padded to the PIN byte count: loads, digest differs
    Missing,   // nothing staged
};

enum class PreIni
{
    None,
    Truncated, // a 45-byte partial file where the companion INI goes
    Foreign,   // a foreign 0.2.4-native-schema INI where the companion INI goes
};

struct Case
{
    std::string id;
    std::string name;

    ModuleKind module = ModuleKind::Active;
    PreIni preIni = PreIni::None;
    bool iniUnwritable = false; // a DIRECTORY named dlssg_sm86.ini
    bool flatLayout = false;    // module under <root>\dlssg_sm86\ instead of <root>\OptiScaler\dlssg_sm86\

    // Todo 13: the pristine bytes this case pins the ladder to. The default is the active stub (the fixture the
    // case-derived module came from); the standby row pins its own stub; m05 pins the REAL PIN.json payload, so
    // its padded fixture keeps the pinned size and differs only in the digest.
    bool pinIsShippedPayload = false;

    std::vector<GpuInformation> adapters;
    bool enabled = true;
    bool externalFg = true;
    bool adaUnlock = false;
    bool optiFgOutput = false;
    bool otherOwner = false; // the child loads the stub under the shipped name before Arm()

    std::string configKernelImage; // "" = unset: the production read falls back to "Auto"
    int configMaxFrames = 3;

    bool secondArm = false;
    bool tamperIni = false;      // overwrite the companion INI between the two calls
    bool secondDiffers = false;  // the second Arm() asks for different settings
    int secondMaxFrames = 3;
    std::string secondKernelImage = "Cubin";

    // Expectations
    std::string expectedStatus;
    std::string expectedDetail;          // exact, when non-empty
    std::string expectedDetailContains;  // substring, when non-empty
    int expectedAttempts = 0;
    bool expectedIni = false;

    bool assertSecond = false;
    bool assertKernelLine = false;
    std::string expectedKernelLine; // "absent" or "KernelImage=PTX"
    bool assertMaxFramesLine = false;
    std::string expectedMaxFramesLine;
    bool assertFirstGolden = false;        // the INI the loader wrote equals FormatIniContent(first settings)
    bool assertSecondGolden = false;       // ... after the second arm, still the first settings (the latch)
    bool assertSecondSecondGolden = false; // ... and it does NOT equal the second settings
    bool assertTamperSurvived = false;
    bool assertForeignAbsent = false;
    bool assertTruncatedMarkerAbsent = false;
    bool assertModuleUnchanged = false;
    int assertAttachLines = -1; // -1 = not asserted
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

AmpereMfgLoader::IniSettings SettingsOf(const Case& c)
{
    AmpereMfgLoader::IniSettings settings;
    settings.MaxGeneratedFrames = c.configMaxFrames;
    settings.KernelImage = c.configKernelImage.empty() ? "Auto" : c.configKernelImage;
    return settings;
}

std::string GoldenIni(const Case& c)
{
    return AmpereMfgLoader::FormatIniContent(SettingsOf(c));
}

std::vector<Case> Cases()
{
    const std::string supported = "; the payload runs on SM75 (RTX 20) or SM86 (RTX 30) only";

    const GpuInformation sm86 = NvidiaRtx("NVIDIA GeForce RTX 3060", AmpereMfgLoader::kArchAmpere, 6);
    const GpuInformation intelIgpu = Adapter("Intel(R) UHD Graphics 630", VendorId::Intel, 0, 0);

    std::vector<Case> cases;

    auto mk = [&cases, &sm86](const std::string& id, const std::string& name)
    {
        Case c;
        c.id = id;
        c.name = name;
        c.adapters = { sm86 };
        cases.push_back(std::move(c));
    };

    // ------------------------------------------------------------------
    // Payload discovery, validation, the companion INI and the load
    // ------------------------------------------------------------------
    mk("m01-missing-payload", "no payload module staged");
    cases.back().module = ModuleKind::Missing;
    cases.back().expectedStatus = "PayloadMissing";
    cases.back().expectedDetailContains = "the payload module dlssg_sm86.dll was not found under";
    cases.back().expectedAttempts = 1;

    mk("m02-tiny-module", "a 1-byte file where the module goes");
    cases.back().module = ModuleKind::Tiny;
    cases.back().expectedStatus = "PayloadIncomplete";
    cases.back().expectedDetailContains = "is not a usable module: the file is smaller than a PE header";
    cases.back().expectedAttempts = 1;

    mk("m03-not-a-pe", "a text file named dlssg_sm86.dll");
    cases.back().module = ModuleKind::Text;
    cases.back().expectedStatus = "PayloadIncomplete";
    cases.back().expectedDetailContains = "is not a usable module: the file does not start with an MZ header";
    cases.back().expectedAttempts = 1;

    mk("m04-wrong-size-truncated", "the active stub cut to 60%: the wrong size, refused by the pin rung");
    cases.back().module = ModuleKind::Truncated;
    cases.back().expectedStatus = "PayloadIncomplete";
    cases.back().expectedDetailContains = "is not the pinned payload: the module is";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = false;   // the pin rung runs before the companion INI is written
    cases.back().assertAttachLines = 0; // the image is never mapped

    mk("m05-wrong-hash-same-size", "the active stub padded to the PIN byte count: the pinned size, a different digest");
    cases.back().module = ModuleKind::Padded;
    cases.back().pinIsShippedPayload = true;
    cases.back().expectedStatus = "PayloadDigestMismatch";
    cases.back().expectedDetailContains =
        "and the pinned digest is c3934a09399f022504227c72df0bf8c0de55f9a08880dddde898c5262cefa838";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = false;   // the pin rung runs before the companion INI is written
    cases.back().assertAttachLines = 0; // the image is never mapped

    mk("m06-standby-role", "the role-2 stub: Loaded as a standby proxy, a named refusal");
    cases.back().module = ModuleKind::Standby;
    cases.back().expectedStatus = "PayloadStandby";
    cases.back().expectedDetailContains = "the payload loaded as a standby proxy (DlssgProxy_Role=2";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true; // the INI is written before the load, and it stays
    cases.back().assertFirstGolden = true;
    cases.back().assertAttachLines = 1;

    mk("m07-flat-layout", "the module under the second candidate folder (<root>\\dlssg_sm86\\)");
    cases.back().flatLayout = true;
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedDetailContains = "m07-flat-layout\\dlssg_sm86\\dlssg_sm86.dll";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertAttachLines = 1;

    mk("m08-ini-unwritable", "a directory occupies dlssg_sm86.ini: the companion INI cannot be written");
    cases.back().iniUnwritable = true;
    cases.back().expectedStatus = "IniWriteFailed";
    cases.back().expectedDetailContains = "for writing";
    cases.back().expectedAttempts = 1;
    cases.back().assertAttachLines = 0; // nothing was loaded

    mk("m09-ini-truncated-preexisting", "a 45-byte truncated INI is already there: it is replaced whole");
    cases.back().preIni = PreIni::Truncated;
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedDetailContains = "payload loaded from ";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertTruncatedMarkerAbsent = true;
    cases.back().assertAttachLines = 1;

    mk("m10-ini-foreign-schema", "a foreign 0.2.4-native INI is already there: no foreign key survives");
    cases.back().preIni = PreIni::Foreign;
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedDetailContains = "payload loaded from ";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertForeignAbsent = true;
    cases.back().assertAttachLines = 1;

    mk("m11-valid-control", "the control row: Loaded, generated INI, module bytes untouched");
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedDetailContains = "payload loaded from ";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertKernelLine = true;
    cases.back().expectedKernelLine = "absent";
    cases.back().assertMaxFramesLine = true;
    cases.back().expectedMaxFramesLine = "MaxGeneratedFrames=3";
    cases.back().assertModuleUnchanged = true;
    cases.back().assertAttachLines = 1;

    // ------------------------------------------------------------------
    // The kernel-image selector, through the production config read
    // ------------------------------------------------------------------
    mk("k01-kernel-auto", "KernelImage unset -> Auto: the key is left out");
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertKernelLine = true;
    cases.back().expectedKernelLine = "absent";

    mk("k02-kernel-ptx", "KernelImage=PTX is written as asked");
    cases.back().configKernelImage = "PTX";
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertKernelLine = true;
    cases.back().expectedKernelLine = "KernelImage=PTX";

    mk("k03-kernel-cubin", "KernelImage=Cubin is written as asked");
    cases.back().configKernelImage = "Cubin";
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertKernelLine = true;
    cases.back().expectedKernelLine = "KernelImage=Cubin";

    mk("k04-kernel-foreign-value", "an unrecognised value falls back to Auto: the key is left out");
    cases.back().configKernelImage = "Rubbish";
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertKernelLine = true;
    cases.back().expectedKernelLine = "absent";

    mk("k05-kernel-lowercase", "\"ptx\" is not \"PTX\": the selector is case-sensitive, the key is left out");
    cases.back().configKernelImage = "ptx";
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertKernelLine = true;
    cases.back().expectedKernelLine = "absent";

    // ------------------------------------------------------------------
    // Conflicting frame-generation owners (eligible SM86 host)
    // ------------------------------------------------------------------
    mk("c01-conflict-ada-unlock", "the Ada unlock is on for this session");
    cases.back().adaUnlock = true;
    cases.back().expectedStatus = "Conflict";
    cases.back().expectedDetail = "the Ada (RTX 40) unlock is active for this session; one frame-generation owner "
                                  "only";

    mk("c02-conflict-other-owner", "another unlocker already holds the DLSSG output");
    cases.back().otherOwner = true;
    cases.back().expectedStatus = "Conflict";
    cases.back().expectedDetail = "another unlocker or frame-generation owner is already present: the payload "
                                  "module is loaded without this loader having loaded it";
    cases.back().assertAttachLines = 1; // the child's own load only: Arm() attempted nothing

    mk("c03-conflict-external-off", "External off while the payload needs game-Streamline ownership");
    cases.back().externalFg = false;
    cases.back().expectedStatus = "Conflict";
    cases.back().expectedDetail = "External frame generation does not own this session; enable [FrameGen] External and restart "
                                  "with no active OptiScaler FG selections so the game's Streamline owns frame generation";

    mk("c04-conflict-optiscaler-dlssg", "OptiScaler's own DLSSG output path is active");
    cases.back().optiFgOutput = true;
    cases.back().expectedStatus = "Conflict";
    cases.back().expectedDetail = "OptiScaler's own DLSSG output path is active; it owns the generated frames for "
                                  "this session";

    // ------------------------------------------------------------------
    // Unsupported adapters
    // ------------------------------------------------------------------
    mk("u01-ineligible-sm80", "A100 SM80 (GA100)");
    cases.back().adapters = { NvidiaRtx("NVIDIA A100-SXM4-40GB", AmpereMfgLoader::kArchAmpere, 0) };
    cases.back().expectedStatus = "Ineligible";
    cases.back().expectedDetail = "adapter 'NVIDIA A100-SXM4-40GB' is SM80 (GA100)" + supported;

    mk("u02-ineligible-gtx16", "GTX 1660 SUPER (TU116): SM75 without the tensor units");
    cases.back().adapters = { NvidiaRtx("NVIDIA GeForce GTX 1660 SUPER", AmpereMfgLoader::kArchTuring,
                                        AmpereMfgLoader::kImplTu116) };
    cases.back().expectedStatus = "Ineligible";
    cases.back().expectedDetail = "adapter 'NVIDIA GeForce GTX 1660 SUPER' is a Turing GTX 16 part: SM75 without "
                                  "the RTX 20 tensor units" +
                                  supported;

    mk("u03-ineligible-arch-missing", "an NVIDIA adapter whose architecture id is missing");
    cases.back().adapters = { NvidiaRtx("NVIDIA GPU", 0, 0) };
    cases.back().expectedStatus = "Ineligible";
    cases.back().expectedDetail = "adapter 'NVIDIA GPU' is an NVIDIA part whose architecture id is missing" + supported;

    mk("u04-ineligible-mixed", "two physical adapters and no correlation to the render device");
    cases.back().adapters = { sm86, intelIgpu };
    cases.back().expectedStatus = "Ineligible";
    cases.back().expectedDetail = "the host reports 2 physical adapters and adapter 'NVIDIA GeForce RTX 3060' is "
                                  "not correlated to this process's render device" +
                                  supported;

    mk("u05-ineligible-unknown-arch", "an NVIDIA architecture this build does not know (0x1C0)");
    cases.back().adapters = { NvidiaRtx("NVIDIA GPU", 0x1C0, 0) };
    cases.back().expectedStatus = "Ineligible";
    cases.back().expectedDetail = "adapter 'NVIDIA GPU' is an NVIDIA architecture this build does not recognise" +
                                  supported;

    // ------------------------------------------------------------------
    // Idempotence and the restart latch
    // ------------------------------------------------------------------
    mk("i01-repeat-init-idempotent", "the INI is tampered between two Arm() calls: neither reload nor rewrite");
    cases.back().secondArm = true;
    cases.back().tamperIni = true;
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedDetailContains = "payload loaded from ";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertSecond = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertTamperSurvived = true;
    cases.back().assertAttachLines = 1;

    mk("i02-restart-latch-options", "the second Arm() asks for different settings: ignored until a restart");
    cases.back().configKernelImage = "PTX";
    cases.back().secondArm = true;
    cases.back().secondDiffers = true;
    cases.back().secondMaxFrames = 5;
    cases.back().secondKernelImage = "Cubin";
    cases.back().expectedStatus = "Loaded";
    cases.back().expectedDetailContains = "payload loaded from ";
    cases.back().expectedAttempts = 1;
    cases.back().expectedIni = true;
    cases.back().assertSecond = true;
    cases.back().assertFirstGolden = true;
    cases.back().assertSecondGolden = true;
    cases.back().assertSecondSecondGolden = true;
    cases.back().assertKernelLine = true;
    cases.back().expectedKernelLine = "KernelImage=PTX";
    cases.back().assertMaxFramesLine = true;
    cases.back().expectedMaxFramesLine = "MaxGeneratedFrames=3";
    cases.back().assertAttachLines = 1;

    return cases;
}

const Case* FindCase(const std::vector<Case>& cases, const std::string& id)
{
    for (const auto& c : cases)
    {
        if (c.id == id)
            return &c;
    }

    return nullptr;
}

void ApplyFacts(const Case& c)
{
    IdentifyGpu::SetAdapters(c.adapters);

    auto* config = Config::Instance();
    config->ExternalFrameGeneration.stored = c.externalFg;
    config->FGDLSSGAmpereMfgUnlock.stored = c.enabled;
    config->FGDLSSGAmpereMfgMaxFrames.stored = c.configMaxFrames;

    if (c.configKernelImage.empty())
        config->FGDLSSGAmpereMfgKernelImage.stored = std::nullopt;
    else
        config->FGDLSSGAmpereMfgKernelImage.stored = c.configKernelImage;

    MfgUnlock::g_enabledForSession = c.adaUnlock;

    auto& state = State::Instance();
    state.activeFgOutput = c.optiFgOutput ? FGOutput::DLSSG : FGOutput::NoFG;
    state.activeFgNvngx = c.optiFgOutput ? FGNvngxReplacement::Nukems : FGNvngxReplacement::None;
}

// ---------------------------------------------------------------------------
// Child mode: one case, the real Arm()
// ---------------------------------------------------------------------------
int RunChild(const std::string& id, const fs::path& root, const fs::path& out)
{
    const auto cases = Cases();
    const Case* found = FindCase(cases, id);

    if (found == nullptr)
    {
        std::printf("unknown child case: %s\n", id.c_str());
        return 1;
    }

    const Case c = *found;
    ApplyFacts(c);

    // Todo 13's pin seam: the parent names the pristine bytes THIS case treats as the pinned payload
    // (AMPERE_MFG_PIN_FILE - the active stub, the standby stub, or the real PIN.json payload for m05), and the
    // ladder compares the staged module against that size and digest
    // (tests/ampere_mfg_payload_pin_seam.h).
    if (!AmpereMfgPinSeam::PinFromEnvironment())
    {
        std::printf("the case's pin source could not be read from AMPERE_MFG_PIN_FILE\n");
        return 1;
    }

    if (c.otherOwner)
    {
        // A real module under the shipped name, loaded before Arm(): this is what "another owner already holds
        // the DLSSG output" looks like from inside the process, and the production
        // GetModuleHandleW(kPayloadModuleName) check detects it - no fixture flag involved.
        const fs::path module = root / "OptiScaler" / "dlssg_sm86" / AmpereMfgLoader::kPayloadModuleName;

        if (LoadLibraryW(module.wstring().c_str()) == nullptr)
        {
            std::printf("the owner stub could not be loaded from %s\n", Narrow(module.wstring()).c_str());
            return 1;
        }
    }

    // The module bytes before the arm, through the production discovery, so "the loader does not touch the
    // payload" is asserted on the file the loader itself picked.
    std::vector<fs::path> tried;
    const fs::path moduleBefore = AmpereMfgLoader::FindPayloadModule(root, tried);

    bool moduleHashOk = false;
    const std::string moduleShaBefore = moduleBefore.empty() ? "" : Sha256OfFile(moduleBefore, moduleHashOk);

    AmpereMfgLoader::ArmOptions options = AmpereMfgLoader::HostOptions();
    options.PackageRoot = root;

    const AmpereMfgLoader::Status first = AmpereMfgLoader::Arm(options);
    const int attemptsAfterFirst = AmpereMfgLoader::LoadAttempts();

    // The companion INI exactly as the loader wrote it, captured before anything else touches it, so the
    // tamper case can still assert the loader's own output.
    const fs::path iniPath = first.IniPath;
    const std::string iniAsWritten = iniPath.empty() ? std::string() : ReadBytes(iniPath);

    bool iniWriteObserved = false;

    if (c.tamperIni && !iniAsWritten.empty() && fs::exists(iniPath))
    {
        WriteText(iniPath, "TAMPERED-BY-THE-MATRIX\n");
        iniWriteObserved = true;
    }

    std::string secondStatus;
    int attemptsAfterSecond = attemptsAfterFirst;

    if (c.secondArm)
    {
        AmpereMfgLoader::ArmOptions second = options;

        if (c.secondDiffers)
        {
            second.Ini.MaxGeneratedFrames = c.secondMaxFrames;
            second.Ini.KernelImage = c.secondKernelImage;
        }

        const auto status = AmpereMfgLoader::Arm(second);
        secondStatus = status.StateText();
        attemptsAfterSecond = AmpereMfgLoader::LoadAttempts();
    }

    // The file on disk after the second arm: the latch must have left it exactly as it was.
    const std::string iniFirst = iniPath.empty() ? std::string() : ReadBytes(iniPath);
    const std::string iniSecond = ReadBytes(iniPath);

    bool shaFirstOk = false;
    bool shaSecondOk = false;
    bool shaWrittenOk = false;
    const std::string shaFirst = iniFirst.empty() ? std::string() : Sha256Hex(iniFirst, shaFirstOk);
    const std::string shaSecond = iniSecond.empty() ? std::string() : Sha256Hex(iniSecond, shaSecondOk);
    const std::string shaWritten = iniAsWritten.empty() ? std::string() : Sha256Hex(iniAsWritten, shaWrittenOk);

    const std::string goldenFirst = GoldenIni(c);

    Case secondCase = c;
    secondCase.configMaxFrames = c.secondMaxFrames;
    secondCase.configKernelImage = c.secondKernelImage;
    const std::string goldenSecond = GoldenIni(secondCase);

    bool moduleHashAfterOk = false;
    const std::string moduleShaAfter = moduleBefore.empty() ? "" : Sha256OfFile(moduleBefore, moduleHashAfterOk);

    auto flag = [](bool value) { return std::string(value ? "true" : "false"); };

    std::string receipt;
    receipt += "case=" + id + "\n";
    receipt += "status=" + std::string(first.StateText()) + "\n";
    receipt += "detail=" + first.Detail + "\n";
    receipt += "expected_pin_bytes=" + std::to_string(AmpereMfgLoader::ExpectedPayloadBytes()) + "\n";
    receipt += "expected_pin_sha256=" + AmpereMfgLoader::ExpectedPayloadSha256() + "\n";
    receipt += "module_path=" + (first.ModulePath.empty() ? std::string("<empty>") : Narrow(first.ModulePath)) + "\n";
    receipt += "ini_path=" + (iniPath.empty() ? std::string("<empty>") : Narrow(iniPath.wstring())) + "\n";
    receipt += "ini_written=" + flag(!iniPath.empty() && fs::exists(iniPath)) + "\n";
    receipt += "load_attempts=" + std::to_string(attemptsAfterFirst) + "\n";
    receipt += "second_status=" + (secondStatus.empty() ? std::string("<none>") : secondStatus) + "\n";
    receipt += "load_attempts_after_second=" + std::to_string(attemptsAfterSecond) + "\n";
    receipt += "module_sha256_before=" + (moduleShaBefore.empty() ? std::string("<none>") : moduleShaBefore) + "\n";
    receipt += "module_sha256_after=" + (moduleShaAfter.empty() ? std::string("<none>") : moduleShaAfter) + "\n";
    receipt += "module_unchanged=" + flag(!moduleShaBefore.empty() && moduleShaBefore == moduleShaAfter) + "\n";
    receipt += "ini_bytes_first=" + (iniFirst.empty() ? std::string("<none>") : std::to_string(iniFirst.size())) + "\n";
    receipt += "ini_bytes_written=" + (iniAsWritten.empty() ? std::string("<none>") : std::to_string(iniAsWritten.size())) +
               "\n";
    receipt += "ini_sha256_written=" + (shaWritten.empty() ? std::string("<none>") : shaWritten) + "\n";
    receipt += "ini_sha256_first=" + (shaFirst.empty() ? std::string("<none>") : shaFirst) + "\n";
    receipt += "ini_sha256_second=" + (shaSecond.empty() ? std::string("<none>") : shaSecond) + "\n";
    receipt += "ini_written_matches_golden=" + flag(!iniAsWritten.empty() && iniAsWritten == goldenFirst) + "\n";
    receipt += "ini_first_matches_golden=" + flag(!iniFirst.empty() && iniFirst == goldenFirst) + "\n";
    receipt += "ini_second_matches_golden=" + flag(!iniSecond.empty() && iniSecond == goldenFirst) + "\n";
    receipt += "ini_second_matches_second_golden=" + flag(!iniSecond.empty() && iniSecond == goldenSecond) + "\n";
    receipt += "kernel_image_line=" + IniLine(iniFirst, "KernelImage=") + "\n";
    receipt += "maxframes_line=" + IniLine(iniFirst, "MaxGeneratedFrames=") + "\n";
    receipt += "foreign_keys_present_after=" + flag(iniSecond.find("[Native]") != std::string::npos ||
                                                    iniSecond.find("MaxGeneratedFrames=99") != std::string::npos ||
                                                    iniSecond.find("Router=Nope") != std::string::npos) +
               "\n";
    receipt += "truncated_marker_present_after=" +
               flag(iniSecond.find("TRUNCATED-PRE-EXISTING") != std::string::npos) + "\n";
    receipt += "tamper_survived=" +
               flag(c.tamperIni && iniWriteObserved && iniSecond == "TAMPERED-BY-THE-MATRIX\n") + "\n";
    receipt += "exit_class=" + std::string(IsFailureStatus(first.StateText()) ? "failure" : "ok") + "\n";

    WriteText(out, receipt);
    std::printf("%s", receipt.c_str());

    return IsFailureStatus(first.StateText()) ? 3 : 0;
}

// ---------------------------------------------------------------------------
// Fixture staging (parent)
// ---------------------------------------------------------------------------
struct Fixtures
{
    fs::path active;
    fs::path standby;
    size_t pinnedBytes = 0;
    std::string activeBytes;
    std::string standbyBytes;
};

std::string StageModule(const fs::path& folder, const Case& c, bool broken, const Fixtures& f)
{
    const fs::path target = folder / AmpereMfgLoader::kPayloadModuleName;

    if (broken)
    {
        WriteText(target, "this is not a PE image\n");
        return Narrow(target.wstring());
    }

    switch (c.module)
    {
    case ModuleKind::Missing: return "<not staged>";
    case ModuleKind::Active: WriteBytes(target, f.activeBytes); break;
    case ModuleKind::Standby: WriteBytes(target, f.standbyBytes); break;
    case ModuleKind::Tiny: WriteBytes(target, std::string(1, 'M')); break;
    case ModuleKind::Text: WriteText(target, "this is not a PE image\n"); break;
    case ModuleKind::Truncated: WriteBytes(target, f.activeBytes.substr(0, f.activeBytes.size() * 6 / 10)); break;
    case ModuleKind::Padded:
    {
        std::string padded = f.activeBytes;
        padded.resize(f.pinnedBytes, '\0');
        WriteBytes(target, padded);
        break;
    }
    }

    return Narrow(target.wstring());
}

void StagePreIni(const fs::path& folder, const Case& c)
{
    const fs::path ini = folder / AmpereMfgLoader::kPayloadIniName;

    if (c.iniUnwritable)
    {
        // A directory where the companion INI goes: ofstream cannot open it, so the loader reports
        // IniWriteFailed instead of writing a payload-visible file.
        std::error_code error;
        fs::create_directory(ini, error);
        return;
    }

    if (c.preIni == PreIni::Truncated)
    {
        WriteText(ini, "; dlssg_sm86.ini TRUNCATED-PRE-EXISTING marker: an interrupted copy left this behind\n"
                       "[General]\nEna");
        return;
    }

    if (c.preIni == PreIni::Foreign)
    {
        WriteText(ini, "; dlssg_sm86.ini FOREIGN-PRE-EXISTING marker: a 0.2.4 native-schema file\n"
                       "[Native]\n"
                       "Enabled=1\n"
                       "\n[FrameGeneration]\n"
                       "MaxGeneratedFrames=99\n"
                       "\n[Compatibility]\n"
                       "Router=Nope\n");
    }
}

std::string StageCase(const fs::path& root, const Case& c, bool broken, const Fixtures& f)
{
    const fs::path folder = c.flatLayout ? root / "dlssg_sm86" : root / "OptiScaler" / "dlssg_sm86";
    const std::string module = StageModule(folder, c, broken, f);

    if (!broken)
        StagePreIni(folder, c);

    return module;
}

int CountLinesWith(const fs::path& path, const std::string& needle)
{
    if (!fs::exists(path))
        return 0;

    int count = 0;

    for (const auto& line : Lines(ReadText(path)))
    {
        if (line.find(needle) != std::string::npos)
            ++count;
    }

    return count;
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

    const DWORD wait = WaitForSingleObject(process.hProcess, 60000);
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
// The receipt of a run
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
std::vector<std::string> g_caseJson;
std::string g_table;

// ---------------------------------------------------------------------------
// Unit-level checks: the kernel-image emission rule of the header
// ---------------------------------------------------------------------------
void Check(const std::string& caseId, const std::string& name, const std::string& expected, const std::string& observed)
{
    const bool ok = expected == observed;
    g_rows.push_back({ caseId, name, expected, observed, ok });

    if (ok)
        std::printf("pass %-30s %-58s = %s\n", caseId.c_str(), name.c_str(), observed.c_str());
    else
        std::printf("FAIL %-30s %-58s expected [%s] observed [%s]\n", caseId.c_str(), name.c_str(), expected.c_str(),
                    observed.c_str());
}

// The emission rule of ClampKernelImage/FormatIniContent, at unit level: PTX and Cubin are written as asked, and
// Auto or anything unrecognised (including the wrong case, which the schema does not accept) leaves the key out
// entirely instead of writing a value the payload would reject.
void RunUnitChecks(const fs::path& evidence)
{
    std::string table = "unit-level: the kernel-image selector and its emission rule (header functions, no child)\n"
                        "pass/FAIL | check | expected | observed\n\n";

    auto row = [&table](const std::string& name, const std::string& expected, const std::string& observed)
    {
        Check("unit-kernel-image", name, expected, observed);
        table += std::string(expected == observed ? "pass" : "FAIL") + " | " + name + " | " + expected + " | " +
                 observed + "\n";
    };

    auto emitted = [](const std::string& value)
    {
        AmpereMfgLoader::IniSettings settings;
        settings.KernelImage = value;
        return IniLine(AmpereMfgLoader::FormatIniContent(settings), "KernelImage=");
    };

    row("ClampKernelImage(\"Auto\")", "", AmpereMfgLoader::ClampKernelImage("Auto"));
    row("ClampKernelImage(\"PTX\")", "PTX", AmpereMfgLoader::ClampKernelImage("PTX"));
    row("ClampKernelImage(\"Cubin\")", "Cubin", AmpereMfgLoader::ClampKernelImage("Cubin"));
    row("ClampKernelImage(\"ptx\") falls back", "", AmpereMfgLoader::ClampKernelImage("ptx"));
    row("ClampKernelImage(\"cubin\") falls back", "", AmpereMfgLoader::ClampKernelImage("cubin"));
    row("ClampKernelImage(\"Rubbish\") falls back", "", AmpereMfgLoader::ClampKernelImage("Rubbish"));
    row("ClampKernelImage(\"\") falls back", "", AmpereMfgLoader::ClampKernelImage(""));
    row("FormatIniContent(KernelImage=\"Auto\") emits no key", "absent", emitted("Auto"));
    row("FormatIniContent(KernelImage=\"PTX\") emits KernelImage=PTX", "KernelImage=PTX", emitted("PTX"));
    row("FormatIniContent(KernelImage=\"Cubin\") emits KernelImage=Cubin", "KernelImage=Cubin", emitted("Cubin"));
    row("FormatIniContent(KernelImage=\"ptx\") emits no key", "absent", emitted("ptx"));
    row("FormatIniContent(KernelImage=\"Rubbish\") emits no key", "absent", emitted("Rubbish"));
    row("FormatIniContent(KernelImage=\"\") emits no key", "absent", emitted(""));

    // The key sits under [Compatibility], right after Router, and the other keys keep their defaults: the
    // factory 0.3.5 schema has no KernelImage key at all, which is what "omitted" has to mean on disk.
    AmpereMfgLoader::IniSettings autoSettings;
    const std::string autoIni = AmpereMfgLoader::FormatIniContent(autoSettings);
    row("the Auto INI keeps Router=Auto", "Router=Auto", IniLine(autoIni, "Router="));
    row("the Auto INI keeps the factory frame count", "MaxGeneratedFrames=3", IniLine(autoIni, "MaxGeneratedFrames="));

    table += "\nchecks=" + std::to_string(g_rows.size()) + "\n";
    WriteText(evidence / "unit-kernel-selector.txt", table);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
} // namespace

int main(int argc, char** argv)
{
    fs::path evidence;
    fs::path scratch;
    fs::path stub;
    fs::path stubStandby;
    fs::path pinnedModule;
    std::string child;
    fs::path childRoot;
    fs::path childOut;
    bool brokenFixture = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };

        if (argument == "--evidence")
            evidence = next();
        else if (argument == "--scratch")
            scratch = next();
        else if (argument == "--stub")
            stub = next();
        else if (argument == "--stub-standby")
            stubStandby = next();
        else if (argument == "--pinned-module")
            pinnedModule = next();
        else if (argument == "--child")
            child = next();
        else if (argument == "--root")
            childRoot = next();
        else if (argument == "--out")
            childOut = next();
        else if (argument == "--broken-fixture")
            brokenFixture = true;
        else
        {
            std::printf("unknown argument: %s\n", argument.c_str());
            return 2;
        }
    }

    if (!child.empty())
        return RunChild(child, childRoot, childOut);

    if (evidence.empty() || scratch.empty() || stub.empty() || stubStandby.empty() || pinnedModule.empty())
    {
        std::printf("--evidence <dir> --scratch <dir> --stub <dll> --stub-standby <dll> --pinned-module <dll> are "
                    "required\n");
        return 2;
    }

    if (!fs::exists(stub) || !fs::exists(stubStandby) || !fs::exists(pinnedModule))
    {
        std::printf("a fixture is missing: stub=%s standby=%s pinned=%s\n", stub.string().c_str(),
                    stubStandby.string().c_str(), pinnedModule.string().c_str());
        return 2;
    }

    std::error_code error;
    fs::create_directories(evidence, error);
    fs::create_directories(scratch, error);
    fs::create_directories(evidence / "cases", error);

    Fixtures fixtures;
    fixtures.active = stub;
    fixtures.standby = stubStandby;
    fixtures.activeBytes = ReadBytes(stub);
    fixtures.standbyBytes = ReadBytes(stubStandby);
    fixtures.pinnedBytes = static_cast<size_t>(fs::file_size(pinnedModule, error));

    if (fixtures.activeBytes.empty() || fixtures.standbyBytes.empty() || fixtures.pinnedBytes == 0)
    {
        std::printf("a fixture could not be read (stub bytes %zu, standby bytes %zu, pinned bytes %zu)\n",
                    fixtures.activeBytes.size(), fixtures.standbyBytes.size(), fixtures.pinnedBytes);
        return 2;
    }

    // ------------------------------------------------------------------
    // 1. Unit-level header checks (no child process)
    // ------------------------------------------------------------------
    RunUnitChecks(evidence);

    // ------------------------------------------------------------------
    // 2. One child process per case
    // ------------------------------------------------------------------
    const auto cases = Cases();
    int failures = 0;
    int brokenCaseFailures = 0;
    int otherFailures = 0;

    std::string counts = "load-attempt probe and attach probe: a refusal must attempt no load and attach nothing;\n"
                         "the latch must keep one attempt and one attach across repeated Arm() calls\n";

    std::printf("=== loader failure matrix (production Arm(), one child process per case)\n");

    for (const auto& c : cases)
    {
        const fs::path caseRoot = scratch / c.id;
        fs::remove_all(caseRoot, error);
        fs::create_directories(caseRoot, error);

        const bool broken = brokenFixture && c.id == kBrokenCase;
        const std::string staged = StageCase(caseRoot, c, broken, fixtures);

        // The stub's own attach log, so "the payload was mapped once" is the payload's evidence, not the
        // loader's bookkeeping.
        const fs::path attachLog = caseRoot / "payload-attach.log";
        SetEnvironmentVariableW(L"AMPERE_MFG_STUB_LOG", attachLog.wstring().c_str());

        // Todo 13: the pin source this case hands the ladder (the seam header explains why the child cannot
        // derive it from the staged bytes itself - m05 stages the corrupted module).
        const fs::path pinSource =
            c.pinIsShippedPayload ? pinnedModule : (c.module == ModuleKind::Standby ? stubStandby : stub);
        SetEnvironmentVariableW(L"AMPERE_MFG_PIN_FILE", pinSource.wstring().c_str());

        const fs::path receipt = evidence / "cases" / (c.id + ".kv");
        const fs::path log = evidence / "cases" / (c.id + ".log");

        const std::string arguments = "--child " + c.id + " --root " + Quote(Narrow(caseRoot.wstring())) + " --out " +
                                      Quote(Narrow(receipt.wstring()));

        const int exitCode = RunChildProcess(OwnExecutablePath(), arguments, log);
        const auto values = ReadReceipt(receipt);

        const std::string status = ValueOf(values, "status");
        const std::string detail = ValueOf(values, "detail");
        const std::string attempts = ValueOf(values, "load_attempts");
        const std::string attemptsSecond = ValueOf(values, "load_attempts_after_second");
        const std::string secondStatus = ValueOf(values, "second_status");
        const std::string iniWritten = ValueOf(values, "ini_written");
        const std::string iniPath = ValueOf(values, "ini_path");
        const std::string iniShaFirst = ValueOf(values, "ini_sha256_first");
        const std::string iniShaSecond = ValueOf(values, "ini_sha256_second");
        const std::string iniBytesFirst = ValueOf(values, "ini_bytes_first");
        const std::string modulePath = ValueOf(values, "module_path");
        const int expectedExit = IsFailureStatus(c.expectedStatus) ? 3 : 0;

        bool caseFailed = false;

        auto check = [&](const std::string& name, const std::string& expected, const std::string& observed)
        {
            const bool ok = expected == observed;
            g_rows.push_back({ c.id, name, expected, observed, ok });
            std::printf("%s %-30s %-56s = %s\n", ok ? "pass" : "FAIL", c.id.c_str(), name.c_str(), observed.c_str());

            if (!ok)
            {
                ++failures;
                caseFailed = true;

                if (c.id == kBrokenCase)
                    ++brokenCaseFailures;
                else
                    ++otherFailures;
            }
        };

        check("child exit", std::to_string(expectedExit), std::to_string(exitCode));
        check("status", c.expectedStatus, status);

        // The wrong-digest row is the one whose fixture must be judged against the SHIPPED pin: its module has
        // exactly the pinned size (the fixture is padded to the pinned byte count), so the only thing that can
        // refuse it is the digest rung, and the digest it compares against has to be PIN.json's.
        if (c.pinIsShippedPayload)
            check("the pin this case used is PIN.json's",
                  std::to_string(AmpereMfgLoader::kPayloadExpectedBytes) + "/" +
                      AmpereMfgLoader::kPayloadExpectedSha256,
                  ValueOf(values, "expected_pin_bytes") + "/" + ValueOf(values, "expected_pin_sha256"));

        if (!c.expectedDetail.empty())
            check("detail", c.expectedDetail, detail);
        else if (!c.expectedDetailContains.empty())
            check("detail contains", "true",
                  detail.find(c.expectedDetailContains) != std::string::npos ? "true" : "false");

        check("load attempts", std::to_string(c.expectedAttempts), attempts);
        check("companion INI written", c.expectedIni ? "true" : "false", iniWritten);

        if (c.expectedIni)
        {
            // The parent hashes the same file the child hashed: the "identical INI sha256" claim is checked twice
            // in-process, and the runner checks it a third time with PowerShell.
            bool hashOk = false;
            const std::string parentHash = fs::exists(iniPath) ? Sha256OfFile(iniPath, hashOk) : std::string();
            check("parent re-hash matches the child's", "true",
                  !parentHash.empty() && parentHash == iniShaFirst ? "true" : "false");

            if (c.assertFirstGolden)
                check("INI the loader wrote equals the generated schema", "true",
                      ValueOf(values, "ini_written_matches_golden"));
        }

        if (c.assertKernelLine)
            check("kernel-image line", c.expectedKernelLine, ValueOf(values, "kernel_image_line"));

        if (c.assertMaxFramesLine)
            check("max-frames line", c.expectedMaxFramesLine, ValueOf(values, "maxframes_line"));

        if (c.assertForeignAbsent)
            check("no foreign INI key survives", "false", ValueOf(values, "foreign_keys_present_after"));

        if (c.assertTruncatedMarkerAbsent)
            check("the truncated INI was replaced whole", "false", ValueOf(values, "truncated_marker_present_after"));

        if (c.assertTamperSurvived)
            check("the latch did not rewrite the INI", "true", ValueOf(values, "tamper_survived"));

        if (c.assertSecondGolden)
            check("second arm keeps the first settings", "true", ValueOf(values, "ini_second_matches_golden"));

        if (c.assertSecondSecondGolden)
            check("the second settings did not take effect", "false",
                  ValueOf(values, "ini_second_matches_second_golden"));

        if (c.assertSecond)
        {
            check("second status", c.expectedStatus, secondStatus);
            check("second arm attempts no load", std::to_string(c.expectedAttempts), attemptsSecond);
            check("the second arm did not touch the INI", "true",
                  iniShaFirst == iniShaSecond && iniShaFirst != "<missing>" ? "true" : "false");
        }

        if (c.assertModuleUnchanged)
            check("the payload bytes are untouched", "true", ValueOf(values, "module_unchanged"));

        if (c.assertAttachLines >= 0)
        {
            const int attaches = CountLinesWith(attachLog, "payload:attach");
            check("payload attaches", std::to_string(c.assertAttachLines), std::to_string(attaches));
        }

        const std::string pass = caseFailed ? "FAIL" : "pass";

        g_table += pass + " | " + c.id + " | expected " + c.expectedStatus + " | observed " + status +
                    " | child exit " + std::to_string(exitCode) + " | attempts " + attempts + " | ini " + iniWritten +
                    "\n";

        g_caseJson.push_back("  {\"case\": \"" + JsonEscape(c.id) + "\", \"name\": \"" + JsonEscape(c.name) +
                             "\", \"pass\": " + (caseFailed ? "false" : "true") + ", \"stagedPayload\": \"" +
                             JsonEscape(staged) + "\", \"expectedStatus\": \"" + JsonEscape(c.expectedStatus) +
                             "\", \"status\": \"" + JsonEscape(status) + "\", \"detail\": \"" + JsonEscape(detail) +
                             "\", \"expectedChildExit\": " + std::to_string(expectedExit) + ", \"childExit\": " +
                             std::to_string(exitCode) + ", \"loadAttempts\": " +
                             (attempts == "<missing>" ? "-1" : attempts) +
                             ", \"expectedLoadAttempts\": " + std::to_string(c.expectedAttempts) +
                             ", \"iniWritten\": " + (iniWritten == "true" ? "true" : "false") +
                             ", \"iniPath\": \"" + JsonEscape(iniPath) + "\", \"iniBytesFirst\": \"" +
                             JsonEscape(iniBytesFirst) + "\", \"iniSha256First\": \"" + JsonEscape(iniShaFirst) +
                             "\", \"iniSha256Second\": \"" + JsonEscape(iniShaSecond) + "\", \"modulePath\": \"" +
                             JsonEscape(modulePath) + "\", \"attaches\": " +
                             std::to_string(c.assertAttachLines >= 0 ? CountLinesWith(attachLog, "payload:attach")
                                                                     : -1) +
                             "}");

        counts += c.id + " load_attempts=" + attempts + " expected=" + std::to_string(c.expectedAttempts) +
                  (c.assertAttachLines >= 0
                       ? " attached=" + std::to_string(CountLinesWith(attachLog, "payload:attach")) + " expected=" +
                             std::to_string(c.assertAttachLines)
                       : "") +
                  "\n";
    }

    SetEnvironmentVariableW(L"AMPERE_MFG_STUB_LOG", nullptr);
    SetEnvironmentVariableW(L"AMPERE_MFG_PIN_FILE", nullptr);

    // ------------------------------------------------------------------
    // Receipts
    // ------------------------------------------------------------------
    {
        std::string json = "{\n \"schema\": 1,\n \"tool\": \"tests/ampere_mfg_failure_matrix_smoke.cpp\",\n";
        json += " \"brokenFixture\": " + std::string(brokenFixture ? "true" : "false") + ",\n";
        json += " \"failures\": " + std::to_string(failures) + ",\n";
        json += " \"cases\": [\n";

        for (size_t i = 0; i < g_caseJson.size(); ++i)
            json += g_caseJson[i] + (i + 1 == g_caseJson.size() ? "" : ",") + "\n";

        json += " ]\n}\n";
        WriteText(evidence / "matrix-cases.json", json);

        std::string rows;

        for (const auto& row : g_rows)
        {
            rows += std::string(row.ok ? "pass " : "FAIL ") + row.caseId + " " + row.name + " expected[" + row.expected +
                    "] observed[" + row.observed + "]\n";
        }

        rows += "\nchecks=" + std::to_string(g_rows.size()) + " failures=" + std::to_string(failures) +
                " brokenCaseFailures=" + std::to_string(brokenCaseFailures) +
                " otherFailures=" + std::to_string(otherFailures) + "\n";
        WriteText(evidence / "matrix-rows.txt", rows);

        std::string table = "loader failure matrix: one child process per case, the production Arm()\n"
                            "pass/FAIL | case | expected | observed | child exit | attempts | ini\n\n" +
                            g_table;
        table += "\ncases=" + std::to_string(cases.size()) + " failures=" + std::to_string(failures) + "\n";
        WriteText(evidence / "matrix-table.txt", table);
        WriteText(evidence / "load-counts.txt", counts);

        std::string hashes;
        std::error_code hashError;

        for (const auto& entry : fs::directory_iterator(evidence / "cases", hashError))
        {
            if (entry.path().extension() != ".kv")
                continue;

            const auto values = ReadReceipt(entry.path());
            const std::string iniPath = ValueOf(values, "ini_path");

            if (iniPath == "<empty>" || iniPath == "<missing>")
                continue;

            bool ok = false;
            const std::string digest = Sha256OfFile(fs::path(Wide(iniPath)), ok);
            hashes += ValueOf(values, "case") + " first=" + ValueOf(values, "ini_sha256_first") + " second=" +
                      ValueOf(values, "ini_sha256_second") + " parent=" + (ok ? digest : std::string("<unread>")) +
                      " bytes=" + ValueOf(values, "ini_bytes_first") + "  " + iniPath + "\n";
        }

        WriteText(evidence / "ini-hashes.txt", hashes.empty() ? std::string("no companion INI was written\n") : hashes);
    }

    std::printf("\nchecks=%d failures=%d brokenCaseFailures=%d otherFailures=%d\n", static_cast<int>(g_rows.size()),
                failures, brokenCaseFailures, otherFailures);

    if (brokenFixture)
    {
        std::string verdict;

        if (failures == 0)
        {
            verdict = "BROKEN FIXTURE NOT DETECTED: every case passed, so the matrix is blind to a corrupted "
                      "module\n";
            std::printf("%s", verdict.c_str());
            WriteText(evidence / "broken-fixture.txt",
                      "the deliberately broken fixture is " + std::string(kBrokenCase) +
                          ": its module is replaced by a text file, so the Loaded expectation must fail\n" + verdict +
                          "exit=2\n");
            return 2;
        }

        if (otherFailures == 0)
        {
            verdict = "BROKEN FIXTURE CAUGHT: " + std::to_string(brokenCaseFailures) +
                      " failure(s), all on the deliberately broken case " + std::string(kBrokenCase) + "\n";
            std::printf("%s", verdict.c_str());
            WriteText(evidence / "broken-fixture.txt",
                      "the deliberately broken fixture is " + std::string(kBrokenCase) +
                          ": its module is replaced by a text file, so the Loaded expectation must fail\n" + verdict +
                          "exit=3 (the designed failure)\n");
            return 3;
        }

        verdict = "BROKEN FIXTURE CAUGHT BUT TOO BROAD: " + std::to_string(otherFailures) +
                  " failure(s) on cases the broken fixture cannot explain\n";
        std::printf("%s", verdict.c_str());
        WriteText(evidence / "broken-fixture.txt",
                  "the deliberately broken fixture is " + std::string(kBrokenCase) + "\n" + verdict + "exit=4\n");
        return 4;
    }

    return failures == 0 ? 0 : 1;
}
