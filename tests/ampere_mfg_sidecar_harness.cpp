// ============================================================================
// tests/ampere_mfg_sidecar_harness.cpp
// Todo 2 of .omo/plans/rtx2030-mfg-integration.md: prove the sidecar load and
// arming contract, including the RENAMED module, in a child-process harness.
//
// What this harness is
// --------------------
// A harness-local reference implementation of the arming seam the production
// loader (todo 3) and the Streamline hook boundary (todo 6) must satisfy:
//
//   * one entry point, Arm(), mirroring the loader's single arming call;
//   * an init boundary that stands in for StreamlineHooks::hkslInit (R6): the
//     payload must be loaded, and its companion INI applied, BEFORE the boundary
//     is crossed (C2). A late arm is a NAMED failure and must not load anything;
//   * the module is loaded by EXPLICIT ABSOLUTE PATH, never by name resolution;
//   * the module's own identity exports are reported (DlssgProxy_Name,
//     DlssgProxy_Role) and role 2 (standby) is a named refusal, because a
//     standby proxy forwards exports but installs nothing (C5);
//   * repeated Arm() calls do not re-load the module and do not re-write the INI;
//   * LoadLibrary returning a handle is LOAD EVIDENCE ONLY (C5). This harness
//     never claims working frame generation, never launches a game and never
//     touches a live install.
//
// Every module load happens in a CHILD PROCESS of this executable: the real
// payload is a 30 MB proxy whose DllMain installs detours and takes a
// process-wide "active proxy" marker, so one variant per process is the only way
// to observe which name loads cleanly.
//
// Cases
// -----
//   arm-before-init   stub payload, armed before the boundary   -> exit 0
//   arm-after-init    stub payload, armed AFTER the boundary    -> exit 1 with
//                     STATUS=ArmAfterInitBoundary and no load attempt (the RED
//                     fixture: it must fail with that named status)
//   idempotence       stub payload, Arm() twice                 -> exit 0, one
//                     load, one INI write
//   real-renamed      the STAGED payload copied as dlssg_sm86.dll
//   real-original     the STAGED payload copied as version.dll
//
// Todo 6 (production path)
// ------------------------
// The cases above measure a harness-local stand-in. The `production-*` cases
// measure the shipped code: OptiScaler/framegen/dlssg/AmpereMfgLoader.cpp is
// compiled into this harness against the same mock seams the todo-5 eligibility
// smoke uses, and each case runs the real Arm() in a child process with fixture
// host facts, with the arming call placed BEFORE the simulated init boundary.
//
//   production-plain           unlock off, plain ownership   -> Disabled, no load, no INI
//   production-external        SM86 + External ownership on -> Loaded, load before the boundary
//   production-external-off    unlock on but External off   -> Conflict, no load, no INI
//   production-missing-payload no module staged             -> PayloadMissing, nothing enabled
//   production-idempotence     two arming calls             -> one load, INI bytes unchanged
//   production-fork-guard      --simulate-fork-guard        -> HookNotInstalled (the audit fixture)
//
// --external-mode runs only the External-ownership case; --fork-guard runs only
// the simulated-fork-guard fixture and exits 1 when the guard makes the hook
// unable to arm (the failure fixture, reported with its named status).
//
// Exit codes (parent mode)
// ------------------------
//   0 = every case behaved as specified
//   1 = a case did not behave as specified (or --fork-guard confirmed the guard)
//   3 = neither real-payload variant loaded cleanly (no bundled_name written)
//   4 = --late-arm: the fixture did NOT fail with the named status (the RED state)
//   3 = --production-red: the pre-todo-6 production path was confirmed (no load,
//       PayloadValidated); 4 = it was not, so the RED run proved nothing
//  64 = usage error, 66 = fail-closed environment error
//
// Usage
// -----
//   ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> [--late-arm]
//   ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> --production-red
//   ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> --external-mode
//   ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> --fork-guard
//   ampere_mfg_sidecar_harness.exe --child <case> --payload <abs> --ini <abs> --log <abs>
//   ampere_mfg_sidecar_harness.exe --child <case> --root <abs> --payload <abs> --log <abs> --production
//
// Evidence receipts are written by the parent (see RECEIPT-todo2.md, RECEIPT-todo6.md).
// ============================================================================

#define NOMINMAX

#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// The PRODUCTION arming path (todo 6)
// ---------------------------------------------------------------------------
// The `production-*` cases measure the shipped loader, not a stand-in: the real
// OptiScaler/framegen/dlssg/AmpereMfgLoader.cpp is compiled into this harness
// against the same seams tests/ampere_mfg_eligibility_smoke.cpp uses - the four
// host sources the loader reads are replaced by tests/ampere_mfg_eligibility_mocks.h
// and the empty headers in tests/ampere_mfg_eligibility_seams/, everything else
// is shipped code. The runner supplies the /I paths and the defines.
#if !defined(AMPERE_MFG_HARNESS_PRODUCTION)
#error "compile this harness through tests\\Run-AmpereMfgSidecarHarness.cmd: the todo-6 cases need the production include paths"
#endif

#include "ampere_mfg_eligibility_mocks.h"
#include <nvapi.h>
#include "AmpereMfgLoader.cpp"

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static std::string Narrow(const std::wstring& text)
{
    if (text.empty())
        return {};

    const int needed = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0, nullptr,
                                          nullptr);
    std::string out(static_cast<size_t>(needed < 0 ? 0 : needed), '\0');

    if (needed > 0)
        WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(), needed, nullptr, nullptr);

    return out;
}

static std::wstring Wide(const std::string& text)
{
    if (text.empty())
        return {};

    const int needed = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(needed < 0 ? 0 : needed), L'\0');

    if (needed > 0)
        MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(), needed);

    return out;
}

static std::string Lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return text;
}

static std::string Trim(const std::string& text)
{
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

static bool StartsWith(const std::string& text, const std::string& prefix)
{
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

static std::vector<std::string> ReadLines(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(in, line))
        lines.push_back(Trim(line));

    while (!lines.empty() && lines.back().empty())
        lines.pop_back();

    return lines;
}

// Machine tokens are printed as "KEY=value" with no space in the value, so the
// parent can assert on exact strings without parsing prose.
static std::map<std::string, std::string> ParseTokens(const std::vector<std::string>& lines)
{
    std::map<std::string, std::string> tokens;

    for (const std::string& line : lines)
    {
        const size_t equals = line.find('=');
        if (equals == std::string::npos || equals == 0)
            continue;
        if (line.find(' ') != std::string::npos)
            continue;
        tokens[line.substr(0, equals)] = line.substr(equals + 1);
    }

    return tokens;
}

// Append-only writer: one CreateFileW/WriteFile per line, so the interleaving of
// the loader's events with the payload's own DllMain line is the real call order.
static void AppendLogLine(const fs::path& log, const std::string& line)
{
    HANDLE file = CreateFileW(log.wstring().c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (file == INVALID_HANDLE_VALUE)
        return;

    std::string text = line;
    text.push_back('\n');

    DWORD written = 0;
    WriteFile(file, text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
    CloseHandle(file);
}

static bool FileExists(const fs::path& path)
{
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

static std::string Sha256Hex(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);

    if (!in)
        return {};

    std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    BCRYPT_ALG_HANDLE algorithm = nullptr;

    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0 || algorithm == nullptr)
        return {};

    unsigned char digest[32] = {};
    const NTSTATUS status = BCryptHash(algorithm, nullptr, 0, reinterpret_cast<PUCHAR>(data.data()),
                                       static_cast<ULONG>(data.size()), digest, sizeof(digest));
    BCryptCloseAlgorithmProvider(algorithm, 0);

    if (status != 0)
        return {};

    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(64);

    for (unsigned char byte : digest)
    {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0xF]);
    }

    return out;
}

static std::string JsonEscape(const std::string& text)
{
    std::string out;

    for (char c : text)
    {
        switch (c)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char buffer[8] = {};
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                out += buffer;
            }
            else
            {
                out.push_back(c);
            }
        }
    }

    return out;
}

static std::string NowIso8601Utc()
{
    SYSTEMTIME now = {};
    GetSystemTime(&now);
    char buffer[40] = {};
    std::snprintf(buffer, sizeof(buffer), "%04u-%02u-%02uT%02u:%02u:%02uZ", now.wYear, now.wMonth, now.wDay, now.wHour,
                  now.wMinute, now.wSecond);
    return buffer;
}

static fs::path OwnExecutablePath()
{
    wchar_t buffer[MAX_PATH * 4] = {};
    GetModuleFileNameW(nullptr, buffer, MAX_PATH * 4);
    return fs::path(buffer);
}

// Any path that looks like a live game install is a hard refusal: the harness
// only ever touches the repository and a scratch copy under the evidence dir.
static bool LooksLikeLiveInstall(const std::string& path)
{
    const std::string lowered = Lower(path);
    return lowered.find("steamlibrary") != std::string::npos || lowered.find("\\steamapps\\") != std::string::npos ||
           lowered.find("epic games") != std::string::npos;
}

// ---------------------------------------------------------------------------
// The arming seam (harness-local stand-in for the production loader's Arm())
// ---------------------------------------------------------------------------

// Status vocabulary the loader adopts (todo 3/6), plus the C2 violation this
// harness has to be able to name.
enum class SidecarStatus
{
    Disabled,
    Ineligible,
    Conflict,
    ArmAfterInitBoundary,
    PayloadMissing,
    PayloadLoadFailed,
    PayloadStandby,
    PayloadLoaded,
};

static const char* StatusName(SidecarStatus status)
{
    switch (status)
    {
    case SidecarStatus::Disabled: return "Disabled";
    case SidecarStatus::Ineligible: return "Ineligible";
    case SidecarStatus::Conflict: return "Conflict";
    case SidecarStatus::ArmAfterInitBoundary: return "ArmAfterInitBoundary";
    case SidecarStatus::PayloadMissing: return "PayloadMissing";
    case SidecarStatus::PayloadLoadFailed: return "PayloadLoadFailed";
    case SidecarStatus::PayloadStandby: return "PayloadStandby";
    case SidecarStatus::PayloadLoaded: return "PayloadLoaded";
    }

    return "Unknown";
}

// Stands in for the game's slInit: once cross() has run, the game has already
// decided whether this platform supports DLSS-G, so arming afterwards is too
// late (see docs/rtx2030-payload-contract.md, "Arming order").
struct InitBoundary
{
    bool crossed = false;
};

struct ArmOutcome
{
    SidecarStatus status = SidecarStatus::Disabled;
    bool loaded = false;
    HMODULE module = nullptr;
    std::string detail;
    std::string identity;
    int role = -1;
};

class ArmingSeam
{
public:
    ArmingSeam(InitBoundary* boundary, fs::path payload, fs::path ini, fs::path log)
        : boundary_(boundary), payload_(std::move(payload)), ini_(std::move(ini)), log_(std::move(log))
    {
    }

    // The single arming entry point. Mirrors the production loader's Arm():
    // validates the payload, applies the companion INI, loads the module from an
    // explicit absolute path, and reports a named status.
    ArmOutcome Arm()
    {
        if (armed_)
        {
            AppendLogLine(log_, "event arm:skipped-already-armed");
            return outcome_;
        }

        AppendLogLine(log_, "event arm:begin payload=" + Narrow(payload_.wstring()) + " ini=" + Narrow(ini_.wstring()));

        // C2: the payload has to be armed before the game's Streamline capability
        // decision. Arming afterwards is a named failure and must not load
        // anything - a late arm is never a silent "enabled".
        if (boundary_->crossed)
        {
            outcome_ = {SidecarStatus::ArmAfterInitBoundary, false, nullptr,
                        "Arm() ran after the Streamline init boundary was crossed; the capability decision has "
                        "already been made (C2)",
                        {}, -1};
            armed_ = true;
            AppendLogLine(log_, "event arm:refused-after-boundary");
            return outcome_;
        }

        if (!FileExists(payload_))
        {
            outcome_ = {SidecarStatus::PayloadMissing, false, nullptr, "payload module not found: " + Narrow(payload_.wstring()), {}, -1};
            armed_ = true;
            AppendLogLine(log_, std::string("event arm:missing path=") + Narrow(payload_.wstring()));
            return outcome_;
        }

        WriteIniIfAbsent();

        ++loadCalls_;
        AppendLogLine(log_, "event arm:load-call path=" + Narrow(payload_.wstring()));

        HMODULE module = LoadLibraryW(payload_.wstring().c_str());
        const DWORD error = module == nullptr ? GetLastError() : 0;
        AppendLogLine(log_, std::string("event arm:load-return handle=") + (module ? "nonnull" : "null") +
                                " error=" + std::to_string(error));

        if (module == nullptr)
        {
            outcome_ = {SidecarStatus::PayloadLoadFailed, false, nullptr,
                        "LoadLibraryW failed with error " + std::to_string(error), {}, -1};
            armed_ = true;
            return outcome_;
        }

        // Identity and role come from the module's own exports (both the real
        // payload and the stub expose them).
        using NameFn = const wchar_t*(__cdecl*)(void);
        using RoleFn = int(__cdecl*)(void);

        if (auto nameFn = reinterpret_cast<NameFn>(GetProcAddress(module, "DlssgProxy_Name")))
        {
            const wchar_t* name = nameFn();
            outcome_.identity = name == nullptr ? std::string() : Narrow(name);
        }

        if (auto roleFn = reinterpret_cast<RoleFn>(GetProcAddress(module, "DlssgProxy_Role")))
            outcome_.role = roleFn();

        // A module that took the standby role forwards its exports but installs
        // nothing: a named refusal, never "installed" (C5).
        if (outcome_.role == 2)
        {
            outcome_.status = SidecarStatus::PayloadStandby;
            outcome_.detail = "the module reported the standby proxy role (another proxy of the family is active)";
            outcome_.module = module;
            armed_ = true;
            return outcome_;
        }

        outcome_.status = SidecarStatus::PayloadLoaded;
        outcome_.loaded = true;
        outcome_.module = module;
        outcome_.detail = "module loaded from an explicit absolute path; load evidence only (C5)";
        armed_ = true;
        return outcome_;
    }

    int LoadCalls() const { return loadCalls_; }
    int IniWrites() const { return iniWrites_; }
    bool Armed() const { return armed_; }

private:
    // The companion INI is generated once, in the 0.3.5 layout, and never
    // re-written (C4 idempotence; the real generator belongs to todo 3).
    void WriteIniIfAbsent()
    {
        if (FileExists(ini_))
            return;

        std::error_code ec;
        fs::create_directories(ini_.parent_path(), ec);

        std::ofstream out(ini_, std::ios::binary);

        if (!out)
            return;

        out << "; Generated by tests/ampere_mfg_sidecar_harness.cpp (harness-local stand-in for the loader's INI "
               "step).\n"
               "; Layout: 0.3.5 (docs/rtx2030-payload-contract.md, \"INI schema keys\").\n"
               "[General]\n"
               "Enabled=1\n"
               "\n"
               "[FrameGeneration]\n"
               "Optimized=1\n"
               "MaxGeneratedFrames=3\n"
               "\n"
               "[Compatibility]\n"
               "Router=Auto\n"
               "\n"
               "[Runtime]\n"
               "Mode=Bundled\n";
        out.close();
        ++iniWrites_;
        AppendLogLine(log_, "event arm:ini-write path=" + Narrow(ini_.wstring()));
    }

    InitBoundary* boundary_ = nullptr;
    fs::path payload_;
    fs::path ini_;
    fs::path log_;
    bool armed_ = false;
    int loadCalls_ = 0;
    int iniWrites_ = 0;
    ArmOutcome outcome_;
};

// Ordering proof straight from the log: the payload's own attach line (written
// by its DllMain) and the loader's load-return line must both precede the
// boundary line.
static std::string ComputeOrder(const std::vector<std::string>& lines)
{
    long long attach = -1;
    long long loadReturn = -1;
    long long boundary = -1;

    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (StartsWith(lines[i], "payload:attach"))
            attach = static_cast<long long>(i);
        if (StartsWith(lines[i], "event arm:load-return"))
            loadReturn = static_cast<long long>(i);
        if (StartsWith(lines[i], "event init:boundary"))
            boundary = static_cast<long long>(i);
    }

    if (loadReturn < 0)
        return "no-load-before-boundary";

    if (boundary < 0)
        return "not-evaluated";

    if (loadReturn < boundary && (attach < 0 || attach < boundary))
        return "load-before-boundary";

    return "VIOLATION";
}

// ---------------------------------------------------------------------------
// Child process
// ---------------------------------------------------------------------------

struct ChildOptions
{
    std::string caseName;
    fs::path payload;
    fs::path ini;
    fs::path log;
};

static std::string Quote(const fs::path& path)
{
    return "\"" + Narrow(path.wstring()) + "\"";
}

// Runs one child and returns its exit code; the child's own events live in its
// --log file, its stdout/stderr in the given file.
static int RunChildProcess(const fs::path& exe, const std::string& arguments, const fs::path& stdioFile,
                           DWORD timeoutMs, bool* timedOut)
{
    if (timedOut != nullptr)
        *timedOut = false;

    std::error_code ec;
    fs::create_directories(stdioFile.parent_path(), ec);

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

    const DWORD wait = WaitForSingleObject(process.hProcess, timeoutMs);
    DWORD exitCode = 0;

    if (wait == WAIT_TIMEOUT)
    {
        TerminateProcess(process.hProcess, 0xDEAD);
        WaitForSingleObject(process.hProcess, 5000);
        if (timedOut != nullptr)
            *timedOut = true;
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

static int RunChildCase(const ChildOptions& options)
{
    // The stub payload records its DllMain attach line into this case's log, so
    // the payload's own load time sits next to the loader's events.
    SetEnvironmentVariableW(L"AMPERE_MFG_STUB_LOG", options.log.wstring().c_str());

    std::error_code ec;
    fs::create_directories(options.log.parent_path(), ec);
    fs::remove(options.log, ec);
    fs::remove(options.ini, ec);

    InitBoundary boundary;
    AppendLogLine(options.log, "child case=" + options.caseName + " pid=" + std::to_string(GetCurrentProcessId()));
    AppendLogLine(options.log, "child payload=" + Narrow(options.payload.wstring()) + " ini=" + Narrow(options.ini.wstring()));

    ArmingSeam seam(&boundary, options.payload, options.ini, options.log);

    const bool lateArmCase = options.caseName == "arm-after-init";
    const bool idempotenceCase = options.caseName == "idempotence";

    if (lateArmCase)
    {
        boundary.crossed = true;
        AppendLogLine(options.log, "event init:boundary");
    }

    const ArmOutcome first = seam.Arm();
    AppendLogLine(options.log, std::string("event arm:result status=") + StatusName(first.status) + " detail=" + first.detail);

    const std::string iniHashAfterFirst = FileExists(options.ini) ? Sha256Hex(options.ini) : std::string();

    bool iniRewritten = false;
    ArmOutcome second;
    bool secondArm = false;

    if (idempotenceCase)
    {
        secondArm = true;
        second = seam.Arm();
        AppendLogLine(options.log, std::string("event arm:second status=") + StatusName(second.status));
        const std::string iniHashAfterSecond = FileExists(options.ini) ? Sha256Hex(options.ini) : std::string();
        iniRewritten = iniHashAfterFirst != iniHashAfterSecond;
    }

    if (!lateArmCase)
    {
        boundary.crossed = true;
        AppendLogLine(options.log, "event init:boundary");
    }

    int stubAttachCount = 0;
    if (first.module != nullptr)
    {
        using AttachCountFn = int(__cdecl*)(void);
        if (auto attachFn = reinterpret_cast<AttachCountFn>(GetProcAddress(first.module, "AmpereMfgStubAttachCount")))
            stubAttachCount = attachFn();
        else
            stubAttachCount = -1; // a real payload carries no attach counter
    }

    const std::vector<std::string> lines = ReadLines(options.log);

    AppendLogLine(options.log, "STATUS=" + std::string(StatusName(first.status)));
    AppendLogLine(options.log, "order=" + ComputeOrder(lines));
    AppendLogLine(options.log, "handle=" + std::string(first.module != nullptr ? "nonnull" : "null"));
    AppendLogLine(options.log, "load_count=" + std::to_string(seam.LoadCalls()));
    AppendLogLine(options.log, "ini_write_count=" + std::to_string(seam.IniWrites()));
    AppendLogLine(options.log, "ini_rewritten=" + std::string(secondArm ? (iniRewritten ? "true" : "false") : "n/a"));
    AppendLogLine(options.log, "stub_attach_count=" + (stubAttachCount < 0 ? std::string("n/a") : std::to_string(stubAttachCount)));
    AppendLogLine(options.log, "role=" + (first.role < 0 ? std::string("n/a") : std::to_string(first.role)));
    AppendLogLine(options.log, "identity=" + (first.identity.empty() ? std::string("empty") : first.identity));

    // INI hashes across the two arm calls: the idempotence receipt needs the bytes
    // around each call, not just the "not rewritten" verdict.
    AppendLogLine(options.log, "ini_sha256_first=" + (iniHashAfterFirst.empty() ? std::string("absent") : iniHashAfterFirst));
    AppendLogLine(options.log,
                  "ini_sha256_second=" +
                      (FileExists(options.ini) ? Sha256Hex(options.ini) : std::string("absent")));
    AppendLogLine(options.log, "ini_bytes=" + std::to_string(FileExists(options.ini) ? fs::file_size(options.ini) : 0));
    AppendLogLine(options.log, "boundary_crossed_before_arm=" + std::string(lateArmCase ? "true" : "false"));

    int exitCode = 0;

    if (lateArmCase)
    {
        // The fixture fails by design when arming happened after the boundary.
        exitCode = first.status == SidecarStatus::ArmAfterInitBoundary ? 1 : 0;
    }
    else if (idempotenceCase)
    {
        const bool ok = first.status == SidecarStatus::PayloadLoaded && seam.LoadCalls() == 1 && seam.IniWrites() == 1 &&
                        !iniRewritten && stubAttachCount == 1;
        exitCode = ok ? 0 : 1;
    }
    else
    {
        exitCode = first.status == SidecarStatus::PayloadLoaded ? 0 : 1;
    }

    AppendLogLine(options.log, "child-exit=" + std::to_string(exitCode));
    return exitCode;
}

// ---------------------------------------------------------------------------
// The PRODUCTION path (todo 6): the shipped loader is the code under test
// ---------------------------------------------------------------------------

// The install decision for the pre-slInit arming hook. StreamlineHooks::hookInterposer installs the slInit
// detour unconditionally (Streamline_Hooks.cpp, entry at 1889, `DetourAttach(&(PVOID&) o_slInit, hkslInit)` at
// 1970), and hkslInit runs the arming call before every return - so the hook is installed AND executed in
// every frame-generation ownership mode, plain and External. --simulate-fork-guard flips this to the
// reference fork's decision, `wilsjo2/main:OptiScaler/hooks/Streamline_Hooks.cpp:1866-1867`
// (`if (State::Instance().externalFrameGeneration) return;`), which never installs the hook when External FG
// ownership is active. That fixture is what the audit has to keep out of the product (C2).
struct HookInstallDecision
{
    bool installed = true;
    std::string detail = "installed: the slInit detour carries the pre-slInit arming call in every ownership mode";
};

static HookInstallDecision ForkGuardDecision()
{
    HookInstallDecision decision;
    decision.installed = false;
    decision.detail = "SIMULATED fork guard (wilsjo2/main:OptiScaler/hooks/Streamline_Hooks.cpp:1866): the install "
                      "was skipped because External FG ownership is active";
    return decision;
}

struct ProductionCaseSpec
{
    std::string name;
    bool unlock = true;
    bool externalFg = true;
    bool stagePayload = true;
    bool secondArm = false;
    bool forkGuard = false;
    bool realPayload = false; // the parent staged the PINNED 30 MB payload, not the stub
    bool clearStage = false;  // remove whatever is staged (the missing-payload case)
};

static ProductionCaseSpec ProductionSpec(const std::string& name)
{
    ProductionCaseSpec spec;
    spec.name = name;

    if (name == "production-plain")
    {
        spec.unlock = false;
        spec.externalFg = false;
    }
    else if (name == "production-external" || name == "production-fork-guard")
    {
        spec.forkGuard = name == "production-fork-guard";
    }
    else if (name == "production-real-external")
    {
        // The shipped bytes, loaded by the shipped arming path: the strongest load proof this host can offer.
        spec.stagePayload = false;
        spec.realPayload = true;
    }
    else if (name == "production-external-off")
    {
        spec.externalFg = false;
    }
    else if (name == "production-missing-payload")
    {
        spec.stagePayload = false;
        spec.clearStage = true;
    }
    else if (name == "production-idempotence")
    {
        spec.secondArm = true;
    }

    return spec;
}

// Ordering proof for the production cases: the payload's own DllMain line (written by the stub while
// Arm() holds LoadLibrary) and the loader's arm-result line both precede the init-boundary line.
static std::string ComputeProductionOrder(const std::vector<std::string>& lines)
{
    long long attach = -1;
    long long armResult = -1;
    long long boundary = -1;

    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (StartsWith(lines[i], "payload:attach"))
            attach = static_cast<long long>(i);
        if (StartsWith(lines[i], "event arm:result"))
            armResult = static_cast<long long>(i);
        if (StartsWith(lines[i], "event init:boundary"))
            boundary = static_cast<long long>(i);
    }

    if (attach < 0)
        return "no-load-before-boundary";

    if (boundary < 0)
        return "not-evaluated";

    if (attach < boundary && armResult >= 0 && armResult < boundary)
        return "load-before-boundary";

    return "VIOLATION";
}

static int RunProductionChildCase(const ProductionCaseSpec& spec, const fs::path& root, const fs::path& stub,
                                  const fs::path& log)
{
    std::error_code ec;
    fs::create_directories(log.parent_path(), ec);
    fs::remove(log, ec);

    SetEnvironmentVariableW(L"AMPERE_MFG_STUB_LOG", log.wstring().c_str());
    // The production no-argument Arm() resolves its package root from the OptiScaler.dll path; point that at the
    // case's scratch tree so the exact entry the Streamline init path calls is what runs here.
    SetEnvironmentVariableW(L"AMPERE_MFG_DLL_PATH", (root / "OptiScaler.dll").wstring().c_str());

    // The mocked host: one physical SM86 adapter, so this process renders on the part the payload supports.
    GpuInformation sm86;
    sm86.name = "NVIDIA GeForce RTX 3060";
    sm86.vendorId = VendorId::Nvidia;
    sm86.softwareAdapter = false;
    sm86.nvidiaArchInfo.architecture_id = AmpereMfgLoader::kArchAmpere;
    sm86.nvidiaArchInfo.implementation = 6; // GA106
    IdentifyGpu::SetAdapters({ sm86 });

    auto* config = Config::Instance();
    config->FGDLSSGAmpereMfgUnlock.stored = spec.unlock;
    config->ExternalFrameGeneration.stored = spec.externalFg;
    config->FGDLSSGAmpereMfgMaxFrames.stored = 3;
    config->FGDLSSGAmpereMfgKernelImage.stored = std::nullopt;

    MfgUnlock::g_enabledForSession = false;
    State::Instance().activeFgOutput = FGOutput::NoFG;
    State::Instance().activeFgNvngx = FGNvngxReplacement::None;

    const fs::path stageDir = root / "OptiScaler" / "dlssg_sm86";
    const fs::path stagedModule = stageDir / AmpereMfgLoader::kPayloadModuleName;
    const fs::path stagedIni = stageDir / AmpereMfgLoader::kPayloadIniName;

    if (spec.clearStage)
    {
        fs::remove(stagedModule, ec);
        fs::remove(stagedIni, ec);
    }
    else if (spec.stagePayload)
    {
        fs::create_directories(stageDir, ec);
        fs::remove(stagedIni, ec);
        fs::copy_file(stub, stagedModule, fs::copy_options::overwrite_existing, ec);
    }
    else
    {
        // The parent staged the real payload; only the companion INI must be gone, so the loader has to write it.
        fs::remove(stagedIni, ec);
    }

    AppendLogLine(log, "child case=" + spec.name + " pid=" + std::to_string(GetCurrentProcessId()));

    // Todo 13's pin seam: the production ladder compares the module it found against the pinned payload. This
    // harness stages either the stub or the real bundled bytes as that payload
    // (tests/ampere_mfg_payload_pin_seam.h):
    //   * a stub case pins the size and digest of what it staged; the shipped constants describe the 30 MB proxy
    //     and would refuse the stub before the load, which is not what those cases measure;
    //   * production-real-external stages the REAL bundled payload, so it pins EXACTLY as the shipped build
    //     does - the header's PIN.json constants - which makes that case the one that measures the production
    //     trust anchor end to end (the ladder's digest rung accepting the shipped module).
    if (spec.realPayload)
    {
        AmpereMfgLoader::g_expectedPayloadBytes = AmpereMfgLoader::kPayloadExpectedBytes;
        AmpereMfgLoader::g_expectedPayloadSha256 = AmpereMfgLoader::kPayloadExpectedSha256;
        AppendLogLine(log, "event pin:shipped-constants bytes=" +
                               std::to_string(AmpereMfgLoader::ExpectedPayloadBytes()));
    }
    else if (fs::exists(stagedModule))
    {
        if (!AmpereMfgPinSeam::PinTo(stagedModule))
            AppendLogLine(log, "event pin:unreadable path=" + Narrow(stagedModule.wstring()));
        else
            AppendLogLine(log, "event pin:staged bytes=" + std::to_string(AmpereMfgLoader::ExpectedPayloadBytes()));
    }

    // The pre-slInit arming seam exactly as StreamlineHooks::hkslInit runs it: the install decision first (and
    // unconditionally), then Arm() BEFORE forwarding to the game's slInit - the boundary below.
    const HookInstallDecision install = spec.forkGuard ? ForkGuardDecision() : HookInstallDecision();

    AppendLogLine(log, std::string("event hook:install installed=") + (install.installed ? "true" : "false") +
                           " mode=" + (spec.externalFg ? "external" : "plain"));

    AmpereMfgLoader::ArmOptions options = AmpereMfgLoader::HostOptions();
    options.PackageRoot = root;

    // The gate verdict at arm time. It has to be read BEFORE Arm(): a successful load puts the module under the
    // shipped name into this process, which the ownership fact "another owner is present" correctly reports as a
    // fact, but it is not the state the gate decided on when it armed.
    const auto decision = AmpereMfgLoader::Evaluate(options, AmpereMfgLoader::HostAdapterFacts(),
                                                   AmpereMfgLoader::HostOwnershipFacts());

    std::string statusText;
    std::string detail;

    if (!install.installed)
    {
        statusText = "HookNotInstalled";
        detail = "the pre-slInit arming hook was not installed for this ownership mode, so the arming call was "
                 "never reached";
        AppendLogLine(log, "event arm:not-reached (the install decision refused the hook)");
    }
    else
    {
        AppendLogLine(log, "event arm:begin");
        // The exact production entry the Streamline init seam calls: Arm() reads the settings itself, latches
        // once per process and resolves the payload from the OptiScaler.dll directory (pointed at this case's
        // scratch tree by AMPERE_MFG_DLL_PATH).
        const auto status = AmpereMfgLoader::Arm();
        statusText = status.StateText();
        detail = status.Detail;

        AppendLogLine(log, std::string("event arm:result status=") + statusText +
                               (status.ModulePath.empty() ? std::string() : " module=" + Narrow(status.ModulePath)));

        if (statusText == "Loaded" || statusText == "BackendInstalled")
            AppendLogLine(log, "event arm:load-return handle=nonnull");
    }

    const std::string iniHashBeforeSecond = FileExists(stagedIni) ? Sha256Hex(stagedIni) : std::string();

    bool iniRewritten = false;
    std::string secondStatus;

    if (install.installed && spec.secondArm)
    {
        const auto second = AmpereMfgLoader::Arm();
        secondStatus = second.StateText();
        const std::string iniHashAfterSecond = FileExists(stagedIni) ? Sha256Hex(stagedIni) : std::string();
        iniRewritten = iniHashBeforeSecond != iniHashAfterSecond;
        AppendLogLine(log, std::string("event arm:second status=") + secondStatus);
    }

    // The game's Streamline capability decision, crossed strictly after arming (C2).
    AppendLogLine(log, "event init:boundary");

    HMODULE module = GetModuleHandleW(stagedModule.wstring().c_str());
    int attachCount = -1;
    int role = -1;
    std::string identity;

    if (module != nullptr)
    {
        using AttachCountFn = int(__cdecl*)(void);
        if (auto attachFn = reinterpret_cast<AttachCountFn>(GetProcAddress(module, "AmpereMfgStubAttachCount")))
            attachCount = attachFn();

        using RoleFn = int(__cdecl*)(void);
        if (auto roleFn = reinterpret_cast<RoleFn>(GetProcAddress(module, "DlssgProxy_Role")))
            role = roleFn();

        using NameFn = const wchar_t*(__cdecl*)(void);
        if (auto nameFn = reinterpret_cast<NameFn>(GetProcAddress(module, "DlssgProxy_Name")))
        {
            const wchar_t* name = nameFn();
            identity = name == nullptr ? std::string() : Narrow(name);
        }
    }

    const std::vector<std::string> lines = ReadLines(log);
    const bool armed = statusText == "Loaded" || statusText == "BackendInstalled";

    AppendLogLine(log, "STATUS=" + statusText);
    AppendLogLine(log, "detail=" + detail);
    AppendLogLine(log, "verdict=" + std::string(AmpereMfgLoader::VerdictName(decision.verdict)));
    AppendLogLine(log, "order=" + ComputeProductionOrder(lines));
    AppendLogLine(log, "handle=" + std::string(module != nullptr ? "nonnull" : "null"));
    AppendLogLine(log, "enabled=" + std::string(armed ? "true" : "false"));
    AppendLogLine(log, "load_count=" + std::to_string(AmpereMfgLoader::LoadAttempts()));
    AppendLogLine(log, "ini_written=" + std::string(iniHashBeforeSecond.empty() ? "false" : "true"));
    AppendLogLine(log, "ini_rewritten=" + std::string(spec.secondArm ? (iniRewritten ? "true" : "false") : "n/a"));
    AppendLogLine(log, "ini_sha256_first=" +
                           (iniHashBeforeSecond.empty() ? std::string("absent") : iniHashBeforeSecond));
    AppendLogLine(log,
                  "ini_sha256_second=" + (FileExists(stagedIni) ? Sha256Hex(stagedIni) : std::string("absent")));
    AppendLogLine(log, "stub_attach_count=" +
                           (attachCount < 0 ? std::string("n/a") : std::to_string(attachCount)));
    AppendLogLine(log, "role=" + (role < 0 ? std::string("n/a") : std::to_string(role)));
    AppendLogLine(log, "identity=" + (identity.empty() ? std::string("empty") : identity));
    AppendLogLine(log, "hook_installed=" + std::string(install.installed ? "true" : "false"));
    AppendLogLine(log, "payload_kind=" + std::string(spec.realPayload ? "pinned" : "stub"));
    AppendLogLine(log, "boundary_crossed_before_arm=false");

    return 0;
}

// ---------------------------------------------------------------------------
// Parent mode
// ---------------------------------------------------------------------------

struct Options
{
    fs::path root;
    fs::path evidence;
    fs::path stub;
    bool lateArmOnly = false;
    bool keepWork = false;
    bool productionRed = false;    // expect the pre-todo-6 production path (no load) - RED confirmation
    bool productionOnly = false;   // run the production cases without the legacy green cases
    bool externalModeOnly = false; // run only the External-ownership production case
    bool forkGuardOnly = false;    // run only the simulated-fork-guard fixture
};

static int Usage()
{
    std::printf("usage:\n"
                "  ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> [--stub <dll>] [--late-arm] "
                "[--keep-work]\n"
                "  ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> --production-red [--keep-work]\n"
                "  ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> --external-mode\n"
                "  ampere_mfg_sidecar_harness.exe --root <repo> --evidence <dir> --fork-guard\n"
                "  ampere_mfg_sidecar_harness.exe --child <case> --payload <abs dll> --ini <abs ini> --log <abs log>\n"
                "  ampere_mfg_sidecar_harness.exe --child <case> --root <abs> --payload <abs dll> --log <abs log> "
                "[--simulate-fork-guard] --production\n");
    return 64;
}

static std::string ChildArguments(const std::string& caseName, const fs::path& payload, const fs::path& ini,
                                 const fs::path& log)
{
    return "--child " + caseName + " --payload " + Quote(payload) + " --ini " + Quote(ini) + " --log " + Quote(log);
}

// The --late-arm invocation: run the fixture by itself and report whether it
// failed with the named status. Exit 1 = detected (the case failed as designed),
// exit 4 = NOT detected (the RED state this todo records first).
static int RunLateArmFixture(const Options& options)
{
    const fs::path workRoot = options.evidence / "work";
    const fs::path work = workRoot / "late-arm";
    const fs::path ini = work / "dlssg_sm86.ini";
    const fs::path log = options.evidence / "late-arm-fixture.log";
    const fs::path stdio = options.evidence / "late-arm-fixture.stdio.txt";

    std::error_code ec;
    fs::create_directories(work, ec);
    fs::remove(log, ec);
    fs::remove(ini, ec);

    bool timedOut = false;
    const int exitCode = RunChildProcess(OwnExecutablePath(), ChildArguments("arm-after-init", options.stub, ini, log),
                                        stdio, 60000, &timedOut);

    const auto tokens = ParseTokens(ReadLines(log));
    const std::string status = tokens.count("STATUS") ? tokens.at("STATUS") : std::string("(no status line)");
    const bool detected = exitCode == 1 && status == "ArmAfterInitBoundary";

    std::printf("late-arm fixture: child-exit=%d status=%s detected=%s timed-out=%s\n", exitCode, status.c_str(),
                detected ? "yes" : "NO", timedOut ? "yes" : "no");

    std::ofstream json(options.evidence / "late-arm-fixture.json", std::ios::binary);    json << "{\n";
    json << " \"case\": \"arm-after-init\",\n";
    json << " \"expectation\": \"nonzero exit with STATUS=ArmAfterInitBoundary, and no load attempt\",\n";
    json << " \"child_exit\": " << exitCode << ",\n";
    json << " \"expected_exit\": 1,\n";
    json << " \"status\": \"" << JsonEscape(status) << "\",\n";
    json << " \"detected\": " << (detected ? "true" : "false") << ",\n";
    json << " \"timed_out\": " << (timedOut ? "true" : "false") << ",\n";
    json << " \"harness_sha256\": \"" << Sha256Hex(OwnExecutablePath()) << "\",\n";
    json << " \"harness_source_sha256\": \"" << Sha256Hex(options.root / "tests" / "ampere_mfg_sidecar_harness.cpp")
         << "\",\n";
    json << " \"timestamp_utc\": \"" << NowIso8601Utc() << "\",\n";
    json << " \"log\": \"" << JsonEscape(Narrow(log.wstring())) << "\"\n";
    json << "}\n";
    json.close();

    if (!options.keepWork)
    {
        std::error_code cleanupError;
        fs::remove_all(workRoot, cleanupError);
    }

    return detected ? 1 : 4;
}

// ---------------------------------------------------------------------------
// Green cases: ordering, idempotence and the real-payload probe
// ---------------------------------------------------------------------------

static constexpr int kNoExitExpectation = -999;

struct CaseResult
{
    std::string name;
    std::string kind;
    std::string childArguments;
    fs::path log;
    fs::path stdio;
    int childExit = -1;
    int expectedExit = kNoExitExpectation;
    bool timedOut = false;
    bool ok = false;
    std::map<std::string, std::string> tokens;
    std::vector<std::string> missing;
};

static std::string TokenOr(const std::map<std::string, std::string>& tokens, const std::string& key,
                           const std::string& fallback)
{
    const auto it = tokens.find(key);
    return it == tokens.end() ? fallback : it->second;
}

static std::string FindLineStartingWith(const std::vector<std::string>& lines, const std::string& prefix)
{
    for (const std::string& line : lines)
    {
        if (StartsWith(line, prefix))
            return line;
    }

    return {};
}

static CaseResult RunCase(const Options& options, const std::string& name, const std::string& kind,
                          const fs::path& payload, const fs::path& ini, int expectedExit,
                          const std::vector<std::pair<std::string, std::string>>& expectedTokens, DWORD timeoutMs)
{
    CaseResult result;
    result.name = name;
    result.kind = kind;
    result.expectedExit = expectedExit;
    result.log = options.evidence / "logs" / (name + ".log");
    result.stdio = options.evidence / "logs" / (name + ".stdio.txt");

    std::error_code ec;
    fs::create_directories(result.log.parent_path(), ec);
    fs::create_directories(ini.parent_path(), ec);
    fs::remove(ini, ec);
    fs::remove(result.log, ec);

    result.childArguments = ChildArguments(name, payload, ini, result.log);
    result.childExit =
        RunChildProcess(OwnExecutablePath(), result.childArguments, result.stdio, timeoutMs, &result.timedOut);
    result.tokens = ParseTokens(ReadLines(result.log));

    for (const auto& expected : expectedTokens)
    {
        const std::string actual = TokenOr(result.tokens, expected.first, "(absent)");
        if (actual != expected.second)
            result.missing.push_back(expected.first + "=" + expected.second + " (saw " + actual + ")");
    }

    const bool exitOk = expectedExit == kNoExitExpectation || result.childExit == expectedExit;
    result.ok = !result.timedOut && exitOk && result.missing.empty();
    return result;
}

// A clean explicit load: a handle, the module's own role export reporting the
// ACTIVE role (1; 2 = standby, measured on the real payload in this todo), and a
// non-empty identity export. LoadLibrary alone is never enough (C5).
static bool IsCleanLoad(const CaseResult& result)
{
    return !result.timedOut && result.childExit == 0 && TokenOr(result.tokens, "handle", "") == "nonnull" &&
           TokenOr(result.tokens, "role", "") == "1" && TokenOr(result.tokens, "identity", "empty") != "empty";
}

// The ordering proof read straight out of the case log: the payload's own
// DllMain line and the loader's load-return line must precede the boundary line.
struct OrderEvidence
{
    std::string order;
    long long attachLine = -1;
    long long loadReturnLine = -1;
    long long boundaryLine = -1;
    std::string attachText;
};

static OrderEvidence CollectOrder(const fs::path& log)
{
    const std::vector<std::string> lines = ReadLines(log);
    OrderEvidence evidence;
    evidence.order = ComputeOrder(lines);

    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (StartsWith(lines[i], "payload:attach"))
        {
            evidence.attachLine = static_cast<long long>(i);
            evidence.attachText = lines[i];
        }
        if (StartsWith(lines[i], "event arm:load-return"))
            evidence.loadReturnLine = static_cast<long long>(i);
        if (StartsWith(lines[i], "event init:boundary"))
            evidence.boundaryLine = static_cast<long long>(i);
    }

    return evidence;
}

static std::string ReadWholeFile(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Reads the pinned sha256 of one staged file out of PIN.json (the pin stays the
// single source of truth; the harness never hardcodes the digest).
static std::string ReadPinnedSha256(const fs::path& pinPath, const std::string& fileName)
{
    const std::string text = ReadWholeFile(pinPath);
    const std::string needle = "\"name\": \"" + fileName + "\"";
    const size_t at = text.find(needle);

    if (at == std::string::npos)
        return {};

    const size_t hashAt = text.find("\"sha256\": \"", at);

    if (hashAt == std::string::npos)
        return {};

    const size_t start = hashAt + std::string("\"sha256\": \"").size();
    const size_t end = text.find('"', start);

    if (end == std::string::npos)
        return {};

    return text.substr(start, end - start);
}

// Replaces one top-level JSON line, keeping the file's indentation and every
// other byte intact.
static bool ReplaceJsonLine(std::string& text, const std::string& key, const std::string& replacement)
{
    const std::string marker = "\"" + key + "\":";
    size_t lineStart = 0;

    while (lineStart <= text.size())
    {
        const size_t lineEnd = text.find('\n', lineStart);
        const size_t end = lineEnd == std::string::npos ? text.size() : lineEnd;
        const std::string line = text.substr(lineStart, end - lineStart);

        if (StartsWith(Trim(line), marker))
        {
            const size_t indentEnd = line.find_first_not_of(" \t");
            const std::string indent = indentEnd == std::string::npos ? std::string() : line.substr(0, indentEnd);
            text.replace(lineStart, end - lineStart, indent + replacement);
            return true;
        }

        if (lineEnd == std::string::npos)
            break;

        lineStart = lineEnd + 1;
    }

    return false;
}

// Writes the chosen bundled_name plus a factual probe note into PIN.json. Todo 2
// is the single writer of that field.
static std::string WriteBundledName(const fs::path& pinPath, const std::string& chosen, const std::string& renamedSummary,
                                    const std::string& originalSummary, bool* wrote)
{
    const std::string before = ReadWholeFile(pinPath);
    std::string after = before;

    const bool nameSet = ReplaceJsonLine(after, "bundled_name", "\"bundled_name\": \"" + chosen + "\",");
    const std::string note = "\"bundled_name_note\": \"Decided by todo 2's explicit-load probe "
                             "(tests/ampere_mfg_sidecar_harness.cpp; receipts in "
                             ".omo/evidence/rtx2030-mfg-integration/02/). Renamed variant " +
                             renamedSummary + "; original-name variant " + originalSummary +
                             ". The name that loaded cleanly by explicit absolute path with "
                             "DlssgProxy_Role=1 (active) is shipped.\",";
    const bool noteSet = ReplaceJsonLine(after, "bundled_name_note", note);

    if (!nameSet || !noteSet)
        return "FAIL-CLOSED: could not locate the bundled_name/bundled_name_note lines in PIN.json";

    if (after == before)
    {
        if (wrote != nullptr)
            *wrote = false;
        return {};
    }

    std::ofstream out(pinPath, std::ios::binary);
    out << after;
    out.close();

    if (wrote != nullptr)
        *wrote = true;
    return {};
}

static std::string DiffLines(const std::string& before, const std::string& after)
{
    std::vector<std::string> beforeLines;
    std::vector<std::string> afterLines;
    {
        std::stringstream stream(before);
        std::string line;
        while (std::getline(stream, line))
            beforeLines.push_back(line);
    }
    {
        std::stringstream stream(after);
        std::string line;
        while (std::getline(stream, line))
            afterLines.push_back(line);
    }

    const size_t count = std::max(beforeLines.size(), afterLines.size());
    std::string out;

    for (size_t i = 0; i < count; ++i)
    {
        const std::string left = i < beforeLines.size() ? beforeLines[i] : std::string();
        const std::string right = i < afterLines.size() ? afterLines[i] : std::string();

        if (left == right)
            continue;

        out += "line " + std::to_string(i + 1) + ":\n- " + left + "\n+ " + right + "\n";
    }

    return out.empty() ? std::string("(no difference: PIN.json already carried the chosen name and note)\n") : out;
}

static std::string VariantSummary(const CaseResult& result)
{
    const std::string status = TokenOr(result.tokens, "STATUS", "(no status)");
    const std::string role = TokenOr(result.tokens, "role", "n/a");
    const std::string identity = TokenOr(result.tokens, "identity", "empty");

    if (result.timedOut)
        return "timed out";

    if (IsCleanLoad(result))
        return "loaded cleanly (status=" + status + ", role=" + role + ", identity=" + identity + ")";

    return "did not load cleanly (exit=" + std::to_string(result.childExit) + ", status=" + status + ", role=" + role +
           ", identity=" + identity + ")";
}

static int RunGreenCases(const Options& options)
{
    const fs::path stagedDir = options.root / "vendor" / "dlssg_sm86";
    const fs::path stagedDll = stagedDir / "dlssg_sm86.dll";
    const fs::path stagedIni = stagedDir / "dlssg_sm86.ini";
    const fs::path pinPath = stagedDir / "PIN.json";
    const fs::path workRoot = options.evidence / "work";
    const fs::path logDir = options.evidence / "logs";

    const std::string stagedBefore = Sha256Hex(stagedDll);
    const std::string iniBefore = Sha256Hex(stagedIni);
    const std::string pinnedSha = ReadPinnedSha256(pinPath, "dlssg_sm86.dll");

    std::printf("pinned payload: staged=%s pinned=%s\n", stagedBefore.c_str(), pinnedSha.c_str());

    if (pinnedSha.empty() || pinnedSha != stagedBefore)
    {
        std::printf("FAIL-CLOSED: the staged payload does not match the sha256 in PIN.json\n");
        return 66;
    }

    std::error_code ec;
    fs::create_directories(workRoot, ec);
    fs::create_directories(logDir, ec);

    // Scratch copies, one per variant. The staged file is never written to.
    const fs::path renamedDir = workRoot / "renamed";
    const fs::path originalDir = workRoot / "original";
    fs::create_directories(renamedDir, ec);
    fs::create_directories(originalDir, ec);
    fs::copy_file(stagedDll, renamedDir / "dlssg_sm86.dll", fs::copy_options::overwrite_existing, ec);
    fs::copy_file(stagedDll, originalDir / "version.dll", fs::copy_options::overwrite_existing, ec);

    const std::string renamedCopySha = Sha256Hex(renamedDir / "dlssg_sm86.dll");
    const std::string originalCopySha = Sha256Hex(originalDir / "version.dll");

    if (renamedCopySha != stagedBefore || originalCopySha != stagedBefore)
    {
        std::printf("FAIL-CLOSED: a scratch copy of the payload differs from the staged file\n");
        return 66;
    }

    std::vector<CaseResult> results;
    results.push_back(RunCase(options, "arm-before-init", "stub", options.stub, workRoot / "stub-before" / "dlssg_sm86.ini",
                              0,
                              {{"STATUS", "PayloadLoaded"},
                               {"order", "load-before-boundary"},
                               {"load_count", "1"},
                               {"ini_write_count", "1"},
                               {"stub_attach_count", "1"}},
                              30000));
    results.push_back(RunCase(options, "arm-after-init", "stub", options.stub, workRoot / "late-arm" / "dlssg_sm86.ini", 1,
                              {{"STATUS", "ArmAfterInitBoundary"},
                               {"order", "no-load-before-boundary"},
                               {"load_count", "0"},
                               {"ini_write_count", "0"},
                               {"stub_attach_count", "0"}},
                              30000));
    results.push_back(RunCase(options, "idempotence", "stub", options.stub, workRoot / "stub-idem" / "dlssg_sm86.ini", 0,
                              {{"STATUS", "PayloadLoaded"},
                               {"order", "load-before-boundary"},
                               {"load_count", "1"},
                               {"ini_write_count", "1"},
                               {"ini_rewritten", "false"},
                               {"stub_attach_count", "1"}},
                              30000));
    results.push_back(RunCase(options, "real-renamed", "real", renamedDir / "dlssg_sm86.dll",
                              renamedDir / "dlssg_sm86.ini", kNoExitExpectation, {}, 180000));
    results.push_back(RunCase(options, "real-original", "real", originalDir / "version.dll",
                              originalDir / "dlssg_sm86.ini", kNoExitExpectation, {}, 180000));

    const CaseResult& armBefore = results[0];
    const CaseResult& armAfter = results[1];
    const CaseResult& idempotence = results[2];
    const CaseResult& renamed = results[3];
    const CaseResult& original = results[4];

    const bool renamedClean = IsCleanLoad(renamed);
    const bool originalClean = IsCleanLoad(original);

    // Probe order is renamed first (todo 1's preferred branch); the first clean
    // load decides bundled_name.
    const std::string chosen = renamedClean ? std::string("dlssg_sm86.dll")
                              : originalClean ? std::string("version.dll")
                                              : std::string();

    const std::string stagedAfter = Sha256Hex(stagedDll);
    const std::string iniAfter = Sha256Hex(stagedIni);
    const bool payloadUntouched = stagedAfter == stagedBefore && iniAfter == iniBefore;

    bool wrotePin = false;
    std::string pinError;
    const std::string pinBefore = ReadWholeFile(pinPath);

    if (!chosen.empty())
        pinError = WriteBundledName(pinPath, chosen, VariantSummary(renamed), VariantSummary(original), &wrotePin);

    const std::string pinAfter = ReadWholeFile(pinPath);

    // ~58 MB of scratch copies do not belong in the evidence tree; the hashes of
    // both copies are captured in the receipt below.
    bool workRemoved = false;
    if (!options.keepWork)
    {
        std::error_code cleanupError;
        workRemoved = fs::remove_all(workRoot, cleanupError) > 0 && !cleanupError;
    }

    // ---- receipts -------------------------------------------------------
    const std::string harnessSha = Sha256Hex(OwnExecutablePath());
    const std::string harnessSourceSha = Sha256Hex(options.root / "tests" / "ampere_mfg_sidecar_harness.cpp");
    const std::string stubSourceSha = Sha256Hex(options.root / "tests" / "ampere_mfg_stub_payload.cpp");
    const std::string stamp = NowIso8601Utc();

    {
        std::ofstream json(options.evidence / "green-cases.json", std::ios::binary);
        json << "{\n";
        json << " \"harness_sha256\": \"" << harnessSha << "\",\n";
        json << " \"harness_source_sha256\": \"" << harnessSourceSha << "\",\n";
        json << " \"stub_source_sha256\": \"" << stubSourceSha << "\",\n";
        json << " \"root\": \"" << JsonEscape(Narrow(options.root.wstring())) << "\",\n";
        json << " \"timestamp_utc\": \"" << stamp << "\",\n";
        json << " \"pinned_payload_sha256\": \"" << stagedBefore << "\",\n";
        json << " \"cases\": [\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const CaseResult& item = results[i];
            json << "  { \"name\": \"" << item.name << "\", \"kind\": \"" << item.kind
                 << "\", \"child_exit\": " << item.childExit << ", \"expected_exit\": " << item.expectedExit
                 << ", \"ok\": " << (item.ok ? "true" : "false") << ", \"timed_out\": "
                 << (item.timedOut ? "true" : "false") << ", \"log\": \""
                 << JsonEscape(Narrow(item.log.wstring())) << "\", \"missing_expectations\": [";

            for (size_t k = 0; k < item.missing.size(); ++k)
                json << "\"" << JsonEscape(item.missing[k]) << "\"" << (k + 1 == item.missing.size() ? "" : ", ");

            json << "], \"tokens\": {";
            size_t written = 0;
            for (const auto& token : item.tokens)
            {
                json << "\"" << JsonEscape(token.first) << "\": \"" << JsonEscape(token.second) << "\""
                     << (++written == item.tokens.size() ? "" : ", ");
            }
            json << "} }" << (i + 1 == results.size() ? "\n" : ",\n");
        }

        const int failures = static_cast<int>(std::count_if(results.begin(), results.end(),
                                                           [](const CaseResult& item) { return !item.ok; }));
        json << " ],\n";
        json << " \"case_count\": " << results.size() << ",\n";
        json << " \"failures\": " << failures << "\n";
        json << "}\n";
    }

    {
        const OrderEvidence before = CollectOrder(armBefore.log);
        const OrderEvidence after = CollectOrder(armAfter.log);

        std::ofstream json(options.evidence / "ordering.json", std::ios::binary);
        json << "{\n";
        json << " \"note\": \"the payload's own DllMain line (payload:attach, written by the stub) and the "
                "loader's load-return line must both precede the init boundary line (C2)\",\n";
        json << " \"harness_sha256\": \"" << harnessSha << "\",\n";
        json << " \"harness_source_sha256\": \"" << harnessSourceSha << "\",\n";
        json << " \"cases\": [\n";
        json << "  { \"name\": \"arm-before-init\", \"order\": \"" << before.order << "\", \"attach_line_index\": "
             << before.attachLine << ", \"load_return_line_index\": " << before.loadReturnLine
             << ", \"boundary_line_index\": " << before.boundaryLine << ", \"attach_line\": \""
             << JsonEscape(before.attachText) << "\", \"ok\": " << (before.order == "load-before-boundary" ? "true" : "false")
             << " },\n";
        json << "  { \"name\": \"arm-after-init\", \"order\": \"" << after.order << "\", \"attach_line_index\": "
             << after.attachLine << ", \"load_return_line_index\": " << after.loadReturnLine
             << ", \"boundary_line_index\": " << after.boundaryLine << ", \"attach_line\": \""
             << JsonEscape(after.attachText) << "\", \"ok\": " << (after.order == "no-load-before-boundary" ? "true" : "false")
             << " }\n";
        json << " ],\n";
        json << " \"load_before_init_proven\": " << (before.order == "load-before-boundary" ? "true" : "false") << "\n";
        json << "}\n";
    }

    {
        const std::vector<std::string> lines = ReadLines(idempotence.log);

        std::ofstream json(options.evidence / "idempotence.json", std::ios::binary);
        json << "{\n";
        json << " \"scope\": \"counting is at the seam and stub level: the harness-local seam counts LoadLibraryW "
                "calls and companion-INI writes, and the stub payload counts its own DllMain attaches; the production "
                "loader's own counting and the real payload's INI handling belong to todos 3/11\",\n";
        json << " \"case\": \"idempotence\",\n";
        json << " \"harness_sha256\": \"" << harnessSha << "\",\n";
        json << " \"harness_source_sha256\": \"" << harnessSourceSha << "\",\n";
        json << " \"child_exit\": " << idempotence.childExit << ",\n";
        json << " \"load_count\": " << TokenOr(idempotence.tokens, "load_count", "0") << ",\n";
        json << " \"ini_write_count\": " << TokenOr(idempotence.tokens, "ini_write_count", "0") << ",\n";
        json << " \"stub_attach_count\": \"" << TokenOr(idempotence.tokens, "stub_attach_count", "n/a") << "\",\n";
        json << " \"ini_rewritten\": " << TokenOr(idempotence.tokens, "ini_rewritten", "n/a") << ",\n";
        json << " \"ini_bytes\": " << TokenOr(idempotence.tokens, "ini_bytes", "0") << ",\n";
        json << " \"ini_sha256_first_arm\": \"" << TokenOr(idempotence.tokens, "ini_sha256_first", "absent") << "\",\n";
        json << " \"ini_sha256_second_arm\": \"" << TokenOr(idempotence.tokens, "ini_sha256_second", "absent") << "\",\n";
        json << " \"second_arm_line\": \"" << JsonEscape(FindLineStartingWith(lines, "event arm:second")) << "\",\n";
        json << " \"second_arm_skipped_line\": \""
             << JsonEscape(FindLineStartingWith(lines, "event arm:skipped-already-armed")) << "\",\n";
        json << " \"ok\": " << (idempotence.ok ? "true" : "false") << "\n";
        json << "}\n";
    }

    {
        std::ofstream json(options.evidence / "real-payload-probe.json", std::ios::binary);
        json << "{\n";
        json << " \"rule\": \"each variant is loaded by EXPLICIT ABSOLUTE PATH in its own child process (the payload "
                "installs detours and takes a process-wide active-proxy marker, so one variant per process is the only "
                "clean observation). Probe order: renamed first. The first variant that loads cleanly (handle, "
                "DlssgProxy_Role=1 active, non-empty DlssgProxy_Name) decides bundled_name; if none loads cleanly, no "
                "bundled_name is written and the harness exits 3. LoadLibrary success is load evidence only (C5).\",\n";
        json << " \"harness_sha256\": \"" << harnessSha << "\",\n";
        json << " \"harness_source_sha256\": \"" << harnessSourceSha << "\",\n";
        json << " \"stub_source_sha256\": \"" << stubSourceSha << "\",\n";
        json << " \"timestamp_utc\": \"" << stamp << "\",\n";
        json << " \"staged_payload\": { \"path\": \"" << JsonEscape(Narrow(stagedDll.wstring())) << "\", \"sha256_before\": \""
             << stagedBefore << "\", \"sha256_after\": \"" << stagedAfter << "\", \"untouched\": "
             << (payloadUntouched ? "true" : "false") << " },\n";
        json << " \"staged_ini_untouched\": " << (iniAfter == iniBefore ? "true" : "false") << ",\n";
        json << " \"copies\": [\n";
        json << "  { \"variant\": \"renamed\", \"path\": \"" << JsonEscape(Narrow((renamedDir / "dlssg_sm86.dll").wstring()))
             << "\", \"sha256\": \"" << renamedCopySha << "\", \"equals_staged\": "
             << (renamedCopySha == stagedBefore ? "true" : "false") << " },\n";
        json << "  { \"variant\": \"original\", \"path\": \"" << JsonEscape(Narrow((originalDir / "version.dll").wstring()))
             << "\", \"sha256\": \"" << originalCopySha << "\", \"equals_staged\": "
             << (originalCopySha == stagedBefore ? "true" : "false") << " }\n";
        json << " ],\n";
        json << " \"variants\": [\n";

        for (size_t i = 0; i < 2; ++i)
        {
            const CaseResult& item = i == 0 ? renamed : original;
            json << "  { \"variant\": \"" << (i == 0 ? "renamed" : "original") << "\", \"case\": \"" << item.name
                 << "\", \"child_exit\": " << item.childExit << ", \"timed_out\": " << (item.timedOut ? "true" : "false")
                 << ", \"clean_load\": " << (IsCleanLoad(item) ? "true" : "false") << ", \"status\": \""
                 << JsonEscape(TokenOr(item.tokens, "STATUS", "")) << "\", \"handle\": \""
                 << JsonEscape(TokenOr(item.tokens, "handle", "")) << "\", \"role\": \""
                 << JsonEscape(TokenOr(item.tokens, "role", "")) << "\", \"identity\": \""
                 << JsonEscape(TokenOr(item.tokens, "identity", "")) << "\", \"order\": \""
                 << JsonEscape(TokenOr(item.tokens, "order", "")) << "\", \"log\": \""
                 << JsonEscape(Narrow(item.log.wstring())) << "\" }" << (i == 0 ? ",\n" : "\n");
        }

        json << " ],\n";
        json << " \"chosen_bundled_name\": " << (chosen.empty() ? "null" : "\"" + chosen + "\"") << ",\n";
        json << " \"decision\": \""
             << JsonEscape(chosen.empty() ? std::string("no variant loaded cleanly: bundled_name not written")
                                         : (renamedClean ? std::string("renamed variant loaded cleanly first")
                                                         : std::string("renamed variant did not load cleanly; the original name did")))
             << "\",\n";
        json << " \"pin_path\": \"" << JsonEscape(Narrow(pinPath.wstring())) << "\",\n";
        json << " \"pin_write_attempted\": " << (chosen.empty() ? "false" : "true") << ",\n";
        json << " \"pin_written\": " << (wrotePin ? "true" : "false") << ",\n";
        json << " \"pin_error\": \"" << JsonEscape(pinError) << "\",\n";
        json << " \"work_dir\": { \"path\": \"" << JsonEscape(Narrow(workRoot.wstring())) << "\", \"kept\": "
             << (options.keepWork ? "true" : "false") << ", \"removed\": " << (workRemoved ? "true" : "false")
             << " }\n";
        json << "}\n";
    }

    {
        // The decision artifacts are the FIRST-write ones: a re-run that does not
        // change PIN.json keeps them and only appends a no-change line, so the
        // original null -> chosen diff is never lost.
        const bool decisionFilesExist = FileExists(options.evidence / "PIN.json.before") &&
                                        FileExists(options.evidence / "PIN.json.after");

        if (wrotePin || !decisionFilesExist)
        {
            std::ofstream beforeFile(options.evidence / "PIN.json.before", std::ios::binary);
            beforeFile << pinBefore;
            beforeFile.close();

            std::ofstream afterFile(options.evidence / "PIN.json.after", std::ios::binary);
            afterFile << pinAfter;
            afterFile.close();
        }

        const fs::path diffPath = options.evidence / "pin.json.diff";

        if (wrotePin || !FileExists(diffPath))
        {
            std::ofstream diffFile(diffPath, std::ios::binary);
            diffFile << "# PIN.json bundled_name/bundled_name_note diff (todo 2 is the single writer of the name)\n";
            diffFile << DiffLines(pinBefore, pinAfter);
            diffFile.close();
        }
        else
        {
            std::ofstream diffFile(diffPath, std::ios::binary | std::ios::app);
            diffFile << "# re-run " << stamp << ": PIN.json already carried \"" << chosen
                     << "\"; no bytes changed (idempotent re-run)\n";
            diffFile.close();
        }
    }

    std::printf("\n%-16s %-6s %s\n", "case", "exit", "result");

    for (const CaseResult& item : results)
    {
        std::printf("%-16s %-6d %s%s\n", item.name.c_str(), item.childExit, item.ok ? "ok" : "FAIL",
                    item.missing.empty() ? "" : (" missing: " + item.missing[0]).c_str());
    }

    const int failures = static_cast<int>(std::count_if(results.begin(), results.end(),
                                                       [](const CaseResult& item) { return !item.ok; }));

    if (!payloadUntouched)
    {
        std::printf("FAIL: the staged payload changed during the probe\n");
        return 1;
    }

    if (chosen.empty())
    {
        std::printf("no variant loaded cleanly; bundled_name NOT written\n");
        return 3;
    }

    if (!pinError.empty())
    {
        std::printf("%s\n", pinError.c_str());
        return 1;
    }

    std::printf("bundled_name=%s (pin %s)\n", chosen.c_str(), wrotePin ? "updated" : "already carried this value");
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Production cases (todo 6)
// ---------------------------------------------------------------------------

// The expectations per case and revision. `red` is the pre-todo-6 production path: it validates the payload
// and writes the companion INI but stops there, so no module is loaded and no handle exists.
static std::vector<std::pair<std::string, std::string>> ProductionExpectations(const std::string& name, bool red)
{
    if (name == "production-plain")
        return { { "STATUS", "Disabled" },
                 { "verdict", "Disabled" },
                 { "load_count", "0" },
                 { "handle", "null" },
                 { "ini_written", "false" },
                 { "enabled", "false" },
                 { "hook_installed", "true" },
                 { "order", "no-load-before-boundary" } };

    if (name == "production-external" || name == "production-idempotence")
    {
        if (red)
        {
            std::vector<std::pair<std::string, std::string>> expected = { { "STATUS", "PayloadValidated" },
                                                                         { "verdict", "Eligible" },
                                                                         { "load_count", "1" },
                                                                         { "handle", "null" },
                                                                         { "ini_written", "true" },
                                                                         { "enabled", "false" },
                                                                         { "hook_installed", "true" },
                                                                         { "order", "no-load-before-boundary" },
                                                                         { "stub_attach_count", "n/a" } };

            if (name == "production-idempotence")
            {
                expected.push_back({ "ini_rewritten", "false" });
                expected.push_back({ "load_count", "1" });
            }

            return expected;
        }

        std::vector<std::pair<std::string, std::string>> expected = { { "STATUS", "Loaded" },
                                                                     { "verdict", "Eligible" },
                                                                     { "load_count", "1" },
                                                                     { "handle", "nonnull" },
                                                                     { "ini_written", "true" },
                                                                     { "enabled", "true" },
                                                                     { "hook_installed", "true" },
                                                                     { "order", "load-before-boundary" },
                                                                     { "stub_attach_count", "1" },
                                                                     { "role", "1" } };

        if (name == "production-idempotence")
        {
            expected.push_back({ "ini_rewritten", "false" });
            expected.push_back({ "boundary_crossed_before_arm", "false" });
        }

        return expected;
    }

    if (name == "production-external-off")
        return { { "STATUS", "Conflict" },
                 { "verdict", "Conflict" },
                 { "load_count", "0" },
                 { "handle", "null" },
                 { "ini_written", "false" },
                 { "enabled", "false" },
                 { "hook_installed", "true" },
                 { "order", "no-load-before-boundary" } };

    if (name == "production-missing-payload")
        return { { "STATUS", "PayloadMissing" },
                 { "verdict", "Eligible" },
                 { "load_count", "1" },
                 { "handle", "null" },
                 { "ini_written", "false" },
                 { "enabled", "false" },
                 { "hook_installed", "true" },
                 { "order", "no-load-before-boundary" } };

    if (name == "production-fork-guard")
        return { { "STATUS", "HookNotInstalled" },
                 { "hook_installed", "false" },
                 { "load_count", "0" },
                 { "handle", "null" },
                 { "ini_written", "false" },
                 { "enabled", "false" },
                 { "order", "no-load-before-boundary" } };

    if (name == "production-real-external")
    {
        // The pinned payload carries no harness attach counter and does not write into the case log, so the order
        // token is not asserted here; the load itself is: handle + the payload's own active role + the companion
        // INI written by the shipped generator. The stub cases above own the ordering proof.
        if (red)
            return { { "STATUS", "PayloadValidated" },
                     { "verdict", "Eligible" },
                     { "load_count", "1" },
                     { "handle", "null" },
                     { "ini_written", "true" },
                     { "enabled", "false" },
                     { "hook_installed", "true" },
                     { "payload_kind", "pinned" } };

        return { { "STATUS", "Loaded" },
                 { "verdict", "Eligible" },
                 { "load_count", "1" },
                 { "handle", "nonnull" },
                 { "ini_written", "true" },
                 { "enabled", "true" },
                 { "hook_installed", "true" },
                 { "role", "1" },
                 { "payload_kind", "pinned" } };
    }

    return {};
}

static CaseResult RunProductionCase(const Options& options, const std::string& name, bool red, DWORD timeoutMs)
{
    CaseResult result;
    result.name = name;
    result.kind = red ? "production-red" : "production";
    result.expectedExit = 0;
    result.log = options.evidence / "logs" / (name + (red ? ".red" : std::string()) + ".log");
    result.stdio = options.evidence / "logs" / (name + (red ? ".red" : std::string()) + ".stdio.txt");

    const fs::path scratch = options.evidence / "work" / (name + (red ? "-red" : std::string()));

    std::error_code ec;
    fs::create_directories(result.log.parent_path(), ec);
    fs::create_directories(scratch, ec);
    fs::remove(result.log, ec);

    if (name == "production-real-external")
    {
        // One scratch copy of the PINNED payload, under the shipped name, so the shipped arming path is what
        // loads it. The staged file itself is never written to.
        const fs::path staged = options.root / "vendor" / "dlssg_sm86" / "dlssg_sm86.dll";
        const fs::path target = scratch / "OptiScaler" / "dlssg_sm86" / "dlssg_sm86.dll";

        fs::create_directories(target.parent_path(), ec);
        fs::copy_file(staged, target, fs::copy_options::overwrite_existing, ec);

        if (!FileExists(target) || Sha256Hex(target) != Sha256Hex(staged))
        {
            std::printf("FAIL-CLOSED: the scratch copy of the pinned payload does not match the staged file\n");
            result.missing.push_back("pinned payload copy");
            result.ok = false;
            return result;
        }
    }

    result.childArguments = "--child " + name + " --root " + Quote(scratch) + " --payload " + Quote(options.stub) +
                            " --log " + Quote(result.log) + " --production" +
                            (name == "production-fork-guard" ? " --simulate-fork-guard" : "");

    result.childExit =
        RunChildProcess(OwnExecutablePath(), result.childArguments, result.stdio, timeoutMs, &result.timedOut);
    result.tokens = ParseTokens(ReadLines(result.log));

    for (const auto& expected : ProductionExpectations(name, red))
    {
        const std::string actual = TokenOr(result.tokens, expected.first, "(absent)");
        if (actual != expected.second)
            result.missing.push_back(expected.first + "=" + expected.second + " (saw " + actual + ")");
    }

    result.ok = !result.timedOut && result.childExit == 0 && result.missing.empty();
    return result;
}

// Runs the production cases. GREEN (default): every expectation of the shipped behaviour. RED
// (--production-red): every expectation of the PRE-todo-6 behaviour, so the run confirms the RED state this
// todo recorded first; exit 3 when confirmed, 4 when it is not (then the RED run proved nothing).
static int RunProductionCases(const Options& options)
{
    const char* names[] = { "production-plain",         "production-external",   "production-external-off",
                           "production-missing-payload", "production-idempotence", "production-real-external" };

    std::vector<CaseResult> results;

    for (const char* name : names)
    {
        const DWORD timeout = std::string(name) == "production-real-external" ? 180000 : 45000;
        results.push_back(RunProductionCase(options, name, options.productionRed, timeout));
    }

    const int failures = static_cast<int>(
        std::count_if(results.begin(), results.end(), [](const CaseResult& item) { return !item.ok; }));

    std::printf("\n%-28s %-6s %s\n", "production case", "exit", "result");

    for (const CaseResult& item : results)
    {
        std::printf("%-28s %-6d %s%s\n", item.name.c_str(), item.childExit, item.ok ? "ok" : "FAIL",
                    item.missing.empty() ? "" : (" missing: " + item.missing[0]).c_str());
    }

    {
        std::ofstream json(options.evidence / "production-cases.json", std::ios::binary);
        json << "{\n";
        json << " \"mode\": \"" << (options.productionRed ? "red-prefix" : "green") << "\",\n";
        json << " \"rule\": \"the production loader OptiScaler/framegen/dlssg/AmpereMfgLoader.cpp is the code "
                "under test; each case runs Arm() in a child process with fixture host facts and the arming call "
                "placed BEFORE the simulated Streamline init boundary. red-prefix = the pre-todo-6 path (no load) "
                "that the todo-6 wiring has to flip.\",\n";
        json << " \"harness_sha256\": \"" << Sha256Hex(OwnExecutablePath()) << "\",\n";
        json << " \"harness_source_sha256\": \""
             << Sha256Hex(options.root / "tests" / "ampere_mfg_sidecar_harness.cpp") << "\",\n";
        json << " \"loader_source_sha256\": \""
             << Sha256Hex(options.root / "OptiScaler" / "framegen" / "dlssg" / "AmpereMfgLoader.cpp") << "\",\n";
        json << " \"loader_header_sha256\": \""
             << Sha256Hex(options.root / "OptiScaler" / "framegen" / "dlssg" / "AmpereMfgLoader.h") << "\",\n";
        json << " \"timestamp_utc\": \"" << NowIso8601Utc() << "\",\n";
        json << " \"cases\": [\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const CaseResult& item = results[i];
            json << "  { \"name\": \"" << item.name << "\", \"child_exit\": " << item.childExit
                 << ", \"ok\": " << (item.ok ? "true" : "false") << ", \"timed_out\": "
                 << (item.timedOut ? "true" : "false") << ", \"log\": \"" << JsonEscape(Narrow(item.log.wstring()))
                 << "\", \"missing_expectations\": [";

            for (size_t k = 0; k < item.missing.size(); ++k)
                json << "\"" << JsonEscape(item.missing[k]) << "\"" << (k + 1 == item.missing.size() ? "" : ", ");

            json << "], \"tokens\": {";
            size_t written = 0;
            for (const auto& token : item.tokens)
            {
                json << "\"" << JsonEscape(token.first) << "\": \"" << JsonEscape(token.second) << "\""
                     << (++written == item.tokens.size() ? "" : ", ");
            }
            json << "} }" << (i + 1 == results.size() ? "\n" : ",\n");
        }

        json << " ],\n";
        json << " \"case_count\": " << results.size() << ",\n";
        json << " \"failures\": " << failures << "\n";
        json << "}\n";
    }

    if (!options.keepWork)
    {
        std::error_code cleanupError;
        fs::remove_all(options.evidence / "work", cleanupError);
    }

    if (options.productionRed)
        return failures == 0 ? 3 : 4;

    return failures == 0 ? 0 : 1;
}

// One production case by name, with its own expectation mode. `confirmFailure` inverts the exit code for a
// fixture that MUST fail: 1 = confirmed with the named status, 0 = the fixture did not fail.
static int RunProductionSingle(const Options& options, const std::string& name, bool confirmFailure)
{
    const bool red = false;
    const CaseResult result = RunProductionCase(options, name, red, 45000);

    std::printf("\n%-28s %-6s %s\n", "production case", "exit", "result");
    std::printf("%-28s %-6d %s%s\n", result.name.c_str(), result.childExit, result.ok ? "ok" : "FAIL",
                result.missing.empty() ? "" : (" missing: " + result.missing[0]).c_str());

    {
        std::ofstream json(options.evidence / (name + ".json"), std::ios::binary);
        json << "{\n";
        json << " \"case\": \"" << name << "\",\n";
        json << " \"mode\": \"" << (confirmFailure ? "failure-fixture" : "green") << "\",\n";
        json << " \"child_exit\": " << result.childExit << ",\n";
        json << " \"ok\": " << (result.ok ? "true" : "false") << ",\n";
        json << " \"expected_ok\": " << (confirmFailure ? "false" : "true") << ",\n";
        json << " \"log\": \"" << JsonEscape(Narrow(result.log.wstring())) << "\",\n";
        json << " \"status\": \"" << JsonEscape(TokenOr(result.tokens, "STATUS", "")) << "\",\n";
        json << " \"hook_installed\": \"" << JsonEscape(TokenOr(result.tokens, "hook_installed", "")) << "\",\n";
        json << " \"harness_source_sha256\": \""
             << Sha256Hex(options.root / "tests" / "ampere_mfg_sidecar_harness.cpp") << "\"\n";
        json << "}\n";
    }

    std::error_code cleanupError;
    fs::remove_all(options.evidence / "work" / name, cleanupError);

    if (confirmFailure)
        return result.ok ? 1 : 0;

    return result.ok ? 0 : 1;
}

int wmain(int argc, wchar_t** argv)
{
    std::vector<std::string> arguments;

    for (int i = 1; i < argc; ++i)
        arguments.push_back(Narrow(argv[i]));

    if (arguments.empty())
        return Usage();

    if (arguments[0] == "--child")
    {
        ChildOptions options;
        bool production = false;
        bool forkGuard = false;
        fs::path productionRoot;

        for (size_t i = 0; i < arguments.size(); ++i)
        {
            const bool hasValue = i + 1 < arguments.size();
            if (arguments[i] == "--child" && hasValue)
                options.caseName = arguments[++i];
            else if (arguments[i] == "--payload" && hasValue)
                options.payload = Wide(arguments[++i]);
            else if (arguments[i] == "--ini" && hasValue)
                options.ini = Wide(arguments[++i]);
            else if (arguments[i] == "--log" && hasValue)
                options.log = Wide(arguments[++i]);
            else if (arguments[i] == "--root" && hasValue)
                productionRoot = Wide(arguments[++i]);
            else if (arguments[i] == "--production")
                production = true;
            else if (arguments[i] == "--simulate-fork-guard")
                forkGuard = true;
            else
            {
                std::printf("unknown child argument: %s\n", arguments[i].c_str());
                return 64;
            }
        }

        if (production)
        {
            if (options.caseName.empty() || options.payload.empty() || options.log.empty() || productionRoot.empty())
            {
                std::printf("production child mode needs --child, --root, --payload and --log\n");
                return 64;
            }

            ProductionCaseSpec spec = ProductionSpec(options.caseName);
            spec.forkGuard = spec.forkGuard || forkGuard;
            return RunProductionChildCase(spec, productionRoot, options.payload, options.log);
        }

        if (options.caseName.empty() || options.payload.empty() || options.ini.empty() || options.log.empty())
        {
            std::printf("child mode needs --child, --payload, --ini and --log\n");
            return 64;
        }

        return RunChildCase(options);
    }

    Options options;

    for (size_t i = 0; i < arguments.size(); ++i)
    {
        const bool hasValue = i + 1 < arguments.size();
        if (arguments[i] == "--root" && hasValue)
            options.root = Wide(arguments[++i]);
        else if (arguments[i] == "--evidence" && hasValue)
            options.evidence = Wide(arguments[++i]);
        else if (arguments[i] == "--stub" && hasValue)
            options.stub = Wide(arguments[++i]);
        else if (arguments[i] == "--late-arm")
            options.lateArmOnly = true;
        else if (arguments[i] == "--production-red")
            options.productionRed = true;
        else if (arguments[i] == "--production-only")
            options.productionOnly = true;
        else if (arguments[i] == "--external-mode")
            options.externalModeOnly = true;
        else if (arguments[i] == "--fork-guard")
            options.forkGuardOnly = true;
        else if (arguments[i] == "--keep-work")
            options.keepWork = true;
        else
        {
            std::printf("unknown argument: %s\n", arguments[i].c_str());
            return Usage();
        }
    }

    // Fail closed on the environment: the resolved root has to look like this
    // repository, and nothing may point at a live game install.
    if (options.root.empty() || options.evidence.empty())
    {
        std::printf("--root and --evidence are required\n");
        return Usage();
    }

    if (!FileExists(options.root / "vendor" / "dlssg_sm86" / "PIN.json") ||
        !FileExists(options.root / "vendor" / "dlssg_sm86" / "dlssg_sm86.dll") ||
        !FileExists(options.root / "docs" / "rtx2030-payload-contract.md"))
    {
        std::printf("FAIL-CLOSED: %s does not look like the repository root (pinned payload or contract doc missing)\n",
                    Narrow(options.root.wstring()).c_str());
        return 66;
    }

    if (LooksLikeLiveInstall(Narrow(options.root.wstring())) || LooksLikeLiveInstall(Narrow(options.evidence.wstring())))
    {
        std::printf("FAIL-CLOSED: refusing to run against a live install path\n");
        return 66;
    }

    if (options.stub.empty())
        options.stub = OwnExecutablePath().parent_path() / "ampere_mfg_stub_payload.dll";

    if (!FileExists(options.stub))
    {
        std::printf("FAIL-CLOSED: stub payload missing: %s\n", Narrow(options.stub.wstring()).c_str());
        return 66;
    }

    if (!options.lateArmOnly && FileExists(OwnExecutablePath().parent_path() / "version.dll"))
    {
        std::printf("FAIL-CLOSED: a version.dll sits next to this executable, which would make the payload forward to "
                    "itself; the probe must run from a directory without one\n");
        return 66;
    }

    std::error_code ec;
    fs::create_directories(options.evidence, ec);

    if (options.lateArmOnly)
        return RunLateArmFixture(options);

    if (options.productionRed || options.productionOnly)
        return RunProductionCases(options);

    if (options.externalModeOnly)
        return RunProductionSingle(options, "production-external", false);

    if (options.forkGuardOnly)
        return RunProductionSingle(options, "production-fork-guard", true);

    const int green = RunGreenCases(options);

    if (green != 0)
        return green;

    return RunProductionCases(options);
}
