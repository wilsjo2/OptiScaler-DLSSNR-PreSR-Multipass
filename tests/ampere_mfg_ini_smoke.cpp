// Host check of the companion-INI generator and the payload-discovery contract in AmpereMfgLoader.h.
// Header-only: no GPU, no payload, no game, no OptiScaler headers.
//
//   cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /DOPTISCALER_RTX40_MFG tests/ampere_mfg_ini_smoke.cpp
//
// Cases, all against the real v0.3.5 schema (docs/rtx2030-payload-contract.md):
//   * MaxGeneratedFrames / Optimized / log level clamping at both bounds (low and high fixtures),
//   * Router / Mode / KernelImage acceptance - including the Auto and PTX cases,
//   * the exact bytes of the generated file for the default settings (0.3.5 layout, no 0.2.x native keys),
//   * payload discovery over a scratch tree in both shipped layouts,
//   * the status vocabulary.
// Receipts go to --evidence <dir> when asked: smoke-cases.json (every row), clamp-fixtures.json (the clamp
// rows with input/expected/observed) and schema-golden.ini (the exact bytes of the default generated INI).
#include "../OptiScaler/framegen/dlssg/AmpereMfgLoader.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace AmpereMfgLoader;

namespace
{
int fails = 0;

struct Row
{
    std::string group;
    std::string name;
    std::string expected;
    std::string observed;
    bool ok;
};

std::vector<Row> rows;

void Record(const char* group, const std::string& name, const std::string& expected, const std::string& observed)
{
    const bool ok = expected == observed;

    if (!ok)
    {
        std::printf("FAIL %s/%s: expected [%s] observed [%s]\n", group, name.c_str(), expected.c_str(),
                    observed.c_str());
        ++fails;
    }

    rows.push_back({group, name, expected, observed, ok});
}

bool Contains(const std::string& text, const std::string& what)
{
    return text.find(what) != std::string::npos;
}

std::string Present(bool present)
{
    return present ? "present" : "absent";
}

std::string Flag(bool value)
{
    return value ? "true" : "false";
}

std::string NowIso()
{
    const std::time_t now = std::time(nullptr);
    std::tm parts {};
    gmtime_s(&parts, &now);

    char buffer[32] {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &parts);
    return buffer;
}

std::string Escape(const std::string& value)
{
    std::string out;

    for (const char character : value)
    {
        switch (character)
        {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += character; break;
        }
    }

    return out;
}

bool WriteText(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << text;
    return static_cast<bool>(file);
}

std::string RelativeTo(const std::filesystem::path& path, const std::filesystem::path& root)
{
    std::error_code error;
    const auto relative = std::filesystem::relative(path, root, error);
    return error ? std::string("<unresolvable>") : relative.string();
}

// The exact file the loader writes for the default settings. Byte-for-byte, LF only.
const char* const kSchemaGolden =
    "; dlssg_sm86.ini - written by OptiScaler from your settings before the payload is loaded.\n"
    "; The other keys of the 0.3.5 schema keep their defaults; restart the game after a change.\n"
    "\n"
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
    "[Logging]\n"
    "Level=1\n"
    "Directory=dlssg_sm86\\logs\n"
    "\n"
    "[Runtime]\n"
    "Mode=Bundled\n";

void ClampIntCases(const char* group, int (*clamp)(int), const std::vector<std::pair<int, int>>& cases)
{
    for (const auto& [input, expected] : cases)
    {
        Record(group, "Clamp(" + std::to_string(input) + ")", std::to_string(expected),
               std::to_string(clamp(input)));
    }
}

void EmittedIntCases(const char* group, const char* key, const std::vector<std::pair<int, std::string>>& cases)
{
    for (const auto& [input, expected] : cases)
    {
        IniSettings settings {};
        settings.MaxGeneratedFrames = input;

        Record(group, key + std::string(" for ") + std::to_string(input), expected,
               [&]
               {
                   // The emitted line, reduced to the value so a failure prints something readable.
                   const std::string ini = FormatIniContent(settings);
                   const std::string prefix = std::string(key) + "=";
                   const auto at = ini.find(prefix);

                   if (at == std::string::npos)
                       return std::string("absent");

                   const auto end = ini.find('\n', at);
                   std::string value = ini.substr(at + prefix.size(), end - at - prefix.size());
                   return value.empty() ? std::string("absent") : value;
               }());
    }
}

void KernelImageCases(const std::vector<std::pair<std::string, std::string>>& cases)
{
    for (const auto& [input, expected] : cases)
    {
        IniSettings settings {};
        settings.KernelImage = input;

        const std::string ini = FormatIniContent(settings);
        std::string observed = "absent";
        const auto at = ini.find("KernelImage=");

        if (at != std::string::npos)
        {
            const auto end = ini.find('\n', at);
            observed = ini.substr(at + std::strlen("KernelImage="), end - at - std::strlen("KernelImage="));
        }

        Record("kernel-image", "[" + input + "]", expected, observed);
    }
}

void DiscoveryCases(const std::filesystem::path& evidenceDir)
{
    const std::filesystem::path scratch = evidenceDir / "scratch" / "discovery";
    std::error_code error;
    std::filesystem::remove_all(scratch, error);

    const std::filesystem::path shipped = scratch / "both" / "OptiScaler" / "dlssg_sm86";
    const std::filesystem::path flat = scratch / "flat" / "dlssg_sm86";
    std::filesystem::create_directories(shipped, error);
    std::filesystem::create_directories(flat, error);
    WriteText(shipped / kPayloadModuleName, "MZ");
    WriteText(flat / kPayloadModuleName, "MZ");

    // Both layouts installed: the shipped release layout wins.
    {
        std::vector<std::filesystem::path> tried;
        const auto found = FindPayloadModule(scratch / "both", tried);
        Record("discovery", "shipped-layout-wins", "OptiScaler\\dlssg_sm86",
               found.empty() ? std::string("<not found>") : RelativeTo(found.parent_path(), scratch / "both"));
        Record("discovery", "shipped-layout-wins/checks", "1", std::to_string(tried.size()));
    }

    // Flat install only: the folder beside the DLL.
    {
        std::vector<std::filesystem::path> tried;
        const auto found = FindPayloadModule(scratch / "flat", tried);
        Record("discovery", "flat-layout", "dlssg_sm86",
               found.empty() ? std::string("<not found>") : RelativeTo(found.parent_path(), scratch / "flat"));
        Record("discovery", "flat-layout/checks", "2", std::to_string(tried.size()));
    }

    // Nothing installed: no module, and every candidate was checked in order.
    {
        std::vector<std::filesystem::path> tried;
        const auto found = FindPayloadModule(scratch / "empty", tried);
        Record("discovery", "empty/result", "<not found>", found.empty() ? "<not found>" : found.string());
        Record("discovery", "empty/checks", "2", std::to_string(tried.size()));

        if (tried.size() == 2)
        {
            Record("discovery", "empty/candidate0", "OptiScaler\\dlssg_sm86",
                   RelativeTo(tried[0].parent_path(), scratch / "empty"));
            Record("discovery", "empty/candidate1", "dlssg_sm86",
                   RelativeTo(tried[1].parent_path(), scratch / "empty"));
        }
    }
}

void VocabularyCases()
{
    const std::pair<State, const char*> expected[] = {
        { State::Disabled, "Disabled" },
        { State::Ineligible, "Ineligible" },
        { State::Conflict, "Conflict" },
        { State::PayloadMissing, "PayloadMissing" },
        { State::PayloadIncomplete, "PayloadIncomplete" },
        { State::PayloadLoadFailed, "PayloadLoadFailed" },
        { State::PayloadStandby, "PayloadStandby" },
        { State::IniWriteFailed, "IniWriteFailed" },
        { State::PayloadValidated, "PayloadValidated" },
        { State::Loaded, "Loaded" },
        { State::BackendInstalled, "BackendInstalled" },
        { State::FeatureCreated, "FeatureCreated" },
        { State::Evaluating, "Evaluating" },
        { State::Presenting, "Presenting" },
    };

    for (const auto& [state, name] : expected)
        Record("state-vocabulary", name, name, StateName(state));

    const Status fresh {};
    Record("state-vocabulary", "default-state", "Disabled", StateName(fresh.state));
    Record("state-vocabulary", "default-state-text", "Disabled", fresh.StateText());
}
} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path evidenceDir;
    std::string root;

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];

        if (argument == "--evidence" && index + 1 < argc)
            evidenceDir = argv[++index];
        else if (argument == "--root" && index + 1 < argc)
            root = argv[++index];
        else
        {
            std::printf("usage: ampere_mfg_ini_smoke [--evidence <dir>] [--root <repo>]\n");
            return 64;
        }
    }

    // 1. MaxGeneratedFrames: the schema allows 0..5; both bounds saturate.
    ClampIntCases("max-generated-frames", &ClampMaxGeneratedFrames,
                  { { -2147483647 - 1, 0 }, { -3, 0 },     { -1, 0 }, { 0, 0 },          { 1, 1 },
                    { 2, 2 },              { 3, 3 },      { 4, 4 },  { 5, 5 },
                    { 6, 5 },              { 9, 5 },      { 2147483647, 5 } });

    EmittedIntCases("max-generated-frames", "MaxGeneratedFrames",
                    { { -3, "0" }, { 0, "0" }, { 1, "1" }, { 3, "3" }, { 5, "5" }, { 9, "5" } });

    // 2. Optimized: in range as asked, otherwise the payload's documented fallback, 1.
    ClampIntCases("optimized", &ClampOptimized,
                  { { -5, 1 }, { -1, 1 }, { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 1 }, { 99, 1 } });

    // 3. Router: the three schema values, otherwise Auto.
    for (const auto& [input, expected] :
         std::vector<std::pair<std::string, std::string>> {
             { "Auto", "Auto" }, { "SM75", "SM75" }, { "SM86", "SM86" }, { "", "Auto" },
             { "SM80", "Auto" }, { "sm75", "Auto" }, { "Unknown", "Auto" } })
    {
        Record("router", "[" + input + "]", expected, ClampRouter(input));
    }

    {
        IniSettings settings {};
        settings.Router = "SM75";
        const std::string ini = FormatIniContent(settings);
        Record("router", "emitted for SM75", "present", Present(Contains(ini, "\nRouter=SM75\n")));
    }

    // 4. KernelImage: PTX and Cubin are written, Auto (and anything else) leaves the key out.
    KernelImageCases({ { "PTX", "PTX" },
                       { "Cubin", "Cubin" },
                       { "Auto", "absent" },
                       { "", "absent" },
                       { "ptx", "absent" },
                       { "Pascal", "absent" } });

    // 5. Mode: the three schema values, otherwise Bundled.
    for (const auto& [input, expected] :
         std::vector<std::pair<std::string, std::string>> { { "Bundled", "Bundled" },
                                                            { "Auto", "Auto" },
                                                            { "Pinned", "Pinned" },
                                                            { "", "Bundled" },
                                                            { "Cache", "Bundled" } })
    {
        Record("mode", "[" + input + "]", expected, ClampMode(input));
    }

    // 6. Log level: 0..3, saturating.
    ClampIntCases("log-level", &ClampLogLevel, { { -1, 0 }, { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 }, { 9, 3 } });

    {
        IniSettings settings {};
        settings.LogLevel = 2;
        Record("log-level", "emitted for 2", "present", Present(Contains(FormatIniContent(settings), "\nLevel=2\n")));
    }

    // 7. The generated file for the default settings is the 0.3.5 schema, byte for byte.
    {
        const std::string ini = FormatIniContent(IniSettings {});
        Record("schema-shape", "default-golden", "identical", ini == kSchemaGolden ? "identical" : "different");
        Record("schema-shape", "lf-only", "true", Flag(ini.find('\r') == std::string::npos));
        Record("schema-shape", "ends-with-newline", "true", Flag(!ini.empty() && ini.back() == '\n'));
        Record("schema-shape", "Enabled=1", "present", Present(Contains(ini, "\nEnabled=1\n")));
        Record("schema-shape", "Directory=dlssg_sm86\\logs", "present",
               Present(Contains(ini, "\nDirectory=dlssg_sm86\\logs\n")));

        // The retired 0.2.x native layout and the keys this loader must never write.
        for (const auto* absent : { "[Backends]", "KernelImage=", "HardwareBilinear=", "ForceSM86Route",
                                    "SimulateAmpere", "SpoofArchToGame", "SM75Family", "Preset=", "CacheDirectory" })
        {
            Record("schema-shape", std::string("no ") + absent, "absent", Present(Contains(ini, absent)));
        }

        // Section order, the one thing a payload INI parser may notice.
        const auto general = ini.find("[General]");
        const auto frameGeneration = ini.find("[FrameGeneration]");
        const auto compatibility = ini.find("[Compatibility]");
        const auto logging = ini.find("[Logging]");
        const auto runtime = ini.find("[Runtime]");
        const bool ordered = general < frameGeneration && frameGeneration < compatibility &&
                             compatibility < logging && logging < runtime;

        Record("schema-shape", "section-order", "true", Flag(ordered));
    }

    // 8. Discovery over a scratch tree: the shipped layout first, the flat one second.
    {
        std::wstring moduleName(kPayloadModuleName);
        Record("names", "module-name-is-bare", "true",
               Flag(moduleName.find_first_of(L"\\/") == std::wstring::npos && !moduleName.empty()));
        Record("names", "module-name-is-a-dll", "true",
               Flag(moduleName.size() > 4 && moduleName.ends_with(L".dll")));

        std::wstring iniName(kPayloadIniName);
        Record("names", "ini-name-is-bare", "true",
               Flag(iniName.find_first_of(L"\\/") == std::wstring::npos && iniName.ends_with(L".ini")));
    }

    // 9. The status vocabulary the receipts and the overlay quote.
    VocabularyCases();

    if (!evidenceDir.empty())
        DiscoveryCases(evidenceDir);

    std::string json;
    json += "{\n  \"tool\": \"ampere_mfg_ini_smoke\",\n";
    json += "  \"timestamp_utc\": \"" + NowIso() + "\",\n";
    json += "  \"root\": \"" + Escape(root) + "\",\n";
    json += "  \"evidence\": \"" + Escape(evidenceDir.string()) + "\",\n";
    json += "  \"cases\": " + std::to_string(rows.size()) + ",\n";
    json += "  \"failed\": " + std::to_string(fails) + ",\n";
    json += std::string("  \"ok\": ") + (fails == 0 ? "true" : "false") + ",\n";
    json += "  \"rows\": [\n";

    std::string clampEntries;

    for (size_t index = 0; index < rows.size(); ++index)
    {
        const auto& row = rows[index];
        const std::string entry = "    { \"group\": \"" + Escape(row.group) + "\", \"name\": \"" + Escape(row.name) +
                                  "\", \"expected\": \"" + Escape(row.expected) + "\", \"observed\": \"" +
                                  Escape(row.observed) + "\", \"ok\": " + (row.ok ? "true" : "false") + " }";
        json += entry + (index + 1 < rows.size() ? "," : "") + "\n";

        if (row.group == "max-generated-frames" || row.group == "optimized" || row.group == "router" ||
            row.group == "kernel-image" || row.group == "mode" || row.group == "log-level")
            clampEntries += entry + ",\n";
    }

    // The clamp subset is a separate array; its last entry must not carry the comma of the full list.
    if (!clampEntries.empty())
        clampEntries.resize(clampEntries.size() - 2);

    json += "  ]\n}\n";

    if (!evidenceDir.empty())
    {
        std::error_code error;
        std::filesystem::create_directories(evidenceDir, error);

        WriteText(evidenceDir / "smoke-cases.json", json);
        WriteText(evidenceDir / "clamp-fixtures.json",
                  "{\n  \"tool\": \"ampere_mfg_ini_smoke\",\n  \"timestamp_utc\": \"" + NowIso() +
                      "\",\n  \"note\": \"clamp rows: input -> emitted value, against the v0.3.5 schema\",\n"
                      "  \"rows\": [\n" +
                      clampEntries + "\n  ]\n}\n");
        WriteText(evidenceDir / "schema-golden.ini", kSchemaGolden);
    }

    if (fails != 0)
    {
        std::printf("FAIL: ampere_mfg_ini_smoke (%d of %d checks)\n", fails, static_cast<int>(rows.size()));
        return 1;
    }

    std::printf("PASS: ampere_mfg_ini_smoke (%d checks: clamp low/high, auto, PTX, schema, discovery, vocabulary)\n",
                static_cast<int>(rows.size()));
    return 0;
}
