// Host check of the shipped [FrameGen] External / [DLSSG] AmpereMfg* config surface (todo 4 of
// .omo/plans/rtx2030-mfg-integration.md).
//
// The runner (tests/Run-AmpereMfgLoader.cmd --config) first runs tools/extract_mfg_config_seam.py, which lifts
// the production declarations and the production read/save statements out of Config.h/Config.cpp verbatim into
// the .inc files this file includes. Nothing below re-states a key or a rule: the field declarations, the
// read/save statements and the CustomOptional/readBool/readInt/readString/GetBoolValue/GetIntValue helpers are
// the shipped text, and the extractor fails when an anchor moves. The INI object is the real
// external/simpleini CSimpleIniA, so "produced INI" below means the bytes the production save would write.
//
//   cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /I <out> /I <repo>/external/simpleini
//      tests\ampere_mfg_config_smoke.cpp
//
// Cases:
//   a  round trip: non-default values (incl. External=true) are written, reloaded and read back unchanged
//   b  AmpereMfgMaxFrames is clamped into 1..5 on load (9 -> 5) and the saved value is the clamped one
//   c  the "value || ampereUnlock" rule: unlock on => the save yields External=true even when the file says
//      false; unlock off => the stored External value survives unchanged
//   d  a save after a load removes none of the four keys and no other key (produced INI compared byte for byte)
//   e  the shipped OptiScaler.ini carries the four keys with the documented defaults, and a save of it leaves
//      the Ada keys and every other key exactly as they were
//
// Exit codes: 0 = every case passed, 1 = at least one row failed, 3 = the --trap-legacy-strip fixture failed as
// designed (the strip assertion sees the v0.8.7 removal list), 2 = that fixture went undetected (blind check).
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include <SimpleIni.h>

// The real HasDefaultValue + CustomOptional (Config.h), included outside the stub struct.
#include "production-config-customoptional.inc"

struct Config
{
    std::vector<std::string> _log;

    // The real FGEnabled declaration and the real [FrameGen] External / [DLSSG] AmpereMfg* declarations.
#include "production-config-fgenabled.inc"
#include "production-ampere-fields.inc"

    std::optional<std::string> readString(std::string section, std::string key, bool lowercase = false);
    std::optional<int> readInt(std::string section, std::string key);
    std::optional<bool> readBool(std::string section, std::string key);

    void LoadFrameGen();
    void SaveFrameGen();
};

static Config* Instance();
static CSimpleIniA ini;

#include "production-config-values.inc"
#include "production-config-read.inc"

static Config g_config;
static Config* Instance()
{
    return &g_config;
}

// These two bodies are the production statements, verbatim.
void Config::LoadFrameGen()
{
#include "production-ampere-read.inc"
}

void Config::SaveFrameGen()
{
#include "production-ampere-save.inc"
}

namespace
{
namespace fs = std::filesystem;

struct Row
{
    std::string caseId;
    std::string name;
    std::string expected;
    std::string observed;
    bool ok;
};

struct ProducedRow
{
    std::string fixture;
    std::string produced;
    bool normalizedIdentical;
    std::string removedLines;
    std::string addedLines;
    std::string values;
};

std::vector<Row> g_rows;
std::vector<ProducedRow> g_produced;
fs::path g_evidence;
int g_fails = 0;

void Check(const std::string& caseId, const std::string& name, const std::string& expected, const std::string& observed)
{
    const bool ok = expected == observed;
    g_rows.push_back({caseId, name, expected, observed, ok});

    if (ok)
        std::printf("pass %s %-44s = %s\n", caseId.c_str(), name.c_str(), observed.c_str());
    else
    {
        ++g_fails;
        std::printf("FAIL %s %-44s expected [%s] observed [%s]\n", caseId.c_str(), name.c_str(), expected.c_str(),
                    observed.c_str());
    }
}

std::string BoolText(bool value)
{
    return value ? "true" : "false";
}

std::string JoinValues(const std::vector<std::string>& values)
{
    std::string joined;

    for (const auto& value : values)
    {
        if (!joined.empty())
            joined += "|";

        joined += value;
    }

    return joined;
}

std::string FormatSection(const char* section, const std::vector<std::pair<const char*, std::string>>& keys)
{
    std::vector<std::string> parts;

    for (const auto& [key, value] : keys)
        parts.push_back(std::format("{}={}", key, value));

    return std::format("{}:{}", section, JoinValues(parts));
}

fs::path IniDir()
{
    return g_evidence / "ini";
}

void EnsureDirs()
{
    std::error_code error;
    fs::create_directories(IniDir(), error);

    if (error)
    {
        std::printf("cannot create %s: %s\n", IniDir().string().c_str(), error.message().c_str());
        std::exit(1);
    }
}

std::string ReadText(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary);

    if (!stream)
        return {};

    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::vector<std::string> NormalizedLines(const std::string& text)
{
    std::vector<std::string> lines;
    std::string current;

    for (const char c : text)
    {
        if (c == '\r')
            continue;

        if (c == '\n')
        {
            lines.push_back(current);
            current.clear();
            continue;
        }

        current += c;
    }

    if (!current.empty())
        lines.push_back(current);

    return lines;
}

// Writes a fixture through SimpleIni's own writer, so the shipped save path writes the same bytes back when it
// preserves every value.
fs::path WriteFixture(const std::string& name,
                      const std::vector<std::tuple<const char*, const char*, const char*>>& settings)
{
    CSimpleIniA writer;

    for (const auto& [section, key, value] : settings)
        writer.SetValue(section, key, value);

    const fs::path path = IniDir() / (name + "-fixture.ini");
    writer.SaveFile(path.c_str());
    return path;
}

// Runs the production read statements against a fixture, exactly as Config::Reload does.
bool LoadFixture(const fs::path& path)
{
    g_config = Config {};
    ini.Reset();

    if (ini.LoadFile(path.c_str()) != SI_OK)
    {
        std::printf("fixture unreadable: %s\n", path.string().c_str());
        std::exit(1);
    }

    g_config.LoadFrameGen();
    return true;
}

// Runs the production save statements against the loaded INI and writes the produced bytes out. `alreadySaved`
// skips the statements when the caller ran them itself (the legacy-strip fixture writes in between).
fs::path SaveProduced(const std::string& name, bool alreadySaved = false)
{
    const fs::path path = IniDir() / (name + "-produced.ini");

    if (!alreadySaved)
        g_config.SaveFrameGen();

    ini.SaveFile(path.c_str());
    return path;
}

std::string ValueIn(const fs::path& path, const char* section, const char* key)
{
    CSimpleIniA reader;

    if (reader.LoadFile(path.c_str()) != SI_OK)
        return "<unloadable>";

    return std::string(reader.GetValue(section, key, "<missing>"));
}

std::set<std::string> KeySet(const fs::path& path)
{
    std::set<std::string> keys;
    CSimpleIniA reader;

    if (reader.LoadFile(path.c_str()) != SI_OK)
        return keys;

    CSimpleIniA::TNamesDepend sections;
    reader.GetAllSections(sections);

    for (const auto& section : sections)
    {
        CSimpleIniA::TNamesDepend sectionKeys;
        reader.GetAllKeys(section.pItem, sectionKeys);

        for (const auto& key : sectionKeys)
            keys.insert(std::format("{}/{}", section.pItem, key.pItem));
    }

    return keys;
}

std::string DescribeKeys(const std::set<std::string>& keys)
{
    std::string text;

    for (const auto& key : keys)
    {
        if (!text.empty())
            text += ",";

        text += key;
    }

    return text;
}

// Every section/key=value pair, so two files can be compared without their formatting getting in the way.
std::set<std::string> ValueMap(const fs::path& path)
{
    std::set<std::string> values;
    CSimpleIniA reader;

    if (reader.LoadFile(path.c_str()) != SI_OK)
        return values;

    CSimpleIniA::TNamesDepend sections;
    reader.GetAllSections(sections);

    for (const auto& section : sections)
    {
        CSimpleIniA::TNamesDepend sectionKeys;
        reader.GetAllKeys(section.pItem, sectionKeys);

        for (const auto& key : sectionKeys)
        {
            CSimpleIniA::TNamesDepend sectionValues;
            reader.GetAllValues(section.pItem, key.pItem, sectionValues);

            for (const auto& value : sectionValues)
                values.insert(std::format("{}/{}={}", section.pItem, key.pItem, value.pItem));
        }
    }

    return values;
}

// Records the produced INI against its fixture: normalized-identical bytes, the lines the save added or removed,
// and the four key values as they came back off disk.
ProducedRow RecordProduced(const std::string& name, const fs::path& fixture, const fs::path& produced)
{
    const auto fixtureLines = NormalizedLines(ReadText(fixture));
    const auto producedLines = NormalizedLines(ReadText(produced));
    const auto fixtureKeys = KeySet(fixture);
    const auto producedKeys = KeySet(produced);

    std::vector<std::string> removed;
    std::vector<std::string> added;
    auto fixtureSorted = fixtureLines;
    auto producedSorted = producedLines;
    std::ranges::sort(fixtureSorted);
    std::ranges::sort(producedSorted);
    std::ranges::set_difference(fixtureSorted, producedSorted, std::back_inserter(removed));
    std::ranges::set_difference(producedSorted, fixtureSorted, std::back_inserter(added));

    std::vector<std::string> values { ValueIn(produced, "FrameGen", "External"),
                                      ValueIn(produced, "DLSSG", "AmpereMfgUnlock"),
                                      ValueIn(produced, "DLSSG", "AmpereMfgMaxFrames"),
                                      ValueIn(produced, "DLSSG", "AmpereMfgKernelImage") };

    ProducedRow row { fixture.string(),
                      produced.string(),
                      fixtureLines == producedLines && fixtureKeys == producedKeys,
                      JoinValues(removed),
                      JoinValues(added),
                      JoinValues(values) };

    g_produced.push_back(row);
    std::printf("      produced %s: %s\n", name.c_str(), row.normalizedIdentical ? "fixture bytes preserved" : "differs");
    return row;
}

// What the four keys hold in the loaded configuration right now.
std::string LoadedValues()
{
    return JoinValues({ BoolText(Instance()->ExternalFrameGeneration.value_or_default()),
                        BoolText(Instance()->FGDLSSGAmpereMfgUnlock.value_or_default()),
                        std::to_string(Instance()->FGDLSSGAmpereMfgMaxFrames.value_or_default()),
                        Instance()->FGDLSSGAmpereMfgKernelImage.value_for_config_or("auto") });
}

std::string ProducedValues(const fs::path& produced)
{
    return JoinValues({ ValueIn(produced, "FrameGen", "External"), ValueIn(produced, "DLSSG", "AmpereMfgUnlock"),
                        ValueIn(produced, "DLSSG", "AmpereMfgMaxFrames"),
                        ValueIn(produced, "DLSSG", "AmpereMfgKernelImage") });
}

// ---------------------------------------------------------------------------
// a) round trip
// ---------------------------------------------------------------------------
bool CaseRoundTrip()
{
    const std::string id = "a";
    const int before = g_fails;

    const auto fixture = WriteFixture("a",
                                      { { "FrameGen", "Enabled", "auto" },
                                        { "FrameGen", "External", "true" },
                                        { "FrameGen", "FGInput", "dlssg" },
                                        { "DLSSG", "AdaMfgUnlock", "false" },
                                        { "DLSSG", "AmpereMfgUnlock", "true" },
                                        { "DLSSG", "AmpereMfgMaxFrames", "4" },
                                        { "DLSSG", "AmpereMfgKernelImage", "PTX" } });

    LoadFixture(fixture);
    Check(id, "loaded values (External|unlock|frames|kernel)", "true|true|4|PTX", LoadedValues());

    const auto produced = SaveProduced("a");
    RecordProduced("a", fixture, produced);
    Check(id, "produced values (External|unlock|frames|kernel)", "true|true|4|PTX",
          ProducedValues(produced));

    // The produced file must read back into the same configuration.
    LoadFixture(produced);
    Check(id, "reloaded values", "true|true|4|PTX", LoadedValues());
    return g_fails == before;
}

// ---------------------------------------------------------------------------
// b) AmpereMfgMaxFrames clamp into 1..5
// ---------------------------------------------------------------------------
bool CaseClamp()
{
    const std::string id = "b";
    const int before = g_fails;

    const std::pair<const char*, std::string> rows[] = { { "9", "5" }, { "6", "5" }, { "5", "5" }, { "4", "4" },
                                                         { "1", "1" }, { "0", "1" }, { "-3", "1" } };

    for (const auto& [input, expected] : rows)
    {
        const auto fixture = WriteFixture(std::format("b-{}", input), { { "FrameGen", "Enabled", "auto" },
                                                                        { "DLSSG", "AmpereMfgMaxFrames", input } });

        LoadFixture(fixture);
        Check(id, std::format("MaxFrames={} loads as", input), expected,
              std::to_string(Instance()->FGDLSSGAmpereMfgMaxFrames.value_or_default()));

        const auto produced = SaveProduced(std::format("b-{}", input));
        Check(id, std::format("MaxFrames={} saves as", input), expected, ValueIn(produced, "DLSSG", "AmpereMfgMaxFrames"));
    }

    // A value the parser cannot read keeps the documented default (3), which the save writes as auto - the same
    // convention every other optional key in this INI uses.
    const auto textFixture = WriteFixture("b-text", { { "FrameGen", "Enabled", "auto" },
                                                      { "DLSSG", "AmpereMfgMaxFrames", "abc" } });
    LoadFixture(textFixture);
    Check(id, "MaxFrames=abc loads as", "3", std::to_string(Instance()->FGDLSSGAmpereMfgMaxFrames.value_or_default()));

    const auto textProduced = SaveProduced("b-text");
    Check(id, "MaxFrames=abc saves as", "auto", ValueIn(textProduced, "DLSSG", "AmpereMfgMaxFrames"));
    return g_fails == before;
}

// ---------------------------------------------------------------------------
// c) the "value || ampereUnlock" rule
// ---------------------------------------------------------------------------
bool CaseOwnership()
{
    const std::string id = "c";
    const int before = g_fails;

    struct Row
    {
        const char* name;
        const char* unlock;
        const char* external; // nullptr = the key is not in the fixture
        const char* expected;
    };

    const Row rows[] = { { "unlock on, External absent", "true", nullptr, "true" },
                         { "unlock on, External=false in the file", "true", "false", "true" },
                         { "unlock on, External=true in the file", "true", "true", "true" },
                         { "unlock off, External=true in the file", "false", "true", "true" },
                         { "unlock off, External=false in the file", "false", "false", "false" },
                         { "unlock off, External absent", "false", nullptr, "false" } };

    for (const auto& row : rows)
    {
        std::vector<std::tuple<const char*, const char*, const char*>> settings { { "FrameGen", "Enabled", "auto" },
                                                                                  { "DLSSG", "AmpereMfgUnlock", row.unlock } };

        if (row.external != nullptr)
            settings.push_back({ "FrameGen", "External", row.external });

        const std::string name = std::format("c-{}{}", row.unlock, row.external == nullptr ? "-absent" : row.external);
        const auto fixture = WriteFixture(name, settings);

        LoadFixture(fixture);
        const auto produced = SaveProduced(name);
        RecordProduced(name, fixture, produced);

        Check(id, row.name, row.expected, ValueIn(produced, "FrameGen", "External"));
    }
    return g_fails == before;
}

// ---------------------------------------------------------------------------
// d) a save after a load removes none of the four keys
// ---------------------------------------------------------------------------
bool CaseStrip(const std::string& id, bool applyLegacyRemoval)
{
    const int before = g_fails;

    const auto fixture = WriteFixture(id, { { "FrameGen", "Enabled", "auto" },
                                            { "FrameGen", "External", "true" },
                                            { "DLSSG", "AdaMfgUnlock", "false" },
                                            { "DLSSG", "AdaTemporalFix", "auto" },
                                            { "DLSSG", "AdaFlipMeteringPatch", "false" },
                                            { "DLSSG", "AmpereMfgUnlock", "true" },
                                            { "DLSSG", "AmpereMfgMaxFrames", "4" },
                                            { "DLSSG", "AmpereMfgKernelImage", "PTX" },
                                            { "DLSSG", "UseGamesReflexMarkers", "auto" } });

    LoadFixture(fixture);
    g_config.SaveFrameGen();

    if (applyLegacyRemoval)
    {
        // The net effect of the v0.8.7 removal list applied by hand: it ran AFTER the [FrameGen] External
        // write, so all retained keys ended up missing from the saved file. The assertions below must catch it.
        ini.Delete("FrameGen", "External");

        for (const auto* key : { "AmpereMfgUnlock", "AmpereMfgMaxFrames", "AmpereMfgKernelImage" })
            ini.Delete("DLSSG", key);
    }

    const auto produced = SaveProduced(id, true);
    const auto row = RecordProduced(id, fixture, produced);

    Check(id, "produced keeps all four keys", "true|true|4|PTX", ProducedValues(produced));

    Check(id, "no key lost or gained", DescribeKeys(KeySet(fixture)), DescribeKeys(KeySet(produced)));
    Check(id, "no line added or removed", "none",
          row.removedLines.empty() && row.addedLines.empty()
              ? "none"
              : std::format("removed[{}] added[{}]", row.removedLines, row.addedLines));
    Check(id, "produced INI bytes equal the fixture", "identical", row.normalizedIdentical ? "identical" : "differs");
    return g_fails == before;
}

// ---------------------------------------------------------------------------
// e) the shipped OptiScaler.ini
// ---------------------------------------------------------------------------
bool CaseShippedIni(const fs::path& shipped)
{
    const std::string id = "e";
    const int before = g_fails;
    const std::string missing = "<missing>";

    Check(id, "[FrameGen] External", "false", ValueIn(shipped, "FrameGen", "External"));
    Check(id, "[DLSSG] AmpereMfgUnlock", "false", ValueIn(shipped, "DLSSG", "AmpereMfgUnlock"));
    Check(id, "[DLSSG] AmpereMfgMaxFrames", "3", ValueIn(shipped, "DLSSG", "AmpereMfgMaxFrames"));
    Check(id, "[DLSSG] AmpereMfgKernelImage", "auto", ValueIn(shipped, "DLSSG", "AmpereMfgKernelImage"));
    Check(id, "no key reads back as missing", "true",
          BoolText(ValueIn(shipped, "FrameGen", "External") != missing &&
                   ValueIn(shipped, "DLSSG", "AmpereMfgUnlock") != missing &&
                   ValueIn(shipped, "DLSSG", "AmpereMfgMaxFrames") != missing &&
                   ValueIn(shipped, "DLSSG", "AmpereMfgKernelImage") != missing));

    // The shipped defaults must load into the documented configuration.
    LoadFixture(shipped);
    Check(id, "shipped defaults load as (External|unlock|frames|kernel)", "false|false|3|auto",
          LoadedValues());

    // Saving the shipped INI must leave every Ada key and the key set untouched.
    const auto produced = SaveProduced("e-shipped");
    RecordProduced("e-shipped", shipped, produced);

    Check(id, "Ada keys unchanged",
          FormatSection("DLSSG", { { "AdaMfgUnlock", ValueIn(shipped, "DLSSG", "AdaMfgUnlock") },
                                   { "AdaTemporalFix", ValueIn(shipped, "DLSSG", "AdaTemporalFix") },
                                   { "AdaFlipMeteringPatch", ValueIn(shipped, "DLSSG", "AdaFlipMeteringPatch") } }),
          FormatSection("DLSSG", { { "AdaMfgUnlock", ValueIn(produced, "DLSSG", "AdaMfgUnlock") },
                                   { "AdaTemporalFix", ValueIn(produced, "DLSSG", "AdaTemporalFix") },
                                   { "AdaFlipMeteringPatch", ValueIn(produced, "DLSSG", "AdaFlipMeteringPatch") } }));

    Check(id, "no key lost or gained", DescribeKeys(KeySet(shipped)), DescribeKeys(KeySet(produced)));

    // The shipped file is saved through SimpleIni's writer, which spaces every pair out as "key = value";
    // compare pairs, not formatting. Everything OUTSIDE the config surface this todo touches must keep its
    // value byte for byte - the Ada keys above all.
    const std::set<std::string> configSurface { "FrameGen/Enabled",
                                                "FrameGen/External",
                                                "DLSSG/AmpereMfgUnlock",
                                                "DLSSG/AmpereMfgMaxFrames",
                                                "DLSSG/AmpereMfgKernelImage" };

    auto OutsideSurface = [&configSurface](std::set<std::string> pairs)
    {
        for (auto it = pairs.begin(); it != pairs.end();)
        {
            const auto separator = it->find('=');
            it = configSurface.contains(it->substr(0, separator)) ? pairs.erase(it) : std::next(it);
        }

        return pairs;
    };

    Check(id, "every key outside this config surface keeps its value",
          DescribeKeys(OutsideSurface(ValueMap(shipped))), DescribeKeys(OutsideSurface(ValueMap(produced))));

    // AmpereMfgMaxFrames=3 is the one documented normalisation on our own keys: the shipped default is written
    // back as auto, exactly like every other optional key of this INI, and reads back as 3 (row below).
    Check(id, "MaxFrames default normalises to auto", "auto", ValueIn(produced, "DLSSG", "AmpereMfgMaxFrames"));

    // The normalising save (the same convention as every other optional key) must still read back as 3.
    LoadFixture(produced);
    Check(id, "saved values reload as (External|unlock|frames|kernel)",
          "false|false|3|auto", LoadedValues());
    return g_fails == before;
}

// ---------------------------------------------------------------------------
// Receipts
// ---------------------------------------------------------------------------
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

void WriteJson(const fs::path& path, const std::vector<std::string>& body)
{
    std::string text;

    for (const auto& line : body)
        text += line + "\n";

    std::ofstream stream(path, std::ios::binary);
    stream << text;
    std::printf("receipt %s\n", path.string().c_str());
}

void WriteReceipts(int exitCode)
{
    std::vector<std::string> cases { "{" };
    cases.push_back(std::format(" \"tool\": \"tests/ampere_mfg_config_smoke.cpp\","));
    cases.push_back(std::format(" \"schema\": 1,"));
    cases.push_back(std::format(" \"evidence\": \"{}\",", JsonEscape(g_evidence.string())));
    cases.push_back(std::format(" \"failures\": {},", g_fails));
    cases.push_back(std::format(" \"exitCode\": {},", exitCode));
    cases.push_back(" \"rows\": [");

    for (size_t i = 0; i < g_rows.size(); ++i)
    {
        const auto& row = g_rows[i];
        const char* comma = i + 1 == g_rows.size() ? "" : ",";
        cases.push_back(std::format("  {{\"case\": \"{}\", \"name\": \"{}\", \"expected\": \"{}\", \"observed\": "
                                    "\"{}\", \"ok\": {}}}{}",
                                    JsonEscape(row.caseId), JsonEscape(row.name), JsonEscape(row.expected),
                                    JsonEscape(row.observed), row.ok ? "true" : "false", comma));
    }

    cases.push_back(" ]");
    cases.push_back("}");
    WriteJson(g_evidence / "config-cases.json", cases);

    std::vector<std::string> produced { "{" };
    produced.push_back(std::format(" \"schema\": 1,"));
    produced.push_back(" \"produced\": [");

    for (size_t i = 0; i < g_produced.size(); ++i)
    {
        const auto& row = g_produced[i];
        const char* comma = i + 1 == g_produced.size() ? "" : ",";
        produced.push_back(std::format("  {{\"fixture\": \"{}\", \"produced\": \"{}\", \"normalizedIdentical\": {}, "
                                       "\"removedLines\": \"{}\", \"addedLines\": \"{}\", \"values\": \"{}\"}}{}",
                                       JsonEscape(row.fixture), JsonEscape(row.produced),
                                       row.normalizedIdentical ? "true" : "false", JsonEscape(row.removedLines),
                                       JsonEscape(row.addedLines), JsonEscape(row.values), comma));
    }

    produced.push_back(" ]");
    produced.push_back("}");
    WriteJson(g_evidence / "config-produced.json", produced);
}
} // namespace

int main(int argc, char** argv)
{
    fs::path root = ".";
    bool trap = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];

        if (argument == "--evidence" && i + 1 < argc)
            g_evidence = argv[++i];
        else if (argument == "--root" && i + 1 < argc)
            root = argv[++i];
        else if (argument == "--trap-legacy-strip")
            trap = true;
        else
        {
            std::printf("unknown argument: %s\n", argument.c_str());
            return 1;
        }
    }

    if (g_evidence.empty())
    {
        std::printf("--evidence <dir> is required\n");
        return 1;
    }

    EnsureDirs();

    if (trap)
    {
        const auto trapRows = g_rows.size();
        const bool stripDetected = !CaseStrip("d-trap", true);
        g_rows.resize(trapRows);
        std::printf("trap: legacy strip %s\n", stripDetected ? "detected as designed" : "went undetected");
        return stripDetected ? 3 : 2;
    }

    std::printf("=== config surface: %s\n", (root / "OptiScaler.ini").string().c_str());
    CaseRoundTrip();
    CaseClamp();
    CaseOwnership();
    CaseStrip("d", false);
    CaseShippedIni(root / "OptiScaler.ini");

    const int failures = g_fails;
    WriteReceipts(failures == 0 ? 0 : 1);
    std::printf("config-cases=%d failures=%d\n", (int) g_rows.size(), failures);
    return failures == 0 ? 0 : 1;
}
