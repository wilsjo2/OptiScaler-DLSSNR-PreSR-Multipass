#include "pch.h"
#include "DlssNr_Presets.h"

#include <Config.h>
#include <Util.h>

#include <SimpleIni.h>

#include <algorithm>
#include <charconv>
#include <format>
#include <optional>
#include <system_error>
#include <type_traits>

namespace DlssNr::Presets
{
namespace
{
constexpr const char* kSection = "DlssNr";

// Every option a preset holds, under its OptiScaler.ini name. Capture and Apply both walk this one list.
template <class Visitor> void Visit(Config& c, Visitor&& v)
{
    // Where NR runs.
    v("RunBeforeSR", c.DlssNrRunBeforeSr);
    v("FinishedPicture", c.DlssNrFinishedPicture);
    v("HdrTransfer", c.DlssNrHdrTransfer);
    v("DeferredDLSS", c.DlssNrDeferredDlss);
    v("PrivateUpscaler", c.DlssNrPrivateUpscaler);
    v("ResidualAcrossRR", c.DlssNrResidualAcrossRr);
    v("ResidualAcrossRRBlend", c.DlssNrResidualAcrossRrBlend);
    v("DenoiseFirst", c.DlssNrDenoiseFirst);
    v("DenoiseFirstStep", c.DlssNrDenoiseFirstStep);
    v("DenoiseFirstEdit", c.DlssNrDenoiseFirstEdit);
    v("DenoiseFirstKernel", c.DlssNrDenoiseFirstKernel);
    v("DenoiseFirstShift", c.DlssNrDenoiseFirstShift);
    v("DenoiseFirstFlipJitter", c.DlssNrDenoiseFirstFlipJitter);
    v("DenoiseFirstNeighbourhoodClamp", c.DlssNrDenoiseFirstNeighbourhoodClamp);
    v("DenoiseFirstFireflyGuard", c.DlssNrDenoiseFirstFireflyGuard);
    // The model of pass 1, which later passes inherit.
    v("Preset", c.DlssNrPreset);
    v("Intensity", c.DlssNrIntensity);
    v("Style", c.DlssNrStyle);
    v("LocalStructure", c.DlssNrLocalStructure);
    v("LocalTone", c.DlssNrLocalTone);
    v("SkinStructure", c.DlssNrSkinStructure);
    v("AutoMask", c.DlssNrAutoMask);
    v("Passes", c.DlssNrPasses);
    v("UnlockPasses", c.DlssNrUnlockPasses);
    v("ReapplyGain", c.DlssNrReapplyGain);
    // Input, resolution and reconstruction.
    v("MatchGuides", c.DlssNrMatchGuides);
    v("RenderMotionScale", c.DlssNrRenderMotionScale);
    v("WorkingScale", c.DlssNrWorkingScale);
    v("ScalingDownscaler", c.DlssNrScalingDownscaler);
    v("Transfer", c.DlssNrTransfer);
    v("SpatialCompression", c.DlssNrSpatialCompression);
    v("SpatialCenterX", c.DlssNrSpatialCenterX);
    v("SpatialCenterY", c.DlssNrSpatialCenterY);
    v("SpatialWorkX", c.DlssNrSpatialWorkX);
    v("SpatialWorkY", c.DlssNrSpatialWorkY);
    v("SpatialOffsetX", c.DlssNrSpatialOffsetX);
    v("SpatialOffsetY", c.DlssNrSpatialOffsetY);
    v("SpatialShiftX", c.DlssNrSpatialShiftX);
    v("SpatialShiftY", c.DlssNrSpatialShiftY);
    // Composition.
    v("TransferStrength", c.DlssNrTransferStrength);
    v("ColourStrength", c.DlssNrColourStrength);
    v("MaxRatio", c.DlssNrMaxRatio);
    v("ReversibleMode", c.DlssNrReversibleMode);
    v("SkinProtection", c.DlssNrSkinProtection);
    v("SkinDetail", c.DlssNrSkinDetail);
    v("SkinColour", c.DlssNrSkinColour);
    v("EnvironmentDetail", c.DlssNrEnvironmentDetail);
    v("EnvironmentColour", c.DlssNrEnvironmentColour);
    v("ReplaceDetailStrength", c.DlssNrReplaceDetailStrength);
    v("ResidualConfidenceSensitivity", c.DlssNrResidualConfidenceSensitivity);
    v("HistoryRejection", c.DlssNrHistoryRejection);
    v("HaloSuppression", c.DlssNrHaloSuppression);
    v("HaloBrightSuppression", c.DlssNrHaloBrightSuppression);
    v("HaloRadius", c.DlssNrHaloRadius);
    // White point and exposure.
    v("WhitePointScale", c.DlssNrWhitePointScale);
    v("WhitePointSource", c.DlssNrWhitePointSource);
    v("WhitePointTrim", c.DlssNrWhitePointTrim);
    v("AutoExposureTrim", c.DlssNrAutoExposureTrim);
    v("AutoExposureHighlightProtection", c.DlssNrAutoExposureHighlightProtection);
    v("ExposureTrimAnchors", c.DlssNrExposureTrimAnchors);
    v("AutoExposureTrimAnchors", c.DlssNrAutoExposureTrimAnchors);
    // Passes 2..30: what each sets apart from pass 1.
    for (unsigned i = 0; i < std::size(c.DlssNrPassOverrides); ++i)
    {
        auto& pass = c.DlssNrPassOverrides[i];
        const auto key = [&](const char* field) { return std::format("Pass{}{}", i + 2, field); };
        if (i < 2)
            v(key("Preset"), pass.preset);
        v(key("Style"), pass.style);
        v(key("Intensity"), pass.intensity);
        v(key("LocalStructure"), pass.structure);
        v(key("LocalTone"), pass.tone);
        v(key("SkinStructure"), pass.skin);
        v(key("AutoMask"), pass.autoMask);
        v(key("WorkingScale"), pass.workingScale);
        v(key("ScalingDownscaler"), pass.scalingDownscaler);
        v(key("Transfer"), pass.transfer);
        v(key("PrivateUpscaler"), pass.privateUpscaler);
        v(key("SpatialCompression"), pass.spatialCompression);
        v(key("SpatialCenterX"), pass.spatialCenterX);
        v(key("SpatialCenterY"), pass.spatialCenterY);
        v(key("SpatialWorkX"), pass.spatialWorkX);
        v(key("SpatialWorkY"), pass.spatialWorkY);
        v(key("SpatialOffsetX"), pass.spatialOffsetX);
        v(key("SpatialOffsetY"), pass.spatialOffsetY);
        v(key("SpatialShiftX"), pass.spatialShiftX);
        v(key("SpatialShiftY"), pass.spatialShiftY);
        v(key("Reapply"), pass.reapply);
        v(key("BorrowTransformer"), pass.borrowTransformer);
        v(key("BorrowBelow"), pass.borrowBelow);
        v(key("BorrowLevel"), pass.borrowLevel);
    }
}

std::string ToText(bool value) { return value ? "true" : "false"; }
std::string ToText(int value) { return std::to_string(value); }
std::string ToText(uint32_t value) { return std::to_string(value); }
std::string ToText(float value) { return std::format("{}", value); } // shortest text that reads back the same
std::string ToText(const std::string& value) { return value; }
std::string ToText(Scaler value) { return std::to_string(static_cast<uint32_t>(value)); }

template <class Number> std::optional<Number> ParseNumber(const std::string& text)
{
    Number value {};
    const auto* end = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    return error == std::errc {} && stop == end ? std::optional<Number>(value) : std::nullopt;
}
void FromText(const std::string& text, std::optional<bool>& out)
{
    if (text == "true" || text == "1")
        out = true;
    else if (text == "false" || text == "0")
        out = false;
}
void FromText(const std::string& text, std::optional<int>& out) { out = ParseNumber<int>(text); }
void FromText(const std::string& text, std::optional<uint32_t>& out) { out = ParseNumber<uint32_t>(text); }
void FromText(const std::string& text, std::optional<float>& out)
{
    out = ParseNumber<float>(text);
    if (out && !std::isfinite(*out))
        out.reset();
}
void FromText(const std::string& text, std::optional<std::string>& out) { out = text; }
void FromText(const std::string& text, std::optional<Scaler>& out)
{
    if (const auto number = ParseNumber<uint32_t>(text); number && *number < static_cast<uint32_t>(Scaler::Count))
        out = static_cast<Scaler>(*number);
}

std::string Trimmed(std::string name)
{
    while (!name.empty() && (name.back() == ' ' || name.back() == '.'))
        name.pop_back();
    while (!name.empty() && name.front() == ' ')
        name.erase(name.begin());
    return name;
}

std::filesystem::path FromUtf8(const std::string& text)
{
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

std::string ToUtf8(const std::filesystem::path& path)
{
    const auto text = path.u8string();
    return std::string(text.begin(), text.end());
}

std::filesystem::path File(const std::string& name) { return Directory() / FromUtf8(Trimmed(name) + ".ini"); }
} // namespace

Values Capture(Config& config)
{
    Values values;
    Visit(config, [&](const std::string& key, auto& option)
          {
              if (option.has_value())
                  values[key] = ToText(option.value());
          });
    return values;
}

void Apply(Config& config, const Values& values)
{
    Visit(config, [&](const std::string& key, auto& option)
          {
              using Type = typename std::remove_reference_t<decltype(option)>::value_type;
              std::optional<Type> parsed;
              if (const auto found = values.find(key); found != values.end())
                  FromText(found->second, parsed);
              // An option that already has this value is left untouched.
              if (static_cast<const std::optional<Type>&>(option) != parsed)
                  option = parsed;
          });
}

std::filesystem::path Directory() { return Util::DllPath().parent_path() / "nr-presets"; }

std::vector<std::string> List()
{
    std::vector<std::string> names;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(Directory(), error))
    {
        if (!entry.is_regular_file(error))
            continue;
        auto extension = ToUtf8(entry.path().extension());
        std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (extension == ".ini")
            names.push_back(ToUtf8(entry.path().stem()));
    }
    std::ranges::sort(names, [](const std::string& a, const std::string& b) { return _stricmp(a.c_str(), b.c_str()) < 0; });
    return names;
}

std::string NameError(const std::string& raw)
{
    const auto name = Trimmed(raw);
    if (name.empty())
        return "Type a name for the preset.";
    if (name.size() > 64)
        return "A preset name can have 64 characters at most.";
    for (const unsigned char c : name)
        if (c < 32 || std::strchr("\\/:*?\"<>|", c))
            return "A preset name cannot contain \\ / : * ? \" < > |";
    for (const char* reserved : { "CON",  "PRN",  "AUX",  "NUL",  "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
                                  "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9" })
        if (!_stricmp(name.c_str(), reserved))
            return "Windows reserves that name; choose another.";
    return {};
}

std::string Cleaned(const std::string& name) { return Trimmed(name); }

bool Exists(const std::string& name)
{
    std::error_code error;
    return NameError(name).empty() && std::filesystem::is_regular_file(File(name), error);
}

std::string Save(const std::string& name, Config& config)
{
    if (const auto problem = NameError(name); !problem.empty())
        return problem;
    std::error_code error;
    std::filesystem::create_directories(Directory(), error);
    CSimpleIniA file;
    file.SetUnicode();
    for (const auto& [key, value] : Capture(config))
        file.SetValue(kSection, key.c_str(), value.c_str());
    // (A preset with every option unset still needs its section, so that the file is a preset.)
    file.SetValue(kSection, "PresetVersion", "1");
    if (file.SaveFile(File(name).c_str()) < 0)
        return "The preset could not be written to " + ToUtf8(File(name)) + ".";
    return {};
}

std::string Read(const std::string& name, Values& values)
{
    if (const auto problem = NameError(name); !problem.empty())
        return problem;
    if (!Exists(name))
        return "There is no preset named \"" + Trimmed(name) + "\".";
    CSimpleIniA file;
    file.SetUnicode();
    if (file.LoadFile(File(name).c_str()) < 0)
        return "The preset \"" + Trimmed(name) + "\" could not be read.";
    CSimpleIniA::TNamesDepend keys;
    file.GetAllKeys(kSection, keys);
    values.clear();
    for (const auto& key : keys)
        if (const char* value = file.GetValue(kSection, key.pItem, nullptr))
            values[key.pItem] = value;
    if (values.empty())
        return "The file of preset \"" + Trimmed(name) + "\" holds no settings.";
    return {};
}

std::string Load(const std::string& name, Config& config)
{
    Values values;
    if (const auto problem = Read(name, values); !problem.empty())
        return problem;
    Apply(config, values);
    return {};
}

std::string Rename(const std::string& from, const std::string& to)
{
    if (const auto problem = NameError(to); !problem.empty())
        return problem;
    if (!Exists(from))
        return "There is no preset named \"" + Trimmed(from) + "\".";
    // A change of letter case only is the same file to Windows: that rename is allowed.
    if (Exists(to) && _stricmp(Trimmed(from).c_str(), Trimmed(to).c_str()))
        return "A preset named \"" + Trimmed(to) + "\" exists already.";
    std::error_code error;
    std::filesystem::rename(File(from), File(to), error);
    return error ? "The preset could not be renamed: " + error.message() : std::string {};
}

std::string Delete(const std::string& name)
{
    if (const auto problem = NameError(name); !problem.empty())
        return problem;
    std::error_code error;
    if (!std::filesystem::remove(File(name), error) || error)
        return "The preset \"" + Trimmed(name) + "\" could not be deleted.";
    return {};
}
} // namespace DlssNr::Presets
