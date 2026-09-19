#include "pch.h"

#include "DlssNr.h"
#include "DlssNrFeature_Vk.h"
#include "DlssNr_ExposureScan.h"
#include "DlssNr_MenuSections.h"
#include <Config.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace DlssNr::MenuSections
{

struct ExposureTrimAnchorUi
{
    float key = 0.0f;
    float trim = 1.0f;
};

static std::vector<ExposureTrimAnchorUi> ParseExposureTrimAnchorsUi(const std::string& text)
{
    std::vector<ExposureTrimAnchorUi> out;
    size_t pos = 0;
    while (pos < text.size() && out.size() < 8)
    {
        const size_t semi = text.find(';', pos);
        const std::string token = text.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
        pos = semi == std::string::npos ? text.size() : semi + 1;
        const size_t colon = token.find(':');
        if (colon == std::string::npos)
            continue;
        try
        {
            const float key = std::stof(token.substr(0, colon));
            const float trim = std::stof(token.substr(colon + 1));
            if (std::isfinite(key) && key > 1e-8f && std::isfinite(trim) && trim > 0.0f)
                out.push_back({ key, std::clamp(trim, 0.25f, 50.0f) });
        }
        catch (...) {}
    }
    std::sort(out.begin(), out.end(),
              [](const ExposureTrimAnchorUi& a, const ExposureTrimAnchorUi& b) { return a.key < b.key; });
    return out;
}

static std::string SerializeExposureTrimAnchorsUi(const std::vector<ExposureTrimAnchorUi>& anchors)
{
    std::string out;
    char buf[64];
    for (const auto& p : anchors)
    {
        snprintf(buf, sizeof(buf), "%.7g:%.7g;", p.key, p.trim);
        out += buf;
    }
    return out;
}

static bool UpsertExposureTrimAnchorUi(std::vector<ExposureTrimAnchorUi>& anchors, float key, float trim)
{
    if (!(std::isfinite(key) && key > 1e-8f))
        return false;
    trim = std::clamp(trim, 0.25f, 50.0f);
    for (auto& p : anchors)
    {
        if (key > p.key * 0.98f && key < p.key * 1.02f)
        {
            p = { key, trim };
            std::sort(anchors.begin(), anchors.end(),
                      [](const ExposureTrimAnchorUi& a, const ExposureTrimAnchorUi& b) { return a.key < b.key; });
            return true;
        }
    }
    if (anchors.size() >= 8)
        return false;
    anchors.push_back({ key, trim });
    std::sort(anchors.begin(), anchors.end(),
              [](const ExposureTrimAnchorUi& a, const ExposureTrimAnchorUi& b) { return a.key < b.key; });
    return true;
}

static float ExposureTrimForUi(float key, float fallback,
                               const std::vector<ExposureTrimAnchorUi>& anchors, bool preview)
{
    fallback = std::clamp(fallback, 0.25f, 50.0f);
    if (preview || anchors.empty() || !(std::isfinite(key) && key > 1e-8f))
        return fallback;
    if (anchors.size() == 1)
        return anchors[0].trim;
    if (key <= anchors.front().key)
        return anchors.front().trim;
    if (key >= anchors.back().key)
        return anchors.back().trim;
    for (size_t i = 0; i + 1 < anchors.size(); ++i)
    {
        const auto& a = anchors[i];
        const auto& b = anchors[i + 1];
        if (key >= a.key && key <= b.key && b.key > a.key * 1.000001f)
        {
            const float t = (std::log(key) - std::log(a.key)) / (std::log(b.key) - std::log(a.key));
            return std::clamp(std::exp(std::log(a.trim) + t * (std::log(b.trim) - std::log(a.trim))),
                              0.25f, 50.0f);
        }
    }
    return anchors.back().trim;
}

static void RenderExposureTrimAnchorControls(CustomOptional<std::string>& storedAnchors,
                                             CustomOptional<bool>& previewSetting, float baseWhitePoint,
                                             float sliderTrim, const char* idSuffix)
{
    auto anchors = ParseExposureTrimAnchorsUi(storedAnchors.value_or_default());
    const bool haveKey = std::isfinite(baseWhitePoint) && baseWhitePoint > 1e-8f;
    const std::string addLabel = std::string("Add Anchor point##") + idSuffix;
    ImGui::BeginDisabled(!haveKey);
    if (ImGui::Button(addLabel.c_str()))
    {
        if (UpsertExposureTrimAnchorUi(anchors, baseWhitePoint, sliderTrim))
            storedAnchors = SerializeExposureTrimAnchorsUi(anchors);
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    bool preview = previewSetting.value_or_default();
    const std::string previewLabel = std::string("Preview Trim value for actual scene##") + idSuffix;
    if (ImGui::Checkbox(previewLabel.c_str(), &preview))
        previewSetting = preview;

    HelpMarker("When enabled, Anchor points are temporarily ignored and the Trim slider is applied"
               "\ndirectly. Tune the current scene, then press Add Anchor point."
               "\nThe calibration axis is Base White Point = PreExposure / Exposure."
               "\nThe preview switch is intentionally not saved across restarts.");

    if (!haveKey)
        ImGui::TextDisabled("Waiting for a valid base white point before an Anchor point can be added.");
    else
    {
        const float effective = ExposureTrimForUi(baseWhitePoint, sliderTrim, anchors, preview);
        ImGui::TextDisabled("Current base white point %.5f -> effective Trim %.2fx%s",
                            baseWhitePoint, effective, preview ? " (preview)" : "");
    }

    if (anchors.empty())
    {
        ImGui::TextDisabled("No Trim Anchor points: the Trim slider is used for every base white point.");
        return;
    }

    if (anchors.size() == 1)
        ImGui::TextDisabled("1 Trim Anchor point: its Trim is used for every base white point.");
    else
        ImGui::TextDisabled("%u Trim Anchor points: Trim is interpolated between base white-point values.",
                            (unsigned int) anchors.size());

    ImGui::PushID(idSuffix);
    for (size_t i = 0; i < anchors.size(); ++i)
    {
        ImGui::PushID((int) i);
        if (ImGui::SmallButton("x"))
        {
            anchors.erase(anchors.begin() + i);
            storedAnchors = SerializeExposureTrimAnchorsUi(anchors);
            ImGui::PopID();
            --i;
            continue;
        }
        ImGui::SameLine();
        ImGui::Text("Base white point %.5f -> Trim %.2fx", anchors[i].key, anchors[i].trim);
        ImGui::PopID();
    }
    ImGui::PopID();
}

void RenderInput(Config* config, float menuResScale)
{
    // Resolution changes rebuild model resources; commit only after releasing the slider.
    static int pendingScale = -1;

    int scalePercent =
        pendingScale >= 0 ? pendingScale : (int) lroundf(config->DlssNrWorkingScale.value_or_default() * 100.0f);

    if (ImGui::SliderInt("Model resolution", &scalePercent, 25, 200, "%d%%"))
        pendingScale = scalePercent;

    if (ImGui::IsItemDeactivatedAfterEdit() && pendingScale >= 0)
    {
        config->DlssNrWorkingScale = std::clamp(pendingScale, 25, 200) / 100.0f;
        pendingScale = -1;
    }

    HelpMarker("50% halves width and height. 100% uses the full input size.");

    if (scalePercent > 100)
    {
        static const char* dsNames[] = { "FSR1",     "Bicubic", "Catmull-Rom", "Lanczos2",
                                         "Lanczos3", "Kaiser2", "Kaiser3",     "MAGIC" };
        int ds = (int) config->DlssNrScalingDownscaler.value_or_default();
        if (ds < 0 || ds >= IM_ARRAYSIZE(dsNames))
            ds = (int) Scaler::Lanczos3;

        if (ImGui::Combo("Downscaler (NR)", &ds, dsNames, IM_ARRAYSIZE(dsNames)))
            config->DlssNrScalingDownscaler = (Scaler) ds;

        HelpMarker("Downsampling filter for resolutions above 100%.");
    }
    {
        const bool reduced = config->DlssNrWorkingScale.value_or_default() < 0.999f;

        if (!reduced)
            ImGui::BeginDisabled();

        static const char* enlargeNames[] = { "Classic", "Matched residual", "Matched residual + DLSS" };
        int enlarge = (int) std::min(config->DlssNrTransfer.value_or_default(), 2u);

        if (ImGui::Combo("Enlargement", &enlarge, enlargeNames, IM_ARRAYSIZE(enlargeNames)))
            config->DlssNrTransfer = (uint32_t) enlarge;

        if (!reduced)
            ImGui::EndDisabled();

        HelpMarker("Below 100%: enlarge the output or the NR difference. DLSS requires post-upscale DX12 processing.");
    }
    static const char* reversibleNames[] = { "Off (soft knee)", "Neutwo proxy + composed", "Neutwo proxy + replace",
                                             "Hybrid proxy + composed", "Hybrid proxy + replace" };
    int reversible = (int) config->DlssNrReversibleMode.value_or_default();
    if (reversible < 0 || reversible > 4)
        reversible = 0;
    if (ImGui::Combo("HDR mapping (experimental)", &reversible, reversibleNames, IM_ARRAYSIZE(reversibleNames)))
        config->DlssNrReversibleMode = (uint32_t) reversible;

    HelpMarker("HDR mapping curve. Replace bypasses strength and highlight controls.");

    {
        {
            const auto ex = DlssNr::GameExposureStatus();
            const bool vk = DlssNr::IsRunningVk();
            const auto autoEx = vk ? DlssNr::AutoExposureStatusVk() : DlssNr::AutoExposureStatus();
            const bool haveExposure = vk ? DlssNr::ExposureOfferedVk() : ex.everOffered;

            const float anchorNow = DlssNr::ExposureScan::BestValue();
            const bool haveAnchor = !DlssNr::ExposureScan::Anchors().empty();

            static const char* sourceNames[] = { "Manual paper white", "Game exposure",
                                                 "Scanned exposure (experimental)",
                                                 "Automatic exposure from HDR frame" };

            int source = (int) config->DlssNrWhitePointSource.value_or_default();

            if (source < 0 || source > 3)
                source = 0;

            if (ImGui::Combo("White point source", &source, sourceNames, IM_ARRAYSIZE(sourceNames)))
            {
                config->DlssNrWhitePointSource = (uint32_t) source;
            }

            HelpMarker("Game exposure uses supplied data. Scanned exposure requires calibration and may select the wrong buffer.");
            if (source == 1)
            {
                if (!vk && ex.seenFrames == 0)
                    ImGui::TextDisabled("Waiting for a frame...");
                else if (!haveExposure)
                    ImGui::TextDisabled("No game exposure available. Using manual paper white.");
                else if (vk)
                    ImGui::TextDisabled("Using game exposure.");
                else if (ex.exposure > 1e-6f)
                {
                    const float baseWhitePoint = ex.preExposure / ex.exposure;
                    const auto anchors =
                        ParseExposureTrimAnchorsUi(config->DlssNrGameExposureTrimAnchors.value_or_default());
                    const float trim = ExposureTrimForUi(
                        baseWhitePoint, config->DlssNrWhitePointTrim.value_or_default(), anchors,
                        config->DlssNrGameExposureTrimPreview.value_or_default());
                    ImGui::TextDisabled("Game exposure %.4f  ->  white point %.2f%s", ex.exposure,
                                        baseWhitePoint * trim,
                                        ex.offeredNow ? "" : "  (held: absent this frame)");
                }
                else
                    ImGui::TextDisabled("Reading exposure...");
            }
            else if (source == 2)
            {
                if (anchorNow <= 0.0f)
                {
                    const unsigned int watching = (unsigned int) DlssNr::ExposureScan::Report().size();

                    if (watching == 0)
                        ImGui::TextDisabled("No exposure candidates found.");
                    else
                        ImGui::TextDisabled("%u exposure candidates",
                                            watching);
                }
                else if (!haveAnchor)
                    ImGui::TextDisabled("Exposure candidate found.");
            }
            else if (source == 3)
            {
                if (autoEx.exposure > 1e-8f)
                {
                    const float baseWhitePoint = autoEx.preExposure / autoEx.exposure;
                    const auto anchors = ParseExposureTrimAnchorsUi(
                        config->DlssNrAutoExposureTrimAnchors.value_or_default());
                    const float trim = ExposureTrimForUi(
                        baseWhitePoint, config->DlssNrAutoExposureTrim.value_or_default(), anchors,
                        config->DlssNrAutoExposureTrimPreview.value_or_default());
                    ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f),
                                       "Automatic exposure %.4f  ->  white point %.2f", autoEx.exposure,
                                       baseWhitePoint * trim);
                    ImGui::TextDisabled("Exposure calculated automatically from a linear HDR frame");
                }
                else
                    ImGui::TextDisabled("Calculating automatic exposure...");
            }
            else if (haveExposure)
            {
                ImGui::TextDisabled("Game exposure is available.");
            }
        }
        const int wpSource = (int) config->DlssNrWhitePointSource.value_or_default();
        static int selectedAnchor = -1;
        auto anchors = DlssNr::ExposureScan::Anchors();
        if (selectedAnchor >= (int) anchors.size())
            selectedAnchor = -1;

        if (wpSource == 2)
        {
            const bool editingRow = selectedAnchor >= 0 && selectedAnchor < (int) anchors.size();
            if (!anchors.empty())
            {
                const float liveScan = DlssNr::ExposureScan::BestValue();

                if (liveScan > 0.0f)
                {
                    const float w = DlssNr::ExposureScan::AnchoredWhitePoint(
                        liveScan, config->DlssNrScanInverted.value_or_default(),
                        config->DlssNrScanTrim.value_or_default());

                    ImGui::TextDisabled("Scan %.5f  ->  white point %.2f   (%u point%s)", liveScan, w,
                                        (unsigned) anchors.size(), anchors.size() == 1 ? "" : "s");
                }
            }
            const bool showPaperWhite = anchors.empty() || editingRow;

            if (showPaperWhite)
            {
                float pw =
                    editingRow ? anchors[selectedAnchor].white : config->DlssNrWhitePointScale.value_or_default();

                char lbl[48];
                if (editingRow)
                    snprintf(lbl, sizeof(lbl), "Paper white (editing point %d)", selectedAnchor + 1);
                else
                    snprintf(lbl, sizeof(lbl), "Paper white");

                if (ImGui::SliderFloat(lbl, &pw, 0.25f, 2000.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
                {
                    if (editingRow)
                    {
                        DlssNr::ExposureScan::AnchorSetWhite(selectedAnchor, pw);
                        config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                    }
                    else
                        config->DlssNrWhitePointScale = pw;
                }

                HelpMarker("Adjust this point, then select Anchor here to save it.");
            }
            if (!anchors.empty())
            {
                float trim = config->DlssNrScanTrim.value_or_default();

                if (ImGui::SliderFloat("Trim (x the scan)", &trim, 0.25f, 4.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
                    config->DlssNrScanTrim = std::clamp(trim, 0.25f, 4.0f);

                ImGui::SameLine();

                if (ImGui::SmallButton("Reset##scantrim"))
                    config->DlssNrScanTrim = 1.0f;

                HelpMarker("Adjust calibrated brightness. Anchoring resets Trim to 1.");
            }
        }
        else if (wpSource == 1)
        {
            float gameTrim = config->DlssNrWhitePointTrim.value_or_default();
            if (ImGui::SliderFloat("Trim (x the game's exposure)", &gameTrim, 0.25f, 50.0f, "%.2fx",
                                   ImGuiSliderFlags_Logarithmic))
                config->DlssNrWhitePointTrim = std::clamp(gameTrim, 0.25f, 50.0f);

            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##wptrim"))
            {
                config->DlssNrWhitePointTrim = 1.0f;
                gameTrim = 1.0f;
            }

            HelpMarker("Exposure multiplier. 1 = unchanged.");

            const auto gameStatus = DlssNr::IsRunningVk() ? DlssNr::GameExposureStatusVk()
                                                           : DlssNr::GameExposureStatus();
            const float baseWhitePoint = gameStatus.exposure > 1e-8f
                ? gameStatus.preExposure / gameStatus.exposure : 0.0f;
            RenderExposureTrimAnchorControls(config->DlssNrGameExposureTrimAnchors,
                                             config->DlssNrGameExposureTrimPreview,
                                             baseWhitePoint, gameTrim, "gameExposureTrim");
        }
        else if (wpSource == 3)
        {
            float autoTrim = config->DlssNrAutoExposureTrim.value_or_default();
            if (ImGui::SliderFloat("Trim (x automatic exposure)", &autoTrim, 0.25f, 50.0f, "%.2fx",
                                   ImGuiSliderFlags_Logarithmic))
                config->DlssNrAutoExposureTrim = std::clamp(autoTrim, 0.25f, 50.0f);

            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##autoexposuretrim"))
            {
                config->DlssNrAutoExposureTrim = 5.0f;
                autoTrim = 5.0f;
            }

            HelpMarker("OptiScaler calculates exposure from the ORIGINAL linear-HDR frame before Neural Rendering."
                       "\nRange: 0.25x to 50.00x. Default: 5.00x"
                       "\nTry to use the highest value that subjectively looks best; excessive values"
                       "\nwill degrade image quality. Anchor points can use different Trim values for"
                       "\ndifferent Base White Point values."
                       "\nAutomatic exposure is available on D3D12 and Vulkan.");

            float protection = config->DlssNrAutoExposureShadowProtection.value_or_default();
            if (ImGui::SliderFloat("Shadow protection from bright highlights", &protection, 0.0f, 100.0f, "%.0f%%"))
                config->DlssNrAutoExposureShadowProtection = protection;

            HelpMarker("Controls how strongly very bright highlights are prevented from driving Automatic exposure."
                       "\n0% keeps the original full-frame arithmetic average."
                       "\n100% uses the strongest soft highlight compression. No tiles are discarded.");

            ImGui::TextDisabled("Metering: highlight-compressed arithmetic average.");

            const auto autoStatus = DlssNr::IsRunningVk() ? DlssNr::AutoExposureStatusVk()
                                                           : DlssNr::AutoExposureStatus();
            const float baseWhitePoint = autoStatus.exposure > 1e-8f
                ? autoStatus.preExposure / autoStatus.exposure : 0.0f;
            RenderExposureTrimAnchorControls(config->DlssNrAutoExposureTrimAnchors,
                                             config->DlssNrAutoExposureTrimPreview,
                                             baseWhitePoint, autoTrim, "automaticExposureTrim");
        }
        else
        {
            float wpScale = config->DlssNrWhitePointScale.value_or_default();

            if (ImGui::SliderFloat("Paper white", &wpScale, 0.25f, 2000.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
                config->DlssNrWhitePointScale = wpScale;

            HelpMarker("Higher values darken the NR input; lower values brighten it.");
        }
        {
            bool meter = config->DlssNrScanMeter.value_or_default();

            if (config->DlssNrWhitePointSource.value_or_default() == 2 &&
                ImGui::Checkbox("Show exposure meter", &meter))
                config->DlssNrScanMeter = meter;

            HelpMarker("Display the scanned value. Does not change the image.");
            if (DlssNr::ExposureScan::Scanning())
            {
                int which = 0;
                float low = 0.0f, high = 0.0f;
                const float live = DlssNr::ExposureScan::BestValue(&which, &low, &high);

                const bool isSource = config->DlssNrWhitePointSource.value_or_default() == 2;
                ImGui::BeginDisabled(live <= 0.0f || !isSource);

                if (ImGui::Button("Anchor here"))
                {
                    const float captureWhite =
                        anchors.empty() ? std::max(0.01f, config->DlssNrWhitePointScale.value_or_default())
                                        : std::max(0.01f, DlssNr::ExposureScan::AnchoredWhitePoint(
                                                              live, config->DlssNrScanInverted.value_or_default(),
                                                              config->DlssNrScanTrim.value_or_default()));

                    if (DlssNr::ExposureScan::AnchorAdd(live, captureWhite))
                    {
                        config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                        config->DlssNrScanTrim = 1.0f;
                        selectedAnchor = -1;
                    }
                }

                ImGui::EndDisabled();

                HelpMarker("Save exposure and white point. Up to 8 calibration points.");

                if (!isSource)
                    ImGui::TextDisabled("Scanned exposure is not the selected white point source.");

                if (!anchors.empty())
                {
                    int active = 0;
                    float bestDist = 1e30f;
                    const float liveLog = std::log(std::max(live, 1e-6f));

                    for (size_t i = 0; i < anchors.size(); ++i)
                    {
                        const float d = std::fabs(std::log(std::max(anchors[i].scan, 1e-6f)) - liveLog);
                        if (d < bestDist)
                        {
                            bestDist = d;
                            active = (int) i;
                        }
                    }

                    for (size_t i = 0; i < anchors.size(); ++i)
                    {
                        ImGui::PushID((int) i);
                        if (ImGui::SmallButton("x"))
                        {
                            DlssNr::ExposureScan::AnchorRemove((int) i);
                            config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                            if (selectedAnchor == (int) i)
                                selectedAnchor = -1;
                            else if (selectedAnchor > (int) i)
                                --selectedAnchor;
                            ImGui::PopID();
                            continue;
                        }

                        ImGui::SameLine();

                        const bool sel = (int) i == selectedAnchor;
                        char row[96];
                        snprintf(row, sizeof(row), "%s scan %.4f  ->  white %.2f%s",
                                 ((int) i == active && isSource) ? ">" : "  ", anchors[i].scan, anchors[i].white,
                                 sel ? "   [editing]" : "");
                        if (ImGui::Selectable(row, sel))
                            selectedAnchor = sel ? -1 : (int) i;

                        ImGui::PopID();
                    }

                    ImGui::TextDisabled("Select a row to edit it; select it again"
                                        " to deselect. > marks the active point.");
                }
                if (anchors.size() == 1)
                {
                    bool inverted = config->DlssNrScanInverted.value_or_default();
                    if (ImGui::Checkbox("Invert exposure tracking", &inverted))
                        config->DlssNrScanInverted = inverted;

                    HelpMarker("Reverse exposure response for single-point calibration.");
                }
                if (ImGui::TreeNode("Advanced"))
                {

                    const auto found = DlssNr::ExposureScan::Report();
                    const char* why = DlssNr::ExposureScan::Status();

                    if (found.empty())
                    {
                        ImGui::TextDisabled("%s",
                                            why != nullptr && why[0] != 0 ? why : "No exposure candidates found.");
                    }
                    else
                    {
                        for (size_t i = 0; i < found.size(); ++i)
                        {
                            const auto& c = found[i];

                            if (c.reads == 0)
                            {
                                ImGui::TextDisabled("%zu. %s -- not read yet", i + 1, c.shape.c_str());
                                continue;
                            }
                            ImGui::TextColored(
                                ImGui::GetStyleColorVec4(c.moves ? ImGuiCol_Text : ImGuiCol_TextDisabled),
                                "%zu. %s = %.5f  (seen %.5f..%.5f) %s", i + 1, c.shape.c_str(), c.latest, c.lowest,
                                c.highest, c.moves ? "MOVES" : "flat so far");
                        }

                        ImGui::TextDisabled("Unverified exposure candidate.");
                    }

                    ImGui::TreePop();
                }
            }
        }
    }
}

} // namespace DlssNr::MenuSections
