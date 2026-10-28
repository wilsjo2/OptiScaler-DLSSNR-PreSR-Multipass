#pragma once

#include <imgui/imgui.h>
#include <algorithm>
#include <string>
#include <vector>
#include <span>
#include <cmath>

namespace DlssNr::PipelineUi
{
enum class Section
{
    Placement,
    Input,
    Model,
    Blend
};
enum class Route
{
    Before,
    After,
    Deferred,
    Finished,
    FinishedBefore
};

struct View
{
    Route route = Route::After;
    bool enabled = true;
    bool rayReconstruction = false;
    bool applyModel = true;
    const char* privateUpscaler = "DLSS";
    unsigned int passes = 1;
    int scalePercent = 100;
};

inline const char* SectionName(Section section)
{
    static constexpr const char* names[] = { "Placement", "Input", "Model passes", "Apply NR edit" };
    return names[(int) section];
}

// Keep both top-level toggles on one row, wrapping their clickable labels in narrow overlays.
inline bool CheckboxWrapped(const char* label, bool* value, float width)
{
    ImGui::PushID(label);
    ImGui::BeginGroup();
    const float right = ImGui::GetCursorPosX() + width;
    bool changed = ImGui::Checkbox("##toggle", value);
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
    ImGui::PushTextWrapPos(right);
    ImGui::TextUnformatted(label);
    ImGui::PopTextWrapPos();
    if (ImGui::IsItemClicked())
    {
        *value = !*value;
        changed = true;
    }
    ImGui::EndGroup();
    ImGui::PopID();
    return changed;
}

inline void DrawTimingBar(double nrMs, double frameMs)
{
    if (!std::isfinite(nrMs) || nrMs < 0.0 || !std::isfinite(frameMs) || frameMs <= 0.0)
    {
        ImGui::TextDisabled("Waiting for frame timing.");
        return;
    }
    const double remaining = std::max(frameMs - nrMs, 0.0);
    const bool overlapping = nrMs > frameMs;
    ImGui::TextWrapped("NR %.2f ms  |  Rest of frame ~%.2f ms", nrMs, remaining);
    const auto text = ImGui::GetStyleColorVec4(ImGuiCol_Text);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(text.x * 0.20f, text.y * 0.55f, text.z * 0.25f, text.w));
    ImGui::ProgressBar(float(nrMs / std::max(frameMs, nrMs)), ImVec2(-1.0f, ImGui::GetFontSize()), "");
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("NR GPU time versus frame interval. Work can overlap.");
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Rendered frame: %.2f ms%s", frameMs,
                        overlapping ? " (NR overlaps/exceeds this interval)" : "");
    ImGui::PopTextWrapPos();
}

// This describes the configured colour/edit flow. It never changes a rendering option.
inline void Draw(const View& view, Section& selected)
{
    ImGui::PushID("NR pipeline chart");
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(ImGui::GetContentRegionAvail().x, 1.0f);
    const float gap = ImGui::GetFontSize() * 1.5f;
    const bool split = view.enabled && (view.route == Route::Deferred || view.route == Route::FinishedBefore);
    const bool finished = view.route == Route::Finished || view.route == Route::FinishedBefore;
    const float nodeWidth =
        split ? std::max((width - gap) * 0.5f, 1.0f) : std::min(width, ImGui::GetFontSize() * 27.0f);
    enum Stage
    {
        GameInput,
        Upscale,
        Effects,
        Prepare,
        Model,
        Apply,
        Enlarge,
        GameOutput
    };
    struct Label
    {
        const char* title;
        std::string detail;
        int section = -1;
    };
    const Label labels[] = {
        { "Game input", "Placement / routing", (int) Section::Placement },
        { view.rayReconstruction ? "RR + Super Resolution" : "Super Resolution",
          split ? "Clean game image" : "Game upscaler" },
        { "Game effects + HUD", "Game rendering" },
        { "Prepare NR input", "HDR / paper white / " + std::to_string(view.scalePercent) + "%", (int) Section::Input },
        { "NR model", std::to_string(view.passes) + (view.passes == 1 ? " pass" : " passes"), (int) Section::Model },
        { "Apply NR edit", view.applyModel ? "Strength / skin" : "Edit hidden; model runs", (int) Section::Blend },
        { "Upscale NR edit", std::string("Separate ") + view.privateUpscaler + " pass (no RR)" },
        { "Game output", "FG / presentation" }
    };
    // Parents encode the two branches directly; ordinary routes are simple ordered stages.
    struct Node
    {
        Stage stage;
        int lane, row, parent, otherParent = -1;
    };
    std::vector<Node> nodes;
    if (split)
    {
        nodes = { { GameInput, 0, 0, -1 },
                  { Prepare, -1, 1, 0 },
                  { Model, -1, 2, 1 },
                  { Enlarge, -1, 3, 2 },
                  { Upscale, 1, 1, 0 } };
        if (finished)
            nodes.push_back({ Effects, 1, 2, 4 });
        nodes.push_back({ Apply, 0, 4, 3, (int) nodes.size() - 1 });
        if (!finished)
            nodes.push_back({ Effects, 0, 5, (int) nodes.size() - 1 });
        nodes.push_back({ GameOutput, 0, finished ? 5 : 6, (int) nodes.size() - 1 });
    }
    else
    {
        static constexpr Stage off[] = { GameInput, Upscale, Effects, GameOutput };
        static constexpr Stage before[] = { GameInput, Prepare, Model, Apply, Upscale, Effects, GameOutput };
        static constexpr Stage after[] = { GameInput, Upscale, Prepare, Model, Apply, Effects, GameOutput };
        static constexpr Stage present[] = { GameInput, Upscale, Effects, Prepare, Model, Apply, GameOutput };
        std::span<const Stage> flow = off;
        if (view.enabled)
            flow = view.route == Route::Before ? std::span(before) : finished ? std::span(present) : std::span(after);
        for (auto stage : flow)
        {
            const int row = (int) nodes.size();
            nodes.push_back({ stage, 0, row, row - 1 });
        }
    }
    const int lastRow = nodes.back().row;

    // Wrap labels inside their nodes so a narrow overlay does not crop either branch.
    const float padding = ImGui::GetStyle().FramePadding.x + 4.0f;
    const float wrapWidth = std::max(nodeWidth - padding * 2.0f, 1.0f);
    float height = 0.0f;
    for (const auto& node : nodes)
    {
        const auto& label = labels[node.stage];
        height = std::max(height, ImGui::CalcTextSize(label.title, nullptr, false, wrapWidth).y +
                                      ImGui::CalcTextSize(label.detail.c_str(), nullptr, false, wrapWidth).y + 12.0f);
    }
    const float step = height + gap;
    const auto topLeft = [&](int index)
    {
        const auto& node = nodes[index];
        const float x = node.lane < 0 ? 0.0f : node.lane > 0 ? width - nodeWidth : (width - nodeWidth) * 0.5f;
        return ImVec2(origin.x + x, origin.y + node.row * step);
    };
    auto* draw = ImGui::GetWindowDrawList();
    const ImU32 lineColour = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    for (int i = 0; i < (int) nodes.size(); ++i)
        for (int parent : { nodes[i].parent, nodes[i].otherParent })
        {
            if (parent < 0)
                continue;
            const auto a = topLeft(parent), b = topLeft(i);
            const ImVec2 start(a.x + nodeWidth * 0.5f, a.y + height), end(b.x + nodeWidth * 0.5f, b.y);
            const float bend = end.y - gap * 0.5f;
            draw->AddLine(start, ImVec2(start.x, bend), lineColour, 1.5f);
            draw->AddLine(ImVec2(start.x, bend), ImVec2(end.x, bend), lineColour, 1.5f);
            draw->AddLine(ImVec2(end.x, bend), end, lineColour, 1.5f);
            draw->AddTriangleFilled(end, ImVec2(end.x - 3, end.y - 5), ImVec2(end.x + 3, end.y - 5), lineColour);
        }
    for (int i = 0; i < (int) nodes.size(); ++i)
    {
        const auto& label = labels[nodes[i].stage];
        const auto at = topLeft(i);
        ImGui::SetCursorScreenPos(at);
        ImGui::PushID(i);
        const bool editable = label.section >= 0;
        const bool chosen = editable && label.section == (int) selected;
        bool hovered = false;
        if (editable)
        {
            if (ImGui::InvisibleButton("stage", ImVec2(nodeWidth, height)))
                selected = (Section) label.section;
            hovered = ImGui::IsItemHovered();
        }
        else
            ImGui::Dummy(ImVec2(nodeWidth, height));
        const auto background = chosen     ? ImGuiCol_HeaderActive
                                : hovered  ? ImGuiCol_ButtonHovered
                                : editable ? ImGuiCol_Button
                                           : ImGuiCol_FrameBg;
        auto fill = ImGui::GetStyleColorVec4(background);
        if (label.section == (int) Section::Model)
        {
            // Preserve the theme's HDR brightness and interaction states, changing only the hue.
            const float brightness = std::max({ fill.x, fill.y, fill.z });
            fill = ImVec4(brightness * 0.20f, brightness * 0.65f, brightness * 0.30f, fill.w);
        }
        draw->AddRectFilled(at, ImVec2(at.x + nodeWidth, at.y + height), ImGui::GetColorU32(fill),
                            ImGui::GetStyle().FrameRounding);
        const auto titleSize = ImGui::CalcTextSize(label.title, nullptr, false, wrapWidth);
        const auto detailSize = ImGui::CalcTextSize(label.detail.c_str(), nullptr, false, wrapWidth);
        const float y = at.y + (height - titleSize.y - detailSize.y) * 0.5f;
        draw->AddText(nullptr, 0.0f, ImVec2(at.x + (nodeWidth - titleSize.x) * 0.5f, y),
                      ImGui::GetColorU32(ImGuiCol_Text), label.title, nullptr, wrapWidth);
        draw->AddText(nullptr, 0.0f, ImVec2(at.x + (nodeWidth - detailSize.x) * 0.5f, y + titleSize.y),
                      ImGui::GetColorU32(editable ? ImGuiCol_Text : ImGuiCol_TextDisabled), label.detail.c_str(),
                      nullptr, wrapWidth);
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + (lastRow + 1) * step));
    ImGui::Dummy(ImVec2(width, 0));
    // Inspection is a tool, not a colour path or a processing stage.
    const auto tool = [&](const char* label, Section section)
    {
        const bool chosen = selected == section;
        if (chosen)
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
        if (ImGui::Button(label))
            selected = section;
        if (chosen)
            ImGui::PopStyleColor();
    };
    if (!view.enabled)
    {
        tool("Prepare input", Section::Input);
        ImGui::SameLine();
        tool("Model passes", Section::Model);
        ImGui::SameLine();
        tool("Apply edit", Section::Blend);
    }
    ImGui::PopID();
}
} // namespace DlssNr::PipelineUi
