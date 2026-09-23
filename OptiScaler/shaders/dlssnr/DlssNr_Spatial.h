#pragma once

#include "DlssNr_Guides.h"
#include "../../../external/peripheral_warp/SpatialWarp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace DlssNr::Spatial
{

struct Settings
{
    bool enabled = false;
    float centerX = 80, centerY = 80;
    float workX = 90, workY = 90;
    float offsetX = 0, offsetY = 0;
    float shiftX = 0, shiftY = 0;
    bool operator==(const Settings& o) const
    {
        const auto same = [](float a, float b) { return a == b || (std::isnan(a) && std::isnan(b)); };
        return enabled == o.enabled && same(centerX, o.centerX) && same(centerY, o.centerY) && same(workX, o.workX) &&
               same(workY, o.workY) && same(offsetX, o.offsetX) && same(offsetY, o.offsetY) && same(shiftX, o.shiftX) &&
               same(shiftY, o.shiftY);
    }
    bool operator!=(const Settings& o) const { return !(*this == o); }
};

template <class ConfigLike> Settings ReadSettings(const ConfigLike& cfg)
{
    Settings s;
    s.enabled = cfg.DlssNrSpatialCompression.value_or_default();
    s.centerX = cfg.DlssNrSpatialCenterX.value_or_default();
    s.centerY = cfg.DlssNrSpatialCenterY.value_or_default();
    s.workX = cfg.DlssNrSpatialWorkX.value_or_default();
    s.workY = cfg.DlssNrSpatialWorkY.value_or_default();
    s.offsetX = cfg.DlssNrSpatialOffsetX.value_or_default();
    s.offsetY = cfg.DlssNrSpatialOffsetY.value_or_default();
    s.shiftX = cfg.DlssNrSpatialShiftX.value_or_default();
    s.shiftY = cfg.DlssNrSpatialShiftY.value_or_default();
    return s;
}

inline float MaxCenterOffset(float centerPercent) { return std::max(0.0f, (100.0f - centerPercent) * .5f - .5f); }

inline float MinimumWorkPercent(float globalScale)
{
    return std::isfinite(globalScale) && globalScale > 0 ? std::clamp(25.0f / globalScale, 25.0f, 100.0f) : 100.0f;
}

inline std::pair<float, float> WorkShiftLimits(float centerPercent, float workPercent, float offsetPercent)
{
    const float c = centerPercent * .01f, w = workPercent * .01f;
    const float halfBand = c * .5f, bandCenter = .5f + offsetPercent * .01f;
    const float periphery[2] = { std::max(0.0f, bandCenter - halfBand), std::max(0.0f, 1.0f - bandCenter - halfBand) };
    const float budget = std::max(0.0f, w - c);
    const int narrow = periphery[0] <= periphery[1] ? 0 : 1;
    float base[2] = {};
    base[narrow] = std::min(.5f * budget, periphery[narrow]);
    base[1 - narrow] = std::min(budget - base[narrow], periphery[1 - narrow]);
    return { -100.0f * std::max(0.0f, std::min(base[1], periphery[0] - base[0])),
             100.0f * std::max(0.0f, std::min(base[0], periphery[1] - base[1])) };
}

inline std::pair<float, float> WorkShiftLimits(const Settings& s, bool yAxis)
{
    return yAxis ? WorkShiftLimits(s.centerY, s.workY, s.offsetY) : WorkShiftLimits(s.centerX, s.workX, s.offsetX);
}

struct Rect
{
    float left = 0, top = 0, right = 1, bottom = 1;
};

struct Layout
{
    Settings settings {};
    bool requested = false, active = false;
    std::string reason;
    uint32_t nativeW = 0, nativeH = 0, ordinaryW = 0, ordinaryH = 0, modelW = 0, modelH = 0;
    float globalScale = 1;
    pw::Layout warp {};
    Rect centerBounds {}, workBounds {}; // Native-frame normalized extents, including offset and shift.
    bool operator==(const Layout& o) const
    {
        return settings == o.settings && nativeW == o.nativeW && nativeH == o.nativeH &&
               (globalScale == o.globalScale || (std::isnan(globalScale) && std::isnan(o.globalScale))) &&
               active == o.active;
    }
    bool operator!=(const Layout& o) const { return !(*this == o); }
};

inline uint32_t EvenExtent(uint32_t native, double fraction)
{
    if (fraction == 1.0)
        return native;
    double value = static_cast<double>(native) * fraction;
    const double nearest = std::round(value);
    if (std::abs(value - nearest) <= 1e-6 * std::max(1.0, std::abs(value)))
        value = nearest;
    uint32_t result = static_cast<uint32_t>(std::ceil(value));
    if (result & 1u)
        ++result;
    if (fraction < 1.0)
        result = std::min(result, native & ~1u);
    return std::max(2u, result);
}

inline Layout Build(const Settings& s, uint32_t nativeW, uint32_t nativeH, float globalScale)
{
    Layout result {};
    result.settings = s;
    result.requested = s.enabled;
    result.nativeW = nativeW;
    result.nativeH = nativeH;
    result.globalScale = globalScale;
    const float ordinaryScale = std::isfinite(globalScale) ? std::clamp(globalScale, .25f, 2.0f) : 1.0f;
    result.ordinaryW = std::max(1u, static_cast<uint32_t>(std::lround(nativeW * ordinaryScale)));
    result.ordinaryH = std::max(1u, static_cast<uint32_t>(std::lround(nativeH * ordinaryScale)));
    result.modelW = result.ordinaryW;
    result.modelH = result.ordinaryH;
    if (!s.enabled)
    {
        result.reason = "disabled";
        return result;
    }
    if (nativeW < 2 || nativeH < 2 || !std::isfinite(globalScale) || globalScale < .25f || globalScale > 2.0f)
    {
        result.reason = "invalid frame or global scale";
        return result;
    }
    const auto validAxis = [](float center, float work, float offset, float shift)
    {
        if (!std::isfinite(center) || !std::isfinite(work) || !std::isfinite(offset) || !std::isfinite(shift) ||
            center <= 0 || work < 25 || work > 100 || center >= work ||
            std::abs(offset) > MaxCenterOffset(center) + 1e-4f)
            return false;
        const auto limits = WorkShiftLimits(center, work, offset);
        return shift >= limits.first - 1e-4f && shift <= limits.second + 1e-4f;
    };
    if (!validAxis(s.centerX, s.workX, s.offsetX, s.shiftX) || !validAxis(s.centerY, s.workY, s.offsetY, s.shiftY))
    {
        result.reason = "invalid centre, work, offset or shift";
        return result;
    }
    const uint32_t rawW = EvenExtent(nativeW, s.workX * .01);
    const uint32_t rawH = EvenExtent(nativeH, s.workY * .01);
    const uint32_t modelW = EvenExtent(nativeW, static_cast<double>(globalScale) * s.workX * .01);
    const uint32_t modelH = EvenExtent(nativeH, static_cast<double>(globalScale) * s.workY * .01);
    if (modelW < nativeW * .25f || modelH < nativeH * .25f || s.centerX * .01f >= static_cast<float>(rawW) / nativeW ||
        s.centerY * .01f >= static_cast<float>(rawH) / nativeH)
    {
        result.reason = "effective model extent below 25% or centre outside work";
        return result;
    }
    if (s.workX == 100 && s.workY == 100)
    {
        result.reason = "work is 100% on both axes";
        return result;
    }
    result.warp.x = pw::BuildAxis(nativeW, rawW, modelW, s.centerX * .01f, s.offsetX * .01f, s.shiftX * .01f);
    result.warp.y = pw::BuildAxis(nativeH, rawH, modelH, s.centerY * .01f, s.offsetY * .01f, s.shiftY * .01f);
    const auto& x = result.warp.x;
    const auto& y = result.warp.y;
    const auto invertible = [](const pw::Axis& axis)
    {
        // A peripheral side needs at least one raw Work texel. At a shift endpoint
        // the reference split may assign zero; snapping can leave a fraction of one.
        return (axis.periphery[0] <= 0 || axis.allotted[0] >= 1.0f) &&
               (axis.periphery[1] <= 0 || axis.allotted[1] >= 1.0f);
    };
    if (!invertible(x) || !invertible(y))
    {
        result.reason = "work shift leaves less than one work pixel on one side";
        return result;
    }
    result.centerBounds = { (x.bandCenter - x.halfBand) / nativeW, (y.bandCenter - y.halfBand) / nativeH,
                            (x.bandCenter + x.halfBand) / nativeW, (y.bandCenter + y.halfBand) / nativeH };
    result.workBounds = { (x.bandCenter - x.halfBand - x.allotted[0]) / nativeW,
                          (y.bandCenter - y.halfBand - y.allotted[0]) / nativeH,
                          (x.bandCenter + x.halfBand + x.allotted[1]) / nativeW,
                          (y.bandCenter + y.halfBand + y.allotted[1]) / nativeH };
    result.modelW = modelW;
    result.modelH = modelH;
    result.active = true;
    result.reason = "active";
    return result;
}

struct alignas(16) Constants
{
    uint32_t mode = 0;
    float unused = 0;
    uint32_t width = 0, height = 0;
    pw::ShaderConstants warp {};
    float depthRect[4] = {}, motionRect[4] = {}, motionScale[2] = {}, padding[6] = {};
};
static_assert(sizeof(Constants) == 256);
static_assert(offsetof(Constants, warp) == 16);

inline Constants MakeConstants(const Layout& layout, uint32_t mode, const GuideRegions& regions, float mvX, float mvY,
                               uint32_t nativeW, uint32_t nativeH)
{
    Constants c {};
    c.mode = mode;
    c.width = mode == 102 ? layout.ordinaryW : layout.modelW;
    c.height = mode == 102 ? layout.ordinaryH : layout.modelH;
    c.warp = pw::MakeShaderConstants(layout.warp);
    c.depthRect[0] = static_cast<float>(regions.depth.x);
    c.depthRect[1] = static_cast<float>(regions.depth.y);
    c.depthRect[2] = static_cast<float>(regions.depth.width);
    c.depthRect[3] = static_cast<float>(regions.depth.height);
    c.motionRect[0] = static_cast<float>(regions.motion.x);
    c.motionRect[1] = static_cast<float>(regions.motion.y);
    c.motionRect[2] = static_cast<float>(regions.motion.width);
    c.motionRect[3] = static_cast<float>(regions.motion.height);
    (void) nativeW;
    (void) nativeH;
    // The caller supplies source-motion units converted to native pixel deltas.
    c.motionScale[0] = mvX;
    c.motionScale[1] = mvY;
    return c;
}

} // namespace DlssNr::Spatial
