#pragma once

// Adapted from PeripheralWarp ABI v2 at 64902dd6a02460e5f6b778504ec2a4005faf4d9c.
// Copyright (c) 2026 Yuri Grib (BeliyG3). MIT licence: see LICENSE.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pw {

struct Axis {
    float nativeExtent = 0, rawExtent = 0, workExtent = 0;
    float bandCenter = 0, workCenter = 0, halfBand = 0, scale = 1;
    float halfSpan[2] = {}, periphery[2] = {}, allotted[2] = {};
    float center[2] = {}, work[2] = {}, compression[2] = {}, edgeSlope[2] = {};
};

// Split the peripheral budget between the two sides, respecting the native side extent.
inline Axis BuildAxis(uint32_t native, uint32_t rawWork, uint32_t model,
                      float centerFraction, float offsetFraction, float shiftFraction) {
    Axis a{};
    a.nativeExtent = static_cast<float>(native);
    a.rawExtent = static_cast<float>(rawWork);
    a.workExtent = static_cast<float>(model);
    a.scale = a.workExtent / a.rawExtent;
    a.halfBand = centerFraction * a.nativeExtent * 0.5f;
    a.bandCenter = (0.5f + offsetFraction) * a.nativeExtent;
    a.halfSpan[0] = a.bandCenter;
    a.halfSpan[1] = a.nativeExtent - a.bandCenter;
    for (int i = 0; i != 2; ++i) a.periphery[i] = std::max(0.0f, a.halfSpan[i] - a.halfBand);
    const float budget = std::max(0.0f, a.rawExtent - 2.0f * a.halfBand);
    const int narrow = a.periphery[0] <= a.periphery[1] ? 0 : 1;
    const int wide = 1 - narrow;
    float base[2] = {};
    base[narrow] = std::min(0.5f * budget, a.periphery[narrow]);
    base[wide] = std::min(budget - base[narrow], a.periphery[wide]);
    const float minShift = -std::max(0.0f, std::min(base[1], a.periphery[0] - base[0]));
    const float maxShift = std::max(0.0f, std::min(base[0], a.periphery[1] - base[1]));
    const float shift = std::clamp(shiftFraction * a.nativeExtent, minShift, maxShift);
    a.allotted[0] = std::clamp(base[0] - shift, 0.0f, a.periphery[0]);
    a.allotted[1] = std::clamp(base[1] + shift, 0.0f, a.periphery[1]);
    if (rawWork == model) {
        // At 100% global scale the centre band should translate by whole texels.
        const float translation = a.halfBand + a.allotted[0] - a.bandCenter;
        const float candidates[3] = {std::round(translation), std::floor(translation), std::ceil(translation)};
        for (float candidate : candidates) {
            const float workCenter = a.bandCenter + candidate;
            const float left = workCenter - a.halfBand;
            const float right = a.rawExtent - workCenter - a.halfBand;
            if (left < -1e-3f || right < -1e-3f ||
                left > a.periphery[0] + 1e-3f || right > a.periphery[1] + 1e-3f) continue;
            a.allotted[0] = std::clamp(left, 0.0f, a.periphery[0]);
            a.allotted[1] = std::clamp(right, 0.0f, a.periphery[1]);
            break;
        }
    }
    for (int i = 0; i != 2; ++i) {
        a.center[i] = a.halfBand / a.halfSpan[i];
        a.work[i] = (a.halfBand + a.allotted[i]) / a.halfSpan[i];
        a.compression[i] = a.periphery[i] > 0 ? a.allotted[i] / a.periphery[i] : 1.0f;
        a.edgeSlope[i] = a.compression[i] * a.compression[i];
    }
    a.workCenter = (a.halfBand + a.allotted[0]) * a.scale;
    return a;
}

inline float PackRadius(float r, float center, float work, float compression, float edgeSlope) {
    r = std::abs(r);
    if (work >= 1 || center >= 1 || r <= center) return r;
    if (r > 1) return work + (r - 1) * edgeSlope;
    const float t = (r - center) / (1 - center);
    return center + (work - center) * t / (compression + (1 - compression) * t);
}

inline float UnpackRadius(float r, float center, float work, float compression, float edgeSlope) {
    r = std::abs(r);
    if (work >= 1 || center >= 1 || r <= center) return r;
    if (r > work) return 1 + (r - work) / edgeSlope;
    const float y = (r - center) / (work - center);
    const float t = compression * y / (1 - (1 - compression) * y);
    return center + (1 - center) * t;
}

inline float Pack(float nativePixel, const Axis& a) {
    const float delta = nativePixel - a.bandCenter;
    const int side = delta < 0 ? 0 : 1;
    const float radius = std::abs(delta) / a.halfSpan[side];
    const float packed = PackRadius(radius, a.center[side], a.work[side], a.compression[side], a.edgeSlope[side]);
    return a.workCenter + (delta < 0 ? -1.0f : 1.0f) * packed * a.halfSpan[side] * a.scale;
}

inline float Unpack(float workPixel, const Axis& a) {
    const float delta = workPixel - a.workCenter;
    const int side = delta < 0 ? 0 : 1;
    const float packed = std::abs(delta) / (a.halfSpan[side] * a.scale);
    const float radius = UnpackRadius(packed, a.center[side], a.work[side], a.compression[side], a.edgeSlope[side]);
    return a.bandCenter + (delta < 0 ? -1.0f : 1.0f) * radius * a.halfSpan[side];
}

struct Layout { Axis x, y; };

struct alignas(16) ShaderConstants {
    float nativeWidth, nativeHeight, workWidth, workHeight;
    float centerFractionX, centerFractionY, workFractionX, workFractionY;
    float compressionX, compressionY, edgeSlopeX, edgeSlopeY;
    uint32_t mode, colorFilter, flags, reserved;
    float bandCenterX, bandCenterY, workCenterX, workCenterY;
    float halfSpanNegX, halfSpanNegY, sideCenterNegX, sideCenterNegY;
    float halfSpanPosX, halfSpanPosY, sideCenterPosX, sideCenterPosY;
    float sideWorkNegX, sideWorkNegY, sideCompressionNegX, sideCompressionNegY;
    float sideWorkPosX, sideWorkPosY, sideCompressionPosX, sideCompressionPosY;
    float sideEdgeSlopeNegX, sideEdgeSlopeNegY, sideEdgeSlopePosX, sideEdgeSlopePosY;
    float workScaleX, workScaleY, centerOffsetX, centerOffsetY;
};
static_assert(sizeof(ShaderConstants) == 176);

inline ShaderConstants MakeShaderConstants(const Layout& l) {
    const Axis& x = l.x; const Axis& y = l.y;
    return {x.nativeExtent, y.nativeExtent, x.workExtent, y.workExtent,
        x.halfBand / (x.nativeExtent * .5f), y.halfBand / (y.nativeExtent * .5f),
        x.rawExtent / x.nativeExtent, y.rawExtent / y.nativeExtent,
        std::min(x.compression[0], x.compression[1]), std::min(y.compression[0], y.compression[1]),
        std::min(x.edgeSlope[0], x.edgeSlope[1]), std::min(y.edgeSlope[0], y.edgeSlope[1]),
        2u, 1u, 1u, 0u,
        x.bandCenter, y.bandCenter, x.workCenter, y.workCenter,
        x.halfSpan[0], y.halfSpan[0], x.center[0], y.center[0],
        x.halfSpan[1], y.halfSpan[1], x.center[1], y.center[1],
        x.work[0], y.work[0], x.compression[0], y.compression[0],
        x.work[1], y.work[1], x.compression[1], y.compression[1],
        x.edgeSlope[0], y.edgeSlope[0], x.edgeSlope[1], y.edgeSlope[1],
        x.scale, y.scale,
        x.bandCenter / x.nativeExtent - .5f, y.bandCenter / y.nativeExtent - .5f};
}

} // namespace pw
