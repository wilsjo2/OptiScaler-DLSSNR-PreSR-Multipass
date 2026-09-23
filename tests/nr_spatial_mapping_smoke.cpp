#include "../OptiScaler/shaders/dlssnr/DlssNr_Spatial.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace DlssNr::Spatial;

static void Near(float a, float b, float tolerance = .01f) {
    assert(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tolerance);
}

static void ProbeAxis(const pw::Axis& axis) {
    for (float pixel : {-64.0f, 0.5f, axis.bandCenter - axis.halfBand,
                        axis.bandCenter, axis.bandCenter + axis.halfBand,
                        axis.nativeExtent - .5f, axis.nativeExtent + 64.0f}) {
        const float packed = pw::Pack(pixel, axis);
        Near(pw::Unpack(packed, axis), pixel);
    }
    float previous = pw::Pack(-64, axis);
    for (int i = 0; i <= 128; ++i) {
        const float packed = pw::Pack((axis.nativeExtent + 128) * i / 128.0f - 64, axis);
        assert(std::isfinite(packed) && packed >= previous - .01f);
        previous = packed;
    }
}

int main() {
    const Settings off{};
    auto layout = Build(off, 3840, 2160, 1.0f);
    assert(!layout.requested && !layout.active && layout.ordinaryW == 3840 && layout.modelW == 3840);

    Settings settings{};
    settings.enabled = true;
    layout = Build(settings, 3840, 2160, 1.0f);
    assert(layout.active && layout.modelW == 3456 && layout.modelH == 1944);
    Near(pw::Pack(0, layout.warp.x), 0);
    Near(pw::Pack(3840, layout.warp.x), 3456);
    Near(pw::Pack(1920, layout.warp.x), 1728);
    // Reference rational shoulder: radius .9 maps to .8666667 at centre .8/work .9.
    Near(pw::Pack(192, layout.warp.x), 64);
    ProbeAxis(layout.warp.x); ProbeAxis(layout.warp.y);
    Near(layout.centerBounds.left, .1f); Near(layout.centerBounds.right, .9f);

    const auto guides = DlssNr::GuideRegions{{16, 8, 2560, 1440}, {64, 32, 3840, 2160}};
    auto constants = MakeConstants(layout, 101, guides, 3.0f, -2.0f, 3840, 2160);
    assert(constants.mode == 101 && constants.width == 3456 && constants.height == 1944);
    assert(constants.depthRect[0] == 16 && constants.depthRect[2] == 2560);
    assert(constants.motionRect[0] == 64 && constants.motionRect[2] == 3840);
    assert(constants.motionScale[0] == 3 && constants.motionScale[1] == -2);
    constants = MakeConstants(layout, 102, guides, 3, -2, 3840, 2160);
    assert(constants.width == 3840 && constants.height == 2160);

    settings.offsetX = 3; settings.offsetY = -2;
    settings.shiftX = 1; settings.shiftY = -1;
    layout = Build(settings, 3840, 2160, 1.0f);
    assert(layout.active && layout.centerBounds.left > .1f && layout.workBounds.left > 0);
    Near(layout.warp.x.workCenter - layout.warp.x.bandCenter,
         std::round(layout.warp.x.workCenter - layout.warp.x.bandCenter), .001f);
    ProbeAxis(layout.warp.x); ProbeAxis(layout.warp.y);
    const auto limits = WorkShiftLimits(settings, false);
    settings.shiftX = limits.first;
    layout = Build(settings, 3840, 2160, 1.0f);
    assert(!layout.active && layout.requested && !layout.reason.empty());

    settings = Settings{}; settings.enabled = true;
    layout = Build(settings, 1921, 1081, .85f);
    assert(layout.active && layout.ordinaryW == 1633 && layout.modelW == 1470);
    ProbeAxis(layout.warp.x); ProbeAxis(layout.warp.y);
    layout = Build(settings, 1921, 1081, 2.0f);
    assert(layout.active && layout.modelW == 3458 && layout.ordinaryW == 3842);
    ProbeAxis(layout.warp.x); ProbeAxis(layout.warp.y);
    layout = Build(settings, 1920, 1080, .25f);
    assert(!layout.active && layout.modelW == layout.ordinaryW);

    settings.workX = settings.workY = 100;
    layout = Build(settings, 1920, 1080, 1.0f);
    assert(!layout.active && layout.modelW == 1920 && layout.reason.find("100%") != std::string::npos);

    settings.centerX = std::numeric_limits<float>::quiet_NaN();
    Settings same = settings;
    assert(settings == same && Build(settings, 1920, 1080, 1.0f) == Build(same, 1920, 1080, 1.0f));
    std::puts("PASS: NR peripheral spatial mapping, bounds, motion constants, bypasses");
}
