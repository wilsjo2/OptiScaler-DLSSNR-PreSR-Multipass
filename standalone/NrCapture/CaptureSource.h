#pragma once
#include "Common.h"
#include <dwmapi.h>
namespace nr {
// Exactly one handle is set. Both capture and presentation use the same geometry.
struct CaptureSource {
    HWND window = nullptr;
    HMONITOR monitor = nullptr;
    CaptureSource(HWND value) : window(value) {}
    CaptureSource(HMONITOR value) : monitor(value) {}
    bool Exists() const {
        MONITORINFO info{sizeof(info)};
        return monitor ? GetMonitorInfoW(monitor, &info) != FALSE : IsWindow(window) != FALSE;
    }
    bool Bounds(RECT& bounds) const {
        if (monitor) {
            MONITORINFO info{sizeof(info)};
            if (!GetMonitorInfoW(monitor, &info)) return false;
            bounds = info.rcMonitor;
        } else {
            POINT origin{};
            if (!GetClientRect(window, &bounds) || !ClientToScreen(window, &origin)) return false;
            OffsetRect(&bounds, origin.x, origin.y);
        }
        return bounds.right > bounds.left && bounds.bottom > bounds.top;
    }
    // On the desktop, focused or not: its picture can be captured and overlaid.
    bool Shown() const {
        if (monitor) return Exists();
        DWORD cloaked = 0; DwmGetWindowAttribute(window, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
        return IsWindow(window) && IsWindowVisible(window) && !IsIconic(window) && !cloaked;
    }
    bool Active() const { return Shown() && (monitor || GetForegroundWindow() == window); }
    HMONITOR Monitor() const { return monitor ? monitor : MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST); }
};
}
