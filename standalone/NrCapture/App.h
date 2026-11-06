#pragma once
#include "Capture.h"
#include "Processor.h"
#include "Presentation.h"
#include "FramePacer.h"
#include <commctrl.h>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <winrt/base.h>
namespace nr {
constexpr UINT StatusMessage = WM_APP + 1;
enum Id { Sources = 100, Refresh, RuntimePath, Browse, Enabled, Strength, Intensity, Structure, LocalTone,
    Codec, Compare, Start, Stop, Status, MotionMode, ReferenceWhite, Passes, WhiteSlider, ModelScale,
    ScaleLabel, Transfer, Upscaler, Colour, SkinStructure, SkinDetail, SkinColour, EnvironmentDetail,
    EnvironmentColour, SeparateSkin, PreviewSkin, InputColour, FpsLimit,
    Tabs, EditPass, InheritPass, UnlockPasses, Style, Preset, AutoMask, HighlightGuard,
    Downscaler, HoldFrame, ApplyModel, DebugView, CompareSplit, CompareZoom, CompareSwap,
    CompareLabels, LabelScale, PassNote, Backend, FgMode, PeakBrightness, BlackLevel, PeakSlider, BlackSlider,
    HaloSuppression, HaloBrightSuppression, HaloRadius,
    ResolutionPass, InheritResolution, ResolutionNote, FoveationPass, InheritFoveation,
    FoveationEnabled, FoveationCenterWidth, FoveationCenterHeight, FoveationWorkWidth, FoveationWorkHeight,
    FoveationPositionX, FoveationPositionY, FoveationShiftX, FoveationShiftY, ShowFoveation, FoveationNote };
inline constexpr const wchar_t* AppearanceNames[]{L"Colour strength", L"Model skin structure", L"Skin detail / lighting",
    L"Skin colour", L"Environment detail / lighting", L"Environment colour"};
class App {
    HWND window = nullptr;
    std::vector<CaptureSource> sources;
    std::jthread worker;
    Handle stop{CreateEventW(nullptr, TRUE, FALSE, nullptr)};
    std::mutex mutex;
    Settings settings = LoadSettings();
    std::wstring status = L"Select a screen or window and the NR runtime. Ctrl+Alt+End stops capture.";
    std::atomic_bool restartRequired = false;
    unsigned selectedPass = 0;
    int buildingPage = -1;
    std::unordered_map<int, HWND> controls;
    std::vector<std::pair<HWND, int>> pages;
    HWND Control(int id) { return controls.at(id); }
    void Say(const std::wstring& message) {
        { std::lock_guard lock(mutex); status = message; }
        PostMessageW(window, StatusMessage, 0, 0);
    }
    void Add(const wchar_t* type, const wchar_t* text, DWORD style, int x, int y, int w, int h, int id) {
        HWND c = CreateWindowExW(0, type, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h,
            window, reinterpret_cast<HMENU>((INT_PTR)id), nullptr, nullptr);
        if (!c) Check(HRESULT_FROM_WIN32(GetLastError()), "Create control");
        if (id) controls[id] = c;
        pages.emplace_back(c, buildingPage);
        SendMessageW(c, WM_SETFONT, (WPARAM)GetStockObject(DEFAULT_GUI_FONT), TRUE);
    }
    void ListSources();
    void ReadControls();
    void Run(CaptureSource source, Settings initial);
    void StartCapture();
    void CreateControls();
    void ShowPage();
    void ShowPass();
    LRESULT Message(UINT message, WPARAM w, LPARAM l);
    static LRESULT CALLBACK Procedure(HWND hwnd, UINT message, WPARAM w, LPARAM l);
public:
    int Main(HINSTANCE instance);
};
}
