#pragma once
#include "Capture.h"
#include "Processor.h"
#include "Presentation.h"
#include "FramePacer.h"
#include "Presets.h"
#include "Calibration.h"
#include <commctrl.h>
#include <thread>
#include <mutex>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <winrt/base.h>
namespace nr {
constexpr UINT StatusMessage = WM_APP + 1;
constexpr UINT CalibrationMessage = WM_APP + 2; // the capture thread has a calibration result for the window
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
    FoveationPositionX, FoveationPositionY, FoveationShiftX, FoveationShiftY, ShowFoveation, FoveationNote,
    ShortcutPass, ReapplyPass, ReapplyGain, BorrowTransformer, BorrowBelow, BorrowLevel, ShortcutNote, BorrowStatus, EditInterval, EditLate,
    Nvfp4Box, Nvfp4Note, Nvfp4Swin256, Nvfp4Swin128, Nvfp4Swin64, Nvfp4Swin32,
    // One box per group of nvfp4/NrNet4.h, in the order of its bits: split blocks (6), transformer (3), input and output block.
    Nvfp4SplitFfwd, Nvfp4SplitFfwdProj, Nvfp4SplitQkv, Nvfp4SplitProj, Nvfp4SplitHead, Nvfp4SplitDec,
    Nvfp4VitQkv, Nvfp4VitAttention, Nvfp4VitProjection, Nvfp4Pre, Nvfp4Post,
    Nvfp4Options,     // text field: NrNet4::Parse items for the kernels' option words (Settings::nvfp4Net)
    // The "Presets" tab: named presets (Presets.h) and the slider calibrator (Calibration.h).
    PresetList, PresetName, PresetLoad, PresetOverwrite, PresetSaveAs, PresetRename, PresetDelete,
    CalibrateGoal, CalibrateStart, CalibrateCancel, CalibrateNote, CalibrateSkin,
    // The "Pacing and latency" tab: Settings::presentMode, latencyMode, editSmoothing, motionZeroBias.
    PresentModeBox, LatencyModeBox, EditSmoothingBox, MotionZeroBiasBox, PacingNote,
    // "Inspect and compare": what the other side of a wipe or side by side shows (Settings::compareModel).
    CompareModelBox };
// The "Model" list in the order shown, as ModelSettings::preset values (Settings::ModelANvfp4 = 4 sits beside Model A).
// 5..14 were removed models: a saved one shows no selection here and is refused when the passes are made.
// Sorted by the precision of the weights: FP8 (NVIDIA's own models and the FP8 pass-in-one models), INT8, NVFP4 (4-bit).
// One version of each pass-in-one model (Settings::CurrentModel): 23, 24, 21, 22.
inline constexpr unsigned ModelChoices[]{0, 1, 2, 3, 23, 24, 25, 27, 26, 28, 4, 21, 22};
inline constexpr const wchar_t* ModelNames[]{L"Automatic",
    L"FP8: Model A (Original)", L"FP8: Model B (Optimized, NR 310.9.16 or newer)", L"FP8: Preset 3 (legacy)",
    L"FP8 2xPass-in-one (RTX 40 and 50 series)", L"FP8 3xPass-in-one (RTX 40 and 50 series)",
    L"INT8: Model A (RTX 30, 40 and 50 series)", L"INT8: Model A - Alternate training (RTX 30, 40 and 50 series)",
    L"INT8 2xPass-in-one (RTX 30, 40 and 50 series)", L"INT8 2xPass-in-one - Alternate training (RTX 30, 40 and 50 series)",
    L"NVFP4: Model A (4-bit kernels, RTX 50 series)",
    L"NVFP4 2xPass-in-one", L"NVFP4 3xPass-in-one"};
static_assert(std::size(ModelNames) == std::size(ModelChoices));
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
    std::atomic_bool capturing = false; // the capture thread is processing frames
    // Between the window and the capture thread, under `mutex`: a calibration to start or cancel, and its result.
    std::optional<Calibration::Request> calibrationRequest;
    bool calibrationCancel = false;
    std::optional<Calibration::Result> calibrationResult;
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
    void ShowSettings(); // every control that shows a saved setting, from `settings` (after a preset or a calibration)
    void ListPresetNames(const std::wstring& select = {});
    std::wstring Chosen(int combo);
    void PresetCommand(int id);
    void StartCalibration();
    void FinishCalibration();
    LRESULT Message(UINT message, WPARAM w, LPARAM l);
    static LRESULT CALLBACK Procedure(HWND hwnd, UINT message, WPARAM w, LPARAM l);
public:
    int Main(HINSTANCE instance);
};
}
