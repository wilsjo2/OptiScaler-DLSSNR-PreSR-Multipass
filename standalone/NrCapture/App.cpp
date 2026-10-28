#include "App.h"
#include <commdlg.h>
namespace nr {
void App::ListSources() {
    sources.clear(); SendMessageW(Control(Sources), CB_RESETCONTENT, 0, 0);
    EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR monitor, HDC, LPRECT, LPARAM context) -> BOOL {
        auto& app = *reinterpret_cast<App*>(context); MONITORINFOEXW info{}; info.cbSize = sizeof(info);
        if (!GetMonitorInfoW(monitor, &info)) return TRUE;
        const auto label = std::format(L"Screen: {} — {}×{}{}", info.szDevice,
            info.rcMonitor.right - info.rcMonitor.left, info.rcMonitor.bottom - info.rcMonitor.top,
            info.dwFlags & MONITORINFOF_PRIMARY ? L" (primary)" : L"");
        const auto index = app.sources.size(); app.sources.emplace_back(monitor);
        SendMessageW(app.Control(Sources), CB_ADDSTRING, 0, (LPARAM)label.c_str());
        if (info.dwFlags & MONITORINFOF_PRIMARY) SendMessageW(app.Control(Sources), CB_SETCURSEL, index, 0);
        return TRUE;
    }, (LPARAM)this);
    EnumWindows([](HWND w, LPARAM context) -> BOOL {
        auto& app = *reinterpret_cast<App*>(context); wchar_t title[512]{}; DWORD process = 0;
        GetWindowThreadProcessId(w, &process);
        if (process == GetCurrentProcessId() || !IsWindowVisible(w) || IsIconic(w) ||
            (GetWindowLongPtrW(w, GWL_EXSTYLE) & WS_EX_TOOLWINDOW) || !GetWindowTextW(w, title, 512)) return TRUE;
        RECT client{}; if (!GetClientRect(w, &client) || client.right < 64 || client.bottom < 64) return TRUE;
        const auto label = std::wstring(L"Window: ") + title;
        app.sources.emplace_back(w); SendMessageW(app.Control(Sources), CB_ADDSTRING, 0, (LPARAM)label.c_str()); return TRUE;
    }, (LPARAM)this);
    if (!sources.empty() && SendMessageW(Control(Sources), CB_GETCURSEL, 0, 0) == CB_ERR)
        SendMessageW(Control(Sources), CB_SETCURSEL, 0, 0);
}
void App::ReadControls() {
    Settings next;
    { std::lock_guard lock(mutex); next = settings; }
    wchar_t path[32768]{}; GetWindowTextW(Control(RuntimePath), path, 32768); next.runtime = path;
    auto number = [&](int id, float low, float high) {
        wchar_t text[64]{}; GetWindowTextW(Control(id), text, 64); wchar_t* end = nullptr;
        float value = wcstof(text, &end);
        if (end == text || *end || !std::isfinite(value) || value < low || value > high)
            throw std::runtime_error(std::format("Enter a number between {} and {}.", low, high));
        return value;
    };
    auto checked = [&](int id) { return SendMessageW(Control(id), BM_GETCHECK, 0, 0) == BST_CHECKED; };
    auto choice = [&](int id) { return (unsigned)SendMessageW(Control(id), CB_GETCURSEL, 0, 0); };
    next.captureBackend = choice(Backend);
    next.frameGeneration = choice(FgMode) + 1;
    auto model = next.PassModel(selectedPass);
    next.strength = number(Strength, 0, 2); model.intensity = number(Intensity, 0, 2);
    next.referenceWhiteNits = number(ReferenceWhite, 0, 10000);
    next.peakBrightnessNits = number(PeakBrightness, 0, 10000);
    next.blackLevelNits = number(BlackLevel, -5, 5);
    // Quadratic slider spacing gives finer control near ordinary SDR/HDR reference whites.
    SendMessageW(Control(WhiteSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(std::sqrt(next.referenceWhiteNits * 100)));
    SendMessageW(Control(PeakSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(std::sqrt(next.peakBrightnessNits * 100)));
    SendMessageW(Control(BlackSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(next.blackLevelNits * 100));
    next.modelScale = (unsigned)SendMessageW(Control(ModelScale), TBM_GETPOS, 0, 0);
    next.transfer = (unsigned)SendMessageW(Control(Transfer), CB_GETCURSEL, 0, 0);
    next.upscaler = (unsigned)SendMessageW(Control(Upscaler), CB_GETCURSEL, 0, 0);
    auto slider = [&](int id) { return (float)SendMessageW(Control(id), TBM_GETPOS, 0, 0) / 100; };
    next.colourStrength = slider(Colour); model.skinStructure = slider(SkinStructure);
    next.skinDetail = slider(SkinDetail); next.skinColour = slider(SkinColour);
    next.environmentDetail = slider(EnvironmentDetail); next.environmentColour = slider(EnvironmentColour);
    next.separateSkin = SendMessageW(Control(SeparateSkin), BM_GETCHECK, 0, 0) == BST_CHECKED;
    next.previewSkin = SendMessageW(Control(PreviewSkin), BM_GETCHECK, 0, 0) == BST_CHECKED;
    for (int id = SkinDetail; id <= EnvironmentColour; ++id) {
        EnableWindow(Control(id), next.separateSkin); EnableWindow(Control(id + 1000), next.separateSkin);
    }
    EnableWindow(Control(PreviewSkin), next.separateSkin);
    model.localStructure = number(Structure, 0, 2); model.localTone = number(LocalTone, 0, 2);
    model.style = choice(Style); model.preset = choice(Preset); model.autoMask = checked(AutoMask);
    if (!selectedPass) next.model = model;
    else if (checked(InheritPass)) next.passOverrides[selectedPass - 1].reset();
    else next.passOverrides[selectedPass - 1] = model;
    next.unlockPasses = checked(UnlockPasses);
    next.highlightGuard = number(HighlightGuard, 1, 8);
    next.haloSuppression = number(HaloSuppression, 0, 1);
    next.haloBrightSuppression = number(HaloBrightSuppression, 0, 1);
    next.haloRadius = number(HaloRadius, 1, 4);
    EnableWindow(Control(Upscaler), next.transfer != 2);
    next.downscaler = choice(Downscaler);
    next.holdFrame = checked(HoldFrame); next.applyModel = checked(ApplyModel);
    next.debugView = choice(DebugView);
    next.compareSplit = slider(CompareSplit); next.compareZoom = slider(CompareZoom);
    next.compareSwap = checked(CompareSwap); next.compareLabels = checked(CompareLabels);
    next.labelScale = slider(LabelScale);
    EnableWindow(Control(Downscaler), next.modelScale > 100);
    next.inputColour = (unsigned)SendMessageW(Control(InputColour), CB_GETCURSEL, 0, 0);
    next.maxFps = (unsigned)std::lround(number(FpsLimit, 0, 360));
    next.enabled = SendMessageW(Control(Enabled), BM_GETCHECK, 0, 0) == BST_CHECKED;
    next.estimateMotion = SendMessageW(Control(MotionMode), BM_GETCHECK, 0, 0) == BST_CHECKED;
    next.tone = (unsigned)SendMessageW(Control(Codec), CB_GETCURSEL, 0, 0);
    next.passes = 1 + (unsigned)SendMessageW(Control(Passes), CB_GETCURSEL, 0, 0);
    next.compare = (unsigned)SendMessageW(Control(Compare), CB_GETCURSEL, 0, 0);
    next.passes = std::clamp(next.passes, 1u, next.unlockPasses ? Settings::MaxPasses : 2u);
    SaveSettings(next);
    { std::lock_guard lock(mutex); settings = next; }
}
void App::StartCapture() {
    if (restartRequired) throw std::runtime_error("Restart the application after the GPU timeout.");
    ReadControls();
    const auto selection = SendMessageW(Control(Sources), CB_GETCURSEL, 0, 0);
    if (selection < 0 || (size_t)selection >= sources.size() || !sources[selection].Exists())
        throw std::runtime_error("Select an available screen or window; use Refresh if it has changed.");
    if (SendMessageW(Control(Backend), CB_GETCURSEL, 0, 0) == 1 && !sources[selection].monitor)
        throw std::runtime_error("DXGI captures whole screens. Select a Screen source or choose Windows Graphics Capture.");
    if (worker.joinable()) { SetEvent(stop.value); worker.join(); }
    ResetEvent(stop.value); Settings current; { std::lock_guard lock(mutex); current = settings; }
    for (int id : {Sources, Refresh, RuntimePath, Browse, Start, Backend, FgMode}) EnableWindow(Control(id), FALSE);
    worker = std::jthread([this, source = sources[selection], current] { Run(source, current); });
}
LRESULT App::Message(UINT m, WPARAM w, LPARAM l) {
    try {
        if (m == WM_CREATE) { CreateControls(); return 0; }
        if (m == WM_NOTIFY && reinterpret_cast<NMHDR*>(l)->idFrom == Tabs && reinterpret_cast<NMHDR*>(l)->code == TCN_SELCHANGE) {
            ShowPage(); return 0;
        }
        if (m == StatusMessage) {
            std::lock_guard lock(mutex); SetWindowTextW(Control(Status), status.c_str());
            if (w) for (int id : {Sources, Refresh, RuntimePath, Browse, Start, Backend, FgMode}) EnableWindow(Control(id), TRUE);
            return 0;
        }
        if (m == WM_HOTKEY || (m == WM_COMMAND && LOWORD(w) == Stop)) { SetEvent(stop.value); return 0; }
        if (m == WM_HSCROLL && l) {
            const int id = GetDlgCtrlID((HWND)l);
            const auto position = SendMessageW((HWND)l, TBM_GETPOS, 0, 0);
            if (id == WhiteSlider || id == PeakSlider) {
                SetWindowTextW(Control(id == WhiteSlider ? ReferenceWhite : PeakBrightness), std::format(L"{:.0f}", position * position / 100.0).c_str());
            } else if (id == BlackSlider) {
                SetWindowTextW(Control(BlackLevel), std::format(L"{:.2f}", position / 100.0).c_str());
            } else if (id == ModelScale) {
                SetWindowTextW(Control(ScaleLabel), std::format(L"Model resolution: {}% per dimension", position).c_str());
            } else if (id >= Colour && id <= EnvironmentColour) {
                SetWindowTextW(Control(id + 1000), std::format(L"{}: {:.2f}", AppearanceNames[id - Colour], position / 100.0).c_str());
            } else if (id == CompareSplit || id == CompareZoom || id == LabelScale) {
                const auto label = id == CompareSplit ? L"Wipe split" : id == CompareZoom ? L"Side-by-side zoom" : L"Label size";
                SetWindowTextW(Control(id + 1000), std::format(L"{}: {:.2f}", label, position / 100.0).c_str());
            } else return 0;
            // Rebuild only after releasing the thumb; keyboard and page changes also apply.
            if (LOWORD(w) != TB_THUMBTRACK) ReadControls();
            return 0;
        }
        if (m == WM_COMMAND) {
            const int id = LOWORD(w), event = HIWORD(w);
            if (id == Refresh) ListSources();
            else if (id == Start) StartCapture();
            else if (id == EditPass && event == CBN_SELCHANGE) {
                ReadControls(); selectedPass = (unsigned)SendMessageW(Control(EditPass), CB_GETCURSEL, 0, 0); ShowPass();
            } else if (id == InheritPass && event == BN_CLICKED) { ReadControls(); ShowPass(); }
            else if (id == UnlockPasses && event == BN_CLICKED) {
                ReadControls(); Settings value; { std::lock_guard lock(mutex); value = settings; }
                SendMessageW(Control(Passes), CB_RESETCONTENT, 0, 0);
                for (unsigned i = 1; i <= (value.unlockPasses ? Settings::MaxPasses : 2); ++i)
                    SendMessageW(Control(Passes), CB_ADDSTRING, 0, (LPARAM)std::to_wstring(i).c_str());
                SendMessageW(Control(Passes), CB_SETCURSEL, value.passes - 1, 0); ShowPass();
            }
            else if (id == Browse) {
                wchar_t file[32768]{}; OPENFILENAMEW picker{sizeof(picker)}; picker.hwndOwner = window;
                picker.lpstrFilter = L"NVIDIA NR runtime\0nvngx_dlssnr.dll\0";
                picker.lpstrFile = file; picker.nMaxFile = 32768; picker.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
                if (GetOpenFileNameW(&picker)) SetWindowTextW(Control(RuntimePath), file);
            } else if ((((id >= Strength && id <= LocalTone) || id == ReferenceWhite || id == FpsLimit || id == HighlightGuard ||
                         id == PeakBrightness || id == BlackLevel || id == HaloSuppression ||
                         id == HaloBrightSuppression || id == HaloRadius) && event == EN_KILLFOCUS) ||
                       ((id == Codec || id == Compare || id == Passes || id == Transfer || id == Upscaler || id == InputColour ||
                         id == Downscaler || id == Style || id == Preset || id == DebugView || id == Backend || id == FgMode) && event == CBN_SELCHANGE) ||
                       ((id == Enabled || id == MotionMode || id == SeparateSkin || id == PreviewSkin || id == AutoMask ||
                         id == HoldFrame || id == ApplyModel || id == CompareSwap || id == CompareLabels) && event == BN_CLICKED)) {
                ReadControls(); if (id == Passes) ShowPass();
            }
            return 0;
        }
        if (m == WM_CLOSE) {
            SetEvent(stop.value); if (worker.joinable()) worker.join();
            UnregisterHotKey(window, 1); DestroyWindow(window); return 0;
        }
        if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    } catch (const std::exception& e) { SetWindowTextW(Control(Status), Wide(e.what()).c_str()); Log(e.what()); }
    return DefWindowProcW(window, m, w, l);
}
LRESULT CALLBACK App::Procedure(HWND hwnd, UINT m, WPARAM w, LPARAM l) {
    auto* self = reinterpret_cast<App*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (m == WM_NCCREATE) {
        self = static_cast<App*>(reinterpret_cast<CREATESTRUCTW*>(l)->lpCreateParams); self->window = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)self);
    }
    return self ? self->Message(m, w, l) : DefWindowProcW(hwnd, m, w, l);
}
int App::Main(HINSTANCE instance) {
    INITCOMMONCONTROLSEX commonControls{sizeof(commonControls), ICC_BAR_CLASSES | ICC_TAB_CLASSES};
    Check(InitCommonControlsEx(&commonControls) ? S_OK : E_FAIL, "Slider controls");
    WNDCLASSW wc{}; wc.lpfnWndProc = Procedure; wc.hInstance = instance; wc.lpszClassName = L"OptiScalerNR.Control";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    if (!RegisterClassW(&wc)) Check(HRESULT_FROM_WIN32(GetLastError()), "Control window class");
    window = CreateWindowW(wc.lpszClassName, L"Display Filter - screen and window overlay", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 986, 785, nullptr, nullptr, instance, this);
    if (!window) Check(HRESULT_FROM_WIN32(GetLastError()), "Control window");
    if (!RegisterHotKey(window, 1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_END))
        throw std::runtime_error("Ctrl+Alt+End is already reserved. Close the other application using it, then retry.");
    ShowWindow(window, SW_SHOWDEFAULT);
    Log("Display Filter controls ready.");
    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0)
        if (!IsDialogMessageW(window, &message)) { TranslateMessage(&message); DispatchMessageW(&message); }
    return 0;
}
}
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    try {
        winrt::init_apartment(winrt::apartment_type::single_threaded);
        int result; { nr::App app; result = app.Main(instance); }
        winrt::uninit_apartment(); return result;
    }
    catch (const winrt::hresult_error& e) { MessageBoxW(nullptr, e.message().c_str(), L"Display Filter", MB_ICONERROR); return 1; }
    catch (const std::exception& e) { MessageBoxW(nullptr, nr::Wide(e.what()).c_str(), L"Display Filter", MB_ICONERROR); return 1; }
}
