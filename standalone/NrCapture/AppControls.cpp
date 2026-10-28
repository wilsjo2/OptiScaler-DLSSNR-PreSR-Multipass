#include "App.h"
namespace nr {
void App::ShowPage() {
    const int selected = TabCtrl_GetCurSel(Control(Tabs));
    for (auto [control, page] : pages) if (page >= 0) ShowWindow(control, page == selected ? SW_SHOW : SW_HIDE);
}
void App::ShowPass() {
    Settings value; { std::lock_guard lock(mutex); value = settings; }
    const auto model = value.PassModel(selectedPass);
    const bool inherit = selectedPass && !value.passOverrides[selectedPass - 1];
    SendMessageW(Control(InheritPass), BM_SETCHECK, inherit ? BST_CHECKED : BST_UNCHECKED, 0);
    EnableWindow(Control(InheritPass), selectedPass != 0);
    for (auto [id, amount] : {std::pair{Intensity, model.intensity}, {Structure, model.localStructure}, {LocalTone, model.localTone}})
        SetWindowTextW(Control(id), std::format(L"{:.2f}", amount).c_str());
    SendMessageW(Control(SkinStructure), TBM_SETPOS, TRUE, (LPARAM)std::lround(model.skinStructure * 100));
    SetWindowTextW(Control(SkinStructure + 1000), std::format(L"Model skin structure: {:.2f}", model.skinStructure).c_str());
    SendMessageW(Control(Style), CB_SETCURSEL, model.style, 0);
    SendMessageW(Control(Preset), CB_SETCURSEL, model.preset, 0);
    SendMessageW(Control(AutoMask), BM_SETCHECK, model.autoMask ? BST_CHECKED : BST_UNCHECKED, 0);
    for (int id : {Intensity, Structure, LocalTone, SkinStructure, Style, Preset, AutoMask}) EnableWindow(Control(id), !inherit);
    SetWindowTextW(Control(PassNote), selectedPass >= value.passes ? L"Inactive pass: edits are saved for when this pass is enabled." :
        inherit ? L"Inherits pass 1, with Local tone 0. Uncheck inheritance to edit." : L"Model changes take effect when an edit is committed.");
}
void App::CreateControls() {
    auto label = [&](const wchar_t* text, int x, int y, int width = 410, int id = 0) { Add(L"STATIC", text, 0, x, y, width, 22, id); };
    auto edit = [&](const wchar_t* text, int id, float value, int x, int y) {
        label(text, x, y + 3, 255); Add(L"EDIT", std::format(L"{:.2f}", value).c_str(), WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, x + 270, y, 115, 25, id);
    };
    auto combo = [&](const wchar_t* text, int id, std::initializer_list<const wchar_t*> choices, unsigned selected, int x, int y) {
        label(text, x, y); Add(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, x, y + 24, 410, 240, id);
        for (auto item : choices) SendMessageW(Control(id), CB_ADDSTRING, 0, (LPARAM)item);
        SendMessageW(Control(id), CB_SETCURSEL, selected, 0);
    };
    auto box = [&](const wchar_t* text, int id, bool checked, int x, int y) {
        Add(L"BUTTON", text, BS_AUTOCHECKBOX | WS_TABSTOP, x, y, 415, 25, id);
        SendMessageW(Control(id), BM_SETCHECK, checked ? BST_CHECKED : BST_UNCHECKED, 0);
    };
    auto slider = [&](const wchar_t* text, int id, float value, int low, int high, int x, int y) {
        label(std::format(L"{}: {:.2f}", text, value).c_str(), x, y, 410, id + 1000);
        Add(TRACKBAR_CLASSW, L"", TBS_NOTICKS | WS_TABSTOP, x, y + 24, 410, 30, id);
        SendMessageW(Control(id), TBM_SETRANGE, FALSE, MAKELPARAM(low, high));
        SendMessageW(Control(id), TBM_SETPOS, TRUE, (LPARAM)std::lround(value * 100));
    };
    label(L"Screen or window", 16, 14);
    Add(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, 16, 39, 560, 250, Sources);
    label(L"Capture backend", 590, 14, 235);
    Add(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, 590, 39, 235, 150, Backend);
    for (auto name : {L"Windows Graphics Capture", L"DXGI Desktop Duplication"})
        SendMessageW(Control(Backend), CB_ADDSTRING, 0, (LPARAM)name);
    SendMessageW(Control(Backend), CB_SETCURSEL, settings.captureBackend, 0);
    Add(L"BUTTON", L"Refresh", WS_TABSTOP, 838, 39, 110, 26, Refresh);
    label(L"NVIDIA NR runtime", 16, 76);
    Add(L"EDIT", settings.runtime.c_str(), WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, 16, 101, 810, 26, RuntimePath);
    Add(L"BUTTON", L"Browse...", WS_TABSTOP, 838, 101, 110, 26, Browse);
    Add(L"BUTTON", L"Start capture", WS_TABSTOP, 16, 143, 135, 30, Start);
    Add(L"BUTTON", L"Stop", WS_TABSTOP, 165, 143, 85, 30, Stop);
    Add(L"BUTTON", L"Enable NR", BS_AUTOCHECKBOX | WS_TABSTOP, 274, 146, 115, 25, Enabled);
    SendMessageW(Control(Enabled), BM_SETCHECK, BST_CHECKED, 0);
    label(L"DLSS FG", 400, 148, 78);
    Add(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, 486, 145, 90, 160, FgMode);
    for (auto name : {L"Off", L"2x", L"3x", L"4x"}) SendMessageW(Control(FgMode), CB_ADDSTRING, 0, (LPARAM)name);
    SendMessageW(Control(FgMode), CB_SETCURSEL, settings.frameGeneration - 1, 0);
    label(L"Target FPS (0 = uncapped)", 590, 148, 230);
    Add(L"EDIT", std::to_wstring(settings.maxFps).c_str(), WS_BORDER | ES_NUMBER | WS_TABSTOP, 835, 145, 110, 25, FpsLimit);
    Add(WC_TABCONTROLW, L"", WS_TABSTOP | WS_CLIPSIBLINGS, 16, 188, 934, 440, Tabs);
    for (auto text : {L"Model passes", L"Input and resolution", L"Composition", L"HDR output", L"Inspect and compare", L"Halo suppression"}) {
        TCITEMW item{}; item.mask = TCIF_TEXT; item.pszText = const_cast<wchar_t*>(text);
        TabCtrl_InsertItem(Control(Tabs), TabCtrl_GetItemCount(Control(Tabs)), &item);
    }
    buildingPage = 0;
    combo(L"Number of passes", Passes, {}, settings.passes - 1, 36, 232);
    for (unsigned i = 1; i <= (settings.unlockPasses ? Settings::MaxPasses : 2); ++i)
        SendMessageW(Control(Passes), CB_ADDSTRING, 0, (LPARAM)std::to_wstring(i).c_str());
    SendMessageW(Control(Passes), CB_SETCURSEL, settings.passes - 1, 0);
    box(L"Unlock up to 10 passes", UnlockPasses, settings.unlockPasses, 36, 292);
    combo(L"Edit pass (including inactive passes)", EditPass, {}, 0, 36, 337);
    for (unsigned i = 1; i <= Settings::MaxPasses; ++i)
        SendMessageW(Control(EditPass), CB_ADDSTRING, 0, (LPARAM)std::format(L"Pass {}", i).c_str());
    SendMessageW(Control(EditPass), CB_SETCURSEL, 0, 0);
    box(L"Inherit pass 1 settings (Local tone 0)", InheritPass, false, 36, 398);
    combo(L"Style", Style, {L"Standard", L"Natural", L"Cinematic"}, settings.model.style, 36, 438);
    combo(L"Model", Preset, {L"Automatic", L"Original", L"Optimized (requires a supporting runtime)", L"Preset 3 (legacy)"}, settings.model.preset, 36, 508);
    label(L"More passes increase GPU time and memory use.", 36, 582);
    edit(L"Model intensity (0-2)", Intensity, settings.model.intensity, 506, 235);
    edit(L"Local structure (0-2)", Structure, settings.model.localStructure, 506, 280);
    edit(L"Local tone (0-2)", LocalTone, settings.model.localTone, 506, 325);
    slider(L"Model skin structure", SkinStructure, settings.model.skinStructure, -100, 200, 506, 376);
    label(L"-1 follows Local structure.", 506, 440);
    box(L"Auto skin mask (model)", AutoMask, settings.model.autoMask, 506, 480);
    Add(L"STATIC", L"", 0, 506, 540, 410, 60, PassNote);

    buildingPage = 1;
    combo(L"Content colour", InputColour, {L"Auto (follows display HDR)", L"Finished SDR (also on an HDR desktop)", L"HDR"}, settings.inputColour, 36, 232);
    combo(L"HDR mapping", Codec, {L"Hybrid + composed", L"Neutwo + composed", L"Off (soft knee)", L"Hybrid + replace", L"Neutwo + replace"}, settings.tone, 36, 302);
    edit(L"Reference white (nits; 0 = desktop)", ReferenceWhite, settings.referenceWhiteNits, 36, 380);
    Add(TRACKBAR_CLASSW, L"", TBS_NOTICKS | WS_TABSTOP, 36, 415, 410, 30, WhiteSlider);
    SendMessageW(Control(WhiteSlider), TBM_SETRANGE, FALSE, MAKELPARAM(0, 1000));
    SendMessageW(Control(WhiteSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(std::sqrt(settings.referenceWhiteNits * 100)));
    box(L"Estimate motion (experimental)", MotionMode, settings.estimateMotion, 36, 472);
    Add(L"STATIC", L"Auto follows the display, not the source content.\r\nScreen capture has no engine depth or motion guides.", 0, 36, 524, 410, 65, 0);
    label(std::format(L"Model resolution: {}% per dimension", settings.modelScale).c_str(), 506, 232, 410, ScaleLabel);
    Add(TRACKBAR_CLASSW, L"", TBS_NOTICKS | WS_TABSTOP, 506, 260, 410, 30, ModelScale);
    SendMessageW(Control(ModelScale), TBM_SETRANGE, FALSE, MAKELPARAM(25, 200));
    SendMessageW(Control(ModelScale), TBM_SETPOS, TRUE, settings.modelScale);
    combo(L"Above 100%: downsampling filter", Downscaler, {L"FSR1", L"Bicubic", L"Catmull-Rom", L"Lanczos2", L"Lanczos3", L"Kaiser2", L"Kaiser3", L"MAGIC"}, settings.downscaler, 506, 307);
    combo(L"Below 100%: edit transfer", Transfer, {L"Classic composition", L"Matched residuals", L"Edge-aware (spatial)"}, settings.transfer, 506, 390);
    combo(L"Below 100%: upscaling", Upscaler, {L"Bilinear", L"Private DLSS SR"}, settings.upscaler, 506, 465);
    EnableWindow(Control(Upscaler), settings.transfer != 2);
    label(L"Output stays at the captured screen/window size.", 506, 548);
    EnableWindow(Control(Downscaler), settings.modelScale > 100);

    buildingPage = 2;
    edit(L"Detail strength (0-2)", Strength, settings.strength, 36, 235);
    slider(L"Colour strength", Colour, settings.colourStrength, 0, 400, 36, 287);
    edit(L"Highlight guard (1-8x)", HighlightGuard, settings.highlightGuard, 36, 368);
    box(L"Separate skin / environment", SeparateSkin, settings.separateSkin, 36, 423);
    box(L"Preview colour-based skin mask", PreviewSkin, false, 36, 463);
    Add(L"STATIC", L"Replace modes bypass strength and highlight controls.\r\nColour: 0 keeps original colours; 1 uses model colours.\r\nAbove 1 increases chroma.\r\nSkin/environment: 0 keeps the original; 1 applies the edit.", 0, 36, 516, 420, 85, 0);
    const float amounts[]{settings.skinDetail, settings.skinColour, settings.environmentDetail, settings.environmentColour};
    for (int id = SkinDetail; id <= EnvironmentColour; ++id) {
        slider(AppearanceNames[id - Colour], id, amounts[id - SkinDetail], 0, 100, 506, 236 + (id - SkinDetail) * 82);
        EnableWindow(Control(id), settings.separateSkin); EnableWindow(Control(id + 1000), settings.separateSkin);
    }
    EnableWindow(Control(PreviewSkin), settings.separateSkin);

    buildingPage = 3;
    edit(L"Peak brightness (nits; 0 = off)", PeakBrightness, settings.peakBrightnessNits, 36, 238);
    Add(TRACKBAR_CLASSW, L"", TBS_NOTICKS | WS_TABSTOP, 36, 277, 410, 30, PeakSlider);
    SendMessageW(Control(PeakSlider), TBM_SETRANGE, FALSE, MAKELPARAM(0, 1000));
    SendMessageW(Control(PeakSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(std::sqrt(settings.peakBrightnessNits * 100)));
    edit(L"Black level (nits; -5 to +5)", BlackLevel, settings.blackLevelNits, 36, 347);
    Add(TRACKBAR_CLASSW, L"", TBS_NOTICKS | WS_TABSTOP, 36, 386, 410, 30, BlackSlider);
    SendMessageW(Control(BlackSlider), TBM_SETRANGE, FALSE, MAKELPARAM(-500, 500));
    SendMessageW(Control(BlackSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(settings.blackLevelNits * 100));
    Add(L"STATIC", L"Peak: softly rolls off highlights above 75% of the target.\r\nBlack: positive lifts shadows; negative darkens them.\r\nBoth at 0 preserve the existing output.", 0, 36, 460, 410, 95, 0);
    Add(L"STATIC", L"Applies after NR on an HDR display, in all HDR mapping modes.\r\n\r\nOriginal comparison, NR bypass and debug views stay unchanged.\r\n\r\nThese controls adjust the filter's output. They do not change Windows or the source application's HDR settings, or recover clipped source detail.", 0, 506, 238, 410, 220, 0);

    buildingPage = 4;
    box(L"Hold captured frame", HoldFrame, false, 36, 232);
    box(L"Apply model edit (uncheck to keep NR running hidden)", ApplyModel, true, 36, 278);
    combo(L"Comparison", Compare, {L"Processed", L"Original", L"Wipe", L"Side by side"}, 0, 36, 330);
    combo(L"Debug view", DebugView, {L"Off", L"Proxy (model input)", L"Raw model output", L"Difference (20x)"}, 0, 36, 415);
    Add(L"STATIC", L"Hold and inspect settings are session-only.\r\nFocus the source to see the output.\r\nCtrl+Alt+End stops capture.", 0, 36, 519, 410, 75, 0);
    slider(L"Wipe split", CompareSplit, settings.compareSplit, 0, 100, 506, 234);
    slider(L"Side-by-side zoom", CompareZoom, settings.compareZoom, 100, 200, 506, 312);
    box(L"Swap comparison sides", CompareSwap, settings.compareSwap, 506, 393);
    box(L"Label comparison sides", CompareLabels, settings.compareLabels, 506, 435);
    slider(L"Label size", LabelScale, settings.labelScale, 50, 500, 506, 482);
    buildingPage = 5;
    edit(L"Dark rims near bright objects (0-1)", HaloSuppression, settings.haloSuppression, 36, 238);
    edit(L"Dark rims near dark objects (0-1)", HaloBrightSuppression, settings.haloBrightSuppression, 36, 300);
    edit(L"Radius (1-4 NR pixels)", HaloRadius, settings.haloRadius, 36, 362);
    Add(L"STATIC", L"Experimental. Lifts fine dark rims while keeping broad lighting.\r\n\r\nBoth strengths at 0 disable the filter.\r\n\r\nCan lighten real shadows; compare before keeping it on.\r\nApplied once after all model passes, before resizing.", 0, 506, 238, 410, 220, 0);
    buildingPage = -1;
    Add(L"STATIC", status.c_str(), 0, 16, 647, 935, 83, Status);
    ShowPass(); ShowPage(); ListSources();
}
}
