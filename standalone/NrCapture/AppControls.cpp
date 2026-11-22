#include "App.h"
#include <bit>
namespace nr {
void App::ShowPage() {
    const int selected = TabCtrl_GetCurSel(Control(Tabs));
    for (auto [control, page] : pages) if (page >= 0) ShowWindow(control, page == selected ? SW_SHOW : SW_HIDE);
}
void App::ShowPass() {
    Settings value; { std::lock_guard lock(mutex); value = settings; }
    for (int id : {EditPass, ResolutionPass, FoveationPass, ShortcutPass})
        SendMessageW(Control(id), CB_SETCURSEL, selectedPass, 0);
    const auto model = value.PassModel(selectedPass);
    const bool inherit = selectedPass && !value.passOverrides[selectedPass - 1];
    SendMessageW(Control(InheritPass), BM_SETCHECK, inherit ? BST_CHECKED : BST_UNCHECKED, 0);
    EnableWindow(Control(InheritPass), selectedPass != 0);
    for (auto [id, amount] : {std::pair{Intensity, model.intensity}, {Structure, model.localStructure}, {LocalTone, model.localTone}})
        SetWindowTextW(Control(id), std::format(L"{:.2f}", amount).c_str());
    SendMessageW(Control(SkinStructure), TBM_SETPOS, TRUE, (LPARAM)std::lround(model.skinStructure * 100));
    SetWindowTextW(Control(SkinStructure + 1000), std::format(L"Model skin structure: {:.2f}", model.skinStructure).c_str());
    SendMessageW(Control(Style), CB_SETCURSEL, model.style, 0);
    SendMessageW(Control(Preset), CB_SETCURSEL, std::find(std::begin(ModelChoices), std::end(ModelChoices), model.preset) - std::begin(ModelChoices), 0);
    SendMessageW(Control(AutoMask), BM_SETCHECK, model.autoMask ? BST_CHECKED : BST_UNCHECKED, 0);
    for (int id : {Intensity, Structure, LocalTone, SkinStructure, Style, Preset, AutoMask}) EnableWindow(Control(id), !inherit);
    SetWindowTextW(Control(PassNote), selectedPass >= value.passes ? L"Inactive pass: edits are saved for when this pass is enabled." :
        inherit ? L"Inherits pass 1, with Local tone 0. Uncheck inheritance to edit." : L"Model changes take effect when an edit is committed.");
    const auto resolution = value.PassResolution(selectedPass);
    const bool inheritResolution = selectedPass && !value.passResolutionOverrides[selectedPass - 1];
    for (int id : {InheritResolution, InheritFoveation}) {
        SendMessageW(Control(id), BM_SETCHECK, inheritResolution ? BST_CHECKED : BST_UNCHECKED, 0);
        EnableWindow(Control(id), selectedPass != 0);
    }
    SendMessageW(Control(ModelScale), TBM_SETPOS, TRUE, resolution.modelScale);
    SetWindowTextW(Control(ScaleLabel), std::format(L"Pass {} resolution: {}% per dimension", selectedPass + 1, resolution.modelScale).c_str());
    SendMessageW(Control(Downscaler), CB_SETCURSEL, resolution.downscaler, 0);
    SendMessageW(Control(Transfer), CB_SETCURSEL, resolution.transfer, 0);
    SendMessageW(Control(Upscaler), CB_SETCURSEL, resolution.upscaler, 0);
    EnableWindow(Control(ModelScale), !inheritResolution);
    EnableWindow(Control(Transfer), !inheritResolution);
    EnableWindow(Control(Upscaler), !inheritResolution && resolution.transfer != 2);
    EnableWindow(Control(Downscaler), !inheritResolution && resolution.modelScale > 100);
    const auto& spatial = resolution.spatial;
    SendMessageW(Control(FoveationEnabled), BM_SETCHECK, spatial.enabled ? BST_CHECKED : BST_UNCHECKED, 0);
    EnableWindow(Control(FoveationEnabled), !inheritResolution);
    for (auto [id, amount] : {std::pair{FoveationCenterWidth, spatial.centerX}, {FoveationCenterHeight, spatial.centerY},
         {FoveationWorkWidth, spatial.workX}, {FoveationWorkHeight, spatial.workY},
         {FoveationPositionX, spatial.offsetX + 50}, {FoveationPositionY, spatial.offsetY + 50},
         {FoveationShiftX, spatial.shiftX}, {FoveationShiftY, spatial.shiftY}}) {
        SetWindowTextW(Control(id), std::format(L"{:.2f}", amount).c_str());
        EnableWindow(Control(id), !inheritResolution && spatial.enabled);
    }
    SendMessageW(Control(ShowFoveation), BM_SETCHECK, value.showFoveation ? BST_CHECKED : BST_UNCHECKED, 0);
    const wchar_t* note = selectedPass >= value.passes ? L"Inactive pass: this profile is saved for later. Show area can preview its grid." :
        inheritResolution ? L"Inherits pass 1 resolution and foveation. Uncheck inheritance to edit this pass." :
        L"This pass has its own resolution, reconstruction and foveation settings.";
    SetWindowTextW(Control(ResolutionNote), note);
    std::wstring foveationNote = note;
    if (!spatial.enabled) foveationNote += L" Foveation is off; Show area displays a regular grid.";
    else if (spatial.workX == 100 && spatial.workY == 100)
        foveationNote += L" Both working extents are 100%, so this pass is not compressed.";
    SetWindowTextW(Control(FoveationNote), foveationNote.c_str());
    const auto shortcut = selectedPass ? value.passShortcuts[selectedPass - 1] : PassShortcut{};
    for (auto [id, on] : {std::pair{ReapplyPass, shortcut.reapply}, {BorrowTransformer, shortcut.borrowTransformer}, {BorrowBelow, shortcut.borrowBelow}})
        SendMessageW(Control(id), BM_SETCHECK, on ? BST_CHECKED : BST_UNCHECKED, 0);
    SendMessageW(Control(BorrowLevel), CB_SETCURSEL, std::bit_width(shortcut.borrowLevel) - 2, 0);
    // Pass 1 always runs the network; re-apply wins over both borrow options.
    EnableWindow(Control(ReapplyPass), selectedPass != 0);
    EnableWindow(Control(BorrowTransformer), selectedPass && !shortcut.reapply);
    EnableWindow(Control(BorrowBelow), selectedPass && !shortcut.reapply);
    EnableWindow(Control(BorrowLevel), selectedPass && !shortcut.reapply && shortcut.borrowBelow);
    SetWindowTextW(Control(ShortcutNote), !selectedPass ? L"Pass 1 always runs the network. Select pass 2 or later." :
        selectedPass >= value.passes ? L"Inactive pass: these choices are saved for when this pass is enabled." :
        shortcut.reapply ? L"This pass runs no network: it repeats the change of the last pass that did, times the gain." :
        shortcut.borrowBelow || shortcut.borrowTransformer ? L"This pass runs the network but reuses an earlier pass's blocks. Borrowing needs the same model, style and size as that pass; otherwise it runs normally." :
        L"This pass runs the whole network.");
    char borrow[256]{}; NrPassBorrow::Status(borrow, sizeof(borrow));
    SetWindowTextW(Control(BorrowStatus), Wide(borrow).c_str());
    SetWindowTextW(Control(Nvfp4Note), Wide(std::string(value.Nvfp4PassInOne() ? "In use: a pass has a 2xPass-in-one or 3xPass-in-one model, which runs all of its own parts (the boxes above are not followed)." :
        value.Nvfp4Pass() ? "In use: a pass has the model \"Model A - NVFP4\"." :
        "Not in use: no pass has the model \"Model A - NVFP4\" (Model passes tab).") +
        "\r\nTransformer feed-forward (4K and above): " + Nvfp4Status() +
        ".\r\nFused blocks: " + Nvfp4SwinStatus() + ".\r\nRest of the network: " + Nvfp4NetStatus() +
        ".\r\nKernel options: <family>.<group>=<number>, separated by spaces (families split, vit, prepost).").c_str());
    // Pacing and latency tab: what of its choices cannot take effect with the other settings, said plainly.
    std::wstring pacing;
    if (value.presentMode == 1 && value.frameGeneration > 1)
        pacing += L"Flip window is not in use while DLSS FG is on: the DLSS FG swapchain presents. ";
    if ((value.editSmoothing > 0 || value.motionZeroBias > 0) && !value.estimateMotion)
        pacing += L"Steady edit and Keep still areas still are not in use: Estimate motion is off. ";
    else if (value.editSmoothing > 0) {
        if (value.HasPassResolutionOverrides()) pacing += L"Steady edit is not in use: a pass has its own resolution settings. ";
        else if (value.PassResolution(0).spatial.enabled) pacing += L"Steady edit is not in use with foveation. ";
        else if (value.editInterval > 1) pacing += L"Steady edit is not in use with an edit interval above 1 (the model does not run every frame). ";
        else pacing += L"Steady edit is in use. ";
    }
    if (value.latencyMode == 1 && value.frameGeneration > 1)
        pacing += L"Low latency with DLSS FG: Reflex and the newest frame are used, but FG still holds one real frame back.";
    SetWindowTextW(Control(PacingNote), pacing.c_str());
}
void App::ShowSettings() {
    Settings value; { std::lock_guard lock(mutex); value = settings; }
    const auto text = [&](int id, float amount) { SetWindowTextW(Control(id), std::format(L"{:.2f}", amount).c_str()); };
    const auto check = [&](int id, bool on) { SendMessageW(Control(id), BM_SETCHECK, on ? BST_CHECKED : BST_UNCHECKED, 0); };
    const auto select = [&](int id, unsigned index) { SendMessageW(Control(id), CB_SETCURSEL, index, 0); };
    const auto slide = [&](int id, const wchar_t* name, float amount) {
        SendMessageW(Control(id), TBM_SETPOS, TRUE, (LPARAM)std::lround(amount * 100));
        SetWindowTextW(Control(id + 1000), std::format(L"{}: {:.2f}", name, amount).c_str());
    };
    SetWindowTextW(Control(FpsLimit), std::to_wstring(value.maxFps).c_str());
    check(UnlockPasses, value.unlockPasses);
    SendMessageW(Control(Passes), CB_RESETCONTENT, 0, 0);
    for (unsigned i = 1; i <= (value.unlockPasses ? Settings::MaxPasses : 2); ++i)
        SendMessageW(Control(Passes), CB_ADDSTRING, 0, (LPARAM)std::to_wstring(i).c_str());
    select(Passes, value.passes - 1);
    select(InputColour, value.inputColour); select(Codec, value.tone);
    text(ReferenceWhite, value.referenceWhiteNits);
    SendMessageW(Control(WhiteSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(std::sqrt(value.referenceWhiteNits * 100)));
    check(MotionMode, value.estimateMotion);
    text(Strength, value.strength); slide(Colour, AppearanceNames[0], value.colourStrength);
    text(HighlightGuard, value.highlightGuard); check(SeparateSkin, value.separateSkin);
    const float amounts[]{value.skinDetail, value.skinColour, value.environmentDetail, value.environmentColour};
    for (int id = SkinDetail; id <= EnvironmentColour; ++id) {
        slide(id, AppearanceNames[id - Colour], amounts[id - SkinDetail]);
        EnableWindow(Control(id), value.separateSkin); EnableWindow(Control(id + 1000), value.separateSkin);
    }
    EnableWindow(Control(PreviewSkin), value.separateSkin);
    text(PeakBrightness, value.peakBrightnessNits);
    SendMessageW(Control(PeakSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(std::sqrt(value.peakBrightnessNits * 100)));
    text(BlackLevel, value.blackLevelNits);
    SendMessageW(Control(BlackSlider), TBM_SETPOS, TRUE, (LPARAM)std::lround(value.blackLevelNits * 100));
    text(HaloSuppression, value.haloSuppression); text(HaloBrightSuppression, value.haloBrightSuppression); text(HaloRadius, value.haloRadius);
    text(ReapplyGain, value.reapplyGain);
    select(EditInterval, std::clamp(value.editInterval, 1u, 4u) - 1); check(EditLate, value.editLate);
    check(Nvfp4Box, value.nvfp4FeedForward);
    const auto bits = [&](int first, unsigned count, unsigned mask) { for (unsigned i = 0; i < count; ++i) check(first + int(i), (mask >> i & 1) != 0); };
    check(Nvfp4Swin32, value.nvfp4Swin & 1); check(Nvfp4Swin64, value.nvfp4Swin & 2); check(Nvfp4Swin128, value.nvfp4Swin & 4); check(Nvfp4Swin256, value.nvfp4Swin & 8);
    bits(Nvfp4SplitFfwd, 6, value.nvfp4Split); bits(Nvfp4VitQkv, 3, value.nvfp4Vit); bits(Nvfp4Pre, 2, value.nvfp4PrePost);
    SetWindowTextW(Control(Nvfp4Options), Wide(value.nvfp4Net).c_str());
    select(PresentModeBox, std::min(value.presentMode, 1u)); select(LatencyModeBox, std::min(value.latencyMode, 1u));
    text(EditSmoothingBox, value.editSmoothing); text(MotionZeroBiasBox, value.motionZeroBias);
    ShowPass(); // the controls of the pass being edited, and the notes
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
    for (auto text : {L"Model passes", L"Input and resolution", L"Composition", L"HDR output", L"Inspect and compare", L"Halo suppression", L"Foveation", L"Faster passes", L"NVFP4", L"Presets", L"Pacing and latency"}) {
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
    // In the order of ModelChoices. Saved values are the runtime preset (0 auto, 1 = Model A, 2 = Model B, 3) or
    // Settings::ModelANvfp4 (4), so saved profiles keep their meaning.
    combo(L"Model", Preset, {}, 0, 36, 508);
    for (auto name : ModelNames) SendMessageW(Control(Preset), CB_ADDSTRING, 0, (LPARAM)name);
    SendMessageW(Control(Preset), CB_SETCURSEL, std::find(std::begin(ModelChoices), std::end(ModelChoices), settings.model.preset) - std::begin(ModelChoices), 0);
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
    combo(L"Edit pass resolution", ResolutionPass, {}, 0, 506, 232);
    for (unsigned i = 1; i <= Settings::MaxPasses; ++i)
        SendMessageW(Control(ResolutionPass), CB_ADDSTRING, 0, (LPARAM)std::format(L"Pass {}", i).c_str());
    box(L"Inherit pass 1 resolution + foveation", InheritResolution, false, 506, 289);
    label(L"", 506, 324, 410, ScaleLabel);
    Add(TRACKBAR_CLASSW, L"", TBS_NOTICKS | WS_TABSTOP, 506, 348, 410, 30, ModelScale);
    SendMessageW(Control(ModelScale), TBM_SETRANGE, FALSE, MAKELPARAM(25, 200));
    SendMessageW(Control(ModelScale), TBM_SETPOS, TRUE, settings.modelScale);
    combo(L"Above 100%: downsampling filter", Downscaler, {L"FSR1", L"Bicubic", L"Catmull-Rom", L"Lanczos2", L"Lanczos3", L"Kaiser2", L"Kaiser3", L"MAGIC"}, settings.downscaler, 506, 388);
    combo(L"Below 100%: edit transfer", Transfer, {L"Classic composition", L"Matched residuals", L"Edge-aware (spatial)", L"Difference"}, settings.transfer, 506, 452);
    combo(L"Below 100%: upscaling", Upscaler, {L"Bilinear", L"Private DLSS SR"}, settings.upscaler, 506, 516);
    EnableWindow(Control(Upscaler), settings.transfer != 2);
    Add(L"STATIC", L"", 0, 506, 578, 410, 38, ResolutionNote);
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
    // What a wipe or side by side puts beside the picture: the original, or the same passes run with another model.
    combo(L"Other side of a wipe or side by side", CompareModelBox, {L"Original picture"}, 0, 36, 400);
    for (auto name : ModelNames) SendMessageW(Control(CompareModelBox), CB_ADDSTRING, 0, (LPARAM)name);
    combo(L"Debug view", DebugView, {L"Off", L"Proxy (model input)", L"Raw model output", L"Difference (20x)"}, 0, 36, 470);
    Add(L"STATIC", L"A model on the other side: every pass runs with it instead of its own model, on the left (swap: right). Side by side then shows the middle of both pictures at full size, without labels.", 0, 36, 540, 410, 80, 0);
    Add(L"STATIC", L"Hold and inspect settings are session-only. The output stays on behind these controls. Ctrl+Alt+End stops capture.", 0, 506, 548, 410, 60, 0);
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
    buildingPage = 6;
    combo(L"Edit pass foveation", FoveationPass, {}, 0, 36, 232);
    for (unsigned i = 1; i <= Settings::MaxPasses; ++i)
        SendMessageW(Control(FoveationPass), CB_ADDSTRING, 0, (LPARAM)std::format(L"Pass {}", i).c_str());
    box(L"Inherit pass 1 resolution + foveation", InheritFoveation, false, 36, 291);
    box(L"Enable foveation for this pass", FoveationEnabled, false, 36, 333);
    edit(L"Focus width (%)", FoveationCenterWidth, 80, 36, 383);
    edit(L"Focus height (%)", FoveationCenterHeight, 80, 36, 428);
    edit(L"Focus centre X (0-100%)", FoveationPositionX, 50, 36, 473);
    edit(L"Focus centre Y (0-100%)", FoveationPositionY, 50, 36, 518);
    edit(L"Working width (%)", FoveationWorkWidth, 90, 506, 235);
    edit(L"Working height (%)", FoveationWorkHeight, 90, 506, 280);
    edit(L"Work shift X (%)", FoveationShiftX, 0, 506, 325);
    edit(L"Work shift Y (%)", FoveationShiftY, 0, 506, 370);
    box(L"Show area (distorted wireframe grid)", ShowFoveation, false, 506, 423);
    Add(L"STATIC", L"Working size reduces outer detail while keeping the focus at this pass's resolution.\r\nWork shift redistributes outer detail.\r\n\r\nShow area previews the selected pass.", 0, 506, 464, 410, 108, 0);
    Add(L"STATIC", L"", 0, 36, 578, 880, 38, FoveationNote);
    buildingPage = 7;
    combo(L"Edit pass", ShortcutPass, {}, 0, 36, 232);
    for (unsigned i = 1; i <= Settings::MaxPasses; ++i)
        SendMessageW(Control(ShortcutPass), CB_ADDSTRING, 0, (LPARAM)std::format(L"Pass {}", i).c_str());
    box(L"Re-apply previous change", ReapplyPass, false, 36, 292);
    box(L"Borrow pass 1's transformer", BorrowTransformer, false, 36, 332);
    box(L"Borrow everything at 1/x size and below", BorrowBelow, false, 36, 372);
    combo(L"Borrow size (1/x)", BorrowLevel, {L"1/2", L"1/4", L"1/8", L"1/16", L"1/32"}, 4, 36, 407);
    edit(L"Re-apply gain (0-2, all passes)", ReapplyGain, settings.reapplyGain, 36, 480);
    // Settings::editInterval: a 100 % pass, or one below 100 % with the "Difference" transfer.
    combo(L"The model runs on (100 % passes and Difference transfer)", EditInterval,
          {L"every frame", L"every 2nd frame (the edit is re-applied between)", L"every 3rd frame", L"every 4th frame"},
          std::clamp(settings.editInterval, 1u, 4u) - 1, 36, 540);
    box(L"Picture first, model after it (edit one frame older)", EditLate, settings.editLate, 36, 600);
    Add(L"STATIC", L"Re-apply runs no network for this pass: it adds the last network pass's change again, times the gain (0.81 fitted; 1 = same change). Consecutive re-apply passes cost one small dispatch.\r\n\r\nBorrowing runs the network but takes the transformer, or every block at 1/x size and below, from the earlier pass of this frame, saving GPU time. Smaller x borrows more and saves more.", 0, 506, 238, 410, 190, 0);
    Add(L"STATIC", L"", 0, 506, 440, 410, 80, ShortcutNote);
    Add(L"STATIC", L"", 0, 506, 530, 410, 80, BorrowStatus);
    buildingPage = 8;
    {
        // The model's network as 4-bit (NVFP4) kernels of ours, a box per part; an unticked part runs NVIDIA's 8-bit kernel.
        Add(L"STATIC", L"Run parts of the network as 4-bit (NVFP4) kernels. Experimental, RTX 50 series only; every part is switched on its own and applies to all passes. All parts but the transformer feed-forward are Model A's.", 0, 36, 232, 880, 36, 0);
        struct Item { int id; const wchar_t* text; int width; bool checked; };
        const auto row = [&](const wchar_t* title, int y, std::initializer_list<Item> items) {
            Add(L"STATIC", title, 0, 36, y + 3, 200, 22, 0);
            int x = 240;
            for (const auto& item : items) {
                Add(L"BUTTON", item.text, BS_AUTOCHECKBOX | WS_TABSTOP, x, y, item.width, 25, item.id);
                SendMessageW(Control(item.id), BM_SETCHECK, item.checked ? BST_CHECKED : BST_UNCHECKED, 0);
                x += item.width + 6;
            }
        };
        const auto bit = [](unsigned mask, unsigned n) { return ((mask >> n) & 1) != 0; };
        row(L"Input and output", 280, {{Nvfp4Pre, L"input block", 110, bit(settings.nvfp4PrePost, 0)}, {Nvfp4Post, L"output block", 110, bit(settings.nvfp4PrePost, 1)}});
        row(L"Fused blocks, width", 314, {{Nvfp4Swin32, L"32", 58, bit(settings.nvfp4Swin, 0)}, {Nvfp4Swin64, L"64", 58, bit(settings.nvfp4Swin, 1)},
            {Nvfp4Swin128, L"128", 58, bit(settings.nvfp4Swin, 2)}, {Nvfp4Swin256, L"256", 58, bit(settings.nvfp4Swin, 3)}});
        row(L"Split blocks, width 512", 348, {{Nvfp4SplitQkv, L"attention", 90, bit(settings.nvfp4Split, 2)}, {Nvfp4SplitProj, L"its projection", 110, bit(settings.nvfp4Split, 3)},
            {Nvfp4SplitFfwd, L"feed-forward", 105, bit(settings.nvfp4Split, 0)}, {Nvfp4SplitFfwdProj, L"its projection", 110, bit(settings.nvfp4Split, 1)},
            {Nvfp4SplitHead, L"bottleneck head", 125, bit(settings.nvfp4Split, 4)}, {Nvfp4SplitDec, L"decoder input", 110, bit(settings.nvfp4Split, 5)}});
        row(L"Transformer", 382, {{Nvfp4VitQkv, L"QKV", 60, bit(settings.nvfp4Vit, 0)}, {Nvfp4VitAttention, L"attention", 90, bit(settings.nvfp4Vit, 1)},
            {Nvfp4VitProjection, L"projection", 95, bit(settings.nvfp4Vit, 2)}, {Nvfp4Box, L"feed-forward", 200, settings.nvfp4FeedForward}});
        // Option words of the kernels (NrNet4::Parse items such as "prepost.pre=0 vit.attention=0 split.qkv=1"); the
        // boxes above decide which groups run, so group lists typed here are overridden by them.
        Add(L"STATIC", L"Kernel options", 0, 36, 453, 200, 22, 0);
        Add(L"EDIT", Wide(settings.nvfp4Net).c_str(), WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, 240, 450, 676, 25, Nvfp4Options);
        Add(L"STATIC", L"", 0, 36, 484, 880, 132, Nvfp4Note);
    }
    buildingPage = 9;
    label(L"Preset", 36, 232);
    Add(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, 36, 256, 410, 300, PresetList);
    Add(L"BUTTON", L"Load", WS_TABSTOP, 36, 294, 130, 28, PresetLoad);
    Add(L"BUTTON", L"Overwrite", WS_TABSTOP, 176, 294, 130, 28, PresetOverwrite);
    Add(L"BUTTON", L"Delete", WS_TABSTOP, 316, 294, 130, 28, PresetDelete);
    label(L"Name", 36, 346);
    Add(L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, 36, 370, 410, 25, PresetName);
    SendMessageW(Control(PresetName), EM_SETLIMITTEXT, 64, 0);
    Add(L"BUTTON", L"Save as new preset", WS_TABSTOP, 36, 406, 200, 28, PresetSaveAs);
    Add(L"BUTTON", L"Rename selected", WS_TABSTOP, 246, 406, 200, 28, PresetRename);
    Add(L"STATIC", L"A preset is the current settings saved under a name: the passes and their models, resolution, composition, HDR output, halo suppression, foveation, faster passes, NVFP4, motion and target FPS.\r\n\r\nIt leaves alone the runtime path, capture backend, DLSS FG, Enable NR and the Inspect and compare tab.", 0, 36, 456, 410, 150, 0);
    label(L"Calibrate the edit pass to this preset", 506, 232);
    Add(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, 506, 256, 410, 300, CalibrateGoal);
    Add(L"BUTTON", L"Calibrate", WS_TABSTOP, 506, 294, 200, 28, CalibrateStart);
    Add(L"BUTTON", L"Cancel", WS_TABSTOP, 716, 294, 100, 28, CalibrateCancel);
    EnableWindow(Control(CalibrateCancel), FALSE);
    Add(L"BUTTON", L"Enable skin settings optimisation", BS_AUTOCHECKBOX | WS_TABSTOP, 506, 330, 410, 24, CalibrateSkin);
    Add(L"STATIC", L"", 0, 506, 360, 410, 118, CalibrateNote);
    Add(L"STATIC", L"Holds the current frame and renders the preset to match on it (all of its passes). Then it tries values of Model intensity, Local structure and Local tone of the pass chosen under \"Edit pass\", and of Detail and Colour strength, until the picture is as close to that preset's as it gets, and leaves the best values in the controls. Skin settings optimisation adds the four skin / environment sliders (switched on) and Highlight guard.\r\n\r\nOther enabled passes keep running as set. Capture must be running.", 0, 506, 482, 410, 134, 0);
    buildingPage = 10;
    // Settings::presentMode, latencyMode, editSmoothing, motionZeroBias. The first takes effect when capture starts.
    combo(L"Presentation with DLSS FG off (set before Start capture)", PresentModeBox,
          {L"Overlay surface (original)", L"Flip window (for NVIDIA Smooth Motion)"}, std::min(settings.presentMode, 1u), 36, 232);
    combo(L"Pacing", LatencyModeBox, {L"Standard (original)", L"Low latency (each new frame at once)"}, std::min(settings.latencyMode, 1u), 36, 296);
    edit(L"Steady edit (0 = off, up to 0.9)", EditSmoothingBox, settings.editSmoothing, 36, 368);
    edit(L"Keep still areas still (0 = off, up to 0.5)", MotionZeroBiasBox, settings.motionZeroBias, 36, 413);
    Add(L"STATIC", L"Steady edit blends the model's change with the previous frame's, moved along the estimated motion, so the effect flickers less (0.7 about halves it on still areas); where the picture itself changed, the new change is used. Keep still areas still makes the motion estimate prefer \"no movement\" where that fits nearly as well; in tests it changed flicker very little, so leave it at 0 unless still areas shimmer. Both need Estimate motion (Input and resolution tab).",
        0, 36, 458, 410, 160, 0);
    Add(L"STATIC", L"Smooth Motion (NVIDIA driver frame generation): DLSS FG Off, Flip window, and Smooth Motion on for display_filter.exe in the NVIDIA App (Graphics, Program settings), or \"smooth_motion_profile.exe on\" from an administrator prompt. The game itself should not use Smooth Motion. Standard pacing with Target FPS at the game's rate gives it the most even frames; Low latency is sooner but passes on uneven arrival.\r\n\r\nLow latency: each captured frame is processed as soon as it arrives (no separate timers), and the newest frame is taken again just before recording. Target FPS still caps the average (10 % room), without holding back frames that arrive unevenly; set it at or above the game's rate, or 0.\r\n\r\nThe model's GPU time is also in the delay: \"Picture first\" (Faster passes tab) takes it out, at one frame of edit lag. DLSS FG holds one real frame back to make the one in between; that cannot be removed.",
        0, 506, 232, 410, 300, 0);
    Add(L"STATIC", L"", 0, 506, 540, 410, 76, PacingNote);
    buildingPage = -1;
    Add(L"STATIC", status.c_str(), 0, 16, 647, 935, 83, Status);
    ListPresetNames(); ShowPass(); ShowPage(); ListSources();
}
}
