#include "Calibration.h"
namespace nr {
namespace {
// What both sides run with while calibrating: the same held frame, the processed picture with nothing drawn over it,
// and the model on every frame (the every-n-th-frame shortcuts would show an edit made with earlier values).
Settings Fixed(Settings s) {
    s.holdFrame = true; s.enabled = true; s.applyModel = true;
    s.compare = 0; s.debugView = 0; s.previewSkin = false; s.showFoveation = false;
    s.editInterval = 1; s.editLate = false;
    return s;
}
// With estimated motion the model blends its history in, so the picture needs some frames to stop changing.
unsigned SettleFrames(const Settings& s) { return s.estimateMotion ? 8u : 0u; }
// The goal has nothing to be judged against yet: it counts as settled when two pictures in a row differ by less than
// 80 dB (this figure / 2000).
constexpr double GoalScale = 2e-5;
constexpr unsigned GoalPictures = 12;
std::wstring Likeness(double difference) { return std::format(L"{:.2f} dB", NrCalibrate::Decibels(difference)); }
}
void Calibration::Start(const Request& request, const Settings& current) {
    if (Active()) throw std::runtime_error("A calibration is running already.");
    if (request.pass >= std::min(current.passes, Settings::MaxPasses))
        throw std::runtime_error(std::format("Pass {} is not active: choose an active pass under \"Edit pass\".", request.pass + 1));
    if (current.Shortcut(request.pass).reapply)
        throw std::runtime_error(std::format("Pass {} re-applies the previous change and runs no model: choose a pass that runs the model.", request.pass + 1));
    goal = Fixed(request.goal); working = Fixed(current);
    goalName = request.goalName; pass = request.pass; skin = request.skin; regions = false; skinNote.clear();
    mask = {}; measured.clear();
    const auto model = working.PassModel(pass);
    // (With "Separate skin / environment" off the four sliders have no effect: the picture is the one they give at 1.)
    const bool separate = working.separateSkin;
    start = {model.intensity, model.localStructure, model.localTone, working.strength, working.colourStrength,
        separate ? working.skinDetail : 1.f, separate ? working.skinColour : 1.f,
        separate ? working.environmentDetail : 1.f, separate ? working.environmentColour : 1.f, working.highlightGuard};
    reader.Reset(); goalPicture = {}; floor = -1; result.reset(); anyRendered = false;
    phase = Phase::Goal;
    Begin(goal, true, GoalPictures);
}
void Calibration::Cancel(const wchar_t* reason) {
    if (!Active()) return;
    phase = Phase::Idle; pending = false;
    result = Result{false, pass, start, reason, skin, regions};
}
Settings Calibration::Applied(Settings s, unsigned pass, const NrCalibrate::Point& values, bool skin, bool regions) {
    auto model = s.PassModel(pass);
    model.intensity = values[NrCalibrate::Intensity];
    model.localStructure = values[NrCalibrate::Structure];
    model.localTone = values[NrCalibrate::Tone];
    if (pass) s.passOverrides[pass - 1] = model; else s.model = model;
    s.strength = values[NrCalibrate::Detail]; s.colourStrength = values[NrCalibrate::Colour];
    if (skin) s.highlightGuard = values[NrCalibrate::Guard];
    if (regions) {
        s.separateSkin = true;
        s.skinDetail = values[NrCalibrate::SkinDetail]; s.skinColour = values[NrCalibrate::SkinColour];
        s.environmentDetail = values[NrCalibrate::EnvironmentDetail]; s.environmentColour = values[NrCalibrate::EnvironmentColour];
    }
    return s;
}
Settings Calibration::Trial(const NrCalibrate::Point& values) const { return Applied(working, pass, values, skin, regions); }
// The composition's own skin mask of the held frame, on the settings in use.
Settings Calibration::MaskFrame() const { auto s = working; s.separateSkin = true; s.previewSkin = true; return s; }
Settings Calibration::Frame() const { return phase == Phase::Goal ? goal : phase == Phase::Mask ? MaskFrame() : Trial(solver.Trial()); }
// `always`: read pictures until they stop changing even when the models' history was not restarted (the goal).
void Calibration::Begin(const Settings& next, bool always, unsigned pictures) {
    // Detail and Colour strength, the skin / environment sliders and Highlight guard are applied after every model
    // pass: a change of those alone leaves the models and their history as they are. (At Detail strength 0 the models
    // do not run at all.)
    auto a = next, b = rendered;
    a.strength = b.strength = 1; a.colourStrength = b.colourStrength = 1;
    a.separateSkin = b.separateSkin = false; a.skinDetail = b.skinDetail = 1; a.skinColour = b.skinColour = 1;
    a.environmentDetail = b.environmentDetail = 1; a.environmentColour = b.environmentColour = 1;
    a.highlightGuard = b.highlightGuard = 2; a.previewSkin = b.previewSkin = false;
    changed = !anyRendered || a != b || next.strength == 0 || rendered.strength == 0;
    settle = changed ? SettleFrames(next) : 0;
    // With history the frames waited above are a start only: the pictures after them are read until they settle.
    settles = always || (changed && settle);
    if (settles) settling.Begin(pictures);
    rendered = next; anyRendered = true;
}
void Calibration::Record(Gpu& gpu, ID3D12Resource* output) {
    if (!Active() || pending) return;
    changed = false;
    if (settle) { --settle; return; }
    auto description = output->GetDesc();
    if (!NrCalibrate::Reader::Readable(description.Format)) {
        Cancel(L"Calibration stopped: the processed picture has a format the calibrator cannot read."); return;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT wanted{}; UINT64 size = 0;
    gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &wanted, nullptr, nullptr, &size);
    if (!readback || size != bytes || wanted.Footprint.Width != layout.Footprint.Width || wanted.Footprint.Height != layout.Footprint.Height ||
        wanted.Footprint.Format != layout.Footprint.Format) {
        // (The loop records a frame only after the one before has completed, so the old buffer is no longer in use.)
        readback = gpu.Buffer(size, D3D12_HEAP_TYPE_READBACK); layout = wanted; bytes = size;
    }
    D3D12_TEXTURE_COPY_LOCATION from{}; from.pResource = output; from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION to{}; to.pResource = readback.Get(); to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = layout;
    gpu.commands->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    pending = true;
}
void Calibration::Completed() {
    if (!Active() || !pending) return;
    pending = false;
    void* data = nullptr; D3D12_RANGE range{0, SIZE_T(bytes)}, none{};
    Check(readback->Map(0, &range, &data), "Read the calibration picture");
    NrCalibrate::Picture picture;
    const auto& f = layout.Footprint;
    const bool read = phase == Phase::Mask ? NrCalibrate::Reader::ReadMask(mask, f.Format, data, f.Width, f.Height, f.RowPitch)
                                           : reader.Read(picture, f.Format, data, f.Width, f.Height, f.RowPitch);
    readback->Unmap(0, &none);
    if (!read) { Cancel(L"Calibration stopped: the processed picture could not be read."); return; }
    if (phase == Phase::Goal) {
        if (!settling.Add(picture, GoalScale)) return; // the next frame is read as well
        floor = settling.Change();
        if (trace) trace(std::format(L"goal read: {}x{} values after {} pictures, the last two differ by {:.3f} dB", picture.width, picture.height,
            settling.Pictures(), NrCalibrate::Decibels(floor)));
        goalPicture = std::move(picture);
        if (!skin) { BeginWorking(); return; }
        phase = Phase::Mask;
        Begin(MaskFrame(), false);
    } else if (phase == Phase::Mask) {
        regions = mask.Fits(goalPicture) && mask.Usable();
        if (!mask.Fits(goalPicture))
            skinNote = L" The skin mask could not be read, so the skin and environment sliders were left as they were.";
        else if (!regions)
            skinNote = std::format(L" Skin is {:.1f} % of the held frame, too little (or too much) to set the skin and environment sliders: "
                L"they were left as they were. Hold a frame with a face in view.", mask.share * 100.0);
        if (trace) trace(std::format(L"skin mask read: {:.2f} % of the frame, {}", mask.share * 100.0, regions ? L"used" : L"not used"));
        BeginWorking();
    } else Judge(picture);
}
// The search begins: the first values to measure. (After the mask the settings in use have run already.)
void Calibration::BeginWorking() {
    NrCalibrate::Solver::Options options; // the sliders' own ranges
    options.skin = skin;
    if (skin && !regions)
        // Without a usable skin mask the four skin / environment sliders stay where they are; Highlight guard is
        // still searched.
        for (unsigned i = NrCalibrate::SkinDetail; i <= NrCalibrate::EnvironmentColour; ++i) options.low[i] = options.high[i] = start[i];
    solver.Start(start, options);
    phase = Phase::Working;
    Begin(Trial(solver.Trial()), false);
}
// A picture of the values on trial: judged once it has settled.
void Calibration::Judge(const NrCalibrate::Picture& picture) {
    const wchar_t* resized = L"Calibration stopped: the picture changed size.";
    if (settles) {
        const double scale = NrCalibrate::Difference(goalPicture, picture);
        if (scale < 0) { Cancel(resized); return; }
        if (!settling.Add(picture, scale)) return;
        settles = false;
    }
    double loss; NrCalibrate::Regions parts;
    if (regions) { parts = NrCalibrate::Difference(goalPicture, picture, mask); loss = parts.Valid() ? parts.Balanced() : -1.0; }
    else loss = NrCalibrate::Difference(goalPicture, picture);
    if (loss < 0) { Cancel(resized); return; }
    if (regions) measured.emplace_back(solver.Trial(), parts);
    if (trace) {
        const auto& v = solver.Trial();
        auto line = std::format(L"try {:3}: intensity {:.2f} structure {:.2f} tone {:.2f} detail {:.2f} colour {:.2f}",
            solver.Trials() + 1, v[0], v[1], v[2], v[3], v[4]);
        if (skin) line += std::format(L" skin {:.2f}/{:.2f} environment {:.2f}/{:.2f} guard {:.2f}", v[5], v[6], v[7], v[8], v[9]);
        if (regions) line += std::format(L" -> skin {:.3f} dB, environment {:.3f} dB", NrCalibrate::Decibels(parts.skin), NrCalibrate::Decibels(parts.environment));
        else line += std::format(L" -> {:.3f} dB", NrCalibrate::Decibels(loss));
        trace(line);
    }
    solver.Report(loss);
    if (solver.Done()) { Finish(); return; }
    Begin(Trial(solver.Trial()), false);
}
void Calibration::Finish() {
    phase = Phase::Idle;
    const auto& best = solver.Best();
    std::wstring text = std::format(L"Calibrated pass {} to \"{}\" with {} measurements.", pass + 1, goalName, solver.Trials());
    const NrCalibrate::Regions* after = nullptr;
    for (auto it = measured.rbegin(); it != measured.rend() && !after; ++it) if (it->first == best) after = &it->second;
    if (after) {
        const auto& before = measured.front().second;
        text += std::format(L" Likeness of the environment: {} before, {} after. Of the skin ({:.1f} % of the frame): {} before, {} after.",
            Likeness(before.environment), Likeness(after->environment), mask.share * 100.0, Likeness(before.skin), Likeness(after->skin));
    } else text += std::format(L" Likeness: {} before, {} after.", Likeness(solver.FirstLoss()), Likeness(solver.BestLoss()));
    if (floor >= 0) text += std::format(L" (Two pictures of the preset to match: {}.)", Likeness(floor));
    const wchar_t* names[]{L"Model intensity", L"Local structure", L"Local tone", L"Detail strength", L"Colour strength",
        L"Skin detail", L"Skin colour", L"Environment detail", L"Environment colour", L"Highlight guard"};
    // The sliders that were searched: the five, Highlight guard with the skin option, and the four skin / environment
    // sliders when the mask allowed it.
    bool any = false;
    for (unsigned i = 0; i < NrCalibrate::Count; ++i) {
        const bool four = i >= NrCalibrate::SkinDetail && i <= NrCalibrate::EnvironmentColour;
        if (i >= NrCalibrate::BasicCount && (!skin || (four && !regions))) continue;
        text += std::format(L"{} {} {:.2f} -> {:.2f}", any ? L"," : L"", names[i], start[i], best[i]);
        any = true;
    }
    text += L".";
    if (regions && !working.separateSkin) text += L" \"Separate skin / environment\" was switched on.";
    text += skinNote;
    Log(Narrow(text));
    measured.clear(); goalPicture = {}; mask.skin = {};
    result = Result{true, pass, best, text, skin, regions};
}
std::wstring Calibration::Status() const {
    if (phase == Phase::Goal) return std::format(L"Calibrating: rendering the preset to match \"{}\" on the held frame...", goalName);
    if (phase == Phase::Mask) return L"Calibrating: reading the skin mask of the held frame...";
    if (phase == Phase::Working) {
        if (!solver.Trials()) return L"Calibrating: measuring the current slider values...";
        return std::format(L"Calibrating pass {}: {} model settings tried, likeness {} (started at {}).", pass + 1,
            solver.ModelTrials(), Likeness(solver.BestLoss()), Likeness(solver.FirstLoss()));
    }
    return {};
}
}
