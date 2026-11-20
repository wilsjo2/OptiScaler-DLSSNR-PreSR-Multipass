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
}
void Calibration::Start(const Request& request, const Settings& current) {
    if (Active()) throw std::runtime_error("A calibration is running already.");
    if (request.pass >= std::min(current.passes, Settings::MaxPasses))
        throw std::runtime_error(std::format("Pass {} is not active: choose an active pass under \"Edit pass\".", request.pass + 1));
    if (current.Shortcut(request.pass).reapply)
        throw std::runtime_error(std::format("Pass {} re-applies the previous change and runs no model: choose a pass that runs the model.", request.pass + 1));
    goal = Fixed(request.goal); working = Fixed(current);
    goalName = request.goalName; pass = request.pass; skin = request.skin;
    const auto model = working.PassModel(pass);
    // (With "Separate skin / environment" off the four sliders have no effect: the picture is the one they give at 1.)
    const bool separate = working.separateSkin;
    start = {model.intensity, model.localStructure, model.localTone, working.strength, working.colourStrength,
        separate ? working.skinDetail : 1.f, separate ? working.skinColour : 1.f,
        separate ? working.environmentDetail : 1.f, separate ? working.environmentColour : 1.f, working.highlightGuard};
    reader.Reset(); goalPicture = {}; floor = -1; result.reset(); anyRendered = false;
    phase = Phase::Goal;
    Begin(goal, 2);
}
void Calibration::Cancel(const wchar_t* reason) {
    if (!Active()) return;
    phase = Phase::Idle; pending = false;
    result = Result{false, pass, start, reason, skin};
}
Settings Calibration::Applied(Settings s, unsigned pass, const NrCalibrate::Point& values, bool skin) {
    auto model = s.PassModel(pass);
    model.intensity = values[NrCalibrate::Intensity];
    model.localStructure = values[NrCalibrate::Structure];
    model.localTone = values[NrCalibrate::Tone];
    if (pass) s.passOverrides[pass - 1] = model; else s.model = model;
    s.strength = values[NrCalibrate::Detail]; s.colourStrength = values[NrCalibrate::Colour];
    if (skin) {
        s.separateSkin = true;
        s.skinDetail = values[NrCalibrate::SkinDetail]; s.skinColour = values[NrCalibrate::SkinColour];
        s.environmentDetail = values[NrCalibrate::EnvironmentDetail]; s.environmentColour = values[NrCalibrate::EnvironmentColour];
        s.highlightGuard = values[NrCalibrate::Guard];
    }
    return s;
}
Settings Calibration::Trial(const NrCalibrate::Point& values) const { return Applied(working, pass, values, skin); }
Settings Calibration::Frame() const { return phase == Phase::Goal ? goal : Trial(solver.Trial()); }
void Calibration::Begin(const Settings& next, unsigned frames) {
    // Detail and Colour strength, the skin / environment sliders and Highlight guard are applied after every model
    // pass: a change of those alone leaves the models and their history as they are. (At Detail strength 0 the models
    // do not run at all.)
    auto a = next, b = rendered;
    a.strength = b.strength = 1; a.colourStrength = b.colourStrength = 1;
    a.separateSkin = b.separateSkin = false; a.skinDetail = b.skinDetail = 1; a.skinColour = b.skinColour = 1;
    a.environmentDetail = b.environmentDetail = 1; a.environmentColour = b.environmentColour = 1;
    a.highlightGuard = b.highlightGuard = 2;
    changed = !anyRendered || a != b || next.strength == 0 || rendered.strength == 0;
    settle = changed ? SettleFrames(next) : 0; measure = frames; measured = 0; sum = {};
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
    const bool read = reader.Read(picture, layout.Footprint.Format, data, layout.Footprint.Width, layout.Footprint.Height, layout.Footprint.RowPitch);
    readback->Unmap(0, &none);
    if (!read) { Cancel(L"Calibration stopped: the processed picture could not be read."); return; }
    if (phase == Phase::Goal && measured == 1) floor = NrCalibrate::Difference(sum, picture);
    if (!sum.Add(picture)) { Cancel(L"Calibration stopped: the picture changed size."); return; }
    ++measured;
    if (--measure) return;
    sum.Scale(1.0f / float(measured));
    Measured();
}
void Calibration::Measured() {
    if (phase == Phase::Goal) {
        goalPicture = std::move(sum);
        if (trace) trace(std::format(L"goal read: {}x{} values, two frames differ by {:.3f} dB", goalPicture.width, goalPicture.height, NrCalibrate::Decibels(floor)));
        NrCalibrate::Solver::Options options; // the sliders' own ranges
        options.skin = skin;
        solver.Start(start, options);
        phase = Phase::Working;
    } else {
        const double loss = NrCalibrate::Difference(goalPicture, sum);
        if (loss < 0) { Cancel(L"Calibration stopped: the picture changed size."); return; }
        if (trace) {
            const auto& v = solver.Trial();
            auto line = std::format(L"try {:3}: intensity {:.2f} structure {:.2f} tone {:.2f} detail {:.2f} colour {:.2f}",
                solver.Trials() + 1, v[0], v[1], v[2], v[3], v[4]);
            if (skin) line += std::format(L" skin {:.2f}/{:.2f} environment {:.2f}/{:.2f} guard {:.2f}", v[5], v[6], v[7], v[8], v[9]);
            trace(line + std::format(L" -> {:.3f} dB", NrCalibrate::Decibels(loss)));
        }
        solver.Report(loss);
        if (solver.Done()) { Finish(); return; }
    }
    Begin(Trial(solver.Trial()), 1);
}
void Calibration::Finish() {
    phase = Phase::Idle;
    const auto& best = solver.Best();
    std::wstring text = std::format(L"Calibrated pass {} to \"{}\" with {} measurements: likeness {:.2f} dB before, {:.2f} dB after",
        pass + 1, goalName, solver.Trials(), NrCalibrate::Decibels(solver.FirstLoss()), NrCalibrate::Decibels(solver.BestLoss()));
    if (floor >= 0) text += std::format(L" (two frames of the preset to match: {:.2f} dB)", NrCalibrate::Decibels(floor));
    text += L".";
    const wchar_t* names[]{L"Model intensity", L"Local structure", L"Local tone", L"Detail strength", L"Colour strength",
        L"Skin detail", L"Skin colour", L"Environment detail", L"Environment colour", L"Highlight guard"};
    const unsigned count = solver.Sliders();
    for (unsigned i = 0; i < count; ++i)
        text += std::format(L" {} {:.2f} -> {:.2f}{}", names[i], start[i], best[i], i + 1 < count ? L"," : L".");
    if (skin && !working.separateSkin) text += L" \"Separate skin / environment\" was switched on.";
    Log(Narrow(text));
    result = Result{true, pass, best, text, skin};
}
std::wstring Calibration::Status() const {
    if (phase == Phase::Goal) return std::format(L"Calibrating: rendering the preset to match \"{}\" on the held frame...", goalName);
    if (phase == Phase::Working) {
        if (!solver.Trials()) return L"Calibrating: measuring the current slider values...";
        return std::format(L"Calibrating pass {}: {} model settings tried, likeness {:.2f} dB (started at {:.2f} dB).", pass + 1,
            solver.ModelTrials(), NrCalibrate::Decibels(solver.BestLoss()), NrCalibrate::Decibels(solver.FirstLoss()));
    }
    return {};
}
}
