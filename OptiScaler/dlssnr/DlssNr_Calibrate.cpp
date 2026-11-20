#include "pch.h"
#include "DlssNr_Calibrate.h"

#include "DlssNr_Capture.h"
#include "DlssNr_Placement.h"
#include "DlssNr_Presets.h"
#include "NrCalibrate.h"
#include "PassProfiles.h"

#include <Config.h>

#include <format>
#include <mutex>
#include <optional>

namespace DlssNr::Calibrate
{
namespace
{
// Between the menu and the NR thread.
std::mutex shared;
struct Request
{
    std::string goal;
    unsigned pass = 0;
    bool skin = false;
};
std::optional<Request> wanted;
bool cancelWanted = false;
Report published;

void Publish(bool active, std::string text)
{
    std::lock_guard lock(shared);
    published = { active, std::move(text) };
}

// NR frames to let pass after a change before a picture is measured: a change of a model slider makes the pass
// anew and its history starts over; the other sliders only change the blend.
constexpr unsigned kSettleModel = 10, kSettleBlend = 2;
// NR frames without a picture before the run gives up.
constexpr unsigned kPatience = 900;

// The NR thread's own state.
struct Run
{
    enum class Phase
    {
        Idle,
        Goal,
        Working
    };
    Phase phase = Phase::Idle;
    std::string goalName;
    unsigned pass = 0;
    bool skin = false, separate = false; // separate: "Separate skin / environment controls" was on at the start
    Presets::Values working; // the settings in use when the run began
    // Session choices forced while calibrating, and what they were.
    bool hold = false, apply = true, skinMask = false;
    uint32_t compare = 0, debugView = 0;
    NrCalibrate::Reader reader;
    NrCalibrate::Picture goalPicture, sum, arrived;
    bool hasArrived = false, unreadable = false;
    NrCalibrate::Solver solver;
    NrCalibrate::Point start {}, applied {};
    double floor = -1; // difference of two frames of the preset to match
    unsigned settle = 0, measure = 0, measured = 0, quiet = 0;
    bool waiting = false;
};
Run run;

void SetValues(Config& config, unsigned pass, const NrCalibrate::Point& values)
{
    if (pass == 0)
    {
        config.DlssNrIntensity = values[NrCalibrate::Intensity];
        config.DlssNrLocalStructure = values[NrCalibrate::Structure];
        config.DlssNrLocalTone = values[NrCalibrate::Tone];
    }
    else
    {
        auto& extra = config.DlssNrPassOverrides[pass - 1];
        extra.intensity = values[NrCalibrate::Intensity];
        extra.structure = values[NrCalibrate::Structure];
        extra.tone = values[NrCalibrate::Tone];
    }
    config.DlssNrTransferStrength = values[NrCalibrate::Detail];
    config.DlssNrColourStrength = values[NrCalibrate::Colour];
    if (run.skin)
    {
        config.DlssNrSkinProtection = true;
        config.DlssNrSkinDetail = values[NrCalibrate::SkinDetail];
        config.DlssNrSkinColour = values[NrCalibrate::SkinColour];
        config.DlssNrEnvironmentDetail = values[NrCalibrate::EnvironmentDetail];
        config.DlssNrEnvironmentColour = values[NrCalibrate::EnvironmentColour];
        config.DlssNrMaxRatio = values[NrCalibrate::Guard];
    }
}

// The same frame and a plain NR picture for both sides.
void ForceSession(Config& config)
{
    config.DlssNrHoldFrame = true;
    config.DlssNrApplyModel = true;
    config.DlssNrCompare = 0u;
    config.DlssNrDebugView = 0u;
    config.DlssNrShowSkinMask = false;
}

void RestoreSession(Config& config)
{
    config.DlssNrHoldFrame = run.hold;
    config.DlssNrApplyModel = run.apply;
    config.DlssNrCompare = run.compare;
    config.DlssNrDebugView = run.debugView;
    config.DlssNrShowSkinMask = run.skinMask;
}

void Measure(unsigned frames, bool modelChanged)
{
    run.settle = modelChanged ? kSettleModel : kSettleBlend;
    run.measure = frames;
    run.measured = 0;
    run.quiet = 0;
    run.sum = {};
    run.waiting = false;
    run.hasArrived = false;
}

void ApplyTrial(Config& config)
{
    const auto& next = run.solver.Trial();
    const bool modelChanged = next[NrCalibrate::Intensity] != run.applied[NrCalibrate::Intensity] ||
                              next[NrCalibrate::Structure] != run.applied[NrCalibrate::Structure] ||
                              next[NrCalibrate::Tone] != run.applied[NrCalibrate::Tone] ||
                              // (At Detail strength 0 the composition leaves the picture alone.)
                              next[NrCalibrate::Detail] == 0 || run.applied[NrCalibrate::Detail] == 0;
    SetValues(config, run.pass, next);
    run.applied = next;
    Measure(1, modelChanged);
}

// Ends the run. `best`: the slider values found; without them the settings go back to what they were at the start.
void End(Config& config, capture::FrameCapture& frames, const std::string& text, const NrCalibrate::Point* best)
{
    if (best)
        SetValues(config, run.pass, *best);
    else
        Presets::Apply(config, run.working);
    RestoreSession(config);
    run.phase = Run::Phase::Idle;
    run.goalPicture = {};
    run.sum = {};
    run.arrived = {};
    if (!frames.isActive())
        frames.release(); // the readback buffers it kept between pictures
    LOG_INFO("NR calibration: {}", text);
    Publish(false, text);
}

void Begin(Config& config, const Request& request)
{
    const auto refuse = [&](const std::string& why) { Publish(false, "Calibration did not start: " + why); };
    if (!config.DlssNrEnabled.value_or_default())
        return refuse("Neural Rendering is switched off.");
    const unsigned passes = std::max(config.DlssNrPasses.value_or_default(), 1u);
    if (request.pass >= passes || request.pass > std::size(config.DlssNrPassOverrides))
        return refuse(std::format("pass {} is not active. Choose an active pass under \"Edit pass\".", request.pass + 1));
    if (Profiles::PassReapplies(config, request.pass))
        return refuse(std::format("pass {} re-applies the previous change and runs no model.", request.pass + 1));
    Presets::Values goal;
    if (const auto problem = Presets::Read(request.goal, goal); !problem.empty())
        return refuse(problem);

    run = {};
    run.goalName = request.goal;
    run.pass = request.pass;
    run.skin = request.skin;
    run.separate = config.DlssNrSkinProtection.value_or_default();
    run.working = Presets::Capture(config);
    run.hold = config.DlssNrHoldFrame.value_or_default();
    run.apply = config.DlssNrApplyModel.value_or_default();
    run.compare = config.DlssNrCompare.value_or_default();
    run.debugView = config.DlssNrDebugView.value_or_default();
    run.skinMask = config.DlssNrShowSkinMask.value_or_default();
    const auto model = Profiles::PassSettings(config, request.pass);
    // (With the separate skin / environment controls off the four sliders have no effect: the picture is the one
    // they give at 1.)
    run.start = { model.intensity,
                  model.localStructure,
                  model.localTone,
                  config.DlssNrTransferStrength.value_or_default(),
                  config.DlssNrColourStrength.value_or_default(),
                  run.separate ? config.DlssNrSkinDetail.value_or_default() : 1.0f,
                  run.separate ? config.DlssNrSkinColour.value_or_default() : 1.0f,
                  run.separate ? config.DlssNrEnvironmentDetail.value_or_default() : 1.0f,
                  run.separate ? config.DlssNrEnvironmentColour.value_or_default() : 1.0f,
                  config.DlssNrMaxRatio.value_or_default() };
    run.applied = run.start;
    // A picture NR hands on before the game's tone mapping has no known scale; a finished picture has.
    run.reader.sceneLight = !config.DlssNrFinishedPicture.value_or_default();

    ForceSession(config);
    Presets::Apply(config, goal);
    run.phase = Run::Phase::Goal;
    Measure(2, true);
    Publish(true, std::format("Calibrating: rendering the preset to match \"{}\" on the held frame...", run.goalName));
}

std::string Likeness(double difference) { return std::format("{:.2f} dB", NrCalibrate::Decibels(difference)); }

// A picture arrived and enough frames were measured: the goal is done, or a set of values was judged.
void Measured(Config& config, capture::FrameCapture& frames)
{
    run.sum.Scale(1.0f / float(run.measured));
    if (run.phase == Run::Phase::Goal)
    {
        run.goalPicture = std::move(run.sum);
        Presets::Apply(config, run.working);
        NrCalibrate::Solver::Options options; // the sliders' own ranges
        options.skin = run.skin;
        run.solver.Start(run.start, options);
        run.phase = Run::Phase::Working;
        // (The models are those of the settings in use again: whatever the first values are, the passes are made anew.)
        SetValues(config, run.pass, run.solver.Trial());
        run.applied = run.solver.Trial();
        Measure(1, true);
        Publish(true, "Calibrating: measuring the current slider values...");
        return;
    }
    const double loss = NrCalibrate::Difference(run.goalPicture, run.sum);
    if (loss < 0)
        return End(config, frames,
                   "Calibration stopped: NR's picture has another size with the preset to match than with the current "
                   "settings (do both run NR at the same place and size?). The settings are as they were.",
                   nullptr);
    run.solver.Report(loss);
    if (!run.solver.Done())
    {
        ApplyTrial(config);
        Publish(true, std::format("Calibrating pass {}: {} model settings tried, likeness {} (started at {}).", run.pass + 1,
                                  run.solver.ModelTrials(), Likeness(run.solver.BestLoss()),
                                  Likeness(run.solver.FirstLoss())));
        return;
    }
    const auto& best = run.solver.Best();
    auto text = std::format("Calibrated pass {} to \"{}\" with {} measurements: likeness {} before, {} after", run.pass + 1,
                            run.goalName, run.solver.Trials(), Likeness(run.solver.FirstLoss()),
                            Likeness(run.solver.BestLoss()));
    if (run.floor >= 0)
        text += std::format(" (two frames of the preset to match: {})", Likeness(run.floor));
    text += ".";
    const unsigned count = run.solver.Sliders();
    for (unsigned i = 0; i < count; ++i)
        text += std::format(" {} {:.2f} -> {:.2f}{}", NrCalibrate::Names[i], run.start[i], best[i],
                            i + 1 < count ? "," : ".");
    if (run.skin && !run.separate)
        text += " \"Separate skin / environment controls\" was switched on.";
    End(config, frames, text, &best);
}
} // namespace

void Start(const std::string& goalPreset, unsigned pass, bool skin)
{
    std::lock_guard lock(shared);
    wanted = Request { goalPreset, pass, skin };
    cancelWanted = false;
    published = { true, "Calibrating: waiting for the next NR frame..." };
}

void Cancel()
{
    std::lock_guard lock(shared);
    cancelWanted = true;
    wanted.reset();
}

Report Read()
{
    std::lock_guard lock(shared);
    return published;
}

void Composed()
{
    if (run.phase != Run::Phase::Idle && run.settle)
        --run.settle;
}

void Tick(Config& config, capture::FrameCapture& frames)
{
    std::optional<Request> request;
    bool cancel = false;
    {
        std::lock_guard lock(shared);
        request = std::move(wanted);
        wanted.reset();
        cancel = cancelWanted;
        cancelWanted = false;
    }
    const bool idle = run.phase == Run::Phase::Idle;
    if (cancel)
    {
        if (idle)
            Publish(false, "Calibration cancelled: nothing was changed.");
        else
            End(config, frames, "Calibration cancelled: the settings are as they were.", nullptr);
        return;
    }
    if (request && idle)
        Begin(config, *request);
    if (run.phase == Run::Phase::Idle)
        return;
    const auto stop = [&](const std::string& why)
    { End(config, frames, "Calibration stopped: " + why + " The settings are as they were.", nullptr); };
    if (!config.DlssNrEnabled.value_or_default())
        return stop("Neural Rendering was switched off.");
    if (++run.quiet > kPatience)
        return stop("NR's picture could not be read (NR is not running, or this placement has no readable picture).");
    if (run.waiting)
    {
        if (!run.hasArrived)
            return;
        run.waiting = run.hasArrived = false;
        run.quiet = 0;
        if (run.unreadable)
            return stop("NR's picture has a format the calibrator cannot read.");
        if (run.phase == Run::Phase::Goal && run.measured == 1)
            run.floor = NrCalibrate::Difference(run.sum, run.arrived);
        if (!run.sum.Add(run.arrived))
            return stop("NR's picture changed size.");
        ++run.measured;
        if (--run.measure == 0)
            return Measured(config, frames);
    }
    if (run.settle)
        return;
    // Ask for this frame's picture; the capture hands it over once the GPU has finished the copy.
    const bool asked = frames.request(1, [](const ReadbackImage& image)
                                      {
                                          bool read = false;
                                          const bool mapped = image.Read(
                                              [&](const void* data, const D3D12_SUBRESOURCE_FOOTPRINT& footprint)
                                              {
                                                  read = run.reader.Read(run.arrived, footprint.Format, data,
                                                                         footprint.Width, footprint.Height,
                                                                         footprint.RowPitch);
                                              });
                                          run.unreadable = !mapped || !read;
                                          run.hasArrived = true;
                                      });
    if (asked)
        run.waiting = true;
}
} // namespace DlssNr::Calibrate
