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
#include <vector>

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

// NR frames to wait after a change before the picture is read. A model slider restarts temporal history: after it
// the pictures are read one after another until they stop changing (NrCalibrate::Settling).
constexpr unsigned kSettleModel = 8, kSettleBlend = 2;
// Pictures read at most while one settles.
constexpr unsigned kSettlePictures = 8, kSettleGoalPictures = 12;
// The goal has nothing to be judged against yet: it counts as settled when two pictures in a row differ by less
// than 80 dB (this figure / 2000).
constexpr double kGoalScale = 2e-5;
// NR frames without a picture before the run gives up.
constexpr unsigned kPatience = 900;

// The NR thread's own state.
struct Run
{
    enum class Phase
    {
        Idle,
        Goal,
        Mask,
        Working
    };
    Phase phase = Phase::Idle;
    std::string goalName;
    unsigned pass = 0;
    // skin: the request's option. regions: the skin mask could be read, so skin and environment are judged
    // separately and their sliders are searched. separate: "Separate skin / environment controls" was on at the start.
    bool skin = false, regions = false, separate = false;
    std::string skinNote;    // why the skin sliders were left alone, when they were
    Presets::Values working; // the settings in use when the run began
    // Session choices forced while calibrating, and what they were.
    bool hold = false, apply = true, skinMask = false;
    uint32_t compare = 0, debugView = 0;
    NrCalibrate::Reader reader;
    NrCalibrate::Picture goalPicture, arrived;
    NrCalibrate::SkinMask mask;
    NrCalibrate::Settling settling;
    bool settlingActive = false;
    bool hasArrived = false, unreadable = false;
    NrCalibrate::Solver solver;
    NrCalibrate::Point start {}, applied {};
    // The skin and environment differences of every set of values measured, to report those of the best one.
    std::vector<std::pair<NrCalibrate::Point, NrCalibrate::Regions>> measured;
    double floor = -1; // difference of the last two pictures of the preset to match
    unsigned settle = 0, quiet = 0;
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
        config.DlssNrMaxRatio = values[NrCalibrate::Guard];
    if (run.regions)
    {
        config.DlssNrSkinProtection = true;
        config.DlssNrSkinDetail = values[NrCalibrate::SkinDetail];
        config.DlssNrSkinColour = values[NrCalibrate::SkinColour];
        config.DlssNrEnvironmentDetail = values[NrCalibrate::EnvironmentDetail];
        config.DlssNrEnvironmentColour = values[NrCalibrate::EnvironmentColour];
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

// The next picture is read `frames` NR frames from now; after a model change, pictures are read until they settle.
void Await(unsigned frames, bool settles, unsigned pictures = kSettlePictures)
{
    run.settle = frames;
    run.settlingActive = settles;
    if (settles)
        run.settling.Begin(pictures);
    run.quiet = 0;
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
    Await(modelChanged ? kSettleModel : kSettleBlend, modelChanged);
}

// Ends the run. `best`: the slider values found; without them the settings go back to what they were at the start.
void End(Config& config, capture::FrameCapture& frames, const std::string& text, const NrCalibrate::Point* best)
{
    Presets::Apply(config, run.working);
    if (best)
        SetValues(config, run.pass, *best);
    RestoreSession(config);
    run.phase = Run::Phase::Idle;
    run.goalPicture = {};
    run.arrived = {};
    run.mask = {};
    run.measured = {};
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
        return refuse(std::format("pass {} is not active. Choose an active pass in the pass tabs.", request.pass + 1));
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
    Await(kSettleModel, true, kSettleGoalPictures);
    Publish(true, std::format("Calibrating: rendering the preset to match \"{}\" on the held frame...", run.goalName));
}

std::string Likeness(double difference) { return std::format("{:.2f} dB", NrCalibrate::Decibels(difference)); }

// The search begins: the settings in use again, and the first values to measure.
void BeginWorking(Config& config)
{
    Presets::Apply(config, run.working);
    NrCalibrate::Solver::Options options; // the sliders' own ranges
    options.skin = run.skin;
    if (run.skin && !run.regions)
        // Without a usable skin mask the four skin / environment sliders stay where they are; Highlight guard is
        // still searched.
        for (unsigned i = NrCalibrate::SkinDetail; i <= NrCalibrate::EnvironmentColour; ++i)
            options.low[i] = options.high[i] = run.start[i];
    run.solver.Start(run.start, options);
    run.phase = Run::Phase::Working;
    SetValues(config, run.pass, run.solver.Trial());
    run.applied = run.solver.Trial();
    // The working settings' history starts again after the goal's.
    Await(kSettleModel, true);
    Publish(true, "Calibrating: measuring the current slider values...");
}

// The goal is read. With the skin option the composition's own skin mask is read next, on the settings in use.
void BeginMask(Config& config)
{
    Presets::Apply(config, run.working);
    config.DlssNrSkinProtection = true;
    config.DlssNrShowSkinMask = true;
    run.phase = Run::Phase::Mask;
    Await(kSettleBlend + 1, false);
    Publish(true, "Calibrating: reading the skin mask of the held frame...");
}

const NrCalibrate::Regions* RegionsOf(const NrCalibrate::Point& values)
{
    for (auto it = run.measured.rbegin(); it != run.measured.rend(); ++it)
        if (it->first == values)
            return &it->second;
    return nullptr;
}

// A picture arrived.
void Arrived(Config& config, capture::FrameCapture& frames)
{
    const auto stop = [&](const std::string& why)
    { End(config, frames, "Calibration stopped: " + why + " The settings are as they were.", nullptr); };

    if (run.phase == Run::Phase::Goal)
    {
        if (!run.settling.Add(run.arrived, kGoalScale))
            return;
        run.floor = run.settling.Change();
        run.goalPicture = std::move(run.arrived);
        run.arrived = {};
        return run.skin ? BeginMask(config) : BeginWorking(config);
    }
    if (run.phase == Run::Phase::Mask)
    {
        config.DlssNrShowSkinMask = false;
        run.regions = run.mask.Fits(run.goalPicture) && run.mask.Usable();
        if (!run.mask.Fits(run.goalPicture))
            run.skinNote = "The skin mask could not be read at this NR placement, so the skin and environment sliders "
                           "were left as they were.";
        else if (!run.regions)
            run.skinNote = std::format("Skin is {:.1f} % of the held frame, too little (or too much) to set the skin "
                                       "and environment sliders: they were left as they were. Hold a frame with a "
                                       "face in view.",
                                       run.mask.share * 100.0);
        return BeginWorking(config);
    }

    // Working: a set of values is judged once its picture has settled.
    if (run.settlingActive)
    {
        const double scale = NrCalibrate::Difference(run.goalPicture, run.arrived);
        if (scale < 0)
            return stop("NR's picture has another size with the preset to match than with the current settings (do "
                        "both run NR at the same place and size?).");
        if (!run.settling.Add(run.arrived, scale))
            return;
        run.settlingActive = false;
    }
    double loss;
    NrCalibrate::Regions regions;
    if (run.regions)
    {
        regions = NrCalibrate::Difference(run.goalPicture, run.arrived, run.mask);
        loss = regions.Valid() ? regions.Balanced() : -1.0;
    }
    else
        loss = NrCalibrate::Difference(run.goalPicture, run.arrived);
    if (loss < 0)
        return stop("NR's picture has another size with the preset to match than with the current settings (do both "
                    "run NR at the same place and size?).");
    if (run.regions)
        run.measured.emplace_back(run.solver.Trial(), regions);
    run.solver.Report(loss);
    if (!run.solver.Done())
    {
        ApplyTrial(config);
        Publish(true, std::format("Calibrating pass {}: {} measurements, likeness {} (started at {}).", run.pass + 1,
                                  run.solver.Trials(), Likeness(run.solver.BestLoss()),
                                  Likeness(run.solver.FirstLoss())));
        return;
    }

    const auto& best = run.solver.Best();
    auto text = std::format("Calibrated pass {} to \"{}\" with {} measurements.", run.pass + 1, run.goalName,
                            run.solver.Trials());
    const auto* after = run.regions ? RegionsOf(best) : nullptr;
    if (after && !run.measured.empty())
    {
        const auto& before = run.measured.front().second;
        text += std::format(" Likeness of the environment: {} before, {} after. Of the skin ({:.1f} % of the frame): {} "
                            "before, {} after.",
                            Likeness(before.environment), Likeness(after->environment), run.mask.share * 100.0,
                            Likeness(before.skin), Likeness(after->skin));
    }
    else
        text += std::format(" Likeness: {} before, {} after.", Likeness(run.solver.FirstLoss()),
                            Likeness(run.solver.BestLoss()));
    if (run.floor >= 0)
        text += std::format(" (Two pictures of the preset to match: {}.)", Likeness(run.floor));
    // The sliders that were searched: the five, Highlight guard with the skin option, and the four skin /
    // environment sliders when the mask allowed it.
    bool any = false;
    for (unsigned i = 0; i < NrCalibrate::Count; ++i)
    {
        const bool four = i >= NrCalibrate::SkinDetail && i <= NrCalibrate::EnvironmentColour;
        if (i >= NrCalibrate::BasicCount && (!run.skin || (four && !run.regions)))
            continue;
        text += std::format("{} {} {:.2f} -> {:.2f}", any ? "," : "", NrCalibrate::Names[i], run.start[i], best[i]);
        any = true;
    }
    text += ".";
    if (run.regions && !run.separate)
        text += " \"Separate skin / environment controls\" was switched on.";
    if (!run.skinNote.empty())
        text += " " + run.skinNote;
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
        Arrived(config, frames);
        if (run.phase == Run::Phase::Idle)
            return;
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
                                                  read = run.phase == Run::Phase::Mask
                                                             ? NrCalibrate::Reader::ReadMask(
                                                                   run.mask, footprint.Format, data, footprint.Width,
                                                                   footprint.Height, footprint.RowPitch)
                                                             : run.reader.Read(run.arrived, footprint.Format, data,
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
