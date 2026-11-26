#pragma once
#include "Processor.h"
#include <dlssnr/NrCalibrate.h>
#include <functional>
namespace nr {
// The slider calibrator's host in the Display Filter (the solver and the picture comparison are NrCalibrate.h's).
// The capture loop asks it which settings to process each frame with, lets it copy the processed picture, and tells
// it when that frame's GPU work has completed. It never changes the application's settings: the result is handed
// back for the window to apply.
//
// The frame is held for the whole run. First the goal (a preset, any number of passes) is rendered on it and read
// back; then the settings in use are rendered with one set of slider values after another. Only the
// chosen pass's Intensity, Local structure and Local tone and the composition's Detail and Colour strength change; the
// other passes in use run as they are set. With Request::skin the Highlight guard is searched as well, and the
// composition's own skin mask is read on the held frame: when it has enough of both parts, skin and environment are
// compared separately (each counts the same whatever its share of the frame) and the four separate skin /
// environment sliders are searched ("Separate skin / environment" is then switched on in the settings in use).
// After a change that restarts the models' history, pictures are read until they stop changing before one is measured.
class Calibration {
public:
    struct Request { Settings goal; std::wstring goalName; unsigned pass = 0; bool skin = false; };
    struct Result {
        bool success = false; unsigned pass = 0;
        NrCalibrate::Point values{};
        std::wstring text;
        bool skin = false;    // Highlight guard is part of `values`
        bool regions = false; // so are the four skin / environment sliders (the skin mask could be used)
    };
    // `values` put onto `settings`: pass `pass`'s three model sliders, the two strengths, with `skin` Highlight guard
    // and with `regions` the skin / environment sliders (switched on).
    static Settings Applied(Settings settings, unsigned pass, const NrCalibrate::Point& values, bool skin, bool regions);
    bool Active() const { return phase != Phase::Idle; }
    void Start(const Request& request, const Settings& working); // throws std::runtime_error with the reason it cannot run
    void Cancel(const wchar_t* reason = L"Calibration cancelled: nothing was changed.");
    Settings Frame() const;                          // the settings of the next frame
    bool ResetHistory() const { return changed; }    // the first frame after a change of settings
    void Record(Gpu& gpu, ID3D12Resource* output);   // after Processor::Run, `output` in COPY_SOURCE, into the open list
    void Completed();                                // once that frame's GPU work has completed
    std::wstring Status() const;
    std::optional<Result> TakeResult() { auto taken = std::move(result); result.reset(); return taken; }
    void Abandon() { (void)readback.Detach(); }      // after a GPU timeout: the copy may still be in use
    std::function<void(const std::wstring&)> trace;  // checks: a line for every measurement
private:
    enum class Phase { Idle, Goal, Mask, Working };
    Phase phase = Phase::Idle;
    Settings goal, working;
    Settings rendered; bool anyRendered = false; // what the last frames were processed with
    std::wstring goalName;
    unsigned pass = 0;
    // skin: the request's option. regions: the skin mask could be used, so skin and environment are judged separately
    // and their sliders are searched.
    bool skin = false, regions = false;
    std::wstring skinNote;             // why the skin / environment sliders were left alone, when they were
    NrCalibrate::Reader reader;
    NrCalibrate::Picture goalPicture;
    NrCalibrate::SkinMask mask;
    NrCalibrate::Settling settling;
    bool settles = false;              // pictures are being read until they stop changing
    NrCalibrate::Solver solver;
    NrCalibrate::Point start{};
    // The skin and environment differences of every set of values measured, to report those of the best one.
    std::vector<std::pair<NrCalibrate::Point, NrCalibrate::Regions>> measured;
    double floor = -1;                 // difference of the last two pictures of the preset to match
    unsigned settle = 0;
    bool changed = true, pending = false;
    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT64 bytes = 0;
    std::optional<Result> result;
    Settings Trial(const NrCalibrate::Point& values) const;
    Settings MaskFrame() const;
    void Begin(const Settings& next, bool always, unsigned pictures = 8);
    void BeginWorking();
    void Judge(const NrCalibrate::Picture& picture);
    void Finish();
};
}
