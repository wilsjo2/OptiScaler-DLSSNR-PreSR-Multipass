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
// other passes in use run as they are set. With Request::skin the composition's separate skin / environment sliders
// and the Highlight guard are searched as well ("Separate skin / environment" is then switched on in the settings in use).
class Calibration {
public:
    struct Request { Settings goal; std::wstring goalName; unsigned pass = 0; bool skin = false; };
    struct Result {
        bool success = false; unsigned pass = 0;
        NrCalibrate::Point values{};
        std::wstring text;
        bool skin = false; // the skin / environment sliders and Highlight guard are part of `values`
    };
    // `values` put onto `settings`: pass `pass`'s three model sliders, the two strengths and, with `skin`, the
    // skin / environment sliders (switched on) and Highlight guard.
    static Settings Applied(Settings settings, unsigned pass, const NrCalibrate::Point& values, bool skin);
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
    enum class Phase { Idle, Goal, Working };
    Phase phase = Phase::Idle;
    Settings goal, working;
    Settings rendered; bool anyRendered = false; // what the last frames were processed with
    std::wstring goalName;
    unsigned pass = 0;
    bool skin = false;
    NrCalibrate::Reader reader;
    NrCalibrate::Picture goalPicture, sum;
    NrCalibrate::Solver solver;
    NrCalibrate::Point start{};
    double floor = -1;                 // difference of two frames of the preset to match: what the picture varies by on its own
    unsigned settle = 0, measure = 0, measured = 0;
    bool changed = true, pending = false;
    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT64 bytes = 0;
    std::optional<Result> result;
    Settings Trial(const NrCalibrate::Point& values) const;
    void Begin(const Settings& next, unsigned frames);
    void Measured();
    void Finish();
};
}
