#pragma once

#include <string>

class Config;
namespace capture
{
class FrameCapture;
}

// The slider calibrator in a game (the solver and the picture comparison are NrCalibrate.h's; the Display Filter has
// its own host of the same). It holds the frame, renders a saved preset (the goal, any number of passes) on it
// and reads NR's picture back; then it tries values of one pass's Intensity, Local structure and Local tone and of
// Detail and Colour strength in the settings in use until NR's picture is as close to the goal's
// as it gets, and leaves the best values in the settings. The other passes in use run as they are set. With `skin`
// the four separate skin / environment sliders (switched on) and Highlight guard are tried as well.
// Direct3D 12 only: the picture is read through the D3D12 NR path's frame capture.
namespace DlssNr::Calibrate
{
struct Report
{
    bool active = false;
    std::string text; // progress while active; the last result otherwise (empty before the first run)
};

// From the menu.
void Start(const std::string& goalPreset, unsigned pass, bool skin);
void Cancel();
Report Read();

// From the D3D12 NR path, on its thread, for the top-level NR instance only.
// Tick: once per NR frame, before the frame reads its settings. Composed: the frame's picture was written.
void Tick(Config& config, capture::FrameCapture& frames);
void Composed();
} // namespace DlssNr::Calibrate
