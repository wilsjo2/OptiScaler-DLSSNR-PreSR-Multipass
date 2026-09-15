#pragma once

// Call under g_nrMutex. This is a render-seam identity, never evidence of GPU submission.
class DlssNrSeamClock
{
    bool nativeStarted = false;
    unsigned long long epoch = 0;

  public:
    unsigned long long AtSeam(bool begin, bool bridge, unsigned long long submitted)
    {
        // Bridges already supply a submitted-frame counter; preserve duplicate detection there.
        if (bridge)
            return submitted;
        // Present may advance between Before and After. Both halves retain the same identity.
        if (!begin)
            return epoch;
        if (!nativeStarted)
        {
            // Seed once for readable diagnostics, not for temporal continuity.
            epoch = submitted;
            nativeStarted = true;
        }
        else
            // A native SR evaluation is ONE rendered frame regardless of how many
            // Presents FG issued. Importing raw Present jumps here makes half-rate
            // history reset every frame with 2X/MFG and prevents every NR skip.
            // Game Reset, Cancel/failed pairs and generation changes invalidate
            // history separately. A raw Present gap alone cannot identify a cut.
            ++epoch;
        return epoch;
    }
};
