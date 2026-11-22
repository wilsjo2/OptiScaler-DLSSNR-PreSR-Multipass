#pragma once
#include "Gpu.h"
#include "CaptureSource.h"
#include <memory>
namespace nr {
class Capture {
public:
    Capture(Gpu& gpu, CaptureSource source, unsigned backend = 0, double maxFps = 0);
    ~Capture();
    void StopProducer(); // Keep queued surfaces alive until processing has drained.
    void Stop(); // Caller has drained its processing queue.
    void SetFrameRate(double fps); // Processing cap, including the FG divisor; zero is uncapped.
    void SetLowLatency(bool on); // Settings::latencyMode: acquire each frame as it arrives (see CapturePacer).
    bool Next(); // Processing thread; previous GPU submission must be complete.
    // As Next, but only a frame of this size is taken (a newer frame of another size waits for the next Next).
    bool NextOfSize(UINT width, UINT height);
    ID3D12Resource* Input() const;
    UINT width = 0, height = 0;
    double timestamp = 0;
    uint64_t receivedFrames = 0, publishedFrames = 0;
    double copyMilliseconds = 0;
    HANDLE Event() const;
    bool Closed() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
