#include "Capture.h"
#include "CaptureQueue.h"
#include "CapturePacer.h"
#include <thread>
#include <atomic>
#include <winrt/Windows.Foundation.h>
namespace nr {
struct Capture::Impl {
    CaptureQueue queue;
    std::unique_ptr<CaptureBackend> backend;
    Handle arrived{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    std::atomic_bool done = false;
    std::atomic<uint64_t> received = 0, published = 0;
    std::atomic<double> copyMs = 0;
    std::mutex errorMutex;
    std::exception_ptr error;
    CapturePacer pacing;
    std::jthread worker;
    Impl(Gpu& gpu, CaptureSource source, unsigned mode, double fps) : queue(gpu), pacing(fps) {
        if (!arrived.value) Check(HRESULT_FROM_WIN32(GetLastError()), "Capture queue event");
        if (mode > 1) throw std::runtime_error("Unknown capture backend.");
        backend = mode ? CreateDxgiCapture(queue.Device(), gpu.adapter.Get(), source) : CreateWgcCapture(queue.Device(), source);
        Log(mode ? "Capture backend: DXGI Desktop Duplication (producer thread)." : "Capture backend: Windows Graphics Capture (producer thread).");
        worker = std::jthread([this](std::stop_token stop) {
            bool apartment = false;
            try {
                winrt::init_apartment(winrt::apartment_type::multi_threaded); apartment = true;
                while (!stop.stop_requested() && !backend->Closed()) {
                    if (!pacing.Wait(stop)) break;
                    const double started = ClockMs();
                    CaptureFrame frame;
                    const bool available = backend->Next(frame);
                    received = backend->receivedFrames;
                    if (!available) continue;
                    queue.Publish(frame);
                    pacing.Submitted(started);
                    copyMs = copyMs.load() + queue.LastCopyMs(); ++published;
                    SetEvent(arrived.value);
                }
            } catch (...) { std::lock_guard lock(errorMutex); error = std::current_exception(); }
            done = true; SetEvent(arrived.value);
            if (apartment) winrt::uninit_apartment();
        });
    }
    void CheckError() { std::lock_guard lock(errorMutex); if (error) std::rethrow_exception(error); }
};
Capture::Capture(Gpu& gpu, CaptureSource source, unsigned backend, double fps) : impl(std::make_unique<Impl>(gpu, source, backend, fps)) {}
Capture::~Capture() { Stop(); }
void Capture::SetFrameRate(double fps) { if (impl) impl->pacing.SetRate(fps); }
void Capture::SetLowLatency(bool on) { if (impl) impl->pacing.SetLowLatency(on); }
void Capture::StopProducer() {
    if (!impl) return;
    impl->worker.request_stop();
    if (impl->worker.joinable()) impl->worker.join();
}
void Capture::Stop() {
    if (!impl) return;
    StopProducer();
    if (!impl->queue.CopiesComplete()) {
        Log("Capture copy still pending: retaining its resources until process exit.");
        impl.release();
    } else impl.reset();
}
bool Capture::Next() {
    if (!impl) return false;
    impl->CheckError();
    receivedFrames = impl->received.load(); publishedFrames = impl->published.load(); copyMilliseconds = impl->copyMs.load();
    if (!impl->queue.Next()) return false;
    width = impl->queue.width; height = impl->queue.height; timestamp = impl->queue.timestamp;
    return true;
}
bool Capture::NextOfSize(UINT w, UINT h) {
    if (!impl) return false;
    impl->CheckError();
    if (!impl->queue.Next(w, h)) return false;
    receivedFrames = impl->received.load(); publishedFrames = impl->published.load(); copyMilliseconds = impl->copyMs.load();
    timestamp = impl->queue.timestamp;
    return true;
}
ID3D12Resource* Capture::Input() const { return impl ? impl->queue.Input() : nullptr; }
HANDLE Capture::Event() const { return impl ? impl->arrived.value : nullptr; }
bool Capture::Closed() const { if (!impl) return true; impl->CheckError(); return impl->done.load(); }
}
