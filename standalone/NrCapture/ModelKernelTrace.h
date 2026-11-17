#pragma once
#include <d3d12.h>
#include <filesystem>
#include <cstddef>
#include <span>

// Probe-only adapter of the earlier DlssNrNative NvAPI wrapper. Tracing forwards
// launches unchanged; full-body recurrence is a separate explicit experiment.
namespace ModelKernelTrace {
struct Report {
    size_t launchCalls = 0, kernels = 0;
    double gpuMs = 0, cpuEvaluateMs = 0, cpuLaunchMs = 0;
    bool perLaunchTiming = false;
};
// Install before constructing Runtime, so model function creation is observed.
// Detours and the NvAPI module are retained until this short-lived probe exits.
void Install();
// Optional probe-only layer between the tracer and the driver, such as the recovered NVFP4 hybrid. Set before
// Install(). It is given each NvAPI interface id with the driver's entry point and returns what callers should use.
void SetDriverLayer(void* (*wrap)(unsigned id, void* original));
// Probe-only JIT control: replace the two bookend kernels with an unchanged
// PTX module generated from the audited DLL. Configure after Runtime::Create.
void LoadBookendControl(ID3D12Device* device, const std::filesystem::path& fatbin);
void CheckBookendControl(unsigned evaluations);
// Feature boundary diagnostics: export pre-lift and post-wide FP16 tensors;
// optionally reuse the captured first pre-lift for subsequent identical frames.
void LoadBookendFeatures(ID3D12Device* device, const std::filesystem::path& fatbin,
                         unsigned width, unsigned height, bool importSelf);
void CaptureBookendFeatures(ID3D12GraphicsCommandList* commands);
void SaveBookendFeatures(const std::filesystem::path& stem);
enum class BookendBridge { AnchorControl, ColourDelta, Affine };
void ConfigureBookendBridge(ID3D12Device* device, std::span<const char> input,
                            const std::filesystem::path& weights, BookendBridge mode);
// false records metadata and only evaluation endpoints; true adds a timestamp
// after each native launch call, which can perturb chained-kernel overlap.
void Begin(ID3D12GraphicsCommandList* commands, ID3D12CommandQueue* queue, bool perLaunchTiming);
void End(ID3D12GraphicsCommandList* commands);
// Call only AFTER the command list's actual submission fence has completed.
// Writes <stem>.csv and <stem>.json. GPU interval attribution is approximate.
Report Collect(const std::filesystem::path& stem);
struct LoopReport {
    unsigned bodies = 1, replayedKernels = 0;
    unsigned long long latentBytes = 0;
};
// Experimental full-body recurrence, restricted to the audited local FP8 DLL.
// Call around EACH Evaluate; EndLoop must succeed BEFORE submitting its list.
// Repeats blocks 1..69, preserving the one pre-block and one final post-block.
void BeginLoop(ID3D12GraphicsCommandList* commands, ID3D12Device* device, unsigned bodies);
LoopReport EndLoop();
// Exception cleanup only; the caller must discard the unsubmitted command list.
void CancelLoop() noexcept;
// Repeat the complete ViT once, then blend z + alpha*(ViT(z)-z), where z is
// the first ViT's output. Encoder/decoder each run once. Alpha 0 is a byte-exact
// control; alpha -1 tests the earlier negative-residual reference experiment.
// Mutually exclusive with bookend, full-body and trace modes. Per-Evaluate API.
void BeginVitLoop(ID3D12GraphicsCommandList* commands, ID3D12Device* device, float alpha = -1.0f);
LoopReport EndVitLoop(); // bodies=2 denotes two ViT traversals; replayedKernels=42
void CancelVitLoop() noexcept;
}
