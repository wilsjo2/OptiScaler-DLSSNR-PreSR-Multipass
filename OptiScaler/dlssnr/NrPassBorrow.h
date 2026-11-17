#pragma once

// Pass borrowing for multi-pass DLSS-NR.
//
// A later network pass skips the network blocks that an earlier pass of the same frame already ran at the same size,
// and takes that pass's tensor at the cut instead ("borrow pass 1's transformer", "borrow everything at 1/x size and
// below"). It works by intercepting the NR runtime's NvAPI CUDA kernel launches (D3D12 only):
//   - the first evaluation of a frame runs unchanged and is recorded as the template;
//   - an evaluation that produces a cut a later evaluation will borrow copies the handoff tensor into a snapshot
//     buffer owned by this module, right after the producing kernel;
//   - a borrowing evaluation checks every launch against the template, skips the cut's launches (returning
//     NVAPI_OK without launching), and copies the snapshot into its own consumer input before the consumer launch.
// Anything unexpected before skipping starts leaves that evaluation unchanged. Anything unexpected after skipping has
// started drops the remaining launches of that evaluation (a chained kernel must never wait on counters that a
// skipped kernel would have published) and latches borrowing off until Reset().
//
// Self-contained: Windows SDK, D3D12, NvAPI and Detours headers only, no precompiled header. Shared by OptiScaler
// (dlssnr/) and the standalone OptiScalerNR app (standalone/NrCapture).

#include <cstddef>
#include <cstdint>

struct ID3D12GraphicsCommandList;
struct ID3D12Object;

namespace NrPassBorrow
{

// Ordered shallowest to deepest; each cut's skipped range contains every shallower cut's range.
// Transformer: the ViT blocks between the two repacks. BelowN: every block at 1/N of the frame size and smaller.
enum class Cut : uint8_t
{
    None = 0,
    Transformer = 1,
    Below32 = 2,
    Below16 = 3,
    Below8 = 4,
    Below4 = 5,
    Below2 = 6,
};

// Map the per-pass tick-boxes to a cut. borrowBelow wins over borrowTransformer; level is 2, 4, 8, 16 or 32
// (other values clamp to the nearest). Neither ticked gives Cut::None.
Cut ChooseCut(bool borrowTransformer, bool borrowBelow, unsigned level);
const char* CutName(Cut cut); // "none", "transformer", "1/32 and below", ...

using LogFn = void (*)(int level, const char* message); // level: 0 info, 1 warning, 2 error
void SetLog(LogFn fn);

// Standalone only: Detour nvapi64!nvapi_QueryInterface so the NR runtime's kernel entry points are wrapped.
// Call before the NR runtime resolves its NvAPI functions (before the first NR runtime/feature is created).
// Returns false and sets *error (static string) when nvapi64 or Detours fails; borrowing is then unavailable and
// nothing else changes. Idempotent.
bool InstallQueryInterfaceHook(const char** error = nullptr);

// Standalone only: a layer between this module and the driver. It is given every entry point the Detoured
// nvapi_QueryInterface returns (interface id, the driver's function) and returns what this module should wrap and call
// instead. This module still sees the runtime's launches unchanged; the layer sees those this module lets through.
// Set it before the NR runtime resolves its NvAPI functions.
using InnerLayerFn = void* (*)(unsigned id, void* original);
void SetInnerLayer(InnerLayerFn layer);

// OptiScaler: call from its existing nvapi_QueryInterface hook with the function pointer the real QueryInterface
// returned. Returns this module's wrapper for CreateCuFunction, DestroyCuFunction, DestroyCuModule and
// LaunchCuKernelChain; any other id, or a null original, comes back unchanged.
void* WrapInterface(uint32_t id, void* original);

// One frame's chain of NGX evaluations, all recorded on `list` by the calling thread.
// cuts[i] is the cut for the i-th network evaluation of the frame. Passes that re-apply instead of evaluating are not
// counted. cuts[0] must be Cut::None. evaluations may be 1 (nothing to borrow; the module stays transparent).
void BeginFrame(ID3D12GraphicsCommandList* list, const Cut* cuts, unsigned evaluations);
void BeginEvaluation(unsigned evaluation); // immediately before that evaluation's NGX evaluate call
enum class Outcome : uint8_t
{
    Native,   // ran unchanged (also when its cut was None, or borrowing was refused before skipping started)
    Borrowed, // skipped its cut and used the snapshot
    Dropped,  // failed after skipping started: its remaining launches were dropped; borrowing is now latched off
};
Outcome EndEvaluation(); // immediately after it
void EndFrame();

// Snapshot buffers are replaced when a cut's tensor grows (a resolution change) and when the device changes. Command
// lists recorded earlier may still refer to the old ones, so they must outlive those lists' GPU work.
// A host that tracks GPU completion calls this after each frame's evaluations (and may call it every frame): it hands
// over up to `capacity` such objects, one reference each, and returns how many (call again while it returns
// `capacity`). The host releases each once every command list recorded so far has finished on the GPU. After the
// first call this module never frees them itself. Without it they are freed a fixed number of borrowing frames later,
// which suits only a host that waits for the GPU each frame.
unsigned TakeRetired(ID3D12Object** objects, unsigned capacity);

// Clear the latched-off state and the recorded template (call when a borrow option changes or the NR features are
// recreated). Snapshot buffers are kept and released later, once the GPU can no longer be using them.
void Reset();

// Short user-facing status, e.g. "Borrowing: transformer (Model A, 310.9.16)", "Off: runtime not audited",
// "Off: hook not installed". Copies into buf (always terminated). Thread-safe.
void Status(char* buf, size_t size);

} // namespace NrPassBorrow
