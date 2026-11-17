#pragma once
#include <cstdint>
#include <filesystem>
#include <string>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;

// NVFP4 v2 replacement of the eight transformer feed-forward pairs of the DLSS-NR model (RTX 50 series only).
// Sits between the model runtime and the driver's CUDA-on-D3D12 entry points: the runtime's own expand and
// contract launches are swallowed and three kernels of ours (pack, expand, contract) are recorded instead.
// Kernel source: nvfp4/kernels/ffn_v2.cu.
//
// Whatever goes wrong before a pair is replaced leaves the runtime's own kernels in use. A failure between the two
// halves of a pair cannot fall back (the runtime's contract would wait for an expand that never ran), so the rest of
// that evaluation's launches are dropped, one frame's result is lost, and the replacement stays off until the next
// Enable().
namespace NrFfnV2 {
struct Options {
    // modelFolders false: the folder with ffn_v2.cubin, ffn_v2.bin and ffn_v2.txt of the model in use (probes).
    // modelFolders true: the folder with ffn_v2.cubin and the folders modelA and modelB, each with ffn_v2.bin and
    // ffn_v2.txt. The model is then recognised from the runtime's launch sequence, per evaluation, and the loaded
    // nvngx_dlssnr.dll must be one of the builds the weights were taken from; anything else runs unchanged.
    std::filesystem::path assets;
    bool modelFolders = false;
    unsigned expandGroups = 8, expandHalves = 2;    // CTA shape of the expand kernel: 16-token groups (2, 4, 8) x 64-output halves (1, 2)
    unsigned contractGroups = 8, contractHalves = 2;
    unsigned stages = 2, stepsPerStage = 2;         // must match the STAGES and KSTEPS the cubin was built with
    // true: our kernels order themselves against each other and against the runtime's neighbours with flags, as the
    // runtime's own kernels do, so they overlap freely (needs the 8 x 2 shapes). false: null UAV barriers around
    // every launch of ours.
    bool chained = true;
    // Extra null UAV barriers on top of that, for experiments: 1 before pack, 2 after pack, 4 after expand, 8 after contract.
    unsigned barriers = 0;
    // Evaluations with fewer tokens (width x height / 3840) are left to the runtime's own kernels; 0: no limit.
    // Measured on an RTX 5090: the replacement is slower below about 1600 tokens, level up to 1920 and 18 us per
    // pair faster at 2160 (4K).
    unsigned minTokens = 0;
    // true: the weights of ffn_v2_trained.bin (the layout and scalars of ffn_v2.bin, other 4-bit codes) where that file
    // sits beside ffn_v2.bin. Both sets are loaded, so this switches at the next evaluation.
    bool trained = false;
};
// Give it each NvAPI interface id with the driver's entry point; it returns what the caller should use.
// Transparent until Enable().
void* WrapNvapi(unsigned id, void* original);
void Enable(const Options& options);        // throws on an unsupported kernel shape
void Disable();                             // takes effect at the next evaluation; a pair in progress is completed
// true: the evaluations that follow keep the runtime's kernels, with everything loaded kept; false again: as Enable()
// left it. For hosts that choose per evaluation; it takes effect at an evaluation's first feed-forward layer.
void Pause(bool paused);
// Destroy everything created on `device`: kernels, weights and buffers. Call only when the GPU can no longer be
// running command lists recorded with them (the host's queue has drained) or once the device is lost. Without it
// these objects keep the device alive.
void ReleaseDevice(ID3D12Device* device);
// Optional, for hosts that know where an evaluation of the model ends: forget what `commands` was in the middle of.
// Without it an evaluation ends where the next begins (the runtime's cc_cb_clear launch), which leaves a kernel
// that some layouts launch before it to the previous evaluation.
void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept;
std::string Status(bool counters = false);  // one line for a status display; counters: pairs replaced so far
void SetLog(void (*log)(const char* line));   // where status changes are reported; standard error by default
bool Failed();                              // a pair failed half-way since the last Enable(); see above
// For a layer above this one that leaves launches of the runtime out (the blocks a pass-in-one model does without), so
// that this layer still knows where it is in the model. RuntimeLaunches: the number of launches the runtime has made in
// this evaluation, the one passed down next included (the model is recognised by that number at the transformer's
// first expand). SkipPair: the runtime's next feed-forward pair was left out; the pair after it is the next block's.
void RuntimeLaunches(ID3D12GraphicsCommandList* commands, unsigned launches, unsigned sourceTokens = 0);
// The numbered set of the evaluations recorded from now on (a pass-in-one model: 2..7): its own pairs, when the asset
// folder has <model folder>/ffn_v2_set<n>.bin (the layout of ffn_v2.bin). kDefaultSet, or a set without that file:
// the first set or the trained one, as Options::trained says.
constexpr unsigned kDefaultSet = ~0u;
void UseSet(unsigned set);
// Whether the file of `set` for Model A is in the asset folder (kDefaultSet: ffn_v2_trained.bin where Options::trained
// asks for it), so that an evaluation with that set runs no other weights in its place. On false, *missing names it.
bool SetComplete(unsigned set, std::string* missing = nullptr);
// Since the start: replacements that failed (the kernels of the runtime carried on, or the evaluation was dropped),
// contracts left to the runtime, and pairs run with other weights than the set asked for. A host that asked for a
// model of ours compares the value before and after an evaluation: any increase means it is not wholly that model.
unsigned long long Faults();
// Buffers are kept per frame size (token count), and recorded command lists refer to them. For hosts that can tell
// when the GPU has finished: RetireIdleSizes sets aside the buffers of every size of `device` beyond the `keep` most
// recently used that no evaluation used among the last `recent` ones, and returns a ticket (0: nothing set aside).
// Call it between evaluations. FreeRetired(device, ticket) frees what that call and earlier ones set aside: call it
// only once every command list recorded before RetireIdleSizes returned has finished on the GPU. Sizes set aside count
// towards the limit of sizes held until they are freed. Without these calls, sizes stay until ReleaseDevice().
uint64_t RetireIdleSizes(ID3D12Device* device, unsigned keep, unsigned recent);
void FreeRetired(ID3D12Device* device, uint64_t ticket);
void SkipPair(ID3D12GraphicsCommandList* commands);
}
