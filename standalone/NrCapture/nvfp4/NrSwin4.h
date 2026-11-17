#pragma once
#include <cstdint>
#include <filesystem>
#include <string>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;

// NVFP4 (4-bit) replacement of the fused Swin blocks of DLSS-NR Model A (RTX 50 series only): the encoder and decoder
// layers of widths 32, 64, 128 and 256, 44 launches per evaluation.
// Sits between the model runtime and the driver's CUDA-on-D3D12 entry points. A launch of one of the runtime's
// cc_tinlayout_fused_swin_* kernels is swapped for one launch of ours with the same grid and block shape, reading the
// same input and writing the same output and flags, so every block is switched on its own: by width, or one at a time.
// Kernel source: nvfp4/kernels/swin4.cu.
//
// Unknown launches keep the runtime's kernel. Optional Level4 records buffer the first three launches of D32/D64,
// then dispatch once at OutView using that evaluation's real final destination. Encoder4 does the same for E32/E64
// at the final Downsample, writing its stored skip and pooled view. An incomplete prefix is replayed.
namespace NrSwin4 {
struct Options {
    // The folder with swin4.txt, swin4.bin and the kernels swin4_<width>_<variant>.cubin.
    std::filesystem::path assets;
    unsigned widths = 0;            // which widths run our kernels: 1 = 32, 2 = 64, 4 = 128, 8 = 256
    uint64_t layers = ~0ull;        // bit n clear: the n-th fused block of the model (0..43, in launch order) stays the runtime's
    unsigned twoTerms = 7;          // width 32: a second 4-bit term for 1 = the block's input, 2 = the MLP's output, 4 = the
                                    // attention's context, 8 = the pooled map of the downsampling layer
    // Which set of weights. 0: swin4.bin. 1: swin4_trained.bin (the same layout, a second set of 4-bit
    // codes). 2..: swin4_set<n>.bin with swin4_set<n>.txt, a pass-in-one model of another shape:
    //   skip <block> ...                         inner blocks (0..43, launch order) that are not launched at all; the next
    //                                            block of the level reads what the last launched one wrote
    //   fp8 <block> offset <o> bytes <b>         a weight record in the runtime's own layout (swin4_set<n>_fp8.bin) for the
    //                                            runtime's own kernel of that block, where ours does not replace it;
    //                                            block 44 is the output block, 45 the input block
    //   level <plan> stages <n> hidden <h> offset <o> bytes <b>
    //                                            plan 1/5/9: Encoder4 E32/E64/E128; plan 137/143/147: Level4 D128/D64/D32;
    //                                            records in swin4_set<n>_level.bin (separate versioned magic);
    //                                            four (C32/64) or six (C128) native launches become one, independent of widths mask
    //   post stages <n> hidden <h> offset <o> bytes <b>
    //                                            PostPair4-v1 in swin4_set<n>_post.bin; original texture/compose interface
    //   pre stages <n> hidden <h> offset <o> bytes <b>
    //                                            PrePair4-v1 in swin4_set<n>_pre.bin; original feature/skip/pool interface
    //   thin pre|post offset <o> bytes <b>         Thin4 v3 records in the same sidecars, thin4_pre/post.cubin.
    //                                            PRE omits the full skip, so requires Thin POST. Common 184-byte POST only;
    //                                            an extended POST after Thin PRE fails closed. Missing lines change nothing.
    // A set whose files are missing is the first set. May change between evaluations.
    unsigned set = 0;
    // true (default): launches are taken in order from the evaluation's cc_cb_clear, each checked against the kernel
    // Model A has at that position, so the very first evaluation runs on these kernels and weights too. The layers are
    // paused for every evaluation that is not one of the 4-bit models, which all run Model A.
    // false: a launch is recognised by the address of the runtime's weight record, learnt from a complete evaluation;
    // the first evaluation after start-up then runs the runtime's own kernels and weights.
    bool byOrder = true;
};
// Give it each NvAPI interface id with the entry point to call below it; it returns what the caller should use.
// Transparent until Enable().
void* WrapNvapi(unsigned id, void* original);
void Enable(const Options& options);
void Disable();
// true: the evaluations that follow keep the runtime's kernels, with everything learnt and loaded kept; false again:
// as Enable() left it. For hosts that choose per evaluation. Call it between evaluations only.
void Pause(bool paused);
// The weight set (Options::set's numbering) of the evaluations that follow, for hosts that choose per evaluation;
// kDefaultSet: Options::set again. A set whose files are missing leaves Options::set in use. Call it between
// evaluations only.
inline constexpr unsigned kDefaultSet = ~0u;
// Returns the effective set after a harness override, for the other families.
unsigned UseSet(unsigned set);
// Whether every weight file of `set` (UseSet's numbering; kDefaultSet: Options::set) is loaded, so that an evaluation
// with that set runs no other weights in its place: its 4-bit records and its records for the runtime's own kernels.
// On false, *missing names the files (relative to the asset folder's parent).
bool SetComplete(unsigned set, std::string* missing = nullptr);
// Launches since the start that ran with the runtime's own weights while this layer was on and not paused: a block
// left to the runtime (a failure, a block that was not recognised, a set without a record for it). A host that asked
// for a model of ours compares the value before and after an evaluation: any increase means the evaluation is not
// wholly that model.
unsigned long long Faults();
// Optional harness override for successive UseSet calls (one per Processor evaluation). kDefaultSet entries keep
// that call's preset selection. Clear with (nullptr, 0) after the frame. Existing hosts need not call this.
void UseSetsForNextEvaluations(const unsigned* sets, unsigned count);
// Opt-in harness diagnostic. Read only after the host has drained the GPU.
void EnablePostProbe(bool enabled);
void WritePostProbe(ID3D12Device* device, const std::filesystem::path& path);
// Opt-in checker-only launch timing. Configure before InstallPassBorrow; collect after a bounded GPU drain.
void EnableLaunchTrace();
void BeginLaunchTrace(ID3D12GraphicsCommandList* commands, ID3D12CommandQueue* queue);
void EndLaunchTrace(ID3D12GraphicsCommandList* commands);
void WriteLaunchTrace(const std::filesystem::path& path);
// Destroy our kernels and weights on `device`. Call only when the GPU can no longer be running command lists recorded
// with them, or once the device is lost.
void ReleaseDevice(ID3D12Device* device);
// For hosts that know where an evaluation of the model ends.
void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept;
std::string Status(bool counters = false);
void SetLog(void (*log)(const char* line));
}
