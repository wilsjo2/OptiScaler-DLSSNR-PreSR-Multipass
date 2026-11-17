#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;

// NVFP4 (4-bit) replacement of the rest of DLSS-NR Model A's pass (RTX 50 series only): what NrSwin4 (fused blocks)
// and NrFfnV2 (transformer feed-forward) do not cover. Three families of kernels, each in groups that are switched
// on their own:
//   split    the width-512 split Swin blocks and the decoder-input projection    ffwd, ffwd_proj, qkv, proj, head, dec
//   vit      the transformer's QKV, attention and projection layers              qkv, attention, projection
//   prepost  the input and the output block                                      pre, post
// Sits between the model runtime and the driver's CUDA-on-D3D12 entry points, above NrSwin4 and NrFfnV2. A launch of
// the runtime is swapped for one launch of ours with the same grid and block shape, on the runtime's own buffers and
// flags, with our weight record. Everything about a launch (which runtime kernel, which kernel of ours, how our
// parameter block is made from the runtime's) comes from the family's manifest, <assets>/<family>4/<family>4.txt.
//
// Whatever is not recognised or goes wrong leaves that launch to the runtime's own kernel. The one exception is a
// pair of our kernels that hand data to each other in a format of their own (the manifest's "with"): if the second
// of such a pair cannot be launched after the first was, that evaluation's picture is wrong; it is logged.
namespace NrNet4 {
enum Family : unsigned { Split, Vit, PrePost, kFamilies };
constexpr unsigned kMaxGroups = 8, kMaxSlots = 128, kMaxSets = 14;
inline constexpr unsigned kDefaultSet = ~0u;
struct FamilyOptions {
    unsigned groups = 0;                            // bit n: the family's n-th group (order as listed above) runs our kernels
    std::array<uint64_t, 2> slots{~0ull, ~0ull};    // bit n clear: the family's n-th launch of an evaluation stays the runtime's
    std::array<int64_t, kMaxGroups> options{-1, -1, -1, -1, -1, -1, -1, -1};    // a group's options word; negative: the manifest's default
    // true: the records of <family>4_trained.bin (the same layout as <family>4.bin, a second set of 4-bit
    // codes) where the folder has that file; otherwise <family>4.bin. May change between evaluations.
    bool trained = false;
};
struct Options {
    // The folder that holds the folders split4, vit4 and prepost4.
    std::filesystem::path assets;
    std::array<FamilyOptions, kFamilies> family;
    // true (default): launches are taken in order from the evaluation's cc_cb_clear, each checked against the kernel
    // Model A has at that position, so the very first evaluation runs on these kernels and weights too. The layers are
    // paused for every evaluation that is not one of the 4-bit models, which all run Model A.
    // false: a launch is recognised by the address of the runtime's weight record, learnt from a complete evaluation;
    // the first evaluation after start-up then runs the runtime's own kernels and weights.
    bool byOrder = true;
    bool Any() const { return family[Split].groups || family[Vit].groups || family[PrePost].groups; }
};
// Settings as text, for a settings file, a command line or an environment variable. Items separated by spaces or
// commas:   <family>=<mask | all | group+group...>   <family>.<group>=<options word>   <family>.slots=<hex>
// <family>.trained=<0 | 1>
// for example "split=all vit=qkv+projection vit.attention=1 prepost=3". Throws std::runtime_error on anything else.
void Parse(const std::string& text, Options& options);
// Give it each NvAPI interface id with the entry point to call below it; it returns what the caller should use.
// Transparent until Enable().
void* WrapNvapi(unsigned id, void* original);
// A family whose assets cannot be read stays off and Status() says why; throws only if none of the wanted ones loaded.
void Enable(const Options& options);
void Disable();
// true: the evaluations that follow keep the runtime's kernels, with everything learnt and loaded kept; false again:
// as Enable() left it. For hosts that choose per evaluation. Call it between evaluations only.
void Pause(bool paused);
// Destroy our kernels and weights on `device`. Call only when the GPU can no longer be running command lists recorded
// with them, or once the device is lost.
void ReleaseDevice(ID3D12Device* device);
// Numbered sets, for pass-in-one models (sets 2..7; a set exists when the asset folder has one of its files):
//   <assets>/net4_set<n>.txt                  lines "skip split <block> ..." and "skip vit <block> ..." ('#' comments):
//                                           "pool vit 2 [min <tokens>]": pool the full padded transformer map by 2,
//                                           run its selected blocks on ceil(H/2)*ceil(W/2) tokens, unpool the learned
//                                           residual onto the head output. Requires vit4/vit_pool.cubin. Absent: off.
//                                           "side split a|b stages <1..8> hidden <64|128|256> file <basename>":
//                                           replace one complete split side with Side4 (record under split4/).
//                                           Cubin: split4/side4_a|b_h<hidden>_s<stages>.cubin. Missing line: unchanged.
//                                             whole blocks that are not launched. split: 0..15 in launch order (eight
//                                             before the transformer, eight after; the first and the last of each
//                                             eight stay), vit: 0..7 (the first and the last stay). The next launched
//                                             block reads the last launched block's result and waits for its flags.
//   <assets>/<family>4/<family>4_set<n>.bin   the family's records for the set (the layout of <family>4.bin)
// UseSet: the set of the evaluations that follow; kDefaultSet or a set that does not exist: none (FamilyOptions::trained
// decides the records, nothing is left out). Blocks are only left out of an evaluation whose launches were recognised.
void UseSet(unsigned set);
// Whether the split and transformer families (those switched on) have the records of `set` loaded (kDefaultSet: the
// trained records where FamilyOptions::trained asks for them), so that an evaluation with that set runs no other
// records in their place. On false, *missing names the file (relative to the asset folder).
bool SetComplete(unsigned set, std::string* missing = nullptr);
// Launches since the start that this layer was on for and did not run as asked: left to the runtime's own kernel and
// weights, or run by our kernel with other records than the set asked for. A host that asked for a model of ours
// compares the value before and after an evaluation: any increase means the evaluation is not wholly that model.
unsigned long long Faults();
// For hosts that know where an evaluation of the model ends.
void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept;
// Checker-only: after a fence, write full/base/final/unpooled stored bytes captured
// when NR_VIT_POOL_PROBE is set. No probe work/resources without that environment flag.
void WritePoolProbe(ID3D12Device* device, const std::filesystem::path& path);
std::string Status(bool counters = false);
void SetLog(void (*log)(const char* line));
}
