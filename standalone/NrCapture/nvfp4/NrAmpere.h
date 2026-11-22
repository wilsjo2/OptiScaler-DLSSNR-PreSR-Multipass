#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;

// The "Ampere set": every one of Model A's 152 weighted launches run by a kernel of ours that RTX 30 (sm_86) can run
// (FP16 tensor-core arithmetic that reproduces NVIDIA's FP8 kernels), tested inside NVIDIA's runtime by launch
// replacement. NVIDIA's runtime itself only runs on RTX 50 (its own kernels are sm_120 only), so on this path the sm_120
// build of the same sources is what runs on the lab's RTX 5090; real RTX 30 runs need an own pass host later.
//
// Sits outermost in the NvAPI wrap chain (outside NrFp8 and the 4-bit layers, which are paused / set 0 for an
// evaluation that uses it). For an evaluation with a set chosen (UseSet), launches are counted from the evaluation's
// cc_cb_clear like NrFp8. Every weighted launch (one listed in fp8_set<n>.txt, NrFp8's manifest, whose records are used
// unchanged) is checked (kernel name, parameter size) and launched with OUR kernel from the family manifests, with a copy
// of the runtime's parameter block in which only the record pointer (and, on the output block, the history blend
// weight pointer) is ours, the runtime's grid and block unless the manifest says otherwise, and the manifest's dynamic
// shared memory (at most 101376 bytes, the sm_86 per-block maximum).
//
// No silent fallback: an evaluation in which any weighted launch did not run our kernel is a fault (Faults()) and the
// host must not show its picture. The runtime's own launch is still issued in that case (a launch left out would leave
// the kernels after it waiting on flags that never come). Development switches can leave chosen launches to NVIDIA's
// kernel (with our record): such an evaluation is counted in Partial(), never in Completed(), and the log says so.
//
// Manifests: every amp_*_manifest.txt in Options::manifests (the four families write amp_fused_ / amp_prepost_ /
// amp_split_ / amp_vit_manifest.txt). Lines ('#' starts a comment; words in any order after the first two):
//   family <name>                                         informational
//   kernel <id> file <cubin> entry <symbol> [shared <bytes>] [grid <g>] [block <b>]
//       a kernel of ours; <id> is any word, unique in its file; <cubin> is a base name such as amp_fused_w256_chained
//       (the chosen build's '.sm120.cubin' / '.sm86.cubin' is appended; a name already ending in .sm120.cubin or
//       .sm86.cubin has that ending replaced), relative to Options::assets
//   launch <numbers> kernel <id> | file <cubin> entry <symbol> | nvidia
//                    [name <runtime kernel>] [params <bytes>] [shared <bytes>] [grid <g>] [block <b>]
//       the runtime launch numbers (counted from cc_cb_clear = 0, as in fp8_set<n>.txt and sim/fp8_launches.json):
//       '12', '3,5,9-11'; a number may appear in one line of one manifest only, except that a second line of the same
//       kind and name may give another kernel for another parameter size (both lines with 'params', sizes different:
//       the input block is 272 bytes on runtime 310.9.16 and 264 on 310.8); the line whose params equal the runtime's
//       block is used. A launch whose block differs from its line's 'params' is a fault. 'nvidia' launches the runtime's own
//       kernel with our record (the layer's dry run; refused unless Options::dry). name/params, when given, must agree
//       with every fp8 set. shared on the launch line wins over the kernel's (default 0); 'shared =' takes the runtime
//       launch's own amount (for loopback tests with NVIDIA's own cubins). A launch without a record (the repacks at 57
//       and 98) may also be given a kernel of ours, with 'name <runtime kernel>' required: it gets the runtime's
//       parameter block unchanged and is optional (not counted among the weighted launches).
//   <g>/<b>: three dimensions 'X Y Z' or 'XxYxZ'; each is a number, '=' (the runtime's), '*k' or '/k' (the runtime's
//       times k, or divided by k rounded up). Default: the runtime's grid and block.
namespace NrAmpere {
enum class Build { Sm120, Sm86 };       // which cubins: <base>.sm120.cubin (RTX 50, lab tests) or <base>.sm86.cubin (RTX 30/40)
struct Options {
    std::filesystem::path records;      // fp8_set<n>.txt / .bin (NrFp8's assets)
    std::filesystem::path assets;       // the cubins
    std::filesystem::path manifests;    // amp_*_manifest.txt (empty: assets)
    Build build = Build::Sm120;
    // Development only (NR_AMP_SLOTS): launches left to NVIDIA's kernel with our record, bit n = launch n; with
    // leaveMissing every weighted launch that has no manifest entry is left. Evaluations count as Partial.
    std::array<uint64_t, 4> leave{};
    bool leaveMissing = false;
    bool dry = false;                   // manifest lines 'nvidia' allowed (NR_AMP_DRY=1)
    std::filesystem::path report;       // NR_AMP_REPORT: Status(true) written here after every evaluation (lab)
};
// Parses the development switches: NR_AMP_SLOTS (hex number over launch numbers, or 'missing'), NR_AMP_DRY,
// NR_AMP_ARCH (sm120 / sm86), NR_AMP_REPORT. Throws on a malformed value.
void FromEnvironment(Options& options);
// Give it each NvAPI interface id with the entry point to call below it; it returns what the caller should use.
void* WrapNvapi(unsigned id, void* original);
// Reads every manifest and every fp8 set's manifest and checks them against each other. Throws when anything is missing
// or does not agree. Records are uploaded and kernels loaded when an evaluation first needs them.
void Enable(const Options& options);
void Disable();
bool Enabled();
// Whether the chosen build can run on the device's GPU (NvAPI GPU_GetArchInfo of the physical GPU whose LUID is the
// device's adapter): "" when it can, otherwise the reason. Cached per device. Call before an evaluation (not from
// inside an NvAPI hook); an evaluation on a device that was not checked, or failed the check, is a fault.
std::string CheckArchitecture(ID3D12Device* device);
// The set of the evaluations that follow (0: none). Between evaluations only.
void UseSet(unsigned set);
bool SetComplete(unsigned set, std::string* missing = nullptr);
unsigned long long Faults();            // evaluations that must not be shown
unsigned long long Completed();         // evaluations in which every weighted launch ran our kernel
unsigned long long Partial();           // evaluations with launches left to NVIDIA by the development switches
std::string Problem();
void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept;
// Frees our records and kernels on `device`. Only when the GPU can no longer run command lists recorded with them.
void ReleaseDevice(ID3D12Device* device);
std::string Status(bool counters = false);
void SetLog(void (*log)(const char* line));
}
