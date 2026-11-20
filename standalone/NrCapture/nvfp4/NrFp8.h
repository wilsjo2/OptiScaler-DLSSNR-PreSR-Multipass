#pragma once
#include <filesystem>
#include <string>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;

// FP8 pass-in-one models of DLSS-NR Model A: NVIDIA's own FP8 kernels, every one of them reading a weight record of
// ours (RTX 40 and 50 series: no kernel of ours is involved).
// Sits between the model runtime and the driver's CUDA-on-D3D12 entry points, outside the 4-bit layers. For an
// evaluation with a set chosen (UseSet), the launches are counted from the evaluation's cc_cb_clear; every launch that
// reads a weight record is checked against the manifest (kernel name, parameter size) and launched with a copy of its
// parameter block whose record pointer is ours (and, on the output block, the history blend weight pointer). The
// runtime's weights are never read by such an evaluation when it completes; if anything does not match (another
// sequence of launches, a failure), the evaluation is counted in Faults() and the host must not show its picture.
//
// Assets: <folder>/fp8_set<n>.txt and fp8_set<n>.bin (n = 1..7):
//   evaluation <launches>                                    launches of the network, cc_cb_clear included (0); the
//                                                            runtime's launches after them are left as they are
//   launch <n> name <kernel> params <size> record <offset> [params <size> record <offset>] offset <o> bytes <b> [blend <p>]
//       the n-th launch, its parameter sizes with where the record pointer sits in each, the record in the .bin
//       (512-aligned), and where the output block's pointer to its history blend weight sits (our own copy of 0.7397)
namespace NrFp8 {
struct Options {
    std::filesystem::path assets;   // the folder with fp8_set<n>.*
};
// Give it each NvAPI interface id with the entry point to call below it; it returns what the caller should use.
// Transparent until Enable() and UseSet().
void* WrapNvapi(unsigned id, void* original);
// Reads every set's manifest in the folder and checks it against its .bin's size. Throws when there is none or one
// cannot be read. A set's records are read and uploaded when an evaluation first uses it (only sets in use take memory).
void Enable(const Options& options);
void Disable();
bool Enabled();
// The set of the evaluations that follow (0: none, the runtime's own records, nothing is counted). Call it between
// evaluations only.
void UseSet(unsigned set);
// Whether `set` was read (on false, *missing names the files).
bool SetComplete(unsigned set, std::string* missing = nullptr);
// Evaluations with a set that did not run wholly on it, and the reason of the last one.
unsigned long long Faults();
std::string Problem();
// Evaluations that ran wholly on a set.
unsigned long long Completed();
// For hosts that know where an evaluation of the model ends (otherwise it ends at the next cc_cb_clear).
void EndEvaluation(ID3D12GraphicsCommandList* commands) noexcept;
// Frees our weights on `device`. Call only when the GPU can no longer be running command lists recorded with them.
void ReleaseDevice(ID3D12Device* device);
std::string Status(bool counters = false);
void SetLog(void (*log)(const char* line));
}
