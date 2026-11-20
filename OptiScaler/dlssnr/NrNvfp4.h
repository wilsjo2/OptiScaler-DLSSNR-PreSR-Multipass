#pragma once
// The 4-bit (NVFP4) kernels and the pass-in-one weight sets of the Display Filter application
// (standalone/NrCapture/nvfp4: NrFfnV2, NrSwin4, NrNet4) for the D3D12 NR path. They sit below pass borrowing in the
// NvAPI hook and replace launches of the NR runtime's Model A with our own kernels; a pass-in-one model is the same kernels
// with a weight set of its own. Self-contained (no precompiled header).
//
// A pass asks for them with a model choice above the runtime's own presets (DlssNr Preset / pass preset):
//   4  Model A with the 4-bit kernels (fused blocks 256 and 128, split blocks, transformer, feed-forward; trained weights)
//   15 NVFP4 3xPass-in-one (weight set 12)
//   16 NVFP4 2xPass-in-one (weight set 13)
//   17 FP8 2xPass-in-one, 18 FP8 3xPass-in-one: NVIDIA's own FP8 kernels with weight records of ours (NrFp8; RTX 40 and
//      50 series; files OptiScaler/nvfp4/fp8/fp8_set1.* and fp8_set2.*). No 4-bit kernel runs in such a pass.
//   19 FP8 2xPass-in-one v2, 20 FP8 3xPass-in-one v2: the same with their v2 weights (fp8_set3.*, fp8_set4.*).
// Any other value from 4 up (5..14 were models that have been removed) is refused, never run as another model.
// The runtime itself is given preset 1 (Model A) for these, because our kernels replace its launches one by one.
// Assets: <OptiScaler.dll's folder>/OptiScaler/nvfp4, laid out as the Display Filter package's nvfp4 folder.
//
// NO FALLBACK. A pass that asks for one of these and does not get it (files missing, another runtime build, the Vulkan
// path, a layer that did not recognise the model) is never shown as a plain Model A pass: OursRan() says so, the
// caller leaves the picture unprocessed, and Problem() gives the reason for the menu and the log.
#include <filesystem>
#include <string>

struct ID3D12GraphicsCommandList;
struct ID3D12Device;

namespace DlssNr::Nvfp4
{
// The numbers are the Display Filter application's (standalone/NrCapture/Common.h).
inline constexpr unsigned ModelANvfp4 = 4,
                          ModelTwoTriple = 15, // one pass with the look of three
                          ModelTwoDouble = 16; // one pass with the look of two
inline constexpr unsigned ModelFp8Double = 17, // FP8: one pass with the look of two
                          ModelFp8Triple = 18, // FP8: one pass with the look of three
                          ModelFp8Double2 = 19, // the same, v2 weights
                          ModelFp8Triple2 = 20;
constexpr bool IsPassInOne(unsigned preset) { return preset == ModelTwoTriple || preset == ModelTwoDouble; }
constexpr bool IsFp8(unsigned preset) { return preset >= ModelFp8Double && preset <= ModelFp8Triple2; }
constexpr unsigned Fp8Set(unsigned preset) { return IsFp8(preset) ? preset - 16 : 0; } // fp8/fp8_set<n>
constexpr bool IsKnown(unsigned preset) { return preset == ModelANvfp4 || IsPassInOne(preset) || IsFp8(preset); }
// Every value from 4 up asks for a 4-bit model: one this build does not have fails visibly (BeginEvaluation).
constexpr bool IsNvfp4(unsigned preset) { return preset >= ModelANvfp4; }
constexpr unsigned WeightSet(unsigned preset) { return IsPassInOne(preset) ? preset - 3 : 0; } // 0: the trained 4-bit codes

void SetLog(void (*log)(const char* line));
// For the NvAPI query hook, inside pass borrowing's wrap: transparent until Configure() switches the layers on, and
// outside the evaluations that ask for them.
void* WrapNvapi(unsigned id, void* original);
// Before a pass's feature is created and before a frame's evaluations: whether any pass wants a 4-bit model. Loads the
// layers the first time one does; they stay loaded (and idle) afterwards.
void Configure(bool wanted, const std::filesystem::path& assets);
// Around every evaluation of the NR model recorded on `list`, whatever the pass's model is.
void BeginEvaluation(unsigned requestedPreset);
void EndEvaluation(ID3D12GraphicsCommandList* list);
// After EndEvaluation() of an evaluation the runtime really made: false when the pass asked for a 4-bit model and our
// kernels did not run it, so its output is the runtime's own model and must not be shown. `whole`: no part of the
// evaluation was borrowed from another pass (a borrowed part runs no kernels of its own). True for other passes.
bool OursRan(bool whole);
// A path that cannot run a 4-bit model at all (Vulkan) says so here instead of running the runtime's own.
void Refuse(const char* reason);
// Why a pass that asks for a 4-bit model is not being applied; empty while it runs or while it is still starting.
std::string Problem();
// The last evaluation that asked for a 4-bit model ran our kernels.
bool Running();
// The feed-forward buffers are kept per picture size. After a frame's evaluations (`evaluations` of them): sets aside
// those of sizes no longer in use and returns a ticket (0: nothing was set aside). Call FreeRetired(device, ticket) once
// every command list recorded so far has finished on the GPU; until then they count towards the limit of sizes held.
unsigned long long RetireIdle(ID3D12Device* device, unsigned evaluations);
void FreeRetired(ID3D12Device* device, unsigned long long ticket);
// Destroys our kernels, weights and buffers on the device; they load again when a pass next asks. Only once the
// device's queue has drained and every NR feature created on it is gone, or the device is lost.
void Release(ID3D12Device* device);
std::string Status();
} // namespace DlssNr::Nvfp4
