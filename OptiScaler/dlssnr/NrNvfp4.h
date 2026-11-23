#pragma once
// Model selection and evaluation for the D3D12 NR path. The game and Display Filter share the native pass host
// for NVFP4 and INT8. FP8 on the game's signed runtime binds the selected weight set to its command list.
//
// A pass asks for them with a model choice above the runtime's own presets (DlssNr Preset / pass preset):
//   4  Model A with the 4-bit kernels (fused blocks 256 and 128, split blocks, transformer, feed-forward; trained
//   weights) 15 NVFP4 3xPass-in-one (weight set 12) 16 NVFP4 2xPass-in-one (weight set 13) 17 FP8 2xPass-in-one, 18 FP8
//   3xPass-in-one: NVIDIA's own FP8 kernels with weight records of ours (NrFp8; RTX 40 and
//      50 series; files OptiScaler/nvfp4/fp8/fp8_set1.* and fp8_set2.*). No 4-bit kernel runs in such a pass.
//   19 FP8 2xPass-in-one v2, 20 FP8 3xPass-in-one v2: the same with their v2 weights (fp8_set3.*, fp8_set4.*).
//   21 NVFP4 2xPass-in-one 4K (weight set 10), 22 NVFP4 3xPass-in-one 4K (weight set 11),
//   23 FP8 2xPass-in-one 4K (fp8_set5.*), 24 FP8 3xPass-in-one 4K (fp8_set6.*): the same models with their 4K weights.
//   25 Original - INT8, 26 INT8 2xPass-in-one: Model A with INT8 weights and INT8 kernels of ours,
//      run wholly on the own pass host (standalone/NrCapture/nvfp4/NrHost: RTX 30, 40 and 50 series; no launch of the
//      NR runtime at all, so such a pass creates no NGX feature). Files OptiScaler/nvfp4/int8.
// Any other value from 4 up (5..14 were models that have been removed) is refused, never run as another model.
// Game FP8 selects signed Model A and substitutes the selected FP8 weight records.
// Assets: <OptiScaler.dll's folder>/OptiScaler/nvfp4, laid out as the Display Filter package's nvfp4 folder.
//
// NO FALLBACK. A pass that asks for one of these and does not get it (files missing, another runtime build, the Vulkan
// path, a layer that did not recognise the model) is never shown as a plain Model A pass: OursRan() says so, the
// caller leaves the picture unprocessed, and Problem() gives the reason for the menu and the log.
#include <filesystem>
#include <memory>
#include <string>

struct ID3D12GraphicsCommandList;
struct ID3D12Device;
struct ID3D12Resource;

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
inline constexpr unsigned ModelTwoDouble4k = 21, // NVFP4 2xPass-in-one with its 4K weights
                          ModelTwoTriple4k = 22, // NVFP4 3xPass-in-one with its 4K weights
                          ModelFp8Double4k = 23, // FP8 2xPass-in-one with its 4K weights
                          ModelFp8Triple4k = 24; // FP8 3xPass-in-one with its 4K weights
inline constexpr unsigned ModelInt8 = 25,       // Model A as INT8 kernels and weights of ours
                          ModelInt8Double = 26; // INT8: one pass with the look of two
constexpr bool IsInt8(unsigned preset) { return preset >= ModelInt8 && preset <= ModelInt8Double; }
constexpr unsigned Int8Set(unsigned preset) { return IsInt8(preset) ? preset - (ModelInt8 - 1) : 0; } // int8/int8_set<n>
constexpr bool IsPassInOne(unsigned preset)
{
    return preset == ModelTwoTriple || preset == ModelTwoDouble || preset == ModelTwoDouble4k || preset == ModelTwoTriple4k;
}
constexpr bool IsFp8(unsigned preset)
{
    return (preset >= ModelFp8Double && preset <= ModelFp8Triple2) || preset == ModelFp8Double4k || preset == ModelFp8Triple4k;
}
constexpr unsigned Fp8Set(unsigned preset) // fp8/fp8_set<n>
{
    return preset == ModelFp8Double4k ? 5 : preset == ModelFp8Triple4k ? 6 : IsFp8(preset) ? preset - 16 : 0;
}
// One version of each model. The numbers 15..20 were earlier weights of the four pass-in-one models: a saved one
// means the model as it is now.
constexpr unsigned Current(unsigned preset)
{
    return preset == ModelTwoDouble ? ModelTwoDouble4k
           : preset == ModelTwoTriple ? ModelTwoTriple4k
           : preset == ModelFp8Double || preset == ModelFp8Double2 ? ModelFp8Double4k
           : preset == ModelFp8Triple || preset == ModelFp8Triple2 ? ModelFp8Triple4k
           : preset;
}
constexpr bool UsesNativeHost(unsigned preset)
{
    return IsInt8(preset) || preset == ModelANvfp4 || IsPassInOne(preset);
}
constexpr bool IsKnown(unsigned preset)
{
    return preset == ModelANvfp4 || IsPassInOne(preset) || IsFp8(preset) || IsInt8(preset);
}
// Every value from 4 up asks for a model of ours: one this build does not have fails visibly (BeginEvaluation).
constexpr bool IsNvfp4(unsigned preset) { return preset >= ModelANvfp4; }
// The NR model list as the menus show it, sorted by the precision of the weights: FP8 (NVIDIA's own models and the
// FP8 pass-in-one models), INT8, NVFP4 (4-bit).
struct ModelEntry
{
    unsigned preset;
    const char* name;
};
inline constexpr ModelEntry kModels[] = {
    { 0, "Automatic" },
    { 1, "FP8 Original" },
    { 2, "FP8 Optimized" },
    { ModelFp8Double4k, "FP8 2xPass-in-one" },
    { ModelFp8Triple4k, "FP8 3xPass-in-one" },
    { ModelInt8, "INT8 Original" },
    { ModelInt8Double, "INT8 2xPass-in-one" },
    { ModelANvfp4, "NVFP4 Original (4-bit kernels)" },
    { ModelTwoDouble4k, "NVFP4 2xPass-in-one" },
    { ModelTwoTriple4k, "NVFP4 3xPass-in-one" },
};
constexpr const char* ModelName(unsigned preset)
{
    for (const auto& model : kModels)
        if (model.preset == Current(preset))
            return model.name;
    return "Unavailable preset";
}
constexpr unsigned WeightSet(unsigned preset) // 0: the trained 4-bit codes
{
    return preset == ModelTwoDouble4k ? 10 : preset == ModelTwoTriple4k ? 11 : IsPassInOne(preset) ? preset - 3 : 0;
}

void SetLog(void (*log)(const char* line));
// For the NvAPI query hook, inside pass borrowing's wrap: transparent until Configure() switches the layers on, and
// outside the evaluations that ask for them.
void* WrapNvapi(unsigned id, void* original);
// Before a pass's feature is created and before a frame's evaluations: whether any pass wants a 4-bit model. Loads the
// layers the first time one does; they stay loaded (and idle) afterwards.
void Configure(bool wanted, const std::filesystem::path& assets);
// Around every evaluation of the NR model recorded on `list`, whatever the pass's model is.
void BeginEvaluation(unsigned requestedPreset, ID3D12GraphicsCommandList* commands);
void EndEvaluation(ID3D12GraphicsCommandList* list);
// True when the requested model completed. Borrowed work is excluded from the expected per-pass layer counts.
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

// INT8 and NVFP4 have one native host per pass: its kernels and weights (shared by every host on the device), its
// buffers and the pass's history, for one model at one picture size.
struct HostPass
{
    ID3D12Resource* colour = nullptr;
    ID3D12Resource* motion = nullptr; // null: no usable motion vectors, the frame is run as a first frame
    ID3D12Resource* output = nullptr;
    unsigned style = 0;
    float intensity = 1.0f, structure = 1.0f, tone = 0.0f, skin = -1.0f;
    bool autoMask = true;
    float mvScaleX = 1.0f, mvScaleY = 1.0f;
    bool reset = false;
};
// `assets`: the folder Configure() takes (<OptiScaler.dll's folder>/OptiScaler/nvfp4). Null when the host cannot be
// made (files missing, a GPU that is not an RTX 30, 40 or 50 series one): Problem() says why. The owner keeps it until
// the GPU has finished every command list an evaluation was recorded on.
std::shared_ptr<void> CreateHost(ID3D12Device* device, unsigned preset, unsigned width, unsigned height,
                                 const std::filesystem::path& assets);
// Records one whole evaluation between BeginEvaluation() and EndEvaluation(). False when it could not: OursRan() then
// answers false and Problem() says why.
bool RunHost(const std::shared_ptr<void>& host, ID3D12GraphicsCommandList* list, const HostPass& pass);
// The driver's own nvapi_QueryInterface (the hook's trampoline), so the host's launches pass none of the layers.
void SetDriverQuery(void* query);

// A model comparison (the Compare control with a model on its other side): which model the last frame showed there
// (0: none) and, when one was asked for and not shown, why.
void SetComparison(unsigned shownModel, const char* note);
unsigned ComparisonShown();
std::string ComparisonNote();
} // namespace DlssNr::Nvfp4
