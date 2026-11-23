// Records a complete Model A pass through NvAPI CUDA-on-D3D12 for INT8, NVFP4 and the hosted FP8 models.
// The selected plan defines kernel arguments and intermediate lifetimes. Each host owns its model buffers,
// GPU texture handles and temporal history for one resolution and weight set.
//
// Kernel sets (NR_HOST_ASSETS, folders separated by ';', each optionally '@<code>'): manifests in the NrAmpere format
// (nvfp4/NrAmpere.h), launch numbers -1..156 (-1 = the leading copy and 156 = the colour step of the Natural and
// Cinematic styles). Codes: sm86 / sm120 (.sm86.cubin / .sm120.cubin), sm86ptx (.sm86.ptx.fatbin, the RTX 30 code
// compiled by the driver; lab only), nvidia (NVIDIA's own cubins as a lab reference; lab only, never shipped).
//
// Environment: NR_HOST=1 on / 0 off (unset: on for RTX 30 and 40, where NVIDIA's runtime refuses the GPU); NR_HOST_ASSETS
// kernel folders (default <exe>/nvfp4/host, where the plan also is); NR_HOST_CODE default code (default: the GPU's
// own: sm86 on RTX 30 and 40, sm120 on RTX 50); NR_HOST_PLAN plan file; NR_FP8_ASSETS records (as NrFp8);
// NR_HOST_LAB=1 allows lab-only codes; NR_HOST_BARRIERS=all a UAV barrier before every launch; NR_HOST_LAYOUT=3108 the
// 310.8 input-block layout (NVIDIA's 310.8 cubins only); NR_HOST_TRACE=<file> our parameter blocks per evaluation;
// NR_HOST_TRACE_RUNTIME=<file> the runtime's (for comparison); NR_HOST_REPORT=<file> counters after each evaluation.
// Every problem throws; the message ends "Nothing was shown." and nothing falls back to NVIDIA's runtime.
#pragma once
#include <d3d12.h>

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace NrHost {

struct Settings {
    unsigned style = 0;                 // 0 Standard, 1 Natural, 2 Cinematic
    float intensity = 1.f, structure = 1.f, tone = 0.f, skin = -1.f;
    bool autoMask = true;
    float mvScaleX = 1.f, mvScaleY = 1.f;
};

struct Options {
    bool enabled = false, fourBit = false, mixedOriginal = false;
    std::filesystem::path plan, records;
    std::string recordsName = "fp8_set";   // <records>/<recordsName><set>.txt and .bin
    std::vector<std::pair<std::filesystem::path, std::string>> kernels;   // folder, code ("" = the default code)
    std::string code;                   // default code; "" = by GPU architecture
    bool lab = false, barriersAll = false;
    unsigned layout = 3109;
    std::filesystem::path trace, report;
    std::filesystem::path timing;       // NR_HOST_TIMING: GPU microseconds per launch (lab)
    bool callerTracksGpuLifetime = false; // Caller holds the host until its command lists can no longer execute.
};
// NR_HOST_* keys; throws on malformed values. `exeDir` gives the defaults (<exe>/nvfp4/host/...).
Options FromEnvironment(const std::filesystem::path& exeDir);
// NR_HOST=1 on, NR_HOST=0 off; unset: on for RTX 30 and 40 (NVIDIA's runtime refuses them), off elsewhere.
bool Requested(ID3D12Device* device = nullptr);
// NVFP4 on RTX 50. Standalone Original can select an FP8 equivalent for individual layers.
Options FourBitOptions(const std::filesystem::path& assets, bool allowMixedOriginal);
// INT8 uses the GPU's architecture and the kernels, plan and quantized records in int8/.
Options Int8Options(const std::filesystem::path& folder);

class Host {
public:
    // Loads the plan, the kernel set(s) and the records of `set`; checks the GPU; allocates for width x height.
    static std::unique_ptr<Host> Create(ID3D12Device* device, const Options& options, unsigned width, unsigned height, unsigned set);
    ~Host();
    // Records one whole evaluation on `commands`. `motion` may be null only when `reset` is true.
    void Evaluate(ID3D12GraphicsCommandList* commands, ID3D12Resource* colour, ID3D12Resource* motion, ID3D12Resource* output,
                  const Settings& settings, bool reset);
    unsigned Width() const;
    unsigned Height() const;
    unsigned Set() const;
    unsigned long long Evaluations() const;
    std::string Status() const;
    struct Impl;
private:
    explicit Host(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl;
};

// After the GPU can no longer run lists that used this device's objects.
void ReleaseDevice(ID3D12Device* device);
void SetLog(void (*log)(const char* line));
// The driver's own nvapi_QueryInterface where the process hooks it (NrPassBorrow::DriverQueryInterface); before the first Create.
void SetDriverQuery(void* query);
// Lab: an NvAPI layer that writes the runtime's own launches (names, shapes, parameter bytes) to NR_HOST_TRACE_RUNTIME.
void* WrapNvapi(unsigned id, void* original);

}  // namespace NrHost
