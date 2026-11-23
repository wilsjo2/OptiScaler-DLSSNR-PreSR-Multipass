#pragma once

// Neural Rendering uses the existing NVIDIA NGX driver dispatcher.

#include <d3d12.h>
#include "DlssNr_ModelParameters.h"
#include <memory>
#include <cstdint>

namespace DlssNr
{
namespace Proxy
{
// Borrowed inputs for one model evaluation; depth and motion retain independent active regions.
struct Frame
{
    ID3D12Resource* color = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* output = nullptr;
    GuideExtent size {};
    GuideRegions guides {};
    bool depthInverted = false, reset = false;
    float mvScaleX = 1.0f, mvScaleY = 1.0f;
};

class Context
{
    struct Impl;
    std::unique_ptr<Impl> _impl;

  public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // True when the driver's nvngx is initialised and exports what this path needs.
    static bool Available();

    // Creation records GPU work. A feature becomes ready only in a later submission epoch.
    unsigned int Prepare(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, unsigned int width,
                         unsigned int height, const ModelSettings& settings, uint64_t submissionEpoch, bool* ready);
    // Returns true when a live pass changes tuning. NGX reads it at creation, the INT8 host at evaluation.
    bool UpdateTuning(const ModelSettings& settings);
    bool HasFeature() const;
    bool Ready(uint64_t submissionEpoch) const;
    void Collect();

    // Each pass owns its feature, parameter map and temporal history. False evaluated means no output.
    unsigned int Run(ID3D12GraphicsCommandList* cmdList, ID3D12Device* device, const Frame& frame,
                     const ModelSettings& settings, uint64_t submissionEpoch, bool* evaluated = nullptr);

    // Retires the current feature and clears the failure latch without immediately freeing GPU work.
    void RetryAfterFailure();

    void Submitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists);
    void ResetRecording(ID3D12CommandList* commands);
    bool Idle();
    void FinishSubmitted();

    // Retires ownership; destruction occurs only after recordings are discarded and GPU work completes.
    // Unresolved ownership is abandoned if this context is destroyed.
    void Release();
};
} // namespace Proxy
} // namespace DlssNr
