#pragma once

#include "../../../OptiScaler/dlssnr/NrNvfp4.h"
#include "NrFfnV2.h"
#include "NrFp8.h"
#include "NrHost.h"
#include "NrNet4.h"
#include "NrSwin4.h"
#include <stdexcept>

namespace NrPassRuntime
{
struct Work
{
    bool swinActive = false;
    unsigned long long swin = 0, net = 0, ffn = 0;
    // Launches that did not run as asked (each layer: Faults()): any during an evaluation and it is not our model.
    unsigned long long swinFaults = 0, netFaults = 0, ffnFaults = 0;
};

inline Work ReadWork()
{
    Work work;
    work.swin = NrSwin4::CompletedWork(work.swinActive);
    work.net = NrNet4::CompletedWork();
    work.ffn = NrFfnV2::CompletedWork();
    work.swinFaults = NrSwin4::Faults();
    work.netFaults = NrNet4::Faults();
    work.ffnFaults = NrFfnV2::Faults();
    return work;
}

enum class Backend { Nvidia, Nvfp4, Fp8, Host, Unavailable };

struct Model
{
    Backend backend;
    unsigned weights;
};

// Both applications use the same selection at creation and evaluation. Standalone can also run FP8 on its own host.
inline Model Select(unsigned preset, bool hostFp8 = false)
{
    using namespace DlssNr::Nvfp4;
    if (IsInt8(preset)) return { Backend::Host, Int8Set(preset) };
    if (IsFp8(preset)) return { hostFp8 ? Backend::Host : Backend::Fp8, Fp8Set(preset) };
    if (preset == ModelANvfp4 || IsPassInOne(preset)) return { Backend::Nvfp4, WeightSet(preset) };
    return { preset <= 3 ? Backend::Nvidia : Backend::Unavailable, 0 };
}

inline bool UsesHost(const Model& model)
{
    return model.backend == Backend::Host || model.backend == Backend::Nvfp4;
}

// Both products retire these hosts through their submission tracker, including discarded command lists.
inline std::shared_ptr<NrHost::Host> CreateTrackedHost(ID3D12Device* device, const Model& model,
                                                       NrHost::Options options, unsigned width, unsigned height)
{
    if (!UsesHost(model)) throw std::invalid_argument("The selected model does not use a native NR host.");
    options.callerTracksGpuLifetime = true;
    return NrHost::Host::Create(device, options, width, height, model.weights);
}

inline void EnableFourBit(bool enabled, bool modelBFeedForward = false)
{
    NrSwin4::Pause(!enabled);
    NrNet4::Pause(!enabled);
    NrFfnV2::Pause(!enabled && !modelBFeedForward);
}

// The lab can override the selected set for one evaluation. All four-bit layers must follow it.
inline unsigned BindFourBitWeights(const Model& model)
{
    const auto requested = model.backend == Backend::Nvfp4 ? model.weights : 0;
    const auto selected = NrSwin4::UseSet(requested ? requested : NrSwin4::kDefaultSet);
    NrNet4::UseSet(selected);
    NrFfnV2::UseSet(selected);
    return selected;
}

inline void Activate(const Model& model, ID3D12GraphicsCommandList* commands, bool modelBFeedForward = false)
{
    // The native plan calls its layers directly. An unrelated NVIDIA evaluation must never inherit them.
    EnableFourBit(false, modelBFeedForward);
    NrFp8::UseSet(model.backend == Backend::Fp8 ? model.weights : 0, commands);
}

inline void End(ID3D12GraphicsCommandList* list)
{
    NrFp8::EndEvaluation(list);
    NrFfnV2::EndEvaluation(list);
    NrSwin4::EndEvaluation(list);
    NrNet4::EndEvaluation(list);
    EnableFourBit(false);
}
} // namespace NrPassRuntime
