#pragma once
#include <dxgi1_6.h>
#include <d3d12.h>
#include <nvapi.h>
#include <functional>
#include <memory>
#include <string>

// A host plan identifies each operation at creation. NVIDIA feature callbacks, where used,
// still identify operations by the function handed to them by the driver.
namespace NrNet4 { struct NativeTarget; }
namespace NrSwin4 { struct NativeTarget; }
namespace NrFfnV2 { struct NativeTarget; }

namespace NrFourBitPlan {
using Launch = decltype(&NvAPI_D3D12_LaunchCuKernelChain);
struct Driver {
    decltype(&NvAPI_D3D12_CreateCuModule) createModule;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction;
    Launch launch;
};
struct Operation {
    std::shared_ptr<NrNet4::NativeTarget> net;
    std::shared_ptr<NrSwin4::NativeTarget> swin;
    std::shared_ptr<NrFfnV2::NativeTarget> ffn;
    NVAPI_CU_KERNEL_LAUNCH_PARAMS base{};
    Launch driver = nullptr;
    std::function<NvAPI_Status(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS&)> firstFallback;
    mutable std::string failure;
    unsigned weightSet = ~0u;
    int swinSlot = -1;
    uint64_t record = 0, blend = 0;
};
// A null function is a logical operation. Only this CPU pipeline receives one. The last
// stage resolves an explicit base kernel or rejects the operation before driver submission.
inline thread_local const Operation* current = nullptr;
inline NvAPI_Status Reject(const char* reason) {
    if (current) current->failure = reason;
    return NVAPI_ERROR;
}
inline unsigned WeightSet(unsigned fallback) { return current ? current->weightSet : fallback; }
inline bool Paused(bool paused) { return !current && paused; }
struct Scope {
    const Operation* previous;
    explicit Scope(const Operation& operation) : previous(current) { operation.failure.clear(); current = &operation; }
    ~Scope() { current = previous; }
};
inline NvAPI_Status Submit(ID3D12GraphicsCommandList* list, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    if (count != 1 || !kernels) return NVAPI_INVALID_ARGUMENT;
    if (kernels->hFunction) return current->driver(list, kernels, count);
    auto base = current->base;
    if (base.paramSize != kernels->paramSize) return NVAPI_ERROR;
    if (!base.hFunction) {
        if (!current->firstFallback) return Reject("no kernel for the selected precision");
        try { return current->firstFallback(list, *kernels); }
        catch (const std::exception& e) { return Reject(e.what()); }
    }
    base.pParams = kernels->pParams;
    return current->driver(list, &base, 1);
}
}

namespace NrPassBorrow {
NvAPI_Status SubmitOwned(NrFourBitPlan::Launch submit, const char* name, ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
}
namespace NrNet4 {
void BindNative(NrFourBitPlan::Operation&, const NrFourBitPlan::Driver&, ID3D12Device*, const std::string&, int planIndex);
NvAPI_Status SubmitNative(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
}
namespace NrSwin4 {
void BindNative(NrFourBitPlan::Operation&, const NrFourBitPlan::Driver&, ID3D12Device*, const std::string&, unsigned set, int swinSlot);
NvAPI_Status SubmitNative(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
}
namespace NrFfnV2 {
void BindNative(NrFourBitPlan::Operation&, const NrFourBitPlan::Driver&, ID3D12Device*, const std::string&, unsigned set, int swinSlot);
NvAPI_Status SubmitNative(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
}
