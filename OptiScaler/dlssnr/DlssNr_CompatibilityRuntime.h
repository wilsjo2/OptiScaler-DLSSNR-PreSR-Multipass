#pragma once

#include <nvsdk_ngx_params.h>
#include <filesystem>
#include <memory>

struct ID3D12Device;
struct ID3D12GraphicsCommandList;

namespace DlssNr
{
// Own this alongside the feature until its recorded GPU work has retired.
// The driver still owns parameter allocation; only NR model calls use this backend.
class CompatibilityRuntime
{
  public:
    using Allocate = NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter**);
    using Destroy = NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter*);

    static std::shared_ptr<CompatibilityRuntime> TryOpen(ID3D12Device* device);
    static std::shared_ptr<CompatibilityRuntime> Open(const std::filesystem::path& path, ID3D12Device* device,
                                                      Allocate allocate, Destroy destroy,
                                                      const std::filesystem::path& dataPath = {});
    ~CompatibilityRuntime();
    CompatibilityRuntime(const CompatibilityRuntime&) = delete;
    CompatibilityRuntime& operator=(const CompatibilityRuntime&) = delete;
    NVSDK_NGX_Result Create(ID3D12GraphicsCommandList*, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
    NVSDK_NGX_Result Evaluate(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, NVSDK_NGX_Parameter*);
    NVSDK_NGX_Result Release(NVSDK_NGX_Handle*);

  private:
    struct Module;
    std::shared_ptr<Module> module;
    ID3D12Device* device = nullptr;
    NVSDK_NGX_Parameter* capabilities = nullptr;
    Destroy destroyParameters = nullptr;
    bool initialized = false;
    CompatibilityRuntime() = default;
};
} // namespace DlssNr
