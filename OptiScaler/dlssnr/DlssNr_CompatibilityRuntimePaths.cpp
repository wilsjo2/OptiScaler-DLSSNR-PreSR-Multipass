#include "pch.h"
#include "DlssNr_CompatibilityRuntime.h"
#include <Config.h>
#include <State.h>
#include <Util.h>
#include <proxies/NVNGX_Proxy.h>
#include <set>

namespace DlssNr
{
std::shared_ptr<CompatibilityRuntime> CompatibilityRuntime::TryOpen(ID3D12Device* device)
{
    std::vector<std::filesystem::path> paths;
    for (const auto& path : NVNGXProxy::FeaturePaths())
        paths.emplace_back(path);
    paths.push_back(Util::ExePath().parent_path());
    paths.push_back(Util::DllPath().parent_path());
    if (Config::Instance()->MainDllPath.has_value())
        paths.emplace_back(Config::Instance()->MainDllPath.value());
    std::set<std::wstring> visited;
    for (const auto& path : paths)
    {
        if (!visited.insert(Util::ToLower(std::filesystem::absolute(path).lexically_normal().wstring())).second)
            continue;
        const auto candidate = path / L"nvngx_dlssnr.dll";
        std::error_code error;
        const auto status = std::filesystem::status(candidate, error);
        if (status.type() == std::filesystem::file_type::not_found) continue;
        // The first installed NR DLL is the selected provider. Surface its load failure.
        return Open(candidate, device, NVNGXProxy::D3D12_GetCapabilityParameters(),
                    NVNGXProxy::D3D12_DestroyParameters(), State::Instance().NVNGX_ApplicationDataPath);
    }
    LOG_INFO("NR compatibility: no supported direct runtime available; preserving driver failure");
    return {};
}
} // namespace DlssNr
