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
    for (const auto& path : State::Instance().NVNGX_FeatureInfo_Paths)
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
        if (auto runtime = Open(path / L"nvngx_dlssnr.dll", device, NVNGXProxy::D3D12_GetCapabilityParameters(),
                                NVNGXProxy::D3D12_DestroyParameters(), State::Instance().NVNGX_ApplicationDataPath))
            return runtime;
    }
    LOG_INFO("NR compatibility: no supported direct runtime available; preserving driver failure");
    return {};
}
} // namespace DlssNr
