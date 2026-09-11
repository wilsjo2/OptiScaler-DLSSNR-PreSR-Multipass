#include "pch.h"

#include "AmpereMfgLoader.h"

#include <Config.h>
#include <State.h>
#include <Util.h>
#include <misc/IdentifyGpu.h>
#include <proxies/Ntdll_Proxy.h>

#include <fstream>
#include <sstream>
#include <mutex>

#ifndef NV_GPU_ARCHITECTURE_GA100
#define NV_GPU_ARCHITECTURE_GA100 0x00000170
#endif

namespace AmpereMfgLoader
{
namespace
{
Status s_status;
bool s_setupAttempted = false;
std::recursive_mutex s_mutex;
} // namespace

Status LastStatus()
{
    std::lock_guard lock(s_mutex);
    return s_status;
}

std::string ResolveAutoKernelImage()
{
    const auto& gpu = IdentifyGpu::getPrimaryGpu();
    const bool onLinux = State::Instance().isRunningOnLinux || gpu.usesVkd3dProton;
    return ResolveAutoKernelImage(static_cast<uint32_t>(gpu.nvidiaArchInfo.architecture_id), gpu.name, onLinux);
}

std::string ResolveRouter()
{
    const auto& gpu = IdentifyGpu::getPrimaryGpu();
    return ResolveRouter(static_cast<uint32_t>(gpu.nvidiaArchInfo.architecture_id), gpu.name);
}

std::string GenerateIniContent()
{
    auto* cfg = Config::Instance();
    const auto& gpu = IdentifyGpu::getPrimaryGpu();
    const bool onLinux = State::Instance().isRunningOnLinux || gpu.usesVkd3dProton;
    const int configuredFrames = cfg->FGDLSSGAmpereMfgMaxFrames.value_or_default();
    const int maxFrames = ResolveMaxGeneratedFrames(configuredFrames, onLinux);

    if (onLinux && configuredFrames == 1)
    {
        LOG_INFO("AmpereMfgLoader: On Linux/Proton with 2X FG (configured max frames 1); SetFlipConfig is stubbed in NvApiHooks to enable clean native 2X FG");
    }

    std::string kernelImg = cfg->FGDLSSGAmpereMfgKernelImage.value_or("Auto");
    if (kernelImg != "PTX" && kernelImg != "Cubin")
        kernelImg = "Auto";

    if (kernelImg == "Auto")
    {
        std::string resolved = ResolveAutoKernelImage();
        if (resolved != "Auto")
        {
            LOG_INFO("AmpereMfgLoader: Auto kernel image resolved to {} for GPU: {}",
                     resolved, IdentifyGpu::getPrimaryGpu().name);
            kernelImg = resolved;
        }
    }

    int hwBilinear = cfg->FGDLSSGAmpereMfgHardwareBilinear.value_or_default() ? 1 : 0;
    std::string router = ResolveRouter();
    int logLevel = 1;

    LOG_INFO("AmpereMfgLoader: Router selected: {} for GPU: {}", router, IdentifyGpu::getPrimaryGpu().name);

    return FormatIniContent(maxFrames, kernelImg, hwBilinear, router, logLevel);
}

void TrySetup()
{
    std::lock_guard lock(s_mutex);
    if (s_setupAttempted)
        return;
    s_setupAttempted = true;

    auto* cfg = Config::Instance();
    if (!cfg->FGDLSSGAmpereMfgUnlock.value_or_default())
        return;

    const auto& gpu = IdentifyGpu::getPrimaryGpu();
    const bool onLinux = State::Instance().isRunningOnLinux || gpu.usesVkd3dProton;
    const int configuredFrames = cfg->FGDLSSGAmpereMfgMaxFrames.value_or_default();

    if (ShouldFallbackToFsrFg(configuredFrames, onLinux, true))
    {
        s_status.Enabled = true;
        s_status.FsrFallbackActive = true;
        s_status.ErrorMessage.clear();
        LOG_INFO("AmpereMfgLoader: On Linux with 1 generated frame, falling back to OptiScaler internal FSR FG instead of sideloading dlssg_sm86");
        return;
    }

    s_status.Enabled = true;

    // Mutual exclusion: fail if Ada MFG unlock is also enabled
    if (cfg->FGDLSSGAdaMfgUnlock.value_or_default())
    {
        s_status.ErrorMessage = "Cannot enable SM86/SM75 MFG while Ada (RTX 40) MFG unlock is enabled.";
        LOG_ERROR("AmpereMfgLoader: {}", s_status.ErrorMessage);
        return;
    }

    // GPU guard: verify Nvidia Turing or Ampere architecture
    if (gpu.vendorId != VendorId::Nvidia)
    {
        s_status.ErrorMessage = "SM86/SM75 MFG requires an NVIDIA GPU.";
        LOG_ERROR("AmpereMfgLoader: {}", s_status.ErrorMessage);
        return;
    }

    const uint32_t archId = static_cast<uint32_t>(gpu.nvidiaArchInfo.architecture_id);
    const bool isAmpere = IsAmpereArch(archId) || (gpu.name.find("RTX 30") != std::string::npos);
    const bool isTuring = IsTuringArch(archId) || (gpu.name.find("RTX 20") != std::string::npos || gpu.name.find("GTX 16") != std::string::npos);

    if (!isAmpere && !isTuring)
    {
        s_status.ErrorMessage = std::format(
            "SM86/SM75 MFG requires an RTX 20 series (Turing) or RTX 30 series (Ampere) GPU. Detected arch 0x{:x} ({}).",
            archId, gpu.name);
        LOG_ERROR("AmpereMfgLoader: {}", s_status.ErrorMessage);
        return;
    }

    // Locate dlssg_sm86.dll
    auto basePath = Util::DllPath().parent_path();
    auto dllPath = std::filesystem::path(cfg->MainDllPath.value_or(basePath.wstring())) /
                   L"dlssg_sm86" / L"dlssg_sm86.dll";
    std::error_code fileError;
    if (!std::filesystem::exists(dllPath, fileError))
        dllPath = basePath / L"OptiScaler" / L"dlssg_sm86" / L"dlssg_sm86.dll";
    if (!std::filesystem::exists(dllPath, fileError))
    {
        dllPath = basePath / L"dlssg_sm86" / L"dlssg_sm86.dll";
    }
    if (!std::filesystem::exists(dllPath, fileError))
    {
        // Fallback: check directly beside OptiScaler DLL
        auto fallbackPath = basePath / L"dlssg_sm86.dll";
        if (std::filesystem::exists(fallbackPath, fileError))
        {
            dllPath = fallbackPath;
        }
        else
        {
            s_status.DllFound = false;
            s_status.ErrorMessage = "dlssg_sm86.dll not found in OptiScaler/dlssg_sm86/ or dlssg_sm86/ subfolders.";
            LOG_ERROR("AmpereMfgLoader: {}", s_status.ErrorMessage);
            return;
        }
    }
    s_status.DllFound = true;

    // Generate and write companion dlssg_sm86.ini beside the DLL
    auto iniPath = dllPath.parent_path() / L"dlssg_sm86.ini";
    try
    {
        std::filesystem::create_directories(iniPath.parent_path());
        std::ofstream iniFile(iniPath, std::ios::out | std::ios::trunc);
        if (!iniFile.is_open())
        {
            s_status.IniWritten = false;
            s_status.ErrorMessage = "Failed to open dlssg_sm86.ini for writing.";
            LOG_ERROR("AmpereMfgLoader: Failed to open {} for writing", wstring_to_string(iniPath.wstring()));
            return;
        }
        iniFile << GenerateIniContent();
        iniFile.close();
        if (!iniFile)
            throw std::runtime_error("Could not finish writing dlssg_sm86.ini");
        s_status.IniWritten = true;
    }
    catch (const std::exception& ex)
    {
        s_status.IniWritten = false;
        s_status.ErrorMessage = std::string("Error writing dlssg_sm86.ini: ") + ex.what();
        LOG_ERROR("AmpereMfgLoader: Exception writing INI: {}", ex.what());
        return;
    }

    // Load dlssg_sm86.dll
    NtdllProxy::Init();
    LOG_INFO("AmpereMfgLoader: Loading {}", wstring_to_string(dllPath.wstring()));
    HMODULE hMod = NtdllProxy::LoadLibraryExW_Ldr(dllPath.c_str(), NULL, 0);
    if (!hMod)
        hMod = LoadLibraryW(dllPath.c_str());

    if (!hMod)
    {
        DWORD err = GetLastError();
        s_status.DllLoaded = false;
        s_status.ErrorMessage = "Failed to load dlssg_sm86.dll (error code " + std::to_string(err) + ").";
        LOG_ERROR("AmpereMfgLoader: Failed to load dlssg_sm86.dll, error: {}", err);
        return;
    }

    s_status.DllLoaded = true;
    s_status.ErrorMessage.clear();
    LOG_INFO("AmpereMfgLoader: SM86 MFG loaded successfully from {}", wstring_to_string(dllPath.wstring()));
}

} // namespace AmpereMfgLoader
