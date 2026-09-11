#pragma once

#include <cstdint>
#include <sstream>
#include <string>

namespace AmpereMfgLoader
{
struct Status
{
    bool Enabled = false;     // Config says to use it
    bool DllFound = false;    // dlssg_sm86.dll found in OptiScaler/dlssg_sm86/
    bool IniWritten = false;  // dlssg_sm86.ini generated and written
    bool DllLoaded = false;   // LoadLibrary succeeded
    bool FsrFallbackActive = false; // 2X FG on Linux: internal FSR FG active
    std::string ErrorMessage; // Human-readable error if anything failed
};

Status LastStatus();

/// Called after DLL initialization, once GPU/environment information is available.
void TrySetup();

/// Resolves the companion INI MaxGeneratedFrames setting.
/// MaxFrames is clamped to [1, 3] (preserving 1 for 2X FG, 2 for 3X FG, 3 for 4X FG).
/// With NvAPI_D3D12_SetFlipConfig stubbed on Linux, both Windows and Linux cleanly
/// support single-frame 2X FG (MaxGeneratedFrames = 1) without artificial elevation.
inline int ResolveMaxGeneratedFrames(int configuredMaxFrames, bool /*onLinux*/ = false)
{
    if (configuredMaxFrames <= 0 || configuredMaxFrames > 3)
        configuredMaxFrames = 3;

    return configuredMaxFrames;
}

/// Returns true if single-frame 2X FG on Linux should fall back to OptiScaler's
/// internal FSR FG pipeline (DLSSG input -> FSR FG output) instead of sideloading dlssg_sm86.
inline bool ShouldFallbackToFsrFg(int configuredMaxFrames, bool onLinux, bool mfgUnlockEnabled)
{
    return onLinux && mfgUnlockEnabled && (configuredMaxFrames == 1);
}

constexpr uint32_t DRS_OVERRIDE_DLSSG_MULTI_FRAME_COUNT_ID = 0x104D6667;
constexpr uint32_t DRS_OVERRIDE_MAX_DLSSG_DYNAMIC_MULTI_FRAME_COUNT_ID = 0x10562D0F;

/// Evaluates whether a DRS query matches a DLSSG multi-frame setting and resolves
/// the overridden value when running on Linux with Ampere MFG unlock enabled.
inline bool TryResolveDrsMultiFrameSetting(uint32_t settingId, int configuredMaxFrames, bool onLinux, bool mfgUnlockEnabled, uint32_t& outValue)
{
    if (!onLinux || !mfgUnlockEnabled)
        return false;

    // Do not override maximum dynamic multi-frame count when configured for single-frame (<= 1),
    // because Streamline requires dynamic max > 1 to enable Dynamic MFG.
    if (settingId == DRS_OVERRIDE_MAX_DLSSG_DYNAMIC_MULTI_FRAME_COUNT_ID && configuredMaxFrames <= 1)
        return false;

    if (settingId == DRS_OVERRIDE_DLSSG_MULTI_FRAME_COUNT_ID ||
        settingId == DRS_OVERRIDE_MAX_DLSSG_DYNAMIC_MULTI_FRAME_COUNT_ID)
    {
        int clamped = configuredMaxFrames;
        if (clamped <= 0 || clamped > 3)
            clamped = 3;
        outValue = static_cast<uint32_t>(clamped);
        return true;
    }

    return false;
}

/// Formats dlssg_sm86.ini content with Native 0.2.3 specification and strict clamping.
inline std::string FormatIniContent(int maxFrames, const std::string& kernelImg, int hwBilinear = 0, const std::string& router = "SM86", int logLevel = 1)
{
    // Native 0.2.3 strictly requires: MaxGeneratedFrames must be 1, 2 or 3
    if (maxFrames <= 0 || maxFrames > 3)
        maxFrames = 3;

    std::string validKernel = kernelImg;
    if (validKernel != "PTX" && validKernel != "Cubin")
        validKernel = "Auto";

    std::string validRouter = router;
    if (validRouter != "SM75" && validRouter != "SM86")
        validRouter = "SM86";

    int validHwBilinear = (hwBilinear == 1) ? 1 : 0;
    int validLogLevel = (logLevel >= 0 && logLevel <= 3) ? logLevel : 1;

    std::ostringstream ss;
    ss << "; Native 0.2.3. Restart the game after changing this file.\n";
    ss << "[Compatibility]\n";
    ss << "Router=" << validRouter << "\n";
    ss << "KernelImage=" << validKernel << "\n";
    ss << "HardwareBilinear=" << validHwBilinear << "\n\n";
    ss << "[FrameGeneration]\n";
    ss << "MaxGeneratedFrames=" << maxFrames << "\n\n";
    ss << "[Logging]\n";
    ss << "Level=" << validLogLevel << "\n";

    return ss.str();
}

/// Checks if an architecture ID represents Turing (SM75).
inline bool IsTuringArch(uint32_t archId)
{
    return (archId == 0x00000160) || ((archId & 0xFFF0) == 0x0160);
}

/// Checks if an architecture ID represents Ampere (SM86).
inline bool IsAmpereArch(uint32_t archId)
{
    return (archId == 0x00000170) || ((archId & 0xFFF0) == 0x0170);
}

/// Resolves router string ("SM75" or "SM86") based on architecture ID and GPU name.
inline std::string ResolveRouter(uint32_t archId, const std::string& gpuName = "")
{
    if (IsTuringArch(archId))
        return "SM75";
    if (IsAmpereArch(archId))
        return "SM86";

    // Fallback: name matching
    if (!gpuName.empty())
    {
        if (gpuName.find("RTX 20") != std::string::npos ||
            gpuName.find("GTX 16") != std::string::npos ||
            gpuName.find("Turing") != std::string::npos)
            return "SM75";

        if (gpuName.find("RTX 30") != std::string::npos ||
            gpuName.find("Ampere") != std::string::npos)
            return "SM86";
    }

    return "SM86";
}

/// Resolves router string ("SM75" or "SM86") for current hardware.
std::string ResolveRouter();

/// Generates dlssg_sm86.ini content from OptiScaler config values.
std::string GenerateIniContent();

/// Resolves optimal kernel image format for current hardware/environment when Auto is requested.
std::string ResolveAutoKernelImage();

inline std::string ResolveAutoKernelImage(uint32_t archId, const std::string& name, bool onLinux)
{
    return onLinux || IsTuringArch(archId) || name.find("RTX 20") != std::string::npos ||
                   name.find("GTX 16") != std::string::npos || name.find("3080 Ti") != std::string::npos ||
                   name.find("3080Ti") != std::string::npos || name.find("Laptop") != std::string::npos ||
                   name.find("Mobile") != std::string::npos
               ? "PTX" : "Auto";
}
} // namespace AmpereMfgLoader
