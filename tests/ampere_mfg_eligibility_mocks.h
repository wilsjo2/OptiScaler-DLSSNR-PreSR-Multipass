// Stand-ins for the production sources AmpereMfgLoader.cpp reads the host from (todo 5 of
// .omo/plans/rtx2030-mfg-integration.md). tests/ampere_mfg_eligibility_smoke.cpp includes this file first,
// then the production loader .cpp itself, so the code under test is the real file: only the four sources it
// reads - the config surface, IdentifyGpu's adapter list, MfgUnlock's session latch and State's FG output -
// are replaced, and the empty headers in tests/ampere_mfg_eligibility_seams/ stop the real ones from being
// pulled in. That is the same seam trick tests/mfg_unlock/run.ps1 uses for the Ada patcher.
//
// The fixtures fill these globals to describe a machine; the real-host case fills them from an actual DXGI +
// NVAPI probe (tests/ampere_mfg_eligibility_smoke.cpp, ProbeHostAdapter).
#pragma once

#define NOMINMAX

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

// The loader logs through spdlog in production; the harness only needs the calls to compile.
#define LOG_INFO(...) ((void) 0)
#define LOG_WARN(...) ((void) 0)
#define LOG_ERROR(...) ((void) 0)
#define LOG_TRACE(...) ((void) 0)

// OptiTypes.h
namespace VendorId
{
enum Value : uint32_t
{
    Invalid = 0,
    Microsoft = 0x1414,
    Nvidia = 0x10DE,
    AMD = 0x1002,
    Intel = 0x8086,
};
};

// Config.h: exactly the members AmpereMfgLoader.cpp reads, with the call shapes the production ones have
// (value_or_default on a WithDefault optional, value_for_config_or on the NoDefault string).
struct MockBool
{
    bool stored = false;
    bool value_or_default() const
    {
        return stored;
    }
};

struct MockInt
{
    int stored = 3;
    int value_or_default() const
    {
        return stored;
    }
};

struct MockNoDefaultString
{
    std::optional<std::string> stored;
    std::string value_for_config_or(std::string fallback)
    {
        return stored.has_value() ? *stored : fallback;
    }
};

enum class FGInput : uint32_t
{
    NoFG,
    Upscaler,
    DLSSG,
    NvngxFG,
};

enum class FGOutput : uint32_t
{
    NoFG,
    FSRFG,
    DLSSG,
    XeFG,
};

enum class FGNvngxReplacement : uint32_t
{
    None,
    Nukems,
    Arturs,
};

template <typename T> struct MockSelection
{
    T stored {};
    T value_or_default() const { return stored; }
};

struct Config
{
    MockSelection<::FGInput> FGInput;
    MockSelection<::FGOutput> FGOutput;
    MockSelection<::FGNvngxReplacement> FGNvngxReplacement;
    MockBool ExternalFrameGeneration;
    MockBool FGDLSSGAmpereMfgUnlock;
    MockInt FGDLSSGAmpereMfgMaxFrames;
    MockNoDefaultString FGDLSSGAmpereMfgKernelImage;

    static Config* Instance()
    {
        static Config config;
        return &config;
    }
};

// MfgUnlock.h: one bool for the gate.
namespace MfgUnlock
{
inline bool g_enabledForSession = false;

inline bool EnabledForSession()
{
    return g_enabledForSession;
}
} // namespace MfgUnlock

// State.h
class State
{
  public:
    FGInput activeFgInput = FGInput::NoFG;
    FGOutput activeFgOutput = FGOutput::NoFG;
    FGNvngxReplacement activeFgNvngx = FGNvngxReplacement::None;

    static State& Instance()
    {
        static State state;
        return state;
    }
};

// misc/IdentifyGpu.h: the same fields of GpuInformation the loader reads, plus the two accessors. The vector's
// first entry is the primary adapter, which is the order IdentifyGpu's own list uses.
struct MockArchInfo
{
    unsigned architecture = 0;
    unsigned architecture_id = 0;
    unsigned implementation = 0;
};

struct GpuInformation
{
    VendorId::Value vendorId = VendorId::Nvidia;
    std::string name = "NVIDIA GeForce RTX 3060";
    bool softwareAdapter = false;
    MockArchInfo nvidiaArchInfo {};
};

namespace IdentifyGpu
{
inline std::vector<GpuInformation> g_adapters { GpuInformation {} };

inline void SetAdapters(std::vector<GpuInformation> adapters)
{
    g_adapters = std::move(adapters);
}

inline const GpuInformation& getPrimaryGpu()
{
    return g_adapters.front();
}

inline const std::vector<GpuInformation>& getAllGpus()
{
    return g_adapters;
}
} // namespace IdentifyGpu

// Util.h
inline std::string wstring_to_string(const std::wstring& text)
{
    std::string out;

    for (const wchar_t c : text)
        out.push_back(static_cast<char>(c));

    return out;
}

namespace Util
{
inline std::wstring DllPath()
{
    // The production default is the directory that holds OptiScaler.dll. The harnesses set AMPERE_MFG_DLL_PATH
    // to a path inside their scratch tree so they can drive the production no-argument Arm() - the exact entry
    // the Streamline init path calls - without going through an explicit ArmOptions::PackageRoot.
    wchar_t buffer[MAX_PATH * 4] = {};
    const DWORD length = GetEnvironmentVariableW(L"AMPERE_MFG_DLL_PATH", buffer, MAX_PATH * 4);

    if (length > 0 && length < MAX_PATH * 4)
        return std::wstring(buffer, length);

    return L"C:\\ampere-mfg-eligibility\\OptiScaler.dll";
}
} // namespace Util

// The payload-pin seam (todo 13), on top of the four host sources above: the harnesses compile the production
// ladder against stub modules, so tests/ampere_mfg_payload_pin_seam.h supplies the pin the ladder compares
// against. It must be seen before AmpereMfgLoader.h is compiled - which this header always is.
#include "ampere_mfg_payload_pin_seam.h"
