#include <cassert>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>

// Mock structures to test the exact decision matrix implemented in Kernel_Hooks.cpp
enum class FGNvngxReplacement
{
    None,
    OptiScaler,
    Nukem,
    Custom
};

struct MockState
{
    bool externalFrameGeneration = false;
    FGNvngxReplacement activeFgNvngx = FGNvngxReplacement::None;
    std::optional<std::wstring> NVNGX_DLSSG_Path;
};

// Simulated fake module pointers
static void* const FAKE_DLL_MODULE = reinterpret_cast<void*>(0x1000);
static void* const FAKE_ORIGINAL_MODULE = reinterpret_cast<void*>(0x2000);
static void* const FAKE_LOADED_PATH_MODULE = reinterpret_cast<void*>(0x3000);

void* ResolveNvngxDlssgModuleA(
    const char* lpModuleName,
    void* originalHandle,
    const MockState& state,
    bool simulatePathLoadSuccess = false)
{
    if (lpModuleName == nullptr)
        return nullptr;

    if (strcmp(lpModuleName, "nvngx_dlssg.dll") == 0)
    {
        auto original = originalHandle;
        if (original != nullptr)
            return original;

        // When external Frame Generation or native DLSSG is active, OptiScaler is not emulating nvngx_dlssg.dll.
        // Avoid returning OptiScaler's dllModule, which would fail Streamline's ProductName verification.
        if (state.externalFrameGeneration ||
            state.activeFgNvngx == FGNvngxReplacement::None)
        {
            if (state.NVNGX_DLSSG_Path.has_value() && simulatePathLoadSuccess)
            {
                return FAKE_LOADED_PATH_MODULE;
            }
            return nullptr;
        }

        return FAKE_DLL_MODULE;
    }

    return originalHandle;
}

void* ResolveNvngxDlssgModuleW(
    const wchar_t* lpModuleName,
    void* originalHandle,
    const MockState& state,
    bool simulatePathLoadSuccess = false)
{
    if (lpModuleName == nullptr)
        return nullptr;

    if (wcscmp(lpModuleName, L"nvngx_dlssg.dll") == 0)
    {
        auto original = originalHandle;
        if (original != nullptr)
            return original;

        if (state.externalFrameGeneration ||
            state.activeFgNvngx == FGNvngxReplacement::None)
        {
            if (state.NVNGX_DLSSG_Path.has_value() && simulatePathLoadSuccess)
            {
                return FAKE_LOADED_PATH_MODULE;
            }
            return nullptr;
        }

        return FAKE_DLL_MODULE;
    }

    return originalHandle;
}

int main()
{
    printf("[TEST] Running KernelHooks nvngx_dlssg module spoofing unit tests...\n");

    // Case 1: External FG is active (e.g. Ampere/Turing dlssg_sm86 or native DLSS-G)
    {
        MockState state;
        state.externalFrameGeneration = true;
        state.activeFgNvngx = FGNvngxReplacement::None;

        // 1a: nvngx_dlssg.dll already loaded in memory -> returns original, never dllModule
        void* resA = ResolveNvngxDlssgModuleA("nvngx_dlssg.dll", FAKE_ORIGINAL_MODULE, state);
        assert(resA == FAKE_ORIGINAL_MODULE);

        void* resW = ResolveNvngxDlssgModuleW(L"nvngx_dlssg.dll", FAKE_ORIGINAL_MODULE, state);
        assert(resW == FAKE_ORIGINAL_MODULE);

        // 1b: nvngx_dlssg.dll not loaded, but NVNGX_DLSSG_Path is set and succeeds -> returns loaded module, never dllModule
        state.NVNGX_DLSSG_Path = L"S:\\games\\nvngx_dlssg.dll";
        resA = ResolveNvngxDlssgModuleA("nvngx_dlssg.dll", nullptr, state, true);
        assert(resA == FAKE_LOADED_PATH_MODULE);

        resW = ResolveNvngxDlssgModuleW(L"nvngx_dlssg.dll", nullptr, state, true);
        assert(resW == FAKE_LOADED_PATH_MODULE);

        // 1c: nvngx_dlssg.dll not loaded and path fails -> returns nullptr, NEVER dllModule
        resA = ResolveNvngxDlssgModuleA("nvngx_dlssg.dll", nullptr, state, false);
        assert(resA == nullptr);

        resW = ResolveNvngxDlssgModuleW(L"nvngx_dlssg.dll", nullptr, state, false);
        assert(resW == nullptr);
    }

    // Case 2: activeFgNvngx is None (native DLSSG or no mod replacement)
    {
        MockState state;
        state.externalFrameGeneration = false;
        state.activeFgNvngx = FGNvngxReplacement::None;

        // Even with externalFrameGeneration = false, if activeFgNvngx == None, OptiScaler must not spoof dllModule
        void* resA = ResolveNvngxDlssgModuleA("nvngx_dlssg.dll", nullptr, state, false);
        assert(resA == nullptr);

        void* resW = ResolveNvngxDlssgModuleW(L"nvngx_dlssg.dll", nullptr, state, false);
        assert(resW == nullptr);
    }

    // Case 3: Internal FG mod replacement is active (e.g. FSR3 FG emulation replacing DLSSG)
    {
        MockState state;
        state.externalFrameGeneration = false;
        state.activeFgNvngx = FGNvngxReplacement::OptiScaler;

        // When OptiScaler is actively emulating nvngx_dlssg and the real DLL is not loaded, it returns dllModule
        void* resA = ResolveNvngxDlssgModuleA("nvngx_dlssg.dll", nullptr, state);
        assert(resA == FAKE_DLL_MODULE);

        void* resW = ResolveNvngxDlssgModuleW(L"nvngx_dlssg.dll", nullptr, state);
        assert(resW == FAKE_DLL_MODULE);

        // Even in replacement mode, if original module was already loaded, prefer original
        resA = ResolveNvngxDlssgModuleA("nvngx_dlssg.dll", FAKE_ORIGINAL_MODULE, state);
        assert(resA == FAKE_ORIGINAL_MODULE);
    }

    // Case 4: Other DLL queries should pass through unchanged
    {
        MockState state;
        void* res = ResolveNvngxDlssgModuleA("user32.dll", FAKE_ORIGINAL_MODULE, state);
        assert(res == FAKE_ORIGINAL_MODULE);

        res = ResolveNvngxDlssgModuleW(L"user32.dll", FAKE_ORIGINAL_MODULE, state);
        assert(res == FAKE_ORIGINAL_MODULE);
    }

    printf("[TEST] All KernelHooks nvngx_dlssg unit tests passed successfully!\n");
    return 0;
}
