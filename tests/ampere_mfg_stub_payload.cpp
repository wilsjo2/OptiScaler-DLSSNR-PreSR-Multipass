// ============================================================================
// tests/ampere_mfg_stub_payload.cpp - stand-in payload for the sidecar harness
// (todo 2 of .omo/plans/rtx2030-mfg-integration.md).
//
// The harness needs a module that (a) RECORDS THE MOMENT IT IS MAPPED, from its
// own DllMain, so the ordering proof rests on the payload's own load time and
// not on the loader's bookkeeping, and (b) exposes the same proxy identity
// exports the real payload exposes (DlssgProxy_Name / DlssgProxy_Role), so the
// harness queries identity and role the same way for the stub and for the real
// module.
//
// Role encoding is the real payload's, measured in this todo: 1 = active (the
// first module of the proxy family in the process), 2 = standby (a later one).
// The stub reports 1.
//
// The stub writes one append-only line into the file named by the environment
// variable AMPERE_MFG_STUB_LOG (the harness sets it to the case's event log, so
// the attach line interleaves with the loader's own events). Kernel32 file APIs
// are used instead of the CRT because this runs in DllMain.
// ============================================================================

#include <windows.h>

#include <string>

static volatile LONG g_attachCount = 0;

static void AppendUtf8Line(const std::wstring& line)
{
    wchar_t path[MAX_PATH * 4];
    const DWORD length = GetEnvironmentVariableW(L"AMPERE_MFG_STUB_LOG", path, MAX_PATH * 4);

    if (length == 0 || length >= MAX_PATH * 4)
        return;

    HANDLE file = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);

    if (file == INVALID_HANDLE_VALUE)
        return;

    const int needed = WideCharToMultiByte(CP_UTF8, 0, line.c_str(), static_cast<int>(line.size()), nullptr, 0, nullptr,
                                          nullptr);
    std::string utf8(static_cast<size_t>(needed < 0 ? 0 : needed), '\0');

    if (needed > 0)
        WideCharToMultiByte(CP_UTF8, 0, line.c_str(), static_cast<int>(line.size()), utf8.data(), needed, nullptr,
                            nullptr);

    utf8.push_back('\n');

    DWORD written = 0;
    WriteFile(file, utf8.data(), static_cast<DWORD>(utf8.size()), &written, nullptr);
    CloseHandle(file);
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID)
{
    if (reason != DLL_PROCESS_ATTACH)
        return TRUE;

    const LONG count = InterlockedIncrement(&g_attachCount);

    wchar_t ownPath[MAX_PATH * 4] = {};
    GetModuleFileNameW(instance, ownPath, MAX_PATH * 4);

    std::wstring line = L"payload:attach pid=" + std::to_wstring(GetCurrentProcessId()) +
                        L" attach_count=" + std::to_wstring(count) + L" module=" + ownPath;
    AppendUtf8Line(line);

    return TRUE;
}

extern "C" __declspec(dllexport) const wchar_t* __cdecl DlssgProxy_Name(void)
{
    return L"ampere_mfg_stub_payload.dll";
}

// 1 = active, 2 = standby. See the role note at the top of this file.
extern "C" __declspec(dllexport) int __cdecl DlssgProxy_Role(void)
{
    return 1;
}

// In-process attach counter, so the harness can assert on the stub's own count
// instead of only on the parsed log.
extern "C" __declspec(dllexport) int __cdecl AmpereMfgStubAttachCount(void)
{
    return static_cast<int>(g_attachCount);
}
