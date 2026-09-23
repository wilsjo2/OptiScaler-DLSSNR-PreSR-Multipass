// ============================================================================
// tests/ampere_mfg_stub_payload_standby.cpp - the standby-role payload fixture
// (todo 11 of .omo/plans/rtx2030-mfg-integration.md).
//
// The failure matrix needs one module of the proxy family that reports the
// standby role (DlssgProxy_Role=2, measured on the real payload in todo 2), so
// the loader's PayloadStandby refusal - "loaded, but it installs nothing" - can
// be observed without a 20/30 card and without the pinned 30 MB payload. This is
// a separate source rather than a role macro on tests/ampere_mfg_stub_payload.cpp
// so the todo-2/5 fixtures stay byte-identical to what their receipts recorded.
//
// It attaches the same way as the active stub (append-only line into the file
// named by AMPERE_MFG_STUB_LOG, kernel32 file APIs because it runs in DllMain),
// so the matrix can count attaches for the standby row with the same probe.
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

    AppendUtf8Line(L"payload:attach pid=" + std::to_wstring(GetCurrentProcessId()) +
                   L" attach_count=" + std::to_wstring(count) + L" module=" + ownPath + L" role=standby");

    return TRUE;
}

extern "C" __declspec(dllexport) const wchar_t* __cdecl DlssgProxy_Name(void)
{
    return L"ampere_mfg_stub_payload_standby.dll";
}

// 2 = standby: another proxy of the family is already active, so this one only
// forwards its exports (the contract's named refusal, never "installed").
extern "C" __declspec(dllexport) int __cdecl DlssgProxy_Role(void)
{
    return 2;
}

extern "C" __declspec(dllexport) int __cdecl AmpereMfgStubAttachCount(void)
{
    return static_cast<int>(g_attachCount);
}
