// smooth_motion_profile.exe: reads, sets or removes NVIDIA's driver Smooth Motion setting for display_filter.exe only.
//
//   smooth_motion_profile query            read-only: the driver's Smooth Motion settings (names, ids, values), and what the
//                                          global profile and the profiles of display_filter.exe and wow.exe hold
//   smooth_motion_profile on               Smooth Motion on for display_filter.exe (creates the profile
//                                          "Display Filter (display_filter.exe)" if the exe has none yet)
//   smooth_motion_profile off              the setting back to the driver's default in display_filter.exe's profile, and the
//                                          profile removed again if this tool created it
//
// Nothing else is changed: no global setting, no other application's profile. The driver does not name these settings;
// the ids are the ones NVIDIA's own present layer (NvPresent64.dll) and the NVIDIA App's backend read.
// On driver 621.24 the driver writes them only for an administrator (or the NVIDIA App): from a normal prompt "on"
// changes nothing and says so (exit code 4).
#include <windows.h>
#include <nvapi.h>
#include <cstdio>
#include <cwchar>
#include <string>
#include <vector>
namespace {
constexpr const wchar_t* kExe = L"display_filter.exe";
constexpr const wchar_t* kProfile = L"Display Filter (display_filter.exe)";
void Fail(const char* what, NvAPI_Status status) {
    NvAPI_ShortString text{}; NvAPI_GetErrorMessage(status, text);
    std::fprintf(stderr, "%s failed: %d %s\n", what, int(status), text);
    std::exit(1);
}
std::wstring Name(NvU32 id) {
    NvAPI_UnicodeString name{};
    if (NvAPI_DRS_GetSettingNameFromId(id, &name) != NVAPI_OK) return L"";
    return std::wstring(reinterpret_cast<const wchar_t*>(name));
}
bool Contains(std::wstring text, const wchar_t* part) {
    for (auto& c : text) c = towlower(c);
    std::wstring wanted(part); for (auto& c : wanted) c = towlower(c);
    return text.find(wanted) != std::wstring::npos;
}
// The driver does not list Smooth Motion among its named settings (NvAPI_DRS_EnumAvailableSettingIds gives 128 public
// ones, none of them Smooth Motion), so the ids are given here. 0xB0D384C0 = Smooth Motion on/off (0/1), set by the NVIDIA
// App per program; 0xB0CC0875 = the APIs Smooth Motion may take (a mask; NVIDIA's own profiles set it to 0 for Explorer,
// Office, media players, and remove one API for some games). Evidence on this PC (Nov 2026): "scan", and both ids appear
// as constants in NvPresent64.dll (the driver's Smooth Motion present layer) and in the NVIDIA App's NvBackend64.dll.
constexpr NvU32 kEnable = 0xB0D384C0, kAllowedApis = 0xB0CC0875;
std::vector<std::pair<NvU32, std::wstring>> SmoothMotionSettings() {
    return {{kEnable, L"Smooth Motion on/off"}, {kAllowedApis, L"Smooth Motion allowed APIs (mask)"}};
}
void PrintValues(NvU32 id) {
    NVDRS_SETTING_VALUES values{}; values.version = NVDRS_SETTING_VALUES_VER; NvU32 max = NVAPI_SETTING_MAX_VALUES;
    if (NvAPI_DRS_EnumAvailableSettingValues(id, &max, &values) != NVAPI_OK) { std::printf("    (values not listed)\n"); return; }
    if (values.settingType != NVDRS_DWORD_TYPE) { std::printf("    type %d (not a number)\n", int(values.settingType)); return; }
    std::printf("    default 0x%X; values:", values.u32DefaultValue);
    for (NvU32 i = 0; i < values.numSettingValues; ++i) std::printf(" 0x%X", values.settingValues[i].u32Value);
    std::printf("\n");
}
void PrintSetting(NvDRSSessionHandle session, NvDRSProfileHandle profile, NvU32 id, const wchar_t* where) {
    NVDRS_SETTING setting{}; setting.version = NVDRS_SETTING_VER;
    const auto s = NvAPI_DRS_GetSetting(session, profile, id, &setting);
    if (s == NVAPI_SETTING_NOT_FOUND) { std::wprintf(L"    %ls: not set\n", where); return; }
    if (s != NVAPI_OK) { std::wprintf(L"    %ls: read failed (%d)\n", where, int(s)); return; }
    std::wprintf(L"    %ls: 0x%X (location %d: 0 this profile, 1 global, 2 base, 3 default)\n", where,
        setting.u32CurrentValue, int(setting.settingLocation));
}
NvDRSProfileHandle FindExeProfile(NvDRSSessionHandle session, const wchar_t* exe, std::wstring* profileName = nullptr) {
    NVDRS_APPLICATION application{}; application.version = NVDRS_APPLICATION_VER; NvDRSProfileHandle profile = nullptr;
    NvAPI_UnicodeString name{}; wcsncpy_s(reinterpret_cast<wchar_t*>(name), NVAPI_UNICODE_STRING_MAX, exe, _TRUNCATE);
    if (NvAPI_DRS_FindApplicationByName(session, name, &profile, &application) != NVAPI_OK) return nullptr;
    if (profileName) {
        NVDRS_PROFILE info{}; info.version = NVDRS_PROFILE_VER;
        if (NvAPI_DRS_GetProfileInfo(session, profile, &info) == NVAPI_OK) *profileName = reinterpret_cast<const wchar_t*>(info.profileName);
    }
    return profile;
}
}
int wmain(int argc, wchar_t** argv) {
    const std::wstring command = argc > 1 ? argv[1] : L"query";
    if (command != L"query" && command != L"on" && command != L"off" && command != L"list" && command != L"scan") {
        std::fprintf(stderr, "Use: smooth_motion_profile query | on | off | list | scan <hex id>...\n"); return 2;
    }
    if (const auto s = NvAPI_Initialize(); s != NVAPI_OK) Fail("NvAPI_Initialize", s);
    if (command == L"list") { // read-only: every setting id the driver knows, with its name where it has one
        NvU32 count = 0; NvAPI_DRS_EnumAvailableSettingIds(nullptr, &count);
        std::vector<NvU32> ids(count + 64); count = NvU32(ids.size());
        if (const auto s = NvAPI_DRS_EnumAvailableSettingIds(ids.data(), &count); s != NVAPI_OK) Fail("EnumAvailableSettingIds", s);
        for (NvU32 i = 0; i < count; ++i) std::wprintf(L"0x%08X %ls\n", ids[i], Name(ids[i]).c_str());
        return 0;
    }
    if (command == L"scan") { // read-only: the given setting ids (hex) in the global profile and in every profile that sets them
        NvDRSSessionHandle session = nullptr;
        if (const auto s = NvAPI_DRS_CreateSession(&session); s != NVAPI_OK) Fail("DRS_CreateSession", s);
        if (const auto s = NvAPI_DRS_LoadSettings(session); s != NVAPI_OK) Fail("DRS_LoadSettings", s);
        std::vector<NvU32> wanted;
        for (int i = 2; i < argc; ++i) wanted.push_back(NvU32(wcstoul(argv[i], nullptr, 16)));
        NvDRSProfileHandle global = nullptr; NvAPI_DRS_GetCurrentGlobalProfile(session, &global);
        for (const auto id : wanted) {
            std::wprintf(L"0x%08X name \"%ls\"\n", id, Name(id).c_str());
            if (global) PrintSetting(session, global, id, L"global profile");
        }
        NvU32 profiles = 0; NvAPI_DRS_GetNumProfiles(session, &profiles);
        for (NvU32 p = 0; p < profiles; ++p) {
            NvDRSProfileHandle profile = nullptr; if (NvAPI_DRS_EnumProfiles(session, p, &profile) != NVAPI_OK) continue;
            NVDRS_PROFILE info{}; info.version = NVDRS_PROFILE_VER; NvAPI_DRS_GetProfileInfo(session, profile, &info);
            for (const auto id : wanted) {
                NVDRS_SETTING setting{}; setting.version = NVDRS_SETTING_VER;
                if (NvAPI_DRS_GetSetting(session, profile, id, &setting) == NVAPI_OK && setting.settingLocation == NVDRS_CURRENT_PROFILE_LOCATION)
                    std::wprintf(L"  \"%ls\" (predefined %u): 0x%08X = 0x%X\n", reinterpret_cast<const wchar_t*>(info.profileName),
                        info.isPredefined, id, setting.u32CurrentValue);
            }
        }
        NvAPI_DRS_DestroySession(session); return 0;
    }
    NvDRSSessionHandle session = nullptr;
    if (const auto s = NvAPI_DRS_CreateSession(&session); s != NVAPI_OK) Fail("DRS_CreateSession", s);
    if (const auto s = NvAPI_DRS_LoadSettings(session); s != NVAPI_OK) Fail("DRS_LoadSettings", s);
    const auto settings = SmoothMotionSettings();
    if (settings.empty()) { std::printf("This driver lists no Smooth Motion setting.\n"); return 3; }
    NvDRSProfileHandle global = nullptr; NvAPI_DRS_GetCurrentGlobalProfile(session, &global);
    std::wstring exeProfileName; NvDRSProfileHandle exeProfile = FindExeProfile(session, kExe, &exeProfileName);
    if (command == L"query") {
        std::wstring wowName; const auto wow = FindExeProfile(session, L"wow.exe", &wowName);
        for (const auto& [id, name] : settings) {
            std::wprintf(L"0x%08X %ls\n", id, name.c_str());
            PrintValues(id);
            if (global) PrintSetting(session, global, id, L"global profile");
            if (exeProfile) PrintSetting(session, exeProfile, id, (L"display_filter.exe in \"" + exeProfileName + L"\"").c_str());
            else std::wprintf(L"    display_filter.exe: no profile\n");
            if (wow) PrintSetting(session, wow, id, (L"wow.exe in \"" + wowName + L"\"").c_str());
        }
        NvAPI_DRS_DestroySession(session); return 0;
    }
    // Only the on/off setting is written; the allowed-API mask is left as the driver has it.
    const NvU32 enableId = kEnable;
    if (command == L"on") {
        // Changes stay in this session until SaveSettings: if anything below fails, the driver keeps nothing.
        bool created = false;
        if (!exeProfile) {
            NVDRS_PROFILE info{}; info.version = NVDRS_PROFILE_VER;
            wcsncpy_s(reinterpret_cast<wchar_t*>(info.profileName), NVAPI_UNICODE_STRING_MAX, kProfile, _TRUNCATE);
            if (const auto s = NvAPI_DRS_CreateProfile(session, &info, &exeProfile); s != NVAPI_OK) Fail("DRS_CreateProfile", s);
            NVDRS_APPLICATION application{}; application.version = NVDRS_APPLICATION_VER;
            wcsncpy_s(reinterpret_cast<wchar_t*>(application.appName), NVAPI_UNICODE_STRING_MAX, kExe, _TRUNCATE);
            wcsncpy_s(reinterpret_cast<wchar_t*>(application.userFriendlyName), NVAPI_UNICODE_STRING_MAX, L"Display Filter", _TRUNCATE);
            if (const auto s = NvAPI_DRS_CreateApplication(session, exeProfile, &application); s != NVAPI_OK) Fail("DRS_CreateApplication", s);
            exeProfileName = kProfile; created = true;
        }
        NVDRS_SETTING setting{}; setting.version = NVDRS_SETTING_VER; setting.settingId = enableId;
        setting.settingType = NVDRS_DWORD_TYPE; setting.u32CurrentValue = 1;
        if (const auto s = NvAPI_DRS_SetSetting(session, exeProfile, &setting); s != NVAPI_OK) {
            // Seen on driver 621.24 from a normal (not administrator) prompt: the driver does not list Smooth Motion
            // among the settings anyone may write (NVAPI_SETTING_NOT_FOUND), and its allowed-API mask needs an
            // administrator (NVAPI_INVALID_USER_PRIVILEGE). The NVIDIA App writes it through its own service.
            if (s == NVAPI_SETTING_NOT_FOUND || s == NVAPI_INVALID_USER_PRIVILEGE) {
                std::fprintf(stderr, "The driver refused the Smooth Motion setting (%d) from this prompt. Nothing was changed.\n"
                    "Turn Smooth Motion on for display_filter.exe in the NVIDIA App (Graphics, Program settings, Add,\n"
                    "display_filter.exe, Smooth Motion On), or run \"smooth_motion_profile on\" from an administrator prompt.\n", int(s));
                NvAPI_DRS_DestroySession(session); return 4;
            }
            Fail("DRS_SetSetting", s);
        }
        if (const auto s = NvAPI_DRS_SaveSettings(session); s != NVAPI_OK) Fail("DRS_SaveSettings", s);
        if (created) std::wprintf(L"Created profile \"%ls\" for %ls.\n", kProfile, kExe);
        std::wprintf(L"Set 0x%08X (Smooth Motion on/off) = 1 in \"%ls\".\n", enableId, exeProfileName.c_str());
    } else {
        if (!exeProfile) { std::printf("display_filter.exe has no profile: nothing to undo.\n"); NvAPI_DRS_DestroySession(session); return 0; }
        if (exeProfileName == kProfile) {
            if (const auto s = NvAPI_DRS_DeleteProfile(session, exeProfile); s != NVAPI_OK) Fail("DRS_DeleteProfile", s);
            std::wprintf(L"Deleted profile \"%ls\".\n", kProfile);
        } else {
            const auto s = NvAPI_DRS_DeleteProfileSetting(session, exeProfile, enableId);
            if (s == NVAPI_INVALID_USER_PRIVILEGE) {
                std::fprintf(stderr, "The driver refused the change from this prompt. Nothing was changed. Set Smooth Motion to\n"
                    "Off (or Use global setting) for display_filter.exe in the NVIDIA App, or run this from an administrator prompt.\n");
                NvAPI_DRS_DestroySession(session); return 4;
            }
            if (s != NVAPI_OK && s != NVAPI_SETTING_NOT_FOUND) Fail("DRS_DeleteProfileSetting", s);
            std::wprintf(L"Removed 0x%08X from \"%ls\" (back to the driver's default).\n", enableId, exeProfileName.c_str());
        }
        if (const auto s = NvAPI_DRS_SaveSettings(session); s != NVAPI_OK) Fail("DRS_SaveSettings", s);
    }
    NvAPI_DRS_DestroySession(session);
    return 0;
}
