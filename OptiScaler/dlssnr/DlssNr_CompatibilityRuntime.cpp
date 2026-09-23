#include "pch.h"
#include "DlssNr_CompatibilityRuntime.h"
#include "DlssNr_RuntimeImports.h"
#include <Logger.h>
#include <d3d12.h>
#include <nvsdk_ngx.h>
#include <psapi.h>
#include <algorithm>
#include <mutex>
#include <map>
#include <string>
#include <vector>
#include <set>
#include <condition_variable>
#include <atomic>
#pragma comment(lib, "psapi.lib")

namespace DlssNr
{
namespace
{
std::recursive_mutex registryMutex;
std::condition_variable_any moduleRetired;
bool moduleRegistered = false;
// Only the model's import is redirected. Outside a direct NR call, even that import
// reports the real path. No process-wide hook, driver patch, or on-disk change.
thread_local HMODULE callerAlias = nullptr;
// Preserve a loader/overlay's existing IAT wrapper for ordinary path queries.
std::atomic<decltype(&GetModuleFileNameW)> originalPathW { &GetModuleFileNameW };
std::atomic<decltype(&GetModuleFileNameA)> originalPathA { &GetModuleFileNameA };
DWORD WINAPI CallerPath(HMODULE queried, LPWSTR path, DWORD capacity)
{
    const auto original = originalPathW.load();
    if (callerAlias && queried == callerAlias)
    {
        constexpr wchar_t alias[] = L"nvngx.dll";
        if (!capacity)
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return 0;
        }
        const DWORD copied = (DWORD) std::min<size_t>(capacity - 1, std::size(alias) - 1);
        memcpy(path, alias, copied * sizeof(wchar_t));
        path[copied] = 0;
        if (capacity < std::size(alias))
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return capacity;
        }
        return copied;
    }
    return original(queried, path, capacity);
}

DWORD WINAPI CallerPathA(HMODULE queried, LPSTR path, DWORD capacity)
{
    const auto original = originalPathA.load();
    if (callerAlias && queried == callerAlias)
    {
        constexpr char alias[] = "nvngx.dll";
        if (!capacity)
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return 0;
        }
        const DWORD copied = (DWORD) std::min<size_t>(capacity - 1, std::size(alias) - 1);
        memcpy(path, alias, copied);
        path[copied] = 0;
        if (capacity < std::size(alias))
        {
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return capacity;
        }
        return copied;
    }
    return original(queried, path, capacity);
}

struct CallerScope
{
    HMODULE previous = callerAlias;
    CallerScope()
    {
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&CallerPath), &callerAlias);
    }
    ~CallerScope() { callerAlias = previous; }
};

bool ReplaceImport(void** slot, void* expected, void* replacement)
{
    if (*slot != expected)
    {
        LOG_ERROR("NR compatibility: import changed before replacement: slot={} expected={} actual={}", (void*) slot,
                  expected, *slot);
        return false;
    }
    DWORD protection = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &protection))
    {
        LOG_ERROR("NR compatibility: import protection change failed: slot={} error={}", (void*) slot, GetLastError());
        return false;
    }
    const bool replaced = InterlockedCompareExchangePointer(slot, replacement, expected) == expected;
    DWORD ignored = 0;
    if (!VirtualProtect(slot, sizeof(void*), protection, &ignored))
        LOG_WARN("NR compatibility: import protection restore failed: slot={} error={}", (void*) slot, GetLastError());
    if (!replaced)
        LOG_ERROR("NR compatibility: import changed concurrently: slot={}", (void*) slot);
    return replaced;
}
} // namespace

struct CompatibilityRuntime::Module
{
    struct Binding
    {
        RuntimeImports::Slot slot;
        void* original;
    };
    HMODULE handle = nullptr;
    bool borrowed = false;
    bool registered = false;
    std::vector<Binding> imports;
    std::filesystem::path path;
    std::recursive_mutex mutex;
    std::set<ID3D12Device*> initializedDevices;
    std::condition_variable_any deviceRetired;
    // The snippet Init_Ext ABI takes driver capabilities, unlike the public NGX Init_Ext ABI.
    using Init = NVSDK_NGX_Result (*)(unsigned long long, const wchar_t*, ID3D12Device*, unsigned int,
                                      NVSDK_NGX_Parameter*);
    Init init = nullptr;
    decltype(&NVSDK_NGX_D3D12_CreateFeature) create = nullptr;
    decltype(&NVSDK_NGX_D3D12_EvaluateFeature) evaluate = nullptr;
    decltype(&NVSDK_NGX_D3D12_ReleaseFeature) release = nullptr;
    using Shutdown = NVSDK_NGX_Result (*)(ID3D12Device*);
    Shutdown shutdown = nullptr;

    ~Module()
    {
        std::lock_guard registryLock(registryMutex);
        for (const auto& binding : imports)
            ReplaceImport(binding.slot.address,
                          binding.slot.wide ? reinterpret_cast<void*>(&CallerPath)
                                            : reinterpret_cast<void*>(&CallerPathA),
                          binding.original);
        if (handle)
            FreeLibrary(handle);
        if (registered)
        {
            moduleRegistered = false;
            moduleRetired.notify_all();
        }
    }
};

std::shared_ptr<CompatibilityRuntime> CompatibilityRuntime::Open(const std::filesystem::path& candidate,
                                                                 ID3D12Device* device, Allocate allocate,
                                                                 Destroy destroy, const std::filesystem::path& dataPath)
{
    if (!device || !allocate || !destroy)
        return {};
    // Cache only live owners. Last-owner shutdown/unload occurs after GPU retirement.
    static std::weak_ptr<Module> liveModule;
    static std::map<ID3D12Device*, std::weak_ptr<CompatibilityRuntime>> devices;
    std::unique_lock lock(registryMutex);
    try
    {
        auto path = std::filesystem::absolute(candidate).lexically_normal();
        auto loaded = liveModule.lock();
        if (!loaded)
            moduleRetired.wait(lock,
                               [&]
                               {
                                   loaded = liveModule.lock();
                                   return loaded || !moduleRegistered;
                               });
        if (loaded && !std::filesystem::equivalent(loaded->path, path))
            return {};
        for (auto it = devices.begin(); it != devices.end();)
            if (it->second.expired())
                it = devices.erase(it);
            else
                ++it;
        if (auto it = devices.find(device); it != devices.end())
            if (auto existing = it->second.lock())
                return existing;
        if (!loaded)
        {
            loaded = std::make_shared<Module>();
            loaded->path = path;
            // Acquire our own reference to this exact module if NGX/another loader already loaded it.
            // Its original owner keeps responsibility for device-wide shutdown.
            loaded->borrowed = GetModuleHandleExW(0, path.c_str(), &loaded->handle) != FALSE;
            if (!loaded->handle)
                loaded->handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (!loaded->handle)
            {
                const auto error = GetLastError();
                if (error != ERROR_FILE_NOT_FOUND && error != ERROR_PATH_NOT_FOUND && error != ERROR_MOD_NOT_FOUND)
                    LOG_ERROR("NR compatibility: cannot load {} ({})", path.string(), error);
                return {};
            }
            auto symbol = [&](const char* name) { return GetProcAddress(loaded->handle, name); };
            loaded->init = reinterpret_cast<Module::Init>(symbol("NVSDK_NGX_D3D12_Init_Ext"));
            loaded->create = reinterpret_cast<decltype(loaded->create)>(symbol("NVSDK_NGX_D3D12_CreateFeature"));
            loaded->evaluate = reinterpret_cast<decltype(loaded->evaluate)>(symbol("NVSDK_NGX_D3D12_EvaluateFeature"));
            loaded->release = reinterpret_cast<decltype(loaded->release)>(symbol("NVSDK_NGX_D3D12_ReleaseFeature"));
            loaded->shutdown = reinterpret_cast<Module::Shutdown>(symbol("NVSDK_NGX_D3D12_Shutdown1"));
            if (!loaded->init || !loaded->create || !loaded->evaluate || !loaded->release || !loaded->shutdown)
            {
                LOG_ERROR("NR compatibility: {} is missing required NR exports", path.string());
                return {};
            }

            MODULEINFO image {};
            std::vector<RuntimeImports::Slot> slots;
            if (!GetModuleInformation(GetCurrentProcess(), loaded->handle, &image, sizeof(image)) ||
                !RuntimeImports::Find({ static_cast<unsigned char*>(image.lpBaseOfDll), image.SizeOfImage }, slots))
            {
                LOG_ERROR("NR compatibility: invalid runtime import table in {}", path.string());
                return {};
            }
            // Resolve named imports from this build; never use a version-specific address.
            loaded->imports.reserve(slots.size());
            void* previousW = nullptr;
            void* previousA = nullptr;
            for (const auto& slot : slots)
            {
                auto*& previous = slot.wide ? previousW : previousA;
                if (!*slot.address || (previous && previous != *slot.address))
                {
                    LOG_ERROR("NR compatibility: conflicting caller-path import targets in {}", path.string());
                    return {};
                }
                previous = *slot.address;
            }
            if (previousW)
                originalPathW = reinterpret_cast<decltype(&GetModuleFileNameW)>(previousW);
            if (previousA)
                originalPathA = reinterpret_cast<decltype(&GetModuleFileNameA)>(previousA);
            for (const auto& slot : slots)
            {
                auto* previous = slot.wide ? previousW : previousA;
                if (!ReplaceImport(slot.address, previous,
                                   slot.wide ? reinterpret_cast<void*>(&CallerPath)
                                             : reinterpret_cast<void*>(&CallerPathA)))
                {
                    LOG_ERROR("NR compatibility: cannot adapt caller-path import in {}", path.string());
                    return {};
                }
                loaded->imports.push_back({ slot, previous });
            }
            liveModule = loaded;
            loaded->registered = moduleRegistered = true;
            moduleRetired.notify_all();
            LOG_INFO("NR compatibility: {} {} with {} named caller-path imports, no helper DLL",
                     loaded->borrowed ? "retained existing runtime" : "loaded", path.string(), slots.size());
        }

        auto owner = std::shared_ptr<CompatibilityRuntime>(new CompatibilityRuntime());
        owner->module = loaded;
        owner->device = device;
        device->AddRef();
        owner->destroyParameters = destroy;
        std::unique_lock runtimeLock(loaded->mutex);
        // An expired weak owner may still be executing its destructor on another thread.
        loaded->deviceRetired.wait(runtimeLock, [&] { return !loaded->initializedDevices.contains(device); });
        auto result = allocate(&owner->capabilities);
        if (result != NVSDK_NGX_Result_Success || !owner->capabilities)
            return {};
        CallerScope caller;
        const auto writablePath = dataPath.empty() ? std::filesystem::temp_directory_path() : dataPath;
        result = loaded->init(0x24480451ull, writablePath.c_str(), device, 0x15, owner->capabilities);
        LOG_INFO("NR compatibility: Init_Ext result=0x{:08X}", (unsigned) result);
        if (result != NVSDK_NGX_Result_Success)
            return {};
        owner->initialized = true;
        loaded->initializedDevices.insert(device);
        devices[device] = owner;
        return owner;
    }
    catch (const std::exception& error)
    {
        LOG_ERROR("NR compatibility: {}", error.what());
        return {};
    }
}

CompatibilityRuntime::~CompatibilityRuntime()
{
    std::lock_guard lock(module->mutex);
    CallerScope caller;
    if (initialized)
    {
        if (!module->borrowed)
        {
            const auto result = module->shutdown(device);
            LOG_INFO("NR compatibility: Shutdown1 result=0x{:08X}", (unsigned) result);
        }
        module->initializedDevices.erase(device);
        module->deviceRetired.notify_all();
    }
    if (capabilities)
        destroyParameters(capabilities);
    if (device)
        device->Release();
}

NVSDK_NGX_Result CompatibilityRuntime::Create(ID3D12GraphicsCommandList* commands, NVSDK_NGX_Parameter* params,
                                              NVSDK_NGX_Handle** feature)
{
    std::lock_guard lock(module->mutex);
    CallerScope caller;
    return module->create(commands, (NVSDK_NGX_Feature) 18, params, feature);
}

NVSDK_NGX_Result CompatibilityRuntime::Evaluate(ID3D12GraphicsCommandList* commands, const NVSDK_NGX_Handle* feature,
                                                NVSDK_NGX_Parameter* params)
{
    std::lock_guard lock(module->mutex);
    CallerScope caller;
    return module->evaluate(commands, feature, params, nullptr);
}

NVSDK_NGX_Result CompatibilityRuntime::Release(NVSDK_NGX_Handle* feature)
{
    std::lock_guard lock(module->mutex);
    CallerScope caller;
    return module->release(feature);
}
} // namespace DlssNr
