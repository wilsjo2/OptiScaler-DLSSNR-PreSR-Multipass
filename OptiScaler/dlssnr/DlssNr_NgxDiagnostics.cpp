#include "pch.h"
#include "DlssNr_NgxDiagnostics.h"
#include <Config.h>
#include <State.h>
#include <Util.h>
#include <proxies/NVNGX_Proxy.h>
#include <atomic>
#include <mutex>
#include <fstream>
#include <set>
#include <map>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

namespace DlssNr::NgxDiagnostics
{
namespace
{
std::atomic_uint active { 0 }, lines { 0 };
std::mutex sinkMutex;
NVSDK_NGX_LoggingInfo original {};
void NVSDK_CONV Callback(const char* message, NVSDK_NGX_Logging_Level level, NVSDK_NGX_Feature feature)
{
    if (!message)
        return;
    // NVIDIA may report feature creation from a worker thread. Bound each capture window.
    if (active.load() && lines.fetch_add(1) < 512)
        LOG_INFO("NR diagnostic NGX [feature={} level={} thread={}]: {}", (unsigned) feature, (unsigned) level,
                 GetCurrentThreadId(), message);
    NVSDK_NGX_LoggingInfo sink;
    {
        std::lock_guard lock(sinkMutex);
        sink = original;
    }
    static thread_local bool forwarding = false;
    if (!forwarding && sink.LoggingCallback && sink.LoggingCallback != Callback &&
        sink.MinimumLoggingLevel != NVSDK_NGX_LOGGING_LEVEL_OFF && level <= sink.MinimumLoggingLevel)
    {
        forwarding = true;
        sink.LoggingCallback(message, level, feature);
        forwarding = false;
    }
}
std::string Sha256(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return "unreadable";
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        return "unavailable";
    if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0)
    {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        return "unavailable";
    }
    unsigned char buffer[65536], digest[32];
    bool ok = true;
    while (file.read((char*) buffer, sizeof(buffer)) || file.gcount())
        if (BCryptHashData(hash, buffer, (ULONG) file.gcount(), 0) < 0)
        {
            ok = false;
            break;
        }
    ok = ok && !file.bad() && BCryptFinishHash(hash, digest, sizeof(digest), 0) >= 0;
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    if (!ok)
        return "unavailable";
    std::string result;
    for (auto byte : digest)
        result += std::format("{:02X}", byte);
    return result;
}
void FileReport(const std::filesystem::path& path, const char* kind, bool hashFile)
{
    std::error_code error;
    const bool exists = std::filesystem::is_regular_file(path, error);
    if (!exists)
    {
        LOG_INFO("NR diagnostic {}: {} [not found, filesystem error={} ]", kind, path.string(), error.value());
        return;
    }
    // Hash unchanged files only once per process, including across Retry and multiple passes.
    struct CachedHash
    {
        uintmax_t size;
        std::filesystem::file_time_type modified;
        std::string hash;
    };
    static std::mutex cacheMutex;
    static std::map<std::wstring, CachedHash> cache;
    const auto size = std::filesystem::file_size(path, error);
    const auto modified = std::filesystem::last_write_time(path, error);
    std::string digest = "not requested";
    if (hashFile)
    {
        std::lock_guard lock(cacheMutex);
        auto key = Util::ToLower(path.lexically_normal().wstring());
        auto found = cache.find(key);
        if (found == cache.end() || found->second.size != size || found->second.modified != modified)
            found = cache.insert_or_assign(key, CachedHash { size, modified, Sha256(path) }).first;
        digest = found->second.hash;
    }
    version_t version {};
    const bool hasVersion = Util::GetFileVersion(path.wstring(), &version);
    LOG_INFO("NR diagnostic {}: {} size={} version={} SHA256={}", kind, path.string(), size,
             hasVersion ? std::format("{}.{}.{}", version.major, version.minor, version.patch) : "unknown", digest);
}
} // namespace
void Install(NVSDK_NGX_LoggingInfo& logging)
{
    std::lock_guard lock(sinkMutex);
    if (logging.LoggingCallback != Callback)
        original = logging;
    logging.LoggingCallback = Callback;
    logging.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
    // Leave DisableOtherLoggingSinks as the game supplied it.
}
Scope::Scope()
{
    if (active.fetch_add(1) == 0)
        lines = 0;
}
Scope::~Scope()
{
    if (active.fetch_sub(1) == 1 && lines.load() > 512)
        LOG_INFO("NR diagnostic: NGX capture limited to 512 lines for this initialization window");
}
void RuntimeReport(ID3D12GraphicsCommandList* commands, ID3D12Device* device, const char* phase)
{
    // Called only for a model creation attempt, never from normal EvaluateFeature.
    try
    {
        LOG_INFO("NR diagnostic {}: device={} commands={} thread={}", phase, (void*) device, (void*) commands,
                 GetCurrentThreadId());
        if (device)
        {
            const auto luid = device->GetAdapterLuid();
            Microsoft::WRL::ComPtr<ID3D12Device> commandDevice;
            const auto hr = commands ? commands->GetDevice(IID_PPV_ARGS(&commandDevice)) : E_POINTER;
            const auto commandLuid = commandDevice ? commandDevice->GetAdapterLuid() : LUID {};
            LOG_INFO("NR diagnostic: adapter LUID={:08X}:{:08X}, device status=0x{:08X}, command device={} "
                     "query=0x{:08X}, command adapter={:08X}:{:08X}, nodes={}",
                     (unsigned) luid.HighPart, luid.LowPart, (unsigned) device->GetDeviceRemovedReason(),
                     (void*) commandDevice.Get(), (unsigned) hr, (unsigned) commandLuid.HighPart, commandLuid.LowPart,
                     device->GetNodeCount());
        }
        FileReport(NVNGXProxy::NVNGXModule_Path(), "driver dispatcher", false);
        std::set<std::filesystem::path> paths;
        for (const auto& path : State::Instance().NVNGX_FeatureInfo_Paths)
        {
            LOG_INFO("NR diagnostic NGX search path: {}", wstring_to_string(path));
            paths.insert(std::filesystem::path(path) / L"nvngx_dlssnr.dll");
        }
        paths.insert(Util::ExePath().parent_path() / L"nvngx_dlssnr.dll");
        paths.insert(Util::DllPath().parent_path() / L"nvngx_dlssnr.dll");
        if (Config::Instance()->MainDllPath.has_value())
            paths.insert(std::filesystem::path(Config::Instance()->MainDllPath.value()) / L"nvngx_dlssnr.dll");
        // A file candidate is not proof that NGX selected/accepted it.
        for (const auto& path : paths)
            FileReport(path, "runtime candidate", true);
        if (const auto module = GetModuleHandleW(L"nvngx_dlssnr.dll"))
        {
            wchar_t path[32768] {};
            const auto length = GetModuleFileNameW(module, path, (DWORD) std::size(path));
            if (length && length < std::size(path))
                FileReport(path, "loaded runtime", true);
            else
                LOG_INFO("NR diagnostic: runtime module found but its path is unavailable");
        }
        else
            LOG_INFO("NR diagnostic: nvngx_dlssnr.dll is not present in the process module list at {}", phase);
    }
    catch (const std::exception& e)
    {
        LOG_INFO("NR diagnostic: runtime inspection unavailable: {}", e.what());
    }
}
} // namespace DlssNr::NgxDiagnostics
