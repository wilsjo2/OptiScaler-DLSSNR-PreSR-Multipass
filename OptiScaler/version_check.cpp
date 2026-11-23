#include "pch.h"

#include "version_check.h"

#include "State.h"
#include "resource.h"

#include <json.hpp>

#include <winhttp.h>

#include <charconv>
#include <optional>
#include <string_view>
#include <thread>
#include <mutex>
#include <format>

namespace
{
struct LatestReleaseInfo
{
    std::string tag;
    std::string url;
    std::string publishedAt;
    feature_version version;
};

#ifdef OPTISCALER_NR_PACKAGE_TAG
constexpr std::string_view packageTag = STRINGIZE(OPTISCALER_NR_PACKAGE_TAG);
constexpr auto packageMarker = "OptiScaler NR package: " STRINGIZE(OPTISCALER_NR_PACKAGE_TAG);
#else
constexpr std::string_view packageTag;
constexpr auto packageMarker = "OptiScaler NR package: untagged";
#endif

std::optional<feature_version> ParseReleaseTag(std::string_view tag)
{
    if (!tag.starts_with('v'))
        return std::nullopt;

    tag.remove_prefix(1);
    feature_version version;
    for (auto* part : { &version.major, &version.minor, &version.patch })
    {
        const char* end = tag.data() + tag.size();
        const auto [next, error] = std::from_chars(tag.data(), end, *part);
        if (error != std::errc {})
            return std::nullopt;
        if (part == &version.patch)
            return next == end ? std::optional(version) : std::nullopt;
        if (next == end || *next != '.')
            return std::nullopt;
        tag.remove_prefix(next - tag.data() + 1);
    }
    return std::nullopt;
}

std::optional<LatestReleaseInfo> FetchLatestRelease(std::string& installedPublishedAt)
{
    HINTERNET session = nullptr;
    HINTERNET connection = nullptr;
    HINTERNET request = nullptr;

    auto cleanup = [&]()
    {
        if (request != nullptr)
        {
            WinHttpCloseHandle(request);
            request = nullptr;
        }
        if (connection != nullptr)
        {
            WinHttpCloseHandle(connection);
            connection = nullptr;
        }
        if (session != nullptr)
        {
            WinHttpCloseHandle(session);
            session = nullptr;
        }
    };

    // Wine is being stoopid and fetch deadlocks somewhere otherwise
    Sleep(1000);

    session = WinHttpOpen(L"OptiScaler Version Check/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                          WINHTTP_NO_PROXY_BYPASS, 0);
    if (session == nullptr)
    {
        LOG_WARN("Version check failed to open WinHTTP session: {}", GetLastError());
        cleanup();
        return std::nullopt;
    }

    WinHttpSetTimeouts(session, 5000, 5000, 5000, 5000);

    connection = WinHttpConnect(session, L"api.github.com", INTERNET_DEFAULT_HTTPS_PORT, 0);
    if (connection == nullptr)
    {
        LOG_WARN("Version check failed to connect: {}", GetLastError());
        cleanup();
        return std::nullopt;
    }

    // The fork also publishes OptiScaler releases as prereleases.
    constexpr auto releasePath = L"/repos/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/releases?per_page=30";
    request = WinHttpOpenRequest(connection, L"GET", releasePath, nullptr, WINHTTP_NO_REFERER,
                                 WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    if (request == nullptr)
    {
        LOG_WARN("Version check failed to open request: {}", GetLastError());
        cleanup();
        return std::nullopt;
    }

    LPCWSTR headers = L"User-Agent: OptiScaler\r\nAccept: application/vnd.github+json\r\nAccept-Encoding: identity\r\n";
    if (!WinHttpSendRequest(request, headers, (DWORD) -1, WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
    {
        LOG_WARN("Version check failed to send request: {}", GetLastError());
        cleanup();
        return std::nullopt;
    }

    if (!WinHttpReceiveResponse(request, nullptr))
    {
        LOG_WARN("Version check failed to receive response: {}", GetLastError());
        cleanup();
        return std::nullopt;
    }

    DWORD statusCode = 0;
    DWORD statusCodeSize = sizeof(statusCode);
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &statusCodeSize, WINHTTP_NO_HEADER_INDEX))
    {
        LOG_WARN("Version check failed to read status code: {}", GetLastError());
        cleanup();
        return std::nullopt;
    }

    if (statusCode != HTTP_STATUS_OK)
    {
        LOG_WARN("Version check returned HTTP status {}", statusCode);
        cleanup();
        return std::nullopt;
    }

    std::string response;
    DWORD available = 0;
    do
    {
        if (!WinHttpQueryDataAvailable(request, &available))
        {
            LOG_WARN("Version check failed querying data availability: {}", GetLastError());
            cleanup();
            return std::nullopt;
        }

        if (available == 0)
            break;

        std::string buffer;
        buffer.resize(available);
        DWORD downloaded = 0;
        if (!WinHttpReadData(request, buffer.data(), available, &downloaded))
        {
            LOG_WARN("Version check failed reading response: {}", GetLastError());
            cleanup();
            return std::nullopt;
        }

        response.append(buffer.data(), downloaded);
    } while (available > 0);

    cleanup();

    try
    {
        const auto releases = nlohmann::json::parse(response);
        if (!releases.is_array())
            return std::nullopt;

        std::optional<LatestReleaseInfo> latest;
        for (const auto& release : releases)
        {
            if (!release.is_object() || release.value("draft", true))
                continue;

            auto tag = release.value("tag_name", std::string {});
            const auto version = ParseReleaseTag(tag);
            if (!version)
                continue;

            auto publishedAt = release.value("published_at", std::string {});
            if (tag == packageTag)
                installedPublishedAt = publishedAt;
            if (!publishedAt.empty() && (!latest || publishedAt > latest->publishedAt))
                latest = LatestReleaseInfo { std::move(tag), release.value("html_url", std::string {}),
                                             std::move(publishedAt), *version };
        }
        return latest;
    }
    catch (const std::exception& ex)
    {
        LOG_WARN("Version check failed to parse response: {}", ex.what());
        return std::nullopt;
    }
}

void FinishVersionCheck()
{
    auto& state = State::Instance();
    std::scoped_lock lock(state.versionCheckMutex);
    state.versionCheckInProgress = false;
    state.versionCheckCompleted = true;
}

void RunVersionCheck(feature_version localVersion)
{
    struct Finalizer
    {
        ~Finalizer() { FinishVersionCheck(); }
    } finalize;

    std::string installedPublishedAt;
    auto release = FetchLatestRelease(installedPublishedAt);
    if (!release.has_value())
    {
        auto& state = State::Instance();
        std::scoped_lock lock(state.versionCheckMutex);
        state.versionCheckError = "Unable to check for updates.";
        state.updateAvailable = false;
        return;
    }

    auto& state = State::Instance();
    // Release v0.8.10 was published after v0.8.91. Use publication order when both releases are known.
    const bool updateAvailable =
        installedPublishedAt.empty() ? release->version > localVersion : release->publishedAt > installedPublishedAt;

    {
        std::scoped_lock lock(state.versionCheckMutex);
        state.latestVersionTag = release->tag;
        state.latestVersionUrl = release->url;
        state.updateAvailable = updateAvailable;
        state.versionCheckError.clear();
    }

    if (updateAvailable)
        LOG_WARN("New OptiScaler release available: {} (current {}).", release->tag, packageTag);
    else
        LOG_INFO("OptiScaler is up to date (current {})", packageTag);
}
} // namespace

const std::string& VersionCheck::CurrentVersionString()
{
    static const std::string version =
        packageTag.empty() ? std::format("{}.{}.{}", VER_MAJOR_VERSION, VER_MINOR_VERSION, VER_HOTFIX_VERSION)
                           : std::string(packageTag.substr(1));
    return version;
}

void VersionCheck::Start()
{
    LOG_DEBUG("{}", packageMarker);
    const auto localVersion = ParseReleaseTag(packageTag);
    if (!localVersion)
        return;

    auto& state = State::Instance();
    {
        std::scoped_lock lock(state.versionCheckMutex);
        if (state.versionCheckInProgress || state.versionCheckCompleted)
            return;

        state.versionCheckInProgress = true;
        state.versionCheckCompleted = false;
        state.updateAvailable = false;
        state.versionCheckError.clear();
        state.latestVersionTag.clear();
        state.latestVersionUrl.clear();
    }

    std::thread(
        [localVersion = *localVersion]()
        {
            try
            {
                RunVersionCheck(localVersion);
            }
            catch (const std::exception& ex)
            {
                LOG_ERROR("Version check failed with exception: {}", ex.what());
                auto& state = State::Instance();
                std::scoped_lock lock(state.versionCheckMutex);
                state.versionCheckError = "Update check failed.";
                state.updateAvailable = false;
            }
            catch (...)
            {
                LOG_ERROR("Version check failed with unknown exception");
                auto& state = State::Instance();
                std::scoped_lock lock(state.versionCheckMutex);
                state.versionCheckError = "Update check failed.";
                state.updateAvailable = false;
            }
        })
        .detach();
}
