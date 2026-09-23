// ============================================================================
// tests/ampere_mfg_payload_pin_seam.h
// Todo 13 of .omo/plans/rtx2030-mfg-integration.md: the payload-pin seam of the smoke harnesses.
//
// The production ladder compares the module it found against the pinned payload
// (AmpereMfgLoader.h, kPayloadExpectedBytes / kPayloadExpectedSha256, from
// vendor/dlssg_sm86/PIN.json) and refuses with the named PayloadDigestMismatch. Those
// constants describe the 30 MB proxy this build bundles - a file no harness can rebuild
// from source - while the harnesses stage stub modules (tests/ampere_mfg_stub_payload.cpp
// and friends) plus deliberately corrupted derivatives of them. Compiling the production
// loader against those fixtures therefore needs exactly one thing replaced: WHICH bytes
// the case treats as the pinned payload.
//
// This header is that replacement. It defines AMPERE_MFG_PAYLOAD_PIN_SEAM, which makes
// AmpereMfgLoader.h leave the pin accessors to this file, and provides them:
// ExpectedPayloadBytes() / ExpectedPayloadSha256() return the size and the lowercase-hex
// digest the case pinned to, read from a file with PinTo(). The production build never
// defines the macro, so the shipped loader has no seam at all and always compares
// against the constants.
//
// The harnesses call PinTo()/PinFromEnvironment() in the CHILD, before Arm():
//   * the failure matrix pins each case to the PRISTINE fixture that case derives from
//     (the active stub, the standby stub, or - for the wrong-digest row - the real
//     PIN.json payload, so the fixture keeps the pinned size and only its digest
//     differs); the parent passes that source in AMPERE_MFG_PIN_FILE, because the child
//     only sees the staged (possibly corrupted) bytes;
//   * the eligibility smoke and the sidecar harness stage the module they pin to, so
//     they pin to that staged module (the eligibility smoke pins the stub it was handed,
//     which is the file the parent staged).
// ============================================================================
#pragma once

// Must be defined before AmpereMfgLoader.h is compiled.
#define AMPERE_MFG_PAYLOAD_PIN_SEAM 1

#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

#include <filesystem>
#include <fstream>
#include <string>

namespace AmpereMfgLoader
{
// The pin the ladder compares against while the seam is active. 0/"" means "no pin was handed in yet", and 0
// is never a passing size: a case that stages a payload calls PinTo()/PinFromEnvironment() before Arm().
inline unsigned long long g_expectedPayloadBytes = 0;
inline std::string g_expectedPayloadSha256;

inline unsigned long long ExpectedPayloadBytes()
{
    return g_expectedPayloadBytes;
}

inline std::string ExpectedPayloadSha256()
{
    return g_expectedPayloadSha256;
}
} // namespace AmpereMfgLoader

namespace AmpereMfgPinSeam
{
// The file's lowercase-hex SHA-256, streamed, through the same CNG provider the loader uses.
inline std::string Sha256Hex(const std::filesystem::path& path, bool& ok)
{
    ok = false;

    std::ifstream stream(path, std::ios::binary);

    if (!stream.is_open())
        return {};

    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;

    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        return {};

    if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0)
    {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        return {};
    }

    unsigned char buffer[64 * 1024] {};
    bool complete = true;

    while (stream.read(reinterpret_cast<char*>(buffer), sizeof(buffer)) || stream.gcount() > 0)
    {
        if (BCryptHashData(hash, buffer, static_cast<ULONG>(stream.gcount()), 0) < 0)
        {
            complete = false;
            break;
        }
    }

    unsigned char raw[32] {};
    complete = complete && !stream.bad() && BCryptFinishHash(hash, raw, sizeof(raw), 0) >= 0;

    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);

    if (!complete)
        return {};

    static constexpr char kHex[] = "0123456789abcdef";
    std::string digest;
    digest.reserve(sizeof(raw) * 2);

    for (const unsigned char byte : raw)
    {
        digest.push_back(kHex[byte >> 4]);
        digest.push_back(kHex[byte & 0x0F]);
    }

    ok = true;
    return digest;
}

// Pins the ladder to the bytes at `path`: their size and digest. False when the file cannot be read.
inline bool PinTo(const std::filesystem::path& path)
{
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);

    if (error)
        return false;

    bool ok = false;
    const std::string digest = Sha256Hex(path, ok);

    if (!ok)
        return false;

    AmpereMfgLoader::g_expectedPayloadBytes = bytes;
    AmpereMfgLoader::g_expectedPayloadSha256 = digest;
    return true;
}

// Pins the ladder to the file named by `variable` (the failure matrix's per-case pin source).
inline bool PinFromEnvironment(const wchar_t* variable = L"AMPERE_MFG_PIN_FILE")
{
    wchar_t buffer[MAX_PATH * 4] = {};
    const DWORD length = GetEnvironmentVariableW(variable, buffer, MAX_PATH * 4);

    if (length == 0 || length >= MAX_PATH * 4)
        return false;

    return PinTo(std::filesystem::path(buffer));
}
} // namespace AmpereMfgPinSeam
