#pragma once

// Recognising an NVIDIA DLSS-G provider (the nvngx_dlssg snippet), wherever it was loaded from.
// Header-only and free of OptiScaler headers so tests/mfg_provider_smoke.cpp can compile it alone.
//
// The lookup by file name misses the copy NVIDIA's driver stores under
//   C:\ProgramData\NVIDIA\NGX\models\dlssg\versions\<n>\files\<hash>.bin
// and a game that renamed its snippet. Both are recognised here. Recognising a module never patches it:
// the signatures in MfgUnlock.cpp stay the only gate.
//
// Adapted from KleberMotta/OptiScaler-DLSS5-MFG-RTX40 74c3bac (mfgunlock/, MIT, from the RenoDX MFG
// Unlock addon by Dreamt / mavismmg). See Licenses/MFGUnlock_LICENSE.txt.

#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace MfgUnlock::Provider
{
// The NGX parameter the unlock is about, present in the DLSS-G snippet builds checked (310.8 and 310.9).
// Used for a module whose path says nothing. The kernel name "dlfg_kernel" would not do: 310.9 renamed
// every kernel and no longer carries it.
inline constexpr std::string_view kMarker = "DLSSG.MultiFrameCountMax";

// Lower-cased, backslash-only, doubled separators collapsed, so the two spellings the loader hands us
// ("models//dlssg" and "models\dlssg") compare equal.
inline std::wstring NormalisePath(std::wstring_view path)
{
    std::wstring out;
    out.reserve(path.size());

    for (wchar_t c : path)
    {
        if (c == L'/')
            c = L'\\';
        else if (c >= L'A' && c <= L'Z')
            c = static_cast<wchar_t>(c - L'A' + L'a');

        if (c == L'\\' && !out.empty() && out.back() == L'\\')
            continue;

        out.push_back(c);
    }

    return out;
}

// The game's own nvngx_dlssg.dll, or anything in the driver's DLSS-G OTA store.
inline bool IsProviderPath(std::wstring_view path)
{
    const std::wstring normalised = NormalisePath(path);
    const std::wstring_view view = normalised;

    const auto slash = view.find_last_of(L'\\');
    const auto name = slash == std::wstring_view::npos ? view : view.substr(slash + 1);

    return name == L"nvngx_dlssg.dll" || view.find(L"\\models\\dlssg\\") != std::wstring_view::npos;
}

// Calls fn(data, size, characteristics) for every section of a mapped 64-bit PE image, using only the
// initialised part of the section. Sections that claim to run past SizeOfImage are skipped rather than
// trusted. Returns false when the image is not a 64-bit PE, or when it is unmapped while this runs.
// `imageSizeOut`, if given, receives SizeOfImage before the first call to fn.
template <typename Fn> bool ForEachSection(void* image, Fn&& fn, size_t* imageSizeOut = nullptr)
{
    if (image == nullptr)
        return false;

    __try
    {
        auto* base = static_cast<uint8_t*>(image);
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);

        if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0)
            return false;

        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);

        if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
            return false;

        const size_t imageSize = nt->OptionalHeader.SizeOfImage;
        const auto* section = IMAGE_FIRST_SECTION(nt);

        if (imageSizeOut != nullptr)
            *imageSizeOut = imageSize;

        for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section)
        {
            const size_t start = section->VirtualAddress;
            const size_t size = section->Misc.VirtualSize;

            if (start > imageSize || size > imageSize - start)
                continue;

            fn(base + start, size, static_cast<DWORD>(section->Characteristics));
        }
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        return false;
    }

    return true;
}

// Whether a mapped PE image holds `needle` inside the initialised part of one of its readable sections.
inline bool ImageContains(const void* image, std::string_view needle)
{
    if (needle.empty())
        return false;

    bool found = false;

    ForEachSection(const_cast<void*>(image),
                   [&](const uint8_t* data, size_t size, DWORD characteristics)
                   {
                       if (found || !(characteristics & IMAGE_SCN_MEM_READ) || size < needle.size())
                           return;

                       found = std::search(data, data + size, needle.begin(), needle.end()) != data + size;
                   });

    return found;
}
} // namespace MfgUnlock::Provider
