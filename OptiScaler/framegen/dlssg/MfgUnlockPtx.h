#pragma once

// Temporal (midpoint) correction of the DLSS-G interpolation kernel on Ada, by rewriting its PTX. The
// alternative to retargeting the Blackwell image (MfgUnlock.cpp, RewriteBlackwellKernels). Header-only and
// free of OptiScaler headers so tests/mfg_ptx_smoke.cpp can compile it alone.
//
// THE PROBLEM. With the frame-count gates open, 3x and 4x produce the right number of frames but not the
// right content: the sm_89 build of Kernel_EstimateIntermMvecsScatter blends the two source frames with a
// compiled-in weight of 0.5 (104 `mul.ftz.f32 ..., 0f3F000000;` in its PTX), so every generated frame
// lands at the midpoint.
//
// THE FIX. Rewrite that PTX so the weight comes from the kernel's own temporal parameter:
//
//     ld.param.f32 %f134, [<entry>_param_0+32];   // t for this frame
//     mov.f32      %f135, 0f3F800000;              // 1.0
//     sub.ftz.f32  %f136, %f135, %f134;            // 1.0 - t
//
// injected after the unique `$L__BB0_3:` label, then the 104 constants replaced with %f136 (first half,
// current -> previous) and %f134 (second half, previous -> current).
//
// The fatbin also carries a precompiled sm_89 cubin, which the driver would load in preference to the
// edited PTX. So the rebuilt fatbin is truncated after the sm_89 PTX entry, dropping the cubin and forcing
// the JIT path, and the PTX entry is re-emitted uncompressed (flags 0x41). The rebuild is placed in new
// memory and every descriptor slot that pointed at the original fatbin is redirected to it.
//
// EXACT PROFILES ONLY. The PTX is identified by its size, entry name and register declaration. A build
// that matches none of the known profiles is left alone; that is the expected outcome for a DLSS-G version
// nobody has looked at.
//
// Technique: dashdogy's RTX40MFG-Unlock (MIT, Copyright 2026 Michael Robles). This code is adapted from
// KleberMotta/OptiScaler-DLSS5-MFG-RTX40 74c3bac (mfgunlock/MfgUnlock_Midpoint.h, MIT, from the RenoDX MFG
// Unlock addon by Dreamt / mavismmg), with bounds checks added on every read of the module and the fatbin.
// See Licenses/MFGUnlock_LICENSE.txt.

#include "MfgUnlockProvider.h"

#include <cstring>
#include <format>
#include <string>
#include <vector>

namespace MfgUnlock::Ptx
{
inline constexpr uint32_t kFatbinMagic = 0xBA55ED50u;
inline constexpr size_t kOuterHeader = 16;
inline constexpr uint32_t kPtxKind = 1;
inline constexpr uint32_t kAdaArch = 89;
inline constexpr uint64_t kUncompressedFlags = 0x41;
inline constexpr size_t kEntryHeaderMin = 64;

// NVIDIA renamed every kernel in 310.9, but the temporal program itself is otherwise unchanged: the PTX
// grew by exactly the accumulated symbol-name delta. Separate, exact profiles, so an unrelated provider
// cannot be admitted because it happens to contain a similar midpoint sequence.
struct TemporalProfile
{
    size_t ptxBytes;
    const char* entryName;
    const char* descriptorName;
    ptrdiff_t entryNameOffset;      // from the descriptor slot to the pointer to the entry name
    ptrdiff_t descriptorNameOffset; // and to the pointer to the descriptor name
};

inline constexpr TemporalProfile kProfiles[] = {
    { 99362, "main_kernel", "dlfg_kernel", 0x10, 0x28 },
    { 99626, "Kernel_EstimateIntermMvecsScatter", "EstimateIntermMvecsScatter", 0x10, -0x08 },
};

inline constexpr size_t kExpectedMidpoints = 104;
inline constexpr char kJoinLabel[] = "$L__BB0_3:";
inline constexpr char kMidpointBits[] = "0f3F000000";
inline constexpr char kMulPrefix[] = "mul.ftz.f32 ";
inline constexpr char kCurrToPrev[] = "%f136";
inline constexpr char kPrevToCurr[] = "%f134";
inline constexpr size_t kMaxPtxBytes = 8u << 20;

inline uint16_t ReadU16(const uint8_t* p)
{
    uint16_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint32_t ReadU32(const uint8_t* p)
{
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint64_t ReadU64(const uint8_t* p)
{
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Plain LZ4 block format: token high nibble = literal run, low nibble = match length - 4, 16-bit
// little-endian back offset, 0xFF continuation bytes.
inline bool Lz4BlockDecompress(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstSize)
{
    size_t in = 0;
    size_t out = 0;

    while (in < srcSize)
    {
        const uint8_t token = src[in++];
        size_t literals = token >> 4;

        if (literals == 15)
        {
            uint8_t ext = 0;
            do
            {
                if (in >= srcSize)
                    return false;
                ext = src[in++];
                literals += ext;
            } while (ext == 0xFF);
        }

        if (literals > srcSize - in || literals > dstSize - out)
            return false;

        std::memcpy(dst + out, src + in, literals);
        in += literals;
        out += literals;

        if (in == srcSize)
            break;

        if (srcSize - in < 2)
            return false;

        const size_t back = static_cast<size_t>(src[in]) | (static_cast<size_t>(src[in + 1]) << 8);
        in += 2;

        if (back == 0 || back > out)
            return false;

        size_t match = 4 + (token & 0x0F);

        if ((token & 0x0F) == 15)
        {
            uint8_t ext = 0;
            do
            {
                if (in >= srcSize)
                    return false;
                ext = src[in++];
                match += ext;
            } while (ext == 0xFF);
        }

        if (match > dstSize - out)
            return false;

        for (size_t i = 0; i < match; ++i)
            dst[out + i] = dst[out + i - back];

        out += match;
    }

    return in == srcSize && out == dstSize;
}

// Locates the sm_89 PTX entry inside a fatbin by walking its entry list rather than trusting fixed offsets.
inline bool FindAdaPtxEntry(const uint8_t* fat, size_t fatSize, size_t& entryOffset)
{
    if (fatSize < kOuterHeader || ReadU32(fat) != kFatbinMagic || ReadU16(fat + 6) != kOuterHeader)
        return false;

    const uint64_t declared = ReadU64(fat + 8);

    if (declared != fatSize - kOuterHeader)
        return false;

    size_t p = kOuterHeader;

    while (fatSize - p >= kEntryHeaderMin)
    {
        const uint32_t kind = ReadU16(fat + p);
        const uint32_t hdr = ReadU32(fat + p + 4);
        const uint64_t payload = ReadU64(fat + p + 8);

        if (hdr < kEntryHeaderMin || payload == 0 || hdr > fatSize - p || payload > fatSize - p - hdr)
            return false;

        if (kind == kPtxKind && ReadU32(fat + p + 28) == kAdaArch)
        {
            entryOffset = p;
            return true;
        }

        p += hdr + static_cast<size_t>(payload);
    }

    return false;
}

// The profile whose PTX size the sm_89 entry carries, or nullptr. Kernel names alone do not identify the
// temporal program, so the exact size is part of the identity.
inline const TemporalProfile* FindProfile(const uint8_t* fat, size_t fatSize)
{
    size_t entry = 0;

    if (!FindAdaPtxEntry(fat, fatSize, entry))
        return nullptr;

    const uint64_t raw = ReadU64(fat + entry + 56);

    for (const auto& profile : kProfiles)
        if (raw == profile.ptxBytes)
            return &profile;

    return nullptr;
}

// Decompresses the Ada PTX, rewrites the blend weights, and re-emits a fatbin that ends after it.
inline bool BuildTemporalFatbin(const uint8_t* fat, size_t fatSize, const TemporalProfile& profile,
                                std::vector<uint8_t>& out, std::string& why)
{
    size_t entry = 0;

    if (!FindAdaPtxEntry(fat, fatSize, entry))
    {
        why = "no sm_89 PTX entry";
        return false;
    }

    const uint32_t hdr = ReadU32(fat + entry + 4);
    const uint64_t payload = ReadU64(fat + entry + 8);
    const uint32_t compressed = ReadU32(fat + entry + 16);
    const uint64_t raw = ReadU64(fat + entry + 56);

    if (compressed == 0 || raw == 0 || raw > kMaxPtxBytes || compressed > payload)
    {
        why = "PTX entry is not compressed as expected";
        return false;
    }

    if (raw != profile.ptxBytes)
    {
        why = std::format("PTX is {} bytes, expected {}", raw, profile.ptxBytes);
        return false;
    }

    std::vector<uint8_t> ptx(static_cast<size_t>(raw));

    if (!Lz4BlockDecompress(fat + entry + hdr, compressed, ptx.data(), ptx.size()))
    {
        why = "LZ4 decompression failed";
        return false;
    }

    const std::string entrySignature = std::string(".entry ") + profile.entryName + "(";
    const std::string parameterName = std::string(profile.entryName) + "_param_0";
    const std::string parameterSignature = std::string(".param .align 8 .b8 ") + parameterName + "[144]";
    const std::string ptxText(reinterpret_cast<const char*>(ptx.data()), ptx.size());

    if (ptxText.find(entrySignature) == std::string::npos || ptxText.find(parameterSignature) == std::string::npos ||
        ptxText.find(".reg .f32 %f<1362>;") == std::string::npos)
    {
        why = "temporal kernel signature changed";
        return false;
    }

    // The join label must be unique, or this is not the kernel it appears to be.
    const char* begin = reinterpret_cast<const char*>(ptx.data());
    const size_t n = ptx.size();
    const size_t labelLen = sizeof(kJoinLabel) - 1;
    size_t label = SIZE_MAX;

    for (size_t i = 0; i + labelLen <= n; ++i)
    {
        if (std::memcmp(begin + i, kJoinLabel, labelLen) != 0)
            continue;

        if (label != SIZE_MAX)
        {
            why = "join label is not unique";
            return false;
        }

        label = i;
    }

    if (label == SIZE_MAX)
    {
        why = "join label not found";
        return false;
    }

    size_t insertion = label + labelLen;

    while (insertion < n && begin[insertion] != '\n')
        ++insertion;

    if (insertion >= n)
    {
        why = "join label has no line end";
        return false;
    }

    ++insertion;

    // Every midpoint constant that terminates a mul.ftz.f32 line.
    const size_t midLen = sizeof(kMidpointBits) - 1;
    const size_t mulLen = sizeof(kMulPrefix) - 1;
    std::vector<size_t> marks;
    marks.reserve(kExpectedMidpoints);

    for (size_t i = 0; i + midLen < n; ++i)
    {
        if (std::memcmp(begin + i, kMidpointBits, midLen) != 0 || begin[i + midLen] != ';')
            continue;

        size_t line = i;

        while (line > 0 && begin[line - 1] != '\n')
            --line;

        if (i - line < mulLen || std::memcmp(begin + line, kMulPrefix, mulLen) != 0)
            continue;

        marks.push_back(i);
    }

    if (marks.size() != kExpectedMidpoints)
    {
        why = std::format("found {} midpoint multiplies, expected {}", marks.size(), kExpectedMidpoints);
        return false;
    }

    if (marks.front() <= insertion)
    {
        why = "first midpoint precedes the injection point";
        return false;
    }

    const std::string temporalInput = "ld.param.f32 %f134, [" + parameterName +
                                      "+32];\r\n"
                                      "mov.f32 %f135, 0f3F800000;\r\n"
                                      "sub.ftz.f32 %f136, %f135, %f134;\r\n";

    std::vector<uint8_t> patched;
    patched.reserve(n + temporalInput.size());

    auto append = [&patched](const void* p, size_t bytes)
    {
        const auto* b = static_cast<const uint8_t*>(p);
        patched.insert(patched.end(), b, b + bytes);
    };

    append(ptx.data(), insertion);
    append(temporalInput.data(), temporalInput.size());

    size_t src = insertion;
    const size_t half = kExpectedMidpoints / 2;

    for (size_t i = 0; i < marks.size(); ++i)
    {
        append(ptx.data() + src, marks[i] - src);
        append(i < half ? kCurrToPrev : kPrevToCurr, 5);
        src = marks[i] + midLen;
    }

    append(ptx.data() + src, n - src);

    const size_t padded = (patched.size() + 7) & ~size_t { 7 };
    const size_t finalSize = entry + hdr + padded;

    out.assign(fat, fat + entry + hdr);
    out.resize(finalSize, 0);
    std::memcpy(out.data() + entry + hdr, patched.data(), patched.size());

    const uint64_t payload64 = padded;
    const uint32_t zero32 = 0;
    const uint64_t zero64 = 0;
    std::memcpy(out.data() + entry + 8, &payload64, sizeof(payload64));
    std::memcpy(out.data() + entry + 16, &zero32, sizeof(zero32));
    std::memcpy(out.data() + entry + 40, &kUncompressedFlags, sizeof(kUncompressedFlags));
    std::memcpy(out.data() + entry + 56, &zero64, sizeof(zero64));

    const uint64_t outer = finalSize - kOuterHeader;
    std::memcpy(out.data() + 8, &outer, sizeof(outer));

    return true;
}

// A NUL-terminated copy of `expected` at `value`, all inside [start, start + imageSize).
inline bool PointsToCString(uintptr_t start, size_t imageSize, uint64_t value, const char* expected)
{
    const size_t len = std::strlen(expected);

    if (value < start || value >= start + imageSize || len + 1 > start + imageSize - value)
        return false;

    return std::memcmp(reinterpret_cast<const char*>(value), expected, len + 1) == 0;
}

// The 8-byte field `displacement` bytes from `slot`, if it lies wholly inside the image.
inline bool ReadRelativePointer(uintptr_t start, size_t imageSize, uintptr_t slot, ptrdiff_t displacement,
                                uint64_t& value)
{
    const uintptr_t end = start + imageSize;
    uintptr_t field = slot;

    if (displacement >= 0)
    {
        if (static_cast<uintptr_t>(displacement) > end - slot)
            return false;
        field = slot + static_cast<uintptr_t>(displacement);
    }
    else
    {
        const uintptr_t distance = static_cast<uintptr_t>(-(displacement + 1)) + 1;
        if (distance > slot - start)
            return false;
        field = slot - distance;
    }

    if (field < start || field > end || sizeof(value) > end - field)
        return false;

    std::memcpy(&value, reinterpret_cast<const void*>(field), sizeof(value));
    return true;
}

struct Result
{
    size_t redirected = 0;   // descriptor slots now pointing at the rebuilt fatbin
    size_t originalSize = 0;
    size_t rebuiltSize = 0;
    std::string detail;      // what was done, or why nothing was
};

// Replaces the temporal-kernel fatbin in every descriptor that references it. The module carries several
// identical descriptors in separate tables and there is no telling which the runtime will pick, so all of
// them are redirected. The rebuild lives in new memory for the rest of the process.
inline bool Apply(void* image, Result& result)
{
    result = {};

    auto* base = static_cast<uint8_t*>(image);
    const uintptr_t start = reinterpret_cast<uintptr_t>(base);
    size_t imageSize = 0;

    std::vector<uint64_t*> slots;
    const uint8_t* fat = nullptr;
    size_t fatSize = 0;
    const TemporalProfile* selected = nullptr;

    const bool valid = Provider::ForEachSection(
        image,
        [&](uint8_t* data, size_t size, DWORD characteristics)
        {
            if (!(characteristics & IMAGE_SCN_MEM_READ) || (characteristics & IMAGE_SCN_MEM_EXECUTE))
                return;

            for (size_t off = 0; off + sizeof(uint64_t) <= size; off += sizeof(uint64_t))
            {
                uint64_t value = 0;
                std::memcpy(&value, data + off, sizeof(value));

                // A fatbin has at least an outer header and one entry header, all inside the image.
                if (value < start || value >= start + imageSize || start + imageSize - value < 1024)
                    continue;

                const auto* candidate = reinterpret_cast<const uint8_t*>(value);

                if (ReadU32(candidate) != kFatbinMagic)
                    continue;

                const TemporalProfile* named = nullptr;

                for (const auto& profile : kProfiles)
                {
                    uint64_t entryName = 0;
                    uint64_t descName = 0;

                    if (!ReadRelativePointer(start, imageSize, reinterpret_cast<uintptr_t>(data + off),
                                             profile.entryNameOffset, entryName) ||
                        !ReadRelativePointer(start, imageSize, reinterpret_cast<uintptr_t>(data + off),
                                             profile.descriptorNameOffset, descName))
                        continue;

                    if (PointsToCString(start, imageSize, entryName, profile.entryName) &&
                        PointsToCString(start, imageSize, descName, profile.descriptorName))
                    {
                        named = &profile;
                        break;
                    }
                }

                if (named == nullptr)
                    continue;

                const uint64_t declared = ReadU64(candidate + 8);

                if (declared > start + imageSize - value - kOuterHeader)
                    continue;

                const size_t total = static_cast<size_t>(declared) + kOuterHeader;

                if (total < 1024 || total > (16u << 20))
                    continue;

                const TemporalProfile* byPtx = FindProfile(candidate, total);

                if (byPtx == nullptr || byPtx != named)
                    continue;

                if (fat == nullptr)
                {
                    fat = candidate;
                    fatSize = total;
                    selected = byPtx;
                }
                else if (candidate != fat)
                {
                    continue; // a second temporal-looking kernel; leave it alone
                }

                slots.push_back(reinterpret_cast<uint64_t*>(data + off));
            }
        },
        &imageSize);

    if (!valid)
    {
        result.detail = "not a readable module image";
        return false;
    }

    if (fat == nullptr || slots.empty() || selected == nullptr)
    {
        result.detail = "no supported temporal-kernel descriptor found";
        return false;
    }

    std::vector<uint8_t> rebuilt;
    std::string why;

    if (!BuildTemporalFatbin(fat, fatSize, *selected, rebuilt, why))
    {
        result.detail = why;
        return false;
    }

    void* mem = VirtualAlloc(nullptr, rebuilt.size(), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (mem == nullptr)
    {
        result.detail = "allocation failed";
        return false;
    }

    std::memcpy(mem, rebuilt.data(), rebuilt.size());

    for (uint64_t* slot : slots)
    {
        DWORD oldProtect = 0;

        if (!VirtualProtect(slot, sizeof(uint64_t), PAGE_READWRITE, &oldProtect))
            continue;

        *slot = reinterpret_cast<uint64_t>(mem);

        DWORD ignored = 0;
        VirtualProtect(slot, sizeof(uint64_t), oldProtect, &ignored);
        ++result.redirected;
    }

    if (result.redirected == 0)
    {
        VirtualFree(mem, 0, MEM_RELEASE);
        result.detail = "no descriptor slot was writable";
        return false;
    }

    result.originalSize = fatSize;
    result.rebuiltSize = rebuilt.size();
    result.detail = std::format("redirected {} {} descriptor(s) from a {}-byte fatbin to a {}-byte rebuild",
                                result.redirected, selected->descriptorName, fatSize, rebuilt.size());
    return true;
}
} // namespace MfgUnlock::Ptx
