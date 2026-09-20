#pragma once

// Forcing the Streamline DLSS-G plugin (sl.dlss_g) onto software frame pacing. Header-only and free of
// OptiScaler headers so tests/mfg_flipmeter_smoke.cpp can compile it alone.
//
// Asking for more than one generated frame while hardware flip metering is on can freeze presentation. The
// plugin has a fallback for when the flip-metering DLL is missing: it logs "FG1 DLL has been detected" and
// writes one byte in its context to say "use software pacing (RSYNC)". Where a build takes that path is
// not something to hard-code:
//
//  1. The byte's offset and its polarity differ between builds. One build clears [ctx+0x38bc] to mean
//     "flip metering off", another sets [ctx+0x44f0] to 1 to mean the same thing.
//  2. The other places that write the byte store either an immediate or a runtime value.
//
// So everything is derived from the binary: find the string, find the one piece of code that logs it, and
// read the (offset, value) the fallback itself writes. That pair IS the wanted state, whatever its polarity.
// Then every other store to that offset is pinned to the same value. Two encodings appear:
//
//     C6 /0 disp32 imm8   mov byte ptr [reg+disp32], imm8    (7 bytes)  the immediate is rewritten
//     40 88 /r disp32     mov byte ptr [reg+disp32], reg8    (7 bytes)  rewritten into the C6 form
//
// The second stores a runtime value, so there is no immediate to change. Its REX prefix is present only to
// name a byte register (spl/bpl/sil/dil), which makes it exactly seven bytes, the length of the C6 form
// with the same base register. That equivalence is the only reason it can be patched in place, so it is the
// only register store handled: a bare 88 /r disp32 is six bytes and a REX.B one needs eight.
//
// Nothing is patched unless the whole derivation is unambiguous: one marker string, one code reference to
// it, and fallback stores in its window that all agree.
//
// Adapted from KleberMotta/OptiScaler-DLSS5-MFG-RTX40 74c3bac (mfgunlock/, MIT, from the RenoDX MFG Unlock
// addon by Dreamt / mavismmg), with the uniqueness, agreement and overlap checks added. See
// Licenses/MFGUnlock_LICENSE.txt.

#include "MfgUnlockProvider.h"

#include <cstring>
#include <vector>

namespace MfgUnlock::Flip
{
inline constexpr std::string_view kMarker = "FG1 DLL has been detected";

inline constexpr size_t kWindow = 0x200;      // how far after the reference the fallback store is looked for
inline constexpr uint32_t kMinField = 0x100;  // the context offset must lie strictly between these
inline constexpr uint32_t kMaxField = 0x20000;
inline constexpr size_t kStoreLength = 7;
inline constexpr size_t kMaxSites = 16;

struct Site
{
    uint8_t* address = nullptr;
    uint8_t original[kStoreLength] = {};
    uint8_t replacement[kStoreLength] = {};
    bool fromRegister = false; // the 40 88 form, rewritten whole; otherwise only an immediate changes
};

struct Plan
{
    uint32_t field = 0; // offset in the plugin's context
    uint8_t value = 0;  // what the fallback writes there
    std::vector<Site> sites;
};

enum class FindResult
{
    Found,
    BadImage,
    NoMarker,
    AmbiguousMarker,
    NoReference,
    AmbiguousReference,
    NoFallback,
    ConflictingFallback,
    NothingToPatch,
    TooManySites,
    OverlappingSites,
};

// Text for the overlay and the log.
inline const char* Describe(FindResult result)
{
    switch (result)
    {
    case FindResult::Found:
        return "found";
    case FindResult::BadImage:
        return "not a readable module image";
    case FindResult::NoMarker:
        return "the fallback message is not in this plugin";
    case FindResult::AmbiguousMarker:
        return "the fallback message appears more than once";
    case FindResult::NoReference:
        return "nothing in the plugin refers to the fallback message";
    case FindResult::AmbiguousReference:
        return "more than one piece of code refers to the fallback message";
    case FindResult::NoFallback:
        return "could not read the fallback's flip-metering state";
    case FindResult::ConflictingFallback:
        return "the fallback writes more than one state";
    case FindResult::NothingToPatch:
        return "no other store to that state that can be patched";
    case FindResult::TooManySites:
        return "too many stores to that state";
    case FindResult::OverlappingSites:
        return "stores to that state overlap";
    }

    return "unknown";
}

// A byte store [reg+disp32] of the C6 /0 form or the 40 88 form: mod = 10, no SIB.
inline bool IsDisp32NoSib(uint8_t modrm) { return modrm >= 0x80 && modrm <= 0xBF && (modrm & 7) != 4; }

inline uint32_t ReadU32(const uint8_t* p)
{
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Derives the wanted state and every store that has to change. Reads only; nothing is written.
inline FindResult FindPlan(void* image, Plan& plan)
{
    plan = {};

    // 1. Exactly one copy of the fallback message.
    std::vector<const uint8_t*> strings;

    const bool valid = Provider::ForEachSection(
        image,
        [&](uint8_t* data, size_t size, DWORD characteristics)
        {
            if (!(characteristics & IMAGE_SCN_MEM_READ) || size < kMarker.size())
                return;

            for (size_t off = 0; off + kMarker.size() <= size; ++off)
                if (std::memcmp(data + off, kMarker.data(), kMarker.size()) == 0)
                    strings.push_back(data + off);
        });

    if (!valid)
        return FindResult::BadImage;

    if (strings.empty())
        return FindResult::NoMarker;

    if (strings.size() > 1)
        return FindResult::AmbiguousMarker;

    // 2. Exactly one lea reg,[rip+disp32] that points at it.
    struct Reference
    {
        uint8_t* at;
        size_t room; // bytes from `at` to the end of its section
    };

    std::vector<Reference> references;
    const auto target = reinterpret_cast<intptr_t>(strings[0]);

    Provider::ForEachSection(image,
                             [&](uint8_t* data, size_t size, DWORD characteristics)
                             {
                                 if (!(characteristics & IMAGE_SCN_MEM_EXECUTE) || size < 7)
                                     return;

                                 for (size_t off = 0; off + 7 <= size; ++off)
                                 {
                                     const uint8_t* p = data + off;

                                     if ((p[0] != 0x48 && p[0] != 0x4C) || p[1] != 0x8D || (p[2] & 0xC7) != 0x05)
                                         continue;

                                     int32_t disp = 0;
                                     std::memcpy(&disp, p + 3, sizeof(disp));

                                     if (reinterpret_cast<intptr_t>(p) + 7 + disp == target)
                                         references.push_back({ data + off, size - off });
                                 }
                             });

    if (references.empty())
        return FindResult::NoReference;

    if (references.size() > 1)
        return FindResult::AmbiguousReference;

    // 3. The state the fallback writes: mov byte ptr [reg+disp32], imm8 in the window after the reference.
    //    Every such store has to agree.
    bool haveWanted = false;
    uint32_t wantField = 0;
    uint8_t wantValue = 0;
    bool conflict = false;

    const size_t window = references[0].room < kWindow ? references[0].room : kWindow;

    for (size_t off = 0; off + kStoreLength <= window; ++off)
    {
        const uint8_t* p = references[0].at + off;

        if (p[0] != 0xC6 || !IsDisp32NoSib(p[1]) || (p[1] & 0x38) != 0)
            continue;

        const uint32_t field = ReadU32(p + 2);

        if (field <= kMinField || field >= kMaxField || p[6] > 1)
            continue;

        if (!haveWanted)
        {
            haveWanted = true;
            wantField = field;
            wantValue = p[6];
        }
        else if (field != wantField || p[6] != wantValue)
        {
            conflict = true;
        }
    }

    if (!haveWanted)
        return FindResult::NoFallback;

    if (conflict)
        return FindResult::ConflictingFallback;

    plan.field = wantField;
    plan.value = wantValue;

    // 4. Every other store to that offset, in the executable sections.
    const uint8_t opposite = static_cast<uint8_t>(1 - wantValue);
    bool tooMany = false;

    Provider::ForEachSection(
        image,
        [&](uint8_t* data, size_t size, DWORD characteristics)
        {
            if (!(characteristics & IMAGE_SCN_MEM_EXECUTE) || size < kStoreLength)
                return;

            for (size_t off = 0; off + kStoreLength <= size; ++off)
            {
                const uint8_t* p = data + off;
                Site site;

                if (p[0] == 0xC6)
                {
                    // Flip the immediate. A store that already writes the wanted value needs nothing.
                    if (!IsDisp32NoSib(p[1]) || (p[1] & 0x38) != 0 || ReadU32(p + 2) != wantField || p[6] != opposite)
                        continue;

                    std::memcpy(site.original, p, kStoreLength);
                    std::memcpy(site.replacement, p, kStoreLength);
                    site.replacement[6] = wantValue;
                }
                else if (p[0] == 0x40 && p[1] == 0x88)
                {
                    // A register store, rewritten whole. REX must be exactly 0x40: any B/R/X/W bit would
                    // change the length or the base register.
                    if (!IsDisp32NoSib(p[2]) || ReadU32(p + 3) != wantField)
                        continue;

                    std::memcpy(site.original, p, kStoreLength);
                    site.replacement[0] = 0xC6;
                    site.replacement[1] = static_cast<uint8_t>(0x80 | (p[2] & 7));
                    std::memcpy(site.replacement + 2, &wantField, sizeof(wantField));
                    site.replacement[6] = wantValue;
                    site.fromRegister = true;
                }
                else
                {
                    continue;
                }

                if (plan.sites.size() >= kMaxSites)
                {
                    tooMany = true;
                    return;
                }

                site.address = data + off;
                plan.sites.push_back(site);
            }
        });

    if (tooMany)
    {
        plan.sites.clear();
        return FindResult::TooManySites;
    }

    if (plan.sites.empty())
        return FindResult::NothingToPatch;

    // Two stores that overlap cannot both be instructions.
    for (size_t i = 0; i < plan.sites.size(); ++i)
        for (size_t j = i + 1; j < plan.sites.size(); ++j)
        {
            const auto a = reinterpret_cast<uintptr_t>(plan.sites[i].address);
            const auto b = reinterpret_cast<uintptr_t>(plan.sites[j].address);

            if (a < b + kStoreLength && b < a + kStoreLength)
            {
                plan.sites.clear();
                return FindResult::OverlappingSites;
            }
        }

    return FindResult::Found;
}

enum class ApplyResult
{
    Patched,
    Mismatch,      // a site no longer holds the bytes that were read
    ProtectFailed, // a page could not be made writable; anything already written was put back
};

inline bool WriteSite(uint8_t* at, const uint8_t* bytes)
{
    DWORD oldProtect = 0;

    if (!VirtualProtect(at, kStoreLength, PAGE_EXECUTE_READWRITE, &oldProtect))
        return false;

    std::memcpy(at, bytes, kStoreLength);

    DWORD ignored = 0;
    VirtualProtect(at, kStoreLength, oldProtect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), at, kStoreLength);
    return true;
}

// Writes every site, or none: each is checked against what was read first, and a failed write puts the
// earlier ones back.
inline ApplyResult Apply(const Plan& plan)
{
    for (const auto& site : plan.sites)
        if (std::memcmp(site.address, site.original, kStoreLength) != 0)
            return ApplyResult::Mismatch;

    for (size_t i = 0; i < plan.sites.size(); ++i)
    {
        if (WriteSite(plan.sites[i].address, plan.sites[i].replacement))
            continue;

        for (size_t back = 0; back < i; ++back)
            WriteSite(plan.sites[back].address, plan.sites[back].original);

        return ApplyResult::ProtectFailed;
    }

    return ApplyResult::Patched;
}
} // namespace MfgUnlock::Flip
