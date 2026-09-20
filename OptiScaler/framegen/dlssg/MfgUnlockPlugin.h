#pragma once

// The Streamline DLSS-G plugin (sl.dlss_g): the frame-count ceiling it applies on top of what
// nvngx_dlssg.dll reports. Header-only and free of OptiScaler headers so tests/mfg_ceiling_smoke.cpp can
// compile it alone.
//
// The plugin starts from its own compiled maximum, then lowers it to the device value cached by the
// Streamline wrapper:
//
//     BA 03 00 00 00   mov   edx, 3
//     3B CA            cmp   ecx, edx
//     0F 42 D1         cmovb edx, ecx      ; edx = min(count, 3)
//
// Most games see the unlocked snippet early enough for that cached value to be 5. A wrapper that cached
// 1 lowers the maximum back to one generated frame, and slDLSSGSetOptions then rejects 3x and 4x with
// eErrorInvalidState (38) even though the snippet accepts them.
//
// The patch turns `cmovb edx, ecx` into `cmovb edx, edx` by changing only its ModRM byte (D1 -> D2):
// the same three-byte instruction, so the instruction boundary is untouched, and a single-byte write.
// The compiled immediate stays in place as a hard bound, so a plugin compiled for 3 stays capped at 3.
//
// Adapted from KleberMotta/OptiScaler-DLSS5-MFG-RTX40 74c3bac (mfgunlock/, MIT, from the RenoDX MFG
// Unlock addon by Dreamt / mavismmg). See Licenses/MFGUnlock_LICENSE.txt.

#include "MfgUnlockProvider.h"

namespace MfgUnlock::Plugin
{
// The whole matched sequence, and the offset of the ModRM byte the patch changes.
inline constexpr size_t kCeilingLength = 10;
inline constexpr size_t kModRmOffset = 9;
inline constexpr uint8_t kModRmOriginal = 0xD1;
inline constexpr uint8_t kModRmPatched = 0xD2;

// After the four bytes of the immediate: cmp ecx,edx / cmovb edx,ecx.
inline constexpr uint8_t kClampTail[] = { 0x3B, 0xCA, 0x0F, 0x42, 0xD1 };

struct CeilingSite
{
    uint8_t* address = nullptr; // the start of `mov edx, imm32`
    uint8_t compiled = 0;       // the plugin's own maximum, in generated frames
};

enum class FindResult
{
    Found,
    None,
    Ambiguous, // more than one match: which one is the clamp is not known, so none is touched
    BadImage,
};

// Looks for the clamp in the executable sections of a mapped plugin. Only a compiled maximum of 1 to 8
// generated frames counts, and the upper three bytes of the immediate must be zero.
inline FindResult FindCeilingSite(void* image, CeilingSite& site)
{
    site = {};
    size_t hits = 0;

    const bool valid = Provider::ForEachSection(
        image,
        [&](uint8_t* data, size_t size, DWORD characteristics)
        {
            if (!(characteristics & IMAGE_SCN_MEM_EXECUTE) || size < kCeilingLength)
                return;

            for (size_t off = 0; off + kCeilingLength <= size; ++off)
            {
                const uint8_t* p = data + off;

                if (p[0] != 0xBA || p[2] != 0 || p[3] != 0 || p[4] != 0)
                    continue;

                if (p[1] == 0 || p[1] > 8)
                    continue;

                if (std::memcmp(p + 5, kClampTail, sizeof(kClampTail)) != 0)
                    continue;

                if (hits++ == 0)
                {
                    site.address = data + off;
                    site.compiled = p[1];
                }
            }
        });

    if (!valid)
    {
        site = {};
        return FindResult::BadImage;
    }

    if (hits == 0)
        return FindResult::None;

    if (hits > 1)
    {
        site = {};
        return FindResult::Ambiguous;
    }

    return FindResult::Found;
}

enum class ApplyResult
{
    Patched,
    Mismatch,      // the ModRM byte is not the original: already patched, or not the instruction found
    ProtectFailed,
};

// Writes the one byte. Checks the original again immediately before writing.
inline ApplyResult ApplyCeilingPatch(const CeilingSite& site)
{
    if (site.address == nullptr)
        return ApplyResult::Mismatch;

    uint8_t* target = site.address + kModRmOffset;

    if (*target != kModRmOriginal)
        return ApplyResult::Mismatch;

    DWORD oldProtect = 0;

    if (!VirtualProtect(target, 1, PAGE_EXECUTE_READWRITE, &oldProtect))
        return ApplyResult::ProtectFailed;

    *target = kModRmPatched;

    DWORD ignored = 0;
    VirtualProtect(target, 1, oldProtect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), target, 1);

    return ApplyResult::Patched;
}
} // namespace MfgUnlock::Plugin
