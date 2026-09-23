// Vendored from optiscaler/OptiScaler @ bb1619ec1eac76e8109d3adf9c795d6d37ba1ba4 (OptiScaler/proxies/XeFGUnlock.h).
// Upstream authorship (GPL-3.0). Do not edit in place - re-fetch the pinned blob instead.
// sha256 of the vendored body (from line 4 down): e6de23a98d7ee34b241b88612581f3aad7f173f8687387dd449398b96c7a80c6
#pragma once

// Unlocks multi frame generation in Intel's libxess_fg.dll.
//
// Intel gates MFG behind "am I the igxess_fg.dll build?" rather than behind an
// actual hardware capability query, so the whole thing comes down to five
// byte-level patches in the provider's own code:
//
//   U1  0x20DA4F   jne -> jmp    the per-frame frame count resolver no longer
//                                falls back to 2X with "XeLL version x.y.z is
//                                too old to support multi-frame generation"
//   U2  0x1A5DE4   je  -> jmp    Settings::mfgAllowed() always returns true, so
//                                the init path stops downgrading model 17/18
//   U3  0x1A517D   mov ebx,3 -> mov ebx,N
//                                raises the default interpolation ceiling
//   U4  0x1A45C2   mov dword [rdi+0x16c], 1 -> mov dword [rdi+0x16c], N
//                                when the provider decides MFG is unavailable
//                                it pins the override to a single interpolated
//                                frame; pin it to the configured count instead
//   U5  0x20973B   mov eax,1 -> mov eax,N
//                                xefgSwapChainGetProperties reports
//                                maxSupportedInterpolations = 1 whenever the
//                                XeLL version is unknown or older than
//                                1.3.0. OptiScaler hides its MFG combo box
//                                entirely for a reported value of 1, so this
//                                is the one that makes the combo appear.
//
// Unlocking the count is not enough on its own. Above 2X the provider hands
// every generated frame of a burst to the swapchain back to back and only paces
// the burst as a whole, which shows up as the frames arriving bunched up and out
// of order - so the present thunk at 0x25C0 is additionally redirected to pace
// each generated frame individually. See XeFGPacing.h; it is a separate patch
// with its own XeFG\ExtraPacing switch and does not touch the five above.
//
// The dll on disk is never modified: we patch the already mapped image, verify
// every write by read-back, and roll the whole set back if any single step
// fails. A failed unlock is not fatal - the provider just behaves as it always
// did.

#include "SysUtils.h"
#include "Logger.h"
#include "Config.h"
#include "XeFGPacing.h"

class XeFGUnlock
{
  public:
    static bool Applied() { return _applied; }

    // Returns true when every patch was applied. Safe to call more than once.
    static bool Apply(HMODULE module)
    {
        if (_applied)
            return true;

        if (module == nullptr)
            return false;

        if (!Config::Instance()->FGXeFGUnlockEnabled.value_or_default())
        {
            LOG_INFO("XeFG unlock: disabled by config (XeFG\\UnlockMFG)");
            return false;
        }

        auto* base = reinterpret_cast<uint8_t*>(module);
        auto* nt = NtHeaders(base);
        auto* text = FindSection(base, ".text");

        if (nt == nullptr || text == nullptr)
        {
            LOG_WARN("XeFG unlock: provider has no usable PE headers, skipping");
            return false;
        }

        if (nt->FileHeader.TimeDateStamp != KnownBuildStamp || nt->OptionalHeader.SizeOfImage != KnownSizeOfImage)
            LOG_WARN("XeFG unlock: unrecognised provider build {:#010x}/{:#x}, relying on per-byte checks",
                     nt->FileHeader.TimeDateStamp, nt->OptionalHeader.SizeOfImage);
        else
            LOG_INFO("XeFG unlock: recognised provider build {:#010x}", KnownBuildStamp);

        int32_t maxInterp = Config::Instance()->FGXeFGMaxInterpolatedFrames.value_or_default();

        if (maxInterp < 1 || maxInterp > MaxReportedInterpolations)
            maxInterp = MaxReportedInterpolations;

        // A reported count of 1 is plain 2X, which is what the unpatched
        // provider already does - so at that setting we leave it alone.
        const bool unlock = maxInterp > 1;

        static const uint8_t u1Old[] = { 0x0F, 0x85, 0xCC, 0x00, 0x00, 0x00 };
        static const uint8_t u1New[] = { 0xE9, 0xCD, 0x00, 0x00, 0x00, 0x90 };

        static const uint8_t u2Old[] = { 0x74, 0x09 };
        static const uint8_t u2New[] = { 0xEB, 0x06 };

        static const uint8_t u3Old[] = { 0xBB, 0x03, 0x00, 0x00, 0x00 };
        static const uint8_t u3New[] = { 0xBB, 0x00, 0x00, 0x00, 0x00 };

        static const uint8_t u4Old[] = { 0xC7, 0x87, 0x6C, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 };
        static const uint8_t u4New[] = { 0xC7, 0x87, 0x6C, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

        static const uint8_t u5Old[] = { 0xB8, 0x01, 0x00, 0x00, 0x00 };
        static const uint8_t u5New[] = { 0xB8, 0x00, 0x00, 0x00, 0x00 };

        // immOffset is the offset of a little endian imm32 inside `replacement`
        // that gets overwritten with the configured interpolation count, so the
        // count is baked into the bytes we write rather than patched in later.
        const Patch patches[] = {
            { 0x20DA4F, u1Old, u1New, sizeof(u1Old), -1, unlock, "U1/frame-count-fallback" },
            { 0x1A5DE4, u2Old, u2New, sizeof(u2Old), -1, unlock, "U2/model-downgrade" },
            { 0x1A517D, u3Old, u3New, sizeof(u3Old), 1, unlock, "U3/default-ceiling" },
            { 0x1A45C2, u4Old, u4New, sizeof(u4Old), 6, unlock, "U4/override-clamp" },
            { 0x20973B, u5Old, u5New, sizeof(u5Old), 1, unlock, "U5/reported-maximum" },
        };

        constexpr int32_t PatchCount = static_cast<int32_t>(sizeof(patches) / sizeof(patches[0]));

        Edited applied[PatchCount] {};
        int32_t appliedCount = 0;
        int32_t skipped = 0;

        for (const auto& patch : patches)
        {
            if (!patch.enabled)
            {
                skipped++;
                continue;
            }

            if (!Contains(text, patch.rva, patch.size))
            {
                LOG_WARN("XeFG unlock: {} at {:#x} falls outside .text, aborting", patch.name, patch.rva);
                Rollback(applied, appliedCount);
                return false;
            }

            uint8_t* dst = base + patch.rva;

            if (memcmp(dst, patch.expected, patch.size) != 0)
            {
                LOG_WARN("XeFG unlock: {} at {:#x} has unexpected bytes ({}), aborting", patch.name, patch.rva,
                         ToHex(dst, patch.size));
                Rollback(applied, appliedCount);
                return false;
            }

            uint8_t buffer[16] {};
            const uint8_t* replacement = patch.replacement;

            if (patch.immOffset >= 0)
            {
                memcpy(buffer, patch.replacement, patch.size);
                buffer[patch.immOffset + 0] = static_cast<uint8_t>(maxInterp & 0xFF);
                buffer[patch.immOffset + 1] = static_cast<uint8_t>((maxInterp >> 8) & 0xFF);
                buffer[patch.immOffset + 2] = static_cast<uint8_t>((maxInterp >> 16) & 0xFF);
                buffer[patch.immOffset + 3] = static_cast<uint8_t>((maxInterp >> 24) & 0xFF);
                replacement = buffer;
            }

            if (!WriteVerified(dst, replacement, patch.size))
            {
                LOG_WARN("XeFG unlock: {} at {:#x} failed write verification, aborting", patch.name, patch.rva);
                Rollback(applied, appliedCount);
                return false;
            }

            applied[appliedCount] = { dst, patch.expected, patch.size };
            appliedCount++;

            LOG_INFO("XeFG unlock: {} patched at {:#x} -> {}", patch.name, patch.rva, ToHex(dst, patch.size));
        }

        _applied = true;

        LOG_INFO("XeFG unlock: {} of {} patches applied ({} skipped), MFG enabled up to {}X", appliedCount, PatchCount,
                 skipped, maxInterp + 1);

        // The provider emits every generated frame of a burst back to back above
        // 2X, which is only reachable now that the multi frame path is unlocked.
        // Independent of the patches above, and harmless if it fails.
        if (unlock)
            XeFGPacing::Install(base);

        return true;
    }

  private:
    // The provider reports an interpolation frame count N, where N=1 is plain 2X.
    //
    // This was 5, justified as "the combo box lists 2X..6X" - which was circular,
    // since that combo box is ours. The patches below write N as a 4 byte
    // immediate, so the encoding has never been what limits this. The ceiling is
    // now the shared sanity bound, and what the user actually gets is decided by
    // XeFG\MaxInterpolatedFrames.
    static constexpr int32_t MaxReportedInterpolations = Config::XeFGMaxInterpolations;

    // Build identity of the libxess_fg.dll these offsets were derived from.
    static constexpr uint32_t KnownBuildStamp = 0x69CB0F4D;
    static constexpr uint32_t KnownSizeOfImage = 0x015ED000;

    struct Patch
    {
        uint32_t rva;
        const uint8_t* expected;
        const uint8_t* replacement;
        uint32_t size;
        int32_t immOffset;
        bool enabled;
        const char* name;
    };

    struct Edited
    {
        uint8_t* dst;
        const uint8_t* original;
        uint32_t size;
    };

    inline static bool _applied = false;

    static IMAGE_NT_HEADERS* NtHeaders(uint8_t* base)
    {
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);

        if (dos->e_magic != IMAGE_DOS_SIGNATURE)
            return nullptr;

        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);

        return nt->Signature == IMAGE_NT_SIGNATURE ? nt : nullptr;
    }

    static const IMAGE_SECTION_HEADER* FindSection(uint8_t* base, const char* name)
    {
        auto* nt = NtHeaders(base);

        if (nt == nullptr)
            return nullptr;

        auto* section = IMAGE_FIRST_SECTION(nt);

        for (WORD i = 0; i < nt->FileHeader.NumberOfSections; i++, section++)
        {
            if (strncmp(reinterpret_cast<const char*>(section->Name), name, IMAGE_SIZEOF_SHORT_NAME) == 0)
                return section;
        }

        return nullptr;
    }

    static bool Contains(const IMAGE_SECTION_HEADER* section, uint32_t rva, uint32_t size)
    {
        uint32_t start = section->VirtualAddress;
        uint32_t end = start + section->Misc.VirtualSize;

        return rva >= start && rva + size <= end;
    }

    static void WriteRaw(uint8_t* dst, const uint8_t* bytes, uint32_t size)
    {
        DWORD oldProtect = 0;

        if (!VirtualProtect(dst, size, PAGE_EXECUTE_READWRITE, &oldProtect))
            return;

        memcpy(dst, bytes, size);
        FlushInstructionCache(GetCurrentProcess(), dst, size);

        DWORD ignored = 0;
        VirtualProtect(dst, size, oldProtect, &ignored);
    }

    static bool WriteVerified(uint8_t* dst, const uint8_t* bytes, uint32_t size)
    {
        WriteRaw(dst, bytes, size);
        return memcmp(dst, bytes, size) == 0;
    }

    static void Rollback(const Edited* applied, int32_t count)
    {
        if (count == 0)
            return;

        // Undo newest first so the image ends up exactly as the provider shipped it.
        for (int32_t i = count - 1; i >= 0; i--)
            WriteRaw(applied[i].dst, applied[i].original, applied[i].size);

        LOG_INFO("XeFG unlock: rolled back {} patch(es)", count);
    }

    static std::string ToHex(const uint8_t* bytes, uint32_t size)
    {
        static const char* digits = "0123456789ABCDEF";

        std::string out;
        out.reserve(size * 3);

        for (uint32_t i = 0; i < size; i++)
        {
            if (i > 0)
                out.push_back(' ');

            out.push_back(digits[bytes[i] >> 4]);
            out.push_back(digits[bytes[i] & 0xF]);
        }

        return out;
    }
};
