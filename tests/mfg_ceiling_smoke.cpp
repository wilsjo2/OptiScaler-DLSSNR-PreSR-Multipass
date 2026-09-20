// Host check of MfgUnlockPlugin.h: finding the Streamline DLSS-G plugin's frame-count clamp and patching
// its one ModRM byte. No GPU and no game needed; the plugin image is built in memory.
// cl /std:c++20 /EHsc tests/mfg_ceiling_smoke.cpp
#include "../OptiScaler/framegen/dlssg/MfgUnlockPlugin.h"

#include <cstdio>
#include <vector>

using namespace MfgUnlock::Plugin;

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            printf("FAIL line %d: %s\n", __LINE__, #c);                                                                \
            ++fails;                                                                                                   \
        }                                                                                                              \
    } while (0)

// A 64-bit PE with .text (executable) at 0x1000 and .rdata (readable data) at 0x1400, 0x400 long each,
// inside a 0x2000-byte image.
struct Image
{
    std::vector<uint8_t> bytes = std::vector<uint8_t>(0x2000, 0x90);

    Image()
    {
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 0x80;

        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + 0x80);
        nt->Signature = IMAGE_NT_SIGNATURE;
        nt->FileHeader.NumberOfSections = 2;
        nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
        nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        nt->OptionalHeader.SizeOfImage = 0x2000;

        auto* s = IMAGE_FIRST_SECTION(nt);
        std::memcpy(s[0].Name, ".text", 5);
        s[0].VirtualAddress = 0x1000;
        s[0].Misc.VirtualSize = 0x400;
        s[0].Characteristics = IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE;
        std::memcpy(s[1].Name, ".rdata", 6);
        s[1].VirtualAddress = 0x1400;
        s[1].Misc.VirtualSize = 0x400;
        s[1].Characteristics = IMAGE_SCN_MEM_READ;
    }

    // mov edx, <ceiling> / cmp ecx, edx / cmovb edx, ecx
    void putClamp(size_t at, uint8_t ceiling = 3)
    {
        const uint8_t seq[] = { 0xBA, ceiling, 0, 0, 0, 0x3B, 0xCA, 0x0F, 0x42, 0xD1 };
        std::memcpy(bytes.data() + at, seq, sizeof(seq));
    }

    IMAGE_NT_HEADERS64* nt() { return reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + 0x80); }
    void* data() { return bytes.data(); }
};

int main()
{
    CeilingSite site;

    // One clamp in the code section is found, with the plugin's own maximum.
    {
        Image img;
        img.putClamp(0x1100, 3);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::Found);
        CHECK(site.address == img.bytes.data() + 0x1100);
        CHECK(site.compiled == 3);
    }
    {
        Image img;
        img.putClamp(0x1100, 5);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::Found);
        CHECK(site.compiled == 5);
    }
    // Right at the end of the section: found. Cut short by the end of the section: not found, and
    // nothing past the section is read.
    {
        Image img;
        img.putClamp(0x1000 + 0x400 - kCeilingLength);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::Found);
    }
    {
        Image img;
        img.putClamp(0x1000 + 0x400 - kCeilingLength + 1);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
    }
    // No match at all.
    {
        Image img;
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
        CHECK(site.address == nullptr);
    }
    // Two matches: refused, and the site is cleared so nothing can be patched by mistake.
    {
        Image img;
        img.putClamp(0x1100);
        img.putClamp(0x1200);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::Ambiguous);
        CHECK(site.address == nullptr);
    }
    // A match in a section that is not executable does not count.
    {
        Image img;
        img.putClamp(0x1500);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
    }
    // A maximum outside 1..8, or an immediate with non-zero upper bytes, is a different instruction.
    {
        Image img;
        img.putClamp(0x1100, 0);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
        img.putClamp(0x1100, 9);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
        img.putClamp(0x1100, 3);
        img.bytes[0x1100 + 2] = 1;
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
    }
    // Different ModRM or a different compare: not the clamp.
    {
        Image img;
        img.putClamp(0x1100);
        img.bytes[0x1100 + 9] = 0xD2;
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
        img.putClamp(0x1100);
        img.bytes[0x1100 + 5] = 0x39;
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
    }
    // A section that claims to run past SizeOfImage is skipped, never read past the buffer.
    {
        Image img;
        IMAGE_FIRST_SECTION(img.nt())[0].Misc.VirtualSize = 0x10000;
        img.putClamp(0x1100);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
    }
    // Not a PE, or no image.
    {
        Image img;
        img.putClamp(0x1100);
        img.bytes[0] = 'X';
        CHECK(FindCeilingSite(img.data(), site) == FindResult::BadImage);
        CHECK(site.address == nullptr);
    }
    CHECK(FindCeilingSite(nullptr, site) == FindResult::BadImage);

    // Applying changes exactly the ModRM byte, once.
    {
        Image img;
        img.putClamp(0x1100, 3);
        const std::vector<uint8_t> before = img.bytes;
        CHECK(FindCeilingSite(img.data(), site) == FindResult::Found);
        CHECK(ApplyCeilingPatch(site) == ApplyResult::Patched);

        size_t changed = 0;
        for (size_t i = 0; i < before.size(); ++i)
            if (before[i] != img.bytes[i])
            {
                ++changed;
                CHECK(i == 0x1100 + kModRmOffset);
                CHECK(before[i] == 0xD1 && img.bytes[i] == 0xD2);
            }
        CHECK(changed == 1);
        CHECK(img.bytes[0x1100 + 1] == 3); // the compiled bound is untouched

        // Already patched: refused, and the site is no longer found as a clamp.
        CHECK(ApplyCeilingPatch(site) == ApplyResult::Mismatch);
        CHECK(FindCeilingSite(img.data(), site) == FindResult::None);
    }
    // An empty site never writes.
    {
        CeilingSite none;
        CHECK(ApplyCeilingPatch(none) == ApplyResult::Mismatch);
    }

    if (fails == 0)
        printf("mfg_ceiling_smoke: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
