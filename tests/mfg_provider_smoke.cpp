// Host check of MfgUnlockProvider.h: which paths and which mapped images count as a DLSS-G provider.
// No GPU and no game needed; the images are built in memory.
// cl /std:c++20 /EHsc tests/mfg_provider_smoke.cpp
#include "../OptiScaler/framegen/dlssg/MfgUnlockProvider.h"

#include <cstdio>
#include <vector>

using namespace MfgUnlock::Provider;

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

// A 64-bit PE with two readable sections: .text at 0x1000 and .rdata at 0x1400, both 0x400 long,
// inside a 0x2000-byte image.
struct Image
{
    std::vector<uint8_t> bytes = std::vector<uint8_t>(0x2000, 0);

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

    void put(size_t at, std::string_view text) { std::memcpy(bytes.data() + at, text.data(), text.size()); }
    IMAGE_SECTION_HEADER* section(int i) { return IMAGE_FIRST_SECTION(nt()) + i; }
    IMAGE_NT_HEADERS64* nt() { return reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + 0x80); }
};

int main()
{
    // Paths: the game's own snippet, the OTA store in both spellings the loader produces, and things
    // that must not match.
    CHECK(IsProviderPath(L"C:\\Game\\bin\\nvngx_dlssg.dll"));
    CHECK(IsProviderPath(L"c:\\game\\bin\\NVNGX_DLSSG.DLL"));
    CHECK(IsProviderPath(L"nvngx_dlssg.dll"));
    CHECK(IsProviderPath(L"C:\\ProgramData\\NVIDIA\\NGX\\models\\dlssg\\versions\\131072\\files\\310_e5a4b2.bin"));
    CHECK(IsProviderPath(L"c:\\programdata/nvidia/ngx/models//dlssg/versions/131072/files/310_e5a4b2.bin"));
    CHECK(!IsProviderPath(L"C:\\ProgramData\\NVIDIA\\NGX\\models\\dlss\\versions\\1\\files\\160_e658700.bin"));
    CHECK(!IsProviderPath(L"C:\\ProgramData\\NVIDIA\\NGX\\models\\dlssd\\versions\\1\\files\\a.bin"));
    CHECK(!IsProviderPath(L"C:\\Game\\bin\\nvngx_dlssg_backup.dll"));
    CHECK(!IsProviderPath(L"C:\\nvngx_dlssg.dll.old\\other.dll"));
    CHECK(!IsProviderPath(L"C:\\Game\\bin\\sl.dlss_g.dll"));
    CHECK(!IsProviderPath(L""));

    // Images: the marker in a readable section is found.
    {
        Image img;
        img.put(0x1400 + 0x40, kMarker);
        CHECK(ImageContains(img.bytes.data(), kMarker));
    }
    {
        Image img;
        img.put(0x1000 + 0x10, kMarker);
        CHECK(ImageContains(img.bytes.data(), kMarker)); // code sections count too
    }
    // Absent.
    {
        Image img;
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    // Present only in the padding past VirtualSize: not initialised data, not counted.
    {
        Image img;
        img.put(0x1400 + 0x400 + 0x10, kMarker);
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    // Split across the end of one section and the start of the next: no match.
    {
        Image img;
        img.put(0x1000 + 0x400 - 5, kMarker.substr(0, 5));
        img.put(0x1400, kMarker.substr(5));
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    // A section that is not readable is not searched.
    {
        Image img;
        img.put(0x1400 + 0x40, kMarker);
        img.section(1)->Characteristics = IMAGE_SCN_MEM_EXECUTE;
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    // A section that claims to run past SizeOfImage is skipped, never read past the buffer.
    {
        Image img;
        img.section(1)->Misc.VirtualSize = 0x10000;
        CHECK(!ImageContains(img.bytes.data(), kMarker));
        img.section(1)->Misc.VirtualSize = 0x400;
        img.section(1)->VirtualAddress = 0x3000;
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    // Not a PE, wrong optional header, no image.
    {
        Image img;
        img.put(0x1400, kMarker);
        img.bytes[0] = 'X';
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    {
        Image img;
        img.put(0x1400, kMarker);
        img.nt()->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR32_MAGIC;
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    {
        Image img;
        img.put(0x1400, kMarker);
        img.nt()->Signature = 0;
        CHECK(!ImageContains(img.bytes.data(), kMarker));
    }
    CHECK(!ImageContains(nullptr, kMarker));
    {
        Image img;
        CHECK(!ImageContains(img.bytes.data(), std::string_view {}));
    }

    if (fails == 0)
        printf("mfg_provider_smoke: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
