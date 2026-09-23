// Manual check, not a test: reports which of the RTX 40 MFG unlock's patches would find their targets in
// real NVIDIA modules. Give it paths to nvngx_dlssg.dll and/or sl.dlss_g.dll (any build; .dll.dlsss and
// the driver's OTA copies work too). Each module is mapped without running any of its code, and anything
// applied is applied to that mapping in this process only. No file is modified.
//
//   cl /std:c++20 /EHsc tests/mfg_real_module_check.cpp
//   mfg_real_module_check.exe C:\path\nvngx_dlssg.dll C:\path\sl.dlss_g.dll
//
// The gate signatures below are copies of the ones in MfgUnlock.cpp; keep them in step.
#include "../OptiScaler/framegen/dlssg/MfgUnlockFlip.h"
#include "../OptiScaler/framegen/dlssg/MfgUnlockPlugin.h"
#include "../OptiScaler/framegen/dlssg/MfgUnlockPtx.h"

#include <cstdio>
#include <sstream>

using namespace MfgUnlock;

// "BB 01 ?? .." style pattern: how many times it matches in the executable sections.
static size_t CountPattern(void* image, const char* pattern)
{
    std::vector<int> bytes;
    std::istringstream stream(pattern);
    std::string token;

    while (stream >> token)
        bytes.push_back(token == "?" ? -1 : static_cast<int>(strtol(token.c_str(), nullptr, 16)));

    size_t hits = 0;

    Provider::ForEachSection(image,
                             [&](uint8_t* data, size_t size, DWORD characteristics)
                             {
                                 if (!(characteristics & IMAGE_SCN_MEM_EXECUTE))
                                     return;

                                 for (size_t off = 0; off + bytes.size() <= size; ++off)
                                 {
                                     bool match = true;

                                     for (size_t i = 0; i < bytes.size() && match; ++i)
                                         if (bytes[i] >= 0 && data[off + i] != bytes[i])
                                             match = false;

                                     hits += match;
                                 }
                             });

    return hits;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("usage: %s <module> [<module> ...]\n", argv[0]);
        return 2;
    }

    for (int a = 1; a < argc; ++a)
    {
        std::wstring path;

        for (const char* c = argv[a]; *c; ++c)
            path += static_cast<wchar_t>(*c);

        HMODULE module = LoadLibraryExW(path.c_str(), nullptr, DONT_RESOLVE_DLL_REFERENCES);
        printf("\n== %s\n", argv[a]);

        if (module == nullptr)
        {
            printf("  could not be mapped (error %lu)\n", GetLastError());
            continue;
        }

        void* image = module;
        const bool ngx = GetProcAddress(module, "NVSDK_NGX_D3D12_PopulateDeviceParameters_Impl") != nullptr ||
                         GetProcAddress(module, "NVSDK_NGX_VULKAN_PopulateDeviceParameters_Impl") != nullptr;

        printf("  DLSS-G snippet: NGX entry point %s, marker %s\n", ngx ? "yes" : "no",
               Provider::ImageContains(image, Provider::kMarker) ? "yes" : "no");

        printf("  frame-count gates: legacy advertise %zu, validate %zu; 310.9 advertise %zu, validate %zu (each must "
               "be exactly 1)\n",
               CountPattern(image, "BB 01 00 00 00 41 B8 03 00 00 00 81 FF B0 01 00 00 44 0F 4C C3"),
               CountPattern(image, "3D B0 01 00 00 7C ? 83 FB 03 76"),
               CountPattern(image, "81 FD B0 01 00 00 0F 8C ? ? ? ? BF 05 00 00 00"),
               CountPattern(image, "3D B0 01 00 00 0F 93 C0"));

        Ptx::Result ptx;
        const bool ptxOk = Ptx::Apply(image, ptx);
        printf("  PTX temporal fix: %s%s\n", ptxOk ? "would apply: " : "not applicable: ", ptx.detail.c_str());

        Plugin::CeilingSite site;
        const auto ceiling = Plugin::FindCeilingSite(image, site);
        printf("  plugin frame-count clamp: %s",
               ceiling == Plugin::FindResult::Found       ? "found"
               : ceiling == Plugin::FindResult::Ambiguous ? "AMBIGUOUS (refused)"
                                                          : "not found");
        if (ceiling == Plugin::FindResult::Found)
            printf(", compiled maximum %u generated frame(s) at +0x%llx", site.compiled,
                   static_cast<unsigned long long>(site.address - static_cast<uint8_t*>(image)));
        printf("\n");

        Flip::Plan plan;
        const auto flip = Flip::FindPlan(image, plan);
        printf("  plugin flip metering: %s", Flip::Describe(flip));
        if (flip == Flip::FindResult::Found)
        {
            printf(": context +0x%X pinned to %u at %zu site(s)\n", plan.field, plan.value, plan.sites.size());

            for (const auto& site2 : plan.sites)
            {
                printf("    +0x%llx %s: ", static_cast<unsigned long long>(site2.address - static_cast<uint8_t*>(image)),
                       site2.fromRegister ? "register store" : "immediate store");

                for (uint8_t b : site2.original)
                    printf("%02X ", b);
                printf("-> ");
                for (uint8_t b : site2.replacement)
                    printf("%02X ", b);
                printf("\n");
            }
        }
        else
        {
            printf("\n");
        }
    }

    return 0;
}
