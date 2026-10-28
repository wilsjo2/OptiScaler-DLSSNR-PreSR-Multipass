#pragma once
#include <windows.h>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace DlssNr::RuntimeImports
{
struct Slot
{
    void** address;
    bool wide;
};

// Inspect the loaded image, not file offsets. Import RVAs may move between runtime builds.
inline bool Find(std::span<unsigned char> image, std::vector<Slot>& slots)
{
    slots.clear();
    auto range = [&](size_t offset, size_t length)
    { return offset <= image.size() && length <= image.size() - offset; };
    if (!range(0, sizeof(IMAGE_DOS_HEADER)))
        return false;
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image.data());
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0 || !range(dos->e_lfanew, sizeof(IMAGE_NT_HEADERS64)))
        return false;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(image.data() + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64 ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC ||
        nt->FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER64) ||
        nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_IMPORT)
        return false;
    const auto directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!directory.VirtualAddress || !range(directory.VirtualAddress, directory.Size))
        return false;

    std::vector<Slot> found;
    for (size_t offset = 0; offset + sizeof(IMAGE_IMPORT_DESCRIPTOR) <= directory.Size;
         offset += sizeof(IMAGE_IMPORT_DESCRIPTOR))
    {
        const auto* descriptor =
            reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(image.data() + directory.VirtualAddress + offset);
        if (!descriptor->Name && !descriptor->FirstThunk && !descriptor->OriginalFirstThunk)
        {
            slots = std::move(found);
            return true;
        }
        if (!descriptor->FirstThunk)
            return false;
        // A bound-only descriptor has no import names left to inspect.
        if (!descriptor->OriginalFirstThunk)
            continue;
        for (size_t index = 0;; ++index)
        {
            const size_t nameOffset = descriptor->OriginalFirstThunk + index * sizeof(IMAGE_THUNK_DATA64);
            const size_t slotOffset = descriptor->FirstThunk + index * sizeof(IMAGE_THUNK_DATA64);
            if (!range(nameOffset, sizeof(IMAGE_THUNK_DATA64)) || !range(slotOffset, sizeof(IMAGE_THUNK_DATA64)))
                return false;
            const auto entry = reinterpret_cast<const IMAGE_THUNK_DATA64*>(image.data() + nameOffset)->u1.AddressOfData;
            if (!entry)
                break;
            if (IMAGE_SNAP_BY_ORDINAL64(entry))
                continue;
            if (!range(entry, sizeof(WORD) + 1))
                return false;
            const auto* name = reinterpret_cast<const char*>(image.data() + entry + sizeof(WORD));
            const auto* end = static_cast<const char*>(memchr(name, 0, image.size() - entry - sizeof(WORD)));
            if (!end)
                return false;
            const std::string_view imported(name, end - name);
            if (imported == "GetModuleFileNameW" || imported == "GetModuleFileNameA")
            {
                if (slotOffset % alignof(void*))
                    return false;
                found.push_back({ reinterpret_cast<void**>(image.data() + slotOffset), imported.back() == 'W' });
            }
        }
    }
    return false; // Unterminated import descriptor table.
}
} // namespace DlssNr::RuntimeImports
