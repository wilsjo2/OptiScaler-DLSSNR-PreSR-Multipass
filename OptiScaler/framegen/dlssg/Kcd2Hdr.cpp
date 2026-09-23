#include "pch.h"
#include "Kcd2Hdr.h"
#include <State.h>
#include <cstring>
#include <mutex>

namespace
{
// KCD2 1.5.6, WHGame.dll. r_HDROutput's callback overwrites r_HDRPipeline when the
// saved HDR setting is applied after user.cfg. Change only its supported-HDR branch
// from scRGB (1) to the game's own HDR10 pipeline (2). HDR-off and support checks stay intact.
constexpr DWORD kTimestamp = 0x6a350e20;
constexpr DWORD kImageSize = 0x5b2d000;
constexpr DWORD kCallbackRva = 0x1dec4c0;
constexpr size_t kSelectionOffset = 0x80;
constexpr unsigned char kCallback[] = {
    0x48, 0x89, 0x5c, 0x24, 0x18, 0x57, 0x48, 0x83, 0xec, 0x20, 0x48, 0x8b, 0xf9, 0x48, 0x8d, 0x15, 0x34, 0x87,
    0xfe, 0x01, 0x48, 0x8b, 0x0d, 0xcd, 0x13, 0xb4, 0x02, 0x48, 0x8b, 0x01, 0xff, 0x90, 0xb8, 0x00, 0x00, 0x00,
    0x48, 0x8b, 0xd8, 0x48, 0x85, 0xc0, 0x74, 0x78, 0x48, 0x8b, 0x17, 0x48, 0x8b, 0xcf, 0x48, 0x89, 0x6c, 0x24,
    0x30, 0x48, 0x89, 0x74, 0x24, 0x38, 0xff, 0x52, 0x10, 0x48, 0x8d, 0x0d, 0x7a, 0xe3, 0x4c, 0x03, 0x40, 0x32,
    0xf6, 0x8b, 0xe8, 0xe8, 0xa0, 0xc6, 0x6f, 0xfe, 0x48, 0x85, 0xc0, 0x74, 0x1c, 0x48, 0x8b, 0xc8, 0xe8, 0xbf,
    0x4d, 0xa7, 0xfe, 0x48, 0x85, 0xc0, 0x74, 0x0f, 0x48, 0x8b, 0x08, 0x48, 0x8b, 0x51, 0x40, 0x48, 0x8b, 0xc8,
    0xff, 0xd2, 0x0f, 0xb6, 0xf0, 0x85, 0xed, 0x48, 0x8b, 0x6c, 0x24, 0x30, 0x74, 0x0c, 0x40, 0x84, 0xf6, 0x74,
    0x07, 0xba, 0x01, 0x00, 0x00, 0x00, 0xeb, 0x10, 0x48, 0x8b, 0x07, 0x33, 0xd2, 0x48, 0x8b, 0xcf, 0xff, 0x50,
    0x38, 0xba, 0xff, 0xff, 0xff, 0xff, 0x48, 0x8b, 0x03, 0x48, 0x8b, 0xcb, 0xff, 0x50, 0x38, 0x48, 0x8b, 0x74,
    0x24, 0x38, 0x48, 0x8b, 0x5c, 0x24, 0x40, 0x48, 0x83, 0xc4, 0x20, 0x5f, 0xc3,
};
static_assert(kCallback[kSelectionOffset - 1] == 0xba && kCallback[kSelectionOffset] == 1);

bool PatchCallback(HMODULE module)
{
    auto* image = reinterpret_cast<unsigned char*>(module);
    auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < sizeof(IMAGE_DOS_HEADER) || dos->e_lfanew > 0x1000)
        return false;
    auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(image + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64 ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC || nt->FileHeader.TimeDateStamp != kTimestamp ||
        nt->OptionalHeader.SizeOfImage != kImageSize)
        return false;

    auto* callback = image + kCallbackRva;
    if (std::memcmp(callback, kCallback, sizeof(kCallback)) != 0)
        return false; // Unknown build or another mod owns this callback: leave it alone.

    auto* selection = callback + kSelectionOffset;
    DWORD previous = 0;
    if (!VirtualProtect(selection, 1, PAGE_EXECUTE_READWRITE, &previous))
        return false;
    *selection = 2;
    DWORD ignored = 0;
    if (!VirtualProtect(selection, 1, previous, &ignored))
        LOG_WARN("KCD2 HDR10: could not restore callback page protection (error {})", GetLastError());
    if (!FlushInstructionCache(GetCurrentProcess(), selection, 1))
        LOG_WARN("KCD2 HDR10: could not flush callback instruction cache (error {})", GetLastError());
    return true;
}
} // namespace

void Kcd2Hdr::ApplyQuirk()
{
    const auto& state = State::Instance();
    if (!state.gameQuirks[GameQuirk::Kcd2DlssgHdr10] || state.activeFgOutput != FGOutput::DLSSG ||
        state.activeFgNvngx != FGNvngxReplacement::None)
        return;
    auto module = GetModuleHandleW(L"WHGame.dll");
    if (!module)
        return;
    static std::once_flag once;
    std::call_once(
        once,
        [module]
        {
            if (PatchCallback(module))
                LOG_INFO("KCD2 HDR10 quirk: HDR enable now selects native HDR10 instead of scRGB for this session");
            else
                LOG_WARN("KCD2 HDR10 quirk not applied: WHGame.dll callback unrecognized or not writable");
        });
}
