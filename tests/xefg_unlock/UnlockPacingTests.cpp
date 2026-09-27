// tests/xefg_unlock/UnlockPacingTests.cpp - opt-in native XeFG unlock + pacing pin.
//
// WHAT THIS PROVES (and what it does not mirror):
//   - Sections A/B drive the ACTUAL headers OptiScaler/proxies/XeFGUnlock.h and
//     OptiScaler/proxies/XeFGPacing.h (compiled into this binary via a stub
//     include dir that only shadows SysUtils.h/Logger.h/Config.h - the two
//     headers under test are the real repo files) against a synthetic PE image
//     built at runtime: config OFF refuses without touching a byte; a mid-table
//     mismatch fails closed with every earlier patch rolled back; an unrecognised
//     build stamp/size refuses before any write and without pacing (T7 fail-closed
//     adaptation vs upstream warn-and-continue); MaxInterpolatedFrames
//     is baked into the U3/U4/U5 immediates on recognised builds; count 1 leaves the provider alone
//     and does not install pacing; pacing OFF refuses; pacing ON rewrites the
//     present thunk on an image carrying the expected bytes and a corrupt thunk
//     refuses. Every expected byte below is a literal transcribed from the header
//     contract, never re-derived from it at runtime.
//   - Sections C/D read the ACTUAL production sources given on the command line
//     and assert the wiring INSIDE the real function bodies: XeFGProxy::HookXeFG
//     must call XeFGUnlock::Apply before resolving any export, XeFGUnlock::Apply
//     must call XeFGPacing::Install, and XeFG_Dx12::Dispatch must feed the
//     provider through XeFGPacing::RenderTimeMs/NoteFedFrameTime. Production
//     Config.h/OptiScaler.ini/menu defaults and limits are pinned as literals.
//   - The runner (tests/xefg_unlock/run.ps1) runs this binary GREEN against the
//     real sources, then re-runs it against disposable copies with the Apply
//     call, the Install call, and the pacing feed each removed (each must fail
//     with its named marker). A seed that exits 0 means this pin no longer
//     detects its defect. GPU/game/provider DLL are never touched: the "provider"
//     is a VirtualAlloc buffer and the latch resets are test-only.
//
// Usage: UnlockPacingTests.exe [XeFG_Proxy.h] [XeFGUnlock.h] [XeFG_Dx12.cpp]
//                               [Config.h] [OptiScaler.ini] [menu_common.cpp]
// Exit codes: 0 = every pin held; 1 = a pin failed (PIN_* marker printed);
//             2 = environment broken (a source file is missing/unreadable).

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Test-only latch reset: XeFGUnlock::_applied is private and latches per process.
// Defining private->public around the REAL headers exposes it so each functional
// case below can start from a clean slate. windows.h and the CRT are already
// loaded above, and the stubs pulled in below declare no private sections, so the
// redefine affects exactly the two headers under test. Production code is untouched.
#define private public
#include <proxies/XeFGUnlock.h>
#include <proxies/XeFGPacing.h>
#undef private

namespace
{
int g_failures = 0;

void Pin(bool held, const char* marker, const char* detail)
{
    if (held)
        return;
    ++g_failures;
    std::printf("%s: %s\n", marker, detail);
}

std::string ReadFile(const char* path, bool& ok)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        ok = false;
        return {};
    }
    std::ostringstream text;
    text << in.rdbuf();
    ok = true;
    return text.str();
}

// Extracts the body (without the outer braces) of the function whose definition
// starts at `signature`. Strings, character literals and comments are skipped so
// braces inside them cannot corrupt the match. Empty when not found.
std::string FunctionBody(const std::string& source, const std::string& signature)
{
    const size_t at = source.find(signature);
    if (at == std::string::npos)
        return {};
    size_t open = source.find('{', at);
    if (open == std::string::npos)
        return {};
    size_t i = open + 1;
    int depth = 1;
    bool lineComment = false, blockComment = false, str = false, ch = false;
    for (; i < source.size(); ++i)
    {
        const char c = source[i];
        const char next = i + 1 < source.size() ? source[i + 1] : '\0';
        if (lineComment)
        {
            if (c == '\n')
                lineComment = false;
            continue;
        }
        if (blockComment)
        {
            if (c == '*' && next == '/')
            {
                blockComment = false;
                ++i;
            }
            continue;
        }
        if (str)
        {
            if (c == '\\')
                ++i;
            else if (c == '"')
                str = false;
            continue;
        }
        if (ch)
        {
            if (c == '\\')
                ++i;
            else if (c == '\'')
                ch = false;
            continue;
        }
        if (c == '/' && next == '/')
            lineComment = true;
        else if (c == '/' && next == '*')
            blockComment = true;
        else if (c == '"')
            str = true;
        else if (c == '\'')
            ch = true;
        else if (c == '{')
            ++depth;
        else if (c == '}')
        {
            if (--depth == 0)
                return source.substr(open + 1, i - open - 1);
        }
    }
    return {};
}

// --- synthetic provider image -------------------------------------------------
// A VirtualAlloc buffer shaped like libxess_fg.dll just enough for the unlock and
// pacing code paths: DOS + NT headers, one .text section covering every patch and
// thunk RVA, and caller-chosen stamp/size fields. Literals below are transcribed
// from OptiScaler/proxies/XeFGUnlock.h (patch table) and XeFGPacing.h (thunk RVAs).

constexpr uint32_t kKnownStamp = 0x69CB0F4D;
constexpr uint32_t kKnownImageSize = 0x015ED000;
constexpr uint32_t kTextVa = 0x1000;
constexpr uint32_t kTextSize = 0x300000;
constexpr size_t kAllocSize = 0x400000;

struct UnlockPatchLiteral
{
    uint32_t rva;
    uint8_t oldBytes[10];
    uint8_t newBytes[10];
    uint32_t size;
    int32_t immOffset;
};

constexpr UnlockPatchLiteral kPatches[] = {
    { 0x20DA4F, { 0x0F, 0x85, 0xCC, 0x00, 0x00, 0x00 }, { 0xE9, 0xCD, 0x00, 0x00, 0x00, 0x90 }, 6, -1 },
    { 0x1A5DE4, { 0x74, 0x09 }, { 0xEB, 0x06 }, 2, -1 },
    { 0x1A517D, { 0xBB, 0x03, 0x00, 0x00, 0x00 }, { 0xBB, 0x00, 0x00, 0x00, 0x00 }, 5, 1 },
    { 0x1A45C2,
      { 0xC7, 0x87, 0x6C, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00 },
      { 0xC7, 0x87, 0x6C, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
      10,
      6 },
    { 0x20973B, { 0xB8, 0x01, 0x00, 0x00, 0x00 }, { 0xB8, 0x00, 0x00, 0x00, 0x00 }, 5, 1 },
};

struct TestImage
{
    uint8_t* base = nullptr;

    bool valid() const { return base != nullptr; }

    static TestImage Create(uint32_t stamp, uint32_t imageSize, bool withPatchBytes, bool withThunkBytes)
    {
        TestImage image;
        image.base = static_cast<uint8_t*>(
            VirtualAlloc(nullptr, kAllocSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (image.base == nullptr)
            return image;
        memset(image.base, 0xCC, kAllocSize);

        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(image.base);
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 0x40;

        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(image.base + 0x40);
        nt->Signature = IMAGE_NT_SIGNATURE;
        nt->FileHeader.NumberOfSections = 1;
        nt->FileHeader.TimeDateStamp = stamp;
        nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER);
        nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR_MAGIC;
        nt->OptionalHeader.SizeOfImage = imageSize;

        auto* section = IMAGE_FIRST_SECTION(nt);
        memset(section->Name, 0, IMAGE_SIZEOF_SHORT_NAME);
        memcpy(section->Name, ".text", 5);
        section->VirtualAddress = kTextVa;
        section->Misc.VirtualSize = kTextSize;

        if (withPatchBytes)
        {
            for (const auto& patch : kPatches)
                memcpy(image.base + patch.rva, patch.oldBytes, patch.size);
        }

        if (withThunkBytes)
        {
            memcpy(image.base + XeFGPacing::PresentThunkRva, XeFGPacing::PresentThunkExpected,
                   XeFGPacing::PresentThunkSize);
            memcpy(image.base + XeFGPacing::SchedThunkRva, XeFGPacing::SchedThunkExpected,
                   XeFGPacing::PresentThunkSize);
            memcpy(image.base + XeFGPacing::TimestampThunkRva, XeFGPacing::TimestampThunkExpected,
                   XeFGPacing::PresentThunkSize);
        }

        return image;
    }

    void Destroy()
    {
        if (base != nullptr)
        {
            VirtualFree(base, 0, MEM_RELEASE);
            base = nullptr;
        }
    }

    uint32_t ReadU32(uint32_t rva) const
    {
        uint32_t value = 0;
        memcpy(&value, base + rva, sizeof(value));
        return value;
    }
};

void ResetState()
{
    XeFGUnlock::_applied = false;
    XeFGPacing::g_enabled = false;
    XeFGPacing::g_native = nullptr;
    XeFGUnlockTestLog::Clear();
}

void SetUnlockConfig(bool enabled, int maxInterp)
{
    Config::Instance()->FGXeFGUnlockEnabled.v = enabled;
    Config::Instance()->FGXeFGMaxInterpolatedFrames.v = maxInterp;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        std::printf("PIN_ENV: expected 6 source paths "
                    "(XeFG_Proxy.h XeFGUnlock.h XeFG_Dx12.cpp Config.h OptiScaler.ini menu_common.cpp), got %d\n",
                    argc - 1);
        return 2;
    }

    bool ok = false;
    const std::string proxy = ReadFile(argv[1], ok);
    if (!ok)
    {
        std::printf("PIN_ENV: unreadable XeFG_Proxy.h: %s\n", argv[1]);
        return 2;
    }
    const std::string unlock = ReadFile(argv[2], ok);
    if (!ok)
    {
        std::printf("PIN_ENV: unreadable XeFGUnlock.h: %s\n", argv[2]);
        return 2;
    }
    const std::string dx12 = ReadFile(argv[3], ok);
    if (!ok)
    {
        std::printf("PIN_ENV: unreadable XeFG_Dx12.cpp: %s\n", argv[3]);
        return 2;
    }
    const std::string configH = ReadFile(argv[4], ok);
    if (!ok)
    {
        std::printf("PIN_ENV: unreadable Config.h: %s\n", argv[4]);
        return 2;
    }
    const std::string ini = ReadFile(argv[5], ok);
    if (!ok)
    {
        std::printf("PIN_ENV: unreadable OptiScaler.ini: %s\n", argv[5]);
        return 2;
    }
    const std::string menu = ReadFile(argv[6], ok);
    if (!ok)
    {
        std::printf("PIN_ENV: unreadable menu_common.cpp: %s\n", argv[6]);
        return 2;
    }

    // --- Section A: real XeFGUnlock::Apply against synthetic PE -----------------
    {
        // F1: config OFF refuses and leaves every patch site untouched.
        ResetState();
        SetUnlockConfig(false, 5);
        TestImage image = TestImage::Create(kKnownStamp, kKnownImageSize, true, false);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F1)");
        if (image.valid())
        {
            const bool applied = XeFGUnlock::Apply(reinterpret_cast<HMODULE>(image.base));
            bool untouched = true;
            for (const auto& patch : kPatches)
            {
                if (memcmp(image.base + patch.rva, patch.oldBytes, patch.size) != 0)
                    untouched = false;
            }
            Pin(!applied && untouched, "PIN_UNLOCK_OFF", "OFF config must refuse without touching the image");
            image.Destroy();
        }
    }

    {
        // F2: a mid-table mismatch fails closed and rolls back earlier patches.
        ResetState();
        SetUnlockConfig(true, 5);
        TestImage image = TestImage::Create(kKnownStamp, kKnownImageSize, true, false);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F2)");
        if (image.valid())
        {
            image.base[kPatches[2].rva] = 0xFF; // corrupt U3 so U1+U2 apply, then abort
            const bool applied = XeFGUnlock::Apply(reinterpret_cast<HMODULE>(image.base));
            // U1/U2 must be rolled back to their original bytes; U3 keeps its
            // corrupt byte (never written); U4/U5 never touched.
            bool restored = memcmp(image.base + kPatches[0].rva, kPatches[0].oldBytes, kPatches[0].size) == 0 &&
                memcmp(image.base + kPatches[1].rva, kPatches[1].oldBytes, kPatches[1].size) == 0 &&
                image.base[kPatches[2].rva] == 0xFF &&
                memcmp(image.base + kPatches[3].rva, kPatches[3].oldBytes, kPatches[3].size) == 0 &&
                memcmp(image.base + kPatches[4].rva, kPatches[4].oldBytes, kPatches[4].size) == 0;
            Pin(!applied && restored, "PIN_UNLOCK_ROLLBACK",
                "partial failure must refuse with every earlier patch rolled back");
            image.Destroy();
        }
    }

    {
        // F3: recognised build applies; U3/U4/U5 carry the configured count.
        ResetState();
        SetUnlockConfig(true, 3);
        TestImage image = TestImage::Create(kKnownStamp, kKnownImageSize, true, false);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F3)");
        if (image.valid())
        {
            const bool applied = XeFGUnlock::Apply(reinterpret_cast<HMODULE>(image.base));
            const uint8_t u1Want[] = { 0xE9, 0xCD, 0x00, 0x00, 0x00, 0x90 };
            const uint8_t u2Want[] = { 0xEB, 0x06 };
            const bool bytes = memcmp(image.base + kPatches[0].rva, u1Want, sizeof(u1Want)) == 0 &&
                memcmp(image.base + kPatches[1].rva, u2Want, sizeof(u2Want)) == 0 &&
                image.ReadU32(kPatches[2].rva + 1) == 3 && image.ReadU32(kPatches[3].rva + 6) == 3 &&
                image.ReadU32(kPatches[4].rva + 1) == 3;
            Pin(applied && bytes, "PIN_UNLOCK_RECOGNISED",
                "recognised build must apply with count 3 in U3/U4/U5");
            image.Destroy();
        }
    }

    {
        // F4: unrecognised build identity (bad stamp/size) refuses BEFORE any
        // write and without installing pacing (T7 fail-closed). This differs
        // from upstream #107/bb1619ec warn-and-continue by reviewed adaptation.
        ResetState();
        SetUnlockConfig(true, 5);
        Config::Instance()->FGXeFGExtraPacing.v = true;
        TestImage image = TestImage::Create(0x12345678, 0x00999999, true, false);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F4)");
        if (image.valid())
        {
            const bool applied = XeFGUnlock::Apply(reinterpret_cast<HMODULE>(image.base));
            bool untouched = true;
            for (const auto& patch : kPatches)
            {
                if (memcmp(image.base + patch.rva, patch.oldBytes, patch.size) != 0)
                    untouched = false;
            }
            Pin(!applied && untouched && !XeFGPacing::g_enabled, "PIN_UNLOCK_BADSTAMP",
                "bad stamp must refuse with unchanged bytes and no pacing");
            image.Destroy();
        }
    }

    {
        // F5: count 1 is plain 2X the provider already does - leave it alone,
        // mark applied, and do NOT install pacing.
        ResetState();
        SetUnlockConfig(true, 1);
        Config::Instance()->FGXeFGExtraPacing.v = true;
        TestImage image = TestImage::Create(kKnownStamp, kKnownImageSize, true, false);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F5)");
        if (image.valid())
        {
            const bool applied = XeFGUnlock::Apply(reinterpret_cast<HMODULE>(image.base));
            bool untouched = true;
            for (const auto& patch : kPatches)
            {
                if (memcmp(image.base + patch.rva, patch.oldBytes, patch.size) != 0)
                    untouched = false;
            }
            Pin(applied && untouched && !XeFGPacing::g_enabled, "PIN_UNLOCK_MIN",
                "count 1 must succeed without patching and without pacing");
            image.Destroy();
        }
    }

    // --- Section B: real XeFGPacing::Install ------------------------------------
    {
        // F6: pacing OFF refuses.
        ResetState();
        Config::Instance()->FGXeFGExtraPacing.v = false;
        TestImage image = TestImage::Create(kKnownStamp, kKnownImageSize, false, true);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F6)");
        if (image.valid())
        {
            const bool installed = XeFGPacing::Install(image.base);
            Pin(!installed && !XeFGPacing::g_enabled, "PIN_PACING_OFF", "pacing OFF must refuse");
            image.Destroy();
        }
    }

    {
        // F7: pacing ON rewrites the present thunk on recognised bytes.
        ResetState();
        Config::Instance()->FGXeFGExtraPacing.v = true;
        TestImage image = TestImage::Create(kKnownStamp, kKnownImageSize, false, true);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F7)");
        if (image.valid())
        {
            const bool installed = XeFGPacing::Install(image.base);
            const bool rewritten = image.base[XeFGPacing::PresentThunkRva] == 0xFF &&
                image.base[XeFGPacing::PresentThunkRva + 1] == 0x25 &&
                memcmp(image.base + XeFGPacing::PresentThunkRva, XeFGPacing::PresentThunkExpected,
                       XeFGPacing::PresentThunkSize) != 0;
            Pin(installed && XeFGPacing::g_enabled && XeFGPacing::g_native != nullptr && rewritten,
                "PIN_PACING_ON", "pacing ON must hook the present thunk on recognised bytes");
            image.Destroy();
        }
    }

    {
        // F8: a corrupt thunk fails closed without enabling.
        ResetState();
        Config::Instance()->FGXeFGExtraPacing.v = true;
        TestImage image = TestImage::Create(kKnownStamp, kKnownImageSize, false, true);
        Pin(image.valid(), "PIN_ENV", "synthetic image allocation failed (F8)");
        if (image.valid())
        {
            image.base[XeFGPacing::PresentThunkRva] = 0x00;
            const bool installed = XeFGPacing::Install(image.base);
            Pin(!installed && !XeFGPacing::g_enabled, "PIN_PACING_ON",
                "corrupt present thunk must refuse pacing");
            image.Destroy();
        }
    }

    // --- Section C: production call-site bindings --------------------------------
    {
        // HookXeFG must patch before resolving: inside the actual HookXeFG
        // definition body, XeFGUnlock::Apply precedes the first export lookup.
        const std::string hookBody = FunctionBody(proxy, "HookXeFG(HMODULE libxefgModule)");
        const size_t apply = hookBody.find("XeFGUnlock::Apply(");
        const size_t firstExport = hookBody.find("GetProcAddress");
        Pin(!hookBody.empty() && apply != std::string::npos && firstExport != std::string::npos &&
                apply < firstExport,
            "PIN_APPLY_ORDER", "HookXeFG must call XeFGUnlock::Apply before any export lookup");
    }

    {
        // The unlock must hand over to pacing: Apply calls XeFGPacing::Install.
        const std::string body = FunctionBody(unlock, "Apply(HMODULE module)");
        Pin(!body.empty() && body.find("XeFGPacing::Install(") != std::string::npos, "PIN_UNLOCK_PACING",
            "XeFGUnlock::Apply must call XeFGPacing::Install");
    }

    {
        // The Dx12 backend must consume pacing: Dispatch feeds RenderTimeMs and
        // reports the fed value back for the pacing report pairing.
        const std::string body = FunctionBody(dx12, "XeFG_Dx12::Dispatch()");
        Pin(!body.empty() && body.find("XeFGPacing::RenderTimeMs()") != std::string::npos &&
                body.find("XeFGPacing::NoteFedFrameTime(") != std::string::npos,
            "PIN_DX12_PACING", "XeFG_Dx12::Dispatch must consume XeFGPacing feed and report");
    }

    // --- Section D: production defaults and limits --------------------------------
    Pin(configH.find("FGXeFGUnlockEnabled { false }") != std::string::npos, "PIN_DEFAULT_OFF",
        "UnlockMFG must default OFF in Config.h");
    Pin(configH.find("FGXeFGMaxInterpolatedFrames { 5 }") != std::string::npos, "PIN_DEFAULT_MAX",
        "MaxInterpolatedFrames must default to 5 in Config.h");
    Pin(configH.find("FGXeFGExtraPacing { true }") != std::string::npos, "PIN_DEFAULT_PACING",
        "ExtraPacing must default ON in Config.h");
    Pin(configH.find("XeFGMaxInterpolations = 31") != std::string::npos, "PIN_LIMIT_31",
        "XeFGMaxInterpolations bound must be 31 in Config.h");
    Pin(ini.find("UnlockMFG=auto") != std::string::npos && ini.find("MaxInterpolatedFrames=auto") != std::string::npos &&
            ini.find("ExtraPacing=auto") != std::string::npos,
        "PIN_INI_DEFAULTS", "OptiScaler.ini must ship UnlockMFG/MaxInterpolatedFrames/ExtraPacing as auto");
    Pin(menu.find("Checkbox(\"Unlock MFG\"") != std::string::npos &&
            menu.find("Checkbox(\"Extra Pacing\"") != std::string::npos,
        "PIN_MENU_UI", "menu must offer Unlock MFG and Extra Pacing checkboxes");

    if (g_failures == 0)
        std::printf("XEFG unlock/pacing pins held: OFF, rollback, bad-stamp, min-1, pacing OFF/ON, "
                    "thunk-refusal, Apply/Install/Dispatch bindings, defaults.\n");

    return g_failures == 0 ? 0 : 1;
}
