// Host check of MfgUnlockFlip.h: deriving the flip-metering fallback state from a Streamline DLSS-G plugin
// image and pinning the stores to it. No GPU and no game needed; the plugin image is built in memory.
// cl /std:c++20 /EHsc tests/mfg_flipmeter_smoke.cpp
#include "../OptiScaler/framegen/dlssg/MfgUnlockFlip.h"

#include <cstdio>

using namespace MfgUnlock::Flip;

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

// .text (executable, NOP-filled) at 0x1000 and .rdata at 0x2000, 0x1000 and 0x400 long, in a 0x3000 image.
// The message is at 0x2100 and one `lea rax,[rip+disp32]` at 0x1100 points at it.
struct Plugin
{
    static constexpr size_t kText = 0x1000;
    static constexpr size_t kRdata = 0x2000;
    static constexpr size_t kMessage = 0x2100;
    static constexpr size_t kLea = 0x1100;

    std::vector<uint8_t> bytes = std::vector<uint8_t>(0x3000, 0);

    Plugin()
    {
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes.data());
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 0x80;
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + 0x80);
        nt->Signature = IMAGE_NT_SIGNATURE;
        nt->FileHeader.NumberOfSections = 2;
        nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
        nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        nt->OptionalHeader.SizeOfImage = 0x3000;
        auto* s = IMAGE_FIRST_SECTION(nt);
        std::memcpy(s[0].Name, ".text", 5);
        s[0].VirtualAddress = kText;
        s[0].Misc.VirtualSize = 0x1000;
        s[0].Characteristics = IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE;
        std::memcpy(s[1].Name, ".rdata", 6);
        s[1].VirtualAddress = kRdata;
        s[1].Misc.VirtualSize = 0x400;
        s[1].Characteristics = IMAGE_SCN_MEM_READ;

        std::fill(bytes.begin() + kText, bytes.begin() + kText + 0x1000, uint8_t { 0x90 });
        put(kMessage, kMarker.data(), kMarker.size());
        lea(kLea, kMessage);
    }

    void put(size_t at, const void* p, size_t n) { std::memcpy(bytes.data() + at, p, n); }

    // lea rax,[rip+disp32] -> `to`
    void lea(size_t at, size_t to, uint8_t rex = 0x48, uint8_t modrm = 0x05)
    {
        const int32_t disp = static_cast<int32_t>(to) - static_cast<int32_t>(at + 7);
        const uint8_t code[3] = { rex, 0x8D, modrm };
        put(at, code, 3);
        put(at + 3, &disp, 4);
    }

    // mov byte ptr [reg+field], imm8: C6 /reg disp32 imm8
    void c6(size_t at, uint32_t field, uint8_t imm, uint8_t modrm = 0x83)
    {
        const uint8_t code[2] = { 0xC6, modrm };
        put(at, code, 2);
        put(at + 2, &field, 4);
        bytes[at + 6] = imm;
    }

    // mov byte ptr [reg+field], reg8: <rex> 88 /r disp32
    void r88(size_t at, uint32_t field, uint8_t modrm = 0xBB, uint8_t rex = 0x40)
    {
        const uint8_t code[3] = { rex, 0x88, modrm };
        put(at, code, 3);
        put(at + 3, &field, 4);
    }

    void* data() { return bytes.data(); }
    IMAGE_NT_HEADERS64* nt() { return reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes.data() + 0x80); }
};

// A plugin as the real one looks: the fallback writes 1 to +0x44f0 right after the log line, and one
// register store elsewhere writes the same field.
static Plugin Typical()
{
    Plugin p;
    p.c6(Plugin::kLea + 0x40, 0x44F0, 1);
    p.c6(Plugin::kLea + 0x80, 0x44F0, 1);
    p.r88(0x1400, 0x44F0);
    return p;
}

int main()
{
    Plan plan;

    // The typical case: the state is read from the fallback, the register store is rewritten whole.
    {
        Plugin p = Typical();
        CHECK(FindPlan(p.data(), plan) == FindResult::Found);
        CHECK(plan.field == 0x44F0 && plan.value == 1);
        CHECK(plan.sites.size() == 1);
        CHECK(plan.sites[0].address == p.bytes.data() + 0x1400);
        CHECK(plan.sites[0].fromRegister);
        const uint8_t want[7] = { 0xC6, 0x83, 0xF0, 0x44, 0x00, 0x00, 0x01 };
        CHECK(std::memcmp(plan.sites[0].replacement, want, 7) == 0);
    }

    // The base register survives the rewrite (modrm rm bits), and a polarity of 0 is followed.
    {
        Plugin p;
        p.c6(Plugin::kLea + 0x40, 0x38BC, 0);
        p.r88(0x1400, 0x38BC, 0x87); // mod=10 reg=0 rm=7
        CHECK(FindPlan(p.data(), plan) == FindResult::Found);
        CHECK(plan.field == 0x38BC && plan.value == 0);
        const uint8_t want[7] = { 0xC6, 0x87, 0xBC, 0x38, 0x00, 0x00, 0x00 };
        CHECK(plan.sites.size() == 1 && std::memcmp(plan.sites[0].replacement, want, 7) == 0);
    }

    // A C6 store of the opposite value has its immediate flipped, and nothing else about it.
    {
        Plugin p = Typical();
        p.c6(0x1500, 0x44F0, 0, 0x85);
        CHECK(FindPlan(p.data(), plan) == FindResult::Found);
        CHECK(plan.sites.size() == 2);
        const auto* c6 = plan.sites[0].fromRegister ? &plan.sites[1] : &plan.sites[0];
        CHECK(!c6->fromRegister && c6->address == p.bytes.data() + 0x1500);
        CHECK(c6->replacement[0] == 0xC6 && c6->replacement[1] == 0x85 && c6->replacement[6] == 1);
        CHECK(std::memcmp(c6->replacement, c6->original, 6) == 0);
    }
    // A C6 store that already writes the wanted value needs nothing; other fields are left alone.
    {
        Plugin p = Typical();
        p.c6(0x1500, 0x44F0, 1);
        p.c6(0x1600, 0x5000, 0);
        p.r88(0x1700, 0x5000);
        CHECK(FindPlan(p.data(), plan) == FindResult::Found && plan.sites.size() == 1);
    }

    // Nothing to patch at all: reported, not claimed.
    {
        Plugin p;
        p.c6(Plugin::kLea + 0x40, 0x44F0, 1);
        CHECK(FindPlan(p.data(), plan) == FindResult::NothingToPatch);
        CHECK(plan.sites.empty());
    }

    // The message: absent, or more than once.
    {
        Plugin p = Typical();
        std::memset(p.bytes.data() + Plugin::kMessage, 0, kMarker.size());
        CHECK(FindPlan(p.data(), plan) == FindResult::NoMarker);
        Plugin q = Typical();
        q.put(0x2200, kMarker.data(), kMarker.size());
        CHECK(FindPlan(q.data(), plan) == FindResult::AmbiguousMarker);
    }

    // The reference: none, two, or the wrong kind of instruction.
    {
        Plugin p = Typical();
        std::fill(p.bytes.begin() + Plugin::kLea, p.bytes.begin() + Plugin::kLea + 7, uint8_t { 0x90 });
        CHECK(FindPlan(p.data(), plan) == FindResult::NoReference);
        Plugin q = Typical();
        q.lea(0x1800, Plugin::kMessage);
        CHECK(FindPlan(q.data(), plan) == FindResult::AmbiguousReference);
        Plugin r = Typical();
        r.lea(Plugin::kLea, Plugin::kMessage, 0x48, 0x0D); // rip-relative needs mod=00 rm=101; 0D is reg=1, still one
        CHECK(FindPlan(r.data(), plan) == FindResult::Found);
        Plugin s = Typical();
        s.lea(Plugin::kLea, Plugin::kMessage, 0x49, 0x05); // REX.B changes the encoding: not matched
        CHECK(FindPlan(s.data(), plan) == FindResult::NoReference);
        Plugin t = Typical();
        t.lea(Plugin::kLea, Plugin::kMessage + 8); // points elsewhere
        CHECK(FindPlan(t.data(), plan) == FindResult::NoReference);
    }

    // The fallback window: none, disagreeing, out of range, wrong shape, or too far away.
    {
        Plugin p;
        p.r88(0x1400, 0x44F0);
        CHECK(FindPlan(p.data(), plan) == FindResult::NoFallback);

        Plugin q;
        q.c6(Plugin::kLea + 0x40, 0x44F0, 1);
        q.c6(Plugin::kLea + 0x80, 0x44F0, 0);
        q.r88(0x1400, 0x44F0);
        CHECK(FindPlan(q.data(), plan) == FindResult::ConflictingFallback);

        Plugin r;
        r.c6(Plugin::kLea + 0x40, 0x44F0, 1);
        r.c6(Plugin::kLea + 0x80, 0x44F8, 1);
        CHECK(FindPlan(r.data(), plan) == FindResult::ConflictingFallback);

        for (uint32_t field : { 0x100u, 0x20000u, 0x40u })
        {
            Plugin f;
            f.c6(Plugin::kLea + 0x40, field, 1);
            f.r88(0x1400, field);
            CHECK(FindPlan(f.data(), plan) == FindResult::NoFallback);
        }

        Plugin i;
        i.c6(Plugin::kLea + 0x40, 0x44F0, 2); // an immediate that is not a flag
        i.r88(0x1400, 0x44F0);
        CHECK(FindPlan(i.data(), plan) == FindResult::NoFallback);

        Plugin sib;
        sib.c6(Plugin::kLea + 0x40, 0x44F0, 1, 0x84); // rm=100: a SIB byte follows, the layout differs
        sib.r88(0x1400, 0x44F0);
        CHECK(FindPlan(sib.data(), plan) == FindResult::NoFallback);

        Plugin reg1;
        reg1.c6(Plugin::kLea + 0x40, 0x44F0, 1, 0x8B); // C6 /1 is not a mov
        reg1.r88(0x1400, 0x44F0);
        CHECK(FindPlan(reg1.data(), plan) == FindResult::NoFallback);

        Plugin tooFar;
        tooFar.c6(Plugin::kLea + kWindow, 0x44F0, 1); // starts at the window's end: out
        tooFar.r88(0x1400, 0x44F0);
        CHECK(FindPlan(tooFar.data(), plan) == FindResult::NoFallback);
        Plugin lastFit;
        lastFit.c6(Plugin::kLea + kWindow - kStoreLength, 0x44F0, 1); // the last position that fits
        lastFit.r88(0x1400, 0x44F0);
        CHECK(FindPlan(lastFit.data(), plan) == FindResult::Found);
    }

    // Register stores of the wrong shape are not rewritten.
    {
        for (int variant = 0; variant < 4; ++variant)
        {
            Plugin p;
            p.c6(Plugin::kLea + 0x40, 0x44F0, 1);
            switch (variant)
            {
            case 0:
                p.r88(0x1400, 0x44F0, 0xBB, 0x41); // REX.B: eight bytes with the same meaning
                break;
            case 1:
                p.r88(0x1400, 0x44F0, 0xBC); // SIB follows
                break;
            case 2:
                p.r88(0x1400, 0x44F0, 0x3B); // mod=00, a shorter form
                break;
            case 3:
            {
                const uint8_t bare[6] = { 0x88, 0xBB, 0xF0, 0x44, 0x00, 0x00 }; // no REX
                p.put(0x1400, bare, sizeof(bare));
                break;
            }
            }
            CHECK(FindPlan(p.data(), plan) == FindResult::NothingToPatch);
        }
    }

    // A section boundary: a store cut off by the end of the code section is not read past it.
    {
        Plugin p = Typical();
        p.bytes[0x1400] = 0x90;
        p.bytes[0x1400 + 1] = 0x90;
        p.bytes[0x1400 + 2] = 0x90; // remove the ordinary register store
        p.r88(Plugin::kText + 0x1000 - 6, 0x44F0); // only six bytes left in the section
        CHECK(FindPlan(p.data(), plan) == FindResult::NothingToPatch);
    }

    // Too many stores.
    {
        Plugin p;
        p.c6(Plugin::kLea + 0x40, 0x44F0, 1);
        for (size_t i = 0; i < kMaxSites; ++i)
            p.r88(0x1400 + i * 0x10, 0x44F0);
        CHECK(FindPlan(p.data(), plan) == FindResult::Found && plan.sites.size() == kMaxSites);
        p.r88(0x1400 + kMaxSites * 0x10, 0x44F0);
        CHECK(FindPlan(p.data(), plan) == FindResult::TooManySites && plan.sites.empty());
    }

    // Not a module.
    {
        Plugin p = Typical();
        p.bytes[0] = 'X';
        CHECK(FindPlan(p.data(), plan) == FindResult::BadImage);
        CHECK(FindPlan(nullptr, plan) == FindResult::BadImage);
    }

    // Applying: writes exactly the replacement bytes, once, and refuses when a site has changed.
    {
        Plugin p = Typical();
        p.c6(0x1500, 0x44F0, 0, 0x85);
        const std::vector<uint8_t> before = p.bytes;
        CHECK(FindPlan(p.data(), plan) == FindResult::Found);
        CHECK(Apply(plan) == ApplyResult::Patched);

        size_t changed = 0;
        for (size_t i = 0; i < before.size(); ++i)
            changed += before[i] != p.bytes[i];
        CHECK(changed == 6 + 1); // the register store (six of its seven bytes differ) and one immediate
        CHECK(std::memcmp(p.bytes.data() + 0x1400, "\xC6\x83\xF0\x44\x00\x00\x01", 7) == 0);
        CHECK(p.bytes[0x1500 + 6] == 1);

        // Applying again: the bytes are no longer what was read.
        CHECK(Apply(plan) == ApplyResult::Mismatch);
        // And a fresh look finds nothing left to change.
        CHECK(FindPlan(p.data(), plan) == FindResult::NothingToPatch);
    }
    {
        Plugin p = Typical();
        p.c6(0x1500, 0x44F0, 0, 0x85);
        CHECK(FindPlan(p.data(), plan) == FindResult::Found);
        p.bytes[0x1500 + 2] ^= 1; // the second site changed after the plan was made
        const std::vector<uint8_t> before = p.bytes;
        CHECK(Apply(plan) == ApplyResult::Mismatch);
        CHECK(p.bytes == before); // nothing was written, not even the first site
    }

    if (fails == 0)
        printf("mfg_flipmeter_smoke: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
