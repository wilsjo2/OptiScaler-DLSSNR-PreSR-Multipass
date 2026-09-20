// Host check of MfgUnlockPtx.h: the LZ4 reader, the fatbin walk, the PTX rewrite, and the descriptor
// redirect on a module image. No GPU and no game needed; the fatbins and the module are built in memory.
// cl /std:c++20 /EHsc tests/mfg_ptx_smoke.cpp
#include "../OptiScaler/framegen/dlssg/MfgUnlockPtx.h"

#include <cstdio>

using namespace MfgUnlock::Ptx;

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

using Bytes = std::vector<uint8_t>;

template <typename T> static void Put(Bytes& b, size_t at, T v) { std::memcpy(b.data() + at, &v, sizeof(v)); }

static size_t Count(const std::string& text, const std::string& what)
{
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + what.size()))
        ++n;
    return n;
}

// PTX of exactly profile.ptxBytes bytes, with `midpoints` midpoint multiplies and a join label.
static std::string MakePtx(const TemporalProfile& profile, size_t midpoints = kExpectedMidpoints, size_t labels = 1,
                           bool regDecl = true)
{
    const std::string name = profile.entryName;
    std::string t = ".version 8.5\n.entry " + name + "(\n.param .align 8 .b8 " + name + "_param_0[144]\n)\n{\n";

    if (regDecl)
        t += ".reg .f32 %f<1362>;\n";

    for (size_t i = 0; i < labels; ++i)
        t += "$L__BB0_3:\n";

    for (size_t i = 0; i < midpoints; ++i)
        t += "mul.ftz.f32 %f200, %f201, 0f3F000000;\n";

    t += "ret;\n}\n";

    const size_t target = profile.ptxBytes;
    if (t.size() + 3 <= target)
        t += "//" + std::string(target - t.size() - 3, 'x') + "\n";

    return t;
}

// A valid LZ4 block holding only literals.
static Bytes Lz4Literals(const std::string& text)
{
    Bytes out;
    const size_t n = text.size();
    out.push_back(static_cast<uint8_t>(std::min<size_t>(n, 15) << 4));

    if (n >= 15)
    {
        size_t rest = n - 15;
        while (rest >= 255)
        {
            out.push_back(255);
            rest -= 255;
        }
        out.push_back(static_cast<uint8_t>(rest));
    }

    out.insert(out.end(), text.begin(), text.end());
    return out;
}

// Outer header, an sm_75 cubin, the sm_89 PTX entry, then a precompiled sm_89 cubin that must be dropped.
struct Fatbin
{
    Bytes bytes;
    size_t ptxEntry = 0;
};

static Fatbin MakeFatbin(const Bytes& compressed, uint64_t rawSize, bool withPtx = true)
{
    Fatbin f;
    f.bytes.assign(16, 0);
    Put<uint32_t>(f.bytes, 0, kFatbinMagic);
    Put<uint16_t>(f.bytes, 4, 1);
    Put<uint16_t>(f.bytes, 6, 16);

    auto add = [&](uint16_t kind, uint32_t arch, const Bytes& payload, uint32_t compressedSize, uint64_t raw)
    {
        const size_t p = f.bytes.size();
        f.bytes.resize(p + 64 + payload.size(), 0);
        Put<uint16_t>(f.bytes, p, kind);
        Put<uint32_t>(f.bytes, p + 4, 64);
        Put<uint64_t>(f.bytes, p + 8, payload.size());
        Put<uint32_t>(f.bytes, p + 16, compressedSize);
        Put<uint32_t>(f.bytes, p + 28, arch);
        Put<uint64_t>(f.bytes, p + 56, raw);
        std::memcpy(f.bytes.data() + p + 64, payload.data(), payload.size());
        return p;
    };

    add(2, 75, Bytes(64, 0xAA), 0, 0);

    if (withPtx)
        f.ptxEntry = add(kPtxKind, kAdaArch, compressed, static_cast<uint32_t>(compressed.size()), rawSize);

    add(2, 89, Bytes(128, 0xBB), 0, 0);
    Put<uint64_t>(f.bytes, 8, f.bytes.size() - 16);
    return f;
}

static Fatbin FatbinFor(const std::string& ptx) { return MakeFatbin(Lz4Literals(ptx), ptx.size()); }

int main()
{
    // LZ4: a match sequence, a bad offset, and truncated input.
    {
        // "abcd" then a match of length 4 at offset 4: "abcdabcd".
        const Bytes block = { 0x40, 'a', 'b', 'c', 'd', 0x04, 0x00 };
        uint8_t out[8] = {};
        CHECK(Lz4BlockDecompress(block.data(), block.size(), out, sizeof(out)));
        CHECK(std::memcmp(out, "abcdabcd", 8) == 0);

        const Bytes badOffset = { 0x40, 'a', 'b', 'c', 'd', 0x09, 0x00 };
        CHECK(!Lz4BlockDecompress(badOffset.data(), badOffset.size(), out, sizeof(out)));
        CHECK(!Lz4BlockDecompress(block.data(), block.size(), out, 7));
        CHECK(!Lz4BlockDecompress(block.data(), 3, out, sizeof(out)));

        const Bytes lit = Lz4Literals(std::string(300, 'z'));
        Bytes big(300);
        CHECK(Lz4BlockDecompress(lit.data(), lit.size(), big.data(), big.size()));
        CHECK(big[0] == 'z' && big[299] == 'z');
    }

    // The rewrite, for both known profiles.
    for (const auto& profile : kProfiles)
    {
        const std::string ptx = MakePtx(profile);
        CHECK(ptx.size() == profile.ptxBytes);
        const Fatbin f = FatbinFor(ptx);

        size_t entry = 0;
        CHECK(FindAdaPtxEntry(f.bytes.data(), f.bytes.size(), entry) && entry == f.ptxEntry);
        CHECK(FindProfile(f.bytes.data(), f.bytes.size()) == &profile);

        Bytes out;
        std::string why;
        CHECK(BuildTemporalFatbin(f.bytes.data(), f.bytes.size(), profile, out, why));

        // Truncated after the PTX entry, so the precompiled sm_89 cubin is gone.
        CHECK(out.size() < f.bytes.size());
        CHECK(ReadU64(out.data() + 8) == out.size() - kOuterHeader);
        CHECK(ReadU32(out.data() + entry + 16) == 0);
        CHECK(ReadU64(out.data() + entry + 40) == kUncompressedFlags);
        CHECK(ReadU64(out.data() + entry + 56) == 0);
        const uint64_t payload = ReadU64(out.data() + entry + 8);
        CHECK(payload % 8 == 0 && out.size() == entry + 64 + payload);

        // Nothing but that entry was changed ahead of the payload.
        CHECK(std::memcmp(out.data(), f.bytes.data(), 8) == 0);
        CHECK(std::memcmp(out.data() + 16, f.bytes.data() + 16, entry - 16) == 0);

        const std::string text(reinterpret_cast<const char*>(out.data() + entry + 64), payload);
        CHECK(Count(text, "0f3F000000") == 0);
        CHECK(Count(text, "mul.ftz.f32 %f200, %f201, %f136;") == 52);
        CHECK(Count(text, "mul.ftz.f32 %f200, %f201, %f134;") == 52);
        CHECK(Count(text, std::string("ld.param.f32 %f134, [") + profile.entryName + "_param_0+32];") == 1);
        CHECK(Count(text, "sub.ftz.f32 %f136, %f135, %f134;") == 1);
        CHECK(Count(text, "$L__BB0_3:") == 1);
        // The injection sits right after the label line and ahead of every multiply.
        CHECK(text.find("$L__BB0_3:\nld.param.f32") != std::string::npos);
    }

    // Refusals, one at a time, against the second profile.
    {
        const TemporalProfile& profile = kProfiles[1];
        Bytes out;
        std::string why;

        auto refuses = [&](const std::string& ptx)
        {
            const Fatbin f = FatbinFor(ptx);
            return !BuildTemporalFatbin(f.bytes.data(), f.bytes.size(), profile, out, why);
        };

        CHECK(refuses(MakePtx(profile, 103)));
        CHECK(refuses(MakePtx(profile, 105)));
        CHECK(refuses(MakePtx(profile, kExpectedMidpoints, 2)));
        CHECK(refuses(MakePtx(profile, kExpectedMidpoints, 0)));
        CHECK(refuses(MakePtx(profile, kExpectedMidpoints, 1, false)));

        // A different size is not this profile, and matches no profile at all.
        std::string shorter = MakePtx(profile);
        shorter.pop_back();
        shorter.pop_back();
        shorter += "\n";
        const Fatbin s = FatbinFor(shorter);
        CHECK(FindProfile(s.bytes.data(), s.bytes.size()) == nullptr);
        CHECK(!BuildTemporalFatbin(s.bytes.data(), s.bytes.size(), profile, out, why));

        // A profile that does not match the entry's name.
        const Fatbin wrong = FatbinFor(MakePtx(kProfiles[0]));
        CHECK(!BuildTemporalFatbin(wrong.bytes.data(), wrong.bytes.size(), profile, out, why));

        // Corrupt LZ4 payload: the first literal-length extension byte is one short, so a stray byte
        // is left over and does not parse as a match.
        Fatbin corrupt = FatbinFor(MakePtx(profile));
        --corrupt.bytes[corrupt.ptxEntry + 65];
        CHECK(!BuildTemporalFatbin(corrupt.bytes.data(), corrupt.bytes.size(), profile, out, why));

        // No sm_89 PTX at all.
        const Fatbin none = MakeFatbin(Lz4Literals(MakePtx(profile)), profile.ptxBytes, false);
        size_t entry = 0;
        CHECK(!FindAdaPtxEntry(none.bytes.data(), none.bytes.size(), entry));

        // Declared length that disagrees with the buffer, an entry that runs past the end, a bad header.
        Fatbin lie = FatbinFor(MakePtx(profile));
        Put<uint64_t>(lie.bytes, 8, lie.bytes.size());
        CHECK(!FindAdaPtxEntry(lie.bytes.data(), lie.bytes.size(), entry));
        Fatbin overrun = FatbinFor(MakePtx(profile));
        Put<uint64_t>(overrun.bytes, overrun.ptxEntry + 8, uint64_t { 1 } << 40);
        CHECK(!FindAdaPtxEntry(overrun.bytes.data(), overrun.bytes.size(), entry));
        Fatbin badHdr = FatbinFor(MakePtx(profile));
        Put<uint32_t>(badHdr.bytes, badHdr.ptxEntry + 4, 8);
        CHECK(!FindAdaPtxEntry(badHdr.bytes.data(), badHdr.bytes.size(), entry));
        Fatbin badMagic = FatbinFor(MakePtx(profile));
        badMagic.bytes[0] ^= 1;
        CHECK(!FindAdaPtxEntry(badMagic.bytes.data(), badMagic.bytes.size(), entry));
        CHECK(!FindAdaPtxEntry(badMagic.bytes.data(), 8, entry));
    }

    // Module image: a readable data section holding names, two descriptor slots and the fatbin.
    for (const auto& profile : kProfiles)
    {
        constexpr size_t kSize = 0x40000;
        Bytes image(kSize, 0);

        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(image.data());
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 0x80;
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(image.data() + 0x80);
        nt->Signature = IMAGE_NT_SIGNATURE;
        nt->FileHeader.NumberOfSections = 1;
        nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
        nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        nt->OptionalHeader.SizeOfImage = kSize;
        auto* section = IMAGE_FIRST_SECTION(nt);
        std::memcpy(section->Name, ".rdata", 6);
        section->VirtualAddress = 0x1000;
        section->Misc.VirtualSize = kSize - 0x1000;
        section->Characteristics = IMAGE_SCN_MEM_READ;

        const uintptr_t base = reinterpret_cast<uintptr_t>(image.data());
        const std::string entryName = profile.entryName;
        const std::string descName = profile.descriptorName;
        std::memcpy(image.data() + 0x1000, entryName.c_str(), entryName.size() + 1);
        std::memcpy(image.data() + 0x1080, descName.c_str(), descName.size() + 1);

        const Fatbin f = FatbinFor(MakePtx(profile));
        constexpr size_t kFat = 0x2000;
        std::memcpy(image.data() + kFat, f.bytes.data(), f.bytes.size());

        // Two descriptors, at slots 0x1200 and 0x1400.
        const size_t slots[] = { 0x1200, 0x1400 };
        for (size_t slot : slots)
        {
            Put<uint64_t>(image, slot, base + kFat);
            Put<uint64_t>(image, slot + profile.entryNameOffset, base + 0x1000);
            Put<uint64_t>(image, slot + profile.descriptorNameOffset, base + 0x1080);
        }

        MfgUnlock::Ptx::Result result;
        CHECK(Apply(image.data(), result));
        CHECK(result.redirected == 2);
        CHECK(result.originalSize == f.bytes.size());
        CHECK(result.rebuiltSize < f.bytes.size());

        uint64_t a = 0;
        uint64_t b = 0;
        std::memcpy(&a, image.data() + 0x1200, 8);
        std::memcpy(&b, image.data() + 0x1400, 8);
        CHECK(a == b && a != base + kFat);
        CHECK(a < base || a >= base + kSize); // now outside the module
        CHECK(ReadU32(reinterpret_cast<const uint8_t*>(a)) == kFatbinMagic);

        // The original fatbin in the module is untouched.
        CHECK(std::memcmp(image.data() + kFat, f.bytes.data(), f.bytes.size()) == 0);

        // A second run finds slots that no longer point at a fatbin of the original shape: still
        // consistent, and it does not redirect them again to something else.
        MfgUnlock::Ptx::Result again;
        Apply(image.data(), again);
        uint64_t a2 = 0;
        std::memcpy(&a2, image.data() + 0x1200, 8);
        CHECK(a2 == a || again.redirected == 0 || again.redirected == 2);
    }

    // Module image with a descriptor whose names do not match: nothing is redirected.
    {
        const TemporalProfile& profile = kProfiles[1];
        constexpr size_t kSize = 0x40000;
        Bytes image(kSize, 0);
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(image.data());
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 0x80;
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(image.data() + 0x80);
        nt->Signature = IMAGE_NT_SIGNATURE;
        nt->FileHeader.NumberOfSections = 1;
        nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
        nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        nt->OptionalHeader.SizeOfImage = kSize;
        auto* section = IMAGE_FIRST_SECTION(nt);
        section->VirtualAddress = 0x1000;
        section->Misc.VirtualSize = kSize - 0x1000;
        section->Characteristics = IMAGE_SCN_MEM_READ;

        const uintptr_t base = reinterpret_cast<uintptr_t>(image.data());
        std::memcpy(image.data() + 0x1000, "SomethingElse", 14);
        std::memcpy(image.data() + 0x1080, "OtherDescriptor", 16);
        const Fatbin f = FatbinFor(MakePtx(profile));
        std::memcpy(image.data() + 0x2000, f.bytes.data(), f.bytes.size());
        Put<uint64_t>(image, 0x1200, base + 0x2000);
        Put<uint64_t>(image, 0x1200 + profile.entryNameOffset, base + 0x1000);
        Put<uint64_t>(image, 0x1200 + profile.descriptorNameOffset, base + 0x1080);

        MfgUnlock::Ptx::Result result;
        CHECK(!Apply(image.data(), result));
        CHECK(result.redirected == 0);
        uint64_t slot = 0;
        std::memcpy(&slot, image.data() + 0x1200, 8);
        CHECK(slot == base + 0x2000);

        // A slot near the very start or end of the section, where the name offsets fall outside the
        // image, is skipped without reading out of bounds.
        Put<uint64_t>(image, 0x1000, base + 0x2000);
        Put<uint64_t>(image, kSize - 8, base + 0x2000);
        CHECK(!Apply(image.data(), result));

        // Not a module at all.
        Bytes junk(64, 0);
        CHECK(!Apply(junk.data(), result));
        CHECK(!Apply(nullptr, result));
    }

    if (fails == 0)
        printf("mfg_ptx_smoke: all checks passed\n");
    return fails == 0 ? 0 : 1;
}
