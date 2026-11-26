#pragma once
// The FP8 pass-in-one weight sets (fp8_set<n>.txt and .bin): read by the D3D12 and Vulkan launch hooks alike.
// The manifest format is described in NrFp8.h.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace NrFp8Sets
{
inline constexpr unsigned kMaxSets = 8;          // fp8_set1 .. fp8_set7
inline constexpr uint16_t kBlendWeight = 0x39eb; // float16 0.7397: the output block's history blend weight our models expect
inline constexpr size_t kMaxParams = 512;

struct Entry
{
    bool present = false;
    std::string name;
    std::vector<std::pair<uint32_t, uint32_t>> layouts; // (parameter block size, offset of the record pointer)
    uint64_t offset = 0, bytes = 0;
    uint32_t blend = 0; // offset of the history blend weight pointer (0: none)
};

struct Set
{
    std::string failure;
    bool present = false;
    unsigned weighted = 0;
    std::vector<Entry> launches;     // one per launch of an evaluation
    std::filesystem::path bin;       // the records, read when an evaluation first uses the set
    uint64_t binBytes = 0;
};

inline std::vector<unsigned char> ReadFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("missing FP8 asset " + path.string());
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (bytes.empty() || !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
        throw std::runtime_error("cannot read FP8 asset " + path.string());
    return bytes;
}

inline Set ReadSet(const std::filesystem::path& folder, unsigned n)
{
    namespace fs = std::filesystem;
    const std::string stem = "fp8_set" + std::to_string(n);
    Set set;
    set.bin = folder / (stem + ".bin");
    std::error_code error;
    set.binBytes = fs::file_size(set.bin, error);
    if (error || !set.binBytes)
        throw std::runtime_error("missing FP8 asset " + set.bin.string());
    std::ifstream file(folder / (stem + ".txt"));
    if (!file)
        throw std::runtime_error("missing FP8 asset " + (folder / (stem + ".txt")).string());
    unsigned evaluation = 0;
    std::string line;
    unsigned number = 0;
    while (std::getline(file, line))
    {
        ++number;
        std::istringstream words(line);
        std::string key;
        if (!(words >> key) || key[0] == '#')
            continue;
        const auto bad = [&](const char* why)
        { return std::runtime_error(stem + ".txt line " + std::to_string(number) + ": " + why); };
        if (key == "evaluation")
        {
            if (!(words >> evaluation) || evaluation == 0 || evaluation > 4096)
                throw bad("a launch count from 1 to 4096");
            set.launches.assign(evaluation, Entry {});
            continue;
        }
        if (key != "launch")
            throw bad("unknown line");
        unsigned launch = 0;
        if (!(words >> launch) || launch == 0 || launch >= set.launches.size())
            throw bad("a launch number after `evaluation`, above 0");
        Entry e;
        e.present = true;
        bool haveOffset = false, haveBytes = false;
        for (std::string word; words >> word;)
        {
            if (word == "name")
            {
                if (!(words >> e.name))
                    throw bad("name");
            }
            else if (word == "params")
            {
                uint32_t size = 0;
                if (!(words >> size) || size < 8 || size > kMaxParams)
                    throw bad("params");
                e.layouts.push_back({ size, 0 });
            }
            else if (word == "record")
            {
                uint32_t at = 0;
                if (!(words >> at) || e.layouts.empty() || e.layouts.back().second || at + 8 > e.layouts.back().first)
                    throw bad("record after params, inside them");
                e.layouts.back().second = at;
            }
            else if (word == "offset")
            {
                if (!(words >> e.offset))
                    throw bad("offset");
                haveOffset = true;
            }
            else if (word == "bytes")
            {
                if (!(words >> e.bytes))
                    throw bad("bytes");
                haveBytes = true;
            }
            else if (word == "blend")
            {
                if (!(words >> e.blend) || e.blend < 8)
                    throw bad("blend");
            }
            else
                throw bad("unknown word");
        }
        if (e.name.empty() || e.layouts.empty() || !haveOffset || !haveBytes || e.bytes == 0)
            throw bad("name, params, record, offset and bytes");
        for (const auto& [size, at] : e.layouts)
            if (!at || (e.blend && e.blend + 8 > size))
                throw bad("a record (and blend) offset for every params");
        if (e.offset % 512 || e.offset > set.binBytes || e.bytes > set.binBytes - e.offset)
            throw bad("a 512-aligned record inside the .bin");
        if (set.launches[launch].present)
            throw bad("a launch given twice");
        set.launches[launch] = std::move(e);
        ++set.weighted;
    }
    if (!evaluation || !set.weighted)
        throw std::runtime_error(stem + ".txt has no launches");
    set.present = true;
    return set;
}

// Where a launch's record pointer sits, or 0 when the kernel or the parameter block is not the manifest's.
// The output block has a second form, which the runtime launches when the pass's intensity is not 1 in the Standard
// style: the same parameters (weight record, history blend weight) followed by 128 bytes more.
inline uint32_t RecordOffset(const Entry& e, const std::string& name, size_t paramSize, std::string& why)
{
    const bool blendForm = e.blend && name == e.name + "_simple_blend_full_rect";
    if (name != e.name && !blendForm)
    {
        why = "is " + name + ", Model A's is " + e.name;
        return 0;
    }
    for (const auto& [size, record] : e.layouts)
        if (paramSize == size + (blendForm ? 128u : 0u))
            return record;
    why = "(" + name + ") has a " + std::to_string(paramSize) + "-byte parameter block";
    return 0;
}
} // namespace NrFp8Sets
