#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace NrHostPlan {
namespace fs = std::filesystem;
constexpr uint16_t kBlendWeight = 0x39eb;         // float16 0.7397, the output block's history blend weight
constexpr uint32_t kMaxShared = 101376;
constexpr int kFirst = -1, kLast = 156, kSlots = kLast - kFirst + 1;
[[noreturn]] inline void Fail(const std::string& what) {
    throw std::runtime_error("the own NR pass host: " + what + ". Nothing was shown.");
}
inline std::vector<unsigned char> ReadAll(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) Fail("missing file " + path.string());
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (bytes.empty() || !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()))) Fail("cannot read " + path.string());
    return bytes;
}
// ---- geometry (host-spec/geom.py) -------------------------------------------------------------------------------------
inline int64_t Up(int64_t n, int64_t a) { return (n + a - 1) / a * a; }
inline int64_t Cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
struct Geometry {
    int64_t W = 0, H = 0, We = 0, He = 0, tokens = 0, T = 0, Gp = 0, count = 0;
    std::array<int64_t, 7> R{}, C{};
    Geometry(unsigned w, unsigned h) : W(w), H(h) {
        const auto align = [](int64_t n) { return Up((n + 1) / 2, 4) % 8 == 4 ? 128 : 64; };
        const int64_t ax = align(W), ay = align(H);
        We = std::max<int64_t>(320, Up(W, ax));
        He = std::max<int64_t>(320, Up(H, ay));
        if (He % (ay * 4) == 0 && We % (ax * 4) == 0) We += ax;
        R[0] = He; C[0] = We;
        for (int k = 0; k < 6; ++k) { R[k + 1] = Up(Cdiv(R[k], 2), 4); C[k + 1] = Up(Cdiv(C[k], 2), 4); }
        tokens = R[6] * C[6];
        T = Cdiv(tokens, 128);
        Gp = Up(tokens, 32) / 16;
    }
    int64_t Symbol(const std::string& s) const {
        if (s == "W") return W;
        if (s == "H") return H;
        if (s == "We") return We;
        if (s == "He") return He;
        if (s == "tokens") return tokens;
        if (s == "T") return T;
        if (s == "Gp") return Gp;
        if (s == "count") return count;
        if (s.size() == 2 && (s[0] == 'R' || s[0] == 'C') && s[1] >= '0' && s[1] <= '6') return (s[0] == 'R' ? R : C)[s[1] - '0'];
        Fail("unknown plan symbol " + s);
    }
};
inline bool Has(const std::string& n, const char* part) { return n.find(part) != std::string::npos; }
inline bool EndsWith(const std::string& n, const std::string& tail) { return n.size() >= tail.size() && n.compare(n.size() - tail.size(), tail.size(), tail) == 0; }
inline std::string Family(const std::string& n) {
    if (n == "cc_cb_clear") return "clear";
    if (n == "cg2r_copy_kernel") return "copy";
    if (n == "cg2r_post_process_kernel") return "grade";
    if (Has(n, "pre_block")) return "pre";
    if (Has(n, "post_block")) return "post";
    if (Has(n, "fused_swin_1h_32")) return "swin32";
    if (Has(n, "fused_swin")) return "swin";
    if (Has(n, "repack_2d_to_1d")) return "repack21";
    if (Has(n, "repack_1d_to_2d")) return "repack12";
    if (Has(n, "ffwd_proj")) return "ffwd_proj";
    if (Has(n, "split_swin_16h_ffwd")) return "ffwd";
    if (Has(n, "split_swin_16h_qkv")) return "sqkv";
    if (Has(n, "proj_pool")) return "proj_pool";
    if (Has(n, "final_head")) return "final_head";
    if (Has(n, "split_swin_16h_proj")) return "sproj";
    if (Has(n, "vit_1d_ffn_expand")) return "expand";
    if (Has(n, "vit_1d_ffn_contract")) return "contract";
    if (Has(n, "vit_1d_qkv")) return "vqkv";
    if (Has(n, "vit_1d_attention")) return "attention";
    if (Has(n, "vit_1d_projection")) return "vproj";
    if (Has(n, "dec_input_upsample")) return "dec";
    Fail("unknown kernel " + n);
}
inline std::string Variant(const std::string& n) {
    for (const char* v : {"inpview_tilesync", "ds_wait", "upsample_tilesync", "outview_wait", "chained", "publish", "tilesync", "wait"})
        if (EndsWith(n, std::string("_") + v + "_fp8") || EndsWith(n, std::string("_") + v)) return v;
    return "plain";
}
inline int SwinLevel(const std::string& n) {
    if (Has(n, "_32_")) return 1;
    if (Has(n, "_64_")) return 2;
    if (Has(n, "_128_")) return 3;
    if (Has(n, "_256_")) return 4;
    Fail("no Swin width in " + n);
}
struct Dims { uint32_t x = 1, y = 1, z = 1; };
inline std::pair<Dims, Dims> Grid(const std::string& name, const Geometry& g, int sx, int sy) {
    const std::string f = Family(name);
    const auto D = [](int64_t x, int64_t y, int64_t z) { return Dims{uint32_t(x), uint32_t(y), uint32_t(z)}; };
    const auto& R = g.R;
    const auto& C = g.C;
    if (f == "clear") return {D(Cdiv(g.count, 256), 1, 1), D(256, 1, 1)};
    if (f == "copy" || f == "grade") return {D(Cdiv(g.W, 16), Cdiv(g.H, 16), 1), D(16, 16, 1)};
    if (f == "pre") return {D(C[0] / 8, R[0] / 8, 1), D(32, 1, 1)};
    if (f == "post") return {D(Cdiv(C[0] + 4, 8), Cdiv(R[0] + 4, 8), 1), D(32, 1, 1)};
    if (f == "swin32" || f == "swin") {
        const int l = SwinLevel(name);
        const int heads = l == 1 ? 1 : l == 2 ? 2 : l == 3 ? 4 : 8;
        return {D(Cdiv(C[l] - std::min(sx, 0), 8), Cdiv(R[l] - std::min(sy, 0), 8), 1), D(32, heads, 1)};
    }
    const int64_t gx5 = Cdiv(C[5], 8), gy5 = Cdiv(R[5], 8);
    if (f == "ffwd") return {D(gx5, gy5, 2), D(32, Has(name, "inpview") ? 4 : 8, 1)};
    if (f == "ffwd_proj") return {D(2 * gx5, gy5, 1), D(32, 4, 1)};
    if (f == "sqkv") return {D(Cdiv(C[5] - std::min(sx, 0), 8), Cdiv(R[5] - std::min(sy, 0), 8), 4), D(32, 4, 1)};
    if (f == "sproj") return {D(2 * gx5, gy5, 1), D(32, Has(name, "outview") ? 4 : 8, 1)};
    if (f == "proj_pool") return {D(2 * gx5, gy5, 1), D(32, 4, 1)};
    if (f == "final_head") return {D(4 * Cdiv(C[6], 8), Cdiv(R[6], 8), 1), D(32, 8, 1)};
    if (f == "repack21") return {D(Up(g.tokens, 32), 1, 1), D(256, 1, 1)};
    if (f == "repack12") return {D(g.tokens, 1, 1), D(256, 1, 1)};
    if (f == "expand") return {D(32 * g.T, 1, 1), D(32, 4, 1)};
    if (f == "contract") return {D(8 * g.T, 1, 4), D(32, 4, 1)};
    if (f == "vqkv") return {D(16 * g.T, 1, 2), D(32, 4, 1)};
    if (f == "attention") return {D(32, Cdiv(g.T, 2), 1), D(32, 4, 1)};
    if (f == "vproj") return {D(8 * g.T, 1, 4), D(32, 4, 1)};
    if (f == "dec") return {D(2 * Cdiv(C[6], 4), Cdiv(R[6], 4), 4), D(32, 2, 1)};
    Fail("no grid rule for " + name);
}
// [(array, int32 words)] a launch owns in the flag region, counter first.
inline std::vector<std::pair<std::string, int64_t>> OwnArrays(const std::string& name, Dims g, int64_t T) {
    const std::string f = Family(name), v = Variant(name);
    const int64_t gx = g.x, gy = g.y, gz = g.z;
    if (f == "swin32" || f == "swin") return (v == "ds_wait" || v == "outview_wait") ? decltype(OwnArrays(name, g, T)){} : decltype(OwnArrays(name, g, T)){{"flags", gx * gy}};
    if (f == "ffwd") return {{"flags", gx * gy * gz}};
    if (f == "ffwd_proj" || f == "proj_pool") return {{"flags", gx * gy}};
    if (f == "sproj") { if (Has(name, "outview")) return {}; return {{"flags", gx * gy}}; }
    if (f == "sqkv") return {{"flags", gx * gy * gz}};
    if (f == "expand") return {{"flags", gx}};
    if (f == "contract" || f == "vqkv") return {{"counter", gx}, {"flags", gx}};
    if (f == "attention") return {{"flags", 32 * T}};
    if (f == "vproj") { if (v == "wait") return {{"counter", gx}}; return {{"counter", gx}, {"flags", gx}}; }
    if (f == "dec") return {{"counter", gx * gy}, {"flags", gx * gy}};
    return {};
}

// ---- the plan ---------------------------------------------------------------------------------------------------------
enum class Code { Zero, U32, I, F, Rcp, TF, TRcp, Ptr, Flags, ClearBase, Record, Blend, Tex, Surf, Mv, Style, Seed };
struct Field {
    uint32_t off = 0, size = 0;
    Code code = Code::Zero;
    uint64_t value = 0;           // U32 constant / tensor index / flags launch
    std::string sym;              // symbol, role, style name, flags array
};
struct LaunchFields {
    int idx = 0;
    std::string name, layout, styles;
    uint32_t params = 0;
    int sx = 0, sy = 0;
    bool barrier = false;
    std::vector<Field> fields;
    std::vector<unsigned char> parameters;
};
struct PlanTensor { std::string name, kind; int a = 0, b = 0; int first = 0, last = 0; };
template<class PlanLaunch> struct ModelPlan {
    std::array<std::array<float, 3>, 14> grades{};
    std::vector<PlanTensor> tensors;
    std::vector<PlanLaunch> launches;
};
inline float BitsToFloat(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

template<class PlanLaunch> ModelPlan<PlanLaunch> ReadPlan(const fs::path& path) {
    std::ifstream file(path);
    if (!file) Fail("missing plan " + path.string());
    ModelPlan<PlanLaunch> p;
    std::map<std::string, size_t> tensorIndex;
    std::string line;
    unsigned number = 0;
    PlanLaunch* cur = nullptr;
    bool header = false;
    while (std::getline(file, line)) {
        ++number;
        std::istringstream w(line);
        std::string key;
        if (!(w >> key) || key[0] == '#') continue;
        const auto bad = [&](const std::string& why) { Fail(path.filename().string() + " line " + std::to_string(number) + ": " + why); };
        if (key == "plan") { std::string name; unsigned version = 0; w >> name >> version; if (name != "modelA-fp8" || version != 1) bad("not a modelA-fp8 version 1 plan"); header = true; continue; }
        if (!header) bad("no plan line");
        if (key == "grade") {
            unsigned k; std::string a, b, c;
            if (!(w >> k >> a >> b >> c) || k >= 14) bad("grade");
            p.grades[k] = {BitsToFloat(uint32_t(std::stoul(a, nullptr, 16))), BitsToFloat(uint32_t(std::stoul(b, nullptr, 16))), BitsToFloat(uint32_t(std::stoul(c, nullptr, 16)))};
        } else if (key == "tensor") {
            PlanTensor t;
            std::string word;
            if (!(w >> t.name >> t.kind)) bad("tensor");
            std::vector<std::string> rest;
            while (w >> word) rest.push_back(word);
            size_t i = 0;
            std::vector<int> nums;
            while (i < rest.size() && rest[i] != "first") nums.push_back(std::stoi(rest[i++]));
            if (i + 4 != rest.size() || rest[i + 2] != "last") bad("tensor ... first <idx> last <idx>");
            t.first = std::stoi(rest[i + 1]); t.last = std::stoi(rest[i + 3]);
            if (nums.size() > 0) t.a = nums[0];
            if (nums.size() > 1) t.b = nums[1];
            if (t.first < kFirst || t.last > kLast || t.last < t.first) bad("invalid tensor lifetime");
            if ((t.kind == "view" || t.kind == "tile") && (t.a < 0 || t.a > 6 || t.b <= 0))
                bad("invalid tensor level or width");
            if ((t.kind == "act" || t.kind == "scratch") && t.a <= 0) bad("invalid tensor extent");
            if (!tensorIndex.emplace(t.name, p.tensors.size()).second) bad("duplicate tensor");
            p.tensors.push_back(t);
        } else if (key == "launch") {
            PlanLaunch l;
            std::string k1, k2, k3, k4, k5;
            if (!(w >> l.idx >> l.name >> k1 >> l.params >> k2 >> l.layout >> k3 >> l.styles >> k4 >> l.sx >> l.sy >> k5) || k1 != "params" || k2 != "layout" ||
                k3 != "styles" || k4 != "shift" || k5 != "barrier") bad("launch");
            int barrier = 0;
            if (!(w >> barrier)) bad("barrier");
            l.barrier = barrier != 0;
            if (l.idx < kFirst || l.idx > kLast) bad("launch number");
            if (!l.params || l.params > 4096) bad("invalid CUDA parameter block size");
            p.launches.push_back(std::move(l));
            cur = &p.launches.back();
        } else if (key == "f") {
            if (!cur) bad("field outside a launch");
            Field f;
            std::string code;
            if (!(w >> f.off >> f.size >> code)) bad("f <off> <size> <code>");
            if (code == "z" || code == "zu") f.code = Code::Zero;
            else if (code == "u32") { std::string h; w >> h; f.code = Code::U32; f.value = std::stoull(h, nullptr, 16); }
            else if (code == "i" || code == "f" || code == "rcp" || code == "tf" || code == "trcp") {
                w >> f.sym;
                f.code = code == "i" ? Code::I : code == "f" ? Code::F : code == "rcp" ? Code::Rcp : code == "tf" ? Code::TF : Code::TRcp;
            } else if (code == "ptr") {
                std::string t; w >> t;
                const auto found = tensorIndex.find(t);
                if (found == tensorIndex.end()) bad("unknown tensor " + t);
                f.code = Code::Ptr; f.value = found->second;
            } else if (code == "flags") { int64_t launch; w >> launch >> f.sym; f.code = Code::Flags; f.value = uint64_t(launch); }
            else if (code == "clearbase") f.code = Code::ClearBase;
            else if (code == "record") f.code = Code::Record;
            else if (code == "blend") f.code = Code::Blend;
            else if (code == "tex") { w >> f.sym; f.code = Code::Tex; }
            else if (code == "surf") { w >> f.sym; f.code = Code::Surf; }
            else if (code == "mv") { w >> f.sym; f.code = Code::Mv; }
            else if (code == "style") { w >> f.sym; f.code = Code::Style; }
            else if (code == "seed") f.code = Code::Seed;
            else bad("unknown code " + code);
            if (!w || !f.size || f.off > cur->params || f.size > cur->params - f.off)
                bad("invalid parameter field");
            const bool pointer = f.code == Code::Ptr || f.code == Code::Flags || f.code == Code::ClearBase ||
                                 f.code == Code::Record || f.code == Code::Blend || f.code == Code::Tex || f.code == Code::Surf;
            if (f.code != Code::Zero && f.size != (pointer ? 8u : 4u)) bad("field has the wrong size");
            cur->fields.push_back(f);
        } else if (key == "end") {
            cur = nullptr;
        } else {
            bad("unknown line " + key);
        }
    }
    if (p.launches.empty() || p.tensors.empty()) Fail("empty plan " + path.string());
    return p;
}

// ---- kernel-set manifests (NrAmpere format; launch numbers -1..156) ----------------------------------------------------
struct Dim { char mode = '='; uint32_t value = 0; };
struct KernelChoice {
    bool present = false;
    fs::path folder;
    std::string code, file, entry, name, where;
    uint32_t params = 0, shared = 0;
    std::array<Dim, 3> grid{}, block{};
    // 'prepare <entry> <factor>': run <entry> (same module) once per host on the launch's record; the kernel then gets
    // the prepared copy (record bytes x factor) at its record field instead of the fp8 record.
    std::string prepare;
    double prepareFactor = 0;
};
inline Dim ParseDim(const std::string& word) {
    Dim d;
    if (word == "=") return d;
    if (word[0] == '*' || word[0] == '/') { d.mode = word[0]; d.value = uint32_t(std::stoul(word.substr(1))); }
    else { d.mode = 'n'; d.value = uint32_t(std::stoul(word)); }
    if (!d.value) throw std::runtime_error("bad dimension " + word);
    return d;
}
inline std::array<Dim, 3> ParseDims(std::istringstream& words) {
    std::string first;
    if (!(words >> first)) throw std::runtime_error("three dimensions");
    std::vector<std::string> parts;
    if (first.find('x') != std::string::npos) {
        std::string part;
        std::istringstream split(first);
        while (std::getline(split, part, 'x')) parts.push_back(part);
    } else {
        parts.push_back(first);
        for (int i = 0; i < 2; ++i) { std::string x; if (!(words >> x)) throw std::runtime_error("three dimensions"); parts.push_back(x); }
    }
    if (parts.size() != 3) throw std::runtime_error("three dimensions");
    return {ParseDim(parts[0]), ParseDim(parts[1]), ParseDim(parts[2])};
}
inline uint32_t Apply(const Dim& d, uint32_t plan) { return d.mode == 'n' ? d.value : d.mode == '*' ? plan * d.value : d.mode == '/' ? (plan + d.value - 1) / d.value : plan; }
inline std::vector<int> ParseLaunchNumbers(const std::string& text) {
    std::vector<int> out;
    std::string part;
    std::istringstream split(text);
    while (std::getline(split, part, ',')) {
        // a range a-b; a leading '-' is a sign (launch -1)
        const size_t dash = part.find('-', 1);
        const int a = std::stoi(part.substr(0, dash));
        const int b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
        if (a < kFirst || b > kLast || b < a) throw std::runtime_error("launch numbers -1..156: " + part);
        for (int n = a; n <= b; ++n) out.push_back(n);
    }
    return out;
}
inline std::string BaseName(std::string file) {
    for (const char* ending : {".sm120.cubin", ".sm86.cubin", ".sm89.cubin", ".sm86.ptx.fatbin"}) {
        const size_t n = strlen(ending);
        if (file.size() > n && file.compare(file.size() - n, n, ending) == 0) return file.substr(0, file.size() - n);
    }
    return file;
}
// All manifests of all folders; per launch number one line per parameter size.
using Choices = std::map<int, std::vector<KernelChoice>>;
inline void ReadFolder(const fs::path& folder, const std::string& code, Choices& choices) {
    std::error_code error;
    if (!fs::is_directory(folder, error)) Fail("kernel folder " + folder.string() + " does not exist");
    unsigned manifests = 0;
    for (const auto& entry : fs::directory_iterator(folder)) {
        const std::string fname = entry.path().filename().string();
        if (!EndsWith(fname, "_manifest.txt")) continue;
        ++manifests;
        std::ifstream file(entry.path());
        std::map<std::string, KernelChoice> kernels;
        std::string line;
        unsigned number = 0;
        while (std::getline(file, line)) {
            ++number;
            const std::string where = fname + " line " + std::to_string(number);
            if (const size_t hash = line.find('#'); hash != std::string::npos) line.resize(hash);
            std::istringstream words(line);
            std::string key;
            if (!(words >> key)) continue;
            try {
                if (key == "family") continue;
                if (key == "kernel") {
                    std::string id;
                    words >> id;
                    KernelChoice k;
                    for (std::string word; words >> word;) {
                        if (word == "file") { words >> k.file; k.file = BaseName(k.file); }
                        else if (word == "entry") words >> k.entry;
                        else if (word == "prepare") { words >> k.prepare >> k.prepareFactor; if (!(k.prepareFactor > 0 && k.prepareFactor <= 16)) throw std::runtime_error("prepare <entry> <factor 0..16>"); }
                        else if (word == "shared") { std::string s; words >> s; k.shared = s == "=" ? 0 : uint32_t(std::stoul(s)); }
                        else if (word == "grid") k.grid = ParseDims(words);
                        else if (word == "block") k.block = ParseDims(words);
                        else throw std::runtime_error("unknown word " + word);
                    }
                    kernels[id] = k;
                    continue;
                }
                if (key != "launch") throw std::runtime_error("unknown line " + key);
                std::string list;
                words >> list;
                const auto numbers = ParseLaunchNumbers(list);
                KernelChoice c;
                std::string kernelId;
                bool haveGrid = false, haveBlock = false, haveShared = false;
                for (std::string word; words >> word;) {
                    if (word == "kernel") words >> kernelId;
                    else if (word == "file") { words >> c.file; c.file = BaseName(c.file); }
                    else if (word == "entry") words >> c.entry;
                    else if (word == "prepare") { words >> c.prepare >> c.prepareFactor; if (!(c.prepareFactor > 0 && c.prepareFactor <= 16)) throw std::runtime_error("prepare <entry> <factor 0..16>"); }
                    else if (word == "name") words >> c.name;
                    else if (word == "params") words >> c.params;
                    else if (word == "shared") { std::string s; words >> s; c.shared = s == "=" ? 0 : uint32_t(std::stoul(s)); haveShared = true; }
                    else if (word == "grid") { c.grid = ParseDims(words); haveGrid = true; }
                    else if (word == "block") { c.block = ParseDims(words); haveBlock = true; }
                    else if (word == "nvidia") throw std::runtime_error("'nvidia' lines need the runtime; the own host has none");
                    else throw std::runtime_error("unknown word " + word);
                }
                if (!kernelId.empty()) {
                    const auto found = kernels.find(kernelId);
                    if (found == kernels.end()) throw std::runtime_error("kernel " + kernelId + " is not declared above");
                    c.file = found->second.file; c.entry = found->second.entry;
                    if (c.prepare.empty()) { c.prepare = found->second.prepare; c.prepareFactor = found->second.prepareFactor; }
                    if (!haveShared) c.shared = found->second.shared;
                    if (!haveGrid) c.grid = found->second.grid;
                    if (!haveBlock) c.block = found->second.block;
                }
                if (c.file.empty() || c.entry.empty()) throw std::runtime_error("kernel <id> or file <base> entry <symbol>");
                if (c.shared > kMaxShared) throw std::runtime_error("shared memory above 101376 bytes");
                c.present = true;
                c.folder = folder;
                c.code = code;
                c.where = folder.filename().string() + "/" + where;
                for (const int n : numbers) {
                    auto& list2 = choices[n];
                    for (const auto& other : list2)
                        if (other.params == c.params) throw std::runtime_error("launch " + std::to_string(n) + " is already given by " + other.where);
                    list2.push_back(c);
                }
            } catch (const std::exception& e) {
                Fail(where + " (" + folder.string() + "): " + e.what());
            }
        }
    }
    if (!manifests) Fail("no *_manifest.txt in kernel folder " + folder.string());
}
inline std::string ModuleFile(const KernelChoice& c) {
    if (c.code == "sm86") return c.file + ".sm86.cubin";
    if (c.code == "sm86ptx") return c.file + ".sm86.ptx.fatbin";
    if (c.code == "sm89") return c.file + ".sm89.cubin";
    return c.file + ".sm120.cubin";                 // sm120, nvidia
}

inline uint64_t TensorBytes(const PlanTensor& t, const Geometry& g) {
    const auto& R = g.R;
    const auto& C = g.C;
    if (t.kind == "tile") return uint64_t((R[t.a] / 4) * (C[t.a] / 4) * 16 * t.b);
    if (t.kind == "view") return uint64_t(R[t.a] * C[t.a] * t.b);
    if (t.kind == "act") return uint64_t(g.Gp * 16 * t.a);
    if (t.kind == "scratch") return uint64_t(t.a * g.Gp * 32768);
    if (t.kind == "scratch_dec") return uint64_t((R[6] / 4) * (C[6] / 4) * 16384);
    Fail("unknown tensor layout " + t.kind);
}
inline void BindParameters(LaunchFields& launch, const Geometry& geo, const auto& layout, uint64_t arena,
                           uint64_t record, uint64_t blend) {
    launch.parameters.resize(launch.params);
    std::erase_if(launch.fields, [&](const Field& f) {
        const auto write = [&](auto value) { memcpy(launch.parameters.data() + f.off, &value, sizeof(value)); };
        switch (f.code) {
        case Code::Zero: break;
        case Code::U32: write(uint32_t(f.value)); break;
        case Code::I: write(int32_t(geo.Symbol(f.sym))); break;
        case Code::F: write(float(geo.Symbol(f.sym))); break;
        case Code::Rcp: write(float(1.0 / double(geo.Symbol(f.sym)))); break;
        case Code::Ptr: write(arena + layout.tensorBase + layout.tensorOffset.at(size_t(f.value))); break;
        case Code::Flags: {
            const auto flags = layout.flagOffset.find({int(int64_t(f.value)), f.sym});
            if (flags == layout.flagOffset.end()) Fail("launch " + std::to_string(launch.idx) + ": missing " + f.sym + " flags");
            write(arena + flags->second);
            break;
        }
        case Code::ClearBase: write(arena); break;
        case Code::Record: write(record); break;
        case Code::Blend: write(blend); break;
        default: return false;
        }
        return true;
    });
}
struct FrameHandles {
    uint64_t colourTex, motionTex, outputTex, outputSurf;
    uint64_t historyTex, historySurf, netoutTex, netoutSurf, inputcopyTex, inputcopySurf;
};
template<class Settings, class Grades>
inline void FrameParameters(LaunchFields& l, const Geometry& geo, const Grades& grades, const Settings& s,
                            const FrameHandles& handles, bool resetFrame, bool graded, uint32_t sinceReset) {
    const auto tex = [&](const std::string& role) -> uint64_t {
        if (role == "colour") return handles.colourTex;
        if (role == "history") return resetFrame ? 0 : handles.historyTex;
        if (role == "motion") return handles.motionTex;
        if (role == "netout") return handles.netoutTex;
        if (role == "inputcopy") return handles.inputcopyTex;
        if (role == "copysrc") return graded ? handles.netoutTex : handles.outputTex;
        Fail("unknown texture role " + role);
    };
    const auto surf = [&](const std::string& role) -> uint64_t {
        if (role == "output") return handles.outputSurf;
        if (role == "history") return handles.historySurf;
        if (role == "netout") return handles.netoutSurf;
        if (role == "inputcopy") return handles.inputcopySurf;
        if (role == "out154") return graded ? handles.netoutSurf : handles.outputSurf;
        Fail("unknown surface role " + role);
    };
    const auto style = [&](const std::string& name) -> uint32_t {
        const auto bits = [](float f) { uint32_t b; memcpy(&b, &f, 4); return b; };
        if (name == "tone") return bits(s.tone);
        if (name == "structure") return bits(s.structure);
        if (name == "stylecode") return bits(float(s.style) / 128.f);
        if (name == "skin") return bits(s.autoMask ? (s.skin >= 0 ? s.skin : s.structure) : -1.f);
        if (name == "other") return bits(s.autoMask ? s.structure : -1.f);
        // NVIDIA's runtime sends 1 here whatever Auto skin mask is (traced with the mask on and off, Skin structure
        // automatic and 0.5): the switch reaches the model only through skin and other (-1 when off). A 0 here made
        // the kernel give the model 0 for both, and the edit lost about a third of its strength.
        if (name == "automask") return 1u;
        if (name == "depthinv") return 0u;
        if (name == "intensity") return bits(s.intensity);
        if (name.rfind("grade", 0) == 0) {
            const auto& gr = grades.at(std::stoi(name.substr(5)));
            const float neutral = gr[0], preset = gr[std::min(s.style, 2u)];
            const float t = std::clamp(s.tone, 0.f, 1.f);
            return bits(neutral + t * (preset - neutral));
        }
        Fail("unknown style field " + name);
    };
    auto& block = l.parameters;
    const auto put32 = [&](uint32_t off, uint32_t v) { memcpy(block.data() + off, &v, 4); };
    const auto putF = [&](uint32_t off, float v) { memcpy(block.data() + off, &v, 4); };
    const auto put64 = [&](uint32_t off, uint64_t v) { memcpy(block.data() + off, &v, 8); };
    for (const Field& f : l.fields) {
        switch (f.code) {
        case Code::TF: putF(f.off, resetFrame ? 0.f : float(geo.Symbol(f.sym))); break;
        case Code::TRcp: putF(f.off, resetFrame ? 0.f : float(1.0 / double(geo.Symbol(f.sym)))); break;
        case Code::Tex: put64(f.off, tex(f.sym)); break;
        case Code::Surf: put64(f.off, surf(f.sym)); break;
        case Code::Mv: putF(f.off, f.sym == "x" ? float(double(s.mvScaleX) / double(geo.W)) : float(double(s.mvScaleY) / double(geo.H))); break;
        case Code::Style: put32(f.off, style(f.sym)); break;
        case Code::Seed: put32(f.off, resetFrame ? 0u : sinceReset); break;
        default: Fail("unexpected dynamic launch field");
        }
    }
}
struct RecordLayout {
    uint64_t blendAt = 0;
    std::map<int, std::pair<uint64_t, std::string>> offsets;
    std::map<int, uint64_t> bytes;
    void Read(const fs::path& txt, const fs::path& bin, const std::vector<unsigned char>& blob) {
        {   // the manifest: launch <n> name <k> params .. record .. offset <o> bytes <b> [blend ..]
            std::ifstream file(txt);
            if (!file) Fail("missing records " + txt.string());
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream w(line);
                std::string key;
                if (!(w >> key) || key != "launch") continue;
                int n = 0;
                if (!(w >> n) || n < 0 || n > 65535) Fail("invalid launch index in " + txt.string());
                std::string name;
                uint64_t offset = 0, bytes = 0;
                bool haveOffset = false;
                for (std::string word; w >> word;) {
                    if (word == "name") w >> name;
                    else if (word == "offset") { w >> offset; haveOffset = true; }
                    else if (word == "bytes") w >> bytes;
                    else { std::string skip; w >> skip; }
                }
                if (!haveOffset || name.empty()) Fail(txt.filename().string() + ": launch " + std::to_string(n) + " has no name or offset");
                if (!bytes || bytes > UINT32_MAX || offsets.contains(n)) Fail("invalid or duplicate record in " + txt.string());
                offsets[n] = {offset, name};
                this->bytes[n] = bytes;
            }
        }

        if (offsets.empty()) Fail("empty records manifest " + txt.string());
        for (const auto& [index, entry] : offsets)
            if (entry.first > blob.size() || bytes.at(index) > blob.size() - entry.first)
                Fail("record " + std::to_string(index) + " extends beyond " + bin.string());
        blendAt = (blob.size() + 511) / 512 * 512;
    }
};
struct ArenaLayout {
    std::map<std::pair<int, std::string>, uint64_t> flagOffset;
    uint64_t flagBytes = 0, tensorBase = 0, arenaBytes = 0;
    std::vector<uint64_t> tensorOffset;
    template<class Launch>
    void Build(Geometry& geo, const ModelPlan<Launch>& plan, const std::vector<Launch*>& order) {
        uint64_t off = 0;
        for (const auto* l : order) {
            if (l->idx < 1 || l->idx > 154) continue;
            const auto [grid, block] = Grid(l->name, geo, l->sx, l->sy);
            for (const auto& [array, words] : OwnArrays(l->name, grid, geo.T)) {
                flagOffset[{l->idx, array}] = off;
                off += uint64_t(Up(4 * words, 512));
            }
        }
        flagBytes = off;
        geo.count = int64_t(off / 4);
        tensorBase = uint64_t(Up(int64_t(off), 4096));
        {   // first-fit, memory reused only after a barrier launch (host-spec/planner.py rule 'barrier')
            std::set<int> barrierLaunches;
            for (const auto* l : order) if (l->barrier) barrierLaunches.insert(l->idx);
            struct Placed { uint64_t off, bytes; int last; };
            std::vector<Placed> placed;
            std::vector<size_t> sorted(plan.tensors.size());
            for (size_t k = 0; k < sorted.size(); ++k) sorted[k] = k;
            std::stable_sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) { return plan.tensors[a].first < plan.tensors[b].first; });
            tensorOffset.assign(plan.tensors.size(), 0);
            uint64_t end = 0;
            for (const size_t k : sorted) {
                const PlanTensor& t = plan.tensors[k];
                const uint64_t bytes = TensorBytes(t, geo);
                std::vector<std::pair<uint64_t, uint64_t>> busy;
                for (const auto& q : placed) {
                    bool free = false;
                    if (q.last < t.first) for (int j = q.last + 1; j <= t.first && !free; ++j) free = barrierLaunches.count(j) != 0;
                    if (!free) busy.push_back({q.off, q.off + q.bytes});
                }
                std::sort(busy.begin(), busy.end());
                uint64_t at = 0;
                for (const auto& [s, e] : busy) {
                    if (at + bytes <= s) break;
                    at = std::max(at, (e + 511) / 512 * 512);
                }
                tensorOffset[k] = at;
                placed.push_back({at, bytes, t.last});
                end = std::max(end, at + bytes);
            }
            arenaBytes = tensorBase + end + 65536;
        }
    }
};
} // namespace NrHostPlan
