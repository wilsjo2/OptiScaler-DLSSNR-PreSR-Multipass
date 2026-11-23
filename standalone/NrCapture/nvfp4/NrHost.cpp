// Own NR pass host; see NrHost.h. Geometry rules: V2/host-spec/geom.py (checked against 19 NVIDIA traces); field
// codes: V2/host_plan.py. Records are uploaded the way NrAmpere.cpp / NrFp8.cpp do it.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <nvapi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "NrHost.h"
#include "NrFourBitPlan.h"
#include "NrWeightUpload.h"

namespace NrHost {
namespace {
using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

constexpr unsigned kCreateModule = 0xad1a677d, kCreateFunction = 0xe2436e22, kLaunch = 0x24973538,
                   kMergedTexture = 0x329fe6e0, kIndependentDescriptor = 0x0ddac234;
constexpr uint16_t kBlendWeight = 0x39eb;         // float16 0.7397, the output block's history blend weight
constexpr uint32_t kMaxShared = 101376;
constexpr int kFirst = -1, kLast = 156, kSlots = kLast - kFirst + 1;
constexpr unsigned kRing = 64;                    // per-evaluation descriptor slots (evaluations in flight at most)
constexpr unsigned kPerEval = 4;                  // colour SRV, motion SRV, output SRV, output UAV
constexpr unsigned kStatic = 6;                   // history SRV/UAV, netout SRV/UAV, inputcopy SRV/UAV

void (*g_log)(const char*) = nullptr;
std::atomic<void*> g_driverQuery = nullptr;
void Log(const std::string& line) {
    const std::string text = "NR host: " + line;
    if (g_log) g_log(text.c_str()); else fprintf(stderr, "%s\n", text.c_str());
}
[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error("the own NR pass host: " + what + ". Nothing was shown.");
}
void Check(HRESULT r, const char* what) {
    if (FAILED(r)) Fail(std::string(what) + " failed (D3D12 status 0x" + [&] { char b[16]; snprintf(b, sizeof b, "%08x", unsigned(r)); return std::string(b); }() + ")");
}
void NvCheck(NvAPI_Status r, const std::string& what) {
    if (r != NVAPI_OK) Fail(what + " failed (NVAPI status " + std::to_string(int(r)) + ")");
}
std::vector<unsigned char> ReadAll(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) Fail("missing file " + path.string());
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (bytes.empty() || !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()))) Fail("cannot read " + path.string());
    return bytes;
}
std::wstring Env(const wchar_t* key) {
    const wchar_t* v = _wgetenv(key);
    return v ? std::wstring(v) : std::wstring();
}
std::string Narrow(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) s.push_back(c < 128 ? char(c) : '?');
    return s;
}

// ---- NvAPI ----------------------------------------------------------------------------------------------------------
using Query = void* (__cdecl*)(unsigned);
struct Api {
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_GetCudaMergedTextureSamplerObject) mergedTexture = nullptr;
    decltype(&NvAPI_D3D12_GetCudaIndependentDescriptorObject) independent = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
    Query query = nullptr;
};
Api& GetApi() {
    static Api api;
    if (api.query) return api;
    // The driver's own entry (not a hooked one: the app's NvAPI layers must not see the host's launches).
    auto query = reinterpret_cast<Query>(g_driverQuery.load());
    std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(&FreeLibrary)> module(nullptr, &FreeLibrary);
    if (!query) {
        module.reset(LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
        if (!module) Fail("the system's nvapi64.dll is not available (no NVIDIA driver?)");
        query = reinterpret_cast<Query>(GetProcAddress(module.get(), "nvapi_QueryInterface"));
        if (!query) Fail("nvapi_QueryInterface is missing from nvapi64.dll");
    }
    const auto initialize = reinterpret_cast<decltype(&NvAPI_Initialize)>(query(0x0150e828));
    Api a;
    a.createModule = reinterpret_cast<decltype(a.createModule)>(query(kCreateModule));
    a.createFunction = reinterpret_cast<decltype(a.createFunction)>(query(kCreateFunction));
    a.launch = reinterpret_cast<decltype(a.launch)>(query(kLaunch));
    a.mergedTexture = reinterpret_cast<decltype(a.mergedTexture)>(query(kMergedTexture));
    a.independent = reinterpret_cast<decltype(a.independent)>(query(kIndependentDescriptor));
    a.destroyModule = reinterpret_cast<decltype(a.destroyModule)>(query(0x41c65285));
    a.destroyFunction = reinterpret_cast<decltype(a.destroyFunction)>(query(0xdf295ea6));
    if (!a.createModule || !a.createFunction || !a.launch || !a.mergedTexture || !a.independent ||
        !a.destroyModule || !a.destroyFunction)
        Fail("the NVIDIA driver lacks the CUDA-on-D3D12 entry points the host needs (update the driver)");
    if (!initialize || initialize() != NVAPI_OK) Fail("NvAPI_Initialize failed");
    module.release(); // The successful dispatch table is used for the rest of the process.
    a.query = query;
    api = a;
    return api;
}

// GPU architecture of the D3D12 adapter (NV_GPU_ARCH_INFO), as NrAmpere::CheckArchitecture.
std::pair<unsigned, unsigned> Architecture(ID3D12Device* device) {
    Api& api = GetApi();
    const auto enumerate = reinterpret_cast<decltype(&NvAPI_EnumPhysicalGPUs)>(api.query(0xe5ac921f));
    const auto logical = reinterpret_cast<decltype(&NvAPI_GetLogicalGPUFromPhysicalGPU)>(api.query(0xadd604d1));
    const auto info = reinterpret_cast<decltype(&NvAPI_GPU_GetLogicalGpuInfo)>(api.query(0x842b066e));
    const auto arch = reinterpret_cast<decltype(&NvAPI_GPU_GetArchInfo)>(api.query(0xd8265d24));
    if (!enumerate || !logical || !info || !arch) Fail("the NvAPI architecture query is unavailable");
    const LUID selected = device->GetAdapterLuid();
    NvPhysicalGpuHandle handles[NVAPI_MAX_PHYSICAL_GPUS]{};
    NvU32 count = 0;
    NvCheck(enumerate(handles, &count), "NvAPI_EnumPhysicalGPUs");
    for (NvU32 i = 0; i < count && i < NVAPI_MAX_PHYSICAL_GPUS; ++i) {
        NvLogicalGpuHandle group = nullptr;
        LUID luid{};
        NV_LOGICAL_GPU_DATA data{};
        data.version = NV_LOGICAL_GPU_DATA_VER;
        data.pOSAdapterId = &luid;
        if (logical(handles[i], &group) != NVAPI_OK || info(group, &data) != NVAPI_OK || luid.LowPart != selected.LowPart ||
            luid.HighPart != selected.HighPart) continue;
        NV_GPU_ARCH_INFO a{};
        a.version = NV_GPU_ARCH_INFO_VER;
        NvCheck(arch(handles[i], &a), "NvAPI_GPU_GetArchInfo");
        return {unsigned(a.architecture), unsigned(a.implementation)};
    }
    Fail("no NVIDIA GPU has the D3D12 adapter's LUID");
}
bool IsAmpere(std::pair<unsigned, unsigned> a) { return a.first == NV_GPU_ARCHITECTURE_GA100 && a.second != NV_GPU_ARCH_IMPLEMENTATION_GA100; }
bool IsAda(std::pair<unsigned, unsigned> a) { return a.first == NV_GPU_ARCHITECTURE_AD100; }
bool IsBlackwell2(std::pair<unsigned, unsigned> a) { return a.first == NV_GPU_ARCHITECTURE_GB200; }

// ---- geometry (host-spec/geom.py) -------------------------------------------------------------------------------------
int64_t Up(int64_t n, int64_t a) { return (n + a - 1) / a * a; }
int64_t Cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
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
bool Has(const std::string& n, const char* part) { return n.find(part) != std::string::npos; }
bool EndsWith(const std::string& n, const std::string& tail) { return n.size() >= tail.size() && n.compare(n.size() - tail.size(), tail.size(), tail) == 0; }
std::string Family(const std::string& n) {
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
std::string Variant(const std::string& n) {
    for (const char* v : {"inpview_tilesync", "ds_wait", "upsample_tilesync", "outview_wait", "chained", "publish", "tilesync", "wait"})
        if (EndsWith(n, std::string("_") + v + "_fp8") || EndsWith(n, std::string("_") + v)) return v;
    return "plain";
}
int SwinLevel(const std::string& n) {
    if (Has(n, "_32_")) return 1;
    if (Has(n, "_64_")) return 2;
    if (Has(n, "_128_")) return 3;
    if (Has(n, "_256_")) return 4;
    Fail("no Swin width in " + n);
}
struct Dims { uint32_t x = 1, y = 1, z = 1; };
std::pair<Dims, Dims> Grid(const std::string& name, const Geometry& g, int sx, int sy) {
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
std::vector<std::pair<std::string, int64_t>> OwnArrays(const std::string& name, Dims g, int64_t T) {
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
struct PlanLaunch {
    int idx = 0;
    std::string name, layout, styles;
    uint32_t params = 0;
    int sx = 0, sy = 0;
    bool barrier = false;
    std::vector<Field> fields;
    std::vector<unsigned char> parameters;
    NVAPI_CU_KERNEL_LAUNCH_PARAMS dispatch {};
    std::unique_ptr<NrFourBitPlan::Operation> fourBit;
};
struct PlanTensor { std::string name, kind; int a = 0, b = 0; int first = 0, last = 0; };
struct Plan {
    std::array<std::array<float, 3>, 14> grades{};
    std::vector<PlanTensor> tensors;
    std::vector<PlanLaunch> launches;
};
float BitsToFloat(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

Plan ReadPlan(const fs::path& path) {
    std::ifstream file(path);
    if (!file) Fail("missing plan " + path.string());
    Plan p;
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
Dim ParseDim(const std::string& word) {
    Dim d;
    if (word == "=") return d;
    if (word[0] == '*' || word[0] == '/') { d.mode = word[0]; d.value = uint32_t(std::stoul(word.substr(1))); }
    else { d.mode = 'n'; d.value = uint32_t(std::stoul(word)); }
    if (!d.value) throw std::runtime_error("bad dimension " + word);
    return d;
}
std::array<Dim, 3> ParseDims(std::istringstream& words) {
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
uint32_t Apply(const Dim& d, uint32_t plan) { return d.mode == 'n' ? d.value : d.mode == '*' ? plan * d.value : d.mode == '/' ? (plan + d.value - 1) / d.value : plan; }
std::vector<int> ParseLaunchNumbers(const std::string& text) {
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
std::string BaseName(std::string file) {
    for (const char* ending : {".sm120.cubin", ".sm86.cubin", ".sm89.cubin", ".sm86.ptx.fatbin"}) {
        const size_t n = strlen(ending);
        if (file.size() > n && file.compare(file.size() - n, n, ending) == 0) return file.substr(0, file.size() - n);
    }
    return file;
}
// All manifests of all folders; per launch number one line per parameter size.
using Choices = std::map<int, std::vector<KernelChoice>>;
void ReadFolder(const fs::path& folder, const std::string& code, Choices& choices) {
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
std::string ModuleFile(const KernelChoice& c) {
    if (c.code == "sm86") return c.file + ".sm86.cubin";
    if (c.code == "sm86ptx") return c.file + ".sm86.ptx.fatbin";
    if (c.code == "sm89") return c.file + ".sm89.cubin";
    return c.file + ".sm120.cubin";                 // sm120, nvidia
}

// ---- per-device objects -----------------------------------------------------------------------------------------------
struct Records { ComPtr<ID3D12Resource> buffer; uint64_t blendAt = 0; std::map<int, std::pair<uint64_t, std::string>> offsets; std::map<int, uint64_t> bytes; };
struct DeviceState {
    std::optional<bool> hostFp8;
    std::map<std::string, NVDX_ObjectHandle> modules;                     // by module path
    std::map<std::pair<std::string, std::string>, NVDX_ObjectHandle> functions;
    std::map<std::string, std::shared_ptr<Records>> records;              // by records file
    std::vector<ComPtr<IUnknown>> retired;                                // objects of destroyed hosts (GPU may still use them)
};
std::mutex g_mutex;
std::map<ID3D12Device*, DeviceState>& Devices() { static auto* d = new std::map<ID3D12Device*, DeviceState>; return *d; }

NVDX_ObjectHandle FunctionFor(ID3D12Device* device, const KernelChoice& c) {
    Api& api = GetApi();
    auto& d = Devices()[device];
    const fs::path path = c.folder / ModuleFile(c);
    const std::string key = path.string();
    if (const auto f = d.functions.find({key, c.entry}); f != d.functions.end()) return f->second;
    NVDX_ObjectHandle module = nullptr;
    if (const auto m = d.modules.find(key); m != d.modules.end()) module = m->second;
    else {
        const auto blob = ReadAll(path);
        NvCheck(api.createModule(device, blob.data(), NvU32(blob.size()), &module), "loading kernel module " + path.filename().string() + " (" + c.code + ")");
        d.modules[key] = module;
    }
    NVDX_ObjectHandle function = nullptr;
    NvCheck(api.createFunction(device, module, c.entry.c_str(), &function), "kernel " + c.entry + " in " + path.filename().string());
    d.functions[{key, c.entry}] = function;
    return function;
}

std::shared_ptr<Records> RecordsFor(ID3D12Device* device, const fs::path& folder, const std::string& prefix, unsigned set) {
    const std::string stem = prefix + std::to_string(set);
    const fs::path bin = folder / (stem + ".bin"), txt = folder / (stem + ".txt");
    auto& d = Devices()[device];
    if (const auto found = d.records.find(bin.string()); found != d.records.end()) return found->second;
    auto r = std::make_shared<Records>();
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
            if (!bytes || bytes > UINT32_MAX || r->offsets.contains(n)) Fail("invalid or duplicate record in " + txt.string());
            r->offsets[n] = {offset, name};
            r->bytes[n] = bytes;
        }
    }
    const auto blob = ReadAll(bin);
    if (r->offsets.empty()) Fail("empty records manifest " + txt.string());
    for (const auto& [index, entry] : r->offsets)
        if (entry.first > blob.size() || r->bytes.at(index) > blob.size() - entry.first)
            Fail("record " + std::to_string(index) + " extends beyond " + bin.string());
    r->blendAt = (blob.size() + 511) / 512 * 512;
    const uint64_t bytes = (r->blendAt + 512 + 4096 + 255) / 256 * 256;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r->buffer)), "record buffer");
    ComPtr<ID3D12Resource> upload;
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "record upload buffer");
    void* mapped = nullptr;
    Check(upload->Map(0, nullptr, &mapped), "record upload map");
    memset(mapped, 0, size_t(bytes));
    memcpy(mapped, blob.data(), blob.size());
    memcpy(static_cast<unsigned char*>(mapped) + r->blendAt, &kBlendWeight, 2);
    upload->Unmap(0, nullptr);
    NrWeightUpload::Copy(device, r->buffer.Get(), upload.Get(), bytes, D3D12_RESOURCE_STATE_COMMON);
    d.records[bin.string()] = r;
    Log("records " + stem + " loaded (" + std::to_string(blob.size() >> 20) + " MB)");
    return r;
}

// ---- runtime launch trace (lab) ----------------------------------------------------------------------------------------
struct TraceState {
    std::mutex mutex;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    std::map<NVDX_ObjectHandle, std::string> names;
    std::ofstream out;
    bool opened = false;
    unsigned long long seq = 0;
};
TraceState& TS() { static auto* t = new TraceState; return *t; }
thread_local bool t_hostLaunching = false;
std::string Hex(const void* data, size_t size) {
    static const char* digits = "0123456789abcdef";
    std::string s(size * 2, '0');
    const auto* b = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) { s[2 * i] = digits[b[i] >> 4]; s[2 * i + 1] = digits[b[i] & 15]; }
    return s;
}
NvAPI_Status __cdecl TraceCreateFunction(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& t = TS();
    const auto r = t.createFunction(device, module, name, out);
    if (r == NVAPI_OK && name && out) { std::lock_guard g(t.mutex); t.names[*out] = name; }
    return r;
}
NvAPI_Status __cdecl TraceLaunch(ID3D12GraphicsCommandList* list, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* k, NvU32 count) {
    auto& t = TS();
    if (!t_hostLaunching && k) {
        std::lock_guard g(t.mutex);
        if (!t.opened) {
            const std::wstring path = Env(L"NR_HOST_TRACE_RUNTIME");
            if (!path.empty()) t.out.open(fs::path(path), std::ios::trunc);
            t.opened = true;
            if (t.out) t.out << "seq,list,name,grid,block,shared,size,params\n";
        }
        if (t.out) {
            for (NvU32 i = 0; i < count; ++i) {
                const auto found = t.names.find(k[i].hFunction);
                t.out << t.seq++ << "," << static_cast<const void*>(list) << "," << (found == t.names.end() ? "?" : found->second) << "," << k[i].gridDim.x << "x" << k[i].gridDim.y << "x"
                      << k[i].gridDim.z << "," << k[i].blockDim.x << "x" << k[i].blockDim.y << "x" << k[i].blockDim.z << "," << k[i].dynSharedMemBytes << "," << k[i].paramSize << ","
                      << Hex(k[i].pParams, k[i].paramSize) << "\n";
            }
            t.out.flush();
        }
    }
    return t.launch(list, k, count);
}
}  // namespace

// ---- the host -----------------------------------------------------------------------------------------------------------
struct Host::Impl {
    ComPtr<ID3D12Device> device;
    Options options;
    unsigned W = 0, H = 0, set = 0;
    Plan plan;
    Geometry geometry{1920, 1080};
    std::vector<PlanLaunch*> order;                 // plan launches for this layout, in idx order
    std::shared_ptr<Records> records;
    // Prepared records ('prepare' in a manifest): one buffer per host, filled by the first evaluation.
    ComPtr<ID3D12Resource> prepared;
    std::map<int, uint64_t> preparedOffset, preparedBytes;
    std::map<int, NVDX_ObjectHandle> prepareFunctions;
    bool preparedDone = false;
    std::map<std::pair<int, std::string>, uint64_t> flagOffset;
    uint64_t flagBytes = 0, tensorBase = 0, arenaBytes = 0;
    std::vector<uint64_t> tensorOffset;
    ComPtr<ID3D12Resource> arena, history, netout, inputcopy;
    ComPtr<ID3D12DescriptorHeap> views, samplers;
    UINT viewStride = 0, samplerStride = 0;
    uint64_t historyTex = 0, historySurf = 0, netoutTex = 0, netoutSurf = 0, inputcopyTex = 0, inputcopySurf = 0;
    unsigned long long evaluations = 0;
    uint32_t sinceReset = 0;                              // the noise seed: evaluations since the last reset frame
    bool historyValid = false;
    bool reportedMissingMotion = false;
    std::ofstream trace;
    // NR_HOST_TIMING: a timestamp before the first launch and after each one; read back kTimingSlots evaluations later.
    static constexpr unsigned kTimingSlots = 4;
    ComPtr<ID3D12QueryHeap> timestamps;
    ComPtr<ID3D12Resource> timingReadback;
    uint64_t timestampFrequency = 0;
    unsigned timingStride = 0;                              // queries per slot
    std::array<std::vector<const PlanLaunch*>, kTimingSlots> timedLaunches;
    std::array<unsigned long long, kTimingSlots> timedEvaluation{};
    std::ofstream timing;
    void WriteTiming(unsigned slot) {
        auto& launches = timedLaunches[slot];
        if (launches.empty() || !timing) return;
        const D3D12_RANGE range{size_t(slot) * timingStride * 8, size_t(slot + 1) * timingStride * 8};
        void* mapped = nullptr;
        if (FAILED(timingReadback->Map(0, &range, &mapped))) return;
        const auto* t = reinterpret_cast<const uint64_t*>(static_cast<const unsigned char*>(mapped) + range.Begin);
        const double us = 1e6 / double(timestampFrequency);
        for (size_t k = 0; k < launches.size(); ++k)
            timing << timedEvaluation[slot] << "," << launches[k]->idx << "," << launches[k]->name << "," << double(t[k + 1] - t[k]) * us << '\n';
        timing << timedEvaluation[slot] << ",total,total," << double(t[launches.size()] - t[0]) * us << '\n';
        const D3D12_RANGE none{0, 0};
        timingReadback->Unmap(0, &none);
        timing.flush();
        launches.clear();
    }
    std::pair<unsigned, unsigned> arch{0, 0};
    std::string kindText;

    D3D12_CPU_DESCRIPTOR_HANDLE ViewCpu(UINT i) const { auto h = views->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * viewStride; return h; }
    D3D12_CPU_DESCRIPTOR_HANDLE SamplerCpu(UINT i) const { auto h = samplers->GetCPUDescriptorHandleForHeapStart(); h.ptr += SIZE_T(i) * samplerStride; return h; }
    uint64_t TexHandle(UINT view, UINT sampler) {
        NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS p{};
        p.structSizeIn = sizeof p;
        p.pDevice = device.Get();
        p.texDesc = ViewCpu(view);
        p.smpDesc = SamplerCpu(sampler);
        NvCheck(GetApi().mergedTexture(&p), "NvAPI_D3D12_GetCudaMergedTextureSamplerObject");
        return p.textureHandle;
    }
    uint64_t SurfHandle(UINT view) {
        NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS p{};
        p.structSizeIn = sizeof p;
        p.pDevice = device.Get();
        p.type = NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SURFACE;
        p.desc = ViewCpu(view);
        NvCheck(GetApi().independent(&p), "NvAPI_D3D12_GetCudaIndependentDescriptorObject");
        return p.handle;
    }
    void Srv(ID3D12Resource* r, UINT at) {
        const auto d = r->GetDesc();
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = d.Format;
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(r, &v, ViewCpu(at));
    }
    void Uav(ID3D12Resource* r, UINT at) {
        const auto d = r->GetDesc();
        D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
        v.Format = d.Format;
        v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(r, nullptr, &v, ViewCpu(at));
    }
    ComPtr<ID3D12Resource> Texture(const char* what) {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = W; d.Height = H; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
        d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        ComPtr<ID3D12Resource> r;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&r)), what);
        return r;
    }
    void EnsureGradingTextures() {
        if (netout) return;
        auto network = Texture("network output texture");
        auto input = Texture("input copy texture");
        Srv(network.Get(), 2); Uav(network.Get(), 3);
        Srv(input.Get(), 4); Uav(input.Get(), 5);
        netoutTex = TexHandle(2, 0); netoutSurf = SurfHandle(3);
        inputcopyTex = TexHandle(4, 0); inputcopySurf = SurfHandle(5);
        netout = std::move(network);
        inputcopy = std::move(input);
    }

};

namespace {
uint64_t TensorBytes(const PlanTensor& t, const Geometry& g) {
    const auto& R = g.R;
    const auto& C = g.C;
    if (t.kind == "tile") return uint64_t((R[t.a] / 4) * (C[t.a] / 4) * 16 * t.b);
    if (t.kind == "view") return uint64_t(R[t.a] * C[t.a] * t.b);
    if (t.kind == "act") return uint64_t(g.Gp * 16 * t.a);
    if (t.kind == "scratch") return uint64_t(t.a * g.Gp * 32768);
    if (t.kind == "scratch_dec") return uint64_t((R[6] / 4) * (C[6] / 4) * 16384);
    Fail("unknown tensor layout " + t.kind);
}
}  // namespace

Host::Host(std::unique_ptr<Impl> i) : impl(std::move(i)) {}
Host::~Host() {
    if (!impl) return;
    try {
        for (unsigned k = 0; k < Impl::kTimingSlots; ++k) impl->WriteTiming(unsigned((impl->evaluations + k) % Impl::kTimingSlots));
        if (impl->options.callerTracksGpuLifetime) return;
        std::lock_guard g(g_mutex);
        auto& d = Devices()[impl->device.Get()];
        for (IUnknown* o : {static_cast<IUnknown*>(impl->arena.Get()), static_cast<IUnknown*>(impl->history.Get()), static_cast<IUnknown*>(impl->netout.Get()),
                            static_cast<IUnknown*>(impl->inputcopy.Get()), static_cast<IUnknown*>(impl->views.Get()), static_cast<IUnknown*>(impl->samplers.Get()),
                            static_cast<IUnknown*>(impl->prepared.Get())})
            if (o) d.retired.emplace_back(o);
    } catch (...) {
    }
}

std::unique_ptr<Host> Host::Create(ID3D12Device* device, const Options& options, unsigned width, unsigned height, unsigned set) {
    std::lock_guard g(g_mutex);
    auto i = std::make_unique<Impl>();
    i->device = device;
    i->options = options;
    i->W = width; i->H = height; i->set = set;
    if (width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION || height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        Fail("frame dimensions exceed D3D12's texture limit");
    if (width < 64 || height < 64) Fail("frame " + std::to_string(width) + "x" + std::to_string(height) + " is too small");
    if (options.fourBit ? (set != 0 && set != 10 && set != 11 && set != 12 && set != 13) : (set < 1 || set > 7)) Fail("set " + std::to_string(set) + " does not exist");
    const std::string setName = options.recordsName + std::to_string(set);
    i->arch = Architecture(device);
    i->plan = ReadPlan(options.plan);
    if (options.kernels.empty()) Fail("no kernel folders (NR_HOST_ASSETS)");
    // Kernel folders and their codes; the architecture rule.
    Choices choices;
    std::map<int, const KernelChoice*> kernelOf;
    std::set<std::string> codes;
    for (const auto& [folder, code0] : options.kernels) {
        std::string code = code0.empty() ? options.code : code0;
        // (RTX 40: the sm86 cubins run there, compute capability 8.6 code on 8.9, until an sm89 build exists.)
        if (code.empty()) code = IsAmpere(i->arch) || IsAda(i->arch) ? "sm86" : IsBlackwell2(i->arch) ? "sm120" : "";
        if (code.empty()) Fail("this GPU (architecture 0x" + [&] { char b[16]; snprintf(b, sizeof b, "%x", i->arch.first); return std::string(b); }() + ") is not an RTX 30, 40 or 50 series GPU");
        const bool labOnly = code == "sm86ptx" || code == "nvidia";
        if (labOnly && !options.lab) Fail("the " + code + " code is a lab reference only (set NR_HOST_LAB=1)");
        const bool fits = code == "sm86" ? IsAmpere(i->arch) || IsAda(i->arch)
                        : code == "sm89" ? IsAda(i->arch)
                        : code == "sm120" || code == "nvidia" ? IsBlackwell2(i->arch)
                        : code == "sm86ptx" ? (IsAmpere(i->arch) || IsAda(i->arch) || IsBlackwell2(i->arch)) : false;
        if (!fits) Fail("the " + code + " kernels cannot run on this GPU (architecture 0x" + [&] { char b[16]; snprintf(b, sizeof b, "%x", i->arch.first); return std::string(b); }() + ")");
        codes.insert(code);
        ReadFolder(folder, code, choices);
    }
    if (options.mixedOriginal && set == 0)
        ReadFolder(options.plan.parent_path() / "mixed", "sm120", choices);
    for (const auto& c : codes) i->kindText += (i->kindText.empty() ? "" : "+") + c;
    // The launches of this layout and their kernels.
    const std::string layout = options.layout == 3108 ? "3108" : "3109";
    int swinSlot = 0;
    for (auto& l : i->plan.launches) {
        if (l.layout != "any" && l.layout != layout) continue;
        i->order.push_back(&l);
        if (options.fourBit) {
            auto& api = GetApi();
            const NrFourBitPlan::Driver driver{api.createModule, api.createFunction, api.destroyModule, api.destroyFunction, api.launch};
            auto& op = l.fourBit = std::make_unique<NrFourBitPlan::Operation>();
            op->driver = api.launch;
            const int block = l.name.starts_with("cc_tinlayout_fused_swin_") ? swinSlot++ : -1;
            op->swinSlot = block;
            op->weightSet = set ? set : ~0u;
            NrNet4::BindNative(*op, driver, device, l.name, l.idx);
            NrSwin4::BindNative(*op, driver, device, l.name, set, block);
            NrFfnV2::BindNative(*op, driver, device, l.name, set, block);
        }
        const auto found = choices.find(l.idx);
        if (l.fourBit && found == choices.end()) continue; // Selected four-bit layer supplies its kernel.
        if (found == choices.end()) Fail("no kernel for launch " + std::to_string(l.idx) + " (" + l.name + ") in the kernel folders");
        const KernelChoice* pick = nullptr;
        for (const auto& c : found->second) if (!c.params || c.params == l.params) { if (pick && pick->params == l.params) continue; pick = &c; if (c.params == l.params) break; }
        if (!pick) Fail("launch " + std::to_string(l.idx) + ": no kernel takes the " + std::to_string(l.params) + "-byte parameter block");
        if (!pick->name.empty() && pick->name != l.name) Fail("launch " + std::to_string(l.idx) + ": the manifest names " + pick->name + ", the plan " + l.name + " (" + pick->where + ")");
        kernelOf[l.idx] = pick;
        l.dispatch.hFunction = FunctionFor(device, *pick);
        if (!pick->prepare.empty()) {
            KernelChoice prep = *pick;
            prep.entry = pick->prepare;
            i->prepareFunctions[l.idx] = FunctionFor(device, prep);
        }
    }
    std::sort(i->order.begin(), i->order.end(), [](const PlanLaunch* a, const PlanLaunch* b) { return a->idx < b->idx; });
    // Records: every weighted launch must have one, under the plan's kernel name.
    if (!options.fourBit) i->records = RecordsFor(device, options.records, options.recordsName, set);
    else if (options.mixedOriginal && set == 0) {
        // Standalone exposes per-layer precision on Original. Its FP8 kernels consume the FP8 representation of the
        // same trained four-bit checkpoint. Pass-in-one presets always use their own four-bit transformer.
        i->records = RecordsFor(device, options.plan.parent_path(), "mixed_set", 1);
        for (auto* l : i->order) {
            if (l->fourBit->record) continue;
            for (const auto& field : l->fields) if (field.code == Code::Record) {
                const auto r = i->records->offsets.find(l->idx);
                if (r == i->records->offsets.end() || r->second.second != l->name)
                    Fail("missing trained FP8 equivalent for launch " + std::to_string(l->idx));
                l->fourBit->record = i->records->buffer->GetGPUVirtualAddress() + r->second.first;
            }
        }
    }
    for (const auto* l : i->order)
        for (const auto& f : l->fields)
            if (!options.fourBit && f.code == Code::Record) {
                const auto r = i->records->offsets.find(l->idx);
                if (r == i->records->offsets.end()) Fail(setName + ".txt has no record for launch " + std::to_string(l->idx));
                if (r->second.second != l->name) Fail(setName + ".txt launch " + std::to_string(l->idx) + " is " + r->second.second + ", the plan's is " + l->name);
            }
    {   // Prepared records.
        uint64_t total = 0;
        for (const auto& [idx, function] : i->prepareFunctions) {
            const auto bytes = i->records->bytes.find(idx);
            if (bytes == i->records->bytes.end() || !bytes->second) Fail("launch " + std::to_string(idx) + " has a 'prepare' but no record bytes in " + setName + ".txt");
            const uint64_t out = (uint64_t(double(bytes->second) * kernelOf.at(idx)->prepareFactor) + 255) / 256 * 256;
            i->preparedOffset[idx] = total;
            i->preparedBytes[idx] = out;
            total += out;
        }
        if (total) {
            D3D12_RESOURCE_DESC d{};
            d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            d.Width = total; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
            d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&i->prepared)), "prepared record buffer");
            Log("prepared records: " + std::to_string(i->prepareFunctions.size()) + " launches, " + std::to_string(total >> 20) + " MB");
        }
    }
    // Geometry, flag region, arena.
    Geometry& geo = i->geometry;
    geo = Geometry(width, height);
    uint64_t off = 0;
    for (const auto* l : i->order) {
        if (l->idx < 1 || l->idx > 154) continue;
        const auto [grid, block] = Grid(l->name, geo, l->sx, l->sy);
        for (const auto& [array, words] : OwnArrays(l->name, grid, geo.T)) {
            i->flagOffset[{l->idx, array}] = off;
            off += uint64_t(Up(4 * words, 512));
        }
    }
    i->flagBytes = off;
    geo.count = int64_t(off / 4);
    // Geometry and kernel selection are fixed for the lifetime of this host.
    for (auto* l : i->order) {
        const KernelChoice base{};
        const auto it = kernelOf.find(l->idx);
        const auto& kernel = it == kernelOf.end() ? base : *it->second;
        const auto [grid, block] = Grid(l->name, geo, l->sx, l->sy);
        auto& dispatch = l->dispatch;
        dispatch.gridDim = {Apply(kernel.grid[0], grid.x), Apply(kernel.grid[1], grid.y), Apply(kernel.grid[2], grid.z)};
        dispatch.blockDim = {Apply(kernel.block[0], block.x), Apply(kernel.block[1], block.y), Apply(kernel.block[2], block.z)};
        dispatch.dynSharedMemBytes = kernel.shared;
        dispatch.paramSize = l->params;
        if (l->fourBit) {
            l->fourBit->base = dispatch;
            dispatch.hFunction = nullptr;
            dispatch.gridDim = {grid.x, grid.y, grid.z};
            dispatch.blockDim = {block.x, block.y, block.z};
            dispatch.dynSharedMemBytes = 0;
        }
    }
    i->tensorBase = uint64_t(Up(int64_t(off), 4096));
    {   // first-fit, memory reused only after a barrier launch (host-spec/planner.py rule 'barrier')
        std::set<int> barrierLaunches;
        for (const auto* l : i->order) if (l->barrier) barrierLaunches.insert(l->idx);
        struct Placed { uint64_t off, bytes; int last; };
        std::vector<Placed> placed;
        std::vector<size_t> sorted(i->plan.tensors.size());
        for (size_t k = 0; k < sorted.size(); ++k) sorted[k] = k;
        std::stable_sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) { return i->plan.tensors[a].first < i->plan.tensors[b].first; });
        i->tensorOffset.assign(i->plan.tensors.size(), 0);
        uint64_t end = 0;
        for (const size_t k : sorted) {
            const PlanTensor& t = i->plan.tensors[k];
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
            i->tensorOffset[k] = at;
            placed.push_back({at, bytes, t.last});
            end = std::max(end, at + bytes);
        }
        i->arenaBytes = i->tensorBase + end + 65536;
    }
    {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = i->arenaBytes; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&i->arena)), "arena buffer");
    }
    i->history = i->Texture("history texture");
    {
        D3D12_DESCRIPTOR_HEAP_DESC h{};
        h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        h.NumDescriptors = kStatic + kRing * kPerEval;
        h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&i->views)), "descriptor heap");
        h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        h.NumDescriptors = 2;
        Check(device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&i->samplers)), "sampler heap");
        i->viewStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        i->samplerStride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        for (UINT k = 0; k < 2; ++k) {
            D3D12_SAMPLER_DESC s{};
            s.Filter = k == 0 ? D3D12_FILTER_MIN_MAG_MIP_POINT : D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            s.AddressU = s.AddressV = s.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            s.MaxAnisotropy = 1;
            s.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            s.MaxLOD = D3D12_FLOAT32_MAX;
            device->CreateSampler(&s, i->SamplerCpu(k));
        }
        i->Srv(i->history.Get(), 0); i->Uav(i->history.Get(), 1);
        const bool historyLinear = Env(L"NR_HOST_HISTORY_SAMPLER") != L"point";
        i->historyTex = i->TexHandle(0, historyLinear ? 1 : 0); i->historySurf = i->SurfHandle(1);
    }
    // Geometry, tensor addresses, synchronization flags and weights stay fixed until this host is retired.
    const uint64_t arena = i->arena->GetGPUVirtualAddress();
    const uint64_t recordBase = i->records ? i->records->buffer->GetGPUVirtualAddress() : 0;
    const uint64_t preparedBase = i->prepared ? i->prepared->GetGPUVirtualAddress() : 0;
    for (auto* l : i->order) {
        l->parameters.resize(l->params);
        std::erase_if(l->fields, [&](const Field& f) {
            const auto write = [&](auto value) { memcpy(l->parameters.data() + f.off, &value, sizeof(value)); };
            switch (f.code) {
            case Code::Zero: break;
            case Code::U32: write(uint32_t(f.value)); break;
            case Code::I: write(int32_t(geo.Symbol(f.sym))); break;
            case Code::F: write(float(geo.Symbol(f.sym))); break;
            case Code::Rcp: write(float(1.0 / double(geo.Symbol(f.sym)))); break;
            case Code::Ptr: write(arena + i->tensorBase + i->tensorOffset.at(size_t(f.value))); break;
            case Code::Flags: {
                const auto flags = i->flagOffset.find({int(int64_t(f.value)), f.sym});
                if (flags == i->flagOffset.end()) Fail("launch " + std::to_string(l->idx) + ": missing " + f.sym + " flags");
                write(arena + flags->second);
                break;
            }
            case Code::ClearBase: write(arena); break;
            case Code::Record: {
                const auto prepared = i->preparedOffset.find(l->idx);
                if (l->fourBit) write(l->fourBit->record);
                else if (prepared != i->preparedOffset.end()) write(preparedBase + prepared->second);
                else write(recordBase + i->records->offsets.at(l->idx).first);
                break;
            }
            case Code::Blend: write(l->fourBit ? l->fourBit->blend : recordBase + i->records->blendAt); break;
            default: return false;
            }
            return true;
        });
    }
    if (!options.timing.empty()) {
        i->timingStride = unsigned(i->order.size()) + 2;
        D3D12_QUERY_HEAP_DESC q{};
        q.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        q.Count = i->timingStride * Impl::kTimingSlots;
        Check(device->CreateQueryHeap(&q, IID_PPV_ARGS(&i->timestamps)), "timestamp query heap");
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = uint64_t(q.Count) * 8; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&i->timingReadback)), "timestamp readback");
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue;
        Check(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "timestamp frequency queue");
        Check(queue->GetTimestampFrequency(&i->timestampFrequency), "timestamp frequency");
        i->timing.open(options.timing, std::ios::trunc);
        if (i->timing) i->timing << "eval,idx,name,us\n";
    }
    if (!options.trace.empty()) {
        i->trace.open(options.trace, std::ios::trunc);
        if (i->trace) i->trace << "eval,idx,name,grid,block,shared,size,params\n";
    }
    Log("pass host ready: " + std::to_string(width) + "x" + std::to_string(height) + ", " + setName + ", kernels " + i->kindText + ", " +
        std::to_string(i->order.size()) + " launches, arena " + std::to_string(i->arenaBytes >> 20) + " MB (flags " + std::to_string(i->flagBytes) + " B), layout " + layout);
    return std::unique_ptr<Host>(new Host(std::move(i)));
}

void Host::Evaluate(ID3D12GraphicsCommandList* commands, ID3D12Resource* colour, ID3D12Resource* motion, ID3D12Resource* output, const Settings& s, bool reset) {
    std::lock_guard g(g_mutex);
    Impl& m = *impl;
    if (!colour || !output) Fail("no colour or output resource");
    const auto cd = colour->GetDesc(), od = output->GetDesc();
    if (cd.Width != m.W || cd.Height != m.H || od.Width != m.W || od.Height != m.H)
        Fail("the colour/output size differs from the host's " + std::to_string(m.W) + "x" + std::to_string(m.H));
    if (s.style > 2) Fail("style " + std::to_string(s.style));
    const bool resetFrame = reset || !m.historyValid || !motion;
    if (motion) m.reportedMissingMotion = false;
    else if (!reset && !m.reportedMissingMotion) {
        Log("motion vectors unavailable, resetting history until they return");
        m.reportedMissingMotion = true;
    }
    // Blending reads the unfiltered input and the raw model result. Keep the latter as temporal history.
    const bool graded = s.style != 0 || s.intensity != 1.f;
    if (graded) m.EnsureGradingTextures();
    const unsigned slot = unsigned(m.evaluations % kRing);
    const UINT base = kStatic + slot * kPerEval;
    m.Srv(colour, base + 0);
    if (!resetFrame) m.Srv(motion, base + 1);
    if (!graded) m.Srv(output, base + 2);
    m.Uav(output, base + 3);
    const uint64_t colourTex = m.TexHandle(base + 0, 0);
    const uint64_t motionTex = motion && !resetFrame ? m.TexHandle(base + 1, 0) : 0;
    const uint64_t outputTex = graded ? 0 : m.TexHandle(base + 2, 0);
    const uint64_t outputSurf = m.SurfHandle(base + 3);
    ID3D12DescriptorHeap* heaps[] = {m.views.Get(), m.samplers.Get()};
    commands->SetDescriptorHeaps(2, heaps);
    const uint64_t recordBase = m.records ? m.records->buffer->GetGPUVirtualAddress() : 0;
    const Geometry& geo = m.geometry;
    const auto tex = [&](const std::string& role) -> uint64_t {
        if (role == "colour") return colourTex;
        if (role == "history") return resetFrame ? 0 : m.historyTex;
        if (role == "motion") return motionTex;
        if (role == "netout") return m.netoutTex;
        if (role == "inputcopy") return m.inputcopyTex;
        if (role == "copysrc") return graded ? m.netoutTex : outputTex;
        Fail("unknown texture role " + role);
    };
    const auto surf = [&](const std::string& role) -> uint64_t {
        if (role == "output") return outputSurf;
        if (role == "history") return m.historySurf;
        if (role == "netout") return m.netoutSurf;
        if (role == "inputcopy") return m.inputcopySurf;
        if (role == "out154") return graded ? m.netoutSurf : outputSurf;
        Fail("unknown surface role " + role);
    };
    const auto style = [&](const std::string& name) -> uint32_t {
        const auto bits = [](float f) { uint32_t b; memcpy(&b, &f, 4); return b; };
        if (name == "tone") return bits(s.tone);
        if (name == "structure") return bits(s.structure);
        if (name == "stylecode") return bits(float(s.style) / 128.f);
        if (name == "skin") return bits(s.autoMask ? (s.skin >= 0 ? s.skin : s.structure) : -1.f);
        if (name == "other") return bits(s.autoMask ? s.structure : -1.f);
        if (name == "automask") return s.autoMask ? 1u : 0u;
        if (name == "depthinv") return 0u;
        if (name == "intensity") return bits(s.intensity);
        if (name.rfind("grade", 0) == 0) {
            const auto& gr = m.plan.grades[std::stoi(name.substr(5))];
            const float neutral = gr[0], preset = gr[std::min(s.style, 2u)];
            const float t = std::clamp(s.tone, 0.f, 1.f);
            return bits(neutral + t * (preset - neutral));
        }
        Fail("unknown style field " + name);
    };
    const auto launch = GetApi().launch;
    const bool timed = m.timestamps != nullptr;
    const unsigned timingSlot = unsigned(m.evaluations % Impl::kTimingSlots);
    unsigned query = timingSlot * m.timingStride;
    if (timed) {
        m.WriteTiming(timingSlot);                          // the evaluation kTimingSlots ago (finished by now)
        m.timedEvaluation[timingSlot] = m.evaluations;
        commands->EndQuery(m.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query++);
    }
    struct HostLaunching { HostLaunching() { t_hostLaunching = true; } ~HostLaunching() { t_hostLaunching = false; } } launching;
    const uint64_t preparedBase = m.prepared ? m.prepared->GetGPUVirtualAddress() : 0;
    if (m.prepared && !m.preparedDone) {     // the first evaluation turns the records into the kernels' own layout
        struct PrepareParams { uint64_t in, out; uint32_t inBytes, outBytes; int32_t launch, pad; } pp{};
        for (const auto& [idx, function] : m.prepareFunctions) {
            pp.in = recordBase + m.records->offsets.at(idx).first;
            pp.out = preparedBase + m.preparedOffset.at(idx);
            pp.inBytes = uint32_t(m.records->bytes.at(idx));
            pp.outBytes = uint32_t(m.preparedBytes.at(idx));
            pp.launch = idx;
            NVAPI_CU_KERNEL_LAUNCH_PARAMS p{};
            p.hFunction = function;
            p.gridDim = {std::max(1u, std::min(1024u, (pp.outBytes + 1023) / 1024)), 1, 1};
            p.blockDim = {256, 1, 1};
            p.pParams = &pp;
            p.paramSize = sizeof pp;
            const auto status = launch(commands, &p, 1);
            if (status != NVAPI_OK) NvCheck(status, "preparing the record of launch " + std::to_string(idx));
        }
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        commands->ResourceBarrier(1, &b);
        m.preparedDone = true;
    }
    for (PlanLaunch* l : m.order) {
        if (l->styles == "graded" && !graded) continue;
        const bool barrier = l->barrier || m.options.barriersAll;
        if (barrier) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.UAV.pResource = nullptr;
            commands->ResourceBarrier(1, &b);
        }
        auto& block = l->parameters;
        const auto put32 = [&](uint32_t off, uint32_t v) { memcpy(block.data() + off, &v, 4); };
        const auto putF = [&](uint32_t off, float v) { memcpy(block.data() + off, &v, 4); };
        const auto put64 = [&](uint32_t off, uint64_t v) { memcpy(block.data() + off, &v, 8); };
        for (const Field& f : l->fields) {
            switch (f.code) {
            case Code::TF: putF(f.off, resetFrame ? 0.f : float(geo.Symbol(f.sym))); break;
            case Code::TRcp: putF(f.off, resetFrame ? 0.f : float(1.0 / double(geo.Symbol(f.sym)))); break;
            case Code::Tex: put64(f.off, tex(f.sym)); break;
            case Code::Surf: put64(f.off, surf(f.sym)); break;
            case Code::Mv: putF(f.off, f.sym == "x" ? float(double(s.mvScaleX) / double(m.W)) : float(double(s.mvScaleY) / double(m.H))); break;
            case Code::Style: put32(f.off, style(f.sym)); break;
            case Code::Seed: put32(f.off, resetFrame ? 0u : m.sinceReset); break;
            default: Fail("unexpected dynamic launch field");
            }
        }
        auto p = l->dispatch;
        p.pParams = block.data();
        if (m.trace)
            m.trace << m.evaluations << "," << l->idx << "," << l->name << "," << p.gridDim.x << "x" << p.gridDim.y << "x" << p.gridDim.z << "," << p.blockDim.x << "x" << p.blockDim.y << "x"
                    << p.blockDim.z << "," << p.dynSharedMemBytes << "," << l->params << "," << Hex(block.data(), l->params) << "\n";
        NvAPI_Status status;
        if (l->fourBit) {
            NrFourBitPlan::Scope scope(*l->fourBit);
            status = NrPassBorrow::SubmitOwned(NrNet4::SubmitNative, l->name.c_str(), commands, &p, 1);
        } else {
            status = launch(commands, &p, 1);
        }
        if (status != NVAPI_OK) NvCheck(status, "launch " + std::to_string(l->idx) + " (" + l->name + ")");
        if (timed) { commands->EndQuery(m.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query++); m.timedLaunches[timingSlot].push_back(l); }
    }
    if (timed)
        commands->ResolveQueryData(m.timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timingSlot * m.timingStride, query - timingSlot * m.timingStride,
                                   m.timingReadback.Get(), uint64_t(timingSlot) * m.timingStride * 8);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    commands->ResourceBarrier(1, &b);
    if (m.trace) m.trace.flush();
    m.historyValid = true;
    m.sinceReset = resetFrame ? 1u : m.sinceReset + 1u;
    ++m.evaluations;
    if (!m.options.report.empty()) {
        std::ofstream r(m.options.report, std::ios::trunc);
        r << "own pass host: " << m.kindText << ", " << m.options.recordsName << m.set << ", " << m.W << "x" << m.H << ", evaluations " << m.evaluations
          << ", nvngx_dlssnr.dll loaded: " << (GetModuleHandleW(L"nvngx_dlssnr.dll") ? "yes" : "no")
          << ", NGX dispatcher loaded: " << (GetModuleHandleW(L"_nvngx.dll") ? "yes" : "no") << "\n";
    }
}

unsigned Host::Width() const { return impl->W; }
unsigned Host::Height() const { return impl->H; }
unsigned Host::Set() const { return impl->set; }
unsigned long long Host::Evaluations() const { return impl->evaluations; }
std::string Host::Status() const {
    return "own pass host: " + impl->kindText + ", " + impl->options.recordsName + std::to_string(impl->set) + ", " + std::to_string(impl->W) + "x" + std::to_string(impl->H) + ", " +
           std::to_string(impl->evaluations) + " evaluations";
}

bool Requested(ID3D12Device* device) {
    const std::wstring v = Env(L"NR_HOST");
    if (!v.empty()) return v != L"0";
    if (!device) return false;
    // Unset: on by itself where NVIDIA's runtime refuses the GPU (below the RTX 50 series), so an FP8 model runs there.
    std::lock_guard g(g_mutex);
    auto& cached = Devices()[device].hostFp8;
    if (cached) return *cached;
    bool own = false;
    try { const auto a = Architecture(device); own = IsAmpere(a) || IsAda(a); } catch (const std::exception&) {}
    cached = own;
    return own;
}

Options FromEnvironment(const fs::path& exeDir) {
    Options o;
    o.enabled = true;
    const std::wstring plan = Env(L"NR_HOST_PLAN");
    o.plan = plan.empty() ? exeDir / "nvfp4" / "host" / "modelA_fp8.plan" : fs::path(plan);
    const std::wstring records = Env(L"NR_FP8_ASSETS");
    o.records = records.empty() ? exeDir / "nvfp4" / "fp8" : fs::path(records);
    o.code = Narrow(Env(L"NR_HOST_CODE"));
    if (!o.code.empty() && o.code != "sm86" && o.code != "sm89" && o.code != "sm120" && o.code != "sm86ptx" && o.code != "nvidia")
        Fail("NR_HOST_CODE must be sm86, sm89, sm120, sm86ptx or nvidia");
    const std::wstring assets = Env(L"NR_HOST_ASSETS");
    std::wstring rest = assets.empty() ? (exeDir / "nvfp4" / "host").wstring() : assets;
    size_t start = 0;
    while (start <= rest.size()) {
        const size_t semi = rest.find(L';', start);
        std::wstring part = rest.substr(start, semi == std::wstring::npos ? std::wstring::npos : semi - start);
        if (!part.empty()) {
            std::string code;
            if (const size_t at = part.rfind(L'@'); at != std::wstring::npos) { code = Narrow(part.substr(at + 1)); part.resize(at); }
            o.kernels.push_back({fs::path(part), code});
        }
        if (semi == std::wstring::npos) break;
        start = semi + 1;
    }
    o.lab = Env(L"NR_HOST_LAB") == L"1";
    const std::wstring barriers = Env(L"NR_HOST_BARRIERS");
    if (!barriers.empty() && barriers != L"all" && barriers != L"inferred") Fail("NR_HOST_BARRIERS must be all or inferred");
    o.barriersAll = barriers == L"all";
    const std::wstring layout = Env(L"NR_HOST_LAYOUT");
    if (!layout.empty() && layout != L"3108" && layout != L"3109") Fail("NR_HOST_LAYOUT must be 3108 or 3109");
    o.layout = layout == L"3108" ? 3108 : 3109;
    o.trace = Env(L"NR_HOST_TRACE");
    o.report = Env(L"NR_HOST_REPORT");
    o.timing = Env(L"NR_HOST_TIMING");
    return o;
}

Options FourBitOptions(const fs::path& assets, bool allowMixedOriginal) {
    Options o;
    o.enabled = o.fourBit = true;
    o.mixedOriginal = allowMixedOriginal;
    const auto folder = assets / "host4";
    o.plan = folder / "modelA_fp8.plan";
    o.recordsName = "nvfp4 set ";
    o.kernels.emplace_back(folder, "sm120");
    o.report = Env(L"NR_HOST_REPORT");
    o.trace = Env(L"NR_HOST_TRACE");
    o.timing = Env(L"NR_HOST_TIMING");
    return o;
}

Options Int8Options(const fs::path& folder) {
    Options o;
    o.enabled = true;
    o.plan = folder / "modelA_i8.plan";
    o.records = folder;
    o.recordsName = "int8_set";
    o.kernels.push_back({folder, std::string()});
    o.code = Narrow(Env(L"NR_HOST_CODE"));
    if (!o.code.empty() && o.code != "sm86" && o.code != "sm120") Fail("NR_HOST_CODE must be sm86 or sm120 for the INT8 models");
    o.barriersAll = Env(L"NR_HOST_BARRIERS") == L"all";
    o.trace = Env(L"NR_HOST_TRACE");
    o.report = Env(L"NR_HOST_REPORT");
    o.timing = Env(L"NR_HOST_TIMING");
    return o;
}

void ReleaseDevice(ID3D12Device* device) {
    NrWeightUpload::Collect();
    std::lock_guard g(g_mutex);
    auto& devices = Devices();
    const auto found = devices.find(device);
    if (found == devices.end()) return;
    auto& state = found->second;
    if (!state.modules.empty()) {
        auto& api = GetApi();
        for (const auto& [key, function] : state.functions) api.destroyFunction(device, function);
        for (const auto& [path, module] : state.modules) api.destroyModule(device, module);
    }
    devices.erase(found);
}
void SetLog(void (*log)(const char*)) { g_log = log; }
void SetDriverQuery(void* query) { g_driverQuery = query; }

void* WrapNvapi(unsigned id, void* original) {
    if (!original || Env(L"NR_HOST_TRACE_RUNTIME").empty()) return original;
    auto& t = TS();
    if (id == kCreateFunction) { t.createFunction = reinterpret_cast<decltype(t.createFunction)>(original); return reinterpret_cast<void*>(&TraceCreateFunction); }
    if (id == kLaunch) { t.launch = reinterpret_cast<decltype(t.launch)>(original); return reinterpret_cast<void*>(&TraceLaunch); }
    return original;
}

}  // namespace NrHost
