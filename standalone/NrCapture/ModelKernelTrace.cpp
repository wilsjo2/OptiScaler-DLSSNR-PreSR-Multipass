#include "ModelKernelTrace.h"
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <detours/detours.h>
#include <nvapi.h>
#include <bcrypt.h>
#include "ModelLoopCopy.h"
#include "ModelBookendBridge.h"
#include "ModelLatentMix.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "bcrypt.lib")

namespace ModelKernelTrace {
namespace {
using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;
using Query = void* (__stdcall*)(unsigned);
constexpr unsigned kCapacity = 4096;
constexpr unsigned kCreateFunction = 0xe2436e22, kLaunch = 0x24973538;
constexpr unsigned kDestroyModule = 0x41c65285, kDestroyFunction = 0xdf295ea6;
constexpr unsigned kCreateModule = 0xad1a677d, kPtxSupported = 0x70c07832;
thread_local bool recordingThread = false;
thread_local bool loopingThread = false;

struct Function { NVDX_ObjectHandle module = nullptr; std::string name; };
struct Kernel {
    std::string name, group, params;
    uintptr_t handle = 0;
    NVAPI_DIM3 grid{}, block{};
    unsigned shared = 0, paramSize = 0;
};
struct Launch {
    std::vector<Kernel> kernels;
    unsigned timestamp = 0;
    NvAPI_Status result = NVAPI_OK;
    double cpuMs = 0;
};
struct SavedKernel {
    NVAPI_CU_KERNEL_LAUNCH_PARAMS launch{};
    std::vector<unsigned char> params;
};
struct Loop {
    bool active = false, replaying = false, verified = false;
    bool vit = false;
    float alpha = -1.0f;
    unsigned bodies = 1, replayed = 0;
    UINT64 bytes = 0;
    ID3D12GraphicsCommandList* commands = nullptr;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12RootSignature> mixRoot;
    ComPtr<ID3D12PipelineState> mixPipeline;
    ComPtr<ID3D12Resource> snapshot;
    UINT64 snapshotBytes = 0, counterBegin = 0, counterEnd = 0;
    std::vector<SavedKernel> saved;
    std::string error;
};
struct State {
    std::recursive_mutex mutex;
    HMODULE module = nullptr;
    Query query = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) create = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    std::map<NVDX_ObjectHandle, Function> functions;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12QueryHeap> heap;
    ComPtr<ID3D12Resource> readback;
    ID3D12GraphicsCommandList* commands = nullptr;
    UINT64 frequency = 0;
    unsigned queries = 0;
    bool installed = false, active = false, pending = false, timed = false;
    std::atomic<bool> failed{false};
    std::vector<Launch> launches;
    Clock::time_point begin;
    double cpuEvaluateMs = 0;
    Loop loop;
    NVDX_ObjectHandle bookendModule = nullptr, bookendPre = nullptr, bookendPost = nullptr;
    NVDX_ObjectHandle bookendImport = nullptr;
    unsigned bookendPreCalls = 0, bookendPostCalls = 0;
    bool bookendFeatures = false, bookendImportSelf = false;
    unsigned bookendWidth = 0, bookendHeight = 0, bookendPaddedWidth = 0, bookendPaddedHeight = 0;
    UINT64 bookendBytes = 0;
    ComPtr<ID3D12Resource> bookendLift, bookendWide, bookendLiftReadback, bookendWideReadback;
    NVDX_ObjectHandle bookendAnchorPost = nullptr, bookendWideOnlyPost = nullptr;
    bool bookendAnchorControl = false, bookendLoop = false;
    unsigned bookendShaderMode = 0, bookendIndex = 0;
    std::vector<SavedKernel> bookendSaved;
    std::string bookendError;
    ComPtr<ID3D12Resource> bookendInput, bookendWeights, bookendNext, bookendAnchor;
    ComPtr<ID3D12Resource> bookendNextReadback, bookendAnchorReadback;
    ComPtr<ID3D12RootSignature> bookendRoot;
    ComPtr<ID3D12PipelineState> bookendPipeline;
};
// Keep hooks and resources alive even if the probe exits through an exception
// before its outstanding command list is retired.
State& Get() { static State* state = new State; return *state; }
void Check(HRESULT status, const char* operation) {
    if (FAILED(status)) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(unsigned(status)));
}
double Milliseconds(Clock::duration time) { return std::chrono::duration<double, std::milli>(time).count(); }
std::string Group(const std::string& name) {
    // Preserve the earlier profiler's ordering: fused pre/post also contain swin.
    constexpr const char* rules[] = {"pre_block", "post_block", "split_swin", "cc_vit_1d", "cg2r_post_process",
        "cg2r_copy", "dec_input_upsample", "cc_cb_clear", "swin_1h_32", "swin_2h_64", "swin_4h_128", "swin_8h_256"};
    for (const auto* rule : rules) if (name.find(rule) != std::string::npos) return rule;
    return "other";
}
std::string Hex(const void* data, size_t size) {
    constexpr char digits[] = "0123456789abcdef";
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::string result(size * 2, '0');
    for (size_t i = 0; i < size; ++i) { result[i * 2] = digits[bytes[i] >> 4]; result[i * 2 + 1] = digits[bytes[i] & 15]; }
    return result;
}
std::string Json(const std::string& text) {
    std::ostringstream output; output << '"';
    for (unsigned char value : text) {
        if (value == '"' || value == '\\') output << '\\' << char(value);
        else if (value < 32) output << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(value) << std::dec;
        else output << char(value);
    }
    output << '"'; return output.str();
}
std::string Csv(const std::string& text) {
    std::string result = "\"";
    for (char value : text) { if (value == '"') result += '"'; result += value; }
    return result + '"';
}

// Canonical sequence actually observed in this DLL's 320, 384, 960 and 3840
// captures. The architecture is fixed; grids and tensor addresses vary.
const std::vector<std::pair<std::string, unsigned>>& Expected() {
    static const auto sequence = [] {
        std::vector<std::pair<std::string, unsigned>> list;
        const auto add = [&](std::string name, unsigned bytes) { list.emplace_back(std::move(name), bytes); };
        add("cc_cb_clear", 16); add("cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8", 264);
        const unsigned counts[]{4, 4, 6, 8};
        for (unsigned level = 0; level < 4; ++level) {
            const unsigned heads = 1u << level;
            const std::string prefix = "cc_tinlayout_fused_swin_" + std::to_string(heads) + "h_" + std::to_string(heads * 32) + "_" + std::to_string(heads) + "_";
            for (unsigned block = 0; block < counts[level]; ++block)
                add(prefix + (block == 0 ? "inpview_tilesync_fp8" : block + 1 == counts[level] ? "ds_wait_fp8" : "chained_fp8"), level == 0 ? 96 : 88);
        }
        for (unsigned block = 0; block < 8; ++block) {
            add(block ? "cc_split_swin_16h_ffwd_512_chained_fp8" : "cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8", 56);
            add(block ? "cc_split_swin_16h_ffwd_proj_512_chained_fp8" : "cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8", 72);
            add("cc_split_swin_16h_qkv_512_chained_fp8", 56);
            add(block == 7 ? "cc_split_swin_16h_proj_pool_512_chained_fp8" : "cc_split_swin_16h_proj_512_chained_fp8", block == 7 ? 80 : 72);
            if (block == 7) add("cc_split_swin_16h_final_head_512_wait_fp8", 40);
        }
        add("cc_vit_1d_repack_2d_to_1d_fp8", 24);
        for (unsigned block = 0; block < 8; ++block) {
            add(block ? "cc_vit_1d_ffn_expand_chained_fp8" : "cc_vit_1d_ffn_expand_publish_fp8", 72);
            add("cc_vit_1d_ffn_contract_chained_fp8", 72); add("cc_vit_1d_qkv_chained_fp8", 80);
            add("cc_vit_1d_attention_chained_fp8", 64);
            add(block == 7 ? "cc_vit_1d_projection_wait_fp8" : "cc_vit_1d_projection_chained_fp8", 72);
        }
        add("cc_vit_1d_repack_1d_to_2d_fp8", 24); add("cc_dec_input_upsample_1024_512_tilesync_fp8", 80);
        for (unsigned block = 0; block < 8; ++block) {
            add("cc_split_swin_16h_ffwd_512_chained_fp8", 56); add("cc_split_swin_16h_ffwd_proj_512_chained_fp8", 72);
            add("cc_split_swin_16h_qkv_512_chained_fp8", 56);
            add(block == 7 ? "cc_split_swin_16h_proj_512_outview_wait_fp8" : "cc_split_swin_16h_proj_512_chained_fp8", 72);
        }
        for (int level = 3; level >= 0; --level) {
            const unsigned heads = 1u << level;
            const std::string prefix = "cc_tinlayout_fused_swin_" + std::to_string(heads) + "h_" + std::to_string(heads * 32) + "_" + std::to_string(heads) + "_";
            for (unsigned block = 0; block < counts[level]; ++block)
                add(prefix + (block == 0 ? "upsample_tilesync_fp8" : block + 1 == counts[level] ? "outview_wait_fp8" : "chained_fp8"), level == 0 ? 96 : 88);
        }
        add("cc_tinlayout_fused_post_block_swin_1h_32_fp8", 184); add("cg2r_copy_kernel", 72);
        return list;
    }();
    return sequence;
}
template<class T> T Parameter(const SavedKernel& kernel, size_t offset) {
    if (offset + sizeof(T) > kernel.params.size()) throw std::runtime_error("Latent-loop parameter bounds");
    T value; memcpy(&value, kernel.params.data() + offset, sizeof value); return value;
}
void UavBarrier(ID3D12GraphicsCommandList* commands) {
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    commands->ResourceBarrier(1, &barrier);
}
void VerifyRuntime(Loop& loop) {
    if (loop.verified) return;
    const auto module = GetModuleHandleW(L"nvngx_dlssnr.dll");
    wchar_t path[32768]{};
    if (!module || !GetModuleFileNameW(module, path, DWORD(std::size(path))))
        throw std::runtime_error("Latent loop requires the audited loaded nvngx_dlssnr.dll");
    std::ifstream input(std::filesystem::path(path), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot verify latent-loop runtime hash");
    struct Hash {
        BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr;
        ~Hash() { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); }
    } hash;
    if (BCryptOpenAlgorithmProvider(&hash.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
        BCryptCreateHash(hash.algorithm, &hash.hash, nullptr, 0, nullptr, 0, 0) < 0)
        throw std::runtime_error("Cannot initialize latent-loop runtime hash");
    std::vector<unsigned char> buffer(65536);
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), std::streamsize(buffer.size()));
        const auto count = input.gcount();
        if (count && BCryptHashData(hash.hash, buffer.data(), ULONG(count), 0) < 0)
            throw std::runtime_error("Cannot hash latent-loop runtime");
    }
    unsigned char digest[32]{};
    if (!input.eof() || BCryptFinishHash(hash.hash, digest, sizeof digest, 0) < 0 ||
        Hex(digest, sizeof digest) != "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e")
        throw std::runtime_error("Latent loop refused: runtime is not the audited 310.8 FP8 DLL");
    loop.verified = true;
}
void InitCopy(Loop& loop, ID3D12Device* device) {
    if (loop.device.Get() == device) return;
    D3D12_ROOT_PARAMETER parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[0].Descriptor.ShaderRegister = 0;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[1].Descriptor.ShaderRegister = 1;
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[2].Constants = {0, 0, 2};
    D3D12_ROOT_SIGNATURE_DESC signature{3, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> bytes, errors; ComPtr<ID3D12RootSignature> root; ComPtr<ID3D12PipelineState> pipeline;
    Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &bytes, &errors), "Latent-copy root signature");
    Check(device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(), IID_PPV_ARGS(&root)), "Latent-copy root signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC description{}; description.pRootSignature = root.Get();
    description.CS = {ModelLoopCopy, sizeof ModelLoopCopy};
    Check(device->CreateComputePipelineState(&description, IID_PPV_ARGS(&pipeline)), "Latent-copy pipeline");
    loop.device = device; loop.root = root; loop.pipeline = pipeline;
    loop.mixRoot.Reset(); loop.mixPipeline.Reset(); loop.snapshot.Reset(); loop.snapshotBytes = 0;
}
void InitLatentMix(Loop& loop, ID3D12Device* device) {
    if (loop.mixPipeline) return;
    D3D12_ROOT_PARAMETER parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[0].Descriptor.ShaderRegister = 0;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[1].Descriptor.ShaderRegister = 1;
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[2].Constants = {0, 0, 3};
    D3D12_ROOT_SIGNATURE_DESC signature{3, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> bytes, errors;
    Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &bytes, &errors), "ViT mix root signature");
    Check(device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(), IID_PPV_ARGS(&loop.mixRoot)), "ViT mix root signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC description{}; description.pRootSignature = loop.mixRoot.Get();
    description.CS = {ModelLatentMix, sizeof ModelLatentMix};
    Check(device->CreateComputePipelineState(&description, IID_PPV_ARGS(&loop.mixPipeline)), "ViT mix pipeline");
}
void ValidateVitCounterIsolation(const Loop& loop) {
    for (size_t index = 1; index < loop.saved.size(); ++index) {
        if (index >= 57 && index <= 98) continue;
        for (size_t offset = 0; offset + 8 <= loop.saved[index].params.size(); offset += 8) {
            const UINT64 pointer = Parameter<UINT64>(loop.saved[index], offset);
            if (pointer >= loop.counterBegin && pointer < loop.counterEnd)
                throw std::runtime_error("ViT counter span is referenced outside ViT; discard this command list");
        }
    }
}
void ReplayVit(State& state) {
    auto& loop = state.loop;
    if (loop.saved.size() != 100) throw std::runtime_error("ViT replay requires the original decoder boundary");
    const auto& inputRepack = loop.saved[57]; const auto& first = loop.saved[58];
    const auto& last = loop.saved[97]; const auto& outputRepack = loop.saved[98]; const auto& decoder = loop.saved[99];
    const UINT rows = Parameter<UINT>(inputRepack, 16), columns = Parameter<UINT>(inputRepack, 20);
    if (!rows || !columns || rows > 256 || columns > 256 || (rows | columns) % 4)
        throw std::runtime_error("Unsupported ViT padded extent");
    const UINT pixels = rows * columns; const UINT64 bytes = UINT64(pixels) * 1024;
    const UINT64 shape = Parameter<UINT64>(inputRepack, 16);
    const UINT64 source = Parameter<UINT64>(outputRepack, 8);
    const auto repackGrid = [](const SavedKernel& kernel, unsigned x) {
        return kernel.launch.gridDim.x == x && kernel.launch.gridDim.y == 1 && kernel.launch.gridDim.z == 1 &&
            kernel.launch.blockDim.x == 256 && kernel.launch.blockDim.y == 1 && kernel.launch.blockDim.z == 1;
    };
    if (!source || source % 16 || source > UINT64_MAX - bytes ||
        !repackGrid(inputRepack, ((pixels + 31) / 32) * 32) || !repackGrid(outputRepack, pixels) ||
        Parameter<UINT64>(outputRepack, 16) != shape || Parameter<UINT64>(decoder, 64) != shape ||
        Parameter<UINT64>(first, 0) != Parameter<UINT64>(inputRepack, 8) ||
        Parameter<UINT64>(last, 16) != Parameter<UINT64>(outputRepack, 0) ||
        Parameter<UINT64>(decoder, 0) != source || Parameter<UINT64>(first, 48) != 0 || Parameter<UINT64>(last, 56) != 0)
        throw std::runtime_error("ViT replay layout/launch/alias contract differs from audited captures");
    loop.counterBegin = Parameter<UINT64>(first, 56); loop.counterEnd = Parameter<UINT64>(decoder, 32);
    const UINT64 allCounters = Parameter<UINT64>(loop.saved[0], 0);
    const UINT64 allCounterBytes = UINT64(Parameter<UINT>(loop.saved[0], 8)) * 4;
    if (!allCounters || allCounters > UINT64_MAX - allCounterBytes || loop.counterBegin <= allCounters ||
        loop.counterEnd <= loop.counterBegin || loop.counterEnd > allCounters + allCounterBytes ||
        (loop.counterBegin | loop.counterEnd) % 512)
        throw std::runtime_error("ViT replay counter allocation differs from audited captures");
    std::set<UINT64> counters;
    UINT64 previousOutput = 0;
    for (size_t index = 58; index <= 97; ++index) {
        const auto& kernel = loop.saved[index];
        if (Parameter<UINT64>(kernel, kernel.params.size() - 8) != shape)
            throw std::runtime_error("ViT replay kernel extent mismatch");
        const size_t operation = (index - 58) % 5;
        const size_t inputOffset = operation == 2 ? 56 : operation == 3 ? 40 : 48;
        const size_t outputOffset = operation == 2 ? 64 : operation == 3 ? 48 : 56;
        const UINT64 input = Parameter<UINT64>(kernel, inputOffset), output = Parameter<UINT64>(kernel, outputOffset);
        if (input != previousOutput) throw std::runtime_error("ViT completion chain differs from audited captures");
        previousOutput = output;
        const auto includeCounter = [&](UINT64 pointer) {
            if (!pointer) return;
            if (pointer < loop.counterBegin || pointer >= loop.counterEnd || pointer % 512)
                throw std::runtime_error("ViT completion pointer escaped its allocation");
            counters.insert(pointer);
        };
        includeCounter(input); includeCounter(output);
        if (operation != 3) includeCounter(Parameter<UINT64>(kernel, operation == 2 ? 40 : 32));
    }
    if (previousOutput || counters.size() != 63 || *counters.begin() != loop.counterBegin ||
        *counters.rbegin() != Parameter<UINT64>(last, 32))
        throw std::runtime_error("ViT replay expected exactly 63 isolated completion allocations");
    ValidateVitCounterIsolation(loop);
    if (loop.snapshotBytes < bytes) {
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC description{}; description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = bytes; description.Height = 1; description.DepthOrArraySize = 1; description.MipLevels = 1;
        description.SampleDesc.Count = 1; description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        description.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        Check(loop.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&loop.snapshot)), "ViT snapshot allocation");
        loop.snapshotBytes = bytes;
    }
    loop.bytes = bytes;
    UavBarrier(loop.commands);
    loop.commands->SetComputeRootSignature(loop.root.Get()); loop.commands->SetPipelineState(loop.pipeline.Get());
    loop.commands->SetComputeRootUnorderedAccessView(0, source);
    loop.commands->SetComputeRootUnorderedAccessView(1, loop.snapshot->GetGPUVirtualAddress());
    const UINT vectors = UINT(bytes / 16), copyGroups = (vectors + 255) / 256, copyX = (std::min)(copyGroups, 65535u);
    const UINT copyConstants[]{vectors, copyX * 256};
    loop.commands->SetComputeRoot32BitConstants(2, 2, copyConstants, 0);
    loop.commands->Dispatch(copyX, (copyGroups + copyX - 1) / copyX, 1); UavBarrier(loop.commands);
    // The original clear writes -1, including partial-rendezvous counters. Only
    // the ViT's 63 allocations are reset; encoder and decoder flags are retained.
    auto clear = loop.saved[0];
    const UINT clearCount = UINT((loop.counterEnd - loop.counterBegin) / 4);
    memcpy(clear.params.data(), &loop.counterBegin, 8); memcpy(clear.params.data() + 8, &clearCount, 4);
    if (clear.launch.blockDim.x != 256 || clear.launch.blockDim.y != 1 || clear.launch.blockDim.z != 1)
        throw std::runtime_error("Unsupported ViT completion-clear geometry");
    clear.launch.gridDim = {(clearCount + 255) / 256, 1, 1};
    const auto launch = [&](const SavedKernel& saved) {
        auto call = saved.launch; call.pParams = saved.params.data();
        if (state.launch(loop.commands, &call, 1) != NVAPI_OK) throw std::runtime_error("Native ViT replay launch failed");
        UavBarrier(loop.commands);
    };
    launch(clear);
    // Reuse both native repacks. This also reconstructs the zero padding that
    // the 1D tensor layout requires for sizes such as the 2160-token 4K latent.
    auto repack = inputRepack; const UINT64 savedAddress = loop.snapshot->GetGPUVirtualAddress();
    memcpy(repack.params.data(), &savedAddress, 8); launch(repack); ++loop.replayed;
    for (size_t index = 58; index <= 98; ++index) { launch(loop.saved[index]); ++loop.replayed; }
    loop.commands->SetComputeRootSignature(loop.mixRoot.Get()); loop.commands->SetPipelineState(loop.mixPipeline.Get());
    loop.commands->SetComputeRootUnorderedAccessView(0, savedAddress);
    loop.commands->SetComputeRootUnorderedAccessView(1, source);
    const UINT words = UINT(bytes / 4), mixGroups = (words + 255) / 256, mixX = (std::min)(mixGroups, 65535u);
    struct { UINT words, rowStride; float alpha; } constants{words, mixX * 256, loop.alpha};
    loop.commands->SetComputeRoot32BitConstants(2, 3, &constants, 0);
    loop.commands->Dispatch(mixX, (mixGroups + mixX - 1) / mixX, 1); UavBarrier(loop.commands);
}
NvAPI_Status __cdecl LaunchKernels(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
void ReplayBody(State& state) {
    auto& loop = state.loop;
    if (loop.saved.size() != 155) throw std::runtime_error("Latent loop requires exactly the captured pre/body/post boundary");
    const auto& clear = loop.saved[0]; const auto& pre = loop.saved[1];
    const auto& first = loop.saved[2]; const auto& last = loop.saved[153]; const auto& post = loop.saved[154];
    const UINT64 counters = Parameter<UINT64>(clear, 0), counterBytes = UINT64(Parameter<UINT>(clear, 8)) * 4;
    const UINT64 skip = Parameter<UINT64>(pre, 216), destination = Parameter<UINT64>(first, 0), source = Parameter<UINT64>(last, 8);
    const UINT rows = Parameter<UINT>(first, 24), columns = Parameter<UINT>(first, 28);
    if (!rows || !columns || rows > 8192 || columns > 8192) throw std::runtime_error("Unsupported latent-loop padded extent");
    const UINT64 bytes = UINT64(rows) * columns * 32;
    if (!counters || counterBytes == 0 || counters > UINT64_MAX - counterBytes || counters + counterBytes != skip ||
        skip > UINT64_MAX - 4 * bytes || destination != skip + 4 * bytes || destination > UINT64_MAX - 4 * bytes ||
        source != destination + 3 * bytes || (destination | source | bytes) % 16 ||
        Parameter<UINT64>(pre, 248) != destination || Parameter<UINT>(pre, 256) != rows || Parameter<UINT>(pre, 260) != columns ||
        Parameter<UINT>(pre, 240) != rows * 2 || Parameter<UINT>(pre, 244) != columns * 2 ||
        Parameter<UINT64>(first, 40) != 0 || Parameter<UINT64>(first, 56) != counters ||
        Parameter<UINT64>(first, 8) != destination + bytes ||
        Parameter<UINT64>(loop.saved[3], 8) != destination + 2 * bytes || Parameter<UINT64>(loop.saved[4], 8) != source ||
        Parameter<UINT64>(last, 24) != Parameter<UINT64>(first, 24) || Parameter<UINT64>(last, 88) != Parameter<UINT64>(first, 24) ||
        Parameter<UINT64>(post, 0) != source || Parameter<UINT64>(post, 8) != skip)
        throw std::runtime_error("Latent loop refused: feature layout/counter/alias contract differs from audited captures");
    for (size_t offset = 0; offset + 8 <= pre.params.size(); offset += 8) {
        const auto pointer = Parameter<UINT64>(pre, offset);
        if (pointer >= counters && pointer < skip) throw std::runtime_error("Pre-block depends on counters that would be cleared");
    }
    for (size_t index = 2; index <= 153; ++index)
        for (size_t offset = 0; offset + 8 <= loop.saved[index].params.size(); offset += 8) {
            const auto pointer = Parameter<UINT64>(loop.saved[index], offset);
            if (pointer >= skip && pointer < destination) throw std::runtime_error("Body aliases the preserved pre-block skip");
        }
    loop.bytes = bytes;
    loop.replaying = true;
    // Audited DLL: EnsureBufferPool (RVA 0x21880) transitions its default-heap
    // UAV buffer to state 8 at 0x21a15, obtains GetGPUVirtualAddress through
    // backend slot +0xb8 (0x5d590), then binds that address at 0x21b17.
    // These feature pointers are offsets into that same D3D12 UAV allocation.
    const auto launch = [&](const SavedKernel& saved) {
        auto call = saved.launch; call.pParams = saved.params.data();
        if (LaunchKernels(loop.commands, &call, 1) != NVAPI_OK) throw std::runtime_error("Native latent-loop replay launch failed");
    };
    for (unsigned repetition = 1; repetition < loop.bodies; ++repetition) {
        UavBarrier(loop.commands);
        loop.commands->SetComputeRootSignature(loop.root.Get()); loop.commands->SetPipelineState(loop.pipeline.Get());
        loop.commands->SetComputeRootUnorderedAccessView(0, source);
        loop.commands->SetComputeRootUnorderedAccessView(1, destination);
        const UINT vectors = UINT(bytes / 16), groups = (vectors + 255) / 256, x = (std::min)(groups, 65535u);
        const UINT constants[]{vectors, x * 256};
        loop.commands->SetComputeRoot32BitConstants(2, 2, constants, 0);
        loop.commands->Dispatch(x, (groups + x - 1) / x, 1);
        UavBarrier(loop.commands);
        launch(clear); UavBarrier(loop.commands);
        for (size_t index = 2; index <= 153; ++index) {
            launch(loop.saved[index]); UavBarrier(loop.commands); ++loop.replayed;
        }
    }
    loop.replaying = false;
}
bool ObserveLoop(State& state, ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    auto& loop = state.loop;
    if (!loopingThread || !loop.active || loop.replaying || loop.bodies == 1) return true;
    try {
        if (commands != loop.commands || count != 1 || !kernels || !loop.error.empty()) throw std::runtime_error("Unexpected native latent-loop launch context");
        const size_t index = loop.saved.size(); const auto& expected = Expected();
        const auto found = state.functions.find(kernels->hFunction);
        if (index >= expected.size() || found == state.functions.end() || found->second.name != expected[index].first ||
            kernels->paramSize != expected[index].second || !kernels->pParams || !kernels->gridDim.x || !kernels->gridDim.y || !kernels->gridDim.z)
            throw std::runtime_error("Unsupported native latent-loop kernel sequence/ABI at launch " + std::to_string(index));
        SavedKernel saved; saved.launch = *kernels;
        const auto* params = static_cast<const unsigned char*>(kernels->pParams);
        saved.params.assign(params, params + kernels->paramSize); loop.saved.push_back(std::move(saved));
        if (loop.vit) { if (index == 99) ReplayVit(state); }
        else if (index == 154) ReplayBody(state); // original post-block pre-barrier has already been recorded
        return true;
    } catch (const std::exception& error) {
        loop.replaying = false; loop.error = error.what(); return false;
    } catch (...) { loop.replaying = false; loop.error = "Native latent-loop capture failed"; return false; }
}

NvAPI_Status __cdecl Create(ID3D12Device* device, NVDX_ObjectHandle module, const char* name, NVDX_ObjectHandle* out) {
    auto& state = Get();
    const auto result = state.create(device, module, name, out);
    if (result == NVAPI_OK && out) {
        try { std::lock_guard lock(state.mutex); state.functions[*out] = {module, name ? name : ""}; }
        catch (...) { state.failed = true; }
    }
    return result;
}
NvAPI_Status __cdecl DestroyFunction(ID3D12Device* device, NVDX_ObjectHandle function) {
    auto& state = Get(); const auto result = state.destroyFunction(device, function);
    if (result == NVAPI_OK) { std::lock_guard lock(state.mutex); state.functions.erase(function); }
    return result;
}
NvAPI_Status __cdecl DestroyModule(ID3D12Device* device, NVDX_ObjectHandle module) {
    auto& state = Get(); const auto result = state.destroyModule(device, module);
    if (result == NVAPI_OK) {
        std::lock_guard lock(state.mutex);
        for (auto it = state.functions.begin(); it != state.functions.end();)
            if (it->second.module == module) it = state.functions.erase(it); else ++it;
    }
    return result;
}
void VerifyAnchor(const State& state, const SavedKernel& post) {
    if (!Parameter<UINT64>(post, 56) || Parameter<float>(post, 64) != 0 || Parameter<float>(post, 68) != 0 ||
        Parameter<float>(post, 72) != float(state.bookendWidth) || Parameter<float>(post, 76) != float(state.bookendHeight) ||
        Parameter<float>(post, 80) != 1.0f / state.bookendWidth || Parameter<float>(post, 84) != 1.0f / state.bookendHeight ||
        Parameter<float>(post, 48) != 0.03125f || Parameter<UINT>(post, 52) != 1 ||
        Parameter<int>(post, 40) != -4 || Parameter<int>(post, 44) != -4)
        throw std::runtime_error("Bookend bridge requires identity source coordinates, SDR and intensity one");
}
void ReplayBookends(State& state, ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& exportedPost) {
    if (state.bookendSaved.size() != 155) throw std::runtime_error("Incomplete bookend loop capture");
    const auto& pre = state.bookendSaved[1]; const auto& post = state.bookendSaved[154];
    VerifyAnchor(state, post);
    const auto counters = Parameter<UINT64>(state.bookendSaved[0], 0);
    const UINT64 counterBytes = UINT64(Parameter<UINT>(state.bookendSaved[0], 8)) * 4;
    if (!counters || !counterBytes || counters + counterBytes != Parameter<UINT64>(pre, 216) || Parameter<UINT>(pre, 200) != 0)
        throw std::runtime_error("Bookend loop requires audited counter arena and reset seed zero");
    const auto launch = [&](const NVAPI_CU_KERNEL_LAUNCH_PARAMS& call) {
        if (state.launch(commands, &call, 1) != NVAPI_OK) throw std::runtime_error("Bookend replay launch failed");
        UavBarrier(commands);
    };
    auto wideOnly = exportedPost; wideOnly.hFunction = state.bookendWideOnlyPost;
    launch(wideOnly);
    D3D12_RESOURCE_BARRIER sources[2]{};
    for (unsigned i = 0; i < 2; ++i) {
        sources[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        sources[i].Transition = {i ? state.bookendWide.Get() : state.bookendLift.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    }
    commands->ResourceBarrier(2, sources);
    commands->SetComputeRootSignature(state.bookendRoot.Get()); commands->SetPipelineState(state.bookendPipeline.Get());
    commands->SetComputeRootShaderResourceView(0, state.bookendLift->GetGPUVirtualAddress());
    commands->SetComputeRootShaderResourceView(1, state.bookendWide->GetGPUVirtualAddress());
    commands->SetComputeRootShaderResourceView(2, state.bookendInput->GetGPUVirtualAddress());
    commands->SetComputeRootShaderResourceView(3, state.bookendWeights->GetGPUVirtualAddress());
    commands->SetComputeRootUnorderedAccessView(4, state.bookendNext->GetGPUVirtualAddress());
    commands->SetComputeRootUnorderedAccessView(5, state.bookendAnchor->GetGPUVirtualAddress());
    const UINT constants[]{state.bookendWidth, state.bookendHeight, state.bookendPaddedWidth, state.bookendPaddedHeight,
        state.bookendShaderMode, 0, 0, 0};
    commands->SetComputeRoot32BitConstants(6, 8, constants, 0);
    commands->Dispatch(state.bookendPaddedWidth / 8, state.bookendPaddedHeight / 8, 1);
    UavBarrier(commands);
    for (auto& barrier : sources) std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    commands->ResourceBarrier(2, sources);
    const auto replay = [&](size_t index, NVDX_ObjectHandle function = nullptr, UINT64 pointer = 0) {
        const auto& saved = state.bookendSaved[index]; auto call = saved.launch;
        auto params = saved.params;
        if (function) {
            call.hFunction = function; const size_t old = params.size(); params.resize(old + 8);
            memcpy(params.data() + old, &pointer, 8); call.paramSize += 8;
        }
        call.pParams = params.data(); launch(call);
    };
    replay(0);
    replay(1, state.bookendImport, state.bookendNext->GetGPUVirtualAddress());
    for (size_t index = 2; index <= 153; ++index) replay(index);
    replay(154, state.bookendAnchorPost, state.bookendAnchor->GetGPUVirtualAddress());
}
NvAPI_Status __cdecl LaunchKernels(ID3D12GraphicsCommandList* commands, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 count) {
    auto& state = Get();
    NVAPI_CU_KERNEL_LAUNCH_PARAMS bookendCall{};
    std::array<unsigned char, 272> bookendParams{};
    if (state.bookendModule) {
        std::lock_guard lock(state.mutex);
        if (!kernels || count != 1) return NVAPI_ERROR;
        const auto found = state.functions.find(kernels->hFunction);
        if (found == state.functions.end()) return NVAPI_ERROR;
        if (state.bookendLoop) {
            const auto& expected = Expected();
            if (!state.bookendError.empty() || state.bookendIndex >= expected.size() ||
                found->second.name != expected[state.bookendIndex].first ||
                kernels->paramSize != expected[state.bookendIndex].second || !kernels->pParams) return NVAPI_ERROR;
            SavedKernel saved; saved.launch = *kernels;
            const auto* p = static_cast<const unsigned char*>(kernels->pParams);
            saved.params.assign(p, p + kernels->paramSize);
            state.bookendSaved.push_back(std::move(saved)); ++state.bookendIndex;
        }
        const bool pre = found->second.name == Expected()[1].first;
        const bool post = found->second.name == Expected()[154].first;
        if (pre || post) {
            if (kernels->paramSize != Expected()[pre ? 1 : 154].second) return NVAPI_ERROR;
            bookendCall = *kernels;
            bookendCall.hFunction = pre ? state.bookendPre : state.bookendPost;
            if (state.bookendFeatures) {
                if (!kernels->pParams || kernels->blockDim.x != 32 || kernels->blockDim.y != 1 || kernels->blockDim.z != 1 ||
                    kernels->gridDim.z != 1 || kernels->gridDim.x != state.bookendPaddedWidth / 8 + (pre ? 0 : 1) ||
                    kernels->gridDim.y != state.bookendPaddedHeight / 8 + (pre ? 0 : 1)) return NVAPI_ERROR;
                memcpy(bookendParams.data(), kernels->pParams, kernels->paramSize);
                const auto u32 = [&](size_t offset) { unsigned value; memcpy(&value, bookendParams.data() + offset, 4); return value; };
                if (pre) {
                    if (u32(240) != state.bookendPaddedHeight || u32(244) != state.bookendPaddedWidth ||
                        u32(208) != state.bookendHeight || u32(212) != state.bookendWidth) return NVAPI_ERROR;
                    if (state.bookendImportSelf && state.bookendPreCalls) bookendCall.hFunction = state.bookendImport;
                } else if (u32(32) != state.bookendPaddedHeight || u32(36) != state.bookendPaddedWidth ||
                           u32(172) != state.bookendWidth || u32(176) != state.bookendHeight ||
                           u32(40) != UINT(-4) || u32(44) != UINT(-4)) return NVAPI_ERROR;
                UINT64 pointer = (pre ? state.bookendLift : state.bookendWide)->GetGPUVirtualAddress();
                if (post && state.bookendAnchorControl) {
                    SavedKernel saved; saved.params.assign(bookendParams.begin(), bookendParams.begin() + 184);
                    try { VerifyAnchor(state, saved); } catch (...) { return NVAPI_ERROR; }
                    bookendCall.hFunction = state.bookendAnchorPost; pointer = state.bookendInput->GetGPUVirtualAddress();
                }
                memcpy(bookendParams.data() + kernels->paramSize, &pointer, sizeof pointer);
                bookendCall.pParams = bookendParams.data(); bookendCall.paramSize += 8;
            }
            kernels = &bookendCall;
            ++(pre ? state.bookendPreCalls : state.bookendPostCalls);
            if (post && state.bookendLoop) {
                try { ReplayBookends(state, commands, bookendCall); return NVAPI_OK; }
                catch (const std::exception& error) { state.bookendError = error.what(); return NVAPI_ERROR; }
                catch (...) { state.bookendError = "Bookend replay failed"; return NVAPI_ERROR; }
            }
        }
    }
    if (loopingThread && state.loop.active) {
        std::lock_guard lock(state.mutex);
        if (!ObserveLoop(state, commands, kernels, count)) return NVAPI_ERROR;
    }
    // Preserve pointer, count, parameter bytes and launch order exactly.
    if (!recordingThread || !state.active || state.commands != commands)
        return state.launch(commands, kernels, count);
    std::lock_guard lock(state.mutex);
    Launch record;
    bool capture = !state.failed;
    if (capture) {
        try {
            if ((!kernels && count) || state.launches.size() >= kCapacity - 2) throw std::runtime_error("Trace capacity exceeded");
            record.kernels.reserve(count);
            for (NvU32 index = 0; index < count; ++index) {
                const auto& source = kernels[index]; Kernel kernel;
                const auto found = state.functions.find(source.hFunction);
                kernel.name = found == state.functions.end() ? "<unobserved function>" : found->second.name;
                kernel.group = Group(kernel.name); kernel.handle = reinterpret_cast<uintptr_t>(source.hFunction);
                kernel.grid = source.gridDim; kernel.block = source.blockDim;
                kernel.shared = source.dynSharedMemBytes; kernel.paramSize = source.paramSize;
                // Pointer payloads allow checking tensor aliases at the ViT boundary.
                // Large/unavailable payloads remain described by paramSize only.
                if (source.pParams && source.paramSize <= 4096) kernel.params = Hex(source.pParams, source.paramSize);
                record.kernels.push_back(std::move(kernel));
            }
        } catch (...) { state.failed = true; capture = false; }
    }
    const auto start = Clock::now();
    const auto result = state.launch(commands, kernels, count);
    const double cpuMs = Milliseconds(Clock::now() - start);
    if (capture) {
        try {
            record.result = result; record.cpuMs = cpuMs;
            if (state.timed) {
                record.timestamp = state.queries++;
                commands->EndQuery(state.heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, record.timestamp);
            }
            state.launches.push_back(std::move(record));
        } catch (...) { state.failed = true; }
    }
    return result;
}
void* __stdcall QueryInterface(unsigned id) {
    auto& state = Get(); void* original = state.query(id);
    if (!original) return original;
    std::lock_guard lock(state.mutex);
    switch (id) {
    case kCreateFunction: return reinterpret_cast<void*>(&Create);
    case kDestroyFunction: return reinterpret_cast<void*>(&DestroyFunction);
    case kDestroyModule: return reinterpret_cast<void*>(&DestroyModule);
    case kLaunch: return reinterpret_cast<void*>(&LaunchKernels);
    default: return original;
    }
}
}

void Install() {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (state.installed) return;
    state.module = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!state.module) throw std::runtime_error("Cannot load system NvAPI for model tracing");
    state.query = reinterpret_cast<Query>(GetProcAddress(state.module, "nvapi_QueryInterface"));
    if (!state.query) throw std::runtime_error("NvAPI query interface unavailable");
    // Resolve native entrypoints before attaching so passthrough always exists.
    state.create = reinterpret_cast<decltype(state.create)>(state.query(kCreateFunction));
    state.destroyFunction = reinterpret_cast<decltype(state.destroyFunction)>(state.query(kDestroyFunction));
    state.destroyModule = reinterpret_cast<decltype(state.destroyModule)>(state.query(kDestroyModule));
    state.launch = reinterpret_cast<decltype(state.launch)>(state.query(kLaunch));
    if (!state.create || !state.destroyFunction || !state.destroyModule || !state.launch)
        throw std::runtime_error("NvAPI CUDA entrypoints unavailable");
    LONG result = DetourTransactionBegin();
    if (result == NO_ERROR) {
        result = DetourUpdateThread(GetCurrentThread());
        if (result == NO_ERROR) result = DetourAttach(reinterpret_cast<PVOID*>(&state.query), reinterpret_cast<PVOID>(&QueryInterface));
        if (result == NO_ERROR) result = DetourTransactionCommit(); else DetourTransactionAbort();
    }
    if (result != NO_ERROR) throw std::runtime_error("NvAPI trace hook installation failed: " + std::to_string(result));
    state.installed = true;
}

void Begin(ID3D12GraphicsCommandList* commands, ID3D12CommandQueue* queue, bool perLaunchTiming) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.installed || !commands || !queue || state.active || state.pending)
        throw std::runtime_error("Invalid model trace Begin state; collect the previous completed sample first");
    ComPtr<ID3D12Device> device; Check(commands->GetDevice(IID_PPV_ARGS(&device)), "Trace command device");
    if (state.device.Get() != device.Get()) {
        D3D12_QUERY_HEAP_DESC query{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kCapacity, 0};
        ComPtr<ID3D12QueryHeap> heap; ComPtr<ID3D12Resource> readback;
        Check(device->CreateQueryHeap(&query, IID_PPV_ARGS(&heap)), "Trace query heap");
        D3D12_HEAP_PROPERTIES memory{}; memory.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{}; buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = UINT64(kCapacity) * sizeof(UINT64); buffer.Height = 1; buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1; buffer.SampleDesc.Count = 1; buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Check(device->CreateCommittedResource(&memory, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&readback)), "Trace readback");
        state.device = device; state.heap = heap; state.readback = readback;
    }
    Check(queue->GetTimestampFrequency(&state.frequency), "Trace queue timestamp frequency");
    if (!state.frequency) throw std::runtime_error("Zero trace timestamp frequency");
    state.launches.clear(); state.launches.reserve(kCapacity - 2);
    state.commands = commands; state.queries = 1; state.timed = perLaunchTiming; state.failed = false;
    commands->EndQuery(state.heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    state.begin = Clock::now(); state.active = true; recordingThread = true;
}
void LoadBookendControl(ID3D12Device* device, const std::filesystem::path& fatbin) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.installed || !device || state.bookendModule || state.loop.active || state.active)
        throw std::runtime_error("Invalid bookend control configuration");
    VerifyRuntime(state.loop);
    const auto supported = reinterpret_cast<decltype(&NvAPI_D3D12_IsFatbinPTXSupported)>(state.query(kPtxSupported));
    const auto createModule = reinterpret_cast<decltype(&NvAPI_D3D12_CreateCuModule)>(state.query(kCreateModule));
    bool available = false;
    if (!supported || !createModule || supported(device, &available) != NVAPI_OK || !available)
        throw std::runtime_error("Native PTX modules are unavailable on this device/driver");
    std::ifstream input(fatbin, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() > 16 * 1024 * 1024)
        throw std::runtime_error("Invalid bookend control fatbin size");
    std::vector<char> bytes(static_cast<size_t>(input.tellg())); input.seekg(0);
    input.read(bytes.data(), std::streamsize(bytes.size()));
    if (!input) throw std::runtime_error("Cannot read bookend control fatbin");
    NVDX_ObjectHandle module = nullptr, pre = nullptr, post = nullptr;
    auto result = createModule(device, bytes.data(), NvU32(bytes.size()), &module);
    if (result == NVAPI_OK) result = state.create(device, module, Expected()[1].first.c_str(), &pre);
    if (result == NVAPI_OK) result = state.create(device, module, Expected()[154].first.c_str(), &post);
    if (result != NVAPI_OK) {
        if (pre) state.destroyFunction(device, pre);
        if (post) state.destroyFunction(device, post);
        if (module) state.destroyModule(device, module);
        throw std::runtime_error("Cannot create native bookend control module/functions: " + std::to_string(result));
    }
    state.functions[pre] = {module, Expected()[1].first};
    state.functions[post] = {module, Expected()[154].first};
    state.bookendModule = module; state.bookendPre = pre; state.bookendPost = post;
}
void CheckBookendControl(unsigned evaluations) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.bookendModule || state.bookendPreCalls != evaluations || state.bookendPostCalls != evaluations)
        throw std::runtime_error("Bookend control did not replace exactly one pre/post pair per evaluation");
    if (state.bookendLoop) {
        if (!state.bookendError.empty()) throw std::runtime_error(state.bookendError + "; discard this command list");
        if (state.bookendIndex != 156 || state.bookendSaved.size() != 156)
            throw std::runtime_error("Incomplete native bookend loop; discard this command list");
        state.bookendIndex = 0; state.bookendSaved.clear();
    }
}
namespace {
ComPtr<ID3D12Resource> BookendBuffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = type;
    D3D12_RESOURCE_DESC description{}; description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = bytes; description.Height = 1; description.DepthOrArraySize = 1;
    description.MipLevels = 1; description.SampleDesc.Count = 1; description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    const auto initial = type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
        type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
    ComPtr<ID3D12Resource> result;
    Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description, initial, nullptr,
        IID_PPV_ARGS(&result)), "Bookend diagnostic buffer");
    return result;
}
}
void LoadBookendFeatures(ID3D12Device* device, const std::filesystem::path& fatbin,
                         unsigned width, unsigned height, bool importSelf) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.installed || !device || state.bookendModule || !width || !height || width > 1024 || height > 1024)
        throw std::runtime_error("Bookend feature diagnostics require an idle probe and dimensions at most 1024");
    VerifyRuntime(state.loop);
    const auto supported = reinterpret_cast<decltype(&NvAPI_D3D12_IsFatbinPTXSupported)>(state.query(kPtxSupported));
    const auto createModule = reinterpret_cast<decltype(&NvAPI_D3D12_CreateCuModule)>(state.query(kCreateModule));
    bool available = false;
    if (!supported || !createModule || supported(device, &available) != NVAPI_OK || !available)
        throw std::runtime_error("Native PTX modules unavailable");
    std::ifstream input(fatbin, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() > 16 * 1024 * 1024) throw std::runtime_error("Invalid feature fatbin");
    std::vector<char> bytes(static_cast<size_t>(input.tellg())); input.seekg(0);
    input.read(bytes.data(), std::streamsize(bytes.size()));
    if (!input) throw std::runtime_error("Cannot read bookend feature fatbin");
    NVDX_ObjectHandle module = nullptr, pre = nullptr, post = nullptr, imported = nullptr;
    auto status = createModule(device, bytes.data(), NvU32(bytes.size()), &module);
    if (status == NVAPI_OK) status = state.create(device, module, "nr_pre_export_lifted_fp16", &pre);
    if (status == NVAPI_OK) status = state.create(device, module, "nr_post_export_wide_fp16", &post);
    if (status == NVAPI_OK) status = state.create(device, module, "nr_pre_from_lifted_fp16", &imported);
    if (status != NVAPI_OK) throw std::runtime_error("Cannot load bookend feature functions: " + std::to_string(status));
    state.bookendWidth = width; state.bookendHeight = height;
    state.bookendPaddedWidth = (std::max)(320u, (width + 31) & ~31u);
    state.bookendPaddedHeight = (std::max)(320u, (height + 31) & ~31u);
    state.bookendBytes = UINT64(state.bookendPaddedWidth) * state.bookendPaddedHeight * 64;
    state.bookendLift = BookendBuffer(device, state.bookendBytes, D3D12_HEAP_TYPE_DEFAULT);
    state.bookendWide = BookendBuffer(device, state.bookendBytes, D3D12_HEAP_TYPE_DEFAULT);
    state.bookendLiftReadback = BookendBuffer(device, state.bookendBytes, D3D12_HEAP_TYPE_READBACK);
    state.bookendWideReadback = BookendBuffer(device, state.bookendBytes, D3D12_HEAP_TYPE_READBACK);
    state.functions[pre] = {module, "nr_pre_export_lifted_fp16"};
    state.functions[post] = {module, "nr_post_export_wide_fp16"};
    state.functions[imported] = {module, "nr_pre_from_lifted_fp16"};
    state.bookendModule = module; state.bookendPre = pre; state.bookendPost = post; state.bookendImport = imported;
    state.bookendFeatures = true; state.bookendImportSelf = importSelf;
}
void CaptureBookendFeatures(ID3D12GraphicsCommandList* commands) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.bookendFeatures) throw std::runtime_error("No bookend features to capture");
    const auto copy = [&](ID3D12Resource* source, ID3D12Resource* target) {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {source, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
        commands->ResourceBarrier(1, &barrier); commands->CopyResource(target, source);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter); commands->ResourceBarrier(1, &barrier);
    };
    copy(state.bookendLift.Get(), state.bookendLiftReadback.Get());
    copy(state.bookendWide.Get(), state.bookendWideReadback.Get());
    if (state.bookendLoop) {
        copy(state.bookendNext.Get(), state.bookendNextReadback.Get());
        copy(state.bookendAnchor.Get(), state.bookendAnchorReadback.Get());
    }
}
void SaveBookendFeatures(const std::filesystem::path& stem) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    const auto save = [&](ID3D12Resource* resource, const wchar_t* suffix) {
        std::ofstream file(stem.wstring() + suffix, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot save bookend features");
        const UINT64 byteCount = resource->GetDesc().Width;
        void* bytes = nullptr; D3D12_RANGE range{0, SIZE_T(byteCount)};
        Check(resource->Map(0, &range, &bytes), "Map bookend features");
        file.write(static_cast<const char*>(bytes), std::streamsize(byteCount));
        D3D12_RANGE none{}; resource->Unmap(0, &none);
        if (!file) throw std::runtime_error("Cannot write bookend features");
    };
    save(state.bookendLiftReadback.Get(), L".pre-lift.fp16");
    save(state.bookendWideReadback.Get(), L".post-wide.fp16");
    if (state.bookendLoop) {
        save(state.bookendNextReadback.Get(), L".next-lift.fp16");
        save(state.bookendAnchorReadback.Get(), L".anchor.rgba16f");
    }
    std::ofstream meta(stem.wstring() + L".features.json");
    meta << "{\"layout\":\"HWC32 FP16\",\"width\":" << state.bookendWidth << ",\"height\":" << state.bookendHeight
         << ",\"padded_width\":" << state.bookendPaddedWidth << ",\"padded_height\":" << state.bookendPaddedHeight
         << ",\"bytes_per_tensor\":" << state.bookendBytes << ",\"pre_import_after_first_frame\":" << (state.bookendImportSelf ? "true" : "false") << "}\n";
    if (!meta) throw std::runtime_error("Cannot write bookend feature metadata");
}
void ConfigureBookendBridge(ID3D12Device* device, std::span<const char> input,
                            const std::filesystem::path& weights, BookendBridge mode) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.bookendFeatures || state.bookendImportSelf || state.bookendPreCalls ||
        input.size() != size_t(state.bookendWidth) * state.bookendHeight * 8)
        throw std::runtime_error("Invalid bookend bridge setup");
    const auto upload = [&](const void* data, size_t size) {
        auto resource = BookendBuffer(device, size, D3D12_HEAP_TYPE_UPLOAD);
        void* mapped = nullptr; D3D12_RANGE none{};
        Check(resource->Map(0, &none, &mapped), "Map bookend upload"); memcpy(mapped, data, size); resource->Unmap(0, nullptr);
        return resource;
    };
    state.bookendInput = upload(input.data(), input.size());
    if (state.create(device, state.bookendModule, "nr_post_from_anchor_fp16", &state.bookendAnchorPost) != NVAPI_OK)
        throw std::runtime_error("Cannot create anchor post kernel");
    state.bookendAnchorControl = mode == BookendBridge::AnchorControl;
    if (state.bookendAnchorControl) return;
    if (state.create(device, state.bookendModule, "nr_post_wide_only_fp16", &state.bookendWideOnlyPost) != NVAPI_OK)
        throw std::runtime_error("Cannot create post-Swin-only kernel");
    std::ifstream file(weights, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != 768) throw std::runtime_error("Expected 768-byte bookend projection/colour weights");
    std::array<char, 768> coefficients{}; file.seekg(0); file.read(coefficients.data(), coefficients.size());
    if (!file) throw std::runtime_error("Cannot read bridge weights");
    state.bookendWeights = upload(coefficients.data(), coefficients.size());
    state.bookendNext = BookendBuffer(device, state.bookendBytes, D3D12_HEAP_TYPE_DEFAULT);
    state.bookendAnchor = BookendBuffer(device, input.size(), D3D12_HEAP_TYPE_DEFAULT);
    state.bookendNextReadback = BookendBuffer(device, state.bookendBytes, D3D12_HEAP_TYPE_READBACK);
    state.bookendAnchorReadback = BookendBuffer(device, input.size(), D3D12_HEAP_TYPE_READBACK);
    D3D12_ROOT_PARAMETER parameters[7]{};
    for (unsigned i = 0; i < 4; ++i) { parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; parameters[i].Descriptor.ShaderRegister = i; }
    for (unsigned i = 4; i < 6; ++i) { parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; parameters[i].Descriptor.ShaderRegister = i - 4; }
    parameters[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[6].Constants = {0, 0, 8};
    D3D12_ROOT_SIGNATURE_DESC signature{7, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> bytes, errors;
    Check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &bytes, &errors), "Bridge signature");
    Check(device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(), IID_PPV_ARGS(&state.bookendRoot)), "Bridge signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature = state.bookendRoot.Get();
    pipeline.CS = {ModelBookendBridge, sizeof ModelBookendBridge};
    Check(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&state.bookendPipeline)), "Bridge pipeline");
    state.bookendShaderMode = mode == BookendBridge::Affine ? 1 : 0;
    state.bookendSaved.reserve(156); state.bookendLoop = true;
}
void End(ID3D12GraphicsCommandList* commands) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.active || !recordingThread || state.commands != commands) throw std::runtime_error("Invalid model trace End state");
    state.cpuEvaluateMs = Milliseconds(Clock::now() - state.begin);
    recordingThread = false; state.active = false;
    commands->EndQuery(state.heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, state.queries++);
    commands->ResolveQueryData(state.heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, state.queries, state.readback.Get(), 0);
    state.pending = true;
}
Report Collect(const std::filesystem::path& stem) {
    auto& state = Get(); std::lock_guard lock(state.mutex);
    if (!state.pending || state.active) throw std::runtime_error("No completed model trace to collect");
    // Submission retirement is an explicit caller contract, unlike the old
    // profiler's heuristic wait for 90 subsequent evaluations.
    UINT64* mapped = nullptr; D3D12_RANGE range{0, SIZE_T(state.queries) * sizeof(UINT64)};
    Check(state.readback->Map(0, &range, reinterpret_cast<void**>(&mapped)), "Trace readback map");
    std::vector<UINT64> ticks(mapped, mapped + state.queries);
    D3D12_RANGE none{0, 0}; state.readback->Unmap(0, &none); state.pending = false;
    if (state.failed) throw std::runtime_error("Kernel trace was incomplete (capture allocation/capacity failure)");
    if (ticks.size() < 2 || !ticks.front() || !std::is_sorted(ticks.begin(), ticks.end()))
        throw std::runtime_error("Invalid completed trace timestamps");
    if (state.launches.empty()) throw std::runtime_error("Trace saw no NVAPI kernel launches; install before Runtime/model initialization");
    Report report; report.launchCalls = state.launches.size(); report.cpuEvaluateMs = state.cpuEvaluateMs;
    report.perLaunchTiming = state.timed;
    const auto milliseconds = [&](UINT64 from, UINT64 to) { return double(to - from) * 1000.0 / double(state.frequency); };
    report.gpuMs = milliseconds(ticks.front(), ticks.back());
    if (!stem.parent_path().empty()) std::filesystem::create_directories(stem.parent_path());
    auto csvPath = stem; csvPath += L".csv"; auto jsonPath = stem; jsonPath += L".json";
    std::ofstream csv(csvPath), json(jsonPath);
    if (!csv || !json) throw std::runtime_error("Cannot create model trace output");
    csv << "launch,kernel,name,group,function,grid_x,grid_y,grid_z,block_x,block_y,block_z,shared_bytes,param_size,launch_result,launch_cpu_ms,launch_interval_gpu_ms,params_hex\n";
    json << std::setprecision(12) << "{\n\"schema\":1,\"per_launch_timing\":" << (state.timed ? "true" : "false")
         << ",\"gpu_ms\":" << report.gpuMs << ",\"cpu_evaluate_ms\":" << report.cpuEvaluateMs
         << ",\"timestamp_frequency\":" << state.frequency << ",\"launches\":[\n";
    std::map<std::string, std::pair<size_t, double>> groups;
    UINT64 previous = ticks.front();
    for (size_t index = 0; index < state.launches.size(); ++index) {
        const auto& launch = state.launches[index]; report.cpuLaunchMs += launch.cpuMs;
        const double gpuMs = state.timed ? milliseconds(previous, ticks.at(launch.timestamp)) : 0;
        if (state.timed) previous = ticks.at(launch.timestamp);
        if (index) json << ",\n";
        json << "{\"index\":" << index << ",\"result\":" << int(launch.result) << ",\"cpu_ms\":" << launch.cpuMs << ",\"interval_gpu_ms\":";
        if (state.timed) json << gpuMs; else json << "null";
        json << ",\"kernels\":[";
        for (size_t kernelIndex = 0; kernelIndex < launch.kernels.size(); ++kernelIndex) {
            const auto& kernel = launch.kernels[kernelIndex]; ++report.kernels; ++groups[kernel.group].first;
            if (!kernelIndex && state.timed) groups[kernel.group].second += gpuMs;
            csv << index << ',' << kernelIndex << ',' << Csv(kernel.name) << ',' << Csv(kernel.group) << ',' << kernel.handle << ','
                << kernel.grid.x << ',' << kernel.grid.y << ',' << kernel.grid.z << ',' << kernel.block.x << ',' << kernel.block.y << ','
                << kernel.block.z << ',' << kernel.shared << ',' << kernel.paramSize << ',' << int(launch.result) << ',' << launch.cpuMs << ',';
            if (state.timed) csv << gpuMs;
            csv << ',' << Csv(kernel.params) << '\n';
            if (kernelIndex) json << ',';
            json << "{\"name\":" << Json(kernel.name) << ",\"group\":" << Json(kernel.group) << ",\"function\":" << kernel.handle
                 << ",\"grid\":[" << kernel.grid.x << ',' << kernel.grid.y << ',' << kernel.grid.z << "],\"block\":["
                 << kernel.block.x << ',' << kernel.block.y << ',' << kernel.block.z << "],\"shared_bytes\":" << kernel.shared
                 << ",\"param_size\":" << kernel.paramSize << ",\"params_hex\":" << Json(kernel.params) << '}';
        }
        json << "]}";
    }
    json << "\n],\"cpu_launch_ms\":" << report.cpuLaunchMs << ",\"kernel_count\":" << report.kernels << ",\"groups\":[";
    bool first = true;
    for (const auto& [group, values] : groups) {
        if (!first) json << ','; first = false;
        json << "{\"name\":" << Json(group) << ",\"kernels\":" << values.first << ",\"attributed_gpu_ms\":";
        if (state.timed) json << values.second; else json << "null";
        json << '}';
    }
    json << "],\"tail_gpu_ms\":" << (state.timed ? milliseconds(previous, ticks.back()) : 0)
         << ",\"timing_note\":\"Per-launch timestamps can change chained-kernel overlap. Each launch interval is attributed to its first kernel group; this is not isolated kernel execution time.\"}\n";
    if (!csv || !json) throw std::runtime_error("Writing model trace failed");
    return report;
}
void BeginLoop(ID3D12GraphicsCommandList* commands, ID3D12Device* device, unsigned bodies) {
    auto& state = Get(); std::lock_guard lock(state.mutex); auto& loop = state.loop;
    if (!state.installed || !commands || !device || loop.active || bodies < 1 || bodies > 3)
        throw std::runtime_error("Invalid native latent-loop Begin state");
    if (bodies > 1) { VerifyRuntime(loop); InitCopy(loop, device); }
    loop.vit = false;
    loop.commands = commands; loop.bodies = bodies; loop.replayed = 0; loop.bytes = 0;
    loop.error.clear(); loop.saved.clear(); loop.saved.reserve(156); loop.replaying = false;
    loop.active = true; loopingThread = true;
}
LoopReport EndLoop() {
    auto& state = Get(); std::lock_guard lock(state.mutex); auto& loop = state.loop;
    if (!loop.active || !loopingThread) throw std::runtime_error("Invalid native latent-loop End state");
    loop.active = false; loopingThread = false;
    if (!loop.error.empty()) throw std::runtime_error(loop.error + "; discard this command list");
    if (loop.bodies > 1 && (loop.saved.size() != Expected().size() || loop.saved.size() != 156 ||
        loop.replayed != (loop.vit ? 42u : (loop.bodies - 1) * 152)))
        throw std::runtime_error("Native latent loop incomplete; discard this command list");
    if (loop.vit) ValidateVitCounterIsolation(loop);
    return {loop.bodies, loop.replayed, loop.bytes};
}
void CancelLoop() noexcept {
    loopingThread = false;
    try {
        auto& state = Get(); std::lock_guard lock(state.mutex);
        state.loop.active = false; state.loop.replaying = false;
    } catch (...) { }
}
void BeginVitLoop(ID3D12GraphicsCommandList* commands, ID3D12Device* device, float alpha) {
    auto& state = Get(); std::lock_guard lock(state.mutex); auto& loop = state.loop;
    if (!state.installed || !commands || !device || loop.active || state.active || state.pending || state.bookendModule ||
        !std::isfinite(alpha) || alpha < -1.0f || alpha > 1.0f)
        throw std::runtime_error("Invalid native ViT-loop Begin state");
    VerifyRuntime(loop); InitCopy(loop, device); InitLatentMix(loop, device);
    loop.commands = commands; loop.bodies = 2; loop.replayed = 0; loop.bytes = 0;
    loop.vit = true; loop.alpha = alpha; loop.counterBegin = 0; loop.counterEnd = 0;
    loop.error.clear(); loop.saved.clear(); loop.saved.reserve(156); loop.replaying = false;
    loop.active = true; loopingThread = true;
}
LoopReport EndVitLoop() { return EndLoop(); }
void CancelVitLoop() noexcept { CancelLoop(); }
}
