// NrKernelLab: runs CUDA kernels through the driver's CUDA-on-D3D12 path (the one the NR runtime uses) from a small
// job script, without the NR runtime. Development tool for the NVFP4 kernels.
//
//   NrKernelLab <job.txt>
//
// Job lines (blank lines and lines starting with # are ignored):
//   buffer NAME BYTES [FILE]              GPU buffer, zero-filled or initialised from FILE (rest zero)
//   fill NAME FILE                        re-initialise a buffer from FILE (rest zero)
//   module NAME FILE                      CUDA module (cubin or PTX fatbin)
//   function NAME MODULE SYMBOL
//   params NAME FILE                      parameter blob for launches
//   patch PARAMS OFFSET BUFFER [ADD]      write (buffer address + ADD) as u64 at OFFSET of the blob
//   launch FUNCTION PARAMS GX GY GZ BX BY BZ SHARED
//   barrier                               UAV barrier: later work waits for earlier work
//   mark LABEL                            GPU timestamp
//   repeat N ... end                      record the enclosed launch/barrier/mark lines N times
//   run                                   execute what was recorded, wait, print the timestamps
//   dump NAME FILE [BYTES]                read a buffer back to FILE
//   sleep MS
// Timestamps print as one line: "marks <label>=<microseconds since first mark> ...".
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <nvapi.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace {
using Blob = std::vector<unsigned char>;
[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error(what); }
void Check(HRESULT result, const char* what) {
    if (FAILED(result)) { char text[160]; sprintf_s(text, "%s failed: 0x%08X", what, unsigned(result)); Fail(text); }
}
void NvCheck(NvAPI_Status status, const char* what) {
    if (status != NVAPI_OK) Fail(std::string(what) + " failed: NvAPI status " + std::to_string(int(status)));
}
Blob ReadFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) Fail("Cannot open " + path);
    Blob bytes(size_t(file.tellg())); file.seekg(0);
    if (!bytes.empty() && !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()))) Fail("Cannot read " + path);
    return bytes;
}
uint64_t Number(const std::string& text) {
    size_t used = 0; const auto value = std::stoull(text, &used, 0);
    if (used != text.size()) Fail("Bad number " + text);
    return value;
}

struct Command { enum Kind { Launch, Barrier, Mark } kind; std::string a, b; NVAPI_DIM3 grid{}, block{}; unsigned shared = 0; };

struct Lab {
    using Query = void* (__cdecl*)(unsigned);
    decltype(&NvAPI_D3D12_CreateCuModule) createModule = nullptr;
    decltype(&NvAPI_D3D12_CreateCuFunction) createFunction = nullptr;
    decltype(&NvAPI_D3D12_LaunchCuKernelChain) launch = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuModule) destroyModule = nullptr;
    decltype(&NvAPI_D3D12_DestroyCuFunction) destroyFunction = nullptr;
    ComPtr<ID3D12Device> device; ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list; ComPtr<ID3D12Fence> fence; UINT64 fenceValue = 0; HANDLE event = nullptr;
    ComPtr<ID3D12QueryHeap> queryHeap; ComPtr<ID3D12Resource> queryReadback; UINT64 frequency = 0;
    static constexpr unsigned kQueries = 16384;
    struct Buffer { ComPtr<ID3D12Resource> gpu; uint64_t bytes = 0; };
    std::map<std::string, Buffer> buffers; std::map<std::string, NVDX_ObjectHandle> modules, functions;
    std::map<std::string, Blob> params; std::vector<std::string> marks; bool recorded = false;

    void Init() {
        const HMODULE nvapi = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!nvapi) Fail("Cannot load nvapi64.dll");
        const auto query = reinterpret_cast<Query>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
        if (!query) Fail("nvapi_QueryInterface missing");
        const auto initialize = reinterpret_cast<NvAPI_Status(__cdecl*)()>(query(0x0150E828));
        if (!initialize) Fail("NvAPI_Initialize missing");
        NvCheck(initialize(), "NvAPI_Initialize");
        createModule = reinterpret_cast<decltype(createModule)>(query(0xad1a677d));
        createFunction = reinterpret_cast<decltype(createFunction)>(query(0xe2436e22));
        launch = reinterpret_cast<decltype(launch)>(query(0x24973538));
        destroyModule = reinterpret_cast<decltype(destroyModule)>(query(0x41c65285));
        destroyFunction = reinterpret_cast<decltype(destroyFunction)>(query(0xdf295ea6));
        if (!createModule || !createFunction || !launch || !destroyModule || !destroyFunction) Fail("NvAPI CUDA entry points missing");

        ComPtr<IDXGIFactory6> factory; Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        ComPtr<IDXGIAdapter1> adapter, chosen;
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{}; adapter->GetDesc1(&desc);
            if (desc.VendorId == 0x10DE && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { chosen = adapter; break; }
        }
        if (!chosen) Fail("No NVIDIA adapter");
        Check(D3D12CreateDevice(chosen.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "D3D12CreateDevice");
        D3D12_COMMAND_QUEUE_DESC desc{}; desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        Check(device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
        Check(device->CreateCommandList(0, desc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        D3D12_QUERY_HEAP_DESC heap{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kQueries, 0};
        Check(device->CreateQueryHeap(&heap, IID_PPV_ARGS(&queryHeap)), "CreateQueryHeap");
        queryReadback = Resource(uint64_t(kQueries) * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        Check(queue->GetTimestampFrequency(&frequency), "GetTimestampFrequency");
    }
    ComPtr<ID3D12Resource> Resource(uint64_t bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
        D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = bytes; desc.Height = 1;
        desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (type == D3D12_HEAP_TYPE_DEFAULT) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = type; ComPtr<ID3D12Resource> out;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out)), "CreateCommittedResource");
        return out;
    }
    void Transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to}; list->ResourceBarrier(1, &barrier);
    }
    void Execute() {
        Check(list->Close(), "Close command list");
        ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
        Check(queue->Signal(fence.Get(), ++fenceValue), "Signal");
        Check(fence->SetEventOnCompletion(fenceValue, event), "SetEventOnCompletion");
        if (WaitForSingleObject(event, 20000) != WAIT_OBJECT_0) {
            char text[160]; sprintf_s(text, "GPU work did not finish within 20 s (device removed reason 0x%08X)", unsigned(device->GetDeviceRemovedReason()));
            Fail(text);
        }
        Check(device->GetDeviceRemovedReason(), "Device removed");
        Check(allocator->Reset(), "Allocator reset"); Check(list->Reset(allocator.Get(), nullptr), "List reset");
    }
    Buffer& Named(const std::string& name) { const auto found = buffers.find(name); if (found == buffers.end()) Fail("Unknown buffer " + name); return found->second; }
    void Fill(Buffer& buffer, const std::string& file) {
        Blob bytes(size_t(buffer.bytes), 0);
        if (!file.empty()) { const auto data = ReadFile(file); if (data.size() > bytes.size()) Fail("File larger than buffer: " + file); memcpy(bytes.data(), data.data(), data.size()); }
        const auto upload = Resource(buffer.bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr; Check(upload->Map(0, nullptr, &mapped), "Map upload"); memcpy(mapped, bytes.data(), bytes.size()); upload->Unmap(0, nullptr);
        Transition(buffer.gpu.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(buffer.gpu.Get(), 0, upload.Get(), 0, buffer.bytes);
        Transition(buffer.gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Execute();
    }
    void Dump(Buffer& buffer, const std::string& file, uint64_t bytes) {
        bytes = bytes ? std::min(bytes, buffer.bytes) : buffer.bytes;
        const auto readback = Resource(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        Transition(buffer.gpu.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), 0, buffer.gpu.Get(), 0, bytes);
        Transition(buffer.gpu.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Execute();
        void* mapped = nullptr; D3D12_RANGE range{0, SIZE_T(bytes)}; Check(readback->Map(0, &range, &mapped), "Map readback");
        std::ofstream out(file, std::ios::binary); out.write(static_cast<const char*>(mapped), std::streamsize(bytes));
        readback->Unmap(0, nullptr); if (!out) Fail("Cannot write " + file);
    }
    void Record(const Command& command) {
        recorded = true;
        if (command.kind == Command::Barrier) { D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; list->ResourceBarrier(1, &barrier); return; }
        if (command.kind == Command::Mark) {
            if (marks.size() >= kQueries) Fail("Too many marks");
            list->EndQuery(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, UINT(marks.size())); marks.push_back(command.a); return;
        }
        const auto function = functions.find(command.a); const auto blob = params.find(command.b);
        if (function == functions.end() || blob == params.end()) Fail("Unknown function or params in launch " + command.a + " " + command.b);
        NVAPI_CU_KERNEL_LAUNCH_PARAMS kernel{}; kernel.hFunction = function->second; kernel.gridDim = command.grid; kernel.blockDim = command.block;
        kernel.dynSharedMemBytes = command.shared; kernel.pParams = blob->second.data(); kernel.paramSize = NvU32(blob->second.size());
        NvCheck(launch(list.Get(), &kernel, 1), "LaunchCuKernelChain");
    }
    void Run() {
        const auto count = UINT(marks.size());
        if (count) list->ResolveQueryData(queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, count, queryReadback.Get(), 0);
        Execute(); recorded = false;
        if (!count) { printf("marks\n"); fflush(stdout); return; }
        void* mapped = nullptr; D3D12_RANGE range{0, SIZE_T(count) * 8}; Check(queryReadback->Map(0, &range, &mapped), "Map timestamps");
        const auto* ticks = static_cast<const UINT64*>(mapped); printf("marks");
        for (UINT i = 0; i < count; ++i) printf(" %s=%.3f", marks[i].c_str(), double(ticks[i] - ticks[0]) * 1e6 / double(frequency));
        printf("\n"); fflush(stdout); queryReadback->Unmap(0, nullptr); marks.clear();
    }
    void Shutdown() {
        if (!device) return;
        for (auto& entry : functions) destroyFunction(device.Get(), entry.second);
        for (auto& entry : modules) destroyModule(device.Get(), entry.second);
        functions.clear(); modules.clear(); buffers.clear();
    }
};

Command ParseRecorded(const std::string& key, std::istringstream& in) {
    Command command{};
    if (key == "barrier") command.kind = Command::Barrier;
    else if (key == "mark") { command.kind = Command::Mark; in >> command.a; }
    else {
        command.kind = Command::Launch; in >> command.a >> command.b >> command.grid.x >> command.grid.y >> command.grid.z
            >> command.block.x >> command.block.y >> command.block.z >> command.shared;
        if (!in || !command.grid.x || !command.grid.y || !command.grid.z || !command.block.x || !command.block.y || !command.block.z) Fail("Bad launch line");
    }
    return command;
}

void RunJob(Lab& lab, const std::string& path) {
    std::ifstream job(path); if (!job) Fail("Cannot open job " + path);
    std::string line; std::vector<Command> block; unsigned repeat = 0; bool inRepeat = false; unsigned lineNumber = 0;
    while (std::getline(job, line)) {
        ++lineNumber; std::istringstream in(line); std::string key; in >> key;
        if (key.empty() || key[0] == '#') continue;
        try {
            if (key == "launch" || key == "barrier" || key == "mark") { const auto command = ParseRecorded(key, in); if (inRepeat) block.push_back(command); else lab.Record(command); }
            else if (key == "repeat") { if (inRepeat) Fail("Nested repeat"); in >> repeat; inRepeat = true; block.clear(); }
            else if (key == "end") { if (!inRepeat) Fail("end without repeat"); for (unsigned i = 0; i < repeat; ++i) for (const auto& command : block) lab.Record(command); inRepeat = false; }
            else if (inRepeat) Fail("Only launch, barrier and mark may be repeated");
            else if (key == "run") lab.Run();
            else if (lab.recorded && (key == "buffer" || key == "fill" || key == "dump")) Fail("Recorded work must be run before " + key);
            else if (key == "buffer") { std::string name, bytes, file; in >> name >> bytes; in >> file; Lab::Buffer buffer; buffer.bytes = std::max<uint64_t>(Number(bytes), 256);
                buffer.gpu = lab.Resource(buffer.bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); lab.Fill(buffer, file); lab.buffers[name] = buffer; }
            else if (key == "fill") { std::string name, file; in >> name >> file; lab.Fill(lab.Named(name), file); }
            else if (key == "module") { std::string name, file; in >> name >> file; const auto bytes = ReadFile(file); NVDX_ObjectHandle module = nullptr;
                NvCheck(lab.createModule(lab.device.Get(), bytes.data(), NvU32(bytes.size()), &module), "CreateCuModule"); lab.modules[name] = module; printf("module %s loaded (%zu bytes)\n", name.c_str(), bytes.size()); fflush(stdout); }
            else if (key == "function") { std::string name, module, symbol; in >> name >> module >> symbol; const auto found = lab.modules.find(module); if (found == lab.modules.end()) Fail("Unknown module " + module);
                NVDX_ObjectHandle function = nullptr; NvCheck(lab.createFunction(lab.device.Get(), found->second, symbol.c_str(), &function), "CreateCuFunction"); lab.functions[name] = function; }
            else if (key == "params") { std::string name, file; in >> name >> file; lab.params[name] = ReadFile(file); }
            else if (key == "patch") { std::string name, offset, buffer, add; in >> name >> offset >> buffer; in >> add; const auto found = lab.params.find(name); if (found == lab.params.end()) Fail("Unknown params " + name);
                const auto at = size_t(Number(offset)); if (at + 8 > found->second.size()) Fail("Patch outside params"); const uint64_t value = lab.Named(buffer).gpu->GetGPUVirtualAddress() + (add.empty() ? 0 : Number(add)); memcpy(found->second.data() + at, &value, 8); }
            else if (key == "dump") { std::string name, file, bytes; in >> name >> file; in >> bytes; lab.Dump(lab.Named(name), file, bytes.empty() ? 0 : Number(bytes)); }
            else if (key == "sleep") { unsigned ms = 0; in >> ms; Sleep(ms); }
            else Fail("Unknown job command " + key);
        } catch (const std::exception& error) { Fail("job line " + std::to_string(lineNumber) + ": " + error.what()); }
    }
    if (lab.recorded) Fail("Job ended with recorded work that was never run");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: NrKernelLab <job.txt>\n"); return 2; }
    Lab lab; int code = 0;
    try { lab.Init(); RunJob(lab, argv[1]); printf("ok\n"); }
    catch (const std::exception& error) { fprintf(stderr, "NrKernelLab: %s\n", error.what()); code = 1; }
    lab.Shutdown(); fflush(stdout);
    return code;
}
