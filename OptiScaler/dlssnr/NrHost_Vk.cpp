#define VK_ENABLE_BETA_EXTENSIONS // Query the CUDA SM version for the packaged cubins.
#include "DlssNr_Image_Vk.h"
#include "NrHost_Vk.h"
#include "NrNvfp4.h"
#include "../../standalone/NrCapture/nvfp4/NrHostPlan.h"
#include "../../standalone/NrCapture/nvfp4/NrPassRuntime.h"
#include "../../standalone/NrCapture/nvfp4/NrFourBitPlan.h"
#include "../../standalone/NrCapture/nvfp4/NrLayerDevice.h"

namespace DlssNr
{
namespace
{
using namespace NrHostPlan;
void Check(VkResult result, const char* operation)
{
    if (result != VK_SUCCESS)
        Fail(std::string(operation) + " failed (VkResult " + std::to_string(result) + ")");
}
template <class T> T Proc(VkDevice device, const char* name)
{
    auto proc = vkGetDeviceProcAddr(device, name);
    if (!proc)
        Fail(std::string("missing Vulkan device function ") + name + "; enable NR before launching the game");
    return reinterpret_cast<T>(proc);
}
struct Allocation
{
    VkDevice device;
    VkBuffer buffer = {};
    VkDeviceMemory memory = {};
    bool mapped = false;
    explicit Allocation(VkDevice d) : device(d) {}
    ~Allocation()
    {
        if (mapped)
            vkUnmapMemory(device, memory);
        if (buffer)
            vkDestroyBuffer(device, buffer, nullptr);
        if (memory)
            vkFreeMemory(device, memory, nullptr);
    }
};
struct Launch : LaunchFields
{
    NVAPI_CU_KERNEL_LAUNCH_PARAMS dispatch {};
    std::unique_ptr<NrFourBitPlan::Operation> fourBit;
};
} // namespace

struct VulkanHost::Impl : NrLayer::Context
{
    VkPhysicalDevice physical;
    VkDevice device;
    VkCommandBuffer cmd {};
    PFN_vkCreateCuModuleNVX createModule;
    PFN_vkCreateCuFunctionNVX createFunction;
    PFN_vkDestroyCuModuleNVX destroyModule;
    PFN_vkDestroyCuFunctionNVX destroyFunction;
    PFN_vkCmdCuLaunchKernelNVX launch;
    PFN_vkGetImageViewHandle64NVX viewHandle;
    PFN_vkGetBufferDeviceAddressKHR bufferAddress;
    std::vector<VkCuModuleNVX> modules;
    std::vector<VkCuFunctionNVX> functions;
    ModelPlan<Launch> plan;
    std::vector<Launch*> order;
    Geometry geometry;
    NrLayer::Memory records, arena;
    VkSampler point {}, linear {};
    ImageVk history, network, inputCopy;
    uint32_t sinceReset = 0;
    bool useFourBit = false;

    Impl(VkPhysicalDevice pd, VkDevice d, unsigned width, unsigned height)
        : physical(pd), device(d), geometry(width, height)
    {
        createModule = Proc<PFN_vkCreateCuModuleNVX>(d, "vkCreateCuModuleNVX");
        createFunction = Proc<PFN_vkCreateCuFunctionNVX>(d, "vkCreateCuFunctionNVX");
        destroyModule = Proc<PFN_vkDestroyCuModuleNVX>(d, "vkDestroyCuModuleNVX");
        destroyFunction = Proc<PFN_vkDestroyCuFunctionNVX>(d, "vkDestroyCuFunctionNVX");
        launch = Proc<PFN_vkCmdCuLaunchKernelNVX>(d, "vkCmdCuLaunchKernelNVX");
        viewHandle = Proc<PFN_vkGetImageViewHandle64NVX>(d, "vkGetImageViewHandle64NVX");
        bufferAddress = Proc<PFN_vkGetBufferDeviceAddressKHR>(d, "vkGetBufferDeviceAddressKHR");
    }
    ~Impl()
    {
        for (auto f : functions)
            destroyFunction(device, f, nullptr);
        for (auto m : modules)
            destroyModule(device, m, nullptr);
        for (auto* i : { &inputCopy, &network, &history })
            i->Destroy(device);
        if (linear)
            vkDestroySampler(device, linear, nullptr);
        if (point)
            vkDestroySampler(device, point, nullptr);
    }

    NrLayer::Memory Allocate(uint64_t bytes, bool mapped)
    {
        auto storage = std::make_shared<Allocation>(device);
        VkBufferCreateInfo ci { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        ci.size = bytes;
        ci.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        Check(vkCreateBuffer(device, &ci, nullptr, &storage->buffer), "create Vulkan model buffer");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, storage->buffer, &req);
        VkMemoryAllocateFlagsInfo flags { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags };
        ai.allocationSize = req.size;
        VkMemoryPropertyFlags host =
            mapped ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0;
        ai.memoryTypeIndex = FindVkMemoryType(physical, req.memoryTypeBits, host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto resultCode = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        if (ai.memoryTypeIndex != UINT32_MAX)
            resultCode = vkAllocateMemory(device, &ai, nullptr, &storage->memory);
        // GPUs with a small CPU-visible VRAM heap may need system memory for mapped weights.
        if (mapped && resultCode == VK_ERROR_OUT_OF_DEVICE_MEMORY)
        {
            const auto remainingTypes = ai.memoryTypeIndex == UINT32_MAX
                                            ? req.memoryTypeBits
                                            : req.memoryTypeBits & ~(1u << ai.memoryTypeIndex);
            ai.memoryTypeIndex = FindVkMemoryType(physical, remainingTypes, host);
            if (ai.memoryTypeIndex != UINT32_MAX)
                resultCode = vkAllocateMemory(device, &ai, nullptr, &storage->memory);
        }
        Check(resultCode, "allocate Vulkan model memory");
        Check(vkBindBufferMemory(device, storage->buffer, storage->memory, 0), "bind Vulkan model buffer");
        VkBufferDeviceAddressInfo bi { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        bi.buffer = storage->buffer;
        NrLayer::Memory result { storage, bufferAddress(device, &bi) };
        if (!result.address)
            Fail("Vulkan buffer device address is unavailable");
        if (mapped)
        {
            Check(vkMapMemory(device, storage->memory, 0, bytes, 0, &result.mapped), "map Vulkan model memory");
            storage->mapped = true;
        }
        return result;
    }
    NrLayer::Memory AllocateMemory(uint64_t bytes, bool upload) override
    {
        // Layer counters start at zero, including those allocated during an evaluation.
        auto memory = Allocate(bytes, true);
        if (!upload)
            memset(memory.mapped, 0, size_t(bytes));
        return memory;
    }
    void Barrier() override
    {
        VkMemoryBarrier barrier { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
    }
    static Impl& Active() { return *static_cast<Impl*>(NrLayer::active); }
    static NvAPI_Status __cdecl Module(ID3D12Device*, const void* data, NvU32 bytes, NVDX_ObjectHandle* out)
    {
        auto& self = Active();
        VkCuModuleTexturingModeCreateInfoNVX mode { VK_STRUCTURE_TYPE_CU_MODULE_TEXTURING_MODE_CREATE_INFO_NVX };
        mode.use64bitTexturing = VK_TRUE;
        VkCuModuleCreateInfoNVX ci { VK_STRUCTURE_TYPE_CU_MODULE_CREATE_INFO_NVX, &mode, bytes, data };
        VkCuModuleNVX module {};
        if (self.createModule(self.device, &ci, nullptr, &module) != VK_SUCCESS)
            return NVAPI_ERROR;
        self.modules.push_back(module);
        *out = reinterpret_cast<NVDX_ObjectHandle>(module);
        return NVAPI_OK;
    }
    static NvAPI_Status __cdecl Function(ID3D12Device*, NVDX_ObjectHandle module, const char* entry,
                                         NVDX_ObjectHandle* out)
    {
        auto& self = Active();
        VkCuFunctionCreateInfoNVX ci { VK_STRUCTURE_TYPE_CU_FUNCTION_CREATE_INFO_NVX };
        ci.module = reinterpret_cast<VkCuModuleNVX>(module);
        ci.pName = entry;
        VkCuFunctionNVX function {};
        if (self.createFunction(self.device, &ci, nullptr, &function) != VK_SUCCESS)
            return NVAPI_ERROR;
        self.functions.push_back(function);
        *out = reinterpret_cast<NVDX_ObjectHandle>(function);
        return NVAPI_OK;
    }
    static NvAPI_Status __cdecl Submit(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels,
                                       NvU32 count)
    {
        auto& self = Active();
        for (unsigned j = 0; j < count; ++j)
        {
            const auto& p = kernels[j];
            size_t size = p.paramSize;
            // CUDA Driver API packed-argument convention used by VK_NVX_binary_import.
            const void* extras[] = { reinterpret_cast<void*>(1), p.pParams, reinterpret_cast<void*>(2), &size,
                                     nullptr };
            VkCuLaunchInfoNVX info { VK_STRUCTURE_TYPE_CU_LAUNCH_INFO_NVX };
            info.function = reinterpret_cast<VkCuFunctionNVX>(p.hFunction);
            info.gridDimX = p.gridDim.x;
            info.gridDimY = p.gridDim.y;
            info.gridDimZ = p.gridDim.z;
            info.blockDimX = p.blockDim.x;
            info.blockDimY = p.blockDim.y;
            info.blockDimZ = p.blockDim.z;
            info.sharedMemBytes = p.dynSharedMemBytes;
            info.extraCount = std::size(extras);
            info.pExtras = extras;
            self.launch(self.cmd, &info);
        }
        return NVAPI_OK;
    }
    void Init(const std::filesystem::path& assets, unsigned preset)
    {
        NrLayer::Scope scope(this);
        useFourBit = !Nvfp4::IsInt8(preset);
        VkPhysicalDeviceCudaKernelLaunchPropertiesNV arch {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV
        };
        VkPhysicalDeviceProperties2 props { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &arch };
        vkGetPhysicalDeviceProperties2(physical, &props);
        std::string code;
        if (arch.computeCapabilityMajor == 12 && arch.computeCapabilityMinor == 0)
            code = "sm120";
        else if (!useFourBit && arch.computeCapabilityMajor == 8 &&
                 (arch.computeCapabilityMinor == 6 || arch.computeCapabilityMinor == 9))
            code = "sm86";
        else
            Fail(useFourBit ? "NVFP4 requires an RTX 50 series GPU" : "INT8 requires an RTX 30, 40 or 50 series GPU");
        auto folder = assets / (useFourBit ? "host4" : "int8");
        auto set = useFourBit ? Nvfp4::WeightSet(preset) : Nvfp4::Int8Set(preset);
        const NrFourBitPlan::Driver driver { Module, Function, nullptr, nullptr, Submit };
        if (useFourBit)
        {
            NrFfnV2::Options ffnOptions;
            ffnOptions.modelFolders = ffnOptions.trained = true;
            ffnOptions.minTokens = 0;
            ffnOptions.assets = assets;
            NrFfnV2::Enable(ffnOptions);
            NrSwin4::Options swinOptions;
            swinOptions.assets = assets / "swin4";
            swinOptions.widths = 12;
            swinOptions.set = 1;
            NrSwin4::Enable(swinOptions);
            NrNet4::Options netOptions;
            netOptions.assets = assets;
            netOptions.family[NrNet4::Split].groups = 63;
            netOptions.family[NrNet4::Vit].groups = 7;
            netOptions.family[NrNet4::Split].trained = netOptions.family[NrNet4::Vit].trained = true;
            NrNet4::Enable(netOptions);
            std::string missing;
            auto weights = set ? set : NrSwin4::kDefaultSet;
            if (!NrSwin4::SetComplete(weights, &missing) || !NrNet4::SetComplete(weights, &missing) ||
                !NrFfnV2::SetComplete(weights, &missing))
                Fail("missing NVFP4 weights: " + missing);
        }
        plan = ReadPlan<Launch>(folder / (useFourBit ? "modelA_fp8.plan" : "modelA_i8.plan"));
        for (auto& op : plan.launches)
            if (op.layout == "any" || op.layout == "3109")
                order.push_back(&op);
        std::sort(order.begin(), order.end(), [](const auto* a, const auto* b) { return a->idx < b->idx; });
        if (order.size() != kSlots)
            Fail("incomplete Vulkan model plan");
        ArenaLayout layout;
        layout.Build(geometry, plan, order);
        arena = Allocate(layout.arenaBytes, false);
        RecordLayout index;
        if (!useFourBit)
        {
            auto stem = "int8_set" + std::to_string(set);
            auto file = folder / (stem + ".bin");
            auto blob = ReadAll(file);
            index.Read(folder / (stem + ".txt"), file, blob);
            records = Allocate(index.blendAt + 4608, true);
            memset(records.mapped, 0, size_t(index.blendAt + 4608));
            memcpy(records.mapped, blob.data(), blob.size());
            memcpy(static_cast<char*>(records.mapped) + index.blendAt, &kBlendWeight, 2);
        }
        Choices choices;
        ReadFolder(folder, code, choices);
        std::map<std::string, NVDX_ObjectHandle> loaded;
        int block = 0, expected = kFirst;
        for (auto* op : order)
        {
            if (op->idx != expected++)
                Fail("missing or duplicate model launch");
            auto [grid, threads] = Grid(op->name, geometry, op->sx, op->sy);
            auto& p = op->dispatch;
            p.gridDim = { grid.x, grid.y, grid.z };
            p.blockDim = { threads.x, threads.y, threads.z };
            p.paramSize = op->params;
            uint64_t record = 0, blend = records.address + index.blendAt;
            if (useFourBit)
            {
                auto& layer = *(op->fourBit = std::make_unique<NrFourBitPlan::Operation>());
                layer.driver = Submit;
                layer.swinSlot = op->name.starts_with("cc_tinlayout_fused_swin_") ? block++ : -1;
                layer.weightSet = set ? set : NrSwin4::kDefaultSet;
                NrNet4::BindNative(layer, driver, nullptr, op->name, op->idx);
                NrSwin4::BindNative(layer, driver, nullptr, op->name, set, layer.swinSlot);
                NrFfnV2::BindNative(layer, driver, nullptr, op->name, set, layer.swinSlot);
                record = layer.record;
                blend = layer.blend;
                layer.base = p;
            }
            else if (std::any_of(op->fields.begin(), op->fields.end(),
                                 [](const auto& f) { return f.code == Code::Record; }))
            {
                auto found = index.offsets.find(op->idx);
                if (found == index.offsets.end() || found->second.second != op->name)
                    Fail("missing record for " + op->name);
                record = records.address + found->second.first;
            }
            BindParameters(*op, geometry, layout, arena.address, record, blend);
            if (op->fourBit && !choices.contains(op->idx))
                continue;
            auto& base = op->fourBit ? op->fourBit->base : p;
            const auto& options = choices.at(op->idx);
            auto pick =
                std::find_if(options.begin(), options.end(), [&](const auto& c) { return c.params == op->params; });
            if (pick == options.end())
                pick = std::find_if(options.begin(), options.end(), [](const auto& c) { return !c.params; });
            if (pick == options.end() || (!pick->name.empty() && pick->name != op->name) || !pick->prepare.empty())
                Fail("incompatible Vulkan kernel manifest for " + op->name);
            auto file = ModuleFile(*pick);
            auto& module = loaded[file];
            if (!module)
            {
                auto blob = ReadAll(folder / file);
                if (Module(nullptr, blob.data(), static_cast<NvU32>(blob.size()), &module) != NVAPI_OK)
                    Fail("loading " + file);
            }
            if (Function(nullptr, module, pick->entry.c_str(), &base.hFunction) != NVAPI_OK)
                Fail("loading " + pick->entry);
            base.gridDim = { Apply(pick->grid[0], grid.x), Apply(pick->grid[1], grid.y), Apply(pick->grid[2], grid.z) };
            base.blockDim = { Apply(pick->block[0], threads.x), Apply(pick->block[1], threads.y),
                              Apply(pick->block[2], threads.z) };
            base.dynSharedMemBytes = pick->shared;
        }
        VkSamplerCreateInfo si { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        Check(vkCreateSampler(device, &si, nullptr, &point), "create Vulkan point sampler");
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        Check(vkCreateSampler(device, &si, nullptr, &linear), "create Vulkan history sampler");
    }
    void Image(ImageVk& image)
    {
        if (image.Valid())
            return;
        if (!image.Ensure(device, physical, unsigned(geometry.W), unsigned(geometry.H), VK_FORMAT_R16G16B16A16_SFLOAT))
            Fail("could not allocate Vulkan model image");
    }
    void BeginImage(ImageVk& image)
    {
        if (image.layout == VK_IMAGE_LAYOUT_GENERAL)
            return;
        VkImageMemoryBarrier b { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image.info.Image;
        b.subresourceRange = image.info.SubresourceRange;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
        image.layout = VK_IMAGE_LAYOUT_GENERAL;
    }
    uint64_t Handle(VkImageView image, VkSampler sampler = {}) const
    {
        VkImageViewHandleInfoNVX info { VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX };
        info.imageView = image;
        info.sampler = sampler;
        info.descriptorType = sampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        return viewHandle(device, &info);
    }
};

VulkanHost::VulkanHost(std::unique_ptr<Impl> state) : impl(std::move(state)) {}
VulkanHost::~VulkanHost() = default;
std::unique_ptr<VulkanHost> VulkanHost::Create(VkPhysicalDevice pd, VkDevice d, const std::filesystem::path& assets,
                                               unsigned preset, unsigned width, unsigned height)
{
    if (!Nvfp4::UsesNativeHost(preset) || width < 64 || height < 64 || width > 16384 || height > 16384)
        Fail("unsupported Vulkan NR model or frame dimensions");
    auto state = std::make_unique<Impl>(pd, d, width, height);
    state->Init(assets, preset);
    return std::unique_ptr<VulkanHost>(new VulkanHost(std::move(state)));
}
void VulkanHost::Evaluate(VkCommandBuffer cmd, VkImageView colour, VkImageView motion, VkImageView output,
                          const ModelSettings& settings, float mvX, float mvY, bool reset)
{
    auto& m = *impl;
    NrLayer::Scope scope(&m);
    m.cmd = cmd;
    reset = reset || !m.sinceReset || !motion;
    const bool graded = settings.style != 0 || settings.intensity != 1.f;
    m.Image(m.history);
    if (graded)
    {
        m.Image(m.network);
        m.Image(m.inputCopy);
    }
    m.BeginImage(m.history);
    if (graded)
    {
        m.BeginImage(m.network);
        m.BeginImage(m.inputCopy);
    }
    FrameHandles h {};
    h.colourTex = m.Handle(colour, m.point);
    h.outputSurf = m.Handle(output);
    if (!reset)
        h.motionTex = m.Handle(motion, m.point);
    h.historyTex = m.Handle(m.history.info.ImageView, m.linear);
    h.historySurf = m.Handle(m.history.info.ImageView);
    if (graded)
    {
        h.netoutTex = m.Handle(m.network.info.ImageView, m.point);
        h.netoutSurf = m.Handle(m.network.info.ImageView);
        h.inputcopyTex = m.Handle(m.inputCopy.info.ImageView, m.point);
        h.inputcopySurf = m.Handle(m.inputCopy.info.ImageView);
    }
    else
        h.outputTex = m.Handle(output, m.point);
    NrHost::Settings tuning { settings.style,
                              settings.intensity,
                              settings.localStructure,
                              settings.localTone,
                              settings.skinStructure,
                              settings.autoMask,
                              mvX,
                              mvY };
    const auto before = m.useFourBit ? NrPassRuntime::ReadWork() : NrPassRuntime::Work {};
    for (auto* op : m.order)
    {
        if (op->styles == "graded" && !graded)
            continue;
        if (op->barrier)
            m.Barrier();
        FrameParameters(*op, m.geometry, m.plan.grades, tuning, h, reset, graded, m.sinceReset);
        auto dispatch = op->dispatch;
        dispatch.pParams = op->parameters.data();
        if (op->fourBit)
        {
            NrFourBitPlan::Scope layer(*op->fourBit);
            if (NrNet4::SubmitNative(nullptr, &dispatch, 1) != NVAPI_OK)
                Fail("Vulkan NVFP4 " + op->name + ": " + op->fourBit->failure);
        }
        else
            Impl::Submit(nullptr, &dispatch, 1);
    }
    if (m.useFourBit)
    {
        NrSwin4::EndEvaluation(nullptr);
        NrNet4::EndEvaluation(nullptr);
        NrFfnV2::EndEvaluation(nullptr);
        const auto after = NrPassRuntime::ReadWork();
        if (!after.swinActive || after.swin <= before.swin || after.ffn <= before.ffn || after.net <= before.net ||
            after.swinFaults != before.swinFaults || after.netFaults != before.netFaults ||
            after.ffnFaults != before.ffnFaults)
            Fail("the Vulkan NVFP4 layers did not complete the selected model");
    }
    m.Barrier();
    m.sinceReset = reset ? 1 : m.sinceReset + 1;
}
} // namespace DlssNr
