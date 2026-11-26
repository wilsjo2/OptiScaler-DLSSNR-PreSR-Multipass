#include "pch.h"

#include "NrFp8_Vk.h"
#include "NrHost_Vk.h"
#include "../../standalone/NrCapture/nvfp4/NrFp8Sets.h"

#include <detours/detours.h>

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace DlssNr::Fp8Vk
{
namespace
{
using namespace NrFp8Sets;
namespace fs = std::filesystem;

// CUDA Driver API packed-argument convention used by VK_NVX_binary_import.
constexpr uintptr_t kParamBuffer = 1, kParamSize = 2;

// A set's records on a device: read and uploaded when an evaluation first uses the set.
struct Weights
{
    bool ready = false;
    std::string failure;
    VulkanMemory memory;
    uint64_t blendAt = 0; // our copy of the history blend weight, after the records
};
struct DeviceState
{
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::array<Weights, kMaxSets> sets;
};
// An evaluation armed on a command buffer, counted from Model A's cc_cb_clear (launch 0).
struct Armed
{
    VkDevice device = VK_NULL_HANDLE;
    unsigned set = 0, index = 0, replaced = 0;
    bool failed = false;
    std::string reason;
};
struct Function
{
    VkDevice device = VK_NULL_HANDLE;
    std::string name;
};

struct State
{
    std::mutex mutex;
    bool hooked = false;
    // The entry points as the driver returned them; a later device is covered when it returns the same ones.
    void* launchTarget = nullptr;
    void* createTarget = nullptr;
    void* destroyTarget = nullptr;
    // Below the hooks once attached.
    PFN_vkCmdCuLaunchKernelNVX launch = nullptr;
    PFN_vkCreateCuFunctionNVX createFunction = nullptr;
    PFN_vkDestroyCuFunctionNVX destroyFunction = nullptr;
    std::unordered_map<VkDevice, DeviceState> devices;
    std::unordered_map<uint64_t, Function> functions;
    std::unordered_map<VkCommandBuffer, Armed> armed;
    std::atomic<unsigned> armedCount { 0 };
    fs::path assets;
    bool read = false;
    std::array<Set, kMaxSets> sets;
    bool loggedLaunch = false;
};
State& S()
{
    static State* state = new State;
    return *state;
}

uint64_t Key(VkCuFunctionNVX function) { return (uint64_t) function; }

Weights& LoadWeights(DeviceState& d, VkDevice device, const Set& set, unsigned n)
{
    Weights& w = d.sets[n];
    if (!w.failure.empty())
        throw std::runtime_error(w.failure);
    if (w.ready)
        return w;
    try
    {
        const auto blob = ReadFile(set.bin);
        if (blob.size() != set.binBytes)
            throw std::runtime_error(set.bin.string() + " changed after it was checked");
        w.blendAt = (blob.size() + 511) / 512 * 512;
        const uint64_t bytes = (w.blendAt + 512 + 4096 + 255) / 256 * 256; // kernels read ahead of a record's end
        w.memory = AllocateVulkanMemory(d.physical, device, bytes, true);
        auto* mapped = static_cast<unsigned char*>(w.memory.mapped);
        memset(mapped, 0, size_t(bytes));
        memcpy(mapped, blob.data(), blob.size());
        memcpy(mapped + w.blendAt, &kBlendWeight, 2);
        w.ready = true;
        LOG_INFO("DLSS-NR Vulkan: FP8 set {} loaded ({} MB)", n, blob.size() >> 20);
        return w;
    }
    catch (const std::exception& error)
    {
        w.memory = {};
        w.failure = error.what();
        throw;
    }
}

VkResult VKAPI_CALL hkCreateCuFunctionNVX(VkDevice device, const VkCuFunctionCreateInfoNVX* info,
                                          const VkAllocationCallbacks* allocator, VkCuFunctionNVX* function)
{
    auto& s = S();
    const auto result = s.createFunction(device, info, allocator, function);
    if (result == VK_SUCCESS && info && info->pName && function)
    {
        std::lock_guard guard(s.mutex);
        s.functions[Key(*function)] = { device, info->pName };
    }
    return result;
}

void VKAPI_CALL hkDestroyCuFunctionNVX(VkDevice device, VkCuFunctionNVX function, const VkAllocationCallbacks* allocator)
{
    auto& s = S();
    {
        std::lock_guard guard(s.mutex);
        s.functions.erase(Key(function));
    }
    s.destroyFunction(device, function, allocator);
}

void VKAPI_CALL hkCmdCuLaunchKernelNVX(VkCommandBuffer cmd, const VkCuLaunchInfoNVX* info)
{
    auto& s = S();
    if (s.armedCount.load(std::memory_order_acquire) == 0 || info == nullptr)
        return s.launch(cmd, info);

    std::unique_lock lock(s.mutex);
    const auto found = s.armed.find(cmd);
    if (found == s.armed.end())
    {
        lock.unlock();
        return s.launch(cmd, info);
    }

    Armed& a = found->second;
    const auto function = s.functions.find(Key(info->function));
    const std::string name = function != s.functions.end() ? function->second.name : "an unknown kernel";

    if (name == "cc_cb_clear")
    {
        if (a.index && !a.failed)
            a.failed = true, a.reason = "the runtime started Model A twice in one evaluation";
        a.index = 1;
        lock.unlock();
        return s.launch(cmd, info);
    }

    // Before the network, after it, or after something did not match: launched as the runtime recorded it.
    const Set& set = s.sets[a.set];
    const unsigned at = a.index;
    if (at)
        ++a.index;
    if (!at || a.failed || at >= set.launches.size() || !set.launches[at].present)
    {
        lock.unlock();
        return s.launch(cmd, info);
    }

    const Entry& e = set.launches[at];
    const void* block = nullptr;
    size_t size = 0;
    for (uint32_t i = 0; info->pExtras && i + 1 < info->extraCount; i += 2)
    {
        const auto key = reinterpret_cast<uintptr_t>(info->pExtras[i]);
        if (key == 0)
            break;
        if (key == kParamBuffer)
            block = info->pExtras[i + 1];
        else if (key == kParamSize && info->pExtras[i + 1])
            size = *static_cast<const size_t*>(info->pExtras[i + 1]);
    }

    if (!s.loggedLaunch)
    {
        s.loggedLaunch = true;
        LOG_INFO("DLSS-NR Vulkan: FP8 launch form: {} packed bytes, {} separate arguments, {} extras", size,
                 info->paramCount, info->extraCount);
    }

    std::string why;
    uint32_t recordAt = 0;
    if (!block || !size)
        why = info->paramCount ? "(" + name + ") passes its " + std::to_string(info->paramCount) +
                                     " arguments one by one, not as one parameter block"
                               : "(" + name + ") has no parameter block";
    else if (size > kMaxParams)
        why = "(" + name + ") has a " + std::to_string(size) + "-byte parameter block";
    else
        recordAt = RecordOffset(e, name, size, why);

    if (!recordAt)
    {
        a.failed = true;
        a.reason = "launch " + std::to_string(at) + " " + why;
        lock.unlock();
        return s.launch(cmd, info);
    }

    const Weights& w = s.devices[a.device].sets[a.set];
    unsigned char edited[kMaxParams];
    memcpy(edited, block, size);
    const uint64_t record = w.memory.address + e.offset;
    memcpy(edited + recordAt, &record, 8);
    if (e.blend)
    {
        uint64_t theirs = 0;
        memcpy(&theirs, edited + e.blend, 8);
        if (theirs)
        {
            const uint64_t ours = w.memory.address + w.blendAt;
            memcpy(edited + e.blend, &ours, 8);
        }
    }
    ++a.replaced;
    lock.unlock();

    // The parameters are consumed when the launch is recorded.
    size_t editedSize = size;
    const void* extras[] = { reinterpret_cast<void*>(kParamBuffer), edited, reinterpret_cast<void*>(kParamSize),
                             &editedSize, nullptr };
    VkCuLaunchInfoNVX own = *info;
    own.paramCount = 0;
    own.pParams = nullptr;
    own.extraCount = static_cast<uint32_t>(std::size(extras));
    own.pExtras = extras;
    s.launch(cmd, &own);
}
} // namespace

void HookDevice(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr)
{
    if (getDeviceProcAddr == nullptr || device == VK_NULL_HANDLE)
        return;

    auto* launch = reinterpret_cast<void*>(getDeviceProcAddr(device, "vkCmdCuLaunchKernelNVX"));
    auto* create = reinterpret_cast<void*>(getDeviceProcAddr(device, "vkCreateCuFunctionNVX"));
    auto* destroy = reinterpret_cast<void*>(getDeviceProcAddr(device, "vkDestroyCuFunctionNVX"));
    if (!launch || !create || !destroy)
    {
        LOG_WARN("DLSS-NR Vulkan: the device has no binary import entry points; FP8 pass-in-one models cannot run");
        return;
    }

    auto& s = S();
    std::lock_guard guard(s.mutex);
    if (!s.hooked)
    {
        s.launchTarget = launch;
        s.createTarget = create;
        s.destroyTarget = destroy;
        s.launch = reinterpret_cast<PFN_vkCmdCuLaunchKernelNVX>(launch);
        s.createFunction = reinterpret_cast<PFN_vkCreateCuFunctionNVX>(create);
        s.destroyFunction = reinterpret_cast<PFN_vkDestroyCuFunctionNVX>(destroy);

        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourAttach(&(PVOID&) s.launch, hkCmdCuLaunchKernelNVX);
        DetourAttach(&(PVOID&) s.createFunction, hkCreateCuFunctionNVX);
        DetourAttach(&(PVOID&) s.destroyFunction, hkDestroyCuFunctionNVX);
        if (const auto result = DetourTransactionCommit(); result != NO_ERROR)
        {
            LOG_ERROR("DLSS-NR Vulkan: could not hook the binary import entry points, error {:X}", result);
            s.launch = nullptr;
            s.createFunction = nullptr;
            s.destroyFunction = nullptr;
            return;
        }
        s.hooked = true;
    }
    else if (launch != s.launchTarget || create != s.createTarget || destroy != s.destroyTarget)
    {
        LOG_WARN("DLSS-NR Vulkan: this device resolves other binary import entry points; FP8 pass-in-one models "
                 "cannot run on it");
        return;
    }

    s.devices[device].physical = physical;
    LOG_INFO("DLSS-NR Vulkan: FP8 pass-in-one launches hooked for device {:X}", (uint64_t) device);
}

void Unhook()
{
    auto& s = S();
    std::lock_guard guard(s.mutex);
    if (!s.hooked)
        return;

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourDetach(&(PVOID&) s.launch, hkCmdCuLaunchKernelNVX);
    DetourDetach(&(PVOID&) s.createFunction, hkCreateCuFunctionNVX);
    DetourDetach(&(PVOID&) s.destroyFunction, hkDestroyCuFunctionNVX);
    if (DetourTransactionCommit() == NO_ERROR)
    {
        s.hooked = false;
        s.devices.clear();
        s.armed.clear();
        s.armedCount = 0;
    }
}

void ReleaseDevice(VkDevice device)
{
    auto& s = S();
    std::lock_guard guard(s.mutex);
    s.devices.erase(device);
    std::erase_if(s.functions, [device](const auto& entry) { return entry.second.device == device; });
    const auto removed = std::erase_if(s.armed, [device](const auto& entry) { return entry.second.device == device; });
    s.armedCount -= static_cast<unsigned>(removed);
}

bool Ready(VkDevice device, unsigned set, const fs::path& assets, std::string& why)
{
    auto& s = S();
    std::lock_guard guard(s.mutex);
    if (!s.read || s.assets != assets)
    {
        std::array<Set, kMaxSets> sets;
        unsigned found = 0;
        for (unsigned n = 1; n < kMaxSets; ++n)
        {
            std::error_code missing;
            if (!fs::exists(assets / ("fp8_set" + std::to_string(n) + ".txt"), missing))
                continue;
            try
            {
                sets[n] = ReadSet(assets, n);
                ++found;
            }
            catch (const std::exception& e)
            {
                sets[n].failure = e.what();
            }
        }
        for (auto& [handle, d] : s.devices)
            for (const auto& w : d.sets)
                if (w.ready && s.read && s.assets != assets)
                {
                    why = "the FP8 asset folder cannot change while a device holds its weights";
                    return false;
                }
        s.sets = std::move(sets);
        s.assets = assets;
        s.read = true;
        LOG_INFO("DLSS-NR Vulkan: {} FP8 set(s) in {}", found, assets.string());
    }

    if (!s.hooked || !s.devices.contains(device))
    {
        why = "Enable NR before Vulkan device creation. FP8 pass-in-one models need the device's kernel launches.";
        return false;
    }
    if (set == 0 || set >= kMaxSets || !s.sets[set].present)
    {
        why = "FP8 model unavailable: " + (set < kMaxSets && !s.sets[set].failure.empty()
                                               ? s.sets[set].failure
                                               : "fp8/fp8_set" + std::to_string(set) + ".txt and .bin");
        return false;
    }
    return true;
}

void Begin(VkCommandBuffer cmd, VkDevice device, unsigned set)
{
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const auto d = s.devices.find(device);
    if (!s.hooked || d == s.devices.end())
        throw std::runtime_error("the FP8 launch hooks do not cover this device");
    if (set == 0 || set >= kMaxSets || !s.sets[set].present)
        throw std::runtime_error("FP8 set " + std::to_string(set) + " is not available");
    LoadWeights(d->second, device, s.sets[set], set);
    Armed armed;
    armed.device = device;
    armed.set = set;
    if (s.armed.insert_or_assign(cmd, std::move(armed)).second)
        s.armedCount.fetch_add(1, std::memory_order_release);
}

bool End(VkCommandBuffer cmd, std::string& why)
{
    auto& s = S();
    std::lock_guard guard(s.mutex);
    const auto found = s.armed.find(cmd);
    if (found == s.armed.end())
    {
        why = "the FP8 evaluation was not armed";
        return false;
    }
    Armed a = std::move(found->second);
    s.armed.erase(found);
    s.armedCount.fetch_sub(1, std::memory_order_release);

    const Set& set = s.sets[a.set];
    if (!a.failed && !a.index)
        a.failed = true, a.reason = "the runtime recorded no Model A network for this pass";
    if (!a.failed && a.index < set.launches.size())
        a.failed = true, a.reason = "the evaluation ended after " + std::to_string(a.index) +
                                    " launches, Model A's network has " + std::to_string(set.launches.size());
    if (!a.failed && a.replaced != set.weighted)
        a.failed = true, a.reason = std::to_string(a.replaced) + " of the " + std::to_string(set.weighted) +
                                    " weight records were the set's";
    if (a.failed)
    {
        why = "FP8 set " + std::to_string(a.set) + ": " + a.reason;
        return false;
    }
    return true;
}
} // namespace DlssNr::Fp8Vk
