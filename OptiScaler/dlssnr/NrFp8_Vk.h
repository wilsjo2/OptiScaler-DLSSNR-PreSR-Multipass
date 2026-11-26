#pragma once

#include <vulkan/vulkan.h>
#include <filesystem>
#include <string>

// FP8 pass-in-one models on native Vulkan. NVIDIA's NR runtime records Model A with its own FP8 kernels through
// vkCmdCuLaunchKernelNVX. For an evaluation armed with a set, every launch that reads a weight record is checked against
// the set's manifest (kernel name, parameter block size) and launched with our record instead; standalone/NrCapture/
// nvfp4/NrFp8.h does the same between the runtime and NvAPI on D3D12.
//
// A launch is never dropped: the runtime's kernels wait for their predecessors without a time limit. When anything does
// not match, the rest of the evaluation runs as the runtime recorded it and End() reports the reason, so the caller
// must not show that evaluation's picture.
namespace DlssNr::Fp8Vk
{
// After vkCreateDevice on a device with binary import and buffer device addresses: hooks its CUDA entry points (one set
// of hooks per process; a later device is covered when it resolves to the same entry points).
void HookDevice(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);
void Unhook();
// Frees the weights held for `device` and forgets its kernels. The device no longer runs work recorded with them.
void ReleaseDevice(VkDevice device);
// Whether `set` can run on `device` (reads the manifests in `assets` the first time); otherwise why not.
bool Ready(VkDevice device, unsigned set, const std::filesystem::path& assets, std::string& why);
// Arms the next evaluation recorded on `cmd` with `set`, loading the set's weights on first use. Throws on failure.
void Begin(VkCommandBuffer cmd, VkDevice device, unsigned set);
// Ends the armed evaluation on `cmd`: true when it ran wholly on the set's records; otherwise `why` says what differed.
bool End(VkCommandBuffer cmd, std::string& why);
} // namespace DlssNr::Fp8Vk
