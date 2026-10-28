#pragma once

#include "DlssNr_Status.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <string>
#include <shaders/dlssnr/DlssNr_Common.h>
#include <nvsdk_ngx.h>

namespace DlssNr
{
inline constexpr unsigned int MaxPassCount = 30;
inline constexpr unsigned int DefaultMaxPassCount = 3;
inline constexpr GUID FinishedColorSpaceKey = {
    0x34a31e7b, 0x84c5, 0x44ef, { 0xa7, 0x4d, 0x6b, 0xd3, 0x60, 0x8c, 0xe5, 0x22 }
};

// Public callbacks route through registered upscaler owners. They do not own GPU state.
std::string FinishedPictureStatus();
bool WaitForFinishedPicture();
void FinishedPictureResetCommandList(ID3D12CommandList* cmd);
void FinishedPictureSubmitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists);
void ApplyToFinishedPicture(IDXGISwapChain* swapchain, ID3D12CommandQueue* queue);
void ApplyToStreamlinePicture(IDXGISwapChain* swapchain, ID3D12Resource* picture, ID3D12CommandQueue* queue);
void ApplyToFinishedPictureDx11(IDXGISwapChain* swapchain);
void FinishedPictureColorSpace(IDXGISwapChain* swapchain, DXGI_COLOR_SPACE_TYPE colorSpace);

std::string DeferredDlssStatus();
std::string DenoiseFirstStatus();
// Outside DllMain only. Returns false rather than releasing a runtime with unresolved owners/work.
bool Shutdown();
} // namespace DlssNr
