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

// XeFG owned application-frame handoff (nr-xefg-088-release, todo 10). The Present-site
// wiring in framegen/xefg/XeFG_Dx12.cpp drives these on the XeFG-retained application
// queue (never State.currentCommandQueue): the capture facts are queried first, the
// composition runs only after the handoff core (dlssnr/DlssNr_XeFGHandoff.h) accepted
// the frame. Swapchain/buffer/colour-space queries must happen before these calls -
// they take the NR locks (see ApplyToFinishedPicture for the ordering rule).
struct XeFGCapture
{
    bool exists = false;    // a pending capture matches this finished picture
    bool submitted = false; // its producer command list was executed
    bool ready = false;     // same application queue, or its readiness fence completed
    bool sameQueue = false; // the capture's producer queue is the application queue
    uint64_t serial = 0;    // the pending capture's slot serial
};
bool XeFGPendingCapture(ID3D12Resource* picture, ID3D12CommandQueue* queue, DXGI_COLOR_SPACE_TYPE space,
                        XeFGCapture& facts);
bool ApplyXeFGPicture(ID3D12Resource* picture, ID3D12CommandQueue* queue, DXGI_COLOR_SPACE_TYPE space);
void XeFGCloseCaptures(uint64_t throughSerial);

std::string DeferredDlssStatus();
// Outside DllMain only. Returns false rather than releasing a runtime with unresolved owners/work.
bool Shutdown();
} // namespace DlssNr
