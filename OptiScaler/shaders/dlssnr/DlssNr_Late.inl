// Included inside namespace DlssNr. All entry points share g_nrMutex.
namespace Late
{
using Microsoft::WRL::ComPtr;
struct Slot
{
    ComPtr<ID3D12Resource> depth, motion, linear, encoded, residual;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ID3D12CommandList* producer = nullptr; // identity only; never dereferenced
    DlssNrFrameInfo frame {};
    uint64_t ready = 0, done = 0, serial = 0;
    bool pending = false, submitted = false, residualOnly = false, sceneLinear = true;
};
std::array<Slot, 4> slots;
ComPtr<ID3D12Device> device;
uint64_t serial = 0, successes = 0;
std::string status = "Waiting for a finished picture.";
bool reset = true;
std::atomic<bool> tracking { false };
inline constexpr GUID colorSpaceKey = { 0x34a31e7b, 0x84c5, 0x44ef, { 0xa7, 0x4d, 0x6b, 0xd3, 0x60, 0x8c, 0xe5, 0x22 } };

void Say(const char* message)
{
    if (status != message)
    {
        status = message;
        LOG_INFO("DLSS-NR finished picture: {}", message);
    }
}

bool Finished(const Slot& slot)
{
    return !slot.fence || (slot.fence->GetCompletedValue() != UINT64_MAX &&
                          slot.fence->GetCompletedValue() >= slot.done);
}

void Cancel()
{
    // Submitted copies may still be in flight; their fences still protect reuse.
    for (auto& slot : slots)
        if (slot.submitted)
            slot.pending = false;
    reset = true;
}

bool Clone(ComPtr<ID3D12Resource>& copy, ID3D12Resource* source)
{
    const auto want = source->GetDesc();
    if (want.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || want.SampleDesc.Count != 1 ||
        want.DepthOrArraySize != 1 || want.MipLevels != 1)
        return false;
    if (copy)
    {
        const auto have = copy->GetDesc();
        if (have.Width != want.Width || have.Height != want.Height ||
            have.Format != TypedGuideFormat(want.Format))
            copy.Reset(); // caller checked the previous GPU fence
    }
    if (!copy)
        copy.Attach(CreateGuideClone(device.Get(), source));
    return copy != nullptr;
}

Slot* Acquire(ID3D12GraphicsCommandList* cmd)
{
    if (!cmd || State::Instance().swapchainInteropApi != SwapchainInteropApi::None)
    { Say("This option needs a native DirectX 12 game."); return nullptr; }
    ComPtr<ID3D12Device> currentDevice;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&currentDevice))))
        return nullptr;
    if (device && device != currentDevice)
    {
        Say("The graphics device changed. Restart the game to use this option.");
        return nullptr;
    }
    device = currentDevice;
    tracking.store(true);
    // Only the lightweight submission hook is needed, including when FG is disabled.
    ResTrack_Dx12::HookLateNrQueue(device.Get());
    Slot* next = nullptr;
    for (auto& slot : slots)
        if (!slot.pending && Finished(slot)) { next = &slot; break; }
    if (!next)
    {
        Say("Waiting for the previous picture to finish.");
        return nullptr;
    }
    auto& slot = *next;
    if (!slot.commands)
    {
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&slot.fence))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&slot.allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot.allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&slot.commands))) || FAILED(slot.commands->Close()))
        {
            slot.commands.Reset(); slot.allocator.Reset(); slot.fence.Reset();
            Say("Could not prepare the finished-picture option.");
            return nullptr;
        }
    }
    return next;
}

void Arm(Slot& slot, ID3D12GraphicsCommandList* cmd)
{
    slot.producer = cmd;
    ID3D12GraphicsCommandList* real = nullptr;
    if (Util::CheckForRealObject(__FUNCTION__, cmd, (IUnknown**) &real)) slot.producer = real;
    slot.serial = ++serial;
    slot.ready = slot.done + 1;
    slot.done = slot.ready;
    slot.submitted = false;
    slot.pending = true;
}

void Capture(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* params, bool rr)
{
    if (!params) return;
    auto* depth = GetResource(params, NVSDK_NGX_Parameter_Depth, "DLSSD.Depth");
    auto* motion = GetResource(params, NVSDK_NGX_Parameter_MotionVectors, "DLSSD.MotionVectors");
    auto* output = GetResource(params, NVSDK_NGX_Parameter_Output, "DLSSD.Output");
    if (!depth || !motion || !output)
    { Cancel(); Say("Waiting for the game's depth and movement data."); return; }
    auto* next = Acquire(cmd);
    if (!next) return;
    auto& slot = *next;
    if (!Clone(slot.depth, depth) || !Clone(slot.motion, motion))
    { Say("The game's depth or movement data is not supported."); return; }
    slot.residualOnly = false;
    // Copy at the NGX seam, where guide states and lifetimes are defined. Keep typed,
    // shader-readable copies until both the producing queue and NR have finished.
    for (auto pair : { std::pair { depth, slot.depth.Get() }, std::pair { motion, slot.motion.Get() } })
    {
        Barrier(cmd, pair.first, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->CopyResource(pair.second, pair.first);
        Barrier(cmd, pair.first, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        // The copy returns to COPY_DEST after the late dispatch.
    }
    slot.frame = {};
    auto& frame = slot.frame;
    unsigned flags = 0, gameReset = 0;
    params->Get(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, &flags);
    params->Get(NVSDK_NGX_Parameter_Reset, &gameReset);
    frame.Reset = gameReset != 0;
    frame.DepthInverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
    frame.MotionVectorsLowResolution = (flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) != 0;
    frame.RayReconstruction = rr;
    frame.OutputWidth = (unsigned) output->GetDesc().Width;
    frame.OutputHeight = output->GetDesc().Height;
    params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, &frame.RenderSubrectWidth);
    params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, &frame.RenderSubrectHeight);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, &frame.DepthSubrectBaseX);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, &frame.DepthSubrectBaseY);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, &frame.MotionSubrectBaseX);
    params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, &frame.MotionSubrectBaseY);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_X, &frame.MvScaleX);
    params->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &frame.MvScaleY);
    frame.ColourIsLinearHdr = false;
    frame.IndependentCommands = true;
    frame.FinishedPicture = true;
    frame.OutputArrivalState = D3D12_RESOURCE_STATE_PRESENT;
    frame.SubmissionEpoch = State::Instance().frameCount;
    Arm(slot, cmd);
}

bool CaptureResidual(ID3D12GraphicsCommandList* cmd, ID3D12Resource* clean, ID3D12Resource* residual,
                     float scale, bool sceneLinear)
{
    auto* next = Acquire(cmd);
    if (!next) return false;
    auto& slot = *next;
    if (!Clone(slot.residual, residual))
    { Say("The upscaled changes could not be saved."); return false; }
    Barrier(cmd, residual, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cmd->CopyResource(slot.residual.Get(), residual);
    Barrier(cmd, residual, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    slot.frame = {};
    slot.frame.OutputWidth = (unsigned) clean->GetDesc().Width;
    slot.frame.OutputHeight = clean->GetDesc().Height;
    slot.frame.PreExposure = scale;
    slot.frame.SubmissionEpoch = State::Instance().frameCount;
    slot.residualOnly = true;
    slot.sceneLinear = sceneLinear;
    Arm(slot, cmd);
    return true;
}

} // namespace Late

void FinishedPictureResetCommandList(ID3D12CommandList* cmd)
{
    std::lock_guard<std::recursive_mutex> lock(g_nrMutex);
    if (g_gpuTime) g_gpuTime->ResetRecording(cmd);
    if (g_ngxTime) g_ngxTime->ResetRecording(cmd);
    if (!Late::tracking.load()) return;
    for (auto& slot : Late::slots)
        if (slot.pending && !slot.submitted && slot.producer == cmd)
        {
            slot.pending = false;
            slot.done = slot.ready - 1; // discarded recording: no GPU signal was promised
            Late::reset = true;
        }
}

bool WaitForFinishedPicture()
{
    if (!Late::tracking.load()) return true;
    std::lock_guard<std::recursive_mutex> lock(g_nrMutex);
    Late::Cancel();
    for (auto& slot : Late::slots)
    {
        if (!slot.submitted || Late::Finished(slot)) continue;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) return false;
        const auto hr = slot.fence->SetEventOnCompletion(slot.done, event);
        const bool finished = SUCCEEDED(hr) && WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
        CloseHandle(event);
        if (!finished) return false;
    }
    return true;
}

std::string FinishedPictureStatus()
{
    std::lock_guard<std::recursive_mutex> lock(g_nrMutex);
    return Late::status;
}

void FinishedPictureSubmitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists)
{
    std::lock_guard<std::recursive_mutex> lock(g_nrMutex);
    if (g_gpuTime) g_gpuTime->Submitted(queue, count, lists);
    if (g_ngxTime) g_ngxTime->Submitted(queue, count, lists);
    if (!Late::tracking.load()) return;
    for (auto& slot : Late::slots)
        if (slot.pending && !slot.submitted)
            for (UINT i = 0; i < count; ++i)
                if (lists[i] == slot.producer)
                {
                    // Signal after ExecuteCommandLists, never when merely recording the copy.
                    slot.submitted = true;
                    if (FAILED(queue->Signal(slot.fence.Get(), slot.ready)))
                    {
                        // The copy already executed. Keep its unsignalled fence protecting
                        // the slot instead of treating this as a discarded recording.
                        slot.pending = false;
                        Late::Say("The graphics queue stopped. Restart the game to retry.");
                    }
                    break;
                }
}

void FinishedPictureColorSpace(IDXGISwapChain* swapchain, DXGI_COLOR_SPACE_TYPE colorSpace)
{
    swapchain->SetPrivateData(Late::colorSpaceKey, sizeof(colorSpace), &colorSpace);
}

void ApplyToFinishedPicture(IDXGISwapChain* swapchain, ID3D12CommandQueue* queue)
{
    std::lock_guard<std::recursive_mutex> lock(g_nrMutex);
    if (!Config::Instance()->DlssNrFinishedPicture.value_or_default() ||
        !Config::Instance()->DlssNrEnabled.value_or_default())
    {
        Late::Cancel();
        return;
    }
    if (!swapchain || !queue || State::Instance().swapchainInteropApi != SwapchainInteropApi::None)
        return;
    Late::ComPtr<IDXGISwapChain3> sc;
    Late::ComPtr<ID3D12Resource> color;
    Late::ComPtr<ID3D12Device> currentDevice;
    if (FAILED(swapchain->QueryInterface(IID_PPV_ARGS(&sc))) ||
        FAILED(sc->GetBuffer(sc->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&color))) ||
        FAILED(queue->GetDevice(IID_PPV_ARGS(&currentDevice))) || currentDevice != Late::device)
        return;
    const auto desc = color->GetDesc();
    auto colorSpace = desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT
        ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    UINT colorSpaceSize = sizeof(colorSpace);
    swapchain->GetPrivateData(Late::colorSpaceKey, &colorSpaceSize, &colorSpace);
    const bool pq = colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    const bool scrgb = colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
    const bool sdr = colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    if ((!sdr && !pq && !scrgb) ||
        (scrgb && desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) ||
        (!scrgb && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && desc.Format != DXGI_FORMAT_R10G10B10A2_UNORM))
    {
        Late::Cancel();
        Late::Say("This screen colour format is not supported.");
        return;
    }
    Late::Slot* latest = nullptr;
    const auto epoch = State::Instance().frameCount;
    const bool residualOnly = Config::Instance()->DlssNrRunBeforeSr.value_or_default() ||
                              Config::Instance()->DlssNrDeferredDlss.value_or_default();
    for (auto& slot : Late::slots)
    {
        if (!slot.pending || !slot.submitted)
            continue;
        if (epoch < slot.frame.SubmissionEpoch || epoch - slot.frame.SubmissionEpoch > 1)
        {
            slot.pending = false;
            Late::reset = true;
            continue;
        }
        if (slot.residualOnly == residualOnly && slot.frame.OutputWidth == desc.Width && slot.frame.OutputHeight == desc.Height &&
            (!latest || slot.serial > latest->serial))
            latest = &slot;
    }
    if (!latest)
        return; // loading screen, another swapchain, or this real frame was already consumed
    auto& slot = *latest;
    // Drop other submitted evaluates from this picture, not their in-flight resources.
    for (auto& other : Late::slots)
        if (other.submitted && other.serial <= slot.serial)
            other.pending = false;
    if (FAILED(queue->Wait(slot.fence.Get(), slot.ready)) || FAILED(slot.allocator->Reset()) ||
        FAILED(slot.commands->Reset(slot.allocator.Get(), nullptr)))
    {
        Late::reset = true;
        Late::Say("Could not prepare the finished picture.");
        return;
    }
    auto* cmd = slot.commands.Get();
    if (!g_compose)
        g_compose = std::make_unique<DlssNr_Dx12>("Neural Rendering", Late::device.Get());
    const auto before = g_nr.successfulDispatches;
    bool appliedResidual = false;
    if (slot.residualOnly)
    {
        if (slot.encoded && (slot.encoded->GetDesc().Width != desc.Width ||
            slot.encoded->GetDesc().Height != desc.Height || slot.encoded->GetDesc().Format != desc.Format))
            slot.encoded.Reset();
        if (!slot.encoded)
            slot.encoded.Attach(CreateScratch(Late::device.Get(), desc.Format, (unsigned) desc.Width, desc.Height));
        if (slot.encoded && Config::Instance()->DlssNrApplyModel.value_or_default())
        {
            Barrier(cmd, slot.residual.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            DlssNrConstants apply {};
            apply.Mode = pq ? 4 : scrgb ? 3 : 2;
            apply.Width = (unsigned) desc.Width; apply.Height = desc.Height;
            apply.WhitePoint = slot.frame.PreExposure;
            apply.TransferStrength = slot.sceneLinear ? 1.0f : 0.0f;
            apply.MaxRatio = std::clamp(Config::Instance()->DlssNrMaxRatio.value_or_default(), 1.0f, 8.0f);
            apply.ShadowFloor = std::clamp(Config::Instance()->DlssNrShadowFloor.value_or_default(), 0.0f, 1.0f);
            appliedResidual = g_compose->DispatchResidualPass(cmd, apply, color.Get(), nullptr,
                slot.residual.Get(), nullptr, slot.encoded.Get(), true);
            if (appliedResidual)
            {
                Barrier(cmd, slot.encoded.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                cmd->CopyResource(color.Get(), slot.encoded.Get());
                Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(cmd, slot.encoded.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT);
            Barrier(cmd, slot.residual.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        }
    }
    else
    {
        auto frame = slot.frame;
        frame.ColourIsLinearHdr = pq || scrgb;
        // Absolute display encodings: 203-nit reference white, in 80-nit scRGB units.
        frame.WhitePointOverride = (pq || scrgb) ? 203.0f / 80.0f : 0.0f;
        frame.Reset |= Late::reset;
        frame.SubmissionEpoch = epoch;
        Barrier(cmd, slot.depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(cmd, slot.motion.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        DlssNrNative::SetPrecision(Config::Instance()->DlssNrPrecision.value_or_default());
        ID3D12Resource* nrColor = color.Get();
        bool colorReady = true;
        DlssNrConstants conversion {};
        conversion.Width = (unsigned) desc.Width;
        conversion.Height = desc.Height;
        if (pq)
        {
            auto ensure = [&](Late::ComPtr<ID3D12Resource>& resource, DXGI_FORMAT format) {
                if (resource && (resource->GetDesc().Width != desc.Width || resource->GetDesc().Height != desc.Height ||
                                 resource->GetDesc().Format != format)) resource.Reset();
                if (!resource) resource.Attach(CreateScratch(Late::device.Get(), format, (unsigned) desc.Width, desc.Height));
                return resource != nullptr;
            };
            colorReady = ensure(slot.linear, DXGI_FORMAT_R16G16B16A16_FLOAT) && ensure(slot.encoded, desc.Format);
            if (colorReady)
            {
                Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                colorReady = g_compose->DispatchResidualPass(cmd, conversion, color.Get(), nullptr, nullptr, nullptr,
                                                           slot.linear.Get(), true);
                Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT);
                // Dispatch reads the converted colour; its transition out of UAV orders the conversion.
                nrColor = slot.linear.Get();
                frame.OutputArrivalState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
        }
        if (colorReady)
            g_compose->Dispatch(cmd, nrColor, slot.depth.Get(), slot.motion.Get(), nrColor, frame, queue);
        if (pq && colorReady && g_nr.successfulDispatches > before && Config::Instance()->DlssNrApplyModel.value_or_default())
        {
            conversion.Mode = 1;
            Barrier(cmd, slot.linear.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (g_compose->DispatchResidualPass(cmd, conversion, slot.linear.Get(), nullptr, color.Get(), nullptr,
                                               slot.encoded.Get(), true))
            {
                Barrier(cmd, slot.encoded.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                cmd->CopyResource(color.Get(), slot.encoded.Get());
                Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Barrier(cmd, slot.encoded.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            Barrier(cmd, color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PRESENT);
            Barrier(cmd, slot.linear.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        Barrier(cmd, slot.motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        Barrier(cmd, slot.depth.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    if (FAILED(cmd->Close()))
    {
        Late::Say("Could not finish the picture. Restart the game to retry.");
        slot.pending = true; // quarantine the slot; do not reuse possibly recorded NR resources
        slot.submitted = false;
        return;
    }
    ID3D12CommandList* lists[] = { cmd };
    queue->ExecuteCommandLists(1, lists);
    slot.done = slot.ready + 1;
    if (FAILED(queue->Signal(slot.fence.Get(), slot.done)))
    {
        Late::Say("The graphics queue stopped. Restart the game to retry.");
        return;
    }
    const bool ran = slot.residualOnly ? appliedResidual : g_nr.successfulDispatches > before;
    Late::reset = !ran;
    Late::Say(!Config::Instance()->DlssNrApplyModel.value_or_default() ? "NR changes are hidden." :
        ran ? (slot.residualOnly ? "Applying the pre-SR changes to the finished picture." :
        "Applying NR to the finished picture.") : "Preparing NR for the finished picture.");
    if (ran && (++Late::successes == 1 || Late::successes % 300 == 0))
        LOG_INFO("DLSS-NR finished picture: {} frames, {}x{}, FG {}", Late::successes,
                 desc.Width, desc.Height, State::Instance().currentFG &&
                 State::Instance().currentFG->IsActive() && !State::Instance().currentFG->IsPaused());
}
