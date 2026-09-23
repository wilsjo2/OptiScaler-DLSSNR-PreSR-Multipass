#include "pch.h"
#include <dlssnr/DlssNr_StreamlinePicture.h>
#include "DlssNr_Dx12_State.h"
#include <atomic>
#include <list>
#include "precompile/DlssNr_Shader.h"
#include "precompile/dlssnr_residual_Shader.h"
#include "precompile/dlssnr_finished_color_Shader.h"
#include "precompile/dlssnr_spatial_Shader.h"
#include "precompile/dlssnr_spatial_guides_Shader.h"

namespace
{
std::recursive_mutex nrOwnersMutex;
std::vector<DlssNr_Dx12*> nrOwners;
std::atomic_uint nrCaptureOutstanding { 0 };
// Only unresolved work at process teardown is intentionally retained. During the session
// retired owners stay registered for submission/reset callbacks until they can be reclaimed.
auto& RetiredNrOwners()
{
    static auto* owners = new std::list<std::unique_ptr<DlssNr_Dx12>>;
    return *owners;
}
unsigned nrNotificationDepth = 0;
void CollectRetiredNrOwners()
{
    static bool collecting = false;
    if (collecting || nrNotificationDepth)
        return;
    collecting = true;
    auto& owners = RetiredNrOwners();
    for (auto it = owners.begin(); it != owners.end();)
    {
        if (!(*it)->ReadyToDestroy())
        {
            ++it;
            continue;
        }
        auto finished = std::move(*it);
        it = owners.erase(it);
        finished.reset(); // May enqueue a child codec; list iterators remain valid.
        LOG_INFO("DLSS-NR: reclaimed retired GPU owner; {} waiting", owners.size());
    }
    collecting = false;
}
struct NrNotificationScope
{
    NrNotificationScope() { ++nrNotificationDepth; }
    ~NrNotificationScope()
    {
        --nrNotificationDepth;
        CollectRetiredNrOwners();
    }
};
DlssNr_Dx12* activeNrOwner = nullptr;
void ActivateNrOwner(DlssNr_Dx12* owner)
{
    if (std::find(nrOwners.begin(), nrOwners.end(), owner) == nrOwners.end())
        nrOwners.push_back(owner);
    activeNrOwner = owner;
}
} // namespace

// ---------------------------------------------------------------------------------------------
// The pass itself. Everything above is what it is made of; everything below is the shape the rest
// of OptiScaler sees.
// ---------------------------------------------------------------------------------------------

DlssNr_Dx12::DlssNr_Dx12(std::string InName, ID3D12Device* InDevice)
    : Shader_Dx12(InName, InDevice), _state(std::make_unique<State>(*this))
{
    if (InDevice == nullptr)
    {
        LOG_ERROR("InDevice is nullptr!");
        return;
    }

    LOG_DEBUG("{0} start!", _name);

    // Five inputs, two outputs, one constant buffer, and a clamped linear sampler.
    //
    // The sampler exists because the model may be run below full resolution, in which case its answer
    // has to be read back at a different size from the frame it is being transferred onto.
    D3D12_STATIC_SAMPLER_DESC sampler {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    if (!SetupRootSignature(InDevice, kSrvCount, kUavCount, 1, 0, 0, 1, &sampler))
    {
        LOG_ERROR("[{0}] Failed to setup root signature", _name);
        return;
    }

    D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(DlssNrConstants));
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

    for (uint32_t i = 0; i < DLSSNR_NUM_OF_HEAPS; ++i)
    {
        auto result = InDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS(&_constantBuffers[i]));

        if (result != S_OK)
        {
            LOG_ERROR("[{0}] CreateCommittedResource error {1:x}", _name, (unsigned int) result);
            return;
        }
    }

    // Precompiled, with no source fallback. The shader used to be compiled at runtime from a string,
    // which would have meant no shader at all for anyone leaving UsePrecompiledShaders at its
    // default.
    if (!CreateComputePipeline(InDevice, &_pipelineState, DlssNr_cso, sizeof(DlssNr_cso), nullptr))
    {
        LOG_ERROR("[{0}] Failed to create the compute pipeline", _name);
        return;
    }

    // Second PSO for the ResidualAcrossRR v2 accumulator (its own blob, same root signature).
    // A failure here is not fatal to the class -- only that experimental mode goes unavailable.
    if (!CreateComputePipeline(InDevice, &_residualPipelineState, dlssnr_residual_cso, sizeof(dlssnr_residual_cso),
                               nullptr))
    {
        _residualPipelineState = nullptr;
        LOG_WARN("[{0}] ResidualAcrossRR compute pipeline unavailable", _name);
    }

    _init = InitHeaps(InDevice, _frameHeaps, DLSSNR_NUM_OF_HEAPS);
    if (_init)
        ResTrack_Dx12::HookLateNrQueue(InDevice); // Observe feature-creation submissions too, before the first Run.
    // Codec-only instances never call Dispatch/ProcessSeam, but still record GPU work.
    std::lock_guard lock(nrOwnersMutex);
    nrOwners.push_back(this);
}

bool DlssNr_Dx12::DispatchPass(ID3D12GraphicsCommandList* InCmdList, const DlssNrConstants& InConstants,
                               ID3D12Resource* InSource, ID3D12Resource* InModel, ID3D12Resource* InOriginal,
                               ID3D12Resource* InMotion, ID3D12Resource* InPrevEdit, ID3D12Resource* OutTarget,
                               ID3D12Resource* OutKeep, uint32_t* immutableSlot)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard stateLock(_state->mutex);
    return DispatchCompute(InCmdList, InConstants, _pipelineState, InSource, InModel, InOriginal, InMotion, InPrevEdit,
                           OutTarget, OutKeep, immutableSlot);
}

bool DlssNr_Dx12::DispatchCompute(ID3D12GraphicsCommandList* InCmdList, const DlssNrConstants& InConstants,
                                  ID3D12PipelineState* pipeline, ID3D12Resource* InSource, ID3D12Resource* InModel,
                                  ID3D12Resource* InOriginal, ID3D12Resource* InMotion, ID3D12Resource* InPrevEdit,
                                  ID3D12Resource* OutTarget, ID3D12Resource* OutKeep, uint32_t* immutableSlot)
{
    _state->lifetime.Record(InCmdList);
    if (!_init || !pipeline || !InCmdList || !_device || !InSource || !OutTarget)
        return false;

    const bool reuse = immutableSlot && *immutableSlot != UINT32_MAX;
    const uint32_t slot = reuse ? *immutableSlot : _heapIndex;
    if (!reuse)
        _heapIndex = (_heapIndex + 1) % DLSSNR_NUM_OF_HEAPS;

    FrameDescriptorHeap& currentHeap = _frameHeaps[slot];
    if (!reuse)
    {

        // Every slot in the table gets a view, whether the mode reads it or not. An unbound descriptor is
        // not an empty read; it is a read from nothing, and the source stands in wherever a mode has
        // nothing of its own to put there.
        ID3D12Resource* const srvs[kSrvCount] = {
            InSource,
            InModel != nullptr ? InModel : InSource,
            InOriginal != nullptr ? InOriginal : InSource,
            InMotion != nullptr ? InMotion : InSource,
            InPrevEdit != nullptr ? InPrevEdit : InSource,
        };

        for (uint32_t i = 0; i < kSrvCount; ++i)
        {
            // Copied depth guides already use an SRV format; the upstream translator maps it back to a DSV.
            const bool translate = srvs[i]->GetDesc().Format != DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            CreateShaderResourceView(_device, srvs[i], currentHeap.GetSrvCPU(i), DXGI_FORMAT_UNKNOWN, translate);
        }

        ID3D12Resource* const uavs[kUavCount] = {
            OutTarget,
            OutKeep != nullptr ? OutKeep : OutTarget,
        };

        for (uint32_t i = 0; i < kUavCount; ++i)
            CreateUnorderedAccessView(_device, uavs[i], currentHeap.GetUavCPU(i), 0);

        if (!CreateConstantsBuffer(_device, _constantBuffers[slot], InConstants, currentHeap.GetCbvCPU(0)))
        {
            LOG_ERROR("[{0}] Failed to create a constants buffer", _name);
            return false;
        }

        if (immutableSlot)
            *immutableSlot = slot;
    }

    ID3D12DescriptorHeap* heaps[] = { currentHeap.GetHeapCSU() };
    InCmdList->SetDescriptorHeaps(_countof(heaps), heaps);
    InCmdList->SetComputeRootSignature(_rootSignature);
    InCmdList->SetPipelineState(pipeline);
    InCmdList->SetComputeRootDescriptorTable(0, currentHeap.GetTableGPUStart());

    // Sized from the constants rather than from a resource, because the pass that shrinks the proxy
    // writes fewer pixels than its source has.
    const UINT dispatchWidth = InConstants.Mode == DlssNrMode_Meter
                                   ? InConstants.Width
                                   : (InConstants.Width + _numThreadsX - 1) / _numThreadsX;
    const UINT dispatchHeight = InConstants.Mode == DlssNrMode_Meter
                                    ? InConstants.Height
                                    : (InConstants.Height + _numThreadsY - 1) / _numThreadsY;
    InCmdList->Dispatch(dispatchWidth, dispatchHeight, 1);

    return true;
}

void DlssNr_Dx12::Retire(std::unique_ptr<DlssNr_Dx12> owner)
{
    if (!owner)
        return;
    if (::State::Instance().isShuttingDown)
    {
        owner.release(); // No locks, GPU calls or destructors under the loader lock.
        return;
    }
    std::lock_guard lock(nrOwnersMutex);
    if (activeNrOwner == owner.get())
        activeNrOwner = nullptr;
    DlssNr::ClearStatus(owner.get());
    {
        std::lock_guard stateLock(owner->_state->mutex);
        owner->_state->late.Cancel();
    }
    RetiredNrOwners().push_back(std::move(owner));
    LOG_INFO("DLSS-NR: retaining retired GPU owner until recordings finish; {} waiting", RetiredNrOwners().size());
}

bool DlssNr_Dx12::ReadyToDestroy()
{
    std::lock_guard lock(_state->mutex);
    _state->CollectEnlargers();
    if (_state->collectingEnlargers)
        return false;
    if (!_state->retiredEnlargers.empty() || (_state->enlarger && !_state->enlarger->lifetime.Idle()))
        return false;
    if (!_state->lifetime.Idle() || !_state->deferredSr.lifetime.Idle())
        return false;
    for (auto& model : _state->nr.models)
        if (!model.Idle())
            return false;
    for (const auto& slot : _state->late.slots)
        if (slot.submitted && !_state->late.Finished(slot))
            return false;
    return _state->late.dx11.Idle();
}

void DlssNr_Dx12::FinishSubmitted()
{
    std::lock_guard lock(_state->mutex);
    _state->lifetime.FinishSubmitted();
    _state->deferredSr.lifetime.FinishSubmitted();
    _state->captureFrames.FinishSubmitted();
    if (_state->enlarger)
        _state->enlarger->lifetime.FinishSubmitted();
    for (auto& old : _state->retiredEnlargers)
        old->lifetime.FinishSubmitted();
    for (auto& model : _state->nr.models)
        model.FinishSubmitted();
}

DlssNr_Dx12::~DlssNr_Dx12()
{
    if (::State::Instance().isShuttingDown)
    {
        _state.release();
        for (auto& heap : _frameHeaps)
            heap.Abandon();
        GpuTime.release();
        return;
    }
    std::lock_guard lock(nrOwnersMutex);
    std::erase(nrOwners, this);
    if (activeNrOwner == this)
        activeNrOwner = nullptr;
    DlssNr::ClearStatus(this);
    const bool finished = _state->WaitForFinishedPicture();
    if (!finished || !_state->lifetime.Idle())
    {
        LOG_WARN("DLSS-NR: abandoning GPU ownership with unresolved command recordings at teardown");
        _state.release();
        for (auto& heap : _frameHeaps)
            heap.Abandon();
        _rootSignature = nullptr;
        _pipelineState = nullptr;
        _constantBuffer = nullptr;
        GpuTime.release();
        return;
    }
    _state.reset();
    for (auto& heap : _frameHeaps)
        heap.ReleaseHeaps();
    if (_finishedColorPipelineState)
        _finishedColorPipelineState->Release();
    for (auto& buffer : _constantBuffers)
    {
        if (buffer != nullptr)
        {
            buffer->Release();
            buffer = nullptr;
        }
    }

    if (_residualPipelineState != nullptr)
    {
        _residualPipelineState->Release();
        _residualPipelineState = nullptr;
    }
    if (_spatialPipelineState != nullptr)
    {
        _spatialPipelineState->Release();
        _spatialPipelineState = nullptr;
    }
    if (_spatialGuidesPipelineState != nullptr)
    {
        _spatialGuidesPipelineState->Release();
        _spatialGuidesPipelineState = nullptr;
    }
}

bool DlssNr_Dx12::SpatialReady()
{
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard stateLock(_state->mutex);
    if (!_init)
        return false;
    if (!_spatialPipelineState)
        CreateComputePipeline(_device, &_spatialPipelineState, dlssnr_spatial_cso, sizeof(dlssnr_spatial_cso), nullptr);
    if (!_spatialGuidesPipelineState)
        CreateComputePipeline(_device, &_spatialGuidesPipelineState, dlssnr_spatial_guides_cso,
                              sizeof(dlssnr_spatial_guides_cso), nullptr);
    return _spatialPipelineState != nullptr && _spatialGuidesPipelineState != nullptr;
}

bool DlssNr_Dx12::DispatchSpatial(ID3D12GraphicsCommandList* cmd, const DlssNr::Spatial::Constants& constants,
                                  ID3D12Resource* source, ID3D12Resource* depthOrAnswer, ID3D12Resource* motion,
                                  ID3D12Resource* target, ID3D12Resource* secondary)
{
    static_assert(sizeof(DlssNr::Spatial::Constants) == sizeof(DlssNrConstants));
    DlssNrConstants bytes {};
    std::memcpy(&bytes, &constants, sizeof(bytes));
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard stateLock(_state->mutex);
    auto* pipeline = constants.mode == 101 ? _spatialGuidesPipelineState : _spatialPipelineState;
    return DispatchCompute(cmd, bytes, pipeline, source, depthOrAnswer, motion, nullptr, nullptr, target, secondary,
                           nullptr);
}

bool DlssNr_Dx12::DispatchResidualPass(ID3D12GraphicsCommandList* InCmdList, const DlssNrConstants& InConstants,
                                       ID3D12Resource* InSource, ID3D12Resource* InModel, ID3D12Resource* InOriginal,
                                       ID3D12Resource* InMotion, ID3D12Resource* OutTarget, bool finishedColor)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard stateLock(_state->mutex);
    if (finishedColor && !_finishedColorPipelineState && _init)
        CreateComputePipeline(_device, &_finishedColorPipelineState, dlssnr_finished_color_cso,
                              sizeof(dlssnr_finished_color_cso), nullptr);
    auto* pipeline = finishedColor ? _finishedColorPipelineState : _residualPipelineState;
    return DispatchCompute(InCmdList, InConstants, pipeline, InSource, InModel, InOriginal, InMotion, nullptr,
                           OutTarget, nullptr, nullptr);
}

bool DlssNr_Dx12::CreateBufferResource(ID3D12Device* device, ID3D12Resource* source, D3D12_RESOURCE_STATES state)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard stateLock(_state->mutex);
    if (device == nullptr || source == nullptr)
        return false;
    auto desc = source->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
        desc.DepthOrArraySize != 1)
        return false;
    desc.MipLevels = 1;
    desc.Alignment = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = (desc.Flags | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) & ~D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    if (_state->buffer != nullptr)
    {
        const auto previous = _state->buffer->GetDesc();
        if (previous.Width == desc.Width && previous.Height == desc.Height && previous.Format == desc.Format &&
            previous.Flags == desc.Flags)
            return true;
        _state->ParkNrResource(_state->buffer);
    }
    const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&_state->buffer))))
        return false;
    _state->bufferState = state;
    return true;
}

void DlssNr_Dx12::SetBufferState(ID3D12GraphicsCommandList* cmdList, D3D12_RESOURCE_STATES state)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard stateLock(_state->mutex);
    _state->lifetime.Record(cmdList);
    Shader_Dx12::SetBufferState(cmdList, state, _state->buffer, &_state->bufferState);
}

ID3D12Resource* DlssNr_Dx12::Buffer() { return _state->buffer; }
bool DlssNr_Dx12::CanRender() const { return _init && _state->buffer != nullptr; }

bool DlssNr_Dx12::Dispatch(ID3D12GraphicsCommandList* cmd, ID3D12Resource* colour, ID3D12Resource* depth,
                           ID3D12Resource* motion, ID3D12Resource* output, const DlssNrFrameInfo& frame,
                           ID3D12CommandQueue* queue)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    ActivateNrOwner(this);
    std::lock_guard stateLock(_state->mutex);
    _state->ConsumeControls();
    struct Publish
    {
        State& s;
        ~Publish() { s.Publish(); }
    } publish { *_state };
    if (!_init || !cmd || !colour || !depth || !motion || !output)
        return false;
    auto info = frame;
    info.PipelineManagedStates = true;
    info.PrivateColorCopy = true;
    if (!info.RenderSubrectWidth)
        info.RenderSubrectWidth = info.Width;
    if (!info.RenderSubrectHeight)
        info.RenderSubrectHeight = info.Height;
    if (colour != output)
    {
        const auto source = colour->GetDesc(), target = output->GetDesc();
        if (source.Width != target.Width || source.Height != target.Height || source.Format != target.Format)
            return false;
        _state->Barrier(cmd, colour, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        _state->Barrier(cmd, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        DlssNr::CopyActiveColor(cmd, output, colour, { (unsigned) target.Width, target.Height });
        _state->Barrier(cmd, colour, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        _state->Barrier(cmd, output, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    const auto before = _state->nr.successfulDispatches;
    _state->Run(cmd, output, depth, motion, output, info, queue);
    return _state->nr.successfulDispatches != before;
}

void DlssNr_Dx12::BeginInputHold(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* params,
                                 const D3D12_RESOURCE_STATES* inputStates)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard lock(_state->mutex);
    _state->BeginInputHold(cmd, params, inputStates);
}

void DlssNr_Dx12::EndInputHold(NVSDK_NGX_Parameter* params)
{
    std::lock_guard lock(_state->mutex);
    _state->inputHold.parameters.Restore(params);
}

bool DlssNr_Dx12::ProcessSeam(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* params, bool beforeUpscale,
                              ID3D12CommandQueue* queue, bool rayReconstruction, unsigned long long submissionEpoch,
                              bool interop, uint32_t featureFlags)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    ActivateNrOwner(this);
    std::lock_guard stateLock(_state->mutex);
    _state->ConsumeControls();
    _state->featureFlags = featureFlags;
    const auto& cfg = *Config::Instance();
    // Both seams reach this scheduler; ordinary passes remain in the shared shader pipeline.
    const auto placement = DlssNr::ResolvePlacement(
        cfg.DlssNrRunBeforeSr.value_or_default(), cfg.DlssNrDeferredDlss.value_or_default(),
        cfg.DlssNrResidualAcrossRr.value_or_default(), cfg.DlssNrFinishedPicture.value_or_default());
    const bool special = placement.finished || placement.deferred;
    if (special)
        _state->EvaluateInternal(cmd, params, beforeUpscale, queue, rayReconstruction, submissionEpoch, interop);
    else
    {
        if (_state->lastFinishedMode != 0)
        {
            _state->lastFinishedMode = 0;
            _state->nr.reset = true;
            if (_state->gpuTime)
                _state->gpuTime->ClearLast();
            if (_state->ngxTime)
                _state->ngxTime->ClearLast();
            _state->lastGpuTime.reset();
            _state->lastNgxTime.reset();
        }
        _state->late.Cancel();
        _state->deferredSr.Cancel();
    }
    _state->Publish();
    return special;
}
void DlssNr_Dx12::DiagnosePipeline(unsigned stage, ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* params,
                                   ID3D12Resource* color, uint32_t flags, bool rr, bool success)
{
    std::lock_guard ownersLock(nrOwnersMutex);
    std::lock_guard stateLock(_state->mutex);
    auto& state = *_state;
    if (stage == 0)
    {
        state.lifetime.Collect();
        static bool previousGameplay = false, previousPhoto = false;
        static unsigned runs = 0;
        DWORD foregroundProcess = 0;
        GetWindowThreadProcessId(GetForegroundWindow(), &foregroundProcess);
        const bool control = foregroundProcess == GetCurrentProcessId() && (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
        const bool gameplay = control && (GetAsyncKeyState(VK_F8) & 0x8000) != 0;
        const bool photo = control && (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
        const char* label = gameplay && !previousGameplay ? "gameplay"
                            : photo && !previousPhoto     ? "photomode"
                                                          : nullptr;
        previousGameplay = gameplay;
        previousPhoto = photo;
        if (label && !state.pipelineCaptureRemaining && !nrCaptureOutstanding)
        {
            if (runs >= 2)
                LOG_WARN("NR pipeline capture: two-run limit reached; restart to capture again");
            else
            {
                ++runs;
                SYSTEMTIME time {};
                GetLocalTime(&time);
                char folder[100];
                std::snprintf(folder, sizeof(folder), "%04u%02u%02u-%02u%02u%02u-%03u-%s-%u", time.wYear, time.wMonth,
                              time.wDay, time.wHour, time.wMinute, time.wSecond, time.wMilliseconds, label, runs);
                state.pipelineCaptureDirectory = Util::DllPath().parent_path() / "nr-pipeline-captures" / folder;
                state.pipelineCaptureRemaining = 4;
                LOG_INFO("NR pipeline capture armed: {} (four frames)", state.pipelineCaptureDirectory.string());
            }
        }
        if (!state.pipelineCaptureRemaining || state.pipelineCapture)
            return;
        auto job = std::make_unique<DlssNr::PipelineCaptureFrame>();
        if (!job->Init(_device))
        {
            state.pipelineCaptureRemaining = 0;
            LOG_ERROR("NR pipeline capture allocation failed");
            return;
        }
        job->directory = state.pipelineCaptureDirectory / std::to_string(4 - state.pipelineCaptureRemaining);
        job->metadata << "stage_semantics before_nr=scene_linear_input after_nr=NR_composed_RR_input "
                         "after_rr=upscaler_output_before_postprocessing\n"
                      << "game_frame " << ::State::Instance().frameCount << " command_list " << cmd << " parameters "
                      << params << " rr " << rr << " feature_flags " << flags << '\n';
        const auto& cfg = *Config::Instance();
        job->metadata << "nr_passes " << cfg.DlssNrPasses.value_or_default() << " working_scale "
                      << cfg.DlssNrWorkingScale.value_or_default() << " nr_history_reset " << state.nr.reset << " hold "
                      << cfg.DlssNrHoldFrame.value_or_default() << '\n';
        for (const char* key :
             { NVSDK_NGX_Parameter_Reset, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
               NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, NVSDK_NGX_Parameter_OutWidth,
               NVSDK_NGX_Parameter_OutHeight, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,
               NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X,
               NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y })
        {
            unsigned value = 0;
            auto result = params->Get(key, &value);
            job->metadata << key << ' ' << value << " get_result " << unsigned(result) << '\n';
        }
        for (const char* key :
             { NVSDK_NGX_Parameter_MV_Scale_X, NVSDK_NGX_Parameter_MV_Scale_Y, NVSDK_NGX_Parameter_Jitter_Offset_X,
               NVSDK_NGX_Parameter_Jitter_Offset_Y, NVSDK_NGX_Parameter_DLSS_Pre_Exposure,
               NVSDK_NGX_Parameter_DLSS_Exposure_Scale, NVSDK_NGX_Parameter_FrameTimeDeltaInMsec })
        {
            float value = 0;
            auto result = params->Get(key, &value);
            job->metadata << key << ' ' << value << " get_result " << unsigned(result) << '\n';
        }
        // Record descriptors of optional RR inputs without assuming their presence.
        for (const char* key : { "DLSS.Input.DiffuseAlbedo", "DLSS.Input.SpecularAlbedo", "GBuffer.Normals",
                                 "GBuffer.Roughness", "MotionVectorsReflection", "DLSSD.SpecularHitDistance",
                                 "DLSS.Input.ColorBeforeParticles", "DLSSD.DiffuseHitDistance" })
        {
            auto* resource = state.GetResource(params, key, key);
            job->metadata << key << " resource " << resource;
            if (resource)
            {
                auto desc = resource->GetDesc();
                job->metadata << " width " << desc.Width << " height " << desc.Height << " format " << desc.Format;
            }
            job->metadata << '\n';
        }
        state.pipelineCapture = job.release();
        ++nrCaptureOutstanding;
        state.lifetime.Record(cmd);
        const auto inputs = DlssNr::ResolveInputStates_Dx12(false);
        state.pipelineCapture->Copy(cmd, _device, "before_nr", color, inputs.color);
        state.pipelineCapture->Copy(cmd, _device, "motion",
                                    state.GetResource(params, NVSDK_NGX_Parameter_MotionVectors, "DLSSD.MotionVectors"),
                                    inputs.motion);
        state.pipelineCapture->Copy(cmd, _device, "depth",
                                    state.GetResource(params, NVSDK_NGX_Parameter_Depth, "DLSSD.Depth"), inputs.depth);
        state.pipelineCapture->Copy(
            cmd, _device, "exposure",
            state.GetResource(params, NVSDK_NGX_Parameter_ExposureTexture, "DLSSD.ExposureTexture"), inputs.exposure);
        return;
    }
    auto* job = state.pipelineCapture;
    if (!job)
        return;
    job->metadata << "stage " << stage << " success " << success << '\n';
    if (stage == 1)
    {
        job->metadata << "nr_model_evaluated " << state.modelRunning << '\n';
        job->Copy(cmd, _device, "after_nr", color, DlssNr::ResolveInputStates_Dx12(false).color);
        return;
    }
    if (success)
        job->Copy(cmd, _device, "after_rr", color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    job->End(cmd);
    state.pipelineCapture = nullptr;
    --state.pipelineCaptureRemaining;
    state.lifetime.Retire(
        [job]
        {
            if (job->Write())
                LOG_INFO("NR pipeline capture saved: {}", job->directory.string());
            else
                LOG_WARN("NR pipeline capture discarded or write failed: {}", job->directory.string());
            delete job;
            --nrCaptureOutstanding;
        });
}

void DlssNr_Dx12::ResetFinishedCommands(ID3D12CommandList* cmd) { _state->FinishedPictureResetCommandList(cmd); }
void DlssNr_Dx12::SubmitFinishedCommands(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists)
{
    _state->FinishedPictureSubmitted(queue, count, lists);
}
bool DlssNr_Dx12::WaitFinished() { return _state->WaitForFinishedPicture(); }
void DlssNr_Dx12::ApplyFinished(ID3D12Resource* picture, ID3D12CommandQueue* queue, DXGI_COLOR_SPACE_TYPE space,
                                bool gameFrameHandoff)
{
    std::lock_guard lock(_state->mutex);
    if (!Config::Instance()->DlssNrFinishedPicture.value_or_default() ||
        !Config::Instance()->DlssNrEnabled.value_or_default())
        _state->late.Cancel();
    else if (picture && queue)
        _state->ApplyFinishedColor(picture, queue, space, gameFrameHandoff);
    _state->Publish();
}
void DlssNr_Dx12::ApplyFinishedDx11(IDXGISwapChain* swapchain)
{
    _state->ApplyToFinishedPictureDx11(swapchain);
    _state->Publish();
}
std::string DlssNr_Dx12::FinishedStatus() { return _state->FinishedPictureStatus(); }
std::string DlssNr_Dx12::DeferredStatus() { return _state->DeferredDlssStatus(); }
namespace DlssNr
{
void FinishedPictureResetCommandList(ID3D12CommandList* cmd)
{
    if (::State::Instance().isShuttingDown)
        return;
    std::lock_guard lock(nrOwnersMutex);
    NrNotificationScope notification;
    const auto owners = nrOwners;
    for (auto* owner : owners)
        owner->ResetFinishedCommands(cmd);
}
void FinishedPictureSubmitted(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists)
{
    if (::State::Instance().isShuttingDown)
        return;
    std::lock_guard lock(nrOwnersMutex);
    NrNotificationScope notification;
    const auto owners = nrOwners;
    for (auto* owner : owners)
        owner->SubmitFinishedCommands(queue, count, lists);
}
bool WaitForFinishedPicture()
{
    if (::State::Instance().isShuttingDown)
        return false;
    std::lock_guard lock(nrOwnersMutex);
    NrNotificationScope notification;
    bool ready = true;
    const auto owners = nrOwners;
    for (auto* owner : owners)
        ready = owner->WaitFinished() && ready;
    return ready;
}
static DXGI_COLOR_SPACE_TYPE ReadFinishedSpace(IDXGISwapChain* swapchain, ID3D12Resource* picture)
{
    auto space = picture->GetDesc().Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709
                                                                             : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    UINT size = sizeof(space);
    swapchain->GetPrivateData(FinishedColorSpaceKey, &size, &space);
    return space;
}
void ApplyToFinishedPicture(IDXGISwapChain* swapchain, ID3D12CommandQueue* queue)
{
    if (::State::Instance().isShuttingDown)
        return;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> chain;
    Microsoft::WRL::ComPtr<ID3D12Resource> picture;
    auto space = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    const auto& config = *Config::Instance();
    // Swapchain calls must precede NR locks: FG Present can submit commands while holding its own lock.
    if (swapchain && queue && config.DlssNrEnabled.value_or_default() &&
        config.DlssNrFinishedPicture.value_or_default())
    {
        if (StreamlinePicture::RenderQueue(swapchain) || FAILED(swapchain->QueryInterface(IID_PPV_ARGS(&chain))) ||
            FAILED(chain->GetBuffer(chain->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&picture))))
            return;
        space = ReadFinishedSpace(swapchain, picture.Get());
    }
    std::lock_guard lock(nrOwnersMutex);
    if (activeNrOwner)
        activeNrOwner->ApplyFinished(picture.Get(), queue, space);
}
void ApplyToStreamlinePicture(IDXGISwapChain* swapchain, ID3D12Resource* picture, ID3D12CommandQueue* queue)
{
    if (::State::Instance().isShuttingDown)
        return;
    if (!swapchain || !picture || !queue)
        return;
    const auto space = ReadFinishedSpace(swapchain, picture);
    std::lock_guard lock(nrOwnersMutex);
    if (activeNrOwner)
        activeNrOwner->ApplyFinished(picture, queue, space, true);
}
void ApplyToFinishedPictureDx11(IDXGISwapChain* swapchain)
{
    if (::State::Instance().isShuttingDown)
        return;
    std::lock_guard lock(nrOwnersMutex);
    if (activeNrOwner)
        activeNrOwner->ApplyFinishedDx11(swapchain);
}
void FinishedPictureColorSpace(IDXGISwapChain* swapchain, DXGI_COLOR_SPACE_TYPE colorSpace)
{
    if (swapchain)
        swapchain->SetPrivateData(FinishedColorSpaceKey, sizeof(colorSpace), &colorSpace);
}
std::string FinishedPictureStatus()
{
    std::lock_guard lock(nrOwnersMutex);
    return activeNrOwner ? activeNrOwner->FinishedStatus() : "Waiting for a finished picture.";
}
std::string DeferredDlssStatus()
{
    std::lock_guard lock(nrOwnersMutex);
    return activeNrOwner ? activeNrOwner->DeferredStatus() : "not started";
}
bool Shutdown()
{
    if (::State::Instance().isShuttingDown)
        return false;
    const auto deadline = GetTickCount64() + 1000;
    do
    {
        {
            std::lock_guard lock(nrOwnersMutex);
            // Callbacks may retire a child codec. Defer owner destruction until traversal ends.
            {
                NrNotificationScope notification;
                for (auto& owner : RetiredNrOwners())
                    owner->FinishSubmitted();
            }
            if (nrOwners.empty())
                return true;
        }
        // Submission/reset hooks must be able to make progress while we drain.
        Sleep(1);
    } while (GetTickCount64() < deadline);
    LOG_WARN("NR shutdown deferred: owners or GPU recordings remain; keeping the NGX runtime alive");
    return false;
}
} // namespace DlssNr
