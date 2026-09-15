// Included in namespace DlssNr by DlssNr_Dx12.cpp, after the private NR helpers.
// Calls below are serialized by g_nrMutex. Nothing writes the game's NGX parameter block.
namespace DeferredSr
{
constexpr unsigned MarkerCount = 16;
struct HalfRate
{
    ID3D12Resource *motion = nullptr, *previousMotion = nullptr, *anchorMotion = nullptr,
                   *previousResidual = nullptr;
    bool ready = false, failed = false, havePrevious = false, previousWasAnchor = false;
    bool motionReadable = false, previousReadable = false, anchorReadable = false;
    bool previousResidualReadable = false, activationLogged = false;
    unsigned long long lastEpoch = 0;
    unsigned long long nrAnchors = 0, skippedNr = 0;
    unsigned long long continuityBreaks = 0;
    ~HalfRate()
    {
        // Generation's GPU completion markers protect all of these lifetimes.
        for (auto* r : { motion, previousMotion, anchorMotion, previousResidual }) if (r) r->Release();
    }
    void Reset() { havePrevious = false; previousWasAnchor = false; }
};
struct Generation
{
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr; // identity/reference only; no private submissions
    unsigned w = 0, h = 0, outW = 0, outH = 0, flags = 0;
    DXGI_FORMAT inputFormat {}, outputFormat {};
    ID3D12Resource *edited = nullptr, *residualInput = nullptr, *residualOutput = nullptr, *clean = nullptr,
                   *composed = nullptr, *exposure = nullptr, *readback = nullptr;
    ID3D12QueryHeap* queries = nullptr;
    volatile UINT64* completed = nullptr;
    bool occupied[MarkerCount] {};
    unsigned nextMarker = 0, lastMarker = 0;
    bool everRecorded = false, smallReadable = false, reset = true, failed = false;
    NVSDK_NGX_Parameter* parameters = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    unsigned long long createEpoch = 0;
    unsigned long long lastBeginEpoch = 0;
    bool began = false;
    std::unique_ptr<DlssNr_Dx12> codec;
    bool halfRequested = false;
    std::string halfStatus;
    bool sampleAndHold = false;
    DlssNrResidualHold hold;
    ID3D12Resource* zeroMotion = nullptr;
    std::unique_ptr<HalfRate> half;

    bool Idle() const { return !everRecorded || completed[lastMarker] != 0; }
    ~Generation()
    {
        half.reset();
        if (zeroMotion) zeroMotion->Release();
        if (feature && NVNGXProxy::D3D12_ReleaseFeature())
            NVNGXProxy::D3D12_ReleaseFeature()(feature);
        if (parameters && NVNGXProxy::D3D12_DestroyParameters())
            NVNGXProxy::D3D12_DestroyParameters()(parameters);
        if (readback && completed) readback->Unmap(0, nullptr);
        for (auto* r : { edited, residualInput, residualOutput, clean, composed, exposure, readback })
            if (r) r->Release();
        if (queries) queries->Release();
        if (queue) queue->Release();
        if (device) device->Release();
    }
};

// Record an actual GPU completion marker after EACH seam. A later CPU frame/Present count alone
// does not prove a resource is no longer in flight. Slots aren't reused until the GPU wrote them.
struct Use
{
    Generation& g;
    ID3D12GraphicsCommandList* cmd;
    unsigned slot;
    bool valid;
    Use(Generation& gen, ID3D12GraphicsCommandList* commands) : g(gen), cmd(commands), slot(g.nextMarker)
    {
        valid = !g.occupied[slot] || g.completed[slot] != 0;
        if (!valid) return;
        g.completed[slot] = 0;
        g.occupied[slot] = true;
        g.lastMarker = slot;
        g.everRecorded = true;
        g.nextMarker = (slot + 1) % MarkerCount;
    }
    ~Use()
    {
        if (!valid) return;
        cmd->EndQuery(g.queries, D3D12_QUERY_TYPE_TIMESTAMP, slot);
        cmd->ResolveQueryData(g.queries, D3D12_QUERY_TYPE_TIMESTAMP, slot, 1,
                              g.readback, slot * sizeof(UINT64));
    }
};

std::unique_ptr<Generation> current;
std::vector<std::unique_ptr<Generation>> retired;
std::string status = "not started";
struct Pending
{
    ID3D12GraphicsCommandList* cmd = nullptr;
    NVSDK_NGX_Parameter* caller = nullptr;
    ID3D12Resource* output = nullptr;
    unsigned long long epoch = 0;
    float scale = 1;
    bool skipNr = false;
    bool half = false;
} pending;

void Say(const std::string& text)
{
    if (status == text) return;
    status = text;
    LOG_INFO("DLSS-NR deferred DLSS: {}", text);
}
void Cancel()
{
    pending = {};
    if (current) { current->reset = true; current->hold.Reset(); if (current->half) current->half->Reset(); }
}
void Collect()
{
    std::erase_if(retired, [](const auto& g) { return g->Idle(); });
}

unsigned UInt(NVSDK_NGX_Parameter* p, const char* key, unsigned fallback = 0)
{
    unsigned value = fallback;
    p->Get(key, &value);
    return value;
}
float Float(NVSDK_NGX_Parameter* p, const char* key, float fallback)
{
    float value = fallback;
    p->Get(key, &value);
    return std::isfinite(value) ? value : fallback;
}

bool Allocate(Generation& g)
{
    g.edited = CreateScratch(g.device, g.inputFormat, g.w, g.h);
    g.residualInput = CreateScratch(g.device, DXGI_FORMAT_R16G16B16A16_FLOAT, g.w, g.h);
    g.residualOutput = CreateScratch(g.device, DXGI_FORMAT_R16G16B16A16_FLOAT, g.outW, g.outH);
    g.clean = CreateScratch(g.device, g.outputFormat, g.outW, g.outH);
    g.composed = CreateScratch(g.device, g.outputFormat, g.outW, g.outH);
    g.exposure = CreateScratch(g.device, DXGI_FORMAT_R32_FLOAT, 1, 1);
    if (!g.edited || !g.residualInput || !g.residualOutput || !g.clean || !g.composed || !g.exposure) return false;
    g.codec = std::make_unique<DlssNr_Dx12>("Deferred NR contribution", g.device);
    if (!g.codec->IsInit()) return false;
    D3D12_QUERY_HEAP_DESC query {};
    query.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    query.Count = MarkerCount;
    if (FAILED(g.device->CreateQueryHeap(&query, IID_PPV_ARGS(&g.queries)))) return false;
    auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto desc = CD3DX12_RESOURCE_DESC::Buffer(MarkerCount * sizeof(UINT64));
    if (FAILED(g.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
              D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g.readback)))) return false;
    void* mapped = nullptr;
    if (FAILED(g.readback->Map(0, nullptr, &mapped))) return false;
    g.completed = static_cast<volatile UINT64*>(mapped);
    for (unsigned i = 0; i < MarkerCount; ++i) g.completed[i] = 0;
    return true;
}

bool CreateHalfRate(Generation& g)
{
    g.half = std::make_unique<HalfRate>();
    auto& h = *g.half;
    h.motion = CreateScratch(g.device, DXGI_FORMAT_R32G32B32A32_FLOAT, g.w, g.h);
    h.previousMotion = CreateScratch(g.device, DXGI_FORMAT_R32G32B32A32_FLOAT, g.w, g.h);
    h.anchorMotion = CreateScratch(g.device, DXGI_FORMAT_R32G32B32A32_FLOAT, g.w, g.h);
    h.previousResidual = CreateScratch(g.device, DXGI_FORMAT_R16G16B16A16_FLOAT, g.outW, g.outH);
    if (!h.motion || !h.previousMotion || !h.anchorMotion || !h.previousResidual) return false;
    h.ready = true;
    return true;
}

bool PrepareHalfRate(Generation& g, ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source,
                     ID3D12Resource* motion, unsigned long long epoch, unsigned long long submittedEpoch)
{
    if (!g.halfRequested) return false;
    if (!(g.flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) ||
        (g.flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered))
    { g.halfStatus = "requires low-resolution, non-jittered motion"; return false; }
    if (!g.half)
    {
        g.halfStatus = "initializing current-raster half-rate resources";
        if (!CreateHalfRate(g))
        {
            g.half->failed = true;
            g.halfStatus = "current-raster half-rate allocation failed";
        }
        return false;
    }
    auto& h = *g.half;
    if (!h.ready || h.failed) return false;
    if (h.havePrevious && epoch != h.lastEpoch + 1) ++h.continuityBreaks;
    if (g.reset || UInt(source, NVSDK_NGX_Parameter_Reset) || epoch != h.lastEpoch + 1) h.Reset();
    h.lastEpoch = epoch;
    auto desc = motion->GetDesc();
    if (!h.activationLogged)
    {
        h.activationLogged = true;
        LOG_INFO("DLSS-NR current-raster half-rate activation: flags 0x{:X}, motion {}x{} format {}, active guides {}x{}, "
                 "output {}x{}, epoch {}, submitted epoch {}", g.flags, desc.Width, desc.Height,
                 (unsigned) desc.Format, g.w, g.h, g.outW, g.outH, epoch, submittedEpoch);
    }
    if (desc.Width < g.w || desc.Height < g.h || desc.MipLevels != 1 || desc.SampleDesc.Count != 1)
    { h.Reset(); g.halfStatus = "unsupported motion texture"; return false; }
    if (h.motionReadable) Barrier(cmd, h.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DlssNrConstants normalize {}; normalize.Mode = DlssNrMode_NormalizeMotion;
    normalize.Width = g.w; normalize.Height = g.h;
    normalize.MvScaleX = Float(source, NVSDK_NGX_Parameter_MV_Scale_X, 1) / g.w;
    normalize.MvScaleY = Float(source, NVSDK_NGX_Parameter_MV_Scale_Y, 1) / g.h;
    bool ok = g.codec->DispatchPass(cmd, normalize, motion, nullptr, nullptr, nullptr, nullptr, h.motion, nullptr);
    Barrier(cmd, h.motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    h.motionReadable = true;
    if (!ok) { h.Reset(); g.halfStatus = "motion normalization failed"; return false; }
    // Skipped frames only need their normalized one-frame field saved for the
    // next anchor. Composing two-frame motion here would never be consumed.
    if (h.havePrevious && !h.previousWasAnchor)
    {
        if (h.anchorReadable) Barrier(cmd, h.anchorMotion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        normalize.Mode = DlssNrMode_ComposeMotion;
        ok = g.codec->DispatchPass(cmd, normalize, h.motion, h.previousMotion, nullptr, nullptr, nullptr, h.anchorMotion, nullptr);
        Barrier(cmd, h.anchorMotion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.anchorReadable = true;
        if (!ok) { h.Reset(); g.halfStatus = "motion composition failed"; return false; }
    }
    g.halfStatus.clear();
    return true;
}

void Before(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source,
            unsigned long long epoch, unsigned long long submittedEpoch,
            ID3D12CommandQueue* queue, bool privateJob = false)
{
    if (pending.cmd && current)
    {
        LOG_DEBUG("DLSS-NR deferred: Before entry with a stale pending (previous After never ran) -> reset. epoch {}", epoch);
        current->reset = true; // abandoned/failed main SR call
    }
    pending = {};
    struct ResetOnGap
    {
        unsigned long long epoch;
        ~ResetOnGap()
        {
            if (!pending.cmd && current)
            {
                LOG_DEBUG("DLSS-NR deferred: Before returned without arming a seam -> reset + hold/half cleared. epoch {}", epoch);
                current->reset = true; current->hold.Reset(); if (current->half) current->half->Reset();
            }
        }
    } resetOnGap { epoch };
    Collect();
    const auto& cfg = *Config::Instance();
    if (cfg.DlssNrUseProxy.value_or_default() || cfg.DlssNrHoldFrame.value_or_default() ||
        cfg.DlssNrDebugView.value_or_default() != 0 || cfg.DlssNrCompare.value_or_default() != 0 ||
        cfg.DlssNrShowSkinMask.value_or_default() ||
        (!cfg.DlssNrApplyModel.value_or_default() && !cfg.DlssNrFinishedPicture.value_or_default()))
    {
        Say("inactive: disable proxy backend, frame hold/debug/compare, and enable Apply model");
        return;
    }
    if (cmd->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT ||
        (!privateJob && (cfg.RestoreComputeSignature.value_or_default() || cfg.RestoreGraphicSignature.value_or_default()) &&
         !D3D12Hooks::CanRestoreRootSignature(cmd)))
    {
        Say("inactive: requires a direct command list with restorable game state");
        return;
    }
    auto* color = GetResource(source, NVSDK_NGX_Parameter_Color, "DLSSD.Color");
    auto* output = GetResource(source, NVSDK_NGX_Parameter_Output, "DLSSD.Output");
    auto* depth = GetResource(source, NVSDK_NGX_Parameter_Depth, "DLSSD.Depth");
    auto* motion = GetResource(source, NVSDK_NGX_Parameter_MotionVectors, "DLSSD.MotionVectors");
    const bool wantsHalf = cfg.DlssNrResidualFg.value_or_default() && !privateJob &&
                           !cfg.DlssNrFinishedPicture.value_or_default();
    if (cfg.DlssNrResidualFg.value_or_default() && !wantsHalf)
    {
        static int lastBlock = -1;
        const int block = privateJob ? 1 : (cfg.DlssNrFinishedPicture.value_or_default() ? 2 : 0);
        if (block != lastBlock)
        {
            lastBlock = block;
            LOG_WARN("DLSS-NR half-rate requested but blocked before activation: {}",
                     privateJob ? "private bridge job" : "finished-picture mode is enabled");
        }
    }
    const bool sampleAndHold = wantsHalf && motion == nullptr;
    if (!color || !output || !depth || (!motion && !sampleAndHold) || color == output)
    {
        Say("inactive: distinct Color/Output, depth and motion are required");
        return;
    }
    for (const char* key : { NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,
         NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X,
         NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,
         NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X,
         NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y })
        if (UInt(source, key) != 0) { Say("inactive: non-zero colour/guide/output offsets"); return; }
    const auto inDesc = color->GetDesc(), outDesc = output->GetDesc();
    const auto active = PreSrColorExtent(inDesc,
        UInt(source, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width),
        UInt(source, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height));
    if (!active || !PreSrColorExtent(outDesc, 0, 0) || inDesc.MipLevels != 1 || outDesc.MipLevels != 1 ||
        active->width > outDesc.Width || active->height > outDesc.Height)
    {
        Say("inactive: unsupported active input/output dimensions");
        return;
    }
    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device)))) return;
    auto* ownerQueue = queue ? queue : (ID3D12CommandQueue*)State::Instance().currentCommandQueue;
    ID3D12Device* queueDevice = nullptr;
    if (!ownerQueue || ownerQueue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT ||
        FAILED(ownerQueue->GetDevice(IID_PPV_ARGS(&queueDevice))) || queueDevice != device)
    {
        if (queueDevice) queueDevice->Release();
        device->Release();
        Say("waiting for a same-device direct queue identity");
        return;
    }
    queueDevice->Release();
    unsigned flags = UInt(source, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags) &
        (NVSDK_NGX_DLSS_Feature_Flags_DepthInverted | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
         NVSDK_NGX_DLSS_Feature_Flags_MVJittered);
    if (sampleAndHold)
        flags = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (current && (current->device != device || current->queue != ownerQueue || current->w != active->width ||
        current->h != active->height || current->outW != outDesc.Width || current->outH != outDesc.Height ||
        current->inputFormat != inDesc.Format || current->outputFormat != outDesc.Format || current->flags != flags ||
        current->halfRequested != wantsHalf ||
        current->sampleAndHold != sampleAndHold))
        retired.push_back(std::move(current));
    if (!current)
    {
        if (retired.size() >= 4) { device->Release(); Say("waiting for retired GPU work; clean SR frame retained"); return; }
        current = std::make_unique<Generation>();
        current->device = device; // take the GetDevice reference
        current->queue = ownerQueue;
        ownerQueue->AddRef();
        current->w = active->width; current->h = active->height;
        current->outW = (unsigned)outDesc.Width; current->outH = outDesc.Height;
        current->inputFormat = inDesc.Format; current->outputFormat = outDesc.Format; current->flags = flags;
        current->halfRequested = wantsHalf;
        current->sampleAndHold = sampleAndHold;
        if (!Allocate(*current)) { current->failed = true; Say("allocation failed; clean SR frame retained"); return; }
    }
    else device->Release();
    auto& g = *current;
    if (g.failed) return;
    // Native seams have a logical per-evaluate identity. Bridges retain the submitted epoch
    // so a second upscale in the same bridge submission is still rejected.
    if (g.began && g.lastBeginEpoch == epoch)
    { g.reset = true; Say("inactive: more than one upscale in a submission epoch"); return; }
    g.began = true;
    g.lastBeginEpoch = epoch;
    Use use(g, cmd);
    if (!use.valid) { Say("waiting for GPU completion slots; clean SR frame retained"); return; }
    if (!g.feature)
    {
        ScopedNrStateEnvelope envelope(cmd);
        if (!NVNGXProxy::InitDx12(g.device) || !NVNGXProxy::D3D12_AllocateParameters() ||
            !NVNGXProxy::D3D12_DestroyParameters() || !NVNGXProxy::D3D12_CreateFeature() ||
            !NVNGXProxy::D3D12_EvaluateFeature() || !NVNGXProxy::D3D12_ReleaseFeature() ||
            NVNGXProxy::D3D12_AllocateParameters()(&g.parameters) != NVSDK_NGX_Result_Success || !g.parameters)
        { g.failed = true; Say("NVIDIA DLSS SR runtime unavailable; no alternative upscaler used"); return; }
        auto* p = g.parameters;
        p->Set(NVSDK_NGX_Parameter_Width, g.w); p->Set(NVSDK_NGX_Parameter_Height, g.h);
        p->Set(NVSDK_NGX_Parameter_OutWidth, g.outW); p->Set(NVSDK_NGX_Parameter_OutHeight, g.outH);
        p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u); p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
        p->Set(NVSDK_NGX_Parameter_PerfQualityValue, (int)UInt(source, NVSDK_NGX_Parameter_PerfQualityValue,
                                                           NVSDK_NGX_PerfQuality_Value_MaxPerf));
        // LDR biased carrier, constant unit exposure, no auto-exposure/sharpening. No main-game presets
        // or feature handle are overwritten. NGX is called directly, bypassing OptiScaler's NR hooks.
        p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, g.flags);
        const auto result = NVNGXProxy::D3D12_CreateFeature()(cmd, NVSDK_NGX_Feature_SuperSampling, p, &g.feature);
        if (result != NVSDK_NGX_Result_Success || !g.feature)
        { g.failed = true; Say("private DLSS creation failed: " + std::to_string((unsigned)result)); return; }
        DlssNrConstants unit {}; unit.Mode = DlssNrMode_UnitExposure; unit.Width = unit.Height = 1;
        if (!g.codec->DispatchPass(cmd, unit, g.edited, nullptr, nullptr, nullptr, nullptr, g.exposure, nullptr))
        { g.failed = true; Say("private exposure initialization failed"); return; }
        Barrier(cmd, g.exposure, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (g.sampleAndHold)
        {
            g.zeroMotion = CreateScratch(g.device, DXGI_FORMAT_R16G16_FLOAT, g.w, g.h);
            unit.Mode = DlssNrMode_ZeroMotion; unit.Width = g.w; unit.Height = g.h;
            if (!g.zeroMotion || !g.codec->DispatchPass(cmd, unit, g.edited, nullptr, nullptr, nullptr,
                                                       nullptr, g.zeroMotion, nullptr))
            { g.failed = true; Say("sample-and-hold guide initialization failed"); return; }
            Barrier(cmd, g.zeroMotion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        g.createEpoch = submittedEpoch;
        Say("private DLSS created; waiting for a later submission epoch");
        return;
    }
    // Synthetic seam ticks cannot prove that a feature's creation commands were submitted.
    if (submittedEpoch == g.createEpoch)
    { LOG_DEBUG("DLSS-NR deferred: waiting after feature creation at submitted epoch {}", submittedEpoch); return; }

    if (g.sampleAndHold)
    {
        if (g.reset || UInt(source, NVSDK_NGX_Parameter_Reset)) g.hold.Reset();
        if (g.hold.CanReuse(epoch))
        {
            pending = {cmd, source, output, epoch,
                       std::max(Float(source, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1), 1e-4f), true, false};
            return; // Apply the held residual to CURRENT clean SR, not a delayed raster.
        }
        g.hold.Reset();
        motion = g.zeroMotion; // private NR/SR only, with temporal history reset below.
    }

    bool half = false;
    if (!g.sampleAndHold)
    {
        ScopedNrStateEnvelope envelope(cmd);
        half = PrepareHalfRate(g, cmd, source, motion, epoch, submittedEpoch);
    }
    if (!half && g.half) g.half->Reset();
    if (half && g.half->havePrevious && g.half->previousWasAnchor)
    {
        pending = { cmd, source, output, epoch,
                    std::max(Float(source, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1), 1e-4f), true, true };
        return; // NR and private residual SR are both skipped; game SR still runs normally.
    }
    auto* nrMotion = half ? (g.half->havePrevious ? g.half->anchorMotion : g.half->motion) : motion;

    const auto arrival = cfg.ColorResourceBarrier.has_value() ?
        (D3D12_RESOURCE_STATES)cfg.ColorResourceBarrier.value() : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    Barrier(cmd, color, arrival, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(cmd, g.edited, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    CopyActiveColor(cmd, g.edited, color, *active);
    Barrier(cmd, color, D3D12_RESOURCE_STATE_COPY_SOURCE, arrival);
    Barrier(cmd, g.edited, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    DlssNrFrameInfo frame {};
    frame.BeforeUpscale = frame.PrivateColorCopy = true;
    frame.IndependentCommands = privateJob;
    frame.SubmissionEpoch = submittedEpoch;
    frame.RenderSubrectWidth = g.w; frame.RenderSubrectHeight = g.h;
    frame.DepthInverted = (flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0;
    frame.ColourIsLinearHdr = (UInt(source, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags) &
        NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0 && FormatCanHoldLinearHdr(outDesc.Format);
    frame.Reset = UInt(source, NVSDK_NGX_Parameter_Reset) != 0 || g.reset || g.sampleAndHold || privateJob;
    frame.MvScaleX = Float(source, NVSDK_NGX_Parameter_MV_Scale_X, 1);
    frame.MvScaleY = Float(source, NVSDK_NGX_Parameter_MV_Scale_Y, 1);
    if (half) { frame.MvScaleX = (float)g.w; frame.MvScaleY = (float)g.h; }
    if (g.sampleAndHold) frame.MvScaleX = frame.MvScaleY = 1.0f;
    frame.PreExposure = std::max(Float(source, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1), 1e-4f);
    frame.ExposureTexture = GetResource(source, NVSDK_NGX_Parameter_ExposureTexture, "ExposureTexture");
    g_nr.exposureOfferedNow = frame.ExposureTexture != nullptr;
    g_nr.exposureEverOffered = g_nr.exposureEverOffered || g_nr.exposureOfferedNow;
    ++g_nr.exposureFrames;
    if (!g_compose) g_compose = std::make_unique<DlssNr_Dx12>("Neural Rendering", g.device);
    const auto before = g_nr.successfulDispatches;
    g_compose->Dispatch(cmd, g.edited, depth, nrMotion, g.edited, frame, queue);
    const bool evaluated = g_nr.successfulDispatches != before;
    if (evaluated)
    {
        ScopedNrStateEnvelope envelope(cmd);
        if (g.smallReadable)
            Barrier(cmd, g.residualInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(cmd, color, arrival, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        DlssNrConstants encode {}; encode.Mode = DlssNrMode_EncodeResidual;
        encode.Width = g.w; encode.Height = g.h; encode.ExposurePreMul = frame.PreExposure;
        bool ok;
        if (cfg.DlssNrFinishedPicture.value_or_default())
        {
            encode.Mode = 5; // finished-colour shader: encode relative changes before FP16 storage
            encode.WhitePoint = frame.PreExposure;
            encode.TransferStrength = frame.ColourIsLinearHdr ? 1.0f : 0.0f;
            encode.MaxRatio = std::clamp(cfg.DlssNrMaxRatio.value_or_default(), 1.0f, 8.0f);
            ok = g.codec->DispatchResidualPass(cmd, encode, color, g.edited, nullptr, nullptr, g.residualInput, true);
        }
        else
            ok = g.codec->DispatchPass(cmd, encode, color, g.edited, nullptr, nullptr, nullptr, g.residualInput, nullptr);
        Barrier(cmd, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, arrival);
        Barrier(cmd, g.residualInput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        g.smallReadable = true;
        if (ok)
        {
            auto* p = g.parameters;
            p->Set(NVSDK_NGX_Parameter_Color, g.residualInput); p->Set(NVSDK_NGX_Parameter_Output, g.residualOutput);
            p->Set(NVSDK_NGX_Parameter_Depth, depth); p->Set(NVSDK_NGX_Parameter_MotionVectors, nrMotion);
            p->Set(NVSDK_NGX_Parameter_ExposureTexture, g.exposure);
            p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, g.w);
            p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, g.h);
            p->Set(NVSDK_NGX_Parameter_Reset, (unsigned)(frame.Reset || g.reset));
            p->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, Float(source, NVSDK_NGX_Parameter_Jitter_Offset_X, 0));
            p->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, Float(source, NVSDK_NGX_Parameter_Jitter_Offset_Y, 0));
            p->Set(NVSDK_NGX_Parameter_MV_Scale_X, frame.MvScaleX);
            p->Set(NVSDK_NGX_Parameter_MV_Scale_Y, frame.MvScaleY);
            p->Set(NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, Float(source, NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, 16.67f) *
                   (half && g.half->havePrevious ? 2.0f : 1.0f));
            p->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
            p->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
            p->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
            pending = { cmd, source, output, epoch, frame.PreExposure, false, half };
            LOG_DEBUG("DLSS-NR deferred: Before armed epoch {} half {} preExp {:.4f} reset-carried {}", epoch, half,
                      frame.PreExposure, frame.Reset);
        }
    }
    else { g.reset = true; Say("waiting for NR evaluation; clean SR frame retained"); }
    Barrier(cmd, g.edited, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

// Background GPU job: produce only the DLSS-upscaled residual. No raster composition,
// regular FG call, or presentation operation is recorded on this queue.
bool ResolvePrivate(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source,
                    [[maybe_unused]] unsigned long long epoch, ID3D12Resource* destination)
{
    const auto pair = pending; pending = {};
    // No epoch match, as in After() -- see the comment there. (Dead path on this branch: async NR
    // was removed in v0.7.1 and nothing calls ResolvePrivate; kept consistent for a future revival.)
    if (!current || current->failed || pair.cmd != cmd || pair.caller != source)
        return false;
    auto& g = *current;
    Use use(g, cmd);
    if (!use.valid) { g.reset = true; return false; }
    const auto result = NVNGXProxy::D3D12_EvaluateFeature()(cmd, g.feature, g.parameters, nullptr);
    if (result != NVSDK_NGX_Result_Success)
    { g.failed = true; Say("asynchronous residual DLSS evaluation failed"); return false; }
    Barrier(cmd, g.residualOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(cmd, destination, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd->CopyResource(destination, g.residualOutput);
    Barrier(cmd, destination, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    Barrier(cmd, g.residualOutput, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    g.reset = false;
    return true;
}

void After(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Parameter* source, unsigned long long epoch)
{
    const auto pair = pending;
    pending = {}; // Consume once, only for the immediately matching successful upscale.
    // Match and consume the immediately preceding Before by resource identity, not Present timing.
    // The pending epoch also owns history continuity if Present changed while DLSS was recording.
    if (!current || current->failed || pair.cmd != cmd || pair.caller != source ||
        pair.output != GetResource(source, NVSDK_NGX_Parameter_Output, "DLSSD.Output"))
    {
        LOG_DEBUG("DLSS-NR deferred: After no-op -> reset. current={} failed={} cmdMatch={} callerMatch={} "
                  "epoch(pending/now)={}/{} outputMatch={}",
                  current != nullptr, current && current->failed, pair.cmd == cmd, pair.caller == source, pair.epoch,
                  epoch, pair.output == GetResource(source, NVSDK_NGX_Parameter_Output, "DLSSD.Output"));
        if (current) current->reset = true;
        return;
    }
    auto& g = *current;
    const auto& cfg = *Config::Instance();
    if ((cfg.RestoreComputeSignature.value_or_default() || cfg.RestoreGraphicSignature.value_or_default()) &&
        !D3D12Hooks::CanRestoreRootSignature(cmd))
    { g.reset = true; Say("inactive: game state cannot be restored after SR"); return; }
    Use use(g, cmd);
    if (!use.valid) { g.reset = true; Say("waiting for GPU completion slots; clean SR frame retained"); return; }
    ScopedNrStateEnvelope envelope(cmd);
    if (!pair.skipNr)
    {
        const auto result = NVNGXProxy::D3D12_EvaluateFeature()(cmd, g.feature, g.parameters, nullptr);
        if (result != NVSDK_NGX_Result_Success)
        { g.failed = true; if (g.half) g.half->Reset(); Say("private DLSS evaluation failed: " + std::to_string((unsigned)result)); return; }
    }
    if (g.reset)
        LOG_DEBUG("DLSS-NR deferred: After fed Reset=1 to the private DLSS SR this frame (history restart). "
                  "epoch {} skipNr {} half {}", epoch, pair.skipNr, pair.half);
    else
        LOG_TRACE("DLSS-NR deferred: After applied epoch {} skipNr {} half {}", epoch, pair.skipNr, pair.half);
    g.reset = false;
    if (cfg.DlssNrFinishedPicture.value_or_default())
    {
        const bool sceneLinear = (UInt(source, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags) &
            NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0 && FormatCanHoldLinearHdr(g.outputFormat);
        if (Late::CaptureResidual(cmd, pair.output, g.residualOutput, pair.scale, sceneLinear))
            Say("running: model before SR; changes saved for the finished picture");
        else { g.reset = true; Say("waiting to save the upscaled changes for the finished picture"); }
        return; // Keep the game's SR output clean: no early composition and no second NR evaluation.
    }

    Barrier(cmd, g.residualOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    bool half = pair.half && g.half && !g.half->failed;
    const auto arrival = cfg.OutputResourceBarrier.has_value() ?
        (D3D12_RESOURCE_STATES)cfg.OutputResourceBarrier.value() : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    Barrier(cmd, pair.output, arrival, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Barrier(cmd, g.clean, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    cmd->CopyResource(g.clean, pair.output);
    Barrier(cmd, g.clean, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12Resource* residual = g.residualOutput;
    ID3D12Resource* fallbackResidual = nullptr;
    ID3D12Resource* fallbackMotion = nullptr;
    DlssNrConstants apply {}; apply.Mode = DlssNrMode_ApplyResidual;
    apply.Width = g.outW; apply.Height = g.outH; apply.ExposurePreMul = pair.scale;
    if (half && pair.skipNr)
    {
        auto& h = *g.half;
        // Keep the game's raster current. Only the previous NR edit is brought
        // forward by the skipped frame's current-to-previous motion field.
        apply.Mode = DlssNrMode_ApplyReprojectedResidual;
        fallbackResidual = h.previousResidual;
        fallbackMotion = h.motion;
    }
    const bool ok = g.codec->DispatchPass(cmd, apply, g.clean, residual, fallbackResidual, fallbackMotion,
                                         nullptr, g.composed, nullptr);
    if (ok)
    {
        Barrier(cmd, g.composed, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(cmd, pair.output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(pair.output, g.composed);
        Barrier(cmd, pair.output, D3D12_RESOURCE_STATE_COPY_DEST, arrival);
        Barrier(cmd, g.composed, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Say(g.sampleAndHold ? "running: sample-and-hold (motion unavailable); each NR residual applied to 2 current frames; no raster delay" :
            half ? "running: NR every second frame; previous NR reprojected onto CURRENT clean raster; no raster delay" :
            "running: " + std::to_string(g.w) + "x" + std::to_string(g.h) + " contribution -> private DLSS -> " +
            std::to_string(g.outW) + "x" + std::to_string(g.outH) + "; applied after SR" +
            (g.halfRequested ? "; half-rate inactive: " + g.halfStatus : ""));
    }
    else { Barrier(cmd, pair.output, D3D12_RESOURCE_STATE_COPY_SOURCE, arrival); Say("composition failed; clean frame retained"); }
    Barrier(cmd, g.clean, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (half && ok && !pair.skipNr)
    {
        auto& h = *g.half;
        // Preserve this anchor only AFTER the previous anchor has been consumed
        // by the midpoint fallback. Known cuts reset havePrevious before use.
        Barrier(cmd, g.residualOutput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(cmd, h.previousResidual, h.previousResidualReadable ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE :
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(h.previousResidual, g.residualOutput);
        Barrier(cmd, h.previousResidual, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(cmd, g.residualOutput, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.previousResidualReadable = true;
    }
    Barrier(cmd, g.residualOutput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (g.sampleAndHold)
    {
        if (ok && !pair.skipNr) g.hold.SampleSucceeded(pair.epoch);
        else g.hold.Reset();
    }
    if (half)
    {
        auto& h = *g.half;
        Barrier(cmd, h.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Barrier(cmd, h.previousMotion, h.previousReadable ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE :
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(h.previousMotion, h.motion);
        Barrier(cmd, h.previousMotion, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(cmd, h.motion, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        h.previousReadable = true; h.havePrevious = ok; h.previousWasAnchor = !pair.skipNr;
        if (ok) { if (pair.skipNr) ++h.skippedNr; else ++h.nrAnchors; }
        // Independent of FG readback or a first successful skip: the old cadence
        // message never fired in the broken 313-anchor / zero-skip sequence.
        if (ok && (h.nrAnchors + h.skippedNr) % 240 == 0)
            LOG_INFO("Current-raster half-rate cadence: {} NR anchors, {} NR skips, {} render-epoch gaps; "
                     "render epoch {}, current Present counter {}", h.nrAnchors, h.skippedNr,
                     h.continuityBreaks, pair.epoch,
                     State::Instance().frameCount);
        if (h.skippedNr == 8 && pair.skipNr)
            LOG_INFO("Current-raster half-rate cadence: {} NR anchor frames, {} skipped NR frames; motion reprojection active",
                     h.nrAnchors, h.skippedNr);
        if (!ok) { h.Reset(); g.reset = true; }
    }
}

void Shutdown()
{
    Cancel();
    if (current) retired.push_back(std::move(current));
    Collect();
    // Never free feature histories, descriptors or surfaces referenced by an unsubmitted/in-flight
    // list. At shutdown only, retain uncompleted generations for process teardown rather than UAF.
    for (auto& g : retired) (void)g.release();
    retired.clear();
}
} // namespace DeferredSr

std::string SynchronousDeferredDlssStatus()
{
    std::lock_guard<std::recursive_mutex> lock(g_nrMutex);
    auto result = DeferredSr::status;
    const auto& cfg = *Config::Instance();
    if (cfg.DlssNrResidualFg.value_or_default())
    {
        if (cfg.DlssNrFinishedPicture.value_or_default())
            result += " (half-rate BLOCKED: finished-picture mode is enabled)";
        else if (!DeferredSr::current)
            result += " (half-rate requested; waiting for deferred SR)";
        else if (DeferredSr::current->sampleAndHold)
            result += " (sample-and-hold; motion reprojection unavailable)";
        else if (!DeferredSr::current->half)
            result += " (half-rate requested; " + (DeferredSr::current->halfStatus.empty() ?
                      std::string("waiting for half-rate resources") : DeferredSr::current->halfStatus) + ")";
        else
        {
            const auto& h = *DeferredSr::current->half;
            result += " (NR frames " + std::to_string(h.nrAnchors) + ", reprojected skips " +
                      std::to_string(h.skippedNr);
            if (!DeferredSr::current->halfStatus.empty())
                result += "; " + DeferredSr::current->halfStatus;
            result += ")";
        }
    }
    return result;
}
