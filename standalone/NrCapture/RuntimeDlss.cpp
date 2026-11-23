#include "Runtime.h"
namespace nr {
void Runtime::ReleaseDlss() {
    if (dlssFeature) { release(dlssFeature); dlssFeature = nullptr; }
    if (dlssParameters) { destroy(dlssParameters); dlssParameters = nullptr; }
}
NVSDK_NGX_Result Runtime::CreateDlss(ID3D12GraphicsCommandList* commands, UINT w, UINT h, UINT outW, UINT outH) {
    ReleaseDlss();
    InitializeDriver();
    if (!std::filesystem::exists(dlssPath))
        throw std::runtime_error("Private DLSS needs the original NVIDIA-signed nvngx_dlss.dll beside the NR runtime or in this application's runtime folder.");
    VerifyNvidia(dlssPath);
    CheckResult(allocate(&dlssParameters), "Private DLSS parameter allocation");
    if (!dlssParameters) throw std::runtime_error("Private DLSS returned no parameter map.");
    auto* p = dlssParameters; dlssWidth = w; dlssHeight = h;
    p->Set(NVSDK_NGX_Parameter_Width, w); p->Set(NVSDK_NGX_Parameter_Height, h);
    p->Set(NVSDK_NGX_Parameter_OutWidth, outW); p->Set(NVSDK_NGX_Parameter_OutHeight, outH);
    p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u); p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
    p->Set(NVSDK_NGX_Parameter_PerfQualityValue, (int)NVSDK_NGX_PerfQuality_Value_MaxQuality);
    // Bounded relative-lighting/chroma carrier, model-resolution motion, unit exposure. No RR or auto exposure.
    p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, (int)NVSDK_NGX_DLSS_Feature_Flags_MVLowRes);
    const auto result = create(commands, NVSDK_NGX_Feature_SuperSampling, p, &dlssFeature);
    Log(std::format("Private DLSS SR {}x{} -> {}x{}: 0x{:08X}; runtime {}", w, h, outW, outH,
        (unsigned)result, Narrow(dlssPath.wstring())));
    return NVSDK_NGX_SUCCEED(result) && !dlssFeature ? NVSDK_NGX_Result_Fail : result;
}
void Runtime::EvaluateDlss(ID3D12GraphicsCommandList* commands, ID3D12Resource* colour, ID3D12Resource* depth,
    ID3D12Resource* motion, ID3D12Resource* exposure, ID3D12Resource* output, bool reset, float frameTimeMs) {
    if (!dlssFeature) throw std::runtime_error("Private DLSS has no valid feature.");
    auto* p = dlssParameters;
    p->Set(NVSDK_NGX_Parameter_Color, colour); p->Set(NVSDK_NGX_Parameter_Output, output);
    p->Set(NVSDK_NGX_Parameter_Depth, depth); p->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
    p->Set(NVSDK_NGX_Parameter_ExposureTexture, exposure);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, dlssWidth);
    p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, dlssHeight);
    p->Set(NVSDK_NGX_Parameter_Reset, (unsigned)reset);
    // Capture is already de-jittered; do not invent camera jitter.
    p->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f); p->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    p->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f); p->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    p->Set(NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, frameTimeMs);
    p->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f); p->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    p->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    CheckResult(evaluate(commands, dlssFeature, p, nullptr), "Private DLSS evaluation");
}
}
