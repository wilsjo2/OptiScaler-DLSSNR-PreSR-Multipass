#pragma once

#include "../shaders/dlssnr/DlssNr_Guides.h"
#include <nvsdk_ngx_params.h>

namespace DlssNr
{
struct ModelSettings
{
    unsigned int preset = 0, style = 0;
    float intensity = 1.0f, localStructure = 1.0f, localTone = 0.0f, skinStructure = -1.0f;
    bool autoMask = true;
    bool operator==(const ModelSettings&) const = default;
};

inline void SetModelTuning(NVSDK_NGX_Parameter* params, const ModelSettings& settings)
{
    params->Set("DLSSNR.Intensity", settings.intensity);
    params->Set("DLSSNR.Style", settings.style);
    params->Set("DLSSNR.LocalStructureStrength", settings.localStructure);
    params->Set("DLSSNR.LocalToneStrength", settings.localTone);
    params->Set("DLSSNR.SkinStructureStrength", settings.skinStructure);
    params->Set("DLSSNR.UseAutoMask", settings.autoMask ? 1u : 0u);
}

inline void SetModelRegions(NVSDK_NGX_Parameter* params, GuideExtent size, const GuideRegions& guides)
{
    params->Set("DLSSNR.ColorSubrectBaseX", 0u);
    params->Set("DLSSNR.ColorSubrectBaseY", 0u);
    params->Set("DLSSNR.ColorSubrectWidth", size.width);
    params->Set("DLSSNR.ColorSubrectHeight", size.height);
    params->Set("DLSSNR.OutputSubrectBaseX", 0u);
    params->Set("DLSSNR.OutputSubrectBaseY", 0u);
    params->Set("DLSSNR.OutputSubrectWidth", size.width);
    params->Set("DLSSNR.OutputSubrectHeight", size.height);
    params->Set("DLSSNR.DepthSubrectBaseX", guides.depth.x);
    params->Set("DLSSNR.DepthSubrectBaseY", guides.depth.y);
    params->Set("DLSSNR.DepthSubrectWidth", guides.depth.width);
    params->Set("DLSSNR.DepthSubrectHeight", guides.depth.height);
    params->Set("DLSSNR.MVecSubrectBaseX", guides.motion.x);
    params->Set("DLSSNR.MVecSubrectBaseY", guides.motion.y);
    params->Set("DLSSNR.MVecSubrectWidth", guides.motion.width);
    params->Set("DLSSNR.MVecSubrectHeight", guides.motion.height);
}
} // namespace DlssNr
