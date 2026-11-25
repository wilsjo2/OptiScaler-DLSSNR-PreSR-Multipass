#include <pch.h>

#include "DLSSFeature_Dx11On12.h"
#include <upscalers/dlss/DLSSFeature_Dx12.h>

// No SetParameters here, deliberately: the FFX bridge variants set "OptiScaler.SupportsUpscaleSize", an FSR 3.1
// escape hatch that makes GetDynamicOutputResolution return without resolving the output size and leaves DLSS
// driven at the wrong resolution.
DLSSFeatureDx11On12::DLSSFeatureDx11On12(unsigned int InHandleId, NVSDK_NGX_Parameter* InParameters,
                                         const std::string& customUpscaler)
    : IFeature_Dx11wDx12(InHandleId, InParameters), IFeature_Dx11(InHandleId, InParameters),
      IFeature(InHandleId, InParameters)
{
    dx12Feature = std::make_unique<DLSSFeatureDx12>(InHandleId, InParameters, customUpscaler);
}
