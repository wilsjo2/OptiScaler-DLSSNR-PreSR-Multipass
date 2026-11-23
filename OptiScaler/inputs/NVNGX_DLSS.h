#pragma once
#include "NVNGX_Parameter.h"

template <typename FeatureType> struct ContextData
{
    std::unique_ptr<FeatureType> feature;
    std::unique_ptr<NVNGX_Parameters> ownedCreateParams;
    NVSDK_NGX_Parameter* createParams = nullptr; // May also refer to the caller's passthrough parameters.
    void ClearCreateParams()
    {
        createParams = nullptr;
        ownedCreateParams.reset();
    }
    int changeBackendCounter = 0;
};
