#pragma once

#include <array>
#include <nvsdk_ngx.h>

// The held frame keeps its sampling/exposure metadata. Original call parameters
// are restored after NR and SR, including failed evaluates.
class NrHoldParameters_Dx12
{
    template <typename T> struct Value
    {
        const char* name;
        T held = 0, live = 0;
        bool valid = false, changed = false;
    };
    std::array<Value<float>, 7> values { { { NVSDK_NGX_Parameter_Jitter_Offset_X },
                                           { NVSDK_NGX_Parameter_Jitter_Offset_Y },
                                           { NVSDK_NGX_Parameter_MV_Scale_X },
                                           { NVSDK_NGX_Parameter_MV_Scale_Y },
                                           { NVSDK_NGX_Parameter_DLSS_Pre_Exposure },
                                           { NVSDK_NGX_Parameter_DLSS_Exposure_Scale },
                                           { NVSDK_NGX_Parameter_FrameTimeDeltaInMsec } } };
    std::array<Value<unsigned int>, 8> regions { { { NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width },
                                                   { NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height },
                                                   { NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X },
                                                   { NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y },
                                                   { NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X },
                                                   { NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y },
                                                   { NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X },
                                                   { NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y } } };
    unsigned int liveReset = 0;
    bool resetChanged = false;

    void ForEach(auto&& operation)
    {
        for (auto& value : values)
            operation(value);
        for (auto& region : regions)
            operation(region);
    }

  public:
    void Capture(NVSDK_NGX_Parameter* params)
    {
        ForEach([&](auto& value) { value.valid = params->Get(value.name, &value.held) == NVSDK_NGX_Result_Success; });
    }
    void Apply(NVSDK_NGX_Parameter* params, bool frozen = true)
    {
        ForEach(
            [&](auto& value)
            {
                value.changed =
                    frozen && value.valid && params->Get(value.name, &value.live) == NVSDK_NGX_Result_Success;
                if (value.changed)
                    params->Set(value.name, value.held);
            });
        liveReset = 0;
        params->Get(NVSDK_NGX_Parameter_Reset, &liveReset);
        params->Set(NVSDK_NGX_Parameter_Reset, 1u);
        resetChanged = true;
    }
    void Restore(NVSDK_NGX_Parameter* params)
    {
        ForEach(
            [&](auto& value)
            {
                if (value.changed)
                    params->Set(value.name, value.live);
                value.changed = false;
            });
        if (resetChanged)
            params->Set(NVSDK_NGX_Parameter_Reset, liveReset);
        resetChanged = false;
    }
};
