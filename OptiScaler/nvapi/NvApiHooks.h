#pragma once

#include <hooks/Reflex_Hooks.h>

#include "NvApiTypes.h"
#include "fakenvapi.h"

class NvApiHooks
{
  public:
    inline static PFN_NvApi_QueryInterface o_NvAPI_QueryInterface = nullptr;
    inline static decltype(&NvAPI_GPU_GetArchInfo) o_NvAPI_GPU_GetArchInfo = nullptr;
    inline static decltype(&NvAPI_DRS_GetSetting) o_NvAPI_DRS_GetSetting = nullptr;

    // inline static UINT64 qiCounter = 0;

    static NvAPI_Status __stdcall hkNvAPI_GPU_GetArchInfo(NvPhysicalGpuHandle hPhysicalGpu,
                                                          NV_GPU_ARCH_INFO* pGpuArchInfo);
    static NvAPI_Status __stdcall hkNvAPI_DRS_GetSetting(NvDRSSessionHandle hSession, NvDRSProfileHandle hProfile,
                                                         NvU32 settingId, NVDRS_SETTING* pSetting);
    static NvAPI_Status __stdcall hkNvAPI_D3D12_SetFlipConfig(void* pCommandQueue, NvU32 dwFlags, void* pParams);
    static void* __stdcall hkNvAPI_QueryInterface(unsigned int InterfaceId);
    static void Hook(HMODULE nvapiModule);
    static void Unhook();
};
