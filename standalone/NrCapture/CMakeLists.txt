cmake_minimum_required(VERSION 3.24)
project(OptiScalerNR LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(REPO "../.." ABSOLUTE)
add_library(nr_capture_core STATIC Common.cpp Colour.cpp Gpu.cpp GpuTimings.cpp Runtime.cpp RuntimeDlss.cpp Processor.cpp ProcessorStages.cpp Motion.cpp Scaling.cpp Spatial.cpp CaptureQueue.cpp FrameGeneration.cpp FrameGenerationFocus.cpp FrameGenerationOutput.cpp
    AdaMfgUnlock.cpp nvfp4/NrFfnV2.cpp nvfp4/NrSwin4.cpp nvfp4/NrNet4.cpp
    "${REPO}/OptiScaler/dlssnr/DlssNr_GpuLifetime.cpp"
    "${REPO}/OptiScaler/dlssnr/DlssNr_CompatibilityRuntime.cpp"
    "${REPO}/OptiScaler/dlssnr/NrPassBorrow.cpp")
get_filename_component(WINDOWS_SDK_ROOT
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]" ABSOLUTE)
find_program(FXC fxc HINTS "${WINDOWS_SDK_ROOT}/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64" REQUIRED)
set(MOTION_SHADERS "${CMAKE_CURRENT_BINARY_DIR}/shaders")
foreach(ENTRY IN ITEMS Luma Estimate Resolve)
    set(DEFINES)
    if(ENTRY STREQUAL "Resolve")
        set(DEFINES /DRESOLVE=1)
    endif()
    set(HEADER "${MOTION_SHADERS}/Motion${ENTRY}.h")
    add_custom_command(OUTPUT "${HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
        COMMAND "${FXC}" /nologo /Tcs_5_0 /E${ENTRY} /O3 /VnMotion${ENTRY} /Fh "${HEADER}"
            ${DEFINES} "${CMAKE_CURRENT_SOURCE_DIR}/Motion.hlsl"
        DEPENDS Motion.hlsl VERBATIM)
    target_sources(nr_capture_core PRIVATE "${HEADER}")
endforeach()
add_custom_command(OUTPUT "${MOTION_SHADERS}/CaptureColour.h"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /ECaptureColour /O3 /VnCaptureColour /Fh "${MOTION_SHADERS}/CaptureColour.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CaptureColour.hlsl"
    DEPENDS CaptureColour.hlsl ModelResize.hlsli "${REPO}/OptiScaler/shaders/dlssnr/precompile/dlssnr.hlsl"
        "${REPO}/OptiScaler/shaders/dlssnr/precompile/dlssnr_quality.hlsli" VERBATIM)
target_sources(nr_capture_core PRIVATE "${MOTION_SHADERS}/CaptureColour.h")
add_custom_command(OUTPUT "${MOTION_SHADERS}/CaptureConvert.h"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /EConvert /O3 /VnCaptureConvert /Fh "${MOTION_SHADERS}/CaptureConvert.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CaptureConvert.hlsl"
    DEPENDS CaptureConvert.hlsl VERBATIM)
target_sources(nr_capture_core PRIVATE "${MOTION_SHADERS}/CaptureConvert.h")
add_custom_command(OUTPUT "${MOTION_SHADERS}/FrameGenerationEncode.h"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /EEncode /O3 /VnFrameGenerationEncode /Fh "${MOTION_SHADERS}/FrameGenerationEncode.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/FrameGenerationOutput.hlsl"
    DEPENDS FrameGenerationOutput.hlsl VERBATIM)
target_sources(nr_capture_core PRIVATE "${MOTION_SHADERS}/FrameGenerationEncode.h")
add_custom_command(OUTPUT "${MOTION_SHADERS}/SpatialPreview.h"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /EPreview /O3 /VnSpatialPreview /Fh "${MOTION_SHADERS}/SpatialPreview.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/SpatialPreview.hlsl"
    DEPENDS SpatialPreview.hlsl "${REPO}/OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_warp.hlsli" VERBATIM)
target_sources(nr_capture_core PRIVATE "${MOTION_SHADERS}/SpatialPreview.h")
target_include_directories(nr_capture_core PRIVATE "${MOTION_SHADERS}")
target_include_directories(nr_capture_core PUBLIC . "${REPO}/OptiScaler" "${REPO}/external/nvngx_dlss_sdk" "${REPO}/external/streamline" "${REPO}/external/nvapi"
    )
target_include_directories(nr_capture_core BEFORE PRIVATE adapters)
# Pass borrowing (NrPassBorrow.cpp) detours nvapi_QueryInterface; only the app and checks install it.
target_include_directories(nr_capture_core PRIVATE "${REPO}/OptiScaler/include" "${REPO}/OptiScaler/dlssnr")
target_compile_definitions(nr_capture_core PUBLIC UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(nr_capture_core PUBLIC /W4 /EHsc /permissive- /utf-8)
target_compile_options(nr_capture_core PUBLIC /Zi)
target_link_options(nr_capture_core INTERFACE /DEBUG)
target_link_libraries(nr_capture_core PUBLIC d3d12 d3d11 dxgi windowsapp dcomp
    wintrust crypt32 advapi32 shell32 ole32 user32 gdi32 comctl32 comdlg32 dwmapi windowscodecs
    "${REPO}/OptiScaler/library/detours/detours.lib")
add_executable(OptiScalerNR WIN32 App.cpp AppControls.cpp AppSession.cpp Capture.cpp WgcCapture.cpp DxgiCapture.cpp Presentation.cpp)
set_target_properties(OptiScalerNR PROPERTIES OUTPUT_NAME display_filter)
target_link_libraries(OptiScalerNR PRIVATE nr_capture_core)
# The executable names its symbol file without the build folder.
target_link_options(OptiScalerNR PRIVATE "/PDBALTPATH:%_PDB%")
add_executable(NrCaptureCheck Check.cpp ReapplyCheck.cpp BorrowCheck.cpp DepthCheck.cpp PassResolutionCheck.cpp QualityCheck.cpp QualityReference.cpp FeatureChecks.cpp HdrOutputCheck.cpp Capture.cpp WgcCapture.cpp DxgiCapture.cpp Presentation.cpp CaptureCheck.cpp MotionCheck.cpp GpuWaitCheck.cpp PerformanceCheck.cpp CaptureQueueCheck.cpp FrameGenerationCheck.cpp FrameGenerationPresentCheck.cpp ResidualCheck.cpp LoggingCheck.cpp)
target_include_directories(NrCaptureCheck BEFORE PRIVATE check_adapters adapters)
target_link_libraries(NrCaptureCheck PRIVATE nr_capture_core dbghelp)
# A local build may add commands: sources in lab/, a folder that is not part of the repository.
file(GLOB NR_LAB_SOURCES CONFIGURE_DEPENDS lab/*.cpp)
if(NR_LAB_SOURCES)
  target_sources(NrCaptureCheck PRIVATE ${NR_LAB_SOURCES})
  target_compile_definitions(NrCaptureCheck PRIVATE NR_LAB_COMMANDS)
endif()
add_executable(NrModelProbe ModelProbe.cpp ModelKernelTrace.cpp ModelOverlap.cpp ModelVitSequence.cpp ModelCapture.cpp Capture.cpp WgcCapture.cpp DxgiCapture.cpp
    nvfp4/DlssNrNative.cpp)
# The recovered v0.7.2 hybrid is kept as it was written; its warnings are not this target's to fix.
set_source_files_properties(nvfp4/DlssNrNative.cpp PROPERTIES COMPILE_OPTIONS "/W0")
add_custom_command(OUTPUT "${MOTION_SHADERS}/ModelOverlapResize.h"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /EPrepare /O3 /VnModelOverlapResize /Fh "${MOTION_SHADERS}/ModelOverlapResize.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ModelOverlapResize.hlsl"
    DEPENDS ModelOverlapResize.hlsl VERBATIM)
target_sources(NrModelProbe PRIVATE "${MOTION_SHADERS}/ModelOverlapResize.h")
add_custom_command(OUTPUT "${MOTION_SHADERS}/ModelLoopCopy.h"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /ECopyLatent /O3 /VnModelLoopCopy /Fh "${MOTION_SHADERS}/ModelLoopCopy.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ModelLoopCopy.hlsl"
    DEPENDS ModelLoopCopy.hlsl VERBATIM)
target_sources(NrModelProbe PRIVATE "${MOTION_SHADERS}/ModelLoopCopy.h")
add_custom_command(OUTPUT "${MOTION_SHADERS}/ModelBookendBridge.h"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /EBridgeBookends /O3 /VnModelBookendBridge /Fh "${MOTION_SHADERS}/ModelBookendBridge.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ModelBookendBridge.hlsl"
    DEPENDS ModelBookendBridge.hlsl VERBATIM)
target_sources(NrModelProbe PRIVATE "${MOTION_SHADERS}/ModelBookendBridge.h")
add_custom_command(OUTPUT "${MOTION_SHADERS}/ModelLatentMix.h"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MOTION_SHADERS}"
    COMMAND "${FXC}" /nologo /Tcs_5_0 /EMixLatent /O3 /VnModelLatentMix /Fh "${MOTION_SHADERS}/ModelLatentMix.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/ModelLatentMix.hlsl"
    DEPENDS ModelLatentMix.hlsl VERBATIM)
target_sources(NrModelProbe PRIVATE "${MOTION_SHADERS}/ModelLatentMix.h")
target_include_directories(NrModelProbe PRIVATE "${REPO}/OptiScaler/include" "${MOTION_SHADERS}")
target_link_directories(NrModelProbe PRIVATE "${REPO}/OptiScaler/library/detours")
target_link_libraries(NrModelProbe PRIVATE nr_capture_core "${REPO}/OptiScaler/library/detours/detours.lib")
# Kernel development harness: runs CUDA kernels through the driver's D3D12 path without the NR runtime.
add_executable(NrKernelLab nvfp4/NrKernelLab.cpp)
target_include_directories(NrKernelLab PRIVATE "${REPO}/external/nvapi")
target_compile_definitions(NrKernelLab PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(NrKernelLab PRIVATE /W4 /EHsc /permissive- /utf-8)
target_link_libraries(NrKernelLab PRIVATE d3d12 dxgi)
add_executable(NrMfgUnlockCheck AdaMfgUnlockCheck.cpp)
target_include_directories(NrMfgUnlockCheck PRIVATE . "${REPO}/OptiScaler" "${REPO}/external/nvngx_dlss_sdk")
target_compile_definitions(NrMfgUnlockCheck PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(NrMfgUnlockCheck PRIVATE /W4 /EHsc /permissive- /utf-8)
