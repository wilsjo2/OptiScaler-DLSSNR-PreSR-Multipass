@echo off
setlocal
rem Rebuild the four NR spatial variants with the bundled DXC and header generator.
set "DXC=%~dp0..\..\shader_tools\dxc.exe"
set "HEADER=%~dp0..\..\shader_tools\create_header.py"
set "SOURCE=%~dp0dlssnr_spatial.hlsl"

"%DXC%" -T cs_6_0 -E CSMain -O3 -Qstrip_debug -Qstrip_reflect "%SOURCE%" -Fo "%~dp0dlssnr_spatial_Shader.cso"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_spatial_Shader.cso" "%~dp0dlssnr_spatial_Shader.h" dlssnr_spatial_cso
if errorlevel 1 exit /b 1

"%DXC%" -T cs_6_0 -E CSMain -O3 -Qstrip_debug -Qstrip_reflect -D SPATIAL_GUIDES "%SOURCE%" -Fo "%~dp0dlssnr_spatial_guides_Shader.cso"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_spatial_guides_Shader.cso" "%~dp0dlssnr_spatial_guides_Shader.h" dlssnr_spatial_guides_cso
if errorlevel 1 exit /b 1

"%DXC%" -spirv -T cs_6_0 -E CSMain -O3 -Qstrip_debug -D VK_MODE -Cc -Vi "%SOURCE%" -Fo "%~dp0dlssnr_spatial_Shader_Vk.spv"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_spatial_Shader_Vk.spv" "%~dp0dlssnr_spatial_Shader_Vk.h" dlssnr_spatial_spv
if errorlevel 1 exit /b 1

"%DXC%" -spirv -T cs_6_0 -E CSMain -O3 -Qstrip_debug -D VK_MODE -D SPATIAL_GUIDES -Cc -Vi "%SOURCE%" -Fo "%~dp0dlssnr_spatial_guides_Shader_Vk.spv"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_spatial_guides_Shader_Vk.spv" "%~dp0dlssnr_spatial_guides_Shader_Vk.h" dlssnr_spatial_guides_spv
if errorlevel 1 exit /b 1

echo NR spatial shaders rebuilt.
