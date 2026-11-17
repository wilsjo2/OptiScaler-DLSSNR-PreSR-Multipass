@echo off
setlocal
set "DXC=%~dp0..\..\shader_tools\dxc.exe"
set "HEADER=%~dp0..\..\shader_tools\create_header.py"
set "SOURCE=%~dp0dlssnr_reapply.hlsl"

"%DXC%" -T cs_6_0 -E CSMain -O3 -Qstrip_debug -Qstrip_reflect "%SOURCE%" -Fo "%~dp0dlssnr_reapply_Shader.cso"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_reapply_Shader.cso" "%~dp0dlssnr_reapply_Shader.h" dlssnr_reapply_cso
if errorlevel 1 exit /b 1

"%DXC%" -spirv -T cs_6_0 -E CSMain -O3 -Qstrip_debug -D VK_MODE -Cc -Vi "%SOURCE%" -Fo "%~dp0dlssnr_reapply_Shader_Vk.spv"
if errorlevel 1 exit /b 1
python "%HEADER%" "%~dp0dlssnr_reapply_Shader_Vk.spv" "%~dp0dlssnr_reapply_Shader_Vk.h" dlssnr_reapply_spv
if errorlevel 1 exit /b 1
echo NR re-apply shaders rebuilt.
