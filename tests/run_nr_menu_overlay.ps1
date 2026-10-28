$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$build = Join-Path $repo 'x64/nr-menu-overlay'
New-Item -ItemType Directory -Force $build | Out-Null
# Exercise the actual frame gate and overlay drawing with real ImGui, without a game or GPU.
$menu = Get-Content -Raw -LiteralPath "$repo/OptiScaler/menu/menu_common.cpp"
$start = $menu.IndexOf('void MenuCommon::BeginMenuFrameIfNeeded(')
$end = $menu.IndexOf('void MenuCommon::RenderSplashWindow(', $start)
if ($start -lt 0 -or $end -le $start) { throw 'Cannot locate the production frame gate.' }
Set-Content -LiteralPath "$build/MenuFrame.cpp" -Value $menu.Substring($start, $end - $start)
$overlay = Get-Content -Raw -LiteralPath "$repo/OptiScaler/dlssnr/DlssNr_MenuOverlay.cpp"
Set-Content -LiteralPath "$build/MenuOverlay.cpp" -Value ([regex]::Replace($overlay, '(?m)^#include[^\r\n]*', ''))
$imgui = "$repo/OptiScaler/include/imgui"
& cl.exe /nologo /std:c++20 /EHsc /DNOMINMAX "/I$build" "/I$imgui" "/I$repo/external/freetype" `
    "/Fo$build/" "/Fe$build/overlay.exe" "$PSScriptRoot/nr_menu_overlay.cpp" `
    "$imgui/imgui.cpp" "$imgui/imgui_draw.cpp" "$imgui/imgui_tables.cpp" "$imgui/imgui_widgets.cpp" `
    "$imgui/misc/freetype/imgui_freetype.cpp" /link "$repo/external/freetype/freetype.lib" user32.lib gdi32.lib imm32.lib dwmapi.lib
if ($LASTEXITCODE) { throw 'Menu overlay test build failed.' }
& "$build/overlay.exe"
if ($LASTEXITCODE) { throw 'Menu overlay test failed.' }
