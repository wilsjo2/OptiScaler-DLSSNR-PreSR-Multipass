param([string]$VcVars)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (!$VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$installation) { throw 'Install Visual Studio C++ build tools, or supply -VcVars.' }
    $VcVars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
}
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-pass-profiles-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null

# Use the production PCH and Config, with the same include roots as the addon.
[xml]$project = Get-Content -LiteralPath (Join-Path $repo 'OptiScaler/OptiScaler.vcxproj') -Raw
$includeRoots = ($project.Project.PropertyGroup | Where-Object {
    $_.Condition -eq "'`$(Configuration)|`$(Platform)'=='Release|x64'" -and $_.IncludePath
}).IncludePath
$includeArgs = $includeRoots.Split(';') | Where-Object { $_ -ne '$(IncludePath)' } | ForEach-Object {
    $path = $_.Replace('$(ProjectDir)', "$repo/OptiScaler/").Replace('$(SolutionDir)', "$repo/")
    '/I "' + $path.Replace('\', '/') + '"'
}
$build = @"
@echo off
call "$VcVars" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++latest /EHsc /O2 /MD $($includeArgs -join ' ') "$PSScriptRoot/nr_pass_profiles_smoke.cpp" /Fe:"$out/nr_pass_profiles_smoke.exe" /Fo:"$out/nr_pass_profiles_smoke.obj"
"@
Set-Content -LiteralPath (Join-Path $out 'build.cmd') -Value $build
Push-Location $repo
try {
    & (Join-Path $out 'build.cmd')
    if ($LASTEXITCODE) { throw 'NR pass profile smoke did not compile.' }
    & (Join-Path $out 'nr_pass_profiles_smoke.exe')
    if ($LASTEXITCODE) { throw 'NR pass profile smoke failed.' }
} finally {
    Pop-Location
    Get-ChildItem -LiteralPath $out -File | Remove-Item -Force
    Remove-Item -LiteralPath $out -Force
}
