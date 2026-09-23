@echo off
setlocal
REM Todo 3 of .omo/plans/rtx2030-mfg-integration.md: the companion-INI generator smoke runner.
REM Compiles tests\ampere_mfg_ini_smoke.cpp (header-only, /DOPTISCALER_RTX40_MFG) and runs the clamp cases
REM (low/high/auto/PTX) against the real 0.3.5 schema, the golden schema check, the discovery candidates and
REM the status vocabulary. Exit 0 = every case passed.
REM Optional first argument = evidence directory (default .omo\evidence\rtx2030-mfg-integration\03), which is
REM how the RED revision wrote its receipts into .omo\evidence\rtx2030-mfg-integration\03\red\. Run from CMD.
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "W=%~dp0.."
set "E=%W%\.omo\evidence\rtx2030-mfg-integration\03"
if not "%~1"=="" set "E=%~1"
set "OUT=%W%\x64\ampere-mfg-ini"
set "EXE=%OUT%\ampere_mfg_ini_smoke.exe"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%W%"

echo === prerequisites
if not exist "%VCVARS%" (echo MISSING "%VCVARS%" & exit /b 1)
if not exist "%W%\tests\ampere_mfg_ini_smoke.cpp" (echo MISSING smoke source & exit /b 1)
if not exist "%W%\OptiScaler\framegen\dlssg\AmpereMfgLoader.h" (echo MISSING loader header & exit /b 1)
if not exist "%E%" mkdir "%E%"
if not exist "%OUT%" mkdir "%OUT%"

call "%VCVARS%" >nul
echo === cl.exe availability
where cl
echo where-cl-exit=%ERRORLEVEL%

if exist "%EXE%" del /Q "%EXE%"
echo === cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /DOPTISCALER_RTX40_MFG tests\ampere_mfg_ini_smoke.cpp
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /DOPTISCALER_RTX40_MFG "%W%\tests\ampere_mfg_ini_smoke.cpp" /Fe:"%EXE%" /Fo:"%OUT%\ampere_mfg_ini_smoke.obj"
set "COMPILE=%ERRORLEVEL%"
echo compile-exit=%COMPILE%
if not "%COMPILE%"=="0" (echo smoke compile failed & exit /b 1)
if not exist "%EXE%" (echo MISSING "%EXE%" & exit /b 1)

echo === ampere_mfg_ini_smoke.exe --evidence "%E%" --root "%W%"
"%EXE%" --evidence "%E%" --root "%W%" > "%E%\smoke-run.log" 2>&1
set "SMOKE=%ERRORLEVEL%"
type "%E%\smoke-run.log"
echo smoke-exit=%SMOKE%
> "%E%\runner-exits.txt" echo compile-exit=%COMPILE%
>> "%E%\runner-exits.txt" echo smoke-exit=%SMOKE%
if not "%SMOKE%"=="0" (echo the INI smoke failed & exit /b 1)

echo ampere-mfg-ini-smoke: OK ^(clamp low/high/auto/PTX, golden schema, discovery, vocabulary^)
endlocal
exit /b 0
