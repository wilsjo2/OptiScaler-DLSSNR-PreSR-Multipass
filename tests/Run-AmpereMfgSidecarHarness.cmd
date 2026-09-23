@echo off
setlocal
REM Todo 2 of .omo/plans/rtx2030-mfg-integration.md: the sidecar load/arming contract runner.
REM Todo 6 adds the PRODUCTION cases: the shipped OptiScaler\framegen\dlssg\AmpereMfgLoader.cpp is compiled
REM into the harness (mock seams, exactly like the todo-5 eligibility smoke) and driven through the arming
REM seam the Streamline init path uses, so "the payload is loaded before the game's capability decision" is
REM measured on shipping code.
REM
REM Verifies the pinned payload against PIN.json, compiles tests\ampere_mfg_sidecar_harness.cpp (with
REM /DAMPERE_MFG_HARNESS_PRODUCTION and the loader include paths) and the stub payload it needs, then:
REM   (default)          late-arm fixture (RED, must fail with the named status) + the green cases and the
REM                      production cases (all must pass)
REM   --late-arm         the late-arm fixture alone
REM   --production-red   the production cases against the PRE-todo-6 behaviour: exit 3 = the RED state is
REM                      confirmed (arming stops before the load), exit 4 = it is not
REM   --external-mode    only the External-FG-ownership production case: hook installed + armed before init
REM   --fork-guard       only the simulated fork-guard fixture (wilsjo2/main Streamline_Hooks.cpp:1866): the
REM                      hook is never installed, so arming never runs - exit 1 confirms the failure
REM Optional second argument = evidence directory (overrides the default).
REM Run from CMD.
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "W=%~dp0.."
set "E=%W%\.omo\evidence\rtx2030-mfg-integration\02"
set "MODE=%~1"
if not "%~2"=="" set "E=%~2"
set "OUT=%W%\x64\ampere-mfg-sidecar"
set "EXE=%OUT%\ampere_mfg_sidecar_harness.exe"
set "SEAMS=%W%\tests\ampere_mfg_eligibility_seams"
set "LOADER_DIR=%W%\OptiScaler\framegen\dlssg"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%W%"

echo === prerequisites
if not exist "%VCVARS%" (echo MISSING "%VCVARS%" & exit /b 1)
if not exist "%W%\vendor\dlssg_sm86\dlssg_sm86.dll" (echo MISSING staged payload & exit /b 1)
if not exist "%W%\vendor\dlssg_sm86\dlssg_sm86.ini" (echo MISSING staged INI & exit /b 1)
if not exist "%W%\vendor\dlssg_sm86\PIN.json" (echo MISSING PIN.json & exit /b 1)
if not exist "%W%\tests\ampere_mfg_sidecar_harness.cpp" (echo MISSING harness source & exit /b 1)
if not exist "%W%\tests\ampere_mfg_stub_payload.cpp" (echo MISSING stub payload source & exit /b 1)
if not exist "%SEAMS%\Config.h" (echo MISSING the production seam stubs & exit /b 1)
if not exist "%LOADER_DIR%\AmpereMfgLoader.cpp" (echo MISSING the production loader & exit /b 1)
if not exist "%E%" mkdir "%E%"
if not exist "%OUT%" mkdir "%OUT%"

echo === python tools\check_payload_pin.py --pin vendor\dlssg_sm86\PIN.json --dir vendor\dlssg_sm86
python tools\check_payload_pin.py --pin "vendor\dlssg_sm86\PIN.json" --dir "vendor\dlssg_sm86"
set "PINCHECK=%ERRORLEVEL%"
echo pin-check-exit=%PINCHECK%
if not "%PINCHECK%"=="0" (echo pinned payload does not match PIN.json & exit /b 1)

call "%VCVARS%" >nul
echo === cl.exe availability
where cl
echo where-cl-exit=%ERRORLEVEL%

if exist "%EXE%" del /Q "%EXE%"
echo === cl ... /DAMPERE_MFG_HARNESS_PRODUCTION /DOPTISCALER_RTX40_MFG /I "%SEAMS%" /I "%LOADER_DIR%" /I "%W%\external\nvapi" tests\ampere_mfg_sidecar_harness.cpp bcrypt.lib
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /DAMPERE_MFG_HARNESS_PRODUCTION /DOPTISCALER_RTX40_MFG /I "%SEAMS%" /I "%LOADER_DIR%" /I "%W%\external\nvapi" "%W%\tests\ampere_mfg_sidecar_harness.cpp" bcrypt.lib /Fe:"%EXE%" /Fo:"%OUT%\ampere_mfg_sidecar_harness.obj"
set "COMPILE=%ERRORLEVEL%"
echo compile-exit=%COMPILE%
if not "%COMPILE%"=="0" (echo harness compile failed & exit /b 1)
if not exist "%EXE%" (echo MISSING "%EXE%" & exit /b 1)

if exist "%OUT%\ampere_mfg_stub_payload.dll" del /Q "%OUT%\ampere_mfg_stub_payload.dll"
echo === cl /LD ... tests\ampere_mfg_stub_payload.cpp -^> ampere_mfg_stub_payload.dll
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /LD "%W%\tests\ampere_mfg_stub_payload.cpp" /Fe:"%OUT%\ampere_mfg_stub_payload.dll" /Fo:"%OUT%\ampere_mfg_stub_payload.obj"
set "STUBCOMPILE=%ERRORLEVEL%"
echo stub-compile-exit=%STUBCOMPILE%
if not "%STUBCOMPILE%"=="0" (echo stub payload compile failed & exit /b 1)
if not exist "%OUT%\ampere_mfg_stub_payload.dll" (echo MISSING stub payload dll & exit /b 1)

if "%MODE%"=="--late-arm" goto late_arm_only
if "%MODE%"=="--production-red" goto production_red
if "%MODE%"=="--external-mode" goto external_mode
if "%MODE%"=="--fork-guard" goto fork_guard
if not "%MODE%"=="" (echo unknown mode: %MODE% & exit /b 2)

echo === late-arm fixture (RED case): arming after the init boundary must fail with the named status
"%EXE%" --root "%W%" --evidence "%E%" --late-arm > "%E%\late-arm-run.log" 2>&1
set "LATEARM=%ERRORLEVEL%"
type "%E%\late-arm-run.log"
echo late-arm-exit=%LATEARM% ^(1 = fixture failed with the named status as designed, 4 = late arming was NOT detected, other = harness error^)
if not "%LATEARM%"=="1" (echo the late-arm fixture did not fail as required ^(exit=%LATEARM%^) & exit /b 1)
findstr /C:"STATUS=ArmAfterInitBoundary" "%E%\late-arm-fixture.log" >nul
if errorlevel 1 (echo the named status STATUS=ArmAfterInitBoundary is not in the receipt & exit /b 1)

echo === green cases: ordering proof, idempotence, real-payload probe + the PRODUCTION cases
"%EXE%" --root "%W%" --evidence "%E%" > "%E%\green-run.log" 2>&1
set "GREEN=%ERRORLEVEL%"
type "%E%\green-run.log"
echo green-exit=%GREEN%
> "%E%\runner-exits.txt" echo late-arm-exit=%LATEARM%
>> "%E%\runner-exits.txt" echo green-exit=%GREEN%
if not "%GREEN%"=="0" (echo green cases failed & exit /b 1)

echo sidecar-harness: OK ^(late-arm fixture failed with the named status; green + production cases passed^)
endlocal
exit /b 0

:late_arm_only
echo === late-arm fixture only (RED case)
"%EXE%" --root "%W%" --evidence "%E%" --late-arm > "%E%\late-arm-run.log" 2>&1
set "LATEARM=%ERRORLEVEL%"
type "%E%\late-arm-run.log"
echo late-arm-exit=%LATEARM% ^(1 = fixture failed with the named status as designed, 4 = late arming was NOT detected, other = harness error^)
> "%E%\runner-exits-late-arm.txt" echo late-arm-exit=%LATEARM%
if not "%LATEARM%"=="1" (echo the late-arm fixture did not fail as required ^(exit=%LATEARM%^) & exit /b 1)
findstr /C:"STATUS=ArmAfterInitBoundary" "%E%\late-arm-fixture.log" >nul
if errorlevel 1 (echo the named status STATUS=ArmAfterInitBoundary is not in the receipt & exit /b 1)
echo sidecar-harness-late-arm: OK ^(nonzero with the named status^)
endlocal
exit /b 0

:production_red
echo === production cases against the PRE-todo-6 arming path (RED confirmation)
"%EXE%" --root "%W%" --evidence "%E%" --production-red > "%E%\production-red-run.log" 2>&1
set "PRODRED=%ERRORLEVEL%"
type "%E%\production-red-run.log"
echo production-red-exit=%PRODRED% ^(3 = the pre-todo-6 path confirmed: no load, PayloadValidated; 4 = it was NOT confirmed^)
> "%E%\runner-exits-production-red.txt" echo production-red-exit=%PRODRED%
if not "%PRODRED%"=="3" (echo the RED state was not confirmed ^(exit=%PRODRED%^) & exit /b 1)
findstr /C:"STATUS=PayloadValidated" "%E%\logs\production-external.red.log" >nul
if errorlevel 1 (echo the production external case did not report PayloadValidated in the RED run & exit /b 1)
echo sidecar-harness-production-red: OK ^(arming stops before the load: RED confirmed^)
endlocal
exit /b 0

:external_mode
echo === External FG ownership production case (hook installed + armed before the init boundary)
"%EXE%" --root "%W%" --evidence "%E%" --external-mode > "%E%\external-mode-run.log" 2>&1
set "EXTERNAL=%ERRORLEVEL%"
type "%E%\external-mode-run.log"
echo external-mode-exit=%EXTERNAL%
> "%E%\runner-exits-external-mode.txt" echo external-mode-exit=%EXTERNAL%
if not "%EXTERNAL%"=="0" (echo the External-mode case failed ^(exit=%EXTERNAL%^) & exit /b 1)
findstr /C:"STATUS=Loaded" "%E%\logs\production-external.log" >nul
if errorlevel 1 (echo the External-mode case did not report STATUS=Loaded & exit /b 1)
findstr /C:"hook_installed=true" "%E%\logs\production-external.log" >nul
if errorlevel 1 (echo the arming hook was not reported installed in External mode & exit /b 1)
findstr /C:"order=load-before-boundary" "%E%\logs\production-external.log" >nul
if errorlevel 1 (echo the payload was not loaded before the init boundary & exit /b 1)
echo sidecar-harness-external-mode: OK ^(hook installed and arming completed before the init boundary^)
endlocal
exit /b 0

:fork_guard
echo === simulated fork guard (wilsjo2/main Streamline_Hooks.cpp:1866): the hook is never installed
"%EXE%" --root "%W%" --evidence "%E%" --fork-guard > "%E%\fork-guard-run.log" 2>&1
set "FORKGUARD=%ERRORLEVEL%"
type "%E%\fork-guard-run.log"
echo fork-guard-exit=%FORKGUARD% ^(1 = the guard broke the arming hook as designed, 0 = the fixture did not fail^)
> "%E%\runner-exits-fork-guard.txt" echo fork-guard-exit=%FORKGUARD%
if not "%FORKGUARD%"=="1" (echo the fork-guard fixture did not fail as required ^(exit=%FORKGUARD%^) & exit /b 1)
findstr /C:"STATUS=HookNotInstalled" "%E%\logs\production-fork-guard.log" >nul
if errorlevel 1 (echo the named status STATUS=HookNotInstalled is not in the receipt & exit /b 1)
echo sidecar-harness-fork-guard: OK ^(the early return leaves the payload unarmed: named failure^)
endlocal
exit /b 0
