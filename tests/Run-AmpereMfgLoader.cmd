@echo off
setlocal
REM Runner for the RTX 20/30 (SM75/SM86) unlock, from .omo/plans/rtx2030-mfg-integration.md.
REM   suite --config         (todo 4) the config-surface runner: it extracts the production declarations and
REM                         read/save statements out of OptiScaler\Config.h / Config.cpp
REM                         (tools\extract_mfg_config_seam.py fails when an anchor moves), compiles
REM                         tests\ampere_mfg_config_smoke.cpp around them with the real SimpleIni, then runs the
REM                         cases a) round trip incl. External=true, b) the 1..5 clamp on AmpereMfgMaxFrames,
REM                         c) the "value || ampereUnlock" rule, d) a save after a load strips none of the four
REM                         keys, e) the shipped OptiScaler.ini. The last step is the designed failure fixture:
REM                         the v0.8.7 removal list applied by hand must make the strip assertions fail (exit 3).
REM   suite --eligibility    (todo 5) the eligibility/ownership matrix: it compiles the PRODUCTION
REM                         AmpereMfgLoader.cpp (tests\ampere_mfg_eligibility_mocks.h stands in for its four host
REM                         sources, tests\ampere_mfg_eligibility_seams\ for the real headers) together with
REM                         tests\ampere_mfg_eligibility_smoke.cpp, builds the stub payload the arm cases stage,
REM                         then runs the matrix and one child process per arm case.
REM   suite --eligibility-red (todo 5, RED) the same, against the evidence copy of the loader with the refusal
REM                         branches removed: the run must fail on the refusal expectations only (exit 3), which
REM                         is what proves the matrix is not blind. Receipts go to the red evidence folder.
REM   suite --matrix         (todo 11, extended by todo 13) the loader failure matrix: it builds the two stub
REM                         payloads (active role 1, standby role 2) plus tests\ampere_mfg_failure_matrix_smoke.cpp
REM                         - the PRODUCTION AmpereMfgLoader.cpp around the same todo-5 mocks - then runs one child
REM                         process per case (missing/tiny/not-a-PE/truncated/wrong-hash payload, standby role, flat
REM                         layout, unwritable INI, truncated and foreign INIs, the kernel-image selector, the four
REM                         conflicting owners, five unsupported adapters, repeated-init idempotence and the
REM                         restart latch) and the unit-level emission rules. The whole set runs TWICE and the two
REM                         receipts are hash-compared (determinism); a broken-fixture run must exit 3 on the
REM                         deliberately corrupted case; the pin checker must reject the wrong-size and wrong-hash
REM                         fixtures by name; todo 13's pin rung adds: the ladder's own constants checked against
REM                         PIN.json (--expect-digest-from-header, exit 0, and exit 1 on a COPY of the header with
REM                         one constant flipped), the pin refusals leaving no companion INI behind, and the
REM                         truncated/wrong-hash rows failing AT THE LOADER with PayloadIncomplete /
REM                         PayloadDigestMismatch instead of reaching the load; then the regressions (INI smoke,
REM                         --config, --eligibility, MSBuild RTX40-MFG) are run from here.
REM Optional second argument = evidence directory (overrides the suite default).
REM Run from CMD.
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "W=%~dp0.."
set "E05=%W%\.omo\evidence\rtx2030-mfg-integration\05"
set "E11=%W%\.omo\evidence\rtx2030-mfg-integration\11"
set "E=%W%\.omo\evidence\rtx2030-mfg-integration\04"
set "SUITE=--config"
if not "%~1"=="" set "SUITE=%~1"
if not "%~2"=="" set "E=%~2"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%W%"

if "%SUITE%"=="--config" goto config

if "%SUITE%"=="--eligibility" goto eligibility
if "%SUITE%"=="--eligibility-red" goto eligibility_red
if "%SUITE%"=="--matrix" goto matrix

echo unknown suite: %SUITE%
echo suites: --config, --eligibility, --eligibility-red, --matrix
endlocal
exit /b 2

REM ===========================================================================
REM --config (todo 4)
REM ===========================================================================
:config
set "OUT=%W%\x64\ampere-mfg-config"
set "EXE=%OUT%\ampere_mfg_config_smoke.exe"

echo === prerequisites
if not exist "%VCVARS%" (echo MISSING "%VCVARS%" & exit /b 1)
if not exist "%W%\tests\ampere_mfg_config_smoke.cpp" (echo MISSING smoke source & exit /b 1)
if not exist "%W%\OptiScaler\Config.cpp" (echo MISSING Config.cpp & exit /b 1)
if not exist "%W%\OptiScaler\Config.h" (echo MISSING Config.h & exit /b 1)
if not exist "%W%\OptiScaler.ini" (echo MISSING OptiScaler.ini & exit /b 1)
if not exist "%E%" mkdir "%E%"
if not exist "%OUT%" mkdir "%OUT%"

echo === python tools\extract_mfg_config_seam.py --config OptiScaler\Config.cpp --header OptiScaler\Config.h --out-dir "%OUT%" --evidence "%E%\seam-extraction.json"
python tools\extract_mfg_config_seam.py --config "OptiScaler\Config.cpp" --header "OptiScaler\Config.h" --out-dir "%OUT%" --evidence "%E%\seam-extraction.json"
set "EXTRACT=%ERRORLEVEL%"
echo extract-exit=%EXTRACT%
if not "%EXTRACT%"=="0" (echo the production anchors moved - the extraction failed & exit /b 1)

call "%VCVARS%" >nul
echo === cl.exe availability
where cl
echo where-cl-exit=%ERRORLEVEL%

cd /d "%OUT%"
if exist "%EXE%" del /Q "%EXE%"
echo === cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /I "%OUT%" /I "%W%\external\simpleini" tests\ampere_mfg_config_smoke.cpp
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /I "%OUT%" /I "%W%\external\simpleini" "%W%\tests\ampere_mfg_config_smoke.cpp" /Fe:"%EXE%" /Fo:"%OUT%\ampere_mfg_config_smoke.obj"
set "COMPILE=%ERRORLEVEL%"
echo compile-exit=%COMPILE%
if not "%COMPILE%"=="0" (echo smoke compile failed & exit /b 1)
if not exist "%EXE%" (echo MISSING "%EXE%" & exit /b 1)

echo === ampere_mfg_config_smoke.exe --evidence "%E%" --root "%W%"
"%EXE%" --evidence "%E%" --root "%W%" > "%E%\config-run.log" 2>&1
set "CONFIG=%ERRORLEVEL%"
type "%E%\config-run.log"
echo config-exit=%CONFIG%

echo === failure fixture: the v0.8.7 removal list must be caught by the strip assertions
"%EXE%" --evidence "%E%" --root "%W%" --trap-legacy-strip > "%E%\config-trap.log" 2>&1
set "TRAP=%ERRORLEVEL%"
type "%E%\config-trap.log"
echo trap-exit=%TRAP% ^(3 = strip detected as designed, 2 = the strip assertions are blind^)

> "%E%\runner-exits.txt" echo extract-exit=%EXTRACT%
>> "%E%\runner-exits.txt" echo compile-exit=%COMPILE%
>> "%E%\runner-exits.txt" echo config-exit=%CONFIG%
>> "%E%\runner-exits.txt" echo trap-exit=%TRAP%

if not "%CONFIG%"=="0" (echo the config cases failed & exit /b 1)
if not "%TRAP%"=="3" (echo the legacy-strip fixture did not fail as required & exit /b 1)

echo === produced INI hashes
powershell -NoProfile -Command "Get-ChildItem -LiteralPath '%E%\ini' -Filter '*.ini' | Sort-Object Name | ForEach-Object { '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name }" > "%E%\produced-ini-hashes.txt"
echo hash-exit=%ERRORLEVEL%
type "%E%\produced-ini-hashes.txt"

echo ampere-mfg-loader --config: OK ^(round trip, clamp 1..5, || ampereUnlock, no strip, shipped INI, legacy-strip fixture caught^)
endlocal
exit /b 0

REM ===========================================================================
REM --eligibility / --eligibility-red (todo 5)
REM ===========================================================================
:eligibility
set "LOADER_DIR=%W%\OptiScaler\framegen\dlssg"
set "REDFLAG="
set "SCRATCH=%W%\x64\ampere-mfg-eligibility\scratch"
set "EXPECT=0"
if "%~2"=="" set "E=%E05%"
goto eligibility_build

:eligibility_red
if "%~2"=="" set "E=%E05%\red"
set "LOADER_DIR=%E%"
set "REDFLAG=--expect-red"
set "SCRATCH=%W%\x64\ampere-mfg-eligibility\scratch-red"
set "EXPECT=3"
goto eligibility_build

:eligibility_build
set "OUT=%W%\x64\ampere-mfg-eligibility"
set "EXE=%OUT%\ampere_mfg_eligibility_smoke.exe"
set "STUB=%OUT%\stub\dlssg_sm86.dll"
set "SEAMS=%W%\tests\ampere_mfg_eligibility_seams"

echo === prerequisites ^(suite %SUITE%^)
if not exist "%VCVARS%" (echo MISSING "%VCVARS%" & exit /b 1)
if not exist "%W%\tests\ampere_mfg_eligibility_smoke.cpp" (echo MISSING smoke source & exit /b 1)
if not exist "%W%\tests\ampere_mfg_eligibility_mocks.h" (echo MISSING mocks header & exit /b 1)
if not exist "%W%\tests\ampere_mfg_stub_payload.cpp" (echo MISSING stub payload source & exit /b 1)
if not exist "%W%\OptiScaler\framegen\dlssg\AmpereMfgLoader.h" (echo MISSING loader header & exit /b 1)
if not exist "%W%\OptiScaler\framegen\dlssg\AmpereMfgLoader.cpp" (echo MISSING loader source & exit /b 1)
if not exist "%LOADER_DIR%\AmpereMfgLoader.cpp" (echo MISSING the loader copy under "%LOADER_DIR%" & exit /b 1)
if not exist "%LOADER_DIR%\AmpereMfgLoader.h" (echo MISSING the loader header under "%LOADER_DIR%" & exit /b 1)
if not exist "%SEAMS%\Config.h" (echo MISSING the seam stubs & exit /b 1)
if not exist "%E%" mkdir "%E%"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%\stub" mkdir "%OUT%\stub"
if not exist "%OUT%\arm" mkdir "%OUT%\arm"
if not exist "%SCRATCH%" mkdir "%SCRATCH%"

python tools\extract_mfg_config_seam.py --config "OptiScaler\Config.cpp" --header "OptiScaler\Config.h" --out-dir "%OUT%" --evidence "%E%\seam-extraction.json"
if errorlevel 1 exit /b 1

call "%VCVARS%" >nul
echo === cl.exe availability
where cl
echo where-cl-exit=%ERRORLEVEL%

echo === stub payload: cl /LD tests\ampere_mfg_stub_payload.cpp -^> "%STUB%"
if exist "%STUB%" del /Q "%STUB%"
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /LD "%W%\tests\ampere_mfg_stub_payload.cpp" /Fe:"%STUB%" /Fo:"%OUT%\stub\ampere_mfg_stub_payload.obj"
set "STUBCOMPILE=%ERRORLEVEL%"
echo stub-compile-exit=%STUBCOMPILE%
if not "%STUBCOMPILE%"=="0" (echo stub payload compile failed & exit /b 1)
if not exist "%STUB%" (echo MISSING "%STUB%" & exit /b 1)

cd /d "%OUT%"
if exist "%EXE%" del /Q "%EXE%"
echo === cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /DOPTISCALER_RTX40_MFG /I "%SEAMS%" /I "%LOADER_DIR%" /I "%W%\external\nvapi" tests\ampere_mfg_eligibility_smoke.cpp dxgi.lib
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /DOPTISCALER_RTX40_MFG /I "%OUT%" /I "%SEAMS%" /I "%LOADER_DIR%" /I "%W%\external\nvapi" "%W%\tests\ampere_mfg_eligibility_smoke.cpp" dxgi.lib /Fe:"%EXE%" /Fo:"%OUT%\ampere_mfg_eligibility_smoke.obj"
set "COMPILE=%ERRORLEVEL%"
echo compile-exit=%COMPILE%
if not "%COMPILE%"=="0" (echo eligibility smoke compile failed & exit /b 1)
if not exist "%EXE%" (echo MISSING "%EXE%" & exit /b 1)

echo === ampere_mfg_eligibility_smoke.exe --evidence "%E%" --scratch "%SCRATCH%" --stub "%STUB%" %REDFLAG%
"%EXE%" --evidence "%E%" --scratch "%SCRATCH%" --stub "%STUB%" %REDFLAG% > "%E%\eligibility-run.log" 2>&1
set "ELIG=%ERRORLEVEL%"
type "%E%\eligibility-run.log"
echo eligibility-exit=%ELIG% ^(expected %EXPECT%^)

> "%E%\runner-exits-eligibility.txt" echo loader-dir=%LOADER_DIR%
>> "%E%\runner-exits-eligibility.txt" echo stub-compile-exit=%STUBCOMPILE%
>> "%E%\runner-exits-eligibility.txt" echo compile-exit=%COMPILE%
>> "%E%\runner-exits-eligibility.txt" echo eligibility-exit=%ELIG%
>> "%E%\runner-exits-eligibility.txt" echo expected-exit=%EXPECT%

if not "%ELIG%"=="%EXPECT%" (echo the eligibility suite did not exit %EXPECT% ^(exit=%ELIG%^) & exit /b 1)

echo === touch checks: the payload was never loaded by this runner and nothing outside the suite moved
findstr /C:"load_attempts=0" "%E%\load-counts.txt" >nul
if errorlevel 1 (echo no load-count receipt with 0 was written & exit /b 1)

if "%SUITE%"=="--eligibility-red" (
    findstr /C:"RED CONFIRMED" "%E%\eligibility-run.log" >nul
    if errorlevel 1 (echo the RED run did not report RED CONFIRMED & exit /b 1)
    echo ampere-mfg-loader --eligibility-red: OK ^(refusals missing as designed, refused rows only^)
) else (
    echo ampere-mfg-loader --eligibility: OK ^(matrix + arm cases, load attempts 0 on every refusal^)
)

endlocal
exit /b 0

REM ===========================================================================
REM --matrix (todo 11): the loader failure matrix, its determinism repeat, the
REM broken-fixture run, the pin checks and the regressions
REM ===========================================================================
:matrix
if "%~2"=="" set "E=%E11%"
set "OUT=%W%\x64\ampere-mfg-matrix"
set "EXE=%OUT%\ampere_mfg_failure_matrix_smoke.exe"
set "ACTIVE=%OUT%\fixtures\dlssg_sm86.dll"
set "STANDBY=%OUT%\fixtures\dlssg_sm86_standby.dll"
set "SCRATCH=%OUT%\scratch"
set "PINCHECK=%OUT%\pin-check"
set "VENDOR=%W%\vendor\dlssg_sm86"
set "M=%E%\matrix"
set "M2=%E%\matrix-repeat"
set "MB=%E%\matrix-broken"
set "MSBUILDPATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
set "SEAMS=%W%\tests\ampere_mfg_eligibility_seams"
set "LOADER_DIR=%W%\OptiScaler\framegen\dlssg"

echo === prerequisites ^(suite --matrix^)
if not exist "%VCVARS%" (echo MISSING "%VCVARS%" & exit /b 1)
if not exist "%W%\tests\ampere_mfg_failure_matrix_smoke.cpp" (echo MISSING the matrix smoke source & exit /b 1)
if not exist "%W%\tests\ampere_mfg_stub_payload.cpp" (echo MISSING the stub payload source & exit /b 1)
if not exist "%W%\tests\ampere_mfg_stub_payload_standby.cpp" (echo MISSING the standby stub source & exit /b 1)
if not exist "%W%\tests\ampere_mfg_eligibility_mocks.h" (echo MISSING the mocks header & exit /b 1)
if not exist "%SEAMS%\Config.h" (echo MISSING the seam stubs & exit /b 1)
if not exist "%W%\OptiScaler\framegen\dlssg\AmpereMfgLoader.cpp" (echo MISSING the loader source & exit /b 1)
if not exist "%W%\OptiScaler\framegen\dlssg\AmpereMfgLoader.h" (echo MISSING the loader header & exit /b 1)
if not exist "%W%\tests\Run-AmpereMfgIniSmoke.cmd" (echo MISSING the INI smoke runner & exit /b 1)
if not exist "%VENDOR%\dlssg_sm86.dll" (echo MISSING the pinned payload ^(the wrong-size/wrong-hash fixtures need its byte count^) & exit /b 1)
if not exist "%VENDOR%\dlssg_sm86.ini" (echo MISSING the pinned INI & exit /b 1)
if not exist "%VENDOR%\THIRD_PARTY_NOTICES.txt" (echo MISSING the pinned notices & exit /b 1)
if not exist "%VENDOR%\PIN.json" (echo MISSING PIN.json & exit /b 1)
if not exist "%MSBUILDPATH%" (echo MISSING "%MSBUILDPATH%" & exit /b 1)
if not exist "%E%" mkdir "%E%"
if not exist "%E%\regression" mkdir "%E%\regression"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%\fixtures" mkdir "%OUT%\fixtures"
if not exist "%SCRATCH%" mkdir "%SCRATCH%"

call "%VCVARS%" >nul
echo === cl.exe availability
where cl
echo where-cl-exit=%ERRORLEVEL%

cd /d "%OUT%"
echo === fixture payloads: the active stub ^(role 1^) and the standby stub ^(role 2^)
if exist "%ACTIVE%" del /Q "%ACTIVE%"
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /LD "%W%\tests\ampere_mfg_stub_payload.cpp" /Fe:"%ACTIVE%" /Fo:"%OUT%\fixtures\active.obj"
set "STUB1=%ERRORLEVEL%"
if exist "%STANDBY%" del /Q "%STANDBY%"
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /LD "%W%\tests\ampere_mfg_stub_payload_standby.cpp" /Fe:"%STANDBY%" /Fo:"%OUT%\fixtures\standby.obj"
set "STUB2=%ERRORLEVEL%"
echo stub-active-exit=%STUB1% stub-standby-exit=%STUB2%
if not "%STUB1%"=="0" (echo the active stub payload did not compile & exit /b 1)
if not "%STUB2%"=="0" (echo the standby stub payload did not compile & exit /b 1)

echo === matrix harness: the production loader + the todo-5 mocks
if exist "%EXE%" del /Q "%EXE%"
cl /nologo /std:c++latest /EHsc /MD /O2 /W4 /DUNICODE /D_UNICODE /DOPTISCALER_RTX40_MFG /I "%SEAMS%" /I "%LOADER_DIR%" /I "%W%\external\nvapi" "%W%\tests\ampere_mfg_failure_matrix_smoke.cpp" /Fe:"%EXE%" /Fo:"%OUT%\matrix.obj"
set "COMPILE=%ERRORLEVEL%"
echo compile-exit=%COMPILE%
if not "%COMPILE%"=="0" (echo the matrix harness did not compile & exit /b 1)
if not exist "%EXE%" (echo MISSING "%EXE%" & exit /b 1)

set FIXARGS=--scratch "%SCRATCH%" --stub "%ACTIVE%" --stub-standby "%STANDBY%" --pinned-module "%VENDOR%\dlssg_sm86.dll"

echo === run 1 of 2 ^(the matrix: one child process per case^)
if not exist "%M%" mkdir "%M%"
"%EXE%" --evidence "%M%" %FIXARGS% > "%M%\matrix-run.log" 2>&1
set "RUN1=%ERRORLEVEL%"
type "%M%\matrix-run.log"
echo matrix-run1-exit=%RUN1%

echo === run 2 of 2 ^(the determinism repeat, same scratch, fresh receipts^)
if not exist "%M2%" mkdir "%M2%"
"%EXE%" --evidence "%M2%" %FIXARGS% > "%M2%\matrix-run.log" 2>&1
set "RUN2=%ERRORLEVEL%"
type "%M2%\matrix-run.log"
echo matrix-run2-exit=%RUN2%

echo === determinism: the two runs must be byte-identical
powershell -NoProfile -Command "$a='%M%'; $b='%M2%'; $n='matrix-cases.json','matrix-rows.txt','matrix-table.txt','load-counts.txt','ini-hashes.txt','unit-kernel-selector.txt'; foreach($f in $n){$h1=(Get-FileHash -LiteralPath (Join-Path $a $f) -Algorithm SHA256).Hash.ToLower(); $h2=(Get-FileHash -LiteralPath (Join-Path $b $f) -Algorithm SHA256).Hash.ToLower(); if($h1 -eq $h2){'IDENTICAL {0} {1}' -f $f,$h1}else{'DIFFERENT {0} {1} {2}' -f $f,$h1,$h2}}" > "%E%\determinism.txt" 2>&1
set "DET=%ERRORLEVEL%"
type "%E%\determinism.txt"
findstr /C:"DIFFERENT" "%E%\determinism.txt" >nul
if not errorlevel 1 (echo the two matrix runs do not match & exit /b 1)
findstr /C:"IDENTICAL" "%E%\determinism.txt" >nul
if errorlevel 1 (echo the determinism comparison produced no verdict & exit /b 1)

echo === independent INI sha256 cross-check ^(PowerShell Get-FileHash vs the child's CNG digest^)
REM m05 is not in this list on purpose: the pin refusal happens BEFORE the companion INI is written, so there is
REM no INI to hash - the block below asserts that absence directly.
powershell -NoProfile -Command "$pairs=@(@('m11-valid-control','%SCRATCH%\m11-valid-control\OptiScaler\dlssg_sm86\dlssg_sm86.ini'),@('m06-standby-role','%SCRATCH%\m06-standby-role\OptiScaler\dlssg_sm86\dlssg_sm86.ini'),@('m07-flat-layout','%SCRATCH%\m07-flat-layout\dlssg_sm86\dlssg_sm86.ini'),@('i01-repeat-init-idempotent','%SCRATCH%\i01-repeat-init-idempotent\OptiScaler\dlssg_sm86\dlssg_sm86.ini'),@('i02-restart-latch-options','%SCRATCH%\i02-restart-latch-options\OptiScaler\dlssg_sm86\dlssg_sm86.ini')); foreach($p in $pairs){$c=$p[0]; $hash=(Get-FileHash -LiteralPath $p[1] -Algorithm SHA256).Hash.ToLower(); $child=(Select-String -LiteralPath (Join-Path '%M%' ('cases\'+$c+'.kv')) -Pattern '^ini_sha256_first=').Line.Split('=')[1]; if($hash -eq $child){'MATCH {0} {1}' -f $c,$hash}else{'MISMATCH {0} file={1} child={2}' -f $c,$hash,$child}}" > "%E%\ini-hash-crosscheck.txt" 2>&1
echo ini-crosscheck-exit=%ERRORLEVEL%
type "%E%\ini-hash-crosscheck.txt"
findstr /C:"MISMATCH" "%E%\ini-hash-crosscheck.txt" >nul
if not errorlevel 1 (echo the independent sha256 cross-check disagreed & exit /b 1)
findstr /C:"MATCH" "%E%\ini-hash-crosscheck.txt" >nul
if errorlevel 1 (echo the sha256 cross-check produced no verdict & exit /b 1)

echo === the pin refusals left no companion INI behind ^(the rung runs before the INI write^)
set "PININI=0"
set "CHECK04=no"
set "CHECK05=no"
if exist "%SCRATCH%\m04-wrong-size-truncated\OptiScaler\dlssg_sm86\dlssg_sm86.ini" set "CHECK04=yes"
if exist "%SCRATCH%\m05-wrong-hash-same-size\OptiScaler\dlssg_sm86\dlssg_sm86.ini" set "CHECK05=yes"
if "%CHECK04%"=="yes" set "PININI=1"
if "%CHECK05%"=="yes" set "PININI=1"
> "%E%\pin-refusal-no-ini.txt" echo pin refusals ^(m04 wrong size, m05 wrong digest^) leaving a companion INI behind: %PININI% ^(0 = none left^)
>> "%E%\pin-refusal-no-ini.txt" echo m04-wrong-size-truncated ini present: %CHECK04%
>> "%E%\pin-refusal-no-ini.txt" echo m05-wrong-hash-same-size ini present: %CHECK05%
echo pin-refusal-ini-left-behind=%PININI% ^(0 = none^)
if not "%PININI%"=="0" (echo a pin refusal left a companion INI behind & exit /b 1)

echo === deliberately broken fixture: m11-valid-control gets a text file as its module
if not exist "%MB%" mkdir "%MB%"
"%EXE%" --evidence "%MB%" %FIXARGS% --broken-fixture > "%MB%\matrix-run.log" 2>&1
set "BROKEN=%ERRORLEVEL%"
type "%MB%\matrix-run.log"
echo matrix-broken-exit=%BROKEN% ^(3 = the broken fixture failed as designed^)
if not exist "%MB%\broken-fixture.txt" (echo the broken-fixture receipt was not written & exit /b 1)
type "%MB%\broken-fixture.txt"
findstr /C:"BROKEN FIXTURE CAUGHT:" "%MB%\broken-fixture.txt" >nul
if errorlevel 1 (echo the broken fixture did not fail as designed & exit /b 1)
findstr /C:"m11-valid-control" "%MB%\broken-fixture.txt" >nul
if errorlevel 1 (echo the broken-fixture receipt does not name the case & exit /b 1)

cd /d "%W%"
echo === pin checks: the arming path validates a PE image, the pin manifest owns size and digest
if exist "%PINCHECK%" rmdir /S /Q "%PINCHECK%"
mkdir "%PINCHECK%\wrong-size"
mkdir "%PINCHECK%\wrong-hash"
copy /Y "%SCRATCH%\m04-wrong-size-truncated\OptiScaler\dlssg_sm86\dlssg_sm86.dll" "%PINCHECK%\wrong-size\dlssg_sm86.dll" >nul
copy /Y "%SCRATCH%\m05-wrong-hash-same-size\OptiScaler\dlssg_sm86\dlssg_sm86.dll" "%PINCHECK%\wrong-hash\dlssg_sm86.dll" >nul
for %%D in (wrong-size wrong-hash) do copy /Y "%VENDOR%\dlssg_sm86.ini" "%PINCHECK%\%%D\dlssg_sm86.ini" >nul
for %%D in (wrong-size wrong-hash) do copy /Y "%VENDOR%\THIRD_PARTY_NOTICES.txt" "%PINCHECK%\%%D\THIRD_PARTY_NOTICES.txt" >nul

set "PYTHONDONTWRITEBYTECODE=1"
python tools\check_payload_pin.py --pin "%VENDOR%\PIN.json" --dir "%PINCHECK%\wrong-size" > "%E%\pin-check-wrong-size.log" 2>&1
set "PIN1=%ERRORLEVEL%"
python tools\check_payload_pin.py --pin "%VENDOR%\PIN.json" --dir "%PINCHECK%\wrong-hash" > "%E%\pin-check-wrong-hash.log" 2>&1
set "PIN2=%ERRORLEVEL%"
echo pin-wrong-size-exit=%PIN1% ^(1 = rejected^)
type "%E%\pin-check-wrong-size.log"
echo pin-wrong-hash-exit=%PIN2% ^(1 = rejected^)
type "%E%\pin-check-wrong-hash.log"
> "%E%\pin-checks.txt" echo wrong-size fixture: check_payload_pin.py exit=%PIN1%
>> "%E%\pin-checks.txt" type "%E%\pin-check-wrong-size.log"
>> "%E%\pin-checks.txt" echo wrong-hash fixture: check_payload_pin.py exit=%PIN2%
>> "%E%\pin-checks.txt" type "%E%\pin-check-wrong-hash.log"
findstr /C:"FAIL dlssg_sm86.dll: size" "%E%\pin-check-wrong-size.log" >nul
if errorlevel 1 (echo the wrong-size fixture was not rejected by size & exit /b 1)
findstr /C:"FAIL dlssg_sm86.dll: sha256" "%E%\pin-check-wrong-hash.log" >nul
if errorlevel 1 (echo the wrong-hash fixture was not rejected by digest & exit /b 1)

echo === the extended checker: the ladder's constants against PIN.json ^(--expect-digest-from-header^)
set "HDR=%W%\OptiScaler\framegen\dlssg\AmpereMfgLoader.h"
python tools\check_payload_pin.py --expect-name-from-header "%HDR%" --expect-digest-from-header "%HDR%" > "%E%\pin-header-check.log" 2>&1
set "PINHDR=%ERRORLEVEL%"
type "%E%\pin-header-check.log"
echo pin-header-check-exit=%PINHDR% ^(0 = both constants equal PIN.json^)
if not "%PINHDR%"=="0" (echo the header constants do not match PIN.json & exit /b 1)
findstr /C:"OK: kPayloadExpectedBytes=30021920" "%E%\pin-header-check.log" >nul
if errorlevel 1 (echo the checker's happy output does not name the pinned size & exit /b 1)

echo === the checker's failure fixture: a COPY of the header with one constant flipped
echo    ^(the shipped header is never modified - the flips only ever live in the copy under pin-check\header-flip^)
if not exist "%PINCHECK%\header-flip" mkdir "%PINCHECK%\header-flip"
copy /Y "%HDR%" "%PINCHECK%\header-flip\digest-flipped.h" >nul
copy /Y "%HDR%" "%PINCHECK%\header-flip\size-flipped.h" >nul
powershell -NoProfile -Command "$d='%PINCHECK%\header-flip'; $p=Join-Path $d 'digest-flipped.h'; (Get-Content -Raw -LiteralPath $p).Replace('c3934a09399f022504227c72df0bf8c0de55f9a08880dddde898c5262cefa838','c3934a09399f022504227c72df0bf8c0de55f9a08880dddde898c5262cefa839') | Set-Content -LiteralPath $p -NoNewline -Encoding ascii; $q=Join-Path $d 'size-flipped.h'; [regex]::Replace((Get-Content -Raw -LiteralPath $q),'kPayloadExpectedBytes\s*=\s*\d+','kPayloadExpectedBytes = 30021921') | Set-Content -LiteralPath $q -NoNewline -Encoding ascii"
echo flip-exit=%ERRORLEVEL%
powershell -NoProfile -Command "Select-String -LiteralPath '%PINCHECK%\header-flip\digest-flipped.h' -Pattern 'kPayloadExpectedSha256\[\] = \"[0-9a-f]+\"' | ForEach-Object { 'digest-flipped copy: ' + $_.Line.Trim() }; Select-String -LiteralPath '%PINCHECK%\header-flip\size-flipped.h' -Pattern 'kPayloadExpectedBytes\s*=' | ForEach-Object { 'size-flipped copy: ' + $_.Line.Trim() }" > "%E%\pin-header-flipped-lines.txt" 2>&1
type "%E%\pin-header-flipped-lines.txt"
python tools\check_payload_pin.py --expect-digest-from-header "%PINCHECK%\header-flip\digest-flipped.h" > "%E%\pin-header-flip-digest.log" 2>&1
set "FLIPD=%ERRORLEVEL%"
python tools\check_payload_pin.py --expect-digest-from-header "%PINCHECK%\header-flip\size-flipped.h" > "%E%\pin-header-flip-size.log" 2>&1
set "FLIPS=%ERRORLEVEL%"
echo pin-header-flip-digest-exit=%FLIPD% ^(1 = rejected^)
type "%E%\pin-header-flip-digest.log"
echo pin-header-flip-size-exit=%FLIPS% ^(1 = rejected^)
type "%E%\pin-header-flip-size.log"
findstr /C:"kPayloadExpectedSha256" "%E%\pin-header-flip-digest.log" >nul
if errorlevel 1 (echo the digest flip was not named by the checker & exit /b 1)
findstr /C:"kPayloadExpectedBytes" "%E%\pin-header-flip-size.log" >nul
if errorlevel 1 (echo the size flip was not named by the checker & exit /b 1)
if not "%FLIPD%"=="1" (echo the digest flip was not rejected & exit /b 1)
if not "%FLIPS%"=="1" (echo the size flip was not rejected & exit /b 1)
> "%E%\pin-header-flips.txt" echo digest-flipped COPY of the header: check_payload_pin.py --expect-digest-from-header exit=%FLIPD%
>> "%E%\pin-header-flips.txt" type "%E%\pin-header-flip-digest.log"
>> "%E%\pin-header-flips.txt" echo size-flipped COPY of the header: check_payload_pin.py --expect-digest-from-header exit=%FLIPS%
>> "%E%\pin-header-flips.txt" type "%E%\pin-header-flip-size.log"

echo === regressions: the INI smoke, --config, --eligibility and MSBuild RTX40-MFG
call "%W%\tests\Run-AmpereMfgIniSmoke.cmd" "%E%\regression\ini" > "%E%\regression\ini-smoke-outer.log" 2>&1
set "REGINI=%ERRORLEVEL%"
echo regression-ini-smoke-exit=%REGINI%
call "%~f0" --config "%E%\regression\config" > "%E%\regression\config-outer.log" 2>&1
set "REGCONFIG=%ERRORLEVEL%"
echo regression-config-exit=%REGCONFIG%
call "%~f0" --eligibility "%E%\regression\eligibility" > "%E%\regression\eligibility-outer.log" 2>&1
set "REGELIG=%ERRORLEVEL%"
echo regression-eligibility-exit=%REGELIG%
"%MSBUILDPATH%" OptiScaler.sln /p:Configuration=Release /p:Platform=x64 /p:OptiScalerRtx40Mfg=true /m:4 /t:Build > "%E%\regression\msbuild.log" 2>&1
set "MSBUILD=%ERRORLEVEL%"
echo regression-msbuild-exit=%MSBUILD%
if not "%MSBUILD%"=="0" (type "%E%\regression\msbuild.log" & echo the RTX40-MFG build failed & exit /b 1)

echo === touched files ^(this suite only; the workspace carries unrelated dirty files as received^)
git status --porcelain -- tests/ampere_mfg_failure_matrix_smoke.cpp tests/ampere_mfg_eligibility_smoke.cpp tests/ampere_mfg_sidecar_harness.cpp tests/ampere_mfg_eligibility_mocks.h tests/ampere_mfg_payload_pin_seam.h tests/Run-AmpereMfgLoader.cmd tools/check_payload_pin.py OptiScaler/framegen/dlssg/AmpereMfgLoader.h OptiScaler/framegen/dlssg/AmpereMfgLoader.cpp > "%E%\touched-files.txt" 2>&1
git diff --stat -- tests/ampere_mfg_failure_matrix_smoke.cpp tests/ampere_mfg_eligibility_smoke.cpp tests/ampere_mfg_sidecar_harness.cpp tests/ampere_mfg_eligibility_mocks.h tests/Run-AmpereMfgLoader.cmd tools/check_payload_pin.py OptiScaler/framegen/dlssg/AmpereMfgLoader.h OptiScaler/framegen/dlssg/AmpereMfgLoader.cpp >> "%E%\touched-files.txt" 2>&1
type "%E%\touched-files.txt"

> "%E%\runner-exits-matrix.txt" echo stub-active-compile-exit=%STUB1%
>> "%E%\runner-exits-matrix.txt" echo stub-standby-compile-exit=%STUB2%
>> "%E%\runner-exits-matrix.txt" echo compile-exit=%COMPILE%
>> "%E%\runner-exits-matrix.txt" echo matrix-run1-exit=%RUN1%
>> "%E%\runner-exits-matrix.txt" echo matrix-run2-exit=%RUN2%
>> "%E%\runner-exits-matrix.txt" echo determinism-exit=%DET%
>> "%E%\runner-exits-matrix.txt" echo matrix-broken-exit=%BROKEN% ^(expected 3^)
>> "%E%\runner-exits-matrix.txt" echo pin-wrong-size-exit=%PIN1% ^(expected 1^)
>> "%E%\runner-exits-matrix.txt" echo pin-wrong-hash-exit=%PIN2% ^(expected 1^)
>> "%E%\runner-exits-matrix.txt" echo pin-header-check-exit=%PINHDR% ^(expected 0^)
>> "%E%\runner-exits-matrix.txt" echo pin-header-flip-digest-exit=%FLIPD% ^(expected 1^)
>> "%E%\runner-exits-matrix.txt" echo pin-header-flip-size-exit=%FLIPS% ^(expected 1^)
>> "%E%\runner-exits-matrix.txt" echo pin-refusal-ini-left-behind=%PININI% ^(expected 0^)
>> "%E%\runner-exits-matrix.txt" echo regression-ini-smoke-exit=%REGINI%
>> "%E%\runner-exits-matrix.txt" echo regression-config-exit=%REGCONFIG%
>> "%E%\runner-exits-matrix.txt" echo regression-eligibility-exit=%REGELIG%
>> "%E%\runner-exits-matrix.txt" echo regression-msbuild-exit=%MSBUILD%
type "%E%\runner-exits-matrix.txt"

set "FAILED=0"
if not "%RUN1%"=="0" set "FAILED=1"
if not "%RUN2%"=="0" set "FAILED=1"
if not "%BROKEN%"=="3" set "FAILED=1"
if not "%PIN1%"=="1" set "FAILED=1"
if not "%PIN2%"=="1" set "FAILED=1"
if not "%PINHDR%"=="0" set "FAILED=1"
if not "%FLIPD%"=="1" set "FAILED=1"
if not "%FLIPS%"=="1" set "FAILED=1"
if not "%PININI%"=="0" set "FAILED=1"
if not "%REGINI%"=="0" set "FAILED=1"
if not "%REGCONFIG%"=="0" set "FAILED=1"
if not "%REGELIG%"=="0" set "FAILED=1"
if not "%FAILED%"=="0" (echo the matrix suite, its fixtures or a regression did not behave as required & exit /b 1)

echo ampere-mfg-loader --matrix: OK ^(27 cases with their exact statuses incl. the pin refusals, twice identical, broken fixture caught, the pin checker green with matching constants and red on both flipped copies, regressions green^)
endlocal
exit /b 0
