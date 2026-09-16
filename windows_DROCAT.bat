@echo off
REM Double-click / terminal launcher for Windows.
REM Self-healing: prepares the versioned environment on first run (via
REM archive\install\install.ps1), repairs it when inconsistent, resolves
REM port conflicts interactively, and launches the web UI.
setlocal EnableDelayedExpansion
set "PYTHONNOUSERSITE=1"
REM Force UTF-8 stdio: on zh-CN/GBK (and other legacy code page) systems the
REM UI server and its child runs would otherwise crash or mojibake on the
REM Unicode status symbols used throughout DROCAT output.
set "PYTHONIOENCODING=utf-8"
set "SCRIPT_DIR=%~dp0"
set "CONDA_BIN="

cls
echo DROCAT - Drosophila Connectome Analysis Toolkit
echo Preparing the versioned environment and launching the UI...
echo.

REM Resolve conda to its FULL path: `call "conda"` does not resolve a
REM quoted bare name through PATH/PATHEXT (the conda.bat shim is never
REM found), so every health check would fail and the launcher would loop
REM through :repair forever.
where conda >nul 2>nul && for /f "delims=" %%c in ('where conda') do if not defined CONDA_BIN set "CONDA_BIN=%%c"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\anaconda3\Scripts\conda.exe"

set "DROCAT_VERSION=4.5.0"
for /f "tokens=3" %%v in ('findstr /C:"APP_VERSION = " "%SCRIPT_DIR%ui\config.py"') do set "DROCAT_VERSION=%%v"
set "DROCAT_VERSION=!DROCAT_VERSION:"=!"
set "ENV_BASE=drocat-!DROCAT_VERSION!"
set "REPAIRED=0"
set "CONFIG_FILE=%SCRIPT_DIR%config.json"
set "CONFIG_LOCAL=%SCRIPT_DIR%config_local.json"
set "ENV_OVERRIDE="
REM Version-specific custom env override: config.json wins per key - it is
REM the file a GitHub-pulled copy edits directly; the gitignored
REM config_local.json is the developer-specific fallback for empty entries.
REM envs.<version> is only consulted for the CURRENT release, so upgrading
REM DROCAT never reuses an older release's custom environment.
if exist "!CONFIG_FILE!" for /f "usebackq delims=" %%e in (`powershell -NoProfile -Command "try { $c = Get-Content -Raw -LiteralPath '!CONFIG_FILE!' | ConvertFrom-Json; if ($c.envs.'!DROCAT_VERSION!') { $c.envs.'!DROCAT_VERSION!' } } catch {}"`) do if not defined ENV_OVERRIDE set "ENV_OVERRIDE=%%e"
if not defined ENV_OVERRIDE if exist "!CONFIG_LOCAL!" for /f "usebackq delims=" %%e in (`powershell -NoProfile -Command "try { $c = Get-Content -Raw -LiteralPath '!CONFIG_LOCAL!' | ConvertFrom-Json; if ($c.envs.'!DROCAT_VERSION!') { $c.envs.'!DROCAT_VERSION!' } } catch {}"`) do set "ENV_OVERRIDE=%%e"

:resolve_reset
set "ENV_NAME="
set /a N=0
:resolve
if !N! EQU 0 (
    set "CAND=!ENV_BASE!"
    if defined ENV_OVERRIDE set "CAND=!ENV_OVERRIDE!"
) else (
    set /a SUFFIX=N+1
    set "CAND=!ENV_BASE!-!SUFFIX!"
)
if defined CONDA_BIN (
    call "!CONDA_BIN!" run -n "!CAND!" python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "ENV_NAME=!CAND!"
        goto env_found
    )
)
REM A configured custom env must be the env used: never silently switch to a
REM default name. If it is missing or not Python 3.11, the installer is the
REM only path that can create or repair it.
if !N! EQU 0 if defined ENV_OVERRIDE (
    echo Environment "!ENV_OVERRIDE!" ^(from envs.!DROCAT_VERSION!^) is missing or is not Python 3.11.
    echo Running the installer to create or repair it...
    goto repair
)
set /a N+=1
if !N! LEQ 20 goto resolve
goto repair

:env_found
call "!CONDA_BIN!" run -n "!ENV_NAME!" python -c "import nicegui,numpy,pandas,neuprint,neuronbridge_client" >nul 2>nul
if errorlevel 1 goto repair
call "!CONDA_BIN!" run -n "!ENV_NAME!" python -m pip check >nul 2>nul
if errorlevel 1 goto repair

REM Persist the resolved environment into the LOCAL config when no custom
REM name was configured, so the auto-found env is pinned for later runs
REM (it only applies while the config.json entry stays empty). The
REM committed config.json is never rewritten.
if not defined ENV_OVERRIDE if exist "!CONFIG_LOCAL!" powershell -NoProfile -Command "try { $c = Get-Content -Raw -LiteralPath '!CONFIG_LOCAL!' | ConvertFrom-Json; if (-not $c.envs) { $c | Add-Member -NotePropertyName 'envs' -NotePropertyValue ([pscustomobject]@{}) -Force }; $c.envs | Add-Member -NotePropertyName '!DROCAT_VERSION!' -NotePropertyValue '!ENV_NAME!' -Force; $c | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath '!CONFIG_LOCAL!' -Encoding UTF8 } catch {}"

REM Token hint: remind users who skipped the installer prompt.
set "NP_TOKEN="
if exist "!CONFIG_FILE!" for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "try { $c = Get-Content -Raw -LiteralPath '!CONFIG_FILE!' | ConvertFrom-Json; if ($c.tokens.neuprint) { $c.tokens.neuprint } } catch {}"`) do set "NP_TOKEN=%%t"
if not defined NP_TOKEN if exist "!CONFIG_LOCAL!" for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "try { $c = Get-Content -Raw -LiteralPath '!CONFIG_LOCAL!' | ConvertFrom-Json; if ($c.tokens.neuprint) { $c.tokens.neuprint } } catch {}"`) do set "NP_TOKEN=%%t"
set "HINT=0"
if not defined NP_TOKEN set "HINT=1"
if defined NP_TOKEN echo !NP_TOKEN! | findstr /C:"YOUR_NEUPRINT_TOKEN_HERE" >nul && set "HINT=1"
if defined NP_TOKEN echo !NP_TOKEN! | findstr /C:"NEUPRINT_TOKEN=''" >nul && set "HINT=1"
if "!HINT!"=="1" echo Tip: the NeuPrint token is not configured yet - set it in the Settings tab or config.json (the CAVE token is optional; only needed for FlyWire FAFB online fetching).
if "!HINT!"=="1" echo      Get a NeuPrint token from: https://neuprint.janelia.org/account
if "!HINT!"=="1" echo      Get a CAVE token from: https://codex.flywire.ai/auth_token
goto launch

:repair
if "!REPAIRED!"=="1" goto error
set "REPAIRED=1"
echo Preparing the pinned DROCAT environment...
where pwsh >nul 2>nul
if not errorlevel 1 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%archive\install\install.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%archive\install\install.ps1"
)
if errorlevel 1 goto error
REM Re-detect conda the same way (full path) after the installer may have
REM installed Miniconda or changed the PATH.
where conda >nul 2>nul && for /f "delims=" %%c in ('where conda') do if not defined CONDA_BIN set "CONDA_BIN=%%c"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\anaconda3\Scripts\conda.exe"
goto resolve_reset

:launch
REM Port-conflict guard: when the UI port is busy, ask the user whether
REM to start on a new port, kill the existing DROCAT process, or cancel.
set "APP_PORT="
for /f "tokens=3" %%p in ('findstr /C:"APP_PORT =" "%SCRIPT_DIR%ui\config.py"') do set "APP_PORT=%%p"
if not defined APP_PORT set "APP_PORT=8080"
if defined DROCAT_UI_PORT set "APP_PORT=!DROCAT_UI_PORT!"

set "OWNER_PID="
REM /C: keeps the regex together: findstr treats an unquoted space as a
REM separator between search terms, so ":!APP_PORT! .*LISTENING" would
REM match EVERY LISTENING line (picking PID 4, System, as the owner).
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":!APP_PORT! .*LISTENING"') do if not defined OWNER_PID set "OWNER_PID=%%p"
if not defined OWNER_PID goto start_ui

set "OWNER_CMD="
for /f "delims=" %%c in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter 'ProcessId=!OWNER_PID!' -ErrorAction SilentlyContinue).CommandLine"') do set "OWNER_CMD=%%c"
REM System-owned ports (PID 4) cannot be inspected; show a fallback label.
if not defined OWNER_CMD set "OWNER_CMD=PID !OWNER_PID! (command line unavailable)"
echo.
echo Port !APP_PORT! is already in use by PID !OWNER_PID!:
echo     !OWNER_CMD!
echo   [1] Start DROCAT on a new port
echo !OWNER_CMD! | findstr /I "ui\app.py drocat" >nul
if errorlevel 1 (
    echo   [2] Not allowed: the process on port !APP_PORT! is not DROCAT - stop it manually, then retry
) else (
    echo   [2] Kill the existing DROCAT process and restart on port !APP_PORT!
)
echo   [3] Cancel
set "CHOICE="
set /p "CHOICE=Your choice [1-3]: "
if "!CHOICE!"=="1" goto pick_new_port
if "!CHOICE!"=="2" (
    echo !OWNER_CMD! | findstr /I "ui\app.py drocat" >nul
    if errorlevel 1 goto conflict_abort
    echo Stopping PID !OWNER_PID!...
    taskkill /PID !OWNER_PID! /F >nul 2>nul
    goto wait_port_free
)

:conflict_abort
echo ERROR: could not resolve the port conflict on !APP_PORT!.
echo Stop the process manually and retry, or set DROCAT_UI_PORT to a free port.
pause
exit /b 1

:pick_new_port
set /a NEW_PORT=!APP_PORT!+1
:pick_new_port_loop
if !NEW_PORT! GTR 65535 goto conflict_abort
netstat -ano | findstr /R /C:":!NEW_PORT! .*LISTENING" >nul 2>nul
if not errorlevel 1 (
    set /a NEW_PORT+=1
    goto pick_new_port_loop
)
echo Port !APP_PORT! is busy; starting on port !NEW_PORT! instead.
set "APP_PORT=!NEW_PORT!"
goto start_ui

:wait_port_free
set /a WAIT=0
:wait_port_free_loop
set "OWNER_PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":!APP_PORT! .*LISTENING"') do if not defined OWNER_PID set "OWNER_PID=%%p"
if not defined OWNER_PID goto start_ui
set /a WAIT+=1
if !WAIT! GEQ 20 goto conflict_abort
ping -n 2 127.0.0.1 >nul
goto wait_port_free_loop

:start_ui
echo Starting DROCAT v!DROCAT_VERSION! in !ENV_NAME! at http://127.0.0.1:!APP_PORT!...
cd /d "%SCRIPT_DIR%"
set "DROCAT_UI_PORT=!APP_PORT!"
call "!CONDA_BIN!" run -n "!ENV_NAME!" --no-capture-output python ui\app.py
if errorlevel 1 goto launch_failed
exit /b 0

:launch_failed
echo.
echo DROCAT could not start. Review the messages above.
pause
exit /b 1

:error
echo ERROR: DROCAT could not prepare a usable environment.
pause
exit /b 1
