@echo off
REM Isolate the windows_DROCAT.bat ownership predicate (lines 147 and 285).
REM Prints MATCH / NO MATCH for each findstr spelling against the command
REM line the launcher actually observed as the owner of port 8080.
setlocal EnableDelayedExpansion

set "SAMPLE1=python  ui\app.py"
set "SAMPLE2=C:\Users\krlen\anaconda3\envs\drocat-4.5.0\python.exe -u C:\Users\krlen\AppData\Local\Temp\tmpAB12.py"

echo Owner command line the launcher saw on port 8080:
echo !SAMPLE1!
echo.

call :probe "[1] batch form  findstr /I ui\app.py drocat" "%SAMPLE1%" /I "ui\app.py drocat"
call :probe "[2] with /C:    findstr /I /C:..."           "%SAMPLE1%" /I /C:"ui\app.py" /C:"drocat"
call :probe "[3] literal /L  findstr /I /L ..."           "%SAMPLE1%" /I /L "ui\app.py drocat"
call :probe "[4] single /C:  findstr /I /C:ui\app.py"     "%SAMPLE1%" /I /C:"ui\app.py"
call :probe "[6] env-path backend, batch form"            "%SAMPLE2%" /I "ui\app.py drocat"

echo.
echo [5] the PowerShell predicate used by :scan_drocat_instances, same string:
powershell -NoProfile -Command "if ('python  ui\app.py' -match 'ui[\\/]app\.py') { '      LISTING MATCHES' } else { '      LISTING DOES NOT MATCH' }"
goto :eof

:probe
echo %~1
echo %~2| findstr %3 %4 %5 %6 %7 >nul
if errorlevel 1 (echo      NO MATCH) else (echo      MATCH)
goto :eof
