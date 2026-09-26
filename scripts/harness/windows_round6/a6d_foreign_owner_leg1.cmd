@echo off
REM A6d leg 1: a NON-DROCAT process owns port 8080 and no DROCAT instance is
REM running. The launcher must refuse to kill it.
REM (Run with DROCAT instances already stopped.)
cd /d E:\DROCAT_test_260925\drocat
echo ============================================================
echo A6d leg 1 - foreign owner on 8080, zero DROCAT instances
echo ============================================================
echo.
echo --- occupying 8080 with the Anaconda BASE interpreter (no 'drocat' and
echo --- no 'ui\app.py' anywhere in its command line) ---
start "foreign-8080" /MIN C:\Users\krlen\anaconda3\python.exe -m http.server 8080
timeout /t 4 /nobreak >nul
netstat -ano | findstr /R /C:":8080 .*LISTENING"
echo.
echo --- running the launcher against that port with choice 3 ---
echo 3> "%TEMP%\r6_c3.txt"
windows_DROCAT.bat < "%TEMP%\r6_c3.txt"
echo BATCH EXIT=%ERRORLEVEL%
echo.
echo --- the foreign listener must still be alive ---
netstat -ano | findstr /R /C:":8080 .*LISTENING"
