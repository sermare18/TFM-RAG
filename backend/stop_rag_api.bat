@echo off
setlocal

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

set "SERVER_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    set "SERVER_PID=%%P"
    goto :stop_server
)

echo No API server appears to be listening on port %PORT%.
exit /b 1

:stop_server
taskkill /PID %SERVER_PID% /T /F >nul
if errorlevel 1 (
    echo Failed to stop API server with PID %SERVER_PID% on port %PORT%.
    exit /b 1
)

echo API server on port %PORT% stopped.
