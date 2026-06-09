@echo off
setlocal

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=rag-cliente"

set "PORT=%~2"
if "%PORT%"=="" set "PORT=8000"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not available in PATH. Open an Anaconda/Miniconda prompt or initialize Conda for cmd.exe first.
    exit /b 1
)

set "PROJECT_ROOT=%~dp0"
set "RUNNER=%PROJECT_ROOT%run_rag_api_server.bat"

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    if not "%%P"=="" (
        echo Port %PORT% is already in use by PID %%P.
        echo Stop the existing server first or choose another port.
        exit /b 1
    )
)

echo Starting API server on http://localhost:%PORT%
echo Conda env: %ENV_NAME%

start "RAG API" cmd /k ""%RUNNER%" %ENV_NAME% %PORT%"
if errorlevel 1 (
    echo Failed to open the API server terminal.
    exit /b 1
)

echo Swagger UI: http://localhost:%PORT%/docs
