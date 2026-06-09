@echo off
setlocal

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=rag-cliente"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not available in PATH. Open an Anaconda/Miniconda prompt or initialize Conda for cmd.exe first.
    exit /b 1
)

set "PROJECT_ROOT=%~dp0"

start "RAG Terminal" cmd /k "cd /d %PROJECT_ROOT% && call conda activate %ENV_NAME% && echo. && echo Project: %PROJECT_ROOT% && echo Conda env: %ENV_NAME%"
