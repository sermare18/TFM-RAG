@echo off
setlocal

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=rag-cliente"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not available in PATH. Open an Anaconda/Miniconda prompt or initialize Conda for cmd.exe first.
    exit /b 1
)

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo Failed to activate Conda environment "%ENV_NAME%".
    exit /b 1
)

echo Conda environment "%ENV_NAME%" is active.
