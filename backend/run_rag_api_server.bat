@echo off
setlocal

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=rag-cliente"

set "PORT=%~2"
if "%PORT%"=="" set "PORT=8000"

set "PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%PROJECT_ROOT%src;%PYTHONPATH%"

cd /d "%PROJECT_ROOT%"

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

python -c "import uvicorn, fastapi" >nul 2>nul
if errorlevel 1 (
    echo Missing API dependencies in Conda environment "%ENV_NAME%".
    echo Install them with:
    echo   python -m pip install -r requirements.txt
    echo   python -m pip install -e .
    exit /b 1
)

echo Project: %PROJECT_ROOT%
echo Conda env: %ENV_NAME%
echo Starting uvicorn on port %PORT%...
echo.

python -m uvicorn rag_cliente.api:app --host 0.0.0.0 --port %PORT%

echo.
echo API server stopped or failed to start.
