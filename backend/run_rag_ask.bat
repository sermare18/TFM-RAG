@echo off
setlocal

if "%~1"=="" (
    echo Usage: run_rag_ask.bat "Your question here"
    exit /b 1
)

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

echo Running RAG question...
python -m rag_cliente.cli ask "%~1"
if errorlevel 1 (
    echo.
    echo Ask command failed. Make sure the Conda env is active and dependencies are installed:
    echo pip install -r requirements.txt
    exit /b 1
)
