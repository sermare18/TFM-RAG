@echo off
setlocal

set "PDF_DIR=%~1"
if "%PDF_DIR%"=="" set "PDF_DIR=data\pdfs"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

echo Running index on "%PDF_DIR%"...
python -m rag_cliente.cli index --doc-dir "%PDF_DIR%"
if errorlevel 1 (
    echo.
    echo Index command failed. Make sure the Conda env is active and dependencies are installed:
    echo pip install -r requirements.txt
    exit /b 1
)

echo.
echo Index finished successfully.
