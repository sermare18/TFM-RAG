@echo off
setlocal

set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=rag-cliente"

set "PYTHON_VERSION=%~2"
if "%PYTHON_VERSION%"=="" set "PYTHON_VERSION=3.11"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda is not available in PATH. Install Miniconda or Anaconda and try again.
    exit /b 1
)

set "PROJECT_ROOT=%~dp0"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%requirements.txt"

if not exist "%REQUIREMENTS_FILE%" (
    echo requirements.txt was not found at: %REQUIREMENTS_FILE%
    exit /b 1
)

echo Creating Conda environment "%ENV_NAME%" with Python %PYTHON_VERSION%...
call conda create -n "%ENV_NAME%" python=%PYTHON_VERSION% -y
if errorlevel 1 exit /b 1

echo Installing dependencies from requirements.txt...
call conda run -n "%ENV_NAME%" python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

call conda run -n "%ENV_NAME%" python -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 exit /b 1

echo.
echo Environment created successfully.
echo Activate it with:
echo conda activate %ENV_NAME%

endlocal
