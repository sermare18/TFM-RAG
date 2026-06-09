param(
    [string]$EnvName = "rag-cliente",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda is not available in PATH. Install Miniconda or Anaconda and try again."
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path $RequirementsFile)) {
    throw "requirements.txt was not found at: $RequirementsFile"
}

Write-Host "Creating Conda environment '$EnvName' with Python $PythonVersion..."
conda create -n $EnvName python=$PythonVersion -y

Write-Host "Installing dependencies from requirements.txt..."
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install -r $RequirementsFile

Write-Host ""
Write-Host "Environment created successfully."
Write-Host "Activate it with:"
Write-Host "conda activate $EnvName"
