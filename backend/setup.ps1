param(
    [string]$EnvName = "rag-cliente",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "El comando anterior ha fallado con codigo $LASTEXITCODE."
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda no esta disponible en PATH."
}

$existingEnvs = conda env list --json | ConvertFrom-Json
$envExists = $existingEnvs.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName }
if (-not $envExists) {
    Write-Host "Creando el entorno '$EnvName' con Python $PythonVersion..."
    Invoke-Checked { conda create -n $EnvName "python=$PythonVersion" -y }
}

Write-Host "Instalando dependencias y el paquete local..."
Invoke-Checked { conda run -n $EnvName python -m pip install --upgrade pip }
Invoke-Checked {
    conda run -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}
Invoke-Checked { conda run -n $EnvName python -m pip install -e $ProjectRoot --no-deps }

Write-Host ""
Write-Host "Entorno preparado. Bedrock sigue desactivado hasta configurar .env."
Write-Host "Usa: .\rag.bat doctor"
