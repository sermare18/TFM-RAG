param(
    [string]$EnvName = "rag-cliente",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "El comando anterior ha fallado con codigo $LASTEXITCODE."
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda no esta disponible en PATH. Instala Miniconda/Anaconda y vuelve a intentarlo."
}

$existingEnvs = conda env list --json | ConvertFrom-Json
$envExists = $existingEnvs.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName }

if (-not $envExists) {
    Write-Host "Creando el entorno '$EnvName' con Python $PythonVersion..."
    Invoke-Checked { conda create -n $EnvName "python=$PythonVersion" -y }
}
else {
    Write-Host "Actualizando el entorno existente '$EnvName'..."
}

Write-Host "Instalando PyTorch 2.12.1 con CUDA 13.0..."
Invoke-Checked { conda run -n $EnvName python -m pip install --upgrade pip }
Invoke-Checked {
    conda run -n $EnvName python -m pip install --upgrade `
        "torch==2.12.1" `
        --index-url $TorchIndexUrl
}

Write-Host "Instalando dependencias y el paquete local..."
Invoke-Checked {
    conda run -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}
Invoke-Checked { conda run -n $EnvName python -m pip install -e $ProjectRoot }

Write-Host "Validando CUDA..."
Invoke-Checked {
    conda run -n $EnvName python -c "import torch; assert torch.cuda.is_available(), 'PyTorch no detecta CUDA'; print(f'OK: {torch.__version__} | {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}')"
}

Write-Host ""
Write-Host "Entorno preparado. Usa:"
Write-Host "  rag.bat gpu"
Write-Host "  rag.bat api"
