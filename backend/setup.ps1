param(
    [string]$EnvName = "rag-cliente",
    [string]$PythonVersion = "3.11",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto"
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

function Test-UsableNvidiaGpu {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $false
    }

    try {
        & $nvidiaSmi.Source --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda no esta disponible en PATH. Instala Miniconda/Anaconda y vuelve a intentarlo."
}

$ResolvedDevice = $Device
if ($Device -eq "auto") {
    $ResolvedDevice = if (Test-UsableNvidiaGpu) { "cuda" } else { "cpu" }
}

$TorchIndexUrl = if ($ResolvedDevice -eq "cuda") {
    "https://download.pytorch.org/whl/cu130"
}
else {
    "https://download.pytorch.org/whl/cpu"
}

Write-Host "Dispositivo solicitado: $Device; variante PyTorch seleccionada: $ResolvedDevice"

$existingEnvs = conda env list --json | ConvertFrom-Json
$envExists = $existingEnvs.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName }

if (-not $envExists) {
    Write-Host "Creando el entorno '$EnvName' con Python $PythonVersion..."
    Invoke-Checked { conda create -n $EnvName "python=$PythonVersion" -y }
}
else {
    Write-Host "Actualizando el entorno existente '$EnvName'..."
}

Write-Host "Instalando explícitamente PyTorch para $ResolvedDevice desde $TorchIndexUrl..."
Invoke-Checked { conda run -n $EnvName python -m pip install --upgrade pip }
Invoke-Checked {
    conda run -n $EnvName python -m pip install --upgrade --force-reinstall `
        "torch==2.12.1" `
        "torchvision==0.27.1" `
        --index-url $TorchIndexUrl
}

Write-Host "Instalando dependencias y el paquete local..."
Invoke-Checked {
    conda run -n $EnvName python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}
Invoke-Checked { conda run -n $EnvName python -m pip install -e $ProjectRoot --no-deps }

if ($ResolvedDevice -eq "cuda") {
    Write-Host "Validando la variante CUDA seleccionada..."
    Invoke-Checked {
        conda run -n $EnvName python -c "import torch; assert torch.version.cuda is not None, 'Se instaló una distribución CPU de PyTorch'; assert torch.cuda.is_available(), 'PyTorch CUDA no detecta una GPU NVIDIA utilizable'; print(f'OK CUDA: {torch.__version__} | {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}')"
    }
}
else {
    Write-Host "Validando la variante CPU seleccionada..."
    Invoke-Checked {
        conda run -n $EnvName python -c "import torch; assert torch.version.cuda is None, f'Se instaló una distribución CUDA de PyTorch ({torch.version.cuda})'; print(f'OK CPU: {torch.__version__}')"
    }
}

Write-Host ""
Write-Host "Entorno preparado. Usa:"
Write-Host "  rag.bat gpu  # muestra la variante de PyTorch y el hardware visible"
Write-Host "  rag.bat api"
