[CmdletBinding()]
param(
    [switch]$Rocm
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Install-Environment {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string[]]$InstallArguments
    )
    if (-not (Test-Path "$Path/bin/python")) {
        & python3 -m venv $Path
    }
    & "$Path/bin/python" -m pip install --upgrade pip
    & "$Path/bin/python" -m pip install @InstallArguments
}

Install-Environment -Path '.venv' -InstallArguments @('-e', '.[dev]')

if ($Rocm) {
    if ([int](& python3 -c 'import sys; print(sys.version_info.minor)') -ne 12) {
        throw 'The pinned ROCm wheels require an active Python 3.12 interpreter.'
    }
    Install-Environment -Path '.venv-rocm72' -InstallArguments @('-r', 'requirements-rocm72.txt', '-e', '.[dev]')
    & .venv-rocm72/bin/python -c "import torch, transformers; print('torch', torch.__version__); print('hip', torch.version.hip); print('transformers', transformers.__version__); print('cuda_available', torch.cuda.is_available())"
}

Write-Host 'VisionModelQuest setup complete. No model weights were downloaded.'

