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
        [Parameter(Mandatory)][string]$BootstrapPython,
        [Parameter(Mandatory)][string[]]$InstallArguments
    )
    if (-not (Test-Path "$Path/bin/python")) {
        & $BootstrapPython -m venv $Path
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create $Path with $BootstrapPython."
        }
    }
    & "$Path/bin/python" -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip in $Path."
    }
    & "$Path/bin/python" -m pip install @InstallArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install dependencies in $Path."
    }
}

function Resolve-Python312 {
    $DirectCommand = Get-Command 'python3.12' -ErrorAction SilentlyContinue
    if ($DirectCommand) {
        $DirectVersion = & $DirectCommand.Source -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>$null
        if ($LASTEXITCODE -eq 0 -and $DirectVersion.Trim() -eq '3.12') {
            return $DirectCommand.Source
        }
    }

    $PyenvCommand = Get-Command 'pyenv' -ErrorAction SilentlyContinue
    if ($PyenvCommand) {
        $InstalledVersions = @(& $PyenvCommand.Source versions --bare 2>$null) |
            Where-Object { $_ -match '^3\.12(?:\.|$)' } |
            Sort-Object { [version]$_ } -Descending
        foreach ($Version in $InstalledVersions) {
            $Prefix = & $PyenvCommand.Source prefix $Version 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $Prefix) {
                continue
            }
            $Candidate = Join-Path $Prefix.Trim() 'bin/python'
            if (-not (Test-Path $Candidate)) {
                continue
            }
            $CandidateVersion = & $Candidate -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
            if ($LASTEXITCODE -eq 0 -and $CandidateVersion.Trim() -eq '3.12') {
                return $Candidate
            }
        }
    }

    throw (
        'The pinned ROCm wheels require Python 3.12. Install python3.12 or a Python 3.12 ' +
        'version through pyenv, then rerun setup.'
    )
}

Install-Environment -Path '.venv' -BootstrapPython 'python3' -InstallArguments @('-e', '.[dev]')

if ($Rocm) {
    $RocmPython = Resolve-Python312
    $RocmBootstrapVersion = & $RocmPython -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    if ($LASTEXITCODE -ne 0 -or $RocmBootstrapVersion.Trim() -ne '3.12') {
        throw "The resolved ROCm bootstrap interpreter is Python $RocmBootstrapVersion, not Python 3.12."
    }
    if (Test-Path '.venv-rocm72/bin/python') {
        $ExistingRocmVersion = & .venv-rocm72/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
        if ($LASTEXITCODE -ne 0 -or $ExistingRocmVersion.Trim() -ne '3.12') {
            throw (
                "Existing .venv-rocm72 uses Python $ExistingRocmVersion. " +
                'Move or remove that environment explicitly, then rerun setup.'
            )
        }
    }
    Install-Environment `
        -Path '.venv-rocm72' `
        -BootstrapPython $RocmPython `
        -InstallArguments @('-r', 'requirements-rocm72.txt', '-e', '.[dev]')
    & .venv-rocm72/bin/python -c "import torch, transformers; print('torch', torch.__version__); print('hip', torch.version.hip); print('transformers', transformers.__version__); print('cuda_available', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) {
        throw 'The ROCm environment was installed, but its runtime verification failed.'
    }
}

Write-Host 'VisionModelQuest setup complete. No model weights were downloaded.'
