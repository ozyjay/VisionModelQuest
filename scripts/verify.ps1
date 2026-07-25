$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path '.venv/bin/python')) {
    throw 'Missing .venv. Run scripts/setup.ps1 first.'
}

& .venv/bin/python -m ruff check .
if ($LASTEXITCODE -ne 0) {
    throw "Ruff failed with exit code $LASTEXITCODE."
}
& .venv/bin/python -m pytest -m 'not hardware and not rocm and not large_model and not long_running'
if ($LASTEXITCODE -ne 0) {
    throw "Pytest failed with exit code $LASTEXITCODE."
}
