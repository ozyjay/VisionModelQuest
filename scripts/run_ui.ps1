$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& bash ./scripts/run_ui.sh @args
if ($LASTEXITCODE -ne 0) {
    throw "The native interface exited with code $LASTEXITCODE."
}
