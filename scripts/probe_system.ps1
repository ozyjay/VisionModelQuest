[CmdletBinding()]
param(
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = if (Test-Path '.venv-rocm72/bin/python') { '.venv-rocm72/bin/python' } elseif (Test-Path '.venv/bin/python') { '.venv/bin/python' } else { 'python3' }
$Arguments = @('-m', 'visionmodelquest.cli', 'probe')
if ($Output) {
    $Arguments += @('--output', $Output)
}
& $Python @Arguments

