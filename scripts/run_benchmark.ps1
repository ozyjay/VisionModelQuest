[CmdletBinding()]
param(
    [ValidateSet('Quick', 'Standard', 'Stability')]
    [string]$Preset = 'Quick',
    [string[]]$Models = @('mock'),
    [string[]]$Fixtures,
    [double]$DurationSeconds,
    [ValidateRange(40, 120)]
    [double]$MaxTemperatureCelsius = 95,
    [switch]$QualityCapture
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$UseRocm = @($Models | Where-Object { $_ -ne 'mock' }).Count -gt 0
$Python = if ($UseRocm) { '.venv-rocm72/bin/python' } else { '.venv/bin/python' }
if (-not (Test-Path $Python)) {
    throw "Missing benchmark environment $Python. Run scripts/setup.ps1 with the appropriate options."
}

$Arguments = @(
    '-m', 'visionmodelquest.cli', 'run',
    '--preset', $Preset.ToLowerInvariant(),
    '--max-temperature-celsius', $MaxTemperatureCelsius.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--models'
) + $Models
if ($Fixtures) {
    $Arguments += @('--fixtures') + $Fixtures
}
if ($PSBoundParameters.ContainsKey('DurationSeconds')) {
    $Arguments += @('--duration-seconds', $DurationSeconds.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($QualityCapture) {
    $Arguments += '--quality-capture'
}
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark failed with exit code $LASTEXITCODE. A partial report was retained."
}
