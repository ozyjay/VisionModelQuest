$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path './scripts/stop_ui.sh') {
    & bash ./scripts/stop_ui.sh
    if ($LASTEXITCODE -ne 0) {
        throw "Could not safely stop the native application processes."
    }
}

$PidFile = 'var/active-worker.pid'

if (-not (Test-Path $PidFile)) {
    Write-Host 'No active VisionModelQuest worker is recorded.'
    exit 0
}

$WorkerPidText = (Get-Content -Raw $PidFile).Trim()
$WorkerPid = 0
if (-not [int]::TryParse($WorkerPidText, [ref]$WorkerPid) -or $WorkerPid -le 1) {
    throw 'The active-worker PID file is invalid; no process was stopped.'
}
$CommandLinePath = "/proc/$WorkerPid/cmdline"
if (-not (Test-Path $CommandLinePath)) {
    Remove-Item $PidFile
    Write-Host 'The recorded worker has already exited.'
    exit 0
}
$CommandLine = (Get-Content -Raw $CommandLinePath).Replace([char]0, ' ')
if ($CommandLine -notmatch 'visionmodelquest\.worker') {
    throw 'The recorded PID is not a VisionModelQuest worker; no process was stopped.'
}
Stop-Process -Id $WorkerPid
Remove-Item $PidFile
Write-Host "Stopped VisionModelQuest worker PID $WorkerPid."
