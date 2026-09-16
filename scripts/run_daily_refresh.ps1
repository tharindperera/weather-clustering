$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"

$ProjectRoot = "D:\Weather and Climate Big Data Analytics Project Using Open-Meteo API\weather-clustering"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDirectory = Join-Path $ProjectRoot "logs\refresh"

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDirectory "refresh_$Timestamp.log"

Write-Host "============================================================"
Write-Host "Recent ECMWF IFS - Daily Automated Refresh"
Write-Host "Started: $(Get-Date)"
Write-Host "============================================================"

Set-Location $ProjectRoot

& $PythonExe "src\refresh_recent_ifs.py" @args *> $LogFile

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "Refresh process finished."
Write-Host "Exit code: $ExitCode"
Write-Host "Log file: $LogFile"

exit $ExitCode
