$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"

$ProjectRoot = "D:\Weather and Climate Big Data Analytics Project Using Open-Meteo API\weather-clustering"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDirectory = Join-Path $ProjectRoot "logs\ingestion"

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDirectory "ingestion_$Timestamp.log"

Write-Host "============================================================"
Write-Host "Weather Clustering - Automated Ingestion"
Write-Host "Started: $(Get-Date)"
Write-Host "============================================================"

Set-Location $ProjectRoot

& $PythonExe "src\ingest_historical.py" @args *> $LogFile

$ExitCode = $LASTEXITCODE

$ManifestPath = Join-Path $ProjectRoot "data\checkpoints\era5_100cities_2016-01-01_2025-12-31\historical_manifest.json"
if (Test-Path $ManifestPath) {
    try {
        $Manifest = Get-Content $ManifestPath -Encoding UTF8 | ConvertFrom-Json
        $SuccessCount = ($Manifest.items.PSObject.Properties | Where-Object { $_.Value.status -eq 'success' }).Count
        if ($SuccessCount -ge 1305) {
            Write-Host "All 1,305 work units successfully completed! Removing scheduled task..."
            Unregister-ScheduledTask -TaskName "WeatherClustering-OpenMeteo-Ingestion" -Confirm:$false -ErrorAction SilentlyContinue
        }
    } catch {}
}

Write-Host ""
Write-Host "Ingestion process finished."
Write-Host "Exit code: $ExitCode"
Write-Host "Log file: $LogFile"

exit $ExitCode
