[CmdletBinding()]
param(
    [string]$RunPath = (Join-Path $PSScriptRoot "runs\full_study\study_20260711_231032\internal_validation\20260711_231106"),
    [string]$PythonExe = "",
    [string]$CsvPath = "",
    [ValidateRange(1, 10000)][int]$NSamples = 200,
    [ValidateRange(1, 10000)][int]$NSteps = 50,
    [ValidateRange(1, 1000000)][int]$NBoot = 1000,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"

if (-not (Test-Path -LiteralPath $RunPath -PathType Container)) {
    throw "Frozen run directory does not exist: $RunPath"
}

if (-not $PythonExe) {
    $pythonCandidates = @(
        (Join-Path $PSScriptRoot "..\mbsaqip_flow\.venv\Scripts\python.exe"),
        (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
        "C:\Program Files\Python313\python.exe"
    )
    $PythonExe = $pythonCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1

    if (-not $PythonExe) {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $PythonExe = $pythonCommand.Source
        }
    }
}

if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Could not find Python. Pass -PythonExe with the project Python executable."
}

$outputPath = Join-Path $RunPath "figures"
$figureArgs = @(
    "-m", "figures.build_all",
    "--run", $RunPath,
    "--out", $outputPath,
    "--device", "cpu",
    "--n-samples", $NSamples,
    "--n-steps", $NSteps,
    "--n-boot", $NBoot,
    "--seed", $Seed
)

if ($CsvPath) {
    if (-not (Test-Path -LiteralPath $CsvPath -PathType Leaf)) {
        throw "CSV file does not exist: $CsvPath"
    }
    $figureArgs += @("--csv", $CsvPath)
} else {
    $figureArgs += "--use-db"
}

Write-Host "Rebuilding figures for: $RunPath"
Write-Host "Python: $PythonExe"
Write-Host "Output: $outputPath"

Push-Location $PSScriptRoot
try {
    & $PythonExe @figureArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "Figure build failed with exit code $exitCode. Review the build summary above."
}

Write-Host "Figure rebuild completed successfully."
Write-Host "Note: the original RUN_MANIFEST.json still records the first build attempt."
