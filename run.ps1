param([switch]$Offline)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$researchPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $researchPython)) {
    $researchPython = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $researchPython)) {
    throw 'Install Python dependencies as described in README.md first.'
}
$researchArguments = @((Join-Path $PSScriptRoot 'run.py'), 'scrc.pipeline')
if ($Offline) { $researchArguments += '--offline' }
& $researchPython @researchArguments
if ($LASTEXITCODE -ne 0) { throw 'Research pipeline failed; inspect reports/pipeline_status.json.' }
