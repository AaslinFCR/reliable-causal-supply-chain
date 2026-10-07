$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) { throw 'Create .venv and install requirements-runtime.txt first; see DEPLOYMENT.md.' }
$env:HOST = '127.0.0.1'
$env:PORT = '8000'
& '.\.venv\Scripts\python.exe' run.py scrc.api.serve
exit $LASTEXITCODE
