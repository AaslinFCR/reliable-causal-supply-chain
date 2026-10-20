# Build dist\RCD_SupplyChain.exe (single file) with PyInstaller.
# Run from this folder:  powershell -ExecutionPolicy Bypass -File build_exe.ps1
$venv = "..\.venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    python -m venv $venv
    & "$venv\Scripts\python" -m pip install -r requirements.txt pyinstaller
}
& "$venv\Scripts\pyinstaller" --noconfirm --onefile --console --name RCD_SupplyChain `
    --collect-submodules rcd `
    --hidden-import matplotlib.backends.backend_pdf `
    --hidden-import matplotlib.backends.backend_agg `
    --workpath ..\build --specpath ..\build --distpath ..\dist run_all.py
