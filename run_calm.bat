@echo off
cd /d "%~dp0"

REM Free the BLE link if a previous rgb_lights run is still holding it
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Where-Object { $_.CommandLine -match 'rgb_lights' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
call .venv\Scripts\activate.bat
python -m rgb_lights calm %*
