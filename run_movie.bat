@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Starting movie mode...
echo Tip: close calibrate-color / the phone LED app first (only one BLE connection).
echo.
python -m rgb_lights movie %*
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
  echo.
  echo Movie mode exited with error %EXITCODE%.
  echo If it says device not found: power the strip on and close calibrate-color.
  pause
)
exit /b %EXITCODE%
