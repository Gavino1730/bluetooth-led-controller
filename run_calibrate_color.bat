@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m rgb_lights calibrate-color %*
