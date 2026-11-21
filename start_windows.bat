@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto ready
where py >nul 2>nul
if errorlevel 1 goto pythoncmd
py -3 -m venv .venv
if errorlevel 1 goto fail
goto install
:pythoncmd
python -m venv .venv
if errorlevel 1 goto fail
:install
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
:ready
.venv\Scripts\python.exe -c "import cv2, numpy, PIL"
if errorlevel 1 goto install
.venv\Scripts\python.exe bot.py
pause
exit /b
:fail
echo Setup failed. Install Python 3.10+ x64 with pip and internet access.
pause
exit /b 1
