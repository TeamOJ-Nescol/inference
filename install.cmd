@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher for Windows ^(`py`^) is required. Install Python 3.12 and rerun this script.
  exit /b 1
)

py -3.12 -m venv .venv
if errorlevel 1 (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Python 3.11 or 3.12 is required. Install one of them and rerun this script.
    exit /b 1
  )
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist model mkdir model

echo.
echo Environment created in .venv.
echo Place the model checkpoint at model\checkpoint_best_ema.pth
echo or set MODEL_PATH before running the server.
