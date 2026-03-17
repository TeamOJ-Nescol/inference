@echo off
setlocal

cd /d "%~dp0"

if not exist .venv (
  echo Virtual environment not found. Run install.cmd first.
  exit /b 1
)

call .venv\Scripts\activate.bat

set "PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%"

if not defined MODEL_PATH (
  set "MODEL_PATH=%CD%\model\checkpoint_best_ema.pth"
)

if not exist "%MODEL_PATH%" (
  echo Model checkpoint not found at %MODEL_PATH%
  echo Set MODEL_PATH or place checkpoint_best_ema.pth under model\.
  exit /b 1
)

python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
