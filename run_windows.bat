@echo off
cd /d "%~dp0"
python main.py
if errorlevel 1 (
  echo.
  echo Neu bao loi thieu thu vien, chay: python -m pip install -r requirements.txt
  pause
)
