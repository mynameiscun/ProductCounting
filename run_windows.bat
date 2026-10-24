@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title ProductCounting - Windows Launcher

REM Always run from the folder containing this BAT file.
cd /d "%~dp0"

set "LOG=%CD%\startup_log.txt"
set "VENV=%CD%\.venv"
set "APP_PY=%VENV%\Scripts\python.exe"

> "%LOG%" echo ProductCounting startup diagnostic
>> "%LOG%" echo Working directory: %CD%

if not exist "main.py" (
    echo [ERROR] Khong tim thay main.py.
    echo Hay GIAI NEN TOAN BO ZIP, roi chay BAT trong thu muc chua main.py.
    >> "%LOG%" echo ERROR: Missing main.py
    goto FAILED
)
if not exist "requirements.txt" (
    echo [ERROR] Khong tim thay requirements.txt.
    >> "%LOG%" echo ERROR: Missing requirements.txt
    goto FAILED
)

echo =========================================
echo  ProductCounting - Khoi dong ung dung
echo =========================================
echo.

REM Reuse existing local Python environment when possible.
if exist "%APP_PY%" (
    "%APP_PY%" --version >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [ERROR] Moi truong .venv cu khong su dung duoc.
        echo Hay xoa thu muc .venv trong Project roi chay lai file BAT.
        >> "%LOG%" echo ERROR: Existing venv Python cannot start
        goto FAILED
    )
) else (
    REM Prefer a well-supported installed Python version.
    set "PY_LAUNCHER="
    where py >nul 2>&1
    if not errorlevel 1 (
        for %%V in (3.12 3.11 3.13 3.10 3) do (
            if not defined PY_LAUNCHER (
                py -%%V -c "import sys; assert (3, 10) <= sys.version_info[:2] < (3, 15)" >nul 2>&1
                if not errorlevel 1 set "PY_LAUNCHER=py -%%V"
            )
        )
    )
    if not defined PY_LAUNCHER (
        python -c "import sys; assert (3, 10) <= sys.version_info[:2] < (3, 15)" >nul 2>&1
        if not errorlevel 1 set "PY_LAUNCHER=python"
    )
    if not defined PY_LAUNCHER (
        echo [ERROR] Khong tim thay Python phu hop ^(3.10 - 3.14^).
        echo Khuyen nghi cai Python 3.12 tu https://www.python.org/downloads/
        echo Sau khi cai, mo lai file BAT. KHONG can Run as Administrator.
        >> "%LOG%" echo ERROR: No supported Python interpreter found
        goto FAILED
    )

    echo [1/3] Tao moi truong Python rieng cho Project...
    echo Selected launcher: !PY_LAUNCHER! >> "%LOG%"
    !PY_LAUNCHER! -m venv "%VENV%" >> "%LOG%" 2>&1
    if errorlevel 1 goto FAILED
)

REM Check application libraries; install only if needed.
"%APP_PY%" -c "import cv2, customtkinter, numpy, tkinter; from PIL import Image" >nul 2>&1
if errorlevel 1 (
    echo [2/3] Cai thu vien con thieu ^(can Internet lan dau^)...
    "%APP_PY%" -m pip install --disable-pip-version-check -r "requirements.txt" >> "%LOG%" 2>&1
    if errorlevel 1 goto FAILED
) else (
    echo [2/3] Thu vien da san sang.
)

"%APP_PY%" -c "import cv2, customtkinter, numpy, tkinter; from PIL import Image" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] Chua nap duoc thu vien hoac tkinter.
    goto FAILED
)

echo [3/3] Mo ung dung ProductCounting...
>> "%LOG%" echo Launching main.py
"%APP_PY%" -u "main.py" >> "%LOG%" 2>&1
if errorlevel 1 goto FAILED

>> "%LOG%" echo Application exited normally.
exit /b 0

:FAILED
echo.
echo [ERROR] Khong the khoi dong ung dung.
echo Chi tiet duoc luu tai: "%LOG%"
echo.
echo ===== NOI DUNG LOI =====
type "%LOG%"
echo ========================
echo.
echo Hay gui file startup_log.txt de kiem tra loi.
pause
exit /b 1
