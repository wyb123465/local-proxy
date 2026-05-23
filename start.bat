@echo off
chcp 65001 >nul
title Local Proxy

:loop
echo ============================================
echo   Local Proxy - Starting...
echo   %date% %time%
echo ============================================
echo.

echo [1/2] Stopping ccx...
taskkill /f /im ccx-windows-amd64.exe 2>nul
if %errorlevel% equ 0 (
    echo   ccx stopped.
) else (
    echo   ccx not running.
)

echo.
echo [2/2] Starting local proxy...
echo.

cd /d "%~dp0"

:: Prefer system Python, fall back to venv
where python >nul 2>nul
if %errorlevel% equ 0 (
    python proxy.py
) else if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe proxy.py
) else (
    echo ERROR: Python not found. Install Python or create venv.
    pause
    exit /b 1
)

echo.
echo [%date% %time%] Proxy crashed, restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto loop
