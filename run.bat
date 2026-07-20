@echo off
title MovieShort AI
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found.
    echo Run setup.bat first to install dependencies.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python main.py

echo.
echo Done.
echo.
pause
