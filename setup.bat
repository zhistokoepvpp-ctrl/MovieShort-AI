@echo off
title MovieShort AI - Setup
cd /d "%~dp0"

echo ========================================
echo     MovieShort AI - Setup
echo ========================================
echo.

:: --- Check Python -----------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo Download Python 3.9-3.11: https://www.python.org/downloads/
    echo Check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set py_ver=%%i
echo [OK] Python %py_ver%

:: --- Check FFmpeg ------------------------------------------
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] FFmpeg not found — needed for video processing.
    winget install Gyan.FFmpeg --source winget >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] FFmpeg installed via winget
    ) else (
        echo.
        echo [WARNING] Could not install FFmpeg automatically.
        echo Download manually: https://www.gyan.dev/ffmpeg/builds/
        echo Extract and add bin\ to system PATH.
        echo.
    )
) else (
    echo [OK] FFmpeg found
)

:: --- Create virtual environment ----------------------------
echo.
echo Creating virtual environment...
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv
)
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Could not create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created

:: --- Install base dependencies -----------------------------
echo.
echo Installing dependencies...
call venv\Scripts\activate.bat

pip install --upgrade pip -q
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Could not install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

:: --- Ask about CUDA ----------------------------------------
echo.
echo +-----------------------------------------------------+
echo | Do you have an NVIDIA GPU?                          |
echo | PyTorch with CUDA speeds up transcription 5-10x!     |
echo | (CUDA: ~3-5 min, CPU only: ~30-40 min per movie)    |
echo +-----------------------------------------------------+
echo.
choice /c YN /n /m "Install PyTorch with CUDA? [Y/N]: "
if errorlevel 2 goto no_cuda
if errorlevel 1 goto yes_cuda

:yes_cuda
echo.
echo Installing PyTorch with CUDA 12.4...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% neq 0 (
    echo [WARNING] Error installing PyTorch CUDA.
    echo Installing CPU version as fallback...
    pip install torch torchvision torchaudio
)
goto verify_cuda

:no_cuda
echo.
echo Installing CPU version of PyTorch...
pip install torch torchvision torchaudio

:verify_cuda
echo.
echo Verifying installation...
python -c "
import torch
cuda = torch.cuda.is_available()
if cuda:
    name = torch.cuda.get_device_name(0)
    print(f'[OK] CUDA ready — {name}')
else:
    print('[INFO] CUDA not available — using CPU')
"
if %errorlevel% neq 0 (
    echo [WARNING] torch import failed, retrying...
    pip install torch torchvision torchaudio
)

:done
echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.
echo To start the program, double-click:
echo   run.bat
echo.
pause
