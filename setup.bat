@echo off
title MovieShort AI - Setup
cd /d "%~dp0"

echo ========================================
echo     MovieShort AI - Setup
echo ========================================
echo.

:: --- Check / Install Python ---------------------------------
python --version >nul 2>&1
if %errorlevel% equ 0 goto python_found

:: --- Auto-download Python if missing ------------------------
echo [INFO] Python not found. Attempting automatic installation...

:: Strategy 1: winget (Windows 10 1809+ / Windows 11)
winget install Python.Python.3.11 --silent --accept-package-agreements >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python installed via winget.
    set "PYTHON_INSTALLED=1"
    goto refresh_python_path
)

:: Strategy 2: download from python.org
echo [INFO] Downloading Python 3.11 from python.org...
curl -sL -o "%TEMP%\python-installer.exe" https://www.python.org/ftp/python/3.11.11/python-3.11.11-amd64.exe
if %errorlevel% neq 0 (
    echo [ERROR] Could not download Python.
    echo Download manually: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo Installing Python (this may take a minute)...
"%TEMP%\python-installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Shortcuts=0
del "%TEMP%\python-installer.exe" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python installation failed.
    pause
    exit /b 1
)
echo [OK] Python installed.
set "PYTHON_INSTALLED=1"

:refresh_python_path
:: Refresh PATH so the newly installed Python is visible
set "PATH=%LocalAppData%\Programs\Python\Python311\;%LocalAppData%\Programs\Python\Python311\Scripts\;%ProgramFiles%\Python311\;%ProgramFiles%\Python311\Scripts\;%PATH%"

python --version >nul 2>&1
if %errorlevel% equ 0 goto python_found

:: Last resort: try py launcher
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python launcher found. Using py.exe.
    set "PYTHON=py"
    goto check_ffmpeg
)

echo [ERROR] Python installed but not found in PATH.
echo Please restart this script or add Python manually to PATH.
pause
exit /b 1

:python_found
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set py_ver=%%i
echo [OK] Python %py_ver%

:: --- Check FFmpeg ------------------------------------------
:check_ffmpeg
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

if defined PYTHON (
    %PYTHON% -m venv venv
) else (
    python -m venv venv
)
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
