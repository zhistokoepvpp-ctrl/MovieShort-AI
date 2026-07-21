@echo off
title MovieShort AI - Setup
cd /d "%~dp0"

echo ========================================
echo     MovieShort AI - Setup
echo ========================================
echo.

:: --- Check Python -------------------------------------------
python --version >nul 2>&1
if %errorlevel% equ 0 goto python_in_path

:: Python not in PATH — try py launcher (installed with Python even without PATH)
:: Try known-stable versions first (3.14 is too new for some deps)
py --version >nul 2>&1
if %errorlevel% equ 0 (
    for %%v in (3.11 3.12 3.13) do (
        py -%%v --version >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON=py -%%v"
            for /f "tokens=2" %%a in ('py -%%v --version 2^>^&1') do set py_ver=%%a
            goto py_found
        )
    )
    :: Fallback to latest (3.14+)
    for /f "tokens=2" %%v in ('py --version 2^>^&1') do set py_ver=%%v
    set "PYTHON=py -3"
    echo [WARNING] Python %py_ver% is very new.
    echo          If pip fails, install Python 3.11/3.12 from python.org and re-run.
    goto py_found
)

:py_found
echo [OK] Python %py_ver% (via py launcher)
goto python_found

:: Search common install directories
echo [INFO] Python not in PATH — scanning standard locations...
for %%d in (
    "C:\Program Files\Python313"
    "C:\Program Files\Python312"
    "C:\Program Files\Python311"
    "C:\Program Files\Python310"
    "C:\Program Files\Python39"
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "%LOCALAPPDATA%\Programs\Python\Python39"
) do (
    if exist "%%~d\python.exe" (
        set "PYTHON="%%~d\python.exe""
        for /f "tokens=2" %%v in ('"%%~d\python.exe" --version 2^>^&1') do set py_ver=%%v
        echo [OK] Python %py_ver% (found in %%~d)
        goto python_found
    )
)

:: Check Microsoft Store Python
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    set "PYTHON="%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe""
    for /f "tokens=2" %%v in ('"%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" --version 2^>^&1') do set py_ver=%%v
    echo [OK] Python %py_ver% (Microsoft Store)
    goto python_found
)

:: No Python found anywhere
cls
echo ========================================
echo     MovieShort AI - Setup
echo ========================================
echo.
echo [ERROR] Python not found!
echo.
echo Python 3.9+ is required to run this program.
echo Download: https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during install.
echo.
echo If Python is already installed but not in PATH:
echo   1. Run the installer again
echo   2. Select "Modify"
echo   3. Check "Add Python to environment variables"
echo.
pause
exit /b 1

:python_in_path
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set py_ver=%%i
echo [OK] Python %py_ver%

:python_found

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

:: Upgrade pip, setuptools, wheel for proper dependency resolution
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Could not install dependencies
    echo.
    echo Common fixes:
    echo - Python 3.14 may be too new for some packages.
    echo   Install Python 3.11/3.12/3.13 from python.org and re-run setup.bat.
    echo - If you have a slow internet connection, some downloads may time out.
    echo   Try running: pip install -r requirements.txt
    echo.
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
