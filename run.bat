@echo off
setlocal
cd /d "%~dp0truthlens\backend"

set PY_CMD=
python --version >nul 2>nul
if %errorlevel% equ 0 (
    set PY_CMD=python
) else (
    py --version >nul 2>nul
    if %errorlevel% equ 0 (
        set PY_CMD=py
    ) else (
        python3 --version >nul 2>nul
        if %errorlevel% equ 0 (
            set PY_CMD=python3
        )
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python 3.10+ is not installed or not working on your system PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo (Make sure to check "Add Python to PATH" during installation)
    echo.
    echo Alternatively, if you have Docker Desktop running:
    echo   docker compose up --build
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment (.venv)...
    %PY_CMD% -m venv .venv
)

call .venv\Scripts\activate.bat
echo Installing requirements...
pip install -q -r requirements.txt

if not exist .env (
    if exist .env.example copy .env.example .env
)

echo =======================================================
echo TruthLens is running!
echo Access the full website at: http://localhost:8000
echo API documentation:          http://localhost:8000/docs
echo =======================================================
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
