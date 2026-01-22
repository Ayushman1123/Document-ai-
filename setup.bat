@echo off
echo ========================================
echo  Document AI System - Setup Script
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.9+
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
cd backend
python -m venv venv

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip
pip install -r requirements.txt

echo [4/5] Setting up environment file...
if not exist .env (
    copy .env.example .env
    echo Created .env file - please add your API keys if needed
)

echo [5/5] Creating data directories...
cd ..
if not exist "data\uploads" mkdir "data\uploads"
if not exist "data\output" mkdir "data\output"
if not exist "data\feedback" mkdir "data\feedback"
if not exist "models" mkdir "models"

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo To run the application:
echo   1. cd backend
echo   2. venv\Scripts\activate
echo   3. python main.py
echo.
echo Then open frontend\index.html in your browser
echo Or run: python -m http.server 5500 in frontend folder
echo.
pause
