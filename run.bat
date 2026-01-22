@echo off
echo ========================================
echo  Document AI System - Runner
echo ========================================
echo.

cd backend
call venv\Scripts\activate.bat

echo Starting backend server on http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py
