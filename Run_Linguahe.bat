@echo off
title Linguahe - One-Click Researcher Tool
color 0b

:: 1. Check if venv exists. If not, create it and install requirements.
if not exist venv (
    echo [FIRST TIME SETUP] Creating virtual environment...
    python -m venv venv
    echo [FIRST TIME SETUP] Installing AI models and libraries...
    echo (This may take a few minutes. Please wait.)
    call venv\Scripts\activate
    pip install -r requirements.txt
)

:: 2. Detect Local IP
for /f "tokens=14" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do set IP=%%a

:: 3. Run the App
echo ==========================================
echo    LINGUAHE IS READY
echo    MOBILE LINK: http://%IP%:5000
echo ==========================================
call venv\Scripts\activate
python app.py
pause