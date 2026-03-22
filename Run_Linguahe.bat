@echo off
title Linguahe - Linguistic Diagnostic Engine
color 0b

:: 1. Check for FFmpeg (Required for Whisper/Librosa)
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] FFmpeg not found! 
    echo Please install FFmpeg and add it to your System PATH.
    echo Download: https://ffmpeg.org/download.html
    pause
    exit
)

:: 2. Check for Virtual Environment
if not exist venv (
    echo ======================================================
    echo [FIRST TIME SETUP] Creating virtual environment...
    python -m venv venv
    echo [FIRST TIME SETUP] Installing AI Models and Audio Tools...
    echo (This will take a few minutes. Please wait.)
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo ======================================================
)

:: 3. Detect Local IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address" ^| findstr /v "192.168.56.1"') do (
    set IP=%%a
    goto :found_ip
)

:found_ip
set IP=%IP:~1%

:: 4. Launching the App
cls
echo ======================================================
echo           LINGUAHE SYSTEM IS ACTIVE
echo ======================================================
echo  LOCAL URL:  http://localhost:5000
echo  MOBILE URL: http://%IP%:5000
echo ======================================================
echo  (Press Ctrl+C to shut down the server safely)
echo.

call venv\Scripts\activate
python app.py
pause