@echo off
setlocal enabledelayedexpansion
title DroneMap v2.0 - Automated Suite

echo ===============================================================================
echo                DroneMap v2.0 - Production Photogrammetry Suite
echo            Single-Pass Drone Video to Georeferenced 3D Model
echo ===============================================================================
echo.

:: 1. Verify virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please create the virtual environment and install requirements first.
    pause
    exit /b 1
)

:: 2. Verify and prepare local test sample video
if not exist "samples\drone_flight_sample.mp4" (
    echo [*] Sample drone video not detected locally. Downloading and preparing sample assets...
    .venv\Scripts\python.exe scripts\download_sample_video.py
    echo.
)

:: 3. Check AI Booster Layer / API Key Status
echo [*] Checking AI Booster Layer & API Credentials...
.venv\Scripts\python.exe scripts\check_ai_status.py
echo.

:: 4. Present Menu
:MENU
echo ===============================================================================
echo  Select an operation:
echo ===============================================================================
echo  [1] Launch Interactive Web Studio (3D Viewer, Copilot Chat, Measurement Tools)
echo  [2] Run 3D Reconstruction on Sample Video (with GNSS Telemetry)
echo  [3] Run 3D Reconstruction on Sample Video (Relative Scale / No Telemetry)
echo  [4] Run Dry-Run Plan Preview on Sample Video
echo  [5] Run Environment Health Check (dronemap doctor)
echo  [6] Run Automated Test Suite (172 tests)
echo  [7] Exit
echo ===============================================================================
set /p CHOICE="Enter choice (1-7) [default: 1]: "

if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" goto SERVE
if "%CHOICE%"=="2" goto RUN_SAMPLE_TELEM
if "%CHOICE%"=="3" goto RUN_SAMPLE_NOTELEM
if "%CHOICE%"=="4" goto DRY_RUN
if "%CHOICE%"=="5" goto DOCTOR
if "%CHOICE%"=="6" goto TEST
if "%CHOICE%"=="7" goto QUIT

echo [!] Invalid selection, please try again.
goto MENU

:SERVE
echo.
echo [*] Launching DroneMap Measurement Studio at http://127.0.0.1:8000 ...
echo [*] Opening browser in 3 seconds...
start "" "http://127.0.0.1:8000"
.venv\Scripts\dronemap.exe serve --port 8000
goto QUIT

:RUN_SAMPLE_TELEM
echo.
echo [*] Starting full 3D reconstruction pipeline with GNSS telemetry...
echo     Video:     samples\drone_flight_sample.mp4
echo     Telemetry: samples\drone_flight_sample.srt
echo.
.venv\Scripts\dronemap.exe run --video samples\drone_flight_sample.mp4 --telemetry samples\drone_flight_sample.srt
echo.
echo [*] Pipeline run finished. Press any key to return to menu...
pause >nul
goto MENU

:RUN_SAMPLE_NOTELEM
echo.
echo [*] Starting 3D reconstruction pipeline in relative / no-telemetry mode...
echo     Video: samples\drone_flight_sample.mp4
echo.
.venv\Scripts\dronemap.exe run --video samples\drone_flight_sample.mp4 --no-telemetry
echo.
echo [*] Pipeline run finished. Press any key to return to menu...
pause >nul
goto MENU

:DRY_RUN
echo.
echo [*] Executing dry-run stage planner preview...
echo.
.venv\Scripts\dronemap.exe run --video samples\drone_flight_sample.mp4 --telemetry samples\drone_flight_sample.srt --dry-run
echo.
pause
goto MENU

:DOCTOR
echo.
echo [*] Running DroneMap environment diagnostics...
echo.
.venv\Scripts\dronemap.exe doctor
echo.
pause
goto MENU

:TEST
echo.
echo [*] Running full automated verification test suite...
echo.
.venv\Scripts\python.exe -m pytest -q
echo.
pause
goto MENU

:QUIT
echo.
echo Thank you for using DroneMap v2.0.
exit /b 0
