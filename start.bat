@echo off
setlocal enabledelayedexpansion
title DroneMap v2.0 - Automated Photogrammetry Suite

echo ===============================================================================
echo                DroneMap v2.0 - Production Photogrammetry Suite
echo          100%% Offline Core + Local GPU AI Stretch + 3D Web Studio
echo ===============================================================================
echo.

:: 1. Verify virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please create the virtual environment and install requirements first.
    pause
    exit /b 1
)

:: 2. Check Offline Core & Local AI GPU Status
.venv\Scripts\python.exe scripts\check_ai_status.py
echo.

:: 3. Present Menu
:MENU
echo ===============================================================================
echo  Select an operation:
echo ===============================================================================
echo  [1] Launch Interactive Web Studio (3D Viewer, Copilot Chat, Measurement Tools)
echo  [2] Run 3D Reconstruction on Dataset #1: Castle Ruins (Toolse Castle, 12s)
echo  [3] Run 3D Reconstruction on Dataset #2: Medieval Fortress Orbit (Haapsalu, 14s)
echo  [4] Run 3D Reconstruction on Dataset #3: Urban Plaza ^& Arch Bridge (Tartu, 14s)
echo  [5] Run 3D Reconstruction on Dataset #4: Botanical Garden Canopy (12s)
echo  [6] Run 3D Reconstruction on Dataset #5: High-Rise City Blocks (Annelinn, 12s)
echo  [7] Run 3D Reconstruction on Dataset #6: Recreation Facility Park (20s)
echo  [8] Download / Refresh All Curated Test Datasets (scripts\download_datasets.py)
echo  [9] Run Environment Health Check (dronemap doctor) ^& Test Suite (pytest)
echo  [0] Exit
echo ===============================================================================
set /p CHOICE="Enter choice (0-9) [default: 1]: "

if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" goto SERVE
if "%CHOICE%"=="2" (
    set VID=samples\drone_castle_survey.mp4
    set TEL=samples\drone_castle_survey.srt
    goto RUN_DATASET
)
if "%CHOICE%"=="3" (
    set VID=samples\drone_fortress_orbit.mp4
    set TEL=samples\drone_fortress_orbit.srt
    goto RUN_DATASET
)
if "%CHOICE%"=="4" (
    set VID=samples\drone_urban_bridge.mp4
    set TEL=samples\drone_urban_bridge.srt
    goto RUN_DATASET
)
if "%CHOICE%"=="5" (
    set VID=samples\drone_botanical_garden.mp4
    set TEL=samples\drone_botanical_garden.srt
    goto RUN_DATASET
)
if "%CHOICE%"=="6" (
    set VID=samples\drone_city_blocks.mp4
    set TEL=samples\drone_city_blocks.srt
    goto RUN_DATASET
)
if "%CHOICE%"=="7" (
    set VID=samples\drone_flight_sample.mp4
    set TEL=samples\drone_flight_sample.srt
    goto RUN_DATASET
)
if "%CHOICE%"=="8" goto DOWNLOAD_ALL
if "%CHOICE%"=="9" goto DOCTOR_AND_TEST
if "%CHOICE%"=="0" goto QUIT

echo [!] Invalid selection, please try again.
goto MENU

:SERVE
echo.
echo [*] Launching DroneMap Measurement Studio at http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
.venv\Scripts\dronemap.exe serve --port 8000
goto QUIT

:RUN_DATASET
echo.
if not exist "%VID%" (
    echo [*] Dataset video %VID% not found locally. Downloading curated datasets...
    .venv\Scripts\python.exe scripts\download_datasets.py
)
echo [*] Starting 3D reconstruction pipeline with GNSS telemetry...
echo     Video:     %VID%
echo     Telemetry: %TEL%
echo.
.venv\Scripts\dronemap.exe run --video "%VID%" --telemetry "%TEL%" --set mesh.refine=false
echo.
echo [*] Pipeline run finished. Press any key to return to menu...
pause >nul
goto MENU

:DOWNLOAD_ALL
echo.
echo [*] Downloading and preparing all curated drone survey datasets...
.venv\Scripts\python.exe scripts\download_datasets.py
echo.
pause
goto MENU

:DOCTOR_AND_TEST
echo.
echo [*] Running DroneMap environment diagnostics...
.venv\Scripts\dronemap.exe doctor
echo.
echo [*] Running automated verification test suite...
.venv\Scripts\python.exe -m pytest -q
echo.
pause
goto MENU

:QUIT
echo.
echo Thank you for using DroneMap v2.0.
exit /b 0
