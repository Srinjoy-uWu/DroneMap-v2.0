@echo off
title DroneMap v2.0 - Measurement Studio
echo ===============================================================================
echo            DroneMap v2.0 - Measurement Studio & AI Copilot
echo ===============================================================================
echo.

:: Check AI Booster Layer / API Key Status
.venv\Scripts\python.exe scripts\check_ai_status.py
echo.

echo [*] Starting Web Studio on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
.venv\Scripts\dronemap.exe serve --port 8000
