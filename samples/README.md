# DroneMap Test Samples

This directory contains local test samples for running, benchmarking, and demonstrating the **DroneMap v2.0** reconstruction pipeline.

## Files

- **`drone_flight_sample.mp4`**: A 20-second 1080p aerial drone survey video (H.264 / 30 fps) capturing an outdoor recreation facility with buildings, paved walkways, trees, and ground terrain.
- **`drone_flight_sample.srt`**: Companion DJI-standard subtitle flight telemetry embedding per-frame GNSS coordinates, relative/absolute altitudes, and gimbal angles.
- **`drone_flight_sample.csv`**: Tabular flight telemetry export with timestamps, coordinates, and drone orientation.

## Usage

### 1. Dry Run (Preview execution stages)
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_flight_sample.mp4 --dry-run
```

### 2. Full Reconstruction with GNSS Georeferencing
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_flight_sample.mp4 --telemetry samples/drone_flight_sample.srt
```

### 3. Reconstruction in Uncalibrated / Metric Relative Mode
```powershell
.venv\Scripts\dronemap.exe run --video samples/drone_flight_sample.mp4 --no-telemetry
```

### 4. Interactive Web Studio
Launch the viewer and upload `drone_flight_sample.mp4` directly via drag-and-drop:
```powershell
.venv\Scripts\dronemap.exe serve
```
