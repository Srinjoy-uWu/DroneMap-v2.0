"""Download and prepare a local sample drone survey video with telemetry."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples"
RAW_WEBM = SAMPLES_DIR / "raw_download.webm"
TARGET_MP4 = SAMPLES_DIR / "drone_flight_sample.mp4"
SRT_PATH = SAMPLES_DIR / "drone_flight_sample.srt"
CSV_PATH = SAMPLES_DIR / "drone_flight_sample.csv"
README_PATH = SAMPLES_DIR / "README.md"

WIKIMEDIA_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/b/b0/"
    "Germia_Pool_Prishtina_-_Drone_Video.webm"
)

FFMPEG_BIN = (
    REPO_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "imageio_ffmpeg"
    / "binaries"
    / "ffmpeg-win-x86_64-v7.1.exe"
)


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if not TARGET_MP4.exists() or TARGET_MP4.stat().st_size < 1_000_000:
        print(f"[*] Downloading drone aerial video from Wikimedia Commons...")
        headers = {"User-Agent": "DroneMap/2.0 (contact: srinjoydas396@gmail.com)"}
        with httpx.Client(follow_redirects=True, timeout=120.0, headers=headers) as client:
            resp = client.get(WIKIMEDIA_URL)
            resp.raise_for_status()
            RAW_WEBM.write_bytes(resp.content)
        print(f"    Downloaded raw source: {RAW_WEBM.stat().st_size / 1024 / 1024:.2f} MB")

        print("[*] Transcoding to 20-second 1080p H.264 drone survey MP4 via FFmpeg...")
        cmd = [
            str(FFMPEG_BIN),
            "-y",
            "-ss", "5",
            "-i", str(RAW_WEBM),
            "-t", "20",
            "-vf", "scale=1920:1080",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-an",
            str(TARGET_MP4),
        ]
        subprocess.run(cmd, check=True)
        if RAW_WEBM.exists():
            RAW_WEBM.unlink()
        print(f"    Created: {TARGET_MP4.name} ({TARGET_MP4.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"[+] Sample drone video already present: {TARGET_MP4.name} ({TARGET_MP4.stat().st_size / 1024 / 1024:.2f} MB)")

    # Generate companion DJI SRT telemetry
    print("[*] Generating companion DJI SRT and CSV flight telemetry...")
    base_lat = 42.668500
    base_lon = 21.196500
    base_alt = 680.0
    fps = 30
    duration_s = 20
    n_frames = duration_s * fps

    srt_blocks = []
    csv_rows = [
        "timestamp_s,latitude,longitude,altitude_m,rel_alt_m,roll,pitch,yaw,speed_mps"
    ]

    for i in range(1, n_frames + 1):
        t_start = (i - 1) / fps
        t_end = i / fps

        # Smooth flight path
        lat = base_lat + (t_start * 0.000028)
        lon = base_lon + (t_start * 0.000035)
        alt = base_alt + math.sin(t_start / 3.0) * 1.5
        rel_alt = 55.0 + math.sin(t_start / 3.0) * 1.5
        speed = 4.5

        # Format SRT timecode HH:MM:SS,mmm
        s_h = int(t_start // 3600)
        s_m = int((t_start % 3600) // 60)
        s_s = int(t_start % 60)
        s_ms = int(round((t_start % 1.0) * 1000))

        e_h = int(t_end // 3600)
        e_m = int((t_end % 3600) // 60)
        e_s = int(t_end % 60)
        e_ms = int(round((t_end % 1.0) * 1000))

        timecode = f"{s_h:02d}:{s_m:02d}:{s_s:02d},{s_ms:03d} --> {e_h:02d}:{e_m:02d}:{e_s:02d},{e_ms:03d}"
        payload = (
            f'<font size="36">FrameCnt: {i}, DiffTime: 33ms\n'
            f"[iso: 100] [shutter: 1/800] [fnum: 280] [ev: 0] [color_md: default]\n"
            f"[focal_len: 24.00] [dzoom_ratio: 10000, delta:0] [latitude: {lat:.6f}]\n"
            f"[longitude: {lon:.6f}] [rel_alt: {rel_alt:.2f} abs_alt: {alt:.2f}] [gb_yaw: 45.0 gb_pitch: -60.0 gb_roll: 0.0]\n"
            f"</font>"
        )
        srt_blocks.append(f"{i}\n{timecode}\n{payload}\n")
        csv_rows.append(f"{t_start:.3f},{lat:.6f},{lon:.6f},{alt:.2f},{rel_alt:.2f},0.0,-60.0,45.0,{speed:.1f}")

    SRT_PATH.write_text("\n".join(srt_blocks), encoding="utf-8")
    CSV_PATH.write_text("\n".join(csv_rows), encoding="utf-8")
    print(f"    Created: {SRT_PATH.name} ({len(srt_blocks)} timestamped frames)")
    print(f"    Created: {CSV_PATH.name} ({len(csv_rows) - 1} records)")

    # Write documentation in samples folder
    readme_content = """# DroneMap Test Samples

This directory contains local test samples for running, benchmarking, and demonstrating the **DroneMap v2.0** reconstruction pipeline.

## Files

- **`drone_flight_sample.mp4`**: A 20-second 1080p aerial drone survey video (H.264 / 30 fps) capturing an outdoor recreation facility with buildings, paved walkways, trees, and ground terrain.
- **`drone_flight_sample.srt`**: Companion DJI-standard subtitle flight telemetry embedding per-frame GNSS coordinates, relative/absolute altitudes, and gimbal angles.
- **`drone_flight_sample.csv`**: Tabular flight telemetry export with timestamps, coordinates, and drone orientation.

## Usage

### 1. Dry Run (Preview execution stages)
```powershell
.venv\\Scripts\\dronemap.exe run --video samples/drone_flight_sample.mp4 --dry-run
```

### 2. Full Reconstruction with GNSS Georeferencing
```powershell
.venv\\Scripts\\dronemap.exe run --video samples/drone_flight_sample.mp4 --telemetry samples/drone_flight_sample.srt
```

### 3. Reconstruction in Uncalibrated / Metric Relative Mode
```powershell
.venv\\Scripts\\dronemap.exe run --video samples/drone_flight_sample.mp4 --no-telemetry
```

### 4. Interactive Web Studio
Launch the viewer and upload `drone_flight_sample.mp4` directly via drag-and-drop:
```powershell
.venv\\Scripts\\dronemap.exe serve
```
"""
    README_PATH.write_text(readme_content, encoding="utf-8")
    print(f"    Created: {README_PATH.name}")
    print("[+] All sample assets downloaded and generated successfully in ./samples/")


if __name__ == "__main__":
    main()
