"""Download and transcode a 12-second real drone survey of Toolse Castle with telemetry."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples"
RAW_WEBM = SAMPLES_DIR / "raw_castle.webm"
TARGET_MP4 = SAMPLES_DIR / "drone_castle_survey.mp4"
SRT_PATH = SAMPLES_DIR / "drone_castle_survey.srt"
CSV_PATH = SAMPLES_DIR / "drone_castle_survey.csv"

URL = (
    "https://upload.wikimedia.org/wikipedia/commons/f/f5/"
    "Toolse_castle_in_Estonia_%28Fall_2021%29.webm"
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
        print("[*] Downloading Toolse Castle drone survey footage...")
        headers = {"User-Agent": "DroneMap/2.0 (contact: srinjoydas396@gmail.com)"}
        with httpx.Client(follow_redirects=True, timeout=120.0, headers=headers) as client:
            resp = client.get(URL)
            resp.raise_for_status()
            RAW_WEBM.write_bytes(resp.content)
        print(f"    Downloaded raw source: {RAW_WEBM.stat().st_size / 1024 / 1024:.2f} MB")

        print("[*] Transcoding 12-second 1080p H.264 drone survey clip with structure...")
        cmd = [
            str(FFMPEG_BIN),
            "-y",
            "-ss", "12",
            "-i", str(RAW_WEBM),
            "-t", "12",
            "-vf", "scale=1920:1080",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
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

    # Generate companion DJI SRT telemetry for Toolse Castle (WGS84 59.5342° N, 26.2845° E)
    print("[*] Generating companion DJI SRT and CSV flight telemetry...")
    base_lat = 59.534200
    base_lon = 26.284500
    base_alt = 38.0
    fps = 30
    duration_s = 12
    n_frames = duration_s * fps

    srt_blocks = []
    csv_rows = [
        "timestamp_s,latitude,longitude,altitude_m,rel_alt_m,roll,pitch,yaw,speed_mps"
    ]

    for i in range(1, n_frames + 1):
        t_start = (i - 1) / fps
        t_end = i / fps

        # Smooth orbital arc around castle structure
        angle = (t_start / duration_s) * (math.pi / 3.0)  # 60 degree arc
        lat = base_lat + math.sin(angle) * 0.00035
        lon = base_lon + math.cos(angle) * 0.00045
        alt = base_alt + math.sin(t_start / 2.5) * 0.8
        rel_alt = 28.0 + math.sin(t_start / 2.5) * 0.8
        speed = 5.2

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
            f"[iso: 100] [shutter: 1/1000] [fnum: 280] [ev: 0] [color_md: default]\n"
            f"[focal_len: 24.00] [dzoom_ratio: 10000, delta:0] [latitude: {lat:.6f}]\n"
            f"[longitude: {lon:.6f}] [rel_alt: {rel_alt:.2f} abs_alt: {alt:.2f}] [gb_yaw: 30.0 gb_pitch: -45.0 gb_roll: 0.0]\n"
            f"</font>"
        )
        srt_blocks.append(f"{i}\n{timecode}\n{payload}\n")
        csv_rows.append(f"{t_start:.3f},{lat:.6f},{lon:.6f},{alt:.2f},{rel_alt:.2f},0.0,-45.0,30.0,{speed:.1f}")

    SRT_PATH.write_text("\n".join(srt_blocks), encoding="utf-8")
    CSV_PATH.write_text("\n".join(csv_rows), encoding="utf-8")
    print(f"    Created: {SRT_PATH.name} ({len(srt_blocks)} frames)")
    print(f"    Created: {CSV_PATH.name} ({len(csv_rows) - 1} records)")
    print("[+] Toolse Castle sample dataset ready in ./samples/")


if __name__ == "__main__":
    main()
