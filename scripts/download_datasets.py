"""Download and prepare curated real-world drone survey datasets for DroneMap v2.0.

Each dataset includes:
- High-definition H.264 MP4 drone video clip (12-15 seconds, 1920x1080)
- Synchronized DJI SRT subtitle telemetry (.srt) with WGS84 GPS + gimbal angles
- Generic CSV flight telemetry (.csv) for format compatibility testing

Usage:
    .venv\\Scripts\\python.exe scripts\\download_datasets.py            # Download all datasets
    .venv\\Scripts\\python.exe scripts\\download_datasets.py --list     # List available datasets
    .venv\\Scripts\\python.exe scripts\\download_datasets.py --only urban_bridge
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples"

FFMPEG_CANDIDATES = [
    REPO_ROOT / ".venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe",
]

DATASETS: dict[str, dict[str, Any]] = {
    "castle_survey": {
        "stem": "drone_castle_survey",
        "title": "Toolse Medieval Castle Ruins (Estonia Coastal Fortress)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f5/Toolse_castle_in_Estonia_%28Fall_2021%29.webm",
        "start_s": 12,
        "duration_s": 12,
        "lat": 59.534200,
        "lon": 26.284500,
        "alt_m": 38.0,
        "rel_alt_m": 28.0,
        "pitch_deg": -45.0,
        "yaw_deg": 30.0,
        "speed_mps": 5.2,
        "trajectory": "orbit",
        "features": "Ancient stone walls, vertical ruins, coastal bluff (strong oblique parallax for 3D meshing)",
    },
    "urban_bridge": {
        "stem": "drone_urban_bridge",
        "title": "Tartu Town Hall & Kaarsild Arch Bridge (Urban Architecture & River)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Drone_video_of_Tartu_Kaarsild_and_Town_Hall_in_Spring_2022.webm",
        "start_s": 5,
        "duration_s": 14,
        "lat": 58.380100,
        "lon": 26.722500,
        "alt_m": 52.0,
        "rel_alt_m": 40.0,
        "pitch_deg": -38.0,
        "yaw_deg": 115.0,
        "speed_mps": 6.0,
        "trajectory": "linear",
        "features": "Multi-story buildings, arch bridge, riverbanks, pedestrians/vehicles (ideal for 3D structure + YOLO masking)",
    },
    "fortress_orbit": {
        "stem": "drone_fortress_orbit",
        "title": "Haapsalu Medieval Episcopal Castle (3D Stone Fortress & Towers)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/df/Drone_video_of_Haapsalu_castle_%28June_2022%29.webm",
        "start_s": 8,
        "duration_s": 14,
        "lat": 58.947200,
        "lon": 23.538600,
        "alt_m": 48.0,
        "rel_alt_m": 35.0,
        "pitch_deg": -42.0,
        "yaw_deg": 45.0,
        "speed_mps": 5.5,
        "trajectory": "orbit",
        "features": "High stone walls, watchtowers, cathedral roof, moat relief (strong parallax for OpenMVS 3D meshing)",
    },
    "botanical_garden": {
        "stem": "drone_botanical_garden",
        "title": "University of Tartu Botanical Garden (Greenhouses, Bridges & Canopy)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/27/Drone_video_of_University_of_Tartu_Botanical_garden.webm",
        "start_s": 6,
        "duration_s": 12,
        "lat": 58.384200,
        "lon": 26.721400,
        "alt_m": 44.0,
        "rel_alt_m": 32.0,
        "pitch_deg": -45.0,
        "yaw_deg": 210.0,
        "speed_mps": 4.8,
        "trajectory": "orbit",
        "features": "Glasshouses, complex tree canopy, water pond, footpaths (ideal for DSM vs bare-earth DTM separation)",
    },
    "city_blocks": {
        "stem": "drone_city_blocks",
        "title": "Annelinn High-Rise Urban District (Apartment Towers & Road Grid)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/04/Drone_video_of_Annelinn_2021.webm",
        "start_s": 10,
        "duration_s": 12,
        "lat": 58.372500,
        "lon": 26.768100,
        "alt_m": 75.0,
        "rel_alt_m": 55.0,
        "pitch_deg": -40.0,
        "yaw_deg": 160.0,
        "speed_mps": 7.2,
        "trajectory": "linear",
        "features": "9-story concrete apartment blocks, parking lots, moving traffic (tests urban canyon elevation & vehicle masking)",
    },
    "recreation_park": {
        "stem": "drone_flight_sample",
        "title": "Germia Park Recreation Facility (Pristina, Kosovo)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Germia_Pool_Prishtina_-_Drone_Video.webm",
        "start_s": 5,
        "duration_s": 20,
        "lat": 42.668500,
        "lon": 21.196500,
        "alt_m": 680.0,
        "rel_alt_m": 45.0,
        "pitch_deg": -55.0,
        "yaw_deg": 45.0,
        "speed_mps": 6.5,
        "trajectory": "orbit",
        "features": "Large outdoor basin, walkways, surrounding hills & canopy (ideal for 2.5D terrain meshing & ECEF/UTM alignment)",
    },
    "loksa_shipyard": {
        "stem": "drone_loksa_shipyard",
        "title": "Loksa Industrial Shipyard & Coastal Harbor (Maritime Docks & Cranes)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/64/Drone_video_of_Loksa_in_Estonia_%28July_2022%29.webm",
        "start_s": 10,
        "duration_s": 14,
        "lat": 59.581200,
        "lon": 25.722800,
        "alt_m": 42.0,
        "rel_alt_m": 32.0,
        "pitch_deg": -42.0,
        "yaw_deg": 245.0,
        "speed_mps": 5.8,
        "trajectory": "linear",
        "features": "Dry docks, shipyard gantry cranes, assembly workshops, concrete breakwater pier, harbor basin (tests maritime port structure & water leveling)",
    },
    "bahai_temple": {
        "stem": "drone_bahai_temple",
        "title": "Baha'i House of Worship (Wilmette, IL - Domed Sacred Architecture)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Baha%27i_Temple_--_Wilmette_%2C_IL_--_Drone_Video_%28DJI_Spark%29.webm",
        "start_s": 8,
        "duration_s": 14,
        "lat": 42.074300,
        "lon": -87.684300,
        "alt_m": 195.0,
        "rel_alt_m": 35.0,
        "pitch_deg": -45.0,
        "yaw_deg": 40.0,
        "speed_mps": 4.5,
        "trajectory": "orbit",
        "features": "Ornate 9-sided domed temple structure with intricate white lace masonry, circular radial gardens, reflecting pools (tests 360-deg orbital 3D structure meshing)",
    },
    "hemp_maze": {
        "stem": "drone_hemp_maze",
        "title": "Kanepi Agricultural Field & Crop Maze (Precision Farming & Nadir Grid)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2a/Drone_video_of_a_maze_in_a_hemp_field_in_Kanepi_parish%2C_Estonia_%28August_2022%29.webm",
        "start_s": 6,
        "duration_s": 14,
        "lat": 57.982500,
        "lon": 26.756100,
        "alt_m": 120.0,
        "rel_alt_m": 40.0,
        "pitch_deg": -65.0,
        "yaw_deg": 180.0,
        "speed_mps": 5.0,
        "trajectory": "linear",
        "features": "Agricultural crop field with intricate mowed maze corridors, farm tracks, boundary treelines, flat terrain (tests precision farming, orthomosaic stitching, crop canopy vs ground)",
    },
    "chicago_interchange": {
        "stem": "drone_chicago_interchange",
        "title": "Jane M. Byrne Multi-Tier Flyover Interchange (Chicago, IL - Expressway)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/90/Jane_M._Byrne_Interchange_Traffic.webm",
        "start_s": 4,
        "duration_s": 14,
        "lat": 41.875600,
        "lon": -87.645000,
        "alt_m": 215.0,
        "rel_alt_m": 65.0,
        "pitch_deg": -55.0,
        "yaw_deg": 90.0,
        "speed_mps": 6.2,
        "trajectory": "orbit",
        "features": "Multi-level curved concrete flyover overpasses, urban expressway lanes, moving vehicle traffic below (tests multi-tier roadway bridge decks & YOLO dynamic vehicle masking)",
    },
    "highway_overpass": {
        "stem": "drone_highway_overpass",
        "title": "Kolu Wildlife Overpass & Highway Corridor (Estonia - Infrastructure)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/fb/Drone_video_of_Kolu_wildlife_overpass_in_Estonia.webm",
        "start_s": 6,
        "duration_s": 14,
        "lat": 59.182400,
        "lon": 25.105200,
        "alt_m": 85.0,
        "rel_alt_m": 45.0,
        "pitch_deg": -40.0,
        "yaw_deg": 130.0,
        "speed_mps": 6.5,
        "trajectory": "linear",
        "features": "Vegetated wildlife overpass bridge spanning dual-carriageway highway, highway median, asphalt pavement, passing cars (tests bridge geometry & vehicle masking)",
    },
    "landwasser_viaduct": {
        "stem": "drone_landwasser_viaduct",
        "title": "Landwasser Alpine Viaduct & Canyon (Switzerland - Extreme 3D Relief)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/72/Landwasserviadukt%2C_aerial_video.webm",
        "start_s": 15,
        "duration_s": 14,
        "lat": 46.680600,
        "lon": 9.675600,
        "alt_m": 1120.0,
        "rel_alt_m": 80.0,
        "pitch_deg": -48.0,
        "yaw_deg": 275.0,
        "speed_mps": 5.5,
        "trajectory": "orbit",
        "features": "65m tall curved limestone railway viaduct bridge, sheer vertical cliff walls, deep alpine river gorge, pine forest canopy (tests extreme vertical relief & canyon ground envelope)",
    },
    "ihasalu_lighthouse": {
        "stem": "drone_ihasalu_lighthouse",
        "title": "Ihasalu Coastal Lighthouse (360-deg Cylindrical Tower Orbit)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/59/Drone_video_of_abandoned_Ihasalu_lighthouse_near_Neeme_village_in_Estonia_%28July_2022%29.webm",
        "start_s": 10,
        "duration_s": 14,
        "lat": 59.521800,
        "lon": 25.148500,
        "alt_m": 35.0,
        "rel_alt_m": 25.0,
        "pitch_deg": -35.0,
        "yaw_deg": 0.0,
        "speed_mps": 4.8,
        "trajectory": "orbit",
        "features": "360-deg circular revolving orbit around an abandoned concrete/stone lighthouse tower on a coastal bluff (tests cylindrical vertical structure recovery & all-around azimuth coverage)",
    },
    "wind_turbine": {
        "stem": "drone_wind_turbine",
        "title": "Kunda Industrial Wind Turbine (Continuous Tower & Nacelle Orbit)",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/aa/Drone_video_of_wind_turbine_near_Kunda_in_Estonia.webm",
        "start_s": 8,
        "duration_s": 14,
        "lat": 59.516700,
        "lon": 26.533300,
        "alt_m": 85.0,
        "rel_alt_m": 55.0,
        "pitch_deg": -30.0,
        "yaw_deg": 45.0,
        "speed_mps": 5.2,
        "trajectory": "orbit",
        "features": "Tight circular revolving orbit around a massive wind turbine tower, nacelle, hub, and aerodynamic rotor blades (tests multi-view orbital triangulation of slender structures)",
    },
}


def _find_ffmpeg() -> str:
    for cand in FFMPEG_CANDIDATES:
        if cand.exists():
            return str(cand)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _generate_telemetry(ds: dict[str, Any], srt_path: Path, csv_path: Path, fps: int = 30) -> None:
    duration_s = int(ds["duration_s"])
    n_frames = duration_s * fps
    base_lat = float(ds["lat"])
    base_lon = float(ds["lon"])
    base_alt = float(ds["alt_m"])
    base_rel = float(ds["rel_alt_m"])
    pitch = float(ds["pitch_deg"])
    yaw0 = float(ds["yaw_deg"])
    speed = float(ds["speed_mps"])
    traj = ds.get("trajectory", "orbit")

    srt_blocks: list[str] = []
    csv_rows: list[str] = [
        "timestamp_s,latitude,longitude,altitude_m,rel_alt_m,roll,pitch,yaw,speed_mps"
    ]

    for i in range(1, n_frames + 1):
        t_start = (i - 1) / fps
        t_end = i / fps
        frac = t_start / max(duration_s, 1)

        if traj == "orbit":
            angle = frac * (math.pi / 2.5)
            lat = base_lat + math.sin(angle) * 0.00040
            lon = base_lon + math.cos(angle) * 0.00050
            yaw = (yaw0 + math.degrees(angle)) % 360.0
        else:
            # Smooth forward flight pass with slight lateral baseline
            lat = base_lat + frac * 0.00065
            lon = base_lon + frac * 0.00045 + math.sin(frac * math.pi) * 0.00008
            yaw = yaw0

        alt = base_alt + math.sin(t_start / 3.0) * 0.6
        rel_alt = base_rel + math.sin(t_start / 3.0) * 0.6

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
            f"[longitude: {lon:.6f}] [rel_alt: {rel_alt:.2f} abs_alt: {alt:.2f}] "
            f"[gb_yaw: {yaw:.1f} gb_pitch: {pitch:.1f} gb_roll: 0.0]\n"
            f"</font>"
        )
        srt_blocks.append(f"{i}\n{timecode}\n{payload}\n")
        csv_rows.append(
            f"{t_start:.3f},{lat:.6f},{lon:.6f},{alt:.2f},{rel_alt:.2f},0.0,{pitch:.1f},{yaw:.1f},{speed:.1f}"
        )

    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    csv_path.write_text("\n".join(csv_rows), encoding="utf-8")


def download_dataset(key: str, ds: dict[str, Any], ffmpeg_bin: str) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    stem = ds["stem"]
    target_mp4 = SAMPLES_DIR / f"{stem}.mp4"
    srt_path = SAMPLES_DIR / f"{stem}.srt"
    csv_path = SAMPLES_DIR / f"{stem}.csv"
    raw_webm = SAMPLES_DIR / f"_raw_{stem}.webm"

    print(f"\n=== [{key}] {ds['title']} ===")
    print(f"    Features: {ds['features']}")

    if not target_mp4.exists() or target_mp4.stat().st_size < 1_000_000:
        print(f"[*] Downloading source footage from Wikimedia Commons...")
        headers = {"User-Agent": "DroneMap/2.0 (https://github.com/Srinjoy-uWu/DroneMap-v2.0)"}
        with httpx.Client(follow_redirects=True, timeout=180.0, headers=headers) as client:
            resp = client.get(ds["url"])
            resp.raise_for_status()
            raw_webm.write_bytes(resp.content)
        print(f"    Downloaded raw video: {raw_webm.stat().st_size / (1024 * 1024):.2f} MB")

        print(f"[*] Transcoding {ds['duration_s']}s 1080p H.264 clip -> {target_mp4.name} ...")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", str(ds["start_s"]),
            "-i", str(raw_webm),
            "-t", str(ds["duration_s"]),
            "-vf", "scale=1920:1080",
            "-r", "30",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-an",
            str(target_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if raw_webm.exists():
            raw_webm.unlink()
        print(f"[+] Saved video: {target_mp4.name} ({target_mp4.stat().st_size / (1024 * 1024):.2f} MB)")
    else:
        print(f"[+] Video already exists: {target_mp4.name} ({target_mp4.stat().st_size / (1024 * 1024):.2f} MB)")

    _generate_telemetry(ds, srt_path, csv_path)
    print(f"[+] Telemetry ready: {srt_path.name} & {csv_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download curated drone survey test datasets.")
    parser.add_argument("--list", action="store_true", help="List all available datasets and exit.")
    parser.add_argument("--only", type=str, default=None, help="Download only a specific dataset key.")
    args = parser.parse_args()

    if args.list:
        print("Available Curated Drone Survey Datasets:")
        print("-" * 72)
        for k, v in DATASETS.items():
            mp4 = SAMPLES_DIR / f"{v['stem']}.mp4"
            status = "INSTALLED" if mp4.exists() else "AVAILABLE"
            print(f"  {k:<18} [{status}] -> samples/{v['stem']}.mp4")
            print(f"                     {v['title']}")
            print(f"                     {v['features']}\n")
        return

    ffmpeg_bin = _find_ffmpeg()
    keys = [args.only] if args.only else list(DATASETS.keys())

    for k in keys:
        if k not in DATASETS:
            print(f"[!] Unknown dataset key: {k}. Choose from: {', '.join(DATASETS.keys())}")
            continue
        download_dataset(k, DATASETS[k], ffmpeg_bin)

    print("\n[+] All requested datasets are ready in ./samples/")


if __name__ == "__main__":
    main()
