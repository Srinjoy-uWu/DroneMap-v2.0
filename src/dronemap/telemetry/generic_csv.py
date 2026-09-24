"""Generic CSV flight-log parser.

Handles CSV exports from:
  - DJI Fly App (flight records)
  - Mission Planner ("Export to CSV" from Telemetry Logs)
  - ArduPilot / APM Planner log exports
  - Any tabular GPS log with recognisable column names

Column name sniffing is case-insensitive and handles underscores, spaces, and
dots as separators.  The parser looks for:

  Latitude:  latitude, lat, GPS.Lat, Lat(WGS84)
  Longitude: longitude, lon, lng, GPS.Lng, Lon(WGS84)
  Altitude:  altitude, alt, GPS.Alt, Alt(m), RelAlt, abs_alt
  Time:      time, timestamp, OSD.flyTime(s), GPS.Date+GPS.Time, offset_time

If no time column is found, synthetic timestamps are assigned (row index × 0.1 s).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Column name aliases
# ---------------------------------------------------------------------------

_LAT_ALIASES = {
    "latitude", "lat", "gps.lat", "lat(wgs84)", "latitude(deg)",
    "osd.lat", "osd.latitude",
}
_LON_ALIASES = {
    "longitude", "lon", "lng", "gps.lng", "gps.lon",
    "lon(wgs84)", "longitude(deg)",
    "osd.lon", "osd.lng", "osd.longitude",
}
_ALT_ALIASES = {
    "altitude", "alt", "gps.alt", "altitude(m)", "alt(m)", "abs_alt",
    "altitudemsl", "altitude_m", "gps.alt(m)",
    "osd.altitude", "osd.alt(m)",
}
_RELALT_ALIASES = {
    "rel_alt", "relativealtitude", "relative_alt", "relalt", "height",
    "heightabovehomepoint(m)",
    "osd.height[d]", "osd.height",
}
_TIME_ALIASES = {
    "time", "timestamp", "time(s)", "osd.flytime(s)", "offset_time",
    "time_boot_ms", "timestamp(ms)", "elapsed",
    # An unrecognised time column is worse than no time column: the synthetic
    # 10 Hz fallback below produces a plausible-looking but wrong timebase that
    # runs short of the real flight, and every fix past its end is then dropped
    # by frame alignment without anything looking broken. Recognise the ordinary
    # spellings.
    "timestamp_s", "time_s", "t", "t_s", "seconds", "sec", "secs",
    "elapsed_s", "elapsed_time", "elapsed(s)", "flight_time", "flighttime",
    "flight_time_s", "frame_time", "frametime",
}
_SPEED_ALIASES = {
    "speed", "groundspeed", "ground_speed", "speed(m/s)", "gps.speed",
    "osd.hspeed(m/s)", "horizontalspeed",
}
_PITCH_ALIASES = {
    "gimbal_pitch", "gimbal.pitch", "osd.gimbal_pitch",
    "gimbal_pitch(°)", "gimbalpitch",
}


def _normalise(name: str) -> str:
    """Lowercase + strip punctuation for alias matching."""
    return re.sub(r"[\s._\-/()°]+", "", name.lower())


def _find_col(header: list[str], aliases: set[str]) -> str | None:
    normalised_aliases = {_normalise(a) for a in aliases}
    for col in header:
        if _normalise(col) in normalised_aliases:
            return col
    return None


def _safe_float(val: str) -> float | None:
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return None


def parse(path: Path) -> list[dict]:
    """Parse a CSV flight log and return a list of fix dicts."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")  # -sig strips BOM

    # Try comma, then semicolon, then tab
    dialect: csv.Dialect | str = "excel"
    sample = text[:4096]
    if sample.count(";") > sample.count(","):
        dialect = "excel-semicolon"
        csv.register_dialect("excel-semicolon", delimiter=";", quotechar='"')
    elif "\t" in sample and sample.count("\t") > sample.count(","):
        dialect = "excel-tab"

    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    if reader.fieldnames is None:
        raise RuntimeError(f"{path.name}: could not read CSV header")

    header = list(reader.fieldnames)

    lat_col  = _find_col(header, _LAT_ALIASES)
    lon_col  = _find_col(header, _LON_ALIASES)
    alt_col  = _find_col(header, _ALT_ALIASES)
    ralt_col = _find_col(header, _RELALT_ALIASES)
    time_col = _find_col(header, _TIME_ALIASES)
    spd_col  = _find_col(header, _SPEED_ALIASES)
    pit_col  = _find_col(header, _PITCH_ALIASES)

    if lat_col is None or lon_col is None:
        raise RuntimeError(
            f"{path.name}: could not find latitude / longitude columns in header:\n"
            f"  {header}\n"
            "Expected one of: " + ", ".join(sorted(_LAT_ALIASES)) + " (case-insensitive)."
        )

    fixes: list[dict] = []
    for row_idx, row in enumerate(reader):
        lat = _safe_float(row.get(lat_col, ""))
        lon = _safe_float(row.get(lon_col, ""))
        if lat is None or lon is None:
            continue

        # Handle MAVLink / ArduPilot integer-scaled lat/lon (1e7)
        if abs(lat) > 90.0 and abs(lat) <= 90e7:
            lat = lat / 1e7
        if abs(lon) > 180.0 and abs(lon) <= 180e7:
            lon = lon / 1e7

        # Strict range validation
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue

        # Altitude. Two different consumers need two different numbers, and
        # collapsing them into one field is how a 200-300 m MSL datum ends up
        # being treated as flying height:
        #   alt_m  - the georeferencing datum, absolute where the log has one.
        #   agl_m  - height above the launch point, used for ground-footprint
        #            and GSD arithmetic. Only set when the log really reports a
        #            relative altitude; never inferred from an absolute one.
        alt_m: float | None = None
        agl_m: float | None = None
        alt_source: str = "none"
        if alt_col:
            alt_m = _safe_float(row.get(alt_col, ""))
            if alt_m is not None:
                alt_source = "absolute"
        if ralt_col:
            agl_m = _safe_float(row.get(ralt_col, ""))
        if alt_m is None and agl_m is not None:
            alt_m = agl_m
            alt_source = "relative_home"

        # Timestamp: distinguish ms epoch, sec epoch, and relative time
        timestamp_s: float = float(row_idx) * 0.1  # synthetic fallback
        time_source = "synthetic_10hz"
        if time_col:
            raw_t = row.get(time_col, "").strip()
            if raw_t.isdigit():
                val = int(raw_t)
                time_source = "column"
                if val > 1_000_000_000_000:  # ms epoch (> year 2001 in ms)
                    timestamp_s = val / 1000.0
                elif val > 1_000_000_000:    # sec epoch (> year 2001 in s)
                    timestamp_s = float(val)
                elif val > 1_000_000:        # sub-epoch ms / microsec
                    timestamp_s = val / 1000.0
                else:
                    timestamp_s = float(val)
            else:
                ts = _safe_float(raw_t)
                if ts is not None:
                    timestamp_s = ts
                    time_source = "column"

        speed_mps   = _safe_float(row.get(spd_col, "")) if spd_col else None
        gimbal_pitch = _safe_float(row.get(pit_col, "")) if pit_col else None

        fixes.append({
            "frame_idx": None,
            "timestamp_s": timestamp_s,
            "time_source": time_source,
            "lat": lat,
            "lon": lon,
            "alt_m": alt_m,
            "agl_m": agl_m,
            "alt_source": alt_source,
            "speed_mps": speed_mps,
            "gimbal_pitch": gimbal_pitch,
        })

    if not fixes:
        raise RuntimeError(
            f"{path.name}: parsed 0 valid rows. "
            "Check that latitude and longitude columns contain numeric values."
        )

    # Sort by timestamp (some exports are not chronological)
    fixes.sort(key=lambda f: f["timestamp_s"])
    return fixes
