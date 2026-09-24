"""DJI SRT subtitle sidecar parser.

DJI drones (Phantom 4, Mavic series, Mini series, Air series) write a ``.srt``
file alongside the video that embeds per-frame GPS, altitude, gimbal angles, and
camera metadata as subtitle text.  The exact format varies slightly across
firmware generations, but the key fields are always present:

Example block (firmware ≥ 1.00.0100)::

    1
    00:00:00,033 --> 00:00:00,066
    <font size="36">FrameCnt: 1, DiffTime: 33ms
    2026-08-31 16:22:01.033
    [iso: 100] [shutter: 1/1000] [fnum: 280] [ev: 0] [color_md : default]
    [focal_len: 24.00] [dzoom_ratio: 10000, delta:0] [latitude: 12.345678]
    [longitude: 77.654321] [rel_alt: 55.200 abs_alt: 283.900] [gb_yaw: -0.2
     gb_pitch: -90.0 gb_roll: 0.0]
    </font>

Older format (no XML tags, plain text)::

    1
    00:00:00,033 --> 00:00:00,066
    GPS(12.345678,77.654321,283.9) BAROMETER(283.9)
    HOME(12.340000,77.650000) D:12.5m H:55.2m
    ...

The parser is deliberately lenient — it uses regex to pick out the fields it
needs from whatever text is present, so minor format variations don't break it.
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns — compiled once
# ---------------------------------------------------------------------------

_RE_TIMECODE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
    r"\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

# Modern format: [latitude: 12.345678]
_RE_LAT_BRACKET = re.compile(r"\[latitude\s*:\s*([+-]?\d+\.?\d*)\]", re.IGNORECASE)
_RE_LON_BRACKET = re.compile(r"\[longitude\s*:\s*([+-]?\d+\.?\d*)\]", re.IGNORECASE)
_RE_ALT_ABS    = re.compile(r"abs_alt\s*:\s*([+-]?\d+\.?\d*)", re.IGNORECASE)
_RE_ALT_REL    = re.compile(r"rel_alt\s*:\s*([+-]?\d+\.?\d*)", re.IGNORECASE)

# Old format: GPS(lat,lon,alt)
_RE_GPS_OLD    = re.compile(r"GPS\(([+-]?\d+\.?\d*),([+-]?\d+\.?\d*),([+-]?\d+\.?\d*)\)")

# Barometer altitude fallback: BAROMETER(283.9)
_RE_BARO       = re.compile(r"BAROMETER\(([+-]?\d+\.?\d*)\)")

# Gimbal
_RE_GB_PITCH   = re.compile(r"gb_pitch\s*:\s*([+-]?\d+\.?\d*)", re.IGNORECASE)

# FrameCnt
_RE_FRAMECNT   = re.compile(r"FrameCnt\s*:\s*(\d+)", re.IGNORECASE)

# DiffTime (ms per frame → speed inference not done here, speed kept None)
_RE_DIFFTIME   = re.compile(r"DiffTime\s*:\s*(\d+)\s*ms", re.IGNORECASE)


def _tc_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    ms_val = int(ms.ljust(3, "0"))  # normalise 1- or 2-digit ms
    return int(h) * 3600 + int(m) * 60 + int(s) + ms_val / 1000.0


def _parse_block(block_lines: list[str]) -> dict | None:
    """Parse one SRT block (sequence number already stripped).

    Returns a fix dict or None if no useful data could be extracted.
    """
    # First line of data after the timecode is the timecode line itself
    timecode_line = None
    text_lines = []
    for line in block_lines:
        if _RE_TIMECODE.search(line):
            timecode_line = line
        else:
            text_lines.append(line)

    if timecode_line is None:
        return None

    m = _RE_TIMECODE.search(timecode_line)
    assert m  # guaranteed by the search above
    timestamp_s = _tc_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))

    text = " ".join(text_lines)

    # --- Frame count ---
    frame_idx: int | None = None
    mc = _RE_FRAMECNT.search(text)
    if mc:
        frame_idx = int(mc.group(1)) - 1  # 0-based

    # --- GPS: modern bracket format ---
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None

    ml = _RE_LAT_BRACKET.search(text)
    if ml:
        lat = float(ml.group(1))
    mo = _RE_LON_BRACKET.search(text)
    if mo:
        lon = float(mo.group(1))

    # Altitude: prefer abs_alt, fall back to rel_alt
    ma = _RE_ALT_ABS.search(text)
    if ma:
        alt_m = float(ma.group(1))
    else:
        mr = _RE_ALT_REL.search(text)
        if mr:
            alt_m = float(mr.group(1))

    # --- GPS: old format ---
    if lat is None:
        mg = _RE_GPS_OLD.search(text)
        if mg:
            lat = float(mg.group(1))
            lon = float(mg.group(2))
            if alt_m is None:
                alt_m = float(mg.group(3))

    # Barometer fallback
    if alt_m is None:
        mb = _RE_BARO.search(text)
        if mb:
            alt_m = float(mb.group(1))

    # Gimbal pitch
    gimbal_pitch: float | None = None
    mgp = _RE_GB_PITCH.search(text)
    if mgp:
        gimbal_pitch = float(mgp.group(1))

    return {
        "frame_idx": frame_idx,
        "timestamp_s": timestamp_s,
        "lat": lat,
        "lon": lon,
        "alt_m": alt_m,
        "speed_mps": None,  # not present in SRT; derived from GPS deltas if needed
        "gimbal_pitch": gimbal_pitch,
    }


def parse(path: Path) -> list[dict]:
    """Parse a DJI .srt file and return a list of fix dicts."""
    text = path.read_text(encoding="utf-8", errors="replace")

    # Strip BOM and normalise line endings
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")

    # Split into blocks separated by blank lines.  Each block is:
    #   <sequence number>
    #   <timecode line>
    #   <text lines...>
    raw_blocks = re.split(r"\n\s*\n", text.strip())

    fixes: list[dict] = []
    for raw in raw_blocks:
        lines = [l for l in raw.splitlines() if l.strip()]
        if not lines:
            continue
        # Drop the sequence-number line (first line that is purely an integer)
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        fix = _parse_block(lines)
        if fix is not None:
            fixes.append(fix)

    if not fixes:
        raise RuntimeError(
            f"{path.name}: could not extract any GPS fixes. "
            "The file may be an unsupported SRT variant — check the raw content."
        )

    return fixes
