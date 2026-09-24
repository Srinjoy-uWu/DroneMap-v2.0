"""Telemetry sidecar parsers.

Supported formats:
  .srt   DJI subtitle sidecar (Phantom, Mavic, Mini — GPS embedded in captions)
  .csv   Generic flight-log CSV (DJI Fly App, Mission Planner, ArduPilot, etc.)
  .tlog  MAVLink binary telemetry — not parsed; a clear error is raised.

All parsers return a list of dicts with *at least* these keys (may be None):

    frame_idx   : int | None   — zero-based video frame index (SRT only)
    timestamp_s : float        — seconds from start of file
    lat         : float | None — WGS-84 latitude, decimal degrees
    lon         : float | None — WGS-84 longitude, decimal degrees
    alt_m       : float | None — altitude MSL, metres
    speed_mps   : float | None — ground speed m/s
    gimbal_pitch: float | None — gimbal pitch, degrees (negative = down)

Call ``load(path)`` and it will auto-detect the format.
"""

from __future__ import annotations

from pathlib import Path


def load(path: str | Path) -> list[dict]:
    """Auto-detect format and parse a telemetry sidecar.

    Returns a list of fix dicts (see module docstring).  Raises ``ValueError``
    for unsupported formats and ``RuntimeError`` if the file is empty or
    completely unparseable.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".srt":
        from .dji_srt import parse
        return parse(path)
    elif suffix == ".csv":
        from .generic_csv import parse
        return parse(path)
    elif suffix == ".tlog":
        raise ValueError(
            f"{path.name}: MAVLink .tlog binary format is not supported. "
            "Export it to CSV from Mission Planner (Telemetry Logs → Export to CSV) "
            "and pass the .csv file instead."
        )
    else:
        raise ValueError(
            f"{path.name}: unrecognised telemetry format {suffix!r}. "
            "Supported: .srt (DJI subtitle), .csv (generic flight log)."
        )


__all__ = ["load"]
