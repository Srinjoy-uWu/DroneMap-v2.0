"""Tests for telemetry parsers."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dronemap.telemetry.dji_srt import parse as parse_srt
from dronemap.telemetry.generic_csv import parse as parse_csv
from dronemap.telemetry import load


# ---------------------------------------------------------------------------
# DJI SRT — modern bracket format
# ---------------------------------------------------------------------------

MODERN_SRT = textwrap.dedent("""\
    1
    00:00:00,033 --> 00:00:00,066
    <font size="36">FrameCnt: 1, DiffTime: 33ms
    2026-08-31 16:22:01.033
    [iso: 100] [shutter: 1/1000] [fnum: 280] [ev: 0]
    [latitude: 12.345678] [longitude: 77.654321]
    [rel_alt: 55.200 abs_alt: 283.900]
    [gb_yaw: -0.2 gb_pitch: -90.0 gb_roll: 0.0]
    </font>

    2
    00:00:00,066 --> 00:00:00,099
    <font size="36">FrameCnt: 2, DiffTime: 33ms
    2026-08-31 16:22:01.066
    [latitude: 12.345700] [longitude: 77.654340]
    [rel_alt: 55.300 abs_alt: 283.950]
    [gb_pitch: -89.5]
    </font>
""")

OLD_SRT = textwrap.dedent("""\
    1
    00:00:00,033 --> 00:00:00,066
    GPS(12.345678,77.654321,283.9) BAROMETER(283.9)
    HOME(12.340000,77.650000) D:12.5m H:55.2m

    2
    00:00:00,066 --> 00:00:00,099
    GPS(12.345700,77.654340,283.95) BAROMETER(283.95)
    HOME(12.340000,77.650000)
""")


class TestDjiSrt:
    def test_modern_format_basic(self, tmp_path: Path):
        srt = tmp_path / "test.srt"
        srt.write_text(MODERN_SRT, encoding="utf-8")
        fixes = parse_srt(srt)
        assert len(fixes) == 2

    def test_modern_lat_lon(self, tmp_path: Path):
        srt = tmp_path / "test.srt"
        srt.write_text(MODERN_SRT, encoding="utf-8")
        fixes = parse_srt(srt)
        assert fixes[0]["lat"] == pytest.approx(12.345678)
        assert fixes[0]["lon"] == pytest.approx(77.654321)

    def test_modern_abs_alt(self, tmp_path: Path):
        srt = tmp_path / "test.srt"
        srt.write_text(MODERN_SRT, encoding="utf-8")
        fixes = parse_srt(srt)
        assert fixes[0]["alt_m"] == pytest.approx(283.9)

    def test_modern_gimbal_pitch(self, tmp_path: Path):
        srt = tmp_path / "test.srt"
        srt.write_text(MODERN_SRT, encoding="utf-8")
        fixes = parse_srt(srt)
        assert fixes[0]["gimbal_pitch"] == pytest.approx(-90.0)

    def test_modern_frame_idx(self, tmp_path: Path):
        srt = tmp_path / "test.srt"
        srt.write_text(MODERN_SRT, encoding="utf-8")
        fixes = parse_srt(srt)
        assert fixes[0]["frame_idx"] == 0  # FrameCnt=1 → 0-based
        assert fixes[1]["frame_idx"] == 1

    def test_modern_timestamp(self, tmp_path: Path):
        srt = tmp_path / "test.srt"
        srt.write_text(MODERN_SRT, encoding="utf-8")
        fixes = parse_srt(srt)
        assert fixes[0]["timestamp_s"] == pytest.approx(0.033)
        assert fixes[1]["timestamp_s"] == pytest.approx(0.066)

    def test_old_format(self, tmp_path: Path):
        srt = tmp_path / "test.srt"
        srt.write_text(OLD_SRT, encoding="utf-8")
        fixes = parse_srt(srt)
        assert len(fixes) == 2
        assert fixes[0]["lat"] == pytest.approx(12.345678)
        assert fixes[0]["lon"] == pytest.approx(77.654321)
        assert fixes[0]["alt_m"] == pytest.approx(283.9)

    def test_empty_file_raises(self, tmp_path: Path):
        srt = tmp_path / "empty.srt"
        srt.write_text("", encoding="utf-8")
        with pytest.raises(RuntimeError, match="could not extract any GPS fixes"):
            parse_srt(srt)

    def test_bom_handled(self, tmp_path: Path):
        srt = tmp_path / "bom.srt"
        srt.write_bytes(b"\xef\xbb\xbf" + MODERN_SRT.encode("utf-8"))
        fixes = parse_srt(srt)
        assert len(fixes) == 2


# ---------------------------------------------------------------------------
# Generic CSV
# ---------------------------------------------------------------------------

COMMA_CSV = textwrap.dedent("""\
    time,latitude,longitude,altitude,speed
    0.0,12.345678,77.654321,283.9,5.5
    0.1,12.345700,77.654340,283.95,5.6
    0.2,12.345720,77.654360,284.0,5.7
""")

SEMI_CSV = textwrap.dedent("""\
    time;lat;lon;alt;speed
    0.0;12.345678;77.654321;283.9;5.5
    0.1;12.345700;77.654340;283.95;5.6
""")

DJI_FLY_CSV = textwrap.dedent("""\
    OSD.flyTime(s),OSD.lat,OSD.lon,OSD.height [D],OSD.hSpeed(m/s)
    0.0,12.345678,77.654321,55.2,5.5
    0.1,12.345700,77.654340,55.3,5.6
""")


class TestGenericCsv:
    def test_comma_csv(self, tmp_path: Path):
        csv = tmp_path / "flight.csv"
        csv.write_text(COMMA_CSV, encoding="utf-8")
        fixes = parse_csv(csv)
        assert len(fixes) == 3
        assert fixes[0]["lat"] == pytest.approx(12.345678)
        assert fixes[0]["lon"] == pytest.approx(77.654321)

    def test_semicolon_csv(self, tmp_path: Path):
        csv = tmp_path / "flight.csv"
        csv.write_text(SEMI_CSV, encoding="utf-8")
        fixes = parse_csv(csv)
        assert len(fixes) == 2
        assert fixes[0]["lat"] == pytest.approx(12.345678)

    def test_speed_parsed(self, tmp_path: Path):
        csv = tmp_path / "flight.csv"
        csv.write_text(COMMA_CSV, encoding="utf-8")
        fixes = parse_csv(csv)
        assert fixes[0]["speed_mps"] == pytest.approx(5.5)

    def test_altitude_parsed(self, tmp_path: Path):
        csv = tmp_path / "flight.csv"
        csv.write_text(COMMA_CSV, encoding="utf-8")
        fixes = parse_csv(csv)
        assert fixes[0]["alt_m"] == pytest.approx(283.9)

    def test_dji_fly_column_names(self, tmp_path: Path):
        csv = tmp_path / "flight.csv"
        csv.write_text(DJI_FLY_CSV, encoding="utf-8")
        fixes = parse_csv(csv)
        assert len(fixes) == 2
        assert fixes[0]["lat"] == pytest.approx(12.345678)

    def test_no_lat_col_raises(self, tmp_path: Path):
        csv = tmp_path / "bad.csv"
        csv.write_text("time,x,y,z\n0,1,2,3\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="latitude"):
            parse_csv(csv)

    def test_sorted_by_timestamp(self, tmp_path: Path):
        unsorted = textwrap.dedent("""\
            time,latitude,longitude
            0.5,12.1,77.1
            0.0,12.0,77.0
            1.0,12.2,77.2
        """)
        csv = tmp_path / "unsorted.csv"
        csv.write_text(unsorted, encoding="utf-8")
        fixes = parse_csv(csv)
        times = [f["timestamp_s"] for f in fixes]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# Auto-detect (load())
# ---------------------------------------------------------------------------

class TestAutoDetect:
    def test_srt_detected(self, tmp_path: Path):
        p = tmp_path / "test.srt"
        p.write_text(MODERN_SRT, encoding="utf-8")
        fixes = load(p)
        assert len(fixes) == 2

    def test_csv_detected(self, tmp_path: Path):
        p = tmp_path / "test.csv"
        p.write_text(COMMA_CSV, encoding="utf-8")
        fixes = load(p)
        assert len(fixes) == 3

    def test_tlog_raises(self, tmp_path: Path):
        p = tmp_path / "test.tlog"
        p.write_bytes(b"\x00\x01\x02\x03")
        with pytest.raises(ValueError, match="MAVLink"):
            load(p)

    def test_unknown_ext_raises(self, tmp_path: Path):
        p = tmp_path / "test.xyz"
        p.write_text("data", encoding="utf-8")
        with pytest.raises(ValueError, match="unrecognised"):
            load(p)
