"""Tests for the run workspace and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dronemap.workspace import (
    GnssMode,
    RunWorkspace,
    StageStatus,
    StageRecord,
    stage_guard,
)


@pytest.fixture()
def ws(tmp_path: Path) -> RunWorkspace:
    return RunWorkspace.create(tmp_path, "test-run-01", config={"frames": {}})


class TestRunWorkspaceCreate:
    def test_creates_manifest(self, ws: RunWorkspace):
        assert ws.manifest_path.exists()

    def test_manifest_has_run_id(self, ws: RunWorkspace):
        assert ws.manifest["run_id"] == "test-run-01"

    def test_stage_dirs_created(self, ws: RunWorkspace, tmp_path: Path):
        for stage in ["frames", "masks", "pose", "georef", "semantics", "dense", "mesh", "export"]:
            assert ws.stage_dir(stage).exists()

    def test_logs_dir_created(self, ws: RunWorkspace):
        assert ws.logs_dir.exists()

    def test_all_stages_pending(self, ws: RunWorkspace):
        for stage in ["frames", "masks", "pose"]:
            assert ws.stage(stage).status is StageStatus.PENDING


class TestRunWorkspaceOpen:
    def test_open_existing(self, ws: RunWorkspace, tmp_path: Path):
        ws2 = RunWorkspace.open(tmp_path, "test-run-01")
        assert ws2.run_id == "test-run-01"

    def test_open_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no run"):
            RunWorkspace.open(tmp_path, "nonexistent-run")

    def test_list_runs(self, ws: RunWorkspace, tmp_path: Path):
        RunWorkspace.create(tmp_path, "run-b", config={})
        ids = RunWorkspace.list_runs(tmp_path)
        assert "test-run-01" in ids
        assert "run-b" in ids
        assert ids == sorted(ids)


class TestManifestOperations:
    def test_set_input(self, ws: RunWorkspace):
        ws.set_input(video="/path/to/video.mp4")
        ws2 = RunWorkspace.open(ws.root.parent.parent, ws.run_id)
        assert ws2.manifest["input"]["video"] == "/path/to/video.mp4"

    def test_set_gnss(self, ws: RunWorkspace):
        ws.set_gnss(GnssMode.RTK, source="test.srt", note="test")
        assert ws.gnss_mode is GnssMode.RTK

    def test_gnss_mode_defaults_to_none(self, ws: RunWorkspace):
        assert ws.gnss_mode is GnssMode.NONE

    def test_set_accuracy(self, ws: RunWorkspace):
        ws.set_accuracy(alignment_rmse_m=1.23)
        ws.load()
        assert ws.manifest["accuracy"]["alignment_rmse_m"] == pytest.approx(1.23)

    def test_atomic_write(self, ws: RunWorkspace):
        """Manifest .tmp file should not persist after a write."""
        ws.set_input(video="x.mp4")
        tmp = ws.manifest_path.with_suffix(".json.tmp")
        assert not tmp.exists()


class TestStageGuard:
    def test_runs_pending_stage(self, ws: RunWorkspace):
        with stage_guard(ws, "frames") as ctx:
            assert ctx is not None
            ctx.metric(n_keyframes=42)
        assert ws.stage("frames").status is StageStatus.OK
        assert ws.stage("frames").metrics["n_keyframes"] == 42

    def test_skips_already_ok(self, ws: RunWorkspace):
        with stage_guard(ws, "frames"):
            pass  # marks OK
        with stage_guard(ws, "frames") as ctx:
            assert ctx is None  # skipped

    def test_force_reruns_ok_stage(self, ws: RunWorkspace):
        with stage_guard(ws, "frames"):
            pass
        with stage_guard(ws, "frames", force=True) as ctx:
            assert ctx is not None

    def test_records_failure(self, ws: RunWorkspace):
        with pytest.raises(RuntimeError, match="boom"):
            with stage_guard(ws, "frames"):
                raise RuntimeError("boom")
        record = ws.stage("frames")
        assert record.status is StageStatus.FAILED
        assert "boom" in (record.error or "")

    def test_duration_recorded(self, ws: RunWorkspace):
        with stage_guard(ws, "frames"):
            pass
        assert ws.stage("frames").duration_s is not None
        assert ws.stage("frames").duration_s >= 0

    def test_note_appended(self, ws: RunWorkspace):
        with stage_guard(ws, "frames") as ctx:
            ctx.note("test note")
        assert "test note" in ws.stage("frames").notes


class TestGnssMode:
    def test_rtk_is_metric_absolute(self):
        assert GnssMode.RTK.is_metric_absolute
        assert GnssMode.PPK.is_metric_absolute
        assert GnssMode.STANDALONE.is_metric_absolute

    def test_none_is_not_metric(self):
        assert not GnssMode.NONE.is_metric_absolute
        assert not GnssMode.SIMULATED.is_metric_absolute

    def test_rtk_expected_accuracy(self):
        assert GnssMode.RTK.expected_horizontal_m == pytest.approx(0.03)
        assert GnssMode.STANDALONE.expected_horizontal_m == pytest.approx(3.0)
        assert GnssMode.NONE.expected_horizontal_m is None
