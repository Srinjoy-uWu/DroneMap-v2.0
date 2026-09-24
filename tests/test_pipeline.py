"""Tests for pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from dronemap.config import load_config
from dronemap.pipeline import (
    STAGE_BY_KEY,
    STAGE_GRAPH,
    StageSpec,
    plan_stages,
    required_tools,
    skip_reason,
    run_pipeline,
    StageNotImplemented,
    load_stage,
)
from dronemap.workspace import RunWorkspace, StageStatus


@pytest.fixture()
def config():
    return load_config()


@pytest.fixture()
def ws(tmp_path: Path, config):
    return RunWorkspace.create(tmp_path, "pipeline-test", config=config.model_dump(mode="json"))


class TestPlanStages:
    def test_all_stages_by_default(self, config):
        # masks and georef may be disabled by default config
        specs = plan_stages(config)
        keys = [s.key for s in specs]
        assert "frames" in keys
        assert "pose" in keys
        assert "dense" in keys
        assert "mesh" in keys
        assert "export" in keys

    def test_only_filter(self, config):
        specs = plan_stages(config, only=["frames", "pose"])
        assert len(specs) == 2
        assert {s.key for s in specs} == {"frames", "pose"}

    def test_skip_filter(self, config):
        specs = plan_stages(config, skip=["masks"])
        assert all(s.key != "masks" for s in specs)

    def test_unknown_stage_raises(self, config):
        with pytest.raises(ValueError, match="unknown stage"):
            plan_stages(config, only=["fake_stage"])

    def test_semantics_disabled_by_default(self, config):
        # semantics.enabled defaults to False
        specs = plan_stages(config)
        assert all(s.key != "semantics" for s in specs)

    def test_semantics_enabled_via_config(self):
        cfg = load_config(overrides=["semantics.enabled=true", "semantics.checkpoint_domain_verified=true"])
        specs = plan_stages(cfg)
        assert any(s.key == "semantics" for s in specs)

    def test_georef_disabled_when_no_telemetry(self):
        cfg = load_config(overrides=["georef.method=none"])
        specs = plan_stages(cfg)
        assert all(s.key != "georef" for s in specs)

    def test_masks_disabled_when_backend_none(self):
        cfg = load_config(overrides=["masks.backend=none"])
        specs = plan_stages(cfg)
        assert all(s.key != "masks" for s in specs)


class TestRequiredTools:
    def test_frames_needs_ffmpeg(self, config):
        specs = plan_stages(config, only=["frames"])
        tools = required_tools(specs)
        assert "ffmpeg" in tools

    def test_pose_needs_colmap(self, config):
        specs = plan_stages(config, only=["pose"])
        tools = required_tools(specs)
        assert "colmap" in tools

    def test_dense_needs_openmvs(self, config):
        specs = plan_stages(config, only=["dense"])
        tools = required_tools(specs)
        assert "openmvs" in tools

    def test_no_duplicates(self, config):
        specs = plan_stages(config)
        tools = required_tools(specs)
        assert len(tools) == len(set(tools))


class TestSkipReason:
    def test_enabled_stage_returns_none(self, config):
        frames_spec = STAGE_BY_KEY["frames"]
        assert skip_reason(config, frames_spec) is None

    def test_georef_reason(self):
        cfg = load_config(overrides=["georef.method=none"])
        spec = STAGE_BY_KEY["georef"]
        reason = skip_reason(cfg, spec)
        assert reason is not None and "none" in reason

    def test_masks_reason_backend_none(self):
        cfg = load_config(overrides=["masks.backend=none"])
        spec = STAGE_BY_KEY["masks"]
        reason = skip_reason(cfg, spec)
        assert reason is not None


class TestRunPipeline:
    def test_dependency_check_fails_on_missing_upstream(self, ws: RunWorkspace, config):
        """Running 'masks' before 'frames' is OK is enabled=False, but should raise if enabled."""
        cfg = load_config()
        # frames is not OK, masks needs it
        specs = plan_stages(cfg, only=["masks"])
        from dronemap.tools import ToolRegistry
        registry = ToolRegistry()  # empty registry (no real tools)

        # masks.enabled checks backend; backend = 'yolo' so masks is in plan
        # But frames is not OK → should raise RuntimeError
        with pytest.raises(RuntimeError, match="needs.*frames"):
            run_pipeline(ws, cfg, registry, specs)

    def test_skip_already_ok_stage(self, ws: RunWorkspace, config):
        """A stage that is already OK should be skipped (ctx=None)."""
        from dronemap.workspace import stage_guard

        # Mark 'frames' as OK manually
        with stage_guard(ws, "frames"):
            pass  # immediately marks OK

        seen_running: list[str] = []

        def on_stage(spec: StageSpec, state: str) -> None:
            if state == "running":
                seen_running.append(spec.key)

        cfg = load_config(overrides=["georef.method=none"])
        specs = plan_stages(cfg, only=["frames"])
        from dronemap.tools import ToolRegistry
        registry = ToolRegistry()

        # frames is already OK and not forced → no stage should actually run
        # (it would fail if load_stage tried, since the module exists but would need real input)
        # We test that 'running' is NOT emitted for an already-OK stage
        try:
            run_pipeline(ws, cfg, registry, specs, on_stage=on_stage)
        except Exception:
            pass  # may fail when the stage actually tries to execute

        assert "frames" not in seen_running

    def test_stage_not_implemented_error(self):
        """load_stage for a missing module should raise StageNotImplemented."""
        fake_spec = StageSpec(
            key="fake",
            title="Fake stage",
            module="stage_does_not_exist",
            wp="WP99",
        )
        with pytest.raises(StageNotImplemented, match="not built yet"):
            load_stage(fake_spec)


class TestStageGraph:
    def test_all_stages_have_keys(self):
        for spec in STAGE_GRAPH:
            assert spec.key
            assert spec.module
            assert spec.wp

    def test_stage_by_key_complete(self):
        for spec in STAGE_GRAPH:
            assert STAGE_BY_KEY[spec.key] is spec

    def test_needs_reference_valid_stages(self):
        valid_keys = set(STAGE_BY_KEY)
        for spec in STAGE_GRAPH:
            for dep in spec.needs:
                assert dep in valid_keys, f"{spec.key} needs unknown stage {dep!r}"

    def test_dense_waits_for_georeferencing_when_enabled(self):
        """Avoid exporting a cloud in arbitrary SfM coordinates with a GIS label."""
        assert "georef" in STAGE_BY_KEY["dense"].needs
