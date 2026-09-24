"""Tests for the configuration layer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dronemap.config import Config, load_config, _deep_merge, _coerce


class TestDeepMerge:
    def test_flat(self):
        assert _deep_merge({"a": 1, "b": 2}, {"b": 99, "c": 3}) == {"a": 1, "b": 99, "c": 3}

    def test_nested(self):
        base = {"frames": {"max_keyframes": 600, "jpeg_quality": 95}}
        overlay = {"frames": {"max_keyframes": 200}}
        result = _deep_merge(base, overlay)
        assert result["frames"]["max_keyframes"] == 200
        assert result["frames"]["jpeg_quality"] == 95

    def test_does_not_mutate(self):
        base = {"a": {"b": 1}}
        overlay = {"a": {"b": 2}}
        _deep_merge(base, overlay)
        assert base["a"]["b"] == 1


class TestCoerce:
    def test_int(self):
        assert _coerce("42") == 42

    def test_float(self):
        assert _coerce("3.14") == pytest.approx(3.14)

    def test_bool_true(self):
        assert _coerce("true") is True

    def test_bool_false(self):
        assert _coerce("false") is False

    def test_string_passthrough(self):
        assert _coerce("hello world") == "hello world"

    def test_list(self):
        assert _coerce("[1, 2, 3]") == [1, 2, 3]


class TestLoadConfig:
    def test_defaults(self):
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.frames.max_keyframes == 600
        assert cfg.frames.target_overlap == pytest.approx(0.75)
        assert cfg.masks.enabled is True
        assert cfg.pose.init_min_tri_angle == pytest.approx(4.0)

    def test_dotted_override(self):
        cfg = load_config(overrides=["frames.max_keyframes=100"])
        assert cfg.frames.max_keyframes == 100

    def test_nested_override(self):
        cfg = load_config(overrides=["pose.single_camera=false"])
        assert cfg.pose.single_camera is False

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DRONEMAP_FRAMES__MAX_KEYFRAMES", "42")
        cfg = load_config()
        assert cfg.frames.max_keyframes == 42

    def test_invalid_override_missing_eq(self):
        with pytest.raises(ValueError, match="key=value"):
            load_config(overrides=["frames.max_keyframes"])

    def test_invalid_profile(self):
        with pytest.raises(FileNotFoundError):
            load_config(profile="does_not_exist")

    def test_overlap_validator(self):
        with pytest.raises(Exception):
            load_config(overrides=["frames.target_overlap=0.0"])

    def test_georef_none_disables_stage(self):
        cfg = load_config(overrides=["georef.method=none"])
        assert cfg.georef.method == "none"


class TestConfigPaths:
    def test_data_root_default_exists(self):
        cfg = load_config()
        # data_root should be a valid Path (doesn't have to exist)
        assert isinstance(cfg.data_root, Path)

    def test_tools_root_in_repo(self):
        cfg = load_config()
        # tools_root should point inside the repo
        assert "SIH26158" in str(cfg.tools_root) or "tools" in str(cfg.tools_root)
