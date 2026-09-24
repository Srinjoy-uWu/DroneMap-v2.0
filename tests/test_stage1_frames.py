"""Tests for Stage 1 streaming video decoding and keyframe extraction."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from dronemap.config import load_config
from dronemap.stage1_frames import _probe_video, FlowSampler, run as run_stage1
from dronemap.tools import ToolRegistry
from dronemap.workspace import RunWorkspace, stage_guard


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Create a short 30-frame synthetic mp4 video with moving circle."""
    video_path = tmp_path / "test_motion.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        # Draw moving textured rectangle to generate optical flow and sharpness
        x = 50 + i * 5
        cv2.rectangle(frame, (x, 80), (x + 60, 140), (200, 200, 200), -1)
        cv2.circle(frame, (x + 30, 110), 15, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return video_path


def test_probe_video(dummy_video: Path):
    info = _probe_video(dummy_video)
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["fps"] == 10.0
    assert info["frame_count"] == 30


def test_flow_sampler():
    sampler = FlowSampler(target_samples=5, total_hint=20)
    for i in range(20):
        gray = np.full((100, 100), 50, dtype=np.uint8)
        cv2.rectangle(gray, (10 + i * 2, 20), (30 + i * 2, 50), 220, -1)
        sampler.observe(i, gray)
    stride = sampler.compute_stride(100)
    assert 2 <= stride <= 30


def test_stage1_streaming_run(dummy_video: Path, tmp_path: Path):
    cfg = load_config()
    cfg.frames.max_keyframes = 10
    cfg.frames.sharpness_window = 3
    ws = RunWorkspace.create(tmp_path, "test_stream_01", config=cfg.model_dump())
    ws.set_input(video=str(dummy_video))

    tools = ToolRegistry.resolve(cfg, require=())
    with stage_guard(ws, "frames") as ctx:
        run_stage1(ws, cfg, tools, ctx)

    assert ws.frames_index.exists()
    keyframes = list(ws.images_dir.glob("*.jpg"))
    assert len(keyframes) > 0
    assert len(keyframes) <= 10
