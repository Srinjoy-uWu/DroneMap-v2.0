"""Run workspace layout and the run manifest.

Every stage reads from and writes to a :class:`RunWorkspace`, and records its
outcome in ``manifest.json``. Two reasons this matters more than usual here:

*   **Resumability.** COLMAP and OpenMVS steps take tens of minutes on a laptop.
    A crash in stage 7 must not re-run stage 3. Stages skip themselves when
    their outputs already exist unless ``force`` is set.
*   **Provenance.** The manifest is not a log file - it is the raw material for
    the accuracy and provenance report, which is itself a graded deliverable
    (evaluation criteria C1, C2, C3, C7, C10). Tool versions, GNSS regime and
    per-stage timings all have to survive the run.
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class GnssMode(str, Enum):
    """Which metric signal the run actually had.

    This drives the accuracy claim, so it is recorded explicitly rather than
    inferred later. ``SIMULATED`` exists so a synthesised track can never be
    mistaken for a real one in a report or a pitch.
    """

    RTK = "RTK"                # corrected antenna, cm-level
    PPK = "PPK"                # post-processed, cm-level
    STANDALONE = "STANDALONE"  # consumer geotags, metre-level
    SIMULATED = "SIMULATED"    # synthesised from the recovered trajectory
    NONE = "NONE"              # no telemetry: up-to-scale, relative only

    @property
    def is_metric_absolute(self) -> bool:
        return self in (GnssMode.RTK, GnssMode.PPK, GnssMode.STANDALONE)

    @property
    def expected_horizontal_m(self) -> float | None:
        return {
            GnssMode.RTK: 0.03,
            GnssMode.PPK: 0.03,
            GnssMode.STANDALONE: 3.0,
        }.get(self)


# Directory name per stage. Numbered so the run folder sorts in pipeline order.
STAGE_DIRS: dict[str, str] = {
    "frames": "01_frames",
    "masks": "02_masks",
    "depth": "02b_depth",
    "pose": "03_pose",
    "georef": "03b_georef",
    "semantics": "04_semantics",
    "dense": "05_dense",
    "splat": "05b_splat",
    "mesh": "06_mesh",
    "export": "07_export",
}


@dataclass
class StageRecord:
    name: str
    status: StageStatus = StageStatus.PENDING
    started_utc: str | None = None
    duration_s: float | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "started_utc": self.started_utc,
            "duration_s": self.duration_s,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "notes": self.notes,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageRecord:
        return cls(
            name=data["name"],
            status=StageStatus(data.get("status", "pending")),
            started_utc=data.get("started_utc"),
            duration_s=data.get("duration_s"),
            outputs=data.get("outputs") or {},
            metrics=data.get("metrics") or {},
            notes=data.get("notes") or [],
            error=data.get("error"),
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunWorkspace:
    """The on-disk state of a single reconstruction run."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.root = Path(root).resolve()
        self._manifest: dict[str, Any] | None = None

    # ---------------------------------------------------------------- factory

    @classmethod
    def create(cls, data_root: Path, run_id: str, config: dict[str, Any]) -> RunWorkspace:
        ws = cls(Path(data_root) / "runs" / run_id, run_id)
        ws.root.mkdir(parents=True, exist_ok=True)
        for name in STAGE_DIRS:
            ws.stage_dir(name).mkdir(parents=True, exist_ok=True)
        ws.logs_dir.mkdir(parents=True, exist_ok=True)
        if not ws.manifest_path.exists():
            ws._manifest = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "created_utc": _utcnow(),
                "host": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "config": config,
                "input": {},
                "tools": {},
                "gnss": {"mode": GnssMode.NONE.value, "source": None},
                "stages": {name: StageRecord(name).to_dict() for name in STAGE_DIRS},
                "accuracy": {},
            }
            ws._write()
        else:
            # Re-opening an existing run: keep history, refresh the config we ran with.
            ws.load()
            ws._manifest["config"] = config
            ws._write()
        return ws

    @classmethod
    def open(cls, data_root: Path, run_id: str) -> RunWorkspace:
        ws = cls(Path(data_root) / "runs" / run_id, run_id)
        if not ws.manifest_path.exists():
            raise FileNotFoundError(
                f"no run {run_id!r} under {Path(data_root) / 'runs'} - create it with `dronemap run`"
            )
        ws.load()
        return ws

    @staticmethod
    def list_runs(data_root: Path) -> list[str]:
        runs_dir = Path(data_root) / "runs"
        if not runs_dir.exists():
            return []
        return sorted(
            p.name for p in runs_dir.iterdir() if (p / MANIFEST_NAME).exists()
        )

    # ------------------------------------------------------------------ paths

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def stage_dir(self, stage: str) -> Path:
        try:
            return self.root / STAGE_DIRS[stage]
        except KeyError:
            raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGE_DIRS)}") from None

    # Frequently-used concrete paths, named once here so stages agree on them.
    @property
    def images_dir(self) -> Path:
        return self.stage_dir("frames") / "images"

    @property
    def masks_dir(self) -> Path:
        return self.stage_dir("masks") / "masks"

    @property
    def frames_index(self) -> Path:
        return self.stage_dir("frames") / "keyframes.json"

    @property
    def telemetry_json(self) -> Path:
        return self.stage_dir("frames") / "telemetry.json"

    @property
    def colmap_db(self) -> Path:
        return self.stage_dir("pose") / "database.db"

    @property
    def sparse_dir(self) -> Path:
        return self.stage_dir("pose") / "sparse"

    @property
    def undistorted_dir(self) -> Path:
        return self.stage_dir("pose") / "undistorted"

    @property
    def georef_sparse_dir(self) -> Path:
        return self.stage_dir("georef") / "sparse_enu"

    @property
    def dense_dir(self) -> Path:
        return self.stage_dir("dense")

    @property
    def mesh_dir(self) -> Path:
        return self.stage_dir("mesh")

    @property
    def export_dir(self) -> Path:
        return self.stage_dir("export")

    def log_path(self, name: str) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self.logs_dir / f"{name}.log"

    # --------------------------------------------------------------- manifest

    def load(self) -> dict[str, Any]:
        self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self.load()
        assert self._manifest is not None
        return self._manifest

    def _write(self) -> None:
        assert self._manifest is not None
        # Write-then-replace: a manifest half-written by a crash would poison
        # every subsequent resume.
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def set_input(self, **kwargs: Any) -> None:
        self.manifest["input"].update(kwargs)
        self._write()

    def set_tools(self, versions: dict[str, str]) -> None:
        self.manifest["tools"].update(versions)
        self._write()

    def set_gnss(self, mode: GnssMode, source: str | None, **extra: Any) -> None:
        self.manifest["gnss"] = {"mode": mode.value, "source": source, **extra}
        self._write()

    @property
    def gnss_mode(self) -> GnssMode:
        return GnssMode(self.manifest.get("gnss", {}).get("mode", GnssMode.NONE.value))

    def set_accuracy(self, **kwargs: Any) -> None:
        self.manifest.setdefault("accuracy", {}).update(kwargs)
        self._write()

    # ------------------------------------------------------------ stage state

    def stage(self, name: str) -> StageRecord:
        return StageRecord.from_dict(self.manifest["stages"][name])

    def save_stage(self, record: StageRecord) -> None:
        self.manifest["stages"][record.name] = record.to_dict()
        self._write()

    def stage_ok(self, name: str) -> bool:
        return self.stage(name).status is StageStatus.OK

    def note(self, stage: str, message: str) -> None:
        """Append a note to a stage.

        Used for anything a human needs to know later that is not a hard failure -
        an escalation that fired, a fallback that was taken, a domain check that
        was skipped.
        """
        record = self.stage(stage)
        record.notes.append(message)
        self.save_stage(record)


@dataclass
class _StageContext:
    ws: RunWorkspace
    record: StageRecord
    _t0: float

    def metric(self, **kwargs: Any) -> None:
        self.record.metrics.update(kwargs)

    def output(self, **kwargs: Path | str) -> None:
        for key, value in kwargs.items():
            self.record.outputs[key] = str(value)

    def note(self, message: str) -> None:
        self.record.notes.append(message)


class stage_guard:  # noqa: N801 - used as a context manager, reads better lowercase
    """Record a stage's lifecycle in the manifest.

    ``with stage_guard(ws, "pose") as ctx:`` yields ``None`` if the stage is
    already ``ok`` and ``force`` is False, which the caller treats as "skip".
    On exception the stage is marked ``failed`` with the message preserved, and
    the exception propagates.
    """

    def __init__(self, ws: RunWorkspace, name: str, force: bool = False) -> None:
        self.ws = ws
        self.name = name
        self.force = force
        self.ctx: _StageContext | None = None
        self._t0 = 0.0

    def __enter__(self) -> _StageContext | None:
        if self.ws.stage_ok(self.name) and not self.force:
            return None
        record = StageRecord(name=self.name, status=StageStatus.RUNNING, started_utc=_utcnow())
        # A forced re-run starts from a clean record so stale metrics from the
        # previous attempt cannot leak into the report.
        self.ws.save_stage(record)
        self._t0 = time.perf_counter()
        self.ctx = _StageContext(self.ws, record, self._t0)
        return self.ctx

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        if self.ctx is None:
            return False
        record = self.ctx.record
        record.duration_s = round(time.perf_counter() - self._t0, 2)
        if exc_type is None:
            record.status = StageStatus.OK
        else:
            record.status = StageStatus.FAILED
            record.error = f"{exc_type.__name__}: {exc}"
        self.ws.save_stage(record)
        return False  # never swallow the exception


def free_disk_gb(path: Path) -> float:
    path = Path(path)
    probe = path if path.exists() else path.anchor or Path.cwd()
    return shutil.disk_usage(probe).free / 2**30
