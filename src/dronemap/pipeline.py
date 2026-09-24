"""Chunk-aware, resumable orchestrator.

The stage graph is declared here as data, not as a hardcoded call sequence, for
three reasons:

*   ``dronemap run --dry-run`` can print the plan without importing (or running)
    anything heavy.
*   Stretch implementations swap in by pointing a stage at a different module -
    the orchestrator does not change.
*   Chunked execution needs to know which stages are per-chunk and which are
    global, and that is a property of the stage, not of the caller.

Chunking is declared now and executes as a single chunk in the Core path. The
progressive live tier fills in the seam later; nothing else has to move.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import Config
from .workspace import RunWorkspace, StageStatus, stage_guard


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage."""

    key: str                       # matches workspace.STAGE_DIRS
    title: str
    module: str                    # dronemap.<module>
    func: str = "run"
    tools: tuple[str, ...] = ()    # tools that must resolve before this runs
    core: bool = True              # part of the Guaranteed Core path
    wp: str = ""                   # work package, for "not built yet" messages
    # Per-chunk stages run once per keyframe chunk; global stages run once over
    # the whole run.
    per_chunk: bool = False
    # Stage is skipped entirely when this returns False.
    enabled: Callable[[Config], bool] = field(default=lambda _c: True)
    # Stages whose output this one consumes. Used for ordering checks and to
    # explain a skip.
    needs: tuple[str, ...] = ()


STAGE_GRAPH: tuple[StageSpec, ...] = (
    StageSpec(
        key="frames",
        title="Decode, deblur-filter and select keyframes",
        module="stage1_frames",
        tools=("ffmpeg",),
        wp="WP2",
        per_chunk=True,
    ),
    StageSpec(
        key="masks",
        title="Mask dynamic objects (YOLOv8-seg)",
        module="stage2_masks",
        wp="WP3",
        per_chunk=True,
        enabled=lambda c: c.masks.enabled and c.masks.backend != "none",
        needs=("frames",),
    ),
    StageSpec(
        key="depth",
        title="Monocular metric depth (Depth Anything V2 / Metric3D v2)",
        module="stage4_depth",
        wp="WP4a",
        enabled=lambda c: c.depth.enabled,
        needs=("frames",),
    ),
    StageSpec(
        key="pose",
        title="Structure-from-motion (COLMAP or MASt3R)",
        module="stage3_pose",
        tools=("colmap",),
        wp="WP4",
        needs=("frames",),
    ),
    StageSpec(
        key="georef",
        title="Metric scale and georeferencing",
        module="stage3_georef",
        tools=("colmap",),
        wp="WP6",
        enabled=lambda c: c.georef.method != "none",
        needs=("pose",),
    ),
    StageSpec(
        key="semantics",
        title="Semantic layers (SegFormer)",
        module="stage4_semantics",
        wp="WP10",
        enabled=lambda c: c.semantics.enabled,
        needs=("frames",),
    ),
    StageSpec(
        key="dense",
        title="Dense point cloud (OpenMVS)",
        module="stage5_dense",
        tools=("openmvs",),
        wp="WP5",
        # When telemetry alignment is enabled, stage 3b refreshes the
        # undistorted COLMAP workspace from the aligned camera model.  Dense
        # reconstruction must wait for that refresh; otherwise the exported
        # cloud would be labelled with a CRS while still being in arbitrary
        # COLMAP coordinates.
        needs=("pose", "georef"),
    ),
    StageSpec(
        key="mesh",
        title="Mesh, refine and texture (OpenMVS)",
        module="stage6_mesh",
        tools=("openmvs",),
        wp="WP5",
        needs=("dense",),
    ),
    StageSpec(
        key="export",
        title="Exports (OBJ/GLB/LAZ/DSM/DTM/ortho) and accuracy report",
        module="stage7_export",
        wp="WP7",
        needs=("mesh",),
    ),
)

STAGE_BY_KEY = {spec.key: spec for spec in STAGE_GRAPH}


class StageNotImplemented(RuntimeError):
    pass


def load_stage(spec: StageSpec) -> Callable:
    """Import a stage's entry point, with a message that says which WP is missing."""
    try:
        module = importlib.import_module(f"dronemap.{spec.module}")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.endswith(spec.module):
            raise StageNotImplemented(
                f"stage {spec.key!r} is not built yet ({spec.wp}: dronemap/{spec.module}.py)"
            ) from exc
        raise  # a genuine missing dependency inside the stage - do not mask it
    try:
        return getattr(module, spec.func)
    except AttributeError as exc:
        raise StageNotImplemented(
            f"dronemap.{spec.module} has no {spec.func}() entry point"
        ) from exc


def plan_stages(
    config: Config,
    only: Sequence[str] | None = None,
    skip: Sequence[str] | None = None,
) -> list[StageSpec]:
    """Resolve which stages will run, in order."""
    unknown = set(only or ()) | set(skip or ())
    unknown -= set(STAGE_BY_KEY)
    if unknown:
        raise ValueError(
            f"unknown stage(s): {', '.join(sorted(unknown))}; known: {', '.join(STAGE_BY_KEY)}"
        )
    selected = []
    for spec in STAGE_GRAPH:
        if only and spec.key not in only:
            continue
        if skip and spec.key in skip:
            continue
        if not spec.enabled(config):
            continue
        selected.append(spec)
    return selected


def required_tools(specs: Sequence[StageSpec]) -> list[str]:
    seen: list[str] = []
    for spec in specs:
        for tool in spec.tools:
            if tool not in seen:
                seen.append(tool)
    return seen


def skip_reason(config: Config, spec: StageSpec) -> str | None:
    """Why a stage is not in the plan, phrased for a human."""
    if spec.enabled(config):
        return None
    return {
        "masks": f"masks.enabled={config.masks.enabled}, backend={config.masks.backend}",
        "georef": "georef.method=none (up-to-scale mode)",
        "semantics": "semantics.enabled=false (awaiting a verified aerial checkpoint)",
    }.get(spec.key, "disabled by config")


def run_pipeline(
    ws: RunWorkspace,
    config: Config,
    tools,  # ToolRegistry; untyped to avoid a circular import  # noqa: ANN001
    specs: Sequence[StageSpec] | None = None,
    force: Sequence[str] | bool = (),
    on_stage: Callable[[StageSpec, str], None] | None = None,
) -> None:
    """Execute stages in order, recording each in the manifest.

    ``force`` is either ``True`` (re-run everything) or a collection of stage
    keys to re-run. Anything already ``ok`` and not forced is skipped, which is
    what makes a crash in stage 7 cheap.
    """
    specs = list(specs if specs is not None else plan_stages(config))
    force_all = force is True
    force_keys = set() if isinstance(force, bool) else set(force)

    for spec in specs:
        forced = force_all or spec.key in force_keys
        # Guard against running a stage whose input never materialised. Cheaper
        # than letting COLMAP fail on an empty image directory.
        for dependency in spec.needs:
            dep_spec = STAGE_BY_KEY[dependency]
            if not dep_spec.enabled(config):
                continue
            if not ws.stage_ok(dependency):
                raise RuntimeError(
                    f"stage {spec.key!r} needs {dependency!r}, which is "
                    f"{ws.stage(dependency).status.value}"
                )

        with stage_guard(ws, spec.key, force=forced) as ctx:
            if ctx is None:
                if on_stage:
                    on_stage(spec, "skipped (already ok)")
                continue
            if on_stage:
                on_stage(spec, "running")
            entry = load_stage(spec)
            entry(ws=ws, config=config, tools=tools, ctx=ctx)
        if on_stage and ws.stage(spec.key).status is StageStatus.OK:
            on_stage(spec, "ok")
