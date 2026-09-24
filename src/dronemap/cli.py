"""Command-line interface.

Command surface:

    dronemap doctor                    environment report; run this first
    dronemap run --video FILE          full pipeline (--dry-run to see the plan)
    dronemap frames|masks|pose|...     one stage, on an existing run
    dronemap runs | show ID            what runs exist, and how far each got
    dronemap config                    the fully resolved configuration
    dronemap serve                     the measurement viewer

Per-stage commands are generated from :data:`dronemap.pipeline.STAGE_GRAPH`, so a
new stage gets a command for free and cannot drift out of sync with the graph.
"""

from __future__ import annotations

import importlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import Config, load_config
from .pipeline import (
    STAGE_BY_KEY,
    STAGE_GRAPH,
    StageSpec,
    plan_stages,
    required_tools,
    run_pipeline,
    skip_reason,
)
from .tools import ToolError, ToolRegistry
from .workspace import GnssMode, RunWorkspace, StageStatus, free_disk_gb

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Single-pass drone video -> georeferenced, metrically accurate 3D model.",
)
# Force UTF-8 output so Unicode arrows/emoji don't crash on Windows cp1252 terminals.
import io as _io
console = Console(file=_io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout)

OK = "[green]OK[/green]"
BAD = "[red]MISSING[/red]"
WARN = "[yellow]--[/yellow]"

_STATUS_COLOUR = {
    StageStatus.OK: "green",
    StageStatus.FAILED: "red",
    StageStatus.RUNNING: "yellow",
    StageStatus.SKIPPED: "bright_black",
    StageStatus.PENDING: "bright_black",
}


# --------------------------------------------------------------------- helpers


def _config(profile: str | None, overrides: list[str] | None, data_root: Path | None) -> Config:
    try:
        config = load_config(profile=profile, overrides=overrides)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    if data_root:
        config.data_root = data_root
    return config


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower() or "run"


def _new_run_id(video: Path | None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{_slug(video.stem)[:24]}-{stamp}" if video else f"run-{stamp}"


def _open_run(config: Config, run_id: str | None) -> RunWorkspace:
    """Open a named run, or the most recent one."""
    if run_id is None:
        existing = RunWorkspace.list_runs(config.data_root)
        if not existing:
            console.print(
                f"[red]no runs under {config.data_root / 'runs'}[/red] - start one with "
                "`dronemap run --video FILE`"
            )
            raise typer.Exit(2)
        run_id = existing[-1]
        console.print(f"[bright_black]using most recent run: {run_id}[/bright_black]")
    try:
        return RunWorkspace.open(config.data_root, run_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _check_disk(config: Config) -> None:
    free = free_disk_gb(config.data_root)
    if free < config.min_free_disk_gb:
        console.print(
            f"[red]only {free:.1f} GB free at {config.data_root}[/red] "
            f"(need {config.min_free_disk_gb:.0f} GB). A full run is ~10-15 GB.\n"
            f"Free space, lower `min_free_disk_gb`, or point `data_root` at another drive."
        )
        raise typer.Exit(2)


def _report_progress(spec: StageSpec, state: str) -> None:
    if state == "running":
        console.rule(f"[bold]{spec.key}[/bold]  {spec.title}")
    elif state == "skipped (already ok)":
        console.print(f"[bright_black]{spec.key:<10} skipped (already complete)[/bright_black]")
    elif state == "ok":
        console.print(f"[green]{spec.key:<10} ok[/green]")


def _print_plan(config: Config, specs: list[StageSpec], ws: RunWorkspace | None) -> None:
    table = Table(title="Planned stages", title_justify="left", show_lines=False)
    table.add_column("#", justify="right", style="bright_black")
    table.add_column("stage")
    table.add_column("what it does")
    table.add_column("tools")
    table.add_column("output directory", style="bright_black")
    table.add_column("status")

    for index, spec in enumerate(specs, start=1):
        if ws is not None and ws.manifest_path.exists():
            record = ws.stage(spec.key)
            status = Text(record.status.value, style=_STATUS_COLOUR[record.status])
        else:
            status = Text("pending", style="bright_black")
        out_dir = ws.stage_dir(spec.key) if ws else Path(f"runs/<id>/{spec.key}")
        table.add_row(
            str(index),
            spec.key,
            spec.title,
            ", ".join(spec.tools) or "-",
            str(out_dir),
            status,
        )
    console.print(table)

    excluded = [s for s in STAGE_GRAPH if s not in specs]
    if excluded:
        console.print("[bright_black]not in this plan:[/bright_black]")
        for spec in excluded:
            reason = skip_reason(config, spec) or "excluded by --only/--skip"
            console.print(f"  [bright_black]{spec.key:<10} {reason}[/bright_black]")


# ---------------------------------------------------------------------- doctor


def _torch_report(table: Table) -> bool:
    try:
        import torch
    except ImportError:
        table.add_row("torch", BAD, "run `uv sync --extra ml`")
        return False
    if not torch.cuda.is_available():
        table.add_row(
            "torch", f"[yellow]{torch.__version__} (CPU only)[/yellow]",
            "CUDA not visible; masking and semantics will be slow",
        )
        return True
    props = torch.cuda.get_device_properties(0)
    table.add_row("torch", OK, f"{torch.__version__}, CUDA {torch.version.cuda}")
    table.add_row("gpu", OK, f"{props.name}, {props.total_memory / 2**30:.1f} GB VRAM")
    return True


def _import_report(table: Table, modules: dict[str, str]) -> None:
    for label, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            table.add_row(label, BAD, f"import {module_name} failed")
            continue
        version = getattr(module, "__version__", "") or ""
        table.add_row(label, OK, str(version))


@app.command()
def doctor(
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
    set_: Optional[list[str]] = typer.Option(None, "--set", help="key=value override"),
) -> None:
    """Check that everything the pipeline needs is present and working."""
    config = _config(profile, set_, None)
    table = Table(show_header=True, header_style="bold")
    table.add_column("component", no_wrap=True)
    table.add_column("state", no_wrap=True)
    table.add_column("detail")

    table.add_row("python", OK, f"{sys.version.split()[0]}  ({sys.executable})")
    ok = _torch_report(table)

    registry = ToolRegistry.resolve(config)

    if registry.colmap:
        cuda = "CUDA" if registry.colmap.has_cuda else "[yellow]no CUDA[/yellow]"
        table.add_row("colmap", OK, f"{registry.colmap.version()}, {cuda}  ({registry.colmap.binary})")
    else:
        table.add_row("colmap", BAD, "run `python scripts/bootstrap.py`")
        ok = False

    if registry.openmvs:
        table.add_row("openmvs", OK, f"{registry.openmvs.version()}  ({registry.openmvs.directory})")
    else:
        table.add_row("openmvs", BAD, "run `python scripts/bootstrap.py`")
        ok = False

    if registry.ffmpeg:
        table.add_row("ffmpeg", OK, f"{registry.ffmpeg.version()}  ({registry.ffmpeg.binary})")
    else:
        table.add_row("ffmpeg", BAD, "run `uv sync` (imageio-ffmpeg bundles a binary)")
        ok = False

    vocab = next(iter(sorted(Path(config.tools_root).glob("**/vocab_tree*.bin"))), None)
    table.add_row(
        "vocab tree", OK if vocab else WARN,
        str(vocab) if vocab else "absent - loop detection will be disabled",
    )

    # Optional: only the Blender ground-truth fixture needs it.
    table.add_row(
        "blender", OK if registry.blender else WARN,
        str(registry.blender) if registry.blender else "optional (synthetic ground-truth fixture)",
    )

    _import_report(table, {
        "opencv": "cv2",
        "open3d": "open3d",
        "pycolmap": "pycolmap",
        "rasterio": "rasterio",
        "laspy": "laspy",
        "pyproj": "pyproj",
        "trimesh": "trimesh",
        "ultralytics": "ultralytics",
        "transformers": "transformers",
        "evo": "evo",
        "fastapi": "fastapi",
    })

    free = free_disk_gb(config.data_root)
    enough = free >= config.min_free_disk_gb
    table.add_row(
        "free disk",
        OK if enough else f"[red]{free:.1f} GB[/red]",
        f"{free:.1f} GB at {config.data_root}  (minimum {config.min_free_disk_gb:.0f} GB; "
        f"a run needs ~10-15 GB)",
    )
    ok = ok and enough

    table.add_row("data root", OK, str(config.data_root))
    table.add_row("tools root", OK, str(config.tools_root))

    console.print(table)
    if ok:
        console.print("\n[green]Ready.[/green] Next: `dronemap run --video <file> --dry-run`")
    else:
        console.print("\n[red]Not ready[/red] - fix the MISSING rows above.")
        raise typer.Exit(1)


# ------------------------------------------------------------------------- run


@app.command()
def run(
    video: Optional[Path] = typer.Option(None, "--video", "-v", help="input drone video"),
    telemetry: Optional[Path] = typer.Option(
        None, "--telemetry", "-t", help="GPS/flight log sidecar (DJI .srt, .csv, .tlog, ...)"
    ),
    no_telemetry: bool = typer.Option(
        False, "--no-telemetry", help="up-to-scale mode: no georeferencing, relative measures only"
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="reuse or name a run"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="e.g. lowvram, smoke"),
    set_: Optional[list[str]] = typer.Option(None, "--set", help="key=value override, repeatable"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
    only: Optional[list[str]] = typer.Option(None, "--only", help="run just these stages"),
    skip: Optional[list[str]] = typer.Option(None, "--skip", help="exclude these stages"),
    force: Optional[list[str]] = typer.Option(
        None, "--force", help="re-run a completed stage; 'all' for everything"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the plan and exit"),
) -> None:
    """Run the pipeline. Resumable: completed stages are skipped unless forced."""
    config = _config(profile, set_, data_root)
    if no_telemetry:
        config.georef.method = "none"

    try:
        specs = plan_stages(config, only=only, skip=skip)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    if not specs:
        console.print("[red]nothing to run[/red] - --only/--skip excluded every stage")
        raise typer.Exit(2)

    if dry_run:
        console.print(f"[bold]profile:[/bold] {profile or 'default'}")
        if video:
            console.print(f"[bold]video:[/bold] {video.resolve()}  "
                          f"{'[red](not found)[/red]' if not video.exists() else ''}")
        console.print(f"[bold]telemetry:[/bold] "
                      f"{telemetry.resolve() if telemetry else ('disabled' if no_telemetry else 'auto-detect')}")
        console.print(f"[bold]run id:[/bold] {run_id or _new_run_id(video) + '  (generated)'}")
        console.print(f"[bold]run root:[/bold] {config.data_root / 'runs' / (run_id or '<id>')}\n")

        ws = None
        if run_id:
            candidate = RunWorkspace(config.data_root / "runs" / run_id, run_id)
            ws = candidate if candidate.manifest_path.exists() else None
        _print_plan(config, specs, ws)

        registry = ToolRegistry.resolve(config)
        needed = required_tools(specs)
        table = Table(title="Required tools", title_justify="left")
        table.add_column("tool")
        table.add_column("state")
        table.add_column("path")
        for name in needed:
            resolved = getattr(registry, name, None)
            binary = getattr(resolved, "binary", None) or getattr(resolved, "directory", None)
            table.add_row(name, OK if resolved else BAD, str(binary) if resolved else "-")
        console.print(table)
        console.print(f"\nfree disk: {free_disk_gb(config.data_root):.1f} GB "
                      f"(minimum {config.min_free_disk_gb:.0f} GB)")
        console.print("[bright_black]dry run - nothing was written[/bright_black]")
        return

    needs_frames = any(spec.key == "frames" for spec in specs)
    if needs_frames and not run_id:
        if video is None:
            console.print("[red]--video is required[/red] (or pass --run-id to resume a run)")
            raise typer.Exit(2)
        if not video.exists():
            console.print(f"[red]video not found:[/red] {video}")
            raise typer.Exit(2)

    _check_disk(config)

    run_id = run_id or _new_run_id(video)
    ws = RunWorkspace.create(config.data_root, run_id, config.model_dump(mode="json"))
    if video:
        ws.set_input(video=str(video.resolve()), telemetry=str(telemetry.resolve()) if telemetry else None)
    if no_telemetry:
        ws.set_gnss(GnssMode.NONE, source=None, reason="--no-telemetry")

    force_arg: object = True if force and "all" in force else set(force or ())
    try:
        registry = ToolRegistry.resolve(config, require=required_tools(specs))
    except ToolError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    ws.set_tools(registry.versions())

    console.print(f"[bold]run[/bold] {run_id}   [bright_black]{ws.root}[/bright_black]")
    try:
        run_pipeline(ws, config, registry, specs, force=force_arg, on_stage=_report_progress)
    except Exception as exc:
        console.print(f"\n[red]{type(exc).__name__}:[/red] {exc}")
        console.print(f"[bright_black]logs: {ws.logs_dir}[/bright_black]")
        console.print(f"[bright_black]resume with: dronemap run --run-id {run_id}[/bright_black]")
        raise typer.Exit(1) from exc

    console.print(f"\n[green]done[/green]  {ws.export_dir}")


# ---------------------------------------------------------- per-stage commands


def _make_stage_command(spec: StageSpec):
    def command(
        run_id: Optional[str] = typer.Option(None, "--run-id", help="defaults to the newest run"),
        profile: Optional[str] = typer.Option(None, "--profile", "-p"),
        set_: Optional[list[str]] = typer.Option(None, "--set"),
        data_root: Optional[Path] = typer.Option(None, "--data-root"),
        force: bool = typer.Option(False, "--force", help="re-run even if already complete"),
    ) -> None:
        config = _config(profile, set_, data_root)
        ws = _open_run(config, run_id)
        try:
            registry = ToolRegistry.resolve(config, require=spec.tools)
            run_pipeline(ws, config, registry, [spec], force=force, on_stage=_report_progress)
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
            raise typer.Exit(1) from exc

    command.__doc__ = f"Stage: {spec.title}."
    return command


for _spec in STAGE_GRAPH:
    app.command(name=_spec.key)(_make_stage_command(_spec))


# ------------------------------------------------------------------ inspection


@app.command()
def runs(
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
) -> None:
    """List runs and how far each got."""
    config = _config(None, None, data_root)
    ids = RunWorkspace.list_runs(config.data_root)
    if not ids:
        console.print(f"[bright_black]no runs under {config.data_root / 'runs'}[/bright_black]")
        return
    table = Table()
    table.add_column("run id")
    table.add_column("created")
    table.add_column("gnss")
    table.add_column("progress")
    for run_id in ids:
        ws = RunWorkspace.open(config.data_root, run_id)
        done = [k for k in STAGE_BY_KEY if ws.stage_ok(k)]
        failed = [k for k in STAGE_BY_KEY if ws.stage(k).status is StageStatus.FAILED]
        progress = f"{len(done)}/{len(STAGE_BY_KEY)} ok"
        if failed:
            progress += f"  [red]failed: {', '.join(failed)}[/red]"
        table.add_row(run_id, ws.manifest.get("created_utc", "?"), ws.gnss_mode.value, progress)
    console.print(table)


@app.command()
def show(
    run_id: Optional[str] = typer.Argument(None, help="defaults to the newest run"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
) -> None:
    """Show a run's stage status, timings and metrics."""
    config = _config(None, None, data_root)
    ws = _open_run(config, run_id)
    console.print(f"[bold]{ws.run_id}[/bold]   [bright_black]{ws.root}[/bright_black]")
    manifest = ws.manifest
    console.print(f"input:  {manifest.get('input') or '-'}")
    console.print(f"gnss:   {manifest.get('gnss') or '-'}")
    console.print(f"tools:  {manifest.get('tools') or '-'}\n")

    table = Table()
    table.add_column("stage")
    table.add_column("status")
    table.add_column("duration", justify="right")
    table.add_column("metrics / notes")
    for key in STAGE_BY_KEY:
        record = ws.stage(key)
        detail = ", ".join(f"{k}={v}" for k, v in record.metrics.items())
        if record.error:
            detail = f"[red]{record.error}[/red]"
        elif record.notes:
            detail = (detail + "  " if detail else "") + "; ".join(record.notes)
        table.add_row(
            key,
            Text(record.status.value, style=_STATUS_COLOUR[record.status]),
            f"{record.duration_s:.1f}s" if record.duration_s else "-",
            detail or "-",
        )
    console.print(table)

    accuracy = manifest.get("accuracy")
    if accuracy:
        console.print("\n[bold]accuracy[/bold]")
        for key, value in accuracy.items():
            console.print(f"  {key}: {value}")


@app.command(name="config")
def show_config(
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
    set_: Optional[list[str]] = typer.Option(None, "--set"),
) -> None:
    """Print the fully resolved configuration (all five layers merged)."""
    import yaml

    config = _config(profile, set_, None)
    console.print(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


@app.command()
def clean(
    run_id: Optional[str] = typer.Argument(None, help="Specific run ID to delete (optional)"),
    all_: bool = typer.Option(False, "--all", "-a", help="Run full safe cleanup (scratch, logs, datasets, previous runs)"),
    intermediate: bool = typer.Option(False, "--intermediate", "-i", help="Prune intermediate depth maps (.dmap) from finished runs"),
    datasets: bool = typer.Option(False, "--datasets", help="Remove unused datasets, keeping only the primary fixture"),
    scratch: bool = typer.Option(False, "--scratch", help="Remove temporary scratch folders and downloaded zips"),
    keep: str = typer.Option("fixture_orbit_hires", "--keep", "-k", help="Run ID to keep when cleaning runs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be cleaned without deleting"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Clean redundant files, scratch directories, intermediate caches, and old runs."""
    from .cleanup import (
        clean_intermediate_depth_maps,
        clean_root_logs,
        clean_runs,
        clean_scratch,
        clean_unused_datasets,
    )

    config = _config(None, None, data_root)
    data_path = Path(config.data_root)
    repo_root = Path(__file__).resolve().parents[2]

    # Mode 1: Delete a specific single run by ID
    if run_id:
        target = data_path / "runs" / run_id
        if not target.exists():
            console.print(f"[red]no such run:[/red] {target}")
            raise typer.Exit(2)
        size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 2**30
        if not yes and not typer.confirm(f"delete {target} ({size_gb:.2f} GB)?"):
            raise typer.Abort()
        shutil.rmtree(target)
        console.print(f"[green]deleted[/green] {target}  ({size_gb:.2f} GB freed)")
        return

    # Mode 2: Multi-target / automated cleanup
    if not (all_ or intermediate or datasets or scratch):
        console.print("[yellow]Specify a run ID or one of --all, --intermediate, --datasets, --scratch[/yellow]")
        console.print("Run `dronemap clean --help` for usage details.")
        return

    total_freed = 0

    if all_ or scratch:
        logs = clean_root_logs(repo_root, dry_run=dry_run)
        if logs:
            console.print(f"[green]Cleaned {len(logs)} root log file(s)[/green]")
        scratch_res = clean_scratch(repo_root, data_path, dry_run=dry_run)
        freed_mb = scratch_res.get("total_freed_bytes", 0) / (1024 * 1024)
        total_freed += scratch_res.get("total_freed_bytes", 0)
        console.print(f"[green]Cleaned scratch & downloads:[/green] {freed_mb:.1f} MB freed")

    if all_ or datasets:
        ds_res = clean_unused_datasets(data_path, keep_datasets=["synthetic_flight"], dry_run=dry_run)
        freed_mb = ds_res.get("total_freed_bytes", 0) / (1024 * 1024)
        total_freed += ds_res.get("total_freed_bytes", 0)
        console.print(f"[green]Cleaned unused datasets:[/green] {freed_mb:.1f} MB freed (kept 'synthetic_flight')")

    if all_:
        run_res = clean_runs(data_path, keep_run_ids=[keep], prune_depth_maps=True, dry_run=dry_run)
        freed_gb = run_res.get("total_freed_bytes", 0) / (1024 * 1024 * 1024)
        total_freed += run_res.get("total_freed_bytes", 0)
        n_del = len(run_res.get("deleted_runs", []))
        console.print(f"[green]Cleaned {n_del} previous run(s) and pruned .dmaps:[/green] {freed_gb:.2f} GB freed (kept '{keep}')")
    elif intermediate:
        runs_dir = data_path / "runs"
        pruned_count = 0
        pruned_bytes = 0
        if runs_dir.exists():
            for r in runs_dir.iterdir():
                if r.is_dir():
                    res = clean_intermediate_depth_maps(r, dry_run=dry_run)
                    pruned_count += res.get("deleted_count", 0)
                    pruned_bytes += res.get("freed_bytes", 0)
        total_freed += pruned_bytes
        console.print(f"[green]Pruned {pruned_count} .dmap depth map cache files:[/green] {pruned_bytes / (1024 * 1024):.1f} MB freed")

    console.print(f"\n[bold green]Total reclaimed:[/bold green] {total_freed / (1024 * 1024 * 1024):.2f} GB")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    data_root: Optional[Path] = typer.Option(None, "--data-root"),
) -> None:
    """Serve the measurement viewer and job API."""
    config = _config(None, None, data_root)
    try:
        server = importlib.import_module("dronemap.api.server")
    except ModuleNotFoundError as exc:
        if exc.name and "dronemap.api" in exc.name:
            console.print("[red]the viewer is not built yet[/red] (WP8: dronemap/api/server.py)")
            raise typer.Exit(1) from exc
        raise
    server.serve(config=config, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
