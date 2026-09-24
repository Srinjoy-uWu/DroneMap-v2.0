"""Discovery and invocation of the external binaries the Core path depends on.

COLMAP and OpenMVS are used as command-line tools rather than libraries, because
prebuilt Windows binaries exist for both and building them from source on Windows
would cost a day we do not have.

Two things here are deliberately defensive:

*   **Flag verification.** COLMAP's published documentation tracks its ``dev``
    branch while we install a tagged release, so flag names can differ. Rather
    than discover that mid-reconstruction, :func:`ColmapTool.supports` parses
    ``colmap <cmd> -h`` and the bootstrap script asserts up front that every flag
    the pipeline uses actually exists.
*   **Log capture with a tail buffer.** These tools are chatty and slow. Full
    output goes to a per-stage log file; the last few lines are kept in memory so
    a failure can report something more useful than "exit code 1".
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# OpenMVS executables, in pipeline order.
OPENMVS_BINARIES = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
)

_EXE = ".exe" if platform.system() == "Windows" else ""


class ToolError(RuntimeError):
    """An external tool exited non-zero, or could not be found."""


class ToolNotFound(ToolError):
    pass


@dataclass
class CompletedTool:
    args: list[str]
    returncode: int
    tail: str
    log_path: Path | None


def _which(name: str, extra_dirs: Iterable[Path] = ()) -> Path | None:
    """Look in explicit directories first, then PATH."""
    for directory in extra_dirs:
        directory = Path(directory)
        if not directory.exists():
            continue
        candidate = directory / f"{name}{_EXE}"
        if candidate.is_file():
            return candidate
        # Release zips nest the binaries a level or two down (bin/, COLMAP-x/bin/).
        for depth in ("*", "*/*", "*/*/*"):
            for hit in directory.glob(f"{depth}/{name}{_EXE}"):
                if hit.is_file():
                    return hit
    found = shutil.which(name)
    return Path(found) if found else None


def _run(
    args: Sequence[str | Path],
    log_path: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    tail_lines: int = 40,
    echo: bool = False,
) -> CompletedTool:
    """Run a command, streaming output to ``log_path`` and keeping a tail in memory."""
    argv = [str(a) for a in args]
    tail: deque[str] = deque(maxlen=tail_lines)

    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", errors="replace")
        log_handle.write(f"\n$ {' '.join(argv)}\n")

    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if log_handle:
                log_handle.write(line + "\n")
            if echo:
                print(line)
        returncode = process.wait()
    except FileNotFoundError as exc:
        raise ToolNotFound(f"executable not found: {argv[0]}") from exc
    finally:
        if log_handle:
            log_handle.close()

    result = CompletedTool(argv, returncode, "\n".join(tail), log_path)
    if check and returncode != 0:
        where = f"\n  log: {log_path}" if log_path else ""
        raise ToolError(
            f"{Path(argv[0]).name} exited {returncode}\n"
            f"  command: {' '.join(argv)}{where}\n"
            f"  last output:\n{_indent(result.tail)}"
        )
    return result


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class ColmapTool:
    """Wrapper around the ``colmap`` multi-command executable."""

    def __init__(self, binary: Path) -> None:
        self.binary = Path(binary)
        self._help_cache: dict[str, str] = {}

    @classmethod
    def find(cls, explicit: Path | None, tools_root: Path) -> ColmapTool:
        binary = Path(explicit) if explicit else _which("colmap", [tools_root, tools_root / "colmap"])
        if not binary or not Path(binary).is_file():
            raise ToolNotFound(
                "COLMAP not found. Run `python scripts/bootstrap.py` to download it, "
                "or set `colmap_bin` in your config."
            )
        return cls(binary)

    def help(self, command: str) -> str:
        """Cached ``colmap <command> -h`` output."""
        if command not in self._help_cache:
            result = _run([self.binary, command, "-h"], check=False, tail_lines=4000)
            self._help_cache[command] = result.tail
        return self._help_cache[command]

    def supports(self, command: str, flag: str) -> bool:
        """Does ``colmap <command>`` accept ``flag``?

        Used by bootstrap to fail loudly at setup time instead of halfway through
        a reconstruction.
        """
        return flag in self.help(command)

    def has_command(self, command: str) -> bool:
        top = _run([self.binary, "-h"], check=False, tail_lines=4000).tail
        return re.search(rf"^\s*{re.escape(command)}\b", top, re.MULTILINE) is not None

    def version(self) -> str:
        text = _run([self.binary, "-h"], check=False, tail_lines=200).tail
        match = re.search(r"COLMAP\s+([0-9][\w.\-+]*)", text)
        return match.group(1) if match else "unknown"

    @property
    def has_cuda(self) -> bool:
        return "with CUDA" in _run([self.binary, "-h"], check=False, tail_lines=200).tail

    def run(
        self,
        command: str,
        args: Sequence[str | Path] = (),
        log_path: Path | None = None,
        check: bool = True,
        echo: bool = False,
    ) -> CompletedTool:
        return _run([self.binary, command, *args], log_path=log_path, check=check, echo=echo)


class OpenMVSTool:
    """Wrapper around the OpenMVS executable suite."""

    def __init__(self, directory: Path, binaries: dict[str, Path]) -> None:
        self.directory = Path(directory)
        self.binaries = binaries

    @classmethod
    def find(cls, explicit: Path | None, tools_root: Path) -> OpenMVSTool:
        search = [Path(explicit)] if explicit else [tools_root, tools_root / "openmvs"]
        binaries: dict[str, Path] = {}
        for name in OPENMVS_BINARIES:
            hit = _which(name, search)
            if hit:
                binaries[name] = hit
        missing = [n for n in OPENMVS_BINARIES if n not in binaries]
        if missing:
            raise ToolNotFound(
                f"OpenMVS binaries not found: {', '.join(missing)}. "
                "Run `python scripts/bootstrap.py` to download them, "
                "or set `openmvs_dir` in your config."
            )
        directory = binaries["DensifyPointCloud"].parent
        return cls(directory, binaries)

    def version(self) -> str:
        # OpenMVS tools print their version banner on a bare invocation, then
        # exit non-zero for missing input - so check=False is expected here.
        text = _run([self.binaries["DensifyPointCloud"]], check=False, tail_lines=60).tail
        match = re.search(r"v([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text)
        return match.group(1) if match else "unknown"

    def run(
        self,
        binary: str,
        args: Sequence[str | Path] = (),
        cwd: Path | None = None,
        log_path: Path | None = None,
        check: bool = True,
        echo: bool = False,
    ) -> CompletedTool:
        if binary not in self.binaries:
            raise ToolNotFound(f"unknown OpenMVS binary {binary!r}")
        return _run(
            [self.binaries[binary], *args],
            cwd=cwd,
            log_path=log_path,
            check=check,
            echo=echo,
        )


class FFmpegTool:
    def __init__(self, binary: Path) -> None:
        self.binary = Path(binary)

    @classmethod
    def find(cls, explicit: Path | None = None) -> FFmpegTool:
        if explicit:
            return cls(Path(explicit))
        on_path = shutil.which("ffmpeg")
        if on_path:
            return cls(Path(on_path))
        # imageio-ffmpeg ships a static build; this is why no admin install is needed.
        try:
            import imageio_ffmpeg

            return cls(Path(imageio_ffmpeg.get_ffmpeg_exe()))
        except Exception as exc:  # pragma: no cover - only when the dep is broken
            raise ToolNotFound(
                "ffmpeg not found and imageio-ffmpeg is unavailable; run `uv sync`"
            ) from exc

    def version(self) -> str:
        text = _run([self.binary, "-version"], check=False, tail_lines=40).tail
        match = re.search(r"ffmpeg version (\S+)", text)
        return match.group(1) if match else "unknown"

    def run(
        self,
        args: Sequence[str | Path],
        log_path: Path | None = None,
        check: bool = True,
    ) -> CompletedTool:
        return _run([self.binary, "-hide_banner", "-nostdin", *args], log_path=log_path, check=check)


def find_blender(explicit: Path | None = None) -> Path | None:
    """Locate Blender, used only by the synthetic ground-truth fixture."""
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    for base in (Path("C:/Program Files/Blender Foundation"), Path("C:/Program Files/Blender")):
        if base.exists():
            hits = sorted(base.glob("*/blender.exe"), reverse=True)
            if hits:
                return hits[0]
    return None


@dataclass
class ToolRegistry:
    """All external tools for a run, resolved once and reused."""

    colmap: ColmapTool | None = None
    openmvs: OpenMVSTool | None = None
    ffmpeg: FFmpegTool | None = None
    blender: Path | None = None

    @classmethod
    def resolve(cls, config, require: Sequence[str] = ()) -> ToolRegistry:  # noqa: ANN001
        """Resolve tools, raising only for those named in ``require``.

        Lets ``dronemap doctor`` report on everything without one missing tool
        masking the rest, while a stage can still hard-require what it needs.
        """
        registry = cls()
        for name, resolver in (
            ("colmap", lambda: ColmapTool.find(config.colmap_bin, config.tools_root)),
            ("openmvs", lambda: OpenMVSTool.find(config.openmvs_dir, config.tools_root)),
            ("ffmpeg", lambda: FFmpegTool.find(config.ffmpeg_bin)),
            ("blender", lambda: find_blender(config.blender_bin)),
        ):
            try:
                setattr(registry, name, resolver())
            except ToolError:
                if name in require:
                    raise
        for name in require:
            if getattr(registry, name, None) is None:
                raise ToolNotFound(f"{name} is required for this stage but was not found")
        return registry

    def versions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.colmap:
            out["colmap"] = self.colmap.version()
            out["colmap_cuda"] = str(self.colmap.has_cuda)
        if self.openmvs:
            out["openmvs"] = self.openmvs.version()
        if self.ffmpeg:
            out["ffmpeg"] = self.ffmpeg.version()
        if self.blender:
            out["blender"] = str(self.blender)
        return out
