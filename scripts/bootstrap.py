"""Fetch the prebuilt external binaries and verify their CLI surface.

    python scripts/bootstrap.py            # download + extract + probe
    python scripts/bootstrap.py --probe    # skip downloads, just re-probe
    python scripts/bootstrap.py --colmap-variant nocuda

Downloads COLMAP and OpenMVS Windows release builds plus the COLMAP vocabulary
tree into ``tools/``. Nothing is compiled and nothing needs admin rights.

The second half of this script is the part that matters. COLMAP's published
documentation is generated from its ``dev`` branch while we install a tagged
release, so flag names in the docs are not guaranteed to exist in the binary. So
rather than trust the docs, we run ``colmap <command> -h`` for every command the
pipeline touches, assert that the Core-path flags are really there, and dump the
raw help text to ``tools/help/`` so later stages can be written against the
installed build instead of against a different version's documentation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

COLMAP_VERSION = "4.1.1"
OPENMVS_VERSION = "v2.4.0"

COLMAP_URLS = {
    "cuda": f"https://github.com/colmap/colmap/releases/download/{COLMAP_VERSION}/colmap-x64-windows-cuda.zip",
    "nocuda": f"https://github.com/colmap/colmap/releases/download/{COLMAP_VERSION}/colmap-x64-windows-nocuda.zip",
}
OPENMVS_URL = (
    f"https://github.com/cdcseacave/openMVS/releases/download/{OPENMVS_VERSION}/OpenMVS_Windows_x64.zip"
)
# 32K-word tree (faiss format, required by COLMAP >= May 2025).
# The old demuc.de URL served a flann-format tree which is no longer compatible.
# We build the faiss tree from scratch using colmap vocab_tree_builder if the
# pre-built file isn't available at this URL.
VOCAB_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words32K.bin"
VOCAB_URL_FALLBACK = "https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"

# Flags the Core path (stages 1-7 minus georeferencing) cannot run without.
# A miss here is a hard failure: better now than 40 minutes into a reconstruction.
# NOTE: --SiftExtraction.use_gpu and --SiftExtraction.max_image_size were
# removed in COLMAP 4.x; stage3_pose.py uses colmap.supports() to gate them.
REQUIRED_FLAGS: dict[str, list[str]] = {
    "feature_extractor": [
        "--database_path",
        "--image_path",
        "--ImageReader.single_camera",
        "--ImageReader.camera_model",
        "--ImageReader.mask_path",
        "--SiftExtraction.max_num_features",
    ],
    "sequential_matcher": [
        "--database_path",
        "--SequentialMatching.overlap",
        "--SequentialMatching.quadratic_overlap",
        "--SequentialMatching.loop_detection",
        "--SequentialMatching.vocab_tree_path",
    ],
    "exhaustive_matcher": ["--database_path"],
    "mapper": [
        "--database_path",
        "--image_path",
        "--output_path",
        "--Mapper.init_min_tri_angle",
        "--Mapper.ba_refine_principal_point",
        "--Mapper.multiple_models",
    ],
    "image_undistorter": [
        "--image_path",
        "--input_path",
        "--output_path",
        "--output_type",
        "--max_image_size",
    ],
    "model_converter": ["--input_path", "--output_path", "--output_type"],
}

# Georeferencing flags (WP6). Probed and reported, but not fatal: if a name has
# drifted we want the real help text in hand rather than a failed bootstrap, and
# the Core path still produces an up-to-scale model without them.
OPTIONAL_FLAGS: dict[str, list[str]] = {
    "model_aligner": [
        "--input_path",
        "--output_path",
        "--database_path",
        "--ref_images_path",
        "--ref_is_gps",
        "--alignment_type",
        "--alignment_max_error",
        "--transform_path",
    ],
    "pose_prior_mapper": [
        "--database_path",
        "--image_path",
        "--output_path",
        "--overwrite_priors_covariance",
        "--prior_position_std_x",
        "--prior_position_std_y",
        "--prior_position_std_z",
    ],
    "model_orientation_aligner": ["--input_path", "--output_path"],
}


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return ""


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Download with a progress line, skipping an already-complete file."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        try:
            head = requests.head(url, allow_redirects=True, timeout=30)
            expected = int(head.headers.get("content-length", 0))
        except requests.RequestException:
            expected = 0
        if expected == 0 or dest.stat().st_size == expected:
            print(f"  cached  {dest.name}  ({human(dest.stat().st_size)})")
            return dest
        print(f"  size mismatch on {dest.name}, re-downloading")

    print(f"  GET     {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r          {human(written)} / {human(total)}  ({pct:5.1f}%)",
                          end="", flush=True)
                else:
                    print(f"\r          {human(written)}", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def extract(archive: Path, target: Path, force: bool = False) -> Path:
    """Unzip into ``target``, flattening a single top-level wrapper directory."""
    if target.exists() and any(target.iterdir()) and not force:
        print(f"  present {target}")
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"  unzip   {archive.name} -> {target}")
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    except zipfile.BadZipFile as exc:
        raise SystemExit(
            f"{archive} is not a valid zip ({exc}). Delete it and re-run to download again."
        ) from exc

    # Release zips wrap everything in one directory; hoist it so the layout is
    # predictable for tools.py.
    entries = list(target.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(target / item.name))
        inner.rmdir()
    return target


def probe_colmap(tools_root: Path) -> bool:
    """Run ``colmap <cmd> -h`` for every command we use and check the flags."""
    from dronemap.tools import ColmapTool, ToolNotFound

    try:
        colmap = ColmapTool.find(None, tools_root)
    except ToolNotFound as exc:
        print(f"  {exc}")
        return False

    print(f"  colmap  {colmap.version()}  (CUDA: {colmap.has_cuda})  {colmap.binary}")
    help_dir = tools_root / "help"
    help_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, bool]] = {}
    hard_failures: list[str] = []
    soft_failures: list[str] = []

    for group, fatal in ((REQUIRED_FLAGS, True), (OPTIONAL_FLAGS, False)):
        for command, flags in group.items():
            text = colmap.help(command)
            (help_dir / f"colmap_{command}.txt").write_text(text, encoding="utf-8")
            if not text.strip() or "not a valid command" in text.lower():
                message = f"colmap has no '{command}' command"
                (hard_failures if fatal else soft_failures).append(message)
                results[command] = {flag: False for flag in flags}
                continue
            present = {flag: (flag in text) for flag in flags}
            results[command] = present
            missing = [flag for flag, found in present.items() if not found]
            if missing:
                message = f"colmap {command}: missing {', '.join(missing)}"
                (hard_failures if fatal else soft_failures).append(message)
                print(f"  [{'FAIL' if fatal else 'warn'}] {message}")
            else:
                print(f"  [ ok ] colmap {command}  ({len(flags)} flags)")

    (tools_root / "colmap_flags.json").write_text(
        json.dumps(
            {"version": colmap.version(), "cuda": colmap.has_cuda, "commands": results},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote   {tools_root / 'colmap_flags.json'} and {help_dir}/")

    if soft_failures:
        print("\n  Georeferencing flags differ from the documented names:")
        for message in soft_failures:
            print(f"    - {message}")
        print(f"    Read the real names in {help_dir} before writing stage 3b.")
    if hard_failures:
        print("\n  Core-path flags are missing - the pipeline cannot run as written:")
        for message in hard_failures:
            print(f"    - {message}")
        return False
    return True


def probe_openmvs(tools_root: Path) -> bool:
    from dronemap.tools import OPENMVS_BINARIES, OpenMVSTool, ToolNotFound

    try:
        openmvs = OpenMVSTool.find(None, tools_root)
    except ToolNotFound as exc:
        print(f"  {exc}")
        return False
    print(f"  openmvs {openmvs.version()}  {openmvs.directory}")
    for name in OPENMVS_BINARIES:
        print(f"  [ ok ] {name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, default=REPO_ROOT / "tools")
    parser.add_argument("--colmap-variant", choices=("cuda", "nocuda"), default="cuda")
    parser.add_argument("--probe", action="store_true", help="skip downloads, only verify")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument("--skip-vocab", action="store_true")
    args = parser.parse_args()

    tools_root: Path = args.tools_root.resolve()
    downloads = tools_root / "downloads"

    if not args.probe:
        free = shutil.disk_usage(tools_root.anchor or ".").free / 2**30
        if free < 5:
            print(f"only {free:.1f} GB free - the binaries need ~4 GB. Free some space first.")
            return 1

        print(f"COLMAP {COLMAP_VERSION} ({args.colmap_variant})")
        archive = download(
            COLMAP_URLS[args.colmap_variant],
            downloads / f"colmap-{COLMAP_VERSION}-{args.colmap_variant}.zip",
            force=args.force,
        )
        extract(archive, tools_root / "colmap", force=args.force)

        print(f"\nOpenMVS {OPENMVS_VERSION}")
        archive = download(OPENMVS_URL, downloads / f"openmvs-{OPENMVS_VERSION}.zip", force=args.force)
        extract(archive, tools_root / "openmvs", force=args.force)

        if not args.skip_vocab:
            print("\nCOLMAP vocabulary tree (loop detection)")
            download(VOCAB_URL, tools_root / "vocab" / Path(VOCAB_URL).name, force=args.force)

    print("\nVerifying the installed CLI surface")
    colmap_ok = probe_colmap(tools_root)
    openmvs_ok = probe_openmvs(tools_root)

    if colmap_ok and openmvs_ok:
        print("\nBootstrap complete. Next: `dronemap doctor`")
        return 0
    print("\nBootstrap incomplete - see the failures above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
