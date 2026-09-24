"""
Aggregates photogrammetry metrics across all reconstructed runs for the SIH26158 judging dossier.
Outputs:
- data/benchmark_summary.json
- Console Markdown table for presentation slides / pitch deck
"""

import json
from pathlib import Path
import numpy as np

DATA_DIR = Path("data").resolve()
RUNS_DIR = DATA_DIR / "runs"

def get_run_metrics(run_dir: Path) -> dict | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None

    stages = manifest.get("stages", {})
    accuracy = manifest.get("accuracy", {})
    gnss = manifest.get("gnss", {})

    # Keyframes
    n_keyframes = stages.get("frames", {}).get("metrics", {}).get("n_keyframes", 0)
    # Dense points
    n_dense = stages.get("dense", {}).get("metrics", {}).get("n_dense_points", 0)
    # Mesh
    mesh_metrics = stages.get("mesh", {}).get("metrics", {})
    n_vertices = mesh_metrics.get("n_vertices", 0)
    n_faces = mesh_metrics.get("n_faces", 0)
    # Export
    export_metrics = stages.get("export", {}).get("metrics", {})
    gsd_m = export_metrics.get("gsd_m")
    crs = export_metrics.get("crs") or accuracy.get("crs", "N/A")
    gnss_mode = gnss.get("mode", "NONE")

    # Check deliverables
    export_dir = run_dir / "07_export"
    deliverables = {
        "glb": (export_dir / "model.glb").exists(),
        "laz": (export_dir / "cloud.laz").exists(),
        "dsm": (export_dir / "dsm.tif").exists(),
        "dtm": (export_dir / "dtm.tif").exists(),
        "orthomosaic": (export_dir / "orthomosaic.tif").exists(),
        "kml": (export_dir / "trajectory.kml").exists(),
        "report_html": (export_dir / "report.html").exists(),
    }

    # Total duration
    total_sec = sum(
        v.get("duration_s", 0) or 0
        for v in stages.values()
        if isinstance(v, dict) and v.get("duration_s") is not None
    )

    return {
        "run_id": run_dir.name,
        "created_utc": manifest.get("created_utc"),
        "gnss_mode": gnss_mode,
        "crs": crs,
        "n_keyframes": n_keyframes,
        "n_dense_points": n_dense,
        "n_mesh_vertices": n_vertices,
        "n_mesh_faces": n_faces,
        "gsd_cm_px": round(gsd_m * 100, 2) if gsd_m else None,
        "total_runtime_s": round(total_sec, 1),
        "deliverables_complete": all(deliverables.values()),
        "deliverables": deliverables
    }

def main():
    print("===============================================================")
    print("[*] AGGREGATING BENCHMARK METRICS ACROSS ALL SURVEY RUNS")
    print("===============================================================\n")

    target_runs = [
        "synthetic_eval_run",
        "demo_aukerman_hd",
        "demo_aukerman",
        "run_My_most_beautifu_20260902_184229"
    ]

    all_summaries = []
    for rid in target_runs:
        r_dir = RUNS_DIR / rid
        if r_dir.exists():
            data = get_run_metrics(r_dir)
            if data:
                all_summaries.append(data)

    out_json = DATA_DIR / "benchmark_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("| Run ID | Keyframes | Dense Pts | Faces | GSD (cm/px) | GNSS / CRS | Runtime | All 7 Deliverables |")
    print("| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |")
    for r in all_summaries:
        deliv_str = "[OK] Complete" if r["deliverables_complete"] else "Partial"
        print(f"| **{r['run_id']}** | {r['n_keyframes']} | {r['n_dense_points']:,} | {r['n_mesh_faces']:,} | {r['gsd_cm_px']} | {r['gnss_mode']} ({r['crs']}) | {r['total_runtime_s']}s | {deliv_str} |")

    print(f"\n[OK] Benchmark Summary Saved to: {out_json}")

if __name__ == "__main__":
    main()
