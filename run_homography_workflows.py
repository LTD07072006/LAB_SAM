"""Run the post-ROI homography/IPM experiments without changing the legacy flow.

The project has two complementary benchmark scripts:

* ``benchmark_homography.py`` compares floor IPM/DLT/RANSAC with ray casting;
* ``benchmark_multiplane_homography.py`` compares floor and multi-plane
  homographies, visibility selection and EMA smoothing.

This entry point runs both scripts on the metric datasets already generated in
``working/`` and writes a compact, machine-readable report.  It deliberately
uses subprocesses instead of importing benchmark internals, so each benchmark
keeps its own command-line contract and optional dependencies.  Existing
datasets, checkpoints and output directories are never modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset: Path
    mesh: Optional[Path] = None
    prediction_summary: Optional[Path] = None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serialisable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def resolve_path(root: Path, value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    path = value.expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def non_empty(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def command_for_multiplane(
    root: Path,
    spec: DatasetSpec,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(args.python),
        str(root / "benchmark_multiplane_homography.py"),
        "--dataset",
        str(spec.dataset),
        "--split",
        args.split,
        "--selection",
        args.selection,
        "--max-records",
        str(args.max_records),
        "--max-visual-records",
        str(args.max_visual_records),
        "--max-contact-images",
        str(args.max_contact_images),
        "--output-dir",
        str(output),
    ]
    if spec.mesh is not None:
        command.extend(["--mesh", str(spec.mesh)])
    if spec.prediction_summary is not None:
        command.extend(["--prediction-summary", str(spec.prediction_summary)])
    return command


def command_for_floor(
    root: Path,
    spec: DatasetSpec,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(args.python),
        str(root / "benchmark_homography.py"),
        "--dataset",
        str(spec.dataset),
        "--split",
        args.split,
        "--selection",
        args.selection,
        "--max-records",
        str(args.max_records),
        "--max-visual-records",
        str(args.max_visual_records),
        "--max-contact-images",
        str(args.max_contact_images),
        "--output-dir",
        str(output),
    ]
    if spec.mesh is not None:
        command.extend(["--mesh", str(spec.mesh)])
    if spec.prediction_summary is not None:
        command.extend(["--prediction-summary", str(spec.prediction_summary)])
    return command


def run_command(command: list[str], cwd: Path) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def finite_metric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def selected_metric(summary: dict[str, Any], key: str) -> dict[str, Any]:
    # ``benchmark_multiplane_homography.py`` keeps its metrics at the top level,
    # while ``benchmark_homography.py`` nests the same table under
    # ``diagnostics.metrics``.  Accept both layouts so the aggregate report does
    # not silently turn the floor-ray rows into missing values.
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        diagnostics = summary.get("diagnostics", {})
        metrics = diagnostics.get("metrics", {}) if isinstance(diagnostics, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    preferred_sources = ["roi_blend", "coarse", "oracle"]
    first_available: Optional[dict[str, Any]] = None
    for source in preferred_sources:
        candidate = metrics.get(f"{source}:{key}")
        if not isinstance(candidate, dict):
            continue
        # The multi-plane benchmark stores all-surface metrics directly.  The
        # floor benchmark stores floor-only and all-surface metrics separately;
        # for that protocol the aggregate must use ``floor_only`` so object
        # contacts do not contaminate the floor IPM/ray comparison.
        if "plane_extraction" not in summary and "floor_only" in candidate:
            selected = candidate["floor_only"]
        elif "all_surfaces" in candidate:
            selected = candidate["all_surfaces"]
        elif "floor_only" in candidate:
            selected = candidate["floor_only"]
        else:
            selected = candidate
        if not isinstance(selected, dict):
            continue
        row = {"source": source, **selected}
        if first_available is None:
            first_available = row
        # Prefer the requested source only when it produced at least one valid
        # estimate.  This matters for asset-backed datasets that have no ROI
        # prediction summary: an empty roi_blend row must fall back to coarse.
        valid = selected.get("valid")
        if isinstance(valid, (int, float)) and valid > 0:
            return row
    if first_available is not None:
        return first_available
    return {"source": None, "mae_m": None, "median_m": None, "p95_m": None}


def compact_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    multi_keys = (
        "floor_ipm",
        "plane_bank_nearest",
        "plane_bank_visible",
        "plane_bank_visible_floor_fallback",
        "floor_ipm_ema",
        "plane_bank_visible_ema",
        "plane_bank_visible_floor_fallback_ema",
    )
    floor_keys = ("analytic", "dlt_all", "ransac_noisy", "ray", "ray_floor")
    is_floor_protocol = "plane_extraction" not in summary
    keys = multi_keys if not is_floor_protocol else floor_keys
    metrics: dict[str, Any] = {}
    for key in keys:
        metric = selected_metric(summary, key)
        metrics[key] = {
            field: metric.get(field)
            for field in (
                "source",
                "records",
                "valid",
                "success_rate",
                "mae_m",
                "median_m",
                "p95_m",
                "under_0.10m",
                "under_0.25m",
                "under_0.50m",
                "under_1.00m",
                "xyz_mae_m",
                "mean_latency_ms",
                "p95_latency_ms",
            )
            if field in metric
        }
    return {
        "summary": str(path),
        "dataset": summary.get("dataset"),
        "split": summary.get("split"),
        "records": summary.get("records"),
        "prediction_summary": summary.get("prediction_summary"),
        "metrics": metrics,
        "visual_outputs": summary.get("visual_outputs", {}),
    }


def markdown_report(
    path: Path,
    reports: Iterable[dict[str, Any]],
    output_root: Path,
) -> None:
    lines = [
        "# Post-ROI homography/IPM benchmark",
        "",
        "This report is generated by `run_homography_workflows.py`.",
        "It compares floor IPM and multi-plane homography against calibrated ray",
        "casting on metric synthetic/asset-backed scenes. It is not a real CCTV",
        "accuracy claim.",
        "",
        f"Output root: `{output_root}`",
        "",
    ]
    for report in reports:
        dataset = Path(str(report.get("dataset", "unknown"))).name
        lines.extend([f"## {dataset}", "", "| Method | Source | Valid | MAE (m) | Median (m) | P95 (m) |", "|---|---|---:|---:|---:|---:|"])
        for method, metric in report.get("metrics", {}).items():
            valid = f"{metric.get('valid', '—')}/{metric.get('records', '—')}"
            def fmt(field: str) -> str:
                value = finite_metric(metric.get(field))
                return "—" if value is None else f"{value:.4f}"
            lines.append(
                f"| `{method}` | {metric.get('source') or '—'} | {valid} | "
                f"{fmt('mae_m')} | {fmt('median_m')} | {fmt('p95_m')} |"
            )
        lines.extend([
            "",
            "Visual outputs:",
            "",
        ])
        for label, value in report.get("visual_outputs", {}).items():
            lines.append(f"- `{label}`: `{value}`")
        lines.append("")
    lines.extend([
        "## Deployment policy",
        "",
        "1. Use floor IPM when the contact point is known to be on the calibrated floor.",
        "2. Use a plane-specific homography only when the plane identity is selected with sufficient confidence.",
        "3. Keep ray casting as the fallback for unknown/non-planar contacts.",
        "4. Apply EMA/EKF after validity and jump gating; smoothing cannot repair a wrong plane choice.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_specs(root: Path, prediction_summary: Optional[Path]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    synthetic = root / "working" / "synthetic_fire_3d_v3"
    if synthetic.is_dir():
        specs.append(DatasetSpec("synthetic_v3", synthetic, synthetic / "room_mesh.json", prediction_summary))
    replicacad = root / "working" / "asset_backed_replicacad_fire_3d_fast"
    if replicacad.is_dir():
        specs.append(DatasetSpec("replicacad", replicacad, replicacad / "room_mesh.json", None))
    return specs


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output") / f"homography_workflows_{today}")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--selection", choices=("head", "even"), default="even")
    parser.add_argument("--max-records", type=int, default=0, help="0 means the complete selected split")
    parser.add_argument("--max-visual-records", type=int, default=40)
    parser.add_argument("--max-contact-images", type=int, default=16)
    parser.add_argument("--prediction-summary", type=Path, default=Path("output/workflow_roi_cpu_regularized_test_full_20261007/summary.json"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--skip-floor", action="store_true", help="Only run the multi-plane benchmark")
    parser.add_argument("--skip-multiplane", action="store_true", help="Only run the floor benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    args.output_root = resolve_path(root, args.output_root)
    args.python = resolve_path(root, args.python)
    args.prediction_summary = resolve_path(root, args.prediction_summary)
    assert args.output_root is not None
    assert args.python is not None
    if args.max_records < 0 or args.max_visual_records < 0 or args.max_contact_images < 0:
        raise ValueError("max values must be non-negative")
    if args.skip_floor and args.skip_multiplane:
        raise ValueError("Cannot skip both benchmark families")
    args.output_root.mkdir(parents=True, exist_ok=True)

    prediction = args.prediction_summary if args.prediction_summary and args.prediction_summary.is_file() else None
    if prediction is None:
        print("prediction_summary=not_found; synthetic run will use manifest noisy/coarse pixels")
    specs = default_specs(root, prediction)
    if not specs:
        raise FileNotFoundError("No default metric dataset found under working/")

    reports: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.dataset.is_dir():
            raise FileNotFoundError(f"Dataset not found: {spec.dataset}")
        if spec.mesh is not None and not spec.mesh.is_file():
            raise FileNotFoundError(f"Mesh not found: {spec.mesh}")
        if not args.skip_multiplane:
            output = args.output_root / f"{spec.name}_multiplane"
            run_command(command_for_multiplane(root, spec, output, args), root)
            reports.append(compact_summary(output / "summary.json"))
        if not args.skip_floor:
            output = args.output_root / f"{spec.name}_floor_ray"
            run_command(command_for_floor(root, spec, output, args), root)
            reports.append(compact_summary(output / "summary.json"))

    combined = {
        "format": "LAB_SAM.run_homography_workflows.v1",
        "output_root": str(args.output_root),
        "split": args.split,
        "selection": args.selection,
        "max_records": args.max_records,
        "reports": reports,
        "notes": [
            "Synthetic and ReplicaCAD values validate metric geometry only.",
            "surface_oracle is intentionally excluded from the compact deployment table.",
            "Homography is valid only on the selected plane; ray casting remains the non-planar fallback.",
        ],
    }
    write_json(args.output_root / "combined_summary.json", combined)
    markdown_report(args.output_root / "combined_report.md", reports, args.output_root)
    print(f"saved_combined_summary={args.output_root / 'combined_summary.json'}")
    print(f"saved_report={args.output_root / 'combined_report.md'}")


if __name__ == "__main__":
    main()
