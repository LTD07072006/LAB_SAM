"""Benchmark only the two requested post-ROI 3D branches.

Both branches use the exact same strict held-out synthetic test observations:
``p_fire_noisy_pixel`` plus the estimated camera calibration.  The Ray Casting
branch intersects the metric room mesh.  The Scene-MLP branch uses the existing
calibration-aware scene-coordinate checkpoint.  No clean pixel fallback is
allowed, and no training is performed here.

The output is intentionally compatible with ``visualize_3d_results.py``.
Metrics are computed on the complete valid test set; the visualizer may later
select five records for a readable demo image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _metric(entries: list[tuple[np.ndarray, Any, float]]) -> dict[str, Any]:
    from scene_coordinate_regression import metric_summary

    valid = [(target, prediction, latency) for target, prediction, latency in entries if prediction is not None]
    if valid:
        targets = np.asarray([item[0] for item in valid], dtype=np.float64)
        predictions = np.asarray([item[1] for item in valid], dtype=np.float64)
        result = metric_summary(targets, predictions).to_dict()
    else:
        result = metric_summary(np.empty((0, 3)), np.empty((0, 3))).to_dict()
    result["attempted"] = int(len(entries))
    result["valid_rate"] = float(len(valid) / max(1, len(entries)))
    latencies = [float(item[2]) for item in entries]
    result["latency_ms"] = {
        "mean": None if not latencies else float(np.mean(latencies)),
        "median": None if not latencies else float(np.median(latencies)),
        "p95": None if not latencies else float(np.percentile(latencies, 95)),
    }
    return result


def _save_csv(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    fields = [
        "branch", "attempted", "valid", "valid_rate", "mae_m", "median_m", "p95_m",
        "xyz_mae_m", "xyz_p95_abs_m", "under_0.10m", "under_0.25m", "under_0.50m",
        "under_1.00m", "latency_mean_ms", "latency_p95_ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, metric in metrics.items():
            latency = metric.get("latency_ms", {})
            writer.writerow(
                {
                    "branch": name,
                    "attempted": metric.get("attempted"),
                    "valid": metric.get("count"),
                    "valid_rate": metric.get("valid_rate"),
                    "mae_m": metric.get("mae_m"),
                    "median_m": metric.get("median_m"),
                    "p95_m": metric.get("p95_m"),
                    "xyz_mae_m": json.dumps(metric.get("xyz_mae_m")),
                    "xyz_p95_abs_m": json.dumps(metric.get("xyz_p95_abs_m")),
                    "under_0.10m": metric.get("under_0.10m"),
                    "under_0.25m": metric.get("under_0.25m"),
                    "under_0.50m": metric.get("under_0.50m"),
                    "under_1.00m": metric.get("under_1.00m"),
                    "latency_mean_ms": latency.get("mean"),
                    "latency_p95_ms": latency.get("p95"),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from benchmark_low_cost_3d import _camera, _load_rows, _ray, _valid_samples
    from mesh_loader import load_triangle_mesh
    from scene_coordinate_regression import SceneCoordinateMLP

    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_rows = _load_rows(dataset, "test")
    samples, skipped = _valid_samples(raw_rows, args.pixel_key, args.camera_source)
    if not samples:
        raise RuntimeError(f"No valid test samples; skipped={skipped}")

    mesh_path = (args.mesh or dataset / "room_mesh.json").expanduser().resolve()
    mesh = load_triangle_mesh(mesh_path)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    mlp = SceneCoordinateMLP.load(checkpoint, device=args.device)

    entries: dict[str, list[tuple[np.ndarray, Any, float]]] = {"ray_casting": [], "scene_mlp": []}
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        ray_location = _ray(sample, mesh)
        ray_point = ray_location.get("point") if ray_location.get("hit") else None
        ray_latency = float(ray_location.get("latency_ms", 0.0))
        entries["ray_casting"].append((sample["target"], ray_point, ray_latency))

        started = time.perf_counter()
        mlp_point = mlp.predict_features(sample["features"].reshape(1, -1))[0]
        mlp_latency = (time.perf_counter() - started) * 1000.0
        entries["scene_mlp"].append((sample["target"], mlp_point, mlp_latency))

        row = sample["row"]
        records.append(
            {
                "sample_id": str(row.get("sample_id", f"test_{index:04d}")),
                "scene_id": str(sample["scene_id"]),
                "frame_index": int(row.get("frame_index", index)),
                "image_path": str((dataset / str(row.get("image_path", ""))).resolve()),
                "image_size": row.get("image_size", [640, 640]),
                "gt_pixel": row.get("p_fire_pixel"),
                "input_pixel": sample["pixel"],
                "gt_xyz": sample["target"],
                "surface": row.get("fire_surface", "unknown"),
                "camera": row.get("camera"),
                "camera_estimated": row.get("camera_estimated"),
                "branches": {
                    "ray": {
                        "pixel": sample["pixel"],
                        "location": ray_location,
                    },
                    "scene_mlp": {
                        "pixel": sample["pixel"],
                        "location": {
                            "point": mlp_point,
                            "hit": True,
                            "status": "valid_scene_mlp",
                            "method": "scene_mlp",
                            "confidence": 1.0,
                            "latency_ms": mlp_latency,
                        },
                    },
                },
            }
        )
        if index == 1 or index == len(samples) or index % 25 == 0:
            print(f"processed={index}/{len(samples)} sample={records[-1]['sample_id']}", flush=True)

    metrics = {name: _metric(values) for name, values in entries.items()}
    summary = {
        "format": "LAB_SAM.ray_vs_scene_mlp_3d_benchmark.v1",
        "protocol": {
            "test_split": "test",
            "same_test_observations": True,
            "pixel_key": args.pixel_key,
            "camera_source": args.camera_source,
            "strict_no_clean_pixel_fallback": True,
            "ray_casting": "undistort observed pixel -> 5-ray cross pattern -> room mesh intersection",
            "scene_mlp": "existing calibration-aware pixel+camera -> XYZ checkpoint",
            "warning": "Synthetic/asset-backed metric geometry; not real CCTV accuracy.",
        },
        "dataset": {
            "root": str(dataset),
            "mesh": str(mesh_path),
            "raw_test_records": len(raw_rows),
            "valid_test_samples": len(samples),
            "skipped": skipped,
            "synthetic_metric_geometry": True,
        },
        "models": {
            "scene_mlp_checkpoint": str(checkpoint),
            "scene_mlp_validation_std_m": mlp.validation_std_m,
        },
        "metrics": metrics,
        "branches": {
            "ray": metrics["ray_casting"],
            "scene_mlp": metrics["scene_mlp"],
        },
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (output / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _save_csv(output / "metrics.csv", metrics)

    print("branch       valid/attempted  MAE(m) median(m) P95(m) under0.25 under0.50 latency_ms")
    for name, metric in metrics.items():
        latency = metric["latency_ms"]
        print(
            f"{name:<13} {metric['count']:>4}/{metric['attempted']:<9} "
            f"{metric['mae_m']:.4f} {metric['median_m']:.4f} {metric['p95_m']:.4f} "
            f"{metric['under_0.25m']:.3f} {metric['under_0.50m']:.3f} {latency['mean']:.3f}"
        )
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=root / "output" / "scene_coordinate_regression_rerun_20261008" / "best_scene_coordinate_mlp.pth")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "benchmark_ray_vs_scene_mlp_20261009")
    parser.add_argument("--pixel-key", choices=("p_fire_noisy_pixel", "p_fire_pixel"), default="p_fire_noisy_pixel")
    parser.add_argument("--camera-source", choices=("estimated", "true"), default="estimated")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for field in ("dataset", "mesh", "checkpoint", "output_dir"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    run(args)


if __name__ == "__main__":
    main()
