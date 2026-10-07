"""Benchmark 2D point and camera noise on one fixed synthetic test split.

This script deliberately does not regenerate images. Every noise profile uses
the same metric fire points, camera poses and mesh, so the difference between
profiles is attributable to the injected error rather than to a different
random scene. It is the fast robustness benchmark for the synthetic 3D branch.

Example::

    python benchmark_synthetic_noise.py \
        --dataset working/synthetic_fire_3d_v3 \
        --split test \
        --point-noise-px 0 1 3 5 10 \
        --calibration-scale 0 1 2 \
        --output working/synthetic_fire_3d_v3/noise_benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from camera_calibration import CameraCalibration
from locator import intersect_ray_with_grid_result
from mesh_loader import load_triangle_mesh
from synthetic_fire_3d import perturb_camera


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def calibration_from_camera(row: dict[str, Any], camera: dict[str, Any]) -> CameraCalibration:
    return CameraCalibration.from_dict(
        {
            "image_size": row["image_size"],
            "intrinsics": {"K": camera["K"], "dist_coeffs": camera.get("dist_coeffs", [])},
            "extrinsics": {
                "R": camera["R_world_to_camera"],
                "camera_position": camera["camera_position"],
            },
        }
    )


def _percentile_or_none(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, percentile)) if len(array) else None


def evaluate_profile(
    rows: list[dict[str, Any]],
    mesh: Any,
    point_noise_px: float,
    calibration_scale: float,
    seed: int,
    max_dist: float,
) -> dict[str, Any]:
    """Evaluate one point/calibration noise combination."""
    errors: list[float] = []
    pixel_errors: list[float] = []
    axis_errors: list[np.ndarray] = []
    statuses: dict[str, int] = {}
    attempts = 0
    hits = 0

    for row_index, row in enumerate(rows):
        if int(row.get("has_fire", 0)) != 1:
            continue
        clean_pixel = row.get("p_fire_pixel")
        truth = row.get("fire_xyz_world")
        if not clean_pixel or not truth:
            continue
        attempts += 1
        # Re-seed per record so every profile sees the same standard-normal
        # draws. Only the requested sigma/scale changes; scene and random
        # direction are therefore held constant across the table.
        rng = np.random.default_rng(int(seed) + row_index * 1009)
        camera = row["camera"]
        noisy_pixel = np.asarray(clean_pixel, dtype=np.float64) + rng.normal(
            0.0, max(0.0, float(point_noise_px)), size=2
        )
        width, height = [int(item) for item in row["image_size"]]
        noisy_pixel[0] = np.clip(noisy_pixel[0], 0.0, width - 1)
        noisy_pixel[1] = np.clip(noisy_pixel[1], 0.0, height - 1)
        pixel_errors.append(float(np.linalg.norm(noisy_pixel - np.asarray(clean_pixel, dtype=np.float64))))

        if calibration_scale > 0.0:
            K, R, position = perturb_camera(
                np.asarray(camera["K"], dtype=np.float64),
                np.asarray(camera["R_world_to_camera"], dtype=np.float64),
                np.asarray(camera["camera_position"], dtype=np.float64),
                rng,
                focal_noise_pct=0.01 * float(calibration_scale),
                principal_noise_px=1.5 * float(calibration_scale),
                rotation_noise_deg=0.25 * float(calibration_scale),
                position_noise_m=0.02 * float(calibration_scale),
            )
            estimated_camera = {
                "K": K,
                "R_world_to_camera": R,
                "camera_position": position,
                "dist_coeffs": camera.get("dist_coeffs", []),
            }
        else:
            estimated_camera = camera

        calibration = calibration_from_camera(row, estimated_camera)
        origin, ray = calibration.geometry().pixel_to_ray(float(noisy_pixel[0]), float(noisy_pixel[1]))
        result = intersect_ray_with_grid_result(origin, ray, mesh, max_dist=float(max_dist))
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if not result.hit or result.point is None:
            continue
        hits += 1
        delta = np.asarray(result.point, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        axis_errors.append(np.abs(delta))
        errors.append(float(np.linalg.norm(delta)))

    values = np.asarray(errors, dtype=np.float64)
    axis = np.asarray(axis_errors, dtype=np.float64) if axis_errors else np.empty((0, 3))
    return {
        "point_noise_px": float(point_noise_px),
        "calibration_scale": float(calibration_scale),
        "attempted": attempts,
        "ray_hits": hits,
        "ray_hit_rate": float(hits / attempts) if attempts else 0.0,
        "pixel_mae_px": float(np.mean(pixel_errors)) if pixel_errors else None,
        "mae_m": float(values.mean()) if len(values) else None,
        "median_m": float(np.median(values)) if len(values) else None,
        "p95_m": _percentile_or_none(values, 95),
        "max_m": float(values.max()) if len(values) else None,
        "under_0.10m": float(np.mean(values <= 0.10)) if len(values) else None,
        "under_0.25m": float(np.mean(values <= 0.25)) if len(values) else None,
        "under_0.50m": float(np.mean(values <= 0.50)) if len(values) else None,
        "under_1.00m": float(np.mean(values <= 1.00)) if len(values) else None,
        "mean_abs_xyz_m": axis.mean(axis=0).tolist() if len(axis) else None,
        "p95_abs_xyz_m": np.percentile(axis, 95, axis=0).tolist() if len(axis) else None,
        "statuses": statuses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--point-noise-px", type=float, nargs="+", default=[0, 1, 3, 5, 10])
    parser.add_argument("--calibration-scale", type=float, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-dist", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    rows = load_jsonl(dataset / "manifest.jsonl")
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    mesh = load_triangle_mesh(dataset / "room_mesh.json")
    profiles = []
    for calibration_scale in args.calibration_scale:
        for point_noise_px in args.point_noise_px:
            profiles.append(
                evaluate_profile(
                    rows,
                    mesh,
                    point_noise_px=float(point_noise_px),
                    calibration_scale=float(calibration_scale),
                    seed=int(args.seed),
                    max_dist=float(args.max_dist),
                )
            )
    result = {
        "dataset": str(dataset),
        "split": args.split,
        "records": len(rows),
        "visible_fire_records": sum(int(row.get("has_fire", 0)) for row in rows),
        "physical_fire_events": sum(int(row.get("fire_event", row.get("has_fire", 0))) for row in rows),
        "occluded_fire_events": sum(
            int(row.get("fire_event", 0) == 1 and row.get("fire_visible", 1) == 0) for row in rows
        ),
        "profiles": profiles,
        "notes": [
            "All profiles use the same fixed manifest and mesh.",
            "calibration_scale=1 matches the generator default calibration perturbation.",
            "Occluded physical events are excluded from 2D-point attempts by design.",
        ],
    }
    output = args.output or dataset / f"noise_benchmark_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
