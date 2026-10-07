"""Evaluate synthetic fire 2D->3D ray casting with clean/noisy points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from camera_calibration import CameraCalibration
from locator import TriangleMesh, intersect_ray_with_grid_result
from mesh_loader import load_triangle_mesh


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_field(
    rows: list[dict[str, Any]],
    mesh: TriangleMesh,
    field: str,
    max_dist: float,
    camera_field: str = "camera",
) -> dict[str, Any]:
    errors = []
    axis_errors = []
    pixel_errors = []
    hits = 0
    attempted = 0
    statuses: dict[str, int] = {}
    physical_events = sum(int(row.get("fire_event", row.get("has_fire", 0))) for row in rows)
    visible_events = sum(int(row.get("fire_visible", row.get("has_fire", 0))) for row in rows)
    occluded_events = sum(
        int(row.get("fire_event", 0) == 1 and row.get("fire_visible", 1) == 0) for row in rows
    )
    missing_observations = 0
    for row in rows:
        if int(row.get("has_fire", 0)) != 1 or not row.get("fire_xyz_world"):
            continue
        pixel = row.get(field)
        if not pixel:
            missing_observations += 1
            continue
        attempted += 1
        camera = row.get(camera_field) or row["camera"]
        calibration = CameraCalibration.from_dict(
            {
                "image_size": row["image_size"],
                "intrinsics": {"K": camera["K"], "dist_coeffs": camera.get("dist_coeffs", [])},
                "extrinsics": {
                    "R": camera["R_world_to_camera"],
                    "camera_position": camera["camera_position"],
                },
            }
        )
        origin, ray = calibration.geometry().pixel_to_ray(float(pixel[0]), float(pixel[1]))
        result = intersect_ray_with_grid_result(origin, ray, mesh, max_dist=max_dist)
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if not result.hit or result.point is None:
            continue
        hits += 1
        truth = np.asarray(row["fire_xyz_world"], dtype=np.float64)
        delta = np.asarray(result.point, dtype=np.float64) - truth
        errors.append(float(np.linalg.norm(delta)))
        axis_errors.append(np.abs(delta))
        clean_pixel = row.get("p_fire_pixel")
        if clean_pixel:
            pixel_errors.append(float(np.linalg.norm(np.asarray(pixel, dtype=np.float64) - clean_pixel)))

    values = np.asarray(errors, dtype=np.float64)
    if len(values):
        metrics = {
            "mae_m": float(values.mean()),
            "median_m": float(np.median(values)),
            "p95_m": float(np.percentile(values, 95)),
            "under_0.05m": float(np.mean(values <= 0.05)),
            "under_0.10m": float(np.mean(values <= 0.10)),
            "under_0.25m": float(np.mean(values <= 0.25)),
            "under_0.50m": float(np.mean(values <= 0.50)),
            "under_1.00m": float(np.mean(values <= 1.00)),
            "max_m": float(values.max()),
        }
        axis_values = np.asarray(axis_errors, dtype=np.float64)
        metrics.update(
            {
                "mean_abs_x_m": float(axis_values[:, 0].mean()),
                "mean_abs_y_m": float(axis_values[:, 1].mean()),
                "mean_abs_z_m": float(axis_values[:, 2].mean()),
                "p95_abs_x_m": float(np.percentile(axis_values[:, 0], 95)),
                "p95_abs_y_m": float(np.percentile(axis_values[:, 1], 95)),
                "p95_abs_z_m": float(np.percentile(axis_values[:, 2], 95)),
                "pixel_mae_px": float(np.mean(pixel_errors)) if pixel_errors else None,
            }
        )
    else:
        metrics = {
            "mae_m": None,
            "median_m": None,
            "p95_m": None,
            "under_0.05m": None,
            "under_0.10m": None,
            "under_0.25m": None,
            "under_0.50m": None,
            "under_1.00m": None,
            "max_m": None,
            "mean_abs_x_m": None,
            "mean_abs_y_m": None,
            "mean_abs_z_m": None,
            "p95_abs_x_m": None,
            "p95_abs_y_m": None,
            "p95_abs_z_m": None,
            "pixel_mae_px": None,
        }
    return {
        "field": field,
        "camera_field": camera_field,
        "attempted": attempted,
        "ray_hits": hits,
        "hit_rate": float(hits / attempted) if attempted else 0.0,
        "physical_fire_events": physical_events,
        "visible_fire_events": visible_events,
        "occluded_fire_events": occluded_events,
        "missing_observations": missing_observations,
        "statuses": statuses,
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-dist", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    rows = load_jsonl(dataset / "manifest.jsonl")
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    mesh = load_triangle_mesh(dataset / "room_mesh.json")
    result = {
        "dataset": str(dataset),
        "split": args.split,
        "records": len(rows),
        "physical_fire_events": sum(int(row.get("fire_event", row.get("has_fire", 0))) for row in rows),
        "visible_fire_events": sum(int(row.get("fire_visible", row.get("has_fire", 0))) for row in rows),
        "occluded_fire_events": sum(
            int(row.get("fire_event", 0) == 1 and row.get("fire_visible", 1) == 0) for row in rows
        ),
        "clean_true_camera": evaluate_field(rows, mesh, "p_fire_pixel", float(args.max_dist), "camera"),
        "noisy_point_true_camera": evaluate_field(
            rows, mesh, "p_fire_noisy_pixel", float(args.max_dist), "camera"
        ),
        "clean_point_estimated_camera": evaluate_field(
            rows, mesh, "p_fire_pixel", float(args.max_dist), "camera_estimated"
        ),
        "noisy_point_estimated_camera": evaluate_field(
            rows, mesh, "p_fire_noisy_pixel", float(args.max_dist), "camera_estimated"
        ),
    }
    output = args.output or (dataset / f"evaluation_{args.split}.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
