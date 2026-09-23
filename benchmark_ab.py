"""Benchmark coarse versus ROI-refined 2D-to-3D localisation.

This runner evaluates the same detector outputs through two paths:

    A: detector point -> ray casting
    B: detector point -> ROIRefiner -> ray casting

The repository does not contain physical 3D labels. Therefore the reported
3D error is explicitly a *geometric proxy*: distance from a predicted ray hit
to the ray hit obtained from the labelled 2D point. It must not be reported as
physical metre accuracy until real 3D ground truth is supplied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_detector import FireDetector  # noqa: E402
from localization import localize_pixels, localize_with_uncertainty  # noqa: E402
from main_localization import default_calibration  # noqa: E402
from camera_calibration import CameraCalibration  # noqa: E402
from mesh_loader import load_triangle_mesh  # noqa: E402
from narrow_localizer import ROIRefinerInference  # noqa: E402
from train_week6 import load_records  # noqa: E402
from locator import GridMap  # noqa: E402


def _finite_point(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return point[:2]


def _summary(values: list[float], thresholds: tuple[float, ...] = ()) -> dict[str, Any]:
    if not values:
        result: dict[str, Any] = {"count": 0}
        for threshold in thresholds:
            result[f"under_{str(threshold).replace('.', '_')}"] = None
        return result
    array = np.asarray(values, dtype=np.float64)
    result = {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }
    for threshold in thresholds:
        result[f"under_{str(threshold).replace('.', '_')}"] = float(
            np.mean(array <= threshold)
        )
    return result


def _metric_summary(errors: list[float], total: int) -> dict[str, Any]:
    result = _summary(errors, (10.0, 25.0))
    result["coverage"] = float(len(errors) / total) if total else 0.0
    result["missing"] = int(total - len(errors))
    # Use explicit names in the report so the unit is never ambiguous.
    if "under_10_0" in result:
        result["pck10"] = result.pop("under_10_0")
    if "under_25_0" in result:
        result["pck25"] = result.pop("under_25_0")
    return result


def _json_value(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _location(camera, grid_map, pixel: Optional[np.ndarray]):
    if pixel is None:
        return None
    result = localize_pixels(
        camera,
        grid_map,
        [pixel],
        max_dist=1000.0,
        step=2.0,
        bisection_iterations=24,
    )
    if not result.hit or result.point is None:
        return {"hit": False, "status": result.status, "point": None, "spread_m": None}
    return {
        "hit": True,
        "status": result.status,
        "point": np.asarray(result.point, dtype=np.float64),
        "spread_m": float(result.spread),
    }


def _uncertainty(camera, grid_map, pixel: Optional[np.ndarray]):
    if pixel is None:
        return None
    result = localize_with_uncertainty(
        camera,
        grid_map,
        pixel,
        pixel_sigma=2.0,
        samples=128,
        seed=42,
        max_dist=1000.0,
        step=2.0,
        bisection_iterations=24,
    )
    if not result.hit or result.point is None:
        return {"hit": False, "spread_m": None, "std_m": None}
    std = None if result.std is None else float(np.linalg.norm(result.std))
    return {
        "hit": True,
        "spread_m": float(result.spread),
        "std_m": std,
        "samples": int(result.samples),
    }


def _run_branch(
    name: str,
    records,
    detector: FireDetector,
    refiner: Optional[ROIRefinerInference],
    camera,
    grid_map: GridMap,
    oracle_locations: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    coarse_errors: list[float] = []
    refined_errors: list[float] = []
    proxy_3d_errors: list[float] = []
    spreads: list[float] = []
    stds: list[float] = []
    latencies: list[float] = []
    detector_latencies: list[float] = []
    refiner_latencies: list[float] = []
    ray_latencies: list[float] = []
    detected_count = 0
    ray_hit_count = 0
    uncertainty_hit_count = 0

    for record in records:
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt_pixel = np.asarray([record.x_norm * width, record.y_norm * height], dtype=np.float64)

        start = time.perf_counter()
        detector_start = time.perf_counter()
        detection = detector.detect(image, warmup=False)
        detector_ms = (time.perf_counter() - detector_start) * 1000.0
        detector_latencies.append(detector_ms)
        coarse = _finite_point(detection.pixel) if detection.detected else None
        if coarse is not None:
            detected_count += 1

        refined = None
        refiner_ms = 0.0
        if name == "B" and coarse is not None and refiner is not None:
            refiner_start = time.perf_counter()
            refined_result = refiner.refine(image, coarse)
            refiner_ms = (time.perf_counter() - refiner_start) * 1000.0
            refiner_latencies.append(refiner_ms)
            refined = _finite_point(refined_result.point)

        used = coarse if name == "A" else refined
        ray_start = time.perf_counter()
        location = _location(camera, grid_map, used)
        uncertainty = _uncertainty(camera, grid_map, used)
        ray_ms = (time.perf_counter() - ray_start) * 1000.0
        ray_latencies.append(ray_ms)
        total_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(total_ms)

        if location is not None and location["hit"]:
            ray_hit_count += 1
        if uncertainty is not None and uncertainty["hit"]:
            uncertainty_hit_count += 1
            if uncertainty["spread_m"] is not None:
                spreads.append(float(uncertainty["spread_m"]))
            if uncertainty["std_m"] is not None:
                stds.append(float(uncertainty["std_m"]))

        gt_location = oracle_locations[str(record.image_path)]
        prediction_point = None if location is None else location.get("point")
        proxy_error = None
        if prediction_point is not None and gt_location is not None:
            proxy_error = float(np.linalg.norm(np.asarray(prediction_point) - gt_location))
            proxy_3d_errors.append(proxy_error)

        coarse_error = None if coarse is None else float(np.linalg.norm(coarse - gt_pixel))
        refined_error = None if refined is None else float(np.linalg.norm(refined - gt_pixel))
        if coarse_error is not None:
            coarse_errors.append(coarse_error)
        if refined_error is not None:
            refined_errors.append(refined_error)

        rows.append({
            "image": str(record.image_path),
            "gt_pixel": gt_pixel,
            "coarse_pixel": coarse,
            "refined_pixel": refined,
            "used_pixel": used,
            "confidence": float(detection.confidence),
            "detected": bool(detection.detected),
            "coarse_error_px": coarse_error,
            "refined_error_px": refined_error,
            "location": location,
            "oracle_2d_location": gt_location,
            "proxy_3d_error_m": proxy_error,
            "uncertainty": uncertainty,
            "latency_ms": total_ms,
            "detector_ms": detector_ms,
            "refiner_ms": refiner_ms,
            "ray_and_uncertainty_ms": ray_ms,
        })

    total = len(records)
    summary = {
        "branch": name,
        "images": total,
        "detected": detected_count,
        "detector_recall_on_gt_fire": float(detected_count / total) if total else 0.0,
        "ray_hits": ray_hit_count,
        "ray_hit_rate_all_images": float(ray_hit_count / total) if total else 0.0,
        "ray_hit_rate_given_detection": float(ray_hit_count / detected_count) if detected_count else 0.0,
        "uncertainty_hits": uncertainty_hit_count,
        "pixel_error_coarse_px": _metric_summary(coarse_errors, total),
        "pixel_error_refined_px": _metric_summary(refined_errors, total),
        "proxy_3d_error_m_vs_oracle_2d": _summary(proxy_3d_errors, (0.5, 1.0, 2.0)),
        "uncertainty_spread_m": _summary(spreads),
        "uncertainty_std_norm_m": _summary(stds),
        "latency_ms": {
            "end_to_end": _summary(latencies),
            "detector": _summary(detector_latencies),
            "refiner": _summary(refiner_latencies),
            "ray_and_uncertainty": _summary(ray_latencies),
        },
        "rows": rows,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "working" / "benchmark_ab_test_fire.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    args = parser.parse_args()

    labels_path = ROOT / "fire-model-data" / "dataset_labels (1).json"
    dataset_root = ROOT / "fire-detection-from-cctv"
    detector_path = ROOT / "fire-model-data" / "best.pth"
    roi_path = ROOT / "week6_roi_result" / "best_roi.pth"
    records, load_stats = load_records(labels_path, dataset_root)
    test_fire = [record for record in records if record.source_split == "test" and record.has_fire]
    test_fire.sort(key=lambda record: str(record.image_path))
    if not test_fire:
        raise RuntimeError("No positive records found in source test split")

    first_size = Image.open(test_fire[0].image_path).size
    if any(Image.open(record.image_path).size != first_size for record in test_fire):
        raise RuntimeError("Test-fire images do not share one image size")
    calibration = CameraCalibration.from_json(args.calibration) if args.calibration else default_calibration(first_size)
    camera = calibration.geometry()
    grid_map = load_triangle_mesh(args.mesh) if args.mesh else GridMap()

    detector = FireDetector(detector_path, device=args.device, threshold=args.threshold, use_amp=False)
    refiner = ROIRefinerInference(roi_path, device=args.device)
    # Warm up before timing so model construction/first-kernel overhead does
    # not dominate the per-image latency summary.
    detector.warmup(repeats=1)
    warm_image = Image.open(test_fire[0].image_path).convert("RGB")
    warm_detection = detector.detect(warm_image, warmup=False)
    if warm_detection.pixel is not None:
        refiner.refine(warm_image, warm_detection.pixel)

    oracle_locations: dict[str, Any] = {}
    for record in test_fire:
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt_pixel = np.asarray([record.x_norm * width, record.y_norm * height], dtype=np.float64)
        oracle = _location(camera, grid_map, gt_pixel)
        oracle_locations[str(record.image_path)] = None if oracle is None or not oracle["hit"] else oracle["point"]

    branch_a = _run_branch("A", test_fire, detector, None, camera, grid_map, oracle_locations)
    branch_b = _run_branch("B", test_fire, detector, refiner, camera, grid_map, oracle_locations)

    paired_coarse: list[float] = []
    paired_refined: list[float] = []
    for row_a, row_b in zip(branch_a["rows"], branch_b["rows"]):
        if row_a["coarse_error_px"] is not None and row_b["refined_error_px"] is not None:
            paired_coarse.append(float(row_a["coarse_error_px"]))
            paired_refined.append(float(row_b["refined_error_px"]))
    paired = {
        "count": len(paired_coarse),
        "coarse_mean_px": float(np.mean(paired_coarse)) if paired_coarse else None,
        "refined_mean_px": float(np.mean(paired_refined)) if paired_refined else None,
        "mean_improvement_px": float(np.mean(np.asarray(paired_coarse) - np.asarray(paired_refined))) if paired_coarse else None,
        "improved_fraction": float(np.mean(np.asarray(paired_refined) < np.asarray(paired_coarse))) if paired_coarse else None,
    }

    report = {
        "protocol": {
            "dataset": "source test/fire only",
            "images": len(test_fire),
            "image_size": list(first_size),
            "calibration": str(args.calibration) if args.calibration else "synthetic default_calibration",
            "surface": str(args.mesh) if args.mesh else "analytic GridMap()",
            "physical_3d_ground_truth": False,
            "proxy_definition": "predicted ray-hit distance to ray-hit from labelled 2D point",
            "latency_definition": "model inference + ROI refinement + ray casting/uncertainty; image I/O excluded",
        },
        "load_stats": dict(load_stats),
        "paired_pixel_comparison": paired,
        "branch_A_coarse": branch_a,
        "branch_B_refined": branch_b,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_value(report), indent=2), encoding="utf-8")

    print(json.dumps(_json_value({
        "output": args.output,
        "protocol": report["protocol"],
        "paired_pixel_comparison": paired,
        "A": {key: value for key, value in branch_a.items() if key != "rows"},
        "B": {key: value for key, value in branch_b.items() if key != "rows"},
    }), indent=2))


if __name__ == "__main__":
    main()
