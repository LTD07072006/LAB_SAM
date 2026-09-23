"""Evaluation harness for the 2D-to-3D localisation contribution.

The repository currently has only a small manual 2D ground-truth set and no
physical 3D ground-truth file. Therefore this script reports pixel/refiner
metrics and, when a 3D label is supplied, honest 3D errors. It supports:

* oracle: ground-truth 2D point -> calibrated ray casting;
* noisy: ground-truth point + controlled detector noise -> ray casting;
* detector: upstream detector -> ROI refiner -> calibrated ray casting.

It never calls a 3D error metric against a synthetic ray-casting output as if
that were a physical ground truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from camera_calibration import CameraCalibration
from config import DEFAULT_CONFIG
from detector_adapter import adapt_local_detector_result
from localization import localize_pixels
from locator import GridMap
from main_localization import Fire3DLocalizationPipeline, default_calibration


def _load_points(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): (float(value[0]), float(value[1])) for key, value in raw.items()}


def _find_point(points: dict[str, tuple[float, float]], image_path: Path):
    if image_path.name in points:
        return points[image_path.name]
    return points.get(str(image_path))


def _load_3d(path: Optional[Path]) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): np.asarray(value, dtype=np.float64).reshape(3) for key, value in raw.items()}


def _error_metrics(errors: list[np.ndarray]) -> dict[str, Any]:
    if not errors:
        return {"count": 0}
    arr = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    norm = np.linalg.norm(arr, axis=1)
    return {
        "count": int(len(arr)),
        "mean_m": float(norm.mean()),
        "median_m": float(np.median(norm)),
        "p95_m": float(np.percentile(norm, 95)),
        "mean_abs_xyz_m": np.mean(np.abs(arr), axis=0).tolist(),
        "under_0_5m": float(np.mean(norm <= 0.5)),
        "under_1m": float(np.mean(norm <= 1.0)),
        "under_2m": float(np.mean(norm <= 2.0)),
    }


def evaluate(args):
    points = _load_points(args.points)
    labels_3d = _load_3d(args.labels_3d)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.labeled_only:
        paths = [path for path in paths if _find_point(points, path) is not None]
    if args.image_limit:
        paths = paths[:args.image_limit]
    if not paths:
        raise FileNotFoundError(f"No images in {args.images}")
    first_size = Image.open(paths[0]).size
    calibration = CameraCalibration.from_json(args.calibration) if args.calibration else default_calibration(first_size)
    geometry = calibration.geometry()
    grid_map = GridMap()

    detector = None
    pipeline = None
    if args.mode == "detector":
        from fire_detector import FireDetector
        detector = FireDetector(args.model, device=args.device, threshold=args.threshold)
        refiner = None
        if args.roi_checkpoint and args.roi_checkpoint.is_file():
            from narrow_localizer import ROIRefinerInference
            refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device)
        pipeline = Fire3DLocalizationPipeline(
            detector, calibration, grid_map=grid_map, refiner=refiner,
            threshold=args.threshold, use_uncertainty=False,
        )

    rows, errors, pixel_errors = [], [], []
    rng = np.random.default_rng(args.seed)
    for image_path in paths:
        gt_pixel = _find_point(points, image_path)
        if gt_pixel is None:
            continue
        image = Image.open(image_path).convert("RGB")
        start = time.perf_counter()
        if args.mode == "oracle":
            used_pixel = np.asarray(gt_pixel, dtype=np.float64)
            result = localize_pixels(geometry, grid_map, [used_pixel],
                                     max_dist=args.max_dist, step=args.step)
            detector_point = None
        elif args.mode == "noisy":
            used_pixel = np.asarray(gt_pixel, dtype=np.float64) + rng.normal(0.0, args.pixel_sigma, 2)
            result = localize_pixels(geometry, grid_map, [used_pixel],
                                     max_dist=args.max_dist, step=args.step)
            detector_point = None
        else:
            output = pipeline.process(image)
            used_pixel = None if output["refined"] is None else np.asarray(output["refined"]["point"], dtype=np.float64)
            detector_point = output["detection"].get("point")
            result = None
            if used_pixel is not None:
                pixel_errors.append(used_pixel - np.asarray(gt_pixel))
        latency = (time.perf_counter() - start) * 1000.0
        row = {
            "image": str(image_path), "gt_pixel": gt_pixel,
            "used_pixel": None if used_pixel is None else used_pixel.tolist(),
            "detector_pixel": detector_point, "latency_ms": latency,
        }
        if result is not None:
            row["location"] = None if not result.hit else result.point.tolist()
        if args.mode == "detector" and output["location"] is not None:
            point = output["location"]["point"]
            row["location"] = None if point is None else np.asarray(point, dtype=np.float64).tolist()
        gt_3d = labels_3d.get(image_path.name, labels_3d.get(str(image_path)))
        predicted_3d = row.get("location")
        if gt_3d is not None and predicted_3d is not None:
            delta = np.asarray(predicted_3d, dtype=np.float64) - gt_3d
            errors.append(delta)
            row["error_3d_m"] = float(np.linalg.norm(delta))
        rows.append(row)

    summary = {
        "mode": args.mode, "images_seen": len(paths), "images_evaluated": len(rows),
        "pixel_error_px": None if not pixel_errors else {
            "mean": float(np.linalg.norm(pixel_errors, axis=1).mean()),
            "median": float(np.median(np.linalg.norm(pixel_errors, axis=1))),
            "p95": float(np.percentile(np.linalg.norm(pixel_errors, axis=1), 95)),
        },
        "3d_error": _error_metrics(errors),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    print(f"saved={args.output}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle", "noisy", "detector"), default="oracle")
    parser.add_argument("--images", type=Path, default=root / "fire-samples")
    parser.add_argument("--points", type=Path, default=root / "ground_truth.json")
    parser.add_argument("--labels-3d", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=root / "fire-model-data" / "best.pth")
    parser.add_argument("--roi-checkpoint", type=Path, default=root / "fire-model-data" / "week6_roi" / "best_roi.pth")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=root / "working" / "evaluation_3d.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG.confidence_threshold)
    parser.add_argument("--pixel-sigma", type=float, default=2.0)
    parser.add_argument("--max-dist", type=float, default=DEFAULT_CONFIG.ray_max_distance)
    parser.add_argument("--step", type=float, default=DEFAULT_CONFIG.ray_coarse_step)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-limit", type=int, default=0)
    parser.add_argument("--all-images", dest="labeled_only", action="store_false",
                        help="Evaluate every image instead of only images present in --points")
    parser.set_defaults(labeled_only=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
