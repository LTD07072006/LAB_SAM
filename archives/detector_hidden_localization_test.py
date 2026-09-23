"""Evaluate detector-predicted pixels against hidden 2D labels and 3D rays.

The detector/localiser never receives the ground-truth pixel. Ground truth is
loaded only by the evaluator after inference to score pixel and 3D errors.

Because ground_truth.json contains 2D pixels rather than measured 3D points,
the 3D reference is the GridMap intersection of the labelled pixel. Therefore
the reported 3D error measures the effect of detector pixel error under the
current camera + GridMap model; it is not an absolute physical 3D error.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG
from fire_detector import FireDetector
from locator import GridMap, intersect_ray_with_grid_result
from main_integrated import make_default_camera


def load_labels(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def localise_pixel(camera, grid, pixel):
    origin, ray = camera.pixel_to_ray(*pixel)
    return intersect_ray_with_grid_result(
        origin,
        ray,
        grid,
        max_dist=DEFAULT_CONFIG.ray_max_distance,
        step=1.0,
        bisection_iterations=DEFAULT_CONFIG.ray_bisection_iterations,
    )


def make_camera_for_image(width, height):
    """Scale the prototype intrinsics from 1280x720 to this image size."""
    camera, _ = make_default_camera()
    sx, sy = float(width) / 1280.0, float(height) / 720.0
    K = camera.K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return type(camera)(K, camera.R, camera.t)


def run(samples_dir: Path, labels_path: Path, model_path: Path,
        threshold=None, device=None):
    labels = load_labels(labels_path)
    detector = FireDetector(
        model_path,
        device=device,
        threshold=(DEFAULT_CONFIG.confidence_threshold if threshold is None else threshold),
    )
    detector.warmup(repeats=2)
    camera, camera_pos = make_default_camera()
    grid = GridMap()

    rows = []
    for name, label in labels.items():
        image_path = samples_dir / name
        if not image_path.exists():
            rows.append({"name": name, "status": "missing_image"})
            continue

        detection = detector.detect(image_path)
        row = {
            "name": name,
            "status": "detected" if detection.detected else "rejected_by_detector",
            "confidence": detection.confidence,
            "size": detection.size,
            "label_pixel": tuple(float(v) for v in label),
            "predicted_pixel": detection.pixel,
        }

        if not detection.detected or detection.pixel is None:
            rows.append(row)
            continue

        label_pixel = np.asarray(label, dtype=np.float64)
        predicted_pixel = np.asarray(detection.pixel, dtype=np.float64)
        row["pixel_error_px"] = float(np.linalg.norm(predicted_pixel - label_pixel))
        row["pixel_dx_px"] = float(predicted_pixel[0] - label_pixel[0])
        row["pixel_dy_px"] = float(predicted_pixel[1] - label_pixel[1])

        # The labelled pixel is used here only to create the evaluation
        # reference. It is never passed into the detector or localisation call.
        image_camera = make_camera_for_image(*detection.size)
        reference = localise_pixel(image_camera, grid, label_pixel)
        estimate = localise_pixel(image_camera, grid, predicted_pixel)
        row["reference_status"] = reference.status
        row["estimate_status"] = estimate.status
        if reference.hit and estimate.hit:
            row["reference_point"] = reference.point.tolist()
            row["estimated_point"] = estimate.point.tolist()
            row["error_3d_m"] = float(np.linalg.norm(estimate.point - reference.point))
            row["range_reference_m"] = float(reference.distance)
            row["range_estimate_m"] = float(estimate.distance)
        rows.append(row)

    detected = [r for r in rows if r.get("status") == "detected"]
    valid_3d = [r for r in detected if "error_3d_m" in r]
    pixel_errors = [r["pixel_error_px"] for r in detected]
    errors_3d = [r["error_3d_m"] for r in valid_3d]

    print("detector_hidden_localization_test")
    print(f"samples={len(rows)}")
    print(f"detected={len(detected)}")
    print(f"rejected_by_detector={sum(r.get('status') == 'rejected_by_detector' for r in rows)}")
    print(f"valid_3d_comparisons={len(valid_3d)}")
    print(f"device={detector.device}")
    print(f"camera_position={np.array2string(camera_pos, precision=4)}")

    if pixel_errors:
        values = np.asarray(pixel_errors)
        print(f"pixel_error_mean_px={values.mean():.6f}")
        print(f"pixel_error_median_px={np.median(values):.6f}")
        print(f"pixel_error_p95_px={np.percentile(values, 95):.6f}")
    if errors_3d:
        values = np.asarray(errors_3d)
        print(f"error_3d_mean_m={values.mean():.6f}")
        print(f"error_3d_median_m={np.median(values):.6f}")
        print(f"error_3d_p95_m={np.percentile(values, 95):.6f}")
        print(f"error_3d_max_m={values.max():.6f}")

    print("per_image:")
    for row in rows:
        if row["status"] == "detected":
            print(
                f"  {row['name']:10} p={row['confidence']:.4f} "
                f"pred=({row['predicted_pixel'][0]:.1f},{row['predicted_pixel'][1]:.1f}) "
                f"gt=({row['label_pixel'][0]:.1f},{row['label_pixel'][1]:.1f}) "
                f"pixel_err={row['pixel_error_px']:.3f}px "
                f"error_3d={row.get('error_3d_m', float('nan')):.4f}m"
            )
        else:
            print(f"  {row['name']:10} p={row.get('confidence', float('nan')):.4f} {row['status']}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_CONFIG.sample_dir)
    parser.add_argument("--labels", type=Path,
                        default=DEFAULT_CONFIG.sample_dir.parent / "ground_truth.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_path)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.samples_dir, args.labels, args.model, args.threshold, args.device)


if __name__ == "__main__":
    main()
