"""Run FireDetector on the expanded fire-samples manifest.

This evaluates classification only for all manifest images. Pixel/3D error is
reported only for the ten manual samples that have ground_truth.json labels.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import DEFAULT_CONFIG
from detector_hidden_localization_test import localise_pixel, make_camera_for_image
from fire_detector import FireDetector
from locator import GridMap
from main_integrated import make_default_camera


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scores(rows, threshold):
    tp = sum(r["label"] == "fire" and r["confidence"] >= threshold for r in rows)
    fp = sum(r["label"] != "fire" and r["confidence"] >= threshold for r in rows)
    fn = sum(r["label"] == "fire" and r["confidence"] < threshold for r in rows)
    tn = sum(r["label"] != "fire" and r["confidence"] < threshold for r in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, len(rows))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


def run(manifest_path: Path, samples_dir: Path, labels_path: Path,
        model_path: Path, device=None, batch_size=16, threshold=0.5):
    manifest = load_manifest(manifest_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    detector = FireDetector(model_path, device=device, threshold=threshold)
    detector.warmup(repeats=2)

    items = []
    for entry in manifest:
        path = samples_dir / entry["file"]
        if path.is_file():
            items.append((entry, path))
    rows = []
    start = time.perf_counter()
    for offset in range(0, len(items), batch_size):
        batch = items[offset:offset + batch_size]
        results = detector.detect_many([path for _, path in batch])
        for (entry, path), result in zip(batch, results):
            rows.append({
                "file": entry["file"],
                "label": entry["label"],
                "source": entry["source"],
                "confidence": result.confidence,
                "detected": result.detected,
                "pixel": result.pixel,
                "size": result.size,
            })
    total_ms = (time.perf_counter() - start) * 1000.0

    print("expanded_test_benchmark")
    print(f"images_manifest={len(manifest)}")
    print(f"images_evaluated={len(rows)}")
    print(f"device={detector.device}")
    print(f"total_inference_ms={total_ms:.3f}")
    print(f"mean_inference_ms_per_image={total_ms / max(1, len(rows)):.3f}")
    print(f"throughput_fps={1000.0 * len(rows) / max(total_ms, 1e-9):.3f}")

    for t in (0.3, 0.5, 0.7):
        print(f"metrics_threshold_{t:.1f}={scores(rows, t)}")

    print("by_source_threshold_0.5:")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    for source in sorted(grouped):
        group = grouped[source]
        print(f"  {source}: n={len(group)} metrics={scores(group, 0.5)} confidence_mean={np.mean([r['confidence'] for r in group]):.4f}")

    # Only the ten manually annotated samples have a physical 2D reference.
    gt_pixels = {name: np.asarray(value, dtype=np.float64) for name, value in labels.items()}
    camera, _ = make_default_camera()
    grid = GridMap()
    coord_rows = []
    for row in rows:
        if row["file"] not in gt_pixels or not row["detected"] or row["pixel"] is None:
            continue
        label_pixel = gt_pixels[row["file"]]
        predicted_pixel = np.asarray(row["pixel"], dtype=np.float64)
        image_camera = make_camera_for_image(*row["size"])
        reference = localise_pixel(image_camera, grid, label_pixel)
        estimate = localise_pixel(image_camera, grid, predicted_pixel)
        item = {
            "file": row["file"],
            "pixel_error_px": float(np.linalg.norm(predicted_pixel - label_pixel)),
            "reference_hit": reference.hit,
            "estimate_hit": estimate.hit,
        }
        if reference.hit and estimate.hit:
            item["error_3d_m"] = float(np.linalg.norm(estimate.point - reference.point))
        coord_rows.append(item)
    if coord_rows:
        pixel = np.asarray([r["pixel_error_px"] for r in coord_rows])
        xyz = np.asarray([r["error_3d_m"] for r in coord_rows if "error_3d_m" in r])
        print(f"manual_coordinate_samples={len(coord_rows)}")
        print(f"manual_pixel_error_mean_px={pixel.mean():.6f}")
        print(f"manual_pixel_error_p95_px={np.percentile(pixel, 95):.6f}")
        if len(xyz):
            print(f"manual_error_3d_mean_m={xyz.mean():.6f}")
            print(f"manual_error_3d_p95_m={np.percentile(xyz, 95):.6f}")
    else:
        print("manual_coordinate_samples=0")

    report = {"thresholds": {str(t): scores(rows, t) for t in (0.3, 0.5, 0.7)},
              "by_source": {k: scores(v, 0.5) for k, v in grouped.items()},
              "rows": rows, "coordinate_rows": coord_rows,
              "timing": {"total_ms": total_ms, "mean_ms": total_ms / max(1, len(rows))}}
    report_path = samples_dir / "expanded_test_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "fire-samples" / "test_manifest.json")
    parser.add_argument("--samples-dir", type=Path, default=root / "fire-samples")
    parser.add_argument("--labels", type=Path, default=root / "ground_truth.json")
    parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.manifest, args.samples_dir, args.labels, args.model, args.device, args.batch_size, args.threshold)


if __name__ == "__main__":
    main()
