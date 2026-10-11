"""Benchmark stability-aware dynamic ROI crops on the existing CCTV split.

The detector is evaluated without a hard stability gate so every positive
frame reaches the refiner.  Only the TTA stability value changes the crop
side; labels are used for metrics after inference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_detector import FireDetector
from narrow_localizer import ROIRefinerInference
from project_paths import CCTV_DATASET
from train_week6 import load_records, split_records


def _stats(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"count": 0, "mae_px": None, "median_px": None, "p95_px": None,
                "max_px": None, "pck10": None, "pck25": None, "outlier_gt25": 0}
    return {
        "count": int(len(x)), "mae_px": float(x.mean()),
        "median_px": float(np.median(x)), "p95_px": float(np.percentile(x, 95)),
        "max_px": float(x.max()), "pck10": float(np.mean(x <= 10.0)),
        "pck25": float(np.mean(x <= 25.0)), "outlier_gt25": int(np.sum(x > 25.0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    records, load_stats = load_records(args.labels, args.dataset_root)
    selected = list(records) if args.split == "all" else split_records(records, seed=args.seed)[args.split]
    if args.max_images > 0:
        selected = selected[:args.max_images]
    detector = FireDetector(
        args.checkpoint, device=args.device, threshold=args.threshold,
        use_amp=False, tta_mode=args.tta_mode, max_stability_px=None,
    )
    refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device)
    detector.warmup(repeats=1)
    factors = tuple(float(value) for value in args.expansion_factors)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for index, record in enumerate(selected, 1):
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt = np.asarray([record.x_norm * width, record.y_norm * height], dtype=np.float64)
        detector_result = detector.detect(image, warmup=False)
        raw = detector_result.raw_pixel if detector_result.raw_pixel is not None else detector_result.pixel
        row: dict[str, Any] = {
            "image": Path(record.image_path).name,
            "image_path": str(record.image_path),
            "has_fire": bool(record.has_fire),
            "ground_truth_pixel": gt.tolist() if record.has_fire else None,
            "detected": bool(detector_result.detected),
            "confidence": float(detector_result.confidence),
            "raw_pixel": raw,
            "stability_px": detector_result.stability_px,
            "raw_error_px": (None if not record.has_fire or raw is None else
                              float(np.linalg.norm(np.asarray(raw) - gt))),
            "variants": {},
        }
        if detector_result.detected and raw is not None:
            for factor in factors:
                t0 = time.perf_counter()
                refined = refiner.refine(
                    image, raw, stability_px=detector_result.stability_px,
                    expansion_factor=factor, max_roi_fraction=args.max_roi_fraction,
                    stability_threshold_px=args.stability_threshold_px,
                )
                point = np.asarray(refined.point, dtype=np.float64)
                error = None if not record.has_fire else float(np.linalg.norm(point - gt))
                left, top, side = refined.roi.left, refined.roi.top, refined.roi.side
                margins = np.asarray([gt[0] - left, left + side - gt[0],
                                      gt[1] - top, top + side - gt[1]])
                row["variants"][str(factor)] = {
                    "point": point.tolist(), "error_px": error,
                    "confidence": float(refined.confidence),
                    "roi_side_px": float(side),
                    "gt_inside_crop": bool(record.has_fire and np.all(margins >= 0.0)),
                    "minimum_gt_margin_px": (float(margins.min()) if record.has_fire else None),
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                }
        rows.append(row)
        if index == 1 or index == len(selected) or index % 25 == 0:
            print(f"processed={index}/{len(selected)}", flush=True)

    report: dict[str, Any] = {
        "format": "LAB_SAM.soft_roi_benchmark.v1",
        "protocol": {
            "split": args.split, "seed": args.seed, "tta_mode": args.tta_mode,
            "hard_gate_disabled": True, "labels_used_only_for_evaluation": True,
            "expansion_factors_px_per_stability": list(factors),
            "max_roi_fraction": args.max_roi_fraction,
            "stability_threshold_px": args.stability_threshold_px,
        },
        "dataset": {"labels": str(args.labels), "root": str(args.dataset_root),
                    "load_stats": load_stats, "records": len(rows),
                    "positive_records": int(sum(row["has_fire"] for row in rows))},
        "checkpoint": {"detector": str(args.checkpoint), "roi": str(args.roi_checkpoint)},
        "raw_detector": {
            "classification_detected": int(sum(row["detected"] for row in rows)),
            "positive_total": int(sum(row["has_fire"] for row in rows)),
            "coarse_error": _stats([row["raw_error_px"] for row in rows if row["raw_error_px"] is not None]),
        },
        "variants": {}, "elapsed_s": time.perf_counter() - started_all, "rows": rows,
    }
    positives = [row for row in rows if row["has_fire"]]
    for factor in factors:
        key = str(factor)
        values = [row["variants"][key]["error_px"] for row in positives
                  if key in row["variants"] and row["variants"][key]["error_px"] is not None]
        margins = [row["variants"][key]["minimum_gt_margin_px"] for row in positives
                   if key in row["variants"] and row["variants"][key]["minimum_gt_margin_px"] is not None]
        variants = [row["variants"][key] for row in positives if key in row["variants"]]
        report["variants"][key] = {
            "localization": _stats(values), "coverage": float(len(values) / max(1, len(positives))),
            "gt_inside_crop_count": int(sum(item["gt_inside_crop"] for item in variants)),
            "gt_outside_crop_count": int(sum(not item["gt_inside_crop"] for item in variants)),
            "minimum_gt_margin_px": _stats(margins),
            "roi_side_px": _stats([float(item["roi_side_px"]) for item in variants]),
            "latency_ms": _stats([float(item["latency_ms"]) for item in variants]),
        }
    # A confidence-only multi-scale selector is a deployable test: no labels
    # participate in choosing the point.  It is reported separately from the
    # fixed-factor rows so a high but miscalibrated heatmap cannot be hidden.
    selected = []
    for row in positives:
        candidates = [row["variants"][str(f)] for f in factors if str(f) in row["variants"]]
        if candidates:
            selected.append(max(candidates, key=lambda item: float(item["confidence"])))
    selected_errors = [item["error_px"] for item in selected if item["error_px"] is not None]
    selected_margins = [item["minimum_gt_margin_px"] for item in selected
                        if item["minimum_gt_margin_px"] is not None]
    report["multi_scale_confidence_select"] = {
        "factors": list(factors), "localization": _stats(selected_errors),
        "coverage": float(len(selected_errors) / max(1, len(positives))),
        "gt_inside_crop_count": int(sum(item["gt_inside_crop"] for item in selected)),
        "gt_outside_crop_count": int(sum(not item["gt_inside_crop"] for item in selected)),
        "minimum_gt_margin_px": _stats(selected_margins),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved": str(path), "variants": report["variants"]}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--roi-checkpoint", type=Path, default=ROOT / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "soft_roi_benchmark")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta-mode", choices=("brightness3", "photometric5"), default="brightness3")
    parser.add_argument("--max-roi-fraction", type=float, default=1.0)
    parser.add_argument("--stability-threshold-px", type=float, default=6.0)
    parser.add_argument("--expansion-factors", type=float, nargs="+", default=(0.0, 4.0, 8.0, 12.0, 16.0))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
