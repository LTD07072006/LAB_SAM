"""Ordered-frame benchmark for the label-free coarse outlier guard."""
from __future__ import annotations

import argparse
import json
import re
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
from temporal_outlier_guard import TemporalCoarseOutlierGuard
from train_week6 import load_records, split_records


def _stats(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"count": 0, "mae_px": None, "median_px": None, "p95_px": None,
                "max_px": None, "pck10": None, "pck25": None, "outlier_gt25": 0}
    return {"count": int(len(x)), "mae_px": float(x.mean()),
            "median_px": float(np.median(x)), "p95_px": float(np.percentile(x, 95)),
            "max_px": float(x.max()), "pck10": float(np.mean(x <= 10)),
            "pck25": float(np.mean(x <= 25)), "outlier_gt25": int(np.sum(x > 25))}


def _frame_number(path: str) -> int:
    match = re.search(r"img_(\d+)", Path(path).name.lower())
    return int(match.group(1)) if match else -1


def run(args: argparse.Namespace) -> dict[str, Any]:
    records, load_stats = load_records(args.labels, args.dataset_root)
    selected = split_records(records, seed=args.seed)[args.split] if args.split != "all" else list(records)
    selected = sorted(selected, key=lambda record: (_frame_number(record.image_path), record.image_path))
    if args.max_images > 0:
        selected = selected[:args.max_images]
    detector = FireDetector(args.checkpoint, device=args.device, threshold=args.threshold,
                            use_amp=False, tta_mode=args.tta_mode, max_stability_px=None)
    refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device)
    guard = TemporalCoarseOutlierGuard(max_jump_px=args.max_jump_px,
                                       max_frame_mean_abs_diff=args.max_frame_mean_abs_diff,
                                       min_stability_px=args.min_stability_px)
    detector.warmup(repeats=1)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, 1):
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt = np.asarray([record.x_norm * width, record.y_norm * height], dtype=np.float64)
        result = detector.detect(image, warmup=False)
        raw = result.raw_pixel if result.raw_pixel is not None else result.pixel
        guarded = guard.update(image, raw, stability_px=result.stability_px)
        row: dict[str, Any] = {
            "image": Path(record.image_path).name, "image_path": str(record.image_path),
            "frame_number": _frame_number(record.image_path), "has_fire": bool(record.has_fire),
            "ground_truth_pixel": gt.tolist() if record.has_fire else None,
            "raw_pixel": raw, "guarded_pixel": guarded.point, "stability_px": result.stability_px,
            "guard_replaced": guarded.replaced, "guard_reason": guarded.reason,
            "jump_px": guarded.jump_px, "frame_diff": guarded.frame_diff,
            "raw_error_px": None if not record.has_fire or raw is None else float(np.linalg.norm(np.asarray(raw) - gt)),
            "guarded_error_px": None if not record.has_fire else float(np.linalg.norm(np.asarray(guarded.point) - gt)),
            "roi_error_px": None,
        }
        if record.has_fire and guarded.point is not None:
            refined = refiner.refine(image, guarded.point)
            row["roi_pixel"] = refined.point
            row["roi_error_px"] = float(np.linalg.norm(np.asarray(refined.point) - gt))
            row["roi_confidence"] = float(refined.confidence)
        rows.append(row)
        if index == 1 or index == len(selected) or index % 25 == 0:
            print(f"processed={index}/{len(selected)}", flush=True)

    positives = [row for row in rows if row["has_fire"]]
    report = {
        "format": "LAB_SAM.temporal_coarse_guard.v1",
        "protocol": {"split": args.split, "ordered_by_filename": True,
                      "tta_mode": args.tta_mode, "labels_used_only_for_evaluation": True,
                      "max_jump_px": args.max_jump_px,
                      "max_frame_mean_abs_diff": args.max_frame_mean_abs_diff,
                      "min_stability_px": args.min_stability_px},
        "dataset": {"labels": str(args.labels), "root": str(args.dataset_root),
                    "load_stats": load_stats, "records": len(rows),
                    "positive_records": len(positives)},
        "raw_coarse": _stats([row["raw_error_px"] for row in positives]),
        "guarded_coarse": _stats([row["guarded_error_px"] for row in positives]),
        "guarded_roi": _stats([row["roi_error_px"] for row in positives if row["roi_error_px"] is not None]),
        "replaced_count": int(sum(row["guard_replaced"] for row in rows)),
        "replaced_positive_count": int(sum(row["has_fire"] and row["guard_replaced"] for row in rows)),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("raw_coarse", "guarded_coarse", "guarded_roi", "replaced_count", "replaced_positive_count", "saved") if key in report}, indent=2))
    print(f"saved={path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--roi-checkpoint", type=Path, default=ROOT / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "temporal_coarse_guard")
    parser.add_argument("--split", choices=("val", "test", "all"), default="test")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta-mode", choices=("brightness3", "photometric5"), default="brightness3")
    parser.add_argument("--max-jump-px", type=float, default=20.0)
    parser.add_argument("--max-frame-mean-abs-diff", type=float, default=35.0)
    parser.add_argument("--min-stability-px", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
