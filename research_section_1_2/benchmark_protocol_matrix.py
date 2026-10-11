"""Separated raw/TTA/guarded coarse and ROI metrics."""
from __future__ import annotations
import argparse, json, sys
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
        return {"count": 0}
    return {"count": int(len(x)), "mae_px": float(x.mean()), "median_px": float(np.median(x)),
            "p95_px": float(np.percentile(x, 95)), "max_px": float(x.max()),
            "pck10": float(np.mean(x <= 10)), "pck25": float(np.mean(x <= 25)),
            "outlier_gt25": int(np.sum(x > 25))}

def run(args: argparse.Namespace) -> dict[str, Any]:
    records, _ = load_records(args.labels, args.dataset_root)
    selected = split_records(records, seed=args.seed)[args.split]
    detector = FireDetector(args.checkpoint, device=args.device, threshold=args.threshold,
                            use_amp=False, tta_mode="brightness3", max_stability_px=None)
    refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device)
    guard = TemporalCoarseOutlierGuard(max_jump_px=args.max_jump_px,
                                       max_frame_mean_abs_diff=args.max_frame_mean_abs_diff,
                                       min_stability_px=args.min_stability_px)
    detector.warmup(repeats=1)
    rows: list[dict[str, Any]] = []
    for record in sorted(selected, key=lambda r: str(r.image_path)):
        image = Image.open(record.image_path).convert("RGB")
        result = detector.detect(image, warmup=False)
        raw = result.raw_pixel if result.raw_pixel is not None else result.pixel
        guarded = guard.update(image, raw if result.detected else None, stability_px=result.stability_px)
        if not record.has_fire or raw is None:
            continue
        gt = np.asarray([record.x_norm * image.width, record.y_norm * image.height], dtype=np.float64)
        roi_raw = refiner.refine(image, raw)
        roi_guarded = refiner.refine(image, guarded.point) if guarded.point is not None else None
        rows.append({"image": Path(record.image_path).name,
                     "raw_error_px": float(np.linalg.norm(np.asarray(raw) - gt)),
                     "guarded_error_px": (None if guarded.point is None else float(np.linalg.norm(np.asarray(guarded.point) - gt))),
                     "roi_raw_error_px": float(np.linalg.norm(np.asarray(roi_raw.point) - gt)),
                     "roi_guarded_error_px": (None if roi_guarded is None else float(np.linalg.norm(np.asarray(roi_guarded.point) - gt))),
                     "guard_replaced": guarded.replaced})
    keys = ("raw_error_px", "guarded_error_px", "roi_raw_error_px", "roi_guarded_error_px")
    report = {"format": "LAB_SAM.protocol_matrix.v1",
              "protocol": {"split": args.split, "tta_mode": "brightness3", "labels_used_only_for_evaluation": True},
              "metrics": {key: _stats([row[key] for row in rows if row[key] is not None]) for key in keys},
              "guard_replaced": int(sum(row["guard_replaced"] for row in rows)), "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "guard_replaced": report["guard_replaced"]}, indent=2))
    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--roi-checkpoint", type=Path, default=ROOT / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "section_1_2_pre_roi_20261010" / "protocol_matrix_test")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-jump-px", type=float, default=20.0)
    parser.add_argument("--max-frame-mean-abs-diff", type=float, default=25.0)
    parser.add_argument("--min-stability-px", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    run(parse_args())
