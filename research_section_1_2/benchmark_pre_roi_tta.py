"""Leakage-safe benchmark for the pre-ROI detector reliability fix.

The benchmark evaluates all records in an official split, including no-fire
images. Photometric TTA changes appearance only; it never consumes the
labelled point. If ``--max-stability-px`` is supplied, a frame can remain a
valid fire detection while its position is rejected for the ROI/3-D stages.
The report therefore separates classification metrics from localisation
coverage and error on accepted positions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_detector import FireDetector
from project_paths import CCTV_DATASET
from train_week6 import load_records, split_records


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _stats(errors: Iterable[float]) -> dict[str, Optional[float]]:
    values = np.asarray(list(errors), dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mae_px": None,
            "median_px": None,
            "p95_px": None,
            "max_px": None,
            "pck10": None,
            "pck25": None,
            "outlier_gt25": 0,
        }
    return {
        "count": int(len(values)),
        "mae_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(values.max()),
        "pck10": float(np.mean(values <= 10.0)),
        "pck25": float(np.mean(values <= 25.0)),
        "outlier_gt25": int(np.sum(values > 25.0)),
    }


def _classification(rows: list[dict[str, Any]]) -> dict[str, float]:
    truth = np.asarray([bool(row["has_fire"]) for row in rows], dtype=bool)
    predicted = np.asarray([bool(row["detected"]) for row in rows], dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "total": float(len(rows)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "accuracy": float((tp + tn) / max(1, len(rows))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
    }


def _position_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    positive = [row for row in rows if row["has_fire"]]
    valid = [row for row in positive if row.get(key) is not None]
    errors = [float(row[key]) for row in valid]
    result = _stats(errors)
    result.update(
        {
            "positive_total": int(len(positive)),
            "coverage": float(len(valid) / max(1, len(positive))),
            "missing": int(len(positive) - len(valid)),
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    records, load_stats = load_records(args.labels, args.dataset_root)
    if args.split == "all":
        selected = list(records)
    else:
        selected = split_records(records, seed=args.seed)[args.split]
    if args.max_images > 0:
        selected = selected[: args.max_images]
    if not selected:
        raise RuntimeError(f"No records selected for split={args.split}")

    detector = FireDetector(
        args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        use_amp=False,
        tta_mode=args.tta_mode,
        tta_reliability_scale_px=args.tta_reliability_scale_px,
        max_stability_px=args.max_stability_px,
    )
    detector.warmup(repeats=1)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, 1):
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        ground_truth = np.asarray([record.x_norm * width, record.y_norm * height], dtype=np.float64)
        started = time.perf_counter()
        result = detector.detect(image, warmup=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        def error(point: Any) -> Optional[float]:
            if not record.has_fire or point is None:
                return None
            values = np.asarray(point, dtype=np.float64).reshape(-1)
            if len(values) < 2 or not np.isfinite(values[:2]).all():
                return None
            return float(np.linalg.norm(values[:2] - ground_truth))

        raw_point = result.raw_pixel if result.raw_pixel is not None else result.pixel
        rows.append(
            {
                "image": Path(record.image_path).name,
                "image_path": str(record.image_path),
                "has_fire": bool(record.has_fire),
                "ground_truth_pixel": ground_truth.tolist() if record.has_fire else None,
                "confidence": float(result.confidence),
                "detected": bool(result.detected),
                "pixel": result.pixel,
                "raw_pixel": raw_point,
                "localization_accepted": bool(result.localization_accepted),
                "localization_status": str(result.localization_status),
                "stability_px": result.stability_px,
                "stability_reference_px": result.stability_reference_px,
                "position_reliability": result.position_reliability,
                "tta_count": int(result.tta_count),
                "latency_ms": float(elapsed_ms),
                "raw_error_px": error(raw_point),
                "accepted_error_px": error(result.pixel),
            }
        )
        if index == 1 or index == len(selected) or index % 50 == 0:
            print(f"processed={index}/{len(selected)}", flush=True)

    status_counts = Counter(str(row["localization_status"]) for row in rows)
    finite_stability = [float(row["stability_px"]) for row in rows if row["stability_px"] is not None]
    latencies = [float(row["latency_ms"]) for row in rows]
    report: dict[str, Any] = {
        "format": "LAB_SAM.pre_roi_tta_benchmark.v1",
        "protocol": {
            "split": args.split,
            "seed": args.seed,
            "threshold": args.threshold,
            "tta_mode": args.tta_mode,
            "tta_reliability_scale_px": args.tta_reliability_scale_px,
            "max_stability_px": args.max_stability_px,
            "labels_are_used_only_for_evaluation": True,
            "ground_truth_never_supplied_to_detector": True,
            "device": args.device,
        },
        "dataset": {
            "labels": str(args.labels),
            "root": str(args.dataset_root),
            "load_stats": load_stats,
            "records": len(rows),
            "positive_records": int(sum(row["has_fire"] for row in rows)),
        },
        "checkpoint": str(args.checkpoint),
        "classification": _classification(rows),
        "localization_raw": _position_metric(rows, "raw_error_px"),
        "localization_accepted": _position_metric(rows, "accepted_error_px"),
        "rejected_positive": int(sum(row["has_fire"] and not row["localization_accepted"] for row in rows)),
        "localization_status_counts": dict(status_counts),
        "stability_px": _stats(finite_stability),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(_json_value(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_value({
        "classification": report["classification"],
        "localization_raw": report["localization_raw"],
        "localization_accepted": report["localization_accepted"],
        "rejected_positive": report["rejected_positive"],
        "latency_ms": report["latency_ms"],
        "saved": str(path),
    }), indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "pre_roi_tta_benchmark")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tta-mode", choices=("none", "brightness3", "photometric5"), default="none")
    parser.add_argument("--tta-reliability-scale-px", type=float, default=8.0)
    parser.add_argument("--max-stability-px", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
