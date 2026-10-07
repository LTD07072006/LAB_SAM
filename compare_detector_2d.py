"""Compare the existing 2D detector with the improved detector before ROI.

Example:

    python compare_detector_2d.py --new fire-model-data/detector_2d_v2/best_detector_2d.pth \
      --max-images 50 --split test --output-dir output/detector_2d_comparison

This script evaluates only the detector stage. It intentionally does not run
ROI refinement or ray casting, so a gain/loss here can be attributed to the
2D input supplied to the downstream stages.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from project_paths import CCTV_DATASET

from detector_2d import Detector2DInference
from fire_detector import FireDetector
from train_week6 import load_records, split_records


ROOT = Path(__file__).resolve().parent
COLOURS = {
    "gt": (46, 204, 113),
    "old": (52, 152, 219),
    "new": (231, 76, 60),
}


@dataclass
class PointPrediction:
    confidence: float = 0.0
    point: Optional[tuple[float, float]] = None
    detected: bool = False
    latency_ms: float = 0.0


def _metric(errors: list[float], predictions: list[PointPrediction], has_fire: list[bool]) -> dict[str, float]:
    predicted = [item.detected for item in predictions]
    tp = sum(bool(p and truth) for p, truth in zip(predicted, has_fire))
    fp = sum(bool(p and not truth) for p, truth in zip(predicted, has_fire))
    fn = sum(bool(not p and truth) for p, truth in zip(predicted, has_fire))
    tn = sum(bool(not p and not truth) for p, truth in zip(predicted, has_fire))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result = {
        "detected": float(sum(predicted)),
        "total": float(len(predictions)),
        "detection_rate": float(sum(predicted) / max(1, len(predictions))),
        "accuracy": float((tp + tn) / max(1, len(predictions))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "mae_px": float(np.mean(errors)) if errors else float("nan"),
        "median_px": float(np.median(errors)) if errors else float("nan"),
        "p95_px": float(np.percentile(errors, 95)) if errors else float("nan"),
        "pck10": float(np.mean(np.asarray(errors) <= 10.0)) if errors else float("nan"),
        "pck25": float(np.mean(np.asarray(errors) <= 25.0)) if errors else float("nan"),
    }
    return result


def _prediction(raw: Any, threshold: float) -> PointPrediction:
    confidence = float(getattr(raw, "confidence", getattr(raw, "p_fire", 0.0)))
    point = getattr(raw, "pixel", getattr(raw, "point", None))
    detected = bool(getattr(raw, "detected", confidence >= threshold)) and confidence >= threshold
    if point is None or not detected:
        return PointPrediction(confidence=confidence, detected=False, latency_ms=float(getattr(raw, "latency_ms", 0.0)))
    values = np.asarray(point, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.all(np.isfinite(values[:2])):
        return PointPrediction(confidence=confidence, detected=False)
    return PointPrediction(
        confidence=confidence,
        point=(float(values[0]), float(values[1])),
        detected=True,
        latency_ms=float(getattr(raw, "latency_ms", 0.0)),
    )


def _draw_marker(draw: ImageDraw.ImageDraw, point: Optional[tuple[float, float]], colour, label: str) -> None:
    if point is None:
        return
    x, y = point
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=3)
    draw.line((x - 2 * radius, y, x + 2 * radius, y), fill=colour, width=2)
    draw.line((x, y - 2 * radius, x, y + 2 * radius), fill=colour, width=2)
    draw.text((x + radius + 2, y - radius - 2), label, fill=colour)


def _contact_sheet(items: list[Image.Image], output: Path, columns: int = 2) -> None:
    if not items:
        return
    width = max(image.width for image in items)
    height = max(image.height for image in items)
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * rows), (20, 20, 20))
    for index, image in enumerate(items):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset", type=Path, default=CCTV_DATASET)
    parser.add_argument("--old", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "detector_2d_comparison")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records, stats = load_records(args.labels, args.dataset)
    splits = split_records(records, seed=args.seed)
    selected = records if args.split == "all" else splits[args.split]
    if args.max_images > 0:
        selected = selected[: args.max_images]
    if not selected:
        raise RuntimeError(f"No records selected: split={args.split}, stats={stats}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    old_detector = FireDetector(args.old, device=device, threshold=args.threshold)
    new_detector = Detector2DInference(args.new, device=device, threshold=args.threshold)
    old_detector.warmup(repeats=1)
    new_detector.warmup(repeats=1)
    output_images: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    for record in selected:
        image = Image.open(record.image_path).convert("RGB")
        gt = (record.x_norm * image.width, record.y_norm * image.height)
        old_start = time.perf_counter()
        old = _prediction(old_detector.detect(image), args.threshold)
        old.latency_ms = (time.perf_counter() - old_start) * 1000.0
        new_start = time.perf_counter()
        new = _prediction(new_detector.detect(image), args.threshold)
        new.latency_ms = (time.perf_counter() - new_start) * 1000.0
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        if record.has_fire:
            _draw_marker(draw, gt, COLOURS["gt"], "GT")
        else:
            draw.text((6, 42), "no_fire (no point GT)", fill=COLOURS["gt"])
        _draw_marker(draw, old.point, COLOURS["old"], "old")
        _draw_marker(draw, new.point, COLOURS["new"], "new")
        draw.rectangle((0, 0, min(canvas.width, 650), 34), fill=(0, 0, 0))
        draw.text(
            (6, 7),
            f"{Path(record.image_path).name} old={old.confidence:.2f} new={new.confidence:.2f}",
            fill=(255, 255, 255),
        )
        output_images.append(canvas)
        row = {
            "image": record.image_path,
            "gt": [gt[0], gt[1]],
            "old": {"confidence": old.confidence, "point": old.point, "detected": old.detected, "latency_ms": old.latency_ms},
            "new": {"confidence": new.confidence, "point": new.point, "detected": new.detected, "latency_ms": new.latency_ms},
            "has_fire": int(record.has_fire),
        }
        for name, prediction in (("old", old), ("new", new)):
            row[name]["error_px"] = (
                float(np.linalg.norm(np.asarray(prediction.point) - np.asarray(gt)))
                if record.has_fire and prediction.point is not None else None
            )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _contact_sheet(output_images, args.output_dir / "detector_2d_contact_sheet.png")
    for index, image in enumerate(output_images):
        image.save(args.output_dir / f"sample_{index:03d}.png")
    summary: dict[str, Any] = {
        "records": len(rows),
        "split": args.split,
        "threshold": args.threshold,
        "stats": stats,
        "branches": {},
        "rows": rows,
    }
    for name in ("old", "new"):
        errors = [row[name]["error_px"] for row in rows if row[name]["error_px"] is not None]
        predictions = [
            PointPrediction(
                confidence=float(row[name]["confidence"]),
                point=tuple(row[name]["point"]) if row[name]["point"] is not None else None,
                detected=bool(row[name]["detected"]),
            )
            for row in rows
        ]
        has_fire = [bool(row["has_fire"]) for row in rows]
        summary["branches"][name] = _metric(errors, predictions, has_fire)
        latencies = [float(row[name]["latency_ms"]) for row in rows]
        summary["branches"][name]["latency_ms_mean"] = float(np.mean(latencies))
        summary["branches"][name]["latency_ms_p95"] = float(np.percentile(latencies, 95))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("2D detector comparison")
    print("branch detected/total F1 precision recall MAE(px) median(px) P95(px) PCK@10 PCK@25 latency(ms)")
    for name in ("old", "new"):
        metric = summary["branches"][name]
        print(
            f"{name:5s} {int(metric['detected']):>8d}/{len(rows):<5d} "
            f"{metric['f1']:.3f} {metric['precision']:.3f} {metric['recall']:.3f} "
            f"{metric['mae_px']:.2f} {metric['median_px']:.2f} {metric['p95_px']:.2f} "
            f"{metric['pck10']:.3f} {metric['pck25']:.3f} {metric['latency_ms_mean']:.2f}"
        )
    print(f"saved_contact_sheet={args.output_dir / 'detector_2d_contact_sheet.png'}")
    print(f"saved_summary={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

