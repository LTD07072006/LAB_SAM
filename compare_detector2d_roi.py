"""Paired benchmark for the improved 2D detector and the ROI refiner.

The benchmark evaluates the same labelled CCTV records with four branches:

    baseline             old detector (best.pth)
    baseline_roi         old detector -> ROIRefiner
    detector_2d           detector_2d_fpn_v2 (best_detector_2d.pth)
    detector_2d_roi       detector_2d_fpn_v2 -> ROIRefiner

The last branch is the direct comparison requested for the new detector used
with the existing ROI model.  All point errors are measured in the original
image coordinate system.  Classification metrics are calculated on every
record; point metrics are calculated on positive records for which the branch
produced a point, with coverage reported separately so missed detections are
not hidden.

This script evaluates the 2D/ROI stage only.  It deliberately does not claim
metre-level 3D accuracy: that requires measured calibration, a metric mesh and
3D ground truth in the same coordinate frame.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from project_paths import CCTV_DATASET

from detector_2d import Detector2DInference
from fire_detector import FireDetector
from narrow_localizer import ROIRefinerInference
from train_week6 import Record, load_records, split_records


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLOURS = {
    "gt": (46, 204, 113),
    "baseline": (52, 152, 219),
    "baseline_roi": (241, 156, 18),
    "detector_2d": (231, 76, 60),
    "detector_2d_roi": (155, 89, 182),
}
BRANCH_ORDER = ("baseline", "baseline_roi", "detector_2d", "detector_2d_roi")


def _allow_linux_checkpoints_on_windows() -> None:
    """Let Windows unpickle checkpoints saved by Kaggle/Linux.

    PyTorch checkpoints can contain an ``argparse.Namespace`` or metadata
    field holding ``pathlib.PosixPath``.  The Windows pathlib implementation
    refuses to instantiate that concrete class during unpickling.  The path
    is metadata only, so mapping it to the local concrete path class is safe
    and keeps the tensor weights unchanged.
    """
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]


@dataclass
class Prediction:
    confidence: float = 0.0
    point: Optional[tuple[float, float]] = None
    detected: bool = False
    detector_ms: float = 0.0
    roi_ms: float = 0.0
    total_ms: float = 0.0
    roi_confidence: Optional[float] = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_point(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(values) < 2 or not np.all(np.isfinite(values[:2])):
        return None
    return float(values[0]), float(values[1])


def _as_prediction(raw: Any, threshold: float, elapsed_ms: float) -> Prediction:
    confidence = float(getattr(raw, "confidence", getattr(raw, "p_fire", 0.0)))
    detected = bool(getattr(raw, "detected", confidence >= threshold)) and confidence >= threshold
    point = _finite_point(getattr(raw, "pixel", getattr(raw, "point", None)))
    if not detected or point is None:
        point = None
        detected = False
    return Prediction(
        confidence=confidence,
        point=point,
        detected=detected,
        detector_ms=float(elapsed_ms),
        total_ms=float(elapsed_ms),
    )


def _run_detector(detector: Any, image: Image.Image, threshold: float, device: torch.device) -> Prediction:
    _sync(device)
    started = time.perf_counter()
    raw = detector.detect(image)
    _sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _as_prediction(raw, threshold, elapsed_ms)


def _run_branch(
    detector: Any,
    refiner: Optional[ROIRefinerInference],
    use_roi: bool,
    image: Image.Image,
    threshold: float,
    device: torch.device,
) -> Prediction:
    prediction = _run_detector(detector, image, threshold, device)
    if not use_roi or not prediction.detected or prediction.point is None or refiner is None:
        return prediction

    _sync(device)
    started = time.perf_counter()
    refined = refiner.refine(image, prediction.point)
    _sync(device)
    roi_ms = (time.perf_counter() - started) * 1000.0
    refined_point = _finite_point(refined.point)
    if refined_point is not None:
        prediction.point = refined_point
        prediction.roi_confidence = float(refined.confidence)
    prediction.roi_ms = float(roi_ms)
    prediction.total_ms = float(prediction.detector_ms + roi_ms)
    return prediction


def _select_records(
    records: Sequence[Record], max_images: int, selection: str
) -> list[Record]:
    if max_images <= 0 or len(records) <= max_images:
        return list(records)
    if selection == "head":
        return list(records[:max_images])
    indexes = np.linspace(0, len(records) - 1, max_images, dtype=int)
    if selection == "even":
        return [records[int(index)] for index in indexes]

    # Diverse selection keeps positive and negative samples represented, then
    # fills the remaining slots in deterministic order.
    groups = {
        0: [index for index, record in enumerate(records) if not record.has_fire],
        1: [index for index, record in enumerate(records) if record.has_fire],
    }
    selected: set[int] = set()
    for group in groups.values():
        if group:
            selected.add(group[len(group) // 2])
    for index in indexes:
        if len(selected) >= max_images:
            break
        selected.add(int(index))
    for index in range(len(records)):
        if len(selected) >= max_images:
            break
        selected.add(index)
    return [records[index] for index in sorted(selected)[:max_images]]


def _marker(draw: ImageDraw.ImageDraw, point: Optional[Sequence[float]], colour, label: str) -> None:
    if point is None:
        return
    x, y = float(point[0]), float(point[1])
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=2)
    draw.line((x - 2 * radius, y, x + 2 * radius, y), fill=colour, width=1)
    draw.line((x, y - 2 * radius, x, y + 2 * radius), fill=colour, width=1)
    draw.text((x + radius + 2, y - radius - 2), label, fill=colour)


def _draw_overlay(
    image: Image.Image,
    record: Record,
    predictions: dict[str, Prediction],
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    if record.has_fire:
        gt = (record.x_norm * canvas.width, record.y_norm * canvas.height)
        _marker(draw, gt, COLOURS["gt"], "GT")
    else:
        draw.text((5, 42), "GT: no_fire", fill=COLOURS["gt"])
    labels = {
        "baseline": "old",
        "baseline_roi": "old+ROI",
        "detector_2d": "new",
        "detector_2d_roi": "new+ROI",
    }
    for name in BRANCH_ORDER:
        prediction = predictions[name]
        _marker(draw, prediction.point, COLOURS[name], labels[name])
    draw.rectangle((0, 0, min(canvas.width, 900), 33), fill=(0, 0, 0))
    confidence_text = " ".join(
        f"{labels[name]}={predictions[name].confidence:.2f}" for name in BRANCH_ORDER
    )
    draw.text((5, 7), f"{Path(record.image_path).name} {confidence_text}", fill=(255, 255, 255))
    return canvas


def _contact_sheet(paths: Sequence[Path], output: Path, columns: int = 3) -> None:
    if not paths:
        return
    tile_size = (420, 320)
    tiles: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            tiles.append(ImageOps.contain(source.convert("RGB"), tile_size))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_size[0] + (tile_size[0] - tile.width) // 2
        y = (index // columns) * tile_size[1] + (tile_size[1] - tile.height) // 2
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _classification_metrics(predictions: Sequence[Prediction], records: Sequence[Record]) -> dict[str, float]:
    truth = np.asarray([bool(record.has_fire) for record in records], dtype=bool)
    predicted = np.asarray([prediction.detected for prediction in predictions], dtype=bool)
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "total": float(len(records)),
        "positive": float(np.sum(truth)),
        "negative": float(np.sum(~truth)),
        "detected": float(np.sum(predicted)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(1, len(records))),
        "specificity": float(tn / max(1, tn + fp)),
    }


def _point_metrics(
    predictions: Sequence[Prediction], records: Sequence[Record]
) -> dict[str, float | None]:
    errors: list[float] = []
    positive_count = sum(int(record.has_fire) for record in records)
    detected_positive = 0
    for prediction, record in zip(predictions, records):
        if not record.has_fire or not prediction.detected or prediction.point is None:
            continue
        detected_positive += 1
        width, height = Image.open(record.image_path).size
        truth = np.asarray((record.x_norm * width, record.y_norm * height), dtype=np.float64)
        errors.append(float(np.linalg.norm(np.asarray(prediction.point) - truth)))
    values = np.asarray(errors, dtype=np.float64)
    result: dict[str, float | None] = {
        "positive_count": float(positive_count),
        "detected_positive": float(detected_positive),
        "point_coverage": float(detected_positive / max(1, positive_count)),
        "mae_px": None,
        "median_px": None,
        "p95_px": None,
        "pck10": None,
        "pck25": None,
    }
    if len(values):
        result.update(
            {
                "mae_px": float(np.mean(values)),
                "median_px": float(np.median(values)),
                "p95_px": float(np.percentile(values, 95.0)),
                "pck10": float(np.mean(values <= 10.0)),
                "pck25": float(np.mean(values <= 25.0)),
            }
        )
    return result


def _paired_roi_metrics(
    base: Sequence[Prediction], roi: Sequence[Prediction], records: Sequence[Record]
) -> dict[str, float | None]:
    before: list[float] = []
    after: list[float] = []
    for base_prediction, roi_prediction, record in zip(base, roi, records):
        if not record.has_fire or not base_prediction.point or not roi_prediction.point:
            continue
        width, height = Image.open(record.image_path).size
        truth = np.asarray((record.x_norm * width, record.y_norm * height), dtype=np.float64)
        before.append(float(np.linalg.norm(np.asarray(base_prediction.point) - truth)))
        after.append(float(np.linalg.norm(np.asarray(roi_prediction.point) - truth)))
    if not before:
        return {"count": 0.0, "base_mae_px": None, "roi_mae_px": None, "improvement_px": None, "improved_fraction": None}
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    return {
        "count": float(len(before)),
        "base_mae_px": float(np.mean(before_array)),
        "roi_mae_px": float(np.mean(after_array)),
        "improvement_px": float(np.mean(before_array - after_array)),
        "improved_fraction": float(np.mean(after_array < before_array)),
    }


def _branch_summary(
    predictions: Sequence[Prediction], records: Sequence[Record]
) -> dict[str, Any]:
    classification = _classification_metrics(predictions, records)
    point = _point_metrics(predictions, records)
    detector_times = np.asarray([item.detector_ms for item in predictions], dtype=np.float64)
    roi_times = np.asarray([item.roi_ms for item in predictions], dtype=np.float64)
    total_times = np.asarray([item.total_ms for item in predictions], dtype=np.float64)
    latency = {
        "detector_mean_ms": float(np.mean(detector_times)) if len(detector_times) else 0.0,
        "detector_p95_ms": float(np.percentile(detector_times, 95)) if len(detector_times) else 0.0,
        "roi_mean_ms": float(np.mean(roi_times)) if len(roi_times) else 0.0,
        "total_mean_ms": float(np.mean(total_times)) if len(total_times) else 0.0,
        "total_p95_ms": float(np.percentile(total_times, 95)) if len(total_times) else 0.0,
    }
    return {"classification": classification, "point": point, "latency": latency}


def _print_summary(summary: dict[str, Any], image_count: int) -> None:
    print("paired detector/ROI benchmark (same labelled records)")
    print(
        "branch              detected  F1     recall  point_cov  MAE(px)  median  P95     PCK@10  PCK@25  total_ms"
    )
    labels = {
        "baseline": "baseline",
        "baseline_roi": "baseline+ROI",
        "detector_2d": "detector_2d",
        "detector_2d_roi": "detector_2d+ROI",
    }
    for name in BRANCH_ORDER:
        item = summary["branches"][name]
        cls = item["classification"]
        point = item["point"]
        latency = item["latency"]
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.3f}"
        print(
            f"{labels[name]:18s} {int(cls['detected']):>4d}/{image_count:<4d} "
            f"{cls['f1']:.3f}  {cls['recall']:.3f}   {point['point_coverage']:.3f}     "
            f"{fmt(point['mae_px']):>7s} {fmt(point['median_px']):>7s} {fmt(point['p95_px']):>7s} "
            f"{fmt(point['pck10']):>7s} {fmt(point['pck25']):>7s} {latency['total_mean_ms']:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset", type=Path, default=CCTV_DATASET)
    parser.add_argument("--baseline", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument(
        "--new",
        type=Path,
        default=Path(r"D:\ITK_SNAP\data\best_detector_2d.pth"),
        help="detector_2d_fpn_v2 checkpoint; accepts a checkpoint outside the project",
    )
    parser.add_argument("--roi", type=Path, default=ROOT / "week6_roi_result" / "best_roi.pth")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all selected records")
    parser.add_argument("--selection", choices=("head", "even", "diverse"), default="diverse")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "detector2d_roi_benchmark",
    )
    parser.add_argument("--contact-sheet-max", type=int, default=30)
    args = parser.parse_args()

    _allow_linux_checkpoints_on_windows()

    for path, label in (
        (args.labels, "labels"),
        (args.baseline, "baseline checkpoint"),
        (args.new, "new detector checkpoint"),
        (args.roi, "ROI checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    records, load_stats = load_records(args.labels, args.dataset)
    split_map = split_records(records, seed=args.seed)
    selected = records if args.split == "all" else split_map[args.split]
    selected = _select_records(selected, args.max_images, args.selection)
    if not selected:
        raise RuntimeError(f"No records selected for split={args.split}; load_stats={load_stats}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")
    print(f"new_checkpoint={args.new}")
    print(f"records={len(selected)} load_stats={dict(load_stats)}")

    baseline = FireDetector(args.baseline, device=device, threshold=args.threshold)
    new_detector = Detector2DInference(args.new, device=device, threshold=args.threshold)
    refiner = ROIRefinerInference(args.roi, device=str(device))
    baseline.warmup(repeats=1)
    new_detector.warmup(repeats=1)

    predictions_by_branch: dict[str, list[Prediction]] = {name: [] for name in BRANCH_ORDER}
    rows: list[dict[str, Any]] = []
    preview_paths: list[Path] = []

    for index, record in enumerate(selected):
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        predictions = {
            "baseline": _run_branch(baseline, None, False, image, args.threshold, device),
            "baseline_roi": _run_branch(baseline, refiner, True, image, args.threshold, device),
            "detector_2d": _run_branch(new_detector, None, False, image, args.threshold, device),
            "detector_2d_roi": _run_branch(new_detector, refiner, True, image, args.threshold, device),
        }
        for name in BRANCH_ORDER:
            predictions_by_branch[name].append(predictions[name])

        width, height = image.size
        gt = [float(record.x_norm * width), float(record.y_norm * height)]
        row: dict[str, Any] = {
            "image": record.image_path,
            "has_fire": int(record.has_fire),
            "gt": gt if record.has_fire else None,
            "branches": {},
        }
        for name in BRANCH_ORDER:
            prediction = predictions[name]
            error = None
            if record.has_fire and prediction.point is not None:
                error = float(np.linalg.norm(np.asarray(prediction.point) - np.asarray(gt)))
            row["branches"][name] = {
                "confidence": prediction.confidence,
                "point": prediction.point,
                "detected": prediction.detected,
                "error_px": error,
                "detector_ms": prediction.detector_ms,
                "roi_ms": prediction.roi_ms,
                "total_ms": prediction.total_ms,
                "roi_confidence": prediction.roi_confidence,
            }

        overlay = _draw_overlay(image, record, predictions)
        preview_path = args.output_dir / "images" / f"sample_{index:04d}.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(preview_path)
        preview_paths.append(preview_path)
        rows.append(row)

    summary: dict[str, Any] = {
        "protocol": {
            "same_records": True,
            "split": args.split,
            "records": len(selected),
            "seed": args.seed,
            "threshold": args.threshold,
            "point_metrics_note": "positive records with a detected point; point_coverage reports missing points",
            "3d_metrics": "not evaluated; no measured calibration/metric mesh/3D ground truth",
        },
        "paths": {
            "labels": str(args.labels),
            "dataset": str(args.dataset),
            "baseline": str(args.baseline),
            "new": str(args.new),
            "roi": str(args.roi),
        },
        "load_stats": dict(load_stats),
        "branches": {},
        "paired_roi": {},
        "rows": rows,
    }
    for name in BRANCH_ORDER:
        summary["branches"][name] = _branch_summary(predictions_by_branch[name], selected)
    summary["paired_roi"] = {
        "baseline_to_baseline_roi": _paired_roi_metrics(
            predictions_by_branch["baseline"], predictions_by_branch["baseline_roi"], selected
        ),
        "detector_2d_to_detector_2d_roi": _paired_roi_metrics(
            predictions_by_branch["detector_2d"], predictions_by_branch["detector_2d_roi"], selected
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _contact_sheet(preview_paths[: max(0, args.contact_sheet_max)], args.output_dir / "contact_sheet.png")
    _print_summary(summary, len(selected))
    print(f"saved_summary={args.output_dir / 'summary.json'}")
    print(f"saved_contact_sheet={args.output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
