"""Benchmark the independent Home Fire YOLO bounding-box branch.

The benchmark keeps the detector branch separate from the existing point,
ROI, and v3 models.  It evaluates *all* fire predictions in each image with
greedy one-to-one IoU matching instead of silently selecting only the most
confident box.  The same predictions are also converted to bottom-center and
bottom-band hypotheses for the later calibrated 3D ray-casting stage.

This script reports 2D detector quality only.  A metre-level 3D claim still
requires measured camera calibration, a metric mesh/GridMap, and 3D ground
truth in the same world coordinate frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from home_fire_detector_adapter import HomeFireYOLO, YOLODetection
from project_paths import D_FIRE_ROOT


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_dataset_root(value: Path) -> Path:
    """Resolve a direct or nested ``train/val/test`` YOLO dataset root."""
    root = value.expanduser().resolve()

    def has_layout(path: Path) -> bool:
        return all(
            (path / split / "images").is_dir() and (path / split / "labels").is_dir()
            for split in ("train", "val", "test")
        )

    if has_layout(root):
        return root
    candidates = [path for path in root.rglob("*") if path.is_dir() and has_layout(path)]
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy layout YOLO train/val/test dưới {root}; "
            "cần <split>/images và <split>/labels."
        )
    candidates.sort(key=lambda path: (len(path.parts), str(path).lower()))
    return candidates[0]


def parse_yolo_label(
    path: Path, source_fire_class: int
) -> list[tuple[float, float, float, float]]:
    """Read fire boxes as clipped normalized ``xyxy`` values.

    YOLO labels use normalized ``x_center, y_center, width, height``.  Invalid
    rows are skipped so one malformed annotation does not stop the benchmark;
    the summary includes the valid annotations that were evaluated.
    """
    boxes: list[tuple[float, float, float, float]] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            xc, yc, width, height = (float(item) for item in fields[1:])
        except ValueError:
            continue
        if class_id != source_fire_class or not np.all(
            np.isfinite([xc, yc, width, height])
        ):
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        x1 = float(np.clip(xc - width / 2.0, 0.0, 1.0))
        y1 = float(np.clip(yc - height / 2.0, 0.0, 1.0))
        x2 = float(np.clip(xc + width / 2.0, 0.0, 1.0))
        y2 = float(np.clip(yc + height / 2.0, 0.0, 1.0))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def image_files(dataset_root: Path, split: str) -> list[Path]:
    image_root = dataset_root / split / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for(image_path: Path, image_root: Path, label_root: Path) -> Path:
    return label_root / image_path.relative_to(image_root).with_suffix(".txt")


def select_images(
    images: Sequence[Path],
    image_root: Path,
    label_root: Path,
    source_fire_class: int,
    max_images: int,
    selection: str,
) -> list[Path]:
    """Select a deterministic subset while retaining a useful diverse mode."""
    if max_images <= 0 or len(images) <= max_images:
        return list(images)
    if selection == "head":
        return list(images[:max_images])

    if selection == "even":
        indexes = np.linspace(0, len(images) - 1, max_images, dtype=int)
        return [images[int(index)] for index in indexes]

    # Diverse selection first spreads samples across annotation cardinality
    # (no fire, one fire, multiple fire boxes), then fills remaining slots
    # evenly.  It remains deterministic and does not inspect predictions.
    groups: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for index, image_path in enumerate(images):
        count = len(
            parse_yolo_label(
                label_path_for(image_path, image_root, label_root),
                source_fire_class,
            )
        )
        groups[0 if count == 0 else 1 if count == 1 else 2].append(index)

    selected: set[int] = set()
    nonempty = [group for group in groups.values() if group]
    for group in nonempty:
        selected.add(group[len(group) // 2])

    even_indexes = np.linspace(0, len(images) - 1, max_images, dtype=int)
    for index in even_indexes:
        if len(selected) >= max_images:
            break
        selected.add(int(index))
    if len(selected) < max_images:
        for index in range(len(images)):
            if len(selected) >= max_images:
                break
            selected.add(index)
    return [images[index] for index in sorted(selected)[:max_images]]


def iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(item) for item in first)
    bx1, by1, bx2, by2 = (float(item) for item in second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(1e-12, area_a + area_b - intersection)


def greedy_match(
    ground_truth: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    iou_threshold: float,
) -> tuple[list[dict], list[int], list[int]]:
    """Greedily match each box at most once, highest IoU first."""
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(ground_truth):
            score = iou_xyxy(prediction, truth)
            if score >= iou_threshold:
                candidates.append((score, prediction_index, truth_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    matches: list[dict] = []
    for score, prediction_index, truth_index in candidates:
        if prediction_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(prediction_index)
        matched_truth.add(truth_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": truth_index,
                "iou": float(score),
            }
        )
    unmatched_predictions = [
        index for index in range(len(predictions)) if index not in matched_predictions
    ]
    unmatched_truth = [index for index in range(len(ground_truth)) if index not in matched_truth]
    return matches, unmatched_predictions, unmatched_truth


def _pixel_box(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    return (
        float(box[0]) * width,
        float(box[1]) * height,
        float(box[2]) * width,
        float(box[3]) * height,
    )


def _normalized_box(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert an original-image pixel ``xyxy`` box to normalized ``xyxy``."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return (
        float(np.clip(float(box[0]) / width, 0.0, 1.0)),
        float(np.clip(float(box[1]) / height, 0.0, 1.0)),
        float(np.clip(float(box[2]) / width, 0.0, 1.0)),
        float(np.clip(float(box[3]) / height, 0.0, 1.0)),
    )


def _draw_cross(draw: ImageDraw.ImageDraw, point: Sequence[float], color: tuple[int, int, int]) -> None:
    x, y = (float(point[0]), float(point[1]))
    draw.line((x - 8, y, x + 8, y), fill=color, width=2)
    draw.line((x, y - 8, x, y + 8), fill=color, width=2)


def draw_sample(
    image: Image.Image,
    ground_truth: Sequence[Sequence[float]],
    predictions: Sequence[YOLODetection],
    matches: Sequence[dict],
    output: Path,
    image_name: str,
) -> None:
    """Draw GT green, prediction red, bottom-center blue, bottom-band orange."""
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    match_by_prediction = {int(item["prediction_index"]): item for item in matches}

    for truth_index, box in enumerate(ground_truth):
        xyxy = _pixel_box(box, width, height)
        draw.rectangle(xyxy, outline=(46, 204, 113), width=3)
        draw.text((xyxy[0] + 3, xyxy[1] + 3), f"GT {truth_index}", fill=(46, 204, 113))

    for prediction_index, prediction in enumerate(predictions):
        x1, y1, x2, y2 = prediction.bbox
        draw.rectangle((x1, y1, x2, y2), outline=(231, 76, 60), width=3)
        matched = match_by_prediction.get(prediction_index)
        suffix = "" if matched is None else f" IoU={float(matched['iou']):.2f}"
        draw.text(
            (x1 + 3, max(0.0, y1 + 3)),
            f"P{prediction_index} {prediction.confidence:.2f}{suffix}",
            fill=(231, 76, 60),
        )
        for point in prediction.bottom_band:
            px, py = (float(point[0]), float(point[1]))
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(241, 156, 18))
        _draw_cross(draw, prediction.bottom_center, (52, 152, 219))

    draw.rectangle((0, 0, min(width, 1400), 32), fill=(0, 0, 0))
    draw.text(
        (6, 8),
        f"{image_name} GT={len(ground_truth)} pred={len(predictions)}",
        fill="white",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def build_contact_sheet(
    image_paths: Iterable[Path],
    output: Path,
    max_tiles: int = 12,
    tile_size: tuple[int, int] = (420, 320),
    columns: int = 3,
) -> None:
    paths = list(image_paths)[:max_tiles]
    if not paths:
        return
    tile_width, tile_height = tile_size
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert("RGB"), tile_size)
        x = (index % columns) * tile_width + (tile_width - tile.width) // 2
        y = (index // columns) * tile_height + (tile_height - tile.height) // 2
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values else 0.0


def point_error_metrics(errors: Sequence[float]) -> dict[str, float]:
    """Summarise matched bbox bottom-contact errors in source-image pixels.

    The metric is intentionally computed only for one-to-one IoU matches. A
    false positive has no valid contact ground truth, while an unmatched
    ground-truth box is already represented by FN in the detector metrics.
    """
    values = np.asarray(list(errors), dtype=np.float64)
    if values.size == 0:
        return {
            "matches": 0.0,
            "mae_px": 0.0,
            "median_px": 0.0,
            "p95_px": 0.0,
            "pck10": 0.0,
            "pck25": 0.0,
        }
    return {
        "matches": float(values.size),
        "mae_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95.0)),
        "pck10": float(np.mean(values <= 10.0)),
        "pck25": float(np.mean(values <= 25.0)),
    }


def valid_contact_band(prediction: YOLODetection, width: int, height: int) -> bool:
    """Whether all proposed bottom-band pixels are valid source-image pixels."""
    points = np.asarray(prediction.bottom_band, dtype=np.float64)
    return bool(
        points.ndim == 2
        and points.shape[1] == 2
        and len(points) > 0
        and np.all(np.isfinite(points))
        and np.all(points[:, 0] >= 0.0)
        and np.all(points[:, 0] <= float(width - 1))
        and np.all(points[:, 1] >= 0.0)
        and np.all(points[:, 1] <= float(height - 1))
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=D_FIRE_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    source_class = parser.add_mutually_exclusive_group(required=True)
    source_class.add_argument(
        "--source-fire-class",
        "--fire-class",
        dest="source_fire_class",
        type=int,
        help=(
            "Verified fire class id in the original YOLO labels. "
            "--fire-class is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--model-fire-class",
        type=int,
        default=None,
        help=(
            "Fire class id in the checkpoint predictions. Defaults to the "
            "source id; use 0 for a --single-fire-class checkpoint."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--selection", choices=("head", "even", "diverse"), default="diverse")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "home_fire_yolo")
    parser.add_argument("--contact-sheet-max", type=int, default=12)
    args = parser.parse_args()

    if args.source_fire_class < 0:
        raise ValueError("--source-fire-class must be non-negative")
    model_fire_class = (
        args.source_fire_class
        if args.model_fire_class is None
        else args.model_fire_class
    )
    if model_fire_class < 0:
        raise ValueError("--model-fire-class must be non-negative")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in [0, 1]")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")

    dataset_root = resolve_dataset_root(args.dataset_root)
    image_root = dataset_root / args.split / "images"
    label_root = dataset_root / args.split / "labels"
    images = select_images(
        image_files(dataset_root, args.split),
        image_root,
        label_root,
        args.source_fire_class,
        args.max_images,
        args.selection,
    )
    if not images:
        raise RuntimeError(f"Không tìm thấy ảnh trong {image_root}")

    detector = HomeFireYOLO(
        args.checkpoint,
        fire_class=model_fire_class,
        threshold=args.threshold,
        device=args.device,
        imgsz=args.imgsz,
    )
    preview_paths: list[Path] = []
    rows: list[dict] = []
    all_ious: list[float] = []
    bottom_errors: list[float] = []
    latencies: list[float] = []
    total_predictions = 0
    total_ground_truth = 0
    valid_bottom_bands = 0
    total_tp = total_fp = total_fn = 0

    for index, image_path in enumerate(images):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        started = time.perf_counter()
        predictions = detector.detect_all(image)
        latency_ms = (time.perf_counter() - started) * 1000.0
        truth = parse_yolo_label(
            label_path_for(image_path, image_root, label_root),
            args.source_fire_class,
        )
        prediction_boxes = [prediction.bbox for prediction in predictions]
        prediction_boxes_normalized = [
            _normalized_box(box, image.width, image.height) for box in prediction_boxes
        ]
        matches, unmatched_predictions, unmatched_truth = greedy_match(
            truth,
            prediction_boxes_normalized,
            args.iou_threshold,
        )
        # A bbox is only a weak proxy for the fire-ground contact.  Evaluate
        # its bottom-center separately so a high IoU cannot hide a poor ray
        # origin for the downstream 3D stage.
        enriched_matches: list[dict] = []
        for match in matches:
            prediction = predictions[int(match["prediction_index"])]
            truth_box = truth[int(match["ground_truth_index"])]
            truth_pixels = _pixel_box(truth_box, image.width, image.height)
            gt_bottom = (
                float((truth_pixels[0] + truth_pixels[2]) * 0.5),
                float(truth_pixels[3]),
            )
            error_px = float(
                np.linalg.norm(
                    np.asarray(prediction.bottom_center, dtype=np.float64)
                    - np.asarray(gt_bottom, dtype=np.float64)
                )
            )
            bottom_errors.append(error_px)
            enriched = dict(match)
            enriched.update(
                {
                    "gt_bottom_center": [gt_bottom[0], gt_bottom[1]],
                    "pred_bottom_center": [
                        float(prediction.bottom_center[0]),
                        float(prediction.bottom_center[1]),
                    ],
                    "bottom_error_px": error_px,
                }
            )
            enriched_matches.append(enriched)
        matches = enriched_matches
        tp = len(matches)
        fp = len(unmatched_predictions)
        fn = len(unmatched_truth)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_predictions += len(predictions)
        total_ground_truth += len(truth)
        valid_bottom_bands += sum(
            valid_contact_band(prediction, image.width, image.height)
            for prediction in predictions
        )
        all_ious.extend(float(item["iou"]) for item in matches)
        latencies.append(float(latency_ms))

        preview_path = args.output_dir / "images" / f"sample_{index:04d}.jpg"
        draw_sample(image, truth, predictions, matches, preview_path, image_path.name)
        preview_paths.append(preview_path)
        rows.append(
            {
                "image": str(image_path),
                "ground_truth_boxes": [list(box) for box in truth],
                "predicted_boxes": [
                    {
                        "bbox_xyxy_pixels": [float(value) for value in prediction.bbox],
                        "confidence": float(prediction.confidence),
                        "class_id": int(prediction.class_id),
                        "bottom_center": [float(value) for value in prediction.bottom_center],
                        "bottom_band": prediction.bottom_band.astype(float).tolist(),
                    }
                    for prediction in predictions
                ],
                "matched_pairs": matches,
                "unmatched_predictions": unmatched_predictions,
                "unmatched_ground_truth": unmatched_truth,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "latency_ms": float(latency_ms),
            }
        )

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    bottom_metrics = point_error_metrics(bottom_errors)
    summary = {
        "branch": "home_fire_yolo_bbox",
        "dataset_root": str(dataset_root),
        "split": args.split,
        "images": len(rows),
        "source_fire_class": args.source_fire_class,
        "model_fire_class": model_fire_class,
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "imgsz": args.imgsz,
        "selection": args.selection,
        "ground_truth_boxes": total_ground_truth,
        "predicted_boxes": total_predictions,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "median_iou": float(np.median(all_ious)) if all_ious else 0.0,
        "p95_iou": percentile(all_ious, 95.0),
        "bottom_center_matches": int(bottom_metrics["matches"]),
        "bottom_center_mae_px": bottom_metrics["mae_px"],
        "bottom_center_median_px": bottom_metrics["median_px"],
        "bottom_center_p95_px": bottom_metrics["p95_px"],
        "bottom_center_pck10": bottom_metrics["pck10"],
        "bottom_center_pck25": bottom_metrics["pck25"],
        "valid_bottom_band_predictions": int(valid_bottom_bands),
        "bottom_band_valid_rate": float(
            valid_bottom_bands / max(1, total_predictions)
        ),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_median_ms": float(np.median(latencies)) if latencies else 0.0,
        "latency_p95_ms": percentile(latencies, 95.0),
        "contact_sheet": str(args.output_dir / "comparison_contact_sheet.png"),
        "rows": rows,
        "note": (
            "2D bbox benchmark only. Bottom-center and bottom-band are weak "
            "geometric hypotheses; physical 3D error requires measured "
            "calibration, metric mesh/GridMap and 3D ground truth."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    build_contact_sheet(
        preview_paths,
        args.output_dir / "comparison_contact_sheet.png",
        max_tiles=max(1, args.contact_sheet_max),
    )

    print(f"dataset_root={dataset_root}")
    print(
        f"loaded_images={len(rows)} "
        f"source_fire_class={args.source_fire_class} "
        f"model_fire_class={model_fire_class}"
    )
    print("branch    TP     FP     FN     precision  recall  F1      meanIoU  medianIoU  P95IoU")
    print(
        f"YOLO      {total_tp:<6}{total_fp:<6}{total_fn:<6}"
        f"{precision:<10.3f}{recall:<8.3f}{f1:<8.3f}"
        f"{summary['mean_iou']:<9.3f}{summary['median_iou']:<11.3f}{summary['p95_iou']:.3f}"
    )
    print(
        "bottom-center (matched boxes only): "
        f"n={int(bottom_metrics['matches'])} "
        f"MAE={bottom_metrics['mae_px']:.2f}px "
        f"median={bottom_metrics['median_px']:.2f}px "
        f"P95={bottom_metrics['p95_px']:.2f}px "
        f"PCK@10={bottom_metrics['pck10']:.3f} "
        f"PCK@25={bottom_metrics['pck25']:.3f}"
    )
    print(
        "bottom-band validity: "
        f"{valid_bottom_bands}/{total_predictions} "
        f"({summary['bottom_band_valid_rate']:.3f})"
    )
    print(
        "latency_ms: "
        f"mean={summary['latency_mean_ms']:.2f} "
        f"median={summary['latency_median_ms']:.2f} "
        f"P95={summary['latency_p95_ms']:.2f}"
    )
    print(f"saved_summary={summary_path}")
    print(f"saved_contact_sheet={args.output_dir / 'comparison_contact_sheet.png'}")
    print(f"saved_images={len(preview_paths)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

