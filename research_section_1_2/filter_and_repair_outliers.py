"""Filter coarse outliers and build label-preserving repair manifests.

Outputs are intentionally additive:

* ``coarse_manifest_brightness3_guarded.json``: raw and optional temporal
  guarded points for ROI training/inference;
* ``hard_tail_train.json``: train positives with high detector error and a
  sampling weight, for hard-tail training;
* ``outlier_review.json``: candidates requiring independent annotation review.

Ground-truth labels are read only for audit metrics.  This script never edits
the source labels JSON and never uses GT to choose the guarded point.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_detector import FireDetector
from project_paths import CCTV_DATASET
from temporal_outlier_guard import TemporalCoarseOutlierGuard
from train_week6 import load_records


def _frame_number(path: str) -> int:
    match = re.search(r"img_(\d+)", Path(path).name.lower())
    return int(match.group(1)) if match else -1


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _point(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(point) < 2 or not np.isfinite(point[:2]).all():
        return None
    return point[:2].copy()


def _error(point: Any, gt: np.ndarray) -> Optional[float]:
    p = _point(point)
    return None if p is None else float(np.linalg.norm(p - gt))


def _make_contact_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    tiles: list[Image.Image] = []
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            continue
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((448, 448), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        for key, colour, label in (
            ("ground_truth_pixel", "lime", "GT"),
            ("raw_pixel", "red", "raw"),
            ("guarded_pixel", "cyan", "guard"),
        ):
            point = _point(row.get(key))
            if point is None:
                continue
            x, y = float(point[0]) * 2.0, float(point[1]) * 2.0
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=colour, width=4)
            draw.line((x - 14, y, x + 14, y), fill=colour, width=2)
            draw.line((x, y - 14, x, y + 14), fill=colour, width=2)
            draw.text((x + 11, y - 13), label, fill=colour, stroke_width=2, stroke_fill="black")
        draw.rectangle((0, 0, 448, 25), fill="black")
        draw.text((5, 5), f"{row['image']}  raw_err={row.get('raw_error_px')}", fill="white")
        tiles.append(image)
    if not tiles:
        return
    columns = 4
    output = Image.new("RGB", (columns * 448, math.ceil(len(tiles) / columns) * 448), "black")
    for index, tile in enumerate(tiles):
        output.paste(tile, ((index % columns) * 448, (index // columns) * 448))
    output.save(path, quality=92)


def run(args: argparse.Namespace) -> dict[str, Any]:
    records, load_stats = load_records(args.labels, args.dataset_root)
    detector = FireDetector(
        args.checkpoint, device=args.device, threshold=args.threshold,
        use_amp=False, tta_mode="brightness3", max_stability_px=None,
    )
    detector.warmup(repeats=1)
    # Guard state is independent per source split.  A no-fire record naturally
    # clears the previous point because update(None) resets after one miss.
    guards: dict[str, TemporalCoarseOutlierGuard] = {}
    last_frame: dict[str, int] = {}
    ordered = sorted(records, key=lambda r: (r.source_split, _frame_number(r.image_path), r.image_path))
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered, 1):
        image = Image.open(record.image_path).convert("RGB")
        result = detector.detect(image, warmup=False)
        raw = result.raw_pixel if result.raw_pixel is not None else result.pixel
        # A raw point retained for diagnostics on a no-fire frame must never
        # bridge two temporal fire segments.
        guard_input = raw if result.detected else None
        split = str(record.source_split)
        guard = guards.setdefault(
            split,
            TemporalCoarseOutlierGuard(
                max_jump_px=args.max_jump_px,
                max_frame_mean_abs_diff=args.max_frame_mean_abs_diff,
                min_stability_px=args.min_stability_px,
            ),
        )
        frame_number = _frame_number(record.image_path)
        guard_enabled = split in set(args.guard_splits)
        if split in last_frame and (frame_number < 0 or last_frame[split] < 0 or frame_number != last_frame[split] + 1):
            guard.reset()
        last_frame[split] = frame_number
        guarded = (
            guard.update(image, guard_input, stability_px=result.stability_px)
            if guard_enabled else
            type("GuardResult", (), {"point": raw if result.detected else None,
                                     "replaced": False, "reason": "guard_disabled_for_split",
                                     "jump_px": None, "frame_diff": None})()
        )
        gt = np.asarray([record.x_norm * image.width, record.y_norm * image.height], dtype=np.float64)
        raw_error = _error(raw, gt) if record.has_fire else None
        guarded_error = _error(guarded.point, gt) if record.has_fire else None
        rows.append({
            "split": split, "image": Path(record.image_path).name,
            "image_path": str(record.image_path), "frame_number": _frame_number(record.image_path),
            "has_fire": bool(record.has_fire),
            "ground_truth_pixel": gt.tolist() if record.has_fire else None,
            "raw_pixel": None if raw is None else [float(v) for v in raw],
            "guarded_pixel": None if guarded.point is None else [float(v) for v in guarded.point],
            "point": None if guarded.point is None else [float(v) for v in guarded.point],
            "confidence": float(result.confidence), "detected": bool(result.detected),
            "stability_px": result.stability_px,
            "position_reliability": result.position_reliability,
            "raw_error_px": raw_error, "guarded_error_px": guarded_error,
            "guard_replaced": bool(guarded.replaced), "guard_reason": guarded.reason,
            "jump_px": guarded.jump_px, "frame_diff": guarded.frame_diff,
            "localization_status": "guarded" if guarded.replaced else str(result.localization_status),
            "size": [int(image.width), int(image.height)],
        })
        if index == 1 or index == len(ordered) or index % 50 == 0:
            print(f"processed={index}/{len(ordered)}", flush=True)

    positives = [row for row in rows if row["has_fire"]]
    train_positive = [row for row in positives if row["split"] == "train"]
    train_errors = np.asarray([row["raw_error_px"] for row in train_positive if row["raw_error_px"] is not None], dtype=np.float64)
    tail_cut = float(np.percentile(train_errors, args.hard_tail_quantile)) if len(train_errors) else float("inf")
    hard_tail = []
    for row in train_positive:
        error = row["raw_error_px"]
        is_tail = error is not None and (float(error) >= max(args.hard_tail_min_px, tail_cut))
        if is_tail:
            hard_tail.append({
                "image_path": row["image_path"], "split": row["split"],
                "point": row["point"], "raw_point": row["raw_pixel"],
                "ground_truth_pixel": row["ground_truth_pixel"],
                "error_px": error, "confidence": row["confidence"],
                "stability_px": row["stability_px"],
                "sample_weight": float(1.0 + min(float(error) / max(args.hard_tail_min_px, 1e-6), args.max_tail_weight - 1.0)),
            })
    review = [row for row in positives if (
        (row["raw_error_px"] is not None and row["raw_error_px"] >= args.review_error_px)
        or row["guard_replaced"]
    )]
    review.sort(key=lambda row: float(row["raw_error_px"] or 0.0), reverse=True)
    output_rows = {row["image_path"]: row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "coarse_manifest_brightness3_guarded.json"
    manifest_path.write_text(json.dumps(_json(output_rows), indent=2, ensure_ascii=False), encoding="utf-8")
    tail_path = args.output_dir / "hard_tail_train.json"
    tail_path.write_text(json.dumps(_json({
        "format": "LAB_SAM.hard_tail_train.v1", "quantile": args.hard_tail_quantile,
        "tail_cut_px": tail_cut, "min_px": args.hard_tail_min_px,
        "max_weight": args.max_tail_weight, "records": hard_tail,
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    review_path = args.output_dir / "outlier_review.json"
    review_path.write_text(json.dumps(_json({
        "format": "LAB_SAM.outlier_review.v1",
        "warning": "No Ground Truth was modified; independent annotation is required before any label change.",
        "records": review,
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    _make_contact_sheet(review[:args.contact_sheet_limit], args.output_dir / "outlier_review_contact_sheet.jpg")
    summary = {
        "format": "LAB_SAM.filter_repair_outliers.v1",
        "protocol": {
            "records": len(rows), "tta_mode": "brightness3", "guard_is_label_free": True,
            "ground_truth_used_only_for_audit": True, "source_labels_unchanged": True,
            "max_jump_px": args.max_jump_px, "max_frame_mean_abs_diff": args.max_frame_mean_abs_diff,
            "min_stability_px": args.min_stability_px,
            "guard_splits": list(args.guard_splits),
        },
        "dataset": {"load_stats": load_stats, "positive_records": len(positives)},
        "audit": {
            "raw_error_px": _stats([row["raw_error_px"] for row in positives if row["raw_error_px"] is not None]),
            "guarded_error_px": _stats([row["guarded_error_px"] for row in positives if row["guarded_error_px"] is not None]),
            "guard_replaced": int(sum(row["guard_replaced"] for row in rows)),
            "guard_replaced_positive": int(sum(row["guard_replaced"] and row["has_fire"] for row in rows)),
            "hard_tail_train_count": len(hard_tail), "hard_tail_cut_px": tail_cut,
            "review_count": len(review),
        },
        "artifacts": {"manifest": str(manifest_path), "hard_tail": str(tail_path),
                       "review": str(review_path), "contact_sheet": str(args.output_dir / "outlier_review_contact_sheet.jpg")},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json(summary), indent=2))
    return summary


def _stats(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"count": 0, "mae_px": None, "median_px": None, "p95_px": None, "max_px": None, "outlier_gt25": 0}
    return {"count": int(len(x)), "mae_px": float(x.mean()), "median_px": float(np.median(x)),
            "p95_px": float(np.percentile(x, 95)), "max_px": float(x.max()),
            "outlier_gt25": int(np.sum(x > 25.0))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "section_1_2_pre_roi_20261010" / "filtered_outliers")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-jump-px", type=float, default=20.0)
    parser.add_argument("--max-frame-mean-abs-diff", type=float, default=25.0)
    parser.add_argument("--min-stability-px", type=float, default=6.0)
    parser.add_argument("--guard-splits", nargs="+", default=("test",),
                        help="Source splits known to be ordered video frames; default: test")
    parser.add_argument("--hard-tail-quantile", type=float, default=90.0)
    parser.add_argument("--hard-tail-min-px", type=float, default=12.0)
    parser.add_argument("--max-tail-weight", type=float, default=5.0)
    parser.add_argument("--review-error-px", type=float, default=25.0)
    parser.add_argument("--contact-sheet-limit", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
