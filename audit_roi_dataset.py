"""Audit the dataset used by the legacy detector -> ROI workflow.

This is a read-only check.  It verifies that ROI training receives real
coarse points from the upstream ``best.pth`` manifest, that train/validation/
test splits are resolved consistently, and that every coarse point lies close
enough to the ground-truth point for the configured square ROI to contain it.

Example:

    .venv\\Scripts\\python.exe audit_roi_dataset.py

The report is JSON so it can be attached to the weekly lab report without
copying values from terminal output.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from project_paths import CCTV_DATASET

from narrow_localizer import _index_coarse_manifest, _lookup_coarse_manifest
from train_week6 import load_records, split_records


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)) if values else None,
        "median": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": float(max(values)) if values else None,
        "within_10px": sum(value <= 10.0 for value in values),
        "within_25px": sum(value <= 25.0 for value in values),
        "within_50px": sum(value <= 50.0 for value in values),
    }


def _record_key(path_text: str) -> str:
    normalized = str(path_text).replace("\\", "/").lower()
    marker = "/img_data/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized


def _audit_split(records: Iterable[Any], indexed_manifest: dict, roi_fraction: float) -> dict[str, Any]:
    rows = list(records)
    positive = [record for record in rows if record.has_fire]
    negative = [record for record in rows if not record.has_fire]
    errors: list[float] = []
    confidences: list[float] = []
    missing: list[str] = []
    outside_roi: list[str] = []

    for record in positive:
        item = _lookup_coarse_manifest(indexed_manifest, record.image_path)
        if not isinstance(item, dict) or item.get("point") is None:
            missing.append(record.image_path)
            continue
        try:
            point = item["point"]
            size = item.get("size")
            if not isinstance(size, (list, tuple)) or len(size) < 2:
                with Image.open(record.image_path) as image:
                    width, height = image.size
            else:
                width, height = float(size[0]), float(size[1])
            coarse_x, coarse_y = float(point[0]), float(point[1])
            gt_x, gt_y = record.x_norm * width, record.y_norm * height
            error = math.hypot(coarse_x - gt_x, coarse_y - gt_y)
            errors.append(error)
            if item.get("confidence") is not None:
                confidences.append(float(item["confidence"]))
            half_side = min(width, height) * float(roi_fraction) / 2.0
            if abs(coarse_x - gt_x) > half_side or abs(coarse_y - gt_y) > half_side:
                outside_roi.append(record.image_path)
        except (TypeError, ValueError, IndexError, OSError):
            missing.append(record.image_path)

    return {
        "records": len(rows),
        "positive": len(positive),
        "negative": len(negative),
        "positive_manifest_points": len(errors),
        "positive_coverage": len(errors) / len(positive) if positive else 0.0,
        "coarse_error_px": _summary(errors),
        "coarse_confidence": _summary(confidences),
        "outside_configured_roi": len(outside_roi),
        "missing_or_invalid_examples": missing[:10],
        "outside_roi_examples": outside_roi[:10],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--coarse-manifest", type=Path, default=root / "week6_roi_result" / "coarse_manifest.json")
    parser.add_argument("--output", type=Path, default=root / "output" / "roi_dataset_audit.json")
    parser.add_argument("--roi-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records, load_stats = load_records(args.labels, args.dataset_root)
    splits = split_records(records, seed=args.seed)
    raw_manifest = json.loads(args.coarse_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        raise ValueError("coarse manifest must contain a JSON object")
    indexed_manifest = _index_coarse_manifest(raw_manifest)

    keys = [_record_key(record.image_path) for record in records]
    duplicate_keys = len(keys) - len(set(keys))
    split_report = {
        name: _audit_split(items, indexed_manifest, args.roi_fraction)
        for name, items in splits.items()
    }
    report = {
        "protocol": {
            "workflow": "best.pth -> coarse point -> ROI refiner",
            "seed": args.seed,
            "roi_fraction": args.roi_fraction,
            "manifest_alias_index": True,
        },
        "paths": {
            "labels": str(args.labels),
            "dataset_root": str(args.dataset_root),
            "coarse_manifest": str(args.coarse_manifest),
        },
        "load_stats": load_stats,
        "manifest": {
            "raw_entries": len(raw_manifest),
            "indexed_aliases": len(indexed_manifest),
        },
        "leakage_checks": {
            "unique_record_keys": len(set(keys)),
            "duplicate_record_keys": duplicate_keys,
            "split_paths": {
                name: [str(record.image_path) for record in items]
                for name, items in splits.items()
            },
        },
        "splits": split_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"records={len(records)} positive={sum(record.has_fire for record in records)}")
    print(f"manifest_entries={len(raw_manifest)} indexed_aliases={len(indexed_manifest)}")
    print(f"duplicate_record_keys={duplicate_keys}")
    for name, values in split_report.items():
        error = values["coarse_error_px"]
        print(
            f"{name}: records={values['records']} positive={values['positive']} "
            f"coverage={values['positive_coverage']:.3f} "
            f"MAE={error['mean'] if error['mean'] is not None else float('nan'):.2f}px "
            f"P95={error['p95'] if error['p95'] is not None else float('nan'):.2f}px "
            f"outside_roi={values['outside_configured_roi']}"
        )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
