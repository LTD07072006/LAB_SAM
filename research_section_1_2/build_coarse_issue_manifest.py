"""Re-label every coarse issue using the leakage-safe workflow split.

The detector manifest stores source folders (train/test), while evaluation
uses a group-aware train/val/test split. This script maps records to that exact
workflow split and emits issue buckets for all positive images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import CCTV_DATASET
from train_week6 import load_records, split_records


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, load_stats = load_records(args.labels, args.dataset_root)
    split_map: dict[str, str] = {}
    for split, values in split_records(records, seed=args.seed).items():
        for record in values:
            split_map[str(Path(record.image_path).resolve()).lower()] = split

    rows: list[dict[str, Any]] = []
    for record in records:
        key = str(Path(record.image_path).resolve()).lower()
        item = manifest.get(record.image_path) or manifest.get(str(Path(record.image_path)))
        if not isinstance(item, dict):
            # Portable fallback by normalised split/class/name suffix.
            suffix = str(Path(record.image_path)).replace("\\", "/").lower()
            item = next((value for path, value in manifest.items()
                         if str(path).replace("\\", "/").lower().endswith(suffix)), {})
        if not record.has_fire or not isinstance(item, dict):
            continue
        raw_error = item.get("raw_error_px")
        if raw_error is None:
            # Derive the audit coordinate in the actual image frame.  The
            # first implementation assumed 224x224, which is true for the
            # current CCTV export but silently mis-scales future inputs.
            try:
                from PIL import Image
                with Image.open(record.image_path) as image:
                    image_width, image_height = image.size
            except (OSError, ValueError):
                image_width, image_height = 224.0, 224.0
            gt = np.asarray([record.x_norm * float(image_width), record.y_norm * float(image_height)])
            point = item.get("raw_pixel", item.get("point"))
            raw_error = None if point is None else float(np.linalg.norm(np.asarray(point) - gt))
        raw_error = float(raw_error) if raw_error is not None else None
        stability = item.get("stability_px")
        confidence = float(item.get("confidence", 0.0))
        workflow_split = split_map.get(key, "unknown")
        categories: list[str] = []
        if raw_error is None:
            categories.append("missing_coarse")
        else:
            for threshold in (5, 8, 10, 12, 15, 20, 25):
                if raw_error > threshold:
                    categories.append(f"error_gt_{threshold}px")
        if stability is not None and float(stability) >= args.stability_px:
            categories.append("tta_unstable")
        if confidence >= args.high_confidence and raw_error is not None and raw_error >= args.high_conf_error_px:
            categories.append("high_confidence_wrong_localization")
        row = {
            "image": Path(record.image_path).name,
            "image_path": str(record.image_path),
            "workflow_split": workflow_split,
            "source_split": record.source_split,
            "raw_pixel": item.get("raw_pixel", item.get("point")),
            "guarded_pixel": item.get("guarded_pixel", item.get("point")),
            "ground_truth_pixel": None,
            "raw_error_px": raw_error,
            "guarded_error_px": item.get("guarded_error_px"),
            "confidence": confidence,
            "stability_px": stability,
            "guard_replaced": bool(item.get("guard_replaced", False)),
            "categories": categories,
            "train_eligible": workflow_split == "train",
            "test_only": workflow_split == "test",
        }
        if row["ground_truth_pixel"] is None:
            try:
                from PIL import Image
                with Image.open(record.image_path) as image:
                    row["ground_truth_pixel"] = [record.x_norm * image.width, record.y_norm * image.height]
            except (OSError, ValueError):
                row["ground_truth_pixel"] = [record.x_norm * 224.0, record.y_norm * 224.0]
        rows.append(row)

    by_split = {split: [row for row in rows if row["workflow_split"] == split] for split in ("train", "val", "test")}
    issue_rows = [row for row in rows if row["categories"]]
    hard_tail_train = [row for row in by_split["train"] if row["raw_error_px"] is not None and row["raw_error_px"] >= args.hard_tail_min_px]
    summary = {
        "format": "LAB_SAM.coarse_issue_manifest.v2",
        "protocol": {"seed": args.seed, "workflow_split_is_group_aware": True,
                      "ground_truth_used_for_audit_only": True, "source_labels_unchanged": True},
        "counts": {
            "positive": len(rows), "issues": len(issue_rows),
            "hard_tail_train": len(hard_tail_train),
            "by_workflow_split": {key: len(value) for key, value in by_split.items()},
            "issues_by_workflow_split": {key: sum(bool(row["categories"]) for row in value) for key, value in by_split.items()},
        },
        "issue_thresholds": {"stability_px": args.stability_px, "high_confidence": args.high_confidence,
                             "high_conf_error_px": args.high_conf_error_px, "hard_tail_min_px": args.hard_tail_min_px},
        "records": rows,
        "issue_records": issue_rows,
        "hard_tail_train": hard_tail_train,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coarse_issue_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "coarse_issues_all.json").write_text(json.dumps(issue_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "hard_tail_train_only.json").write_text(json.dumps(hard_tail_train, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": summary["counts"], "saved": str(args.output_dir)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "output" / "section_1_2_pre_roi_20261010" / "filtered_outliers_final2" / "coarse_manifest_brightness3_guarded.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "section_1_2_pre_roi_20261010" / "coarse_issue_manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-px", type=float, default=6.0)
    parser.add_argument("--high-confidence", type=float, default=0.99)
    parser.add_argument("--high-conf-error-px", type=float, default=10.0)
    parser.add_argument("--hard-tail-min-px", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
