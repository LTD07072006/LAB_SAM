"""Audit the project's classification/regression label contract.

Semantic contract for the point detector::

    has_fire=1 -> fire and a valid normalized p_fire point is required
    has_fire=0 -> no fire; p_fire is ignored by regression

For the separate Home Fire YOLO branch, original class 1 is fire and original
class 0 is no fire. The one-class YOLO training view remaps original class 1
to model class 0 and leaves no-fire images with empty label files.

This script is stdlib-only and can run before installing torch, timm or
Ultralytics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def _binary(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number not in (0.0, 1.0):
        return None
    return int(number)


def _valid_point(value: Any) -> bool:
    try:
        point = list(value)
        values = [float(point[0]), float(point[1])]
    except (TypeError, ValueError, IndexError):
        return False
    return all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in values)


def _resolve_image(raw_path: str, root: Path) -> Optional[Path]:
    raw = Path(raw_path)
    candidates = [raw]
    normalised = raw_path.replace("\\", "/")
    match = re.search(r"(?:^|/)img_data/(train|val|test)/(.+)$", normalised, re.IGNORECASE)
    if match:
        split, relative = match.groups()
        candidates.extend(
            [
                root / split / relative,
                root / "img_data" / split / relative,
                root / "data" / "data" / "img_data" / split / relative,
                root / "data" / "img_data" / split / relative,
            ]
        )
    candidates.append(root / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = [path for path in root.rglob(raw.name) if path.is_file()]
    return matches[0].resolve() if len(matches) == 1 else None


def audit_json(labels_path: Path, dataset_root: Optional[Path]) -> dict[str, Any]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("labels JSON must contain a list")
    counts: Counter[str] = Counter()
    unresolved = 0
    resolved_paths: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            counts["invalid_record"] += 1
            continue
        if "has_fire" not in item:
            counts["missing_has_fire"] += 1
            continue
        label = _binary(item.get("has_fire"))
        if label is None:
            counts["non_binary_has_fire"] += 1
            continue
        counts[f"class_{label}"] += 1
        if label == 1:
            if _valid_point(item.get("p_fire")):
                counts["positive_valid_p_fire"] += 1
            else:
                counts["positive_missing_or_invalid_p_fire"] += 1
        else:
            raw_point = item.get("p_fire")
            if raw_point is not None:
                if _valid_point(raw_point):
                    values = [float(raw_point[0]), float(raw_point[1])]
                    if any(abs(value) > 1e-12 for value in values):
                        counts["negative_nonzero_p_fire_ignored"] += 1
                else:
                    counts["negative_invalid_p_fire_ignored"] += 1
        if dataset_root is not None:
            resolved = _resolve_image(str(item.get("image_path", "")), dataset_root)
            if resolved is None:
                unresolved += 1
            else:
                resolved_paths.add(str(resolved).lower())
    return {
        "path": str(labels_path.resolve()),
        "records": len(payload),
        "counts": dict(counts),
        "unresolved_images": unresolved,
        "unique_resolved_images": len(resolved_paths),
        "contract_ok": (
            counts["non_binary_has_fire"] == 0
            and counts["missing_has_fire"] == 0
            and counts["positive_missing_or_invalid_p_fire"] == 0
        ),
    }


def _resolve_yolo_root(value: Path) -> Path:
    def has_layout(path: Path) -> bool:
        return all(
            (path / split / "images").is_dir() and (path / split / "labels").is_dir()
            for split in SPLITS
        )

    root = value.resolve()
    if has_layout(root):
        return root
    candidates = [path for path in root.rglob("*") if path.is_dir() and has_layout(path)]
    if not candidates:
        raise FileNotFoundError(f"YOLO layout not found below {root}")
    return sorted(candidates, key=lambda path: (len(path.parts), str(path).lower()))[0]


def audit_yolo(root_value: Path, source_fire_class: int) -> dict[str, Any]:
    root = _resolve_yolo_root(root_value)
    counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for split in SPLITS:
        label_root = root / split / "labels"
        image_root = root / split / "images"
        counts[f"{split}_images"] = sum(
            1 for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_paths = [path for path in label_root.rglob("*.txt") if path.is_file()]
        counts[f"{split}_label_files"] = len(label_paths)
        for label_path in label_paths:
            rows = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            if not rows:
                counts["empty_label_files"] += 1
            for row in rows:
                fields = row.split()
                if len(fields) != 5:
                    counts["malformed_rows"] += 1
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except (TypeError, ValueError):
                    counts["malformed_rows"] += 1
                    continue
                class_counts[str(class_id)] += 1
                if class_id == source_fire_class:
                    counts["source_fire_boxes"] += 1
                if class_id == 0:
                    counts["source_class_0_boxes"] += 1
                if class_id == 1:
                    counts["source_class_1_boxes"] += 1
                if any(not math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
                    counts["invalid_geometry"] += 1
    return {
        "root": str(root),
        "source_fire_class": source_fire_class,
        "class_bbox_counts": dict(class_counts),
        "counts": dict(counts),
        "recommended_one_class_mapping": {
            "source_fire_class": source_fire_class,
            "model_fire_class": 0,
            "source_no_fire_class": 0 if source_fire_class == 1 else 1,
            "negative_images_use_empty_label": True,
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=project_root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=project_root / "fire-detection-from-cctv")
    parser.add_argument("--yolo-root", type=Path, default=project_root / "home-fire-dataset")
    parser.add_argument("--source-fire-class", type=int, default=1)
    parser.add_argument("--output", type=Path, default=project_root / "working" / "fire_label_audit.json")
    args = parser.parse_args()

    report = {
        "classification_contract": {"fire": 1, "no_fire": 0},
        "regression_contract": "p_fire is required only when has_fire=1; coordinate loss is masked for has_fire=0",
        "json_labels": audit_json(args.labels.resolve(), args.dataset_root.resolve()),
        "home_fire_yolo": audit_yolo(args.yolo_root, args.source_fire_class),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
