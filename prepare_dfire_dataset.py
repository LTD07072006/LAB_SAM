"""Prepare an optional D-Fire detector dataset without touching old data.

The script only reads ``datasets/D-Fire`` (or ``--source``) and writes a new
manifest under ``working/dfire_detector``. It never modifies ``archive.zip``,
the existing CCTV datasets, video folders, checkpoints or Week 6 outputs.

D-Fire is an object-detection dataset, so it can improve classification and
coarse bounding-box training. It does not contain the project's ``p_fire``
bottom-contact labels or metric XYZ values and must not be mixed into the ROI
or final 3D test set without additional annotation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional

from project_paths import D_FIRE_ROOT, D_FIRE_ZIP


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _find_label(image: Path) -> Optional[Path]:
    """Find a YOLO txt annotation next to common D-Fire image layouts."""

    candidates = [
        image.with_suffix(".txt"),
        Path(str(image).replace("\\images\\", "\\labels\\")).with_suffix(".txt"),
        Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_yolo(label_path: Optional[Path]) -> list[dict[str, Any]]:
    if label_path is None:
        return []
    boxes: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id, x_center, y_center, width, height = map(float, fields)
        except ValueError:
            continue
        values = [class_id, x_center, y_center, width, height]
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            continue
        boxes.append({
            "class_id": int(class_id),
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "line": line_number,
        })
    return boxes


def _split_for(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for split in ("train", "val", "valid", "test"):
        if split in parts:
            return "val" if split == "valid" else split
    return "all"


def build_manifest(source: Path, output: Path, copy_images: bool = False) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"D-Fire source not found: {source}. Put the extracted dataset there "
            "or pass --source explicitly."
        )
    image_root = output / "images" if copy_images else source
    records: list[dict[str, Any]] = []
    skipped = {"no_label": 0, "bad_label": 0, "no_fire_class": 0}
    for image in sorted(_iter_images(source)):
        label = _find_label(image)
        if label is None:
            skipped["no_label"] += 1
            continue
        boxes = _parse_yolo(label)
        if not boxes:
            skipped["bad_label"] += 1
            continue
        # D-Fire's public object labels are fire/smoke. Keep all boxes but
        # identify fire-like classes using the conventional class-0 default;
        # verify names.yaml/data.yaml before training a particular branch.
        fire_boxes = [box for box in boxes if box["class_id"] == 0]
        if not fire_boxes:
            skipped["no_fire_class"] += 1
        relative = image.relative_to(source).as_posix()
        target_image = image_root / Path(relative)
        if copy_images:
            target_image.parent.mkdir(parents=True, exist_ok=True)
            if not target_image.exists():
                shutil.copy2(image, target_image)
        records.append({
            "sample_id": f"dfire_{len(records):07d}",
            "image_path": str(target_image if copy_images else image),
            "source_image": str(image),
            "label_path": str(label),
            "split": _split_for(image),
            "has_fire": int(bool(fire_boxes)),
            "boxes_yolo": boxes,
            "p_fire": None,
            "fire_xyz_world": None,
            "annotation_source": "D-Fire YOLO object annotation",
        })
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "D-Fire",
        "source": str(source),
        "output": str(output),
        "records": len(records),
        "positive_fire_records": sum(record["has_fire"] for record in records),
        "skipped": skipped,
        "warning": "2D object labels only; p_fire and metric 3D XYZ are intentionally null.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=D_FIRE_ROOT,
        help="Extracted D-Fire root; use prepare_home_fire_dataset.py for a ZIP",
    )
    parser.add_argument("--zip", type=Path, default=None, help="Optional D-Fire ZIP; creates a manifest without extracting")
    parser.add_argument("--output", type=Path, default=root / "working" / "dfire_detector")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into the new working folder")
    args = parser.parse_args()
    if args.zip is None and not args.source.exists() and D_FIRE_ZIP.is_file():
        args.zip = D_FIRE_ZIP
    if args.zip is not None:
        from prepare_home_fire_dataset import build_manifests

        summary_path = args.output / "summary.json"
        summary = build_manifests(
            archive=args.zip,
            root=None,
            manifest_out=args.output / "manifest.jsonl",
            weak_out=args.output / "weak_points.jsonl",
            summary_out=summary_path,
            preview_out=args.output / "preview.jpg",
            preview_count=18,
            fire_class=1,
            weak_weight=0.25,
            seed=42,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    summary = build_manifest(args.source, args.output, copy_images=args.copy_images)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
