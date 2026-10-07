"""Audit and prepare the 1.9 GB Home Fire YOLO dataset.

This dataset is deliberately kept separate from ``dataset_labels (1).json``.
Its labels are bounding boxes, while the main 2D-to-3D pipeline needs a
calibrated bottom/contact point ``p_fire``.  The script therefore produces:

* an image-level JSONL manifest with every YOLO box;
* a weak point JSONL manifest where a box becomes its bottom centre;
* a compact summary and an optional contact sheet for checking class meaning.

The script works directly from an extracted dataset root or from the ZIP
archive, so a 1.9 GB duplicate is not created merely to inspect it.

Examples::

    python prepare_home_fire_dataset.py --zip home-fire-dataset.zip \
        --manifest-out working/home_fire_manifest.jsonl \
        --weak-out working/home_fire_weak_points.jsonl \
        --preview-out working/home_fire_preview.jpg

For this project the original annotation convention is ``1 = fire`` and
``0 = no fire``.  Therefore use ``--fire-class 1`` when creating weak contact
points.  This option refers to the original dataset class; it is not the
class id of the later one-class YOLO checkpoint (that checkpoint uses
``0 = fire`` after remapping).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from project_paths import D_FIRE_ZIP

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_IMAGES_RE = re.compile(r"(?:^|/)(train|val|test)/images/(.+)$", re.IGNORECASE)
SPLIT_LABELS_RE = re.compile(r"(?:^|/)(train|val|test)/labels/(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageItem:
    split: str
    relative_name: str
    image_path: Optional[Path] = None
    archive_member: Optional[str] = None


def _normalise_member(value: str) -> str:
    return value.replace("\\", "/").lstrip("/")


def _json_number(value: float) -> float:
    # Avoid serialising -0.0 and keep manifests readable.
    return float(round(0.0 if abs(value) < 1e-12 else value, 8))


def _find_split_suffix(path_text: str, kind: str) -> Optional[Tuple[str, str]]:
    pattern = SPLIT_IMAGES_RE if kind == "images" else SPLIT_LABELS_RE
    match = pattern.search(_normalise_member(path_text))
    if not match:
        return None
    return match.group(1).lower(), match.group(2)


def _iter_root_items(root: Path) -> Tuple[List[ImageItem], Dict[Tuple[str, str], Path]]:
    images: List[ImageItem] = []
    labels: Dict[Tuple[str, str], Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        image_match = _find_split_suffix(str(path.relative_to(root)), "images")
        if image_match and path.suffix.lower() in IMAGE_EXTENSIONS:
            split, relative_name = image_match
            images.append(ImageItem(split, relative_name, image_path=path.resolve()))
            continue
        label_match = _find_split_suffix(str(path.relative_to(root)), "labels")
        if label_match and path.suffix.lower() == ".txt":
            split, relative_name = label_match
            labels[(split, Path(relative_name).with_suffix("").as_posix().lower())] = path.resolve()
    return images, labels


def _iter_zip_items(archive: Path) -> Tuple[List[ImageItem], Dict[Tuple[str, str], str]]:
    images: List[ImageItem] = []
    labels: Dict[Tuple[str, str], str] = {}
    with zipfile.ZipFile(archive) as zf:
        for raw_name in zf.namelist():
            name = _normalise_member(raw_name)
            image_match = _find_split_suffix(name, "images")
            if image_match and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                split, relative_name = image_match
                images.append(ImageItem(split, relative_name, archive_member=name))
                continue
            label_match = _find_split_suffix(name, "labels")
            if label_match and Path(name).suffix.lower() == ".txt":
                split, relative_name = label_match
                labels[(split, Path(relative_name).with_suffix("").as_posix().lower())] = name
    return images, labels


def _read_text(path: Optional[Path], zf: Optional[zipfile.ZipFile], member: Optional[str]) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8", errors="replace")
    if zf is None or member is None:
        return ""
    with zf.open(member, "r") as handle:
        return handle.read().decode("utf-8", errors="replace")


def _open_image(item: ImageItem, zf: Optional[zipfile.ZipFile]):
    if item.image_path is not None:
        return Image.open(item.image_path).convert("RGB")
    if zf is None or item.archive_member is None:
        raise FileNotFoundError(f"No image source for {item}")
    with zf.open(item.archive_member, "r") as handle:
        # ``copy`` is required because the ZIP stream closes after this block.
        return Image.open(handle).convert("RGB").copy()


def _parse_yolo(text: str, *, split: str, label_name: str, stats: Counter) -> List[Dict[str, object]]:
    boxes: List[Dict[str, object]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            stats["malformed_rows"] += 1
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except (TypeError, ValueError):
            stats["malformed_rows"] += 1
            continue
        values = (x_center, y_center, width, height)
        if class_id < 0 or any(value != value or value in (float("inf"), float("-inf")) for value in values):
            stats["invalid_values"] += 1
            continue
        if width <= 0 or height <= 0:
            stats["invalid_geometry"] += 1
            continue
        # Keep an invalid row visible in the summary but do not pass it into a
        # trainer. Small floating-point boundary noise is safely clipped.
        if any(value < -1e-6 or value > 1.0 + 1e-6 for value in values):
            stats["out_of_range"] += 1
            continue
        x_center = min(1.0, max(0.0, x_center))
        y_center = min(1.0, max(0.0, y_center))
        width = min(1.0, max(0.0, width))
        height = min(1.0, max(0.0, height))
        x1 = max(0.0, x_center - width / 2.0)
        y1 = max(0.0, y_center - height / 2.0)
        x2 = min(1.0, x_center + width / 2.0)
        y2 = min(1.0, y_center + height / 2.0)
        if x2 <= x1 or y2 <= y1:
            stats["invalid_geometry"] += 1
            continue
        boxes.append({
            "class_id": class_id,
            "bbox_xywh": [_json_number(x_center), _json_number(y_center), _json_number(width), _json_number(height)],
            "bbox_xyxy": [_json_number(x1), _json_number(y1), _json_number(x2), _json_number(y2)],
            "point_bottom_center": [_json_number(x_center), _json_number(y2)],
            "label_line": line_number,
        })
        stats[f"class_{class_id}"] += 1
    return boxes


def _sample_items(items: Sequence[ImageItem], count: int, seed: int) -> List[ImageItem]:
    if count <= 0:
        return []
    by_split: Dict[str, List[ImageItem]] = defaultdict(list)
    for item in items:
        by_split[item.split].append(item)
    rng = random.Random(seed)
    selected: List[ImageItem] = []
    # Keep the preview representative of all official splits.
    for split in SPLITS:
        group = list(by_split.get(split, []))
        rng.shuffle(group)
        take = min(len(group), max(1, count // max(1, len(SPLITS))))
        selected.extend(group[:take])
    remaining = [item for item in items if item not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def _make_preview(
    items: Sequence[ImageItem],
    box_map: Dict[Tuple[str, str], List[Dict[str, object]]],
    zf: Optional[zipfile.ZipFile],
    output: Path,
    seed: int,
) -> None:
    selected = _sample_items(items, len(items), seed)
    if not selected:
        return
    columns = 4
    tile_w, tile_h = 360, 285
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    colors = [(45, 170, 70), (220, 70, 40), (60, 100, 220), (220, 150, 35), (150, 70, 180)]
    for index, item in enumerate(selected):
        try:
            image = _open_image(item, zf)
        except Exception:
            continue
        tile = ImageOps.contain(image, (tile_w - 20, tile_h - 55))
        x0 = (index % columns) * tile_w + 10
        y0 = (index // columns) * tile_h + 28
        sheet.paste(tile, (x0 + (tile_w - 20 - tile.width) // 2, y0))
        scale_x = tile.width / max(1, image.width)
        scale_y = tile.height / max(1, image.height)
        image_x = x0 + (tile_w - 20 - tile.width) // 2
        image_y = y0
        for box in box_map.get((item.split, Path(item.relative_name).with_suffix("").as_posix().lower()), []):
            x1, y1, x2, y2 = box["bbox_xyxy"]
            px1 = image_x + int(float(x1) * image.width * scale_x)
            py1 = image_y + int(float(y1) * image.height * scale_y)
            px2 = image_x + int(float(x2) * image.width * scale_x)
            py2 = image_y + int(float(y2) * image.height * scale_y)
            color = colors[int(box["class_id"]) % len(colors)]
            draw.rectangle((px1, py1, px2, py2), outline=color, width=3)
            draw.ellipse((px2 - 4, py2 - 4, px2 + 4, py2 + 4), fill=color)
        title = f"{item.split} | {Path(item.relative_name).name}"
        draw.text((x0, 7 + (index // columns) * tile_h), _safe_title_limit(title), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def _safe_title_limit(value: str, limit: int = 48) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def build_manifests(
    *,
    archive: Optional[Path],
    root: Optional[Path],
    manifest_out: Path,
    weak_out: Path,
    summary_out: Path,
    preview_out: Optional[Path],
    preview_count: int,
    fire_class: Optional[int],
    weak_weight: float,
    seed: int,
) -> Dict[str, object]:
    if bool(archive) == bool(root):
        raise ValueError("Specify exactly one of --zip or --root")
    zf: Optional[zipfile.ZipFile] = None
    if archive:
        archive = archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
        items, label_index = _iter_zip_items(archive)
        zf = zipfile.ZipFile(archive)
    else:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        items, label_index = _iter_root_items(root)

    stats: Counter = Counter()
    split_counts: Counter = Counter()
    class_counts: Counter = Counter()
    box_map: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    image_records: List[Dict[str, object]] = []
    weak_records: List[Dict[str, object]] = []
    try:
        for item in sorted(items, key=lambda x: (x.split, x.relative_name.lower())):
            key = (item.split, Path(item.relative_name).with_suffix("").as_posix().lower())
            label_ref = label_index.get(key)
            if label_ref is None:
                stats["missing_label"] += 1
                continue
            label_text = _read_text(label_ref if isinstance(label_ref, Path) else None, zf, label_ref if isinstance(label_ref, str) else None)
            boxes = _parse_yolo(label_text, split=item.split, label_name=str(label_ref), stats=stats)
            if not boxes:
                stats["empty_or_invalid_label"] += 1
            try:
                image = _open_image(item, zf)
                image_width, image_height = image.size
                image.close()
            except Exception:
                stats["unreadable_image"] += 1
                continue
            box_map[key] = boxes
            split_counts[item.split] += 1
            for box in boxes:
                class_counts[str(box["class_id"])] += 1
            record: Dict[str, object] = {
                "image_id": f"{item.split}/{item.relative_name}",
                "split": item.split,
                "image_path": str(item.image_path) if item.image_path else None,
                "archive_path": str(archive) if archive else None,
                "archive_member": item.archive_member,
                "image_width": image_width,
                "image_height": image_height,
                "boxes": boxes,
                "num_boxes": len(boxes),
                "fire_class_id": fire_class,
                "has_fire": (any(int(box["class_id"]) == fire_class for box in boxes) if fire_class is not None else None),
                "source": "home_fire_yolo",
            }
            image_records.append(record)
            # A YOLO class is not automatically a fire label.  Until the
            # dataset's class mapping has been checked visually or from its
            # metadata, do not manufacture ``has_fire=true`` records: doing
            # so would silently contaminate point/ROI training with the other
            # class (for example smoke, person, or background equipment).
            if fire_class is None:
                stats["weak_points_skipped_unknown_class"] += len(boxes)
            else:
                for box in boxes:
                    if int(box["class_id"]) != fire_class:
                        stats["weak_points_skipped_non_fire_class"] += 1
                        continue
                    weak_records.append({
                        "image_id": record["image_id"],
                        "split": item.split,
                        "image_path": record["image_path"],
                        "archive_path": record["archive_path"],
                        "archive_member": record["archive_member"],
                        "image_width": image_width,
                        "image_height": image_height,
                        "class_id": int(box["class_id"]),
                        "bbox": box["bbox_xywh"],
                        "p_fire": box["point_bottom_center"],
                        "has_fire": True,
                        "source": "home_fire_yolo",
                        "point_source": "bbox_bottom_weak",
                        "label_quality": "weak",
                        "weight": float(weak_weight),
                    })
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        weak_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        with manifest_out.open("w", encoding="utf-8", newline="\n") as handle:
            for record in image_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        with weak_out.open("w", encoding="utf-8", newline="\n") as handle:
            for record in weak_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary = {
            "source": str(archive or root),
            "fire_class_id": fire_class,
            "weak_point_weight": weak_weight,
            "image_count": len(image_records),
            "weak_point_count": len(weak_records),
            "split_counts": dict(split_counts),
            "class_bbox_counts": dict(class_counts),
            "checks": dict(stats),
            "weak_manifest_usable": fire_class is not None and bool(weak_records),
            "weak_manifest_note": (
                "Only the selected fire class was converted to weak bottom-center points."
                if fire_class is not None
                else "No weak points were generated because --fire-class was not supplied."
            ),
            "manifest": str(manifest_out.resolve()),
            "weak_manifest": str(weak_out.resolve()),
        }
        summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if preview_out:
            preview_items = _sample_items(items, preview_count, seed)
            _make_preview(preview_items, box_map, zf, preview_out, seed)
        return summary
    finally:
        if zf is not None:
            zf.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    project_root = Path(__file__).resolve().parent
    source.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Home Fire/D-Fire YOLO ZIP archive",
    )
    source.add_argument("--root", type=Path, help="Extracted dataset root")
    parser.add_argument("--manifest-out", type=Path, default=project_root / "working" / "home_fire_manifest.jsonl")
    parser.add_argument("--weak-out", type=Path, default=project_root / "working" / "home_fire_weak_points.jsonl")
    parser.add_argument("--summary-out", type=Path, default=project_root / "working" / "home_fire_summary.json")
    parser.add_argument("--preview-out", type=Path, default=project_root / "working" / "home_fire_preview.jpg")
    parser.add_argument("--preview-count", type=int, default=18)
    parser.add_argument(
        "--fire-class",
        "--source-fire-class",
        dest="fire_class",
        type=int,
        default=None,
        help=(
            "Original dataset fire class id; use 1 for this project "
            "(0 means no fire). --fire-class is kept as a compatibility alias."
        ),
    )
    parser.add_argument("--weak-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.zip is None and args.root is None:
        args.zip = D_FIRE_ZIP
    summary = build_manifests(
        archive=args.zip,
        root=args.root,
        manifest_out=args.manifest_out,
        weak_out=args.weak_out,
        summary_out=args.summary_out,
        preview_out=args.preview_out if args.preview_count > 0 else None,
        preview_count=args.preview_count,
        fire_class=args.fire_class,
        weak_weight=args.weak_weight,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
