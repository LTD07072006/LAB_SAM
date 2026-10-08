"""Build a leakage-safe manifest for real iPhone/webcam fire experiments.

The script does not create or control a flame.  It only validates annotations
made from images/video frames.  A record must contain the 2D bottom-contact
pixel and a metric ``fire_xyz_world`` measured in the same coordinate frame as
the supplied camera calibration and room mesh.

Accepted annotation formats:

* JSON list of records;
* ``{"records": [...]}``;
* JSON mapping image names to record dictionaries or ``[u, v]`` pixels;
* CSV with ``image_path,p_fire_x,p_fire_y,X,Y,Z`` (aliases are accepted).

Example::

    .venv\\Scripts\\python.exe build_real_flame_manifest.py ^
      --dataset-root working\\iphone_room_session ^
      --annotations annotations.json ^
      --output working\\iphone_room_session\\manifest.jsonl ^
      --require-xyz

Images may remain in their original folder.  ``image_path`` is stored relative
to ``--dataset-root`` whenever possible, which keeps the manifest portable.
Splitting is scene/sequence based: adjacent frames from one scene never enter
different train/val/test partitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
SOURCE_TYPES = {"led", "display", "real_flame", "marker", "synthetic"}


def _finite(value: Any, size: int, name: str) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if len(array) < size or not np.all(np.isfinite(array[:size])):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return [float(item) for item in array[:size]]


def _read_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        return [dict(item) for item in raw["records"] if isinstance(item, dict)]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        records: list[dict[str, Any]] = []
        for image, value in raw.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("image_path", image)
            else:
                record = {"image_path": image, "p_fire_pixel": value}
            records.append(record)
        return records
    raise ValueError(f"Unsupported annotation JSON structure: {path}")


def _first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _image_path(root: Path, raw: Any) -> Path:
    if raw is None:
        raise ValueError("record has no image_path/image field")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"annotated image does not exist: {path}")
    return path


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _pixel_from_record(record: dict[str, Any], image_size: tuple[int, int]) -> list[float]:
    raw = _first(record, "p_fire_pixel", "pixel", "p_fire", "point_2d")
    if raw is None:
        x = _first(record, "p_fire_x", "pixel_x", "u")
        y = _first(record, "p_fire_y", "pixel_y", "v")
        raw = [x, y]
    pixel = _finite(raw, 2, "p_fire_pixel")
    # p_fire in the original dataset is normalized.  Pixel annotations outside
    # [0,1] are kept as pixels, so ordinary small images remain unambiguous.
    if all(0.0 <= value <= 1.0 for value in pixel):
        pixel = [pixel[0] * image_size[0], pixel[1] * image_size[1]]
    if not (-1.0 <= pixel[0] <= image_size[0] + 1.0 and -1.0 <= pixel[1] <= image_size[1] + 1.0):
        raise ValueError(f"p_fire_pixel is outside image bounds {image_size}: {pixel}")
    return pixel


def _xyz_from_record(record: dict[str, Any], required: bool) -> Optional[list[float]]:
    raw = _first(record, "fire_xyz_world", "xyz", "point_3d", "world_xyz")
    if raw is None:
        raw = [
            _first(record, "X", "x", "world_x"),
            _first(record, "Y", "y", "world_y"),
            _first(record, "Z", "z", "world_z"),
        ]
    if raw is None or any(value in (None, "") for value in raw):
        if required:
            raise ValueError("record has no fire_xyz_world; metric 3D benchmark needs XYZ")
        return None
    return _finite(raw, 3, "fire_xyz_world")


def _scene_id(record: dict[str, Any], image: Path) -> str:
    value = _first(record, "scene_id", "sequence_id", "session_id", "video_id")
    return str(value if value is not None else image.parent.name or "scene_default")


def _stable_split(scene_id: str, seed: int, ratios: tuple[float, float, float]) -> str:
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _parse_ratios(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3 or any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise ValueError("--split-ratios needs three non-negative values")
    total = float(sum(values))
    return tuple(float(value / total) for value in values)  # type: ignore[return-value]


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = args.dataset_root.expanduser().resolve()
    annotation_path = args.annotations.expanduser().resolve()
    raw_records = _read_annotations(annotation_path)
    if not raw_records:
        raise ValueError(f"No annotations found in {annotation_path}")
    ratios = _parse_ratios(args.split_ratios)
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_records):
        image = _image_path(root, _first(raw, "image_path", "image", "file", "path"))
        key = str(image).lower()
        if key in seen_paths:
            raise ValueError(f"Duplicate image annotation: {image}")
        seen_paths.add(key)
        with Image.open(image) as opened:
            width, height = opened.size
        pixel = _pixel_from_record(raw, (width, height))
        xyz = _xyz_from_record(raw, args.require_xyz)
        scene = _scene_id(raw, image)
        source_type = str(_first(raw, "source_type", "source", "fire_source") or args.source_type).lower()
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}, got {source_type!r}")
        raw_frame_index = _first(raw, "frame_index", "frame", "index")
        frame_index = int(index if raw_frame_index is None else raw_frame_index)
        requested_split = str(_first(raw, "split") or "").lower().strip()
        if requested_split and requested_split not in SPLITS:
            raise ValueError(f"Invalid split {requested_split!r} for {image}")
        normalized.append(
            {
                "sample_id": str(_first(raw, "sample_id", "id") or f"{scene}_frame_{frame_index:06d}"),
                "scene_id": scene,
                "frame_index": frame_index,
                "split": requested_split,
                "image_path": _relative_path(image, root),
                "image_size": [width, height],
                "has_fire": int(
                    1
                    if _first(raw, "has_fire", "class_id", "label") is None
                    else _first(raw, "has_fire", "class_id", "label")
                ),
                "fire_visible": int(_first(raw, "fire_visible") if _first(raw, "fire_visible") is not None else 1),
                "p_fire_pixel": pixel,
                "fire_xyz_world": xyz,
                "camera_id": _first(raw, "camera_id") or args.camera_id,
                "source_type": source_type,
                "label_uncertainty_px": float(
                    args.label_uncertainty_px
                    if _first(raw, "label_uncertainty_px", "pixel_sigma") is None
                    else _first(raw, "label_uncertainty_px", "pixel_sigma")
                ),
                "annotation_source": str(_first(raw, "annotation_source") or annotation_path.name),
            }
        )

    # A scene may not straddle splits. Explicitly provided splits are checked;
    # unlabelled scenes receive a deterministic split together.
    scene_splits: dict[str, str] = {}
    for record in normalized:
        if record["split"]:
            scene = str(record["scene_id"])
            old = scene_splits.get(scene)
            if old is not None and old != record["split"]:
                raise ValueError(f"Scene {scene!r} appears in multiple splits: {old}, {record['split']}")
            scene_splits[scene] = str(record["split"])
    for record in normalized:
        scene = str(record["scene_id"])
        record["split"] = scene_splits.setdefault(scene, _stable_split(scene, args.seed, ratios))
        # Keep a normalized coordinate too; downstream code can use either.
        width, height = record["image_size"]
        record["p_fire"] = [record["p_fire_pixel"][0] / width, record["p_fire_pixel"][1] / height]
        if record["fire_xyz_world"] is None and args.require_xyz:
            raise ValueError(f"record has no XYZ after normalization: {record['sample_id']}")

    normalized.sort(key=lambda item: (str(item["split"]), str(item["scene_id"]), int(item["frame_index"])))
    counts = {split: sum(record["split"] == split for record in normalized) for split in SPLITS}
    scenes = {split: sorted({record["scene_id"] for record in normalized if record["split"] == split}) for split in SPLITS}
    info = {
        "format": "LAB_SAM.real_flame_manifest.v1",
        "dataset_root": str(root),
        "annotation_source": str(annotation_path),
        "records": len(normalized),
        "records_by_split": counts,
        "scenes_by_split": {key: len(value) for key, value in scenes.items()},
        "source_types": sorted({record["source_type"] for record in normalized}),
        "metric_xyz_present": sum(record["fire_xyz_world"] is not None for record in normalized),
        "warning": "Real-room accuracy is valid only when calibration, mesh and XYZ labels share one measured coordinate frame.",
    }
    return normalized, info


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-xyz", action="store_true", help="Reject annotations without metric XYZ")
    parser.add_argument("--source-type", choices=sorted(SOURCE_TYPES), default="led")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--label-uncertainty-px", type=float, default=3.0)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.label_uncertainty_px < 0.0:
        raise ValueError("--label-uncertainty-px must be >= 0")
    records, info = build_manifest(args)
    output = (args.output or (args.dataset_root / "manifest.jsonl")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    info_path = output.with_name("dataset_info.json")
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "dataset_info": str(info_path), **info}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
