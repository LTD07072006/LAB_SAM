"""Build PnP 2D-3D correspondences from the existing fire labels.

The existing ``dataset_labels (1).json`` contains ``p_fire`` (normalized 2D
pixels) but no metric 3D coordinate. This utility joins those labels with a
separate measured-anchor file. It intentionally refuses to invent 3D values.

Anchor file format::

    {
      "metadata": {
        "status": "measured_real_room",
        "camera_id": "cctv_01",
        "units": "metres"
      },
      "records": [
        {
          "image": "img_103.jpg",
          "xyz": [1.0, 0.0, 0.0],
          "anchor_id": "floor_A",
          "camera_id": "cctv_01"
        }
      ]
    }

The 3D point in each record must be the physical location represented by the
2D ``p_fire`` point in the corresponding image. For a fixed CCTV, multiple
images may be pooled only when they share the same ``camera_id`` and camera
pose. If the camera moves, create a separate group per frame/pose.

Example::

    .venv\\Scripts\\python.exe build_pnp_correspondences.py ^
      --labels "fire-model-data\\dataset_labels (1).json" ^
      --dataset-root datasets\\fire-detection-from-cctv ^
      --anchors room_anchor_xyz.json ^
      --intrinsics camera_intrinsics.json ^
      --camera-id cctv_01 ^
      --output working\\pnp_cctv_01.json

Then solve the pose with ``calibrate_room_pose_pnp.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalise_path(value: Any) -> str:
    return str(value).replace("\\", "/").lower()


def _aliases(value: Any) -> list[str]:
    text = _normalise_path(value)
    aliases = [text, Path(text).name]
    for marker in ("/img_data/", "/images/"):
        if marker in text:
            suffix = text.split(marker, 1)[1]
            aliases.extend([suffix, marker.strip("/") + "/" + suffix])
    parts = text.split("/")
    for split in ("train", "val", "test"):
        if split in parts:
            index = parts.index(split)
            aliases.append("/".join(parts[index:]))
            break
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_intrinsics(path: Path) -> dict[str, Any]:
    """Load checkerboard intrinsics in either repository JSON form.

    Accepted forms are ``{"intrinsics": {"K": ...}}`` and the shorter
    ``{"K": ...}``.  Keeping this separate from the label/anchor records
    avoids silently treating a 3D anchor file as a camera calibration file.
    """
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Intrinsics file must be a JSON object: {path}")
    value = data.get("intrinsics", data)
    if not isinstance(value, dict) or "K" not in value:
        raise ValueError(
            f"Intrinsics file must contain K or intrinsics.K: {path}"
        )
    try:
        K = np.asarray(value["K"], dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"K must be a 3x3 numeric matrix: {path}") from exc
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"K contains invalid focal lengths: {path}")
    try:
        dist = np.asarray(value.get("dist_coeffs", []), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dist_coeffs must be numeric: {path}") from exc
    if len(dist) not in (0, 4, 5, 8, 12, 14):
        raise ValueError(
            "dist_coeffs must have 0, 4, 5, 8, 12 or 14 values "
            f"(got {len(dist)}): {path}"
        )
    return {
        "K": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": data.get("image_size", value.get("image_size")),
        "source": str(path.resolve()),
    }


def _load_labels(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Labels must be a JSON list: {path}")
    return data


def _load_anchor_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = _load_json(path)
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        records = data["records"]
    elif isinstance(data, dict):
        records = []
        for key, value in data.items():
            if key == "metadata":
                continue
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("image", key)
            else:
                record = {"image": key, "xyz": value}
            records.append(record)
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Anchor file must be a JSON object/list: {path}")
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, [record for record in records if isinstance(record, dict)]


def _xyz(record: dict[str, Any]) -> Optional[list[float]]:
    value = record.get("xyz", record.get("fire_xyz_world", record.get("point")))
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(point) < 3 or not np.all(np.isfinite(point[:3])):
        return None
    return [float(value) for value in point[:3]]


def _find_image(root: Path, raw_path: Any, cache: dict[str, Path]) -> Optional[Path]:
    if raw_path is None:
        return None
    raw = Path(str(raw_path))
    if raw.is_file():
        return raw.resolve()
    for alias in _aliases(raw_path):
        if alias in cache:
            return cache[alias]
    candidates = []
    for relative in _aliases(raw_path):
        candidate = (root / relative).resolve()
        if candidate.is_file():
            candidates.append(candidate)
        candidate = (root / "data" / "data" / "img_data" / relative).resolve()
        if candidate.is_file():
            candidates.append(candidate)
        candidate = (root / "data" / "img_data" / relative).resolve()
        if candidate.is_file():
            candidates.append(candidate)
    if not candidates:
        basename = raw.name.lower()
        candidates = [
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name.lower() == basename
        ]
    if not candidates:
        return None
    result = candidates[0]
    for alias in _aliases(raw_path):
        cache.setdefault(alias, result)
    return result


def _build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        for alias in _aliases(relative):
            index.setdefault(alias, path.resolve())
    return index


def _pixel(label: dict[str, Any], image: Path) -> Optional[list[float]]:
    value = label.get("p_fire_pixel", label.get("p_fire"))
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        return None
    width, height = Image.open(image).size
    if np.all((point[:2] >= 0.0) & (point[:2] <= 1.0)):
        point = point[:2] * np.asarray([width, height], dtype=np.float64)
    else:
        point = point[:2]
    return [float(point[0]), float(point[1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help=(
            "Optional checkerboard calibration JSON. Accepts K/dist_coeffs "
            "under intrinsics or at the top level."
        ),
    )
    parser.add_argument("--camera-id", default=None, help="Select one fixed-camera group")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels_path = args.labels.expanduser().resolve()
    root = args.dataset_root.expanduser().resolve()
    anchors_path = args.anchors.expanduser().resolve()
    intrinsics_path = (
        None if args.intrinsics is None else args.intrinsics.expanduser().resolve()
    )
    labels = _load_labels(labels_path)
    anchor_metadata, anchor_records = _load_anchor_records(anchors_path)
    intrinsics = None if intrinsics_path is None else _load_intrinsics(intrinsics_path)
    anchor_index: dict[str, list[dict[str, Any]]] = {}
    for record in anchor_records:
        xyz = _xyz(record)
        image = record.get("image", record.get("image_path", record.get("sample_id")))
        if xyz is None or image is None:
            continue
        for alias in _aliases(image):
            anchor_index.setdefault(alias, []).append(record)

    image_index = _build_image_index(root)
    pairs: list[dict[str, Any]] = []
    skipped = {"not_fire": 0, "no_anchor": 0, "missing_image": 0, "bad_2d": 0, "camera_mismatch": 0}
    selected_camera = None if args.camera_id is None else str(args.camera_id)
    for label in labels:
        if int(label.get("has_fire", 0)) != 1:
            skipped["not_fire"] += 1
            continue
        split = str(label.get("split", ""))
        raw_image = label.get("image_path", label.get("image"))
        image = _find_image(root, raw_image, image_index)
        if image is None:
            skipped["missing_image"] += 1
            continue
        matching = []
        for alias in _aliases(raw_image):
            matching.extend(anchor_index.get(alias, []))
        if not matching:
            skipped["no_anchor"] += 1
            continue
        record = matching[0]
        camera_id = record.get("camera_id", anchor_metadata.get("camera_id"))
        if selected_camera is not None and str(camera_id) != selected_camera:
            skipped["camera_mismatch"] += 1
            continue
        pixel = _pixel(label, image)
        xyz = _xyz(record)
        if pixel is None:
            skipped["bad_2d"] += 1
            continue
        if args.split != "all" and split and split != args.split:
            continue
        pairs.append(
            {
                "image": str(image),
                "image_name": image.name,
                "sample_id": record.get("sample_id", image.stem),
                "anchor_id": record.get("anchor_id"),
                "camera_id": camera_id,
                "image_size": list(Image.open(image).size),
                "object_point_world": xyz,
                "image_point": pixel,
            }
        )

    if args.max_pairs > 0:
        pairs = pairs[: int(args.max_pairs)]
    if len(pairs) < 4:
        raise RuntimeError(
            f"Only {len(pairs)} valid 2D-3D pairs; solvePnP needs at least 4. "
            f"skipped={skipped}"
        )
    image_sizes = {tuple(pair["image_size"]) for pair in pairs}
    output = {
        "metadata": {
            "status": str(anchor_metadata.get("status", "measured_real_room")),
            "units": str(anchor_metadata.get("units", "metres")),
            "coordinate_system": anchor_metadata.get("coordinate_system", "room frame"),
            "camera_id": selected_camera or anchor_metadata.get("camera_id"),
            "source_labels": str(labels_path),
            "source_anchors": str(anchors_path),
            "source_intrinsics": None if intrinsics_path is None else str(intrinsics_path),
            "pair_count": len(pairs),
            "skipped": skipped,
            "warning": "All pairs must share one fixed camera pose for one PnP solve.",
        },
        "image_size": list(next(iter(image_sizes))) if len(image_sizes) == 1 else None,
        "object_points_world": [pair["object_point_world"] for pair in pairs],
        "image_points": [pair["image_point"] for pair in pairs],
        "records": pairs,
    }
    if intrinsics is not None:
        output["intrinsics"] = {
            "K": intrinsics["K"],
            "dist_coeffs": intrinsics["dist_coeffs"],
        }
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"pairs": len(pairs), "skipped": skipped, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
