"""Validate measured-room assets before a physical 3D benchmark.

This is intentionally strict: a file marked ``template`` or ``synthetic`` is
not accepted as real calibration/mesh/ground truth.  The validator checks
schema, finite numeric values, units and coordinate-frame metadata, but it
cannot prove that a measurement was physically taken.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REJECTED_TOKENS = ("template", "synthetic", "provisional", "example", "placeholder")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _metadata(value: dict[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _check_status(path: Path, metadata: dict[str, Any]) -> None:
    text = " ".join(str(v).lower() for v in metadata.values())
    if any(token in text for token in REJECTED_TOKENS):
        raise ValueError(f"{path} is marked as template/synthetic/provisional")
    status = str(metadata.get("status", "")).lower()
    if status != "measured_real_room":
        raise ValueError(
            f"{path} metadata.status must be 'measured_real_room' (got {status!r})"
        )
    if str(metadata.get("units", "")).lower() not in {"metres", "meters", "m"}:
        raise ValueError(f"{path} must declare metres as units")
    if not metadata.get("coordinate_system"):
        raise ValueError(f"{path} must declare coordinate_system")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(shape)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_camera(path: Path) -> dict[str, Any]:
    data = _read(path)
    metadata = _metadata(data)
    _check_status(path, metadata)
    K = _finite_array(data.get("intrinsics", {}).get("K"), (3, 3), "camera K")
    R = _finite_array(data.get("extrinsics", {}).get("R"), (3, 3), "camera R")
    position = _finite_array(data.get("extrinsics", {}).get("camera_position"), (3,), "camera position")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{path} has invalid focal length")
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-2):
        raise ValueError(f"{path} rotation is not orthonormal")
    if not np.isfinite(position).all():
        raise ValueError(f"{path} camera position is invalid")
    return {"type": "camera", "path": str(path), "image_size": data.get("image_size")}


def validate_mesh(path: Path) -> dict[str, Any]:
    data = _read(path)
    metadata = _metadata(data)
    _check_status(path, metadata)
    vertices = _finite_array(data.get("vertices"), (-1, 3), "mesh vertices")
    faces = np.asarray(data.get("faces"), dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{path} has invalid mesh vertices/faces")
    return {"type": "mesh", "path": str(path), "vertices": len(vertices), "faces": len(faces)}


def validate_labels(path: Path) -> dict[str, Any]:
    data = _read(path)
    metadata = _metadata(data)
    _check_status(path, metadata)
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain non-empty records")
    valid = 0
    cameras: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        xyz = _finite_array(record.get("fire_xyz_world", record.get("xyz")), (3,), "fire XYZ")
        image = record.get("image", record.get("image_path", record.get("sample_id")))
        if image is None or not str(image):
            raise ValueError(f"{path} contains a record without image/sample_id")
        if record.get("camera_id") is not None:
            cameras.add(str(record["camera_id"]))
        valid += 1
    if valid == 0:
        raise ValueError(f"{path} contains no valid XYZ records")
    return {"type": "ground_truth_3d", "path": str(path), "records": valid, "camera_ids": sorted(cameras)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--labels-3d", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        validate_camera(args.camera.expanduser().resolve()),
        validate_mesh(args.mesh.expanduser().resolve()),
        validate_labels(args.labels_3d.expanduser().resolve()),
    ]
    output = {"valid": True, "assets": reports, "warning": "Schema validation is not a substitute for measurement audit."}
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
