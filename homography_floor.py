"""Inverse-perspective mapping (IPM) for a planar room floor.

The main localisation pipeline uses calibrated ray casting against a mesh.  A
floor homography is a useful independent baseline when the target is known to
touch one planar floor (Z=0):

    undistorted pixel -> H_image_to_floor -> (X, Y, 0)

This module deliberately does not infer a metric scale from an image.  The
camera intrinsics/extrinsics or measured 2D-3D floor correspondences must
already be expressed in the same room coordinate frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape}") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def estimate_homography(
    image_points: Sequence[Sequence[float]],
    floor_points_xy: Sequence[Sequence[float]],
) -> np.ndarray:
    """Estimate ``floor_xy -> image`` H with a numerically stable DLT.

    At least four non-collinear correspondences are required.  The result is
    normalised so ``H[2, 2]`` is one whenever that value is non-zero.
    """

    image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    floor = np.asarray(floor_points_xy, dtype=np.float64).reshape(-1, 2)
    if len(image) != len(floor) or len(image) < 4:
        raise ValueError("Homography needs at least four matching 2D points")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(floor)):
        raise ValueError("Homography points must be finite")
    if np.linalg.matrix_rank(floor - floor.mean(axis=0)) < 2:
        raise ValueError("Floor points are collinear; choose points spread over the floor")

    rows: list[list[float]] = []
    for (u, v), (x, y) in zip(image, floor):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homography = vh[-1].reshape(3, 3)
    scale = homography[2, 2]
    if abs(scale) > 1e-12:
        homography = homography / scale
    if abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("Estimated homography is singular")
    return homography


@dataclass
class FloorHomography:
    """Bidirectional projective map between the floor and image plane."""

    H_floor_to_image: np.ndarray
    H_image_to_floor: np.ndarray
    source: str = "unknown"

    def __post_init__(self) -> None:
        self.H_floor_to_image = _matrix(self.H_floor_to_image, (3, 3), "H_floor_to_image")
        self.H_image_to_floor = _matrix(self.H_image_to_floor, (3, 3), "H_image_to_floor")
        if abs(float(np.linalg.det(self.H_floor_to_image))) < 1e-12:
            raise ValueError("H_floor_to_image is singular")

    @classmethod
    def from_camera(
        cls,
        K: Sequence[Sequence[float]],
        R_world_to_camera: Sequence[Sequence[float]],
        t_world_to_camera: Sequence[float] | Sequence[Sequence[float]],
    ) -> "FloorHomography":
        """Build H for ``X_camera = R @ X_world + t`` and world ``Z=0``."""

        K_array = _matrix(K, (3, 3), "K")
        R = _matrix(R_world_to_camera, (3, 3), "R_world_to_camera")
        t = np.asarray(t_world_to_camera, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(t)):
            raise ValueError("t_world_to_camera contains non-finite values")
        # World floor point [X, Y, 0, 1] projects with K [r1 r2 t].
        H = K_array @ np.column_stack((R[:, 0], R[:, 1], t))
        if abs(float(np.linalg.det(H))) < 1e-12:
            raise ValueError("Camera pose cannot produce an invertible floor homography")
        return cls(H, np.linalg.inv(H), source="camera_pose")

    @classmethod
    def from_calibration(cls, calibration: Any) -> "FloorHomography":
        """Build from a repository ``CameraCalibration``-like object."""

        return cls.from_camera(calibration.K, calibration.R, calibration.t)

    @classmethod
    def from_correspondences(
        cls,
        image_points: Sequence[Sequence[float]],
        floor_points_xy: Sequence[Sequence[float]],
    ) -> "FloorHomography":
        H = estimate_homography(image_points, floor_points_xy)
        return cls(H, np.linalg.inv(H), source="measured_floor_correspondences")

    @staticmethod
    def _apply(H: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not np.all(np.isfinite(values)):
            raise ValueError("Projection points must be finite")
        homogeneous = np.column_stack((values, np.ones(len(values)))) @ H.T
        denominator = homogeneous[:, 2]
        if np.any(np.abs(denominator) < 1e-12):
            raise ValueError("Projection intersects the homography horizon")
        return homogeneous[:, :2] / denominator[:, None]

    def pixel_to_floor_xy(self, pixels: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
        """Map ideal/undistorted image pixels to metric floor ``(X,Y)``."""

        return self._apply(self.H_image_to_floor, pixels)

    def pixel_to_floor_xyz(self, pixels: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
        xy = self.pixel_to_floor_xy(pixels)
        return np.column_stack((xy, np.zeros(len(xy), dtype=np.float64)))

    def floor_xy_to_pixel(self, points_xy: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
        return self._apply(self.H_floor_to_image, points_xy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "plane": "world Z=0",
            "H_floor_to_image": self.H_floor_to_image.tolist(),
            "H_image_to_floor": self.H_image_to_floor.tolist(),
        }


def localize_floor_pixel(calibration: Any, pixel: Sequence[float]) -> np.ndarray:
    """Undistort one pixel and map it to ``(X,Y,0)`` using camera pose."""

    point = np.asarray(pixel, dtype=np.float64).reshape(1, 2)
    ideal = calibration.undistort_pixels(point)
    return FloorHomography.from_calibration(calibration).pixel_to_floor_xyz(ideal)[0]


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, help="Camera JSON with intrinsics/extrinsics")
    parser.add_argument(
        "--correspondences",
        type=Path,
        help="JSON with image_points and floor_points_xy; used instead of --calibration",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.calibration) == bool(args.correspondences):
        raise ValueError("Provide exactly one of --calibration or --correspondences")

    if args.calibration is not None:
        from camera_calibration import CameraCalibration

        homography = FloorHomography.from_calibration(
            CameraCalibration.from_json(args.calibration.expanduser().resolve())
        )
    else:
        data = _load_json(args.correspondences.expanduser().resolve())
        image_points = data.get("image_points", data.get("image_points_px"))
        floor_points = data.get("floor_points_xy", data.get("object_points_xy"))
        if image_points is None or floor_points is None:
            raise ValueError("Correspondence JSON needs image_points and floor_points_xy")
        homography = FloorHomography.from_correspondences(image_points, floor_points)

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(homography.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved={args.output.expanduser().resolve()}")


if __name__ == "__main__":
    _main()
