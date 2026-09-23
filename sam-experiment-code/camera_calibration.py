"""Camera intrinsics/extrinsics and distortion correction for localisation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from locator import CameraGeometry


def _array(data: Any, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
    value = np.asarray(data, dtype=np.float64)
    if shape is not None:
        value = value.reshape(shape)
    return value


@dataclass
class CameraCalibration:
    """Calibration in the same convention used by ``CameraGeometry``.

    ``R`` maps world coordinates to camera coordinates and ``t`` is the
    matching translation, i.e. ``X_camera = R @ X_world + t``.
    """

    K: np.ndarray
    dist_coeffs: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: Optional[tuple[int, int]] = None

    def __post_init__(self):
        self.K = _array(self.K, (3, 3))
        self.dist_coeffs = _array(self.dist_coeffs).reshape(-1)
        self.R = _array(self.R, (3, 3))
        self.t = _array(self.t).reshape(3, 1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        intrinsics = data.get("intrinsics", data)
        K = intrinsics.get("K", intrinsics.get("camera_matrix"))
        if K is None:
            raise ValueError("Calibration JSON needs K or camera_matrix")
        distortion = intrinsics.get("dist_coeffs", intrinsics.get("distortion", intrinsics.get("D", [])))
        extrinsics = data.get("extrinsics", data)
        R = extrinsics.get("R", extrinsics.get("rotation"))
        if R is None:
            raise ValueError("Calibration JSON needs R or rotation")
        t = extrinsics.get("t", extrinsics.get("translation"))
        camera_position = extrinsics.get("camera_position", data.get("camera_position"))
        if t is None and camera_position is not None:
            R_array = _array(R, (3, 3))
            t = -R_array @ _array(camera_position).reshape(3, 1)
        if t is None:
            raise ValueError("Calibration JSON needs t/translation or camera_position")
        size = data.get("image_size", intrinsics.get("image_size"))
        image_size = None if size is None else (int(size[0]), int(size[1]))
        return cls(K, distortion, R, t, image_size)

    @classmethod
    def from_json(cls, path: Path | str) -> "CameraCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def geometry(self) -> CameraGeometry:
        return CameraGeometry(self.K, self.R, self.t)

    def camera_position(self) -> np.ndarray:
        return (-self.R.T @ self.t).reshape(3)

    def undistort_pixels(self, pixels: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
        """Return ideal pixels, using OpenCV when present and a NumPy fallback."""
        points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if not len(self.dist_coeffs) or np.allclose(self.dist_coeffs, 0.0):
            return points.copy()
        try:
            import cv2
            result = cv2.undistortPoints(
                points.reshape(-1, 1, 2), self.K, self.dist_coeffs,
                P=self.K,
            )
            return result.reshape(-1, 2)
        except ImportError:
            pass
        return self._undistort_numpy(points)

    def _undistort_numpy(self, pixels: np.ndarray, iterations: int = 12) -> np.ndarray:
        coeffs = np.pad(self.dist_coeffs, (0, max(0, 5 - len(self.dist_coeffs))))
        k1, k2, p1, p2, k3 = coeffs[:5]
        inv_k = np.linalg.inv(self.K)
        homogeneous = np.column_stack([pixels, np.ones(len(pixels))]) @ inv_k.T
        xd, yd = homogeneous[:, 0], homogeneous[:, 1]
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x = (xd - 2.0 * p1 * x * y - p2 * (r2 + 2.0 * x * x)) / np.maximum(radial, 1e-12)
            y = (yd - p1 * (r2 + 2.0 * y * y) - 2.0 * p2 * x * y) / np.maximum(radial, 1e-12)
        return np.column_stack([self.K[0, 0] * x + self.K[0, 2], self.K[1, 1] * y + self.K[1, 2]])
