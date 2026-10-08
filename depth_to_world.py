"""Convert a monocular depth sample into the calibrated world frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class DepthWorldEstimate:
    point: Optional[np.ndarray]
    depth_value: Optional[float]
    std_m: Optional[np.ndarray]
    success: bool
    units: str = "unknown"
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": None if self.point is None else np.asarray(self.point).tolist(),
            "depth_value": self.depth_value,
            "std_m": None if self.std_m is None else np.asarray(self.std_m).tolist(),
            "success": bool(self.success),
            "units": self.units,
            "reason": self.reason,
        }


def sample_depth(
    depth_map: np.ndarray,
    pixel: Sequence[float],
    radius_px: int = 2,
) -> tuple[Optional[float], Optional[float], int]:
    """Return median depth, robust spread and number of valid samples."""

    values = np.asarray(depth_map, dtype=np.float64)
    if values.ndim != 2:
        return None, None, 0
    u, v = np.asarray(pixel, dtype=np.float64).reshape(2)
    center_x, center_y = int(round(u)), int(round(v))
    radius = max(0, int(radius_px))
    x0, x1 = max(0, center_x - radius), min(values.shape[1], center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(values.shape[0], center_y + radius + 1)
    patch = values[y0:y1, x0:x1].reshape(-1)
    patch = patch[np.isfinite(patch) & (patch > 0.0)]
    if not len(patch):
        return None, None, 0
    median = float(np.median(patch))
    mad = float(1.4826 * np.median(np.abs(patch - median))) if len(patch) > 1 else 0.0
    return median, max(mad, 1e-4), int(len(patch))


def depth_pixel_to_world(
    calibration: Any,
    pixel: Sequence[float],
    depth_value: float,
    units: str = "camera_z",
    scale: float = 1.0,
    offset: float = 0.0,
    depth_std_m: Optional[float] = None,
) -> DepthWorldEstimate:
    """Convert one depth value to world XYZ.

    ``camera_z`` means optical-axis depth in metres after ``scale``/``offset``;
    ``ray_range`` means Euclidean distance from the camera along the ray.
    ``relative`` is rejected unless the caller explicitly supplies a metric
    scale/offset policy.  This avoids presenting relative foundation-model
    output as a metric 3D result.
    """

    mode = str(units).strip().lower()
    if mode not in {"relative", "camera_z", "ray_range"}:
        return DepthWorldEstimate(None, None, None, False, mode, "unknown_depth_units")
    try:
        value = float(depth_value) * float(scale) + float(offset)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("depth must be finite and positive after scaling")
        if mode == "relative" and abs(float(scale) - 1.0) < 1e-12 and abs(float(offset)) < 1e-12:
            return DepthWorldEstimate(None, value, None, False, mode, "relative_depth_has_no_metric_scale")
        ideal = calibration.undistort_pixels(np.asarray(pixel, dtype=np.float64).reshape(1, 2))[0]
        origin, direction = calibration.geometry().pixel_to_ray(float(ideal[0]), float(ideal[1]))
        if mode == "ray_range":
            world = origin + direction * value
        else:
            normalized = np.linalg.inv(calibration.K) @ np.array([ideal[0], ideal[1], 1.0])
            camera_xyz = normalized * (value / max(normalized[2], 1e-9))
            world = calibration.R.T @ (camera_xyz - calibration.t.reshape(3))
        if not np.all(np.isfinite(world)):
            raise ValueError("world point is non-finite")
        sensitivity = max(float(depth_std_m or 0.0), 0.01)
        return DepthWorldEstimate(
            world,
            value,
            np.full(3, sensitivity, dtype=np.float64),
            True,
            mode,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError, AttributeError) as exc:
        return DepthWorldEstimate(None, None, None, False, mode, str(exc))


def depth_map_pixel_to_world(
    calibration: Any,
    depth_map: np.ndarray,
    pixel: Sequence[float],
    units: str = "camera_z",
    scale: float = 1.0,
    offset: float = 0.0,
    radius_px: int = 2,
) -> DepthWorldEstimate:
    value, spread, count = sample_depth(depth_map, pixel, radius_px)
    if value is None:
        return DepthWorldEstimate(None, None, None, False, units, "no_valid_depth_patch")
    result = depth_pixel_to_world(calibration, pixel, value, units, scale, offset, spread)
    if result.std_m is not None and count:
        result.std_m = result.std_m * (1.0 + 1.0 / np.sqrt(count))
    return result


__all__ = ["DepthWorldEstimate", "depth_map_pixel_to_world", "depth_pixel_to_world", "sample_depth"]
