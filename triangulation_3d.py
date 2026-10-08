"""Mesh-free multi-view triangulation for the post-ROI fire pipeline.

The detector/ROI stage supplies one image point per frame.  When the same
fire event is visible in at least two frames with different camera poses, the
corresponding rays can be triangulated directly into a metric 3D point.  This
module deliberately does not use a room mesh, a floor-plane assumption, or a
learned depth model.

It is an offline/multi-view research branch.  In a live system the caller can
use a causal window (current frame plus previous frames) instead of all frames
from a sequence.  The result is only metric when the camera intrinsics,
extrinsics and world coordinate frame are metric and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class TriangulationEstimate:
    """Result of linear multi-view triangulation."""

    point: np.ndarray | None
    success: bool
    reprojection_rmse_px: float | None = None
    used_views: int = 0
    condition_number: float | None = None
    positive_depth_rate: float | None = None
    reason: str | None = None
    null_space_gap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": None if self.point is None else np.asarray(self.point).tolist(),
            "success": bool(self.success),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "used_views": int(self.used_views),
            "condition_number": self.condition_number,
            "positive_depth_rate": self.positive_depth_rate,
            "reason": self.reason,
            "null_space_gap": self.null_space_gap,
        }


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape}") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def projection_matrix(calibration: Any) -> np.ndarray:
    """Return ``P = K [R | t]`` for a repository calibration object."""

    K = _matrix(calibration.K, (3, 3), "K")
    R = _matrix(calibration.R, (3, 3), "R")
    t = _matrix(calibration.t, (3, 1), "t")
    return K @ np.column_stack((R, t))


def _project(P: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    projected = P @ np.concatenate((point, [1.0]))
    if abs(float(projected[2])) < 1e-12:
        return None
    result = projected[:2] / projected[2]
    return result if np.all(np.isfinite(result)) else None


def triangulate_dlt(
    pixels: Sequence[Sequence[float]],
    calibrations: Sequence[Any],
    *,
    min_views: int = 2,
    max_reprojection_rmse_px: float | None = None,
) -> TriangulationEstimate:
    """Triangulate one world point from calibrated 2D observations.

    Pixels are undistorted before the DLT solve.  The returned reprojection
    error is measured in ideal-pixel coordinates.  A positive-depth check is
    included because a numerically valid DLT point behind the cameras is not a
    usable localisation.
    """

    try:
        observations = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError) as exc:
        return TriangulationEstimate(None, False, reason=f"invalid_pixels:{exc}")
    if len(observations) != len(calibrations):
        return TriangulationEstimate(None, False, reason="pixels_calibrations_length_mismatch")
    if len(observations) < max(2, int(min_views)):
        return TriangulationEstimate(None, False, used_views=len(observations), reason="need_more_views")
    if not np.all(np.isfinite(observations)):
        return TriangulationEstimate(None, False, used_views=len(observations), reason="non_finite_pixels")

    matrices: list[np.ndarray] = []
    ideal_pixels: list[np.ndarray] = []
    for calibration, pixel in zip(calibrations, observations):
        try:
            ideal = np.asarray(calibration.undistort_pixels(pixel.reshape(1, 2))[0], dtype=np.float64)
            matrix = projection_matrix(calibration)
        except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return TriangulationEstimate(None, False, used_views=len(matrices), reason=f"invalid_calibration:{exc}")
        if not np.all(np.isfinite(ideal)):
            return TriangulationEstimate(None, False, used_views=len(matrices), reason="non_finite_ideal_pixel")
        matrices.append(matrix)
        ideal_pixels.append(ideal)

    rows: list[np.ndarray] = []
    for matrix, (u, v) in zip(matrices, ideal_pixels):
        rows.append(u * matrix[2] - matrix[0])
        rows.append(v * matrix[2] - matrix[1])
    system = np.asarray(rows, dtype=np.float64)
    try:
        _, singular_values, vh = np.linalg.svd(system)
    except np.linalg.LinAlgError as exc:
        return TriangulationEstimate(None, False, used_views=len(matrices), reason=f"svd_failed:{exc}")
    if len(singular_values) < 4 or singular_values[-1] < 1e-14:
        # A small last singular value is expected for a good homogeneous
        # solution; an all-zero/degenerate system is the actual failure mode.
        if not np.all(np.isfinite(singular_values)):
            return TriangulationEstimate(None, False, used_views=len(matrices), reason="invalid_singular_values")
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        return TriangulationEstimate(None, False, used_views=len(matrices), reason="point_at_infinity")
    point = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(point)):
        return TriangulationEstimate(None, False, used_views=len(matrices), reason="non_finite_point")

    errors: list[float] = []
    positive: list[bool] = []
    for matrix, calibration, pixel in zip(matrices, calibrations, ideal_pixels):
        predicted = _project(matrix, point)
        if predicted is None:
            return TriangulationEstimate(None, False, used_views=len(matrices), reason="invalid_reprojection")
        errors.append(float(np.linalg.norm(predicted - pixel)))
        try:
            camera_point = calibration.R @ point + np.asarray(calibration.t, dtype=np.float64).reshape(3)
            positive.append(bool(camera_point[2] > 1e-8))
        except (AttributeError, ValueError):
            positive.append(True)
    reprojection = float(np.sqrt(np.mean(np.square(errors))))
    positive_rate = float(np.mean(positive)) if positive else 0.0
    # For homogeneous DLT, the smallest singular value is the expected
    # null-space direction.  Dividing by it would label an exact, noiseless
    # solution as "ill-conditioned" simply because its residual is numerical
    # zero.  Keep two diagnostics separate:
    #   * condition_number: scale conditioning excluding the null singularity;
    #   * null_space_gap: separation between the physical rank and null space,
    #     where larger is safer.
    condition = None
    null_space_gap = None
    if len(singular_values) >= 3:
        rank_scale = max(float(singular_values[-2]), 1e-15)
        condition = float(singular_values[0] / rank_scale)
        null_space_gap = float(singular_values[-2] / max(float(singular_values[-1]), 1e-15))
    if positive_rate < 1.0:
        return TriangulationEstimate(
            point,
            False,
            reprojection,
            len(matrices),
            condition,
            positive_rate,
            "point_not_in_front_of_all_cameras",
            null_space_gap,
        )
    if max_reprojection_rmse_px is not None and reprojection > float(max_reprojection_rmse_px):
        return TriangulationEstimate(
            point,
            False,
            reprojection,
            len(matrices),
            condition,
            positive_rate,
            "reprojection_error_too_large",
            null_space_gap,
        )
    return TriangulationEstimate(
        point,
        True,
        reprojection,
        len(matrices),
        condition,
        positive_rate,
        None,
        null_space_gap,
    )


__all__ = ["TriangulationEstimate", "projection_matrix", "triangulate_dlt"]
