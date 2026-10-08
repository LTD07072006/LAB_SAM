"""Metric pixel-to-world regression helpers.

This module keeps the physical projection as the source of truth.  The
``GPRResidualMapper`` learns only a residual correction on top of a calibrated
homography/ray-casting estimate; it is not a replacement for camera geometry.

The scikit-learn dependency is optional.  When it is unavailable, the class
falls back to a small inverse-distance residual regressor so the rest of the
benchmark remains runnable on a clean CPU environment.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from homography_floor import FloorHomography


def _finite_point(value: Any, dimensions: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(point) < dimensions or not np.all(np.isfinite(point[:dimensions])):
        return None
    return point[:dimensions].copy()


@dataclass
class ProjectionEstimate:
    """A world-coordinate estimate and a compact uncertainty description."""

    point: Optional[np.ndarray]
    std_m: Optional[np.ndarray] = None
    confidence: float = 0.0
    method: str = "unknown"
    success: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": None if self.point is None else np.asarray(self.point).tolist(),
            "std_m": None if self.std_m is None else np.asarray(self.std_m).tolist(),
            "confidence": float(self.confidence),
            "method": self.method,
            "success": bool(self.success),
            "reason": self.reason,
        }


class HomographyMapper:
    """Map image pixels to the metric floor using :class:`FloorHomography`."""

    def __init__(
        self,
        homography: FloorHomography,
        calibration: Any = None,
        default_std_m: float = 0.08,
    ) -> None:
        self.homography = homography
        self.calibration = calibration
        self.default_std_m = float(max(1e-6, default_std_m))

    @classmethod
    def from_calibration(cls, calibration: Any, default_std_m: float = 0.08) -> "HomographyMapper":
        return cls(FloorHomography.from_calibration(calibration), calibration, default_std_m)

    @classmethod
    def from_correspondences(
        cls,
        image_points: Sequence[Sequence[float]],
        floor_points_xy: Sequence[Sequence[float]],
        default_std_m: float = 0.08,
    ) -> "HomographyMapper":
        return cls(
            FloorHomography.from_correspondences(image_points, floor_points_xy),
            None,
            default_std_m,
        )

    def predict(self, pixel: Sequence[float]) -> ProjectionEstimate:
        point = _finite_point(pixel, 2)
        if point is None:
            return ProjectionEstimate(None, method="ipm", reason="invalid_pixel")
        try:
            if self.calibration is not None:
                point = self.calibration.undistort_pixels(point.reshape(1, 2))[0]
            xyz = self.homography.pixel_to_floor_xyz(point.reshape(1, 2))[0]
            return ProjectionEstimate(
                xyz,
                np.full(3, self.default_std_m, dtype=np.float64),
                confidence=1.0,
                method="ipm",
                success=True,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ProjectionEstimate(None, method="ipm", reason=str(exc))


class GPRResidualMapper:
    """Learn ``ground_truth_xyz - physical_estimate_xyz`` from calibration data.

    Features contain normalized pixel coordinates and the physical base point.
    Including the base point makes the correction useful across a room while
    preserving the calibrated geometry.  The model is deliberately capped by
    ``max_samples`` because exact Gaussian processes scale cubically.
    """

    def __init__(
        self,
        feature_mean: Optional[np.ndarray] = None,
        feature_scale: Optional[np.ndarray] = None,
        default_std_m: float = 0.20,
    ) -> None:
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = None if feature_scale is None else np.asarray(feature_scale, dtype=np.float64)
        self.default_std_m = float(max(1e-6, default_std_m))
        self.backend = "unfitted"
        self.models: list[Any] = []
        self.train_features = np.empty((0, 5), dtype=np.float64)
        self.train_residuals = np.empty((0, 3), dtype=np.float64)

    @staticmethod
    def make_features(
        pixels: Sequence[Sequence[float]] | Sequence[float],
        base_xyz: Sequence[Sequence[float]] | Sequence[float],
        image_sizes: Sequence[Sequence[float]] | Sequence[float] = (640.0, 640.0),
    ) -> np.ndarray:
        pixel_array = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        base_array = np.asarray(base_xyz, dtype=np.float64).reshape(-1, 3)
        sizes = np.asarray(image_sizes, dtype=np.float64)
        if sizes.ndim == 1:
            sizes = np.repeat(sizes.reshape(1, 2), len(pixel_array), axis=0)
        sizes = sizes.reshape(-1, 2)
        if len(pixel_array) != len(base_array) or len(pixel_array) != len(sizes):
            raise ValueError("pixels, base_xyz and image_sizes must have the same length")
        if np.any(sizes <= 0.0) or not np.all(np.isfinite(sizes)):
            raise ValueError("image_sizes must be finite and positive")
        features = np.column_stack((pixel_array / sizes, base_array))
        if not np.all(np.isfinite(features)):
            raise ValueError("GPR features contain non-finite values")
        return features

    @staticmethod
    def _subsample(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
        if maximum <= 0 or len(values) <= maximum:
            return np.arange(len(values), dtype=np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(len(values), size=int(maximum), replace=False))

    def fit(
        self,
        pixels: Sequence[Sequence[float]],
        base_xyz: Sequence[Sequence[float]],
        target_xyz: Sequence[Sequence[float]],
        image_sizes: Sequence[Sequence[float]] | Sequence[float] = (640.0, 640.0),
        max_samples: int = 256,
        seed: int = 42,
    ) -> "GPRResidualMapper":
        features = self.make_features(pixels, base_xyz, image_sizes)
        targets = np.asarray(target_xyz, dtype=np.float64).reshape(-1, 3)
        bases = np.asarray(base_xyz, dtype=np.float64).reshape(-1, 3)
        if len(features) != len(targets) or len(targets) != len(bases) or len(features) < 4:
            raise ValueError("GPR residual fitting needs at least four matching samples")
        if not np.all(np.isfinite(targets)):
            raise ValueError("target_xyz contains non-finite values")
        indices = self._subsample(features, max_samples, seed)
        features = features[indices]
        residuals = targets[indices] - bases[indices]
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = features.std(axis=0)
        self.feature_scale = np.maximum(self.feature_scale, 1e-6)
        standardized = (features - self.feature_mean) / self.feature_scale
        self.train_features = standardized
        self.train_residuals = residuals
        self.models = []

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.ones(standardized.shape[1]),
                length_scale_bounds=(1e-2, 1e2),
            ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
            for axis in range(3):
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    n_restarts_optimizer=0,
                    random_state=seed + axis,
                )
                model.fit(standardized, residuals[:, axis])
                self.models.append(model)
            self.backend = "sklearn_gpr"
        except (ImportError, ValueError, RuntimeError):
            # IDW is deterministic, dependency-free, and adequate as a
            # fallback correction for a small calibration grid.
            self.backend = "idw_residual"
        return self

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("GPRResidualMapper is not fitted")
        return (features - self.feature_mean) / self.feature_scale

    def predict(
        self,
        pixel: Sequence[float],
        base_xyz: Sequence[float],
        image_size: Sequence[float] = (640.0, 640.0),
    ) -> ProjectionEstimate:
        base = _finite_point(base_xyz, 3)
        point = _finite_point(pixel, 2)
        if base is None or point is None:
            return ProjectionEstimate(None, method="gpr_residual", reason="missing_base_or_pixel")
        if self.backend == "unfitted":
            return ProjectionEstimate(None, method="gpr_residual", reason="not_fitted")
        feature = self.make_features(point, base, image_size)
        standardized = self._standardize(feature)
        if self.backend == "sklearn_gpr":
            means: list[float] = []
            stds: list[float] = []
            for model in self.models:
                mean, std = model.predict(standardized, return_std=True)
                means.append(float(mean[0]))
                stds.append(float(max(std[0], self.default_std_m * 0.25)))
            residual = np.asarray(means, dtype=np.float64)
            std = np.asarray(stds, dtype=np.float64)
        else:
            distances = np.linalg.norm(self.train_features - standardized[0], axis=1)
            exact = np.flatnonzero(distances < 1e-9)
            if len(exact):
                residual = self.train_residuals[int(exact[0])]
                std = np.full(3, self.default_std_m * 0.5, dtype=np.float64)
            else:
                weights = 1.0 / np.maximum(distances, 1e-6) ** 2
                weights /= weights.sum()
                residual = weights @ self.train_residuals
                variance = weights @ (self.train_residuals - residual) ** 2
                std = np.sqrt(np.maximum(variance, (self.default_std_m * 0.25) ** 2))
        return ProjectionEstimate(
            base + residual,
            std,
            confidence=float(np.clip(1.0 / (1.0 + float(np.mean(std))), 0.0, 1.0)),
            method="gpr_residual",
            success=True,
        )

    def save(self, path: Path | str) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> "GPRResidualMapper":
        with Path(path).expanduser().resolve().open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__} in {path}")
        return value


__all__ = ["GPRResidualMapper", "HomographyMapper", "ProjectionEstimate"]
