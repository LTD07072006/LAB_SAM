"""Multi-ray fire localisation and 3D uncertainty estimation.

The detector can provide a point, a bounding box, or a binary mask. This
module converts several image points into rays, intersects them with the
known scene surface, removes 3D outliers, and reports a position together
with a practical uncertainty estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from locator import CameraGeometry, GridMap, RayHit, intersect_ray_with_grid_result
from point_filter import robust_filter_points


@dataclass
class FireLocalization:
    """Backward-compatible result plus multi-ray diagnostics."""

    hit: bool
    point: Optional[np.ndarray] = None
    status: str = "no_hit"
    points: np.ndarray = None
    kept_indices: np.ndarray = None
    spread: float = 0.0
    confidence: float = 0.0
    covariance: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    samples: int = 0


def bottom_contact_pixels(bbox: Sequence[float], columns: int = 5, bottom_fraction: float = 0.18) -> np.ndarray:
    """Sample a horizontal band near the bottom of ``(x1,y1,x2,y2)``.

    The bottom band is more meaningful for ground-contact localisation than
    the box centre. It also gives the 3D stage several rays to aggregate.
    """
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64).reshape(4)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    xs = np.linspace(x1 + 0.15 * width, x2 - 0.15 * width, max(2, int(columns)))
    ys = np.full_like(xs, y2 - 0.5 * min(height, max(1.0, bottom_fraction * height)))
    return np.column_stack([xs, ys])


def mask_bottom_contact_pixels(mask: np.ndarray, columns: int = 7, quantile: float = 0.85) -> np.ndarray:
    """Extract representative bottom-contour pixels from a binary mask."""
    array = np.asarray(mask).astype(bool)
    if array.ndim != 2 or not array.any():
        return np.empty((0, 2), dtype=np.float64)
    ys, xs = np.where(array)
    threshold = np.quantile(ys, np.clip(quantile, 0.0, 1.0))
    bottom = np.column_stack([xs[ys >= threshold], ys[ys >= threshold]])
    if len(bottom) == 0:
        return np.empty((0, 2), dtype=np.float64)
    selected = []
    for x in np.linspace(bottom[:, 0].min(), bottom[:, 0].max(), max(2, int(columns))):
        band = bottom[np.abs(bottom[:, 0] - x) <= max(1.0, (np.ptp(bottom[:, 0]) / max(columns, 1)))]
        selected.append(band[np.argmax(band[:, 1])] if len(band) else bottom[np.argmin(np.abs(bottom[:, 0] - x))])
    return np.asarray(selected, dtype=np.float64)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.maximum(weights, 1e-9)
    result = []
    for axis in range(values.shape[1]):
        order = np.argsort(values[:, axis])
        cumulative = np.cumsum(weights[order])
        result.append(values[order[np.searchsorted(cumulative, cumulative[-1] * 0.5)], axis])
    return np.asarray(result, dtype=np.float64)


def localize_pixels(
    camera: CameraGeometry,
    grid_map: GridMap,
    pixels: Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
    max_dist: float = 1000.0,
    step: float = 2.0,
    bisection_iterations: int = 24,
) -> FireLocalization:
    """Localise several 2D points and robustly aggregate valid 3D hits."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(pixels) == 0:
        return FireLocalization(False, status="no_pixels", points=np.empty((0, 3)), kept_indices=np.empty(0, dtype=int))
    if weights is None:
        weights = np.ones(len(pixels), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(pixels):
        raise ValueError("weights must have one value per pixel")

    origins, rays = camera.pixels_to_rays(pixels)
    hits = []
    hit_weights = []
    for origin, ray, weight in zip(origins, rays, weights):
        result: RayHit = intersect_ray_with_grid_result(
            origin, ray, grid_map, max_dist=max_dist, step=step,
            bisection_iterations=bisection_iterations,
        )
        if result.hit and result.point is not None and np.all(np.isfinite(result.point)):
            hits.append(np.asarray(result.point, dtype=np.float64))
            hit_weights.append(float(weight))
    if not hits:
        return FireLocalization(False, status="no_valid_intersection", points=np.empty((0, 3)), kept_indices=np.empty(0, dtype=int), samples=len(pixels))

    points = np.asarray(hits, dtype=np.float64)
    filtered = robust_filter_points(points, min_points=3)
    kept_weights = np.asarray(hit_weights)[filtered.kept_indices]
    center = _weighted_median(filtered.points, kept_weights) if len(filtered.points) else filtered.center
    centered = filtered.points - center
    covariance = np.cov(centered.T, aweights=kept_weights, ddof=0) if len(filtered.points) >= 2 else np.zeros((3, 3), dtype=np.float64)
    covariance = np.atleast_2d(covariance)
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    confidence = float(np.clip(filtered.confidence * (len(points) / len(pixels)), 0.0, 1.0))
    return FireLocalization(
        True,
        point=center,
        status="valid_multi_ray",
        points=filtered.points,
        kept_indices=filtered.kept_indices,
        spread=filtered.spread,
        confidence=confidence,
        covariance=covariance,
        std=std,
        samples=len(pixels),
    )


def localize_with_uncertainty(
    camera: CameraGeometry,
    grid_map: GridMap,
    pixel: Sequence[float],
    pixel_sigma: float = 2.0,
    samples: int = 128,
    seed: int = 42,
    **kwargs,
) -> FireLocalization:
    """Monte Carlo propagation from 2D pixel uncertainty to 3D position."""
    base = np.asarray(pixel, dtype=np.float64).reshape(2)
    rng = np.random.default_rng(seed)
    cloud = base + rng.normal(0.0, max(0.0, float(pixel_sigma)), size=(max(1, int(samples)), 2))
    result = localize_pixels(camera, grid_map, cloud, **kwargs)
    result.samples = len(cloud)
    return result
