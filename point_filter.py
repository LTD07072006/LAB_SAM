"""Robust aggregation of 3D ray intersections."""
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np

@dataclass
class PointFilterResult:
    points: np.ndarray
    center: Optional[np.ndarray]
    kept_indices: np.ndarray
    spread: float
    confidence: float

def robust_filter_points(points: Iterable[Iterable[float]], mad_scale=3.5, min_points=3):
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.size == 0:
        return PointFilterResult(np.empty((0, 3)), None, np.empty(0, dtype=int), 0.0, 0.0)
    arr = arr.reshape(-1, 3)
    if len(arr) < min_points:
        center = np.median(arr, axis=0)
        spread = float(np.median(np.linalg.norm(arr - center, axis=1)))
        return PointFilterResult(arr, center, np.arange(len(arr)), spread, 0.5)
    median = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - median), axis=0)
    scale = 1.4826 * np.maximum(mad, 1e-6)
    keep = np.all(np.abs(arr - median) / scale <= mad_scale, axis=1)
    if not np.any(keep):
        keep[:] = True
    kept = arr[keep]
    center = np.median(kept, axis=0)
    distances = np.linalg.norm(kept - center, axis=1)
    spread = float(np.median(distances)) if len(distances) else 0.0
    confidence = float(np.clip((len(kept) / len(arr)) * np.exp(-spread / 2.0), 0.0, 1.0))
    return PointFilterResult(kept, center, np.flatnonzero(keep), spread, confidence)

def bcc_drop_filter(points_3d, dc_ratio=0.15):
    del dc_ratio
    return [tuple(p) for p in robust_filter_points(points_3d).points]
