"""Normalize outputs from any upstream 2D fire detector.

The 3D localisation contribution should not depend on a particular detector
implementation. This adapter accepts the local ``FireDetector`` result,
YOLO-like dictionaries, or a small custom dictionary containing a point,
bounding box, and/or mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class Detection2D:
    detected: bool
    confidence: float
    point: Optional[tuple[float, float]]
    bbox: Optional[tuple[float, float, float, float]]
    mask: Optional[np.ndarray]
    image_size: tuple[int, int]
    latency_ms: float = 0.0
    source: str = "external"

    @property
    def p_fire(self) -> float:
        return self.confidence

    def contact_pixels(self, columns: int = 7, bottom_fraction: float = 0.18) -> np.ndarray:
        """Return candidate image pixels at the fire-ground contact.

        Mask geometry is preferred over a box, and a detector point remains a
        valid fallback. The returned pixels are deliberately still in the
        original image coordinate system.
        """
        from localization import bottom_contact_pixels, mask_bottom_contact_pixels

        if self.mask is not None:
            points = mask_bottom_contact_pixels(self.mask, columns=columns)
            if len(points):
                return points
        if self.bbox is not None:
            return bottom_contact_pixels(self.bbox, columns=columns, bottom_fraction=bottom_fraction)
        if self.point is not None:
            return np.asarray([self.point], dtype=np.float64)
        return np.empty((0, 2), dtype=np.float64)


def _pair(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) < 2 or not np.all(np.isfinite(array[:2])):
        return None
    return float(array[0]), float(array[1])


def _bbox(value: Any) -> Optional[tuple[float, float, float, float]]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) < 4 or not np.all(np.isfinite(array[:4])):
        return None
    return tuple(float(x) for x in array[:4])


def _get(raw: Any, *names: str, default=None):
    if isinstance(raw, dict):
        for name in names:
            if name in raw:
                return raw[name]
        return default
    for name in names:
        if hasattr(raw, name):
            return getattr(raw, name)
    return default


def adapt_detection(raw: Any, image_size: Sequence[int], threshold: float = 0.5,
                    source: str = "external") -> Detection2D:
    """Convert a detector result to the stable ``Detection2D`` contract."""
    width, height = int(image_size[0]), int(image_size[1])
    confidence = float(np.clip(_get(raw, "confidence", "p_fire", "score", default=0.0), 0.0, 1.0))
    point = _pair(_get(raw, "point", "pixel", "center", default=None))
    bbox = _bbox(_get(raw, "bbox", "box", "xyxy", default=None))
    mask_value = _get(raw, "mask", "segmentation", default=None)
    mask = None if mask_value is None else np.asarray(mask_value).astype(bool)
    detected_value = _get(raw, "detected", "is_fire", default=None)
    detected = bool(confidence >= threshold) if detected_value is None else bool(detected_value)
    if point is None and bbox is not None:
        point = (float((bbox[0] + bbox[2]) * 0.5), float(bbox[3]))
    return Detection2D(
        detected=detected and confidence >= threshold,
        confidence=confidence,
        point=point if detected and confidence >= threshold else None,
        bbox=bbox if detected and confidence >= threshold else None,
        mask=mask if detected and confidence >= threshold else None,
        image_size=(width, height),
        latency_ms=float(_get(raw, "latency_ms", default=0.0) or 0.0),
        source=source,
    )


def adapt_local_detector_result(raw: Any, image_size: Sequence[int], threshold: float = 0.5) -> Detection2D:
    """Adapter for the repository's ``fire_detector.DetectionResult``."""
    return adapt_detection(raw, image_size, threshold=threshold, source="local_fire_detector")
