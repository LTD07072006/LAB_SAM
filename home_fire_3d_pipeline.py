"""Independent 2D-to-3D pipeline for the Home Fire YOLO branch.

The detector is intentionally injected into this module.  That keeps the
geometry code independent from both the legacy ``FireDetector`` and the
Week-6 ROI entry point::

    YOLO bbox -> bottom-band pixels -> undistortion -> camera ray
    -> metric mesh/GridMap intersection -> robust 3D point
    -> uncertainty -> optional temporal filtering/tracking

All pixels passed to calibration are in the original image coordinate system.
Ultralytics may resize an image internally, but ``HomeFireYOLO`` converts its
boxes back to source-image pixels before this module sees them.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from camera_calibration import CameraCalibration
from config import DEFAULT_CONFIG
from detector_adapter import Detection2D, adapt_detection, adapt_local_detector_result
from localization import localize_pixels, localize_with_uncertainty
from locator import GridMap
from temporal_filter import DetectionSmoother
from tracking_3d import Fire3DTracker


def default_calibration(image_size: tuple[int, int] = (1280, 720)) -> CameraCalibration:
    """Return the historical synthetic camera as an explicit fallback.

    This is useful for a software smoke test only.  It must not be used to
    report physical metre errors for the YOLO dataset.
    """

    width, height = image_size
    K = np.array(
        [
            [800.0, 0.0, width / 2.0],
            [0.0, 800.0, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_position = np.array([0.0, -25.0, 30.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    forward = (target - camera_position) / np.linalg.norm(target - camera_position)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    t = -R @ camera_position.reshape(3, 1)
    return CameraCalibration(K, np.zeros(5), R, t, (width, height))


def _json_value(value: Any) -> Any:
    """Convert NumPy/dataclass-adjacent values into JSON-safe values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class Fire3DLocalizationPipeline:
    """Run detector, optional ROI refinement, ray casting and tracking.

    ``detector`` can be an object exposing ``detect(image)`` or a callable.
    It must return a value accepted by :func:`detector_adapter.adapt_detection`.
    ``HomeFireYOLO.detect`` already returns the repository's ``Detection2D``
    contract, so no YOLO-specific code is needed here.
    """

    def __init__(
        self,
        detector: Any,
        calibration: CameraCalibration,
        grid_map: Optional[GridMap] = None,
        refiner: Optional[Any] = None,
        threshold: float = 0.5,
        use_uncertainty: bool = True,
        tracker: Optional[Fire3DTracker] = None,
        temporal: Optional[DetectionSmoother] = None,
        ray_kwargs: Optional[dict[str, Any]] = None,
        max_refine_shift_px: Optional[float] = 80.0,
        refine_blend: float = 0.75,
        min_refine_confidence: float = 0.15,
    ) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if max_refine_shift_px is not None and float(max_refine_shift_px) < 0.0:
            raise ValueError("max_refine_shift_px must be non-negative or None")
        if not 0.0 <= float(refine_blend) <= 1.0:
            raise ValueError("refine_blend must be in [0, 1]")
        if not 0.0 <= float(min_refine_confidence) <= 1.0:
            raise ValueError("min_refine_confidence must be in [0, 1]")
        self.detector = detector
        self.calibration = calibration
        self.geometry = calibration.geometry()
        self.grid_map = grid_map if grid_map is not None else GridMap()
        self.refiner = refiner
        self.threshold = float(threshold)
        self.use_uncertainty = bool(use_uncertainty)
        self.tracker = tracker
        self.temporal = temporal
        self.ray_kwargs = dict(ray_kwargs or {})
        self.max_refine_shift_px = (
            None if max_refine_shift_px is None else float(max_refine_shift_px)
        )
        self.refine_blend = float(refine_blend)
        self.min_refine_confidence = float(min_refine_confidence)

    def _detect(self, image: Image.Image) -> Detection2D:
        if callable(self.detector):
            raw = self.detector(image)
        elif hasattr(self.detector, "detect"):
            try:
                raw = self.detector.detect(image, warmup=False)
            except TypeError:
                # Some external detectors expose only detect(image).
                raw = self.detector.detect(image)
        else:
            raw = self.detector

        if raw.__class__.__name__ == "DetectionResult":
            return adapt_local_detector_result(raw, image.size, self.threshold)
        if isinstance(raw, Detection2D):
            return raw
        return adapt_detection(raw, image.size, self.threshold)

    def process(self, image: Image.Image | np.ndarray | str | Path) -> dict[str, Any]:
        """Process one image and return serialisable diagnostics laterally."""

        image_path: Optional[str]
        if isinstance(image, (str, Path)):
            image_path = str(image)
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        else:
            image_path = None
            if not isinstance(image, Image.Image):
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[2] < 3:
                    raise ValueError("image array must have shape HxWx3 or HxWx4")
                image = Image.fromarray(array[..., :3].astype(np.uint8))
            image = image.convert("RGB")

        start = time.perf_counter()
        detection = self._detect(image)

        temporal_state = None
        if self.temporal is not None:
            temporal_state = self.temporal.update(detection.point, detection.confidence)
            if temporal_state.confirmed and temporal_state.pixel is not None:
                detection.point = temporal_state.pixel
            elif not temporal_state.confirmed:
                return self._result(
                    image_path, detection, None, None, start, temporal_state
                )

        if not detection.detected:
            return self._result(image_path, detection, None, None, start, temporal_state)

        refined = None
        if self.refiner is not None and detection.point is not None:
            refined = self.refiner.refine(image, detection.point)

        primary_pixel, refinement = self._resolve_refinement(detection, refined)

        contact_pixels = detection.contact_pixels(
            columns=int(self.ray_kwargs.get("columns", DEFAULT_CONFIG.multi_ray_columns)),
            bottom_fraction=float(
                self.ray_kwargs.get(
                    "bottom_fraction", DEFAULT_CONFIG.multi_ray_bottom_fraction
                )
            ),
        )
        if detection.mask is None and detection.bbox is None and primary_pixel is not None:
            # Point-only detectors have no justified neighbouring contact
            # geometry: propagate uncertainty around the one refined point.
            contact_pixels = primary_pixel.reshape(1, 2)
        elif len(contact_pixels) == 0 and primary_pixel is not None:
            contact_pixels = primary_pixel.reshape(1, 2)
        elif primary_pixel is not None and (
            detection.bbox is not None or detection.mask is not None
        ):
            # Keep YOLO's bottom band for multi-ray robustness, but include the
            # refined contact point as the highest-priority geometric sample.
            contact_pixels = np.vstack([primary_pixel, contact_pixels])

        if len(contact_pixels) == 0:
            return self._result(
                image_path,
                detection,
                refined,
                None,
                start,
                temporal_state,
                refinement=refinement,
            )

        # Calibration owns distortion correction.  It receives source-image
        # pixels, never the detector's internal 640x640 letterbox coordinates.
        undistorted = self.calibration.undistort_pixels(contact_pixels)
        location_kwargs = {
            "max_dist": float(
                self.ray_kwargs.get("max_dist", DEFAULT_CONFIG.ray_max_distance)
            ),
            "step": float(self.ray_kwargs.get("step", DEFAULT_CONFIG.ray_coarse_step)),
            "bisection_iterations": int(
                self.ray_kwargs.get(
                    "bisection_iterations", DEFAULT_CONFIG.ray_bisection_iterations
                )
            ),
        }
        if self.use_uncertainty and len(undistorted) == 1:
            location = localize_with_uncertainty(
                self.geometry,
                self.grid_map,
                undistorted[0],
                pixel_sigma=float(
                    self.ray_kwargs.get(
                        "pixel_sigma", DEFAULT_CONFIG.uncertainty_pixel_sigma
                    )
                ),
                samples=int(
                    self.ray_kwargs.get(
                        "samples", DEFAULT_CONFIG.uncertainty_samples
                    )
                ),
                **location_kwargs,
            )
        else:
            location = localize_pixels(
                self.geometry,
                self.grid_map,
                undistorted,
                **location_kwargs,
            )

        track_state = None
        if self.tracker is not None:
            track_state = self.tracker.update(
                location.point if location.hit else None,
                location.covariance if location.hit else None,
                location.confidence if location.hit else 0.0,
            )
            if location.hit and track_state.accepted and track_state.point is not None:
                location.point = track_state.point.copy()
                location.covariance = track_state.covariance
                location.std = np.sqrt(np.maximum(np.diag(track_state.covariance), 0.0))
                location.status = "valid_tracked"

        return self._result(
            image_path,
            detection,
            refined,
            location,
            start,
            temporal_state,
            track_state,
            refinement,
        )

    def _resolve_refinement(
        self,
        detection: Detection2D,
        refined: Any,
    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        """Guard and blend an optional ROI point against the YOLO point.

        The YOLO bottom-center remains the safety baseline.  A refiner is
        accepted only when its heatmap confidence is sufficient and its shift
        is plausible.  The final point is a confidence-weighted blend rather
        than an unconditional replacement, which prevents a bad ROI crop from
        moving a ray to a completely different part of a narrow scene.
        """

        coarse = None
        if detection.point is not None:
            coarse = np.asarray(detection.point, dtype=np.float64).reshape(2)
        info: dict[str, Any] = {
            "enabled": self.refiner is not None,
            "used": False,
            "reason": "no_refiner" if self.refiner is None else "coarse_only",
            "coarse_point": None if coarse is None else coarse.copy(),
            "refined_point": None,
            "shift_px": None,
            "confidence": None,
            "weight": 0.0,
            "point": None if coarse is None else coarse.copy(),
        }
        if coarse is None or refined is None:
            return coarse, info

        candidate = np.asarray(getattr(refined, "point", None), dtype=np.float64).reshape(-1)
        if len(candidate) < 2 or not np.all(np.isfinite(candidate[:2])):
            info["reason"] = "invalid_refined_point"
            return coarse, info
        candidate = candidate[:2]
        confidence = float(np.clip(getattr(refined, "confidence", 1.0), 0.0, 1.0))
        shift = float(np.linalg.norm(candidate - coarse))
        info["refined_point"] = candidate.copy()
        info["shift_px"] = shift
        info["confidence"] = confidence

        if confidence < self.min_refine_confidence:
            info["reason"] = "low_refine_confidence"
            return coarse, info
        if self.max_refine_shift_px is not None and shift > self.max_refine_shift_px:
            info["reason"] = "refine_shift_too_large"
            return coarse, info

        weight = float(np.clip(self.refine_blend * confidence, 0.0, 1.0))
        point = (1.0 - weight) * coarse + weight * candidate
        info.update(
            {
                "used": True,
                "reason": "accepted_blend",
                "weight": weight,
                "point": point.copy(),
            }
        )
        return point, info

    @staticmethod
    def _result(
        image_path: Optional[str],
        detection: Detection2D,
        refined: Any,
        location: Any,
        start: float,
        temporal_state: Any = None,
        track_state: Any = None,
        refinement: Any = None,
    ) -> dict[str, Any]:
        return {
            "image_path": image_path,
            "detection": asdict(detection),
            "refined": None
            if refined is None
            else {
                "point": refined.point,
                "confidence": refined.confidence,
                "roi": asdict(refined.roi),
            },
            "refinement": refinement,
            "location": None
            if location is None
            else {
                "hit": location.hit,
                "status": location.status,
                "point": location.point,
                "confidence": location.confidence,
                "spread": location.spread,
                "std": location.std,
                "covariance": location.covariance,
                "samples": location.samples,
                "ray_points": len(location.points)
                if location.points is not None
                else 0,
            },
            "temporal": None
            if temporal_state is None
            else asdict(temporal_state),
            "track": None if track_state is None else asdict(track_state),
            "latency_ms": (time.perf_counter() - start) * 1000.0,
        }


__all__ = [
    "Fire3DLocalizationPipeline",
    "_json_value",
    "default_calibration",
]
