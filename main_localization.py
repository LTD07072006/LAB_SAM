"""New detector-independent 2D-to-3D fire localisation workflow.

Pipeline:
    upstream detector -> Detection2D adapter -> ROI refiner -> undistortion
    -> bottom-contact/multi-ray casting -> robust 3D aggregation
    -> uncertainty -> optional temporal tracking.

The local ``FireDetector`` is only a convenient upstream baseline. Replace it
with a detector owned by another project by passing a callable that returns a
dict accepted by ``detector_adapter.adapt_detection``.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from camera_calibration import CameraCalibration
from config import DEFAULT_CONFIG
from detector_adapter import Detection2D, adapt_detection, adapt_local_detector_result
from localization import (
    FireLocalization,
    localize_pixels,
    localize_pixels_ray_plane,
    localize_with_uncertainty,
)
from locator import CameraGeometry, GridMap
from mesh_loader import load_triangle_mesh
from project_paths import FIRE_SAMPLES
from tracking_3d import Fire3DTracker
from temporal_filter import DetectionSmoother


def default_calibration(image_size: tuple[int, int] = (1280, 720)) -> CameraCalibration:
    """Create the old synthetic camera as an explicit fallback only."""
    width, height = image_size
    K = np.array([[800.0, 0.0, width / 2.0], [0.0, 800.0, height / 2.0], [0.0, 0.0, 1.0]])
    camera_position = np.array([0.0, -25.0, 30.0])
    target = np.array([0.0, 0.0, 0.0])
    forward = (target - camera_position) / np.linalg.norm(target - camera_position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    t = -R @ camera_position.reshape(3, 1)
    return CameraCalibration(K, np.zeros(5), R, t, (width, height))


def _json_value(value: Any):
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
    """One-frame/sequence pipeline with detector and refiner boundaries."""

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
        roi_expansion_factor: float = 0.0,
        roi_max_fraction: float = 1.0,
    ):
        self.detector = detector
        self.calibration = calibration
        self.geometry = calibration.geometry()
        self.grid_map = grid_map or GridMap()
        self.refiner = refiner
        self.threshold = float(threshold)
        self.use_uncertainty = bool(use_uncertainty)
        self.tracker = tracker
        self.temporal = temporal
        self.ray_kwargs = dict(ray_kwargs or {})
        self.roi_expansion_factor = float(roi_expansion_factor)
        self.roi_max_fraction = float(roi_max_fraction)
        if not np.isfinite(self.roi_expansion_factor) or self.roi_expansion_factor < 0.0:
            raise ValueError("roi_expansion_factor must be finite and non-negative")
        if not np.isfinite(self.roi_max_fraction) or self.roi_max_fraction <= 0.0:
            raise ValueError("roi_max_fraction must be finite and positive")

    def _localize(
        self,
        pixels: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
    ) -> FireLocalization:
        """Localise pixels with the configured plane or mesh backend.

        ``RayPlaneIntersectionLocator`` is the low-latency path.  A failed
        plane hypothesis is optionally handed to the existing mesh/grid
        backend so a bad floor prior does not silently produce a metre-scale
        coordinate.
        """

        backend = str(self.ray_kwargs.get("backend", "mesh")).lower()
        if backend not in {"mesh", "plane"}:
            raise ValueError(f"Unsupported ray backend: {backend!r}")
        common = dict(
            max_dist=float(self.ray_kwargs.get("max_dist", DEFAULT_CONFIG.ray_max_distance)),
            step=float(self.ray_kwargs.get("step", DEFAULT_CONFIG.ray_coarse_step)),
            bisection_iterations=int(
                self.ray_kwargs.get("bisection_iterations", DEFAULT_CONFIG.ray_bisection_iterations)
            ),
        )
        if backend == "mesh":
            return localize_pixels(
                self.geometry,
                self.grid_map,
                pixels,
                weights=weights,
                **common,
            )

        plane = tuple(float(value) for value in self.ray_kwargs.get("plane", (0.0, 0.0, 1.0, 0.0)))
        if len(plane) != 4:
            raise ValueError("ray_kwargs['plane'] must contain (a,b,c,d)")
        plane_result = localize_pixels_ray_plane(
            self.calibration,
            pixels,
            plane=plane,
            weights=weights,
            device=self.ray_kwargs.get("plane_device"),
            eps=float(self.ray_kwargs.get("plane_eps", 1e-8)),
            extrinsic_convention=str(
                self.ray_kwargs.get("plane_extrinsic_convention", "world_to_camera")
            ),
            # ``pixels`` has already passed through CameraCalibration above.
            undistort=False,
        )
        if plane_result.hit or not bool(self.ray_kwargs.get("plane_fallback", True)):
            return plane_result

        reason = plane_result.fallback_reason or plane_result.status
        mesh_result = localize_pixels(
            self.geometry,
            self.grid_map,
            pixels,
            weights=weights,
            **common,
        )
        mesh_result.backend = "mesh_fallback"
        mesh_result.fallback_reason = reason
        mesh_result.total_rays = len(np.asarray(pixels).reshape(-1, 2))
        if mesh_result.hit:
            mesh_result.status = "valid_mesh_fallback_from_plane"
        else:
            mesh_result.status = "mesh_fallback_failed_after_plane"
        return mesh_result

    def _detect(self, image: Image.Image) -> Detection2D:
        if callable(self.detector):
            raw = self.detector(image)
        elif hasattr(self.detector, "detect"):
            try:
                raw = self.detector.detect(image, warmup=False)
            except TypeError:
                # External detectors often expose detect(image) only.
                raw = self.detector.detect(image)
        else:
            raw = self.detector
        if raw.__class__.__name__ == "DetectionResult":
            return adapt_local_detector_result(raw, image.size, self.threshold)
        return adapt_detection(raw, image.size, self.threshold)

    def process(self, image: Image.Image | np.ndarray | str | Path) -> dict[str, Any]:
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = Image.open(image).convert("RGB")
        else:
            image_path = None
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image).astype(np.uint8)[..., :3])
            image = image.convert("RGB")

        start = time.perf_counter()
        detection = self._detect(image)
        temporal_state = None
        # A stability gate retains the fire alert but rejects this frame's
        # position. Do not let a previous temporal point bypass that reject.
        soft_crop = (
            self.refiner is not None
            and self.roi_expansion_factor > 0.0
            and detection.detected
            and detection.point is None
            and detection.raw_point is not None
        )
        if not detection.localization_accepted and not soft_crop:
            if self.temporal is not None:
                temporal_state = self.temporal.update(None, detection.confidence)
            return self._result(image_path, detection, None, None, start, temporal_state)
        if soft_crop:
            detection.point = detection.raw_point
            detection.localization_status = "soft_crop"
        if self.temporal is not None:
            temporal_state = self.temporal.update(detection.point, detection.confidence)
            if temporal_state.confirmed and temporal_state.pixel is not None:
                detection.point = temporal_state.pixel
            elif not temporal_state.confirmed:
                return self._result(image_path, detection, None, None, start, temporal_state)
        if not detection.detected:
            return self._result(image_path, detection, None, None, start, temporal_state)

        refined = None
        if self.refiner is not None and detection.point is not None:
            refined = self.refiner.refine(
                image,
                detection.point,
                stability_px=detection.stability_px,
                expansion_factor=self.roi_expansion_factor,
                max_roi_fraction=self.roi_max_fraction,
            )

        # For segmentation/bbox detectors, preserve geometric bottom contour.
        # For point-only detectors, use the refiner point and uncertainty
        # propagation rather than inventing an arbitrary physical width.
        if refined is not None:
            primary_pixel = np.asarray(refined.point, dtype=np.float64)
        elif detection.point is not None:
            primary_pixel = np.asarray(detection.point, dtype=np.float64)
        else:
            primary_pixel = None
        contact_pixels = detection.contact_pixels(
            columns=int(self.ray_kwargs.get("columns", 7)),
            bottom_fraction=float(self.ray_kwargs.get("bottom_fraction", 0.18)),
        )
        if detection.mask is None and detection.bbox is None and primary_pixel is not None:
            # Point-only upstream detector: the refined point is the only
            # geometrically meaningful input. Do not silently ray-cast the
            # stale coarse point.
            contact_pixels = primary_pixel.reshape(1, 2)
        elif len(contact_pixels) == 0 and primary_pixel is not None:
            contact_pixels = primary_pixel.reshape(1, 2)
        elif primary_pixel is not None and (detection.bbox is not None or detection.mask is not None):
            # Refined point replaces the uncertain box-bottom midpoint while
            # retaining neighbouring contour rays for robust aggregation.
            contact_pixels = np.vstack([primary_pixel, contact_pixels])
        if len(contact_pixels) == 0:
            return self._result(image_path, detection, refined, None, start, temporal_state)

        undistorted = self.calibration.undistort_pixels(contact_pixels)
        configured_backend = str(self.ray_kwargs.get("backend", "mesh")).lower()
        if configured_backend not in {"mesh", "plane"}:
            raise ValueError(f"Unsupported ray backend: {configured_backend!r}")
        if self.use_uncertainty and len(undistorted) == 1 and configured_backend == "mesh":
            location = localize_with_uncertainty(
                self.geometry, self.grid_map, undistorted[0],
                pixel_sigma=float(self.ray_kwargs.get("pixel_sigma", DEFAULT_CONFIG.uncertainty_pixel_sigma)),
                samples=int(self.ray_kwargs.get("samples", DEFAULT_CONFIG.uncertainty_samples)),
                max_dist=float(self.ray_kwargs.get("max_dist", DEFAULT_CONFIG.ray_max_distance)),
                step=float(self.ray_kwargs.get("step", DEFAULT_CONFIG.ray_coarse_step)),
                bisection_iterations=int(self.ray_kwargs.get("bisection_iterations", DEFAULT_CONFIG.ray_bisection_iterations)),
            )
        elif self.use_uncertainty and len(undistorted) == 1:
            # Plane intersections are algebraic and cheap, so propagate the
            # single-pixel uncertainty as one vectorised cloud as well.
            rng = np.random.default_rng(int(self.ray_kwargs.get("seed", 42)))
            sigma = max(
                0.0,
                float(self.ray_kwargs.get("pixel_sigma", DEFAULT_CONFIG.uncertainty_pixel_sigma)),
            )
            samples = max(1, int(self.ray_kwargs.get("samples", DEFAULT_CONFIG.uncertainty_samples)))
            cloud = undistorted[0] + rng.normal(0.0, sigma, size=(samples, 2))
            location = self._localize(cloud)
            location.samples = samples
        else:
            location = self._localize(undistorted)

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

        return self._result(image_path, detection, refined, location, start, temporal_state, track_state)

    @staticmethod
    def _result(image_path, detection, refined, location, start, temporal_state=None, track_state=None):
        return {
            "image_path": image_path,
            "detection": asdict(detection),
            "refined": None if refined is None else {
                "point": refined.point, "confidence": refined.confidence,
                "roi": asdict(refined.roi),
            },
            "location": None if location is None else {
                "hit": location.hit, "status": location.status,
                "point": location.point, "confidence": location.confidence,
                "spread": location.spread, "std": location.std,
                "covariance": location.covariance, "samples": location.samples,
                "ray_points": len(location.points) if location.points is not None else 0,
                "backend": getattr(location, "backend", "mesh"),
                "valid_rays": getattr(location, "valid_rays", 0),
                "total_rays": getattr(location, "total_rays", 0),
                "fallback_reason": getattr(location, "fallback_reason", None),
            },
            "temporal": None if temporal_state is None else asdict(temporal_state),
            "track": None if track_state is None else asdict(track_state),
            "latency_ms": (time.perf_counter() - start) * 1000.0,
        }


def run_samples(args):
    from fire_detector import FireDetector

    detector = FireDetector(
        args.model,
        device=args.device,
        threshold=args.threshold,
        tta_mode=args.tta_mode,
        tta_reliability_scale_px=args.tta_reliability_scale_px,
        max_stability_px=args.max_stability_px,
    )
    sample_paths = sorted(
        p for p in args.samples.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not sample_paths:
        raise FileNotFoundError(f"No images found in {args.samples}")
    first_size = Image.open(sample_paths[0]).size
    calibration = CameraCalibration.from_json(args.calibration) if args.calibration else default_calibration(first_size)
    refiner = None
    if args.roi_checkpoint and args.roi_checkpoint.is_file():
        from narrow_localizer import ROIRefinerInference
        refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device)
    tracker = None if not args.sequence else Fire3DTracker(
        DEFAULT_CONFIG.tracker_alpha, DEFAULT_CONFIG.tracker_gate_m, DEFAULT_CONFIG.tracker_max_missed,
    )
    temporal = None if not args.sequence else DetectionSmoother(
        DEFAULT_CONFIG.temporal_alpha, DEFAULT_CONFIG.temporal_window,
        DEFAULT_CONFIG.temporal_min_hits, threshold=args.threshold,
    )
    grid_map = load_triangle_mesh(args.mesh) if args.mesh else GridMap()
    pipeline = Fire3DLocalizationPipeline(
        detector, calibration, grid_map=grid_map, refiner=refiner, threshold=args.threshold,
        use_uncertainty=not args.no_uncertainty, tracker=tracker, temporal=temporal,
        ray_kwargs={
            "backend": args.ray_backend,
            "plane": tuple(args.plane),
            "plane_fallback": not args.no_plane_fallback,
            "plane_device": args.plane_device,
        },
        roi_expansion_factor=args.roi_expansion_factor,
        roi_max_fraction=args.roi_max_fraction,
    )
    outputs = []
    for path in sample_paths:
        result = pipeline.process(path)
        outputs.append(result)
        loc = result["location"]
        print(f"{path.name:24} p={result['detection']['confidence']:.3f} "
              f"refined={result['refined'] is not None} "
              f"status={loc['status'] if loc else 'not_localized'} "
              f"latency={result['latency_ms']:.1f}ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_value(outputs), indent=2), encoding="utf-8")
    print(f"saved={args.output} images={len(outputs)}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=FIRE_SAMPLES)
    parser.add_argument("--model", type=Path, default=root / "fire-model-data" / "best.pth")
    parser.add_argument("--roi-checkpoint", type=Path, default=root / "week6_roi_result" / "best_roi.pth")
    parser.add_argument("--calibration", type=Path, default=None, help="JSON calibration; default is synthetic fallback")
    parser.add_argument("--mesh", type=Path, default=None, help="JSON triangle mesh in the calibration world frame")
    parser.add_argument("--output", type=Path, default=root / "working" / "localization_results.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG.confidence_threshold)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--tta-mode",
        choices=("none", "brightness3", "photometric5"),
        default="none",
        help="Optional geometry-preserving photometric TTA before ROI",
    )
    parser.add_argument("--tta-reliability-scale-px", type=float, default=8.0)
    parser.add_argument(
        "--max-stability-px",
        type=float,
        default=None,
        help="Reject unstable coarse localisation; calibrate on validation",
    )
    parser.add_argument(
        "--roi-expansion-factor", type=float, default=0.0,
        help="Soft-gate ROI expansion in px per stability_px; 0 keeps hard-gate behavior",
    )
    parser.add_argument(
        "--roi-max-fraction", type=float, default=1.0,
        help="Maximum dynamic ROI side as a fraction of the longer image dimension",
    )
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument(
        "--ray-backend",
        choices=("mesh", "plane"),
        default="mesh",
        help="3D intersection backend; plane uses closed-form ray-plane intersections",
    )
    parser.add_argument(
        "--plane",
        type=float,
        nargs=4,
        metavar=("A", "B", "C", "D"),
        default=(0.0, 0.0, 1.0, 0.0),
        help="World plane coefficients a,b,c,d for aX+bY+cZ+d=0",
    )
    parser.add_argument(
        "--no-plane-fallback",
        action="store_true",
        help="Do not fall back to the mesh backend when a plane hit is invalid",
    )
    parser.add_argument(
        "--plane-device",
        default=None,
        help="Torch device for batched plane intersections (for example cpu or cuda)",
    )
    parser.add_argument("--sequence", action="store_true", help="Enable temporal smoothing and 3D tracking for ordered video frames")
    args = parser.parse_args()
    run_samples(args)


if __name__ == "__main__":
    main()
