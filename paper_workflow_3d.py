"""Paper-inspired 2D-to-3D fire localisation benchmark.

This is the safe, reproducible bridge between the experiments already in this
repository and the physical-room experiment proposed in the papers supplied
with the project:

    noisy 2D query -> ROI refinement -> calibration/undistortion
        -> multi-ray mesh intersection -> robust 3D point -> tracking metrics

The default run uses the metric synthetic dataset.  It is deliberately kept
separate from the real-CCTV claim: synthetic camera/mesh/XYZ labels validate
the geometry and robustness implementation, while a real-room result is only
accepted when the user supplies measured calibration, a measured mesh and
marker-based 3D labels.

The mixed ROI checkpoint is used as a downstream point refiner only. It does
not replace the upstream fire/no-fire detector.

Example (quick synthetic run)::

    .venv\\Scripts\\python.exe paper_workflow_3d.py ^
      --dataset working\\synthetic_fire_3d_v3 ^
      --roi-checkpoint output\\roi_domain_experiments\\mixed\\best_roi.pth ^
      --output-dir output\\paper_workflow_synthetic ^
      --split test --max-records 24 --device cpu

Example with a real measured room::

    .venv\\Scripts\\python.exe paper_workflow_3d.py ^
      --dataset path\\to\\manifest_dataset ^
      --calibration measured_camera.json --mesh measured_room_mesh.json ^
      --labels-3d measured_ground_truth_3d.json ^
      --calibration-source external --coarse-source detector ^
      --detector-checkpoint fire-model-data\\best.pth ^
      --roi-checkpoint output\\roi_domain_experiments\\mixed\\best_roi.pth ^
      --output-dir output\\paper_workflow_real ^
      --split test --device cuda

The manifest contract is intentionally close to synthetic_fire_3d.py:
``image_path``, ``p_fire_pixel``, ``p_fire_noisy_pixel``,
``fire_xyz_world``, ``camera`` and ``scene_id``.  A real manifest may omit
the per-frame camera object when --calibration-source external is used.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from camera_calibration import CameraCalibration
from ekf_tracker import Fire3DEKF
from homography_floor import FloorHomography
from localization import bottom_contact_pixels, localize_pixels
from locator import intersect_ray_with_grid_result
from mesh_loader import load_triangle_mesh
from tracking_3d import Fire3DTracker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BRANCHES = ("coarse", "roi_raw", "roi_safe", "roi_blend", "ipm")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _as_point(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(point) < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return point[:2].copy()


def _as_xyz(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(point) < 3 or not np.all(np.isfinite(point[:3])):
        return None
    return point[:3].copy()


def _surface_name(row: dict[str, Any]) -> str:
    """Return a stable surface label for geometry-validity accounting."""

    value = row.get("fire_surface", row.get("surface_type", row.get("surface", "unknown")))
    text = str(value or "unknown").strip().lower()
    return {
        "ground": "floor",
        "floor_plane": "floor",
        "floor_surface": "floor",
    }.get(text, text)


def _row_pixel(row: dict[str, Any], field: str = "p_fire_pixel") -> Optional[np.ndarray]:
    """Read a pixel label, accepting either pixel or normalized coordinates."""
    point = _as_point(row.get(field))
    # Synthetic manifests use ``null`` for a missed detector observation, but
    # a literal ``[0, 0]`` is also an invalid sentinel in older manifests.
    # Never turn that miss into the top-left image pixel.
    if point is not None and not (
        field == "p_fire_noisy_pixel" and np.allclose(point, 0.0, atol=1e-12)
    ):
        return point
    normalized_field = "p_fire" if field == "p_fire_pixel" else "p_fire_noisy"
    normalized = _as_point(row.get(normalized_field))
    if normalized is not None and normalized_field == "p_fire_noisy" and np.allclose(
        normalized, 0.0, atol=1e-12
    ) and row.get("noise", {}).get("detected_observation") is False:
        return None
    size = np.asarray(row.get("image_size", [0, 0]), dtype=np.float64).reshape(-1)
    if normalized is None or len(size) < 2 or np.any(size[:2] <= 0):
        return None
    if np.all((normalized >= 0.0) & (normalized <= 1.0)):
        return normalized * size[:2]
    return normalized


def _image_path(dataset: Path, raw: Any) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (dataset / path).resolve()


def _load_manifest(dataset: Path, split: str) -> list[dict[str, Any]]:
    path = dataset / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and str(row.get("split", "")) != split:
                continue
            if int(row.get("has_fire", 0)) != 1:
                continue
            if _row_pixel(row) is None:
                continue
            row["_image_path"] = str(_image_path(dataset, row.get("image_path", "")))
            if not Path(row["_image_path"]).is_file():
                continue
            rows.append(row)
    rows.sort(key=lambda item: (str(item.get("scene_id", "")), int(item.get("frame_index", 0))))
    if not rows:
        raise RuntimeError(
            f"No visible positive records with a valid 2D label in {path} split={split!r}"
        )
    return rows


def _select_records(rows: list[dict[str, Any]], maximum: int, selection: str) -> list[dict[str, Any]]:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    if selection == "even":
        indexes = np.linspace(0, len(rows) - 1, int(maximum), dtype=int)
        selected = [rows[int(index)] for index in indexes]
    elif selection == "diverse":
        # Greedy farthest-point selection in normalized pixel/3D space.  It
        # keeps the smoke run representative without scanning every frame in
        # the downstream neural model.
        points = np.asarray([_row_pixel(row) for row in rows], dtype=np.float64)
        image_sizes = np.asarray([row.get("image_size", [640, 640]) for row in rows], dtype=np.float64)
        points = points / np.maximum(image_sizes, 1.0)
        selected_indices = [0]
        distances = np.full(len(rows), np.inf, dtype=np.float64)
        for _ in range(1, int(maximum)):
            last = points[selected_indices[-1]]
            distances = np.minimum(distances, np.linalg.norm(points - last, axis=1))
            distances[selected_indices] = -1.0
            selected_indices.append(int(np.argmax(distances)))
        selected = [rows[index] for index in sorted(selected_indices)]
    else:
        raise ValueError(f"Unknown selection policy: {selection}")
    return selected


def _calibration_from_row(row: dict[str, Any], source: str) -> CameraCalibration:
    camera = row.get("camera") if source == "true" else row.get("camera_estimated")
    if not isinstance(camera, dict):
        raise ValueError(f"Manifest row has no {source} camera calibration")
    return CameraCalibration.from_dict(
        {
            "image_size": row.get("image_size"),
            "intrinsics": {
                "K": camera.get("K"),
                "dist_coeffs": camera.get("dist_coeffs", []),
            },
            "extrinsics": {
                "R": camera.get("R_world_to_camera", camera.get("R")),
                "camera_position": camera.get("camera_position"),
            },
        }
    )


def _load_external_calibration(path: Path) -> CameraCalibration:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    metadata_text = " ".join(str(value).lower() for value in metadata.values())
    status = str(metadata.get("status", data.get("status", "unknown"))).lower()
    if any(token in f"{status} {metadata_text}" for token in ("provisional", "synthetic", "template", "example", "placeholder")):
        raise ValueError(
            f"Refusing provisional/synthetic/template calibration for external evaluation: {path}"
        )
    if status != "measured_real_room":
        raise ValueError(
            "External calibration must declare metadata.status="
            f"'measured_real_room' (got {status!r}): {path}"
        )
    return CameraCalibration.from_dict(data)


def _load_external_3d(path: Optional[Path]) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        metadata = raw.get("metadata", {})
        if isinstance(metadata, dict):
            status = str(metadata.get("status", "unknown")).lower()
            if status != "measured_real_room":
                raise ValueError(
                    "External 3D labels must declare metadata.status="
                    f"'measured_real_room' (got {status!r}): {path}"
                )
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        entries = raw["records"]
    elif isinstance(raw, dict):
        entries = [{"key": key, "value": value} for key, value in raw.items()]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"3D labels must be a JSON object/list: {path}")
    result: dict[str, np.ndarray] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key", entry.get("image", entry.get("image_path", entry.get("sample_id"))))
        value = entry.get("value", entry.get("fire_xyz_world", entry.get("xyz", entry.get("point"))))
        if key is None:
            continue
        point = _as_xyz(value)
        if point is not None:
            key_text = str(key).replace("\\", "/").lower()
            result[key_text] = point
            result[Path(key_text).name] = point
    return result


def _load_mesh(path: Path, external: bool) -> Any:
    """Load a mesh and reject synthetic/template metadata for real evaluation."""
    if external:
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        metadata_text = " ".join(str(value).lower() for value in metadata.values())
        status = str(metadata.get("status", data.get("status", "unknown"))).lower()
        if any(token in f"{status} {metadata_text}" for token in ("provisional", "synthetic", "template", "example", "placeholder")):
            raise ValueError(
                f"Refusing provisional/synthetic/template mesh for external evaluation: {path}"
            )
        if status != "measured_real_room":
            raise ValueError(
                "External mesh must declare metadata.status="
                f"'measured_real_room' (got {status!r}): {path}"
            )
    return load_triangle_mesh(path)


def _lookup_external_3d(labels: dict[str, np.ndarray], row: dict[str, Any]) -> Optional[np.ndarray]:
    candidates = [
        str(row.get("sample_id", "")),
        Path(str(row.get("image_path", ""))).name,
        str(row.get("image_path", "")).replace("\\", "/").lower(),
    ]
    for candidate in candidates:
        if candidate.lower() in labels:
            return labels[candidate.lower()].copy()
    return None


def _candidate_pixels(point: np.ndarray, pattern: str, radius_px: float) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(2)
    radius = max(0.0, float(radius_px))
    if pattern == "single" or radius == 0.0:
        return point.reshape(1, 2)
    if pattern == "cross":
        offsets = np.asarray(
            [[0.0, 0.0], [radius, 0.0], [-radius, 0.0], [0.0, radius], [0.0, -radius]],
            dtype=np.float64,
        )
    elif pattern == "square":
        offsets = np.asarray(
            [
                [0.0, 0.0],
                [radius, 0.0],
                [-radius, 0.0],
                [0.0, radius],
                [0.0, -radius],
                [radius, radius],
                [radius, -radius],
                [-radius, radius],
                [-radius, -radius],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown ray pattern: {pattern}")
    return point.reshape(1, 2) + offsets


def _raycast(
    calibration: CameraCalibration,
    mesh: Any,
    point: Optional[np.ndarray],
    pattern: str,
    radius_px: float,
    max_dist: float,
    step: float,
) -> tuple[Optional[dict[str, Any]], float, dict[str, int], float]:
    if point is None:
        return None, 0.0, {}, 0.0
    started = time.perf_counter()
    pixels = _candidate_pixels(point, pattern, radius_px)
    undistorted = calibration.undistort_pixels(pixels)
    origins, rays = calibration.geometry().pixels_to_rays(undistorted)
    statuses: dict[str, int] = {}
    raw_hits: list[tuple[np.ndarray, float]] = []
    for origin, ray in zip(origins, rays):
        result = intersect_ray_with_grid_result(
            origin,
            ray,
            mesh,
            max_dist=float(max_dist),
            step=float(step),
        )
        statuses[result.status] = statuses.get(result.status, 0) + 1
        if result.hit and result.point is not None:
            raw_hits.append((np.asarray(result.point, dtype=np.float64), 1.0))
    location = localize_pixels(
        calibration.geometry(),
        mesh,
        undistorted,
        weights=np.ones(len(undistorted), dtype=np.float64),
        max_dist=float(max_dist),
        step=float(step),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    hit_rate = float(len(raw_hits) / max(1, len(pixels)))
    if not location.hit or location.point is None:
        return (
            {
                "hit": False,
                "status": location.status,
                "point": None,
                "spread_m": float(location.spread),
                "confidence": float(location.confidence),
                "ray_count": int(len(pixels)),
                "ray_hits": int(len(raw_hits)),
            },
            hit_rate,
            statuses,
            latency_ms,
        )
    return (
        {
            "hit": True,
            "status": location.status,
            "point": np.asarray(location.point, dtype=np.float64),
            "spread_m": float(location.spread),
            "confidence": float(location.confidence),
            "std_m": None if location.std is None else np.asarray(location.std),
            "ray_count": int(len(pixels)),
            "ray_hits": int(len(raw_hits)),
        },
        hit_rate,
        statuses,
        latency_ms,
    )


def _ipm_localize(
    calibration: CameraCalibration,
    homography: FloorHomography,
    point: Optional[np.ndarray],
) -> tuple[Optional[dict[str, Any]], float]:
    """Map one undistorted pixel to the metric floor with IPM.

    IPM is a planar baseline, so it is only physically valid when the fire
    contact point lies on the annotated floor plane.  The caller records that
    assumption separately instead of silently treating a table/cabinet hit as
    a floor point.
    """

    if point is None:
        return None, 0.0
    started = time.perf_counter()
    try:
        ideal = calibration.undistort_pixels(np.asarray(point, dtype=np.float64).reshape(1, 2))
        xyz = homography.pixel_to_floor_xyz(ideal)[0]
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return {
            "hit": False,
            "status": "ipm_projection_failed",
            "point": None,
            "spread_m": None,
            "confidence": 0.0,
            "ray_count": 1,
            "ray_hits": 0,
            "projection_ms": (time.perf_counter() - started) * 1000.0,
        }, 0.0
    if not np.all(np.isfinite(xyz)):
        return {
            "hit": False,
            "status": "ipm_non_finite",
            "point": None,
            "spread_m": None,
            "confidence": 0.0,
            "ray_count": 1,
            "ray_hits": 0,
            "projection_ms": (time.perf_counter() - started) * 1000.0,
        }, 0.0
    return {
        "hit": True,
        "status": "valid_ipm_floor",
        "point": np.asarray(xyz, dtype=np.float64),
        "spread_m": 0.0,
        "confidence": 1.0,
        "std_m": np.zeros(3, dtype=np.float64),
        "ray_count": 1,
        "ray_hits": 1,
        "projection_ms": (time.perf_counter() - started) * 1000.0,
    }, 1.0


def _safe_roi_point(
    coarse: Optional[np.ndarray],
    refined: Optional[np.ndarray],
    confidence: float,
    max_shift_px: float,
    min_heatmap_confidence: float,
    blend_alpha: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
    if coarse is None or refined is None:
        return coarse, coarse, {
            "fallback": True,
            "reason": "missing_coarse_or_refined",
            "shift_px": None,
            "blend_weight": 0.0,
        }
    shift = float(np.linalg.norm(refined - coarse))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if shift > max_shift_px:
        return coarse.copy(), coarse.copy(), {
            "fallback": True,
            "reason": "shift_exceeds_limit",
            "shift_px": shift,
            "blend_weight": 0.0,
        }
    if confidence < min_heatmap_confidence:
        return coarse.copy(), coarse.copy(), {
            "fallback": True,
            "reason": "heatmap_confidence_below_limit",
            "shift_px": shift,
            "blend_weight": 0.0,
        }
    confidence_weight = np.clip(
        (confidence - min_heatmap_confidence) / max(1e-6, 1.0 - min_heatmap_confidence),
        0.0,
        1.0,
    )
    shift_weight = math.exp(-shift / max(max_shift_px, 1e-6))
    weight = float(np.clip(blend_alpha * confidence_weight * shift_weight, 0.0, 1.0))
    blended = coarse + weight * (refined - coarse)
    return refined.copy(), blended, {
        "fallback": False,
        "reason": "accepted",
        "shift_px": shift,
        "blend_weight": weight,
    }


def _branch_metrics(
    rows: list[dict[str, Any]],
    branch: str,
    ipm_eval_surface: str = "floor",
    include_surface_breakdown: bool = True,
) -> dict[str, Any]:
    """Compute metrics while making IPM's planar assumption explicit.

    IPM can numerically project any pixel to Z=0, but that result is a valid
    floor measurement only when the annotated contact is on the floor.  The
    default IPM 3D metrics therefore exclude table/cabinet/column contacts;
    their all-surface diagnostic is retained for transparency.
    """
    pixel_errors: list[float] = []
    xyz_deltas: list[np.ndarray] = []
    all_xyz_deltas: list[np.ndarray] = []
    ray_attempts = ray_hits = 0
    ipm_attempts = ipm_successes = 0
    location_attempts = location_successes = 0
    latencies: dict[str, list[float]] = defaultdict(list)
    fallback = 0
    shifts: list[float] = []
    spreads: list[float] = []
    confidences: list[float] = []
    status_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    for row in rows:
        surface = _surface_name(row)
        surface_counts[surface] = surface_counts.get(surface, 0) + 1
        branch_data = row.get("branches", {}).get(branch, {})
        predicted = _as_point(branch_data.get("pixel"))
        gt_pixel = _as_point(row.get("gt_pixel"))
        if predicted is not None and gt_pixel is not None:
            pixel_errors.append(float(np.linalg.norm(predicted - gt_pixel)))
        location = branch_data.get("location") or {}
        point_3d = _as_xyz(location.get("point"))
        gt_xyz = _as_xyz(row.get("gt_xyz"))
        if point_3d is not None and gt_xyz is not None:
            delta = point_3d - gt_xyz
            all_xyz_deltas.append(delta)
            if (
                branch != "ipm"
                or ipm_eval_surface == "all"
                or surface == ipm_eval_surface
            ):
                xyz_deltas.append(delta)
        ray_count = int(location.get("ray_count", 0))
        ray_hit_count = int(location.get("ray_hits", 0))
        if branch == "ipm":
            ipm_attempts += 1
            ipm_successes += int(bool(location.get("hit")))
        else:
            ray_attempts += ray_count
            ray_hits += ray_hit_count
        if location:
            location_attempts += 1
            location_successes += int(bool(location.get("hit")))
        for key in ("roi_ms", "ray_ms", "total_ms"):
            if branch_data.get(key) is not None:
                latencies[key].append(float(branch_data[key]))
        if branch_data.get("fallback"):
            fallback += 1
        if branch_data.get("shift_px") is not None:
            shifts.append(float(branch_data["shift_px"]))
        if location.get("spread_m") is not None:
            spreads.append(float(location["spread_m"]))
        if location.get("confidence") is not None:
            confidences.append(float(location["confidence"]))
        status = str(location.get("status", "no_result"))
        status_counts[status] = status_counts.get(status, 0) + 1

    pixel = np.asarray(pixel_errors, dtype=np.float64)
    deltas = np.asarray(xyz_deltas, dtype=np.float64).reshape(-1, 3) if xyz_deltas else np.empty((0, 3))
    all_deltas = (
        np.asarray(all_xyz_deltas, dtype=np.float64).reshape(-1, 3)
        if all_xyz_deltas
        else np.empty((0, 3))
    )
    norms = np.linalg.norm(deltas, axis=1) if len(deltas) else np.empty(0)
    all_norms = np.linalg.norm(all_deltas, axis=1) if len(all_deltas) else np.empty(0)

    def percentile(values: np.ndarray, q: float) -> Optional[float]:
        return None if not len(values) else float(np.percentile(values, q))

    def mean(values: np.ndarray) -> Optional[float]:
        return None if not len(values) else float(values.mean())

    result: dict[str, Any] = {
        "branch": branch,
        "samples": len(rows),
        "pixel_samples": int(len(pixel)),
        "pixel_mae_px": mean(pixel),
        "pixel_median_px": percentile(pixel, 50),
        "pixel_p95_px": percentile(pixel, 95),
        "pck10": None if not len(pixel) else float(np.mean(pixel <= 10.0)),
        "pck25": None if not len(pixel) else float(np.mean(pixel <= 25.0)),
        "ray_attempts": ray_attempts,
        "ray_hits": ray_hits,
        "ray_hit_rate": float(ray_hits / ray_attempts) if ray_attempts else None,
        "ipm_projection_attempts": ipm_attempts,
        "ipm_projection_successes": ipm_successes,
        "ipm_projection_success_rate": (
            float(ipm_successes / ipm_attempts) if ipm_attempts else None
        ),
        "location_attempts": location_attempts,
        "location_successes": location_successes,
        "location_success_rate": float(location_successes / location_attempts)
        if location_attempts
        else 0.0,
        "surface_counts": surface_counts,
        "ipm_eval_surface": ipm_eval_surface if branch == "ipm" else None,
        "geometry_eligible_samples": int(len(norms)),
        "geometry_excluded_samples": int(len(all_norms) - len(norms))
        if branch == "ipm"
        else 0,
        "three_d_samples": int(len(norms)),
        "three_d_mae_m": mean(norms),
        "three_d_median_m": percentile(norms, 50),
        "three_d_p95_m": percentile(norms, 95),
        "three_d_all_surfaces_samples": int(len(all_norms)),
        "three_d_all_surfaces_mae_m": mean(all_norms),
        "three_d_all_surfaces_median_m": percentile(all_norms, 50),
        "three_d_all_surfaces_p95_m": percentile(all_norms, 95),
        "under_0.10m": None if not len(norms) else float(np.mean(norms <= 0.10)),
        "under_0.25m": None if not len(norms) else float(np.mean(norms <= 0.25)),
        "under_0.50m": None if not len(norms) else float(np.mean(norms <= 0.50)),
        "under_1.00m": None if not len(norms) else float(np.mean(norms <= 1.00)),
        "xyz_mae_m": None if not len(deltas) else np.mean(np.abs(deltas), axis=0),
        "xyz_p95_abs_m": None if not len(deltas) else np.percentile(np.abs(deltas), 95, axis=0),
        "fallback_count": fallback,
        "fallback_rate": float(fallback / max(1, len(rows))),
        "mean_shift_px": mean(np.asarray(shifts)),
        "p95_shift_px": percentile(np.asarray(shifts), 95),
        "mean_spread_m": mean(np.asarray(spreads)),
        "mean_location_confidence": mean(np.asarray(confidences)),
        "status_counts": status_counts,
        "latency_ms": {
            key: {
                "mean": mean(np.asarray(values)),
                "median": percentile(np.asarray(values), 50),
                "p95": percentile(np.asarray(values), 95),
            }
            for key, values in latencies.items()
        },
    }
    if include_surface_breakdown:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_surface_name(row)].append(row)
        result["by_surface"] = {
            surface: _branch_metrics(
                surface_rows,
                branch,
                ipm_eval_surface=ipm_eval_surface,
                include_surface_breakdown=False,
            )
            for surface, surface_rows in sorted(grouped.items())
        }
    return result


def _trajectory_metrics(
    rows: list[dict[str, Any]],
    branch: str,
    temporal_filter: str = "ema",
    ipm_eval_surface: str = "floor",
) -> dict[str, Any]:
    if temporal_filter not in {"ema", "ekf"}:
        raise ValueError(f"Unknown temporal filter: {temporal_filter}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scene_id", row.get("sequence_id", "unknown")))].append(row)
    pixel_jitter: list[float] = []
    xyz_jitter: list[float] = []
    jumps: list[float] = []
    valid_frames = total_frames = evaluated_frames = 0
    surface_excluded_frames = 0
    tracked_jitter: list[float] = []
    for sequence_rows in grouped.values():
        sequence_rows.sort(key=lambda item: int(item.get("frame_index", 0)))
        tracker = (
            Fire3DTracker(alpha=0.35, gate_m=3.0, max_missed=3)
            if temporal_filter == "ema"
            else Fire3DEKF(
                dt=1.0,
                process_accel_std=0.35,
                measurement_std_m=0.25,
                gate_mahalanobis2=16.27,
                max_missed=3,
            )
        )
        previous_pixel: Optional[np.ndarray] = None
        previous_xyz: Optional[np.ndarray] = None
        previous_tracked: Optional[np.ndarray] = None
        for row in sequence_rows:
            total_frames += 1
            if (
                branch == "ipm"
                and ipm_eval_surface != "all"
                and _surface_name(row) != ipm_eval_surface
            ):
                surface_excluded_frames += 1
                previous_pixel = None
                previous_xyz = None
                previous_tracked = None
                continue
            evaluated_frames += 1
            data = row.get("branches", {}).get(branch, {})
            pixel = _as_point(data.get("pixel"))
            xyz = _as_xyz((data.get("location") or {}).get("point"))
            if pixel is not None:
                if previous_pixel is not None:
                    pixel_jitter.append(float(np.linalg.norm(pixel - previous_pixel)))
                previous_pixel = pixel
            if xyz is not None:
                valid_frames += 1
                if previous_xyz is not None:
                    distance = float(np.linalg.norm(xyz - previous_xyz))
                    xyz_jitter.append(distance)
                    jumps.append(float(distance > 0.50))
                previous_xyz = xyz
                location = data.get("location") or {}
                measurement_std = location.get("std_m")
                if temporal_filter == "ekf":
                    state = tracker.update(
                        xyz,
                        covariance=measurement_std,
                        confidence=float(location.get("confidence", 0.0)),
                    )
                    tracked_point = state.position
                else:
                    state = tracker.update(
                        xyz,
                        confidence=float(location.get("confidence", 0.0)),
                    )
                    tracked_point = state.point
                if tracked_point is not None and state.accepted:
                    if previous_tracked is not None:
                        tracked_jitter.append(float(np.linalg.norm(tracked_point - previous_tracked)))
                    previous_tracked = tracked_point.copy()
            else:
                tracker.update(None)
                previous_xyz = None
                previous_tracked = None

    def avg(values: list[float]) -> Optional[float]:
        return None if not values else float(np.mean(values))

    def median(values: list[float]) -> Optional[float]:
        return None if not values else float(np.median(values))

    def p95(values: list[float]) -> Optional[float]:
        return None if not values else float(np.percentile(values, 95))

    return {
        "branch": branch,
        "filter": temporal_filter,
        "sequences": len(grouped),
        "frames": total_frames,
        "evaluated_frames": evaluated_frames,
        "surface_excluded_frames": surface_excluded_frames,
        "valid_frame_rate": float(valid_frames / max(1, evaluated_frames)),
        "filter_scope": ipm_eval_surface if branch == "ipm" else "all_surfaces",
        "pixel_jitter_median_px": median(pixel_jitter),
        "pixel_jitter_p95_px": p95(pixel_jitter),
        "three_d_jitter_median_m": median(xyz_jitter),
        "three_d_jitter_p95_m": p95(xyz_jitter),
        "three_d_jump_rate_over_0.5m": avg(jumps),
        "tracked_3d_jitter_median_m": avg(tracked_jitter),
        "tracked_3d_jitter_p95_m": p95(tracked_jitter),
    }


def _draw_marker(draw: ImageDraw.ImageDraw, point: Optional[np.ndarray], color: tuple[int, int, int], scale: float) -> None:
    if point is None:
        return
    x, y = float(point[0] * scale), float(point[1] * scale)
    radius = max(3.0, 5.0 * scale)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=max(1, int(2 * scale)))
    draw.line((x - radius * 1.5, y, x + radius * 1.5, y), fill=color, width=max(1, int(scale)))
    draw.line((x, y - radius * 1.5, x, y + radius * 1.5), fill=color, width=max(1, int(scale)))


def _write_contact_sheet(rows: list[dict[str, Any]], output: Path, maximum: int = 16) -> None:
    rows = rows[:maximum]
    if not rows:
        return
    thumb_w = 320
    margin = 6
    columns = min(4, len(rows))
    thumb_h = int(round(thumb_w * rows[0]["image_size"][1] / rows[0]["image_size"][0]))
    label_h = 42
    sheet = Image.new("RGB", (columns * (thumb_w + margin) + margin, math.ceil(len(rows) / columns) * (thumb_h + label_h + margin) + margin), (22, 22, 22))
    draw_sheet = ImageDraw.Draw(sheet)
    colors = {
        "gt_pixel": (40, 220, 80),
        "coarse": (60, 130, 255),
        "roi_raw": (255, 165, 30),
        "roi_blend": (235, 50, 50),
    }
    for index, row in enumerate(rows):
        path = Path(row["image_path"])
        image = Image.open(path).convert("RGB")
        scale = thumb_w / max(1, image.width)
        image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = margin + (index % columns) * (thumb_w + margin)
        y = margin + (index // columns) * (thumb_h + label_h + margin)
        image_draw = ImageDraw.Draw(image)
        _draw_marker(image_draw, _as_point(row.get("gt_pixel")), colors["gt_pixel"], scale)
        for branch in ("coarse", "roi_raw", "roi_blend"):
            _draw_marker(image_draw, _as_point((row.get("branches", {}).get(branch) or {}).get("pixel")), colors[branch], scale)
        sheet.paste(image, (x, y))
        draw_sheet.text(
            (x + 3, y + thumb_h + 3),
            f"{Path(row['image_path']).name}  coarse=blue ROI=orange blend=red GT=green",
            fill=(240, 240, 240),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{float(value):.4f}"
    return str(value)


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nPaper-inspired 2D -> 3D benchmark")
    print("branch          pixel_MAE  PCK@10  mesh_hit  IPM_ok  3D_MAE(m)  median(m)  P95(m)  lat_P95(ms)")
    available = summary.get("branches", {})
    for branch in BRANCHES:
        if branch not in available:
            continue
        item = available[branch]
        latency = item.get("latency_ms", {}).get("total_ms", {}).get("p95")
        print(
            f"{branch:<15} "
            f"{_format_metric(item.get('pixel_mae_px')):>9} "
            f"{_format_metric(item.get('pck10')):>7} "
            f"{_format_metric(item.get('ray_hit_rate')):>8} "
            f"{_format_metric(item.get('ipm_projection_success_rate')):>7} "
            f"{_format_metric(item.get('three_d_mae_m')):>10} "
            f"{_format_metric(item.get('three_d_median_m')):>10} "
            f"{_format_metric(item.get('three_d_p95_m')):>8} "
            f"{_format_metric(latency):>13}"
        )
    print("\n3D metrics are valid for this run only as synthetic metric-geometry diagnostics.")
    print("A physical-room claim requires measured calibration + measured mesh + marker XYZ labels.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    rows = _load_manifest(dataset, args.split)
    rows = _select_records(rows, args.max_records, args.selection)
    mesh_path = args.mesh.expanduser().resolve() if args.mesh else dataset / "room_mesh.json"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = _load_mesh(mesh_path, external=args.calibration_source == "external")

    external_calibration = None
    if args.calibration_source == "external":
        if args.calibration is None:
            raise ValueError("--calibration is required with --calibration-source external")
        external_calibration = _load_external_calibration(args.calibration.expanduser().resolve())
    external_3d = _load_external_3d(args.labels_3d.expanduser().resolve() if args.labels_3d else None)

    detector = None
    if args.coarse_source == "detector":
        if args.detector_checkpoint is None:
            raise ValueError("--detector-checkpoint is required with --coarse-source detector")
        from fire_detector import FireDetector

        detector = FireDetector(args.detector_checkpoint, device=args.device, threshold=args.threshold)
        detector.warmup(repeats=1)

    refiner = None
    if args.roi_checkpoint is not None:
        from narrow_localizer import ROIRefinerInference

        refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device)

    ipm_homography = None
    if args.enable_ipm:
        if args.calibration_source == "external":
            ipm_homography = FloorHomography.from_calibration(external_calibration)
        else:
            # The synthetic manifest may use a per-frame camera pose.  The
            # homography is therefore built inside the frame loop below.
            ipm_homography = True

    processed: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["_image_path"])
        image = Image.open(image_path).convert("RGB")
        image_size = [image.width, image.height]
        if args.calibration_source == "external":
            calibration = external_calibration
        else:
            calibration = _calibration_from_row(row, args.calibration_source)
        frame_homography = (
            FloorHomography.from_calibration(calibration)
            if args.enable_ipm and ipm_homography is True
            else ipm_homography
        )
        gt_pixel = _row_pixel(row)
        gt_xyz = _as_xyz(row.get("fire_xyz_world"))
        if gt_xyz is None:
            gt_xyz = _lookup_external_3d(external_3d, row)
        row_result: dict[str, Any] = {
            "sample_id": row.get("sample_id", image_path.stem),
            "scene_id": row.get("scene_id", row.get("sequence_id", image_path.stem)),
            "frame_index": int(row.get("frame_index", index - 1)),
            "image_path": str(image_path),
            "image_size": image_size,
            "gt_pixel": gt_pixel,
            "gt_xyz": gt_xyz,
            "fire_surface": row.get("fire_surface", row.get("surface_type", "unknown")),
            "branches": {},
        }
        coarse: Optional[np.ndarray] = None
        detector_ms = 0.0
        if args.coarse_source == "manifest":
            coarse = _row_pixel(row, "p_fire_noisy_pixel")
        elif args.coarse_source == "gt_noise":
            coarse = gt_pixel.copy() if gt_pixel is not None else None
            if coarse is not None:
                rng = np.random.default_rng(args.seed + index * 7919)
                coarse += rng.normal(0.0, args.gt_noise_px, 2)
        else:
            detector_start = time.perf_counter()
            result = detector.detect(image, warmup=False)
            detector_ms = (time.perf_counter() - detector_start) * 1000.0
            coarse = None if result.pixel is None else np.asarray(result.pixel, dtype=np.float64)
            row_result["detector"] = {
                "confidence": float(result.confidence),
                "detected": bool(result.detected),
                "pixel": coarse,
                "latency_ms": detector_ms,
            }

        refined: Optional[np.ndarray] = None
        roi_confidence: Optional[float] = None
        roi_ms: Optional[float] = None
        if refiner is not None and coarse is not None:
            roi_start = time.perf_counter()
            refined_result = refiner.refine(image, coarse)
            roi_ms = (time.perf_counter() - roi_start) * 1000.0
            refined = np.asarray(refined_result.point, dtype=np.float64)
            roi_confidence = float(refined_result.confidence)

        safe, blended, reliability = _safe_roi_point(
            coarse,
            refined,
            roi_confidence if roi_confidence is not None else 0.0,
            args.max_shift_px,
            args.min_heatmap_confidence,
            args.blend_alpha,
        )
        points = {
            "coarse": coarse,
            "roi_raw": refined,
            "roi_safe": safe,
            "roi_blend": blended,
        }
        all_points = dict(points)
        if args.enable_ipm:
            all_points["ipm"] = blended
        for branch, point in all_points.items():
            if branch == "ipm":
                ipm_data, ipm_hit_rate = _ipm_localize(calibration, frame_homography, point)
                branch_data = {
                    "pixel": point,
                    "roi_confidence": roi_confidence,
                    "shift_px": reliability.get("shift_px"),
                    "blend_weight": reliability.get("blend_weight"),
                    "fallback": False,
                    "fallback_reason": None,
                    "location": ipm_data,
                    "ray_hit_rate": ipm_hit_rate,
                    "ray_statuses": {} if ipm_data is None else {str(ipm_data.get("status")): 1},
                    "roi_ms": roi_ms or 0.0,
                    "ray_ms": 0.0,
                    "total_ms": detector_ms + (roi_ms or 0.0) + (0.0 if ipm_data is None else float(ipm_data.get("projection_ms", 0.0))),
                }
                row_result["branches"][branch] = branch_data
                continue
            ray_data, ray_hit_rate, statuses, ray_ms = _raycast(
                calibration,
                mesh,
                point,
                args.ray_pattern,
                args.ray_radius_px,
                args.max_dist,
                args.step,
            )
            branch_data: dict[str, Any] = {
                "pixel": point,
                "roi_confidence": roi_confidence,
                "shift_px": reliability.get("shift_px") if branch != "coarse" else None,
                "blend_weight": reliability.get("blend_weight") if branch == "roi_blend" else None,
                "fallback": bool(reliability.get("fallback")) if branch == "roi_safe" else False,
                "fallback_reason": reliability.get("reason") if branch == "roi_safe" else None,
                "location": ray_data,
                "ray_hit_rate": ray_hit_rate,
                "ray_statuses": statuses,
                "roi_ms": roi_ms if branch != "coarse" else 0.0,
                "ray_ms": ray_ms,
                "total_ms": detector_ms + (roi_ms or 0.0) + ray_ms,
            }
            row_result["branches"][branch] = branch_data
        processed.append(row_result)
        if index == 1 or index == len(rows) or index % 10 == 0:
            print(f"processed={index}/{len(rows)} image={image_path.name}", flush=True)

    summary = {
        "format": "LAB_SAM.paper_workflow_3d.v2",
        "method": {
            "coarse_source": args.coarse_source,
            "roi_checkpoint": None if args.roi_checkpoint is None else str(args.roi_checkpoint),
            "detector_checkpoint": None if args.detector_checkpoint is None else str(args.detector_checkpoint),
            "calibration_source": args.calibration_source,
            "mesh": str(mesh_path),
            "ray_pattern": args.ray_pattern,
            "ray_radius_px": args.ray_radius_px,
            "max_shift_px": args.max_shift_px,
            "min_heatmap_confidence": args.min_heatmap_confidence,
            "blend_alpha": args.blend_alpha,
            "ipm_enabled": bool(args.enable_ipm),
            "ipm_plane": "world Z=0 (floor only)",
            "temporal_filter": args.temporal_filter,
        },
        "dataset": {
            "root": str(dataset),
            "split": args.split,
            "records_selected": len(processed),
            "selection": args.selection,
            "synthetic_metric_geometry": args.calibration_source != "external",
            "warning": "Synthetic metric results are not physical-room accuracy.",
        },
        "branches": {
            branch: _branch_metrics(
                processed,
                branch,
                ipm_eval_surface=args.ipm_eval_surface,
            )
            for branch in BRANCHES
            if branch in (processed[0]["branches"] if processed else {})
        },
        "sequence_stability": {
            branch: _trajectory_metrics(
                processed,
                branch,
                args.temporal_filter,
                ipm_eval_surface=args.ipm_eval_surface,
            )
            for branch in BRANCHES
            if branch in (processed[0]["branches"] if processed else {})
        },
        "elapsed_seconds": time.perf_counter() - started_all,
        "records": processed,
    }
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-records", type=int, default=24)
    parser.add_argument("--selection", choices=("even", "diverse"), default="even")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "paper_workflow_synthetic")
    parser.add_argument("--coarse-source", choices=("manifest", "gt_noise", "detector"), default="manifest")
    parser.add_argument("--gt-noise-px", type=float, default=3.0)
    parser.add_argument("--detector-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--roi-checkpoint",
        type=Path,
        default=root / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth",
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Disable the optional ROI refiner; useful for geometry-only smoke tests",
    )
    parser.add_argument("--calibration-source", choices=("true", "estimated", "external"), default="true")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--labels-3d", type=Path, default=None)
    parser.add_argument("--ray-pattern", choices=("single", "cross", "square"), default="cross")
    parser.add_argument("--ray-radius-px", type=float, default=2.0)
    parser.add_argument("--max-dist", type=float, default=100.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--max-shift-px", type=float, default=40.0)
    parser.add_argument("--min-heatmap-confidence", type=float, default=0.05)
    parser.add_argument("--blend-alpha", type=float, default=0.75)
    parser.add_argument(
        "--enable-ipm",
        action="store_true",
        help="Add an independent planar-floor IPM branch alongside mesh ray casting",
    )
    parser.add_argument(
        "--ipm-eval-surface",
        choices=("floor", "all"),
        default="floor",
        help=(
            "Surface subset used for IPM 3D error metrics. 'floor' is the "
            "physically valid default; 'all' is diagnostic only."
        ),
    )
    parser.add_argument(
        "--temporal-filter",
        choices=("ema", "ekf"),
        default="ema",
        help="Temporal filter used for sequence stability metrics",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.roi_checkpoint = args.roi_checkpoint.expanduser().resolve() if args.roi_checkpoint else None
    if args.no_roi:
        args.roi_checkpoint = None
    args.detector_checkpoint = args.detector_checkpoint.expanduser().resolve() if args.detector_checkpoint else None
    args.mesh = args.mesh.expanduser().resolve() if args.mesh else None
    args.calibration = args.calibration.expanduser().resolve() if args.calibration else None
    args.labels_3d = args.labels_3d.expanduser().resolve() if args.labels_3d else None
    if args.max_records < 0:
        raise ValueError("--max-records must be >= 0")
    if args.roi_checkpoint is not None and not args.roi_checkpoint.is_file():
        raise FileNotFoundError(f"ROI checkpoint not found: {args.roi_checkpoint}")
    summary = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "sequence_metrics.json", summary["sequence_stability"])
    _write_contact_sheet(summary["records"], args.output_dir / "comparison_contact_sheet.png")
    _print_summary(summary)
    print(f"saved_summary={args.output_dir / 'summary.json'}")
    print(f"saved_contact_sheet={args.output_dir / 'comparison_contact_sheet.png'}")


if __name__ == "__main__":
    main()
