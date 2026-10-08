"""Priority-1 benchmark for the protected Ray Casting post-ROI workflow.

This is an additive benchmark.  It does not modify checkpoints, source
datasets, or older output directories.  It evaluates the following items on
the same scene-separated synthetic test split:

* 1/3/5/7 bottom-band rays;
* confidence-dependent pixel uncertainty (the synthetic detector confidence
  is used as a proxy because no real ROI heatmap is stored in this dataset);
* Ray-Casting-first routing with the existing mixed Scene-MLP as fallback;
* EMA and EKF temporal filtering;
* controlled mesh vertex noise.

The intended real workflow is:

    detector -> ROI bottom-band candidates -> undistortion -> multi-ray mesh
    intersection -> robust 3D aggregation -> EMA/EKF -> protected fallback.

The synthetic manifest contains metric geometry, but it is not real CCTV.  In
particular, ``observation_confidence`` is a generated confidence proxy, not a
confidence emitted by the real ROI checkpoint.

Example::

    .venv\\Scripts\\python.exe benchmark_priority1_raycasting.py ^
      --dataset working\\synthetic_fire_3d_v3 ^
      --output-dir output\\priority1_raycasting_20261008
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from camera_calibration import CameraCalibration
from ekf_tracker import Fire3DEKF
from locator import TriangleMesh, intersect_ray_with_grid_result
from mesh_loader import load_triangle_mesh
from physics_scene_coordinate import PhysicsSceneCoordinateMLP, build_samples
from tracking_3d import Fire3DTracker


CONDITIONS = ("clean_true", "noisy_true", "clean_estimated", "noisy_estimated")
PIXEL_KEYS = {
    "clean_true": ("p_fire_pixel", "camera"),
    "noisy_true": ("p_fire_noisy_pixel", "camera"),
    "clean_estimated": ("p_fire_pixel", "camera_estimated"),
    "noisy_estimated": ("p_fire_noisy_pixel", "camera_estimated"),
}
RAY_COUNTS = (1, 3, 5, 7)
MESH_NOISE_M = (0.0, 0.005, 0.02, 0.05, 0.10)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _finite_point(value: Any, dimensions: int) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(array) < dimensions or not np.all(np.isfinite(array[:dimensions])):
        return None
    return array[:dimensions].copy()


def _load_rows(dataset: Path, split: str) -> list[dict[str, Any]]:
    path = dataset / f"{split}.jsonl"
    if not path.is_file():
        path = dataset / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {split}.jsonl or manifest.jsonl under {dataset}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if split == "all" or str(row.get("split", split)) == split:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows for split={split!r} in {path}")
    return rows


def _camera(row: dict[str, Any], source: str) -> Optional[CameraCalibration]:
    value = row.get(source)
    if not isinstance(value, dict):
        return None
    try:
        return CameraCalibration.from_dict(
            {
                "image_size": row.get("image_size", [640, 640]),
                "intrinsics": {
                    "K": value.get("K"),
                    "dist_coeffs": value.get("dist_coeffs", []),
                },
                "extrinsics": {
                    "R": value.get("R_world_to_camera", value.get("R")),
                    "camera_position": value.get("camera_position"),
                },
            }
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _visible_rows(rows: Iterable[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    pixel_key, camera_key = PIXEL_KEYS[condition]
    result: list[dict[str, Any]] = []
    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            continue
        pixel = _finite_point(row.get(pixel_key), 2)
        target = _finite_point(row.get("fire_xyz_world"), 3)
        if pixel is None or target is None:
            continue
        size = _finite_point(row.get("image_size", [640, 640]), 2)
        calibration = _camera(row, camera_key)
        if size is None or calibration is None:
            continue
        if np.any(pixel < -0.5) or np.any(pixel > size + 0.5):
            continue
        result.append(row)
    return result


def _stable_seed(text: str, base: int = 37) -> int:
    value = int(base)
    for index, character in enumerate(str(text)):
        value = (value * 131 + (index + 1) * ord(character)) % 2_147_483_647
    return value


def _roi_confidence(row: dict[str, Any], condition: str) -> float:
    """Return the synthetic confidence proxy used by this benchmark."""

    if condition.startswith("clean_"):
        return 1.0
    noise = row.get("noise", {})
    try:
        value = float(noise.get("observation_confidence", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return float(np.clip(value, 0.0, 1.0))


def _pixel_sigma_px(row: dict[str, Any], condition: str, base_sigma_px: float) -> float:
    """Map ROI confidence to an input-pixel uncertainty.

    The mapping is intentionally conservative: a high-confidence ROI is given
    0.5x the base sigma and a low-confidence ROI 2.0x the base sigma.  This is
    a controllable proxy until a real ROI heatmap/entropy is available.
    """

    confidence = _roi_confidence(row, condition)
    multiplier = 0.5 + 1.5 * (1.0 - confidence)
    return float(max(0.05, base_sigma_px * multiplier))


def _candidate_pattern(count: int) -> np.ndarray:
    patterns = {
        1: [[0.0, 0.0]],
        3: [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
        5: [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-0.5, 0.7], [0.5, 0.7]],
        7: [
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [-0.5, 0.7],
            [0.5, 0.7],
            [-0.5, -0.7],
            [0.5, -0.7],
        ],
    }
    if count not in patterns:
        raise ValueError(f"Unsupported ray count: {count}")
    return np.asarray(patterns[count], dtype=np.float64)


def _candidate_pixels(
    row: dict[str, Any],
    condition: str,
    count: int,
    base_sigma_px: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Generate a deterministic bottom-band proxy around the observed point."""

    pixel_key, _ = PIXEL_KEYS[condition]
    base = _finite_point(row.get(pixel_key), 2)
    if base is None:
        return np.empty((0, 2)), np.empty((0,)), 0.0, {"status": "missing_pixel"}
    image_size = _finite_point(row.get("image_size", [640, 640]), 2)
    bbox = _finite_point(row.get("bbox_xyxy"), 4)
    if image_size is None:
        return np.empty((0, 2)), np.empty((0,)), 0.0, {"status": "missing_image_size"}

    sigma = _pixel_sigma_px(row, condition, base_sigma_px)
    pattern = _candidate_pattern(count)
    if bbox is not None:
        box_width = max(1.0, float(abs(bbox[2] - bbox[0])))
        horizontal_span = min(0.22 * box_width, max(1.0, 1.5 * sigma))
    else:
        horizontal_span = max(1.0, 1.5 * sigma)

    # A fixed per-record jitter makes the benchmark deterministic while still
    # representing imperfect bottom-contour/heatmap samples.
    rng = np.random.default_rng(int(seed))
    jitter = rng.normal(0.0, sigma * 0.18, size=(count, 2))
    jitter[0] = 0.0
    offsets = pattern.copy()
    offsets[:, 0] *= horizontal_span
    offsets[:, 1] *= 0.55 * sigma
    candidates = base[None, :] + offsets + jitter
    width, height = image_size
    candidates[:, 0] = np.clip(candidates[:, 0], 0.0, width - 1.0)
    candidates[:, 1] = np.clip(candidates[:, 1], 0.0, height - 1.0)

    confidence = _roi_confidence(row, condition)
    # The center candidate is the most reliable; off-center candidates are
    # down-weighted according to the same confidence-aware uncertainty.
    distances = np.linalg.norm(candidates - base[None, :], axis=1)
    weights = np.exp(-0.5 * (distances / max(sigma, 0.05)) ** 2)
    weights *= max(0.05, confidence)
    weights = weights / max(float(weights.sum()), 1e-12)
    metadata = {
        "status": "ok",
        "pixel_key": pixel_key,
        "confidence_proxy": confidence,
        "pixel_sigma_px": sigma,
        "base_pixel": base,
        "pattern": pattern,
        "heatmap_proxy": "centered_bottom_band_with_confidence_scaled_jitter",
    }
    return candidates, weights, confidence, metadata


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    weights = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-9)
    output: list[float] = []
    for axis in range(3):
        order = np.argsort(values[:, axis])
        cumulative = np.cumsum(weights[order])
        output.append(float(values[order[np.searchsorted(cumulative, cumulative[-1] * 0.5)], axis]))
    return np.asarray(output, dtype=np.float64)


def _aggregate_hits(
    hits: list[np.ndarray],
    weights: np.ndarray,
    pixel_sigma_px: float,
    ray_distances: list[float],
    focal_px: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    if not hits:
        return None, None, 0.0, float("inf")
    points = np.asarray(hits, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(points) >= 3:
        median = np.median(points, axis=0)
        mad = np.median(np.abs(points - median), axis=0)
        scale = 1.4826 * np.maximum(mad, 1e-5)
        keep = np.all(np.abs(points - median) / scale <= 3.5, axis=1)
        if not np.any(keep):
            keep[:] = True
        points = points[keep]
        weights = weights[keep]
    center = _weighted_median(points, weights)
    centered = points - center
    if len(points) >= 2:
        covariance = np.cov(centered.T, aweights=weights, ddof=0)
        covariance = np.atleast_2d(covariance)
        std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        depth = max(float(np.median(ray_distances)), 1.0)
        sigma_m = depth * float(pixel_sigma_px) / max(float(focal_px), 1.0)
        std = np.asarray([sigma_m, sigma_m, max(0.5 * sigma_m, 0.01)], dtype=np.float64)
    spread = float(np.median(np.linalg.norm(centered, axis=1))) if len(points) else float("inf")
    return center, np.maximum(std, 1e-4), spread, float(np.linalg.norm(std))


def _ray_localize(
    row: dict[str, Any],
    condition: str,
    mesh: TriangleMesh,
    ray_count: int,
    base_sigma_px: float,
    seed: int,
    max_dist: float,
) -> dict[str, Any]:
    _, camera_key = PIXEL_KEYS[condition]
    calibration = _camera(row, camera_key)
    target = _finite_point(row.get("fire_xyz_world"), 3)
    if calibration is None or target is None:
        return {
            "point": None,
            "std_m": None,
            "target": target,
            "hit_count": 0,
            "ray_count": ray_count,
            "hit_rate": 0.0,
            "spread_m": None,
            "uncertainty_m": None,
            "confidence": _roi_confidence(row, condition),
            "latency_ms": None,
            "candidates": [],
            "status": "invalid_camera_or_target",
        }

    pixels, weights, confidence, candidate_meta = _candidate_pixels(
        row,
        condition,
        ray_count,
        base_sigma_px,
        seed,
    )
    started = time.perf_counter()
    geometry = calibration.geometry()
    hits: list[np.ndarray] = []
    hit_weights: list[float] = []
    distances: list[float] = []
    statuses: Counter[str] = Counter()
    for pixel, weight in zip(pixels, weights):
        origin, direction = geometry.pixel_to_ray(float(pixel[0]), float(pixel[1]))
        result = intersect_ray_with_grid_result(origin, direction, mesh, max_dist=max_dist)
        statuses[result.status] += 1
        if result.hit and result.point is not None and np.all(np.isfinite(result.point)):
            hits.append(np.asarray(result.point, dtype=np.float64))
            hit_weights.append(float(weight))
            distances.append(float(result.distance))
    focal = float(np.mean([calibration.K[0, 0], calibration.K[1, 1]]))
    point, std, spread, uncertainty = _aggregate_hits(
        hits,
        np.asarray(hit_weights, dtype=np.float64),
        float(candidate_meta.get("pixel_sigma_px", base_sigma_px)),
        distances,
        focal,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    hit_rate = float(len(hits) / max(1, len(pixels)))
    # The quality score is deliberately conservative and is used only by the
    # protected router, never as a claimed probability of correctness.
    quality = float(np.clip(confidence * hit_rate * np.exp(-spread / 0.25), 0.0, 1.0)) if point is not None else 0.0
    return {
        "point": point,
        "std_m": std,
        "target": target,
        "hit_count": len(hits),
        "ray_count": len(pixels),
        "hit_rate": hit_rate,
        "spread_m": None if not np.isfinite(spread) else float(spread),
        "uncertainty_m": None if not np.isfinite(uncertainty) else float(uncertainty),
        "confidence": confidence,
        "quality": quality,
        "latency_ms": elapsed,
        "candidates": pixels,
        "candidate_weights": weights,
        "candidate_meta": candidate_meta,
        "status_counts": dict(statuses),
        "status": "valid_multi_ray" if point is not None else "no_valid_intersection",
    }


def _metric_summary(targets: Sequence[np.ndarray], predictions: Sequence[Optional[np.ndarray]]) -> dict[str, Any]:
    pairs = [
        (np.asarray(target, dtype=np.float64).reshape(3), np.asarray(prediction, dtype=np.float64).reshape(3))
        for target, prediction in zip(targets, predictions)
        if target is not None and prediction is not None and np.all(np.isfinite(prediction))
    ]
    attempted = sum(1 for target in targets if target is not None)
    if not pairs:
        return {
            "count": 0,
            "attempted": attempted,
            "valid_rate": 0.0,
            "mae_m": None,
            "median_m": None,
            "p95_m": None,
            "xyz_mae_m": None,
            "xyz_p95_abs_m": None,
            "under_0.10m": None,
            "under_0.25m": None,
            "under_0.50m": None,
            "under_1.00m": None,
        }
    truth = np.asarray([item[0] for item in pairs])
    predicted = np.asarray([item[1] for item in pairs])
    delta = predicted - truth
    errors = np.linalg.norm(delta, axis=1)
    return {
        "count": int(len(errors)),
        "attempted": int(attempted),
        "valid_rate": float(len(errors) / max(1, attempted)),
        "mae_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "xyz_mae_m": np.mean(np.abs(delta), axis=0).tolist(),
        "xyz_p95_abs_m": np.percentile(np.abs(delta), 95, axis=0).tolist(),
        "under_0.10m": float(np.mean(errors <= 0.10)),
        "under_0.25m": float(np.mean(errors <= 0.25)),
        "under_0.50m": float(np.mean(errors <= 0.50)),
        "under_1.00m": float(np.mean(errors <= 1.00)),
    }


def _stability(records: Sequence[dict[str, Any]], point_key: str, jump_threshold_m: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get(point_key) is not None:
            grouped[str(record["scene_id"])].append(record)
    steps: list[float] = []
    for values in grouped.values():
        values.sort(key=lambda item: int(item.get("frame_index", 0)))
        for previous, current in zip(values, values[1:]):
            steps.append(float(np.linalg.norm(np.asarray(current[point_key]) - np.asarray(previous[point_key]))))
    if not steps:
        return {
            "step_count": 0,
            "mean_step_m": None,
            "median_step_m": None,
            "p95_step_m": None,
            "jump_threshold_m": float(jump_threshold_m),
            "jump_rate": None,
        }
    values = np.asarray(steps, dtype=np.float64)
    return {
        "step_count": int(len(values)),
        "mean_step_m": float(np.mean(values)),
        "median_step_m": float(np.median(values)),
        "p95_step_m": float(np.percentile(values, 95)),
        "jump_threshold_m": float(jump_threshold_m),
        "jump_rate": float(np.mean(values > float(jump_threshold_m))),
    }


def _apply_ema(records: Sequence[dict[str, Any]], alpha: float, gate_m: float, max_missed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scene_id"])].append(record)
    output: list[dict[str, Any]] = []
    for scene_id, values in grouped.items():
        values.sort(key=lambda item: int(item.get("frame_index", 0)))
        tracker = Fire3DTracker(alpha=alpha, gate_m=gate_m, max_missed=max_missed)
        for record in values:
            state = tracker.update(
                point=record.get("ray_point"),
                covariance=record.get("ray_std_m"),
                confidence=float(record.get("ray_quality", 0.0)),
            )
            prediction = None if state.point is None else np.asarray(state.point, dtype=np.float64).copy()
            output.append({**record, "ema_point": prediction, "ema_accepted": bool(state.accepted)})
    return output


def _apply_ekf(
    records: Sequence[dict[str, Any]],
    process_accel_std: float,
    measurement_std_m: float,
    gate_mahalanobis2: float,
    max_missed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scene_id"])].append(record)
    output: list[dict[str, Any]] = []
    for scene_id, values in grouped.items():
        values.sort(key=lambda item: int(item.get("frame_index", 0)))
        tracker = Fire3DEKF(
            dt=1.0,
            process_accel_std=process_accel_std,
            measurement_std_m=measurement_std_m,
            gate_mahalanobis2=gate_mahalanobis2,
            max_missed=max_missed,
        )
        for record in values:
            state = tracker.update(
                point=record.get("ray_point"),
                covariance=record.get("ray_std_m"),
                confidence=float(record.get("ray_quality", 0.0)),
            )
            prediction = None if state.position is None else np.asarray(state.position, dtype=np.float64).copy()
            output.append(
                {
                    **record,
                    "ekf_point": prediction,
                    "ekf_accepted": bool(state.accepted),
                    "ekf_mahalanobis2": state.mahalanobis2,
                }
            )
    return output


def _filter_metrics(records: Sequence[dict[str, Any]], key: str, jump_threshold_m: float) -> dict[str, Any]:
    targets = [record.get("target") for record in records]
    predictions = [record.get(key) for record in records]
    result = _metric_summary(targets, predictions)
    result["stability"] = _stability(records, key, jump_threshold_m)
    return result


def _select_filter_parameters(
    validation_records: Sequence[dict[str, Any]],
    jump_threshold_m: float,
) -> dict[str, Any]:
    """Select EMA/EKF parameters using validation scenes only."""

    ema_candidates = (0.15, 0.25, 0.35, 0.50, 0.70)
    best_ema: Optional[dict[str, Any]] = None
    for alpha in ema_candidates:
        filtered = _apply_ema(validation_records, alpha, gate_m=1.0, max_missed=3)
        metrics = _filter_metrics(filtered, "ema_point", jump_threshold_m)
        score = float(metrics["mae_m"]) if metrics["mae_m"] is not None else float("inf")
        if best_ema is None or score < best_ema["score"]:
            best_ema = {"alpha": alpha, "score": score, "metrics": metrics}

    best_ekf: Optional[dict[str, Any]] = None
    for process_std in (0.05, 0.15, 0.35, 0.60):
        for measurement_std in (0.10, 0.25, 0.50, 0.75):
            filtered = _apply_ekf(
                validation_records,
                process_accel_std=process_std,
                measurement_std_m=measurement_std,
                gate_mahalanobis2=16.27,
                max_missed=3,
            )
            metrics = _filter_metrics(filtered, "ekf_point", jump_threshold_m)
            score = float(metrics["mae_m"]) if metrics["mae_m"] is not None else float("inf")
            if best_ekf is None or score < best_ekf["score"]:
                best_ekf = {
                    "process_accel_std": process_std,
                    "measurement_std_m": measurement_std,
                    "gate_mahalanobis2": 16.27,
                    "score": score,
                    "metrics": metrics,
                }
    if best_ema is None or best_ekf is None:
        raise RuntimeError("Could not select temporal filter parameters")
    return {"ema": best_ema, "ekf": best_ekf}


def _load_scene_mlp_predictions(
    rows: Sequence[dict[str, Any]],
    condition: str,
    checkpoint: Optional[Path],
    device: str,
) -> dict[str, dict[str, Any]]:
    if checkpoint is None or not checkpoint.is_file():
        return {}
    # ``PhysicsSceneCoordinateMLP`` uses the explicit condition contract and
    # does not fall back from a missing noisy observation to a clean label.
    samples, _ = build_samples(rows, condition)
    if not samples:
        return {}
    model = PhysicsSceneCoordinateMLP.load(checkpoint, device=device)
    prediction = model.predict(samples)
    output: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        sample_id = str(sample.row.get("sample_id"))
        output[sample_id] = {
            "point": prediction["xyz"][index],
            "std_m": prediction["std_m"][index],
            "sigma_norm_m": float(np.linalg.norm(prediction["std_m"][index])),
        }
    return output


def _protected_route(
    records: Sequence[dict[str, Any]],
    mlp_predictions: dict[str, dict[str, Any]],
    min_hit_rate: float,
    max_spread_m: float,
    min_ray_quality: float,
    max_mlp_sigma_m: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        ray_ok = (
            record.get("ray_point") is not None
            and float(record.get("ray_hit_rate", 0.0)) >= min_hit_rate
            and float(record.get("ray_spread_m") or 0.0) <= max_spread_m
            and float(record.get("ray_quality", 0.0)) >= min_ray_quality
        )
        point = None
        source = "not_localized"
        if ray_ok:
            point = record["ray_point"]
            source = "ray_casting_protected"
        else:
            mlp = mlp_predictions.get(str(record["sample_id"]))
            if mlp is not None and mlp["point"] is not None and mlp["sigma_norm_m"] <= max_mlp_sigma_m:
                point = mlp["point"]
                source = "scene_mlp_fallback"
        output.append({**record, "route_point": point, "route_source": source, "ray_protected": ray_ok})
    return output


def _describe(values: Sequence[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _condition_benchmark(
    rows: Sequence[dict[str, Any]],
    condition: str,
    mesh: TriangleMesh,
    ray_count: int,
    base_sigma_px: float,
    seed: int,
    max_dist: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _visible_rows(rows, condition):
        result = _ray_localize(
            row,
            condition,
            mesh,
            ray_count,
            base_sigma_px,
            seed=_stable_seed(row.get("sample_id", ""), seed),
            max_dist=max_dist,
        )
        records.append(
            {
                "sample_id": row.get("sample_id"),
                "scene_id": row.get("scene_id", "unknown"),
                "frame_index": int(row.get("frame_index", 0)),
                "surface": row.get("fire_surface", "unknown"),
                "condition": condition,
                "target": result["target"],
                "ray_point": result["point"],
                "ray_std_m": result["std_m"],
                "ray_hit_count": result["hit_count"],
                "ray_count": result["ray_count"],
                "ray_hit_rate": result["hit_rate"],
                "ray_spread_m": result["spread_m"],
                "ray_uncertainty_m": result["uncertainty_m"],
                "ray_quality": result.get("quality", 0.0),
                "roi_confidence_proxy": result["confidence"],
                "pixel_sigma_px": result.get("candidate_meta", {}).get("pixel_sigma_px"),
                "candidate_pixels": result.get("candidates"),
                "candidate_weights": result.get("candidate_weights"),
                "candidate_meta": result.get("candidate_meta"),
                "ray_latency_ms": result["latency_ms"],
                "status": result["status"],
            }
        )
    targets = [record["target"] for record in records]
    points = [record["ray_point"] for record in records]
    valid = [record for record in records if record["ray_point"] is not None]
    by_surface: dict[str, Any] = {}
    for surface in sorted({str(record["surface"]) for record in records}):
        group = [record for record in records if str(record["surface"]) == surface]
        by_surface[surface] = _metric_summary(
            [item["target"] for item in group],
            [item["ray_point"] for item in group],
        )
    summary = {
        "condition": condition,
        "ray_count": ray_count,
        "records": len(records),
        "metrics": _metric_summary(targets, points),
        "by_surface": by_surface,
        "ray_hit_rate": float(len(valid) / max(1, len(records))),
        "mean_candidate_hit_rate": float(np.mean([item["ray_hit_rate"] for item in records])) if records else None,
        "mean_spread_m": float(np.mean([item["ray_spread_m"] for item in valid])) if valid else None,
        "mean_uncertainty_m": float(np.mean([item["ray_uncertainty_m"] for item in valid])) if valid else None,
        "latency_ms": _describe([float(item["ray_latency_ms"]) for item in records]),
        "confidence_proxy": {
            "mean": float(np.mean([item["roi_confidence_proxy"] for item in records])) if records else None,
            "min": float(np.min([item["roi_confidence_proxy"] for item in records])) if records else None,
            "max": float(np.max([item["roi_confidence_proxy"] for item in records])) if records else None,
        },
    }
    return records, summary


def _mesh_from_noise(base_vertices: np.ndarray, faces: np.ndarray, sigma_m: float, seed: int) -> TriangleMesh:
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(sigma_m), size=base_vertices.shape) if sigma_m > 0 else np.zeros_like(base_vertices)
    return TriangleMesh(base_vertices + noise, faces)


def _plot_outputs(summary: dict[str, Any], output: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths: list[str] = []
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    ray_table = summary["ray_count_benchmark"]
    for condition in CONDITIONS:
        values = [ray_table[str(count)][condition]["metrics"]["mae_m"] for count in RAY_COUNTS]
        hit_rates = [ray_table[str(count)][condition]["ray_hit_rate"] for count in RAY_COUNTS]
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        axis.plot(RAY_COUNTS, values, marker="o", label="3D MAE (m)")
        axis.set_xlabel("Number of rays")
        axis.set_ylabel("3D MAE (m)")
        axis.set_xticks(RAY_COUNTS)
        axis.grid(alpha=0.25)
        axis.set_title(f"Multi-ray Ray Casting — {condition}")
        twin = axis.twinx()
        twin.plot(RAY_COUNTS, hit_rates, marker="s", color="tab:orange", label="Ray hit rate")
        twin.set_ylabel("Ray hit rate")
        twin.set_ylim(0.0, 1.05)
        figure.tight_layout()
        path = figures / f"multi_ray_{condition}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))

    mesh_table = summary["mesh_noise_benchmark"]["noisy_estimated"]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for count in RAY_COUNTS:
        values = [mesh_table[str(sigma)][str(count)]["metrics"]["mae_m"] for sigma in MESH_NOISE_M]
        axis.plot(MESH_NOISE_M, values, marker="o", label=f"{count} rays")
    axis.set_xlabel("Mesh vertex noise σ (m)")
    axis.set_ylabel("3D MAE (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Ray Casting sensitivity to mesh noise — noisy estimated")
    figure.tight_layout()
    path = figures / "mesh_noise_sensitivity.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(str(path))

    temporal = summary["temporal_benchmark"]
    labels = ["raw ray", "EMA", "EKF"]
    values = [
        temporal["raw"]["metrics"]["mae_m"],
        temporal["ema"]["metrics"]["mae_m"],
        temporal["ekf"]["metrics"]["mae_m"],
    ]
    jumps = [
        temporal["raw"]["metrics"]["stability"]["jump_rate"],
        temporal["ema"]["metrics"]["stability"]["jump_rate"],
        temporal["ekf"]["metrics"]["stability"]["jump_rate"],
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    positions = np.arange(len(labels))
    axis.bar(positions - 0.18, values, width=0.36, label="3D MAE (m)")
    twin = axis.twinx()
    twin.bar(positions + 0.18, jumps, width=0.36, color="tab:orange", label="Jump rate")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("3D MAE (m)")
    twin.set_ylabel("Jump rate")
    twin.set_ylim(0.0, 1.05)
    axis.set_title("Temporal filtering — noisy estimated, 5 rays")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = figures / "temporal_filtering.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(str(path))
    return paths


def _write_html(summary: dict[str, Any], output: Path, figure_paths: Sequence[str]) -> Path:
    def fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return html.escape(str(value))

    rows: list[str] = []
    for count in RAY_COUNTS:
        for condition in CONDITIONS:
            item = summary["ray_count_benchmark"][str(count)][condition]
            metric = item["metrics"]
            rows.append(
                "<tr>"
                f"<td>{count}</td><td>{html.escape(condition)}</td>"
                f"<td>{fmt(metric['mae_m'])}</td><td>{fmt(metric['median_m'])}</td>"
                f"<td>{fmt(metric['p95_m'])}</td><td>{fmt(item['ray_hit_rate'])}</td>"
                f"<td>{fmt(item['latency_ms']['mean'])}</td></tr>"
            )
    images = "".join(
        f'<p><img src="{html.escape(Path(path).relative_to(output).as_posix())}" alt="benchmark figure" style="max-width:720px"></p>'
        for path in figure_paths
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Priority 1 Ray Casting Benchmark</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:.45rem;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}img{{display:block;margin:1rem 0}}</style></head>
<body><h1>Priority 1 Ray Casting Benchmark</h1>
<p>Metric geometry synthetic benchmark. Confidence is a generated ROI proxy; it is not a real ROI heatmap.</p>
<table><thead><tr><th>Rays</th><th>Condition</th><th>MAE (m)</th><th>Median (m)</th><th>P95 (m)</th><th>Hit rate</th><th>Latency (ms)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>{images}</body></html>"""
    path = output / "priority1_report.html"
    path.write_text(document, encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    test_rows = _load_rows(dataset, "test")
    val_rows = _load_rows(dataset, "val")
    mesh_path = dataset / "room_mesh.json"
    base_mesh = load_triangle_mesh(mesh_path)
    mesh_data = json.loads(mesh_path.read_text(encoding="utf-8"))
    base_vertices = np.asarray(mesh_data["vertices"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh_data["faces"], dtype=np.int64).reshape(-1, 3)

    # Select temporal parameters on validation scenes only, using the most
    # realistic synthetic condition and the default 5-ray configuration.
    val_records, _ = _condition_benchmark(
        val_rows,
        "noisy_estimated",
        base_mesh,
        ray_count=5,
        base_sigma_px=args.base_sigma_px,
        seed=args.seed + 1000,
        max_dist=args.max_dist,
    )
    filter_parameters = _select_filter_parameters(val_records, args.jump_threshold_m)

    ray_count_benchmark: dict[str, dict[str, Any]] = {}
    all_test_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for count in RAY_COUNTS:
        ray_count_benchmark[str(count)] = {}
        all_test_records[str(count)] = {}
        for condition in CONDITIONS:
            records, result = _condition_benchmark(
                test_rows,
                condition,
                base_mesh,
                ray_count=count,
                base_sigma_px=args.base_sigma_px,
                seed=args.seed,
                max_dist=args.max_dist,
            )
            ray_count_benchmark[str(count)][condition] = result
            all_test_records[str(count)][condition] = records

    # The protected route uses the existing mixed Scene-MLP only when the ray
    # quality gate rejects the geometric result. It never overwrites a valid
    # Ray-Casting result merely because the MLP reports a small uncertainty.
    checkpoint = args.scene_mlp_checkpoint.expanduser().resolve() if args.scene_mlp_checkpoint else None
    mlp_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    if checkpoint is not None and checkpoint.is_file():
        for condition in CONDITIONS:
            mlp_predictions[condition] = _load_scene_mlp_predictions(
                test_rows, condition, checkpoint, args.device
            )
    else:
        mlp_predictions = {condition: {} for condition in CONDITIONS}

    route_summary: dict[str, Any] = {}
    route_records: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        records = all_test_records["5"][condition]
        routed = _protected_route(
            records,
            mlp_predictions.get(condition, {}),
            min_hit_rate=args.min_hit_rate,
            max_spread_m=args.max_spread_m,
            min_ray_quality=args.min_ray_quality,
            max_mlp_sigma_m=args.max_mlp_sigma_m,
        )
        route_records[condition] = routed
        route_summary[condition] = {
            "ray_only": _metric_summary([item["target"] for item in routed], [item["ray_point"] for item in routed]),
            "protected_route": _metric_summary([item["target"] for item in routed], [item["route_point"] for item in routed]),
            "sources": dict(Counter(item["route_source"] for item in routed)),
            "ray_protection_rate": float(np.mean([item["ray_protected"] for item in routed])) if routed else 0.0,
            "route_policy": {
                "min_hit_rate": args.min_hit_rate,
                "max_spread_m": args.max_spread_m,
                "min_ray_quality": args.min_ray_quality,
                "max_mlp_sigma_m": args.max_mlp_sigma_m,
            },
        }

    # Temporal filtering uses the realistic noisy+estimated stream and the
    # best default ray count. Parameter selection remains validation-only.
    temporal_input = all_test_records["5"]["noisy_estimated"]
    ema_records = _apply_ema(
        temporal_input,
        alpha=float(filter_parameters["ema"]["alpha"]),
        gate_m=args.ema_gate_m,
        max_missed=args.max_missed,
    )
    ekf_records = _apply_ekf(
        temporal_input,
        process_accel_std=float(filter_parameters["ekf"]["process_accel_std"]),
        measurement_std_m=float(filter_parameters["ekf"]["measurement_std_m"]),
        gate_mahalanobis2=float(filter_parameters["ekf"]["gate_mahalanobis2"]),
        max_missed=args.max_missed,
    )
    raw_records = [{**record, "raw_point": record.get("ray_point")} for record in temporal_input]
    temporal_summary = {
        "condition": "noisy_estimated",
        "ray_count": 5,
        "parameter_selection": filter_parameters,
        "raw": {"metrics": _filter_metrics(raw_records, "raw_point", args.jump_threshold_m)},
        "ema": {"metrics": _filter_metrics(ema_records, "ema_point", args.jump_threshold_m)},
        "ekf": {"metrics": _filter_metrics(ekf_records, "ekf_point", args.jump_threshold_m)},
    }

    mesh_noise_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        mesh_noise_summary[condition] = {}
        for sigma in MESH_NOISE_M:
            sigma_key = str(float(sigma))
            mesh_noise_summary[condition][sigma_key] = {}
            noisy_mesh = _mesh_from_noise(base_vertices, faces, float(sigma), args.seed + 7000 + int(sigma * 10000))
            for count in RAY_COUNTS:
                _, result = _condition_benchmark(
                    test_rows,
                    condition,
                    noisy_mesh,
                    ray_count=count,
                    base_sigma_px=args.base_sigma_px,
                    seed=args.seed,
                    max_dist=args.max_dist,
                )
                mesh_noise_summary[condition][sigma_key][str(count)] = result

    # Flatten the principal tables for spreadsheet/report use.
    csv_path = output / "priority1_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "table", "condition", "ray_count", "mesh_noise_m", "mae_m", "median_m", "p95_m",
            "hit_rate", "under_0.25m", "under_0.50m", "under_1.00m", "latency_ms", "source_counts",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for count in RAY_COUNTS:
            for condition in CONDITIONS:
                item = ray_count_benchmark[str(count)][condition]
                metric = item["metrics"]
                writer.writerow({
                    "table": "ray_count",
                    "condition": condition,
                    "ray_count": count,
                    "mesh_noise_m": 0.0,
                    "mae_m": metric["mae_m"],
                    "median_m": metric["median_m"],
                    "p95_m": metric["p95_m"],
                    "hit_rate": item["ray_hit_rate"],
                    "under_0.25m": metric["under_0.25m"],
                    "under_0.50m": metric["under_0.50m"],
                    "under_1.00m": metric["under_1.00m"],
                    "latency_ms": item["latency_ms"]["mean"],
                    "source_counts": "",
                })
        for condition in CONDITIONS:
            for sigma in MESH_NOISE_M:
                for count in RAY_COUNTS:
                    item = mesh_noise_summary[condition][str(float(sigma))][str(count)]
                    metric = item["metrics"]
                    writer.writerow({
                        "table": "mesh_noise",
                        "condition": condition,
                        "ray_count": count,
                        "mesh_noise_m": sigma,
                        "mae_m": metric["mae_m"],
                        "median_m": metric["median_m"],
                        "p95_m": metric["p95_m"],
                        "hit_rate": item["ray_hit_rate"],
                        "under_0.25m": metric["under_0.25m"],
                        "under_0.50m": metric["under_0.50m"],
                        "under_1.00m": metric["under_1.00m"],
                        "latency_ms": item["latency_ms"]["mean"],
                        "source_counts": "",
                    })
        for condition, item in route_summary.items():
            metric = item["protected_route"]
            writer.writerow({
                "table": "protected_route",
                "condition": condition,
                "ray_count": 5,
                "mesh_noise_m": 0.0,
                "mae_m": metric["mae_m"],
                "median_m": metric["median_m"],
                "p95_m": metric["p95_m"],
                "hit_rate": metric["valid_rate"],
                "under_0.25m": metric["under_0.25m"],
                "under_0.50m": metric["under_0.50m"],
                "under_1.00m": metric["under_1.00m"],
                "latency_ms": "",
                "source_counts": json.dumps(item["sources"], ensure_ascii=False),
            })

    summary: dict[str, Any] = {
        "format": "LAB_SAM.priority1_raycasting_benchmark.v1",
        "dataset": {
            "root": str(dataset),
            "mesh": str(mesh_path),
            "test_records": len(test_rows),
            "test_visible_conditions": {
                condition: len(_visible_rows(test_rows, condition)) for condition in CONDITIONS
            },
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry; not real CCTV accuracy.",
            "confidence_warning": "noise.observation_confidence is a synthetic ROI confidence proxy; no real heatmap is present.",
        },
        "workflow": {
            "multi_ray": "confidence-scaled bottom-band proxy -> undistorted rays -> mesh intersection -> MAD + weighted median",
            "routing": "valid Ray Casting is protected; Scene-MLP is fallback only",
            "temporal": "validation-selected EMA and EKF on noisy_estimated 5-ray stream",
            "mesh_noise": "independent Gaussian vertex perturbation with topology unchanged",
        },
        "configuration": {
            "ray_counts": list(RAY_COUNTS),
            "mesh_noise_m": list(MESH_NOISE_M),
            "base_sigma_px": args.base_sigma_px,
            "max_dist": args.max_dist,
            "seed": args.seed,
        },
        "ray_count_benchmark": ray_count_benchmark,
        "protected_route_benchmark": route_summary,
        "temporal_benchmark": temporal_summary,
        "mesh_noise_benchmark": mesh_noise_summary,
        "artifacts": {
            "metrics_csv": str(csv_path),
            "scene_mlp_checkpoint": None if checkpoint is None else str(checkpoint),
        },
    }
    figure_paths = _plot_outputs(summary, output)
    summary["artifacts"]["figures"] = figure_paths
    report_path = _write_html(summary, output, figure_paths)
    summary["artifacts"]["html_report"] = str(report_path)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    # Write compact per-frame records separately so the main summary remains
    # readable and the inputs to later visualisation are reproducible.
    records_path = output / "ray_records_noisy_estimated_5.json"
    records_path.write_text(
        json.dumps(all_test_records["5"]["noisy_estimated"], indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    summary["artifacts"]["frame_records"] = str(records_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"saved_summary={summary_path}")
    print(f"saved_csv={csv_path}")
    print(f"saved_report={report_path}")
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nPriority 1 — Ray Casting-first benchmark")
    print("ray_count / condition              MAE(m)  median(m)  P95(m)  hit-rate")
    for count in RAY_COUNTS:
        for condition in CONDITIONS:
            item = summary["ray_count_benchmark"][str(count)][condition]
            metric = item["metrics"]
            print(
                f"{count:>3} / {condition:<18} "
                f"{metric['mae_m'] if metric['mae_m'] is not None else float('nan'):>7.3f} "
                f"{metric['median_m'] if metric['median_m'] is not None else float('nan'):>9.3f} "
                f"{metric['p95_m'] if metric['p95_m'] is not None else float('nan'):>7.3f} "
                f"{item['ray_hit_rate']:>7.1%}"
            )
    temporal = summary["temporal_benchmark"]
    print("\nTemporal — noisy_estimated, 5 rays")
    for name in ("raw", "ema", "ekf"):
        metric = temporal[name]["metrics"]
        stability = metric["stability"]
        print(
            f"{name:<5} MAE={metric['mae_m']:.3f}m "
            f"jump-rate={stability['jump_rate'] if stability['jump_rate'] is not None else float('nan'):.1%}"
        )
    print("\nProtected routing")
    for condition, item in summary["protected_route_benchmark"].items():
        metric = item["protected_route"]
        print(f"{condition:<18} MAE={metric['mae_m']:.3f}m sources={item['sources']}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "priority1_raycasting_20261008",
    )
    parser.add_argument(
        "--scene-mlp-checkpoint",
        type=Path,
        default=root / "output" / "physics_scene_all_rerun_20261008" / "checkpoints" / "physics_scene_mlp_v2_pinhole_domain.pth",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-sigma-px", type=float, default=3.0)
    parser.add_argument("--max-dist", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--min-hit-rate", type=float, default=0.60)
    parser.add_argument("--max-spread-m", type=float, default=0.35)
    parser.add_argument("--min-ray-quality", type=float, default=0.05)
    parser.add_argument("--max-mlp-sigma-m", type=float, default=1.50)
    parser.add_argument("--ema-gate-m", type=float, default=1.0)
    parser.add_argument("--max-missed", type=int, default=3)
    parser.add_argument("--jump-threshold-m", type=float, default=0.50)
    args = parser.parse_args()
    if args.base_sigma_px < 0 or args.max_dist <= 0:
        raise ValueError("base-sigma-px must be non-negative and max-dist must be positive")
    if not 0 <= args.min_hit_rate <= 1 or not 0 <= args.min_ray_quality <= 1:
        raise ValueError("hit-rate and ray-quality thresholds must be in [0, 1]")
    if args.max_spread_m <= 0 or args.max_mlp_sigma_m <= 0 or args.jump_threshold_m <= 0:
        raise ValueError("spread, MLP sigma and jump thresholds must be positive")
    summary = run(args)
    _print_summary(summary)


if __name__ == "__main__":
    main()
