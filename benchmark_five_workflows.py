"""Benchmark the five post-ROI fire-localisation workflows on one manifest.

The branches are deliberately evaluated on the same selected records:

``ray``
    Calibrated multi-ray casting against a mesh, with the existing robust
    multi-ray aggregation and outlier rejection.
``ipm``
    Homography/IPM projection to the floor plane ``Z=0``.
``gpr``
    A GPR/IDW residual correction trained on the *train* split only, on top of
    the Ray Casting estimate.
``depth``
    Optional monocular/precomputed depth projection.  It is marked unavailable
    when no metric depth backend is supplied; relative depth is never reported
    as metric XYZ without an explicit scale.
``fusion``
    Uncertainty-aware combination that gives Ray Casting priority and rejects
    auxiliary estimates that disagree too much.  EMA and EKF filtered versions
    are also recorded for sequence stability.

This script does not delete or modify source datasets.  Synthetic metric data
is a geometry smoke test; real-room metre accuracy requires measured camera,
mesh and ``fire_xyz_world`` labels in one coordinate frame.

Quick CPU smoke test::

    .venv\\Scripts\\python.exe benchmark_five_workflows.py ^
      --dataset working\\synthetic_fire_3d_v3 ^
      --split test --max-records 12 --no-roi --depth-backend none ^
      --output-dir output\\five_workflows_smoke

Then render PNG/HTML/PLY views::

    .venv\\Scripts\\python.exe visualize_3d_results.py ^
      --summary output\\five_workflows_smoke\\summary.json ^
      --dataset working\\synthetic_fire_3d_v3 ^
      --output-dir output\\five_workflows_smoke\\visualization ^
      --write-ply
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
from PIL import Image, ImageDraw, ImageOps

from camera_calibration import CameraCalibration
from depth_to_world import depth_map_pixel_to_world
from ekf_tracker import Fire3DEKF
from homography_floor import FloorHomography
from localization import localize_pixels
from mesh_loader import load_triangle_mesh
from monocular_depth_adapter import MonocularDepthAdapter
from paper_workflow_3d import (
    _as_point,
    _as_xyz,
    _calibration_from_row,
    _image_path,
    _load_external_3d,
    _load_external_calibration,
    _load_manifest,
    _raycast,
    _row_pixel,
    _safe_roi_point,
    _select_records,
)
from pixel_world_regressor import GPRResidualMapper, ProjectionEstimate
from tracking_3d import Fire3DTracker
from uncertainty_fusion import WorldCandidate, fuse_world_candidates


BRANCHES = ("ray", "ipm", "gpr", "depth", "fusion", "fusion_ema", "fusion_ekf")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {
    "gt": (31, 157, 85),
    "ray": (40, 120, 208),
    "ipm": (23, 162, 184),
    "gpr": (148, 103, 189),
    "depth": (255, 127, 14),
    "fusion": (214, 39, 40),
}


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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _calibration_for_row(row: dict[str, Any], args: argparse.Namespace, external: Optional[CameraCalibration]) -> CameraCalibration:
    if args.calibration_source == "external":
        if external is None:
            raise ValueError("external calibration is not loaded")
        return external
    return _calibration_from_row(row, args.calibration_source)


def _mesh_for_args(dataset: Path, args: argparse.Namespace) -> tuple[Any, Path]:
    if args.mesh:
        mesh_path = args.mesh.expanduser().resolve()
    elif args.floor_only and (dataset / "room_floor_mesh.json").is_file():
        mesh_path = dataset / "room_floor_mesh.json"
    else:
        mesh_path = dataset / "room_mesh.json"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    return load_triangle_mesh(mesh_path), mesh_path


def _pixel_error(point: Any, gt: Optional[np.ndarray]) -> Optional[float]:
    value = _as_point(point)
    return None if value is None or gt is None else float(np.linalg.norm(value - gt))


def _projection_to_location(estimate: ProjectionEstimate) -> dict[str, Any]:
    return {
        "hit": bool(estimate.success and estimate.point is not None),
        "status": "valid_" + estimate.method if estimate.success else estimate.reason or "no_result",
        "point": estimate.point,
        "std_m": estimate.std_m,
        "spread_m": None if estimate.std_m is None else float(np.linalg.norm(estimate.std_m)),
        "confidence": float(estimate.confidence),
        "method": estimate.method,
    }


def _branch_location(point: Optional[np.ndarray], std: Optional[np.ndarray], confidence: float, method: str, reason: Optional[str] = None) -> dict[str, Any]:
    return {
        "hit": point is not None,
        "status": "valid_" + method if point is not None else reason or "no_result",
        "point": point,
        "std_m": std,
        "spread_m": None if std is None else float(np.linalg.norm(std)),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "method": method,
        "reason": reason,
    }


def _candidate_from_location(method: str, location: Optional[dict[str, Any]], prior: float) -> WorldCandidate:
    if not location or not location.get("hit"):
        return WorldCandidate(method, None, confidence=0.0, available=False, prior=prior, reason="not_available")
    return WorldCandidate(
        method,
        _as_xyz(location.get("point")),
        _as_xyz(location.get("std_m")),
        float(location.get("confidence", 0.0)),
        available=True,
        prior=prior,
    )


def _ray_for_pixel(calibration: CameraCalibration, mesh: Any, point: Optional[np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    location, hit_rate, statuses, latency = _raycast(
        calibration,
        mesh,
        point,
        args.ray_pattern,
        args.ray_radius_px,
        args.max_dist,
        args.step,
    )
    if location is None:
        location = {"hit": False, "status": "no_pixel", "point": None, "ray_count": 0, "ray_hits": 0}
    location = dict(location)
    location.update({"ray_hit_rate": hit_rate, "ray_statuses": statuses, "latency_ms": latency})
    return location


def _ipm_for_pixel(calibration: CameraCalibration, point: Optional[np.ndarray]) -> dict[str, Any]:
    if point is None:
        return _branch_location(None, None, 0.0, "ipm", "no_pixel")
    started = time.perf_counter()
    try:
        ideal = calibration.undistort_pixels(np.asarray(point, dtype=np.float64).reshape(1, 2))
        world = FloorHomography.from_calibration(calibration).pixel_to_floor_xyz(ideal)[0]
        return {
            **_branch_location(world, np.full(3, 0.08), 1.0, "ipm"),
            "projection_ms": (time.perf_counter() - started) * 1000.0,
            "ipm_success": True,
        }
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return {
            **_branch_location(None, None, 0.0, "ipm", str(exc)),
            "projection_ms": (time.perf_counter() - started) * 1000.0,
            "ipm_success": False,
        }


def _depth_for_pixel(
    adapter: MonocularDepthAdapter,
    image: Image.Image,
    image_path: Path,
    sample_id: str,
    calibration: CameraCalibration,
    point: Optional[np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    if point is None:
        return {**_branch_location(None, None, 0.0, "depth", "no_pixel"), "depth_ms": 0.0, "backend": adapter.backend}
    prediction = adapter.predict(image, image_path, sample_id)
    if not prediction.success or prediction.depth_map is None:
        return {
            **_branch_location(None, None, 0.0, "depth", prediction.reason or "depth_unavailable"),
            "depth_ms": (time.perf_counter() - started) * 1000.0,
            "backend": prediction.backend,
            "depth_prediction": prediction.to_dict(),
        }
    estimate = depth_map_pixel_to_world(
        calibration,
        prediction.depth_map,
        point,
        units=prediction.units,
        scale=args.depth_scale,
        offset=args.depth_offset,
        radius_px=args.depth_radius_px,
    )
    location = _branch_location(estimate.point, estimate.std_m, prediction.confidence if estimate.success else 0.0, "depth", estimate.reason)
    location.update({
        "depth_ms": (time.perf_counter() - started) * 1000.0,
        "backend": prediction.backend,
        "depth_prediction": prediction.to_dict(),
        "depth_value": estimate.depth_value,
    })
    return location


def _fit_gpr(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    mesh: Any,
    calibration_override: Optional[CameraCalibration] = None,
) -> tuple[Optional[GPRResidualMapper], dict[str, Any]]:
    if not args.enable_gpr:
        return None, {"enabled": False, "reason": "disabled"}
    pixels: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sizes: list[list[int]] = []
    for row in rows:
        if str(row.get("split")) != args.gpr_fit_split:
            continue
        gt = _as_xyz(row.get("fire_xyz_world"))
        noisy = _row_pixel(row, "p_fire_noisy_pixel")
        if noisy is None:
            noisy = _row_pixel(row, "p_fire_pixel")
        if gt is None or noisy is None:
            continue
        calibration = calibration_override or _calibration_from_row(row, "true")
        ray = _ray_for_pixel(calibration, mesh, noisy, args)
        base = _as_xyz(ray.get("point"))
        if base is None:
            continue
        pixels.append(noisy)
        bases.append(base)
        targets.append(gt)
        sizes.append(list(row.get("image_size", [640, 640])))
    if len(pixels) < 4:
        return None, {"enabled": True, "fitted": False, "samples": len(pixels), "reason": "need_at_least_four_ray_hits"}
    mapper = GPRResidualMapper(default_std_m=args.gpr_default_std_m)
    mapper.fit(pixels, bases, targets, sizes, max_samples=args.gpr_max_samples, seed=args.seed)
    model_path = args.output_dir / "gpr_residual.pkl"
    mapper.save(model_path)
    return mapper, {
        "enabled": True,
        "fitted": True,
        "backend": mapper.backend,
        "samples": len(pixels),
        "fit_split": args.gpr_fit_split,
        "model": str(model_path),
    }


def _apply_temporal_filters(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scene_id", "unknown"))].append(row)
    for sequence in grouped.values():
        sequence.sort(key=lambda item: int(item.get("frame_index", 0)))
        ema = Fire3DTracker(alpha=args.ema_alpha, gate_m=args.temporal_gate_m, max_missed=args.max_missed)
        ekf = Fire3DEKF(
            dt=args.ekf_dt,
            process_accel_std=args.ekf_process_std,
            measurement_std_m=args.ekf_measurement_std,
            gate_mahalanobis2=args.ekf_gate,
            max_missed=args.max_missed,
        )
        for row in sequence:
            fusion_branch = row.get("branches", {}).get("fusion", {}) or {}
            location = fusion_branch.get("location", fusion_branch) or {}
            point = _as_xyz(location.get("point")) if location.get("hit", location.get("success", False)) else None
            std = _as_xyz(location.get("std_m"))
            confidence = float(location.get("confidence", 0.0))
            ema_state = ema.update(point, confidence=confidence)
            ekf_state = ekf.update(point, covariance=std, confidence=confidence)
            ema_location = _branch_location(
                None if not ema_state.accepted else ema_state.point,
                None if ema_state.covariance is None else np.sqrt(np.maximum(np.diag(ema_state.covariance), 0.0)),
                ema_state.confidence,
                "fusion_ema",
                "filtered_or_rejected" if not ema_state.accepted else None,
            )
            ekf_location = _branch_location(
                None if not ekf_state.accepted else ekf_state.position,
                None if ekf_state.covariance is None else np.sqrt(np.maximum(np.diag(ekf_state.covariance)[:3], 0.0)),
                ekf_state.confidence,
                "fusion_ekf",
                "filtered_or_rejected" if not ekf_state.accepted else None,
            )
            row["branches"]["fusion_ema"] = {
                "pixel": fusion_branch.get("pixel"),
                "location": ema_location,
                "latency_ms": fusion_branch.get("latency_ms", 0.0),
            }
            row["branches"]["fusion_ekf"] = {
                "pixel": fusion_branch.get("pixel"),
                "location": ekf_location,
                "latency_ms": fusion_branch.get("latency_ms", 0.0),
            }


def _metrics(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    pixel_errors: list[float] = []
    deltas: list[np.ndarray] = []
    locations = 0
    successes = 0
    hits = 0
    attempts = 0
    latencies: list[float] = []
    fallback = 0
    status_counts: dict[str, int] = {}
    for row in rows:
        data = row.get("branches", {}).get(branch, {}) or {}
        pixel = _as_point(data.get("pixel"))
        gt_pixel = _as_point(row.get("gt_pixel"))
        if pixel is not None and gt_pixel is not None:
            pixel_errors.append(float(np.linalg.norm(pixel - gt_pixel)))
        location = data.get("location", data)
        point = _as_xyz(location.get("point"))
        gt = _as_xyz(row.get("gt_xyz"))
        if point is not None and gt is not None:
            deltas.append(point - gt)
        locations += int(bool(location))
        successes += int(bool(location.get("hit")))
        attempts += int(location.get("ray_count", 0))
        hits += int(location.get("ray_hits", 0))
        for key in ("latency_ms", "total_ms", "projection_ms", "depth_ms"):
            if data.get(key) is not None:
                latencies.append(float(data[key]))
        fallback += int(bool(data.get("fallback")))
        status = str(location.get("status", "no_result"))
        status_counts[status] = status_counts.get(status, 0) + 1
    pixels = np.asarray(pixel_errors, dtype=np.float64)
    xyz = np.asarray(deltas, dtype=np.float64).reshape(-1, 3) if deltas else np.empty((0, 3))
    norms = np.linalg.norm(xyz, axis=1) if len(xyz) else np.empty(0)
    metric: dict[str, Any] = {
        "branch": branch,
        "samples": len(rows),
        "pixel_samples": int(len(pixels)),
        "pixel_mae_px": None if not len(pixels) else float(pixels.mean()),
        "pixel_median_px": None if not len(pixels) else float(np.median(pixels)),
        "pixel_p95_px": None if not len(pixels) else float(np.percentile(pixels, 95)),
        "pck10": None if not len(pixels) else float(np.mean(pixels <= 10.0)),
        "pck25": None if not len(pixels) else float(np.mean(pixels <= 25.0)),
        "location_success_rate": float(successes / max(1, locations)),
        "ray_hit_rate": None if not attempts else float(hits / attempts),
        "ray_attempts": attempts,
        "ray_hits": hits,
        "three_d_samples": int(len(norms)),
        "three_d_mae_m": None if not len(norms) else float(norms.mean()),
        "three_d_median_m": None if not len(norms) else float(np.median(norms)),
        "three_d_p95_m": None if not len(norms) else float(np.percentile(norms, 95)),
        "under_0.10m": None if not len(norms) else float(np.mean(norms <= 0.10)),
        "under_0.25m": None if not len(norms) else float(np.mean(norms <= 0.25)),
        "under_0.50m": None if not len(norms) else float(np.mean(norms <= 0.50)),
        "under_1.00m": None if not len(norms) else float(np.mean(norms <= 1.00)),
        "xyz_mae_m": None if not len(xyz) else np.mean(np.abs(xyz), axis=0),
        "xyz_p95_abs_m": None if not len(xyz) else np.percentile(np.abs(xyz), 95, axis=0),
        "fallback_count": fallback,
        "fallback_rate": float(fallback / max(1, len(rows))),
        "status_counts": status_counts,
        "latency_ms": {
            "mean": None if not latencies else float(np.mean(latencies)),
            "median": None if not latencies else float(np.median(latencies)),
            "p95": None if not latencies else float(np.percentile(latencies, 95)),
        },
    }
    return metric


def _sequence_metrics(rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scene_id", "unknown"))].append(row)
    jumps: list[float] = []
    jitter: list[float] = []
    valid = total = 0
    for sequence in grouped.values():
        sequence.sort(key=lambda item: int(item.get("frame_index", 0)))
        previous = None
        for row in sequence:
            total += 1
            branch_data = row.get("branches", {}).get(branch, {}) or {}
            location = branch_data.get("location", branch_data) or {}
            point = _as_xyz(location.get("point")) if location.get("hit") else None
            if point is None:
                previous = None
                continue
            valid += 1
            if previous is not None:
                distance = float(np.linalg.norm(point - previous))
                jitter.append(distance)
                jumps.append(float(distance > 0.50))
            previous = point
    return {
        "branch": branch,
        "sequences": len(grouped),
        "frames": total,
        "valid_frame_rate": float(valid / max(1, total)),
        "three_d_jitter_median": None if not jitter else float(np.median(jitter)),
        "three_d_jitter_p95": None if not jitter else float(np.percentile(jitter, 95)),
        "jump_rate_over_0.5m": None if not jumps else float(np.mean(jumps)),
    }


def _marker(draw: ImageDraw.ImageDraw, point: Optional[np.ndarray], color: tuple[int, int, int], label: str) -> None:
    if point is None:
        return
    x, y = [float(v) for v in point[:2]]
    radius = 6
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.text((x + radius + 2, y - radius - 2), label, fill=color)


def _contact_sheet(rows: list[dict[str, Any]], output: Path, maximum: int = 16) -> None:
    selected = rows if maximum <= 0 else rows[:maximum]
    if not selected:
        return
    tiles: list[Image.Image] = []
    for row in selected:
        path = Path(str(row["image_path"]))
        if not path.is_file():
            continue
        with Image.open(path) as opened:
            image = opened.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        _marker(draw, _as_point(row.get("gt_pixel")), COLORS["gt"], "GT")
        for branch in ("ray", "ipm", "gpr", "depth", "fusion"):
            point = _as_point((row.get("branches", {}).get(branch, {}) or {}).get("pixel"))
            _marker(draw, point, COLORS[branch], branch)
        tiles.append(ImageOps.contain(image, (420, 320)))
    if not tiles:
        return
    columns = 4
    rows_count = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 420, rows_count * 320), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * 420 + (420 - tile.width) // 2
        y = (index // columns) * 320 + (320 - tile.height) // 2
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    rows = _load_manifest(dataset, args.split)
    rows = _select_records(rows, args.max_records, args.selection)
    mesh, mesh_path = _mesh_for_args(dataset, args)
    external_calibration = None
    if args.calibration_source == "external":
        if args.calibration is None:
            raise ValueError("--calibration is required in external mode")
        external_calibration = _load_external_calibration(args.calibration.expanduser().resolve())
    external_3d = _load_external_3d(args.labels_3d.expanduser().resolve() if args.labels_3d else None)

    refiner = None
    if args.roi_checkpoint is not None and not args.no_roi:
        from narrow_localizer import ROIRefinerInference

        refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device or "cpu")
    detector = None
    if args.coarse_source == "detector":
        if args.detector_checkpoint is None:
            raise ValueError("--detector-checkpoint is required with --coarse-source detector")
        from fire_detector import FireDetector

        detector = FireDetector(args.detector_checkpoint, device=args.device, threshold=args.threshold)
        detector.warmup(repeats=1)
    depth_adapter = MonocularDepthAdapter(
        backend=args.depth_backend,
        map_root=args.depth_map_root,
        model_name=args.depth_model,
        units=args.depth_units,
    )
    try:
        fit_rows = _load_manifest(dataset, args.gpr_fit_split)
    except (FileNotFoundError, RuntimeError) as exc:
        fit_rows = []
        gpr_missing_reason = f"fit_split_unavailable: {exc}"
    else:
        gpr_missing_reason = None
    gpr, gpr_info = _fit_gpr(fit_rows, args, mesh, external_calibration)
    if gpr_missing_reason is not None and gpr_info.get("reason") == "need_at_least_four_ray_hits":
        gpr_info["reason"] = gpr_missing_reason

    processed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["_image_path"])
        image = Image.open(image_path).convert("RGB")
        calibration = _calibration_for_row(row, args, external_calibration)
        gt_pixel = _row_pixel(row, "p_fire_pixel")
        gt_xyz = _as_xyz(row.get("fire_xyz_world"))
        if gt_xyz is None:
            gt_xyz = _lookup_external_xyz(external_3d, row)
        if args.coarse_source == "manifest":
            coarse = _row_pixel(row, "p_fire_noisy_pixel")
            if coarse is None and args.allow_ground_truth_coarse_fallback:
                coarse = gt_pixel
        elif args.coarse_source == "gt_noise":
            coarse = None if gt_pixel is None else gt_pixel + np.random.default_rng(args.seed + index).normal(0.0, args.gt_noise_px, 2)
        else:
            result = detector.detect(image, warmup=False)
            coarse = None if result.pixel is None else np.asarray(result.pixel, dtype=np.float64)
        refined = None
        roi_confidence = 0.0
        if refiner is not None and coarse is not None:
            refined_result = refiner.refine(image, coarse)
            refined = _as_point(refined_result.point)
            roi_confidence = float(refined_result.confidence)
        safe, blended, reliability = _safe_roi_point(
            coarse,
            refined,
            roi_confidence,
            args.max_shift_px,
            args.min_heatmap_confidence,
            args.blend_alpha,
        )
        points = {"coarse": coarse, "refined": refined, "safe": safe, "blend": blended}
        projection_pixel = blended if blended is not None else coarse
        ray = _ray_for_pixel(calibration, mesh, projection_pixel, args)
        ipm = _ipm_for_pixel(calibration, projection_pixel)
        gpr_location = _branch_location(None, None, 0.0, "gpr", "not_fitted")
        if gpr is not None and projection_pixel is not None and ray.get("point") is not None:
            gpr_estimate = gpr.predict(projection_pixel, ray["point"], row.get("image_size", [640, 640]))
            gpr_location = _projection_to_location(gpr_estimate)
        depth = _depth_for_pixel(depth_adapter, image, image_path, str(row.get("sample_id", image_path.stem)), calibration, projection_pixel, args)
        fusion_result = fuse_world_candidates(
            [
                _candidate_from_location("ray", ray, 1.0),
                _candidate_from_location("ipm", ipm, 0.65),
                _candidate_from_location("gpr", gpr_location, 0.55),
                _candidate_from_location("depth", depth, 0.25),
            ],
            ray_gate_m=args.fusion_ray_gate_m,
            max_uncertainty_m=args.fusion_max_uncertainty_m,
        )
        fusion = fusion_result.to_dict()
        fusion.update({"hit": fusion_result.success, "status": "valid_fusion" if fusion_result.success else fusion_result.reason})
        base_record = {
            "sample_id": row.get("sample_id", image_path.stem),
            "scene_id": row.get("scene_id", image_path.parent.name),
            "frame_index": int(row.get("frame_index", index - 1)),
            "image_path": str(image_path),
            "image_size": list(image.size),
            "gt_pixel": gt_pixel,
            "gt_xyz": gt_xyz,
            "surface": row.get("fire_surface", row.get("surface_type", "unknown")),
            "coarse_pixel": coarse,
            "roi_pixel": refined,
            "blend_pixel": blended,
            "roi_confidence": roi_confidence,
            "roi_reliability": reliability,
            "branches": {
                "ray": {"pixel": projection_pixel, "location": ray, "latency_ms": ray.get("latency_ms", 0.0)},
                "ipm": {"pixel": projection_pixel, "location": ipm, "latency_ms": ipm.get("projection_ms", 0.0)},
                "gpr": {"pixel": projection_pixel, "location": gpr_location, "latency_ms": 0.0},
                "depth": {"pixel": projection_pixel, "location": depth, "latency_ms": depth.get("depth_ms", 0.0)},
                "fusion": {"pixel": projection_pixel, "location": fusion, "fallback": fusion_result.fallback, "latency_ms": sum(float(item.get("latency_ms", 0.0)) for item in (ray, ipm, depth))},
            },
        }
        processed.append(base_record)
        if index == 1 or index == len(rows) or index % 10 == 0:
            print(f"processed={index}/{len(rows)} image={image_path.name}", flush=True)
    _apply_temporal_filters(processed, args)

    summary = {
        "format": "LAB_SAM.benchmark_five_workflows.v1",
        "method": {
            "coarse_source": args.coarse_source,
            "roi_checkpoint": None if args.roi_checkpoint is None else str(args.roi_checkpoint),
            "mesh": str(mesh_path),
            "calibration_source": args.calibration_source,
            "ray_pattern": args.ray_pattern,
            "ray_radius_px": args.ray_radius_px,
            "gpr": gpr_info,
            "depth_backend": args.depth_backend,
            "depth_units": args.depth_units,
            "fusion": "ray-protected uncertainty-weighted fusion",
            "temporal_filters": ["ema", "ekf"],
        },
        "dataset": {
            "root": str(dataset),
            "split": args.split,
            "records_selected": len(processed),
            "selection": args.selection,
            "metric_xyz_records": sum(_as_xyz(row.get("gt_xyz")) is not None for row in processed),
            "warning": "Synthetic/asset-backed metrics are geometry validation, not real-CCTV accuracy.",
        },
        "branches": {branch: _metrics(processed, branch) for branch in BRANCHES},
        "sequence_stability": {branch: _sequence_metrics(processed, branch) for branch in ("fusion", "fusion_ema", "fusion_ekf")},
        "records": processed,
    }
    return summary


def _lookup_external_xyz(labels: dict[str, np.ndarray], row: dict[str, Any]) -> Optional[np.ndarray]:
    for key in (str(row.get("sample_id", "")), Path(str(row.get("image_path", ""))).name, str(row.get("image_path", "")).replace("\\", "/").lower()):
        if key.lower() in labels:
            return labels[key.lower()].copy()
    return None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-records", type=int, default=24)
    parser.add_argument("--selection", choices=("even", "diverse"), default="even")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "five_workflows")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--floor-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Prefer room_floor_mesh.json when a dataset provides it")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--labels-3d", type=Path, default=None)
    parser.add_argument("--calibration-source", choices=("true", "estimated", "external"), default="true")
    parser.add_argument("--coarse-source", choices=("manifest", "gt_noise", "detector"), default="manifest")
    parser.add_argument(
        "--allow-ground-truth-coarse-fallback",
        action="store_true",
        help="Use p_fire_pixel only when manifest has no noisy/detector point; off by default to avoid oracle leakage",
    )
    parser.add_argument("--gt-noise-px", type=float, default=3.0)
    parser.add_argument("--detector-checkpoint", type=Path, default=None)
    parser.add_argument("--roi-checkpoint", type=Path, default=root / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth")
    parser.add_argument("--no-roi", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ray-pattern", choices=("single", "cross", "square"), default="cross")
    parser.add_argument("--ray-radius-px", type=float, default=2.0)
    parser.add_argument("--max-dist", type=float, default=100.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--max-shift-px", type=float, default=40.0)
    parser.add_argument("--min-heatmap-confidence", type=float, default=0.05)
    parser.add_argument("--blend-alpha", type=float, default=0.75)
    parser.add_argument("--enable-gpr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpr-fit-split", choices=("train", "val"), default="train")
    parser.add_argument("--gpr-max-samples", type=int, default=128)
    parser.add_argument("--gpr-default-std-m", type=float, default=0.20)
    parser.add_argument("--depth-backend", choices=("none", "maps", "transformers"), default="none")
    parser.add_argument("--depth-map-root", type=Path, default=None)
    parser.add_argument("--depth-model", default=None)
    parser.add_argument("--depth-units", choices=("relative", "camera_z", "ray_range"), default="relative")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-offset", type=float, default=0.0)
    parser.add_argument("--depth-radius-px", type=int, default=2)
    parser.add_argument("--fusion-ray-gate-m", type=float, default=1.0)
    parser.add_argument("--fusion-max-uncertainty-m", type=float, default=4.0)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--temporal-gate-m", type=float, default=3.0)
    parser.add_argument("--ekf-dt", type=float, default=1.0)
    parser.add_argument("--ekf-process-std", type=float, default=0.35)
    parser.add_argument("--ekf-measurement-std", type=float, default=0.25)
    parser.add_argument("--ekf-gate", type=float, default=16.27)
    parser.add_argument("--max-missed", type=int, default=3)
    parser.add_argument("--max-contact-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.roi_checkpoint = args.roi_checkpoint.expanduser().resolve() if args.roi_checkpoint else None
    args.detector_checkpoint = args.detector_checkpoint.expanduser().resolve() if args.detector_checkpoint else None
    args.mesh = args.mesh.expanduser().resolve() if args.mesh else None
    args.calibration = args.calibration.expanduser().resolve() if args.calibration else None
    args.labels_3d = args.labels_3d.expanduser().resolve() if args.labels_3d else None
    args.depth_map_root = args.depth_map_root.expanduser().resolve() if args.depth_map_root else None
    if args.max_records < 0 or args.max_missed < 0:
        raise ValueError("--max-records and --max-missed must be >= 0")
    if args.calibration_source == "external" and (args.calibration is None or args.labels_3d is None):
        raise ValueError("external mode requires --calibration and --labels-3d")
    if not args.no_roi and args.roi_checkpoint is not None and not args.roi_checkpoint.is_file():
        raise FileNotFoundError(f"ROI checkpoint not found: {args.roi_checkpoint}")
    summary = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "comparison_metrics.json", summary["branches"])
    _write_json(args.output_dir / "sequence_metrics.json", summary["sequence_stability"])
    _contact_sheet(summary["records"], args.output_dir / "comparison_contact_sheet.png", args.max_contact_images)
    print("branch    3D_MAE(m)  median(m)  P95(m)  ray_hit  success")
    for branch in BRANCHES:
        metric = summary["branches"][branch]
        print(
            f"{branch:<11} {metric['three_d_mae_m']!s:>9} {metric['three_d_median_m']!s:>10} "
            f"{metric['three_d_p95_m']!s:>8} {metric['ray_hit_rate']!s:>8} {metric['location_success_rate']:.3f}"
        )
    print(f"saved_summary={args.output_dir / 'summary.json'}")
    print(f"saved_contact_sheet={args.output_dir / 'comparison_contact_sheet.png'}")


if __name__ == "__main__":
    main()
