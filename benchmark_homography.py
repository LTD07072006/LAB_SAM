"""Benchmark floor homography/IPM alternatives against calibrated ray casting.

The script uses the metric fire manifests already present in this project and
keeps the floor-plane assumption explicit. It compares:

- analytic H from camera calibration;
- DLT from four spread-out floor correspondences;
- DLT from all visible floor correspondences;
- RANSAC after synthetic marker-click noise/outliers;
- one static H reused within a scene;
- calibrated mesh ray casting as the reference.

Homography is physically valid only for the nominated plane. S3DIS, TUM-RGB-D,
HSSD and ReplicaCAD assets are useful room/depth sources, but the downloaded
raw datasets do not contain fire-contact labels. The asset-backed ReplicaCAD
manifest generated in this project does, so it can be evaluated directly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from camera_calibration import CameraCalibration
from locator import intersect_ray_with_grid_result
from mesh_loader import load_triangle_mesh


METHODS = ("analytic", "dlt4", "dlt_all", "ransac_noisy", "scene_static", "ray", "ray_floor")
SOURCES = ("coarse", "roi_blend", "oracle")
COLORS = {
    "gt": (31, 157, 85),
    "analytic": (40, 120, 208),
    "dlt4": (242, 142, 43),
    "dlt_all": (148, 103, 189),
    "ransac_noisy": (214, 39, 40),
    "scene_static": (23, 162, 184),
    "ray": (40, 40, 40),
    "ray_floor": (96, 96, 96),
}


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def point(value: Any, dimensions: int = 2) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(array) < dimensions or not np.all(np.isfinite(array[:dimensions])):
        return None
    return array[:dimensions].copy()


def load_rows(dataset: Path, split: str) -> list[dict[str, Any]]:
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and str(row.get("split", "")) != split:
                continue
            if int(row.get("has_fire", 0)) != 1:
                continue
            if point(row.get("p_fire_pixel")) is None:
                continue
            if point(row.get("fire_xyz_world"), 3) is None:
                continue
            raw_image = Path(str(row.get("image_path", "")))
            image_path = raw_image if raw_image.is_absolute() else dataset / raw_image
            row["_image_path"] = str(image_path.resolve())
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No metric fire records in {manifest} split={split!r}")
    rows.sort(key=lambda item: (str(item.get("scene_id", "")), int(item.get("frame_index", 0))))
    return rows


def select_rows(rows: list[dict[str, Any]], maximum: int, policy: str) -> list[dict[str, Any]]:
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    if policy == "head":
        return list(rows[:maximum])
    indices = np.linspace(0, len(rows) - 1, int(maximum), dtype=int)
    return [rows[int(index)] for index in indices]


def load_prediction_index(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Prediction summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["sample_id"]): row
        for row in payload.get("records", [])
        if row.get("sample_id") is not None
    }


def calibration_from_row(row: dict[str, Any]) -> CameraCalibration:
    camera = row.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"Missing camera calibration for {row.get('sample_id')}")
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


def resolve_mesh(dataset: Path, override: Optional[Path]) -> Path:
    path = override.expanduser().resolve() if override else dataset / "room_mesh.json"
    if not path.is_file():
        raise FileNotFoundError(f"Room mesh not found: {path}")
    return path.resolve()


def resolve_floor_mesh(dataset: Path, mesh_path: Path) -> Path:
    candidate = dataset / "room_floor_mesh.json"
    return candidate.resolve() if candidate.is_file() else mesh_path


def floor_bounds(mesh: Any) -> tuple[float, float, float, float, float]:
    """Estimate a stable XY rectangle and nominal Z for the lowest floor."""

    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths[:, None], 1e-12)
    centers = triangles.mean(axis=1)
    horizontal = (np.abs(normals[:, 2]) >= 0.985) & (lengths > 1e-8)
    if not np.any(horizontal):
        raise RuntimeError("Mesh has no horizontal faces for a floor homography")
    values = centers[horizontal]
    z_values = values[:, 2]
    if float(z_values.min()) <= 0.0 <= float(z_values.max()) and abs(float(np.median(z_values))) < 0.15:
        z_floor = 0.0
    else:
        z_floor = float(np.percentile(z_values, 10.0))
    layer = values[np.abs(values[:, 2] - z_floor) <= 0.08]
    if len(layer) < 4:
        layer = values
    x0, x1 = np.percentile(layer[:, 0], [1.0, 99.0])
    y0, y1 = np.percentile(layer[:, 1], [1.0, 99.0])
    if x1 - x0 < 0.25 or y1 - y0 < 0.25:
        raise RuntimeError("Floor bounds are too small for a stable homography")
    return float(x0), float(x1), float(y0), float(y1), float(z_floor)


def project_world(points_xyz: np.ndarray, calibration: CameraCalibration) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    camera_points = (calibration.R @ points.T + calibration.t).T
    depth = camera_points[:, 2]
    valid = depth > 1e-7
    safe = np.maximum(depth, 1e-7)
    pixels = np.column_stack(
        (
            calibration.K[0, 0] * camera_points[:, 0] / safe + calibration.K[0, 2],
            calibration.K[1, 1] * camera_points[:, 1] / safe + calibration.K[1, 2],
        )
    )
    return pixels, valid


def floor_anchors(
    calibration: CameraCalibration,
    bounds: tuple[float, float, float, float, float],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1, z = bounds
    margin_x = max(0.02 * (x1 - x0), 0.02)
    margin_y = max(0.02 * (y1 - y0), 0.02)
    xs = np.linspace(x0 + margin_x, x1 - margin_x, max(3, int(grid_size)))
    ys = np.linspace(y0 + margin_y, y1 - margin_y, max(3, int(grid_size)))
    floor_xy = np.asarray(list(itertools.product(xs, ys)), dtype=np.float64)
    floor_xyz = np.column_stack((floor_xy, np.full(len(floor_xy), z)))
    pixels, valid_depth = project_world(floor_xyz, calibration)
    width, height = calibration.image_size or (0, 0)
    visible = valid_depth.copy()
    if width > 0 and height > 0:
        visible &= (
            (pixels[:, 0] >= 4.0)
            & (pixels[:, 0] < width - 4.0)
            & (pixels[:, 1] >= 4.0)
            & (pixels[:, 1] < height - 4.0)
        )
    return floor_xy[visible], pixels[visible]


def normalize_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    scale = math.sqrt(2.0) / max(float(distances.mean()), 1e-12)
    transform = np.asarray(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ transform.T
    return homogeneous[:, :2] / homogeneous[:, 2:3], transform


def dlt_homography(floor_xy: np.ndarray, image_xy: np.ndarray) -> np.ndarray:
    source = np.asarray(floor_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(image_xy, dtype=np.float64).reshape(-1, 2)
    if len(source) != len(target) or len(source) < 4:
        raise ValueError("DLT needs at least four matching correspondences")
    source_n, source_transform = normalize_points(source)
    target_n, target_transform = normalize_points(target)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source_n, target_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    normalized = vh[-1].reshape(3, 3)
    homography = np.linalg.inv(target_transform) @ normalized @ source_transform
    if abs(float(homography[2, 2])) > 1e-12:
        homography /= homography[2, 2]
    if not np.all(np.isfinite(homography)) or abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("DLT produced a singular homography")
    return homography


def project_floor(homography: np.ndarray, floor_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(floor_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(homography).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def unproject_floor(homography: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64).reshape(3, 3))
    homogeneous = np.column_stack((values, np.ones(len(values)))) @ inverse.T
    denominator = homogeneous[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Pixel lies on the homography horizon")
    return homogeneous[:, :2] / denominator[:, None]


def analytic_homography(calibration: CameraCalibration, plane_z: float) -> np.ndarray:
    translation = calibration.R[:, 2] * float(plane_z) + calibration.t.reshape(3)
    homography = calibration.K @ np.column_stack(
        (calibration.R[:, 0], calibration.R[:, 1], translation)
    )
    if abs(float(np.linalg.det(homography))) < 1e-12:
        raise ValueError("Analytic homography is singular")
    return homography / homography[2, 2]


def spread_indices(points: np.ndarray, maximum: int = 4) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) <= maximum:
        return np.arange(len(values), dtype=int)
    # Farthest-point sampling alone can select a nearly collinear quadruple
    # when the visible grid is strongly foreshortened.  Four points still fit
    # a projective transform exactly, but the resulting H is then unstable
    # away from those four points.  Build candidates from the four diagonal
    # extrema and choose the largest-area non-degenerate quadrilateral.
    scores = (
        values[:, 0] + values[:, 1],
        values[:, 0] - values[:, 1],
        -values[:, 0] + values[:, 1],
        -values[:, 0] - values[:, 1],
        values[:, 0],
        -values[:, 0],
        values[:, 1],
        -values[:, 1],
    )
    candidates = list(dict.fromkeys(int(np.argmin(score)) for score in scores))
    if len(candidates) < maximum:
        distances = np.full(len(values), np.inf, dtype=np.float64)
        selected = [candidates[0]]
        for _ in range(len(candidates), maximum):
            distances = np.minimum(
                distances,
                np.linalg.norm(values - values[selected[-1]], axis=1),
            )
            distances[selected] = -1.0
            selected.append(int(np.argmax(distances)))
        candidates = list(dict.fromkeys(candidates + selected))

    def quadrilateral_score(indices: tuple[int, ...]) -> float:
        points = values[list(indices)]
        center = points.mean(axis=0)
        order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
        ordered = points[order]
        area = 0.5 * abs(
            np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1))
            - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
        )
        singular_values = np.linalg.svd(points - center, compute_uv=False)
        if len(singular_values) < 2 or singular_values[1] < 1e-8:
            return -1.0
        return float(area * singular_values[1])

    best: Optional[tuple[int, ...]] = None
    best_score = -1.0
    for combination in itertools.combinations(candidates, maximum):
        score = quadrilateral_score(combination)
        if score > best_score:
            best, best_score = combination, score
    if best is not None:
        return np.asarray(best, dtype=int)
    return np.asarray(candidates[:maximum], dtype=int)


def noisy_correspondences(
    floor_xy: np.ndarray,
    image_xy: np.ndarray,
    image_size: tuple[int, int],
    sigma_px: float,
    outlier_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noisy = np.asarray(image_xy, dtype=np.float64).copy()
    noisy += rng.normal(0.0, max(0.0, float(sigma_px)), size=noisy.shape)
    mask = rng.random(len(noisy)) < float(np.clip(outlier_fraction, 0.0, 1.0))
    width, height = image_size
    if np.any(mask):
        noisy[mask] = np.column_stack(
            (
                rng.uniform(0.0, max(1.0, width - 1.0), int(mask.sum())),
                rng.uniform(0.0, max(1.0, height - 1.0), int(mask.sum())),
            )
        )
    return np.asarray(floor_xy), noisy, mask


def ransac_homography(
    floor_xy: np.ndarray,
    image_xy: np.ndarray,
    threshold_px: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import cv2

        cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
        matrix, mask = cv2.findHomography(
            np.asarray(floor_xy, dtype=np.float64),
            np.asarray(image_xy, dtype=np.float64),
            cv2.RANSAC,
            float(threshold_px),
        )
        if matrix is not None:
            inliers = np.ones(len(floor_xy), dtype=bool) if mask is None else np.asarray(mask).reshape(-1).astype(bool)
            return np.asarray(matrix, dtype=np.float64), inliers, "opencv_ransac"
    except (ImportError, ValueError, np.linalg.LinAlgError):
        pass
    return dlt_homography(floor_xy, image_xy), np.ones(len(floor_xy), dtype=bool), "dlt_fallback"


def ray_point(
    calibration: CameraCalibration,
    mesh: Any,
    pixel_value: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], str]:
    if pixel_value is None:
        return None, "no_pixel"
    ideal = calibration.undistort_pixels(np.asarray(pixel_value).reshape(1, 2))[0]
    origin, direction = calibration.geometry().pixel_to_ray(float(ideal[0]), float(ideal[1]))
    hit = intersect_ray_with_grid_result(origin, direction, mesh, max_dist=100.0, step=0.25)
    if not hit.hit:
        return None, hit.status
    return np.asarray(hit.point, dtype=np.float64), hit.status


def homography_point(
    homography: np.ndarray,
    calibration: CameraCalibration,
    pixel_value: Optional[np.ndarray],
    plane_z: float,
) -> tuple[Optional[np.ndarray], str]:
    if pixel_value is None:
        return None, "no_pixel"
    try:
        ideal = calibration.undistort_pixels(np.asarray(pixel_value).reshape(1, 2))
        xy = unproject_floor(homography, ideal)[0]
        return np.asarray([xy[0], xy[1], plane_z], dtype=np.float64), "valid_homography"
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, str(exc)


def source_points(row: dict[str, Any], prediction: Optional[dict[str, Any]]) -> dict[str, Optional[np.ndarray]]:
    coarse = point(row.get("p_fire_noisy_pixel"))
    if prediction is not None:
        prediction_coarse = point(prediction.get("coarse_pixel"))
        if prediction_coarse is not None:
            coarse = prediction_coarse
    roi_blend = point(prediction.get("blend_pixel")) if prediction is not None else None
    return {
        "coarse": coarse,
        "roi_blend": roi_blend,
        "oracle": point(row.get("p_fire_pixel")),
    }


def scene_static_homographies(
    rows: list[dict[str, Any]],
    bounds: tuple[float, float, float, float, float],
    grid_size: int,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scene_id", "unknown"))].append(row)
    result: dict[str, np.ndarray] = {}
    for scene_id, scene_rows in grouped.items():
        calibration = calibration_from_row(scene_rows[0])
        floor_xy, image_xy = floor_anchors(calibration, bounds, grid_size)
        if len(floor_xy) >= 4:
            result[scene_id] = dlt_homography(floor_xy, image_xy)
    return result


def evaluate(
    rows: list[dict[str, Any]],
    mesh: Any,
    floor_mesh: Any,
    bounds: tuple[float, float, float, float, float],
    prediction_index: dict[str, dict[str, Any]],
    grid_size: int,
    anchor_noise_px: float,
    outlier_fraction: float,
    ransac_threshold_px: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, CameraCalibration]]:
    static_h = scene_static_homographies(rows, bounds, grid_size)
    processed: list[dict[str, Any]] = []
    calibrations: dict[str, CameraCalibration] = {}
    reprojection: dict[str, list[float]] = defaultdict(list)
    ransac_backends: dict[str, int] = defaultdict(int)

    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", Path(str(row.get("image_path", ""))).stem))
        calibration = calibration_from_row(row)
        calibrations[sample_id] = calibration
        floor_xy, image_xy = floor_anchors(calibration, bounds, grid_size)
        if len(floor_xy) < 4:
            raise RuntimeError(f"Only {len(floor_xy)} visible floor anchors for {sample_id}")
        selected = spread_indices(image_xy, 4)
        h_analytic = analytic_homography(calibration, bounds[4])
        h_dlt4 = dlt_homography(floor_xy[selected], image_xy[selected])
        h_dlt_all = dlt_homography(floor_xy, image_xy)
        noisy_floor, noisy_pixels, outlier_mask = noisy_correspondences(
            floor_xy,
            image_xy,
            tuple(int(value) for value in row.get("image_size", [640, 640])),
            anchor_noise_px,
            outlier_fraction,
            seed + index * 7919,
        )
        h_ransac, inlier_mask, backend = ransac_homography(
            noisy_floor, noisy_pixels, ransac_threshold_px, seed + index * 7919
        )
        ransac_backends[backend] += 1
        homographies = {
            "analytic": h_analytic,
            "dlt4": h_dlt4,
            "dlt_all": h_dlt_all,
            "ransac_noisy": h_ransac,
        }
        scene_id = str(row.get("scene_id", "unknown"))
        if scene_id in static_h:
            homographies["scene_static"] = static_h[scene_id]

        prediction = prediction_index.get(sample_id)
        sources = source_points(row, prediction)
        detail: dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "frame_index": int(row.get("frame_index", index)),
            "image_path": row.get("_image_path"),
            "surface": row.get("fire_surface", row.get("surface_type")),
            "is_floor": str(row.get("fire_surface", row.get("surface_type", ""))).lower()
            in {"floor", "ground", "floor_plane"},
            "gt_pixel": point(row.get("p_fire_pixel")),
            "gt_xyz": point(row.get("fire_xyz_world"), 3),
            "sources": sources,
            "anchor_count": int(len(floor_xy)),
            "anchor_outliers": int(outlier_mask.sum()),
            "ransac_inliers": int(inlier_mask.sum()),
            "methods": {},
        }
        for source_name, source_pixel in sources.items():
            for method, homography in homographies.items():
                started = time.perf_counter()
                estimate, status = homography_point(homography, calibration, source_pixel, bounds[4])
                detail["methods"][f"{source_name}:{method}"] = {
                    "point": estimate,
                    "status": status,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            started = time.perf_counter()
            estimate, status = ray_point(calibration, mesh, source_pixel)
            detail["methods"][f"{source_name}:ray"] = {
                "point": estimate,
                "status": status,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
            started = time.perf_counter()
            estimate, status = ray_point(calibration, floor_mesh, source_pixel)
            detail["methods"][f"{source_name}:ray_floor"] = {
                "point": estimate,
                "status": status,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        detail["anchors"] = {
            "image_pixels": image_xy,
            "dlt4_pixels": image_xy[selected],
            "noisy_pixels": noisy_pixels,
        }
        for method, homography in homographies.items():
            residual = np.linalg.norm(project_floor(homography, floor_xy) - image_xy, axis=1)
            reprojection[method].extend(residual.tolist())
        processed.append(detail)

    metrics: dict[str, Any] = {}
    for source_name in SOURCES:
        for method in METHODS:
            key = f"{source_name}:{method}"
            metrics[key] = {
                "floor_only": metric_for(processed, key, floor_only=True),
                "all_surfaces": metric_for(processed, key, floor_only=False),
            }
    diagnostics = {
        "records": len(processed),
        "scenes": len({row["scene_id"] for row in processed}),
        "floor_records": int(sum(bool(row["is_floor"]) for row in processed)),
        "non_floor_records": int(sum(not row["is_floor"] for row in processed)),
        "anchor_count_mean": float(np.mean([row["anchor_count"] for row in processed])),
        "ransac_inlier_mean": float(np.mean([row["ransac_inliers"] for row in processed])),
        "ransac_backends": dict(ransac_backends),
        "anchor_reprojection_px": {
            method: {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}
            for method, values in reprojection.items()
            if values
        },
        "metrics": metrics,
    }
    return processed, diagnostics, calibrations


def metric_for(rows: list[dict[str, Any]], key: str, floor_only: bool) -> dict[str, Any]:
    selected = [row for row in rows if not floor_only or row["is_floor"]]
    errors: list[float] = []
    deltas: list[np.ndarray] = []
    latencies: list[float] = []
    for row in selected:
        result = row["methods"].get(key, {})
        estimate = point(result.get("point"), 3)
        truth = point(row.get("gt_xyz"), 3)
        if estimate is not None and truth is not None:
            delta = estimate - truth
            deltas.append(delta)
            errors.append(float(np.linalg.norm(delta)))
        if result.get("latency_ms") is not None:
            latencies.append(float(result["latency_ms"]))
    values = np.asarray(errors, dtype=np.float64)
    xyz = np.asarray(deltas, dtype=np.float64).reshape(-1, 3) if deltas else np.empty((0, 3))
    return {
        "records": len(selected),
        "valid": int(len(values)),
        "success_rate": float(len(values) / max(1, len(selected))),
        "mae_m": None if not len(values) else float(values.mean()),
        "median_m": None if not len(values) else float(np.median(values)),
        "p95_m": None if not len(values) else float(np.percentile(values, 95)),
        "under_0.10m": None if not len(values) else float(np.mean(values <= 0.10)),
        "under_0.25m": None if not len(values) else float(np.mean(values <= 0.25)),
        "under_0.50m": None if not len(values) else float(np.mean(values <= 0.50)),
        "under_1.00m": None if not len(values) else float(np.mean(values <= 1.00)),
        "xyz_mae_m": None if not len(xyz) else np.mean(np.abs(xyz), axis=0),
        "mean_latency_ms": None if not latencies else float(np.mean(latencies)),
        "p95_latency_ms": None if not latencies else float(np.percentile(latencies, 95)),
    }


def project_estimate_to_pixel(value: Any, calibration: CameraCalibration) -> Optional[np.ndarray]:
    xyz = point(value, 3)
    if xyz is None:
        return None
    pixels, valid = project_world(xyz.reshape(1, 3), calibration)
    return pixels[0] if bool(valid[0]) else None


def draw_marker(draw: ImageDraw.ImageDraw, value: Any, color: tuple[int, int, int], label: str) -> None:
    pixel_value = point(value)
    if pixel_value is None:
        return
    x, y = float(pixel_value[0]), float(pixel_value[1])
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    draw.text((x + radius + 2, y - radius - 2), label, fill=color)


def contact_sheet(
    rows: list[dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    output: Path,
    maximum: int,
) -> None:
    tiles: list[Image.Image] = []
    selected = rows if maximum <= 0 else rows[:maximum]
    for row in selected:
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        draw_marker(draw, row.get("gt_pixel"), COLORS["gt"], "GT")
        source = row["sources"].get("roi_blend")
        if source is None:
            source = row["sources"].get("coarse")
        draw_marker(draw, source, (220, 220, 220), "input")
        calibration = calibrations[row["sample_id"]]
        source_name = "roi_blend" if row["sources"].get("roi_blend") is not None else "coarse"
        for method in ("analytic", "dlt_all", "ransac_noisy", "ray_floor"):
            value = row["methods"].get(f"{source_name}:{method}", {}).get("point")
            draw_marker(draw, project_estimate_to_pixel(value, calibration), COLORS[method], method)
        tiles.append(ImageOps.contain(image, (420, 320)))
    if not tiles:
        return
    columns = 4
    sheet = Image.new("RGB", (columns * 420, math.ceil(len(tiles) / columns) * 320), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * 420 + (420 - tile.width) // 2
        y = (index // columns) * 320 + (320 - tile.height) // 2
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def static_plots(
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    mesh: Any,
    output_dir: Path,
    maximum: int,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return outputs

    names = ("analytic", "dlt4", "dlt_all", "ransac_noisy", "scene_static", "ray", "ray_floor")
    metrics = diagnostics["metrics"]
    values = []
    for name in names:
        metric = metrics.get(f"roi_blend:{name}", metrics[f"coarse:{name}"])["floor_only"]
        values.append(np.nan if metric["mae_m"] is None else metric["mae_m"])
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=150)
    axis.bar(list(names), values, color=[np.asarray(COLORS[name]) / 255.0 for name in names])
    axis.set_ylabel("3D MAE (m)")
    axis.set_title("Floor homography/IPM versus calibrated ray casting")
    axis.tick_params(axis="x", rotation=22)
    figure.tight_layout()
    comparison = output_dir / "homography_error_comparison.png"
    figure.savefig(comparison, bbox_inches="tight")
    plt.close(figure)
    outputs["error_comparison_png"] = str(comparison)

    selected = rows if maximum <= 0 else rows[:maximum]
    figure, axis = plt.subplots(figsize=(9, 7), dpi=150)
    for row in selected:
        if not row["is_floor"]:
            continue
        truth = point(row["gt_xyz"], 3)
        if truth is not None:
            axis.scatter(truth[0], truth[1], c=[np.asarray(COLORS["gt"]) / 255.0], marker="o", s=28)
        for name in ("analytic", "dlt_all", "ransac_noisy", "ray_floor"):
            source_name = "roi_blend" if row["sources"].get("roi_blend") is not None else "coarse"
            value = point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                axis.scatter(value[0], value[1], c=[np.asarray(COLORS[name]) / 255.0], marker="x", s=24)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title("Top view of floor estimates")
    axis.axis("equal")
    figure.tight_layout()
    floor_map = output_dir / "homography_floor_map.png"
    figure.savefig(floor_map, bbox_inches="tight")
    plt.close(figure)
    outputs["floor_map_png"] = str(floor_map)

    figure = plt.figure(figsize=(11, 8), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    triangles = mesh.vertices[mesh.faces]
    axis.add_collection3d(
        Poly3DCollection(triangles, alpha=0.12, facecolor="#8c8c8c", edgecolor="#777777", linewidth=0.25)
    )
    for row in selected:
        truth = point(row["gt_xyz"], 3)
        if truth is not None:
            axis.scatter(*truth, c=[np.asarray(COLORS["gt"]) / 255.0], marker="o", s=28)
        for name in ("analytic", "dlt_all", "ransac_noisy", "ray_floor"):
            source_name = "roi_blend" if row["sources"].get("roi_blend") is not None else "coarse"
            value = point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                axis.scatter(*value, c=[np.asarray(COLORS[name]) / 255.0], marker="x", s=30)
    low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title("Homography/IPM 3D estimates on the room mesh")
    figure.tight_layout()
    overview = output_dir / "homography_3d_overview.png"
    figure.savefig(overview, bbox_inches="tight")
    plt.close(figure)
    outputs["overview_3d_png"] = str(overview)

    first = next((row for row in rows if row.get("anchors", {}).get("image_pixels") is not None), None)
    if first is not None:
        try:
            with Image.open(first["image_path"]) as opened:
                anchor_image = opened.convert("RGB").copy()
            draw = ImageDraw.Draw(anchor_image)
            for value in first["anchors"]["image_pixels"]:
                x, y = (float(v) for v in value)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(70, 170, 240))
            for value in first["anchors"]["dlt4_pixels"]:
                x, y = (float(v) for v in value)
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 70, 40), width=2)
            anchor_path = output_dir / "floor_anchor_layout.png"
            anchor_image.save(anchor_path)
            outputs["floor_anchor_layout_png"] = str(anchor_path)
        except (OSError, KeyError, TypeError):
            pass
    return outputs


def write_ply(rows: list[dict[str, Any]], mesh: Any, output: Path, maximum: int) -> None:
    selected = rows if maximum <= 0 else rows[:maximum]
    points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for row in selected:
        truth = point(row["gt_xyz"], 3)
        if truth is not None:
            points.append((truth, COLORS["gt"]))
        for name in ("analytic", "dlt4", "dlt_all", "ransac_noisy", "scene_static", "ray", "ray_floor"):
            source_name = "roi_blend" if row["sources"].get("roi_blend") is not None else "coarse"
            value = point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                points.append((value, COLORS[name]))
    vertices, faces = mesh.vertices, mesh.faces
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices) + len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} 150 150 150" for v in vertices)
    lines.extend(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]} {c[1]} {c[2]}" for p, c in points)
    lines.extend(f"3 {int(f[0])} {int(f[1])} {int(f[2])}" for f in faces)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_html(rows: list[dict[str, Any]], mesh: Any, output: Path, maximum: int) -> None:
    selected = rows if maximum <= 0 else rows[:maximum]
    traces: list[dict[str, Any]] = [
        {
            "type": "mesh3d",
            "x": mesh.vertices[:, 0].tolist(),
            "y": mesh.vertices[:, 1].tolist(),
            "z": mesh.vertices[:, 2].tolist(),
            "i": mesh.faces[:, 0].tolist(),
            "j": mesh.faces[:, 1].tolist(),
            "k": mesh.faces[:, 2].tolist(),
            "name": "Room mesh",
            "opacity": 0.22,
            "color": "#8c8c8c",
        }
    ]
    for name in ("gt", "analytic", "dlt_all", "ransac_noisy", "ray_floor"):
        values: list[np.ndarray] = []
        labels: list[str] = []
        for row in selected:
            if name == "gt":
                value = point(row["gt_xyz"], 3)
            else:
                source_name = "roi_blend" if row["sources"].get("roi_blend") is not None else "coarse"
                value = point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                values.append(value)
                labels.append(str(row["sample_id"]))
        if not values:
            continue
        array = np.asarray(values)
        color = COLORS["gt"] if name == "gt" else COLORS[name]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": array[:, 0].tolist(),
                "y": array[:, 1].tolist(),
                "z": array[:, 2].tolist(),
                "text": labels,
                "name": name,
                "marker": {"size": 5, "color": "rgb(%d,%d,%d)" % color},
            }
        )
    payload = json.dumps(traces, ensure_ascii=False)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Homography 3D benchmark</title>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>"
        "</head><body><div id='scene' style='width:100%;height:95vh'></div>"
        "<script>const traces="
        + payload
        + ";Plotly.newPlot('scene',traces,{title:'Homography/IPM 3D benchmark',"
        "scene:{aspectmode:'data',xaxis:{title:'X (m)'},yaxis:{title:'Y (m)'},"
        "zaxis:{title:'Z (m)'}},margin:{l:0,r:0,b:0,t:45}});</script>"
        "</body></html>"
    )
    output.write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(load_rows(dataset, args.split), args.max_records, args.selection)
    mesh_path = resolve_mesh(dataset, args.mesh)
    floor_path = resolve_floor_mesh(dataset, mesh_path)
    mesh = load_triangle_mesh(mesh_path)
    floor_mesh = load_triangle_mesh(floor_path)
    bounds = floor_bounds(floor_mesh)
    prediction_path = args.prediction_summary.expanduser().resolve() if args.prediction_summary else None
    prediction_index = load_prediction_index(prediction_path)
    processed, diagnostics, calibrations = evaluate(
        rows,
        mesh,
        floor_mesh,
        bounds,
        prediction_index,
        args.grid_size,
        args.anchor_noise_px,
        args.outlier_fraction,
        args.ransac_threshold_px,
        args.seed,
    )
    contact_sheet(processed, calibrations, output_dir / "homography_contact_sheet.jpg", args.max_contact_images)
    visuals = static_plots(processed, diagnostics, mesh, output_dir, args.max_visual_records)
    visuals["contact_sheet"] = str(output_dir / "homography_contact_sheet.jpg")
    ply_path = output_dir / "homography_annotations.ply"
    write_ply(processed, floor_mesh, ply_path, args.max_visual_records)
    visuals["ply"] = str(ply_path)
    html_path = output_dir / "homography_3d_interactive.html"
    write_html(processed, floor_mesh, html_path, args.max_visual_records)
    visuals["interactive_html"] = str(html_path)
    summary = {
        "format": "LAB_SAM.benchmark_homography.v1",
        "dataset": str(dataset),
        "mesh": str(mesh_path),
        "floor_mesh": str(floor_path),
        "split": args.split,
        "records": len(processed),
        "prediction_summary": None if prediction_path is None else str(prediction_path),
        "plane": {"nominal_z_m": bounds[4], "bounds_xy": list(bounds[:4])},
        "protocol": {
            "analytic": "closed-form H from camera calibration",
            "dlt4": "four spread-out floor correspondences",
            "dlt_all": "normalised DLT from visible floor correspondences",
            "ransac_noisy": {
                "anchor_noise_px": args.anchor_noise_px,
                "outlier_fraction": args.outlier_fraction,
                "threshold_px": args.ransac_threshold_px,
            },
            "scene_static": "one DLT H from first frame of each scene",
            "ray": "single-ray intersection with full mesh, including obstacles",
            "ray_floor": "single-ray intersection with the floor-only mesh",
            "warning": "Homography is physically valid only on the nominated floor plane.",
        },
        "diagnostics": diagnostics,
        "visual_outputs": visuals,
        "records_detail": processed,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"dataset={dataset}")
    print(
        f"records={len(processed)} floor={diagnostics['floor_records']} "
        f"non_floor={diagnostics['non_floor_records']}"
    )
    print("source/method                 floor_MAE(m)  median(m)  P95(m)  success")
    for source_name in SOURCES:
        for method in METHODS:
            metric = diagnostics["metrics"][f"{source_name}:{method}"]["floor_only"]
            print(
                f"{source_name + ':' + method:28} {str(metric['mae_m']):>11} "
                f"{str(metric['median_m']):>10} {str(metric['p95_m']):>8} "
                f"{metric['success_rate']:.3f}"
            )
    print(f"saved_summary={output_dir / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--prediction-summary", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--selection", choices=("head", "even"), default="even")
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--anchor-noise-px", type=float, default=1.5)
    parser.add_argument("--outlier-fraction", type=float, default=0.15)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--max-visual-records", type=int, default=40)
    parser.add_argument("--max-contact-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261007)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_records < 0 or args.grid_size < 3 or args.max_visual_records < 0 or args.max_contact_images < 0:
        raise ValueError("max values must be non-negative and grid-size must be >= 3")
    if args.anchor_noise_px < 0 or not 0.0 <= args.outlier_fraction <= 1.0 or args.ransac_threshold_px <= 0:
        raise ValueError("Invalid noise or RANSAC parameters")
    run(args)


if __name__ == "__main__":
    main()

