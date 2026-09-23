"""End-to-end fire detection, localisation and evacuation prototype."""
import argparse
from pathlib import Path
import numpy as np
from config import DEFAULT_CONFIG
from fire_detector import FireDetector
from locator import GridMap, intersect_ray_with_grid_result
from point_filter import robust_filter_points
from temporal_filter import DetectionSmoother
from localization import bottom_contact_pixels, localize_pixels, localize_with_uncertainty
from tracking_3d import Fire3DTracker

def make_default_camera():
    K = [[800, 0, 640], [0, 800, 360], [0, 0, 1]]
    camera_pos = np.array([0.0, -25.0, 30.0]); target = np.array([0.0, 0.0, 0.0])
    forward = (target - camera_pos) / np.linalg.norm(target - camera_pos)
    right = np.cross(forward, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(forward, right); R = np.vstack([right, down, forward])
    from locator import CameraGeometry
    return CameraGeometry(K, R, -R @ camera_pos.reshape(3, 1)), camera_pos

def simulate_evacuation_path(start_pos, fire_pos, grid_map, exit_xy=(25.0, 25.0), safe_radius=10.0, max_steps=80, step_size=3.0):
    current = np.asarray(start_pos, dtype=np.float64).copy(); fire_xy = np.asarray(fire_pos)[:2]
    target = np.array([exit_xy[0], exit_xy[1], grid_map.get_elevation(*exit_xy)]); path = [current.copy()]
    for _ in range(max_steps):
        to_exit = target[:2] - current[:2]; distance = np.linalg.norm(to_exit)
        if distance < step_size: break
        direction = to_exit / max(distance, 1e-9); delta = current[:2] - fire_xy; fire_distance = np.linalg.norm(delta)
        if fire_distance < safe_radius:
            direction += 2.0 * (safe_radius - fire_distance) / safe_radius * delta / max(fire_distance, 1e-6)
            direction /= max(np.linalg.norm(direction), 1e-9)
        current[:2] += step_size * direction; current[2] = grid_map.get_elevation(*current[:2]) + 1.0; path.append(current.copy())
    path.append(target); return np.asarray(path)

def process_frame(detector, camera, grid_map, smoother, frame, tracker=None):
    detection = detector.detect(frame, warmup=True); state = smoother.update(detection.pixel, detection.confidence)
    if not state.confirmed or state.pixel is None: return {"detection": detection, "temporal": state, "location": None}
    # A point-only detector has no bbox/mask, so use a small horizontal
    # neighbourhood. For a future bbox/mask detector, pass its bottom pixels
    # directly to localize_pixels/localize_with_uncertainty.
    px = np.asarray(state.pixel, dtype=np.float64)
    half_span = max(2.0, min(detection.size) * 0.015)
    pixels = np.column_stack([
        np.linspace(px[0] - half_span, px[0] + half_span, DEFAULT_CONFIG.multi_ray_columns),
        np.full(DEFAULT_CONFIG.multi_ray_columns, px[1]),
    ])
    weights = np.linspace(0.8, 1.0, len(pixels))
    if DEFAULT_CONFIG.use_3d_uncertainty:
        location = localize_with_uncertainty(
            camera, grid_map, state.pixel,
            pixel_sigma=DEFAULT_CONFIG.uncertainty_pixel_sigma,
            samples=DEFAULT_CONFIG.uncertainty_samples,
            max_dist=DEFAULT_CONFIG.ray_max_distance,
            step=DEFAULT_CONFIG.ray_coarse_step,
            bisection_iterations=DEFAULT_CONFIG.ray_bisection_iterations,
        )
    else:
        location = localize_pixels(
            camera, grid_map, pixels, weights=weights,
            max_dist=DEFAULT_CONFIG.ray_max_distance,
            step=DEFAULT_CONFIG.ray_coarse_step,
            bisection_iterations=DEFAULT_CONFIG.ray_bisection_iterations,
        )
    if tracker is not None and location.hit:
        track = tracker.update(location.point, location.covariance, location.confidence)
        if track.accepted and track.point is not None:
            location.point = track.point
            location.covariance = track.covariance
            location.std = np.sqrt(np.maximum(np.diag(track.covariance), 0.0))
            location.confidence = max(location.confidence, track.confidence)
            location.status = "valid_tracked"
        else:
            location.status = "rejected_temporal_outlier"
    return {"detection": detection, "temporal": state, "location": location}

def run_samples(sample_dir: Path, model_path: Path):
    detector = FireDetector(model_path, threshold=DEFAULT_CONFIG.confidence_threshold); camera, camera_pos = make_default_camera(); grid_map = GridMap()
    smoother = DetectionSmoother(DEFAULT_CONFIG.temporal_alpha, DEFAULT_CONFIG.temporal_window, DEFAULT_CONFIG.temporal_min_hits, threshold=DEFAULT_CONFIG.confidence_threshold)
    tracker = Fire3DTracker(DEFAULT_CONFIG.tracker_alpha, DEFAULT_CONFIG.tracker_gate_m, DEFAULT_CONFIG.tracker_max_missed)
    paths = sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}); points = []
    for path in paths:
        result = process_frame(detector, camera, grid_map, smoother, path, tracker); detection = result["detection"]; location = result["location"]
        if location and location.hit: points.append(location.point)
        uncertainty = "" if not location or location.std is None else f" std={np.array2string(location.std, precision=2)}"
        print(f"{path.name:16} p={detection.confidence:.3f} confirmed={result['temporal'].confirmed} status={location.status if location else 'not_confirmed'}{uncertainty}")
    filtered = robust_filter_points(points); print(f"raw_points={len(points)} kept={len(filtered.points)} center={filtered.center} confidence={filtered.confidence:.3f}")
    if filtered.center is not None:
        start = np.array([-20.0, -10.0, grid_map.get_elevation(-20.0, -10.0) + 1.0]); print(f"evacuation_waypoints={len(simulate_evacuation_path(start, filtered.center, grid_map))} camera={camera_pos}")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--samples", type=Path, default=DEFAULT_CONFIG.sample_dir); parser.add_argument("--model", type=Path, default=DEFAULT_CONFIG.model_path); args = parser.parse_args(); run_samples(args.samples, args.model)

if __name__ == "__main__": main()
