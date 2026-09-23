"""Synthetic regression test for pixel-noise to 3D localisation."""
import argparse, time
import numpy as np
from locator import CameraGeometry, GridMap, intersect_ray_with_grid_result

def look_at(camera_pos, target):
    forward = np.asarray(target) - np.asarray(camera_pos); forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-9: right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right); up = np.cross(right, forward)
    return np.vstack([right, up, forward])

def run(n=500, pixel_noise=2.0, seed=42):
    rng = np.random.default_rng(seed); grid = GridMap(); camera_pos = np.array([0.0, 0.0, 20.0]); K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    errors, latencies, misses = [], [], 0
    for _ in range(n):
        x, y = rng.uniform(-15., 15., size=2); gt = np.array([x, y, grid.get_elevation(x, y)]); R = look_at(camera_pos, gt); camera = CameraGeometry(K, R, -R @ camera_pos.reshape(3, 1))
        pixel_camera = R @ (gt - camera_pos); pixel = K @ pixel_camera; pixel = pixel[:2] / pixel[2] + rng.uniform(-pixel_noise, pixel_noise, size=2); C, ray = camera.pixel_to_ray(*pixel)
        t0 = time.perf_counter(); result = intersect_ray_with_grid_result(C, ray, grid, step=1.0); latencies.append((time.perf_counter()-t0)*1000)
        if not result.hit: misses += 1
        else: errors.append(float(np.linalg.norm(result.point - gt)))
    print(f"runs={n} hits={n-misses} misses={misses} noise_px={pixel_noise}")
    if errors: print(f"error_mean_m={np.mean(errors):.4f} error_p95_m={np.percentile(errors,95):.4f}")
    print(f"latency_mean_ms={np.mean(latencies):.4f} latency_p95_ms={np.percentile(latencies,95):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", type=int, default=500); parser.add_argument("--noise", type=float, default=2.0); args = parser.parse_args(); run(args.runs, args.noise)
