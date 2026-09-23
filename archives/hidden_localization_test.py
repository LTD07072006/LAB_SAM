"""Hidden-ground-truth test for monocular ray-to-GridMap localisation.

The localisation function receives only a pixel. The sampled 3D point is
used after localisation exclusively to calculate the error statistics.
"""
import argparse
from pathlib import Path

import numpy as np

from locator import CameraGeometry, GridMap, intersect_ray_with_grid_result


def make_test_camera():
    """Match the default camera used by main_integrated.py."""
    K = np.array([[800.0, 0.0, 640.0],
                  [0.0, 800.0, 360.0],
                  [0.0, 0.0, 1.0]])
    camera_pos = np.array([0.0, -25.0, 30.0])
    target = np.array([0.0, 0.0, 0.0])
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    return CameraGeometry(K, R, -R @ camera_pos.reshape(3, 1)), camera_pos


def project_world_point(camera: CameraGeometry, point):
    """Project a world point to a pixel, returning None if behind camera."""
    point = np.asarray(point, dtype=np.float64).reshape(3)
    point_camera = camera.R @ point + camera.t.ravel()
    if point_camera[2] <= 1e-9:
        return None
    pixel = camera.K @ point_camera
    return pixel[:2] / pixel[2]


def run(samples=1000, seed=20260921, pixel_noise=0.0,
        xy_min=-15.0, xy_max=15.0, image_width=1280, image_height=720):
    rng = np.random.default_rng(seed)
    grid = GridMap()
    camera, camera_pos = make_test_camera()

    errors = []
    distances = []
    records = []
    rejected_projection = 0
    misses = 0

    for sample_id in range(samples):
        # Hidden ground truth: not passed to the localisation module.
        x, y = rng.uniform(xy_min, xy_max, size=2)
        ground_truth = np.array([x, y, grid.get_elevation(x, y)], dtype=np.float64)
        pixel = project_world_point(camera, ground_truth)
        if pixel is None or not (0 <= pixel[0] < image_width and 0 <= pixel[1] < image_height):
            rejected_projection += 1
            continue

        observed_pixel = pixel + rng.normal(0.0, pixel_noise, size=2)

        # The module under test receives only camera geometry, observed pixel,
        # and the map. It never receives ground_truth.
        cam_origin, ray = camera.pixel_to_ray(*observed_pixel)
        result = intersect_ray_with_grid_result(
            cam_origin,
            ray,
            grid,
            max_dist=1000.0,
            step=1.0,
            bisection_iterations=24,
        )

        if not result.hit:
            misses += 1
            continue

        error = float(np.linalg.norm(result.point - ground_truth))
        errors.append(error)
        distances.append(float(result.distance))
        if len(records) < 5:
            records.append((sample_id, ground_truth.copy(), pixel.copy(), result.point.copy(), error))

    print("hidden_ground_truth_test")
    print(f"requested_samples={samples}")
    print(f"evaluated_samples={len(errors)}")
    print(f"rejected_outside_image={rejected_projection}")
    print(f"localisation_misses={misses}")
    print(f"camera_position={np.array2string(camera_pos, precision=4)}")
    print(f"pixel_noise_std={pixel_noise:.3f}")

    if errors:
        values = np.asarray(errors)
        print(f"hit_rate={len(errors) / max(1, samples - rejected_projection):.4f}")
        print(f"error_mean_m={values.mean():.6f}")
        print(f"error_median_m={np.median(values):.6f}")
        print(f"error_p95_m={np.percentile(values, 95):.6f}")
        print(f"error_max_m={values.max():.6f}")
        print(f"range_mean_m={np.mean(distances):.6f}")
        print("sample_checks:")
        for sample_id, gt, px, estimate, error in records:
            print(
                f"  id={sample_id} pixel=({px[0]:.3f},{px[1]:.3f}) "
                f"gt=({gt[0]:.4f},{gt[1]:.4f},{gt[2]:.4f}) "
                f"estimate=({estimate[0]:.4f},{estimate[1]:.4f},{estimate[2]:.4f}) "
                f"error_m={error:.6f}"
            )
    else:
        print("No valid localisation results to score.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--pixel-noise", type=float, default=0.0,
                        help="Gaussian pixel noise standard deviation")
    args = parser.parse_args()
    run(args.samples, args.seed, args.pixel_noise)


if __name__ == "__main__":
    main()
