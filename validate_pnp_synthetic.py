"""Validate the room-camera PnP implementation on a fixed synthetic pose.

This test creates measured-like room anchors, projects them through one known
camera, adds optional image noise, then recovers the camera with OpenCV
``solvePnP``.  It validates the geometry and JSON contract only; the result is
never a real-room accuracy claim.

Example::

    .venv\\Scripts\\python.exe validate_pnp_synthetic.py ^
      --output-dir working\\pnp_synthetic_validation ^
      --pixel-noise 0.25

The output contains a correspondence JSON that can also be passed to
``calibrate_room_pose_pnp.py`` with ``--status synthetic_validation``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _look_at(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build world-to-camera R with x=right, y=down, z=forward."""
    camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return np.vstack([right, down, forward])


def _project(
    points: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    camera_position: np.ndarray,
) -> np.ndarray:
    camera_points = (R @ (points - camera_position).T).T
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("Synthetic anchor is behind the camera")
    return np.column_stack(
        [
            K[0, 0] * camera_points[:, 0] / camera_points[:, 2] + K[0, 2],
            K[1, 1] * camera_points[:, 1] / camera_points[:, 2] + K[1, 2],
        ]
    )


def _rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    delta = R_est @ R_true.T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.pixel_noise < 0:
        raise ValueError("--pixel-noise must be non-negative")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "This validation needs opencv-python. Install it in the project venv."
        ) from exc

    rng = np.random.default_rng(args.seed)
    image_size = [1920, 1080]
    K = np.array(
        [[1380.0, 0.0, 960.0], [0.0, 1370.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    camera_position = np.array([0.35, -4.2, 2.45], dtype=np.float64)
    R_true = _look_at(camera_position, np.array([0.0, 2.2, 1.0], dtype=np.float64))

    # Floor points plus points at different heights make the pose observable;
    # they mimic a set of room markers rather than fire labels.
    object_points = np.array(
        [
            [-2.2, 1.0, 0.0],
            [-0.8, 1.0, 0.0],
            [0.8, 1.0, 0.0],
            [2.2, 1.0, 0.0],
            [-2.2, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.2, 3.0, 0.0],
            [-1.6, 5.0, 0.0],
            [1.6, 5.0, 0.0],
            [-2.0, 2.0, 1.6],
            [2.0, 2.0, 1.6],
            [0.0, 4.5, 2.2],
        ],
        dtype=np.float64,
    )
    image_points_clean = _project(object_points, K, R_true, camera_position)
    image_points = image_points_clean + rng.normal(
        0.0, float(args.pixel_noise), size=image_points_clean.shape
    )
    rvec_true, _ = cv2.Rodrigues(R_true)
    tvec_true = (-R_true @ camera_position).reshape(3, 1)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed on the synthetic fixture")
    R_est, _ = cv2.Rodrigues(rvec)
    camera_est = (-R_est.T @ tvec.reshape(3, 1)).reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reprojection = np.linalg.norm(
        projected.reshape(-1, 2) - image_points, axis=1
    )
    camera_error = float(np.linalg.norm(camera_est - camera_position))
    rotation_error = _rotation_error_deg(R_est, R_true)
    t_error = float(np.linalg.norm(tvec.reshape(3) - tvec_true.reshape(3)))

    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    correspondence = {
        "metadata": {
            "status": "synthetic_validation",
            "units": "metres",
            "coordinate_system": "synthetic room frame",
            "camera_id": "synthetic_fixed_camera",
            "warning": "This fixture validates PnP only; it is not measured-room data.",
        },
        "image_size": image_size,
        "intrinsics": {"K": K.tolist(), "dist_coeffs": dist.tolist()},
        "object_points_world": object_points.tolist(),
        "image_points": image_points.tolist(),
        "records": [
            {
                "anchor_id": f"anchor_{index:02d}",
                "object_point_world": point.tolist(),
                "image_point": pixel.tolist(),
            }
            for index, (point, pixel) in enumerate(zip(object_points, image_points))
        ],
    }
    result = {
        "metadata": correspondence["metadata"],
        "seed": int(args.seed),
        "pixel_noise_sigma": float(args.pixel_noise),
        "correspondence_count": int(len(object_points)),
        "ground_truth": {
            "K": K,
            "R_world_to_camera": R_true,
            "t": tvec_true,
            "camera_position": camera_position,
        },
        "estimated": {
            "R_world_to_camera": R_est,
            "t": tvec,
            "camera_position": camera_est,
        },
        "errors": {
            "reprojection_mean_px": float(reprojection.mean()),
            "reprojection_median_px": float(np.median(reprojection)),
            "reprojection_p95_px": float(np.percentile(reprojection, 95)),
            "camera_position_error_m": camera_error,
            "rotation_error_deg": rotation_error,
            "translation_vector_error_m": t_error,
        },
    }
    output_dir = args.output_dir.expanduser().resolve()
    (output_dir / "correspondences.json").write_text(
        json.dumps(correspondence, indent=2, default=_json_default), encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(result["errors"], indent=2))
    print(f"saved={output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
