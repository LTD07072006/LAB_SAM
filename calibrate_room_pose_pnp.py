"""Estimate a calibrated camera pose from measured room-marker pairs.

Intrinsics should first be obtained with a checkerboard using Zhang's method.
The input then contains measured room-frame 3D marker points and their image
pixels from one frame. The output follows this repository convention:
``X_camera = R @ X_world + t``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_intrinsics(value: object, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Read K/distortion from a path or an already-loaded JSON object."""
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    else:
        data = value
    if not isinstance(data, dict):
        raise ValueError(f"Intrinsics must be a JSON object: {source}")
    nested = data.get("intrinsics", data)
    if not isinstance(nested, dict) or "K" not in nested:
        raise ValueError(
            f"Intrinsics must contain K or intrinsics.K: {source}"
        )
    try:
        K = np.asarray(nested["K"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(nested.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid K/dist_coeffs in {source}") from exc
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"K contains invalid focal lengths: {source}")
    if len(dist) not in (0, 4, 5, 8, 12, 14):
        raise ValueError(
            "dist_coeffs must have 0, 4, 5, 8, 12 or 14 values "
            f"(got {len(dist)}): {source}"
        )
    return K, dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="Optional checkerboard JSON; overrides correspondences.intrinsics",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status",
        choices=("measured_real_room", "synthetic_validation"),
        default="measured_real_room",
        help="Metadata only; use synthetic_validation for generated test pairs",
    )
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python to run solvePnP calibration") from exc

    correspondences_path = args.correspondences.expanduser().resolve()
    data = json.loads(correspondences_path.read_text(encoding="utf-8"))
    object_points = np.asarray(data["object_points_world"], dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(data["image_points"], dtype=np.float64).reshape(-1, 2)
    if len(object_points) != len(image_points) or len(object_points) < 4:
        raise ValueError("Need at least four matching object_points_world/image_points")
    if args.intrinsics is not None:
        K, dist = _load_intrinsics(args.intrinsics, str(args.intrinsics))
        intrinsics_source = str(args.intrinsics.expanduser().resolve())
    elif "intrinsics" in data:
        K, dist = _load_intrinsics(data["intrinsics"], "correspondences.intrinsics")
        intrinsics_source = "correspondences.intrinsics"
    else:
        raise ValueError(
            "No camera intrinsics found. Add --intrinsics camera.json or "
            "build correspondences with --intrinsics."
        )
    # A rank-deficient set is not invalid, but this warning is useful because
    # four coplanar floor marks are more ambiguous than a spatially spread set.
    object_rank = int(np.linalg.matrix_rank(object_points - object_points.mean(axis=0)))
    if object_rank < 3:
        print(
            "warning: object points are coplanar/low-rank; use 6-10 well-spread "
            "anchors including height variation when possible"
        )
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP did not find a pose")
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    camera_position = (-R.T @ t).reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    reprojection_error = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    output = {
        "metadata": {
            "status": args.status,
            "units": "metres",
            "coordinate_system": "same as input object_points_world",
            "pose_method": "solvePnP",
            "correspondence_count": int(len(object_points)),
            "intrinsics_source": intrinsics_source,
            "object_point_rank": object_rank,
        },
        "image_size": data.get("image_size"),
        "intrinsics": {"K": K, "dist_coeffs": dist.reshape(-1)},
        "extrinsics": {"R": R, "t": t, "camera_position": camera_position},
        "reprojection_error_px": {
            "mean": float(reprojection_error.mean()),
            "median": float(np.median(reprojection_error)),
            "p95": float(np.percentile(reprojection_error, 95)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value,
        ),
        encoding="utf-8",
    )
    print(json.dumps(output["reprojection_error_px"], indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
