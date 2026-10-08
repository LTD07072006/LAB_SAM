"""Benchmark two low-cost post-ROI 3D directions without Homography/IPM.

Branches:

* ``knn_idw``: local interpolation from the training split's metric samples;
* ``scene_mlp``: the existing learned pixel+camera-to-XYZ checkpoint;
* ``ray`` and ``ipm``: reference baselines on the same strict test detections;
* ``triangulation_*``: mesh-free multi-view DLT using adjacent observations.

Unlike the older convenience loader, this benchmark never substitutes a clean
ground-truth pixel when ``p_fire_noisy_pixel`` is missing. Such detector misses
are excluded and counted, avoiding oracle leakage in the primary test results.
The clean-pixel/true-pose triangulation is included only as an explicit upper-
bound diagnostic.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from camera_calibration import CameraCalibration
from homography_floor import FloorHomography
from localization import localize_pixels
from mesh_loader import load_triangle_mesh
from scene_coordinate_regression import (
    FEATURE_NAMES,
    SceneCoordinateMLP,
    make_feature_vector,
    metric_summary,
)
from triangulation_3d import triangulate_dlt


LOW_COST_BRANCHES = (
    "knn_idw",
    "scene_mlp",
    "ray",
    "ipm",
    "triangulation_noisy_estimated",
    "triangulation_noisy_true",
    "triangulation_clean_true_oracle",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _load_rows(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / f"{split}.jsonl"
    if not path.is_file():
        path = root / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No {split}.jsonl or manifest.jsonl under {root}")
    rows = []
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


def _point(value: Any, dims: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(point) < dims or not np.all(np.isfinite(point[:dims])):
        return None
    return point[:dims].copy()


def _strict_pixel(row: dict[str, Any], key: str) -> Optional[np.ndarray]:
    """Return a real observation only; never fall back to a clean label."""
    pixel = _point(row.get(key), 2)
    if pixel is None or np.allclose(pixel, 0.0, atol=1e-12):
        return None
    image_size = _point(row.get("image_size", [640, 640]), 2)
    if image_size is None or np.any(image_size <= 0):
        return None
    width, height = image_size
    if not (-0.5 <= pixel[0] < width + 0.5 and -0.5 <= pixel[1] < height + 0.5):
        return None
    return pixel


def _camera(row: dict[str, Any], source: str) -> Optional[CameraCalibration]:
    camera = row.get("camera_estimated" if source == "estimated" else "camera")
    if not isinstance(camera, dict):
        return None
    try:
        return CameraCalibration.from_dict(
            {
                "image_size": row.get("image_size"),
                "intrinsics": {"K": camera.get("K"), "dist_coeffs": camera.get("dist_coeffs", [])},
                "extrinsics": {
                    "R": camera.get("R_world_to_camera", camera.get("R")),
                    "camera_position": camera.get("camera_position"),
                },
            }
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _valid_samples(rows: list[dict[str, Any]], pixel_key: str, camera_source: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            skipped["not_visible_fire"] += 1
            continue
        pixel = _strict_pixel(row, pixel_key)
        if pixel is None:
            skipped["missing_or_invalid_observation"] += 1
            continue
        target = _point(row.get("fire_xyz_world"), 3)
        if target is None:
            skipped["missing_metric_xyz"] += 1
            continue
        calibration = _camera(row, camera_source)
        if calibration is None:
            skipped["invalid_camera"] += 1
            continue
        features = make_feature_vector(row, pixel, camera_source)
        if features is None:
            skipped["invalid_features"] += 1
            continue
        samples.append({
            "row": row,
            "pixel": pixel,
            "target": target,
            "camera": calibration,
            "features": features,
            "scene_id": str(row.get("scene_id", "unknown")),
        })
    return samples, dict(skipped)


class KNNIDW:
    """Dependency-free, train-scene-only local scene-coordinate mapper."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        values = np.asarray(features, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
        labels = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
        if len(values) < 2 or len(values) != len(labels):
            raise ValueError("KNN-IDW needs at least two matching training samples")
        self.mean = values.mean(axis=0)
        self.scale = np.maximum(values.std(axis=0), 1e-5)
        self.features = (values - self.mean) / self.scale
        self.targets = labels
        self.k: int = 3

    def _neighbors(self, features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = (np.asarray(features, dtype=np.float64).reshape(-1, self.features.shape[1]) - self.mean) / self.scale
        distances = np.linalg.norm(query[:, None, :] - self.features[None, :, :], axis=2)
        indices = np.argpartition(distances, kth=min(k - 1, distances.shape[1] - 1), axis=1)[:, :k]
        selected_distances = np.take_along_axis(distances, indices, axis=1)
        order = np.argsort(selected_distances, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        selected_distances = np.take_along_axis(selected_distances, order, axis=1)
        return indices, selected_distances

    def predict(self, features: np.ndarray, k: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = int(k or self.k)
        indices, distances = self._neighbors(features, count)
        weights = 1.0 / np.maximum(distances, 1e-4) ** 2
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        neighbors = self.targets[indices]
        mean = np.sum(neighbors * weights[:, :, None], axis=1)
        variance = np.sum(weights[:, :, None] * (neighbors - mean[:, None, :]) ** 2, axis=1)
        return mean, np.sqrt(np.maximum(variance, 1e-6)), distances[:, 0]


def _location(point: Optional[np.ndarray], method: str, latency_ms: float, **extra: Any) -> dict[str, Any]:
    return {
        "hit": point is not None,
        "status": "valid_" + method if point is not None else extra.pop("reason", "no_result"),
        "point": point,
        "method": method,
        "confidence": 1.0 if point is not None else 0.0,
        "latency_ms": float(latency_ms),
        **extra,
    }


def _ray(sample: dict[str, Any], mesh: Any) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        offsets = np.asarray([[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]], dtype=np.float64)
        ideal = sample["camera"].undistort_pixels(sample["pixel"].reshape(1, 2) + offsets)
        result = localize_pixels(sample["camera"].geometry(), mesh, ideal, max_dist=100.0, step=0.25)
        latency = (time.perf_counter() - started) * 1000.0
        return _location(
            result.point if result.hit else None,
            "ray",
            latency,
            ray_count=5,
            ray_hits=int(len(result.points)),
            reason=result.status,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError, AttributeError) as exc:
        return _location(None, "ray", (time.perf_counter() - started) * 1000.0, reason=str(exc), ray_count=5, ray_hits=0)


def _ipm(sample: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        camera = sample["camera"]
        ideal = camera.undistort_pixels(sample["pixel"].reshape(1, 2))
        point = FloorHomography.from_calibration(camera).pixel_to_floor_xyz(ideal)[0]
        return _location(point, "ipm", (time.perf_counter() - started) * 1000.0)
    except (TypeError, ValueError, np.linalg.LinAlgError, AttributeError) as exc:
        return _location(None, "ipm", (time.perf_counter() - started) * 1000.0, reason=str(exc))


def _metrics(entries: list[tuple[np.ndarray, Optional[np.ndarray], float]]) -> dict[str, Any]:
    selected = [(gt, pred, latency) for gt, pred, latency in entries if pred is not None]
    if selected:
        gt = np.asarray([item[0] for item in selected], dtype=np.float64)
        pred = np.asarray([item[1] for item in selected], dtype=np.float64)
        metrics = metric_summary(gt, pred).to_dict()
    else:
        metrics = metric_summary(np.empty((0, 3)), np.empty((0, 3))).to_dict()
    metrics["valid_rate"] = float(len(selected) / max(1, len(entries)))
    metrics["latency_ms"] = {
        "mean": None if not entries else float(np.mean([entry[2] for entry in entries])),
        "median": None if not entries else float(np.median([entry[2] for entry in entries])),
        "p95": None if not entries else float(np.percentile([entry[2] for entry in entries], 95)),
    }
    return metrics


def _sequence_triangulation(
    rows: list[dict[str, Any]],
    key: str,
    camera_source: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            continue
        pixel = _strict_pixel(row, key)
        calibration = _camera(row, camera_source)
        target = _point(row.get("fire_xyz_world"), 3)
        if pixel is not None and calibration is not None and target is not None:
            grouped[str(row.get("scene_id", "unknown"))].append({"row": row, "pixel": pixel, "camera": calibration, "target": target})

    outputs: dict[str, dict[str, Any]] = {}
    metric_entries: list[tuple[np.ndarray, Optional[np.ndarray], float]] = []
    baseline_lengths: list[float] = []
    reprojections: list[float] = []
    failed: dict[str, int] = defaultdict(int)
    for scene_id, views in grouped.items():
        views.sort(key=lambda item: int(item["row"].get("frame_index", 0)))
        targets = np.asarray([view["target"] for view in views])
        if len(targets) and float(np.max(np.linalg.norm(targets - targets[0], axis=1))) > 1e-4:
            failed["non_static_ground_truth_within_sequence"] += 1
            continue
        if len(views) < 2:
            failed["fewer_than_two_observations"] += 1
            continue
        camera_positions = np.asarray([view["camera"].camera_position() for view in views])
        pairwise = [
            float(np.linalg.norm(camera_positions[i] - camera_positions[j]))
            for i in range(len(views))
            for j in range(i + 1, len(views))
        ]
        baseline_lengths.append(max(pairwise, default=0.0))
        started = time.perf_counter()
        result = triangulate_dlt([view["pixel"] for view in views], [view["camera"] for view in views])
        latency = (time.perf_counter() - started) * 1000.0
        point = result.point if result.success else None
        metric_entries.append((views[0]["target"], point, latency))
        if result.reprojection_rmse_px is not None:
            reprojections.append(float(result.reprojection_rmse_px))
        if not result.success:
            failed[str(result.reason or "triangulation_failed")] += 1
        outputs[scene_id] = {
            **result.to_dict(),
            "latency_ms_total": latency,
            "baseline_max_m": max(pairwise, default=0.0),
            "scene_id": scene_id,
        }
    return outputs, {
        "metrics": _metrics(metric_entries),
        "scenes_attempted": len(grouped),
        "scenes_triangulated": sum(item[1] is not None for item in metric_entries),
        "baseline_max_m_median": None if not baseline_lengths else float(np.median(baseline_lengths)),
        "baseline_max_m_p95": None if not baseline_lengths else float(np.percentile(baseline_lengths, 95)),
        "reprojection_rmse_px_median": None if not reprojections else float(np.median(reprojections)),
        "reprojection_rmse_px_p95": None if not reprojections else float(np.percentile(reprojections, 95)),
        "failures": dict(failed),
        "camera_source": camera_source,
        "pixel_key": key,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = {split: _load_rows(dataset, split) for split in ("train", "val", "test")}
    samples: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, dict[str, int]] = {}
    for split in raw:
        samples[split], skipped[split] = _valid_samples(raw[split], args.pixel_key, args.camera_source)
    scene_sets = [{sample["scene_id"] for sample in samples[split]} for split in ("train", "val", "test")]
    overlap = (scene_sets[0] & scene_sets[1]) | (scene_sets[0] & scene_sets[2]) | (scene_sets[1] & scene_sets[2])
    if overlap:
        raise RuntimeError(f"Scene leakage between splits: {sorted(overlap)[:10]}")
    if len(samples["train"]) < 2 or not samples["val"] or not samples["test"]:
        raise RuntimeError(f"Insufficient valid data after strict miss filtering: { {k: len(v) for k, v in samples.items()} }")

    train = samples["train"]
    val = samples["val"]
    test = samples["test"]
    knn = KNNIDW(np.asarray([item["features"] for item in train]), np.asarray([item["target"] for item in train]))
    k_candidates = sorted({min(k, len(train)) for k in (1, 3, 5, 9, 15, 25)})
    validation_by_k: dict[str, float] = {}
    for k in k_candidates:
        prediction, _, _ = knn.predict(np.asarray([item["features"] for item in val]), k)
        validation_by_k[str(k)] = float(np.linalg.norm(prediction - np.asarray([item["target"] for item in val]), axis=1).mean())
    selected_k = min(k_candidates, key=lambda k: validation_by_k[str(k)])
    knn.k = selected_k

    checkpoint = args.checkpoint.expanduser().resolve()
    mlp = SceneCoordinateMLP.load(checkpoint, device=args.device) if checkpoint.is_file() else None
    if mlp is None and not args.allow_missing_mlp:
        raise FileNotFoundError(f"Scene MLP checkpoint not found: {checkpoint}; pass --allow-missing-mlp to skip it")

    mesh_path = (args.mesh or (dataset / "room_mesh.json")).expanduser().resolve()
    mesh = load_triangle_mesh(mesh_path)
    test_features = np.asarray([item["features"] for item in test], dtype=np.float32)
    knn_prediction, knn_std, knn_distance = knn.predict(test_features)
    mlp_prediction = None if mlp is None else mlp.predict_features(test_features)

    records: list[dict[str, Any]] = []
    branch_entries: dict[str, list[tuple[np.ndarray, Optional[np.ndarray], float]]] = {name: [] for name in LOW_COST_BRANCHES}
    groups_for_visuals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, sample in enumerate(test):
        row = sample["row"]
        started = time.perf_counter()
        knn_loc = _location(knn_prediction[index], "knn_idw", (time.perf_counter() - started) * 1000.0,
                            std_m=knn_std[index], nearest_training_distance=float(knn_distance[index]), k=selected_k)
        branch_entries["knn_idw"].append((sample["target"], knn_prediction[index], knn_loc["latency_ms"]))

        if mlp_prediction is None:
            mlp_loc = _location(None, "scene_mlp", 0.0, reason="checkpoint_not_supplied")
        else:
            started = time.perf_counter()
            point = mlp.predict_features(sample["features"].reshape(1, -1))[0]
            mlp_loc = _location(point, "scene_mlp", (time.perf_counter() - started) * 1000.0,
                                std_m=mlp.validation_std_m)
        branch_entries["scene_mlp"].append((sample["target"], mlp_loc["point"], mlp_loc["latency_ms"]))

        ray_loc = _ray(sample, mesh)
        ipm_loc = _ipm(sample)
        branch_entries["ray"].append((sample["target"], ray_loc["point"], ray_loc["latency_ms"]))
        branch_entries["ipm"].append((sample["target"], ipm_loc["point"], ipm_loc["latency_ms"]))

        record = {
            "sample_id": str(row.get("sample_id", f"test_{index:04d}")),
            "scene_id": sample["scene_id"],
            "frame_index": int(row.get("frame_index", index)),
            "image_path": str((dataset / str(row.get("image_path", ""))).resolve()),
            "image_size": row.get("image_size", [640, 640]),
            "gt_pixel": row.get("p_fire_pixel"),
            "input_pixel": sample["pixel"],
            "gt_xyz": sample["target"],
            "surface": row.get("fire_surface", "unknown"),
            "camera": row.get("camera"),
            "camera_estimated": row.get("camera_estimated"),
            "branches": {
                "knn_idw": {"pixel": sample["pixel"], "location": knn_loc},
                "scene_mlp": {"pixel": sample["pixel"], "location": mlp_loc},
                "ray": {"pixel": sample["pixel"], "location": ray_loc},
                "ipm": {"pixel": sample["pixel"], "location": ipm_loc},
                # Multi-view estimates are attached to the first frame of
                # each scene below.  Keep each variant separate so the
                # visualizer never silently overwrites noisy/true/oracle
                # results with the last loop iteration.
                "triangulation_noisy_estimated": {"pixel": sample["pixel"], "location": None},
                "triangulation_noisy_true": {"pixel": sample["pixel"], "location": None},
                "triangulation_clean_true_oracle": {"pixel": sample["pixel"], "location": None},
            },
        }
        records.append(record)
        groups_for_visuals[sample["scene_id"]].append(record)
        if index == 0 or index + 1 == len(test) or (index + 1) % 25 == 0:
            print(f"test={index + 1}/{len(test)} sample={record['sample_id']}", flush=True)

    tri_est, tri_est_info = _sequence_triangulation(raw["test"], args.triangulation_pixel_key, "estimated")
    tri_true, tri_true_info = _sequence_triangulation(raw["test"], args.triangulation_pixel_key, "true")
    tri_oracle, tri_oracle_info = _sequence_triangulation(raw["test"], "p_fire_pixel", "true")
    for name, results, info in (
        ("triangulation_noisy_estimated", tri_est, tri_est_info),
        ("triangulation_noisy_true", tri_true, tri_true_info),
        ("triangulation_clean_true_oracle", tri_oracle, tri_oracle_info),
    ):
        branch_entries[name] = []
        for scene_id, result in results.items():
            views = groups_for_visuals.get(scene_id, [])
            if views:
                first = min(views, key=lambda item: int(item["frame_index"]))
                first["branches"][name]["location"] = _location(
                    result.get("point") if result.get("success") else None,
                    "triangulation",
                    float(result.get("latency_ms_total", 0.0)),
                    reprojection_rmse_px=result.get("reprojection_rmse_px"),
                    baseline_max_m=result.get("baseline_max_m"),
                    used_views=result.get("used_views"),
                    reason=result.get("reason") or "multi_view_result_attached_to_first_frame",
                    variant=name,
                )
        info["metrics"]["valid_rate"] = float(info["scenes_triangulated"] / max(1, info["scenes_attempted"]))
        info["metrics"]["latency_ms"]["per_view_mean"] = (
            None if not info["scenes_triangulated"] else float(
                np.mean([item["latency_ms_total"] / max(1, item["used_views"]) for item in results.values() if item.get("success")])
            )
        )

    metrics = {name: _metrics(entries) for name, entries in branch_entries.items()}
    # The sequence-level methods use one independent estimate per scene, not
    # duplicated per-frame coordinates.
    metrics["triangulation_noisy_estimated"] = tri_est_info["metrics"]
    metrics["triangulation_noisy_true"] = tri_true_info["metrics"]
    metrics["triangulation_clean_true_oracle"] = tri_oracle_info["metrics"]
    summary = {
        "format": "LAB_SAM.low_cost_post_roi_3d_benchmark.v1",
        "method": {
            "description": "KNN/IDW scene-coordinate regression and multi-view triangulation; no homography or mesh is used in these branches",
            "mesh": str(mesh_path),
            "checkpoint": str(checkpoint) if mlp is not None else None,
            "pixel_key": args.pixel_key,
            "camera_source": args.camera_source,
            "feature_names": list(FEATURE_NAMES),
        },
        "dataset": {
            "root": str(dataset),
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry result, not real CCTV accuracy.",
            "strict_detector_miss_policy": "missing noisy observation is excluded; no clean-GT fallback",
            "valid_samples": {split: len(samples[split]) for split in samples},
            "raw_records": {split: len(raw[split]) for split in raw},
            "skipped": skipped,
            "scenes": {split: len(scene_sets[index]) for index, split in enumerate(("train", "val", "test"))},
            "split_scene_overlap": 0,
        },
        "training": {
            "knn_idw": {"k_candidates": k_candidates, "validation_mae_by_k_m": validation_by_k, "selected_k": selected_k,
                        "train_samples": len(train), "train_scenes": len(scene_sets[0]), "backend": "NumPy IDW; no new dependency"},
            "scene_mlp": {"checkpoint": str(checkpoint) if mlp is not None else None,
                          "status": "loaded" if mlp is not None else "skipped"},
        },
        "triangulation": {
            "noisy_estimated_pose": tri_est_info,
            "noisy_true_pose": tri_true_info,
            "clean_pixel_true_pose_oracle": tri_oracle_info,
            "warning": "Triangulation needs camera motion/parallax and a common metric world frame. Clean-pixel/true-pose is an oracle diagnostic only.",
        },
        "branches": metrics,
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (output / "test_predictions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (output / "branch_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print("branch                              MAE(m) median(m) P95(m) valid")
    for name, metric in metrics.items():
        print(f"{name:<36} {str(metric.get('mae_m')):>8} {str(metric.get('median_m')):>9} {str(metric.get('p95_m')):>8} {metric.get('valid_rate', 0.0):.3f}")
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "low_cost_3d_benchmark_20261008")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=root / "output" / "scene_coordinate_regression_rerun_20261008" / "best_scene_coordinate_mlp.pth")
    parser.add_argument("--allow-missing-mlp", action="store_true")
    parser.add_argument("--pixel-key", choices=("p_fire_noisy_pixel", "p_fire_pixel"), default="p_fire_noisy_pixel")
    parser.add_argument("--triangulation-pixel-key", choices=("p_fire_noisy_pixel", "p_fire_pixel"), default="p_fire_noisy_pixel")
    parser.add_argument("--camera-source", choices=("estimated", "true"), default="estimated")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
