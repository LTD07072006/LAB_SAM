"""Train/benchmark learned pixel-to-world scene coordinates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from camera_calibration import CameraCalibration
from homography_floor import FloorHomography
from localization import localize_pixels
from mesh_loader import load_triangle_mesh
from scene_coordinate_regression import FEATURE_NAMES, SceneCoordinateMLP, build_samples, load_manifest_rows, make_feature_vector, metric_summary


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def _calibration(row: dict[str, Any], source: str) -> CameraCalibration:
    camera = row.get("camera_estimated" if source == "estimated" else "camera")
    if not isinstance(camera, dict):
        raise ValueError("missing camera calibration")
    return CameraCalibration.from_dict({
        "image_size": row.get("image_size"),
        "intrinsics": {"K": camera["K"], "dist_coeffs": camera.get("dist_coeffs", [])},
        "extrinsics": {"R": camera["R_world_to_camera"], "camera_position": camera["camera_position"]},
    })


def _location(point: Any, method: str, start: float, reason: str | None = None) -> dict[str, Any]:
    point = None if point is None else np.asarray(point, dtype=np.float64)
    return {"hit": point is not None, "status": f"valid_{method}" if point is not None else (reason or "no_result"), "point": point, "method": method, "confidence": 1.0 if point is not None else 0.0, "latency_ms": (time.perf_counter() - start) * 1000.0}


def _ray(row: dict[str, Any], pixel: np.ndarray, mesh: Any, source: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        cal = _calibration(row, source)
        radius_px = 2.0
        offsets = np.asarray(
            [[0.0, 0.0], [radius_px, 0.0], [-radius_px, 0.0], [0.0, radius_px], [0.0, -radius_px]],
            dtype=np.float64,
        )
        result = localize_pixels(
            cal.geometry(), mesh,
            cal.undistort_pixels(pixel.reshape(1, 2) + offsets),
            max_dist=100.0, step=0.25,
        )
        output = _location(result.point if result.hit else None, "ray_single", start, result.status)
        # ``localize_pixels`` keeps the robustly retained hit cloud.  It is a
        # conservative diagnostic (outliers may already be removed), while
        # the learned branch itself never uses this mesh result.
        output.update({"ray_count": 5, "ray_hits": int(len(result.points)), "spread_m": float(result.spread)})
        return output
    except (TypeError, ValueError, np.linalg.LinAlgError, AttributeError) as exc:
        return _location(None, "ray_single", start, str(exc))


def _ipm(row: dict[str, Any], pixel: np.ndarray, source: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        cal = _calibration(row, source)
        point = FloorHomography.from_calibration(cal).pixel_to_floor_xyz(cal.undistort_pixels(pixel.reshape(1, 2)))[0]
        output = _location(point, "ipm_floor", start)
        output.update({"ray_count": 1, "ray_hits": 1, "spread_m": 0.0})
        return output
    except (TypeError, ValueError, np.linalg.LinAlgError, AttributeError) as exc:
        return _location(None, "ipm_floor", start, str(exc))


def _learned(model: SceneCoordinateMLP, row: dict[str, Any], pixel: np.ndarray, source: str) -> dict[str, Any]:
    start = time.perf_counter()
    features = make_feature_vector(row, pixel, source)
    if features is None:
        return _location(None, "scene_mlp", start, "invalid_features")
    output = _location(model.predict_features(features.reshape(1, -1))[0], "scene_mlp", start)
    output["std_m"] = model.validation_std_m
    return output


def _choose(samples: list[Any], maximum: int) -> list[Any]:
    if maximum <= 0 or len(samples) <= maximum:
        return samples
    return [samples[int(index)] for index in np.linspace(0, len(samples) - 1, maximum, dtype=int)]


def _evaluate(model: SceneCoordinateMLP, samples: list[Any], mesh: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = _choose(samples, args.max_test_records)
    records = []
    for index, sample in enumerate(selected, start=1):
        row, pixel = sample.row, sample.pixel.astype(np.float64)
        records.append({
            "sample_id": row.get("sample_id", f"sample_{index:04d}"), "scene_id": sample.scene_id,
            "frame_index": int(row.get("frame_index", index - 1)), "image_path": str((args.dataset / row["image_path"]).resolve()),
            "image_size": row.get("image_size", [640, 640]), "gt_pixel": row.get("p_fire_pixel"), "input_pixel": pixel,
            "gt_xyz": sample.target_xyz.astype(np.float64), "surface": row.get("fire_surface", "unknown"),
            "branches": {"scene_mlp": {"pixel": pixel, "location": _learned(model, row, pixel, args.camera_source)}, "ray": {"pixel": pixel, "location": _ray(row, pixel, mesh, args.camera_source)}, "ipm": {"pixel": pixel, "location": _ipm(row, pixel, args.camera_source)}},
        })
        if index == 1 or index == len(selected) or index % 25 == 0:
            print(f"evaluated={index}/{len(selected)} sample={records[-1]['sample_id']}", flush=True)
    return records


def _metrics(records: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    truth, prediction, latency = [], [], []
    for record in records:
        current = record["branches"][branch]["location"]
        if current.get("point") is not None:
            truth.append(record["gt_xyz"]); prediction.append(current["point"])
        latency.append(float(current.get("latency_ms", 0.0)))
    result = metric_summary(np.asarray(truth), np.asarray(prediction)).to_dict()
    result.update({"branch": branch, "records": len(records), "valid_rate": len(prediction) / max(1, len(records)), "latency_ms": {"mean": float(np.mean(latency)), "median": float(np.median(latency)), "p95": float(np.percentile(latency, 95))}})
    return result


def run(args: argparse.Namespace) -> None:
    args.dataset, args.output_dir = args.dataset.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = {split: load_manifest_rows(args.dataset, split) for split in ("train", "val", "test")}
    sample_sets = {split: build_samples(rows[split], args.pixel_key, args.camera_source) for split in rows}
    train, val, test = (sample_sets[split][0] for split in ("train", "val", "test"))
    scene_sets = [{sample.scene_id for sample in group} for group in (train, val, test)]
    overlap = (scene_sets[0] & scene_sets[1]) | (scene_sets[0] & scene_sets[2]) | (scene_sets[1] & scene_sets[2])
    if overlap:
        raise RuntimeError(f"scene leakage detected: {sorted(overlap)[:5]}")
    mesh_path = (args.mesh or args.dataset / "room_mesh.json").resolve()
    mesh = load_triangle_mesh(mesh_path)
    model = SceneCoordinateMLP(hidden_dim=args.hidden_dim, device=args.device)
    history = model.fit(train, val, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed)
    checkpoint = args.output_dir / "best_scene_coordinate_mlp.pth"
    model.save(checkpoint, metadata={"dataset": str(args.dataset), "pixel_key": args.pixel_key, "camera_source": args.camera_source})
    records = _evaluate(model, test, mesh, args)
    branch_metrics = {branch: _metrics(records, branch) for branch in ("scene_mlp", "ray", "ipm")}
    summary = {"format": "LAB_SAM.scene_coordinate_regression.v1", "method": {"description": "pixel+camera metadata to XYZ; scene_mlp is independent of homography/ray casting", "pixel_key": args.pixel_key, "camera_source": args.camera_source, "checkpoint": str(checkpoint), "mesh": str(mesh_path), "mesh_for_reference_baselines": str(mesh_path), "feature_names": list(FEATURE_NAMES)}, "dataset": {"root": str(args.dataset), "synthetic_metric_geometry": True, "split": "test", "samples": {split: len(sample_sets[split][0]) for split in sample_sets}, "scenes": {split: len({sample.scene_id for sample in sample_sets[split][0]}) for split in sample_sets}, "skipped": {split: sample_sets[split][1] for split in sample_sets}, "warning": "Synthetic/asset-backed metric result; not real CCTV accuracy."}, "training": {"epochs_requested": args.epochs, "epochs_ran": len(history), "best_val_mae_m": min((item["val_mae_m"] for item in history), default=None), "device": args.device}, "branches": branch_metrics, "records": records}
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_default), encoding="utf-8")
    (args.output_dir / "test_predictions.json").write_text(json.dumps(records, indent=2, default=_default), encoding="utf-8")
    print("branch       3D_MAE(m) median(m) P95(m) valid")
    for branch, item in branch_metrics.items():
        print(f"{branch:<12} {item['mae_m']!s:>9} {item['median_m']!s:>9} {item['p95_m']!s:>8} {item['valid_rate']:.3f}")
    print(f"saved_summary={args.output_dir / 'summary.json'}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "scene_coordinate_regression_20261008")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--pixel-key", choices=("p_fire_noisy_pixel", "p_fire_pixel"), default="p_fire_noisy_pixel")
    parser.add_argument("--camera-source", choices=("estimated", "true"), default="estimated")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-test-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch-size and hidden-dim must be positive")
    run(args)


if __name__ == "__main__":
    main()
