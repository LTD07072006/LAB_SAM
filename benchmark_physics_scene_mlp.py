"""Benchmark the physics-informed scene MLP against Ray Casting and IPM.

The benchmark is intentionally post-ROI. It trains one depth/ray model under
three regimes and tests every model on the same four input conditions:

    clean  -> clean_true
    noisy  -> noisy_estimated
    mixed  -> clean/noisy x true/estimated

The learned model predicts metric camera depth, a bounded correction to the
pinhole ray and uncertainty. A validation-only uncertainty gate chooses
between the learned point, mesh Ray Casting, floor IPM and not_localized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from camera_calibration import CameraCalibration
from homography_floor import FloorHomography
from locator import TriangleMesh, intersect_ray_with_grid_result
from physics_scene_coordinate import (
    PhysicsSceneCoordinateMLP,
    PhysicsSceneSample,
    build_samples,
    metric_summary,
)
from scene_coordinate_regression import load_manifest_rows


CONDITIONS = ("clean_true", "noisy_true", "clean_estimated", "noisy_estimated")
TRAIN_MODES: dict[str, tuple[str, ...]] = {
    "clean": ("clean_true",),
    "noisy": ("noisy_estimated",),
    "mixed": CONDITIONS,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _check_scene_split(sample_sets: dict[str, list[PhysicsSceneSample]]) -> None:
    scene_sets = {
        split: {sample.scene_id for sample in samples}
        for split, samples in sample_sets.items()
    }
    overlap = (
        (scene_sets["train"] & scene_sets["val"])
        | (scene_sets["train"] & scene_sets["test"])
        | (scene_sets["val"] & scene_sets["test"])
    )
    if overlap:
        raise RuntimeError(f"Scene leakage detected: {sorted(overlap)[:10]}")


def _balanced(samples: list[PhysicsSceneSample], enabled: bool) -> list[PhysicsSceneSample]:
    if not enabled or not samples:
        return samples
    groups: dict[str, list[PhysicsSceneSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.surface].append(sample)
    target = max(len(group) for group in groups.values())
    result: list[PhysicsSceneSample] = []
    for surface in sorted(groups):
        group = groups[surface]
        repeats = int(math.ceil(target / len(group)))
        result.extend((group * repeats)[:target])
    return result


def _calibration(sample: PhysicsSceneSample) -> CameraCalibration:
    source = "camera_estimated" if sample.condition.endswith("estimated") else "camera"
    value = sample.row[source]
    return CameraCalibration.from_dict(
        {
            "image_size": sample.row.get("image_size"),
            "intrinsics": {
                "K": value["K"],
                "dist_coeffs": value.get("dist_coeffs", []),
            },
            "extrinsics": {
                "R": value["R_world_to_camera"],
                "camera_position": value["camera_position"],
            },
        }
    )


def _describe(values: Iterable[float]) -> dict[str, Optional[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _run_physics(
    model: PhysicsSceneCoordinateMLP,
    samples: list[PhysicsSceneSample],
) -> dict[str, Any]:
    if not samples:
        return {
            "metrics": metric_summary(np.empty((0, 3)), np.empty((0, 3))),
            "predictions": [],
            "latency_ms": _describe([]),
        }
    started = time.perf_counter()
    prediction = model.predict(samples)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    per_sample_ms = elapsed_ms / max(1, len(samples))
    entries: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sigma = float(np.linalg.norm(prediction["std_m"][index]))
        entries.append(
            {
                "sample_id": sample.row.get("sample_id"),
                "scene_id": sample.scene_id,
                "surface": sample.surface,
                "condition": sample.condition,
                "gt_xyz": sample.target_xyz,
                "point": prediction["xyz"][index],
                "std_m": prediction["std_m"][index],
                "sigma_norm_m": sigma,
                "confidence": float(1.0 / (1.0 + sigma)),
                "depth_m": float(prediction["depth_m"][index]),
                "corrected_ray_xy": prediction["corrected_ray_xy"][index],
                "latency_ms": per_sample_ms,
            }
        )
    return {
        "metrics": metric_summary(
            np.asarray([sample.target_xyz for sample in samples], dtype=np.float32),
            prediction["xyz"],
        ),
        "predictions": entries,
        "latency_ms": _describe([per_sample_ms] * len(samples)),
    }


def _run_ray_and_ipm(
    samples: list[PhysicsSceneSample],
    mesh: TriangleMesh,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ray_latencies: list[float] = []
    ipm_latencies: list[float] = []
    for sample in samples:
        calibration = _calibration(sample)
        geometry = calibration.geometry()

        started = time.perf_counter()
        origin, direction = geometry.pixel_to_ray(float(sample.pixel[0]), float(sample.pixel[1]))
        ray_hit = intersect_ray_with_grid_result(origin, direction, mesh, max_dist=1000.0)
        ray_latency = (time.perf_counter() - started) * 1000.0
        ray_point = None if not ray_hit.hit else np.asarray(ray_hit.point, dtype=np.float32)

        started = time.perf_counter()
        try:
            ipm_point = FloorHomography.from_calibration(calibration).pixel_to_floor_xyz(
                [sample.pixel]
            )[0].astype(np.float32)
        except (ValueError, np.linalg.LinAlgError):
            ipm_point = None
        ipm_latency = (time.perf_counter() - started) * 1000.0

        records.append(
            {
                "sample_id": sample.row.get("sample_id"),
                "scene_id": sample.scene_id,
                "surface": sample.surface,
                "condition": sample.condition,
                "gt_xyz": sample.target_xyz,
                "ray_point": ray_point,
                "ray_status": ray_hit.status,
                "ray_distance_m": float(ray_hit.distance),
                "ray_latency_ms": ray_latency,
                "ipm_point": ipm_point,
                "ipm_latency_ms": ipm_latency,
            }
        )
        ray_latencies.append(ray_latency)
        ipm_latencies.append(ipm_latency)

    def baseline_metrics(key: str) -> dict[str, Any]:
        valid = [record for record in records if record[key] is not None]
        return {
            "metrics": metric_summary(
                np.asarray([record["gt_xyz"] for record in valid], dtype=np.float32).reshape(-1, 3)
                if valid
                else np.empty((0, 3)),
                np.asarray([record[key] for record in valid], dtype=np.float32).reshape(-1, 3)
                if valid
                else np.empty((0, 3)),
            ),
            "valid_rate": float(len(valid) / max(1, len(records))),
        }

    return {
        "records": records,
        "ray": {**baseline_metrics("ray_point"), "latency_ms": _describe(ray_latencies)},
        "ipm": {**baseline_metrics("ipm_point"), "latency_ms": _describe(ipm_latencies)},
    }


def _route(records: list[dict[str, Any]], threshold_m: float) -> dict[str, Any]:
    selected_points: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sources: Counter[str] = Counter()
    output_records: list[dict[str, Any]] = []
    for record in records:
        point = None
        source = "not_localized"
        if record["physics_point"] is not None and record["sigma_norm_m"] <= threshold_m:
            point = record["physics_point"]
            source = "physics_mlp"
        elif record["ray_point"] is not None:
            point = record["ray_point"]
            source = "ray_casting"
        elif record["ipm_point"] is not None:
            point = record["ipm_point"]
            source = "ipm"
        if point is not None:
            selected_points.append(np.asarray(point, dtype=np.float32))
            targets.append(np.asarray(record["gt_xyz"], dtype=np.float32))
        sources[source] += 1
        output_records.append(
            {**record, "selected_point": point, "selected_source": source}
        )
    return {
        "metrics": metric_summary(
            np.asarray(targets, dtype=np.float32).reshape(-1, 3)
            if targets
            else np.empty((0, 3)),
            np.asarray(selected_points, dtype=np.float32).reshape(-1, 3)
            if selected_points
            else np.empty((0, 3)),
        ),
        "valid_rate": float(len(selected_points) / max(1, len(records))),
        "sources": dict(sources),
        "records": output_records,
    }


def _choose_threshold(
    validation_records: list[dict[str, Any]],
    thresholds: Iterable[float],
) -> dict[str, Any]:
    best: Optional[dict[str, Any]] = None
    for threshold in thresholds:
        routed = _route(validation_records, float(threshold))
        mae = routed["metrics"].get("mae_m")
        score = float(mae) if mae is not None else float("inf")
        candidate = {
            "threshold_m": float(threshold),
            "metrics": routed["metrics"],
            "valid_rate": routed["valid_rate"],
            "sources": routed["sources"],
        }
        if best is None:
            best = candidate
            continue
        best_score = float(best["metrics"].get("mae_m") or float("inf"))
        if (score, -routed["valid_rate"]) < (best_score, -best["valid_rate"]):
            best = candidate
    if best is None:
        raise RuntimeError("No uncertainty threshold candidates")
    return best


def _sample_records(
    physics: list[dict[str, Any]],
    baselines: dict[str, Any],
    routed: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_id = {str(item["sample_id"]): item for item in baselines["records"]}
    route_by_id = {str(item["sample_id"]): item for item in routed["records"]}
    output: list[dict[str, Any]] = []
    for item in physics:
        sample_id = str(item["sample_id"])
        baseline = baseline_by_id[sample_id]
        route = route_by_id[sample_id]
        output.append(
            {
                "sample_id": item["sample_id"],
                "scene_id": item["scene_id"],
                "surface": item["surface"],
                "condition": item["condition"],
                "gt_xyz": item["gt_xyz"],
                "branches": {
                    "scene_mlp": {
                        "point": item["point"],
                        "uncertainty_std_m": item["std_m"],
                        "uncertainty_norm_m": item["sigma_norm_m"],
                        "confidence": item["confidence"],
                    },
                    "ray": {
                        "point": baseline["ray_point"],
                        "status": baseline["ray_status"],
                    },
                    "ipm": {"point": baseline["ipm_point"]},
                    "fusion": {
                        "point": route["selected_point"],
                        "source": route["selected_source"],
                    },
                },
            }
        )
    return output


def _write_visualization_summary(
    output: Path,
    dataset: Path,
    mode: str,
    condition: str,
    records: list[dict[str, Any]],
) -> Path:
    path = output / "visualization_summaries" / f"{mode}_{condition}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "LAB_SAM.physics_scene_visualization.v1",
        "dataset": {
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry, not real CCTV accuracy.",
        },
        "method": {"mesh": str(dataset / "room_mesh.json")},
        "records": records,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "train_mode",
        "test_condition",
        "sample_count",
        "physics_mae_m",
        "physics_median_m",
        "physics_p95_m",
        "physics_latency_ms",
        "ray_mae_m",
        "ray_valid_rate",
        "ray_latency_ms",
        "ipm_mae_m",
        "ipm_valid_rate",
        "ipm_latency_ms",
        "route_mae_m",
        "route_valid_rate",
        "threshold_m",
        "physics_source_count",
        "ray_source_count",
        "ipm_source_count",
        "not_localized_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "histories").mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(parents=True, exist_ok=True)

    rows = {split: load_manifest_rows(dataset, split) for split in ("train", "val", "test")}
    all_samples: dict[str, dict[str, list[PhysicsSceneSample]]] = {}
    skipped: dict[str, dict[str, dict[str, int]]] = {}
    for condition in CONDITIONS:
        all_samples[condition] = {}
        skipped[condition] = {}
        for split in ("train", "val", "test"):
            all_samples[condition][split], skipped[condition][split] = build_samples(
                rows[split], condition
            )

    mesh_path = dataset / "room_mesh.json"
    mesh_data = json.loads(mesh_path.read_text(encoding="utf-8"))
    mesh = TriangleMesh(mesh_data["vertices"], mesh_data["faces"])
    thresholds = np.linspace(
        float(args.min_uncertainty_m),
        float(args.max_uncertainty_m),
        max(2, int(args.threshold_steps)),
    )
    results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for mode, train_conditions in TRAIN_MODES.items():
        train_samples = [
            sample
            for condition in train_conditions
            for sample in all_samples[condition]["train"]
        ]
        val_samples = [
            sample
            for condition in train_conditions
            for sample in all_samples[condition]["val"]
        ]
        _check_scene_split(
            {
                "train": train_samples,
                "val": val_samples,
                "test": [
                    sample
                    for condition in CONDITIONS
                    for sample in all_samples[condition]["test"]
                ],
            }
        )
        train_samples = _balanced(train_samples, args.balance_surfaces)
        model = PhysicsSceneCoordinateMLP(hidden_dim=args.hidden_dim, device=args.device)
        print(
            f"[physics-mlp] mode={mode} train={len(train_samples)} "
            f"val={len(val_samples)} conditions={','.join(train_conditions)}",
            flush=True,
        )
        history = model.fit(
            train_samples,
            val_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        checkpoint = output / "checkpoints" / f"physics_scene_mlp_{mode}.pth"
        model.save(
            checkpoint,
            metadata={
                "train_mode": mode,
                "train_conditions": list(train_conditions),
                "surface_balanced": bool(args.balance_surfaces),
            },
        )
        (output / "histories" / f"physics_scene_mlp_{mode}.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

        validation_records: list[dict[str, Any]] = []
        for condition in train_conditions:
            physics = _run_physics(model, all_samples[condition]["val"])
            baselines = _run_ray_and_ipm(all_samples[condition]["val"], mesh)
            for prediction, baseline in zip(physics["predictions"], baselines["records"]):
                validation_records.append(
                    {
                        **prediction,
                        "physics_point": prediction["point"],
                        "ray_point": baseline["ray_point"],
                        "ipm_point": baseline["ipm_point"],
                    }
                )
        selected_gate = _choose_threshold(validation_records, thresholds)

        mode_result: dict[str, Any] = {
            "train_conditions": list(train_conditions),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "checkpoint": str(checkpoint),
            "epochs_ran": len(history),
            "best_val_mae_m": min((item["val_mae_m"] for item in history), default=None),
            "gate_selected_on_validation": selected_gate,
            "test_conditions": {},
        }

        for condition in CONDITIONS:
            samples = all_samples[condition]["test"]
            print(
                f"[test] mode={mode} condition={condition} samples={len(samples)}",
                flush=True,
            )
            physics = _run_physics(model, samples)
            baselines = _run_ray_and_ipm(samples, mesh)
            combined: list[dict[str, Any]] = []
            for prediction, baseline in zip(physics["predictions"], baselines["records"]):
                combined.append(
                    {
                        **prediction,
                        "physics_point": prediction["point"],
                        "ray_point": baseline["ray_point"],
                        "ipm_point": baseline["ipm_point"],
                    }
                )
            routed = _route(combined, float(selected_gate["threshold_m"]))
            records = _sample_records(physics["predictions"], baselines, routed)
            summary_path = _write_visualization_summary(
                output, dataset, mode, condition, records
            )
            prediction_path = output / "predictions" / f"{mode}_{condition}.json"
            prediction_path.write_text(
                json.dumps(
                    {
                        "physics": physics,
                        "baselines": baselines,
                        "route": routed,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            mode_result["test_conditions"][condition] = {
                "sample_count": len(samples),
                "physics": physics["metrics"],
                "physics_latency_ms": physics["latency_ms"],
                "ray": baselines["ray"],
                "ipm": baselines["ipm"],
                "route": {
                    "metrics": routed["metrics"],
                    "valid_rate": routed["valid_rate"],
                    "sources": routed["sources"],
                    "threshold_m": selected_gate["threshold_m"],
                },
                "visualization_summary": str(summary_path),
            }
            source_counts = Counter(routed["sources"])
            csv_rows.append(
                {
                    "train_mode": mode,
                    "test_condition": condition,
                    "sample_count": len(samples),
                    "physics_mae_m": physics["metrics"].get("mae_m"),
                    "physics_median_m": physics["metrics"].get("median_m"),
                    "physics_p95_m": physics["metrics"].get("p95_m"),
                    "physics_latency_ms": physics["latency_ms"].get("mean"),
                    "ray_mae_m": baselines["ray"]["metrics"].get("mae_m"),
                    "ray_valid_rate": baselines["ray"].get("valid_rate"),
                    "ray_latency_ms": baselines["ray"]["latency_ms"].get("mean"),
                    "ipm_mae_m": baselines["ipm"]["metrics"].get("mae_m"),
                    "ipm_valid_rate": baselines["ipm"].get("valid_rate"),
                    "ipm_latency_ms": baselines["ipm"]["latency_ms"].get("mean"),
                    "route_mae_m": routed["metrics"].get("mae_m"),
                    "route_valid_rate": routed["valid_rate"],
                    "threshold_m": selected_gate["threshold_m"],
                    "physics_source_count": source_counts.get("physics_mlp", 0),
                    "ray_source_count": source_counts.get("ray_casting", 0),
                    "ipm_source_count": source_counts.get("ipm", 0),
                    "not_localized_count": source_counts.get("not_localized", 0),
                }
            )
        results[mode] = mode_result

    metrics_path = output / "physics_scene_metrics.csv"
    _write_csv(metrics_path, csv_rows)
    summary = {
        "format": "LAB_SAM.physics_scene_cross_benchmark.v1",
        "dataset": {
            "root": str(dataset),
            "mesh": str(mesh_path),
            "raw_records": {split: len(rows[split]) for split in rows},
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry results, not real CCTV accuracy.",
            "split_policy": "scene-separated train/val/test; no adjacent-frame leakage",
        },
        "conditions": list(CONDITIONS),
        "train_modes": {key: list(value) for key, value in TRAIN_MODES.items()},
        "physics_model": {
            "predicts": ["metric_camera_depth", "bounded_ray_xy_correction", "xyz_log_variance"],
            "physical_layer": "C + R_world_to_camera.T @ ([ray_x, ray_y, 1] * depth)",
            "loss": "coordinate + depth + ray + reprojection + heteroscedastic residual",
            "domain_generalization": "mixed clean/noisy and true/estimated camera conditions",
        },
        "skipped": skipped,
        "results": results,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "visualization_summaries": str(output / "visualization_summaries"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "physics_scene_cross_20261008_v1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance-surfaces",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-uncertainty-m", type=float, default=0.10)
    parser.add_argument("--max-uncertainty-m", type=float, default=3.00)
    parser.add_argument("--threshold-steps", type=int, default=15)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch-size and hidden-dim must be positive")
    if (
        args.threshold_steps < 2
        or args.min_uncertainty_m <= 0
        or args.max_uncertainty_m <= args.min_uncertainty_m
    ):
        raise ValueError("uncertainty threshold range is invalid")
    run(args)


if __name__ == "__main__":
    main()
