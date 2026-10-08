"""Benchmark SIREN and categorical-depth Scene-MLP variants.

This is an additive benchmark for the post-ROI branch.  It keeps the current
Ray Casting implementation untouched and evaluates four learned controls on
the same metric synthetic fire-point split:

* ``relu_scalar``: ReLU/GELU-style control with a scalar depth head;
* ``siren_scalar``: SIREN hidden layers with a scalar depth head;
* ``relu_categorical``: conventional hidden layers with categorical depth;
* ``siren_categorical``: SIREN plus categorical depth.

The test conditions are the same as the previous physics benchmark:
``clean_true``, ``noisy_true``, ``clean_estimated`` and ``noisy_estimated``.
Every condition is compared to mesh Ray Casting on the exact same records.

The categorical head is inspired by CaDDN's depth-distribution idea, but this
is not the full CaDDN detector: the input is one post-ROI point plus camera
metadata and the distribution is used only to estimate one metric depth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from benchmark_physics_scene_v2 import _run_ray_ipm
from physics_scene_coordinate_v2 import (
    PhysicsSceneV2Sample,
    augment_domain_samples,
    build_samples,
    metric_summary,
)
from physics_scene_siren_categorical import PhysicsSceneSirenCategorical
from physics_scene_siren_categorical import set_seed
from scene_coordinate_regression import load_manifest_rows


CONDITIONS = ("clean_true", "noisy_true", "clean_estimated", "noisy_estimated")
MODES = (
    "relu_scalar",
    "siren_scalar",
    "relu_categorical",
    "siren_categorical",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


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


def _check_scene_split(sample_sets: dict[str, Sequence[PhysicsSceneV2Sample]]) -> None:
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


def _balanced(
    samples: Sequence[PhysicsSceneV2Sample],
    enabled: bool,
) -> list[PhysicsSceneV2Sample]:
    """Balance fire contact surfaces by repeat-only sampling."""

    values = list(samples)
    if not enabled or not values:
        return values
    groups: dict[str, list[PhysicsSceneV2Sample]] = defaultdict(list)
    for sample in values:
        groups[sample.surface].append(sample)
    target = max(len(group) for group in groups.values())
    result: list[PhysicsSceneV2Sample] = []
    for surface in sorted(groups):
        group = groups[surface]
        repeats = int(math.ceil(target / len(group)))
        result.extend((group * repeats)[:target])
    return result


def _run_model(
    model: PhysicsSceneSirenCategorical,
    samples: Sequence[PhysicsSceneV2Sample],
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
    targets = np.asarray([sample.target_xyz for sample in samples], dtype=np.float32)
    entries: list[dict[str, Any]] = []
    probability = prediction["depth_probabilities"]
    bin_correct: list[float] = []
    depth_errors: list[float] = []
    for index, sample in enumerate(samples):
        sigma = float(np.linalg.norm(prediction["std_m"][index]))
        entropy = float(prediction["depth_entropy"][index])
        depth = float(prediction["depth_m"][index])
        depth_errors.append(abs(depth - float(sample.target_depth_m)))
        predicted_bin = None
        target_bin = None
        if model.depth_mode == "categorical" and probability.shape[1] == model.depth_bins:
            predicted_bin = int(np.argmax(probability[index]))
            target_bin = int(
                np.clip(
                    np.searchsorted(
                        model.depth_centers.detach().cpu().numpy(),
                        float(sample.target_depth_m),
                        side="left",
                    ),
                    0,
                    model.depth_bins - 1,
                )
            )
            bin_correct.append(float(predicted_bin == target_bin))
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
                "depth_m": depth,
                "target_depth_m": float(sample.target_depth_m),
                "depth_error_m": abs(depth - float(sample.target_depth_m)),
                "depth_entropy": entropy,
                "predicted_depth_bin": predicted_bin,
                "target_depth_bin": target_bin,
                "corrected_ray_xy": prediction["corrected_ray_xy"][index],
                "projected_pixel": prediction["projected_pixel"][index],
                "target_pixel": sample.target_pixel_under_camera,
                "reprojection_error_px": float(
                    np.linalg.norm(
                        prediction["projected_pixel"][index]
                        - sample.target_pixel_under_camera
                    )
                ),
                "latency_ms": per_sample_ms,
            }
        )
    return {
        "metrics": metric_summary(targets, prediction["xyz"]),
        "predictions": entries,
        "latency_ms": _describe([per_sample_ms] * len(samples)),
        "depth_error_m": _describe(depth_errors),
        "mean_depth_entropy": float(np.mean(prediction["depth_entropy"])),
        "depth_bin_accuracy": (
            float(np.mean(bin_correct)) if bin_correct else None
        ),
        "reprojection_error_px": _describe(item["reprojection_error_px"] for item in entries),
    }


def _surface_metrics(
    records: Sequence[dict[str, Any]],
    point_key: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for surface in sorted({str(item.get("surface", "unknown")) for item in records}):
        valid = [
            item
            for item in records
            if str(item.get("surface", "unknown")) == surface
            and item.get(point_key) is not None
        ]
        output[surface] = metric_summary(
            np.asarray([item["gt_xyz"] for item in valid], dtype=np.float32).reshape(-1, 3)
            if valid
            else np.empty((0, 3)),
            np.asarray([item[point_key] for item in valid], dtype=np.float32).reshape(-1, 3)
            if valid
            else np.empty((0, 3)),
        )
    return output


def _merge(
    model_result: dict[str, Any],
    baseline_result: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_id = {
        str(item["sample_id"]): item for item in baseline_result["records"]
    }
    merged: list[dict[str, Any]] = []
    for item in model_result["predictions"]:
        baseline = baseline_by_id[str(item["sample_id"])]
        merged.append(
            {
                **item,
                "physics_point": item["point"],
                "ray_point": baseline["ray_point"],
                "ipm_point": baseline["ipm_point"],
                "ray_status": baseline["ray_status"],
                "ipm_status": baseline["ipm_status"],
            }
        )
    return merged


def _route(records: Sequence[dict[str, Any]], threshold_m: float) -> dict[str, Any]:
    selected: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sources: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for record in records:
        point = None
        source = "not_localized"
        if (
            record.get("physics_point") is not None
            and float(record.get("sigma_norm_m", float("inf"))) <= threshold_m
        ):
            point = record["physics_point"]
            source = "scene_mlp"
        elif record.get("ray_point") is not None:
            point = record["ray_point"]
            source = "ray_casting"
        elif record.get("ipm_point") is not None:
            point = record["ipm_point"]
            source = "ipm_floor"
        if point is not None:
            selected.append(np.asarray(point, dtype=np.float32))
            targets.append(np.asarray(record["gt_xyz"], dtype=np.float32))
        sources[source] += 1
        output.append({**record, "selected_point": point, "selected_source": source})
    return {
        "metrics": metric_summary(
            np.asarray(targets, dtype=np.float32).reshape(-1, 3)
            if targets
            else np.empty((0, 3)),
            np.asarray(selected, dtype=np.float32).reshape(-1, 3)
            if selected
            else np.empty((0, 3)),
        ),
        "valid_rate": float(len(selected) / max(1, len(records))),
        "sources": dict(sources),
        "records": output,
    }


def _choose_threshold(
    records: Sequence[dict[str, Any]],
    thresholds: Iterable[float],
) -> dict[str, Any]:
    best: Optional[dict[str, Any]] = None
    for threshold in thresholds:
        routed = _route(records, float(threshold))
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


def _visualization_records(
    records: Sequence[dict[str, Any]],
    route: dict[str, Any],
) -> list[dict[str, Any]]:
    route_by_id = {str(item["sample_id"]): item for item in route["records"]}
    result: list[dict[str, Any]] = []
    for item in records:
        routed = route_by_id[str(item["sample_id"])]
        result.append(
            {
                "sample_id": item["sample_id"],
                "scene_id": item["scene_id"],
                "gt_xyz": item["gt_xyz"],
                "branches": {
                    "scene_mlp": {
                        "point": item["physics_point"],
                        "confidence": item["confidence"],
                        "uncertainty_norm_m": item["sigma_norm_m"],
                    },
                    "ray": {
                        "point": item["ray_point"],
                        "status": item["ray_status"],
                    },
                    "ipm": {
                        "point": item["ipm_point"],
                        "status": item["ipm_status"],
                    },
                    "fusion": {
                        "point": routed["selected_point"],
                        "source": routed["selected_source"],
                    },
                },
            }
        )
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "test_condition",
        "surface",
        "sample_count",
        "scene_mlp_mae_m",
        "scene_mlp_median_m",
        "scene_mlp_p95_m",
        "scene_mlp_under_0.25m",
        "scene_mlp_under_0.50m",
        "scene_mlp_under_1.00m",
        "scene_mlp_xyz_mae_m",
        "scene_mlp_latency_ms",
        "scene_mlp_depth_mae_m",
        "scene_mlp_depth_entropy",
        "scene_mlp_depth_bin_accuracy",
        "scene_mlp_reprojection_px",
        "ray_mae_m",
        "ray_median_m",
        "ray_p95_m",
        "ray_valid_rate",
        "ray_latency_ms",
        "ipm_mae_m",
        "ipm_valid_rate",
        "route_mae_m",
        "route_valid_rate",
        "gate_threshold_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_reference(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, Any] = {"available": True, "path": str(path), "results": {}}
    results = data.get("results", {})
    if "mixed" in results:
        output["results"]["scene_mlp_v1_mixed"] = {
            condition: details.get("physics", {})
            for condition, details in results["mixed"].get("test_conditions", {}).items()
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "histories", "predictions", "visualization_summaries"):
        (output / name).mkdir(parents=True, exist_ok=True)

    rows = {split: load_manifest_rows(dataset, split) for split in ("train", "val", "test")}
    all_samples: dict[str, dict[str, list[PhysicsSceneV2Sample]]] = {}
    skipped: dict[str, dict[str, dict[str, int]]] = {}
    for condition in CONDITIONS:
        all_samples[condition] = {}
        skipped[condition] = {}
        for split in ("train", "val", "test"):
            all_samples[condition][split], skipped[condition][split] = build_samples(
                rows[split], condition, canonicalize=True
            )

    mesh_path = dataset / "room_mesh.json"
    mesh_data = json.loads(mesh_path.read_text(encoding="utf-8"))
    from locator import TriangleMesh

    mesh = TriangleMesh(mesh_data["vertices"], mesh_data["faces"])
    thresholds = np.linspace(
        float(args.min_uncertainty_m),
        float(args.max_uncertainty_m),
        max(2, int(args.threshold_steps)),
    )

    train_pool = [
        sample
        for condition in CONDITIONS
        for sample in all_samples[condition]["train"]
    ]
    val_pool = [
        sample
        for condition in CONDITIONS
        for sample in all_samples[condition]["val"]
    ]
    test_pool = {
        condition: all_samples[condition]["test"] for condition in CONDITIONS
    }
    _check_scene_split(
        {"train": train_pool, "val": val_pool, "test": [sample for values in test_pool.values() for sample in values]}
    )
    train_pool = _balanced(train_pool, args.balance_surfaces)

    results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for mode in MODES:
        backbone = "siren" if mode.startswith("siren") else "relu"
        depth_mode = "categorical" if mode.endswith("categorical") else "scalar"
        # Seed immediately before construction so every ablation gets a
        # reproducible, mode-independent initialization.  ``fit`` seeds the
        # dataloader and training operations too, but doing it only there is
        # too late: the first model's random initialization would otherwise
        # consume the RNG stream before later modes are created.
        set_seed(args.seed)
        model = PhysicsSceneSirenCategorical(
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            backbone=backbone,
            depth_mode=depth_mode,
            depth_bins=args.depth_bins,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            omega_0=args.omega_0,
            device=args.device,
        )
        print(
            f"[siren-categorical] mode={mode} train={len(train_pool)} "
            f"val={len(val_pool)} depth_mode={depth_mode} backbone={backbone}",
            flush=True,
        )
        history = model.fit(
            train_pool,
            val_pool,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        checkpoint = output / "checkpoints" / f"scene_{mode}.pth"
        model.save(
            checkpoint,
            metadata={
                "mode": mode,
                "train_conditions": list(CONDITIONS),
                "surface_balanced": bool(args.balance_surfaces),
                "dataset": str(dataset),
            },
        )
        (output / "histories" / f"scene_{mode}.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

        validation_model = _run_model(model, val_pool)
        validation_baseline = _run_ray_ipm(val_pool, mesh)
        selected_gate = _choose_threshold(
            _merge(validation_model, validation_baseline), thresholds
        )
        mode_result: dict[str, Any] = {
            "backbone": backbone,
            "depth_mode": depth_mode,
            "train_conditions": list(CONDITIONS),
            "train_samples": len(train_pool),
            "val_samples": len(val_pool),
            "checkpoint": str(checkpoint),
            "epochs_ran": len(history),
            "best_val_mae_m": min((item["val_mae_m"] for item in history), default=None),
            "gate_selected_on_validation": selected_gate,
            "test_conditions": {},
        }

        for condition in CONDITIONS:
            samples = test_pool[condition]
            print(
                f"[siren-categorical-test] mode={mode} condition={condition} "
                f"samples={len(samples)}",
                flush=True,
            )
            model_result = _run_model(model, samples)
            baseline_result = _run_ray_ipm(samples, mesh)
            merged = _merge(model_result, baseline_result)
            route = _route(merged, float(selected_gate["threshold_m"]))
            visualization_path = output / "visualization_summaries" / f"{mode}_{condition}.json"
            visualization_path.write_text(
                json.dumps(
                    {
                        "format": "LAB_SAM.siren_categorical_visualization.v1",
                        "dataset": {
                            "synthetic_metric_geometry": True,
                            "warning": "Synthetic/asset-backed geometry, not real CCTV accuracy.",
                        },
                        "method": {"mesh": str(mesh_path), "mode": mode},
                        "records": _visualization_records(merged, route),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            prediction_path = output / "predictions" / f"{mode}_{condition}.json"
            prediction_path.write_text(
                json.dumps(
                    {
                        "model": model_result,
                        "baselines": baseline_result,
                        "route": route,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )
            physics_metrics = model_result["metrics"]
            ray_metrics = baseline_result["ray"]["metrics"]
            ipm_metrics = baseline_result["ipm"]["metrics"]
            mode_result["test_conditions"][condition] = {
                "sample_count": len(samples),
                "scene_mlp": {
                    **physics_metrics,
                    "latency_ms": model_result["latency_ms"],
                    "depth_error_m": model_result["depth_error_m"],
                    "mean_depth_entropy": model_result["mean_depth_entropy"],
                    "depth_bin_accuracy": model_result["depth_bin_accuracy"],
                    "reprojection_error_px": model_result["reprojection_error_px"],
                    "by_surface": _surface_metrics(merged, "physics_point"),
                },
                "ray": {
                    **ray_metrics,
                    "valid_rate": baseline_result["ray"]["valid_rate"],
                    "latency_ms": baseline_result["ray"]["latency_ms"],
                    "by_surface": _surface_metrics(baseline_result["records"], "ray_point"),
                },
                "ipm_floor_only": {
                    **ipm_metrics,
                    "valid_rate": baseline_result["ipm"]["valid_rate"],
                    "latency_ms": baseline_result["ipm"]["latency_ms"],
                },
                "route": {
                    "metrics": route["metrics"],
                    "valid_rate": route["valid_rate"],
                    "sources": route["sources"],
                    "threshold_m": selected_gate["threshold_m"],
                },
                "visualization_summary": str(visualization_path),
            }
            csv_rows.append(
                {
                    "mode": mode,
                    "test_condition": condition,
                    "surface": "all",
                    "sample_count": len(samples),
                    "scene_mlp_mae_m": physics_metrics.get("mae_m"),
                    "scene_mlp_median_m": physics_metrics.get("median_m"),
                    "scene_mlp_p95_m": physics_metrics.get("p95_m"),
                    "scene_mlp_under_0.25m": physics_metrics.get("under_0.25m"),
                    "scene_mlp_under_0.50m": physics_metrics.get("under_0.50m"),
                    "scene_mlp_under_1.00m": physics_metrics.get("under_1.00m"),
                    "scene_mlp_xyz_mae_m": physics_metrics.get("xyz_mae_m"),
                    "scene_mlp_latency_ms": model_result["latency_ms"].get("mean"),
                    "scene_mlp_depth_mae_m": model_result["depth_error_m"].get("mean"),
                    "scene_mlp_depth_entropy": model_result["mean_depth_entropy"],
                    "scene_mlp_depth_bin_accuracy": model_result["depth_bin_accuracy"],
                    "scene_mlp_reprojection_px": model_result["reprojection_error_px"].get("mean"),
                    "ray_mae_m": ray_metrics.get("mae_m"),
                    "ray_median_m": ray_metrics.get("median_m"),
                    "ray_p95_m": ray_metrics.get("p95_m"),
                    "ray_valid_rate": baseline_result["ray"]["valid_rate"],
                    "ray_latency_ms": baseline_result["ray"]["latency_ms"].get("mean"),
                    "ipm_mae_m": ipm_metrics.get("mae_m"),
                    "ipm_valid_rate": baseline_result["ipm"]["valid_rate"],
                    "route_mae_m": route["metrics"].get("mae_m"),
                    "route_valid_rate": route["valid_rate"],
                    "gate_threshold_m": selected_gate["threshold_m"],
                }
            )
        results[mode] = mode_result

    metrics_path = output / "siren_categorical_metrics.csv"
    _write_csv(metrics_path, csv_rows)
    summary = {
        "format": "LAB_SAM.siren_categorical_scene_benchmark.v1",
        "dataset": {
            "root": str(dataset),
            "mesh": str(mesh_path),
            "raw_records": {split: len(rows[split]) for split in rows},
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry results, not real CCTV accuracy.",
            "split_policy": "scene-separated train/val/test; no adjacent-frame leakage",
        },
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "data": {
            "train_samples_before_surface_balance": len(
                [sample for condition in CONDITIONS for sample in all_samples[condition]["train"]]
            ),
            "train_samples_after_surface_balance": len(train_pool),
            "val_samples": len(val_pool),
            "test_samples": {condition: len(test_pool[condition]) for condition in CONDITIONS},
            "surface_balance": bool(args.balance_surfaces),
        },
        "model_design": {
            "siren": "sin(omega_0 * Linear(x)) with SIREN initialization",
            "categorical_depth": "softmax depth bins; expected depth feeds the pinhole physical layer",
            "depth_range_m": [args.depth_min_m, args.depth_max_m],
            "depth_bins": args.depth_bins,
            "physical_layer": "C + R_world_to_camera.T @ ([corrected_ray_x, corrected_ray_y, 1] * depth)",
            "loss": "XYZ + depth/cross-entropy + ray + pixel reprojection + heteroscedastic residual",
            "ray_casting": "same pixel, calibration and mesh for every learned mode",
        },
        "skipped": skipped,
        "references": {
            "previous_scene_mlp_v1": _load_reference(
                Path(args.reference_v1).expanduser().resolve()
            ),
            "previous_physics_v2": str(args.reference_v2),
        },
        "results": results,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "checkpoints": str(output / "checkpoints"),
            "visualization_summaries": str(output / "visualization_summaries"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print("mode                  condition          Scene-MLP MAE   Ray MAE   Ray hit")
    for mode in MODES:
        for condition in CONDITIONS:
            item = results[mode]["test_conditions"][condition]
            print(
                f"{mode:<21} {condition:<18} "
                f"{item['scene_mlp']['mae_m']!s:>13} "
                f"{item['ray']['mae_m']!s:>9} "
                f"{item['ray']['valid_rate']:.3f}",
                flush=True,
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
        default=root / "output" / "siren_categorical_scene_20261008",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--depth-bins", type=int, default=64)
    parser.add_argument("--depth-min-m", type=float, default=6.0)
    parser.add_argument("--depth-max-m", type=float, default=20.0)
    parser.add_argument("--omega-0", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-surfaces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-uncertainty-m", type=float, default=0.10)
    parser.add_argument("--max-uncertainty-m", type=float, default=3.00)
    parser.add_argument("--threshold-steps", type=int, default=15)
    parser.add_argument(
        "--reference-v1",
        type=Path,
        default=root / "output" / "physics_scene_cross_20261008_v1" / "summary.json",
    )
    parser.add_argument(
        "--reference-v2",
        type=Path,
        default=root / "output" / "physics_scene_cross_20261008_v2_fair" / "summary.json",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch-size and hidden-dim must be positive")
    if args.hidden_layers <= 0 or args.depth_bins < 2:
        raise ValueError("hidden-layers and depth-bins must be positive")
    if args.threshold_steps < 2 or args.min_uncertainty_m <= 0 or args.max_uncertainty_m <= args.min_uncertainty_m:
        raise ValueError("uncertainty threshold range is invalid")
    run(args)


if __name__ == "__main__":
    main()
