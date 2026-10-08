"""Strict post-ROI 3D ablations for the low-cost synthetic experiment.

The experiment separates two error sources for both a learned scene-coordinate
mapper and multi-view triangulation:

    pixel:  p_fire_pixel          vs p_fire_noisy_pixel
    camera: camera (true pose)    vs camera_estimated (calibration noise)

The four combinations are evaluated on the same scene-separated test split.
Metrics are also broken down by ``fire_surface`` so that a good floor result
cannot hide failures on a table, cabinet or column.

This file intentionally writes to a new output directory and never edits or
deletes the source dataset, checkpoints or older benchmark artifacts.
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
from scene_coordinate_regression import (
    FEATURE_NAMES,
    SceneCoordinateMLP,
    SceneCoordinateSample,
    make_feature_vector,
    metric_summary,
)
from triangulation_3d import triangulate_dlt


VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("clean_true", "p_fire_pixel", "true"),
    ("noisy_true", "p_fire_noisy_pixel", "true"),
    ("clean_estimated", "p_fire_pixel", "estimated"),
    ("noisy_estimated", "p_fire_noisy_pixel", "estimated"),
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
    rows: list[dict[str, Any]] = []
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
    """Return only the requested observation; never substitute clean labels."""

    pixel = _point(row.get(key), 2)
    image_size = _point(row.get("image_size", [640, 640]), 2)
    if pixel is None or image_size is None or np.any(image_size <= 0):
        return None
    width, height = image_size
    if np.allclose(pixel, 0.0, atol=1e-12):
        return None
    if not (-0.5 <= pixel[0] < width + 0.5 and -0.5 <= pixel[1] < height + 0.5):
        return None
    return pixel


def _calibration(row: dict[str, Any], source: str) -> Optional[CameraCalibration]:
    value = row.get("camera" if source == "true" else "camera_estimated")
    if not isinstance(value, dict):
        return None
    try:
        return CameraCalibration.from_dict(
            {
                "image_size": row.get("image_size"),
                "intrinsics": {
                    "K": value.get("K"),
                    "dist_coeffs": value.get("dist_coeffs", []),
                },
                "extrinsics": {
                    "R": value.get("R_world_to_camera", value.get("R")),
                    "camera_position": value.get("camera_position"),
                },
            }
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _valid_samples(
    rows: Iterable[dict[str, Any]],
    pixel_key: str,
    camera_source: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build strict samples for one ablation condition."""

    samples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            skipped["not_visible_fire"] += 1
            continue
        pixel = _strict_pixel(row, pixel_key)
        target = _point(row.get("fire_xyz_world"), 3)
        calibration = _calibration(row, camera_source)
        if pixel is None:
            skipped["missing_or_invalid_pixel"] += 1
            continue
        if target is None:
            skipped["missing_xyz"] += 1
            continue
        if calibration is None:
            skipped["invalid_camera"] += 1
            continue
        features = make_feature_vector(row, pixel, camera_source)
        if features is None:
            skipped["invalid_features"] += 1
            continue
        samples.append(
            {
                "row": row,
                "pixel": pixel,
                "target": target,
                "camera": calibration,
                "features": features,
                "scene_id": str(row.get("scene_id", "unknown")),
                "surface": str(row.get("fire_surface", "unknown")),
            }
        )
    return samples, dict(skipped)


def _check_scene_split(sample_sets: dict[str, list[dict[str, Any]]]) -> None:
    scene_sets = {
        split: {sample["scene_id"] for sample in samples}
        for split, samples in sample_sets.items()
    }
    overlap = (scene_sets["train"] & scene_sets["val"]) | (
        scene_sets["train"] & scene_sets["test"]
    ) | (scene_sets["val"] & scene_sets["test"])
    if overlap:
        raise RuntimeError(f"Scene leakage detected: {sorted(overlap)[:10]}")


def _surface_counts(samples: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(sample["surface"] for sample in samples).items()))


def _as_mlp_samples(samples: Iterable[dict[str, Any]]) -> list[SceneCoordinateSample]:
    """Adapt benchmark dictionaries to the shared MLP training contract."""

    return [
        SceneCoordinateSample(
            row=sample["row"],
            features=np.asarray(sample["features"], dtype=np.float32),
            target_xyz=np.asarray(sample["target"], dtype=np.float32),
            pixel=np.asarray(sample["pixel"], dtype=np.float32),
            scene_id=str(sample["scene_id"]),
        )
        for sample in samples
    ]


def _balanced_train_samples(samples: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    """Moderately oversample rare surfaces without changing validation/test."""

    if not enabled or not samples:
        return samples
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["surface"]].append(sample)
    target = max(len(group) for group in groups.values())
    balanced: list[dict[str, Any]] = []
    for surface in sorted(groups):
        group = groups[surface]
        repeats = int(math.ceil(target / len(group)))
        balanced.extend((group * repeats)[:target])
    return balanced


def _metrics(
    entries: list[tuple[np.ndarray, Optional[np.ndarray], float]],
) -> dict[str, Any]:
    valid = [(truth, prediction, latency) for truth, prediction, latency in entries if prediction is not None]
    if valid:
        truth = np.asarray([item[0] for item in valid], dtype=np.float64)
        prediction = np.asarray([item[1] for item in valid], dtype=np.float64)
        result = metric_summary(truth, prediction).to_dict()
    else:
        result = metric_summary(np.empty((0, 3)), np.empty((0, 3))).to_dict()
    latencies = [float(item[2]) for item in entries]
    result["attempted"] = len(entries)
    result["valid_count"] = len(valid)
    result["valid_rate"] = float(len(valid) / max(1, len(entries)))
    result["latency_ms"] = {
        "mean": None if not latencies else float(np.mean(latencies)),
        "median": None if not latencies else float(np.median(latencies)),
        "p95": None if not latencies else float(np.percentile(latencies, 95)),
    }
    return result


def _by_surface(
    entries: list[tuple[str, np.ndarray, Optional[np.ndarray], float]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[np.ndarray, Optional[np.ndarray], float]]] = defaultdict(list)
    for surface, target, prediction, latency in entries:
        grouped[surface].append((target, prediction, latency))
    return {surface: _metrics(values) for surface, values in sorted(grouped.items())}


def _rotation_error_deg(true_R: np.ndarray, estimated_R: np.ndarray) -> float:
    relative = np.asarray(estimated_R, dtype=np.float64) @ np.asarray(true_R, dtype=np.float64).T
    trace_value = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace_value)))


def _calibration_noise_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    focal_errors: list[float] = []
    principal_errors: list[float] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row.get("scene_id", "unknown")), int(row.get("frame_index", 0)))
        if key in seen:
            continue
        seen.add(key)
        true = row.get("camera")
        estimated = row.get("camera_estimated")
        if not isinstance(true, dict) or not isinstance(estimated, dict):
            continue
        try:
            true_position = np.asarray(true["camera_position"], dtype=np.float64).reshape(3)
            estimated_position = np.asarray(estimated["camera_position"], dtype=np.float64).reshape(3)
            true_R = np.asarray(true["R_world_to_camera"], dtype=np.float64).reshape(3, 3)
            estimated_R = np.asarray(estimated["R_world_to_camera"], dtype=np.float64).reshape(3, 3)
            true_K = np.asarray(true["K"], dtype=np.float64).reshape(3, 3)
            estimated_K = np.asarray(estimated["K"], dtype=np.float64).reshape(3, 3)
        except (KeyError, TypeError, ValueError):
            continue
        position_errors.append(float(np.linalg.norm(estimated_position - true_position)))
        rotation_errors.append(_rotation_error_deg(true_R, estimated_R))
        focal_errors.append(float(abs(estimated_K[0, 0] - true_K[0, 0]) / max(1e-9, true_K[0, 0])))
        principal_errors.append(
            float(np.linalg.norm(estimated_K[:2, 2] - true_K[:2, 2]))
        )

    def describe(values: list[float], scale: float = 1.0) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "p95": None}
        array = np.asarray(values, dtype=np.float64) * scale
        return {
            "count": len(array),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
        }

    return {
        "camera_position_error_m": describe(position_errors),
        "rotation_error_deg": describe(rotation_errors),
        "focal_relative_error_pct": describe(focal_errors, 100.0),
        "principal_point_error_px": describe(principal_errors),
    }


def _run_mlp_variant(
    variant: str,
    pixel_key: str,
    camera_source: str,
    rows: dict[str, list[dict[str, Any]]],
    output: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_sets: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        sample_sets[split], skipped[split] = _valid_samples(rows[split], pixel_key, camera_source)
        if pixel_key == "p_fire_pixel":
            # Pixel-clean and pixel-noisy comparisons must use the exact same
            # images. A clean label is not allowed to gain extra samples when
            # the synthetic detector missed its noisy observation.
            noisy_samples, _ = _valid_samples(
                rows[split], "p_fire_noisy_pixel", camera_source
            )
            noisy_ids = {
                str(sample["row"].get("sample_id", "")) for sample in noisy_samples
            }
            before = len(sample_sets[split])
            sample_sets[split] = [
                sample
                for sample in sample_sets[split]
                if str(sample["row"].get("sample_id", "")) in noisy_ids
            ]
            skipped[split]["clean_sample_not_observed_by_noisy_detector"] = (
                before - len(sample_sets[split])
            )
    _check_scene_split(sample_sets)
    if len(sample_sets["train"]) < 2 or not sample_sets["val"] or not sample_sets["test"]:
        raise RuntimeError(f"Not enough strict samples for MLP {variant}: { {k: len(v) for k, v in sample_sets.items()} }")

    train_for_fit = _balanced_train_samples(sample_sets["train"], args.balance_surfaces)
    train_mlp_samples = _as_mlp_samples(train_for_fit)
    val_mlp_samples = _as_mlp_samples(sample_sets["val"])
    model = SceneCoordinateMLP(hidden_dim=args.hidden_dim, device=args.device)
    history = model.fit(
        train_mlp_samples,
        val_mlp_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    checkpoint = output / "checkpoints" / f"scene_mlp_{variant}.pth"
    model.save(
        checkpoint,
        metadata={
            "variant": variant,
            "pixel_key": pixel_key,
            "camera_source": camera_source,
            "surface_balanced": bool(args.balance_surfaces),
        },
    )
    (output / "histories" / f"scene_mlp_{variant}.json").write_text(
        json.dumps(history, indent=2, default=_json_default), encoding="utf-8"
    )

    entries: list[tuple[str, np.ndarray, Optional[np.ndarray], float]] = []
    predictions: list[dict[str, Any]] = []
    for sample in sample_sets["test"]:
        started = time.perf_counter()
        prediction = model.predict_features(sample["features"].reshape(1, -1))[0]
        latency = (time.perf_counter() - started) * 1000.0
        entries.append((sample["surface"], sample["target"], prediction, latency))
        predictions.append(
            {
                "variant": variant,
                "sample_id": sample["row"].get("sample_id"),
                "scene_id": sample["scene_id"],
                "surface": sample["surface"],
                "pixel_key": pixel_key,
                "camera_source": camera_source,
                "input_pixel": sample["pixel"],
                "gt_xyz": sample["target"],
                "pred_xyz": prediction,
                "latency_ms": latency,
            }
        )

    ungrouped = [(target, prediction, latency) for _, target, prediction, latency in entries]
    result = {
        "pixel_key": pixel_key,
        "camera_source": camera_source,
        "strict_samples": {split: len(sample_sets[split]) for split in sample_sets},
        "strict_scenes": {
            split: len({sample["scene_id"] for sample in sample_sets[split]})
            for split in sample_sets
        },
        "skipped": skipped,
        "surface_counts": {split: _surface_counts(sample_sets[split]) for split in sample_sets},
        "train_samples_after_surface_balance": len(train_mlp_samples),
        "checkpoint": str(checkpoint),
        "epochs_ran": len(history),
        "best_val_mae_m": min((item["val_mae_m"] for item in history), default=None),
        "metrics": _metrics(ungrouped),
        "metrics_by_surface": _by_surface(entries),
    }
    return {"variant": variant, "mlp": result}, predictions


def _triangulation_variant(
    variant: str,
    pixel_key: str,
    camera_source: str,
    rows: list[dict[str, Any]],
    min_baseline_m: float,
    max_reprojection_px: float,
    max_condition_number: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            continue
        pixel = _strict_pixel(row, pixel_key)
        target = _point(row.get("fire_xyz_world"), 3)
        camera = _calibration(row, camera_source)
        if pixel is None or target is None or camera is None:
            skipped["missing_strict_view"] += 1
            continue
        grouped[str(row.get("scene_id", "unknown"))].append(
            {
                "row": row,
                "pixel": pixel,
                "target": target,
                "camera": camera,
                "surface": str(row.get("fire_surface", "unknown")),
            }
        )

    raw_entries: list[tuple[str, np.ndarray, Optional[np.ndarray], float]] = []
    gated_entries: list[tuple[str, np.ndarray, Optional[np.ndarray], float]] = []
    records: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()
    gate_reasons: Counter[str] = Counter()
    baseline_values: list[float] = []
    reprojection_values: list[float] = []
    condition_values: list[float] = []

    for scene_id, views in sorted(grouped.items()):
        views.sort(key=lambda item: int(item["row"].get("frame_index", 0)))
        surface = Counter(view["surface"] for view in views).most_common(1)[0][0]
        target = views[0]["target"]
        if any(np.linalg.norm(view["target"] - target) > 1e-4 for view in views[1:]):
            failure_reasons["non_static_ground_truth"] += 1
            continue
        if len(views) < 2:
            failure_reasons["fewer_than_two_views"] += 1
            continue
        positions = np.asarray([view["camera"].camera_position() for view in views])
        pairwise = [
            float(np.linalg.norm(positions[i] - positions[j]))
            for i in range(len(views))
            for j in range(i + 1, len(views))
        ]
        baseline = max(pairwise, default=0.0)
        started = time.perf_counter()
        estimate = triangulate_dlt(
            [view["pixel"] for view in views],
            [view["camera"] for view in views],
        )
        latency = (time.perf_counter() - started) * 1000.0
        raw_point = estimate.point if estimate.success else None
        raw_entries.append((surface, target, raw_point, latency))
        if estimate.reprojection_rmse_px is not None:
            reprojection_values.append(float(estimate.reprojection_rmse_px))
        if estimate.condition_number is not None and np.isfinite(estimate.condition_number):
            condition_values.append(float(estimate.condition_number))
        baseline_values.append(baseline)
        if not estimate.success:
            failure_reasons[str(estimate.reason or "triangulation_failed")] += 1

        gate_reason: Optional[str] = None
        if baseline < min_baseline_m:
            gate_reason = "baseline_below_minimum"
        elif estimate.point is None or not estimate.success:
            gate_reason = str(estimate.reason or "triangulation_failed")
        elif estimate.reprojection_rmse_px is not None and estimate.reprojection_rmse_px > max_reprojection_px:
            gate_reason = "reprojection_above_maximum"
        elif estimate.condition_number is not None and estimate.condition_number > max_condition_number:
            gate_reason = "condition_number_above_maximum"
        if gate_reason is not None:
            gate_reasons[gate_reason] += 1
            gated_point = None
        else:
            gated_point = estimate.point
        gated_entries.append((surface, target, gated_point, latency))
        records.append(
            {
                "variant": variant,
                "scene_id": scene_id,
                "surface": surface,
                "pixel_key": pixel_key,
                "camera_source": camera_source,
                "views": len(views),
                "baseline_max_m": baseline,
                "reprojection_rmse_px": estimate.reprojection_rmse_px,
                "condition_number": estimate.condition_number,
                "null_space_gap": estimate.null_space_gap,
                "positive_depth_rate": estimate.positive_depth_rate,
                "raw_success": bool(estimate.success),
                "raw_point": estimate.point,
                "gated_point": gated_point,
                "gate_reason": gate_reason,
                "gt_xyz": target,
                "latency_ms": latency,
            }
        )

    raw_ungrouped = [(target, prediction, latency) for _, target, prediction, latency in raw_entries]
    gated_ungrouped = [(target, prediction, latency) for _, target, prediction, latency in gated_entries]
    result = {
        "pixel_key": pixel_key,
        "camera_source": camera_source,
        "scenes_attempted": len(raw_entries),
        "strict_views": sum(len(values) for values in grouped.values()),
        "skipped": dict(skipped),
        "metrics_raw": _metrics(raw_ungrouped),
        "metrics_raw_by_surface": _by_surface(raw_entries),
        "metrics_gated": _metrics(gated_ungrouped),
        "metrics_gated_by_surface": _by_surface(gated_entries),
        "gates": {
            "min_baseline_m": min_baseline_m,
            "max_reprojection_rmse_px": max_reprojection_px,
            "max_condition_number": max_condition_number,
        },
        "baseline_max_m": _describe(baseline_values),
        "reprojection_rmse_px": _describe(reprojection_values),
        "condition_number": _describe(condition_values),
        "failure_reasons": dict(failure_reasons),
        "gate_reasons": dict(gate_reasons),
    }
    return {"variant": variant, "triangulation": result}, records


def _describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def _write_csv(path: Path, results: dict[str, Any]) -> None:
    fields = [
        "variant",
        "mlp_mae_m",
        "mlp_median_m",
        "mlp_p95_m",
        "mlp_valid_rate",
        "tri_raw_mae_m",
        "tri_raw_median_m",
        "tri_raw_p95_m",
        "tri_raw_valid_rate",
        "tri_gated_mae_m",
        "tri_gated_median_m",
        "tri_gated_p95_m",
        "tri_gated_valid_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant in (item[0] for item in VARIANTS):
            item = results[variant]
            mlp = item["mlp"]["metrics"]
            tri_raw = item["triangulation"]["metrics_raw"]
            tri_gated = item["triangulation"]["metrics_gated"]
            writer.writerow(
                {
                    "variant": variant,
                    "mlp_mae_m": mlp.get("mae_m"),
                    "mlp_median_m": mlp.get("median_m"),
                    "mlp_p95_m": mlp.get("p95_m"),
                    "mlp_valid_rate": mlp.get("valid_rate"),
                    "tri_raw_mae_m": tri_raw.get("mae_m"),
                    "tri_raw_median_m": tri_raw.get("median_m"),
                    "tri_raw_p95_m": tri_raw.get("p95_m"),
                    "tri_raw_valid_rate": tri_raw.get("valid_rate"),
                    "tri_gated_mae_m": tri_gated.get("mae_m"),
                    "tri_gated_median_m": tri_gated.get("median_m"),
                    "tri_gated_p95_m": tri_gated.get("p95_m"),
                    "tri_gated_valid_rate": tri_gated.get("valid_rate"),
                }
            )


def _plot_results(output: Path, results: dict[str, Any]) -> list[str]:
    """Write compact scientific plots when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [item[0] for item in VARIANTS]
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.25

    def values(path: tuple[str, ...]) -> list[float]:
        values_list: list[float] = []
        for label in labels:
            value: Any = results[label]
            for key in path:
                value = value[key]
            values_list.append(np.nan if value is None else float(value))
        return values_list

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width, values(("mlp", "metrics", "mae_m")), width, label="Scene-MLP")
    axis.bar(positions, values(("triangulation", "metrics_raw", "mae_m")), width, label="Triangulation raw")
    axis.bar(positions + width, values(("triangulation", "metrics_gated", "mae_m")), width, label="Triangulation gated")
    axis.set_xticks(positions, labels, rotation=20)
    axis.set_ylabel("3D MAE (m)")
    axis.set_title("Pixel / camera ablation on the same held-out scenes")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    mae_path = figure_dir / "ablation_mae.png"
    fig.savefig(mae_path, dpi=160)
    plt.close(fig)

    surfaces = sorted(
        {
            surface
            for label in labels
            for surface in results[label]["mlp"]["metrics_by_surface"]
        }
    )
    matrix = np.full((len(surfaces), len(labels)), np.nan, dtype=np.float64)
    for column, label in enumerate(labels):
        for row, surface in enumerate(surfaces):
            value = results[label]["mlp"]["metrics_by_surface"].get(surface, {}).get("mae_m")
            if value is not None:
                matrix[row, column] = float(value)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=20)
    axis.set_yticks(np.arange(len(surfaces)), surfaces)
    axis.set_title("Scene-MLP 3D MAE by fire surface")
    axis.set_xlabel("Ablation")
    axis.set_ylabel("Surface")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="MAE (m)")
    fig.tight_layout()
    surface_path = figure_dir / "scene_mlp_surface_mae.png"
    fig.savefig(surface_path, dpi=160)
    plt.close(fig)
    return [str(mae_path), str(surface_path)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "histories").mkdir(parents=True, exist_ok=True)

    rows = {split: _load_rows(dataset, split) for split in ("train", "val", "test")}
    calibration_noise = _calibration_noise_summary(rows["test"])
    results: dict[str, Any] = {}
    mlp_predictions: list[dict[str, Any]] = []
    triangulation_records: list[dict[str, Any]] = []

    for variant, pixel_key, camera_source in VARIANTS:
        print(f"[MLP] {variant}: pixel={pixel_key} camera={camera_source}", flush=True)
        mlp_result, predictions = _run_mlp_variant(
            variant, pixel_key, camera_source, rows, output, args
        )
        print(f"[triangulation] {variant}: pixel={pixel_key} camera={camera_source}", flush=True)
        tri_result, records = _triangulation_variant(
            variant,
            pixel_key,
            camera_source,
            rows["test"],
            args.min_baseline_m,
            args.max_reprojection_px,
            args.max_condition_number,
        )
        results[variant] = {**mlp_result, **tri_result}
        mlp_predictions.extend(predictions)
        triangulation_records.extend(records)
        print(
            f"  MLP MAE={results[variant]['mlp']['metrics']['mae_m']}m "
            f"Tri gated MAE={results[variant]['triangulation']['metrics_gated']['mae_m']}m "
            f"valid={results[variant]['triangulation']['metrics_gated']['valid_rate']:.3f}",
            flush=True,
        )

    summary = {
        "format": "LAB_SAM.post_roi_3d_ablation.v1",
        "dataset": {
            "root": str(dataset),
            "synthetic_metric_geometry": True,
            "warning": "These are synthetic/asset-backed geometry results, not real CCTV accuracy.",
            "split_policy": "scene-separated train/val/test; no adjacent-frame leakage",
            "raw_records": {split: len(rows[split]) for split in rows},
        },
        "ablation": {
            "pixel_clean": "p_fire_pixel",
            "pixel_noisy": "p_fire_noisy_pixel; missing observations are excluded",
            "camera_true": "camera",
            "camera_estimated": "camera_estimated",
            "surface_breakdown": ["floor", "left_table", "central_cabinet", "right_column"],
            "calibration_noise_on_test": calibration_noise,
        },
        "training": {
            "mlp_epochs_requested": args.epochs,
            "mlp_batch_size": args.batch_size,
            "mlp_hidden_dim": args.hidden_dim,
            "mlp_surface_balanced": bool(args.balance_surfaces),
            "device": args.device,
            "feature_names": list(FEATURE_NAMES),
        },
        "triangulation_policy": {
            "min_baseline_m": args.min_baseline_m,
            "max_reprojection_rmse_px": args.max_reprojection_px,
            "max_condition_number": args.max_condition_number,
            "raw_and_gated_metrics_are_both_reported": True,
            "note": "A gated triangulation result is rejected when baseline, reprojection or numerical conditioning is unsafe.",
        },
        "variants": results,
        "artifacts": {
            "mlp_predictions": str(output / "mlp_predictions.json"),
            "triangulation_records": str(output / "triangulation_records.json"),
        },
    }
    figure_paths = _plot_results(output, results)
    summary["artifacts"]["figures"] = figure_paths
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (output / "mlp_predictions.json").write_text(
        json.dumps(mlp_predictions, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (output / "triangulation_records.json").write_text(
        json.dumps(triangulation_records, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_csv(output / "ablation_metrics.csv", results)

    print("variant          MLP_MAE  TRI_RAW_MAE  TRI_GATED_MAE  TRI_GATED_VALID")
    for variant, _, _ in VARIANTS:
        item = results[variant]
        mlp = item["mlp"]["metrics"]
        raw = item["triangulation"]["metrics_raw"]
        gated = item["triangulation"]["metrics_gated"]
        print(
            f"{variant:<16} {str(mlp['mae_m']):>7} {str(raw['mae_m']):>12} "
            f"{str(gated['mae_m']):>14} {gated['valid_rate']:.3f}"
        )
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ablation_3d_20261008_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance-surfaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample rare surfaces in MLP training only; validation/test stay untouched.",
    )
    parser.add_argument("--min-baseline-m", type=float, default=0.05)
    parser.add_argument("--max-reprojection-px", type=float, default=8.0)
    parser.add_argument("--max-condition-number", type=float, default=1e12)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch-size and hidden-dim must be positive")
    if args.min_baseline_m < 0 or args.max_reprojection_px <= 0 or args.max_condition_number <= 0:
        raise ValueError("triangulation gates must be positive")
    run(args)


if __name__ == "__main__":
    main()

