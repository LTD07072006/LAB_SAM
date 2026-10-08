"""Benchmark Physics Scene-MLP v2 against mesh Ray Casting.

This runner is intentionally independent from ``benchmark_physics_scene_mlp.py``
so the v1 results remain reproducible.  It evaluates the same visible-fire
test records under four post-ROI conditions and compares:

* ``pinhole_only``: physical layer, no domain augmentation, no geometry prior;
* ``pinhole_domain``: canonical camera representation plus geometry-preserving
  camera/pixel domain augmentation;
* ``pinhole_domain_geometry``: the previous branch plus geometry-error prior
  and pixel reprojection loss;
* ``pinhole_full_mixed_geometry``: strongest branch, trained on clean/noisy and
  true/estimated observations plus domain augmentation.

Ray Casting is evaluated on the exact same samples.  IPM is deliberately
limited to ``fire_surface == floor`` because a floor homography is invalid for
table, cabinet and column contact points.

The benchmark is synthetic/asset-backed metric geometry.  It is not a claim of
real CCTV accuracy.  Auxiliary TUM RGB-D data are audited separately because
they provide depth/pose but no fire-source XYZ labels.
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

from camera_calibration import CameraCalibration
from homography_floor import FloorHomography
from locator import TriangleMesh, intersect_ray_with_grid_result
from physics_scene_coordinate_v2 import (
    PhysicsSceneCoordinateMLPv2,
    PhysicsSceneV2Sample,
    augment_domain_samples,
    build_samples,
    metric_summary,
)
from scene_coordinate_regression import load_manifest_rows


CONDITIONS = ("clean_true", "noisy_true", "clean_estimated", "noisy_estimated")


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


def _calibration(sample: PhysicsSceneV2Sample) -> CameraCalibration:
    # Ray Casting uses the same camera source as the sample condition.
    source = "camera_estimated" if sample.condition.endswith("estimated") else "camera"
    value = sample.row.get(source)
    if not isinstance(value, dict):
        raise ValueError(f"Missing calibration source {source} for {sample.row.get('sample_id')}")
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


def _run_model(
    model: PhysicsSceneCoordinateMLPv2,
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
    entries: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sigma = float(np.linalg.norm(prediction["std_m"][index]))
        projected = prediction["projected_pixel"][index]
        target_pixel = sample.target_pixel_under_camera
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
                "projected_pixel": projected,
                "target_pixel": target_pixel,
                "reprojection_error_px": float(np.linalg.norm(projected - target_pixel)),
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
        "reprojection_error_px": _describe(item["reprojection_error_px"] for item in entries),
    }


def _run_ray_ipm(
    samples: Sequence[PhysicsSceneV2Sample],
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

        ipm_point = None
        ipm_status = "not_applicable_non_floor"
        started = time.perf_counter()
        if sample.surface == "floor":
            try:
                ipm_point = FloorHomography.from_calibration(calibration).pixel_to_floor_xyz(
                    [calibration.undistort_pixels([sample.pixel])[0]]
                )[0].astype(np.float32)
                ipm_status = "valid_floor_ipm"
            except (ValueError, np.linalg.LinAlgError):
                ipm_status = "invalid_floor_ipm"
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
                "ipm_status": ipm_status,
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
            "valid_count": int(len(valid)),
            "surface_counts": dict(Counter(record["surface"] for record in valid)),
        }

    return {
        "records": records,
        "ray": {**baseline_metrics("ray_point"), "latency_ms": _describe(ray_latencies)},
        "ipm": {**baseline_metrics("ipm_point"), "latency_ms": _describe(ipm_latencies)},
        "ipm_status_counts": dict(Counter(record["ipm_status"] for record in records)),
    }


def _surface_metrics(records: Sequence[dict[str, Any]], point_key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    surfaces = sorted({str(record.get("surface", "unknown")) for record in records})
    for surface in surfaces:
        valid = [record for record in records if record.get("surface") == surface and record.get(point_key) is not None]
        output[surface] = metric_summary(
            np.asarray([record["gt_xyz"] for record in valid], dtype=np.float32).reshape(-1, 3)
            if valid
            else np.empty((0, 3)),
            np.asarray([record[point_key] for record in valid], dtype=np.float32).reshape(-1, 3)
            if valid
            else np.empty((0, 3)),
        )
    return output


def _route(records: Sequence[dict[str, Any]], threshold_m: float) -> dict[str, Any]:
    selected: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sources: Counter[str] = Counter()
    routed_records: list[dict[str, Any]] = []
    for record in records:
        point = None
        source = "not_localized"
        if record.get("physics_point") is not None and float(record["sigma_norm_m"]) <= threshold_m:
            point = record["physics_point"]
            source = "physics_scene_mlp_v2"
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
        routed_records.append({**record, "selected_point": point, "selected_source": source})
    return {
        "metrics": metric_summary(
            np.asarray(targets, dtype=np.float32).reshape(-1, 3) if targets else np.empty((0, 3)),
            np.asarray(selected, dtype=np.float32).reshape(-1, 3) if selected else np.empty((0, 3)),
        ),
        "valid_rate": float(len(selected) / max(1, len(records))),
        "sources": dict(sources),
        "records": routed_records,
    }


def _choose_threshold(records: Sequence[dict[str, Any]], values: Iterable[float]) -> dict[str, Any]:
    best: Optional[dict[str, Any]] = None
    for threshold in values:
        routed = _route(records, float(threshold))
        score = routed["metrics"].get("mae_m")
        score_value = float(score) if score is not None else float("inf")
        candidate = {
            "threshold_m": float(threshold),
            "metrics": routed["metrics"],
            "valid_rate": routed["valid_rate"],
            "sources": routed["sources"],
        }
        if best is None or score_value < float(best["metrics"].get("mae_m") or float("inf")):
            best = candidate
    if best is None:
        raise RuntimeError("No uncertainty gate threshold was evaluated")
    return best


def _merge_records(
    model_result: dict[str, Any],
    baseline_result: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_id = {str(item["sample_id"]): item for item in baseline_result["records"]}
    output: list[dict[str, Any]] = []
    for prediction in model_result["predictions"]:
        base = baseline_by_id[str(prediction["sample_id"])]
        output.append(
            {
                **prediction,
                "physics_point": prediction["point"],
                "ray_point": base["ray_point"],
                "ipm_point": base["ipm_point"],
                "ray_status": base["ray_status"],
                "ipm_status": base["ipm_status"],
            }
        )
    return output


def _visualization_records(records: Sequence[dict[str, Any]], route: dict[str, Any]) -> list[dict[str, Any]]:
    routed_by_id = {str(item["sample_id"]): item for item in route["records"]}
    output: list[dict[str, Any]] = []
    for record in records:
        routed = routed_by_id[str(record["sample_id"])]
        output.append(
            {
                "sample_id": record["sample_id"],
                "scene_id": record["scene_id"],
                "frame_index": record.get("frame_index", 0),
                "gt_xyz": record["gt_xyz"],
                "branches": {
                    "scene_mlp": {
                        "point": record["physics_point"],
                        "uncertainty_std_m": record["std_m"],
                        "confidence": record["confidence"],
                    },
                    "ray": {"point": record["ray_point"], "status": record["ray_status"]},
                    "ipm": {"point": record["ipm_point"], "status": record["ipm_status"]},
                    "fusion": {
                        "point": routed["selected_point"],
                        "source": routed["selected_source"],
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
    records: Sequence[dict[str, Any]],
    route: dict[str, Any],
) -> Path:
    path = output / "visualization_summaries" / f"{mode}_{condition}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "LAB_SAM.physics_scene_visualization.v2",
        "dataset": {
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry, not real CCTV accuracy.",
        },
        "method": {"mesh": str(dataset / "room_mesh.json")},
        "records": _visualization_records(records, route),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "train_mode",
        "test_condition",
        "surface",
        "sample_count",
        "physics_mae_m",
        "physics_median_m",
        "physics_p95_m",
        "physics_under_0.25m",
        "physics_under_0.50m",
        "physics_under_1.00m",
        "physics_latency_ms",
        "physics_reprojection_px",
        "ray_mae_m",
        "ray_median_m",
        "ray_p95_m",
        "ray_valid_rate",
        "ray_latency_ms",
        "ipm_mae_m",
        "ipm_median_m",
        "ipm_p95_m",
        "ipm_valid_rate",
        "ipm_latency_ms",
        "route_mae_m",
        "route_valid_rate",
        "gate_threshold_m",
        "ray_source_count",
        "scene_mlp_source_count",
        "ipm_source_count",
        "not_localized_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_tum(dataset_root: Path) -> dict[str, Any]:
    """Return a small non-fire audit for the downloaded TUM RGB-D sequence."""

    sequence = dataset_root / "datasets" / "open_3d" / "tum_rgbd" / "rgbd_dataset_freiburg1_desk"
    if not sequence.is_dir():
        return {"available": False, "reason": f"missing {sequence}"}

    def data_lines(path: Path) -> list[str]:
        if not path.is_file():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]

    rgb = data_lines(sequence / "rgb.txt")
    depth = data_lines(sequence / "depth.txt")
    gt = data_lines(sequence / "groundtruth.txt")
    return {
        "available": True,
        "sequence": str(sequence),
        "rgb_frames": len(rgb),
        "depth_frames": len(depth),
        "groundtruth_pose_rows": len(gt),
        "has_fire_xyz": False,
        "use": "camera/depth/trajectory sanity check only; not mixed into fire accuracy",
    }


def _make_train_pool(
    mode: str,
    all_samples: dict[str, dict[str, list[PhysicsSceneV2Sample]]],
    args: argparse.Namespace,
) -> tuple[list[PhysicsSceneV2Sample], list[PhysicsSceneV2Sample], dict[str, Any]]:
    clean_train = all_samples["clean_true"]["train"]
    clean_val = all_samples["clean_true"]["val"]
    if mode == "pinhole_only":
        return clean_train, clean_val, {
            "base_conditions": ["clean_true"],
            "domain_augmentation": False,
            "canonicalize": False,
            "geometry_prior": False,
        }

    augmented_train = augment_domain_samples(
        clean_train,
        copies_per_sample=args.augment_copies,
        seed=args.seed + 17,
        focal_pct=args.focal_pct,
        principal_px=args.principal_px,
        rotation_deg=args.rotation_deg,
        position_m=args.position_m,
        point_sigma_px=args.point_sigma_px,
        canonicalize=True,
    )
    augmented_val = augment_domain_samples(
        clean_val,
        copies_per_sample=max(1, min(args.val_augment_copies, args.augment_copies)),
        seed=args.seed + 37,
        focal_pct=args.focal_pct,
        principal_px=args.principal_px,
        rotation_deg=args.rotation_deg,
        position_m=args.position_m,
        point_sigma_px=args.point_sigma_px,
        canonicalize=True,
    )
    train_pool = list(clean_train) + augmented_train
    val_pool = list(clean_val) + augmented_val
    metadata: dict[str, Any] = {
        "base_conditions": ["clean_true"],
        "domain_augmentation": True,
        "domain_augmented_train": len(augmented_train),
        "domain_augmented_val": len(augmented_val),
        "canonicalize": True,
        "geometry_prior": mode == "pinhole_domain_geometry",
    }
    if mode == "pinhole_full_mixed_geometry":
        # Keep the source conditions explicit, then add the same geometry-safe
        # camera perturbations.  This is the strongest robustness configuration.
        train_pool.extend(
            sample
            for condition in CONDITIONS
            if condition != "clean_true"
            for sample in all_samples[condition]["train"]
        )
        val_pool.extend(
            sample
            for condition in CONDITIONS
            if condition != "clean_true"
            for sample in all_samples[condition]["val"]
        )
        metadata["base_conditions"] = list(CONDITIONS)
        metadata["geometry_prior"] = True
    return train_pool, val_pool, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "histories", "predictions", "visualization_summaries"):
        (output / name).mkdir(parents=True, exist_ok=True)

    rows = {split: load_manifest_rows(dataset, split) for split in ("train", "val", "test")}
    # Keep both representations.  ``pinhole_only`` intentionally receives the
    # raw pixel representation, while the domain/generalization branches use
    # the canonical-camera representation even for clean samples.  Reusing one
    # bank here would make the ablation labels misleading.
    all_samples: dict[str, dict[str, list[PhysicsSceneV2Sample]]] = {}
    all_samples_raw: dict[str, dict[str, list[PhysicsSceneV2Sample]]] = {}
    skipped: dict[str, dict[str, dict[str, int]]] = {}
    for condition in CONDITIONS:
        all_samples[condition] = {}
        all_samples_raw[condition] = {}
        skipped[condition] = {}
        for split in ("train", "val", "test"):
            all_samples[condition][split], skipped[condition][split] = build_samples(
                rows[split], condition, canonicalize=True
            )
            all_samples_raw[condition][split], _ = build_samples(
                rows[split], condition, canonicalize=False
            )

    mesh_path = dataset / "room_mesh.json"
    mesh_data = json.loads(mesh_path.read_text(encoding="utf-8"))
    mesh = TriangleMesh(mesh_data["vertices"], mesh_data["faces"])
    thresholds = np.linspace(
        float(args.min_uncertainty_m),
        float(args.max_uncertainty_m),
        max(2, int(args.threshold_steps)),
    )
    mode_names = (
        "pinhole_only",
        "pinhole_domain",
        "pinhole_domain_geometry",
        "pinhole_full_mixed_geometry",
    )
    results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for mode in mode_names:
        sample_bank = all_samples_raw if mode == "pinhole_only" else all_samples
        train_samples, val_samples, mode_meta = _make_train_pool(mode, sample_bank, args)
        # Augmented samples retain the original scene id.  They do not introduce
        # a new scene, so the scene-level leakage check remains meaningful.
        _check_scene_split(
            {
                "train": train_samples,
                "val": val_samples,
                "test": [sample for condition in CONDITIONS for sample in sample_bank[condition]["test"]],
            }
        )
        model = PhysicsSceneCoordinateMLPv2(
            hidden_dim=args.hidden_dim,
            device=args.device,
            canonicalize=bool(mode_meta["canonicalize"]),
            geometry_prior=bool(mode_meta["geometry_prior"]),
        )
        print(
            f"[physics-v2] mode={mode} train={len(train_samples)} val={len(val_samples)} "
            f"canonicalize={mode_meta['canonicalize']} geometry_prior={mode_meta['geometry_prior']}",
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
        checkpoint = output / "checkpoints" / f"physics_scene_mlp_v2_{mode}.pth"
        model.save(
            str(checkpoint),
            metadata={"mode": mode, **mode_meta, "epochs_ran": len(history)},
        )
        (output / "histories" / f"physics_scene_mlp_v2_{mode}.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
        )

        validation_records: list[dict[str, Any]] = []
        # Gate selection uses the same source conditions used by the model.  For
        # the full mixed branch, all four are available; otherwise clean_true
        # plus the domain-augmented validation pool are used.
        validation_model = _run_model(model, val_samples)
        validation_baselines = _run_ray_ipm(sample_bank["clean_true"]["val"], mesh)
        base_by_id = {str(item["sample_id"]): item for item in validation_baselines["records"]}
        for item in validation_model["predictions"]:
            base = base_by_id.get(str(item["sample_id"]))
            if base is None:
                # Domain-augmented samples have no valid ray baseline in the
                # original scene; keep them for validation of the MLP but do
                # not invent a ray/IPM fallback for them.
                validation_records.append({**item, "physics_point": item["point"], "ray_point": None, "ipm_point": None})
                continue
            validation_records.append(
                {
                    **item,
                    "physics_point": item["point"],
                    "ray_point": base["ray_point"],
                    "ipm_point": base["ipm_point"],
                    "sigma_norm_m": item["sigma_norm_m"],
                }
            )
        selected_gate = _choose_threshold(validation_records, thresholds)

        mode_result: dict[str, Any] = {
            "mode": mode,
            "training": mode_meta,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "checkpoint": str(checkpoint),
            "epochs_ran": len(history),
            "best_val_mae_m": min((item["val_mae_m"] for item in history), default=None),
            "gate_selected_on_validation": selected_gate,
            "test_conditions": {},
        }

        for condition in CONDITIONS:
            samples = sample_bank[condition]["test"]
            print(f"[test-v2] mode={mode} condition={condition} samples={len(samples)}", flush=True)
            model_result = _run_model(model, samples)
            baseline_result = _run_ray_ipm(samples, mesh)
            merged = _merge_records(model_result, baseline_result)
            route = _route(merged, float(selected_gate["threshold_m"]))
            summary_path = _write_visualization_summary(output, dataset, mode, condition, merged, route)
            prediction_path = output / "predictions" / f"{mode}_{condition}.json"
            prediction_path.write_text(
                json.dumps(
                    {"physics_v2": model_result, "baselines": baseline_result, "route": route},
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                ),
                encoding="utf-8",
            )

            mode_result["test_conditions"][condition] = {
                "sample_count": len(samples),
                "physics_v2": model_result["metrics"],
                "physics_v2_latency_ms": model_result["latency_ms"],
                "physics_v2_reprojection_error_px": model_result["reprojection_error_px"],
                "physics_v2_by_surface": _surface_metrics(merged, "physics_point"),
                "ray": {
                    **baseline_result["ray"],
                    "by_surface": _surface_metrics(baseline_result["records"], "ray_point"),
                },
                "ipm_floor_only": {
                    **baseline_result["ipm"],
                    "by_surface": _surface_metrics(baseline_result["records"], "ipm_point"),
                    "status_counts": baseline_result["ipm_status_counts"],
                },
                "route": {
                    "metrics": route["metrics"],
                    "valid_rate": route["valid_rate"],
                    "sources": route["sources"],
                    "threshold_m": selected_gate["threshold_m"],
                },
                "visualization_summary": str(summary_path),
            }
            source_counts = Counter(route["sources"])
            physics_metric = model_result["metrics"]
            ray_metric = baseline_result["ray"]["metrics"]
            ipm_metric = baseline_result["ipm"]["metrics"]
            for surface in sorted({sample.surface for sample in samples}):
                surface_samples = [sample for sample in samples if sample.surface == surface]
                surface_model = _surface_metrics(merged, "physics_point").get(surface, {})
                surface_ray = _surface_metrics(baseline_result["records"], "ray_point").get(surface, {})
                surface_ipm = _surface_metrics(baseline_result["records"], "ipm_point").get(surface, {})
                csv_rows.append(
                    {
                        "train_mode": mode,
                        "test_condition": condition,
                        "surface": surface,
                        "sample_count": len(surface_samples),
                        "physics_mae_m": surface_model.get("mae_m"),
                        "physics_median_m": surface_model.get("median_m"),
                        "physics_p95_m": surface_model.get("p95_m"),
                        "physics_under_0.25m": surface_model.get("under_0.25m"),
                        "physics_under_0.50m": surface_model.get("under_0.50m"),
                        "physics_under_1.00m": surface_model.get("under_1.00m"),
                        "physics_latency_ms": model_result["latency_ms"].get("mean"),
                        "physics_reprojection_px": model_result["reprojection_error_px"].get("mean"),
                        "ray_mae_m": surface_ray.get("mae_m"),
                        "ray_median_m": surface_ray.get("median_m"),
                        "ray_p95_m": surface_ray.get("p95_m"),
                        "ray_valid_rate": float(sum(1 for r in baseline_result["records"] if r["surface"] == surface and r["ray_point"] is not None) / max(1, len(surface_samples))),
                        "ray_latency_ms": baseline_result["ray"]["latency_ms"].get("mean"),
                        "ipm_mae_m": surface_ipm.get("mae_m"),
                        "ipm_median_m": surface_ipm.get("median_m"),
                        "ipm_p95_m": surface_ipm.get("p95_m"),
                        "ipm_valid_rate": float(sum(1 for r in baseline_result["records"] if r["surface"] == surface and r["ipm_point"] is not None) / max(1, len(surface_samples))),
                        "ipm_latency_ms": baseline_result["ipm"]["latency_ms"].get("mean"),
                        "route_mae_m": route["metrics"].get("mae_m"),
                        "route_valid_rate": route["valid_rate"],
                        "gate_threshold_m": selected_gate["threshold_m"],
                        "ray_source_count": source_counts.get("ray_casting", 0),
                        "scene_mlp_source_count": source_counts.get("physics_scene_mlp_v2", 0),
                        "ipm_source_count": source_counts.get("ipm_floor", 0),
                        "not_localized_count": source_counts.get("not_localized", 0),
                    }
                )

        results[mode] = mode_result

    metrics_path = output / "physics_scene_v2_metrics.csv"
    _write_csv(metrics_path, csv_rows)
    summary = {
        "format": "LAB_SAM.physics_scene_cross_benchmark.v2",
        "dataset": {
            "root": str(dataset),
            "mesh": str(mesh_path),
            "raw_records": {split: len(rows[split]) for split in rows},
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry results, not real CCTV accuracy.",
            "split_policy": "scene-separated train/val/test; no adjacent-frame leakage",
        },
        "auxiliary_open_3d": _audit_tum(root),
        "conditions": list(CONDITIONS),
        "models": [
            "pinhole_only",
            "pinhole_domain",
            "pinhole_domain_geometry",
            "pinhole_full_mixed_geometry",
        ],
        "v2_components": {
            "pinhole_physical_layer": "C + R_world_to_camera.T @ ([corrected_ray_x, corrected_ray_y, 1] * metric_depth)",
            "domain_generalization": "canonical camera ray plus geometry-preserving focal/principal-point/pose/pixel perturbation",
            "geometry_error_prior": "bounded metric sensitivity weight and pixel reprojection loss",
            "ipm_policy": "IPM is valid only for fire_surface=floor; non-floor rows are not_applicable",
        },
        "skipped": skipped,
        "results": results,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "visualization_summaries": str(output / "visualization_summaries"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
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
        default=root / "output" / "physics_scene_cross_20261008_v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--val-augment-copies", type=int, default=1)
    parser.add_argument("--focal-pct", type=float, default=0.05)
    parser.add_argument("--principal-px", type=float, default=10.0)
    parser.add_argument("--rotation-deg", type=float, default=2.0)
    parser.add_argument("--position-m", type=float, default=0.05)
    parser.add_argument("--point-sigma-px", type=float, default=2.0)
    parser.add_argument("--min-uncertainty-m", type=float, default=0.10)
    parser.add_argument("--max-uncertainty-m", type=float, default=3.00)
    parser.add_argument("--threshold-steps", type=int, default=15)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        raise ValueError("epochs, batch-size and hidden-dim must be positive")
    if args.augment_copies < 0 or args.val_augment_copies < 0:
        raise ValueError("augmentation copies cannot be negative")
    if args.threshold_steps < 2 or args.min_uncertainty_m <= 0 or args.max_uncertainty_m <= args.min_uncertainty_m:
        raise ValueError("uncertainty threshold range is invalid")
    run(args)


if __name__ == "__main__":
    main()
