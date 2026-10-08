"""Aggregate and rank the current 2D and 3D model experiments.

The benchmark scripts in this repository intentionally write separate summaries
because they use different protocols.  This utility is the read-only index over
those summaries.  It does not retrain a model or rerun a 3D benchmark by
default.  It keeps pixel errors (2D) and metric errors (3D) in separate tables,
and records camera condition, coverage, and oracle status on every row.

Typical use from ``D:\\LAB\\SAM_Experiment``::

    .venv\\Scripts\\python.exe compare_all_models.py

To regenerate the CCTV 2D comparison before aggregating::

    .venv\\Scripts\\python.exe compare_all_models.py --refresh-2d

The default registry uses the latest complete artifact for each model family.
``--include-legacy`` adds older full benchmark outputs when they are present.
All rows, including rows not printed to the console, are written to JSON and
CSV in the selected output directory.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parent
# Use a stable writable location so repeated runs update one current report
# instead of colliding with an older date-stamped CSV that may still be open.
DEFAULT_OUTPUT = ROOT / "output" / "compare_all_models_current"


# The last terminal comparison was produced before compare_v3_roi wrote a
# machine-readable summary.  These values are retained as a clearly labelled
# snapshot so an aggregate run remains complete without silently rerunning a
# detector.  ``--refresh-2d`` replaces this snapshot with fresh values.
TERMINAL_2D_SNAPSHOT = {
    "source": "output/workflow_2d_then_3d_20261008/2d_compare",
    "protocol": "compare_v3_roi terminal snapshot on CCTV test split",
    "dataset": "real_cctv_labeled_test",
    "total": 36,
    "rows": [
        {"model": "coarse detector", "branch": "coarse", "detected": 36, "mae_px": 9.41, "median_px": 7.32, "p95_px": 17.07, "pck10": 0.75, "pck25": 0.97},
        {"model": "ROI refiner", "branch": "roi", "detected": 36, "mae_px": 9.63, "median_px": 6.87, "p95_px": 18.43, "pck10": 0.75, "pck25": 0.97},
        {"model": "v3 detector", "branch": "v3", "detected": 26, "mae_px": 64.68, "median_px": 53.46, "p95_px": 131.73, "pck10": 0.19, "pck25": 0.38},
    ],
}


@dataclass
class Row:
    """One comparable metric observation."""

    dimension: str  # 2d or 3d
    model: str
    branch: str
    dataset: str
    comparison_group: str
    condition: str
    camera_source: str
    pixel_input: str
    metric: str
    units: str
    mae: Optional[float]
    median: Optional[float]
    p95: Optional[float]
    valid_count: Optional[int]
    total_count: Optional[int]
    valid_rate: Optional[float]
    pck10: Optional[float]
    pck25: Optional[float]
    under_025m: Optional[float]
    oracle: bool
    known_camera: bool
    source: str
    protocol: str
    notes: str = ""
    rank: Optional[int] = None


def _json(path: Path, errors: list[str]) -> Optional[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            errors.append(f"not an object: {path}")
            return None
        return value
    except FileNotFoundError:
        errors.append(f"missing: {path}")
    except Exception as exc:  # malformed old artifacts should not stop a run
        errors.append(f"unreadable: {path} ({exc})")
    return None


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _int(value: Any) -> Optional[int]:
    n = _num(value)
    return None if n is None else int(round(n))


def _source(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _metric_values(metric: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    return _num(metric.get("mae_m", metric.get("three_d_mae_m"))), _num(
        metric.get("median_m", metric.get("three_d_median_m"))
    ), _num(metric.get("p95_m", metric.get("three_d_p95_m")))


def _coverage(metric: dict[str, Any], total_hint: Any = None) -> tuple[Optional[int], Optional[int], Optional[float]]:
    valid = _int(
        metric.get("valid_count", metric.get("valid", metric.get("count", metric.get("three_d_samples"))))
    )
    total = _int(metric.get("total_count", metric.get("records", metric.get("attempted", total_hint))))
    rate = _num(metric.get("valid_rate", metric.get("success_rate")))
    if rate is None and valid is not None and total:
        rate = valid / total
    if total is None and valid is not None and rate and 0.0 < rate <= 1.0:
        total = int(round(valid / rate))
    return valid, total, rate


def _dataset_name(path: Path, summary: dict[str, Any]) -> str:
    dataset = summary.get("dataset")
    if isinstance(dataset, dict):
        dataset = dataset.get("root") or dataset.get("name")
    if isinstance(dataset, str) and dataset:
        lower = dataset.lower().replace("\\", "/")
        if "synthetic_fire_3d_v3" in lower:
            return "synthetic_fire_3d_v3_test"
        if "replicacad" in lower or "asset_backed" in lower:
            return "replicacad_asset_geometry"
        if "cctv" in lower or "fire_smoke" in lower:
            return "real_cctv"
        return Path(dataset).name or dataset
    text = str(path).lower()
    if "replicacad" in text or "asset_backed" in text:
        return "replicacad_asset_geometry"
    return "synthetic_fire_3d_v3_test" if "3d" in text else "unknown"


def _append(rows: list[Row], **kwargs: Any) -> None:
    rows.append(Row(**kwargs))


def _row_2d(
    rows: list[Row], *, model: str, branch: str, dataset: str, condition: str,
    metric: dict[str, Any], source: str, protocol: str, notes: str = "",
    total_hint: Any = None, pck10: Any = None, pck25: Any = None,
) -> None:
    valid, total, rate = _coverage(metric, total_hint)
    _append(
        rows,
        dimension="2d", model=model, branch=branch, dataset=dataset,
        comparison_group=f"2d/{dataset}", condition=condition,
        camera_source="not_applicable", pixel_input="not_applicable",
        metric="mae_px", units="px", mae=_num(metric.get("mae_px")),
        median=_num(metric.get("median_px")), p95=_num(metric.get("p95_px")),
        valid_count=valid, total_count=total, valid_rate=rate,
        pck10=_num(metric.get("pck10", pck10)), pck25=_num(metric.get("pck25", pck25)),
        under_025m=None, oracle=False, known_camera=False, source=source,
        protocol=protocol, notes=notes,
    )


def _row_3d(
    rows: list[Row], *, model: str, branch: str, dataset: str, condition: str,
    camera_source: str, pixel_input: str, metric: dict[str, Any], source: str,
    protocol: str, oracle: bool = False, known_camera: Optional[bool] = None,
    notes: str = "", total_hint: Any = None,
) -> None:
    mae, median, p95 = _metric_values(metric)
    valid, total, rate = _coverage(metric, total_hint)
    if known_camera is None:
        known_camera = str(camera_source).lower() in {"true", "measured", "oracle"}
    _append(
        rows,
        dimension="3d", model=model, branch=branch, dataset=dataset,
        comparison_group=f"3d/{dataset}/{condition}/oracle={str(bool(oracle)).lower()}",
        condition=condition, camera_source=str(camera_source), pixel_input=str(pixel_input),
        metric="mae_m", units="m", mae=mae, median=median, p95=p95,
        valid_count=valid, total_count=total, valid_rate=rate,
        pck10=None, pck25=None, under_025m=_num(metric.get("under_0.25m")),
        oracle=bool(oracle), known_camera=bool(known_camera), source=source,
        protocol=protocol, notes=notes,
    )


def _add_roi_2d(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "roi_best_rerun_20261008" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    for item in summary.get("results", []):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "roi"))
        for dataset_key, payload in (item.get("datasets") or {}).items():
            if not isinstance(payload, dict) or not isinstance(payload.get("metric"), dict):
                continue
            metric = dict(payload["metric"])
            metric["count"] = payload.get("samples")
            dataset = {
                "real_test": "real_cctv_positive_test",
                "synthetic_test": "synthetic_fire_positive_test",
                "experiment_test": "roi_experiment_test",
            }.get(dataset_key, dataset_key)
            _row_2d(
                rows, model=tag, branch="roi_refiner", dataset=dataset,
                condition=dataset_key, metric=metric, source=_source(path),
                protocol="saved ROI checkpoint re-evaluation", notes=str(item.get("checkpoint", "")),
                total_hint=payload.get("samples"),
            )


def _add_terminal_2d(rows: list[Row], errors: list[str], refresh_summary: Optional[Path] = None) -> None:
    data: dict[str, Any] = TERMINAL_2D_SNAPSHOT
    source = TERMINAL_2D_SNAPSHOT["source"]
    if refresh_summary and refresh_summary.is_file():
        loaded = _json(refresh_summary, errors)
        if loaded:
            data = loaded
            source = _source(refresh_summary)
    for item in data.get("rows", []):
        metric = dict(item)
        metric["count"] = item.get("detected")
        _row_2d(
            rows, model=str(item.get("model", item.get("branch", "2d"))),
            branch=str(item.get("branch", "unknown")),
            dataset=str(data.get("dataset", "real_cctv_labeled_test")),
            condition="cctv_test", metric=metric, source=source,
            protocol=str(data.get("protocol", "compare_v3_roi terminal comparison")),
            notes="snapshot is used unless --refresh-2d is supplied",
            total_hint=data.get("total"),
        )


def _add_five_workflow(rows: list[Row], errors: list[str], path: Path, label: str, oracle: bool = False) -> None:
    summary = _json(path, errors)
    if not summary:
        return
    method = summary.get("method") or {}
    dataset = _dataset_name(path, summary)
    camera = str(method.get("calibration_source", "unknown"))
    pixel_input = "roi" if method.get("roi_checkpoint") else "coarse"
    total = (summary.get("dataset") or {}).get("metric_xyz_records") or (summary.get("dataset") or {}).get("records_selected")
    for branch, metric in (summary.get("branches") or {}).items():
        if not isinstance(metric, dict) or _num(metric.get("three_d_mae_m")) is None:
            continue
        _row_3d(
            rows, model=label, branch=str(branch), dataset=dataset,
            condition=f"{pixel_input}_mixed_camera_{camera}", camera_source=camera,
            pixel_input=pixel_input, metric=metric, source=_source(path),
            protocol="five-workflow mesh benchmark", oracle=oracle,
            notes=f"three_d_samples={metric.get('three_d_samples')}", total_hint=total,
        )


def _add_low_cost(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "low_cost_scene_mlp_rerun_20261008" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    method = summary.get("method") or {}
    camera = str(method.get("camera_source", "estimated"))
    valid_test = (summary.get("dataset") or {}).get("valid_samples", {}).get("test")
    tri_info = summary.get("triangulation") or {}
    tri_key = {
        "triangulation_noisy_estimated": "noisy_estimated_pose",
        "triangulation_noisy_true": "noisy_true_pose",
        "triangulation_clean_true_oracle": "clean_pixel_true_pose_oracle",
    }
    for branch, metric in (summary.get("branches") or {}).items():
        if not isinstance(metric, dict) or _num(metric.get("mae_m")) is None:
            continue
        is_oracle = "oracle" in branch.lower()
        branch_camera = camera
        branch_pixel = str(method.get("pixel_key", "noisy_pixel"))
        branch_condition = f"noisy_{branch_camera}"
        if branch == "triangulation_noisy_true":
            branch_camera = "true"
            branch_condition = "noisy_true"
        elif branch == "triangulation_clean_true_oracle":
            branch_camera = "true"
            branch_pixel = "clean_pixel_oracle"
            branch_condition = "clean_true_oracle"
        _row_3d(
            rows, model="low-cost post-ROI", branch=str(branch), dataset=dataset,
            condition=branch_condition, camera_source=branch_camera,
            pixel_input=branch_pixel, metric=metric,
            source=_source(path), protocol="KNN/IDW, Scene-MLP, ray and triangulation benchmark",
            oracle=is_oracle,
            notes=(
                f"raw_test_records={(summary.get('dataset') or {}).get('raw_records', {}).get('test')}; "
                f"scenes_attempted={tri_info.get(tri_key.get(branch, ''), {}).get('scenes_attempted')}"
                if branch in tri_key else ""
            ),
            total_hint=(
                tri_info.get(tri_key[branch], {}).get("scenes_attempted")
                if branch in tri_key else valid_test
            ),
        )


def _add_scene_coordinate(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "scene_coordinate_regression_rerun_20261008" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    method = summary.get("method") or {}
    camera = str(method.get("camera_source", "estimated"))
    pixel = str(method.get("pixel_key", "noisy_pixel"))
    total = (summary.get("dataset") or {}).get("raw_records", {}).get("test")
    for branch, metric in (summary.get("branches") or {}).items():
        if isinstance(metric, dict) and _num(metric.get("mae_m")) is not None:
            _row_3d(
                rows, model="scene-coordinate regression", branch=str(branch), dataset=dataset,
                condition=f"{pixel}_{camera}", camera_source=camera, pixel_input=pixel,
                metric=metric, source=_source(path), protocol="scene-coordinate regression benchmark",
                total_hint=total,
            )


def _add_ablation(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "ablation_3d_all_rerun_20261008" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    for variant, details in (summary.get("variants") or {}).items():
        if not isinstance(details, dict):
            continue
        mlp = details.get("mlp") or {}
        metric = mlp.get("metrics") if isinstance(mlp, dict) else None
        if isinstance(metric, dict) and _num(metric.get("mae_m")) is not None:
            camera = str(mlp.get("camera_source", "unknown"))
            pixel = str(mlp.get("pixel_key", "unknown"))
            _row_3d(
                rows, model="Scene-MLP ablation", branch="mlp", dataset=dataset,
                condition=str(variant), camera_source=camera, pixel_input=pixel,
                metric=metric, source=_source(path), protocol="strict scene-separated ablation",
                total_hint=(metric.get("attempted") or metric.get("count")),
            )
        tri = details.get("triangulation") or {}
        tri_metric = tri.get("metrics_gated") or tri.get("metrics_raw") or tri.get("metrics")
        if isinstance(tri_metric, dict) and _num(tri_metric.get("mae_m")) is not None:
            camera = str(tri.get("camera_source", "unknown"))
            pixel = str(tri.get("pixel_key", "unknown"))
            _row_3d(
                rows, model="triangulation ablation", branch="triangulation", dataset=dataset,
                condition=str(variant), camera_source=camera, pixel_input=pixel,
                metric=tri_metric, source=_source(path), protocol="gated triangulation ablation",
                oracle=str(variant) == "clean_true", total_hint=tri_metric.get("attempted"),
                notes="clean_true is an oracle diagnostic",
            )


def _add_temporal(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "temporal_scene_mlp_rerun_20261008" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    for condition, details in (summary.get("variants") or {}).items():
        if not isinstance(details, dict):
            continue
        camera = str(details.get("camera_source", "unknown"))
        pixel = str(details.get("pixel_key", "unknown"))
        test = details.get("test") or {}
        entries: list[tuple[str, Any]] = []
        if isinstance(test.get("raw"), dict):
            entries.append(("raw", test["raw"]))
        selected = test.get("selected") or {}
        if isinstance(selected, dict):
            for name, value in selected.items():
                entries.append((str(name), value))
        for branch, value in entries:
            metric = value.get("metrics") if isinstance(value, dict) and isinstance(value.get("metrics"), dict) else value
            if not isinstance(metric, dict) or _num(metric.get("mae_m")) is None:
                continue
            _row_3d(
                rows, model="temporal Scene-MLP", branch=branch, dataset=dataset,
                condition=str(condition), camera_source=camera, pixel_input=pixel,
                metric=metric, source=_source(path), protocol="temporal EMA/EKF selection",
                total_hint=metric.get("attempted"), notes=str(value.get("parameter", "")) if isinstance(value, dict) else "",
            )


def _add_physics(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "physics_scene_all_rerun_20261008" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    for mode, result in (summary.get("results") or {}).items():
        if not isinstance(result, dict):
            continue
        for condition, details in (result.get("test_conditions") or {}).items():
            if not isinstance(details, dict):
                continue
            metric = details.get("physics_v2")
            if not isinstance(metric, dict) or _num(metric.get("mae_m")) is None:
                continue
            _row_3d(
                rows, model=f"Physics Scene-MLP v2 ({mode})", branch="physics_v2",
                dataset=dataset, condition=str(condition), camera_source=str(condition).split("_")[-1],
                pixel_input=str(condition).split("_")[0], metric=metric,
                source=_source(path), protocol="physics-aware pinhole scene benchmark",
                total_hint=details.get("sample_count"),
            )


def _add_siren(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "siren_categorical_scene_20261008_full" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    for mode, result in (summary.get("results") or {}).items():
        if not isinstance(result, dict):
            continue
        for condition, details in (result.get("test_conditions") or {}).items():
            if not isinstance(details, dict):
                continue
            metric = details.get("scene_mlp")
            if not isinstance(metric, dict) or _num(metric.get("mae_m")) is None:
                continue
            _row_3d(
                rows, model=f"{mode} scene model", branch="scene_mlp", dataset=dataset,
                condition=str(condition), camera_source=str(condition).split("_")[-1],
                pixel_input=str(condition).split("_")[0], metric=metric,
                source=_source(path), protocol="SIREN/ReLU scalar/categorical scene benchmark",
                total_hint=metric.get("count"),
            )


def _add_multiplane(rows: list[Row], errors: list[str]) -> None:
    path = ROOT / "output" / "multiplane_cpu_mixed_20261008" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    prediction_path = Path(str(summary.get("prediction_summary", "")))
    camera = "true"
    if prediction_path.is_file():
        pred = _json(prediction_path, errors)
        if pred:
            camera = str((pred.get("method") or {}).get("calibration_source", camera))
    protocol = str((summary.get("protocol") or {}).get("warning", "multi-plane homography"))
    for key, value in (summary.get("metrics") or {}).items():
        if not isinstance(value, dict) or not isinstance(value.get("all_surfaces"), dict):
            continue
        if ":" not in key:
            continue
        source_name, method = key.split(":", 1)
        metric = value["all_surfaces"]
        oracle = source_name == "oracle" or "surface_oracle" in method
        _row_3d(
            rows, model="multi-plane homography", branch=method, dataset=dataset,
            condition=source_name, camera_source=camera, pixel_input=source_name,
            metric=metric, source=_source(path), protocol=protocol, oracle=oracle,
            total_hint=metric.get("records"), notes="planar visibility/selection protocol",
        )


def _add_priority1_raycasting(rows: list[Row], errors: list[str]) -> None:
    """Add the latest protected Ray-casting and ray-count benchmark.

    This benchmark is separate from the five-workflow summary: it evaluates
    multi-ray aggregation, protected routing, and temporal filters under the
    same synthetic mesh protocol.  It is therefore indexed as its own model
    family instead of overwriting the workflow rows.
    """
    path = ROOT / "output" / "priority1_raycasting_20261008b" / "summary.json"
    summary = _json(path, errors)
    if not summary:
        return
    dataset = _dataset_name(path, summary)
    conditions = (summary.get("dataset") or {}).get("test_visible_conditions") or {}

    def condition_parts(condition: str) -> tuple[str, str, bool]:
        bits = str(condition).split("_")
        pixel = "noisy" if bits and bits[0] == "noisy" else "clean"
        camera = bits[-1] if bits else "unknown"
        return pixel, camera, pixel == "clean" and camera == "true"

    for condition, details in (summary.get("protected_route_benchmark") or {}).items():
        if not isinstance(details, dict):
            continue
        pixel, camera, oracle = condition_parts(str(condition))
        total = conditions.get(condition)
        for branch in ("ray_only", "protected_route"):
            metric = details.get(branch)
            if not isinstance(metric, dict) or _num(metric.get("mae_m")) is None:
                continue
            _row_3d(
                rows, model="Priority1 Ray-casting", branch=branch,
                dataset=dataset, condition=str(condition), camera_source=camera,
                pixel_input=pixel, metric=metric, source=_source(path),
                protocol="multi-ray protected-route benchmark", oracle=oracle,
                total_hint=total, notes="valid Ray-casting is protected; Scene-MLP is fallback only",
            )

    for ray_count, details in (summary.get("ray_count_benchmark") or {}).items():
        if not isinstance(details, dict):
            continue
        for condition, metric_bundle in details.items():
            if not isinstance(metric_bundle, dict):
                continue
            metric = metric_bundle.get("metrics")
            if not isinstance(metric, dict) or _num(metric.get("mae_m")) is None:
                continue
            pixel, camera, oracle = condition_parts(str(condition))
            _row_3d(
                rows, model="Priority1 Ray-casting", branch=f"{ray_count}_rays",
                dataset=dataset, condition=str(condition), camera_source=camera,
                pixel_input=pixel, metric=metric, source=_source(path),
                protocol="ray-count benchmark", oracle=oracle,
                total_hint=conditions.get(condition), notes=f"ray_count={ray_count}",
            )

    temporal = summary.get("temporal_benchmark") or {}
    condition = str(temporal.get("condition", "noisy_estimated"))
    pixel, camera, oracle = condition_parts(condition)
    total = conditions.get(condition)
    for branch in ("raw", "ema", "ekf"):
        value = temporal.get(branch)
        metric = value.get("metrics") if isinstance(value, dict) else None
        if not isinstance(metric, dict) or _num(metric.get("mae_m")) is None:
            continue
        _row_3d(
            rows, model="Priority1 Ray-casting temporal", branch=branch,
            dataset=dataset, condition=condition, camera_source=camera,
            pixel_input=pixel, metric=metric, source=_source(path),
            protocol="validation-selected EMA/EKF on 5-ray stream", oracle=oracle,
            total_hint=total, notes="temporal benchmark parameters selected on validation",
        )


def _add_legacy_workflows(rows: list[Row], errors: list[str]) -> None:
    # These are useful historical baselines; they are opt-in so repeated smoke
    # runs do not dominate the default current-model table.
    paths = [
        (ROOT / "output" / "asset_backed_cpu_mixed_20261008" / "summary.json", "asset-backed five-workflow"),
        (ROOT / "output" / "post_roi_3d_benchmark" / "full_ema_mixed" / "summary.json", "paper post-ROI workflow"),
        (ROOT / "output" / "3d_rerun_standard_mixed_20261008" / "summary.json", "standard mixed five-workflow"),
        (ROOT / "output" / "3d_rerun_cpu_mixed_estimated_20261008" / "summary.json", "estimated-camera five-workflow"),
    ]
    for path, label in paths:
        if path.is_file():
            _add_five_workflow(rows, errors, path, label)


def _refresh_2d(output: Path, errors: list[str]) -> Optional[Path]:
    """Run the existing CCTV comparison and save a machine-readable summary."""
    try:
        import compare_v3_roi as compare
    except Exception as exc:
        errors.append(f"cannot import compare_v3_roi for --refresh-2d: {exc}")
        return None
    args = argparse.Namespace(
        labels=ROOT / "fire-model-data" / "dataset_labels (1).json",
        dataset=compare.CCTV_DATASET,
        baseline=ROOT / "fire-model-data" / "best.pth",
        roi=ROOT / "week6_roi_result" / "best_roi.pth",
        v3=compare.default_v3_checkpoint(), output_dir=output / "2d_refresh",
        split="test", max_images=0, selection="even", device=None, threshold=0.5, seed=42,
    )
    try:
        rows = compare.evaluate(args)
        result: dict[str, Any] = {
            "dataset": "real_cctv_labeled_test", "total": len(rows),
            "protocol": "fresh compare_v3_roi evaluation", "rows": [],
        }
        for branch in ("coarse", "roi", "v3"):
            errors_px = []
            for row in rows:
                prediction = getattr(row, branch)
                error = compare._prediction_error(prediction, row.gt)
                if error is not None:
                    errors_px.append(float(error))
            metric = compare._metric(errors_px, len(rows))
            result["rows"].append({"model": branch, "branch": branch, **metric})
        path = output / "2d_refresh" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception as exc:
        errors.append(f"--refresh-2d failed: {exc}")
        return None


def _rank(rows: list[Row]) -> None:
    groups: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        if row.mae is None:
            continue
        groups.setdefault((row.dimension, row.comparison_group), []).append(row)
    for group_rows in groups.values():
        # The requested rank order is weakest to strongest (largest error first).
        group_rows.sort(key=lambda item: item.mae if item.mae is not None else float("-inf"), reverse=True)
        for rank, row in enumerate(group_rows, 1):
            row.rank = rank


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{float(value):.{digits}f}"


def _print_table(title: str, rows: list[Row], limit: int) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no rows)")
        return
    print("rank  model                                  branch                 dataset                         cond                 MAE      median      P95   valid  cov    oracle")
    shown = sorted(
        rows,
        key=lambda x: (x.comparison_group, -(x.mae if x.mae is not None else float("-inf"))),
    )
    if limit > 0:
        shown = shown[:limit]
    for row in shown:
        model = row.model[:38]
        branch = row.branch[:22]
        dataset = row.dataset[:30]
        cond = row.condition[:20]
        valid = f"{row.valid_count}/{row.total_count}" if row.valid_count is not None else "-"
        cov = _fmt(row.valid_rate, 3)
        print(f"{str(row.rank or '-'):>4}  {model:<38} {branch:<22} {dataset:<30} {cond:<20} {_fmt(row.mae):>8} {_fmt(row.median):>10} {_fmt(row.p95):>9} {valid:>8} {cov:>5} {str(row.oracle):>6}")
    if limit > 0 and len(rows) > limit:
        print(f"... {len(rows) - limit} more rows are in comparison.csv/json (use --print-limit 0)")


def _write(rows: list[Row], diagnostics: list[dict[str, Any]], errors: list[str], output: Path, registry: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = {
        "format": "LAB_SAM.compare_all_models.v1",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "registry": registry,
        "row_count": len(rows),
        "rows": [asdict(row) for row in rows],
        "diagnostics": diagnostics,
        "errors_and_missing_artifacts": errors,
        "notes": [
            "2D rows are pixel MAE and 3D rows are metric XYZ MAE in metres.",
            "Rows are ranked from weakest to strongest only inside their comparison_group; datasets and camera conditions are not silently mixed.",
            "oracle=true rows are upper-bound diagnostics and are excluded from the non-oracle candidate table.",
            "known_camera=true means the saved evaluation used a true/measured camera pose rather than an estimated one.",
        ],
    }
    (output / "comparison.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = list(Row.__dataclass_fields__.keys())
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-legacy", action="store_true", help="include older full benchmark representatives")
    parser.add_argument("--exclude-oracle", action="store_true", help="omit oracle/upper-bound rows from printed and saved tables")
    parser.add_argument("--refresh-2d", action="store_true", help="rerun compare_v3_roi on the labelled CCTV test split")
    parser.add_argument("--print-limit", type=int, default=60, help="rows per printed table; 0 prints all")
    args = parser.parse_args()

    errors: list[str] = []
    rows: list[Row] = []
    registry: list[str] = []
    refresh_summary = _refresh_2d(args.output_dir, errors) if args.refresh_2d else None
    if args.refresh_2d:
        registry.append("fresh compare_v3_roi via --refresh-2d")

    before = len(rows); _add_roi_2d(rows, errors); registry.append("roi_best_rerun_20261008/summary.json"); registry.append(f"roi rows={len(rows)-before}")
    before = len(rows); _add_terminal_2d(rows, errors, refresh_summary); registry.append(f"CCTV terminal rows={len(rows)-before}")
    before = len(rows); _add_five_workflow(rows, errors, ROOT / "output" / "workflow_2d_then_3d_20261008" / "3d_workflows" / "summary.json", "workflow 2D->3D current"); registry.append(f"five-workflow current rows={len(rows)-before}")
    before = len(rows); _add_low_cost(rows, errors); registry.append(f"low-cost current rows={len(rows)-before}")
    before = len(rows); _add_scene_coordinate(rows, errors); registry.append(f"scene-coordinate current rows={len(rows)-before}")
    before = len(rows); _add_ablation(rows, errors); registry.append(f"ablation current rows={len(rows)-before}")
    before = len(rows); _add_temporal(rows, errors); registry.append(f"temporal current rows={len(rows)-before}")
    before = len(rows); _add_physics(rows, errors); registry.append(f"physics current rows={len(rows)-before}")
    before = len(rows); _add_siren(rows, errors); registry.append(f"SIREN/ReLU current rows={len(rows)-before}")
    before = len(rows); _add_multiplane(rows, errors); registry.append(f"multi-plane current rows={len(rows)-before}")
    before = len(rows); _add_priority1_raycasting(rows, errors); registry.append(f"priority1 Ray-casting rows={len(rows)-before}")
    if args.include_legacy:
        before = len(rows); _add_legacy_workflows(rows, errors); registry.append(f"legacy workflow rows={len(rows)-before}")

    diagnostics: list[dict[str, Any]] = []
    inference = _json(ROOT / "output" / "real_fire_inference" / "summary.json", errors)
    if inference:
        diagnostics.append({"name": "real media detector domain shift", "source": _source(ROOT / "output" / "real_fire_inference" / "summary.json"), "metric": "detection_rate", "value": (inference.get("image_summary") or {}).get("detection_rate"), "units": "fraction", "ranked": False})

    _rank(rows)
    saved_rows = [row for row in rows if not (args.exclude_oracle and row.oracle)]
    _write(saved_rows, diagnostics, errors, args.output_dir, registry)

    print(f"aggregated_rows={len(saved_rows)} output_dir={args.output_dir}")
    print(f"missing_or_parse_warnings={len(errors)}")
    _print_table("2D ranking (weakest -> strongest; pixel MAE; lower is better)", [r for r in saved_rows if r.dimension == "2d"], args.print_limit)
    _print_table("3D ranking (weakest -> strongest; metre MAE; conditions/camera/oracle stay in separate groups)", [r for r in saved_rows if r.dimension == "3d"], args.print_limit)
    candidates = [r for r in saved_rows if r.dimension == "3d" and not r.oracle and r.camera_source == "estimated" and (r.valid_rate is None or r.valid_rate >= 0.80)]
    _print_table("3D non-oracle estimated-camera candidates (coverage >= 0.80)", candidates, args.print_limit)
    if diagnostics:
        print("\nUnranked diagnostics")
        for item in diagnostics:
            print(f"{item['name']}: {item['metric']}={item['value']} {item['units']}")
    if errors:
        print("\nWarnings (see comparison.json for the full list)")
        for item in errors[:12]:
            print(f"- {item}")
    print(f"saved_json={args.output_dir / 'comparison.json'}")
    print(f"saved_csv={args.output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
