"""Evaluate EMA and 3D Kalman filtering after the scene-coordinate MLP.

This is a strict follow-up to ``benchmark_3d_ablations.py``.  The MLP
checkpoints are trained once per pixel/pose condition and loaded here.  EMA
and Kalman parameters are selected on the validation scenes only; the test
scenes remain untouched until the final report.

The input remains a post-ROI pixel plus camera metadata.  Missing noisy
observations are not replaced with clean labels and are counted as misses.
All output is written to a new directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from benchmark_3d_ablations import (
    VARIANTS,
    _load_rows,
    _valid_samples,
)
from ekf_tracker import Fire3DEKF
from scene_coordinate_regression import SceneCoordinateMLP, metric_summary


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _group_predictions(
    samples: list[dict[str, Any]],
    model: SceneCoordinateMLP,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        prediction = model.predict_features(
            np.asarray(sample["features"], dtype=np.float32).reshape(1, -1)
        )[0].astype(np.float64)
        records.append(
            {
                "scene_id": sample["scene_id"],
                "surface": sample["surface"],
                "frame_index": int(sample["row"].get("frame_index", 0)),
                "sample_id": sample["row"].get("sample_id"),
                "target": np.asarray(sample["target"], dtype=np.float64),
                "raw": prediction,
                "mlp_latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    return records


def _apply_ema(records: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["scene_id"])].append(record)
    for scene_id, values in groups.items():
        values.sort(key=lambda item: item["frame_index"])
        state: Optional[np.ndarray] = None
        for record in values:
            started = time.perf_counter()
            measurement = np.asarray(record["raw"], dtype=np.float64)
            state = measurement.copy() if state is None else alpha * measurement + (1.0 - alpha) * state
            output.append(
                {
                    **record,
                    "filter": "ema",
                    "parameter": {"alpha": alpha},
                    "prediction": state.copy(),
                    "filter_latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
    return output


def _apply_ekf(
    records: list[dict[str, Any]],
    process_accel_std: float,
    measurement_std_m: float,
    gate_mahalanobis2: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["scene_id"])].append(record)
    for scene_id, values in groups.items():
        values.sort(key=lambda item: item["frame_index"])
        tracker = Fire3DEKF(
            dt=1.0,
            process_accel_std=process_accel_std,
            measurement_std_m=measurement_std_m,
            gate_mahalanobis2=gate_mahalanobis2,
            max_missed=3,
        )
        for record in values:
            started = time.perf_counter()
            state = tracker.update(
                point=np.asarray(record["raw"], dtype=np.float64),
                covariance=None,
                confidence=1.0,
            )
            prediction = None if state.position is None else state.position.copy()
            output.append(
                {
                    **record,
                    "filter": "ekf",
                    "parameter": {
                        "process_accel_std": process_accel_std,
                        "measurement_std_m": measurement_std_m,
                        "gate_mahalanobis2": gate_mahalanobis2,
                    },
                    "prediction": prediction,
                    "filter_latency_ms": (time.perf_counter() - started) * 1000.0,
                    "accepted": bool(state.accepted),
                    "mahalanobis2": state.mahalanobis2,
                }
            )
    return output


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("prediction") is not None]
    if valid:
        truth = np.asarray([record["target"] for record in valid], dtype=np.float64)
        prediction = np.asarray([record["prediction"] for record in valid], dtype=np.float64)
        result = metric_summary(truth, prediction).to_dict()
    else:
        result = metric_summary(np.empty((0, 3)), np.empty((0, 3))).to_dict()
    latencies = [float(record.get("filter_latency_ms", 0.0)) for record in records]
    result.update(
        {
            "attempted": len(records),
            "valid_count": len(valid),
            "valid_rate": float(len(valid) / max(1, len(records))),
            "latency_ms": {
                "mean": None if not latencies else float(np.mean(latencies)),
                "median": None if not latencies else float(np.median(latencies)),
                "p95": None if not latencies else float(np.percentile(latencies, 95)),
            },
        }
    )
    return result


def _surface_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["surface"])].append(record)
    return {surface: _metrics(values) for surface, values in sorted(grouped.items())}


def _temporal_stability(records: list[dict[str, Any]], jump_threshold_m: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("prediction") is not None:
            grouped[str(record["scene_id"])].append(record)
    steps: list[float] = []
    sequences_with_step = 0
    for values in grouped.values():
        values.sort(key=lambda item: item["frame_index"])
        if len(values) >= 2:
            sequences_with_step += 1
        for previous, current in zip(values, values[1:]):
            steps.append(float(np.linalg.norm(current["prediction"] - previous["prediction"])))
    if not steps:
        return {
            "sequence_count_with_step": sequences_with_step,
            "step_count": 0,
            "mean_step_m": None,
            "median_step_m": None,
            "p95_step_m": None,
            "jump_threshold_m": jump_threshold_m,
            "jump_rate": None,
        }
    array = np.asarray(steps, dtype=np.float64)
    return {
        "sequence_count_with_step": sequences_with_step,
        "step_count": len(array),
        "mean_step_m": float(np.mean(array)),
        "median_step_m": float(np.median(array)),
        "p95_step_m": float(np.percentile(array, 95)),
        "jump_threshold_m": jump_threshold_m,
        "jump_rate": float(np.mean(array > jump_threshold_m)),
    }


def _evaluate_parameter(records: list[dict[str, Any]], filtered: list[dict[str, Any]], jump_threshold_m: float) -> dict[str, Any]:
    result = _metrics(filtered)
    result["metrics_by_surface"] = _surface_metrics(filtered)
    result["stability"] = _temporal_stability(filtered, jump_threshold_m)
    return result


def _select_best(candidates: list[tuple[dict[str, Any], dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    available = [item for item in candidates if item[1].get("mae_m") is not None]
    if not available:
        raise RuntimeError("No valid temporal-filter candidate on validation")
    return min(
        available,
        key=lambda item: (
            float(item[1]["mae_m"]),
            float(item[1]["stability"].get("mean_step_m") or float("inf")),
        ),
    )


def _candidate_filters(records: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    candidates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for alpha in args.ema_alphas:
        parameter = {"filter": "ema", "alpha": float(alpha)}
        candidates.append((parameter, _apply_ema(records, float(alpha))))
    for process_std in args.ekf_process_stds:
        for measurement_std in args.ekf_measurement_stds:
            for gate in args.ekf_gates:
                parameter = {
                    "filter": "ekf",
                    "process_accel_std": float(process_std),
                    "measurement_std_m": float(measurement_std),
                    "gate_mahalanobis2": float(gate),
                }
                candidates.append(
                    (
                        parameter,
                        _apply_ekf(
                            records,
                            float(process_std),
                            float(measurement_std),
                            float(gate),
                        ),
                    )
                )
    return candidates


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "variant",
        "raw_mae_m",
        "ema_mae_m",
        "ema_median_m",
        "ema_p95_m",
        "ema_mean_step_m",
        "ema_jump_rate",
        "ekf_mae_m",
        "ekf_median_m",
        "ekf_p95_m",
        "ekf_mean_step_m",
        "ekf_jump_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for variant, _, _ in VARIANTS:
            item = summary["variants"][variant]
            raw = item["test"]["raw"]
            ema = item["test"]["selected"]["ema"]["metrics"]
            ekf = item["test"]["selected"]["ekf"]["metrics"]
            writer.writerow(
                {
                    "variant": variant,
                    "raw_mae_m": raw.get("mae_m"),
                    "ema_mae_m": ema.get("mae_m"),
                    "ema_median_m": ema.get("median_m"),
                    "ema_p95_m": ema.get("p95_m"),
                    "ema_mean_step_m": ema["stability"].get("mean_step_m"),
                    "ema_jump_rate": ema["stability"].get("jump_rate"),
                    "ekf_mae_m": ekf.get("mae_m"),
                    "ekf_median_m": ekf.get("median_m"),
                    "ekf_p95_m": ekf.get("p95_m"),
                    "ekf_mean_step_m": ekf["stability"].get("mean_step_m"),
                    "ekf_jump_rate": ekf["stability"].get("jump_rate"),
                }
            )


def _plot(summary: dict[str, Any], output: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [item[0] for item in VARIANTS]
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.25

    def get_values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for label in labels:
            value: Any = summary["variants"][label]["test"]
            for key in path:
                value = value[key]
            result.append(np.nan if value is None else float(value))
        return result

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width, get_values(("raw", "mae_m")), width, label="raw MLP")
    axis.bar(positions, get_values(("selected", "ema", "metrics", "mae_m")), width, label="EMA")
    axis.bar(positions + width, get_values(("selected", "ekf", "metrics", "mae_m")), width, label="EKF")
    axis.set_xticks(positions, labels, rotation=20)
    axis.set_ylabel("3D MAE (m)")
    axis.set_title("Temporal filtering selected on validation, measured on test")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    mae_path = figure_dir / "temporal_mae.png"
    fig.savefig(mae_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width / 2, get_values(("selected", "ema", "metrics", "stability", "mean_step_m")), width, label="EMA")
    axis.bar(positions + width / 2, get_values(("selected", "ekf", "metrics", "stability", "mean_step_m")), width, label="EKF")
    axis.set_xticks(positions, labels, rotation=20)
    axis.set_ylabel("Mean consecutive step (m)")
    axis.set_title("Temporal jitter after filtering")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    jitter_path = figure_dir / "temporal_jitter.png"
    fig.savefig(jitter_path, dpi=160)
    plt.close(fig)
    return [str(mae_path), str(jitter_path)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    ablation_dir = args.ablation_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_predictions").mkdir(parents=True, exist_ok=True)

    rows = {split: _load_rows(dataset, split) for split in ("train", "val", "test")}
    summary: dict[str, Any] = {
        "format": "LAB_SAM.temporal_filter_ablation.v1",
        "dataset": {
            "root": str(dataset),
            "synthetic_metric_geometry": True,
            "warning": "Synthetic/asset-backed geometry result, not real CCTV accuracy.",
            "split_policy": "scene-separated; filter parameters selected on validation only",
        },
        "filter_search": {
            "ema_alphas": [float(value) for value in args.ema_alphas],
            "ekf_process_stds": [float(value) for value in args.ekf_process_stds],
            "ekf_measurement_stds": [float(value) for value in args.ekf_measurement_stds],
            "ekf_gates": [float(value) for value in args.ekf_gates],
            "jump_threshold_m": args.jump_threshold_m,
        },
        "variants": {},
    }
    csv_summary: dict[str, Any] = {"variants": summary["variants"]}

    for variant, pixel_key, camera_source in VARIANTS:
        checkpoint = ablation_dir / "checkpoints" / f"scene_mlp_{variant}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing MLP checkpoint for {variant}: {checkpoint}")
        model = SceneCoordinateMLP.load(checkpoint, device=args.device)
        split_records: dict[str, list[dict[str, Any]]] = {}
        skipped: dict[str, dict[str, int]] = {}
        for split in ("train", "val", "test"):
            samples, skipped[split] = _valid_samples(rows[split], pixel_key, camera_source)
            split_records[split] = _group_predictions(samples, model)

        val_raw_metrics = _metrics(
            [
                {
                    **record,
                    "prediction": record["raw"],
                    "filter_latency_ms": 0.0,
                }
                for record in split_records["val"]
            ]
        )
        val_candidates = _candidate_filters(split_records["val"], args)
        evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for parameter, filtered in val_candidates:
            evaluated.append(
                (
                    parameter,
                    _evaluate_parameter(filtered, filtered, args.jump_threshold_m),
                )
            )
        ema_best_parameter, ema_best_metrics = _select_best(
            [item for item in evaluated if item[0]["filter"] == "ema"]
        )
        ekf_best_parameter, ekf_best_metrics = _select_best(
            [item for item in evaluated if item[0]["filter"] == "ekf"]
        )

        test_raw_records = [
            {
                **record,
                "filter": "raw",
                "prediction": record["raw"],
                "filter_latency_ms": 0.0,
            }
            for record in split_records["test"]
        ]
        test_ema_records = _apply_ema(
            split_records["test"], float(ema_best_parameter["alpha"])
        )
        test_ekf_records = _apply_ekf(
            split_records["test"],
            float(ekf_best_parameter["process_accel_std"]),
            float(ekf_best_parameter["measurement_std_m"]),
            float(ekf_best_parameter["gate_mahalanobis2"]),
        )
        test_ema_metrics = _evaluate_parameter(test_ema_records, test_ema_records, args.jump_threshold_m)
        test_ekf_metrics = _evaluate_parameter(test_ekf_records, test_ekf_records, args.jump_threshold_m)
        test_raw_metrics = _evaluate_parameter(test_raw_records, test_raw_records, args.jump_threshold_m)
        summary["variants"][variant] = {
            "pixel_key": pixel_key,
            "camera_source": camera_source,
            "checkpoint": str(checkpoint),
            "skipped": skipped,
            "valid_samples": {split: len(values) for split, values in split_records.items()},
            "valid_scenes": {
                split: len({record["scene_id"] for record in values})
                for split, values in split_records.items()
            },
            "validation": {
                "raw": val_raw_metrics,
                "selected": {
                    "ema": {"parameter": ema_best_parameter, "metrics": ema_best_metrics},
                    "ekf": {"parameter": ekf_best_parameter, "metrics": ekf_best_metrics},
                },
            },
            "test": {
                "raw": test_raw_metrics,
                "selected": {
                    "ema": {"parameter": ema_best_parameter, "metrics": test_ema_metrics},
                    "ekf": {"parameter": ekf_best_parameter, "metrics": test_ekf_metrics},
                },
            },
        }
        selected_rows = test_ema_records + test_ekf_records
        (output / "selected_predictions" / f"{variant}.json").write_text(
            json.dumps(selected_rows, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        print(
            f"{variant}: raw={test_raw_metrics['mae_m']}m "
            f"EMA={test_ema_metrics['mae_m']}m "
            f"EKF={test_ekf_metrics['mae_m']}m",
            flush=True,
        )

    figure_paths = _plot(summary, output)
    summary["artifacts"] = {
        "summary": str(output / "summary.json"),
        "figures": figure_paths,
        "selected_predictions": str(output / "selected_predictions"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_csv(output / "temporal_metrics.csv", summary)
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--ablation-dir", type=Path, default=root / "output" / "ablation_3d_all_rerun_20261008")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "temporal_filter_ablation_20261008_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jump-threshold-m", type=float, default=0.5)
    parser.add_argument("--ema-alphas", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.5, 0.7])
    parser.add_argument("--ekf-process-stds", type=float, nargs="+", default=[0.05, 0.15, 0.35, 0.6])
    parser.add_argument("--ekf-measurement-stds", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--ekf-gates", type=float, nargs="+", default=[9.21, 16.27, 25.0])
    args = parser.parse_args()
    if args.jump_threshold_m <= 0:
        raise ValueError("jump-threshold-m must be positive")
    run(args)


if __name__ == "__main__":
    main()

