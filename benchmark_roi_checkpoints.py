"""Full held-out 2D benchmark for the available ROI checkpoints.

The upstream detector is evaluated once on the same positive CCTV test images.
Every ROI checkpoint then receives exactly the same image and coarse point.
This keeps the comparison focused on ROI refinement instead of changing the
test set or the upstream detector between runs.

The script writes a new machine-readable report and never changes checkpoints
or datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _metric(errors: list[float], total: int) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    result: dict[str, Any] = {
        "detected": int(len(values)),
        "total": int(total),
        "coverage": float(len(values) / total) if total else 0.0,
        "mae_px": None,
        "median_px": None,
        "p95_px": None,
        "pck10": None,
        "pck25": None,
    }
    if len(values):
        result.update(
            {
                "mae_px": float(values.mean()),
                "median_px": float(np.median(values)),
                "p95_px": float(np.percentile(values, 95)),
                "pck10": float(np.mean(values <= 10.0)),
                "pck25": float(np.mean(values <= 25.0)),
            }
        )
    return result


def _latency(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return {
            key: checkpoint.get(key)
            for key in (
                "architecture",
                "backbone",
                "experiment",
                "epoch",
                "val_loss",
                "roi_fraction",
                "heatmap_size",
                "temperature",
                "dropout",
                "weight_decay",
                "early_stopping",
                "coarse_definition",
            )
            if key in checkpoint
        }
    except Exception as exc:
        return {"metadata_error": str(exc)}


def _format(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    from compare_v3_roi import _normalise_prediction, _records_for_split, _run_detector
    from fire_detector import FireDetector
    from narrow_localizer import ROIRefinerInference
    from train_week6 import load_records

    records, load_stats = load_records(args.labels, args.dataset)
    records = _records_for_split(records, args.split, args.seed)
    if args.max_images > 0:
        records = records[: args.max_images]
    if not records:
        raise RuntimeError(f"No positive records found for split={args.split}")

    checkpoint_items = {
        "week6_baseline": args.week6,
        "standard_real": args.standard_real,
        "standard_synthetic": args.standard_synthetic,
        "standard_mixed": args.standard_mixed,
        "cpu_regularized_mixed": args.mixed,
    }
    for name, path in checkpoint_items.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing ROI checkpoint for {name}: {path}")

    # Soft-gate mode keeps the raw TTA point available and lets the refiner
    # enlarge its crop.  A hard gate would remove the very samples this
    # experiment is designed to recover.
    detector_gate = None if args.roi_expansion_factor > 0.0 else args.max_stability_px
    detector = FireDetector(
        args.baseline,
        device=args.device,
        threshold=args.threshold,
        tta_mode=args.tta_mode,
        tta_reliability_scale_px=args.tta_reliability_scale_px,
        max_stability_px=detector_gate,
    )
    if hasattr(detector, "warmup"):
        detector.warmup(repeats=1)
    refiners = {
        name: ROIRefinerInference(path, device=args.device)
        for name, path in checkpoint_items.items()
    }

    per_image: list[dict[str, Any]] = []
    branch_errors: dict[str, list[float]] = {"coarse": []}
    branch_latency: dict[str, list[float]] = {"coarse": []}
    for name in checkpoint_items:
        branch_errors[name] = []
        branch_latency[name] = []

    for index, record in enumerate(records, 1):
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt = np.asarray([float(record.x_norm * width), float(record.y_norm * height)], dtype=np.float64)
        coarse = _run_detector(detector, image, args.threshold)
        coarse_error = None
        if coarse.point is not None:
            coarse_error = float(np.linalg.norm(np.asarray(coarse.point) - gt))
            branch_errors["coarse"].append(coarse_error)
        branch_latency["coarse"].append(float(coarse.latency_ms))
        row: dict[str, Any] = {
            "image": Path(record.image_path).name,
            "image_path": str(record.image_path),
            "gt_pixel": gt,
            "coarse": {
                "point": coarse.point,
                "confidence": coarse.confidence,
                "detected": coarse.detected,
                "stability_px": coarse.stability_px,
                "position_reliability": coarse.position_reliability,
                "stability_reference_px": coarse.stability_reference_px,
                "raw_point": coarse.raw_point,
                "localization_accepted": coarse.localization_accepted,
                "localization_status": coarse.localization_status,
                "error_px": coarse_error,
                "latency_ms": coarse.latency_ms,
            },
            "roi": {},
        }
        for name, refiner in refiners.items():
            if coarse.point is None:
                row["roi"][name] = {
                    "point": None,
                    "confidence": 0.0,
                    "detected": False,
                    "error_px": None,
                    "latency_ms": None,
                    "status": "upstream_coarse_miss",
                }
                continue
            started = time.perf_counter()
            refined = refiner.refine(
                image,
                coarse.point,
                stability_px=coarse.stability_px,
                expansion_factor=args.roi_expansion_factor,
                max_roi_fraction=args.roi_max_fraction,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            point = np.asarray(refined.point, dtype=np.float64)
            error_px = float(np.linalg.norm(point - gt))
            branch_errors[name].append(error_px)
            branch_latency[name].append(latency_ms)
            row["roi"][name] = {
                "point": point,
                "confidence": float(refined.confidence),
                "detected": True,
                "error_px": error_px,
                "latency_ms": latency_ms,
                "status": "valid_roi",
                "roi_side_px": float(refined.roi.side),
            }
        per_image.append(row)
        if index == 1 or index == len(records) or index % 10 == 0:
            print(f"processed={index}/{len(records)} image={row['image']}", flush=True)

    metrics: dict[str, Any] = {}
    for name, errors in branch_errors.items():
        metrics[name] = {
            **_metric(errors, len(records)),
            "latency": _latency(branch_latency[name]),
        }

    mixed = metrics["cpu_regularized_mixed"]
    for name, item in metrics.items():
        if name == "cpu_regularized_mixed" or item["mae_px"] is None or mixed["mae_px"] is None:
            item["mae_improvement_vs_cpu_regularized_mixed_px"] = 0.0 if name == "cpu_regularized_mixed" else None
        else:
            item["mae_improvement_vs_cpu_regularized_mixed_px"] = float(item["mae_px"] - mixed["mae_px"])

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "LAB_SAM.roi_checkpoint_benchmark.v1",
        "protocol": {
            "split": args.split,
            "same_positive_images_for_all_branches": True,
            "upstream_detector_run_once": True,
            "no_retraining": True,
            "threshold": args.threshold,
            "tta_mode": args.tta_mode,
            "tta_reliability_scale_px": args.tta_reliability_scale_px,
            "max_stability_px": args.max_stability_px,
            "roi_expansion_factor": args.roi_expansion_factor,
            "roi_max_fraction": args.roi_max_fraction,
            "detector_gate_effective": detector_gate,
            "device": args.device,
            "warning": "ROI metrics are evaluated on visible positive fire images after the upstream detector. Overall fire classification recall is separate.",
        },
        "dataset": {
            "labels": str(args.labels),
            "root": str(args.dataset),
            "loaded_stats": load_stats,
            "positive_test_images": len(records),
        },
        "checkpoints": {
            name: {"path": str(path), "metadata": _checkpoint_meta(path)}
            for name, path in checkpoint_items.items()
        },
        "metrics": metrics,
        "per_image": per_image,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    (output / "per_image.json").write_text(
        json.dumps(per_image, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    fields = [
        "branch", "detected", "total", "coverage", "mae_px", "median_px", "p95_px",
        "pck10", "pck25", "latency_mean_ms", "latency_p95_ms",
        "mae_improvement_vs_cpu_regularized_mixed_px",
    ]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, item in metrics.items():
            writer.writerow(
                {
                    "branch": name,
                    "detected": item["detected"],
                    "total": item["total"],
                    "coverage": item["coverage"],
                    "mae_px": item["mae_px"],
                    "median_px": item["median_px"],
                    "p95_px": item["p95_px"],
                    "pck10": item["pck10"],
                    "pck25": item["pck25"],
                    "latency_mean_ms": item["latency"]["mean_ms"],
                    "latency_p95_ms": item["latency"]["p95_ms"],
                    "mae_improvement_vs_cpu_regularized_mixed_px": item["mae_improvement_vs_cpu_regularized_mixed_px"],
                }
            )

    print("branch                 detected/total  MAE(px) median(px) P95(px) PCK@10 PCK@25 latency(ms)")
    for name, item in metrics.items():
        print(
            f"{name:<22} {item['detected']:>3}/{item['total']:<3} "
            f"{_format(item['mae_px']):>8} {_format(item['median_px']):>10} "
            f"{_format(item['p95_px']):>8} {_format(item['pck10']):>7} {_format(item['pck25']):>7} "
            f"{_format(item['latency']['mean_ms']):>10}"
        )
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "fire-detection-from-cctv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--week6", type=Path, default=ROOT / "week6_roi_result" / "best_roi.pth")
    parser.add_argument("--standard-real", type=Path, default=ROOT / "archives" / "cleanup_20261008" / "pth" / "output" / "roi_domain_experiments" / "real" / "best_roi.pth")
    parser.add_argument("--standard-synthetic", type=Path, default=ROOT / "archives" / "cleanup_20261008" / "pth" / "output" / "roi_domain_experiments" / "synthetic" / "best_roi.pth")
    parser.add_argument("--standard-mixed", type=Path, default=ROOT / "archives" / "cleanup_20261008" / "pth" / "output" / "roi_domain_experiments" / "mixed" / "best_roi.pth")
    parser.add_argument("--mixed", type=Path, default=ROOT / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "roi_checkpoint_benchmark_20261009")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all positive images")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tta-mode",
        choices=("none", "brightness3", "photometric5"),
        default="none",
        help="Optional photometric TTA for the upstream coarse detector",
    )
    parser.add_argument("--tta-reliability-scale-px", type=float, default=8.0)
    parser.add_argument("--max-stability-px", type=float, default=None)
    parser.add_argument(
        "--roi-expansion-factor", type=float, default=0.0,
        help="Soft-gate crop expansion in pixels per TTA stability_px; 0 disables it",
    )
    parser.add_argument(
        "--roi-max-fraction", type=float, default=1.0,
        help="Maximum dynamic crop side as a fraction of the longer image dimension",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for field in ("labels", "dataset", "baseline", "week6", "standard_real", "standard_synthetic", "standard_mixed", "mixed", "output_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
