"""Audit the apparent ROI metric jump between the saved report and fresh runs.

The historical benchmark in ``benchmark tổng hợp.docx`` reports the ROI
checkpoints on a saved/simulated coarse-point protocol.  The fresh benchmark
``benchmark_roi_checkpoints.py`` runs ``best.pth`` first and feeds its actual
detector point to every ROI checkpoint.  These are different experiments.

This read-only audit evaluates the same positive CCTV test records under both
protocols and writes a new report.  It never modifies checkpoints, datasets or
existing output directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "little") ^ int(seed)


def _simulated_coarse(
    gt: np.ndarray,
    width: int,
    height: int,
    noise_px: float,
    image_path: str,
    seed: int,
) -> np.ndarray:
    """Match roi_dataset_loader._simulated_coarse exactly."""
    if noise_px <= 0.0:
        return gt.copy()
    rng = np.random.default_rng(_stable_seed(image_path, seed))
    delta = rng.normal(
        loc=0.0,
        scale=[float(noise_px) / max(width, 1), float(noise_px) / max(height, 1)],
        size=2,
    )
    return np.clip(gt + delta * np.asarray([width, height], dtype=np.float64), 0.01 * np.asarray([width, height]), 0.99 * np.asarray([width, height]))


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


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        keys = (
            "architecture",
            "experiment",
            "epoch",
            "val_loss",
            "backbone",
            "roi_fraction",
            "heatmap_size",
            "temperature",
            "dropout",
            "weight_decay",
            "coarse_definition",
        )
        return {key: checkpoint.get(key) for key in keys if key in checkpoint}
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"metadata_error": str(exc)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "checkpoint",
        "coarse_protocol",
        "coarse_mae_px",
        "refined_mae_px",
        "refined_median_px",
        "refined_p95_px",
        "refined_pck10",
        "refined_pck25",
        "detected",
        "total",
        "difference_vs_docx_px",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_high_error_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image",
        "actual_detector_coarse_error_px",
        "cpu_mixed_refined_error_px",
        "standard_mixed_refined_error_px",
        "gt_pixel",
        "actual_detector_coarse",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from compare_v3_roi import _records_for_split, _run_detector
    from fire_detector import FireDetector
    from narrow_localizer import ROIRefinerInference
    from train_week6 import load_records

    records, load_stats = load_records(args.labels, args.dataset)
    records = _records_for_split(records, "test", args.seed)
    if not records:
        raise RuntimeError("No positive test records were resolved")

    checkpoint_items = {
        "week6_baseline": args.week6,
        "standard_real": args.standard_real,
        "standard_synthetic": args.standard_synthetic,
        "standard_mixed": args.standard_mixed,
        "cpu_regularized_mixed": args.mixed,
    }
    for name, path in checkpoint_items.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint {name}: {path}")

    detector = FireDetector(args.baseline, device=args.device, threshold=args.threshold)
    if hasattr(detector, "warmup"):
        detector.warmup(repeats=1)
    refiners = {
        name: ROIRefinerInference(path, device=args.device)
        for name, path in checkpoint_items.items()
    }

    # Values transcribed from benchmark tổng hợp.docx. They are used only as
    # an audit reference; fresh numbers below are computed from the images.
    docx_reference = {
        "standard_real": 5.236342324150933,
        "standard_synthetic": 2.7684032050046055,
        "standard_mixed": 4.2813825607299805,
        "cpu_regularized_mixed": 4.329145908355713,
        "week6_baseline": 6.4112181133694115,
    }

    protocols = ("actual_detector", "simulated_gaussian_3px")
    per_image: list[dict[str, Any]] = []
    errors: dict[str, dict[str, list[float]]] = {
        name: {protocol: [] for protocol in protocols}
        for name in checkpoint_items
    }
    coarse_errors: dict[str, list[float]] = {protocol: [] for protocol in protocols}

    for index, record in enumerate(records, 1):
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt = np.asarray(
            [float(record.x_norm * width), float(record.y_norm * height)],
            dtype=np.float64,
        )
        prediction = _run_detector(detector, image, args.threshold)
        actual_coarse = None if prediction.point is None else np.asarray(prediction.point, dtype=np.float64)
        simulated_coarse = _simulated_coarse(
            gt,
            width,
            height,
            args.simulated_noise_px,
            str(record.image_path),
            args.seed,
        )
        coarse_points = {
            "actual_detector": actual_coarse,
            "simulated_gaussian_3px": simulated_coarse,
        }
        row: dict[str, Any] = {
            "image": Path(record.image_path).name,
            "image_path": str(record.image_path),
            "gt_pixel": gt,
            "coarse": {
                "actual_detector": actual_coarse,
                "simulated_gaussian_3px": simulated_coarse,
                "actual_confidence": float(prediction.confidence),
                "actual_detected": bool(prediction.detected),
            },
            "roi": {},
        }
        for protocol, coarse in coarse_points.items():
            if coarse is not None:
                coarse_error = float(np.linalg.norm(coarse - gt))
                coarse_errors[protocol].append(coarse_error)
            for name, refiner in refiners.items():
                row["roi"].setdefault(name, {})
                if coarse is None:
                    row["roi"][name][protocol] = {
                        "point": None,
                        "error_px": None,
                        "status": "upstream_detector_miss",
                    }
                    continue
                refined = refiner.refine(image, coarse)
                point = np.asarray(refined.point, dtype=np.float64)
                error_px = float(np.linalg.norm(point - gt))
                errors[name][protocol].append(error_px)
                row["roi"][name][protocol] = {
                    "point": point,
                    "error_px": error_px,
                    "confidence": float(refined.confidence),
                    "status": "valid_roi",
                }
        per_image.append(row)
        if index == 1 or index == len(records) or index % 10 == 0:
            print(f"processed={index}/{len(records)} image={row['image']}", flush=True)

    metrics: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for name in checkpoint_items:
        metrics[name] = {}
        for protocol in protocols:
            item = _metric(errors[name][protocol], len(records))
            coarse_values = coarse_errors[protocol]
            coarse_metric = _metric(coarse_values, len(records))
            metrics[name][protocol] = {
                "coarse": coarse_metric,
                "refined": item,
                "difference_vs_docx_px": (
                    None
                    if item["mae_px"] is None
                    else float(item["mae_px"] - docx_reference[name])
                ),
            }
            csv_rows.append(
                {
                    "checkpoint": name,
                    "coarse_protocol": protocol,
                    "coarse_mae_px": coarse_metric["mae_px"],
                    "refined_mae_px": item["mae_px"],
                    "refined_median_px": item["median_px"],
                    "refined_p95_px": item["p95_px"],
                    "refined_pck10": item["pck10"],
                    "refined_pck25": item["pck25"],
                    "detected": item["detected"],
                    "total": item["total"],
                    "difference_vs_docx_px": metrics[name][protocol]["difference_vs_docx_px"],
                }
            )

    high_error = sorted(
        [
            {
                "image": row["image"],
                "gt_pixel": row["gt_pixel"],
                "actual_detector_coarse": row["coarse"]["actual_detector"],
                "actual_detector_coarse_error_px": (
                    None
                    if row["coarse"]["actual_detector"] is None
                    else float(np.linalg.norm(np.asarray(row["coarse"]["actual_detector"]) - np.asarray(row["gt_pixel"])))
                ),
                "cpu_mixed_refined_error_px": row["roi"]["cpu_regularized_mixed"]["actual_detector"]["error_px"],
                "standard_mixed_refined_error_px": row["roi"]["standard_mixed"]["actual_detector"]["error_px"],
            }
            for row in per_image
        ],
        key=lambda item: float(item["cpu_mixed_refined_error_px"] or -1.0),
        reverse=True,
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "LAB_SAM.roi_metric_discrepancy_audit.v1",
        "source_report": "D:/ITK_SNAP/data/benchmark tổng hợp.docx",
        "protocol": {
            "test_positive_records": len(records),
            "same_test_images": True,
            "device": args.device,
            "threshold": args.threshold,
            "simulated_noise_px": args.simulated_noise_px,
            "actual_detector_definition": "best.pth produces coarse point, then the ROI refiner receives that point",
            "historical_report_definition": "saved/simulated coarse point; reconstructed as deterministic Gaussian noise around p_fire",
            "warning": "The two coarse protocols are not interchangeable. The 9.x px result is the end-to-end detector-to-ROI result; the 4.x px result is the easier simulated-coarse result.",
        },
        "dataset": {
            "labels": str(args.labels),
            "root": str(args.dataset),
            "loaded_stats": load_stats,
        },
        "checkpoints": {
            name: {"path": str(path), "metadata": _checkpoint_meta(path), "docx_reference_mae_px": docx_reference[name]}
            for name, path in checkpoint_items.items()
        },
        "metrics": metrics,
        "high_error_images": high_error[:10],
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
    (output / "high_error_images.json").write_text(
        json.dumps(high_error, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _write_csv(output / "metrics.csv", csv_rows)
    _write_high_error_csv(output / "high_error_images.csv", high_error)

    print("checkpoint protocol refined_MAE_px difference_vs_DOCX_px")
    for row in csv_rows:
        print(
            f"{row['checkpoint']:<24} {row['coarse_protocol']:<24} "
            f"{row['refined_mae_px']!s:<16} {row['difference_vs_docx_px']!s}"
        )
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "fire-detection-from-cctv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--week6", type=Path, default=ROOT / "week6_roi_result" / "best_roi.pth")
    parser.add_argument("--standard-real", type=Path, default=ROOT / "archives" / "cleanup_20261008" / "pth" / "output" / "roi_domain_experiments" / "real" / "best_roi.pth")
    parser.add_argument("--standard-synthetic", type=Path, default=ROOT / "archives" / "cleanup_20261008" / "pth" / "output" / "roi_domain_experiments" / "synthetic" / "best_roi.pth")
    parser.add_argument("--standard-mixed", type=Path, default=ROOT / "archives" / "cleanup_20261008" / "pth" / "output" / "roi_domain_experiments" / "mixed" / "best_roi.pth")
    parser.add_argument("--mixed", type=Path, default=ROOT / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "audit_roi_metric_discrepancy_20261009")
    parser.add_argument("--simulated-noise-px", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for field in (
        "labels",
        "dataset",
        "baseline",
        "week6",
        "standard_real",
        "standard_synthetic",
        "standard_mixed",
        "mixed",
        "output_dir",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    run(args)


if __name__ == "__main__":
    main()
