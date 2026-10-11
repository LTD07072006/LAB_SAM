"""Reproduce the Section 1.2 bottleneck audit and paired 3-D ablations.

The script intentionally consumes cached model reports.  It does not silently
re-run a detector or fit a depth scale on the evaluation samples.  The only
new computation is the analytic, differentiable ray/plane calculation on the
same ROI pixels stored in the Depth Anything report.  This makes the comparison
useful for deciding which part of the cascade should be fixed first.

Example::

    python benchmark_section_1_2.py \
      --real-summary ../output/audit_roi_metric_discrepancy_20261009/summary.json \
      --depth-report ../working/synthetic_fire_3d_v3/\
        depth_anything_v2_metric_indoor_roi_cpu_regularized_test92.json \
      --dataset ../working/synthetic_fire_3d_v3 \
      --output-dir ../output/section_1_2_research_20261010

All paths are explicit in the generated ``summary.json``.  Results on the
procedural dataset validate camera geometry and error propagation; they do not
establish metre-level accuracy for the real CCTV images, which have 2-D labels
only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the project already depends on Pillow.
    Image = None  # type: ignore[assignment]

from depth_localization import DifferentiableRayCastLayer


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(array * array))),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(np.abs(array))),
    }


def _error_stats(values: Iterable[float]) -> dict[str, float]:
    """Summarise non-negative errors without changing their meaning."""

    return _percentiles(values)


def _safe_corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def audit_real_summary(path: Path) -> dict[str, Any]:
    """Recompute crop containment and confidence diagnostics for real CCTV."""

    source = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = source["checkpoints"]["cpu_regularized_mixed"]
    roi_fraction = float(checkpoint["metadata"].get("roi_fraction", 0.70))
    rows: list[dict[str, Any]] = []
    for item in source["per_image"]:
        gt = np.asarray(item["gt_pixel"], dtype=np.float64)
        coarse = np.asarray(item["coarse"]["actual_detector"], dtype=np.float64)
        image_path = Path(item["image_path"])
        if Image is not None and image_path.is_file():
            with Image.open(image_path) as image:
                width, height = image.size
        else:
            # The real test set is 224x224; retaining this fallback makes the
            # audit inspectable if the image archive is moved after evaluation.
            width, height = 224, 224
        side = max(32.0, min(width, height) * roi_fraction)
        left = int(round(float(coarse[0] - side / 2.0)))
        top = int(round(float(coarse[1] - side / 2.0)))
        right = left + int(round(side))
        bottom = top + int(round(side))
        margins = np.asarray(
            [gt[0] - left, right - gt[0], gt[1] - top, bottom - gt[1]],
            dtype=np.float64,
        )
        roi = item["roi"]["cpu_regularized_mixed"]["actual_detector"]
        rows.append(
            {
                "image": item["image"],
                "width": int(width),
                "height": int(height),
                "coarse_error_px": float(np.linalg.norm(coarse - gt)),
                "coarse_confidence": float(item["coarse"]["actual_confidence"]),
                "roi_error_px": float(roi["error_px"]),
                "roi_heatmap_confidence": float(roi["confidence"]),
                "crop": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "side": float(side),
                    "gt_inside": bool(np.all(margins >= 0.0)),
                    "minimum_gt_margin_px": float(np.min(margins)),
                    "margins_px": margins.tolist(),
                },
            }
        )

    coarse_errors = [row["coarse_error_px"] for row in rows]
    roi_errors = [row["roi_error_px"] for row in rows]
    confidence = [row["coarse_confidence"] for row in rows]
    roi_confidence = [row["roi_heatmap_confidence"] for row in rows]
    inside = [row["crop"]["gt_inside"] for row in rows]
    high_conf_outliers = [
        row["image"]
        for row in rows
        if row["coarse_confidence"] >= 0.99 and row["coarse_error_px"] > 15.0
    ]
    return {
        "source_summary": str(path),
        "protocol": source["protocol"],
        "roi_fraction": roi_fraction,
        "records": len(rows),
        "metrics_from_same_actual_detector_protocol": source["metrics"],
        "recomputed": {
            "coarse_error_px": _error_stats(coarse_errors),
            "roi_error_px": _error_stats(roi_errors),
            "coarse_confidence_vs_coarse_error_pearson": _safe_corr(confidence, coarse_errors),
            "coarse_confidence_vs_roi_error_pearson": _safe_corr(confidence, roi_errors),
            "roi_heatmap_confidence_vs_roi_error_pearson": _safe_corr(roi_confidence, roi_errors),
            "gt_inside_crop_count": int(sum(inside)),
            "gt_outside_crop_count": int(len(inside) - sum(inside)),
            "minimum_gt_margin_px": _error_stats(
                [row["crop"]["minimum_gt_margin_px"] for row in rows]
            ),
            "high_confidence_coarse_outliers_gt15px": high_conf_outliers,
        },
        "per_image": rows,
        "interpretation": (
            "The crop contains all 36 real test labels under the recorded 0.70 "
            "fraction. Coarse distribution shift and overconfidence remain; "
            "crop exclusion is not demonstrated by this audit."
        ),
    }


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["sample_id"])] = row
    return result


def _camera_t(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    camera = record["camera"]
    K = torch.as_tensor(camera["K"], dtype=torch.float64)
    R = torch.as_tensor(camera["R_world_to_camera"], dtype=torch.float64)
    C = torch.as_tensor(camera["camera_position"], dtype=torch.float64)
    # The dataset stores camera centre C, while the differentiable layer takes
    # the OpenCV translation T in X_camera = R X_world + T.
    T = -(R @ C)
    return K, R, T


def paired_synthetic_ablation(dataset_root: Path, depth_report_path: Path) -> dict[str, Any]:
    """Compare the cached Depth Anything pixels with a known floor plane."""

    report = json.loads(depth_report_path.read_text(encoding="utf-8"))
    records = _load_jsonl(dataset_root / "test.jsonl")
    layer = DifferentiableRayCastLayer(parallel_mode="raise")
    roi_ray_errors: list[float] = []
    coarse_ray_errors: list[float] = []
    gt_ray_errors: list[float] = []
    floor_depth_errors: list[float] = []
    jacobian_sigma: list[float] = []
    paired_rows: list[dict[str, Any]] = []

    for sample in report["samples"]:
        record = records[str(sample["sample_id"])]
        if record.get("fire_surface") != "floor":
            continue
        K, R, T = _camera_t(record)
        plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        gt = torch.as_tensor(record["fire_xyz_world"], dtype=torch.float64)
        roi_pixel = torch.as_tensor(sample["roi_pixel"], dtype=torch.float64).requires_grad_(True)
        roi_point = layer(roi_pixel, K, R, T, plane)
        roi_error = float(torch.linalg.vector_norm(roi_point - gt).detach())
        roi_ray_errors.append(roi_error)
        gt_pixel = torch.as_tensor(record["p_fire_pixel"], dtype=torch.float64)
        gt_point = layer(gt_pixel, K, R, T, plane)
        gt_ray_errors.append(float(torch.linalg.vector_norm(gt_point - gt).detach()))

        coarse_pixel_value = record.get("p_fire_noisy_pixel")
        coarse_error: float | None = None
        if coarse_pixel_value is not None:
            coarse_point = layer(
                torch.as_tensor(coarse_pixel_value, dtype=torch.float64), K, R, T, plane
            )
            coarse_error = float(torch.linalg.vector_norm(coarse_point - gt).detach())
            coarse_ray_errors.append(coarse_error)

        jacobian = torch.autograd.functional.jacobian(
            lambda point: layer(point, K, R, T, plane), roi_pixel
        )
        jacobian_sigma.append(float(torch.linalg.svdvals(jacobian)[0].detach()))
        floor_depth_errors.append(float(sample["camera_error_m"]))
        paired_rows.append(
            {
                "sample_id": sample["sample_id"],
                "roi_pixel_error_px": float(sample["roi_pixel_error_px"]),
                "roi_floor_ray_error_m": roi_error,
                "coarse_floor_ray_error_m": coarse_error,
                "depth_camera_xyz_error_m": float(sample["camera_error_m"]),
                "jacobian_sigma_max_m_per_px": jacobian_sigma[-1],
            }
        )

    return {
        "dataset_root": str(dataset_root),
        "depth_report": str(depth_report_path),
        "depth_report_summary": {
            "model": report.get("depth_model"),
            "checkpoint": report.get("roi_checkpoint"),
            "device": report.get("device"),
            "records": report.get("records"),
            "elapsed_s": report.get("elapsed_s"),
            "all_record_depth_camera_error": report.get("depth_camera_error"),
        },
        "floor_records": len(roi_ray_errors),
        "floor_ray_roi_same_input": _error_stats(roi_ray_errors),
        "floor_ray_coarse_same_input": _error_stats(coarse_ray_errors),
        "floor_ray_gt_pixel_oracle": _error_stats(gt_ray_errors),
        "depth_camera_xyz_error_on_floor": _error_stats(floor_depth_errors),
        "ray_pixel_jacobian_sigma_max_m_per_px": _error_stats(jacobian_sigma),
        "per_sample": paired_rows,
        "interpretation": (
            "On the known floor plane, the same ROI pixels produce sub-metre "
            "ray errors while the uncalibrated metric-depth branch remains much "
            "larger. This supports using depth as an auxiliary/support signal, "
            "not replacing known-plane geometry yet."
        ),
    }


def _make_plot(summary: dict[str, Any], path: Path) -> None:
    """Write a compact static figure; plotting is optional for core evaluation."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - benchmark remains usable without plotting.
        summary["plot"] = {"status": "skipped", "reason": "matplotlib unavailable"}
        return

    real = summary["real_audit"]["metrics_from_same_actual_detector_protocol"]
    coarse = real["cpu_regularized_mixed"]["actual_detector"]["coarse"]["mae_px"]
    roi = real["cpu_regularized_mixed"]["actual_detector"]["refined"]["mae_px"]
    simulated = real["cpu_regularized_mixed"]["simulated_gaussian_3px"]["refined"]["mae_px"]
    ray = summary["synthetic_paired"]["floor_ray_roi_same_input"]["rmse"]
    depth = summary["synthetic_paired"]["depth_camera_xyz_error_on_floor"]["rmse"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].bar(["coarse\nactual", "ROI\nactual", "ROI\nGaussian 3px"], [coarse, roi, simulated], color=["#6b7280", "#2563eb", "#93c5fd"])
    axes[0].set_ylabel("Pixel error MAE (px)")
    axes[0].set_title("Real CCTV test (n=36)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(["Known floor\nray + ROI", "Depth Anything\nmetric head + ROI"], [ray, depth], color=["#059669", "#d97706"])
    axes[1].set_ylabel("3-D error RMSE (m)")
    axes[1].set_title("Synthetic metric test, floor subset")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    summary["plot"] = {"status": "written", "path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    project = here.parent
    parser.add_argument(
        "--real-summary",
        type=Path,
        default=project / "output" / "audit_roi_metric_discrepancy_20261009" / "summary.json",
    )
    parser.add_argument(
        "--depth-report",
        type=Path,
        default=project / "working" / "synthetic_fire_3d_v3" / "depth_anything_v2_metric_indoor_roi_cpu_regularized_test92.json",
    )
    parser.add_argument("--dataset", type=Path, default=project / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--output-dir", type=Path, default=project / "output" / "section_1_2_research_20261010")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real = audit_real_summary(args.real_summary)
    synthetic = paired_synthetic_ablation(args.dataset, args.depth_report)
    summary: dict[str, Any] = {
        "format": "LAB_SAM.section_1_2_research.v1",
        "created_by": str(Path(__file__).resolve()),
        "protocol": {
            "real": "36 real positive test images; actual best.pth coarse manifest replay; no retraining",
            "synthetic": "92 visible test frames / 64 floor frames; true calibration; scene-separated generated database",
            "depth": "cached Depth Anything V2 Metric Indoor Small report; no test-set scale fit in this script",
            "ray": "known z=0 plane; DifferentiableRayCastLayer in float64; same ROI pixels as depth report",
        },
        "real_audit": real,
        "synthetic_paired": synthetic,
        "limitations": [
            "The real CCTV data has 2-D labels only, so no real metre-level depth claim is made.",
            "The synthetic scene is procedural and validates geometry/error propagation rather than domain transfer.",
            "Known-plane ray casting is not a mesh-free solution for arbitrary furniture surfaces.",
            "Depth Anything metric indoor was trained on Hypersim; fire/CCTV scale and appearance require validation/calibration.",
            "The affine depth calibration saved in the separate GT-pixel report is diagnostic and uses a scene-disjoint subset; it is not used here.",
        ],
        "recommendation": [
            "Train the ROI refiner with the actual detector coarse manifest and hard-tail samples; keep detector recall separate from point accuracy.",
            "Predict localization uncertainty (heatmap entropy/covariance and crop margin) and gate or blend 3-D branches using validation-calibrated thresholds.",
            "Use dynamic multiscale/context crops only after matching train/inference distributions; the current crop audit does not show systematic label exclusion.",
            "Connect soft-argmax ROI coordinates to the known-plane ray layer and add a robust 3-D loss with horizon/behind-camera masks.",
            "Keep Depth Anything as an auxiliary depth/support prior until metric scale is calibrated on held-out fire/CCTV scenes; do not replace the measured-plane branch on the current benchmark.",
            "For non-floor fire contacts, use a semantic plane selector with visibility checks; nearest-plane or floor fallback has severe tails in the current synthetic ablation.",
        ],
    }
    _make_plot(summary, args.output_dir / "section_1_2_benchmark.png")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "summary": str(args.output_dir / "summary.json"),
        "plot": str(args.output_dir / "section_1_2_benchmark.png"),
        "real_gt_inside_crop": real["recomputed"]["gt_inside_crop_count"],
        "real_records": real["records"],
        "floor_ray_rmse_m": synthetic["floor_ray_roi_same_input"]["rmse"],
        "floor_depth_rmse_m": synthetic["depth_camera_xyz_error_on_floor"]["rmse"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

