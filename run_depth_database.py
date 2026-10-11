"""Run DepthBasedLocator on the existing synthetic fire database.

The benchmark uses the annotated ``p_fire_pixel`` to isolate depth error from
ROI-refiner error.  It reports the raw relative-depth result and an explicitly
diagnostic affine fit ``z_gt = alpha * depth_raw + beta`` learned on one group
of scenes and evaluated on disjoint scenes.  The affine fit is not a claim
that the foundation model is metric; it is a quick way to quantify the scale
ambiguity on the supplied synthetic data.

Example:
    D:\\LAB\\.venv\\Scripts\\python.exe run_depth_database.py --max-samples 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_localization import DepthBasedLocator


DEFAULT_ROOT = Path(__file__).resolve().parent / "working" / "synthetic_fire_3d_v3"
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def finite_record(record: dict[str, Any]) -> bool:
    return bool(
        record.get("fire_visible", 0)
        and record.get("p_fire_pixel") is not None
        and record.get("fire_xyz_world") is not None
        and record.get("camera") is not None
    )


def load_records(root: Path, split: str, max_samples: int) -> list[dict[str, Any]]:
    path = root / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [record for record in records if finite_record(record)]
    # Keep scenes contiguous so the calibration/evaluation split cannot share
    # adjacent frames from one camera trajectory.
    records.sort(key=lambda item: (str(item.get("scene_id", "")), int(item.get("frame_index", 0))))
    if max_samples > 0:
        records = records[:max_samples]
    return records


def camera_point_from_world(record: dict[str, Any]) -> np.ndarray:
    camera = record["camera"]
    R = np.asarray(camera["R_world_to_camera"], dtype=np.float64)
    C = np.asarray(camera["camera_position"], dtype=np.float64).reshape(3)
    world = np.asarray(record["fire_xyz_world"], dtype=np.float64).reshape(3)
    return R @ (world - C)


def camera_to_world(record: dict[str, Any], point_camera: np.ndarray) -> np.ndarray:
    camera = record["camera"]
    R = np.asarray(camera["R_world_to_camera"], dtype=np.float64)
    C = np.asarray(camera["camera_position"], dtype=np.float64).reshape(3)
    return C + R.T @ np.asarray(point_camera, dtype=np.float64).reshape(3)


def metric_summary(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {"count": 0, "rmse_m": math.nan, "median_m": math.nan, "p90_m": math.nan}
    return {
        "count": int(errors.size),
        "rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
    }


def fit_affine(records: list[dict[str, Any]]) -> tuple[float, float]:
    x = np.asarray([item["depth_raw"] for item in records], dtype=np.float64)
    y = np.asarray([item["gt_camera_z"] for item in records], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2 or np.ptp(x[finite]) < 1e-8:
        # The model can be locally constant on very small subsets.  In that
        # case preserve its observed mean depth rather than exploding the fit.
        return 0.0, float(np.mean(y[finite])) if finite.any() else 1.0
    alpha, beta = np.polyfit(x[finite], y[finite], deg=1)
    return float(alpha), float(beta)


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = load_records(root, args.split, args.max_samples)
    if len(records) < 2:
        raise RuntimeError("Need at least two visible-fire records for the benchmark")

    print(f"dataset={root}")
    print(f"split={args.split} visible_records={len(records)} device={device}")
    print(f"loading={args.model}")
    load_start = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model)
    locator_cache: dict[tuple[float, ...], DepthBasedLocator] = {}
    print(f"model_load_s={time.perf_counter() - load_start:.3f}")

    results: list[dict[str, Any]] = []
    infer_start = time.perf_counter()
    for index, record in enumerate(records, start=1):
        K = np.asarray(record["camera"]["K"], dtype=np.float32)
        key = tuple(float(value) for value in K.reshape(-1))
        locator = locator_cache.get(key)
        if locator is None:
            locator = DepthBasedLocator(
                model=model,
                K_matrix=torch.from_numpy(K),
                processor=processor,
                device=device,
                use_amp=(device.type == "cuda"),
                invalid_depth="nearest_valid",
            )
            locator_cache[key] = locator

        image_path = root / record["image_path"]
        image = Image.open(image_path).convert("RGB")
        point_2d = torch.tensor(record["p_fire_pixel"], dtype=torch.float32, device=device)
        sample_start = time.perf_counter()
        details = locator.locate_with_details(image, point_2d)
        if details is None:
            continue
        gt_camera = camera_point_from_world(record)
        raw_camera = details.point_3d.detach().cpu().numpy().astype(np.float64)
        results.append(
            {
                "sample_id": record["sample_id"],
                "scene_id": record.get("scene_id"),
                "depth_raw": float(details.depth.detach().cpu()),
                "pred_camera": raw_camera.tolist(),
                "gt_camera": gt_camera.tolist(),
                "gt_camera_z": float(gt_camera[2]),
                "gt_world": record["fire_xyz_world"],
                "infer_s": time.perf_counter() - sample_start,
            }
        )
        print(
            f"[{index:03d}/{len(records):03d}] {record['sample_id']} "
            f"raw_depth={float(details.depth):.4f} gt_z={gt_camera[2]:.4f} "
            f"infer_s={results[-1]['infer_s']:.3f}",
            flush=True,
        )

    if not results:
        raise RuntimeError("No successful DepthBasedLocator predictions")
    results.sort(key=lambda item: str(item["sample_id"]))

    # Split by scene, not by adjacent frames.  The fit is diagnostic only and
    # uses the ground-truth z scale present in this synthetic database.
    scene_ids = sorted({str(item["scene_id"]) for item in results})
    split_index = max(1, min(len(scene_ids) - 1, int(round(len(scene_ids) * args.calibration_fraction))))
    calibration_scenes = set(scene_ids[:split_index])
    calibration = [item for item in results if str(item["scene_id"]) in calibration_scenes]
    evaluation = [item for item in results if str(item["scene_id"]) not in calibration_scenes]
    alpha, beta = fit_affine(calibration)
    records_by_id = {str(record["sample_id"]): record for record in records}

    raw_world_errors: list[float] = []
    calibrated_world_errors: list[float] = []
    calibrated_z_errors: list[float] = []
    for item in evaluation:
        raw_camera = np.asarray(item["pred_camera"], dtype=np.float64)
        gt_world = np.asarray(item["gt_world"], dtype=np.float64)
        source_record = records_by_id[str(item["sample_id"])]
        raw_world = camera_to_world(source_record, raw_camera)
        # Replace only the camera-axis depth with the calibrated metric value.
        calibrated_camera = raw_camera * ((alpha * item["depth_raw"] + beta) / max(item["depth_raw"], 1e-8))
        calibrated_world = camera_to_world(source_record, calibrated_camera)
        raw_world_errors.append(float(np.linalg.norm(raw_world - gt_world)))
        calibrated_world_errors.append(float(np.linalg.norm(calibrated_world - gt_world)))
        calibrated_z_errors.append(float(alpha * item["depth_raw"] + beta - item["gt_camera_z"]))

    raw_depth = np.asarray([item["depth_raw"] for item in results], dtype=np.float64)
    gt_z = np.asarray([item["gt_camera_z"] for item in results], dtype=np.float64)
    corr = float(np.corrcoef(raw_depth, gt_z)[0, 1]) if np.ptp(raw_depth) > 1e-8 else math.nan
    report: dict[str, Any] = {
        "model": args.model,
        "dataset_root": str(root),
        "split": args.split,
        "device": str(device),
        "records_requested": len(records),
        "records_succeeded": len(results),
        "scenes": len(scene_ids),
        "inference_total_s": time.perf_counter() - infer_start,
        "inference_mean_s": float(np.mean([item["infer_s"] for item in results])),
        "raw_depth_gt_z_correlation": corr,
        "diagnostic_affine_calibration": {
            "formula": "gt_camera_z = alpha * depth_raw + beta",
            "alpha": alpha,
            "beta": beta,
            "calibration_scenes": sorted(calibration_scenes),
            "calibration_count": len(calibration),
            "evaluation_count": len(evaluation),
            "evaluation_camera_z_rmse_m": float(np.sqrt(np.mean(np.square(calibrated_z_errors))))
            if calibrated_z_errors
            else math.nan,
            "evaluation_world_error": metric_summary(np.asarray(calibrated_world_errors)),
        },
        "raw_world_error_interpreting_relative_depth_as_m": metric_summary(np.asarray(raw_world_errors)),
        "samples": results,
    }
    output = args.output or (root / f"depth_anything_v2_{args.split}_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: value for k, value in report.items() if k != "samples"}, indent=2))
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

