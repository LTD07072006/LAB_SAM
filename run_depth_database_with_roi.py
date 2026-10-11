"""Evaluate DepthBasedLocator using the existing ROI Refiner checkpoint.

The first benchmark uses ``p_fire_pixel`` to isolate depth quality.  This
runner adds the saved ``week6_roi_result/best_roi.pth`` and reports the change
when the locator samples the ROI prediction instead.  It deliberately keeps
the depth model fixed and reuses one model forward per RGB frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from depth_localization import DepthBasedLocator
from narrow_localizer import ROIRefinerInference


DATASET = PROJECT / "working" / "synthetic_fire_3d_v3"
ROI_CHECKPOINT = PROJECT / "week6_roi_result" / "best_roi.pth"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--roi-checkpoint", type=Path, default=ROI_CHECKPOINT)
    parser.add_argument("--depth-model", default=MODEL_NAME)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def record_is_valid(record: dict[str, Any]) -> bool:
    return bool(record.get("fire_visible") and record.get("p_fire_pixel") and record.get("fire_xyz_world"))


def camera_xyz(record: dict[str, Any]) -> np.ndarray:
    R = np.asarray(record["camera"]["R_world_to_camera"], dtype=np.float64)
    C = np.asarray(record["camera"]["camera_position"], dtype=np.float64)
    W = np.asarray(record["fire_xyz_world"], dtype=np.float64)
    return R @ (W - C)


def main() -> int:
    cli = args()
    root = cli.dataset_root.resolve()
    records = [
        json.loads(line)
        for line in (root / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record_is_valid(record)]
    records.sort(key=lambda item: item["sample_id"])
    if cli.max_samples > 0:
        records = records[: cli.max_samples]
    if not records:
        raise RuntimeError("No visible-fire records in test.jsonl")

    device = torch.device(cli.device)
    print(f"dataset={root}")
    print(f"records={len(records)} device={device}")
    print(f"roi_checkpoint={cli.roi_checkpoint}")
    roi = ROIRefinerInference(cli.roi_checkpoint, device=str(device))
    processor = AutoImageProcessor.from_pretrained(cli.depth_model)
    depth_model = AutoModelForDepthEstimation.from_pretrained(cli.depth_model)
    locators: dict[tuple[float, ...], DepthBasedLocator] = {}
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for i, record in enumerate(records, 1):
        image_path = root / record["image_path"]
        image = Image.open(image_path).convert("RGB")
        # A coarse point is intentionally taken from the supplied detector-like
        # noisy observation; this exercises the actual ROI refiner path rather
        # than centering a crop exactly on the ground-truth pixel.
        coarse = record.get("p_fire_noisy_pixel") or record["p_fire_pixel"]
        refined = roi.refine(image, coarse)
        K_np = np.asarray(record["camera"]["K"], dtype=np.float32)
        key = tuple(float(v) for v in K_np.reshape(-1))
        locator = locators.get(key)
        if locator is None:
            locator = DepthBasedLocator(
                depth_model,
                torch.from_numpy(K_np),
                processor=processor,
                device=device,
                use_amp=device.type == "cuda",
                invalid_depth="nearest_valid",
            )
            locators[key] = locator
        truth_pixel = np.asarray(record["p_fire_pixel"], dtype=np.float64)
        refined_pixel = np.asarray(refined.point, dtype=np.float64)
        pixel_error = float(np.linalg.norm(refined_pixel - truth_pixel))
        details = locator.locate_with_details(image, torch.from_numpy(refined_pixel.astype(np.float32)).to(device))
        if details is None:
            continue
        gt_camera = camera_xyz(record)
        pred = details.point_3d.detach().cpu().numpy().astype(np.float64)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "scene_id": record.get("scene_id"),
                "roi_pixel": refined_pixel.tolist(),
                "gt_pixel": truth_pixel.tolist(),
                "roi_pixel_error_px": pixel_error,
                "roi_confidence": float(refined.confidence),
                "depth_m": float(details.depth.detach().cpu()),
                "pred_camera_xyz": pred.tolist(),
                "gt_camera_xyz": gt_camera.tolist(),
                "camera_error_m": float(np.linalg.norm(pred - gt_camera)),
            }
        )
        print(
            f"[{i:03d}/{len(records):03d}] {record['sample_id']} "
            f"roi_px_err={pixel_error:.2f} depth={float(details.depth):.3f} "
            f"xyz_err={rows[-1]['camera_error_m']:.3f}",
            flush=True,
        )

    if not rows:
        raise RuntimeError("No successful ROI + depth predictions")
    px = np.asarray([row["roi_pixel_error_px"] for row in rows], dtype=np.float64)
    xyz = np.asarray([row["camera_error_m"] for row in rows], dtype=np.float64)
    report = {
        "dataset": str(root),
        "depth_model": cli.depth_model,
        "roi_checkpoint": str(cli.roi_checkpoint),
        "device": str(device),
        "records": len(rows),
        "elapsed_s": time.perf_counter() - start,
        "roi_pixel": {
            "mae_px": float(px.mean()),
            "median_px": float(np.median(px)),
            "p90_px": float(np.percentile(px, 90)),
            "pck10": float(np.mean(px <= 10.0)),
            "pck25": float(np.mean(px <= 25.0)),
        },
        "depth_camera_error": {
            "rmse_m": float(np.sqrt(np.mean(xyz * xyz))),
            "median_m": float(np.median(xyz)),
            "p90_m": float(np.percentile(xyz, 90)),
        },
        "warning": "DepthAnything prediction is evaluated at ROI-refiner pixels; no affine metric recalibration is applied.",
        "samples": rows,
    }
    output = cli.output or root / "depth_anything_v2_metric_indoor_roi_test.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

