"""Compute floor-plane ray errors from the saved ROI benchmark report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from depth_localization import DifferentiableRayCastLayer


root = Path(r"D:\LAB\SAM_Experiment\working\synthetic_fire_3d_v3")
report = json.loads((root / "depth_anything_v2_metric_indoor_roi_test92.json").read_text(encoding="utf-8"))
records = {
    item["sample_id"]: item
    for item in (
        json.loads(line)
        for line in (root / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
}
layer = DifferentiableRayCastLayer()
roi_errors = []
gt_errors = []
for row in report["samples"]:
    record = records[row["sample_id"]]
    if record.get("fire_surface") != "floor":
        continue
    camera = record["camera"]
    K = torch.tensor(camera["K"], dtype=torch.float64)
    R = torch.tensor(camera["R_world_to_camera"], dtype=torch.float64)
    C = torch.tensor(camera["camera_position"], dtype=torch.float64)
    T = -(R @ C)
    plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    target = torch.tensor(record["fire_xyz_world"], dtype=torch.float64)
    roi_pixel = torch.tensor(row["roi_pixel"], dtype=torch.float64)
    gt_pixel = torch.tensor(record["p_fire_pixel"], dtype=torch.float64)
    roi_errors.append(float(torch.linalg.vector_norm(layer(roi_pixel, K, R, T, plane) - target)))
    gt_errors.append(float(torch.linalg.vector_norm(layer(gt_pixel, K, R, T, plane) - target)))

for name, values in (("roi_pixel", roi_errors), ("gt_pixel", gt_errors)):
    values_np = np.asarray(values, dtype=np.float64)
    print(
        name,
        "count", len(values),
        "rmse_m", float(np.sqrt(np.mean(values_np * values_np))),
        "median_m", float(np.median(values_np)),
        "p90_m", float(np.percentile(values_np, 90)),
    )
