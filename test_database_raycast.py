"""Validate DifferentiableRayCastLayer against the synthetic floor labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_localization import DifferentiableRayCastLayer


root = Path(__file__).resolve().parent / "working" / "synthetic_fire_3d_v3"
records = [
    json.loads(line)
    for line in (root / "test.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
records = [
    item
    for item in records
    if item.get("fire_visible") and item.get("fire_surface") == "floor"
]
layer = DifferentiableRayCastLayer()
errors = []
gradient_norms = []
for record in records:
    camera = record["camera"]
    K = torch.tensor(camera["K"], dtype=torch.float64)
    R = torch.tensor(camera["R_world_to_camera"], dtype=torch.float64)
    C = torch.tensor(camera["camera_position"], dtype=torch.float64)
    T = -(R @ C)
    pixel = torch.tensor(record["p_fire_pixel"], dtype=torch.float64, requires_grad=True)
    plane = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    predicted = layer(pixel, K, R, T, plane)
    target = torch.tensor(record["fire_xyz_world"], dtype=torch.float64)
    errors.append(float(torch.linalg.vector_norm(predicted - target).detach()))
    predicted.square().sum().backward()
    gradient_norms.append(float(pixel.grad.abs().sum()))

print("floor_records", len(records))
print("ray_rmse_m", float(np.sqrt(np.mean(np.square(errors)))))
print("ray_max_m", float(np.max(errors)))
print("gradient_min_l1", float(np.min(gradient_norms)))
print("gradient_all_finite", bool(np.isfinite(gradient_norms).all()))
