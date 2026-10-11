import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_localization import DepthBasedLocator

root = Path(__file__).resolve().parent / "working" / "synthetic_fire_3d_v3"
record = json.loads(root.joinpath("test.jsonl").read_text(encoding="utf-8").splitlines()[0])
print("sample", record["sample_id"], record["image_path"], record["p_fire_pixel"], record["fire_xyz_world"])
name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
t0 = time.perf_counter()
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModelForDepthEstimation.from_pretrained(name)
print("load_s", time.perf_counter() - t0)
K = torch.tensor(record["camera"]["K"], dtype=torch.float32)
locator = DepthBasedLocator(model, K, processor=processor, use_amp=False, invalid_depth="nearest_valid")
image_path = root / record["image_path"]
image = Image.open(image_path).convert("RGB")
print("image", image.size)
t0 = time.perf_counter()
details = locator.locate_with_details(image, torch.tensor(record["p_fire_pixel"], dtype=torch.float32))
print("infer_s", time.perf_counter() - t0)
assert details is not None
print("depth", float(details.depth), "map", tuple(details.depth_map.shape))
print("cam_xyz", details.point_3d.tolist())
R = torch.tensor(record["camera"]["R_world_to_camera"], dtype=torch.float64)
camera_position = torch.tensor(record["camera"]["camera_position"], dtype=torch.float64)
point_world = camera_position + R.T @ details.point_3d.double()
print("world_xyz", point_world.tolist())
print("gt", record["fire_xyz_world"])
