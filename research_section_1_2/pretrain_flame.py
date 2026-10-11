"""Pretrain the spatial fire detector's confidence/backbone on FLAME.

FLAME has image-level Fire/No_Fire labels only.  The coordinate and heatmap
heads are deliberately excluded from the loss; CCTV `p_fire` remains the only
source of point supervision in the later fine-tuning stage.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from train_week6 import MEAN, STD, SpatialFireModel, seed_everything


class FlameDataset(Dataset):
    def __init__(self, root: Path, max_samples: int = 0, train: bool = True):
        paths = []
        for cls, label in (("Fire", 1.0), ("No_Fire", 0.0)):
            paths.extend((path, label) for path in sorted((root / "Training" / cls).glob("*"))
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        rng = random.Random(42)
        rng.shuffle(paths)
        if max_samples > 0:
            paths = paths[:max_samples]
        self.items = paths
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.train:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
        return tensor, torch.tensor(label, dtype=torch.float32)


def matching_backbone_init(model: nn.Module, checkpoint: Path) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state) if isinstance(state, dict) else state
    current = model.state_dict()
    copied = 0
    for raw_key, value in state.items():
        key = str(raw_key)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            current[key] = value
            copied += 1
    model.load_state_dict(current, strict=True)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flame-root", type=Path, default=ROOT / "datasets" / "FLAME")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "dataset_integration_20261010" / "flame_pretrain")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seed_everything(42)
    device = torch.device(args.device)
    dataset = FlameDataset(args.flame_root, max_samples=args.max_samples, train=True)
    if not dataset:
        raise RuntimeError(f"No FLAME images found under {args.flame_root}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SpatialFireModel(pretrained=False).to(device)
    copied = matching_backbone_init(model, args.init_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = seen = 0
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs["confidence_logit"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = len(labels)
            loss_sum += float(loss.detach()) * n
            correct += int(((torch.sigmoid(outputs["confidence_logit"]) >= 0.5) == (labels >= 0.5)).sum())
            seen += n
        row = {"epoch": epoch, "loss": loss_sum / max(1, seen), "accuracy": correct / max(1, seen)}
        history.append(row)
        print(row, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "spatial_heatmap_v2", "backbone": model.backbone_name,
        "model": model.state_dict(), "epoch": args.epochs,
        "pretraining": "FLAME image-level confidence only",
        "flame_samples": len(dataset), "backbone_tensors_transferred": copied,
        "history": history, "image_size": (224, 224),
    }
    torch.save(checkpoint, args.output_dir / "best_flame_pretrained.pth")
    (args.output_dir / "summary.json").write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"samples": len(dataset), "copied": copied, "checkpoint": str(args.output_dir / "best_flame_pretrained.pth")}, indent=2))


if __name__ == "__main__":
    main()
