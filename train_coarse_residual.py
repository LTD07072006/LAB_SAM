"""Train a small image-conditioned residual correction head for coarse points.

The legacy detector remains frozen.  The correction head sees the detector's
actual coarse point and the frozen image embedding, then predicts a bounded
pixel offset.  Validation/test labels are used only for evaluation; the
training manifest must contain train rows only for weighting.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from fire_detector import FireGrounder
from train_week6 import load_records, split_records

ROOT = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def aliases(value: str) -> list[str]:
    s = str(value).replace("\\", "/").lower()
    out = [s]
    marker = "/img_data/"
    if marker in s:
        out.append(s[s.index(marker)+1:])
    if "/" in s:
        out.append(s.rsplit("/", 1)[-1])
    return list(dict.fromkeys(out))


def index_manifest(payload: Any) -> dict[str, dict]:
    rows = payload.get("records", []) if isinstance(payload, dict) else payload
    result: dict[str, dict] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not row.get("image_path"):
            continue
        for key in aliases(row["image_path"]):
            result.setdefault(key, row)
    return result


class ResidualDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, weights: dict[str, float]):
        self.rows, self.train, self.weights = rows, train, weights
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        image = Image.open(r["image_path"]).convert("RGB")
        x = self.tf(image)
        w, h = image.size
        coarse = np.asarray(r["raw_pixel"], dtype=np.float32)
        gt = np.asarray(r["ground_truth_pixel"], dtype=np.float32)
        # normalized input and normalized residual; this is resolution-safe.
        inp = torch.tensor([coarse[0] / max(w, 1), coarse[1] / max(h, 1)], dtype=torch.float32)
        target = torch.tensor([(gt[0]-coarse[0]) / max(w, 1), (gt[1]-coarse[1]) / max(h, 1)], dtype=torch.float32)
        return x, inp, target, torch.tensor([float(w), float(h)]), r["image_path"], torch.tensor(self.weights.get(aliases(r["image_path"])[0], 1.0))


class ResidualHead(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim + 2, 128), nn.SiLU(), nn.Dropout(0.10), nn.Linear(128, 2))

    def forward(self, feature: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        # A tanh bound prevents a learned correction from making catastrophic jumps.
        return 0.20 * torch.tanh(self.net(torch.cat([feature, coarse], dim=1)))


@torch.no_grad()
def evaluate(backbone: nn.Module, head: ResidualHead, loader: DataLoader, device: torch.device) -> dict[str, float]:
    backbone.eval(); head.eval(); base, corrected = [], []
    for images, coarse, target, sizes, *_ in loader:
        images, coarse, target, sizes = images.to(device), coarse.to(device), target.to(device), sizes.to(device)
        feat = backbone(images)
        delta = head(feat, coarse)
        base.append(torch.linalg.vector_norm(target * sizes, dim=1).cpu())
        corrected.append(torch.linalg.vector_norm((target-delta) * sizes, dim=1).cpu())
    b, c = torch.cat(base), torch.cat(corrected)
    return {"base_mae_px": float(b.mean()), "corrected_mae_px": float(c.mean()), "base_pck10": float((b<=10).float().mean()), "corrected_pck10": float((c<=10).float().mean()), "base_pck25": float((b<=25).float().mean()), "corrected_pck25": float((c<=25).float().mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "output/section_1_2_pre_roi_20261010/coarse_issue_manifest/coarse_issue_manifest.json")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "fire-model-data/best.pth")
    ap.add_argument("--out", type=Path, default=ROOT / "output/dataset_integration_20261010/coarse_residual_head")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--device", default="cpu")
    args = ap.parse_args(); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    payload = json.loads(args.manifest.read_text(encoding="utf-8")); idx = index_manifest(payload)
    records, _ = load_records(ROOT / "fire-model-data/dataset_labels (1).json", ROOT / "datasets/fire-detection-from-cctv/data/data")
    splits = split_records(records, 42)
    rows_by_split: dict[str, list[dict]] = {}
    for split, recs in splits.items():
        rows = []
        for rec in recs:
            if not rec.has_fire: continue
            item = next((idx[k] for k in aliases(rec.image_path) if k in idx), None)
            if item and item.get("raw_pixel") and item.get("ground_truth_pixel"):
                rows.append({**item, "image_path": rec.image_path})
        rows_by_split[split] = rows
    # Weight only train hard-tail rows.  No val/test row can enter this map.
    weights = {}
    for row in rows_by_split["train"]:
        e = float(row.get("raw_error_px", 0.0)); weights[aliases(row["image_path"])[0]] = 1.0 + (1.0 if e > 10 else 0.5 if e > 7 else 0.0)
    device = torch.device(args.device)
    detector = FireGrounder(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False); detector.load_state_dict(ckpt["model"], strict=True); detector.eval()
    for p in detector.backbone.parameters(): p.requires_grad = False
    with torch.no_grad(): feature_dim = int(detector.backbone(torch.zeros(1,3,224,224,device=device)).shape[1])
    head = ResidualHead(feature_dim).to(device); opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loaders = {s: DataLoader(ResidualDataset(rows_by_split[s], s=="train", weights), args.batch_size, shuffle=s=="train", num_workers=0) for s in ("train","val","test")}
    args.out.mkdir(parents=True, exist_ok=True); history=[]; best=float("inf")
    for epoch in range(1, args.epochs+1):
        head.train(); losses=[]
        for images, coarse, target, _, _, weight in loaders["train"]:
            images, coarse, target, weight = images.to(device), coarse.to(device), target.to(device), weight.to(device)
            with torch.no_grad(): feat=detector.backbone(images)
            delta=head(feat,coarse); per=F.smooth_l1_loss(delta,target,reduction="none").mean(1); loss=(per*weight).sum()/weight.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 5.0); opt.step(); losses.append(float(loss.detach()))
        val=evaluate(detector.backbone,head,loaders["val"],device); history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val":val}); print(epoch,history[-1],flush=True)
        if val["corrected_mae_px"] < best: best=val["corrected_mae_px"]; torch.save({"model":head.state_dict(),"feature_dim":feature_dim,"epoch":epoch,"val":val},args.out/"best_residual_head.pth")
    best_ck=torch.load(args.out/"best_residual_head.pth",map_location=device,weights_only=False); head.load_state_dict(best_ck["model"]); report={"best_epoch":best_ck["epoch"],"val":best_ck["val"],"test":evaluate(detector.backbone,head,loaders["test"],device),"counts":{s:len(v) for s,v in rows_by_split.items()}}
    (args.out/"history.json").write_text(json.dumps(history,indent=2)); (args.out/"metrics.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
