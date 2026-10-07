"""Train a small, isolated ROI pilot on weak YOLO bottom-center labels.

This is deliberately a pilot branch.  It does not replace the calibrated
``best.pth -> ROI -> ray casting`` workflow and it never overwrites the
existing ROI checkpoint.  YOLO supplies bounding boxes, so the bottom centre
of one verified fire box is used as a weak proxy for the fire-ground point.

The default run reads at most 1,200 label files per split and selects:

    400 train + 50 validation + 50 test = 500 images

The original YOLO class ``1`` is treated as fire.  Change ``--fire-class``
only after checking the dataset's class convention.

Example:

    .venv\\Scripts\\python.exe train_home_fire_roi_pilot.py \
        --epochs 50 --device cuda

The output is written to ``output/home_fire_roi_pilot_500_ep50`` by default.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from project_paths import D_FIRE_ROOT
from narrow_localizer import (
    NarrowROIDataset,
    ROIRefiner,
    load_backbone_from_checkpoint,
    roi_metrics,
    roi_loss,
    run_roi_epoch,
)
from train_week6 import Record, seed_everything


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _read_fire_boxes(label_path: Path, fire_class: int) -> list[tuple[float, float]]:
    """Read normalized bottom-center points for one YOLO label file."""
    points: list[tuple[float, float]] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return points

    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            class_id = int(float(fields[0]))
            x_center, y_center, width, height = map(float, fields[1:5])
        except ValueError:
            continue
        values = np.asarray([x_center, y_center, width, height], dtype=np.float64)
        if class_id != fire_class or not np.all(np.isfinite(values)):
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        # YOLO stores the box centre and size in normalized coordinates.
        # ROI receives a contact-point proxy at the centre of the lower edge.
        point = (float(np.clip(x_center, 0.0, 1.0)),
                 float(np.clip(y_center + height / 2.0, 0.0, 1.0)))
        points.append(point)
    return points


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _candidate_records(
    dataset_root: Path,
    split: str,
    fire_class: int,
    scan_limit: int,
) -> list[Record]:
    """Inspect only the first ``scan_limit`` label files in one split."""
    label_dir = dataset_root / split / "labels"
    image_dir = dataset_root / split / "images"
    if not label_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(f"Missing YOLO split folders: {label_dir} / {image_dir}")

    label_paths = sorted(path for path in label_dir.iterdir() if path.is_file())
    if scan_limit > 0:
        label_paths = label_paths[:scan_limit]

    records: list[Record] = []
    for label_path in label_paths:
        # A single fire box makes the weak point unambiguous.  Images with
        # multiple fire boxes are left for a later multi-target branch.
        points = _read_fire_boxes(label_path, fire_class)
        if len(points) != 1:
            continue
        image_path = _find_image(image_dir, label_path.stem)
        if image_path is None:
            continue
        x_norm, y_norm = points[0]
        records.append(
            Record(
                image_path=str(image_path),
                has_fire=1,
                x_norm=x_norm,
                y_norm=y_norm,
                source_split=split,
                group=split,
            )
        )
    return records


def _select(records: Iterable[Record], count: int, seed: int) -> list[Record]:
    records = list(records)
    if len(records) < count:
        raise RuntimeError(f"Only {len(records)} usable single-fire images found; need {count}.")
    rng = random.Random(seed)
    return rng.sample(records, count)


def build_pilot_records(args: argparse.Namespace) -> dict[str, list[Record]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    all_records: dict[str, list[Record]] = {}
    for offset, split in enumerate(("train", "val", "test")):
        all_records[split] = _candidate_records(
            dataset_root,
            split,
            args.fire_class,
            args.scan_limit,
        )
        print(
            f"candidate_split={split} candidates={len(all_records[split])} "
            f"scan_limit={args.scan_limit}"
        )

    selected = {
        "train": _select(all_records["train"], args.train_count, args.seed + 1),
        "val": _select(all_records["val"], args.val_count, args.seed + 2),
        "test": _select(all_records["test"], args.test_count, args.seed + 3),
    }
    total = sum(len(items) for items in selected.values())
    print(
        f"pilot_images={total} train={len(selected['train'])} "
        f"val={len(selected['val'])} test={len(selected['test'])}"
    )
    return selected


def _record_to_json(record: Record) -> dict[str, object]:
    return {
        "image_path": record.image_path,
        "split": record.source_split,
        "has_fire": 1,
        "p_fire": [record.x_norm, record.y_norm],
        "point_source": "yolo_bbox_bottom_center_weak",
        "label_quality": "weak",
        "weight": 0.15,
    }


def _save_model(model: ROIRefiner, output: Path, epoch: int, val_loss: float, args: argparse.Namespace) -> None:
    torch.save(
        {
            "architecture": "narrow_roi_localizer_v1_yolo_weak_pilot",
            "backbone": model.backbone_name,
            "model": model.state_dict(),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "roi_fraction": float(args.roi_fraction),
            "heatmap_size": int(model.heatmap_size),
            "temperature": float(model.temperature),
            "source": "YOLO bbox bottom-center weak labels",
            "fire_class": int(args.fire_class),
            "weak_label_weight_note": "Pilot-only; current ROI loss uses these records as targets.",
        },
        output,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = build_pilot_records(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_root": str(args.dataset_root.resolve()),
        "fire_class": args.fire_class,
        "scan_limit_per_split": args.scan_limit,
        "counts": {key: len(value) for key, value in selected.items()},
        "records": [_record_to_json(record) for items in selected.values() for record in items],
        "note": "Weak YOLO bbox bottom-center labels; not ground-truth contact points.",
    }
    (args.output_dir / "pilot_manifest_500.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    loaders = {
        "train": DataLoader(
            NarrowROIDataset(
                selected["train"],
                train=True,
                repeats=args.repeats,
                roi_fraction=args.roi_fraction,
                noise_std=args.noise_std,
                seed=args.seed,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
        "val": DataLoader(
            NarrowROIDataset(
                selected["val"],
                train=False,
                repeats=1,
                roi_fraction=args.roi_fraction,
                noise_std=args.noise_std,
                seed=args.seed + 1,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            NarrowROIDataset(
                selected["test"],
                train=False,
                repeats=1,
                roi_fraction=args.roi_fraction,
                noise_std=args.noise_std,
                seed=args.seed + 2,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        ),
    }

    model = ROIRefiner(pretrained=False).to(device)
    if args.init_checkpoint:
        load_backbone_from_checkpoint(model, args.init_checkpoint.expanduser().resolve())
    if args.freeze_epochs > 0:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    print(f"device={device} epochs={args.epochs} output={args.output_dir.resolve()}")

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
        train_loss, train_metric = run_roi_epoch(model, loaders["train"], device, optimizer)
        val_loss, val_metric = run_roi_epoch(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metric": train_metric,
            "val_metric": val_metric,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={train_loss['total']:.5f} "
            f"val={val_loss['total']:.5f} MAE={val_metric['mae_px']:.2f}px "
            f"PCK10={val_metric['pck10']:.3f} PCK25={val_metric['pck25']:.3f}",
            flush=True,
        )
        if val_loss["total"] < best_val:
            best_val = float(val_loss["total"])
            best_epoch = epoch
            _save_model(model, args.output_dir / "best_roi_500_yolo.pth", epoch, best_val, args)
        _save_model(model, args.output_dir / "last_roi_500_yolo.pth", epoch, float(val_loss["total"]), args)
        (args.output_dir / "history_roi_500_yolo.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    best_checkpoint = torch.load(
        args.output_dir / "best_roi_500_yolo.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"], strict=True)
    test_loss, test_metric = run_roi_epoch(model, loaders["test"], device)
    elapsed = time.perf_counter() - started
    result = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_metric": test_metric,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "counts": {key: len(value) for key, value in selected.items()},
        "warning": "This test is YOLO weak-label evaluation, not calibrated CCTV contact-point evaluation.",
    }
    (args.output_dir / "test_metrics_roi_500_yolo.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"best_epoch={best_epoch} test_loss={test_loss} test_metric={test_metric}")
    print(f"elapsed_seconds={elapsed:.1f} saved={args.output_dir.resolve()}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=D_FIRE_ROOT)
    parser.add_argument("--fire-class", type=int, default=1)
    parser.add_argument("--train-count", type=int, default=400)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--scan-limit", type=int, default=1200)
    parser.add_argument("--init-checkpoint", type=Path, default=root / "fire-model-data" / "best.pth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "home_fire_roi_pilot_500_ep50",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--roi-fraction", type=float, default=0.70)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if min(args.train_count, args.val_count, args.test_count) <= 0:
        raise ValueError("train/val/test counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
