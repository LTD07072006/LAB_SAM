"""Train and compare ROIRefiner on real, synthetic and mixed domains.

The three experiments share the same ROI architecture and optimizer settings:

``real``
    Real ``p_fire`` labels. If no detector manifest is supplied, a reproducible
    pixel perturbation is used as the coarse point.
``synthetic``
    Synthetic ``p_fire_pixel`` is the clean target and
    ``p_fire_noisy_pixel`` is the coarse detector observation.
``mixed``
    A balanced union of real and synthetic training records. Validation is
    balanced across the two domains; the final report evaluates each domain
    separately, so domain transfer is visible instead of being hidden in one
    aggregate score.

This script trains only the point-refinement branch. Fire/no-fire
classification and detector recall remain upstream metrics.

Example (a quick CPU smoke run)::

    .venv\\Scripts\\python.exe train_roi_domain_experiments.py \\
      --experiments real synthetic mixed --epochs 2 --max-train-per-domain 32 \\
      --max-val-per-domain 16 --max-test-per-domain 16 --device cpu

Example (full run)::

    .venv\\Scripts\\python.exe train_roi_domain_experiments.py \\
      --experiments real synthetic mixed --epochs 30 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from project_paths import CCTV_DATASET
from narrow_localizer import ROIRefiner, load_backbone_from_checkpoint, roi_metrics, roi_loss, run_roi_epoch
from roi_dataset_loader import (
    DomainBundle,
    ROIRecord,
    ROIRefinerDataset,
    assert_disjoint_splits,
    limit_records,
    load_real_domain,
    load_synthetic_domain,
    records_manifest,
)


SPLITS = ("train", "val", "test")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def combine_records(*record_lists: Sequence[ROIRecord]) -> list[ROIRecord]:
    result: list[ROIRecord] = []
    seen: set[str] = set()
    for records in record_lists:
        for record in records:
            key = str(Path(record.image_path).resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(record)
    return result


def balance_domains(
    left: Sequence[ROIRecord],
    right: Sequence[ROIRecord],
    seed: int,
    maximum_per_domain: int = 0,
) -> list[ROIRecord]:
    """Return an equal-size domain union for mixed validation/training."""
    left_items = list(left)
    right_items = list(right)
    if maximum_per_domain > 0:
        left_items = limit_records(left_items, maximum_per_domain, seed + 11)
        right_items = limit_records(right_items, maximum_per_domain, seed + 17)
    count = min(len(left_items), len(right_items))
    if count <= 0:
        raise RuntimeError(f"Cannot balance empty domains: left={len(left_items)} right={len(right_items)}")
    left_items = limit_records(left_items, count, seed + 23)
    right_items = limit_records(right_items, count, seed + 29)
    merged = left_items + right_items
    random.Random(seed + 31).shuffle(merged)
    return merged


def prepare_domains(args: argparse.Namespace) -> tuple[DomainBundle, DomainBundle]:
    real = load_real_domain(
        args.real_labels,
        args.real_dataset,
        seed=args.seed,
        coarse_manifest=args.real_coarse_manifest,
        coarse_noise_px=args.real_coarse_noise_px,
    )
    synthetic = load_synthetic_domain(args.synthetic_dataset, seed=args.seed)

    for domain in (real, synthetic):
        domain.splits = {
            split: limit_records(domain.splits.get(split, []), getattr(args, f"max_{split}_per_domain"), args.seed + len(split))
            for split in SPLITS
        }
        assert_disjoint_splits(domain.splits, domain.name)

    if any(not real.splits[split] for split in SPLITS):
        raise RuntimeError(f"Real domain has an empty split: {real.counts()} stats={real.stats}")
    if any(not synthetic.splits[split] for split in SPLITS):
        raise RuntimeError(f"Synthetic domain has an empty split: {synthetic.counts()} stats={synthetic.stats}")
    return real, synthetic


def build_experiment_splits(
    name: str,
    real: DomainBundle,
    synthetic: DomainBundle,
    args: argparse.Namespace,
) -> dict[str, list[ROIRecord]]:
    if name == "real":
        return {split: list(real.splits[split]) for split in SPLITS}
    if name == "synthetic":
        return {split: list(synthetic.splits[split]) for split in SPLITS}
    if name == "mixed":
        return {
            "train": balance_domains(real.splits["train"], synthetic.splits["train"], args.seed + 101),
            "val": balance_domains(real.splits["val"], synthetic.splits["val"], args.seed + 103),
            # The mixed model is not selected on test. This split is retained
            # only for the optional combined summary; final metrics are run
            # separately on real and synthetic test records below.
            "test": balance_domains(real.splits["test"], synthetic.splits["test"], args.seed + 107),
        }
    raise ValueError(f"Unknown experiment: {name}")


def make_loader(
    records: Sequence[ROIRecord],
    train: bool,
    args: argparse.Namespace,
    seed_offset: int,
) -> DataLoader:
    dataset = ROIRefinerDataset(
        records,
        train=train,
        repeats=args.repeats if train else 1,
        roi_fraction=args.roi_fraction,
        coarse_jitter_px=args.coarse_jitter_px if train else 0.0,
        flip_p=args.flip_p if train else 0.0,
        seed=args.seed + seed_offset,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


def save_checkpoint(
    path: Path,
    model: ROIRefiner,
    experiment: str,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "architecture": "narrow_roi_localizer_v1_domain_experiment",
            "backbone": model.backbone_name,
            "model": model.state_dict(),
            "experiment": experiment,
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "roi_fraction": float(args.roi_fraction),
            "heatmap_size": int(model.heatmap_size),
            "temperature": float(model.temperature),
            "dropout": float(model.dropout_p),
            "weight_decay": float(args.weight_decay),
            "early_stopping": {
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
            },
            "coarse_definition": "clean p_fire target with detector/coarse point input",
        },
        path,
    )


@torch.no_grad()
def evaluate_loader(model: ROIRefiner, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    loss, metric = run_roi_epoch(model, loader, device)
    return {"loss": loss, "metric": metric, "samples": len(loader.dataset)}


def train_one(
    experiment: str,
    real: DomainBundle,
    synthetic: DomainBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(args.seed)
    splits = build_experiment_splits(experiment, real, synthetic, args)
    experiment_dir = args.output_dir / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        experiment_dir / "dataset_manifest.json",
        {
            "experiment": experiment,
            "counts": {split: len(records) for split, records in splits.items()},
            "source_domains": records_manifest((real, synthetic)),
            "used_records": {
                split: [record.to_dict() for record in records]
                for split, records in splits.items()
            },
        },
    )

    loaders = {
        "train": make_loader(splits["train"], True, args, 201),
        "val": make_loader(splits["val"], False, args, 203),
        "test": make_loader(splits["test"], False, args, 205),
        "real_test": make_loader(real.splits["test"], False, args, 207),
        "synthetic_test": make_loader(synthetic.splits["test"], False, args, 209),
    }

    model = ROIRefiner(
        backbone=args.backbone,
        pretrained=False,
        heatmap_size=args.heatmap_size,
        temperature=args.temperature,
        dropout=args.dropout,
    ).to(device)
    resume_checkpoint = args.resume_checkpoint if args.resume_checkpoint and experiment == args.resume_experiment else None
    resume_history = args.resume_history if args.resume_history and experiment == args.resume_experiment else None
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    if resume_checkpoint:
        resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        history_path = Path(resume_history).expanduser().resolve() if resume_history else experiment_dir / "history.json"
        if history_path.is_file():
            loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_history, list):
                raise ValueError(f"Resume history must be a JSON list: {history_path}")
            history = loaded_history
        if history:
            start_epoch = int(history[-1]["epoch"]) + 1
            best_row = min(history, key=lambda row: float(row["val_loss"]["total"]))
            best_val = float(best_row["val_loss"]["total"])
            best_epoch = int(best_row["epoch"])
        else:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val = float(checkpoint.get("val_loss", float("inf")))
            best_epoch = int(checkpoint.get("epoch", 0))
        print(
            f"resumed_experiment={experiment} checkpoint={resume_checkpoint} "
            f"start_epoch={start_epoch} best_epoch={best_epoch} best_val={best_val:.6f}",
            flush=True,
        )
    elif args.init_checkpoint:
        load_backbone_from_checkpoint(model, args.init_checkpoint)

    if args.freeze_epochs > 0 and start_epoch <= args.freeze_epochs:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    elif start_epoch > args.freeze_epochs:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=max(1, args.lr_patience),
        min_lr=args.min_lr,
    )
    best_path = experiment_dir / "best_roi.pth"
    started = time.perf_counter()
    stale_epochs = 0
    print(
        f"experiment={experiment} device={device} "
        f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
        f"real_test={len(loaders['real_test'].dataset)} "
        f"synthetic_test={len(loaders['synthetic_test'].dataset)}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
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
            f"experiment={experiment} epoch={epoch:03d} "
            f"train={train_loss['total']:.5f} val={val_loss['total']:.5f} "
            f"MAE={val_metric['mae_px']:.2f}px PCK10={val_metric['pck10']:.3f} "
            f"PCK25={val_metric['pck25']:.3f}",
            flush=True,
        )
        scheduler.step(float(val_loss["total"]))
        improved = float(val_loss["total"]) < best_val - args.min_delta
        if improved:
            best_val = float(val_loss["total"])
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(best_path, model, experiment, epoch, best_val, args)
        else:
            stale_epochs += 1
        save_checkpoint(experiment_dir / "last_roi.pth", model, experiment, epoch, float(val_loss["total"]), args)
        write_json(experiment_dir / "history.json", history)
        if args.patience > 0 and stale_epochs >= args.patience:
            print(
                f"early_stop experiment={experiment} epoch={epoch} "
                f"best_epoch={best_epoch} best_val={best_val:.6f} "
                f"stale_epochs={stale_epochs}",
                flush=True,
            )
            break

    if not best_path.is_file():
        raise FileNotFoundError(f"No best checkpoint available for {experiment}: {best_path}")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = {
        "validation": evaluate_loader(model, loaders["val"], device),
        "experiment_test": evaluate_loader(model, loaders["test"], device),
        "real_test": evaluate_loader(model, loaders["real_test"], device),
        "synthetic_test": evaluate_loader(model, loaders["synthetic_test"], device),
    }
    result = {
        "experiment": experiment,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "stopped_epoch": int(history[-1]["epoch"]) if history else 0,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "counts": {split: len(records) for split, records in splits.items()},
        "metrics": metrics,
        "checkpoint": str(best_path),
        "notes": [
            "MAE/PCK are 2D ROI point metrics, not 3D metric errors.",
            "The ROI branch is evaluated only on visible positive fire records.",
            "Classification recall and occluded-fire visibility are upstream metrics.",
        ],
    }
    write_json(experiment_dir / "test_metrics.json", result)
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-labels", type=Path, default=root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--real-dataset", type=Path, default=CCTV_DATASET)
    parser.add_argument("--synthetic-dataset", type=Path, default=root / "working" / "synthetic_fire_3d_v3")
    parser.add_argument("--real-coarse-manifest", type=Path, default=None)
    parser.add_argument("--real-coarse-noise-px", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "roi_domain_experiments")
    parser.add_argument("--experiments", nargs="+", choices=("real", "synthetic", "mixed"), default=["real", "synthetic", "mixed"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--roi-fraction", type=float, default=0.70)
    parser.add_argument("--coarse-jitter-px", type=float, default=0.5)
    parser.add_argument("--flip-p", type=float, default=0.20)
    parser.add_argument("--backbone", default="mobilenetv4_conv_medium")
    parser.add_argument("--heatmap-size", type=int, default=28)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-history", type=Path, default=None)
    parser.add_argument("--resume-experiment", choices=("real", "synthetic", "mixed"), default="mixed")
    parser.add_argument("--max-train-per-domain", type=int, default=0)
    parser.add_argument("--max-val-per-domain", type=int, default=0)
    parser.add_argument("--max-test-per-domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.real_labels = args.real_labels.expanduser().resolve()
    args.real_dataset = args.real_dataset.expanduser().resolve()
    args.synthetic_dataset = args.synthetic_dataset.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.init_checkpoint = args.init_checkpoint.expanduser().resolve() if args.init_checkpoint else None
    args.resume_checkpoint = args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint else None
    args.resume_history = args.resume_history.expanduser().resolve() if args.resume_history else None
    if min(args.epochs, args.batch_size, args.repeats) <= 0:
        raise ValueError("epochs, batch-size and repeats must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.patience < 0 or args.lr_patience < 0:
        raise ValueError("patience values must be >= 0")
    if args.min_delta < 0.0 or args.min_lr < 0.0:
        raise ValueError("min-delta and min-lr must be >= 0")
    if args.max_train_per_domain < 0 or args.max_val_per_domain < 0 or args.max_test_per_domain < 0:
        raise ValueError("max-* limits must be >= 0")
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real, synthetic = prepare_domains(args)
    write_json(args.output_dir / "source_domains.json", records_manifest((real, synthetic)))
    print(f"real_counts={real.counts()} synthetic_counts={synthetic.counts()}", flush=True)
    results: dict[str, Any] = {}
    for experiment in args.experiments:
        results[experiment] = train_one(experiment, real, synthetic, args, device)
    write_json(
        args.output_dir / "comparison_summary.json",
        {
            "experiments": list(args.experiments),
            "real_counts": real.counts(),
            "synthetic_counts": synthetic.counts(),
            "results": results,
        },
    )
    print(json.dumps(results, indent=2, ensure_ascii=False, default=json_default))
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
