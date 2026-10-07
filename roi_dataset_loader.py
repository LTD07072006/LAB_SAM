"""Unified real/synthetic dataset loader for the ROI fire-point refiner.

The loader exposes one contract to the training script:

``coarse_xy_norm`` -> simulated detector/ROI input
``gt_xy_norm``     -> clean fire-contact target

Real labels contain ``p_fire`` but usually do not contain a detector output,
so a deterministic pixel perturbation is generated for the real-only and
mixed experiments. Synthetic records contain both ``p_fire_pixel`` and
``p_fire_noisy_pixel``; the latter is used directly and records with a missed
synthetic detection are excluded from ROI regression.

The module intentionally keeps fire classification outside the ROI task. Only
visible positive fire records enter the ROI refiner. Upstream detector recall
and occluded events must be reported by a separate detector/visibility metric.
"""

from __future__ import annotations

import json
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from narrow_localizer import _crop_square, _prepare_image
from train_week6 import Record, load_records, split_records


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ROIRecord:
    """Portable record consumed by :class:`ROIRefinerDataset`."""

    image_path: str
    gt_xy_norm: tuple[float, float]
    coarse_xy_norm: tuple[float, float]
    split: str
    source: str
    group: str
    sample_id: str
    point_source: str
    image_width: int
    image_height: int

    @property
    def has_fire(self) -> int:
        return 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomainBundle:
    name: str
    splits: dict[str, list[ROIRecord]]
    stats: dict[str, Any]

    def counts(self) -> dict[str, int]:
        return {split: len(self.splits.get(split, [])) for split in SPLITS}


def _finite_norm(value: Any) -> Optional[tuple[float, float]]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(array) < 2 or not np.all(np.isfinite(array[:2])):
        return None
    if np.any(array[:2] < 0.0) or np.any(array[:2] > 1.0):
        return None
    return float(array[0]), float(array[1])


def _pixel_to_norm(value: Any, width: int, height: int) -> Optional[tuple[float, float]]:
    try:
        pixel = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(pixel) < 2 or not np.all(np.isfinite(pixel[:2])):
        return None
    return _finite_norm((pixel[0] / max(width, 1), pixel[1] / max(height, 1)))


def _image_size(path: Path, fallback: Optional[Sequence[int]] = None) -> tuple[int, int]:
    if fallback is not None and len(fallback) >= 2:
        width, height = int(fallback[0]), int(fallback[1])
        if width > 0 and height > 0:
            return width, height
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "little") ^ int(seed)


def _simulated_coarse(
    gt_xy_norm: tuple[float, float],
    width: int,
    height: int,
    noise_px: float,
    seed: int,
) -> tuple[float, float]:
    if noise_px <= 0.0:
        return gt_xy_norm
    rng = np.random.default_rng(seed)
    delta = rng.normal(
        loc=0.0,
        scale=[float(noise_px) / max(width, 1), float(noise_px) / max(height, 1)],
        size=2,
    )
    return tuple(float(value) for value in np.clip(np.asarray(gt_xy_norm) + delta, 0.01, 0.99))


def _normalise_manifest_point(item: Any, width: int, height: int) -> Optional[tuple[float, float]]:
    """Read either ``point`` or a direct normalized/pixel coarse value."""
    if isinstance(item, dict):
        point = item.get("point", item.get("p_fire_noisy_pixel", item.get("p_fire_noisy")))
        size = item.get("size", item.get("image_size", [width, height]))
        if point is None:
            return None
        try:
            values = np.asarray(point, dtype=np.float64).reshape(-1)
            size_values = np.asarray(size, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if len(values) < 2:
            return None
        if len(size_values) >= 2 and np.max(np.abs(values[:2])) > 1.5:
            values = values[:2] / np.maximum(size_values[:2], 1.0)
        return _finite_norm(values[:2])
    if item is None:
        return None
    try:
        values = np.asarray(item, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(values) < 2:
        return None
    if np.max(np.abs(values[:2])) > 1.5:
        values = values[:2] / np.maximum(np.asarray([width, height], dtype=np.float64), 1.0)
    return _finite_norm(values[:2])


def _manifest_aliases(path_text: str) -> list[str]:
    normalised = str(path_text).replace("\\", "/").lower()
    aliases = [normalised]
    marker = "/img_data/"
    if marker in normalised:
        suffix = normalised.split(marker, 1)[1]
        aliases.extend([suffix, "img_data/" + suffix])
    parts = normalised.split("/")
    for split in SPLITS:
        if split in parts:
            index = parts.index(split)
            suffix = "/".join(parts[index:])
            aliases.append(suffix)
            break
    aliases.append(parts[-1])
    return list(dict.fromkeys(aliases))


def load_coarse_manifest(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Coarse manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Coarse manifest must be a JSON object: {path}")
    indexed: dict[str, Any] = {}
    for raw_key, value in data.items():
        for alias in _manifest_aliases(str(raw_key)):
            indexed.setdefault(alias, value)
    return indexed


def _lookup_manifest(indexed: dict[str, Any], image_path: str) -> Any:
    for alias in _manifest_aliases(image_path):
        if alias in indexed:
            return indexed[alias]
    return None


def load_synthetic_domain(dataset_dir: Path, seed: int = 42) -> DomainBundle:
    """Load visible positive synthetic records using clean/noisy pixel pairs."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}")

    splits: dict[str, list[ROIRecord]] = {split: [] for split in SPLITS}
    stats: dict[str, Any] = {
        "raw_records": 0,
        "loaded": 0,
        "skipped_no_fire": 0,
        "skipped_occluded": 0,
        "skipped_missing_gt": 0,
        "skipped_missing_coarse": 0,
        "skipped_missing_image": 0,
        "skipped_invalid": 0,
    }
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["raw_records"] += 1
        row = json.loads(line)
        split = str(row.get("split", ""))
        if split not in splits:
            stats["skipped_invalid"] += 1
            continue
        if int(row.get("has_fire", 0)) != 1:
            if int(row.get("fire_event", 0)) == 1 and int(row.get("fire_visible", 0)) == 0:
                stats["skipped_occluded"] += 1
            else:
                stats["skipped_no_fire"] += 1
            continue
        relative_path = row.get("image_path")
        if not relative_path:
            stats["skipped_missing_image"] += 1
            continue
        image_path = (dataset_dir / str(relative_path)).resolve()
        if not image_path.is_file():
            stats["skipped_missing_image"] += 1
            continue
        width, height = _image_size(image_path, row.get("image_size"))
        gt = _pixel_to_norm(row.get("p_fire_pixel"), width, height)
        coarse = _pixel_to_norm(row.get("p_fire_noisy_pixel"), width, height)
        if gt is None:
            stats["skipped_missing_gt"] += 1
            continue
        if coarse is None:
            stats["skipped_missing_coarse"] += 1
            continue
        sample_id = str(row.get("sample_id", image_path.stem))
        splits[split].append(
            ROIRecord(
                image_path=str(image_path),
                gt_xy_norm=gt,
                coarse_xy_norm=coarse,
                split=split,
                source="synthetic",
                group=str(row.get("scene_id", sample_id)),
                sample_id=sample_id,
                point_source="synthetic_p_fire_noisy_pixel",
                image_width=width,
                image_height=height,
            )
        )
        stats["loaded"] += 1

    stats["counts"] = {split: len(values) for split, values in splits.items()}
    stats["seed"] = int(seed)
    return DomainBundle("synthetic", splits, stats)


def load_real_domain(
    labels_path: Path,
    dataset_root: Path,
    seed: int = 42,
    coarse_manifest: Optional[Path] = None,
    coarse_noise_px: float = 3.0,
) -> DomainBundle:
    """Load real ``p_fire`` labels and derive a reproducible coarse input.

    If ``coarse_manifest`` is supplied, its detector point is used and records
    without a point are excluded. Otherwise, the coarse point is generated by
    adding Gaussian pixel noise to the clean annotation. This isolates ROI
    training from classification recall while remaining explicit in metadata.
    """
    raw_records, load_stats = load_records(Path(labels_path), Path(dataset_root))
    split_records_map = split_records(raw_records, seed=seed)
    manifest = load_coarse_manifest(coarse_manifest)
    splits: dict[str, list[ROIRecord]] = {split: [] for split in SPLITS}
    stats: dict[str, Any] = {
        "loader": load_stats,
        "raw_positive_records": sum(record.has_fire for record in raw_records),
        "skipped_no_fire": 0,
        "skipped_missing_coarse": 0,
        "coarse_mode": "manifest" if manifest else "deterministic_gaussian_from_p_fire",
        "coarse_noise_px": float(coarse_noise_px),
    }
    for split, records in split_records_map.items():
        for index, record in enumerate(records):
            if record.has_fire != 1:
                stats["skipped_no_fire"] += 1
                continue
            image_path = Path(record.image_path)
            if not image_path.is_file():
                stats["skipped_missing_coarse"] += 1
                continue
            width, height = _image_size(image_path)
            gt = _finite_norm((record.x_norm, record.y_norm))
            if gt is None:
                stats["skipped_missing_coarse"] += 1
                continue
            point_source = "real_simulated_coarse"
            if manifest:
                value = _lookup_manifest(manifest, str(image_path))
                coarse = _normalise_manifest_point(value, width, height)
                point_source = "real_detector_manifest"
            else:
                coarse = _simulated_coarse(
                    gt,
                    width,
                    height,
                    float(coarse_noise_px),
                    _stable_seed(str(image_path), seed),
                )
            if coarse is None:
                stats["skipped_missing_coarse"] += 1
                continue
            sample_id = image_path.stem
            splits[split].append(
                ROIRecord(
                    image_path=str(image_path.resolve()),
                    gt_xy_norm=gt,
                    coarse_xy_norm=coarse,
                    split=split,
                    source="real",
                    group=record.group,
                    sample_id=sample_id,
                    point_source=point_source,
                    image_width=width,
                    image_height=height,
                )
            )
    stats["counts"] = {split: len(values) for split, values in splits.items()}
    stats["seed"] = int(seed)
    return DomainBundle("real", splits, stats)


def limit_records(records: Sequence[ROIRecord], maximum: int, seed: int) -> list[ROIRecord]:
    """Deterministically cap a split without changing its labels."""
    records = list(records)
    if maximum <= 0 or len(records) <= maximum:
        return records
    rng = random.Random(seed)
    selected = rng.sample(records, int(maximum))
    return sorted(selected, key=lambda record: record.sample_id)


def assert_disjoint_splits(splits: dict[str, Sequence[ROIRecord]], label: str) -> None:
    owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            key = str(Path(record.image_path).resolve()).lower()
            previous = owners.get(key)
            if previous is not None and previous != split:
                raise RuntimeError(f"{label} leakage: {key} appears in {previous} and {split}")
            owners[key] = split
            group_key = str(record.group)
            previous_group = group_owners.get(group_key)
            if previous_group is not None and previous_group != split:
                raise RuntimeError(
                    f"{label} group leakage: {group_key} appears in {previous_group} and {split}"
                )
            group_owners[group_key] = split


class ROIRefinerDataset(Dataset):
    """Crop around coarse input and return target in ROI-relative coordinates."""

    def __init__(
        self,
        records: Sequence[ROIRecord],
        train: bool,
        repeats: int = 1,
        roi_fraction: float = 0.70,
        coarse_jitter_px: float = 0.0,
        flip_p: float = 0.20,
        seed: int = 42,
    ):
        self.records = list(records)
        if not self.records:
            raise ValueError("ROIRefinerDataset received no records")
        self.train = bool(train)
        self.repeats = max(1, int(repeats if train else 1))
        self.roi_fraction = float(roi_fraction)
        self.coarse_jitter_px = max(0.0, float(coarse_jitter_px))
        self.flip_p = float(np.clip(flip_p, 0.0, 1.0))
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records) * self.repeats

    def __getitem__(self, index: int):
        record = self.records[index % len(self.records)]
        rng = np.random.default_rng(self.seed + int(index) * 1009)
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt_norm = np.asarray(record.gt_xy_norm, dtype=np.float64)
        coarse_norm = np.asarray(record.coarse_xy_norm, dtype=np.float64)

        if self.train and self.coarse_jitter_px > 0.0:
            coarse_norm = coarse_norm + rng.normal(
                0.0,
                [self.coarse_jitter_px / max(width, 1), self.coarse_jitter_px / max(height, 1)],
                size=2,
            )
            coarse_norm = np.clip(coarse_norm, 0.01, 0.99)

        if self.train and rng.random() < self.flip_p:
            image = TF.hflip(image)
            gt_norm[0] = 1.0 - gt_norm[0]
            coarse_norm[0] = 1.0 - coarse_norm[0]

        if self.train:
            image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
            image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
            image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.90, 1.10)))

        coarse_px = coarse_norm * np.asarray([width, height], dtype=np.float64)
        gt_px = gt_norm * np.asarray([width, height], dtype=np.float64)
        crop, box = _crop_square(image, coarse_px, self.roi_fraction)
        target_rel = np.clip((gt_px - np.asarray([box.left, box.top])) / box.side, 0.0, 1.0)
        meta = torch.tensor(
            [
                box.left,
                box.top,
                box.side,
                width,
                height,
                gt_px[0],
                gt_px[1],
                coarse_px[0],
                coarse_px[1],
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(target_rel, dtype=torch.float32)
        return _prepare_image(crop), target, meta, record.image_path


def records_manifest(domains: Iterable[DomainBundle]) -> dict[str, Any]:
    return {
        domain.name: {
            "counts": domain.counts(),
            "stats": domain.stats,
            "records": {
                split: [record.to_dict() for record in domain.splits.get(split, [])]
                for split in SPLITS
            },
        }
        for domain in domains
    }
