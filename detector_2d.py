"""Higher-resolution 2D fire detector used before the ROI refiner.

This module deliberately does not replace the existing ``best.pth`` model.
It provides a separate checkpoint contract for the detector stage:

    image -> fire confidence + coarse fire point -> ROIRefiner

The important differences from the old GAP-based detector are:

* letterbox resizing preserves the original aspect ratio and keeps the point
  transform invertible;
* a MobileNetV4 feature pyramid fuses reduction 4/8/16 maps for localisation;
* the heatmap is kept at roughly one quarter of the input resolution instead
  of being produced from the final low-resolution feature map only;
* confidence and point losses are separated, and point loss is positive-only;
* checkpoint metadata records image size and preprocessing so inference can
  convert the point back to the original image coordinates correctly.

The trainer consumes the existing ``dataset_labels (1).json`` p_fire labels.
The YOLO ``home-fire-dataset`` is intentionally not mixed here: it has bbox
labels rather than the point/contact labels required by this detector.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from project_paths import CCTV_DATASET


ARCHITECTURE = "detector_2d_fpn_v2"
DEFAULT_BACKBONE = "mobilenetv4_conv_medium"
# (width, height) for PIL and the checkpoint. 640x640 keeps more detail in
# small flames before the coarse point is passed to ROIRefiner.
DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_TEMPERATURE = 0.055
DEFAULT_FPN_CHANNELS = 96
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class LetterboxMeta:
    """Geometry needed to map a point between original and model images."""

    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale: float
    pad_x: float
    pad_y: float

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.scale,
                self.pad_x,
                self.pad_y,
                float(self.target_width),
                float(self.target_height),
                float(self.original_width),
                float(self.original_height),
            ],
            dtype=torch.float32,
        )


def _size_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        value = (value, value)
    values = tuple(int(item) for item in value)
    if len(values) != 2 or min(values) <= 0:
        raise ValueError(f"Invalid image size: {value!r}")
    return values


def resolve_torch_device(value: Optional[str | torch.device] = None) -> torch.device:
    """Normalize local/Kaggle device spellings to a valid ``torch.device``.

    Ultralytics commonly uses ``0`` for the first GPU, while PyTorch expects
    ``cuda:0``. Accept both forms so the same command works in Kaggle and in
    the project virtual environment. A requested CUDA device fails loudly
    instead of silently falling back to CPU.
    """
    if isinstance(value, torch.device):
        device = value
    elif value is None or str(value).strip().lower() in {"", "auto"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        text = str(value).strip().lower()
        if text.isdigit():
            text = f"cuda:{text}"
        device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({device}) but torch.cuda.is_available() is False. "
            "In Kaggle select GPU T4 before running the training cell."
        )
    return device


def letterbox_image(
    image: Image.Image,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, LetterboxMeta]:
    """Resize with padding, retaining an exact inverse point transform."""
    image = image.convert("RGB")
    target_width, target_height = _size_pair(image_size)
    original_width, original_height = image.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image has an invalid size")

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), fill)
    pad_x = (target_width - resized_width) / 2.0
    pad_y = (target_height - resized_height) / 2.0
    canvas.paste((resized), (int(round(pad_x)), int(round(pad_y))))
    # Use the actual integer paste position. This avoids a one-pixel mismatch
    # when a rounded half-pixel pad is used for odd dimensions.
    actual_pad_x = float(int(round(pad_x)))
    actual_pad_y = float(int(round(pad_y)))
    return canvas, LetterboxMeta(
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
        scale=scale,
        pad_x=actual_pad_x,
        pad_y=actual_pad_y,
    )


def point_to_letterbox(
    xy_norm: Sequence[float],
    meta: LetterboxMeta,
) -> tuple[float, float]:
    """Convert an original-image normalized point into model-image coords."""
    x = float(xy_norm[0]) * meta.original_width
    y = float(xy_norm[1]) * meta.original_height
    x = (x * meta.scale + meta.pad_x) / meta.target_width
    y = (y * meta.scale + meta.pad_y) / meta.target_height
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


def coords_from_letterbox(coords: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Vectorized inverse transform for ``[batch, 2]`` normalized coords."""
    scale = meta[:, 0].clamp_min(1e-8)
    pad_x, pad_y = meta[:, 1], meta[:, 2]
    target_width, target_height = meta[:, 3], meta[:, 4]
    original_width, original_height = meta[:, 5].clamp_min(1.0), meta[:, 6].clamp_min(1.0)
    x_px = (coords[:, 0] * target_width - pad_x) / scale
    y_px = (coords[:, 1] * target_height - pad_y) / scale
    x_norm = x_px / original_width
    y_norm = y_px / original_height
    return torch.stack((x_norm.clamp(0.0, 1.0), y_norm.clamp(0.0, 1.0)), dim=1)


def _feature_dicts(feature_info: Any) -> list[dict[str, Any]]:
    if feature_info is None:
        return []
    getter = getattr(feature_info, "get_dicts", None)
    if callable(getter):
        return [dict(item) for item in getter()]
    try:
        return [dict(item) for item in feature_info]
    except (TypeError, ValueError):
        return []


def _closest_unique_indices(reductions: Sequence[int], targets: Sequence[int]) -> list[int]:
    """Select distinct feature maps closest to requested reductions."""
    available = set(range(len(reductions)))
    selected: list[int] = []
    for target in targets:
        if not available:
            selected.append(min(range(len(reductions)), key=lambda i: abs(int(reductions[i]) - target)))
            continue
        index = min(available, key=lambda i: abs(int(reductions[i]) - target))
        selected.append(index)
        available.remove(index)
    return selected


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class Detector2DFPNModel(nn.Module):
    """MobileNetV4 FPN with high-resolution point heatmap and classifier."""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        fpn_channels: int = DEFAULT_FPN_CHANNELS,
        temperature: float = DEFAULT_TEMPERATURE,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = str(backbone)
        self.image_size = _size_pair(image_size)
        self.fpn_channels = int(fpn_channels)
        self.temperature = float(temperature)
        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=bool(pretrained),
            features_only=True,
        )
        metadata = _feature_dicts(getattr(self.backbone, "feature_info", None))
        if not metadata:
            raise RuntimeError(f"Backbone {self.backbone_name!r} không có feature_info")
        reductions = [int(item.get("reduction", 0)) for item in metadata]
        channels = [int(item.get("num_chs", 0)) for item in metadata]
        if any(value <= 0 for value in channels):
            raise RuntimeError(f"feature_info không hợp lệ: {metadata!r}")

        self.reduction4_index, self.reduction8_index, self.reduction16_index, self.reduction32_index = (
            _closest_unique_indices(reductions, (4, 8, 16, 32))
        )
        self.lat4 = nn.Conv2d(channels[self.reduction4_index], self.fpn_channels, 1)
        self.lat8 = nn.Conv2d(channels[self.reduction8_index], self.fpn_channels, 1)
        self.lat16 = nn.Conv2d(channels[self.reduction16_index], self.fpn_channels, 1)
        self.fpn_neck = nn.Sequential(
            ConvBNAct(self.fpn_channels, self.fpn_channels),
            ConvBNAct(self.fpn_channels, self.fpn_channels),
        )
        self.heatmap_head = nn.Conv2d(self.fpn_channels, 1, 1)
        self.class_head = nn.Sequential(
            nn.Linear(channels[self.reduction32_index], 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(256, 1),
        )
        target_width, target_height = self.image_size
        self.heatmap_size = (max(16, target_height // 4), max(16, target_width // 4))

    @staticmethod
    def _soft_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        batch, _, height, width = logits.shape
        probability = F.softmax(logits.flatten(1) / max(float(temperature), 1e-5), dim=1)
        probability = probability.view(batch, height, width)
        xs = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        x = (probability * xs.view(1, 1, width)).sum(dim=(1, 2))
        y = (probability * ys.view(1, height, 1)).sum(dim=(1, 2))
        return torch.stack((x, y), dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        f4 = features[self.reduction4_index]
        f8 = features[self.reduction8_index]
        f16 = features[self.reduction16_index]
        f32 = features[self.reduction32_index]

        p4 = self.lat4(f4)
        p8 = F.interpolate(self.lat8(f8), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p16 = F.interpolate(self.lat16(f16), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fpn_neck(p4 + p8 + p16)
        heatmap_logits = self.heatmap_head(fused)
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=self.heatmap_size,
            mode="bilinear",
            align_corners=False,
        )
        cls_feature = F.adaptive_avg_pool2d(f32, 1).flatten(1)
        confidence_logit = self.class_head(cls_feature).squeeze(1)
        return {
            "confidence_logit": confidence_logit,
            "coord": self._soft_argmax(heatmap_logits, self.temperature),
            "heatmap_logits": heatmap_logits,
        }


class Detector2DDataset(Dataset):
    """Dataset with synchronized point geometry and aspect-ratio-safe resize."""

    def __init__(
        self,
        records: Sequence[Any],
        train: bool = False,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        flip_p: float = 0.30,
    ):
        self.records = list(records)
        self.train = bool(train)
        self.image_size = _size_pair(image_size)
        self.flip_p = float(flip_p)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        x_norm, y_norm = record.x_norm, record.y_norm

        if self.train and random.random() < self.flip_p:
            image = TF.hflip(image)
            if record.has_fire:
                x_norm = 1.0 - x_norm

        if self.train:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.82, 1.18))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.18))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.88, 1.12))

        image, meta = letterbox_image(image, self.image_size)
        input_xy = point_to_letterbox((x_norm, y_norm), meta) if record.has_fire else (0.0, 0.0)
        # This is the target in the current (possibly flipped) image frame.
        # Keeping it synchronized makes train-time metrics meaningful too.
        original_xy = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
        target = torch.tensor([float(record.has_fire), *input_xy], dtype=torch.float32)
        return tensor, target, original_xy, meta.as_tensor(), record.image_path


def gaussian_targets(
    target_xy: torch.Tensor,
    positive: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=target_xy.device, dtype=target_xy.dtype),
        torch.arange(width, device=target_xy.device, dtype=target_xy.dtype),
        indexing="ij",
    )
    cx = target_xy[:, 0] * max(width - 1, 1)
    cy = target_xy[:, 1] * max(height - 1, 1)
    heat = torch.exp(
        -((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2)
        / (2.0 * sigma**2)
    )
    return heat[:, None] * positive[:, None, None, None]


class Detector2DLoss(nn.Module):
    """Confidence + positive-only coordinate + weighted heatmap loss."""

    def __init__(
        self,
        positive_weight: float = 1.0,
        lambda_coord: float = 7.0,
        lambda_heatmap: float = 1.0,
        heatmap_sigma: float = 2.0,
    ):
        super().__init__()
        self.lambda_coord = float(lambda_coord)
        self.lambda_heatmap = float(lambda_heatmap)
        self.heatmap_sigma = float(heatmap_sigma)
        self.confidence = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(max(1.0, positive_weight))))

    def forward(self, outputs: Mapping[str, torch.Tensor], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        positive = targets[:, 0] > 0.5
        # Move the fixed class-balance tensor with the output on first use.
        self.confidence.pos_weight = self.confidence.pos_weight.to(outputs["confidence_logit"].device)
        confidence_loss = self.confidence(outputs["confidence_logit"], targets[:, 0])
        if positive.any():
            coordinate_loss = F.smooth_l1_loss(
                outputs["coord"][positive], targets[positive, 1:], beta=0.02
            )
        else:
            coordinate_loss = outputs["coord"].sum() * 0.0

        _, _, height, width = outputs["heatmap_logits"].shape
        target_heatmap = gaussian_targets(
            targets[:, 1:], positive.float(), height, width, sigma=self.heatmap_sigma
        )
        pixel_loss = F.binary_cross_entropy_with_logits(
            outputs["heatmap_logits"], target_heatmap, reduction="none"
        )
        # Emphasize the gaussian centre without making the huge background
        # dominate the point signal.
        pixel_weight = 1.0 + 4.0 * target_heatmap
        heatmap_loss = (pixel_loss * pixel_weight).mean()
        total = confidence_loss + self.lambda_coord * coordinate_loss + self.lambda_heatmap * heatmap_loss
        return {
            "total": total,
            "confidence": confidence_loss,
            "coordinate": coordinate_loss,
            "heatmap": heatmap_loss,
        }


@dataclass
class Detector2DResult:
    """Stable inference result for the improved detector."""

    confidence: float
    pixel: Optional[tuple[float, float]]
    detected: bool
    size: tuple[int, int]
    latency_ms: float = 0.0

    @property
    def p_fire(self) -> float:
        return self.confidence

    @property
    def point(self) -> Optional[tuple[float, float]]:
        return self.pixel


class Detector2DInference:
    """Load ``detector_2d_fpn_v2`` and return original-image pixel points."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str | torch.device] = None,
        threshold: float = 0.5,
        use_amp: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.device = resolve_torch_device(device)
        self.threshold = float(threshold)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("detector_2d checkpoint phải là một mapping")
        architecture = checkpoint.get("architecture")
        if architecture != ARCHITECTURE:
            raise ValueError(
                f"Checkpoint có architecture={architecture!r}, cần {ARCHITECTURE!r}"
            )
        self.image_size = _size_pair(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
        self.model = Detector2DFPNModel(
            backbone=str(checkpoint.get("backbone", DEFAULT_BACKBONE)),
            image_size=self.image_size,
            fpn_channels=int(checkpoint.get("fpn_channels", DEFAULT_FPN_CHANNELS)),
            temperature=float(checkpoint.get("temperature", DEFAULT_TEMPERATURE)),
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self._warm = False

    @staticmethod
    def _to_image(item: Any) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, (str, Path)):
            return Image.open(item).convert("RGB")
        if isinstance(item, np.ndarray):
            array = item if item.dtype == np.uint8 else np.clip(item, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                return Image.fromarray(array, mode="L").convert("RGB")
            return Image.fromarray(array[..., :3]).convert("RGB")
        raise TypeError(f"Unsupported image input: {type(item)!r}")

    def warmup(self, repeats: int = 2) -> None:
        dummy = torch.zeros(
            (1, 3, self.image_size[1], self.image_size[0]),
            device=self.device,
        )
        with torch.inference_mode():
            for _ in range(max(1, int(repeats))):
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.use_amp else contextlib.nullcontext()
                )
                with context:
                    self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._warm = True

    @torch.inference_mode()
    def detect_many(self, items: Sequence[Any], warmup: bool = False) -> list[Detector2DResult]:
        if not items:
            return []
        if warmup and not self._warm:
            self.warmup()
        images = [self._to_image(item) for item in items]
        sizes = [image.size for image in images]
        letterboxed: list[Image.Image] = []
        metas: list[torch.Tensor] = []
        for image in images:
            prepared, meta = letterbox_image(image, self.image_size)
            letterboxed.append(prepared)
            metas.append(meta.as_tensor())
        batch = torch.stack(
            [TF.normalize(TF.to_tensor(image), MEAN, STD) for image in letterboxed]
        ).to(self.device)
        metadata = torch.stack(metas).to(self.device)
        start = time.perf_counter()
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp else contextlib.nullcontext()
        )
        with context:
            raw = self.model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        probabilities = torch.sigmoid(raw["confidence_logit"]).float()
        original_coords = coords_from_letterbox(raw["coord"].float(), metadata)
        results: list[Detector2DResult] = []
        for probability, coordinate, (width, height) in zip(probabilities, original_coords, sizes):
            score = float(probability.item())
            x_pixel = float(coordinate[0].item()) * width
            y_pixel = float(coordinate[1].item()) * height
            point = (
                float(np.clip(x_pixel, 0.0, max(0, width - 1))),
                float(np.clip(y_pixel, 0.0, max(0, height - 1))),
            )
            detected = score >= self.threshold
            results.append(
                Detector2DResult(
                    confidence=score,
                    pixel=point if detected else None,
                    detected=detected,
                    size=(width, height),
                    latency_ms=elapsed_ms / len(images),
                )
            )
        return results

    def detect(self, item: Any, warmup: bool = False) -> Detector2DResult:
        return self.detect_many([item], warmup=warmup)[0]


@torch.no_grad()
def detector_metrics(
    confidence_logit: torch.Tensor,
    coords: torch.Tensor,
    targets: torch.Tensor,
    original_xy: torch.Tensor,
    meta: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probability = torch.sigmoid(confidence_logit)
    true_positive = targets[:, 0] > 0.5
    predicted_positive = probability >= float(threshold)
    tp = int((predicted_positive & true_positive).sum())
    fp = int((predicted_positive & ~true_positive).sum())
    fn = int((~predicted_positive & true_positive).sum())
    tn = int((~predicted_positive & ~true_positive).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    result = {
        "accuracy": float((predicted_positive == true_positive).float().mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "detected": float(predicted_positive.sum()),
        "total": float(len(targets)),
    }
    if not true_positive.any():
        result.update(
            {
                "mae_px": float("nan"),
                "median_px": float("nan"),
                "p95_px": float("nan"),
                "mae_x_px": float("nan"),
                "mae_y_px": float("nan"),
                "pck10": float("nan"),
                "pck25": float("nan"),
            }
        )
        return result

    predicted_original = coords_from_letterbox(coords, meta)[true_positive]
    truth = original_xy[true_positive]
    dimensions = torch.stack((meta[true_positive, 5], meta[true_positive, 6]), dim=1)
    delta_px = (predicted_original - truth) * dimensions
    radial_error = torch.linalg.vector_norm(delta_px, dim=1)
    result.update(
        {
            "mae_px": float(radial_error.mean()),
            "median_px": float(radial_error.median()),
            "p95_px": float(torch.quantile(radial_error, 0.95)),
            "mae_x_px": float(delta_px[:, 0].abs().mean()),
            "mae_y_px": float(delta_px[:, 1].abs().mean()),
            "pck10": float((radial_error <= 10.0).float().mean()),
            "pck25": float((radial_error <= 25.0).float().mean()),
        }
    )
    return result


def _mean_losses(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    return {key: float(np.mean([item[key] for item in values])) for key in values[0]}


def _merge_metric_batches(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Merge metrics from batches using counts instead of averaging ratios."""
    if not values:
        return {}
    tp = sum(int(item["tp"]) for item in values)
    fp = sum(int(item["fp"]) for item in values)
    fn = sum(int(item["fn"]) for item in values)
    tn = sum(int(item["tn"]) for item in values)
    total = max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result = {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "detected": float(tp + fp),
        "total": float(total),
    }
    for key in ("mae_px", "median_px", "p95_px", "mae_x_px", "mae_y_px", "pck10", "pck25"):
        finite = [float(item[key]) for item in values if np.isfinite(item[key])]
        result[key] = float(np.mean(finite)) if finite else float("nan")
    return result


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Detector2DLoss,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    threshold: float = 0.5,
    accumulation_steps: int = 1,
) -> tuple[dict[str, float], dict[str, float]]:
    training = optimizer is not None
    accumulation_steps = max(1, int(accumulation_steps))
    model.train(training)
    # When the backbone is frozen, do not update its BatchNorm running
    # statistics while the new heads are being warmed up.
    if training and any(not parameter.requires_grad for parameter in model.backbone.parameters()):
        model.backbone.eval()
    losses: list[dict[str, float]] = []
    confidence_values: list[torch.Tensor] = []
    coordinate_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    original_values: list[torch.Tensor] = []
    metadata_values: list[torch.Tensor] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, targets, original_xy, meta, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        original_xy = original_xy.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        with _autocast(device):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
        if training:
            loss_for_backward = loss_dict["total"] / accumulation_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            is_update = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if is_update:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append({key: float(value.detach().float().cpu()) for key, value in loss_dict.items()})
        confidence_values.append(outputs["confidence_logit"].detach().float().cpu())
        coordinate_values.append(outputs["coord"].detach().float().cpu())
        target_values.append(targets.detach().float().cpu())
        original_values.append(original_xy.detach().float().cpu())
        metadata_values.append(meta.detach().float().cpu())
    if not losses:
        raise RuntimeError("DataLoader produced no batches")
    merged_metric = detector_metrics(
        torch.cat(confidence_values),
        torch.cat(coordinate_values),
        torch.cat(target_values),
        torch.cat(original_values),
        torch.cat(metadata_values),
        threshold=threshold,
    )
    return _mean_losses(losses), merged_metric


def _set_backbone_trainable(model: Detector2DFPNModel, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = bool(trainable)


def _checkpoint_state(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}
    for key in ("model", "state_dict", "model_state_dict", "weights", "net"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def load_backbone_initialization(model: Detector2DFPNModel, checkpoint_path: Path) -> dict[str, int]:
    """Transfer only matching backbone tensors from an older checkpoint.

    The old ``best.pth`` and ``best_spatial.pth`` use different heads, so a
    strict whole-model load would be wrong. Matching backbone tensors are safe
    to reuse and leave the new FPN/head randomly initialized.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _checkpoint_state(checkpoint)
    model_state = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        for prefix in ("module.", "model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if not key.startswith("backbone."):
            continue
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            transferred[key] = value
    model_state.update(transferred)
    model.load_state_dict(model_state, strict=True)
    return {"matched": len(transferred), "available": sum(1 for key in state if str(key).endswith("weight") or str(key).endswith("bias"))}


def _make_scaler(device: torch.device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    path: Path,
    model: Detector2DFPNModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_loss: float,
    split_info: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "backbone": model.backbone_name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "image_size": model.image_size,
            "heatmap_size": model.heatmap_size,
            "temperature": model.temperature,
            "fpn_channels": model.fpn_channels,
            "mean": MEAN,
            "std": STD,
            "split_info": dict(split_info),
            "args": vars(args),
        },
        path,
    )


def _load_records_and_splits(labels: Path, dataset_root: Path, seed: int):
    # Reuse the project's path resolution and leakage-safe official split logic.
    from train_week6 import load_records, split_records

    records, load_stats = load_records(labels, dataset_root)
    splits = split_records(records, seed=seed)
    return records, load_stats, splits


def _default_init_checkpoint(root: Path) -> Optional[Path]:
    candidates = (
        root / "fire-model-data" / "week6_spatial" / "best_spatial.pth",
        root / "fire-model-data" / "best.pth",
    )
    return next((path for path in candidates if path.is_file()), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the improved 2D fire detector before ROI")
    root = Path(__file__).resolve().parent
    parser.add_argument("--labels", type=Path, default=root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "fire-model-data" / "detector_2d_v2",
    )
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--image-size", "--imgsz", dest="image_size", type=int, default=640)
    parser.add_argument("--fpn-channels", type=int, default=DEFAULT_FPN_CHANNELS)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=2 if Path("/kaggle").is_dir() else 0,
        help="DataLoader workers; 2 is a safe Kaggle T4 default",
    )
    parser.add_argument(
        "--grad-accumulation",
        type=int,
        default=1,
        help="Accumulate this many mini-batches before an optimizer step",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=_default_init_checkpoint(root),
        help="optional old detector checkpoint; matching backbone weights are transferred only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="auto, cpu, cuda, cuda:0, or 0 (0 is normalized to cuda:0)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="run one train and validation epoch")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_torch_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    image_size = (int(args.image_size), int(args.image_size))
    records, load_stats, splits = _load_records_and_splits(args.labels, args.dataset_root, args.seed)
    if not records:
        raise RuntimeError(f"Không load được record nào: {load_stats}")
    split_info = {
        name: {
            "count": len(items),
            "fire": sum(int(record.has_fire) for record in items),
            "no_fire": sum(int(not record.has_fire) for record in items),
        }
        for name, items in splits.items()
    }
    if any(split_info[name]["count"] == 0 for name in ("train", "val", "test")):
        raise RuntimeError(f"Empty split: {split_info}")

    train_loader = DataLoader(
        Detector2DDataset(splits["train"], train=True, image_size=image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Detector2DDataset(splits["val"], image_size=image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Detector2DDataset(splits["test"], image_size=image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = Detector2DFPNModel(
        backbone=args.backbone,
        image_size=image_size,
        fpn_channels=args.fpn_channels,
        # When an existing checkpoint is supplied, use it as the local
        # backbone initialization and avoid an unnecessary timm download.
        pretrained=not args.no_pretrained and args.init_checkpoint is None,
    ).to(device)
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(f"init checkpoint không tồn tại: {args.init_checkpoint}")
        transfer_info = load_backbone_initialization(model, args.init_checkpoint)
        print(f"backbone_init={args.init_checkpoint} transfer={transfer_info}")
    positive = split_info["train"]["fire"]
    negative = split_info["train"]["no_fire"]
    criterion = Detector2DLoss(positive_weight=negative / max(1, positive))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("backbone.")
                ],
                "lr": args.lr,
            },
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr / 100.0
    )
    scaler = _make_scaler(device)
    print(
        f"architecture={ARCHITECTURE} device={device} image_size={image_size} "
        f"amp={device.type == 'cuda'} batch={args.batch_size} "
        f"grad_accumulation={max(1, args.grad_accumulation)} workers={args.workers}"
    )
    print(f"load_stats={load_stats}")
    print(f"split_info={split_info}")

    if args.smoke_test:
        _set_backbone_trainable(model, False)
        train_loss, train_metric = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.threshold,
            args.grad_accumulation,
        )
        val_loss, val_metric = run_epoch(model, val_loader, criterion, device, threshold=args.threshold)
        print(f"smoke_train_loss={train_loss}")
        print(f"smoke_train_metric={train_metric}")
        print(f"smoke_val_loss={val_loss}")
        print(f"smoke_val_metric={val_metric}")
        return

    best = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        _set_backbone_trainable(model, epoch > args.freeze_epochs)
        train_loss, train_metric = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.threshold,
            args.grad_accumulation,
        )
        val_loss, val_metric = run_epoch(model, val_loader, criterion, device, threshold=args.threshold)
        scheduler.step()
        line = {
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "train_metric": train_metric,
            "val_metric": val_metric,
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(line)
        print(
            f"epoch={epoch:03d} train={train_loss['total']:.5f} val={val_loss['total']:.5f} "
            f"val_f1={val_metric['f1']:.3f} val_mae={val_metric['mae_px']:.2f}px "
            f"val_pck10={val_metric['pck10']:.3f} val_pck25={val_metric['pck25']:.3f}"
        )
        if val_loss["total"] < best:
            best = val_loss["total"]
            save_checkpoint(
                args.output_dir / "best_detector_2d.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best,
                split_info,
                args,
            )
        save_checkpoint(
            args.output_dir / "last_detector_2d.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss["total"],
            split_info,
            args,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best_checkpoint = args.output_dir / "best_detector_2d.pth"
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    test_loss, test_metric = run_epoch(model, test_loader, criterion, device, threshold=args.threshold)
    print(f"test_loss={test_loss}")
    print(f"test_metric={test_metric}")
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps({"architecture": ARCHITECTURE, "loss": test_loss, "metric": test_metric}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

