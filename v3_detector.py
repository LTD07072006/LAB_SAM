"""Inference adapter for the Week 6 v3 FPN fire-point detector.

The v3 model is intentionally separate from ``fire_detector.FireDetector``.
Its coordinate branch keeps spatial information by fusing backbone features at
reduction 4 and reduction 8, producing a 64x64 heatmap, while a reduction-32
feature map supplies the fire/no-fire classification logit.  The reported
point is obtained with a differentiable soft-argmax over the heatmap.

This file is also the compatibility boundary for old v3 checkpoints.  It
accepts checkpoints saved as ``{"model": state_dict}``,
``{"state_dict": state_dict}``, or a raw state dict and removes the common
``module.`` prefix produced by ``DataParallel``.

A checkpoint cannot be reconstructed from its filename alone.  If
``torch.load`` rejects a file, the caller receives an actionable error rather
than silently comparing a different architecture.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ARCHITECTURE = "mobilenetv4_fpn_heatmap_v3"
DEFAULT_BACKBONE = "mobilenetv4_conv_medium"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_HEATMAP_SIZE = 64
DEFAULT_TEMPERATURE = 0.07
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class V3Detection:
    """Detection returned in the original image's pixel coordinate system."""

    confidence: float
    pixel: Optional[tuple[float, float]]
    detected: bool
    size: tuple[int, int]
    latency_ms: float = 0.0
    heatmap: Optional[np.ndarray] = None

    @property
    def point(self) -> Optional[tuple[float, float]]:
        return self.pixel

    @property
    def p_fire(self) -> float:
        return self.confidence

    @property
    def u(self) -> Optional[int]:
        return None if self.pixel is None else int(round(self.pixel[0]))

    @property
    def v(self) -> Optional[int]:
        return None if self.pixel is None else int(round(self.pixel[1]))

    def __iter__(self):
        yield self.confidence
        yield self.u
        yield self.v


def _feature_dicts(feature_info: Any) -> list[dict[str, Any]]:
    """Return timm feature metadata across several timm releases."""
    if feature_info is None:
        return []
    getter = getattr(feature_info, "get_dicts", None)
    if callable(getter):
        return [dict(item) for item in getter()]
    try:
        return [dict(item) for item in feature_info]
    except (TypeError, ValueError):
        return []


def _closest_index(reductions: Sequence[int], target: int) -> int:
    if not reductions:
        raise ValueError("Backbone không cung cấp feature_info")
    return min(range(len(reductions)), key=lambda index: abs(int(reductions[index]) - target))


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class V3FPNModel(nn.Module):
    """MobileNetV4 feature pyramid and spatial point head.

    ``features_only=True`` exposes intermediate maps.  The two maps closest
    to reductions 4 and 8 are fused for localisation; the map closest to
    reduction 32 is used only by the classifier.  The explicit resize to
    64x64 makes the checkpoint contract independent of the input image size.
    """

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        heatmap_size: int = DEFAULT_HEATMAP_SIZE,
        temperature: float = DEFAULT_TEMPERATURE,
        pretrained: bool = False,
        fpn_channels: int = 96,
    ):
        super().__init__()
        self.backbone_name = str(backbone)
        self.heatmap_size = int(heatmap_size)
        self.temperature = float(temperature)
        self.fpn_channels = int(fpn_channels)

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=bool(pretrained),
            features_only=True,
        )
        metadata = _feature_dicts(getattr(self.backbone, "feature_info", None))
        if not metadata:
            raise RuntimeError(
                f"Backbone {self.backbone_name!r} không có feature_info; "
                "không thể xác định reduction-4/8/32."
            )
        reductions = [int(item.get("reduction", 0)) for item in metadata]
        channels = [int(item.get("num_chs", 0)) for item in metadata]
        if any(value <= 0 for value in channels):
            raise RuntimeError(f"feature_info không hợp lệ: {metadata!r}")

        self.reduction4_index = _closest_index(reductions, 4)
        self.reduction8_index = _closest_index(reductions, 8)
        self.reduction32_index = _closest_index(reductions, 32)

        # Distinct lateral modules keep the two spatial scales explicit in the
        # state dict and make it easy to inspect/replace the FPN later.
        # Names and channel widths match the released v3 checkpoint:
        # lat4/lat8 -> neck -> heatmap_head.
        self.lat4 = nn.Conv2d(channels[self.reduction4_index], self.fpn_channels, 1)
        self.lat8 = nn.Conv2d(channels[self.reduction8_index], self.fpn_channels, 1)
        self.neck = ConvBNAct(self.fpn_channels, self.fpn_channels, 3)
        self.heatmap_head = nn.Conv2d(self.fpn_channels, 1, 1)

        # Reduction-32 is deliberately kept on the classification path.  The
        # classifier does not pool the localisation features into coordinates.
        # The released v3 checkpoint uses a 256-wide MLP/BN classifier from
        # the reduction-32 feature (960 channels for MobileNetV4 medium).
        self.cls_head = nn.Sequential(
            nn.Linear(channels[self.reduction32_index], 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _soft_argmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        batch, _, height, width = logits.shape
        probabilities = F.softmax(logits.flatten(1) / max(float(temperature), 1e-5), dim=1)
        probabilities = probabilities.view(batch, height, width)
        xs = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        ys = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        x = (probabilities * xs.view(1, 1, width)).sum(dim=(1, 2))
        y = (probabilities * ys.view(1, height, 1)).sum(dim=(1, 2))
        return torch.stack((x, y), dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        reduction4 = features[self.reduction4_index]
        reduction8 = features[self.reduction8_index]
        reduction32 = features[self.reduction32_index]

        p4 = self.lat4(reduction4)
        p8 = self.lat8(reduction8)
        p8 = F.interpolate(p8, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.neck(p4 + p8)
        heatmap_logits = self.heatmap_head(fused)
        heatmap_logits = F.interpolate(
            heatmap_logits,
            size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear",
            align_corners=False,
        )

        classification_feature = F.adaptive_avg_pool2d(reduction32, 1).flatten(1)
        confidence_logit = self.cls_head(classification_feature).squeeze(1)
        return {
            "confidence_logit": confidence_logit,
            "coord": self._soft_argmax(heatmap_logits, self.temperature),
            "heatmap_logits": heatmap_logits,
            "class_logit": confidence_logit,
        }


def _unwrap_state_dict(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract state dict and metadata from common torch checkpoint layouts."""
    if isinstance(checkpoint, Mapping):
        metadata = dict(checkpoint)
        for key in ("model", "state_dict", "model_state_dict", "weights", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return dict(candidate), metadata
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
            if tensor_values:
                return dict(checkpoint), metadata
    raise ValueError("Checkpoint v3 không chứa state_dict/model hợp lệ")


def _clean_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "model.", "net."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def load_v3_checkpoint(checkpoint_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint v3: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Không đọc được checkpoint v3 {path}. File có thể bị hỏng/ghi toàn byte 0x00; "
            "hãy chép lại best_v3_fpn.pth từ Kaggle hoặc GitHub. Chi tiết: "
            f"{exc}"
        ) from exc
    state, metadata = _unwrap_state_dict(checkpoint)
    return _clean_state_dict(state), metadata


class V3Detector:
    """Load and run the v3 FPN checkpoint through a stable detector API."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.5,
        use_amp: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = float(threshold)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        state, metadata = load_v3_checkpoint(self.checkpoint_path, self.device)
        self.metadata = metadata
        backbone = str(metadata.get("backbone", metadata.get("backbone_name", DEFAULT_BACKBONE)))
        image_size_value = metadata.get("image_size", metadata.get("img_size", DEFAULT_IMAGE_SIZE))
        if isinstance(image_size_value, int):
            image_size_value = (image_size_value, image_size_value)
        self.image_size = tuple(int(value) for value in image_size_value)
        heatmap_size = int(metadata.get("heatmap_size", metadata.get("output_size", DEFAULT_HEATMAP_SIZE)))
        temperature = float(metadata.get("temperature", metadata.get("softargmax_temperature", DEFAULT_TEMPERATURE)))
        fpn_channels = int(metadata.get("hidden_channels", metadata.get("fpn_channels", 96)))

        self.model = V3FPNModel(
            backbone=backbone,
            heatmap_size=heatmap_size,
            temperature=temperature,
            pretrained=False,
            fpn_channels=fpn_channels,
        ).to(self.device)
        try:
            result = self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            # Older v3 saves sometimes have DataParallel prefixes or a small
            # metadata mismatch.  Accept only a checkpoint with substantial
            # parameter overlap; never silently use random model weights.
            result = self.model.load_state_dict(state, strict=False)
            expected = set(self.model.state_dict())
            loaded = expected.intersection(state)
            overlap = len(loaded) / max(1, len(expected))
            if overlap < 0.80:
                raise RuntimeError(
                    "Checkpoint v3 không khớp kiến trúc FPN đã phục dựng "
                    f"(overlap={overlap:.1%}). Cần đúng code train v3 hoặc checkpoint tương ứng. "
                    f"Lỗi strict ban đầu: {exc}"
                ) from exc
            print(
                f"WARNING: tải v3 với strict=False; loaded={len(loaded)}/{len(expected)} "
                f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
            )
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self._warm = False

    @staticmethod
    def _to_image(item: Any) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, (str, Path)):
            return Image.open(item).convert("RGB")
        if isinstance(item, np.ndarray):
            array = item if item.dtype == np.uint8 else np.clip(item, 0, 255).astype(np.uint8)
            return Image.fromarray(array[..., :3]).convert("RGB")
        raise TypeError(f"Unsupported image input: {type(item)!r}")

    def warmup(self, repeats: int = 2) -> None:
        dummy = torch.zeros((1, 3, self.image_size[1], self.image_size[0]), device=self.device)
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
    def detect_many(self, items: Sequence[Any], warmup: bool = False) -> list[V3Detection]:
        if not items:
            return []
        if warmup and not self._warm:
            self.warmup()
        images = [self._to_image(item) for item in items]
        sizes = [image.size for image in images]
        batch = torch.stack([self.transform(image) for image in images]).to(self.device)
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
        confidence = torch.sigmoid(raw["confidence_logit"]).float().cpu().numpy()
        coordinates = raw["coord"].float().cpu().numpy()
        heatmaps = torch.sigmoid(raw["heatmap_logits"]).float().cpu().numpy()[:, 0]
        result: list[V3Detection] = []
        for probability, coordinate, heatmap, (width, height) in zip(confidence, coordinates, heatmaps, sizes):
            # Match the training convention used by train_week6 and
            # fire_detector: normalized labels are multiplied by width/height
            # and then clipped to the last valid pixel.
            point = (
                float(np.clip(coordinate[0] * width, 0.0, max(0, width - 1))),
                float(np.clip(coordinate[1] * height, 0.0, max(0, height - 1))),
            )
            score = float(probability)
            detected = score >= self.threshold
            result.append(V3Detection(
                confidence=score,
                pixel=point if detected else None,
                detected=detected,
                size=(width, height),
                latency_ms=elapsed_ms / len(images),
                heatmap=heatmap,
            ))
        return result

    def detect(self, item: Any, warmup: bool = False) -> V3Detection:
        return self.detect_many([item], warmup=warmup)[0]


# Names accepted by compare_v3_roi.py and by older experiment notebooks.
V3FireDetector = V3Detector
FireDetectorV3 = V3Detector
FPNFireDetector = V3Detector


def load_detector(checkpoint_path: Union[str, Path], device: Optional[str] = None, threshold: float = 0.5) -> V3Detector:
    return V3Detector(checkpoint_path, device=device, threshold=threshold)


__all__ = [
    "ARCHITECTURE",
    "V3Detection",
    "V3FPNModel",
    "V3Detector",
    "V3FireDetector",
    "FireDetectorV3",
    "FPNFireDetector",
    "load_detector",
]
