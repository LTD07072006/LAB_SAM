"""Fire detector adapter for legacy, spatial and improved 2D checkpoints.

The ``detector_2d_fpn_v2`` branch is intentionally exposed through this same
adapter so ROI and 3D code can consume the new detector without knowing which
2D architecture produced the coarse point.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

@dataclass
class DetectionResult:
    confidence: float
    pixel: Optional[tuple]
    detected: bool
    size: tuple
    latency_ms: float = 0.0
    @property
    def p_fire(self): return self.confidence
    @property
    def u(self): return None if self.pixel is None else int(self.pixel[0])
    @property
    def v(self): return None if self.pixel is None else int(self.pixel[1])
    def __iter__(self):
        yield self.confidence; yield self.u; yield self.v

class FireGrounder(nn.Module):
    def __init__(self, backbone="mobilenetv4_conv_medium", pretrained=False, in_features=1280):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.head = nn.Linear(in_features, 3)
    def forward(self, x): return torch.sigmoid(self.head(self.backbone(x)))

class FireDetector:
    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None,
                 threshold=0.5, use_amp=True):
        self.model_path = Path(model_path)
        if not self.model_path.exists(): raise FileNotFoundError(self.model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = float(threshold)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.architecture = checkpoint.get("architecture", "legacy_gap_v1")
        backbone = checkpoint.get("backbone", "mobilenetv4_conv_medium")
        self._detector_2d = None
        if self.architecture == "detector_2d_fpn_v2":
            # Keep the old FireDetector API as the integration boundary. This
            # lets all existing ROI/3D scripts consume the higher-resolution
            # detector without changing their downstream contracts.
            from detector_2d import Detector2DInference

            self._detector_2d = Detector2DInference(
                self.model_path,
                device=self.device,
                threshold=self.threshold,
                use_amp=self.use_amp,
            )
            self.model = self._detector_2d.model
            self.spatial_head = True
        elif self.architecture == "spatial_heatmap_v2":
            from train_week6 import SpatialFireModel
            self.model = SpatialFireModel(backbone=backbone, pretrained=False).to(self.device)
            self.spatial_head = True
        elif self.architecture.startswith("spatial_heatmap"):
            raise ValueError(
                f"Unsupported spatial checkpoint architecture {self.architecture!r}; "
                "retrain with train_week6.py (spatial_heatmap_v2)."
            )
        else:
            self.model = FireGrounder(backbone=backbone, pretrained=False).to(self.device)
            self.spatial_head = False
        # Detector2DInference already loaded and validated this checkpoint.
        # The other legacy branches still load their state dict here.
        if self._detector_2d is None:
            self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        if self.device.type == "cuda": torch.backends.cudnn.benchmark = True
        image_size = checkpoint.get("image_size", (224, 224))
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        image_size = tuple(int(value) for value in image_size)
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError(f"Invalid checkpoint image_size: {image_size!r}")
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._warm = False

    def _to_image(self, item):
        if isinstance(item, (str, Path)): return Image.open(item).convert("RGB")
        if isinstance(item, Image.Image): return item.convert("RGB")
        if isinstance(item, np.ndarray):
            array = item if item.dtype == np.uint8 else np.clip(item, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                return Image.fromarray(array, mode="L").convert("RGB")
            return Image.fromarray(array[..., :3]).convert("RGB")
        raise TypeError(f"Unsupported image input: {type(item)!r}")

    def warmup(self, repeats=3):
        if self._detector_2d is not None:
            self._detector_2d.warmup(repeats=repeats)
            self._warm = True
            return
        dummy = torch.zeros((1, 3, self.image_size[1], self.image_size[0]), device=self.device)
        with torch.inference_mode():
            for _ in range(max(1, int(repeats))):
                context = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
                with context: self.model(dummy)
        if self.device.type == "cuda": torch.cuda.synchronize()
        self._warm = True

    def detect_many(self, items: Sequence, warmup=False):
        if not items:
            return []
        if self._detector_2d is not None:
            results = self._detector_2d.detect_many(items, warmup=warmup)
            self._warm = self._detector_2d._warm
            return [
                DetectionResult(
                    confidence=result.confidence,
                    pixel=result.pixel,
                    detected=result.detected,
                    size=result.size,
                    latency_ms=result.latency_ms,
                )
                for result in results
            ]
        if warmup and not self._warm: self.warmup()
        images = [self._to_image(item) for item in items]
        sizes = [image.size for image in images]
        tensor = torch.stack([self.transform(image) for image in images]).to(self.device)
        with torch.inference_mode():
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with context:
                raw = self.model(tensor)
                if self.spatial_head:
                    predictions = torch.cat([
                        torch.sigmoid(raw["confidence_logit"]).unsqueeze(1),
                        raw["coord"],
                    ], dim=1).float().cpu().numpy()
                else:
                    predictions = raw.float().cpu().numpy()
        output = []
        for pred, (width, height) in zip(predictions, sizes):
            confidence = float(pred[0])
            pixel = (float(np.clip(pred[1], 0.0, 1.0) * width), float(np.clip(pred[2], 0.0, 1.0) * height))
            output.append(DetectionResult(confidence, pixel if confidence >= self.threshold else None,
                                          confidence >= self.threshold, (width, height)))
        return output

    def detect(self, item, warmup=False):
        if warmup and not self._warm: self.warmup()
        return self.detect_many([item], warmup=False)[0]
