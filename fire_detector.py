"""Fire detector adapter for legacy, spatial and improved 2D checkpoints.

The ``detector_2d_fpn_v2`` branch is intentionally exposed through this same
adapter so ROI and 3D code can consume the new detector without knowing which
2D architecture produced the coarse point.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
from contextlib import nullcontext
import math
import time
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageEnhance
from torchvision import transforms

@dataclass
class DetectionResult:
    """Stable detector result passed to ROI and 3-D stages.

    ``confidence`` is the fire/no-fire score.  It must not be interpreted as
    localisation reliability: the legacy checkpoint can be very confident
    while its point is an outlier. ``stability_px`` measures appearance-view
    disagreement, NOT calibrated position error. ``position_reliability``
    is an exponential heuristic, NOT a probability of being correct. An
    optional stability gate rejects localisation while retaining the fire
    alert and the predicted pixel for diagnostics. Unmeasured fields are None.
    """

    confidence: float
    pixel: Optional[tuple[float, float]]
    detected: bool
    size: tuple[int, int]
    latency_ms: float = 0.0
    stability_px: Optional[float] = None
    stability_reference_px: Optional[float] = None
    raw_pixel: Optional[tuple[float, float]] = None
    tta_count: int = 1
    tta_valid_count: int = 1
    tta_confidence: Optional[float] = None
    position_reliability: Optional[float] = None
    localization_accepted: bool = True
    localization_status: str = "not_checked"

    @property
    def p_fire(self) -> float:
        return self.confidence

    @property
    def u(self) -> Optional[int]:
        return None if self.pixel is None else int(self.pixel[0])

    @property
    def v(self) -> Optional[int]:
        return None if self.pixel is None else int(self.pixel[1])

    @property
    def localisation_reliability(self) -> Optional[float]:
        """Alias using the British spelling used by some pipeline scripts."""

        return self.position_reliability

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
                 threshold: float = 0.5, use_amp: bool = True,
                 tta_mode: str = "none", tta_reliability_scale_px: float = 8.0,
                 max_stability_px: Optional[float] = None,
                 stability_reference_size: float = 224.0):
        self.model_path = Path(model_path)
        if not self.model_path.exists(): raise FileNotFoundError(self.model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.tta_mode = self._normalise_tta_mode(tta_mode)
        self.tta_reliability_scale_px = float(tta_reliability_scale_px)
        self.stability_reference_size = float(stability_reference_size)
        if any(not math.isfinite(value) or value <= 0.0 for value in
               (self.tta_reliability_scale_px, self.stability_reference_size)):
            raise ValueError("Reliability scale and reference size must be finite and positive")
        self.max_stability_px = None if max_stability_px is None else float(max_stability_px)
        if self.max_stability_px is not None:
            if not math.isfinite(self.max_stability_px) or self.max_stability_px < 0.0:
                raise ValueError("max_stability_px must be finite and nonnegative")
            if self.tta_mode == "none":
                raise ValueError("A stability gate requires photometric TTA")
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
            # Checkpoints store (width, height); torchvision expects (height, width).
            transforms.Resize((self.image_size[1], self.image_size[0])), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._warm = False

    @staticmethod
    def _normalise_tta_mode(value: str) -> str:
        """Validate the small, deterministic photometric TTA vocabulary."""

        mode = str(value or "none").strip().lower().replace("-", "_")
        aliases = {
            "off": "none",
            "false": "none",
            "brightness": "brightness3",
            "brightness_3": "brightness3",
            "photometric": "photometric5",
            "photometric_5": "photometric5",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"none", "brightness3", "photometric5"}:
            raise ValueError(
                f"Unsupported tta_mode={value!r}; use 'none', 'brightness3', or 'photometric5'"
            )
        return mode

    def _tta_images(self, image: Image.Image) -> list[Image.Image]:
        """Create geometry-preserving appearance variants for one image.

        Brightness/contrast changes leave the pixel coordinate system intact,
        unlike flip TTA.  The three-view setting is the validation-selected
        default for the recommended real-time compromise.
        """

        image = image.convert("RGB")
        if self.tta_mode == "none":
            return [image]
        variants = [image]
        variants.extend(
            [
                ImageEnhance.Brightness(image).enhance(1.15),
                ImageEnhance.Brightness(image).enhance(0.85),
            ]
        )
        if self.tta_mode == "photometric5":
            variants.extend(
                [
                    ImageEnhance.Contrast(image).enhance(1.15),
                    ImageEnhance.Contrast(image).enhance(0.85),
                ]
            )
        return variants

    @staticmethod
    def _aggregate_tta(
        predictions: Sequence[tuple[float, Optional[tuple[float, float]]]],
        size: tuple[int, int],
        threshold: float,
        reliability_scale_px: float,
        max_stability_px: Optional[float],
        stability_reference_size: float,
        latency_ms: float,
    ) -> DetectionResult:
        """Aggregate appearance views and derive a point-stability signal."""

        if not predictions:
            return DetectionResult(
                confidence=0.0,
                pixel=None,
                detected=False,
                size=size,
                latency_ms=latency_ms,
                localization_accepted=False,
                localization_status="no_tta_prediction",
            )
        scores = np.asarray([float(item[0]) for item in predictions], dtype=np.float64)
        scores_valid = bool(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all())
        # Invalid model outputs must never become a high-confidence alert or
        # a zero-disagreement point from the one remaining valid view.
        confidence = float(scores.mean()) if scores_valid else 0.0
        points = np.asarray(
            [item[1] for item in predictions if item[1] is not None
             and np.asarray(item[1]).shape == (2,)
             and np.isfinite(item[1]).all()],
            dtype=np.float64,
        ).reshape(-1, 2)
        if len(points):
            centre = np.mean(points, axis=0)
            stability = float(np.mean(np.linalg.norm(points - centre, axis=1)))
            raw_point = (
                float(np.clip(centre[0], 0.0, max(0, size[0] - 1))),
                float(np.clip(centre[1], 0.0, max(0, size[1] - 1))),
            )
        else:
            stability = float("inf")
            raw_point = None
        size_scale = max(float(max(size)), 1.0) / max(float(stability_reference_size), 1e-6)
        effective_scale = max(float(reliability_scale_px) * size_scale, 1e-6)
        effective_gate = (
            None
            if max_stability_px is None
            else max(float(max_stability_px) * size_scale, 0.0)
        )
        reliability = (
            float(np.exp(-stability / effective_scale))
            if np.isfinite(stability)
            else 0.0
        )
        detected = bool(confidence >= float(threshold))
        complete = len(points) == len(predictions) and len(points) >= 2
        accepted = bool(
            detected
            and raw_point is not None
            and complete
            and (effective_gate is None or stability <= effective_gate)
        )
        if not scores_valid:
            status = "invalid_tta_confidence"
        elif not detected:
            status = "below_threshold"
        elif not complete:
            status = "incomplete_tta"
        else:
            status = "stable" if accepted else "unstable"
        return DetectionResult(
            confidence=confidence,
            pixel=raw_point if accepted else None,
            detected=detected,
            size=(int(size[0]), int(size[1])),
            latency_ms=float(latency_ms),
            stability_px=stability if np.isfinite(stability) else None,
            stability_reference_px=effective_gate,
            raw_pixel=raw_point,
            tta_count=len(predictions),
            tta_valid_count=len(points),
            tta_confidence=confidence,
            position_reliability=reliability,
            localization_accepted=accepted,
            localization_status=status,
        )

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
            # The improved detector already batches its own inputs.  TTA is
            # intentionally implemented in this adapter as a separate,
            # opt-in path so the old checkpoint and all existing call sites
            # retain their exact behaviour by default.
            if self.tta_mode != "none":
                return self._detect_many_tta(items, warmup=warmup)
            results = self._detector_2d.detect_many(items, warmup=warmup)
            self._warm = self._detector_2d._warm
            return [
                DetectionResult(
                    confidence=result.confidence,
                    pixel=result.pixel,
                    detected=result.detected,
                    size=result.size,
                    latency_ms=result.latency_ms,
                    stability_px=getattr(result, "stability_px", None),
                    stability_reference_px=getattr(result, "stability_reference_px", None),
                    raw_pixel=getattr(result, "raw_pixel", result.pixel),
                    tta_count=int(getattr(result, "tta_count", 1)),
                    tta_valid_count=int(getattr(result, "tta_valid_count", 1)),
                    tta_confidence=getattr(result, "tta_confidence", None),
                    position_reliability=getattr(result, "position_reliability", None),
                    localization_accepted=bool(result.detected and result.pixel is not None
                                               and getattr(result, "localization_accepted", True)),
                    localization_status=str(getattr(result, "localization_status", "not_checked")),
                )
                for result in results
            ]
        if warmup and not self._warm: self.warmup()
        if self.tta_mode != "none":
            return self._detect_many_tta(items, warmup=False)
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
            valid = bool(np.isfinite(pred).all())
            confidence = float(pred[0]) if valid else 0.0
            detected = valid and confidence >= self.threshold
            pixel = (float(np.clip(pred[1] * width, 0.0, width - 1)),
                     float(np.clip(pred[2] * height, 0.0, height - 1))) if valid else None
            output.append(
                DetectionResult(
                    confidence=confidence,
                    pixel=pixel if detected else None,
                    detected=detected,
                    size=(width, height),
                    raw_pixel=pixel if detected else None,
                    localization_accepted=bool(detected and pixel is not None),
                    localization_status=("not_checked" if detected else
                                         "below_threshold" if valid else "invalid_prediction"),
                )
            )
        return output

    def _detect_many_tta(self, items: Sequence, warmup: bool = False) -> list[DetectionResult]:
        """Run photometric TTA without changing the detector's public API.

        We process variants in one model batch per view.  This is faster than
        invoking ``detect`` repeatedly and keeps the point aggregation
        independent of image sizes.  The method is deliberately private so a
        caller can use it simply by constructing ``FireDetector(...,
        tta_mode='brightness3')``.
        """

        if not items:
            return []
        if warmup and not self._warm:
            self.warmup(repeats=1)
        images = [self._to_image(item) for item in items]
        variants = [self._tta_images(image) for image in images]
        view_count = len(variants[0])
        if any(len(value) != view_count for value in variants):
            raise RuntimeError("TTA variants must have the same count for all inputs")
        sizes = [image.size for image in images]
        predictions: list[list[tuple[float, Optional[tuple[float, float]]]]] = [
            [] for _ in images
        ]
        start = time.perf_counter()
        if self._detector_2d is not None:
            # Detector2DInference already accepts a batch. Flatten all image
            # and TTA dimensions into one call instead of launching one FPN
            # forward per appearance view.  The original index mapping is
            # deterministic and preserves each image's coordinate system.
            flattened = [
                variants[item_index][view_index]
                for item_index in range(len(images))
                for view_index in range(view_count)
            ]
            flat_results = self._detector_2d.detect_many(flattened, warmup=False)
            for item_index in range(len(images)):
                for view_index in range(view_count):
                    result = flat_results[item_index * view_count + view_index]
                    predictions[item_index].append((
                        float(result.confidence),
                        None if result.pixel is None else tuple(float(value) for value in result.pixel),
                    ))
        else:
            # The legacy checkpoint uses a fixed resize transform. Flattening
            # the views into one batch removes Python/model-launch overhead
            # and is substantially faster for CPU real-time streams.
            flattened = [variant for item_variants in variants for variant in item_variants]
            tensor = torch.stack([self.transform(image) for image in flattened]).to(self.device)
            with torch.inference_mode():
                context = torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
                with context:
                    raw = self.model(tensor)
                    if self.spatial_head:
                        view_predictions = torch.cat(
                            [
                                torch.sigmoid(raw["confidence_logit"]).unsqueeze(1),
                                raw["coord"],
                            ],
                            dim=1,
                        ).float().cpu().numpy()
                    else:
                        view_predictions = raw.float().cpu().numpy()
            for item_index, item_variants in enumerate(variants):
                for view_index in range(view_count):
                    pred = view_predictions[item_index * view_count + view_index]
                    confidence = float(pred[0])
                    width, height = sizes[item_index]
                    point = (
                        float(np.clip(pred[1], 0.0, 1.0) * width),
                        float(np.clip(pred[2], 0.0, 1.0) * height),
                    )
                    predictions[item_index].append((confidence, point))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._warm = True
        return [
            self._aggregate_tta(
                item_predictions,
                size,
                self.threshold,
                self.tta_reliability_scale_px,
                self.max_stability_px,
                self.stability_reference_size,
                elapsed_ms / max(1, len(images)),
            )
            for item_predictions, size in zip(predictions, sizes)
        ]

    def detect(self, item, warmup=False):
        if warmup and not self._warm: self.warmup()
        return self.detect_many([item], warmup=False)[0]
