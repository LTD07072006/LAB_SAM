"""YOLO-to-2D/3D adapter for the independent Home Fire branch.

The adapter keeps detection and geometry separate: YOLO selects a bbox, then
the box is converted to a bottom-center or bottom-band contact hypothesis.
The result implements the repository's ``Detection2D`` contract and can be
passed to the existing multi-ray localisation code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from detector_adapter import Detection2D


@dataclass(frozen=True)
class YOLODetection:
    """One YOLO detection in the original image coordinate system.

    ``Ultralytics`` may resize/letterbox internally, but ``boxes.xyxy`` is
    returned in the source-image coordinate system.  Keeping that convention
    here is important: calibration and ray casting must receive original
    pixels, not 640x640 model-space pixels.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    bottom_center: tuple[float, float]
    bottom_band: np.ndarray


def _as_xyxy(value: Any) -> Optional[tuple[float, float, float, float]]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) < 4 or not np.all(np.isfinite(array[:4])):
        return None
    return tuple(float(item) for item in array[:4])


def bottom_band_points(
    bbox: Sequence[float],
    columns: int = 5,
    vertical_offsets: Sequence[float] = (0.05, 0.10, 0.15),
    image_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Return several pixels near the bbox base in original-image pixels.

    ``image_size`` is optional so the helper remains useful for generic
    geometry tests.  The YOLO adapter supplies it in production to guarantee
    that every contact hypothesis is a valid source-image pixel before
    calibration and ray casting.
    """
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64).reshape(4)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    xs = np.linspace(x1 + 0.20 * width, x2 - 0.20 * width, max(2, int(columns)))
    points = []
    for offset in vertical_offsets:
        points.extend((float(x), float(y2 - float(offset) * height)) for x in xs)
    result = np.asarray(points, dtype=np.float64)
    if image_size is not None and len(result):
        image_width, image_height = (int(image_size[0]), int(image_size[1]))
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_size must contain positive width and height")
        result[:, 0] = np.clip(result[:, 0], 0.0, float(image_width - 1))
        result[:, 1] = np.clip(result[:, 1], 0.0, float(image_height - 1))
    return result


def _image_size(image: Any) -> tuple[int, int]:
    if hasattr(image, "size") and not isinstance(image, np.ndarray):
        size = image.size
        return int(size[0]), int(size[1])
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("image must have height and width")
    return int(array.shape[1]), int(array.shape[0])


def _tensor_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor/array-like object without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class HomeFireYOLO:
    """Lazy Ultralytics wrapper returning a stable ``Detection2D`` object."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        fire_class: Optional[int] = None,
        threshold: float = 0.35,
        device: Optional[str] = None,
        imgsz: int = 640,
        contact_columns: int = 5,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.fire_class = fire_class
        self.threshold = float(threshold)
        self.device = device
        self.imgsz = int(imgsz)
        self.contact_columns = max(2, int(contact_columns))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Chưa cài Ultralytics. Trong .venv chạy: python -m pip install ultralytics"
                ) from exc
            self._model = YOLO(str(self.checkpoint))
        return self._model

    def _predict(self, image: Any):
        kwargs = {"source": image, "conf": self.threshold, "imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        return self.model.predict(**kwargs)[0]

    def detect_all(self, image: Any) -> list[YOLODetection]:
        """Return every prediction belonging to ``fire_class``.

        The 3D stage can later select one target, or aggregate several
        targets.  Returning all boxes here also makes bbox metrics meaningful
        for images containing more than one fire region.
        """
        result = self._predict(image)
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _tensor_numpy(boxes.xyxy)
        scores = _tensor_numpy(boxes.conf).reshape(-1)
        classes = _tensor_numpy(boxes.cls).astype(int).reshape(-1)
        width, height = _image_size(image)
        detections: list[YOLODetection] = []
        for box, score, class_id in zip(xyxy, scores, classes):
            score = float(score)
            class_id = int(class_id)
            if self.fire_class is not None and class_id != self.fire_class:
                continue
            if score < self.threshold:
                continue
            xyxy_box = _as_xyxy(box)
            if xyxy_box is None:
                continue
            x1, y1, x2, y2 = xyxy_box
            # Ultralytics normally returns source-image coordinates, but
            # clipping here protects calibration from malformed outputs and
            # from small floating-point/letterbox boundary overshoots.
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            x1 = float(np.clip(x1, 0.0, width - 1.0))
            x2 = float(np.clip(x2, 0.0, width - 1.0))
            y1 = float(np.clip(y1, 0.0, height - 1.0))
            y2 = float(np.clip(y2, 0.0, height - 1.0))
            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 1.0:
                continue
            xyxy_box = (x1, y1, x2, y2)
            point = (float((x1 + x2) * 0.5), float(y2))
            detections.append(
                YOLODetection(
                    bbox=xyxy_box,
                    confidence=score,
                    class_id=class_id,
                    bottom_center=point,
                    bottom_band=bottom_band_points(
                        xyxy_box,
                        columns=self.contact_columns,
                        image_size=(width, height),
                    ),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def detect(self, image: Any) -> Detection2D:
        width, height = _image_size(image)
        started = time.perf_counter()
        candidates = self.detect_all(image)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not candidates:
            return Detection2D(False, 0.0, None, None, None, (width, height), elapsed, "home_fire_yolo")
        selected = candidates[0]
        return Detection2D(
            detected=True,
            confidence=selected.confidence,
            point=selected.bottom_center,
            bbox=selected.bbox,
            mask=None,
            image_size=(width, height),
            latency_ms=elapsed,
            source=f"home_fire_yolo_class_{selected.class_id}",
            contact_candidates=selected.bottom_band,
        )

    def contact_pixels(self, image: Any) -> np.ndarray:
        """Predict once and return the selected bbox's bottom-band candidates."""
        detection = self.detect(image)
        if not detection.detected or detection.bbox is None:
            return np.empty((0, 2), dtype=np.float64)
        width, height = _image_size(image)
        return bottom_band_points(
            detection.bbox,
            columns=self.contact_columns,
            image_size=(width, height),
        )
