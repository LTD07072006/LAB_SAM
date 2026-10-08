"""Optional monocular-depth adapter for the post-ROI 3D benchmark.

Depth models are deliberately isolated from the calibrated Ray Casting path.
The adapter can consume precomputed ``.npy``/image depth maps immediately and
reports ``unavailable`` when a foundation model has not been installed.  This
prevents a missing optional dependency from silently changing the primary
geometry result.

Supported sources:

* ``none``: explicit unavailable result;
* ``maps``: precomputed maps under ``map_root``; filenames are matched by
  sample id or image stem;
* ``transformers``: an optional Hugging Face depth-estimation pipeline.  The
  caller must install ``transformers`` and provide a model name.  Its output
  is relative depth unless the selected model documents metric depth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


@dataclass
class DepthPrediction:
    depth_map: Optional[np.ndarray]
    confidence: float
    units: str
    backend: str
    success: bool
    reason: Optional[str] = None
    source_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": None if self.depth_map is None else list(self.depth_map.shape),
            "confidence": float(self.confidence),
            "units": self.units,
            "backend": self.backend,
            "success": bool(self.success),
            "reason": self.reason,
            "source_path": self.source_path,
        }


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if not archive.files:
            raise ValueError("NPZ depth archive is empty")
        value = archive[archive.files[0]]
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        value = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    elif suffix == ".json":
        value = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported depth-map extension: {path.suffix}")
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2 or not np.any(np.isfinite(value)):
        raise ValueError(f"Depth map must be a finite 2D array: {path}")
    return value


class MonocularDepthAdapter:
    """Unified, optional interface for depth maps and depth-estimation models."""

    def __init__(
        self,
        backend: str = "none",
        map_root: Optional[Path | str] = None,
        model_name: Optional[str] = None,
        units: str = "relative",
        cache_size: int = 8,
    ) -> None:
        self.backend = str(backend).strip().lower()
        if self.backend not in {"none", "maps", "transformers"}:
            raise ValueError("backend must be none, maps, or transformers")
        self.map_root = None if map_root is None else Path(map_root).expanduser().resolve()
        self.model_name = model_name
        self.units = str(units).strip().lower()
        if self.units not in {"relative", "camera_z", "ray_range"}:
            raise ValueError("units must be relative, camera_z, or ray_range")
        self.cache_size = max(0, int(cache_size))
        self._cache: dict[str, DepthPrediction] = {}
        self._pipeline: Any = None
        self._init_error: Optional[str] = None
        if self.backend == "maps" and (self.map_root is None or not self.map_root.is_dir()):
            self._init_error = f"depth map root does not exist: {self.map_root}"

    @property
    def available(self) -> bool:
        if self.backend == "none":
            return False
        if self.backend == "maps":
            return self._init_error is None
        return self._load_transformers_pipeline() is not None

    def _load_transformers_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self._init_error is not None:
            return None
        if not self.model_name:
            self._init_error = "transformers backend needs --depth-model"
            return None
        try:
            from transformers import pipeline

            self._pipeline = pipeline("depth-estimation", model=self.model_name)
            return self._pipeline
        except Exception as exc:
            self._init_error = f"could not load transformers depth model: {exc}"
            return None

    @staticmethod
    def _candidate_names(image_path: Path, sample_id: Optional[str]) -> list[str]:
        names: list[str] = []
        if sample_id:
            names.extend([sample_id, Path(sample_id).stem])
        names.extend([image_path.name, image_path.stem])
        return list(dict.fromkeys(names))

    def _find_map(self, image_path: Path, sample_id: Optional[str]) -> Optional[Path]:
        if self.map_root is None:
            return None
        candidates = self._candidate_names(image_path, sample_id)
        extensions = (".npy", ".npz", ".png", ".tif", ".tiff", ".json")
        for name in candidates:
            path = self.map_root / name
            if path.is_file():
                return path
            for extension in extensions:
                path = self.map_root / f"{name}{extension}"
                if path.is_file():
                    return path
        for name in candidates:
            for extension in extensions:
                matches = list(self.map_root.rglob(f"{name}{extension}"))
                if matches:
                    return matches[0]
        return None

    def predict(
        self,
        image: Image.Image | np.ndarray,
        image_path: Optional[Path | str] = None,
        sample_id: Optional[str] = None,
    ) -> DepthPrediction:
        """Return a depth map for one image without changing its geometry."""

        key = str(sample_id or image_path or "memory")
        if key in self._cache:
            return self._cache[key]
        path = None if image_path is None else Path(image_path)
        try:
            if self.backend == "none":
                result = DepthPrediction(None, 0.0, self.units, self.backend, False, "backend_disabled")
            elif self.backend == "maps":
                map_path = self._find_map(path or Path("image"), sample_id)
                if map_path is None:
                    result = DepthPrediction(None, 0.0, self.units, self.backend, False, "depth_map_not_found")
                else:
                    result = DepthPrediction(
                        _load_array(map_path),
                        1.0,
                        self.units,
                        self.backend,
                        True,
                        source_path=str(map_path),
                    )
            else:
                depth_pipeline = self._load_transformers_pipeline()
                if depth_pipeline is None:
                    result = DepthPrediction(None, 0.0, self.units, self.backend, False, self._init_error)
                else:
                    source = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8))
                    raw = depth_pipeline(source)
                    depth = raw.get("depth") if isinstance(raw, dict) else None
                    if depth is None:
                        raise ValueError("transformers result has no depth field")
                    result = DepthPrediction(
                        np.asarray(depth, dtype=np.float32),
                        0.5,
                        self.units,
                        self.backend,
                        True,
                        source_path=None if path is None else str(path),
                    )
        except Exception as exc:
            result = DepthPrediction(None, 0.0, self.units, self.backend, False, str(exc))
        if self.cache_size:
            self._cache[key] = result
            while len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))
        return result


__all__ = ["DepthPrediction", "MonocularDepthAdapter"]
