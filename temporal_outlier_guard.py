"""Label-free temporal guard for coarse detector outliers.

The guard is deliberately separate from the detector checkpoint.  It is a
streaming heuristic: a large coarse jump is replaced by the previous accepted
point only when the current frame remains visually close to the previous one
and the detector's photometric TTA is unstable.  It must be evaluated on
ordered video frames; it should not be used on shuffled still images.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GuardResult:
    point: Optional[tuple[float, float]]
    replaced: bool
    reason: str
    jump_px: Optional[float] = None
    frame_diff: Optional[float] = None


class TemporalCoarseOutlierGuard:
    """Reject isolated coarse jumps without consulting a ground-truth label."""

    def __init__(
        self,
        *,
        max_jump_px: float = 20.0,
        max_frame_mean_abs_diff: float = 35.0,
        min_stability_px: float = 6.0,
        max_missed_frames: int = 0,
    ) -> None:
        values = (max_jump_px, max_frame_mean_abs_diff, min_stability_px)
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("guard thresholds must be finite and non-negative")
        self.max_jump_px = float(max_jump_px)
        self.max_frame_mean_abs_diff = float(max_frame_mean_abs_diff)
        self.min_stability_px = float(min_stability_px)
        self.max_missed_frames = max(0, int(max_missed_frames))
        self._last_point: Optional[np.ndarray] = None
        self._image: Optional[np.ndarray] = None
        self._missed = 0

    def reset(self) -> None:
        self._last_point = None
        self._image = None
        self._missed = 0

    @staticmethod
    def _array(image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
        else:
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[-1] >= 3:
                array = array[..., :3]
            array = np.asarray(array, dtype=np.float32)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("image must have shape HxWx3")
        return array

    @staticmethod
    def _point(value: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if value is None:
            return None
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(point) < 2 or not np.isfinite(point[:2]).all():
            return None
        return point[:2].copy()

    def update(
        self,
        image: Image.Image | np.ndarray,
        point: Optional[Sequence[float]],
        *,
        stability_px: Optional[float] = None,
    ) -> GuardResult:
        current_image = self._array(image)
        current = self._point(point)
        if current is None:
            self._missed += 1
            if self._missed > self.max_missed_frames:
                self.reset()
            return GuardResult(None, False, "missing_point")

        if self._last_point is None or self._image is None:
            self._last_point = current
            self._image = current_image
            self._missed = 0
            return GuardResult(tuple(current), False, "initialised")

        jump = float(np.linalg.norm(current - self._last_point))
        if current_image.shape != self._image.shape:
            frame_diff = float("inf")
        else:
            frame_diff = float(np.mean(np.abs(current_image - self._image)))
        stability = -float("inf") if stability_px is None else float(stability_px)
        replace = bool(
            jump >= self.max_jump_px
            and frame_diff <= self.max_frame_mean_abs_diff
            and stability >= self.min_stability_px
        )
        if replace:
            chosen = self._last_point.copy()
            reason = "isolated_jump_replaced"
        else:
            chosen = current
            reason = "accepted_current"
            self._last_point = current
        self._image = current_image
        self._missed = 0
        return GuardResult(tuple(chosen), replace, reason, jump, frame_diff)
