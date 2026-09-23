"""Low-cost temporal stabilisation for detections from a video stream."""
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np

@dataclass
class TemporalState:
    pixel: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False

class DetectionSmoother:
    def __init__(self, alpha=0.35, window=5, min_hits=3, max_misses=2, threshold=0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.min_hits = max(1, int(min_hits))
        self.max_misses = max(0, int(max_misses))
        self.threshold = float(threshold)
        self.history = deque(maxlen=max(1, int(window)))
        self.state = TemporalState()

    def reset(self):
        self.history.clear()
        self.state = TemporalState()

    def update(self, pixel: Optional[Sequence[float]], confidence: float) -> TemporalState:
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if pixel is not None and confidence >= self.threshold:
            point = np.asarray(pixel, dtype=np.float64).reshape(2)
            old = point if self.state.pixel is None else np.asarray(self.state.pixel)
            smoothed = point if self.state.pixel is None else self.alpha * point + (1.0 - self.alpha) * old
            self.history.append(smoothed)
            consensus = np.median(np.asarray(self.history), axis=0)
            self.state.pixel = (float(consensus[0]), float(consensus[1]))
            self.state.confidence = confidence if self.state.hits == 0 else max(confidence, self.state.confidence * 0.8)
            self.state.hits += 1
            self.state.misses = 0
            self.state.confirmed = self.state.hits >= self.min_hits
        else:
            self.state.misses += 1
            if self.state.misses > self.max_misses:
                self.reset()
        return TemporalState(**self.state.__dict__)
