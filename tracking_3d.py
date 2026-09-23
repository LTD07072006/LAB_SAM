"""Lightweight 3D temporal filter with outlier gating."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Track3DState:
    point: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    confidence: float = 0.0
    age: int = 0
    missed: int = 0
    accepted: bool = False


class Fire3DTracker:
    """Constant-velocity exponential tracker with innovation gating."""

    def __init__(self, alpha: float = 0.35, gate_m: float = 3.0, max_missed: int = 3):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.gate_m = float(max(0.0, gate_m))
        self.max_missed = int(max(0, max_missed))
        self.state = Track3DState()

    def reset(self):
        self.state = Track3DState()

    def update(self, point=None, covariance=None, confidence: float = 0.0) -> Track3DState:
        if point is None:
            self.state.missed += 1
            self.state.accepted = False
            if self.state.missed > self.max_missed:
                self.reset()
            return Track3DState(**self.state.__dict__)

        measurement = np.asarray(point, dtype=np.float64).reshape(3)
        measurement_cov = np.asarray(covariance, dtype=np.float64).reshape(3, 3) if covariance is not None else np.eye(3) * 0.25
        if self.state.point is None:
            self.state.point = measurement.copy()
            self.state.velocity = np.zeros(3, dtype=np.float64)
            self.state.covariance = measurement_cov
            self.state.confidence = float(confidence)
            self.state.age = 1
            self.state.missed = 0
            self.state.accepted = True
            return Track3DState(**self.state.__dict__)

        predicted = self.state.point + self.state.velocity
        innovation = measurement - predicted
        innovation_norm = float(np.linalg.norm(innovation))
        if innovation_norm > self.gate_m:
            self.state.missed += 1
            self.state.accepted = False
            if self.state.missed > self.max_missed:
                self.reset()
            return Track3DState(**self.state.__dict__)

        old_point = self.state.point.copy()
        self.state.point = self.alpha * measurement + (1.0 - self.alpha) * predicted
        self.state.velocity = self.state.point - old_point
        self.state.covariance = self.alpha * measurement_cov + (1.0 - self.alpha) * self.state.covariance
        self.state.confidence = max(float(confidence), self.state.confidence * (1.0 - 0.25 * self.alpha))
        self.state.age += 1
        self.state.missed = 0
        self.state.accepted = True
        return Track3DState(**self.state.__dict__)
