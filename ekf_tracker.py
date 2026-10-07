"""Constant-velocity 3D Kalman filter for fire-location sequences.

The existing :mod:`tracking_3d` module is an EMA-style robust smoother.  This
module adds a conventional 6-state Kalman filter with Mahalanobis innovation
gating.  It is useful when the localisation stage also provides a per-frame
spread/covariance estimate from multi-ray intersection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EKFState:
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    confidence: float = 0.0
    age: int = 0
    missed: int = 0
    accepted: bool = False
    mahalanobis2: Optional[float] = None


def _copy_state(state: EKFState) -> EKFState:
    return EKFState(
        position=None if state.position is None else state.position.copy(),
        velocity=None if state.velocity is None else state.velocity.copy(),
        covariance=None if state.covariance is None else state.covariance.copy(),
        confidence=float(state.confidence),
        age=int(state.age),
        missed=int(state.missed),
        accepted=bool(state.accepted),
        mahalanobis2=None if state.mahalanobis2 is None else float(state.mahalanobis2),
    )


class Fire3DEKF:
    """6D constant-velocity Kalman filter with outlier rejection."""

    def __init__(
        self,
        dt: float = 1.0,
        process_accel_std: float = 0.35,
        measurement_std_m: float = 0.25,
        gate_mahalanobis2: float = 16.27,
        max_missed: int = 3,
    ) -> None:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.process_accel_std = float(max(1e-6, process_accel_std))
        self.measurement_std_m = float(max(1e-6, measurement_std_m))
        self.gate_mahalanobis2 = float(max(0.0, gate_mahalanobis2))
        self.max_missed = int(max(0, max_missed))
        self.state = EKFState()

        dt2 = self.dt * self.dt
        self.F = np.eye(6, dtype=np.float64)
        self.F[:3, 3:] = np.eye(3) * self.dt
        q = self.process_accel_std**2
        self.Q = q * np.block(
            [
                [np.eye(3) * (dt2 * dt2 / 4.0), np.eye(3) * (dt2 * self.dt / 2.0)],
                [np.eye(3) * (dt2 * self.dt / 2.0), np.eye(3) * dt2],
            ]
        )
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:, :3] = np.eye(3)

    def reset(self) -> None:
        self.state = EKFState()

    def _predict_in_place(self) -> None:
        if self.state.position is None or self.state.velocity is None:
            return
        x = np.concatenate((self.state.position, self.state.velocity))
        covariance = self.state.covariance
        if covariance is None:
            covariance = np.eye(6, dtype=np.float64)
        x = self.F @ x
        covariance = self.F @ covariance @ self.F.T + self.Q
        covariance = 0.5 * (covariance + covariance.T)
        self.state.position = x[:3]
        self.state.velocity = x[3:]
        self.state.covariance = covariance

    def _measurement_covariance(self, covariance: Optional[np.ndarray]) -> np.ndarray:
        if covariance is None:
            return np.eye(3, dtype=np.float64) * self.measurement_std_m**2
        value = np.asarray(covariance, dtype=np.float64)
        if value.shape == (3,):
            value = np.diag(np.maximum(value, 1e-6) ** 2)
        else:
            value = value.reshape(3, 3)
        if not np.all(np.isfinite(value)):
            return np.eye(3, dtype=np.float64) * self.measurement_std_m**2
        value = 0.5 * (value + value.T)
        value += np.eye(3, dtype=np.float64) * 1e-6
        return value

    def update(
        self,
        point=None,
        covariance: Optional[np.ndarray] = None,
        confidence: float = 0.0,
    ) -> EKFState:
        """Predict, gate and update one 3D measurement.

        A missing or rejected point advances the prediction but does not
        correct it.  The returned state is a copy safe to store in JSON.
        """

        if self.state.position is None:
            if point is None:
                self.state.accepted = False
                self.state.missed += 1
                return _copy_state(self.state)
            measurement = np.asarray(point, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(measurement)):
                return self.update(None, covariance, confidence)
            measurement_cov = self._measurement_covariance(covariance)
            self.state.position = measurement.copy()
            self.state.velocity = np.zeros(3, dtype=np.float64)
            self.state.covariance = np.zeros((6, 6), dtype=np.float64)
            self.state.covariance[:3, :3] = measurement_cov
            self.state.covariance[3:, 3:] = np.eye(3, dtype=np.float64)
            self.state.confidence = float(np.clip(confidence, 0.0, 1.0))
            self.state.age = 1
            self.state.missed = 0
            self.state.accepted = True
            self.state.mahalanobis2 = 0.0
            return _copy_state(self.state)

        self._predict_in_place()
        self.state.mahalanobis2 = None
        if point is None:
            self.state.missed += 1
            self.state.accepted = False
            if self.state.missed > self.max_missed:
                self.reset()
            return _copy_state(self.state)

        measurement = np.asarray(point, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(measurement)):
            return self.update(None, covariance, confidence)
        measurement_cov = self._measurement_covariance(covariance)
        x = np.concatenate((self.state.position, self.state.velocity))
        P = self.state.covariance if self.state.covariance is not None else np.eye(6)
        innovation = measurement - self.H @ x
        S = self.H @ P @ self.H.T + measurement_cov
        S = 0.5 * (S + S.T)
        try:
            solved = np.linalg.solve(S, innovation)
            d2 = float(innovation @ solved)
        except np.linalg.LinAlgError:
            d2 = float("inf")
        self.state.mahalanobis2 = d2
        if d2 > self.gate_mahalanobis2:
            self.state.missed += 1
            self.state.accepted = False
            if self.state.missed > self.max_missed:
                self.reset()
            return _copy_state(self.state)

        try:
            gain = P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            gain = P @ self.H.T @ np.linalg.pinv(S)
        x = x + gain @ innovation
        identity = np.eye(6, dtype=np.float64)
        # Joseph form is more stable for small/ill-conditioned covariances.
        residual = identity - gain @ self.H
        P = residual @ P @ residual.T + gain @ measurement_cov @ gain.T
        self.state.position = x[:3]
        self.state.velocity = x[3:]
        self.state.covariance = 0.5 * (P + P.T)
        self.state.confidence = max(
            float(np.clip(confidence, 0.0, 1.0)),
            self.state.confidence * 0.9,
        )
        self.state.age += 1
        self.state.missed = 0
        self.state.accepted = True
        return _copy_state(self.state)

