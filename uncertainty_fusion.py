"""Uncertainty-aware fusion of independent 3D localisation candidates.

Ray Casting remains the preferred source.  IPM, GPR residual correction and
monocular depth are auxiliary hypotheses; a candidate that disagrees strongly
with a reliable ray result is rejected rather than averaged into a bad point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass
class WorldCandidate:
    method: str
    point: Optional[np.ndarray]
    std_m: Optional[np.ndarray] = None
    confidence: float = 0.0
    available: bool = True
    prior: float = 1.0
    reason: Optional[str] = None

    def normalized(self) -> "WorldCandidate":
        point = None if self.point is None else np.asarray(self.point, dtype=np.float64).reshape(-1)
        if point is not None and (len(point) < 3 or not np.all(np.isfinite(point[:3]))):
            point = None
        std = None if self.std_m is None else np.asarray(self.std_m, dtype=np.float64).reshape(-1)
        if std is None or len(std) < 3 or not np.all(np.isfinite(std[:3])):
            std = np.full(3, 0.25, dtype=np.float64)
        std = np.maximum(std[:3], 1e-3)
        return WorldCandidate(
            self.method,
            None if point is None else point[:3].copy(),
            std,
            float(np.clip(self.confidence, 0.0, 1.0)),
            bool(self.available and point is not None),
            float(max(1e-6, self.prior)),
            self.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        normalized = self.normalized()
        return {
            "method": normalized.method,
            "point": None if normalized.point is None else normalized.point.tolist(),
            "std_m": None if normalized.std_m is None else normalized.std_m.tolist(),
            "confidence": normalized.confidence,
            "available": normalized.available,
            "prior": normalized.prior,
            "reason": normalized.reason,
        }


@dataclass
class FusionResult:
    point: Optional[np.ndarray]
    std_m: Optional[np.ndarray]
    confidence: float
    success: bool
    method: str = "fusion"
    used_methods: tuple[str, ...] = ()
    rejected_methods: tuple[str, ...] = ()
    weights: Optional[dict[str, float]] = None
    fallback: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": None if self.point is None else np.asarray(self.point).tolist(),
            "std_m": None if self.std_m is None else np.asarray(self.std_m).tolist(),
            "confidence": float(self.confidence),
            "success": bool(self.success),
            "method": self.method,
            "used_methods": list(self.used_methods),
            "rejected_methods": list(self.rejected_methods),
            "weights": self.weights or {},
            "fallback": bool(self.fallback),
            "reason": self.reason,
        }


DEFAULT_PRIORS = {
    "ray": 1.00,
    "ray_casting": 1.00,
    "ipm": 0.65,
    "gpr": 0.55,
    "gpr_residual": 0.55,
    "depth": 0.25,
    "monocular_depth": 0.25,
}


def _prior(method: str, overrides: Optional[dict[str, float]]) -> float:
    if overrides and method in overrides:
        return float(max(1e-6, overrides[method]))
    return float(DEFAULT_PRIORS.get(method, 0.5))


def fuse_world_candidates(
    candidates: Iterable[WorldCandidate],
    ray_gate_m: float = 1.0,
    max_uncertainty_m: float = 4.0,
    priors: Optional[dict[str, float]] = None,
) -> FusionResult:
    """Fuse candidates while protecting a valid Ray Casting estimate.

    Weights are ``prior * confidence / variance``.  If a ray candidate exists,
    non-ray candidates farther than ``ray_gate_m`` from it are rejected.  If
    ray is unavailable, the most confident remaining candidate becomes the
    reference and the same gate is relaxed to the available uncertainty.
    """

    normalized = [candidate.normalized() for candidate in candidates]
    available = [candidate for candidate in normalized if candidate.available and candidate.point is not None]
    if not available:
        return FusionResult(None, None, 0.0, False, reason="no_available_candidates")
    ray = next((candidate for candidate in available if candidate.method in {"ray", "ray_casting"}), None)
    reference = ray or max(available, key=lambda item: item.confidence * item.prior)
    rejected: list[str] = []
    accepted: list[WorldCandidate] = []
    for candidate in available:
        distance = float(np.linalg.norm(candidate.point - reference.point))
        tolerance = float(ray_gate_m)
        if candidate is not reference:
            uncertainty = float(np.linalg.norm(reference.std_m) + np.linalg.norm(candidate.std_m))
            tolerance = max(tolerance, 2.0 * uncertainty)
        if distance > tolerance and candidate is not reference:
            rejected.append(candidate.method)
        else:
            accepted.append(candidate)
    if not accepted:
        accepted = [reference]
    weights: dict[str, float] = {}
    raw_weights: list[float] = []
    for candidate in accepted:
        variance = float(np.mean(np.maximum(candidate.std_m, 1e-3) ** 2))
        value = _prior(candidate.method, priors) * max(candidate.confidence, 0.05) / max(variance, 1e-6)
        if candidate is ray:
            value *= 2.0
        raw_weights.append(value)
    raw = np.asarray(raw_weights, dtype=np.float64)
    raw /= max(float(raw.sum()), 1e-12)
    for candidate, weight in zip(accepted, raw):
        weights[candidate.method] = float(weight)
    points = np.asarray([candidate.point for candidate in accepted], dtype=np.float64)
    fused = np.sum(points * raw[:, None], axis=0)
    dispersion = np.sqrt(np.sum(raw[:, None] * (points - fused) ** 2, axis=0))
    uncertainty = np.sqrt(
        np.sum(
            np.asarray([(weight * candidate.std_m) ** 2 for candidate, weight in zip(accepted, raw)], dtype=np.float64),
            axis=0,
        )
    )
    std = np.maximum(dispersion + uncertainty, 1e-4)
    std = np.minimum(std, float(max_uncertainty_m))
    confidence = float(np.clip(np.sum(raw * np.asarray([item.confidence for item in accepted])) * np.exp(-float(np.mean(dispersion))), 0.0, 1.0))
    fallback = ray is None or len(accepted) == 1
    reason = "ray_protected_fusion" if ray is not None else "best_available_candidate"
    return FusionResult(
        fused,
        std,
        confidence,
        True,
        used_methods=tuple(item.method for item in accepted),
        rejected_methods=tuple(rejected),
        weights=weights,
        fallback=fallback,
        reason=reason,
    )


__all__ = ["FusionResult", "WorldCandidate", "fuse_world_candidates"]
