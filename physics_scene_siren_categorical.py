"""SIREN and categorical-depth scene coordinates for the post-ROI branch.

This module is an additive experiment for the existing physics-informed
Scene-MLP code.  It keeps the same input contract as
``physics_scene_coordinate_v2.py`` and changes only the learnable head:

* ``scalar`` depth: a conventional scalar log-depth head;
* ``categorical`` depth: a CaDDN-inspired categorical distribution over depth
  bins, with the expected depth used by the pinhole physical layer;
* ``relu`` or ``siren`` hidden backbone.

The physical layer remains explicit:

    X_world = C + R_world_to_camera.T @ ([ray_x, ray_y, 1] * depth)

This is not a reimplementation of the full CaDDN detector.  CaDDN predicts
per-pixel depth distributions and uses them to construct a frustum/BEV
representation for autonomous-driving 3D detection.  Here the same useful
idea is adapted to one post-ROI fire contact point with known camera metadata.
The implementation is deliberately separate so existing checkpoints remain
reproducible.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from physics_scene_coordinate_v2 import (
    PhysicsSceneV2Sample,
    V2_FEATURE_NAMES,
    build_samples,
    metric_summary,
)


class SineLayer:
    """Small wrapper kept for documentation; the real layer is made in Torch."""


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


class PhysicsSceneSirenCategorical:
    """Physics-constrained scene model with optional SIREN and depth bins.

    Parameters are intentionally explicit so checkpoints can be reconstructed
    without importing the old model class.  ``depth_mode`` is either
    ``scalar`` or ``categorical``; ``backbone`` is either ``relu`` or
    ``siren``.  The ``relu + scalar`` configuration is the control ablation.
    """

    FORMAT = "LAB_SAM.physics_scene_siren_categorical.v1"
    DELTA_LIMIT = 0.35

    def __init__(
        self,
        input_dim: int = len(V2_FEATURE_NAMES),
        hidden_dim: int = 160,
        hidden_layers: int = 3,
        backbone: str = "siren",
        depth_mode: str = "categorical",
        depth_bins: int = 80,
        depth_min_m: float = 6.0,
        depth_max_m: float = 20.0,
        omega_0: float = 20.0,
        device: str = "cpu",
    ) -> None:
        if backbone not in {"relu", "siren"}:
            raise ValueError("backbone must be 'relu' or 'siren'")
        if depth_mode not in {"scalar", "categorical"}:
            raise ValueError("depth_mode must be 'scalar' or 'categorical'")
        if int(hidden_dim) <= 0 or int(hidden_layers) <= 0:
            raise ValueError("hidden_dim and hidden_layers must be positive")
        if int(depth_bins) < 2:
            raise ValueError("depth_bins must be at least 2")
        if float(depth_min_m) <= 0 or float(depth_max_m) <= float(depth_min_m):
            raise ValueError("invalid depth range")

        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PhysicsSceneSirenCategorical requires PyTorch") from exc

        self.torch = torch
        self.nn = nn
        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.backbone = str(backbone)
        self.depth_mode = str(depth_mode)
        self.depth_bins = int(depth_bins)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.omega_0 = float(omega_0)

        if self.depth_mode == "categorical":
            centers = np.linspace(
                self.depth_min_m,
                self.depth_max_m,
                self.depth_bins,
                dtype=np.float32,
            )
            self.depth_centers = torch.from_numpy(centers).to(self.device)
            output_dim = self.depth_bins + 5  # logits + ray correction + logvar
        else:
            self.depth_centers = torch.empty(0, device=self.device)
            output_dim = 6  # scalar log-depth + ray correction + logvar

        layers: list[Any] = []
        if self.backbone == "siren":
            layers.append(self._make_sine_layer(self.input_dim, self.hidden_dim, True))
            for _ in range(self.hidden_layers - 1):
                layers.append(self._make_sine_layer(self.hidden_dim, self.hidden_dim, False))
        else:
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            # Literal ReLU control for the SIREN ablation.  The previous v2
            # branch uses GELU, but this new control must match the ReLU
            # baseline described by the SIREN reference.
            layers.append(nn.ReLU())
            for _ in range(self.hidden_layers - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
        self.backbone_net = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden_dim, output_dim)
        self.model = nn.Sequential(self.backbone_net, self.head).to(self.device)

        self.feature_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.feature_scale = np.ones(self.input_dim, dtype=np.float32)
        self.log_depth_mean = 0.0
        self.log_depth_scale = 1.0
        self.logvar_min = -8.0
        self.logvar_max = 4.0
        self.history: list[dict[str, float]] = []

    def _make_sine_layer(self, in_dim: int, out_dim: int, is_first: bool) -> Any:
        """Create a SIREN layer with the initialization from the paper."""

        torch = self.torch
        nn = self.nn
        omega_0 = float(self.omega_0)

        class _Sine(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
                self.omega_0 = omega_0
                with torch.no_grad():
                    if is_first:
                        bound = 1.0 / max(1, in_dim)
                    else:
                        bound = float(np.sqrt(6.0 / max(1, in_dim)) / omega_0)
                    self.linear.weight.uniform_(-bound, bound)
                    if self.linear.bias is not None:
                        self.linear.bias.uniform_(-bound, bound)

            def forward(self, value: Any) -> Any:
                return torch.sin(self.omega_0 * self.linear(value))

        return _Sine()

    def _normalise_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        return (values - self.feature_mean) / self.feature_scale

    def _tensors(self, samples: Sequence[PhysicsSceneV2Sample]) -> tuple[Any, ...]:
        torch = self.torch
        features = torch.from_numpy(
            self._normalise_features(np.asarray([s.features for s in samples], dtype=np.float32))
        ).to(self.device)
        observed_xy = torch.from_numpy(
            np.asarray([s.observed_ray_xy for s in samples], dtype=np.float32)
        ).to(self.device)
        target_xyz = torch.from_numpy(
            np.asarray([s.target_xyz for s in samples], dtype=np.float32)
        ).to(self.device)
        target_ray_xy = torch.from_numpy(
            np.asarray([s.target_ray_xy for s in samples], dtype=np.float32)
        ).to(self.device)
        target_depth = torch.from_numpy(
            np.asarray([s.target_depth_m for s in samples], dtype=np.float32)
        ).to(self.device)
        target_log_depth = torch.log(torch.clamp(target_depth, min=1e-4))
        cameras = torch.from_numpy(
            np.asarray([s.camera_position for s in samples], dtype=np.float32)
        ).to(self.device)
        rotations = torch.from_numpy(
            np.asarray([s.rotation_world_to_camera for s in samples], dtype=np.float32)
        ).to(self.device)
        intrinsics = torch.from_numpy(
            np.asarray([s.intrinsic for s in samples], dtype=np.float32)
        ).to(self.device)
        target_pixels = torch.from_numpy(
            np.asarray([s.target_pixel_under_camera for s in samples], dtype=np.float32)
        ).to(self.device)
        image_sizes = torch.from_numpy(
            np.asarray([s.row.get("image_size", [640, 640]) for s in samples], dtype=np.float32)
        ).to(self.device)
        return (
            features,
            observed_xy,
            target_xyz,
            target_ray_xy,
            target_depth,
            target_log_depth,
            cameras,
            rotations,
            intrinsics,
            target_pixels,
            image_sizes,
        )

    def _depth_and_uncertainty(self, output: Any) -> tuple[Any, Any, Any, Any]:
        """Return depth, depth probabilities, entropy and log-variance."""

        torch = self.torch
        if self.depth_mode == "categorical":
            logits = output[:, : self.depth_bins]
            probabilities = torch.softmax(logits, dim=1)
            depth = (probabilities * self.depth_centers[None, :]).sum(dim=1)
            entropy = -(
                probabilities * torch.log(torch.clamp(probabilities, min=1e-8))
            ).sum(dim=1) / float(np.log(self.depth_bins))
            logvar = output[:, self.depth_bins + 2 : self.depth_bins + 5]
            return depth, probabilities, entropy, torch.clamp(
                logvar, min=self.logvar_min, max=self.logvar_max
            )

        log_depth = output[:, 0] * float(self.log_depth_scale) + float(self.log_depth_mean)
        depth = torch.exp(torch.clamp(log_depth, min=-5.0, max=5.0))
        logvar = output[:, 3:6]
        return depth, None, torch.zeros_like(depth), torch.clamp(
            logvar, min=self.logvar_min, max=self.logvar_max
        )

    def _physical(self, output: Any, tensors: tuple[Any, ...]) -> dict[str, Any]:
        torch = self.torch
        observed_xy, cameras, rotations = tensors[1], tensors[6], tensors[7]
        depth, probabilities, entropy, logvar = self._depth_and_uncertainty(output)
        correction_start = self.depth_bins if self.depth_mode == "categorical" else 1
        corrected_xy = observed_xy + self.DELTA_LIMIT * torch.tanh(
            output[:, correction_start : correction_start + 2]
        )
        camera_point = torch.cat(
            [corrected_xy, torch.ones((len(output), 1), device=self.device)], dim=1
        ) * depth[:, None]
        world_point = cameras + torch.bmm(
            rotations.transpose(1, 2), camera_point[:, :, None]
        ).squeeze(-1)
        std_m = torch.exp(0.5 * logvar)
        return {
            "xyz": world_point,
            "std_m": std_m,
            "depth_m": depth,
            "corrected_ray_xy": corrected_xy,
            "depth_probabilities": probabilities,
            "depth_entropy": entropy,
            "logvar": logvar,
        }

    def _project(self, world_point: Any, tensors: tuple[Any, ...]) -> Any:
        torch = self.torch
        cameras, rotations, intrinsics = tensors[6], tensors[7], tensors[8]
        camera_point = torch.bmm(
            rotations, (world_point - cameras)[:, :, None]
        ).squeeze(-1)
        projected_xy = camera_point[:, :2] / torch.clamp(camera_point[:, 2:3], min=1e-5)
        return torch.stack(
            [
                intrinsics[:, 0, 0] * projected_xy[:, 0] + intrinsics[:, 0, 2],
                intrinsics[:, 1, 1] * projected_xy[:, 1] + intrinsics[:, 1, 2],
            ],
            dim=1,
        )

    def _loss(self, output: Any, tensors: tuple[Any, ...]) -> tuple[Any, dict[str, float]]:
        torch = self.torch
        physical = self._physical(output, tensors)
        target_xyz = tensors[2]
        target_ray_xy = tensors[3]
        target_depth = tensors[4]
        target_log_depth = tensors[5]
        target_pixels = tensors[9]
        image_sizes = tensors[10]

        coord_loss = torch.nn.functional.smooth_l1_loss(
            physical["xyz"], target_xyz, beta=0.20
        )
        corrected_xy = physical["corrected_ray_xy"]
        ray_loss = torch.nn.functional.smooth_l1_loss(
            corrected_xy, target_ray_xy, beta=0.01
        )

        projected_pixels = self._project(physical["xyz"], tensors)
        pixel_delta = (projected_pixels - target_pixels) / torch.clamp(image_sizes, min=1.0)
        reprojection_loss = torch.nn.functional.smooth_l1_loss(
            pixel_delta, torch.zeros_like(pixel_delta), beta=0.004
        )

        if self.depth_mode == "categorical":
            centers = self.depth_centers
            target_bins = torch.bucketize(target_depth.detach(), centers)
            target_bins = torch.clamp(target_bins, 0, self.depth_bins - 1).long()
            logits = output[:, : self.depth_bins]
            categorical_loss = torch.nn.functional.cross_entropy(logits, target_bins)
            depth_scale = max(self.depth_max_m - self.depth_min_m, 1e-3)
            expected_depth_loss = torch.nn.functional.smooth_l1_loss(
                (physical["depth_m"] - self.depth_min_m) / depth_scale,
                (target_depth - self.depth_min_m) / depth_scale,
                beta=0.02,
            )
            depth_loss = categorical_loss + 0.75 * expected_depth_loss
            depth_metric = categorical_loss
        else:
            predicted_log_depth = output[:, 0]
            target_log_depth_norm = (
                target_log_depth - float(self.log_depth_mean)
            ) / float(self.log_depth_scale)
            depth_loss = torch.nn.functional.smooth_l1_loss(
                predicted_log_depth, target_log_depth_norm, beta=0.25
            )
            depth_metric = depth_loss

        residual = physical["xyz"] - target_xyz
        logvar = physical["logvar"]
        nll = 0.5 * (torch.exp(-logvar) * residual.pow(2) + logvar).mean()
        total = (
            coord_loss
            + 0.20 * depth_loss
            + 0.35 * ray_loss
            + 0.50 * reprojection_loss
            + 0.05 * nll
        )
        metrics = {
            "coord_loss": float(coord_loss.detach().cpu()),
            "depth_loss": float(depth_loss.detach().cpu()),
            "depth_primary_loss": float(depth_metric.detach().cpu()),
            "ray_loss": float(ray_loss.detach().cpu()),
            "reprojection_loss": float(reprojection_loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "mean_sigma_m": float(
                torch.linalg.norm(physical["std_m"], dim=1).mean().detach().cpu()
            ),
            "mean_depth_entropy": float(physical["depth_entropy"].mean().detach().cpu()),
        }
        return total, metrics

    def fit(
        self,
        train_samples: Sequence[PhysicsSceneV2Sample],
        val_samples: Sequence[PhysicsSceneV2Sample],
        epochs: int = 60,
        batch_size: int = 64,
        patience: int = 15,
        learning_rate: float = 8e-4,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        if not train_samples or not val_samples:
            raise ValueError("SIREN/categorical model requires non-empty train and val samples")
        set_seed(seed)
        train_features = np.asarray([s.features for s in train_samples], dtype=np.float32)
        self.feature_mean = train_features.mean(axis=0)
        self.feature_scale = np.maximum(train_features.std(axis=0), 1e-5).astype(np.float32)
        train_depths = np.asarray([s.target_depth_m for s in train_samples], dtype=np.float32)
        log_depths = np.log(np.maximum(train_depths, 1e-4))
        self.log_depth_mean = float(log_depths.mean())
        self.log_depth_scale = float(max(log_depths.std(), 1e-3))
        if self.depth_mode == "categorical":
            # Fit the range from train only.  The small margin prevents a target
            # exactly at the observed extreme from being clipped to the edge.
            self.depth_min_m = float(max(0.1, train_depths.min() - 0.5))
            self.depth_max_m = float(train_depths.max() + 0.5)
            centers = np.linspace(
                self.depth_min_m,
                self.depth_max_m,
                self.depth_bins,
                dtype=np.float32,
            )
            self.depth_centers = self.torch.from_numpy(centers).to(self.device)

        train_tensors = self._tensors(train_samples)
        val_tensors = self._tensors(val_samples)
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(
            TensorDataset(*train_tensors),
            batch_size=max(1, int(batch_size)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, int(patience // 3)),
            min_lr=1e-5,
        )
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        stale = 0
        self.history = []

        for epoch in range(1, max(1, int(epochs)) + 1):
            self.model.train()
            train_values: list[float] = []
            parts: dict[str, list[float]] = {}
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                output = self.model(batch[0])
                loss, metrics = self._loss(output, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite SIREN/categorical loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                train_values.append(float(loss.detach().cpu()))
                for key, value in metrics.items():
                    parts.setdefault(key, []).append(value)

            self.model.eval()
            with torch.inference_mode():
                val_output = self.model(val_tensors[0])
                val_physical = self._physical(val_output, val_tensors)
                val_loss, val_parts = self._loss(val_output, val_tensors)
            val_metric = metric_summary(
                np.asarray([s.target_xyz for s in val_samples], dtype=np.float32),
                val_physical["xyz"].detach().cpu().numpy(),
            )
            val_mae = float(
                val_metric["mae_m"] if val_metric["mae_m"] is not None else val_loss.cpu()
            )
            record: dict[str, float] = {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(np.mean(train_values)),
                "val_loss": float(val_loss.detach().cpu()),
                "val_mae_m": val_mae,
                "val_mean_sigma_m": float(
                    torch.linalg.norm(val_physical["std_m"], dim=1).mean().detach().cpu()
                ),
                "val_mean_depth_entropy": float(
                    val_physical["depth_entropy"].mean().detach().cpu()
                ),
            }
            record.update({f"train_{key}": float(np.mean(values)) for key, values in parts.items()})
            record.update({f"val_{key}": float(value) for key, value in val_parts.items()})
            self.history.append(record)
            scheduler.step(val_mae)
            if val_mae < best_val - 1e-6:
                best_val = val_mae
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= max(1, int(patience)):
                break

        self.model.load_state_dict(best_state)
        return self.history

    def predict(self, samples: Sequence[PhysicsSceneV2Sample]) -> dict[str, np.ndarray]:
        if not samples:
            return {
                "xyz": np.empty((0, 3), dtype=np.float32),
                "std_m": np.empty((0, 3), dtype=np.float32),
                "depth_m": np.empty((0,), dtype=np.float32),
                "corrected_ray_xy": np.empty((0, 2), dtype=np.float32),
                "depth_entropy": np.empty((0,), dtype=np.float32),
                "depth_probabilities": np.empty((0, self.depth_bins), dtype=np.float32),
                "projected_pixel": np.empty((0, 2), dtype=np.float32),
            }
        tensors = self._tensors(samples)
        self.model.eval()
        with self.torch.inference_mode():
            output = self.model(tensors[0])
            physical = self._physical(output, tensors)
            projected = self._project(physical["xyz"], tensors)
        probabilities = physical["depth_probabilities"]
        if probabilities is None:
            probabilities_array = np.empty((len(samples), self.depth_bins), dtype=np.float32)
        else:
            probabilities_array = probabilities.detach().cpu().numpy().astype(np.float32)
        return {
            "xyz": physical["xyz"].detach().cpu().numpy().astype(np.float32),
            "std_m": physical["std_m"].detach().cpu().numpy().astype(np.float32),
            "depth_m": physical["depth_m"].detach().cpu().numpy().astype(np.float32),
            "corrected_ray_xy": physical["corrected_ray_xy"].detach().cpu().numpy().astype(np.float32),
            "depth_entropy": physical["depth_entropy"].detach().cpu().numpy().astype(np.float32),
            "depth_probabilities": probabilities_array,
            "projected_pixel": projected.detach().cpu().numpy().astype(np.float32),
        }

    def save(self, path: Path | str, metadata: Optional[dict[str, Any]] = None) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "format": self.FORMAT,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "hidden_layers": self.hidden_layers,
                "backbone": self.backbone,
                "depth_mode": self.depth_mode,
                "depth_bins": self.depth_bins,
                "depth_min_m": self.depth_min_m,
                "depth_max_m": self.depth_max_m,
                "omega_0": self.omega_0,
                "feature_names": list(V2_FEATURE_NAMES),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "log_depth_mean": self.log_depth_mean,
                "log_depth_scale": self.log_depth_scale,
                "logvar_min": self.logvar_min,
                "logvar_max": self.logvar_max,
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "PhysicsSceneSirenCategorical":
        checkpoint = __import__("torch").load(
            Path(path).expanduser().resolve(), map_location=device, weights_only=False
        )
        model = cls(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            hidden_layers=int(checkpoint.get("hidden_layers", 3)),
            backbone=str(checkpoint["backbone"]),
            depth_mode=str(checkpoint["depth_mode"]),
            depth_bins=int(checkpoint.get("depth_bins", 80)),
            depth_min_m=float(checkpoint.get("depth_min_m", 6.0)),
            depth_max_m=float(checkpoint.get("depth_max_m", 20.0)),
            omega_0=float(checkpoint.get("omega_0", 20.0)),
            device=device,
        )
        model.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        model.feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
        model.log_depth_mean = float(checkpoint.get("log_depth_mean", 0.0))
        model.log_depth_scale = float(checkpoint.get("log_depth_scale", 1.0))
        model.logvar_min = float(checkpoint.get("logvar_min", -8.0))
        model.logvar_max = float(checkpoint.get("logvar_max", 4.0))
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        return model


__all__ = [
    "PhysicsSceneSirenCategorical",
    "PhysicsSceneV2Sample",
    "V2_FEATURE_NAMES",
    "build_samples",
    "metric_summary",
    "set_seed",
]
