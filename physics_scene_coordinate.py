"""Physics-informed scene-coordinate regression for the post-ROI branch.

The model predicts a metric camera depth and a small correction to the
observed pinhole ray. A physical layer converts those quantities to XYZ:

    X_world = C + R.T @ ([ray_x, ray_y, 1] * depth)

This is deliberately complementary to mesh Ray Casting. The learned branch
is a fast fallback when a detector pixel or calibration is noisy; Ray Casting
remains the independently evaluated geometric baseline.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


PHYSICS_FEATURE_NAMES = (
    "ray_cam_x",
    "ray_cam_y",
    "ray_cam_z",
    "ray_world_x",
    "ray_world_y",
    "ray_world_z",
    "camera_x_norm",
    "camera_y_norm",
    "camera_z_norm",
    "fx_norm",
    "fy_norm",
    "cx_norm",
    "cy_norm",
    "r00",
    "r01",
    "r02",
    "r10",
    "r11",
    "r12",
    "r20",
    "r21",
    "r22",
)
DELTA_LIMIT = 0.35


@dataclass(frozen=True)
class PhysicsSceneSample:
    row: dict[str, Any]
    condition: str
    features: np.ndarray
    target_xyz: np.ndarray
    target_depth_m: float
    target_ray_xy: np.ndarray
    observed_ray_xy: np.ndarray
    camera_position: np.ndarray
    rotation_world_to_camera: np.ndarray
    intrinsic: np.ndarray
    pixel: np.ndarray
    target_pixel_under_camera: np.ndarray
    scene_id: str
    surface: str


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


def _finite(value: Any, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _pixel(row: dict[str, Any], key: str) -> Optional[np.ndarray]:
    value = _finite(row.get(key), (2,))
    if value is None:
        return None
    size = _finite(row.get("image_size", [640, 640]), (2,))
    if size is None or np.any(size <= 0) or np.any(value < -0.5) or np.any(value > size + 0.5):
        return None
    if np.allclose(value, 0.0, atol=1e-12):
        return None
    return value


def _camera(row: dict[str, Any], source: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    value = row.get("camera_estimated" if source == "estimated" else "camera")
    if not isinstance(value, dict):
        return None
    K = _finite(value.get("K"), (3, 3))
    R = _finite(value.get("R_world_to_camera", value.get("R")), (3, 3))
    C = _finite(value.get("camera_position"), (3,))
    if K is None or R is None or C is None:
        return None
    if abs(float(np.linalg.det(K))) < 1e-12:
        return None
    return K, R, C


def _condition_spec(condition: str) -> tuple[str, str]:
    values = {
        "clean_true": ("p_fire_pixel", "true"),
        "noisy_true": ("p_fire_noisy_pixel", "true"),
        "clean_estimated": ("p_fire_pixel", "estimated"),
        "noisy_estimated": ("p_fire_noisy_pixel", "estimated"),
    }
    if condition not in values:
        raise ValueError(f"Unknown physics condition: {condition}")
    return values[condition]


def build_samples(
    rows: Iterable[dict[str, Any]],
    condition: str,
) -> tuple[list[PhysicsSceneSample], dict[str, int]]:
    """Build visible-fire samples for one pixel/pose condition."""

    pixel_key, camera_source = _condition_spec(condition)
    samples: list[PhysicsSceneSample] = []
    skipped: dict[str, int] = {}

    def skip(name: str) -> None:
        skipped[name] = skipped.get(name, 0) + 1

    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            skip("not_visible_fire")
            continue
        pixel = _pixel(row, pixel_key)
        target = _finite(row.get("fire_xyz_world"), (3,))
        camera = _camera(row, camera_source)
        image_size = _finite(row.get("image_size", [640, 640]), (2,))
        if pixel is None:
            skip("missing_or_invalid_pixel")
            continue
        if target is None:
            skip("missing_xyz")
            continue
        if camera is None or image_size is None:
            skip("invalid_camera")
            continue
        K, R, C = camera
        try:
            ray_cam = np.linalg.solve(K, np.array([pixel[0], pixel[1], 1.0], dtype=np.float64))
        except np.linalg.LinAlgError:
            skip("singular_intrinsic")
            continue
        if abs(float(ray_cam[2])) < 1e-12:
            skip("invalid_ray")
            continue
        ray_cam = ray_cam / ray_cam[2]
        ray_world = R.T @ ray_cam
        ray_world_norm = float(np.linalg.norm(ray_world))
        if ray_world_norm < 1e-12:
            skip("invalid_world_ray")
            continue
        ray_world = ray_world / ray_world_norm

        target_camera = R @ (target - C)
        if target_camera[2] <= 1e-6:
            skip("target_behind_camera")
            continue
        target_ray_xy = target_camera[:2] / target_camera[2]
        target_pixel = np.array(
            [
                K[0, 0] * target_ray_xy[0] + K[0, 2],
                K[1, 1] * target_ray_xy[1] + K[1, 2],
            ],
            dtype=np.float64,
        )
        width, height = image_size
        values = np.concatenate(
            [
                ray_cam,
                ray_world,
                C / np.array([8.0, 14.0, 4.0], dtype=np.float64),
                np.array(
                    [
                        K[0, 0] / width,
                        K[1, 1] / height,
                        K[0, 2] / width,
                        K[1, 2] / height,
                    ],
                    dtype=np.float64,
                ),
                R.reshape(-1),
            ]
        )
        if values.shape != (len(PHYSICS_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
            skip("invalid_features")
            continue
        samples.append(
            PhysicsSceneSample(
                row=row,
                condition=condition,
                features=values.astype(np.float32),
                target_xyz=target.astype(np.float32),
                target_depth_m=float(target_camera[2]),
                target_ray_xy=target_ray_xy.astype(np.float32),
                observed_ray_xy=ray_cam[:2].astype(np.float32),
                camera_position=C.astype(np.float32),
                rotation_world_to_camera=R.astype(np.float32),
                intrinsic=K.astype(np.float32),
                pixel=pixel.astype(np.float32),
                target_pixel_under_camera=target_pixel.astype(np.float32),
                scene_id=str(row.get("scene_id", "unknown")),
                surface=str(row.get("fire_surface", "unknown")),
            )
        )
    return samples, skipped


def metric_summary(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1, 3)
    if len(truth) != len(predicted):
        raise ValueError("targets and predictions must have equal length")
    if not len(truth):
        return {
            "count": 0,
            "mae_m": None,
            "median_m": None,
            "p95_m": None,
            "xyz_mae_m": None,
            "under_0.10m": None,
            "under_0.25m": None,
            "under_0.50m": None,
            "under_1.00m": None,
        }
    delta = predicted - truth
    errors = np.linalg.norm(delta, axis=1)
    return {
        "count": int(len(errors)),
        "mae_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "xyz_mae_m": np.mean(np.abs(delta), axis=0).tolist(),
        "under_0.10m": float(np.mean(errors <= 0.10)),
        "under_0.25m": float(np.mean(errors <= 0.25)),
        "under_0.50m": float(np.mean(errors <= 0.50)),
        "under_1.00m": float(np.mean(errors <= 1.00)),
    }


class PhysicsSceneCoordinateMLP:
    """Physics-constrained depth/ray model with heteroscedastic uncertainty."""

    def __init__(
        self,
        input_dim: int = len(PHYSICS_FEATURE_NAMES),
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PhysicsSceneCoordinateMLP requires PyTorch") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 6),
        ).to(self.device)
        self.feature_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.feature_scale = np.ones(self.input_dim, dtype=np.float32)
        self.log_depth_mean = 0.0
        self.log_depth_scale = 1.0
        self.history: list[dict[str, float]] = []

    def _normalise_features(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_scale

    def _tensors(self, samples: Sequence[PhysicsSceneSample]) -> tuple[Any, ...]:
        torch = self.torch
        features = torch.from_numpy(
            self._normalise_features(np.asarray([sample.features for sample in samples], dtype=np.float32))
        ).to(self.device)
        observed_xy = torch.from_numpy(
            np.asarray([sample.observed_ray_xy for sample in samples], dtype=np.float32)
        ).to(self.device)
        target_xyz = torch.from_numpy(
            np.asarray([sample.target_xyz for sample in samples], dtype=np.float32)
        ).to(self.device)
        target_ray_xy = torch.from_numpy(
            np.asarray([sample.target_ray_xy for sample in samples], dtype=np.float32)
        ).to(self.device)
        target_depth = torch.from_numpy(
            np.log(np.asarray([sample.target_depth_m for sample in samples], dtype=np.float32))
        ).to(self.device)
        cameras = torch.from_numpy(
            np.asarray([sample.camera_position for sample in samples], dtype=np.float32)
        ).to(self.device)
        rotations = torch.from_numpy(
            np.asarray([sample.rotation_world_to_camera for sample in samples], dtype=np.float32)
        ).to(self.device)
        target_pixel = torch.from_numpy(
            np.asarray([sample.target_pixel_under_camera for sample in samples], dtype=np.float32)
        ).to(self.device)
        sizes = torch.from_numpy(
            np.asarray(
                [sample.row.get("image_size", [640, 640]) for sample in samples],
                dtype=np.float32,
            )
        ).to(self.device)
        return features, observed_xy, target_xyz, target_ray_xy, target_depth, cameras, rotations, target_pixel, sizes

    def _physical(self, output: Any, observed_xy: Any, cameras: Any, rotations: Any) -> tuple[Any, Any, Any]:
        torch = self.torch
        log_depth = output[:, 0] * float(self.log_depth_scale) + float(self.log_depth_mean)
        depth = torch.exp(torch.clamp(log_depth, min=-5.0, max=5.0))
        corrected_xy = observed_xy + DELTA_LIMIT * torch.tanh(output[:, 1:3])
        camera_point = torch.cat(
            [corrected_xy, torch.ones((len(output), 1), device=self.device)], dim=1
        ) * depth[:, None]
        world_point = cameras + torch.bmm(rotations.transpose(1, 2), camera_point[:, :, None]).squeeze(-1)
        logvar = torch.clamp(output[:, 3:], min=-8.0, max=4.0)
        std_m = torch.exp(0.5 * logvar)
        return world_point, std_m, depth

    def _loss(self, output: Any, tensors: tuple[Any, ...]) -> tuple[Any, dict[str, float]]:
        torch = self.torch
        (
            _features,
            observed_xy,
            target_xyz,
            target_ray_xy,
            target_log_depth,
            cameras,
            rotations,
            target_pixel,
            image_sizes,
        ) = tensors
        world_point, std_m, _depth = self._physical(output, observed_xy, cameras, rotations)
        log_depth_pred = output[:, 0]
        log_depth_target = (target_log_depth - float(self.log_depth_mean)) / float(self.log_depth_scale)
        corrected_xy = observed_xy + DELTA_LIMIT * torch.tanh(output[:, 1:3])
        target_residual = world_point - target_xyz
        coord_loss = torch.nn.functional.smooth_l1_loss(world_point, target_xyz, beta=0.25)
        depth_loss = torch.nn.functional.smooth_l1_loss(log_depth_pred, log_depth_target, beta=0.25)
        ray_loss = torch.nn.functional.smooth_l1_loss(corrected_xy, target_ray_xy, beta=0.01)
        camera_point = torch.bmm(
            rotations, (world_point - cameras)[:, :, None]
        ).squeeze(-1)
        projected_xy = camera_point[:, :2] / torch.clamp(camera_point[:, 2:3], min=1e-5)
        target_xy = target_pixel.clone()
        target_xy[:, 0] = (target_xy[:, 0] - 0.0)  # retain explicit pixel target in diagnostics
        reprojection_norm = torch.cat(
            [
                projected_xy[:, :1] - target_ray_xy[:, :1],
                projected_xy[:, 1:2] - target_ray_xy[:, 1:2],
            ],
            dim=1,
        )
        reprojection_loss = torch.nn.functional.smooth_l1_loss(
            reprojection_norm, torch.zeros_like(reprojection_norm), beta=0.002
        )
        nll = 0.5 * (
            torch.exp(-torch.clamp(output[:, 3:], min=-8.0, max=4.0))
            * target_residual.pow(2)
            + torch.clamp(output[:, 3:], min=-8.0, max=4.0)
        ).mean()
        total = coord_loss + 0.25 * depth_loss + 0.50 * ray_loss + 0.25 * reprojection_loss + 0.05 * nll
        metrics = {
            "coord_loss": float(coord_loss.detach().cpu()),
            "depth_loss": float(depth_loss.detach().cpu()),
            "ray_loss": float(ray_loss.detach().cpu()),
            "reprojection_loss": float(reprojection_loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "mean_sigma_m": float(torch.linalg.norm(std_m, dim=1).mean().detach().cpu()),
        }
        return total, metrics

    def fit(
        self,
        train_samples: Sequence[PhysicsSceneSample],
        val_samples: Sequence[PhysicsSceneSample],
        epochs: int = 120,
        batch_size: int = 64,
        patience: int = 20,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        if not train_samples or not val_samples:
            raise ValueError("Physics MLP requires non-empty train and validation samples")
        set_seed(seed)
        features = np.asarray([sample.features for sample in train_samples], dtype=np.float32)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = np.maximum(features.std(axis=0), 1e-5).astype(np.float32)
        log_depths = np.log(np.asarray([sample.target_depth_m for sample in train_samples], dtype=np.float32))
        self.log_depth_mean = float(log_depths.mean())
        self.log_depth_scale = float(max(log_depths.std(), 1e-3))
        train_tensors = self._tensors(train_samples)
        val_tensors = self._tensors(val_samples)
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        # The geometry tensors are deterministic metadata; a TensorDataset makes
        # shuffled mini-batches explicit and avoids re-reading JSON.
        train_dataset = TensorDataset(*train_tensors)
        loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        stale = 0
        self.history = []
        for epoch in range(1, max(1, int(epochs)) + 1):
            self.model.train()
            train_losses: list[float] = []
            loss_parts: dict[str, list[float]] = {}
            for batch in loader:
                features_batch = batch[0]
                optimizer.zero_grad(set_to_none=True)
                output = self.model(features_batch)
                loss, parts = self._loss(output, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite physics loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                for key, value in parts.items():
                    loss_parts.setdefault(key, []).append(value)
            self.model.eval()
            with torch.inference_mode():
                val_output = self.model(val_tensors[0])
                val_world, val_std, _ = self._physical(
                    val_output,
                    val_tensors[1],
                    val_tensors[5],
                    val_tensors[6],
                )
                val_loss, val_parts = self._loss(val_output, val_tensors)
            target_val = np.asarray([sample.target_xyz for sample in val_samples], dtype=np.float32)
            val_metrics = metric_summary(target_val, val_world.cpu().numpy())
            val_mae = float(val_metrics["mae_m"] if val_metrics["mae_m"] is not None else val_loss.cpu())
            record = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(val_loss.cpu()),
                "val_mae_m": val_mae,
                "val_mean_sigma_m": float(torch.linalg.norm(val_std, dim=1).mean().cpu()),
            }
            record.update({f"train_{key}": float(np.mean(values)) for key, values in loss_parts.items()})
            record.update({f"val_{key}": float(value) for key, value in val_parts.items()})
            self.history.append(record)
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

    def predict(self, samples: Sequence[PhysicsSceneSample]) -> dict[str, np.ndarray]:
        if not samples:
            return {
                "xyz": np.empty((0, 3), dtype=np.float32),
                "std_m": np.empty((0, 3), dtype=np.float32),
                "depth_m": np.empty((0,), dtype=np.float32),
                "corrected_ray_xy": np.empty((0, 2), dtype=np.float32),
            }
        tensors = self._tensors(samples)
        self.model.eval()
        with self.torch.inference_mode():
            output = self.model(tensors[0])
            xyz, std_m, depth = self._physical(output, tensors[1], tensors[5], tensors[6])
            corrected = tensors[1] + DELTA_LIMIT * self.torch.tanh(output[:, 1:3])
        return {
            "xyz": xyz.cpu().numpy().astype(np.float32),
            "std_m": std_m.cpu().numpy().astype(np.float32),
            "depth_m": depth.cpu().numpy().astype(np.float32),
            "corrected_ray_xy": corrected.cpu().numpy().astype(np.float32),
        }

    def save(self, path: Path | str, metadata: Optional[dict[str, Any]] = None) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "format": "LAB_SAM.physics_scene_mlp.v1",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "feature_names": list(PHYSICS_FEATURE_NAMES),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "log_depth_mean": self.log_depth_mean,
                "log_depth_scale": self.log_depth_scale,
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "PhysicsSceneCoordinateMLP":
        import torch

        checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)
        model = cls(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            device=device,
        )
        model.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        model.feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
        model.log_depth_mean = float(checkpoint["log_depth_mean"])
        model.log_depth_scale = float(checkpoint["log_depth_scale"])
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        return model


__all__ = [
    "DELTA_LIMIT",
    "PHYSICS_FEATURE_NAMES",
    "PhysicsSceneCoordinateMLP",
    "PhysicsSceneSample",
    "build_samples",
    "metric_summary",
    "set_seed",
]
