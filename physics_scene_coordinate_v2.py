"""Physics-informed scene coordinates with camera-domain canonicalization.

This is an independent v2 branch for the post-ROI experiments.  It combines
three ideas that are useful for the fire-point problem, while keeping the
existing Ray Casting implementation as an external baseline:

1. A pinhole physical layer converts predicted metric depth and a bounded ray
   correction into world XYZ.
2. The input ray is reprojected to a canonical camera before the MLP sees it;
   training can additionally randomize focal length, principal point, pose
   and pixel noise to model camera-domain shift.
3. The loss uses a geometry-error prior.  Samples for which one pixel maps to
   a larger metric displacement receive a bounded higher weight, and the
   predicted world point is reprojected to the target pixel in pixel units.

The module is deliberately independent of image appearance.  It consumes the
post-ROI point and calibration metadata, so results can be compared fairly
with mesh Ray Casting on exactly the same samples.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional, Sequence

import numpy as np


CANONICAL_IMAGE_SIZE = np.array([640.0, 640.0], dtype=np.float64)
CANONICAL_K = np.array(
    [[600.0, 0.0, 320.0], [0.0, 600.0, 320.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
ROOM_SCALE = np.array([8.0, 14.0, 4.0], dtype=np.float64)
DELTA_LIMIT = 0.35

V2_FEATURE_NAMES = (
    "canonical_u_norm",
    "canonical_v_norm",
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


@dataclass(frozen=True)
class PhysicsSceneV2Sample:
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
    geometry_weight: float
    pixel_sensitivity_m_per_px: float
    canonical_pixel: np.ndarray
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
    return array if np.all(np.isfinite(array)) else None


def _condition_spec(condition: str) -> tuple[str, str]:
    values = {
        "clean_true": ("p_fire_pixel", "true"),
        "noisy_true": ("p_fire_noisy_pixel", "true"),
        "clean_estimated": ("p_fire_pixel", "estimated"),
        "noisy_estimated": ("p_fire_noisy_pixel", "estimated"),
    }
    if condition not in values:
        raise ValueError(f"Unknown condition: {condition}")
    return values[condition]


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


def _row_pixel(row: dict[str, Any], key: str) -> Optional[np.ndarray]:
    value = row.get(key)
    pixel = _finite(value, (2,))
    if pixel is None:
        return None
    size = _finite(row.get("image_size", [640, 640]), (2,))
    if size is None or np.any(size <= 0) or np.any(pixel < -0.5) or np.any(pixel > size + 0.5):
        return None
    return pixel


def _project(target_xyz: np.ndarray, K: np.ndarray, R: np.ndarray, C: np.ndarray) -> Optional[np.ndarray]:
    target_camera = R @ (target_xyz - C)
    if target_camera[2] <= 1e-6:
        return None
    return np.array(
        [
            K[0, 0] * target_camera[0] / target_camera[2] + K[0, 2],
            K[1, 1] * target_camera[1] / target_camera[2] + K[1, 2],
        ],
        dtype=np.float64,
    )


def _rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def _make_sample(
    row: dict[str, Any],
    condition: str,
    target_xyz: np.ndarray,
    pixel: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    canonicalize: bool = True,
) -> Optional[PhysicsSceneV2Sample]:
    try:
        ray_cam = np.linalg.solve(K, np.array([pixel[0], pixel[1], 1.0], dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    if abs(float(ray_cam[2])) < 1e-12:
        return None
    ray_cam = ray_cam / ray_cam[2]
    ray_world = R.T @ ray_cam
    ray_world_norm = float(np.linalg.norm(ray_world))
    if ray_world_norm < 1e-12:
        return None
    ray_world = ray_world / ray_world_norm

    target_camera = R @ (target_xyz - C)
    if target_camera[2] <= 1e-6:
        return None
    target_ray_xy = target_camera[:2] / target_camera[2]
    target_pixel = _project(target_xyz, K, R, C)
    if target_pixel is None:
        return None

    canonical_h = CANONICAL_K @ ray_cam
    if abs(float(canonical_h[2])) < 1e-12:
        return None
    canonical_pixel = canonical_h[:2] / canonical_h[2]
    pixel_sensitivity = float(
        np.sqrt((target_camera[2] / max(float(K[0, 0]), 1e-6)) ** 2
                + (target_camera[2] / max(float(K[1, 1]), 1e-6)) ** 2)
    )
    # A bounded prior emphasizes geometrically sensitive samples without
    # allowing far-away points to dominate the entire objective.
    geometry_weight = 1.0 + 2.0 * float(np.clip(pixel_sensitivity / 0.01, 0.0, 3.0))

    width, height = np.asarray(row.get("image_size", [640, 640]), dtype=np.float64)
    view_pixel = canonical_pixel if canonicalize else pixel
    values = np.concatenate(
        [
            view_pixel / CANONICAL_IMAGE_SIZE,
            ray_cam,
            ray_world,
            C / ROOM_SCALE,
            np.array([K[0, 0] / width, K[1, 1] / height, K[0, 2] / width, K[1, 2] / height]),
            R.reshape(-1),
        ]
    )
    if values.shape != (len(V2_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        return None
    return PhysicsSceneV2Sample(
        row=row,
        condition=condition,
        features=values.astype(np.float32),
        target_xyz=target_xyz.astype(np.float32),
        target_depth_m=float(target_camera[2]),
        target_ray_xy=target_ray_xy.astype(np.float32),
        observed_ray_xy=ray_cam[:2].astype(np.float32),
        camera_position=C.astype(np.float32),
        rotation_world_to_camera=R.astype(np.float32),
        intrinsic=K.astype(np.float32),
        pixel=pixel.astype(np.float32),
        target_pixel_under_camera=target_pixel.astype(np.float32),
        geometry_weight=float(geometry_weight),
        pixel_sensitivity_m_per_px=float(pixel_sensitivity),
        canonical_pixel=canonical_pixel.astype(np.float32),
        scene_id=str(row.get("scene_id", "unknown")),
        surface=str(row.get("fire_surface", "unknown")),
    )


def build_samples(
    rows: Iterable[dict[str, Any]],
    condition: str,
    canonicalize: bool = True,
) -> tuple[list[PhysicsSceneV2Sample], dict[str, int]]:
    """Build visible-fire samples for one pixel/calibration condition."""

    pixel_key, camera_source = _condition_spec(condition)
    result: list[PhysicsSceneV2Sample] = []
    skipped: dict[str, int] = {}

    def skip(name: str) -> None:
        skipped[name] = skipped.get(name, 0) + 1

    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            skip("not_visible_fire")
            continue
        target = _finite(row.get("fire_xyz_world"), (3,))
        pixel = _row_pixel(row, pixel_key)
        camera = _camera(row, camera_source)
        if target is None:
            skip("missing_xyz")
            continue
        if pixel is None:
            skip("missing_pixel")
            continue
        if camera is None:
            skip("invalid_camera")
            continue
        sample = _make_sample(row, condition, target, pixel, *camera, canonicalize=canonicalize)
        if sample is None:
            skip("invalid_projection")
            continue
        result.append(sample)
    return result, skipped


def augment_domain_samples(
    samples: Sequence[PhysicsSceneV2Sample],
    copies_per_sample: int = 1,
    seed: int = 42,
    focal_pct: float = 0.05,
    principal_px: float = 10.0,
    rotation_deg: float = 2.0,
    position_m: float = 0.05,
    point_sigma_px: float = 2.0,
    canonicalize: bool = True,
) -> list[PhysicsSceneV2Sample]:
    """Generate camera-domain shifts without changing the metric target.

    The target point is reprojected through a perturbed camera.  This is a
    geometry-preserving augmentation, unlike an image-only affine transform
    that would silently invalidate the camera model.
    """

    rng = np.random.default_rng(int(seed))
    augmented: list[PhysicsSceneV2Sample] = []
    for sample_index, base in enumerate(samples):
        for copy_index in range(max(0, int(copies_per_sample))):
            K = np.asarray(base.intrinsic, dtype=np.float64).copy()
            K[0, 0] *= 1.0 + float(rng.normal(0.0, focal_pct))
            K[1, 1] *= 1.0 + float(rng.normal(0.0, focal_pct))
            K[0, 2] += float(rng.normal(0.0, principal_px))
            K[1, 2] += float(rng.normal(0.0, principal_px))
            delta = _rotation_xyz(
                *np.deg2rad(rng.normal(0.0, rotation_deg, size=3))
            )
            R = delta @ np.asarray(base.rotation_world_to_camera, dtype=np.float64)
            C = np.asarray(base.camera_position, dtype=np.float64) + rng.normal(0.0, position_m, size=3)
            clean_pixel = _project(np.asarray(base.target_xyz, dtype=np.float64), K, R, C)
            if clean_pixel is None:
                continue
            observed_pixel = clean_pixel + rng.normal(0.0, point_sigma_px, size=2)
            row = dict(base.row)
            row["sample_id"] = f"{base.row.get('sample_id', sample_index)}__domain_{copy_index}"
            sample = _make_sample(
                row,
                "domain_augmented",
                np.asarray(base.target_xyz, dtype=np.float64),
                observed_pixel,
                K,
                R,
                C,
                canonicalize=canonicalize,
            )
            if sample is not None:
                augmented.append(sample)
    return augmented


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
            "xyz_p95_abs_m": None,
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
        "xyz_p95_abs_m": np.percentile(np.abs(delta), 95, axis=0).tolist(),
        "under_0.10m": float(np.mean(errors <= 0.10)),
        "under_0.25m": float(np.mean(errors <= 0.25)),
        "under_0.50m": float(np.mean(errors <= 0.50)),
        "under_1.00m": float(np.mean(errors <= 1.00)),
    }


class PhysicsSceneCoordinateMLPv2:
    """Canonicalized physical MLP with geometry-error-prior supervision."""

    def __init__(
        self,
        input_dim: int = len(V2_FEATURE_NAMES),
        hidden_dim: int = 160,
        device: str = "cpu",
        canonicalize: bool = True,
        geometry_prior: bool = True,
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("PhysicsSceneCoordinateMLPv2 requires PyTorch") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.canonicalize = bool(canonicalize)
        self.geometry_prior = bool(geometry_prior)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
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

    def _tensors(self, samples: Sequence[PhysicsSceneV2Sample]) -> tuple[Any, ...]:
        torch = self.torch
        features = torch.from_numpy(
            self._normalise_features(np.asarray([s.features for s in samples], dtype=np.float32))
        ).to(self.device)
        observed_xy = torch.from_numpy(np.asarray([s.observed_ray_xy for s in samples], dtype=np.float32)).to(self.device)
        target_xyz = torch.from_numpy(np.asarray([s.target_xyz for s in samples], dtype=np.float32)).to(self.device)
        target_ray_xy = torch.from_numpy(np.asarray([s.target_ray_xy for s in samples], dtype=np.float32)).to(self.device)
        target_log_depth = torch.from_numpy(
            np.log(np.asarray([s.target_depth_m for s in samples], dtype=np.float32))
        ).to(self.device)
        cameras = torch.from_numpy(np.asarray([s.camera_position for s in samples], dtype=np.float32)).to(self.device)
        rotations = torch.from_numpy(
            np.asarray([s.rotation_world_to_camera for s in samples], dtype=np.float32)
        ).to(self.device)
        intrinsics = torch.from_numpy(np.asarray([s.intrinsic for s in samples], dtype=np.float32)).to(self.device)
        target_pixels = torch.from_numpy(
            np.asarray([s.target_pixel_under_camera for s in samples], dtype=np.float32)
        ).to(self.device)
        image_sizes = torch.from_numpy(
            np.asarray([s.row.get("image_size", [640, 640]) for s in samples], dtype=np.float32)
        ).to(self.device)
        geometry_weights = torch.from_numpy(
            np.asarray([s.geometry_weight for s in samples], dtype=np.float32)
        ).to(self.device)
        return (
            features,
            observed_xy,
            target_xyz,
            target_ray_xy,
            target_log_depth,
            cameras,
            rotations,
            intrinsics,
            target_pixels,
            image_sizes,
            geometry_weights,
        )

    def _physical(self, output: Any, observed_xy: Any, cameras: Any, rotations: Any) -> tuple[Any, Any, Any, Any]:
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
        return world_point, std_m, depth, corrected_xy

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
            intrinsics,
            target_pixels,
            image_sizes,
            geometry_weights,
        ) = tensors
        world_point, std_m, _depth, corrected_xy = self._physical(output, observed_xy, cameras, rotations)
        prior = geometry_weights if self.geometry_prior else torch.ones_like(geometry_weights)
        coord_per_sample = torch.nn.functional.smooth_l1_loss(
            world_point, target_xyz, beta=0.20, reduction="none"
        ).mean(dim=1)
        coord_loss = (coord_per_sample * prior).mean() / torch.clamp(prior.mean(), min=1.0)
        log_depth_pred = output[:, 0]
        target_log_depth_norm = (target_log_depth - float(self.log_depth_mean)) / float(self.log_depth_scale)
        depth_loss = torch.nn.functional.smooth_l1_loss(log_depth_pred, target_log_depth_norm, beta=0.25)
        ray_loss = torch.nn.functional.smooth_l1_loss(corrected_xy, target_ray_xy, beta=0.01)

        camera_point = torch.bmm(rotations, (world_point - cameras)[:, :, None]).squeeze(-1)
        projected_xy = camera_point[:, :2] / torch.clamp(camera_point[:, 2:3], min=1e-5)
        projected_pixels = torch.stack(
            [
                intrinsics[:, 0, 0] * projected_xy[:, 0] + intrinsics[:, 0, 2],
                intrinsics[:, 1, 1] * projected_xy[:, 1] + intrinsics[:, 1, 2],
            ],
            dim=1,
        )
        pixel_delta_norm = (projected_pixels - target_pixels) / torch.clamp(image_sizes, min=1.0)
        reprojection_per_sample = torch.nn.functional.smooth_l1_loss(
            pixel_delta_norm, torch.zeros_like(pixel_delta_norm), beta=0.004, reduction="none"
        ).mean(dim=1)
        reprojection_loss = (reprojection_per_sample * prior).mean() / torch.clamp(prior.mean(), min=1.0)

        residual = world_point - target_xyz
        logvar = torch.clamp(output[:, 3:], min=-8.0, max=4.0)
        nll_per_sample = 0.5 * (
            torch.exp(-logvar) * residual.pow(2) + logvar
        ).mean(dim=1)
        nll = (nll_per_sample * prior).mean() / torch.clamp(prior.mean(), min=1.0)
        total = coord_loss + 0.20 * depth_loss + 0.35 * ray_loss + 0.50 * reprojection_loss + 0.05 * nll
        metrics = {
            "coord_loss": float(coord_loss.detach().cpu()),
            "depth_loss": float(depth_loss.detach().cpu()),
            "ray_loss": float(ray_loss.detach().cpu()),
            "reprojection_loss": float(reprojection_loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "mean_sigma_m": float(torch.linalg.norm(std_m, dim=1).mean().detach().cpu()),
            "mean_geometry_weight": float(prior.mean().detach().cpu()),
        }
        return total, metrics

    def fit(
        self,
        train_samples: Sequence[PhysicsSceneV2Sample],
        val_samples: Sequence[PhysicsSceneV2Sample],
        epochs: int = 80,
        batch_size: int = 64,
        patience: int = 15,
        learning_rate: float = 1.5e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        if not train_samples or not val_samples:
            raise ValueError("Physics Scene-MLP v2 requires non-empty train and validation samples")
        set_seed(seed)
        features = np.asarray([s.features for s in train_samples], dtype=np.float32)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = np.maximum(features.std(axis=0), 1e-5).astype(np.float32)
        log_depths = np.log(np.asarray([s.target_depth_m for s in train_samples], dtype=np.float32))
        self.log_depth_mean = float(log_depths.mean())
        self.log_depth_scale = float(max(log_depths.std(), 1e-3))
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
            self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, int(patience // 3)), min_lr=1e-5
        )
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        stale = 0
        self.history = []
        for epoch in range(1, max(1, int(epochs)) + 1):
            self.model.train()
            train_losses: list[float] = []
            parts_values: dict[str, list[float]] = {}
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                output = self.model(batch[0])
                loss, parts = self._loss(output, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite v2 physics loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                for key, value in parts.items():
                    parts_values.setdefault(key, []).append(value)
            self.model.eval()
            with torch.inference_mode():
                val_output = self.model(val_tensors[0])
                val_world, val_std, _depth, _corrected = self._physical(
                    val_output, val_tensors[1], val_tensors[5], val_tensors[6]
                )
                val_loss, val_parts = self._loss(val_output, val_tensors)
            val_metrics = metric_summary(
                np.asarray([s.target_xyz for s in val_samples], dtype=np.float32),
                val_world.detach().cpu().numpy(),
            )
            val_mae = float(val_metrics["mae_m"] if val_metrics["mae_m"] is not None else val_loss.cpu())
            record: dict[str, float] = {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(val_loss.detach().cpu()),
                "val_mae_m": val_mae,
                "val_mean_sigma_m": float(torch.linalg.norm(val_std, dim=1).mean().detach().cpu()),
            }
            record.update({f"train_{key}": float(np.mean(values)) for key, values in parts_values.items()})
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
                "projected_pixel": np.empty((0, 2), dtype=np.float32),
            }
        tensors = self._tensors(samples)
        self.model.eval()
        with self.torch.inference_mode():
            output = self.model(tensors[0])
            xyz, std_m, depth, corrected = self._physical(
                output, tensors[1], tensors[5], tensors[6]
            )
            camera_point = self.torch.bmm(tensors[6], (xyz - tensors[5])[:, :, None]).squeeze(-1)
            projected_xy = camera_point[:, :2] / self.torch.clamp(camera_point[:, 2:3], min=1e-5)
            projected = self.torch.stack(
                [
                    tensors[7][:, 0, 0] * projected_xy[:, 0] + tensors[7][:, 0, 2],
                    tensors[7][:, 1, 1] * projected_xy[:, 1] + tensors[7][:, 1, 2],
                ],
                dim=1,
            )
        return {
            "xyz": xyz.cpu().numpy().astype(np.float32),
            "std_m": std_m.cpu().numpy().astype(np.float32),
            "depth_m": depth.cpu().numpy().astype(np.float32),
            "corrected_ray_xy": corrected.cpu().numpy().astype(np.float32),
            "projected_pixel": projected.cpu().numpy().astype(np.float32),
        }

    def save(self, path: str, metadata: Optional[dict[str, Any]] = None) -> None:
        destination = __import__("pathlib").Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "format": "LAB_SAM.physics_scene_mlp.v2",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "feature_names": list(V2_FEATURE_NAMES),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "log_depth_mean": self.log_depth_mean,
                "log_depth_scale": self.log_depth_scale,
                "canonicalize": self.canonicalize,
                "geometry_prior": self.geometry_prior,
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PhysicsSceneCoordinateMLPv2":
        import torch

        checkpoint = torch.load(__import__("pathlib").Path(path).expanduser().resolve(), map_location=device, weights_only=False)
        model = cls(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            device=device,
            canonicalize=bool(checkpoint.get("canonicalize", True)),
            geometry_prior=bool(checkpoint.get("geometry_prior", True)),
        )
        model.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        model.feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
        model.log_depth_mean = float(checkpoint["log_depth_mean"])
        model.log_depth_scale = float(checkpoint["log_depth_scale"])
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        return model


__all__ = [
    "CANONICAL_K",
    "DELTA_LIMIT",
    "PhysicsSceneCoordinateMLPv2",
    "PhysicsSceneV2Sample",
    "V2_FEATURE_NAMES",
    "augment_domain_samples",
    "build_samples",
    "metric_summary",
    "set_seed",
]
