"""Lightweight learned 2D-to-3D scene-coordinate regression.

This module implements a deliberately separate post-ROI branch:

    ROI/coarse pixel + camera calibration metadata -> metric XYZ

It does not call Homography/IPM, does not intersect a mesh, and does not use
the fire image itself.  The small MLP learns a scene prior from metric
synthetic/asset-backed records.  That makes it useful as a low-cost feasibility
experiment, but it is not a general monocular 3D reconstruction method: a
pixel ray can correspond to multiple depths unless the scene prior is strong.

The implementation is dependency-light apart from PyTorch, which is already
used by the detector/ROI training code.  It intentionally keeps calibration
features explicit so the model cannot silently confuse pixel coordinates from
different camera resolutions or poses.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


FEATURE_NAMES = (
    "u_norm",
    "v_norm",
    "fx_norm",
    "fy_norm",
    "cx_norm",
    "cy_norm",
    "cam_x",
    "cam_y",
    "cam_z",
    "r00",
    "r01",
    "r02",
    "r10",
    "r11",
    "r12",
    "r20",
    "r21",
    "r22",
    "ray_x",
    "ray_y",
    "ray_z",
)


@dataclass(frozen=True)
class SceneCoordinateSample:
    """One valid visible-fire training/evaluation sample."""

    row: dict[str, Any]
    features: np.ndarray
    target_xyz: np.ndarray
    pixel: np.ndarray
    scene_id: str


@dataclass
class RegressionMetrics:
    """Metric summary for a set of predicted scene coordinates."""

    count: int
    mae_m: Optional[float]
    median_m: Optional[float]
    p95_m: Optional[float]
    xyz_mae_m: Optional[list[float]]
    xyz_p95_abs_m: Optional[list[float]]
    under_0_10m: Optional[float]
    under_0_25m: Optional[float]
    under_0_50m: Optional[float]
    under_1_00m: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mae_m": self.mae_m,
            "median_m": self.median_m,
            "p95_m": self.p95_m,
            "xyz_mae_m": self.xyz_mae_m,
            "xyz_p95_abs_m": self.xyz_p95_abs_m,
            "under_0.10m": self.under_0_10m,
            "under_0.25m": self.under_0_25m,
            "under_0.50m": self.under_0_50m,
            "under_1.00m": self.under_1_00m,
        }


def set_seed(seed: int) -> None:
    """Make the small CPU experiment repeatable."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def load_manifest_rows(dataset_root: Path | str, split: str = "all") -> list[dict[str, Any]]:
    """Load split JSONL files, falling back to ``manifest.jsonl``."""

    root = Path(dataset_root).expanduser().resolve()
    names = ["manifest.jsonl"] if split == "all" else [f"{split}.jsonl", "manifest.jsonl"]
    path = next((root / name for name in names if (root / name).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No manifest JSONL found under {root}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if split == "all" or str(row.get("split", split)) == split:
                    rows.append(row)
    if not rows:
        raise RuntimeError(f"No records for split={split!r} in {path}")
    return rows


def _finite_array(value: Any, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError):
        return None
    return array if np.all(np.isfinite(array)) else None


def _camera_dict(row: dict[str, Any], camera_source: str) -> Optional[dict[str, Any]]:
    source = "camera_estimated" if camera_source == "estimated" else "camera"
    value = row.get(source)
    return value if isinstance(value, dict) else None


def make_feature_vector(
    row: dict[str, Any],
    pixel: Sequence[float],
    camera_source: str = "estimated",
) -> Optional[np.ndarray]:
    """Create a calibration-aware feature vector for one observed pixel.

    The camera position is scaled by a fixed room-scale convention before the
    training standardisation.  The world ray is included as a useful inductive
    feature, but no ray/mesh intersection is performed here.
    """

    camera = _camera_dict(row, camera_source)
    image_size = _finite_array(row.get("image_size", [640, 640]), (2,))
    pixel_array = _finite_array(pixel, (2,))
    if camera is None or image_size is None or pixel_array is None or np.any(image_size <= 0.0):
        return None
    K = _finite_array(camera.get("K"), (3, 3))
    R = _finite_array(camera.get("R_world_to_camera", camera.get("R")), (3, 3))
    position = _finite_array(camera.get("camera_position"), (3,))
    if K is None or R is None or position is None:
        return None
    try:
        ray_camera = np.linalg.solve(K, np.array([pixel_array[0], pixel_array[1], 1.0], dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    ray_world = R.T @ ray_camera
    norm = float(np.linalg.norm(ray_world))
    if norm < 1e-12:
        return None
    ray_world /= norm
    width, height = image_size
    # These scales are feature normalisations, not a claim about the room
    # dimensions.  The later train-set standardisation remains authoritative.
    values = np.concatenate(
        [
            pixel_array / np.array([width, height]),
            np.array([K[0, 0] / width, K[1, 1] / height, K[0, 2] / width, K[1, 2] / height]),
            position / np.array([8.0, 14.0, 4.0]),
            R.reshape(-1),
            ray_world,
        ]
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        return None
    return values.astype(np.float32)


def _row_pixel(row: dict[str, Any], pixel_key: str) -> Optional[np.ndarray]:
    value = row.get(pixel_key)
    if value is None and pixel_key == "p_fire_noisy_pixel":
        value = row.get("p_fire_pixel")
    return _finite_array(value, (2,))


def build_samples(
    rows: Iterable[dict[str, Any]],
    pixel_key: str = "p_fire_noisy_pixel",
    camera_source: str = "estimated",
) -> tuple[list[SceneCoordinateSample], dict[str, int]]:
    """Keep only visible fire records with a pixel, camera and XYZ target."""

    samples: list[SceneCoordinateSample] = []
    skipped: dict[str, int] = {}
    for row in rows:
        if not int(row.get("has_fire", 0)) or not int(row.get("fire_visible", 1)):
            skipped["not_visible_fire"] = skipped.get("not_visible_fire", 0) + 1
            continue
        pixel = _row_pixel(row, pixel_key)
        target = _finite_array(row.get("fire_xyz_world"), (3,))
        if pixel is None:
            skipped["missing_pixel"] = skipped.get("missing_pixel", 0) + 1
            continue
        if target is None:
            skipped["missing_xyz"] = skipped.get("missing_xyz", 0) + 1
            continue
        features = make_feature_vector(row, pixel, camera_source)
        if features is None:
            skipped["invalid_camera"] = skipped.get("invalid_camera", 0) + 1
            continue
        samples.append(
            SceneCoordinateSample(
                row=row,
                features=features,
                target_xyz=target.astype(np.float32),
                pixel=pixel.astype(np.float32),
                scene_id=str(row.get("scene_id", "unknown")),
            )
        )
    return samples, skipped


def metric_summary(targets: np.ndarray, predictions: np.ndarray) -> RegressionMetrics:
    """Calculate Euclidean and per-axis metric errors."""

    truth = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1, 3)
    if len(truth) != len(predicted):
        raise ValueError("targets and predictions must have equal length")
    if len(truth) == 0:
        return RegressionMetrics(0, None, None, None, None, None, None, None, None, None)
    delta = predicted - truth
    norms = np.linalg.norm(delta, axis=1)
    return RegressionMetrics(
        count=len(norms),
        mae_m=float(norms.mean()),
        median_m=float(np.median(norms)),
        p95_m=float(np.percentile(norms, 95)),
        xyz_mae_m=np.mean(np.abs(delta), axis=0).tolist(),
        xyz_p95_abs_m=np.percentile(np.abs(delta), 95, axis=0).tolist(),
        under_0_10m=float(np.mean(norms <= 0.10)),
        under_0_25m=float(np.mean(norms <= 0.25)),
        under_0_50m=float(np.mean(norms <= 0.50)),
        under_1_00m=float(np.mean(norms <= 1.00)),
    )


class SceneCoordinateMLP:
    """Small Torch MLP with serialisable feature/target normalisation."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dim: int = 96,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError("SceneCoordinateMLP requires PyTorch") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        ).to(self.device)
        self.feature_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.feature_scale = np.ones(self.input_dim, dtype=np.float32)
        self.target_mean = np.zeros(3, dtype=np.float32)
        self.target_scale = np.ones(3, dtype=np.float32)
        self.validation_std_m = np.full(3, 0.25, dtype=np.float32)
        self.history: list[dict[str, float]] = []

    def _normalise_fit(self, samples: Sequence[SceneCoordinateSample]) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray([sample.features for sample in samples], dtype=np.float32)
        targets = np.asarray([sample.target_xyz for sample in samples], dtype=np.float32)
        if len(features) == 0:
            raise ValueError("Cannot fit scene-coordinate model on zero samples")
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = np.maximum(features.std(axis=0), 1e-5)
        self.target_mean = targets.mean(axis=0)
        self.target_scale = np.maximum(targets.std(axis=0), 1e-5)
        return self._normalise(features, targets)

    def _normalise(self, features: np.ndarray, targets: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        x = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_scale
        if targets is None:
            return x, None
        y = (np.asarray(targets, dtype=np.float32) - self.target_mean) / self.target_scale
        return x, y

    def _denormalise_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.target_scale + self.target_mean

    def fit(
        self,
        train_samples: Sequence[SceneCoordinateSample],
        val_samples: Sequence[SceneCoordinateSample],
        epochs: int = 160,
        batch_size: int = 64,
        patience: int = 28,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        """Train with validation early stopping and return the history."""

        set_seed(seed)
        x_train, y_train = self._normalise_fit(train_samples)
        x_val, y_val = self._normalise(
            np.asarray([sample.features for sample in val_samples], dtype=np.float32),
            np.asarray([sample.target_xyz for sample in val_samples], dtype=np.float32),
        )
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=max(1, int(batch_size)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        x_val_tensor = torch.from_numpy(x_val).to(self.device)
        y_val_tensor = torch.from_numpy(y_val).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        criterion = torch.nn.SmoothL1Loss(beta=0.5)
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        stale = 0
        self.history = []
        for epoch in range(1, max(1, int(epochs)) + 1):
            self.model.train()
            train_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(x_batch), y_batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite scene-coordinate loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.inference_mode():
                val_output = self.model(x_val_tensor)
                val_loss = float(criterion(val_output, y_val_tensor).cpu())
                val_world = self._denormalise_targets(val_output.cpu().numpy())
            val_metrics = metric_summary(
                np.asarray([sample.target_xyz for sample in val_samples], dtype=np.float32),
                val_world,
            )
            val_mae = float(val_metrics.mae_m if val_metrics.mae_m is not None else val_loss)
            record = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses) if train_losses else float("nan")),
                "val_loss": val_loss,
                "val_mae_m": val_mae,
            }
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
        if val_samples:
            val_predictions = self.predict_samples(val_samples)
            val_delta = val_predictions - np.asarray([sample.target_xyz for sample in val_samples], dtype=np.float32)
            self.validation_std_m = np.maximum(np.std(val_delta, axis=0), 0.02).astype(np.float32)
        return self.history

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        values, _ = self._normalise(np.asarray(features, dtype=np.float32))
        tensor = self.torch.from_numpy(values).to(self.device)
        self.model.eval()
        with self.torch.inference_mode():
            output = self.model(tensor).detach().cpu().numpy()
        return self._denormalise_targets(output)

    def predict_samples(self, samples: Sequence[SceneCoordinateSample]) -> np.ndarray:
        if not samples:
            return np.empty((0, 3), dtype=np.float32)
        return self.predict_features(np.asarray([sample.features for sample in samples], dtype=np.float32))

    def save(self, path: Path | str, metadata: Optional[dict[str, Any]] = None) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "format": "LAB_SAM.scene_coordinate_mlp.v1",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "target_mean": self.target_mean,
                "target_scale": self.target_scale,
                "validation_std_m": self.validation_std_m,
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "SceneCoordinateMLP":
        import torch

        checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)
        model = cls(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            device=device,
        )
        model.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        model.feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
        model.target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        model.target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
        model.validation_std_m = np.asarray(checkpoint.get("validation_std_m", [0.25, 0.25, 0.25]), dtype=np.float32)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        return model


class UncertaintySceneCoordinateMLP(SceneCoordinateMLP):
    """Scene-coordinate MLP that predicts XYZ and a per-axis uncertainty.

    The first three outputs are the normalised XYZ mean.  The last three are
    log-variances in the same normalised target space.  Predicting uncertainty
    in normalised space keeps the loss numerically well behaved when the room
    has a long Y axis and a much shorter Z axis.  ``predict_with_uncertainty``
    converts both the mean and standard deviation back to metres.

    This class is intentionally additive: existing ``SceneCoordinateMLP``
    checkpoints and callers remain compatible.
    """

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dim: int = 96,
        device: str = "cpu",
    ) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        nn = self.torch.nn
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 6),
        ).to(self.device)
        self.logvar_min = -8.0
        self.logvar_max = 4.0

    def _split_output(self, output: Any) -> tuple[Any, Any]:
        logvar = self.torch.clamp(
            output[..., 3:], min=self.logvar_min, max=self.logvar_max
        )
        return output[..., :3], logvar

    def _world_from_output(self, output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(output, dtype=np.float32)
        mean_norm = values[..., :3]
        logvar = np.clip(values[..., 3:], self.logvar_min, self.logvar_max)
        mean_world = self._denormalise_targets(mean_norm)
        std_norm = np.exp(0.5 * logvar)
        std_world = std_norm * self.target_scale.reshape(1, 3)
        return mean_world.astype(np.float32), std_world.astype(np.float32)

    def fit(
        self,
        train_samples: Sequence[SceneCoordinateSample],
        val_samples: Sequence[SceneCoordinateSample],
        epochs: int = 160,
        batch_size: int = 64,
        patience: int = 28,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        """Train heteroscedastic regression with validation early stopping."""

        if not train_samples or not val_samples:
            raise ValueError("Uncertainty MLP requires non-empty train and val samples")
        set_seed(seed)
        x_train, y_train = self._normalise_fit(train_samples)
        x_val, y_val = self._normalise(
            np.asarray([sample.features for sample in val_samples], dtype=np.float32),
            np.asarray([sample.target_xyz for sample in val_samples], dtype=np.float32),
        )
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=max(1, int(batch_size)),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        x_val_tensor = torch.from_numpy(x_val).to(self.device)
        y_val_tensor = torch.from_numpy(y_val).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        stale = 0
        self.history = []

        def nll_loss(output: Any, target: Any) -> Any:
            mean_norm, logvar = self._split_output(output)
            squared = (mean_norm - target) ** 2
            # Constant 0.5 is omitted from model selection but retained for the
            # usual Gaussian negative log-likelihood interpretation.
            return 0.5 * (torch.exp(-logvar) * squared + logvar).mean()

        for epoch in range(1, max(1, int(epochs)) + 1):
            self.model.train()
            train_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = nll_loss(self.model(x_batch), y_batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite uncertainty loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            self.model.eval()
            with torch.inference_mode():
                val_output = self.model(x_val_tensor)
                val_loss = float(nll_loss(val_output, y_val_tensor).cpu())
                val_world, val_std = self._world_from_output(val_output.cpu().numpy())
            val_metrics = metric_summary(
                np.asarray([sample.target_xyz for sample in val_samples], dtype=np.float32),
                val_world,
            )
            val_mae = float(val_metrics.mae_m if val_metrics.mae_m is not None else val_loss)
            record = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses) if train_losses else float("nan")),
                "val_loss": val_loss,
                "val_mae_m": val_mae,
                "val_mean_sigma_m": float(np.mean(np.linalg.norm(val_std, axis=1))),
            }
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
        val_predictions, _ = self.predict_with_uncertainty(
            np.asarray([sample.features for sample in val_samples], dtype=np.float32)
        )
        val_delta = val_predictions - np.asarray(
            [sample.target_xyz for sample in val_samples], dtype=np.float32
        )
        self.validation_std_m = np.maximum(np.std(val_delta, axis=0), 0.02).astype(np.float32)
        return self.history

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, _ = self._normalise(np.asarray(features, dtype=np.float32))
        tensor = self.torch.from_numpy(values).to(self.device)
        self.model.eval()
        with self.torch.inference_mode():
            output = self.model(tensor).detach().cpu().numpy()
        return self._world_from_output(output)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(features)
        return mean

    def save(self, path: Path | str, metadata: Optional[dict[str, Any]] = None) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "format": "LAB_SAM.scene_coordinate_mlp_uncertainty.v1",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "target_mean": self.target_mean,
                "target_scale": self.target_scale,
                "validation_std_m": self.validation_std_m,
                "logvar_min": self.logvar_min,
                "logvar_max": self.logvar_max,
                "state_dict": self.model.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "UncertaintySceneCoordinateMLP":
        import torch

        checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)
        model = cls(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            device=device,
        )
        model.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        model.feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
        model.target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        model.target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float32)
        model.validation_std_m = np.asarray(
            checkpoint.get("validation_std_m", [0.25, 0.25, 0.25]), dtype=np.float32
        )
        model.logvar_min = float(checkpoint.get("logvar_min", -8.0))
        model.logvar_max = float(checkpoint.get("logvar_max", 4.0))
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        return model


__all__ = [
    "FEATURE_NAMES",
    "RegressionMetrics",
    "SceneCoordinateMLP",
    "UncertaintySceneCoordinateMLP",
    "SceneCoordinateSample",
    "build_samples",
    "load_manifest_rows",
    "make_feature_vector",
    "metric_summary",
    "set_seed",
]
