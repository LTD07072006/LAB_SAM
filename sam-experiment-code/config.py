"""Shared runtime configuration for the fire-localisation pipeline."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class RuntimeConfig:
    model_path: Path = ROOT / "fire-model-data" / "best.pth"
    sample_dir: Path = ROOT / "fire-samples"
    confidence_threshold: float = 0.50
    temporal_alpha: float = 0.35
    temporal_window: int = 5
    temporal_min_hits: int = 3
    ray_max_distance: float = 1000.0
    ray_coarse_step: float = 2.0
    ray_bisection_iterations: int = 24
    multi_ray_columns: int = 5
    multi_ray_bottom_fraction: float = 0.18
    use_3d_uncertainty: bool = True
    uncertainty_pixel_sigma: float = 2.0
    uncertainty_samples: int = 128
    tracker_alpha: float = 0.35
    tracker_gate_m: float = 3.0
    tracker_max_missed: int = 3

DEFAULT_CONFIG = RuntimeConfig()
