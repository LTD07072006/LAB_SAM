"""Central paths for the local LAB_SAM datasets and experiment outputs.

The large datasets are kept outside the source tree's tracked code.  This
module also keeps a compatibility fallback for an older checkout where the
two original folders still sit at the project root.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def _existing_or_new(new_path: Path, legacy_path: Path) -> Path:
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    # Prefer the new layout for future error messages and newly created data.
    return new_path


CCTV_DATASET = _existing_or_new(
    DATASETS_ROOT / "fire-detection-from-cctv",
    PROJECT_ROOT / "fire-detection-from-cctv",
)
FIRE_SAMPLES = _existing_or_new(
    DATASETS_ROOT / "fire-samples",
    PROJECT_ROOT / "fire-samples",
)
D_FIRE_ROOT = _existing_or_new(
    DATASETS_ROOT / "D-Fire",
    PROJECT_ROOT / "home-fire-dataset",
)

# Dataset archives are unpacked in-place under ``datasets``.  Keep directory
# paths here so callers do not depend on removable ZIP files.
ARCHIVE_ROOT = DATASETS_ROOT / "archive"
D_FIRE_ROOT_EXTRACTED = DATASETS_ROOT / "D-Fire"
FIRE_SMOKE_ROOT = DATASETS_ROOT / "fire_smoke_dataset" / "FIRE-SMOKE-DATASET"
# Backward-compatible names for callers that imported the old constants.
# They now resolve to extracted directories when available.
ARCHIVE_ZIP = ARCHIVE_ROOT if ARCHIVE_ROOT.exists() else DATASETS_ROOT / "archive.zip"
D_FIRE_ZIP = D_FIRE_ROOT_EXTRACTED if D_FIRE_ROOT_EXTRACTED.exists() else DATASETS_ROOT / "D-Fire.zip"
FIRE_SMOKE_ZIP = FIRE_SMOKE_ROOT if FIRE_SMOKE_ROOT.exists() else DATASETS_ROOT / "FIRE-SMOKE-DATASET.zip"
SYNTHETIC_3D_DATASET = PROJECT_ROOT / "working" / "synthetic_fire_3d_v3"


def under_project(path: str | Path) -> Path:
    """Resolve a relative path against the project root."""

    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value
