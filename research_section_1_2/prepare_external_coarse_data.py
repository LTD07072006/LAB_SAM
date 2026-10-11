"""Prepare external fire datasets without inventing CCTV point labels.

The output manifest is intentionally explicit: D-Fire box bottom-centres are
weak auxiliary targets, while FIRE-SMOKE-DATASET samples are classification
only. Neither source is merged into the leakage-safe CCTV validation/test set.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


def _dfire_rows(path: Path, output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for image_name in names:
            if "/images/" not in image_name or not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            stem = Path(image_name).stem
            split = next((item for item in ('train', 'val', 'test') if f'data/{item}/images/' in image_name), 'unknown')
            label_name = image_name.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            if label_name not in archive.namelist():
                continue
            with archive.open(image_name) as stream:
                with Image.open(stream) as image:
                    width, height = image.size
            boxes: list[dict[str, Any]] = []
            for line in archive.read(label_name).decode('utf-8', 'ignore').splitlines():
                values = line.split()
                if len(values) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, values[:5])
                # YOLO coordinates are normalized. Bottom centre is a useful
                # weak point proxy for fire boxes, but not CCTV Ground Truth.
                boxes.append({
                    'class_id': int(cls),
                    'weak_point_norm': [cx, min(1.0, cy + bh / 2.0)],
                    'bbox_norm': [cx, cy, bw, bh],
                })
            if boxes:
                rows.append({
                    'image_path': f'zip://{path.name}!/{image_name}',
                    'source': 'D-Fire', 'split': split, 'image_size': [width, height],
                    'boxes': boxes, 'point_supervision': 'weak_bottom_center',
                })
    return rows


def _dfire_rows_from_directory(root: Path) -> list[dict[str, Any]]:
    """Read an extracted D-Fire tree using the same weak-label policy."""
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        image_root = root / "data" / split / "images"
        label_root = root / "data" / split / "labels"
        for image_path in sorted(image_root.glob("*")):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            boxes: list[dict[str, Any]] = []
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                values = line.split()
                if len(values) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, values[:5])
                boxes.append({
                    "class_id": int(cls),
                    "weak_point_norm": [cx, min(1.0, cy + bh / 2.0)],
                    "bbox_norm": [cx, cy, bw, bh],
                })
            if boxes:
                rows.append({
                    "image_path": str(image_path.resolve()),
                    "source": "D-Fire", "split": split,
                    "image_size": [width, height], "boxes": boxes,
                    "point_supervision": "weak_bottom_center",
                })
    return rows


def _fire_smoke_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset = root / 'FIRE-SMOKE-DATASET'
    for split in ('Train', 'Test'):
        for class_name, class_id in (('Fire', 1), ('Smoke', 0), ('Neutral', 0)):
            folder = dataset / split / class_name
            for path in sorted(folder.glob('*')):
                if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                    continue
                rows.append({
                    'image_path': str(path.resolve()), 'source': 'FIRE-SMOKE-DATASET',
                    'split': split.lower(), 'source_class': class_name.lower(),
                    'class_id': class_id, 'has_fire': class_id,
                    'p_fire': None, 'point_supervision': 'none_classification_only',
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--project-root', type=Path, default=root)
    parser.add_argument('--output', type=Path, default=root / 'output' / 'dataset_integration_20261010' / 'external_coarse_manifest.json')
    args = parser.parse_args()
    project = args.project_root.resolve()
    dfire_zip = project / 'datasets' / 'D-Fire.zip'
    dfire_root = project / 'datasets' / 'D-Fire'
    if dfire_zip.is_file():
        rows = _dfire_rows(dfire_zip, project / 'datasets')
    elif dfire_root.is_dir():
        rows = _dfire_rows_from_directory(dfire_root)
    else:
        raise FileNotFoundError(f'Neither extracted D-Fire directory nor archive exists: {dfire_root}')
    rows.extend(_fire_smoke_rows(project / 'datasets' / 'fire_smoke_dataset'))
    payload = {
        'format': 'LAB_SAM.external_coarse_manifest.v1',
        'policy': {
            'external_data_auxiliary_only': True,
            'D_Fire_bottom_center_is_weak_target': True,
            'fire_smoke_classification_only': True,
            'excluded_from_real_validation_and_test': True,
        },
        'counts': {
            'total': len(rows),
            'dfire': sum(row['source'] == 'D-Fire' for row in rows),
            'fire_smoke': sum(row['source'] == 'FIRE-SMOKE-DATASET' for row in rows),
        },
        'records': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'counts': payload['counts'], 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
