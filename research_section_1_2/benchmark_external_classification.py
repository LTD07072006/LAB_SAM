"""Evaluate the existing coarse detector on the newly downloaded domain data.

This is a classification/domain-shift check only. The external datasets do
not carry the CCTV fire-base point, so this script never reports pixel MAE.
"""
from __future__ import annotations

import argparse
import json
import zipfile
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fire_detector import FireDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--checkpoint', type=Path, default=root / 'fire-model-data' / 'best.pth')
    parser.add_argument('--dataset-root', type=Path, default=root / 'datasets' / 'fire_smoke_dataset' / 'FIRE-SMOKE-DATASET')
    parser.add_argument('--max-per-class', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=root / 'output' / 'dataset_integration_20261010' / 'external_classification_benchmark.json')
    args = parser.parse_args()
    detector = FireDetector(args.checkpoint, device=args.device, threshold=0.5, use_amp=False, tta_mode='none')
    rows = []
    for split in ('Train', 'Test'):
        for cls, label in (('Fire', 1), ('Smoke', 0), ('Neutral', 0)):
            paths = sorted((args.dataset_root / split / cls).glob('*'))[:max(0, args.max_per_class)]
            for path in paths:
                try:
                    image = Image.open(path).convert('RGB')
                    result = detector.detect(image)
                    rows.append({'split': split.lower(), 'class': cls.lower(), 'label': label,
                                 'confidence': result.confidence, 'predicted_fire': int(result.detected),
                                 'image': str(path)})
                except (OSError, ValueError):
                    continue
    y = np.asarray([row['label'] for row in rows], dtype=np.int64)
    p = np.asarray([row['predicted_fire'] for row in rows], dtype=np.int64)
    report = {'format': 'LAB_SAM.external_classification_benchmark.v1',
              'protocol': {'pixel_metrics': False, 'point_labels_available': False,
                           'threshold': detector.threshold},
              'count': len(rows), 'accuracy': float(np.mean(y == p)) if len(y) else None,
              'by_split_class': {}, 'rows': rows}
    for key in sorted({(row['split'], row['class']) for row in rows}):
        mask = np.asarray([(r['split'], r['class']) == key for r in rows])
        report['by_split_class'][f'{key[0]}_{key[1]}'] = {'count': int(mask.sum()),
            'positive_rate': float(p[mask].mean()) if mask.any() else None,
            'accuracy': float((y[mask] == p[mask]).mean()) if mask.any() else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('count', 'accuracy', 'by_split_class')}, indent=2))


if __name__ == '__main__':
    main()
