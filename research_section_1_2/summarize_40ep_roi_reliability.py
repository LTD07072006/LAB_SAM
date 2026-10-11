from __future__ import annotations
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'output' / 'dataset_integration_20261010'
coarse_history = json.loads((out / 'coarse_fpn_640_hardtail_40ep' / 'history.json').read_text(encoding='utf-8'))
roi_history = json.loads((out / 'roi_finetune_actual_coarse_40ep' / 'history_roi.json').read_text(encoding='utf-8'))
best_coarse = min(coarse_history, key=lambda row: row['val_metric']['mae_px'])
best_roi = min(roi_history, key=lambda row: row['val_metric']['mae_px'])
report = {
    'format': 'LAB_SAM.coarse_roi_40ep_decision.v1',
    'coarse_run': {
        'config': {'epochs': 40, 'image_size': 640, 'freeze_epochs': 3, 'hard_tail_weight': 1.5,
                   'init_checkpoint': str(root / 'fire-model-data' / 'best.pth')},
        'best_epoch': best_coarse['epoch'], 'best_validation': best_coarse['val_metric'],
        'production_baseline': {'mae_px': 4.8981751303522145, 'pck25': 1.0},
        'accepted_for_production': False,
        'checkpoint': str(out / 'coarse_fpn_640_hardtail_40ep' / 'best_detector_2d.pth'),
    },
    'roi_run': {
        'config': {'epochs': 12, 'lr': 1e-5, 'freeze_epochs': 3, 'repeats': 2,
                   'noise_std': 0.04, 'hard_tail_boost': 2},
        'best_epoch': best_roi['epoch'], 'best_validation': best_roi['val_metric'],
        'production_baseline': {'mae_px': 4.700112850877399, 'pck25': 1.0},
        'accepted_for_production': False,
        'checkpoint': str(out / 'roi_finetune_actual_coarse_40ep' / 'best_roi.pth'),
    },
    'production_checkpoints': {
        'coarse': str(root / 'fire-model-data' / 'best.pth'),
        'roi': str(root / 'output' / 'roi_domain_experiments_cpu_regularized' / 'mixed' / 'best_roi.pth'),
    },
    'manual_review_required': {
        'ground_truth_edit': False,
        'review_images': 'Use coarse_issue_manifest/coarse_issues_all.csv; prioritize test rows, raw_error_px >25, high-confidence wrong localization, and ROI/coarse shift.',
        'img_243': 'Do not edit automatically; independent annotation is required before any label revision.',
    },
}
(out / 'coarse_roi_40ep_decision.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
