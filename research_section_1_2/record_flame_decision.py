from __future__ import annotations
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'output' / 'dataset_integration_20261010'
history = json.loads((out / 'cctv_finetune_from_flame' / 'history.json').read_text(encoding='utf-8'))
best = min(history, key=lambda row: row['val_metric']['mae_px'])
decision = {
    'format': 'LAB_SAM.FLAME_training_decision.v1',
    'flame_pretrain': str(out / 'flame_pretrain' / 'best_flame_pretrained.pth'),
    'cctv_finetune': str(out / 'cctv_finetune_from_flame' / 'best_spatial.pth'),
    'flame_samples': 512,
    'hard_tail_train_samples': 24,
    'best_epoch': best['epoch'],
    'best_validation': best['val_metric'],
    'baseline_validation': {'mae_px': 4.8981751303522145, 'pck25': 1.0},
    'accepted_for_production': False,
    'production_checkpoint': str(root / 'fire-model-data' / 'best.pth'),
    'reason': 'FLAME is image-level classification only; pilot representation transfer regressed CCTV point localization severely.',
    'policy': 'Keep FLAME checkpoint for research. Do not use FLAME to create p_fire labels or replace the CCTV checkpoint.'
}
(out / 'flame_training_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
print(json.dumps(decision, indent=2))
