"""Benchmark a validation-selected coarse checkpoint before invoking ROI.

All positives contribute to model coordinate MAE, including detection misses.
ROI-input coverage and accepted-only MAE are reported separately. Error bins
are fixed by the older brightness3 coarse audit, never by the candidate.
This existing test is a development benchmark after repeated experimentation,
not an independent final estimate of deployment performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fire_detector import FireDetector
from project_paths import CCTV_DATASET
from train_week6 import FireDataset, SpatialFireModel, load_records, split_records


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [r for r in rows if r["has_fire"]]
    finite = [r for r in positives if r["error_px"] is not None]
    accepted = [r for r in finite if r["detected"]]
    errors = np.asarray([r["error_px"] for r in finite], dtype=float)
    def metrics(e: np.ndarray) -> dict[str, Any]:
        if not len(e):
            return {"count": 0, "mae_px": None, "p95_px": None, "max_px": None, "pck10": None, "pck25": None, "outliers_gt25": 0}
        return {"count": len(e), "mae_px": float(e.mean()), "p95_px": float(np.percentile(e, 95)),
                "max_px": float(e.max()), "pck10": float(np.mean(e <= 10)),
                "pck25": float(np.mean(e <= 25)), "outliers_gt25": int(np.sum(e > 25))}
    tp = sum(r["detected"] for r in positives)
    fp = sum(r["detected"] for r in rows if not r["has_fire"])
    return {"total": len(rows), "positives": len(positives), "invalid_positive_coordinates": len(positives)-len(finite),
            "tp": tp, "fn": len(positives)-tp, "fp": fp, "tn": len(rows)-len(positives)-fp,
            "input_coverage": len(accepted)/max(1, len(positives)),
            "all_positive_coordinates": metrics(errors),
            "accepted_only": metrics(np.asarray([r["error_px"] for r in accepted])),
            "effective_pck25": sum(r["error_px"] <= 25 for r in accepted)/max(1, len(positives))}


@torch.inference_mode()
def candidate_rows(checkpoint: Path, splits: dict, batch_size: int, device: str) -> dict[str, list[dict]]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("architecture") == "detector_2d_fpn_v2":
        from detector_2d import Detector2DDataset, Detector2DFPNModel, coords_from_letterbox
        model = Detector2DFPNModel(backbone=ck['backbone'], image_size=ck['image_size'],
            fpn_channels=ck['fpn_channels'], temperature=ck['temperature'],
            multiscale_heatmap=ck.get('multiscale_heatmap', False), direct_coordinate_head=ck.get('direct_coordinate_head', False), pretrained=False).to(device)
        model.load_state_dict(ck['model'], strict=True); model.eval()
        output = {}
        for split, records in splits.items():
            values = []
            loader = DataLoader(Detector2DDataset(records, image_size=ck['image_size']), batch_size, shuffle=False)
            for images, targets, original_xy, meta, paths, _ in loader:
                predictions = model(images.to(device))
                probabilities = torch.sigmoid(predictions['confidence_logit']).cpu().numpy()
                xy = coords_from_letterbox(predictions['coord'].cpu(), meta).numpy()
                for path, target, size, gt_xy, probability, point in zip(paths, targets.numpy(), meta[:,5:7].numpy(), original_xy.numpy(), probabilities, xy):
                    valid = bool(np.isfinite(point).all() and np.isfinite(probability))
                    gt = gt_xy * size; pixel = point * size
                    values.append({'image_path':path, 'split':split, 'has_fire':bool(target[0]),
                        'confidence':float(probability) if np.isfinite(probability) else None,
                        'detected':bool(valid and probability>=0.5), 'pixel':pixel.tolist() if valid else None,
                        'ground_truth_pixel':gt.tolist() if target[0] else None,
                        'error_px':float(np.linalg.norm(pixel-gt)) if valid and target[0] else None,
                        'width':int(size[0]), 'height':int(size[1])})
            output[split] = values
        return output
    if ck["architecture"] != "spatial_heatmap_v2":
        raise ValueError("Expected spatial_heatmap_v2 checkpoint")
    model = SpatialFireModel(backbone=ck["backbone"], pretrained=False).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    output = {}
    for split, records in splits.items():
        rows = []
        loader = DataLoader(FireDataset(records, image_size=tuple(ck["image_size"])), batch_size, shuffle=False)
        for images, targets, sizes, paths, _ in loader:
            result = model(images.to(device)); conf = torch.sigmoid(result["confidence_logit"]).cpu().numpy()
            coord = result["coord"].cpu().numpy()
            for p, t, size, probability, xy in zip(paths, targets.numpy(), sizes.numpy(), conf, coord):
                valid = bool(np.isfinite(xy).all() and np.isfinite(probability))
                pixel = xy * size; gt = t[1:] * size
                rows.append({"image_path": p, "split": split, "has_fire": bool(t[0]),
                             "confidence": float(probability) if np.isfinite(probability) else None,
                             "detected": bool(valid and probability >= 0.5),
                             "pixel": pixel.tolist() if valid else None,
                             "ground_truth_pixel": gt.tolist() if t[0] else None,
                             "error_px": float(np.linalg.norm(pixel-gt)) if valid and t[0] else None,
                             "width": int(size[0]), "height": int(size[1])})
        output[split] = rows
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accept-mae", type=float, default=9.6270)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, stats = load_records(ROOT / "fire-model-data/dataset_labels (1).json", CCTV_DATASET)
    splits = split_records(records, 42)
    # train is a diagnostic only; val selected the weights; test never updates them.
    rows = candidate_rows(args.checkpoint, splits, args.batch_size, args.device)
    audit_file = ROOT / "output/section_1_2_pre_roi_20261010/coarse_issue_manifest/coarse_issue_manifest.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    by_path = {str(Path(r["image_path"]).resolve()).lower(): r for r in audit["records"]}
    for values in rows.values():
        for row in values:
            a = by_path.get(str(Path(row["image_path"]).resolve()).lower(), {})
            error = a.get("raw_error_px")
            row["baseline_tta_error_px"] = error
            row["baseline_error_bin"] = "missing" if error is None else "0-5" if error<=5 else "5-10" if error<=10 else "10-20" if error<=20 else ">20"
            gt = row["ground_truth_pixel"]
            row["region"] = None if gt is None else f"x{min(2,int(gt[0]/row['width']*3))}_y{min(2,int(gt[1]/row['height']*3))}"
    report: dict[str, Any] = {"protocol": {"seed": 42, "split": "source-test + train frame-window holdout", "sequence_id_available": False,
        "selection": "validation only", "test_role": "existing development benchmark, requires fresh sequence holdout for deployment claim",
        "coarse_only": True, "tta": "none", "temporal_guard": False,
        "mae_definition": "mean Euclidean pixel distance on every positive, including detection misses",
        "error_bins": "fixed older brightness3 coarse errors", "roi_executed": False}, "load_stats": stats,
        "checkpoint": str(args.checkpoint.resolve()), "metrics": {s: summarize(v) for s,v in rows.items()},
        "bins": {s: {b: summarize([r for r in v if r['baseline_error_bin']==b and r['has_fire']]) for b in ('0-5','5-10','10-20','>20')} for s,v in rows.items()},
        "regions": {s: {reg: summarize([r for r in v if r['region']==reg]) for reg in sorted({r['region'] for r in v if r['region']})} for s,v in rows.items()}}
    test = report["metrics"]["test"]; mae = test["all_positive_coordinates"]["mae_px"]
    meets = mae is not None and math.isfinite(mae) and mae <= args.accept_mae and test["invalid_positive_coordinates"] == 0
    # This is the user's acceptance rule, not a target used to select epochs.
    report["decision"] = {"threshold_mae_px": args.accept_mae, "meets_requested_coarse_mae": meets,
        "status": "accepted_for_roi_trial" if meets else "rejected_for_roi", "production_overwritten": False,
        "warning": "Compare recall and outliers separately; coarse MAE does not prove end-to-end ROI improvement."}
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    export = {k: source[k] for k in ('architecture','backbone','model','epoch','val_loss','image_size','split_info')}
    for key in ('temperature','fpn_channels','multiscale_heatmap','heatmap_size','mean','std'):
        if key in source:
            export[key] = source[key]
    export['benchmark_decision'] = report['decision']; export['selection_metric']='validation_mae'
    export['direct_coordinate_head'] = bool(source.get('direct_coordinate_head', False))
    export_path = args.output_dir / ("coarse_before_roi.pth" if meets else "coarse_candidate_rejected.pth")
    torch.save(export, export_path)
    # Confirm the public upstream adapter can load the exact exported checkpoint.
    detector = FireDetector(export_path, device=args.device, tta_mode="none")
    probe = detector.detect(splits['test'][0].image_path)
    source_detector = FireDetector(args.checkpoint, device=args.device, tta_mode='none')
    source_probe = source_detector.detect(splits['test'][0].image_path)
    if probe.pixel != source_probe.pixel or abs(probe.confidence-source_probe.confidence)>1e-6:
        raise RuntimeError('Exported model predictions differ from the source checkpoint')
    report['export'] = {"path": str(export_path.resolve()), "sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
                       "adapter_load_verified": True, "probe_detected": probe.detected, "probe_pixel": probe.pixel}
    manifest = {r['image_path']: {"point": r['pixel'] if r['detected'] else None, "raw_pixel": r['pixel'],
                                "confidence": r['confidence'], "detected": r['detected'], "size": [r['width'],r['height']],
                                "checkpoint": str(export_path.resolve()), "split": r['split']} for v in rows.values() for r in v}
    (args.output_dir / 'actual_coarse_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (args.output_dir / 'per_image.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    (args.output_dir / 'benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    csv_rows = [{k: r[k] for k in ('image_path','split','has_fire','confidence','detected','error_px','baseline_tta_error_px','baseline_error_bin','region')} for v in rows.values() for r in v]
    with (args.output_dir / 'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({"metrics":report['metrics'],"decision":report['decision'],"export":report['export']},indent=2))


if __name__ == '__main__': main()
