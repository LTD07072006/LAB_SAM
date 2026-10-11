"""Leakage-safe coarse curriculum: positive heatmap warm-up -> 640 fine stage.

Only workflow-train positive images update the model. Validation/test and the
older audit manifest are never used as labels; the manifest contributes only
train sample weights derived from its raw coarse error bins.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from detector_2d import (Detector2DDataset, Detector2DFPNModel, Detector2DLoss,
    load_backbone_initialization, run_epoch, save_checkpoint)
from train_week6 import load_records, split_records
from project_paths import CCTV_DATASET

ROOT=Path(__file__).resolve().parent

def aliases(value: str) -> list[str]:
    s=str(value).replace('\\','/').lower(); out=[s]
    if '/img_data/' in s: out.append(s[s.index('/img_data/')+1:])
    if '/' in s: out.append(s.rsplit('/',1)[-1])
    return list(dict.fromkeys(out))

def manifest_index(path: Path) -> dict[str,dict]:
    raw=json.loads(path.read_text(encoding='utf-8')); rows=raw.get('records',raw) if isinstance(raw,dict) else raw; out={}
    for row in rows if isinstance(rows,list) else []:
        if isinstance(row,dict) and row.get('image_path'):
            for key in aliases(row['image_path']): out.setdefault(key,row)
    return out

def weights_for(records, manifest):
    weights={}
    for r in records:
        if not r.has_fire: continue
        row=next((manifest[k] for k in aliases(r.image_path) if k in manifest),{})
        try: error=float(row.get('raw_error_px',0.0))
        except (TypeError,ValueError): error=0.0
        # Explicit bins: hard tail gets a larger, bounded multiplier.
        weight=1.0 if error<=5 else 1.5 if error<=10 else 2.0 if error<=20 else 3.0
        # Region-aware emphasis is train-only and derived from the immutable label.
        if r.x_norm < 1/3 and r.y_norm >= 2/3: weight*=1.5
        weights[str(Path(r.image_path).resolve()).lower()]=weight
    return weights

def positive_loader(records, image_size, weights, augment, batch):
    return DataLoader(Detector2DDataset([r for r in records if r.has_fire], train=True,
        image_size=(image_size,image_size), flip_p=0.0, sample_weights=weights, augment=augment),
        batch_size=batch, shuffle=True, num_workers=0)

def eval_loader(records, image_size, batch):
    return DataLoader(Detector2DDataset(records, train=False,image_size=(image_size,image_size),augment=False),
        batch_size=batch,shuffle=False,num_workers=0)

def train_stage(model, train_loader, val_loader, device, epochs, freeze, args, out):
    for p in model.backbone.parameters(): p.requires_grad=not freeze
    for p in model.class_head.parameters(): p.requires_grad=not freeze
    head=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    groups=[{'params':head,'lr':args.head_lr}]
    if not freeze: groups.append({'params':model.backbone.parameters(),'lr':args.backbone_lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,epochs),eta_min=args.head_lr/100)
    criterion=Detector2DLoss(lambda_coord=args.lambda_coord,lambda_heatmap=0.1,
        heatmap_sigma=args.heatmap_sigma,confidence_weight=0.0 if freeze else 1.0,distribution_weight=args.distribution_weight,temperature=args.temperature,
        direct_weight=args.direct_weight,consistency_weight=args.consistency_weight)
    history=[]; best=float('inf'); best_state=None
    for epoch in range(1,epochs+1):
        model.train();
        if freeze: model.backbone.eval()
        train_loss,train_metric=run_epoch(model,train_loader,criterion,device,optimizer,threshold=.5)
        val_loss,val_metric=run_epoch(model,val_loader,criterion,device,threshold=.5)
        scheduler.step()
        row={'epoch':epoch,'train':train_loss,'val':val_loss,'train_metric':train_metric,'val_metric':val_metric}
        history.append(row); print(f'{out.name} epoch={epoch} val_mae={val_metric["mae_px"]:.2f} pck25={val_metric["pck25"]:.3f}',flush=True)
        score=float(val_metric['mae_px'])
        if np.isfinite(score) and score<best:
            best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save_checkpoint(out/'best_detector_2d.pth',model,optimizer,scheduler,epoch,float(val_loss['total']),{'stage':out.name},args)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    if best_state is not None: model.load_state_dict(best_state,strict=True)
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return model,history,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--device',default='cpu')
    ap.add_argument('--stage320-epochs',type=int,default=0); ap.add_argument('--stage640-epochs',type=int,default=5); ap.add_argument('--unfreeze-epochs',type=int,default=35)
    ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--head-lr',type=float,default=3e-4); ap.add_argument('--backbone-lr',type=float,default=1e-5)
    ap.add_argument('--lambda-coord',type=float,default=20.0); ap.add_argument('--heatmap-sigma',type=float,default=2.0); ap.add_argument('--distribution-weight',type=float,default=.5); ap.add_argument('--temperature',type=float,default=.2)
    ap.add_argument('--direct-weight',type=float,default=10.0); ap.add_argument('--consistency-weight',type=float,default=2.0)
    args=ap.parse_args(); args.selection_metric='mae'; random.seed(42);np.random.seed(42);torch.manual_seed(42);device=torch.device(args.device); args.output_dir.mkdir(parents=True,exist_ok=True)
    labels=ROOT/'fire-model-data/dataset_labels (1).json'; records,_=load_records(labels,CCTV_DATASET); splits=split_records(records,42)
    manifest=manifest_index(ROOT/'output/section_1_2_pre_roi_20261010/coarse_issue_manifest/coarse_issue_manifest.json'); weights=weights_for(splits['train'],manifest)
    model=Detector2DFPNModel(image_size=(320,320),fpn_channels=64,temperature=args.temperature,multiscale_heatmap=True,direct_coordinate_head=True,pretrained=False); transfer=load_backbone_initialization(model,ROOT/'fire-model-data/best.pth'); model.to(device)
    stage320=args.output_dir/'stage320'; stage320.mkdir(parents=True,exist_ok=True)
    train320=positive_loader(splits['train'],320,weights,False,args.batch_size); val320=eval_loader(splits['val'],320,args.batch_size)
    model,h320,b320=train_stage(model,train320,val320,device,args.stage320_epochs,True,args,stage320)
    stage640=args.output_dir/'stage640'; stage640.mkdir(parents=True,exist_ok=True)
    # Same head weights, new output grid. Spatial layers are resolution agnostic.
    state=model.state_dict(); model640=Detector2DFPNModel(image_size=(640,640),fpn_channels=64,temperature=args.temperature,multiscale_heatmap=True,direct_coordinate_head=True,pretrained=False); model640.load_state_dict(state,strict=True); model640.to(device)
    train640=positive_loader(splits['train'],640,weights,False,args.batch_size); val640=eval_loader(splits['val'],640,args.batch_size)
    model640,h640,b640=train_stage(model640,train640,val640,device,args.stage640_epochs,True,args,stage640)
    if args.unfreeze_epochs>0:
        stage640uf=args.output_dir/'stage640_unfrozen'; stage640uf.mkdir(parents=True,exist_ok=True)
        # Joint stage includes no-fire images; positive-only training must not
        # turn the confidence branch into an always-fire classifier.
        joint=DataLoader(Detector2DDataset(splits['train'],train=True,image_size=(640,640),
            flip_p=0.0,sample_weights=weights,augment=False),args.batch_size,shuffle=True,num_workers=0)
        model640,huf,buf=train_stage(model640,joint,val640,device,args.unfreeze_epochs,False,args,stage640uf)
        best_path=stage640uf/'best_detector_2d.pth'
    else: best_path=stage640/'best_detector_2d.pth'
    report={'protocol':{'warmup_positive_only':True,'joint_stage_includes_no_fire':True,'classification_frozen_warmup':True,'augmentation_stage320':False,'augmentation_stage640':False,'hard_tail_train_only':True,'val_test_not_used_for_updates':True,'roi_executed':False},'counts':{k:len(v) for k,v in splits.items()},'transfer':transfer,'best_val_mae':buf if args.unfreeze_epochs else b640,'selected_checkpoint':str(best_path.resolve()),'stage320_best_val_mae':b320 if h320 else None,'stage640_best_val_mae':b640,'acceptance':{'target_test_mae_px':9.627,'status':'pending_test_benchmark'}}
    (args.output_dir/'curriculum_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
