"""Sixteen training images only: diagnostic overfit, never an accuracy claim.

Backbone features are cached in eval mode. This is exactly head-only learning;
no validation/test image or label contributes to the smoke optimizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector_2d import (Detector2DDataset, Detector2DFPNModel, Detector2DLoss,
    coords_from_letterbox, load_backbone_initialization, letterbox_image, point_to_letterbox)
from train_week6 import load_records, split_records, seed_everything
from project_paths import CCTV_DATASET


def geometry_check() -> dict:
    worst = 0.0
    for size in ((224,224),(317,191),(191,317),(641,479)):
        _, meta = letterbox_image(Image.new('RGB', size), (640,640))
        xy = torch.tensor([[0.0,0.0],[0.5,0.5],[0.99,0.99],[0.03,0.90]])
        transformed = torch.tensor([point_to_letterbox(p,meta) for p in xy])
        inverse = coords_from_letterbox(transformed,meta.as_tensor().repeat(len(xy),1))
        worst = max(worst,float(((inverse-xy)*torch.tensor(size)).abs().max()))
    if worst > 0.001:
        raise AssertionError(f'Letterbox roundtrip failed: {worst}')
    return {'roundtrip_max_px':worst, 'note':'Uses existing project normalized-coordinate convention; labels unchanged.'}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=160)
    ap.add_argument('--image-size',type=int,default=640)
    args=ap.parse_args(); seed_everything(42); args.output_dir.mkdir(parents=True,exist_ok=True)
    records,_=load_records(ROOT/'fire-model-data/dataset_labels (1).json',CCTV_DATASET)
    splits=split_records(records,42); positives=sorted([r for r in splits['train'] if r.has_fire],key=lambda r:(r.y_norm,r.x_norm,r.image_path))
    selected=[positives[i] for i in np.linspace(0,len(positives)-1,16,dtype=int)]
    assert all(r.image_path not in {p.image_path for s in ('val','test') for p in splits[s]} for r in selected)
    model=Detector2DFPNModel(image_size=(args.image_size,)*2,fpn_channels=32,temperature=0.2,multiscale_heatmap=True,direct_coordinate_head=True,pretrained=False)
    transfer=load_backbone_initialization(model,ROOT/'fire-model-data/best.pth')
    for p in model.backbone.parameters(): p.requires_grad=False
    model.eval(); cache=[]
    with torch.no_grad():
        for images,target,original,meta,paths,weights in DataLoader(Detector2DDataset(selected,image_size=model.image_size),2,False):
            cache.append((model.backbone(images),target,original,meta,paths))
    # Concatenate all 16 cached samples. Every smoke step then optimizes the
    # complete diagnostic set instead of cycling through one of eight tiny
    # batches, which otherwise makes edge samples oscillate.
    all_features=[torch.cat([item[0][j] for item in cache],dim=0) for j in range(len(cache[0][0]))]
    all_targets=torch.cat([item[1] for item in cache],dim=0)
    all_original=torch.cat([item[2] for item in cache],dim=0)
    all_meta=torch.cat([item[3] for item in cache],dim=0)
    all_paths=[path for item in cache for path in item[4]]
    cache=[(all_features,all_targets,all_original,all_meta,all_paths)]
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    criterion=Detector2DLoss(lambda_coord=20,lambda_heatmap=0.1,heatmap_sigma=2,
        confidence_weight=0.2,distribution_weight=1,temperature=model.temperature,direct_weight=10,consistency_weight=2)
    history=[]
    def evaluate(step: int) -> list[dict]:
        model.eval(); rows=[]
        with torch.no_grad():
            for features,target,original,meta,paths in cache:
                result=model.forward_heads(features)
                xy=coords_from_letterbox(result['coord'],meta)*meta[:,5:7]; gt=original*meta[:,5:7]
                for path,point,label,confidence in zip(paths,xy,gt,torch.sigmoid(result['confidence_logit'])):
                    rows.append({'image_path':path,'pixel':point.tolist(),'ground_truth_pixel':label.tolist(),
                        'error_px':float(torch.linalg.vector_norm(point-label)), 'confidence':float(confidence)})
        errors=np.array([r['error_px'] for r in rows]); item={'step':step,'mae_px':float(errors.mean()),'max_px':float(errors.max()),'pck25':float(np.mean(errors<=25))}
        history.append(item); print(json.dumps(item),flush=True); return rows
    initial=evaluate(0)
    for step in range(1,args.steps+1):
        features,target,*_=cache[0]; model.train();model.backbone.eval()
        result=model.forward_heads(features); loss=criterion(result,target)['total']
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%40==0: evaluate(step)
    rows=evaluate(args.steps)
    accepted=history[-1]['mae_px']<=9.627 and history[-1]['pck25']==1.0
    report={'protocol':'train-only diagnostic overfit; no held-out generalization claim',
        'counts':{'train_smoke':16,'val_used':0,'test_used':0},'image_size':model.image_size,
        'geometry':geometry_check(),'backbone_transfer':transfer,'history':history,'per_image':rows,
        'gate':{'mae_limit_px':9.627,'pck25_required':1.0,'passed':accepted,'long_run_allowed':accepted}}
    # Explicit smoke naming prevents accidentally promoting an overfit model.
    torch.save({'architecture':'detector_2d_fpn_v2','backbone':model.backbone_name,'model':model.state_dict(),
        'image_size':model.image_size,'fpn_channels':model.fpn_channels,'temperature':model.temperature,
        'multiscale_heatmap':True,'smoke_only':True},args.output_dir/'smoke_only.pth')
    (args.output_dir/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(4*280,4*280),'white'); draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        with Image.open(r['image_path']) as im:
            im=im.convert('RGB');w,h=im.size;im=im.resize((260,240));x=(i%4)*280;y=(i//4)*280;sheet.paste(im,(x,y))
        for key,color in (('ground_truth_pixel','lime'),('pixel','red')):
            px,py=r[key];px=x+px/w*260;py=y+py/h*240;draw.ellipse((px-4,py-4,px+4,py+4),outline=color,width=2)
        draw.text((x,y+245),f"{Path(r['image_path']).name}: {r['error_px']:.2f}px",fill='black')
    sheet.save(args.output_dir/'smoke_contact_sheet.jpg')
    print(json.dumps(report['gate']),flush=True)


if __name__=='__main__': main()
