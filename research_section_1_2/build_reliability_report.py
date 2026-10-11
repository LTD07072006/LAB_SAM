from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
manifest_path = root / 'output' / 'section_1_2_pre_roi_20261010' / 'coarse_issue_manifest' / 'coarse_issues_all.json'
rows = json.loads(manifest_path.read_text(encoding='utf-8'))
def finite(row, key):
    try:
        x=float(row.get(key)); return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

issues=[]
for row in rows:
    error=finite(row,'raw_error_px'); confidence=finite(row,'confidence'); stability=finite(row,'stability_px')
    if row.get('workflow_split') == 'test' and (error is not None and error > 25 or 'high_confidence_wrong_localization' in row.get('categories', [])):
        issues.append({**row,'manual_priority':'P0_test_outlier'})
    elif row.get('workflow_split') == 'test' and (stability is not None and stability >= 6 or error is not None and error > 15):
        issues.append({**row,'manual_priority':'P1_test_uncertain'})
for row in rows:
    if row.get('workflow_split') == 'train' and 'high_confidence_wrong_localization' in row.get('categories', []):
        row_copy={**row,'manual_priority':'P2_train_audit'}
        if not any(x.get('image_path') == row_copy.get('image_path') for x in issues): issues.append(row_copy)

errors=np.array([finite(r,'raw_error_px') for r in rows if finite(r,'raw_error_px') is not None])
conf=np.array([finite(r,'confidence') for r in rows if finite(r,'confidence') is not None])
stability=np.array([finite(r,'stability_px') for r in rows if finite(r,'stability_px') is not None])
paired=[(finite(r,'confidence'),finite(r,'raw_error_px')) for r in rows if finite(r,'confidence') is not None and finite(r,'raw_error_px') is not None]
correlation=float(np.corrcoef(np.array(paired).T)[0,1]) if len(paired)>1 else None
report={'format':'LAB_SAM.reliability_report.v1', 'counts':{
    'positive_records':len(rows), 'manual_review_rows':len(issues),
    'manual_test_rows':sum(x.get('workflow_split')=='test' for x in issues),
    'test_gt25_rows':sum(x.get('workflow_split')=='test' and (finite(x,'raw_error_px') or 0)>25 for x in rows),
}, 'statistics':{
    'raw_error_mae_px':float(errors.mean()) if len(errors) else None,
    'raw_error_p95_px':float(np.percentile(errors,95)) if len(errors) else None,
    'confidence_error_pearson':correlation,
    'stability_median_px':float(np.median(stability)) if len(stability) else None,
    'stability_ge6_count':int(np.sum(stability>=6)) if len(stability) else 0,
}, 'policy':{
    'video_ordered_frames':'temporal guard allowed with raw/guarded/reason logging',
    'independent_or_shuffled_images':'no temporal guard; use uncertainty/hard gate',
    'ground_truth_edit':'manual dual annotation only; never automatic',
}, 'manual_review':sorted(issues,key=lambda x:({'P0_test_outlier':0,'P1_test_uncertain':1,'P2_train_audit':2}.get(x.get('manual_priority'),9), -(finite(x,'raw_error_px') or 0)))}
out=root/'output'/'dataset_integration_20261010'; out.mkdir(parents=True,exist_ok=True)
(out/'reliability_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
fields=['manual_priority','image','image_path','workflow_split','raw_error_px','guarded_error_px','confidence','stability_px','guard_replaced','categories']
with (out/'manual_review_queue.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    for row in report['manual_review']:
        row=dict(row); row['categories']=';'.join(row.get('categories',[])); writer.writerow({key:row.get(key) for key in fields})
print(json.dumps({'counts':report['counts'],'statistics':report['statistics']},indent=2))
