import json, pathlib, re, statistics
root=pathlib.Path(r'D:\LAB\SAM_Experiment')
outs=root/'output'

def load(rel): return json.loads((outs/rel).read_text(encoding='utf-8'))

def p(title,v): print(title, json.dumps(v, ensure_ascii=False, indent=2, default=str))
# selected metrics compact
roi=load('roi_checkpoint_benchmark_20261009/summary.json')
print('ROI')
for k in ['coarse','week6_baseline','standard_real','standard_mixed','cpu_regularized_mixed']:
 m=roi['metrics'][k]; print(k, {x:m.get(x) for x in ['mae_px','median_px','p95_px','pck10','pck25']})

five=load('five_workflows/summary.json')
print('\nFIVE')
for k,v in five['branches'].items():
 print(k, {x:v.get(x) for x in ['three_d_samples','three_d_mae_m','three_d_median_m','three_d_p95_m','under_0.25m','under_0.50m','location_success_rate','ray_hit_rate','latency_ms']})
print('seq')
for k,v in five.get('sequence_stability',{}).items(): print(k,v)

low=load('low_cost_3d_benchmark_20261008/summary.json')
print('\nLOW')
for k,v in low['branches'].items():
 print(k,{x:v.get(x) for x in ['samples','three_d_mae_m','three_d_median_m','three_d_p95_m','under_0.25m','under_0.50m','valid_rate','location_success_rate','latency_ms']})
print('tri')
for k,v in low['triangulation'].items():
 print(k, v)

ab=load('ablation_3d_20261008_v1/summary.json')
print('\nABLATION')
for k,v in ab['variants'].items():
 print(k,'mlp',v['mlp']['metrics'],'tri_raw',v['triangulation']['metrics_raw'],'tri_gate',v['triangulation']['metrics_gated'])

# priority only selected nested
pri=load('priority1_raycasting_20261008b/summary.json')
print('\nPRIORITY')
for cond in ['clean_true','noisy_true','clean_estimated','noisy_estimated']:
 print(cond)
 for rays in ['1','3','5','7']:
  x=pri['ray_count_benchmark'][rays][cond] if rays in pri['ray_count_benchmark'] else None
  if x: print('rays',rays,{z:x['metrics'].get(z) for z in ['valid_rate','mae_m','median_m','p95_m','under_0.25m','under_0.50m']}, 'lat',x.get('latency_ms',{}))
 print('protected', {k: v for k,v in pri['protected_route_benchmark'][cond].items() if k in ['ray_only','protected_route','ray_protection_rate','sources','route_policy']})

# depth json
for fn in ['depth_anything_v2_metric_indoor_test_visible92.json','depth_anything_v2_metric_indoor_roi_test92.json','depth_anything_v2_metric_indoor_test24.json','depth_anything_v2_metric_indoor_roi_test24.json']:
 p=root/'working'/'synthetic_fire_3d_v3'/fn
 if p.exists():
  d=json.loads(p.read_text(encoding='utf-8'))
  print('\nDEPTH',fn,'keys',list(d))
  for k,v in d.items():
   if isinstance(v,(int,float,str,bool)) or v is None: print(k,v)
   elif isinstance(v,dict):
    # recursively show likely metrics keys shallow
    print(k, {kk:vv for kk,vv in v.items() if isinstance(vv,(int,float,str,bool)) or vv is None})
