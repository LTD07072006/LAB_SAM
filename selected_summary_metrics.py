import json, pathlib, statistics
root=pathlib.Path(r'D:\LAB\SAM_Experiment\output')
for rel in ['benchmark_ray_vs_scene_mlp_20261009/summary.json','five_workflows/summary.json','low_cost_3d_benchmark_20261008/summary.json','physics_scene_cross_20261008_v2/summary.json','siren_categorical_scene_20261008/summary.json','temporal_filter_ablation_20261008_v1/summary.json']:
 d=json.loads((root/rel).read_text(encoding='utf-8')); print('\n',rel)
 if 'metrics' in d and rel.startswith('benchmark_ray'):
  for k,v in d['metrics'].items(): print(k, json.dumps(v,ensure_ascii=False))
 if rel.startswith('physics'):
  for model,v in d['results'].items():
   print(model, 'conditions=',list(v.keys()) if isinstance(v,dict) else type(v))
   if isinstance(v,dict):
    for cond, met in v.items():
     if isinstance(met,dict): print(' ',cond, {k:met.get(k) for k in ['mae_m','median_m','p95_m','valid_rate','under_0.25m','under_0.50m','latency_ms'] if k in met})
 if rel.startswith('siren'):
  for model,v in d['results'].items():
   print(model, json.dumps(v,ensure_ascii=False)[:12000])
 if rel.startswith('temporal'):
  for variant,v in d['variants'].items():
   print(variant,json.dumps(v,ensure_ascii=False)[:6000])
