import json, pathlib
p=pathlib.Path(r'D:\LAB\SAM_Experiment\output\physics_scene_cross_20261008_v2\summary.json')
d=json.loads(p.read_text(encoding='utf-8'))
for model, md in d['results'].items():
 print('\nMODEL',model)
 tc=md.get('test_conditions',{})
 for cond, val in tc.items():
  if isinstance(val,dict):
   print('COND',cond)
   for k in ['sample_count','mae_m','median_m','p95_m','valid_rate','under_0.25m','under_0.50m','latency_ms']:
    if k in val: print(k, val[k])
   for branch in ['scene_mlp','ray','ipm_floor_only','route','fusion']:
    if branch in val and isinstance(val[branch],dict):
     print(branch,{k:val[branch].get(k) for k in ['mae_m','median_m','p95_m','valid_rate','under_0.25m','under_0.50m','latency_ms']})
