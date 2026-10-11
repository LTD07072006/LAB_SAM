import json, pathlib
root=pathlib.Path(r'D:\LAB\SAM_Experiment\output')
files=[
'roi_checkpoint_benchmark_20261009/summary.json',
'priority1_raycasting_20261008b/summary.json',
'benchmark_ray_vs_scene_mlp_20261009/summary.json',
'five_workflows/summary.json',
'low_cost_3d_benchmark_20261008/summary.json',
'ablation_3d_20261008_v1/summary.json',
'temporal_filter_ablation_20261008_v1/summary.json',
'physics_scene_cross_20261008_v2/summary.json',
'siren_categorical_scene_20261008/summary.json',
'multiplane_cpu_mixed_20261008/summary.json',
'workflow_2d_then_3d_20261008/3d_workflows/summary.json',
'audit_roi_metric_discrepancy_20261009/summary.json'
]
for rel in files:
 p=root/rel
 print('\n===',rel,'exists',p.exists(),'===')
 if not p.exists(): continue
 try: d=json.loads(p.read_text(encoding='utf-8'))
 except Exception as e: print(e); continue
 print('top keys:',list(d)[:40])
 for k,v in d.items():
  if isinstance(v,(str,int,float,bool)) or v is None:
   print(k,':',v)
  elif isinstance(v,dict):
   print(k,': dict keys',list(v)[:20])
  elif isinstance(v,list): print(k,': list len',len(v))
