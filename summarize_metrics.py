import json, pathlib, pprint, math
root=pathlib.Path(r'D:\LAB\SAM_Experiment\output')
paths=['roi_checkpoint_benchmark_20261009/summary.json','priority1_raycasting_20261008b/summary.json','benchmark_ray_vs_scene_mlp_20261009/summary.json','five_workflows/summary.json','low_cost_3d_benchmark_20261008/summary.json','ablation_3d_20261008_v1/summary.json','temporal_filter_ablation_20261008_v1/summary.json','physics_scene_cross_20261008_v2/summary.json','siren_categorical_scene_20261008/summary.json','multiplane_cpu_mixed_20261008/summary.json','workflow_2d_then_3d_20261008/3d_workflows/summary.json','audit_roi_metric_discrepancy_20261009/summary.json']
for rel in paths:
    p=root/rel; d=json.loads(p.read_text(encoding='utf-8'))
    print('\n=== '+rel+' ===')
    if rel.startswith('roi_') or rel.startswith('audit_'):
      print('metrics=',json.dumps(d.get('metrics'),ensure_ascii=True,indent=2)[:12000])
      print('protocol=',json.dumps(d.get('protocol'),ensure_ascii=True,indent=2)[:3000])
    elif rel.startswith('priority'):
      for key in ['ray_count_benchmark','protected_route_benchmark','temporal_benchmark','mesh_noise_benchmark']:
       print(key,'=',json.dumps(d.get(key),ensure_ascii=True,indent=2)[:14000])
    elif rel.startswith('benchmark_ray'):
      print('metrics=',json.dumps(d.get('metrics'),ensure_ascii=True,indent=2)[:10000])
      print('protocol=',json.dumps(d.get('protocol'),ensure_ascii=True,indent=2)[:4000])
    elif rel.startswith('five_') or rel.startswith('workflow_'):
      print('method=',json.dumps(d.get('method'),ensure_ascii=True,indent=2)[:5000])
      print('branches=',json.dumps(d.get('branches'),ensure_ascii=True,indent=2)[:20000])
      print('sequence=',json.dumps(d.get('sequence_stability'),ensure_ascii=True,indent=2)[:12000])
    elif rel.startswith('low_cost'):
      print('branches=',json.dumps(d.get('branches'),ensure_ascii=True,indent=2)[:20000])
      print('triangulation=',json.dumps(d.get('triangulation'),ensure_ascii=True,indent=2)[:6000])
    elif rel.startswith('ablation'):
      for variant,v in d.get('variants',{}).items():
       print(variant,'mlp=',json.dumps(v.get('mlp',{}).get('metrics'),ensure_ascii=True),'tri_raw=',json.dumps(v.get('triangulation',{}).get('metrics_raw'),ensure_ascii=True),'tri_gate=',json.dumps(v.get('triangulation',{}).get('metrics_gated'),ensure_ascii=True))
    elif rel.startswith('temporal'):
      for variant,v in d.get('variants',{}).items(): print(variant,'=',json.dumps(v,ensure_ascii=True)[:8000])
    elif rel.startswith('physics'):
      print('results=',json.dumps(d.get('results'),ensure_ascii=True,indent=2)[:18000])
    elif rel.startswith('siren'):
      print('results=',json.dumps(d.get('results'),ensure_ascii=True,indent=2)[:18000])
    elif rel.startswith('multiplane'):
      for k,v in d.get('metrics',{}).items(): print(k,'=',json.dumps(v,ensure_ascii=True))
