import json, pathlib
p=pathlib.Path(r'D:\LAB\SAM_Experiment\output\multiplane_cpu_mixed_20261008\summary.json'); d=json.loads(p.read_text(encoding='utf-8'))
for k,v in d['metrics'].items():
 if k in ['roi_blend:floor_ipm','roi_blend:surface_oracle','roi_blend:plane_bank_nearest','roi_blend:plane_bank_visible','roi_blend:plane_bank_visible_floor_fallback','roi_blend:floor_ipm_ema','roi_blend:plane_bank_visible_floor_fallback_ema','coarse:floor_ipm','coarse:plane_bank_visible_floor_fallback']:
  if isinstance(v,dict):
   print(k,{x:v.get(x) for x in ['records','valid','success_rate','mae_m','median_m','p95_m','under_0.10m','under_0.25m','under_0.50m','mean_latency_ms','p95_latency_ms']})
