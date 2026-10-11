import json, pathlib
p=pathlib.Path(r'D:\LAB\SAM_Experiment\output\temporal_filter_ablation_20261008_v1\summary.json'); d=json.loads(p.read_text(encoding='utf-8'))
for cond,v in d['variants'].items():
 print('\n',cond)
 for k in ['raw','ema','ekf','selected']:
  if k not in v: continue
  x=v[k]
  if isinstance(x,dict) and 'metrics' in x:
   m=x['metrics']; st=x.get('stability',{})
   print(k,{z:m.get(z) for z in ['mae_m','median_m','p95_m','under_0.25m','under_0.50m','valid_rate']}, {z:st.get(z) for z in ['mean_step_m','median_step_m','p95_step_m','jump_rate']})
