import json, pathlib, statistics, math
p=pathlib.Path(r'D:\LAB\SAM_Experiment\output\roi_checkpoint_benchmark_20261009\summary.json')
d=json.loads(p.read_text(encoding='utf-8'))
for name in ['cpu_regularized_mixed','week6_baseline']:
 rows=d['per_image']
 # inspect keys first
 print(name,'first keys',rows[0].keys())
 break
# likely per image has all checkpoint values
for r in d['per_image']:
 if 'img_243' in str(r): print('IMG243',json.dumps(r,indent=2,ensure_ascii=False))
# print max errors for cpu and coarse/refined fields
for name in ['coarse','week6_baseline','standard_real','standard_mixed','cpu_regularized_mixed']:
 vals=[]
 for r in d['per_image']:
  entry=r.get(name)
  if isinstance(entry,dict):
   for key in ['error_px','pixel_error_px','refined_error_px','mae_px']:
    if key in entry:
      vals.append((float(entry[key]),r.get('image_name',r.get('sample_id')),entry)); break
 if vals:
  print(name,'n',len(vals),'top')
  for item in sorted(vals,reverse=True)[:5]: print(item[0],item[1])
