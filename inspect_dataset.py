import json, pathlib
root=pathlib.Path(r'D:\LAB\SAM_Experiment\working\synthetic_fire_3d_v3')
for fn in ['dataset_info.json','evaluation_test.json','noise_benchmark_test.json']:
 d=json.loads((root/fn).read_text(encoding='utf-8')); print('\n',fn); print(json.dumps(d,indent=2,ensure_ascii=False)[:12000])
