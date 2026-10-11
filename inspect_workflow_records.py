import json, pathlib, collections
for rel in ['D:\\LAB\\SAM_Experiment\\output\\five_workflows\\summary.json','D:\\LAB\\SAM_Experiment\\output\\workflow_2d_then_3d_20261008\\3d_workflows\\summary.json']:
 d=json.loads(pathlib.Path(rel).read_text(encoding='utf-8'))
 print('\n',rel)
 rec=d.get('records',[])
 print('records',len(rec))
 # print first record keys and nested
 if rec:
  print('keys',rec[0].keys()); print(json.dumps(rec[0],ensure_ascii=False)[:5000])
 for field in ['roi_reliability','roi_pixel','blend_pixel','roi_confidence','branches']:
  vals=[r.get(field) for r in rec if field in r]
  if vals: print(field,'count',len(vals),'sample',vals[0])
