from pathlib import Path
import json, statistics, math
root=Path(r'D:\LAB\SAM_Experiment\working\synthetic_fire_3d_v3')
rows=[json.loads(x) for x in (root/'test.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
vis=[r for r in rows if r.get('fire_visible') and r.get('p_fire_pixel') and r.get('fire_xyz_world')]
print('test rows',len(rows),'visible',len(vis),'surfaces', {s:sum(r.get('fire_surface')==s for r in vis) for s in sorted(set(r.get('fire_surface') for r in vis))},'scenes',len(set(r['scene_id'] for r in vis)))
# records in depth samples stats
for fn in ['depth_anything_v2_metric_indoor_test_visible92.json','depth_anything_v2_metric_indoor_roi_test92.json']:
 d=json.loads((root/fn).read_text(encoding='utf-8')); print(fn,'samples',len(d.get('samples',[])), 'scene_ids',len(set(x.get('scene_id') for x in d['samples'])))
