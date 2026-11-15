"""Assemble a shadow terrain-park with the fixed base sets, so the unchanged mirror tool can rebuild the _mir sets."""
import shutil, json, joblib, os
from pathlib import Path
S = Path(r'C:/Users/PC/AppData/Local/Temp/claude/74893d75b5436cbb/cf3ca1e7-ff31-44d4-97d3-238f0a3bcf48/scratchpad')
TP = Path(r'C:/Users/PC/projects/terrain-park'); SH = S / 'shadow' / 'terrain-park'
if SH.exists(): shutil.rmtree(SH)
(SH / 'catalog' / 'gifs').mkdir(parents=True); (SH / 'tools' / 'mirror').mkdir(parents=True)
(SH / 'data' / '2026-10-30' / 'x2').mkdir(parents=True); (SH / 'data' / '2026-10-30' / 'props').mkdir(parents=True)
shutil.copy(TP / 'catalog' / 'make_catalog.py', SH / 'catalog'); shutil.copy(TP / 'catalog' / 'categories.json', SH / 'catalog')
shutil.copy(TP / 'catalog' / 'gifs' / 'index.json', SH / 'catalog' / 'gifs')
for f in ('mirror_sets.py', 'x2_mirror_map.json'): shutil.copy(TP / 'tools' / 'mirror' / f, SH / 'tools' / 'mirror')
shutil.copy(TP / 'data' / '2026-10-30' / 'props' / 'ramp.json', SH / 'data' / '2026-10-30' / 'props')
shutil.copytree(TP / 'data' / 'train_sets', SH / 'data' / 'train_sets')
for f in (S / 'rr_plain').glob('record_20261030_121*.npz'): shutil.copy(f, SH / 'data' / '2026-10-30' / 'x2')
# the X2 model: make_catalog looks in REPO.parent
os.environ['X2_RECORDER'] = r'C:/Users/PC/projects/agibot-x2-record-and-replay'
TS = SH / 'data' / 'train_sets'
# stairs: v1 with 113941 replaced by the re-retargeted clip (props stay v1's)
cl = joblib.load(TS / 'tp_stairs_all_v1.pkl'); new = joblib.load(S / 'stairs_v1_1' / 'tp_stairs_all_v1.pkl')
cl['tp_stairs_113941'] = new['tp_stairs_113941']; joblib.dump(cl, TS / 'tp_stairs_all_v1.pkl')
me = json.load(open(TS / 'tp_stairs_all_v1_meta.json')); mn = json.load(open(S / 'stairs_v1_1' / 'tp_stairs_all_v1_meta.json'))
me['tp_stairs_113941'] = mn['tp_stairs_113941']
me['tp_stairs_113941']['fix_v1_1'] = ('re-retargeted 2026-11-12 with a fresh IK state: in v1 the warm start carried over from take 113923 '
                                     'pinned waist yaw/pitch/roll at their limits (2.38, 0.35, -0.52) for 3.4 s and pitched the pelvis 46-52 deg')
json.dump(me, open(TS / 'tp_stairs_all_v1_meta.json', 'w'), indent=1)
# 2x2: drop pico_20261029_211326_0_2x2 (left hip yaw pinned at 3.43 rad through the whole take)
c2 = joblib.load(TS / 'pico_2026-10-29_2x2_train.pkl'); c2.pop('pico_20261029_211326_0_2x2'); joblib.dump(c2, TS / 'pico_2026-10-29_2x2_train.pkl')
print('shadow ready', len(cl), len(c2))
