import joblib, json, shutil, numpy as np
SH = 'shadow/terrain-park/data/train_sets/'; TS = 'C:/Users/PC/projects/terrain-park/data/train_sets/'
def diff(a, b):
    out = {}
    for k in set(a) | set(b):
        if k not in a or k not in b: out[k] = 'only in ' + ('new' if k in b else 'old'); continue
        x, y = a[k], b[k]
        if len(x['dof']) != len(y['dof']): out[k] = 'len %d->%d' % (len(x['dof']), len(y['dof'])); continue
        d = max(float(np.abs(np.asarray(x[f], float) - np.asarray(y[f], float)).max()) for f in ('root_trans_offset', 'root_rot', 'dof'))
        if d > 0: out[k] = round(d, 5)
    return out
for n in ('tp_t4x4u_y20_mir', 'tp_ramp_base_x30_mir', 'tp_stairs_all_v1_mir', 'pico_2026-10-29_2x2_train_mir', 'tp_driveway_v1_mir', 'tp_stairs_all_v1', 'pico_2026-10-29_2x2_train', 'tp_driveway_v1'):
    d = diff(joblib.load(TS + n + '.pkl'), joblib.load(SH + n + '.pkl')); print(n, 'changed clips:', len(d), dict(list(d.items())[:8]))
for n in ('tp_t4x4u_y20_mir_props', 'tp_stairs_all_v1_mir_props', 'pico_2026-10-29_2x2_train_mir_props'):
    a = json.load(open(TS + n + '.json')); b = json.load(open(SH + n + '.json')); print(n, 'props changed:', [k for k in b if a.get(k) != b[k]], 'removed', [k for k in a if k not in b])
# write v1_1 files
W = [('tp_stairs_all_v1.pkl', 'tp_stairs_all_v1_1.pkl'), ('tp_stairs_all_v1_props.json', 'tp_stairs_all_v1_1_props.json'), ('tp_stairs_all_v1_meta.json', 'tp_stairs_all_v1_1_meta.json'),
     ('tp_stairs_all_v1_mir.pkl', 'tp_stairs_all_v1_1_mir.pkl'), ('tp_stairs_all_v1_mir_props.json', 'tp_stairs_all_v1_1_mir_props.json'), ('tp_stairs_all_v1_mir_meta.json', 'tp_stairs_all_v1_1_mir_meta.json'),
     ('pico_2026-10-29_2x2_train.pkl', 'pico_2026-10-29_2x2_train_v1_1.pkl'), ('pico_2026-10-29_2x2_train_mir.pkl', 'pico_2026-10-29_2x2_train_v1_1_mir.pkl'),
     ('pico_2026-10-29_2x2_train_mir_props.json', 'pico_2026-10-29_2x2_train_v1_1_mir_props.json'),
     ('tp_driveway_v1.pkl', 'tp_driveway_v1_1.pkl'), ('tp_driveway_v1_props.json', 'tp_driveway_v1_1_props.json'),
     ('tp_driveway_v1_mir.pkl', 'tp_driveway_v1_1_mir.pkl'), ('tp_driveway_v1_mir_props.json', 'tp_driveway_v1_1_mir_props.json')]
import os
for a, b in W:
    assert not os.path.exists(TS + b), b
    shutil.copy(SH + a, TS + b)
print('written', len(W))
print('driveway plane old', json.load(open(TS + 'tp_driveway_v1_props.json'))['tp_driveway_121023'][0], 'new', json.load(open(TS + 'tp_driveway_v1_1_props.json'))['tp_driveway_121023'][0])
