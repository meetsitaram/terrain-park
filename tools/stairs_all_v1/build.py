"""Build tp_stairs_all_v1 from the body-to-body X2 retargets of the 2026-10-30 Pico stair takes (both stair places).
Steps (the terrain-park / build_train recipe): 50 Hz resample + Gaussian smooth, untracked joints zeroed, floor levelled,
stairs fitted to the X2 planted feet, x + 10 m, checks. Kinematic only (no tracker rollout)."""
import numpy as np, mujoco, glob, os, json, joblib
from scipy.spatial.transform import Rotation as sRot, Slerp
XML = r'C:/Users/PC/projects/agibot-x2-record-and-replay/sim/x2_ultra.xml'
TPB = r'C:/Users/PC/projects/terrain-park/data/2026-10-30/x2_body_to_body'
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
FPS = 50; SIG = 0.04 * FPS
UNTR = ('left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_wrist_yaw_joint',
        'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'head_yaw_joint', 'head_pitch_joint')
UIDX = [m.jnt_qposadr[m.joint(n).id] for n in UNTR]
HJ = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
JR = np.array([m.jnt_range[j] for j in HJ]); JN = [m.joint(j).name for j in HJ]
FEET = []
for side in ('left', 'right'):
    bid = m.body(f'{side}_ankle_roll_link').id
    g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1][0]
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    FEET.append((g, v[::max(1, len(v) // 300)]))
P2 = '114221 114241 114251 114258 114308 114315 114324 114330'.split()
SKIP = {'113723': 'no stair contact (turn-around on the floor at the stair foot)',
        '114456': 'no stair contact (walk up to the stair foot and stop)'}
ST2 = json.load(open(r'C:/Users/PC/projects/terrain-park/data/2026-10-30/station_stairs2/stairs_x2.json'))['props'][0]
# cuts (s): a mis-levelled standing stretch at the very start or end, or a tracking jump (114037 at 4.86 s)
TRIM = {'113941': (0.36, None), '114146': (0.36, None), '113730': (None, 6.24), '114037': (None, 4.70)}
FR = {1: (np.array([1.33, 2.92]), np.radians(21.0)), 2: (np.array(ST2['origin'][:2]), ST2['yaw'])}


def resample(t, q):
    tu = np.arange(t[0], t[-1], 1.0 / FPS)
    out = np.stack([np.interp(tu, t, q[:, j]) for j in range(q.shape[1])], 1)
    w = q[:, 3:7].copy()
    for i in range(1, len(w)):
        if (w[i] * w[i - 1]).sum() < 0: w[i] *= -1
    keep = np.r_[True, np.diff(t) > 0]
    out[:, 3:7] = Slerp(t[keep], sRot.from_quat(w[keep][:, [1, 2, 3, 0]]))(tu).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(out)):
        if (out[i, 3:7] * out[i - 1, 3:7]).sum() < 0: out[i, 3:7] *= -1
    k = np.arange(-int(3 * SIG), int(3 * SIG) + 1); g = np.exp(-0.5 * (k / SIG) ** 2); g /= g.sum()
    pad = np.pad(out, ((len(k) // 2, len(k) // 2), (0, 0)), mode='edge')
    out = np.stack([np.convolve(pad[:, j], g, mode='valid') for j in range(out.shape[1])], 1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def feet(q, o, up, n):
    """(N,2,7): centre s, centre l, sole z, smin, smax, lmin, lmax of the bottom 1.5 cm slab, stair frame."""
    R = np.zeros((len(q), 2, 7))
    for i in range(len(q)):
        d.qpos[:38] = q[i]; mujoco.mj_kinematics(m, d)
        for k, (g, v) in enumerate(FEET):
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T; zmin = w[:, 2].min()
            sl = w[w[:, 2] < zmin + 0.015][:, :2] - o; s = sl @ up; l = sl @ n
            R[i, k] = [s.mean(), l.mean(), zmin, s.min(), s.max(), l.min(), l.max()]
    return R


def episodes(R):
    """Planted stretches: sole speed < 8 cm/s for >= 8 frames and not the clearly higher foot in the air."""
    E = []
    for f in range(2):
        v = np.r_[0, np.linalg.norm(np.diff(R[:, f, :3], axis=0), axis=1) * FPS]; v = np.convolve(v, np.ones(5) / 5, 'same')
        ok = v < 0.08; i = 0
        while i < len(ok):
            if ok[i]:
                j = i
                while j + 1 < len(ok) and ok[j + 1]: j += 1
                if j - i + 1 >= 8: E.append((f, i, j))
                i = j + 1
            else:
                i += 1
    return E


T = {}
for k in sorted([os.path.basename(f)[16:22] for f in glob.glob('b2b/record_*.npz')] + P2):
    if k == '104425': continue
    z = np.load((TPB if k in P2 else 'b2b') + '/record_20261030_%s.npz' % k)
    q = resample(z['t'].astype(float), z['qpos'].astype(float)); q[:, UIDX] = 0.0
    if k in TRIM:
        a_, b_ = TRIM[k]; q = q[int(round((a_ or 0) * FPS)):(int(round(b_ * FPS)) if b_ else None)]
    pl = 2 if k in P2 else 1; o, y = FR[pl]; up = np.array([np.cos(y), np.sin(y)]); n = np.array([-up[1], up[0]])
    R = feet(q, o, up, n)
    T[k] = dict(q=q, pl=pl, o=o, y=y, up=up, n=n, R=R, E=[(f, i, j, np.median(R[i:j + 1, f], axis=0)) for f, i, j in episodes(R)])

# place 1: floor line along the climb, then per-take level shift, then tops and edges from the levelled plants
A = np.array([e[3] for t in T.values() if t['pl'] == 1 for e in t['E']])
fl = A[(A[:, 2] < 0.12) & (A[:, 0] < -0.15)]; kf, z0 = np.polyfit(fl[:, 0], fl[:, 2], 1)
print('place 1 floor: %.1f cm at the stair foot, slope %.1f cm/m along the climb (n %d plants)' % (100 * z0, 100 * kf, len(fl)))
zfloor = lambda s: z0 + kf * np.minimum(s, 0.0)
tops = (0.18, 0.27)
for it in range(3):
    for k, t in T.items():
        if t['pl'] == 2:
            t['dz'] = 0.0; t['zcorr'] = np.full(len(t['q']), -ST2['origin'][2]); continue
        s = (t['q'][:, :2] - t['o']) @ t['up']; base = -zfloor(s); err = []
        for f, i, j, md in t['E']:
            zz = md[2] + np.median(base[i:j + 1]); lv = min((0.0, tops[0], tops[1]), key=lambda h: abs(zz - h)); err.append(zz - lv)
        t['dz'] = -float(np.median(err)) if err else 0.0
        t['zcorr'] = base + t['dz']
    Z, SS = [], []
    for t in T.values():
        if t['pl'] != 1: continue
        for f, i, j, md in t['E']:
            Z.append(md[2] + np.median(t['zcorr'][i:j + 1])); SS.append(md)
    Z = np.array(Z); SS = np.array(SS)
    s1 = (Z > 0.10) & (Z < (tops[0] + tops[1]) / 2) & (SS[:, 0] < 0.6); s2 = Z >= (tops[0] + tops[1]) / 2
    tops = (float(np.median(Z[s1])), float(np.median(Z[s2])))
e1 = float(np.percentile(SS[s1 & (SS[:, 0] > -0.15), 3], 10)); e2 = float(np.percentile(SS[s2, 3], 3))
l1a, l1b = float(np.percentile(SS[s1, 5], 3)), float(np.percentile(SS[s1, 6], 97))
s2max, l2a, l2b = float(SS[s2, 4].max()), float(SS[s2, 5].min()), float(SS[s2, 6].max())
print('place 1 stairs: step 1 top %.1f cm (n %d), front edge s=%.2f, lateral %.2f..%.2f | deck top %.1f cm (n %d), edge s=%.2f, far %.2f, lateral %.2f..%.2f'
      % (100 * tops[0], s1.sum(), e1, l1a, l1b, 100 * tops[1], s2.sum(), e2, s2max, l2a, l2b))
print('per-take level shifts, place 1 (cm): min %.1f med %.1f max %.1f' % tuple(100 * np.percentile([t['dz'] for t in T.values() if t['pl'] == 1], [0, 50, 100])))
w1c = (l1a + l1b) / 2; w1 = max(l1b - l1a + 0.10, 0.8)
r2 = ST2['steps'][0]['run'] + ST2['steps'][1]['run']
SURF = {1: [(e1, e2, w1c - w1 / 2, w1c + w1 / 2, tops[0]), (e2, s2max + 0.3, l2a - 0.3, l2b + 0.3, tops[1])],
        2: [(0.0, r2, -0.4, 0.4, ST2['steps'][0]['top']), (ST2['steps'][0]['run'], r2, -0.4, 0.4, ST2['steps'][1]['top'])]}


def boxes(t, dx):
    B = []
    for s0, s1_, l0, l1, top in SURF[t['pl']]:
        c = t['o'] + t['up'] * (s0 + s1_) / 2 + t['n'] * (l0 + l1) / 2
        B.append(dict(kind='step', center=[float(c[0] + dx), float(c[1]), top / 2], half=[(s1_ - s0) / 2, (l1 - l0) / 2, top / 2], yaw=float(t['y'])))
    return B


def heights_under(pl, md):
    hs = {0.0}
    for s0, s1_, l0, l1, top in SURF[pl]:
        if md[4] > s0 and md[3] < s1_ and md[6] > l0 and md[5] < l1: hs.add(top)
    return hs


clips, props, checks = {}, {}, {}
for k, t in T.items():
    q = t['q'].copy(); q[:, 2] += t['zcorr']; name = 'tp_stairs_' + k; ch = dict(place=t['pl'])
    errs = []; ch['eps'] = []
    for f, i, j, md in t['E']:
        zz = md[2] + np.median(t['zcorr'][i:j + 1]); h = min(heights_under(t['pl'], md), key=lambda a: abs(zz - a)); errs.append(zz - h)
        ch['eps'].append([f, i, j, sorted({0.0} | {x[4] for x in SURF[t['pl']]}).index(h), round(float(md[0]), 3), round(float(md[1]), 3)])
    errs = np.array(errs); ch['contacts'] = len(errs)
    ch['sink_max_cm'] = round(float(max(-100 * errs.min(), 0)), 1); ch['float_max_cm'] = round(float(max(100 * errs.max(), 0)), 1)
    ch['contacts_over_2cm'] = int((np.abs(errs) > 0.02).sum())
    dof = q[:, 7:]; over = np.maximum(JR[:, 0] - dof, dof - JR[:, 1]).max(0)
    ch['joint_over_rad'] = round(float(max(over.max(), 0)), 3); ch['joint_worst'] = JN[int(over.argmax())] if over.max() > 0 else ''
    ch['root_speed_max'] = round(float((np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) * FPS).max()), 2)
    ch['root_acc_max'] = round(float((np.linalg.norm(np.diff(q[:, :3], 2, axis=0), axis=1) * FPS * FPS).max()), 1)
    ch['dof_vel_max'] = round(float(np.abs(np.diff(dof, axis=0)).max() * FPS), 1)
    ch['level_shift_cm'] = round(100 * t['dz'], 1)
    ch['ok'] = bool(ch['contacts_over_2cm'] == 0 and ch['joint_over_rad'] <= 0.0 and ch['root_speed_max'] < 2.5)
    checks[name] = ch
    if k in SKIP:
        ch['excluded'] = SKIP[k]; continue
    qq = q.copy(); qq[:, 0] += 10.0
    clips[name] = dict(root_trans_offset=qq[:, :3].astype(np.float32), root_rot=qq[:, [4, 5, 6, 3]].astype(np.float32), dof=qq[:, 7:].astype(np.float32), fps=FPS)
    props[name] = boxes(t, 10.0)
joblib.dump(clips, 'all_candidates.pkl'); json.dump(props, open('all_candidates_props.json', 'w'), indent=1); json.dump(checks, open('checks.json', 'w'), indent=1)
json.dump(dict(SURF={str(a): b for a, b in SURF.items()}, FR={str(a): [list(map(float, b[0])), float(b[1])] for a, b in FR.items()}, floor=[float(z0), float(kf)]), open('geom.json', 'w'))
print('candidate clips', len(clips), 'seconds %.0f' % sum(len(c['dof']) / FPS for c in clips.values()))
print('pass all checks:', sum(c['ok'] and 'excluded' not in c for c in checks.values()))
print('fail: id contacts over2 sink float joint_over worst root_speed dofvel')
for k, c in checks.items():
    if not c['ok']:
        print(k[10:], c['contacts'], c['contacts_over_2cm'], c['sink_max_cm'], c['float_max_cm'], c['joint_over_rad'], c['joint_worst'], c['root_speed_max'], c['dof_vel_max'])
old = joblib.load(r'C:/Users/PC/projects/terrain-park/data/train_sets/tp_stairs_x10.pkl')
for k in ('tp_stairs_114221', 'tp_stairs_114330'):
    a, b = old[k], clips[k]; nmin = min(len(a['dof']), len(b['dof']))
    print(k, len(a['dof']), len(b['dof']), 'max diff vs tp_stairs_x10: root %.4f dof %.4f rot %.4f' % (
        np.abs(a['root_trans_offset'][:nmin] - b['root_trans_offset'][:nmin]).max(), np.abs(a['dof'][:nmin] - b['dof'][:nmin]).max(),
        np.abs(a['root_rot'][:nmin] - b['root_rot'][:nmin]).max()))

if os.environ.get('DBG'):
    print('step-1 plant smin sorted:', np.round(np.sort(SS[s1, 3])[:8], 2), 'centre s:', np.round(np.sort(SS[s1, 0])[:6], 2))
    print('deck plant smin sorted:', np.round(np.sort(SS[s2, 3])[:8], 2))
    for k, t in T.items():
        if checks['tp_stairs_' + k]['ok']: continue
        N = len(t['q']); out = []
        for f, i, j, md in t['E']:
            zz = md[2] + np.median(t['zcorr'][i:j + 1]); h = min(heights_under(t['pl'], md), key=lambda a: abs(zz - a))
            if abs(zz - h) > 0.02: out.append('%s t%.1f-%.1f/%.1f s%.2f l%.2f z%.1f h%.1f' % ('LR'[f], i / FPS, j / FPS, N / FPS, md[0], md[1], 100 * zz, 100 * h))
        v = np.linalg.norm(np.diff(t['q'][:, :3], axis=0), axis=1) * FPS
        if v.max() > 2.5: out.append('root speed>2.5 at t=%s' % np.round(np.where(v > 2.5)[0] / FPS, 2)[:6])
        print(k, ' | '.join(out))
