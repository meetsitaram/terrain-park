"""Retarget artifact check: every base take of the Pico sets (+ _mir partners) vs the raw Pico recording.
Usage: python artcheck.py [set1.pkl,set2.pkl...]  -> artcheck.json in cwd + summary print."""
import numpy as np, joblib, mujoco, json, sys, os, re
from scipy.spatial.transform import Rotation as R
TP = r'C:/Users/PC/projects/terrain-park'; TS = os.environ.get('TSDIR', TP + '/data/train_sets/')
m = mujoco.MjModel.from_xml_path(r'C:/Users/PC/projects/agibot-x2-record-and-replay/sim/x2_ultra.xml'); d = mujoco.MjData(m)
HJ = [j for j in range(m.njnt) if m.jnt_type[j] == 3]
DN = {int(m.jnt_qposadr[j]) - 7: m.joint(j).name.replace('_joint', '') for j in HJ}
LO = np.array([m.jnt_range[j][0] for j in HJ]); HI = np.array([m.jnt_range[j][1] for j in HJ])
BH = m.body('head_pitch_link').id; BP = m.body('pelvis').id
J = {n: i for i, n in enumerate(["Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee", "Spine2", "Left_Ankle", "Right_Ankle",
     "Spine3", "Left_Foot", "Right_Foot", "Neck", "Left_Collar", "Right_Collar", "Head"])}
TRACKED = [i for i in range(31) if not re.search('wrist|head', DN[i])]
LEG = list(range(12))

SETS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['tp_stairs_all_v1', 'pico_2026-10-29_2x2_train', 'pico_2026-10-29_2x2_test',
                                                       'tp_driveway_v1', 'tp_t4x4u_y20', 'acp_tp_ramp_sh2_x30']
MIR = {'tp_stairs_all_v1': 'tp_stairs_all_v1_mir', 'pico_2026-10-29_2x2_train': 'pico_2026-10-29_2x2_train_mir', 'tp_driveway_v1': 'tp_driveway_v1_mir',
       'tp_t4x4u_y20': 'tp_t4x4u_y20_mir', 'acp_tp_ramp_sh2_x30': 'tp_ramp_base_x30_mir'}


def rec_of(name):
    m_ = re.search(r'(2026\d{4})_(\d{6})', name)
    if m_: return m_.group(1), m_.group(2)
    return '20261030', re.search(r'_(\d{6})', name).group(1)


def x2feat(c):
    T = len(c['dof']); lean = np.zeros(T); hp = np.zeros(T); tilt = np.zeros(T)
    for i in range(T):
        d.qpos[:3] = c['root_trans_offset'][i]; x, y, z, w = c['root_rot'][i]; d.qpos[3:7] = [w, x, y, z]; d.qpos[7:] = c['dof'][i]
        mujoco.mj_kinematics(m, d); v = d.xpos[BH] - d.xpos[BP]
        lean[i] = np.degrees(np.arccos(v[2] / np.linalg.norm(v))); hp[i] = v[2]
        tilt[i] = np.degrees(np.arccos(np.clip(d.xmat[BP].reshape(3, 3)[2, 2], -1, 1)))
    return lean, hp, tilt


def align(c, x2):
    """offset (s) of clip frame 0 in x2 time, by dof match on tracked joints + root height change."""
    t = x2['t'].astype(float); q = x2['qpos'].astype(float); fps = float(c['fps'])
    tu = np.arange(t[0], t[-1], 1 / fps); X = np.stack([np.interp(tu, t, q[:, 7 + j]) for j in TRACKED], 1)
    C = np.asarray(c['dof'], float)[:, TRACKED]; n = len(C); best = (1e9, 0)
    for s in range(-int(0.5 * fps), max(1, len(X) - n + int(0.5 * fps))):
        a, b = max(0, s), min(len(X), s + n)
        if b - a < 0.6 * n: continue
        e = np.abs(X[a:b] - C[a - s:b - s]).mean()
        if e < best[0]: best = (e, s)
    return tu[0] + best[1] / fps, best[0]


def human(raw, x2t0, tq):
    """human lean (deg, Pelvis->Head from vertical) and head-pelvis height at times tq (x2 time base = raw stamp seconds)."""
    ts = (raw['stamp_ns'] - raw['stamp_ns'][0]) / 1e9
    b = raw['body'].astype(float); ok = raw['body_ok'].astype(float)
    v = b[:, J['Head'], :3] - b[:, J['Pelvis'], :3]
    lean = np.degrees(np.arccos(v[:, 1] / np.linalg.norm(v, axis=1)))
    hy = raw['headset'][:, 1].astype(float) - float(raw['floor_y'])
    # human limb motion: knee positions relative to pelvis, speed m/s
    kn = np.concatenate([b[:, J['Left_Knee'], :3] - b[:, J['Pelvis'], :3], b[:, J['Right_Knee'], :3] - b[:, J['Pelvis'], :3],
                         b[:, J['Left_Ankle'], :3] - b[:, J['Pelvis'], :3], b[:, J['Right_Ankle'], :3] - b[:, J['Pelvis'], :3]], 1)
    ksp = np.r_[0, np.linalg.norm(np.diff(kn, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-3)]
    f = lambda a: np.interp(tq, ts, a)
    return f(lean), f(v[:, 1]), f(hy), f(ksp), f(ok)


def frozen(dof, fps, min_s=0.5):
    """per joint list of (start, end) frames where the value is constant (<1e-4 rad/frame) for >= min_s."""
    out = {}
    st = np.abs(np.diff(dof, axis=0)) < 1e-4
    for j in range(15):
        i = 0; s = st[:, j]
        while i < len(s):
            if s[i]:
                k = i
                while k + 1 < len(s) and s[k + 1]: k += 1
                if (k - i + 2) >= min_s * fps: out.setdefault(j, []).append((i, k + 1))
                i = k + 1
            else: i += 1
    return out


res = {}
for S in SETS:
    cl = joblib.load(TS + S + '.pkl'); mir = joblib.load(TS + MIR[S] + '.pkl') if S in MIR else {}
    for k in cl:
        if '_sh' in k: continue
        c = {a: np.asarray(v, float) if a != 'fps' else v for a, v in cl[k].items()}; fps = float(c['fps'])
        day, hm = rec_of(k); x2f = '%s/data/%s-%s-%s/x2/record_%s_%s.npz' % (TP, day[:4], day[4:6], day[6:], day, hm)
        rawf = x2f.replace('/x2/', '/raw/')
        x2 = np.load(x2f.replace(TP + '/data/2026-10-30/x2', os.environ['X2OV']) if os.environ.get('X2OV') and 'driveway' in k else x2f); raw = np.load(rawf)
        t0, aerr = align(c, x2); hs = float(x2['height_scale'])
        T = len(c['dof']); tq = t0 + np.arange(T) / fps
        hl, hdz, hy, ksp, ok = human(raw, t0, tq)
        r = dict(set=S, T=T, fps=fps, t0=round(t0, 2), align_err=round(float(aerr), 3), hscale=round(hs, 3))
        for tag, cc in (('base', c), ('mir', mir.get(k + '_mir'))):
            if cc is None: continue
            cc = {a: np.asarray(v, float) if a != 'fps' else v for a, v in cc.items()}
            xl, xdz, tilt = x2feat(cc); dl = np.abs(xl - hl)
            fz = frozen(cc['dof'], fps); dof_ = cc['dof']
            moving = ksp > 0.15                            # human legs move relative to the pelvis
            fzflag = np.zeros(T, bool); fzj = set()
            for j, segs in fz.items():
                for a, b in segs:
                    if moving[a:b + 1].mean() > 0.5: fzflag[a:b + 1] = True; fzj.add('%s=%.2f' % (DN[j], dof_[a, j]))
            dof = cc['dof']; atlim = (np.minimum(dof - LO, HI - dof) < 1e-3)
            imp = np.zeros(T, bool); impj = set()
            for j in (2, 8, 12):                           # hip yaw / waist yaw far out
                bad = np.abs(dof[:, j]) > 1.2
                if bad.any(): imp |= bad; impj.add(DN[j])
            bad = atlim[:, 12:15].sum(1) >= 2                # two or more waist joints pinned at their limits together
            if bad.any(): imp |= bad; impj.add('waist_2_at_limit')
            leanflag = dl > 20
            fl = leanflag | fzflag | imp
            idx = np.where(fl)[0]
            r[tag] = dict(n_flag=int(fl.sum()), n_lean=int(leanflag.sum()), n_frozen=int(fzflag.sum()), n_implaus=int(imp.sum()),
                          first=int(idx[0]) if len(idx) else -1, last=int(idx[-1]) if len(idx) else -1,
                          frozen_joints=sorted(fzj), implaus_joints=sorted(impj),
                          lean_x2_max=round(float(xl.max()), 1), lean_h_max=round(float(hl.max()), 1), dlean_max=round(float(dl.max()), 1),
                          tilt_max=round(float(tilt.max()), 1), human_ok_min=round(float(ok.min()), 2),
                          hdz_h=round(float(np.median(hdz)) * hs, 3), hdz_x2=round(float(np.median(xdz)), 3),
                          head_h_min=round(float(hy.min()), 2), flag_frames=[int(x) for x in idx])
        res[k] = r
        b = r['base']
        print('%-30s al%.3f T%4d flag %4d (lean %d froz %d imp %d) %s..%s fz%s imp%s lean x2 %.0f h %.0f d %.0f tilt %.0f | mir %s' % (
            k, aerr, T, b['n_flag'], b['n_lean'], b['n_frozen'], b['n_implaus'], b['first'], b['last'], b['frozen_joints'], b['implaus_joints'],
            b['lean_x2_max'], b['lean_h_max'], b['dlean_max'], b['tilt_max'], r.get('mir', {}).get('n_flag', '-')), flush=True)
json.dump(res, open(os.environ.get('OUT', 'artcheck.json'), 'w'))
