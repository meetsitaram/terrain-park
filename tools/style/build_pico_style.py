"""Compact style-reference set from the user's Pico takes only (X2 retargets, corrected v1_1 sets + mirrors).

    python tools/style/build_pico_style.py [segment|review|all] [--config my.json]

segment: cut every take into labelled windows (flat_walk, ramp, logs, stairs, turn_in_place, side_walk), check each
         window, select per category for variety and cleanliness up to the seconds target, write
         data/style/pico_style_v1.pkl (joblib, clip format + category, terrain, source) and pico_style_v1_meta.json.
review:  one contact sheet PNG and one MuJoCo GIF per category in data/style/review/.

Length target (user, 2026-11-13): similar to the Hiking-in-the-Wild reference (one G1 npz, 18,982 frames at 50 fps =
380 s), so about 60 s per category, windows of 1.5-5 s cut at double support, mirrors included in the total.
CPU only, at most 3 threads. Sources: data/train_sets (see SETS). No BONES-SEED data.
"""
import os, sys, json, math, re
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------- settings (override with --config file.json)
CONFIG = json.loads("""{
 "total_seconds_target": 380,
 "per_category": {
  "flat_walk":     {"seconds": 63, "n_max": 40},
  "ramp":          {"seconds": 63, "n_max": 40},
  "logs":          {"seconds": 63, "n_max": 40},
  "stairs":        {"seconds": 63, "n_max": 40},
  "turn_in_place": {"seconds": 63, "n_max": 40},
  "side_walk":     {"seconds": 63, "n_max": 40}
 },
 "win_min_s": 1.5, "win_max_s": 5.0,
 "max_uses_per_take": 2,
 "flat": {"v_fwd": [0.3, 1.2], "yaw_rate": 0.3, "prop_clear_m": 0.3, "clear_frac_min": 0.85},
 "turn": {"yaw_rate": 0.5, "speed": 0.15, "min_turn_deg": 45},
 "side": {"lateral_share": 0.6, "speed": 0.1},
 "checks": {"foot_err_m": 0.02, "foot_err_pct": 95, "sink_m": 0.02, "joint_tol_rad": 0.01,
            "root_step_m": 0.05, "root_yaw_step_rad": 0.15, "lean_pct": 99.0},
 "gif": {"w": 400, "h": 300, "fps": 10, "max_bytes": 1500000}
}""")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TS = REPO / 'data' / 'train_sets'
OUT = REPO / 'data' / 'style'
sys.path.insert(0, str(REPO / 'catalog'))
import make_catalog as mc          # noqa: E402  (frames, X2 model, GIF helpers)

CATS = ['flat_walk', 'ramp', 'logs', 'stairs', 'turn_in_place', 'side_walk']
# (set, props json or None, prop kind, ramp height map)
SETS = [('tp_stairs_all_v1_1', 'tp_stairs_all_v1_1_props', 'stairs', None),
        ('tp_stairs_all_v1_1_mir', 'tp_stairs_all_v1_1_mir_props', 'stairs', None),
        ('pico_2026-10-29_2x2_train_v1_1', 'pico_2026-10-29_2x2_props', '2x2', None),
        ('pico_2026-10-29_2x2_train_v1_1_mir', 'pico_2026-10-29_2x2_train_v1_1_mir_props', '2x2', None),
        ('tp_t4x4u_y20', 'tp_t4x4u_y20_props', '4x4', None),
        ('tp_t4x4u_y20_mir', 'tp_t4x4u_y20_mir_props', '4x4', None),
        ('acp_tp_ramp_sh2_x30', None, 'ramp', 'tp_ramp_hmap_x30'),          # base takes only (no _sh)
        ('tp_ramp_base_x30_mir', None, 'ramp', 'tp_ramp_hmap_x30_mir'),
        ('tp_driveway_v1_1', 'tp_driveway_v1_1_props', 'driveway', None),
        ('tp_driveway_v1_1_mir', 'tp_driveway_v1_1_mir_props', 'driveway', None)]
SOLE_R = 0.005
_HM = {}


def hmap(name):
    if name not in _HM:
        h = np.load(TS / (name + '.npz'))
        _HM[name] = (float(h['x0']), float(h['y0']), float(h['res']), np.asarray(h['grid'], float))
    return _HM[name]


def ramp_frame(hm):
    """as make_catalog.ramp_prop, on the given (possibly mirrored) height map."""
    x0, y0, res, g = hm; hmax = g.max()
    gx, gy = np.gradient(g, res); mag = np.hypot(gx, gy)
    m = (g > 0.03) & (g < 0.85 * hmax) & (mag > 0.05) & (mag < 0.5)
    yaw = math.atan2(gy[m].mean(), gx[m].mean()); u = np.array([math.cos(yaw), math.sin(yaw)]); v = np.array([-u[1], u[0]])
    I, J = np.where(m); xy = np.c_[x0 + res * I, y0 + res * J]
    k, c = np.polyfit(xy @ u, g[m], 1)
    return u * (-c / k) + v * np.median(xy @ v), yaw


# ---------------------------------------------------------------- loading
def load():
    import joblib
    takes = []
    for st, pj, kind, hm in SETS:
        cl = joblib.load(TS / (st + '.pkl'))
        pr = json.load(open(TS / (pj + '.json'))) if pj else {}
        for k, c in cl.items():
            if kind == 'ramp' and '_sh' in k:
                continue
            tk = dict(id=k, set=st, prop=kind, mir=k.endswith('_mir'), fps=float(c['fps']),
                      pos=np.asarray(c['root_trans_offset'], float), quat=np.asarray(c['root_rot'], float),
                      dof=np.asarray(c['dof'], float), boxes=[], plane=None, hmap=hm)
            for b in pr.get(k, []):
                if b.get('kind') == 'plane':
                    tk['plane'] = b
                else:
                    tk['boxes'].append(b)
            takes.append(tk)
    return takes


def frame_of(tk):
    if tk['prop'] in ('4x4', '2x2', 'stairs'):
        return mc.base_frame(dict(prop='stairs1' if tk['prop'] == 'stairs' else tk['prop'], boxes=tk['boxes']), None)
    if tk['prop'] == 'ramp':
        return ramp_frame(hmap(tk['hmap']))
    p = tk['plane']; return tk['pos'][:, :2].mean(0), math.atan2(p['b'], p['a'])


def surface(tk, xy):
    """ground height under xy (N,2) and whether it is the floor."""
    z = np.zeros(len(xy)); flat = np.ones(len(xy), bool)
    if tk['prop'] == 'driveway':
        p = tk['plane']; return p['a'] * xy[:, 0] + p['b'] * xy[:, 1] + p['c'], np.zeros(len(xy), bool)
    if tk['hmap']:
        x0, y0, res, g = hmap(tk['hmap'])
        fi = np.clip((xy[:, 0] - x0) / res, 0, g.shape[0] - 1.001); fj = np.clip((xy[:, 1] - y0) / res, 0, g.shape[1] - 1.001)
        i, j = fi.astype(int), fj.astype(int); a, b = fi - i, fj - j
        z = g[i, j] * (1 - a) * (1 - b) + g[i + 1, j] * a * (1 - b) + g[i, j + 1] * (1 - a) * b + g[i + 1, j + 1] * a * b
        flat = z < 0.01
    for bx in tk['boxes']:
        c, s = math.cos(bx['yaw']), math.sin(bx['yaw']); d = xy - np.array(bx['center'][:2])
        lx, ly = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
        inside = (np.abs(lx) <= bx['half'][0]) & (np.abs(ly) <= bx['half'][1])
        top = bx['center'][2] + bx['half'][2]
        z = np.where(inside, np.maximum(z, top), z); flat &= ~inside
    return z, flat


def prop_dist(tk, xy):
    if tk['prop'] == 'driveway':
        return np.zeros(len(xy))
    d = np.full(len(xy), 99.0)
    for bx in tk['boxes']:
        c, s = math.cos(bx['yaw']), math.sin(bx['yaw']); q = xy - np.array(bx['center'][:2])
        lx, ly = np.abs(q[:, 0] * c + q[:, 1] * s) - bx['half'][0], np.abs(-q[:, 0] * s + q[:, 1] * c) - bx['half'][1]
        d = np.minimum(d, np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)))
    if tk['hmap']:
        x0, y0, res, g = hmap(tk['hmap']); I, J = np.where(g[::3, ::3] > 0.01)
        P = np.c_[x0 + 3 * res * I, y0 + 3 * res * J]
        for a in range(0, len(xy), 500):
            d[a:a + 500] = np.minimum(d[a:a + 500], np.sqrt(((xy[a:a + 500, None] - P[None]) ** 2).sum(-1)).min(1))
    return d


# ---------------------------------------------------------------- per-frame features
class Model:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mc.x2_spec().compile(); self.d = mujoco.MjData(self.m)
        m = self.m
        self.feet = []
        for side in ('left', 'right'):
            b = m.body(side + '_ankle_roll_link').id
            self.feet.append([g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE])
        self.head, self.pelvis = m.body('head_pitch_link').id, m.body('pelvis').id
        jid = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.rng = m.jnt_range[jid]; self.jname = [m.joint(j).name for j in jid]


def features(tk, M):
    mj, m, d = M.mj, M.m, M.d
    T, fps = len(tk['pos']), tk['fps']
    sole = np.zeros((T, 2, len(M.feet[0]), 3)); lean = np.zeros(T)
    for i in range(T):
        d.qpos[:] = mc.to_qpos(tk, i); mj.mj_kinematics(m, d)
        for f in range(2):
            sole[i, f] = d.geom_xpos[M.feet[f]]
        v = d.xpos[M.head] - d.xpos[M.pelvis]; lean[i] = math.degrees(math.acos(v[2] / np.linalg.norm(v)))
    S = sole.reshape(-1, 3)
    sz, sflat = surface(tk, S[:, :2])
    clr = (S[:, 2] - SOLE_R - sz).reshape(T, 2, -1)
    fl = sflat.reshape(T, 2, -1); sz = sz.reshape(T, 2, -1)
    clear = clr.min(2)                                         # (T,2) lowest sole point above its surface
    k = np.argmin(clr, 2)
    fsurf = np.take_along_axis(sz, k[..., None], 2)[..., 0]    # surface height under the lowest point
    fflat = fl.all(2)
    cen = sole.mean(2)                                         # (T,2,3)
    fv = np.gradient(cen, axis=0) * fps; fsp = np.hypot(fv[..., 0], fv[..., 1]); fvz = np.abs(fv[..., 2])
    fsp = np.stack([mc.smooth(fsp[:, f], 3) for f in range(2)], 1); fvz = np.stack([mc.smooth(fvz[:, f], 3) for f in range(2)], 1)
    stance = (fsp < 0.25) & (fvz < 0.15)
    ds = stance.all(1)
    # contact episodes (as tools/stairs_all_v1/build.py): median sole height of the episode against the nearest
    # surface height under the foot (boxes: the discrete levels under any sole point; height map / plane: the range)
    zlow = sole[..., 2].min(2) - SOLE_R
    eps = []
    for f in range(2):
        for i, j in runs(stance[:, f]):
            if j - i < 12:                 # shorter 'stances' are slow swing apexes (episode errors of 5-20 cm)
                continue
            zm = float(np.median(zlow[i:j + 1, f])); H = sz[i:j + 1, f]
            if tk['prop'] in ('ramp', 'driveway'):
                lo, hi = float(np.median(H.min(1))), float(np.median(H.max(1)))
                e = 0.0 if lo <= zm <= hi else (zm - hi if zm > hi else zm - lo)
            else:
                lv = np.unique(np.round(H, 3)); e = float((zm - lv)[np.argmin(np.abs(zm - lv))])
            eps.append((f, i, j, e))
    o, yaw0 = frame_of(tk)
    P = mc.smooth(tk['pos'][:, :2], 0.2 * fps)
    psi = np.unwrap(mc.yaw_xyzw(tk['quat']))
    V = mc.smooth(np.gradient(P, axis=0) * fps, 0.3 * fps); sp = np.hypot(*V.T)
    yr = mc.smooth(np.gradient(mc.smooth(psi, 0.3 * fps)) * fps, 0.3 * fps)
    c, s = np.cos(psi), np.sin(psi)
    vf, vl = V[:, 0] * c + V[:, 1] * s, -V[:, 0] * s + V[:, 1] * c
    pd = prop_dist(tk, tk['pos'][:, :2])
    fpd = np.minimum(prop_dist(tk, cen[:, 0, :2]), prop_dist(tk, cen[:, 1, :2]))
    # joints, root jumps
    over = np.maximum(M.rng[:, 0] - tk['dof'], tk['dof'] - M.rng[:, 1]).clip(0)
    step = np.r_[0, np.linalg.norm(np.diff(tk['pos'], axis=0), axis=1)]
    ystep = np.r_[0, np.abs(np.diff(psi))]
    return dict(T=T, fps=fps, eps=eps, clear=clear, fsurf=fsurf, fflat=fflat, cen=cen, stance=stance, ds=ds, sp=sp, vf=vf, vl=vl, yr=yr,
                psi=psi, pd=pd, fpd=fpd, over=over, step=step, ystep=ystep, lean=lean, o=o, yaw0=yaw0)


# ---------------------------------------------------------------- windows
def runs(mask):
    out = []; a = None
    for i, x in enumerate(np.r_[mask, False]):
        if x and a is None:
            a = i
        elif not x and a is not None:
            out.append((a, i - 1)); a = None
    return out


def ds_points(F):
    """middle frames of double-support phases."""
    return np.array([(a + b) // 2 for a, b in runs(F['ds'])], int)


def snap_out(F, a, b):
    D = ds_points(F)
    lo, hi = D[D <= a], D[D >= b]
    return (int(lo.max()) if len(lo) else None), (int(hi.min()) if len(hi) else None)


def fit_len(F, a, b, mode):
    """window [a,b] on double-support points, 1.5-5 s: grow outwards to the next DS points while short; chunk when long."""
    D = ds_points(F); fps = F['fps']; lo_n, hi_n = CONFIG['win_min_s'] * fps, CONFIG['win_max_s'] * fps
    if a is None or b is None or b <= a:
        return []
    while b - a < lo_n:
        L, R = D[D < a], D[D > b]
        if not len(L) and not len(R):
            return []
        if len(R) and (not len(L) or R.min() - b <= a - L.max()):
            b = int(R.min())
        else:
            a = int(L.max())
    if b - a <= hi_n:
        return [(a, b)]
    if mode == 'event':                          # keep it whole only if it fits; else chunk too
        pass
    out = []; s = a
    inner = D[(D > a) & (D <= b)]
    while True:
        cand = inner[(inner > s) & (inner - s <= hi_n)]
        if not len(cand):
            break
        e = int(cand.max())
        if e - s >= lo_n:
            out.append((s, e))
        s = e
        if b - s < lo_n:
            break
    return out


def steady_windows(F, mask, min_core):
    """runs of mask, cut inward to double-support points."""
    out = []
    D = ds_points(F)
    for a, b in runs(mask):
        if b - a < min_core:
            continue
        inn = D[(D >= a) & (D <= b)]
        if len(inn) < 2:
            continue
        out += fit_len(F, int(inn.min()), int(inn.max()), 'steady')
    return out


def segment(tk, F):
    fps = F['fps']; C = CONFIG; W = []
    T = F['T']
    floor_both = F['fflat'].all(1) & (tk['prop'] != 'driveway')
    # flat walk
    fw = floor_both & (F['vf'] >= C['flat']['v_fwd'][0]) & (F['vf'] <= C['flat']['v_fwd'][1]) & (np.abs(F['yr']) < C['flat']['yaw_rate']) \
        & (F['pd'] > C['flat']['prop_clear_m']) & (F['fpd'] > C['flat']['prop_clear_m']) & (np.abs(F['vl']) < 0.5 * np.abs(F['vf']) + 1e-9)
    for a, b in runs(fw):
        if b - a >= 0.5 * fps:
            A, B = snap_out(F, a, b)
            W += [('flat_walk', x, y) for x, y in fit_len(F, A, B, 'event')]
    # turn in place
    tm = (np.abs(F['yr']) > C['turn']['yaw_rate']) & (F['sp'] < C['turn']['speed'])
    for a, b in runs(tm):
        if b - a < 0.3 * fps:
            continue
        A, B = snap_out(F, a, b)
        W += [('turn_in_place', x, y) for x, y in fit_len(F, A, B, 'event')]
    # side walk
    n6 = 0.6 * fps; spS = mc.smooth(F['sp'], n6); vlS = mc.smooth(np.abs(F['vl']), n6); vfS = mc.smooth(np.abs(F['vf']), n6)
    sm = (spS > C['side']['speed']) & (vlS > 0.7 * np.hypot(vlS, vfS))
    for a, b in runs(sm):
        if b - a >= 0.5 * fps:
            A, B = snap_out(F, a, b)
            W += [('side_walk', x, y) for x, y in fit_len(F, A, B, 'event')]
    # terrain events
    if tk['prop'] in ('4x4', '2x2'):
        o, yaw = F['o'], F['yaw0']
        hl = tk['boxes'][0]['half'][0]
        fx = [mc.to_frame(F['cen'][:, f, :2], o, yaw) for f in range(2)]
        ev = []
        for f in range(2):
            x = fx[f][:, 0]; sgn = np.sign(x)
            for i in np.where(sgn[1:] * sgn[:-1] < 0)[0]:
                if abs(fx[f][i, 1]) <= hl + 0.15:
                    ev.append(i)
        ev.sort(); groups = []
        for i in ev:
            if groups and i - groups[-1][-1] < 2.5 * fps:
                groups[-1].append(i)
            else:
                groups.append([i])
        for g in groups:
            A, B = snap_out(F, g[0], g[-1])
            W += [('logs', x, y) for x, y in fit_len(F, A, B, 'event')]
    elif tk['prop'] in ('stairs', 'ramp'):
        lev = np.where(F['stance'], F['fsurf'], np.nan)
        for f in range(2):                                     # carry the last stance level through swing
            for i in range(1, T):
                if np.isnan(lev[i, f]):
                    lev[i, f] = lev[i - 1, f]
            first = np.where(~np.isnan(lev[:, f]))[0]
            if len(first):
                lev[:first[0], f] = lev[first[0], f]
        lev = np.nan_to_num(lev)
        off = (np.abs(lev) > 0.03).any(1) if tk['prop'] == 'ramp' else None
        if tk['prop'] == 'stairs':
            l0, l1 = lev[0].max(), lev[-1].max()
            off = (np.abs(lev - l0) > 0.03).any(1) | (np.abs(lev - l1) > 0.03).any(1)
            # only the part that changes level: from the first change away from the start to the last before the end
            idx = np.where((np.abs(lev - l0) > 0.03).any(1))[0]; idx2 = np.where((np.abs(lev - l1) > 0.03).any(1))[0]
            if len(idx) and len(idx2) and abs(l1 - l0) > 0.03:
                off = np.zeros(T, bool); off[idx[0]:idx2[-1] + 1] = True
        for a, b in runs(off):
            if b - a < 0.2 * fps:
                continue
            A, B = snap_out(F, a, b)
            W += [(tk['prop'], x, y) for x, y in fit_len(F, A, B, 'event')]
    elif tk['prop'] == 'driveway':
        wm = F['sp'] > 0.25
        W += [('ramp', a, b) for a, b in steady_windows(F, wm, 0.8 * fps)]
    return W


def describe(tk, F, cat, a, b, lean_hi):
    C = CONFIG['checks']; fps = F['fps']; sl = slice(a, b + 1)
    E = np.array([e for f, i, j, e in F['eps'] if j >= a and i <= b] or [0.0])
    fe = float(np.abs(E).max())                       # every contact episode in the window within 2 cm of its surface
    sink = float(max(0.0, -E.min()))
    jo = float(F['over'][sl].max()); jw = M_.jname[int(F['over'][sl].max(0).argmax())]
    rs, ys = float(F['step'][a + 1:b + 1].max()), float(F['ystep'][a + 1:b + 1].max())
    ln = float(F['lean'][sl].max())
    checks = dict(foot_err_max_cm=round(fe * 100, 2), sink_max_cm=round(sink * 100, 2), joint_over_rad=round(jo, 3), joint_worst=jw,
                  root_step_max_m=round(rs, 3), root_yaw_step_max_rad=round(ys, 3), lean_max_deg=round(ln, 1), lean_hi_deg=round(lean_hi, 1),
                  ds_start=bool(F['ds'][a]), ds_end=bool(F['ds'][b]))
    ok = fe <= C['foot_err_m'] and sink <= C['sink_m'] and jo <= C['joint_tol_rad'] and rs <= C['root_step_m'] \
        and ys <= C['root_yaw_step_rad'] and ln <= lean_hi and checks['ds_start'] and checks['ds_end']
    checks['ok'] = bool(ok)
    clean = fe / C['foot_err_m'] + max(0, sink) / C['sink_m'] + ln / lean_hi + rs / C['root_step_m']
    # variety features
    dz = float(tk['pos'][b, 2] - tk['pos'][a, 2]); spd = float(F['sp'][sl].mean())
    head = float(mc.wrap(F['psi'][a] - F['yaw0'])); turn = float(F['psi'][b] - F['psi'][a])
    lat = float(np.sign(F['vl'][sl].mean()))
    side = 'mir' if tk['mir'] else 'orig'
    if cat in ('stairs', 'ramp'):
        dirn = 'up' if dz > 0.05 else 'down' if dz < -0.05 else 'across'
    elif cat == 'turn_in_place':
        dirn = 'left' if turn > 0 else 'right'
    elif cat == 'side_walk':
        dirn = 'left' if lat > 0 else 'right'
    elif cat == 'logs':
        off = abs(float(mc.wrap(math.atan2(*(F['cen'][b, 0, :2] - F['cen'][a, 0, :2])[::-1]) - F['psi'][a])))
        dirn = 'forward' if off < math.radians(50) else 'sideways' if off < math.radians(130) else 'backward'
    else:
        dirn = 'left' if F['yr'][sl].mean() > 0.05 else 'right' if F['yr'][sl].mean() < -0.05 else 'straight'
    ang = abs(head) if cat in ('logs', 'stairs', 'ramp') else 0.0
    ang = min(ang, math.pi - ang) if cat == 'logs' else ang
    feat = dict(dir=dirn, mirror=side, speed='slow' if spd < 0.5 else 'normal', prop=tk['prop'],
                angle='square' if math.degrees(ang) < 25 else 'oblique')
    nums = dict(seconds=round((b - a) / fps, 2), dz_m=round(dz, 3), speed_mps=round(spd, 2), heading_vs_prop_deg=round(math.degrees(head), 1),
                turn_deg=round(math.degrees(turn), 1), lateral_share=round(lat_share(F, a, b), 2),
                flat_frac=round(float(np.mean(F['fflat'][sl].all(1) & (F['pd'][sl] > CONFIG['flat']['prop_clear_m']) & (F['fpd'][sl] > CONFIG['flat']['prop_clear_m'])
                                              & (tk['prop'] != 'driveway'))), 2),
                floor_all=bool(F['fflat'][sl].all() and tk['prop'] != 'driveway'),
                vf_mean=round(float(F['vf'][sl].mean()), 2), yaw_rate_absmean=round(float(np.abs(F['yr'][sl]).mean()), 2))
    return dict(take=tk['id'], set=tk['set'], cat=cat, a=int(a), b=int(b), t0=round(a / fps, 2), t1=round(b / fps, 2), checks=checks,
                clean=round(clean, 3), feat=feat, nums=nums)


def lat_share(F, a, b):
    d = np.abs(np.c_[F['vf'][a:b + 1], F['vl'][a:b + 1]]); t = d.sum()
    return float(d[:, 1].sum() / t) if t > 0 else 0.0


def valid(w):
    c, n = w['cat'], w['nums']
    if c == 'turn_in_place':                 # on the floor only: no step onto or over a prop inside the window
        return n['floor_all'] and abs(n['turn_deg']) >= CONFIG['turn']['min_turn_deg']
    if c == 'flat_walk':
        f = CONFIG['flat']
        return n['floor_all'] and n['flat_frac'] >= f['clear_frac_min'] and f['v_fwd'][0] <= n['vf_mean'] <= f['v_fwd'][1] and n['yaw_rate_absmean'] < f['yaw_rate']
    if c == 'side_walk':
        return n['lateral_share'] > CONFIG['side']['lateral_share']
    if c == 'stairs':
        return abs(n['dz_m']) > 0.05 or w['feat']['dir'] == 'across'
    return True


def select(cands, cat):
    cfg = CONFIG['per_category'][cat]; lim = cfg['seconds']; nmax = cfg['n_max']
    pool = [w for w in cands if w['cat'] == cat and w['checks']['ok'] and valid(w)]
    chosen = []; tot = 0.0; uses = {}; seen = {k: set() for k in ('dir', 'mirror', 'speed', 'prop', 'angle')}
    combos = set()

    def overlaps(w):
        return any(c['take'] == w['take'] and not (w['b'] <= c['a'] or w['a'] >= c['b']) for c in chosen)
    while len(chosen) < nmax:
        best = None
        for w in pool:
            if w in chosen or overlaps(w) or tot + w['nums']['seconds'] > lim:
                continue
            base = w['take'].replace('_mir', '')
            if uses.get(base, 0) >= CONFIG['max_uses_per_take']:
                continue
            new = sum(w['feat'][k] not in seen[k] for k in seen)
            combo = tuple(w['feat'][k] for k in ('dir', 'mirror', 'speed', 'prop'))
            score = (new, combo not in combos, -uses.get(base, 0), -sum(c['feat']['mirror'] == w['feat']['mirror'] for c in chosen), -w['clean'])
            if best is None or score > best[0]:
                best = (score, w)
        if best is None:
            break
        w = best[1]; chosen.append(w); tot += w['nums']['seconds']
        b = w['take'].replace('_mir', ''); uses[b] = uses.get(b, 0) + 1
        for k in seen:
            seen[k].add(w['feat'][k])
        combos.add(tuple(w['feat'][k] for k in ('dir', 'mirror', 'speed', 'prop')))
    return chosen, pool


M_ = None


def cmd_segment():
    global M_
    import joblib
    M_ = Model()
    takes = load(); print('takes', len(takes), flush=True)
    feats = {}
    for tk in takes:
        feats[tk['id']] = features(tk, M_)
    lean_all = np.concatenate([F['lean'] for F in feats.values()])
    lean_hi = float(np.percentile(lean_all, CONFIG['checks']['lean_pct']))
    print('lean p50 %.1f p95 %.1f hi(p%.1f) %.1f' % (np.percentile(lean_all, 50), np.percentile(lean_all, 95), CONFIG['checks']['lean_pct'], lean_hi))
    cands = []
    for tk in takes:
        F = feats[tk['id']]
        for cat, a, b in segment(tk, F):
            cands.append(describe(tk, F, cat, a, b, lean_hi))
    print('candidates', len(cands))
    sel = {}; stats = {}
    for cat in CATS:
        ch, pool = select(cands, cat); sel[cat] = ch
        allc = [w for w in cands if w['cat'] == cat]
        fails = {}
        for w in allc:
            if not w['checks']['ok']:
                for k, bad in (('foot', w['checks']['foot_err_max_cm'] > 2), ('sink', w['checks']['sink_max_cm'] > 2), ('joint', w['checks']['joint_over_rad'] > 0.01),
                               ('lean', w['checks']['lean_max_deg'] > lean_hi), ('root', w['checks']['root_step_max_m'] > 0.05 or w['checks']['root_yaw_step_max_rad'] > 0.15)):
                    fails[k] = fails.get(k, 0) + bad
        stats[cat] = dict(candidates=len(allc), clean=len(pool), clean_seconds=round(sum(w['nums']['seconds'] for w in pool), 1),
                          fail_reasons=fails, chosen=len(ch), seconds=round(sum(w['nums']['seconds'] for w in ch), 1),
                          takes=len({w['take'].replace('_mir', '') for w in ch}), mirrored=sum(w['feat']['mirror'] == 'mir' for w in ch))
        print(cat, json.dumps(stats[cat]))
    OUT.mkdir(parents=True, exist_ok=True)
    byid = {t['id']: t for t in takes}
    clips = {}; meta = []
    for cat in CATS:
        for k, w in enumerate(sel[cat]):
            tk = byid[w['take']]; a, b = w['a'], w['b']
            name = '%s_%02d_%s_%.1f' % (cat, k, w['take'], w['t0'])
            terr = dict(prop=tk['prop'], boxes=tk['boxes'])
            if tk['hmap']:
                terr['hmap'] = tk['hmap'] + '.npz'
            if tk['plane']:
                terr['plane'] = tk['plane']
            clips[name] = dict(root_trans_offset=tk['pos'][a:b + 1].astype(np.float32), root_rot=tk['quat'][a:b + 1].astype(np.float32),
                               dof=tk['dof'][a:b + 1].astype(np.float32), fps=tk['fps'], category=cat, terrain=terr,
                               source=dict(set=tk['set'], take=tk['id'], start_frame=a, end_frame=b))
            meta.append(dict(clip=name, **{x: w[x] for x in ('take', 'set', 't0', 't1', 'feat', 'nums', 'checks', 'clean')}, category=cat))
    joblib.dump(clips, OUT / 'pico_style_v1.pkl', compress=3)
    tot = sum(s['seconds'] for s in stats.values())
    json.dump(dict(generated_by='tools/style/build_pico_style.py', config=CONFIG, lean_hi_deg=round(lean_hi, 1), total_seconds=round(tot, 1),
                   stats=stats, clips=meta), open(OUT / 'pico_style_v1_meta.json', 'w'), indent=1)
    json.dump(cands, open(OUT / 'candidates.json', 'w'))
    json.dump({k: [[int(f), int(i), int(j), round(e, 4)] for f, i, j, e in F['eps']] for k, F in feats.items()}, open(OUT / 'episodes_debug.json', 'w'))
    print('total seconds %.1f clips %d' % (tot, len(clips)))


# ---------------------------------------------------------------- review
def cmd_review():
    import joblib, mujoco
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    meta = json.load(open(OUT / 'pico_style_v1_meta.json')); clips = joblib.load(OUT / 'pico_style_v1.pkl')
    R = OUT / 'review'; R.mkdir(exist_ok=True)
    G = CONFIG['gif']; Wd, Ht = G['w'], G['h']
    for cat in CATS:
        ms = [m for m in meta['clips'] if m['category'] == cat]
        if not ms:
            continue
        # contact sheet
        n = len(ms); nc = min(6, n); nr = int(math.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.7 * nr), squeeze=False)
        for ax in axs.ravel():
            ax.axis('off')
        for k, m in enumerate(ms):
            c = clips[m['clip']]; ax = axs.ravel()[k]; ax.axis('on')
            tk = dict(prop=c['terrain']['prop'], boxes=c['terrain']['boxes'], hmap=c['terrain'].get('hmap', '').replace('.npz', '') or None,
                      plane=c['terrain'].get('plane'), pos=np.asarray(c['root_trans_offset'], float))
            P = tk['pos'][:, :2]; ctr = P.mean(0)
            for bx in tk['boxes']:
                hx, hy = bx['half'][:2]; cc, s = math.cos(bx['yaw']), math.sin(bx['yaw'])
                cor = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
                ax.add_patch(Polygon(np.array(bx['center'][:2]) + np.c_[cor[:, 0] * cc - cor[:, 1] * s, cor[:, 0] * s + cor[:, 1] * cc],
                                     fc='#d8b48a', ec='#8a6a40', lw=0.6))
            if tk['hmap']:
                x0, y0, res, g = hmap(tk['hmap'])
                ax.imshow(g.T, origin='lower', extent=[x0, x0 + res * g.shape[0], y0, y0 + res * g.shape[1]], cmap='copper_r', alpha=0.5, vmin=0, vmax=0.3)
            q = np.asarray(c['root_rot'], float); psi = mc.yaw_xyzw(q)
            ax.plot(P[:, 0], P[:, 1], '-', color='#2a78d6', lw=1.4)
            ax.plot(P[0, 0], P[0, 1], 'o', color='#1baf7a', ms=4); ax.plot(P[-1, 0], P[-1, 1], 's', color='#e34948', ms=4)
            for i in range(0, len(P), max(1, len(P) // 6)):
                ax.arrow(P[i, 0], P[i, 1], 0.15 * math.cos(psi[i]), 0.15 * math.sin(psi[i]), head_width=0.05, color='#555', lw=0.5)
            r = max(1.0, np.abs(P - ctr).max() + 0.4)
            ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r); ax.set_aspect('equal')
            ax.tick_params(labelsize=5)
            f = m['feat']; ck = m['checks']
            ax.set_title('%d %s %.1f-%.1fs\n%s %s %s %s | foot %.1fcm lean %.0f' % (k + 1, mc.short(m['take']) if hasattr(mc, 'short') else m['take'],
                         m['t0'], m['t1'], f['dir'], f['mirror'], f['speed'], f['prop'], ck['foot_err_max_cm'], ck['lean_max_deg']), fontsize=6)
        fig.suptitle('%s: %d windows, %.1f s (top view, world frame; blue = root path, green start, red end, arrows = heading)' % (
            cat, n, sum(m['nums']['seconds'] for m in ms)), fontsize=8)
        fig.tight_layout(); fig.savefig(R / ('sheet_%s.png' % cat), dpi=110); plt.close(fig)
        # GIF: windows one after another, fixed camera per window (small GIF deltas)
        tot = sum(m['nums']['seconds'] for m in ms)
        frames, labels, wid = [], [], []                    # rendered once, real time at G['fps']; faster playback drops frames
        for k, m in enumerate(ms):
            c = clips[m['clip']]
            tk = dict(prop=c['terrain']['prop'], boxes=c['terrain']['boxes'], pos=np.asarray(c['root_trans_offset'], float),
                      quat=np.asarray(c['root_rot'], float), dof=np.asarray(c['dof'], float))
            ctx = {}
            if tk['prop'] == 'ramp':
                ctx['hmap'] = hmap(c['terrain']['hmap'].replace('.npz', ''))
            if tk['prop'] == 'driveway':
                ctx['driveway'] = c['terrain']['plane']
            mp = dict(tk, prop='stairs1' if tk['prop'] == 'stairs' else tk['prop'])
            mdl = mc.build_model(mp, ctx); d = mujoco.MjData(mdl)
            T = len(tk['pos']); fpsc = float(c['fps'])
            idx = np.clip((np.arange(0, T / fpsc, 1 / G['fps']) * fpsc).astype(int), 0, T - 1)
            P = tk['pos']; ctr = P.mean(0); ext = np.abs(P[:, :2] - ctr[:2]).max()
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 2.6 + 1.2 * ext; cam.elevation = -20
            cam.azimuth = math.degrees(float(np.mean(mc.yaw_xyzw(tk['quat'])))) + 115
            cam.lookat[:] = [ctr[0], ctr[1], ctr[2] - 0.25]
            with mujoco.Renderer(mdl, Ht, Wd) as r:
                for i in idx:
                    d.qpos[:] = mc.to_qpos(tk, i); mujoco.mj_forward(mdl, d)
                    r.update_scene(d, cam); frames.append(r.render().copy()); wid.append(k)
                    labels.append('%s %d/%d  %s  %.1f-%.1f s  %s %s' % (cat, k + 1, len(ms), m['take'], m['t0'], m['t1'], m['feat']['dir'], m['feat']['mirror']))
        wid = np.array(wid)
        for colors, spd in ((48, 1), (48, 2), (32, 3), (32, 4), (24, 5), (24, 6)):
            pick = [i for k in range(len(ms)) for i in np.where(wid == k)[0][::spd]]
            l2 = 'MuJoCo kinematic playback, not the training simulator' + ('' if spd == 1 else '  (x%d speed)' % spd)
            ims = [mc.caption(Image.fromarray(frames[i]), labels[i], l2) for i in pick]
            sheet = Image.new('RGB', (Wd * 8, Ht))
            for j, im in enumerate(ims[::max(1, len(ims) // 8)][:8]):
                sheet.paste(im, (j * Wd, 0))
            pal = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
            qs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
            path = R / ('play_%s.gif' % cat)
            qs[0].save(path, save_all=True, append_images=qs[1:], duration=int(1000 / G['fps']), loop=0, optimize=False, disposal=1)
            nb = os.path.getsize(path)
            print('gif', cat, len(ms), 'windows', '%.1f s' % tot, 'x%d' % spd, nb // 1000, 'kB', flush=True)
            if nb <= G['max_bytes']:
                break
        # strips for checking: 4 frames per window
        st = Image.new('RGB', (Wd // 2 * 4, Ht // 2 * len(ms)))
        off = 0
        for k, m in enumerate(ms):
            nfr = sum(1 for l in labels if l.startswith('%s %d/' % (cat, k + 1)))
            for j, fr in enumerate((0.05, 0.35, 0.65, 0.95)):
                st.paste(Image.fromarray(frames[off + int(fr * (nfr - 1))]).resize((Wd // 2, Ht // 2)), (j * Wd // 2, k * Ht // 2))
            off += nfr
        st.save(R / ('strip_%s.png' % cat))


if __name__ == '__main__':
    a = sys.argv[1:]
    if '--config' in a:
        i = a.index('--config'); CONFIG.update(json.load(open(a[i + 1]))); del a[i:i + 2]
    what = a[0] if a else 'all'
    if what in ('segment', 'all'):
        cmd_segment()
    if what in ('review', 'all'):
        cmd_review()
