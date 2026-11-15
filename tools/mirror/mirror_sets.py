"""Left/right mirrored versions of the Pico terrain takes (X2 retargets), with their mirrored props.

    python tools/mirror/mirror_sets.py map      # mirror map from the MJCF + forward-kinematics check -> x2_mirror_map.json
    python tools/mirror/mirror_sets.py sets     # map + mirrored sets, props, height map, checks -> data/train_sets/*_mir*
    python tools/mirror/mirror_sets.py gifs     # 6 catalogue-style GIFs + 1 side-by-side -> catalog/gifs_mirror/
    python tools/mirror/mirror_sets.py all

Mirror. Every take is mirrored about the vertical plane through its prop's own axis (the catalogue's prop frame,
catalog/make_catalog.py base_frame: beams = the crossing axis through the beam centre; stairs = the up axis through the
centre of the front edge of step 1; ramp = the up-slope axis through the foot of the slope at its width centre;
driveway = the uphill axis through the centre of the takes). In that frame: root y -> -y, root rotation R -> M R M
(M = diag(1,-1,1)), dof via x2_mirror_map.json (left <-> right partner, sign from the joint axes). The props are mirrored
the same way: boxes (centre reflected, yaw -> 2 psi - yaw), the ramp height map (resampled, bilinear), the driveway plane.
A mirrored take must always be paired with its mirrored prop (the *_mir props / height map), never the original.

Sources: data/train_sets/tp_t4x4u_y20, pico_2026-10-29_2x2_train, tp_stairs_all_v1 (+meta), acp_tp_ramp_sh2_x30 (base
takes only: the _sh variants are stride/height variants of the same 15 takes), data/2026-10-30/x2/record_20261030_121[01]*
(driveway, 50 fps nearest-sample as the catalogue). CPU only, at most 3 threads.
"""
import os, sys, json, math, copy
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TS = REPO / 'data' / 'train_sets'
sys.path.insert(0, str(REPO / 'catalog'))
import make_catalog as mc          # noqa: E402  (loading, prop frames, ramp / driveway fits, classification, GIF helpers)

MAP_JSON = HERE / 'x2_mirror_map.json'
CLAMP_TOL = 0.02
SETS = {   # out set name: (source description, props kind)
    '4x4': 'tp_t4x4u_y20_mir', '2x2': 'pico_2026-10-29_2x2_train_mir', 'stairs': 'tp_stairs_all_v1_mir',
    'ramp': 'tp_ramp_base_x30_mir', 'driveway': 'tp_driveway_v1_mir'}


# ---------------------------------------------------------------- quaternions (xyzw)
def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0); bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qz(psi):
    return np.array([0, 0, math.sin(psi / 2), math.cos(psi / 2)])


def qmir_y(q):                    # M R M with M = diag(1,-1,1): (x,y,z,w) -> (-x, y, -z, w)
    return q * np.array([-1, 1, -1, 1])


def mirror_quat(q, psi):          # R' = Rz(psi) M Rz(-psi) R M
    out = qmul(qz(psi), qmir_y(qmul(qz(-psi), q)))
    s = np.sign(np.sum(out[1:] * out[:-1], -1)); s = np.r_[1, np.cumprod(np.where(s == 0, 1, s))]   # keep the sign continuous
    return out * s[:, None]


def mirror_xy(xy, o, psi):        # reflection about the line through o with direction psi
    c, s = math.cos(2 * psi), math.sin(2 * psi); S = np.array([[c, s], [s, -c]])
    return (np.asarray(xy)[..., :2] - o) @ S.T + o


# ---------------------------------------------------------------- mirror map from the MJCF
def model():
    import mujoco
    return mc.x2_spec().compile()


def partner(name):
    return name.replace('left_', '@').replace('right_', 'left_').replace('@', 'right_')


def build_map(seed=0, n=20):
    import mujoco
    m = model(); d = mujoco.MjData(m)
    names = [m.joint(j).name for j in range(1, m.njnt)]
    assert len(names) == 31
    mujoco.mj_kinematics(m, d)                                   # qpos0: world axes of every joint
    M = np.diag([1., -1., 1.])
    axes = {m.joint(j).name: d.xaxis[j].copy() for j in range(1, m.njnt)}
    perm, sign, rows = [], [], []
    for i, n_ in enumerate(names):
        p = partner(n_); a = axes[n_]; b = axes[p]
        pa = -(M @ a)                                            # an axis is a pseudovector: under a reflection it maps to -M a
        dot = float(pa @ b)
        assert abs(abs(dot) - 1) < 1e-3, (n_, a, b)
        perm.append(names.index(p)); sign.append(int(round(dot)))
        jid = m.joint(n_).id
        rows.append(dict(joint=n_, index=i, partner=p, partner_index=names.index(p), sign=int(round(dot)), axis_dot=round(dot, 6),
                         axis_world_qpos0=[round(float(x), 6) for x in a], range=[round(float(x), 4) for x in m.jnt_range[jid]]))
    perm = np.array(perm); sign = np.array(sign, float)
    # numeric check: FK of the mirrored pose = mirrored body positions / orientations of the original pose
    bpart = [m.body(partner(m.body(b).name)).id for b in range(m.nbody)]
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]

    def fk_check(mm):
        dd = mujoco.MjData(mm); rng = np.random.default_rng(seed)
        ep = np.zeros(mm.nbody); er = np.zeros(mm.nbody)
        for _ in range(n):
            q = np.zeros(38); q[:3] = rng.uniform(-2, 2, 3) + [0, 0, 1]
            qq = rng.normal(size=4); qq /= np.linalg.norm(qq); q[3:7] = qq      # wxyz
            q[7:] = rng.uniform(lo, hi)
            dd.qpos[:] = q; mujoco.mj_kinematics(mm, dd); P = dd.xpos.copy(); R = dd.xmat.reshape(-1, 3, 3).copy()
            qm = q.copy(); qm[1] = -q[1]
            xyzw = qmir_y(q[[4, 5, 6, 3]]); qm[3:7] = xyzw[[3, 0, 1, 2]]
            qm[7:] = sign * q[7:][perm]
            dd.qpos[:] = qm; mujoco.mj_kinematics(mm, dd); P2 = dd.xpos.copy(); R2 = dd.xmat.reshape(-1, 3, 3).copy()
            for b in range(1, mm.nbody):
                ep[b] = max(ep[b], float(np.linalg.norm(P2[bpart[b]] - M @ P[b])))
                er[b] = max(er[b], float(np.abs(R2[bpart[b]] - M @ R[b] @ M).max()))
        return ep, er
    # (a) a symmetrized copy of the model (every left body / joint set to the exact mirror of its right partner):
    #     tests the map alone; (b) the real model: adds the CAD's own small left/right differences
    ms = model()
    for b in range(1, ms.nbody):
        nm = ms.body(b).name
        if nm.startswith('left_'):
            p = ms.body(partner(nm)).id
            ms.body_pos[b] = M @ ms.body_pos[p]; ms.body_quat[b] = ms.body_quat[p] * np.array([1, -1, 1, -1])
    for j in range(1, ms.njnt):
        nm = ms.joint(j).name
        if nm.startswith('left_'):
            p = ms.joint(partner(nm)).id
            ms.jnt_axis[j] = sign[j - 1] * -(M @ ms.jnt_axis[p]); ms.jnt_pos[j] = M @ ms.jnt_pos[p]
    eps, ers = fk_check(ms)
    ep, er = fk_check(m)
    worst_p, worst_r = float(eps.max()), float(ers.max())
    real = {m.body(b).name: round(1000 * float(ep[b]), 3) for b in range(1, m.nbody)}
    legs = [b for b in range(1, m.nbody) if not any(s in m.body(b).name for s in ('shoulder', 'elbow', 'wrist'))]
    real_worst_legs_torso = float(ep[legs].max())
    res = dict(generated_by='tools/mirror/mirror_sets.py map', model=str(mc.x2_spec.__module__ and 'agibot-x2-record-and-replay/sim/x2_ultra.xml'),
               convention='dof_mirrored[i] = sign[i] * dof[perm[i]]; dof in MuJoCo order (31 joints after the free joint). Mirror about '
                          'the robot sagittal x-z plane. Root: pos y -> -y, quat xyzw (x,y,z,w) -> (-x,y,-z,w) (R -> M R M, M=diag(1,-1,1)).',
               rule='sign = (-M a_joint) . a_partner with a = joint axis in world at qpos0 (axis is a pseudovector): pitch (y) keeps '
                    'its sign, roll (x) and yaw (z) flip.',
               perm=perm.tolist(), sign=sign.astype(int).tolist(), joints=rows,
               fk_check=dict(poses=n, seed=seed, bodies=m.nbody - 1,
                             symmetrized_model=dict(max_body_pos_err_mm=round(worst_p * 1000, 6), max_body_rotmat_err=round(worst_r, 8),
                                                    passed=bool(worst_p < 1e-3)),
                             real_model=dict(max_body_pos_err_mm=round(1000 * float(ep.max()), 3),
                                             max_legs_pelvis_torso_head_mm=round(1000 * real_worst_legs_torso, 3),
                                             per_body_mm=real, max_body_rotmat_err=round(float(er.max()), 5),
                                             note='x2_ultra.xml is not exactly left/right symmetric (CAD export): left_hip_roll / hip_yaw '
                                                  'offsets differ by 0.6 mm, shoulder_yaw / wrist_yaw / wrist_pitch by 0.5-0.7 mm and '
                                                  '0.003 rad (0.35 deg) in orientation; the arm residual is that, not the map.')))
    json.dump(res, open(MAP_JSON, 'w'), indent=1)
    print('map: FK check %d poses, %d bodies. Symmetrized model: max body position error %.2e mm, rotation %.1e -> %s. '
          'Real model: max %.2f mm (legs/pelvis/torso/head %.2f mm), worst %s'
          % (n, m.nbody - 1, worst_p * 1000, worst_r, 'PASS' if worst_p < 1e-3 else 'FAIL', 1000 * ep.max(), 1000 * real_worst_legs_torso,
             sorted(real.items(), key=lambda x: -x[1])[:4]))
    return perm, sign, m


# ---------------------------------------------------------------- props
def mirror_box(b, o, psi):
    b2 = copy.deepcopy(b); c = np.array(b['center'], float)
    c[:2] = mirror_xy(c[:2], o, psi); b2['center'] = [float(x) for x in c]
    b2['yaw'] = float(mc.wrap(2 * psi - b['yaw']))
    return b2


def box_diff(A, B):
    """Largest difference (m) between two box lists, boxes matched in order; a box is the same under yaw + pi."""
    w = 0.0
    for a, b in zip(A, B):
        def corners(x):
            hx, hy = x['half'][:2]; c, s = math.cos(x['yaw']), math.sin(x['yaw'])
            k = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
            return np.array(x['center'][:2]) + np.c_[k[:, 0] * c - k[:, 1] * s, k[:, 0] * s + k[:, 1] * c]
        ca, cb = corners(a), corners(b)
        w = max(w, max(np.linalg.norm(ca[i] - cb, axis=1).min() for i in range(4)), abs(a['center'][2] - b['center'][2]))
    return w


def mirror_hmap(hm, o, psi):
    x0, y0, res, g = hm
    I, J = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]), indexing='ij')
    XY = np.c_[x0 + res * I.ravel(), y0 + res * J.ravel()]
    src = mirror_xy(XY, o, psi)                                  # a reflection is its own inverse
    return bilinear(hm, src).reshape(g.shape)


def bilinear(hm, xy):
    x0, y0, res, g = hm
    fi = (xy[:, 0] - x0) / res; fj = (xy[:, 1] - y0) / res
    inside = (fi >= 0) & (fi <= g.shape[0] - 1) & (fj >= 0) & (fj <= g.shape[1] - 1)
    i0 = np.clip(np.floor(fi).astype(int), 0, g.shape[0] - 2); j0 = np.clip(np.floor(fj).astype(int), 0, g.shape[1] - 2)
    ti = np.clip(fi - i0, 0, 1); tj = np.clip(fj - j0, 0, 1)
    h = (g[i0, j0] * (1 - ti) * (1 - tj) + g[i0 + 1, j0] * ti * (1 - tj) + g[i0, j0 + 1] * (1 - ti) * tj + g[i0 + 1, j0 + 1] * ti * tj)
    return np.where(inside, h, 0.0)


def mirror_plane(dw, o, psi):
    c2, s2 = math.cos(2 * psi), math.sin(2 * psi); S = np.array([[c2, s2], [s2, -c2]])
    ab = np.array([dw['a'], dw['b']]); ab2 = S @ ab                      # z = ab . p = ab . (o + S(p' - o)) for the mirrored point
    out = dict(dw); out['a'], out['b'] = float(ab2[0]), float(ab2[1]); out['c'] = float(dw['c'] + ab @ (o - S @ o))
    return out


# ---------------------------------------------------------------- terrain height and checks
def terrain_fn(prop, boxes, ctx):
    if prop == 'ramp':
        return lambda xy: bilinear(ctx['hmap'], xy)
    if prop == 'driveway':
        dw = ctx['driveway']; return lambda xy: dw['a'] * xy[:, 0] + dw['b'] * xy[:, 1] + dw['c']

    def f(xy):
        h = np.zeros(len(xy))
        for b in boxes:
            c, s = math.cos(b['yaw']), math.sin(b['yaw']); d = xy - np.array(b['center'][:2])
            lx = d[:, 0] * c + d[:, 1] * s; ly = -d[:, 0] * s + d[:, 1] * c
            inn = (np.abs(lx) <= b['half'][0]) & (np.abs(ly) <= b['half'][1])
            h = np.where(inn, np.maximum(h, b['center'][2] + b['half'][2]), h)
        return h
    return f


class Feet:
    def __init__(self, m):
        import mujoco
        self.mj = mujoco; self.m = m; self.d = mujoco.MjData(m)
        self.feet = [m.body('left_ankle_roll_link').id, m.body('right_ankle_roll_link').id]
        self.sole = {b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == 2 and m.geom_pos[g][2] < -0.06]
                     for b in self.feet}

    def run(self, pos, quat, dof):
        T = len(pos); A = np.zeros((T, 2, 3)); S = np.zeros((T, 2, len(self.sole[self.feet[0]]), 3)); R = self.m.geom_size[self.sole[self.feet[0]][0]][0]
        for i in range(T):
            self.d.qpos[:3] = pos[i]; self.d.qpos[3] = quat[i, 3]; self.d.qpos[4:7] = quat[i, :3]; self.d.qpos[7:] = dof[i]
            self.mj.mj_kinematics(self.m, self.d)
            for k, b in enumerate(self.feet):
                A[i, k] = self.d.xpos[b]
                for g, gid in enumerate(self.sole[b]):
                    S[i, k, g] = self.d.geom_xpos[gid]
        return A, S, R


def contact_errors(feet, c, prop, boxes, ctx):
    """Stance episodes per foot (ankle body horizontal and vertical speed < 0.2 m/s for >= 0.12 s); per episode the
    median clearance of the lowest sole sphere above the terrain under it (m). Same procedure for originals and mirrors."""
    fps = float(c['fps']); A, S, r = feet.run(c['root_trans_offset'], c['root_rot'], c['dof']); h = terrain_fn(prop, boxes, ctx)
    out = []
    for k in range(2):
        v = np.gradient(A[:, k], axis=0) * fps; st = (np.hypot(v[:, 0], v[:, 1]) < 0.2) & (np.abs(v[:, 2]) < 0.2)
        pts = S[:, k].reshape(-1, 3); clr = (pts[:, 2] - r - h(pts[:, :2])).reshape(S.shape[0], S.shape[2]).min(1)
        i = 0
        while i < len(st):
            if st[i]:
                j = i
                while j + 1 < len(st) and st[j + 1]:
                    j += 1
                if (j - i + 1) / fps >= 0.12:
                    out.append((k, i, j, float(np.median(clr[i:j + 1]))))
                i = j + 1
            else:
                i += 1
    return out


def root_checks(c):
    fps = float(c['fps']); p = np.asarray(c['root_trans_offset']); q = np.asarray(c['root_rot'])
    sp = (np.linalg.norm(np.diff(p, axis=0), axis=1) * fps).max()
    ang = 2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], 1)), 0, 1)); w = (ang * fps).max()
    return float(sp), float(w)


# ---------------------------------------------------------------- the sets
def load_all():
    takes = mc.load_takes()
    ctx = dict(ramp=mc.ramp_prop(), hmap=mc.ramp_hmap()); ctx['driveway'] = mc.fit_driveway(takes)
    mc.classify(takes, ctx)
    return takes, ctx


def mirror_ctx(ctx):
    o = np.array(ctx['ramp']['origin'][:2]); psi = ctx['ramp']['yaw']
    x0, y0, res, g = ctx['hmap']
    dw = ctx['driveway']; od = np.array(dw['centre']); pd = math.atan2(dw['b'], dw['a'])
    cm = dict(ramp=dict(ctx['ramp']), hmap=(x0, y0, res, mirror_hmap(ctx['hmap'], o, psi)), driveway=mirror_plane(dw, od, pd))
    return cm


def mirror_take(tk, ctx, perm, sign):
    o, psi = mc.base_frame(tk, ctx); o = np.asarray(o, float)
    pos = tk['pos'].copy(); pos[:, :2] = mirror_xy(pos[:, :2], o, psi)
    quat = mirror_quat(tk['quat'], psi)
    dof = sign * tk['dof'][:, perm]
    boxes = [mirror_box(b, o, psi) for b in tk['boxes']]
    t2 = dict(tk); t2.update(id=tk['id'] + '_mir', pos=pos, quat=quat, dof=dof, boxes=boxes, mirror=dict(origin=o.tolist(), psi=float(psi)))
    for k in ('category', 'numbers', 'frame', 'P'):
        t2.pop(k, None)
    return t2


def run_sets():
    import joblib
    perm, sign, m = build_map()
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    takes, ctx = load_all(); cm = mirror_ctx(ctx); feet = Feet(m)
    mir = [mirror_take(t, ctx, perm, sign) for t in takes]
    mc.classify(mir, cm)
    report = dict(sets={}, props={}, clips={})
    out = {k: ({}, {}, {}) for k in SETS}                         # clips, props, meta
    smeta = json.load(open(TS / 'tp_stairs_all_v1_meta.json'))
    for t0, t in zip(takes, mir):
        key = 'stairs' if t['prop'].startswith('stairs') else t['prop']
        rec = dict(prop=t['prop'], category_orig=t0['category'], category_mir=t['category'])
        # joint limits
        over = np.maximum(lo - t['dof'], t['dof'] - hi); worst = float(over.max())
        over0 = float(np.maximum(lo - t0['dof'], t0['dof'] - hi).max())
        rec['joint_over_orig_rad'] = round(max(over0, 0), 4); rec['joint_over_mir_rad'] = round(max(worst, 0), 4)
        rec['joint_worst_mir'] = mc_jname(m, int(np.unravel_index(over.argmax(), over.shape)[1])) if worst > 0 else ''
        rec['clamped_values'] = int(((over > 1e-6) & (over <= CLAMP_TOL)).sum())      # 1e-10 rad float noise at a limit (as in the originals) is not counted
        rec['at_limit_noise_values'] = int(((over > 0) & (over <= 1e-6)).sum())
        drop = []
        if worst > CLAMP_TOL:
            drop.append('joint %s beyond its limit by %.3f rad' % (rec['joint_worst_mir'], worst))
        dof = np.clip(t['dof'], lo, hi)
        # contacts on the mirrored prop vs the original on its own prop
        c0 = dict(root_trans_offset=t0['pos'], root_rot=t0['quat'], dof=t0['dof'], fps=t0['fps'])
        c1 = dict(root_trans_offset=t['pos'], root_rot=t['quat'], dof=dof, fps=t['fps'])
        e0 = contact_errors(feet, c0, t0['prop'], t0['boxes'], ctx); e1 = contact_errors(feet, c1, t['prop'], t['boxes'], cm)
        a0 = np.array([e[3] for e in e0]); a1 = np.array([e[3] for e in e1])
        rec.update(contacts_orig=len(e0), contacts_mir=len(e1), over2cm_orig=int((np.abs(a0) > 0.02).sum()), over2cm_mir=int((np.abs(a1) > 0.02).sum()),
                   err_max_orig_cm=round(100 * float(np.abs(a0).max()) if len(a0) else 0, 1), err_max_mir_cm=round(100 * float(np.abs(a1).max()) if len(a1) else 0, 1))
        p0 = {(1 - e[0], e[1], e[2]): e[3] for e in e0}                # mirrored left foot = original right foot, same frames
        p1 = {(e[0], e[1], e[2]): e[3] for e in e1}
        rec['pairs_matched'] = '%d / %d' % (len(set(p0) & set(p1)), max(len(p0), len(p1)))
        rec['pair_diff_max_cm'] = round(100 * max([abs(p0[k] - p1[k]) for k in set(p0) & set(p1)] or [0]), 2)
        if rec['over2cm_mir'] > rec['over2cm_orig']:
            drop.append('%d contacts beyond 2 cm on the mirrored prop (original %d)' % (rec['over2cm_mir'], rec['over2cm_orig']))
        sp0, w0 = root_checks(c0); sp1, w1 = root_checks(c1)
        rec.update(root_speed_max_orig=round(sp0, 2), root_speed_max_mir=round(sp1, 2), root_angvel_max_orig=round(w0, 2), root_angvel_max_mir=round(w1, 2))
        if sp1 > max(2.5, sp0 + 0.01) or w1 > w0 + 0.01:
            drop.append('root jump (speed %.2f m/s, angular %.2f rad/s)' % (sp1, w1))
        rec['dropped'] = '; '.join(drop)
        report['clips'][t['id']] = rec
        if drop:
            continue
        cl, pr, me = out[key]
        dt = np.asarray(_src_clip(t0)['dof']).dtype if t0['prop'] != 'driveway' else np.float64
        cl[t['id']] = dict(root_trans_offset=t['pos'].astype(dt), root_rot=t['quat'].astype(dt), dof=dof.astype(dt),
                           fps=int(t['fps']) if float(t['fps']).is_integer() else t['fps'])
        if t['prop'] == 'driveway':
            dw = cm['driveway']; pr[t['id']] = [dict(kind='plane', a=dw['a'], b=dw['b'], c=dw['c'], note='z = a x + b y + c (world), mirrored driveway plane')]
        elif t['prop'] == 'ramp':
            pr[t['id']] = []
        else:
            pr[t['id']] = t['boxes']
        if key == 'stairs':
            mm = {a: b for a, b in smeta[t0['id']].items() if a != 'checks'}
            mm.update(mirror_of=t0['id'], checks_mirror={a: rec[a] for a in ('contacts_mir', 'over2cm_mir', 'err_max_mir_cm', 'joint_over_mir_rad', 'clamped_values', 'root_speed_max_mir')})
            me[t['id']] = mm
    # props: symmetric or not (mirrored prop vs original, metres)
    for t0, t in zip(takes, mir):
        if t['prop'] in ('ramp', 'driveway'):
            continue
        report['props'].setdefault(t['prop'], []).append(box_diff(t['boxes'], t0['boxes']))
    report['props'] = {p: dict(max_m=round(max(v), 3), median_m=round(float(np.median(v)), 3)) for p, v in report['props'].items()}
    g0 = ctx['hmap'][3]; g1 = cm['hmap'][3]
    report['props']['ramp'] = dict(max_m=round(float(np.abs(g1 - g0).max()), 3), cells_over_2cm=int((np.abs(g1 - g0) > 0.02).sum()), cells=int(g0.size))
    dw, dm = ctx['driveway'], cm['driveway']
    report['props']['driveway'] = dict(da=dm['a'] - dw['a'], db=dm['b'] - dw['b'], dz_at_centre_m=round(float(dm['a'] * dw['centre'][0] + dm['b'] * dw['centre'][1] + dm['c'] - (dw['a'] * dw['centre'][0] + dw['b'] * dw['centre'][1] + dw['c'])), 4))
    # write
    for key, name in SETS.items():
        cl, pr, me = out[key]
        joblib.dump(cl, TS / (name + '.pkl'))
        json.dump(pr, open(TS / (name + '_props.json'), 'w'))
        if me:
            json.dump(me, open(TS / (name + '_meta.json'), 'w'), indent=1)
        report['sets'][name] = dict(clips=len(cl), seconds=round(sum(len(c['dof']) / float(c['fps']) for c in cl.values()), 1))
    x0, y0, res, g = cm['hmap']
    np.savez(TS / 'tp_ramp_hmap_x30_mir.npz', x0=x0, y0=y0, res=res, grid=g.astype(np.asarray(np.load(TS / 'tp_ramp_hmap_x30.npz')['grid']).dtype))
    # the original driveway takes as a set too (no set held them before), with their plane
    import joblib as jl
    dcl = {t['id']: dict(root_trans_offset=t['pos'], root_rot=t['quat'], dof=t['dof'], fps=50) for t in takes if t['prop'] == 'driveway'}
    jl.dump(dcl, TS / 'tp_driveway_v1.pkl')
    json.dump({k: [dict(kind='plane', a=dw['a'], b=dw['b'], c=dw['c'], note='z = a x + b y + c (world), driveway plane fitted to the X2 ankles (catalogue)')] for k in dcl},
              open(TS / 'tp_driveway_v1_props.json', 'w'))
    report['sets']['tp_driveway_v1 (originals)'] = dict(clips=len(dcl))
    cat_same = sum(r['category_orig'] == r['category_mir'] for r in report['clips'].values())
    report['category_same'] = '%d / %d' % (cat_same, len(report['clips']))
    json.dump(report, open(HERE / 'mirror_report.json', 'w'), indent=1)
    by = {}
    for k, r in report['clips'].items():
        b = by.setdefault(r['prop'], dict(n=0, kept=0, clamped_clips=0, clamped_values=0, dropped=[], over2_orig=0, over2_mir=0, pairdiff=0.0))
        b['n'] += 1; b['kept'] += not r['dropped']; b['clamped_clips'] += r['clamped_values'] > 0 and not r['dropped']; b['clamped_values'] += r['clamped_values']
        b['over2_orig'] += r['over2cm_orig']; b['over2_mir'] += r['over2cm_mir']; b["pairdiff"] = max(b["pairdiff"], r["pair_diff_max_cm"]); b["unmatched"] = b.get("unmatched", 0) + (r["pairs_matched"].split(" / ")[0] != r["pairs_matched"].split(" / ")[1])
        if r['dropped']:
            b['dropped'].append('%s: %s' % (k, r['dropped']))
    for p, b in by.items():
        print(p, json.dumps(b))
    print('props', json.dumps(report['props']))
    print('sets', json.dumps(report['sets']))
    print('categories same after mirroring:', report['category_same'])
    diff = [(k, r['category_orig'], r['category_mir']) for k, r in report['clips'].items() if r['category_orig'] != r['category_mir']]
    for x in diff[:10]:
        print('  category changed', x)


def mc_jname(m, i):
    return m.joint(i + 1).name


_SRC = {}


def _src_clip(t0):
    import joblib
    f = t0['source'].split(' ')[0]
    if f not in _SRC:
        _SRC[f] = joblib.load(TS / f)
    return _SRC[f][t0['id']]


# ---------------------------------------------------------------- GIFs
def render_frames(tk, ctx, az_sign=1, n=None):
    import mujoco
    m = mc.build_model(tk, ctx); d = mujoco.MjData(m)
    W, H, FPS, MAXS = 400, 300, 12.5, 6.0
    T = len(tk['pos']); dur = T / tk['fps']; speed = max(1.0, dur / MAXS)
    idx = np.clip((np.arange(0, dur / speed, 1 / FPS) * speed * tk['fps']).astype(int), 0, T - 1)
    look = mc.smooth(tk['pos'], 1.0 * tk['fps'])
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    psi = tk['mirror']['psi'] if 'mirror' in tk else tk['frame'][1]
    cam.distance, cam.elevation, cam.azimuth = 2.9, -18, mc.D(psi) + az_sign * 125          # mirrored camera for the mirrored take
    frames = []
    with mujoco.Renderer(m, H, W) as r:
        for i in idx:
            d.qpos[:] = mc.to_qpos(tk, i); mujoco.mj_forward(m, d)
            cam.lookat[:] = [look[i, 0], look[i, 1], look[i, 2] - 0.2]
            r.update_scene(d, cam); frames.append(r.render().copy())
    return frames, speed


def strip(frames, path, W=400, H=300):
    from PIL import Image
    pick = [frames[int(len(frames) * f)] for f in (0.1, 0.35, 0.6, 0.9)]
    st = Image.new('RGB', (W * 4 if pick[0].shape[1] == W else pick[0].shape[1] * 2, H * (1 if pick[0].shape[1] == W else 2)))
    w = pick[0].shape[1]
    for k, f in enumerate(pick):
        st.paste(Image.fromarray(f), ((k % (4 if w == W else 2)) * w, (0 if w == W else (k // 2) * H)))
    st.save(path)


def run_gifs():
    perm, sign, m = build_map()
    takes, ctx = load_all(); cm = mirror_ctx(ctx)
    keep = set(); import joblib
    for name in SETS.values():
        keep |= set(joblib.load(TS / (name + '.pkl')))
    mir = [mirror_take(t, ctx, perm, sign) for t in takes]
    mc.classify(mir, cm)
    for t, t0 in zip(mir, takes):
        t['mirror']['psi'] = t0['frame'][1]; t0['mirror_psi'] = t0['frame'][1]
    cat = json.load(open(REPO / 'catalog' / 'categories.json'))['props']
    outd = REPO / 'catalog' / 'gifs_mirror'; (outd / 'strips').mkdir(parents=True, exist_ok=True)
    picks = {}
    for p in ('4x4', '2x2', 'stairs1', 'stairs2', 'ramp', 'driveway'):                    # the catalogue's first GIF take per prop
        first = next(iter(cat[p]['categories'].values()))
        tid = first['gifs'][0]['take'] if first.get('gifs') else first['takes'][0]
        picks[p] = tid
    index = {}
    for p, tid in picks.items():
        t = next(x for x in mir if x['id'] == tid + '_mir')
        if t['id'] not in keep:
            print('skip (dropped)', t['id']); continue
        fr, speed = render_frames(t, cm, az_sign=-1)
        l1 = 'MIRRORED %s / %s / %s' % (mc.PROP_NAME[p], t['category'].split(' (walk-up')[0], t['id'])
        l2 = 'mirrored take on its mirrored prop; kinematic playback' + ('' if speed < 1.02 else '  (x%.1f speed)' % speed)
        path = outd / ('%s_mir_%s.gif' % (p, mc.short(tid)))
        nb = mc.encode(fr, path, l1, l2, (400, 300))
        strip(fr, outd / 'strips' / (path.stem + '.png'))
        index[path.name] = dict(take=t['id'], prop=p, bytes=nb, frames=len(fr))
        print('gif', path.name, nb // 1000, 'kB', flush=True)
    # side by side, stairs side approach: original (catalogue camera) | mirror (mirrored camera)
    tid = cat['stairs1']['categories']['side approach then up']['takes'][0]
    t0 = next(x for x in takes if x['id'] == tid); t = next(x for x in mir if x['id'] == tid + '_mir')
    f0, sp = render_frames(t0, ctx, az_sign=1); f1, _ = render_frames(t, cm, az_sign=-1)
    n = min(len(f0), len(f1)); fr = [np.concatenate([f0[i], np.full((300, 4, 3), 255, np.uint8), f1[i]], 1) for i in range(n)]
    path = outd / ('stairs1_side_approach_orig_vs_mir_%s.gif' % mc.short(tid))
    nb = mc.encode(fr, path, 'LEFT original %s  |  RIGHT mirrored %s_mir (camera mirrored too)' % (tid, tid),
                   'Stairs place 1, side approach then up; kinematic playback', (804, 300))
    strip(fr, outd / 'strips' / (path.stem + '.png'), W=804)
    index[path.name] = dict(take=tid, bytes=nb, frames=n)
    json.dump(index, open(outd / 'index.json', 'w'), indent=1)
    print('gif', path.name, nb // 1000, 'kB')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what == 'map':
        build_map()
    if what in ('sets', 'all'):
        run_sets()
    if what in ('gifs', 'all'):
        run_gifs()
    print('DONE', what)
