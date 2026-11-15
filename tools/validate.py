"""Validate retargeted X2 takes against their prop, without looking: per-take numbers and a sheet of the worst frames.

    python tools/validate.py X2_DIR RAW_DIR PROPS_JSON [OUT.png]

Per frame, for each X2 foot (ankle_roll_link visual mesh), the surface under it (the prop: stairs, ramp, or the floor)
and the foot's lowest point above it (clearance). The human recording says when a foot is planted (still, the lower
one). Per take:
  floating   frames where a planted foot's X2 sole is > 5 cm above its surface
  sinking    frames where any foot point is > 2 cm below its surface
  contact    planted frames with the X2 sole within 2 cm of the surface
  tilt       median angle between a planted X2 sole and the surface (deg; > 5 is visibly wrong on a slope)
  legs       frames with a knee bent backward (fails); hip / ankle joints at a limit are reported, not failed
A take passes when floating < 5%, sinking < 5%, contact >= 80%, tilt <= 5 deg, legs 0. The worst frame of each
failing take (most floating or deepest sinking) is rendered side-on next to the prop into OUT.png.
"""
import json, os, sys
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _x2 import X2_XML  # noqa: E402

x2dir, rawdir, props_path = sys.argv[1:4]
out_png = sys.argv[4] if len(sys.argv) > 4 else None
js = json.load(open(props_path)); st = js['props'][0]
yaw = st['yaw']; u = np.array([np.cos(yaw), np.sin(yaw)]); o = np.array(st['origin'][:2]); fz = st['origin'][2]


def level(xy):
    """(N,) which surface is under world points xy: 0 floor, 1.. steps (stairs), 1 slope / 2 deck (ramp)."""
    s = (xy - o) @ u; w = (xy - o) @ np.array([-u[1], u[0]]); lv = np.zeros(len(xy), int)
    if st['kind'] == 'ramp':
        lv = np.where((s > 0) & (s < st['slope_run']), 1, np.where((s >= st['slope_run']) & (s < st['slope_run'] + st['deck_run']), 2, 0))
    elif st['kind'] == 'stairs':
        a = 0.0
        for k, stp in enumerate(st['steps'], 1):
            lv = np.where(s >= a, k, lv); a += stp['run']
        lv = np.where(s > a, 0, lv)
    return np.where(np.abs(w) <= st['width'] / 2, lv, 0)


def surface(xy):
    """(N,) surface height and (N, 3) normal under world points xy (N, 2)."""
    s = (xy - o) @ u; nrm = np.tile([0.0, 0.0, 1.0], (len(xy), 1)); h = np.full(len(xy), fz)
    if st['kind'] == 'ramp':
        on = (s > 0) & (s < st['slope_run'] + st['deck_run'])
        h = fz + np.where(on, np.clip(s / st['slope_run'], 0, 1) * st['rise'], 0.0)
        sl = on & (s < st['slope_run']); th = np.arctan2(st['rise'], st['slope_run'])
        nrm[sl] = [-np.sin(th) * u[0], -np.sin(th) * u[1], np.cos(th)]
    elif st['kind'] == 'stairs':
        a = 0.0
        for stp in st['steps']:
            h = np.where(s >= a, fz + stp['top'], h); a += stp['run']
    w = (xy - o) @ np.array([-u[1], u[0]])
    return np.where(np.abs(w) <= st['width'] / 2, h, fz), nrm


m = mujoco.MjModel.from_xml_path(str(X2_XML)); d = mujoco.MjData(m)
feet = []
for side in ('left', 'right'):
    bid = m.body(f'{side}_ankle_roll_link').id
    g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1][0]
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    feet.append((g, v[::max(1, len(v) // 120)]))
lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
# the 12 sole contact spheres per foot (the X2's foot collision geometry)
soles = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f'{sd}_ankle_roll_link').id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
         for sd in ('left', 'right')]
FH = dict(on=0, edge=0, sunk=0, floating=0, other=0); FH_LEVEL = dict(match=0, wrong=0)
# sole normal in each foot geom's frame: world up with the X2 standing in its default pose (feet flat)
mujoco.mj_resetData(m, d); d.qpos[2] = 1.0; mujoco.mj_kinematics(m, d)
sole_n_local = [d.geom_xmat[g].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0]) for g, _ in feet]
# leg joints (hip, knee, ankle) whose limits count, the knee's straight-leg limit excepted
leg_j = [i for i in range(m.njnt - 1) if any(k in m.joint(i + 1).name for k in ('hip', 'ankle'))]
knees = [m.joint(f'{s}_knee_joint').qposadr[0] for s in ('left', 'right')]
rows, worst = [], []
for nm in js['takes']:
    q = np.load(os.path.join(x2dir, nm))['qpos']
    raw = np.load(os.path.join(rawdir, nm)); b = raw['body'].astype(float); t = raw['stamp_ns'] * 1e-9
    dt = np.maximum(np.diff(t), 1e-3)
    planted = []
    for hk in (10, 11):
        sp = np.convolve(np.r_[0, np.linalg.norm(np.diff(b[:, hk, [0, 2]], axis=0), axis=1) / dt], np.ones(9) / 9, 'same')
        planted.append((sp < 0.08) & (b[:, hk, 1] <= b[:, 21 - hk, 1] + 0.01))
    fl = sk = cont = n_pl = legs = lim = 0; tilts = []; frames = 0; wf = (0.0, 0)
    for i in range(0, len(q), 2):
        d.qpos[:q.shape[1]] = q[i]; mujoco.mj_kinematics(m, d); frames += 1
        sink = 0.0; floated = False
        for k, (g, v) in enumerate(feet):
            R = d.geom_xmat[g].reshape(3, 3); w = d.geom_xpos[g] + v @ R.T
            hs, nrm = surface(w[:, :2]); c = w[:, 2] - hs
            sink = min(sink, c.min())
            if planted[k][i]:
                n_pl += 1; cl = c.min(); cont += abs(cl) < 0.02
                if cl > 0.05:
                    floated = True
                sole_n = R @ sole_n_local[k]
                tilts.append(np.degrees(np.arccos(np.clip(sole_n @ nrm[np.argmin(c)], -1, 1))))
        fl += floated; sk += sink < -0.02
        j = q[i, 7:]
        legs += bool((q[i, knees] < -0.02).any())                     # a knee bent backward
        lim += bool(((j[leg_j] <= lo[leg_j] + 0.01) | (j[leg_j] >= hi[leg_j] - 0.01)).any())
        score = max(-sink, 0.0) + (0.05 if floated else 0.0)
        if score > wf[0]:
            wf = (score, i)
    r = dict(take=nm[16:22], floating=100 * fl / frames, sinking=100 * sk / frames, contact=100 * cont / max(n_pl, 1),
             tilt=float(np.median(tilts)) if tilts else 0.0, legs=legs, limits=100 * lim / frames)
    r['pass'] = r['floating'] < 5 and r['sinking'] < 5 and r['contact'] >= 80 and r['tilt'] <= 5 and legs == 0
    rows.append(r); worst.append((nm, wf[1]))
    # footholds: each planted stretch of each foot, judged at its middle frame by the 12 sole points
    for k in range(2):
        pl = planted[k]; idx = np.where(pl)[0]
        if not len(idx):
            continue
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for rn in runs:
            if len(rn) < 8:
                continue
            i = int(rn[len(rn) // 2]); d.qpos[:q.shape[1]] = q[i]; mujoco.mj_kinematics(m, d)
            pts = np.array([d.geom_xpos[g] - [0, 0, m.geom_size[g][0]] for g in soles[k]])   # sphere bottoms
            hs, _ = surface(pts[:, :2]); c = pts[:, 2] - hs
            sup, ins, flt = (np.abs(c) <= 0.015), (c < -0.015), (c > 0.015)
            hang = flt & (hs < np.max(hs[sup]) - 0.02) if sup.any() else np.zeros(len(c), bool)
            if sup.sum() >= 9: kind = 'on'
            elif ins.sum() >= 3: kind = 'sunk'
            elif sup.any() and hang.sum() >= 3: kind = 'edge'
            elif not sup.any() and not ins.any(): kind = 'floating'
            else: kind = 'other'
            FH[kind] += 1
            hl = b[i, 10 + k, 1]; fy = float(raw['floor_y'])
            r.setdefault('fh', []).append(kind)
            lv = np.bincount(level(pts[:, :2]), minlength=4).argmax()
            r.setdefault('lv', []).append(int(lv))
print(f"{'take':7s} {'floating':>9s} {'sinking':>8s} {'contact':>8s} {'tilt':>7s} {'knee bk':>8s} {'at limit':>9s}  result")
for r in rows:
    print(f"{r['take']:7s} {r['floating']:8.0f}% {r['sinking']:7.0f}% {r['contact']:7.0f}% {r['tilt']:5.1f}deg {r['legs']:8d} {r['limits']:8.0f}%  {'PASS' if r['pass'] else 'fail'}")
print(f"{sum(r['pass'] for r in rows)} / {len(rows)} takes pass")
tot = sum(FH.values())
print(f"footholds (12 sole points each, at the middle of each planted stretch): {tot}")
for k_, v_ in FH.items():
    print(f"  {k_:9s} {v_:4d}  {100 * v_ / max(tot, 1):3.0f}%")
names = {'stairs': ['floor'] + [f'step {i + 1}' for i in range(len(st.get('steps', [])))], 'ramp': ['floor', 'slope', 'deck']}.get(st['kind'], ['floor'])
allv = [v for r in rows for v in r.get('lv', [])]
print('  surface under the footholds: ' + ', '.join(f'{names[i]} {allv.count(i)}' for i in range(len(names))))
if out_png:
    from PIL import Image
    spec = mujoco.MjSpec.from_file(str(X2_XML))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.05], pos=[0, 0, fz], rgba=[0.35, 0.35, 0.4, 1])
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    qz = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    if st['kind'] == 'ramp':
        L, H = st['slope_run'], st['rise']; th = np.arctan2(H, L); c = o + u * L / 2; qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, np.array(qz), np.array([np.cos(-th / 2), 0, np.sin(-th / 2), 0]))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[np.hypot(L, H) / 2, st['width'] / 2, 0.01], pos=[c[0], c[1], fz + H / 2 - 0.01], quat=qq, rgba=[0.72, 0.55, 0.33, 1])
        c = o + u * (L + st['deck_run'] / 2)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[st['deck_run'] / 2, st['width'] / 2, H / 2], pos=[c[0], c[1], fz + H / 2], quat=qz, rgba=[0.64, 0.49, 0.33, 1])
    elif st['kind'] == 'stairs':
        total = sum(s_['run'] for s_ in st['steps']); a = 0.0
        for s_ in st['steps']:
            Lb = total - a; c = o + u * (a + Lb / 2)
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[Lb / 2, st['width'] / 2, s_['top'] / 2], pos=[c[0], c[1], fz + s_['top'] / 2], quat=qz, rgba=[0.72, 0.55, 0.33, 1])
            a += s_['run']
    mm = spec.compile(); dd = mujoco.MjData(mm); rr = mujoco.Renderer(mm, 320, 320); tiles = []
    for r, (nm, i) in zip(rows, worst):
        if r['pass']:
            continue
        q = np.load(os.path.join(x2dir, nm))['qpos']; dd.qpos[:q.shape[1]] = q[i]; mujoco.mj_forward(mm, dd)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [*q[i, :2], fz + 0.25]; cam.distance = 1.4; cam.elevation = -5; cam.azimuth = np.degrees(yaw) + 90
        rr.update_scene(dd, cam); tiles.append(rr.render().copy())
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        Image.fromarray(np.concatenate([np.concatenate(tiles[k:k + 6], 1) for k in range(0, len(tiles), 6)], 0)).save(out_png)
        print('worst frames of failing takes ->', out_png)
