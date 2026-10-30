"""Fit a ramp as ONE prop from the X2's planted feet: flat floor, a slope, then a level deck (boxes and slopes only).

    python fit_ramp_x2.py X2_DIR RAW_DIR FIRST LAST OUT.json

Planted = the human foot still and lower (the human recording decides), sole height = the X2 foot's lowest visual-mesh
vertex. The ramp axis is the principal direction of the planted feet. Along it, the profile h(s) = 0 before s0,
rises linearly to H at s1, stays H after (the deck), fitted by grid search on (s0, s1) with H = the median sole height
beyond s1, minimising the median absolute error. Width = planted spread across the axis + 10 cm, at least 0.6 m.
Written as one prop: kind 'ramp', origin (foot of the slope, width centre, floor), yaw, width, slope_run, rise,
deck_run, floor z.
"""
import glob, json, os, sys
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _x2 import X2_XML  # noqa: E402

x2dir, rawdir, first, last, out = sys.argv[1:6]
names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(x2dir, 'record_*.npz')) if first <= os.path.basename(f)[16:22] <= last)
m = mujoco.MjModel.from_xml_path(str(X2_XML)); d = mujoco.MjData(m)
feet = []
for side in ('left', 'right'):
    bid = m.body(f'{side}_ankle_roll_link').id
    g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 1][0]
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    feet.append((g, v[::max(1, len(v) // 150)]))
R = []
for nm in names:
    z = np.load(os.path.join(x2dir, nm)); q = z['qpos']
    raw = np.load(os.path.join(rawdir, nm)); b = raw['body'].astype(float); t = raw['stamp_ns'] * 1e-9
    dt = np.maximum(np.diff(t), 1e-3)
    for k, hk in ((0, 10), (1, 11)):
        sp = np.convolve(np.r_[0, np.linalg.norm(np.diff(b[:, hk, [0, 2]], axis=0), axis=1) / dt], np.ones(9) / 9, 'same')
        for i in np.where((sp < 0.08) & (b[:, hk, 1] <= b[:, 21 - hk, 1] + 0.01))[0]:
            d.qpos[:q.shape[1]] = q[i]; mujoco.mj_kinematics(m, d)
            g, vv = feet[k]; w = d.geom_xpos[g] + vv @ d.geom_xmat[g].reshape(3, 3).T
            R.append(np.r_[w[:, :2].mean(0), w[:, 2].min()])
R = np.array(R)
c = R[:, :2].mean(0); _, V = np.linalg.eigh(np.cov((R[:, :2] - c).T)); u = V[:, -1]
s = (R[:, :2] - c) @ u
if np.corrcoef(s, R[:, 2])[0, 1] < 0:
    u = -u; s = -s                                       # uphill = +s
n = np.array([-u[1], u[0]]); qq = (R[:, :2] - c) @ n
floor = float(np.percentile(R[:, 2], 10))
hz = R[:, 2] - floor
best = None
for s0 in np.arange(np.percentile(s, 2), np.percentile(s, 70), 0.02):
    for s1 in np.arange(s0 + 0.2, np.percentile(s, 98), 0.02):
        top = hz[s > s1]
        if len(top) < 30:
            continue
        H = float(np.median(top))
        pred = np.clip((s - s0) / (s1 - s0), 0, 1) * H
        err = np.median(np.abs(hz - pred))
        if best is None or err < best[0]:
            best = (err, s0, s1, H)
err, s0, s1, H = best
pred = np.clip((s - s0) / (s1 - s0), 0, 1) * H; res = hz - pred
w0, w1 = np.percentile(qq, [3, 97]); width = max(w1 - w0 + 0.10, 0.6)
origin = c + u * s0 + n * (w0 + w1) / 2
deck = float(np.percentile(s, 98) - s1 + 0.06)
prop = dict(kind='ramp', origin=[float(origin[0]), float(origin[1]), floor], yaw=float(np.arctan2(u[1], u[0])), width=float(width),
            slope_run=float(s1 - s0), rise=H, deck_run=deck, slope_deg=float(np.degrees(np.arctan2(H, s1 - s0))), scale='x2',
            frame='retargeted X2 world (MuJoCo, z up), metres; origin = foot of the slope, width centre, at the floor')
json.dump(dict(takes=names, props=[prop]), open(out, 'w'), indent=1)
print(f'{len(names)} takes, {len(R)} planted-foot frames')
print(f'ramp: floor {100 * floor:+.1f} cm, slope {s1 - s0:.2f} m rising {100 * H:.1f} cm ({prop["slope_deg"]:.1f} deg), deck {deck:.2f} m, width {width:.2f} m')
print(f'fit: median |error| {100 * err:.1f} cm; within 2 cm {100 * np.mean(np.abs(res) < 0.02):.0f}%, floating > 2 cm {100 * np.mean(res > 0.02):.0f}%, sunk < -2 cm {100 * np.mean(res < -0.02):.0f}%')
print('->', out)
