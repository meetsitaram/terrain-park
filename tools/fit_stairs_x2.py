"""Fit a staircase as ONE prop from the X2's planted feet, the step of each foot decided by the human recording.

    python fit_stairs_x2.py X2_DIR RAW_DIR FIRST LAST OUT.json

Human side (clean): the planted lower foot's sole height (Foot joint - 3 cm above the take's floor) clusters into levels
(gaps > 6 cm): floor, step 1, step 2, ... Each X2 frame (the retarget is one X2 frame per human sample) takes its
human foot's level. X2 side (geometry): per level, the X2 foot's sole height (lowest visual-mesh vertex) and footprint.
The staircase: one climb direction (level centroids), each step's top = its median X2 sole height minus the floor's,
its front edge where its feet start (3rd percentile along the climb), the last step running to its feet's far end;
width = the widest level's plant spread + 10 cm, at least 0.8 m. Written as one prop: kind 'stairs', origin (front
edge of step 1, centre of the width, floor height), yaw, width, steps [{rise, run}] (rise relative to the step below).
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
    feet.append((bid, g, v[::max(1, len(v) // 150)]))
H, R = [], []                                            # human sole height; X2 (cx, cy, sole z, along-span pts)
for nm in names:
    z = np.load(os.path.join(x2dir, nm)); q = z['qpos']
    raw = np.load(os.path.join(rawdir, nm)); b = raw['body'].astype(float); fy = float(raw['floor_y']); t = raw['stamp_ns'] * 1e-9
    assert len(b) == len(q), (nm, len(b), len(q))
    dt = np.maximum(np.diff(t), 1e-3)
    for k, hk in ((0, 10), (1, 11)):
        o = 21 - hk
        sp = np.convolve(np.r_[0, np.linalg.norm(np.diff(b[:, hk, [0, 2]], axis=0), axis=1) / dt], np.ones(9) / 9, 'same')
        for i in np.where((sp < 0.08) & (b[:, hk, 1] <= b[:, o, 1] + 0.01))[0]:
            d.qpos[:q.shape[1]] = q[i]; mujoco.mj_kinematics(m, d)
            _, g, vv = feet[k]; w = d.geom_xpos[g] + vv @ d.geom_xmat[g].reshape(3, 3).T
            H.append(b[i, hk, 1] - fy - 0.03); R.append(np.r_[w[:, :2].mean(0), w[:, 2].min()]); R[-1] = (R[-1], w[:, :2])
hs = np.array(H); order = np.sort(hs); cuts = np.where(np.diff(order) > 0.06)[0]
edges = np.r_[order[0] - 1, (order[cuts] + order[cuts + 1]) / 2, order[-1] + 1]
lev = np.digitize(hs, edges[1:-1])
keep = [l for l in range(len(edges) - 1) if (lev == l).sum() > 30]
print('human levels: ' + ', '.join(f'{100 * np.median(hs[lev == l]):.0f} cm (n {(lev == l).sum()})' for l in keep))
C = np.array([r[0] for r in R]); FP = [r[1] for r in R]
floor = float(np.median(C[lev == keep[0], 2]))
cent = [C[lev == l, :2].mean(0) for l in keep]
up = cent[-1] - cent[1] if len(keep) > 2 else cent[1] - cent[0]; up /= np.linalg.norm(up); n = np.array([-up[1], up[0]])
yaw = float(np.arctan2(up[1], up[0]))
starts, ends, tops, widths, nn = [], [], [], [], []
for l in keep[1:]:
    pts = np.concatenate([FP[i] for i in np.where(lev == l)[0]])
    a = pts @ up; qq = pts @ n
    starts.append(np.percentile(a, 3)); ends.append(np.percentile(a, 97))
    tops.append(float(np.median(C[lev == l, 2]) - floor)); widths.append((np.percentile(qq, 3), np.percentile(qq, 97)))
    nn.append(int((lev == l).sum()))
steps, prev = [], 0.0
for j in range(len(tops)):
    run = (starts[j + 1] - starts[j]) if j + 1 < len(tops) else (ends[j] - starts[j])
    steps.append(dict(rise=round(tops[j] - prev, 4), run=round(float(run), 4), top=round(tops[j], 4), plants=nn[j])); prev = tops[j]
q0 = min(w[0] for w in widths); q1 = max(w[1] for w in widths); width = max(q1 - q0 + 0.10, 0.8)
origin = up * starts[0] + n * (q0 + q1) / 2
prop = dict(kind='stairs', origin=[float(origin[0]), float(origin[1]), floor], yaw=yaw, width=float(width), steps=steps,
            scale='x2', frame='retargeted X2 world (MuJoCo, z up), metres; origin = front edge of step 1, centre of the width, at the floor')
json.dump(dict(takes=names, props=[prop]), open(out, 'w'), indent=1)
print(f'stairs: floor {100 * floor:+.1f} cm, heading {np.degrees(yaw):.0f} deg, width {width:.2f} m')
for j, s in enumerate(steps):
    print(f'  step {j + 1}: rise {100 * s["rise"]:.1f} cm (top {100 * s["top"]:.1f} cm), run {s["run"]:.2f} m, {s["plants"]} plant frames')
print('->', out)
