"""Play X2 takes on a fitted prop (fit_stairs_x2.py stairs, fit_ramp_x2.py ramp), MuJoCo viewer.

    python tools/view_props.py data/2026-10-30/x2_body_to_body data/2026-10-30/props/ramp.json

Stairs are drawn as one stacked box per step, a ramp as a slope and a level deck. Keys: N next, P previous, space pause."""
import json, os, sys, time
import numpy as np, mujoco, mujoco.viewer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _x2 import KEY_N, KEY_P, KEY_SPACE, X2_XML  # noqa: E402

x2dir, js = sys.argv[1], json.load(open(sys.argv[2]))
st = js['props'][0]
clips = [(n, np.load(os.path.join(x2dir, n))) for n in js['takes']]
spec = mujoco.MjSpec.from_file(str(X2_XML))
fz = st['origin'][2]
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.05], pos=[0, 0, fz], rgba=[0.35, 0.35, 0.4, 1])
spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
yaw = st['yaw']; u = np.array([np.cos(yaw), np.sin(yaw)]); o = np.array(st['origin'][:2])
if st['kind'] == 'ramp':                                  # a slope (thin tilted box, top face = the slope) and a level deck
    L, H = st['slope_run'], st['rise']; th = np.arctan2(H, L); T = 0.02; Ls = np.hypot(L, H)
    c = o + u * L / 2; zc = fz + H / 2 - T / 2 * np.cos(th)
    qy = np.array([np.cos(-th / 2), 0, np.sin(-th / 2), 0]); qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    qq = np.zeros(4); mujoco.mju_mulQuat(qq, qz, qy)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[Ls / 2, st['width'] / 2, T / 2], pos=[c[0], c[1], zc], quat=qq,
                            rgba=[0.72, 0.55, 0.33, 1], contype=0, conaffinity=0)
    c = o + u * (L + st['deck_run'] / 2)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[st['deck_run'] / 2, st['width'] / 2, H / 2], pos=[c[0], c[1], fz + H / 2],
                            quat=qz, rgba=[0.64, 0.49, 0.33, 1], contype=0, conaffinity=0)
    st['steps'] = []
total = sum(s['run'] for s in st['steps']); a = 0.0
for j, s in enumerate(st['steps']):
    L = total - a; c = o + u * (a + L / 2)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, st['width'] / 2, s['top'] / 2], pos=[c[0], c[1], fz + s['top'] / 2],
                            quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], rgba=[0.72 - 0.08 * j, 0.55 - 0.06 * j, 0.33, 1],
                            contype=0, conaffinity=0)
    a += s['run']
m = spec.compile(); d = mujoco.MjData(m)
keys = []; i, tp, paused, last = 0, 0.0, False, time.perf_counter()
print(len(clips), 'takes; N next, P previous, space pause', flush=True)
with mujoco.viewer.launch_passive(m, d, key_callback=keys.append) as viewer:
    viewer.cam.distance, viewer.cam.elevation = 2.4, -12
    viewer.cam.azimuth = np.degrees(yaw) + 90
    while viewer.is_running():
        now = time.perf_counter(); tp += 0.0 if paused else now - last; last = now
        while keys:
            k = keys.pop(0)
            if k in (KEY_N, KEY_P): i, tp = (i + (1 if k == KEY_N else -1)) % len(clips), 0.0
            elif k == KEY_SPACE: paused = not paused
        name, z = clips[i]; t, q = z['t'], z['qpos']
        if tp > t[-1] + 0.5: i, tp = (i + 1) % len(clips), 0.0; name, z = clips[i]; t, q = z['t'], z['qpos']
        j = min(np.searchsorted(t, tp), len(t) - 1)
        with viewer.lock():
            d.qpos[:q.shape[1]] = q[j]; mujoco.mj_forward(m, d)
            viewer.cam.lookat[:] = [*q[j, :2], fz + 0.4]
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT, 'take\ntime\nkeys',
                              f'{i + 1}/{len(clips)} {name[7:-4]}\n{tp:4.1f} / {t[-1]:.1f} s\nN next  P prev  space pause'))
        viewer.sync(); time.sleep(0.01)
