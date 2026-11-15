"""Driveway v1_1: fresh re-retarget (rr_plain) -> 50 Hz interp + slerp + Gaussian smoothing (sigma 0.04 s) and the 8 untracked
joints zeroed (the tp_stairs_all_v1 / build_train recipe), written as uniform-50 Hz npz for the shadow catalogue loader."""
import numpy as np, glob, os, sys
sys.argv = ['x']; import importlib.util
src = open(r'C:/Users/PC/projects/terrain-park/tools/stairs_all_v1/build.py').read().split('\nT = {}')[0]   # helpers only
exec(src)
OUT = 'shadow/terrain-park/data/2026-10-30/x2'
for f in sorted(glob.glob('rr_plain/record_20261030_121*.npz')):
    z = np.load(f); t = z['t'].astype(float); q = z['qpos'].astype(float)
    g = np.diff(t); print(os.path.basename(f)[16:22], 'stream gaps > 0.1 s:', np.round(t[1:][g > 0.1], 2), np.round(g[g > 0.1], 2), end=' ')
    qq = resample(t - t[0], q); qq[:, UIDX] = 0.0
    sp = np.linalg.norm(np.diff(qq[:, :3], axis=0), axis=1) * FPS; dv = np.abs(np.diff(qq[:, 7:], axis=0)).max() * FPS
    print('-> T %d speed max %.2f dofvel max %.1f' % (len(qq), sp.max(), dv))
    np.savez_compressed(os.path.join(OUT, os.path.basename(f)), t=np.arange(len(qq)) / FPS + 1e-7, qpos=qq, height_scale=z['height_scale'], floor_y=z['floor_y'])
