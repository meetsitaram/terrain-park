"""Re-retarget single Pico takes with a FRESH X2Retargeter per take (no warm start carried over from the previous take).
python reret.py RAW_DIR OUT_DIR CALIB_HHMMSS b2b|plain HHMMSS [HHMMSS ...]"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, r'C:/Users/PC/projects/loco-manip-apparatus/sim/capture')
from retarget_x2 import calibrate, retarget  # noqa: E402
from x2_recorder.pico_mocap.x2_ik import X2Retargeter  # noqa: E402

raw, out, cal, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)


def load(f):
    d = np.load(f)
    return dict(name=f.stem, t=(d['stamp_ns'] - d['stamp_ns'][0]) * 1e-9, body=d['body'], ok=d['body_ok'].astype(bool), floor=float(d['floor_y']))


calt = load(next(raw.glob('record_*_%s.npz' % cal)))
for k in sys.argv[5:]:
    tk = load(next(raw.glob('record_*_%s.npz' % k)))
    rt = X2Retargeter(); rt.human_stance = True; rt.body_to_body = (mode == 'b2b')
    calibrate(rt, [calt]); t0 = time.perf_counter()
    q = retarget(rt, tk)
    np.savez_compressed(out / (tk['name'] + '.npz'), t=tk['t'], qpos=q.astype(np.float32), height_scale=rt.height_scale, floor_y=tk['floor'])
    print(k, 'hs %.3f' % rt.height_scale, 'waist_yaw |max| %.2f hip_yaw |max| %.2f %.2f  (%.0f s)' % (
        np.abs(q[:, 19]).max(), np.abs(q[:, 9]).max(), np.abs(q[:, 15]).max(), time.perf_counter() - t0), flush=True)
