"""Fit a pinhole camera to a spec picture from 2D-3D point pairs (model inches), so the Fusion render can use the
same viewpoint. Searches the focal length; solvePnP for the pose. Prints eye, target, up and the vertical field of
view, and the reprojection error per point.

Run: python cad/tools/camfit.py   (cameras in CAMS below; writes cad/fusion/cameras.json)
"""
import json
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
SPEC = os.path.join(HERE, '..', '..', 'build', 'spec')

# name: (image file, image size, [(model point in, pixel)])
CAMS = {
    'overview': ('img_p2_x29.png', (1500, 449), [
        ((2, 97.75, 40.25), (210, 165)),          # back post at the start end, top
        ((2, -1.75, 40.25), (88, 336)),           # front post at the start end, top
        ((48, 96, 101), (415, 50)),               # door wall, back end, top
        ((48, 0, 101), (357, 155)),               # door wall, front end, top
        ((430, -1.75, 88.25), (1495, 231)),       # landing guard, front-right corner, top
        ((338, -1.75, 88.25), (1335, 234)),       # landing guard, front-left post, top
        ((430, 97.75, 88.25), (1370, 73)),        # landing guard, back-right corner, top
        ((338, 97.75, 88.25), (1232, 73)),        # landing guard, back-left post, top
        ((430, -1.75, 46.25), (1452, 320)),       # front-right corner at landing deck height
    ]),
}


def fit(img_size, pairs):
    obj = np.array([p for p, _ in pairs], np.float64)
    img = np.array([q for _, q in pairs], np.float64)
    w, h = img_size
    best = None
    for f in np.geomspace(300, 20000, 400):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        if best is None or err.mean() < best[0]:
            best = (err.mean(), f, rv, tv, err)
    e, f, rv, tv, err = best
    R, _ = cv2.Rodrigues(rv)
    eye = (-R.T @ tv).ravel()
    fwd = R.T @ np.array([0, 0, 1.0])                                # camera looks along +z
    up = -(R.T @ np.array([0, 1.0, 0]))                              # image y points down
    vfov = 2 * math.degrees(math.atan(h / 2 / f))
    return dict(eye=eye.tolist(), fwd=fwd.tolist(), up=up.tolist(), focal_px=f, vfov_deg=vfov, err_px=err.tolist(),
                size=img_size)


if __name__ == '__main__':
    out = {}
    for name, (fn, size, pairs) in CAMS.items():
        r = fit(size, pairs)
        out[name] = r
        print(name, 'eye', [round(v, 1) for v in r['eye']], 'fwd', [round(v, 3) for v in r['fwd']],
              'vfov %.1f deg' % r['vfov_deg'], 'err px', [round(v, 1) for v in r['err_px']])
    with open(os.path.join(HERE, '..', 'fusion', 'cameras.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
