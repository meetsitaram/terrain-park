"""Recover the spec apparatus' main dimensions and the camera together from one picture (least squares on
reprojection error, weak priors pulling towards our current params). Lane length is fixed by the spec text
(9 x 4 ft); everything else is free.

Run: python cad/tools/layoutfit.py
"""
import math
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import params as P

L = 432.0
# unknown dims and priors (inches, absolute heights above the venue floor)
NAMES = ['W', 'Hr', 'Xd', 'Hw', 'Xl', 'Gh', 'Lh']
PRIOR = dict(W=96.0, Hr=P.DECK_Z + P.RAIL['height'], Xd=48.0, Hw=P.DECK_Z + 97.0, Xl=336.0,
             Gh=P.DECK_Z + P.LANDING['h'] + P.LANDING['guard_h'], Lh=P.DECK_Z + P.LANDING['h'])
SIGMA = dict(W=24.0, Hr=12.0, Xd=24.0, Hw=24.0, Xl=48.0, Gh=24.0, Lh=24.0)   # prior widths (in)

# (point builder from dims d, pixel) for the p.2 overview picture
PTS = [
    (lambda d: (0, 0, d['Hr']), (90, 337)),            # start end, front post top
    (lambda d: (0, d['W'], d['Hr']), (208, 163)),      # start end, back post top
    (lambda d: (d['Xd'], d['W'], d['Hw']), (393, 50)),  # door wall, back end, top
    (lambda d: (d['Xd'], 0, d['Hw']), (357, 155)),     # door wall, front end, top
    (lambda d: (L, 0, d['Gh']), (1497, 233)),          # landing guard, front-right, top
    (lambda d: (L, d['W'], d['Gh']), (1371, 72)),      # landing guard, back-right, top
    (lambda d: (d['Xl'], d['W'], d['Gh']), (1237, 72)),  # landing guard, back-left, top
    (lambda d: (d['Xl'], 0, d['Gh']), (1335, 234)),    # landing guard, front-left, top
    (lambda d: (L, 0, d['Lh']), (1452, 320)),          # landing front-right at landing deck height
    # front rail post pairs at the 4 ft section boundaries (tops)
    (lambda d: (48, 0, d['Hr']), (233, 335)),
    (lambda d: (96, 0, d['Hr']), (388, 335)),
    (lambda d: (144, 0, d['Hr']), (536, 335)),
    (lambda d: (192, 0, d['Hr']), (690, 335)),
    (lambda d: (240, 0, d['Hr']), (835, 335)),
]
SIZE = (1500, 449)


def unpack(x):
    d = dict(zip(NAMES, x[:len(NAMES)]))
    rv, tv, lf = x[len(NAMES):len(NAMES) + 3], x[len(NAMES) + 3:len(NAMES) + 6], x[-1]
    return d, rv, tv, math.exp(lf)


def resid(x):
    d, rv, tv, f = unpack(x)
    w, h = SIZE
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    obj = np.array([fn(d) for fn, _ in PTS], float)
    img = np.array([q for _, q in PTS], float)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
    r = (proj.reshape(-1, 2) - img).ravel()
    pri = [(d[k] - PRIOR[k]) / SIGMA[k] * 3.0 for k in NAMES]        # ~3 px per prior sigma
    return np.concatenate([r, pri])


def main():
    # start from the plain PnP fit with prior dims
    import camfit
    d0 = PRIOR
    pairs = [(fn(d0), q) for fn, q in PTS]
    c = camfit.fit(SIZE, pairs)
    R = np.stack([np.cross(c['fwd'], c['up']), -np.array(c['up']), c['fwd']])
    rv, _ = cv2.Rodrigues(R)
    tv = -R @ np.array(c['eye'])
    x0 = np.concatenate([[d0[k] for k in NAMES], rv.ravel(), tv.ravel(), [math.log(c['focal_px'])]])
    res = least_squares(resid, x0, method='lm', max_nfev=20000)
    d, rv, tv, f = unpack(res.x)
    r = resid(res.x)[:2 * len(PTS)].reshape(-1, 2)
    print('reprojection px:', [round(float(np.linalg.norm(e)), 1) for e in r])
    for k in NAMES:
        print('  %-3s %7.1f in  (prior %.1f)' % (k, d[k], PRIOR[k]))
    Rm, _ = cv2.Rodrigues(rv)
    eye = (-Rm.T @ tv).ravel()
    print('eye', eye.round(1), 'fwd', (Rm.T @ [0, 0, 1]).round(3), 'f', round(f), 'vfov %.1f' % (2 * math.degrees(math.atan(SIZE[1] / 2 / f))))
    return d, eye, Rm, f


if __name__ == '__main__':
    main()
