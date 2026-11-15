"""Overlay the model's wireframe on a spec picture through a fitted camera (cad/fusion/cameras.json): the quickest
way to see where our layout differs from NIST's. Writes build/overlay_<name>.png.

Run: python cad/tools/overlay.py overview [--fit]   (--fit re-runs camfit first)
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
import model as M
import params as P
import camfit

COL = {'lumber': (0, 140, 255), 'sheet': (255, 0, 200), 'buy': (0, 200, 0), 'print': (0, 200, 0)}


def box_edges(shape):
    if shape[0] != 'box':
        return []
    _, c, ax, size = shape
    c = np.array(c)
    ax = [np.array(a) * s / 2 for a, s in zip(ax, size)]
    corners = {}
    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-1, 1):
                corners[(i, j, k)] = c + i * ax[0] + j * ax[1] + k * ax[2]
    edges = []
    for a, pa in corners.items():
        for d in range(3):
            if a[d] == -1:
                b = list(a)
                b[d] = 1
                edges.append((pa, corners[tuple(b)]))
    return edges


def camera_matrices(cam):
    w, h = cam['size']
    f = cam['focal_px']
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    fwd = np.array(cam['fwd'])
    up = np.array(cam['up'])
    right = np.cross(fwd, up)
    R = np.stack([right, -up, fwd])                                    # world -> camera rows
    t = -R @ np.array(cam['eye'])
    return K, R, t


def draw(name, poses=None, groups=None):
    cams = json.load(open(os.path.join(HERE, '..', 'fusion', 'cameras.json'), encoding='utf-8'))
    cam = cams[name]
    fn = camfit.CAMS[name][0]
    img = cv2.imread(os.path.join(camfit.SPEC, fn))
    img = cv2.addWeighted(img, 0.55, np.full_like(img, 255), 0.45, 0)
    K, R, t = camera_matrices(cam)
    m = M.build(poses=poses or P.POSE)
    for p in m.pieces:
        if groups and not any(p.group.startswith(g) for g in groups):
            continue
        col = COL.get(p.kind, (128, 128, 128))
        for s in p.shapes:
            for a, b in box_edges(s):
                pa = R @ a + t
                pb = R @ b + t
                if pa[2] <= 1 or pb[2] <= 1:
                    continue
                qa = K @ pa
                qb = K @ pb
                cv2.line(img, (int(qa[0] / qa[2]), int(qa[1] / qa[2])), (int(qb[0] / qb[2]), int(qb[1] / qb[2])), col, 1,
                         cv2.LINE_AA)
    for pt, px in camfit.CAMS[name][2]:
        q = K @ (R @ np.array(pt, float) + t)
        cv2.circle(img, (int(px[0]), int(px[1])), 5, (0, 0, 255), 2)
        cv2.line(img, (int(px[0]), int(px[1])), (int(q[0] / q[2]), int(q[1] / q[2])), (0, 0, 255), 2)
    out = os.path.join(HERE, '..', '..', 'build', 'overlay_%s.png' % name)
    cv2.imwrite(out, img)
    return out


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'overview'
    if '--fit' in sys.argv:
        r = camfit.fit(camfit.CAMS[name][1], camfit.CAMS[name][2])
        cams = json.load(open(os.path.join(HERE, '..', 'fusion', 'cameras.json'), encoding='utf-8'))
        cams[name] = r
        json.dump(cams, open(os.path.join(HERE, '..', 'fusion', 'cameras.json'), 'w', encoding='utf-8'), indent=1)
    print(draw(name, poses={'door hinge': 80.0, 'start gate': 60.0, 'side gate': 60.0},
               groups=['01', '02', '03', '05', '06', '07', '08', '09', '10', '12']))
