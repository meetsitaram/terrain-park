"""Interference check on the pure-Python model (inches).

Broad phase: piece AABBs. Narrow phase: sample a 16^3 grid over the two pieces' overlapping AABB and count points
inside both solids (union of primitives minus their box cuts, exact for boxes, cylinders/cones, spheres).
Reports overlaps above VOL in^3. Run: python cad/tools/check_clash.py [--all]
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model as M

VOL = 0.05          # in^3: below this, it's a touching face / rounding
N = 16              # samples per axis over the overlap box

# intended overlaps (regex on 'group / name' of either piece) and why
EXPECTED = [
    (r'switch body', 'push-button bodies pass through a drilled hole into the box / panel'),
    (r'knee brace', '45 deg end cuts modelled square: the corner tips touch the post'),
    (r'lampholder socket.*bulb|bulb.*lampholder socket', 'bulb thread screwed into the socket'),
    (r'spring hinge', 'hinge leaves sit in mortises in the jamb and the door edge'),
    (r'target \d ring.*centre|centre.*target \d ring', 'sticker layers'),
    (r'display window', 'window inside the bezel'),
    (r'latch bolt', 'latch bolt in the strike mortise'),
    (r'post base', 'post base flanges are thin steel under the post / beside the skirt'),
    (r'pipe strap', 'strap wraps the nipple'),
    (r'valve body.*mount block|mount block.*valve body', 'valve body sits in a notch in the block (nipples rest on it)'),
    (r'rose .*lever|lever.*rose ', 'lever neck through the rose'),
    (r'brace cleat', 'brace foot bears on the end of its cleat'),
    (r'gate hinges', 'hinge leaves screwed to the post and the stile'),
    (r'post kicker', 'kicker end cut against the post'),
    (r'upper stand / brace.*countertop|countertop.*upper stand / brace', 'brace foot cut flat on the countertop'),
    (r'stairs/handrails / back mid rail.*landing edge guard|landing edge guard.*back mid rail', 'rail end cut to the guard'),
]


def expected(a, b):
    k = a + ' || ' + b
    for rx, why in EXPECTED:
        if re.search(rx, k, re.I):
            return why
    return None


def inside(shape, p):
    k = shape[0]
    if k == 'box':
        _, c, ax, size = shape
        d = p - np.asarray(c)
        ok = np.ones(len(p), bool)
        for a, s in zip(ax, size):
            ok &= np.abs(d @ np.asarray(M.unit(a))) <= s / 2 - 1e-6
        return ok
    if k == 'cyl':
        _, a, b, r0, r1 = shape
        a, b = np.asarray(a), np.asarray(b)
        L = np.linalg.norm(b - a)
        u = (b - a) / L
        t = (p - a) @ u
        rad = np.linalg.norm((p - a) - np.outer(t, u), axis=1)
        r = r0 + (r1 - r0) * np.clip(t / L, 0, 1)
        return (t > 1e-6) & (t < L - 1e-6) & (rad < r - 1e-6)
    _, c, r = shape
    return np.linalg.norm(p - np.asarray(c), axis=1) < r - 1e-6


def solid(piece, p):
    ok = np.zeros(len(p), bool)
    for s in piece.shapes:
        ok |= inside(s, p)
    for s in piece.cuts:
        ok &= ~inside(s, p)
    return ok


def shape_aabb(s):
    k = s[0]
    if k == 'box':
        _, c, ax, size = s
        e = [sum(abs(M.unit(ax[j])[i]) * size[j] / 2 for j in range(3)) for i in range(3)]
        return [c[i] - e[i] for i in range(3)], [c[i] + e[i] for i in range(3)]
    if k == 'cyl':
        _, a, b, r0, r1 = s
        r = max(r0, r1)
        return [min(a[i], b[i]) - r for i in range(3)], [max(a[i], b[i]) + r for i in range(3)]
    _, c, r = s
    return [c[i] - r for i in range(3)], [c[i] + r for i in range(3)]


def piece_aabb(p):
    bs = [shape_aabb(s) for s in p.shapes]
    return [min(b[0][i] for b in bs) for i in range(3)], [max(b[1][i] for b in bs) for i in range(3)]


def run(m=None, skip=None):
    m = m or M.build()
    boxes = [piece_aabb(p) for p in m.pieces]
    order = sorted(range(len(m.pieces)), key=lambda i: boxes[i][0][0])
    rows = []
    for ii, i in enumerate(order):
        lo_i, hi_i = boxes[i]
        for j in order[ii + 1:]:
            lo_j, hi_j = boxes[j]
            if lo_j[0] >= hi_i[0]:
                break
            lo = [max(lo_i[k], lo_j[k]) for k in range(3)]
            hi = [min(hi_i[k], hi_j[k]) for k in range(3)]
            if any(hi[k] - lo[k] <= 0.02 for k in range(3)):
                continue
            pi, pj = m.pieces[i], m.pieces[j]
            if skip and skip(pi, pj):
                continue
            g = [np.linspace(lo[k], hi[k], N + 2)[1:-1] for k in range(3)]
            pts = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
            both = solid(pi, pts) & solid(pj, pts)
            if not both.any():
                continue
            cell = np.prod([(hi[k] - lo[k]) / N for k in range(3)])
            vol = both.sum() * cell
            if vol > VOL:
                q = pts[both]
                span = q.max(0) - q.min(0)
                rows.append((vol, '%s / %s' % (pi.group, pi.name), '%s / %s' % (pj.group, pj.name),
                             [round(float(s), 2) for s in span], [round(float(c), 1) for c in q.mean(0)]))
    rows.sort(key=lambda r: -r[0])
    return rows


if __name__ == '__main__':
    rows = run()
    bad = [r for r in rows if not expected(r[1], r[2])]
    print('%d overlaps > %.2f in^3: %d expected (see EXPECTED), %d to look at' % (len(rows), VOL, len(rows) - len(bad),
                                                                                  len(bad)))
    for vol, a, b, span, c in (rows if '--all' in sys.argv else bad):
        print('%7.2f in3  %s  <->  %s   span %s at %s' % (vol, a, b, span, c))
