"""Support check: every piece must be held by something. Builds a contact graph (pieces whose solids touch,
tested by inflating each piece by GAP and sampling the overlap) and walks it from the floor (pieces with
z_min ~ 0). Anything not reached is floating; anything reached only through a single thin contact is listed
as a 'weak' hold for review.

Moving parts are expected to touch their parent (hinge, slide), so they count as supported if they touch.
Run: python cad/tools/check_support.py
"""
import os
import sys
from collections import defaultdict, deque


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model as M
from check_clash import piece_aabb

GAP = 0.08            # in: faces closer than this are 'touching' (screwed / resting)


def obb(shape):
    k = shape[0]
    if k == 'box':
        _, c, ax, size = shape
        return c, [M.unit(v) for v in ax], [x / 2 for x in size]
    if k == 'cyl':
        _, a, b, r0, r1 = shape
        ex, ey, ez = M.frame_along(a, b, [0.3, 0.5, 0.8])
        r = max(r0, r1)
        return M.lerp(a, b, 0.5), [ex, ey, ez], [r, r, M.norm(M.sub(b, a)) / 2]   # bounding box: lenient
    _, c, r = shape
    return c, [M.X, M.Y, M.Z], [r] * 3


def separation(o1, o2):
    """Largest gap along the 15 SAT axes (<= 0: overlapping / touching)."""
    c1, a1, h1 = o1
    c2, a2, h2 = o2
    d = M.sub(c2, c1)
    axes = list(a1) + list(a2)
    for u in a1:
        for v in a2:
            w = M.cross(u, v)
            n = M.norm(w)
            if n > 1e-6:
                axes.append(M.mul(w, 1 / n))
    gap = -1e9
    for L in axes:
        r1 = sum(h1[i] * abs(M.dot(a1[i], L)) for i in range(3))
        r2 = sum(h2[i] * abs(M.dot(a2[i], L)) for i in range(3))
        gap = max(gap, abs(M.dot(d, L)) - r1 - r2)
    return gap


def contacts(m):
    """Pieces touch if any of their primitives are within GAP (exact SAT on boxes; cylinders and spheres as their
    bounding boxes (lenient)). Box cuts are ignored here (check_clash covers overlaps)."""
    boxes = [piece_aabb(p) for p in m.pieces]
    obbs = [[obb(s) for s in p.shapes] for p in m.pieces]
    order = sorted(range(len(m.pieces)), key=lambda i: boxes[i][0][0])
    edges = defaultdict(set)
    for ii, i in enumerate(order):
        lo_i, hi_i = boxes[i]
        for j in order[ii + 1:]:
            lo_j, hi_j = boxes[j]
            if lo_j[0] > hi_i[0] + GAP:
                break
            if any(lo_j[k] > hi_i[k] + GAP or lo_i[k] > hi_j[k] + GAP for k in (1, 2)):
                continue
            if any(separation(a, b) <= GAP for a in obbs[i] for b in obbs[j]):
                edges[i].add(j)
                edges[j].add(i)
    return boxes, edges


def run(m=None):
    m = m or M.build()
    boxes, edges = contacts(m)
    ground = [i for i, b in enumerate(boxes) if b[0][2] < 0.05]
    seen = set(ground)
    q = deque(ground)
    while q:
        i = q.popleft()
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                q.append(j)
    floating = [i for i in range(len(m.pieces)) if i not in seen]
    single = [i for i in seen if len(edges[i]) == 1 and i not in ground]
    return m, floating, single, edges


if __name__ == '__main__':
    m, floating, single, edges = run()
    print('%d pieces, %d floating, %d held by a single contact' % (len(m.pieces), len(floating), len(single)))
    for i in floating:
        p = m.pieces[i]
        print('  FLOATING  %s / %s   touches: %s' % (p.group, p.name,
                                                     [m.pieces[j].name for j in edges[i]][:4] or 'nothing'))
    if '--single' in sys.argv:
        for i in single:
            p = m.pieces[i]
            j = next(iter(edges[i]))
            print('  single    %s / %s  <- %s / %s' % (p.group, p.name, m.pieces[j].group, m.pieces[j].name))


def end_support(m, edges):
    """Boards longer than 20 in must touch something at BOTH ends (a rail held at one end is a cantilever: the
    user caught guard rails ending in mid-air). End zone: the last 3 in of the board's length."""
    import numpy as np
    bad = []
    for i, p in enumerate(m.pieces):
        if p.kind != 'lumber' or p.part or not p.cut or p.cut.get('length', 0) < 20:
            continue
        sh = p.shapes[0]
        if sh[0] != 'box':
            continue
        _, c, ax, size = sh
        k = int(np.argmax(size))                                      # long axis
        L = size[k]
        for sgn in (-1, 1):
            cz = [c[j] + sgn * (L / 2 - 1.5) * ax[k][j] for j in range(3)]
            sz = list(size)
            sz[k] = 3.0
            zone = obb(('box', cz, ax, tuple(sz)))
            held = any(separation(zone, obb(s)) <= GAP for j in edges[i] for s in m.pieces[j].shapes)
            upright_top = abs(ax[k][2]) > 0.9 and sgn * ax[k][2] > 0   # a post's top may be free
            on_floor = cz[2] - sum(abs(ax[j][2]) * sz[j] / 2 for j in range(3)) < 0.1
            if not held and not on_floor and not upright_top:         # ends on the floor are held by the floor
                bad.append((p.group, p.name, 'end at %s' % [round(v, 1) for v in cz]))
    return bad


if __name__ == '__main__':
    loose = end_support(m, edges)
    print('%d loose board ends' % len(loose))
    for g, n, w in loose:
        print('  LOOSE END %s / %s  %s' % (g, n, w))
