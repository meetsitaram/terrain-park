"""Place real fasteners at every wood joint of the model (cad/model.py), from the geometry itself.

For each pair of touching wood pieces (boxes; cut pieces use their uncut box), find the shared face and decide
the joint the way a builder would:
  - sheet (ply/OSB/hardboard) on framing          -> 1-5/8 in screws along the framing member, 6 in o.c.
  - board face to board (one member <= 2 in thick) -> two (or three for wide faces) 3 in screws through the
                                                     thinner member into the other
  - thick to thick (4x4/6x6 to post/beam/rim)      -> 1/2 in carriage bolts through both
  - a joist end butting into a beam/rim            -> joist hanger (plus the 3 in screws are skipped)
  - stair stringer top against the landing beam    -> stair-stringer connector
Each fastener: dict(kind, pos [in], dir (drive direction, unit), length [in], group (the unit it belongs to: the
later-built piece), pieces (i, j)). `build_fasteners(m)` returns the list; `to_pieces()` turns them into model
pieces (screw heads + shanks as small cylinders) for renders and the BOM.
"""
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model as M
from params import LUMBER

GAP = 0.08
WOOD = ('lumber', 'sheet')
# build order (stage prefix) decides which piece is "later": a screw goes through the later piece into the earlier
ORDER = ['01', '02', '03', '04', '05', '06', '07', '08', '10', '09', '11', '12', '13', '14']


def obb(sh):
    _, c, ax, size = sh
    A = np.array([np.array(v, float) / np.linalg.norm(v) for v in ax])
    return np.array(c, float), A, np.array(size, float) / 2


def overlap_patch(a, b):
    """Touching boxes a, b: return (n, patch_center, patch_axes(2), patch_half(2), depth_a, depth_b) where n is the
    normal of a's face that b touches (pointing from a to b), or None if they don't share a face."""
    ca, Aa, ha = a
    cb, Ab, hb = b
    best = None
    for k in range(3):                                       # candidate face normals: a's axes
        n = Aa[k]
        d = np.dot(cb - ca, n)
        s = 1 if d >= 0 else -1
        rb = sum(hb[i] * abs(np.dot(Ab[i], n)) for i in range(3))
        gap = abs(d) - ha[k] - rb
        if abs(gap) > GAP:
            continue
        # b must overlap a's face in the other two directions
        u, v = Aa[(k + 1) % 3], Aa[(k + 2) % 3]
        ok = True
        lo, hi = [], []
        for t, ht in ((u, ha[(k + 1) % 3]), (v, ha[(k + 2) % 3])):
            pb = np.dot(cb - ca, t)
            eb = sum(hb[i] * abs(np.dot(Ab[i], t)) for i in range(3))
            l, h = max(-ht, pb - eb), min(ht, pb + eb)
            if h - l < 0.3:
                ok = False
                break
            lo.append(l)
            hi.append(h)
        if not ok:
            continue
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if best is None or area > best[-1]:
            pc = ca + n * s * ha[k] + u * (lo[0] + hi[0]) / 2 + v * (lo[1] + hi[1]) / 2
            best = (n * s, pc, (u, v), ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2), area)
    return best


def thickness_along(o, n):
    c, A, h = o
    return 2 * sum(h[i] * abs(np.dot(A[i], n)) for i in range(3))


def stage_rank(p):
    pre = p.group[:2]
    return ORDER.index(pre) if pre in ORDER else 99


def build_fasteners(m=None):
    m = m or M.build()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
    from check_support import contacts
    boxes, edges = contacts(m)
    out = []
    done = set()
    for i, js in edges.items():
        for j in js:
            if (j, i) in done or i == j:
                continue
            done.add((i, j))
            a, b = m.pieces[i], m.pieces[j]
            if a.kind not in WOOD or b.kind not in WOOD:
                continue
            if a.part or b.part:                                    # moving parts (gates) have their own hardware
                if a.part != b.part:
                    continue
            sa = next((s for s in a.shapes if s[0] == 'box'), None)
            sb = next((s for s in b.shapes if s[0] == 'box'), None)
            if sa is None or sb is None:
                continue
            # the later-built piece is the one screwed through
            later, earlier, ol, oe = (a, b, obb(sa), obb(sb))
            if (stage_rank(b), j) > (stage_rank(a), i):
                later, earlier, ol, oe = b, a, obb(sb), obb(sa)
            res = overlap_patch(oe, ol)                              # face of the earlier piece touched by the later
            if res is None:
                res2 = overlap_patch(ol, oe)
                if res2 is None:
                    continue
                n, pc, (u, v), (hu, hv), area = res2
                n = -n
            else:
                n, pc, (u, v), (hu, hv), area = res
            # n points from earlier into later; drive direction is -n (from the later piece's outer face inwards)
            t_later = thickness_along(ol, n)
            t_earlier = thickness_along(oe, n)
            grp = later.group
            sheet = later.kind == 'sheet' or earlier.kind == 'sheet'
            if sheet:
                # screws through the sheet (thin) into the framing member, along the patch's long axis, 6 in o.c.
                sh_piece = later if later.kind == 'sheet' else earlier
                long_ax, long_h, short_h = (u, hu, hv) if hu >= hv else (v, hv, hu)
                k = max(1, int(round(2 * long_h / 6.0)))
                n_drive = -n if sh_piece is later else n
                t_sh = t_later if sh_piece is later else t_earlier
                for q in range(k):
                    off = -long_h + (q + 0.5) * 2 * long_h / k
                    p = pc + long_ax * off
                    head = p + (-n_drive) * t_sh                    # head on the sheet's outer face
                    out.append(dict(kind='screw 1-5/8 in', pos=head.tolist(), dir=n_drive.tolist(), length=1.625,
                                    group=grp, pieces=(i, j)))
                continue
            # joist end into a beam/rim/post face: a joist hanger (either piece may be the joist)
            hung = False
            for jo, ot in ((later, earlier), (earlier, later)):
                o = obb(next(s_ for s_ in jo.shapes if s_[0] == 'box'))
                lk = int(np.argmax(o[2]))
                if (abs(np.dot(o[1][lk], n)) > 0.9 and o[2][lk] > 6 and min(o[2]) * 2 <= 1.6
                        and ot.kind == 'lumber' and jo.stock in ('2x8', '2x10', '2x12')
                        and not jo.name.startswith('stringer')):
                    out.append(dict(kind='joist hanger', pos=pc.tolist(), dir=(-n).tolist(), length=0, group=grp,
                                    pieces=(i, j), stock=jo.stock))
                    hung = True
                    break
            if hung:
                continue
            if later.name.startswith('stringer') and earlier.stock in ('2x8', '2x10', '2x12'):
                out.append(dict(kind='stair stringer connector', pos=pc.tolist(), dir=(-n).tolist(), length=0,
                                group=grp, pieces=(i, j), stock=later.stock))
                continue
            if max(t_later, t_earlier) > 12:                         # end grain butting a face (along the board)
                if later.stock == earlier.stock and min(t_later, t_earlier) > 12:
                    continue                                         # boards butted end to end over a post
                long_ax, long_h = (u, hu) if hu >= hv else (v, hv)
                thin_is_later = t_later <= t_earlier
                n_drive = -n if thin_is_later else n
                t_thin = min(t_later, t_earlier)
                for off in ((-long_h / 3, long_h / 3) if long_h > 1.5 else (0.0,)):
                    p = pc + long_ax * off
                    out.append(dict(kind='structural screw 6 in' if t_thin > 2 else 'screw 3 in',
                                    pos=(p + (-n_drive) * t_thin).tolist(), dir=n_drive.tolist(),
                                    length=6.0 if t_thin > 2 else 3.0, group=grp, pieces=(i, j)))
                continue
            if t_later >= 3.4 and t_earlier >= 3.4:                  # 4x4 / 6x6 / doubled: carriage bolts
                long_ax, long_h = (u, hu) if hu >= hv else (v, hv)
                for off in ((-long_h / 3, long_h / 3) if long_h > 2.5 else (0.0,)):
                    p = pc + long_ax * off
                    head = p + n * t_later
                    out.append(dict(kind='carriage bolt 1/2 in', pos=head.tolist(), dir=(-n).tolist(),
                                    length=t_later + t_earlier + 1.0, group=grp, pieces=(i, j)))
                continue
            # screws: through the thinner member
            thin_is_later = t_later <= t_earlier
            t_thin = min(t_later, t_earlier)
            n_drive = -n if thin_is_later else n
            L = 3.0 if t_thin <= 1.6 else (4.0 if t_thin <= 2.6 else 6.0)
            if L == 6.0:
                k6 = 'structural screw 6 in'
            long_ax, long_h, short_h = (u, hu, hv) if hu >= hv else (v, hv, hu)
            k = 2 if long_h < 6 else 3
            for q in range(k):
                off = -long_h * 0.6 + q * (1.2 * long_h / (k - 1))
                p = pc + long_ax * off
                head = p + (-n_drive) * t_thin
                out.append(dict(kind='screw %s in' % ('3' if L == 3.0 else '4') if L < 6 else 'structural screw 6 in',
                                pos=head.tolist(), dir=n_drive.tolist(), length=L, group=grp, pieces=(i, j)))
    return out


def to_pieces(fs):
    """Fasteners -> model pieces (for renders): screw = head disc + shank; bolt = round head + shank;
    hanger = U-strap box."""
    pcs = []
    for f in fs:
        p, d = np.array(f['pos']), np.array(f['dir'])
        g = f['group'].split('/')[0] + '/' + f['group'].split('/', 1)[1] + ' fasteners'
        if 'screw' in f['kind']:
            shapes = [('cyl', (p - d * 0.06).tolist(), (p + d * 0.02).tolist(), 0.17, 0.17),
                      ('cyl', p.tolist(), (p + d * f['length']).tolist(), 0.07, 0.05)]
            look = 'fastener'
        elif f['kind'].startswith('carriage'):
            shapes = [('cyl', (p - d * 0.25).tolist(), p.tolist(), 0.5, 0.42),
                      ('cyl', p.tolist(), (p + d * f['length']).tolist(), 0.25, 0.25)]
            look = 'fastener'
        elif f['kind'].startswith('stair'):
            ex, ey, ez = M.frame_along(p.tolist(), (p + d).tolist(), [0, 0, 1])
            shapes = [M.box_shape((p + d * 0.5).tolist(), (ex, ey, ez), (1.0, 1.75, 6.0))]
            look = 'fastener'
        else:                                                         # joist hanger
            ex, ey, ez = M.frame_along(p.tolist(), (p + d).tolist(), [0, 0, 1])
            shapes = [M.box_shape((p + d * 0.75).tolist(), (ex, ey, ez), (1.75, 6.5 if f.get('stock') == '2x8' else 8.0,
                                                                            1.6))]
            look = 'fastener'
        pcs.append(M.Piece(g, f['kind'], shapes, 'buy', None, look, buy=f['kind'], mass=0.01))
    return pcs


if __name__ == '__main__':
    import time
    t = time.time()
    fs = build_fasteners()
    c = Counter(f['kind'] for f in fs)
    print('%d fasteners in %.1f s' % (len(fs), time.time() - t))
    for k, v in sorted(c.items()):
        print('  %5d %s' % (v, k))
    by = Counter(f['group'].split('/')[0] for f in fs)
    print(sorted(by.items()))
