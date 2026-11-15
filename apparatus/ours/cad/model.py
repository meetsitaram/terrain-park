"""The apparatus as a flat list of pieces, in pure Python (no Fusion): the one model that the Fusion build
(cad/fusion/apparatus.py), the cut list / BOM (cad/tools/bom.py) and the simulator export (sim/) all read.

Every piece is a solid made of primitives (oriented boxes, cylinders/cones, spheres) with optional box cuts, so
Fusion can build it as BRep with the TemporaryBRepManager (no sketches, no timeline per board) and MuJoCo can
use the same primitives as collision geoms.

Units: INCHES inside this file (what the stores sell); Piece.mm() converts. Frame: see params.py.
"""
import math

import params as P
from params import LUMBER, SHEET, DECK_Z

# ---------------------------------------------------------------- vectors (inches)

def add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def norm(a): return math.sqrt(dot(a, a))
def unit(a): return mul(a, 1.0 / norm(a))
def lerp(a, b, t): return add(a, mul(sub(b, a), t))


def rotz(v, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def rot_axis(v, k, deg):
    """Rodrigues: rotate v about unit axis k."""
    k = unit(k)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return add(add(mul(v, c), mul(cross(k, v), s)), mul(k, dot(k, v) * (1 - c)))


X, Y, Z = [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]
I3 = (X, Y, Z)

# ---------------------------------------------------------------- shapes
# box:    ('box', centre, (ex, ey, ez), (sx, sy, sz))         right-handed unit axes, full sizes
# cyl:    ('cyl', p0, p1, r0, r1)                              cylinder (r0 == r1) or cone frustum
# sphere: ('sphere', centre, r)


def box_shape(c, axes, size):
    return ('box', list(c), tuple(list(a) for a in axes), tuple(size))


def ext(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box from extents."""
    return box_shape([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], I3,
                     (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)))


def frame_along(a, b, up):
    """(ex, ey, ez): ez along a->b, ey = `up` made square to it, ex = ey x ez."""
    ez = unit(sub(b, a))
    u = sub(up, mul(ez, dot(up, ez)))
    if norm(u) < 1e-9:
        u = sub(X, mul(ez, dot(X, ez)))
    ey = unit(u)
    ex = cross(ey, ez)
    return ex, ey, ez


def frame_normal(n, up=Z):
    """(eu, ev, n): a plate frame with normal n, ev as close to `up` as possible (eu horizontal-ish)."""
    n = unit(n)
    v = sub(up, mul(n, dot(up, n)))
    if norm(v) < 1e-9:
        v = sub(Y, mul(n, dot(Y, n)))
    ev = unit(v)
    eu = cross(ev, n)
    return eu, ev, n


class Piece:
    __slots__ = ('group', 'name', 'shapes', 'cuts', 'kind', 'stock', 'look', 'mat', 'cut', 'buy', 'mass', 'part')

    def __init__(self, group, name, shapes, kind, stock=None, look='pine', mat=None, cuts=(), cut=None, buy=None,
                 mass=None, part=None):
        self.group = group            # Fusion component path 'Stage/Unit', e.g. '01 Deck/S3 deck panel'
        self.name = name              # body name
        self.shapes = list(shapes)    # unioned primitives
        self.cuts = list(cuts)        # boxes subtracted from the union
        self.kind = kind              # 'lumber' | 'sheet' | 'buy' | 'print' | 'other'
        self.stock = stock            # '2x4', 'ply', 'osb', ...  (lumber / sheet)
        self.look = look              # appearance key (fusion/apparatus.py LOOKS)
        self.mat = mat                # physical material key (density for mass and sim)
        self.cut = cut                # cut-list info, e.g. dict(length=45.0, ends='square') or sheet dims
        self.buy = buy                # BOM key for bought items
        self.mass = mass              # kg, for bought items whose mass is not geometry x density
        self.part = part              # moving-part tag for the sim / posing, e.g. 'door leaf'


def rot_about(pt, anchor, axis, deg):
    return add(anchor, rot_axis(sub(pt, anchor), axis, deg))


def rotate_shape(sh, anchor, axis, deg):
    k = sh[0]
    if k == 'box':
        _, c, ax, size = sh
        return ('box', rot_about(c, anchor, axis, deg), tuple(rot_axis(a, axis, deg) for a in ax), size)
    if k == 'cyl':
        _, a, b, r0, r1 = sh
        return ('cyl', rot_about(a, anchor, axis, deg), rot_about(b, anchor, axis, deg), r0, r1)
    _, c, r = sh
    return ('sphere', rot_about(c, anchor, axis, deg), r)


def translate_shape(sh, d):
    k = sh[0]
    if k == 'box':
        return ('box', add(sh[1], d), sh[2], sh[3])
    if k == 'cyl':
        return ('cyl', add(sh[1], d), add(sh[2], d), sh[3], sh[4])
    return ('sphere', add(sh[1], d), sh[2])


class Model:
    def __init__(self):
        self.pieces = []
        self.joints = []              # dicts: name, type, part, parent, anchor, axis, range, spring...
        self.notes = {}               # group -> list of build notes (fasteners, order) for the guide
        self.fasteners = []           # dicts: group, kind, count, where

    def add(self, p):
        self.pieces.append(p)
        return p

    # ---- lumber and sheets
    def lumber(self, group, name, stock, a, b, up=Z, flat=False, look='pine', ends='square', part=None, cuts=()):
        """A board between end centres a and b. On edge (default): the wide face stands along `up`.
        flat=True: lies flat (thin along `up`)."""
        thin, wide = LUMBER[stock]
        ex, ey, ez = frame_along(a, b, up)
        sx, sy = (wide, thin) if flat else (thin, wide)
        L = norm(sub(b, a))
        c = lerp(a, b, 0.5)
        return self.add(Piece(group, name, [box_shape(c, (ex, ey, ez), (sx, sy, L))], 'lumber', stock, look,
                              mat='softwood', cut=dict(length=L, ends=ends), part=part, cuts=cuts))

    def post(self, group, name, x, y, z0, z1, stock='4x4', look='pine', part=None):
        return self.lumber(group, name, stock, [x, y, z0], [x, y, z1], up=Y, part=part)

    def sheet(self, group, name, stock, shape, cuts=(), look=None, part=None, dims=None):
        t = {'ply': SHEET['ply'], 'osb': SHEET['osb'], 'hard': SHEET['hard']}[stock]
        sz = sorted(shape[3])
        return self.add(Piece(group, name, [shape], 'sheet', stock, look or stock,
                              mat={'ply': 'plywood', 'osb': 'osb', 'hard': 'hardboard'}[stock],
                              cuts=cuts, cut=dict(dims=dims or (sz[2], sz[1]), t=t), part=part))

    def item(self, group, name, shapes, buy, look, mass=None, part=None, mat=None, cuts=()):
        return self.add(Piece(group, name, shapes, 'buy', None, look, mat=mat, buy=buy, mass=mass, part=part,
                              cuts=cuts))

    def printed(self, group, name, shapes, look='petg', part=None, buy=None):
        return self.add(Piece(group, name, shapes, 'print', None, look, mat='petg', part=part, buy=buy))

    def note(self, group, text):
        self.notes.setdefault(group, []).append(text)

    def screws(self, group, kind, count, where):
        self.fasteners.append(dict(group=group, kind=kind, count=count, where=where))

    def pose(self, poses):
        """Rotate hinged parts (and their child parts) by poses = {joint name: degrees} (renders / pictures)."""
        for j in self.joints:
            v = poses.get(j['name'])
            if not v or j['type'] not in ('hinge', 'slide'):
                continue
            parts = {j['part']} | {k['part'] for k in self.joints if k.get('parent') == j['part']}
            if j['type'] == 'hinge':
                f = lambda sh: rotate_shape(sh, j['anchor'], j['axis'], v)
            else:                                                    # slide: v in inches along the axis
                d = mul(unit(j['axis']), v)
                f = lambda sh: translate_shape(sh, d)
            for p in self.pieces:
                if p.part in parts:
                    p.shapes = [f(sh) for sh in p.shapes]
                    p.cuts = [f(sh) for sh in p.cuts]


# ---------------------------------------------------------------- deck (one panel per 4 ft section)

def deck(m):
    t2, w4 = LUMBER['2x4']
    W = P.LANE['width']
    for i in range(P.LANE['sections']):
        x0 = i * P.LANE['section']
        g = '01 Deck/S%d deck panel' % (i + 1)
        zc = w4 / 2
        for k, xr in enumerate((x0 + t2 / 2, x0 + 48 - t2 / 2)):
            m.lumber(g, 'end rim %d' % (k + 1), '2x4', [xr, 0, zc], [xr, W, zc], up=Z)
        oc = P.FLOOR['oc']
        ys = [t2 / 2] + [oc * j for j in range(1, int(W / oc))] + [W - t2 / 2]
        for k, y in enumerate(ys):
            m.lumber(g, 'joist %d' % (k + 1), '2x4', [x0 + t2, y, zc], [x0 + 48 - t2, y, zc], up=Z)
        m.sheet(g, 'deck sheet', 'ply', ext(x0, x0 + 48, 0, W, w4, w4 + SHEET['ply']))
        m.screws(g, 'deck screw 3 in', len(ys) * 2 * 2, 'two through each end rim into every joist end')
        m.screws(g, 'deck screw 1-5/8 in', 2 * 17 + len(ys) * 7, 'sheet to frame: 6 in o.c. on the rims, 8 in o.c. on joists')
        m.note(g, 'Glue (construction adhesive) on every joist and rim top before the sheet goes on.')


# ---------------------------------------------------------------- lane walls (posts, rails, cap) and landing guards

def post_xs(i):
    """x of the two corner posts of section i (each 4 ft module has its own posts, spec p.2: double posts)."""
    x0 = i * P.LANE['section']
    pt = LUMBER['4x4'][0]
    return x0 + pt / 2, x0 + P.LANE['section'] - pt / 2


def rails(m):
    """Lane walls as one module per 4 ft section: two 4x4 posts per side bolted to the deck panel's side joist,
    three 2x4 rails on their outside face, a flat 2x6 cap. Special modules: the side gate (S6 front), the stair
    (S8 front: tall posts, the stair rails and skirt are in stairs()), the landing (S9: tall posts, guard rails)."""
    W = P.LANE['width']
    pt = LUMBER['4x4'][0]
    t2, w4 = LUMBER['2x4']
    lane_top = DECK_Z + P.RAIL['height']
    land_top = DECK_Z + P.LANDING['h']
    tall_top = land_top + P.LANDING['guard_h']
    n = P.LANE['sections']
    i_stair = int(P.X['stairs'][0] // P.LANE['section'])
    i_land = int(P.X['landing'][0] // P.LANE['section'])
    i_gate = int(P.GATES['side']['x'][0] // P.LANE['section'])
    cw = LUMBER['2x6'][1]
    for side, yc, yr in (('front', -pt / 2, -pt - t2 / 2), ('back', W + pt / 2, W + pt + t2 / 2)):
        for i in range(n):
            g = '02 Lane walls/%s S%d' % (side, i + 1)
            xa, xb = post_xs(i)
            x0, x1 = i * P.LANE['section'], (i + 1) * P.LANE['section']
            tall = i >= i_land or (i == i_stair and side == 'front')
            for k, x in enumerate((xa, xb)):
                m.post(g, 'post %d' % (k + 1), x, yc, 0.0, tall_top if tall else lane_top)
            m.screws(g, 'carriage bolt 1/2 x 6 in', 4, 'each post through the deck side joist (2 per post)')
            if i == i_stair and side == 'front':
                continue                                            # stair rails + skirt: stairs()
            if i >= i_land:
                for h in (12.0, 24.0, P.LANDING['guard_h']):
                    z = land_top + h - w4 / 2
                    m.lumber(g, 'guard rail %d' % h, '2x4', [x0, yr, z], [x1, yr, z], up=Z)
                m.screws(g, 'deck screw 3 in', 12, 'guard rails: 2 screws into each post')
                continue
            if i == i_gate and side == 'front':
                continue                                            # gates()
            for h in P.RAIL['rails']:
                z = DECK_Z + h - w4 / 2
                m.lumber(g, 'rail %d' % h, '2x4', [x0, yr, z], [x1, yr, z], up=Z)
            m.screws(g, 'deck screw 3 in', 12, 'rails: 2 screws into each post (top rail flush with the post tops)')
    gates(m)
    # end guard across the landing (x = 432) and the landing edge guard above the deck (x = 384, behind the stair)
    g = '02 Lane walls/landing end guard'
    x1 = P.X['landing'][1]
    xr = x1 + t2 / 2                                                 # rails on the S9 corner posts' end faces
    m.post(g, 'post middle', x1 + t2 + pt / 2, W / 2, 0.0, tall_top)   # outside the rails, bolted to the deck end
    for h in (12.0, 24.0, P.LANDING['guard_h']):
        z = land_top + h - w4 / 2
        m.lumber(g, 'rail %d' % h, '2x4', [xr, -pt, z], [xr, W + pt, z], up=Z)
    g = '02 Lane walls/landing edge guard'
    ys0 = P.STAIRS['y0'] + 1.5 + P.STAIRS['width']                      # back edge of the stair
    xg = P.X['landing'][0] - SHEET['osb'] - pt / 2                      # outside the landing skirt
    m.post(g, 'post at stair', xg, ys0 + 1.5 + pt / 2, DECK_Z, tall_top)   # behind the stair's back handrail
    m.post(g, 'post at back', xg, W - pt / 2, DECK_Z, tall_top)              # against the back lane post
    for h in (12.0, 24.0, P.LANDING['guard_h']):
        z = land_top + h - w4 / 2
        m.lumber(g, 'rail %d' % h, '2x4', [xg - pt / 2 - t2 / 2, ys0 + 1.5, z], [xg - pt / 2 - t2 / 2, W, z], up=Z)


def gate_panel(m, g, part, hinge, along, length, outward, heights=(12.0, 24.0, 36.0)):
    """A rail gate: two 2x4 stiles and 2x4 rails, hinged at `hinge` (x, y) on a vertical axis, closed along the
    unit vector `along`; it opens towards `outward`. Two gate hinges + a gate latch."""
    t2, w4 = LUMBER['2x4']
    h0, h1 = DECK_Z + 1.0, DECK_Z + max(heights)
    a = [hinge[0], hinge[1], 0]
    for k, d in enumerate((1.75, length - 1.75)):
        c = add(a, mul(along, d))
        m.lumber(g, 'stile %d' % (k + 1), '2x4', [c[0], c[1], h0], [c[0], c[1], h1], up=[along[0], along[1], 0],
                 part=part)
    for h in heights:
        z = DECK_Z + h - w4 / 2
        p0, p1 = add(a, mul(along, 3.5)), add(a, mul(along, length - 3.5))
        m.lumber(g, 'rail %d' % h, '2x4', [p0[0], p0[1], z], [p1[0], p1[1], z], up=Z, part=part)
    if length > 60:                                                  # long gate: diagonal brace against sag
        off = mul(unit([outward[0], outward[1], 0]), LUMBER['2x4'][0])   # on the outside face of the rails
        lo_ = add(add(a, mul(along, length - 3.5)), off)
        hi_ = add(add(a, mul(along, 3.5)), off)
        m.lumber(g, 'diagonal brace', '2x4', [lo_[0], lo_[1], DECK_Z + heights[0] + 0.0],
                 [hi_[0], hi_[1], DECK_Z + max(heights) - 3.5], up=[-along[1], along[0], 0], flat=True,
                 part=part, ends='angled both')
    side_v = [-along[1], along[0], 0]
    m.item(g, 'gate hinges', [box_shape(add(a, [0, 0, DECK_Z + 8.0]), (along, side_v, Z), (0.75, 2.6, 4.0)),
                              box_shape(add(a, [0, 0, DECK_Z + 30.0]), (along, side_v, Z), (0.75, 2.6, 4.0))],
           'gate hinge', 'galvanized', mass=0.4)
    ax = cross([along[0], along[1], 0], [outward[0], outward[1], 0])
    m.joints.append(dict(name=part, type='hinge', part=part, parent=None, anchor=[hinge[0], hinge[1], DECK_Z],
                         axis=unit(ax), range=(0.0, 100.0), note='rail gate'))


def gates(m):
    W = P.LANE['width']
    pt = LUMBER['4x4'][0]
    t2 = LUMBER['2x4'][0]
    # start gate: hinged on the back corner post, closes the back half of the start end, opens outwards (-X)
    # start end (spec p.1/2): a fixed rail panel across the back half, from the back corner post to a middle post,
    # and a gate hinged on that middle post closing the front half; it opens outwards (-X)
    G = P.GATES['start']
    w4 = LUMBER['2x4'][1]
    ym = G['middle_y']
    g = '02 Lane walls/start end panel'
    m.post(g, 'middle post', pt / 2, ym, DECK_Z + 0.1, DECK_Z + P.RAIL['height'])
    m.item(g, 'middle post base', [ext(pt / 2 - 1.9, pt / 2 + 1.9, ym - 1.9, ym + 1.9, DECK_Z, DECK_Z + 0.1)],
           'post base 4x4', 'galvanized', mass=0.5)
    xr = -t2 / 2                                                     # rails on the posts' start-side face
    for h in P.RAIL['rails']:
        z = DECK_Z + h - w4 / 2
        m.lumber(g, 'rail %d' % h, '2x4', [xr, ym - pt / 2, z], [xr, W + pt, z], up=Z)
    m.screws(g, 'deck screw 3 in', 12, 'rails: 2 screws into each post')
    gate_panel(m, '02 Lane walls/start gate (moving)', 'start gate', [xr - t2 / 2 - t2 / 2 - 0.25, ym - pt / 2],
               [0, -1, 0], ym - pt / 2 + pt - 0.5, [-1, 0, 0])
    # side gate in the front rail at C: hinged on the post at x = 242, opens outwards (-Y)
    x0, x1 = P.GATES['side']['x']
    gate_panel(m, '02 Lane walls/side gate (moving)', 'side gate', [x1 - pt - 0.25, -pt / 2], [-1, 0, 0],
               x1 - x0 - 2 * pt - 0.5, [0, -1, 0])                 # hinged on the right post (spec p.2)


# ---------------------------------------------------------------- door wall (task 2) with the self-closing door

def door_wall(m):
    D = P.DOOR
    W = P.LANE['width']
    t2, w4 = LUMBER['2x4']
    xc = P.X['door_wall']
    x0, x1 = xc - w4 / 2, xc + w4 / 2
    g = '03 Door wall/wall frame'
    z0 = DECK_Z
    stud = 92.625                                                    # precut stud
    ztp = z0 + t2 + stud                                             # underside of the top plates
    ro0, ro1 = D['y0'], D['y0'] + D['rough'][0]
    m.lumber(g, 'bottom plate front', '2x4', [xc, 0, z0 + t2 / 2], [xc, ro0, z0 + t2 / 2], up=X)
    m.lumber(g, 'bottom plate back', '2x4', [xc, ro1, z0 + t2 / 2], [xc, W, z0 + t2 / 2], up=X)
    m.note(g, 'Frame the bottom plate in one piece, then saw it out between the jack studs after the wall is up.')
    m.lumber(g, 'top plate', '2x4', [xc, 0, ztp + t2 / 2], [xc, W, ztp + t2 / 2], up=X)
    m.lumber(g, 'cap plate', '2x4', [xc, 0, ztp + 1.5 * t2], [xc, W, ztp + 1.5 * t2], up=X)
    ro_top = z0 + t2 + D['rough'][1]
    studs = [t2 / 2, 16.0, ro0 - t2 * 1.5, ro1 + t2 * 1.5, 80.0, W - t2 / 2]
    for k, y in enumerate(studs):
        m.lumber(g, 'stud %d' % (k + 1) if k not in (2, 3) else 'king stud %d' % (k - 1), '2x4',
                 [xc, y, z0 + t2], [xc, y, ztp], up=X)
    for k, y in enumerate((ro0 - t2 / 2, ro1 + t2 / 2)):
        m.lumber(g, 'jack stud %d' % (k + 1), '2x4', [xc, y, z0 + t2], [xc, y, ro_top], up=X)
    hy0, hy1 = ro0 - t2, ro1 + t2
    hh = LUMBER['2x6'][1]
    for k, dx in enumerate((-w4 / 2 + t2 / 2, w4 / 2 - t2 / 2)):
        m.lumber(g, 'header %d' % (k + 1), '2x6', [xc + dx, hy0, ro_top + hh / 2], [xc + dx, hy1, ro_top + hh / 2], up=Z)
    for k, y in enumerate((ro0 + 8.0, (ro0 + ro1) / 2, ro1 - 8.0)):
        m.lumber(g, 'cripple %d' % (k + 1), '2x4', [xc, y, ro_top + hh], [xc, y, ztp], up=X)
    # A-frame braces at both ends, both sides (spec pictures), feet on 2x4 cleats screwed to the deck
    for y in (t2 / 2, W - t2 / 2):
        for s in (-1, 1):
            top = [xc + s * (w4 / 2 + SHEET['osb'] + 0.6), y, z0 + 84.0]
            foot = [xc + s * 30.0, y, z0 + w4 / 2]
            m.lumber(g, 'brace %s %s' % ('front' if y < W / 2 else 'back', 'start side' if s < 0 else 'far side'), '2x4',
                     top, foot, up=[s, 0, 1], ends='miter both')
            m.lumber(g, 'brace cleat %s %s' % ('front' if y < W / 2 else 'back', 'start' if s < 0 else 'far'), '2x4',
                     [xc + s * 18.0, y + (1.0 if y < W / 2 else -1.0), z0 + t2 / 2],
                     [xc + s * 29.0, y + (1.0 if y < W / 2 else -1.0), z0 + t2 / 2], up=Z, flat=True)
    # OSB skins both faces, door opening cut out
    gs = '03 Door wall/wall skins'
    to = SHEET['osb']
    opening = ext(x0 - 2, x1 + 2, ro0, ro1, z0 - 1, ro_top)
    for side, xa, xb in (('start side', x0 - to, x0), ('far side', x1, x1 + to)):
        for k, (ya, yb) in enumerate(((0, 48.0), (48.0, W))):
            m.sheet(gs, 'OSB %s %d' % (side, k + 1), 'osb', ext(xa, xb, ya, yb, z0, z0 + 96.0), cuts=[opening],
                    dims=(96.0, 48.0))
    m.screws(gs, 'screw 1-5/8 in', 4 * 30, 'OSB to studs and plates, 6 in o.c. edges / 12 in field')
    m.screws(g, 'deck screw 3 in', 60, 'plates to studs (2 per end), bottom plate to deck (16 in o.c.), braces')

    # prehung door unit: jambs, stops, slab, 3 spring hinges, lever set
    gj = '03 Door wall/door jamb (buy)'
    jt = 0.75
    jx0, jx1 = x0 - to, x1 + to
    sh = 0.25
    jy0, jy1 = ro0 + sh, ro1 - sh                                     # jamb outer faces
    head_z = z0 + 81.25
    for k, (ya, yb) in enumerate(((jy0, jy0 + jt), (jy1 - jt, jy1))):
        m.item(gj, 'side jamb %d' % (k + 1), [ext(jx0, jx1, ya, yb, z0, head_z)], 'prehung door', 'paint white')
    m.item(gj, 'head jamb', [ext(jx0, jx1, jy0, jy1, head_z, head_z + jt)], 'prehung door', 'paint white')
    slab_t = D['slab'][2]
    sx0 = jx0 + 0.0625                                               # slab start-side face: swings towards the start
    sx1 = sx0 + slab_t
    stop = [ext(sx1 + 0.0625, sx1 + 0.4375, jy0 + jt, jy0 + jt + 0.5, z0, head_z),
            ext(sx1 + 0.0625, sx1 + 0.4375, jy1 - jt - 0.5, jy1 - jt, z0, head_z),
            ext(sx1 + 0.0625, sx1 + 0.4375, jy0 + jt, jy1 - jt, head_z - 0.5, head_z)]
    m.item(gj, 'door stop', stop, 'prehung door', 'paint white')
    for side, xa, xb in (('start side', jx0 - 0.5, jx0), ('far side', jx1, jx1 + 0.5)):
        cas = [ext(xa, xb, jy0 - 2.25, jy0, z0, head_z + jt + 2.25),
               ext(xa, xb, jy1, jy1 + 2.25, z0, head_z + jt + 2.25),
               ext(xa, xb, jy0 - 2.25, jy1 + 2.25, head_z + jt, head_z + jt + 2.25)]
        m.item(gj, 'casing %s' % side, cas, 'casing', 'paint white')

    # door leaf: hinges on the back (+Y) jamb, lever on the front edge (spec 'view from the starting location');
    # it swings towards the start side (spec overview pictures show it open on that side)
    gd = '03 Door wall/door leaf (moving)'
    sw = 35.75                                                       # a '36 in' slab is 35-3/4 wide
    hy = jy1 - jt - 0.125                                            # hinge edge (back jamb)
    ly = hy - sw                                                     # latch edge (front)
    slab = ext(sx0, sx1, ly, hy, z0 + 0.5, z0 + 0.5 + 79.5)
    m.item(gd, 'door slab 36 x 80', [slab], 'prehung door', 'door grey', mass=14.0, part='door leaf')
    zl = z0 + D['lever_h']
    yl = ly + 2.375                                                  # 2-3/8 in backset from the latch edge
    gl = '03 Door wall/door lever (moving)'
    for side, xs, sgn in (('start side', sx0, -1), ('far side', sx1, 1)):
        rose = ('cyl', [xs, yl, zl], [xs + sgn * 0.45, yl, zl], 1.3, 1.3)
        m.item(gd, 'rose %s' % side, [rose], 'lever passage set', 'satin nickel', mass=0.1, part='door leaf')
        neck = ('cyl', [xs + sgn * 0.4, yl, zl], [xs + sgn * 2.2, yl, zl], 0.4, 0.4)
        lever = box_shape([xs + sgn * 2.2, yl + 2.3, zl], I3, (0.6, 4.8, 0.55))   # points towards the hinges
        m.item(gl, 'lever %s' % side, [neck, lever], 'lever passage set', 'satin nickel', mass=0.25,
               part='door lever')
    m.item(gl, 'latch bolt', [ext(sx0 + 0.3, sx1 - 0.3, ly - 0.45, ly, zl - 0.5, zl + 0.5)], 'lever passage set',
           'satin nickel', part='door lever')
    gh = '03 Door wall/spring hinges (buy)'
    for k, zh in enumerate((z0 + 0.5 + 7.0, z0 + 0.5 + 40.0, z0 + 0.5 + 79.5 - 11.0)):
        knuckle = ('cyl', [sx0 - 0.12, hy + 0.12, zh - 2.0], [sx0 - 0.12, hy + 0.12, zh + 2.0], 0.3, 0.3)
        leaf_d = ext(sx0, sx0 + 1.5, hy - 0.06, hy, zh - 2.0, zh + 2.0)
        leaf_f = ext(sx0, sx0 + 1.5, hy, hy + 0.06, zh - 2.0, zh + 2.0)
        m.item(gh, 'spring hinge %d' % (k + 1), [knuckle, leaf_d, leaf_f], 'spring hinge 4 in', 'satin nickel', mass=0.4)
    # opening = rotation about -Z through the knuckles: the latch edge moves to -X (towards the start)
    m.joints.append(dict(name='door hinge', type='hinge', part='door leaf', parent=None,
                         anchor=[sx0 - 0.12, hy + 0.12, z0], axis=[0, 0, -1], range=(0.0, 95.0),
                         spring=dict(kind='torsion', k_nm_per_rad=1.2, preload_nm=1.5,
                                     note='3 x 4 in spring hinges: ~0.5 N m each at closed, ~1.5 N m at 90 deg'),
                         damping=1.0))
    m.joints.append(dict(name='door lever', type='hinge', part='door lever', parent='door leaf',
                         anchor=[sx0, yl, zl], axis=[1, 0, 0], range=(0.0, 45.0),
                         spring=dict(kind='return', k_nm_per_rad=0.6, preload_nm=0.3, note='latch return spring'),
                         latch=dict(release_deg=30.0)))
    m.note(gd, 'Hang the prehung unit in the rough opening: shim behind each hinge and the latch, 2 screws per shim '
              'point through the jamb into the jack stud. Replace the plain hinges with the spring hinges, '
              'set the spring tension 3/4 turn (until the door closes and latches from 90 deg).')

    # keypad (task 1): 4x4 post on a post base, plywood panel, 10 LED push buttons, 5-digit display
    kp = P.KEYPAD
    gk = '04 Keypad/keypad post'
    px, py = kp['pos']
    top = DECK_Z + kp['center_h'] + kp['board'][1] / 2
    m.post(gk, 'keypad post', px, py, DECK_Z + 0.1, top)
    m.item(gk, 'post base', [ext(px - 1.9, px + 1.9, py - 1.9, py + 1.9, DECK_Z, DECK_Z + 0.1),
                             ext(px - 1.9, px + 1.9, py - 1.9, py - 1.8, DECK_Z, DECK_Z + 3.0),
                             ext(px - 1.9, px + 1.9, py + 1.8, py + 1.9, DECK_Z, DECK_Z + 3.0)],
           'post base 4x4', 'galvanized', mass=0.5)
    bw, bh, bt = kp['board']
    bx1 = px - 1.75 - 3.5                                            # panel on a 1x4 box frame, start side of the post
    zc = DECK_Z + kp['center_h']
    m.sheet(gk, 'keypad panel', 'ply', ext(bx1 - bt, bx1, py - bw / 2, py + bw / 2, zc - bh / 2, zc + bh / 2),
            dims=(bh, bw))
    plate_box(m, gk, 'keypad', [bx1 - bt / 2, py, zc], ([0, -1, 0], Z, [-1, 0, 0]), (bw, bh, bt))
    pitch = kp['pitch']
    keys = [(r, c) for r in range(3) for c in range(3)] + [(3, 1)]
    labels = '1234567890'
    for n, (r, c) in enumerate(keys):
        yb = py + (1 - c) * pitch                                    # columns left (+Y) to right as seen from -X
        zb = zc + 2.2 - r * pitch
        button(m, '04 Keypad/key %s (moving)' % labels[n], [bx1 - bt, yb, zb], [-1, 0, 0], 'key ' + labels[n],
               lit=labels[n] in '13592')                             # the spec picture shows code 13592 lit
    dw, dh = kp['display']
    m.item(gk, 'display bezel', [ext(bx1 - bt - 0.4, bx1 - bt, py - dw / 2 - 0.3, py + dw / 2 + 0.3,
                                     zc + 7.4, zc + 7.4 + dh + 0.6)], '5-digit display', 'black plastic', mass=0.05)
    m.item(gk, 'display window', [ext(bx1 - bt - 0.42, bx1 - bt - 0.38, py - dw / 2, py + dw / 2,
                                      zc + 7.7, zc + 7.7 + dh)], '5-digit display', 'display red')
    tie_blocks(m, gk, 'keypad post', [px + 1.75, py], [P.X['door_wall'] - 1.75 - SHEET['osb'], py], (20.0, 40.0),
               'keypad post ties: toe-screw into the post, screw through the OSB into a stud / added blocking')
    m.item(gk, 'electronics box', [ext(bx1, bx1 + 3.1, py - 6.5, py - 1.5, zc - 10.0, zc - 4.0)],
           'project box', 'grey plastic', mass=0.4)


def tie_blocks(m, g, name, a_xy, b_xy, heights, note):
    """2x4 blocking on edge from a freestanding post face (a_xy) to something solid (b_xy) at each height above the
    deck: a post on a post base is not a moment connection, so a robot pushing on a task board would rack it."""
    for h in heights:
        z = DECK_Z + h
        m.lumber(g, '%s tie %d' % (name, h), '2x4', [a_xy[0], a_xy[1], z], [b_xy[0], b_xy[1], z], up=Z)
    m.screws(g, 'deck screw 3 in', 4 * len(heights), note)


def plate_box(m, g, name, c, axes, size, depth=3.5):
    """1x4 frame behind a panel (centre c, axes (u, v, n), size (w, h, t)): turns a flat board into a shallow box
    so push-button switch bodies and wiring fit behind it."""
    u, v, n = axes
    w, h, t = size
    back = add(c, mul(n, -(t / 2 + depth / 2)))
    for nm, off, sz, L in (('top', mul(v, h / 2 - 0.375), (w, 0.75, depth), w),
                           ('bottom', mul(v, -h / 2 + 0.375), (w, 0.75, depth), w),
                           ('left', mul(u, -w / 2 + 0.375), (0.75, h - 1.5, depth), h - 1.5),
                           ('right', mul(u, w / 2 - 0.375), (0.75, h - 1.5, depth), h - 1.5)):
        m.add(Piece(g, '%s frame %s' % (name, nm), [box_shape(add(back, off), (u, v, n), sz)], 'lumber', '1x4', 'pine',
                    mat='softwood', cut=dict(length=L, ends='square')))


def button(m, group, face_pt, n, part, lit=False):
    """60 mm LED dome push button through a panel: face_pt on the panel face, n = outward normal. The cap is
    its own moving part (spring return)."""
    n = unit(n)
    bezel_r, cap_r = (4.3 / 2, 3.5 / 2)                               # 100 mm LED dome button
    a = face_pt
    m.item(group, 'bezel',
           [('cyl', a, add(a, mul(n, 0.3)), bezel_r, bezel_r)], 'LED dome button 100 mm', 'black plastic', mass=0.03)
    cap = [('cyl', add(a, mul(n, 0.3)), add(a, mul(n, 0.6)), cap_r, cap_r),
           ('cyl', add(a, mul(n, 0.6)), add(a, mul(n, 1.1)), cap_r, cap_r * 0.6)]          # dome
    m.item(group, 'cap', cap, 'LED dome button 100 mm', 'button yellow' if lit else 'white plastic', mass=0.02,
           part=part)                                                # lit = LED on (spec pictures: yellow)
    m.item(group, 'switch body', [('cyl', a, add(a, mul(n, -2.0)), 1.3, 1.3)], 'LED dome button 100 mm',
           'black plastic', mass=0.04)
    m.joints.append(dict(name=part, type='slide', part=part, anchor=a, axis=mul(n, -1), range=(0.0, 0.16),
                         spring=dict(kind='linear', k_n_per_m=600.0, preload_n=3.0)))


# ---------------------------------------------------------------- task boards: lightbulb (3), valve (4), button (5)

def task_board(m, kind, x, y, face, mount='post', tilt_deg=None):
    """A 2 x 2 ft plywood board at wall height and a second one leaning at floor height, on a 4x4 post
    (mount='post', boards on the post face at (x, y)) or on the far face of the door wall (mount='wall').
    face: unit horizontal vector the boards face (towards the robot)."""
    B = P.BOARD
    f = unit(list(face))
    side = cross(Z, f)                                               # board u axis (horizontal)
    g = '05 Task boards/%s board' % kind
    K = B.get('by_kind', {}).get(kind, {})                           # per-board size overrides
    if mount == 'wall':
        x = P.X['door_wall'] + LUMBER['2x4'][1] / 2 + SHEET['osb'] - 1.75   # so 'post face' = wall face
    else:
        top = DECK_Z + B['upper_center_h'] + K.get('plate', B['plate'])[1] / 2 + 2.0
        m.post(g, '%s post' % kind, x, y, DECK_Z + 0.1, top)
        m.item(g, 'post base', [ext(x - 1.9, x + 1.9, y - 1.9, y + 1.9, DECK_Z, DECK_Z + 0.1)], 'post base 4x4',
               'galvanized', mass=0.5)
        W = P.LANE['width']
        rail_in = LUMBER['4x4'][0]                                   # lane rails sit outside the lane posts
        if y < 24.0:
            tie_blocks(m, g, '%s post' % kind, [x, y - 1.75], [x, -rail_in], (10.25, 22.25),
                       'ties from the post to the front rails (screw through the rail into the tie)')
        elif y > W - 24.0:
            tie_blocks(m, g, '%s post' % kind, [x, y + 1.75], [x, W + rail_in], (10.25, 22.25),
                       'ties from the post to the back rails (screw through the rail into the tie)')
        else:                                                        # mid-lane: two kickers to the deck, sideways
            for sgn in (-1, 1):
                m.lumber(g, '%s post kicker %s' % (kind, 'front' if sgn < 0 else 'back'), '2x4',
                         [x, y + sgn * 1.75, DECK_Z + 22.0], [x, y + sgn * 20.0, DECK_Z + 0.45], up=X,
                         ends='angled both')
            m.screws(g, 'lag screw 3/8 x 4 in', 4, 'kickers: one lag into the post, one through a cleat into the deck')
    pw, ph, pt = K.get('plate', B['plate'])
    dpt = 3.5 if kind == 'button' else 0.0                           # button boards are shallow boxes (switch bodies)
    # upper board on the post's face side, turned a little in its plane (spec p.3: 'multiple orientations')
    spin = B['spin'].get(kind, 0.0)
    su, sv = rot_axis(side, f, spin), rot_axis(Z, f, spin)
    c_up = add([x, y, DECK_Z + B['upper_center_h']], mul(f, 1.75 + dpt + pt / 2))
    m.sheet(g, 'upper board', 'ply', box_shape(c_up, (su, sv, f), (pw, ph, pt)), dims=(pw, ph))
    if dpt:
        plate_box(m, g, 'upper board', c_up, (su, sv, f), (pw, ph, pt), dpt)
    face_up = add(c_up, mul(f, pt / 2))
    # lower board leaning back: bottom edge on the deck, top edge against the post
    tilt = math.radians(tilt_deg if tilt_deg is not None else B['lower_tilt_deg'])
    n_lo = unit(add(mul(f, math.cos(tilt)), mul(Z, math.sin(tilt))))
    v_lo = cross(n_lo, side)
    if v_lo[2] < 0:
        v_lo = mul(v_lo, -1)
    # back face (plate, or its box frame) rests on the deck at the bottom and on the post/wall face at the top
    bf = 1.75 + ph * math.sin(tilt)                                  # bottom-back edge, from the post centre
    B0 = add([x, y, DECK_Z], mul(f, bf))
    c_lo = add(add(B0, mul(v_lo, ph / 2)), mul(n_lo, dpt + pt / 2))
    m.sheet(g, 'lower board', 'ply', box_shape(c_lo, (side, v_lo, n_lo), (pw, ph, pt)), dims=(pw, ph))
    if dpt:
        plate_box(m, g, 'lower board', c_lo, (side, v_lo, n_lo), (pw, ph, pt), dpt)
    face_lo = add(c_lo, mul(n_lo, pt / 2))
    # stop cleat in front of the bottom edge (2x4 flat, screwed to the deck): the board can't slide out
    t_all = dpt + pt
    front_at = bf + t_all * math.cos(tilt) + 0.05                    # just in front of the bottom-front edge
    cc = add([x, y, DECK_Z + 0.75], mul(f, front_at + 1.75))
    m.lumber(g, 'lower board cleat', '2x4', add(cc, mul(side, -pw / 2 + 1)), add(cc, mul(side, pw / 2 - 1)), up=Z,
             flat=True)
    m.screws(g, 'screw 1-5/8 in', 16, '8 per board into the post / cleat')
    sp = K.get('spacing', B['spacing'])
    spots = [(0, 0), (-sp, sp), (sp, sp), (-sp, -sp), (sp, -sp)]
    if kind == 'valve':                                              # a painted 2x4 '+' cross, valves on its arms
        spots = [(0, 0), (-sp, 0), (sp, 0), (0, sp), (0, -sp)]
    for lvl, ctr, nn, uu, vv in (('upper', face_up, f, su, sv), ('lower', face_lo, n_lo, side, v_lo)):
        if kind == 'valve':
            gc = '05 Task boards/valve %s cross' % lvl
            for nm, c0, sz, look in (('arm left-right', [0, 0], (20.0, 3.5), 'paint blue'),
                                     ('arm up', [0, 5.875], (3.5, 8.25), 'paint orange'),
                                     ('arm down', [0, -5.875], (3.5, 8.25), 'paint purple')):
                cc = add(add(add(ctr, mul(uu, c0[0])), mul(vv, c0[1])), mul(nn, 0.75))
                m.add(Piece(gc, nm, [box_shape(cc, (uu, vv, nn), (sz[0], sz[1], 1.5))], 'lumber', '2x4', look,
                            mat='softwood', cut=dict(length=max(sz), ends='square, painted')))
            ctr = add(ctr, mul(nn, 1.5))                             # valves stand on the cross
        for k, (u, v) in enumerate(spots):
            p = add(add(ctr, mul(uu, u)), mul(vv, v))
            gi = '05 Task boards/%s %s %d' % (kind, lvl, k + 1)
            if kind == 'lightbulb':
                lightbulb(m, gi, p, nn, '%s bulb %s %d' % (kind, lvl, k + 1), seated=(k % 2 == 0))
            elif kind == 'valve':
                valve(m, gi, p, nn, uu, vv, k, '%s %s %d' % (kind, lvl, k + 1))
            else:
                button(m, gi + ' (moving)', p, nn, 'button %s %d' % (lvl, k + 1))


def lightbulb(m, g, p, n, part, seated=True):
    """Porcelain keyless lampholder (E26) on the board and a 3D-printed A19 bulb. seated=False: the bulb is
    loose (in the bulb box instead), only the holder is on the board."""
    n = unit(n)
    m.item(g, 'lampholder base', [('cyl', p, add(p, mul(n, 0.9)), 2.0, 1.6)], 'porcelain lampholder', 'porcelain',
           mass=0.15)
    m.item(g, 'lampholder socket', [('cyl', add(p, mul(n, 0.9)), add(p, mul(n, 2.1)), 0.85, 0.85)],
           'porcelain lampholder', 'porcelain')
    if seated:
        b0 = add(p, mul(n, 1.6))
        bulb(m, g, b0, n, part)


def bulb(m, g, b0, n, part):
    """3D-printed A19 bulb: E26 thread section (printed), neck, 60 mm globe. b0 = thread base point."""
    thread = ('cyl', b0, add(b0, mul(n, 1.0)), 0.51, 0.51)
    neck = ('cyl', add(b0, mul(n, 1.0)), add(b0, mul(n, 2.1)), 0.6, 1.05)
    globe = ('sphere', add(b0, mul(n, 3.1)), 1.18)
    m.printed(g, 'bulb (print)', [thread, neck, globe], look='bulb', part=part, buy='printed bulb')
    m.joints.append(dict(name=part, type='screw', part=part, anchor=b0, axis=n, range=(0.0, 720.0),
                         pitch_in=1 / 7.0, note='E26: 7 threads per inch; seated after about 2 turns'))


def valve(m, g, p, n, u, v, k, part):
    """3/4 in lever ball valve on short nipples, held by a painted 2x4 block. Orientation varies per spot k:
    pipe along u, along v, or along the board normal (spec: 'mounted in multiple orientations')."""
    n = unit(n)
    look_blk = ('paint yellow', 'paint blue', 'paint blue', 'paint orange', 'paint purple')[k]
    pipe_ax = (n, u, v, v, u)[k]                                     # spot 0 (centre): pipe sticks out of the board
    if k == 0:
        stem = u
        c = add(p, mul(n, 3.0))
        m.item(g, 'mount block', [box_shape(add(p, mul(n, 0.75)), (u, v, n), (3.5, 3.5, 1.5))], 'block', look_blk)
        nip = [('cyl', add(p, mul(n, 1.5)), add(c, mul(n, -1.3)), 0.525, 0.525),
               ('cyl', add(c, mul(n, 1.3)), add(c, mul(n, 3.0)), 0.525, 0.525)]
    else:
        stem = n
        blk_t = 1.5 if k in (1, 4) else 3.0                         # two heights of blocks
        c = add(p, mul(n, blk_t + 0.525))                           # nipples rest on the block
        m.item(g, 'mount block', [box_shape(add(p, mul(n, blk_t / 2)), (u, v, n), (5.0, 3.5, blk_t) if pipe_ax is u
                                            else (3.5, 5.0, blk_t))], 'block', look_blk)
        nip = [('cyl', add(c, mul(pipe_ax, -1.3)), add(c, mul(pipe_ax, -3.0)), 0.525, 0.525),
               ('cyl', add(c, mul(pipe_ax, 1.3)), add(c, mul(pipe_ax, 3.0)), 0.525, 0.525)]
        m.item(g, 'pipe strap', [box_shape(add(add(c, mul(pipe_ax, -2.4)), mul(n, 0.03)), (pipe_ax, cross(n, pipe_ax), n),
                                           (0.75, 1.6, 1.11))],      # one-hole strap over the nipple, screwed to the block
               'pipe strap', 'galvanized')
    m.item(g, 'nipples', nip, 'nipple 3/4 x close', 'galvanized', mass=0.1)
    body = [('cyl', add(c, mul(pipe_ax, -1.3)), add(c, mul(pipe_ax, 1.3)), 0.75, 0.75),
            ('sphere', c, 0.85)]
    m.item(g, 'valve body', body, 'ball valve 3/4 in', 'brass', mass=0.35)
    s0 = add(c, mul(stem, 0.8))
    m.item(g, 'stem', [('cyl', add(c, mul(stem, 0.6)), add(s0, mul(stem, 0.4)), 0.22, 0.22)], 'ball valve 3/4 in', 'brass',
           part=part)
    perp = cross(stem, pipe_ax)                                      # closed: lever square to the pipe
    lever_c = add(add(s0, mul(stem, 0.45)), mul(perp, 2.3))
    lever_a = add(s0, mul(stem, 0.45))
    m.item(g, 'lever (red grip)', [box_shape(lever_c, frame_along(lever_a, add(lever_a, mul(perp, 4.6)), stem),
                                             (0.7, 0.18, 4.6))], 'ball valve 3/4 in', 'red vinyl', mass=0.05, part=part)
    m.joints.append(dict(name=part, type='hinge', part=part, anchor=c, axis=stem, range=(0.0, 90.0),
                         friction_nm=1.5, note='quarter turn; closed = lever square to the pipe, open = parallel'))


# ---------------------------------------------------------------- mobility: stepovers (A), ramps (B), wall (C)

def stepovers(m):
    S = P.STEPOVER
    s = LUMBER['4x4'][0]
    for k, (a, b) in enumerate(S['pieces']):
        g = '06 Stepover/stepover %d' % (k + 1)
        m.lumber(g, '4x4 stepover', '4x4', [a[0], a[1], DECK_Z + s / 2], [b[0], b[1], DECK_Z + s / 2], up=Z,
                 ends='square, laid loose')


def ramps(m):
    R = P.RAMPS
    cell = R['cell']
    th = math.radians(R['slope_deg'])
    t = SHEET['ply']
    peak = R['peak']
    rib_h = peak - t / math.cos(th)                                  # rib top line under the sheet
    tt, ww = LUMBER['2x8']
    for idx, dirn in enumerate(R['layout']):
        ci, cj = idx % 2, idx // 2
        x0 = P.X['ramps'][0] + ci * cell
        y0 = cj * cell
        g = '07 Crossing ramps/cell %d%d %s ridge' % (ci + 1, cj + 1, dirn.upper())
        a_ax = X if dirn == 'x' else Y                               # ridge direction
        b_ax = Y if dirn == 'x' else X                               # slope direction
        org = [x0, y0, DECK_Z]
        P_ = lambda a, b, z: add(add(org, mul(a_ax, a)), add(mul(b_ax, b), [0, 0, z]))
        # three triangle ribs (2x8 cut to a 15 deg gable), square to the ridge, at both ends and the middle
        nrib = R['ribs']
        for k, a in enumerate([tt / 2] + [cell * j / (nrib - 1) for j in range(1, nrib - 1)] + [cell - tt / 2]):
            c = P_(a, cell / 2, ww / 2)
            rib = box_shape(c, (cross(Z, b_ax), Z, b_ax), (tt, ww, cell))
            cuts = []
            for sgn in (-1, 1):                                      # the two slopes: big boxes above the roof line
                n_up = unit(add(mul(b_ax, sgn * math.sin(th)), mul(Z, math.cos(th))))     # outward + up
                eb = unit(add(mul(b_ax, sgn * math.cos(th)), mul(Z, -math.sin(th))))   # down the slope
                ridge_pt = P_(a, cell / 2, rib_h)
                mid = lerp(ridge_pt, P_(a, cell / 2 + sgn * cell / 2, rib_h - cell / 2 * math.tan(th)), 0.5)
                cuts.append(box_shape(add(mid, mul(n_up, 10.0)), (cross(n_up, eb), n_up, eb), (tt + 2, 20.0, cell + 4)))
            m.add(Piece(g, 'rib %d (2x8 cut to 15 deg)' % (k + 1), [rib], 'lumber', '2x8', 'pine', mat='softwood',
                        cuts=cuts, cut=dict(length=cell, ends='gable 15 deg, peak %.2f in' % rib_h)))
        # ridge board (2x4 flat under the peak) and two 3/4 ply skins
        for sgn in (-1, 1):
            n_up = unit(add(mul(b_ax, sgn * math.sin(th)), mul(Z, math.cos(th))))     # outward + up
            eb = unit(add(mul(b_ax, sgn * math.cos(th)), mul(Z, -math.sin(th))))   # down the slope
            run = cell / 2 / math.cos(th)
            ridge_top = P_(cell / 2, cell / 2, peak)
            c = add(add(ridge_top, mul(eb, run / 2)), mul(n_up, -t / 2))
            ax = (cross(n_up, eb), n_up, eb)                         # right-handed; the first axis lies along the ridge
            bevel = [ext(-1e3, 1e3, -1e3, 1e3, -1e3, DECK_Z),                       # foot: ripped 15 deg
                     box_shape(P_(cell / 2, cell / 2 - sgn * 50.0, 0), (a_ax, b_ax, Z) if dot(cross(a_ax, b_ax), Z) > 0
                               else (b_ax, a_ax, Z), (100.0, 100.0, 400.0))]          # ridge: beyond the centre line
            m.sheet(g, 'skin %s' % ('A' if sgn < 0 else 'B'), 'ply', box_shape(c, ax, (cell, t, run + 0.3)),
                    cuts=bevel, dims=(cell, round(run + 0.3, 3)))
        m.screws(g, 'screw 1-5/8 in', 2 * nrib * 4, 'skins to ribs, 4 per rib per side')
        m.screws(g, 'deck screw 3 in', 2 * nrib, 'ribs toe-screwed to the deck, 2 per rib')
        m.note(g, 'Cut the ribs from 2x8: mark the peak in the middle, %.2f in high, snap lines to both bottom '
                  'corners, cut with a circular saw set to 0 deg.' % rib_h)


def wall_obstacle(m):
    W = P.WALL_OBSTACLE
    g = '08 Flat ground/small wall obstacle'
    t2, w4 = LUMBER['2x4']
    x = W['x']
    m.lumber(g, 'bottom plate', '2x4', [x, 0, DECK_Z + t2 / 2], [x, W['length'], DECK_Z + t2 / 2], up=X)
    m.lumber(g, 'top plate', '2x4', [x, 0, DECK_Z + W['h'] - t2 / 2], [x, W['length'], DECK_Z + W['h'] - t2 / 2], up=X)
    for k, y in enumerate((t2 / 2, W['length'] / 2, W['length'] - t2 / 2)):
        m.lumber(g, 'stud %d' % (k + 1), '2x4', [x, y, DECK_Z + t2], [x, y, DECK_Z + W['h'] - t2], up=X)
    for side, s in (('start side', -1), ('far side', 1)):
        xa = x + s * w4 / 2
        m.sheet(g, 'skin %s' % side, 'ply', ext(min(xa, xa + s * SHEET['ply']), max(xa, xa + s * SHEET['ply']),
                                                  0, W['length'], DECK_Z, DECK_Z + W['h']), dims=(W['length'], W['h']))


# ---------------------------------------------------------------- stairs (D) and landing

def stairs(m):
    S = P.STAIRS
    r, h, nr = S['run'], S['rise'], S['risers']
    xl = P.X['landing'][0]
    xs = xl - (nr - 1) * r                                          # face of the first riser cut (x_1)
    y0 = S['y0'] + 1.5                                              # handrail board between the posts and the stair
    y1 = y0 + S['width']
    tt, tw = LUMBER['2x12']
    th = math.atan2(h, r)
    g = '09 Stairs/stringers'
    xk = lambda k: xs + (k - 1) * r                                 # riser k cut line (k = 1 .. nr)
    zk = lambda k: DECK_Z + h * k - 1.5                             # seat k (under a 1.5 in tread)
    ez = [math.cos(th), 0, math.sin(th)]
    ey = [-math.sin(th), 0, math.cos(th)]
    ex = cross(ey, ez)
    top_line = lambda x: zk(1) + (x - xk(1)) * h / r
    xa, xb = xk(1) - 20.0, xl + 10.0
    mid = [(xa + xb) / 2, 0, top_line((xa + xb) / 2)]
    Lb = (xb - xa) / math.cos(th)
    ns = S['stringers']
    for k in range(ns):
        y = y0 + tt / 2 + k * (S['width'] - tt) / (ns - 1)
        c = add([mid[0], y, mid[2]], mul(ey, -tw / 2))
        cuts = [ext(xk(1) - 200, xk(1), y - 1, y + 1, -10, 200),     # plumb cut at the first riser
                ext(xk(1) - 200, xl + 200, y - 1, y + 1, -200, DECK_Z),   # floor cut
                ext(xl, xl + 200, y - 1, y + 1, -10, 200)]           # plumb cut against the landing rim
        for kk in range(1, nr):
            cuts.append(ext(xk(kk), xk(kk + 1), y - 1, y + 1, zk(kk), zk(kk) + 30))
        L_cut = (xl - (xk(1) - (zk(1) - DECK_Z) * r / h)) / math.cos(th)
        m.add(Piece(g, 'stringer %d' % (k + 1), [box_shape(c, (ex, ey, ez), (tt, tw, Lb))], 'lumber', '2x12', 'pine',
                    mat='softwood', cuts=cuts,
                    cut=dict(length=math.ceil(L_cut + 2), ends='stair stringer: %d risers x %.2f, run %.2f' % (nr, h, r))))
    g = '09 Stairs/treads and risers'
    ply = SHEET['ply']
    for k in range(1, nr):
        ztop = DECK_Z + h * k
        xf = xk(k) - ply - 0.5                                       # 1/2 in nosing past the riser face
        td = r + 0.5                                                 # tread depth: run + nosing (riser behind it)
        stock = '2x10' if td <= 9.25 else '2x12'
        m.add(Piece(g, 'tread %d' % k, [ext(xf, xf + td, y0, y1, ztop - 1.5, ztop)], 'lumber', stock, 'pine',
                    mat='softwood', cut=dict(length=S['width'], ends='square; rip to %.2f in' % td)))
    for k in range(1, nr + 1):
        zb = DECK_Z + h * (k - 1)
        zt = zk(k) if k < nr else DECK_Z + h * nr - SHEET['ply']
        m.sheet(g, 'riser %d' % k, 'ply', ext(xk(k) - ply, xk(k), y0, y1, zb, zt), dims=(S['width'], round(zt - zb, 3)))
    m.screws(g, 'deck screw 3 in', (nr - 1) * 2 * ns * 2, 'each tread board: 2 screws into each stringer')
    m.screws(g, 'screw 1-5/8 in', nr * 2 * ns, 'risers: 2 per stringer')
    m.screws('09 Stairs/stringers', 'joist hanger / stair angle', ns, 'stringer tops to the landing rim')
    # handrails: a 4x4 newel post at the foot of each side; the front rails then screw to the inside of the
    # lane posts at x = 290 and 338, the back rails end on the landing edge guard post
    gr = '09 Stairs/handrails'
    nose = lambda x: DECK_Z + h * (1 + (x - xk(1)) / r)
    xn = xk(1) - 4.0                                                # back newel, in front of the first riser
    xa_front = P.X['stairs'][0] + LUMBER['4x4'][0]                   # front rails start on the S8 front post
    for side, yn, y, xb_ in (('front', None, y0 - 0.75, xl),
                             ('back', y1 + 1.5 + 1.75, y1 + 0.75, xl - SHEET['osb'] - 3.5)):
        # both rails on the stair side of their posts (graspable from the treads)
        xa_ = xn - 1.75 if yn is not None else xa_front             # rails start on the newel's / post's face
        if yn is None:
            pass                                                    # the S8 front posts carry the front rails
        else:
            m.post(gr, '%s newel post' % side, xn, yn, DECK_Z + 0.1, nose(xn) + S['handrail_h'] + 1.0)
            m.item(gr, '%s newel base' % side, [ext(xn - 1.9, xn + 1.9, yn - 1.9, yn + 1.9, DECK_Z, DECK_Z + 0.1)],
                   'post base 4x4', 'galvanized', mass=0.5)
        for hh, nm in ((S['handrail_h'], 'handrail'), (S['handrail_h'] / 2, 'mid rail')):
            m.lumber(gr, '%s %s' % (side, nm), '2x4', [xa_, y, nose(xa_) + hh - 1.75], [xb_, y, nose(xb_) + hh - 1.75],
                     up=Z, ends='angled %.1f deg' % math.degrees(th))
    pl = LUMBER['4x4'][0]
    xs0 = P.X['stairs'][0]
    m.sheet(gr, 'stair side skirt', 'osb', ext(xs0, xl, -pl - SHEET['osb'], -pl, DECK_Z, DECK_Z + h * nr),
            dims=(xl - xs0, h * nr))                                # closes the stair front side (spec p.2)
    m.screws(gr, 'carriage bolt 1/2 x 6 in', 4, 'newel posts through the first stringer / a deck cleat')
    m.screws(gr, 'deck screw 3 in', 16, 'rails to newels and posts, 2 per crossing')
    m.note(gr, 'Front rails screw to the newel and to the inside of the lane posts at x = 290 and 338; the back '
              'rails run from the back newel to the landing edge guard post.')


def landing(m):
    L = P.LANDING
    x0, x1 = P.X['landing']
    W = P.LANE['width']
    tt, ww = LUMBER['2x8']
    top = DECK_Z + L['h']
    jt = top - SHEET['ply']                                          # frame top
    g = '10 Landing/frame'
    beams = (x0, (x0 + x1) / 2 - tt, x1 - 2 * tt)                   # doubled 2x8 beams along Y (x of first ply)
    for k, xb in enumerate(beams):
        for j in range(2):
            xc = xb + tt * (j + 0.5)
            m.lumber(g, 'beam %d ply %d' % (k + 1, j + 1), '2x8', [xc, 0, jt - ww / 2], [xc, W, jt - ww / 2], up=Z)
        for y in (1.75, W / 2, W - 1.75):
            xp = (x0 + 1.75) if k == 0 else (x1 - 1.75) if k == 2 else xb + tt   # flush with the landing edges
            m.post(g, 'post %d-%d' % (k + 1, int(y)), xp, y, DECK_Z + 0.1, jt - ww)
            m.item(g, 'post base', [ext(xp - 1.9, xp + 1.9, y - 1.9, y + 1.9, DECK_Z, DECK_Z + 0.1)],
                   'post base 4x4', 'galvanized', mass=0.5)
    oc = L['oc']
    ys = [tt / 2] + [oc * j for j in range(1, int(W / oc))] + [W - tt / 2]
    for bay in range(2):
        xa = beams[bay] + 2 * tt
        xb = beams[bay + 1]
        for k, y in enumerate(ys):
            m.lumber(g, 'joist bay %d-%d' % (bay + 1, k + 1), '2x8', [xa, y, jt - ww / 2], [xb, y, jt - ww / 2], up=Z)
    m.screws(g, 'joist hanger 2x8 + nails', 2 * 2 * len(ys), 'both ends of every joist')
    m.screws(g, 'carriage bolt 1/2 x 6 in', 2 * 3 * 3, 'beam plies through-bolted, 3 per beam; posts toe-screwed')
    gs = '10 Landing/deck sheets'
    for k in range(int(round((x1 - x0) / 48.0))):
        m.sheet(gs, 'landing sheet %d' % (k + 1), 'ply', ext(x0 + 48 * k, x0 + 48 * (k + 1), 0, W, jt, top))
    # skirt on the face towards the stairs/lane (behind the stair) so feet can't go under the landing
    ys0 = P.STAIRS['y0'] + 1.5 + P.STAIRS['width']
    m.sheet(gs, 'front skirt', 'osb', ext(x0 - SHEET['osb'], x0, ys0, W, DECK_Z, jt), dims=(W - ys0, jt - DECK_Z))
    so = SHEET['osb']
    pl = LUMBER['4x4'][0]                                            # outside the lane / guard posts
    for nm, sh, dims in (('lane-side skirt front', ext(x0, x1 + pl, -pl - so, -pl, DECK_Z, top), (x1 - x0, top - DECK_Z)),
                         ('lane-side skirt back', ext(x0, x1 + pl, W + pl, W + pl + so, DECK_Z, top),
                          (x1 - x0, top - DECK_Z)),
                         ('end skirt', ext(x1, x1 + so, -pl, W + pl, DECK_Z, top), (W, top - DECK_Z))):
        m.sheet(gs, nm, 'osb', sh, dims=dims)


# ---------------------------------------------------------------- location buttons (1-5) on caps / guard tops

def location_buttons(m):
    """Five location buttons (spec p.3, circled): a 100 mm dome button in a project box on a short 2x6 pad that
    sits on the top rail and the post top (or the landing guard top rail)."""
    pt = LUMBER['4x4'][0]
    t2 = LUMBER['2x4'][0]
    W = P.LANE['width']
    lane_top = DECK_Z + P.RAIL['height']
    land_top = DECK_Z + P.LANDING['h'] + P.LANDING['guard_h']
    yf, yb = -LUMBER['2x6'][1] / 2, W + LUMBER['2x6'][1] / 2         # over post + rail, flush with the lane
    spots = [('1', 118.0, yf, lane_top), ('2', 118.0, yb, lane_top), ('3', 228.0, yb, lane_top),
             ('4', 160.0, yf, lane_top), ('5', 410.0, yb, land_top)]
    for nm, x, y, top in spots:
        g = '11 Location buttons/location %s' % nm
        m.lumber(g, 'pad', '2x6', [x - 6.0, y, top + 0.75], [x + 6.0, y, top + 0.75], up=Z, flat=True)
        z = top + 1.5
        m.item(g, 'box', [ext(x - 2.25, x + 2.25, y - 2.25, y + 2.25, z, z + 2.5)], 'project box', 'grey plastic',
               mass=0.3)
        button(m, g + ' (moving)', [x, y, z + 2.5], [0, 0, 1], 'location button %s' % nm)


# ---------------------------------------------------------------- demonstration counter

def counter(m):
    C = P.COUNTER
    W = P.LANE['width']
    yb = W - 4.0                                                     # cabinet backs (room for the stand legs)
    yf = yb - C['depth']
    z0 = DECK_Z
    x = C['x0']
    pb = 0.75
    for pos, code, w, kind in C['bases']:
        g = '12 Counter/%s %s %s (buy)' % (pos, code, kind)
        sides = [ext(x, x + pb, yf, yb, z0, z0 + C['base_h']), ext(x + w - pb, x + w, yf, yb, z0, z0 + C['base_h'])]
        toe = [ext(x - 1, x + w + 1, yf - 1, yf + 3.0, z0 - 1, z0 + C['toe'])]
        m.item(g, 'sides', sides, 'base cabinet ' + code, 'birch', mass=None, mat='particleboard', cuts=toe)
        m.item(g, 'bottom', [ext(x + pb, x + w - pb, yf + 3.0, yb - 0.25, z0 + C['toe'], z0 + C['toe'] + pb)],
               'base cabinet ' + code, 'birch', mat='particleboard')
        m.item(g, 'back', [ext(x + pb, x + w - pb, yb - 0.25, yb, z0 + C['toe'], z0 + C['base_h'])],
               'base cabinet ' + code, 'birch', mat='hardboard')
        m.item(g, 'toe kick', [ext(x + pb, x + w - pb, yf + 3.0, yf + 3.75, z0, z0 + C['toe'])],
               'base cabinet ' + code, 'birch', mat='particleboard')
        m.item(g, 'face frame', [ext(x, x + w, yf - 0.75, yf, z0 + C['base_h'] - 1.5, z0 + C['base_h']),
                                 ext(x, x + 1.5, yf - 0.75, yf, z0 + C['toe'], z0 + C['base_h']),
                                 ext(x + w - 1.5, x + w, yf - 0.75, yf, z0 + C['toe'], z0 + C['base_h'])],
               'base cabinet ' + code, 'birch', mat='hardwood')
        zb, zt = z0 + C['toe'] + pb, z0 + C['base_h'] - 1.5
        if kind == 'doors':
            m.item(g, 'shelf', [ext(x + pb, x + w - pb, yf + 1.0, yb - 0.25, (zb + zt) / 2 - 0.375, (zb + zt) / 2 + 0.375)],
                   'base cabinet ' + code, 'birch', mat='particleboard')
            for j, (xa, xh, sgn) in enumerate(((x + 0.25, x + 0.25, 1), (x + w / 2 + 0.125, x + w - 0.25, -1))):
                part = '%s %s door %d' % (pos, code, j + 1)
                gd = '12 Counter/%s %s door %d (moving)' % (pos, code, j + 1)
                dw = w / 2 - 0.375
                m.item(gd, 'door', [ext(xa, xa + dw, yf - 1.5, yf - 0.75, zb - 0.5, zt + 1.0)], 'base cabinet ' + code,
                       'birch', mat='hardwood', part=part)
                hx = xa + dw * (0.85 if sgn > 0 else 0.15)
                m.item(gd, 'pull', [ext(hx - 0.2, hx + 0.2, yf - 2.6, yf - 1.5, zt - 4.5, zt - 0.5)], 'cabinet pull',
                       'satin nickel', part=part)
                m.joints.append(dict(name=part, type='hinge', part=part, anchor=[xh, yf - 0.75, zb], axis=[0, 0, -sgn],
                                     range=(0.0, 110.0), note='concealed hinges, soft close'))
        else:
            heights = (6.0, 9.0, zt - zb - 15.0 - 0.25)
            zc = zt + 1.0
            for j, hh in enumerate(heights):
                part = '%s %s drawer %d' % (pos, code, j + 1)
                gd = '12 Counter/%s %s drawer %d (moving)' % (pos, code, j + 1)
                za = zc - hh
                m.item(gd, 'front', [ext(x + 0.25, x + w - 0.25, yf - 1.5, yf - 0.75, za + 0.125, zc - 0.125)],
                       'base cabinet ' + code, 'birch', mat='hardwood', part=part)
                bx = [ext(x + 1.5, x + w - 1.5, yf - 0.75, yb - 1.0, za + 1.0, za + 1.5),
                      ext(x + 1.5, x + 2.0, yf - 0.75, yb - 1.0, za + 1.0, zc - 1.5),
                      ext(x + w - 2.0, x + w - 1.5, yf - 0.75, yb - 1.0, za + 1.0, zc - 1.5),
                      ext(x + 1.5, x + w - 1.5, yb - 1.5, yb - 1.0, za + 1.0, zc - 1.5)]
                m.item(gd, 'box', bx, 'base cabinet ' + code, 'birch', mat='plywood', part=part)
                m.item(gd, 'pull', [ext(x + w / 2 - 3, x + w / 2 + 3, yf - 2.6, yf - 1.5, zc - 2.0, zc - 1.2)],
                       'cabinet pull', 'satin nickel', part=part)
                m.joints.append(dict(name=part, type='slide', part=part, anchor=[x + w / 2, yf, za], axis=[0, -1, 0],
                                     range=(0.0, 20.0), note='full-extension side-mount slides'))
                zc = za
        x += w
    g = '12 Counter/countertop (buy)'
    ct = C['top']
    m.item(g, 'laminate countertop 8 ft', [ext(C['x0'], C['x0'] + ct[0], yb - ct[1], yb, z0 + C['base_h'],
                                                z0 + C['base_h'] + ct[2])], 'countertop 8 ft', 'countertop', mass=30.0)
    # upper cabinet on a 4x4 stand behind the counter
    code, uw, uh, ud = C['upper']
    xc = C['x0'] + C['bases'][0][2] + C['bases'][1][2] / 2          # over the drawer base
    g = '12 Counter/upper stand'
    yl = W - 2.0
    zt_counter = z0 + C['base_h'] + ct[2]
    zu0 = zt_counter + C['upper_bottom_h']
    for k, xl in enumerate((xc - uw / 2 + 1.75, xc + uw / 2 - 1.75)):
        m.post(g, 'leg %d' % (k + 1), xl, yl, DECK_Z + 0.1, zu0 + uh)
        m.item(g, 'leg %d post base' % (k + 1), [ext(xl - 1.9, xl + 1.9, yl - 1.9, yl + 1.9, DECK_Z, DECK_Z + 0.1)],
               'post base 4x4', 'galvanized', mass=0.5)
    tie_blocks(m, g, 'stand leg 2', [xc + uw / 2 - 1.75, yl + 1.75], [xc + uw / 2 - 1.75, W + LUMBER['4x4'][0]],
               (10.25, 22.25), 'leg 2 ties to the back rails; leg 1 bolts to the lane post beside it (2 carriage bolts)')
    m.lumber(g, 'cross rail', '2x4', [xc - uw / 2, yl - 1.75 - 0.75, zu0 - 1.75], [xc + uw / 2, yl - 1.75 - 0.75, zu0 - 1.75],
             up=Z)
    for s in (-1, 1):
        yb = yl - 1.75 - 0.75                                        # on the leg's front face
        a = [xc + s * (uw / 2 - 1.75), yb, zu0 - 7.0]                # below the cross rail
        b = [xc + s * (uw / 2 + 16.0), yb, zt_counter + 1.0]        # foot on the back edge of the countertop
        m.lumber(g, 'brace %s' % ('left' if s < 0 else 'right'), '2x4', a, b, up=Z, ends='miter both')
    g = '12 Counter/%s upper cabinet (buy)' % code
    yb2 = yl - 1.75
    yf2 = yb2 - ud
    xa = xc - uw / 2
    car = [ext(xa, xa + pb, yf2, yb2, zu0, zu0 + uh), ext(xa + uw - pb, xa + uw, yf2, yb2, zu0, zu0 + uh),
           ext(xa, xa + uw, yf2, yb2, zu0, zu0 + pb), ext(xa, xa + uw, yf2, yb2, zu0 + uh - pb, zu0 + uh),
           ext(xa, xa + uw, yb2 - 0.25, yb2, zu0, zu0 + uh)]
    m.item(g, 'carcass', car, 'wall cabinet ' + code, 'birch', mat='particleboard', mass=14.0)
    m.item(g, 'shelf', [ext(xa + pb, xa + uw - pb, yf2 + 0.5, yb2 - 0.25, zu0 + 6.0, zu0 + 6.75)], 'wall cabinet ' + code,
           'birch', mat='particleboard')
    for j, s in enumerate((-1, 1)):
        part = '%s door %d' % (code, j + 1)
        gd = '12 Counter/%s door %d (moving)' % (code, j + 1)
        hx = xa if s < 0 else xa + uw
        xd0, xd1 = (hx, hx + 0.75) if s < 0 else (hx - 0.75, hx)     # doors shown open 90 deg, flat to the sides
        m.item(gd, 'door (shown open)', [ext(xd0 + s * 0.75, xd1 + s * 0.75, yf2 - uw / 2 + 0.25, yf2, zu0, zu0 + uh)],
               'wall cabinet ' + code, 'birch', mat='hardwood', part=part)
        m.joints.append(dict(name=part, type='hinge', part=part, anchor=[hx, yf2, zu0], axis=[0, 0, 1 if s > 0 else -1],
                             range=(0.0, 110.0), note='shown open'))
    # drill targets on the back panel (5, quincunx)
    gt = '12 Counter/drill targets'
    tc = [xc, yb2 - 0.25, zu0 + 6.75 + (uh - 6.75) / 2]
    for k, (du, dv) in enumerate(((0, 0), (-4, 4), (4, 4), (-4, -4), (4, -4))):
        p = [tc[0] + du, tc[1], tc[2] + dv]
        m.item(gt, 'target %d ring' % (k + 1), [('cyl', p, add(p, [0, -0.12, 0]), 1.9, 1.9)], 'target sticker',
               'black plastic')
        m.item(gt, 'target %d centre' % (k + 1), [('cyl', p, add(p, [0, -0.14, 0]), 0.8, 0.8)], 'target sticker',
               'white plastic')


def counter_objects(m):
    """Totes, milk crates, green cylinders, drill, the bulb box, items for the demonstration tasks."""
    C = P.COUNTER
    W = P.LANE['width']
    yb = W - 4.0
    zc = DECK_Z + C['base_h'] + C['top'][2]
    ymid = yb - C['top'][1] / 2
    x0 = C['x0']
    for k, xc in enumerate((x0 + 12.25, x0 + 96 - 12.25)):
        tote(m, '13 Objects/clear tote %d (free)' % (k + 1), [xc, ymid, zc], 'clear tote %d' % (k + 1))
    zu_top = zc + C['upper_bottom_h'] + C['upper'][2]
    tote(m, '13 Objects/clear tote 3 (free)', [x0 + 48.0, W - 2.0 - 1.75 - C['upper'][3] / 2 + 1.0, zu_top], 'clear tote 3')
    for k, xc in enumerate((x0 + 31.0, x0 + 65.0)):
        crate(m, '13 Objects/milk crate %d (free)' % (k + 1), [xc, ymid, zc], 'milk crate %d' % (k + 1))
    for k, (dx, dy) in enumerate(((-3, -2), (0, -4), (3, -2), (-1.5, 1.5), (1.5, 1.5))):
        p = [x0 + 48.0 + dx, ymid - 5 + dy, zc]
        m.printed('13 Objects/green cylinder %d (free)' % (k + 1), 'cylinder', [('cyl', p, add(p, [0, 0, 2.5]), 1.0, 1.0)],
                  look='print green', part='green cylinder %d' % (k + 1), buy='printed cylinder')
    # grey tote inside the left base cabinet, drill in the right base cabinet
    yf = yb - C['depth']
    tote(m, '13 Objects/grey tote (free)', [x0 + 15.0, yf + 11.0, DECK_Z + C['toe'] + 0.75], 'grey tote', small=True)
    zb_, zt_ = DECK_Z + C['toe'] + 0.75, DECK_Z + C['base_h'] - 1.5
    drill(m, '13 Objects/drill (free)', [x0 + 76.0, yf + 10.0, (zb_ + zt_) / 2 + 0.375 + 9.0], 'drill')   # on the shelf
    # bulb box at the lightbulb board: loose bulbs to pick and screw in
    T = P.TASK['lightbulb']
    bx, by = 134.0, 90.0                                              # beside the lower bulb board, behind stepover 2
    g = '13 Objects/bulb box (free)'
    m.item(g, 'box', [ext(bx - 6, bx + 6, by - 4, by + 4, DECK_Z, DECK_Z + 0.25),
                      ext(bx - 6, bx + 6, by - 4, by - 3.75, DECK_Z, DECK_Z + 4),
                      ext(bx - 6, bx + 6, by + 3.75, by + 4, DECK_Z, DECK_Z + 4),
                      ext(bx - 6, bx - 5.75, by - 4, by + 4, DECK_Z, DECK_Z + 4),
                      ext(bx + 5.75, bx + 6, by - 4, by + 4, DECK_Z, DECK_Z + 4)], 'small bin', 'grey plastic',
           mass=0.3, part='bulb box')
    for k in range(4):
        b0 = [bx - 4.5 + k * 3.0, by, DECK_Z + 1.5]
        bulb_loose(m, '13 Objects/loose bulb %d (free)' % (k + 1), b0, 'loose bulb %d' % (k + 1))


def bulb_loose(m, g, b0, part):
    n = [0, 1, 0.0]
    thread = ('cyl', add(b0, [0, -2.5, 0]), add(b0, [0, -1.5, 0]), 0.51, 0.51)
    neck = ('cyl', add(b0, [0, -1.5, 0]), add(b0, [0, -0.4, 0]), 0.6, 1.05)
    globe = ('sphere', add(b0, [0, 0.6, 0]), 1.18)
    m.printed(g, 'bulb (print)', [thread, neck, globe], look='bulb', part=part, buy='printed bulb')


def tote(m, g, base_c, part, small=False):
    L, Wd, H = (14.0, 10.0, 8.0) if small else (23.5, 16.0, 12.5)
    t = 0.12
    x, y, z = base_c
    sh = [ext(x - L / 2, x + L / 2, y - Wd / 2, y + Wd / 2, z, z + t),
          ext(x - L / 2, x + L / 2, y - Wd / 2, y - Wd / 2 + t, z, z + H),
          ext(x - L / 2, x + L / 2, y + Wd / 2 - t, y + Wd / 2, z, z + H),
          ext(x - L / 2, x - L / 2 + t, y - Wd / 2, y + Wd / 2, z, z + H),
          ext(x + L / 2 - t, x + L / 2, y - Wd / 2, y + Wd / 2, z, z + H),
          ext(x - L / 2 - 0.4, x + L / 2 + 0.4, y - Wd / 2 - 0.4, y + Wd / 2 + 0.4, z + H, z + H + 0.6)]
    m.item(g, 'tote', sh, 'grey tote' if small else 'clear latching tote', 'grey plastic' if small else 'clear plastic',
           mass=0.6 if small else 1.4, part=part)


def crate(m, g, base_c, part):
    L, Wd, H = 13.0, 13.0, 11.0
    t = 0.25
    x, y, z = base_c
    sh = [ext(x - L / 2, x + L / 2, y - Wd / 2, y + Wd / 2, z, z + t),
          ext(x - L / 2, x + L / 2, y - Wd / 2, y - Wd / 2 + t, z, z + H),
          ext(x - L / 2, x + L / 2, y + Wd / 2 - t, y + Wd / 2, z, z + H),
          ext(x - L / 2, x - L / 2 + t, y - Wd / 2, y + Wd / 2, z, z + H),
          ext(x + L / 2 - t, x + L / 2, y - Wd / 2, y + Wd / 2, z, z + H)]
    cuts = [ext(x - 3, x + 3, y - Wd, y + Wd, z + H - 3.0, z + H - 1.5),     # hand holes on two sides
            ext(x - L, x + L, y - 3, y + 3, z + H - 3.0, z + H - 1.5)]
    m.item(g, 'milk crate', sh, 'milk crate', 'crate blue', mass=1.6, part=part, cuts=cuts)


def drill(m, g, c, part):
    x, y, z = c
    sh = [ext(x - 3.5, x + 3.5, y - 1.3, y + 1.3, z, z + 2.8),                 # motor body (horizontal, along -Y? keep X)
          ext(x - 1.0, x + 1.0, y - 1.1, y + 1.1, z - 6.0, z),                 # handle
          ext(x - 2.4, x + 2.4, y - 1.9, y + 1.9, z - 9.0, z - 6.0),           # battery
          ('cyl', [x - 3.5, y, z + 1.4], [x - 5.2, y, z + 1.4], 0.75, 0.6),    # chuck
          ('cyl', [x - 5.2, y, z + 1.4], [x - 8.2, y, z + 1.4], 0.12, 0.12)]   # 1/4 in bit
    m.item(g, 'cordless drill', sh, 'cordless drill', 'drill yellow', mass=1.5, part=part)


# ---------------------------------------------------------------- gantry

def gantry(m):
    """Mobile A-frame gantry (spec p.1): two side frames (4x4 base beam on two locking casters, 4x4 post, two 2x4
    A-braces), a doubled 2x10 crossbeam across the lane with 2x4 knee braces, a beam trolley with a hook for the
    robot's safety rope. Base beams run outside the lane rails, so it rolls along the whole lane; parked at the
    start. NOT a certified fall-arrest anchor: see the guide."""
    G = P.GANTRY
    W = P.LANE['width']
    s4 = LUMBER['4x4'][0]
    tt, ww = LUMBER['2x10']
    t2, w4 = LUMBER['2x4']
    x0 = G['x']
    rail_out = LUMBER['4x4'][0] + LUMBER['2x4'][0]
    zb = G['caster_h']                                               # underside of the base beams
    ztop = G['clear_h'] + ww                                         # top of the crossbeam
    for side, y in (('front', -rail_out - G['side_gap']), ('back', W + rail_out + G['side_gap'])):
        g = '14 Gantry/%s frame' % side
        L = G['base_len']
        m.lumber(g, 'base beam', '4x4', [x0 - L / 2, y, zb + s4 / 2], [x0 + L / 2, y, zb + s4 / 2], up=Z)
        for k, xc in enumerate((x0 - L / 2 + 4.0, x0 + L / 2 - 4.0)):
            m.item(g, 'caster %d' % (k + 1), [ext(xc - 2.0, xc + 2.0, y - 2.0, y + 2.0, zb - 0.25, zb),
                                              ('cyl', [xc, y - 0.6, 2.0], [xc, y + 0.6, 2.0], 2.0, 2.0)],
                   'locking caster 4 in', 'black plastic', mass=0.8)
        m.post(g, 'post', x0, y, zb + s4, ztop)
        for sgn in (-1, 1):                                          # A-braces in the frame's XZ plane
            m.lumber(g, 'A-brace %s' % ('start' if sgn < 0 else 'lane'), '2x4',
                     [x0 + sgn * (L / 2 - 6.0), y, zb + s4 + 0.5], [x0 + sgn * s4 / 2, y, zb + s4 + 30.0], up=Y,
                     ends='miter both')
        m.item(g, 'mending plates', [ext(x0 - 10.0, x0 + 10.0, y - s4 / 2 - 0.06, y - s4 / 2, zb + 0.5, zb + 3.0),
                                     ext(x0 - 10.0, x0 + 10.0, y + s4 / 2, y + s4 / 2 + 0.06, zb + 0.5, zb + 3.0)],
               'mending plate', 'galvanized', mass=0.3)
        # knee brace post -> crossbeam (YZ plane), inboard
        sy = 1 if side == 'front' else -1
        m.lumber(g, 'knee brace', '2x4', [x0, y + sy * s4 / 2, ztop - ww - 24.0], [x0, y + sy * (s4 / 2 + 24.0), ztop - ww],
                 up=X, ends='45 deg both')
        m.screws(g, 'carriage bolt 1/2 x 6 in', 2, 'post to base beam through a 2x4 cleat each side')
        m.screws(g, 'lag screw 3/8 x 4 in', 4 + 8, 'A-braces and knee brace (2 per end); casters (4 per caster)')
    g = '14 Gantry/crossbeam'
    yf, yb = -rail_out - G['side_gap'], W + rail_out + G['side_gap']
    for j, dx in enumerate((-(s4 + tt) / 2, (s4 + tt) / 2)):         # doubled 2x10 sandwiching the post tops
        m.lumber(g, 'crossbeam ply %d' % (j + 1), '2x10', [x0 + dx, yf - s4 / 2, ztop - ww / 2],
                 [x0 + dx, yb + s4 / 2, ztop - ww / 2], up=Z)
    m.screws(g, 'carriage bolt 1/2 x 8 in', 4, 'crossbeam plies through each post top, 2 bolts per post')
    m.item(g, 'beam trolley', [ext(x0 - 3.0, x0 + 3.0, W / 2 - 2.0, W / 2 + 2.0, ztop - ww - 3.0, ztop - ww)],
           'beam trolley', 'galvanized', mass=3.0, part='trolley')
    m.item(g, 'chain and hook', [('cyl', [x0, W / 2, ztop - ww - 3.0], [x0, W / 2, ztop - ww - 30.0], 0.25, 0.25),
                                 ('cyl', [x0, W / 2 - 0.6, ztop - ww - 30.0], [x0, W / 2 + 0.6, ztop - ww - 30.0], 0.9, 0.9)],
           'safety rope', 'black plastic', mass=1.0, part='hook')
    m.note(g, 'The trolley holds the robot safety rope. Lumber is fine for steadying a robot; a fall arrest load '
              '(dynamic factor 2-3 on 120 kg) is checked in cad/tools/beam.py. Lock all four casters before a trial.')


# ---------------------------------------------------------------- build all

STAGES = [('deck', deck), ('rails', rails), ('door', door_wall), ('tasks', None), ('mobility', None),
          ('stairs', None), ('buttons', location_buttons), ('counter', counter), ('objects', counter_objects),
          ('gantry', gantry)]


def tasks(m):
    for kind, T in P.TASK.items():
        task_board(m, kind, T['x'], T['y'], T['face'], T['mount'], T.get('tilt'))


def mobility(m):
    stepovers(m)
    ramps(m)
    wall_obstacle(m)


def stairs_landing(m):
    stairs(m)
    landing(m)


STAGES = [('deck', deck), ('rails', rails), ('door', door_wall), ('tasks', tasks), ('mobility', mobility),
          ('stairs', stairs_landing), ('buttons', location_buttons), ('counter', counter),
          ('objects', counter_objects), ('gantry', gantry)]


def build(stages=None, poses=None):
    m = Model()
    for name, fn in STAGES:
        if stages is None or name in stages:
            fn(m)
    stain(m)
    if poses:
        m.pose(poses)
    return m


def stain(m):
    """Railings are stained dark brown (spec pictures): exterior semi-transparent stain on the lumber."""
    for p in m.pieces:
        if p.kind == 'lumber' and p.group.startswith(P.RAIL['stain_groups']):
            p.look = 'stain cedar 2'


def stage_prefixes(stage):
    m = Model()
    dict(STAGES)[stage](m)
    return sorted({p.group.split('/')[0] for p in m.pieces})


if __name__ == '__main__':
    m = build()
    from collections import Counter
    print(len(m.pieces), 'pieces,', len(m.joints), 'joints')
    print(Counter(p.kind for p in m.pieces))
    print(Counter(p.group.split('/')[0] for p in m.pieces))
