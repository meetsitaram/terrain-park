"""'Stairs' terrain section (user, 2026-11-11): flights with varied riser height, tread depth, stair width and step count, side walls
and rails; straight, L-shaped (90 deg turn with a rest landing, keeps climbing), U-shaped switchbacks, and rounded amphitheatre steps.
Preview only; not used by any training run.

Easy -> hard: risers 8 -> 22 cm, treads 40 -> 22 cm, width 2.0 -> 0.7 m, 3 -> 15 steps per flight, nosing 0-3 cm.
Sides per flight: open, solid walls stepping up with the stairs (knee 0.30 m, waist 0.70 m or house 2.44 m above each tread),
handrails on posts (apparatus style, 0.9 m above the nosing line), or one of each.
Each tile also returns routes: (start pose x, y, yaw), then waypoints (x, y) in order, the last one the target, from the foot of the stairs to the top or down the far side."""
import math

import numpy as np
import trimesh

from gen_new_terrains import FOUR_BY_FOUR, bar, block, ground, lerp


def oriented_box(cx, cy, cz, sx, sy, sz, yaw):
    b = trimesh.creation.box(extents=(sx, sy, sz))
    b.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
    b.apply_translation((cx, cy, cz))
    return b


def flight(x0, y0, yaw, n, rise, run, width, z0, nosing):
    """n solid steps from (x0, y0) at height z0, climbing along yaw. Returns (meshes, end x, end y, end z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    parts = []
    for i in range(n):
        h = z0 + rise * (i + 1)
        a = run * i + run / 2 - nosing / 2
        parts.append(oriented_box(x0 + c * a, y0 + s * a, h / 2, run + nosing, width, h, yaw))
    return parts, x0 + c * run * n, y0 + s * run * n, z0 + rise * n


def stepped_sides(x0, y0, yaw, n, rise, run, width, z0, style, height, rng, landing=None):
    """Side walls stepping up with each tread, or handrails following the slope, on both sides of a flight.
    landing: (length, z) adds a flat section after the last step."""
    c, s = math.cos(yaw), math.sin(yaw)
    nx, ny = -s, c
    parts = []
    for sgn in (-1, 1):
        st = style if style != "mixed" else ("wall", "rails")[sgn > 0]
        if st == "open":
            continue
        off = sgn * (width / 2 + 0.05)
        if st == "wall":
            for i in range(n):
                top = z0 + rise * (i + 1) + height
                a = run * i + run / 2
                parts.append(oriented_box(x0 + c * a + nx * off, y0 + s * a + ny * off, top / 2, run, 0.08, top, yaw))
            if landing:
                L, z = landing
                a = run * n + L / 2
                parts.append(oriented_box(x0 + c * a + nx * off, y0 + s * a + ny * off, (z + height) / 2, L, 0.08, z + height, yaw))
        else:
            length = run * n
            n_p = max(2, int(length / 1.2) + 1)
            for k in range(n_p):
                f = k / (n_p - 1)
                px, py, pz = x0 + c * length * f + nx * off, y0 + s * length * f + ny * off, z0 + rise * n * f
                parts.append(block(px, py, FOUR_BY_FOUR, FOUR_BY_FOUR, pz + 0.9, yaw))
            p0 = (x0 + nx * off, y0 + ny * off, z0 + 0.9)
            p1 = (x0 + c * length + nx * off, y0 + s * length + ny * off, z0 + rise * n + 0.9)
            parts.append(bar(p0, p1, size=0.05))
            if landing:
                L, z = landing
                p2 = (p1[0] + c * L, p1[1] + s * L, z + 0.9)
                parts += [bar(p1, p2, size=0.05), block(p2[0], p2[1], FOUR_BY_FOUR, FOUR_BY_FOUR, z + 0.9, yaw)]
    return parts


def ring_sector(r0, r1, a0, a1, h, cx=0.0, cy=0.0, n=28):
    """A solid annular sector from radius r0 to r1, angles a0..a1, from the floor to height h (one amphitheatre step)."""
    ang = np.linspace(a0, a1, n)
    inner = np.c_[cx + r0 * np.cos(ang), cy + r0 * np.sin(ang)]
    outer = np.c_[cx + r1 * np.cos(ang), cy + r1 * np.sin(ang)]
    ring = np.vstack([inner, outer[::-1]])  # polygon outline, counter-clockwise for a0 < a1
    m = len(ring)
    v = np.vstack([np.c_[ring, np.zeros(m)], np.c_[ring, np.full(m, h)]])
    f = []
    for i in range(n - 1):  # top and bottom as quads between the inner and outer arcs
        a, b, c_, d = i, i + 1, 2 * n - 2 - i, 2 * n - 1 - i
        f += [(a, b, c_), (a, c_, d), (a + m, c_ + m, b + m), (a + m, d + m, c_ + m)]
    for i in range(m):  # side walls
        j = (i + 1) % m
        f += [(i, j, j + m), (i, j + m, i + m)]
    mesh = trimesh.Trimesh(v, f, process=True)
    mesh.fix_normals()
    return mesh


def stairs(d, seed):
    rng = np.random.default_rng(seed)
    rise = rng.uniform(lerp(0.08, 0.15, d), lerp(0.12, 0.22, d))
    run = rng.uniform(lerp(0.32, 0.22, d), lerp(0.40, 0.30, d))
    width = rng.uniform(lerp(1.4, 0.7, d), lerp(2.0, 1.1, d))
    n = int(rng.integers(int(lerp(3, 7, d)), int(lerp(6, 15, d)) + 1))
    nosing = rng.uniform(0.0, 0.03)
    style = str(rng.choice(["open", "wall", "rails", "mixed"]))
    height = float(rng.choice([0.30, 0.70, 2.44]))
    kind = str(rng.choice(["straight", "L", "U", "amphitheatre"], p=[0.3, 0.25, 0.25, 0.2]))
    o, routes = [], []
    if kind == "straight":  # up, landing, down
        n = max(2, min(n, int(3.0 / run)))
        land = max(width, 0.9) + rng.uniform(0.0, 0.5)
        x0, y0 = -3.4, rng.uniform(-1.5, 1.5)
        up, ex, ey, zt = flight(x0, y0, 0.0, n, rise, run, width, 0.0, nosing)
        o += up + [block(ex + land / 2, ey, land, width, zt, 0.0)]
        o += stepped_sides(x0, y0, 0.0, n, rise, run, width, 0.0, style, height, rng, landing=(land, zt))
        x_dn = ex + land + run * n
        dn, _, _, _ = flight(x_dn, ey, math.pi, n, rise, run, width, 0.0, nosing)
        o += dn + stepped_sides(x_dn, ey, math.pi, n, rise, run, width, 0.0, style, height, rng)
        routes = [((x0 - 0.45, y0, 0.0), (ex + land / 2, ey), (min(x_dn + 0.5, 3.8), ey)),
                  ((min(x_dn + 0.5, 3.8), ey, math.pi), (ex + land / 2, ey), (x0 - 0.45, y0))]
    elif kind == "L":  # up, rest landing, turn 90 deg (left or right), up again, top platform
        n1 = max(2, min(n // 2 + 1, int(3.2 / run)))
        n2 = max(2, min(n - n1 + 2, int(3.0 / run)))
        turn = rng.choice((-1, 1))
        x0, y0 = -3.4, -2.6 * turn
        up1, ex, ey, z1 = flight(x0, y0, 0.0, n1, rise, run, width, 0.0, nosing)
        o += up1 + stepped_sides(x0, y0, 0.0, n1, rise, run, width, 0.0, style, height, rng)
        o.append(block(ex + width / 2, ey, width, width, z1, 0.0))  # square rest landing
        yaw2 = turn * math.pi / 2
        sx, sy = ex + width / 2, ey + turn * width / 2
        up2, fx, fy, z2 = flight(sx, sy, yaw2, n2, rise, run, width, z1, nosing)
        o += up2 + stepped_sides(sx, sy, yaw2, n2, rise, run, width, z1, style, height, rng, landing=(1.0, z2))
        o.append(oriented_box(fx + math.cos(yaw2) * 0.5, fy + math.sin(yaw2) * 0.5, z2 / 2, 1.0, width, z2, yaw2))  # top platform
        routes = [((x0 - 0.45, y0, 0.0), (ex + width / 2, ey), (fx + math.cos(yaw2) * 0.5, fy + math.sin(yaw2) * 0.5))]
    elif kind == "U":  # up, wide landing, turn 180 deg, up again alongside, top platform
        n1 = max(2, min(n // 2 + 1, int(4.5 / run)))
        n2 = n1
        x0, y0 = -3.0, -width / 2 - 0.05
        up1, ex, ey, z1 = flight(x0, y0, 0.0, n1, rise, run, width, 0.0, nosing)
        o += up1 + stepped_sides(x0, y0, 0.0, n1, rise, run, width, 0.0, style, height, rng)
        lw = 2 * width + 0.1
        o.append(block(ex + width / 2, 0.0, width, lw, z1, 0.0))  # landing across both flights
        sx, sy = ex, width / 2 + 0.05
        up2, fx, fy, z2 = flight(sx, sy, math.pi, n2, rise, run, width, z1, nosing)
        o += up2 + stepped_sides(sx, sy, math.pi, n2, rise, run, width, z1, style, height, rng, landing=(1.0, z2))
        o.append(block(fx - 0.5, fy, 1.0, width, z2, 0.0))
        routes = [((x0 - 0.45, y0, 0.0), (ex + width / 2, y0), (ex + width / 2, sy), (fx - 0.5, fy))]
    else:  # amphitheatre: concentric curved steps rising outward over a half or three-quarter circle, a flat stage in the middle
        n = max(3, min(n, int(3.2 / run)))
        r_stage = rng.uniform(0.8, 1.5)
        span = rng.choice((math.pi, 1.5 * math.pi))
        a0 = math.pi / 2 - span / 2 + math.pi  # open side faces -x (the spawn side)
        cx, cy = -0.5, 0.0
        for i in range(n):
            o.append(ring_sector(r_stage + run * i, r_stage + run * (i + 1), a0 - span / 2 + span / 2, a0 + span, rise * (i + 1), cx, cy))
        if rng.random() < 0.5:  # aisle rails or walls at the two ends of the seating
            for a in (a0 + span / 2 - span / 2, a0 + span):
                pass
        tier = r_stage + run * (n // 2 + 0.5)  # walk along one tier, following its curve, then down to the stage
        arc = [(cx + tier * math.cos(a), cy + tier * math.sin(a)) for a in np.linspace(a0 + 0.15, a0 + span - 0.15, 10)]
        tier_route = ((cx, cy, a0 + 0.15), arc[0]) + tuple(arc[1:]) + ((cx, cy),)
        routes = [tier_route, ((cx, cy, 0.0), (cx + (r_stage + run * n - run / 2) * math.cos(a0 + span / 2),
                                  cy + (r_stage + run * n - run / 2) * math.sin(a0 + span / 2))),
                  ((cx + (r_stage + run * n - run / 2) * math.cos(a0 + span / 2), cy + (r_stage + run * n - run / 2) * math.sin(a0 + span / 2),
                    a0 + span / 2 + math.pi), (cx, cy))]
    stairs.last_routes = routes
    info = dict(kind=kind, rise=round(rise, 3), run=round(run, 3), width=round(width, 2), steps=n, nosing=round(nosing, 3),
                sides=style if kind != "amphitheatre" else "open", side_height=height)
    targets = [(r[-1][0], r[-1][1], "route end") for r in routes]
    return [ground()], o, targets, info
