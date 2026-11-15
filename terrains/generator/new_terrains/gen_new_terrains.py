"""Preview of the proposed extra training terrains (docs/TODO_extended_training.md, To do 1). Not used by any training run.

Each generator takes a difficulty in [0, 1] and a seed and returns (ground meshes, obstacle meshes, targets) for one 8 x 8 m tile,
in metres, tile centre at the origin, robot spawn at the centre facing +x. Obstacles are separate meshes so that, in Isaac Lab, they can
carry their own physics material (side-face friction) and a filtered contact sensor (leg-on-obstacle vs leg-on-ground).
The same functions are meant to become Isaac Lab MeshTerrain generators once the user signs off on the look.
"""
import math

import numpy as np
import trimesh

TILE = 8.0
FOUR_BY_FOUR = 0.089  # 3.5 in


def lerp(a, b, d):
    return a + (b - a) * d


def ground():
    g = trimesh.creation.box(extents=(TILE, TILE, 0.1))
    g.apply_translation((0, 0, -0.05))
    return g


def beam(cx, cy, length, size, yaw, raise_one_end=0.0, height=None):
    """A square beam lying on the ground (or with one end raised: a slanted 4x4)."""
    h = size if height is None else height
    b = trimesh.creation.box(extents=(length, size, h))
    pitch = math.atan2(raise_one_end, length) if raise_one_end else 0.0
    b.apply_translation((0, 0, h / 2))
    if pitch:
        b.apply_transform(trimesh.transformations.rotation_matrix(-pitch, (0, 1, 0), point=(-length / 2, 0, 0)))
    b.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
    b.apply_translation((cx, cy, 0))
    parts = [b]
    if raise_one_end:  # the block it rests on
        blk = trimesh.creation.box(extents=(0.25, 0.25, raise_one_end))
        blk.apply_translation((length / 2 * math.cos(yaw) + cx, length / 2 * math.sin(yaw) + cy, raise_one_end / 2))
        parts.append(blk)
    return parts


def wedge(cx, cy, length, width, slope_deg, yaw, ridge=False):
    """A ramp (one face) or a gable (two faces to a ridge), width across, slope along length."""
    h = (length / (2 if ridge else 1)) * math.tan(math.radians(slope_deg))
    if ridge:
        prof = [(-length / 2, 0), (0, h), (length / 2, 0)]
    else:
        prof = [(-length / 2, 0), (length / 2, h), (length / 2, 0)]
    v, f = [], []
    for y in (-width / 2, width / 2):
        for (x, z) in prof:
            v.append((x, y, z))
    n = len(prof)
    f += [(0, 1, 2)[::-1], (n, n + 1, n + 2)]
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i)]
    m = trimesh.Trimesh(v, f, process=True)
    m.fix_normals()
    m.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
    m.apply_translation((cx, cy, 0))
    return m


def block(cx, cy, sx, sy, h, yaw):
    b = trimesh.creation.box(extents=(sx, sy, h))
    b.apply_translation((0, 0, h / 2))
    b.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
    b.apply_translation((cx, cy, 0))
    return b


def wall(x0, y0, x1, y1, h=1.0, t=0.08):
    L = math.hypot(x1 - x0, y1 - y0)
    w = trimesh.creation.box(extents=(L, t, h))
    w.apply_translation((0, 0, h / 2))
    w.apply_transform(trimesh.transformations.rotation_matrix(math.atan2(y1 - y0, x1 - x0), (0, 0, 1)))
    w.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, 0))
    return w


def ring_points(rng, n, r0=1.2, r1=3.4):
    """Obstacle centres around the spawn, so every target direction crosses some."""
    a = rng.uniform(0, 2 * math.pi, n)
    r = rng.uniform(r0, r1, n)
    return np.c_[r * np.cos(a), r * np.sin(a)], a


# ---------------------------------------------------------------- the five types
def single_beams(d, seed):
    """A few separate 4x4-like beams: size 5 -> 15 cm, crossing angle up to 60 deg, some slanted (one end raised up to 20 cm)."""
    rng = np.random.default_rng(seed)
    obs = []
    pts, ang = ring_points(rng, 6)
    for (x, y), a in zip(pts, ang):
        size = lerp(0.05, 0.15, d) * rng.uniform(0.85, 1.0)
        cross = math.radians(rng.uniform(0, lerp(15, 60, d)))  # angle away from square to the path out of the centre
        yaw = a + math.pi / 2 + cross * rng.choice((-1, 1))
        slant = lerp(0.0, 0.20, d) * rng.uniform(0.5, 1.0) if rng.random() < lerp(0.0, 0.5, d) else 0.0
        obs += beam(x, y, rng.uniform(1.2, 2.4), size, yaw, raise_one_end=slant)
    for _ in range(rng.integers(0, 3) if rng.random() < lerp(0.3, 0.8, d) else 0):  # apparatus-style brace stands
        bx, by = ring_points(rng, 1, 1.2, 3.4)[0][0]
        obs += brace_stand(bx, by, rng.uniform(0, math.pi), rng)
    ang_ = np.random.default_rng(seed + 7).uniform(0, 2 * math.pi)
    turn_ = ang_ + np.random.default_rng(seed + 8).choice((-1, 1)) * np.random.default_rng(seed + 9).uniform(0.6, 1.8)
    single_beams.last_routes = [((-3.6 * math.cos(ang_), -3.6 * math.sin(ang_), ang_), (0.0, 0.0), (3.4 * math.cos(turn_), 3.4 * math.sin(turn_)))]
    return [ground()], obs, []


def beam_fields(d, seed):
    """Several beams at mixed angles; spacing 1.2 -> 0.5 m; includes V pairs like the apparatus stepovers."""
    rng = np.random.default_rng(seed)
    obs = []
    spacing = lerp(1.2, 0.5, d)
    n_rows = int(6.4 / spacing)
    for i in range(n_rows):
        x = -3.2 + (i + 0.5) * spacing
        if abs(x) < 0.35:
            continue  # clear spawn
        size = lerp(0.05, 0.12, d) * rng.uniform(0.8, 1.0)
        if rng.random() < 0.35:  # a V pair, ~8 deg each side, like the apparatus
            for s in (-1, 1):
                obs += beam(x, s * 1.6, 2.6, size, math.pi / 2 + s * math.radians(rng.uniform(5, 12)))
        else:
            yaw = math.pi / 2 + math.radians(rng.uniform(-lerp(10, 50, d), lerp(10, 50, d)))
            obs += beam(x, rng.uniform(-1.5, 1.5), rng.uniform(2.5, 5.0), size, yaw,
                        raise_one_end=lerp(0, 0.15, d) * rng.random() if rng.random() < 0.3 * d else 0.0)
    if rng.random() < lerp(0.2, 0.6, d):  # a brace stand at the side of the field
        obs += brace_stand(rng.uniform(-3.0, 3.0), rng.choice((-3.2, 3.2)), rng.uniform(0, math.pi), rng)
    ang_ = np.random.default_rng(seed + 7).uniform(0, 2 * math.pi)
    turn_ = ang_ + np.random.default_rng(seed + 8).choice((-1, 1)) * np.random.default_rng(seed + 9).uniform(0.6, 1.8)
    beam_fields.last_routes = [((-3.6 * math.cos(ang_), -3.6 * math.sin(ang_), ang_), (0.0, 0.0), (3.4 * math.cos(turn_), 3.4 * math.sin(turn_)))]
    return [ground()], obs, []


def blocks(d, seed):
    """Loose blocks smaller and larger than a 4x4: 5 -> 25 cm tall, more of them as the level rises."""
    rng = np.random.default_rng(seed)
    obs = []
    n = int(lerp(10, 34, d))
    pts, _ = ring_points(rng, n, 0.7, 3.6)
    for x, y in pts:
        h = rng.uniform(0.05, lerp(0.10, 0.25, d))
        sx, sy = rng.uniform(0.08, 0.45), rng.uniform(0.08, 0.45)
        obs.append(block(x, y, sx, sy, h, rng.uniform(0, math.pi)))
    ang_ = np.random.default_rng(seed + 7).uniform(0, 2 * math.pi)
    turn_ = ang_ + np.random.default_rng(seed + 8).choice((-1, 1)) * np.random.default_rng(seed + 9).uniform(0.6, 1.8)
    blocks.last_routes = [((-3.6 * math.cos(ang_), -3.6 * math.sin(ang_), ang_), (0.0, 0.0), (3.4 * math.cos(turn_), 3.4 * math.sin(turn_)))]
    return [ground()], obs, []


def bar(p0, p1, size=FOUR_BY_FOUR):
    """A square bar (4x4 by default) between two 3-D points."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    b = trimesh.creation.box(extents=(size, size, L))
    b.apply_transform(trimesh.geometry.align_vectors((0, 0, 1), (p1 - p0) / L))
    b.apply_translation((p0 + p1) / 2)
    return b


def brace_stand(x, y, yaw, rng, board=None):
    """An upright board or post held by slanted 4x4 braces down to the floor, like the apparatus button-post A-frame:
    the braces are low slanted bars (up to ~0.55 m) that catch shins. board: True = a 0.6 x 1.2 m board on the post."""
    c, s = math.cos(yaw), math.sin(yaw)
    h = rng.uniform(1.1, 1.5)
    parts = [block(x, y, FOUR_BY_FOUR, FOUR_BY_FOUR, h, yaw)]
    if board if board is not None else rng.random() < 0.6:
        bd = trimesh.creation.box(extents=(0.02, rng.uniform(0.5, 0.9), rng.uniform(0.6, 1.0)))
        bd.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
        bd.apply_translation((x + 0.06 * c, y + 0.06 * s, h - 0.45))
        parts.append(bd)
    reach, top = rng.uniform(0.45, 0.65), rng.uniform(0.45, 0.65)
    two_ways = rng.random() < 0.5  # braces front and back, or left and right
    for sgn in (-1, 1):
        dx, dy = (c * sgn, s * sgn) if two_ways else (-s * sgn, c * sgn)
        parts.append(bar((x + dx * reach, y + dy * reach, 0.04), (x + dx * 0.05, y + dy * 0.05, top)))
    return parts


def rails_around(x0, x1, y0, y1, rng, height=None, sides="all"):
    """Apparatus-style side rails (4x4 posts every ~1.2 m, 1-2 rails) around a rectangle, open at the two ends along x."""
    from_side = []
    h = height if height is not None else rng.choice((0.30, 0.70, 1.0))
    for y in ((y0, y1) if sides in ("all", "sides") else ()):
        n = max(2, int((x1 - x0) / 1.2) + 1)
        for xx in np.linspace(x0, x1, n):
            from_side.append(block(xx, y, FOUR_BY_FOUR, FOUR_BY_FOUR, h, 0.0))
        for z in ([h - 0.03] if h < 0.5 else [h - 0.03, h / 2]):
            r = trimesh.creation.box(extents=(x1 - x0, 0.04, 0.06))
            r.apply_translation(((x0 + x1) / 2, y, z))
            from_side.append(r)
    return from_side


def wedges(d, seed):
    """Wedges and ramps, 8 -> 25 deg. Each tile has one big feature, with side rails on some:
    a crossing-ramp pinwheel (2 x 2 gables, cells 1.2 -> 2.0 m, like the apparatus), or a long ramp up to a plateau and down
    (ramps 1.5 -> 3 m); plus scattered single wedges and narrow slanted strips, and brace stands on some tiles."""
    rng = np.random.default_rng(seed)
    obs = []
    slope = lambda: rng.uniform(lerp(8, 15, d), lerp(12, 25, d))
    big = rng.integers(2)
    if big == 0:  # pinwheel of 4 gables, bigger cells
        cell = rng.uniform(1.2, 2.0)
        cx0, cy0 = 3.6 - 2 * cell, 3.6 - 2 * cell
        for k, (i, j) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            obs.append(wedge(cx0 + (i + 0.5) * cell, cy0 + (j + 0.5) * cell, cell, cell, slope(), (0, math.pi / 2, math.pi / 2, 0)[k], ridge=True))
        if rng.random() < 0.5:
            obs += rails_around(cx0, cx0 + 2 * cell, cy0 - 0.08, cy0 + 2 * cell + 0.08, rng)
        busy = (cx0 - 0.2, cy0 - 0.2)
        # routes through the main walk area: in at the open end, across the ridges, out the far end
        ya, yb = cy0 + cell * rng.uniform(0.4, 0.8), cy0 + cell * rng.uniform(1.2, 1.6)
        mx, my = cx0 + cell, cy0 + cell  # pinwheel centre
        loop_r = cell * 0.75
        loop = [(mx + loop_r * math.cos(a), my + loop_r * math.sin(a)) for a in np.linspace(math.pi, 3 * math.pi, 13)]
        routes = [((cx0 - 0.5, ya, 0.0), (cx0 + cell, ya), (cx0 + cell, yb), (cx0 + 2 * cell + 0.5, yb)),  # across x, turn, across
                  ((mx + rng.uniform(-0.4, 0.4) * cell, cy0 - 0.5, math.pi / 2), (mx, my), (mx + rng.uniform(-0.4, 0.4) * cell, cy0 + 2 * cell + 0.5)),  # across y
                  ((cx0 - 0.5, cy0 - 0.5, math.pi / 4), (cx0 + 2 * cell + 0.4, cy0 + 2 * cell + 0.4)),     # diagonal over all the ridges
                  ((loop[0][0] - 0.6, loop[0][1], 0.0),) + tuple(loop)]                                       # a loop round the centre
    else:  # long ramp up, plateau, ramp down, along x, in the upper half of the tile
        L_r, sl = rng.uniform(1.5, 3.0), slope()
        h = L_r * math.tan(math.radians(sl)) * 0.5
        w = rng.uniform(1.0, 2.0)
        yc = rng.uniform(1.2, 3.6 - w / 2)
        obs.append(wedge(-3.6 + L_r / 4, yc, L_r / 2, w, sl, 0.0))
        obs.append(block(-3.6 + L_r / 2 + 0.6, yc, 1.2, w, h, 0.0))
        obs.append(wedge(-3.6 + L_r / 2 + 1.2 + L_r / 4, yc, L_r / 2, w, sl, math.pi))
        if rng.random() < 0.6:
            obs += rails_around(-3.6, -3.6 + L_r + 1.2, yc - w / 2 - 0.06, yc + w / 2 + 0.06, rng)
        busy = (9.0, yc - w / 2 - 0.2)
        # routes through the main walk area: up the ramp between the rails, over the plateau, down the far side (and back)
        x_end = -3.6 + L_r + 1.2 + 0.5
        routes = [((-3.85, yc, 0.0), (-3.6 + L_r / 2 + 0.6, yc), (x_end, yc), (x_end, yc - w / 2 - 0.8)),   # up, plateau, down, turn out
                  ((x_end, yc - w / 2 - 0.8, math.pi / 2), (x_end, yc), (-3.85, yc))]
    # scattered narrow slanted strips (0.35 -> 0.15 m wide) and single wedges, away from the big feature
    pts, ang = ring_points(rng, 9, 1.0, 3.3)
    if rng.random() < lerp(0.3, 0.8, d):
        for _ in range(rng.integers(1, 3)):
            bx, by = rng.uniform(-3.0, 0.5), rng.uniform(-3.2, -1.0)
            obs += brace_stand(bx, by, rng.uniform(0, math.pi), rng)
    for (x, y), a in zip(pts, ang):
        if (x > busy[0] and y > busy[1]) or y < -0.9 and x < 0.8:
            continue
        kind = rng.random()
        if kind < 0.45:
            obs.append(wedge(x, y, rng.uniform(0.6, 1.4), lerp(0.35, 0.15, d), slope(), a + rng.uniform(-1.2, 1.2), ridge=rng.random() < 0.5))
        else:
            obs.append(wedge(x, y, rng.uniform(0.8, 1.6), rng.uniform(lerp(1.2, 0.6, d), 1.2), slope(), a + rng.uniform(-0.8, 0.8),
                             ridge=rng.random() < 0.4))
    targets = [(r[-1][0], r[-1][1], "route end") for r in routes]
    wedges.last_routes = routes  # (start pose x, y, yaw) -> target (x, y); a share of episodes uses these
    return [ground()], obs, targets


def tight_spots(d, seed):
    """Narrow lanes, wall gaps and pockets (1.0 -> 0.6 m), a beam inside; targets beside and behind the robot (sidestep, turn in place)."""
    rng = np.random.default_rng(seed)
    w = lerp(1.0, 0.6, d)
    obs = []
    # an L-shaped lane from the spawn: forward 2.5 m, then left 2.5 m
    obs += [wall(-0.6, -w / 2, 2.5 + w / 2, -w / 2), wall(-0.6, w / 2, 2.5 - w / 2, w / 2),
            wall(2.5 + w / 2, -w / 2, 2.5 + w / 2, 3.0), wall(2.5 - w / 2, w / 2, 2.5 - w / 2, 3.0),
            wall(-0.6, -w / 2, -0.6, w / 2)]  # closed behind: the robot must turn in place to leave by the lane
    obs += beam(1.4, 0.0, w, lerp(0.05, 0.10, d), math.pi / 2 + math.radians(rng.uniform(-20, 20)))
    # a pocket on the other side: a gap in a wall, narrower than the shoulders turned square (sidestep through)
    gap = lerp(0.75, 0.50, d)
    obs += [wall(-3.4, -1.5, -1.0 - gap / 2, -1.5), wall(-1.0 + gap / 2, -1.5, 1.0, -1.5)]
    obs += [wall(-3.4, -1.5, -3.4, -3.4), wall(-3.4, -3.4, 1.0, -3.4), wall(1.0, -3.4, 1.0, -1.5)]
    obs.append(block(-2.2, -2.5, 0.4, 0.4, lerp(0.08, 0.2, d), 0.3))
    targets = [(2.5, 2.6, "turn left"), (-0.25, 0.0, "behind: turn in place"), (-1.0, -2.6, "sidestep through the gap")]
    tight_spots.last_routes = [((-0.3, 0.0, math.pi), (2.5, 0.0), (2.5, 2.6)),                         # facing the dead end: turn, lane, left
                               ((-1.0, -0.8, math.pi / 2), (-1.0, -1.2), (-1.0, -2.0), (-2.6, -2.95))]   # facing away: turn, through the gap, pocket
    return [ground()], obs, targets


TYPES = [("single beams (5-15 cm, up to 60 deg, slanted 4x4s)", single_beams),
         ("beam fields (spacing 1.2 -> 0.5 m, V pairs)", beam_fields),
         ("blocks (5-25 cm, smaller and larger than a 4x4)", blocks),
         ("wedges (8-25 deg: ramps, ridges, narrow strips, pinwheel)", wedges),
         ("tight spots (lanes 1.0 -> 0.6 m, gaps, pockets)", tight_spots)]


def side(x0, x1, y, style, rng, height):
    """One side of a corridor along x at y, up to `height` (G1 knee 0.30 m, G1 waist 0.70 m, house wall 2.44 m = 8 ft).
    Styles: 'wall' (solid), 'rails' (4x4 posts every 1.0-1.6 m with 1-3 horizontal rails, like the apparatus lane rails),
    'posts' (posts only)."""
    parts = []
    if style == "wall":
        parts.append(wall(x0, y, x1, y, h=height, t=rng.uniform(0.05, 0.12)))
    else:
        pitch = rng.uniform(1.0, 1.6)
        n = max(2, int((x1 - x0) / pitch) + 1)
        for x in np.linspace(x0, x1, n):
            parts.append(block(x, y, FOUR_BY_FOUR, FOUR_BY_FOUR, height, 0.0))
        if style == "rails":
            zs = [height - 0.03] + list(rng.uniform(0.15, height - 0.1, rng.integers(0, 3))) if height > 0.25 else [height - 0.03]
            for z in zs:
                r = trimesh.creation.box(extents=(x1 - x0, 0.04, 0.06))
                r.apply_translation(((x0 + x1) / 2, y + 0.06 * np.sign(y), z))
                parts.append(r)
    return parts


def log_corridors(d, seed):
    """Long corridor (7.6 m) with logs across it at mixed angles. Spacing 1.4 -> 0.45 m, log height 6 -> 15 cm, angle up to 50 deg.
    Sides: solid walls, apparatus-style rails on posts, low kerbs or posts only, the same or different on each side. Every gap along the centre line is kept >= 0.30 m (one foot length plus margin),
    so a clean path with no foot on a log always exists. Spawn at one end, target at the other."""
    rng = np.random.default_rng(seed)
    width = lerp(1.5, 1.0, d)
    L = 7.6
    styles = ("wall", "rails", "posts")
    heights = {"knee": 0.30, "waist": 0.70, "house": 2.44}  # G1 knee, G1 waist, 8 ft house corridor
    if rng.random() < 0.3:  # a house corridor: solid walls to 8 ft on both sides
        sa = sb = "wall"; ha = hb = heights["house"]
    else:
        sa = styles[rng.integers(len(styles))]
        sb = sa if rng.random() < 0.5 else styles[rng.integers(len(styles))]  # same or different style on the two sides
        ha = heights[("knee", "waist")[rng.integers(2)]]
        hb = ha if rng.random() < 0.5 else heights[("knee", "waist", "house")[rng.integers(3)]]
    obs = side(-L / 2, L / 2, -width / 2, sa, rng, ha) + side(-L / 2, L / 2, width / 2, sb, rng, hb)
    x = -L / 2 + 1.0
    last_hi = -1e9
    prev_straight = False
    while x < L / 2 - 0.8:
        size = rng.uniform(lerp(0.06, 0.10, d), lerp(0.08, 0.15, d))
        if rng.random() < 0.3 and x > last_hi + 0.3 and not prev_straight:  # a straight log along the corridor, off to one side
            ln = rng.uniform(0.8, 2.0)
            # off-centre by 0.15 m up to (half width - 0.3 m): the lane on the far side stays >= half the corridor wide
            off = rng.uniform(0.15, max(0.16, width / 2 - 0.30)) * rng.choice((-1, 1))
            obs += beam(x + ln / 2, off, ln, size, 0.0)
            last_hi = x + ln
            x += ln + rng.uniform(lerp(0.6, 0.3, d), lerp(1.0, 0.5, d))
            prev_straight = True
            continue
        prev_straight = False
        ang = math.radians(rng.uniform(-lerp(10, 50, d), lerp(10, 50, d)))
        span = width / max(math.cos(ang), 0.3) + 0.4  # long enough to reach both walls
        lo = x - abs(math.tan(ang)) * width / 2 - size / 2  # extent of the log along the corridor at its two ends
        hi = x + abs(math.tan(ang)) * width / 2 + size / 2
        if lo - last_hi >= lerp(0.40, 0.26, d) or last_hi < -1e8:
            obs += beam(x, 0.0, span, size, math.pi / 2 + ang,
                        raise_one_end=lerp(0, 0.10, d) * rng.random() if rng.random() < 0.25 * d else 0.0)
            last_hi = hi
        x += rng.uniform(lerp(1.3, 0.30, d), lerp(1.7, 0.50, d))
    targets = [(L / 2 - 0.4, 0.0, "end of corridor")]
    log_corridors.last_routes = [((-L / 2 + 0.3, 0.0, 0.0), (L / 2 - 0.4, 0.0)),
                                 ((-L / 2 + 0.3, 0.0, math.pi), (L / 2 - 0.4, 0.0))]  # second: spawned facing the wrong way, turn in place first
    return [ground()], obs, targets


TYPES.append(("log corridors (7.6 m, logs 6-15 cm, clear gap 0.40 -> 0.26 m)", log_corridors))
