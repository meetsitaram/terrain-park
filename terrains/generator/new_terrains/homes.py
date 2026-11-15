"""'Homes' terrain section, after the user's photos (2026-11-11): patio, exterior door with threshold, hallway, kitchen clutter, laundry step.
All dimensions are estimates from the photos (not measured). Preview only; not used by any training run."""
import math

import numpy as np
import trimesh

from gen_new_terrains import block, ground, lerp, wall


def cyl(x, y, z, r, h, axis=(0, 0, 1)):
    c = trimesh.creation.cylinder(radius=r, height=h, sections=16)
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    if abs(a[2]) < 0.999:
        c.apply_transform(trimesh.geometry.align_vectors((0, 0, 1), a))
    c.apply_translation((x, y, z))
    return c


def lifted(m, z):
    m.apply_translation((0, 0, z))
    return m


def chair(x, y, yaw, seat=0.46):
    s, parts = 0.42, []
    for dx in (-s / 2 + 0.02, s / 2 - 0.02):
        for dy in (-s / 2 + 0.02, s / 2 - 0.02):
            parts.append(block(dx, dy, 0.035, 0.035, seat, 0.0))
    parts.append(lifted(block(0, 0, s, s, 0.04, 0.0), seat))
    parts.append(lifted(block(-s / 2 + 0.02, 0, 0.03, s, 0.45, 0.0), seat))
    for p in parts:
        p.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
        p.apply_translation((x, y, 0))
    return parts


def table(x0, y0, sx, sy, h=0.75):
    parts = [block(x, y, 0.05, 0.05, h, 0.0) for x in (x0 + 0.05, x0 + sx - 0.05) for y in (y0 + 0.05, y0 + sy - 0.05)]
    parts.append(lifted(block(x0 + sx / 2, y0 + sy / 2, sx, sy, 0.04, 0.0), h - 0.04))
    return parts


def wall_with_openings(x0, y0, x1, y1, openings, h=2.44, t=0.10):
    """Straight wall with door openings (centre along the wall, width) and a header above each at 2.03 m."""
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    parts, s = [], 0.0
    for a, b in sorted((c - w / 2, c + w / 2) for c, w in openings) + [(L, L)]:
        if a - s > 0.01:
            parts.append(wall(x0 + ux * s, y0 + uy * s, x0 + ux * a, y0 + uy * a, h=h, t=t))
        if b > a:
            parts.append(lifted(wall(x0 + ux * a, y0 + uy * a, x0 + ux * b, y0 + uy * b, h=h - 2.03, t=t), 2.03))
        s = b
    return parts


def door_leaf(hx, hy, width, open_deg, closed_yaw):
    """Door hinged at (hx, hy); closed it lies along closed_yaw (across the opening); open_deg swings it towards closed_yaw - 90 deg."""
    a = closed_yaw - math.radians(open_deg)
    return wall(hx, hy, hx + width * math.cos(a), hy + width * math.sin(a), h=2.0, t=0.04)


def home_rooms(d, seed):
    rng = np.random.default_rng(seed)
    o = []
    door_w = lerp(0.95, 0.76, d)   # clear doorway
    hall_w = lerp(1.15, 0.90, d)   # hallway width
    thr = lerp(0.03, 0.08, d)      # exterior threshold rise
    step = lerp(0.12, 0.21, d)     # laundry step and patio step
    n_clutter = int(lerp(3, 12, d))
    # exterior wall at x = -1 with the exterior door (y = -2) and a sliding-door opening (y = 1.6)
    o += wall_with_openings(-1.0, -4.0, -1.0, 4.0, [(2.0, door_w), (5.6, 1.5)])
    o.append(block(-1.25, -2.0, 0.55, door_w + 0.2, thr, 0.0))  # raised threshold slab
    for s in (-1, 1):  # paver kerbs either side of the approach
        o.append(block(-2.3, -2.0 + s * (door_w / 2 + 0.12), 1.8, 0.10, 0.10, 0.0))
    o.append(block(-1.25, 1.6, 0.5, 1.2, step, 0.0))  # tiled step at the sliding door
    o.append(door_leaf(-0.95 + 0.03, -2.0 + door_w / 2, door_w, rng.uniform(75, 100), -math.pi / 2))  # door open inward, against the wall side
    # patio: planter kerb, loose bricks, a chair
    o += [block(-2.3, 0.2, 1.1, 0.08, 0.12, 0.0), block(-2.85, -0.35, 0.08, 1.1, 0.12, 0.0)]
    for _ in range(int(lerp(3, 8, d))):
        bx, by, yw = rng.uniform(-3.6, -1.6), rng.uniform(-1.2, 3.4), rng.uniform(0, math.pi)
        o.append(block(bx, by, 0.20, 0.06, 0.10, yw) if rng.random() < 0.3 else block(bx, by, 0.20, 0.10, 0.06, yw))
    o += chair(-3.0, 2.6, rng.uniform(0, 6.28))
    # hallway along +x with side doorways, a closet at the end, maybe a plant pot
    yh0, yh1 = -2.0 - hall_w / 2, -2.0 + hall_w / 2
    o += wall_with_openings(-0.95, yh0, 4.0, yh0, [(3.15 + 0.95, 0.8)])            # laundry doorway at x = 3.15
    o += wall_with_openings(-0.95, yh1, 4.0, yh1, [(1.3 + 0.95, 0.8), (2.65 + 0.95, 0.8)])  # kitchen doorways at x = 1.3, 2.65
    o.append(block(3.9, -2.0, 0.2, hall_w, 2.0, 0.0))
    if rng.random() < 0.7 and hall_w > 1.0:
        o.append(cyl(0.1, yh1 - 0.17, 0.2, 0.15, 0.4))  # plant pot narrowing the hallway (wide hallways only)
    # kitchen north of the hallway
    o += [wall(4.0, yh1, 4.0, 4.0), wall(-0.95, 4.0, 4.0, 4.0)]
    o.append(block(3.65, 1.5, 0.6, 3.0, 0.9, 0.0))  # counter
    o.append(lifted(block(3.35 - 0.22, 1.2 + rng.uniform(-0.5, 0.5), 0.45, 0.5, 0.18, 0.0), 0.62))  # open drawer
    cy_ = 2.3 + rng.uniform(-0.3, 0.4)
    o.append(lifted(wall(3.35, cy_, 3.35 - 0.4 * math.cos(0.5), cy_ + 0.4 * math.sin(0.5), h=0.62, t=0.02), 0.1))  # open cabinet door
    o += table(0.0, 2.2, 1.0, 0.8)
    o += chair(0.5, 1.8, math.pi / 2 + rng.uniform(-0.4, 0.4)) + chair(1.25, 2.6, math.pi + rng.uniform(-0.4, 0.4))
    keep_out = [(-0.3, 1.4, 1.75, 3.25), (2.95, 0.0, 4.0, 3.2), (0.9, yh1, 1.7, 1.4), (1.3, 1.0, 2.6, 3.6)]  # table, counter, doorway path
    placed = 0
    for _ in range(200):
        if placed >= n_clutter:
            break
        cx, cy, k = rng.uniform(0.0, 3.0), rng.uniform(yh1 + 0.4, 3.6), rng.integers(5)
        if any(a <= cx <= b and c <= cy <= d_ for a, c, b, d_ in keep_out):
            continue
        placed += 1
        if k == 0:
            o.append(block(cx, cy, 0.45, 0.35, 0.30, rng.uniform(0, 3)))  # cardboard box
        elif k == 1:
            o.append(block(cx, cy, 0.24, 0.12, 0.10, rng.uniform(0, 3)))  # tissue box
        elif k == 2:
            a = rng.uniform(0, math.pi)  # pool noodle, 3 segments, slightly bent
            for s_ in (-0.45, 0.0, 0.45):
                o.append(cyl(cx + s_ * math.cos(a), cy + s_ * math.sin(a) + 0.08 * (s_ != 0), 0.035, 0.035, 0.48, (math.cos(a), math.sin(a), 0)))
        elif k == 3:
            o.append(block(cx, cy, 0.32, 0.22, 0.42, rng.uniform(0, 3)))  # backpack
        else:
            o.append(block(cx, cy, 0.40, 0.30, 0.12, rng.uniform(0, 3)))  # bag
    # laundry alcove south of the hallway: step up to a door, washer, shoe rack, mat, shoes
    xl0, xl1 = 2.4, 3.9  # 1.5 m wide: shoe rack + washer leave a 0.58 m lane
    o += [wall(xl0, yh0, xl0, -4.0), wall(xl1, yh0, xl1, -4.0)]
    o.append(block(3.15, yh0 - 0.05, 0.8, 0.30, lerp(0.05, 0.12, d), 0.0))  # step at the turn into the laundry (sill / step down)
    o.append(block((xl0 + xl1) / 2 - 0.1, -3.75, 0.85, 0.45, step, 0.0))
    o.append(block(xl1 - 0.30, -3.0, 0.6, 0.65, 0.9, 0.0))  # washer
    o.append(block(xl0 + 0.16, -3.1, 0.32, 0.6, 0.45, 0.0))  # shoe rack
    o.append(block((xl0 + xl1) / 2, -2.85, 0.6, 0.45, 0.012, 0.0))
    for _ in range(int(lerp(2, 6, d))):
        o.append(block(rng.uniform(xl0 + 0.45, xl1 - 0.75), rng.uniform(-3.45, yh0 - 0.2), 0.28, 0.11, 0.09, rng.uniform(0, 3)))  # shoes (steppable)
    targets = [(-0.3, -2.0, "through the door"), (3.2, -2.0, "end of hallway"), (3.1, -3.75, "laundry step"),
               (1.5, 3.2, "kitchen"), (-2.5, 1.6, "patio")]
    home_rooms.last_routes = [
        ((-3.2, -2.0, 0.0), (-1.6, -2.0), (-0.6, -2.0), (-0.1, -2.12), (0.4, -2.12), (3.2, -2.0)),   # patio, threshold, door, hallway
        ((3.55, -2.0, 0.0), (3.0, -2.0), (3.0, yh0 - 0.2), (3.0, -3.4), (3.0, -3.75)),              # dead end: turn right, sill, laundry step
        ((-0.55, -2.15, math.pi), (1.3, -2.0), (1.3, yh1 + 0.3), (2.0, 1.2), (2.2, 3.3)),           # facing the door: turn, hallway, kitchen
        ((-3.0, 1.6, 0.0), (-1.6, 1.6), (-0.6, 1.6), (0.6, 1.2))]                                 # patio, tiled step, sliding door
    return [ground()], o, targets


def home_spawns():
    """Spawn poses (x, y, yaw) that face a dead end, so the first move is to turn in place, back out, or sidestep."""
    return [(3.55, -2.0, 0.0, "hallway end, facing the closet: turn round, or turn right into the laundry"),
            (3.15, -3.35, -math.pi / 2, "laundry, facing the step and door: turn round and leave"),
            (2.85, 3.55, math.pi / 2, "kitchen corner, facing the wall: turn, sidestep the counter"),
            (-0.55, -2.0 - 0.2, math.pi, "just inside the door, facing it: turn round into the hallway")]
