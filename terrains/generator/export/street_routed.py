"""Street-front generator variant used by the extended_v1 export. A copy of hiking-in-the-wild/review/new_terrains/street.py
(2026-11-11) with the routes planned first and the obstacles placed clear of them. The original file is not changed.

Same content and the same difficulty ranges as the original: street, gutter, curb 10 -> 18 cm, sidewalk with heaved slabs, driveway
apron ramp and slope to the garage, garage threshold, lawn with edging and bumps, river-rock bed, planter kerb, front walk, porch step
12 -> 20 cm, door threshold, house wall with garage and front door openings, hydrant, sign pole, crate, parked car.

What differs from street.py:
1. Routes. Four per tile, planned before anything is placed:
     A  street -> up the curb -> lawn -> porch step                 (crosses the curb at y = ya, random 1.0 .. 1.5)
     B  street -> up the driveway apron -> beside the car -> garage threshold      (y = -1.225)
     C  porch (facing the door) -> down the step -> rock bed -> sidewalk -> down the curb   (y = yc, random 3.0 .. 3.3)
     D  garage threshold (facing in) -> down the driveway and the apron -> street
   The original's three routes ran through the planter bed, the hydrant group and the parked car on most seeds.
2. Hydrant, sign pole and crate: each gets its own place along the sidewalk, at least MARGIN from every route (the original fixed
   the pole 0.6 m and the crate 1.1 m from the hydrant). One that finds no place after 400 tries is left out.
3. Bushes and boulders: drawn from the original ranges, kept only if at least MARGIN (plus their radius) from every route.
4. Parked car: centre line moved 0.15 m away from the house-side jamb (y = -2.55 instead of -2.4) and its position along the driveway
   limited to 0.6 .. 1.2 m (was 0.6 .. 1.4), so a 0.85 m lane stays between the car and the garage jamb and the nose stays
   outside the garage wall.
5. A garage floor slab behind the threshold (x 3.45 .. 4.0, at driveway-top height), so the threshold is a step and not the edge of a
   drop of up to 0.55 m to z = 0.
6. Porch poses at (3.2, 2.7) instead of (3.1 / 3.2, 2.6): the middle of the doorway, 0.49 m from each jamb.
7. House wall: built here (house_wall) instead of homes.wall_with_openings, so each door header starts 2.03 m above the floor of
   its own doorway. In street.py the headers are at z = 2.03 while the garage floor and the porch rise to 0.45 m at the hard levels,
   which leaves 1.58 m of headroom and puts the header inside the body-height band.
8. Spawn poses are the four route starts of the tile (street_front.last_spawns)."""
import math

import numpy as np
import trimesh

from gen_new_terrains import block, ground, lerp, wall  # noqa: F401
from homes import cyl, lifted
from street import slab, tag

MARGIN = 0.42  # m, route centre line to the surface of anything taller than a step (0.35 m asked, plus raster slack)


def _seg_dist(px, py, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def route_dist(px, py, routes):
    return min(_seg_dist(px, py, a, b) for rt in routes for a, b in zip(rt[:-1], rt[1:]))


def house_wall(x, openings, t=0.10):
    """Wall along y at x from -4 to 4 with door openings (centre y, width, floor height there). As homes.wall_with_openings, but the
    header of each opening starts 2.03 m above that doorway's floor and the wall top is 2.44 m above the highest doorway floor."""
    top = 2.44 + max(fz for _, _, fz in openings)
    parts, s0 = [], -4.0
    for yc, w, fz in sorted(openings):
        if yc - w / 2 - s0 > 0.01:
            parts.append(wall(x, s0, x, yc - w / 2, h=top, t=t))
        parts.append(lifted(wall(x, yc - w / 2, x, yc + w / 2, h=top - (2.03 + fz), t=t), 2.03 + fz))
        s0 = yc + w / 2
    if 4.0 - s0 > 0.01:
        parts.append(wall(x, s0, x, 4.0, h=top, t=t))
    for m in parts:
        m.metadata["group"] = "wall_with_openings"
    return parts


def street_front(d, seed):
    rng = np.random.default_rng(seed)
    o = []
    curb = lerp(0.10, 0.18, d)       # curb height above the gutter
    heave = lerp(0.0, 0.03, d)       # sidewalk slab offsets at the joints
    porch = lerp(0.12, 0.20, d)      # porch step
    y_drive0, y_drive1 = -3.6, -0.6  # driveway band along y
    # ---- routes first
    ya, yc, yb = rng.uniform(1.0, 1.5), rng.uniform(3.0, 3.3), -1.225
    px_, py_ = 3.2, 2.7              # on the porch, middle of the doorway
    gx = 3.38                        # on the garage threshold
    routes = [
        ((-3.2, ya, 0.0), (-1.5, ya), (1.2, ya), (2.3, py_), (px_, py_)),      # A: street, up the curb, lawn, porch step
        ((-3.2, yb, 0.0), (-1.9, yb), (3.0, yb), (gx, yb)),                    # B: street, up the apron, beside the car, garage threshold
        ((px_, py_, 0.0), (2.3, py_), (1.9, yc), (-1.5, yc), (-3.0, yc)),      # C: porch facing the door: turn, step down, rock bed, curb down
        ((gx, yb, 0.0), (3.0, yb), (-1.9, yb), (-3.0, yb))]                    # D: garage threshold facing in: turn, driveway and apron down
    skipped = []
    # ---- ground surfaces (as in street.py)
    o.append(tag(slab(-2.6, -2.2, -4.0, 4.0, 0.03, 0.0), "paving"))  # gutter: a shallow V dropping 3 cm towards the curb
    for k, y in enumerate(np.arange(-4.0, 4.0, 1.2)):                # sidewalk slabs 1.2 m long, each heaved a little
        y1 = min(y + 1.2, 4.0)
        if y1 > y_drive0 and y < y_drive1:
            continue
        h = curb + rng.uniform(-heave, heave)
        o.append(tag(slab(-2.2, -0.8, y + 0.005, y1 - 0.005, h, h + rng.uniform(-heave, heave) * 0.5), "paving"))
    o.append(tag(slab(-2.2, -1.6, y_drive0, y_drive1, 0.0, curb), "paving"))               # driveway apron (curb cut)
    o.append(tag(slab(-1.6, -0.8, y_drive0, y_drive1, curb, curb + 0.01), "paving"))
    o.append(tag(slab(-0.8, 3.3, y_drive0, y_drive1, curb + 0.01, curb + 0.01 + lerp(0.10, 0.30, d)), "paving"))  # driveway slope
    gz = curb + 0.01 + lerp(0.10, 0.30, d)
    o.append(tag(slab(3.3, 3.45, y_drive0 + 0.2, y_drive1 - 0.2, gz + lerp(0.02, 0.06, d)), "paving"))          # garage threshold
    o.append(tag(slab(3.45, 4.0, y_drive0 + 0.2, y_drive1 - 0.2, gz), "paving"))                               # garage floor (new)
    yard_z = curb + 0.02
    o.append(tag(slab(-0.8, 3.4, y_drive1, 4.0, yard_z), "lawn"))
    o.append(block(-0.75, (y_drive1 + 4.0) / 2, 0.04, 4.0 - y_drive1, yard_z + lerp(0.02, 0.05, d), 0.0))       # lawn edging
    o.append(block(1.3, y_drive1 + 0.03, 4.2, 0.04, yard_z + lerp(0.02, 0.05, d), 0.0))
    for _ in range(int(lerp(4, 14, d))):  # turf bumps (a few cm)
        bx, by = rng.uniform(-0.5, 1.8), rng.uniform(y_drive1 + 0.3, 3.8)
        o.append(tag(lifted(block(bx, by, rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8), rng.uniform(0.015, lerp(0.03, 0.06, d)), rng.uniform(0, 3)), yard_z), "lawn"))
    for _ in range(int(lerp(25, 80, d))):  # river-rock bed: small stones (3-8 cm)
        r = rng.uniform(0.03, 0.08) / 2
        o.append(tag(lifted(trimesh.creation.icosphere(subdivisions=1, radius=1.0).apply_scale((r * 1.4, r, r * 0.8)), 0)
                 .apply_translation((rng.uniform(0.3, 1.9), rng.uniform(2.6, 3.9), yard_z + r * 0.4)), "rock"))
    for _ in range(int(lerp(1, 3, d))):    # boulders, clear of the routes
        for _try in range(200):
            r, bx, by = rng.uniform(0.12, 0.2), rng.uniform(2.0, 2.8), rng.uniform(1.0, 1.9)
            if route_dist(bx, by, routes) >= r + MARGIN:
                o.append(tag(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation((bx, by, yard_z + r * 0.5)), "rock"))
                break
        else:
            skipped.append("boulder")
    o.append(tag(slab(1.9, 3.0, y_drive1, 2.4, yard_z + 0.01), "paving"))                                          # walk slabs
    o += [block(2.45, 1.0, 1.0, 0.07, yard_z + 0.10, 0.0), block(1.95, 1.4, 0.07, 0.8, yard_z + 0.10, 0.0)]  # planter kerb
    for _ in range(int(lerp(1, 3, d))):    # bushes (solid blobs), clear of the routes
        for _try in range(200):
            r, bx, by = rng.uniform(0.3, 0.5), rng.uniform(2.3, 3.1), rng.uniform(1.2, 1.9)
            if route_dist(bx, by, routes) >= r + MARGIN:
                o.append(tag(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_scale((1, 1, 0.8)).apply_translation((bx, by, yard_z + r * 0.5)), "plant"))
                break
        else:
            skipped.append("bush")
    o.append(tag(slab(3.0, 3.4, 1.9, 3.3, yard_z + porch), "paving"))                                              # porch step
    o.append(slab(3.4, 3.48, 2.3, 3.1, yard_z + porch + lerp(0.01, 0.04, d)))                                      # door threshold
    # house front wall; the two door headers sit 2.03 m above the floor of their own doorway (garage floor, door threshold)
    o += tag(house_wall(3.5, [(-2.1, 2.6, gz + lerp(0.02, 0.06, d)), (2.7, 0.85, yard_z + porch + lerp(0.01, 0.04, d))]), "wall")
    # ---- street furniture on the sidewalk, each clear of the routes and of the others
    placed = []

    def place(x, r, name):
        for _try in range(400):
            y = rng.uniform(-0.15, 3.8)
            if route_dist(x, y, routes) >= r + MARGIN and all(math.hypot(x - qx, y - qy) >= r + qr + 0.10 for qx, qy, qr in placed):
                placed.append((x, y, r))
                return y
        skipped.append(name)
        return None

    hy = place(-1.05, 0.17, "hydrant")
    if hy is not None:
        o += [cyl(-1.05, hy, curb + 0.3, 0.11, 0.6), cyl(-1.05, hy, curb + 0.5, 0.06, 0.32, (0, 1, 0))]
    sy = place(-1.0, 0.36, "sign pole")
    if sy is not None:
        o += [cyl(-1.0, sy, curb + 1.1, 0.03, 2.2), lifted(block(-1.0, sy, 0.5, 0.5, 0.12, 0.0), curb)]
    cy = place(-1.2, 0.24, "crate")
    if cy is not None:
        o.append(lifted(block(-1.2, cy, 0.33, 0.33, 0.28, 0.3), curb))
    # ---- parked car on the driveway, leaving the lane of route B / D beside it
    if rng.random() < lerp(0.4, 0.9, d):
        car_x, car_y = rng.uniform(0.6, 1.2), -2.55
        o.append(tag(lifted(block(car_x, car_y, 4.4, 1.8, 1.25, 0.0), gz * 0.5 + 0.18), "car"))  # body (0.18 m ground clearance)
        for dx in (-1.35, 1.35):
            for dy in (-0.75, 0.75):
                o.append(tag(cyl(car_x + dx, car_y + dy, gz * 0.5 + 0.33, 0.33, 0.22, (0, 1, 0)), "tyre"))
    for m in o:
        m.metadata.setdefault("kind", "object")
    targets = [(px_, py_, "up the curb, over the lawn, onto the porch"), (gx, yb, "up the driveway apron to the garage threshold"),
               (-3.0, yc, "off the porch, across the rock bed, down the curb"), (-3.0, yb, "down the driveway and the apron to the street")]
    street_front.last_routes = routes
    street_front.last_spawns = [(-3.2, ya, 0.0, "street, facing the curb: step up"),
                                (-3.2, yb, 0.0, "street at the driveway: up the apron"),
                                (px_, py_, 0.0, "porch, facing the door: turn round and step down"),
                                (gx, yb, 0.0, "garage threshold, facing in: turn round, walk down beside the car")]
    street_front.last_skipped = skipped
    return [ground()], o, targets


def street_spawns():
    """Spawn poses (x, y, yaw, note) of the tile generated last."""
    return list(street_front.last_spawns)
