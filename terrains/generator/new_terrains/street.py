"""'Street front' terrain section, after the user's photos (2026-11-11): street, gutter and curb, a sidewalk with heaved slabs, a driveway
with a curb-cut apron, a garage threshold, a lawn with edging and bumps, a river-rock bed, a planter kerb, a front walk and a porch step,
plus a hydrant, a sign pole, a crate and a parked car. Dimensions are estimates from the photos (not measured). Preview only.

Layout on one 8 m tile (x towards the house, y along the street): street x < -2.6, gutter -2.6..-2.2, curb face at -2.2,
sidewalk -2.2..-0.8, front yard -0.8..3.4, house front wall at 3.4 (garage door and front door in it)."""
import math

import numpy as np
import trimesh

from gen_new_terrains import block, ground, lerp, wall
from homes import cyl, lifted


def slab(x0, x1, y0, y1, z0, z1=None, zr0=None, zr1=None):
    """A box-like slab whose top may slope: z0 at x0 and z1 at x1 (or zr0/zr1 along y). Bottom at 0."""
    z1 = z0 if z1 is None else z1
    v = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                  [x0, y0, z0], [x1, y0, z1], [x1, y1, z1], [x0, y1, z0]], float)
    if zr0 is not None:
        v[4:, 2] = [zr0, zr0, zr1, zr1]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    m = trimesh.Trimesh(v, f, process=True)
    m.fix_normals()
    return m


def tag(ms, kind):
    for m in (ms if isinstance(ms, list) else [ms]):
        m.metadata["kind"] = kind
    return ms


def street_front(d, seed):
    rng = np.random.default_rng(seed)
    o = []
    curb = lerp(0.10, 0.18, d)       # curb height above the gutter
    heave = lerp(0.0, 0.03, d)       # sidewalk slab offsets at the joints
    porch = lerp(0.12, 0.20, d)      # porch step
    y_drive0, y_drive1 = -3.6, -0.6  # driveway band along y
    # gutter: a shallow V dropping 3 cm towards the curb; street at 0
    o.append(tag(slab(-2.6, -2.2, -4.0, 4.0, 0.03, 0.0), "paving"))
    # sidewalk at curb height, slabs 1.2 m long along the street, each heaved a little (a trip edge at every joint)
    for k, y in enumerate(np.arange(-4.0, 4.0, 1.2)):
        y1 = min(y + 1.2, 4.0)
        if y1 > y_drive0 and y < y_drive1:  # the driveway apron replaces the sidewalk here (ramp from gutter up to the yard)
            continue
        h = curb + rng.uniform(-heave, heave)
        o.append(tag(slab(-2.2, -0.8, y + 0.005, y1 - 0.005, h, h + rng.uniform(-heave, heave) * 0.5), "paving"))
    # driveway apron: curb cut, rising from the gutter to sidewalk height (~10-17 deg), then the driveway rising to the garage
    o.append(tag(slab(-2.2, -1.6, y_drive0, y_drive1, 0.0, curb), "paving"))
    o.append(tag(slab(-1.6, -0.8, y_drive0, y_drive1, curb, curb + 0.01), "paving"))
    o.append(tag(slab(-0.8, 3.3, y_drive0, y_drive1, curb + 0.01, curb + 0.01 + lerp(0.10, 0.30, d)), "paving"))  # driveway slope to the garage
    gz = curb + 0.01 + lerp(0.10, 0.30, d)
    o.append(tag(slab(3.3, 3.45, y_drive0 + 0.2, y_drive1 - 0.2, gz + lerp(0.02, 0.06, d)), "paving"))         # garage threshold
    # front yard ground (lawn) at sidewalk height with bumps; lawn edging strip along the sidewalk and the driveway
    yard_z = curb + 0.02
    o.append(tag(slab(-0.8, 3.4, y_drive1, 4.0, yard_z), "lawn"))
    o.append(block(-0.75, (y_drive1 + 4.0) / 2, 0.04, 4.0 - y_drive1, yard_z + lerp(0.02, 0.05, d), 0.0))
    o.append(block(1.3, y_drive1 + 0.03, 4.2, 0.04, yard_z + lerp(0.02, 0.05, d), 0.0))
    for _ in range(int(lerp(4, 14, d))):  # turf bumps and dips (a few cm)
        bx, by = rng.uniform(-0.5, 1.8), rng.uniform(y_drive1 + 0.3, 3.8)
        o.append(tag(lifted(block(bx, by, rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8), rng.uniform(0.015, lerp(0.03, 0.06, d)), rng.uniform(0, 3)), yard_z), "lawn"))
    # river-rock bed between the lawn and the walk: many small stones (3-8 cm), a few boulders
    for _ in range(int(lerp(25, 80, d))):
        r = rng.uniform(0.03, 0.08) / 2
        o.append(tag(lifted(trimesh.creation.icosphere(subdivisions=1, radius=1.0).apply_scale((r * 1.4, r, r * 0.8)), 0)
                 .apply_translation((rng.uniform(0.3, 1.9), rng.uniform(2.6, 3.9), yard_z + r * 0.4)), "rock"))
    for _ in range(int(lerp(1, 3, d))):
        r = rng.uniform(0.12, 0.2)
        o.append(tag(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation((rng.uniform(2.0, 2.8), rng.uniform(1.0, 1.9), yard_z + r * 0.5)), "rock"))
    # front walk from the driveway to the porch, planter bed with a kerb, bushes (solid blobs), porch step and door threshold
    o.append(tag(slab(1.9, 3.0, y_drive1, 2.4, yard_z + 0.01), "paving"))                                          # walk slabs
    o += [block(2.45, 1.0, 1.0, 0.07, yard_z + 0.10, 0.0), block(1.95, 1.4, 0.07, 0.8, yard_z + 0.10, 0.0)]  # planter kerb
    for _ in range(int(lerp(1, 3, d))):
        r = rng.uniform(0.3, 0.5)
        o.append(tag(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_scale((1, 1, 0.8)).apply_translation((rng.uniform(2.3, 3.1), rng.uniform(1.2, 1.9), yard_z + r * 0.5)), "plant"))
    o.append(tag(slab(3.0, 3.4, 1.9, 3.3, yard_z + porch), "paving"))                                              # porch step
    o.append(slab(3.4, 3.48, 2.3, 3.1, yard_z + porch + lerp(0.01, 0.04, d)))                       # door threshold
    # house front wall with a garage door opening and a front door opening (2.44 m)
    from homes import wall_with_openings
    o += tag(wall_with_openings(3.5, -4.0, 3.5, 4.0, [(1.9, 2.6), (6.7, 0.85)]), "wall")
    # street furniture: hydrant, sign pole with a base, a crate, a parked car on the driveway
    hx, hy = -1.05, rng.uniform(1.0, 3.5)
    o += [cyl(hx, hy, curb + 0.3, 0.11, 0.6), cyl(hx, hy, curb + 0.5, 0.06, 0.32, (0, 1, 0))]
    o += [cyl(-1.0, hy - 0.6, curb + 1.1, 0.03, 2.2), lifted(block(-1.0, hy - 0.6, 0.5, 0.5, 0.12, 0.0), curb)]
    o.append(lifted(block(-1.2, hy - 1.1, 0.33, 0.33, 0.28, 0.3), curb))
    if rng.random() < lerp(0.4, 0.9, d):
        car_x = rng.uniform(0.6, 1.4)
        o.append(tag(lifted(block(car_x, (y_drive0 + y_drive1) / 2 - 0.3, 4.4, 1.8, 1.25, 0.0), gz * 0.5 + 0.18), "car"))  # body (0.18 m ground clearance)
        for dx in (-1.35, 1.35):
            for dy in (-0.75, 0.75):
                o.append(tag(cyl(car_x + dx, (y_drive0 + y_drive1) / 2 - 0.3 + dy, gz * 0.5 + 0.33, 0.33, 0.22, (0, 1, 0)), "tyre"))
    for m in o:
        m.metadata.setdefault("kind", "object")
    targets = [(-0.3, 1.5, "up the curb onto the sidewalk"), (2.9, -2.1, "up the driveway into the garage"),
               (3.2, 2.6, "onto the porch"), (1.0, 3.3, "across the rock bed")]
    street_front.last_routes = [
        ((-3.2, 1.5, 0.0), (-1.5, 1.5), (1.0, 1.5), (2.4, 2.3), (3.2, 2.6)),      # street, up the curb, lawn, onto the walk, porch step
        ((-3.2, -2.1, 0.0), (-1.9, -2.1), (-0.5, -0.9), (2.9, -0.9), (2.9, -2.1)), # street, up the apron, beside the car, turn into the garage
        ((3.1, 2.6, 0.0), (2.6, 2.0), (2.6, -0.4), (-1.5, -0.4), (-3.0, -0.4))]    # porch facing the door: turn, walk, driveway, down to the street
    return [ground()], o, targets


def street_spawns():
    """Spawn poses (x, y, yaw): some facing the curb from the street, some facing dead ends (garage back, front door)."""
    return [(-3.2, 1.5, 0.0, "street, facing the curb: step up"),
            (-3.2, -2.1, 0.0, "street at the driveway: up the apron"),
            (3.1, 2.6, 0.0, "porch, facing the door: turn round and step down"),
            (2.9, -2.1, 0.0, "garage threshold, facing in: turn round, walk down beside the car")]
