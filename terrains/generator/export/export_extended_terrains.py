#!/usr/bin/env python
"""Export the extended training terrain map (docs/TODO_extended_training.md, "To do 1") to a simulator-neutral dataset.

CPU only. Imports the generators in review/new_terrains read-only (no bytecode written there), tags every primitive they make,
rasterises each 8 x 8 m tile at 2 cm, lists every primitive as a box or a mesh, checks the routes with check_routes.py and
writes one .npz + one .json per tile type.

  PYTHONDONTWRITEBYTECODE=1 python -B export_extended_terrains.py --out OUT/extended_v1 --workers 6
"""
import os
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import time

import numpy as np
import trimesh
from scipy import ndimage

GEN_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "new_terrains")
TILE = 8.0
RES = 0.02
N = int(round(TILE / RES))          # 400 cells
X0 = -TILE / 2 + RES / 2            # centre of cell 0 (-3.99 m)
LEVELS = 10
MAX_STEP = 0.26                     # same as check_routes.py: taller than this above its support = not steppable
BODY_BAND = (0.05, 1.60)            # obstacle_mask: geometry between floor + 0.05 m and floor + 1.6 m
SPAWN_CLEAR = 0.12                  # spawn poses keep at least this distance from obstacle_mask cells
EDGE = 3.9                          # route points further out than this are pulled back into the tile
ROLE = {"floor": 0, "low": 1, "tall": 2}

TYPE_KEYS = ["single_beams", "beam_fields", "blocks", "wedges", "tight_spots", "log_corridors", "homes", "street_front", "stairs"]
VARIANTS = {"single_beams": 3, "beam_fields": 3, "blocks": 3, "wedges": 4, "tight_spots": 2, "log_corridors": 4, "homes": 3,
            "street_front": 3, "stairs": 8}
STAIR_KINDS = ["straight", "L", "U", "amphitheatre"]
WEDGE_KINDS = ["pinwheel", "long_ramp"]

G = homes = street = stairs = CR = SR = None
ROUTE_CLEAR = 0.35                  # street_front routes: least distance to an obstacle_mask cell along the whole route
FNS = {}
CALLS = []  # per-tile log of generator helper calls (corridor sides, rails)


# ------------------------------------------------------------------ generators: import read-only, tag primitives
def _each(res):
    if isinstance(res, trimesh.Trimesh):
        yield res
    elif isinstance(res, (list, tuple)):
        for r in res:
            yield from _each(r)


def _wrap(fn, prim=None, group=None, force_prim=False, log=None, special=None):
    def w(*a, **k):
        res = fn(*a, **k)
        if special == "beam":
            for i, m in enumerate(res):
                m.metadata["prim"] = "beam" if i == 0 else "beam_support"
            return res
        for m in _each(res):
            if prim:
                if force_prim:
                    m.metadata["prim"] = prim
                else:
                    m.metadata.setdefault("prim", prim)
            if group:
                m.metadata.setdefault("group", group)
        if log:
            CALLS.append(log(a, k))
        return res
    w.__wrapped__ = fn
    return w


def load_generators(gen_dir):
    """Import the generator modules and replace their helper functions (in this process only) by tagging wrappers."""
    global G, homes, street, stairs, CR, SR, FNS
    sys.path.insert(0, gen_dir)
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    import gen_new_terrains as G_
    import homes as homes_
    import street as street_
    import stairs as stairs_
    import check_routes as CR_
    import street_routed as SR_  # our variant of street.py (this directory): routes first, obstacles clear of them
    G, homes, street, stairs, CR, SR = G_, homes_, street_, stairs_, CR_, SR_
    table = [
        (G.ground, dict(prim="ground")),
        (G.beam, dict(special="beam")),
        (G.wedge, dict(prim="wedge")),
        (G.block, dict(prim="block")),
        (G.wall, dict(prim="wall")),
        (G.bar, dict(prim="bar")),
        (G.brace_stand, dict(prim="board", group="brace_stand")),
        (G.rails_around, dict(prim="rail", group="rail_fence",
                              log=lambda a, k: ("rails_around", dict(height=k.get("height")))),),
        (G.side, dict(prim="rail", group="corridor_side", log=lambda a, k: ("side", dict(y=float(a[2]), style=str(a[3]), height=float(a[5]))))),
        (homes.cyl, dict(prim="cylinder")),
        (homes.chair, dict(group="chair")),
        (homes.table, dict(group="table")),
        (homes.wall_with_openings, dict(group="wall_with_openings")),
        (homes.door_leaf, dict(group="door_leaf")),
        (street.slab, dict(prim="slab")),
        (stairs.oriented_box, dict(prim="box")),
        (stairs.flight, dict(prim="stair_step", force_prim=True)),
        (stairs.stepped_sides, dict(group="stair_side")),
        (stairs.ring_sector, dict(prim="ring_sector")),
    ]
    for orig, kw in table:
        w = _wrap(orig, **kw)
        for M in (G, homes, street, stairs, SR):
            for name, val in list(vars(M).items()):
                if val is orig:
                    setattr(M, name, w)
    FNS = {"single_beams": G.single_beams, "beam_fields": G.beam_fields, "blocks": G.blocks, "wedges": G.wedges,
           "tight_spots": G.tight_spots, "log_corridors": G.log_corridors, "homes": homes.home_rooms,
           "street_front": SR.street_front, "stairs": stairs.stairs}


def label_of(m):
    md = m.metadata
    kind, group, prim = md.get("kind"), md.get("group"), md.get("prim", "object")
    if kind in ("car", "tyre", "rock", "plant", "lawn"):
        return kind
    if group == "wall_with_openings":
        return "door_header" if m.bounds[0, 2] > 1.0 else "wall"
    if group == "rail_fence" or group == "corridor_side" or group == "stair_side":
        sub = {"block": "post", "rail": "rail", "wall": "wall", "box": "wall", "bar": "handrail"}.get(prim, prim)
        return group + ":" + sub
    if group == "brace_stand":
        return "brace_stand:" + {"block": "post", "bar": "brace", "board": "board"}.get(prim, prim)
    if group:
        return group + (":" + prim if prim not in ("wall",) else "")
    if kind == "paving" and prim == "slab":
        return "paving_slab"
    return prim


# ------------------------------------------------------------------ geometry helpers
def quat_wxyz(R):
    M = np.eye(4)
    M[:3, :3] = R
    q = np.asarray(trimesh.transformations.quaternion_from_matrix(M), float)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def as_box(m):
    """(centre, half extents, R) if the mesh is an exact box, else None. R columns = box axes; z axis the most vertical one."""
    v = np.asarray(m.vertices, float)
    if len(v) != 8:
        return None
    d = v - v[0]
    for i, j, k in itertools.combinations(range(1, 8), 3):
        e = d[[i, j, k]]
        n = np.linalg.norm(e, axis=1)
        if n.min() < 1e-9:
            continue
        u = e / n[:, None]
        if abs(u[0] @ u[1]) > 1e-6 or abs(u[0] @ u[2]) > 1e-6 or abs(u[1] @ u[2]) > 1e-6:
            continue
        corners = np.array([v[0] + a * e[0] + b * e[1] + c * e[2] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        dist = np.linalg.norm(corners[:, None] - v[None], axis=2)
        if dist.min(1).max() < 1e-6 and dist.min(0).max() < 1e-6:
            h = n / 2
            iz = int(np.argmax(np.abs(u[:, 2])))
            z = u[iz] * (1.0 if u[iz, 2] >= 0 else -1.0)
            rest = [a for a in range(3) if a != iz]
            ix = rest[0] if abs(u[rest[0], 0]) >= abs(u[rest[1], 0]) else rest[1]
            iy = rest[0] if ix == rest[1] else rest[1]
            x = u[ix] * (1.0 if u[ix, 0] >= 0 else -1.0)
            y = np.cross(z, x)
            R = np.c_[x, y, z]
            return v.mean(0), np.array([h[ix], h[iy], h[iz]]), R
    return None


def raster_mesh(v, f):
    """Top and bottom surface of a mesh sampled at the cell centres inside its bounding box.
    Returns (i0, j0, top, bot) or None; top is -inf and bot +inf where the mesh does not cover the cell centre."""
    lo, hi = v[:, :2].min(0), v[:, :2].max(0)
    i0 = max(0, int(math.ceil((lo[0] - X0) / RES - 1e-9)))
    i1 = min(N - 1, int(math.floor((hi[0] - X0) / RES + 1e-9)))
    j0 = max(0, int(math.ceil((lo[1] - X0) / RES - 1e-9)))
    j1 = min(N - 1, int(math.floor((hi[1] - X0) / RES + 1e-9)))
    if i1 < i0 or j1 < j0:
        return None
    top = np.full((i1 - i0 + 1, j1 - j0 + 1), -np.inf)
    bot = np.full((i1 - i0 + 1, j1 - j0 + 1), np.inf)
    eps = 1e-9
    for t in v[f]:
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = t
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < 1e-12:
            continue
        a0 = max(i0, int(math.ceil((min(x1, x2, x3) - X0) / RES - 1e-9)))
        a1 = min(i1, int(math.floor((max(x1, x2, x3) - X0) / RES + 1e-9)))
        b0 = max(j0, int(math.ceil((min(y1, y2, y3) - X0) / RES - 1e-9)))
        b1 = min(j1, int(math.floor((max(y1, y2, y3) - X0) / RES + 1e-9)))
        if a1 < a0 or b1 < b0:
            continue
        xs = (X0 + RES * np.arange(a0, a1 + 1))[:, None]
        ys = (X0 + RES * np.arange(b0, b1 + 1))[None, :]
        l1 = ((y2 - y3) * (xs - x3) + (x3 - x2) * (ys - y3)) / det
        l2 = ((y3 - y1) * (xs - x3) + (x1 - x3) * (ys - y3)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -eps) & (l2 >= -eps) & (l3 >= -eps)
        if not inside.any():
            continue
        z = l1 * z1 + l2 * z2 + l3 * z3
        ts = top[a0 - i0:a1 - i0 + 1, b0 - j0:b1 - j0 + 1]
        bs = bot[a0 - i0:a1 - i0 + 1, b0 - j0:b1 - j0 + 1]
        np.maximum(ts, np.where(inside, z, -np.inf), out=ts)
        np.minimum(bs, np.where(inside, z, np.inf), out=bs)
    if not np.isfinite(top).any():
        return None
    return i0, j0, top, bot


def cell_of(x, y):
    return int(np.clip(math.floor((x + TILE / 2) / RES), 0, N - 1)), int(np.clip(math.floor((y + TILE / 2) / RES), 0, N - 1))


def lerp(a, b, d):
    return a + (b - a) * d


def params_for(key, d):
    """The difficulty-dependent generator parameters at difficulty d (the formulas of the generators; random draws are within these)."""
    r = lambda x: round(float(x), 4)
    if key == "single_beams":
        return dict(n_beams=6, beam_size_m=[r(lerp(.05, .15, d) * .85), r(lerp(.05, .15, d))], crossing_angle_max_deg=r(lerp(15, 60, d)),
                    slant_raise_max_m=r(lerp(0, .20, d)), slant_probability=r(lerp(0, .5, d)), beam_length_m=[1.2, 2.4],
                    brace_stand_probability=r(lerp(.3, .8, d)))
    if key == "beam_fields":
        return dict(row_spacing_m=r(lerp(1.2, .5, d)), beam_size_m=[r(lerp(.05, .12, d) * .8), r(lerp(.05, .12, d))],
                    angle_max_deg=r(lerp(10, 50, d)), v_pair_probability=0.35, v_pair_half_angle_deg=[5, 12], v_pair_beam_length_m=2.6,
                    slant_raise_max_m=r(lerp(0, .15, d)), slant_probability=r(.3 * d), brace_stand_probability=r(lerp(.2, .6, d)))
    if key == "blocks":
        return dict(n_blocks=int(lerp(10, 34, d)), block_height_m=[0.05, r(lerp(.10, .25, d))], block_side_m=[0.08, 0.45])
    if key == "wedges":
        return dict(slope_deg=[r(lerp(8, 15, d)), r(lerp(12, 25, d))], pinwheel_cell_m=[1.2, 2.0], long_ramp_length_m=[1.5, 3.0],
                    narrow_strip_width_m=r(lerp(.35, .15, d)), wide_wedge_width_m=[r(lerp(1.2, .6, d)), 1.2],
                    brace_stand_probability=r(lerp(.3, .8, d)))
    if key == "tight_spots":
        return dict(lane_width_m=r(lerp(1.0, .6, d)), wall_gap_m=r(lerp(.75, .50, d)), lane_beam_size_m=r(lerp(.05, .10, d)),
                    pocket_block_height_m=r(lerp(.08, .2, d)), wall_height_m=1.0, wall_thickness_m=0.08)
    if key == "log_corridors":
        return dict(corridor_width_m=r(lerp(1.5, 1.0, d)), corridor_length_m=7.6, log_size_m=[r(lerp(.06, .10, d)), r(lerp(.08, .15, d))],
                    log_angle_max_deg=r(lerp(10, 50, d)), min_clear_gap_m=r(lerp(.40, .26, d)), log_spacing_m=[r(lerp(1.3, .30, d)), r(lerp(1.7, .50, d))],
                    slant_raise_max_m=r(lerp(0, .10, d)), side_heights_m=dict(knee=0.30, waist=0.70, house=2.44))
    if key == "homes":
        return dict(door_width_m=r(lerp(.95, .76, d)), hall_width_m=r(lerp(1.15, .90, d)), threshold_rise_m=r(lerp(.03, .08, d)),
                    step_rise_m=r(lerp(.12, .21, d)), laundry_sill_m=r(lerp(.05, .12, d)), n_clutter=int(lerp(3, 12, d)),
                    n_bricks=int(lerp(3, 8, d)), n_shoes=int(lerp(2, 6, d)))
    if key == "street_front":
        return dict(curb_m=r(lerp(.10, .18, d)), slab_heave_m=r(lerp(0, .03, d)), porch_step_m=r(lerp(.12, .20, d)),
                    driveway_rise_m=r(lerp(.10, .30, d)), garage_threshold_m=r(lerp(.02, .06, d)), n_turf_bumps=int(lerp(4, 14, d)),
                    n_stones=int(lerp(25, 80, d)), car_probability=r(lerp(.4, .9, d)))
    if key == "stairs":
        return dict(rise_m=[r(lerp(.08, .15, d)), r(lerp(.12, .22, d))], run_m=[r(lerp(.32, .22, d)), r(lerp(.40, .30, d))],
                    width_m=[r(lerp(1.4, .7, d)), r(lerp(2.0, 1.1, d))], steps=[int(lerp(3, 7, d)), int(lerp(6, 15, d))], nosing_m=[0.0, 0.03])
    return {}


# ------------------------------------------------------------------ one tile
def generate(key, d, seed):
    del CALLS[:]
    fn = FNS[key]
    out = fn(d, seed)
    g, o, targets = out[0], out[1], out[2]
    info = out[3] if len(out) > 3 else None
    routes = [tuple(tuple(float(c) for c in p) for p in rt) for rt in getattr(fn, "last_routes", [])]
    return list(g), list(o), list(targets), info, routes, list(CALLS)


def variant_kind(key, info, routes):
    if key == "stairs":
        return info["kind"]
    if key == "wedges":
        return "pinwheel" if len(routes) == 4 else "long_ramp"
    return None


def build_tile(task):
    key, level, variant, seed = task
    d = level / (LEVELS - 1)
    g, o, targets, info, routes, calls = generate(key, d, seed)
    prims = [(m, 0) for m in g] + [(m, 1) for m in o]

    # ---- rasterise every primitive, then stack the grounded ones into the floor raster
    H = np.zeros((N, N))
    recs = []
    for m, gen_group in prims:
        v, f = np.asarray(m.vertices, float), np.asarray(m.faces)
        md = m.metadata
        recs.append(dict(m=m, v=v, f=f, gen_group=gen_group, label=label_of(m), prim=md.get("prim", "object"), group=md.get("group"),
                         ras=raster_mesh(v, f), box=as_box(m), role=None, in_raster=False))
    order = sorted(range(len(recs)), key=lambda i: (recs[i]["v"][:, 2].min(), recs[i]["v"][:, 2].max()))
    pending = list(order)
    for _ in range(5):
        left = []
        for i in pending:
            r = recs[i]
            if r["prim"] == "ground":
                r["role"], r["in_raster"] = "floor", True
                continue
            if r["ras"] is None:
                left.append(i)
                continue
            i0, j0, top, bot = r["ras"]
            F = np.isfinite(top)
            Hs = H[i0:i0 + top.shape[0], j0:j0 + top.shape[1]]
            gap = (bot - Hs)[F]
            if np.percentile(gap, 90) > 0.04:
                left.append(i)
                continue
            rise = float((top - Hs)[F].max())
            floor_like = (r["prim"] in ("wedge", "slab", "ring_sector", "stair_step")
                          or (key == "stairs" and r["prim"] in ("block", "box") and r["group"] is None)
                          or (key == "wedges" and r["prim"] == "block" and r["group"] is None))
            if floor_like:
                r["role"] = "floor"
            elif rise <= MAX_STEP + 1e-6:
                r["role"] = "low"
            else:
                r["role"] = "tall"
                continue
            r["in_raster"] = True
            np.maximum(Hs, np.where(F, top, -np.inf), out=Hs)
        if len(left) == len(pending):
            break
        pending = left
    Htop = H.copy()
    mask = np.zeros((N, N), bool)
    for r in recs:
        if r["ras"] is None:
            if r["role"] is None:
                r["role"] = "tall" if r["v"][:, 2].max() > MAX_STEP else "low"
            continue
        i0, j0, top, bot = r["ras"]
        F = np.isfinite(top)
        Hs = H[i0:i0 + top.shape[0], j0:j0 + top.shape[1]]
        if r["role"] is None:  # not resting on the floor stack: overhang, lifted or slanted
            r["role"] = "low" if float((top - Hs)[F].max()) <= MAX_STEP + 1e-6 else "tall"
        Ts = Htop[i0:i0 + top.shape[0], j0:j0 + top.shape[1]]
        np.maximum(Ts, np.where(F, top, -np.inf), out=Ts)
        if not r["in_raster"]:
            blk = F & (top - Hs > MAX_STEP) & (bot < Hs + BODY_BAND[1]) & (top > Hs + BODY_BAND[0])
            mask[i0:i0 + top.shape[0], j0:j0 + top.shape[1]] |= blk
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool))
    # distance to the nearest obstacle cell, capped at 9.99 m (also the value on tiles without any)
    clear = np.minimum(ndimage.distance_transform_edt(~mask) * RES, 9.99) if mask.any() else np.full((N, N), 9.99)

    # ---- routes: check_routes.py on the full mesh, then our own cell check
    full = trimesh.util.concatenate([m for m, _ in prims])
    kept, dropped = [], []

    def point_info(x, y):
        inside = bool(abs(x) <= TILE / 2 and abs(y) <= TILE / 2)
        ci, cj = cell_of(x, y)
        return inside, bool(mask[ci, cj]), float(H[ci, cj]), float(clear[ci, cj])

    for ri, rt in enumerate(routes):
        fixed = None
        if any(abs(c) > EDGE + 1e-9 for p in rt for c in p[:2]):  # small fix: pinwheel routes end 0.1 m outside the tile
            fixed = "points beyond +-%.1f m pulled back to the tile (generator gave %s)" % (
                EDGE, "; ".join("(%.2f, %.2f)" % (p[0], p[1]) for p in rt if max(abs(p[0]), abs(p[1])) > EDGE + 1e-9))
            rt = tuple(tuple(float(np.clip(c, -EDGE, EDGE)) if k < 2 else c for k, c in enumerate(p)) for p in rt)
        pts = [(rt[0][0], rt[0][1])] + [(w[0], w[1]) for w in rt[1:]]
        rec = dict(source_index=ri, start=[rt[0][0], rt[0][1], rt[0][2]], waypoints=[[p[0], p[1]] for p in pts[1:]],
                   target=[pts[-1][0], pts[-1][1]],
                   length_m=float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:]))))
        infos = [point_info(*p) for p in pts]
        if fixed:
            rec["fixed"] = fixed
        rec["z"] = [round(i[2], 4) for i in infos]
        rec["clearance_m"] = [round(i[3], 3) for i in infos]
        along = []
        for a, b in zip(pts[:-1], pts[1:]):
            for q in np.linspace(0.0, 1.0, max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / RES) + 1)):
                ci, cj = cell_of(a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)
                along.append(clear[ci, cj])
        rec["min_clearance_along_route_m"] = round(float(min(along)), 3)
        res = CR.route_blocked(full, rt)
        reason = None
        if res:
            reason = dict(by="check_routes.py", why=res[2], at=[round(res[0], 3), round(res[1], 3)])
        elif not all(i[0] for i in infos):
            reason = dict(by="export check", why="route point outside the tile")
        elif any(i[1] for i in infos):
            k = [i[1] for i in infos].index(True)
            reason = dict(by="export check", why="route point on an obstacle cell", at=[round(pts[k][0], 3), round(pts[k][1], 3)])
        elif key == "street_front" and rec["min_clearance_along_route_m"] < ROUTE_CLEAR:
            reason = dict(by="export check", why="closer than %.2f m to an obstacle cell along the route" % ROUTE_CLEAR)
        if reason:
            rec["dropped"] = reason
            dropped.append(rec)
        else:
            kept.append(rec)

    # ---- spawn poses
    cand = []
    if key in ("single_beams", "beam_fields", "blocks", "wedges", "tight_spots"):
        cand.append((0.0, 0.0, 0.0, "generator", "tile centre, facing +x"))
    if key == "homes":
        cand += [(s[0], s[1], s[2], "home_spawns", s[3]) for s in homes.home_spawns()]
    if key == "street_front":
        cand += [(s[0], s[1], s[2], "street_spawns", s[3]) for s in SR.street_spawns()]
    for k, rec in enumerate(kept):
        cand.append((rec["start"][0], rec["start"][1], rec["start"][2], "route", "start of route %d" % k))
    spawns, spawns_rejected = [], []
    yy, xx = np.meshgrid(np.arange(N), np.arange(N))
    for x, y, yaw, src, note in cand:
        inside, onobs, z, cl = point_info(x, y)
        ci, cj = cell_of(x, y)
        near = (xx - ci) ** 2 + (yy - cj) ** 2 <= (0.20 / RES) ** 2
        rng_ = float(H[near].max() - H[near].min())
        s = dict(x=float(x), y=float(y), yaw=float(yaw), z=round(z, 4), clearance_m=round(cl, 3), floor_range_r20_m=round(rng_, 3),
                 source=src, note=note)
        if (not inside) or onobs or cl < SPAWN_CLEAR:
            s["rejected"] = "outside tile" if not inside else "closer than %.2f m to an obstacle cell" % SPAWN_CLEAR
            spawns_rejected.append(s)
        elif any(abs(s["x"] - q["x"]) < 1e-6 and abs(s["y"] - q["y"]) < 1e-6 and abs(s["yaw"] - q["yaw"]) < 1e-6 for q in spawns):
            continue
        else:
            spawns.append(s)

    # ---- primitive tables
    boxes, meshes = [], []
    for r in recs:
        if r["box"] is not None:
            c, h, R = r["box"]
            upright = abs(R[2, 2] - 1.0) < 1e-9
            boxes.append(dict(center=c, half=h, quat=quat_wxyz(R), yaw=math.atan2(R[1, 0], R[0, 0]), upright=upright,
                              role=ROLE[r["role"]], in_raster=r["in_raster"], label=r["label"], gen_group=r["gen_group"]))
        else:
            v = r["v"]
            meshes.append(dict(v=v.astype(np.float32), f=r["f"].astype(np.int32), role=ROLE[r["role"]], in_raster=r["in_raster"],
                               label=r["label"], gen_group=r["gen_group"], aabb_c=(v.min(0) + v.max(0)) / 2, aabb_h=(v.max(0) - v.min(0)) / 2))

    # ---- realised parameters
    labels = [r["label"] for r in recs]
    realised = dict(primitive_counts={l: labels.count(l) for l in sorted(set(labels))})
    vk = variant_kind(key, info, routes)
    if info:
        realised["stairs"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()}
    if key == "wedges":
        realised["main_feature"] = vk
        wd = [r for r in recs if r["prim"] == "wedge"]
        if vk == "pinwheel":
            cells = wd[:4]
            ext = cells[0]["v"].max(0) - cells[0]["v"].min(0)
            realised["pinwheel"] = dict(cell_m=round(float(max(ext[0], ext[1])), 3), cells=4,
                                        crest_m=[round(float(c["v"][:, 2].max()), 3) for c in cells],
                                        slope_deg=[round(math.degrees(math.atan2(float(c["v"][:, 2].max()), max(ext[0], ext[1]) / 2)), 2) for c in cells],
                                        rails=any(r["group"] == "rail_fence" for r in recs))
        else:
            plat = [r for r in recs if r["prim"] == "block" and r["group"] is None][0]
            c, h, R = plat["box"]
            realised["long_ramp"] = dict(plateau_height_m=round(float(2 * h[2]), 3), length_m=round(float(2 * h[0]), 3), width_m=round(float(2 * h[1]), 3),
                                         rails=any(r["group"] == "rail_fence" for r in recs))
    if key == "street_front":
        realised["generator"] = "terrain_export/street_routed.py"
        realised["left_out_no_room"] = list(getattr(SR.street_front, "last_skipped", []))
    if key == "log_corridors":
        realised["sides"] = [c[1] for c in calls if c[0] == "side"]
    rails = [c[1] for c in calls if c[0] == "rails_around"]
    if rails:
        realised["rail_fences"] = len(rails)

    tile_id = "%s_L%d_v%d" % (key, level, variant)
    meta = dict(tile_id=tile_id, type=key, level=level, difficulty=d, seed=seed, variant=variant, variant_kind=vk,
                size_m=[TILE, TILE], resolution_m=RES, shape=[N, N], origin=[0.0, 0.0, 0.0], cell0_center=[X0, X0],
                params=params_for(key, d), realised=realised, n_boxes=len(boxes), n_meshes=len(meshes),
                floor_height_range_m=[round(float(H.min()), 4), round(float(H.max()), 4)], top_height_max_m=round(float(Htop.max()), 4),
                spawns=spawns, spawns_rejected=spawns_rejected, routes=kept,
                targets=[dict(x=float(t[0]), y=float(t[1]), label=str(t[2])) for t in targets
                         if point_info(t[0], t[1])[0] and not point_info(t[0], t[1])[1]],
                raw_targets_unchecked=[dict(x=float(t[0]), y=float(t[1]), label=str(t[2]),
                                            why="outside the tile" if not point_info(t[0], t[1])[0] else "on an obstacle cell")
                                       for t in targets if not point_info(t[0], t[1])[0] or point_info(t[0], t[1])[1]])
    return dict(key=key, meta=meta, height=H.astype(np.float32), height_top=Htop.astype(np.float32), mask=mask, boxes=boxes,
                meshes=meshes, dropped=[dict(tile_id=tile_id, type=key, level=level, seed=seed, **r) for r in dropped],
                n_routes_defined=len(routes))


# ------------------------------------------------------------------ the whole set
def plan_tasks():
    """(type, level, variant, seed). Seeds are 10000 * (type index + 1) + 100 * level + k; for stairs and wedges k is scanned upward
    until every stair kind / wedge main feature appears the same number of times per level."""
    tasks = []
    for ti, key in enumerate(TYPE_KEYS):
        for level in range(LEVELS):
            d = level / (LEVELS - 1)
            base = 10000 * (ti + 1) + 100 * level
            if key in ("stairs", "wedges"):
                kinds = STAIR_KINDS if key == "stairs" else WEDGE_KINDS
                per = VARIANTS[key] // len(kinds)
                got = {k: [] for k in kinds}
                k = 0
                while any(len(v) < per for v in got.values()):
                    _, _, _, info, routes, _ = generate(key, d, base + k)
                    vk = variant_kind(key, info, routes)
                    if len(got[vk]) < per:
                        got[vk].append(base + k)
                    k += 1
                    if k > 99:
                        raise RuntimeError("seed scan failed for %s level %d" % (key, level))
                seeds = [s for kk in kinds for s in got[kk]]
            else:
                seeds = [base + k for k in range(VARIANTS[key])]
            tasks += [(key, level, v, s) for v, s in enumerate(seeds)]
    return tasks


def write_type(out, key, tiles):
    labels = sorted({b["label"] for t in tiles for b in t["boxes"]} | {m["label"] for t in tiles for m in t["meshes"]})
    lid = {l: i for i, l in enumerate(labels)}
    B = [b for t in tiles for b in t["boxes"]]
    M = [m for t in tiles for m in t["meshes"]]
    box_start = np.cumsum([0] + [len(t["boxes"]) for t in tiles]).astype(np.int32)
    mesh_start = np.cumsum([0] + [len(t["meshes"]) for t in tiles]).astype(np.int32)
    arr = dict(
        height=np.stack([t["height"] for t in tiles]),
        height_top=np.stack([t["height_top"] for t in tiles]),
        obstacle_mask=np.stack([t["mask"] for t in tiles]),
        tile_level=np.array([t["meta"]["level"] for t in tiles], np.int32),
        tile_difficulty=np.array([t["meta"]["difficulty"] for t in tiles], np.float64),
        tile_seed=np.array([t["meta"]["seed"] for t in tiles], np.int32),
        tile_variant=np.array([t["meta"]["variant"] for t in tiles], np.int32),
        tile_id=np.array([t["meta"]["tile_id"] for t in tiles]),
        resolution=np.float32(RES), size=np.array([TILE, TILE], np.float32), cell0_center=np.array([X0, X0], np.float32),
        labels=np.array(labels),
        box_start=box_start,
        box_center=np.array([b["center"] for b in B], np.float32).reshape(-1, 3),
        box_half=np.array([b["half"] for b in B], np.float32).reshape(-1, 3),
        box_quat=np.array([b["quat"] for b in B], np.float32).reshape(-1, 4),
        box_yaw=np.array([b["yaw"] for b in B], np.float32),
        box_upright=np.array([b["upright"] for b in B], bool),
        box_role=np.array([b["role"] for b in B], np.uint8),
        box_in_raster=np.array([b["in_raster"] for b in B], bool),
        box_label=np.array([lid[b["label"]] for b in B], np.int16),
        box_gen_group=np.array([b["gen_group"] for b in B], np.uint8),
        mesh_start=mesh_start,
        mesh_vert_start=np.cumsum([0] + [len(m["v"]) for m in M]).astype(np.int32),
        mesh_face_start=np.cumsum([0] + [len(m["f"]) for m in M]).astype(np.int32),
        mesh_vertices=(np.concatenate([m["v"] for m in M]) if M else np.zeros((0, 3), np.float32)),
        mesh_faces=(np.concatenate([m["f"] for m in M]) if M else np.zeros((0, 3), np.int32)),
        mesh_role=np.array([m["role"] for m in M], np.uint8),
        mesh_in_raster=np.array([m["in_raster"] for m in M], bool),
        mesh_label=np.array([lid[m["label"]] for m in M], np.int16),
        mesh_gen_group=np.array([m["gen_group"] for m in M], np.uint8),
        mesh_aabb_center=np.array([m["aabb_c"] for m in M], np.float32).reshape(-1, 3),
        mesh_aabb_half=np.array([m["aabb_h"] for m in M], np.float32).reshape(-1, 3),
    )
    np.savez_compressed(out / (key + ".npz"), **arr)
    metas = []
    for i, t in enumerate(tiles):
        m = dict(t["meta"])
        m["npz"] = key + ".npz"
        m["index"] = i
        metas.append(m)
    json.dump(dict(type=key, npz=key + ".npz", n_tiles=len(tiles), labels=labels, tiles=metas), open(out / (key + ".json"), "w"), indent=1)
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-dir", default=GEN_DIR_DEFAULT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated tile types (for a test run)")
    ap.add_argument("--levels", default=None, help="comma-separated level indices (for a test run)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_generators(args.gen_dir)
    t0 = time.time()
    tasks = plan_tasks()
    if args.only:
        tasks = [t for t in tasks if t[0] in args.only.split(",")]
    if args.levels:
        tasks = [t for t in tasks if t[1] in [int(x) for x in args.levels.split(",")]]
    print("tiles planned: %d (%.1f s)" % (len(tasks), time.time() - t0), flush=True)
    with mp.get_context("fork").Pool(max(1, min(args.workers, 8))) as pool:
        results = []
        for k, r in enumerate(pool.imap(build_tile, tasks, chunksize=2)):
            results.append(r)
            if (k + 1) % 30 == 0:
                print("  %d / %d tiles (%.0f s)" % (k + 1, len(tasks), time.time() - t0), flush=True)
    summary, dropped = {}, []
    for key in TYPE_KEYS:
        tiles = [r for r in results if r["key"] == key]
        if not tiles:
            continue
        metas = write_type(out, key, tiles)
        d_ = [x for t in tiles for x in t["dropped"]]
        dropped += d_
        kinds = {}
        for m in metas:
            if m["variant_kind"]:
                kinds[m["variant_kind"]] = kinds.get(m["variant_kind"], 0) + 1
        summary[key] = dict(npz=key + ".npz", json=key + ".json", n_tiles=len(tiles), levels=sorted({m["level"] for m in metas}),
                            variants_per_level=VARIANTS[key], variant_kinds=kinds,
                            routes_defined=sum(t["n_routes_defined"] for t in tiles), routes_dropped=len(d_),
                            routes_dropped_by_check_routes=sum(1 for x in d_ if x["dropped"]["by"] == "check_routes.py"),
                            routes_kept=sum(len(m["routes"]) for m in metas),
                            routes_fixed_kept=sum(1 for m in metas for r in m["routes"] if "fixed" in r),
                            tiles_without_route=sum(1 for m in metas if not m["routes"]),
                            min_routes_per_tile=min(len(m["routes"]) for m in metas),
                            routes_kept_per_level=[sum(len(m["routes"]) for m in metas if m["level"] == l) for l in range(LEVELS)],
                            spawns=sum(len(m["spawns"]) for m in metas), spawns_rejected=sum(len(m["spawns_rejected"]) for m in metas),
                            tiles_without_spawn=sum(1 for m in metas if not m["spawns"]),
                            n_boxes=sum(m["n_boxes"] for m in metas), n_meshes=sum(m["n_meshes"] for m in metas))
        print(key, json.dumps(summary[key]), flush=True)
    json.dump(dict(note="routes defined by the generators but left out of the dataset", n=len(dropped), routes=dropped),
              open(out / "routes_dropped.json", "w"), indent=1)
    # a layout for laying the tiles out side by side (used by the preview): column = type variant, row = level
    col, layout = 0, {}
    for key in TYPE_KEYS:
        if key in summary:
            layout[key] = dict(first_column=col, columns=VARIANTS[key])
            col += VARIANTS[key]
    index = dict(
        name="extended_v1", made=time.strftime("%Y-%m-%d %H:%M:%S"), units="m, rad", tile_size_m=[TILE, TILE], resolution_m=RES,
        raster_shape=[N, N], cell0_center=[X0, X0], levels=LEVELS, difficulty_of_level=[l / (LEVELS - 1) for l in range(LEVELS)],
        edge_clamp_m=EDGE, roles={"0": "floor", "1": "low (<= %.2f m above its support)" % MAX_STEP, "2": "tall"}, max_step_m=MAX_STEP,
        obstacle_mask_band_m=list(BODY_BAND), spawn_clearance_m=SPAWN_CLEAR, quat_order="wxyz", street_route_clearance_m=ROUTE_CLEAR,
        generators=dict(street_front="terrain_export/street_routed.py (variant of review/new_terrains/street.py)", others="review/new_terrains (unchanged)"),
        types=summary, layout=dict(note="suggested field: tile (type, level, variant) at x = level * 8 m, y = (first_column + variant) * 8 m",
                                   columns=layout, total_columns=col),
        totals=dict(tiles=sum(s["n_tiles"] for s in summary.values()), routes_defined=sum(s["routes_defined"] for s in summary.values()),
                    routes_dropped=sum(s["routes_dropped"] for s in summary.values()), routes_kept=sum(s["routes_kept"] for s in summary.values()),
                    routes_fixed_kept=sum(s["routes_fixed_kept"] for s in summary.values()),
                    spawns=sum(s["spawns"] for s in summary.values())),
        command="PYTHONDONTWRITEBYTECODE=1 python -B export_extended_terrains.py --out %s --workers %d" % (args.out, args.workers))
    json.dump(index, open(out / "index.json", "w"), indent=1)
    print("totals", json.dumps(index["totals"]), "  %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
