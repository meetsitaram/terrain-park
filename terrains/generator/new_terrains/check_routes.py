"""Check every commanded route against the geometry it runs through: a route is blocked where the walking surface jumps by more than
MAX_STEP between samples 5 cm apart, along the centre line or 0.2 m to either side (the robot's half width). Walls, closed lane ends,
furniture and kerbs too tall to step show up as jumps; stairs (risers <= 22 cm), ramps and 4x4s do not.
python -I review/new_terrains/check_routes.py   (CPU only)"""
import math
import pathlib
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_new_terrains as G  # noqa: E402
from homes import home_rooms  # noqa: E402
from street import street_front  # noqa: E402
from stairs import stairs as _stairs  # noqa: E402

MAX_STEP = 0.26
HALF_W = 0.20


def stairs_fn(d, seed):
    g, o, t, _ = _stairs(d, seed)
    stairs_fn.last_routes = _stairs.last_routes
    return g, o, t


SECTIONS = [(n, f) for n, f in G.TYPES] + [("homes", home_rooms), ("street front", street_front), ("stairs", stairs_fn)]


def heights(mesh, xy):
    o = np.c_[xy, np.full(len(xy), 20.0)]
    loc, idx, _ = mesh.ray.intersects_location(o, np.tile((0, 0, -1.0), (len(xy), 1)), multiple_hits=False)
    z = np.zeros(len(xy))
    if len(idx):
        z[np.asarray(idx)] = np.asarray(loc).reshape(-1, 3)[:, 2]
    return z


def first_hit(mesh, origins, dirs):
    loc, idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    out = np.full((len(origins), 3), np.nan)
    if len(idx):
        out[np.asarray(idx)] = np.asarray(loc).reshape(-1, 3)
    return out


def route_blocked(mesh, route):
    """Walk the route in 5 cm steps along the centre line and 0.2 m to each side, tracking the surface height h:
    blocked where something stands in the way at body height (h + 0.3, 0.7, 1.2 m: walls, closed ends, furniture), where the next
    surface is more than MAX_STEP above h (too tall to step up), or where it drops more than MAX_STEP (an edge to fall off).
    Door headers (2.03 m) and anything else above head height do not block. Returns None or (x, y, reason)."""
    pts = np.array([(route[0][0], route[0][1])] + [(w[0], w[1]) for w in route[1:]], float)
    for off in (0.0, -HALF_W, HALF_W):
        h = None
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            u = (b - a) / L
            nrm = np.array([-u[1], u[0]])
            n = max(2, int(L / 0.05) + 1)
            xy = a + nrm * off + (b - a) * np.linspace(0, 1, n)[:, None]
            if h is None:
                h0 = first_hit(mesh, np.array([[xy[0, 0], xy[0, 1], 1.0]]), np.array([[0, 0, -1.0]]))[0, 2]
                h = 0.0 if np.isnan(h0) else float(h0)
            for k in range(n - 1):
                step = xy[k + 1] - xy[k]
                ds = float(np.linalg.norm(step))
                if ds < 1e-9:
                    continue
                dirn = np.array([step[0] / ds, step[1] / ds, 0.0])
                org = np.array([[xy[k, 0], xy[k, 1], h + z] for z in (0.30, 0.70, 1.20)])
                hit = first_hit(mesh, org, np.tile(dirn, (3, 1)))
                dist = np.linalg.norm(hit[:, :2] - org[:, :2], axis=1)
                if np.any(dist < ds + 0.02):
                    return float(xy[k, 0]), float(xy[k, 1]), "obstacle at body height"
                nz = first_hit(mesh, np.array([[xy[k + 1, 0], xy[k + 1, 1], h + MAX_STEP + 0.01]]), np.array([[0, 0, -1.0]]))[0, 2]
                nz = 0.0 if np.isnan(nz) else float(nz)
                if h - nz > MAX_STEP:
                    return float(xy[k + 1, 0]), float(xy[k + 1, 1]), "drop of %.2f m" % (h - nz)
                h = nz
    return None


def check(levels=(0.0, 0.5, 1.0), seeds=range(6), verbose=True):
    bad, total = [], 0
    for name, fn in SECTIONS:
        for d in levels:
            for sd in seeds:
                g, o, _ = fn(d, seed=sd)
                mesh = trimesh.util.concatenate(list(g) + list(o))
                for i, rt in enumerate(getattr(fn, "last_routes", [])):
                    total += 1
                    r = route_blocked(mesh, rt)
                    if r:
                        bad.append((name, d, sd, i, r))
    if verbose:
        print("routes checked: %d, blocked: %d" % (total, len(bad)))
        for b in bad[:80]:
            print("  %-28s level %.1f seed %d route %d: %s at (%.2f, %.2f)" % (b[0][:28], b[1], b[2], b[3], b[4][2], b[4][0], b[4][1]))
    return bad, total


if __name__ == "__main__":
    check()
