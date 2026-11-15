"""Draw a commanded route (start pose, then waypoints) as a red line hugging the surface, with arrowheads on each leg."""
import math

import numpy as np
import pyvista as pv
import trimesh


def surface_heights(meshes, xy):
    """Top surface height at each (x, y) from the given trimesh meshes (0 where nothing)."""
    if not meshes:
        return np.zeros(len(xy))
    m = trimesh.util.concatenate(meshes)
    o = np.c_[xy, np.full(len(xy), 20.0)]
    d = np.tile((0, 0, -1.0), (len(xy), 1))
    loc, idx, _ = m.ray.intersects_location(o, d, multiple_hits=False)
    z = np.zeros(len(xy))
    if len(idx):
        z[np.asarray(idx)] = np.asarray(loc).reshape(-1, 3)[:, 2]
    return z


def draw_route(p, route, meshes=None, lift=0.12, ox=0.0, oy=0.0, width=4, arrows=True, step=0.1, color="#d23a2a"):
    pts = np.array([(route[0][0], route[0][1])] + [(w[0], w[1]) for w in route[1:]], float)
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        n = max(2, int(L / step) + 1)
        xy = a + (b - a) * np.linspace(0, 1, n)[:, None]
        z = surface_heights(meshes, xy) + lift
        line = np.c_[xy[:, 0] + ox, xy[:, 1] + oy, z]
        p.add_mesh(pv.lines_from_points(line), color=color, line_width=width)
        if arrows and L > 0.2:
            dx, dy = (b - a) / L
            p.add_mesh(pv.Cone(center=(line[-1, 0] - dx * 0.12, line[-1, 1] - dy * 0.12, line[-1, 2]), direction=(dx, dy, 0),
                               height=0.28, radius=0.1), color=color)
