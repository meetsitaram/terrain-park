"""Preview of the combined training field: the paper's terrain field as generated for the current run (review/terrain/terrain.ply,
10 difficulty rows x 20 columns of 8 m tiles), with the proposed new sections added as 10 extra columns (2 per type), same difficulty
rows. A separate map for a later run: the current run's config and terrain are untouched. CPU only."""
import json
import pathlib
import sys

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import gen_new_terrains as G  # noqa: E402

T = HERE.parent / "terrain"
L = json.load(open(T / "layout.json"))
O = np.load(T / "layout.npz")["origins"]  # (rows, cols, 3); row -> x, column -> y
S = L["size"][0]
rows, cols = L["rows"], L["cols"]

paper = pv.read(T / "terrain.ply").triangulate()
pts, faces = paper.points, paper.faces.reshape(-1, 4)[:, 1:]
keep = pts[faces][:, :, 2].max(1) < 1.5  # tile-edge walls hidden, as in the earlier terrain review
paper = pv.PolyData(pts, np.c_[np.full(int(keep.sum()), 3), faces[keep]].ravel())

from homes import home_rooms  # noqa: E402
from street import street_front  # noqa: E402
from stairs import stairs as _stairs  # noqa: E402
from routes_draw import draw_route  # noqa: E402


def stairs_fn(d, seed):
    g, o, t, _ = _stairs(d, seed)
    stairs_fn.last_routes = _stairs.last_routes
    return g, o, t


new_cols = [(name, fn) for name, fn in G.TYPES + [("homes", home_rooms), ("street front", street_front), ("stairs", stairs_fn)]
            for _ in range(2)]  # 9 x 2 = 18
y0 = O[0, -1, 1] + S
ground_new, obs_new, route_lines = [], [], []
for j, (name, fn) in enumerate(new_cols):
    for r in range(rows):
        d = r / (rows - 1)
        g, o, _ = fn(d, seed=100 * j + r)
        ox, oy = O[r, 0, 0], y0 + j * S
        for rt in getattr(fn, "last_routes", []):
            route_lines.append((rt, ox, oy))
        for m in g:
            m = m.copy(); m.apply_translation((ox, oy, 0)); ground_new.append(m)
        for m in o:
            k = m.metadata.get("kind", "object") if hasattr(m, "metadata") else "object"
            m = m.copy(); m.apply_translation((ox, oy, 0)); m.metadata["kind"] = k; obs_new.append(m)


def merge(ms):
    v, f, off = [], [], 0
    for m in ms:
        v.append(m.vertices); f.append(m.faces + off); off += len(m.vertices)
    v, f = np.vstack(v), np.vstack(f)
    return pv.PolyData(v, np.c_[np.full(len(f), 3), f].ravel())


gnew = merge(ground_new)
COL = {"paving": "#d6d3cc", "lawn": "#7fa35a", "rock": "#8b7d6b", "plant": "#5e7d4a", "wall": "#9fc4c9", "car": "#8e2a2a", "tyre": "#222222",
       "object": "#c79a5b"}
by_kind = {}
for m in obs_new:
    by_kind.setdefault(m.metadata.get("kind", "object"), []).append(m)
onew = {k: merge(v) for k, v in by_kind.items()}
font = ImageFont.truetype("arial.ttf", 26)
small = ImageFont.truetype("arial.ttf", 18)


def shot(cam, size, labels=True):
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("#c9ccd1")
    p.add_mesh(paper, color="#b9bcc2", ambient=0.3, diffuse=0.75)
    p.add_mesh(gnew, color="#9ea3aa", ambient=0.3, diffuse=0.75)
    for k, mesh in onew.items():
        p.add_mesh(mesh, color=COL.get(k, "#c79a5b"), ambient=0.3, diffuse=0.8)
    for (rt, ox_, oy_) in route_lines:  # commanded routes (with their turns) through each section's main walk area
        draw_route(p, rt, meshes=None, lift=1.0, ox=ox_, oy=oy_, width=2, arrows=False, step=10.0)
    p.remove_all_lights()
    p.add_light(pv.Light(position=(-60, -120, 120), focal_point=(0, 0, 0), intensity=1.0, light_type="scene light"))
    p.add_light(pv.Light(position=(80, 60, 60), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light"))
    p.camera_position = cam
    im = p.screenshot(return_img=True)
    p.close()
    return Image.fromarray(im)


# 1) the whole field from above, with column labels
xs, ys = O[:, 0, 0], np.r_[O[0, :, 1], y0 + np.arange(len(new_cols)) * S]
cx, cy = xs.mean(), ys.mean()
span = ys.max() - ys.min() + S
top = shot([(cx - 1, cy, span * 1.05), (cx, cy, 0), (1, 0, 0)], (2400, 900))
p = pv.Plotter(off_screen=True)
p.close()
dr = ImageDraw.Draw(top)
dr.rectangle((0, 0, 2400, 40), fill=(20, 24, 30))
dr.text((14, 8), "Combined training field (preview): left, the paper's 20 columns as in the current run; right, 18 new columns: beams, beam fields, blocks, wedges, tight spots, log corridors, homes, street front, stairs; red lines: routes (wood = new obstacles). Rows: difficulty, easy at the bottom.",
        fill=(235, 235, 235), font=small)
top.save(HERE / "combined_field_top.jpg", quality=90)

# 2) perspective over the whole field
shot([(cx - 95, cy - 120, 85), (cx, cy + 10, 0), (0, 0, 1)], (2400, 1200)).save(HERE / "combined_field_persp.jpg", quality=90)

# 3) the seam: last paper columns (boxes, slopes) next to the new ones, medium-to-hard rows
seam_y = y0 - S / 2
shot([(O[5, 0, 0] - 30, seam_y - 22, 22), (O[6, 0, 0], seam_y + 18, 0), (0, 0, 1)], (2000, 1100)).save(HERE / "combined_field_seam.jpg", quality=90)
ny0, ny1 = y0 - S / 2, y0 + len(new_cols) * S - S / 2
ncx, ncy = xs.mean(), (ny0 + ny1) / 2
shot([(ncx - 70, ncy - 80, 70), (ncx, ncy, 0), (0, 0, 1)], (2400, 1300)).save(HERE / "combined_field_new_section.jpg", quality=90)
print("wrote combined_field_top / persp / seam / new_section", flush=True)
