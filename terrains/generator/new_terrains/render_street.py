"""Preview of the 'street front' section (was homes; reused) at easy / medium / hard (top view and two perspectives) plus close-ups at hard.
Walls are cut at 1.1 m in these views (dollhouse view) except homes_full_height.jpg. Blue figures: dead-end spawn poses (arrow = facing). Red posts: targets. CPU only."""
import math
import pathlib
import sys

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from street import street_front, street_spawns  # noqa: E402
home_rooms, home_spawns = street_front, street_spawns
from render_new_terrains import pv_mesh  # noqa: E402

OUT = pathlib.Path(__file__).parent
font = ImageFont.truetype("arial.ttf", 22)


def figure_at(p, x, y, yaw, color):
    c, s = math.cos(yaw), math.sin(yaw)
    for off in (-0.1, 0.1):
        p.add_mesh(pv.Cylinder(center=(x - s * off, y + c * off, 0.36), direction=(0, 0, 1), radius=0.06, height=0.72), color=color)
    body = pv.Box(bounds=(-0.1, 0.1, -0.2, 0.2, 0.72, 1.15)).rotate_z(math.degrees(yaw), inplace=False).translate((x, y, 0), inplace=False)
    p.add_mesh(body, color=color)
    p.add_mesh(pv.Sphere(radius=0.1, center=(x, y, 1.24)), color="#2b2b2b")
    p.add_mesh(pv.Arrow(start=(x, y, 0.03), direction=(c, s, 0), scale=0.6), color="#2f6fd6")


def shot(g, o, t, cam, size, cut=None):
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("#c9ccd1")
    for m in g:
        p.add_mesh(pv_mesh(m), color="#4a4d52", ambient=0.3, diffuse=0.75)
    for m in o:
        pm = pv_mesh(m)
        if cut is not None and pm.bounds[5] > cut:  # dollhouse view: walls cut at `cut` metres so the rooms can be seen
            pm = pm.clip(normal=(0, 0, 1), origin=(0, 0, cut), invert=True)
        if pm.n_points:
            col = {"paving": "#c9c6bf", "lawn": "#7fa35a", "rock": "#8b7d6b", "plant": "#5e7d4a", "wall": "#9fc4c9",
                   "car": "#8e2a2a", "tyre": "#222222"}.get(m.metadata.get("kind"), "#c79a5b")
            p.add_mesh(pm, color=col, ambient=0.3, diffuse=0.8)
    for (x, y, yaw, _) in home_spawns():
        figure_at(p, x, y, yaw, "#8fb7ff")
    for (x, y, _) in t:
        p.add_mesh(pv.Cylinder(center=(x, y, 0.9), direction=(0, 0, 1), radius=0.03, height=1.0), color="#d23a2a")
    p.remove_all_lights()
    p.add_light(pv.Light(position=(-20, -40, 50), focal_point=(0, 0, 0), intensity=1.0, light_type="scene light"))
    p.add_light(pv.Light(position=(30, 20, 30), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light"))
    p.camera_position = cam
    im = p.screenshot(return_img=True)
    p.close()
    return Image.fromarray(im)


W, H = 700, 520
cams = [[(0.0, -0.01, 15.0), (0.0, 0.0, 0.0), (1, 0, 0)],
        [(-6.0, -6.5, 6.5), (0.6, -0.4, 0.0), (0, 0, 1)],
        [(7.0, -4.5, 6.5), (1.0, 0.0, 0.0), (0, 0, 1)]]
sheet = Image.new("RGB", (3 * W, 50 + 3 * H), (20, 24, 30))
dr = ImageDraw.Draw(sheet)
for k, (name, d) in enumerate((("easy", 0.0), ("medium", 0.5), ("hard", 1.0))):
    g, o, t = home_rooms(d, seed=4)
    for i, cam in enumerate(cams):
        sheet.paste(shot(g, o, t, cam, (W, H)), (k * W, 50 + i * H))
    dr.text((k * W + 10, 12), name, fill=(235, 235, 235), font=font)
sheet.save(OUT / "street_sheet.jpg", quality=90)
g, o, t = home_rooms(1.0, seed=4)
shot(g, o, t, [(-4.6, 3.6, 1.6), (-1.6, 1.4, 0.1), (0, 0, 1)], (1400, 900)).save(OUT / "street_closeup_curb.jpg", quality=90)
shot(g, o, t, [(-4.4, -5.6, 2.6), (0.4, -2.0, 0.2), (0, 0, 1)], (1400, 900)).save(OUT / "street_closeup_drive.jpg", quality=90)
shot(g, o, t, [(0.2, 4.6, 2.2), (2.6, 2.2, 0.2), (0, 0, 1)], (1400, 900)).save(OUT / "street_closeup_porch.jpg", quality=90)
shot(g, o, t, [(-6.0, -6.5, 6.5), (0.6, -0.4, 0.0), (0, 0, 1)], (1400, 900), cut=None).save(OUT / "street_full.jpg", quality=90)
print("done")
