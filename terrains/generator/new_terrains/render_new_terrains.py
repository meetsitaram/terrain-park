"""Render the proposed terrains at easy / medium / hard with a G1-sized figure for scale. CPU only (VTK off-screen)."""
import pathlib
import sys

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_new_terrains as G  # noqa: E402

OUT = pathlib.Path(__file__).parent
font = ImageFont.truetype("arial.ttf", 22)
small = ImageFont.truetype("arial.ttf", 17)


def pv_mesh(m):
    return pv.PolyData(np.asarray(m.vertices), np.c_[np.full(len(m.faces), 3), m.faces].ravel())


def figure(p):
    """A G1-sized stand-in at the spawn (1.32 m, 0.45 m shoulders), facing +x, for scale only."""
    for c, r, h in (((0, -0.1, 0.36), 0.06, 0.72), ((0, 0.1, 0.36), 0.06, 0.72)):
        p.add_mesh(pv.Cylinder(center=c, direction=(0, 0, 1), radius=r, height=h), color="#f2f2f2")
    p.add_mesh(pv.Box(bounds=(-0.1, 0.1, -0.2, 0.2, 0.72, 1.15)), color="#f2f2f2")
    p.add_mesh(pv.Sphere(radius=0.1, center=(0.02, 0, 1.24)), color="#2b2b2b")
    p.add_mesh(pv.Arrow(start=(0.15, 0, 0.02), direction=(1, 0, 0), scale=0.5), color="#2f6fd6")


def shot(ground, obs, targets, cam, size):
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("#c9ccd1")
    for g in ground:
        p.add_mesh(pv_mesh(g), color="#9ea3aa", ambient=0.3, diffuse=0.75)
    for o in obs:
        p.add_mesh(pv_mesh(o), color="#c79a5b", ambient=0.3, diffuse=0.8, show_edges=False)
    figure(p)
    for (x, y, label) in targets:
        p.add_mesh(pv.Cylinder(center=(x, y, 0.9), direction=(0, 0, 1), radius=0.03, height=1.0), color="#d23a2a")
    p.remove_all_lights()
    p.add_light(pv.Light(position=(-20, -40, 50), focal_point=(0, 0, 0), intensity=1.0, light_type="scene light"))
    p.add_light(pv.Light(position=(30, 20, 30), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light"))
    p.camera_position = cam
    im = p.screenshot(return_img=True)
    p.close()
    return Image.fromarray(im)


levels = (("easy", 0.0), ("medium", 0.5), ("hard", 1.0))
W, H = 560, 400
cam_persp = [(-6.5, -6.5, 5.5), (0.3, 0.3, 0.0), (0, 0, 1)]
sheet = Image.new("RGB", (260 + 3 * W, 46 + len(G.TYPES) * H), (20, 24, 30))
dr = ImageDraw.Draw(sheet)
for k, (name, _) in enumerate(levels):
    dr.text((270 + k * W, 10), name, fill=(235, 235, 235), font=font)
for i, (title, fn) in enumerate(G.TYPES):
    words, line, lines = title.split(), "", []
    for wd in words:
        if len(line) + len(wd) > 24:
            lines.append(line); line = ""
        line += wd + " "
    lines.append(line)
    for j, l in enumerate(lines):
        dr.text((10, 46 + i * H + H // 2 - 30 + 22 * j), l, fill=(235, 235, 235), font=small)
    for k, (_, d) in enumerate(levels):
        g, o, t = fn(d, seed=7 + i)
        sheet.paste(shot(g, o, t, cam_persp, (W, H)), (260 + k * W, 46 + i * H))
    print("rendered", title, flush=True)
dr.text((10, sheet.size[1] - 26), "Preview only, not in training. Wood colour: obstacles (separate meshes). White figure: G1 size, blue arrow: facing. Red posts: targets.",
        fill=(200, 200, 200), font=small)
sheet.save(OUT / "new_terrains_sheet.jpg", quality=90)

# close-ups at hard: one per type, larger
for i, (title, fn) in enumerate(G.TYPES):
    g, o, t = fn(1.0, seed=7 + i)
    shot(g, o, t, [(-3.6, -3.0, 2.2), (0.8, 0.4, 0.1), (0, 0, 1)], (1280, 800)).save(OUT / f"closeup_{i + 1}_{fn.__name__}.jpg", quality=90)
print("done", flush=True)
