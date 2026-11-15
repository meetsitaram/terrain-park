"""Stairs section preview: one row per kind (straight, L, U, amphitheatre) x easy / medium / hard, labelled with riser, tread, width,
steps and sides. Blue figure: route start (arrow = facing); red line and post: the route. CPU only."""
import pathlib
import sys

import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import stairs as S  # noqa: E402
from render_new_terrains import pv_mesh  # noqa: E402
from render_home import figure_at
from routes_draw import draw_route  # noqa: E402

OUT = pathlib.Path(__file__).parent
font = ImageFont.truetype("arial.ttf", 22)
small = ImageFont.truetype("arial.ttf", 16)


def shot(g, o, routes, cam, size):
    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background("#c9ccd1")
    for m in g:
        p.add_mesh(pv_mesh(m), color="#9ea3aa", ambient=0.3, diffuse=0.75)
    for m in o:
        p.add_mesh(pv_mesh(m), color="#c79a5b", ambient=0.3, diffuse=0.8)
    r0 = routes[0]
    sx, sy, syaw = r0[0]
    tx, ty = r0[-1]
    figure_at(p, sx, sy, syaw, "#8fb7ff")
    for k_, rt in enumerate(routes):  # every route on the tile, one colour each
        draw_route(p, rt, meshes=list(o), color=("#d23a2a", "#2f6fd6", "#2e9e4a", "#8a3fc7")[k_ % 4])
    p.add_mesh(pv.Cylinder(center=(tx, ty, 2.0), direction=(0, 0, 1), radius=0.035, height=1.4), color="#d23a2a")
    p.remove_all_lights()
    p.add_light(pv.Light(position=(-20, -40, 50), focal_point=(0, 0, 0), intensity=1.0, light_type="scene light"))
    p.add_light(pv.Light(position=(30, 20, 30), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light"))
    p.camera_position = cam
    im = p.screenshot(return_img=True)
    p.close()
    return Image.fromarray(im)


def seed_for(kind, k):
    """First seed whose tile is of this kind (the kind is drawn from the seed)."""
    for sd in range(k * 1000, k * 1000 + 1000):
        info = S.stairs(0.5, sd)[3]
        if info["kind"] == kind and (kind == "amphitheatre" or info["sides"] != "open"):
            return sd


W, H = 640, 470
kinds = ("straight", "L", "U", "amphitheatre")
sheet = Image.new("RGB", (3 * W, 50 + len(kinds) * H), (20, 24, 30))
dr = ImageDraw.Draw(sheet)
for k, (name, d) in enumerate((("easy", 0.0), ("medium", 0.5), ("hard", 1.0))):
    dr.text((k * W + 10, 12), name, fill=(235, 235, 235), font=font)
for i, kind in enumerate(kinds):
    sd = seed_for(kind, i + 1)
    for k, d in enumerate((0.0, 0.5, 1.0)):
        g, o, t, info = S.stairs(d, seed=sd)
        im = shot(g, o, S.stairs.last_routes, [(-8.0, -8.0, 7.5), (-0.2, 0.0, 0.4), (0, 0, 1)], (W, H))
        side = info["sides"] + ("" if info["sides"] in ("open", "rails") else " %.2f m" % info["side_height"])
        ImageDraw.Draw(im).text((8, H - 24), "%s: riser %.0f cm, tread %.0f cm, width %.2f m, %d steps, sides %s" % (
            info["kind"], info["rise"] * 100, info["run"] * 100, info["width"], info["steps"], side), fill=(20, 20, 20), font=small)
        sheet.paste(im, (k * W, 50 + i * H))
sheet.save(OUT / "stairs_sheet.jpg", quality=90)
print("done")
