"""Wedges section: 4 tiles x easy / medium / hard. Blue figure: route start (arrow = facing); red post: route end through the main
walk area (between the rails on the long ramps, across the ridges on the pinwheels). CPU only."""
import math, pathlib, sys
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_new_terrains as G
from render_new_terrains import pv_mesh
from render_home import figure_at
from routes_draw import draw_route

OUT = pathlib.Path(__file__).parent
font = ImageFont.truetype("arial.ttf", 22)


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
    p.add_mesh(pv.Cylinder(center=(tx, ty, 0.9), direction=(0, 0, 1), radius=0.035, height=1.0), color="#d23a2a")
    p.remove_all_lights()
    p.add_light(pv.Light(position=(-20, -40, 50), focal_point=(0, 0, 0), intensity=1.0, light_type="scene light"))
    p.add_light(pv.Light(position=(30, 20, 30), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light"))
    p.camera_position = cam
    im = p.screenshot(return_img=True)
    p.close()
    return Image.fromarray(im)


W, H = 640, 470
seeds = (2, 5, 9, 14)
sheet = Image.new("RGB", (3 * W, 50 + len(seeds) * H), (20, 24, 30))
dr = ImageDraw.Draw(sheet)
for k, (name, d) in enumerate((("easy", 0.0), ("medium", 0.5), ("hard", 1.0))):
    dr.text((k * W + 10, 12), name, fill=(235, 235, 235), font=font)
    for i, sd in enumerate(seeds):
        g, o, t = G.wedges(d, seed=sd)
        sheet.paste(shot(g, o, G.wedges.last_routes, [(-8.5, -8.0, 8.5), (0.0, 0.3, 0.0), (0, 0, 1)], (W, H)), (k * W, 50 + i * H))
sheet.save(OUT / "wedges_sheet.jpg", quality=90)
print("done")
