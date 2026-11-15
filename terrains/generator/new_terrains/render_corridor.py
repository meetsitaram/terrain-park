"""Preview of the log-corridor section: four side styles x easy / medium / hard, from behind the spawn, plus a top view. CPU only."""
import pathlib, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import gen_new_terrains as G
from render_new_terrains import shot  # importing also re-renders the earlier sheet (cheap)

OUT = pathlib.Path(__file__).parent
font = ImageFont.truetype("arial.ttf", 22)
W, H = 640, 400
levels = (("easy", 0.0), ("medium", 0.5), ("hard", 1.0))
seeds = (3, 8, 21, 34, 5)  # four tiles, different side styles
sheet = Image.new("RGB", (3 * W, 50 + (len(seeds) + 1) * H), (20, 24, 30))
dr = ImageDraw.Draw(sheet)
for k, (name, d) in enumerate(levels):
    dr.text((k * W + 10, 12), name, fill=(235, 235, 235), font=font)
    for i, sd in enumerate(seeds + (seeds[0],)):
        g, o, t = G.log_corridors(d, seed=sd)
        for m in g + o:
            m.apply_translation((3.4, 0, 0))
        t = [(x + 3.4, y, l) for (x, y, l) in t]
        cam = [(3.4, -0.01, 15.5), (3.4, 0.0, 0.0), (1, 0, 0)] if i == len(seeds) else [(-2.2, -0.35, 3.4), (3.6, 0.0, 0.0), (0, 0, 1)]
        sheet.paste(shot(g, o, t, cam, (W, H)), (k * W, 50 + i * H))
sheet.save(OUT / "log_corridor_sheet.jpg", quality=90)
print("done")
