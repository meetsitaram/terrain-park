"""Pixel-grid crops of a spec picture, for reading off point coordinates by eye (step 1 of the spec-camera fit).

Splits the picture into overlapping vertical strips, upsamples each 2x and draws a labelled grid: thin blue lines
every 25 px, red every 100 px, labels in original-image pixels. Open a strip, find a feature with a known model
position (post top, deck corner, guard corner), read its pixel from the grid, and add the pair to camfit.CAMS.

Run: python cad/tools/grid.py build/spec/img_p2_x29.png [--strips 3] [--step 25] [--scale 2]
     -> build/spec/grid_<name>_<i>.png
"""
import argparse
import os

from PIL import Image, ImageDraw


def grid(path, strips=3, step=25, scale=2, overlap=90):
    im = Image.open(path).convert('RGB')
    W, H = im.size
    w = (W + overlap * (strips - 1)) // strips
    name = os.path.splitext(os.path.basename(path))[0]
    out = []
    for i in range(strips):
        x0 = max(0, min(W - w, i * (w - overlap)))
        x1 = x0 + w
        c = im.crop((x0, 0, x1, H)).resize((w * scale, H * scale), Image.LANCZOS)
        d = ImageDraw.Draw(c)
        for gx in range(x0 - x0 % step, x1, step):
            X = (gx - x0) * scale
            d.line([(X, 0), (X, H * scale)], fill=(255, 0, 0) if gx % 100 == 0 else (0, 160, 255), width=1)
            if gx % (2 * step) == 0:
                d.text((X + 2, 2), str(gx), fill=(255, 0, 0))
        for gy in range(0, H, step):
            Y = gy * scale
            d.line([(0, Y), (c.size[0], Y)], fill=(255, 0, 0) if gy % 100 == 0 else (0, 160, 255), width=1)
            d.text((2, Y + 2), str(gy), fill=(255, 0, 0))
        p = os.path.join(os.path.dirname(path), 'grid_%s_%d.png' % (name, i))
        c.save(p)
        out.append(p)
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('image')
    a.add_argument('--strips', type=int, default=3)
    a.add_argument('--step', type=int, default=25)
    a.add_argument('--scale', type=int, default=2)
    o = a.parse_args()
    for p in grid(o.image, o.strips, o.step, o.scale):
        print(p)
