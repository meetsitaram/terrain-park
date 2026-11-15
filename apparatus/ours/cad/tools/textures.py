"""Procedural tileable textures for the Fusion appearances (OSB, sanded plywood, laminate countertop).

Run with the desktop Python: writes cad/fusion/textures/*.png. Each image covers TILE_IN x TILE_IN inches of the
real surface (set as the appearance's real-world scale in apparatus.py).
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.join(os.path.dirname(__file__), '..', 'fusion', 'textures')
N = 1024
TILE_IN = 24.0


def tileable_noise(n, scale, seed):
    """Smooth tileable value noise in [0, 1] (bilinear upsample of a random grid, wrapped)."""
    rng = np.random.default_rng(seed)
    g = rng.random((scale, scale))
    img = Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return a


def osb(seed=1):
    """Oriented strand board: overlapping elongated flakes, mostly along one direction, warm tan tones."""
    rng = random.Random(seed)
    base = Image.new('RGB', (N, N), (196, 160, 104))
    d = ImageDraw.Draw(base)
    px_in = N / TILE_IN
    for _ in range(5200):
        L = rng.uniform(2.0, 5.0) * px_in                         # flakes 2-5 in long, ~1 in wide
        W = rng.uniform(0.4, 1.1) * px_in
        ang = rng.gauss(0, 18) + (90 if rng.random() < 0.25 else 0)
        cx, cy = rng.uniform(0, N), rng.uniform(0, N)
        tone = rng.gauss(0, 1)
        k = 12 * tone + rng.uniform(-6, 6)
        col = (int(max(150, min(230, 204 + k))), int(max(115, min(195, 166 + k * 0.95))),
               int(max(70, min(140, 108 + k * 0.8))))
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        pts = []
        for u, v in ((-L / 2, -W / 2), (L / 2, -W / 2 * rng.uniform(0.5, 1)), (L / 2, W / 2), (-L / 2, W / 2 * rng.uniform(0.5, 1))):
            pts.append((cx + u * ca - v * sa, cy + u * sa + v * ca))
        for ox in (-N, 0, N):                                       # draw wrapped copies: tileable
            for oy in (-N, 0, N):
                q = [(x + ox, y + oy) for x, y in pts]
                if all(x < -50 or x > N + 50 for x, _ in q) or all(y < -50 or y > N + 50 for _, y in q):
                    continue
                d.polygon(q, fill=col, outline=(col[0] - 22, col[1] - 24, col[2] - 22))
    img = base.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(img, dtype=np.float32)
    a *= (0.92 + 0.12 * tileable_noise(N, 8, seed))[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def plywood(seed=2):
    """Sanded pine/birch face veneer: pale, long soft grain with a few darker growth lines."""
    rng = np.random.default_rng(seed)
    y = np.arange(N)[:, None] / N
    x = np.arange(N)[None, :] / N
    warp = 0.18 * tileable_noise(N, 4, seed) + 0.05 * tileable_noise(N, 12, seed + 7)
    t = (x + warp) * 14
    lines = 0.5 + 0.5 * np.sin(2 * math.pi * t)
    lines = lines ** 10
    n = tileable_noise(N, 48, seed + 1)
    v = 0.90 - 0.10 * lines - 0.05 * n
    r, g, b = 236 * v, 210 * v, 160 * v
    a = np.stack([r, g, b], -1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).rotate(90)


def countertop(seed=3):
    """Grey stone-look laminate."""
    n1 = tileable_noise(N, 64, seed)
    n2 = tileable_noise(N, 16, seed + 5)
    rng = np.random.default_rng(seed)
    speck = (rng.random((N, N)) > 0.985).astype(np.float32)
    v = 0.48 + 0.12 * n1 + 0.08 * n2 - 0.15 * speck
    a = np.stack([v * 200, v * 196, v * 192], -1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def stained_pine(seed=4, base=(205, 118, 52)):
    """Pine stained with BEHR PREMIUM ST-533 Cedar Naturaltone (semi-transparent): orange-brown, darker
    latewood lines along the board, a little blotch."""
    x = np.arange(N)[None, :] / N
    warp = 0.10 * tileable_noise(N, 4, seed) + 0.04 * tileable_noise(N, 16, seed + 3)
    t = (x + warp) * 18
    lines = (0.5 + 0.5 * np.sin(2 * math.pi * t)) ** 8
    n = tileable_noise(N, 40, seed + 1)
    v = 1.0 - 0.28 * lines - 0.08 * n
    a = np.stack([base[0] * v, base[1] * v, base[2] * v], -1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).rotate(90)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (('osb', osb), ('plywood', plywood), ('countertop', countertop), ('stain_cedar_2', stained_pine)):
        fn().save(os.path.join(OUT, name + '.png'))
        print('wrote', name)
