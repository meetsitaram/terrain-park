"""Side-by-side and blended comparison of a Fusion spec-camera render with the spec picture.

The render keeps the viewport aspect with Fusion's perspectiveAngle = the spec picture's horizontal field of view,
so: scale the render width to the spec width, crop the centre band to the spec height.
Run: python cad/tools/compare.py overview   -> build/compare_overview.png (spec / ours / 50-50 blend)
"""
import json
import os
import sys

from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..', '..')


def crop_render(name):
    """Scale the render so 1 px = 1 spec px (both are pinhole images about the same axis), crop the spec frame."""
    cam = json.load(open(os.path.join(HERE, '..', 'fusion', 'cameras.json'), encoding='utf-8'))[name]
    W, H = cam['size']
    f = cam['focal_px']
    r = Image.open(os.path.join(ROOT, 'build', 'shots', 'spec_%s.png' % name)).convert('RGB')
    rw, rh = r.size
    tan_v = max(W / 2 / f * rh / rw, H / 2 / f)                      # same rule as view.spec_camera
    s = f * tan_v / (rh / 2)
    r = r.resize((int(round(rw * s)), int(round(rh * s))), Image.LANCZOS)
    left, top = (r.size[0] - W) // 2, (r.size[1] - H) // 2
    return r.crop((left, top, left + W, top + H))


def main(name):
    cam = json.load(open(os.path.join(HERE, '..', 'fusion', 'cameras.json'), encoding='utf-8'))[name]
    spec = Image.open(os.path.join(ROOT, 'build', 'spec', cam['image'])).convert('RGB')
    ours = crop_render(name)
    blend = Image.blend(spec, ours, 0.5)
    W, H = spec.size
    out = Image.new('RGB', (W, 3 * H + 60), 'white')
    d = ImageDraw.Draw(out)
    for k, (im, lab) in enumerate(((spec, 'spec (NIST rules PDF)'), (ours, 'our Fusion model, same camera'),
                                   (blend, '50/50 blend'))):
        out.paste(im, (0, k * (H + 20) + 20))
        d.text((6, k * (H + 20) + 4), lab, fill=(0, 0, 0))
    p = os.path.join(ROOT, 'build', 'compare_%s.png' % name)
    out.save(p)
    ours.save(os.path.join(ROOT, 'build', 'ours_%s.png' % name))
    return p


if __name__ == '__main__':
    print(main(sys.argv[1] if len(sys.argv) > 1 else 'overview'))
