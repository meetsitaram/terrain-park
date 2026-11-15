"""Camera helpers: set a view (target, eye direction, extent) and save PNGs into build/shots.
Coordinates in inches (model frame); Fusion camera in cm."""
import os

import adsk.core

import fxlib as fx
import params as P

SHOTS = os.path.join(fx.REPO, 'build', 'shots')

# named views matching the spec pictures (target in inches, eye direction, extent in inches, perspective?)
VIEWS = {
    'overview': ([216, 48, 30], [-0.55, -1.0, 0.75], 300, False),    # p.2 overview: from the front, above
    'overview_persp': ([216, 48, 20], [-0.55, -1.0, 0.75], 420, True),
    'start': ([60, 48, 50], [-1.0, -0.02, 0.05], 120, True),         # 'view from the starting location'
    'door': ([48, 48, 50], [-1.0, -0.25, 0.12], 130, False),
    'tasks_s2': ([70, 50, 30], [0.6, -1.0, 0.5], 110, False),
    'ramps': ([144, 48, 10], [-0.6, -1.0, 0.8], 140, False),
    'counter': ([260, 80, 50], [0.0, -1.0, 0.25], 120, False),        # demonstration counter
    'stairs': ([350, 40, 40], [-0.8, -1.0, 0.6], 170, False),
    'top': ([216, 48, 0], [0, 0, 1], 460, False),
    'task_bulb': ([60, 82.5, 34], [1.0, -0.15, 0.08], 44, False),
    'task_valve': ([243, 80, 34], [-1.0, -0.15, 0.08], 44, False),
    'task_button': ([134, 48, 34], [-1.0, -0.15, 0.08], 44, False),
    'task_keypad': ([70, 15, 50], [-1.0, -0.1, 0.05], 30, False),
    'stairs_close': ([310, 22, 25], [-0.3, -1.0, 0.5], 70, False),
    'bulbs': ([56, 82, 30], [1.0, -0.35, 0.3], 70, False),
    'ramps_side': ([144, 48, 5], [0.0, -1.0, 0.25], 110, False),
}


def camera(target, eye_dir, extent, persp=False):
    """Set type and field first, assign, then eye/target/up and assign again: Fusion moves the eye when the field
    changes (LESSONS.md)."""
    vp = fx.app.activeViewport
    t = [c * P.IN / 10 for c in target]
    d = fx.unit(eye_dir)
    dist = extent * P.IN / 10 * (1.6 if persp else 3.0)
    cam = vp.camera
    cam.isFitView = False
    cam.cameraType = (adsk.core.CameraTypes.PerspectiveCameraType if persp
                      else adsk.core.CameraTypes.OrthographicCameraType)
    if persp:
        cam.perspectiveAngle = 0.75
    vp.camera = cam
    adsk.doEvents()
    cam = vp.camera
    cam.target = adsk.core.Point3D.create(*t)
    cam.eye = adsk.core.Point3D.create(*[t[i] + d[i] * dist for i in range(3)])
    up = [0, 0, 1] if abs(d[2]) < 0.99 else [0, 1, 0]
    cam.upVector = adsk.core.Vector3D.create(*up)
    if not persp:
        cam.viewExtents = extent * P.IN / 10
    vp.camera = cam
    vp.refresh()
    return vp


def settle(seconds=1.5):
    """Let Fusion finish drawing (textures load asynchronously; an early save shows flat grey bodies)."""
    import time
    t0 = time.time()
    while time.time() - t0 < seconds:
        adsk.doEvents()
        time.sleep(0.05)
    fx.app.activeViewport.refresh()
    adsk.doEvents()


def show(hide=()):
    """Show every stage component except those whose name (or 'stage/unit' path) starts with one of `hide`."""
    fx.init()
    top = fx.top_occurrence(create=False)
    for o in top.component.occurrences:
        o.isLightBulbOn = not any(o.component.name.startswith(h) for h in hide)
        for u in o.component.occurrences:
            u.isLightBulbOn = not any((o.component.name + '/' + u.component.name).startswith(h) for h in hide)


def shot(name, view=None, w=1600, h=900, hide=('14 Gantry',)):
    fx.init()
    show(hide)
    os.makedirs(SHOTS, exist_ok=True)
    target, d, ext, persp = VIEWS[view or name]
    vp = camera(target, d, ext, persp)
    settle()
    path = os.path.join(SHOTS, name + '.png')
    vp.saveAsImageFile(path, w, h)
    return path


# ---------------------------------------------------------------- spec-matched cameras (cad/tools/layoutfit.py)

def _cams():
    import json
    return json.load(open(os.path.join(fx.REPO, 'cad', 'fusion', 'cameras.json'), encoding='utf-8'))


def spec_camera(name, angle=None, ext_scale=1.0):
    """Perspective camera from cameras.json (eye/fwd/up in inches, focal in px of the spec picture). The target
    goes on the view axis at the scene centre's depth; Fusion's viewExtents (not perspectiveAngle) sets what
    is visible, so it is set to the spec picture's half-width at that depth (ext_scale to fine-tune)."""
    import math
    fx.init()
    c = _cams()[name]
    vp = fx.app.activeViewport
    k = P.IN / 10
    centre = [216.0, 48.0, 30.0]
    dist = sum((centre[i] - c['eye'][i]) * c['fwd'][i] for i in range(3))
    eye = [v * k for v in c['eye']]
    tgt = [(c['eye'][i] + c['fwd'][i] * dist) * k for i in range(3)]
    # Fusion's perspectiveAngle is the VERTICAL field of view of the viewport, and changing it moves the eye
    # along the view axis (it keeps the visible extents): set the angle first, then eye/target (LESSONS.md).
    # The viewport width shows the spec picture's full width.
    half_w = c['size'][0] / 2 / c['focal_px']                       # tan of the spec half-width angle
    half_h = c['size'][1] / 2 / c['focal_px']
    vfov = 2 * math.atan(max(half_w * vp.height / vp.width, half_h) * ext_scale)   # cover the whole spec picture
    cam = vp.camera
    cam.isFitView = False
    cam.cameraType = adsk.core.CameraTypes.PerspectiveCameraType
    cam.perspectiveAngle = angle if angle is not None else vfov
    vp.camera = cam
    adsk.doEvents()
    cam = vp.camera
    cam.eye = adsk.core.Point3D.create(*eye)
    cam.target = adsk.core.Point3D.create(*tgt)
    cam.upVector = adsk.core.Vector3D.create(*c['up'])
    vp.camera = cam
    vp.refresh()
    return vp, c


def probe(name, pts, angle=None):
    """Where Fusion puts model points (inches) in the viewport, and the viewport size: for fov calibration."""
    vp, c = spec_camera(name, angle)
    settle(1.0)
    vp = fx.app.activeViewport
    out = []
    for p in pts:
        q = vp.modelToViewSpace(adsk.core.Point3D.create(*[v * P.IN / 10 for v in p]))
        out.append((round(q.x, 1), round(q.y, 1)))
    return dict(w=vp.width, h=vp.height, pts=out, angle=vp.camera.perspectiveAngle)


def spec_shot(name, w=1500, h=None, hide=(), angle=None, ext_scale=1.0):
    """Render through the spec camera; the image keeps the viewport aspect (crop to the spec picture afterwards
    with tools/compare.py)."""
    show(hide)
    render_style()
    vp, c = spec_camera(name, angle, ext_scale)
    settle()
    hh = int(round(w * vp.height / vp.width))
    path = os.path.join(SHOTS, 'spec_%s.png' % name)
    os.makedirs(SHOTS, exist_ok=True)
    vp.saveAsImageFile(path, w, hh)
    return path, vp.width, vp.height


def render_style(env='PhotoBooth', style='ShadedWithVisibleEdgesOnly'):
    """Lighting environment + visual style for the pictures. The spec renders are SketchUp-like: flat light and
    dark outlines, so 'shaded with visible edges only' under a soft environment is the closest."""
    fx.init()
    fx.app.lightingEnvironment = getattr(adsk.core.LightingEnvironments, env + 'LightingEnvironment')
    fx.app.activeViewport.visualStyle = getattr(adsk.core.VisualStyles, style + 'VisualStyle')
    return env, style
