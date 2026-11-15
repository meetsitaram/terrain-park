"""Export the apparatus (cad/model.py) to a MuJoCo MJCF: sim/apparatus.xml (+ meshes/).

- Static pieces -> geoms on one fixed body 'apparatus' (welded to the world).
- Moving parts -> bodies with hinge / slide joints from model.joints (springs, damping, ranges); a part whose
  joint has a 'parent' part hangs under that part's body (door lever under the door leaf).
- '(free)' objects (totes, crates, cylinders, drill, loose bulbs) -> free bodies.
- Box cuts aligned with the piece's own axes are cut exactly (the box is split into boxes); other cuts (ramp
  bevels, stair notches, rib gables) become meshes via trimesh/manifold; a concave result is visual-only and
  the parts around it carry the contact (treads, risers, skins).
- Units: metres. Frame: model frame (X along the lane, Z up, floor 0) unless --frame is given: the export can
  place the apparatus so a spawn point lands at the world origin facing a heading (robot runs).

Run: python sim/export_mjcf.py [--spawn X Y] [--heading DEG] [--deck-at-zero] [--out file.xml]
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cad'))
import model as M
import params as P

IN = P.IN / 1000.0                                                   # inches -> metres
TEX = os.path.join(HERE, '..', 'cad', 'fusion', 'textures')

DENSITY = dict(softwood=450.0, plywood=560.0, osb=620.0, hardboard=900.0, particleboard=680.0, hardwood=700.0,
               petg=1270.0)
# look -> (rgba, texture png or None)
LOOK = {
    'pine': ((0.86, 0.72, 0.52, 1), None), 'ply': ((0.85, 0.75, 0.56, 1), 'plywood.png'),
    'osb': ((0.80, 0.65, 0.42, 1), 'osb.png'), 'birch': ((0.88, 0.80, 0.64, 1), 'plywood.png'),
    'countertop': ((0.45, 0.44, 0.43, 1), 'countertop.png'), 'stain cedar 2': ((0.72, 0.42, 0.20, 1), 'stain_cedar_2.png'),
    'paint white': ((0.95, 0.95, 0.95, 1), None), 'door grey': ((0.55, 0.56, 0.58, 1), None),
    'satin nickel': ((0.75, 0.75, 0.76, 1), None), 'galvanized': ((0.70, 0.71, 0.72, 1), None),
    'brass': ((0.85, 0.68, 0.30, 1), None), 'red vinyl': ((0.80, 0.10, 0.10, 1), None),
    'porcelain': ((0.95, 0.94, 0.90, 1), None), 'bulb': ((0.97, 0.97, 0.94, 1), None),
    'petg': ((0.92, 0.92, 0.92, 1), None), 'print green': ((0.25, 0.68, 0.29, 1), None),
    'white plastic': ((0.96, 0.96, 0.96, 1), None), 'black plastic': ((0.08, 0.08, 0.08, 1), None),
    'grey plastic': ((0.55, 0.56, 0.57, 1), None), 'button yellow': ((0.95, 0.82, 0.10, 1), None),
    'display red': ((0.85, 0.10, 0.10, 1), None), 'clear plastic': ((0.85, 0.90, 0.95, 0.35), None),
    'crate blue': ((0.10, 0.25, 0.77, 1), None), 'drill yellow': ((0.95, 0.72, 0.02, 1), None),
    'rope': ((0.95, 0.95, 0.90, 1), None), 'paint blue': ((0.16, 0.25, 0.72, 1), None),
    'paint yellow': ((0.95, 0.85, 0.10, 1), None), 'paint purple': ((0.42, 0.18, 0.69, 1), None),
    'paint orange': ((0.94, 0.49, 0.10, 1), None), 'hard': ((0.54, 0.42, 0.28, 1), None),
    'fastener': ((1.0, 0.45, 0.05, 1), None),                     # hardware highlighted in build animations
}
FRICTION = '0.8 0.02 0.001" solref="0.005 1" solimp="0.95 0.99 0.001'   # stiff wood: feet sink < 3 mm


def quat_from_axes(ax):
    R = np.array(ax, float).T                                       # columns = local axes in world
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def right_handed(ax):
    a = [np.array(v, float) / np.linalg.norm(v) for v in ax]
    if np.dot(np.cross(a[0], a[1]), a[2]) < 0:
        a[2] = -a[2]
    return a


def aligned(sh, cut):
    """True if the cut box's axes are parallel to the shape box's axes."""
    A = [np.array(v) / np.linalg.norm(v) for v in sh[2]]
    B = [np.array(v) / np.linalg.norm(v) for v in cut[2]]
    return all(max(abs(np.dot(a, b)) for b in B) > 0.9999 for a in A)


def split_box(sh, cuts):
    """Exact box minus aligned boxes -> list of boxes (local slab decomposition)."""
    _, c, ax, size = sh
    A = [np.array(v) / np.linalg.norm(v) for v in ax]
    c = np.array(c, float)
    lo, hi = -np.array(size) / 2, np.array(size) / 2
    boxes = [(lo.copy(), hi.copy())]
    for cut in cuts:
        cc = np.array(cut[1], float)
        B = [np.array(v) / np.linalg.norm(v) for v in cut[2]]
        half = np.zeros(3)
        for i in range(3):                                           # cut extent along each local axis
            half[i] = sum(abs(np.dot(A[i], B[j])) * cut[3][j] / 2 for j in range(3))
        ctr = np.array([np.dot(cc - c, A[i]) for i in range(3)])
        clo, chi = ctr - half, ctr + half
        out = []
        for blo, bhi in boxes:
            if np.any(chi <= blo) or np.any(clo >= bhi):
                out.append((blo, bhi))
                continue
            rlo, rhi = blo.copy(), bhi.copy()
            for i in range(3):                                       # peel slabs off the outside of the cut
                if clo[i] > rlo[i]:
                    a, b = rlo.copy(), rhi.copy()
                    b[i] = clo[i]
                    out.append((a, b))
                    rlo[i] = clo[i]
                if chi[i] < rhi[i]:
                    a, b = rlo.copy(), rhi.copy()
                    a[i] = chi[i]
                    out.append((a, b))
                    rhi[i] = chi[i]
        boxes = [(a, b) for a, b in out if np.all(b - a > 1e-4)]
    res = []
    for blo, bhi in boxes:
        mid = (blo + bhi) / 2
        res.append(('box', list(c + sum(mid[i] * A[i] for i in range(3))), ax, tuple(bhi - blo)))
    return res


def shape_mesh(sh):
    import trimesh
    if sh[0] == 'box':
        _, c, ax, size = sh
        T = np.eye(4)
        T[:3, :3] = np.array(right_handed(ax)).T
        T[:3, 3] = c
        return trimesh.creation.box(extents=size, transform=T)
    if sh[0] == 'cyl':
        _, a, b, r0, r1 = sh
        a, b = np.array(a), np.array(b)
        r = (r0 + r1) / 2
        return trimesh.creation.cylinder(radius=r, segment=[a, b], sections=24)
    _, c, r = sh
    return trimesh.creation.icosphere(radius=r, subdivisions=2).apply_translation(c)


def piece_mesh(p):
    import trimesh
    m = trimesh.boolean.union([shape_mesh(s) for s in p.shapes]) if len(p.shapes) > 1 else shape_mesh(p.shapes[0])
    for c in p.cuts:
        m = trimesh.boolean.difference([m, shape_mesh(c)])
    return m


TWIN_DEPTH = 0.04          # m: thin walkable boards get a collision twin this much thicker, downwards
THIN = 0.025               # m: half-thickness below which a board is "thin"


def twin_box(sh):
    """Thin, roughly horizontal box (walkable) -> a thicker box with the same top face, extended downwards, or None.
    Robots' 5 mm foot spheres tunnel into a 19 mm board at a 5 ms step and get pushed out through the bottom face
    (LESSONS.md); a thick twin behaves like a plane."""
    _, c, ax, size = sh
    A = [np.array(v, float) / np.linalg.norm(v) for v in ax]
    k = int(np.argmin(size))
    if size[k] * IN / 2 >= THIN or abs(A[k][2]) < 0.5:
        return None
    up = A[k] if A[k][2] > 0 else -A[k]
    extra = TWIN_DEPTH / IN                                          # inches
    nc = np.array(c, float) - up * extra / 2
    ns = list(size)
    ns[k] = size[k] + extra
    return ('box', list(nc), ax, tuple(ns))


class Writer:
    def __init__(self, xform):
        self.xform = xform                                            # (R 3x3, t 3) applied to model points (metres)
        self.assets = []
        self.meshes = 0
        self.mats = {}

    def mat(self, look):
        if look in self.mats:
            return self.mats[look]
        rgba, tex = LOOK.get(look, ((0.6, 0.6, 0.6, 1), None))
        name = 'm_' + look.replace(' ', '_')
        if tex:
            self.assets.append('<texture name="t_%s" type="2d" file="%s"/>' % (name, os.path.join(TEX, tex).replace('\\', '/')))
            self.assets.append('<material name="%s" texture="t_%s" texuniform="true" texrepeat="1.6 1.6" rgba="1 1 1 %g"/>'
                               % (name, name, rgba[3]))
        else:
            self.assets.append('<material name="%s" rgba="%g %g %g %g"/>' % ((name,) + tuple(rgba)))
        self.mats[look] = name
        return name

    def pt(self, p, frame_origin):
        """model point (inches) -> body-local metres (body at frame_origin, same orientation)."""
        return np.array(p, float) * IN - frame_origin

    def geoms(self, p, origin, collide=True):
        mat = self.mat(p.look)
        vol = max(volume_of(p), 1e-9)
        dens = (p.mass / vol) if p.mass else DENSITY.get(p.mat or 'softwood', 500.0)
        dn = ' density="%.1f"' % dens
        out = []
        aligned_cuts = all(any(aligned(s, c) for s in p.shapes if s[0] == 'box') for c in p.cuts) if p.cuts else True
        shapes = p.shapes
        if p.cuts and aligned_cuts and all(s[0] == 'box' for s in p.shapes):
            shapes = [b for s in p.shapes for b in split_box(s, p.cuts)]
        elif p.cuts:
            tw = twin_box(p.shapes[0]) if (collide and p.shapes[0][0] == 'box') else None
            if tw is None:
                return self.mesh_geom(p, origin, mat, collide, dn)
            _, tc, tax, tsz = tw                                     # cut sheet (ramp skin): mesh looks, box contact
            return self.mesh_geom(p, origin, mat, False, dn) + [
                '<geom type="box" size="%.5f %.5f %.5f" pos="%s" quat="%s" group="3" rgba="0 0 0 0" friction="%s" '
                'density="0"/>' % (tsz[0] / 2 * IN, tsz[1] / 2 * IN, tsz[2] / 2 * IN, fmt(self.pt(tc, origin)),
                                   fmt(quat_from_axes(right_handed(tax))), FRICTION)]
        con = '' if collide else ' contype="0" conaffinity="0"'
        for s in shapes:
            if s[0] == 'box':
                tw = twin_box(s) if collide else None
                if tw is not None:                                   # collision twin (hidden group 3), visual-only board
                    _, tc, tax, tsz = tw
                    out.append('<geom type="box" size="%.5f %.5f %.5f" pos="%s" quat="%s" group="3" rgba="0 0 0 0" '
                               'friction="%s" density="0"/>'
                               % (tsz[0] / 2 * IN, tsz[1] / 2 * IN, tsz[2] / 2 * IN, fmt(self.pt(tc, origin)),
                                  fmt(quat_from_axes(right_handed(tax))), FRICTION))
                    con_s = ' contype="0" conaffinity="0"'
                else:
                    con_s = con
                _, c, ax, size = s
                q = quat_from_axes(right_handed(ax))
                out.append('<geom type="box" size="%.5f %.5f %.5f" pos="%s" quat="%s" material="%s" friction="%s"%s%s/>'
                           % (size[0] / 2 * IN, size[1] / 2 * IN, size[2] / 2 * IN, fmt(self.pt(c, origin)), fmt(q),
                              mat, FRICTION, con_s, dn))
            elif s[0] == 'cyl':
                _, a, b, r0, r1 = s
                if np.linalg.norm(np.subtract(b, a)) < 1e-6:
                    continue
                out.append('<geom type="cylinder" fromto="%s %s" size="%.5f" material="%s" friction="%s"%s%s/>'
                           % (fmt(self.pt(a, origin)), fmt(self.pt(b, origin)), (r0 + r1) / 2 * IN, mat, FRICTION, con,
                              dn))
            else:
                _, c, r = s
                out.append('<geom type="sphere" pos="%s" size="%.5f" material="%s"%s%s/>'
                           % (fmt(self.pt(c, origin)), r * IN, mat, con, dn))
        return out

    def mesh_geom(self, p, origin, mat, collide, dn=''):
        m = piece_mesh(p)
        m.apply_scale(IN)
        m.apply_translation(-origin)
        name = 'mesh%d' % self.meshes
        self.meshes += 1
        os.makedirs(os.path.join(HERE, 'meshes'), exist_ok=True)
        f = os.path.join(HERE, 'meshes', name + '.stl')
        m.export(f)
        self.assets.append('<mesh name="%s" file="%s"/>' % (name, f.replace('\\', '/')))
        hull_ratio = m.volume / max(m.convex_hull.volume, 1e-12)
        con = '' if (collide and hull_ratio > 0.97) else ' contype="0" conaffinity="0"'
        return ['<geom type="mesh" mesh="%s" material="%s" friction="%s"%s%s/>' % (name, mat, FRICTION, con, dn)]


def fmt(v):
    return ' '.join('%.5f' % x for x in v)


def volume_of(p):
    vol = 0.0
    for s in p.shapes:
        if s[0] == 'box':
            vol += np.prod(s[3])
        elif s[0] == 'cyl':
            vol += math.pi * ((s[3] + s[4]) / 2) ** 2 * np.linalg.norm(np.subtract(s[2], s[1]))
        else:
            vol += 4 / 3 * math.pi * s[2] ** 3
    return vol * IN ** 3


def mass_of(p):
    if p.mass:
        return p.mass
    vol = 0.0
    for s in p.shapes:
        if s[0] == 'box':
            vol += np.prod(s[3])
        elif s[0] == 'cyl':
            vol += math.pi * ((s[3] + s[4]) / 2) ** 2 * np.linalg.norm(np.subtract(s[2], s[1]))
        else:
            vol += 4 / 3 * math.pi * s[2] ** 3
    return vol * (IN ** 3) * DENSITY.get(p.mat or 'softwood', 500.0)


def export(path, poses=None, frame=(0.0, 0.0, 0.0, 0.0), floor=True, skip=()):
    """frame = (x, y, z, yaw_deg): where the model origin sits in the world (metres, degrees)."""
    m = M.build(stages=[st for st, _ in M.STAGES if st not in skip], poses=poses or P.POSE)
    w = Writer(None)
    by_part = defaultdict(list)
    free = defaultdict(list)
    static = []
    for p in m.pieces:
        if '(free)' in p.group:
            free[p.group].append(p)
        elif p.part:
            by_part[p.part].append(p)
        else:
            static.append(p)
    joints = {j['part']: j for j in m.joints if j['type'] in ('hinge', 'slide', 'screw')}
    children = defaultdict(list)
    for part in by_part:
        j = joints.get(part)
        children[j.get('parent') if j else None].append(part)

    def body(part, parent_origin):
        j = joints.get(part)
        anchor = np.array(j['anchor'], float) * IN if j else np.array(by_part[part][0].shapes[0][1]) * IN
        lines = ['<body name="%s" pos="%s">' % (part.replace(' ', '_'), fmt(anchor - parent_origin))]
        if j:
            ax = np.array(j['axis'], float)
            ax /= np.linalg.norm(ax)
            spring = j.get('spring') or {}
            if j['type'] in ('hinge', 'screw'):
                lo, hi = j.get('range', (0, 90))
                k = spring.get('k_nm_per_rad', 0.0)
                pre = spring.get('preload_nm', 0.0)
                ref = -pre / k if k else 0.0                      # spring pulls to below 0 -> preload at closed
                fr = j.get('friction_nm', 0.05)
                lines.append('<joint name="%s" type="hinge" axis="%s" range="%g %g" stiffness="%g" springref="%g" '
                             'damping="%g" frictionloss="%g" limited="true"/>'
                             % (part.replace(' ', '_'), fmt(ax), math.radians(lo), math.radians(hi), k, ref,
                                j.get('damping', 0.05), fr))
            else:
                lo, hi = j.get('range', (0, 0.1))
                k = spring.get('k_n_per_m', 0.0)
                pre = spring.get('preload_n', 0.0)
                lines.append('<joint name="%s" type="slide" axis="%s" range="%g %g" stiffness="%g" springref="%g" '
                             'damping="%g" limited="true"/>'
                             % (part.replace(' ', '_'), fmt(ax), lo * IN, hi * IN, k, -pre / k if k else 0.0,
                                j.get('damping', 2.0)))
        tot = sum(mass_of(p) for p in by_part[part])
        for p in by_part[part]:
            lines += w.geoms(p, anchor)
        for ch in children.get(part, []):
            lines += body(ch, anchor)
        lines.append('</body>')
        return lines

    x, y, z, yaw = frame
    root = ['<body name="apparatus" pos="%g %g %g" euler="0 0 %g">' % (x, y, z, math.radians(yaw))]
    for p in static:
        root += w.geoms(p, np.zeros(3), collide='(buy)' not in p.group or True)
    for part in children.get(None, []):
        root += body(part, np.zeros(3))
    root.append('</body>')
    # free objects must be top-level bodies: place them in world coordinates through the frame
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    for g, ps in free.items():
        c = np.array(ps[0].shapes[0][1], float) * IN
        cw = np.array([x + cy * c[0] - sy * c[1], y + sy * c[0] + cy * c[1], z + c[2]])
        tot = sum(mass_of(p) for p in ps)
        root.append('<body name="%s" pos="%s" euler="0 0 %g"><freejoint/>'
                    % (g.split('/')[1].replace(' ', '_').replace('(', '').replace(')', ''), fmt(cw), math.radians(yaw)))
        for p in ps:
            root += w.geoms(p, c)
        root.append('</body>')
    floor_z = 0.0 if not floor else 0.0
    xml = ['<mujoco model="loco-manip apparatus">', '<compiler angle="radian" autolimits="true"/>',
           '<option timestep="0.002" integrator="implicitfast"/>',
           '<visual><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>'
           '<quality shadowsize="8192"/><map znear="0.01"/></visual>',
           '<asset>', '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>',
           '<texture name="floor" type="2d" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>',
           '<material name="floor" texture="floor" texuniform="true" texrepeat="5 5" reflectance="0.2"/>'] + w.assets + ['</asset>',
           '<worldbody>',
           '<light pos="5 -5 12" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>',
           '<light pos="12 8 10" dir="-0.4 -0.3 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>']
    if floor:
        xml.append('<geom name="venue_floor" type="plane" size="40 40 0.1" pos="0 0 %g" material="floor"/>' % (z if z else floor_z))
    excl = ['<exclude body1="apparatus" body2="%s"/>' % part.replace(' ', '_') for part in by_part]
    for part in by_part:                                             # and a child part never hits its parent part
        j = joints.get(part)
        if j and j.get('parent'):
            excl.append('<exclude body1="%s" body2="%s"/>' % (j['parent'].replace(' ', '_'), part.replace(' ', '_')))
    xml += root + ['</worldbody>', '<contact>'] + excl + ['</contact>', '</mujoco>']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml))
    return path, len(static), len(by_part), len(free), w.meshes


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'apparatus.xml'))
    ap.add_argument('--spec-pose', action='store_true', help='door/gates/cabinets as in the spec pictures')
    a = ap.parse_args()
    print(export(a.out, poses=P.SPEC_POSE if a.spec_pose else None))
