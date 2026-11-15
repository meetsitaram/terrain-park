"""Fusion-side helpers for the loco-manipulation apparatus. Inputs are repo millimetres.

Adapted from CDX-3R's fxlib.py: pure-Python math (no numpy/scipy inside Fusion) plus a Part
builder that turns parameters into Fusion timeline features (extrudes, lofts, pipes, revolves).
One Part = one component.

Frame: mm, X along the lane from the start, Y left, Z up, floor Z = 0. Fusion is Z-up too, so the
top component has no rotation; the only conversion is mm -> cm (P(), matrix()).
"""
import adsk.core
import adsk.fusion
import math
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # folder holding cad/ (apparatus/ours)
TOP = 'Apparatus'                                          # must differ from the document (root) name
VI = adsk.core.ValueInput
NEW = adsk.fusion.FeatureOperations.NewBodyFeatureOperation
JOIN = adsk.fusion.FeatureOperations.JoinFeatureOperation
CUT = adsk.fusion.FeatureOperations.CutFeatureOperation

IN = 25.4                                                  # mm per inch: cut lists are in inches, the model in mm

app = None
design = None


def init():
    global app, design
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    return design


# ---------------------------------------------------------------- vectors

def add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def norm(a): return math.sqrt(dot(a, a))
def unit(a): return mul(a, 1.0 / norm(a))
def mat_vec(R, v): return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {'x': [[1, 0, 0], [0, c, -s], [0, s, c]],
            'y': [[c, 0, s], [0, 1, 0], [-s, 0, c]],
            'z': [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]


def linspace(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [a]


def interp(x, xs, ys):
    """numpy.interp: clamps outside the table."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def frame_of(d):
    """(ex, ey, ez) with ez along d."""
    ez = unit(d)
    a = [1, 0, 0] if abs(ez[0]) < 0.9 else [0, 1, 0]
    ex = unit(cross(a, ez))
    return ex, cross(ez, ex), ez


# ---------------------------------------------------------------- Fusion basics

def P(p):
    return adsk.core.Point3D.create(p[0] / 10.0, p[1] / 10.0, p[2] / 10.0)


def matrix(R=None, t=(0, 0, 0)):
    m = adsk.core.Matrix3D.create()
    if R is not None:
        cols = [[R[0][j], R[1][j], R[2][j]] for j in range(3)]
        m.setWithCoordinateSystem(P(t), *[adsk.core.Vector3D.create(*c) for c in cols])
    else:
        m.translation = adsk.core.Vector3D.create(t[0] / 10.0, t[1] / 10.0, t[2] / 10.0)
    return m


def coll(items):
    c = adsk.core.ObjectCollection.create()
    for i in items:
        c.add(i)
    return c


def appearance(hexcol, base_prefix='Plastic - Matte'):
    """A flat-colour appearance (copy of a library appearance with its albedo replaced)."""
    name = 'lma ' + hexcol
    a = design.appearances.itemByName(name)
    if a:
        return a
    lib = app.materialLibraries.itemByName('Fusion Appearance Library')
    base = next(lib.appearances.item(i) for i in range(lib.appearances.count)
                if lib.appearances.item(i).name.startswith(base_prefix))
    a = design.appearances.addByCopy(base, name)
    h = hexcol.lstrip('#')
    prop = adsk.core.ColorProperty.cast(a.appearanceProperties.itemById('opaque_albedo'))
    prop.value = adsk.core.Color.create(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    return a


def library_appearance(name_prefix):
    """A textured appearance straight from the Fusion library (wood, OSB, metal...), copied into the design once."""
    a = design.appearances.itemByName(name_prefix)
    if a:
        return a
    lib = app.materialLibraries.itemByName('Fusion Appearance Library')
    src = next((lib.appearances.item(i) for i in range(lib.appearances.count)
                if lib.appearances.item(i).name.startswith(name_prefix)), None)
    if src is None:
        raise KeyError('no library appearance starting with %r' % name_prefix)
    return design.appearances.addByCopy(src, src.name)


def top_occurrence(create=True):
    """The one top component everything lives in. Z-up repo frame = Fusion frame: no rotation."""
    root = design.rootComponent
    for o in root.occurrences:
        if o.component.name == TOP:
            return o
    if not create:
        return None
    occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = TOP
    snapshot()
    occ.isGroundToParent = True
    return occ


def snapshot():
    if design.snapshots.hasPendingSnapshot:
        design.snapshots.add()


def group(parent, name, R=None, t=(0, 0, 0)):
    """An empty component used to hold parts, optionally in its own frame. Replaces one of the same name."""
    for o in list(parent.occurrences):
        if o.component.name == name:
            o.deleteMe()
    occ = parent.occurrences.addNewComponent(matrix(R, t))
    occ.component.name = name
    snapshot()
    return occ.component


AX = {'z': ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
      'x': ([0, 1, 0], [0, 0, 1], [1, 0, 0]),
      'y': ([1, 0, 0], [0, 0, 1], [0, 1, 0])}
BASES = (([0, 0, 1], 'xYConstructionPlane'), ([1, 0, 0], 'yZConstructionPlane'), ([0, 1, 0], 'xZConstructionPlane'))


def circle(r):
    return lambda sk, to: sk.sketchCurves.sketchCircles.addByCenterRadius(to(0, 0), r / 10.0)


def rect(u0, u1, v0, v1):
    return lambda sk, to: sk.sketchCurves.sketchLines.addTwoPointRectangle(to(u0, v0), to(u1, v1))


def poly(pts):
    """Closed polygon through (u, v) points."""
    def draw(sk, to):
        ln = sk.sketchCurves.sketchLines
        q = [to(*c) for c in pts]
        for i in range(len(q)):
            ln.addByTwoPoints(q[i], q[(i + 1) % len(q)])
    return draw


def hexagon(af):
    r = af / math.sqrt(3)                                  # circumradius of a hex with across-flats af
    return poly([(r * math.cos(math.radians(30 + 60 * i)), r * math.sin(math.radians(30 + 60 * i))) for i in range(6)])


# ---------------------------------------------------------------- Part builder

class Part:
    """One component. Coordinates are in the parent frame (mm)."""

    def __init__(self, parent, name, color=None, material=None):
        occ = parent.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        self.occ, self.comp = occ, occ.component
        self.comp.name = name
        self.color = color                                 # '#rrggbb' or a library appearance name prefix
        self.material = material                           # physical material name (mass, density) or None
        self._planes, self._refs, self._paths = {}, None, {}

    # -- planes and sketches
    def _hidden_sketch(self, plane, name):
        sk = self.comp.sketches.add(plane)
        sk.name = name
        sk.isVisible = False
        return sk

    def refs(self):
        if self._refs is None:
            self._refs = self._hidden_sketch(self.comp.xYConstructionPlane, 'refs')
        return self._refs

    def plane(self, origin, ez, ex):
        """A plane through origin with normal ez."""
        ez = unit(ez)
        for axis, base in BASES:
            if abs(abs(dot(ez, axis)) - 1) < 1e-9:
                bp = getattr(self.comp, base)
                g = bp.geometry.normal
                d = dot(origin, [g.x, g.y, g.z])
                key = (base, round(d, 4))
                if key not in self._planes:
                    if abs(d) < 1e-6:
                        self._planes[key] = bp
                    else:
                        ci = self.comp.constructionPlanes.createInput()
                        ci.setByOffset(bp, VI.createByString(f'{d:.4f} mm'))
                        pl = self.comp.constructionPlanes.add(ci)
                        pl.isLightBulbOn = False
                        self._planes[key] = pl
                return self._planes[key]
        ey = cross(ez, ex)
        rs = self.refs()
        pts = [origin, add(origin, mul(unit(ex), 20)), add(origin, mul(unit(ey), 20))]
        sps = [rs.sketchPoints.add(rs.modelToSketchSpace(P(p))) for p in pts]
        ci = self.comp.constructionPlanes.createInput()
        ci.setByThreePoints(*sps)
        pl = self.comp.constructionPlanes.add(ci)
        pl.isLightBulbOn = False
        return pl

    # -- extrusions
    def prism(self, origin, frame, z0, z1, draw, pick=None, name='prism', op=NEW, targets=None):
        """Sketch in the plane origin + z0*ez, extrude to z1 along ez. draw(sketch, to).
        op/targets: e.g. a cut feature (op=CUT) limited to the target bodies."""
        ex, ey, ez = [unit(v) for v in frame]
        o = add(origin, mul(ez, z0))
        pl = self.plane(o, ez, ex)
        sk = self._hidden_sketch(pl, name)
        to = lambda u, v: sk.modelToSketchSpace(P(add(o, add(mul(ex, u), mul(ey, v)))))
        draw(sk, to)
        profs = [sk.profiles.item(i) for i in range(sk.profiles.count)]
        if pick:
            profs = pick(profs)
        ei = self.comp.features.extrudeFeatures.createInput(coll(profs), op)
        if targets:
            ei.participantBodies = list(targets)
        zs = sk.transform.getAsCoordinateSystem()[3]
        direction = (adsk.fusion.ExtentDirections.PositiveExtentDirection if dot([zs.x, zs.y, zs.z], ez) > 0
                     else adsk.fusion.ExtentDirections.NegativeExtentDirection)
        ei.setOneSideExtent(adsk.fusion.DistanceExtentDefinition.create(VI.createByReal((z1 - z0) / 10.0)), direction)
        f = self.comp.features.extrudeFeatures.add(ei)
        f.name = name
        return f

    def box(self, x0, x1, y0, y1, z0, z1, name='box'):
        f = self.prism([0, 0, 0], AX['z'], z0, z1, rect(x0, x1, y0, y1), name=name)
        return list(f.bodies)

    def board(self, a, b, w, t, up=(0, 0, 1), name='board'):
        """A piece of lumber from end centre a to end centre b: cross-section w (along `up` x axis) by t (along `up`).
        E.g. a 2x4 joist on edge: w = 1.5 in, t = 3.5 in, up = Z."""
        d = sub(b, a)
        ez = unit(d)
        ey = unit(sub(list(up), mul(ez, dot(up, ez))))
        ex = cross(ey, ez)
        f = self.prism(a, (ex, ey, ez), 0, norm(d), rect(-w / 2, w / 2, -t / 2, t / 2), name=name)
        return list(f.bodies)

    def _frame(self, axis):
        return AX[axis] if isinstance(axis, str) else axis

    def cyl(self, axis, c, r, length, name='cylinder'):
        f = self.prism(c, self._frame(axis), -length / 2, length / 2, circle(r), name=name)
        return list(f.bodies)

    def ann(self, axis, c, ro, ri, length, chamfer=0.0, name='ring'):
        def draw(sk, to):
            sk.sketchCurves.sketchCircles.addByCenterRadius(to(0, 0), ro / 10.0)
            sk.sketchCurves.sketchCircles.addByCenterRadius(to(0, 0), ri / 10.0)
        f = self.prism(c, self._frame(axis), -length / 2, length / 2, draw,
                       lambda ps: [p for p in ps if p.profileLoops.count == 2], name=name)
        if chamfer:
            self.chamfer(f, chamfer)
        return list(f.bodies)

    def plate(self, outline, z0, thickness, bevel=0.0, origin=(0, 0, 0), frame=None, name='plate'):
        """Polygon outline (u, v) extruded from z0 along the frame's ez; optional chamfer on both faces."""
        f = self.prism(list(origin), frame or AX['z'], z0, z0 + thickness, poly(outline), name=name)
        if bevel:
            self.chamfer(f, bevel)
        return list(f.bodies)

    def chamfer(self, f, d):
        edges = [e for faces in (f.startFaces, f.endFaces) for face in faces for e in face.edges]
        ch = self.comp.features.chamferFeatures
        try:
            try:
                ci = ch.createInput2()
                ci.chamferEdgeSets.addEqualDistanceChamferEdgeSet(coll(edges), VI.createByReal(d / 10.0), True)
            except AttributeError:
                ci = ch.createInput(coll(edges), True)
                ci.setToEqualDistance(VI.createByReal(d / 10.0))
            ch.add(ci)
        except Exception:
            pass  # a chamfer that cannot be built is cosmetic; keep the solid

    # -- sweeps
    def paths(self):
        if 'xy' not in self._paths:
            self._paths['xy'] = self._hidden_sketch(self.comp.xYConstructionPlane, 'paths')
        return self._paths['xy']

    def _pipe(self, curve_entity, r, name):
        path = self.comp.features.createPath(curve_entity, False)
        pi = self.comp.features.pipeFeatures.createInput(path, NEW)
        pi.sectionSize = VI.createByReal(2 * r / 10.0)
        f = self.comp.features.pipeFeatures.add(pi)
        f.name = name
        return list(f.bodies)

    def pipe(self, points, r, name='pipe'):
        """Round rod along straight segments (a railing, a handle bar), joined into one body."""
        sk = self.paths()
        bodies = []
        for a, b in zip(points[:-1], points[1:]):
            if norm(sub(b, a)) >= 0.8:
                ln = sk.sketchCurves.sketchLines.addByTwoPoints(sk.modelToSketchSpace(P(a)), sk.modelToSketchSpace(P(b)))
                bodies += self._pipe(ln, r, name)
        return self.join(bodies)

    def revolve(self, profile_uv, origin, frame, deg=360.0, name='revolve'):
        """Revolve a closed (u, v) polygon about the sketch's v axis (u = radius, v = height along frame's ey).
        For lightbulbs, knobs, button caps."""
        ex, ey, ez = [unit(v) for v in frame]
        pl = self.plane(origin, ez, ex)
        sk = self._hidden_sketch(pl, name)
        to = lambda u, v: sk.modelToSketchSpace(P(add(origin, add(mul(ex, u), mul(ey, v)))))
        poly(profile_uv)(sk, to)
        vs = [v for _, v in profile_uv]
        axis = sk.sketchCurves.sketchLines.addByTwoPoints(to(0, min(vs)), to(0, max(vs)))
        axis.isConstruction = True
        rv = self.comp.features.revolveFeatures
        ri = rv.createInput(sk.profiles.item(0), axis, NEW)
        ri.setAngleExtent(False, VI.createByString(f'{deg} deg'))
        f = rv.add(ri)
        f.name = name
        return list(f.bodies)

    # -- bodies
    def join(self, bodies):
        """Join into bodies[0]; any tool that cannot be joined stays a separate body."""
        bodies = [b for b in bodies if b and b.isValid]
        if len(bodies) < 2:
            return bodies
        target, left = bodies[0], []
        cf = self.comp.features.combineFeatures
        for tool in bodies[1:]:
            try:
                ci = cf.createInput(target, coll([tool]))
                ci.operation = JOIN
                ci.isKeepToolBodies = False
                target = cf.add(ci).bodies.item(0)
            except Exception:
                left.append(tool)
        return [target] + left

    def hole(self, frame, center, d, length, target, name='hole'):
        """A cut feature (not a tool body): a Ø d cylinder along frame's ez, centred on center."""
        return self.prism(center, self._frame(frame), -length / 2, length / 2, circle(d / 2),
                          name=name, op=CUT, targets=[target])

    def axis(self, which):
        return getattr(self.comp, which + 'ConstructionAxis')

    def circular(self, entities, which, count, step_deg, name='pattern'):
        """Circular pattern of features or bodies about the component's X/Y/Z axis."""
        cp = self.comp.features.circularPatternFeatures
        ci = cp.createInput(coll(entities), self.axis(which))
        ci.quantity = VI.createByReal(count)
        ci.totalAngle = VI.createByString(f'{step_deg * (count - 1) if count > 1 else 0} deg')
        ci.isSymmetric = False
        f = cp.add(ci)
        f.name = name
        return f

    def rect_pattern(self, entities, which, count, spacing, name='pattern'):
        """Linear pattern of bodies/features along the component's X/Y/Z axis: count copies, spacing mm apart
        (joists at 16 in on centre, stair treads, deck screws)."""
        rp = self.comp.features.rectangularPatternFeatures
        ri = rp.createInput(coll(entities), self.axis(which), VI.createByReal(count), VI.createByReal(spacing / 10.0),
                            adsk.fusion.PatternDistanceType.SpacingPatternDistanceType)
        f = rp.add(ri)
        f.name = name
        return f

    def mirror(self, entities, plane='yZ', name='mirror'):
        mf = self.comp.features.mirrorFeatures
        f = mf.add(mf.createInput(coll(entities), getattr(self.comp, plane + 'ConstructionPlane')))
        f.name = name
        return f

    def cut(self, targets, tools):
        """Subtract tool bodies from each target; tools are consumed."""
        cf = self.comp.features.combineFeatures
        out = []
        for t in targets:
            ci = cf.createInput(t, coll(tools))
            ci.operation = CUT
            ci.isKeepToolBodies = True
            out.append(cf.add(ci).bodies.item(0))
        for tool in tools:
            if tool.isValid:
                self.comp.features.removeFeatures.add(tool)
        return out

    def finish(self, body_name=None):
        """Names (optional), appearance and physical material on every body. Never call on '(buy)' parts with a
        name: bought hardware keeps its body names."""
        a = None
        if self.color:
            a = appearance(self.color) if self.color.startswith('#') else library_appearance(self.color)
        mat = physical_material(self.material) if self.material else None
        n = self.comp.bRepBodies.count
        for i in range(n):
            b = self.comp.bRepBodies.item(i)
            if body_name:
                b.name = body_name if n == 1 else f'{body_name} {i + 1}'
            if mat:
                b.material = mat
            if a:
                b.appearance = a
        return self


def physical_material(name_prefix):
    """A physical material (density for mass properties) from the Fusion material library."""
    m = design.materials.itemByName(name_prefix)
    if m:
        return m
    lib = app.materialLibraries.itemByName('Fusion Material Library')
    src = next((lib.materials.item(i) for i in range(lib.materials.count)
                if lib.materials.item(i).name.startswith(name_prefix)), None)
    if src is None:
        raise KeyError('no physical material starting with %r' % name_prefix)
    return design.materials.addByCopy(src, src.name)


# ---------------------------------------------------------------- body helpers (from CDX-3R shoulder.py / concepts.py)

def merge_all(p, sliver=0.05):
    """Join every body of the part into the largest (drop slivers): robust against stale body references."""
    bodies_ = sorted(p.comp.bRepBodies, key=lambda b: -b.volume)
    for b in bodies_[1:]:
        if b.volume < sliver:
            p.comp.features.removeFeatures.add(b)
    main_ = bodies_[0]
    rest = [b for b in p.comp.bRepBodies if b != main_]
    if rest:
        cf = p.comp.features.combineFeatures
        ci = cf.createInput(main_, coll(rest))
        ci.operation = JOIN
        ci.isKeepToolBodies = False
        cf.add(ci)
    return p.comp.bRepBodies.item(0)


def is_sharp(e):
    """True unless the two faces meet smoothly (already rounded / tangent)."""
    if e.faces.count != 2:
        return False
    mid = e.pointOnEdge
    n = []
    for f in e.faces:
        ok, v = f.evaluator.getNormalAtPoint(mid)
        if not ok:
            return True
        n.append(v)
    return abs(n[0].dotProduct(n[1])) < 0.995


def _concave_small_cylinder(f, r_max):
    """A hole / socket / pocket: a cylindrical face (r < r_max mm) whose normal points at its axis."""
    g = f.geometry
    if not isinstance(g, adsk.core.Cylinder) or g.radius * 10 >= r_max:
        return False
    pt = f.pointOnFace
    ok, nrm = f.evaluator.getNormalAtPoint(pt)
    if not ok:
        return True
    a = g.axis
    a.normalize()
    v = g.origin.vectorTo(pt)
    radial = v.copy()
    t = a.copy()
    t.scaleBy(v.dotProduct(a))
    radial.subtract(t)
    return nrm.dotProduct(radial) < 0


def chamfer_edge(p, e):
    ch = p.comp.features.chamferFeatures
    for d in (1.0, 0.6):
        try:
            try:
                ci = ch.createInput2()
                ci.chamferEdgeSets.addEqualDistanceChamferEdgeSet(coll([e]), VI.createByReal(d / 10.0), False)
            except AttributeError:
                ci = ch.createInput(coll([e]), False)
                ci.setToEqualDistance(VI.createByReal(d / 10.0))
            ch.add(ci)
            return True
        except Exception:
            continue
    return False


def _one_by_one(p, ff, edges, r, chamfer):
    """Fillet each edge alone (r, then 2/3 r); with chamfer=True, fall back to a small chamfer.
    Returns (done, skipped_count, skipped_edges)."""
    done, left = 0, []
    for e in edges:
        if not e.isValid:
            continue
        ok = False
        for rr in (r, r * 2 / 3):
            try:
                fi = ff.createInput()
                try:
                    fi.edgeSetInputs.addConstantRadiusEdgeSet(coll([e]), VI.createByReal(rr / 10.0), False)
                except AttributeError:
                    fi.addConstantRadiusEdgeSet(coll([e]), VI.createByReal(rr / 10.0), False)
                ff.add(fi)
                ok = True
                break
            except Exception:
                continue
        if not ok and chamfer and e.isValid:
            ok = chamfer_edge(p, e)
        if ok:
            done += 1
        else:
            left.append(e)
    return done, len(left), left


def fillet(p, edges, r):
    """Fillet edges at r; if Fusion rejects the set, one at a time and skip what can't take it. (done, skipped)."""
    if not edges:
        return 0, 0
    ff = p.comp.features.filletFeatures
    try:
        fi = ff.createInput()
        try:
            fi.edgeSetInputs.addConstantRadiusEdgeSet(coll(edges), VI.createByReal(r / 10.0), False)
        except AttributeError:
            fi.addConstantRadiusEdgeSet(coll(edges), VI.createByReal(r / 10.0), False)
        ff.add(fi)
        return len(edges), 0
    except Exception:
        edges = sorted(edges, key=lambda e: -e.length)
        first = _one_by_one(p, ff, edges, r, chamfer=False)
        second = _one_by_one(p, ff, first[2], r, chamfer=True)
        return first[0] + second[0], second[1]


def round_edges(p, body, r=2.0, r_hole=30.0, min_len=3.0):
    """Round every sharp edge except hole / socket / pocket edges (fits stay exact). For printed parts and eased
    lumber edges; cut fit features after this, never before."""
    edges = []
    for e in body.edges:
        if e.length * 10 < min_len or not is_sharp(e):
            continue
        if any(_concave_small_cylinder(f, r_hole) for f in e.faces):
            continue
        edges.append(e)
    return fillet(p, edges, r)


# ---------------------------------------------------------------- bought hardware (components end in '(buy)')

def hw_bolt(p, c_head, frame, length, head, name, dia):
    """Bolt / lag / carriage bolt along frame's ez: head (dia, height) behind c_head, shank in front."""
    hd, hh = head
    b = p.prism(c_head, frame, -hh, 0, circle(hd / 2), name=name).bodies.item(0)
    p.prism(c_head, frame, 0, length, circle(dia / 2), name=name + ' shank', op=JOIN, targets=[b])
    b.name = name
    return b


def hw_nut(p, c, frame, af, h, dia, name):
    b = p.prism(c, frame, 0, h, hexagon(af), name=name).bodies.item(0)
    p.prism(c, frame, -1, h + 1, circle(dia / 2), name='thread', op=CUT, targets=[b])
    b.name = name
    return b


def hw_washer(p, c, frame, od, idia, t, name):
    b = p.ann(frame, add(c, mul(unit(frame[2]), t / 2)), od / 2, idia / 2, t, name=name)[0]
    b.name = name
    return b


# ---------------------------------------------------------------- checks and exports

def world_bbox(bodies, occ):
    """Bounding box in root space (mm), from a tessellation (boundingBox can be stale or short)."""
    lo, hi = [1e9] * 3, [-1e9] * 3
    for b in bodies:
        proxy = b.createForAssemblyContext(occ)
        if isinstance(proxy, adsk.fusion.BRepBody):
            calc = proxy.meshManager.createMeshCalculator()
            calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.NormalQualityTriangleMesh)
            xyz = calc.calculate().nodeCoordinatesAsDouble
            box = ([min(xyz[i::3]) for i in range(3)], [max(xyz[i::3]) for i in range(3)])
        else:
            bb = proxy.boundingBox
            box = (bb.minPoint.asArray(), bb.maxPoint.asArray())
        for i in range(3):
            lo[i], hi[i] = min(lo[i], box[0][i] * 10), max(hi[i], box[1][i] * 10)
    return lo, hi


def save_archive(name):
    """Export the active design as cad/fusion/models/<name>.f3d."""
    import os
    path = os.path.join(REPO, 'cad', 'fusion', 'models', name + '.f3d')
    em = design.exportManager
    em.execute(em.createFusionArchiveExportOptions(path))
    return path
