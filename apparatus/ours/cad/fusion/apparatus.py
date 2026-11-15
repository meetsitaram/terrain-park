"""Build the apparatus in Fusion from the pure-Python model (cad/model.py).

main(stage): rebuilds one stage (see model.STAGES) inside the top component 'Loco-manip apparatus'. Each stage
owns its group components (e.g. '01 Deck'), so a stage can be rebuilt alone. One stage per MCP call.

Every piece becomes one BRep body made with the TemporaryBRepManager (boxes, cylinders, spheres, booleans) and
added in one base feature per unit component: fast, no sketch per board. Bodies get a physical material
(density -> mass) and an appearance (textured wood, OSB, plywood, metal, plastic).
"""
import os
import time

import adsk.core
import adsk.fusion

import fxlib as fx
import model as M
import params as P

TEX = os.path.join(fx.REPO, 'cad', 'fusion', 'textures')

# look -> (library appearance to copy, colour '#rrggbb' or texture png, texture real-world scale in inches)
LOOKS = {
    'pine': ('Pine', None, None),
    'stain cedar 2': ('Pine', 'stain_cedar_2.png', 18.0),        # BEHR ST-533 Cedar Naturaltone on pine
    'ply': ('Pine', 'plywood.png', 24.0),
    'osb': ('Pine', 'osb.png', 24.0),
    'hard': ('Plastic - Matte (Gray)', '#8a6a48', None),
    'birch': ('Pine', 'plywood.png', 30.0),
    'countertop': ('Granite - Black and White', 'countertop.png', 24.0),
    'paint white': ('Paint - Enamel Glossy (White)', None, None),
    'door grey': ('Paint - Enamel Glossy (Grey)', '#8c8f93', None),
    'satin nickel': ('Stainless Steel - Satin', None, None),
    'galvanized': ('Steel - Galvanized (Small)', None, None),
    'brass': ('Brass - Polished', None, None),
    'red vinyl': ('Plastic - Glossy (Red)', None, None),
    'porcelain': ('Plastic - Glossy (White)', '#f2efe6', None),
    'bulb': ('Plastic - Matte (White)', '#f4f4ee', None),
    'petg': ('Plastic - Matte (White)', '#e8e8e8', None),
    'print green': ('Plastic - Matte (Green)', '#3fae49', None),
    'white plastic': ('Plastic - Glossy (White)', None, None),
    'black plastic': ('Plastic - Matte (Black)', None, None),
    'grey plastic': ('Plastic - Matte (Gray)', None, None),
    'button yellow': ('Plastic - Glossy (Yellow)', '#f2d21b', None),
    'display red': ('LED (Red)', None, None),
    'clear plastic': ('Glass - Frosted Light', None, None),
    'crate blue': ('Plastic - Glossy (Blue)', '#1b3fc4', None),
    'drill yellow': ('Plastic - Glossy (Yellow)', '#f2b705', None),
    'rope': ('Fabric (White)', None, None),
    'paint blue': ('Paint - Enamel Glossy (Blue)', '#2a3fb8', None),
    'paint yellow': ('Paint - Enamel Glossy (Yellow)', None, None),
    'paint purple': ('Paint - Enamel Glossy (Blue)', '#6b2fb0', None),
    'paint orange': ('Paint - Enamel Glossy (Red)', '#ef7d1a', None),
}

# physical material key -> (Fusion material library name, density kg/m^3 to set)
MATS = {
    'softwood': ('Softwood, Lumber', 450.0),        # SPF / SYP framing lumber, kiln dried
    'plywood': ('Plywood, Sheathing', 560.0),
    'osb': ('Particleboard', 620.0),
    'hardboard': ('Particleboard', 900.0),
    'particleboard': ('Particleboard', 680.0),
    'hardwood': ('Oak, Red', 700.0),
    'petg': ('ABS Plastic', 1270.0),
}

_cache = {}


def look(key):
    if key in _cache and _cache[key].isValid:
        return _cache[key]
    name = 'lma ' + key
    a = fx.design.appearances.itemByName(name)
    if a is None:
        base, col, scale = LOOKS.get(key, ('Plastic - Matte (Gray)', None, None))
        lib = fx.app.materialLibraries.itemByName('Fusion Appearance Library')
        a = fx.design.appearances.addByCopy(lib.appearances.itemByName(base), name)
        if col and col.startswith('#'):
            h = col.lstrip('#')
            c = adsk.core.Color.create(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
            for pid in ('opaque_albedo', 'layered_diffuse', 'surface_albedo', 'transparent_color', 'metal_f0'):
                p = a.appearanceProperties.itemById(pid)
                if p and isinstance(p, adsk.core.ColorProperty):
                    try:
                        if p.connectedTexture is None:
                            p.value = c
                    except Exception:
                        pass
        elif col and col.startswith('tint:'):
            h = col[5:].lstrip('#')
            c = adsk.core.Color.create(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
            p = a.appearanceProperties.itemById('opaque_albedo')
            tp = p.connectedTexture.properties
            tp.itemById('common_Tint_toggle').value = True
            tp.itemById('common_Tint_color').value = c
        elif col:
            for pid in ('opaque_albedo', 'layered_diffuse', 'surface_albedo'):
                p = a.appearanceProperties.itemById(pid)
                if p and isinstance(p, adsk.core.ColorProperty) and p.connectedTexture:
                    tp = p.connectedTexture.properties
                    tp.itemById('unifiedbitmap_Bitmap').value = os.path.join(TEX, col).replace('\\', '/')
                    for sid in ('texture_RealWorldScaleX', 'texture_RealWorldScaleY'):
                        s = tp.itemById(sid)
                        if s:
                            s.value = scale
    _cache[key] = a
    return a


def material(key):
    if key is None:
        return None
    nm, dens = MATS[key]
    name = 'lma ' + key
    m = fx.design.materials.itemByName(name)
    if m is None:
        lib = fx.app.materialLibraries.itemByName('Fusion Material Library')
        m = fx.design.materials.addByCopy(lib.materials.itemByName(nm), name)
        p = m.materialProperties.itemById('structural_Density')
        if p:
            p.value = dens / 1000.0                     # Fusion stores g/cm^3... value in database units
    return m


def cm(p):
    return adsk.core.Point3D.create(p[0] * P.IN / 10, p[1] * P.IN / 10, p[2] * P.IN / 10)


def vec(v):
    return adsk.core.Vector3D.create(v[0], v[1], v[2])


def temp_body(shape, tb):
    kind = shape[0]
    if kind == 'box':
        _, c, axes, size = shape
        obb = adsk.core.OrientedBoundingBox3D.create(cm(c), vec(axes[0]), vec(axes[1]),
                                                     size[0] * P.IN / 10, size[1] * P.IN / 10, size[2] * P.IN / 10)
        return tb.createBox(obb)
    if kind == 'cyl':
        _, a, b, r0, r1 = shape
        return tb.createCylinderOrCone(cm(a), r0 * P.IN / 10, cm(b), r1 * P.IN / 10)
    if kind == 'sphere':
        _, c, r = shape
        return tb.createSphere(cm(c), r * P.IN / 10)
    raise ValueError(kind)


def piece_body(p, tb):
    body = None
    for s in p.shapes:
        b = temp_body(s, tb)
        if body is None:
            body = b
        else:
            tb.booleanOperation(body, b, adsk.fusion.BooleanTypes.UnionBooleanType)
    for s in p.cuts:
        tb.booleanOperation(body, temp_body(s, tb), adsk.fusion.BooleanTypes.DifferenceBooleanType)
    return body


def top():
    occ = fx.top_occurrence()
    return occ.component


def child(parent, name):
    for o in parent.occurrences:
        if o.component.name == name:
            return o.component
    occ = parent.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = name
    return occ.component


def clear_stage(parent, prefixes):
    for o in list(parent.occurrences):
        if any(o.component.name == pf for pf in prefixes):
            o.deleteMe()


def build_units(parent, pieces):
    """pieces of one stage -> stage component / unit component / bodies."""
    tb = adsk.fusion.TemporaryBRepManager.get()
    units = {}
    for p in pieces:
        units.setdefault(p.group, []).append(p)
    n = 0
    for gname, ps in units.items():
        stage_name, unit_name = gname.split('/', 1)
        sc = child(parent, stage_name)
        occ = sc.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        comp = occ.component
        comp.name = unit_name
        made = []
        for p in ps:
            try:
                made.append((p, comp.bRepBodies.add(piece_body(p, tb))))   # direct design: no base feature
            except Exception:
                made.append((p, None))
        for p, b in made:
            if b is None:
                continue
            b.name = p.name
            mt = material(p.mat)
            if mt:
                b.material = mt
            b.appearance = look(p.look)
            n += 1
    return n


def main(stage, poses=None):
    t0 = time.time()
    fx.init()
    doc = fx.app.activeDocument
    if doc.name != 'Loco-manip apparatus':
        raise RuntimeError('active document is %r, open "Loco-manip apparatus" first' % doc.name)
    if fx.design.designType != adsk.fusion.DesignTypes.DirectDesignType:
        fx.design.designType = adsk.fusion.DesignTypes.DirectDesignType   # no timeline: see LESSONS.md
    root_top = top()
    m = M.Model()
    dict(M.STAGES)[stage](m)
    M.stain(m)
    m.pose(poses if poses is not None else P.POSE)
    prefixes = sorted({p.group.split('/')[0] for p in m.pieces})
    clear_stage(root_top, prefixes)
    n = build_units(root_top, m.pieces)
    return dict(stage=stage, groups=prefixes, bodies=n, pieces=len(m.pieces), seconds=round(time.time() - t0, 1))


def rebuild_all(stages=None, poses=None):
    """Every stage in order (about 15 s). Removes stray top-level occurrences first."""
    fx.init()
    root = fx.design.rootComponent
    for o in list(root.occurrences):
        if o.component.name != fx.TOP:
            o.deleteMe()
    return [main(s, poses) for s, _ in M.STAGES if stages is None or s in stages]


def audit():
    """Bodies per stage that kept Fusion's default look/material (a sign the finishing pass was lost)."""
    fx.init()
    top_c = fx.top_occurrence(create=False).component
    out = {}
    for so in top_c.occurrences:
        bad = total = 0
        for o in so.component.allOccurrences:
            for b in o.component.bRepBodies:
                total += 1
                if not b.appearance or not b.appearance.name.startswith('lma '):
                    bad += 1
        out[so.component.name] = (total, bad)
    return out
