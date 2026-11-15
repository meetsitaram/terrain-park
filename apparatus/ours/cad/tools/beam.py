"""Structural spot checks for the apparatus (simple beam theory, US lumber design values).

Every case: a simply supported span with a uniform load and/or a concentrated load at the worst spot, bending
stress against the allowable (NDS-style Fb x size x repetitive x duration factors), shear, and deflection against
L/360 (live). Loads from params.DESIGN_LOAD: a 120 kg robot plus a helper -> 400 lbf on one foot, 60 psf.

Run: python cad/tools/beam.py   (prints a table; --md writes docs/checks/structure.md)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import params as P

E_LUMBER = 1.4e6          # psi, No.2 SPF / SYP
FB_BASE = 875.0           # psi, No.2 SPF base value (conservative for SYP too)
FV = 135.0                # psi shear
CF = {'2x4': 1.5, '2x6': 1.3, '2x8': 1.2, '2x10': 1.1, '2x12': 1.0, '4x4': 1.5, '6x6': 1.0}
CD = 1.0                  # normal duration (robot trials of minutes count as short-term; stay at 1.0)
# 3/4 (23/32) span-rated plywood, strength axis across supports, per foot of width (APA PDS)
PLY = dict(EI=286000.0, FbS=1120.0, FvIb=290.0)   # lb-in^2/ft, lb-in/ft, lb/ft (Sturd-I-Floor 24 oc, 23/32)

P_FOOT = P.DESIGN_LOAD['point_lbf']
Q_PSF = P.DESIGN_LOAD['live_psf']


def section(nom, plies=1):
    b, d = P.LUMBER[nom]
    b *= plies
    return b, d, b * d ** 3 / 12, b * d ** 2 / 6


def lumber_case(name, nom, L, trib_in=0.0, point=0.0, plies=1, repetitive=False, depth=None, note=''):
    """L in inches; trib_in tributary width for the uniform live load; point at mid-span."""
    b, d, I, S = section(nom, plies)
    if depth:                                                       # e.g. a cut stair stringer: the throat depth
        d = depth
        I, S = b * d ** 3 / 12, b * d ** 2 / 6
    w = Q_PSF / 144.0 * trib_in                                     # lb/in
    M = w * L ** 2 / 8 + point * L / 4
    V = w * L / 2 + point / 2
    fb = M / S
    Fb = FB_BASE * CF.get(nom, 1.0) * CD * (1.15 if repetitive else 1.0)
    fv = 1.5 * V / (b * d)
    defl = 5 * w * L ** 4 / (384 * E_LUMBER * I) + point * L ** 3 / (48 * E_LUMBER * I)
    lim = L / 360
    ok = fb <= Fb and fv <= FV and defl <= lim
    return dict(name=name, member='%s%s' % ('%dx ' % plies if plies > 1 else '', nom), span=L, fb=fb, Fb=Fb, fv=fv,
                Fv=FV, defl=defl, lim=lim, ok=ok, note=note)


def ply_case(name, L, point=0.0, width_ft=None, note=''):
    """Plywood strip over span L (in), point load at mid-span plus the live load. A foot load spreads across the
    sheet: effective strip width = span + 4 in foot contact (one-way slab rule of thumb), unless width_ft given.
    Deflection limit L/240 under a concentrated load (APA concentrated-load practice), L/360 uniform only."""
    if width_ft is None:
        width_ft = (L + 4.0) / 12.0
    w = Q_PSF / 144.0 * 12 * width_ft                               # lb/in on the strip
    M = w * L ** 2 / 8 + point * L / 4
    V = w * L / 2 + point / 2
    cap_M = PLY['FbS'] * width_ft
    cap_V = PLY['FvIb'] * width_ft
    EI = PLY['EI'] * width_ft
    defl = 5 * w * L ** 4 / (384 * EI) + point * L ** 3 / (48 * EI)
    lim = L / (240 if point else 360)
    ok = M <= cap_M and V <= cap_V and defl <= lim
    return dict(name=name, member='3/4 ply (%.0f in strip)' % (width_ft * 12), span=L, fb=M, Fb=cap_M, fv=V, Fv=cap_V,
                defl=defl, lim=lim, ok=ok, note=note, units='lb-in / lb')


def cases():
    out = []
    oc = P.FLOOR['oc']
    # a robot foot spreads a point load over roughly a 1 ft wide strip of 3/4 plywood (conservative for a sheet)
    out.append(ply_case('Lane deck sheet between joists', oc - 1.5, P_FOOT,
                        note='deck panels sit on the floor; the sheet spans between 2x4 joists at %g in o.c.' % oc))
    out.append(ply_case('Landing sheet between joists', P.LANDING['oc'] - 1.5, P_FOOT))
    rib = P.RAMPS['cell'] / (P.RAMPS['ribs'] - 1)
    out.append(ply_case('Ramp skin between ribs', rib - 1.5, P_FOOT,
                        note='%d ribs per 4 ft cell (%g in o.c.); strength axis across the ribs' % (P.RAMPS['ribs'], rib)))
    # landing framing
    bay = (P.X['landing'][1] - P.X['landing'][0]) / 2 - 3.0 - 1.5
    out.append(lumber_case('Landing joist', P.LANDING['joist'], bay, trib_in=P.LANDING['oc'], point=P_FOOT,
                           repetitive=True))
    out.append(lumber_case('Landing beam (between posts)', '2x8', 44.0, trib_in=48.0, point=2 * P_FOOT, plies=2,
                           note='two joists land on it; posts at the lane edges and the middle'))
    # stairs
    S = P.STAIRS
    th = math.atan2(S['rise'], S['run'])
    L_h = (S['risers'] - 1) * S['run'] + 8.25                       # horizontal run of the stringer
    L_s = L_h / math.cos(th)
    throat = P.LUMBER['2x12'][1] - S['rise'] * math.cos(th)
    ns = S['stringers']
    sp = (S['width'] - 1.5) / (ns - 1)
    out.append(lumber_case('Stair stringer (cut 2x12, throat %.1f in)' % throat, '2x12', L_s, trib_in=sp,
                           point=0.6 * P_FOOT, depth=throat, repetitive=True,
                           note='%d stringers at %.1f in; a foot on a 2-board tread loads a stringer with ~60%% of it; '
                                'conservative: only the throat depth counts' % (ns, sp)))
    out.append(lumber_case('Stair tread 2x10 flat', '2x10', sp - 1.5, point=P_FOOT, depth=1.5,
                           note='one 2x10 laid flat between stringers (b = 9.25, d = 1.5)'))
    out[-1]['member'] = '2x10 flat'
    b, d = 9.25, 1.5
    t = out[-1]
    I, Sx = b * d ** 3 / 12, b * d ** 2 / 6
    L = t['span']
    t['fb'] = P_FOOT * L / 4 / Sx
    t['fv'] = 1.5 * P_FOOT / 2 / (b * d)
    t['defl'] = P_FOOT * L ** 3 / (48 * E_LUMBER * I)
    t['Fb'] = FB_BASE * 1.3 * 1.15                                  # flat use factor ~1.15
    t['ok'] = t['fb'] <= t['Fb'] and t['fv'] <= FV and t['defl'] <= t['lim']
    # gantry: fall arrest on the ridge
    G = P.GANTRY
    fall = P.ROBOT['mass_kg'] * 9.81 * 2.5 / 4.448                  # lbf: 120 kg x dynamic factor 2.5
    rail_out = P.LUMBER['4x4'][0] + P.LUMBER['2x4'][0]
    span = P.LANE['width'] + 2 * (rail_out + G['side_gap'])
    out.append(lumber_case('Gantry crossbeam (rope at mid-span)', '2x10', span, point=fall, plies=2,
                           note='mobile gantry; 120 kg x 2.5 dynamic factor = %.0f lbf at the middle' % fall))
    return out


def table(rows):
    lines = ['| Member | Section | Span (in) | Bending (act / allow) | Shear (act / allow) | Deflection (in, limit L/360) | OK |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        u = r.get('units')
        bend = '%.0f / %.0f %s' % (r['fb'], r['Fb'], 'lb-in' if u else 'psi')
        shear = '%.0f / %.0f %s' % (r['fv'], r['Fv'], 'lb' if u else 'psi')
        lines.append('| %s | %s | %.1f | %s | %s | %.3f / %.3f | %s |' % (
            r['name'], r['member'], r['span'], bend, shear, r['defl'], r['lim'], 'yes' if r['ok'] else '**NO**'))
    return '\n'.join(lines)


if __name__ == '__main__':
    rows = cases()
    print(table(rows))
    for r in rows:
        if r['note']:
            print('-', r['name'] + ':', r['note'])
    if '--md' in sys.argv:
        out = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'checks')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'structure.md'), 'w', encoding='utf-8') as f:
            f.write('# Structural checks\n\nGenerated by `cad/tools/beam.py`. Design loads: %d lbf on one foot, %d psf '
                    'live, gantry rope 120 kg x 2.5.\n\n' % (P_FOOT, Q_PSF))
            f.write(table(rows) + '\n\n')
            for r in rows:
                if r['note']:
                    f.write('- **%s**: %s\n' % (r['name'], r['note']))
