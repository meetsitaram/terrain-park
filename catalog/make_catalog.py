"""Visual catalogue of the Pico terrain captures (X2 retargets), grouped by prop and approach shape.

    python catalog/make_catalog.py            # everything: categories.json, top-view sheets, GIFs, README.md, index.html
    python catalog/make_catalog.py classify   # categories.json only
    python catalog/make_catalog.py plots | gifs | docs

Inputs (this repo): data/train_sets/*.pkl + *_props.json, tp_stairs_all_v1_meta.json, tp_ramp_hmap_x30.npz,
data/2026-10-30/props/ramp.json, data/2026-10-30/x2/record_20261030_1210*-1211* (driveway). X2 model: X2_RECORDER
(default: sibling folder agibot-x2-record-and-replay, as tools/_x2.py). CPU only, at most 3 threads.

Frames. Every take is put in a prop-aligned frame: origin at the prop, +x = the crossing direction.
  beams (4x4, 2x2 log): +x across the beam, turned so that the take crosses towards +x (a 180 deg turn, never a mirror:
      left stays left); stairs: +x up the stairs, origin at the front edge of step 1; ramp: +x up the slope, origin at
      the foot of the slope; driveway: +x uphill, origin at the centre of the takes.
Numbers (degrees, + = to the left / counter-clockwise seen from above):
  body = pelvis heading, path = direction of travel; *_x = at the crossing, against +x; turn_before / turn_after = change
  of the body heading from the start to the crossing and from the crossing to the end; sideways_share / backward_share =
  share of the travelled distance with the travel direction 50-130 deg / over 130 deg off the body heading.
"""
import os, sys, json, glob, math, collections
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '3')
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TS = REPO / 'data' / 'train_sets'
PROPS = [('4x4', '4x4 beam'), ('2x2', '2x2 log'), ('stairs1', 'Stairs, place 1'), ('stairs2', 'Stairs, place 2'),
         ('ramp', 'Ramp'), ('driveway', 'Driveway')]
PROP_NAME = dict(PROPS)
RAMP_DX = 30.0            # the ramp training set is the 2026-10-30 X2 world shifted by +30 m in x (checked against the height map)
D = math.degrees


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def smooth(x, n):
    n = max(1, int(n)) | 1
    if n == 1:
        return x
    k = np.ones(n) / n
    xp = np.concatenate([np.repeat(x[:1], n // 2, 0), x, np.repeat(x[-1:], n // 2, 0)])
    return np.stack([np.convolve(xp[:, i], k, 'valid') for i in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(xp, k, 'valid')


def yaw_xyzw(q):
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------- loading
def load_takes():
    import joblib
    takes = []

    def add(tid, prop, src, c, boxes, **kw):
        takes.append(dict(id=tid, prop=prop, source=src, fps=float(c['fps']), pos=np.asarray(c['root_trans_offset'], float),
                          quat=np.asarray(c['root_rot'], float), dof=np.asarray(c['dof'], float), boxes=boxes, **kw))

    for prop, f, pj in (('4x4', 'tp_t4x4u_y20', 'tp_t4x4u_y20_props'), ('2x2', 'pico_2026-10-29_2x2_train', 'pico_2026-10-29_2x2_props')):
        cl = joblib.load(TS / (f + '.pkl')); pr = json.load(open(TS / (pj + '.json')))
        for k in cl:
            add(k, prop, f + '.pkl', cl[k], pr[k])
    cl = joblib.load(TS / 'tp_stairs_all_v1.pkl'); pr = json.load(open(TS / 'tp_stairs_all_v1_props.json'))
    meta = json.load(open(TS / 'tp_stairs_all_v1_meta.json'))
    for k in cl:
        add(k, 'stairs%d' % meta[k]['place'], 'tp_stairs_all_v1.pkl', cl[k], pr[k], meta={a: b for a, b in meta[k].items() if a != 'checks'})
    cl = joblib.load(TS / 'acp_tp_ramp_sh2_x30.pkl')
    for k in cl:
        if '_sh' not in k:
            add(k, 'ramp', 'acp_tp_ramp_sh2_x30.pkl (base takes)', cl[k], [])
    for f in sorted(glob.glob(str(REPO / 'data' / '2026-10-30' / 'x2' / 'record_20261030_121[01]*.npz'))):
        z = np.load(f); t, q = z['t'], z['qpos']
        tt = np.arange(0, t[-1], 0.02); idx = np.clip(np.searchsorted(t, tt), 0, len(t) - 1)   # 50 fps, nearest sample
        q = q[idx]
        c = dict(fps=50, root_trans_offset=q[:, :3], root_rot=q[:, [4, 5, 6, 3]], dof=q[:, 7:])
        add('tp_driveway_' + Path(f).stem[-6:], 'driveway', 'data/2026-10-30/x2/' + Path(f).name, c, [])
    return takes


def ramp_prop():
    """Ramp frame from the training height map itself: +x = mean uphill gradient of the slope cells, origin = foot of
    the slope at its width centre. (props/ramp.json shifted by +30 m is a different, wider fit, turned against this one.)"""
    x0, y0, res, g = ramp_hmap(); hmax = g.max()
    gx, gy = np.gradient(g, res); mag = np.hypot(gx, gy)
    m = (g > 0.03) & (g < 0.85 * hmax) & (mag > 0.05) & (mag < 0.5)          # the slope, without its side walls
    yaw = math.atan2(gy[m].mean(), gx[m].mean()); u = np.array([math.cos(yaw), math.sin(yaw)]); v = np.array([-u[1], u[0]])
    I, J = np.where(m); xy = np.c_[x0 + res * I, y0 + res * J]
    k, c = np.polyfit(xy @ u, g[m], 1)
    o = u * (-c / k) + v * np.median(xy @ v)
    old = json.load(open(REPO / 'data' / '2026-10-30' / 'props' / 'ramp.json'))['props'][0]
    return dict(origin=[float(o[0]), float(o[1]), 0.0], yaw=float(yaw), slope_deg=D(math.atan(k)), rise=float(hmax),
                source='tp_ramp_hmap_x30.npz', yaw_vs_ramp_json_deg=round(D(wrap(yaw - old['yaw'])), 1),
                origin_vs_ramp_json_shifted_m=[round(float(o[0] - old['origin'][0] - RAMP_DX), 2), round(float(o[1] - old['origin'][1]), 2)])


def hmap_at(hm, xy):
    x0, y0, res, g = hm
    i = np.clip(np.round((xy[:, 0] - x0) / res).astype(int), 0, g.shape[0] - 1)
    j = np.clip(np.round((xy[:, 1] - y0) / res).astype(int), 0, g.shape[1] - 1)
    return g[i, j]


def ramp_hmap():
    h = np.load(TS / 'tp_ramp_hmap_x30.npz')
    return float(h['x0']), float(h['y0']), float(h['res']), np.asarray(h['grid'], float)      # grid[ix, iy]


# ---------------------------------------------------------------- X2 model (also used for the driveway plane fit)
def x2_spec():
    import mujoco
    cands = []
    if os.environ.get('X2_RECORDER'):
        cands.append(Path(os.environ['X2_RECORDER']) / 'sim' / 'x2_ultra.xml')
    cands += [REPO.parent / 'agibot-x2-record-and-replay' / 'sim' / 'x2_ultra.xml']
    for p in cands:
        if p.exists():
            return mujoco.MjSpec.from_file(str(p))
    raise FileNotFoundError('X2 model not found: set X2_RECORDER to the agibot-x2-record-and-replay checkout')


def to_qpos(tk, i):
    q = np.empty(38)
    q[:3] = tk['pos'][i]; q[3] = tk['quat'][i, 3]; q[4:7] = tk['quat'][i, :3]; q[7:] = tk['dof'][i]   # xyzw -> wxyz
    return q


def fit_driveway(takes):
    """Plane z = a x + b y + c through the X2's own planted feet (lower ankle body per frame), lowered by the ankle
    height measured the same way on the flat 4x4 takes."""
    import mujoco
    m = x2_spec().compile(); d = mujoco.MjData(m)
    feet = [i for i in range(m.nbody) if 'ankle_roll' in m.body(i).name]

    def low(tk, step=5):
        out = []
        for i in range(0, len(tk['pos']), step):
            d.qpos[:] = to_qpos(tk, i); mujoco.mj_kinematics(m, d)
            p = d.xpos[feet]; out.append(p[np.argmin(p[:, 2])].copy())
        return np.array(out)
    flat = np.concatenate([low(t)[:, 2] for t in takes if t['prop'] == '4x4'][:10])
    ankle = float(np.median(flat))
    P = np.concatenate([low(t) for t in takes if t['prop'] == 'driveway'])
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    co = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    for _ in range(3):                                    # drop swing-foot outliers
        r = P[:, 2] - A @ co; keep = np.abs(r - np.median(r)) < 0.03
        co = np.linalg.lstsq(A[keep], P[keep, 2], rcond=None)[0]
    r = P[:, 2] - A @ co
    return dict(a=float(co[0]), b=float(co[1]), c=float(co[2] - ankle), ankle_height=ankle, slope_deg=D(math.atan(math.hypot(co[0], co[1]))),
                resid_p90_cm=float(np.percentile(np.abs(r[keep]), 90) * 100), centre=[float(P[:, 0].mean()), float(P[:, 1].mean())],
                note='fitted here to the X2 ankle bodies of the 6 driveway takes; props/driveway.json is at human scale')


# ---------------------------------------------------------------- frames and numbers
def base_frame(tk, ctx):
    """origin (2,), yaw of +x, footprint polygons in the prop frame."""
    p = tk['prop']
    if p in ('4x4', '2x2'):
        b = tk['boxes'][0]
        return np.array(b['center'][:2]), b['yaw'] + np.pi / 2
    if p.startswith('stairs'):
        b = tk['boxes'][0]; u = np.array([math.cos(b['yaw']), math.sin(b['yaw'])])
        return np.array(b['center'][:2]) - u * b['half'][0], b['yaw']
    if p == 'ramp':
        r = ctx['ramp']; return np.array(r['origin'][:2]), r['yaw']
    dw = ctx['driveway']; return np.array(dw['centre']), math.atan2(dw['b'], dw['a'])


def to_frame(xy, o, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    d = xy - o
    return np.c_[d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c]


def footprints(tk, ctx, o, yaw):
    out = []
    for b in tk['boxes']:
        hx, hy = b['half'][:2]; c, s = math.cos(b['yaw']), math.sin(b['yaw'])
        cor = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        w = np.array(b['center'][:2]) + np.c_[cor[:, 0] * c - cor[:, 1] * s, cor[:, 0] * s + cor[:, 1] * c]
        out.append(to_frame(w, o, yaw))
    if tk['prop'] == 'ramp':                                    # extent of the raised cells of the height map (for the axes)
        x0, y0, res, g = ctx['hmap']; I, J = np.where(g > 0.01)
        F = to_frame(np.c_[x0 + res * I, y0 + res * J], o, yaw)
        out.append(np.array([[F[:, 0].min(), F[:, 1].min()], [F[:, 0].max(), F[:, 1].max()]]))
    return out


def kinematics(tk, o, yaw):
    fps = tk['fps']
    P = smooth(to_frame(tk['pos'][:, :2], o, yaw), 0.3 * fps)
    psi = smooth(np.unwrap(yaw_xyzw(tk['quat']) - yaw), 0.6 * fps)
    psi = psi - 2 * np.pi * round(float(psi[0]) / (2 * np.pi))
    V = np.gradient(P, axis=0) * fps; V = smooth(V, 0.4 * fps)
    sp = np.hypot(*V.T); phi = np.arctan2(V[:, 1], V[:, 0])
    return P, psi, phi, sp


def shares(psi, phi, sp, fps):
    mv = sp > 0.15
    dist = sp / fps
    tot = float(dist[mv].sum()) or 1e-9
    off = np.abs(wrap(phi - psi))
    return float(dist[mv & (off > math.radians(50)) & (off <= math.radians(130))].sum() / tot), \
        float(dist[mv & (off > math.radians(130))].sum() / tot)


def crossings(P, half_len, band=0.12):
    """Frames where the root passes x = 0 (from beyond -band to beyond +band or back) within the beam's length."""
    out = []; side = 0; last = 0
    for i, x in enumerate(P[:, 0]):
        s = 1 if x > band else (-1 if x < -band else 0)
        if s == 0:
            continue
        if side and s != side:
            j = last + int(np.argmin(np.abs(P[last:i + 1, 0])))
            if abs(P[j, 1]) <= half_len + 0.15:
                out.append((j, s))
        side = s; last = i
    return out


def seg_dir(P, i, n):
    a, b = max(0, i - n), min(len(P) - 1, i + n)
    d = P[b] - P[a]
    return math.atan2(d[1], d[0])


def mean_ang(psi, i, n):
    return float(np.mean(psi[max(0, i - n):i + n + 1]))


TURN = 40.0      # deg: a body turn of at least this much before or after the crossing counts as a turn
OBL = 25.0       # deg: path or body this far off the square crossing direction counts as oblique
SIDE = 50.0      # deg: travel direction this far off the body heading counts as sideways


def analyse_beam(tk, ctx):
    o, yaw = base_frame(tk, ctx); fps = tk['fps']; hl = tk['boxes'][0]['half'][0]
    P, psi, phi, sp = kinematics(tk, o, yaw)
    cr = crossings(P, hl)
    if (cr and cr[0][1] < 0) or (not cr and P[:, 0].mean() > 0):
        yaw += np.pi; P, psi, phi, sp = kinematics(tk, o, yaw); cr = crossings(P, hl)
    n3 = int(0.3 * fps); ss, bs = shares(psi, phi, sp, fps)
    num = dict(crossings=len(cr), seconds=round(len(P) / fps, 1), path_m=round(float(np.hypot(*np.diff(P, axis=0).T).sum()), 2),
               sideways_share=round(ss, 2), backward_share=round(bs, 2), body_start_x=round(D(mean_ang(psi, 0, n3)), 0),
               body_end_x=round(D(mean_ang(psi, len(P) - 1, n3)), 0), closest_m=round(float(np.abs(P[:, 0]).min()), 2))
    if not cr:
        num['turn_total'] = round(num['body_end_x'] - num['body_start_x'], 0)
        num['start_x_m'] = round(float(P[0, 0]), 2); num['end_x_m'] = round(float(P[-1, 0]), 2)
        return 'no crossing (walk-up, stand at the beam or pass alongside)', num, (o, yaw)
    i = cr[0][0]
    path_x = D(seg_dir(P, i, int(0.4 * fps))); body_x = D(mean_ang(psi, i, n3))
    body_x = (body_x + 180) % 360 - 180
    tb = D(mean_ang(psi, i, n3) - mean_ang(psi, 0, n3)); ta = D(mean_ang(psi, len(P) - 1, n3) - mean_ang(psi, i, n3))
    if len(cr) > 1:
        j = cr[-1][0]; ta = D(mean_ang(psi, len(P) - 1, n3) - mean_ang(psi, j, n3))
    off = (body_x - path_x + 180) % 360 - 180
    num.update(path_x=round(path_x, 0), body_x=round(body_x, 0), body_vs_path=round(off, 0), turn_before=round(tb, 0), turn_after=round(ta, 0),
               cross_y_m=round(float(P[i, 1]), 2))
    if len(cr) > 1:
        cat = 'over and back'
    elif SIDE <= abs(off) <= 180 - SIDE:
        cat = 'sidestep over'
    elif abs(off) > 180 - SIDE:
        cat = 'backward over'
    elif abs(tb) >= TURN and abs(ta) >= TURN:
        cat = 'C / U turn' if tb * ta > 0 else 'S turn'
    elif abs(tb) >= TURN:
        cat = 'turn, then cross'
    elif abs(ta) >= TURN:
        cat = 'cross, then turn'
    elif abs(body_x) >= OBL or abs(path_x) >= OBL + 10:
        cat = 'oblique over'
    else:
        cat = 'straight over'
    return cat, num, (o, yaw)


def analyse_ramp(tk, ctx):
    r = ctx['ramp']; o, yaw = base_frame(tk, ctx); fps = tk['fps']
    P, psi, phi, sp = kinematics(tk, o, yaw)
    h = hmap_at(ctx['hmap'], tk['pos'][:, :2]); hmax = ctx['hmap'][3].max()      # ground height under the root (training height map)
    on_deck = h > 0.9 * hmax; on_slope = (h > 0.01) & ~on_deck
    z = smooth(tk['pos'][:, 2], 0.3 * fps); n3 = int(0.3 * fps); ss, bs = shares(psi, phi, sp, fps)
    num = dict(seconds=round(len(P) / fps, 1), path_m=round(float(np.hypot(*np.diff(P, axis=0).T).sum()), 2), sideways_share=round(ss, 2),
               backward_share=round(bs, 2), root_z_gain_m=round(float(z[-1] - z[0]), 2), root_z_range_m=round(float(z.max() - z.min()), 2),
               slope_share=round(float(on_slope.mean()), 2), deck_share=round(float(on_deck.mean()), 2), ground_max_m=round(float(h.max()), 2),
               start_on_deck=bool(on_deck[:n3].mean() > 0.5), end_on_deck=bool(on_deck[-n3:].mean() > 0.5))
    idx = np.where(on_slope)[0]
    if len(idx) >= n3:
        i0, i1 = idx[0], idx[-1]; im = idx[len(idx) // 2]
        d = P[i1] - P[i0]; path = D(math.atan2(d[1], d[0])); body = (D(mean_ang(psi, im, n3)) + 180) % 360 - 180
        num.update(path_on_slope_x=round(path, 0), body_on_slope_x=round(body, 0),
                   turn_before=round(D(mean_ang(psi, i0, n3) - mean_ang(psi, 0, n3)), 0),
                   turn_after=round(D(mean_ang(psi, len(P) - 1, n3) - mean_ang(psi, i1, n3)), 0))
    return num, (o, yaw), P


def analyse_stairs(tk, ctx):
    o, yaw = base_frame(tk, ctx); fps = tk['fps']
    P, psi, phi, sp = kinematics(tk, o, yaw); ss, bs = shares(psi, phi, sp, fps); n3 = int(0.3 * fps)
    z = smooth(tk['pos'][:, 2], 0.3 * fps); m = tk['meta']
    num = dict(seconds=round(len(P) / fps, 1), path_m=round(float(np.hypot(*np.diff(P, axis=0).T).sum()), 2), sideways_share=round(ss, 2),
               backward_share=round(bs, 2), root_z_gain_m=round(float(z[-1] - z[0]), 2), body_start_x=round(D(mean_ang(psi, 0, n3)), 0),
               body_end_x=round(D(mean_ang(psi, len(P) - 1, n3)), 0), levels=m['levels'], turn_low_deg=m['turn_low_deg'],
               turn_high_deg=m['turn_high_deg'], facing_off_deg=m['facing_off_deg'], stop_and_go=m['stop_and_go'])
    return m['cls'], num, (o, yaw)


def analyse_driveway(tk, ctx):
    o, yaw = base_frame(tk, ctx); fps = tk['fps']
    P, psi, phi, sp = kinematics(tk, o, yaw); ss, bs = shares(psi, phi, sp, fps); n3 = int(0.3 * fps)
    d = P[-1] - P[0]; net = D(math.atan2(d[1], d[0])); z = tk['pos'][:, 2]
    num = dict(seconds=round(len(P) / fps, 1), path_m=round(float(np.hypot(*np.diff(P, axis=0).T).sum()), 2), sideways_share=round(ss, 2),
               backward_share=round(bs, 2), net_travel_x=round(net, 0), root_z_gain_m=round(float(z[-1] - z[0]), 2),
               turn_total=round(D(mean_ang(psi, len(P) - 1, n3) - mean_ang(psi, 0, n3)), 0), slope_deg=round(ctx['driveway']['slope_deg'], 1))
    a = abs(net)
    cat = 'straight uphill' if a < 30 else 'straight downhill' if a > 150 else 'diagonal across the slope'
    return cat, num, (o, yaw)


def classify(takes, ctx):
    for tk in takes:
        p = tk['prop']
        if p in ('4x4', '2x2'):
            cat, num, fr = analyse_beam(tk, ctx)
        elif p.startswith('stairs'):
            cat, num, fr = analyse_stairs(tk, ctx)
        elif p == 'ramp':
            num, fr, P = analyse_ramp(tk, ctx); cat = ramp_category(num)
        else:
            cat, num, fr = analyse_driveway(tk, ctx)
        tk['category'], tk['numbers'], tk['frame'] = cat, num, fr
        tk['P'] = to_frame(tk['pos'][:, :2], *fr)
    return takes


def ramp_category(n):
    if 'path_on_slope_x' not in n:
        return 'passes the ramp, not on it'
    p, b = n['path_on_slope_x'], n['body_on_slope_x']; up = n['root_z_gain_m'] > 0
    if abs(n['root_z_gain_m']) < 0.10:
        return 'over the ramp side to side'
    off = lambda a: abs(a) if up else 180 - abs(a)                   # off the slope line, in the direction of travel
    way = 'up' if up else 'down'
    if abs(n['turn_before']) >= TURN:
        return 'turn, then %s' % way
    if abs(n['turn_after']) >= TURN:
        return '%s, then turn' % way
    return ('oblique %s' if max(off(p), off(b)) >= OBL else 'straight %s') % way


# ---------------------------------------------------------------- grouping, representatives
HUES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e1e0d9'
BEAM_ORDER = ['straight over', 'oblique over', 'turn, then cross', 'cross, then turn', 'C / U turn', 'S turn', 'sidestep over', 'backward over',
              'over and back', 'no crossing (walk-up, stand at the beam or pass alongside)']
STAIR_ORDER = ['straight up', 'straight down', 'single step up, straight', 'single step down, straight', 'side approach then up',
               'down then turn', 'C (up)', 'C (down)', 'S (up)', 'S (down)']
RAMP_ORDER = ['straight up', 'oblique up', 'turn, then up', 'up, then turn', 'straight down', 'oblique down', 'turn, then down',
              'down, then turn', 'over the ramp side to side', 'passes the ramp, not on it']
DRIVE_ORDER = ['straight uphill', 'straight downhill', 'diagonal across the slope']
ORDER = {'4x4': BEAM_ORDER, '2x2': BEAM_ORDER, 'stairs1': STAIR_ORDER, 'stairs2': STAIR_ORDER, 'ramp': RAMP_ORDER, 'driveway': DRIVE_ORDER}
PREFIX = ('tp_t4x4_', 'pico_20261029_', 'tp_stairs_', 'tp_ramp_', 'tp_driveway_')


def short(tid):
    for p in PREFIX:
        if tid.startswith(p):
            tid = tid[len(p):]
    return tid.replace('_2x2', '')


def slug(s):
    return ''.join(ch if ch.isalnum() else '_' for ch in s.lower()).strip('_').replace('__', '_').replace('__', '_')


def groups(takes):
    out = collections.OrderedDict()
    for prop, _ in PROPS:
        ts = [t for t in takes if t['prop'] == prop]
        cats = sorted({t['category'] for t in ts}, key=lambda c: (ORDER[prop].index(c) if c in ORDER[prop] else 99, c))
        if ts:
            out[prop] = collections.OrderedDict((c, [t for t in ts if t['category'] == c]) for c in cats)
    return out


def style(prop, cats):
    """category -> (colour, linestyle). Stairs: one hue per shape, dashed = down; otherwise one hue per category."""
    st = {}
    if prop.startswith('stairs'):
        fam = lambda c: c.replace(' (up)', '').replace(' (down)', '').replace(' up', '').replace(' down', '').replace('down ', '').replace(',', '').strip()
        fams = []
        for c in STAIR_ORDER + list(cats):
            if fam(c) not in fams:
                fams.append(fam(c))
        for c in cats:
            st[c] = (HUES[fams.index(fam(c)) % 8], '--' if 'down' in c else '-')
    else:
        for i, c in enumerate(cats):
            st[c] = (HUES[i % 8], '-' if i < 8 else '--')
    return st


def resample(P, n=40):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if s[-1] < 1e-6:
        return np.repeat(P[:1], n, 0)
    u = np.linspace(0, s[-1], n)
    return np.c_[np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])]


def representatives(ts):
    """The medoid take (smallest summed path distance to the others) and, for 4 or more takes, the one farthest from it."""
    R = [resample(t['P']) for t in ts]
    Dm = np.array([[np.hypot(*(a - b).T).mean() for b in R] for a in R])
    i = int(np.argmin(Dm.sum(1))); out = [ts[i]]
    if len(ts) >= 4:
        out.append(ts[int(np.argmax(Dm[i]))])
    return out


# ---------------------------------------------------------------- top-view plots
def draw_prop(ax, prop, tk, ctx):
    from matplotlib.patches import Polygon
    for k, poly in enumerate(footprints(tk, ctx, *tk['frame']) if prop != 'ramp' else []):
        ax.add_patch(Polygon(poly, closed=True, fc='#efeee8', ec='#3d3d3a', lw=1.2, zorder=1))
    if prop == 'ramp':                                          # the training height map, in the ramp frame
        x0, y0, res, g = ctx['hmap']; o, yaw = tk['frame']
        X, Y = np.meshgrid(x0 + res * np.arange(g.shape[0]), y0 + res * np.arange(g.shape[1]), indexing='ij')
        F = to_frame(np.c_[X.ravel(), Y.ravel()], o, yaw)
        FX, FY = F[:, 0].reshape(g.shape), F[:, 1].reshape(g.shape)
        ax.contourf(FX, FY, g, levels=[0.01, 0.05, 0.10, 0.15, 0.25], colors=['#efeee8', '#e3e1d6', '#d5d2c3', '#c5c1ae'], zorder=1)
        ax.contour(FX, FY, g, levels=[0.01], colors=['#3d3d3a'], linewidths=1.0, zorder=1.2)
        ax.annotate('', xy=(0.95, 0), xytext=(0.15, 0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1), zorder=1.5)
        ax.text(0.55, 0.06, 'up (darker = higher)', color=MUTED, fontsize=7, ha='center', zorder=1.5)
    if prop.startswith('stairs'):
        ax.text(-0.05, 0.0, 'up →', color=MUTED, fontsize=7, ha='right', va='center', zorder=1.5)
    if prop == 'driveway':
        ax.annotate('', xy=(0.6, 0), xytext=(-0.6, 0), arrowprops=dict(arrowstyle='->', color=MUTED, lw=1), zorder=1.5)
        ax.text(0, 0.08, 'uphill, %.1f deg' % ctx['driveway']['slope_deg'], color=MUTED, fontsize=7, ha='center', zorder=1.5)


def draw_take(ax, tk, color, ls='-', lw=1.3, alpha=0.9, label=None):
    P = smooth(tk['P'], 0.2 * tk['fps'])
    ax.plot(P[:, 0], P[:, 1], ls, color=color, lw=lw, alpha=alpha, zorder=3, solid_capstyle='round')
    ax.plot(*P[0], 'o', color=color, ms=4.5, mec='white', mew=0.6, alpha=min(1, alpha + 0.1), zorder=4)
    k = max(2, int(0.25 * tk['fps']))
    d = P[-1] - P[-k]; d = d / (np.hypot(*d) + 1e-9) * 0.001
    ax.annotate('', xy=P[-1] + d, xytext=P[-1] - d * 60, zorder=4,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=0, mutation_scale=9, alpha=min(1, alpha + 0.1), shrinkA=0, shrinkB=0))
    if label:
        ax.annotate(label, P[0], xytext=(3, 3), textcoords='offset points', fontsize=6, color=INK, zorder=5)


def limits(ts, ctx):
    pts = np.concatenate([t['P'] for t in ts] + footprints(ts[0], ctx, *ts[0]['frame']))
    lo, hi = pts.min(0) - 0.25, pts.max(0) + 0.25
    if ts[0]['prop'] in ('4x4', '2x2'):                        # the beam is 2.4 m long; show what the takes use
        tp = np.concatenate([t['P'] for t in ts]); lo[1], hi[1] = tp[:, 1].min() - 0.3, tp[:, 1].max() + 0.3
    return lo, hi


def style_ax(ax, lo, hi, title):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
    ax.set_title(title, fontsize=8.5, color=INK, loc='left', pad=4)
    ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)


def plots(takes, ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    (HERE / 'panels').mkdir(exist_ok=True)
    for prop, cats in groups(takes).items():
        ts = [t for c in cats.values() for t in c]
        lo, hi = limits(ts, ctx); st = style(prop, list(cats))
        asp = (hi[1] - lo[1]) / (hi[0] - lo[0])
        ncol = 4 if asp > 0.8 else 3
        n = len(cats) + 1; nrow = math.ceil(n / ncol)
        pw = 16.0 / ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(16, nrow * (pw * 0.9 * asp + 0.75) + 1.3), dpi=100, squeeze=False, layout='constrained')
        ax = axs[0, 0]; draw_prop(ax, prop, ts[0], ctx)
        for c, tl in cats.items():
            for t in tl:
                draw_take(ax, t, st[c][0], st[c][1], lw=1.0, alpha=0.55)
        style_ax(ax, lo, hi, 'All %d takes, coloured by category' % len(ts))
        hs = [Line2D([0], [0], color=st[c][0], ls=st[c][1], lw=2, label='%s (%d)' % (c.split(' (walk-up')[0], len(tl))) for c, tl in cats.items()]
        for k, (c, tl) in enumerate(cats.items(), 1):
            ax = axs[k // ncol, k % ncol]; draw_prop(ax, prop, ts[0], ctx)
            for j, t in enumerate(tl):
                draw_take(ax, t, HUES[j % 8], '-' if j < 8 else '--', label=short(t['id']))
            style_ax(ax, lo, hi, '%s  (%d)' % (c, len(tl)))
            ax.plot([lo[0], lo[0] + 0.012 * (hi[0] - lo[0])], [hi[1]] * 2, color=st[c][0], lw=7, solid_capstyle='butt', zorder=6)
            ax.spines['top'].set_color(st[c][0]); ax.spines['top'].set_linewidth(2.5)
        for k in range(n, nrow * ncol):
            axs[k // ncol, k % ncol].axis('off')
        fig.suptitle('%s: %d takes in %d categories. Top view in the prop frame (metres), +x = %s. Dot = start, arrow = end.' % (
            PROP_NAME[prop], len(ts), len(cats), {'4x4': 'crossing direction (each take turned to cross left to right; no mirroring)',
                                                    '2x2': 'crossing direction (each take turned to cross left to right; no mirroring)',
                                                    'ramp': 'up the slope', 'driveway': 'uphill'}.get(prop, 'up the stairs')),
            fontsize=10.5, color=INK, x=0.01, ha='left')
        fig.legend(handles=hs, loc='outside lower left', ncol=min(5, len(hs)), fontsize=8, frameon=False,
                   title='Categories (takes); category panels colour each take separately, with its id at the start', title_fontsize=8, alignment='left')
        fig.savefig(HERE / ('topview_%s.png' % prop), dpi=100, facecolor='white'); plt.close(fig)
        for c, tl in cats.items():                              # the same panel on its own, to sit next to the GIFs
            fig, ax = plt.subplots(figsize=(4.2, 4.2 * 0.9 * asp + 0.7), dpi=100, layout='constrained'); draw_prop(ax, prop, ts[0], ctx)
            for j, t in enumerate(tl):
                draw_take(ax, t, HUES[j % 8], '-' if j < 8 else '--', label=short(t['id']))
            style_ax(ax, lo, hi, '%s  (%d)' % (c.split(' (walk-up')[0], len(tl)))
            fig.savefig(HERE / 'panels' / ('%s_%s.png' % (prop, slug(c))), dpi=100, facecolor='white'); plt.close(fig)
        print('plot', prop, n - 1, 'categories', flush=True)


# ---------------------------------------------------------------- MuJoCo GIFs
def build_model(tk, ctx):
    import mujoco
    G = mujoco.mjtGeom
    spec = x2_spec()
    try:
        spec.add_texture(name='cat_grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.80, 0.81, 0.83], rgb2=[0.74, 0.75, 0.78], width=256, height=256)
        mat = spec.add_material(name='cat_grid', texrepeat=[1, 1], texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'cat_grid'
        ground = dict(material='cat_grid')
    except Exception:
        ground = dict(rgba=[0.78, 0.79, 0.81, 1])
    floor = [g for g in spec.geoms if g.name == 'floor']
    for g in floor:                                             # the model's own floor: out of the way, ours replaces it
        g.pos = [0, 0, -50]
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    wb = spec.worldbody; c0 = tk['pos'][:, :2].mean(0)
    off = dict(contype=0, conaffinity=0)
    wb.add_light(pos=[c0[0], c0[1], 5], dir=[0.2, 0.2, -1], diffuse=[0.7, 0.7, 0.7])
    if tk['prop'] == 'driveway':
        dw = ctx['driveway']; a, b = dw['a'], dw['b']
        n = np.array([-a, -b, 1.0]); n /= np.linalg.norm(n)
        z0 = a * c0[0] + b * c0[1] + dw['c']
        ax = np.cross([0, 0, 1], n); s = np.linalg.norm(ax); ang = math.asin(s)
        q = np.r_[math.cos(ang / 2), ax / (s + 1e-12) * math.sin(ang / 2)]
        wb.add_geom(type=G.mjGEOM_BOX, size=[8, 8, 0.05], pos=(np.r_[c0, z0] - n * 0.05).tolist(), quat=q.tolist(), **ground, **off)
        return spec.compile()
    wb.add_geom(type=G.mjGEOM_PLANE, size=[30, 30, 0.05], pos=[c0[0], c0[1], -0.004 if tk['prop'] == 'ramp' else 0.0], **ground, **off)
    for j, b in enumerate(tk['boxes']):
        wb.add_geom(type=G.mjGEOM_BOX, size=list(b['half']), pos=list(b['center']), quat=[math.cos(b['yaw'] / 2), 0, 0, math.sin(b['yaw'] / 2)],
                    rgba=[0.74 - 0.07 * j, 0.55 - 0.05 * j, 0.32, 1], **off)
    if tk['prop'] == 'ramp':                                    # the training height map itself
        x0, y0, res, g = ctx['hmap']; zmax = float(g.max())
        spec.add_hfield(name='cat_ramp', nrow=g.shape[1], ncol=g.shape[0], size=[g.shape[0] * res / 2, g.shape[1] * res / 2, zmax, 0.01],
                        userdata=(g.T / zmax).ravel().tolist())
        wb.add_geom(type=G.mjGEOM_HFIELD, hfieldname='cat_ramp', pos=[x0 + g.shape[0] * res / 2, y0 + g.shape[1] * res / 2, 0],
                    rgba=[0.74, 0.58, 0.36, 1], **off)
    return spec.compile()


def caption(img, l1, l2):
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img, 'RGBA'); W, H = img.size
    try:
        f = ImageFont.load_default(size=10 if W >= 400 else 9)
    except TypeError:
        f = ImageFont.load_default()
    d.rectangle([0, H - 27, W, H], fill=(20, 20, 20, 190))
    d.text((4, H - 26), l1, font=f, fill=(255, 255, 255, 255)); d.text((4, H - 13), l2, font=f, fill=(215, 215, 210, 255))
    return img


def encode(frames, path, l1, l2, size, colors=64):
    from PIL import Image
    ims = [caption(Image.fromarray(f).resize(size, Image.LANCZOS) if f.shape[1] != size[0] else caption_copy(f), l1, l2) for f in frames]
    k = max(1, len(ims) // 8); sheet = Image.new('RGB', (size[0] * len(ims[::k]), size[1]))
    for i, im in enumerate(ims[::k]):
        sheet.paste(im, (i * size[0], 0))
    pal = sheet.quantize(colors=colors, method=Image.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    q[0].save(path, save_all=True, append_images=q[1:], duration=80, loop=0, optimize=False, disposal=1)
    return os.path.getsize(path)


def caption_copy(f):
    from PIL import Image
    return Image.fromarray(f).copy()


def render_gif(tk, ctx, path, strips=None):
    import mujoco
    m = build_model(tk, ctx); d = mujoco.MjData(m)
    W, H, FPS, MAXS = 400, 300, 12.5, 6.0
    T = len(tk['pos']); dur = T / tk['fps']; speed = max(1.0, dur / MAXS)
    idx = np.clip((np.arange(0, dur / speed, 1 / FPS) * speed * tk['fps']).astype(int), 0, T - 1)
    look = smooth(tk['pos'], 1.0 * tk['fps'])
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 2.9, -18, D(tk['frame'][1]) + 125
    frames = []
    with mujoco.Renderer(m, H, W) as r:
        for i in idx:
            d.qpos[:] = to_qpos(tk, i); mujoco.mj_forward(m, d)
            cam.lookat[:] = [look[i, 0], look[i, 1], look[i, 2] - 0.2]
            r.update_scene(d, cam); frames.append(r.render().copy())
    l1 = '%s / %s / %s' % (PROP_NAME[tk['prop']], tk['category'].split(' (walk-up')[0], tk['id'])
    l2 = 'MuJoCo kinematic playback, not the training simulator' + ('' if speed < 1.02 else '  (x%.1f speed)' % speed)
    size = (W, H); n = encode(frames, path, l1, l2, size)
    if n > 1.5e6:
        size = (320, 240); n = encode(frames, path, l1, l2, size, 48)
    if strips:
        from PIL import Image
        pick = [frames[int(len(frames) * f)] for f in (0.1, 0.35, 0.6, 0.9)]
        st = Image.new('RGB', (W * 4, H))
        for k, f in enumerate(pick):
            st.paste(Image.fromarray(f), (k * W, 0))
        st.save(Path(strips) / (Path(path).stem + '.png'))
    return dict(file='gifs/' + Path(path).name, bytes=n, size=list(size), frames=len(frames), speed=round(speed, 2), seconds=round(dur, 1))


def gifs(takes, ctx):
    (HERE / 'gifs').mkdir(exist_ok=True)
    out = {}; failed = {}
    strips = os.environ.get('CATALOG_STRIPS')
    for prop, cats in groups(takes).items():
        for c, tl in cats.items():
            for k, tk in enumerate(representatives(tl)):
                name = '%s_%s_%d_%s.gif' % (prop, slug(c)[:40], k + 1, short(tk['id']))
                try:
                    info = render_gif(tk, ctx, HERE / 'gifs' / name, strips)
                    info.update(take=tk['id'], role='medoid (typical)' if k == 0 else 'farthest from the medoid')
                    out.setdefault(prop, {}).setdefault(c, []).append(info)
                    print('gif', name, info['bytes'] // 1000, 'kB', info['size'], 'x%.1f' % info['speed'], flush=True)
                except Exception as e:                           # noqa: BLE001
                    failed[tk['id']] = repr(e); print('FAILED', name, repr(e), flush=True)
    json.dump(dict(gifs=out, failed=failed), open(HERE / 'gifs' / 'index.json', 'w'), indent=1)
    return out


# ---------------------------------------------------------------- categories.json, README.md, index.html
PREVIOUS = {'4x4': '17 straight, 19 L turns, 9 sideways walks, 4 C / U turns, 3 S, 4 oblique (by net body yaw and sideways share, not relative to the beam)',
            '2x2': '6 straight, 2 C, 1 L, 1 oblique', 'ramp': '10 straight, 3 oblique, 2 L (15 base takes)'}
COLS = {'beam': ['body_x', 'path_x', 'turn_before', 'turn_after', 'sideways_share', 'backward_share', 'seconds'],
        'stairs': ['turn_low_deg', 'turn_high_deg', 'facing_off_deg', 'root_z_gain_m', 'sideways_share', 'seconds'],
        'ramp': ['path_on_slope_x', 'body_on_slope_x', 'turn_before', 'turn_after', 'root_z_gain_m', 'sideways_share', 'seconds'],
        'driveway': ['net_travel_x', 'root_z_gain_m', 'turn_total', 'seconds']}
ABS = {'body_x', 'path_x', 'turn_before', 'turn_after', 'turn_low_deg', 'turn_high_deg', 'facing_off_deg', 'turn_total'}


def kind(prop):
    return 'beam' if prop in ('4x4', '2x2') else 'stairs' if prop.startswith('stairs') else prop


def cell(tl, key):
    v = [t['numbers'][key] for t in tl if key in t['numbers'] and not isinstance(t['numbers'][key], (bool, str))]
    if not v:
        return '-'
    v = np.abs(v) if key in ABS else np.array(v, float)
    f = (lambda x: '%.2f' % x) if ('share' in key or key.endswith('_m')) else (lambda x: '%.0f' % x if key != 'seconds' else '%.1f' % x)
    return f(np.median(v)) if len(v) == 1 else '%s (%s to %s)' % (f(np.median(v)), f(v.min()), f(v.max()))


def head(key):
    return ('abs %s' % key if key in ABS else key).replace('_', ' ')


def write_categories(takes, ctx, gi):
    g = groups(takes)
    doc = dict(
        generated_by='catalog/make_catalog.py',
        thresholds=dict(turn_deg=TURN, oblique_deg=OBL, sideways_deg=SIDE, across_slope_max_z_gain_m=0.10),
        conventions='angles in degrees, + = to the left seen from above; *_x = against +x of the prop frame (beams: the crossing direction; '
                    'stairs / ramp: up; driveway: uphill); body = pelvis heading, path = direction of travel; turn_before / turn_after = body '
                    'heading change start -> crossing -> end; shares are of the travelled distance',
        driveway_plane=ctx['driveway'], ramp_frame=ctx['ramp'],
        props={p: dict(name=PROP_NAME[p], takes=sum(len(v) for v in cats.values()), topview='topview_%s.png' % p,
                       categories={c: dict(count=len(tl), takes=[t['id'] for t in tl], panel='panels/%s_%s.png' % (p, slug(c)),
                                           gifs=gi.get(p, {}).get(c, [])) for c, tl in cats.items()}) for p, cats in g.items()},
        takes={t['id']: dict(prop=t['prop'], source=t['source'], category=t['category'], frame=dict(origin=[round(float(x), 3) for x in t['frame'][0]],
                             yaw_deg=round(D(t['frame'][1]), 1)), numbers=t['numbers']) for t in takes})
    json.dump(doc, open(HERE / 'categories.json', 'w'), indent=1)
    return doc


NOTES = """## How the takes were grouped

Every take is put in a frame fixed to its prop (origin at the prop, +x = the crossing direction) and measured there:

- **body** = pelvis heading, **path** = direction of travel, both against +x at the crossing (`body_x`, `path_x`).
- **turn before / after** = change of the body heading from the start to the crossing and from the crossing to the end.
- **sideways / backward share** = share of the travelled distance with the travel direction 50 to 130 deg / over 130 deg off the body heading.
- Beams (4x4, 2x2 log): crossing = the root passing the beam line within the beam's length. Rules, first match wins: no crossing; more than one
  crossing (over and back); travel 50 to 130 deg off the body at the crossing (sidestep over); over 130 deg (backward over); a body turn of
  {turn:.0f} deg or more both before and after (C / U turn if the same way, S turn if opposite ways); only before (turn, then cross); only after
  (cross, then turn); body {obl:.0f} deg or more off square, or path {obl2:.0f} deg or more (oblique over); otherwise straight over.
  Each beam take is turned by 180 deg if needed so that it crosses towards +x; nothing is mirrored, so left and right turns stay apart in the panels.
- Stairs: the class comes from `data/train_sets/tp_stairs_all_v1_meta.json` (per take: levels touched, turn on the floor, turn on the top, facing
  against the stair line); it was not re-derived here. Numbers added: root height gain, sideways share, duration.
- Ramp (15 base takes of `acp_tp_ramp_sh2_x30.pkl`, names without `_sh`): root height gain under 0.10 m from start to end = over the ramp
  side to side (floor, a few centimetres up the slope, floor again); otherwise up or down, with "turn" for a body turn of {turn:.0f} deg or more before or after the slope and "oblique" for body or path
  {obl:.0f} deg or more off the slope line.
- Driveway (6 takes, `data/2026-10-30/x2`, resampled to 50 fps): net travel within 30 deg of uphill / downhill = straight, otherwise diagonal. The
  ground is a plane fitted here to the X2's own planted feet ({slope:.1f} deg); `props/driveway.json` is at human scale.

In the forward step-overs the travel direction sits a steady 10 to 20 deg to the left of the pelvis heading (`body_vs_path` in `categories.json`),
on both capture days and whichever way the beam is crossed, while the sidestep takes are square to the beam. That looks like a pelvis-yaw offset
of the retarget rather than a property of the walks; "oblique" is therefore decided on the body heading first.

GIFs are MuJoCo kinematic playback (root pose and 31 joints set per frame, no physics, not the training simulator), camera three-quarter view
following the root, 12.5 fps. The ramp is drawn from the training height map (`tp_ramp_hmap_x30.npz`) in the plots and in the GIFs, and
the ramp frame (origin at the foot of the slope, +x up) is taken from that map too. `props/ramp.json` shifted by +30 m is a different fit: about
2.2 m wide and turned against the height map's slope (`ramp_frame` in `categories.json` has the difference); the height map's ramp is about 0.8 m wide.
"""


def docs(takes, ctx):
    gi = json.load(open(HERE / 'gifs' / 'index.json'))['gifs'] if (HERE / 'gifs' / 'index.json').exists() else {}
    doc = write_categories(takes, ctx, gi); g = groups(takes)
    notes = NOTES.format(turn=TURN, obl=OBL, obl2=OBL + 10, slope=ctx['driveway']['slope_deg'])
    md = ['# Pico terrain captures: catalogue by prop and approach shape', '',
          'Generated by `python catalog/make_catalog.py` (working tree only). %d takes on %d props; every take with its category and numbers is in '
          '`categories.json`.' % (len(takes), len(g)), '', '| Prop | Takes | Categories (takes) |', '| --- | --- | --- |']
    for p, cats in g.items():
        md.append('| %s | %d | %s |' % (PROP_NAME[p], sum(map(len, cats.values())), ', '.join('%s (%d)' % (c.split(' (walk-up')[0], len(tl)) for c, tl in cats.items())))
    md += ['', notes]
    css = ('body{font:14px/1.45 system-ui,Segoe UI,sans-serif;color:#1a1a19;background:#fff;max-width:1640px;margin:20px auto;padding:0 16px}'
           'table{border-collapse:collapse;margin:8px 0 16px}td,th{border:1px solid #e1e0d9;padding:4px 8px;text-align:left;vertical-align:top;font-size:13px}'
           'th{background:#f6f5f0}img{max-width:100%}.cat{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;margin:6px 0 22px}'
           '.cat img{border:1px solid #e1e0d9}h2{margin-top:36px;border-top:2px solid #1a1a19;padding-top:10px}h3{margin:18px 0 4px}'
           '.ids{color:#6b6a63;font-size:12px;max-width:1200px}pre{white-space:pre-wrap;background:#f6f5f0;padding:10px;font:12px/1.4 ui-monospace,monospace}')
    ht = ['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Pico terrain captures: catalogue</title><style>%s</style></head><body>' % css,
          '<h1>Pico terrain captures: catalogue by prop and approach shape</h1>',
          '<p>%d takes on %d props. Generated by <code>catalog/make_catalog.py</code>; all numbers per take in <code>categories.json</code>.</p>' % (len(takes), len(g)),
          '<table><tr><th>Prop</th><th>Takes</th><th>Categories (takes)</th></tr>']
    for p, cats in g.items():
        ht.append('<tr><td><a href="#%s">%s</a></td><td>%d</td><td>%s</td></tr>' % (p, PROP_NAME[p], sum(map(len, cats.values())),
                  ', '.join('%s (%d)' % (c.split(' (walk-up')[0], len(tl)) for c, tl in cats.items())))
    ht.append('</table>')
    for p, cats in g.items():
        cols = COLS[kind(p)]; n = sum(map(len, cats.values()))
        md += ['## %s (%d takes)' % (PROP_NAME[p], n), '']
        ht += ['<h2 id="%s">%s (%d takes)</h2>' % (p, PROP_NAME[p], n)]
        if p in PREVIOUS:
            md += ['Earlier classification: %s.' % PREVIOUS[p], '']; ht.append('<p>Earlier classification: %s.</p>' % PREVIOUS[p])
        md += ['![%s top view](topview_%s.png)' % (PROP_NAME[p], p), '',
               '| Category | Takes | ' + ' | '.join(head(c) for c in cols) + ' | Take ids |', '| --- | --- |' + ' --- |' * (len(cols) + 1)]
        ht += ['<img src="topview_%s.png" alt="%s top view">' % (p, PROP_NAME[p]),
               '<table><tr><th>Category</th><th>Takes</th>' + ''.join('<th>%s</th>' % head(c) for c in cols) + '<th>Take ids</th></tr>']
        for c, tl in cats.items():
            ids = ', '.join(short(t['id']) for t in tl)
            md.append('| %s | %d | %s | %s |' % (c, len(tl), ' | '.join(cell(tl, k).replace('|', '\\|') for k in cols), ids))
            ht.append('<tr><td>%s</td><td>%d</td>%s<td class="ids">%s</td></tr>' % (c, len(tl), ''.join('<td>%s</td>' % cell(tl, k) for k in cols), ids))
        md += ['', 'Median (min to max) over the takes of the category; abs = absolute value. Ids are short for `%s`.' % tl[0]['id'], '']
        ht += ['</table><p class="ids">Median (min to max) over the takes of the category; abs = absolute value.</p>']
        for c, tl in cats.items():
            gl = gi.get(p, {}).get(c, [])
            md += ['### %s: %s (%d)' % (PROP_NAME[p], c, len(tl)), '',
                   '<img src="panels/%s_%s.png" height="300"> ' % (p, slug(c)) + ' '.join('<img src="%s" height="300">' % x['file'] for x in gl), '']
            ht += ['<h3>%s (%d)</h3><div class="cat"><img src="panels/%s_%s.png" style="height:300px">' % (c, len(tl), p, slug(c))
                   + ''.join('<img src="%s" style="height:300px" alt="%s">' % (x['file'], x['take']) for x in gl) + '</div>']
            if gl:
                cap = '; '.join('`%s` %s%s' % (short(x['take']), x['role'], '' if x['speed'] < 1.02 else ', x%.1f speed' % x['speed']) for x in gl)
                md += ['GIFs: ' + cap + '.', '']; ht.append('<p class="ids">GIFs: %s.</p>' % cap.replace('`', ''))
    ht += ['<h2>How the takes were grouped</h2><pre>%s</pre></body></html>' % notes.replace('<', '&lt;')]
    open(HERE / 'README.md', 'w', encoding='utf8').write('\n'.join(md) + '\n')
    open(HERE / 'index.html', 'w', encoding='utf8').write('\n'.join(ht) + '\n')
    for p, cats in g.items():
        print(p, {c.split(' (walk-up')[0]: len(tl) for c, tl in cats.items()})


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    takes = load_takes()
    ctx = dict(ramp=ramp_prop(), hmap=ramp_hmap())
    ctx['driveway'] = fit_driveway(takes)
    classify(takes, ctx)
    if what == 'debug':
        for tk in takes:
            print(tk['prop'], tk['id'], '|', tk['category'], '|', json.dumps(tk['numbers']))
    if what in ('all', 'plots'):
        plots(takes, ctx)
    if what in ('all', 'gifs'):
        gifs(takes, ctx)
    if what == 'onegif':                                         # python make_catalog.py onegif <take id> <out.gif>
        tk = [t for t in takes if t['id'] == sys.argv[2]][0]
        print(render_gif(tk, ctx, sys.argv[3], os.environ.get('CATALOG_STRIPS')))
    if what in ('all', 'classify', 'plots', 'gifs', 'docs'):
        docs(takes, ctx)
    print('DONE', what)
