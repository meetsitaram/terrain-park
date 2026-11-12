"""Classify the candidate clips by path shape in the stair frame, write tp_stairs_all_v1 (clips passing every check),
its props / meta files and the top-view PNG."""
import numpy as np, json, joblib, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = joblib.load('all_candidates.pkl'); P = json.load(open('all_candidates_props.json')); CH = json.load(open('checks.json')); G = json.load(open('geom.json'))
FPS = 50; TH = 50.0


def info(name):
    c = C[name]; ch = CH[name]; pl = str(ch['place']); o = np.array(G['FR'][pl][0]) + [10.0, 0.0]; y = G['FR'][pl][1]
    up = np.array([np.cos(y), np.sin(y)]); n = np.array([-up[1], up[0]]); S = G['SURF'][pl]; mid = (S[0][0] + S[1][0]) / 2
    r = c['root_trans_offset'][:, :2].astype(float); s = (r - o) @ up; l = (r - o) @ n
    x, yy, z, w = c['root_rot'].astype(float).T; yaw = np.unwrap(np.arctan2(2 * (w * z + x * yy), 1 - 2 * (yy * yy + z * z)))
    psi = np.degrees(yaw - y); k = np.ones(15) / 15; ps = np.convolve(np.pad(psi, 7, mode='edge'), k, 'valid')
    t1, t2 = S[0][4], S[1][4]; zr = c['root_trans_offset'][:, 2].astype(float)
    zs = np.convolve(np.pad(zr, 12, mode='edge'), np.ones(25) / 25, 'valid') - 0.685; seq = []; cur = None
    for h in zs:      # support level from the root height, with a 2 cm dead band
        cand = 0 if h < t1 / 2 - 0.02 else 2 if h > (t1 + t2) / 2 + 0.02 else 1 if t1 / 2 + 0.02 < h < (t1 + t2) / 2 - 0.02 else cur
        if cand is None: cand = 0 if h < t1 / 2 else 2 if h > (t1 + t2) / 2 else 1
        if cand != cur: seq.append(cand); cur = cand
    mono = all(b > a for a, b in zip(seq, seq[1:])) or all(b < a for a, b in zip(seq, seq[1:]))
    upw = seq[-1] > seq[0]
    low = s < mid
    # heading change on the low side (floor and first half of the flight) and on the high side
    def turn(mask):
        idx = np.where(mask)[0]
        return float(ps[idx[-1]] - ps[idx[0]]) if len(idx) > 5 else 0.0
    tl, th = turn(low), turn(~low)
    zone = (s > S[0][0] - 0.15) & (s < S[1][0] + 0.15)
    off = float(np.median((ps[zone] - (0.0 if upw else 180.0) + 180) % 360 - 180)) if zone.any() else 0.0
    sp = np.convolve(np.linalg.norm(np.diff(r, axis=0), axis=1) * FPS, np.ones(15) / 15, 'same'); st = (sp < 0.05).astype(int)
    pauses = sum(0.5 < i / FPS < len(s) / FPS - 0.8 for i in np.where(np.diff(st) == 1)[0])
    d = dict(seq=''.join(map(str, seq)), turn_low=round(tl), turn_high=round(th), facing_off=round(float(off)), s=s, l=l, place=int(pl),
             T=len(s) / FPS, pauses=int(pauses), approach_deg=round(abs(tl)) if upw else None)
    if len(seq) < 2: cls = 'other'
    elif not mono: cls = 'continuous up-and-down'
    elif 40 <= abs(off): cls = 'sideways or oblique body, ' + ('up' if upw else 'down')
    elif not (0 in seq and 2 in seq):
        cls = 'single step ' + ('up' if upw else 'down') + (', straight' if max(abs(tl), abs(th)) < TH else ' with turn')
    elif abs(tl) < TH and abs(th) < TH: cls = 'straight ' + ('up' if upw else 'down')
    elif abs(tl) >= TH and abs(th) >= TH: cls = ('C' if tl * th > 0 else 'S') + (' (up)' if upw else ' (down)')
    elif abs(tl) >= TH: cls = 'side approach then up' if upw else 'down then turn'
    else: cls = 'L: up then turn on top' if upw else 'L: turn on top then down'
    d['cls'] = cls
    return d


I = {k: info(k) for k in C}
meta = {}
for k, d in I.items():
    ch = CH[k]
    meta[k] = dict(cls=d['cls'], place=d['place'], levels=d['seq'], turn_low_deg=d['turn_low'], turn_high_deg=d['turn_high'], facing_off_deg=d['facing_off'],
                   seconds=round(d['T'], 1), stop_and_go=bool(d['pauses'] >= 2 and d['T'] > 10), in_set=bool(ch['ok']), checks={a: ch[a] for a in ch if a not in ('eps', 'place')})
for k, ch in CH.items():
    if k not in meta:
        meta[k] = dict(cls='other (no stair contact)', place=ch['place'], in_set=False, excluded=ch.get('excluded', ''))
keep = [k for k in C if CH[k]['ok']]
joblib.dump({k: C[k] for k in keep}, 'tp_stairs_all_v1.pkl')
json.dump({k: P[k] for k in keep}, open('tp_stairs_all_v1_props.json', 'w'), indent=1)
json.dump(meta, open('tp_stairs_all_v1_meta.json', 'w'), indent=1)
by = collections.defaultdict(list)
for k in sorted(meta): by[meta[k]['cls']].append(k[10:] + ('p' if meta[k].get('stop_and_go') else '') + ('' if meta[k]['in_set'] else '*'))
for c in sorted(by, key=lambda c: -len(by[c])): print('%-28s %2d (in set %2d): %s' % (c, len(by[c]), sum('*' not in x for x in by[c]), ' '.join(by[c])))
print('set:', len(keep), 'clips, %.0f s' % sum(len(C[k]['dof']) / FPS for k in keep))
for k in sorted(I):
    d = I[k]
    if d['cls'] in ('side approach then up', 'down then turn') or d['cls'][0] in 'CSL':
        print(k[10:], d['cls'], 'levels', d['seq'], 'turn low %+d high %+d' % (d['turn_low'], d['turn_high']), 'facing', d['facing_off'], 'ok' if CH[k]['ok'] else 'FAIL')

# ---- top view, stair frame (place 2 clips overlaid in their own stair frame)
classes = sorted(by, key=lambda c: -len(by[c])); cm = plt.get_cmap('tab20'); col = {c: cm(i) for i, c in enumerate(classes)}
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
for ax, pl in zip(axs, (1, 2)):
    for s0, s1, l0, l1, top in G['SURF'][str(pl)]:
        ax.add_patch(plt.Rectangle((s0, l0), s1 - s0, l1 - l0, fc=(0.75, 0.75, 0.75, 0.35 if top < 0.2 else 0.6), ec='k', lw=1.5))
        ax.text(s0 + 0.02, l1 - 0.08, 'top %.1f cm' % (100 * top), fontsize=9)
    for k, d in I.items():
        if d['place'] != pl: continue
        ok = CH[k]['ok']
        ax.plot(d['s'], d['l'], color=col[d['cls']], lw=1.8 if ok else 1.0, ls='-' if ok else ':')
        ax.plot(d['s'][0], d['l'][0], 'o', color=col[d['cls']], ms=5); ax.annotate('', xy=(d['s'][-1], d['l'][-1]), xytext=(d['s'][-6], d['l'][-6]), arrowprops=dict(arrowstyle='->', color=col[d['cls']]))
        ax.text(d['s'][-1], d['l'][-1], k[12:], fontsize=7, color=col[d['cls']])
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_xlabel('s: along the climb (m), 0 = front edge of step 1'); ax.set_ylabel('l: to the left (m)')
    ax.set_title('stair place %d: root paths (o start, arrow end; dotted = failed a check, not in the set)' % pl)
axs[0].legend(handles=[plt.Line2D([], [], color=col[c], lw=3, label='%s (%d)' % (c, len(by[c]))) for c in classes], loc='upper left', fontsize=9)
plt.tight_layout(); plt.savefig('tp_stairs_all_v1_topview.png', dpi=80)
