"""Cut list and bill of materials from the piece model (cad/model.py).

- Lumber: every board with its cut length (whole 1/8 in, rounded up), packed into store lengths (8-16 ft) by
  first-fit decreasing with a 1/8 in saw kerf; the cheapest mix per section size.
- Sheets: every sheet part packed onto 4 x 8 sheets (guillotine shelf packing, kerf 1/8 in).
- Bought items, fasteners (rounded up to store packs) and 3D prints counted per BOM key.

Run: python cad/tools/bom.py      -> build/cutlist.json and prints a summary
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import model as M
import params as P

KERF = 0.125
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'build')


def eighth(x):
    return math.ceil(x * 8 - 1e-6) / 8


def fmt_in(x):
    w = int(x)
    f = round((x - w) * 8)
    if f == 8:
        w, f = w + 1, 0
    if f == 0:
        return '%d"' % w
    g = math.gcd(f, 8)
    return '%d-%d/%d"' % (w, f // g, 8 // g) if w else '%d/%d"' % (f // g, 8 // g)


def stock_lengths(nom):
    lens = [8, 10, 12, 14, 16]
    if nom == '6x6':
        lens = [8, 10, 12, 14, 16]
    return [L * 12 for L in lens]


def pack_lumber(cuts, nom):
    """cuts: [(length, label)] -> list of boards [(stock_len, [(len, label)...])]. FFD; for each new board pick the
    shortest stock length that fits the longest remaining cut, then fill."""
    remaining = sorted(cuts, key=lambda c: -c[0])
    boards = []
    lens = stock_lengths(nom)
    for L, lab in remaining:
        placed = False
        for b in boards:
            used = sum(c[0] for c in b[1]) + KERF * len(b[1])
            if used + L <= b[0]:
                b[1].append((L, lab))
                placed = True
                break
        if not placed:
            fit = [s for s in lens if s >= L]
            if not fit:
                raise ValueError('%s cut %.1f in longer than any stock length' % (nom, L))
            boards.append([fit[0] if L > 96 else 96, [(L, lab)]])
    # try to merge: upgrade a board to a longer stock length if that absorbs another 8 ft board's cuts
    return boards


def pack_sheets(parts, W=48.0, H=96.0):
    """parts: [(a, b, label)] rectangles (any rotation) -> list of sheets, each a list of placed parts.
    Shelf packing along the 96 in length."""
    rects = sorted(parts, key=lambda r: -max(r[0], r[1]))
    sheets = []                                                      # each: dict(shelves=[[height, used_w]], parts=[])
    for a, b, lab in rects:
        long_, short = max(a, b), min(a, b)
        if long_ > H + 1e-6 or short > W + 1e-6:
            raise ValueError('sheet part %s %.1f x %.1f does not fit a 4x8' % (lab, a, b))
        done = False
        for orient in ((short, long_), (long_, short)):              # (width across the sheet, height along it)
            w, h = orient
            if w > W + 1e-6 or h > H + 1e-6:
                continue
            for s in sheets:
                for shelf in s['shelves']:
                    if h <= shelf[0] + 1e-6 and shelf[1] + w <= W + 1e-6:
                        shelf[1] += w + KERF
                        s['parts'].append((a, b, lab))
                        done = True
                        break
                if not done and sum(sh[0] + KERF for sh in s['shelves']) + h <= H + 1e-6:
                    s['shelves'].append([h, w + KERF])
                    s['parts'].append((a, b, lab))
                    done = True
                if done:
                    break
            if done:
                break
        if not done:
            w, h = (short, long_)
            sheets.append(dict(shelves=[[h, w + KERF]], parts=[(a, b, lab)]))
    return sheets


# fastener packs as sold (count per pack) and what each kind is
FASTENER_PACKS = {
    'deck screw 3 in': ('deck screw 3 in (#10, star drive, exterior)', 350),         # ~5 lb box
    'deck screw 1-5/8 in': ('deck screw 1-5/8 in (#8, star drive, exterior)', 500),
    'screw 1-5/8 in': ('deck screw 1-5/8 in (#8, star drive, exterior)', 500),
    'carriage bolt 1/2 x 6 in': ('carriage bolt 1/2 x 6 in + nut + washer', 1),
    'carriage bolt 1/2 x 8 in': ('carriage bolt 1/2 x 8 in + nut + washer', 1),
    'carriage bolt 1/2 x 10 in': ('carriage bolt 1/2 x 10 in + nut + washer', 1),
    'carriage bolt 1/2 x 12 in': ('carriage bolt 1/2 x 12 in + nut + washer', 1),
    'lag screw 3/8 x 4 in': ('lag screw 3/8 x 4 in + washer', 1),
    'joist hanger 2x8 + nails': ('joist hanger 2x8 (Simpson LUS28) + 10d hanger nails', 1),
    'face-mount hanger, double 2x10': ('face-mount hanger double 2x10 (Simpson LUS210-2)', 1),
    'joist hanger / stair angle': ('stair stringer connector (Simpson LSCZ)', 1),
}


def build():
    m = M.build()
    lumber = defaultdict(list)
    sheet_parts = defaultdict(list)
    buy = Counter()
    buy_where = defaultdict(set)
    prints = Counter()
    for p in m.pieces:
        stage = p.group.split('/')[0]
        if p.kind == 'lumber':
            L = eighth(p.cut['length'])
            lumber[p.stock].append((L, '%s: %s' % (p.group.split('/', 1)[1], p.name)))
        elif p.kind == 'sheet':
            a, b = p.cut['dims']
            sheet_parts[p.stock].append((eighth(a), eighth(b), '%s: %s' % (p.group.split('/', 1)[1], p.name)))
        elif p.kind == 'buy':
            buy[p.buy] += 1
            buy_where[p.buy].add(stage)
        elif p.kind == 'print':
            prints[p.buy or p.name] += 1
    # bought items are counted per piece; several pieces make one item
    PER_ITEM = {'prehung door': 7, 'lever passage set': 5, 'LED push button 60 mm': 3, 'porcelain lampholder': 2,
                'ball valve 3/4 in': 3, 'base cabinet B30': 9, 'base cabinet DB30': 14, 'base cabinet B36': 9,
                'wall cabinet W3030': 4, '5-digit display': 2, 'casing': 2, 'cabinet pull': 1, 'target sticker': 2}
    items = {}
    for k, n in buy.items():
        per = PER_ITEM.get(k, 1)
        items[k] = dict(qty=max(1, round(n / per)), where=sorted(buy_where[k]))
    fast = Counter()
    fast_where = defaultdict(set)
    for f in m.fasteners:
        fast[f['kind']] += f['count']
        fast_where[f['kind']].add(f['group'].split('/')[0])
    fasteners = {}
    for k, n in fast.items():
        desc, pack = FASTENER_PACKS.get(k, (k, 1))
        e = fasteners.setdefault(desc, dict(count=0, pack=pack, where=set()))
        e['count'] += n
        e['where'] |= fast_where[k]
    for e in fasteners.values():
        e['count'] = int(math.ceil(e['count'] * 1.1))                # 10 % spare
        e['packs'] = int(math.ceil(e['count'] / e['pack']))
        e['where'] = sorted(e['where'])
    boards = {nom: pack_lumber(c, nom) for nom, c in lumber.items()}
    sheets = {k: pack_sheets(v) for k, v in sheet_parts.items()}
    return m, dict(lumber={nom: dict(cuts=sorted(c, key=lambda x: (-x[0], x[1])),
                                     boards=[dict(stock=b[0], cuts=b[1]) for b in bs])
                           for (nom, c), bs in ((kv, boards[kv[0]]) for kv in lumber.items())},
                   sheets={k: dict(parts=v, sheets=[s['parts'] for s in sheets[k]]) for k, v in sheet_parts.items()},
                   items=items, fasteners=fasteners, prints=dict(prints))


def summary(d):
    print('LUMBER')
    for nom, e in sorted(d['lumber'].items()):
        c = Counter(b['stock'] // 12 for b in e['boards'])
        print('  %-5s %3d cuts -> %s' % (nom, len(e['cuts']), ', '.join('%d x %d ft' % (n, L) for L, n in sorted(c.items()))))
    print('SHEETS')
    for k, e in d['sheets'].items():
        print('  %-4s %3d parts -> %d sheets' % (k, len(e['parts']), len(e['sheets'])))
    print('ITEMS')
    for k, e in sorted(d['items'].items()):
        print('  %3d  %s' % (e['qty'], k))
    print('FASTENERS')
    for k, e in sorted(d['fasteners'].items()):
        print('  %4d (%d packs of %d)  %s' % (e['count'], e['packs'], e['pack'], k))
    print('PRINTS', d['prints'])


if __name__ == '__main__':
    m, d = build()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'cutlist.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, default=list)
    summary(d)
