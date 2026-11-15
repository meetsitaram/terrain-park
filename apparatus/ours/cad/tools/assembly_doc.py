"""Write docs/ASSEMBLY.md: the build guide, one section per step of cad/assembly_steps.py.

For each step the parts list comes from the model itself (cad/model.py + cad/fasteners.py), matched to the step's
sub-steps the same way the animation does (first sub-step whose regex finds '<group> / <name>'), so the guide, the
animations (sim/assembly_anim.py) and the BOM (cad/tools/shop.py) never disagree.

Run: python cad/tools/assembly_doc.py
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import assembly_steps as AS
import fasteners as F
import model as M
from bom import eighth, fmt_in

# the how-to text per step (the parts lists are generated)
GUIDE = {
    '01-deck': dict(
        steps=['Each panel is 4 x 8 ft: cut 2 end rims and 9 joists from 2x4 (lengths below).',
               'Stand the rims on edge, 48 in apart; set the joists between them on edge, 12 in on centre, ends flush.',
               'Drive 2 x 3 in screws through the rim into the end of every joist. Check the diagonals match (square).',
               'Run a bead of subfloor adhesive on every top edge, drop the 3/4 plywood on, flush to the rims.',
               'Screw the sheet down with 1-5/8 in screws every 6 in along every joist and rim.',
               'Build 9 panels. Set them end to end on a level floor (shim low spots), and screw neighbouring rims '
               'together with 3 in screws every 12 in.'],
        check='All 9 panels flat within 1/8 in across the joints; total lane 36 ft x 8 ft.'),
    '02-walls': dict(
        steps=['Stand 4x4 posts outside both long sides of the deck, 2 per 4 ft section, tops 36 in above the deck.',
               'Bolt each post through the outside joist with 2 carriage bolts (drill 1/2 in, washer + nut inside).',
               'Screw three 2x4 rails per section to the inside faces of the posts, tops at 12, 24 and 36 in.',
               'At the start end: a fixed rail panel on the back half, a hinged gate on the front half.',
               'At section 6 (counter side), hang the side gate.',
               'Stain all rails and posts (BEHR ST-533 Cedar Naturaltone, 2 coats).'],
        check='Posts plumb; rails level; gates swing freely and latch.'),
    '03-door-wall': dict(
        steps=['Frame the wall flat on the deck: bottom and top plates, studs 16 in o.c., king + jack studs and a '
               'header around the 38 x 82-1/2 in rough opening.',
               'Screw the frame together with 3 in screws (2 per stud end).',
               'Stand it up at x = 96 in (8 ft from the start), screw the bottom plate to the deck joists.',
               'Skin both faces with OSB: 1-5/8 in screws every 6 in on edges, 12 in in the field.',
               'Brace each end with two 2x4 A-braces down to deck cleats.'],
        check='Wall plumb both ways; opening square (diagonals equal).'),
    '04-door': dict(
        steps=['Set the 36 x 80 prehung unit in the opening; shim behind the hinges and the strike; screw the jambs.',
               'Swap the 3 plain hinges for spring hinges; set the tension so the door closes from 90 deg.',
               'Fit the lever set at 36 in; trim both faces with casing.'],
        check='Door swings toward the start, closes and latches on its own from any angle.'),
    '05-keypad': dict(
        steps=['Screw a post base to the deck (over a joist) at the right of the door; set the 4x4 post.',
               'Tie the post to the door wall with two 2x4 ties.',
               'Build the plywood panel box; mount 10 dome buttons, the display and the electronics box.'],
        check='Panel does not move when a button is hit hard.'),
    '06-task-boards': dict(
        steps=['Lightbulb boards: screw the plywood boards to the door wall, lampholders at wall and floor height.',
               'Button post: post base + 4x4, kickers to the deck, two boxed boards with 5 buttons each.',
               'Valve post: two boards; screw on the painted 2x4 crosses; mount 10 ball valves on close nipples '
               'strapped to wood blocks.'],
        check='Every button clicks; every valve turns a full quarter turn without hitting the board.'),
    '07-terrain': dict(
        steps=['Screw the two 4x4 stepovers to the deck in a V (6 in structural screws).',
               'Cut 2x8 ribs to a 15 deg gable (rise 6-7/16 in over 24 in); 5 ribs per 4 x 4 ft cell, in hangers '
               'or screwed through the end ribs.',
               'Skin each ramp face with 3/4 plywood; ridges of neighbouring cells cross at 90 deg (pinwheel).',
               'Build the small wall obstacle (2x4 frame, plywood both faces) and screw it down.'],
        check='Walk every ramp: no flex, no proud screw heads.'),
    '08-landing': dict(
        steps=['Screw post bases to the deck over joists; set the 4x4 posts.',
               'Through-bolt doubled 2x8 beams to the posts.',
               'Hang 2x8 joists between the beams, 12 in o.c. (LUS28 hangers, 10 nails each).',
               'Glue and screw the 3/4 plywood deck; skin the open sides with OSB.',
               'Guard posts and three rails around the open sides.'],
        check='Stand on every corner: no movement. Deck 44 in above the lane deck.'),
    '09-stairs': dict(
        steps=['Cut 4 stringers from 2x12: 6 risers of 7-5/16 in, 5 treads of 8-3/4 in.',
               'Hang the tops on the landing beam with stringer connectors; screw the feet to the deck.',
               'Screw 2x10 treads (2 x 3 in per stringer) and plywood risers.',
               'Newel posts, handrails and the side skirt.'],
        check='Every tread level; rise varies by less than 3/16 in.'),
    '10-buttons': dict(steps=['Screw the 5 location button pads (2x6) where the plan shows; mount the buttons.'],
                       check='Buttons sit flush enough that a robot foot cannot catch the pad.'),
    '11-counter': dict(
        steps=['Set the base cabinets (B30, DB36, B30) on the deck along the back wall of section 7; level and screw '
               'them together and to the deck.',
               'Set the 8 ft countertop; screw up through the cabinet corner blocks.',
               'Build the 4x4 stand with braces; hang the W3024 wall cabinet 15 in above the counter.',
               'Stick the drill targets inside the wall cabinet.'],
        check='Drawers and doors open fully; nothing rocks.'),
    '12-objects': dict(
        steps=['Place the totes, crates, cylinders, drill and bulbs as in the spec pictures.',
               'Roll the gantry to the start; lock the casters.'],
        check='Compare with the spec photos (docs/spec).'),
}

DISCREPANCIES = [
    ('Stairs are steeper than code', 'The spec flight fits one 4 ft section: 6 risers of 7-5/16 in and 8-3/4 in '
     'treads, about 40 deg. Residential code (IRC) asks for a 7-3/4 in rise at most and a 10 in tread at least. The '
     'robots are meant to be tested on this stair, so the model keeps the spec. People should use the side gate, '
     'not the stairs.'),
    ('Landing guard is 36 in, not 42 in', 'Matches the spec pictures. Code asks 42 in for a walking surface more '
     'than 30 in up. Fine for robots; add a temporary top rail if people work up there.'),
    ('Landing guard posts were attached to beam end grain', 'The first fastener pass put 1/2 in bolts through the '
     'guard posts into the ends of the landing beams, which does not work (a bolt cannot go along a beam). The '
     'generator now uses 6 in structural screws through the post into the beam end, plus the post base at the '
     'bottom.'),
    ('Landing guards were built before the landing', 'They came in with the lane walls (step 2), floating in the '
     'air. Moved to the end of step 8.'),
    ('Landing posts stand on the lane deck', 'Each post carries up to about 400 lb with a robot on the landing. Set '
     'every post base over a deck joist, or add 2x4 blocking under the plywood at each post.'),
    ('Lane posts are screwed to a single 2x4 joist', 'A 120 kg robot falling into a rail loads the post hard at '
     'the deck. Use 2 carriage bolts per post through the outside joist (in the BOM) and add a 2x4 block inside '
     'the joist at each post.'),
    ('Gantry beam trolley needs a steel flange', 'A beam trolley rides on a steel I-beam flange; the A-frame beam '
     'is wood. Either bolt a short steel beam under it, or skip the trolley and hang the safety line from a rated '
     'eye bolt through the beam. The nylon rope is not a rated fall-arrest line; for a 120 kg robot use a rated '
     'sling or chain hoist (2:1 margin or better) and lock all four casters.'),
    ('Deck panels need a level floor', 'The 2x4 joists bear straight on the floor. Shim low spots under the rims, '
     'otherwise the panel joints step and robots trip on them.'),
    ('Gates need latches', 'The spec shows hinged gates but no latch; a robot leaning on an unlatched gate walks '
     'out of the lane. Two post latches are in the BOM.'),
]


def step_parts():
    m = M.build()
    fs = F.build_fasteners(m)
    pcs = list(m.pieces) + F.to_pieces(fs)
    meta = [None] * len(m.pieces) + fs
    keys = ['%s / %s' % (p.group, p.name) for p in pcs]
    when = {}
    for si, st in enumerate(AS.STEPS):
        for ki, sub in enumerate(st['subs']):
            r = re.compile(sub[1], re.I)
            for i, k in enumerate(keys):
                if i not in when and r.search(k):
                    when[i] = (si, ki)
    out = defaultdict(lambda: dict(lumber=Counter(), sheet=Counter(), buy=Counter(), fast=Counter()))
    for i, (si, ki) in when.items():
        p = pcs[i]
        e = out[si]
        if meta[i] is not None:
            e['fast'][meta[i]['kind']] += 1
        elif p.kind == 'lumber':
            e['lumber'][(p.stock, eighth(p.cut['length']))] += 1
        elif p.kind == 'sheet':
            a, b = p.cut['dims']
            e['sheet'][(p.stock, eighth(max(a, b)), eighth(min(a, b)))] += 1
        elif p.kind == 'buy':
            e['buy'][p.buy] += 1
    return out


SHEET = {'ply': '3/4 plywood', 'osb': '7/16 OSB', 'hardboard': '1/4 hardboard'}
FAST = {'screw 1-5/8 in': '1-5/8 in deck screws', 'screw 3 in': '3 in deck screws',
        'structural screw 6 in': '6 in structural screws', 'carriage bolt 1/2 in': '1/2 in carriage bolts + nut + washer',
        'joist hanger': 'LUS28 joist hangers', 'stair stringer connector': 'LSCZ stringer connectors'}


def main():
    parts = step_parts()
    bom = json.load(open(os.path.join(ROOT, 'build', 'bom.json'), encoding='utf-8'))
    L = ['# Assembly guide', '',
         'How to build the loco-manipulation test lane, in 12 steps, from the first deck panel to the objects on the '
         'counter. Each step has an animation (rendered in MuJoCo from the same model as the drone shots and the '
         'robot runs), the parts it uses, and the order of work. Parts lists and screw counts come from the model, '
         'so they match the [bill of materials](BOM.md) (materials total **${:,.0f}**, about **${:,.0f}** with tax).'
         .format(bom['total'], bom['total_with_tax']), '',
         '**Whole build in 30 s:** [build_timelapse.mp4](videos/build_timelapse.mp4)', '',
         '## Before you start', '',
         '- **Tools:** circular saw or miter saw, drill/driver + impact driver (T25 bit, 1/2 in auger bit), 4 ft level, '
         'speed square, tape, clamps, caulk gun, hammer (hanger nails), socket set (3/4 in), jig saw (stringers).',
         '- **Safety:** glasses and hearing protection for cutting; two people for walls, the landing and the door.',
         '- **Lumber is nominal:** a 2x4 is 1-1/2 x 3-1/2 in, a 4x4 3-1/2 in square. All lengths below are actual cut '
         'lengths.',
         '- **Space:** about 40 x 12 ft of flat, level floor, plus room to roll the gantry.', '',
         '## Steps', '']
    for si, st in enumerate(AS.STEPS):
        L.append('- [%s. %s](#%s)' % (st['id'][:2], st['title'],
                                       re.sub(r'[^a-z0-9 -]', '', ('%s %s' % (st['id'][:2], st['title'])).lower()).replace(' ', '-')))
    L.append('')
    for si, st in enumerate(AS.STEPS):
        g = GUIDE[st['id']]
        e = parts[si]
        L += ['## %s %s' % (st['id'][:2], st['title']), '',
              '![%s](images/assembly/%s.gif)' % (st['title'], st['id']), '']
        need = []
        for (stock, ln), n in sorted(e['lumber'].items(), key=lambda kv: (kv[0][0], -kv[0][1])):
            need.append('%d × %s @ %s' % (n, stock, fmt_in(ln)))
        for (stock, a, b), n in sorted(e['sheet'].items()):
            need.append('%d × %s %s × %s' % (n, SHEET.get(stock, stock), fmt_in(a), fmt_in(b)))
        if need:
            L += ['**Cut:** ' + '; '.join(need), '']
        if e['buy']:
            L += ['**Parts:** ' + ', '.join('%s (%d)' % (k, n) if n > 1 else k for k, n in sorted(e['buy'].items())), '']
        if e['fast']:
            L += ['**Hardware:** ' + ', '.join('%d %s' % (n, FAST.get(k, k)) for k, n in sorted(e['fast'].items())), '']
        L += ['**Do:**', '']
        L += ['%d. %s' % (i + 1, s) for i, s in enumerate(g['steps'])]
        L += ['', '**Check:** ' + g['check'], '']
    L += ['## Things that would not work as modelled', '',
          'Found while turning the model into a build sequence. Each one is either fixed in the model or called out '
          'in the step.', '']
    for t, d in DISCREPANCIES:
        L += ['- **%s.** %s' % (t, d)]
    L += ['', '---', 'Regenerate: `python sim/assembly_anim.py` (animations), `python cad/tools/assembly_doc.py` '
          '(this page).', '']
    open(os.path.join(ROOT, 'docs', 'ASSEMBLY.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('wrote docs/ASSEMBLY.md')


if __name__ == '__main__':
    main()
