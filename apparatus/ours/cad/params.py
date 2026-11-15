"""Single source of truth for the apparatus layout and lumber (pure Python: Fusion, cut list, BOM and sim all
import it).

Frame: X along the lane from the start end, Y to the left when facing +X (Y = 0 is the front rail line, the
side the spec's overview pictures are drawn from), Z up, Z = 0 is the venue floor. Values here are INCHES
(what the stores sell); multiply by IN for mm.

Source of the layout: the 2026 Humanoids Loco-Manipulation rules PDF (docs/spec), pictures on pages 2-4.
NIST's own plans were not released when this was written (checked 2026-10-06): every dimension marked
'(est.)' is read off the pictures or chosen by standard practice and should be swapped once NIST publishes.
"""
import math

IN = 25.4

# ---------------------------------------------------------------- nominal lumber -> actual size (in)
LUMBER = {
    '2x4': (1.5, 3.5), '2x6': (1.5, 5.5), '2x8': (1.5, 7.25), '2x10': (1.5, 9.25), '2x12': (1.5, 11.25),
    '4x4': (3.5, 3.5), '6x6': (5.5, 5.5), '1x4': (0.75, 3.5), '1x6': (0.75, 5.5),
}
# Real boards also vary: +-1/16 in on width/thickness, crowns and twist, and pressure-treated stock is wetter
# (shrinks ~1/8 in as it dries). Design rule: never rely on a nominal-sum fit; leave 1/8 in play in joints that
# stack boards (door rough opening, stair seats are cut to the line on each board), and size cut lengths from
# measured stock where it matters (see the build guide's 'measure your lumber' step).
SHEET = dict(w=48.0, l=96.0, ply=0.75, osb=0.71875, hard=0.25)     # 4x8 sheets; 23/32 OSB; 3/4 ply
STOCK_LENGTHS = (8, 10, 12, 14, 16)                                # ft, lumber the stores stock

# ---------------------------------------------------------------- overall lane
LANE = dict(
    sections=9, section=48.0,           # 9 sections of 4 ft (spec p.2)
    width=96.0,                         # lane inside width 8 ft (est.: crossing ramps are 8 x 8 ft and fill it)
)
LANE['length'] = LANE['sections'] * LANE['section']      # 432 in = 36 ft

# Floor: one portable deck panel per 4 ft section, sitting on the venue floor (fully supported, so the
# plywood span between joists is what carries a robot foot): 2x4 joists on edge, 12 in o.c., 3/4 plywood.
FLOOR = dict(joist='2x4', oc=12.0, sheet='ply', glue=True)        # 12 in o.c.: robot foot point loads (beam.py)
FLOOR['h'] = LUMBER['2x4'][1] + SHEET['ply']             # 4.25 in: deck top above the venue floor
DECK_Z = FLOOR['h']

# Lane walls: 4x4 posts every 4 ft outside the deck edge, three 2x4 rails (spec pictures: 3 rails + cap)
RAIL = dict(post='4x4', rail='2x4', height=36.0, rails=(12.0, 24.0, 36.0),    # rail tops above the deck (spec fit)
            stain='BEHR PREMIUM ST-533 Cedar Naturaltone',           # semi-transparent stain on all railings (spec look)
            stain_groups=('02 Lane walls', '09 Stairs/handrails'),
            )                                                                # no cap (spec): top rail flush with the posts

# ---------------------------------------------------------------- the sections (x ranges along the lane, in)
# Read off the spec overview by fitting its camera (cad/tools/layoutfit.py, docs/NOTES.md): the door wall sits
# 2 sections from the start and the landing is the last section.
#   0-96     S1-2  start area (keypad on the start side of the door wall)
#   96       door wall (task 2); lightbulb boards on its far face (task 3)
#   96-144   S3    stepover obstacles (A), location buttons 1 (front) and 2 (back)
#   144-240  S4-5  crossing ramps 8 x 8 ft (B); button task board (5) at their start, valves (4) at their end
#   240-336  S6-7  flat ground 8 ft (C), small wall obstacle, side gate; demonstration counter on the back side
#   336-384  S8    stairs (D), front half
#   384-432  S9    landing, location button 5 on its back guard
X = dict(door_wall=96.0, stepover=(96.0, 144.0), ramps=(144.0, 240.0), flat=(240.0, 336.0),
         stairs=(336.0, 384.0), landing=(384.0, 432.0))

# ---------------------------------------------------------------- door wall (task 2) and keypad (task 1)
DOOR = dict(
    slab=(36.0, 80.0, 1.375),           # standard 36 x 80 prehung interior door ("3 ft x 7 ft" in the spec)
    rough=(38.0, 82.5),                 # rough opening for a 36 x 80 prehung unit
    y0=28.0,                            # rough opening y0 .. y0 + 38 (spec 'view from start': 31 in of wall left of it)
    hinge_side='back',                  # hinges on the back (+Y) jamb, lever on the front edge; swings towards the
                                        # start side as in the spec overview (the text says 'pushed open': the
                                        # return trip pushes it; flip with a mirrored prehung unit if NIST says so)
    wall_h=96.0,                        # stud wall height above the deck (8 ft studs, est.)
    stud='2x4', oc=16.0, skin='osb',
    lever_h=36.0,                       # lever handle height above the deck (code: 34-48)
    spring_hinges=3,                    # 3 x 4 in self-closing spring hinges (spec: torsion springs)
    brace='2x4',                        # diagonal A-frame braces each end (spec pictures)
)
KEYPAD = dict(
    board=(17.0, 26.0, 0.75),           # w, h, t plywood panel on a 4x4 post (spec 'view from start')
    center_h=53.0,                      # board centre above the deck (spec 'view from start')
    pos=(78.0, 15.0),                   # x, y of the post (start side of the door wall, right of the door)
    button_d=100.0 / IN,                # 100 mm LED dome push buttons (spec 'view from start': ~4.3 in bezels)
    pitch=4.7,                          # button grid pitch (spec picture)
    display=(9.0, 3.8),                 # 5-digit 7-segment display (w, h), near the top (spec picture)
)

# ---------------------------------------------------------------- task boards (3 lightbulb, 4 valve, 5 button)
BOARD = dict(
    plate=(24.0, 24.0, 0.75),           # 2 x 2 ft plywood square, 5 items in a quincunx (spec pictures)
    upper_center_h=48.0,                # "wall height" board centre above the deck (est.)
    lower_tilt_deg=20.0,                # "floor" board leans back from vertical (est.)
    lower_center_h=12.0,
    post='4x4',
    spacing=8.0,                        # item pitch from the centre to the corners (in)
    by_kind=dict(button=dict(plate=(15.0, 15.0, 0.75), spacing=4.75)),   # button board: about 4 posts wide (spec p.3)
    spin=dict(lightbulb=-10.0, valve=6.0, button=0.0),   # upper boards turned in-plane (spec p.3 pictures)
)
TASK = dict(
    # lightbulb boards hang on the far face of the door wall, back part (spec p.2 picture: '3' beside the door)
    lightbulb=dict(x=None, y=82.5, face=(1, 0, 0), mount='wall'),
    button=dict(x=142.0, y=48.0, face=(-1, 0, 0), mount='post'),   # start of the ramps, lane centre, faces the start
    valve=dict(x=251.5, y=80.0, face=(-1, 0, 0), mount='post', tilt=10.0),   # counter's left end (spec p.2)
)

# ---------------------------------------------------------------- mobility
STEPOVER = dict(lumber='4x4',          # two 4x4s in a V from the door wall towards the button post (spec p.2/3):
    pieces=(((117.5, 10.0), (122.0, 41.0)),  # door -> button 1 (front) crosses one, -> button 2 (back) the other
            ((117.5, 86.0), (122.0, 55.0))))
RAMPS = dict(
    slope_deg=15.0, cell=48.0,          # 2 x 2 cells of 4 x 4 ft; each cell a hill: two 15 deg faces to a ridge
    layout=('x', 'y', 'y', 'x'),        # ridge directions per cell (0,0),(1,0),(0,1),(1,1): pinwheel (est.)
    skin='ply', frame='2x8', ribs=5,      # 5 ribs per cell: 12 in o.c. under the 3/4 ply skins (beam.py)
)
RAMPS['peak'] = RAMPS['cell'] / 2 * math.tan(math.radians(RAMPS['slope_deg']))   # 6.43 in ridge height
WALL_OBSTACLE = dict(x=290.0, length=36.0, h=36.0, t=3.5)    # small wall from the front rail into C, beside the gate

STAIRS = dict(
    rise=44.0 / 6, risers=6,            # 6 x 7-1/3 in = 44 in to the landing (spec picture fit: ~44.5)
    run=8.75, width=42.0,               # the flight fits section 8 (spec): 5 treads x 8-3/4 in. Steeper than IRC
                                        # (rise <= 7-3/4, run >= 10): ~40 deg; switch to 10 in treads if NIST does
    y0=0.0,                             # front half of the lane; turn left (+Y) at the top to button 5
    nosing=1.0,
    stringer='2x12', stringers=4,       # 4 cut 2x12 stringers (beam.py: 3 overstress the throat)
    handrail_h=36.0,                    # above the tread nose line
)
STAIRS['total_rise'] = STAIRS['rise'] * STAIRS['risers']
LANDING = dict(
    h=STAIRS['total_rise'],             # deck top 42 in above the lane deck... (relative to DECK_Z)
    joist='2x8', oc=12.0, beam='2x8', post='4x4', sheet='ply',
    guard_h=36.0,                       # guard top above the landing (spec picture fit ~36 in; IRC asks 42)
)

# ---------------------------------------------------------------- demonstration counter (back wall of C)
COUNTER = dict(
    x0=255.5, depth=24.0, top=(96.0, 25.0, 1.5),       # 8 ft laminate countertop
    bases=(('left', 'B30', 30.0, 'doors'), ('middle', 'DB36', 36.0, 'drawers3'), ('right', 'B30', 30.0, 'doors')),
                                                       # stock base cabinets (spec p.4 close-up)
    base_h=34.5, toe=4.0,
    upper=('W3024', 30.0, 24.0, 12.0),                 # wall cabinet on a 4x4 stand (drill targets inside)
    upper_bottom_h=15.0,                               # gap between countertop and the upper cabinet (spec p.4)
)

# ---------------------------------------------------------------- gantry (fall arrest rope over the lane)
GANTRY = dict(                          # mobile A-frame gantry on casters (spec p.1 picture), parked at the start
    x=-30.0,                            # parked position (frame centre) just before the start end; rolls along X
    clear_h=114.0,                      # crossbeam underside above the floor (9.5 ft): robot + rope + hook (est.)
    base_len=72.0,                      # 4x4 base beam per side, outside the lane rails
    side_gap=9.0,                       # base beam centre this far outside the lane rail faces
    post='4x4', beam='2x10', base='4x4', brace='2x4',
    caster_h=5.0,                       # 4 in locking swivel plate casters: plate to floor
)

GATES = dict(
    start=dict(middle_y=48.0),          # start end: fixed rail panel over the back half, gate on the front half (spec)
    side=dict(x=(240.0, 288.0)),        # the front rail module of section 6 is a gate (spec p.2), staff access
)
POSE = {'door hinge': 0.0, 'start gate': 0.0, 'side gate': 0.0}     # renders override these (view.py)
# the spec pictures' state of the moving parts (door open towards the start, gates open, demo cabinets open)
SPEC_POSE = {'door hinge': 90.0, 'start gate': 55.0, 'side gate': 60.0, 'left B30 door 1': 90.0,
             'right B30 door 1': 90.0, 'middle DB36 drawer 3': 10.0}       # upper doors: modelled open 90 deg

# ---------------------------------------------------------------- robot envelope (spec p.5)
ROBOT = dict(mass_kg=120.0, door_w=36.0)
DESIGN_LOAD = dict(point_lbf=400.0, live_psf=60.0)   # deck/stairs/landing: 120 kg robot + a person helping


def mm(v):
    """inches (number or tuple) -> mm"""
    return tuple(c * IN for c in v) if isinstance(v, (tuple, list)) else v * IN
