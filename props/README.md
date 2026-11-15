# Props

The props behind every terrain in this repo, with the numbers needed to rebuild or simulate them. Three sources:

1. **Pico capture props**, fitted to where the feet went (`data/2026-10-29/props.json`, `data/2026-10-30/props/`): see the main
   README, "Formats".
2. **Extended map props** (`terrains/`), metres, drawn per tile from the ranges below; level `d` = 0..1 moves each range from
   its first to its second value. Code: `terrains/generator/new_terrains/gen_new_terrains.py` (`beam`, `wedge`, `block`, `wall`,
   `bar`, `brace_stand`, `rails_around`, `side`), `stairs.py` (`flight`, `stepped_sides`), `homes.py`, `street.py`. Every placed
   prop is in the tile files as a box or mesh with its label (`box_label`, `mesh_label`).
3. **Our apparatus props** (`apparatus/ours/`), inches, from `apparatus/ours/cad/params.py`; geometry `cad/model.py`, STL pieces in
   `sim/meshes/`, MuJoCo model `sim/apparatus.xml`.

## Lumber (actual sizes, `cad/params.py` `LUMBER`)

| Nominal | Actual (in) | Actual (m) |
| --- | --- | --- |
| 4x4 | 3.5 x 3.5 | 0.089 x 0.089 (`FOUR_BY_FOUR` in the generator) |
| 2x4 / 2x6 / 2x8 / 2x10 / 2x12 | 1.5 x 3.5 / 5.5 / 7.25 / 9.25 / 11.25 | 0.038 x 0.089 ... 0.286 |
| Sheets | 48 x 96; 3/4 ply, 23/32 OSB | 1.22 x 2.44; 19 / 18 mm |

## Extended map props (metres)

| Prop | Parameters |
| --- | --- |
| Beam / 4x4 (`single_beams`) | 6 per tile on a ring 1.2-3.4 m from the centre; square section 0.05 -> 0.15 (x 0.85-1.0); length 1.2-2.4; crossing angle 0 -> up to 15-60 deg off square; one end raised up to 0.20 on up to half of them |
| Beam field (`beam_fields`) | rows at spacing 1.2 -> 0.5; section 0.05 -> 0.12; length 2.5-5.0, yaw +-10 -> +-50 deg; 35% V pairs (2.6 m, 5-12 deg each side, like the apparatus stepover) |
| Blocks (`blocks`) | 10 -> 34 per tile, 0.7-3.6 m from the centre; height 0.05 to 0.10 -> 0.25; sides 0.08-0.45; any yaw |
| Wedge / ramp (`wedges`) | slope 8-12 -> 15-25 deg; pinwheel of 4 gables (cells 1.2-2.0, ridge directions like the apparatus) or a long ramp 1.5-3.0 up to a plateau and down, width 1.0-2.0; narrow slanted strips 0.35 wide; side rails on some |
| Brace stand | 4x4 post 1.1-1.5 tall, optional board 0.02 x 0.5-0.9 x 0.6-1.0, two slanted 4x4 braces reaching 0.45-0.65 out to 0.45-0.65 high (the apparatus button-post A-frame) |
| Rails | 4x4 posts every 1.0-1.6 (1.2 in `rails_around`), height 0.30 / 0.70 / 1.0; rails 0.04 x 0.06 |
| Tight spots | L lane 1.0 -> 0.6 wide, closed behind; a 0.05 -> 0.10 beam inside; wall gap 0.75 -> 0.50; walls 1.0 x 0.08 |
| Log corridor | 7.6 long, 1.5 -> 1.0 wide; logs 0.06-0.08 -> 0.10-0.15, spacing 1.4 -> 0.45, up to 50 deg; sides wall / rails / posts at 0.30, 0.70 or 2.44; every gap on the centre line >= 0.30 |
| Stairs | rise 0.08-0.12 -> 0.15-0.22, run 0.32-0.40 -> 0.22-0.30, width 1.4-2.0 -> 0.7-1.1, 3-6 -> 7-15 steps, nosing 0-0.03; straight (up, landing, down), L, U, amphitheatre; sides open / wall / rails / mixed at 0.30 / 0.70 / 2.44 |
| Homes, street front | walls 2.44 x 0.10 with door openings and leaves, chairs (seat 0.46), tables (0.75), thresholds, kerbs, porch steps, driveway slabs, cars (`homes.py`, `street.py`) |

The exact values of every tile are in its `<type>.json` (`params`, `seed`).

## Our apparatus props (inches; 1 in = 25.4 mm)

| Prop | Parameters (`cad/params.py`) |
| --- | --- |
| Lane | 9 sections of 48 = 432 long, 96 wide inside; deck 2x4 joists 12 in o.c. + 3/4 ply, top 4.25 above the floor |
| Rails | 4x4 posts, 2x4 rails at 12 / 24 / 36 above the deck |
| Door wall | at x 96; 36 x 80 prehung door, rough opening 38 x 82.5 at y 28; wall 96 tall, 2x4 studs 16 o.c., OSB skin; 3 spring hinges |
| Stepover | two 4x4s in a V from the door wall towards the button post: (117.5, 10) to (122, 41) and (117.5, 86) to (122, 55) |
| Crossing ramps | 2 x 2 cells of 48 x 48, each a hill of two 15 deg faces to a ridge (peak 6.43), pinwheel layout x, y, y, x; 2x8 frame, 5 ribs per cell, 3/4 ply skin |
| Wall obstacle | at x 290: 36 long, 36 tall, 3.5 thick |
| Stairs | 6 risers of 7-1/3 (44 total), run 8.75, width 42, nosing 1, 4 cut 2x12 stringers, handrail 36 |
| Landing | 44 above the deck, 2x8 joists 12 o.c., 4x4 posts, guard 36 |
| Task boards, keypad, counter, gantry | `KEYPAD`, `BOARD`, `TASK`, `COUNTER`, `GANTRY` in `params.py` |

Values marked "(est.)" in `params.py` were read off the challenge spec's pictures before the organisers published plans.
Build guide and bill of materials: `apparatus/ours/docs/ASSEMBLY.md`, `BOM.md`.
