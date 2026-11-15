# extended_v1: extended training terrain map (simulator-neutral export)

Made 2026-11-11 from the preview generators in `hiking-in-the-wild/review/new_terrains/` (the plan: `docs/TODO_extended_training.md`,
"To do 1"). CPU only; no simulator was used. 330 tiles of 8 x 8 m, 9 tile types, 10 difficulty levels, 549 checked routes. 9.4 MB.

`preview_top.png` and `preview_closeups.png` are drawings made from these files (**preview, not a simulator render**).

## Files

| File | What |
|---|---|
| `index.json` | Set-level facts: sizes, level -> difficulty, counts per type, the command used. |
| `<type>.npz` | All tiles of one type: rasters, boxes, meshes (numpy only, no pickle). |
| `<type>.json` | Per-tile metadata of that type: level, seed, generator parameters, spawn poses, routes, targets. |
| `routes_dropped.json` | The 131 generator routes left out, each with the reason and the place. |
| `load_example.py` | Loads one tile with numpy and prints its shapes. |
| `preview_top.png`, `preview_closeups.png` | Top-down drawings of every tile / of one tile per type at levels 0, 4, 9. |

Types: `single_beams`, `beam_fields`, `blocks`, `wedges`, `tight_spots`, `log_corridors`, `homes`, `street_front`, `stairs`.

## Frame and units

Metres and radians. Each tile has its own frame: origin at the tile centre, x and y from -4 to +4, z up, the flat ground top at z = 0.
Yaw is about +z, 0 = facing +x. Quaternions are `wxyz`. Tiles are independent; `index.json` `layout` only suggests a field
(x = level * 8 m, y = column * 8 m).

## `<type>.npz` keys (N tiles, B boxes, K meshes in the file)

Rasters, `[tile, ix, iy]`, 400 x 400 cells of 2 cm; cell (ix, iy) has its centre at x = -3.99 + 0.02 ix, y = -3.99 + 0.02 iy
(`cell0_center`, `resolution`, `size` are in the file):

- `height` float32 (N, 400, 400): the **floor raster**. Top surface of every primitive flagged `in_raster` (ground, stair treads and
  landings, wedges, ramps, slabs, and low things resting on them: beams, blocks, kerbs, thresholds, stones).
- `height_top` float32: top surface of everything, tall things included (walls to 2.44 m). For drawing and vision, not for physics.
- `obstacle_mask` bool: cells where something that is not in the floor raster stands more than 0.26 m above the floor and
  between floor + 0.05 m and floor + 1.6 m (door headers at 2.03 m are not in it), grown by one cell.

Per tile: `tile_id`, `tile_level` (0..9), `tile_difficulty` (level / 9), `tile_seed`, `tile_variant`.

Boxes (every box-shaped primitive of the generator, the ones in the raster too). Boxes of tile i are rows
`box_start[i] : box_start[i+1]`:

- `box_center` (B, 3), `box_half` (B, 3) half extents along the box's own axes, `box_quat` (B, 4) wxyz.
- `box_yaw` (B,) and `box_upright` (B,): for upright boxes (`box_upright` true) the yaw alone gives the pose.
- `box_in_raster` bool: true = already in `height`. **Collision geometry = `height` + the boxes and meshes with `in_raster` false**
  (adding the others too counts them twice). Or use a flat floor at z = 0 and all boxes and meshes, ignoring the raster.
- `box_role` uint8: 0 floor (ground, treads, landings, plateaus, slabs), 1 low (at most 0.26 m above what it stands on: steppable
  by the rule of `check_routes.py`), 2 tall. Role 1 or 2 = obstacle. Role 1 with `in_raster` false = slanted or lifted low
  pieces (a 4x4 with one end raised, a low rail on posts): a raster cannot show the gap under them.
- `box_label` int16: index into `labels` (e.g. `beam`, `wall`, `door_leaf`, `door_header`, `brace_stand:brace`, `rail_fence:post`,
  `corridor_side:rail`, `stair_side:handrail`, `stair_step`, `car`, `chair:block`).
- `box_gen_group` uint8: 0 = the generator's ground list, 1 = its obstacle list (the generators call everything but the ground slab an
  obstacle; `box_role` is the finer split).

Meshes (primitives that are not boxes: wedges, sloped slabs, amphitheatre steps, cylinders, stones, bushes). Meshes of tile i are
`mesh_start[i] : mesh_start[i+1]`; mesh m has vertices `mesh_vertices[mesh_vert_start[m] : mesh_vert_start[m+1]]` and faces
`mesh_faces[mesh_face_start[m] : mesh_face_start[m+1]]` (indices into its own vertices). `mesh_role`, `mesh_in_raster`, `mesh_label`,
`mesh_gen_group` as for boxes; `mesh_aabb_center`, `mesh_aabb_half` give an axis-aligned box around each.

## `<type>.json`, per tile

`tile_id`, `type`, `npz`, `index` (row in the npz), `level`, `difficulty`, `seed`, `variant`, `variant_kind` (stairs: straight / L / U /
amphitheatre; wedges: pinwheel / long_ramp), `size_m`, `resolution_m`, `shape`, `origin`, `cell0_center`,
`params` (the difficulty-dependent generator ranges at this level), `realised` (primitive counts; stairs: rise, run, width, steps,
nosing, sides; wedges: pinwheel cell size, crest and slope per cell, or the ramp plateau; corridors: style and height of each side),
`floor_height_range_m`, `top_height_max_m`, and:

- `spawns`: `x, y, yaw`, floor height `z`, `clearance_m` (distance to the nearest `obstacle_mask` cell, capped at 9.99),
  `floor_range_r20_m` (floor height range within 0.2 m: above about 0.05 the pose is on a slope or beside a step), `source`
  (`generator` = tile centre facing +x, `home_spawns` / `street_spawns` = the dead-end spawns of those modules, `route` = a route start).
  Kept only if not on an obstacle cell and at least 0.12 m clear; the others are in `spawns_rejected`. Filter on `clearance_m`
  for a wider robot.
- `routes`: `start` (x, y, yaw), `waypoints` (x, y) in order, `target` (= last waypoint), `length_m`, floor `z` and `clearance_m` at each
  point, `source_index` (index in the generator's `last_routes`), `fixed` when the export changed it. Every route here passed
  `check_routes.route_blocked`, and all its points are inside the tile and off obstacle cells.
- `targets`: the generator's own target list, as given (not cleaned: `inside_tile`, `on_obstacle_cell` flags). Use `routes`.

## How it was made

```
cd terrains/generator/export
PYTHONDONTWRITEBYTECODE=1 python -B export_extended_terrains.py --out OUT/extended_v1 --workers 6
python -B make_preview.py  --data OUT/extended_v1
python -B verify_export.py --data OUT/extended_v1 --gen-dir ../new_terrains
```

About 20 s on 6 cores. The script imports the generator modules unchanged and calls each with `difficulty = level / 9` and
`seed = 10000 * (type index + 1) + 100 * level + k` (k = 0, 1, ...; for stairs and wedges k is scanned upward until every stair kind /
wedge main feature appears equally often per level). Each primitive is rasterised at the cell centres; a primitive goes into the floor
raster if it rests on the floor stack (no gap under it) and is floor-like or at most 0.26 m high.

## Counts

| Type | Tiles | Per level | Routes defined | Dropped | Kept | Tiles without a route |
|---|---|---|---|---|---|---|
| single_beams | 30 | 3 seeds | 30 | 1 | 29 | 1 |
| beam_fields (V pairs, slanted 4x4s) | 30 | 3 seeds | 30 | 0 | 30 | 0 |
| blocks | 30 | 3 seeds | 30 | 0 | 30 | 0 |
| wedges (pinwheel, long ramp, narrow strips) | 40 | 2 pinwheel + 2 long ramp | 120 | 21 | 99 | 2 |
| tight_spots (lane, dead end, wall gap, pocket) | 20 | 2 seeds | 40 | 0 | 40 | 0 |
| log_corridors | 40 | 4 seeds | 80 | 0 | 80 | 0 |
| homes | 30 | 3 seeds | 120 | 19 | 101 | 0 |
| street_front | 30 | 3 seeds | 90 | 71 | 19 | 13 |
| stairs | 80 | 2 each: straight, L, U, amphitheatre | 140 | 19 | 121 | 0 |
| **Total** | **330** | 33 | **680** | **131** | **549** | 16 |

- All 131 drops are `check_routes.py` findings (119 "obstacle at body height", 12 drops of more than 0.26 m); see
  `routes_dropped.json`. The checker's own default sample (3 levels x 6 seeds) gives 327 routes, 61 blocked; this set uses other
  seeds and all 10 levels, hence the different numbers.
- 51 kept routes were changed: points more than 3.9 m from the tile centre were pulled back to 3.9 m (the pinwheel routes end
  at x or y = 4.1, outside the tile) and then checked. Nothing else was repaired.
- Street front keeps only 19 of 90 routes: the generator puts the hydrant, sign pole, bushes and the car across its fixed routes on
  most seeds. That needs a generator fix, not an export fix.
- 858 spawn poses; 32 rejected (25 of them the street "garage threshold" spawn, which is inside the parked car when there is one).
  One wedges tile has no spawn.
- Corridor sides in the 40 tiles (80 sides): wall 2.44 m 38, wall 0.70 m 7, wall 0.30 m 6, rails on posts 0.30 m 9 / 0.70 m 3,
  posts only 0.30 m 6 / 0.70 m 11.

## Not in the set

- **The paper's own terrains** (`ROUGH_TERRAINS_CFG`, the 20 columns of the current run). `review/make_terrain.py` builds them with
  Isaac Lab's `TerrainGenerator` inside Isaac Sim, which was not allowed here. Its earlier output still exists
  (`review/terrain/terrain.ply`, 176 MB, with `layout.json` / `layout.npz`) and could be rasterised into this format without Isaac.
- Stairs: straight, L, U and amphitheatre are all in `stairs.py` and all exported. The amphitheatre's aisle rails are an empty stub
  in the generator, so amphitheatres have open sides only.
- Not implemented in the generators, so absent: roll wedges (slope across the path), a randomised tight-spot layout (the two seeds
  per level differ only in the lane beam's angle), beam spacing rules for hard beam fields (beams pile up at high levels).
- No textures, materials or friction values. No home photos, address or house number.

## Held-out apparatuses

The two challenge apparatuses (`apparatus/ours` in this repo; NIST: see `apparatus/NIST.md`) are not in this set as layouts or routes.

- The four generator modules read no file at all: no CAD, no apparatus raster, wall list or route (checked by reading them and by
  searching for file access). Every dimension is a constant or a random draw.
- No tile combines the apparatus sequence (door, 4x4 stepovers, ramp cells, wall, counter, stairs, landing): beams are only in the
  beam, tight-spot and corridor tiles, pinwheels only in wedge tiles, stair flights only in stair tiles, doors and the counter
  only in homes. The apparatus lane is about 11 x 2.35 m; a tile is 8 x 8 m.
- No route here follows an apparatus route; routes come from the generators' own formulas.

Apparatus-like single parts that do appear (allowed as re-arranged parts):

| Part | Here | Apparatus, for comparison |
|---|---|---|
| 4x4 stepover | beams 5 to 15 cm square, 1.2 to 5 m long, flat or one end raised up to 0.20 m | 8.9 cm (ours), 12.6 cm above the deck (NIST) |
| V of two beams | beam fields: two 2.6 m beams at y = +-1.6, each 5 to 12 deg off square | ours: two 0.8 m beams about 8 deg off square; NIST: about 1.35 m each, about 45 deg |
| Crossing-ramp / pinwheel cells | wedges: 2 x 2 gable cells of 1.27 to 1.94 m, 8.8 to 25 deg, crest 0.10 to 0.39 m (20 tiles, 9 with rails) | ours: four cells of about 1.2 m, 15 deg, crest 0.16 m |
| Rails on 4x4 posts | 8.9 cm posts every 1.0 to 1.6 m, 4 x 6 cm rails, 0.30 / 0.70 / 1.0 m (wedges, corridors); stair handrails 0.9 m | side rails and posts |
| Brace stand | 8.9 cm post 1.1 to 1.5 m, two slanted 4x4 braces reaching 0.45 to 0.65 m out and up, board on 60 % (45 tiles) | button-post A-frame |
| Stair flight | risers 8.4 to 21.6 cm, treads 24 to 40 cm, 3 to 15 steps | ours: six risers of 18.6 cm; NIST: five of 20 to 21 cm |
| Door, counter | homes: doorway 0.76 to 0.95 m with an open leaf, a 0.9 m counter 0.6 x 3.0 m | 0.97 m door, counter |

The pinwheel is the closest part: it has four cells like our apparatus's ramp section. No exported pinwheel has both a cell
under 1.35 m and all four slopes between 13 and 17 deg, and no straight flight has 5 or 6 risers of 17 to 21.5 cm.

## Checks run on the files (`verify_export.py`)

- All 330 rasters: 400 x 400 = 8 m / 2 cm, float32, finite, `height_top >= height`; offsets and quaternions consistent.
- All 858 spawn poses and 2,135 route points: inside the tile, off obstacle cells, and not inside any tall box at 0.3 / 0.7 / 1.2 m above
  the floor (0 failures). Smallest clearance: spawns 0.126 m, route points 0.02 m (5 routes pass closer than 0.2 m to an obstacle
  cell at a waypoint; they passed the route checker, which tests the centre line and 0.2 m to each side).
- 4 random tiles per type: `height_top` against rays cast down on the generator mesh: at most 0.55 % of cells differ by more than
  1 cm (cells on the edge of a wall); every box rebuilt from centre / half / quat matches the generator's vertices to 2e-7 m.
- Not checked: nothing was loaded in a simulator; spawn poses were not tested against a robot's footprint or height.
