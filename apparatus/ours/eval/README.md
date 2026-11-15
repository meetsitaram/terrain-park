# Our apparatus (IEEE Humanoids 2026 loco-manipulation apparatus, from our CAD)

Held out of training as a whole (docs/EVAL.md). Collected 2026-11-11 by the eval monitor.

| File | What |
|---|---|
| `cad_20mm_walk.npz` | The physics floor. `grid[ix, iy]` float32 heights in metres, 734 x 269 cells of 2 cm; `x0`, `y0` are the centre of cell (0, 0) (-1.9444, -1.4583 m); `res` 0.02. Tall obstacle cells (class obstacle and over 0.5 m) are set to the floor, 0. Range -0.108 (ground around the deck) to 1.118 m (landing). |
| `cad_20mm.npz` | The same raster with the tall cells kept (walls, posts, rails: up to 2.44 m). For vision and drawing, not for physics. |
| `cad_walls.json` | 344 boxes (`center`, `half`, `quat` wxyz, metres): every static CAD geom that reaches above 0.5 m, stands on a tall cell and is within 1 m of the route. Walls, door jambs, posts, rails, braces. `geom` is the id in the source scene. |
| `route.json` | The eval route: `spawn` (x, y in m, yaw in deg), 22 `waypoints` (name, x, y, heading_deg, hold_s, section), `sections`, `beams` (centre lines of the two 4x4s, which a trial must cross), `door` (hinge, open leaf, door plane). |

- **Source:** copies of `ssr-x2/assets/route/` (byte-identical to `loco-manip-apparatus/build/hmap/`), made by
  `loco-manip-apparatus/sim/pipeline/hmap.py` (raster: rays straight down on `sim/scene_x2_env_mobility.xml`) and `pipeline/walls.py` (boxes).
  The CAD is `loco-manip-apparatus/cad/`.
- **Units and frame:** metres. x along the lane from the start end, y across, z up; the lane deck top is z = 0. The deck is about
  11 m by 2.35 m (x 0..11, y 0..2.35). The route source is in inches in the same frame (1 in = 0.0254 m).
- **What is in it:** start area, door wall with a 0.97 m door, two 4x4 stepovers (8.9 cm tall; front one at x 3.02..3.10 on y 0.69,
  back one at x 3.04..3.10 on y 1.6), four crossing ramp cells (15 deg, crest 0.16 m, x 3.7..6.1), the small wall, the counter,
  six risers of 0.186 m (the last 0.17) to the landing at 1.118 m, side rails and posts.
- **Known property of the walk raster:** obstacle cells up to 0.5 m stay in it (351 cells on the deck, e.g. the feet of diagonal
  braces and the bulb box), so a brace is both a spike in the floor and a box. It is the same file ACP and SSR trained and
  were scored with on 2026-11-02, so it is kept as is.
- **Door:** hinge at (2.374, 1.651), leaf 0.92 m, standing open at 90 degrees along -x (toward the start); the source scene has it
  baked in that pose and it is among the wall boxes. The doorway is clear from y 0.73 to 1.60 (0.87 m) at body height; the
  route passes at y 1.18. The slanted piece outside the start (x -1.2..0) is the start gate, open, off the route.
- **4x4s:** front beam from (2.985, 0.257) to (3.098, 1.041), back beam from (3.097, 1.416) to (2.984, 2.183) (centre lines fitted
  to the raster). There is a 0.37 m gap between their ends, so a robot could go between them; the eval fails a trial whose
  pelvis path does not cross both centre lines within the beams' lengths.
- **Route:** `loco-manip-apparatus/sim/x2_route.py` `ROUTES['mobility']` (copied in `ssr-x2/ssr/route.py`): door -> front 4x4 ->
  button 1 -> back over it -> back 4x4 -> button 2 -> ramps to button 3 -> diagonal to button 4 -> ramps -> round the wall ->
  stairs -> landing, turn left -> button 5 at the landing's far end. 19.2 m. Stations (buttons) hold 1 s.
  **Five waypoints differ from the source**, after `eval/check_routes.py` (robot half width 0.25 m):
  waypoint 1 "through the door" x 104 -> 107 in; waypoints 2 and 6 (the slot between the door wall and the front 4x4, 0.51 m
  wide) x 107 / 108 -> 109 in, the middle of the slot (0.20 / 0.23 m from the wall before, 0.25 m now); waypoints 7 and 8
  (either side of the back 4x4) from (108, 60) and (129, 64) in to (2.80, 1.60) and (3.38, 1.78) m, so the leg between them
  crosses the back beam 0.27 m from its end (0.14 m before) while staying 0.25 m from the sloping board west of the beam
  (user, 2026-11-11: it has to step over both 4x4s). Check result: no segment blocked, tightest clearance 0.25 m; the front beam
  is crossed twice (0.36 m from its end, 82 degrees), the back beam once (0.27 m, 81 degrees); no arrival disc reaches a beam.
