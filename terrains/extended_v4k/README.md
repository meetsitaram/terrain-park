# extended_v4k

The terrain set the X2 perception policies trained on (2026-11-12). Same files and format as `../extended_v1/` (read its
README for keys, frame and units); `load_example.py` loads one tile.

- Tiles: the 330 tiles of extended_v1, with the 60 straight, L and U stairs tiles replaced by the inset versions (stair foot
  2.5-3.0 m inside the tile; same ids, seeds and stair parameters; some straight flights lose 1-6 steps).
- Routes: 2,382 (extended_v1's own, the 1,116 prop routes and the 700 inset-stairs routes), after the spawn fix: starts and spawns
  closer than 0.40 m to anything rising 0.25-1.5 m above the start floor moved or dropped. Per type: `v4j_report.json`.
- Made by `../generator/addons/make_terrain_v4j.py` and `make_terrain_v4k.py` (see `../README.md`, "Rebuild").
- `README_v4j.json`, `v4j_report.json`: the build note and per-type counts of the spawn fix.
