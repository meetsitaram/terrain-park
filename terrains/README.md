# Extended training terrain

Procedural 8 x 8 m terrain tiles built from everyday props (4x4 timbers, wedges and ramps, blocks, stairs, walls, rails,
furniture), used to train the X2 perception policies (`x2-perception`). Simulator-neutral: numpy rasters, boxes and meshes,
plus checked routes per tile.

| Folder | What | Size |
| --- | --- | --- |
| `extended_v1/` | The base map (2026-11-11): 330 tiles, 9 types, 10 levels, 549 checked routes. Format and frame: `extended_v1/README.md` | 9.4 MB |
| `extended_v4k/` | The set the policies trained on (2026-11-12): extended_v1 + the add-ons below, 330 tiles, 2,382 routes. Same format | 8.4 MB |
| `generator/new_terrains/` | The tile generators (one function per type: difficulty in [0, 1] and a seed -> ground and obstacle meshes, routes), `check_routes.py`, preview renderers | |
| `generator/export/` | `export_extended_terrains.py` (generators -> `extended_v1`), `verify_export.py`, `make_preview.py`, `street_routed.py` | |
| `generator/addons/` | `make_terrain_v4j.py`, `make_terrain_v4k.py` (extended_v1 + add-ons -> `extended_v4k`) | |

Tile types: `single_beams`, `beam_fields`, `blocks`, `wedges`, `tight_spots`, `log_corridors`, `homes`, `street_front`, `stairs`.
The props and their parameter ranges: `../props/README.md`.

## extended_v4k = extended_v1 + add-ons

| Add-on | What it changes |
| --- | --- |
| `extended_v1_2_stairs_inset` | The 60 straight, L and U stairs tiles regenerated with the stair foot 2.5-3.0 m inside the tile (same ids, seeds and stair parameters), 700 routes in 11 kinds (side approach, oblique, descend, L, C, S, ...). Amphitheatre tiles stay those of extended_v1 |
| `extended_v1_1_prop_routes` | 1,116 extra routes on the single-beam, beam-field, log-corridor, wedge and block tiles (straight, oblique, side approach with a 70-90 deg turn near the prop, L, C, S, along a beam, across a wedge) |
| spawn fix (v4j) | Route starts and spawns closer than 0.40 m (planar) to anything rising 0.25-1.5 m above the start floor are moved or dropped (`extended_v4k/v4j_report.json`) |

Routes per type in extended_v4k: single_beams 268, beam_fields 268, blocks 240, wedges 484, tight_spots 36, log_corridors 124,
homes 101, street_front 120, stairs 741. An earlier add-on, `extended_v1_1_stair_routes` (526 routes on the original stair tiles),
was superseded by the inset stairs and is not in v4k.

## Load

```
pip install numpy
python terrains/extended_v1/load_example.py wedges 0
python terrains/extended_v4k/load_example.py stairs 3
```

Collision geometry = `height` raster + the boxes and meshes with `in_raster` false (or a flat floor and all boxes and meshes).
Details: `extended_v1/README.md`.

## Rebuild

```
pip install numpy scipy trimesh            # pyvista and pillow for the preview renders
cd terrains/generator/export
python -B export_extended_terrains.py --out OUT/extended_v1 --workers 6      # --gen-dir defaults to ../new_terrains
python -B verify_export.py --data OUT/extended_v1 --gen-dir ../new_terrains
python -B make_preview.py --data OUT/extended_v1
cd ../addons
python make_terrain_v4j.py OUT/extended_v1 STAIRS_INSET OUT/extended_v4j
EXTRA_ROUTES=PROP_ROUTES python make_terrain_v4k.py OUT/extended_v1 STAIRS_INSET OUT/extended_v4k
```

The export runs on Linux (it uses multiprocessing with `fork`). extended_v1 rebuilds from the
generators with the seeds in `index.json` / `<type>.json`. **The scripts that made the two add-on folders (`extended_v1_2_stairs_inset`,
`extended_v1_1_prop_routes`) are not in this repo:** they were one-off scripts and were not kept. Their output is merged into
`extended_v4k/` (stairs tiles in `stairs.npz` / `stairs.json`, extra routes in each `<type>.json`), so v4k is complete as data; only
the step from extended_v1 to v4k cannot be re-run from scratch.
