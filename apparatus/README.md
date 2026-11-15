# Apparatus

`ours/` is our own model of the 2026 IEEE Humanoids Loco-Manipulation Challenge apparatus: a parametric CAD drawn to the
challenge spec's pictures, its MuJoCo model, the height raster and wall boxes the X2 policies are scored on, and the eval route.
It is a copy of the design in [meetsitaram/loco-manip-apparatus](https://github.com/meetsitaram/loco-manip-apparatus) (which also has
the X2 simulations, the perception pipeline, renders and videos). The official NIST testbed is not here: `NIST.md` says where to get
it.

| Path | What |
| --- | --- |
| `ours/cad/params.py` | Every dimension, in inches (lane, deck, rails, door wall, stepover 4x4s, crossing ramps, wall obstacle, stairs, landing, task boards, counter, gantry) |
| `ours/cad/model.py` | Builds the apparatus from `params.py`: 675 pieces (lumber, sheets, bought parts, 3D-printed bulbs and cylinders), 54 joints. `python ours/cad/model.py` prints the counts |
| `ours/cad/assembly_steps.py`, `fasteners.py` | The build in 12 steps (shared by the build guide) and the fastener schedule |
| `ours/cad/tools/` | Cut list and shop sheets (`shop.py`), BOM (`bom.py`), beam and deck load checks (`beam.py`), clash and support checks, procedural textures |
| `ours/cad/fusion/` | Fusion 360 scripts that draw the model (`apparatus.py`, `fxlib.py`, `view.py`) and its textures. The saved `.f3d` is not copied (see below) |
| `ours/sim/apparatus.xml` | MuJoCo model of the apparatus (858 geoms, door, gates and task parts jointed), `meshes/` its 32 STL pieces (ramp bevels, stair notches) |
| `ours/sim/export_mjcf.py` | Rebuilds `apparatus.xml` from the CAD (`--spawn X Y --heading DEG --deck-at-zero --out file.xml`) |
| `ours/eval/` | The eval course: `cad_20mm_walk.npz` (physics floor raster, 734 x 269 cells of 2 cm), `cad_20mm.npz` (with tall cells), `cad_walls.json` (344 wall boxes), `route.json` (22 waypoints, 19.2 m, buttons 1-5). Format: `ours/eval/README.md` |
| `ours/docs/ASSEMBLY.md`, `BOM.md` | Build guide and bill of materials (images and videos are linked from the loco-manip-apparatus repo) |

## Load

```
pip install mujoco numpy
python -m mujoco.viewer --mjcf=apparatus/ours/sim/apparatus.xml
python -c "import numpy as np, json; z = np.load('apparatus/ours/eval/cad_20mm_walk.npz'); print(z['grid'].shape, z['res']); print(len(json.load(open('apparatus/ours/eval/route.json'))['waypoints']))"
```

Frames: the CAD and `apparatus.xml` are in the model frame (metres in the MJCF; X along the lane from the start end, Y left,
Z up, venue floor 0). The eval files are in the same x, y with the lane deck top at z = 0 (`ours/eval/README.md`).

The raster and the wall boxes were made by `sim/pipeline/hmap.py` and `sim/pipeline/walls.py` in loco-manip-apparatus (rays
straight down on the X2 mobility scene); those need the X2 robot model and are not copied here.

## Not included

- The challenge spec (`docs/spec` in loco-manip-apparatus) and the rules: the organisers' documents (see `NIST.md`).
- 3D-print files: the printed parts (bulbs, green cylinders) are drawn parametrically in `model.py`; there are no separate STL or
  slicer files (`prints/` in loco-manip-apparatus is empty).
- Build animations (39 MB of GIFs) and videos: linked, not copied.
- The saved Fusion model `cad/fusion/models/loco-manip-apparatus.f3d` (5.2 MB, in loco-manip-apparatus): Fusion files can carry
  account metadata that could not be checked here; running `cad/fusion/apparatus.py` in Fusion 360 redraws it from the parameters.
