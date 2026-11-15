# The official NIST testbed (not in this repo)

The official 2026 IEEE Humanoids Loco-Manipulation Challenge testbed is the organisers' CAD. It has no licence file, so it is
not redistributed here, and nothing derived from it (meshes, MJCF, height rasters, wall boxes) is either. Get it from the
organisers:

- **Source:** [github.com/peterso/ieee-humanoids-2026-loco-manipulation-challenge](https://github.com/peterso/ieee-humanoids-2026-loco-manipulation-challenge),
  `CAD-Files/LocoManipulationTestbed.dae` (also `.fbx`, `.usdz`); the rules PDF is in the same repo. Our import was built and
  checked against commit `5312b6d` (2026-10-23).
- From the importer: "the official CAD ... no license file: read from a local clone, never copied" (`sim/nist/import_nist.py`
  in loco-manip-apparatus).

## How our eval imports it

All steps read a local clone and write to git-ignored build folders:

1. Clone the challenge repo next to [loco-manip-apparatus](https://github.com/meetsitaram/loco-manip-apparatus) and check out `5312b6d`.
2. `cd loco-manip-apparatus/sim; python -m nist.import_nist` - DAE (inches) to parts in metres, visual meshes and convex /
   voxel collision pieces in `build/nist/parts/`.
3. `python -m nist.scene_nist --no-robot` - the MuJoCo scene with the door, keypad, bulbs, crusher, drawers jointed:
   `build/nist/scene_x2_nist_norobot.xml`.
4. The eval course is derived from that scene the same way ours was (`eval/collect_nist.py` in x2-perception): a top-surface
   raster of 2 cm, tall obstacle cells set to the floor (`nist_20mm_walk.npz`), the tall geoms as 1,920 wall boxes (door open at
   -1.65 rad), and `route.json` from `NIST_ROUTE` in `sim/pipeline/routepass.py`: 35 waypoints, 22.5 m, 11 stations.

Renders of our own runs in the testbed are ours and are published in loco-manip-apparatus (e.g. `docs/videos/p5_nist_route_walk.mp4`).
