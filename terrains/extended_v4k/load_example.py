"""Load one tile of this terrain set with numpy only and print its shapes.   python load_example.py [type] [tile index]"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
kind = sys.argv[1] if len(sys.argv) > 1 else "wedges"
i = int(sys.argv[2]) if len(sys.argv) > 2 else 0

Z = np.load(HERE / (kind + ".npz"))
meta = json.load(open(HERE / (kind + ".json")))["tiles"][i]

height = Z["height"][i]                # (400, 400) float32, metres, [ix, iy]; floor raster
height_top = Z["height_top"][i]        # everything, tall things included
mask = Z["obstacle_mask"][i]           # bool
res = float(Z["resolution"])
x = Z["cell0_center"][0] + res * np.arange(height.shape[0])  # cell centre coordinates, the same for y

b0, b1 = Z["box_start"][i], Z["box_start"][i + 1]
center, half, quat = Z["box_center"][b0:b1], Z["box_half"][b0:b1], Z["box_quat"][b0:b1]  # quat wxyz
extra = ~Z["box_in_raster"][b0:b1]     # boxes to add on top of the raster for collision
labels = Z["labels"][Z["box_label"][b0:b1]]

m0, m1 = Z["mesh_start"][i], Z["mesh_start"][i + 1]
meshes = [(Z["mesh_vertices"][Z["mesh_vert_start"][m]:Z["mesh_vert_start"][m + 1]],
           Z["mesh_faces"][Z["mesh_face_start"][m]:Z["mesh_face_start"][m + 1]]) for m in range(m0, m1)]

print("tile", meta["tile_id"], "| type", meta["type"], "| level", meta["level"], "| difficulty %.3f" % meta["difficulty"], "| seed", meta["seed"],
      "| kind", meta["variant_kind"])
print("height", height.shape, height.dtype, "min %.3f max %.3f m" % (height.min(), height.max()), "| resolution", res, "m | size",
      height.shape[0] * res, "x", height.shape[1] * res, "m | x from %.2f to %.2f" % (x[0], x[-1]))
print("height_top", height_top.shape, "max %.3f m" % height_top.max(), "| obstacle_mask", mask.shape, "cells set:", int(mask.sum()))
print("boxes", center.shape, half.shape, quat.shape, "| not in raster:", int(extra.sum()), "| labels:", sorted(set(labels.tolist())))
print("meshes", len(meshes), "| not in raster:", int((~Z["mesh_in_raster"][m0:m1]).sum()),
      "| first:", (meshes[0][0].shape, meshes[0][1].shape) if meshes else None)
print("spawns", [(round(s["x"], 2), round(s["y"], 2), round(s["yaw"], 2)) for s in meta["spawns"]])
for r in meta["routes"]:
    print("route: start", [round(v, 2) for v in r["start"]], "->", len(r["waypoints"]), "waypoints -> target", [round(v, 2) for v in r["target"]],
          "| %.1f m" % r["length_m"])
print("params", meta["params"])
