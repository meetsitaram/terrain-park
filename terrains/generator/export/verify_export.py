#!/usr/bin/env python
"""Checks on the exported set, read back from disk. CPU only.

  python -B verify_export.py --data OUT/extended_v1 [--gen-dir .../review/new_terrains]
1. every raster: shape = size / resolution, float32, finite; box quaternions unit; offsets consistent.
2. every spawn pose and every route point: inside the tile, not on an obstacle_mask cell, and (independent of the mask)
   not inside any tall box that is not in the raster, tested at floor + 0.3 / 0.7 / 1.2 m.
3. with --gen-dir: for some tiles per type, regenerate the tile, cast rays straight down on the mesh of the generator (embree, as
   check_routes.py does) and compare with height_top; and rebuild each box from centre / half / quat and compare its corners
   with the box vertices of the generator."""
import argparse
import json
import pathlib
import sys

sys.dont_write_bytecode = True
import numpy as np


def quat_R(q):
    w, x, y, z = [float(a) for a in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gen-dir", default=None)
    ap.add_argument("--per-type", type=int, default=4)
    args = ap.parse_args()
    D = pathlib.Path(args.data)
    index = json.load(open(D / "index.json"))
    res, n = index["resolution_m"], index["raster_shape"][0]
    assert abs(index["tile_size_m"][0] / res - n) < 1e-9
    bad = 0
    tot = dict(tiles=0, spawns=0, route_points=0, routes=0, targets=0, boxes=0, meshes=0)
    street_min = 9.0
    min_clear_route, min_clear_spawn = 9.0, 9.0
    for key in index["types"]:
        Z = np.load(D / (key + ".npz"))
        Z = {k: Z[k] for k in Z.files}
        metas = json.load(open(D / (key + ".json")))["tiles"]
        H, HT, MK = Z["height"], Z["height_top"], Z["obstacle_mask"]
        assert H.shape == (len(metas), n, n) and H.dtype == np.float32, (key, H.shape, H.dtype)
        assert HT.shape == H.shape and MK.shape == H.shape and MK.dtype == bool
        assert np.isfinite(H).all() and np.isfinite(HT).all() and (HT >= H - 1e-6).all() and H.min() >= -1e-6
        assert abs(float(Z["size"][0]) / float(Z["resolution"]) - n) < 1e-4
        assert Z["box_start"][-1] == len(Z["box_center"]) and Z["mesh_start"][-1] == len(Z["mesh_role"])
        assert Z["mesh_vert_start"][-1] == len(Z["mesh_vertices"]) and Z["mesh_face_start"][-1] == len(Z["mesh_faces"])
        assert np.abs(np.linalg.norm(Z["box_quat"], axis=1) - 1).max() < 1e-5 and (Z["box_half"] > 0).all()
        assert np.isfinite(Z["box_center"]).all() and np.isfinite(Z["mesh_vertices"]).all()
        tot["boxes"] += len(Z["box_center"])
        tot["meshes"] += len(Z["mesh_role"])
        for m in metas:
            i = m["index"]
            assert str(Z["tile_id"][i]) == m["tile_id"] and int(Z["tile_level"][i]) == m["level"] and int(Z["tile_seed"][i]) == m["seed"]
            tot["tiles"] += 1
            tall = [b for b in range(Z["box_start"][i], Z["box_start"][i + 1]) if not Z["box_in_raster"][b] and Z["box_role"][b] == 2]
            Rs = [quat_R(Z["box_quat"][b]) for b in tall]

            def check(x, y, what):
                nonlocal bad, street_min
                if not (abs(x) <= 4.0 and abs(y) <= 4.0):
                    bad += 1
                    print("OUTSIDE", m["tile_id"], what, x, y)
                    return
                ci, cj = min(n - 1, int((x + 4.0) / res)), min(n - 1, int((y + 4.0) / res))
                if MK[i, ci, cj]:
                    bad += 1
                    print("ON OBSTACLE CELL", m["tile_id"], what, x, y)
                z0 = float(H[i, ci, cj])
                for b, R in zip(tall, Rs):
                    for dz in (0.3, 0.7, 1.2):
                        loc = (np.array([x, y, z0 + dz]) - Z["box_center"][b]) @ R
                        if (np.abs(loc) <= Z["box_half"][b]).all():
                            bad += 1
                            print("INSIDE BOX", m["tile_id"], what, x, y, str(Z["labels"][Z["box_label"][b]]))
                            return

            for s in m["spawns"]:
                check(s["x"], s["y"], "spawn")
                tot["spawns"] += 1
                min_clear_spawn = min(min_clear_spawn, s["clearance_m"])
            for g in m["targets"]:
                check(g["x"], g["y"], "target")
                tot["targets"] += 1
            if key == "street_front":
                assert len(m["routes"]) >= 2, (m["tile_id"], len(m["routes"]))
                for r in m["routes"]:
                    assert r["min_clearance_along_route_m"] >= 0.35, (m["tile_id"], r["min_clearance_along_route_m"])
                    assert any(abs(s["x"] - r["start"][0]) < 1e-9 and abs(s["y"] - r["start"][1]) < 1e-9 for s in m["spawns"]), m["tile_id"]
                    street_min = min(street_min, r["min_clearance_along_route_m"])
            for r in m["routes"]:
                tot["routes"] += 1
                assert r["target"] == r["waypoints"][-1]
                for p in [r["start"][:2]] + r["waypoints"]:
                    check(p[0], p[1], "route point")
                    tot["route_points"] += 1
                min_clear_route = min(min_clear_route, min(r["clearance_m"]))
    print("read back:", tot)
    print("points outside / on obstacle cells / inside tall boxes:", bad)
    print("street_front: every tile has >= 2 routes, each with a spawn at its start; least clearance along a route %.3f m" % street_min)
    print("smallest clearance to an obstacle cell: spawns %.3f m, route points %.3f m" % (min_clear_spawn, min_clear_route))

    if args.gen_dir:
        import export_extended_terrains as E
        import trimesh
        E.load_generators(args.gen_dir)
        xs = E.X0 + E.RES * np.arange(n)
        xy = np.array([(x, y) for x in xs for y in xs])
        rng = np.random.default_rng(0)
        for key in index["types"]:
            Z = np.load(D / (key + ".npz"))
            Z = {k: Z[k] for k in ("height_top", "box_start", "box_quat", "box_center", "box_half")}
            metas = json.load(open(D / (key + ".json")))["tiles"]
            worst_frac, worst_box, worst_p99 = 0.0, 0.0, 0.0
            for m in [metas[k] for k in rng.choice(len(metas), min(args.per_type, len(metas)), replace=False)]:
                i = m["index"]
                g, o, _, _, _, _ = E.generate(key, m["difficulty"], m["seed"])
                mesh = trimesh.util.concatenate(g + o)
                z = E.CR.heights(mesh, xy).reshape(n, n)
                diff = np.abs(z - Z["height_top"][i])
                worst_frac = max(worst_frac, float((diff > 0.01).mean()))
                worst_p99 = max(worst_p99, float(np.percentile(diff, 99)))
                bi = Z["box_start"][i]
                for msh in g + o:
                    if E.as_box(msh) is None:
                        continue
                    R = quat_R(Z["box_quat"][bi])
                    s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
                    cn = Z["box_center"][bi] + (s * Z["box_half"][bi]) @ R.T
                    d = np.linalg.norm(cn[:, None] - np.asarray(msh.vertices)[None], axis=2)
                    worst_box = max(worst_box, float(max(d.min(1).max(), d.min(0).max())))
                    bi += 1
                assert bi == Z["box_start"][i + 1]
            print("%-14s height_top vs rays on the generator mesh: cells differing > 1 cm: %.3f %% (worst tile), 99th pct |diff| %.4f m; "
                  "box corners vs generator vertices: max %.2e m" % (key, 100 * worst_frac, worst_p99, worst_box))
    print("VERIFY", "OK" if bad == 0 else "FAILED")


if __name__ == "__main__":
    main()
