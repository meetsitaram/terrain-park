"""v4j terrain set (CPU): extended_v1 with (1) the stair tiles v0-v5 replaced by extended_v1_2_stairs_inset (same tile ids,
index kept at the extended_v1 row so the generator's draws are unchanged), amphitheatre tiles v6-v7 from extended_v1; (2) route
starts (and spawns) closer than MIN_CLEAR m (planar) to a box or mesh that rises into the body's height band (0.25-1.5 m above
the start floor) dropped: spawn-into-props fix (v4i: 23.5% of homes robots had a non-foot contact over 50 N in their first 0.2 s).
usage: make_terrain_v4j.py <extended_v1> <stairs_inset> <out>"""
import json, math, os, sys
import numpy as np

V1, INS, OUT = sys.argv[1:4]
MIN_CLEAR = float(os.environ.get("MIN_CLEAR", "0.40"))
TIGHT_MIN = float(os.environ.get("TIGHT_MIN", "0.30"))
BAND = (0.25, 1.5)
TYPES = ["single_beams", "beam_fields", "blocks", "wedges", "tight_spots", "log_corridors", "homes", "street_front", "stairs"]
os.makedirs(OUT, exist_ok=True)


def qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def solids(z, i):
    """Per tile: list of (corners (8, 3)) for boxes and mesh AABBs."""
    out = []
    for b in range(int(z["box_start"][i]), int(z["box_start"][i + 1])):
        R = qrot(z["box_quat"][b]); h = z["box_half"][b]
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
        out.append(c @ R.T + z["box_center"][b])
    # meshes (ramps, wedges, logs) are left out: their bounding boxes cover walkable ramp surfaces (28 false hits on wedges)
    return out


def clearance(p, zf, sol):
    """Planar distance from p (x, y) to the xy hull (bounding rectangle along the solid's own axes, approximated by the
    convex hull of its corners) of every solid overlapping the height band above floor zf."""
    best = 1e3
    for c in sol:
        lo, hi = c[:, 2].min(), c[:, 2].max()
        if hi < zf + BAND[0] or lo > zf + BAND[1]:
            continue
        pts = c[:, :2]
        # distance to the convex hull of the projected corners
        from scipy.spatial import ConvexHull
        try:
            hull = pts[ConvexHull(pts).vertices]
        except Exception:
            hull = pts
        inside, d = True, 1e3
        for a, b in zip(hull, np.roll(hull, -1, 0)):
            e = b - a; L = max(np.dot(e, e), 1e-12)
            t = np.clip(np.dot(p - a, e) / L, 0, 1)
            d = min(d, np.linalg.norm(p - (a + t * e)))
            if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < 0:
                inside = False
        best = min(best, 0.0 if inside else d)
    return best


report = {}
for tt in TYPES:
    z1 = np.load(f"{V1}/{tt}.npz", allow_pickle=True)
    j1 = json.load(open(f"{V1}/{tt}.json"))
    tiles = j1["tiles"]
    zsrc = {i: (z1, i) for i in range(len(tiles))}
    if tt == "stairs":
        zi = np.load(f"{INS}/stairs.npz", allow_pickle=True)
        ji = json.load(open(f"{INS}/stairs.json"))["tiles"]
        row_of = {t["tile_id"]: t["index"] for t in tiles}
        for t in ji:
            r = row_of[t["tile_id"]]
            assert tiles[r]["level"] == t["level"], t["tile_id"]
            zsrc[r] = (zi, t["index"])
            t = dict(t); t["index"] = r; t["source_set"] = "extended_v1_2_stairs_inset"; t["npz"] = "stairs.npz"
            tiles[r] = t
        # merged npz: per-tile arrays and the box / mesh blocks in extended_v1 row order
        labels = list(z1["labels"])
        for l in zi["labels"]:
            if l not in labels:
                labels.append(l)
        out = {k: [] for k in ("height", "height_top", "obstacle_mask", "tile_level", "tile_difficulty", "tile_seed", "tile_variant", "tile_id")}
        boxk = [k for k in z1.files if k.startswith("box_") and k != "box_start"]
        meshk = ["mesh_role", "mesh_in_raster", "mesh_label", "mesh_gen_group", "mesh_aabb_center", "mesh_aabb_half"]
        B = {k: [] for k in boxk}; M = {k: [] for k in meshk}; V, F = [], []
        bs, ms, mvs, mfs = [0], [0], [0], [0]
        for r in range(len(tiles)):
            z, i = zsrc[r]
            lab = list(z["labels"])
            for k in out:
                out[k].append(z[k][i])
            b0, b1 = int(z["box_start"][i]), int(z["box_start"][i + 1])
            for k in boxk:
                a = z[k][b0:b1]
                if k == "box_label":
                    a = np.array([labels.index(lab[x]) if x >= 0 else x for x in a], np.int16)
                B[k].append(a)
            bs.append(bs[-1] + b1 - b0)
            m0, m1 = int(z["mesh_start"][i]), int(z["mesh_start"][i + 1])
            for m in range(m0, m1):
                for k in meshk:
                    a = z[k][m]
                    if k == "mesh_label":
                        a = labels.index(lab[a]) if a >= 0 else a
                    M[k].append(a)
                v0, v1 = int(z["mesh_vert_start"][m]), int(z["mesh_vert_start"][m + 1])
                f0, f1 = int(z["mesh_face_start"][m]), int(z["mesh_face_start"][m + 1])
                V.append(z["mesh_vertices"][v0:v1]); F.append(z["mesh_faces"][f0:f1])
                mvs.append(mvs[-1] + v1 - v0); mfs.append(mfs[-1] + f1 - f0)
            ms.append(ms[-1] + m1 - m0)
        res = {k: np.stack(v) for k, v in out.items()}
        for k in boxk:
            res[k] = np.concatenate(B[k])
        for k in meshk:
            res[k] = np.array(M[k], dtype=z1[k].dtype).reshape((-1,) + z1[k].shape[1:])
        res.update(box_start=np.array(bs, np.int32), mesh_start=np.array(ms, np.int32), mesh_vert_start=np.array(mvs, np.int32),
                   mesh_face_start=np.array(mfs, np.int32), mesh_vertices=np.concatenate(V).astype(np.float32) if V else z1["mesh_vertices"][:0],
                   mesh_faces=np.concatenate(F).astype(np.int32) if F else z1["mesh_faces"][:0],
                   resolution=z1["resolution"], size=z1["size"], cell0_center=z1["cell0_center"], labels=np.array(labels))
        np.savez_compressed(f"{OUT}/stairs.npz", **res)
        zload = np.load(f"{OUT}/stairs.npz", allow_pickle=True)
        zsrc = {r: (zload, r) for r in range(len(tiles))}
    else:
        dst = f"{OUT}/{tt}.npz"
        if not os.path.exists(dst):
            os.symlink(f"{V1}/{tt}.npz", dst)
    # spawn-into-props filter
    n0 = n1 = s0 = s1 = moved = tight = 0
    near = []
    for r, t in enumerate(tiles):
        z, i = zsrc[r]
        sol = solids(z, i)
        keep = []
        for rt in t.get("routes", []):
            n0 += 1
            zs = list(rt.get("z", [0.0] * (len(rt["waypoints"]) + 1)))
            c = clearance(np.array(rt["start"][:2], float), zs[0], sol)
            near.append(c)
            if c >= MIN_CLEAR:
                keep.append(rt); n1 += 1
                continue
            # move the start forward along the route (5 cm steps, before the last waypoint) to the first point with clearance
            pts = [np.array(rt["start"][:2], float)] + [np.array(w[:2], float) for w in rt["waypoints"]]
            new, best = None, (c, 0, pts[0])
            for k in range(len(pts) - 1):
                a_, b_ = pts[k], pts[k + 1]
                L = np.linalg.norm(b_ - a_)
                last = k == len(pts) - 2
                for u in np.arange(0.05, L - (0.6 if last else 0.0), 0.05):
                    q = a_ + (b_ - a_) * u / L
                    cq = clearance(q, zs[k], sol)
                    if cq > best[0]:
                        best = (cq, k, q)
                    if cq >= MIN_CLEAR:
                        new = (k, q); break
                if new:
                    break
            if new is None and best[0] >= TIGHT_MIN:   # really tight: keep it, start at the widest point (user: tight routes stay)
                new = best[1:]; tight += 1
            if new is None:
                t.setdefault("routes_dropped_v4j", []).append(dict(rt, start_clearance_m=round(float(c), 3)))
                continue
            k, q = new
            r2 = dict(rt); yaw = math.atan2(pts[k + 1][1] - q[1], pts[k + 1][0] - q[0])
            r2["start"] = [float(q[0]), float(q[1]), yaw]
            r2["waypoints"] = rt["waypoints"][k:]
            r2["z"] = [zs[k]] + zs[k + 1:]
            r2["start_moved_v4j_m"] = round(float(np.linalg.norm(q - pts[0])), 2)
            keep.append(r2); n1 += 1; moved += 1
        t["routes"] = keep
        sp = []
        for s in t.get("spawns", []):
            s0 += 1
            c = clearance(np.array([s["x"], s["y"]], float), s.get("z", 0.0), sol)
            if c >= MIN_CLEAR:
                sp.append(dict(s, clearance_props_m=round(float(c), 3))); s1 += 1
        if sp:
            t["spawns"] = sp
        else:
            t["spawns_all_near_props"] = True   # kept as they were, so the tile still has a spawn
    j1["tiles"] = tiles
    json.dump(j1, open(f"{OUT}/{tt}.json", "w"))
    near = np.array(near) if near else np.zeros(1)
    report[tt] = dict(routes_in=n0, routes_kept=n1, starts_moved=moved, tight_kept=tight, spawns_in=s0, spawns_kept=s1,
                      start_clear_p5_p50=np.percentile(near, [5, 50]).round(2).tolist(), under_0p34=float((near < 0.34).mean()))
    print(tt, report[tt], flush=True)
json.dump(report, open(f"{OUT}/v4j_report.json", "w"), indent=1)
idx = {"note": "v4j terrain set, made by make_terrain_v4j.py from extended_v1 + extended_v1_2_stairs_inset; spawn / route start "
       "clearance >= %.2f m to props in the 0.25-1.5 m band" % MIN_CLEAR}
json.dump(idx, open(f"{OUT}/README_v4j.json", "w"))
