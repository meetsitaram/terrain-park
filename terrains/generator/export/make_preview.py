#!/usr/bin/env python
"""Top-down preview of the exported set, drawn from the files on disk (numpy + OpenCV, CPU only). Not a simulator render.

  python -B make_preview.py --data OUT/extended_v1
writes preview_top.png (every tile; columns = type variants, rows = difficulty level, easy at the bottom)
and preview_closeups.png (one tile per type at levels 0, 4 and 9)."""
import argparse
import json
import math
import pathlib

import cv2
import numpy as np
from matplotlib import colormaps

TILE = 8.0
MAGENTA = (255, 0, 255)   # BGR
CYAN = (255, 200, 0)
ORANGE = (0, 140, 255)
TALL_FILL = (70, 62, 58)
H_MAX = 3.5
CMAP = (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]  # BGR


def colour(h):
    v = np.sqrt(np.clip(h, 0, H_MAX) / H_MAX)
    img = CMAP[(v * 255).astype(int)]
    img[h <= 1e-4] = (226, 226, 226)  # flat ground at z = 0: light grey
    return img


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_corners(c, h, q):
    R = quat_R(q)
    s = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    return c + (s * h) @ R.T


def render_tile(Z, i, meta, P):
    """One tile as a P x P BGR image: x to the right, y up."""
    h = Z["height"][i]
    k = max(1, int(math.floor(h.shape[0] / P)))
    hh = h[:h.shape[0] // k * k, :h.shape[1] // k * k].reshape(h.shape[0] // k, k, h.shape[1] // k, k).max((1, 3)) if k > 1 else h
    img = colour(np.flipud(hh.T))
    img = cv2.resize(img, (P, P), interpolation=cv2.INTER_AREA if img.shape[0] > P else cv2.INTER_NEAREST)
    img = np.ascontiguousarray(img)
    S = 8  # sub-pixel bits

    def px(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.c_[(xy[:, 0] + TILE / 2) / TILE * P, (TILE / 2 - xy[:, 1]) / TILE * P]

    def ipx(xy):
        return np.round(px(xy) * (1 << S)).astype(np.int32)

    polys = []
    for b in range(Z["box_start"][i], Z["box_start"][i + 1]):
        if Z["box_in_raster"][b]:
            continue
        cn = box_corners(Z["box_center"][b], Z["box_half"][b], Z["box_quat"][b])
        polys.append((cn, int(Z["box_role"][b])))
    for m in range(Z["mesh_start"][i], Z["mesh_start"][i + 1]):
        if Z["mesh_in_raster"][m]:
            continue
        v = Z["mesh_vertices"][Z["mesh_vert_start"][m]:Z["mesh_vert_start"][m + 1]]
        polys.append((v, int(Z["mesh_role"][m])))
    lw = 1 if P < 300 else 2
    for cn, role in sorted(polys, key=lambda t: t[0][:, 2].max()):
        hull = cv2.convexHull(ipx(cn[:, :2]).reshape(-1, 1, 2))
        overhead = cn[:, 2].min() > 1.6
        if role == 2 and not overhead:
            cv2.fillPoly(img, [hull], TALL_FILL, lineType=cv2.LINE_AA, shift=S)
            cv2.polylines(img, [hull], True, (255, 255, 255), 1, cv2.LINE_AA, shift=S)
        elif role == 2:
            cv2.polylines(img, [hull], True, (160, 160, 160), 1, cv2.LINE_AA, shift=S)
        else:
            cv2.polylines(img, [hull], True, ORANGE, lw, cv2.LINE_AA, shift=S)

    def pose(x, y, yaw, col, r):
        c = ipx([(x, y)])[0]
        cv2.circle(img, tuple(int(a) for a in c), int(r * (1 << S)), col, lw, cv2.LINE_AA, shift=S)
        e = ipx([(x + 0.35 * math.cos(yaw), y + 0.35 * math.sin(yaw))])[0]
        cv2.line(img, tuple(int(a) for a in c), tuple(int(a) for a in e), col, lw, cv2.LINE_AA, shift=S)

    for s in meta["spawns"]:
        if s["source"] != "route":
            pose(s["x"], s["y"], s["yaw"], CYAN, max(2.0, 0.12 / TILE * P))
    dash = 0.22  # m
    for rt in meta["routes"]:
        pts = np.array([rt["start"][:2]] + rt["waypoints"], float)
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            n = max(1, int(L / dash))
            for q in range(0, n, 2):
                p0, p1 = a + (b - a) * q / n, a + (b - a) * min(q + 1, n) / n
                cv2.line(img, tuple(int(v) for v in ipx([p0])[0]), tuple(int(v) for v in ipx([p1])[0]), MAGENTA, lw, cv2.LINE_AA, shift=S)
        a, b = pts[-2], pts[-1]
        u = (b - a) / max(1e-9, np.linalg.norm(b - a))
        nrm = np.array([-u[1], u[0]])
        tri = np.array([b, b - 0.30 * u + 0.13 * nrm, b - 0.30 * u - 0.13 * nrm])
        cv2.fillPoly(img, [ipx(tri).reshape(-1, 1, 2)], MAGENTA, lineType=cv2.LINE_AA, shift=S)
        pose(rt["start"][0], rt["start"][1], rt["start"][2], MAGENTA, max(2.0, 0.12 / TILE * P))
    cv2.rectangle(img, (0, 0), (P - 1, P - 1), (120, 120, 120), 1)
    return img


def text(img, s, org, scale=0.5, col=(235, 235, 235), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def legend(img, x, y):
    w = 260
    bar = CMAP[np.linspace(0, 255, w).astype(int)][None].repeat(14, 0)
    img[y:y + 14, x:x + w] = bar
    for m in (0.05, 0.25, 1.0, 2.0, 3.5):
        u = x + int(math.sqrt(m / H_MAX) * (w - 1))
        cv2.line(img, (u, y + 14), (u, y + 19), (235, 235, 235), 1)
        text(img, "%g" % m, (u - 10, y + 33), 0.4)
    text(img, "floor raster height (m); flat ground z=0 is light grey", (x + w + 14, y + 13), 0.45)
    xx = x + w + 430
    cv2.rectangle(img, (xx, y), (xx + 22, y + 14), TALL_FILL, -1)
    cv2.rectangle(img, (xx, y), (xx + 22, y + 14), (255, 255, 255), 1)
    text(img, "tall box/mesh (not in raster)", (xx + 30, y + 13), 0.45)
    xx += 270
    cv2.rectangle(img, (xx, y), (xx + 22, y + 14), ORANGE, 1)
    text(img, "low slanted/lifted box (not in raster)", (xx + 30, y + 13), 0.45)
    xx += 320
    for q in range(0, 40, 12):
        cv2.line(img, (xx + q, y + 7), (xx + q + 6, y + 7), MAGENTA, 2, cv2.LINE_AA)
    text(img, "commanded route (drawing only, NOT geometry / not a foothold); circle = start pose", (xx + 50, y + 13), 0.45)
    xx += 700
    cv2.circle(img, (xx + 8, y + 7), 5, CYAN, 1, cv2.LINE_AA)
    text(img, "other spawn pose", (xx + 22, y + 13), 0.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--px", type=int, default=160)
    ap.add_argument("--px-close", type=int, default=440)
    args = ap.parse_args()
    D = pathlib.Path(args.data)
    index = json.load(open(D / "index.json"))
    keys = list(index["types"].keys())
    data = {k: (np.load(D / (k + ".npz")), json.load(open(D / (k + ".json")))["tiles"]) for k in keys}
    levels = index["levels"]
    P = args.px
    ncol = index["layout"]["total_columns"]
    left, top = 46, 96
    img = np.full((top + levels * P + 6, left + ncol * P + 6, 3), (30, 24, 20), np.uint8)
    text(img, "extended_v1 terrain set: PREVIEW, NOT A SIMULATOR RENDER.  %d tiles of 8 x 8 m, 2 cm rasters. Columns: tile type and variant; rows: difficulty level (L0 easy, bottom)."
         % index["totals"]["tiles"], (10, 22), 0.6)
    legend(img, 10, 34)
    for k in keys:
        Z, metas = data[k]
        Z = {n_: Z[n_] for n_ in Z.files}
        c0 = index["layout"]["columns"][k]["first_column"]
        nc = index["layout"]["columns"][k]["columns"]
        text(img, k, (left + c0 * P + 4, top - 8), 0.55, (255, 255, 255), 1)
        cv2.line(img, (left + c0 * P + 2, top - 3), (left + (c0 + nc) * P - 3, top - 3), (200, 200, 200), 1)
        for m in metas:
            t = render_tile(Z, m["index"], m, P)
            r = levels - 1 - m["level"]
            c = c0 + m["variant"]
            img[top + r * P:top + (r + 1) * P, left + c * P:left + (c + 1) * P] = t
            if m["variant_kind"]:
                text(img, m["variant_kind"][:10], (left + c * P + 3, top + r * P + 12), 0.36, (0, 0, 0), 1)
        data[k] = (Z, metas)
    for l in range(levels):
        text(img, "L%d" % l, (8, top + (levels - 1 - l) * P + P // 2), 0.55)
    cv2.imwrite(str(D / "preview_top.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print("wrote preview_top.png", img.shape)

    P = args.px_close
    lv = [l for l in (0, 4, 9) if l < levels]
    img = np.full((60 + len(lv) * P, 46 + len(keys) * P, 3), (30, 24, 20), np.uint8)
    text(img, "extended_v1 close-ups: PREVIEW, NOT A SIMULATOR RENDER. One tile (variant 0) per type at levels 0, 4, 9. Magenta dashes = commanded routes (drawing only, not geometry).",
         (10, 22), 0.6)
    for ci, k in enumerate(keys):
        Z, metas = data[k]
        text(img, k, (46 + ci * P + 6, 50), 0.6, (255, 255, 255), 1)
        for ri, l in enumerate(reversed(lv)):
            m = [t for t in metas if t["level"] == l and t["variant"] == 0]
            if m:
                img[60 + ri * P:60 + (ri + 1) * P, 46 + ci * P:46 + (ci + 1) * P] = render_tile(Z, m[0]["index"], m[0], P)
    for ri, l in enumerate(reversed(lv)):
        text(img, "L%d" % l, (8, 60 + ri * P + P // 2), 0.6)
    cv2.imwrite(str(D / "preview_closeups.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print("wrote preview_closeups.png", img.shape)


if __name__ == "__main__":
    main()
