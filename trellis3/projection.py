"""
Projection multi-view texturing: repaint a baked GLB's base-colour texture from N posed photos.

Same idea as the "Projection MultiView Texturing" node in ComfyUI-Trellis2 (re-implemented here, CPU only):
each texel's 3D position is projected into every posed view; a texel takes colour from the views that see it
(depth-tested), face it, and have the subject inside their cut-out mask. Per-view weights are blended, and the
original TRELLIS texel keeps a small share so texels no view sees stay untouched.

Camera convention (glTF, Y up, model front = +Z), orthographic:
    yaw 0 front, 45 three-quarter, 90 profile (camera on +X, the face points to image-left), 180 back.
    image right = (cos y, 0, -sin y), image up = +Y, direction to camera = (sin y, 0, cos y).
Poses are only nominal, so each view is auto-fitted (yaw +-15 deg, scale, shift) to the mesh silhouette first.

    python -m trellis3.projection mesh.glb out.glb --view front.png:0 --view tq.png:45 --view back.png:180
"""
import argparse
import json
import math
import os
import time

import numpy as np

CANVAS = 256


# ------------------------------------------------------------------ geometry
def view_basis(yaw_deg):
    y = math.radians(yaw_deg)
    return (np.array([math.cos(y), 0.0, -math.sin(y)]), np.array([0.0, 1.0, 0.0]),
            np.array([math.sin(y), 0.0, math.cos(y)]))


def project(points, centre, yaw_deg, scale, dx, dy, cx, cy):
    """points (N,3) -> pixel x, y and depth toward the camera (larger = nearer)."""
    r, up, d = view_basis(yaw_deg)
    q = points - centre
    return cx + dx + scale * (q @ r), cy + dy - scale * (q @ up), q @ d


def rasterize_uv(uv, faces, H, W):
    """Face id (+1, 0 = empty) per texel and barycentric weights at texel centres."""
    import cv2
    ids = np.zeros((H, W), np.int32)
    px = np.stack([uv[..., 0] * (W - 1), (1.0 - uv[..., 1]) * (H - 1)], -1)   # (F,3,2)
    for i in range(len(faces)):
        cv2.fillConvexPoly(ids, px[i].round().astype(np.int32), int(i + 1))
    ys, xs = np.nonzero(ids)
    f = ids[ys, xs] - 1
    a, b, c = px[f, 0], px[f, 1], px[f, 2]
    p = np.stack([xs, ys], -1).astype(np.float64)
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    bary = np.clip(np.stack([1 - w1 - w2, w1, w2], -1), 0, None)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-9)
    return ids, ys, xs, f, bary


def texel_geometry(mesh, uv_faces, H, W):
    """3D position and unit normal of every texel covered by a face. Returns (ys, xs, pos, nrm)."""
    faces = np.asarray(mesh.faces)
    ids, ys, xs, f, bary = rasterize_uv(uv_faces, faces, H, W)
    v = np.asarray(mesh.vertices, np.float64)
    n = np.asarray(mesh.vertex_normals, np.float64)
    pos = np.einsum("nk,nkc->nc", bary, v[faces[f]])
    nrm = np.einsum("nk,nkc->nc", bary, n[faces[f]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    return ys, xs, pos, nrm


# ------------------------------------------------------------------ view fitting
def _alpha_canvas(alpha):
    import cv2
    h, w = alpha.shape
    cw = max(8, round(CANVAS * w / h))
    return cv2.resize((alpha > 127).astype(np.uint8), (cw, CANVAS), interpolation=cv2.INTER_AREA) > 0


def _silhouette(points, centre, yaw, s, dx, dy, cw):
    import cv2
    x, y, _ = project(points, centre, yaw, s, dx, dy, cw / 2, CANVAS / 2)
    m = np.zeros((CANVAS, cw), np.uint8)
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    ok = (xi >= 0) & (xi < cw) & (yi >= 0) & (yi < CANVAS)
    m[yi[ok], xi[ok]] = 1
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m > 0


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def fit_view(points, alpha, yaw0, yaw_range=15.0, seed=0, yaw_margin=0.02):
    """Fit yaw, scale and 2D shift of an orthographic view to the cut-out silhouette.
    Returns dict(yaw, scale, dx, dy, iou) in NATIVE image pixels (scale = px per model unit)."""
    rng = np.random.default_rng(seed)
    sub = points[rng.choice(len(points), min(len(points), 250_000), replace=False)]
    centre = (sub.min(0) + sub.max(0)) / 2
    target = _alpha_canvas(alpha)
    cw = target.shape[1]
    ys, xs = np.nonzero(target)
    if len(ys) == 0:
        raise ValueError("view has an empty cut-out mask")
    s0 = (ys.max() - ys.min() + 1) / max(1e-9, sub[:, 1].max() - sub[:, 1].min())
    best = None
    nominal = None
    for yo in sorted(np.arange(-yaw_range, yaw_range + 0.1, 5.0), key=abs):
        r, up, _ = view_basis(yaw0 + yo)
        q = (sub - centre)
        px_, py_ = q @ r, q @ up
        dx = (xs.min() + xs.max()) / 2 - cw / 2 - s0 * (px_.min() + px_.max()) / 2
        dy = (ys.min() + ys.max()) / 2 - CANVAS / 2 + s0 * (py_.min() + py_.max()) / 2
        sc = _iou(_silhouette(sub, centre, yaw0 + yo, s0, dx, dy, cw), target)
        if nominal is None:                      # yo == 0 comes first: the nominal pose is the default
            nominal = best = (sc, yaw0 + yo, s0, dx, dy)
        elif sc > best[0] + yaw_margin:         # off-nominal yaw must win clearly (symmetric shapes are yaw-blind)
            best = (sc, yaw0 + yo, s0, dx, dy)
    sc, yaw, s, dx, dy = best
    free_yaw = yaw != yaw0
    for step in (4.0, 2.0, 1.0, 0.5):
        improved = True
        while improved:
            improved = False
            for k, delta in (("s", s * 0.01 * step), ("dx", step), ("dy", step), ("yaw", step * 0.5 if free_yaw else 0.0)):
                if delta == 0.0:
                    continue
                for sign in (-1, 1):
                    cand = dict(s=s, dx=dx, dy=dy, yaw=yaw)
                    cand[k] += sign * delta
                    v = _iou(_silhouette(sub, centre, cand["yaw"], cand["s"], cand["dx"], cand["dy"], cw), target)
                    if v > sc + 1e-4:
                        sc, s, dx, dy, yaw = v, cand["s"], cand["dx"], cand["dy"], cand["yaw"]
                        improved = True
    f = alpha.shape[0] / CANVAS
    return {"yaw": float(yaw), "scale": float(s * f), "dx": float(dx * f), "dy": float(dy * f),
            "iou": round(sc, 4), "centre": centre}


# ------------------------------------------------------------------ weights and blend
def facing_weight(nrm, d, power=3.0, floor=0.1):
    cos = nrm @ d
    return np.clip((cos - floor) / (1 - floor), 0, 1) ** power


def depth_visible(x, y, z, size, extent, kernel=5, tol=0.02):
    """Splat the texel cloud into a z-buffer (nearest wins) and test each texel against it."""
    import cv2
    zmin = z.min()
    xi, yi = np.clip(x.round().astype(int), 0, size[1] - 1), np.clip(y.round().astype(int), 0, size[0] - 1)
    order = np.argsort(z)                                  # far first, near last: last write wins
    zb = np.full(size, -1e9, np.float32)
    zb[yi[order], xi[order]] = z[order]
    filled = cv2.dilate(zb, np.ones((kernel, kernel), np.uint8))
    return z >= filled[yi, xi] - tol * extent


def blend(base, view_colours, view_weights, base_weight=0.2, sharp=2.0):
    """base (N,3) float; per-view colours (N,3) and weights (N,). Weighted mean, base keeps `base_weight`."""
    num = base * base_weight
    den = np.full(len(base), base_weight, np.float64)
    for c, w in zip(view_colours, view_weights):
        ww = np.asarray(w, np.float64) ** sharp
        num = num + c * ww[:, None]
        den = den + ww
    return num / den[:, None], den - base_weight


def sample_image(img, x, y):
    import cv2
    mx = x.astype(np.float32).reshape(-1, 1)
    my = y.astype(np.float32).reshape(-1, 1)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(len(x), -1)


def _erode_alpha(alpha, frac=0.012):
    import cv2
    k = max(3, int(min(alpha.shape) * frac) | 1)
    return cv2.GaussianBlur(cv2.erode((alpha > 127).astype(np.uint8) * 255, np.ones((k, k), np.uint8)), (k, k), 0)


def fill_gutter(tex, covered):
    """Copy the nearest covered texel into uncovered texels so mip/bilinear sampling has no black fringe."""
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    return tex[idx[0], idx[1]]


# ------------------------------------------------------------------ driver
def load_views(specs):
    from PIL import Image
    out = []
    for s in specs:
        path, _, yaw = s.rpartition(":") if ":" in s[2:] else (s, "", "0")
        im = Image.open(path).convert("RGBA")
        out.append((path, float(yaw), np.asarray(im)))
    return out


def project_texture(glb_in, glb_out, view_specs, base_weight=0.2, sharp=2.0, power=3.0, fit=True, log=print):
    import trimesh
    t0 = time.time()
    scene = trimesh.load(glb_in)
    geom = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    mat = geom.visual.material
    base_tex = np.asarray(mat.baseColorTexture.convert("RGB")).copy()
    H, W = base_tex.shape[:2]
    uv = np.asarray(geom.visual.uv, np.float64)
    uv_faces = uv[np.asarray(geom.faces)]
    ys, xs, pos, nrm = texel_geometry(geom, uv_faces, H, W)
    log(f"texels covered: {len(ys):,} of {H * W:,} ({H}x{W})")
    base = base_tex[ys, xs].astype(np.float64)
    extent = float(np.ptp(pos[:, 1]))
    cols, wts, report = [], [], []
    for path, yaw, rgba in load_views(view_specs):
        alpha = rgba[..., 3]
        if alpha.min() == 255:
            raise ValueError(f"{path} has no cut-out alpha; run the cut-out step first")
        fitp = fit_view(pos, alpha, yaw) if fit else None
        if fitp is None:
            fitp = {"yaw": yaw, "scale": rgba.shape[0] / max(extent, 1e-9), "dx": 0.0, "dy": 0.0, "iou": None,
                    "centre": (pos.min(0) + pos.max(0)) / 2}
        h, w = alpha.shape
        x, y, z = project(pos, fitp["centre"], fitp["yaw"], fitp["scale"], fitp["dx"], fitp["dy"], w / 2, h / 2)
        vis = depth_visible(x * 0.5, y * 0.5, z, (h // 2 + 1, w // 2 + 1), extent)
        _, _, d = view_basis(fitp["yaw"])
        wf = facing_weight(nrm, d, power)
        wa = sample_image(_erode_alpha(alpha), x, y)[:, 0] / 255.0
        inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
        wt = wf * wa * vis * inside
        col = sample_image(np.ascontiguousarray(rgba[..., :3]), x, y).astype(np.float64)
        cols.append(col)
        wts.append(wt)
        report.append({"view": path, "yaw_nominal": yaw, "yaw_fit": round(fitp["yaw"], 1), "iou": fitp["iou"],
                       "texels_seen_pct": round(100 * float((wt > 0.05).mean()), 1)})
        log(f"  view {os.path.basename(path)}: yaw {yaw:g} -> {fitp['yaw']:.1f}, IoU {fitp['iou']}, "
            f"seen {report[-1]['texels_seen_pct']}%")
    new, strength = blend(base, cols, wts, base_weight, sharp)
    out_tex = base_tex.copy()
    out_tex[ys, xs] = np.clip(new, 0, 255).astype(np.uint8)
    covered = np.zeros((H, W), bool)
    covered[ys, xs] = True
    out_tex = fill_gutter(out_tex, covered)
    from PIL import Image
    mat.baseColorTexture = Image.fromarray(out_tex)
    scene.export(glb_out)
    stat = {"glb_in": glb_in, "glb_out": glb_out, "views": report, "base_weight": base_weight, "sharp": sharp,
            "texels_changed_pct": round(100 * float((strength > 0.05).mean()), 1), "seconds": round(time.time() - t0, 1)}
    json.dump(stat, open(glb_out + ".projection.json", "w"), indent=1)
    log(json.dumps(stat))
    return stat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb_in")
    ap.add_argument("glb_out")
    ap.add_argument("--view", action="append", required=True, metavar="IMG:YAW",
                    help="cut-out RGBA view and its nominal yaw in degrees (repeat per view; front 0, three-quarter 45, "
                         "profile 90, back 180)")
    ap.add_argument("--base-weight", type=float, default=0.2, help="share kept by the TRELLIS texel (default 0.2)")
    ap.add_argument("--sharp", type=float, default=2.0, help="view weight exponent; higher = less blending")
    ap.add_argument("--no-fit", action="store_true", help="trust the nominal poses")
    a = ap.parse_args(argv)
    project_texture(a.glb_in, a.glb_out, a.view, a.base_weight, a.sharp, fit=not a.no_fit)


if __name__ == "__main__":
    main()
