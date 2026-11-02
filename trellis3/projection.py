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
                    if k == "yaw" and abs(cand["yaw"] - yaw0) > yaw_range:
                        continue                      # never leave the allowed window around the nominal pose
                    v = _iou(_silhouette(sub, centre, cand["yaw"], cand["s"], cand["dx"], cand["dy"], cw), target)
                    if v > sc + 1e-4:
                        sc, s, dx, dy, yaw = v, cand["s"], cand["dx"], cand["dy"], cand["yaw"]
                        improved = True
    f = alpha.shape[0] / CANVAS
    return {"yaw": float(yaw), "scale": float(s * f), "dx": float(dx * f), "dy": float(dy * f),
            "iou": round(sc, 4), "centre": centre}


# ------------------------------------------------------------------ weights and blend
def facing_weight(nrm, d, power=3.0, floor=0.1, signed=False):
    """Weight from the angle between surface normal and view direction.

    TRELLIS meshes have ragged, inconsistently wound faces, so the sign of a vertex normal is unreliable
    (about 40% point inward on a real head). The depth test already restricts painting to the surface nearest the
    camera, so by default only |cos| is used; `signed=True` zeroes back-facing normals (clean meshes).
    """
    cos = nrm @ d
    if not signed:
        cos = np.abs(cos)
    return np.clip((cos - floor) / (1 - floor), 0, 1) ** power


def depth_visible(x, y, z, size, extent, kernel=5, tol=0.02):
    """Splat the texel cloud into a z-buffer (nearest wins) and test each texel against it."""
    zb = depth_buffer(x, y, z, size)
    return depth_test(zb, x, y, z, extent, kernel, tol)


def depth_buffer(x, y, z, size, zb=None, chunk=4_000_000):
    """Accumulate the nearest depth per pixel (nearest = larger z). Chunked so 8K texel clouds fit in RAM."""
    if zb is None:
        zb = np.full(size, -1e9, np.float32)
    for i in range(0, len(x), chunk):
        xi = np.clip(np.rint(x[i:i + chunk]).astype(np.int32), 0, size[1] - 1)
        yi = np.clip(np.rint(y[i:i + chunk]).astype(np.int32), 0, size[0] - 1)
        zc = z[i:i + chunk].astype(np.float32)
        o = np.argsort(zc)                                  # far first, near last: last write wins
        xi, yi, zc = xi[o], yi[o], zc[o]
        zb[yi, xi] = np.maximum(zb[yi, xi], zc)
    return zb


def depth_test(zb, x, y, z, extent, kernel=5, tol=0.02):
    import cv2
    filled = cv2.dilate(zb, np.ones((kernel, kernel), np.uint8))
    xi = np.clip(np.rint(x).astype(np.int32), 0, zb.shape[1] - 1)
    yi = np.clip(np.rint(y).astype(np.int32), 0, zb.shape[0] - 1)
    return z >= filled[yi, xi] - tol * extent


def blend(base, view_colours, view_weights, tau=0.15, sharp=2.0):
    """base (N,3) float; per-view colours (N,3) and weights (N,) in [0,1].

    The views are averaged against each other (weights ** sharp), so a grazing view cannot be drowned by the
    original texel. How much of the original survives is decided by confidence = min(1, strongest view weight / tau):
    texels no view sees keep the original, texels one view sees well take that view. Returns (colour, confidence).
    """
    ws = [np.asarray(w, np.float64) for w in view_weights]
    num = np.zeros_like(base, dtype=np.float64)
    den = np.zeros(len(base), np.float64)
    for c, w in zip(view_colours, ws):
        ww = w ** sharp
        num += c * ww[:, None]
        den += ww
    view = num / np.maximum(den, 1e-12)[:, None]
    conf = np.clip(np.max(np.stack(ws), axis=0) / tau, 0.0, 1.0) if ws else np.zeros(len(base))
    return view * conf[:, None] + base * (1.0 - conf)[:, None], conf


def sample_image(img, x, y, width=2048):
    """Bilinear lookup of img (H,W[,C]) at float pixel coords. Returns (N, C). Maps are laid out as a 2-D grid
    because cv2.remap rejects more than 32767 rows."""
    import cv2
    n = len(x)
    rows = -(-n // width)
    mx = np.zeros(rows * width, np.float32)
    my = np.zeros(rows * width, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(img, mx.reshape(rows, width), my.reshape(rows, width), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(rows * width, -1)[:n]


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
def parse_view_spec(spec):
    """'path:yaw[:weight]' -> (path, yaw, weight). A Windows drive colon in the path is left alone."""
    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    parts = spec.rsplit(":", 2)
    if len(parts) == 3 and num(parts[1]) is not None and num(parts[2]) is not None and len(parts[0]) > 1:
        return parts[0], num(parts[1]), num(parts[2])
    parts = spec.rsplit(":", 1)
    if len(parts) == 2 and num(parts[1]) is not None and len(parts[0]) > 1:
        return parts[0], num(parts[1]), 1.0
    return spec, 0.0, 1.0


def load_views(specs):
    from PIL import Image
    out = []
    for s in specs:
        path, yaw, weight = parse_view_spec(s)
        im = Image.open(path).convert("RGBA")
        out.append((path, yaw, np.asarray(im), weight))
    return out


class Accumulator:
    """Running weighted blend of the views: avoids holding every view's colours at once (8K = 30M texels)."""

    def __init__(self, n, sharp=2.0):
        self.num = np.zeros((n, 3), np.float32)
        self.den = np.zeros(n, np.float32)
        self.maxw = np.zeros(n, np.float32)
        self.sharp = sharp

    def add(self, col, w, sl=slice(None)):
        """Add one view's colours/weights for the texels in `sl` (a chunk of the full texel list)."""
        w = w.astype(np.float32)
        ww = w ** self.sharp
        self.num[sl] += col.astype(np.float32) * ww[:, None]
        self.den[sl] += ww
        np.maximum(self.maxw[sl], w, out=self.maxw[sl])

    def result(self, base, tau):
        view = self.num / np.maximum(self.den, 1e-12)[:, None]
        conf = np.clip(self.maxw / tau, 0.0, 1.0)
        return view * conf[:, None] + base.astype(np.float32) * (1.0 - conf)[:, None], conf


def project_texture(glb_in, glb_out, view_specs, tau=0.15, sharp=2.0, power=3.0, fit=True, size=None,
                    out_png=None, front_only_mask=None, chunk=4_000_000, log=print):
    """Repaint the base-colour texture of `glb_in` from posed RGBA views.

    size: output texture side (default: the GLB's own texture; a missing texture starts as neutral grey).
    front_only_mask: UV-space image; white texels (eyes, mouth interior) are painted by the FIRST view only.
    out_png: also write the new base-colour texture as a PNG.
    """
    import trimesh
    from PIL import Image
    t0 = time.time()
    scene = trimesh.load(glb_in, process=False)
    geom = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    mat = geom.visual.material
    tex0 = getattr(mat, "baseColorTexture", None)
    if tex0 is None:
        tex0 = Image.new("RGB", (size or 2048,) * 2, (128, 128, 128))
    if size and tex0.size != (size, size):
        tex0 = tex0.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    base_tex = np.asarray(tex0.convert("RGB")).copy()
    H, W = base_tex.shape[:2]
    uv = np.asarray(geom.visual.uv, np.float64)
    uv_faces = uv[np.asarray(geom.faces)]
    ys, xs, pos, nrm = texel_geometry(geom, uv_faces, H, W)
    pos, nrm = pos.astype(np.float32), nrm.astype(np.float32)
    n = len(ys)
    log(f"texels covered: {n:,} of {H * W:,} ({H}x{W})")
    base = base_tex[ys, xs]
    extent = float(np.ptp(pos[:, 1]))
    centre0 = ((pos.min(0) + pos.max(0)) / 2).astype(np.float64)
    front_only = None
    if front_only_mask is not None:
        m = Image.open(front_only_mask).convert("L").resize((W, H), Image.Resampling.NEAREST)
        front_only = (np.asarray(m)[ys, xs] > 127)
        log(f"front-only texels: {int(front_only.sum()):,}")
    acc = Accumulator(n, sharp)
    report = []
    for vi, (path, yaw, rgba, vweight) in enumerate(load_views(view_specs)):
        alpha = rgba[..., 3]
        if alpha.min() == 255:
            raise ValueError(f"{path} has no cut-out alpha; run the cut-out step first")
        fitp = fit_view(pos, alpha, yaw) if fit else None
        if fitp is None:
            fitp = {"yaw": yaw, "scale": rgba.shape[0] / max(extent, 1e-9), "dx": 0.0, "dy": 0.0, "iou": None,
                    "centre": centre0}
        h, w = alpha.shape
        zb = np.full((h // 2 + 1, w // 2 + 1), -1e9, np.float32)
        for i in range(0, n, chunk):                     # pass 1: nearest-surface depth buffer
            x, y, z = project(pos[i:i + chunk], fitp["centre"], fitp["yaw"], fitp["scale"], fitp["dx"],
                              fitp["dy"], w / 2, h / 2)
            depth_buffer(x * 0.5, y * 0.5, z, zb.shape, zb)
        _, _, d = view_basis(fitp["yaw"])
        ealpha = _erode_alpha(alpha)
        rgb = np.ascontiguousarray(rgba[..., :3])
        seen = 0
        for i in range(0, n, chunk):                     # pass 2: weights and colour per chunk
            sl = slice(i, i + chunk)
            x, y, z = project(pos[sl], fitp["centre"], fitp["yaw"], fitp["scale"], fitp["dx"], fitp["dy"],
                              w / 2, h / 2)
            vis = depth_test(zb, x * 0.5, y * 0.5, z, extent)
            wf = facing_weight(nrm[sl], d, power)
            wa = sample_image(ealpha, x, y)[:, 0] / 255.0
            inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
            wt = wf * wa * vis * inside * vweight
            if front_only is not None and vi > 0:
                wt = np.where(front_only[sl], 0.0, wt)
            acc.add(sample_image(rgb, x, y), wt, sl)
            seen += int((wt > 0.05).sum())
        report.append({"view": path, "weight": vweight, "yaw_nominal": yaw, "yaw_fit": round(fitp["yaw"], 1), "iou": fitp["iou"],
                       "texels_seen_pct": round(100 * seen / n, 1)})
        log(f"  view {os.path.basename(path)}: yaw {yaw:g} -> {fitp['yaw']:.1f}, IoU {fitp['iou']}, "
            f"seen {report[-1]['texels_seen_pct']}%")
    new, conf = acc.result(base, tau)
    out_tex = base_tex.copy()
    out_tex[ys, xs] = np.clip(new, 0, 255).astype(np.uint8)
    covered = np.zeros((H, W), bool)
    covered[ys, xs] = True
    out_tex = fill_gutter(out_tex, covered)
    img = Image.fromarray(out_tex)
    if out_png:
        img.save(out_png)
    mat.baseColorTexture = img
    scene.export(glb_out)
    stat = {"glb_in": glb_in, "glb_out": glb_out, "texture": [W, H], "views": report, "tau": tau, "sharp": sharp,
            "front_only_mask": front_only_mask, "out_png": out_png,
            "texels_changed_pct": round(100 * float((conf > 0.05).mean()), 1), "seconds": round(time.time() - t0, 1)}
    json.dump(stat, open(glb_out + ".projection.json", "w"), indent=1)
    log(json.dumps(stat))
    return stat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb_in")
    ap.add_argument("glb_out")
    ap.add_argument("--view", action="append", required=True, metavar="IMG:YAW",
                    help="cut-out RGBA view, its nominal yaw in degrees and an optional weight: IMG:YAW[:WEIGHT] (repeat per view; "
                         "front 0, three-quarter 45, profile 90, back 180; the mirrored profile is 270)")
    ap.add_argument("--tau", type=float, default=0.15, help="view weight at which a texel is fully repainted (default 0.15); "
                                                              "lower repaints more, higher keeps more of the TRELLIS texel")
    ap.add_argument("--sharp", type=float, default=2.0, help="view weight exponent; higher = less blending")
    ap.add_argument("--no-fit", action="store_true", help="trust the nominal poses")
    ap.add_argument("--size", type=int, default=None, help="output texture side in pixels (default: the GLB's own, e.g. 8192)")
    ap.add_argument("--out-png", default=None, help="also write the new base-colour texture as a PNG")
    ap.add_argument("--front-only-mask", default=None,
                    help="UV-space mask image; white texels (eyes, mouth interior) are painted by the first view only")
    a = ap.parse_args(argv)
    project_texture(a.glb_in, a.glb_out, a.view, a.tau, a.sharp, fit=not a.no_fit, size=a.size,
                    out_png=a.out_png, front_only_mask=a.front_only_mask)


if __name__ == "__main__":
    main()
