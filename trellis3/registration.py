"""
Feature-based registration of photo views to a mesh (orthographic camera solved from facial landmarks).

Silhouette IoU can fit an outline while the features sit a centimetre off (a ghosted double moustache). This module
solves each view's camera from corresponding landmarks instead:

    mesh landmarks   3D points (eye centres, nose tip, mouth corners, chin, ear) lifted from picks in a splat render of
                     the mesh whose camera is known exactly (`render_splat`, `lift`)
    image landmarks  2D picks on each cut-out view (by eye, or from any detector)

`solve_camera` fits yaw, scale, in-plane roll and shift by least squares so the projected mesh landmarks land on the
image landmarks; the residual (pixels, and as a share of the face height) is reported per view and per point.
Mirrored views reuse the original picks: x is flipped and left/right labels swap (`mirror_picks`).
"""
import json
import math

import numpy as np

SWAP = {"eye_vl": "eye_vr", "eye_vr": "eye_vl", "mouth_vl": "mouth_vr", "mouth_vr": "mouth_vl",
        "ear_vl": "ear_vr", "ear_vr": "ear_vl", "brow_vl": "brow_vr", "brow_vr": "brow_vl"}


def _basis(yaw_deg):
    y = math.radians(yaw_deg)
    return (np.array([math.cos(y), 0.0, -math.sin(y)]), np.array([0.0, 1.0, 0.0]),
            np.array([math.sin(y), 0.0, math.cos(y)]))


def project_roll(points, centre, yaw_deg, scale, roll_deg, tx, ty):
    """Orthographic projection with in-plane roll. Returns x, y (pixels, y down) and depth toward the camera."""
    r, up, d = _basis(yaw_deg)
    q = np.asarray(points, np.float64) - centre
    u, v = q @ r, -(q @ up)
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    return scale * (c * u - s * v) + tx, scale * (s * u + c * v) + ty, q @ d


def solve_camera(mesh_pts, img_pts, centre, yaw0, yaw_range=15.0, fix_yaw=False, affine=False, warp=True):
    """Least-squares camera from N >= 4 landmark correspondences.
    mesh_pts (N,3), img_pts (N,2) in image pixels.
    affine=False: similarity camera (yaw, scale, roll, shift).
    affine=True:  yaw + a full 2x2 matrix M + shift, which also absorbs an aspect difference between the photo face and
                  the mesh face (needs N >= 5 to stay determined). Returns a dict that `project_cam` understands."""
    from scipy.optimize import least_squares
    P, p = np.asarray(mesh_pts, np.float64), np.asarray(img_pts, np.float64)
    if len(P) < (5 if affine else 4):
        raise ValueError("need at least %d landmarks to solve this camera" % (5 if affine else 4))
    r0, up0, _ = _basis(yaw0)
    q = P - centre
    u0, v0 = q @ r0, -(q @ up0)
    A = np.c_[u0, v0]
    ca, cb = A.mean(0), p.mean(0)
    s0 = float(np.sqrt(((p - cb) ** 2).sum() / max(1e-12, ((A - ca) ** 2).sum())))
    n = len(P)
    ylo, yhi = (yaw0 - (0 if fix_yaw else yaw_range) - 1e-6), (yaw0 + (0 if fix_yaw else yaw_range) + 1e-6)

    if not affine:
        def resid(x):
            yaw, ls, roll, tx, ty = x
            px, py, _ = project_roll(P, centre, yaw, math.exp(ls), roll, tx, ty)
            return np.r_[px - p[:, 0], py - p[:, 1]]
        x0 = np.array([yaw0, math.log(s0), 0.0, cb[0] - s0 * ca[0], cb[1] - s0 * ca[1]])
        lo = np.array([ylo, -50, -25, -1e9, -1e9])
        hi = np.array([yhi, 50, 25, 1e9, 1e9])
        sol = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi))
        yaw, ls, roll, tx, ty = sol.x
        rr = resid(sol.x)
        per = np.hypot(rr[:n], rr[n:])
        return {"kind": "similarity", "yaw": float(yaw), "scale": float(math.exp(ls)), "roll": float(roll),
                "tx": float(tx), "ty": float(ty), "rms_px": float(np.sqrt((per ** 2).mean())),
                "resid_px": [round(float(v), 1) for v in per]}

    def resid_a(x):
        yaw, m00, m01, m10, m11, tx, ty = x
        r, up, _ = _basis(yaw)
        qq = P - centre
        u, v = qq @ r, -(qq @ up)
        return np.r_[m00 * u + m01 * v + tx - p[:, 0], m10 * u + m11 * v + ty - p[:, 1]]
    x0 = np.array([yaw0, s0, 0.0, 0.0, s0, cb[0] - s0 * ca[0], cb[1] - s0 * ca[1]])
    lo = np.array([ylo, 0.2 * s0, -0.6 * s0, -0.6 * s0, 0.2 * s0, -1e9, -1e9])
    hi = np.array([yhi, 5 * s0, 0.6 * s0, 0.6 * s0, 5 * s0, 1e9, 1e9])
    sol = least_squares(resid_a, np.clip(x0, lo, hi), bounds=(lo, hi))
    yaw, m00, m01, m10, m11, tx, ty = sol.x
    rr = resid_a(sol.x)
    per = np.hypot(rr[:n], rr[n:])
    out = {"kind": "affine", "yaw": float(yaw), "M": [[float(m00), float(m01)], [float(m10), float(m11)]],
           "tx": float(tx), "ty": float(ty), "rms_px": float(np.sqrt((per ** 2).mean())),
           "resid_px": [round(float(v), 1) for v in per]}
    if warp:                                              # residual (target - affine prediction) interpolated by a TPS in (u, v)
        r, up, _ = _basis(yaw)
        qq = P - centre
        uv = np.c_[qq @ r, -(qq @ up)]
        out["warp"] = {"uv": uv.tolist(), "res": (-np.c_[rr[:n], rr[n:]]).tolist()}
    return out


_WARPS = {}


def _warp_of(cam):
    """Thin-plate-spline interpolator of the landmark residuals (exact at the picks, linear far away)."""
    key = id(cam)
    if key not in _WARPS:
        from scipy.interpolate import RBFInterpolator
        w = cam["warp"]
        _WARPS[key] = RBFInterpolator(np.asarray(w["uv"]), np.asarray(w["res"]), kernel="thin_plate_spline", degree=1)
    return _WARPS[key]


def project_cam(points, cam, centre=None):
    """Project (N,3) points with a camera dict from `solve_camera`. Returns x, y (pixels) and depth toward the camera."""
    c = np.asarray(cam.get("centre", centre), np.float64)
    if cam.get("kind") == "affine":
        r, up, d = _basis(cam["yaw"])
        q = np.asarray(points, np.float64) - c
        u, v = q @ r, -(q @ up)
        M = cam["M"]
        x = M[0][0] * u + M[0][1] * v + cam["tx"]
        y = M[1][0] * u + M[1][1] * v + cam["ty"]
        w = cam.get("warp")
        if w:
            res = _warp_of(cam)(np.c_[u, v])
            x, y = x + res[:, 0], y + res[:, 1]
        return x, y, q @ d
    return project_roll(points, c, cam["yaw"], cam["scale"], cam["roll"], cam["tx"], cam["ty"])


def face_region_weight(pos, mesh3d, centre, a_max=1.4, b_min=-1.8, b_max=2.5, soft=0.2, depth_ed=1.7):
    """Soft 0..1 weight of the face region, measured in eye-distance units around the eyes in the front plane:
    |a| <= a_max sideways, b from b_min (above the brows) to b_max (below the chin), a = (u - eye mid)/eye distance,
    b = (v - eye line)/eye distance. Hair above the forehead and the bust below the neck fall outside."""
    r, up, _ = _basis(0.0)
    e1, e2 = np.asarray(mesh3d["eye_vl"]) - centre, np.asarray(mesh3d["eye_vr"]) - centre
    u1, u2 = e1 @ r, e2 @ r
    vl = -(((e1 + e2) / 2) @ up)
    ed = abs(u2 - u1)
    q = np.asarray(pos, np.float64) - centre
    a = (q @ r - (u1 + u2) / 2) / ed
    b = (-(q @ up) - vl) / ed

    def ramp(x, lo, hi):
        return np.clip((x - lo) / soft, 0, 1) * np.clip((hi - x) / soft, 0, 1)
    # depth gate (front direction +Z): keep texels in front of a plane `depth_ed` eye-distances behind the eyes (ears, cheeks,
    # forehead), so hair on the back of the head that happens to share the same front-plane (a, b) is excluded
    z = q @ np.array([0.0, 0.0, 1.0])
    ze = ((np.asarray(mesh3d["eye_vl"]) - centre) @ np.array([0.0, 0.0, 1.0]))
    zg = np.clip((z - (ze - depth_ed * ed)) / (soft * ed), 0, 1)
    return (ramp(a, -a_max, a_max) * ramp(b, b_min, b_max) * zg).astype(np.float32)


def mirror_picks(picks, width):
    """Picks of the horizontally flipped image: x -> width-1-x and left/right labels swap."""
    return {SWAP.get(k, k): [width - 1 - v[0], v[1]] for k, v in picks.items()}


# ------------------------------------------------------------------ splat render of the mesh (known camera)
def render_splat(pos, colour, nrm, centre, yaw, size=1400, scale=None, shade=True):
    """Render the texel cloud as an image from an exactly known orthographic camera.
    Returns (rgb uint8 (size,size,3), depth (size,size) float32 nearer = larger, dict(scale, tx, ty))."""
    import cv2
    r, up, d = _basis(yaw)
    q = pos.astype(np.float64) - centre
    u, v, z = q @ r, -(q @ up), q @ d
    if scale is None:
        ext = max(np.ptp(u), np.ptp(v))
        scale = 0.92 * size / ext
    tx, ty = size / 2 - scale * (u.min() + u.max()) / 2, size / 2 - scale * (v.min() + v.max()) / 2
    xi = np.clip(np.rint(scale * u + tx).astype(np.int32), 0, size - 1)
    yi = np.clip(np.rint(scale * v + ty).astype(np.int32), 0, size - 1)
    order = np.argsort(z)
    img = np.zeros((size, size, 3), np.float32)
    dep = np.full((size, size), -1e9, np.float32)
    col = colour.astype(np.float32)
    if shade:
        lam = np.clip(np.abs(nrm @ d), 0.15, 1.0)[:, None]
        col = col * (0.55 + 0.45 * lam)
    img[yi[order], xi[order]] = col[order]
    dep[yi[order], xi[order]] = z[order].astype(np.float32)
    mask = (dep > -1e8).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    for c in range(3):                                   # fill pinholes between splats with the local max
        img[..., c] = np.where(mask > 0, img[..., c], cv2.dilate(img[..., c], k, iterations=2))
    dep = np.where(mask > 0, dep, cv2.dilate(dep, k, iterations=2))
    return np.clip(img, 0, 255).astype(np.uint8), dep, {"scale": float(scale), "tx": float(tx), "ty": float(ty)}


def lift(px, py, depth, cam, centre, yaw, radius=7):
    """Back-project an image pick (pixel) of a `render_splat` image to 3D using its depth buffer. The depth is the NEAREST
    surface within `radius` pixels: gaps between splats show farther surfaces (the back of the head) through them."""
    xi, yi = int(round(px)), int(round(py))
    patch = depth[max(0, yi - radius):yi + radius + 1, max(0, xi - radius):xi + radius + 1]
    z = float(patch.max())
    r, up, d = _basis(yaw)
    u = (px - cam["tx"]) / cam["scale"]
    v = (py - cam["ty"]) / cam["scale"]
    return (np.asarray(centre, np.float64) + u * r - v * up + z * d)


def load_landmarks(path):
    return json.load(open(path, encoding="utf-8"))
