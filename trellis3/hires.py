"""
CPU nearest-surface queries against the full-resolution decoder mesh.

Why: the Metal bake maps every texel back to "the original mesh" through a
BVH (mtlbvh), and mtlbvh is unstable above ~200K faces. trellis-mac therefore
pre-simplifies to 200K faces *before* the bake, so the BVH only ever sees the
simplified surface and the detail of the 800K+ decoder mesh is lost. This
module gives the same two things the BVH provided, on CPU, from the full
mesh:

  1. snapping texel positions from the simplified surface onto the
     full-res surface before sampling voxel attributes (sharper colour), and
  2. a tangent-space normal map baked from the full-res surface onto the
     simplified, UV-unwrapped mesh (detail that survives decimation).

The query is a k-nearest-centroid search (scipy cKDTree) followed by an exact
point-triangle closest-point test over the k candidates. For decoder meshes,
whose triangles are voxel-sized and uniform, k=8 finds the true nearest
triangle in practice; it is not a guaranteed-exact BVH. Fully vectorised.
"""

import time

import numpy as np


def _closest_point_on_triangles(p, a, b, c):
    """Closest point on triangle (a,b,c) to p, all [..., 3]. Ericson, RTCD 5.1.5.

    Returns (closest [...,3], barycentric [...,3] as (wa, wb, wc)).
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # Start from the face-interior solution and overwrite region by region.
    denom = 1.0 / np.where(va + vb + vc == 0, 1e-30, va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w

    # edge AB
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.where(d1 - d3 == 0, 0.0, d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3))
    u = np.where(m, 1 - t, u); v = np.where(m, t, v); w = np.where(m, 0, w)
    # edge AC
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.where(d2 - d6 == 0, 0.0, d2 / np.where(d2 - d6 == 0, 1.0, d2 - d6))
    u = np.where(m, 1 - t, u); v = np.where(m, 0, v); w = np.where(m, t, w)
    # edge BC
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    s = (d4 - d3) + (d5 - d6)
    t = np.where(s == 0, 0.0, (d4 - d3) / np.where(s == 0, 1.0, s))
    u = np.where(m, 0, u); v = np.where(m, 1 - t, v); w = np.where(m, t, w)
    # vertices
    m = (d1 <= 0) & (d2 <= 0)
    u = np.where(m, 1, u); v = np.where(m, 0, v); w = np.where(m, 0, w)
    m = (d3 >= 0) & (d4 <= d3)
    u = np.where(m, 0, u); v = np.where(m, 1, v); w = np.where(m, 0, w)
    m = (d6 >= 0) & (d5 <= d6)
    u = np.where(m, 0, u); v = np.where(m, 0, v); w = np.where(m, 1, w)

    bary = np.stack([u, v, w], axis=-1)
    closest = a * u[..., None] + b * v[..., None] + c * w[..., None]
    return closest, bary


class NearestSurface:
    """Nearest point / normal on a triangle mesh.

    Two query modes:
      exact=False (default): nearest *vertex* via cKDTree. On a decoder mesh
          (voxel-sized, uniform triangles) the error is at most half an edge,
          below the voxel size used for colour sampling. ~4M points in seconds.
      exact=True: k nearest centroids, then an exact point-triangle test over
          the candidates. Matches trimesh.proximity to 1e-5 but is ~100x
          slower in numpy; use for small meshes or validation.
    """

    def __init__(self, vertices, faces, k=8, exact=False):
        from scipy.spatial import cKDTree
        self.v = np.ascontiguousarray(vertices, dtype=np.float32)
        self.f = np.ascontiguousarray(faces, dtype=np.int64)
        self.k = min(k, len(self.f))
        self.exact = exact
        tri = self.v[self.f]                      # [F,3,3]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(n, axis=1, keepdims=True)
        self.face_normals = n / np.maximum(area2, 1e-20)
        vn = np.zeros_like(self.v)
        for i in range(3):                        # area-weighted vertex normals
            np.add.at(vn, self.f[:, i], n)
        self.vertex_normals = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)
        self._vtree = cKDTree(self.v)
        self._ctree = None
        if exact:
            self._ctree = cKDTree(tri.mean(axis=1))
        # one incident face per vertex, for face_id in vertex mode
        self._vface = np.full(len(self.v), -1, dtype=np.int64)
        self._vface[self.f[:, 2]] = np.arange(len(self.f))
        self._vface[self.f[:, 1]] = np.arange(len(self.f))
        self._vface[self.f[:, 0]] = np.arange(len(self.f))

    def query(self, points, chunk=500_000, exact=None):
        """Return (closest [M,3], face_id [M], bary [M,3], dist [M])."""
        points = np.asarray(points, dtype=np.float32)
        if not (self.exact if exact is None else exact):
            d, vi = self._vtree.query(points, k=1, workers=-1)
            fid = self._vface[vi]
            bary = (self.f[fid] == vi[:, None]).astype(np.float32)
            return self.v[vi], fid, bary, d.astype(np.float32)
        if self._ctree is None:
            from scipy.spatial import cKDTree
            self._ctree = cKDTree(self.v[self.f].mean(axis=1))
        out_p, out_f, out_b, out_d = [], [], [], []
        for s in range(0, len(points), chunk):
            p = points[s:s + chunk]
            _, cand = self._ctree.query(p, k=self.k, workers=-1)
            cand = cand.reshape(len(p), -1)       # [m,k] even when k == 1
            tri = self.v[self.f[cand]]            # [m,k,3,3]
            cp, bary = _closest_point_on_triangles(p[:, None, :], tri[:, :, 0], tri[:, :, 1], tri[:, :, 2])
            d = np.linalg.norm(cp - p[:, None, :], axis=-1)
            best = d.argmin(axis=1)
            r = np.arange(len(p))
            out_p.append(cp[r, best]); out_f.append(cand[r, best])
            out_b.append(bary[r, best]); out_d.append(d[r, best])
        return (np.concatenate(out_p), np.concatenate(out_f),
                np.concatenate(out_b), np.concatenate(out_d))

    def normal(self, face_id, bary, smooth=True):
        if not smooth:
            return self.face_normals[face_id]
        n = (self.vertex_normals[self.f[face_id]] * bary[..., None]).sum(axis=1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


# ---------------------------------------------------------------------------
# Normal map bake
# ---------------------------------------------------------------------------

def _face_tangents(vertices, faces, uvs):
    """Per-face tangent and bitangent from UV derivatives. [F,3] each."""
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    t0, t1, t2 = (uvs[faces[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = 1.0 / np.where(np.abs(det) < 1e-12, 1e-12, det)
    T = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    B = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-20)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-20)
    return T, B


def vertex_normals_welded(vertices, faces):
    """Smooth, area-weighted vertex normals of the low mesh, welded by POSITION so UV-seam copies of a vertex share one normal (what the exporter's own vertex
    normals are: computed before the UV split). [V,3]."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    eps = max(float(np.ptp(V, axis=0).max()), 1e-12) * 1e-6
    _, inv = np.unique(np.round(V / eps).astype(np.int64), axis=0, return_inverse=True)
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[F[:, k]], fn)
    n = acc[inv]
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)).astype(np.float32)


def _vertex_tangents(vertices, faces, uvs, normals):
    """Per-vertex tangent (orthogonalised against `normals`) and handedness, accumulated from the UV-derivative face tangents like a renderer's tangent generator does
    (per UV-split vertex). [V,3], [V,1]."""
    T, B = _face_tangents(vertices, faces, uvs)
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)[:, None]
    nf = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-20)
    hand = np.sign((np.cross(nf, T) * B).sum(1, keepdims=True))
    hand[hand == 0] = 1
    tv = np.zeros((len(V), 3)); hv = np.zeros((len(V), 1))
    for k in range(3):
        np.add.at(tv, F[:, k], T * area)
        np.add.at(hv, F[:, k], hand * area)
    t = tv - normals * (tv * normals).sum(1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-20)
    h = np.where(hv >= 0, 1.0, -1.0)
    return t.astype(np.float32), h.astype(np.float32)


def _barycentric(p, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-30, 1e-30, d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], axis=1).astype(np.float32)


def _rasterize(vertices, faces, uvs, texture_size, v_down):
    """UV rasteriser that also returns the covering face id per texel."""
    from backends.texture_baker import _rasterize_uv_triangles
    # Reuse the upstream rasteriser for positions; recover face ids with a
    # second pass keyed on the face index written into a float channel.
    uv = uvs.copy()
    if not v_down:
        uv[:, 1] = 1.0 - uv[:, 1]
    ids = np.arange(len(faces), dtype=np.float32)[:, None].repeat(3, axis=1)
    # Encode the face id as a fake "vertex" so interpolation returns it exactly
    # (all three corners carry the same value).
    fake_v = np.zeros((len(faces) * 3, 3), dtype=np.float32)
    fake_v[:, 0] = ids.reshape(-1)
    fake_f = np.arange(len(faces) * 3).reshape(-1, 3)
    fake_uv = uv[faces].reshape(-1, 2)
    positions, mask = _rasterize_uv_triangles(vertices, faces, uv, texture_size)
    fid_img, _ = _rasterize_uv_triangles(fake_v, fake_f, fake_uv, texture_size)
    face_ids = np.rint(fid_img[..., 0]).astype(np.int64)
    return positions, mask, face_ids


def bake_normal_map(low_vertices, low_faces, low_uvs, hires, texture_size=2048,
                    v_down=False, smooth=True, dilate=8, log=print, low_normals=None, shading="smooth"):
    """Tangent-space normal map of `hires` baked onto the low mesh's UV layout.

    v_down=True uses the KDTree baker's row convention (row = v); False uses
    the trimesh/glTF convention (row = 1 - v).
    Returns (uint8 [H,W,3], mask [H,W]).

    shading="smooth" (default): the map is encoded in the frame a glTF renderer rebuilds, i.e. the INTERPOLATED VERTEX normal of the low mesh (`low_normals`, else welded
    area-weighted normals of the low mesh) and tangents generated per vertex from the UV derivatives. The exported GLB carries smooth vertex normals, so a map encoded
    against the FLAT face normal ("flat", the pre-2026-11-02 behaviour, kept for comparison) is decoded against a different base normal and prints the low-poly triangle
    layout as crystal shards (tests/test_normal_map_frame.py measures both).
    """
    t0 = time.time()
    V = np.asarray(low_vertices, dtype=np.float32)
    F = np.asarray(low_faces, dtype=np.int64)
    UV = np.asarray(low_uvs, dtype=np.float32)
    positions, mask, face_ids = _rasterize(V, F, UV, texture_size, v_down)
    log(f"    normal map: rasterised {mask.sum():,} texels ({time.time() - t0:.1f}s)")

    T, B = _face_tangents(V, F, UV)
    lf = face_ids[mask]
    p = positions[mask]
    cp, fid, bary, _ = hires.query(p)
    n_hi = hires.normal(fid, bary, smooth=smooth)

    tri = V[F]
    if shading == "flat":   # old behaviour: face normal, face tangents
        n_lo = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n_lo /= np.maximum(np.linalg.norm(n_lo, axis=1, keepdims=True), 1e-20)
        t, b, n = T[lf], B[lf], n_lo[lf]
        t = t - n * (t * n).sum(1, keepdims=True)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-20)
        handed = np.sign((np.cross(n, t) * b).sum(1, keepdims=True))
        handed[handed == 0] = 1
        b = np.cross(n, t) * handed
    else:   # what the renderer interpolates: vertex normals, vertex tangents, handedness
        Nv = np.asarray(low_normals, dtype=np.float32) if low_normals is not None else vertex_normals_welded(V, F)
        Tv, Hv = _vertex_tangents(V, F, UV, Nv)
        bary_lo = _barycentric(p, tri[lf])
        fl = F[lf]
        n = (Nv[fl] * bary_lo[..., None]).sum(1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        t = (Tv[fl] * bary_lo[..., None]).sum(1)
        t = t - n * (t * n).sum(1, keepdims=True)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-20)
        handed = np.where((Hv[fl] * bary_lo[..., None]).sum(1) >= 0, 1.0, -1.0)
        b = np.cross(n, t) * handed
    # Normal map convention: +Y of the map follows +V in the texture (glTF/OpenGL).
    ts = np.stack([(n_hi * t).sum(1), (n_hi * b).sum(1), (n_hi * n).sum(1)], axis=1)
    ts[:, 2] = np.maximum(ts[:, 2], 0.0)           # never point into the surface
    ts /= np.maximum(np.linalg.norm(ts, axis=1, keepdims=True), 1e-20)

    H = W = texture_size
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 2] = 1.0
    img[mask] = ts
    if dilate:
        from scipy.ndimage import binary_dilation, uniform_filter
        cur = mask.copy()
        for _ in range(dilate):
            grown = binary_dilation(cur, iterations=1)
            new = grown & ~cur
            if not new.any():
                break
            for c in range(3):
                blurred = uniform_filter(img[..., c], size=3)
                img[..., c][new] = blurred[new]
            cur = grown
    out = ((np.clip(img, -1, 1) * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)
    log(f"    normal map: done ({time.time() - t0:.1f}s)")
    return out, mask


def attach_normal_map(obj, normal_img):
    """Set normalTexture on every PBR material of a Trimesh/Scene."""
    from PIL import Image
    from trellis3.postbake import _materials
    n = 0
    for mat in _materials(obj):
        if hasattr(mat, "normalTexture"):
            mat.normalTexture = Image.fromarray(normal_img, "RGB")
            n += 1
    return n


def glb_to_decoder_frame(vertices):
    """Undo to_glb's Y-up -> Z-up conversion so positions match the decoder mesh."""
    v = np.asarray(vertices, dtype=np.float32).copy()
    y = -v[:, 2].copy()
    z = v[:, 1].copy()
    v[:, 1], v[:, 2] = y, z
    return v
